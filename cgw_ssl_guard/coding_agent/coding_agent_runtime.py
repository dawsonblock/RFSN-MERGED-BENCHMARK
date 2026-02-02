"""Serial decision runtime for the coding agent.

This module implements the main control loop following the CGW/SSL
serial decision architecture. The runtime:

1. Collects proposals from all generators
2. Submits them to the thalamic gate
3. Gate selects exactly one winner
4. Commits the winner to CGW (atomic swap)
5. Executes the action (blocking)
6. Reports result and prepares for next cycle

HARD CONSTRAINTS (NON-NEGOTIABLE):
- Exactly one decision per cycle
- No parallel decisions
- Atomic commit only
- Execution happens OUTSIDE the decision loop
- Tool execution is blocking with respect to decisions
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..event_bus import SimpleEventBus
from ..cgw_state import CGWRuntime
from ..thalamic_gate import ThalamusGate
from ..types import Candidate, ForcedCandidate, SelectionReason, SelfModel
from ..monitors import SerialityMonitor

from .action_types import (
    ActionPayload,
    CodingAction,
    CycleResult,
    ExecutionResult,
)
from .executor import BlockingExecutor, ExecutorConfig
from .proposal_generators import (
    ProposalContext,
    ProposalGenerator,
    SafetyProposalGenerator,
    PlannerProposalGenerator,
    IdleProposalGenerator,
)

# Optional LLM-backed proposal generators (only enabled when API keys exist)
from .llm_adapter import validate_api_keys, create_llm_caller
from .llm_integration import LLMAnalysisGenerator, LLMPatchGenerator, LLMDecisionAdvisor

# Advisory-only simulation gate (never executes tools; only adjusts scores)
from .sim.simulation_gate import SimulationGate

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the coding agent runtime."""
    
    # Cycle limits
    max_cycles: int = 100
    max_patches: int = 10
    max_test_runs: int = 20
    
    # Timeouts (in seconds)
    cycle_timeout: float = 600.0
    total_timeout: float = 3600.0
    
    # Goal
    goal: str = "Fix failing tests"
    
    # Competition cooldown
    cooldown_ms: int = 100

    # Advisory cognition
    enable_simulation_gate: bool = True


@dataclass
class AgentResult:
    """Final result of the coding agent run."""
    
    success: bool
    final_action: CodingAction
    cycles_executed: int
    total_time_ms: float
    
    # Outcome details
    tests_passing: bool = False
    patches_applied: int = 0
    
    # Cycle history for replay
    cycle_history: List[CycleResult] = field(default_factory=list)
    
    # Error info
    error: Optional[str] = None
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        status = "SUCCESS" if self.success else "FAILURE"
        return (
            f"[{status}] {self.final_action.value} after {self.cycles_executed} cycles "
            f"({self.total_time_ms:.1f}ms). Tests passing: {self.tests_passing}. "
            f"Patches applied: {self.patches_applied}."
        )


class CodingAgentRuntime:
    """Serial decision controller for autonomous coding tasks.
    
    This runtime enforces the CGW architecture's key invariants:
    - One decision per cycle (single-slot workspace)
    - Forced signals bypass competition (safety/abort)
    - Blocking execution (no tool overlap)
    - Event emission for replay
    
    Usage:
        runtime = CodingAgentRuntime(config=AgentConfig(goal="Fix tests"))
        result = runtime.run_until_done()
    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        event_bus: Optional[SimpleEventBus] = None,
        gate: Optional[ThalamusGate] = None,
        cgw: Optional[CGWRuntime] = None,
        executor: Optional[BlockingExecutor] = None,
        generators: Optional[List[ProposalGenerator]] = None,
    ):
        self.config = config or AgentConfig()
        
        # Core CGW components
        self.event_bus = event_bus or SimpleEventBus()
        self.gate = gate or ThalamusGate(self.event_bus)
        self.cgw = cgw or CGWRuntime(self.event_bus)
        self.executor = executor or BlockingExecutor(config=ExecutorConfig())
        
        # Generators (proposal sources)
        self.generators = generators or self._create_default_generators()

        # Advisory-only cognition layer (adjusts proposal scores; never executes)
        self.sim_gate = SimulationGate(event_bus=self.event_bus) if self.config.enable_simulation_gate else None
        
        # Seriality monitor for debugging
        self.seriality_monitor = SerialityMonitor()
        self.event_bus.on("CGW_COMMIT", self.seriality_monitor.on_commit)
        
        # State
        self._context = self._create_initial_context()
        self._cycle_history: List[CycleResult] = []
        self._is_running = False
        self._start_time: float = 0.0
        
        # Metrics
        self._patches_applied = 0
        self._test_runs = 0
        
        # Event subscribers
        self._on_cycle_complete: List[Callable[[CycleResult], None]] = []
    
    def _create_default_generators(self) -> List[ProposalGenerator]:
        """Create the default set of proposal generators."""
        gens: List[ProposalGenerator] = [
            SafetyProposalGenerator(),   # Safety always first
            PlannerProposalGenerator(),  # Main planner
            IdleProposalGenerator(),     # Fallback
        ]

        # If any LLM API key is configured, enable LLM-backed proposal generators.
        # They ONLY propose candidates; execution remains governed by the executor spine.
        keys = validate_api_keys()
        if any(keys.values()):
            llm = create_llm_caller()
            # Insert after the planner so they compete with planner proposals.
            gens.insert(2, LLMAnalysisGenerator(llm_caller=llm))
            gens.insert(3, LLMPatchGenerator(llm_caller=llm))
            gens.insert(4, LLMDecisionAdvisor(llm_caller=llm))

        return gens
    
    def _create_initial_context(self) -> ProposalContext:
        """Create the initial proposal context."""
        return ProposalContext(
            cycle_id=0,
            goal=self.config.goal,
        )
    
    def on_cycle_complete(self, callback: Callable[[CycleResult], None]) -> None:
        """Register a callback for cycle completion events."""
        self._on_cycle_complete.append(callback)
    
    def inject_forced_signal(self, action: CodingAction, reason: str = "FORCED") -> str:
        """Inject a forced signal that bypasses competition.
        
        Use this for safety signals, emergency aborts, or other
        high-priority overrides.
        
        Returns:
            The slot_id of the injected signal.
        """
        payload = ActionPayload(
            action=action,
            context={"reason": reason, "forced": True},
        )
        return self.gate.inject_forced_signal(
            source_module="forced_override",
            content_payload=payload.to_bytes(),
            reason=reason,
        )
    
    def tick(self) -> CycleResult:
        """Execute exactly one decision cycle.
        
        This is the core of the serial decision architecture:
        1. Collect proposals from all generators
        2. Gate selects winner (forced signals first)
        3. Commit to CGW (atomic swap)
        4. Execute action (blocking)
        5. Update context with results
        6. Return cycle result
        
        Returns:
            CycleResult with the decision, execution result, and metadata.
        
        Raises:
            RuntimeError: If tick() is called while already running.
        """
        cycle_start = time.time()
        self._context.cycle_id += 1
        
        # --- Phase 1: Collect Proposals ---
        for generator in self.generators:
            candidates = generator.generate(self._context)
            for candidate in candidates:
                self.gate.submit_candidate(candidate)

        # --- Phase 1b: Advisory Simulation Gate (score adjustment only) ---
        # Mimics short human "mental rehearsal" ONLY when impact/uncertainty is high.
        # NEVER runs shell, NEVER applies patches, NEVER bypasses forced signals.
        if self.sim_gate is not None and self.gate.candidates:
            self.sim_gate.adjust_candidates(self.gate.candidates, self._context)
        
        # --- Phase 2: Gate Selection ---
        decision_start = time.time()
        winner, reason = self.gate.select_winner()
        decision_time_ms = (time.time() - decision_start) * 1000
        
        if winner is None:
            # Idle cycle - no winner selected
            return CycleResult(
                cycle_id=self._context.cycle_id,
                action=CodingAction.IDLE,
                payload=ActionPayload(action=CodingAction.IDLE),
                execution_result=None,
                decision_time_ms=decision_time_ms,
            )
        
        # --- Phase 3: Parse Payload ---
        payload = ActionPayload.from_bytes(winner.content_payload)
        was_forced = isinstance(winner, ForcedCandidate)
        
        # --- Phase 4: Commit to CGW ---
        self_model = SelfModel(
            goals=[self.config.goal],
            active_intentions=[payload.action.value],
        )
        self.cgw.update(winner, reason, self_model)
        
        # --- Phase 5: Execute (BLOCKING) ---
        execution_start = time.time()
        execution_result = self.executor.execute(payload)
        execution_time_ms = (time.time() - execution_start) * 1000
        
        # --- Phase 6: Update Context ---
        self._update_context(payload.action, execution_result)
        
        # --- Phase 7: Create Cycle Result ---
        # Get losers from gate state (already cleared, so check event)
        losers = []  # Would need to track from event
        
        result = CycleResult(
            cycle_id=self._context.cycle_id,
            action=payload.action,
            payload=payload,
            execution_result=execution_result,
            slot_id=winner.slot_id,
            was_forced=was_forced,
            losers=losers,
            decision_time_ms=decision_time_ms,
            execution_time_ms=execution_time_ms,
        )
        
        # Store in history
        self._cycle_history.append(result)
        
        # Notify subscribers
        for callback in self._on_cycle_complete:
            try:
                callback(result)
            except Exception as e:
                logger.warning(f"Cycle callback error: {e}")
        
        # Log
        logger.info(
            f"Cycle {result.cycle_id}: {result.action.value} "
            f"(forced={was_forced}, decision={decision_time_ms:.1f}ms, "
            f"exec={execution_time_ms:.1f}ms)"
        )
        
        return result
    
    def _update_context(self, action: CodingAction, result: ExecutionResult) -> None:
        """Update the proposal context based on execution results."""
        self._context.last_action = action
        self._context.last_result = result
        
        # Update test state
        if action in (CodingAction.RUN_TESTS, CodingAction.RUN_FOCUSED_TESTS):
            self._test_runs += 1
            self._context.tests_passing = result.tests_failed == 0 and result.tests_passed > 0
            self._context.failing_tests = result.failing_tests
            self._context.test_output = result.output
        
        # Update patch state
        if action == CodingAction.APPLY_PATCH and result.patch_applied:
            self._patches_applied += 1
            self._context.patches_applied = self._patches_applied
        
        # Check for resource limits triggering safety
        if self._patches_applied >= self.config.max_patches:
            self._context.safety_triggered = True
            self._context.safety_reason = f"Max patches ({self.config.max_patches}) reached"
        if self._test_runs >= self.config.max_test_runs:
            self._context.safety_triggered = True
            self._context.safety_reason = f"Max test runs ({self.config.max_test_runs}) reached"
    
    def run_until_done(self, max_cycles: Optional[int] = None) -> AgentResult:
        """Run cycles until FINALIZE, ABORT, or max cycles reached.
        
        Args:
            max_cycles: Override for config.max_cycles.
            
        Returns:
            AgentResult with success status and cycle history.
        """
        if self._is_running:
            raise RuntimeError("CodingAgentRuntime is already running")
        
        self._is_running = True
        self._start_time = time.time()
        max_cycles = max_cycles or self.config.max_cycles
        
        try:
            while len(self._cycle_history) < max_cycles:
                # Check total timeout
                elapsed = time.time() - self._start_time
                if elapsed > self.config.total_timeout:
                    logger.warning("Total timeout reached")
                    self.inject_forced_signal(CodingAction.ABORT, "timeout")
                
                # Execute one cycle
                result = self.tick()
                
                # Check for terminal actions
                if result.action == CodingAction.FINALIZE:
                    return self._create_result(success=True, final_action=result.action)
                
                if result.action == CodingAction.ABORT:
                    return self._create_result(success=False, final_action=result.action)
            
            # Max cycles reached
            logger.warning(f"Max cycles ({max_cycles}) reached")
            return self._create_result(
                success=False,
                final_action=CodingAction.ABORT,
                error=f"Max cycles ({max_cycles}) reached",
            )
            
        finally:
            self._is_running = False
    
    def _create_result(
        self,
        success: bool,
        final_action: CodingAction,
        error: Optional[str] = None,
    ) -> AgentResult:
        """Create the final agent result."""
        total_time_ms = (time.time() - self._start_time) * 1000
        return AgentResult(
            success=success,
            final_action=final_action,
            cycles_executed=len(self._cycle_history),
            total_time_ms=total_time_ms,
            tests_passing=self._context.tests_passing,
            patches_applied=self._patches_applied,
            cycle_history=self._cycle_history.copy(),
            error=error,
        )
    
    def verify_seriality(self) -> bool:
        """Verify that no cycle had more than one commit.
        
        Returns:
            True if seriality invariant was maintained.
        """
        for cycle_id, count in self.seriality_monitor.commits_per_cycle.items():
            if count > 1:
                logger.error(f"Seriality violation: Cycle {cycle_id} had {count} commits")
                return False
        return True
    
    def get_cycle_history(self) -> List[CycleResult]:
        """Get the history of all executed cycles for replay."""
        return self._cycle_history.copy()
    
    def get_context(self) -> ProposalContext:
        """Get the current proposal context."""
        return self._context
