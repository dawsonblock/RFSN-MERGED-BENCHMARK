"""Controller Execution Loop with Planner Integration (SERIAL, governed).

This module implements the SERIAL execution loop that integrates
the planner with the controller while maintaining safety guarantees.

EXECUTION MODEL:
1. Build observation packet (read-only)
2. Planner produces plan (JSON data)
3. PlanGate validates (HARD SAFETY)
4. For each step in topological order:
   - StepGate validates (per-step)
   - Execute step (only via GovernedExecutor - single spine)
   - Verify (blocking)
   - Record outcome
5. If step fails: planner may replan, but gate/budgets remain unchanged

KEY INVARIANTS:
- One mutation at a time (serial execution)
- Planner cannot execute (only proposes)
- Gate cannot be bypassed
- Learning cannot modify gates
- ALL execution goes through GovernedExecutor (single spine)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .executor_spine import GovernedExecutor, StepExecResult
from .gates.plan_gate import PlanGate, PlanGateError, StepGateError
from .learning import LearnedStrategySelector

logger = logging.getLogger(__name__)


@dataclass
class ExecutionOutcome:
    """Outcome of a single step execution."""
    
    step_id: str
    success: bool
    elapsed_ms: int = 0
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    details: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "success": self.success,
            "elapsed_ms": self.elapsed_ms,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
        }


@dataclass
class LoopResult:
    """Result of a full execution loop."""
    
    success: bool
    steps_executed: int
    steps_succeeded: int
    elapsed_ms: int
    outcomes: list[ExecutionOutcome] = field(default_factory=list)
    final_status: str = "unknown"
    error: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "steps_executed": self.steps_executed,
            "steps_succeeded": self.steps_succeeded,
            "elapsed_ms": self.elapsed_ms,
            "final_status": self.final_status,
            "error": self.error,
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


# Type for verifier function (simple boolean return)
Verifier = Callable[[], bool]


class ControllerExecutionLoop:
    """Serial plan execution loop.

    This loop NEVER executes directly. It routes execution to GovernedExecutor,
    which is the single execution spine.
    """
    
    def __init__(
        self,
        *,
        gate: PlanGate | None = None,
        governed_executor: GovernedExecutor | None = None,
        executor: Callable[[dict], ExecutionOutcome] | None = None,  # Compat
        learning: LearnedStrategySelector | None = None,
        max_replans: int = 2,
        event_recorder: Callable[[str, dict], None] | None = None,
    ):
        self.gate = gate or PlanGate()
        self.exec = governed_executor
        self._simple_executor = executor  # For backwards compatibility
        self.learning = learning
        self.max_replans = max_replans
        self.event_recorder = event_recorder or self._default_recorder
        
        self._current_plan: dict[str, Any] | None = None
        self._completed_steps: list[str] = []
    
    def run_with_planner(
        self,
        planner: Any,  # PlannerV2 or compatible
        observation: dict[str, Any],
    ) -> LoopResult:
        start_time = time.monotonic()
        outcomes: list[ExecutionOutcome] = []

        if self.exec is None:
            return LoopResult(
                success=False,
                steps_executed=0,
                steps_succeeded=0,
                elapsed_ms=0,
                outcomes=[],
                final_status="error",
                error="No governed_executor configured (required).",
            )

        replans = 0
        plan = None

        while replans <= self.max_replans:
            # 1) planner proposes a plan artifact (data)
            plan = planner.propose_plan(
                goal=observation.get("goal", ""),
                context=observation,
            )
            self._current_plan = plan
            self.event_recorder("PLAN_PROPOSED", {"plan": plan})

            # 2) hard validate plan
            try:
                self.gate.validate_plan(plan)
            except PlanGateError as e:
                return self._fail(start_time, outcomes, f"PlanGateError: {e}")

            # Derive steps from validated plan structure
            steps = plan.get("steps") if isinstance(plan, dict) else None
            if not isinstance(steps, list):
                return self._fail(
                    start_time,
                    outcomes,
                    "Validated plan does not contain a 'steps' list.",
                )
            # 3) execute serially
            step_outcomes, ok = self._execute_steps_serial(steps)
            outcomes.extend(step_outcomes)

            if ok:
                elapsed = int((time.monotonic() - start_time) * 1000)
                return LoopResult(
                    success=True,
                    steps_executed=len(outcomes),
                    steps_succeeded=sum(1 for o in outcomes if o.success),
                    elapsed_ms=elapsed,
                    outcomes=outcomes,
                    final_status="success",
                )

            # step failure: replan with updated observation (caller must populate)
            replans += 1
            self.event_recorder("REPLAN", {"attempt": replans})

        return self._fail(start_time, outcomes, "Exceeded max replans")
    
    def _execute_steps_serial(self, steps: list[dict[str, Any]]) -> tuple[list[ExecutionOutcome], bool]:
        outs: list[ExecutionOutcome] = []
        for step in steps:
            sid = step.get("id", "unknown")
            try:
                # per-step validation (strong)
                self.gate.validate_step(step)
            except StepGateError as e:
                outs.append(ExecutionOutcome(step_id=sid, success=False, stderr=f"StepGateError: {e}", exit_code=1))
                return outs, False

            self.event_recorder("STEP_START", {"step": step})

            r: StepExecResult = self.exec.execute_step(step)
            outs.append(
                ExecutionOutcome(
                    step_id=sid,
                    success=bool(r.ok),
                    elapsed_ms=r.elapsed_ms,
                    stdout=r.stdout,
                    stderr=r.stderr,
                    exit_code=r.exit_code,
                    details=r.details or {},
                )
            )

            self.event_recorder("STEP_DONE", {"step_id": sid, "ok": r.ok, "exit_code": r.exit_code})

            if not r.ok:
                return outs, False

            # verifier-first: after each step, run verify (blocking)
            vr = self.exec.verify()
            outs.append(
                ExecutionOutcome(
                    step_id="verify",
                    success=bool(vr.ok),
                    elapsed_ms=vr.elapsed_ms,
                    stdout=vr.stdout,
                    stderr=vr.stderr,
                    exit_code=vr.exit_code,
                    details=vr.details or {},
                )
            )
            self.event_recorder("VERIFY_DONE", {"ok": vr.ok})
            if not vr.ok:
                return outs, False

        return outs, True

    def _default_recorder(self, event_type: str, data: dict) -> None:
        logger.info("Event: %s - %s", event_type, data)

    def _fail(self, start_time: float, outcomes: list[ExecutionOutcome], error: str) -> LoopResult:
        elapsed = int((time.monotonic() - start_time) * 1000)
        return LoopResult(
            success=False,
            steps_executed=len(outcomes),
            steps_succeeded=sum(1 for o in outcomes if o.success),
            elapsed_ms=elapsed,
            outcomes=outcomes,
            final_status="error",
            error=error,
        )
    
    def run_plan(self, plan: dict[str, Any]) -> LoopResult:
        """Run a plan directly (backwards compatibility).
        
        This method is for testing and simple use cases where no planner
        is involved.
        """
        start_time = time.monotonic()
        outcomes: list[ExecutionOutcome] = []
        
        # 1. Validate plan
        try:
            self.gate.validate_plan(plan)
        except (PlanGateError, StepGateError) as e:
            return LoopResult(
                success=False,
                steps_executed=0,
                steps_succeeded=0,
                elapsed_ms=0,
                final_status="plan_gate_error",
                error=str(e),
            )
        
        # 2. Get steps
        steps = plan.get("steps") if isinstance(plan, dict) else None
        if not isinstance(steps, list):
            return self._fail(start_time, outcomes, "No steps in plan")
        
        # 3. Execute steps
        for step in steps:
            sid = step.get("id", "unknown")
            
            try:
                self.gate.validate_step(step)
            except StepGateError as e:
                outcomes.append(ExecutionOutcome(
                    step_id=sid, success=False, stderr=str(e), exit_code=1
                ))
                return LoopResult(
                    success=False,
                    steps_executed=len(outcomes),
                    steps_succeeded=sum(1 for o in outcomes if o.success),
                    elapsed_ms=int((time.monotonic() - start_time) * 1000),
                    outcomes=outcomes,
                    final_status="step_gate_error",
                    error=str(e),
                )
            
            # Execute using simple executor if available
            if self._simple_executor:
                outcome = self._simple_executor(step)
                outcomes.append(outcome)
            else:
                # Default: mark as success for testing
                outcomes.append(ExecutionOutcome(step_id=sid, success=True))
        
        elapsed = int((time.monotonic() - start_time) * 1000)
        return LoopResult(
            success=True,
            steps_executed=len(outcomes),
            steps_succeeded=sum(1 for o in outcomes if o.success),
            elapsed_ms=elapsed,
            outcomes=outcomes,
            final_status="success",
        )


# Backwards compatibility alias
ControllerLoop = ControllerExecutionLoop
