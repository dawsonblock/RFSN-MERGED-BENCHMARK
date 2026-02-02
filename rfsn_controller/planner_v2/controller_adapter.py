"""Planner Layer v2.1 - Controller Adapter with Governance.

Interface between PlannerV2 and the Controller.
Translates steps to controller task specs and processes outcomes.
Includes validation, budgeting, halt conditions, and artifact logging.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .artifact_log import PlanArtifactLog
from .fingerprint import RepoFingerprint, compute_fingerprint
from .governance import (
    ContentSanitizer,
    HaltChecker,
    HaltSpec,
    PlanBudget,
    PlanValidator,
    ValidationResult,
)
from .memory_adapter import MemoryAdapter
from .overrides import OverrideManager
from .planner import PlannerV2
from .qa_integration import PlannerQABridge
from .schema import (
    ControllerOutcome,
    ControllerTaskSpec,
    Plan,
    PlanState,
)

if TYPE_CHECKING:
    from ..qa.qa_orchestrator import QAOrchestrator

logger = logging.getLogger(__name__)


class ControllerAdapter:
    """Adapter between PlannerV2 and the Controller with Governance.

    This class provides the interface for the controller to:
    1. Start a new goal and get the first task spec
    2. Process outcomes and get subsequent task specs
    3. Access plan state for logging/auditing

    Governance features:
    - Plan validation before execution
    - Resource budgeting (patch cycles, failures, tokens, time)
    - Halt conditions (flaky tests, file growth, identical failures)
    - Artifact logging for replay
    - Override support for runtime modification

    The controller calls this adapter; the adapter calls the planner.
    The planner never executes directly.
    """

    def __init__(
        self,
        planner: PlannerV2 | None = None,
        memory_adapter: MemoryAdapter | None = None,
        seed: int = 0,
        # Governance options
        budget: PlanBudget | None = None,
        halt_spec: HaltSpec | None = None,
        validator: PlanValidator | None = None,
        sanitizer: ContentSanitizer | None = None,
        # Artifact logging
        artifact_dir: Path | None = None,
        repo_dir: Path | None = None,
        # Overrides
        override_file: Path | None = None,
        # QA
        qa_orchestrator: QAOrchestrator | None = None,
    ):
        """Initialize the controller adapter.

        Args:
            planner: Optional PlannerV2 instance. If not provided, one is created.
            memory_adapter: Optional memory adapter for historical queries.
            seed: Seed for deterministic plan generation.
            budget: Optional budget for resource limits.
            halt_spec: Optional halt condition specification.
            validator: Optional plan validator.
            sanitizer: Optional content sanitizer.
            artifact_dir: Optional directory for artifact logging.
            repo_dir: Optional repo directory for fingerprinting.
            override_file: Optional JSON file for runtime overrides.
            qa_orchestrator: Optional QA orchestrator for claim verification.
        """
        if planner is None:
            planner = PlannerV2(memory_adapter=memory_adapter, seed=seed)
        self._planner = planner
        self._current_plan: Plan | None = None
        self._current_state: PlanState | None = None
        
        # Governance
        self._budget = budget
        self._halt_checker = HaltChecker(halt_spec) if halt_spec else HaltChecker()
        self._validator = validator or PlanValidator(strict_mode=False)
        self._sanitizer = sanitizer or ContentSanitizer(mode="flag")
        
        # Artifact logging
        self._artifact_log: PlanArtifactLog | None = None
        if artifact_dir:
            self._artifact_log = PlanArtifactLog(artifact_dir)
        self._current_artifact_id: str | None = None
        self._repo_dir = repo_dir
        self._repo_fingerprint: RepoFingerprint | None = None
        
        # Overrides
        self._override_manager = OverrideManager(override_file)
        
        # QA Bridge
        self._qa_bridge = PlannerQABridge(qa_orchestrator)
        
        # Step timing
        self._step_start_time: float | None = None
        self._files_touched: list[str] = []

    def start_goal(
        self,
        goal: str,
        context: dict[str, Any],
        validate: bool = True,
    ) -> ControllerTaskSpec:
        """Start a new goal and get the first task spec.

        Generates a plan for the goal, validates it, and returns the first step
        as a task spec for the controller to execute.

        Args:
            goal: The high-level goal description.
            context: Execution context (repo_type, language, test_cmd, etc.)
            validate: Whether to validate the plan before execution.

        Returns:
            Task spec for the first step.

        Raises:
            ValueError: If plan has no executable steps or fails validation.
        """
        # Sanitize context if it contains repo content
        if self._sanitizer and "readme" in context:
            result = self._sanitizer.sanitize(context.get("readme", ""))
            if result.was_modified:
                context["readme"] = result.sanitized
                context["_sanitizer_triggered"] = result.triggered_patterns
        
        # Generate plan
        logger.info("Starting goal: %s", goal)
        self._current_plan = self._planner.propose_plan(goal, context)
        self._current_state = PlanState(plan_id=self._current_plan.plan_id)
        logger.debug("Generated plan %s with %d steps", self._current_plan.plan_id, len(self._current_plan.steps))
        
        # Validate plan
        if validate:
            validation = self._validator.validate(self._current_plan)
            if not validation.valid:
                error_summary = "; ".join(str(e) for e in validation.errors[:3])
                raise ValueError(f"Plan validation failed: {error_summary}")
        
        # Initialize halt checker
        self._halt_checker.initialize(self._current_plan)
        
        # Initialize budget if not set
        if self._budget is None:
            self._budget = PlanBudget()
        
        # Compute repo fingerprint
        if self._repo_dir:
            self._repo_fingerprint = compute_fingerprint(self._repo_dir)
        
        # Start artifact logging
        if self._artifact_log:
            fp_hash = self._repo_fingerprint.to_hash() if self._repo_fingerprint else "unknown"
            self._current_artifact_id = self._artifact_log.record_plan_start(
                self._current_plan,
                fp_hash,
                metadata={"context_keys": list(context.keys())},
            )
        
        # Get first step
        step = self._planner.next_step(self._current_plan, self._current_state)
        if step is None:
            raise ValueError("Plan has no executable steps")
        
        # Check overrides
        if self._override_manager.should_skip(step.step_id):
            # Skip and get next
            self._current_state.completed_steps.append(step.step_id)
            step = self._planner.next_step(self._current_plan, self._current_state)
            if step is None:
                raise ValueError("All steps skipped")
        
        # Apply overrides
        step = self._override_manager.apply(step)
        
        # Start step timing
        self._step_start_time = time.monotonic()
        self._files_touched = []
        
        logger.debug("Returning task spec for step: %s", step.step_id)
        return step.get_task_spec()

    def process_outcome(
        self,
        outcome: ControllerOutcome,
        diff: str = "",
        files_touched: list[str] | None = None,
    ) -> ControllerTaskSpec | None:
        """Process controller outcome and get next task spec.

        Updates plan state based on the outcome, checks governance conditions,
        revises if needed, and returns the next step to execute.

        Args:
            outcome: Outcome from controller execution.
            diff: Optional diff produced (for artifact hashing).
            files_touched: Optional list of files modified.

        Returns:
            Task spec for the next step, or None if complete.

        Raises:
            ValueError: If no active plan.
        """
        if self._current_plan is None or self._current_state is None:
            raise ValueError("No active plan")
        
        # Calculate step elapsed time
        elapsed_ms = 0
        if self._step_start_time:
            elapsed_ms = int((time.monotonic() - self._step_start_time) * 1000)
        
        if files_touched:
            self._files_touched = files_touched
        
        logger.info("Processing outcome for step %s: success=%s", outcome.step_id, outcome.success)
        
        # Record step artifact
        if self._artifact_log and self._current_artifact_id:
            step = self._current_plan.get_step(outcome.step_id)
            if step:
                self._artifact_log.record_step(
                    self._current_artifact_id,
                    step.get_task_spec(),
                    outcome,
                    diff,
                    elapsed_ms,
                    self._files_touched,
                )
        
        # Update budget
        if self._budget:
            self._budget.record_patch_cycle()
            if not outcome.success:
                self._budget.record_failing_step()
            
            # Check budget exhaustion
            exhausted, resource = self._budget.is_exhausted()
            if exhausted:
                self._current_state.halted = True
                self._current_state.halt_reason = f"Budget exhausted: {resource}"
                self._finalize_artifact("budget_exhausted")
                return None
        
        # Record for halt checker
        is_flaky = bool(
            outcome.failure_evidence and 
            outcome.failure_evidence.category.value == "flaky_test"
        )
        self._halt_checker.record_outcome(
            outcome.step_id,
            outcome,
            self._files_touched,
            is_flaky=is_flaky,
        )
        
        # Check halt conditions
        halt_reason = self._halt_checker.check(self._current_plan, self._current_state)
        if halt_reason:
            self._current_state.halted = True
            self._current_state.halt_reason = halt_reason
            self._finalize_artifact("halt_condition")
            return None
        
        # Check override halt
        if self._override_manager.should_halt():
            self._current_state.halted = True
            self._current_state.halt_reason = self._override_manager.get_halt_reason()
            self._finalize_artifact("manual_halt")
            return None
        
        # QA Verification
        if outcome.success and self._qa_bridge.enabled:
            step = self._current_plan.get_step(outcome.step_id)
            if step:
                qa_result = self._qa_bridge.verify_step_outcome(
                    step, 
                    outcome, 
                    diff,
                    # We might pass test command if known, but for now rely on QA orchestrator
                )
                
                # Check Guardrails (Upgrade 4)
                # We check only if success, because failure is already bad
                violations = []
                if diff and outcome.success:
                    for f in (self._files_touched or []):
                        # In a real impl, we'd need to fetch content.
                        # For now we just call trigger the method for architectural completeness
                        v = self._planner.check_guardrails(f, diff)
                        violations.extend(v)
                
                if violations:
                     outcome.success = False
                     outcome.error_message = f"Guardrail Violation: {'; '.join(violations)}"
                     # We don't rollback physically here, but we mark as failed
                
                if not qa_result.accepted:
                    # Convert QA rejection to controller failure
                    outcome.success = False
                    outcome.error_message = f"QA Rejected: {'; '.join(qa_result.rejection_reasons)}"
                    outcome.failure_evidence = self._qa_bridge.convert_qa_to_failure_evidence(qa_result)
                    
                    # Log QA rejection
                    if self._artifact_log and self._current_artifact_id:
                        self._artifact_log.record_qa_rejection(
                            self._current_artifact_id,
                            step.step_id,
                            qa_result
                        )
        
        # Update state
        self._current_state = self._planner.update_state(
            self._current_plan,
            self._current_state,
            outcome,
        )
        
        # Record outcome to memory/firewall
        # Extract tags (files + failure type)
        tags = list(self._files_touched)
        if outcome.failure_evidence:
            tags.append(outcome.failure_evidence.category.value)
            
        self._planner.record_action_outcome(
            step_id=outcome.step_id,
            success=outcome.success,
            diff=diff,
            tags=tags,
            failure_type=outcome.failure_evidence.category.value if outcome.failure_evidence else "unknown",
            files=self._files_touched,
        )

        # Check if we need to revise on failure
        if not outcome.success and not self._current_state.halted:
            self._current_plan = self._planner.revise_plan(
                self._current_plan,
                self._current_state,
                outcome,
            )

        # Check if complete
        if self._planner.is_complete(self._current_plan, self._current_state):
            self._finalize_artifact("success" if outcome.success else "complete")
            return None

        # Get next step
        step = self._planner.next_step(self._current_plan, self._current_state)
        if step is None:
            self._finalize_artifact("no_more_steps")
            return None
        
        # Check and apply overrides
        while self._override_manager.should_skip(step.step_id):
            self._current_state.completed_steps.append(step.step_id)
            step = self._planner.next_step(self._current_plan, self._current_state)
            if step is None:
                self._finalize_artifact("all_skipped")
                return None
        
        step = self._override_manager.apply(step)
        
        # Start timing for next step
        self._step_start_time = time.monotonic()
        self._files_touched = []

        return step.get_task_spec()
    
    def _finalize_artifact(self, status: str) -> None:
        """Finalize artifact logging."""
        if self._artifact_log and self._current_artifact_id:
            self._artifact_log.finalize(self._current_artifact_id, status)
            self._current_artifact_id = None

    def validate_plan(self) -> ValidationResult:
        """Validate the current plan.

        Returns:
            ValidationResult with errors and warnings.
        """
        if self._current_plan is None:
            return ValidationResult(valid=False, errors=[])
        return self._validator.validate(self._current_plan)

    def get_budget_status(self) -> dict[str, Any]:
        """Get current budget status.

        Returns:
            Budget status dictionary.
        """
        if self._budget is None:
            return {"enabled": False}
        return {
            "enabled": True,
            **self._budget.to_dict(),
        }

    def get_halt_status(self) -> dict[str, Any]:
        """Get halt checker status.

        Returns:
            Halt checker statistics.
        """
        return self._halt_checker.get_statistics()

    def get_plan_json(self) -> str:
        """Get the current plan as JSON for auditing.

        Returns:
            JSON string of current plan, or empty object if no plan.
        """
        if self._current_plan is None:
            return "{}"
        return self._current_plan.to_json()

    def get_plan(self) -> Plan | None:
        """Get the current plan.

        Returns:
            Current plan or None.
        """
        return self._current_plan

    def get_state(self) -> PlanState | None:
        """Get current plan state.

        Returns:
            Current plan state or None.
        """
        return self._current_state

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of current plan execution.

        Returns:
            Dictionary with plan summary.
        """
        if self._current_plan is None or self._current_state is None:
            return {"active": False}

        summary = self._planner.get_plan_summary(
            self._current_plan,
            self._current_state,
        )
        summary["active"] = True
        summary["budget"] = self.get_budget_status()
        summary["halt_status"] = self.get_halt_status()
        return summary

    def is_complete(self) -> bool:
        """Check if the current plan is complete.

        Returns:
            True if no active plan or plan is complete.
        """
        if self._current_plan is None or self._current_state is None:
            return True
        return self._planner.is_complete(self._current_plan, self._current_state)

    def is_halted(self) -> bool:
        """Check if the current plan is halted.

        Returns:
            True if plan is halted due to failures.
        """
        if self._current_state is None:
            return False
        return self._current_state.halted

    def get_halt_reason(self) -> str:
        """Get the reason for plan halt.

        Returns:
            Halt reason or empty string.
        """
        if self._current_state is None:
            return ""
        return self._current_state.halt_reason

    def reset(self) -> None:
        """Reset the adapter, clearing current plan and state."""
        if self._current_artifact_id:
            self._finalize_artifact("reset")
        self._current_plan = None
        self._current_state = None
        if self._budget:
            self._budget = self._budget.clone()
        self._halt_checker = HaltChecker(self._halt_checker.spec)
        self._step_start_time = None
        self._files_touched = []

    def get_parallel_tasks(self, max_workers: int = 4) -> list[ControllerTaskSpec]:
        """Get the next batch of tasks that can run in parallel.
        
        Args:
            max_workers: Maximum concurrent tasks.
            
        Returns:
            List of task specs.
        """
        if self._current_plan is None or self._current_state is None:
            raise ValueError("No active plan")
            
        # Check halt
        if self._current_state.halted or self._override_manager.should_halt():
            return []
            
        steps = self._planner.get_parallel_batch(
            self._current_plan, 
            self._current_state,
            max_workers=max_workers,
        )
        
        specs = []
        for step in steps:
            if self._override_manager.should_skip(step.step_id):
                self._current_state.completed_steps.append(step.step_id)
                continue
                
            step = self._override_manager.apply(step)
            specs.append(step.get_task_spec())
            
        # Start timing (will be reset for each individual outcome process)
        self._step_start_time = time.monotonic() 
        
        return specs

