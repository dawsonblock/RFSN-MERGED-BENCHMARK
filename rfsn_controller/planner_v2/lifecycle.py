"""Planner Layer v2 - Step Lifecycle State Machine.

This module manages step state transitions with defined failure policy.

State Machine:
    PENDING -> ACTIVE (deps met) or BLOCKED (deps not met)
    BLOCKED -> PENDING (blocking dep completes)
    ACTIVE -> DONE (success) or FAILED (failure)
    FAILED -> PENDING (revised) or SKIPPED (scope reduced) or HALT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .schema import Plan, RiskLevel, Step, StepStatus

if TYPE_CHECKING:
    pass


class StepLifecycle:
    """Manages step state transitions.

    Implements the step lifecycle state machine with failure policy:
    - First failure: attempt revision
    - Second failure same step: reduce scope or skip if non-critical
    - Third failure same step: HALT plan (no infinite loops)
    """

    MAX_STEP_FAILURES = 2

    @staticmethod
    def can_activate(step: Step, plan: Plan) -> tuple[bool, str]:
        """Check if a step can transition to ACTIVE.

        A step can activate if:
        - It is currently PENDING
        - All its dependencies are DONE

        Args:
            step: The step to check.
            plan: The plan containing the step.

        Returns:
            Tuple of (can_activate, reason).
        """
        if step.status != StepStatus.PENDING:
            return False, f"Step is {step.status.value}, not PENDING"

        # Check all dependencies are DONE
        for dep_id in step.dependencies:
            dep = plan.get_step(dep_id)
            if dep is None:
                return False, f"Unknown dependency: {dep_id}"
            if dep.status != StepStatus.DONE:
                return False, f"Dependency {dep_id} is {dep.status.value}"

        return True, "OK"

    @staticmethod
    def activate(step: Step) -> None:
        """Transition step to ACTIVE.

        Args:
            step: The step to activate.
        """
        step.status = StepStatus.ACTIVE

    @staticmethod
    def complete(step: Step, result: dict[str, Any]) -> None:
        """Transition step to DONE.

        Args:
            step: The step to complete.
            result: The execution result.
        """
        step.status = StepStatus.DONE
        step.result = result

    @staticmethod
    def fail(step: Step, error: str) -> bool:
        """Transition step to FAILED.

        Increments failure count and checks if retry is possible.

        Args:
            step: The step that failed.
            error: Error message describing the failure.

        Returns:
            True if step can be retried, False if max failures reached.
        """
        step.status = StepStatus.FAILED
        step.failure_count += 1
        step.result = {"error": error}
        return step.failure_count < StepLifecycle.MAX_STEP_FAILURES

    @staticmethod
    def skip(step: Step, reason: str) -> None:
        """Transition step to SKIPPED.

        Used for non-critical steps that fail multiple times.

        Args:
            step: The step to skip.
            reason: Reason for skipping.
        """
        step.status = StepStatus.SKIPPED
        step.result = {"skip_reason": reason}

    @staticmethod
    def reset_for_retry(step: Step) -> None:
        """Reset step to PENDING for retry after revision.

        Clears the result but preserves failure_count.

        Args:
            step: The step to reset.
        """
        step.status = StepStatus.PENDING
        step.result = None

    @staticmethod
    def block(step: Step, reason: str) -> None:
        """Transition step to BLOCKED.

        Used when dependencies are not yet met.

        Args:
            step: The step to block.
            reason: Reason for blocking.
        """
        step.status = StepStatus.BLOCKED
        step.result = {"blocked_reason": reason}

    @staticmethod
    def unblock(step: Step) -> None:
        """Transition step from BLOCKED to PENDING.

        Called when blocking dependencies complete.

        Args:
            step: The step to unblock.
        """
        if step.status == StepStatus.BLOCKED:
            step.status = StepStatus.PENDING
            step.result = None

    @staticmethod
    def can_skip(step: Step) -> bool:
        """Check if a step can be safely skipped.

        Only LOW risk steps can be skipped.

        Args:
            step: The step to check.

        Returns:
            True if step can be skipped safely.
        """
        return step.risk_level == RiskLevel.LOW

    @staticmethod
    def should_halt(step: Step) -> bool:
        """Check if plan should halt due to step failures.

        Args:
            step: The step to check.

        Returns:
            True if failures exceed threshold.
        """
        return step.failure_count >= StepLifecycle.MAX_STEP_FAILURES
