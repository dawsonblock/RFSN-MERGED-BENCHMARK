"""Halt Conditions - Plan-level stop conditions.

Triggers plan halt on patterns like:
- Repeated test flakiness
- Widening file touches
- Dependency creep
- Repeated identical failures
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..schema import ControllerOutcome, Plan, PlanState


@dataclass
class StepHistory:
    """History for a single step across attempts."""
    step_id: str
    failure_signatures: list[str] = field(default_factory=list)
    files_touched: set[str] = field(default_factory=set)
    flaky_count: int = 0


@dataclass
class HaltSpec:
    """Specification for plan halt conditions."""
    
    # Maximum consecutive flaky test failures
    max_consecutive_flaky: int = 3
    
    # Maximum rate of file touches per step (files / steps)
    max_file_touch_growth_rate: float = 2.0
    
    # Maximum new dependencies added during plan execution
    max_dependency_creep: int = 5
    
    # Maximum identical failure signatures in a row
    max_identical_failures: int = 3
    
    # Maximum total steps (absolute limit)
    max_total_steps: int = 50
    
    def to_dict(self) -> dict:
        return {
            "max_consecutive_flaky": self.max_consecutive_flaky,
            "max_file_touch_growth_rate": self.max_file_touch_growth_rate,
            "max_dependency_creep": self.max_dependency_creep,
            "max_identical_failures": self.max_identical_failures,
            "max_total_steps": self.max_total_steps,
        }


class HaltChecker:
    """Checks halt conditions during plan execution."""
    
    def __init__(self, spec: HaltSpec | None = None):
        """Initialize with halt specification.
        
        Args:
            spec: Halt conditions. Uses defaults if None.
        """
        self.spec = spec or HaltSpec()
        self.step_history: dict[str, StepHistory] = {}
        self.failure_signatures: list[str] = []
        self.files_touched_per_step: list[int] = []
        self.initial_dependencies: set[str] = set()
        self.current_dependencies: set[str] = set()
        self.flaky_streak: int = 0
        self.total_steps: int = 0
    
    def initialize(self, plan: Plan) -> None:
        """Initialize checker with plan.
        
        Args:
            plan: The plan being executed.
        """
        self.initial_dependencies = set()
        for step in plan.steps:
            self.step_history[step.step_id] = StepHistory(step_id=step.step_id)
        self.total_steps = 0
    
    def record_outcome(
        self,
        step_id: str,
        outcome: ControllerOutcome,
        files_touched: list[str],
        is_flaky: bool = False,
    ) -> None:
        """Record a step outcome for halt checking.
        
        Args:
            step_id: The step that completed.
            outcome: The controller outcome.
            files_touched: Files modified in this step.
            is_flaky: Whether this was a flaky test failure.
        """
        self.total_steps += 1
        self.files_touched_per_step.append(len(files_touched))
        
        # Track step history
        if step_id in self.step_history:
            history = self.step_history[step_id]
            history.files_touched.update(files_touched)
            if outcome.error_message:
                history.failure_signatures.append(outcome.error_message[:100])
        
        # Track failure signatures
        if not outcome.success and outcome.error_message:
            self.failure_signatures.append(outcome.error_message[:100])
        
        # Track flaky streak
        if is_flaky:
            self.flaky_streak += 1
        else:
            self.flaky_streak = 0
    
    def add_dependency(self, dep: str) -> None:
        """Record a new dependency added during execution.
        
        Args:
            dep: The dependency identifier.
        """
        self.current_dependencies.add(dep)
    
    def check(self, plan: Plan, state: PlanState) -> str | None:
        """Check all halt conditions.
        
        Args:
            plan: The current plan.
            state: Current plan state.
            
        Returns:
            Halt reason if any condition triggered, None otherwise.
        """
        # Check flaky streak
        if self.flaky_streak >= self.spec.max_consecutive_flaky:
            return f"Flaky test streak: {self.flaky_streak} consecutive flaky failures"
        
        # Check file touch growth rate
        if len(self.files_touched_per_step) >= 3:
            avg_files = sum(self.files_touched_per_step) / len(self.files_touched_per_step)
            if avg_files > self.spec.max_file_touch_growth_rate:
                return f"File touch rate too high: {avg_files:.1f} files/step (max: {self.spec.max_file_touch_growth_rate})"
        
        # Check dependency creep
        new_deps = self.current_dependencies - self.initial_dependencies
        if len(new_deps) > self.spec.max_dependency_creep:
            return f"Dependency creep: {len(new_deps)} new dependencies (max: {self.spec.max_dependency_creep})"
        
        # Check identical failures
        if len(self.failure_signatures) >= self.spec.max_identical_failures:
            recent = self.failure_signatures[-self.spec.max_identical_failures:]
            if len(set(recent)) == 1:
                return f"Identical failures: same error {self.spec.max_identical_failures} times in a row"
        
        # Check total steps
        if self.total_steps >= self.spec.max_total_steps:
            return f"Maximum steps reached: {self.total_steps} (max: {self.spec.max_total_steps})"
        
        return None
    
    def get_statistics(self) -> dict:
        """Get halt checker statistics."""
        return {
            "total_steps": self.total_steps,
            "flaky_streak": self.flaky_streak,
            "avg_files_per_step": (
                sum(self.files_touched_per_step) / max(1, len(self.files_touched_per_step))
            ),
            "new_dependencies": len(self.current_dependencies - self.initial_dependencies),
            "unique_failure_signatures": len(set(self.failure_signatures)),
        }
