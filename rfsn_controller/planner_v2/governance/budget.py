"""Plan Budget - Resource limits for plan execution.

Manages per-plan caps for:
- Patch cycles
- Failing steps  
- Token consumption
- Wall-clock time

The planner queries remaining budget to simplify plans when resources are tight.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


class BudgetExhausted(Exception):
    """Raised when plan budget is exhausted."""
    
    def __init__(self, resource: str, limit: float, used: float):
        self.resource = resource
        self.limit = limit
        self.used = used
        super().__init__(f"Budget exhausted: {resource} ({used:.1f}/{limit:.1f})")


@dataclass
class PlanBudget:
    """Resource limits for plan execution.
    
    Tracks usage against limits and determines when to simplify plans.
    """
    
    # Limits
    max_patch_cycles: int = 10
    max_failing_steps: int = 3
    max_total_tokens: int = 50000
    max_wall_clock_sec: float = 600.0
    
    # Current usage
    patch_cycles_used: int = 0
    failing_steps_used: int = 0
    tokens_used: int = 0
    start_time: float | None = None
    
    # Thresholds
    simplify_threshold: float = 0.3  # Simplify when <30% remaining
    warning_threshold: float = 0.2  # Warn when <20% remaining
    
    def __post_init__(self) -> None:
        if self.start_time is None:
            self.start_time = time.monotonic()
    
    @property
    def elapsed_sec(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        return time.monotonic() - self.start_time
    
    def record_patch_cycle(self) -> None:
        """Record a patch cycle."""
        self.patch_cycles_used += 1
    
    def record_failing_step(self) -> None:
        """Record a failing step."""
        self.failing_steps_used += 1
    
    def record_tokens(self, tokens: int) -> None:
        """Record token usage."""
        self.tokens_used += tokens
    
    def is_exhausted(self) -> tuple[bool, str | None]:
        """Check if any budget is exhausted.
        
        Returns:
            Tuple of (is_exhausted, resource_name).
        """
        if self.patch_cycles_used >= self.max_patch_cycles:
            return True, "patch_cycles"
        if self.failing_steps_used >= self.max_failing_steps:
            return True, "failing_steps"
        if self.tokens_used >= self.max_total_tokens:
            return True, "tokens"
        if self.elapsed_sec >= self.max_wall_clock_sec:
            return True, "wall_clock"
        return False, None
    
    def check_and_raise(self) -> None:
        """Check budget and raise if exhausted."""
        exhausted, resource = self.is_exhausted()
        if exhausted:
            limits = {
                "patch_cycles": (self.max_patch_cycles, self.patch_cycles_used),
                "failing_steps": (self.max_failing_steps, self.failing_steps_used),
                "tokens": (self.max_total_tokens, self.tokens_used),
                "wall_clock": (self.max_wall_clock_sec, self.elapsed_sec),
            }
            limit, used = limits[resource]
            raise BudgetExhausted(resource, limit, used)
    
    def remaining_fraction(self) -> float:
        """Get minimum remaining fraction across all resources.
        
        Returns:
            Fraction between 0.0 (exhausted) and 1.0 (full).
        """
        fractions = [
            1.0 - (self.patch_cycles_used / max(1, self.max_patch_cycles)),
            1.0 - (self.failing_steps_used / max(1, self.max_failing_steps)),
            1.0 - (self.tokens_used / max(1, self.max_total_tokens)),
            1.0 - (self.elapsed_sec / max(1, self.max_wall_clock_sec)),
        ]
        return max(0.0, min(fractions))
    
    def remaining_by_resource(self) -> dict[str, float]:
        """Get remaining fraction for each resource."""
        return {
            "patch_cycles": 1.0 - (self.patch_cycles_used / max(1, self.max_patch_cycles)),
            "failing_steps": 1.0 - (self.failing_steps_used / max(1, self.max_failing_steps)),
            "tokens": 1.0 - (self.tokens_used / max(1, self.max_total_tokens)),
            "wall_clock": 1.0 - (self.elapsed_sec / max(1, self.max_wall_clock_sec)),
        }
    
    def should_simplify_plan(self) -> bool:
        """Check if plan should be simplified due to tight budget.
        
        Returns:
            True if remaining < simplify_threshold.
        """
        return self.remaining_fraction() < self.simplify_threshold
    
    def should_warn(self) -> bool:
        """Check if warning should be issued.
        
        Returns:
            True if remaining < warning_threshold.
        """
        return self.remaining_fraction() < self.warning_threshold
    
    def get_tight_resources(self) -> list[str]:
        """Get list of resources that are running low."""
        tight = []
        remaining = self.remaining_by_resource()
        for resource, fraction in remaining.items():
            if fraction < self.simplify_threshold:
                tight.append(resource)
        return tight
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "max_patch_cycles": self.max_patch_cycles,
            "max_failing_steps": self.max_failing_steps,
            "max_total_tokens": self.max_total_tokens,
            "max_wall_clock_sec": self.max_wall_clock_sec,
            "patch_cycles_used": self.patch_cycles_used,
            "failing_steps_used": self.failing_steps_used,
            "tokens_used": self.tokens_used,
            "elapsed_sec": self.elapsed_sec,
            "remaining_fraction": self.remaining_fraction(),
            "should_simplify": self.should_simplify_plan(),
        }
    
    @classmethod
    def from_config(
        cls,
        max_steps: int = 10,
        max_failures: int = 3,
        max_tokens: int = 50000,
        max_time_sec: float = 600.0,
    ) -> PlanBudget:
        """Create budget from config values."""
        return cls(
            max_patch_cycles=max_steps,
            max_failing_steps=max_failures,
            max_total_tokens=max_tokens,
            max_wall_clock_sec=max_time_sec,
        )
    
    def clone(self) -> PlanBudget:
        """Create a copy with reset usage."""
        return PlanBudget(
            max_patch_cycles=self.max_patch_cycles,
            max_failing_steps=self.max_failing_steps,
            max_total_tokens=self.max_total_tokens,
            max_wall_clock_sec=self.max_wall_clock_sec,
        )
