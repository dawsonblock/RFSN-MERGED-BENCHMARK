"""Budget tracking and enforcement for the RFSN controller.

This module provides resource budget management to prevent runaway executions
and ensure predictable resource consumption. It tracks:
- Steps (main loop iterations)
- LLM calls
- Tokens (prompt + completion)
- Time (wall-clock seconds)
- Subprocess calls

Budget enforcement supports soft warnings (at configurable thresholds) and
hard limits that raise exceptions to halt execution.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# Lazy import to avoid circular dependency
def _log_budget_event(event_type: str, resource: str, current: int, limit: int) -> None:
    """Log budget events to the global event logger if available."""
    try:
        from .events import (
            log_budget_exceeded_global,
            log_budget_warning_global,
        )
        
        if event_type == "warning":
            percentage = (current / limit * 100) if limit > 0 else 0
            log_budget_warning_global(resource, current, limit, percentage)
        elif event_type == "exceeded":
            log_budget_exceeded_global(resource, current, limit)
    except ImportError:
        pass  # Events module not available


class BudgetState(Enum):
    """Budget consumption state."""
    
    ACTIVE = "active"         # Under warning threshold
    WARNING = "warning"       # At or above warning threshold, but under limit
    EXCEEDED = "exceeded"     # At or above hard limit (operation blocked)
    EXHAUSTED = "exhausted"   # All budgets consumed, no further operations allowed


class BudgetExceeded(Exception):
    """Raised when a budget limit is exceeded.
    
    This exception signals that the controller should halt execution
    because resource limits have been reached.
    """
    
    def __init__(
        self,
        resource: str,
        current: int,
        limit: int,
        message: str | None = None,
    ):
        self.resource = resource
        self.current = current
        self.limit = limit
        self.message = message or f"Budget exceeded for {resource}: {current}/{limit}"
        super().__init__(self.message)
    
    def __repr__(self) -> str:
        return f"BudgetExceeded({self.resource!r}, current={self.current}, limit={self.limit})"


@dataclass
class Budget:
    """Resource budget tracker with consumption tracking and enforcement.
    
    This class tracks resource consumption across multiple dimensions and
    provides methods for checking limits, getting current state, and
    enforcing hard limits via exceptions.
    
    All operations are thread-safe.
    
    Attributes:
        max_steps: Maximum number of main loop iterations.
        max_llm_calls: Maximum number of LLM API calls.
        max_tokens: Maximum total tokens (prompt + completion).
        max_time_seconds: Maximum wall-clock execution time in seconds.
        max_subprocess_calls: Maximum number of subprocess invocations.
        warning_threshold: Percentage (0.0-1.0) at which WARNING state is triggered.
    """
    
    # Hard limits (0 or None means unlimited)
    max_steps: int = 0
    max_llm_calls: int = 0
    max_tokens: int = 0
    max_time_seconds: float = 0
    max_subprocess_calls: int = 0
    
    # Warning threshold (default: 80%)
    warning_threshold: float = 0.8
    
    # Internal counters
    _steps: int = field(default=0, repr=False)
    _llm_calls: int = field(default=0, repr=False)
    _tokens: int = field(default=0, repr=False)
    _subprocess_calls: int = field(default=0, repr=False)
    _start_time: float | None = field(default=None, repr=False)
    
    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    
    # Optional callbacks for state changes
    _on_warning: Callable[[str, int, int], None] | None = field(default=None, repr=False)
    _on_exceeded: Callable[[str, int, int], None] | None = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize the start time for time tracking."""
        if self._start_time is None:
            self._start_time = time.time()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Consumption tracking methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def record_step(self, count: int = 1) -> None:
        """Record step(s) consumed and check limits.
        
        Args:
            count: Number of steps to record.
            
        Raises:
            BudgetExceeded: If step limit is exceeded.
        """
        with self._lock:
            self._steps += count
            self._check_and_enforce("steps", self._steps, self.max_steps)
    
    def record_llm_call(self, tokens: int = 0) -> None:
        """Record an LLM call and optional token usage.
        
        Args:
            tokens: Number of tokens used in this call.
            
        Raises:
            BudgetExceeded: If LLM call or token limit is exceeded.
        """
        with self._lock:
            self._llm_calls += 1
            self._tokens += tokens
            self._check_and_enforce("llm_calls", self._llm_calls, self.max_llm_calls)
            self._check_and_enforce("tokens", self._tokens, self.max_tokens)
    
    def record_tokens(self, count: int) -> None:
        """Record token consumption without incrementing call count.
        
        Args:
            count: Number of tokens to record.
            
        Raises:
            BudgetExceeded: If token limit is exceeded.
        """
        with self._lock:
            self._tokens += count
            self._check_and_enforce("tokens", self._tokens, self.max_tokens)
    
    def record_subprocess_call(self, count: int = 1) -> None:
        """Record subprocess call(s) and check limits.
        
        Args:
            count: Number of subprocess calls to record.
            
        Raises:
            BudgetExceeded: If subprocess call limit is exceeded.
        """
        with self._lock:
            self._subprocess_calls += count
            self._check_and_enforce("subprocess_calls", self._subprocess_calls, self.max_subprocess_calls)
    
    def check_time_budget(self) -> None:
        """Check if time budget is exceeded.
        
        This should be called periodically during execution.
        
        Raises:
            BudgetExceeded: If time limit is exceeded.
        """
        if self.max_time_seconds <= 0:
            return
        
        with self._lock:
            elapsed = self.elapsed_seconds
            if elapsed >= self.max_time_seconds:
                _log_budget_event("exceeded", "time_seconds", int(elapsed), int(self.max_time_seconds))
                if self._on_exceeded:
                    self._on_exceeded("time_seconds", int(elapsed), int(self.max_time_seconds))
                raise BudgetExceeded(
                    resource="time_seconds",
                    current=int(elapsed),
                    limit=int(self.max_time_seconds),
                )
            elif elapsed >= self.max_time_seconds * self.warning_threshold:
                logger.warning(
                    f"Budget warning: time_seconds at {elapsed:.1f}/{self.max_time_seconds:.1f}"
                )
                _log_budget_event("warning", "time_seconds", int(elapsed), int(self.max_time_seconds))
                if self._on_warning:
                    self._on_warning("time_seconds", int(elapsed), int(self.max_time_seconds))
    
    # ─────────────────────────────────────────────────────────────────────────
    # Query methods
    # ─────────────────────────────────────────────────────────────────────────
    
    @property
    def steps(self) -> int:
        """Current step count."""
        with self._lock:
            return self._steps
    
    @property
    def llm_calls(self) -> int:
        """Current LLM call count."""
        with self._lock:
            return self._llm_calls
    
    @property
    def tokens(self) -> int:
        """Current token count."""
        with self._lock:
            return self._tokens
    
    @property
    def subprocess_calls(self) -> int:
        """Current subprocess call count."""
        with self._lock:
            return self._subprocess_calls
    
    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time in seconds since budget tracking started."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time
    
    @property
    def remaining_steps(self) -> int | None:
        """Remaining steps, or None if unlimited."""
        if self.max_steps <= 0:
            return None
        with self._lock:
            return max(0, self.max_steps - self._steps)
    
    @property
    def remaining_llm_calls(self) -> int | None:
        """Remaining LLM calls, or None if unlimited."""
        if self.max_llm_calls <= 0:
            return None
        with self._lock:
            return max(0, self.max_llm_calls - self._llm_calls)
    
    @property
    def remaining_tokens(self) -> int | None:
        """Remaining tokens, or None if unlimited."""
        if self.max_tokens <= 0:
            return None
        with self._lock:
            return max(0, self.max_tokens - self._tokens)
    
    @property
    def remaining_time_seconds(self) -> float | None:
        """Remaining time in seconds, or None if unlimited."""
        if self.max_time_seconds <= 0:
            return None
        return max(0.0, self.max_time_seconds - self.elapsed_seconds)
    
    @property
    def remaining_subprocess_calls(self) -> int | None:
        """Remaining subprocess calls, or None if unlimited."""
        if self.max_subprocess_calls <= 0:
            return None
        with self._lock:
            return max(0, self.max_subprocess_calls - self._subprocess_calls)
    
    def get_state(self) -> BudgetState:
        """Get the current overall budget state.
        
        Returns:
            The most severe state across all tracked resources.
        """
        states = [
            self._get_resource_state("steps", self._steps, self.max_steps),
            self._get_resource_state("llm_calls", self._llm_calls, self.max_llm_calls),
            self._get_resource_state("tokens", self._tokens, self.max_tokens),
            self._get_resource_state("subprocess_calls", self._subprocess_calls, self.max_subprocess_calls),
        ]
        
        # Check time separately
        if self.max_time_seconds > 0:
            elapsed = self.elapsed_seconds
            if elapsed >= self.max_time_seconds:
                states.append(BudgetState.EXCEEDED)
            elif elapsed >= self.max_time_seconds * self.warning_threshold:
                states.append(BudgetState.WARNING)
        
        # Return the most severe state
        if BudgetState.EXCEEDED in states or BudgetState.EXHAUSTED in states:
            # Check if ALL resources are exhausted
            all_exhausted = all(
                s in (BudgetState.EXCEEDED, BudgetState.EXHAUSTED) 
                for s in states if s != BudgetState.ACTIVE
            )
            return BudgetState.EXHAUSTED if all_exhausted and BudgetState.EXCEEDED in states else BudgetState.EXCEEDED
        if BudgetState.WARNING in states:
            return BudgetState.WARNING
        return BudgetState.ACTIVE
    
    def get_resource_states(self) -> dict[str, BudgetState]:
        """Get the state of each tracked resource.
        
        Returns:
            Dictionary mapping resource names to their states.
        """
        with self._lock:
            states = {
                "steps": self._get_resource_state("steps", self._steps, self.max_steps),
                "llm_calls": self._get_resource_state("llm_calls", self._llm_calls, self.max_llm_calls),
                "tokens": self._get_resource_state("tokens", self._tokens, self.max_tokens),
                "subprocess_calls": self._get_resource_state(
                    "subprocess_calls", self._subprocess_calls, self.max_subprocess_calls
                ),
            }
            
            # Check time
            if self.max_time_seconds > 0:
                elapsed = self.elapsed_seconds
                if elapsed >= self.max_time_seconds:
                    states["time_seconds"] = BudgetState.EXCEEDED
                elif elapsed >= self.max_time_seconds * self.warning_threshold:
                    states["time_seconds"] = BudgetState.WARNING
                else:
                    states["time_seconds"] = BudgetState.ACTIVE
            
            return states
    
    def get_usage_summary(self) -> dict[str, dict[str, float]]:
        """Get a summary of resource usage.
        
        Returns:
            Dictionary with usage information for each resource.
        """
        with self._lock:
            return {
                "steps": {
                    "current": self._steps,
                    "limit": self.max_steps,
                    "remaining": self.max_steps - self._steps if self.max_steps > 0 else -1,
                    "percentage": (self._steps / self.max_steps * 100) if self.max_steps > 0 else 0,
                },
                "llm_calls": {
                    "current": self._llm_calls,
                    "limit": self.max_llm_calls,
                    "remaining": self.max_llm_calls - self._llm_calls if self.max_llm_calls > 0 else -1,
                    "percentage": (self._llm_calls / self.max_llm_calls * 100) if self.max_llm_calls > 0 else 0,
                },
                "tokens": {
                    "current": self._tokens,
                    "limit": self.max_tokens,
                    "remaining": self.max_tokens - self._tokens if self.max_tokens > 0 else -1,
                    "percentage": (self._tokens / self.max_tokens * 100) if self.max_tokens > 0 else 0,
                },
                "time_seconds": {
                    "current": self.elapsed_seconds,
                    "limit": self.max_time_seconds,
                    "remaining": self.max_time_seconds - self.elapsed_seconds if self.max_time_seconds > 0 else -1,
                    "percentage": (self.elapsed_seconds / self.max_time_seconds * 100) if self.max_time_seconds > 0 else 0,
                },
                "subprocess_calls": {
                    "current": self._subprocess_calls,
                    "limit": self.max_subprocess_calls,
                    "remaining": self.max_subprocess_calls - self._subprocess_calls if self.max_subprocess_calls > 0 else -1,
                    "percentage": (self._subprocess_calls / self.max_subprocess_calls * 100) if self.max_subprocess_calls > 0 else 0,
                },
            }
    
    def is_within_budget(self) -> bool:
        """Check if all resources are within their limits.
        
        Returns:
            True if all resources are under their limits.
        """
        state = self.get_state()
        return state in (BudgetState.ACTIVE, BudgetState.WARNING)
    
    def reset(self) -> None:
        """Reset all counters and restart the time tracker.
        
        This is useful for testing or starting a new execution phase.
        """
        with self._lock:
            self._steps = 0
            self._llm_calls = 0
            self._tokens = 0
            self._subprocess_calls = 0
            self._start_time = time.time()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Callback registration
    # ─────────────────────────────────────────────────────────────────────────
    
    def on_warning(self, callback: Callable[[str, int, int], None]) -> None:
        """Register a callback for warning state transitions.
        
        Args:
            callback: Function called with (resource_name, current, limit).
        """
        self._on_warning = callback
    
    def on_exceeded(self, callback: Callable[[str, int, int], None]) -> None:
        """Register a callback for exceeded state transitions.
        
        Args:
            callback: Function called with (resource_name, current, limit).
        """
        self._on_exceeded = callback
    
    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────
    
    def _get_resource_state(self, name: str, current: int, limit: int) -> BudgetState:
        """Get the state for a single resource.
        
        Args:
            name: Resource name for logging.
            current: Current consumption.
            limit: Hard limit (0 means unlimited).
            
        Returns:
            The resource's current state.
        """
        if limit <= 0:
            return BudgetState.ACTIVE
        
        if current >= limit:
            return BudgetState.EXCEEDED
        elif current >= limit * self.warning_threshold:
            return BudgetState.WARNING
        return BudgetState.ACTIVE
    
    def _check_and_enforce(self, resource: str, current: int, limit: int) -> None:
        """Check resource usage and enforce limits.
        
        Args:
            resource: Resource name.
            current: Current consumption.
            limit: Hard limit (0 means unlimited).
            
        Raises:
            BudgetExceeded: If limit is exceeded.
        """
        if limit <= 0:
            return  # Unlimited
        
        if current >= limit:
            logger.error(f"Budget exceeded: {resource} at {current}/{limit}")
            _log_budget_event("exceeded", resource, current, limit)
            if self._on_exceeded:
                self._on_exceeded(resource, current, limit)
            raise BudgetExceeded(resource=resource, current=current, limit=limit)
        
        if current >= limit * self.warning_threshold:
            logger.warning(f"Budget warning: {resource} at {current}/{limit}")
            _log_budget_event("warning", resource, current, limit)
            if self._on_warning:
                self._on_warning(resource, current, limit)


def create_budget(
    max_steps: int = 0,
    max_llm_calls: int = 0,
    max_tokens: int = 0,
    max_time_seconds: float = 0,
    max_subprocess_calls: int = 0,
    warning_threshold: float = 0.8,
) -> Budget:
    """Factory function to create a Budget instance.
    
    Args:
        max_steps: Maximum steps (0 = unlimited).
        max_llm_calls: Maximum LLM calls (0 = unlimited).
        max_tokens: Maximum tokens (0 = unlimited).
        max_time_seconds: Maximum execution time (0 = unlimited).
        max_subprocess_calls: Maximum subprocess calls (0 = unlimited).
        warning_threshold: Percentage for warning state (default 0.8 = 80%).
        
    Returns:
        Configured Budget instance.
    """
    return Budget(
        max_steps=max_steps,
        max_llm_calls=max_llm_calls,
        max_tokens=max_tokens,
        max_time_seconds=max_time_seconds,
        max_subprocess_calls=max_subprocess_calls,
        warning_threshold=warning_threshold,
    )


# Singleton for global budget access (optional pattern for simple integrations)
_global_budget: Budget | None = None
_global_budget_lock = threading.Lock()


def get_global_budget() -> Budget | None:
    """Get the global budget instance, if set.
    
    Returns:
        The global Budget instance or None if not set.
    """
    with _global_budget_lock:
        return _global_budget


def set_global_budget(budget: Budget | None) -> None:
    """Set the global budget instance.
    
    Args:
        budget: Budget instance to set globally, or None to clear.
    """
    global _global_budget
    with _global_budget_lock:
        _global_budget = budget


def record_subprocess_call_global() -> None:
    """Record a subprocess call to the global budget if set.
    
    This is a convenience function for use in modules that don't have
    direct access to the budget instance.
    """
    budget = get_global_budget()
    if budget is not None:
        budget.record_subprocess_call()


def record_llm_call_global(tokens: int = 0) -> None:
    """Record an LLM call to the global budget if set.
    
    Args:
        tokens: Number of tokens consumed.
    """
    budget = get_global_budget()
    if budget is not None:
        budget.record_llm_call(tokens)


def check_time_budget_global() -> None:
    """Check time budget on the global budget if set.
    
    Raises:
        BudgetExceeded: If time limit is exceeded.
    """
    budget = get_global_budget()
    if budget is not None:
        budget.check_time_budget()
