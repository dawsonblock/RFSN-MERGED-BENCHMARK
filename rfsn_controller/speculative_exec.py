"""Speculative execution for predictive preloading.

Pre-runs likely next steps to reduce perceived latency.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SpeculativeResult:
    """Result of a speculative computation."""
    
    key: str
    result: Any
    computed_at: float
    hit: bool = False  # Whether it was actually used
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self.computed_at


@dataclass
class SpeculativeExecutor:
    """Pre-compute likely next operations.
    
    Uses prediction to start work before it's needed:
    1. Pre-parse likely files
    2. Pre-generate likely prompts
    3. Pre-run likely commands
    """
    
    max_concurrent: int = 4
    max_cache_size: int = 50
    max_age_seconds: float = 300.0  # 5 minutes
    
    _cache: OrderedDict = field(default_factory=OrderedDict)
    _pending: dict[str, asyncio.Task] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _stats: dict[str, int] = field(default_factory=lambda: {
        "hits": 0, "misses": 0, "speculative_runs": 0
    })
    
    def speculate(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        priority: int = 0,
    ) -> None:
        """Start speculative computation if not already running.
        
        Args:
            key: Unique key for this computation.
            compute_fn: Function to compute the result.
            priority: Higher priority tasks run first.
        """
        with self._lock:
            # Already computed or pending?
            if key in self._cache or key in self._pending:
                return
            
            # Too many pending?
            if len(self._pending) >= self.max_concurrent:
                # Could implement priority queue here
                return
        
        # Start background computation
        async def run_speculative():
            try:
                result = compute_fn()
                with self._lock:
                    self._cache[key] = SpeculativeResult(
                        key=key,
                        result=result,
                        computed_at=time.time(),
                    )
                    self._stats["speculative_runs"] += 1
                    self._evict_old()
            finally:
                with self._lock:
                    self._pending.pop(key, None)
        
        # Schedule in background
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                task = loop.create_task(run_speculative())
                with self._lock:
                    self._pending[key] = task
        except RuntimeError:
            # No event loop, run synchronously in thread
            threading.Thread(
                target=lambda: asyncio.run(run_speculative()),
                daemon=True,
            ).start()
    
    def get(self, key: str) -> Any | None:
        """Get speculated result if available.
        
        Args:
            key: Key to look up.
            
        Returns:
            Cached result or None.
        """
        with self._lock:
            if key in self._cache:
                result = self._cache[key]
                
                # Check age
                if result.age_seconds > self.max_age_seconds:
                    del self._cache[key]
                    return None
                
                result.hit = True
                self._stats["hits"] += 1
                return result.result
            
            self._stats["misses"] += 1
            return None
    
    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Any],
    ) -> Any:
        """Get cached result or compute now.
        
        Args:
            key: Key to look up.
            compute_fn: Function to compute if not cached.
            
        Returns:
            Cached or computed result.
        """
        result = self.get(key)
        if result is not None:
            return result
        
        # Compute now
        return compute_fn()
    
    def _evict_old(self) -> None:
        """Remove old entries to stay under limit."""
        while len(self._cache) > self.max_cache_size:
            self._cache.popitem(last=False)
    
    @property
    def hit_rate(self) -> float:
        """Get the cache hit rate."""
        total = self._stats["hits"] + self._stats["misses"]
        return self._stats["hits"] / total if total > 0 else 0.0


@dataclass
class PredictivePreloader:
    """Predict and preload likely next operations.
    
    Learns from patterns to predict what will be needed next.
    """
    
    executor: SpeculativeExecutor = field(default_factory=SpeculativeExecutor)
    
    # Pattern tracking
    _sequences: list[list[str]] = field(default_factory=list)
    _current_sequence: list[str] = field(default_factory=list)
    _transitions: dict[str, dict[str, int]] = field(default_factory=dict)
    
    def record_action(self, action: str) -> None:
        """Record an action to learn patterns.
        
        Args:
            action: The action that occurred.
        """
        self._current_sequence.append(action)
        
        # Build transition counts
        if len(self._current_sequence) >= 2:
            prev = self._current_sequence[-2]
            if prev not in self._transitions:
                self._transitions[prev] = {}
            self._transitions[prev][action] = (
                self._transitions[prev].get(action, 0) + 1
            )
    
    def predict_next(self, current_action: str, top_k: int = 3) -> list[str]:
        """Predict likely next actions.
        
        Args:
            current_action: Current action.
            top_k: Number of predictions to return.
            
        Returns:
            List of likely next actions, most likely first.
        """
        if current_action not in self._transitions:
            return []
        
        transitions = self._transitions[current_action]
        sorted_actions = sorted(
            transitions.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return [action for action, _ in sorted_actions[:top_k]]
    
    def preload_for_action(
        self,
        current_action: str,
        loaders: dict[str, Callable[[], Any]],
    ) -> None:
        """Preload data for predicted next actions.
        
        Args:
            current_action: Current action.
            loaders: Map of action names to loader functions.
        """
        predictions = self.predict_next(current_action)
        
        for predicted_action in predictions:
            if predicted_action in loaders:
                self.executor.speculate(
                    key=f"preload:{predicted_action}",
                    compute_fn=loaders[predicted_action],
                )


# Common file preloading patterns
def create_file_preloader(repo_dir: str) -> PredictivePreloader:
    """Create a preloader configured for file operations.
    
    Args:
        repo_dir: Repository directory.
        
    Returns:
        Configured preloader.
    """
    preloader = PredictivePreloader()
    
    # Pre-configure common patterns
    # After reading a test file, likely need test runner
    preloader._transitions["read_test"] = {"run_test": 5, "read_source": 3}
    # After a test failure, likely need to read source
    preloader._transitions["test_fail"] = {"read_source": 5, "generate_patch": 4}
    # After patching, likely run tests again
    preloader._transitions["apply_patch"] = {"run_test": 5}
    
    return preloader


# Global executor instance
_executor: SpeculativeExecutor | None = None
_executor_lock = threading.Lock()


def get_speculative_executor() -> SpeculativeExecutor:
    """Get the global speculative executor."""
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = SpeculativeExecutor()
        return _executor
