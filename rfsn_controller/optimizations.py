"""Advanced optimizations for the RFSN controller.

This module provides:
1. Lazy loading of heavy dependencies
2. Subprocess connection pooling
3. Response compression for caching
4. Early termination heuristics
5. Retry with exponential backoff
"""

from __future__ import annotations

import gzip
import hashlib
import os
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, TypeVar

# ============================================================================
# LAZY LOADING
# ============================================================================

class LazyModule:
    """Lazy-loading wrapper for heavy modules.
    
    Only imports the module when first accessed, saving startup time.
    """
    
    def __init__(self, module_name: str):
        self._module_name = module_name
        self._module = None
        self._lock = threading.Lock()
    
    def _load(self):
        if self._module is None:
            with self._lock:
                if self._module is None:
                    import importlib
                    self._module = importlib.import_module(self._module_name)
        return self._module
    
    def __getattr__(self, name: str):
        return getattr(self._load(), name)


# Pre-configured lazy modules for heavy dependencies
lazy_numpy = LazyModule("numpy")
lazy_pandas = LazyModule("pandas") 
lazy_torch = LazyModule("torch")
lazy_transformers = LazyModule("transformers")


def lazy_import(module_name: str) -> LazyModule:
    """Create a lazy-loading module wrapper.
    
    Args:
        module_name: Name of the module to lazily import.
        
    Returns:
        LazyModule wrapper.
    """
    return LazyModule(module_name)


# ============================================================================
# SUBPROCESS POOL
# ============================================================================
# SECURITY NOTE: This module has been refactored to eliminate interactive shell
# usage. The previous implementation used persistent bash shells ("/bin/bash -i")
# which posed shell injection risks. The new implementation uses direct argv-based
# command execution only, with no shell=True or interactive shells.
# ============================================================================

@dataclass
class CommandResult:
    """Result of a command execution.
    
    Security: This class represents the result of direct argv-based command
    execution. No shell interpolation is involved.
    """
    stdout: str
    stderr: str
    returncode: int
    elapsed_time: float = 0.0
    
    @property
    def ok(self) -> bool:
        """Check if command succeeded (returncode == 0)."""
        return self.returncode == 0


@dataclass
class SubprocessPool:
    """Pool for managing concurrent command execution.
    
    Security Improvements (Phase 1 Refactor):
    -----------------------------------------
    - REMOVED: Persistent interactive bash shells ("/bin/bash -i")
    - REMOVED: Writing commands to shell stdin (shell injection vector)
    - ADDED: Direct argv-based command execution using subprocess.run
    - ADDED: Explicit shell=False enforcement
    - ADDED: Concurrency limiting via semaphore
    
    This pool now provides:
    1. Concurrency control (max parallel commands)
    2. Timeout management
    3. Resource tracking
    
    All commands are executed directly via subprocess.run with argv lists,
    never through a shell interpreter.
    """
    
    max_workers: int = 4
    default_timeout: float = 60.0  # seconds
    
    # Internal state - managed via __post_init__
    _semaphore: threading.Semaphore | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _active_count: int = field(default=0, repr=False)
    _total_executed: int = field(default=0, repr=False)
    
    def __post_init__(self):
        """Initialize the subprocess pool with secure defaults."""
        self._semaphore = threading.Semaphore(self.max_workers)
        self._lock = threading.Lock()
        self._active_count = 0
        self._total_executed = 0
    
    def _validate_argv(self, argv: list[str]) -> None:
        """Validate command is a proper argv list.
        
        Security: Rejects shell wrappers and ensures argv-only execution.
        
        Args:
            argv: Command arguments as list.
            
        Raises:
            ValueError: If argv is invalid or contains shell wrapper patterns.
        """
        if not isinstance(argv, list):
            raise ValueError("Command must be a list of strings (argv format)")
        
        if len(argv) == 0:
            raise ValueError("Command argv list cannot be empty")
        
        if not all(isinstance(arg, str) for arg in argv):
            raise ValueError("All command arguments must be strings")
        
        # Security: Reject shell wrapper patterns that could bypass argv safety
        # These patterns indicate an attempt to invoke a shell interpreter
        shell_wrappers = {"sh", "bash", "dash", "zsh", "ksh", "csh", "tcsh"}
        base_cmd = os.path.basename(argv[0])
        if len(argv) >= 2 and base_cmd in shell_wrappers:
            # Check for -c flag (shell command execution)
            if "-c" in argv[1:3]:
                raise ValueError(
                    f"Shell wrapper detected: {base_cmd} -c is not allowed. "
                    "Use direct argv execution instead."
                )
            # Check for -i flag (interactive shell)
            if "-i" in argv[1:3]:
                raise ValueError(
                    f"Interactive shell detected: {base_cmd} -i is not allowed. "
                    "Use direct argv execution instead."
                )
    
    def run_command(
        self,
        argv: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        capture_output: bool = True,
    ) -> CommandResult:
        """Execute a command using direct argv-based execution.
        
        Security: This method executes commands directly via subprocess.run
        with shell=False. No shell interpolation occurs. Commands are validated
        to reject shell wrappers like "sh -c" or "bash -i".
        
        Args:
            argv: Command as list of strings (e.g., ["ls", "-la", "/tmp"]).
                  Must NOT be a shell wrapper (sh -c, bash -i, etc.).
            cwd: Working directory for command execution.
            env: Environment variables (merged with current environment).
            timeout: Maximum execution time in seconds. Defaults to pool timeout.
            capture_output: Whether to capture stdout/stderr.
            
        Returns:
            CommandResult with stdout, stderr, returncode, and timing.
            
        Raises:
            ValueError: If argv is invalid or contains shell patterns.
            TimeoutError: If command exceeds timeout.
        """
        # Security: Validate argv format and reject shell wrappers
        self._validate_argv(argv)
        
        timeout = timeout or self.default_timeout
        
        # Acquire semaphore slot for concurrency control
        self._semaphore.acquire()
        try:
            with self._lock:
                self._active_count += 1
            
            start_time = time.time()
            
            try:
                # Security: Execute with shell=False (explicit for clarity)
                # This ensures the command is executed directly via execve(),
                # not through a shell interpreter.
                result = subprocess.run(
                    argv,
                    cwd=cwd,
                    env={**os.environ, **(env or {})} if env else None,
                    capture_output=capture_output,
                    text=True,
                    timeout=timeout,
                    shell=False, check=False,  # SECURITY: Explicit shell=False
                )
                
                elapsed = time.time() - start_time
                
                return CommandResult(
                    stdout=result.stdout or "",
                    stderr=result.stderr or "",
                    returncode=result.returncode,
                    elapsed_time=elapsed,
                )
                
            except subprocess.TimeoutExpired as e:
                elapsed = time.time() - start_time
                return CommandResult(
                    stdout=e.stdout or "" if hasattr(e, 'stdout') else "",
                    stderr=e.stderr or "" if hasattr(e, 'stderr') else "",
                    returncode=-1,
                    elapsed_time=elapsed,
                )
        finally:
            with self._lock:
                self._active_count -= 1
                self._total_executed += 1
            self._semaphore.release()
    
    def acquire(self) -> bool:
        """Acquire an execution slot from the pool.
        
        Returns:
            True if slot acquired, False if pool is at capacity.
            
        Note: This method is provided for backward API compatibility.
        For new code, prefer using run_command() directly.
        """
        acquired = self._semaphore.acquire(blocking=False)
        if acquired:
            with self._lock:
                self._active_count += 1
        return acquired
    
    def release(self, worker: Any = None) -> None:
        """Release an execution slot back to the pool.
        
        Args:
            worker: Ignored. Kept for backward API compatibility.
            
        Note: This method is provided for backward API compatibility.
        For new code, prefer using run_command() directly.
        """
        with self._lock:
            if self._active_count > 0:
                self._active_count -= 1
        self._semaphore.release()
    
    def cleanup(self) -> None:
        """Perform cleanup operations.
        
        Note: With the refactored design using direct subprocess.run,
        there are no persistent processes to clean up. This method is
        retained for backward API compatibility and future use.
        """
        # No persistent processes to clean up in the new secure design.
        # Each command runs in its own subprocess which terminates on completion.
        pass
    
    def shutdown(self) -> None:
        """Shutdown the pool.
        
        Note: With the refactored design using direct subprocess.run,
        there are no persistent processes to terminate. This method is
        retained for backward API compatibility.
        """
        # No persistent processes to terminate in the new secure design.
        # Reset internal state for clean shutdown.
        with self._lock:
            self._active_count = 0
    
    @property
    def active_count(self) -> int:
        """Get the number of currently active command executions."""
        with self._lock:
            return self._active_count
    
    @property
    def total_executed(self) -> int:
        """Get the total number of commands executed."""
        with self._lock:
            return self._total_executed


# Global subprocess pool
_subprocess_pool: SubprocessPool | None = None


def get_subprocess_pool() -> SubprocessPool:
    """Get the global subprocess pool.
    
    Returns:
        The shared SubprocessPool instance for concurrent command execution.
        
    Security: The pool uses direct argv-based execution with shell=False.
    No interactive shells or shell wrappers are used.
    """
    global _subprocess_pool
    if _subprocess_pool is None:
        _subprocess_pool = SubprocessPool()
    return _subprocess_pool


# ============================================================================
# RESPONSE COMPRESSION
# ============================================================================

def compress_response(content: str) -> bytes:
    """Compress a response string for storage.
    
    Args:
        content: The string to compress.
        
    Returns:
        Compressed bytes.
    """
    return gzip.compress(content.encode("utf-8"))


def decompress_response(data: bytes) -> str:
    """Decompress a stored response.
    
    Args:
        data: Compressed bytes.
        
    Returns:
        Decompressed string.
    """
    return gzip.decompress(data).decode("utf-8")


def compress_if_large(content: str, threshold: int = 1000) -> tuple[bool, bytes]:
    """Compress content only if it's large enough to benefit.
    
    Args:
        content: The string to potentially compress.
        threshold: Minimum size in bytes to trigger compression.
        
    Returns:
        (is_compressed, data) tuple.
    """
    content_bytes = content.encode("utf-8")
    if len(content_bytes) >= threshold:
        compressed = gzip.compress(content_bytes)
        # Only use compression if it actually saves space
        if len(compressed) < len(content_bytes) * 0.9:
            return True, compressed
    return False, content_bytes


def decompress_if_needed(is_compressed: bool, data: bytes) -> str:
    """Decompress data if it was compressed.
    
    Args:
        is_compressed: Whether the data is compressed.
        data: The data bytes.
        
    Returns:
        Decoded string.
    """
    if is_compressed:
        return gzip.decompress(data).decode("utf-8")
    return data.decode("utf-8")


# ============================================================================
# EARLY TERMINATION HEURISTICS
# ============================================================================

@dataclass
class TerminationHeuristics:
    """Heuristics for early termination to save compute time.
    
    Tuned for fast feedback: stops quickly on repeated failures or stale patches.
    """
    
    # Minimum steps before considering early termination
    min_steps: int = 2  # Reduced from 3 for faster feedback
    
    # Maximum consecutive failures before terminating
    max_consecutive_failures: int = 3  # Reduced from 5
    
    # Maximum similar patches (by hash) before terminating
    max_similar_patches: int = 2  # Reduced from 3
    
    # Success rate threshold (if below this after min_steps, terminate)
    min_success_rate: float = 0.05
    
    # Internal state
    _patch_hashes: deque = field(default_factory=lambda: deque(maxlen=20))
    _consecutive_failures: int = 0
    _total_attempts: int = 0
    _successful_attempts: int = 0
    
    def __post_init__(self):
        self._patch_hashes = deque(maxlen=20)
        self._consecutive_failures = 0
        self._total_attempts = 0
        self._successful_attempts = 0
    
    def record_attempt(self, diff: str, success: bool) -> None:
        """Record a patch attempt.
        
        Args:
            diff: The patch diff.
            success: Whether the patch succeeded.
        """
        self._total_attempts += 1
        
        if success:
            self._successful_attempts += 1
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
        
        # Track patch hash
        patch_hash = hashlib.sha256(diff.encode()).hexdigest()[:16]
        self._patch_hashes.append(patch_hash)
    
    def should_terminate(self) -> tuple[bool, str]:
        """Check if we should terminate early.
        
        Returns:
            (should_terminate, reason) tuple.
        """
        # Check consecutive failures
        if self._consecutive_failures >= self.max_consecutive_failures:
            return True, f"Too many consecutive failures ({self._consecutive_failures})"
        
        # Check similar patches
        if len(self._patch_hashes) >= self.max_similar_patches:
            recent = list(self._patch_hashes)[-self.max_similar_patches:]
            if len(set(recent)) == 1:
                return True, "Repeated identical patches"
        
        # Check success rate after minimum steps
        if self._total_attempts >= self.min_steps:
            rate = self._successful_attempts / self._total_attempts
            if rate < self.min_success_rate:
                return True, f"Success rate too low ({rate:.1%})"
        
        return False, ""
    
    def reset(self) -> None:
        """Reset heuristics state."""
        self._patch_hashes.clear()
        self._consecutive_failures = 0
        self._total_attempts = 0
        self._successful_attempts = 0


# ============================================================================
# RETRY WITH BACKOFF
# ============================================================================

T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retry with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay between retries.
        max_delay: Maximum delay between retries.
        exponential_base: Base for exponential backoff.
        retryable_exceptions: Tuple of exceptions to retry on.
        
    Returns:
        Decorator function.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay,
                        )
                        time.sleep(delay)
                    else:
                        raise
            
            raise last_exception  # Should never reach here
        
        return wrapper
    return decorator


# ============================================================================
# MEMOIZATION WITH TTL
# ============================================================================

def memoize_with_ttl(ttl_seconds: float = 300.0, maxsize: int = 128):
    """Decorator for memoization with time-to-live.
    
    Args:
        ttl_seconds: Time-to-live for cached values.
        maxsize: Maximum cache size.
        
    Returns:
        Decorator function.
    """
    def decorator(func):
        cache: dict[str, tuple[float, Any]] = {}
        lock = threading.Lock()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = str((args, tuple(sorted(kwargs.items()))))
            
            now = time.time()
            
            with lock:
                # Check cache
                if key in cache:
                    timestamp, value = cache[key]
                    if now - timestamp < ttl_seconds:
                        return value
                    else:
                        del cache[key]
                
                # Compute value
                value = func(*args, **kwargs)
                
                # Store in cache
                if len(cache) >= maxsize:
                    # Remove oldest entry
                    oldest_key = min(cache, key=lambda k: cache[k][0])
                    del cache[oldest_key]
                
                cache[key] = (now, value)
                return value
        
        def clear_cache():
            with lock:
                cache.clear()
        
        wrapper.clear_cache = clear_cache
        return wrapper
    
    return decorator


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def batch_process(
    items: list[Any],
    processor: Callable[[Any], Any],
    batch_size: int = 10,
    max_workers: int = 4,
) -> list[Any]:
    """Process items in batches with parallel execution.
    
    Args:
        items: Items to process.
        processor: Function to apply to each item.
        batch_size: Number of items per batch.
        max_workers: Maximum parallel workers.
        
    Returns:
        List of processed results in same order.
    """
    from concurrent.futures import ThreadPoolExecutor
    
    results = [None] * len(items)
    
    def process_item(idx_item):
        idx, item = idx_item
        return idx, processor(item)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, result in executor.map(process_item, enumerate(items)):
            results[idx] = result
    
    return results


# ============================================================================
# RESOURCE LIMITS
# ============================================================================

@dataclass
class ResourceLimits:
    """Resource limits for operations."""
    
    max_memory_mb: int = 4096
    max_cpu_seconds: float = 300.0
    max_file_size_mb: int = 100
    max_output_size_mb: int = 10
    
    def check_memory(self) -> bool:
        """Check if memory usage is within limits."""
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            memory_mb = usage.ru_maxrss / 1024  # Convert to MB
            return memory_mb < self.max_memory_mb
        except Exception:
            return True
    
    def limit_output(self, output: str) -> str:
        """Limit output size to prevent memory issues."""
        max_chars = self.max_output_size_mb * 1024 * 1024
        if len(output) > max_chars:
            return output[:max_chars] + f"\n... [truncated {len(output) - max_chars} chars]"
        return output


# ============================================================================
# INITIALIZATION
# ============================================================================

def init_optimizations() -> None:
    """Initialize all optimization systems."""
    # Cleanup interval configurable via env var (default: 30s)
    cleanup_interval = int(os.getenv("RFSN_CACHE_CLEANUP_INTERVAL", "30"))
    
    # Start subprocess pool cleanup thread
    def cleanup_loop():
        while True:
            time.sleep(cleanup_interval)
            try:
                pool = get_subprocess_pool()
                pool.cleanup()
            except Exception:
                pass
    
    thread = threading.Thread(target=cleanup_loop, daemon=True)
    thread.start()


def shutdown_optimizations() -> None:
    """Shutdown all optimization systems."""
    global _subprocess_pool
    if _subprocess_pool:
        _subprocess_pool.shutdown()
        _subprocess_pool = None
