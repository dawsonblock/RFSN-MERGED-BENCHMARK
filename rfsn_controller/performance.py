"""Docker image pre-warming and worktree pooling for performance optimization.

This module provides:
1. Pre-warming of Docker images in the background
2. Worktree pooling for faster parallel patch evaluation
3. Async utilities for non-blocking operations
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Empty, Queue

from .sandbox import Sandbox

# ============================================================================
# DOCKER IMAGE PRE-WARMING
# ============================================================================

# Default images to pre-warm
DEFAULT_BUILDPACK_IMAGES: list[str] = [
    "python:3.11-slim",
    "python:3.12-slim",
    "node:20-slim",
    "node:22-slim",
    "rust:1.75-slim",
    "golang:1.22-alpine",
    "mcr.microsoft.com/dotnet/sdk:8.0",
    "maven:3.9-eclipse-temurin-21",
    "ruby:3.3-slim",
]

_prewarm_thread: threading.Thread | None = None
_prewarm_status: dict[str, str] = {}  # image -> status (pending/pulling/ready/failed)
_prewarm_lock = threading.Lock()


def _pull_image(image: str) -> bool:
    """Pull a Docker image if not already present.
    
    Args:
        image: Docker image name to pull.
        
    Returns:
        True if image is now available.
    """
    global _prewarm_status
    
    with _prewarm_lock:
        _prewarm_status[image] = "checking"
    
    try:
        # Check if image exists locally
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            timeout=10, check=False,
        )
        
        if result.returncode == 0:
            with _prewarm_lock:
                _prewarm_status[image] = "ready"
            return True
        
        # Pull the image
        with _prewarm_lock:
            _prewarm_status[image] = "pulling"
        
        result = subprocess.run(
            ["docker", "pull", image],
            capture_output=True,
            timeout=300, check=False,  # 5 minute timeout per image
        )
        
        if result.returncode == 0:
            with _prewarm_lock:
                _prewarm_status[image] = "ready"
            return True
        else:
            with _prewarm_lock:
                _prewarm_status[image] = "failed"
            return False
            
    except subprocess.TimeoutExpired:
        with _prewarm_lock:
            _prewarm_status[image] = "timeout"
        return False
    except FileNotFoundError:
        # Docker not installed
        with _prewarm_lock:
            _prewarm_status[image] = "docker_not_found"
        return False
    except Exception as e:
        with _prewarm_lock:
            _prewarm_status[image] = f"error: {str(e)[:50]}"
        return False


def prewarm_images(
    images: list[str] | None = None,
    max_workers: int = 3,
    blocking: bool = False,
) -> None:
    """Pre-warm Docker images in the background.
    
    Args:
        images: List of images to prewarm. Uses defaults if None.
        max_workers: Number of parallel pull threads.
        blocking: If True, wait for all images to be pulled.
    """
    global _prewarm_thread
    
    if images is None:
        images = DEFAULT_BUILDPACK_IMAGES.copy()
    
    def _prewarm_worker():
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(_pull_image, images)
    
    if blocking:
        _prewarm_worker()
    else:
        # Start background thread
        _prewarm_thread = threading.Thread(target=_prewarm_worker, daemon=True)
        _prewarm_thread.start()


def get_prewarm_status() -> dict[str, str]:
    """Get current pre-warming status for all images.
    
    Returns:
        Dict mapping image name to status string.
    """
    with _prewarm_lock:
        return _prewarm_status.copy()


def is_image_ready(image: str) -> bool:
    """Check if an image is ready (pre-warmed or available).
    
    Args:
        image: Docker image name.
        
    Returns:
        True if image is ready for use.
    """
    with _prewarm_lock:
        status = _prewarm_status.get(image)
        if status == "ready":
            return True
    
    # Check directly if not in cache
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            timeout=5, check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


# ============================================================================
# WORKTREE POOL
# ============================================================================

@dataclass
class WorktreePool:
    """Pool of reusable git worktrees for faster parallel patch evaluation.
    
    Instead of creating/destroying worktrees for each patch, this pool
    maintains a set of pre-created worktrees that can be reused.
    """
    
    sb: Sandbox
    pool_size: int = 5
    
    _available: Queue = field(default_factory=Queue)
    _in_use: set[str] = field(default_factory=set)
    _initialized: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def __post_init__(self):
        """Initialize the pool with worktrees."""
        self._available = Queue()
        self._in_use = set()
        self._lock = threading.Lock()
    
    def initialize(self) -> None:
        """Create the pool of worktrees.
        
        Call this once after creating the pool.
        """
        if self._initialized:
            return
        
        for i in range(self.pool_size):
            try:
                wt = self._create_worktree(f"pool_{i:02d}")
                self._available.put(wt)
            except Exception as e:
                print(f"[WorktreePool] Failed to create worktree {i}: {e}")
        
        self._initialized = True
    
    def _create_worktree(self, suffix: str) -> str:
        """Create a new worktree with the given suffix."""
        wt_path = os.path.join(self.sb.root, f"wt_{suffix}")
        
        # Use subprocess directly for reliability
        result = subprocess.run(
            ["git", "worktree", "add", "--detach", wt_path],
            cwd=self.sb.repo_dir,
            capture_output=True,
            timeout=60, check=False,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create worktree: {result.stderr.decode()}")
        
        return wt_path
    
    def acquire(self, timeout: float = 30.0) -> str | None:
        """Acquire a worktree from the pool.
        
        Args:
            timeout: Maximum time to wait for a worktree.
            
        Returns:
            Path to worktree, or None if timeout.
        """
        try:
            wt = self._available.get(timeout=timeout)
            with self._lock:
                self._in_use.add(wt)
            return wt
        except Empty:
            return None
    
    def release(self, wt_path: str) -> None:
        """Release a worktree back to the pool.
        
        The worktree is reset to match the main repo HEAD.
        
        Args:
            wt_path: Path to the worktree to release.
        """
        with self._lock:
            if wt_path not in self._in_use:
                return  # Already released or not from this pool
            self._in_use.discard(wt_path)
        
        # Reset worktree to clean state
        try:
            subprocess.run(
                ["git", "reset", "--hard", "HEAD"],
                cwd=wt_path,
                capture_output=True,
                timeout=30, check=False,
            )
            subprocess.run(
                ["git", "clean", "-fd"],
                cwd=wt_path,
                capture_output=True,
                timeout=30, check=False,
            )
            self._available.put(wt_path)
        except Exception as e:
            # If reset fails, create a new worktree
            print(f"[WorktreePool] Reset failed, recreating: {e}")
            try:
                shutil.rmtree(wt_path, ignore_errors=True)
                # Extract suffix from path
                suffix = os.path.basename(wt_path).replace("wt_", "")
                new_wt = self._create_worktree(f"new_{suffix}")
                self._available.put(new_wt)
            except Exception:
                pass  # Pool size will be reduced
    
    def destroy(self) -> None:
        """Destroy all worktrees in the pool."""
        # Drain available queue
        while True:
            try:
                wt = self._available.get_nowait()
                self._cleanup_worktree(wt)
            except Empty:
                break
        
        # Cleanup in-use worktrees
        with self._lock:
            for wt in self._in_use:
                self._cleanup_worktree(wt)
            self._in_use.clear()
        
        self._initialized = False
    
    def _cleanup_worktree(self, wt_path: str) -> None:
        """Remove a worktree completely."""
        try:
            subprocess.run(
                ["git", "worktree", "remove", "--force", wt_path],
                cwd=self.sb.repo_dir,
                capture_output=True,
                timeout=30, check=False,
            )
        except Exception:
            pass
        
        if os.path.exists(wt_path):
            shutil.rmtree(wt_path, ignore_errors=True)
    
    @property
    def available_count(self) -> int:
        """Number of available worktrees."""
        return self._available.qsize()
    
    @property
    def in_use_count(self) -> int:
        """Number of worktrees currently in use."""
        with self._lock:
            return len(self._in_use)
    
    def __enter__(self) -> WorktreePool:
        self.initialize()
        return self
    
    def __exit__(self, *args) -> None:
        self.destroy()


# ============================================================================
# CONTEXT MANAGERS FOR EASY USE
# ============================================================================

class PooledWorktree:
    """Context manager for using a worktree from a pool."""
    
    def __init__(self, pool: WorktreePool, timeout: float = 30.0):
        self.pool = pool
        self.timeout = timeout
        self.wt_path: str | None = None
    
    def __enter__(self) -> str | None:
        self.wt_path = self.pool.acquire(timeout=self.timeout)
        return self.wt_path
    
    def __exit__(self, *args) -> None:
        if self.wt_path:
            self.pool.release(self.wt_path)
            self.wt_path = None


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def start_prewarm_background() -> None:
    """Start pre-warming default images in background.
    
    Safe to call multiple times - will only start once.
    """
    global _prewarm_thread
    
    if _prewarm_thread is not None and _prewarm_thread.is_alive():
        return  # Already running
    
    prewarm_images(blocking=False)


def create_worktree_pool(sb: Sandbox, pool_size: int = 5) -> WorktreePool:
    """Create and initialize a worktree pool.
    
    Args:
        sb: The sandbox to create worktrees in.
        pool_size: Number of worktrees to pre-create.
        
    Returns:
        Initialized WorktreePool.
    """
    pool = WorktreePool(sb=sb, pool_size=pool_size)
    pool.initialize()
    return pool
