"""Worktree Manager for isolated SWE-bench task execution.

Manages Git worktrees to provide isolated environments for parallel task
execution. Each task runs in its own worktree with no shared mutable state.

INVARIANTS:
1. Worktrees are created in a temporary directory
2. Worktrees are destroyed after task completion
3. No cross-worktree filesystem access
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class WorktreeHandle:
    """Handle to an active worktree.
    
    Attributes:
        path: Absolute path to the worktree directory
        branch: Name of the detached branch
        task_id: Unique identifier for the task using this worktree
        base_commit: The commit SHA the worktree is based on
    """
    path: Path
    branch: str
    task_id: str
    base_commit: str


@dataclass
class WorktreeManager:
    """Manages isolated Git worktrees for parallel task execution.
    
    Each worktree is created in a temporary directory and destroyed
    after the task completes. The manager tracks active worktrees
    to ensure cleanup on exit.
    """
    
    repo_path: Path
    """Path to the main repository"""
    
    base_dir: Path | None = None
    """Base directory for worktrees (default: system temp)"""
    
    active_worktrees: dict[str, WorktreeHandle] = field(default_factory=dict)
    """Map of task_id -> WorktreeHandle for cleanup"""
    
    def __post_init__(self) -> None:
        self.repo_path = Path(self.repo_path).resolve()
        if self.base_dir is None:
            self.base_dir = Path(tempfile.gettempdir()) / "rfsn_worktrees"
        self.base_dir = Path(self.base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def create(
        self,
        task_id: str | None = None,
        commit: str = "HEAD",
    ) -> WorktreeHandle:
        """Create an isolated worktree for a task.
        
        Args:
            task_id: Unique identifier for this task (generated if None)
            commit: Git commit/ref to base the worktree on
            
        Returns:
            WorktreeHandle with the path to the new worktree
            
        Raises:
            RuntimeError: If worktree creation fails
        """
        if task_id is None:
            task_id = str(uuid.uuid4())[:8]
        
        # Get the full commit SHA
        result = subprocess.run(
            ["git", "rev-parse", commit],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to resolve commit {commit}: {result.stderr}")
        base_commit = result.stdout.strip()
        
        # Create unique branch name for this worktree
        branch_name = f"rfsn-wt-{task_id}"
        worktree_path = self.base_dir / f"wt-{task_id}"
        
        # Remove any stale worktree at this path
        if worktree_path.exists():
            self._force_remove_worktree(worktree_path)
        
        # Create the worktree with detached HEAD
        result = subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree_path), base_commit],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create worktree: {result.stderr}")
        
        handle = WorktreeHandle(
            path=worktree_path,
            branch=branch_name,
            task_id=task_id,
            base_commit=base_commit,
        )
        self.active_worktrees[task_id] = handle
        return handle
    
    def destroy(self, handle: WorktreeHandle) -> None:
        """Destroy a worktree and clean up resources.
        
        Args:
            handle: The WorktreeHandle to destroy
        """
        self._force_remove_worktree(handle.path)
        self.active_worktrees.pop(handle.task_id, None)
    
    def _force_remove_worktree(self, path: Path) -> None:
        """Force remove a worktree, handling git bookkeeping."""
        # Try git worktree remove first
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(path)],
            cwd=self.repo_path,
            capture_output=True,
        )
        # Force delete if still exists
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        
        # Prune stale worktree entries
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=self.repo_path,
            capture_output=True,
        )
    
    def cleanup_all(self) -> int:
        """Destroy all active worktrees.
        
        Returns:
            Number of worktrees cleaned up
        """
        count = 0
        for handle in list(self.active_worktrees.values()):
            try:
                self.destroy(handle)
                count += 1
            except Exception:
                pass  # Best effort cleanup
        return count
    
    @contextmanager
    def worktree(
        self,
        task_id: str | None = None,
        commit: str = "HEAD",
    ) -> Iterator[WorktreeHandle]:
        """Context manager for automatic worktree cleanup.
        
        Usage:
            with manager.worktree(task_id="task-1") as wt:
                # Work in wt.path
                pass
            # Worktree is automatically destroyed
        
        Args:
            task_id: Unique identifier for this task
            commit: Git commit/ref to base the worktree on
            
        Yields:
            WorktreeHandle for the created worktree
        """
        handle = self.create(task_id=task_id, commit=commit)
        try:
            yield handle
        finally:
            self.destroy(handle)
    
    def apply_patch(self, handle: WorktreeHandle, diff: str) -> tuple[bool, str]:
        """Apply a patch to a worktree.
        
        Args:
            handle: The worktree to apply the patch to
            diff: The unified diff to apply
            
        Returns:
            (success, message) tuple
        """
        result = subprocess.run(
            ["git", "apply", "--check", "-"],
            cwd=handle.path,
            input=diff,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False, f"Patch check failed: {result.stderr}"
        
        result = subprocess.run(
            ["git", "apply", "-"],
            cwd=handle.path,
            input=diff,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False, f"Patch apply failed: {result.stderr}"
        
        return True, "Patch applied successfully"
    
    def run_command(
        self,
        handle: WorktreeHandle,
        cmd: list[str],
        timeout: int = 300,
    ) -> tuple[int, str, str]:
        """Run a command in a worktree.
        
        Args:
            handle: The worktree to run the command in
            cmd: Command as argv list (no shell)
            timeout: Timeout in seconds
            
        Returns:
            (exit_code, stdout, stderr) tuple
        """
        try:
            result = subprocess.run(
                cmd,
                cwd=handle.path,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout}s"
        except Exception as e:
            return -1, "", f"Command failed: {e}"
