"""
WorkspaceManager: Decoupled workspace and git operations.

This module provides secure workspace management with path validation,
git worktree support, and filesystem hygiene. It ensures all operations
stay within the designated repository boundaries and follow security best
practices.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .structured_logging import get_logger

logger = get_logger(__name__)


@dataclass
class GitResult:
    """Result of a git operation."""
    
    returncode: int
    stdout: str
    stderr: str
    success: bool
    
    @property
    def output(self) -> str:
        """Get combined output."""
        return self.stdout + self.stderr


class WorkspaceManager:
    """
    Manages the lifecycle of a git-based workspace.
    
    The WorkspaceManager handles:
    - Path validation and security checks
    - Git operations (clone, worktree, diff, etc.)
    - Filesystem hygiene (forbidden paths, cleanup)
    - Isolated testing via git worktrees
    
    Security Features:
    - Path traversal prevention
    - Forbidden path filtering (.git, node_modules, etc.)
    - Command timeout enforcement
    - Proper error handling
    
    Example:
        >>> manager = WorkspaceManager(root="/tmp/rfsn-workspace", repo_dir="/tmp/rfsn-workspace/repo")
        >>> safe_path = manager.resolve_path("src/main.py")
        >>> if manager.is_safe_path("src/main.py"):
        ...     # Safe to operate on this path
        ...     pass
        >>> worktree_path = manager.make_worktree("feature-branch")
        >>> # Do isolated testing in worktree
        >>> manager.cleanup_worktree(worktree_path)
    """

    def __init__(self, root: str, repo_dir: str):
        """
        Initialize the workspace manager.
        
        Args:
            root: Root directory for workspace operations
            repo_dir: Path to the git repository
        """
        self.root = Path(root).resolve()
        self.repo_dir = Path(repo_dir).resolve()
        self.worktree_counter = 0
        self.worktrees: list[Path] = []
        self.forbidden_prefixes = [
            ".git/",
            "node_modules/",
            ".venv/",
            "venv/",
            "__pycache__/",
            ".pytest_cache/",
            ".mypy_cache/",
            ".ruff_cache/",
            "dist/",
            "build/",
            "*.egg-info/"
        ]
        
        logger.info("Workspace manager initialized", root=str(self.root), repo_dir=str(self.repo_dir))

    def resolve_path(self, path: str) -> str:
        """
        Resolve and validate a path to ensure it's within the repo.
        
        This method prevents path traversal attacks by ensuring all resolved
        paths stay within the repository boundaries.
        
        Args:
            path: Path to resolve (absolute or relative)
            
        Returns:
            Resolved absolute path string
            
        Raises:
            ValueError: If path traversal is detected
        """
        if os.path.isabs(path):
            target = Path(path).resolve()
        else:
            target = (self.repo_dir / path).resolve()
            
        if not target.is_relative_to(self.repo_dir):
            logger.error("Path traversal blocked", path=path, target=str(target))
            raise ValueError(f"Security Violation: Path traversal attempt blocked: {path}")
            
        return str(target)

    def is_safe_path(self, p: str) -> bool:
        """
        Return True if the relative path is outside forbidden prefixes.
        
        Args:
            p: Path to check (relative or absolute)
            
        Returns:
            True if path is safe to operate on
        """
        # Normalize path - handle both ./ and .\ prefixes
        p = p.replace("\\", "/")
        while p.startswith("./"):
            p = p[2:]
        
        # Check against forbidden prefixes
        for pref in self.forbidden_prefixes:
            # Strip trailing slashes and wildcards for comparison
            check_pref = pref.rstrip("/").rstrip("*")
            if p.startswith(check_pref + "/") or p == check_pref:
                logger.debug("Path rejected as unsafe", path=p, prefix=pref)
                return False
            
        return True

    def run_git(
        self, 
        args: list[str], 
        timeout: int = 60,
        cwd: str | None = None
    ) -> GitResult:
        """
        Execute a git command safely.
        
        Args:
            args: Git command arguments (without 'git' prefix)
            timeout: Command timeout in seconds
            cwd: Working directory (defaults to repo_dir)
            
        Returns:
            GitResult with command output and status
        """
        cmd = ["git"] + args
        work_dir = Path(cwd) if cwd else self.repo_dir
        
        try:
            logger.debug("Running git command", args=args, cwd=str(work_dir))
            
            result = subprocess.run(
                cmd,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=timeout, check=False
            )
            
            success = result.returncode == 0
            
            if not success:
                logger.warning("Git command failed", args=args, stderr=result.stderr)
            
            return GitResult(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                success=success
            )
            
        except subprocess.TimeoutExpired:
            logger.error("Git command timed out", args=args, timeout=timeout)
            return GitResult(
                returncode=-1,
                stdout="",
                stderr="git command timeout",
                success=False
            )
        except Exception as e:
            logger.error("Git command error", args=args, error=str(e))
            return GitResult(
                returncode=-1,
                stdout="",
                stderr=str(e),
                success=False
            )

    def make_worktree(self, branch: str, base_branch: str = "HEAD") -> str:
        """
        Create a new git worktree for isolated testing.
        
        Git worktrees allow multiple working directories for the same repository,
        enabling isolated patch testing without affecting the main workspace.
        
        Args:
            branch: Name for the new worktree branch
            base_branch: Base branch to create worktree from (default: HEAD)
            
        Returns:
            Path to the created worktree
            
        Raises:
            RuntimeError: If worktree creation fails
        """
        self.worktree_counter += 1
        wt_name = f"wt-{self.worktree_counter}"
        wt_path = self.root / wt_name
        
        # Create worktree
        result = self.run_git([
            "worktree", "add",
            "-b", f"tmp-{self.worktree_counter}",
            str(wt_path),
            base_branch
        ])
        
        if not result.success:
            logger.error("Failed to create worktree", path=str(wt_path), error=result.stderr)
            raise RuntimeError(f"Failed to create worktree: {result.stderr}")
        
        self.worktrees.append(wt_path)
        logger.info("Created worktree", path=str(wt_path), branch=branch)
        
        return str(wt_path)

    def cleanup_worktree(self, worktree_path: str):
        """
        Remove a specific git worktree.
        
        Args:
            worktree_path: Path to the worktree to remove
        """
        wt_path = Path(worktree_path)
        
        if wt_path not in self.worktrees:
            logger.warning("Worktree not managed by this instance", path=str(wt_path))
            return
        
        try:
            # Remove worktree via git
            result = self.run_git(["worktree", "remove", "--force", str(wt_path)])
            
            if result.success:
                self.worktrees.remove(wt_path)
                logger.info("Removed worktree", path=str(wt_path))
            else:
                logger.error("Failed to remove worktree via git", path=str(wt_path), error=result.stderr)
                # Fallback: remove directory manually
                if wt_path.exists():
                    shutil.rmtree(wt_path, ignore_errors=True)
                    self.worktrees.remove(wt_path)
                    
        except Exception as e:
            logger.error("Error removing worktree", path=str(wt_path), error=str(e))

    def cleanup(self):
        """
        Cleanup the entire workspace.
        
        This removes all worktrees and optionally the root workspace directory.
        """
        # Cleanup all worktrees
        for wt_path in list(self.worktrees):
            self.cleanup_worktree(str(wt_path))
        
        # Remove root if it exists and is managed by us
        if self.root.exists() and self.root != self.repo_dir:
            try:
                shutil.rmtree(self.root, ignore_errors=True)
                logger.info("Cleaned up workspace", root=str(self.root))
            except Exception as e:
                logger.error("Error cleaning up workspace", root=str(self.root), error=str(e))

    def get_file_tree(self, include_hidden: bool = False) -> list[str]:
        """
        Get list of all files in the repository.
        
        Args:
            include_hidden: Whether to include hidden files/directories
            
        Returns:
            List of relative file paths
        """
        result = self.run_git(["ls-files"])
        
        if not result.success:
            logger.error("Failed to list files", error=result.stderr)
            return []
        
        files = result.stdout.strip().split("\n")
        
        if not include_hidden:
            files = [f for f in files if self.is_safe_path(f)]
        
        return files

    def get_diff(self, file_path: str | None = None) -> str:
        """
        Get git diff for the repository or specific file.
        
        Args:
            file_path: Optional specific file to diff
            
        Returns:
            Diff output as string
        """
        args = ["diff"]
        if file_path:
            args.append(file_path)
        
        result = self.run_git(args)
        return result.stdout if result.success else ""
