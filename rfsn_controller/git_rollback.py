"""Git-based rollback system for safe hypothesis testing.

Provides snapshot/restore capabilities for exploring multiple patch
candidates without polluting the repository state.
"""

from __future__ import annotations

import subprocess
import time
import uuid
from dataclasses import dataclass


@dataclass
class Snapshot:
    """A git snapshot for rollback."""
    
    snapshot_id: str
    label: str
    created_at: float
    stash_ref: str | None = None  # For stash-based snapshots
    commit_ref: str | None = None  # For commit-based snapshots


class GitRollbackError(Exception):
    """Error during git rollback operations."""
    pass


class GitRollbackManager:
    """Manage git snapshots for safe hypothesis testing.
    
    Uses git stash for lightweight, fast snapshots that can be
    quickly restored during beam search exploration.
    
    Example:
        rollback = GitRollbackManager()
        
        # Create snapshot before modifying
        snap = rollback.create_snapshot(repo_path, "before_patch_1")
        
        # Apply patch and test...
        apply_patch(patch)
        result = run_tests()
        
        # Rollback to try different patch
        rollback.restore(repo_path, snap.snapshot_id)
    """
    
    SNAPSHOT_PREFIX = "rfsn_beam_"
    
    def __init__(self, max_snapshots: int = 20):
        """Initialize the rollback manager.
        
        Args:
            max_snapshots: Maximum snapshots to keep per repo.
        """
        self.max_snapshots = max_snapshots
        self._snapshots: dict[str, Snapshot] = {}
    
    def _run_git(
        self,
        repo_path: str,
        args: list[str],
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run a git command in the repo.
        
        Args:
            repo_path: Path to the git repository.
            args: Git command arguments.
            check: Whether to raise on non-zero exit.
            
        Returns:
            Completed process result.
            
        Raises:
            GitRollbackError: If command fails and check=True.
        """
        cmd = ["git", "-C", repo_path, *args]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if check and result.returncode != 0:
                raise GitRollbackError(
                    f"Git command failed: {' '.join(args)}\n"
                    f"stderr: {result.stderr}"
                )
            return result
        except subprocess.TimeoutExpired as e:
            raise GitRollbackError(f"Git command timed out: {' '.join(args)}") from e
    
    def create_snapshot(self, repo_path: str, label: str = "") -> Snapshot:
        """Create a snapshot of the current repo state.
        
        Uses git stash to save all changes (staged, unstaged, untracked).
        
        Args:
            repo_path: Path to the git repository.
            label: Optional label for the snapshot.
            
        Returns:
            Snapshot object with ID for restoration.
        """
        snapshot_id = f"{self.SNAPSHOT_PREFIX}{uuid.uuid4().hex[:8]}"
        stash_msg = f"{snapshot_id}:{label}" if label else snapshot_id
        
        # Stash all changes including untracked files
        # Use --include-untracked to capture new files
        result = self._run_git(
            repo_path,
            ["stash", "push", "--include-untracked", "-m", stash_msg],
            check=False,
        )
        
        # Check if anything was stashed (ternary for lint compliance)
        stash_ref = None if "No local changes to save" in result.stdout else "stash@{0}"
        
        snapshot = Snapshot(
            snapshot_id=snapshot_id,
            label=label,
            created_at=time.time(),
            stash_ref=stash_ref,
        )
        
        self._snapshots[snapshot_id] = snapshot
        
        # Cleanup old snapshots if over limit
        self._cleanup_if_needed(repo_path)
        
        return snapshot
    
    def restore(self, repo_path: str, snapshot_id: str) -> bool:
        """Restore repo to a previous snapshot.
        
        Args:
            repo_path: Path to the git repository.
            snapshot_id: ID of the snapshot to restore.
            
        Returns:
            True if restored successfully.
            
        Raises:
            GitRollbackError: If snapshot not found or restore fails.
        """
        if snapshot_id not in self._snapshots:
            raise GitRollbackError(f"Snapshot not found: {snapshot_id}")
        
        snapshot = self._snapshots[snapshot_id]
        
        # First, clear any current changes
        self._run_git(repo_path, ["reset", "--hard", "HEAD"])
        self._run_git(repo_path, ["clean", "-fd"])
        
        if snapshot.stash_ref:
            # Find the stash by message
            stash_list = self._run_git(
                repo_path,
                ["stash", "list"],
                check=False,
            )
            
            stash_index = None
            for line in stash_list.stdout.splitlines():
                if snapshot_id in line:
                    # Extract index from "stash@{N}: ..."
                    stash_index = line.split(":")[0]
                    break
            
            if stash_index:
                # Pop the stash to restore changes
                self._run_git(
                    repo_path,
                    ["stash", "pop", stash_index],
                    check=False,
                )
        
        return True
    
    def discard(self, repo_path: str, snapshot_id: str) -> bool:
        """Discard a snapshot without restoring.
        
        Args:
            repo_path: Path to the git repository.
            snapshot_id: ID of the snapshot to discard.
            
        Returns:
            True if discarded successfully.
        """
        if snapshot_id not in self._snapshots:
            return False
        
        snapshot = self._snapshots.pop(snapshot_id)
        
        if snapshot.stash_ref:
            # Find and drop the stash
            stash_list = self._run_git(
                repo_path,
                ["stash", "list"],
                check=False,
            )
            
            for line in stash_list.stdout.splitlines():
                if snapshot_id in line:
                    stash_index = line.split(":")[0]
                    self._run_git(
                        repo_path,
                        ["stash", "drop", stash_index],
                        check=False,
                    )
                    break
        
        return True
    
    def list_snapshots(self, repo_path: str | None = None) -> list[Snapshot]:
        """List available snapshots.
        
        Args:
            repo_path: Optional path to sync with git stash list.
            
        Returns:
            List of Snapshot objects.
        """
        return list(self._snapshots.values())
    
    def _cleanup_if_needed(self, repo_path: str) -> None:
        """Remove old snapshots if over the limit.
        
        Args:
            repo_path: Path to the git repository.
        """
        if len(self._snapshots) <= self.max_snapshots:
            return
        
        # Sort by creation time, oldest first
        sorted_snaps = sorted(
            self._snapshots.values(),
            key=lambda s: s.created_at,
        )
        
        # Remove oldest until under limit
        to_remove = len(self._snapshots) - self.max_snapshots
        for snap in sorted_snaps[:to_remove]:
            self.discard(repo_path, snap.snapshot_id)
    
    def cleanup_all(self, repo_path: str) -> int:
        """Remove all RFSN snapshots from the repo.
        
        Args:
            repo_path: Path to the git repository.
            
        Returns:
            Number of snapshots cleaned up.
        """
        count = 0
        
        # Clear tracked snapshots
        for snap_id in list(self._snapshots.keys()):
            if self.discard(repo_path, snap_id):
                count += 1
        
        # Also clean any orphaned stashes
        stash_list = self._run_git(
            repo_path,
            ["stash", "list"],
            check=False,
        )
        
        indices_to_drop = []
        for line in stash_list.stdout.splitlines():
            if self.SNAPSHOT_PREFIX in line:
                stash_index = line.split(":")[0]
                indices_to_drop.append(stash_index)
        
        # Drop in reverse order to maintain indices
        for stash_index in reversed(indices_to_drop):
            self._run_git(
                repo_path,
                ["stash", "drop", stash_index],
                check=False,
            )
            count += 1
        
        return count
    
    def save_working_state(self, repo_path: str) -> Snapshot:
        """Save current working state before beam search.
        
        Creates a special snapshot that preserves all current changes
        so beam search can explore without losing work.
        
        Args:
            repo_path: Path to the git repository.
            
        Returns:
            Snapshot for later restoration.
        """
        return self.create_snapshot(repo_path, "beam_search_start")
    
    def restore_working_state(self, repo_path: str, snapshot: Snapshot) -> bool:
        """Restore working state after beam search.
        
        Args:
            repo_path: Path to the git repository.
            snapshot: Snapshot from save_working_state.
            
        Returns:
            True if restored successfully.
        """
        return self.restore(repo_path, snapshot.snapshot_id)
