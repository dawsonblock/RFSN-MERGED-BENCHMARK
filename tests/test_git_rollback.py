"""Tests for git_rollback module."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from rfsn_controller.git_rollback import (
    GitRollbackError,
    GitRollbackManager,
    Snapshot,
)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    
    # Initialize git repo
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    
    # Create initial commit
    (repo / "file.txt").write_text("initial content")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    
    return repo


class TestGitRollbackManager:
    """Tests for GitRollbackManager."""
    
    def test_create_snapshot(self, git_repo: Path) -> None:
        """Test creating a snapshot."""
        mgr = GitRollbackManager()
        
        # Make a change
        (git_repo / "file.txt").write_text("modified content")
        
        # Create snapshot
        snap = mgr.create_snapshot(str(git_repo), "test_label")
        
        assert snap.snapshot_id.startswith("rfsn_beam_")
        assert snap.label == "test_label"
        assert snap.stash_ref == "stash@{0}"
    
    def test_create_snapshot_no_changes(self, git_repo: Path) -> None:
        """Test creating snapshot with no local changes."""
        mgr = GitRollbackManager()
        
        # No changes made
        snap = mgr.create_snapshot(str(git_repo), "empty")
        
        # Should still create a snapshot marker
        assert snap.snapshot_id.startswith("rfsn_beam_")
        assert snap.stash_ref is None  # No stash created
    
    def test_restore_snapshot(self, git_repo: Path) -> None:
        """Test restoring a snapshot."""
        mgr = GitRollbackManager()
        
        # Make a change
        (git_repo / "file.txt").write_text("modified content")
        
        # Create snapshot (stashes changes)
        snap = mgr.create_snapshot(str(git_repo), "before_mod")
        
        # File should be back to original after stash
        content = (git_repo / "file.txt").read_text()
        assert content == "initial content"
        
        # Restore snapshot
        mgr.restore(str(git_repo), snap.snapshot_id)
        
        # Changes should be back
        content = (git_repo / "file.txt").read_text()
        assert content == "modified content"
    
    def test_restore_nonexistent_snapshot(self, git_repo: Path) -> None:
        """Test error when restoring nonexistent snapshot."""
        mgr = GitRollbackManager()
        
        with pytest.raises(GitRollbackError, match="Snapshot not found"):
            mgr.restore(str(git_repo), "nonexistent_id")
    
    def test_discard_snapshot(self, git_repo: Path) -> None:
        """Test discarding a snapshot."""
        mgr = GitRollbackManager()
        
        # Make a change and snapshot
        (git_repo / "file.txt").write_text("modified")
        snap = mgr.create_snapshot(str(git_repo), "to_discard")
        
        # Discard
        result = mgr.discard(str(git_repo), snap.snapshot_id)
        assert result is True
        
        # Should no longer be in snapshots
        assert snap.snapshot_id not in [s.snapshot_id for s in mgr.list_snapshots()]
    
    def test_list_snapshots(self, git_repo: Path) -> None:
        """Test listing snapshots."""
        mgr = GitRollbackManager()
        
        # Create multiple snapshots
        (git_repo / "file.txt").write_text("mod1")
        snap1 = mgr.create_snapshot(str(git_repo), "snap1")
        
        (git_repo / "file.txt").write_text("mod2")
        snap2 = mgr.create_snapshot(str(git_repo), "snap2")
        
        snaps = mgr.list_snapshots()
        assert len(snaps) == 2
        assert set(s.label for s in snaps) == {"snap1", "snap2"}
    
    def test_cleanup_old_snapshots(self, git_repo: Path) -> None:
        """Test automatic cleanup of old snapshots."""
        mgr = GitRollbackManager(max_snapshots=2)
        
        # Create 3 snapshots (exceeds max of 2)
        for i in range(3):
            (git_repo / "file.txt").write_text(f"mod{i}")
            mgr.create_snapshot(str(git_repo), f"snap{i}")
        
        # Should only have 2 snapshots
        assert len(mgr.list_snapshots()) == 2
    
    def test_cleanup_all(self, git_repo: Path) -> None:
        """Test cleaning up all snapshots."""
        mgr = GitRollbackManager()
        
        # Create multiple snapshots
        for i in range(3):
            (git_repo / "file.txt").write_text(f"mod{i}")
            mgr.create_snapshot(str(git_repo), f"snap{i}")
        
        # Cleanup all
        count = mgr.cleanup_all(str(git_repo))
        
        assert count >= 3
        assert len(mgr.list_snapshots()) == 0
    
    def test_save_restore_working_state(self, git_repo: Path) -> None:
        """Test save/restore working state convenience methods."""
        mgr = GitRollbackManager()
        
        # Make changes
        (git_repo / "file.txt").write_text("working changes")
        (git_repo / "new_file.txt").write_text("new file")
        
        # Save working state
        snap = mgr.save_working_state(str(git_repo))
        
        # Files should be stashed
        assert (git_repo / "file.txt").read_text() == "initial content"
        assert not (git_repo / "new_file.txt").exists()
        
        # Restore working state
        mgr.restore_working_state(str(git_repo), snap)
        
        # Changes should be back
        assert (git_repo / "file.txt").read_text() == "working changes"
        assert (git_repo / "new_file.txt").exists()
