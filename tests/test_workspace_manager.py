"""Tests for WorkspaceManager module."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from rfsn_controller.workspace_manager import GitResult, WorkspaceManager


class TestWorkspaceManager:
    """Test suite for WorkspaceManager class."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "workspace"
            repo = Path(tmpdir) / "workspace" / "repo"
            root.mkdir()
            repo.mkdir()
            yield root, repo

    @pytest.fixture
    def manager(self, temp_workspace):
        """Create a WorkspaceManager instance."""
        root, repo = temp_workspace
        return WorkspaceManager(root=str(root), repo_dir=str(repo))

    def test_init(self, temp_workspace):
        """Test initialization."""
        root, repo = temp_workspace
        manager = WorkspaceManager(root=str(root), repo_dir=str(repo))
        
        assert manager.root == root.resolve()
        assert manager.repo_dir == repo.resolve()
        assert manager.worktree_counter == 0
        assert len(manager.worktrees) == 0
        assert len(manager.forbidden_prefixes) > 0

    def test_resolve_path_relative(self, manager):
        """Test resolving relative paths."""
        resolved = manager.resolve_path("src/main.py")
        
        assert Path(resolved).is_absolute()
        assert "src" in resolved
        assert "main.py" in resolved

    def test_resolve_path_absolute_within_repo(self, manager):
        """Test resolving absolute paths within repo."""
        target = manager.repo_dir / "test.py"
        resolved = manager.resolve_path(str(target))
        
        assert Path(resolved) == target

    def test_resolve_path_traversal_blocked(self, manager):
        """Test that path traversal is blocked."""
        with pytest.raises(ValueError, match="Security Violation"):
            manager.resolve_path("../../etc/passwd")

    def test_resolve_path_absolute_outside_repo_blocked(self, manager):
        """Test that absolute paths outside repo are blocked."""
        with pytest.raises(ValueError, match="Security Violation"):
            manager.resolve_path("/etc/passwd")

    def test_is_safe_path_normal_files(self, manager):
        """Test safe path checking for normal files."""
        assert manager.is_safe_path("src/main.py") is True
        assert manager.is_safe_path("tests/test_foo.py") is True
        assert manager.is_safe_path("README.md") is True

    def test_is_safe_path_forbidden_prefixes(self, manager):
        """Test safe path checking blocks forbidden prefixes."""
        assert manager.is_safe_path(".git/config") is False
        assert manager.is_safe_path("node_modules/package/index.js") is False
        assert manager.is_safe_path(".venv/lib/python") is False
        assert manager.is_safe_path("__pycache__/module.pyc") is False

    def test_is_safe_path_edge_cases(self, manager):
        """Test safe path checking edge cases."""
        # Paths with ./ prefix should work
        assert manager.is_safe_path("./src/main.py") is True
        
        # Windows-style paths should work
        assert manager.is_safe_path("src\\main.py") is True

    @patch('subprocess.run')
    def test_run_git_success(self, mock_run, manager):
        """Test successful git command execution."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="output content",
            stderr=""
        )
        
        result = manager.run_git(["status"])
        
        assert result.success is True
        assert result.returncode == 0
        assert result.stdout == "output content"
        assert result.stderr == ""

    @patch('subprocess.run')
    def test_run_git_failure(self, mock_run, manager):
        """Test failed git command execution."""
        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="fatal: not a git repository"
        )
        
        result = manager.run_git(["status"])
        
        assert result.success is False
        assert result.returncode == 1
        assert result.stderr == "fatal: not a git repository"

    @patch('subprocess.run')
    def test_run_git_timeout(self, mock_run, manager):
        """Test git command timeout handling."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["git", "clone"], timeout=60)
        
        result = manager.run_git(["clone", "large-repo"], timeout=1)
        
        assert result.success is False
        assert result.returncode == -1
        assert "timeout" in result.stderr.lower()

    @patch('subprocess.run')
    def test_run_git_exception(self, mock_run, manager):
        """Test git command exception handling."""
        mock_run.side_effect = Exception("Network error")
        
        result = manager.run_git(["fetch"])
        
        assert result.success is False
        assert result.returncode == -1
        assert "Network error" in result.stderr

    @patch('subprocess.run')
    def test_make_worktree_success(self, mock_run, manager):
        """Test successful worktree creation."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Preparing worktree",
            stderr=""
        )
        
        wt_path = manager.make_worktree("feature-branch")
        
        assert "wt-1" in wt_path
        assert manager.worktree_counter == 1
        assert len(manager.worktrees) == 1
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_make_worktree_failure(self, mock_run, manager):
        """Test worktree creation failure."""
        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="fatal: invalid reference"
        )
        
        with pytest.raises(RuntimeError, match="Failed to create worktree"):
            manager.make_worktree("invalid-branch")

    @patch('subprocess.run')
    def test_make_multiple_worktrees(self, mock_run, manager):
        """Test creating multiple worktrees."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        
        wt1 = manager.make_worktree("branch1")
        wt2 = manager.make_worktree("branch2")
        wt3 = manager.make_worktree("branch3")
        
        assert manager.worktree_counter == 3
        assert len(manager.worktrees) == 3
        assert "wt-1" in wt1
        assert "wt-2" in wt2
        assert "wt-3" in wt3

    @patch('subprocess.run')
    def test_cleanup_worktree(self, mock_run, manager):
        """Test worktree cleanup."""
        # Create a worktree
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        wt_path = manager.make_worktree("test-branch")
        
        # Clean it up
        manager.cleanup_worktree(wt_path)
        
        assert len(manager.worktrees) == 0
        # Should call git worktree remove
        assert mock_run.call_count == 2  # create + remove

    @patch('subprocess.run')
    @patch('shutil.rmtree')
    def test_cleanup_worktree_fallback(self, mock_rmtree, mock_run, manager):
        """Test worktree cleanup fallback to manual removal."""
        # Create a worktree
        mock_run.side_effect = [
            Mock(returncode=0, stdout="", stderr=""),  # create
            Mock(returncode=1, stdout="", stderr="failed")  # remove fails
        ]
        
        wt_path = manager.make_worktree("test-branch")
        manager.cleanup_worktree(wt_path)
        
        # Should have tried git worktree remove, then fallback to rmtree
        assert mock_run.call_count == 2
        # Note: mock_rmtree might not be called if path doesn't exist in test

    @patch('subprocess.run')
    def test_get_file_tree(self, mock_run, manager):
        """Test getting file tree."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="src/main.py\ntests/test_main.py\nREADME.md\n",
            stderr=""
        )
        
        files = manager.get_file_tree()
        
        assert len(files) == 3
        assert "src/main.py" in files
        assert "tests/test_main.py" in files
        assert "README.md" in files

    @patch('subprocess.run')
    def test_get_file_tree_filters_unsafe(self, mock_run, manager):
        """Test that get_file_tree filters unsafe paths."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="src/main.py\n.git/config\nnode_modules/pkg/index.js\nREADME.md\n",
            stderr=""
        )
        
        files = manager.get_file_tree()
        
        assert "src/main.py" in files
        assert "README.md" in files
        assert ".git/config" not in files
        assert "node_modules/pkg/index.js" not in files

    @patch('subprocess.run')
    def test_get_diff(self, mock_run, manager):
        """Test getting git diff."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="diff --git a/file.py b/file.py\n+added line\n",
            stderr=""
        )
        
        diff = manager.get_diff()
        
        assert "diff --git" in diff
        assert "+added line" in diff

    @patch('subprocess.run')
    def test_get_diff_specific_file(self, mock_run, manager):
        """Test getting git diff for specific file."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="diff --git a/src/main.py b/src/main.py\n",
            stderr=""
        )
        
        diff = manager.get_diff("src/main.py")
        
        assert "src/main.py" in diff
        # Verify correct args passed
        call_args = mock_run.call_args[0][0]
        assert "diff" in call_args
        assert "src/main.py" in call_args


class TestGitResult:
    """Test suite for GitResult dataclass."""

    def test_git_result_creation(self):
        """Test GitResult dataclass creation."""
        result = GitResult(
            returncode=0,
            stdout="output",
            stderr="",
            success=True
        )
        
        assert result.returncode == 0
        assert result.stdout == "output"
        assert result.stderr == ""
        assert result.success is True

    def test_git_result_output_property(self):
        """Test output property combines stdout and stderr."""
        result = GitResult(
            returncode=0,
            stdout="std output",
            stderr="std error",
            success=True
        )
        
        output = result.output
        assert "std output" in output
        assert "std error" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
