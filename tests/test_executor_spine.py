"""Tests for the unified executor spine."""

import tempfile
from pathlib import Path

from rfsn_controller.executor_spine import GovernedExecutor


def test_governed_executor_basic():
    """Test basic GovernedExecutor functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        executor = GovernedExecutor(
            repo_dir=tmpdir,
            verify_argv=["pwd"],
            timeout_sec=10,
        )
        
        # Test that executor was created
        assert executor.repo_dir == str(Path(tmpdir).resolve())
        assert executor.verify_argv == ["pwd"]
        assert executor.timeout_sec == 10


def test_governed_executor_read_file():
    """Test read_file step type."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test file
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("Hello, World!")
        
        executor = GovernedExecutor(repo_dir=tmpdir, timeout_sec=10)
        
        step = {
            "id": "read_test",
            "type": "read_file",
            "path": "test.txt",
        }
        
        result = executor.execute_step(step)
        
        assert result.ok is True
        assert result.stdout == "Hello, World!"
        assert result.exit_code == 0


def test_governed_executor_grep():
    """Test grep step type."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        (Path(tmpdir) / "file1.txt").write_text("hello world")
        (Path(tmpdir) / "file2.txt").write_text("hello python")
        
        executor = GovernedExecutor(repo_dir=tmpdir, timeout_sec=10)
        
        step = {
            "id": "grep_test",
            "type": "grep",
            "query": "hello",
        }
        
        result = executor.execute_step(step)
        
        assert result.ok is True
        assert "hello" in result.stdout


def test_governed_executor_unknown_step_type():
    """Test handling of unknown step type."""
    with tempfile.TemporaryDirectory() as tmpdir:
        executor = GovernedExecutor(repo_dir=tmpdir, timeout_sec=10)
        
        step = {
            "id": "unknown_test",
            "type": "unknown_type",
        }
        
        result = executor.execute_step(step)
        
        assert result.ok is False
        assert "Unknown step type" in result.stderr


def test_governed_executor_path_traversal_protection():
    """Test that path traversal is blocked."""
    with tempfile.TemporaryDirectory() as tmpdir:
        executor = GovernedExecutor(repo_dir=tmpdir, timeout_sec=10)
        
        step = {
            "id": "traversal_test",
            "type": "read_file",
            "path": "../../../etc/passwd",
        }
        
        result = executor.execute_step(step)
        
        assert result.ok is False
        assert "escapes repo" in result.stderr or "not found" in result.stderr


def test_governed_executor_empty_diff():
    """Test that empty diff is rejected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        executor = GovernedExecutor(repo_dir=tmpdir, timeout_sec=10)
        
        step = {
            "id": "patch_test",
            "type": "apply_patch",
            "diff": "",
        }
        
        result = executor.execute_step(step)
        
        assert result.ok is False
        assert "non-empty diff" in result.stderr

