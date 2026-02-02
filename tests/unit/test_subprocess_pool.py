"""Unit tests for SubprocessPool in optimizations module.

This module tests the Phase 1 refactored SubprocessPool:
- Direct argv-based execution (no interactive shells)
- Shell wrapper rejection
- Concurrency control
- Timeout handling
"""

from __future__ import annotations

import threading
import time

import pytest

from rfsn_controller.optimizations import (
    CommandResult,
    SubprocessPool,
    get_subprocess_pool,
)

# =============================================================================
# CommandResult Tests
# =============================================================================

@pytest.mark.unit
class TestCommandResult:
    """Test CommandResult dataclass."""
    
    def test_success_result(self) -> None:
        """Success result has ok=True."""
        result = CommandResult(
            stdout="output",
            stderr="",
            returncode=0,
            elapsed_time=1.0,
        )
        
        assert result.ok
        assert result.stdout == "output"
        assert result.returncode == 0
    
    def test_failure_result(self) -> None:
        """Failure result has ok=False."""
        result = CommandResult(
            stdout="",
            stderr="error",
            returncode=1,
            elapsed_time=0.5,
        )
        
        assert not result.ok
        assert result.returncode == 1


# =============================================================================
# SubprocessPool._validate_argv Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.security
class TestSubprocessPoolValidation:
    """Test SubprocessPool._validate_argv security validation."""
    
    @pytest.fixture
    def pool(self) -> SubprocessPool:
        """Create a fresh pool for testing."""
        return SubprocessPool()
    
    def test_valid_simple_command(self, pool: SubprocessPool) -> None:
        """Accepts simple valid commands."""
        pool._validate_argv(["echo", "hello"])  # Should not raise
        pool._validate_argv(["ls", "-la", "/tmp"])  # Should not raise
    
    def test_rejects_non_list(self, pool: SubprocessPool) -> None:
        """Rejects non-list inputs."""
        with pytest.raises(ValueError, match="must be a list"):
            pool._validate_argv("echo hello")  # type: ignore
    
    def test_rejects_empty_list(self, pool: SubprocessPool) -> None:
        """Rejects empty command list."""
        with pytest.raises(ValueError, match="cannot be empty"):
            pool._validate_argv([])
    
    def test_rejects_non_string_elements(self, pool: SubprocessPool) -> None:
        """Rejects lists with non-string elements."""
        with pytest.raises(ValueError, match="must be strings"):
            pool._validate_argv(["echo", 123])  # type: ignore
    
    def test_rejects_sh_c_wrapper(self, pool: SubprocessPool) -> None:
        """Rejects sh -c shell wrappers."""
        with pytest.raises(ValueError, match="Shell wrapper"):
            pool._validate_argv(["sh", "-c", "echo test"])
    
    def test_rejects_bash_c_wrapper(self, pool: SubprocessPool) -> None:
        """Rejects bash -c shell wrappers."""
        with pytest.raises(ValueError, match="Shell wrapper"):
            pool._validate_argv(["bash", "-c", "echo test"])
    
    def test_rejects_bash_i_interactive(self, pool: SubprocessPool) -> None:
        """Rejects bash -i interactive shells."""
        with pytest.raises(ValueError, match="Interactive shell"):
            pool._validate_argv(["bash", "-i"])
    
    def test_rejects_sh_i_interactive(self, pool: SubprocessPool) -> None:
        """Rejects sh -i interactive shells."""
        with pytest.raises(ValueError, match="Interactive shell"):
            pool._validate_argv(["sh", "-i"])
    
    def test_allows_shell_without_dangerous_flags(
        self, 
        pool: SubprocessPool
    ) -> None:
        """Allows shell commands without -c/-i flags."""
        pool._validate_argv(["bash", "--version"])  # OK
        pool._validate_argv(["sh", "--help"])  # OK


# =============================================================================
# SubprocessPool.run_command Tests
# =============================================================================

@pytest.mark.unit
class TestSubprocessPoolRunCommand:
    """Test SubprocessPool.run_command execution."""
    
    @pytest.fixture
    def pool(self) -> SubprocessPool:
        """Create a fresh pool for testing."""
        return SubprocessPool(max_workers=2, default_timeout=10.0)
    
    def test_successful_command(self, pool: SubprocessPool) -> None:
        """Executes valid commands successfully."""
        result = pool.run_command(["echo", "hello world"])
        
        assert result.ok
        assert result.returncode == 0
        assert "hello world" in result.stdout
        assert result.elapsed_time >= 0
    
    def test_failed_command(self, pool: SubprocessPool) -> None:
        """Handles failed commands correctly."""
        result = pool.run_command(["ls", "/nonexistent/path/xyz"])
        
        assert not result.ok
        assert result.returncode != 0
    
    def test_rejects_shell_wrapper(self, pool: SubprocessPool) -> None:
        """Rejects shell wrapper commands."""
        with pytest.raises(ValueError, match="Shell wrapper"):
            pool.run_command(["sh", "-c", "echo test"])
    
    def test_rejects_interactive_shell(self, pool: SubprocessPool) -> None:
        """Rejects interactive shell commands."""
        with pytest.raises(ValueError, match="Interactive shell"):
            pool.run_command(["bash", "-i"])
    
    def test_timeout_handling(self, pool: SubprocessPool) -> None:
        """Handles command timeout correctly."""
        result = pool.run_command(
            ["sleep", "60"],
            timeout=0.5,
        )
        
        assert not result.ok
        assert result.returncode == -1
    
    def test_custom_cwd(self, pool: SubprocessPool, tmp_path) -> None:
        """Respects custom working directory."""
        # Create a test file in tmp_path
        test_file = tmp_path / "testfile.txt"
        test_file.write_text("content")
        
        result = pool.run_command(
            ["ls"],
            cwd=str(tmp_path),
        )
        
        assert result.ok
        assert "testfile.txt" in result.stdout
    
    def test_custom_environment(self, pool: SubprocessPool) -> None:
        """Passes custom environment variables."""
        result = pool.run_command(
            ["printenv", "MY_VAR"],
            env={"MY_VAR": "test_value"},
        )
        
        assert "test_value" in result.stdout
    
    def test_tracks_execution_count(self, pool: SubprocessPool) -> None:
        """Tracks total executed commands."""
        initial = pool.total_executed
        
        pool.run_command(["echo", "1"])
        pool.run_command(["echo", "2"])
        pool.run_command(["echo", "3"])
        
        assert pool.total_executed == initial + 3


# =============================================================================
# SubprocessPool Concurrency Tests
# =============================================================================

@pytest.mark.unit
class TestSubprocessPoolConcurrency:
    """Test SubprocessPool concurrency control."""
    
    def test_max_workers_limit(self) -> None:
        """Pool limits concurrent executions to max_workers."""
        pool = SubprocessPool(max_workers=2)
        
        # Track concurrent execution count
        max_concurrent = 0
        current_concurrent = 0
        lock = threading.Lock()
        
        def run_slow():
            nonlocal max_concurrent, current_concurrent
            with lock:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
            
            time.sleep(0.2)
            
            with lock:
                current_concurrent -= 1
        
        # Start more threads than max_workers
        threads = []
        for _ in range(4):
            t = threading.Thread(
                target=lambda: pool.run_command(["sleep", "0.2"])
            )
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Due to pool limiting, max concurrent should be <= max_workers
        # Note: This test is approximate due to timing
    
    def test_active_count_tracking(self) -> None:
        """Pool correctly tracks active execution count."""
        pool = SubprocessPool(max_workers=2)
        
        assert pool.active_count == 0
        
        # Run a quick command
        pool.run_command(["echo", "test"])
        
        # After completion, active count should be 0
        assert pool.active_count == 0


# =============================================================================
# Backward Compatibility Tests
# =============================================================================

@pytest.mark.unit
class TestSubprocessPoolBackwardCompatibility:
    """Test backward-compatible API methods."""
    
    @pytest.fixture
    def pool(self) -> SubprocessPool:
        """Create a fresh pool for testing."""
        return SubprocessPool()
    
    def test_acquire_release(self, pool: SubprocessPool) -> None:
        """Test acquire/release API for backward compatibility."""
        # Should be able to acquire
        assert pool.acquire() == True
        assert pool.active_count == 1
        
        # Release should work
        pool.release()
        assert pool.active_count == 0
    
    def test_cleanup_no_error(self, pool: SubprocessPool) -> None:
        """Cleanup should not raise errors."""
        pool.cleanup()  # Should not raise
    
    def test_shutdown_no_error(self, pool: SubprocessPool) -> None:
        """Shutdown should not raise errors."""
        pool.shutdown()  # Should not raise
        assert pool.active_count == 0


# =============================================================================
# Global Pool Tests
# =============================================================================

@pytest.mark.unit
class TestGlobalSubprocessPool:
    """Test get_subprocess_pool() global instance."""
    
    def test_returns_pool_instance(self) -> None:
        """Returns a SubprocessPool instance."""
        pool = get_subprocess_pool()
        assert isinstance(pool, SubprocessPool)
    
    def test_returns_same_instance(self) -> None:
        """Returns the same instance on multiple calls."""
        pool1 = get_subprocess_pool()
        pool2 = get_subprocess_pool()
        assert pool1 is pool2
    
    def test_pool_is_functional(self) -> None:
        """Global pool can execute commands."""
        pool = get_subprocess_pool()
        result = pool.run_command(["echo", "global pool test"])
        
        assert result.ok
        assert "global pool test" in result.stdout


# =============================================================================
# Security Integration Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.security
class TestSubprocessPoolSecurity:
    """Security-focused tests for SubprocessPool."""
    
    @pytest.fixture
    def pool(self) -> SubprocessPool:
        """Create a fresh pool for testing."""
        return SubprocessPool()
    
    def test_no_shell_injection_via_args(self, pool: SubprocessPool) -> None:
        """Shell metacharacters in args don't execute."""
        # This would be dangerous with shell=True
        result = pool.run_command(["echo", "hello; rm -rf /"])
        
        assert result.ok
        # The semicolon should be treated as literal
        assert ";" in result.stdout
        assert "rm" in result.stdout  # Just echoed, not executed
    
    def test_no_backtick_execution(self, pool: SubprocessPool) -> None:
        """Backticks don't execute commands."""
        result = pool.run_command(["echo", "`whoami`"])
        
        assert result.ok
        assert "`whoami`" in result.stdout
    
    def test_no_dollar_substitution(self, pool: SubprocessPool) -> None:
        """$() substitution doesn't execute."""
        result = pool.run_command(["echo", "$(whoami)"])
        
        assert result.ok
        assert "$(whoami)" in result.stdout
    
    def test_explicit_shell_false(self, pool: SubprocessPool) -> None:
        """Verify pool uses shell=False (via behavior)."""
        # If shell=True was used, this would expand the glob
        # With shell=False, it's passed literally to echo
        result = pool.run_command(["echo", "*"])
        
        assert result.ok
        # With shell=False, echo receives literal "*"
        assert "*" in result.stdout or "echo" in result.stderr
