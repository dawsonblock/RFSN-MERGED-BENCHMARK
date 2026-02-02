"""Unit tests for exec_utils module.

This module tests:
- _validate_argv() security validation
- safe_run() command execution
- Environment sanitization
- Docker command building
- Shell wrapper rejection
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rfsn_controller.exec_utils import (
    SAFE_ENV_VARS,
    ExecResult,
    _get_safe_env,
    _validate_argv,
    docker_exec_argv,
    parse_command_string,
    safe_run,
    safe_run_string,
)

# =============================================================================
# _validate_argv Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.security
class TestValidateArgv:
    """Test _validate_argv security validation."""
    
    def test_valid_simple_command(self) -> None:
        """Accepts simple valid commands."""
        _validate_argv(["echo", "hello"])  # Should not raise
        _validate_argv(["ls", "-la", "/tmp"])  # Should not raise
    
    def test_reject_python_exec_string(self) -> None:
        """Reject python -c commands that execute code strings."""
        with pytest.raises(ValueError):
            _validate_argv(["python", "-c", "print('hello')"])
        with pytest.raises(ValueError):
            _validate_argv(["python3", "-c", "import sys"])
    
    def test_rejects_non_list(self) -> None:
        """Rejects non-list inputs."""
        with pytest.raises(ValueError, match="must be a list"):
            _validate_argv("echo hello")  # type: ignore
        
        with pytest.raises(ValueError, match="must be a list"):
            _validate_argv(("echo", "hello"))  # type: ignore
    
    def test_rejects_empty_list(self) -> None:
        """Rejects empty command list."""
        with pytest.raises(ValueError, match="must not be empty"):
            _validate_argv([])
    
    def test_rejects_non_string_elements(self) -> None:
        """Rejects lists with non-string elements."""
        with pytest.raises(ValueError, match="must be a string"):
            _validate_argv(["echo", 123])  # type: ignore
        
        with pytest.raises(ValueError, match="must be a string"):
            _validate_argv(["ls", None])  # type: ignore
    
    def test_rejects_sh_c_wrapper(self) -> None:
        """Rejects sh -c shell wrappers."""
        with pytest.raises(ValueError, match="Shell wrapper detected"):
            _validate_argv(["sh", "-c", "echo test"])
        
        with pytest.raises(ValueError, match="Shell wrapper detected"):
            _validate_argv(["/bin/sh", "-c", "rm -rf /"])
    
    def test_rejects_bash_c_wrapper(self) -> None:
        """Rejects bash -c shell wrappers."""
        with pytest.raises(ValueError, match="Shell wrapper detected"):
            _validate_argv(["bash", "-c", "echo test"])
        
        with pytest.raises(ValueError, match="Shell wrapper detected"):
            _validate_argv(["/bin/bash", "-c", "dangerous command"])
    
    def test_rejects_dash_wrapper(self) -> None:
        """Rejects dash -c shell wrappers."""
        with pytest.raises(ValueError, match="Shell wrapper detected"):
            _validate_argv(["dash", "-c", "echo test"])
    
    def test_rejects_zsh_wrapper(self) -> None:
        """Rejects zsh -c shell wrappers."""
        with pytest.raises(ValueError, match="Shell wrapper detected"):
            _validate_argv(["zsh", "-c", "echo test"])
    
    def test_rejects_ksh_wrapper(self) -> None:
        """Rejects ksh -c shell wrappers."""
        with pytest.raises(ValueError, match="Shell wrapper detected"):
            _validate_argv(["ksh", "-c", "echo test"])
    
    def test_allows_shell_without_c_flag(self) -> None:
        """Allows shell commands without -c flag (e.g., --version)."""
        # These should not raise as they don't have -c
        _validate_argv(["bash", "--version"])
        _validate_argv(["sh", "--help"])


# =============================================================================
# safe_run Tests
# =============================================================================

@pytest.mark.unit
class TestSafeRun:
    """Test safe_run command execution."""
    
    def test_successful_command(self, test_cwd: Path) -> None:
        """Executes valid commands successfully."""
        result = safe_run(
            ["echo", "hello world"],
            cwd=str(test_cwd),
            check_global_allowlist=False,
        )
        
        assert result.ok
        assert result.exit_code == 0
        assert "hello world" in result.stdout
        assert result.command == ["echo", "hello world"]
    
    def test_failed_command(self, test_cwd: Path) -> None:
        """Handles failed commands correctly."""
        result = safe_run(
            ["ls", "/nonexistent/directory/path"],
            cwd=str(test_cwd),
            check_global_allowlist=False,
        )
        
        assert not result.ok
        assert result.exit_code != 0
    
    def test_rejects_string_command(self, test_cwd: Path) -> None:
        """Rejects string commands instead of lists."""
        with pytest.raises(ValueError, match="must be a list"):
            safe_run(
                "echo hello",  # type: ignore
                cwd=str(test_cwd),
                check_global_allowlist=False,
            )
    
    def test_rejects_shell_wrapper(self, test_cwd: Path) -> None:
        """Rejects shell wrapper commands."""
        with pytest.raises(ValueError, match="Shell wrapper"):
            safe_run(
                ["sh", "-c", "echo test"],
                cwd=str(test_cwd),
                check_global_allowlist=False,
            )
    
    def test_timeout_handling(self, test_cwd: Path) -> None:
        """Handles command timeout correctly."""
        result = safe_run(
            ["sleep", "10"],
            cwd=str(test_cwd),
            timeout_sec=1,
            check_global_allowlist=False,
        )
        
        assert not result.ok
        assert result.timed_out
        assert result.exit_code == -1
    
    def test_custom_environment(self, test_cwd: Path) -> None:
        """Passes custom environment variables."""
        result = safe_run(
            ["printenv", "CUSTOM_VAR"],
            cwd=str(test_cwd),
            env={"CUSTOM_VAR": "test_value"},
            check_global_allowlist=False,
        )
        
        assert "test_value" in result.stdout
    
    def test_local_allowlist_blocks(self, test_cwd: Path) -> None:
        """Local allowlist blocks non-allowed commands."""
        result = safe_run(
            ["echo", "test"],
            cwd=str(test_cwd),
            allowed_commands={"ls", "cat"},  # echo not in list
            check_global_allowlist=False,
        )
        
        assert not result.ok
        assert "not in allowed list" in result.stderr
    
    def test_local_allowlist_allows(self, test_cwd: Path) -> None:
        """Local allowlist allows listed commands."""
        result = safe_run(
            ["echo", "test"],
            cwd=str(test_cwd),
            allowed_commands={"echo", "ls"},
            check_global_allowlist=False,
        )
        
        assert result.ok


# =============================================================================
# Environment Sanitization Tests
# =============================================================================

@pytest.mark.unit
class TestEnvironmentSanitization:
    """Test environment variable sanitization."""
    
    def test_get_safe_env_filters_variables(
        self, 
        monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_get_safe_env only includes safe variables."""
        # Set some env vars
        monkeypatch.setenv("PATH", "/usr/bin")
        monkeypatch.setenv("HOME", "/tmp/testhome")
        monkeypatch.setenv("DANGEROUS_VAR", "should_be_filtered")
        monkeypatch.setenv("AWS_SECRET_KEY", "secret")
        
        safe_env = _get_safe_env()
        
        assert "PATH" in safe_env
        assert "HOME" in safe_env
        assert "DANGEROUS_VAR" not in safe_env
        assert "AWS_SECRET_KEY" not in safe_env
    
    def test_safe_env_vars_constant(self) -> None:
        """SAFE_ENV_VARS contains expected variables."""
        assert "PATH" in SAFE_ENV_VARS
        assert "HOME" in SAFE_ENV_VARS
        assert "LANG" in SAFE_ENV_VARS


# =============================================================================
# parse_command_string Tests
# =============================================================================

@pytest.mark.unit
class TestParseCommandString:
    """Test command string parsing."""
    
    def test_simple_command(self) -> None:
        """Parses simple commands correctly."""
        result = parse_command_string("echo hello world")
        assert result == ["echo", "hello", "world"]
    
    def test_quoted_arguments(self) -> None:
        """Handles quoted arguments correctly."""
        result = parse_command_string('echo "hello world"')
        assert result == ["echo", "hello world"]
    
    def test_complex_command(self) -> None:
        """Parses complex commands with various quoting."""
        result = parse_command_string("ls -la '/path/with spaces'")
        assert result == ["ls", "-la", "/path/with spaces"]
    
    def test_invalid_quoting(self) -> None:
        """Raises error for invalid quoting."""
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_command_string("echo 'unterminated")


# =============================================================================
# safe_run_string Tests
# =============================================================================

@pytest.mark.unit
class TestSafeRunString:
    """Test safe_run_string compatibility wrapper."""
    
    def test_executes_string_command(self, test_cwd: Path) -> None:
        """Executes string commands by parsing to argv."""
        result = safe_run_string(
            "echo hello",
            cwd=str(test_cwd),
            check_global_allowlist=False,
        )
        
        assert result.ok
        assert "hello" in result.stdout
    
    def test_rejects_shell_wrapper_string(self, test_cwd: Path) -> None:
        """Rejects shell wrappers even in string form."""
        with pytest.raises(ValueError, match="Shell wrapper"):
            safe_run_string(
                "sh -c 'echo test'",
                cwd=str(test_cwd),
                check_global_allowlist=False,
            )


# =============================================================================
# docker_exec_argv Tests
# =============================================================================

@pytest.mark.unit
class TestDockerExecArgv:
    """Test Docker command building."""
    
    def test_basic_command(self) -> None:
        """Builds basic docker exec command."""
        result = docker_exec_argv("my-container", ["echo", "hello"])
        
        assert result == ["docker", "exec", "my-container", "echo", "hello"]
    
    def test_with_workdir(self) -> None:
        """Adds workdir option."""
        result = docker_exec_argv(
            "container", 
            ["ls"], 
            workdir="/app"
        )
        
        assert "-w" in result
        assert "/app" in result
    
    def test_with_user(self) -> None:
        """Adds user option."""
        result = docker_exec_argv(
            "container", 
            ["id"], 
            user="www-data"
        )
        
        assert "-u" in result
        assert "www-data" in result
    
    def test_with_env(self) -> None:
        """Adds environment variables."""
        result = docker_exec_argv(
            "container",
            ["printenv"],
            env={"FOO": "bar", "BAZ": "qux"},
        )
        
        assert "-e" in result
        assert "FOO=bar" in result
        assert "BAZ=qux" in result
    
    def test_validates_inner_command(self) -> None:
        """Validates inner command against shell wrappers."""
        with pytest.raises(ValueError, match="Shell wrapper"):
            docker_exec_argv(
                "container",
                ["sh", "-c", "echo test"],  # Should be rejected
            )
    
    def test_full_options(self) -> None:
        """Builds command with all options."""
        result = docker_exec_argv(
            "my-container",
            ["python", "script.py"],
            workdir="/app",
            user="ubuntu",
            env={"DEBUG": "1"},
        )
        
        expected_parts = [
            "docker", "exec",
            "-w", "/app",
            "-u", "ubuntu",
            "-e", "DEBUG=1",
            "my-container",
            "python", "script.py",
        ]
        
        # Check all expected parts are present
        for part in expected_parts:
            assert part in result


# =============================================================================
# ExecResult Tests
# =============================================================================

@pytest.mark.unit
class TestExecResult:
    """Test ExecResult dataclass."""
    
    def test_success_result(self) -> None:
        """Success result has correct attributes."""
        result = ExecResult(
            ok=True,
            exit_code=0,
            stdout="output",
            stderr="",
            command=["echo", "test"],
        )
        
        assert result.ok
        assert result.exit_code == 0
        assert not result.timed_out
    
    def test_failure_result(self) -> None:
        """Failure result has correct attributes."""
        result = ExecResult(
            ok=False,
            exit_code=1,
            stdout="",
            stderr="error message",
            command=["false"],
        )
        
        assert not result.ok
        assert result.exit_code == 1
    
    def test_timeout_result(self) -> None:
        """Timeout result has correct attributes."""
        result = ExecResult(
            ok=False,
            exit_code=-1,
            stdout="",
            stderr="",
            command=["sleep", "100"],
            timed_out=True,
        )
        
        assert not result.ok
        assert result.timed_out


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.security
class TestSecurityIntegration:
    """Integration tests for security invariants."""
    
    def test_cannot_execute_shell_injection(self, test_cwd: Path) -> None:
        """Verifies shell injection is not possible."""
        # This would be dangerous if shell=True was used
        malicious = "hello; rm -rf /"
        
        result = safe_run(
            ["echo", malicious],
            cwd=str(test_cwd),
            check_global_allowlist=False,
        )
        
        # The semicolon should be treated as literal, not command separator
        assert result.ok
        assert ";" in result.stdout
        assert "rm" in result.stdout  # Just echoed, not executed
    
    def test_cannot_use_backticks(self, test_cwd: Path) -> None:
        """Verifies backtick command substitution doesn't work."""
        result = safe_run(
            ["echo", "`whoami`"],
            cwd=str(test_cwd),
            check_global_allowlist=False,
        )
        
        # Backticks should be treated as literal
        assert result.ok
        assert "`whoami`" in result.stdout
    
    def test_cannot_use_dollar_substitution(self, test_cwd: Path) -> None:
        """Verifies $(cmd) substitution doesn't work."""
        result = safe_run(
            ["echo", "$(whoami)"],
            cwd=str(test_cwd),
            check_global_allowlist=False,
        )
        
        # Should be treated as literal
        assert result.ok
        assert "$(whoami)" in result.stdout
