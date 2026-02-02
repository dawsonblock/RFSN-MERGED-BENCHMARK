"""Tests for shell idiom validation in model_validator.

These tests ensure that the validator correctly rejects commands
containing shell idioms that are incompatible with shell=False execution.
"""

import pytest

from rfsn_controller.model_validator import ModelOutputValidator


class TestShellIdiomValidation:
    """Test shell idiom detection and rejection."""

    @pytest.fixture
    def validator(self):
        """Create a ModelOutputValidator instance."""
        return ModelOutputValidator()

    def test_reject_command_chaining_with_ampersand(self, validator):
        """Test that && command chaining is rejected."""
        output = """{
            "mode": "tool_request",
            "requests": [
                {"tool": "sandbox.run", "args": {"cmd": "npm install && npm test"}}
            ],
            "why": "Install and test"
        }"""

        result = validator.validate(output)

        assert not result.is_valid
        assert "Shell idiom" in result.validation_error
        assert (
            "&&" in result.validation_error or "command chaining" in result.validation_error.lower()
        )
        assert "shell=False" in result.why

    def test_reject_command_chaining_with_or(self, validator):
        """Test that || command chaining is rejected."""
        output = """{
            "mode": "tool_request",
            "requests": [
                {"tool": "sandbox.run", "args": {"cmd": "cargo build || echo failed"}}
            ],
            "why": "Build with fallback"
        }"""

        result = validator.validate(output)

        assert not result.is_valid
        assert "Shell idiom" in result.validation_error

    def test_reject_pipe_operator(self, validator):
        """Test that pipe operator is rejected."""
        output = """{
            "mode": "tool_request",
            "requests": [
                {"tool": "sandbox.run", "args": {"cmd": "pytest | tee output.txt"}}
            ],
            "why": "Run tests and save output"
        }"""

        result = validator.validate(output)

        assert not result.is_valid
        assert "Shell idiom" in result.validation_error
        assert "pipe" in result.validation_error.lower()

    def test_reject_output_redirect(self, validator):
        """Test that output redirect is rejected."""
        output = """{
            "mode": "tool_request",
            "requests": [
                {"tool": "sandbox.run", "args": {"cmd": "npm test > output.txt"}}
            ],
            "why": "Save test output"
        }"""

        result = validator.validate(output)

        assert not result.is_valid
        assert "Shell idiom" in result.validation_error
        assert "redirect" in result.validation_error.lower()

    def test_reject_input_redirect(self, validator):
        """Test that input redirect is rejected."""
        output = """{
            "mode": "tool_request",
            "requests": [
                {"tool": "sandbox.run", "args": {"cmd": "python script.py < input.txt"}}
            ],
            "why": "Run with input"
        }"""

        result = validator.validate(output)

        assert not result.is_valid
        assert "Shell idiom" in result.validation_error

    def test_reject_command_substitution_dollar_paren(self, validator):
        """Test that $() command substitution is rejected."""
        output = """{
            "mode": "tool_request",
            "requests": [
                {"tool": "sandbox.run", "args": {"cmd": "echo $(date)"}}
            ],
            "why": "Print date"
        }"""

        result = validator.validate(output)

        assert not result.is_valid
        assert "Shell idiom" in result.validation_error
        assert "substitution" in result.validation_error.lower()

    def test_reject_backtick_command_substitution(self, validator):
        """Test that backtick command substitution is rejected."""
        output = """{
            "mode": "tool_request",
            "requests": [
                {"tool": "sandbox.run", "args": {"cmd": "echo `date`"}}
            ],
            "why": "Print date"
        }"""

        result = validator.validate(output)

        assert not result.is_valid
        assert "Shell idiom" in result.validation_error

    def test_reject_cd_command(self, validator):
        """Test that cd command is rejected."""
        output = """{
            "mode": "tool_request",
            "requests": [
                {"tool": "sandbox.run", "args": {"cmd": "cd src && pytest"}}
            ],
            "why": "Change to src and test"
        }"""

        result = validator.validate(output)

        assert not result.is_valid
        assert "Shell idiom" in result.validation_error

    def test_reject_cd_command_standalone(self, validator):
        """Test that standalone cd command is rejected."""
        output = """{
            "mode": "tool_request",
            "requests": [
                {"tool": "sandbox.run", "args": {"cmd": "cd tests"}}
            ],
            "why": "Change directory"
        }"""

        result = validator.validate(output)

        assert not result.is_valid
        assert "Shell idiom" in result.validation_error
        assert "cd" in result.validation_error.lower()

    def test_reject_inline_env_var(self, validator):
        """Test that inline environment variable assignment is rejected."""
        output = """{
            "mode": "tool_request",
            "requests": [
                {"tool": "sandbox.run", "args": {"cmd": "FOO=bar pytest"}}
            ],
            "why": "Set env var and test"
        }"""

        result = validator.validate(output)

        assert not result.is_valid
        assert "Shell idiom" in result.validation_error
        assert "environment variable" in result.validation_error.lower()

    def test_accept_simple_command(self, validator):
        """Test that simple commands are accepted."""
        output = """{
            "mode": "tool_request",
            "requests": [
                {"tool": "sandbox.run", "args": {"cmd": "pytest tests/test_example.py"}}
            ],
            "why": "Run specific test"
        }"""

        result = validator.validate(output)

        assert result.is_valid
        assert result.mode == "tool_request"

    def test_accept_command_with_flags(self, validator):
        """Test that commands with flags are accepted."""
        output = """{
            "mode": "tool_request",
            "requests": [
                {"tool": "sandbox.run", "args": {"cmd": "npm install --save-dev jest"}}
            ],
            "why": "Install dev dependency"
        }"""

        result = validator.validate(output)

        assert result.is_valid
        assert result.mode == "tool_request"

    def test_accept_python_module_invocation(self, validator):
        """Test that python -m commands are accepted."""
        output = """{
            "mode": "tool_request",
            "requests": [
                {"tool": "sandbox.run", "args": {"cmd": "python -m pytest tests/"}}
            ],
            "why": "Run tests via module"
        }"""

        result = validator.validate(output)

        assert result.is_valid
        assert result.mode == "tool_request"

    def test_corrective_feedback_includes_guidance(self, validator):
        """Test that corrective feedback provides actionable guidance."""
        output = """{
            "mode": "tool_request",
            "requests": [
                {"tool": "sandbox.run", "args": {"cmd": "npm install && npm test"}}
            ],
            "why": "Install and test"
        }"""

        result = validator.validate(output)

        assert not result.is_valid
        # Check that corrective feedback includes helpful guidance
        assert "shell=False" in result.why
        assert "separate" in result.why.lower() or "split" in result.why.lower()
        assert "tool_request" in result.why.lower()

    def test_non_run_commands_not_checked(self, validator):
        """Test that non-run tool requests are not checked for shell idioms."""
        # Even if the path contains special characters, it should pass
        # since we only check sandbox.run commands
        output = """{
            "mode": "tool_request",
            "requests": [
                {"tool": "sandbox.read_file", "args": {"path": "file && name.txt"}}
            ],
            "why": "Read file with special chars in name"
        }"""

        result = validator.validate(output)

        # Should be valid since we only check sandbox.run commands
        assert result.is_valid

    def test_multiple_requests_first_invalid(self, validator):
        """Test that validation stops at first invalid request."""
        output = """{
            "mode": "tool_request",
            "requests": [
                {"tool": "sandbox.run", "args": {"cmd": "npm install && npm test"}},
                {"tool": "sandbox.read_file", "args": {"path": "README.md"}}
            ],
            "why": "Multiple operations"
        }"""

        result = validator.validate(output)

        assert not result.is_valid
        assert "Shell idiom" in result.validation_error


class TestShellIdiomDetection:
    """Test the _detect_shell_idioms method directly."""

    @pytest.fixture
    def validator(self):
        """Create a ModelOutputValidator instance."""
        return ModelOutputValidator()

    def test_detect_double_ampersand(self, validator):
        """Test detection of && operator."""
        has_idiom, desc = validator._detect_shell_idioms("npm install && npm test")
        assert has_idiom
        assert desc is not None

    def test_detect_pipe(self, validator):
        """Test detection of pipe operator."""
        has_idiom, desc = validator._detect_shell_idioms("cat file.txt | grep pattern")
        assert has_idiom
        assert desc is not None

    def test_detect_redirect_out(self, validator):
        """Test detection of output redirect."""
        has_idiom, desc = validator._detect_shell_idioms("echo hello > file.txt")
        assert has_idiom
        assert desc is not None

    def test_detect_redirect_in(self, validator):
        """Test detection of input redirect."""
        has_idiom, desc = validator._detect_shell_idioms("python script.py < input.txt")
        assert has_idiom
        assert desc is not None

    def test_detect_command_substitution(self, validator):
        """Test detection of $() command substitution."""
        has_idiom, desc = validator._detect_shell_idioms("echo $(pwd)")
        assert has_idiom
        assert desc is not None

    def test_detect_backtick_substitution(self, validator):
        """Test detection of backtick command substitution."""
        has_idiom, desc = validator._detect_shell_idioms("echo `pwd`")
        assert has_idiom
        assert desc is not None

    def test_detect_cd_command(self, validator):
        """Test detection of cd command."""
        has_idiom, desc = validator._detect_shell_idioms("cd /tmp")
        assert has_idiom
        assert desc is not None

    def test_detect_inline_env_var(self, validator):
        """Test detection of inline environment variable."""
        has_idiom, desc = validator._detect_shell_idioms("DEBUG=1 pytest")
        assert has_idiom
        assert desc is not None

    def test_no_idiom_in_simple_command(self, validator):
        """Test that simple commands have no idioms."""
        has_idiom, desc = validator._detect_shell_idioms("pytest tests/")
        assert not has_idiom
        assert desc is None

    def test_no_idiom_with_flags(self, validator):
        """Test that commands with flags have no idioms."""
        has_idiom, desc = validator._detect_shell_idioms("npm install --save-dev jest")
        assert not has_idiom
        assert desc is None

    def test_double_greater_than_detected_as_redirect(self, validator):
        """Test that >> (append redirect) is detected as a shell redirect idiom."""
        has_idiom, desc = validator._detect_shell_idioms("echo hello >> output.txt")
        assert has_idiom
        assert desc is not None
        assert "redirect" in desc.lower()

        # Note: The model_validator uses simple regex patterns for performance
        # and may have false positives with quoted strings. The command_normalizer
        # uses shlex for more accurate detection. This is acceptable since the
        # model_validator provides a first-line defense with corrective feedback.
        # The command itself would be blocked at execution time if it actually
        # contains shell idioms.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
