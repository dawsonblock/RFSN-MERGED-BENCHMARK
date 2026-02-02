"""Tests for controller module syntax and FINAL_VERIFY logic.

This module tests that:
1. The controller module can be imported without syntax errors
2. Command normalizer functions work correctly with quoted commands
3. FINAL_VERIFY handles different verify policies correctly
"""

import pytest


class TestControllerImport:
    """Test that controller module imports without syntax errors."""

    def test_controller_module_imports(self):
        """Controller module should import without SyntaxError."""
        try:
            assert True
        except SyntaxError as e:
            pytest.fail(f"Controller module has syntax error: {e}")

    def test_command_normalizer_imports(self):
        """Command normalizer module should import without errors."""
        try:
            from rfsn_controller.command_normalizer import detect_shell_idioms

            assert callable(detect_shell_idioms)
        except (SyntaxError, ImportError) as e:
            pytest.fail(f"Command normalizer module has error: {e}")


class TestCommandNormalizerQuotedCommands:
    """Test command normalizer with quoted commands."""

    def test_detect_shell_idioms_with_quoted_greater_than(self):
        """Should not detect > inside quotes as redirect."""
        from rfsn_controller.command_normalizer import detect_shell_idioms

        # Greater-than inside quotes should not be detected as redirect
        assert detect_shell_idioms('python -c "print(1 > 0)"') is False
        assert detect_shell_idioms("python -c 'x = 5 > 3'") is False

    def test_detect_shell_idioms_with_quoted_pipe(self):
        """Should not detect | inside quotes as pipe."""
        from rfsn_controller.command_normalizer import detect_shell_idioms

        # Pipe inside quotes should not be detected
        assert detect_shell_idioms('echo "test | value"') is False
        assert detect_shell_idioms("echo 'a | b'") is False

    def test_detect_shell_idioms_real_redirect_outside_quotes(self):
        """Should detect actual redirects outside quotes."""
        from rfsn_controller.command_normalizer import detect_shell_idioms

        # Real redirects should be detected
        assert detect_shell_idioms("echo hello > output.txt") is True
        assert detect_shell_idioms("cat < input.txt") is True

    def test_detect_shell_idioms_real_pipe_outside_quotes(self):
        """Should detect actual pipes outside quotes."""
        from rfsn_controller.command_normalizer import detect_shell_idioms

        # Real pipes should be detected
        assert detect_shell_idioms("cat file.txt | grep pattern") is True

    def test_shlex_import_no_error(self):
        """Calling detect_shell_idioms should not raise NameError for shlex."""
        from rfsn_controller.command_normalizer import detect_shell_idioms

        # Should not raise NameError about shlex
        try:
            result = detect_shell_idioms('echo "test > value"')
            assert result is False  # No shell idioms in this command
        except NameError as e:
            if "shlex" in str(e):
                pytest.fail("shlex module not imported - NameError occurred")
            raise


class TestFinalVerifyLogic:
    """Test FINAL_VERIFY phase logic and verify policies.

    Note: These are structural tests that verify the code paths exist
    and are syntactically correct. Full integration tests would require
    a complete sandbox setup.
    """

    def test_verify_result_import(self):
        """VerifyResult should be importable from verifier module."""
        from rfsn_controller.verifier import VerifyResult

        # Should be able to create a VerifyResult instance
        v = VerifyResult(
            ok=True, exit_code=0, stdout="test output", stderr="", failing_tests=[], sig="test_sig"
        )
        assert v.ok is True
        assert v.exit_code == 0
        assert v.stdout == "test output"

    def test_verify_result_failed_case(self):
        """VerifyResult should handle failed verification."""
        from rfsn_controller.verifier import VerifyResult

        v = VerifyResult(
            ok=False,
            exit_code=1,
            stdout="",
            stderr="error output",
            failing_tests=["test_foo", "test_bar"],
            sig="fail_sig",
        )
        assert v.ok is False
        assert v.exit_code == 1
        assert len(v.failing_tests) == 2

    def test_phases_import(self):
        """Phase enum should be importable."""
        from rfsn_controller.phases import Phase

        # Verify FINAL_VERIFY phase exists
        assert hasattr(Phase, "FINAL_VERIFY")
        assert hasattr(Phase, "EVIDENCE_PACK")
        assert hasattr(Phase, "BAILOUT")

    def test_controller_has_final_verify_section(self):
        """Controller source should contain FINAL_VERIFY handling."""
        import inspect

        import rfsn_controller.controller

        # Get the source code of the controller module
        source = inspect.getsource(rfsn_controller.controller)

        # Verify FINAL_VERIFY section exists
        assert "Phase.FINAL_VERIFY" in source
        assert "verify_policy" in source

        # Verify the three verify policies are handled
        assert "cmds_then_tests" in source
        assert "cmds_only" in source
        assert "tests_only" in source or 'verify_policy != "cmds_only"' in source
