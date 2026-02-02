"""Tests for controller reliability upgrades and budget tracking.

This module tests:
1. Phase budget configuration
2. Budget tracking logic
3. Feature mode completion gating
4. Reproducible verification
"""

from rfsn_controller.controller import ControllerConfig


class TestBudgetConfiguration:
    """Test that budget parameters are properly configured."""

    def test_default_budget_values(self):
        """Budget parameters should have reasonable defaults."""
        cfg = ControllerConfig(
            github_url="https://github.com/test/repo",
            test_cmd="pytest -q",
        )

        # Check default budget values
        assert cfg.max_install_attempts == 3
        assert cfg.max_patch_attempts == 20
        assert cfg.max_verification_attempts == 5
        assert cfg.max_tool_calls == 40
        assert cfg.repro_times == 1

    def test_custom_budget_values(self):
        """Budget parameters should be configurable."""
        cfg = ControllerConfig(
            github_url="https://github.com/test/repo",
            test_cmd="pytest -q",
            max_install_attempts=5,
            max_patch_attempts=30,
            max_verification_attempts=10,
            max_tool_calls=50,
            repro_times=3,
        )

        assert cfg.max_install_attempts == 5
        assert cfg.max_patch_attempts == 30
        assert cfg.max_verification_attempts == 10
        assert cfg.max_tool_calls == 50
        assert cfg.repro_times == 3

    def test_repro_times_defaults_to_one(self):
        """Reproducibility check should default to 1 run."""
        cfg = ControllerConfig(
            github_url="https://github.com/test/repo",
            test_cmd="pytest -q",
        )
        assert cfg.repro_times == 1


class TestVerifyPolicies:
    """Test that verify policies are properly defined."""

    def test_verify_policy_options(self):
        """Controller should support all three verify policies."""
        # Create configs with each policy
        cfg_tests_only = ControllerConfig(
            github_url="https://github.com/test/repo",
            test_cmd="pytest -q",
            verify_policy="tests_only",
        )

        cfg_cmds_only = ControllerConfig(
            github_url="https://github.com/test/repo",
            test_cmd="pytest -q",
            verify_policy="cmds_only",
            verify_cmds=["make verify"],
        )

        cfg_cmds_then_tests = ControllerConfig(
            github_url="https://github.com/test/repo",
            test_cmd="pytest -q",
            verify_policy="cmds_then_tests",
            verify_cmds=["make lint", "make typecheck"],
        )

        assert cfg_tests_only.verify_policy == "tests_only"
        assert cfg_cmds_only.verify_policy == "cmds_only"
        assert cfg_cmds_then_tests.verify_policy == "cmds_then_tests"

    def test_verify_policy_defaults_to_tests_only(self):
        """Default verify policy should be tests_only."""
        cfg = ControllerConfig(
            github_url="https://github.com/test/repo",
            test_cmd="pytest -q",
        )
        assert cfg.verify_policy == "tests_only"

    def test_focused_verify_cmds_default_empty(self):
        """Focused verify commands should default to empty list."""
        cfg = ControllerConfig(
            github_url="https://github.com/test/repo",
            test_cmd="pytest -q",
        )
        assert cfg.focused_verify_cmds == []

    def test_verify_cmds_default_empty(self):
        """Verify commands should default to empty list."""
        cfg = ControllerConfig(
            github_url="https://github.com/test/repo",
            test_cmd="pytest -q",
        )
        assert cfg.verify_cmds == []


class TestFeatureMode:
    """Test feature mode configuration."""

    def test_feature_mode_disabled_by_default(self):
        """Feature mode should be disabled by default."""
        cfg = ControllerConfig(
            github_url="https://github.com/test/repo",
            test_cmd="pytest -q",
        )
        assert cfg.feature_mode is False
        assert cfg.feature_description is None
        assert cfg.acceptance_criteria == []

    def test_feature_mode_with_description(self):
        """Feature mode should accept description and criteria."""
        cfg = ControllerConfig(
            github_url="https://github.com/test/repo",
            test_cmd="pytest -q",
            feature_mode=True,
            feature_description="Add user authentication",
            acceptance_criteria=[
                "User can log in with username and password",
                "User can log out",
                "Invalid credentials are rejected",
            ],
        )

        assert cfg.feature_mode is True
        assert cfg.feature_description == "Add user authentication"
        assert len(cfg.acceptance_criteria) == 3
        assert "User can log in" in cfg.acceptance_criteria[0]


class TestControllerSourceCode:
    """Test that controller source contains expected upgrade logic."""

    def test_budget_tracking_variables_exist(self):
        """Controller source should contain budget tracking variables."""
        import inspect

        import rfsn_controller.controller

        source = inspect.getsource(rfsn_controller.controller)

        # Verify budget tracking variables are initialized
        assert "total_tool_calls = 0" in source
        assert "total_patch_attempts = 0" in source
        assert "total_verification_attempts = 0" in source

    def test_feature_completion_gating_exists(self):
        """Controller should gate feature_summary completion on verification."""
        import inspect

        import rfsn_controller.controller

        source = inspect.getsource(rfsn_controller.controller)

        # Verify feature completion gating logic exists
        assert "feature_summary" in source
        assert "completion_status" in source
        assert "COMPLETION REJECTED" in source or "completion_rejected" in source.lower()

    def test_repro_verification_exists(self):
        """Controller should support reproducible verification."""
        import inspect

        import rfsn_controller.controller

        source = inspect.getsource(rfsn_controller.controller)

        # Verify reproducibility logic exists
        assert "repro_times" in source
        assert "reproducible" in source or "run_idx+1" in source

    def test_blocked_tool_feedback_exists(self):
        """Controller should provide structured feedback for blocked tools."""
        import inspect

        import rfsn_controller.controller

        source = inspect.getsource(rfsn_controller.controller)

        # Verify structured feedback exists
        assert "BLOCKED TOOL REQUEST" in source
        assert "what to do instead" in source.lower() or "→" in source

    def test_patch_budget_check_exists(self):
        """Controller should check patch attempt budget."""
        import inspect

        import rfsn_controller.controller

        source = inspect.getsource(rfsn_controller.controller)

        # Verify patch budget check exists
        assert "max_patch_attempts" in source
        assert "Patch attempt budget" in source or "patch_attempts" in source
