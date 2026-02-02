"""Tests for security improvements and verification enhancements."""

from rfsn_controller.command_allowlist import ALLOWED_COMMANDS, is_command_allowed
from rfsn_controller.goals import GoalFactory, GoalType
from rfsn_controller.patch_hygiene import PatchHygieneConfig
from rfsn_controller.tool_manager import ToolRequest


class TestCommandAllowlist:
    """Test command allowlist security."""

    def test_cd_is_not_allowed(self):
        """Test that cd command is not in allowlist."""
        assert "cd" not in ALLOWED_COMMANDS

    def test_cd_command_blocked(self):
        """Test that cd command is blocked by is_command_allowed."""
        allowed, reason = is_command_allowed("cd /tmp")
        assert not allowed
        assert "not in allowlist" in reason.lower() or "blocked" in reason.lower()

    def test_safe_commands_still_allowed(self):
        """Test that safe commands are still allowed."""
        safe_commands = ["ls", "cat file.txt", "grep pattern file.txt", "python test.py"]
        for cmd in safe_commands:
            allowed, reason = is_command_allowed(cmd)
            assert allowed, f"Command '{cmd}' should be allowed but got: {reason}"


class TestToolSignatureHashing:
    """Test deterministic tool signature hashing."""

    def test_dict_args_deterministic(self):
        """Test that dict args produce deterministic signatures."""
        tool1 = ToolRequest(tool="test", args={"b": 2, "a": 1})
        tool2 = ToolRequest(tool="test", args={"a": 1, "b": 2})

        # Same dict with different key order should produce same signature
        assert tool1.signature() == tool2.signature()

    def test_list_args_deterministic(self):
        """Test that list args produce deterministic signatures."""
        tool1 = ToolRequest(tool="test", args={"items": [1, 2, 3]})
        tool2 = ToolRequest(tool="test", args={"items": [1, 2, 3]})

        assert tool1.signature() == tool2.signature()

    def test_nested_dict_list_deterministic(self):
        """Test that nested dict/list args produce deterministic signatures."""
        tool1 = ToolRequest(tool="test", args={"config": {"z": [3, 2, 1], "a": "value"}})
        tool2 = ToolRequest(tool="test", args={"config": {"a": "value", "z": [3, 2, 1]}})

        assert tool1.signature() == tool2.signature()

    def test_different_values_different_signatures(self):
        """Test that different values produce different signatures."""
        tool1 = ToolRequest(tool="test", args={"value": 1})
        tool2 = ToolRequest(tool="test", args={"value": 2})

        assert tool1.signature() != tool2.signature()


class TestVerifyGoal:
    """Test verify_cmd/smoke test goal creation."""

    def test_create_verify_goal(self):
        """Test that create_verify_goal works correctly."""
        goal = GoalFactory.create_verify_goal(
            command="./smoke_tests.sh",
            timeout=120,
            required=False,
        )

        assert goal.goal_type == GoalType.CUSTOM
        assert goal.command == "./smoke_tests.sh"
        assert goal.description == "Smoke test succeeds"
        assert goal.timeout == 120
        assert goal.required is False


class TestPatchHygieneConfig:
    """Test configurable hygiene thresholds."""

    def test_repair_mode_config(self):
        """Test repair mode configuration."""
        config = PatchHygieneConfig.for_repair_mode()

        assert config.max_lines_changed == 200
        assert config.max_files_changed == 5
        assert config.allow_test_deletion is False
        assert config.allow_test_modification is False

    def test_feature_mode_config(self):
        """Test feature mode configuration."""
        config = PatchHygieneConfig.for_feature_mode()

        assert config.max_lines_changed == 500
        assert config.max_files_changed == 15
        assert config.allow_test_deletion is False
        assert config.allow_test_modification is True

    def test_feature_mode_java_adjustment(self):
        """Test feature mode with Java language adjustment."""
        config = PatchHygieneConfig.for_feature_mode(language="java")

        # Java gets +200 lines for boilerplate
        assert config.max_lines_changed == 700
        assert config.max_files_changed == 15
        assert config.language == "java"

    def test_feature_mode_node_adjustment(self):
        """Test feature mode with Node.js language adjustment."""
        config = PatchHygieneConfig.for_feature_mode(language="node")

        # Node gets +100 lines for config files
        assert config.max_lines_changed == 600
        assert config.max_files_changed == 15
        assert config.language == "node"

    def test_custom_config(self):
        """Test custom configuration."""
        config = PatchHygieneConfig.custom(
            max_lines_changed=1000,
            max_files_changed=20,
            allow_test_deletion=True,
            allow_test_modification=True,
            language="python",
        )

        assert config.max_lines_changed == 1000
        assert config.max_files_changed == 20
        assert config.allow_test_deletion is True
        assert config.allow_test_modification is True
        assert config.language == "python"

    def test_forbidden_dirs_always_strict(self):
        """Test that forbidden dirs cannot be overridden."""
        config1 = PatchHygieneConfig.for_repair_mode()
        config2 = PatchHygieneConfig.for_feature_mode()
        config3 = PatchHygieneConfig.custom(
            max_lines_changed=10000,
            max_files_changed=100,
        )

        # All configs should have the same strict forbidden dirs
        assert config1.forbidden_dirs == config2.forbidden_dirs
        assert config2.forbidden_dirs == config3.forbidden_dirs
        assert ".git/" in config1.forbidden_dirs
        assert "node_modules/" in config1.forbidden_dirs
        assert ".env" in config1.forbidden_dirs

    def test_forbidden_patterns_always_strict(self):
        """Test that forbidden patterns cannot be overridden."""
        config1 = PatchHygieneConfig.for_repair_mode()
        config2 = PatchHygieneConfig.for_feature_mode()

        # All configs should have the same strict forbidden patterns
        assert config1.forbidden_file_patterns == config2.forbidden_file_patterns
        assert ".env" in config1.forbidden_file_patterns
        assert "*.key" in config1.forbidden_file_patterns
        assert "secrets.yml" in config1.forbidden_file_patterns
