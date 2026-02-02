"""Tests for allowlist profiles, hygiene policies, and shell idiom detection."""

from rfsn_controller.allowlist_profiles import (
    commands_for_language,
    commands_for_project,
)
from rfsn_controller.command_normalizer import detect_shell_idioms, get_shell_idiom_error_message
from rfsn_controller.patch_hygiene import PatchHygieneConfig


class TestAllowlistProfiles:
    """Tests for language-scoped command allowlists."""

    def test_python_profile_includes_python_commands(self):
        """Python profile should include Python-specific commands."""
        cmds = commands_for_language("python")
        assert "python" in cmds
        assert "pytest" in cmds
        assert "pip" in cmds
        assert "git" in cmds  # Base commands are included

    def test_python_profile_excludes_rust_commands(self):
        """Python profile should not include Rust commands."""
        cmds = commands_for_language("python")
        assert "cargo" not in cmds
        assert "rustc" not in cmds

    def test_rust_profile_includes_cargo(self):
        """Rust profile should include cargo command."""
        cmds = commands_for_language("rust")
        assert "cargo" in cmds
        assert "rustc" in cmds
        assert "git" in cmds  # Base commands are included

    def test_rust_profile_excludes_python_commands(self):
        """Rust profile should not include Python-specific commands."""
        cmds = commands_for_language("rust")
        assert "pytest" not in cmds
        assert "pip" not in cmds

    def test_node_profile_includes_npm(self):
        """Node profile should include npm and related commands."""
        cmds = commands_for_language("node")
        assert "npm" in cmds
        assert "yarn" in cmds
        assert "node" in cmds

    def test_cd_never_in_any_profile(self):
        """cd command should never be in any profile."""
        languages = ["python", "node", "rust", "go", "java", "dotnet"]
        for lang in languages:
            cmds = commands_for_language(lang)
            assert "cd" not in cmds, f"cd found in {lang} profile"

    def test_base_commands_in_all_profiles(self):
        """Base commands should be in all profiles."""
        languages = ["python", "node", "rust", "go", "java", "dotnet"]
        for lang in languages:
            cmds = commands_for_language(lang)
            assert "git" in cmds
            assert "cat" in cmds
            assert "grep" in cmds

    def test_commands_for_project_dict_format(self):
        """commands_for_project should handle dict format."""
        project_info = {"language": "python"}
        cmds = commands_for_project(project_info)
        assert "python" in cmds
        assert "pytest" in cmds

    def test_commands_for_project_object_format(self):
        """commands_for_project should handle object format."""

        class ProjectInfo:
            def __init__(self):
                self.language = "rust"

        project_info = ProjectInfo()
        cmds = commands_for_project(project_info)
        assert "cargo" in cmds

    def test_commands_for_project_defaults_to_python(self):
        """commands_for_project should default to Python for unknown."""
        project_info = {"language": "unknown"}
        cmds = commands_for_project(project_info)
        assert "python" in cmds
        assert "pytest" in cmds


class TestShellIdiomDetection:
    """Tests for shell idiom detection."""

    def test_detects_command_chaining_and(self):
        """Should detect && command chaining."""
        assert detect_shell_idioms("npm install && npm test") is True

    def test_detects_command_chaining_or(self):
        """Should detect || command chaining."""
        assert detect_shell_idioms("npm test || exit 1") is True

    def test_detects_semicolon(self):
        """Should detect semicolon separator."""
        assert detect_shell_idioms("cd foo; pytest") is True

    def test_detects_pipe(self):
        """Should detect pipe operator."""
        assert detect_shell_idioms("cat file.txt | grep test") is True

    def test_detects_redirect(self):
        """Should detect redirect operators."""
        assert detect_shell_idioms("pytest > output.txt") is True
        assert detect_shell_idioms("cat < input.txt") is True

    def test_detects_command_substitution(self):
        """Should detect command substitution."""
        assert detect_shell_idioms("echo $(pwd)") is True
        assert detect_shell_idioms("echo `pwd`") is True

    def test_detects_newlines(self):
        """Should detect multi-line commands."""
        assert detect_shell_idioms("npm install\nnpm test") is True

    def test_detects_cd_command(self):
        """Should detect cd command."""
        assert detect_shell_idioms("cd foo && pytest") is True
        assert detect_shell_idioms("cd tests") is True

    def test_detects_inline_env_vars(self):
        """Should detect inline environment variables."""
        assert detect_shell_idioms("FOO=bar pytest") is True
        assert detect_shell_idioms("NODE_ENV=test npm test") is True

    def test_accepts_simple_commands(self):
        """Should accept simple commands without shell idioms."""
        assert detect_shell_idioms("pytest") is False
        assert detect_shell_idioms("python -m pytest") is False
        assert detect_shell_idioms("npm test") is False
        assert detect_shell_idioms("cargo test") is False

    def test_accepts_commands_with_arguments(self):
        """Should accept commands with arguments."""
        assert detect_shell_idioms("pytest -v tests/") is False
        assert detect_shell_idioms("npm test -- --coverage") is False

    def test_error_message_generation(self):
        """Should generate helpful error messages."""
        msg = get_shell_idiom_error_message("npm install && npm test")
        assert "command chaining" in msg
        assert "shell=False" in msg


class TestHygieneProfiles:
    """Tests for hygiene policy profiles."""

    def test_repair_mode_strict_caps(self):
        """Repair mode should have strict caps."""
        policy = PatchHygieneConfig.for_repair_mode()
        assert policy.max_lines_changed == 200
        assert policy.max_files_changed == 5
        assert policy.allow_test_modification is False
        assert policy.allow_test_deletion is False
        assert policy.allow_lockfile_changes is False

    def test_feature_mode_flexible_caps(self):
        """Feature mode should have more flexible caps."""
        policy = PatchHygieneConfig.for_feature_mode()
        assert policy.max_lines_changed == 500
        assert policy.max_files_changed == 15
        assert policy.allow_test_modification is True
        assert policy.allow_test_deletion is False
        assert policy.allow_lockfile_changes is False

    def test_feature_mode_java_adjustment(self):
        """Feature mode for Java should have higher line cap."""
        policy = PatchHygieneConfig.for_feature_mode(language="java")
        assert policy.max_lines_changed == 700  # 500 + 200
        assert policy.language == "java"

    def test_feature_mode_node_adjustment(self):
        """Feature mode for Node should have higher line cap."""
        policy = PatchHygieneConfig.for_feature_mode(language="node")
        assert policy.max_lines_changed == 600  # 500 + 100
        assert policy.language == "node"

    def test_custom_policy_with_lockfile_changes(self):
        """Custom policy should allow lockfile changes if specified."""
        policy = PatchHygieneConfig.custom(
            max_lines_changed=1000,
            max_files_changed=20,
            allow_lockfile_changes=True,
        )
        assert policy.max_lines_changed == 1000
        assert policy.max_files_changed == 20
        assert policy.allow_lockfile_changes is True

    def test_forbidden_dirs_always_strict(self):
        """Forbidden directories should be strict in all modes."""
        repair = PatchHygieneConfig.for_repair_mode()
        feature = PatchHygieneConfig.for_feature_mode()

        # Both should forbid .git/
        assert ".git/" in repair.forbidden_dirs
        assert ".git/" in feature.forbidden_dirs

        # Both should forbid node_modules/
        assert "node_modules/" in repair.forbidden_dirs
        assert "node_modules/" in feature.forbidden_dirs


class TestIntegration:
    """Integration tests for policy enforcement."""

    def test_allowlist_enforced_in_sandbox_run(self):
        """Test that sandbox _run enforces allowed_commands."""
        from rfsn_controller.allowlist_profiles import commands_for_language
        from rfsn_controller.sandbox import _run

        # Get Python allowlist (should not include cargo)
        python_cmds = commands_for_language("python")

        # Try to run a blocked command with Python allowlist
        exit_code, stdout, stderr = _run(
            "cargo test", cwd="/tmp", timeout_sec=1, allowed_commands=python_cmds
        )

        # Should be blocked
        assert exit_code != 0
        assert "not allowed" in stderr.lower()

        # Try to run an allowed command
        exit_code, stdout, stderr = _run(
            "python --version", cwd="/tmp", timeout_sec=5, allowed_commands=python_cmds
        )

        # Should succeed (or fail for legitimate reasons, not blocking)
        # If python is not installed, it will fail with "not in allowlist" first
        # so we check it's not blocked by allowlist
        if exit_code != 0:
            # If it failed, it must not be due to language allowlist blocking.
            assert "is not allowed for this project type" not in stderr.lower()

    def test_hygiene_policy_selection_by_mode(self):
        """Test that hygiene policy is correctly selected based on mode."""
        # This tests the logic without running full controller
        repair_policy = PatchHygieneConfig.for_repair_mode(language="python")
        feature_policy = PatchHygieneConfig.for_feature_mode(language="python")

        # Verify repair is stricter
        assert repair_policy.max_lines_changed < feature_policy.max_lines_changed
        assert repair_policy.max_files_changed < feature_policy.max_files_changed
        assert not repair_policy.allow_test_modification
        assert feature_policy.allow_test_modification

        # Test language adjustment for Java in feature mode
        java_feature = PatchHygieneConfig.for_feature_mode(language="java")
        assert java_feature.max_lines_changed == 700  # 500 + 200

    def test_lockfile_detection_includes_custom_lock_files(self):
        """Test that .lock files are treated as lockfiles even if not in explicit list."""
        from rfsn_controller.patch_hygiene import validate_patch_hygiene

        # Create a patch that modifies a custom .lock file
        diff = """--- a/custom-deps.lock
+++ b/custom-deps.lock
@@ -1,1 +1,1 @@
-old-version
+new-version
"""

        # With allow_lockfile_changes=False, should be rejected
        config = PatchHygieneConfig(
            max_lines_changed=1000,
            max_files_changed=10,
            allow_lockfile_changes=False,
        )
        result = validate_patch_hygiene(diff, config)
        assert not result.is_valid
        assert any("Cannot modify file" in v or "*.lock" in v for v in result.violations)

        # With allow_lockfile_changes=True, should be allowed
        config_allow = PatchHygieneConfig(
            max_lines_changed=1000,
            max_files_changed=10,
            allow_lockfile_changes=True,
        )
        result_allow = validate_patch_hygiene(diff, config_allow)
        assert result_allow.is_valid
