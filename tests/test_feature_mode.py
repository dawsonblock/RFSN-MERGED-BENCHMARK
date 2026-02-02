"""Tests for feature mode functionality.

Tests model validation, goal creation, prompt building, and feature workflow for the
feature engineering mode that enables the agent to implement new features from scratch.
"""

import pytest

from rfsn_controller.goals import DEFAULT_FEATURE_SUBGOALS, GoalFactory, GoalType
from rfsn_controller.model_validator import ModelOutputValidator
from rfsn_controller.prompt import MODE_FEATURE, build_model_input


class TestModelValidator:
    """Test model validator with feature_summary mode."""

    def test_feature_summary_valid(self):
        """Test valid feature_summary mode."""
        validator = ModelOutputValidator()
        output = '{"mode": "feature_summary", "summary": "Implemented user authentication with JWT tokens and session management", "completion_status": "complete"}'
        result = validator.validate(output)

        assert result.is_valid
        assert result.mode == "feature_summary"
        assert "authentication" in result.summary
        assert result.completion_status == "complete"

    def test_feature_summary_empty_summary(self):
        """Test feature_summary with empty summary."""
        validator = ModelOutputValidator()
        output = '{"mode": "feature_summary", "summary": "", "completion_status": "complete"}'
        result = validator.validate(output)

        assert not result.is_valid
        assert result.mode == "tool_request"
        assert "summary cannot be empty" in result.validation_error

    def test_feature_summary_too_short(self):
        """Test feature_summary with too short summary."""
        validator = ModelOutputValidator()
        output = '{"mode": "feature_summary", "summary": "Done", "completion_status": "complete"}'
        result = validator.validate(output)

        assert not result.is_valid
        assert result.mode == "tool_request"
        assert "at least 20 characters" in result.validation_error

    def test_feature_summary_invalid_status(self):
        """Test feature_summary with invalid completion_status."""
        validator = ModelOutputValidator()
        output = '{"mode": "feature_summary", "summary": "This is a valid length summary for testing purposes", "completion_status": "invalid"}'
        result = validator.validate(output)

        assert not result.is_valid
        assert result.mode == "tool_request"
        assert "Invalid completion_status" in result.validation_error

    def test_feature_summary_all_statuses(self):
        """Test all valid completion statuses."""
        validator = ModelOutputValidator()
        statuses = ["complete", "partial", "blocked", "in_progress"]

        for status in statuses:
            output = f'{{"mode": "feature_summary", "summary": "This is a detailed test summary that meets length requirements", "completion_status": "{status}"}}'
            result = validator.validate(output)
            assert result.is_valid, f"Status {status} should be valid"
            assert result.completion_status == status

    def test_shell_idiom_detection(self):
        """Test detection of shell idioms."""
        validator = ModelOutputValidator()

        # Test various shell idioms that should be detected
        test_cases = [
            ("npm install && npm test", "&&"),
            ("cat file.txt | grep pattern", "|"),
            ("echo hello > output.txt", ">"),
            ("cat < input.txt", "<"),
            ("result=$(command)", "$()"),
            ("result=`command`", "backtick"),
            ("cd /tmp", "cd"),
            ("cd src", "cd"),
            ("ENV_VAR=value python script.py", "environment variable"),
        ]

        for text, keyword in test_cases:
            has_idiom, description = validator._detect_shell_idioms(text)
            assert has_idiom, f"Should detect shell idiom in: {text} (expected keyword: {keyword})"
            assert description is not None, f"Description should not be None for: {text}"

    def test_shell_idiom_no_false_positives(self):
        """Test that normal commands don't trigger false positives."""
        validator = ModelOutputValidator()

        # These should NOT be detected as shell idioms
        safe_cases = [
            "npm install",
            "python -m pytest tests/",
            "git status",
            "grep -r pattern .",
            "echo 'Hello World'",
            "python script.py --flag=value",
        ]

        for text in safe_cases:
            has_idiom, _ = validator._detect_shell_idioms(text)
            assert not has_idiom, f"Should NOT detect shell idiom in: {text}"


class TestFeatureGoals:
    """Test feature goal creation."""

    def test_create_feature_goal(self):
        """Test creating a feature goal."""
        goal = GoalFactory.create_feature_goal(
            description="Add user authentication",
            acceptance_criteria=["Users can log in", "Users can log out"],
            timeout=600,
        )

        # Check type by verifying it has feature-specific attributes
        assert hasattr(goal, "acceptance_criteria")
        assert hasattr(goal, "subgoals")
        assert goal.description == "Add user authentication"
        assert len(goal.acceptance_criteria) == 2
        assert goal.timeout == 600
        assert len(goal.subgoals) == 4  # Default subgoals

    def test_feature_goal_default_subgoals(self):
        """Test default subgoals are created."""
        goal = GoalFactory.create_feature_goal(
            description="Test feature", acceptance_criteria=["Must work"]
        )

        # Should match the centralized constant
        assert goal.subgoals == DEFAULT_FEATURE_SUBGOALS
        assert "scaffold" in goal.subgoals[0]
        assert "implement" in goal.subgoals[1]
        assert "tests" in goal.subgoals[2]
        assert "docs" in goal.subgoals[3]

    def test_feature_goal_validation_empty_description(self):
        """Test that empty description raises ValueError."""
        with pytest.raises(ValueError, match="description cannot be empty"):
            GoalFactory.create_feature_goal(description="", acceptance_criteria=["Must work"])

    def test_feature_goal_validation_no_criteria(self):
        """Test that no acceptance criteria raises ValueError."""
        with pytest.raises(ValueError, match="At least one acceptance criterion is required"):
            GoalFactory.create_feature_goal(description="Test feature", acceptance_criteria=[])

    def test_feature_goal_validation_empty_criteria(self):
        """Test that all empty criteria raises ValueError."""
        with pytest.raises(ValueError, match="All acceptance criteria are empty"):
            GoalFactory.create_feature_goal(
                description="Test feature", acceptance_criteria=["", "  ", ""]
            )


class TestPromptBuilding:
    """Test prompt building for feature mode."""

    def test_build_feature_mode_prompt(self):
        """Test building prompt in feature mode."""
        state = {
            "mode": MODE_FEATURE,
            "goal": "Implement feature: User authentication",
            "feature_description": "Add login/logout functionality",
            "acceptance_criteria": ["Users can log in", "Sessions persist"],
            "completed_subgoals": ["scaffold: Created auth module"],
            "current_subgoal": "implement: Write login logic",
            "test_cmd": "pytest -q",
            "focus_test_cmd": "pytest -q tests/test_auth.py",
            "failure_output": "",
            "repo_tree": "src/\ntests/",
            "constraints": "Follow security best practices",
            "files_block": "",
            "observations": "",
        }

        prompt = build_model_input(state)

        assert "FEATURE_DESCRIPTION" in prompt
        assert "ACCEPTANCE_CRITERIA" in prompt
        assert "COMPLETED_SUBGOALS" in prompt
        assert "CURRENT_SUBGOAL" in prompt
        assert "User authentication" in prompt
        assert "login/logout functionality" in prompt
        assert "Users can log in" in prompt

    def test_build_repair_mode_prompt(self):
        """Test building prompt in repair mode (original behavior)."""
        state = {
            "goal": "Make tests pass",
            "intent": "fix_bug",
            "subgoal": "Fix import error",
            "test_cmd": "pytest -q",
            "focus_test_cmd": "pytest -q tests/test_main.py",
            "failure_output": "ImportError: No module named 'foo'",
            "repo_tree": "src/\ntests/",
            "constraints": "Minimal changes",
            "files_block": "",
            "observations": "",
        }

        prompt = build_model_input(state)

        assert "INTENT" in prompt
        assert "SUBGOAL" in prompt
        assert "fix_bug" in prompt
        assert "Fix import error" in prompt
        # Feature mode sections should not be present
        assert "FEATURE_DESCRIPTION" not in prompt
        assert "ACCEPTANCE_CRITERIA" not in prompt

    def test_build_prompt_missing_required_keys(self):
        """Test that missing required keys raises KeyError."""
        state = {
            "goal": "Make tests pass",
            # Missing other required keys
        }

        with pytest.raises(KeyError, match="Missing required state keys"):
            build_model_input(state)

    def test_build_repair_mode_missing_intent(self):
        """Test that repair mode without intent raises KeyError."""
        state = {
            "goal": "Make tests pass",
            "test_cmd": "pytest -q",
            "focus_test_cmd": "pytest -q tests/test_main.py",
            "failure_output": "ImportError",
            "repo_tree": "src/",
            "constraints": "Minimal",
            "files_block": "",
            # Missing intent and subgoal
        }

        with pytest.raises(KeyError, match="Repair mode requires"):
            build_model_input(state)


class TestGoalType:
    """Test GoalType enum includes FEATURE."""

    def test_feature_goal_type_exists(self):
        """Test FEATURE goal type exists."""
        assert hasattr(GoalType, "FEATURE")
        assert GoalType.FEATURE.value == "feature"
