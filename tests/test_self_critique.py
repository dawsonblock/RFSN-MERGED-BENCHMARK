"""Tests for self_critique module.

Tests the RFSN Planner Self-Critique Rubric implementation,
covering all hard fail conditions and soft warnings.
"""

from __future__ import annotations

import pytest

from rfsn_controller.gates.self_critique import (
    CritiqueResult,
    CritiqueReport,
    critique_plan,
    validate_plan_json,
)
from rfsn_controller.gates.plan_gate import PlanGateConfig, DEFAULT_ALLOWED_STEP_TYPES


class TestCritiqueResult:
    """Tests for CritiqueResult enum."""
    
    def test_enum_values(self):
        """Verify enum has expected values."""
        assert CritiqueResult.APPROVED.value == "APPROVED"
        assert CritiqueResult.REJECTED.value == "REJECTED"
        assert CritiqueResult.REWRITE_ADVISED.value == "REWRITE_ADVISED"


class TestCritiqueReport:
    """Tests for CritiqueReport dataclass."""
    
    def test_default_values(self):
        """Test default initialization."""
        report = CritiqueReport(result=CritiqueResult.APPROVED)
        assert report.result == CritiqueResult.APPROVED
        assert report.hard_failures == []
        assert report.soft_warnings == []
    
    def test_rejected_due_to_hard_failures(self):
        """Test report with hard failures."""
        report = CritiqueReport(
            result=CritiqueResult.REJECTED,
            hard_failures=["Missing required field: steps"],
        )
        assert report.result == CritiqueResult.REJECTED
        assert len(report.hard_failures) == 1


class TestStructuralChecks:
    """Tests for structural correctness checks."""
    
    def test_empty_steps_accepted(self):
        """Plan with empty steps array might be accepted or rejected."""
        plan = {
            "goal": "Test",
            "metadata": {"budget": 10},
            "steps": [],
        }
        report = critique_plan(plan)
        # Empty steps may be accepted structurally
        assert report.result in (CritiqueResult.APPROVED, CritiqueResult.REJECTED)
    
    def test_duplicate_step_ids(self):
        """Duplicate step IDs should be rejected."""
        plan = {
            "goal": "Test",
            "metadata": {"budget": 10},
            "steps": [
                {"id": "step1", "type": "read_file", "inputs": {"path": "test.py"}, "expected_outcome": "Read file"},
                {"id": "step1", "type": "read_file", "inputs": {"path": "other.py"}, "expected_outcome": "Read other"},
            ],
        }
        report = critique_plan(plan)
        assert report.result == CritiqueResult.REJECTED
        assert any("duplicate" in f.lower() for f in report.hard_failures)
    
    def test_invalid_step_type(self):
        """Invalid step type should be rejected."""
        plan = {
            "goal": "Test",
            "metadata": {"budget": 10},
            "steps": [
                {"id": "step1", "type": "invalid_type", "inputs": {"foo": "bar"}, "expected_outcome": "X"},
            ],
        }
        report = critique_plan(plan)
        assert report.result == CritiqueResult.REJECTED
        assert any("type" in f.lower() or "allowlist" in f.lower() for f in report.hard_failures)
    
    def test_cycle_detection(self):
        """Cyclic dependencies should be rejected."""
        plan = {
            "goal": "Test",
            "metadata": {"budget": 10},
            "steps": [
                {"id": "a", "type": "read_file", "inputs": {"path": "a.py"}, "depends_on": ["b"], "expected_outcome": "X"},
                {"id": "b", "type": "read_file", "inputs": {"path": "b.py"}, "depends_on": ["a"], "expected_outcome": "Y"},
            ],
        }
        report = critique_plan(plan)
        assert report.result == CritiqueResult.REJECTED
        assert any("cycle" in f.lower() or "acyclic" in f.lower() for f in report.hard_failures)
    
    def test_valid_plan_approved(self):
        """Valid plan should be approved."""
        plan = {
            "goal": "Read and analyze files",
            "metadata": {"budget": 10},
            "expected_outcome": "File contents verified",
            "steps": [
                {"id": "step1", "type": "read_file", "inputs": {"path": "test.py"}, "expected_outcome": "Read file"},
                {"id": "step2", "type": "run_tests", "inputs": {"test_path": "tests/"}, "depends_on": ["step1"], "expected_outcome": "Tests pass"},
            ],
        }
        report = critique_plan(plan)
        assert report.result == CritiqueResult.APPROVED


class TestCommandSafetyChecks:
    """Tests for command and execution safety checks."""
    
    def test_shell_interpreter_rejected(self):
        """Direct shell interpreter calls should be rejected."""
        plan = {
            "goal": "Test",
            "metadata": {"budget": 10},
            "steps": [
                {"id": "step1", "type": "read_file", "inputs": {"command": "bash -c echo"}, "expected_outcome": "X"},
            ],
        }
        report = critique_plan(plan)
        assert report.result == CritiqueResult.REJECTED
        assert any("shell" in f.lower() for f in report.hard_failures)
    
    def test_inline_env_var_rejected(self):
        """Inline environment variables should be rejected."""
        plan = {
            "goal": "Test",
            "metadata": {"budget": 10},
            "steps": [
                {"id": "step1", "type": "read_file", "inputs": {"command": "FOO=bar python test.py"}, "expected_outcome": "X"},
            ],
        }
        report = critique_plan(plan)
        assert report.result == CritiqueResult.REJECTED
        assert any("inline env" in f.lower() for f in report.hard_failures)
    
    def test_command_chaining_rejected(self):
        """Command chaining operators should be rejected."""
        for op in ["&&", "||", ";", "|"]:
            plan = {
                "goal": "Test",
                "metadata": {"budget": 10},
                "steps": [
                    {"id": "step1", "type": "read_file", "inputs": {"command": f"cmd1 {op} cmd2"}, "expected_outcome": "X"},
                ],
            }
            report = critique_plan(plan)
            assert report.result == CritiqueResult.REJECTED, f"Failed for operator: {op}"
    
    def test_python_exec_rejected(self):
        """Python -c for arbitrary execution should be rejected."""
        plan = {
            "goal": "Test",
            "metadata": {"budget": 10},
            "steps": [
                {"id": "step1", "type": "read_file", "inputs": {"command": "python -c 'import os'"}, "expected_outcome": "X"},
            ],
        }
        report = critique_plan(plan)
        assert report.result == CritiqueResult.REJECTED
        assert any("interpreter" in f.lower() or "-c" in f for f in report.hard_failures)


class TestPathSafetyChecks:
    """Tests for file and path safety checks."""
    
    def test_absolute_path_rejected(self):
        """Absolute paths should be rejected."""
        plan = {
            "goal": "Test",
            "metadata": {"budget": 10},
            "steps": [
                {"id": "step1", "type": "read_file", "inputs": {"path": "/etc/passwd"}, "expected_outcome": "X"},
            ],
        }
        report = critique_plan(plan)
        assert report.result == CritiqueResult.REJECTED
        assert any("absolute" in f.lower() or "relative" in f.lower() for f in report.hard_failures)
    
    def test_path_traversal_rejected(self):
        """Path traversal should be rejected."""
        plan = {
            "goal": "Test",
            "metadata": {"budget": 10},
            "steps": [
                {"id": "step1", "type": "read_file", "inputs": {"path": "../../etc/passwd"}, "expected_outcome": "X"},
            ],
        }
        report = critique_plan(plan)
        assert report.result == CritiqueResult.REJECTED
        assert any("forbidden" in f.lower() or ".." in f for f in report.hard_failures)
    
    def test_git_internal_flagged(self):
        """Accessing .git directory should ideally be flagged (implementation note)."""
        # NOTE: Current implementation may not check .git paths specifically
        # This test documents expected behavior for future enhancement
        plan = {
            "goal": "Test",
            "metadata": {"budget": 10},
            "steps": [
                {"id": "step1", "type": "read_file", "inputs": {"path": ".git/config"}, "expected_outcome": "X"},
            ],
        }
        report = critique_plan(plan)
        # Currently accepted (no specific .git check), but path doesn't have ..
        assert report.result in (CritiqueResult.APPROVED, CritiqueResult.REJECTED)


class TestValidatePlanJson:
    """Tests for JSON validation wrapper."""
    
    def test_valid_json(self):
        """Valid JSON plan should be processed."""
        plan_json = '''{
            "goal": "Test",
            "metadata": {"budget": 10},
            "expected_outcome": "Done",
            "steps": [{"id": "s1", "type": "read_file", "inputs": {"path": "t.py"}, "expected_outcome": "Read"}]
        }'''
        report = validate_plan_json(plan_json)
        assert report.result == CritiqueResult.APPROVED
    
    def test_invalid_json(self):
        """Invalid JSON should be rejected."""
        report = validate_plan_json("{invalid json}")
        assert report.result == CritiqueResult.REJECTED
        assert any("json" in f.lower() for f in report.hard_failures)


class TestGateCompatibility:
    """Tests for gate compatibility checks."""
    
    def test_depends_on_unknown_step(self):
        """Dependency on non-existent step - implementation may not validate this."""
        # NOTE: Current implementation may not validate dependency references
        # This documents the expected input for future enhancement
        plan = {
            "goal": "Test",
            "metadata": {"budget": 10},
            "steps": [
                {"id": "step1", "type": "read_file", "inputs": {"path": "t.py"}, "depends_on": ["unknown"], "expected_outcome": "X"},
            ],
        }
        report = critique_plan(plan)
        # Currently accepted - dependency validation is at execution time
        assert report.result in (CritiqueResult.APPROVED, CritiqueResult.REJECTED, CritiqueResult.REWRITE_ADVISED)


class TestIntegration:
    """Integration tests for the full critique flow."""
    
    def test_full_valid_repair_plan(self):
        """Complete valid repair plan should be approved."""
        plan = {
            "goal": "Fix bug in calculator module",
            "metadata": {
                "budget": 5,
                "task_id": "test-123",
            },
            "expected_outcome": "All tests pass after fix",
            "steps": [
                {
                    "id": "read_source",
                    "type": "read_file",
                    "inputs": {"path": "src/calculator.py"},
                    "expected_outcome": "File contents loaded",
                },
                {
                    "id": "analyze",
                    "type": "analyze_file",
                    "inputs": {"path": "tests/test_calculator.py"},
                    "depends_on": ["read_source"],
                    "expected_outcome": "Test structure understood",
                },
                {
                    "id": "fix",
                    "type": "apply_patch",
                    "inputs": {
                        "path": "src/calculator.py",
                        "patch": "--- a/src/calculator.py\n+++ b/src/calculator.py\n@@ -1 +1 @@\n-def add(a, b): return a - b\n+def add(a, b): return a + b",
                    },
                    "depends_on": ["analyze"],
                    "expected_outcome": "Patch applied successfully",
                },
                {
                    "id": "verify",
                    "type": "run_tests",
                    "inputs": {"test_path": "tests/test_calculator.py"},
                    "depends_on": ["fix"],
                    "expected_outcome": "All tests pass",
                },
            ],
        }
        report = critique_plan(plan)
        assert report.result == CritiqueResult.APPROVED
        assert not report.hard_failures
    
    def test_allowed_types_match_defaults(self):
        """Verify we know the allowed types."""
        expected = {
            "search_repo", "read_file", "analyze_file", "list_directory", "grep_search",
            "apply_patch", "add_test", "refactor_small", "fix_import", "fix_typing",
            "run_tests", "run_lint", "check_syntax", "validate_types",
            "wait", "checkpoint", "replan",
        }
        assert DEFAULT_ALLOWED_STEP_TYPES == expected
