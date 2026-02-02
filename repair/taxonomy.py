"""Repair taxonomy - bug classification ontology."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any


# Core repair taxonomy - expandable
TAXONOMY = [
    "dependency_pin_or_version",
    "import_path_or_module",
    "api_signature_mismatch",
    "logic_branch_or_condition",
    "off_by_one_or_index",
    "none_handling_or_optional",
    "type_or_attr_error",
    "format_or_encoding",
    "time_or_timezone",
    "test_expectation_update",
    "mock_or_fixture_issue",
    "io_path_or_permissions",
    "performance_timeout",
    "concurrency_race",
    "resource_leak",
    "config_or_env_value",
]


# Repair strategies mapped to taxonomy
REPAIR_STRATEGIES: Dict[str, Dict[str, Any]] = {
    "dependency_pin_or_version": {
        "fix_template": "Update version constraint or pin dependency",
        "files_pattern": ["requirements*.txt", "setup.py", "pyproject.toml", "setup.cfg"],
        "prompt_hint": "Check version compatibility and update dependency constraints.",
    },
    "import_path_or_module": {
        "fix_template": "Fix import statement or module path",
        "files_pattern": ["*.py"],
        "prompt_hint": "Verify import paths and module structure.",
    },
    "api_signature_mismatch": {
        "fix_template": "Align function call with expected signature",
        "files_pattern": ["*.py"],
        "prompt_hint": "Check function signatures and update call sites.",
    },
    "logic_branch_or_condition": {
        "fix_template": "Fix conditional logic or branching",
        "files_pattern": ["*.py"],
        "prompt_hint": "Review conditional statements and edge cases.",
    },
    "off_by_one_or_index": {
        "fix_template": "Fix array/list indexing or loop bounds",
        "files_pattern": ["*.py"],
        "prompt_hint": "Check loop bounds and array indices carefully.",
    },
    "none_handling_or_optional": {
        "fix_template": "Add null/None checks or handle Optional types",
        "files_pattern": ["*.py"],
        "prompt_hint": "Add defensive None checks before access.",
    },
    "type_or_attr_error": {
        "fix_template": "Fix type mismatch or missing attribute",
        "files_pattern": ["*.py"],
        "prompt_hint": "Verify types and attribute existence.",
    },
    "format_or_encoding": {
        "fix_template": "Fix string encoding or format issues",
        "files_pattern": ["*.py"],
        "prompt_hint": "Check encoding declarations and string handling.",
    },
    "test_expectation_update": {
        "fix_template": "Update test assertions to match new behavior",
        "files_pattern": ["test_*.py", "*_test.py", "tests/*.py"],
        "prompt_hint": "Review test expectations against implementation.",
    },
    "mock_or_fixture_issue": {
        "fix_template": "Fix mock setup or fixture configuration",
        "files_pattern": ["test_*.py", "*_test.py", "conftest.py"],
        "prompt_hint": "Verify mock targets and fixture scope.",
    },
}


@dataclass(frozen=True)
class RepairHypothesis:
    """A hypothesis about what type of repair is needed."""
    kind: str
    rationale: str
    likely_files: List[str]
    hints: Dict[str, Any]
    
    @property
    def strategy(self) -> Dict[str, Any]:
        """Get the repair strategy for this hypothesis."""
        return REPAIR_STRATEGIES.get(self.kind, {})
