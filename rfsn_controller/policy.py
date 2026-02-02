"""Heuristic policy for selecting repair intents and subgoals."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .verifier import VerifyResult


@dataclass
class PolicyDecision:
    """Decision container with intent, subgoal, and focus test command."""

    intent: str
    subgoal: str
    focus_test_cmd: str
    confidence: float = 1.0  # Confidence score for this decision


# Error pattern regexes for better classification
ERROR_PATTERNS = {
    "import": [
        r"ModuleNotFoundError",
        r"ImportError",
        r"No module named",
        r"cannot import name",
    ],
    "type": [
        r"TypeError",
        r"unsupported operand type",
        r"object of type",
        r"expected.*got",
    ],
    "attribute": [
        r"AttributeError",
        r"has no attribute",
        r"object has no attribute",
    ],
    "key": [
        r"KeyError",
        r"key not found",
    ],
    "index": [
        r"IndexError",
        r"list index out of range",
        r"string index out of range",
    ],
    "value": [
        r"ValueError",
        r"invalid literal",
        r"could not convert",
    ],
    "name": [
        r"NameError",
        r"name.*is not defined",
    ],
    "syntax": [
        r"SyntaxError",
        r"invalid syntax",
    ],
    "assertion": [
        r"AssertionError",
        r"assert",
    ],
    "zero_division": [
        r"ZeroDivisionError",
        r"division by zero",
    ],
}


def _classify_error(blob: str) -> list[str]:
    """Classify the error type(s) from the error output.

    Args:
        blob: Combined stdout/stderr from test failure.

    Returns:
        List of error type categories found.
    """
    found = []
    for category, patterns in ERROR_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, blob, re.IGNORECASE):
                found.append(category)
                break
    return found


def _extract_error_context(blob: str) -> dict:
    """Extract contextual information from error output.

    Args:
        blob: Combined stdout/stderr from test failure.

    Returns:
        Dictionary with context clues.
    """
    context = {
        "has_traceback": "Traceback" in blob,
        "has_assert": "AssertionError" in blob or "assert " in blob,
        "line_numbers": re.findall(r"line \d+", blob),
        "file_paths": re.findall(r'File "([^"]+)"', blob),
        "error_messages": re.findall(r"\b[A-Z][a-zA-Z]*Error:?[^\n]*", blob),
    }
    return context


def _choose_intent_from_categories(categories: list[str], context: dict) -> tuple[str, str, float]:
    """Choose intent and subgoal based on error categories.

    Args:
        categories: List of error type categories.
        context: Contextual information from error output.

    Returns:
        Tuple of (intent, subgoal, confidence).
    """
    if not categories:
        return "general_fix", "reduce_failing_tests", 0.5

    # Priority-based matching
    if "import" in categories:
        return "dependency_or_import_fix", "fix_imports", 0.9
    if "name" in categories:
        return "name_fix", "resolve_undefined_names", 0.85
    if "syntax" in categories:
        return "syntax_fix", "correct_syntax_errors", 0.95
    if "attribute" in categories:
        return "attribute_error_fix", "fix_missing_attr", 0.85
    if "type" in categories:
        return "type_error_fix", "reduce_type_errors", 0.8
    if "key" in categories:
        return "key_error_fix", "handle_missing_keys", 0.8
    if "index" in categories:
        return "index_error_fix", "fix_index_bounds", 0.8
    if "value" in categories:
        return "value_error_fix", "validate_inputs", 0.75
    if "zero_division" in categories:
        return "zero_division_fix", "add_division_checks", 0.9
    if "assertion" in categories:
        return "logic_fix", "reduce_assertions", 0.7

    # Default fallback
    return "general_fix", "reduce_failing_tests", 0.5


def choose_policy(test_cmd: str, v: VerifyResult) -> PolicyDecision:
    """Choose an intent and subgoal based on test output heuristics.

    The policy inspects the failure output for common error types and returns
    a PolicyDecision. It also determines a focus test command to run for
    faster feedback (only the first failing test when available).

    Args:
        test_cmd: The full test command configured.
        v: The latest VerifyResult from running tests.

    Returns:
        A PolicyDecision with chosen intent, subgoal, and focus test command.
    """
    blob = (v.stdout or "") + "\n" + (v.stderr or "")

    # Classify error types
    categories = _classify_error(blob)
    context = _extract_error_context(blob)

    # Choose intent and subgoal based on classification
    intent, subgoal, confidence = _choose_intent_from_categories(categories, context)

    # focus on the first failing file for speed
    if v.failing_tests:
        # Handle both "path::test" and "path/test.py" formats
        first_test = v.failing_tests[0]
        if "::" in first_test:
            test_file = first_test.split("::", 1)[0]
        else:
            test_file = first_test
        focus = f"pytest -q {test_file}"
    else:
        focus = test_cmd

    return PolicyDecision(intent, subgoal, focus, confidence)
