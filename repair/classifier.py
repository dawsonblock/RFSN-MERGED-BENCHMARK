"""Failure classifier - maps test output to repair hypotheses."""
from __future__ import annotations
import re
from typing import List
from .taxonomy import RepairHypothesis, TAXONOMY


# Pattern-based classification rules
_CLASSIFICATION_PATTERNS = [
    # Import/Module errors
    (r"ModuleNotFoundError|ImportError|No module named", "import_path_or_module"),
    
    # API/Signature errors  
    (r"TypeError: .* (takes|positional arguments|got an unexpected keyword|missing \d+ required)", "api_signature_mismatch"),
    
    # Attribute errors
    (r"AttributeError: .* has no attribute|AttributeError: '.*' object", "type_or_attr_error"),
    
    # Index/Key errors
    (r"KeyError:|IndexError:|list index out of range", "off_by_one_or_index"),
    
    # None/Optional handling
    (r"NoneType|AttributeError: 'NoneType'|TypeError: 'NoneType'", "none_handling_or_optional"),
    
    # Test assertion failures
    (r"AssertionError:|assert .* ==|Expected.*but got|FAILED", "test_expectation_update"),
    
    # Encoding issues
    (r"UnicodeDecodeError|UnicodeEncodeError|codec can't", "format_or_encoding"),
    
    # File/IO issues
    (r"FileNotFoundError|PermissionError|IsADirectoryError|NotADirectoryError", "io_path_or_permissions"),
    
    # Timeout/Performance
    (r"Timeout|timed out|TimeoutError|deadline exceeded", "performance_timeout"),
    
    # Dependency issues
    (r"version|incompatible|requirement|pkg_resources", "dependency_pin_or_version"),
    
    # Mock/Fixture issues
    (r"fixture|mock|patch|MagicMock|call_count", "mock_or_fixture_issue"),
    
    # Concurrency
    (r"deadlock|race|concurrent|ThreadError|Lock", "concurrency_race"),
]


def classify_failure(test_output: str, failing_files: List[str]) -> List[RepairHypothesis]:
    """
    Classify a test failure into repair hypotheses.
    
    Uses pattern matching on test output to identify likely
    repair categories. Returns ranked hypotheses.
    
    Args:
        test_output: The test output/error message
        failing_files: List of files that likely need fixing
        
    Returns:
        List of RepairHypothesis, ranked by confidence
    """
    hits: List[RepairHypothesis] = []
    out = (test_output or "").strip()
    
    if not out:
        # No output - default to logic error
        return [
            RepairHypothesis(
                kind="logic_branch_or_condition",
                rationale="no test output available; assuming logic issue",
                likely_files=failing_files[:] if failing_files else [],
                hints={},
            )
        ]

    # Apply pattern matching
    seen_kinds: set[str] = set()
    for pattern, kind in _CLASSIFICATION_PATTERNS:
        if re.search(pattern, out, re.IGNORECASE):
            if kind not in seen_kinds:
                seen_kinds.add(kind)
                hits.append(
                    RepairHypothesis(
                        kind=kind,
                        rationale=f"matched pattern: {pattern[:50]}",
                        likely_files=failing_files[:] if failing_files else [],
                        hints={"pattern": pattern, "match": re.search(pattern, out, re.IGNORECASE).group(0)[:200] if re.search(pattern, out, re.IGNORECASE) else ""},
                    )
                )

    # Fallback: logic/condition mismatch
    if not hits:
        hits.append(
            RepairHypothesis(
                kind="logic_branch_or_condition",
                rationale="no strong exception signature; assume logic/test mismatch",
                likely_files=failing_files[:] if failing_files else [],
                hints={},
            )
        )

    # Filter to valid taxonomy entries and limit
    filtered = [h for h in hits if h.kind in TAXONOMY]
    return filtered[:5]


def extract_error_signature(test_output: str, max_length: int = 500) -> str:
    """
    Extract a compact error signature from test output.
    
    Useful for similarity matching and indexing.
    """
    out = (test_output or "").strip()
    
    # Try to find the most informative lines
    lines = out.split("\n")
    error_lines = []
    
    for line in lines:
        line = line.strip()
        if any(kw in line for kw in ["Error", "Exception", "FAILED", "assert", "Traceback"]):
            error_lines.append(line)
    
    if error_lines:
        signature = "\n".join(error_lines[:10])
    else:
        signature = out
    
    return signature[:max_length]
