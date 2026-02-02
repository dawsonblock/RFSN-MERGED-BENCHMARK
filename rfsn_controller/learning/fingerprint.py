"""Failure Fingerprinting.

Extracts stable fingerprints from test failures and lint errors
for use in episodic memory and strategy selection.

The fingerprint enables:
- Recognizing similar failure patterns across runs
- Mapping failures to successful strategies
- Avoiding strategies that consistently fail for certain patterns
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FailureFingerprint:
    """Immutable fingerprint of a failure pattern.
    
    Fields are designed to be stable across minor variations
    while still capturing the essential failure signature.
    """
    
    # Coarse category (IMPORT_ERROR, API_MISMATCH, NULL_DEREF, etc.)
    category: str
    
    # Stable test names (sorted for consistency)
    failing_tests: tuple[str, ...]
    
    # Lint codes (sorted)
    lint_codes: tuple[str, ...]
    
    # Files involved (sorted basenames only)
    affected_files: tuple[str, ...]
    
    # Error class from stack trace (e.g., "TypeError", "AttributeError")
    error_class: str | None = None
    
    # Normalized stack signature (top 3 frames)
    stack_signature: str | None = None
    
    def __hash__(self) -> int:
        return hash((
            self.category,
            self.failing_tests,
            self.lint_codes,
            self.affected_files,
            self.error_class,
        ))
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "category": self.category,
            "failing_tests": list(self.failing_tests),
            "lint_codes": list(self.lint_codes),
            "affected_files": list(self.affected_files),
            "error_class": self.error_class,
            "stack_signature": self.stack_signature,
        }


# Pattern matchers for categorization
CATEGORY_PATTERNS = {
    "IMPORT_ERROR": [
        r"ImportError",
        r"ModuleNotFoundError",
        r"cannot import name",
        r"No module named",
    ],
    "TYPE_ERROR": [
        r"TypeError",
        r"type.*expected",
        r"incompatible type",
    ],
    "ATTRIBUTE_ERROR": [
        r"AttributeError",
        r"has no attribute",
        r"'NoneType'.*attribute",
    ],
    "ASSERTION_ERROR": [
        r"AssertionError",
        r"assert.*failed",
        r"expected.*got",
    ],
    "INDEX_ERROR": [
        r"IndexError",
        r"list index out of range",
        r"KeyError",
    ],
    "NULL_DEREF": [
        r"NoneType",
        r"null",
        r"undefined",
    ],
    "SYNTAX_ERROR": [
        r"SyntaxError",
        r"invalid syntax",
        r"EOL while scanning",
    ],
    "LINT_ERROR": [
        r"E\d{3}",  # PEP8 codes
        r"W\d{3}",
        r"F\d{3}",  # Pyflakes
    ],
    "TIMEOUT": [
        r"timeout",
        r"timed out",
        r"deadline exceeded",
    ],
}


def categorize_failure(
    test_output: str,
    lint_errors: list[str] | None = None,
) -> str:
    """Categorize a failure into a coarse class.
    
    Args:
        test_output: Test failure output (stack traces, etc.)
        lint_errors: List of lint error messages.
        
    Returns:
        Category string (e.g., "IMPORT_ERROR", "TYPE_ERROR").
    """
    combined = test_output.lower()
    if lint_errors:
        combined += " ".join(lint_errors).lower()
    
    for category, patterns in CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern.lower(), combined):
                return category
    
    return "UNKNOWN"


def extract_error_class(stack_trace: str) -> str | None:
    """Extract error class from stack trace.
    
    Args:
        stack_trace: Python stack trace.
        
    Returns:
        Error class name (e.g., "TypeError") or None.
    """
    # Match "ErrorType: message" at end of traceback
    match = re.search(r"(\w+Error|\w+Exception|\w+Warning):", stack_trace)
    if match:
        return match.group(1)
    return None


def normalize_stack_signature(stack_trace: str, max_frames: int = 3) -> str:
    """Extract normalized stack signature from trace.
    
    Normalizes by:
    - Taking only top N frames
    - Removing line numbers
    - Keeping only file + function names
    
    Args:
        stack_trace: Full stack trace.
        max_frames: Max frames to include.
        
    Returns:
        Normalized signature string.
    """
    # Match Python traceback frames
    frame_pattern = r'File "([^"]+)", line \d+, in (\w+)'
    frames = re.findall(frame_pattern, stack_trace)
    
    # Keep only basename and function
    normalized = []
    for file_path, func_name in frames[:max_frames]:
        basename = file_path.split("/")[-1]
        normalized.append(f"{basename}:{func_name}")
    
    return "->".join(normalized)


def fingerprint_failure(
    failing_tests: list[str] | None = None,
    lint_errors: list[str] | None = None,
    stack_trace: str | None = None,
    affected_files: list[str] | None = None,
) -> FailureFingerprint:
    """Create a fingerprint from failure information.
    
    Args:
        failing_tests: List of failing test names.
        lint_errors: List of lint error messages.
        stack_trace: Optional stack trace.
        affected_files: Optional list of affected file paths.
        
    Returns:
        FailureFingerprint instance.
    """
    # Normalize test names (take just the test function/method name)
    tests = []
    for test in (failing_tests or [])[:5]:
        # Strip path prefixes, keep test name
        parts = test.replace("::", ".").split(".")
        tests.append(parts[-1] if parts else test)
    
    # Extract lint codes
    codes = []
    for err in (lint_errors or [])[:5]:
        # Extract error codes like E501, F401, W293
        matches = re.findall(r"[A-Z]\d{3,4}", err)
        codes.extend(matches)
    
    # Normalize file paths to basenames
    files = []
    for f in (affected_files or [])[:10]:
        files.append(f.split("/")[-1])
    
    # Categorize
    combined_output = "\n".join([
        stack_trace or "",
        "\n".join(lint_errors or []),
        "\n".join(failing_tests or []),
    ])
    category = categorize_failure(combined_output, lint_errors)
    
    # Extract error class
    error_class = None
    if stack_trace:
        error_class = extract_error_class(stack_trace)
    
    # Normalize stack signature
    stack_sig = None
    if stack_trace:
        stack_sig = normalize_stack_signature(stack_trace)
    
    return FailureFingerprint(
        category=category,
        failing_tests=tuple(sorted(set(tests))),
        lint_codes=tuple(sorted(set(codes))),
        affected_files=tuple(sorted(set(files))),
        error_class=error_class,
        stack_signature=stack_sig,
    )


def compute_fingerprint_hash(fingerprint: FailureFingerprint) -> str:
    """Compute stable hash of fingerprint for database keys.
    
    Args:
        fingerprint: Fingerprint to hash.
        
    Returns:
        Hex digest string.
    """
    key_parts = [
        fingerprint.category,
        ",".join(fingerprint.failing_tests[:5]),
        ",".join(fingerprint.lint_codes[:5]),
    ]
    key_str = "|".join(key_parts)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]
