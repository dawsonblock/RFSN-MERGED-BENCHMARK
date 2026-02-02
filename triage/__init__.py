"""
Test Failure Triage Package

Classification and analysis of test failures.
"""

from .failures import (
    FailureTriage,
    FailureClassification,
    FailureType,
    FailureSeverity
)

__all__ = [
    "FailureTriage",
    "FailureClassification",
    "FailureType",
    "FailureSeverity",
]
