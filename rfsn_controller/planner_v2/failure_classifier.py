"""Failure Taxonomy Engine - Classify failures into typed categories.

This module provides the taxonomy and logic for classifying execution failures
into specific types (BUILD_ERROR, TEST_FAILURE, etc.) to enable intelligent
revision strategies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class FailureType(Enum):
    """Strict taxonomy of failure types."""
    
    BUILD_ERROR = "build_error"
    TEST_FAILURE = "test_failure"
    FLAKY_TEST = "flaky_test"
    LOGIC_REGRESSION = "logic_regression"
    API_MISUSE = "api_misuse"
    DEPENDENCY_CONFLICT = "dependency_conflict"
    ENVIRONMENT_MISMATCH = "environment_mismatch"
    SYNTAX_ERROR = "syntax_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class FailureReport:
    """Structured report of a classified failure."""
    
    failure_type: FailureType
    confidence: float
    evidence: list[str] = field(default_factory=list)
    primary_error: str = ""
    
    def to_dict(self):
        return {
            "failure_type": self.failure_type.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "primary_error": self.primary_error,
        }


class FailureClassifier:
    """Classifies execution failures into the taxonomy."""
    
    def __init__(self):
        # Compiled regex patterns for classification
        self._patterns = [
            (FailureType.SYNTAX_ERROR, re.compile(r"SyntaxError:|IndentationError:", re.I)),
            (FailureType.BUILD_ERROR, re.compile(r"context\s+deadline\s+exceeded|build\s+failed|error:\s+linking|compiler\s+error", re.I)),
            (FailureType.DEPENDENCY_CONFLICT, re.compile(r"ModuleNotFoundError:|ImportError:|ResolutionError:|VersionConflict", re.I)),
            (FailureType.API_MISUSE, re.compile(r"TypeError:|AttributeError:|NameError:|ArgumentError", re.I)),
            (FailureType.TEST_FAILURE, re.compile(r"FAILED\s+tests/|AssertionError:|FAIL:", re.I)),
            (FailureType.TIMEOUT, re.compile(r"TimeLimitExceeded|TimeoutError|timed\s+out", re.I)),
        ]

    def classify(self, stdout: str, stderr: str, exit_code: int) -> FailureReport:
        """Classify a failure based on output logs and exit code.
        
        Args:
            stdout: Standard output log.
            stderr: Standard error log.
            exit_code: Process exit code.
            
        Returns:
            FailureReport with classification details.
        """
        combined = f"{stdout}\n{stderr}"
        
        # 1. Check strict patterns
        for f_type, pattern in self._patterns:
            matches = pattern.findall(combined)
            if matches:
                # Found a match
                return FailureReport(
                    failure_type=f_type,
                    confidence=0.9,
                    evidence=matches[:5],
                    primary_error=matches[0]
                )
        
        # 2. Fallback heuristics based on exit code
        if exit_code == 124: # Timeout
            return FailureReport(
                failure_type=FailureType.TIMEOUT,
                confidence=1.0, 
                evidence=["Exit code 124"],
                primary_error="Process timed out"
            )
            
        # 3. Default to UNKNOWN
        return FailureReport(
            failure_type=FailureType.UNKNOWN,
            confidence=0.1,
            evidence=[],
            primary_error="Unknown failure pattern"
        )

    def refine_classification(self, report: FailureReport, verified_history: list[dict]) -> FailureReport:
        """Refine classification based on historical verification (e.g. flaky tests).
        
        Args:
            report: The initial report.
            verified_history: List of past outcomes for this test/step.
            
        Returns:
            Refined FailureReport.
        """
        # Logic to detect FLAKY_TEST or LOGIC_REGRESSION
        # Simple heuristic: if it failed similarly before and then passed without change -> Flaky
        # If it passed before and now fails after a change -> Regression
        
        return report
