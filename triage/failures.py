"""
Test Failure Triage System

Classifies and analyzes test failures to determine:
- Is this a real bug or flaky test?
- What type of failure (assertion, exception, timeout)?
- Is this related to the patch or pre-existing?
- Confidence in failure classification
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any

from runner.tests import TestResult, TestStatus
from rfsn_controller.structured_logging import get_logger

logger = get_logger(__name__)


class FailureType(str, Enum):
    """Types of test failures"""
    ASSERTION = "assertion"         # Assertion failed
    EXCEPTION = "exception"          # Uncaught exception
    TIMEOUT = "timeout"              # Test timed out
    IMPORT_ERROR = "import_error"    # Import/module error
    SYNTAX_ERROR = "syntax_error"    # Syntax error
    TYPE_ERROR = "type_error"        # Type error
    ATTRIBUTE_ERROR = "attribute_error"  # Attribute error
    KEY_ERROR = "key_error"          # Key/index error
    SETUP_ERROR = "setup_error"      # Test setup failed
    TEARDOWN_ERROR = "teardown_error"  # Test teardown failed
    FLAKY = "flaky"                  # Inconsistent results
    UNKNOWN = "unknown"              # Unknown failure type


class FailureSeverity(str, Enum):
    """Severity of test failure"""
    CRITICAL = "critical"     # Blocks basic functionality
    HIGH = "high"             # Major feature broken
    MEDIUM = "medium"         # Moderate impact
    LOW = "low"               # Minor issue
    FLAKY = "flaky"          # Intermittent failure


@dataclass
class FailureClassification:
    """Classification of a test failure"""
    test_id: str
    failure_type: FailureType
    severity: FailureSeverity
    is_regression: bool
    confidence: float  # 0.0-1.0
    error_message: str
    root_cause: Optional[str] = None
    suggested_fix: Optional[str] = None
    metadata: Dict[str, Any] = None


class FailureTriage:
    """Triage and classify test failures"""
    
    def __init__(self):
        self.flaky_tests = set()  # Known flaky tests
        self.baseline_failures = {}  # test_id -> FailureClassification
    
    def classify_failure(
        self,
        test: TestResult,
        is_baseline: bool = False
    ) -> FailureClassification:
        """Classify a single test failure"""
        
        # Determine failure type
        failure_type = self._determine_failure_type(test)
        
        # Determine severity
        severity = self._determine_severity(test, failure_type)
        
        # Check if regression (only if not baseline)
        is_regression = False
        if not is_baseline:
            is_regression = test.test_id not in self.baseline_failures
        
        # Extract error message
        error_msg = test.error or test.traceback or "Unknown error"
        
        # Determine confidence
        confidence = self._calculate_confidence(test, failure_type)
        
        # Suggest fix
        suggested_fix = self._suggest_fix(failure_type, error_msg)
        
        classification = FailureClassification(
            test_id=test.test_id,
            failure_type=failure_type,
            severity=severity,
            is_regression=is_regression,
            confidence=confidence,
            error_message=error_msg[:500],  # Truncate
            suggested_fix=suggested_fix,
            metadata={
                "duration_ms": test.duration_ms,
                "status": test.status.value
            }
        )
        
        # Store baseline failures
        if is_baseline and test.status == TestStatus.FAILED:
            self.baseline_failures[test.test_id] = classification
        
        return classification
    
    def _determine_failure_type(self, test: TestResult) -> FailureType:
        """Determine the type of failure"""
        
        content = (test.error + test.traceback + test.output).lower()
        
        # Check for timeout
        if test.status == TestStatus.TIMEOUT or "timeout" in content:
            return FailureType.TIMEOUT
        
        # Check for specific error types
        if "assertionerror" in content or "assert " in content:
            return FailureType.ASSERTION
        
        if "importerror" in content or "modulenotfounderror" in content:
            return FailureType.IMPORT_ERROR
        
        if "syntaxerror" in content:
            return FailureType.SYNTAX_ERROR
        
        if "typeerror" in content:
            return FailureType.TYPE_ERROR
        
        if "attributeerror" in content:
            return FailureType.ATTRIBUTE_ERROR
        
        if "keyerror" in content or "indexerror" in content:
            return FailureType.KEY_ERROR
        
        if "setup" in content and "failed" in content:
            return FailureType.SETUP_ERROR
        
        if "teardown" in content and "failed" in content:
            return FailureType.TEARDOWN_ERROR
        
        # Check if known flaky test
        if test.test_id in self.flaky_tests:
            return FailureType.FLAKY
        
        # Default to exception
        if "exception" in content or "error" in content:
            return FailureType.EXCEPTION
        
        return FailureType.UNKNOWN
    
    def _determine_severity(
        self,
        test: TestResult,
        failure_type: FailureType
    ) -> FailureSeverity:
        """Determine severity of failure"""
        
        # Flaky tests are low severity
        if failure_type == FailureType.FLAKY:
            return FailureSeverity.FLAKY
        
        # Import/syntax errors are critical
        if failure_type in [FailureType.IMPORT_ERROR, FailureType.SYNTAX_ERROR]:
            return FailureSeverity.CRITICAL
        
        # Setup/teardown errors are high
        if failure_type in [FailureType.SETUP_ERROR, FailureType.TEARDOWN_ERROR]:
            return FailureSeverity.HIGH
        
        # Type/attribute errors are medium-high
        if failure_type in [FailureType.TYPE_ERROR, FailureType.ATTRIBUTE_ERROR]:
            return FailureSeverity.MEDIUM
        
        # Assertions and other exceptions depend on test name
        content = test.test_id.lower()
        if any(keyword in content for keyword in ['critical', 'core', 'essential']):
            return FailureSeverity.HIGH
        elif any(keyword in content for keyword in ['integration', 'e2e']):
            return FailureSeverity.MEDIUM
        else:
            return FailureSeverity.LOW
    
    def _calculate_confidence(
        self,
        test: TestResult,
        failure_type: FailureType
    ) -> float:
        """Calculate confidence in classification"""
        
        confidence = 0.5  # Base confidence
        
        # Higher confidence for clear error types
        if failure_type in [
            FailureType.ASSERTION,
            FailureType.IMPORT_ERROR,
            FailureType.SYNTAX_ERROR,
            FailureType.TYPE_ERROR
        ]:
            confidence += 0.3
        
        # Higher confidence if we have traceback
        if test.traceback:
            confidence += 0.2
        
        # Lower confidence for unknown or flaky
        if failure_type in [FailureType.UNKNOWN, FailureType.FLAKY]:
            confidence -= 0.3
        
        return max(0.0, min(1.0, confidence))
    
    def _suggest_fix(
        self,
        failure_type: FailureType,
        error_msg: str
    ) -> Optional[str]:
        """Suggest potential fix based on failure type"""
        
        suggestions = {
            FailureType.IMPORT_ERROR: "Check imports and dependencies",
            FailureType.SYNTAX_ERROR: "Fix syntax errors in code",
            FailureType.TYPE_ERROR: "Check function argument types",
            FailureType.ATTRIBUTE_ERROR: "Verify object attributes exist",
            FailureType.KEY_ERROR: "Check dictionary keys or list indices",
            FailureType.ASSERTION: "Review assertion logic and expected values",
            FailureType.TIMEOUT: "Optimize performance or increase timeout",
            FailureType.FLAKY: "Investigate test stability and dependencies"
        }
        
        return suggestions.get(failure_type)
    
    def triage_stage_results(
        self,
        tests: List[TestResult],
        is_baseline: bool = False
    ) -> List[FailureClassification]:
        """Triage all failures in a stage"""
        
        classifications = []
        
        for test in tests:
            if test.status in [TestStatus.FAILED, TestStatus.ERROR, TestStatus.TIMEOUT]:
                classification = self.classify_failure(test, is_baseline)
                classifications.append(classification)
        
        logger.info(
            f"Triaged {len(classifications)} failures: "
            f"{sum(1 for c in classifications if c.is_regression)} regressions"
        )
        
        return classifications
    
    def identify_regressions(
        self,
        validation_classifications: List[FailureClassification]
    ) -> List[FailureClassification]:
        """Identify tests that regressed (new failures)"""
        
        regressions = [c for c in validation_classifications if c.is_regression]
        
        # Sort by severity
        severity_order = {
            FailureSeverity.CRITICAL: 0,
            FailureSeverity.HIGH: 1,
            FailureSeverity.MEDIUM: 2,
            FailureSeverity.LOW: 3,
            FailureSeverity.FLAKY: 4
        }
        
        regressions.sort(key=lambda c: severity_order[c.severity])
        
        return regressions
    
    def mark_as_flaky(self, test_id: str):
        """Mark a test as flaky"""
        self.flaky_tests.add(test_id)
        logger.info(f"Marked test as flaky: {test_id}")
    
    def get_failure_summary(
        self,
        classifications: List[FailureClassification]
    ) -> Dict[str, Any]:
        """Get summary of failures"""
        
        by_type = {}
        by_severity = {}
        
        for c in classifications:
            by_type[c.failure_type.value] = by_type.get(c.failure_type.value, 0) + 1
            by_severity[c.severity.value] = by_severity.get(c.severity.value, 0) + 1
        
        return {
            "total_failures": len(classifications),
            "regressions": sum(1 for c in classifications if c.is_regression),
            "by_type": by_type,
            "by_severity": by_severity,
            "avg_confidence": sum(c.confidence for c in classifications) / len(classifications) if classifications else 0
        }


if __name__ == "__main__":
    # Test triage
    from runner.tests import TestResult, TestStatus
    
    triage = FailureTriage()
    
    # Mock test result
    test = TestResult(
        test_id="tests/test_math.py::test_divide",
        status=TestStatus.FAILED,
        duration_ms=100,
        error="ZeroDivisionError: division by zero",
        traceback="Traceback...\nZeroDivisionError: division by zero"
    )
    
    classification = triage.classify_failure(test, is_baseline=True)
    print(f"Failure type: {classification.failure_type}")
    print(f"Severity: {classification.severity}")
    print(f"Confidence: {classification.confidence}")
    print(f"Suggestion: {classification.suggested_fix}")
