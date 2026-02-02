"""Tests for Failure Taxonomy Engine."""

import pytest

from rfsn_controller.planner_v2.failure_classifier import (
    FailureClassifier,
    FailureType,
)


class TestFailureClassifier:
    
    @pytest.fixture
    def classifier(self):
        return FailureClassifier()
    
    def test_classify_syntax_error(self, classifier):
        stdout = "File 'main.py', line 10"
        stderr = "SyntaxError: invalid syntax"
        report = classifier.classify(stdout, stderr, 1)
        
        assert report.failure_type == FailureType.SYNTAX_ERROR
        assert report.confidence == 0.9
        assert "SyntaxError:" in report.primary_error

    def test_classify_build_error(self, classifier):
        stdout = "compiling..."
        stderr = "error: linking with `cc` failed: exit code: 1"
        report = classifier.classify(stdout, stderr, 1)
        
        assert report.failure_type == FailureType.BUILD_ERROR
    
    def test_classify_test_failure(self, classifier):
        stdout = "FAILED tests/test_core.py::test_basic"
        stderr = ""
        report = classifier.classify(stdout, stderr, 1)
        
        assert report.failure_type == FailureType.TEST_FAILURE
        
    def test_classify_timeout(self, classifier):
        stdout = ""
        stderr = ""
        report = classifier.classify(stdout, stderr, 124)
        
        assert report.failure_type == FailureType.TIMEOUT
        assert report.confidence == 1.0

    def test_classify_unknown(self, classifier):
        stdout = "Something went wrong"
        stderr = "Just a generic error"
        report = classifier.classify(stdout, stderr, 1)
        
        assert report.failure_type == FailureType.UNKNOWN
