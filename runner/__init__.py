"""
Test Runner Package

Staged test execution with artifact capture and Docker support.
"""

from .tests import (
    StagedTestRunner,
    TestRunConfig,
    TestStage,
    TestStatus,
    TestResult,
    StageResult,
    run_staged_tests
)

from .artifacts import (
    ArtifactCapture,
    TestArtifact,
    ArtifactCollection,
    extract_stack_traces,
    extract_error_messages
)

__all__ = [
    # Test Runner
    "StagedTestRunner",
    "TestRunConfig",
    "TestStage",
    "TestStatus",
    "TestResult",
    "StageResult",
    "run_staged_tests",
    # Artifacts
    "ArtifactCapture",
    "TestArtifact",
    "ArtifactCollection",
    "extract_stack_traces",
    "extract_error_messages",
]
