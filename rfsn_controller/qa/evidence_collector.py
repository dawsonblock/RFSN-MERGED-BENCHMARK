"""Evidence collector for QA claims.
from __future__ import annotations

Controller-driven evidence gathering. NOT an LLM.
Uses existing tools to collect evidence for challenged claims.
"""

import logging
from collections.abc import Callable
from typing import Any

from .qa_types import (
    ClaimVerdict,
    DeltaMapEvidence,
    Evidence,
    EvidenceType,
    PolicyCheckEvidence,
    StaticCheckEvidence,
    TestResultEvidence,
    Verdict,
)

logger = logging.getLogger(__name__)


class EvidenceCollector:
    """Collects evidence to resolve challenged claims.
    
    This is NOT an LLM. It uses controller tools to gather
    deterministic, verifiable evidence.
    """

    def __init__(
        self,
        *,
        test_runner: Callable[[str], dict[str, Any]] | None = None,
        delta_tracker: Any | None = None,  # TestDeltaTracker
        hygiene_validator: Callable[[str], dict[str, Any]] | None = None,
        static_checker: Callable[[str], dict[str, Any]] | None = None,
        coverage_analyzer: Any | None = None,  # CoverageAnalyzer
        timeout_ms: int = 60000,
    ):
        """Initialize collector.
        
        Args:
            test_runner: Function(test_cmd) -> {exit_code, failing_tests, ...}
            delta_tracker: TestDeltaTracker instance for delta computation.
            hygiene_validator: Function(diff) -> {is_valid, violations, ...}
            static_checker: Function(tool) -> {exit_code, issues, ...}
            coverage_analyzer: CoverageAnalyzer for coverage evidence.
            timeout_ms: Timeout for evidence collection.
        """
        self.test_runner = test_runner
        self.delta_tracker = delta_tracker
        self.hygiene_validator = hygiene_validator
        self.static_checker = static_checker
        self.coverage_analyzer = coverage_analyzer
        self.timeout_ms = timeout_ms

    def collect_for_verdict(
        self,
        verdict: ClaimVerdict,
        *,
        diff: str = "",
        test_cmd: str = "",
    ) -> list[Evidence]:
        """Collect evidence for a challenged claim.
        
        Args:
            verdict: The CHALLENGE verdict with evidence_request.
            diff: The patch diff (for hygiene checks).
            test_cmd: Test command to run.
        
        Returns:
            List of collected evidence.
        """
        if verdict.verdict != Verdict.CHALLENGE:
            return []

        evidence: list[Evidence] = []
        request = (verdict.evidence_request or "").lower()

        # Parse what evidence is needed
        if "test_result" in request or "test" in request:
            if self.test_runner and test_cmd:
                try:
                    result = self.test_runner(test_cmd)
                    evidence.append(TestResultEvidence(
                        command=test_cmd,
                        exit_code=result.get("exit_code", 1),
                        failing_tests=result.get("failing_tests", []),
                        passing_tests=result.get("passing_tests", []),
                        duration_ms=result.get("duration_ms", 0),
                    ).to_evidence())
                except Exception as e:
                    logger.warning("Failed to run tests: %s", e)

        if "delta_map" in request or "delta" in request:
            if self.delta_tracker:
                try:
                    # Get current failing tests from test runner result
                    current_failing = set()
                    for ev in evidence:
                        if ev.type == EvidenceType.TEST_RESULT:
                            current_failing = set(ev.data.get("failing_tests", []))
                            break

                    fixed, regressed = self.delta_tracker.compute_delta(current_failing)
                    evidence.append(DeltaMapEvidence(
                        fixed=list(fixed),
                        regressed=list(regressed),
                        still_failing=list(current_failing - fixed),
                    ).to_evidence())
                except Exception as e:
                    logger.warning("Failed to compute delta: %s", e)

        if "policy_check" in request or "policy" in request or "hygiene" in request:
            if self.hygiene_validator and diff:
                try:
                    result = self.hygiene_validator(diff)
                    evidence.append(PolicyCheckEvidence(
                        is_valid=result.get("is_valid", False),
                        violations=result.get("violations", []),
                        diff_stats=result.get("diff_stats", {}),
                    ).to_evidence())
                except Exception as e:
                    logger.warning("Failed hygiene check: %s", e)

        if "static_check" in request or "static" in request or "lint" in request:
            if self.static_checker:
                try:
                    result = self.static_checker("ruff")  # Default to ruff
                    evidence.append(StaticCheckEvidence(
                        tool="ruff",
                        exit_code=result.get("exit_code", 1),
                        issues=result.get("issues", []),
                    ).to_evidence())
                except Exception as e:
                    logger.warning("Failed static check: %s", e)

        if "coverage" in request or "untested" in request:
            if self.coverage_analyzer:
                try:
                    # Get touched files from diff
                    touched_files = self._parse_touched_files(diff)
                    report = self.coverage_analyzer.analyze_patch(touched_files)
                    evidence.append(Evidence(
                        type=EvidenceType.COVERAGE_SIGNAL,
                        data=report.as_evidence_data(),
                    ))
                except Exception as e:
                    logger.warning("Failed coverage analysis: %s", e)

        return evidence

    def collect_all(
        self,
        verdicts: list[ClaimVerdict],
        *,
        diff: str = "",
        test_cmd: str = "",
    ) -> dict[str, list[Evidence]]:
        """Collect evidence for all challenged claims.
        
        Args:
            verdicts: List of verdicts (will collect for CHALLENGE only).
            diff: The patch diff.
            test_cmd: Test command.
        
        Returns:
            Dict of claim_id -> evidence list.
        """
        result: dict[str, list[Evidence]] = {}

        for verdict in verdicts:
            if verdict.verdict == Verdict.CHALLENGE:
                evidence = self.collect_for_verdict(
                    verdict,
                    diff=diff,
                    test_cmd=test_cmd,
                )
                result[verdict.claim_id] = evidence

        return result

    def create_test_result(
        self,
        command: str,
        exit_code: int,
        failing_tests: list[str],
        **kwargs,
    ) -> Evidence:
        """Helper to create test result evidence."""
        return TestResultEvidence(
            command=command,
            exit_code=exit_code,
            failing_tests=failing_tests,
            **kwargs,
        ).to_evidence()

    def create_delta_map(
        self,
        fixed: list[str],
        regressed: list[str],
        still_failing: list[str] | None = None,
    ) -> Evidence:
        """Helper to create delta map evidence."""
        return DeltaMapEvidence(
            fixed=fixed,
            regressed=regressed,
            still_failing=still_failing or [],
        ).to_evidence()

    def create_policy_check(
        self,
        is_valid: bool,
        violations: list[str] | None = None,
        diff_stats: dict[str, int] | None = None,
    ) -> Evidence:
        """Helper to create policy check evidence."""
        return PolicyCheckEvidence(
            is_valid=is_valid,
            violations=violations or [],
            diff_stats=diff_stats or {},
        ).to_evidence()

    def _parse_touched_files(self, diff: str) -> list[str]:
        """Parse touched file paths from a diff string."""
        files = []
        if not diff:
            return files
        for line in diff.splitlines():
            if line.startswith('diff --git'):
                # Extract filename from "diff --git a/foo.py b/foo.py"
                parts = line.split()
                if len(parts) >= 4:
                    files.append(parts[3].lstrip('b/'))
        return files


def create_evidence_collector(**kwargs) -> EvidenceCollector:
    """Factory function for EvidenceCollector."""
    return EvidenceCollector(**kwargs)
