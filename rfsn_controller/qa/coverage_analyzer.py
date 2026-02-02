"""Coverage analyzer for QA evidence.
from __future__ import annotations

Parses pytest-cov output to detect:
- Which files are touched by a patch
- Which files have test coverage
- Which touched files are NOT covered by tests (low confidence)
"""

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CoverageReport:
    """Coverage analysis for touched files."""

    touched_files: list[str] = field(default_factory=list)
    covered_files: list[str] = field(default_factory=list)
    untested_files: list[str] = field(default_factory=list)
    coverage_map: dict[str, float] = field(default_factory=dict)  # file -> coverage %
    total_coverage: float = 0.0
    confidence: str = "unknown"  # "high", "medium", "low", "unknown"

    @property
    def has_untested_changes(self) -> bool:
        return len(self.untested_files) > 0

    def as_evidence_data(self) -> dict[str, Any]:
        return {
            "touched_files": self.touched_files,
            "covered_files": self.covered_files,
            "untested_changes": self.untested_files,
            "coverage_map": self.coverage_map,
            "total_coverage": self.total_coverage,
            "coverage_confidence": self.confidence,
        }


class CoverageAnalyzer:
    """Analyzes test coverage for patch validation."""

    # Files that are typically not covered by tests
    IGNORE_PATTERNS = [
        r"^test_",
        r"_test\.py$",
        r"conftest\.py$",
        r"setup\.py$",
        r"__init__\.py$",
    ]

    # Thresholds for confidence levels
    HIGH_COVERAGE_THRESHOLD = 80.0
    MEDIUM_COVERAGE_THRESHOLD = 50.0

    def __init__(
        self,
        *,
        cwd: str | None = None,
        timeout_seconds: int = 60,
        cov_report_path: str | None = None,
        enable_cache: bool = True,
    ):
        """Initialize analyzer.
        
        Args:
            cwd: Working directory for running coverage.
            timeout_seconds: Timeout for coverage command.
            cov_report_path: Path to existing coverage JSON report.
            enable_cache: Enable caching of coverage data (default: True).
        """
        self.cwd = cwd
        self.timeout_seconds = timeout_seconds
        self.cov_report_path = cov_report_path
        self.enable_cache = enable_cache
        self._cached_coverage_data: dict[str, Any] | None = None
        self._cache_loaded: bool = False

    def analyze_patch(
        self,
        touched_files: list[str],
        *,
        coverage_data: dict[str, Any] | None = None,
    ) -> CoverageReport:
        """Analyze coverage for files touched by a patch.
        
        Args:
            touched_files: List of files modified by the patch.
            coverage_data: Optional pre-loaded coverage data.
        
        Returns:
            CoverageReport with analysis.
        """
        # Filter out test files from analysis
        source_files = [
            f for f in touched_files
            if not self._is_test_file(f)
        ]

        if coverage_data is None:
            coverage_data = self._load_coverage_data()

        if coverage_data is None:
            return CoverageReport(
                touched_files=touched_files,
                confidence="unknown",
            )

        covered: list[str] = []
        untested: list[str] = []
        coverage_map: dict[str, float] = {}

        # Check coverage for each touched file
        for file_path in source_files:
            file_cov = self._get_file_coverage(file_path, coverage_data)
            if file_cov is not None:
                coverage_map[file_path] = file_cov
                if file_cov > 0:
                    covered.append(file_path)
                else:
                    untested.append(file_path)
            else:
                # File not in coverage report = untested
                untested.append(file_path)

        # Calculate total coverage for touched files
        if coverage_map:
            total_cov = sum(coverage_map.values()) / len(coverage_map)
        else:
            total_cov = 0.0

        # Determine confidence level
        if not source_files:
            confidence = "high"  # Only test files touched
        elif not untested:
            if total_cov >= self.HIGH_COVERAGE_THRESHOLD:
                confidence = "high"
            elif total_cov >= self.MEDIUM_COVERAGE_THRESHOLD:
                confidence = "medium"
            else:
                confidence = "low"
        else:
            confidence = "low"

        return CoverageReport(
            touched_files=touched_files,
            covered_files=covered,
            untested_files=untested,
            coverage_map=coverage_map,
            total_coverage=total_cov,
            confidence=confidence,
        )

    def _is_test_file(self, file_path: str) -> bool:
        """Check if file is a test file."""
        basename = os.path.basename(file_path)
        for pattern in self.IGNORE_PATTERNS:
            if re.search(pattern, basename):
                return True
        return False

    def _load_coverage_data(self) -> dict[str, Any] | None:
        """Load coverage data from JSON report or run coverage.
        
        Uses cached data if available and caching is enabled.
        """
        # Return cached data if available
        if self.enable_cache and self._cache_loaded:
            return self._cached_coverage_data

        data = None

        # Try loading from specified path
        if self.cov_report_path and os.path.exists(self.cov_report_path):
            try:
                with open(self.cov_report_path) as f:
                    data = json.load(f)
            except Exception as e:
                logger.warning("Failed to load coverage report: %s", e)

        # Try common coverage report locations
        if data is None:
            common_paths = [
                "coverage.json",
                ".coverage.json",
                "htmlcov/coverage.json",
            ]

            for path in common_paths:
                full_path = os.path.join(self.cwd or ".", path)
                if os.path.exists(full_path):
                    try:
                        with open(full_path) as f:
                            data = json.load(f)
                            break
                    except Exception:
                        continue

        # Cache the result
        if self.enable_cache:
            self._cached_coverage_data = data
            self._cache_loaded = True

        return data

    def invalidate_cache(self) -> None:
        """Invalidate cached coverage data."""
        self._cached_coverage_data = None
        self._cache_loaded = False

    def _get_file_coverage(
        self,
        file_path: str,
        coverage_data: dict[str, Any],
    ) -> float | None:
        """Get coverage percentage for a file.
        
        Args:
            file_path: File to check.
            coverage_data: Coverage JSON data.
        
        Returns:
            Coverage percentage (0-100) or None if not found.
        """
        # Handle different coverage JSON formats
        files = coverage_data.get("files", {})

        # Try exact match first
        if file_path in files:
            file_data = files[file_path]
            return self._extract_coverage_pct(file_data)

        # Try basename match
        basename = os.path.basename(file_path)
        for cov_path, file_data in files.items():
            if os.path.basename(cov_path) == basename:
                return self._extract_coverage_pct(file_data)

        # Try partial path match
        for cov_path, file_data in files.items():
            if file_path.endswith(cov_path) or cov_path.endswith(file_path):
                return self._extract_coverage_pct(file_data)

        return None

    def _extract_coverage_pct(self, file_data: dict[str, Any]) -> float:
        """Extract coverage percentage from file data."""
        # Format 1: {"summary": {"percent_covered": X}}
        if "summary" in file_data:
            return file_data["summary"].get("percent_covered", 0.0)

        # Format 2: {"coverage": X}
        if "coverage" in file_data:
            return file_data["coverage"]

        # Format 3: Calculate from executed/missing lines
        executed = len(file_data.get("executed_lines", []))
        missing = len(file_data.get("missing_lines", []))
        total = executed + missing
        if total > 0:
            return (executed / total) * 100.0

        return 0.0

    def run_coverage(
        self,
        test_cmd: str = "pytest",
        *,
        output_path: str = "coverage.json",
    ) -> dict[str, Any] | None:
        """Run coverage and return data.
        
        Args:
            test_cmd: Test command (typically pytest).
            output_path: Path for JSON output.
        
        Returns:
            Coverage data dict or None on failure.
        """
        cov_cmd = f"{test_cmd} --cov --cov-report=json:{output_path} -q"

        try:
            import shlex
            subprocess.run(
                shlex.split(cov_cmd),
                shell=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=self.cwd, check=False,
            )

            if os.path.exists(output_path):
                with open(output_path) as f:
                    return json.load(f)
        except Exception as e:
            logger.warning("Failed to run coverage: %s", e)

        return None


def create_coverage_analyzer(**kwargs) -> CoverageAnalyzer:
    """Factory function for CoverageAnalyzer."""
    return CoverageAnalyzer(**kwargs)


def coverage_evidence_fn(
    touched_files: list[str],
    coverage_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create coverage evidence for EvidenceCollector.
    
    Args:
        touched_files: Files touched by patch.
        coverage_data: Optional pre-loaded coverage data.
    
    Returns:
        Evidence data dict.
    """
    analyzer = CoverageAnalyzer()
    report = analyzer.analyze_patch(touched_files, coverage_data=coverage_data)
    return report.as_evidence_data()
