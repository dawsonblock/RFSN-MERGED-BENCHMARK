"""Tests for coverage analyzer."""

import pytest


class TestCoverageReport:
    """Tests for CoverageReport dataclass."""

    def test_has_untested_changes_true(self):
        """Detects untested files."""
        from rfsn_controller.qa import CoverageReport

        report = CoverageReport(
            touched_files=["src/foo.py"],
            untested_files=["src/foo.py"],
        )
        assert report.has_untested_changes

    def test_has_untested_changes_false(self):
        """No untested files detected."""
        from rfsn_controller.qa import CoverageReport

        report = CoverageReport(
            touched_files=["src/foo.py"],
            covered_files=["src/foo.py"],
            untested_files=[],
        )
        assert not report.has_untested_changes

    def test_as_evidence_data(self):
        """Converts to evidence dict."""
        from rfsn_controller.qa import CoverageReport

        report = CoverageReport(
            touched_files=["src/foo.py"],
            covered_files=["src/foo.py"],
            untested_files=[],
            coverage_map={"src/foo.py": 85.0},
            total_coverage=85.0,
            confidence="high",
        )
        data = report.as_evidence_data()
        assert data["coverage_confidence"] == "high"
        assert data["total_coverage"] == 85.0
        assert "src/foo.py" in data["touched_files"]


class TestCoverageAnalyzer:
    """Tests for CoverageAnalyzer."""

    def test_is_test_file_detection(self):
        """Correctly identifies test files."""
        from rfsn_controller.qa import CoverageAnalyzer

        analyzer = CoverageAnalyzer()
        assert analyzer._is_test_file("test_foo.py")
        assert analyzer._is_test_file("foo_test.py")
        assert analyzer._is_test_file("conftest.py")
        assert not analyzer._is_test_file("foo.py")
        assert not analyzer._is_test_file("src/utils.py")

    def test_analyze_patch_with_coverage_data(self):
        """Analyzes patch with mock coverage data."""
        from rfsn_controller.qa import CoverageAnalyzer

        analyzer = CoverageAnalyzer()
        coverage_data = {
            "files": {
                "src/foo.py": {"summary": {"percent_covered": 85.0}},
                "src/bar.py": {"summary": {"percent_covered": 0.0}},
            }
        }
        report = analyzer.analyze_patch(
            ["src/foo.py", "src/bar.py", "test_foo.py"],
            coverage_data=coverage_data,
        )

        assert "src/foo.py" in report.covered_files
        assert "src/bar.py" in report.untested_files
        assert "test_foo.py" not in report.covered_files  # Test file filtered
        assert report.confidence == "low"  # Has untested files

    def test_high_confidence_all_covered(self):
        """High confidence when all files well-covered."""
        from rfsn_controller.qa import CoverageAnalyzer

        analyzer = CoverageAnalyzer()
        coverage_data = {
            "files": {
                "src/foo.py": {"summary": {"percent_covered": 90.0}},
                "src/bar.py": {"summary": {"percent_covered": 85.0}},
            }
        }
        report = analyzer.analyze_patch(
            ["src/foo.py", "src/bar.py"],
            coverage_data=coverage_data,
        )

        assert report.confidence == "high"
        assert report.total_coverage == 87.5

    def test_medium_confidence_moderate_coverage(self):
        """Medium confidence for moderate coverage."""
        from rfsn_controller.qa import CoverageAnalyzer

        analyzer = CoverageAnalyzer()
        coverage_data = {
            "files": {
                "src/foo.py": {"summary": {"percent_covered": 60.0}},
            }
        }
        report = analyzer.analyze_patch(["src/foo.py"], coverage_data=coverage_data)

        assert report.confidence == "medium"

    def test_unknown_confidence_no_data(self):
        """Unknown confidence when no coverage data."""
        from rfsn_controller.qa import CoverageAnalyzer

        analyzer = CoverageAnalyzer()
        report = analyzer.analyze_patch(["src/foo.py"])

        assert report.confidence == "unknown"

    def test_only_test_files_high_confidence(self):
        """High confidence if only test files touched."""
        from rfsn_controller.qa import CoverageAnalyzer

        analyzer = CoverageAnalyzer()
        coverage_data = {"files": {}}
        report = analyzer.analyze_patch(
            ["test_foo.py", "test_bar.py"],
            coverage_data=coverage_data,
        )

        assert report.confidence == "high"

    def test_extract_coverage_executed_lines(self):
        """Extracts coverage from executed/missing lines format."""
        from rfsn_controller.qa import CoverageAnalyzer

        analyzer = CoverageAnalyzer()
        file_data = {
            "executed_lines": [1, 2, 3, 4],
            "missing_lines": [5],
        }
        pct = analyzer._extract_coverage_pct(file_data)
        assert pct == 80.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
