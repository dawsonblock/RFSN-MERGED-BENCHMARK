"""Unit tests for vNext hardening components.

Tests cover:
- URL validation
- Tool request deduplication and hashing
- Patch hygiene gates
- Stall detection
"""

import os
import sys
import unittest

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rfsn_controller.patch_hygiene import PatchHygieneConfig, validate_patch_hygiene
from rfsn_controller.stall_detector import StallState
from rfsn_controller.tool_manager import ToolRequest, ToolRequestConfig, ToolRequestManager
from rfsn_controller.url_validation import validate_github_url


class TestURLValidation:
    """Tests for GitHub URL validation."""

    def test_valid_github_urls(self):
        """Test that valid GitHub URLs are accepted."""
        valid_urls = [
            "https://github.com/owner/repo",
            "https://github.com/owner/repo.git",
            "https://github.com/owner-name/repo-name",
            "https://github.com/owner123/repo456",
            "http://github.com/owner/repo",  # http is also accepted by the regex
        ]
        for url in valid_urls:
            is_valid, normalized, error = validate_github_url(url)
            assert is_valid, f"URL should be valid: {url}"
            assert normalized is not None
            assert error is None

    def test_invalid_github_urls(self):
        """Test that invalid GitHub URLs are rejected."""
        invalid_urls = [
            "https://gitlab.com/owner/repo",  # Wrong domain
            "github.com/owner/repo",  # Missing scheme
            "https://github.com/owner/repo/extra/path",  # Extra path
            "https://github.com/owner/",  # Missing repo
            "https://github.com//repo",  # Missing owner
            "ftp://github.com/owner/repo",  # Wrong scheme
            "https://notgithub.com/owner/repo",  # Wrong domain
        ]
        for url in invalid_urls:
            is_valid, normalized, error = validate_github_url(url)
            assert not is_valid, f"URL should be invalid: {url}"
            assert normalized is None
            assert error is not None

    def test_url_normalization(self):
        """Test that URLs are normalized correctly."""
        test_cases = [
            (
                "https://github.com/owner/repo.git",
                "https://github.com/owner/repo.git",
            ),  # .git is NOT stripped
            ("https://github.com/owner/repo", "https://github.com/owner/repo"),
        ]
        for input_url, expected_normalized in test_cases:
            is_valid, normalized, error = validate_github_url(input_url)
            assert is_valid
            assert normalized == expected_normalized


class TestToolDedupe:
    """Tests for tool request deduplication and hashing."""

    def test_tool_request_signature_unique(self):
        """Test that different tool requests have different signatures."""
        req1 = ToolRequest(tool="sandbox.read_file", args={"path": "test.py"})
        req2 = ToolRequest(tool="sandbox.read_file", args={"path": "other.py"})
        req3 = ToolRequest(tool="sandbox.grep", args={"pattern": "test"})

        sig1 = req1.signature()
        sig2 = req2.signature()
        sig3 = req3.signature()

        assert sig1 != sig2, "Different paths should have different signatures"
        assert sig1 != sig3, "Different tools should have different signatures"
        assert sig2 != sig3, "Different tools should have different signatures"

    def test_tool_request_signature_consistent(self):
        """Test that identical tool requests have the same signature."""
        req1 = ToolRequest(tool="sandbox.read_file", args={"path": "test.py"})
        req2 = ToolRequest(tool="sandbox.read_file", args={"path": "test.py"})

        assert req1.signature() == req2.signature(), "Identical requests should have same signature"

    def test_tool_request_signature_args_order(self):
        """Test that signature is independent of argument order."""
        req1 = ToolRequest(tool="sandbox.test", args={"a": 1, "b": 2})
        req2 = ToolRequest(tool="sandbox.test", args={"b": 2, "a": 1})

        assert req1.signature() == req2.signature(), "Signature should be order-independent"

    def test_tool_manager_deduplication(self):
        """Test that tool manager blocks duplicate requests."""
        config = ToolRequestConfig(max_total_requests_per_run=10)
        manager = ToolRequestManager(config)

        # First request should be allowed
        is_allowed, reason = manager.should_allow_request("sandbox.read_file", {"path": "test.py"})
        assert is_allowed
        assert reason is None

        # Register the request
        manager.register_request("sandbox.read_file", {"path": "test.py"})

        # Duplicate request should be blocked
        is_allowed, reason = manager.should_allow_request("sandbox.read_file", {"path": "test.py"})
        assert not is_allowed
        assert "Duplicate request blocked" in reason

    def test_tool_manager_quota(self):
        """Test that tool manager enforces quota limits."""
        config = ToolRequestConfig(max_total_requests_per_run=3)
        manager = ToolRequestManager(config)

        # First 3 requests should be allowed
        for i in range(3):
            is_allowed, reason = manager.should_allow_request(
                "sandbox.read_file", {"path": f"test{i}.py"}
            )
            assert is_allowed, f"Request {i} should be allowed"
            manager.register_request("sandbox.read_file", {"path": f"test{i}.py"})

        # 4th request should be blocked by quota
        is_allowed, reason = manager.should_allow_request("sandbox.read_file", {"path": "test3.py"})
        assert not is_allowed
        assert "quota exceeded" in reason.lower()

    def test_tool_manager_filter_requests(self):
        """Test that tool manager filters request lists."""
        config = ToolRequestConfig(max_total_requests_per_run=10, max_requests_per_response=2)
        manager = ToolRequestManager(config)

        requests = [
            {"tool": "sandbox.read_file", "args": {"path": "test.py"}},
            {"tool": "sandbox.read_file", "args": {"path": "test.py"}},  # Duplicate
            {"tool": "sandbox.grep", "args": {"pattern": "test"}},
        ]

        allowed, blocked = manager.filter_requests(requests)

        # First: requests are truncated to max_requests_per_response=2
        # Then: duplicates are filtered out
        # Result: only 1 unique request allowed (the first read_file)
        assert len(allowed) == 1
        assert allowed[0]["tool"] == "sandbox.read_file"


class TestPatchHygiene:
    """Tests for patch hygiene gates."""

    def test_valid_patch(self):
        """Test that a valid patch passes hygiene gates."""
        diff = """--- a/test.py
+++ b/test.py
@@ -1,3 +1,3 @@
-def old_func():
+def new_func():
     return 1
"""
        result = validate_patch_hygiene(diff)
        assert result.is_valid
        assert len(result.violations) == 0

    def test_too_many_files(self):
        """Test that patches with too many files are rejected."""
        # Create a diff with 6 files (default max is 5)
        diff = ""
        for i in range(6):
            diff += f"""--- a/test{i}.py
+++ b/test{i}.py
@@ -0,0 +1 @@
+pass
"""
        result = validate_patch_hygiene(diff, PatchHygieneConfig(max_files_changed=5))
        assert not result.is_valid
        assert any("Too many files changed" in v for v in result.violations)

    def test_too_many_lines(self):
        """Test that patches with too many lines are rejected."""
        # Create a diff with 251 lines (default max is 250)
        diff = "--- a/test.py\n+++ b/test.py\n"
        for i in range(251):
            diff += f"+line {i}\n"

        result = validate_patch_hygiene(diff, PatchHygieneConfig(max_lines_changed=250))
        assert not result.is_valid
        assert any("Too many lines changed" in v for v in result.violations)

    def test_forbidden_directories(self):
        """Test that patches touching forbidden directories are rejected."""
        forbidden_dirs = [".git/", "node_modules/", "__pycache__/"]
        for forbidden_dir in forbidden_dirs:
            diff = f"""--- a/{forbidden_dir}test.py
+++ b/{forbidden_dir}test.py
@@ -0,0 +1 @@
+pass
"""
            result = validate_patch_hygiene(diff)
            # Check if the file path was parsed correctly
            # The parser extracts the path after '+++ b/' or '--- a/'
            # So for '.git/test.py', the filepath would be '.git/test.py'
            # And it should start with '.git/'
            assert not result.is_valid, f"Should reject patch in {forbidden_dir}"
            assert any(forbidden_dir in v for v in result.violations)

    def test_forbidden_file_patterns(self):
        """Test that patches touching forbidden file patterns are rejected."""
        forbidden_files = [".env", "secrets.yml", "id_rsa"]
        for forbidden_file in forbidden_files:
            diff = f"""--- a/{forbidden_file}
+++ b/{forbidden_file}
@@ -0,0 +1 @@
+SECRET_KEY=value
"""
            result = validate_patch_hygiene(diff)
            assert not result.is_valid, f"Should reject patch to {forbidden_file}"

    def test_test_deletion(self):
        """Test that test file deletion is rejected."""
        diff = """--- a/test_file.py
+++ /dev/null
@@ -1,1 +0,0 @@
-def test_func():
-    pass
"""
        result = validate_patch_hygiene(diff)
        assert not result.is_valid
        assert any("Cannot delete test file" in v for v in result.violations)

    def test_skip_patterns(self):
        """Test that patches with skip patterns are rejected."""
        skip_patterns = ["@pytest.mark.skip", "@unittest.skip", "@pytest.mark.xfail"]
        for pattern in skip_patterns:
            diff = f"""--- a/test_file.py
+++ b/test_file.py
@@ -1,3 +1,4 @@
 def test_func():
     assert True
+    {pattern}(reason="temporarily disabled")
     pass
"""
            result = validate_patch_hygiene(diff)
            assert not result.is_valid, f"Should reject patch with {pattern}"

    def test_debug_patterns(self):
        """Test that patches with debug patterns are rejected."""
        debug_patterns = ['print("debug"', "pdb.set_trace", "breakpoint()"]
        for pattern in debug_patterns:
            diff = f"""--- a/test.py
+++ b/test.py
@@ -1,3 +1,4 @@
 def test_func():
+    {pattern}
     assert True
"""
            result = validate_patch_hygiene(diff)
            assert not result.is_valid, f"Should reject patch with {pattern}"


class TestStallDetector(unittest.TestCase):
    """Test stall detection logic."""

    def test_stall_state_initialization(self):
        """Test that stall state initializes correctly."""
        stall_state = StallState(stall_threshold=3)
        assert stall_state.stall_threshold == 3
        assert stall_state.iterations_without_improvement == 0
        assert stall_state.failing_tests_count == 0
        assert stall_state.failing_test_id is None
        assert stall_state.error_signature == ""

    def test_stall_detection_no_improvement(self):
        """Test that stall is detected after N iterations without improvement."""
        stall_state = StallState(stall_threshold=3)

        # First iteration (improvement from 0 to 5)
        is_stalled = stall_state.update(failing_count=5, test_id="test_1", sig="sig1")
        assert not is_stalled
        assert stall_state.iterations_without_improvement == 0

        # Second iteration (no improvement)
        is_stalled = stall_state.update(failing_count=5, test_id="test_1", sig="sig1")
        assert not is_stalled
        assert stall_state.iterations_without_improvement == 1

        # Third iteration (no improvement)
        is_stalled = stall_state.update(failing_count=5, test_id="test_1", sig="sig1")
        assert not is_stalled
        assert stall_state.iterations_without_improvement == 2

        # Fourth iteration (no improvement - should stall)
        is_stalled = stall_state.update(failing_count=5, test_id="test_1", sig="sig1")
        assert is_stalled
        assert stall_state.iterations_without_improvement == 3

    def test_stall_detection_with_improvement(self):
        """Test that stall is not detected when there's improvement."""
        stall_state = StallState(stall_threshold=3)

        # First iteration
        stall_state.update(failing_count=5, test_id="test_1", sig="sig1")

        # Second iteration (improved - fewer failing tests)
        is_stalled = stall_state.update(failing_count=3, test_id="test_1", sig="sig1")
        assert not is_stalled
        assert stall_state.iterations_without_improvement == 0
        assert stall_state.failing_tests_count == 3

    def test_stall_score(self):
        """Test that stall score is computed correctly."""
        stall_state = StallState()

        stall_state.update(failing_count=5, test_id="test_1", sig="sig1")
        score = stall_state.get_score()
        assert score == (5, True)  # (failing_count, has_error_signature)

        stall_state.update(failing_count=3, test_id="test_1", sig="sig1")
        score = stall_state.get_score()
        assert score == (3, True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
