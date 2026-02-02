"""Go buildpack implementation.
from __future__ import annotations

Handles Go repositories with go modules.
"""

import hashlib
import re

from .base import (
    Buildpack,
    BuildpackContext,
    BuildpackType,
    DetectResult,
    FailureInfo,
    Step,
    TestPlan,
)


class GoBuildpack(Buildpack):
    """Buildpack for Go repositories."""

    def __init__(self):
        """Initialize the Go buildpack."""
        super().__init__()
        self._buildpack_type = BuildpackType.GO

    def detect(self, ctx: BuildpackContext) -> DetectResult | None:
        """Detect if this is a Go repository."""
        go_indicators = ["go.mod", "go.sum", "Gopkg.toml", "Gopkg.lock"]

        found_indicators = []
        for indicator in go_indicators:
            if indicator in ctx.files or any(f == indicator or f.endswith("/" + indicator) for f in ctx.repo_tree):
                found_indicators.append(indicator)

        if not found_indicators:
            return None

        confidence = 0.7
        if "go.mod" in found_indicators:
            confidence += 0.2
        if "go.sum" in found_indicators:
            confidence += 0.1

        return DetectResult(
            buildpack_type=BuildpackType.GO,
            confidence=min(confidence, 1.0),
            metadata={"indicators": found_indicators},
        )

    def image(self) -> str:
        """Return the Docker image for Go."""
        return "golang:1.22-alpine"

    def sysdeps_whitelist(self) -> list[str]:
        """Return Go-specific system dependencies."""
        common = ["build-essential", "pkg-config", "git", "ca-certificates"]
        go_extras = ["gcc", "musl-dev"]
        return common + go_extras

    def install_plan(self, ctx: BuildpackContext) -> list[Step]:
        """Generate Go installation steps."""
        steps = []

        if "go.mod" in ctx.files:
            steps.append(Step(
                argv=["go", "mod", "download"],
                description="Download Go modules",
                timeout_sec=300,
                network_required=True,
            ))

        steps.append(Step(
            argv=["go", "build", "./..."],
            description="Build Go packages",
            timeout_sec=300,
            network_required=False,
        ))

        return steps

    def test_plan(self, ctx: BuildpackContext, focus_file: str | None = None) -> TestPlan:
        """Generate Go test execution plan."""
        if focus_file:
            argv = ["go", "test", "-v", focus_file]
        else:
            argv = ["go", "test", "-v", "./..."]

        return TestPlan(
            argv=argv,
            description="Run go test",
            timeout_sec=120,
            network_required=False,
            focus_file=focus_file,
        )

    def parse_failures(self, stdout: str, stderr: str) -> FailureInfo:
        """Parse Go test output for failures."""
        failing_tests = []
        likely_files = []
        error_type = None
        error_message = None

        output = stdout + "\n" + stderr

        # Parse go test failures
        fail_pattern = r"--- FAIL:\s+(\S+)"
        for match in re.finditer(fail_pattern, output):
            test_name = match.group(1)
            failing_tests.append(test_name)

        # Parse file references
        file_pattern = r"(\S+\.go):(\d+):"
        for match in re.finditer(file_pattern, output):
            file_path = match.group(1)
            if file_path not in likely_files:
                likely_files.append(file_path)

        signature_input = "\n".join(failing_tests) + "\n" + (error_type or "")
        signature = hashlib.sha256(signature_input.encode()).hexdigest()[:16]

        return FailureInfo(
            failing_tests=failing_tests,
            likely_files=likely_files,
            signature=signature,
            error_type=error_type,
            error_message=error_message,
        )

    def focus_plan(self, failure: FailureInfo) -> TestPlan | None:
        """Generate focused test plan based on failure."""
        if not failure.failing_tests:
            return None

        test_name = failure.failing_tests[0]
        return TestPlan(
            argv=["go", "test", "-v", "-run", test_name, "./..."],
            description=f"Focus test on {test_name}",
            timeout_sec=120,
            network_required=False,
            focus_file=None,
        )
