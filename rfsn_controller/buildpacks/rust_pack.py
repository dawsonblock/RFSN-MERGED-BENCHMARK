"""Rust buildpack implementation.
from __future__ import annotations

Handles Rust repositories with Cargo.
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


class RustBuildpack(Buildpack):
    """Buildpack for Rust repositories."""

    def __init__(self):
        """Initialize the Rust buildpack."""
        super().__init__()
        self._buildpack_type = BuildpackType.RUST

    def detect(self, ctx: BuildpackContext) -> DetectResult | None:
        """Detect if this is a Rust repository."""
        rust_indicators = ["Cargo.toml", "Cargo.lock", "rust-toolchain", "rust-toolchain.toml"]

        found_indicators = []
        for indicator in rust_indicators:
            if indicator in ctx.files or any(f == indicator or f.endswith("/" + indicator) for f in ctx.repo_tree):
                found_indicators.append(indicator)

        if not found_indicators:
            return None

        confidence = 0.7
        if "Cargo.toml" in found_indicators:
            confidence += 0.2
        if "Cargo.lock" in found_indicators:
            confidence += 0.1

        return DetectResult(
            buildpack_type=BuildpackType.RUST,
            confidence=min(confidence, 1.0),
            metadata={"indicators": found_indicators},
        )

    def image(self) -> str:
        """Return the Docker image for Rust."""
        return "rust:1.76-slim"

    def sysdeps_whitelist(self) -> list[str]:
        """Return Rust-specific system dependencies."""
        common = ["build-essential", "pkg-config", "git", "ca-certificates"]
        rust_extras = ["libssl-dev", "cmake"]
        return common + rust_extras

    def install_plan(self, ctx: BuildpackContext) -> list[Step]:
        """Generate Rust installation steps."""
        steps = []

        steps.append(Step(
            argv=["cargo", "fetch"],
            description="Fetch Rust dependencies",
            timeout_sec=300,
            network_required=True,
        ))

        steps.append(Step(
            argv=["cargo", "build"],
            description="Build Rust project",
            timeout_sec=600,
            network_required=False,
        ))

        return steps

    def test_plan(self, ctx: BuildpackContext, focus_file: str | None = None) -> TestPlan:
        """Generate Rust test execution plan."""
        if focus_file:
            argv = ["cargo", "test", "--", focus_file]
        else:
            argv = ["cargo", "test"]

        return TestPlan(
            argv=argv,
            description="Run cargo test",
            timeout_sec=300,
            network_required=False,
            focus_file=focus_file,
        )

    def parse_failures(self, stdout: str, stderr: str) -> FailureInfo:
        """Parse Rust test output for failures."""
        failing_tests = []
        likely_files = []
        error_type = None
        error_message = None

        output = stdout + "\n" + stderr

        # Parse cargo test failures
        fail_pattern = r"test\s+(\S+)\s+\.\.\.\s+FAILED"
        for match in re.finditer(fail_pattern, output):
            test_name = match.group(1)
            failing_tests.append(test_name)

        # Parse file references
        file_pattern = r"-->\s+(\S+\.rs):(\d+):"
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
            argv=["cargo", "test", test_name],
            description=f"Focus test on {test_name}",
            timeout_sec=300,
            network_required=False,
            focus_file=None,
        )
