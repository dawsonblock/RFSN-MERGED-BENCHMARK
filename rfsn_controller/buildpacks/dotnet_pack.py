""".NET buildpack implementation.
from __future__ import annotations

Handles .NET repositories with dotnet CLI.
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


class DotnetBuildpack(Buildpack):
    """Buildpack for .NET repositories."""

    def __init__(self):
        """Initialize the .NET buildpack."""
        super().__init__()
        self._buildpack_type = BuildpackType.DOTNET

    def detect(self, ctx: BuildpackContext) -> DetectResult | None:
        """Detect if this is a .NET repository."""
        dotnet_indicators = [
            "*.csproj",
            "*.fsproj",
            "*.sln",
            "global.json",
            "nuget.config",
            "Directory.Build.props",
        ]

        found_indicators = []
        for indicator in dotnet_indicators:
            if indicator.startswith("*"):
                ext = indicator[1:]
                if any(f.endswith(ext) for f in ctx.repo_tree):
                    found_indicators.append(indicator)
            elif indicator in ctx.files or any(f == indicator or f.endswith("/" + indicator) for f in ctx.repo_tree):
                found_indicators.append(indicator)

        if not found_indicators:
            return None

        confidence = 0.7
        if "*.csproj" in found_indicators or "*.sln" in found_indicators:
            confidence += 0.2

        return DetectResult(
            buildpack_type=BuildpackType.DOTNET,
            confidence=min(confidence, 1.0),
            metadata={"indicators": found_indicators},
        )

    def image(self) -> str:
        """Return the Docker image for .NET."""
        return "mcr.microsoft.com/dotnet/sdk:8.0"

    def sysdeps_whitelist(self) -> list[str]:
        """Return .NET-specific system dependencies."""
        common = ["build-essential", "pkg-config", "git", "ca-certificates"]
        return common

    def install_plan(self, ctx: BuildpackContext) -> list[Step]:
        """Generate .NET installation steps."""
        steps = []

        steps.append(Step(
            argv=["dotnet", "restore"],
            description="Restore .NET dependencies",
            timeout_sec=300,
            network_required=True,
        ))

        steps.append(Step(
            argv=["dotnet", "build", "--no-restore"],
            description="Build .NET project",
            timeout_sec=300,
            network_required=False,
        ))

        return steps

    def test_plan(self, ctx: BuildpackContext, focus_file: str | None = None) -> TestPlan:
        """Generate .NET test execution plan."""
        argv = ["dotnet", "test", "--no-build"]

        return TestPlan(
            argv=argv,
            description="Run dotnet test",
            timeout_sec=300,
            network_required=False,
            focus_file=focus_file,
        )

    def parse_failures(self, stdout: str, stderr: str) -> FailureInfo:
        """Parse .NET test output for failures."""
        failing_tests = []
        likely_files = []
        error_type = None
        error_message = None

        output = stdout + "\n" + stderr

        # Parse dotnet test failures
        fail_pattern = r"Failed\s+(\S+)"
        for match in re.finditer(fail_pattern, output):
            test_name = match.group(1)
            failing_tests.append(test_name)

        # Parse file references
        file_pattern = r"(\S+\.cs):line\s+(\d+)"
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
            argv=["dotnet", "test", "--filter", test_name],
            description=f"Focus test on {test_name}",
            timeout_sec=300,
            network_required=False,
            focus_file=None,
        )
