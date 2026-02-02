"""C++ buildpack implementation.
from __future__ import annotations

Handles C++ repositories with CMake or Make.
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


class CppBuildpack(Buildpack):
    """Buildpack for C++ repositories."""

    def __init__(self):
        """Initialize the C++ buildpack."""
        super().__init__()
        self._buildpack_type = BuildpackType.CPP

    def detect(self, ctx: BuildpackContext) -> DetectResult | None:
        """Detect if this is a C++ repository."""
        cpp_indicators = [
            "CMakeLists.txt",
            "Makefile",
            "configure.ac",
            "configure",
            "meson.build",
            "conanfile.txt",
            "vcpkg.json",
        ]

        found_indicators = []
        for indicator in cpp_indicators:
            if indicator in ctx.files or any(f == indicator or f.endswith("/" + indicator) for f in ctx.repo_tree):
                found_indicators.append(indicator)

        # Also check for .cpp or .hpp files
        has_cpp_files = any(f.endswith(".cpp") or f.endswith(".hpp") or f.endswith(".cc") for f in ctx.repo_tree)
        if has_cpp_files:
            found_indicators.append("*.cpp files")

        if not found_indicators:
            return None

        confidence = 0.6
        if "CMakeLists.txt" in found_indicators:
            confidence += 0.3
        if "Makefile" in found_indicators:
            confidence += 0.2

        return DetectResult(
            buildpack_type=BuildpackType.CPP,
            confidence=min(confidence, 1.0),
            metadata={"indicators": found_indicators},
        )

    def image(self) -> str:
        """Return the Docker image for C++."""
        return "gcc:13"

    def sysdeps_whitelist(self) -> list[str]:
        """Return C++-specific system dependencies."""
        common = ["build-essential", "pkg-config", "git", "ca-certificates"]
        cpp_extras = ["cmake", "make", "libssl-dev", "libboost-all-dev"]
        return common + cpp_extras

    def install_plan(self, ctx: BuildpackContext) -> list[Step]:
        """Generate C++ installation steps."""
        steps = []

        if "CMakeLists.txt" in ctx.files:
            steps.append(Step(
                argv=["cmake", "-B", "build", "-S", "."],
                description="Configure with CMake",
                timeout_sec=300,
                network_required=False,
            ))
            steps.append(Step(
                argv=["cmake", "--build", "build"],
                description="Build with CMake",
                timeout_sec=600,
                network_required=False,
            ))
        elif "Makefile" in ctx.files:
            steps.append(Step(
                argv=["make"],
                description="Build with Make",
                timeout_sec=600,
                network_required=False,
            ))

        return steps

    def test_plan(self, ctx: BuildpackContext, focus_file: str | None = None) -> TestPlan:
        """Generate C++ test execution plan."""
        if "CMakeLists.txt" in ctx.files:
            argv = ["ctest", "--test-dir", "build", "--output-on-failure"]
        else:
            argv = ["make", "test"]

        return TestPlan(
            argv=argv,
            description="Run C++ tests",
            timeout_sec=300,
            network_required=False,
            focus_file=focus_file,
        )

    def parse_failures(self, stdout: str, stderr: str) -> FailureInfo:
        """Parse C++ test output for failures."""
        failing_tests = []
        likely_files = []
        error_type = None
        error_message = None

        output = stdout + "\n" + stderr

        # Parse CTest failures
        ctest_pattern = r"(\d+).*?Failed"
        for match in re.finditer(ctest_pattern, output):
            test_num = match.group(1)
            failing_tests.append(f"Test {test_num}")

        # Parse file references
        file_pattern = r"(\S+\.(?:cpp|hpp|cc|h)):(\d+)"
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
        return None  # C++ test focusing is complex
