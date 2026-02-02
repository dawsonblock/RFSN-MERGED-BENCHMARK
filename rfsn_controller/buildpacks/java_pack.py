"""Java buildpack implementation.
from __future__ import annotations

Handles Java repositories with Maven or Gradle.
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


class JavaBuildpack(Buildpack):
    """Buildpack for Java repositories."""

    def __init__(self):
        """Initialize the Java buildpack."""
        super().__init__()
        self._buildpack_type = BuildpackType.JAVA

    def detect(self, ctx: BuildpackContext) -> DetectResult | None:
        """Detect if this is a Java repository."""
        java_indicators = [
            "pom.xml",
            "build.gradle",
            "build.gradle.kts",
            "gradlew",
            "mvnw",
            "settings.gradle",
            "settings.gradle.kts",
        ]

        found_indicators = []
        for indicator in java_indicators:
            if indicator in ctx.files or any(f == indicator or f.endswith("/" + indicator) for f in ctx.repo_tree):
                found_indicators.append(indicator)

        if not found_indicators:
            return None

        confidence = 0.7
        if "pom.xml" in found_indicators:
            confidence += 0.2
        if "build.gradle" in found_indicators or "build.gradle.kts" in found_indicators:
            confidence += 0.2

        return DetectResult(
            buildpack_type=BuildpackType.JAVA,
            confidence=min(confidence, 1.0),
            metadata={"indicators": found_indicators},
        )

    def image(self) -> str:
        """Return the Docker image for Java."""
        return "eclipse-temurin:21-jdk"

    def sysdeps_whitelist(self) -> list[str]:
        """Return Java-specific system dependencies."""
        common = ["build-essential", "pkg-config", "git", "ca-certificates"]
        return common

    def install_plan(self, ctx: BuildpackContext) -> list[Step]:
        """Generate Java installation steps."""
        steps = []

        # Check for Gradle
        if "build.gradle" in ctx.files or "build.gradle.kts" in ctx.files:
            if "gradlew" in ctx.files:
                steps.append(Step(
                    argv=["./gradlew", "build", "-x", "test"],
                    description="Build with Gradle wrapper",
                    timeout_sec=600,
                    network_required=True,
                ))
            else:
                steps.append(Step(
                    argv=["gradle", "build", "-x", "test"],
                    description="Build with Gradle",
                    timeout_sec=600,
                    network_required=True,
                ))
        # Check for Maven
        elif "pom.xml" in ctx.files:
            if "mvnw" in ctx.files:
                steps.append(Step(
                    argv=["./mvnw", "compile", "-DskipTests"],
                    description="Compile with Maven wrapper",
                    timeout_sec=600,
                    network_required=True,
                ))
            else:
                steps.append(Step(
                    argv=["mvn", "compile", "-DskipTests"],
                    description="Compile with Maven",
                    timeout_sec=600,
                    network_required=True,
                ))

        return steps

    def test_plan(self, ctx: BuildpackContext, focus_file: str | None = None) -> TestPlan:
        """Generate Java test execution plan."""
        # Check for Gradle
        if "build.gradle" in ctx.files or "build.gradle.kts" in ctx.files:
            if "gradlew" in ctx.files:
                argv = ["./gradlew", "test"]
            else:
                argv = ["gradle", "test"]
        # Check for Maven
        elif "pom.xml" in ctx.files:
            if "mvnw" in ctx.files:
                argv = ["./mvnw", "test"]
            else:
                argv = ["mvn", "test"]
        else:
            argv = ["mvn", "test"]

        return TestPlan(
            argv=argv,
            description="Run Java tests",
            timeout_sec=300,
            network_required=False,
            focus_file=focus_file,
        )

    def parse_failures(self, stdout: str, stderr: str) -> FailureInfo:
        """Parse Java test output for failures."""
        failing_tests = []
        likely_files = []
        error_type = None
        error_message = None

        output = stdout + "\n" + stderr

        # Parse JUnit failures
        junit_pattern = r"(FAILURE|ERROR).*?(\S+Test\S*)"
        for match in re.finditer(junit_pattern, output):
            test_name = match.group(2)
            failing_tests.append(test_name)

        # Parse file references
        file_pattern = r"at\s+(\S+)\((\S+\.java):(\d+)\)"
        for match in re.finditer(file_pattern, output):
            file_path = match.group(2)
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
            argv=["mvn", "test", f"-Dtest={test_name}"],
            description=f"Focus test on {test_name}",
            timeout_sec=300,
            network_required=False,
            focus_file=None,
        )
