"""Node.js buildpack implementation.
from __future__ import annotations

Handles Node.js repositories with npm, yarn, or pnpm dependency managers.
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


class NodeBuildpack(Buildpack):
    """Buildpack for Node.js repositories."""

    def __init__(self):
        """Initialize the Node.js buildpack."""
        super().__init__()
        self._buildpack_type = BuildpackType.NODE

    def detect(self, ctx: BuildpackContext) -> DetectResult | None:
        """Detect if this is a Node.js repository."""
        node_indicators = [
            "package.json",
            "package-lock.json",
            "yarn.lock",
            "pnpm-lock.yaml",
            "bun.lockb",
            ".nvmrc",
            ".node-version",
        ]

        found_indicators = []
        for indicator in node_indicators:
            if indicator in ctx.files or any(f == indicator or f.endswith("/" + indicator) for f in ctx.repo_tree):
                found_indicators.append(indicator)

        if not found_indicators:
            return None

        confidence = 0.6
        if "package.json" in found_indicators:
            confidence += 0.3
        if "package-lock.json" in found_indicators or "yarn.lock" in found_indicators:
            confidence += 0.1

        return DetectResult(
            buildpack_type=BuildpackType.NODE,
            confidence=min(confidence, 1.0),
            metadata={"indicators": found_indicators},
        )

    def image(self) -> str:
        """Return the Docker image for Node.js."""
        return "node:20-slim"

    def sysdeps_whitelist(self) -> list[str]:
        """Return Node.js-specific system dependencies."""
        common = ["build-essential", "pkg-config", "git", "ca-certificates"]
        node_extras = ["python3", "libssl-dev"]
        return common + node_extras

    def install_plan(self, ctx: BuildpackContext) -> list[Step]:
        """Generate Node.js installation steps."""
        steps = []

        # Detect package manager
        if "pnpm-lock.yaml" in ctx.files:
            steps.append(Step(
                argv=["npm", "install", "-g", "pnpm"],
                description="Install pnpm",
                timeout_sec=120,
                network_required=True,
            ))
            steps.append(Step(
                argv=["pnpm", "install"],
                description="Install dependencies with pnpm",
                timeout_sec=300,
                network_required=True,
            ))
        elif "yarn.lock" in ctx.files:
            steps.append(Step(
                argv=["npm", "install", "-g", "yarn"],
                description="Install yarn",
                timeout_sec=120,
                network_required=True,
            ))
            steps.append(Step(
                argv=["yarn", "install"],
                description="Install dependencies with yarn",
                timeout_sec=300,
                network_required=True,
            ))
        else:
            steps.append(Step(
                argv=["npm", "install"],
                description="Install dependencies with npm",
                timeout_sec=300,
                network_required=True,
            ))

        return steps

    def test_plan(self, ctx: BuildpackContext, focus_file: str | None = None) -> TestPlan:
        """Generate Node.js test execution plan."""
        ctx.files.get("package.json", "")
        
        if focus_file:
            argv = ["npm", "test", "--", focus_file]
        else:
            argv = ["npm", "test"]

        return TestPlan(
            argv=argv,
            description="Run npm test",
            timeout_sec=120,
            network_required=False,
            focus_file=focus_file,
        )

    def parse_failures(self, stdout: str, stderr: str) -> FailureInfo:
        """Parse Node.js test output for failures."""
        failing_tests = []
        likely_files = []
        error_type = None
        error_message = None

        output = stdout + "\n" + stderr

        # Parse Jest failures
        jest_pattern = r"FAIL\s+(.+\.(?:js|ts|tsx))"
        for match in re.finditer(jest_pattern, output):
            file_path = match.group(1)
            failing_tests.append(file_path)
            likely_files.append(file_path)

        # Parse Mocha failures
        mocha_pattern = r"failing\s+\d+.*?\n\s+\d+\)\s+(.+)"
        for match in re.finditer(mocha_pattern, output, re.DOTALL):
            test_name = match.group(1).strip()
            failing_tests.append(test_name)

        # Parse error type
        error_pattern = r"(TypeError|ReferenceError|SyntaxError|Error):"
        error_match = re.search(error_pattern, output)
        if error_match:
            error_type = error_match.group(1)

        signature_input = "\n".join(failing_tests) + "\n" + (error_type or "")
        signature = hashlib.sha256(signature_input.encode()).hexdigest()[:16]

        return FailureInfo(
            failing_tests=failing_tests,
            likely_files=list(set(likely_files)),
            signature=signature,
            error_type=error_type,
            error_message=error_message,
        )

    def focus_plan(self, failure: FailureInfo) -> TestPlan | None:
        """Generate focused test plan based on failure."""
        if not failure.likely_files:
            return None

        focus_file = failure.likely_files[0]
        return TestPlan(
            argv=["npm", "test", "--", focus_file],
            description=f"Focus test on {focus_file}",
            timeout_sec=120,
            network_required=False,
            focus_file=focus_file,
        )
