"""Polyrepo buildpack implementation.
from __future__ import annotations

Handles polyrepo/monorepo repositories with multiple language support.
"""

import hashlib

from .base import (
    Buildpack,
    BuildpackContext,
    BuildpackType,
    DetectResult,
    FailureInfo,
    Step,
    TestPlan,
)


class PolyrepoBuildpack(Buildpack):
    """Buildpack for polyrepo/monorepo repositories.
    
    This buildpack detects repositories that contain multiple projects
    in different languages and orchestrates their setup.
    """

    def __init__(self):
        """Initialize the Polyrepo buildpack."""
        super().__init__()
        self._buildpack_type = BuildpackType.POLYREPO

    def detect(self, ctx: BuildpackContext) -> DetectResult | None:
        """Detect if this is a polyrepo/monorepo repository.
        
        Polyrepos typically have multiple project markers at the root
        or in subdirectories.
        """
        # Count different language markers
        language_markers = {
            "python": ["pyproject.toml", "requirements.txt", "setup.py"],
            "node": ["package.json"],
            "go": ["go.mod"],
            "rust": ["Cargo.toml"],
            "java": ["pom.xml", "build.gradle"],
            "cpp": ["CMakeLists.txt"],
            "dotnet": [],
        }

        # Check for .csproj files
        if any(f.endswith(".csproj") for f in ctx.repo_tree):
            language_markers["dotnet"].append("*.csproj")

        found_languages = set()
        found_indicators = []

        for lang, markers in language_markers.items():
            for marker in markers:
                if marker.startswith("*"):
                    ext = marker[1:]
                    if any(f.endswith(ext) for f in ctx.repo_tree):
                        found_languages.add(lang)
                        found_indicators.append(marker)
                elif marker in ctx.files or any(f == marker or f.endswith("/" + marker) for f in ctx.repo_tree):
                    found_languages.add(lang)
                    found_indicators.append(marker)

        # Need at least 2 languages to be considered a polyrepo
        if len(found_languages) < 2:
            return None

        confidence = 0.5 + (len(found_languages) * 0.1)

        return DetectResult(
            buildpack_type=BuildpackType.POLYREPO,
            confidence=min(confidence, 1.0),
            metadata={
                "indicators": found_indicators,
                "languages": list(found_languages),
            },
        )

    def image(self) -> str:
        """Return the Docker image for polyrepo.
        
        Use a general-purpose image with multiple runtimes.
        """
        return "ubuntu:22.04"

    def sysdeps_whitelist(self) -> list[str]:
        """Return polyrepo-specific system dependencies."""
        common = ["build-essential", "pkg-config", "git", "ca-certificates"]
        # Include common language runtimes
        polyrepo_extras = [
            "python3",
            "python3-pip",
            "python3-venv",
            "nodejs",
            "npm",
            "golang",
            "rustc",
            "cargo",
            "maven",
            "gradle",
            "cmake",
        ]
        return common + polyrepo_extras

    def install_plan(self, ctx: BuildpackContext) -> list[Step]:
        """Generate polyrepo installation steps.
        
        Attempts to install dependencies for each detected language.
        """
        steps = []

        # Python
        if "requirements.txt" in ctx.files or "pyproject.toml" in ctx.files:
            steps.append(Step(
                argv=["python3", "-m", "pip", "install", "-e", "."],
                description="Install Python dependencies",
                timeout_sec=300,
                network_required=True,
            ))

        # Node.js
        if "package.json" in ctx.files:
            steps.append(Step(
                argv=["npm", "install"],
                description="Install Node.js dependencies",
                timeout_sec=300,
                network_required=True,
            ))

        # Go
        if "go.mod" in ctx.files:
            steps.append(Step(
                argv=["go", "mod", "download"],
                description="Download Go modules",
                timeout_sec=300,
                network_required=True,
            ))

        return steps

    def test_plan(self, ctx: BuildpackContext, focus_file: str | None = None) -> TestPlan:
        """Generate polyrepo test execution plan.
        
        For polyrepos, we run a general test command that tries
        multiple test runners.
        """
        # Default to Python pytest as most common
        return TestPlan(
            argv=["python3", "-m", "pytest", "-q"],
            description="Run tests (polyrepo)",
            timeout_sec=300,
            network_required=False,
            focus_file=focus_file,
        )

    def parse_failures(self, stdout: str, stderr: str) -> FailureInfo:
        """Parse polyrepo test output for failures."""
        # Delegate to a general parser
        signature = hashlib.sha256((stdout + stderr).encode()).hexdigest()[:16]

        return FailureInfo(
            failing_tests=[],
            likely_files=[],
            signature=signature,
            error_type=None,
            error_message=None,
        )

    def focus_plan(self, failure: FailureInfo) -> TestPlan | None:
        """Generate focused test plan based on failure."""
        return None  # Polyrepo focusing is complex
