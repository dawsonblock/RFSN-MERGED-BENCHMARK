"""Python buildpack implementation.
from __future__ import annotations

Handles Python repositories with pip, poetry, or other dependency managers.
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


class PythonBuildpack(Buildpack):
    """Buildpack for Python repositories."""

    def __init__(self):
        """Initialize the Python buildpack."""
        super().__init__()
        self._buildpack_type = BuildpackType.PYTHON

    def detect(self, ctx: BuildpackContext) -> DetectResult | None:
        """Detect if this is a Python repository.

        Args:
            ctx: Buildpack context with repo information.

        Returns:
            DetectResult if Python detected, None otherwise.
        """
        # Check for Python indicator files
        python_indicators = [
            "pyproject.toml",
            "requirements.txt",
            "setup.py",
            "setup.cfg",
            "Pipfile",
            "poetry.lock",
            "requirements.lock",
            "tox.ini",
            "noxfile.py",
            "conftest.py",  # pytest configuration
            "pytest.ini",  # pytest configuration
            "py.typed",  # typed packages
        ]

        found_indicators = []
        for indicator in python_indicators:
            # Check exact filename match in files dict
            if indicator in ctx.files or any(f == indicator or f.endswith("/" + indicator) for f in ctx.repo_tree):
                found_indicators.append(indicator)

        if not found_indicators:
            return None

        # Calculate confidence based on indicators
        confidence = 0.6
        if "pyproject.toml" in found_indicators:
            confidence += 0.2
        if "requirements.txt" in found_indicators:
            confidence += 0.2
        if "setup.py" in found_indicators:
            confidence += 0.1
        if "conftest.py" in found_indicators:
            confidence += 0.1

        confidence = min(confidence, 1.0)

        return DetectResult(
            buildpack_type=BuildpackType.PYTHON,
            confidence=confidence,
            metadata={"indicators": found_indicators},
        )

    def image(self) -> str:
        """Return the Docker image for Python.

        Returns:
            Docker image tag (alpine-based for smaller size).
        """
        return "python:3.11-slim"

    def sysdeps_whitelist(self) -> list[str]:
        """Return Python-specific system dependencies.

        Returns:
            List of allowed system packages.
        """
        # Common core
        common = ["build-essential", "pkg-config", "git", "ca-certificates"]
        # Python extras for common wheels
        python_extras = [
            "libssl-dev",
            "libffi-dev",
            "zlib1g-dev",
            "libbz2-dev",
            "liblzma-dev",
            "libxml2-dev",
            "libxslt1-dev",
            "libjpeg-dev",
            "libpng-dev",
        ]
        return common + python_extras

    def install_plan(self, ctx: BuildpackContext) -> list[Step]:
        """Generate Python installation steps.

        Args:
            ctx: Buildpack context with repo information.

        Returns:
            List of installation steps.
        """
        steps = []

        # Upgrade pip, setuptools, wheel
        steps.append(
            Step(
                argv=["python", "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"],
                description="Upgrade pip, setuptools, wheel",
                timeout_sec=180,
                network_required=True,
            )
        )

        # Install pytest for testing
        steps.append(
            Step(
                argv=["python", "-m", "pip", "install", "pytest"],
                description="Install pytest",
                timeout_sec=120,
                network_required=True,
            )
        )

        # Check for pyproject.toml
        if "pyproject.toml" in ctx.files:
            # Try editable install first
            steps.append(
                Step(
                    argv=["python", "-m", "pip", "install", "-e", "."],
                    description="Install package in editable mode",
                    timeout_sec=300,
                    network_required=True,
                )
            )

        # Check for requirements.txt
        if "requirements.txt" in ctx.files:
            steps.append(
                Step(
                    argv=["python", "-m", "pip", "install", "-r", "requirements.txt"],
                    description="Install from requirements.txt",
                    timeout_sec=300,
                    network_required=True,
                )
            )

        # Check for Pipfile
        if "Pipfile" in ctx.files:
            steps.append(
                Step(
                    argv=["pip", "install", "pipenv"],
                    description="Install pipenv",
                    timeout_sec=120,
                    network_required=True,
                )
            )
            steps.append(
                Step(
                    argv=["pipenv", "install", "--system"],
                    description="Install dependencies from Pipfile",
                    timeout_sec=300,
                    network_required=True,
                )
            )

        # Check for poetry.lock
        if "poetry.lock" in ctx.files:
            steps.append(
                Step(
                    argv=["pip", "install", "poetry"],
                    description="Install poetry",
                    timeout_sec=120,
                    network_required=True,
                )
            )
            steps.append(
                Step(
                    argv=["poetry", "install", "--no-interaction"],
                    description="Install dependencies with poetry",
                    timeout_sec=300,
                    network_required=True,
                )
            )

        return steps

    def test_plan(self, ctx: BuildpackContext, focus_file: str | None = None) -> TestPlan:
        """Generate Python test execution plan.

        Args:
            ctx: Buildpack context with repo information.
            focus_file: Optional file to focus tests on.

        Returns:
            TestPlan with command and configuration.
        """
        # Check for pytest
        if (
            "pytest" in ctx.files.get("pyproject.toml", "")
            or "pytest" in ctx.files.get("setup.cfg", "")
            or "pytest.ini" in ctx.files
        ):
            if focus_file:
                argv = ["python", "-m", "pytest", "-q", focus_file]
            else:
                argv = ["python", "-m", "pytest", "-q"]
            return TestPlan(
                argv=argv,
                description="Run pytest",
                timeout_sec=120,
                network_required=False,
                focus_file=focus_file,
            )

        # Check for unittest
        if "unittest" in ctx.files.get("pyproject.toml", "") or "setup.py" in ctx.files:
            if focus_file:
                argv = ["python", "-m", "unittest", "-q", focus_file]
            else:
                argv = ["python", "-m", "unittest", "discover", "-q"]
            return TestPlan(
                argv=argv,
                description="Run unittest",
                timeout_sec=120,
                network_required=False,
                focus_file=focus_file,
            )

        # Default to pytest
        if focus_file:
            argv = ["python", "-m", "pytest", "-q", focus_file]
        else:
            argv = ["python", "-m", "pytest", "-q"]
        return TestPlan(
            argv=argv,
            description="Run pytest (default)",
            timeout_sec=120,
            network_required=False,
            focus_file=focus_file,
        )

    def parse_failures(self, stdout: str, stderr: str) -> FailureInfo:
        """Parse Python test output for failures.

        Args:
            stdout: Standard output from test execution.
            stderr: Standard error from test execution.

        Returns:
            FailureInfo with parsed failure details.
        """
        failing_tests = []
        likely_files = []
        error_type = None
        error_message = None

        output = stdout + "\n" + stderr

        # Parse pytest failures
        pytest_pattern = r"FAILED\s+([^\s]+)::([^\s]+)"
        for match in re.finditer(pytest_pattern, output):
            module = match.group(1)
            test_name = match.group(2)
            failing_tests.append(f"{module}::{test_name}")
            likely_files.append(module.replace(".", "/") + ".py")

        # Parse unittest failures
        unittest_pattern = r"FAIL:\s+([^\s]+)\s+\(([^\s]+)\)"
        for match in re.finditer(unittest_pattern, output):
            test_name = match.group(1)
            module = match.group(2)
            failing_tests.append(f"{module}.{test_name}")
            likely_files.append(module.replace(".", "/") + ".py")

        # Parse error type from traceback
        error_pattern = r"([A-Z][a-zA-Z]+Error):"
        error_match = re.search(error_pattern, output)
        if error_match:
            error_type = error_match.group(1)

        # Parse error message
        message_pattern = r"[A-Z][a-zA-Z]+Error:\s*(.+?)(?:\n|$)"
        message_match = re.search(message_pattern, output)
        if message_match:
            error_message = message_match.group(1).strip()

        # Generate signature
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
        """Generate focused test plan based on failure.

        Args:
            failure: Failure information from test execution.

        Returns:
            TestPlan for focused testing, or None if not possible.
        """
        if not failure.likely_files:
            return None

        # Focus on the first likely file
        focus_file = failure.likely_files[0]
        return TestPlan(
            argv=["python", "-m", "pytest", "-q", focus_file],
            description=f"Focus test on {focus_file}",
            timeout_sec=120,
            network_required=False,
            focus_file=focus_file,
        )

    def get_verification_goals(self, ctx: BuildpackContext) -> list[str]:
        """Get verification goals for Python.

        Args:
            ctx: Buildpack context with repo information.

        Returns:
            List of goal names.
        """
        goals = ["test"]

        # Check for linting tools
        pyproject = ctx.files.get("pyproject.toml", "")
        if "ruff" in pyproject or "flake8" in pyproject or "pylint" in pyproject:
            goals.append("lint")

        # Check for type checking
        if "mypy" in pyproject:
            goals.append("typecheck")

        return goals

    def get_services_required(self, ctx: BuildpackContext) -> list[str]:
        """Get required external services for Python.

        Args:
            ctx: Buildpack context with repo information.

        Returns:
            List of service names (postgres, redis, mysql, mongodb, etc.).
        """
        services = []

        # Check dependency files for service indicators
        requirements = ctx.files.get("requirements.txt", "").lower()
        pyproject = ctx.files.get("pyproject.toml", "").lower()
        setup_py = ctx.files.get("setup.py", "").lower()

        all_deps = requirements + pyproject + setup_py

        # PostgreSQL
        if any(
            dep in all_deps
            for dep in ["psycopg2", "sqlalchemy", "django.db.backends.postgresql", "asyncpg"]
        ):
            services.append("postgres")

        # Redis
        if any(dep in all_deps for dep in ["redis", "django-redis", "celery[redis]", "aioredis"]):
            services.append("redis")

        # MySQL
        if any(
            dep in all_deps
            for dep in ["pymysql", "mysqlclient", "django.db.backends.mysql", "aiomysql"]
        ):
            services.append("mysql")

        # MongoDB
        if any(dep in all_deps for dep in ["pymongo", "motor", "mongomock"]):
            services.append("mongodb")

        # Elasticsearch
        if any(dep in all_deps for dep in ["elasticsearch", "elasticsearch-dsl"]):
            services.append("elasticsearch")

        # RabbitMQ
        if any(dep in all_deps for dep in ["pika", "kombu", "celery"]):
            services.append("rabbitmq")

        return services
