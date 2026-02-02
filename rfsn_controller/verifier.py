"""Test runner integration for the RFSN controller.
from __future__ import annotations

Implements multi-predicate goal verification:
- Tests (required)
- Lint (optional)
- Typecheck (optional)
- Repro (optional)
"""

import os
from dataclasses import dataclass, field

from .parsers import error_signature, parse_pytest_failures
from .sandbox import Sandbox, docker_test, run_cmd


@dataclass
class VerifyResult:
    """Wrapper for test run results."""

    ok: bool
    exit_code: int
    stdout: str
    stderr: str
    failing_tests: list[str] = field(default_factory=list)
    sig: str = ""
    predicate_name: str = "tests"  # Name of the verification predicate
    skipped: bool = False  # Whether verification was skipped
    # Test delta tracking
    tests_fixed: list[str] = field(default_factory=list)  # fail → pass
    tests_regressed: list[str] = field(default_factory=list)  # pass → fail


@dataclass
class VerifySummary:
    """Summary of all verification checks."""

    all_passed: bool
    results: list[VerifyResult]
    total_checks: int
    passed_checks: int
    failed_checks: int


class TestDeltaTracker:
    """Track test state transitions across patch attempts.

    Used to detect regressions (previously passing tests now failing)
    and measure progress (previously failing tests now passing).
    """

    def __init__(self, baseline_failing: set):
        """Initialize with baseline failing test set.

        Args:
            baseline_failing: Set of test names that were failing before patch.
        """
        self.baseline_failing = set(baseline_failing)

    def compute_delta(self, current_failing: set) -> tuple:
        """Compute test state transitions.

        Args:
            current_failing: Set of test names failing after patch.

        Returns:
            Tuple of (tests_fixed, tests_regressed) as lists.
        """
        fixed = self.baseline_failing - current_failing
        regressed = current_failing - self.baseline_failing
        return list(fixed), list(regressed)

    def has_regressions(self, current_failing: set) -> bool:
        """Check if any previously-passing test now fails.

        Args:
            current_failing: Set of test names failing after patch.

        Returns:
            True if regressions detected.
        """
        return bool(current_failing - self.baseline_failing)

    def get_regression_count(self, current_failing: set) -> int:
        """Get count of regressed tests.

        Args:
            current_failing: Set of test names failing after patch.

        Returns:
            Number of tests that regressed.
        """
        return len(current_failing - self.baseline_failing)

    def get_fixed_count(self, current_failing: set) -> int:
        """Get count of fixed tests.

        Args:
            current_failing: Set of test names failing after patch.

        Returns:
            Number of tests that were fixed.
        """
        return len(self.baseline_failing - current_failing)


class Verifier:
    """Multi-predicate verifier for goal validation."""

    def __init__(
        self,
        sb: Sandbox,
        test_cmd: str,
        lint_cmd: str | None = None,
        typecheck_cmd: str | None = None,
        repro_cmd: str | None = None,
        verify_cmd: str | None = None,
        docker_image: str = "python:3.11-slim",
        cpu: float = 2.0,
        mem_mb: int = 4096,
        pids: int = 256,
        read_only: bool = False,
        use_docker: bool = False,
    ):
        """Initialize the verifier.

        Args:
            sb: The sandbox containing the repo.
            test_cmd: Test command (required).
            lint_cmd: Lint command (optional).
            typecheck_cmd: Typecheck command (optional).
            repro_cmd: Repro command (optional).
            verify_cmd: Smoke test command for feature verification (optional).
            docker_image: Docker image to use.
            cpu: CPU limit.
            mem_mb: Memory limit in MB.
            pids: Process ID limit.
            read_only: Whether to mount repo as read-only.
            use_docker: Whether to use Docker for verification.
        """
        self.sb = sb
        self.test_cmd = test_cmd
        self.lint_cmd = lint_cmd
        self.typecheck_cmd = typecheck_cmd
        self.repro_cmd = repro_cmd
        self.verify_cmd = verify_cmd
        self.docker_image = docker_image
        self.cpu = cpu
        self.mem_mb = mem_mb
        self.pids = pids
        self.read_only = read_only
        self.use_docker = use_docker

    def verify_all(
        self,
        timeout_sec: int = 120,
        focus_test_file: str | None = None,
    ) -> VerifySummary:
        """Run all enabled verification predicates in order (fast-to-slow).

        Verification order:
        1. Focus test (if provided)
        2. Full tests
        3. Verify/Smoke test (if enabled)
        4. Repro (if enabled)
        5. Lint (if enabled)
        6. Typecheck (if enabled)

        Args:
            timeout_sec: Timeout for each verification step.
            focus_test_file: Specific test file to run first.

        Returns:
            VerifySummary with all results.
        """
        results = []

        # 1. Focus test (fastest)
        if focus_test_file:
            focus_cmd = f"pytest -q {focus_test_file}"
            result = self._run_verify(focus_cmd, "focus_test", timeout_sec)
            results.append(result)
            if not result.ok:
                # Focus test failed, skip remaining checks
                return VerifySummary(
                    all_passed=False,
                    results=results,
                    total_checks=len(results),
                    passed_checks=sum(1 for r in results if r.ok),
                    failed_checks=sum(1 for r in results if not r.ok),
                )

        # 2. Full tests (required)
        result = self._run_verify(self.test_cmd, "tests", timeout_sec)
        results.append(result)
        if not result.ok:
            # Tests failed, skip optional checks
            return VerifySummary(
                all_passed=False,
                results=results,
                total_checks=len(results),
                passed_checks=sum(1 for r in results if r.ok),
                failed_checks=sum(1 for r in results if not r.ok),
            )

        # 3. Verify/Smoke test (optional)
        if self.verify_cmd:
            result = self._run_verify(self.verify_cmd, "verify", timeout_sec)
            results.append(result)
            if not result.ok:
                return VerifySummary(
                    all_passed=False,
                    results=results,
                    total_checks=len(results),
                    passed_checks=sum(1 for r in results if r.ok),
                    failed_checks=sum(1 for r in results if not r.ok),
                )

        # 4. Repro (optional)
        if self.repro_cmd:
            result = self._run_verify(self.repro_cmd, "repro", timeout_sec)
            results.append(result)
            if not result.ok:
                return VerifySummary(
                    all_passed=False,
                    results=results,
                    total_checks=len(results),
                    passed_checks=sum(1 for r in results if r.ok),
                    failed_checks=sum(1 for r in results if not r.ok),
                )

        # 5. Lint (optional)
        if self.lint_cmd:
            result = self._run_verify(self.lint_cmd, "lint", timeout_sec)
            results.append(result)
            if not result.ok:
                return VerifySummary(
                    all_passed=False,
                    results=results,
                    total_checks=len(results),
                    passed_checks=sum(1 for r in results if r.ok),
                    failed_checks=sum(1 for r in results if not r.ok),
                )

        # 6. Typecheck (optional)
        if self.typecheck_cmd:
            result = self._run_verify(self.typecheck_cmd, "typecheck", timeout_sec)
            results.append(result)
            if not result.ok:
                return VerifySummary(
                    all_passed=False,
                    results=results,
                    total_checks=len(results),
                    passed_checks=sum(1 for r in results if r.ok),
                    failed_checks=sum(1 for r in results if not r.ok),
                )

        # All checks passed
        return VerifySummary(
            all_passed=True,
            results=results,
            total_checks=len(results),
            passed_checks=len(results),
            failed_checks=0,
        )

    def _run_verify(
        self,
        cmd: str,
        predicate_name: str,
        timeout_sec: int,
    ) -> VerifyResult:
        """Run a single verification predicate.

        Args:
            cmd: Command to run.
            predicate_name: Name of the predicate.
            timeout_sec: Timeout for the command.

        Returns:
            VerifyResult with execution status.
        """
        if self.use_docker:
            docker_result = docker_test(
                self.sb,
                cmd,
                timeout_sec=timeout_sec,
                docker_image=self.docker_image,
                cpu=self.cpu,
                mem_mb=self.mem_mb,
                pids=self.pids,
                read_only=self.read_only,
            )
            stdout = docker_result.stdout
            stderr = docker_result.stderr
            exit_code = docker_result.exit_code
            ok = docker_result.ok
        else:
            r = run_cmd(self.sb, cmd, timeout_sec=timeout_sec)
            stdout = r.get("stdout") or ""
            stderr = r.get("stderr") or ""
            try:
                exit_code = int(r.get("exit_code") or 1)
            except (ValueError, TypeError):
                exit_code = 1
            ok = bool(r.get("ok"))

        # Parse test failures and signature for test predicates
        failing_tests = []
        sig = ""
        if predicate_name in ["tests", "focus_test"]:
            out = stdout + stderr
            failing_tests = parse_pytest_failures(out)
            sig = error_signature(stdout, stderr)

        # Validate verification integrity check (Fail-Closed)
        # If the verifier detected a security violation/blocking, we must abort, 
        # not just mark as failed test.
        if "Command blocked by security policy" in stderr or "Security Violation" in stderr:
             raise RuntimeError(f"Fail-Closed: Security violation detected during verification ({predicate_name}). Session aborted.")

        return VerifyResult(
            ok=ok,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            failing_tests=failing_tests,
            sig=sig,
            predicate_name=predicate_name,
        )


def run_tests(
    sb: Sandbox, test_cmd: str, timeout_sec: int = 120, allow_skip: bool = False
) -> VerifyResult:
    """Run a test command and return structured results.

    Args:
        sb: The sandbox in which to run the tests.
        test_cmd: Command line to invoke the tests (e.g. "pytest -q").
        timeout_sec: Maximum seconds to wait for the command.
        allow_skip: If True, return success if tests don't exist yet.

    Returns:
        A VerifyResult with status and details.
    """
    r = run_cmd(sb, test_cmd, timeout_sec=timeout_sec)
    out = (r.get("stdout") or "") + (r.get("stderr") or "")

    # Check if tests don't exist yet (common in feature mode)
    if allow_skip:
        no_tests_indicators = [
            "no tests ran",
            "no test",
            "collected 0 items",
            "cannot find",
            "does not exist",
        ]
        if any(indicator in out.lower() for indicator in no_tests_indicators):
            return VerifyResult(
                ok=True,
                exit_code=0,
                stdout=r.get("stdout") or "",
                stderr=r.get("stderr") or "",
                failing_tests=[],
                sig="",
                skipped=True,
            )

    fails = parse_pytest_failures(out)
    sig = error_signature(r.get("stdout") or "", r.get("stderr") or "")
    try:
        exit_code = int(r.get("exit_code") or 1)
    except (ValueError, TypeError):
        exit_code = 1
    return VerifyResult(
        ok=bool(r.get("ok")),
        exit_code=exit_code,
        stdout=r.get("stdout") or "",
        stderr=r.get("stderr") or "",
        failing_tests=fails,
        sig=sig,
        skipped=False,
    )


def get_default_python_commands(
    repo_dir: str,
) -> dict[str, str | None]:
    """Get default verification commands for Python projects.

    Args:
        repo_dir: Path to the repository.

    Returns:
        Dict with keys: test, lint, typecheck, repro.
    """
    commands = {
        "test": "pytest -q",
        "lint": None,
        "typecheck": None,
        "repro": None,
    }

    # Check for ruff (lint)
    if os.path.exists(os.path.join(repo_dir, "pyproject.toml")):
        try:
            with open(os.path.join(repo_dir, "pyproject.toml")) as f:
                content = f.read()
                if "ruff" in content.lower():
                    commands["lint"] = "ruff check ."
        except Exception:
            pass

    # Check for mypy (typecheck)
    if os.path.exists(os.path.join(repo_dir, "pyproject.toml")):
        try:
            with open(os.path.join(repo_dir, "pyproject.toml")) as f:
                content = f.read()
                if "mypy" in content.lower():
                    commands["typecheck"] = "mypy ."
        except Exception:
            pass

    # Default repro command (run tests twice to check for flakiness)
    commands["repro"] = "pytest -q --repeat=2" if commands["test"] else None

    return commands
