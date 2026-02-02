"""Verification Manager for RFSN Controller.

Manages test execution and verification workflow. Extracted from controller.py
to reduce complexity and improve testability.

Responsibilities:
- Running tests in isolation
- Collecting test output
- Determining pass/fail status
- Tracking test execution metrics
- Managing verification retries
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .structured_logging import get_logger

logger = get_logger(__name__)


@dataclass
class VerificationResult:
    """Result of a verification run."""

    success: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    failing_tests: list[str] = field(default_factory=list)
    passing_tests: list[str] = field(default_factory=list)
    test_count: int = 0
    error_signature: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationConfig:
    """Configuration for verification."""

    test_command: list[str]
    timeout_seconds: int = 120
    max_retries: int = 3
    verify_multiple_times: int = 1  # Run verification N times
    working_directory: Path = field(default_factory=lambda: Path.cwd())


class VerificationManager:
    """
    Manages test execution and verification workflow.

    This manager handles all aspects of running tests to verify patches:
    - Executing test commands in isolation
    - Parsing test output to determine pass/fail
    - Tracking which tests failed
    - Managing timeouts and retries
    - Collecting metrics

    Example:
        >>> config = VerificationConfig(
        ...     test_command=["pytest", "-q"],
        ...     timeout_seconds=60
        ... )
        >>> manager = VerificationManager(config)
        >>> result = await manager.verify_patch(worktree_path)
        >>> print(f"Tests passed: {result.success}")
    """

    def __init__(self, config: VerificationConfig):
        """
        Initialize verification manager.

        Args:
            config: Verification configuration
        """
        self.config = config
        self._stats = {
            "total_verifications": 0,
            "successful_verifications": 0,
            "failed_verifications": 0,
            "timeout_count": 0,
            "retry_count": 0,
        }

    async def verify_patch(
        self,
        worktree_path: Path,
        additional_env: dict[str, str] | None = None,
    ) -> VerificationResult:
        """
        Run tests to verify a patch in the given worktree.

        Args:
            worktree_path: Path to git worktree with applied patch
            additional_env: Optional environment variables

        Returns:
            VerificationResult with test outcomes

        Example:
            >>> result = await manager.verify_patch(Path("/tmp/worktree-1"))
            >>> if result.success:
            ...     print("All tests passed!")
            >>> else:
            ...     print(f"Failed: {result.failing_tests}")
        """
        logger.info(
            "Starting patch verification",
            worktree=str(worktree_path),
            command=self.config.test_command,
        )

        self._stats["total_verifications"] += 1

        # Run verification (with optional retries)
        for attempt in range(self.config.max_retries):
            try:
                result = await self._run_tests_once(
                    worktree_path,
                    additional_env=additional_env,
                    attempt=attempt,
                )

                # If successful or definitively failed, return
                if result.success or result.exit_code != -1:
                    if result.success:
                        self._stats["successful_verifications"] += 1
                    else:
                        self._stats["failed_verifications"] += 1
                    return result

                # Timeout occurred, retry
                logger.warning(
                    "Verification timeout, retrying",
                    attempt=attempt + 1,
                    max_retries=self.config.max_retries,
                )
                self._stats["retry_count"] += 1

            except Exception as e:
                logger.error("Verification error", error=str(e), attempt=attempt + 1)
                if attempt == self.config.max_retries - 1:
                    # Last attempt, return failure
                    return VerificationResult(
                        success=False,
                        exit_code=-1,
                        stdout="",
                        stderr=f"Verification error: {e!s}",
                        duration_seconds=0.0,
                        error_signature=str(e),
                    )

        # All retries exhausted
        self._stats["timeout_count"] += 1
        return VerificationResult(
            success=False,
            exit_code=-1,
            stdout="",
            stderr="Verification timed out after all retries",
            duration_seconds=0.0,
        )

    async def _run_tests_once(
        self,
        worktree_path: Path,
        additional_env: dict[str, str] | None = None,
        attempt: int = 0,
    ) -> VerificationResult:
        """
        Run tests once and collect results.

        Args:
            worktree_path: Path to worktree
            additional_env: Optional environment variables
            attempt: Attempt number (for logging)

        Returns:
            VerificationResult
        """
        import time

        start_time = time.time()

        # Prepare environment
        env = dict(additional_env or {})
        env["PYTEST_CURRENT_TEST"] = ""  # Clear pytest state

        try:
            # Run test command
            proc = await asyncio.create_subprocess_exec(
                *self.config.test_command,
                cwd=str(worktree_path),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Wait with timeout
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.timeout_seconds,
            )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            exit_code = proc.returncode or 0

            duration = time.time() - start_time

            # Parse test output
            failing_tests = self._parse_failing_tests(stdout, stderr)
            passing_tests = self._parse_passing_tests(stdout, stderr)
            error_sig = self._extract_error_signature(stderr)

            success = exit_code == 0 and len(failing_tests) == 0

            logger.info(
                "Verification complete",
                success=success,
                exit_code=exit_code,
                duration_seconds=round(duration, 2),
                failing_count=len(failing_tests),
                passing_count=len(passing_tests),
            )

            return VerificationResult(
                success=success,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                failing_tests=failing_tests,
                passing_tests=passing_tests,
                test_count=len(failing_tests) + len(passing_tests),
                error_signature=error_sig,
                metadata={
                    "attempt": attempt,
                    "worktree": str(worktree_path),
                },
            )

        except TimeoutError:
            duration = time.time() - start_time
            logger.warning(
                "Verification timeout",
                timeout=self.config.timeout_seconds,
                duration=duration,
            )
            return VerificationResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Timeout after {self.config.timeout_seconds}s",
                duration_seconds=duration,
            )

    def _parse_failing_tests(self, stdout: str, stderr: str) -> list[str]:
        """
        Extract list of failing tests from output.

        Args:
            stdout: Standard output
            stderr: Standard error

        Returns:
            List of failing test names
        """
        failing = []
        combined = stdout + "\n" + stderr

        # Common patterns for test failures
        patterns = [
            "FAILED ",  # pytest
            "FAIL:",  # unittest
            "✗ ",  # modern test runners
            "❌ ",  # modern test runners
        ]

        for line in combined.split("\n"):
            for pattern in patterns:
                if pattern in line:
                    # Extract test name (simple heuristic)
                    parts = line.split(pattern)
                    if len(parts) > 1:
                        test_name = parts[1].split()[0].strip()
                        if test_name and test_name not in failing:
                            failing.append(test_name)

        return failing

    def _parse_passing_tests(self, stdout: str, stderr: str) -> list[str]:
        """
        Extract list of passing tests from output.

        Args:
            stdout: Standard output
            stderr: Standard error

        Returns:
            List of passing test names
        """
        passing = []
        combined = stdout + "\n" + stderr

        patterns = [
            "PASSED ",  # pytest
            "ok ",  # unittest
            "✓ ",  # modern test runners
            "✅ ",  # modern test runners
        ]

        for line in combined.split("\n"):
            for pattern in patterns:
                if pattern in line:
                    parts = line.split(pattern)
                    if len(parts) > 1:
                        test_name = parts[1].split()[0].strip()
                        if test_name and test_name not in passing:
                            passing.append(test_name)

        return passing

    def _extract_error_signature(self, stderr: str) -> str | None:
        """
        Extract a signature from error output for deduplication.

        Args:
            stderr: Standard error output

        Returns:
            Error signature or None
        """
        if not stderr:
            return None

        # Look for common error patterns
        error_lines = []
        for line in stderr.split("\n"):
            line_lower = line.lower()
            if any(
                keyword in line_lower
                for keyword in [
                    "error:",
                    "exception:",
                    "traceback",
                    "assert",
                    "failed",
                ]
            ):
                error_lines.append(line.strip())

        if error_lines:
            # Return first few error lines as signature
            return "\n".join(error_lines[:3])

        return None

    async def verify_multiple_times(
        self,
        worktree_path: Path,
        times: int | None = None,
    ) -> list[VerificationResult]:
        """
        Run verification multiple times for reliability testing.

        Args:
            worktree_path: Path to worktree
            times: Number of times to run (defaults to config value)

        Returns:
            List of verification results
        """
        times = times or self.config.verify_multiple_times
        results = []

        logger.info("Running multi-verification", times=times)

        for i in range(times):
            result = await self.verify_patch(worktree_path)
            results.append(result)

            if not result.success:
                logger.warning(
                    "Multi-verification failed",
                    run=i + 1,
                    total=times,
                )

        return results

    def get_stats(self) -> dict[str, Any]:
        """
        Get verification statistics.

        Returns:
            Dictionary of stats
        """
        total = self._stats["total_verifications"]
        success_rate = (
            self._stats["successful_verifications"] / total if total > 0 else 0.0
        )

        return {
            **self._stats,
            "success_rate": success_rate,
        }
