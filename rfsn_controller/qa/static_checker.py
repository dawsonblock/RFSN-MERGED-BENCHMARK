"""Static checker wrapper for QA evidence.
from __future__ import annotations

Wraps mypy, ruff, and other static analysis tools for use as evidence.
"""

import json
import logging
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StaticCheckResult:
    """Result from running a static checker."""

    tool: str
    exit_code: int
    issues: list[dict[str, Any]]
    raw_output: str = ""

    @property
    def passed(self) -> bool:
        return self.exit_code == 0

    @property
    def issue_count(self) -> int:
        return len(self.issues)


class StaticChecker:
    """Wrapper for static analysis tools."""

    SUPPORTED_TOOLS = {"ruff", "mypy", "flake8", "pylint"}

    def __init__(
        self,
        *,
        timeout_seconds: int = 30,
        cwd: str | None = None,
    ):
        """Initialize checker.
        
        Args:
            timeout_seconds: Timeout for tool execution.
            cwd: Working directory for tool execution.
        """
        self.timeout_seconds = timeout_seconds
        self.cwd = cwd

    def check(
        self,
        tool: str,
        *,
        files: list[str] | None = None,
        extra_args: list[str] | None = None,
    ) -> StaticCheckResult:
        """Run a static checker.
        
        Args:
            tool: Tool name (ruff, mypy, flake8, pylint).
            files: Specific files to check (default: current dir).
            extra_args: Additional arguments.
        
        Returns:
            StaticCheckResult with issues.
        """
        if tool not in self.SUPPORTED_TOOLS:
            logger.warning("Unsupported tool: %s", tool)
            return StaticCheckResult(tool=tool, exit_code=1, issues=[])

        try:
            if tool == "ruff":
                return self._run_ruff(files, extra_args)
            elif tool == "mypy":
                return self._run_mypy(files, extra_args)
            elif tool == "flake8":
                return self._run_flake8(files, extra_args)
            else:
                return self._run_generic(tool, files, extra_args)
        except Exception as e:
            logger.warning("Static check failed: %s", e)
            return StaticCheckResult(tool=tool, exit_code=1, issues=[], raw_output=str(e))

    def _run_ruff(
        self,
        files: list[str] | None,
        extra_args: list[str] | None,
    ) -> StaticCheckResult:
        """Run ruff with JSON output."""
        cmd = ["ruff", "check", "--output-format=json"]
        if extra_args:
            cmd.extend(extra_args)
        cmd.extend(files or ["."])

        result = self._execute(cmd)

        issues = []
        if result["stdout"]:
            try:
                issues = json.loads(result["stdout"])
            except json.JSONDecodeError:
                # Parse line-by-line if not JSON
                issues = self._parse_line_output(result["stdout"])

        return StaticCheckResult(
            tool="ruff",
            exit_code=result["exit_code"],
            issues=issues,
            raw_output=result["stdout"],
        )

    def _run_mypy(
        self,
        files: list[str] | None,
        extra_args: list[str] | None,
    ) -> StaticCheckResult:
        """Run mypy."""
        cmd = ["mypy", "--no-error-summary"]
        if extra_args:
            cmd.extend(extra_args)
        cmd.extend(files or ["."])

        result = self._execute(cmd)
        issues = self._parse_mypy_output(result["stdout"])

        return StaticCheckResult(
            tool="mypy",
            exit_code=result["exit_code"],
            issues=issues,
            raw_output=result["stdout"],
        )

    def _run_flake8(
        self,
        files: list[str] | None,
        extra_args: list[str] | None,
    ) -> StaticCheckResult:
        """Run flake8."""
        cmd = ["flake8"]
        if extra_args:
            cmd.extend(extra_args)
        cmd.extend(files or ["."])

        result = self._execute(cmd)
        issues = self._parse_line_output(result["stdout"])

        return StaticCheckResult(
            tool="flake8",
            exit_code=result["exit_code"],
            issues=issues,
            raw_output=result["stdout"],
        )

    def _run_generic(
        self,
        tool: str,
        files: list[str] | None,
        extra_args: list[str] | None,
    ) -> StaticCheckResult:
        """Run a generic tool."""
        cmd = [tool]
        if extra_args:
            cmd.extend(extra_args)
        cmd.extend(files or ["."])

        result = self._execute(cmd)
        issues = self._parse_line_output(result["stdout"])

        return StaticCheckResult(
            tool=tool,
            exit_code=result["exit_code"],
            issues=issues,
            raw_output=result["stdout"],
        )

    def _execute(self, cmd: list[str]) -> dict[str, Any]:
        """Execute a command and return result."""
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=self.cwd, check=False,
            )
            return {
                "exit_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        except subprocess.TimeoutExpired:
            return {"exit_code": 124, "stdout": "", "stderr": "Timeout"}
        except FileNotFoundError:
            return {"exit_code": 127, "stdout": "", "stderr": f"Tool not found: {cmd[0]}"}

    def _parse_line_output(self, output: str) -> list[dict[str, Any]]:
        """Parse line-based output into issues."""
        issues = []
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            # Try to parse file:line:col format
            if ":" in line:
                parts = line.split(":", 3)
                if len(parts) >= 3:
                    issues.append({
                        "file": parts[0],
                        "line": int(parts[1]) if parts[1].isdigit() else 0,
                        "message": parts[-1].strip(),
                    })
                else:
                    issues.append({"message": line})
            else:
                issues.append({"message": line})
        return issues

    def _parse_mypy_output(self, output: str) -> list[dict[str, Any]]:
        """Parse mypy output into issues."""
        issues = []
        for line in output.splitlines():
            line = line.strip()
            if not line or line.startswith("Found"):
                continue
            # mypy format: file.py:line: error: message
            if ": error:" in line or ": warning:" in line:
                parts = line.split(":", 3)
                if len(parts) >= 4:
                    issues.append({
                        "file": parts[0],
                        "line": int(parts[1]) if parts[1].isdigit() else 0,
                        "severity": "error" if "error" in parts[2] else "warning",
                        "message": parts[3].strip(),
                    })
        return issues

    def check_all(
        self,
        tools: list[str] | None = None,
        *,
        files: list[str] | None = None,
        parallel: bool = True,
    ) -> dict[str, StaticCheckResult]:
        """Run multiple static checkers.
        
        Args:
            tools: List of tools to run (default: ruff only).
            files: Files to check.
            parallel: Run checks in parallel (default: True).
        
        Returns:
            Dict of tool -> result.
        """
        tools = tools or ["ruff"]

        if parallel and len(tools) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            results = {}
            with ThreadPoolExecutor(max_workers=min(len(tools), 4)) as executor:
                future_to_tool = {
                    executor.submit(self.check, tool, files=files): tool
                    for tool in tools
                }
                for future in as_completed(future_to_tool):
                    tool = future_to_tool[future]
                    try:
                        results[tool] = future.result()
                    except Exception as e:
                        results[tool] = StaticCheckResult(
                            tool=tool, exit_code=1, issues=[], raw_output=str(e)
                        )
            return results
        else:
            return {tool: self.check(tool, files=files) for tool in tools}


def create_static_checker(**kwargs) -> StaticChecker:
    """Factory function for StaticChecker."""
    return StaticChecker(**kwargs)


def static_checker_evidence_fn(tool: str = "ruff") -> Callable[[str], dict[str, Any]]:
    """Create evidence function for EvidenceCollector.
    
    Args:
        tool: Static checker to use.
    
    Returns:
        Function suitable for EvidenceCollector.static_checker.
    """
    checker = StaticChecker()

    def evidence_fn(requested_tool: str) -> dict[str, Any]:
        result = checker.check(requested_tool or tool)
        return {
            "exit_code": result.exit_code,
            "issues": result.issues,
            "tool": result.tool,
        }

    return evidence_fn
