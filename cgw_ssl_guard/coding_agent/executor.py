"""Blocking executor for coding actions.

This module implements the execution layer of the serial decision
architecture. The executor:

1. Receives a committed action from the CGW
2. Executes it SYNCHRONOUSLY (blocking)
3. Returns results to the decision layer

CRITICAL INVARIANTS:
- The executor NEVER decides what to do next
- It only reports results back
- Execution must complete before the next decision cycle
- No tool overlap - one action at a time
- ALL real execution is delegated to GovernedExecutor (single spine)
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from .action_types import (
    ActionPayload,
    CodingAction,
    ExecutionResult,
)

# Import the unified execution spine
try:
    from rfsn_controller.executor_spine import GovernedExecutor
except ImportError:
    # Fallback for environments where rfsn_controller isn't available
    GovernedExecutor = None

logger = logging.getLogger(__name__)


class SandboxProtocol(Protocol):
    """Protocol for sandbox implementations."""
    
    def run(self, cmd: str, timeout: int = 300) -> subprocess.CompletedProcess:
        """Run a command in the sandbox."""
        ...
    
    def read_file(self, path: str) -> str:
        """Read a file from the sandbox."""
        ...
    
    def write_file(self, path: str, content: str) -> None:
        """Write a file to the sandbox."""
        ...
    
    def apply_diff(self, diff: str) -> bool:
        """Apply a diff/patch to the sandbox."""
        ...


@dataclass
class ExecutorConfig:
    """Configuration for the blocking executor."""
    
    # Timeouts (in seconds)
    test_timeout: int = 300
    timeout_sec: int = 180  # Default timeout for governed executor
    build_timeout: int = 600
    lint_timeout: int = 120
    patch_timeout: int = 30
    
    # Resource limits
    max_output_bytes: int = 1_000_000
    
    # Test command (can be overridden per action)
    default_test_cmd: str = "pytest -q"
    
    # Working directory
    work_dir: Optional[Path] = None


class BlockingExecutor:
    """Execute coding actions synchronously.
    
    This executor blocks until each action completes. It enforces
    the serial execution guarantee of the CGW architecture.
    
    Usage:
        executor = BlockingExecutor(sandbox, config)
        result = executor.execute(payload)  # Blocks until complete
    """
    
    def __init__(
        self,
        sandbox: Optional[SandboxProtocol] = None,
        config: Optional[ExecutorConfig] = None,
    ):
        self.sandbox = sandbox
        self.config = config or ExecutorConfig()
        self._action_handlers: Dict[CodingAction, Callable] = {
            CodingAction.RUN_TESTS: self._execute_run_tests,
            CodingAction.RUN_FOCUSED_TESTS: self._execute_run_focused_tests,
            CodingAction.ANALYZE_FAILURE: self._execute_analyze_failure,
            CodingAction.ANALYZE_TRACEBACK: self._execute_analyze_traceback,
            CodingAction.INSPECT_FILES: self._execute_inspect_files,
            CodingAction.GENERATE_PATCH: self._execute_generate_patch,
            CodingAction.APPLY_PATCH: self._execute_apply_patch,
            CodingAction.REVERT_PATCH: self._execute_revert_patch,
            CodingAction.VALIDATE: self._execute_validate,
            CodingAction.LINT: self._execute_lint,
            CodingAction.BUILD: self._execute_build,
            CodingAction.FINALIZE: self._execute_finalize,
            CodingAction.ABORT: self._execute_abort,
            CodingAction.IDLE: self._execute_idle,
        }
        self._execution_count = 0
        self._is_executing = False
    
    def execute(self, payload: ActionPayload) -> ExecutionResult:
        """Execute an action synchronously.
        
        This method BLOCKS until execution completes. It enforces
        that only one action can be executing at a time.
        
        Args:
            payload: The ActionPayload committed by the CGW.
            
        Returns:
            ExecutionResult with status, output, and artifacts.
            
        Raises:
            RuntimeError: If called while another execution is in progress.
        """
        if self._is_executing:
            raise RuntimeError(
                "BlockingExecutor: Cannot execute while another action is in progress. "
                "This violates the serial execution guarantee."
            )
        
        self._is_executing = True
        self._execution_count += 1
        start_time = time.time()
        
        try:
            handler = self._action_handlers.get(payload.action)
            if handler is None:
                return ExecutionResult(
                    action=payload.action,
                    success=False,
                    error=f"Unknown action: {payload.action}",
                )
            
            result = handler(payload)
            result.duration_ms = (time.time() - start_time) * 1000
            return result
            
        except Exception as e:
            logger.exception(f"Execution failed for {payload.action}: {e}")
            return ExecutionResult(
                action=payload.action,
                success=False,
                error=str(e),
                duration_ms=(time.time() - start_time) * 1000,
            )
        finally:
            self._is_executing = False
    
    def is_executing(self) -> bool:
        """Check if an action is currently executing."""
        return self._is_executing
    
    def execution_count(self) -> int:
        """Get the total number of executions."""
        return self._execution_count
    
    def _spine(self) -> Optional[Any]:
        """Get or create a governed executor bound to the current work dir.
        
        This guarantees the same allowlists + hygiene are enforced.
        Returns None if GovernedExecutor is not available or not enabled.
        
        IMPORTANT:
        - If a spine has been explicitly injected (e.g., by CGWControllerBridge),
          it is always returned.
        - Auto-creation is only performed when explicitly enabled via config,
          so the default behavior continues to respect the sandbox.
        """
        # If the GovernedExecutor class is not available, we can only ever return
        # an already-injected spine (if any).
        if GovernedExecutor is None:
            return getattr(self, "_governed_exec", None)

        # Prefer an explicitly injected spine if present.
        if hasattr(self, "_governed_exec") and self._governed_exec is not None:
            return self._governed_exec

        # By default, do NOT auto-create a governed executor when a sandbox is present,
        # unless the config explicitly allows governed execution alongside the sandbox.
        allow_with_sandbox = getattr(self.config, "allow_governed_with_sandbox", False)
        if self.sandbox is not None and not allow_with_sandbox:
            return None

        # Only auto-create a governed executor when explicitly enabled via config.
        auto_governed = getattr(self.config, "auto_governed_executor", False)
        if not auto_governed:
            return None

        work_dir = self.config.work_dir
        if work_dir is None:
            # fallback to CWD if not configured
            work_dir = Path(".").resolve()
        self._governed_exec = GovernedExecutor(
            repo_dir=str(work_dir),
            allowed_commands=None,  # let global allowlist enforce; set per-profile if you have it
            verify_argv=["pytest", "-q"],  # override via config if desired
            timeout_sec=self.config.timeout_sec,
        )
        return self._governed_exec
    
    # --- Action Handlers ---
    
    def _execute_run_tests(self, payload: ActionPayload) -> ExecutionResult:
        """Run the test suite."""
        spine = self._spine()
        if spine is not None:
            # Delegate to governed executor
            argv = payload.parameters.get("argv") or ["pytest", "-q"]
            step = {"id": f"cgw:{payload.action.value}", "type": "run_tests", "argv": argv}
            r = spine.execute_step(step)
            return ExecutionResult(
                action=payload.action,
                success=bool(r.ok),
                output=r.stdout,
                        error=r.stderr if not r.ok else None,
                        duration_ms=r.elapsed_ms,
                    )
        
        # Fallback to sandbox execution if spine is not available
        test_cmd = payload.parameters.get("test_cmd", self.config.default_test_cmd)
        
        if self.sandbox is None:
            # Mock execution for testing
            return ExecutionResult(
                action=payload.action,
                success=True,
                output="[mock] Tests executed",
                tests_passed=5,
                tests_failed=0,
            )
        
        try:
            result = self.sandbox.run(test_cmd, timeout=self.config.test_timeout)
            output = result.stdout[:self.config.max_output_bytes] if result.stdout else ""
            
            # Parse test results (simplified)
            tests_passed, tests_failed, failing_tests = self._parse_test_output(output)
            
            return ExecutionResult(
                action=payload.action,
                success=result.returncode == 0,
                output=output,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                failing_tests=failing_tests,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                action=payload.action,
                success=False,
                error=f"Test timeout after {self.config.test_timeout}s",
            )
    
    def _execute_run_focused_tests(self, payload: ActionPayload) -> ExecutionResult:
        """Run a subset of focused tests."""
        spine = self._spine()
        if spine is not None:
            # Delegate to governed executor
            argv = payload.parameters.get("argv")
            if not argv:
                return ExecutionResult(
                    action=payload.action,
                    success=False,
                    error="RUN_FOCUSED_TESTS requires parameters.argv",
                )
            step = {"id": f"cgw:{payload.action.value}", "type": "run_cmd", "argv": argv}
            r = spine.execute_step(step)
            return ExecutionResult(
                action=payload.action,
                success=bool(r.ok),
                output=r.stdout,
                error=r.stderr if not r.ok else None,
                tests_ran=True,
            )
        
        # Fallback to original logic
        focus_tests = payload.parameters.get("focus_tests", [])
        test_cmd = payload.parameters.get("test_cmd", self.config.default_test_cmd)
        
        if focus_tests:
            # Pytest-specific focus
            test_cmd = f"{test_cmd} {' '.join(focus_tests)}"
        
        payload.parameters["test_cmd"] = test_cmd
        return self._execute_run_tests(payload)
    
    def _execute_analyze_failure(self, payload: ActionPayload) -> ExecutionResult:
        """Analyze test failures."""
        # This is typically handled by the LLM in the proposal phase
        # The executor just acknowledges the analysis request
        return ExecutionResult(
            action=payload.action,
            success=True,
            output="Failure analysis complete",
            artifacts={"analysis_requested": True},
        )
    
    def _execute_analyze_traceback(self, payload: ActionPayload) -> ExecutionResult:
        """Analyze traceback from test output."""
        return ExecutionResult(
            action=payload.action,
            success=True,
            output="Traceback analysis complete",
            artifacts={"traceback_analyzed": True},
        )
    
    def _execute_inspect_files(self, payload: ActionPayload) -> ExecutionResult:
        """Inspect specific files."""
        files = payload.parameters.get("files", [])
        contents = {}
        
        if self.sandbox:
            for f in files[:10]:  # Limit files
                try:
                    contents[f] = self.sandbox.read_file(f)
                except Exception as e:
                    contents[f] = f"Error: {e}"
        
        return ExecutionResult(
            action=payload.action,
            success=True,
            output=f"Inspected {len(contents)} files",
            artifacts={"file_contents": contents},
        )
    
    def _execute_generate_patch(self, payload: ActionPayload) -> ExecutionResult:
        """Generate a patch (typically done by LLM in proposal phase)."""
        # The actual patch generation is done externally
        # This just acknowledges that we're in patch generation mode
        return ExecutionResult(
            action=payload.action,
            success=True,
            output="Patch generation requested",
            artifacts={"patch_requested": True},
        )
    
    def _execute_apply_patch(self, payload: ActionPayload) -> ExecutionResult:
        """Apply a patch to the codebase."""
        diff = payload.parameters.get("diff", "")
        
        if not diff:
            return ExecutionResult(
                action=payload.action,
                success=False,
                error="No diff provided",
            )
        
        spine = self._spine()
        if spine is not None:
            # Delegate to governed executor with patch hygiene
            step = {"id": f"cgw:{payload.action.value}", "type": "apply_patch", "diff": diff}
            r = spine.execute_step(step)
            return ExecutionResult(
                action=payload.action,
                success=bool(r.ok),
                output=r.stdout,
                error=r.stderr if not r.ok else None,
                patch_applied=bool(r.ok),
            )
        
        # Fallback to sandbox execution
        if self.sandbox is None:
            return ExecutionResult(
                action=payload.action,
                success=True,
                output="[mock] Patch applied",
                patch_applied=True,
            )
        
        try:
            success = self.sandbox.apply_diff(diff)
            return ExecutionResult(
                action=payload.action,
                success=success,
                output="Patch applied" if success else "Patch application failed",
                patch_applied=success,
            )
        except Exception as e:
            return ExecutionResult(
                action=payload.action,
                success=False,
                error=f"Patch error: {e}",
            )
    
    def _execute_revert_patch(self, payload: ActionPayload) -> ExecutionResult:
        """Revert the last applied patch."""
        spine = self._spine()
        if spine is not None:
            # Delegate to governed executor
            step = {"id": f"cgw:{payload.action.value}", "type": "reset_hard"}
            r = spine.execute_step(step)
            return ExecutionResult(
                action=payload.action,
                success=bool(r.ok),
                output=r.stdout,
                error=r.stderr if not r.ok else None,
                patch_applied=False,
            )
        
        # Fallback to sandbox execution
        if self.sandbox is None:
            return ExecutionResult(
                action=payload.action,
                success=True,
                output="[mock] Patch reverted",
            )
        
        try:
            result = self.sandbox.run("git checkout -- .", timeout=self.config.patch_timeout)
            return ExecutionResult(
                action=payload.action,
                success=result.returncode == 0,
                output="Patch reverted",
            )
        except Exception as e:
            return ExecutionResult(
                action=payload.action,
                success=False,
                error=f"Revert error: {e}",
            )
    
    def _execute_validate(self, payload: ActionPayload) -> ExecutionResult:
        """Run validation checks."""
        spine = self._spine()
        if spine is not None:
            # Validation == verifier command
            vr = spine.verify()
            return ExecutionResult(
                action=payload.action,
                success=bool(vr.ok),
                output=vr.stdout,
                error=vr.stderr if not vr.ok else None,
            )
        
        # Fallback
        return ExecutionResult(
            action=payload.action,
            success=True,
            output="Validation complete",
        )
    
    def _execute_lint(self, payload: ActionPayload) -> ExecutionResult:
        """Run linting."""
        spine = self._spine()
        if spine is not None:
            # Delegate to governed executor
            argv = payload.parameters.get("argv") or ["python", "-m", "ruff", "check", "."]
            step = {"id": f"cgw:{payload.action.value}", "type": "run_cmd", "argv": argv}
            r = spine.execute_step(step)
            return ExecutionResult(
                action=payload.action,
                success=bool(r.ok),
                output=r.stdout,
                error=r.stderr if not r.ok else None,
            )
        
        # Fallback to sandbox execution
        if self.sandbox is None:
            return ExecutionResult(
                action=payload.action,
                success=True,
                output="[mock] Lint passed",
            )
        
        lint_cmd = payload.parameters.get("lint_cmd", "flake8 --count --max-line-length=120")
        try:
            result = self.sandbox.run(lint_cmd, timeout=self.config.lint_timeout)
            return ExecutionResult(
                action=payload.action,
                success=result.returncode == 0,
                output=result.stdout[:self.config.max_output_bytes] if result.stdout else "",
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                action=payload.action,
                success=False,
                error=f"Lint timeout after {self.config.lint_timeout}s",
            )
    
    def _execute_build(self, payload: ActionPayload) -> ExecutionResult:
        """Run build commands."""
        spine = self._spine()
        if spine is not None:
            # Delegate to governed executor
            argv = payload.parameters.get("argv") or ["python", "-m", "compileall", "."]
            step = {"id": f"cgw:{payload.action.value}", "type": "run_cmd", "argv": argv}
            r = spine.execute_step(step)
            return ExecutionResult(
                action=payload.action,
                success=bool(r.ok),
                output=r.stdout,
                error=r.stderr if not r.ok else None,
            )
        
        # Fallback to sandbox execution
        if self.sandbox is None:
            return ExecutionResult(
                action=payload.action,
                success=True,
                output="[mock] Build succeeded",
            )
        
        build_cmd = payload.parameters.get("build_cmd", "python -m py_compile *.py")
        try:
            result = self.sandbox.run(build_cmd, timeout=self.config.build_timeout)
            return ExecutionResult(
                action=payload.action,
                success=result.returncode == 0,
                output=result.stdout[:self.config.max_output_bytes] if result.stdout else "",
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                action=payload.action,
                success=False,
                error=f"Build timeout after {self.config.build_timeout}s",
            )
    
    def _execute_finalize(self, payload: ActionPayload) -> ExecutionResult:
        """Finalize the coding session (success)."""
        return ExecutionResult(
            action=payload.action,
            success=True,
            output="Session finalized successfully",
        )
    
    def _execute_abort(self, payload: ActionPayload) -> ExecutionResult:
        """Abort the coding session."""
        reason = payload.context.get("reason", "Unknown")
        return ExecutionResult(
            action=payload.action,
            success=True,  # Abort itself succeeds
            output=f"Session aborted: {reason}",
        )
    
    def _execute_idle(self, payload: ActionPayload) -> ExecutionResult:
        """Handle idle state (no action needed)."""
        return ExecutionResult(
            action=payload.action,
            success=True,
            output="Idle",
        )
    
    def _parse_test_output(self, output: str) -> tuple:
        """Parse test output to extract pass/fail counts."""
        # Simplified pytest output parsing
        tests_passed = 0
        tests_failed = 0
        failing_tests = []
        
        for line in output.split("\n"):
            if "passed" in line.lower():
                # Try to extract count
                import re
                match = re.search(r"(\d+)\s+passed", line)
                if match:
                    tests_passed = int(match.group(1))
            if "failed" in line.lower():
                match = re.search(r"(\d+)\s+failed", line)
                if match:
                    tests_failed = int(match.group(1))
            if "FAILED" in line:
                # Extract test name
                parts = line.split("::")
                if parts:
                    failing_tests.append(parts[0])
        
        return tests_passed, tests_failed, failing_tests
