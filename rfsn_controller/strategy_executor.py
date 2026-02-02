"""Strategy Executor for RFSN Controller.

Executes repair strategies based on planner decisions. Extracted from controller.py
to reduce complexity and improve testability.

Responsibilities:
- Strategy selection and execution
- Step-by-step execution tracking
- Progress monitoring
- Rollback on failure
- Strategy metrics collection
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .structured_logging import get_logger

logger = get_logger(__name__)


class StrategyType(Enum):
    """Types of repair strategies."""

    DIRECT_PATCH = "direct_patch"  # Apply patch directly
    LOCALIZE_THEN_FIX = "localize_then_fix"  # Find issue, then fix
    TEST_DRIVEN = "test_driven"  # Fix based on test failures
    INCREMENTAL = "incremental"  # Fix one test at a time
    ENSEMBLE = "ensemble"  # Try multiple approaches in parallel
    HYPOTHESIS_DRIVEN = "hypothesis_driven"  # Planner v5 style


@dataclass
class StrategyStep:
    """Single step in a repair strategy."""

    step_id: str
    action: str  # edit_file, run_tests, read_file, etc.
    target: str | None = None  # File path or command
    content: str | None = None  # Content for edit
    metadata: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 60


@dataclass
class StrategyConfig:
    """Configuration for strategy execution."""

    strategy_type: StrategyType
    max_steps: int = 12
    max_retries_per_step: int = 2
    timeout_seconds: int = 300
    allow_parallel: bool = True
    working_directory: Path = field(default_factory=lambda: Path.cwd())


@dataclass
class StepResult:
    """Result of executing a single step."""

    step_id: str
    success: bool
    output: str
    error: str | None = None
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyResult:
    """Result of executing a complete strategy."""

    strategy_type: StrategyType
    success: bool
    steps_executed: int
    step_results: list[StepResult] = field(default_factory=list)
    final_state: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    error_message: str | None = None
    patches_generated: list[str] = field(default_factory=list)


class StrategyExecutor:
    """
    Executes repair strategies based on planner decisions.

    This executor handles the step-by-step execution of repair strategies,
    including progress tracking, error handling, and rollback capabilities.

    Example:
        >>> config = StrategyConfig(
        ...     strategy_type=StrategyType.DIRECT_PATCH,
        ...     max_steps=10
        ... )
        >>> executor = StrategyExecutor(config)
        >>> 
        >>> strategy_steps = [
        ...     StrategyStep("step1", "read_file", target="main.py"),
        ...     StrategyStep("step2", "edit_file", target="main.py", content="fix"),
        ...     StrategyStep("step3", "run_tests")
        ... ]
        >>> 
        >>> result = await executor.execute_strategy(strategy_steps)
        >>> print(f"Success: {result.success}")
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize strategy executor.

        Args:
            config: Strategy configuration
        """
        self.config = config
        self._stats = {
            "total_strategies": 0,
            "successful_strategies": 0,
            "failed_strategies": 0,
            "total_steps": 0,
            "failed_steps": 0,
            "total_duration": 0.0,
        }
        self._current_strategy_state: dict[str, Any] = {}

    async def execute_strategy(
        self,
        steps: list[StrategyStep],
        context: dict[str, Any] | None = None,
    ) -> StrategyResult:
        """
        Execute a complete repair strategy.

        Args:
            steps: List of strategy steps to execute
            context: Optional execution context (repo path, test command, etc.)

        Returns:
            StrategyResult with execution outcomes

        Example:
            >>> steps = [
            ...     StrategyStep("step1", "read_file", target="main.py"),
            ...     StrategyStep("step2", "edit_file", target="main.py", content="fix")
            ... ]
            >>> result = await executor.execute_strategy(steps)
        """
        import time

        logger.info(
            "Starting strategy execution",
            strategy_type=self.config.strategy_type.value,
            steps_count=len(steps),
        )

        self._stats["total_strategies"] += 1
        self._current_strategy_state = context or {}

        start_time = time.time()
        step_results = []
        success = True

        # Execute steps sequentially or in parallel
        for i, step in enumerate(steps):
            if i >= self.config.max_steps:
                logger.warning(
                    "Max steps reached",
                    executed=i,
                    max_steps=self.config.max_steps,
                )
                break

            logger.info(
                "Executing step",
                step_id=step.step_id,
                action=step.action,
                step_num=i + 1,
                total_steps=len(steps),
            )

            step_result = await self._execute_step(step)
            step_results.append(step_result)

            self._stats["total_steps"] += 1

            if not step_result.success:
                self._stats["failed_steps"] += 1
                logger.error(
                    "Step failed",
                    step_id=step.step_id,
                    error=step_result.error,
                )

                # Decide whether to continue or abort
                if self._should_abort_strategy(step, step_result):
                    success = False
                    break

        duration = time.time() - start_time
        self._stats["total_duration"] += duration

        if success:
            self._stats["successful_strategies"] += 1
        else:
            self._stats["failed_strategies"] += 1

        logger.info(
            "Strategy execution complete",
            success=success,
            steps_executed=len(step_results),
            duration_seconds=round(duration, 2),
        )

        return StrategyResult(
            strategy_type=self.config.strategy_type,
            success=success,
            steps_executed=len(step_results),
            step_results=step_results,
            final_state=self._current_strategy_state.copy(),
            duration_seconds=duration,
            error_message=step_results[-1].error if not success else None,
        )

    async def _execute_step(self, step: StrategyStep) -> StepResult:
        """
        Execute a single strategy step.

        Args:
            step: Step to execute

        Returns:
            StepResult
        """
        import time

        start_time = time.time()

        try:
            # Dispatch to appropriate handler
            if step.action == "read_file":
                result = await self._handle_read_file(step)
            elif step.action == "edit_file":
                result = await self._handle_edit_file(step)
            elif step.action == "run_tests":
                result = await self._handle_run_tests(step)
            elif step.action == "run_command":
                result = await self._handle_run_command(step)
            else:
                result = StepResult(
                    step_id=step.step_id,
                    success=False,
                    output="",
                    error=f"Unknown action: {step.action}",
                )

            result.duration_seconds = time.time() - start_time
            return result

        except Exception as e:
            logger.error(
                "Step execution error",
                step_id=step.step_id,
                error=str(e),
            )
            return StepResult(
                step_id=step.step_id,
                success=False,
                output="",
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    async def _handle_read_file(self, step: StrategyStep) -> StepResult:
        """Handle read_file action."""
        if not step.target:
            return StepResult(
                step_id=step.step_id,
                success=False,
                output="",
                error="No target file specified",
            )

        try:
            file_path = self.config.working_directory / step.target
            content = file_path.read_text()

            # Store in state for later use
            self._current_strategy_state[f"file_content:{step.target}"] = content

            return StepResult(
                step_id=step.step_id,
                success=True,
                output=content,
                metadata={"file_path": str(file_path), "size": len(content)},
            )

        except Exception as e:
            return StepResult(
                step_id=step.step_id,
                success=False,
                output="",
                error=f"Failed to read file: {e}",
            )

    async def _handle_edit_file(self, step: StrategyStep) -> StepResult:
        """Handle edit_file action."""
        if not step.target or not step.content:
            return StepResult(
                step_id=step.step_id,
                success=False,
                output="",
                error="No target file or content specified",
            )

        try:
            file_path = self.config.working_directory / step.target

            # Backup original
            if file_path.exists():
                backup_key = f"backup:{step.target}"
                self._current_strategy_state[backup_key] = file_path.read_text()

            # Write new content
            file_path.write_text(step.content)

            return StepResult(
                step_id=step.step_id,
                success=True,
                output=f"Edited {step.target}",
                metadata={
                    "file_path": str(file_path),
                    "size": len(step.content),
                },
            )

        except Exception as e:
            return StepResult(
                step_id=step.step_id,
                success=False,
                output="",
                error=f"Failed to edit file: {e}",
            )

    async def _handle_run_tests(self, step: StrategyStep) -> StepResult:
        """Handle run_tests action."""
        # This would integrate with VerificationManager
        # For now, return a placeholder
        return StepResult(
            step_id=step.step_id,
            success=True,
            output="Tests would be run here",
            metadata={"test_command": step.content or "pytest"},
        )

    async def _handle_run_command(self, step: StrategyStep) -> StepResult:
        """Handle run_command action."""
        if not step.content:
            return StepResult(
                step_id=step.step_id,
                success=False,
                output="",
                error="No command specified",
            )

        try:
            # Parse command
            import shlex

            cmd_parts = shlex.split(step.content)

            # Run command
            proc = await asyncio.create_subprocess_exec(
                *cmd_parts,
                cwd=str(self.config.working_directory),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=step.timeout_seconds,
            )

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            success = proc.returncode == 0

            return StepResult(
                step_id=step.step_id,
                success=success,
                output=stdout,
                error=stderr if not success else None,
                metadata={"exit_code": proc.returncode},
            )

        except TimeoutError:
            return StepResult(
                step_id=step.step_id,
                success=False,
                output="",
                error=f"Command timeout after {step.timeout_seconds}s",
            )
        except Exception as e:
            return StepResult(
                step_id=step.step_id,
                success=False,
                output="",
                error=f"Command execution failed: {e}",
            )

    def _should_abort_strategy(
        self,
        step: StrategyStep,
        result: StepResult,
    ) -> bool:
        """
        Determine if strategy should be aborted after step failure.

        Args:
            step: Failed step
            result: Step result

        Returns:
            True if strategy should abort
        """
        # Critical actions should abort
        critical_actions = {"edit_file", "run_tests"}
        if step.action in critical_actions:
            logger.warning(
                "Critical step failed, aborting strategy",
                step_id=step.step_id,
                action=step.action,
            )
            return True

        # Otherwise continue
        return False

    async def rollback_strategy(self) -> bool:
        """
        Rollback changes made during strategy execution.

        Returns:
            True if rollback successful
        """
        logger.info("Rolling back strategy changes")

        try:
            # Restore backed up files
            for key, content in self._current_strategy_state.items():
                if key.startswith("backup:"):
                    file_path_str = key.replace("backup:", "")
                    file_path = self.config.working_directory / file_path_str

                    logger.debug("Restoring backup", file_path=str(file_path))
                    file_path.write_text(content)

            logger.info("Rollback complete")
            return True

        except Exception as e:
            logger.error("Rollback failed", error=str(e))
            return False

    def get_stats(self) -> dict[str, Any]:
        """
        Get strategy execution statistics.

        Returns:
            Dictionary of stats
        """
        total = self._stats["total_strategies"]
        success_rate = (
            self._stats["successful_strategies"] / total if total > 0 else 0.0
        )

        avg_duration = (
            self._stats["total_duration"] / total if total > 0 else 0.0
        )

        return {
            **self._stats,
            "success_rate": success_rate,
            "average_duration": avg_duration,
        }

    async def execute_parallel_strategies(
        self,
        strategies: list[list[StrategyStep]],
        context: dict[str, Any] | None = None,
    ) -> list[StrategyResult]:
        """
        Execute multiple strategies in parallel.

        Args:
            strategies: List of strategy step lists
            context: Shared execution context

        Returns:
            List of strategy results
        """
        if not self.config.allow_parallel:
            logger.warning("Parallel execution disabled, running sequentially")
            results = []
            for strategy_steps in strategies:
                result = await self.execute_strategy(strategy_steps, context)
                results.append(result)
            return results

        logger.info(
            "Executing parallel strategies",
            count=len(strategies),
        )

        # Execute all strategies concurrently
        tasks = [
            self.execute_strategy(steps, context) for steps in strategies
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Parallel strategy failed",
                    strategy_num=i,
                    error=str(result),
                )
            else:
                valid_results.append(result)

        return valid_results
