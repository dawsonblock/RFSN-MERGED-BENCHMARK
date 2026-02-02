"""Parallel Executor - Concurrent step execution.

Executes independent steps in parallel to improve throughput.
Respects resource limits and maintains state consistency.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from .schema import ControllerOutcome, ControllerTaskSpec, Plan, PlanState, Step, StepStatus


@dataclass
class ParallelBatch:
    """A batch of steps that can run in parallel."""
    
    steps: list[Step]
    batch_id: int
    
    @property
    def size(self) -> int:
        return len(self.steps)


@dataclass
class ParallelResult:
    """Result of parallel batch execution."""
    
    batch_id: int
    outcomes: list[ControllerOutcome]
    elapsed_ms: int
    errors: list[str] = field(default_factory=list)
    
    @property
    def all_success(self) -> bool:
        return all(o.success for o in self.outcomes)
    
    @property
    def any_success(self) -> bool:
        return any(o.success for o in self.outcomes)


# Type alias for controller execution function
ControllerFn = Callable[[ControllerTaskSpec], ControllerOutcome]


class ParallelStepExecutor:
    """Executes independent steps in parallel."""
    
    def __init__(
        self,
        max_workers: int = 4,
        use_async: bool = False,
    ):
        """Initialize the executor.
        
        Args:
            max_workers: Maximum concurrent executions.
            use_async: Use asyncio instead of threads.
        """
        self._max_workers = max_workers
        self._use_async = use_async
        self._executor: ThreadPoolExecutor | None = None
    
    def find_parallel_batches(
        self,
        plan: Plan,
        state: PlanState,
    ) -> list[ParallelBatch]:
        """Find steps that can run in parallel.
        
        Steps are parallelizable if:
        1. All their dependencies are DONE
        2. They don't depend on each other
        3. Their allowed_files don't overlap significantly
        
        Args:
            plan: The plan.
            state: Current plan state.
            
        Returns:
            List of parallel batches in execution order.
        """
        batches = []
        remaining = [s for s in plan.steps if s.status == StepStatus.PENDING]
        done_ids = set(state.completed_steps)
        batch_id = 0
        
        while remaining:
            # Find steps whose dependencies are all done
            ready = []
            for step in remaining:
                deps_satisfied = all(d in done_ids for d in step.dependencies)
                if deps_satisfied:
                    ready.append(step)
            
            if not ready:
                break
            
            # Filter for non-conflicting steps
            parallel_group = self._filter_non_conflicting(ready)
            
            if parallel_group:
                batches.append(ParallelBatch(
                    steps=parallel_group,
                    batch_id=batch_id,
                ))
                batch_id += 1
                
                # Mark these as "done" for next iteration
                for step in parallel_group:
                    done_ids.add(step.step_id)
                    remaining = [s for s in remaining if s.step_id != step.step_id]
            else:
                break
        
        return batches
    
    def _filter_non_conflicting(self, steps: list[Step]) -> list[Step]:
        """Filter steps to keep only non-conflicting ones.
        
        Two steps conflict if they touch the same files.
        
        Args:
            steps: Candidate steps.
            
        Returns:
            Non-conflicting subset.
        """
        if len(steps) <= 1:
            return steps
        
        result = [steps[0]]
        touched_files = set(steps[0].allowed_files)
        
        for step in steps[1:]:
            step_files = set(step.allowed_files)
            if not step_files & touched_files:
                result.append(step)
                touched_files.update(step_files)
            
            if len(result) >= self._max_workers:
                break
        
        return result
    
    def execute_batch(
        self,
        batch: ParallelBatch,
        controller_fn: ControllerFn,
    ) -> ParallelResult:
        """Execute a batch of steps in parallel.
        
        Args:
            batch: Steps to execute.
            controller_fn: Function to execute each step.
            
        Returns:
            ParallelResult with all outcomes.
        """
        import time
        start = time.monotonic()
        
        if self._use_async:
            outcomes = self._execute_async(batch.steps, controller_fn)
        else:
            outcomes = self._execute_threaded(batch.steps, controller_fn)
        
        elapsed = int((time.monotonic() - start) * 1000)
        
        return ParallelResult(
            batch_id=batch.batch_id,
            outcomes=outcomes,
            elapsed_ms=elapsed,
        )
    
    def _execute_threaded(
        self,
        steps: list[Step],
        controller_fn: ControllerFn,
    ) -> list[ControllerOutcome]:
        """Execute steps using thread pool.
        
        Args:
            steps: Steps to execute.
            controller_fn: Execution function.
            
        Returns:
            List of outcomes.
        """
        outcomes = []
        
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_step = {
                executor.submit(controller_fn, step.get_task_spec()): step
                for step in steps
            }
            
            for future in as_completed(future_to_step):
                try:
                    outcome = future.result()
                    outcomes.append(outcome)
                except Exception as e:
                    step = future_to_step[future]
                    outcomes.append(ControllerOutcome(
                        step_id=step.step_id,
                        success=False,
                        error_message=str(e),
                    ))
        
        return outcomes
    
    def _execute_async(
        self,
        steps: list[Step],
        controller_fn: ControllerFn,
    ) -> list[ControllerOutcome]:
        """Execute steps using asyncio.
        
        Args:
            steps: Steps to execute.
            controller_fn: Execution function.
            
        Returns:
            List of outcomes.
        """
        async def run_step(step: Step) -> ControllerOutcome:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                controller_fn,
                step.get_task_spec(),
            )
        
        async def run_all() -> list[ControllerOutcome]:
            tasks = [run_step(s) for s in steps]
            # Use return_exceptions=True to capture errors instead of raising
            return await asyncio.gather(*tasks, return_exceptions=True) # type: ignore
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        results = loop.run_until_complete(run_all())
        
        outcomes = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                outcomes.append(ControllerOutcome(
                    step_id=steps[i].step_id,
                    success=False,
                    error_message=str(result),
                ))
            else:
                outcomes.append(result)
        
        return outcomes
    
    def merge_outcomes(
        self,
        state: PlanState,
        outcomes: list[ControllerOutcome],
    ) -> PlanState:
        """Merge parallel outcomes into plan state.
        
        Args:
            state: Current state.
            outcomes: Outcomes from parallel execution.
            
        Returns:
            Updated state.
        """
        for outcome in outcomes:
            if outcome.success:
                if outcome.step_id not in state.completed_steps:
                    state.completed_steps.append(outcome.step_id)
            elif outcome.step_id not in state.failed_steps:
                state.failed_steps.append(outcome.step_id)
        
        return state


class ParallelExecutionConfig:
    """Configuration for parallel execution."""
    
    def __init__(
        self,
        enabled: bool = True,
        max_workers: int = 8,  # Increased default
        min_batch_size: int = 2,
        max_batch_size: int = 8, # Increased default
        use_async: bool = False,
    ):
        self.enabled = enabled
        self.max_workers = max_workers
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.use_async = use_async
    
    def create_executor(self) -> ParallelStepExecutor:
        """Create an executor with this configuration."""
        return ParallelStepExecutor(
            max_workers=self.max_workers,
            use_async=self.use_async,
        )
