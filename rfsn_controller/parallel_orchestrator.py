"""Multi-Worktree Parallel Orchestrator.

Runs N isolated controller instances in parallel, each with:
- Separate working directory (git worktree)
- Dedicated controller process
- Independent logs/evidence collection
- One-attempt-per-task model

Safety: Each worker stays strictly serial internally.
Parallelism is only across tasks, never within a task.

The orchestrator implements the RFSN parallel execution strategy:
1. Partition tasks across N workers
2. Each worker gets an isolated worktree
3. All workers execute in parallel
4. Results are aggregated with deterministic tie-breaking

Usage:
    from rfsn_controller.parallel_orchestrator import run_parallel_benchmark
    
    results = run_parallel_benchmark(
        tasks=tasks,
        max_workers=4,
        output_dir=Path("runs"),
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rfsn_controller.structured_logging import get_logger

if TYPE_CHECKING:
    from eval.run import EvalResult, SWEBenchTask, TaskStatus

logger = get_logger(__name__)


@dataclass
class WorkerConfig:
    """Configuration for a single worker."""
    
    worker_id: int
    worktree_dir: Path
    output_dir: Path
    cache_dir: Path
    
    # Inherited from main config
    max_time_per_task: int = 1800
    max_steps_per_task: int = 50
    max_patches_per_task: int = 20


@dataclass
class WorkerResult:
    """Result from a single worker's task execution."""
    
    task_id: str
    worker_id: int
    success: bool
    status: str  # TaskStatus value
    
    # Artifact paths
    patch_path: Path | None
    controller_log: Path | None
    events_jsonl: Path | None
    evidence_dir: Path | None
    
    # Metrics
    runtime_seconds: float
    failing_tests: int
    gate_rejections: int
    security_violations: list[str]
    
    # For tie-breaking
    patch_size: int = 0
    files_touched: int = 0
    patch_hash: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "worker_id": self.worker_id,
            "success": self.success,
            "status": self.status,
            "patch_path": str(self.patch_path) if self.patch_path else None,
            "controller_log": str(self.controller_log) if self.controller_log else None,
            "events_jsonl": str(self.events_jsonl) if self.events_jsonl else None,
            "evidence_dir": str(self.evidence_dir) if self.evidence_dir else None,
            "runtime_seconds": self.runtime_seconds,
            "failing_tests": self.failing_tests,
            "gate_rejections": self.gate_rejections,
            "security_violations": self.security_violations,
            "patch_size": self.patch_size,
            "files_touched": self.files_touched,
            "patch_hash": self.patch_hash,
        }


class WorktreeManager:
    """Manages isolated git worktrees for parallel execution."""
    
    def __init__(self, base_dir: Path, num_workers: int):
        """Initialize worktree manager.
        
        Args:
            base_dir: Base directory for worktrees.
            num_workers: Number of workers to create worktrees for.
        """
        self.base_dir = base_dir
        self.num_workers = num_workers
        self.worktrees: dict[int, Path] = {}
        
    def setup_worktrees(self) -> dict[int, Path]:
        """Create worktree directories for each worker.
        
        Returns:
            Mapping of worker ID to worktree path.
        """
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        for worker_id in range(self.num_workers):
            worktree_path = self.base_dir / f"worker_{worker_id}"
            worktree_path.mkdir(parents=True, exist_ok=True)
            self.worktrees[worker_id] = worktree_path
            logger.debug(f"Created worktree for worker {worker_id}: {worktree_path}")
        
        return self.worktrees
    
    def cleanup_worktrees(self) -> None:
        """Remove all worktree directories."""
        for worker_id, worktree_path in self.worktrees.items():
            try:
                if worktree_path.exists():
                    shutil.rmtree(worktree_path)
                    logger.debug(f"Cleaned up worktree for worker {worker_id}")
            except Exception as e:
                logger.warning(f"Failed to cleanup worktree {worker_id}: {e}")


class ParallelOrchestrator:
    """Orchestrates parallel task execution across isolated workers."""
    
    def __init__(
        self,
        max_workers: int = 4,
        output_dir: Path = Path("runs"),
        worktree_base: Path | None = None,
    ):
        """Initialize parallel orchestrator.
        
        Args:
            max_workers: Maximum number of parallel workers.
            output_dir: Base directory for output artifacts.
            worktree_base: Base directory for worktrees (temp if not specified).
        """
        self.max_workers = max_workers
        self.output_dir = output_dir
        
        # Use temp directory if not specified
        if worktree_base is None:
            self.worktree_base = Path(tempfile.mkdtemp(prefix="rfsn_worktrees_"))
            self._cleanup_worktrees_on_exit = True
        else:
            self.worktree_base = worktree_base
            self._cleanup_worktrees_on_exit = False
        
        self.worktree_manager = WorktreeManager(self.worktree_base, max_workers)
        self.results: list[WorkerResult] = []
        
    async def run_worker(
        self,
        worker_id: int,
        tasks: list["SWEBenchTask"],
        config: WorkerConfig,
    ) -> list[WorkerResult]:
        """Run a single worker on its assigned tasks.
        
        Args:
            worker_id: Worker identifier.
            tasks: Tasks assigned to this worker.
            config: Worker configuration.
            
        Returns:
            List of results from this worker.
        """
        from eval.run import EvalConfig, EvalRunner, TaskStatus
        
        results = []
        
        # Create worker-specific directories
        worker_output = config.output_dir / f"worker_{worker_id}"
        worker_output.mkdir(parents=True, exist_ok=True)
        
        # Create evaluation runner for this worker
        eval_config = EvalConfig(
            work_dir=config.worktree_dir,
            results_dir=worker_output,
            cache_dir=config.cache_dir,
            parallel_tasks=1,  # Serial within worker
            max_time_per_task=config.max_time_per_task,
            max_steps_per_task=config.max_steps_per_task,
            max_patches_per_task=config.max_patches_per_task,
        )
        
        runner = EvalRunner(eval_config)
        
        for task in tasks:
            start_time = time.time()
            logger.info(f"[Worker {worker_id}] Starting task {task.task_id}")
            
            try:
                # Run the task
                eval_result = await runner.run_task(task)
                
                # Collect artifacts
                task_dir = worker_output / task.task_id
                patch_path = task_dir / "patch.diff" if task_dir.exists() else None
                controller_log = task_dir / "controller.log" if task_dir.exists() else None
                events_jsonl = task_dir / "events.jsonl" if task_dir.exists() else None
                evidence_dir = task_dir / "evidence" if (task_dir / "evidence").exists() else None
                
                # Calculate patch metrics for tie-breaking
                patch_size = 0
                files_touched = 0
                patch_hash = ""
                
                if eval_result.final_patch:
                    patch_size = len(eval_result.final_patch)
                    files_touched = eval_result.final_patch.count("diff --git")
                    patch_hash = hashlib.sha256(
                        eval_result.final_patch.encode()
                    ).hexdigest()[:16]
                
                result = WorkerResult(
                    task_id=task.task_id,
                    worker_id=worker_id,
                    success=eval_result.success,
                    status=eval_result.status.value,
                    patch_path=patch_path,
                    controller_log=controller_log,
                    events_jsonl=events_jsonl,
                    evidence_dir=evidence_dir,
                    runtime_seconds=time.time() - start_time,
                    failing_tests=eval_result.tests_failed,
                    gate_rejections=eval_result.gate_rejections,
                    security_violations=eval_result.security_violations,
                    patch_size=patch_size,
                    files_touched=files_touched,
                    patch_hash=patch_hash,
                )
                
                results.append(result)
                logger.info(
                    f"[Worker {worker_id}] Completed task {task.task_id}: "
                    f"success={result.success}, time={result.runtime_seconds:.1f}s"
                )
                
            except Exception as e:
                logger.exception(f"[Worker {worker_id}] Task {task.task_id} failed: {e}")
                results.append(WorkerResult(
                    task_id=task.task_id,
                    worker_id=worker_id,
                    success=False,
                    status="ERROR",
                    patch_path=None,
                    controller_log=None,
                    events_jsonl=None,
                    evidence_dir=None,
                    runtime_seconds=time.time() - start_time,
                    failing_tests=0,
                    gate_rejections=0,
                    security_violations=[],
                ))
        
        return results
    
    def partition_tasks(
        self,
        tasks: list["SWEBenchTask"],
    ) -> list[list["SWEBenchTask"]]:
        """Partition tasks among workers using round-robin.
        
        Args:
            tasks: All tasks to partition.
            
        Returns:
            List of task lists, one per worker.
        """
        partitions: list[list["SWEBenchTask"]] = [[] for _ in range(self.max_workers)]
        
        for i, task in enumerate(tasks):
            worker_id = i % self.max_workers
            partitions[worker_id].append(task)
        
        for i, partition in enumerate(partitions):
            logger.info(f"Worker {i}: {len(partition)} tasks")
        
        return partitions
    
    async def run(
        self,
        tasks: list["SWEBenchTask"],
    ) -> list[WorkerResult]:
        """Run all tasks in parallel across workers.
        
        Args:
            tasks: Tasks to execute.
            
        Returns:
            Aggregated results from all workers.
        """
        logger.info(f"Starting parallel execution: {len(tasks)} tasks, {self.max_workers} workers")
        
        # Setup worktrees
        worktrees = self.worktree_manager.setup_worktrees()
        
        # Partition tasks
        partitions = self.partition_tasks(tasks)
        
        # Create worker configs
        worker_configs = {
            worker_id: WorkerConfig(
                worker_id=worker_id,
                worktree_dir=worktree_path,
                output_dir=self.output_dir,
                cache_dir=self.output_dir / "cache",
            )
            for worker_id, worktree_path in worktrees.items()
        }
        
        # Run workers in parallel
        worker_tasks = [
            self.run_worker(worker_id, partitions[worker_id], worker_configs[worker_id])
            for worker_id in range(self.max_workers)
            if partitions[worker_id]  # Skip workers with no tasks
        ]
        
        all_results = await asyncio.gather(*worker_tasks, return_exceptions=True)
        
        # Flatten and collect results
        for result in all_results:
            if isinstance(result, Exception):
                logger.error(f"Worker failed with exception: {result}")
            else:
                self.results.extend(result)
        
        # Cleanup worktrees if needed
        if self._cleanup_worktrees_on_exit:
            self.worktree_manager.cleanup_worktrees()
        
        logger.info(f"Parallel execution complete: {len(self.results)} results")
        return self.results


def select_best_patch(attempts: list[WorkerResult], task_id: str) -> WorkerResult | None:
    """Select the best patch from multiple attempts using deterministic tie-breaking.
    
    Tie-break order:
    1. PASS beats non-PASS
    2. Fewer failing tests
    3. Smaller diff (patch_size)
    4. Fewer files touched
    5. Lexicographic patch hash (deterministic)
    
    Args:
        attempts: List of results for the same task.
        task_id: Task identifier for logging.
        
    Returns:
        Best result, or None if no attempts.
    """
    if not attempts:
        return None
    
    # Filter to matching task_id
    matching = [a for a in attempts if a.task_id == task_id]
    if not matching:
        return None
    
    if len(matching) == 1:
        return matching[0]
    
    # Sort using tie-break criteria
    def sort_key(r: WorkerResult) -> tuple:
        # Lower is better for all criteria
        return (
            0 if r.status == "PASS" else 1,     # PASS first
            r.failing_tests,                     # Fewer failing tests
            r.patch_size,                        # Smaller patches
            r.files_touched,                     # Fewer files
            r.patch_hash,                        # Lexicographic hash
        )
    
    sorted_attempts = sorted(matching, key=sort_key)
    best = sorted_attempts[0]
    
    logger.info(
        f"Selected best patch for {task_id}: worker={best.worker_id}, "
        f"status={best.status}, failing={best.failing_tests}, size={best.patch_size}"
    )
    
    return best


async def run_parallel_benchmark(
    tasks: list["SWEBenchTask"],
    max_workers: int = 4,
    output_dir: Path = Path("runs"),
) -> list[WorkerResult]:
    """Run tasks in parallel across isolated worktrees.
    
    Args:
        tasks: SWE-bench tasks to run.
        max_workers: Number of parallel workers.
        output_dir: Base output directory.
        
    Returns:
        List of worker results.
    """
    orchestrator = ParallelOrchestrator(
        max_workers=max_workers,
        output_dir=output_dir,
    )
    
    return await orchestrator.run(tasks)


def aggregate_results(
    worker_results: list[WorkerResult],
) -> dict[str, WorkerResult]:
    """Aggregate results from multiple workers, selecting best per task.
    
    Args:
        worker_results: All results from all workers.
        
    Returns:
        Mapping of task_id to best result.
    """
    # Group by task_id
    by_task: dict[str, list[WorkerResult]] = {}
    for result in worker_results:
        if result.task_id not in by_task:
            by_task[result.task_id] = []
        by_task[result.task_id].append(result)
    
    # Select best for each task
    aggregated = {}
    for task_id, attempts in by_task.items():
        best = select_best_patch(attempts, task_id)
        if best:
            aggregated[task_id] = best
    
    return aggregated
