"""Evaluation harness for SWE-bench tasks.

This module implements the outer task loop that SWE-bench scores:
1. Ingest issue (problem statement + repo snapshot)
2. Localize likely files/regions
3. Plan micro-steps
4. Propose action
5. Gate validates
6. Execute
7. Evaluate
8. Store outcome + update priors
9. Stop when solved or budget exhausted
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess  # legacy; avoid direct use (see _run_argv)
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.loop import run_episode
from agent.profiles import Profile, load_profile
from agent.types import AgentState, Phase, Proposal
from memory.log import append_event
from rfsn_controller.structured_logging import get_logger

logger = get_logger(__name__)


class TaskStatus(Enum):
    """Machine-readable status codes for task results.
    
    Used for generating results.json with consistent, parseable status codes.
    """
    
    PASS = "PASS"                            # All tests pass
    FAIL_TESTS = "FAIL_TESTS"                # Tests executed but failed
    REJECTED_BY_GATE = "REJECTED_BY_GATE"    # Plan rejected by PlanGate
    SECURITY_VIOLATION = "SECURITY_VIOLATION"  # Security constraint violated
    ERROR = "ERROR"                          # Unhandled exception
    TIMEOUT = "TIMEOUT"                      # Task exceeded time limit
    INVALID = "INVALID"                      # Tests didn't run / setup failed


def _run_argv(workdir: Path, argv: list[str], *, timeout_sec: int) -> tuple[int, str, str]:
    """Hardened argv-only execution for eval harness.

    The evaluation harness must never use shell strings. This wrapper routes
    through rfsn_controller.exec_utils.safe_run which enforces:
      - argv-only (no shell=True)
      - no shell wrappers (sh -c / bash -c)
      - allowlist checks
    """
    try:
        from rfsn_controller.exec_utils import safe_run
        r = safe_run(argv=argv, cwd=str(workdir), timeout_sec=timeout_sec)
        return (0 if r.ok else int(r.exit_code), r.stdout or "", r.stderr or "")
    except Exception as e:
        return (1, "", f"safe_run failed: {e}")


def setup_repo(task: SWEBenchTask, workdir: Path) -> bool:
    """Clone and checkout repository for a SWE-bench task.
    
    Args:
        task: The SWE-bench task containing repo and commit info
        workdir: Directory to clone into
        
    Returns:
        True if setup succeeded, False otherwise
    """
    repo_url = f"https://github.com/{task.repo}.git"
    
    # Clean up existing directory if it exists
    if workdir.exists():
        shutil.rmtree(workdir)
    
    workdir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Cloning {task.repo} to {workdir}")
    
    try:
        # Clone with minimal history for speed
        rc, out, err = _run_argv(Path.cwd(), ["git", "clone", "--depth", "100", repo_url, str(workdir)], timeout_sec=300)
        if rc != 0:
            logger.error(f"Clone failed: {err}")
            return False
        
        # Fetch the specific commit if not in shallow clone
        _run_argv(workdir, ["git", "fetch", "--depth", "100", "origin", task.base_commit], timeout_sec=120)
        
        # Checkout the base commit
        rc, out, err = _run_argv(workdir, ["git", "checkout", task.base_commit], timeout_sec=60)
        if rc != 0:
            logger.warning(f"Checkout specific commit failed: {err}")
            # Try without specific commit - use HEAD
            logger.info("Using HEAD instead of specific commit")
        else:
            logger.info(f"Checked out commit {task.base_commit[:8]}")
        
        return True
        
    except subprocess.TimeoutExpired:
        logger.error("Repository setup timed out")
        return False
    except Exception as e:
        logger.error(f"Repository setup failed: {e}")
        return False


from .types import SWEBenchTask

# SWEBenchTask is now imported from .types



@dataclass
class EvalResult:
    """Result of evaluating a single task."""
    
    task_id: str
    success: bool
    resolution_time: float
    steps_taken: int
    patches_tried: int
    tests_passed: int
    tests_failed: int
    
    # Machine-readable status
    status: TaskStatus = TaskStatus.ERROR
    
    # Gate and security tracking
    gate_rejections: int = 0
    security_violations: list[str] = field(default_factory=list)
    verification_ran: bool = False
    
    # Detailed tracking
    localization_hits: list[dict[str, Any]] = field(default_factory=list)
    patch_history: list[dict[str, Any]] = field(default_factory=list)
    test_history: list[dict[str, Any]] = field(default_factory=list)
    error_message: str | None = None
    final_patch: str | None = None
    
    # Resource usage
    llm_calls: int = 0
    llm_tokens: int = 0
    subprocess_calls: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "success": self.success,
            "status": self.status.value,
            "resolution_time": self.resolution_time,
            "steps_taken": self.steps_taken,
            "patches_tried": self.patches_tried,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "gate_rejections": self.gate_rejections,
            "security_violations": self.security_violations,
            "verification_ran": self.verification_ran,
            "localization_hits": self.localization_hits,
            "patch_history": self.patch_history,
            "test_history": self.test_history,
            "error_message": self.error_message,
            "final_patch": self.final_patch,
            "llm_calls": self.llm_calls,
            "llm_tokens": self.llm_tokens,
            "subprocess_calls": self.subprocess_calls,
        }


@dataclass
class EvalConfig:
    """Configuration for evaluation run."""
    
    # Task selection
    dataset: str = "swebench_lite"  # or "swebench_verified"
    task_ids: Optional[List[str]] = None
    max_tasks: Optional[int] = None
    
    # Profile
    profile_name: str = "swebench_lite"
    
    # Resource limits
    max_time_per_task: int = 1800  # 30 minutes
    max_steps_per_task: int = 50
    max_patches_per_task: int = 20
    
    # Directories
    work_dir: Path = Path("./eval_runs")
    results_dir: Path = Path("./eval_results")
    cache_dir: Path = Path("./eval_cache")
    
    # Parallelism
    parallel_tasks: int = 1
    
    # Logging
    verbose: bool = True
    save_artifacts: bool = True


class EvalRunner:
    """Main evaluation runner for SWE-bench tasks."""
    
    def __init__(self, config: EvalConfig):
        """Initialize evaluation runner.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.profile = load_profile(config.profile_name)
        
        # Create directories
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized EvalRunner with profile {config.profile_name}")
    
    async def run_task(self, task: SWEBenchTask) -> EvalResult:
        """Run a single SWE-bench task.
        
        This is the outer task loop that SWE-bench scores:
        1. Ingest issue + repo snapshot
        2. Localize likely files/regions
        3. Plan micro-steps
        4. Propose action
        5. Gate validates
        6. Execute
        7. Evaluate
        8. Store outcome + update priors
        9. Stop when solved or budget exhausted
        
        Args:
            task: SWE-bench task to run
            
        Returns:
            Evaluation result
        """
        start_time = time.time()
        
        logger.info(f"Starting task {task.task_id}")
        logger.info(f"Repo: {task.repo}, commit: {task.base_commit}")
        
        # Setup repository - clone and checkout
        workdir = self.config.work_dir / task.task_id
        if not setup_repo(task, workdir):
            return EvalResult(
                task_id=task.task_id,
                success=False,
                resolution_time=time.time() - start_time,
                steps_taken=0,
                patches_tried=0,
                tests_passed=0,
                tests_failed=0,
                error_message="Failed to setup repository",
            )
        
        # Initialize agent state
        from agent.types import RepoFingerprint, BudgetState
        
        repo = RepoFingerprint(
            repo_id=f"{task.repo}@{task.base_commit}",
            commit_sha=task.base_commit,
            workdir=str(workdir),
            language="python",
        )
        
        budget = BudgetState(
            max_rounds=self.profile.max_rounds,
            max_patch_attempts=self.profile.max_patch_attempts,
            max_test_runs=self.profile.max_test_runs,
            max_model_calls=self.profile.max_model_calls,
        )
        
        state = AgentState(
            task_id=task.task_id,
            repo=repo,
            phase=Phase.INGEST,
            budget=budget,
            notes={
                "problem_statement": task.problem_statement,
                "test_patch": task.test_patch,
            },
        )
        
        # Log task start
        append_event(state, {
            "event": "task_start",
            "task_id": task.task_id,
            "repo": task.repo,
            "base_commit": task.base_commit,
        })
        
        try:
            # Import DeepSeek agent functions
            from agent.deepseek_agent import propose, gate, execute
            
            # Run agent episode (the inner loop)
            final_state = run_episode(
                profile=self.profile,
                state=state,
                propose_fn=propose,
                gate_fn=gate,
                exec_fn=execute,
            )
            
            # Calculate metrics
            resolution_time = time.time() - start_time
            
            # Build result
            result = EvalResult(
                task_id=task.task_id,
                success=(final_state.phase == Phase.DONE and 
                        final_state.notes.get("solved", False)),
                resolution_time=resolution_time,
                steps_taken=final_state.budget.round_idx,
                patches_tried=final_state.budget.patch_attempts,
                tests_passed=final_state.notes.get("tests_passed", 0),
                tests_failed=final_state.notes.get("tests_failed", 0),
                localization_hits=final_state.localization_hits,
                patch_history=final_state.notes.get("patch_history", []),
                test_history=final_state.notes.get("test_history", []),
                llm_calls=final_state.budget.model_calls,
                llm_tokens=final_state.notes.get("llm_tokens", 0),
                subprocess_calls=final_state.notes.get("subprocess_calls", 0),
            )
            
            logger.info(f"Task {task.task_id} completed: "
                       f"success={result.success}, "
                       f"steps={result.steps_taken}, "
                       f"time={resolution_time:.1f}s")
            
            # Log task end
            append_event(state, {
                "event": "task_complete",
                "task_id": task.task_id,
                "success": result.success,
                "resolution_time": resolution_time,
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed with error: {e}")
            
            # Log failure
            append_event(state, {
                "event": "task_error",
                "task_id": task.task_id,
                "error": str(e),
            })
            
            # Return failure result
            resolution_time = time.time() - start_time
            return EvalResult(
                task_id=task.task_id,
                success=False,
                resolution_time=resolution_time,
                steps_taken=state.budget.round_idx,
                patches_tried=0,
                tests_passed=0,
                tests_failed=0,
                error_message=str(e),
            )
    
    async def run_batch(self, tasks: List[SWEBenchTask]) -> List[EvalResult]:
        """Run a batch of tasks.
        
        Args:
            tasks: List of tasks to run
            
        Returns:
            List of evaluation results
        """
        results = []
        
        if self.config.parallel_tasks > 1:
            # Parallel execution
            semaphore = asyncio.Semaphore(self.config.parallel_tasks)
            
            async def run_with_semaphore(task: SWEBenchTask) -> EvalResult:
                async with semaphore:
                    return await self.run_task(task)
            
            tasks_coros = [run_with_semaphore(task) for task in tasks]
            raw_results = await asyncio.gather(*tasks_coros, return_exceptions=True)
            
            # Convert exceptions to failed results
            for i, result in enumerate(raw_results):
                if isinstance(result, BaseException):
                    results.append(EvalResult(
                        task_id=tasks[i].task_id,
                        success=False,
                        resolution_time=0.0,
                        steps_taken=0,
                        patches_tried=0,
                        tests_passed=0,
                        tests_failed=0,
                        error_message=str(result),
                    ))
                else:
                    results.append(result)
        else:
            # Serial execution
            for task in tasks:
                result = await self.run_task(task)
                results.append(result)
        
        return results
    
    def save_results(self, results: List[EvalResult], run_id: str):
        """Save evaluation results to disk.
        
        Args:
            results: List of evaluation results
            run_id: Unique identifier for this run
        """
        # Save individual results
        results_file = self.config.results_dir / f"{run_id}_results.jsonl"
        with open(results_file, "w") as f:
            for result in results:
                f.write(json.dumps(result.to_dict()) + "\n")
        
        # Save summary
        summary = {
            "run_id": run_id,
            "total_tasks": len(results),
            "successful_tasks": sum(1 for r in results if r.success),
            "failed_tasks": sum(1 for r in results if not r.success),
            "total_time": sum(r.resolution_time for r in results),
            "avg_steps": sum(r.steps_taken for r in results) / len(results),
            "avg_patches": sum(r.patches_tried for r in results) / len(results),
            "total_llm_calls": sum(r.llm_calls for r in results),
            "total_llm_tokens": sum(r.llm_tokens for r in results),
        }
        
        summary_file = self.config.results_dir / f"{run_id}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved results to {results_file}")
        logger.info(f"Success rate: {summary['successful_tasks']}/{summary['total_tasks']}")


async def run_eval(config: EvalConfig) -> List[EvalResult]:
    """Run evaluation with given configuration.
    
    Args:
        config: Evaluation configuration
        
    Returns:
        List of evaluation results
    """
    runner = EvalRunner(config)
    
    # Load tasks (placeholder - will be implemented in eval/swebench.py)
    from .swebench import load_tasks
    tasks = load_tasks(config.dataset, config.task_ids, config.max_tasks)
    
    logger.info(f"Loaded {len(tasks)} tasks from {config.dataset}")
    
    # Run tasks
    results = await runner.run_batch(tasks)
    
    # Save results
    run_id = f"{config.dataset}_{int(time.time())}"
    runner.save_results(results, run_id)
    
    return results
