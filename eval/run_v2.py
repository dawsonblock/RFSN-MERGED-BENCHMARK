"""SWE-bench evaluation harness v2 - unified architecture.

This module routes evaluation through:
1. Dataset loader (strict mode)
2. Episode runner (single authority loop)
3. Gate adapter (PlanGate only)
4. Upstream intelligence (propose_v2)

The gate is untouched. All intelligence is upstream.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Any

import agent.llm_patcher
import eval.dataset_loader
import eval.strictness
import orchestrator.episode_runner

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Machine-readable status codes."""
    PASS = "PASS"
    FAIL_TESTS = "FAIL_TESTS"
    REJECTED_BY_GATE = "REJECTED_BY_GATE"
    SECURITY_VIOLATION = "SECURITY_VIOLATION"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    INVALID = "INVALID"


@dataclass
class EvalResult:
    """Result of evaluating a single task."""
    instance_id: str
    passed: bool
    attempts: int
    runtime: float
    status: TaskStatus
    test_output_tail: str = ""
    gate_rejections: int = 0
    security_violations: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "passed": self.passed,
            "attempts": self.attempts,
            "runtime": self.runtime,
            "status": self.status.value,
            "test_output_tail": self.test_output_tail[-500:],
            "gate_rejections": self.gate_rejections,
            "security_violations": self.security_violations,
        }


def _process_task_worker(task_obj: Any, dataset_name: str, model_name: str, max_attempts: int):
    """Worker function for parallel task execution.
    Must be at module level for pickling.
    """
    t0 = time.time()
    try:
        llm_patch_fn = agent.llm_patcher.get_llm_patch_fn(model_name)
        
        task_dict = task_obj.to_dict()
        repo_url = f"https://github.com/{task_obj.repo}.git"
        
        run_result = orchestrator.episode_runner.run_one_task(
            task_dict,
            repo_url,
            llm_patch_fn=llm_patch_fn,
            max_attempts=max_attempts,
        )
        
        runtime = time.time() - t0
        if run_result.invalid:
            status = TaskStatus.INVALID
        elif run_result.passed:
            status = TaskStatus.PASS
        else:
            status = TaskStatus.FAIL_TESTS
        
        return EvalResult(
            instance_id=task_obj.instance_id,
            passed=run_result.passed,
            attempts=run_result.attempts,
            runtime=runtime,
            status=status,
            test_output_tail=run_result.test_output[-2000:] if run_result.test_output else "",
            gate_rejections=getattr(run_result, "gate_rejections", 0),
            security_violations=getattr(run_result, "security_violations", 0),
        )
    except Exception as e:
        import traceback
        logging.error("Task %s failed: %s\n%s", task_obj.instance_id, e, traceback.format_exc())
        return EvalResult(
            instance_id=task_obj.instance_id,
            passed=False,
            attempts=0,
            runtime=time.time() - t0,
            status=TaskStatus.ERROR,
            test_output_tail=str(e),
        )


def run_eval(
    dataset_name: str = "swebench_lite.jsonl",
    task_ids: list[str] | None = None,
    max_tasks: int | None = None,
    llm_patch_fn: Callable | None = None,
    max_attempts: int = 6,
    results_dir: str = "./eval_results",
    use_parallel: bool = False,
    max_workers: int | None = None,
) -> list[EvalResult]:
    """
    Run SWE-bench evaluation.
    
    Args:
        dataset_name: Dataset file name
        task_ids: Optional list of specific task IDs
        max_tasks: Maximum tasks to run
        llm_patch_fn: Patch generation function
        max_attempts: Max attempts per task
        results_dir: Directory to save results
        
    Returns:
        List of EvalResult objects
    """
    model_name = os.environ.get("RFSN_MODEL", "deepseek")
    if llm_patch_fn is None:
        llm_patch_fn = agent.llm_patcher.get_llm_patch_fn(model_name)
    
    # Load tasks
    if task_ids:
        # Filter None values from load_task_by_id
        tasks_raw = [eval.dataset_loader.load_task_by_id(dataset_name, tid) for tid in task_ids]
        tasks = [t for t in tasks_raw if t is not None]
    else:
        tasks = list(eval.dataset_loader.iter_tasks(dataset_name))
    
    if max_tasks:
        tasks = tasks[:max_tasks]
    
    if not tasks:
        if eval.strictness.strict_benchmark_mode():
            raise RuntimeError("No tasks to run. Check dataset file exists.")
        logger.warning("No tasks found. Strict mode is OFF.")
        return []
    
    model_name = os.environ.get("RFSN_MODEL", "deepseek")
    logger.info("Running %d tasks from %s (parallel=%s, model=%s)", len(tasks), dataset_name, use_parallel, model_name)
    
    results: list[EvalResult] = []
    passed_count = 0
    
    if use_parallel:
        workers = max_workers or min(multiprocessing.cpu_count(), len(tasks))
        logger.info("ENVIRONMENT: Using ProcessPoolExecutor with %d workers", workers)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_task = {
                executor.submit(_process_task_worker, t, dataset_name, model_name, max_attempts): t 
                for t in tasks
            }
            for future in as_completed(future_to_task):
                res = future.result()
                results.append(res)
                if res.passed:
                    passed_count += 1
                logger.info("Task %s finished: %s (total passed: %d)", res.instance_id, res.status.value, passed_count)
    else:
        for i, task_obj in enumerate(tasks):
            logger.info("[%d/%d] %s", i + 1, len(tasks), task_obj.instance_id)
            t0 = time.time()
            try:
                task_dict = task_obj.to_dict()
                repo_url = f"https://github.com/{task_obj.repo}.git"
                run_result = orchestrator.episode_runner.run_one_task(
                    task_dict,
                    repo_url,
                    llm_patch_fn=llm_patch_fn,
                    max_attempts=max_attempts,
                )
                runtime = time.time() - t0
                if run_result.invalid:
                    status = TaskStatus.INVALID
                elif run_result.passed:
                    status = TaskStatus.PASS
                else:
                    status = TaskStatus.FAIL_TESTS
                
                res = EvalResult(
                    instance_id=task_obj.instance_id,
                    passed=run_result.passed,
                    attempts=run_result.attempts,
                    runtime=runtime,
                    status=status,
                    test_output_tail=run_result.test_output[-2000:] if run_result.test_output else "",
                    gate_rejections=getattr(run_result, "gate_rejections", 0),
                    security_violations=getattr(run_result, "security_violations", 0),
                )
            except Exception as e:
                logger.error("Task %s failed: %s", task_obj.instance_id, e)
                res = EvalResult(
                    instance_id=task_obj.instance_id,
                    passed=False,
                    attempts=0,
                    runtime=time.time() - t0,
                    status=TaskStatus.ERROR,
                    test_output_tail=str(e),
                )
            
            results.append(res)
            if res.passed:
                passed_count += 1
            logger.info("  -> %s (%.1fs)", res.status.value, res.runtime)
    
    # Summary
    total = len(results)
    pass_rate = (passed_count / total * 100) if total > 0 else 0
    logger.info("=" * 50)
    logger.info("Results: %d/%d passed (%.1f%%)", passed_count, total, pass_rate)
    
    # Save results
    os.makedirs(results_dir, exist_ok=True)
    run_id = f"eval_{int(time.time())}"
    results_path = os.path.join(results_dir, f"{run_id}.json")
    
    with open(results_path, "w") as f:
        json.dump({
            "run_id": run_id,
            "dataset": dataset_name,
            "total_tasks": total,
            "passed": passed_count,
            "pass_rate": pass_rate,
            "results": [r.to_dict() for r in results],
        }, f, indent=2)
    
    logger.info("Results saved to: %s", results_path)
    
    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="RFSN SWE-bench Evaluation")
    parser.add_argument("--dataset", default="swebench_lite.jsonl", help="Dataset file name")
    parser.add_argument("--task-id", action="append", dest="task_ids", help="Specific task ID(s)")
    parser.add_argument("--max-tasks", type=int, help="Maximum tasks to run")
    parser.add_argument("--max-attempts", type=int, default=6, help="Max attempts per task")
    parser.add_argument("--parallel", action="store_true", help="Run tasks in parallel")
    parser.add_argument("--max-workers", type=int, default=None, help="Max parallel workers")
    parser.add_argument("--results-dir", default="./eval_results", help="Results directory")
    parser.add_argument("--model", default="deepseek", help="Model to use (deepseek/gemini)")
    parser.add_argument("--non-strict", action="store_true", help="Disable strict mode")
    
    args = parser.parse_args()
    
    if args.non_strict:
        os.environ["RFSN_STRICT_BENCH"] = "0"
    
    results = run_eval(
        dataset_name=args.dataset,
        task_ids=args.task_ids,
        max_tasks=args.max_tasks,
        llm_patch_fn = agent.llm_patcher.get_llm_patch_fn(args.model),
        max_attempts=args.max_attempts,
        results_dir=args.results_dir,
        use_parallel=args.parallel,
        max_workers=args.max_workers,
    )
    
    # Exit with failure if any task failed in strict mode
    if eval.strictness.strict_benchmark_mode():
        passed = sum(1 for r in results if r.passed)
        if passed < len(results):
            sys.exit(1)


if __name__ == "__main__":
    main()
