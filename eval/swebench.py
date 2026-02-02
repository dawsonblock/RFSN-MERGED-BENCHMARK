"""SWE-bench dataset adapter.

This module handles loading and parsing SWE-bench tasks from various formats.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .types import SWEBenchTask
import os
from rfsn_controller.structured_logging import get_logger

logger = get_logger(__name__)

# Dataset paths (can be configured)
DATASET_PATHS = {
    "swebench_lite": "datasets/swebench_lite.jsonl",
    "swebench_verified": "datasets/swebench_verified.jsonl",
    "swebench_full": "datasets/swebench_full.jsonl",
}


def load_tasks(
    dataset: str,
    task_ids: Optional[List[str]] = None,
    max_tasks: Optional[int] = None,
    strict: bool = False,
) -> List[SWEBenchTask]:
    """Load SWE-bench tasks from dataset.
    
    Args:
        dataset: Dataset name (swebench_lite, swebench_verified, etc.)
        task_ids: Optional list of specific task IDs to load
        max_tasks: Optional maximum number of tasks to load
        strict: If True, fail hard on missing datasets (no fallbacks)
        
    Returns:
        List of SWE-bench tasks
        
    Raises:
        SystemExit: In strict mode, if dataset is missing or corrupted.
    """
    dataset_path = Path(DATASET_PATHS.get(dataset, dataset))
    # Strict mode ensures missing dataset is a hard error.
    # Environment variable override via RFSN_BENCH_STRICT.
    strict = strict or (os.environ.get("RFSN_BENCH_STRICT", "").lower() in {"1", "true", "yes"})
    
    if not dataset_path.exists():
        msg = f"Dataset file {dataset_path} not found"
        if strict:
            logger.error(f"[STRICT MODE] {msg}")
            raise SystemExit(f"FATAL: {msg}. Strict mode requires real datasets.")
        logger.warning(msg + ", creating sample tasks")
        return _create_sample_tasks(max_tasks or 5)
    
    tasks = []
    parse_errors = 0
    
    try:
        with open(dataset_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    task = _parse_task(data)
                    
                    # Filter by task_ids if specified
                    if task_ids and task.task_id not in task_ids:
                        continue
                    
                    tasks.append(task)
                    
                    # Stop if we've reached max_tasks
                    if max_tasks and len(tasks) >= max_tasks:
                        break
                        
                except json.JSONDecodeError as e:
                    parse_errors += 1
                    if strict:
                        logger.error(f"[STRICT MODE] Failed to parse line {line_num}: {e}")
                        raise SystemExit(f"FATAL: Dataset corrupted at line {line_num}")
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue
                    
    except SystemExit:
        raise  # Re-raise SystemExit
    except Exception as e:
        if strict:
            logger.error(f"[STRICT MODE] Failed to load dataset {dataset_path}: {e}")
            raise SystemExit(f"FATAL: Failed to load dataset: {e}")
        logger.error(f"Failed to load dataset {dataset_path}: {e}")
        return _create_sample_tasks(max_tasks or 5)
    
    if len(tasks) == 0 and strict:
        logger.error(f"[STRICT MODE] No valid tasks loaded from {dataset}")
        raise SystemExit(f"FATAL: No valid tasks in dataset {dataset}")
    
    logger.info(f"Loaded {len(tasks)} tasks from {dataset}")
    if parse_errors > 0:
        logger.warning(f"Encountered {parse_errors} parse errors during loading")
    return tasks


def _parse_task(data: dict) -> SWEBenchTask:
    """Parse a single task from JSON data.
    
    Args:
        data: Task data dictionary
        
    Returns:
        Parsed SWEBenchTask
    """
    # Handle different field naming conventions
    task_id = data.get("instance_id") or data.get("task_id") or ""
    repo = data.get("repo") or ""
    base_commit = data.get("base_commit") or data.get("version") or ""
    problem_statement = data.get("problem_statement") or data.get("text") or ""
    test_patch = data.get("test_patch") or data.get("patch") or ""
    hints_text = data.get("hints_text") or data.get("hints")
    
    return SWEBenchTask(
        task_id=task_id,
        repo=repo,
        base_commit=base_commit,
        problem_statement=problem_statement,
        test_patch=test_patch,
        hints_text=hints_text,
        instance_id=task_id,
        created_at=data.get("created_at", ""),
        version=data.get("version", "1.0"),
    )


def _create_sample_tasks(n: int = 5) -> List[SWEBenchTask]:
    """Create sample tasks for testing.
    
    Args:
        n: Number of sample tasks to create
        
    Returns:
        List of sample tasks
    """
    logger.info(f"Creating {n} sample tasks for testing")
    
    samples = []
    
    for i in range(n):
        task = SWEBenchTask(
            task_id=f"sample-{i+1}",
            repo="test/sample-repo",
            base_commit=f"abc123{i}",
            problem_statement=f"""
# Sample Issue {i+1}

## Problem Description
This is a sample issue for testing the SWE-bench evaluation harness.

## Expected Behavior
The function should return the correct result.

## Actual Behavior
The function returns an incorrect result or raises an exception.

## Steps to Reproduce
1. Call the function with test inputs
2. Observe the incorrect output
            """.strip(),
            test_patch=f"""
diff --git a/test_sample.py b/test_sample.py
--- a/test_sample.py
+++ b/test_sample.py
@@ -1,5 +1,9 @@
 def test_sample():
-    assert sample_function() == "old_value"
+    assert sample_function() == "new_value"
+
+def test_sample_edge_case():
+    assert sample_function("edge") == "edge_value"
             """.strip(),
            hints_text=f"Check the sample_function in sample.py line {i*10}",
            instance_id=f"sample-{i+1}",
            version="1.0",
        )
        samples.append(task)
    
    return samples


def export_results_to_swebench_format(
    results: List[dict],
    output_path: Path,
) -> None:
    """Export evaluation results in SWE-bench format for scoring.
    
    Args:
        results: List of evaluation result dictionaries
        output_path: Path to write results file
    """
    with open(output_path, "w") as f:
        for result in results:
            # Convert to SWE-bench format
            swebench_result = {
                "instance_id": result["task_id"],
                "model_patch": result.get("final_patch", ""),
                "model_name_or_path": "rfsn-controller",
                "success": result.get("success", False),
            }
            f.write(json.dumps(swebench_result) + "\n")
    
    logger.info(f"Exported {len(results)} results to {output_path}")


def load_results_from_file(results_path: Path) -> List[dict]:
    """Load evaluation results from JSONL file.
    
    Args:
        results_path: Path to results file
        
    Returns:
        List of result dictionaries
    """
    results = []
    
    with open(results_path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    return results
