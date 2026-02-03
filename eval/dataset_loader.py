"""SWE-bench dataset loader with strict mode enforcement."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, Optional, Dict, Any
import json
import os

from .strictness import strict_benchmark_mode


@dataclass
class SWEBenchTask:
    """A single SWE-bench task."""
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    test_patch: str
    hints: Optional[Dict[str, Any]] = None
    fail_to_pass: Optional[list[str]] = None
    pass_to_pass: Optional[list[str]] = None
    patch: Optional[str] = None  # Gold patch for comparison
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
            "test_patch": self.test_patch,
            "hints": self.hints,
            "FAIL_TO_PASS": self.fail_to_pass,
            "PASS_TO_PASS": self.pass_to_pass,
            "patch": self.patch,
        }



def _required_path(name: str) -> str:
    """Get dataset path, enforcing strict mode."""
    p = os.path.join("datasets", name)
    if not os.path.exists(p):
        if strict_benchmark_mode():
            raise FileNotFoundError(
                f"Missing dataset file: {p}. Strict mode is ON. "
                f"Place the dataset file under datasets/. "
                f"Set RFSN_STRICT_BENCH=0 for local development."
            )
    return p


def iter_tasks(dataset_name: str) -> Iterator[SWEBenchTask]:
    """
    Iterate over tasks in a SWE-bench dataset.
    
    Args:
        dataset_name: 'swebench_lite.jsonl' | 'swebench_verified.jsonl' | 'swebench_full.jsonl'
        
    Yields:
        SWEBenchTask objects
        
    Raises:
        FileNotFoundError: If dataset missing and strict mode is ON
    """
    path = _required_path(dataset_name)
    if not os.path.exists(path):
        return iter(())  # non-strict local dev only

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            
            # Parse FAIL_TO_PASS and PASS_TO_PASS - they're JSON strings in the JSONL
            fail_to_pass_raw = obj.get("FAIL_TO_PASS")
            pass_to_pass_raw = obj.get("PASS_TO_PASS")
            
            fail_to_pass = None
            if fail_to_pass_raw:
                if isinstance(fail_to_pass_raw, str):
                    try:
                        fail_to_pass = json.loads(fail_to_pass_raw)
                    except json.JSONDecodeError:
                        fail_to_pass = [fail_to_pass_raw]  # Single test
                else:
                    fail_to_pass = fail_to_pass_raw  # Already a list
            
            pass_to_pass = None
            if pass_to_pass_raw:
                if isinstance(pass_to_pass_raw, str):
                    try:
                        pass_to_pass = json.loads(pass_to_pass_raw)
                    except json.JSONDecodeError:
                        pass_to_pass = [pass_to_pass_raw]  # Single test
                else:
                    pass_to_pass = pass_to_pass_raw  # Already a list
            
            yield SWEBenchTask(
                instance_id=obj.get("instance_id") or obj.get("id") or "unknown",
                repo=obj["repo"],
                base_commit=obj["base_commit"],
                problem_statement=obj.get("problem_statement") or obj.get("description") or "",
                test_patch=obj.get("test_patch") or "",
                hints=obj.get("hints") or obj.get("metadata"),
                fail_to_pass=fail_to_pass,
                pass_to_pass=pass_to_pass,
                patch=obj.get("patch"),  # Gold patch
            )


def load_task_by_id(dataset_name: str, instance_id: str) -> Optional[SWEBenchTask]:
    """Load a specific task by instance_id."""
    for task in iter_tasks(dataset_name):
        if task.instance_id == instance_id:
            return task
    return None


def count_tasks(dataset_name: str) -> int:
    """Count tasks in a dataset."""
    return sum(1 for _ in iter_tasks(dataset_name))
