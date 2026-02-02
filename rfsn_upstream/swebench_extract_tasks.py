"""SWE-bench Task Extractor.

Converts SWE-bench dataset from HuggingFace to the JSONL format
expected by swebench_runner.py.

INVARIANTS:
1. Output is deterministic (same input = same output)
2. Each line in output is valid JSON
3. All required fields are present
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TaskRecord:
    """A single task record for JSONL output."""
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    test_patch: str = ""
    test_cmd: str = "pytest"
    hints_text: str = ""
    metadata: dict[str, Any] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
            "test_patch": self.test_patch,
            "test_cmd": self.test_cmd,
            "hints_text": self.hints_text,
            "metadata": self.metadata or {},
        }


def load_from_huggingface(
    dataset_name: str = "princeton-nlp/SWE-bench",
    split: str = "test",
    limit: int = 0,
) -> Iterator[TaskRecord]:
    """Load tasks from HuggingFace dataset.
    
    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split (test, dev, train)
        limit: Maximum records to load (0 = all)
        
    Yields:
        TaskRecord for each entry
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "datasets library not installed. "
            "Install with: pip install datasets"
        ) from e
    
    dataset = load_dataset(dataset_name, split=split)
    
    for count, entry in enumerate(dataset):
        # Extract fields from SWE-bench format
        instance_id = entry.get("instance_id", "")
        repo = entry.get("repo", "")
        
        # Build GitHub URL from repo name
        repo_url = f"https://github.com/{repo}" if "/" in repo and not repo.startswith("http") else repo
        
        yield TaskRecord(
            instance_id=instance_id,
            repo=repo_url,
            base_commit=entry.get("base_commit", ""),
            problem_statement=entry.get("problem_statement", ""),
            test_patch=entry.get("test_patch", ""),
            test_cmd="pytest",  # Default for Python repos
            hints_text=entry.get("hints_text", ""),
            metadata={
                "created_at": entry.get("created_at", ""),
                "version": entry.get("version", ""),
                "environment_setup_commit": entry.get("environment_setup_commit", ""),
            },
        )
        
        if limit > 0 and count >= limit - 1:
            break


def load_from_jsonl(path: Path, limit: int = 0) -> Iterator[TaskRecord]:
    """Load tasks from existing JSONL file.
    
    Args:
        path: Path to JSONL file
        limit: Maximum records to load (0 = all)
        
    Yields:
        TaskRecord for each entry
    """
    count = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            
            data = json.loads(line)
            yield TaskRecord(
                instance_id=data.get("instance_id", ""),
                repo=data.get("repo", ""),
                base_commit=data.get("base_commit", ""),
                problem_statement=data.get("problem_statement", ""),
                test_patch=data.get("test_patch", ""),
                test_cmd=data.get("test_cmd", "pytest"),
                hints_text=data.get("hints_text", ""),
                metadata=data.get("metadata"),
            )
            
            count += 1
            if limit > 0 and count >= limit:
                break


def load_from_json(path: Path, limit: int = 0) -> Iterator[TaskRecord]:
    """Load tasks from JSON array file.
    
    Args:
        path: Path to JSON file with array of tasks
        limit: Maximum records to load (0 = all)
        
    Yields:
        TaskRecord for each entry
    """
    with open(path) as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        data = [data]
    
    for i, entry in enumerate(data):
        if limit > 0 and i >= limit:
            break
        
        repo = entry.get("repo", "")
        if "/" in repo and not repo.startswith("http"):
            repo = f"https://github.com/{repo}"
        
        yield TaskRecord(
            instance_id=entry.get("instance_id", f"task_{i}"),
            repo=repo,
            base_commit=entry.get("base_commit", ""),
            problem_statement=entry.get("problem_statement", ""),
            test_patch=entry.get("test_patch", ""),
            test_cmd=entry.get("test_cmd", "pytest"),
            hints_text=entry.get("hints_text", ""),
            metadata=entry.get("metadata"),
        )


def extract_tasks(
    source: str,
    output: Path,
    limit: int = 0,
    split: str = "test",
) -> int:
    """Extract tasks to JSONL file.
    
    Args:
        source: HuggingFace dataset name or path to JSON/JSONL file
        output: Output JSONL path
        limit: Maximum tasks (0 = all)
        split: Dataset split (for HuggingFace)
        
    Returns:
        Number of tasks extracted
    """
    # Determine source type
    source_path = Path(source)
    
    if source_path.exists():
        # Load from file
        if source_path.suffix == ".jsonl":
            tasks = load_from_jsonl(source_path, limit)
        elif source_path.suffix == ".json":
            tasks = load_from_json(source_path, limit)
        else:
            raise ValueError(f"Unknown file format: {source_path.suffix}")
    else:
        # Load from HuggingFace
        tasks = load_from_huggingface(source, split, limit)
    
    # Write to JSONL
    count = 0
    with open(output, "w") as f:
        for task in tasks:
            f.write(json.dumps(task.to_dict()) + "\n")
            count += 1
    
    return count


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Extract SWE-bench tasks to JSONL format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from HuggingFace (default)
  python -m rfsn_upstream.swebench_extract_tasks --output tasks.jsonl

  # Extract limited set for testing
  python -m rfsn_upstream.swebench_extract_tasks --output tasks.jsonl --limit 10

  # Extract from local JSON file
  python -m rfsn_upstream.swebench_extract_tasks --source data.json --output tasks.jsonl

  # Extract specific split
  python -m rfsn_upstream.swebench_extract_tasks --split dev --output dev_tasks.jsonl
""",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="princeton-nlp/SWE-bench",
        help="HuggingFace dataset name or path to JSON/JSONL file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum tasks to extract (0 = all)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "dev", "train"],
        help="Dataset split (for HuggingFace, default: test)",
    )
    
    args = parser.parse_args()
    
    try:
        count = extract_tasks(
            source=args.source,
            output=args.output,
            limit=args.limit,
            split=args.split,
        )
        print(f"Extracted {count} tasks to {args.output}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
