"""Common types for evaluation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class SWEBenchTask:
    """A single SWE-bench task instance."""
    
    task_id: str
    repo: str
    base_commit: str
    problem_statement: str
    test_patch: str
    hints_text: Optional[str] = None
    
    # Metadata
    instance_id: str = ""
    created_at: str = ""
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
            "test_patch": self.test_patch,
            "hints_text": self.hints_text,
            "instance_id": self.instance_id,
            "created_at": self.created_at,
            "version": self.version,
        }
