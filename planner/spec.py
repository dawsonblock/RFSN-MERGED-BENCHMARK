"""Planner specification types."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class RepairStep:
    """A single step in a repair plan."""
    intent: str
    files: List[str]
    hypothesis: str


@dataclass
class Plan:
    """A complete repair plan with steps and metadata."""
    task_id: str
    bug_summary: str
    steps: List[RepairStep]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
