from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass(frozen=True)
class Candidate:
    key: str
    patch: str  # unified diff
    meta: Dict[str, Any]

@dataclass(frozen=True)
class DiffStats:
    files_changed: int
    lines_changed: int
    paths: List[str]
