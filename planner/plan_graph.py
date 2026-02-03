from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class PlanNode:
    name: str
    allowed_next: List[str]

PLAN_GRAPH = {
    "analyze": PlanNode("analyze", ["search"]),
    "search": PlanNode("search", ["patch", "abort"]),
    "patch": PlanNode("patch", ["test"]),
    "test": PlanNode("test", ["commit", "rollback"]),
    "commit": PlanNode("commit", []),
    "rollback": PlanNode("rollback", [])
}

def validate_transition(current: str, nxt: str) -> bool:
    return nxt in PLAN_GRAPH[current].allowed_next
