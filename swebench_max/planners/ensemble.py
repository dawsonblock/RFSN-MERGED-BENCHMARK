from dataclasses import dataclass
from typing import Dict, Any, List

from swebench_max.llm.client import LLMClient, LLMConfig
from swebench_max.prompts import PLANNER_SYSTEM, PLANNER_USER, SKEPTIC_SYSTEM, SKEPTIC_USER

@dataclass(frozen=True)
class PlannerSpec:
    name: str
    weight: float
    llm: LLMConfig

def _render_planner_user(issue: Dict[str, Any], context: str, failures: str, forbid_prefixes: List[str]) -> str:
    return PLANNER_USER.format(
        problem_statement=issue.get("problem_statement", ""),
        context=context,
        failures=failures,
        forbid_prefixes=", ".join(forbid_prefixes),
    )

def _render_skeptic_user(issue: Dict[str, Any], patch: str, failures: str, forbid_prefixes: List[str]) -> str:
    return SKEPTIC_USER.format(
        problem_statement=issue.get("problem_statement", ""),
        patch=patch,
        failures=failures,
        forbid_prefixes=", ".join(forbid_prefixes),
    )

def propose_patches(
    issue: Dict[str, Any],
    planners: List[PlannerSpec],
    context: str,
    failures: str,
    forbid_prefixes: List[str],
    per_planner: int = 4,
) -> List[str]:
    patches: List[str] = []
    user = _render_planner_user(issue, context, failures, forbid_prefixes)

    for ps in planners:
        client = LLMClient(ps.llm)
        # cheap diversity: bump temp slightly per sample
        for i in range(per_planner):
            txt = client.complete(PLANNER_SYSTEM, user)
            if txt and "diff --git" in txt:
                patches.append(txt.strip() + "\n")
    return patches

def skeptic_rewrite(
    issue: Dict[str, Any],
    skeptic_llm: LLMConfig,
    patch: str,
    failures: str,
    forbid_prefixes: List[str],
) -> str:
    client = LLMClient(skeptic_llm)
    user = _render_skeptic_user(issue, patch, failures, forbid_prefixes)
    txt = client.complete(SKEPTIC_SYSTEM, user)
    if txt and "diff --git" in txt:
        return txt.strip() + "\n"
    return patch
