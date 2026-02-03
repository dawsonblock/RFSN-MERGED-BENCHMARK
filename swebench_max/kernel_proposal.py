from dataclasses import dataclass
from typing import Any, Dict

from swebench_max.diff_stats import compute_diff_stats

@dataclass(frozen=True)
class KernelProposal:
    action_allowed: bool
    patch: str
    diff_stats: Dict[str, Any]

def to_kernel_proposal(patch: str, allow: bool = True) -> KernelProposal:
    diff = compute_diff_stats(patch)
    return KernelProposal(
        action_allowed=allow,
        patch=patch,
        diff_stats={"files_changed": diff.files_changed, "lines_changed": diff.lines_changed, "paths": diff.paths},
    )
