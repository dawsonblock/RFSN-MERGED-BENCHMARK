from typing import List
from swebench_max.diff_stats import compute_diff_stats

def forbid_paths_filter(patch: str, forbid_prefixes: List[str]) -> bool:
    diff = compute_diff_stats(patch)
    for p in diff.paths:
        for pref in forbid_prefixes:
            if p.startswith(pref):
                return False
    return True
