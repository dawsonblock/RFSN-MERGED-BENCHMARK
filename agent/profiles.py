"""Profile system for SWE-bench agent.

Profiles control all behavioral knobs WITHOUT changing code.
Two profiles: swebench_lite (fast iteration) and swebench_verified (strict).

Same agent, different constraints.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import yaml
from pathlib import Path


@dataclass(frozen=True)
class Profile:
    """Agent profile controlling all behavioral knobs.
    
    This is the ONLY way to change agent behavior between
    SWE-bench Lite and Verified. No code changes.
    """
    name: str

    # Budgets
    max_rounds: int
    max_patch_attempts: int
    max_test_runs: int
    max_model_calls: int

    # Patch search
    patch_candidates_per_round: int
    max_files_touched: int
    max_diff_lines: int

    # Testing policy
    test_stage_cap: int  # Don't run beyond this stage
    require_full_suite_for_finalize: bool
    forbid_test_modifications: bool

    # Evidence policy
    require_citations_for_edits: bool
    localization_top_k: int

    # Safety knobs
    allow_vendor_edits: bool
    allow_ci_edits: bool
    
    # Bandit/learning knobs
    enable_bandit: bool
    enable_outcome_learning: bool


def load_profile(path: str | Path) -> Profile:
    """Load profile from YAML file.
    
    Args:
        path: Path to profile YAML
        
    Returns:
        Profile object
        
    Example:
        >>> profile = load_profile("profiles/swebench_lite.yaml")
        >>> print(profile.max_rounds)
        8
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    return Profile(
        name=cfg["name"],
        max_rounds=int(cfg["budgets"]["max_rounds"]),
        max_patch_attempts=int(cfg["budgets"]["max_patch_attempts"]),
        max_test_runs=int(cfg["budgets"]["max_test_runs"]),
        max_model_calls=int(cfg["budgets"]["max_model_calls"]),
        patch_candidates_per_round=int(cfg["patch_search"]["candidates_per_round"]),
        max_files_touched=int(cfg["patch_search"]["max_files_touched"]),
        max_diff_lines=int(cfg["patch_search"]["max_diff_lines"]),
        test_stage_cap=int(cfg["tests"]["stage_cap"]),
        require_full_suite_for_finalize=bool(cfg["tests"]["require_full_suite_for_finalize"]),
        forbid_test_modifications=bool(cfg["tests"]["forbid_test_modifications"]),
        require_citations_for_edits=bool(cfg["evidence"]["require_citations_for_edits"]),
        localization_top_k=int(cfg["localization"]["top_k"]),
        allow_vendor_edits=bool(cfg["safety"]["allow_vendor_edits"]),
        allow_ci_edits=bool(cfg["safety"]["allow_ci_edits"]),
        enable_bandit=bool(cfg.get("learning", {}).get("enable_bandit", True)),
        enable_outcome_learning=bool(cfg.get("learning", {}).get("enable_outcome_learning", True)),
    )
