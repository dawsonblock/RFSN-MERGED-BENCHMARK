import os
import subprocess
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

from swebench_max.candidate import Candidate
from swebench_max.diff_stats import compute_diff_stats
from swebench_max.targeted_tests import targeted_tests
from swebench_max.static_risk import static_risk_score

@dataclass
class EvalResult:
    candidate_key: str
    ok_apply: bool
    ok_compile: bool
    ok_unit_smoke: bool
    targeted_passed: int
    targeted_failed: int
    score: float
    diff_stats: dict
    notes: List[str]

def _run_shell(cmd: str, cwd: str, timeout: int = 900) -> Tuple[bool, str]:
    p = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True, timeout=timeout)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    return (p.returncode == 0), out

def _apply_patch(repo: str, patch: str) -> bool:
    # Use git apply, no commit.
    p = subprocess.run(["git", "apply", "--whitespace=nowarn"], cwd=repo, input=patch, text=True, capture_output=True)
    return p.returncode == 0

def _reset(repo: str):
    subprocess.run(["git", "reset", "--hard"], cwd=repo, capture_output=True)
    subprocess.run(["git", "clean", "-fd"], cwd=repo, capture_output=True)

def evaluate_candidate(
    repo_root: str,
    cand: Candidate,
    cfg: Dict[str, Any],
) -> EvalResult:
    notes: List[str] = []
    _reset(repo_root)

    ok_apply = _apply_patch(repo_root, cand.patch)
    if not ok_apply:
        return EvalResult(
            candidate_key=cand.key,
            ok_apply=False,
            ok_compile=False,
            ok_unit_smoke=False,
            targeted_passed=0,
            targeted_failed=0,
            score=-999.0,
            diff_stats={},
            notes=["apply_failed"],
        )

    diff = compute_diff_stats(cand.patch)
    diff_stats = {"files_changed": diff.files_changed, "lines_changed": diff.lines_changed, "paths": diff.paths}

    forbid = cfg.get("forbid_paths_prefix", [])
    risk = static_risk_score(cand.patch, forbid)

    ok_compile, _ = _run_shell(cfg["tests"]["smoke_cmd"], repo_root, timeout=600)
    ok_unit_smoke, _ = _run_shell(cfg["tests"]["unit_smoke_cmd"], repo_root, timeout=900)

    tgt_cmds = targeted_tests(diff, repo_root, cfg["tests"]["targeted_max"])
    tp = 0
    tf = 0
    for c in tgt_cmds:
        ok, _ = _run_shell(c, repo_root, timeout=900)
        if ok:
            tp += 1
        else:
            tf += 1

    # Weighted score
    w = cfg["ranker"]
    score = 0.0
    score += (w["w_compile"] if ok_compile else -w["w_compile"])
    score += (w["w_unit_smoke"] if ok_unit_smoke else -w["w_unit_smoke"])
    score += w["w_targeted_tests"] * (tp - tf) / max(1, (tp + tf))
    score += w["w_static_risk"] * (-risk)  # risk is negative for worse
    score += w["w_diff_size"] * (diff.lines_changed / 100.0)

    if not ok_unit_smoke and tp == 0:
        notes.append("weak_candidate")

    return EvalResult(
        candidate_key=cand.key,
        ok_apply=True,
        ok_compile=ok_compile,
        ok_unit_smoke=ok_unit_smoke,
        targeted_passed=tp,
        targeted_failed=tf,
        score=score,
        diff_stats=diff_stats,
        notes=notes,
    )
