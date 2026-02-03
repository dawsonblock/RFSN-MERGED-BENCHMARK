import concurrent.futures as cf
import json
import os
import time
from typing import Dict, Any, List, Tuple

from swebench_max.worktree_pool import WorktreePool
from swebench_max.candidate import Candidate
from swebench_max.evaluator import evaluate_candidate, EvalResult
from swebench_max.diff_stats import compute_diff_stats

def _load_yaml(path: str) -> Dict[str, Any]:
    # minimal yaml: require PyYAML installed; if not, fail loudly.
    import yaml  # type: ignore
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def generate_candidates_stub(issue: Dict[str, Any], cfg: Dict[str, Any]) -> List[Candidate]:
    """
    Replace this with your real planner(s).
    Must return unified diffs.
    """
    return []

def select_best(results: List[EvalResult]) -> EvalResult:
    return sorted(results, key=lambda r: r.score, reverse=True)[0]

def swebench_max_run(repo_root: str, issue_json: str, cfg_path: str) -> Dict[str, Any]:
    cfg = _load_yaml(cfg_path)
    issue = json.loads(issue_json)

    pool = WorktreePool(repo_root, max_parallel=cfg["worktrees"]["max_parallel"])
    worktrees = [pool.create(i) for i in range(cfg["worktrees"]["max_parallel"])]

    try:
        all_results: List[EvalResult] = []
        candidates_seen = 0

        for rnd in range(cfg["rounds"]):
            cands = generate_candidates_stub(issue, cfg)
            cands = cands[: cfg["candidates_per_round"]]
            if not cands:
                break

            batch: List[Tuple[Candidate, str]] = []
            for i, cand in enumerate(cands):
                wt = worktrees[i % len(worktrees)]
                batch.append((cand, wt.path))

            with cf.ThreadPoolExecutor(max_workers=len(worktrees)) as ex:
                futs = []
                for cand, path in batch:
                    futs.append(ex.submit(evaluate_candidate, path, cand, cfg))

                for f in cf.as_completed(futs):
                    all_results.append(f.result())
                    candidates_seen += 1
                    if candidates_seen >= cfg["max_total_candidates"]:
                        break

            if candidates_seen >= cfg["max_total_candidates"]:
                break

        if not all_results:
            return {"status": "no_candidates"}

        best = select_best(all_results)

        return {
            "status": "ok",
            "best": {
                "candidate_key": best.candidate_key,
                "score": best.score,
                "ok_compile": best.ok_compile,
                "ok_unit_smoke": best.ok_unit_smoke,
                "targeted_passed": best.targeted_passed,
                "targeted_failed": best.targeted_failed,
                "diff_stats": best.diff_stats,
                "notes": best.notes,
            },
            "all": [
                {
                    "candidate_key": r.candidate_key,
                    "score": r.score,
                    "ok_compile": r.ok_compile,
                    "ok_unit_smoke": r.ok_unit_smoke,
                    "tp": r.targeted_passed,
                    "tf": r.targeted_failed,
                }
                for r in sorted(all_results, key=lambda x: x.score, reverse=True)[:20]
            ],
        }

    finally:
        pool.cleanup()
