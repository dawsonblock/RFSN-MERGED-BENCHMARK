"""
SWE-bench MAX Mode Orchestrator

Multi-worktree parallel candidate evaluation with:
- Rounds of candidate generation
- Parallel patch evaluation
- Best candidate selection
- Result recording for learning
"""
import concurrent.futures as cf
import json
import os
from typing import Dict, Any, List, Tuple, Optional

from swebench_max.worktree_pool import WorktreePool
from swebench_max.candidate import Candidate
from swebench_max.evaluator import evaluate_candidate, EvalResult
from swebench_max.diff_stats import compute_diff_stats
from swebench_max.candidate_generator import (
    generate_candidates,
    record_candidate_result,
    GeneratorState,
)


def _load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    import yaml  # type: ignore
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_best(results: List[EvalResult]) -> Optional[EvalResult]:
    """Select the best candidate by score."""
    if not results:
        return None
    return sorted(results, key=lambda r: r.score, reverse=True)[0]


def swebench_max_run(
    repo_root: str,
    issue_json: str,
    cfg_path: str,
) -> Dict[str, Any]:
    """
    Run SWE-bench MAX mode:
    1. Create worktree pool for parallel evaluation
    2. Generate candidates using ensemble planners
    3. Evaluate candidates in parallel
    4. Record results for learning
    5. Return best candidate
    """
    cfg = _load_yaml(cfg_path)
    issue = json.loads(issue_json)
    
    # Initialize worktree pool
    pool = WorktreePool(repo_root, max_parallel=cfg["worktrees"]["max_parallel"])
    worktrees = [pool.create(i) for i in range(cfg["worktrees"]["max_parallel"])]
    
    # Generator state persists across rounds
    gen_state = GeneratorState()
    
    try:
        all_results: List[EvalResult] = []
        candidates_seen = 0
        
        for rnd in range(cfg["rounds"]):
            # Generate candidates using ensemble planners
            cands = generate_candidates(
                issue=issue,
                cfg=cfg,
                state=gen_state,
                repo_root=repo_root,
            )
            cands = cands[:cfg["candidates_per_round"]]
            
            if not cands:
                print(f"Round {rnd}: No candidates generated, stopping")
                break
            
            print(f"Round {rnd}: Evaluating {len(cands)} candidates")
            
            # Assign candidates to worktrees
            batch: List[Tuple[Candidate, str]] = []
            for i, cand in enumerate(cands):
                wt = worktrees[i % len(worktrees)]
                batch.append((cand, wt.path))
            
            # Evaluate in parallel
            round_results: List[Tuple[Candidate, EvalResult]] = []
            with cf.ThreadPoolExecutor(max_workers=len(worktrees)) as ex:
                futs = {
                    ex.submit(evaluate_candidate, path, cand, cfg): cand
                    for cand, path in batch
                }
                
                for f in cf.as_completed(futs):
                    cand = futs[f]
                    result = f.result()
                    round_results.append((cand, result))
                    all_results.append(result)
                    candidates_seen += 1
                    
                    if candidates_seen >= cfg["max_total_candidates"]:
                        break
            
            # Record results for learning (updates gen_state)
            for cand, result in round_results:
                success = result.ok_unit_smoke and result.targeted_failed == 0
                record_candidate_result(cand, success, gen_state, issue, cfg)
            
            # Check if we found a likely winner
            best_so_far = select_best(all_results)
            if best_so_far and best_so_far.score > 8.0:
                print(f"Round {rnd}: Found high-scoring candidate ({best_so_far.score:.2f}), stopping early")
                break
            
            if candidates_seen >= cfg["max_total_candidates"]:
                print(f"Round {rnd}: Reached max candidates ({candidates_seen}), stopping")
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
            "rounds_completed": gen_state.round_idx,
            "candidates_evaluated": candidates_seen,
        }
    
    finally:
        pool.cleanup()
