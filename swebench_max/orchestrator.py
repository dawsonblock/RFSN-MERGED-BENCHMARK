"""
SWE-bench MAX Mode Orchestrator

Multi-worktree parallel candidate evaluation with:
- 3-planner ensemble patch proposals
- Log-based failure retrieval into prompts
- Skeptic rewrite on best subset
- Dedup so budget isn't wasted
- Only emits candidates; kernel/gate decides execution
"""
import concurrent.futures as cf
import json
import os
from typing import Dict, Any, List, Tuple, Optional

from swebench_max.worktree_pool import WorktreePool
from swebench_max.candidate import Candidate
from swebench_max.evaluator import evaluate_candidate, EvalResult
from swebench_max.diff_stats import compute_diff_stats
from swebench_max.dedup import PatchDeduper
from swebench_max.retrieval_memory import extract_fail_signals, quick_repo_context
from swebench_max.planners.ensemble import PlannerSpec, propose_patches, skeptic_rewrite
from swebench_max.llm.client import LLMConfig
from swebench_max.filters import forbid_paths_filter


def _load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    import yaml  # type: ignore
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _planner_specs_from_cfg(cfg: Dict[str, Any]) -> List[PlannerSpec]:
    """
    cfg example:
    planners:
      - name: primary
        weight: 1.0
    and llms:
      primary: { provider, model, api_key_env, base_url, ... }
    """
    out: List[PlannerSpec] = []
    llms = cfg.get("llms", {})
    for p in cfg.get("planners", []):
        name = p["name"]
        w = float(p.get("weight", 1.0))
        llm_cfg = llms.get(name) or llms.get("default")
        if not llm_cfg:
            raise ValueError(f"Missing llm config for planner: {name} (cfg.llms.{name} or cfg.llms.default)")
        out.append(
            PlannerSpec(
                name=name,
                weight=w,
                llm=LLMConfig(
                    provider=llm_cfg["provider"],
                    model=llm_cfg["model"],
                    base_url=llm_cfg.get("base_url"),
                    api_key_env=llm_cfg.get("api_key_env", "OPENAI_API_KEY"),
                    timeout_s=int(llm_cfg.get("timeout_s", 120)),
                    max_tokens=int(llm_cfg.get("max_tokens", 2000)),
                    temperature=float(llm_cfg.get("temperature", 0.2)),
                ),
            )
        )
    return out


def _skeptic_llm_from_cfg(cfg: Dict[str, Any]) -> LLMConfig:
    llm_cfg = (cfg.get("llms", {}) or {}).get("skeptic") or (cfg.get("llms", {}) or {}).get("default")
    if not llm_cfg:
        raise ValueError("Missing llm config for skeptic (cfg.llms.skeptic or cfg.llms.default)")
    return LLMConfig(
        provider=llm_cfg["provider"],
        model=llm_cfg["model"],
        base_url=llm_cfg.get("base_url"),
        api_key_env=llm_cfg.get("api_key_env", "OPENAI_API_KEY"),
        timeout_s=int(llm_cfg.get("timeout_s", 120)),
        max_tokens=int(llm_cfg.get("max_tokens", 2000)),
        temperature=float(llm_cfg.get("temperature", 0.2)),
    )


def generate_candidates(issue: Dict[str, Any], cfg: Dict[str, Any]) -> List[Candidate]:
    """
    Real generator:
    - retrieval from logs
    - repo context snapshot
    - 3-planner ensemble proposals
    - skeptic rewrite on top N
    - dedup by normalized hash
    """
    repo_root = cfg.get("_repo_root", ".")
    task_id = str(issue.get("task_id", "unknown"))
    forbid = cfg.get("forbid_paths_prefix", [])

    # Retrieval (cheap, high-signal)
    fail_snips = extract_fail_signals(log_dir=os.path.join(repo_root, "logs"), task_id=task_id, limit=6)
    failures = "\n\n".join(fail_snips) if fail_snips else "(none)"

    context = quick_repo_context(repo_root, max_files=30)

    planners = _planner_specs_from_cfg(cfg)
    skeptic_llm = _skeptic_llm_from_cfg(cfg)

    per_planner = max(1, int(cfg.get("candidates_per_round", 12) // max(1, len(planners))))
    raw_patches = propose_patches(
        issue=issue,
        planners=planners,
        context=context,
        failures=failures,
        forbid_prefixes=forbid,
        per_planner=per_planner,
    )

    # Safety filter on forbidden paths
    raw_patches = [p for p in raw_patches if forbid_paths_filter(p, forbid)]

    dedup = PatchDeduper()
    kept: List[str] = []
    for p in raw_patches:
        if dedup.add(p):
            kept.append(p)
        if len(kept) >= int(cfg.get("candidates_per_round", 12)):
            break

    # Skeptic rewrite on top subset to reduce risk / increase pass-rate
    rewritten: List[str] = []
    for p in kept[: min(len(kept), 8)]:
        rp = skeptic_rewrite(issue, skeptic_llm, p, failures, forbid)
        if dedup.add(rp):
            rewritten.append(rp)

    final = kept + rewritten
    final = final[: int(cfg.get("candidates_per_round", 12))]

    cands: List[Candidate] = []
    for i, patch in enumerate(final):
        cands.append(
            Candidate(
                key=f"{task_id}:cand{i}",
                patch=patch,
                meta={
                    "task_id": task_id,
                    "failures": failures,
                },
            )
        )
    return cands


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
    4. Return best candidate
    """
    cfg = _load_yaml(cfg_path)
    cfg["_repo_root"] = repo_root  # Inject for generator
    issue = json.loads(issue_json)

    # Initialize worktree pool
    pool = WorktreePool(repo_root, max_parallel=cfg["worktrees"]["max_parallel"])
    worktrees = [pool.create(i) for i in range(cfg["worktrees"]["max_parallel"])]

    try:
        all_results: List[EvalResult] = []
        candidates_seen = 0

        for rnd in range(cfg["rounds"]):
            # Generate candidates using ensemble planners
            cands = generate_candidates(issue=issue, cfg=cfg)
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
            with cf.ThreadPoolExecutor(max_workers=len(worktrees)) as ex:
                futs = {
                    ex.submit(evaluate_candidate, path, cand, cfg): cand
                    for cand, path in batch
                }

                for f in cf.as_completed(futs):
                    result = f.result()
                    all_results.append(result)
                    candidates_seen += 1

                    if candidates_seen >= cfg["max_total_candidates"]:
                        break

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
            "rounds_completed": rnd + 1,
            "candidates_evaluated": candidates_seen,
        }

    finally:
        pool.cleanup()
