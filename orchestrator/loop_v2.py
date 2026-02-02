"""
Advanced orchestrator v2 - full agent loop.

This orchestrator integrates all upstream modules:
- Planner selection (Thompson sampling)
- Failure classification (repair taxonomy)
- Skill routing (repo-specific heads)
- Retrieval (failure index)
- Learning (bandit updates)

The gate remains untouched - all intelligence is upstream.
"""
from __future__ import annotations

from typing import Dict, Any, Callable, List

from repair.classifier import classify_failure
from skills.router import select_skill_heads
from retrieval.failure_index import FailureIndex, FailureRecord
from retrieval.recall import build_retrieval_context

from learning.planner_bandit import PlannerSelector, register_planner, PLANNERS
from learning.outcomes import Outcome, score

# Import default planner - register it
from planner.planner import generate_plan as planner_v1_generate_plan

# Register default planner
register_planner("planner_v1", planner_v1_generate_plan)

# Global instances (can be replaced with dependency injection)
selector = PlannerSelector()
failure_index = FailureIndex()


def run_episode_v2(
    task: Dict[str, Any],
    patch_generator: Callable[[Any, Dict[str, Any]], List[Any]],
    executor: Callable[[Any], Outcome],
) -> bool:
    """
    Advanced agent loop with full upstream intelligence.
    
    This orchestrator:
    1. Selects planner using Thompson sampling
    2. Classifies failure type using heuristics
    3. Retrieves similar failures from index
    4. Routes skill heads based on repo fingerprint
    5. Generates N patch candidates (beam search)
    6. Executes sequentially (serial authority preserved)
    7. Updates planner bandit + writes failure index on success
    
    Args:
        task: Task dict with repo, test_output, failing_files, etc.
        patch_generator: Function(plan, context) -> list of patch candidates
        executor: Function(patch) -> Outcome
        
    Returns:
        True if task was solved, False otherwise
    """
    repo = task.get("repo", "unknown")
    repo_fp = task.get("repo_fingerprint", repo)
    test_output = task.get("test_output", "")
    failing_files = task.get("failing_files", []) or []

    # 1. Classify failure type
    hypotheses = classify_failure(test_output, failing_files)
    
    # 2. Build retrieval context from failure index
    retrieval_ctx = build_retrieval_context(repo, test_output, failure_index)
    
    # 3. Select skill heads based on repo
    skill_heads = select_skill_heads({"repo_fingerprint": repo_fp}, k=2)
    
    # 4. Select planner using Thompson sampling
    planner_name = selector.pick()
    planner_fn = PLANNERS.get(planner_name, planner_v1_generate_plan)
    
    # 5. Generate plan
    plan = planner_fn(task, retrieval_ctx)
    
    # Attach upstream context to plan metadata
    plan.metadata["repair_hypotheses"] = [h.kind for h in hypotheses]
    plan.metadata["skill_heads"] = [h.name for h in skill_heads]
    plan.metadata["retrieval"] = retrieval_ctx
    
    # 6. Generate patch candidates
    context = {
        "hypotheses": hypotheses,
        "skill_heads": skill_heads,
        "retrieval": retrieval_ctx,
    }
    candidates = patch_generator(plan, context)
    
    # 7. Execute candidates sequentially
    best_delta = 0
    for cand in candidates:
        result = executor(cand)
        best_delta = min(best_delta, result.test_delta)
        
        # Update planner bandit
        r = score(result)
        selector.update(planner_name, success=result.passed, weight=abs(r) if abs(r) > 0 else 1.0)
        
        if result.passed:
            # Write successful pattern into failure index for later recall
            sig = (test_output or "")[:2000]
            failure_index.add(FailureRecord(
                repo=repo,
                signature=sig,
                patch_summary=getattr(cand, "summary", "patch accepted"),
                metadata={
                    "planner": planner_name,
                    "hypotheses": [h.kind for h in hypotheses],
                    "skills": [h.name for h in skill_heads],
                },
            ))
            return True
    
    return False


def get_orchestrator_stats() -> Dict[str, Any]:
    """Get statistics about the orchestrator state."""
    return {
        "planner_stats": selector.get_statistics(),
        "failure_index_size": failure_index.size(),
        "registered_planners": list(PLANNERS.keys()),
    }
