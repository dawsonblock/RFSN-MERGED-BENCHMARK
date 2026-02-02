"""Recall utilities - build retrieval context for planning."""
from __future__ import annotations
from typing import Dict, Any, List
from .failure_index import FailureIndex


def build_retrieval_context(
    repo: str, 
    test_output: str, 
    index: FailureIndex,
    k: int = 5,
) -> Dict[str, Any]:
    """
    Build retrieval context for the planner.
    
    Queries the failure index for similar past failures
    and formats results for use in planning.
    
    Args:
        repo: Current repository name
        test_output: The test failure output
        index: FailureIndex to query
        k: Number of results to retrieve
        
    Returns:
        Dict with retrieval_hits list
    """
    hits = index.query(test_output, k=k, repo_bias=repo)
    
    return {
        "retrieval_hits": [
            {
                "repo": h.repo,
                "signature": h.signature[:600],
                "patch_summary": h.patch_summary[:600],
                "metadata": h.metadata,
            }
            for h in hits
        ],
        "has_same_repo_hits": any(h.repo == repo for h in hits),
        "total_hits": len(hits),
    }


def format_retrieval_for_prompt(context: Dict[str, Any]) -> str:
    """
    Format retrieval context as prompt text.
    
    Args:
        context: Dict from build_retrieval_context
        
    Returns:
        Formatted string for inclusion in LLM prompts
    """
    hits = context.get("retrieval_hits", [])
    if not hits:
        return ""
    
    lines = ["## Similar Past Fixes\n"]
    
    for i, hit in enumerate(hits[:3], 1):
        lines.append(f"### Example {i}")
        if hit.get("repo"):
            lines.append(f"Repo: {hit['repo']}")
        lines.append(f"Failure pattern: {hit.get('signature', 'N/A')[:200]}")
        lines.append(f"Fix approach: {hit.get('patch_summary', 'N/A')[:200]}")
        lines.append("")
    
    return "\n".join(lines)


def extract_retrieval_insights(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract actionable insights from retrieval results.
    
    Args:
        context: Dict from build_retrieval_context
        
    Returns:
        Dict with extracted insights
    """
    hits = context.get("retrieval_hits", [])
    
    insights = {
        "common_fix_patterns": [],
        "suggested_files": [],
        "confidence_boost": 0.0,
    }
    
    if not hits:
        return insights
    
    # Extract common patterns
    for hit in hits:
        summary = hit.get("patch_summary", "")
        if "import" in summary.lower():
            insights["common_fix_patterns"].append("import_fix")
        if "assert" in summary.lower():
            insights["common_fix_patterns"].append("assertion_update")
        if "type" in summary.lower():
            insights["common_fix_patterns"].append("type_fix")
    
    # Boost confidence if same-repo matches
    if context.get("has_same_repo_hits"):
        insights["confidence_boost"] = 0.1
    
    return insights
