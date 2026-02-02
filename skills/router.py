"""Skill router - selects appropriate skill heads for a context."""
from __future__ import annotations
from typing import Dict, Any, List
from .heads import SKILL_HEADS, SkillHead


def select_skill_heads(ctx: Dict[str, Any], k: int = 2) -> List[SkillHead]:
    """
    Select the most appropriate skill heads for a given context.
    
    Args:
        ctx: Context dict with repo_fingerprint, files, etc.
        k: Maximum number of skill heads to return
        
    Returns:
        List of applicable SkillHead objects, sorted by priority
    """
    hits = [h for h in SKILL_HEADS if h.applies(ctx)]
    
    # Sort by priority (higher = more specific)
    hits.sort(key=lambda h: h.priority, reverse=True)
    
    return hits[:k]


def merge_skill_constraints(heads: List[SkillHead]) -> Dict[str, Any]:
    """
    Merge constraints from multiple skill heads.
    
    For numeric constraints, uses the most restrictive value.
    For prompts, concatenates all.
    """
    if not heads:
        return {"max_files": 5, "max_lines": 200, "prompt_suffix": ""}
    
    max_files = min(h.patch_style.get("max_files", 10) for h in heads)
    max_lines = min(h.patch_style.get("max_lines", 500) for h in heads)
    
    prompts = [h.prompt_suffix for h in heads if h.prompt_suffix]
    combined_prompt = "\n\n".join(prompts)
    
    return {
        "max_files": max_files,
        "max_lines": max_lines,
        "prompt_suffix": combined_prompt,
        "skill_names": [h.name for h in heads],
    }


def get_repo_fingerprint(repo: str, files: List[str] | None = None) -> str:
    """
    Generate a fingerprint for skill matching.
    
    Combines repo name with file patterns to help skill selection.
    """
    parts = [repo.lower()]
    
    if files:
        # Add common package indicators
        for f in files[:50]:  # Limit to avoid huge fingerprints
            f_lower = f.lower()
            if "requirements" in f_lower or "setup.py" in f_lower:
                # Could parse these for dependencies
                pass
            if "django" in f_lower:
                parts.append("django")
            if "fastapi" in f_lower:
                parts.append("fastapi")
            if "numpy" in f_lower or "np." in f_lower:
                parts.append("numpy")
            if "pandas" in f_lower or "pd." in f_lower:
                parts.append("pandas")
    
    return " ".join(set(parts))
