"""Skill heads - repo-specific repair strategies."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Callable, List


@dataclass
class SkillHead:
    """
    A skill head provides repo-specific guidance for patch generation.
    
    Each skill head defines:
    - Applicability condition
    - Prompt suffix for LLM guidance
    - Patch constraints (max files, max lines)
    """
    name: str
    applies: Callable[[Dict[str, Any]], bool]
    prompt_suffix: str
    patch_style: Dict[str, Any]
    priority: int = 0  # Higher = more specific


def _repo_contains(pkg: str) -> Callable[[Dict[str, Any]], bool]:
    """Create a matcher that checks if repo fingerprint contains a package."""
    def _fn(ctx: Dict[str, Any]) -> bool:
        fingerprint = ctx.get("repo_fingerprint", "") or ""
        return pkg.lower() in fingerprint.lower()
    return _fn


def _has_file_pattern(pattern: str) -> Callable[[Dict[str, Any]], bool]:
    """Create a matcher that checks for file patterns."""
    def _fn(ctx: Dict[str, Any]) -> bool:
        files = ctx.get("files", []) or []
        return any(pattern in f for f in files)
    return _fn


# Registry of skill heads - ordered by priority
SKILL_HEADS: List[SkillHead] = [
    # Default minimal fix skill (always applies)
    SkillHead(
        name="minimal_fix",
        applies=lambda ctx: True,
        prompt_suffix=(
            "Constraints:\n"
            "- Keep changes minimal and focused.\n"
            "- Prefer local fix over refactor.\n"
            "- Do not change public APIs unless tests demand it.\n"
            "- Avoid introducing new dependencies.\n"
        ),
        patch_style={"max_files": 3, "max_lines": 120},
        priority=0,
    ),
    
    # pytest-specific skill
    SkillHead(
        name="pytest_fix",
        applies=_has_file_pattern("test_"),
        prompt_suffix=(
            "pytest constraints:\n"
            "- Use pytest idioms (fixtures, parametrize).\n"
            "- Keep test isolation.\n"
            "- Check fixture scope carefully.\n"
        ),
        patch_style={"max_files": 4, "max_lines": 150},
        priority=10,
    ),
    
    # typing/mypy-aware skill
    SkillHead(
        name="typing_safe",
        applies=_repo_contains("mypy"),
        prompt_suffix=(
            "Type constraints:\n"
            "- Keep typing consistent with existing annotations.\n"
            "- Avoid Any unless unavoidable.\n"
            "- Use Optional for nullable types.\n"
        ),
        patch_style={"max_files": 4, "max_lines": 160},
        priority=20,
    ),
    
    # pandas-aware skill
    SkillHead(
        name="pandas_vectorized",
        applies=_repo_contains("pandas"),
        prompt_suffix=(
            "pandas constraints:\n"
            "- Prefer vectorized operations.\n"
            "- Avoid per-row Python loops.\n"
            "- Be careful with dtype conversions.\n"
        ),
        patch_style={"max_files": 5, "max_lines": 200},
        priority=30,
    ),
    
    # numpy-aware skill
    SkillHead(
        name="numpy_safe",
        applies=_repo_contains("numpy"),
        prompt_suffix=(
            "numpy constraints:\n"
            "- Maintain dtype consistency.\n"
            "- Use broadcasting where appropriate.\n"
            "- Avoid unnecessary copies.\n"
        ),
        patch_style={"max_files": 4, "max_lines": 180},
        priority=25,
    ),
    
    # Django-aware skill
    SkillHead(
        name="django_orm",
        applies=_repo_contains("django"),
        prompt_suffix=(
            "Django constraints:\n"
            "- Use ORM methods properly.\n"
            "- Be aware of lazy evaluation.\n"
            "- Check migration state.\n"
        ),
        patch_style={"max_files": 6, "max_lines": 250},
        priority=30,
    ),
    
    # FastAPI-aware skill
    SkillHead(
        name="fastapi_async",
        applies=_repo_contains("fastapi"),
        prompt_suffix=(
            "FastAPI constraints:\n"
            "- Use async/await consistently.\n"
            "- Validate Pydantic models.\n"
            "- Check dependency injection.\n"
        ),
        patch_style={"max_files": 5, "max_lines": 200},
        priority=30,
    ),
]
