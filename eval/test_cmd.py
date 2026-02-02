"""Test command derivation for SWE-bench tasks."""
from __future__ import annotations
from typing import List, Dict, Any, Optional


def derive_test_command(hints: Optional[Dict[str, Any]]) -> List[str]:
    """
    Derive the correct test command for a SWE-bench task.
    
    Prefers explicit dataset-provided commands if present.
    Fallback is pytest in quiet mode.
    
    Args:
        hints: Task hints/metadata from the dataset
        
    Returns:
        Command as list of strings (for subprocess)
    """
    if hints:
        # Common keys in swebench variants
        for k in ("test_cmd", "test_command", "pytest_cmd", "command"):
            v = hints.get(k)
            if isinstance(v, str) and v.strip():
                # Conservative split for simple commands
                return v.strip().split()

        # Sometimes nested under env
        env = hints.get("env") if isinstance(hints.get("env"), dict) else None
        if env:
            v = env.get("test_cmd") or env.get("test_command")
            if isinstance(v, str) and v.strip():
                return v.strip().split()

    # Fallback: pytest in quiet mode
    return ["python", "-m", "pytest", "-q"]


def derive_test_command_for_repo(repo: str, hints: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Derive test command with repo-specific defaults.
    
    Uses dataset hints first, then repo-specific heuristics.
    """
    # Try hints first
    cmd = derive_test_command(hints)
    if cmd != ["python", "-m", "pytest", "-q"]:
        return cmd
    
    # Repo-specific defaults
    repo_lower = repo.lower()
    
    if "django" in repo_lower:
        return ["python", "-m", "django", "test", "--settings=tests.settings"]
    if "flask" in repo_lower:
        return ["python", "-m", "pytest", "tests/"]
    if "requests" in repo_lower:
        return ["python", "-m", "pytest", "tests/"]
    if "numpy" in repo_lower:
        return ["python", "-m", "pytest", "numpy/"]
    if "pandas" in repo_lower:
        return ["python", "-m", "pytest", "pandas/tests/"]
    if "scikit-learn" in repo_lower or "sklearn" in repo_lower:
        return ["python", "-m", "pytest", "sklearn/"]
    
    return cmd
