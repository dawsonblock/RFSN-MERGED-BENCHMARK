"""Test command derivation for SWE-bench tasks."""
from __future__ import annotations
from typing import List, Dict, Any, Optional


def derive_test_command(hints: Optional[Dict[str, Any]], fail_to_pass: Optional[List[str]] = None, pass_to_pass: Optional[List[str]] = None) -> List[str]:
    """
    Derive the correct test command for a SWE-bench task.
    
    Prefers explicit dataset-provided commands if present.
    Falls back to running specific FAIL_TO_PASS tests if provided.
    Final fallback is pytest in quiet mode.
    
    Args:
        hints: Task hints/metadata from the dataset
        fail_to_pass: List of test paths that should pass after the fix (from SWE-bench)
        pass_to_pass: List of test paths that should continue to pass
        
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

    # Use FAIL_TO_PASS tests from SWE-bench (run ALL specified tests)
    all_tests = []
    if fail_to_pass:
        all_tests.extend(fail_to_pass)
    if pass_to_pass:
        all_tests.extend(pass_to_pass)
    
    if all_tests:
        # Run pytest with the specific test paths, suppress deprecation warnings
        return ["python", "-m", "pytest", "-xvs", "-W", "ignore::DeprecationWarning"] + all_tests

    # Fallback: pytest in quiet mode
    return ["python", "-m", "pytest", "-q"]


def derive_test_command_for_repo(
    repo: str, 
    hints: Optional[Dict[str, Any]] = None,
    fail_to_pass: Optional[List[str]] = None,
    pass_to_pass: Optional[List[str]] = None,
) -> List[str]:
    """
    Derive test command with repo-specific defaults.
    
    Uses dataset hints first, then FAIL_TO_PASS tests, then repo-specific heuristics.
    
    Args:
        repo: Repository name (e.g., "astropy/astropy")
        hints: Task hints/metadata from the dataset
        fail_to_pass: List of test paths that should pass after the fix
        pass_to_pass: List of test paths that should continue to pass
    """
    repo_lower = repo.lower()
    
    # Astropy needs special handling FIRST - before generic logic
    # Disable doctest to avoid importing extension-dependent modules
    if "astropy" in repo_lower:
        # Build command with the specific test paths if available
        if fail_to_pass or pass_to_pass:
            test_paths = []
            if fail_to_pass:
                test_paths.extend(fail_to_pass)
            if pass_to_pass:
                test_paths.extend(pass_to_pass)
            # Disable doctest collection to avoid importing extension-dependent modules
            return [
                "python", "-m", "pytest", "-xvs",
                "-W", "ignore::DeprecationWarning",
                "-p", "no:doctest",  # Disable doctest plugin
                "--ignore=astropy/io",  # Skip modules with extension dependencies
                "--ignore=astropy/convolution",
                "--ignore=astropy/cosmology",
                "--ignore=astropy/table",
                "--ignore=astropy/wcs",
                "--ignore=astropy/coordinates",
                "--ignore=astropy/time",
                "--ignore=astropy/timeseries",
            ] + test_paths
    
    # Try hints first, then SWE-bench test specs
    cmd = derive_test_command(hints, fail_to_pass, pass_to_pass)
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
