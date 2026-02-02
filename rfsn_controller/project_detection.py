"""Project type detection for automatic setup and testing.

Detects project type from repository structure and provides
appropriate setup and test commands.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from functools import lru_cache

from .buildpacks import BuildpackContext, get_all_buildpacks


@dataclass
class ProjectType:
    """Detected project type with setup and test commands."""

    name: str
    setup_commands: list[str]
    test_commands: list[str]
    language: str


@dataclass
class InstallResult:
    """Result of an installation attempt."""

    success: bool
    command: str
    output: str
    error: str
    failure_reason: str | None = None


def classify_install_failure(stderr: str) -> str:
    """Classify installation failure type.

    Args:
        stderr: Error output from installation.

    Returns:
        Failure reason string.
    """
    if not stderr:
        return "unknown"

    stderr_lower = stderr.lower()

    # Check for system library issues
    if any(
        x in stderr_lower
        for x in ["command not found", "no such file", "cannot find -l", "library not found"]
    ):
        return "missing_system_libs"

    # Check for Python version mismatch
    if any(
        x in stderr_lower
        for x in ["python_requires", "requires python", "not supported", "version"]
    ):
        return "python_version_mismatch"

    # Check for pip resolution issues
    if any(x in stderr_lower for x in ["resolutionerror", "dependency conflict", "could not find"]):
        return "pip_resolution_error"

    # Check for network issues
    if any(x in stderr_lower for x in ["connection refused", "timeout", "network", "ssl"]):
        return "network_error"

    # Check for permission issues
    if any(x in stderr_lower for x in ["permission denied", "access denied", "eacces"]):
        return "permission_error"

    return "unknown"


@lru_cache(maxsize=32)
def detect_project_type(repo_dir: str) -> ProjectType | None:
    """Detect the project type from repository structure.

    Args:
        repo_dir: Path to the repository.

    Returns:
        ProjectType if detected, None otherwise.
    """
    # Create context for buildpacks
    # For performance, we don't fully populate files dict, just keys
    try:
        files_list = []
        for root, _, filenames in os.walk(repo_dir):
            rel_root = os.path.relpath(root, repo_dir)
            if rel_root == ".":
                rel_root = ""
            for f in filenames:
                path = os.path.join(rel_root, f) if rel_root else f
                files_list.append(path)
            # Don't walk too deep for detection
            if rel_root.count(os.sep) > 2:
                continue
    except OSError:
        files_list = []

    # Create dummy map (keys only) for detection
    files_map = {f: "" for f in files_list}
    
    ctx = BuildpackContext(
        repo_dir=repo_dir,
        repo_tree=files_list,
        files=files_map,
    )

    # Iterate through all buildpacks
    best_result = None
    best_pack = None

    for pack in get_all_buildpacks():
        result = pack.detect(ctx)
        if result and (best_result is None or result.confidence > best_result.confidence):
            best_result = result
            best_pack = pack
            if result.confidence >= 1.0:
                break
    
    if best_result and best_pack:
        # Convert Buildpack steps to legacy ProjectType commands
        setup_steps = best_pack.get_safe_install_plan(ctx)
        setup_cmds = [shlex.join(s.argv) for s in setup_steps]
        
        test_plan = best_pack.test_plan(ctx)
        test_cmds = [shlex.join(test_plan.argv)]
        
        return ProjectType(
            name=best_result.buildpack_type.value,
            setup_commands=setup_cmds,
            test_commands=test_cmds,
            language=best_result.buildpack_type.value,  # Approximation
        )

    return None


def get_default_test_command(repo_dir: str) -> str | None:
    """Get a default test command for the repository.

    Args:
        repo_dir: Path to the repository.

    Returns:
        Test command string or None if unable to determine.
    """
    project_type = detect_project_type(repo_dir)
    if project_type and project_type.test_commands:
        return project_type.test_commands[0]
    return None


def get_setup_commands(repo_dir: str) -> list[str]:
    """Get setup commands for the repository.

    Args:
        repo_dir: Path to the repository.

    Returns:
        List of setup command strings.
    """
    commands = []
    
    # Check for generic setup script (Legacy fallback or explicit override)
    # Note: CppBuildpack handles this now, but keeping it here ensures
    # it works even if no specific buildpack matches.
    if os.path.exists(os.path.join(repo_dir, "setup.sh")):
        commands.append("bash setup.sh")

    project_type = detect_project_type(repo_dir)
    if project_type:
        commands.extend(project_type.setup_commands)
    
    # Deduplicate while preserving order
    seen = set()
    unique_commands = []
    for cmd in commands:
        if cmd not in seen:
            unique_commands.append(cmd)
            seen.add(cmd)
            
    return unique_commands

