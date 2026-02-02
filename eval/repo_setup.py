"""Repository setup for SWE-bench evaluation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import os
import subprocess
import tempfile
import shutil


@dataclass
class RepoWorkspace:
    """A cloned repository workspace."""
    path: str
    repo: str
    base_commit: str


def _run(cmd: List[str], cwd: str, timeout_s: int = 600) -> subprocess.CompletedProcess:
    """Run a command with timeout."""
    return subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def clone_repo(repo_url: str, base_commit: str, work_root: str = ".work") -> RepoWorkspace:
    """
    Clone a repository and checkout a specific commit.
    
    Args:
        repo_url: Git URL to clone
        base_commit: Commit SHA to checkout
        work_root: Directory to create workspaces in
        
    Returns:
        RepoWorkspace with path to cloned repo
        
    Raises:
        RuntimeError: If clone or checkout fails
    """
    os.makedirs(work_root, exist_ok=True)
    d = tempfile.mkdtemp(prefix="rfsn_repo_", dir=work_root)
    
    # Try shallow clone first (faster)
    p = _run(["git", "clone", "--depth", "1", repo_url, d], cwd=".")
    if p.returncode != 0:
        # Fallback: full clone (some commits not in shallow)
        shutil.rmtree(d, ignore_errors=True)
        d = tempfile.mkdtemp(prefix="rfsn_repo_", dir=work_root)
        p = _run(["git", "clone", repo_url, d], cwd=".")
        if p.returncode != 0:
            raise RuntimeError(f"git clone failed:\n{p.stdout}")

    # Fetch all refs to ensure we can checkout the base commit
    p = _run(["git", "fetch", "--all", "--tags"], cwd=d)
    if p.returncode != 0:
        raise RuntimeError(f"git fetch failed:\n{p.stdout}")

    p = _run(["git", "checkout", base_commit], cwd=d)
    if p.returncode != 0:
        raise RuntimeError(f"git checkout {base_commit} failed:\n{p.stdout}")

    return RepoWorkspace(path=d, repo=repo_url, base_commit=base_commit)


def hard_reset_clean(ws: RepoWorkspace) -> None:
    """
    Reset workspace to clean state.
    
    Performs git reset --hard and git clean -fdx.
    """
    _run(["git", "reset", "--hard"], cwd=ws.path)
    _run(["git", "clean", "-fdx"], cwd=ws.path)


def apply_patch_text(ws: RepoWorkspace, patch_text: str) -> str:
    """
    Apply a patch blob safely.
    
    Uses git apply --check first, then git apply --3way.
    
    Args:
        ws: Workspace to apply patch in
        patch_text: The patch content
        
    Returns:
        Status string: "APPLIED_OK", "EMPTY_PATCH", or error message
    """
    if not patch_text.strip():
        return "EMPTY_PATCH"

    patch_path = os.path.join(ws.path, ".rfsn_tmp.patch")
    with open(patch_path, "w", encoding="utf-8") as f:
        f.write(patch_text)

    try:
        # Validate patch first
        p1 = _run(["git", "apply", "--check", patch_path], cwd=ws.path)
        if p1.returncode != 0:
            return "APPLY_CHECK_FAILED\n" + p1.stdout

        # Apply with 3-way merge for better handling
        p2 = _run(["git", "apply", "--3way", patch_path], cwd=ws.path)
        if p2.returncode != 0:
            return "APPLY_3WAY_FAILED\n" + p2.stdout

        return "APPLIED_OK"
    finally:
        # Clean up temp file
        if os.path.exists(patch_path):
            os.remove(patch_path)


def cleanup_workspace(ws: RepoWorkspace) -> None:
    """Remove workspace directory."""
    if ws.path and os.path.exists(ws.path):
        shutil.rmtree(ws.path, ignore_errors=True)
