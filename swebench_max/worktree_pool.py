import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Worktree:
    path: str
    branch: str

def _run(cmd: List[str], cwd: Optional[str] = None) -> str:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed: {cmd}\n{p.stdout}\n{p.stderr}")
    return p.stdout

class WorktreePool:
    """
    Upstream only. Never commits to main.
    Uses git worktrees to evaluate candidates in parallel safely.
    """
    def __init__(self, repo_root: str, max_parallel: int):
        self.repo_root = os.path.abspath(repo_root)
        self.max_parallel = max_parallel
        self.base_dir = os.path.join(self.repo_root, ".rfsn_worktrees")
        os.makedirs(self.base_dir, exist_ok=True)
        self._worktrees: List[Worktree] = []

    def create(self, idx: int) -> Worktree:
        branch = f"rfsn_wt_{idx}"
        path = os.path.join(self.base_dir, branch)
        if os.path.exists(path):
            self.remove(branch, path)

        _run(["git", "worktree", "add", "-b", branch, path], cwd=self.repo_root)
        wt = Worktree(path=path, branch=branch)
        self._worktrees.append(wt)
        return wt

    def remove(self, branch: str, path: str):
        try:
            _run(["git", "worktree", "remove", "--force", path], cwd=self.repo_root)
        except Exception:
            pass
        try:
            _run(["git", "branch", "-D", branch], cwd=self.repo_root)
        except Exception:
            pass

    def cleanup(self):
        for wt in list(self._worktrees):
            self.remove(wt.branch, wt.path)
        self._worktrees.clear()
