from __future__ import annotations

import hashlib
import re
import subprocess
import time


def _safe(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:80] if len(s) > 80 else s


def _git_rev(repo_path: str) -> str | None:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5, check=False,
        )
        if p.returncode == 0:
            return p.stdout.strip()
    except Exception:
        return None
    return None


def make_run_id(
    *,
    repo_path: str,
    goal: str,
    profile: str | None = None,
    ts_unix: int | None = None,
) -> str:
    ts = int(ts_unix if ts_unix is not None else time.time())
    commit = _git_rev(repo_path) or "n/a"
    base = f"{ts}:{commit}:{goal}:{profile or ''}"
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:12]
    return f"{ts}-{_safe(profile or 'run')}-{digest}"
