import glob
import json
import os
import re
from typing import Dict, Any, List, Tuple

def _safe_read(path: str, max_bytes: int = 200_000) -> str:
    try:
        with open(path, "rb") as f:
            b = f.read(max_bytes)
        return b.decode("utf-8", errors="replace")
    except Exception:
        return ""

def extract_fail_signals(log_dir: str, task_id: str, limit: int = 6) -> List[str]:
    """
    Looks for recent episodes mentioning task_id and extracts short failure snippets.
    Expected logs layout: logs/episodes/*.json (your earlier script).
    """
    out: List[Tuple[float, str]] = []
    for p in glob.glob(os.path.join(log_dir, "episodes", "*.json")):
        try:
            st = os.stat(p).st_mtime
            j = json.loads(_safe_read(p))
            if str(j.get("task_id", "")) != str(task_id):
                continue
            # capture short fields if present
            parts = []
            for k in ["error", "failure", "traceback", "pytest_output", "notes"]:
                v = j.get(k)
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip()[:800])
            if not parts:
                continue
            snippet = "\n---\n".join(parts)
            out.append((st, snippet))
        except Exception:
            continue
    out.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in out[:limit]]

def quick_repo_context(repo_root: str, max_files: int = 40) -> str:
    """
    Low-cost context: list key files + small excerpts from likely areas.
    """
    # Prefer README + top-level package modules
    candidates = []
    for name in ["README.md", "pyproject.toml", "setup.cfg", "setup.py"]:
        p = os.path.join(repo_root, name)
        if os.path.exists(p):
            candidates.append(p)

    # include small sample of recent python files
    py_files = []
    for root, _, files in os.walk(repo_root):
        if "/.git/" in root.replace("\\", "/"):
            continue
        for f in files:
            if f.endswith(".py") and "tests" not in root:
                py_files.append(os.path.join(root, f))
    py_files.sort(key=lambda p: os.stat(p).st_mtime if os.path.exists(p) else 0, reverse=True)
    candidates.extend(py_files[: max(0, max_files - len(candidates))])

    chunks = []
    for p in candidates[:max_files]:
        rel = os.path.relpath(p, repo_root)
        txt = _safe_read(p, max_bytes=6000)
        if txt.strip():
            chunks.append(f"FILE: {rel}\n{txt[:2000]}")
    return "\n\n".join(chunks)
