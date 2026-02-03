"""Stable Repo Fingerprint.

Computes a deterministic fingerprint based on dependency/config files,
not commit prefixes. This ensures repair cards and upstream learner
can generalize across commits in the same repo family.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def _read(path: Path, max_bytes: int = 250_000) -> bytes:
    """Read file contents up to max_bytes, return empty on failure."""
    try:
        return path.read_bytes()[:max_bytes]
    except Exception:
        return b""


def compute_repo_fingerprint(repo_dir: str) -> str:
    """
    Deterministic fingerprint based on common dependency/config files.
    Stable across commits unless dependencies/config actually change.
    
    Args:
        repo_dir: Path to repository root
        
    Returns:
        16-char hex string fingerprint
    """
    root = Path(repo_dir)
    candidates = [
        "pyproject.toml",
        "poetry.lock",
        "requirements.txt",
        "requirements-dev.txt",
        "setup.cfg",
        "setup.py",
        "Pipfile",
        "Pipfile.lock",
        "tox.ini",
        "pytest.ini",
        "conftest.py",
    ]

    h = hashlib.sha256()
    for name in candidates:
        h.update(name.encode("utf-8"))
        h.update(_read(root / name))
    return h.hexdigest()[:16]
