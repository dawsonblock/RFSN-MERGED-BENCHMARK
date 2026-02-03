"""Feature extraction for contextual bandit.

Extracts context from task metadata, tracebacks, and repo fingerprints.
Produces fixed-length numeric feature vectors for LinUCB.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Context:
    """Contextual features for upstream learner decisions."""

    repo: str
    task_id: str
    bucket: str
    error_type: str
    top_module: str
    top_symbol: str
    test_hint: str
    repo_fingerprint: str
    # Repair card context (optional, defaults for backwards compat)
    rc_k: int = 0
    rc_top_score: float = 0.0
    rc_top_wr: float = 0.5


_ERR_RE = re.compile(r"\b([A-Za-z_]+Error)\b")
_MOD_RE = re.compile(r'File "([^"]+\.py)"')
_SYM_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")


def _safe_read(path: Path, max_bytes: int = 250_000) -> bytes:
    """Safely read file contents up to max_bytes."""
    try:
        b = path.read_bytes()
        return b[:max_bytes]
    except Exception:
        return b""


def repo_fingerprint(repo_dir: str) -> str:
    """Deterministic repo fingerprint based on common dependency/config files.

    Cheap proxy for 'project type' without expensive analysis.
    """
    root = Path(repo_dir)
    candidates = [
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
        "requirements.txt",
        "requirements-dev.txt",
        "Pipfile",
        "poetry.lock",
        "tox.ini",
        "pytest.ini",
    ]
    h = hashlib.sha256()
    for name in candidates:
        h.update(name.encode("utf-8"))
        h.update(_safe_read(root / name))
    return h.hexdigest()[:16]


def parse_failure_signals(test_output: str) -> dict[str, str]:
    """Extract failure signals from test output.

    Returns dict with error_type, top_module, top_symbol, test_hint.
    """
    t = test_output or ""
    m = _ERR_RE.search(t)
    error_type = m.group(1) if m else "UnknownError"

    mod = _MOD_RE.search(t)
    top_module = mod.group(1).replace("\\", "/") if mod else ""

    # crude: pick a symbol near the error mention
    top_symbol = ""
    if error_type in t:
        idx = t.find(error_type)
        window = t[max(0, idx - 200) : idx + 200]
        syms = [s for s in _SYM_RE.findall(window) if len(s) <= 64]
        top_symbol = syms[0] if syms else ""

    # tests hint: first failing test name line if present
    test_hint = ""
    for line in t.splitlines():
        if "FAILED " in line:
            test_hint = line.strip()[:160]
            break

    return {
        "error_type": error_type,
        "top_module": top_module,
        "top_symbol": top_symbol,
        "test_hint": test_hint,
    }


def featurize(ctx: Context) -> list[float]:
    """Fixed-length numeric feature vector for contextual bandit.

    15 dimensions, all in [0,1].
    """

    def h1(s: str, mod: int) -> float:
        if not s:
            return 0.0
        x = int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)
        return float(x % mod) / float(mod)

    return [
        h1(ctx.repo, 997),
        h1(ctx.bucket, 997),
        h1(ctx.error_type, 997),
        h1(ctx.top_module, 997),
        h1(ctx.top_symbol, 997),
        h1(ctx.test_hint, 997),
        h1(ctx.repo_fingerprint, 997),
        # simple interactions
        h1(ctx.bucket + "|" + ctx.error_type, 997),
        h1(ctx.bucket + "|" + ctx.top_module, 997),
        h1(ctx.error_type + "|" + ctx.top_symbol, 997),
        h1(ctx.repo + "|" + ctx.repo_fingerprint, 997),
        # repair-card features (3 dims)
        min(float(max(ctx.rc_k, 0)), 5.0) / 5.0,
        min(float(max(ctx.rc_top_score, 0.0)), 10.0) / 10.0,
        float(min(max(ctx.rc_top_wr, 0.0), 1.0)),
        1.0,  # bias
    ]
