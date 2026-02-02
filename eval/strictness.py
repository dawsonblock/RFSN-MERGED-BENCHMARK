"""Strict benchmark mode enforcement."""
from __future__ import annotations
import os


def strict_benchmark_mode() -> bool:
    """
    Check if strict benchmark mode is enabled.
    
    Default is STRICT (CI-safe). In strict mode:
    - Missing datasets cause hard failure
    - No sample task fallback
    - All validation errors are fatal
    
    Set RFSN_STRICT_BENCH=0 for local development only.
    """
    v = os.environ.get("RFSN_STRICT_BENCH", "1").strip().lower()
    return v not in ("0", "false", "no")


def require_strict(msg: str) -> None:
    """Raise error if in strict mode."""
    if strict_benchmark_mode():
        raise RuntimeError(f"[STRICT MODE] {msg}")
