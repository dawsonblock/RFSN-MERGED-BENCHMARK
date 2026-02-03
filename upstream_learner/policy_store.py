"""Policy store for upstream learner.

Atomic JSON persistence to .rfsn_state/policy/upstream_policy.json.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


def _atomic_write(path: str, obj: dict[str, Any]) -> None:
    """Atomically write JSON to path using tmp + rename."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def load_json(path: str, default: dict[str, Any]) -> dict[str, Any]:
    """Load JSON from path, returning default on error."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


@dataclass
class PolicyStore:
    """Persistent storage for upstream policy."""

    path: str = ".rfsn_state/policy/upstream_policy.json"

    def load(self) -> dict[str, Any]:
        """Load policy from disk."""
        return load_json(self.path, default={"version": 1, "bandits": {}})

    def save(self, policy: dict[str, Any]) -> None:
        """Save policy to disk atomically."""
        _atomic_write(self.path, policy)
