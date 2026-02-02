from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Any

from .thompson import ThompsonBandit, BetaArm
from .state_store import atomic_write_json, read_json


class PersistentThompsonBandit(ThompsonBandit):
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.load()

    def save(self) -> None:
        payload: Dict[str, Any] = {
            "arms": {name: {"a": arm.a, "b": arm.b} for name, arm in self.arms.items()}
        }
        atomic_write_json(self.path, payload)

    def load(self) -> None:
        payload = read_json(self.path, default={"arms": {}})
        arms = payload.get("arms", {}) if isinstance(payload, dict) else {}
        if not isinstance(arms, dict):
            return
        for name, ab in arms.items():
            try:
                a = float(ab.get("a", 1.0))
                b = float(ab.get("b", 1.0))
                self.arms[name] = BetaArm(a=a, b=b)
            except Exception:
                continue
