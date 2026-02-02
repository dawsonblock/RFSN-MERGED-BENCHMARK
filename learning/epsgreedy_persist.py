from __future__ import annotations

from typing import Dict, Any

from .bandit import StrategyBandit
from .state_store import atomic_write_json, read_json


class PersistentStrategyBandit(StrategyBandit):
    def __init__(self, path: str, epsilon: float = 0.1):
        super().__init__(epsilon=epsilon)
        self.path = path
        self.load()

    def save(self) -> None:
        atomic_write_json(self.path, {"epsilon": self.epsilon, "stats": self.stats})

    def load(self) -> None:
        payload: Dict[str, Any] = read_json(self.path, default={"epsilon": self.epsilon, "stats": {}})
        eps = payload.get("epsilon", self.epsilon)
        try:
            self.epsilon = float(eps)
        except Exception:
            pass
        stats = payload.get("stats", {})
        if isinstance(stats, dict):
            self.stats = stats
