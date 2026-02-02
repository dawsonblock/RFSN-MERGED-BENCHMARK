"""Strategy bandit for upstream learning."""
from __future__ import annotations
import random
from typing import Dict


class StrategyBandit:
    """
    Simple epsilon-greedy bandit for strategy selection.
    
    Tracks success/failure statistics for different strategies
    and selects the best-performing one (with exploration).
    """
    
    def __init__(self, epsilon: float = 0.1):
        self.stats: Dict[str, Dict[str, float]] = {}
        self.epsilon = epsilon

    def select(self) -> str:
        """Select a strategy using epsilon-greedy."""
        if not self.stats:
            return "default"
        
        # Epsilon-greedy: explore with probability epsilon
        if random.random() < self.epsilon:
            return random.choice(list(self.stats.keys()))
        
        # Exploit: choose best strategy
        return max(self.stats, key=lambda k: self.stats[k]["score"])

    def update(self, strategy: str, reward: float) -> None:
        """Update statistics for a strategy."""
        if strategy not in self.stats:
            self.stats[strategy] = {"score": 0.0, "count": 0}
        self.stats[strategy]["count"] += 1
        # Running average
        n = self.stats[strategy]["count"]
        old_score = self.stats[strategy]["score"]
        self.stats[strategy]["score"] = old_score + (reward - old_score) / n
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Return current statistics."""
        return dict(self.stats)
