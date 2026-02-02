"""Thompson sampling for planner selection."""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict


@dataclass
class BetaArm:
    """
    Beta distribution arm for Thompson sampling.
    
    Uses Beta(a, b) where:
    - a = 1 + successes
    - b = 1 + failures
    """
    a: float = 1.0  # success + prior
    b: float = 1.0  # failure + prior

    def sample(self) -> float:
        """Sample from Beta(a, b) distribution."""
        return random.betavariate(self.a, self.b)

    def update(self, success: bool, weight: float = 1.0) -> None:
        """Update the arm based on observed outcome."""
        if success:
            self.a += weight
        else:
            self.b += weight
    
    @property
    def mean(self) -> float:
        """Expected value of the arm."""
        return self.a / (self.a + self.b)
    
    @property
    def count(self) -> int:
        """Number of trials (excluding prior)."""
        return int(self.a + self.b - 2)


class ThompsonBandit:
    """
    Thompson Sampling bandit for strategy selection.
    
    Uses Beta-Bernoulli model for each arm. Maintains
    posterior distributions and samples to select arms.
    """
    
    def __init__(self):
        self.arms: Dict[str, BetaArm] = {}

    def ensure(self, name: str) -> None:
        """Ensure an arm exists."""
        if name not in self.arms:
            self.arms[name] = BetaArm()

    def choose(self, options: list[str]) -> str:
        """
        Choose an arm using Thompson sampling.
        
        Samples from each arm's posterior and picks the highest.
        """
        for o in options:
            self.ensure(o)
        
        scored = [(o, self.arms[o].sample()) for o in options]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[0][0]

    def update(self, name: str, success: bool, weight: float = 1.0) -> None:
        """Update an arm's posterior based on outcome."""
        self.ensure(name)
        self.arms[name].update(success, weight)
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all arms."""
        return {
            name: {
                "mean": arm.mean,
                "count": arm.count,
                "a": arm.a,
                "b": arm.b,
            }
            for name, arm in self.arms.items()
        }
