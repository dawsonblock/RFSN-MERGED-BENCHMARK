from dataclasses import dataclass
import math
import random
from typing import Dict, Any, List

@dataclass(frozen=True)
class Arm:
    key: str
    features: Dict[str, float]

class ThompsonBandit:
    """
    Contextual bandit.
    NO authority.
    NO side effects.
    Consumes episode logs only.
    """

    def __init__(self):
        self.counts: Dict[str, int] = {}
        self.values: Dict[str, float] = {}

    def select(self, arms: List[Arm]) -> Arm:
        best_arm = None
        best_score = -float("inf")

        for arm in arms:
            n = self.counts.get(arm.key, 0)
            v = self.values.get(arm.key, 0.0)
            bonus = math.sqrt(math.log(1 + sum(self.counts.values())) / (1 + n))
            score = v + random.gauss(0, bonus)

            if score > best_score:
                best_score = score
                best_arm = arm

        return best_arm

    def update(self, arm_key: str, reward: float):
        n = self.counts.get(arm_key, 0)
        v = self.values.get(arm_key, 0.0)
        self.counts[arm_key] = n + 1
        self.values[arm_key] = v + (reward - v) / (n + 1)
