"""Planner bandit for selecting among multiple planners."""
from __future__ import annotations
from typing import Dict, Any, Callable
from .thompson import ThompsonBandit

# Type alias for planner functions
PlannerFn = Callable[[Dict[str, Any], Dict[str, Any]], Any]

# Registry of available planners
PLANNERS: Dict[str, PlannerFn] = {}


def register_planner(name: str, fn: PlannerFn) -> None:
    """Register a planner function."""
    PLANNERS[name] = fn


def get_planner(name: str) -> PlannerFn | None:
    """Get a registered planner by name."""
    return PLANNERS.get(name)


class PlannerSelector:
    """
    Selects among registered planners using Thompson sampling.
    
    Learns which planner works best over time based on
    observed success/failure outcomes.
    """
    
    def __init__(self):
        self.bandit = ThompsonBandit()

    def pick(self) -> str:
        """Pick a planner using Thompson sampling."""
        options = list(PLANNERS.keys()) or ["planner_v1"]
        return self.bandit.choose(options)

    def update(self, planner_name: str, success: bool, weight: float = 1.0) -> None:
        """Update the bandit based on planner outcome."""
        self.bandit.update(planner_name, success, weight)
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all planners."""
        return self.bandit.get_statistics()
