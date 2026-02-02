"""Outcome types and scoring."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Outcome:
    """Result of executing a patch."""
    passed: bool
    test_delta: int  # negative = fewer failures
    runtime: float
    error_message: str = ""


def score(outcome: Outcome) -> float:
    """
    Score an outcome for learning.
    
    Returns:
        Float in range [-1, 1] where:
        - 1.0 = complete success
        - 0.0 = neutral
        - negative = regression
    """
    if outcome.passed:
        return 1.0
    
    # Reward progress (fewer failing tests)
    if outcome.test_delta < 0:
        return min(0.5, -0.1 * outcome.test_delta)
    
    # Penalize regression
    if outcome.test_delta > 0:
        return max(-1.0, -0.2 * outcome.test_delta)
    
    # No change
    return -0.1


def score_patch_quality(
    outcome: Outcome,
    patch_size: int,
    files_touched: int,
) -> float:
    """
    Score that also considers patch quality.
    
    Prefers smaller, more focused patches.
    """
    base = score(outcome)
    
    if not outcome.passed:
        return base
    
    # Bonus for minimal patches
    size_penalty = min(0.2, patch_size / 1000 * 0.1)
    file_penalty = min(0.1, files_touched * 0.02)
    
    return base - size_penalty - file_penalty
