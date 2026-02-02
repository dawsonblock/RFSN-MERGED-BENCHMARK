"""Learned Strategy Selector.

Combines fingerprinting, bandit selection, and quarantine
to provide intelligent strategy recommendations.

This is the main entry point for the learning layer.
It coordinates all learning components while respecting
the constraint that learning only operates in proposal space.

SAFETY INVARIANT: This module NEVER executes anything.
It only returns recommendations. The controller decides
whether to follow them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .fingerprint import (
    FailureFingerprint,
    compute_fingerprint_hash,
    fingerprint_failure,
)
from .quarantine import QuarantineLane
from .strategy_bandit import StrategyBandit

logger = logging.getLogger(__name__)


@dataclass
class StrategyRecommendation:
    """Recommendation from the learning layer.
    
    This is a SUGGESTION only. The controller/gate has final authority.
    """
    
    # Recommended strategy
    strategy: str
    
    # Confidence (0-1)
    confidence: float
    
    # Fingerprint context
    fingerprint_hash: str
    
    # Alternative strategies (fallbacks)
    alternatives: list[str]
    
    # Strategies to avoid
    quarantined: set[str]
    
    # Reasoning (for debugging/audit)
    reasoning: str
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "strategy": self.strategy,
            "confidence": round(self.confidence, 3),
            "fingerprint": self.fingerprint_hash,
            "alternatives": self.alternatives,
            "quarantined": list(self.quarantined),
            "reasoning": self.reasoning,
        }


class LearnedStrategySelector:
    """Main entry point for strategy learning.
    
    Combines:
    - Fingerprinting: Recognizes failure patterns
    - Bandit: Selects strategies via Thompson Sampling
    - Quarantine: Excludes risky strategies
    
    Usage:
        selector = LearnedStrategySelector()
        
        # Get recommendation
        rec = selector.recommend(
            failing_tests=["test_foo"],
            lint_errors=["E501: line too long"],
        )
        
        # Use rec.strategy in planner
        # After outcome:
        selector.update(rec, success=True, regression=False)
    """
    
    def __init__(
        self,
        strategies: list[str] | None = None,
    ):
        """Initialize selector.
        
        Args:
            strategies: List of strategy names to use.
        """
        self.bandit = StrategyBandit(strategies=strategies)
        self.quarantine = QuarantineLane()
        
        # Cache of fingerprints for update correlation
        self._fingerprint_cache: dict[str, FailureFingerprint] = {}
    
    def recommend(
        self,
        failing_tests: list[str] | None = None,
        lint_errors: list[str] | None = None,
        stack_trace: str | None = None,
        affected_files: list[str] | None = None,
        exclude: set[str] | None = None,
    ) -> StrategyRecommendation:
        """Get a strategy recommendation for the current failure.
        
        Args:
            failing_tests: List of failing test names.
            lint_errors: List of lint error messages.
            stack_trace: Optional stack trace.
            affected_files: Files involved in failure.
            exclude: Additional strategies to exclude.
            
        Returns:
            StrategyRecommendation with suggested strategy.
        """
        # Build fingerprint
        fingerprint = fingerprint_failure(
            failing_tests=failing_tests,
            lint_errors=lint_errors,
            stack_trace=stack_trace,
            affected_files=affected_files,
        )
        fp_hash = compute_fingerprint_hash(fingerprint)
        
        # Cache for later update
        self._fingerprint_cache[fp_hash] = fingerprint
        
        # Get quarantined strategies
        quarantined = self.quarantine.get_quarantined_strategies(fp_hash)
        all_excluded = (exclude or set()) | quarantined
        
        # Select via bandit
        strategy = self.bandit.select(fp_hash, exclude=all_excluded)
        
        # Get stats for confidence
        stats = self.bandit.get_stats(fp_hash)
        strategy_stats = stats.get(strategy, {})
        confidence = strategy_stats.get("success_rate", 0.5)
        
        # Get alternatives
        alternatives = self._get_alternatives(fp_hash, exclude=all_excluded | {strategy})
        
        # Build reasoning
        reasoning = self._build_reasoning(fingerprint, strategy, quarantined)
        
        logger.info(
            "Recommended strategy '%s' (confidence=%.2f) for %s",
            strategy, confidence, fingerprint.category
        )
        
        return StrategyRecommendation(
            strategy=strategy,
            confidence=confidence,
            fingerprint_hash=fp_hash,
            alternatives=alternatives[:3],
            quarantined=quarantined,
            reasoning=reasoning,
        )
    
    def update(
        self,
        recommendation: StrategyRecommendation,
        success: bool,
        regression: bool = False,
        partial_reward: float | None = None,
    ) -> None:
        """Update learning based on outcome.
        
        Args:
            recommendation: The recommendation that was followed.
            success: Whether the strategy succeeded.
            regression: Whether the strategy caused a regression.
            partial_reward: Optional partial reward.
        """
        fp_hash = recommendation.fingerprint_hash
        strategy = recommendation.strategy
        
        # Update bandit
        self.bandit.update(
            fp_hash,
            strategy,
            success=success,
            regression=regression,
            partial_reward=partial_reward,
        )
        
        # Update quarantine
        self.quarantine.record_outcome(
            strategy,
            fp_hash,
            success=success,
            regression=regression,
        )
        
        logger.info(
            "Updated learning for %s/%s: success=%s, regression=%s",
            strategy, fp_hash[:8], success, regression
        )
    
    def _get_alternatives(
        self,
        fp_hash: str,
        exclude: set[str],
    ) -> list[str]:
        """Get alternative strategies ranked by expected reward."""
        stats = self.bandit.get_stats(fp_hash)
        
        candidates = [
            (s, data.get("mean_reward", 0.0))
            for s, data in stats.items()
            if s not in exclude
        ]
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in candidates]
    
    def _build_reasoning(
        self,
        fingerprint: FailureFingerprint,
        strategy: str,
        quarantined: set[str],
    ) -> str:
        """Build human-readable reasoning for recommendation."""
        parts = [
            f"Category: {fingerprint.category}",
            f"Selected: {strategy}",
        ]
        
        if fingerprint.error_class:
            parts.append(f"Error: {fingerprint.error_class}")
        
        if quarantined:
            parts.append(f"Quarantined: {', '.join(quarantined)}")
        
        return "; ".join(parts)
    
    def get_stats(self) -> dict:
        """Get combined statistics from all components."""
        return {
            "bandit": self.bandit.get_stats(),
            "quarantine": {
                "global": list(self.quarantine._global_quarantine),
            },
            "fingerprints_cached": len(self._fingerprint_cache),
        }
    
    def force_quarantine(self, strategy: str, reason: str = "manual") -> None:
        """Force-quarantine a strategy globally.
        
        Args:
            strategy: Strategy to quarantine.
            reason: Reason for quarantine.
        """
        self.quarantine.force_quarantine(strategy, reason)
