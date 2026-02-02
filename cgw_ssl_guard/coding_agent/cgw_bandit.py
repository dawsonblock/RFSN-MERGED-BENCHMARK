"""CGW Strategy Bandit Integration.

Integrates the StrategyBandit with the CGW proposal scoring mechanism.
The bandit learns which action types work best for different failure patterns
and adjusts proposal saliency accordingly.

Key features:
- UCB/Thompson Sampling for action selection
- Saliency boosting for high-performing strategies
- Automatic exclusion of failing strategies
- Metrics export for dashboard
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class BanditArm:
    """A single arm in the multi-armed bandit."""
    
    name: str
    alpha: float = 1.0  # Beta dist param (successes + 1)
    beta: float = 1.0   # Beta dist param (failures + 1)
    pulls: int = 0
    total_reward: float = 0.0
    
    def sample(self) -> float:
        """Thompson Sampling: draw from Beta posterior."""
        import random
        return random.betavariate(self.alpha, self.beta)
    
    def update(self, reward: float) -> None:
        """Update arm with observed reward (0-1 scale)."""
        self.pulls += 1
        self.total_reward += reward
        self.alpha += reward
        self.beta += (1.0 - reward)
    
    def mean_reward(self) -> float:
        """Mean of Beta posterior."""
        return self.alpha / (self.alpha + self.beta)
    
    def ucb(self, total_pulls: int, c: float = 2.0) -> float:
        """Upper confidence bound."""
        import math
        if self.pulls == 0:
            return float('inf')
        exploration = c * math.sqrt(math.log(total_pulls + 1) / self.pulls)
        return self.mean_reward() + exploration
    
    def as_dict(self) -> Dict[str, Any]:
        """Export arm statistics."""
        return {
            "name": self.name,
            "alpha": self.alpha,
            "beta": self.beta,
            "pulls": self.pulls,
            "total_reward": self.total_reward,
            "mean_reward": self.mean_reward(),
        }


@dataclass
class CGWBanditConfig:
    """Configuration for the CGW bandit."""
    
    # Arms (action types) to track
    action_types: List[str] = None
    
    # Exploration bonus for underexplored arms
    exploration_bonus: float = 0.1
    
    # Decay factor for old observations
    decay_factor: float = 0.99
    
    # Minimum saliency boost
    min_boost: float = 0.8
    
    # Maximum saliency boost
    max_boost: float = 1.5
    
    # Database path for persistence
    db_path: Optional[str] = None
    
    def __post_init__(self):
        if self.action_types is None:
            self.action_types = [
                "RUN_TESTS",
                "ANALYZE_FAILURE",
                "GENERATE_PATCH",
                "APPLY_PATCH",
                "VALIDATE",
                "INSPECT_FILES",
                "LINT",
                "BUILD",
            ]


class CGWBandit:
    """Multi-armed bandit for CGW action selection.
    
    This bandit learns which action types perform best in different
    contexts and provides saliency boosts for proposals.
    
    Usage:
        bandit = CGWBandit()
        
        # Before gate selection, boost proposals
        saliency = bandit.get_saliency_boost("RUN_TESTS", context_hash)
        
        # After execution, update bandit
        bandit.update("RUN_TESTS", reward=1.0 if success else 0.0)
    """
    
    def __init__(
        self,
        config: Optional[CGWBanditConfig] = None,
    ):
        self.config = config or CGWBanditConfig()
        self._arms: Dict[str, BanditArm] = {}
        self._context_arms: Dict[str, Dict[str, BanditArm]] = {}
        self._total_pulls = 0
        self._db: Optional[sqlite3.Connection] = None
        
        # Initialize arms
        for action_type in self.config.action_types:
            self._arms[action_type] = BanditArm(name=action_type)
        
        # Load from database if configured
        if self.config.db_path:
            self._init_db()
            self._load_from_db()
    
    def _init_db(self) -> None:
        """Initialize SQLite database."""
        db_path = self.config.db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._db = sqlite3.connect(db_path)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS bandit_arms (
                name TEXT PRIMARY KEY,
                alpha REAL NOT NULL,
                beta REAL NOT NULL,
                pulls INTEGER NOT NULL,
                total_reward REAL NOT NULL,
                updated_ts INTEGER NOT NULL
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS context_arms (
                context_hash TEXT NOT NULL,
                action_type TEXT NOT NULL,
                alpha REAL NOT NULL,
                beta REAL NOT NULL,
                pulls INTEGER NOT NULL,
                total_reward REAL NOT NULL,
                updated_ts INTEGER NOT NULL,
                PRIMARY KEY (context_hash, action_type)
            )
        """)
        self._db.commit()
    
    def _load_from_db(self) -> None:
        """Load arm state from database."""
        if not self._db:
            return
        
        cursor = self._db.execute(
            "SELECT name, alpha, beta, pulls, total_reward FROM bandit_arms"
        )
        for row in cursor:
            name, alpha, beta, pulls, total_reward = row
            if name in self._arms:
                arm = self._arms[name]
                arm.alpha = alpha
                arm.beta = beta
                arm.pulls = pulls
                arm.total_reward = total_reward
                self._total_pulls += pulls
    
    def _save_arm(self, arm: BanditArm) -> None:
        """Persist arm state to database."""
        if not self._db:
            return
        
        import time
        self._db.execute("""
            INSERT OR REPLACE INTO bandit_arms 
            (name, alpha, beta, pulls, total_reward, updated_ts)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            arm.name, arm.alpha, arm.beta, 
            arm.pulls, arm.total_reward, int(time.time())
        ))
        self._db.commit()
    
    def select_action(
        self,
        *,
        method: str = "thompson",
        exclude: Optional[Set[str]] = None,
    ) -> str:
        """Select an action using the bandit.
        
        Args:
            method: Selection method ("thompson" or "ucb")
            exclude: Action types to exclude from selection
            
        Returns:
            Selected action type name.
        """
        exclude = exclude or set()
        candidates = [
            arm for arm in self._arms.values()
            if arm.name not in exclude
        ]
        
        if not candidates:
            return self.config.action_types[0]
        
        if method == "thompson":
            # Thompson Sampling
            samples = [(arm, arm.sample()) for arm in candidates]
            best = max(samples, key=lambda x: x[1])
            return best[0].name
        else:
            # UCB
            scores = [(arm, arm.ucb(self._total_pulls)) for arm in candidates]
            best = max(scores, key=lambda x: x[1])
            return best[0].name
    
    def get_saliency_boost(
        self,
        action_type: str,
        context_hash: Optional[str] = None,
    ) -> float:
        """Get saliency boost for an action type.
        
        Args:
            action_type: The action type to get boost for.
            context_hash: Optional context hash for context-specific boosts.
            
        Returns:
            Multiplier for proposal saliency (0.8 - 1.5).
        """
        if action_type not in self._arms:
            return 1.0
        
        arm = self._arms[action_type]
        
        # Base boost from mean reward
        mean = arm.mean_reward()
        
        # Add exploration bonus for underexplored arms
        if arm.pulls < 5:
            exploration = self.config.exploration_bonus * (5 - arm.pulls) / 5
        else:
            exploration = 0.0
        
        # Calculate boost
        boost = self.config.min_boost + (
            (self.config.max_boost - self.config.min_boost) * (mean + exploration)
        )
        
        return max(self.config.min_boost, min(self.config.max_boost, boost))
    
    def update(
        self,
        action_type: str,
        reward: float,
        context_hash: Optional[str] = None,
    ) -> None:
        """Update bandit with observed reward.
        
        Args:
            action_type: The action type that was executed.
            reward: Reward value (0.0 = failure, 1.0 = success).
            context_hash: Optional context hash for context-specific learning.
        """
        if action_type not in self._arms:
            logger.warning(f"Unknown action type: {action_type}")
            return
        
        arm = self._arms[action_type]
        arm.update(reward)
        self._total_pulls += 1
        
        # Persist to database
        self._save_arm(arm)
        
        logger.debug(
            f"Bandit update: {action_type} reward={reward:.2f} "
            f"mean={arm.mean_reward():.3f}"
        )
    
    def decay_all(self, factor: Optional[float] = None) -> None:
        """Apply decay to all arms for non-stationarity."""
        factor = factor or self.config.decay_factor
        for arm in self._arms.values():
            arm.alpha = 1.0 + (arm.alpha - 1.0) * factor
            arm.beta = 1.0 + (arm.beta - 1.0) * factor
            self._save_arm(arm)
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all arms."""
        return {
            name: arm.as_dict()
            for name, arm in self._arms.items()
        }
    
    def get_best_action(self) -> str:
        """Get the action with highest mean reward."""
        best = max(self._arms.values(), key=lambda a: a.mean_reward())
        return best.name
    
    def get_avoidance_set(
        self,
        threshold: float = 0.3,
        min_pulls: int = 3,
    ) -> Set[str]:
        """Get action types to avoid (consistently failing).
        
        Args:
            threshold: Maximum mean reward to include in avoidance.
            min_pulls: Minimum pulls to consider for avoidance.
            
        Returns:
            Set of action type names to avoid.
        """
        avoid = set()
        for arm in self._arms.values():
            if arm.pulls >= min_pulls and arm.mean_reward() < threshold:
                avoid.add(arm.name)
        return avoid
    
    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        lines = [
            "# HELP cgw_bandit_pulls_total Total pulls per arm",
            "# TYPE cgw_bandit_pulls_total counter",
        ]
        for arm in self._arms.values():
            lines.append(
                f'cgw_bandit_pulls_total{{action="{arm.name}"}} {arm.pulls}'
            )
        
        lines.extend([
            "",
            "# HELP cgw_bandit_mean_reward Mean reward per arm",
            "# TYPE cgw_bandit_mean_reward gauge",
        ])
        for arm in self._arms.values():
            lines.append(
                f'cgw_bandit_mean_reward{{action="{arm.name}"}} {arm.mean_reward():.4f}'
            )
        
        return "\n".join(lines)
    
    def close(self) -> None:
        """Close database connection."""
        if self._db:
            self._db.close()
            self._db = None


# === Singleton Access ===

_bandit_instance: Optional[CGWBandit] = None


def get_cgw_bandit(
    db_path: Optional[str] = None,
) -> CGWBandit:
    """Get or create the global CGW bandit instance.
    
    Args:
        db_path: Optional database path for persistence.
                 Defaults to ~/.cgw/bandit.db
    """
    global _bandit_instance
    
    if _bandit_instance is None:
        if db_path is None:
            home = os.path.expanduser("~")
            db_path = os.path.join(home, ".cgw", "bandit.db")
        
        config = CGWBanditConfig(db_path=db_path)
        _bandit_instance = CGWBandit(config=config)
    
    return _bandit_instance


def reset_bandit() -> None:
    """Reset the global bandit instance (for testing)."""
    global _bandit_instance
    if _bandit_instance:
        _bandit_instance.close()
    _bandit_instance = None


# === Integration with Proposal Generators ===

class BanditBoostMixin:
    """Mixin to add bandit-based saliency boosting to generators.
    
    Usage:
        class MyGenerator(ProposalGenerator, BanditBoostMixin):
            def generate(self, context):
                candidates = self._generate_candidates(context)
                return self.apply_bandit_boost(candidates)
    """
    
    def apply_bandit_boost(
        self,
        candidates: List[Any],
        context_hash: Optional[str] = None,
    ) -> List[Any]:
        """Apply bandit-based saliency boost to candidates.
        
        Args:
            candidates: List of Candidate objects.
            context_hash: Optional context hash for context-specific boosts.
            
        Returns:
            Candidates with boosted saliency.
        """
        bandit = get_cgw_bandit()
        
        for candidate in candidates:
            if hasattr(candidate, 'content') and 'action' in candidate.content:
                action_type = candidate.content['action']
                if isinstance(action_type, str):
                    boost = bandit.get_saliency_boost(action_type, context_hash)
                    candidate.saliency *= boost
        
        return candidates


def record_action_outcome(
    action_type: str,
    success: bool,
    partial_reward: float = 0.0,
    context_hash: Optional[str] = None,
) -> None:
    """Record an action outcome to the bandit.
    
    Args:
        action_type: The action type that was executed.
        success: Whether the action succeeded.
        partial_reward: Partial reward for partial success (0-1).
        context_hash: Optional context hash.
    """
    bandit = get_cgw_bandit()
    
    if success:
        reward = 1.0
    elif partial_reward > 0:
        reward = partial_reward
    else:
        reward = 0.0
    
    bandit.update(action_type, reward, context_hash)
