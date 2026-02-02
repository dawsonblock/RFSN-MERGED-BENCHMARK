"""Structural learning policy using Thompson Sampling.

This module implements a contextual bandit policy that learns which
actions are most likely to succeed based on past outcomes.
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BanditArm:
    """A single arm in the bandit.
    
    Uses Beta distribution parameters for Thompson Sampling.
    """
    
    name: str
    pulls: int = 0
    successes: int = 0
    alpha: float = 1.0  # Beta prior (successes + 1)
    beta: float = 1.0   # Beta prior (failures + 1)
    
    def update(self, reward: float) -> None:
        """Update the arm based on observed reward.
        
        Args:
            reward: Reward value (0.0 to 1.0).
        """
        self.pulls += 1
        # For binary rewards, treat > 0.5 as success
        if reward > 0.5:
            self.successes += 1
            self.alpha += reward
        else:
            self.beta += (1.0 - reward)
    
    def sample(self, rng: random.Random) -> float:
        """Sample from the posterior Beta distribution.
        
        Args:
            rng: Random number generator.
            
        Returns:
            Sampled probability value.
        """
        return rng.betavariate(self.alpha, self.beta)
    
    def mean(self) -> float:
        """Get the mean of the posterior distribution."""
        return self.alpha / (self.alpha + self.beta)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "name": self.name,
            "pulls": self.pulls,
            "successes": self.successes,
            "alpha": self.alpha,
            "beta": self.beta,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BanditArm:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            pulls=data.get("pulls", 0),
            successes=data.get("successes", 0),
            alpha=data.get("alpha", 1.0),
            beta=data.get("beta", 1.0),
        )


# Default action templates for code repair
DEFAULT_ARMS = [
    "run_tests",           # Run test suite
    "search_stacktrace",   # Search for error patterns
    "read_file",           # Read source file
    "apply_patch",         # Apply a code patch
    "refactor_module",     # Refactor module structure
    "update_deps",         # Update dependencies
    "add_tests",           # Add new tests
    "fix_imports",         # Fix import statements
    "fix_types",           # Fix type annotations
    "rollback",            # Rollback last change
]


@dataclass
class ContextFeatures:
    """Features that describe the current context for action selection."""
    
    feature_mode: str = "repair"
    error_type: str = "unknown"
    repo_size: str = "medium"  # small, medium, large
    language: str = "python"
    last_outcome: str = "none"  # success, fail, none
    step_number: int = 0
    
    def to_vector(self) -> list[float]:
        """Convert to numeric feature vector."""
        # Simple one-hot encoding
        mode_map = {"analysis": 0, "repair": 1, "refactor": 2, "feature": 3}
        error_map = {"unknown": 0, "assertion": 1, "import": 2, "type": 3, "syntax": 4}
        size_map = {"small": 0, "medium": 1, "large": 2}
        outcome_map = {"none": 0, "success": 1, "fail": 2}
        
        return [
            mode_map.get(self.feature_mode, 0) / 3.0,
            error_map.get(self.error_type, 0) / 4.0,
            size_map.get(self.repo_size, 1) / 2.0,
            1.0 if self.language == "python" else 0.0,
            outcome_map.get(self.last_outcome, 0) / 2.0,
            min(self.step_number / 12.0, 1.0),
        ]


class ThompsonBandit:
    """Thompson Sampling bandit for action selection.
    
    Features:
    - Deterministic with fixed seed
    - Persistence to SQLite
    - Context-aware action ordering
    """
    
    def __init__(
        self,
        seed: int = 1337,
        arm_names: list[str] | None = None,
    ) -> None:
        """Initialize the bandit.
        
        Args:
            seed: Random seed for reproducibility.
            arm_names: List of arm names (default: DEFAULT_ARMS).
        """
        self.seed = seed
        self.rng = random.Random(seed)
        self.arms: dict[str, BanditArm] = {}
        
        # Initialize arms
        arm_names = arm_names or DEFAULT_ARMS
        for name in arm_names:
            self.arms[name] = BanditArm(name=name)
    
    def choose(
        self,
        available: list[str] | None = None,
        context: ContextFeatures | None = None,
    ) -> str:
        """Choose the best arm using Thompson Sampling.
        
        Args:
            available: Optional list of available arm names.
            context: Optional context features (currently logged but not used).
            
        Returns:
            Name of the chosen arm.
        """
        candidates = available or list(self.arms.keys())
        
        # Sample from each arm's posterior
        samples: list[tuple[str, float]] = []
        for name in candidates:
            if name not in self.arms:
                # Create new arm with default priors
                self.arms[name] = BanditArm(name=name)
            
            sample = self.arms[name].sample(self.rng)
            samples.append((name, sample))
        
        # Choose arm with highest sample
        samples.sort(key=lambda x: x[1], reverse=True)
        return samples[0][0]
    
    def choose_top_k(
        self,
        k: int,
        available: list[str] | None = None,
    ) -> list[str]:
        """Choose top k arms by Thompson Sampling.
        
        Args:
            k: Number of arms to return.
            available: Optional list of available arm names.
            
        Returns:
            List of top k arm names.
        """
        candidates = available or list(self.arms.keys())
        
        samples: list[tuple[str, float]] = []
        for name in candidates:
            if name not in self.arms:
                self.arms[name] = BanditArm(name=name)
            
            sample = self.arms[name].sample(self.rng)
            samples.append((name, sample))
        
        samples.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in samples[:k]]
    
    def update(self, arm: str, reward: float) -> None:
        """Update an arm based on observed reward.
        
        Args:
            arm: Name of the arm.
            reward: Reward value (0.0 to 1.0).
        """
        if arm not in self.arms:
            self.arms[arm] = BanditArm(name=arm)
        
        self.arms[arm].update(reward)
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics about all arms.
        
        Returns:
            Dictionary with arm statistics.
        """
        return {
            name: {
                "pulls": arm.pulls,
                "successes": arm.successes,
                "mean": arm.mean(),
            }
            for name, arm in self.arms.items()
        }
    
    def save(self, db_path: str) -> None:
        """Save bandit state to SQLite database.
        
        Args:
            db_path: Path to the database file.
        """
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bandit_arms (
                name TEXT PRIMARY KEY,
                pulls INTEGER,
                successes INTEGER,
                alpha REAL,
                beta REAL
            )
        """)
        
        # Upsert each arm
        for arm in self.arms.values():
            cursor.execute("""
                INSERT OR REPLACE INTO bandit_arms (name, pulls, successes, alpha, beta)
                VALUES (?, ?, ?, ?, ?)
            """, (arm.name, arm.pulls, arm.successes, arm.alpha, arm.beta))
        
        conn.commit()
        conn.close()
    
    def load(self, db_path: str) -> None:
        """Load bandit state from SQLite database.
        
        Args:
            db_path: Path to the database file.
        """
        if not Path(db_path).exists():
            return
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT name, pulls, successes, alpha, beta FROM bandit_arms")
            for row in cursor.fetchall():
                name, pulls, successes, alpha, beta = row
                self.arms[name] = BanditArm(
                    name=name,
                    pulls=pulls,
                    successes=successes,
                    alpha=alpha,
                    beta=beta,
                )
        except sqlite3.OperationalError:
            pass  # Table doesn't exist yet
        
        conn.close()
    
    def to_json(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "seed": self.seed,
            "arms": {name: arm.to_dict() for name, arm in self.arms.items()},
        }
    
    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ThompsonBandit:
        """Create from JSON data."""
        bandit = cls(seed=data.get("seed", 1337))
        for name, arm_data in data.get("arms", {}).items():
            bandit.arms[name] = BanditArm.from_dict(arm_data)
        return bandit


def create_policy(
    db_path: str | None = None,
    seed: int = 1337,
) -> ThompsonBandit:
    """Create a configured policy instance.
    
    Args:
        db_path: Optional path to load/save state.
        seed: Random seed.
        
    Returns:
        Configured ThompsonBandit.
    """
    policy = ThompsonBandit(seed=seed)
    
    if db_path:
        policy.load(db_path)
    
    return policy
