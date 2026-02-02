"""Strategy bandit with Thompson Sampling for patch approach selection.
from __future__ import annotations

Implements:
- Multi-armed bandit over patch strategies (temperature, prompt style, etc.)
- Thompson Sampling for exploration/exploitation balance
- Negative memory: tracks failure patterns for avoidance
- Failure feature extraction from error signatures
"""

import hashlib
import json
import logging
import math
import os
import random
import sqlite3
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FailureFeatures:
    """Extracted features from a failure for negative memory attribution."""

    error_class: str  # e.g., "AssertionError", "TypeError"
    stack_signature: str  # Hash of stack trace locations
    touched_files: list[str]  # Files modified by the patch
    test_file: str | None  # Failing test file
    error_message_prefix: str  # First 100 chars of error
    
    def as_dict(self) -> dict[str, Any]:
        return {
            "error_class": self.error_class,
            "stack_signature": self.stack_signature,
            "touched_files": self.touched_files,
            "test_file": self.test_file,
            "error_message_prefix": self.error_message_prefix,
        }
    
    def feature_hash(self) -> str:
        """Hash for similarity matching."""
        content = f"{self.error_class}:{self.stack_signature}:{self.test_file or ''}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class StrategyArm:
    """A strategy arm in the bandit."""

    name: str
    alpha: float = 1.0  # Beta prior successes + 1
    beta: float = 1.0   # Beta prior failures + 1
    total_reward: float = 0.0
    pulls: int = 0
    
    def sample(self) -> float:
        """Draw from Beta posterior (Thompson Sampling)."""
        return random.betavariate(self.alpha, self.beta)
    
    def update(self, reward: float) -> None:
        """Update arm with observed reward (0-1 scale)."""
        reward = max(0.0, min(1.0, reward))
        self.alpha += reward
        self.beta += (1.0 - reward)
        self.total_reward += reward
        self.pulls += 1
    
    @property
    def mean_reward(self) -> float:
        """Mean of Beta posterior."""
        return self.alpha / (self.alpha + self.beta)
    
    @property
    def ucb(self) -> float:
        """Upper confidence bound for comparison."""
        if self.pulls == 0:
            return float('inf')
        mean = self.mean_reward
        exploration = math.sqrt(2 * math.log(self.pulls + 1) / self.pulls)
        return mean + exploration


@dataclass
class NegativeMemoryEntry:
    """Record of a failed attempt for avoidance learning."""

    failure_features: FailureFeatures
    strategy_name: str
    patch_hash: str
    error_count: int = 1
    last_seen_ts: int = 0


class StrategyBandit:
    """Multi-armed bandit for patch strategy selection.
    
    Strategies include:
    - temperature variations (0.0, 0.3, 0.7, 1.0)
    - prompt styles (minimal, verbose, structured)
    - patch granularity (single-hunk, multi-hunk, whole-file)
    """
    
    DEFAULT_STRATEGIES = [
        "temp_0.0",
        "temp_0.3",
        "temp_0.7",
        "temp_1.0",
        "prompt_minimal",
        "prompt_verbose",
        "prompt_structured",
        "granularity_surgical",
        "granularity_moderate",
    ]
    
    def __init__(
        self,
        *,
        strategies: list[str] | None = None,
        exploration_bonus: float = 0.1,
        decay_factor: float = 0.99,
    ):
        """Initialize bandit.
        
        Args:
            strategies: List of strategy names (uses defaults if None).
            exploration_bonus: Bonus for underexplored arms.
            decay_factor: Decay applied to old observations.
        """
        strategy_list = strategies or self.DEFAULT_STRATEGIES
        self.arms: dict[str, StrategyArm] = {
            name: StrategyArm(name=name) for name in strategy_list
        }
        self.exploration_bonus = exploration_bonus
        self.decay_factor = decay_factor
        self.total_pulls = 0
    
    def select_strategy(self, *, exclude: set[str] | None = None) -> str:
        """Select a strategy using Thompson Sampling.
        
        Args:
            exclude: Strategy names to exclude from selection.
        
        Returns:
            Selected strategy name.
        """
        exclude = exclude or set()
        candidates = {k: v for k, v in self.arms.items() if k not in exclude}
        
        if not candidates:
            # Fall back to any arm if all excluded
            candidates = self.arms
        
        # Thompson Sampling: sample from each arm's posterior
        samples = {name: arm.sample() for name, arm in candidates.items()}
        
        # Add exploration bonus for underexplored arms
        for name, arm in candidates.items():
            if arm.pulls < 3:
                samples[name] += self.exploration_bonus * (3 - arm.pulls)
        
        selected = max(samples, key=lambda k: samples[k])
        logger.debug(
            "Selected strategy %s (sample=%.3f, pulls=%d)",
            selected,
            samples[selected],
            self.arms[selected].pulls,
        )
        return selected
    
    def update(self, strategy: str, reward: float) -> None:
        """Update arm with observed reward.
        
        Args:
            strategy: Strategy name.
            reward: Reward value (0.0 = failure, 1.0 = success).
        """
        if strategy not in self.arms:
            logger.warning("Unknown strategy: %s", strategy)
            return
        
        self.arms[strategy].update(reward)
        self.total_pulls += 1
        
        logger.debug(
            "Updated %s: reward=%.2f, mean=%.3f, pulls=%d",
            strategy,
            reward,
            self.arms[strategy].mean_reward,
            self.arms[strategy].pulls,
        )
    
    def get_stats(self) -> dict[str, dict[str, float]]:
        """Get statistics for all arms.
        
        Returns:
            Dictionary of arm name -> stats dict.
        """
        return {
            name: {
                "mean_reward": arm.mean_reward,
                "pulls": arm.pulls,
                "alpha": arm.alpha,
                "beta": arm.beta,
                "ucb": arm.ucb,
            }
            for name, arm in self.arms.items()
        }
    
    def decay_all(self) -> None:
        """Apply decay to all arms (for non-stationarity)."""
        for arm in self.arms.values():
            arm.alpha = 1.0 + (arm.alpha - 1.0) * self.decay_factor
            arm.beta = 1.0 + (arm.beta - 1.0) * self.decay_factor


class NegativeMemoryStore:
    """SQLite-backed store for failure patterns to avoid.
    
    Tracks:
    - Error class + stack signature combinations
    - Strategies that consistently fail for certain patterns
    - Patch approaches to avoid for similar contexts
    """
    
    def __init__(self, db_path: str):
        """Initialize store.
        
        Args:
            db_path: Path to SQLite database.
        """
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init_schema()
    
    def _init_schema(self) -> None:
        """Initialize database schema."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS negative_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_hash TEXT NOT NULL,
                error_class TEXT NOT NULL,
                stack_signature TEXT NOT NULL,
                test_file TEXT,
                strategy_name TEXT NOT NULL,
                patch_hash TEXT NOT NULL,
                error_count INTEGER NOT NULL DEFAULT 1,
                last_seen_ts INTEGER NOT NULL,
                features_json TEXT NOT NULL,
                UNIQUE(feature_hash, strategy_name, patch_hash)
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_negative_error_class
            ON negative_memory (error_class, strategy_name)
        """)
        self.conn.commit()
    
    def record_failure(
        self,
        features: FailureFeatures,
        strategy: str,
        patch_hash: str,
        timestamp: int,
    ) -> None:
        """Record a failure for negative memory.
        
        Args:
            features: Extracted failure features.
            strategy: Strategy that was used.
            patch_hash: Hash of the patch that failed.
            timestamp: Current timestamp.
        """
        feature_hash = features.feature_hash()
        try:
            self.conn.execute("""
                INSERT INTO negative_memory (
                    feature_hash, error_class, stack_signature, test_file,
                    strategy_name, patch_hash, error_count, last_seen_ts, features_json
                ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(feature_hash, strategy_name, patch_hash) DO UPDATE SET
                    error_count = error_count + 1,
                    last_seen_ts = excluded.last_seen_ts
            """, (
                feature_hash,
                features.error_class,
                features.stack_signature,
                features.test_file,
                strategy,
                patch_hash,
                timestamp,
                json.dumps(features.as_dict()),
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            logger.warning("Failed to record negative memory: %s", e)
    
    def get_failed_strategies(
        self,
        error_class: str,
        *,
        min_failures: int = 2,
        limit: int = 10,
    ) -> list[tuple[str, int]]:
        """Get strategies that frequently fail for an error class.
        
        Args:
            error_class: The error class to look up.
            min_failures: Minimum failure count to include.
            limit: Maximum strategies to return.
        
        Returns:
            List of (strategy_name, failure_count) tuples.
        """
        cur = self.conn.cursor()
        cur.execute("""
            SELECT strategy_name, SUM(error_count) as total_failures
            FROM negative_memory
            WHERE error_class = ?
            GROUP BY strategy_name
            HAVING total_failures >= ?
            ORDER BY total_failures DESC
            LIMIT ?
        """, (error_class, min_failures, limit))
        return cur.fetchall()
    
    def should_avoid_strategy(
        self,
        error_class: str,
        strategy: str,
        *,
        threshold: int = 3,
    ) -> bool:
        """Check if a strategy should be avoided for an error class.
        
        Args:
            error_class: The error class.
            strategy: Strategy to check.
            threshold: Failure count threshold.
        
        Returns:
            True if strategy should be avoided.
        """
        cur = self.conn.cursor()
        cur.execute("""
            SELECT SUM(error_count)
            FROM negative_memory
            WHERE error_class = ? AND strategy_name = ?
        """, (error_class, strategy))
        result = cur.fetchone()[0]
        return (result or 0) >= threshold
    
    def get_avoidance_set(
        self,
        error_class: str,
        *,
        threshold: int = 3,
    ) -> set[str]:
        """Get set of strategies to avoid for an error class.
        
        Args:
            error_class: The error class.
            threshold: Failure count threshold.
        
        Returns:
            Set of strategy names to avoid.
        """
        failed = self.get_failed_strategies(error_class, min_failures=threshold)
        return {name for name, _ in failed}
    
    def close(self) -> None:
        """Close database connection."""
        try:
            self.conn.close()
        except Exception:
            pass


def extract_failure_features(
    *,
    stderr: str,
    stdout: str,
    patch_diff: str,
    test_file: str | None = None,
) -> FailureFeatures:
    """Extract features from a test failure for negative memory.
    
    Args:
        stderr: Standard error from test run.
        stdout: Standard output from test run.
        patch_diff: The patch that was applied.
        test_file: Known failing test file (if any).
    
    Returns:
        FailureFeatures extracted from the failure.
    """
    combined = (stderr or "") + "\n" + (stdout or "")
    
    # Extract error class
    error_class = "Unknown"
    for line in combined.splitlines():
        line = line.strip()
        # Common Python error patterns
        if ": " in line and line[0].isupper():
            parts = line.split(": ", 1)
            if parts[0].endswith("Error") or parts[0].endswith("Exception"):
                error_class = parts[0].split(".")[-1]  # Remove module path
                break
    
    # Extract stack signature (hash of file:line locations)
    stack_lines = []
    for line in combined.splitlines():
        if "File " in line and ", line " in line:
            # Extract filename and line number
            try:
                parts = line.split('"')
                if len(parts) >= 2:
                    filename = parts[1].split("/")[-1]
                    line_part = line.split(", line ")[1].split(",")[0]
                    stack_lines.append(f"{filename}:{line_part}")
            except (IndexError, ValueError):
                pass
    stack_signature = hashlib.sha256(
        ":".join(stack_lines[:10]).encode()
    ).hexdigest()[:12]
    
    # Extract touched files from patch
    touched_files = []
    for line in (patch_diff or "").splitlines():
        if line.startswith("+++ b/") or line.startswith("--- a/"):
            path = line.split("/", 1)[-1] if "/" in line else line[6:]
            if path and path not in touched_files:
                touched_files.append(path)
    
    # Error message prefix
    error_message = ""
    for line in combined.splitlines():
        if error_class in line and ": " in line:
            error_message = line.split(": ", 1)[-1][:100]
            break
    
    return FailureFeatures(
        error_class=error_class,
        stack_signature=stack_signature,
        touched_files=touched_files[:10],
        test_file=test_file,
        error_message_prefix=error_message,
    )


class LearningOrchestrator:
    """Coordinates strategy bandit with negative memory for informed selection."""
    
    def __init__(
        self,
        *,
        bandit: StrategyBandit | None = None,
        negative_store: NegativeMemoryStore | None = None,
        avoidance_threshold: int = 3,
    ):
        """Initialize orchestrator.
        
        Args:
            bandit: Strategy bandit (creates default if None).
            negative_store: Negative memory store (optional).
            avoidance_threshold: Failures before avoiding strategy.
        """
        self.bandit = bandit or StrategyBandit()
        self.negative_store = negative_store
        self.avoidance_threshold = avoidance_threshold
    
    def select_strategy(
        self,
        *,
        error_class: str | None = None,
    ) -> str:
        """Select strategy with negative memory awareness.
        
        Args:
            error_class: Current error class for avoidance lookup.
        
        Returns:
            Selected strategy name.
        """
        avoid = set()
        
        if error_class and self.negative_store:
            avoid = self.negative_store.get_avoidance_set(
                error_class,
                threshold=self.avoidance_threshold,
            )
            if avoid:
                logger.info(
                    "Avoiding strategies for %s: %s",
                    error_class,
                    ", ".join(sorted(avoid)),
                )
        
        return self.bandit.select_strategy(exclude=avoid)
    
    def record_outcome(
        self,
        *,
        strategy: str,
        success: bool,
        features: FailureFeatures | None = None,
        patch_hash: str = "",
        timestamp: int = 0,
    ) -> None:
        """Record strategy outcome.
        
        Args:
            strategy: Strategy that was used.
            success: Whether the attempt succeeded.
            features: Failure features (for failures only).
            patch_hash: Hash of the patch.
            timestamp: Current timestamp.
        """
        # Update bandit
        reward = 1.0 if success else 0.0
        self.bandit.update(strategy, reward)
        
        # Record negative memory for failures
        if not success and features and self.negative_store:
            self.negative_store.record_failure(
                features=features,
                strategy=strategy,
                patch_hash=patch_hash,
                timestamp=timestamp,
            )
