"""Reward function for SWE-bench task outcomes.

Defines a structured reward function with binary success and optional
shaping rewards for curriculum learning.

INVARIANTS:
1. Final reward is always in [-1.0, 1.0] range
2. Binary success (tests pass + gate accepts) = 1.0
3. Shaping rewards are optional and configurable
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OutcomeType(Enum):
    """Types of task outcomes."""
    SUCCESS = "success"  # Tests pass, gate accepts, patch applies
    PARTIAL = "partial"  # Some progress but not complete
    REJECTED = "rejected"  # Gate rejected the proposal
    FAILED = "failed"  # Tests failed or patch didn't apply
    ERROR = "error"  # System error (not counted)


@dataclass
class TaskOutcome:
    """Structured outcome of a SWE-bench task attempt.
    
    Attributes:
        outcome_type: The type of outcome
        tests_passed: Number of tests that passed
        tests_total: Total number of tests
        patch_applied: Whether the patch applied cleanly
        gate_accepted: Whether the verification gate accepted
        rejection_reason: Reason for gate rejection (if any)
        error_message: System error message (if any)
        metadata: Additional outcome metadata
    """
    outcome_type: OutcomeType
    tests_passed: int = 0
    tests_total: int = 0
    patch_applied: bool = False
    gate_accepted: bool = False
    rejection_reason: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def test_pass_rate(self) -> float:
        """Calculate test pass rate (0.0 to 1.0)."""
        if self.tests_total == 0:
            return 0.0
        return self.tests_passed / self.tests_total
    
    @property
    def is_success(self) -> bool:
        """Check if this is a full success."""
        return (
            self.outcome_type == OutcomeType.SUCCESS
            and self.patch_applied
            and self.gate_accepted
            and self.tests_passed == self.tests_total
        )


@dataclass
class RewardConfig:
    """Configuration for the reward function.
    
    Attributes:
        success_reward: Reward for full success
        partial_reward_base: Base reward for partial progress
        rejection_penalty: Penalty for gate rejection
        failure_penalty: Penalty for test failure
        shaping_enabled: Whether to use shaping rewards
        patch_apply_bonus: Bonus for clean patch application
        test_progress_weight: Weight for partial test progress
    """
    success_reward: float = 1.0
    partial_reward_base: float = 0.0
    rejection_penalty: float = -0.1
    failure_penalty: float = -0.2
    shaping_enabled: bool = True
    patch_apply_bonus: float = 0.1
    test_progress_weight: float = 0.3
    
    def validate(self) -> None:
        """Validate configuration values."""
        assert -1.0 <= self.success_reward <= 1.0
        assert -1.0 <= self.partial_reward_base <= 1.0
        assert -1.0 <= self.rejection_penalty <= 0.0
        assert -1.0 <= self.failure_penalty <= 0.0


def compute_reward(
    outcome: TaskOutcome,
    config: RewardConfig | None = None,
) -> float:
    """Compute the reward for a task outcome.
    
    The reward function follows these rules:
    1. Full success (tests pass, gate accepts, patch applies) = success_reward
    2. Gate rejection = rejection_penalty
    3. Test failure = failure_penalty + shaping rewards
    4. Shaping rewards (if enabled):
       - patch_apply_bonus if patch applied cleanly
       - test_progress_weight * test_pass_rate for partial progress
    
    Args:
        outcome: The task outcome to evaluate
        config: Reward configuration (uses defaults if None)
        
    Returns:
        Reward value in [-1.0, 1.0] range
    """
    if config is None:
        config = RewardConfig()
    
    # Handle system errors (no learning signal)
    if outcome.outcome_type == OutcomeType.ERROR:
        return 0.0
    
    # Full success
    if outcome.is_success:
        return config.success_reward
    
    # Start with base reward for outcome type
    reward = 0.0
    
    if outcome.outcome_type == OutcomeType.REJECTED:
        reward = config.rejection_penalty
    elif outcome.outcome_type == OutcomeType.FAILED:
        reward = config.failure_penalty
    elif outcome.outcome_type == OutcomeType.PARTIAL:
        reward = config.partial_reward_base
    
    # Apply shaping rewards if enabled
    if config.shaping_enabled:
        # Bonus for clean patch application
        if outcome.patch_applied:
            reward += config.patch_apply_bonus
        
        # Partial credit for test progress
        if outcome.test_pass_rate > 0:
            reward += config.test_progress_weight * outcome.test_pass_rate
    
    # Clamp to valid range
    return max(-1.0, min(1.0, reward))


def create_success_outcome(
    tests_total: int,
    metadata: dict[str, Any] | None = None,
) -> TaskOutcome:
    """Create a successful task outcome.
    
    Args:
        tests_total: Total number of tests
        metadata: Optional metadata
        
    Returns:
        TaskOutcome representing full success
    """
    return TaskOutcome(
        outcome_type=OutcomeType.SUCCESS,
        tests_passed=tests_total,
        tests_total=tests_total,
        patch_applied=True,
        gate_accepted=True,
        metadata=metadata or {},
    )


def create_failure_outcome(
    tests_passed: int,
    tests_total: int,
    patch_applied: bool = True,
    error_message: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> TaskOutcome:
    """Create a failed task outcome.
    
    Args:
        tests_passed: Number of tests that passed
        tests_total: Total number of tests
        patch_applied: Whether the patch applied
        error_message: Optional error message
        metadata: Optional metadata
        
    Returns:
        TaskOutcome representing failure
    """
    return TaskOutcome(
        outcome_type=OutcomeType.FAILED,
        tests_passed=tests_passed,
        tests_total=tests_total,
        patch_applied=patch_applied,
        gate_accepted=False,
        error_message=error_message,
        metadata=metadata or {},
    )


def create_rejection_outcome(
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> TaskOutcome:
    """Create a gate rejection outcome.
    
    Args:
        reason: Reason for rejection
        metadata: Optional metadata
        
    Returns:
        TaskOutcome representing gate rejection
    """
    return TaskOutcome(
        outcome_type=OutcomeType.REJECTED,
        rejection_reason=reason,
        metadata=metadata or {},
    )
