"""Adaptive patch budget control with escalation ladder.
from __future__ import annotations

Implements the Patch Budget Controller as specified:
- Surgical mode (default): tiny diffs only
- Escalation mode: widens limits only after stagnation is proven
- Hard ceiling: never exceeds unless user explicitly opts in
"""

import logging
from dataclasses import dataclass, field
from enum import IntEnum

logger = logging.getLogger(__name__)


class BudgetTier(IntEnum):
    """Escalation tiers for patch budget."""

    SURGICAL = 0  # Default: minimal changes
    MODERATE = 1  # After 2 stagnant attempts
    EXPANDED = 2  # Structural failures only
    CEILING = 3  # Hard stop, user opt-in only


# Tier limits: (max_lines, max_files)
TIER_LIMITS: dict[BudgetTier, tuple[int, int]] = {
    BudgetTier.SURGICAL: (80, 3),
    BudgetTier.MODERATE: (150, 5),
    BudgetTier.EXPANDED: (300, 8),
    BudgetTier.CEILING: (500, 15),
}


@dataclass
class PatchBudgetController:
    """Adaptive patch budget with escalation ladder.

    Starts in surgical mode (small patches only). Escalates to higher
    tiers only after detecting stagnation (consecutive failures with
    identical failing test sets).

    The CEILING tier requires explicit user opt-in via `user_ceiling_override`.

    Attributes:
        current_tier: Current escalation tier (0-3).
        consecutive_stagnant: Count of consecutive stagnant attempts.
        last_failing_tests: Set of failing test names from last attempt.
        user_ceiling_override: If True, allow escalation to CEILING tier.
        stagnation_threshold: Number of stagnant attempts before escalation (default: 2).
    """

    current_tier: BudgetTier = BudgetTier.SURGICAL
    consecutive_stagnant: int = 0
    last_failing_tests: set[str] = field(default_factory=set)
    user_ceiling_override: bool = False
    stagnation_threshold: int = 2

    def get_limits(self) -> tuple[int, int]:
        """Get current (max_lines, max_files) limits.

        Returns:
            Tuple of (max_lines_changed, max_files_changed).
        """
        return TIER_LIMITS[self.current_tier]

    def record_attempt(
        self,
        failing_tests: set[str],
        success: bool,
    ) -> None:
        """Record a patch attempt and update stagnation tracking.

        Args:
            failing_tests: Set of test names that failed after this attempt.
            success: Whether the patch was accepted (tests passed).
        """
        if success:
            # Success resets stagnation counter but keeps current tier
            self.consecutive_stagnant = 0
            self.last_failing_tests = set()
            logger.info(
                "Patch succeeded at tier %s, resetting stagnation counter",
                self.current_tier.name,
            )
            return

        # Check for stagnation: same failing tests as last attempt
        is_stagnant = (
            len(self.last_failing_tests) > 0
            and failing_tests == self.last_failing_tests
        )

        if is_stagnant:
            self.consecutive_stagnant += 1
            logger.info(
                "Stagnant attempt %d/%d (same %d failing tests)",
                self.consecutive_stagnant,
                self.stagnation_threshold,
                len(failing_tests),
            )
        else:
            # Different failures - reset stagnation but record new baseline
            self.consecutive_stagnant = 1
            logger.debug(
                "New failure pattern: %d tests (was %d)",
                len(failing_tests),
                len(self.last_failing_tests),
            )

        self.last_failing_tests = failing_tests.copy()

    def should_escalate(self) -> bool:
        """Check if escalation is warranted based on stagnation.

        Returns:
            True if consecutive stagnant attempts >= threshold and not at max tier.
        """
        if self.consecutive_stagnant < self.stagnation_threshold:
            return False

        # Check if we can escalate (not already at ceiling or ceiling locked)
        max_tier = (
            BudgetTier.CEILING if self.user_ceiling_override else BudgetTier.EXPANDED
        )
        return self.current_tier < max_tier

    def escalate(self) -> bool:
        """Attempt to escalate to the next tier.

        Returns:
            True if escalation succeeded, False if already at maximum tier.
        """
        max_tier = (
            BudgetTier.CEILING if self.user_ceiling_override else BudgetTier.EXPANDED
        )

        if self.current_tier >= max_tier:
            logger.warning(
                "Cannot escalate beyond %s (ceiling_override=%s)",
                self.current_tier.name,
                self.user_ceiling_override,
            )
            return False

        old_tier = self.current_tier
        self.current_tier = BudgetTier(self.current_tier + 1)
        self.consecutive_stagnant = 0  # Reset after escalation

        old_limits = TIER_LIMITS[old_tier]
        new_limits = TIER_LIMITS[self.current_tier]

        logger.info(
            "Escalated: %s → %s (lines: %d→%d, files: %d→%d)",
            old_tier.name,
            self.current_tier.name,
            old_limits[0],
            new_limits[0],
            old_limits[1],
            new_limits[1],
        )
        return True

    def reset(self) -> None:
        """Reset controller to initial state (surgical mode)."""
        self.current_tier = BudgetTier.SURGICAL
        self.consecutive_stagnant = 0
        self.last_failing_tests = set()
        logger.debug("Patch budget controller reset to SURGICAL")

    def get_state_summary(self) -> dict:
        """Get current state for logging/telemetry.

        Returns:
            Dictionary with current tier, limits, and stagnation info.
        """
        max_lines, max_files = self.get_limits()
        return {
            "tier": self.current_tier.name,
            "tier_value": int(self.current_tier),
            "max_lines": max_lines,
            "max_files": max_files,
            "consecutive_stagnant": self.consecutive_stagnant,
            "stagnation_threshold": self.stagnation_threshold,
            "ceiling_override": self.user_ceiling_override,
            "can_escalate": self.should_escalate(),
        }


def create_patch_budget_controller(
    *,
    user_ceiling_override: bool = False,
    stagnation_threshold: int = 2,
    initial_tier: BudgetTier | None = None,
) -> PatchBudgetController:
    """Factory function to create a PatchBudgetController.

    Args:
        user_ceiling_override: Allow escalation to CEILING tier.
        stagnation_threshold: Stagnant attempts before escalation.
        initial_tier: Starting tier (defaults to SURGICAL).

    Returns:
        Configured PatchBudgetController instance.
    """
    tier = initial_tier if initial_tier is not None else BudgetTier.SURGICAL
    return PatchBudgetController(
        current_tier=tier,
        user_ceiling_override=user_ceiling_override,
        stagnation_threshold=stagnation_threshold,
    )
