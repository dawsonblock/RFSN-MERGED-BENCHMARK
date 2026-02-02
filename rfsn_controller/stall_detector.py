"""Stall detection for the repair loop.
from __future__ import annotations

Tracks improvement over iterations and detects when the controller
is making no progress for an extended period.
"""

from dataclasses import dataclass


@dataclass
class StallState:
    """Track state for stall detection."""

    failing_tests_count: int = 0
    failing_test_id: str | None = None
    error_signature: str = ""
    iterations_without_improvement: int = 0
    stall_threshold: int = 3

    def update(self, failing_count: int, test_id: str | None, sig: str) -> bool:
        """Update state and return True if stall detected.

        Args:
            failing_count: Number of failing tests.
            test_id: Top failing test ID.
            sig: Error signature.

        Returns:
            True if stall detected, False otherwise.
        """
        # Check if there's any improvement
        improved = (
            failing_count < self.failing_tests_count
            or test_id != self.failing_test_id
            or sig != self.error_signature
        )

        if improved:
            self.failing_tests_count = failing_count
            self.failing_test_id = test_id
            self.error_signature = sig
            self.iterations_without_improvement = 0
            return False
        else:
            self.iterations_without_improvement += 1
            return self.iterations_without_improvement >= self.stall_threshold

    def get_score(self) -> tuple[int, bool]:
        """Get current score tuple for comparison.

        Returns:
            Tuple of (failing_count, signature_changed_bool).
        """
        return (self.failing_tests_count, self.error_signature != "")

    def is_stalled(self) -> bool:
        """Check if currently stalled."""
        return self.iterations_without_improvement >= self.stall_threshold

    def reset(self) -> None:
        """Reset stall state."""
        self.failing_tests_count = 0
        self.failing_test_id = None
        self.error_signature = ""
        self.iterations_without_improvement = 0
