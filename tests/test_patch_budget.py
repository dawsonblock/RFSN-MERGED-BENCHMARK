"""Unit tests for PatchBudgetController, DiffMinimizer, and TestDeltaTracker."""

import pytest


class TestPatchBudgetController:
    """Tests for PatchBudgetController adaptive escalation."""

    def test_starts_at_surgical(self):
        """Budget controller starts at SURGICAL tier with low limits."""
        from rfsn_controller.patch_budget import (
            BudgetTier,
            create_patch_budget_controller,
        )

        budget = create_patch_budget_controller()
        assert budget.current_tier == BudgetTier.SURGICAL
        assert budget.get_limits() == (80, 3)

    def test_no_escalation_on_progress(self):
        """Successful patches don't trigger escalation."""
        from rfsn_controller.patch_budget import (
            BudgetTier,
            create_patch_budget_controller,
        )

        budget = create_patch_budget_controller()
        
        # Record success
        budget.record_attempt(failing_tests=set(), success=True)
        
        assert budget.current_tier == BudgetTier.SURGICAL
        assert not budget.should_escalate()

    def test_escalate_after_two_stagnant(self):
        """Escalation after 2 consecutive stagnant failures."""
        from rfsn_controller.patch_budget import (
            BudgetTier,
            create_patch_budget_controller,
        )

        budget = create_patch_budget_controller(stagnation_threshold=2)
        failing = {"test_foo", "test_bar"}
        
        # First failure sets baseline
        budget.record_attempt(failing_tests=failing, success=False)
        assert not budget.should_escalate()
        
        # Second identical failure triggers stagnation
        budget.record_attempt(failing_tests=failing, success=False)
        assert budget.should_escalate()
        
        # Escalate
        assert budget.escalate()
        assert budget.current_tier == BudgetTier.MODERATE
        assert budget.get_limits() == (150, 5)

    def test_ceiling_requires_override(self):
        """CEILING tier requires explicit override."""
        from rfsn_controller.patch_budget import (
            BudgetTier,
            create_patch_budget_controller,
        )

        # Without override, max is EXPANDED
        budget = create_patch_budget_controller(user_ceiling_override=False)
        budget.current_tier = BudgetTier.EXPANDED
        
        assert not budget.escalate()  # Cannot escalate without override
        assert budget.current_tier == BudgetTier.EXPANDED
        
        # With override, can reach CEILING
        budget = create_patch_budget_controller(user_ceiling_override=True)
        budget.current_tier = BudgetTier.EXPANDED
        budget.consecutive_stagnant = 2
        
        assert budget.escalate()
        assert budget.current_tier == BudgetTier.CEILING
        assert budget.get_limits() == (500, 15)

    def test_reset_clears_state(self):
        """Reset returns to initial state."""
        from rfsn_controller.patch_budget import (
            BudgetTier,
            create_patch_budget_controller,
        )

        budget = create_patch_budget_controller()
        budget.current_tier = BudgetTier.MODERATE
        budget.consecutive_stagnant = 5
        budget.last_failing_tests = {"test_a"}
        
        budget.reset()
        
        assert budget.current_tier == BudgetTier.SURGICAL
        assert budget.consecutive_stagnant == 0
        assert len(budget.last_failing_tests) == 0


class TestDiffMinimizer:
    """Tests for DiffMinimizer patch shrinking."""

    def test_drops_whitespace_only_hunks(self):
        """Hunks with only whitespace changes are dropped."""
        from rfsn_controller.diff_minimizer import DiffMinimizer

        minimizer = DiffMinimizer()
        # True whitespace-only: only adds/removes blank lines
        diff = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,3 +1,4 @@
 def foo():
+
     pass
 end
"""
        result = minimizer.minimize(diff)
        assert result.dropped_hunks == 1
        assert "def foo():" not in result.minimized  # Hunk was dropped

    def test_preserves_functional_changes(self):
        """Functional code changes are preserved."""
        from rfsn_controller.diff_minimizer import DiffMinimizer

        minimizer = DiffMinimizer()
        diff = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 def foo():
-    return 1
+    return 2
 end
"""
        result = minimizer.minimize(diff)
        assert result.dropped_hunks == 0
        assert "return 2" in result.minimized

    def test_splits_independent_files(self):
        """Multi-file diffs can be split."""
        from rfsn_controller.diff_minimizer import DiffMinimizer

        minimizer = DiffMinimizer()
        diff = """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1 +1 @@
-x = 1
+x = 2
diff --git a/b.py b/b.py
--- a/b.py
+++ b/b.py
@@ -1 +1 @@
-y = 1
+y = 2
"""
        parts = minimizer.split_independent(diff)
        assert len(parts) == 2
        assert any("a.py" in p for p in parts)
        assert any("b.py" in p for p in parts)


class TestDeltaTracker:
    """Tests for TestDeltaTracker regression detection."""

    def test_detects_fix(self):
        """Fixed tests are tracked correctly."""
        from rfsn_controller.verifier import TestDeltaTracker

        tracker = TestDeltaTracker({"test_a", "test_b"})
        
        fixed, regressed = tracker.compute_delta({"test_b"})
        
        assert "test_a" in fixed
        assert len(regressed) == 0
        assert tracker.get_fixed_count({"test_b"}) == 1

    def test_detects_regression(self):
        """Regressions are detected correctly."""
        from rfsn_controller.verifier import TestDeltaTracker

        tracker = TestDeltaTracker({"test_a"})
        
        fixed, regressed = tracker.compute_delta({"test_a", "test_b"})
        
        assert len(fixed) == 0
        assert "test_b" in regressed
        assert tracker.has_regressions({"test_a", "test_b"})
        assert tracker.get_regression_count({"test_a", "test_b"}) == 1

    def test_no_false_positives(self):
        """No regressions when test set unchanged."""
        from rfsn_controller.verifier import TestDeltaTracker

        tracker = TestDeltaTracker({"test_a", "test_b"})
        
        assert not tracker.has_regressions({"test_a", "test_b"})
        assert tracker.get_regression_count({"test_a", "test_b"}) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
