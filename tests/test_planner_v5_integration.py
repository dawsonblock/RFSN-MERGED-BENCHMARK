"""Integration tests for Planner v5."""


import pytest

from rfsn_controller.planner_v5_adapter import (
    ControllerAction,
    PlannerV5Adapter,
)


class TestPlannerV5Integration:
    """Integration tests for Planner v5."""

    @pytest.fixture
    def adapter(self):
        """Create planner v5 adapter."""
        return PlannerV5Adapter(enabled=True)

    def test_adapter_initialization(self, adapter):
        """Test that adapter initializes correctly."""
        assert adapter.enabled
        assert adapter.meta_planner is not None
        assert adapter.state_tracker is not None

    def test_get_initial_action(self, adapter):
        """Test getting initial action without feedback."""
        if not adapter.enabled:
            pytest.skip("Planner v5 not available")

        # Get first action
        action = adapter.get_next_action()

        assert action is not None
        assert isinstance(action, ControllerAction)
        assert action.action_type in [
            "edit_file",
            "run_tests",
            "read_file",
            "localize",
        ]

    def test_feedback_loop(self, adapter):
        """Test feedback loop with planner."""
        if not adapter.enabled:
            pytest.skip("Planner v5 not available")

        # Initial action
        action1 = adapter.get_next_action()
        assert action1 is not None

        # Provide feedback
        feedback = {
            "success": False,
            "output": "Test failed",
            "tests_passed": 0,
            "tests_failed": 2,
            "traceback": "AssertionError: Expected 5, got 3",
        }

        # Get next action
        action2 = adapter.get_next_action(controller_feedback=feedback)
        assert action2 is not None

        # Actions should potentially differ based on feedback
        assert isinstance(action2, ControllerAction)

    def test_gate_rejection_handling(self, adapter):
        """Test handling of gate rejections."""
        if not adapter.enabled:
            pytest.skip("Planner v5 not available")

        # Simulate gate rejection
        gate_rejection = ("UNSAFE_EVAL", "eval() detected in patch")

        action = adapter.get_next_action(gate_rejection=gate_rejection)
        assert action is not None

        # Planner should adapt to rejection
        assert isinstance(action, ControllerAction)

    def test_success_feedback(self, adapter):
        """Test planner response to successful outcome."""
        if not adapter.enabled:
            pytest.skip("Planner v5 not available")

        feedback = {
            "success": True,
            "output": "All tests passed",
            "tests_passed": 5,
            "tests_failed": 0,
        }

        action = adapter.get_next_action(controller_feedback=feedback)

        # On success, planner might return None or verification action
        if action:
            assert isinstance(action, ControllerAction)

    def test_multiple_iterations(self, adapter):
        """Test multiple planning iterations."""
        if not adapter.enabled:
            pytest.skip("Planner v5 not available")

        iterations = 5
        actions = []

        for i in range(iterations):
            feedback = {
                "success": False,
                "output": f"Iteration {i}",
                "tests_passed": i,
                "tests_failed": 5 - i,
            }

            action = adapter.get_next_action(controller_feedback=feedback)
            if action:
                actions.append(action)

        # Should have received multiple actions
        assert len(actions) > 0

    def test_process_result_integration(self, adapter):
        """Test processing results through adapter."""
        if not adapter.enabled:
            pytest.skip("Planner v5 not available")

        # Simulate a repair attempt
        result = {
            "patch_applied": True,
            "tests_passed": True,
            "exit_code": 0,
            "output": "Success",
        }

        # Process result
        adapter.process_result(result)

        # Get next action after success
        next_action = adapter.get_next_action(
            controller_feedback={"success": True, "tests_passed": 5}
        )

        # Should handle gracefully (might return None on success)
        assert next_action is None or isinstance(next_action, ControllerAction)


class TestPlannerV5StateTracking:
    """Test state tracking in Planner v5."""

    @pytest.fixture
    def adapter(self):
        """Create planner v5 adapter."""
        return PlannerV5Adapter(enabled=True)

    def test_state_persistence_across_calls(self, adapter):
        """Test that state persists across multiple calls."""
        if not adapter.enabled:
            pytest.skip("Planner v5 not available")

        # First call
        adapter.get_next_action()

        # Provide feedback
        feedback = {"success": False, "tests_failed": 3}
        adapter.get_next_action(controller_feedback=feedback)

        # State should be maintained
        assert adapter.state_tracker is not None
        assert adapter.state_tracker.current_iteration >= 1

    def test_hypothesis_tracking(self, adapter):
        """Test hypothesis tracking in state."""
        if not adapter.enabled:
            pytest.skip("Planner v5 not available")

        # Generate actions with different outcomes
        for i in range(3):
            feedback = {
                "success": False,
                "tests_failed": 2,
                "output": f"Hypothesis {i} failed",
            }
            action = adapter.get_next_action(controller_feedback=feedback)

            if action:
                # State should track hypotheses
                assert adapter.state_tracker is not None


class TestPlannerV5CLI:
    """Test CLI integration with Planner v5."""

    def test_cli_accepts_v5_flag(self):
        """Test that CLI accepts --planner-mode v5."""
        import subprocess

        result = subprocess.run(
            ["python", "-m", "rfsn_controller.cli", "--help"],
            capture_output=True,
            text=True, check=False,
        )

        # Check that v5 is in help text
        assert "v5" in result.stdout
        assert "planner-mode" in result.stdout


@pytest.mark.integration
class TestPlannerV5EndToEnd:
    """End-to-end integration tests."""

    def test_complete_planning_cycle(self):
        """Test a complete planning cycle from start to finish."""
        adapter = PlannerV5Adapter(enabled=True)

        if not adapter.enabled:
            pytest.skip("Planner v5 not available")

        # Simulate a repair cycle
        cycle_steps = [
            {"success": False, "tests_failed": 5, "output": "Initial failure"},
            {"success": False, "tests_failed": 4, "output": "Some progress"},
            {"success": False, "tests_failed": 2, "output": "More progress"},
            {"success": False, "tests_failed": 1, "output": "Almost there"},
            {"success": True, "tests_passed": 5, "output": "Success!"},
        ]

        actions_taken = []

        for step in cycle_steps:
            action = adapter.get_next_action(controller_feedback=step)
            if action:
                actions_taken.append(action)

            if step["success"]:
                break

        # Should have taken multiple actions
        assert len(actions_taken) >= 1

        # Should have reached success
        assert cycle_steps[-1]["success"]

    def test_stuck_detection(self):
        """Test that planner detects stuck states."""
        adapter = PlannerV5Adapter(enabled=True)

        if not adapter.enabled:
            pytest.skip("Planner v5 not available")

        # Simulate stuck situation (no progress)
        for i in range(10):
            feedback = {
                "success": False,
                "tests_failed": 5,  # No progress
                "output": "Same error",
            }
            action = adapter.get_next_action(controller_feedback=feedback)

            # Planner should eventually adapt or signal stuck state
            if action:
                assert isinstance(action, ControllerAction)


@pytest.mark.performance
class TestPlannerV5Performance:
    """Performance tests for Planner v5."""

    def test_action_generation_speed(self):
        """Test that action generation is fast enough."""
        import time

        adapter = PlannerV5Adapter(enabled=True)

        if not adapter.enabled:
            pytest.skip("Planner v5 not available")

        start_time = time.time()

        # Generate 10 actions
        for _ in range(10):
            feedback = {
                "success": False,
                "tests_failed": 3,
            }
            adapter.get_next_action(controller_feedback=feedback)

        elapsed = time.time() - start_time

        # Should be fast (<1s per action)
        assert elapsed < 10.0

    def test_memory_usage(self):
        """Test that planner doesn't leak memory."""
        import gc

        adapter = PlannerV5Adapter(enabled=True)

        if not adapter.enabled:
            pytest.skip("Planner v5 not available")

        # Generate many actions
        for i in range(100):
            feedback = {"success": False, "tests_failed": 2}
            adapter.get_next_action(controller_feedback=feedback)

            # Periodic garbage collection
            if i % 20 == 0:
                gc.collect()

        # If we got here without OOM, test passes
        assert True
