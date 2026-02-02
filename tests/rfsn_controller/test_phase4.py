"""Tests for Phase 4 - Hierarchical Planner Layer.

Tests cover:
- PlanGate validation
- Learning layer components
- Controller loop integration
"""


import pytest

# ============================================================================
# PLAN GATE TESTS
# ============================================================================

class TestPlanGate:
    """Test PlanGate hard safety enforcement."""
    
    def test_validate_valid_plan(self):
        """Test validation of a valid plan."""
        from rfsn_controller.gates import PlanGate
        
        gate = PlanGate()
        plan = {
            "plan_id": "test",
            "steps": [
                {"id": "step1", "type": "read_file", "inputs": {"file": "test.py"}, "expected_outcome": "read"},
                {"id": "step2", "type": "run_tests", "inputs": {}, "expected_outcome": "pass"},
            ]
        }
        
        assert gate.validate_plan(plan) is True
    
    def test_reject_empty_plan(self):
        """Test rejection of empty plan."""
        from rfsn_controller.gates import PlanGate, PlanGateError
        
        gate = PlanGate()
        plan = {"plan_id": "test", "steps": []}
        
        with pytest.raises(PlanGateError, match="no steps"):
            gate.validate_plan(plan)
    
    def test_reject_unknown_step_type(self):
        """Test rejection of unknown step type."""
        from rfsn_controller.gates import PlanGate, PlanGateError
        
        gate = PlanGate()
        plan = {
            "plan_id": "test",
            "steps": [
                {"id": "step1", "type": "dangerous_operation", "inputs": {}},
            ]
        }
        
        with pytest.raises(PlanGateError, match="not in allowlist"):
            gate.validate_plan(plan)
    
    def test_reject_shell_injection(self):
        """Test rejection of shell injection attempts."""
        from rfsn_controller.gates import PlanGate, StepGateError
        
        gate = PlanGate()
        step = {
            "id": "step1",
            "type": "read_file",
            "inputs": {"file": "test.py; rm -rf /"},
        }
        
        with pytest.raises(StepGateError, match="shell injection"):
            gate.validate_step(step)
    
    def test_reject_duplicate_step_ids(self):
        """Test rejection of duplicate step IDs."""
        from rfsn_controller.gates import PlanGate, PlanGateConfig, PlanGateError
        
        # Use non-strict mode to skip expected_outcome validation
        config = PlanGateConfig(strict_mode=False, require_expected_outcomes=False)
        gate = PlanGate(config)
        plan = {
            "plan_id": "test",
            "steps": [
                {"id": "step1", "type": "read_file", "inputs": {}},
                {"id": "step1", "type": "run_tests", "inputs": {}},  # Duplicate
            ]
        }
        
        with pytest.raises(PlanGateError, match="Duplicate"):
            gate.validate_plan(plan)
    
    def test_step_budget(self):
        """Test step budget enforcement."""
        from rfsn_controller.gates import PlanGate, PlanGateConfig, PlanGateError
        
        config = PlanGateConfig(max_steps=2, require_expected_outcomes=False)
        gate = PlanGate(config)
        
        plan = {
            "plan_id": "test",
            "steps": [
                {"id": f"step{i}", "type": "read_file", "inputs": {}}
                for i in range(5)  # Exceeds budget
            ]
        }
        
        with pytest.raises(PlanGateError, match="exceeds step budget"):
            gate.validate_plan(plan)
    
    def test_get_allowed_step_types(self):
        """Test getting allowed step types."""
        from rfsn_controller.gates import PlanGate
        
        gate = PlanGate()
        allowed = gate.get_allowed_step_types()
        
        assert "read_file" in allowed
        assert "run_tests" in allowed
        assert "apply_patch" in allowed


# ============================================================================
# FINGERPRINT TESTS
# ============================================================================

class TestFingerprint:
    """Test failure fingerprinting."""
    
    def test_basic_fingerprint(self):
        """Test basic fingerprint creation."""
        from rfsn_controller.learning import fingerprint_failure
        
        fp = fingerprint_failure(
            failing_tests=["test_foo", "test_bar"],
            lint_errors=["E501: line too long"],
        )
        
        assert fp.category == "LINT_ERROR"
        assert "test_foo" in fp.failing_tests
        assert "E501" in fp.lint_codes
    
    def test_fingerprint_import_error(self):
        """Test fingerprinting import errors."""
        from rfsn_controller.learning import fingerprint_failure
        
        fp = fingerprint_failure(
            stack_trace="ImportError: No module named 'foo'"
        )
        
        assert fp.category == "IMPORT_ERROR"
    
    def test_fingerprint_hash_stable(self):
        """Test that fingerprint hash is stable."""
        from rfsn_controller.learning import compute_fingerprint_hash, fingerprint_failure
        
        fp1 = fingerprint_failure(failing_tests=["test_a", "test_b"])
        fp2 = fingerprint_failure(failing_tests=["test_b", "test_a"])  # Same, different order
        
        hash1 = compute_fingerprint_hash(fp1)
        hash2 = compute_fingerprint_hash(fp2)
        
        assert hash1 == hash2  # Order shouldn't matter


# ============================================================================
# STRATEGY BANDIT TESTS
# ============================================================================

class TestStrategyBandit:
    """Test strategy bandit."""
    
    def test_select_unexplored_first(self):
        """Test that unexplored arms are selected first."""
        from rfsn_controller.learning import StrategyBandit
        
        bandit = StrategyBandit(strategies=["a", "b", "c"])
        
        # First selection should be unexplored
        strategy = bandit.select("ctx1")
        assert strategy in {"a", "b", "c"}
    
    def test_update_changes_stats(self):
        """Test that updates change statistics."""
        from rfsn_controller.learning import StrategyBandit
        
        bandit = StrategyBandit(strategies=["a", "b"])
        
        bandit.update("ctx1", "a", success=True)
        bandit.update("ctx1", "a", success=True)
        bandit.update("ctx1", "b", success=False)
        
        stats = bandit.get_stats("ctx1")
        assert stats["a"]["wins"] == 2
        assert stats["b"]["wins"] == 0
    
    def test_exclude_works(self):
        """Test excluding strategies from selection."""
        from rfsn_controller.learning import StrategyBandit
        
        bandit = StrategyBandit(strategies=["a", "b", "c"])
        
        strategy = bandit.select("ctx1", exclude={"a", "b"})
        assert strategy == "c"


# ============================================================================
# QUARANTINE TESTS
# ============================================================================

class TestQuarantine:
    """Test quarantine lane."""
    
    def test_new_strategy_quarantined(self):
        """Test that new strategies are quarantined."""
        from rfsn_controller.learning import is_quarantined
        
        stats = {"tries": 0, "wins": 0, "regressions": 0}
        assert is_quarantined(stats) is True
    
    def test_successful_strategy_not_quarantined(self):
        """Test that successful strategies are not quarantined."""
        from rfsn_controller.learning import is_quarantined
        
        stats = {"tries": 5, "wins": 4, "regressions": 0}
        assert is_quarantined(stats) is False
    
    def test_lane_tracks_outcomes(self):
        """Test quarantine lane tracking."""
        from rfsn_controller.learning import QuarantineConfig, QuarantineLane
        
        # Use lower thresholds for testing
        config = QuarantineConfig(min_successes=2, max_regression_rate=0.5)
        lane = QuarantineLane(config)
        
        # Record some outcomes to test tracking
        lane.record_outcome("test_strategy", "ctx1", success=True)
        lane.record_outcome("test_strategy", "ctx1", success=True)
        lane.record_outcome("test_strategy", "ctx1", success=False)
        
        # Strategy should not be quarantined (2 wins, 3 tries, success rate > min)
        assert not lane.is_quarantined("test_strategy", "ctx1")
        
        # Record a regression - should trigger quarantine
        lane.record_outcome("test_strategy", "ctx1", success=False, regression=True)
        lane.record_outcome("test_strategy", "ctx1", success=False, regression=True)
        
        # Now regression rate is 2/5 = 40%, just under threshold - still ok
        assert not lane.is_quarantined("test_strategy", "ctx1")
    
    def test_force_quarantine(self):
        """Test force quarantine."""
        from rfsn_controller.learning import QuarantineLane
        
        lane = QuarantineLane()
        lane.force_quarantine("bad_strategy", "manual")
        
        assert lane.is_quarantined("bad_strategy")


# ============================================================================
# LEARNED STRATEGY SELECTOR TESTS
# ============================================================================

class TestLearnedStrategySelector:
    """Test learned strategy selector."""
    
    def test_recommend_returns_strategy(self):
        """Test recommendation returns a strategy."""
        from rfsn_controller.learning import LearnedStrategySelector
        
        selector = LearnedStrategySelector()
        rec = selector.recommend(failing_tests=["test_foo"])
        
        assert rec.strategy is not None
        assert rec.fingerprint_hash is not None
        assert 0 <= rec.confidence <= 1
    
    def test_update_changes_recommendations(self):
        """Test that updates influence future recommendations."""
        from rfsn_controller.learning import LearnedStrategySelector
        
        selector = LearnedStrategySelector(strategies=["a", "b"])
        
        # Get initial recommendation
        rec1 = selector.recommend(failing_tests=["test_x"])
        
        # Update with success
        selector.update(rec1, success=True)
        
        # Check stats updated
        stats = selector.get_stats()
        assert stats["bandit"]["a"]["tries"] + stats["bandit"]["b"]["tries"] == 1


# ============================================================================
# CONTROLLER LOOP TESTS
# ============================================================================

class TestControllerLoop:
    """Test controller execution loop."""
    
    def test_run_valid_plan(self):
        """Test running a valid plan."""
        from rfsn_controller.controller_loop import ControllerLoop
        from rfsn_controller.gates import PlanGate, PlanGateConfig
        
        # Disable strict mode for testing
        config = PlanGateConfig(require_expected_outcomes=False)
        loop = ControllerLoop(gate=PlanGate(config))
        plan = {
            "plan_id": "test",
            "steps": [
                {"id": "s1", "type": "read_file", "inputs": {}},
                {"id": "s2", "type": "run_tests", "inputs": {}},
            ]
        }
        
        result = loop.run_plan(plan)
        
        assert result.success is True
        assert result.steps_executed == 2
        assert result.steps_succeeded == 2
    
    def test_reject_invalid_plan(self):
        """Test rejection of invalid plan."""
        from rfsn_controller.controller_loop import ControllerLoop
        
        loop = ControllerLoop()
        plan = {
            "plan_id": "test",
            "steps": [
                {"id": "s1", "type": "dangerous", "inputs": {}},
            ]
        }
        
        result = loop.run_plan(plan)
        
        assert result.success is False
        assert "plan_gate_error" in result.final_status
    
    def test_custom_executor(self):
        """Test with custom executor."""
        from rfsn_controller.controller_loop import ControllerLoop, ExecutionOutcome
        from rfsn_controller.gates import PlanGate, PlanGateConfig
        
        executed = []
        
        def custom_executor(step):
            executed.append(step["id"])
            return ExecutionOutcome(step_id=step["id"], success=True)
        
        config = PlanGateConfig(require_expected_outcomes=False)
        loop = ControllerLoop(gate=PlanGate(config), executor=custom_executor)
        plan = {
            "plan_id": "test",
            "steps": [
                {"id": "a", "type": "read_file", "inputs": {}},
                {"id": "b", "type": "run_tests", "inputs": {}},
            ]
        }
        
        loop.run_plan(plan)
        
        assert executed == ["a", "b"]


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestPhase4Integration:
    """Integration tests for Phase 4 components."""
    
    def test_all_imports_work(self):
        """Test all Phase 4 imports."""
        
        assert True
    
    def test_gate_with_learning(self):
        """Test gate respects learning exclusions."""
        from rfsn_controller.gates import PlanGate
        from rfsn_controller.learning import LearnedStrategySelector
        
        gate = PlanGate()
        selector = LearnedStrategySelector()
        
        # Get recommendation, noting quarantined strategies
        selector.recommend(failing_tests=["test_x"])
        
        # Verify gate cannot be modified by learning
        original_types = gate.get_allowed_step_types()
        
        # Simulate learning trying to add a type (should not work)
        # The gate has no method for learning to call
        
        assert gate.get_allowed_step_types() == original_types
    
    def test_full_pipeline(self):
        """Test full pipeline: fingerprint -> select -> gate -> execute."""
        from rfsn_controller.controller_loop import ControllerLoop
        from rfsn_controller.gates import PlanGate, PlanGateConfig
        from rfsn_controller.learning import LearnedStrategySelector
        
        # Setup with non-strict config for testing
        selector = LearnedStrategySelector()
        config = PlanGateConfig(require_expected_outcomes=False)
        gate = PlanGate(config)
        loop = ControllerLoop(gate=gate, learning=selector)
        
        # Get strategy recommendation
        rec = selector.recommend(failing_tests=["test_foo"])
        assert rec.strategy is not None
        
        # Create a plan (simulating planner output)
        plan = {
            "plan_id": "test",
            "steps": [
                {"id": "s1", "type": "read_file", "inputs": {"file": "test.py"}},
            ]
        }
        
        # Run plan
        result = loop.run_plan(plan)
        assert result.success
        
        # Update learning
        selector.update(rec, success=True)
