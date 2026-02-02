"""
Tests for RFSN Planner v5 components.
"""

from uuid import UUID

import pytest

from rfsn_controller.planner_v5 import (
    ActionType,
    ExpectedEffect,
    GateRejectionType,
    HypothesisOutcome,
    MetaPlanner,
    Proposal,
    ProposalIntent,
    ProposalPlanner,
    RiskLevel,
    ScoringEngine,
    StateTracker,
    Target,
    TestExpectation,
)


class TestProposal:
    """Test proposal data structures and validation."""

    def test_minimal_valid_proposal(self):
        """Test creating a minimal valid proposal."""
        proposal = Proposal(
            intent=ProposalIntent.ANALYZE,
            hypothesis="This is a test hypothesis for validation",
            action_type=ActionType.READ_FILE,
            target=Target(path="test.py"),
            change_summary="Test summary",
            expected_effect=ExpectedEffect(
                tests=TestExpectation.UNCHANGED,
                behavior="No changes"
            ),
            risk_level=RiskLevel.LOW,
            rollback_plan="No changes to roll back"
        )
        
        assert isinstance(proposal.proposal_id, UUID)
        assert proposal.intent == ProposalIntent.ANALYZE
        assert proposal.risk_level == RiskLevel.LOW

    def test_proposal_validation_missing_hypothesis(self):
        """Test that proposals without hypothesis are rejected."""
        with pytest.raises(ValueError, match="Hypothesis must be"):
            Proposal(
                hypothesis="",  # Too short
                action_type=ActionType.READ_FILE,
                target=Target(path="test.py"),
            )

    def test_proposal_validation_mutation_requires_rollback(self):
        """Test that mutations require rollback plan."""
        with pytest.raises(ValueError, match="Rollback plan required"):
            Proposal(
                hypothesis="Test hypothesis for mutation",
                action_type=ActionType.EDIT_FILE,
                target=Target(path="test.py"),
                change_summary="Edit file",
                rollback_plan="",  # Missing
            )

    def test_proposal_to_dict(self):
        """Test proposal serialization."""
        proposal = Proposal(
            hypothesis="Test serialization hypothesis",
            action_type=ActionType.RUN_TESTS,
            target=Target(path="tests/"),
            change_summary="Run tests",
            rollback_plan="No changes made",
        )
        
        data = proposal.to_dict()
        assert data["intent"] == "analyze"
        assert data["action_type"] == "run_tests"
        assert "proposal_id" in data
        assert data["target"]["path"] == "tests/"

    def test_proposal_is_mutation(self):
        """Test mutation detection."""
        edit = Proposal(
            hypothesis="Edit file for testing",
            action_type=ActionType.EDIT_FILE,
            target=Target(path="test.py"),
            change_summary="Edit file content",
            rollback_plan="Revert changes made",
        )
        assert edit.is_mutation()
        
        read = Proposal(
            hypothesis="Read file for testing",
            action_type=ActionType.READ_FILE,
            target=Target(path="test.py"),
            change_summary="Read file content",
            rollback_plan="No changes to revert",
        )
        assert not read.is_mutation()


class TestStateTracker:
    """Test state tracking functionality."""

    def test_initial_state(self):
        """Test initial state tracker setup."""
        state = StateTracker()
        
        assert not state.has_reproduction()
        assert len(state.failing_tests) == 0
        assert state.current_iteration == 0
        assert not state.is_stuck()

    def test_record_hypothesis(self):
        """Test recording hypothesis outcomes."""
        state = StateTracker()
        proposal_id = UUID("12345678-1234-5678-1234-567812345678")
        
        state.record_hypothesis(
            proposal_id=proposal_id,
            hypothesis="Test hypothesis",
            outcome=HypothesisOutcome.CONFIRMED
        )
        
        assert len(state.hypotheses_tried) == 1
        assert state.hypotheses_tried[0].outcome == HypothesisOutcome.CONFIRMED

    def test_stuck_detection_iterations(self):
        """Test stuck detection via iteration budget."""
        state = StateTracker(iteration_budget=10)
        
        for i in range(10):
            state.increment_iteration()
        
        assert state.is_stuck()

    def test_stuck_detection_same_failure(self):
        """Test stuck detection via repeated failures."""
        state = StateTracker(stuck_threshold=2)
        
        state.record_failure_signature("sig1")
        assert not state.is_stuck()
        
        state.record_failure_signature("sig1")
        assert state.is_stuck()

    def test_suspect_file_tracking(self):
        """Test suspect file management."""
        state = StateTracker()
        
        state.add_suspect_file("module.py", confidence=0.9)
        state.add_suspect_file("other.py", confidence=0.5)
        
        assert state.get_top_suspect_file() == "module.py"

    def test_risk_budget(self):
        """Test risk budget management."""
        state = StateTracker(risk_budget=2)
        
        assert state.can_afford_risk("low")
        assert state.can_afford_risk("medium")
        
        state.spend_risk("medium")
        assert state.can_afford_risk("medium")
        
        state.spend_risk("medium")
        assert not state.can_afford_risk("medium")
        assert state.can_afford_risk("low")  # Low always allowed


class TestProposalPlanner:
    """Test proposal generation."""

    def test_propose_reproduce(self):
        """Test reproduction proposal generation."""
        state = StateTracker()
        planner = ProposalPlanner(state)
        
        proposal = planner.propose_reproduce(test_nodeid="tests/test_x.py::test_y")
        
        assert proposal.intent == ProposalIntent.TEST
        assert proposal.action_type == ActionType.RUN_TESTS
        assert "test_x.py::test_y" in proposal.hypothesis

    def test_propose_localize(self):
        """Test localization proposal generation."""
        state = StateTracker()
        planner = ProposalPlanner(state)
        
        proposal = planner.propose_localize_file(
            file_path="module.py",
            reason="Appears in traceback"
        )
        
        assert proposal.intent == ProposalIntent.ANALYZE
        assert proposal.action_type == ActionType.READ_FILE
        assert proposal.target.path == "module.py"

    def test_propose_add_guard(self):
        """Test guard proposal generation."""
        state = StateTracker()
        planner = ProposalPlanner(state)
        
        proposal = planner.propose_add_guard(
            file_path="module.py",
            symbol="function_name",
            guard_type="none_check",
            expected_behavior="Prevents AttributeError"
        )
        
        assert proposal.intent == ProposalIntent.REPAIR
        assert proposal.action_type == ActionType.EDIT_FILE
        assert "none_check" in proposal.hypothesis.lower()

    def test_extract_traceback_file(self):
        """Test traceback parsing."""
        state = StateTracker()
        planner = ProposalPlanner(state)
        
        traceback = '''
  File "/usr/lib/python3.12/something.py", line 10, in stdlib_func
  File "src/module.py", line 123, in my_function
    x.foo()
AttributeError: 'NoneType' object has no attribute 'foo'
        '''
        
        file_path = planner.extract_traceback_file(traceback)
        assert file_path == "src/module.py"

    def test_extract_exception_type(self):
        """Test exception type extraction."""
        state = StateTracker()
        planner = ProposalPlanner(state)
        
        output = "AttributeError: 'NoneType' object has no attribute 'foo'"
        exc_type = planner.extract_exception_type(output)
        
        assert exc_type == "AttributeError"


class TestScoringEngine:
    """Test proposal scoring."""

    def test_score_traceback_relevance(self):
        """Test scoring based on traceback relevance."""
        scorer = ScoringEngine(
            traceback_frames=[("module.py", 123)],
        )
        
        relevant = Proposal(
            hypothesis="Fix issue in module.py",
            action_type=ActionType.EDIT_FILE,
            target=Target(path="module.py"),
            change_summary="Fix issue in code",
            rollback_plan="Revert changes made",
        )
        
        irrelevant = Proposal(
            hypothesis="Fix issue in other.py",
            action_type=ActionType.EDIT_FILE,
            target=Target(path="other.py"),
            change_summary="Fix issue in code",
            rollback_plan="Revert changes made",
        )
        
        relevant_score = scorer.score_proposal(relevant)
        irrelevant_score = scorer.score_proposal(irrelevant)
        
        assert relevant_score.traceback_relevance > irrelevant_score.traceback_relevance

    def test_score_guard_quality(self):
        """Test scoring based on guard presence."""
        scorer = ScoringEngine()
        
        with_guard = Proposal(
            hypothesis="Add None check to prevent AttributeError",
            action_type=ActionType.EDIT_FILE,
            target=Target(path="module.py"),
            change_summary="Add None check before accessing attributes",
            rollback_plan="Revert guard",
        )
        
        score = scorer.score_proposal(with_guard)
        assert score.guard_quality > 0.0

    def test_select_best_candidate(self):
        """Test selecting best proposal from candidates."""
        scorer = ScoringEngine(
            traceback_frames=[("module.py", 123)],
        )
        
        candidates = [
            Proposal(
                hypothesis="Fix in module.py with None check",
                action_type=ActionType.EDIT_FILE,
                target=Target(path="module.py"),
                change_summary="Add None check guard",
                rollback_plan="Revert changes made",
            ),
            Proposal(
                hypothesis="Refactor entire codebase",
                action_type=ActionType.EDIT_FILE,
                target=Target(path="other.py"),
                change_summary="Large refactor change",
                rollback_plan="Revert changes made",
            ),
        ]
        
        best = scorer.select_best(candidates, top_n=1)
        assert len(best) == 1
        assert best[0].target.path == "module.py"


class TestMetaPlanner:
    """Test meta-planner strategy layer."""

    def test_initial_phase_is_reproduce(self):
        """Test that planner starts in reproduce phase."""
        meta = MetaPlanner()
        
        proposal = meta.next_proposal()
        
        assert proposal.intent == ProposalIntent.TEST
        assert proposal.action_type == ActionType.RUN_TESTS

    def test_phase_transition_reproduce_to_localize(self):
        """Test transition from reproduce to localize."""
        state = StateTracker()
        state.reproduction_confirmed = True
        state.repro_command = "pytest tests/"
        state.add_suspect_file("module.py", confidence=0.9)
        
        meta = MetaPlanner(state_tracker=state)
        
        proposal = meta.next_proposal()
        
        assert proposal.intent == ProposalIntent.ANALYZE
        assert proposal.action_type == ActionType.READ_FILE

    def test_handle_gate_rejection_ordering(self):
        """Test handling ordering violation rejection."""
        meta = MetaPlanner()
        
        # Submit a patch proposal (wrong - should reproduce first)
        # Gate rejects with ORDERING_VIOLATION
        proposal = meta.next_proposal(
            gate_rejection=("ordering_violation", "Must reproduce before patching")
        )
        
        # Should go back to reproduce phase
        assert proposal.intent == ProposalIntent.TEST

    def test_handle_gate_rejection_schema(self):
        """Test handling schema violation rejection."""
        state = StateTracker()
        meta = MetaPlanner(state_tracker=state)
        
        # Multiple schema violations
        for _ in range(2):
            state.gate_rejections.append((GateRejectionType.SCHEMA_VIOLATION, "Missing field"))
        
        proposal = meta.next_proposal(
            gate_rejection=("schema_violation", "Missing rollback_plan")
        )
        
        # Should enter safe mode (analyze only)
        assert proposal.action_type in {ActionType.READ_FILE, ActionType.SEARCH_REPO}

    def test_process_feedback_update_state(self):
        """Test feedback processing updates state."""
        meta = MetaPlanner()
        
        feedback = {
            "success": False,
            "output": "FAILED tests/test_x.py::test_y",
            "tests_failed": 1,
            "tests_passed": 0,
            "traceback": 'File "module.py", line 10\nAttributeError: ...',
        }
        
        meta._process_feedback(feedback)
        
        assert "tests/test_x.py::test_y" in meta.state_tracker.failing_tests
        assert "AttributeError" in meta.state_tracker.exception_types

    def test_stuck_detection(self):
        """Test that meta-planner detects stuck state."""
        state = StateTracker(iteration_budget=5)
        meta = MetaPlanner(state_tracker=state)
        
        # Exhaust iteration budget
        for _ in range(6):
            proposal = meta.next_proposal()
        
        # Should be stuck
        assert state.is_stuck()
        assert "stuck" in proposal.hypothesis.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
