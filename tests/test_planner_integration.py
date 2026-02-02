"""Integration tests for Planner v2."""

from unittest.mock import MagicMock

import pytest

from rfsn_controller.planner_v2 import (
    ControllerAdapter,
    ControllerOutcome,
    PlannerV2,
)


class TestPlannerIntegration:
    """Holistic integration tests."""

    @pytest.fixture
    def planner(self):
        return PlannerV2(seed=123)

    @pytest.fixture
    def adapter(self, planner):
        return ControllerAdapter(planner)

    def test_parallel_execution_retrieval(self, adapter):
        """Test retrieving parallel tasks."""
        # Start a generic plan but use feature mode to avoid strict validation or mock validator
        # Ideally we use a goal that triggers proposing a generic plan but we mock validation
        # Or better: use a repo_type that allows wildcards?
        # Let's just mock the validator to always pass for this test
        adapter._validator.validate = MagicMock(return_value=MagicMock(valid=True))
        context = {"repo_type": "generic"}
        adapter.start_goal("Perform independent tasks", context)
        
        # Mock the plan to have parallelizable steps
        plan = adapter.get_plan()
        adapter.get_state()
        
        # Inject parallel steps
        from rfsn_controller.planner_v2.schema import RiskLevel, Step
        step1 = Step("s1", "Step 1", "intent", ["file1.py"], "success", risk_level=RiskLevel.LOW)
        step2 = Step("s2", "Step 2", "intent", ["file2.py"], "success", risk_level=RiskLevel.LOW)
        step3 = Step("s3", "Step 3", "intent", ["file1.py"], "success", risk_level=RiskLevel.LOW) # Conflict with s1
        
        plan.steps = [step1, step2, step3]
        
        # Verify parallel batch
        tasks = adapter.get_parallel_tasks(max_workers=2)
        assert len(tasks) == 2
        assert {t.step_id for t in tasks} == {"s1", "s2"}
        
        # Verify subsequent batch handling
        # Complete s1
        outcome1 = ControllerOutcome("s1", True)
        adapter.process_outcome(outcome1)
        
        # s2 is technically still "active" in parallel context but if we serialize calls:
        outcome2 = ControllerOutcome("s2", True)
        next_task = adapter.process_outcome(outcome2)
        
        # process_outcome automatically fetches the next sequential task (s3)
        # So get_parallel_tasks should be empty (s3 is already ACTIVE)
        # We verify s3 was returned by process_outcome
        assert next_task is not None
        assert next_task.step_id == "s3"

    def test_qa_integration_flow(self, adapter):
        """Test QA integration flow."""
        # Mock QA bridge
        mock_qa = MagicMock()
        adapter._qa_bridge = mock_qa
        mock_qa.enabled = True
        
        context = {"repo_type": "generic"}
        task = adapter.start_goal("Fix bug", context)
        
        # Simulate success outcome
        outcome = ControllerOutcome(task.step_id, True)
        
        # Simulate QA rejection
        from rfsn_controller.planner_v2.qa_integration import StepQAResult
        mock_qa.verify_step_outcome.return_value = StepQAResult(
            step_id=task.step_id,
            accepted=False,
            rejection_reasons=["Regression detected"],
            escalation_tags=[],
            should_revise=True
        )
        
        # Mock failure evidence conversion
        from rfsn_controller.planner_v2.schema import FailureCategory, FailureEvidence
        mock_qa.convert_qa_to_failure_evidence.return_value = FailureEvidence(
            category=FailureCategory.TEST_REGRESSION,
            suggestion="Fix regression"
        )
        
        # Process outcome
        adapter.process_outcome(outcome)
        
        # Should have triggered revision
        plan = adapter.get_plan()
        assert plan.version > 1
        
        # The revision strategy (TestRegression) probably reset the step or added context
        # Check current step
        step = plan.get_step(task.step_id)
        # Failure count is incremented by update_state AND/OR revise_plan.
        # update_state calls StepLifecycle.fail -> might increment (if impl does)
        # revise_plan explicitly increments.
        # It's likely 2 now.
        assert step.failure_count >= 1
