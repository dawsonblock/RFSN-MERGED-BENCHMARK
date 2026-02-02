"""Tests for the planner_v2 module.

Tests cover:
- Schema serialization/deserialization
- Step lifecycle state machine
- Plan generation for different goal types
- Controller adapter integration
"""

import json

from rfsn_controller.planner_v2 import (
    ControllerAdapter,
    ControllerOutcome,
    MemoryAdapter,
    Plan,
    PlannerV2,
    PlanState,
    RiskLevel,
    Step,
    StepLifecycle,
    StepStatus,
)

# =============================================================================
# Schema Tests
# =============================================================================


class TestStepStatus:
    """Tests for StepStatus enum."""

    def test_all_statuses_exist(self) -> None:
        """Verify all expected statuses exist."""
        assert StepStatus.PENDING.value == "PENDING"
        assert StepStatus.ACTIVE.value == "ACTIVE"
        assert StepStatus.DONE.value == "DONE"
        assert StepStatus.FAILED.value == "FAILED"
        assert StepStatus.SKIPPED.value == "SKIPPED"
        assert StepStatus.BLOCKED.value == "BLOCKED"


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_all_levels_exist(self) -> None:
        """Verify all expected risk levels exist."""
        assert RiskLevel.LOW.value == "LOW"
        assert RiskLevel.MED.value == "MED"
        assert RiskLevel.HIGH.value == "HIGH"


class TestStep:
    """Tests for Step dataclass."""

    def test_create_step(self) -> None:
        """Test creating a step."""
        step = Step(
            step_id="test-step",
            title="Test Step",
            intent="Do something",
            allowed_files=["src/*.py"],
            success_criteria="Tests pass",
        )
        assert step.step_id == "test-step"
        assert step.status == StepStatus.PENDING
        assert step.failure_count == 0

    def test_step_to_dict_and_back(self) -> None:
        """Test step serialization."""
        step = Step(
            step_id="test-step",
            title="Test Step",
            intent="Do something",
            allowed_files=["src/*.py"],
            success_criteria="Tests pass",
            dependencies=["other-step"],
            risk_level=RiskLevel.MED,
        )
        d = step.to_dict()
        restored = Step.from_dict(d)

        assert restored.step_id == step.step_id
        assert restored.dependencies == step.dependencies
        assert restored.risk_level == step.risk_level

    def test_get_task_spec(self) -> None:
        """Test generating task spec from step."""
        step = Step(
            step_id="test-step",
            title="Test Step",
            intent="Do something",
            allowed_files=["src/*.py"],
            success_criteria="Tests pass",
            verify="pytest -q",
        )
        spec = step.get_task_spec()

        assert spec.step_id == "test-step"
        assert spec.intent == "Do something"
        assert spec.verify_cmd == "pytest -q"


class TestPlan:
    """Tests for Plan dataclass."""

    def test_create_plan(self) -> None:
        """Test creating a plan."""
        plan = Plan(
            plan_id="test-plan",
            goal="Fix bug",
            steps=[
                Step(
                    step_id="step1",
                    title="Step 1",
                    intent="First step",
                    allowed_files=[],
                    success_criteria="Done",
                )
            ],
            created_at="2026-01-25T00:00:00Z",
        )
        assert plan.plan_id == "test-plan"
        assert len(plan.steps) == 1

    def test_plan_to_json_and_back(self) -> None:
        """Test plan JSON serialization."""
        plan = Plan(
            plan_id="test-plan",
            goal="Fix bug",
            steps=[
                Step(
                    step_id="step1",
                    title="Step 1",
                    intent="First step",
                    allowed_files=["*.py"],
                    success_criteria="Done",
                )
            ],
            created_at="2026-01-25T00:00:00Z",
            assumptions=["Bug exists"],
        )
        json_str = plan.to_json()
        restored = Plan.from_json(json_str)

        assert restored.plan_id == plan.plan_id
        assert restored.assumptions == plan.assumptions
        assert len(restored.steps) == 1

    def test_get_step(self) -> None:
        """Test getting step by ID."""
        plan = Plan(
            plan_id="test-plan",
            goal="Fix bug",
            steps=[
                Step(step_id="a", title="A", intent="A", allowed_files=[], success_criteria=""),
                Step(step_id="b", title="B", intent="B", allowed_files=[], success_criteria=""),
            ],
            created_at="2026-01-25T00:00:00Z",
        )
        assert plan.get_step("a") is not None
        assert plan.get_step("a").title == "A"
        assert plan.get_step("nonexistent") is None


class TestPlanState:
    """Tests for PlanState dataclass."""

    def test_create_state(self) -> None:
        """Test creating plan state."""
        state = PlanState(plan_id="test-plan")
        assert state.current_step_idx == 0
        assert state.halted is False

    def test_state_to_dict_and_back(self) -> None:
        """Test state serialization."""
        state = PlanState(
            plan_id="test-plan",
            completed_steps=["step1"],
            consecutive_failures=1,
        )
        d = state.to_dict()
        restored = PlanState.from_dict(d)

        assert restored.completed_steps == ["step1"]
        assert restored.consecutive_failures == 1


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestStepLifecycle:
    """Tests for StepLifecycle state machine."""

    def test_can_activate_no_deps(self) -> None:
        """Test activation check with no dependencies."""
        step = Step(
            step_id="a",
            title="A",
            intent="A",
            allowed_files=[],
            success_criteria="",
            dependencies=[],
        )
        plan = Plan(
            plan_id="test",
            goal="",
            steps=[step],
            created_at="",
        )
        can, _ = StepLifecycle.can_activate(step, plan)
        assert can is True

    def test_can_activate_deps_not_done(self) -> None:
        """Test activation blocked by pending dependency."""
        step_a = Step(step_id="a", title="A", intent="", allowed_files=[], success_criteria="")
        step_b = Step(
            step_id="b",
            title="B",
            intent="",
            allowed_files=[],
            success_criteria="",
            dependencies=["a"],
        )
        plan = Plan(plan_id="test", goal="", steps=[step_a, step_b], created_at="")

        can, _ = StepLifecycle.can_activate(step_b, plan)
        assert can is False

    def test_can_activate_deps_done(self) -> None:
        """Test activation allowed when deps are done."""
        step_a = Step(step_id="a", title="A", intent="", allowed_files=[], success_criteria="")
        step_a.status = StepStatus.DONE
        step_b = Step(
            step_id="b",
            title="B",
            intent="",
            allowed_files=[],
            success_criteria="",
            dependencies=["a"],
        )
        plan = Plan(plan_id="test", goal="", steps=[step_a, step_b], created_at="")

        can, _ = StepLifecycle.can_activate(step_b, plan)
        assert can is True

    def test_activate(self) -> None:
        """Test step activation."""
        step = Step(step_id="a", title="A", intent="", allowed_files=[], success_criteria="")
        StepLifecycle.activate(step)
        assert step.status == StepStatus.ACTIVE

    def test_complete(self) -> None:
        """Test step completion."""
        step = Step(step_id="a", title="A", intent="", allowed_files=[], success_criteria="")
        StepLifecycle.complete(step, {"ok": True})
        assert step.status == StepStatus.DONE
        assert step.result == {"ok": True}

    def test_fail_can_retry(self) -> None:
        """Test failure with retry possible."""
        step = Step(step_id="a", title="A", intent="", allowed_files=[], success_criteria="")
        can_retry = StepLifecycle.fail(step, "Error")
        assert step.status == StepStatus.FAILED
        assert step.failure_count == 1
        assert can_retry is True

    def test_fail_max_reached(self) -> None:
        """Test failure at max retries."""
        step = Step(step_id="a", title="A", intent="", allowed_files=[], success_criteria="")
        step.failure_count = 1
        can_retry = StepLifecycle.fail(step, "Error again")
        assert step.failure_count == 2
        assert can_retry is False

    def test_skip(self) -> None:
        """Test step skipping."""
        step = Step(step_id="a", title="A", intent="", allowed_files=[], success_criteria="")
        StepLifecycle.skip(step, "Non-critical")
        assert step.status == StepStatus.SKIPPED

    def test_can_skip_low_risk(self) -> None:
        """Test skippable check for low risk."""
        step = Step(
            step_id="a",
            title="A",
            intent="",
            allowed_files=[],
            success_criteria="",
            risk_level=RiskLevel.LOW,
        )
        assert StepLifecycle.can_skip(step) is True

    def test_cannot_skip_high_risk(self) -> None:
        """Test skippable check for high risk."""
        step = Step(
            step_id="a",
            title="A",
            intent="",
            allowed_files=[],
            success_criteria="",
            risk_level=RiskLevel.HIGH,
        )
        assert StepLifecycle.can_skip(step) is False


# =============================================================================
# Planner Tests
# =============================================================================


class TestPlannerV2:
    """Tests for PlannerV2 class."""

    def test_propose_repair_plan(self) -> None:
        """Test generating a repair plan."""
        planner = PlannerV2(seed=42)
        plan = planner.propose_plan(
            "Fix failing test test_login",
            {"test_cmd": "pytest", "failing_test_file": "tests/test_auth.py"},
        )

        assert plan.plan_id.startswith("plan-")
        assert len(plan.steps) >= 6
        assert plan.steps[0].step_id == "analyze-failure"

    def test_propose_feature_plan(self) -> None:
        """Test generating a feature plan."""
        planner = PlannerV2(seed=42)
        plan = planner.propose_plan(
            "Add user preferences API",
            {"test_cmd": "pytest"},
        )

        assert len(plan.steps) >= 6
        assert "understand" in plan.steps[0].step_id

    def test_propose_generic_plan(self) -> None:
        """Test generating a generic plan."""
        planner = PlannerV2(seed=42)
        plan = planner.propose_plan(
            "Refactor database layer",
            {"test_cmd": "pytest"},
        )

        assert len(plan.steps) >= 3

    def test_next_step(self) -> None:
        """Test getting next step."""
        planner = PlannerV2(seed=42)
        plan = planner.propose_plan("Fix test", {})
        state = PlanState(plan_id=plan.plan_id)

        step = planner.next_step(plan, state)
        assert step is not None
        assert step.status == StepStatus.ACTIVE

    def test_next_step_halted(self) -> None:
        """Test next step returns None when halted."""
        planner = PlannerV2(seed=42)
        plan = planner.propose_plan("Fix test", {})
        state = PlanState(plan_id=plan.plan_id, halted=True)

        step = planner.next_step(plan, state)
        assert step is None

    def test_update_state_success(self) -> None:
        """Test updating state on success."""
        planner = PlannerV2(seed=42)
        plan = planner.propose_plan("Fix test", {})
        state = PlanState(plan_id=plan.plan_id)

        step = planner.next_step(plan, state)
        outcome = ControllerOutcome(step_id=step.step_id, success=True)

        state = planner.update_state(plan, state, outcome)
        assert step.step_id in state.completed_steps
        assert step.status == StepStatus.DONE

    def test_update_state_failure(self) -> None:
        """Test updating state on failure."""
        planner = PlannerV2(seed=42)
        plan = planner.propose_plan("Fix test", {})
        state = PlanState(plan_id=plan.plan_id)

        step = planner.next_step(plan, state)
        outcome = ControllerOutcome(
            step_id=step.step_id,
            success=False,
            error_message="Error",
        )

        state = planner.update_state(plan, state, outcome)
        assert step.step_id in state.failed_steps
        assert step.status == StepStatus.FAILED

    def test_revise_plan_first_failure(self) -> None:
        """Test plan revision on first failure."""
        planner = PlannerV2(seed=42)
        plan = planner.propose_plan("Fix test", {})
        state = PlanState(plan_id=plan.plan_id)

        step = planner.next_step(plan, state)
        StepLifecycle.fail(step, "Error")

        failure = ControllerOutcome(
            step_id=step.step_id,
            success=False,
            # Evidence required for revision strategy selection
            failure_evidence=None,  # Will trigger ScopeReductionRevision (catch-all)
        )
        revised = planner.revise_plan(plan, state, failure)

        assert revised.version == 2
        revised_step = revised.get_step(step.step_id)
        # Without evidence, it falls back to ScopeReduction which SKIPS non-critical steps
        assert revised_step.status == StepStatus.SKIPPED

    def test_is_complete_all_done(self) -> None:
        """Test completion check when all done."""
        planner = PlannerV2(seed=42)
        plan = Plan(
            plan_id="test",
            goal="Test",
            steps=[
                Step(
                    step_id="a",
                    title="A",
                    intent="",
                    allowed_files=[],
                    success_criteria="",
                    status=StepStatus.DONE,
                )
            ],
            created_at="",
        )
        state = PlanState(plan_id="test")

        assert planner.is_complete(plan, state) is True

    def test_is_complete_halted(self) -> None:
        """Test completion check when halted."""
        planner = PlannerV2(seed=42)
        plan = planner.propose_plan("Fix test", {})
        state = PlanState(plan_id=plan.plan_id, halted=True)

        assert planner.is_complete(plan, state) is True


# =============================================================================
# Controller Adapter Tests
# =============================================================================


class TestControllerAdapter:
    """Tests for ControllerAdapter class."""

    def test_start_goal(self) -> None:
        """Test starting a goal."""
        adapter = ControllerAdapter(seed=42)
        spec = adapter.start_goal("Fix test", {"test_cmd": "pytest"})

        assert spec.step_id is not None
        assert adapter.get_plan() is not None
        assert adapter.get_state() is not None

    def test_process_outcome_success(self) -> None:
        """Test processing successful outcome."""
        adapter = ControllerAdapter(seed=42)
        spec = adapter.start_goal("Fix test", {})

        outcome = ControllerOutcome(step_id=spec.step_id, success=True)
        next_spec = adapter.process_outcome(outcome)

        # Should have a next step
        assert next_spec is not None or adapter.is_complete()

    def test_process_outcome_completes(self) -> None:
        """Test processing outcomes until complete."""
        adapter = ControllerAdapter(seed=42)
        spec = adapter.start_goal("Fix test", {})

        # Process all steps with success
        while spec is not None:
            outcome = ControllerOutcome(step_id=spec.step_id, success=True)
            spec = adapter.process_outcome(outcome)

        assert adapter.is_complete() is True

    def test_get_plan_json(self) -> None:
        """Test getting plan as JSON."""
        adapter = ControllerAdapter(seed=42)
        adapter.start_goal("Fix test", {})

        json_str = adapter.get_plan_json()
        data = json.loads(json_str)

        assert "plan_id" in data
        assert "steps" in data

    def test_get_summary(self) -> None:
        """Test getting plan summary."""
        adapter = ControllerAdapter(seed=42)
        adapter.start_goal("Fix test", {})

        summary = adapter.get_summary()
        assert summary["active"] is True
        assert "total_steps" in summary

    def test_reset(self) -> None:
        """Test adapter reset."""
        adapter = ControllerAdapter(seed=42)
        adapter.start_goal("Fix test", {})
        adapter.reset()

        assert adapter.get_plan() is None
        assert adapter.get_state() is None


# =============================================================================
# Memory Adapter Tests
# =============================================================================


class TestMemoryAdapter:
    """Tests for MemoryAdapter class."""

    def test_no_memory(self) -> None:
        """Test with no memory store."""
        adapter = MemoryAdapter()
        priors = adapter.query_decomposition_priors("repair", "python", "python")
        assert priors == []

    def test_has_memory(self) -> None:
        """Test memory availability check."""
        adapter = MemoryAdapter()
        assert adapter.has_memory() is False

    def test_similarity_score(self) -> None:
        """Test goal similarity calculation."""
        adapter = MemoryAdapter()
        score = adapter.get_similarity_score(
            "Fix failing test",
            "Fix broken test",
        )
        # Should have some overlap
        assert score > 0.0
        assert score <= 1.0

    def test_similarity_score_identical(self) -> None:
        """Test similarity of identical goals."""
        adapter = MemoryAdapter()
        score = adapter.get_similarity_score("Fix test", "Fix test")
        assert score == 1.0
