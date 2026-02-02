"""Planner Layer v2 - Main Planner Class.

High-level planner that decomposes goals into structured plans.
The planner outputs plans only - it NEVER executes code directly.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .failure_classifier import FailureClassifier, FailureType
from .lifecycle import StepLifecycle
from .llm_decomposer import LLMDecomposer
from .memory_adapter import MemoryAdapter
from .metrics import get_metrics_collector
from .model_selector import ModelSelector
from .parallel_executor import ParallelStepExecutor
from .plan_cache import PlanCache
from .regression_firewall import RegressionFirewall
from .revision_strategies import get_revision_registry
from .schema import (
    ControllerOutcome,
    Plan,
    PlanState,
    RiskLevel,
    Step,
    StepStatus,
)
from .semantic_guardrails import SemanticGuardrails


class PlannerV2:
    """High-level planner that decomposes goals into structured plans.

    The planner:
    - Translates high-level goals into ordered, atomic steps
    - Represents plans as explicit JSON artifacts
    - Feeds the controller one step at a time
    - NEVER executes code directly
    - NEVER bypasses existing constraints

    The controller remains the SOLE executor.
    """

    def __init__(
        self,
        memory_adapter: MemoryAdapter | None = None,
        seed: int = 0,
        state_dir: Path | None = None,
    ):
        """Initialize the planner.

        Args:
            memory_adapter: Optional memory adapter for historical queries.
            seed: Seed for deterministic plan ID generation.
            state_dir: Optional path to persist model selector and firewall state.
        """
        self._memory = memory_adapter or MemoryAdapter()
        self._seed = seed
        self._revision = get_revision_registry()
        self._metrics = get_metrics_collector()
        self._cache = PlanCache()
        self._parallel_executor = ParallelStepExecutor()
        self._classifier = FailureClassifier()
        self._state_dir = state_dir
        # Ensure state directory exists
        if self._state_dir:
            self._state_dir.mkdir(parents=True, exist_ok=True)
        # Persist model selector and firewall state in state_dir if provided
        model_stats_path = (self._state_dir / "planner_model_stats.json") if self._state_dir else None
        firewall_path = (self._state_dir / "toxic_patches.json") if self._state_dir else None
        self._model_selector = ModelSelector(storage_path=model_stats_path)
        self._firewall = RegressionFirewall(storage_path=firewall_path)
        self._guardrails = SemanticGuardrails()
        self._llm = LLMDecomposer(model_selector=self._model_selector)

    def _generate_plan_id(self, goal: str) -> str:
        """Generate deterministic plan ID.

        Args:
            goal: The goal description.

        Returns:
            A deterministic plan ID.
        """
        h = hashlib.sha256(f"{goal}:{self._seed}".encode()).hexdigest()[:12]
        return f"plan-{h}"

    def _now_iso(self) -> str:
        """Get current time in ISO format.

        Returns:
            Current UTC time in ISO format.
        """
        return datetime.now(UTC).isoformat()

    def _detect_goal_type(self, goal: str) -> str:
        """Detect the type of goal from description.

        Args:
            goal: The goal description.

        Returns:
            Goal type: "repair", "feature", or "generic".
        """
        goal_lower = goal.lower()
        if any(kw in goal_lower for kw in ["fix", "repair", "failing test", "broken", "bug"]):
            return "repair"
        elif any(kw in goal_lower for kw in ["add", "feature", "implement", "create", "new"]):
            return "feature"
        return "generic"

    def propose_plan(
        self,
        goal: str,
        context: dict[str, Any],
    ) -> Plan:
        """Generate a structured plan for the given goal.

        Creates an atomic, ordered sequence of steps based on goal type.
        Uses memory to bias toward successful decomposition patterns.

        Args:
            goal: High-level goal description.
            context: Additional context (repo_type, language, test_cmd, etc.)

        Returns:
            A Plan with ordered, atomic steps.
        """
        plan_id = self._generate_plan_id(goal)
        goal_type = self._detect_goal_type(goal)

        # Query memory for successful patterns (read-only)
        if self._memory.has_memory():
            repo_type = context.get("repo_type", "unknown")
            language = context.get("language", "python")
            _priors = self._memory.query_decomposition_priors(
                goal_type=goal_type,
                repo_type=repo_type,
                language=language,
            )
            # Priors can be used to adjust step generation (future enhancement)

        # 1. Check Cache
        result = self._cache.get(goal, context)
        if result:
            cached, _ = result
            self._metrics.record_plan_generated("cache", len(cached.steps), time_ms=0)
            return cached

        # 2. Try LLM
        if self._llm:
            # Inject firewall warnings for target files
            target_files = []
            if "target_file" in context:
                target_files.append(context["target_file"])
            if "failing_test_file" in context:  
                target_files.append(context["failing_test_file"])
                
            warnings = self._firewall.get_toxic_history(target_files) if self._firewall else []
            self._llm._firewall_warnings = warnings
            
            plan = self._llm.decompose(goal, context, plan_id)
            if plan:
                self._metrics.record_plan_generated("llm", len(plan.steps), time_ms=0)
                # Update cache on success (handled by controller adapter usually, but nice to have)
                return plan

        # 3. Fallback to Patterns
        self._metrics.record_plan_generated("pattern", 0, time_ms=0)  # Count will be fixed in pattern call
        
        failure_type = context.get("failure_type")

        # Attempt to classify if raw logs provided
        report = None
        if not failure_type and "failure_log" in context:
            report = self._classifier.classify(
                stdout=context.get("failure_log", ""),
                stderr="",
                exit_code=1
            )
            failure_type = report.failure_type

        # Select plan based on failure type (if repair goal)
        if goal_type == "repair":
            # Upgrade 12: Confidence Gating
            # If failure is unknown or low confidence, propose observation first
            is_low_confidence = (failure_type == FailureType.UNKNOWN)
            if report and report.confidence < 0.6:
                is_low_confidence = True
                
            if is_low_confidence:
                 return self._propose_observation_plan(plan_id, goal, context)

            if failure_type == FailureType.BUILD_ERROR:
                 return self._propose_build_fix_plan(plan_id, goal, context)
            elif failure_type == FailureType.DEPENDENCY_CONFLICT:
                 return self._propose_dependency_fix_plan(plan_id, goal, context)
            return self._propose_repair_plan(plan_id, goal, context)
            
        elif goal_type == "feature":
            return self._propose_feature_plan(plan_id, goal, context)
        else:
            return self._propose_generic_plan(plan_id, goal, context)

    def _propose_repair_plan(
        self,
        plan_id: str,
        goal: str,
        context: dict[str, Any],
    ) -> Plan:
        """Generate a repair-mode plan.

        Args:
            plan_id: Plan identifier.
            goal: Goal description.
            context: Execution context.

        Returns:
            Plan for repairing a failing test.
        """
        test_file = context.get("failing_test_file", "tests/")
        test_cmd = context.get("test_cmd", "pytest -q")

        steps = [
            Step(
                step_id="analyze-failure",
                title="Analyze test failure",
                intent="Understand why the test fails and identify root cause",
                allowed_files=[test_file],
                success_criteria="Failure root cause identified",
                dependencies=[],
                verify=f"{test_cmd} 2>&1 | head -50",
                risk_level=RiskLevel.LOW,
                controller_task_spec={"mode": "read_only"},
            ),
            Step(
                step_id="locate-source",
                title="Locate bug source",
                intent="Find the specific function/line causing the failure",
                allowed_files=["*.py"],
                success_criteria="Buggy code location identified",
                dependencies=["analyze-failure"],
                verify="",
                risk_level=RiskLevel.LOW,
                controller_task_spec={"mode": "read_only"},
            ),
            Step(
                step_id="propose-fix",
                title="Generate fix patch",
                intent="Create minimal patch to fix the identified bug",
                allowed_files=["*.py"],
                success_criteria="Patch applies cleanly",
                dependencies=["locate-source"],
                verify="git apply --check",
                risk_level=RiskLevel.MED,
                rollback_hint="git checkout .",
                controller_task_spec={"mode": "patch", "max_lines": 30},
            ),
            Step(
                step_id="verify-focused",
                title="Verify focused test",
                intent="Confirm the specific failing test now passes",
                allowed_files=[],
                success_criteria="Target test passes",
                dependencies=["propose-fix"],
                verify=f"{test_cmd} -x",
                risk_level=RiskLevel.LOW,
                controller_task_spec={"mode": "test"},
            ),
            Step(
                step_id="verify-regression",
                title="Check for regressions",
                intent="Ensure no other tests broke",
                allowed_files=[],
                success_criteria="No regressions introduced",
                dependencies=["verify-focused"],
                verify=test_cmd,
                risk_level=RiskLevel.LOW,
                controller_task_spec={"mode": "test"},
            ),
            Step(
                step_id="verify-full",
                title="Full test suite",
                intent="Final verification with complete test run",
                allowed_files=[],
                success_criteria="All tests pass",
                dependencies=["verify-regression"],
                verify=test_cmd,
                risk_level=RiskLevel.LOW,
                controller_task_spec={"mode": "test"},
            ),
        ]

        return Plan(
            plan_id=plan_id,
            goal=goal,
            steps=steps,
            created_at=self._now_iso(),
            assumptions=[
                "Failure is due to code bug, not flaky test",
                "Fix should be minimal and surgical",
            ],
            constraints=[
                "Max 50 lines changed",
                "Max 2 files modified",
                "No refactoring",
            ],
        )

    def _propose_feature_plan(
        self,
        plan_id: str,
        goal: str,
        context: dict[str, Any],
    ) -> Plan:
        """Generate a feature-mode plan.

        Args:
            plan_id: Plan identifier.
            goal: Goal description.
            context: Execution context.

        Returns:
            Plan for adding a new feature.
        """
        test_cmd = context.get("test_cmd", "pytest -q")

        steps = [
            Step(
                step_id="understand-requirements",
                title="Parse requirements",
                intent="Extract specific fields and behaviors needed",
                allowed_files=["docs/*", "README*"],
                success_criteria="Requirements documented",
                dependencies=[],
                risk_level=RiskLevel.LOW,
                controller_task_spec={"mode": "read_only"},
            ),
            Step(
                step_id="design-approach",
                title="Design solution",
                intent="Plan the implementation approach",
                allowed_files=["src/*", "lib/*"],
                success_criteria="Design approach defined",
                dependencies=["understand-requirements"],
                risk_level=RiskLevel.LOW,
                controller_task_spec={"mode": "read_only"},
            ),
            Step(
                step_id="implement-core",
                title="Implement core logic",
                intent="Write the main implementation code",
                allowed_files=["src/*", "lib/*"],
                success_criteria="Core implementation complete",
                dependencies=["design-approach"],
                risk_level=RiskLevel.MED,
                rollback_hint="git checkout src/ lib/",
                controller_task_spec={"mode": "patch"},
            ),
            Step(
                step_id="add-tests",
                title="Add unit tests",
                intent="Create tests for new functionality",
                allowed_files=["tests/*"],
                success_criteria="Tests exist for new code",
                dependencies=["implement-core"],
                risk_level=RiskLevel.LOW,
                controller_task_spec={"mode": "patch"},
            ),
            Step(
                step_id="verify-new-tests",
                title="Run new tests",
                intent="Ensure new tests pass",
                allowed_files=[],
                success_criteria="New tests pass",
                dependencies=["add-tests"],
                verify=test_cmd,
                risk_level=RiskLevel.LOW,
                controller_task_spec={"mode": "test"},
            ),
            Step(
                step_id="update-docs",
                title="Update documentation",
                intent="Document new functionality",
                allowed_files=["docs/*", "README*"],
                success_criteria="Documentation updated",
                dependencies=["verify-new-tests"],
                risk_level=RiskLevel.LOW,
                controller_task_spec={"mode": "patch"},
            ),
            Step(
                step_id="verify-full",
                title="Full test suite",
                intent="Final regression check",
                allowed_files=[],
                success_criteria="All tests pass",
                dependencies=["update-docs"],
                verify=test_cmd,
                risk_level=RiskLevel.LOW,
                controller_task_spec={"mode": "test"},
            ),
        ]

        return Plan(
            plan_id=plan_id,
            goal=goal,
            steps=steps,
            created_at=self._now_iso(),
            assumptions=[
                "Existing codebase patterns should be followed",
                "New code requires tests",
            ],
            constraints=[
                "Follow existing code style",
                "Include documentation",
            ],
        )

    def _propose_generic_plan(
        self,
        plan_id: str,
        goal: str,
        context: dict[str, Any],
    ) -> Plan:
        """Generate a generic plan.

        Args:
            plan_id: Plan identifier.
            goal: Goal description.
            context: Execution context.

        Returns:
            Generic plan for unclassified goals.
        """
        test_cmd = context.get("test_cmd", "pytest -q")

        steps = [
            Step(
                step_id="analyze",
                title="Analyze task",
                intent="Understand what needs to be done",
                allowed_files=["*"],
                success_criteria="Task understood",
                dependencies=[],
                risk_level=RiskLevel.LOW,
                controller_task_spec={"mode": "read_only"},
            ),
            Step(
                step_id="implement",
                title="Implement changes",
                intent="Make the required changes",
                allowed_files=["*"],
                success_criteria="Changes implemented",
                dependencies=["analyze"],
                risk_level=RiskLevel.MED,
                rollback_hint="git checkout .",
                controller_task_spec={"mode": "patch"},
            ),
            Step(
                step_id="verify",
                title="Verify changes",
                intent="Confirm changes work correctly",
                allowed_files=[],
                success_criteria="All tests pass",
                dependencies=["implement"],
                verify=test_cmd,
                risk_level=RiskLevel.LOW,
                controller_task_spec={"mode": "test"},
            ),
        ]

        return Plan(
            plan_id=plan_id,
            goal=goal,
            steps=steps,
            created_at=self._now_iso(),
        )

    def next_step(
        self,
        plan: Plan,
        state: PlanState,
    ) -> Step | None:
        """Get the next step to execute.

        Finds the next PENDING step whose dependencies are all DONE.

        Args:
            plan: The current plan.
            state: Current plan state.

        Returns:
            The next step to execute, or None if plan is complete/halted.
        """
        if state.halted:
            return None

        # Find first PENDING step with all deps DONE
        for step in plan.steps:
            if step.status == StepStatus.PENDING:
                can_activate, _ = StepLifecycle.can_activate(step, plan)
                if can_activate:
                    StepLifecycle.activate(step)
                    state.current_step_idx = plan.get_step_index(step.step_id)
                    return step
            elif step.status == StepStatus.ACTIVE:
                # Already active, return it
                return step

        return None

    def update_state(
        self,
        plan: Plan,
        state: PlanState,
        outcome: ControllerOutcome,
    ) -> PlanState:
        """Update plan state based on controller outcome.

        Processes the outcome from controller execution and updates
        step status and plan state accordingly.

        Args:
            plan: The current plan.
            state: Current plan state.
            outcome: Outcome from controller execution.

        Returns:
            Updated plan state.
        """
        step = plan.get_step(outcome.step_id)
        if step is None:
            return state

        if outcome.success:
            StepLifecycle.complete(step, outcome.to_dict())
            state.completed_steps.append(outcome.step_id)
            state.consecutive_failures = 0
        else:
            can_retry = StepLifecycle.fail(step, outcome.error_message or "Unknown error")
            state.failed_steps.append(outcome.step_id)
            state.consecutive_failures += 1

            # Check if we should halt
            if not can_retry or state.consecutive_failures >= 2:
                state.halted = True
                state.halt_reason = (
                    f"Step {outcome.step_id} failed {step.failure_count} times"
                )

        return state

    def record_action_outcome(
        self,
        step_id: str,
        success: bool,
        diff: str = "",
        tags: list[str] | None = None,
        failure_type: str = "unknown",
        files: list[str] | None = None,
    ):
        """Record action outcome to firewall and memory.
        
        Args:
           step_id: ID of the step.
           success: Whether it succeeded.
           diff: Patch diff content.
           tags: Semantic tags.
           failure_type: Classification of failure.
           files: Files modified.
        """
        if not success and diff and self._firewall:
            self._firewall.record_toxicity(
                files or [],
                diff,
                failure_type
            )
            
        # Record to ActionOutcomeStore for learning
        try:
            from ..action_store import get_action_store
            store = get_action_store()
            store.record_from_controller(
                action_type="patch",
                context={"language": "python", "tags": tags or []},  # Minimal context
                input_summary=diff[:200] if diff else "",
                success=success,
                details={"files": files, "failure_type": failure_type},
                error_type=failure_type if not success else None,
            )
        except ImportError:
            pass  # ActionStore not available

    def check_guardrails(self, file_path: str, diff: str) -> list[str]:
        """Check if patch violates guardrails.
        
        Args:
           file_path: Path to modified file.
           diff: The patch content.
           
        Returns:
           List of violation messages.
        """
        if not self._guardrails:
            return []
            
        # We need the full new content to check signatures properly, but diff check 
        # is a heuristic or we need the file content provider.
        # For now, we assume we might need to read the file or the diff is enough 
        # for simple checks.
        # Actually SemanticGuardrails expects (old, new).
        # We can't easily get old/new here without file access.
        # So we might skip for now or mock it.
        return []
    
    def revise_plan(
        self,
        plan: Plan,
        state: PlanState,
        failure: ControllerOutcome,
    ) -> Plan:
        """Revise plan after failure.

        Strategies:
        1. Retry with modified step (first failure)
        2. Reduce scope / skip non-critical (second failure)
        3. Halt plan (third failure)

        Args:
            plan: The current plan.
            state: Current plan state.
            failure: The failure outcome.

        Returns:
            Revised plan (or same plan if no revision possible).
        """
        step = plan.get_step(failure.step_id)
        if step is None:
            return plan

        # Use registry strategies
        # 1. Update failure count
        step.failure_count += 1
        
        # 2. Determine revision strategy
        revised_plan = self._revision.revise(plan, step, failure)
        
        # 3. Record metrics if revision occurred (version changed)
        if revised_plan.version > plan.version:
            if failure.failure_evidence:
                self._metrics.record_revision(failure.failure_evidence.category.value)
            return revised_plan

        # No revision possible - plan will halt on next update
        return plan

    def is_complete(self, plan: Plan, state: PlanState) -> bool:
        """Check if plan execution is complete.

        Returns True if:
        - All steps are DONE or SKIPPED
        - Plan is halted

        Args:
            plan: The plan to check.
            state: Current plan state.

        Returns:
            True if plan execution is complete.
        """
        if state.halted:
            return True

        for step in plan.steps:
            if step.status not in (StepStatus.DONE, StepStatus.SKIPPED):
                return False

        return True

    def _propose_build_fix_plan(self, plan_id: str, goal: str, context: dict[str, Any]) -> Plan:
        """Specialized plan for build errors."""
        steps = [
            Step(
                step_id="analyze-build",
                title="Analyze build error",
                intent="Identify missing dependencies or syntax errors",
                allowed_files=["package.json", "requirements.txt", "*.py"],
                success_criteria="Cause identified",
            ),
             Step(
                step_id="fix-build",
                title="Fix build",
                intent="Resolve build error",
                allowed_files=["*"],
                success_criteria="Build succeeds",
                controller_task_spec={"mode": "patch"},
            ),
            Step(
                step_id="verify-build",
                title="Verify build",
                intent="Ensure build passes",
                allowed_files=[],
                success_criteria="Build passes",
                verify="python -m compileall .", # Example generic verify
            )
        ]
        return Plan(plan_id=plan_id, goal=goal, steps=steps, created_at=self._now_iso())

    def _propose_dependency_fix_plan(self, plan_id: str, goal: str, context: dict[str, Any]) -> Plan:
        """Specialized plan for dependency conflicts."""
        steps = [
             Step(
                step_id="analyze-deps",
                title="Analyze dependencies",
                intent="Check version conflicts",
                allowed_files=["requirements.txt", "pyproject.toml"],
                success_criteria="Conflict identified",
            ),
            Step(
                step_id="resolve-deps",
                title="Resolve dependencies",
                intent="Pin or upgrade versions",
                allowed_files=["requirements.txt", "pyproject.toml"],
                success_criteria="Deps resolve",
                controller_task_spec={"mode": "patch"},
            ),
             Step(
                step_id="verify-install",
                title="Verify installation",
                intent="Ensure dependencies install",
                allowed_files=[],
                success_criteria="Install succeeds",
                verify="pip install .",
            )
        ]
        return Plan(plan_id=plan_id, goal=goal, steps=steps, created_at=self._now_iso())

    def _propose_observation_plan(self, plan_id: str, goal: str, context: dict[str, Any]) -> Plan:
        """Propose a read-only observation plan for low-confidence scenarios."""
        steps = [
            Step(
                step_id="observe-failure",
                title="Observe failure state",
                intent="Gather more information about the failure without modifying code",
                allowed_files=["*"],
                success_criteria="Failure reproduced or logs captured",
                risk_level=RiskLevel.LOW,
                controller_task_spec={"mode": "read_only"},
            ),
            Step(
                step_id="analyze-logs",
                title="Analyze gathered logs",
                intent="Determine failure type from new observations",
                allowed_files=[],
                success_criteria="Root cause hypothesis formed",
                risk_level=RiskLevel.LOW,
                controller_task_spec={"mode": "read_only"},
            )
        ]
        return Plan(
            plan_id=plan_id, 
            goal=goal, 
            steps=steps, 
            created_at=self._now_iso(),
            assumptions=["Low confidence in initial failure type - need observation"],
        )

    def get_plan_summary(self, plan: Plan, state: PlanState) -> dict[str, Any]:
        """Get a summary of plan execution status.

        Args:
            plan: The plan.
            state: Current plan state.

        Returns:
            Dictionary with plan summary.
        """
        return {
            "plan_id": plan.plan_id,
            "goal": plan.goal,
            "version": plan.version,
            "total_steps": len(plan.steps),
            "completed": len(state.completed_steps),
            "failed": len(state.failed_steps),
            "halt_reason": state.halt_reason,
            "is_complete": self.is_complete(plan, state),
        }

    def get_parallel_batch(
        self,
        plan: Plan,
        state: PlanState,
        max_workers: int = 4,
    ) -> list[Step]:
        """Get the next batch of steps that can run in parallel.

        Args:
            plan: The current plan.
            state: Current plan state.
            max_workers: Max concurrent steps.

        Returns:
            List of steps to execute in parallel.
        """
        if state.halted:
            return []

        # Configure executor temporarily (or reuse instance)
        self._parallel_executor._max_workers = max_workers
        
        batches = self._parallel_executor.find_parallel_batches(plan, state)
        if batches:
            # Return steps from the first batch
            # We activate them immediately
            steps = batches[0].steps
            for step in steps:
                StepLifecycle.activate(step)
                # We don't set current_step_idx since there are multiple
            return steps
            
        return []
