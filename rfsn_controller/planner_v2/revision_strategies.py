"""Revision Strategies - Evidence-aware plan revision.

Uses failure evidence to intelligently revise failed steps
instead of blind retry. Each strategy handles a specific failure type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from .schema import ControllerOutcome, FailureCategory, FailureEvidence, Plan, RiskLevel, Step, StepStatus
from .tool_registry import get_tool_registry


class RevisionStrategy(Protocol):
    """Protocol for revision strategies."""
    
    def can_handle(self, category: FailureCategory) -> bool:
        """Check if this strategy handles the failure category."""
        ...
    
    def revise(
        self,
        plan: Plan,
        step: Step,
        evidence: FailureEvidence,
        failure_count: int,
    ) -> Plan:
        """Revise the plan based on failure evidence."""
        ...


@dataclass
class RevisionResult:
    """Result of a revision attempt."""
    
    revised_plan: Plan
    strategy_used: str
    changes_made: list[str]
    should_retry: bool = True


class BaseRevisionStrategy(ABC):
    """Base class for revision strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging."""
        ...
    
    @property
    @abstractmethod
    def handled_categories(self) -> list[FailureCategory]:
        """Categories this strategy can handle."""
        ...
    
    def can_handle(self, category: FailureCategory) -> bool:
        """Check if this strategy handles the failure category."""
        return category in self.handled_categories
    
    @abstractmethod
    def revise(
        self,
        plan: Plan,
        step: Step,
        evidence: FailureEvidence,
        failure_count: int,
    ) -> Plan:
        """Revise the plan based on failure evidence."""
        ...
    
    def _clone_plan(self, plan: Plan) -> Plan:
        """Create a copy of the plan for modification."""
        return Plan.from_dict(plan.to_dict())
    
    def _update_step(self, plan: Plan, step_id: str, **updates) -> None:
        """Update a step in place."""
        for step in plan.steps:
            if step.step_id == step_id:
                for key, value in updates.items():
                    if hasattr(step, key):
                        setattr(step, key, value)
                break


class TestRegressionRevision(BaseRevisionStrategy):
    """Handles test regression failures."""
    
    @property
    def name(self) -> str:
        return "test_regression"
    
    @property
    def handled_categories(self) -> list[FailureCategory]:
        return [FailureCategory.TEST_REGRESSION, FailureCategory.FLAKY_TEST]
    
    def revise(
        self,
        plan: Plan,
        step: Step,
        evidence: FailureEvidence,
        failure_count: int,
    ) -> Plan:
        """Revise for test failures.
        
        Strategies:
        1. First failure: Add context about failing tests
        2. Second failure: Reduce scope to focus on specific test
        3. Third+: Add isolation step before retry
        """
        revised = self._clone_plan(plan)
        
        if failure_count == 1:
            # Add failing test info to step context
            self._update_step(
                revised,
                step.step_id,
                controller_task_spec={
                    **(step.controller_task_spec or {}),
                    "failing_tests": evidence.top_failing_tests,
                    "hint": "Focus on fixing these specific tests",
                },
                status=StepStatus.PENDING,  # Reset for retry
            )
            print(f"DEBUG: Updated step {step.step_id} to PENDING")
        elif failure_count == 2:
            # Reduce scope - focus on first failing test only
            if evidence.top_failing_tests:
                self._update_step(
                    revised,
                    step.step_id,
                    intent=f"Fix ONLY {evidence.top_failing_tests[0]}",
                    success_criteria=f"Test {evidence.top_failing_tests[0]} passes",
                    verify=get_tool_registry().get_tool("pytest").format_command(target=evidence.top_failing_tests[0]),
                    status=StepStatus.PENDING,  # Reset for retry
                )
        else:
            # Add isolation step
            isolation_step = Step(
                step_id=f"{step.step_id}-isolate",
                title="Isolate test failure",
                intent="Run failing test in isolation to understand dependencies",
                allowed_files=step.allowed_files,
                success_criteria="Test behavior understood",
                verify=get_tool_registry().get_tool("pytest-focused").format_command(
                    target=evidence.top_failing_tests[0] if evidence.top_failing_tests else ""
                ),
                risk_level=RiskLevel.LOW,
            )
            # Insert before the failing step
            step_idx = plan.get_step_index(step.step_id)
            revised.steps.insert(step_idx, isolation_step)
        
        revised.version += 1
        return revised


class CompileErrorRevision(BaseRevisionStrategy):
    """Handles compilation/syntax errors."""
    
    @property
    def name(self) -> str:
        return "compile_error"
    
    @property
    def handled_categories(self) -> list[FailureCategory]:
        return [
            FailureCategory.COMPILATION_ERROR,
            FailureCategory.TYPE_ERROR,
            FailureCategory.LINT_ERROR,
        ]
    
    def revise(
        self,
        plan: Plan,
        step: Step,
        evidence: FailureEvidence,
        failure_count: int,
    ) -> Plan:
        """Revise for compile errors.
        
        Strategies:
        1. First failure: Add error line and codes to context
        2. Second failure: Add syntax check step before
        3. Third+: Reduce allowed files to affected ones only
        """
        revised = self._clone_plan(plan)
        
        if failure_count == 1:
            # Add error details to context
            context_update = {
                **(step.controller_task_spec or {}),
                "error_line": evidence.error_line,
                "error_codes": evidence.error_codes,
                "stack_trace": evidence.stack_trace_head[:200],
                "hint": "Fix the syntax/type error at the indicated line",
            }
            self._update_step(revised, step.step_id, controller_task_spec=context_update, status=StepStatus.PENDING)
            
        elif failure_count == 2:
            # Add syntax check step
            check_step = Step(
                step_id=f"{step.step_id}-syntax-check",
                title="Verify syntax before changes",
                intent="Check that files compile before making changes",
                allowed_files=evidence.affected_files or step.allowed_files,
                success_criteria="All files have valid syntax",
                verify=get_tool_registry().get_tool("py-compile").format_command(
                    target=" ".join(evidence.affected_files[:3])
                ),
                risk_level=RiskLevel.LOW,
                status=StepStatus.PENDING,
            )
            step_idx = plan.get_step_index(step.step_id)
            revised.steps.insert(step_idx, check_step)
            
            if evidence.affected_files:
                affected_file = evidence.affected_files[0] if evidence.affected_files else 'unknown'
                self._update_step(
                    revised,
                    step.step_id,
                    allowed_files=evidence.affected_files[:3],
                    intent=f"Fix error at line {evidence.error_line} in {affected_file}",
                    status=StepStatus.PENDING,
                )
        
        revised.version += 1
        return revised


class ImportErrorRevision(BaseRevisionStrategy):
    """Handles import and dependency errors."""
    
    @property
    def name(self) -> str:
        return "import_error"
    
    @property
    def handled_categories(self) -> list[FailureCategory]:
        return [FailureCategory.IMPORT_ERROR, FailureCategory.MISSING_DEPENDENCY]
    
    def revise(
        self,
        plan: Plan,
        step: Step,
        evidence: FailureEvidence,
        failure_count: int,
    ) -> Plan:
        """Revise for import errors.
        
        Add dependency resolution step before the failing step.
        """
        revised = self._clone_plan(plan)
        
        # Add import check step
        import_step = Step(
            step_id=f"{step.step_id}-check-imports",
            title="Resolve import issues",
            intent="Check and fix import statements",
            allowed_files=evidence.affected_files or step.allowed_files,
            success_criteria="All imports resolve correctly",
            verify="python -c 'import sys; sys.exit(0)'",
            controller_task_spec={
                "hint": "Fix import statements or add missing __init__.py",
            },
            status=StepStatus.PENDING,
        )
        
        step_idx = plan.get_step_index(step.step_id)
        revised.steps.insert(step_idx, import_step)
        revised.version += 1
        
        return revised


class ScopeReductionRevision(BaseRevisionStrategy):
    """Generic scope reduction for any failure type."""
    
    @property
    def name(self) -> str:
        return "scope_reduction"
    
    @property
    def handled_categories(self) -> list[FailureCategory]:
        return [FailureCategory.UNKNOWN, FailureCategory.TIMEOUT]
    
    def revise(
        self,
        plan: Plan,
        step: Step,
        evidence: FailureEvidence,
        failure_count: int,
    ) -> Plan:
        """Reduce scope by splitting step or reducing file set."""
        revised = self._clone_plan(plan)
        
        if len(step.allowed_files) > 1:
            # Reduce to single file focus
            self._update_step(
                revised,
                step.step_id,
                allowed_files=step.allowed_files[:1],
                intent=f"Focus on: {step.allowed_files[0]}",
                status=StepStatus.PENDING,  # Reset for retry
            )
        else:
            # Skip non-critical step
            for s in revised.steps:
                if s.step_id == step.step_id and s.risk_level == RiskLevel.LOW:
                    s.status = StepStatus.SKIPPED
        
        revised.version += 1
        return revised


class RevisionStrategyRegistry:
    """Registry of all revision strategies."""
    
    def __init__(self):
        self._strategies: list[BaseRevisionStrategy] = [
            TestRegressionRevision(),
            CompileErrorRevision(),
            ImportErrorRevision(),
            ScopeReductionRevision(),  # Catch-all
        ]
    
    def get_strategy(self, category: FailureCategory) -> BaseRevisionStrategy | None:
        """Get the best strategy for a failure category."""
        for strategy in self._strategies:
            if strategy.can_handle(category):
                return strategy
        return self._strategies[-1]  # Return catch-all
    
    def revise(
        self,
        plan: Plan,
        step: Step,
        outcome: ControllerOutcome,
    ) -> Plan:
        """Revise plan using the appropriate strategy.
        
        Args:
            plan: Current plan.
            step: Failed step.
            outcome: Controller outcome with failure evidence.
            
        Returns:
            Revised plan.
        """
        evidence = outcome.failure_evidence
        if not evidence:
            # No evidence, use generic scope reduction
            evidence = FailureEvidence(category=FailureCategory.UNKNOWN)
        
        strategy = self.get_strategy(evidence.category)
        assert strategy is not None  # get_strategy always returns catch-all
        
        return strategy.revise(
            plan,
            step,
            evidence,
            step.failure_count,
        )
    
    def register(self, strategy: BaseRevisionStrategy) -> None:
        """Register a new strategy (at front for priority)."""
        self._strategies.insert(0, strategy)


# Singleton registry
_registry = RevisionStrategyRegistry()


def get_revision_registry() -> RevisionStrategyRegistry:
    """Get the global revision strategy registry."""
    return _registry
