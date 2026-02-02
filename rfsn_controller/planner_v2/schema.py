"""Planner Layer v2 - Schema definitions.

This module defines the core data structures for plans and steps.
All types are serializable to JSON for auditability and replay.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StepStatus(Enum):
    """Status of a plan step."""

    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    DONE = "DONE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    BLOCKED = "BLOCKED"


class RiskLevel(Enum):
    """Risk level for a step based on blast radius and uncertainty."""

    LOW = "LOW"
    MED = "MED"
    HIGH = "HIGH"


class FailureCategory(Enum):
    """Taxonomy of failure types for intelligent revision.
    
    Categorizing failures allows the planner to revise intelligently
    instead of blindly rewriting the same step.
    """
    
    TEST_REGRESSION = "test_regression"
    COMPILATION_ERROR = "compilation_error"
    LINT_ERROR = "lint_error"
    TYPE_ERROR = "type_error"
    MISSING_DEPENDENCY = "missing_dependency"
    IMPORT_ERROR = "import_error"
    TIMEOUT = "timeout"
    SANDBOX_VIOLATION = "sandbox_violation"
    PATCH_REJECTED = "patch_rejected"
    PATCH_CONFLICT = "patch_conflict"
    VERIFICATION_FAILED = "verification_failed"
    PERMISSION_DENIED = "permission_denied"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    FLAKY_TEST = "flaky_test"
    UNKNOWN = "unknown"


@dataclass
class FailureEvidence:
    """Minimal extracted evidence for intelligent revision.
    
    This structured evidence helps the planner understand:
    - What specifically failed
    - Where the failure occurred
    - What caused it
    """
    
    category: FailureCategory
    top_failing_tests: list[str] = field(default_factory=list)
    stack_trace_head: str = ""  # First 500 chars
    error_codes: list[str] = field(default_factory=list)
    affected_files: list[str] = field(default_factory=list)
    error_line: int | None = None
    suggestion: str = ""  # Brief fix suggestion if available
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "category": self.category.value,
            "top_failing_tests": self.top_failing_tests,
            "stack_trace_head": self.stack_trace_head,
            "error_codes": self.error_codes,
            "affected_files": self.affected_files,
            "error_line": self.error_line,
            "suggestion": self.suggestion,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FailureEvidence:
        """Deserialize from dictionary."""
        return cls(
            category=FailureCategory(data.get("category", "unknown")),
            top_failing_tests=data.get("top_failing_tests", []),
            stack_trace_head=data.get("stack_trace_head", ""),
            error_codes=data.get("error_codes", []),
            affected_files=data.get("affected_files", []),
            error_line=data.get("error_line"),
            suggestion=data.get("suggestion", ""),
        )
    
    @classmethod
    def from_error_output(cls, stdout: str, stderr: str, exit_code: int) -> FailureEvidence:
        """Create evidence from error output.
        
        Parses stdout/stderr to extract structured evidence.
        """
        combined = f"{stdout}\n{stderr}"
        category = cls._detect_category(combined, exit_code)
        
        return cls(
            category=category,
            top_failing_tests=cls._extract_failing_tests(combined),
            stack_trace_head=cls._extract_stack_trace(combined),
            error_codes=cls._extract_error_codes(combined),
            affected_files=cls._extract_affected_files(combined),
            error_line=cls._extract_error_line(combined),
        )
    
    @staticmethod
    def _detect_category(output: str, exit_code: int) -> FailureCategory:
        """Detect failure category from output."""
        output_lower = output.lower()
        
        if "syntaxerror" in output_lower or "indentationerror" in output_lower:
            return FailureCategory.COMPILATION_ERROR
        if "typeerror" in output_lower or "mypy" in output_lower:
            return FailureCategory.TYPE_ERROR
        if "importerror" in output_lower or "modulenotfounderror" in output_lower:
            return FailureCategory.IMPORT_ERROR
        if "flake8" in output_lower or "ruff" in output_lower or "pylint" in output_lower:
            return FailureCategory.LINT_ERROR
        if "timeout" in output_lower or exit_code == 124:
            return FailureCategory.TIMEOUT
        if "permission denied" in output_lower:
            return FailureCategory.PERMISSION_DENIED
        if "failed" in output_lower and "test" in output_lower:
            return FailureCategory.TEST_REGRESSION
        if "conflict" in output_lower or "hunk failed" in output_lower:
            return FailureCategory.PATCH_CONFLICT
        
        return FailureCategory.UNKNOWN
    
    @staticmethod
    def _extract_failing_tests(output: str) -> list[str]:
        """Extract failing test names."""
        import re
        # Match pytest-style test names
        pattern = r"(test_\w+|Test\w+::\w+)"
        matches = re.findall(pattern, output)
        return list(set(matches))[:5]  # Top 5 unique
    
    @staticmethod
    def _extract_stack_trace(output: str) -> str:
        """Extract first part of stack trace."""
        if "Traceback" in output:
            idx = output.find("Traceback")
            return output[idx:idx + 500]
        return output[:500]
    
    @staticmethod
    def _extract_error_codes(output: str) -> list[str]:
        """Extract error codes like E501, F401, etc."""
        import re
        pattern = r"\b([A-Z]\d{3,4})\b"
        matches = re.findall(pattern, output)
        return list(set(matches))[:10]
    
    @staticmethod
    def _extract_affected_files(output: str) -> list[str]:
        """Extract file paths from output."""
        import re
        pattern = r"[\w/.-]+\.py"
        matches = re.findall(pattern, output)
        return list(set(matches))[:10]
    
    @staticmethod
    def _extract_error_line(output: str) -> int | None:
        """Extract line number from error."""
        import re
        match = re.search(r"line\s+(\d+)", output, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None


@dataclass
class ControllerTaskSpec:
    """What planner sends to controller for one step execution.

    This is the interface payload that the controller consumes.
    Controller executes exactly one step per cycle.
    """

    step_id: str
    intent: str
    allowed_files: list[str]
    success_criteria: str
    verify_cmd: str | None = None
    context_hints: dict[str, Any] = field(default_factory=dict)
    mode: str = "patch"  # read_only, patch, test
    max_lines: int = 50

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "step_id": self.step_id,
            "intent": self.intent,
            "allowed_files": self.allowed_files,
            "success_criteria": self.success_criteria,
            "verify_cmd": self.verify_cmd,
            "context_hints": self.context_hints,
            "mode": self.mode,
            "max_lines": self.max_lines,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ControllerTaskSpec:
        """Deserialize from dictionary."""
        return cls(
            step_id=data["step_id"],
            intent=data["intent"],
            allowed_files=data.get("allowed_files", []),
            success_criteria=data.get("success_criteria", ""),
            verify_cmd=data.get("verify_cmd"),
            context_hints=data.get("context_hints", {}),
            mode=data.get("mode", "patch"),
            max_lines=data.get("max_lines", 50),
        )


@dataclass
class ControllerOutcome:
    """What controller returns to planner after step execution.

    This is the structured outcome payload the planner uses to
    update plan state and decide next actions.
    """

    step_id: str
    success: bool
    patch_applied: bool = False
    tests_passed: bool = False
    error_message: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    failure_evidence: FailureEvidence | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "step_id": self.step_id,
            "success": self.success,
            "patch_applied": self.patch_applied,
            "tests_passed": self.tests_passed,
            "error_message": self.error_message,
            "metrics": self.metrics,
            "failure_evidence": self.failure_evidence.to_dict() if self.failure_evidence else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ControllerOutcome:
        """Deserialize from dictionary."""
        evidence = None
        if data.get("failure_evidence"):
            evidence = FailureEvidence.from_dict(data["failure_evidence"])
        return cls(
            step_id=data["step_id"],
            success=data["success"],
            patch_applied=data.get("patch_applied", False),
            tests_passed=data.get("tests_passed", False),
            error_message=data.get("error_message"),
            metrics=data.get("metrics", {}),
            failure_evidence=evidence,
        )


@dataclass
class Step:
    """A single atomic step in the execution plan.

    Each step must be independently executable by the controller.
    Steps declare intent, affected files, success criteria, and dependencies.
    """

    step_id: str
    title: str
    intent: str
    allowed_files: list[str]
    success_criteria: str
    dependencies: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    verify: str = ""
    risk_level: RiskLevel = RiskLevel.LOW
    rollback_hint: str = ""
    hypothesis: str = ""
    controller_task_spec: dict[str, Any] | None = None
    status: StepStatus = StepStatus.PENDING
    result: dict[str, Any] | None = None
    failure_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "step_id": self.step_id,
            "title": self.title,
            "intent": self.intent,
            "allowed_files": self.allowed_files,
            "success_criteria": self.success_criteria,
            "dependencies": self.dependencies,
            "inputs": self.inputs,
            "verify": self.verify,
            "risk_level": self.risk_level.value,
            "rollback_hint": self.rollback_hint,
            "hypothesis": self.hypothesis,
            "controller_task_spec": self.controller_task_spec,
            "status": self.status.value,
            "result": self.result,
            "failure_count": self.failure_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Step:
        """Deserialize from dictionary."""
        return cls(
            step_id=data["step_id"],
            title=data["title"],
            intent=data["intent"],
            allowed_files=data.get("allowed_files", []),
            success_criteria=data.get("success_criteria", ""),
            dependencies=data.get("dependencies", []),
            inputs=data.get("inputs", []),
            verify=data.get("verify", ""),
            risk_level=RiskLevel(data.get("risk_level", "LOW")),
            rollback_hint=data.get("rollback_hint", ""),
            hypothesis=data.get("hypothesis", ""),
            controller_task_spec=data.get("controller_task_spec"),
            status=StepStatus(data.get("status", "PENDING")),
            result=data.get("result"),
            failure_count=data.get("failure_count", 0),
        )

    def get_task_spec(self) -> ControllerTaskSpec:
        """Generate a ControllerTaskSpec from this step."""
        return ControllerTaskSpec(
            step_id=self.step_id,
            intent=self.intent,
            allowed_files=self.allowed_files,
            success_criteria=self.success_criteria,
            verify_cmd=self.verify if self.verify else None,
            context_hints=self.controller_task_spec or {},
        )


@dataclass
class Plan:
    """A structured, ordered plan for achieving a goal.

    Plans are explicit artifacts (not prompts) that can be serialized,
    audited, and replayed. The planner outputs plans; the controller executes.
    """

    plan_id: str
    goal: str
    steps: list[Step]
    created_at: str
    assumptions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at,
            "assumptions": self.assumptions,
            "constraints": self.constraints,
            "version": self.version,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Plan:
        """Deserialize from dictionary."""
        return cls(
            plan_id=data["plan_id"],
            goal=data["goal"],
            steps=[Step.from_dict(s) for s in data.get("steps", [])],
            created_at=data["created_at"],
            assumptions=data.get("assumptions", []),
            constraints=data.get("constraints", []),
            version=data.get("version", 1),
        )

    @classmethod
    def from_json(cls, json_str: str) -> Plan:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def get_step(self, step_id: str) -> Step | None:
        """Get a step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_step_index(self, step_id: str) -> int:
        """Get the index of a step by ID."""
        for i, step in enumerate(self.steps):
            if step.step_id == step_id:
                return i
        return -1


@dataclass
class PlanState:
    """Runtime state tracking for an active plan.

    Tracks progress through the plan, completed/failed steps,
    revision count, and halt conditions.
    """

    plan_id: str
    current_step_idx: int = 0
    completed_steps: list[str] = field(default_factory=list)
    failed_steps: list[str] = field(default_factory=list)
    revision_count: int = 0
    consecutive_failures: int = 0
    halted: bool = False
    halt_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "plan_id": self.plan_id,
            "current_step_idx": self.current_step_idx,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "revision_count": self.revision_count,
            "consecutive_failures": self.consecutive_failures,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanState:
        """Deserialize from dictionary."""
        return cls(
            plan_id=data["plan_id"],
            current_step_idx=data.get("current_step_idx", 0),
            completed_steps=data.get("completed_steps", []),
            failed_steps=data.get("failed_steps", []),
            revision_count=data.get("revision_count", 0),
            consecutive_failures=data.get("consecutive_failures", 0),
            halted=data.get("halted", False),
            halt_reason=data.get("halt_reason", ""),
        )
