"""Coding-specific action types for the serial decision agent.

This module defines the atomic actions that can be committed to the
CGW during a coding workflow. Each action represents exactly one
decision that the agent can make per cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class CodingAction(Enum):
    """Atomic actions for the coding workflow.

    These actions follow the serial decision model:
    - Exactly one action is committed per cycle
    - Each action has a corresponding execution phase
    - Results are reported before the next decision
    """

    # Test execution
    RUN_TESTS = "RUN_TESTS"
    RUN_FOCUSED_TESTS = "RUN_FOCUSED_TESTS"

    # Analysis
    ANALYZE_FAILURE = "ANALYZE_FAILURE"
    ANALYZE_TRACEBACK = "ANALYZE_TRACEBACK"
    INSPECT_FILES = "INSPECT_FILES"

    # Patch generation and application
    GENERATE_PATCH = "GENERATE_PATCH"
    APPLY_PATCH = "APPLY_PATCH"
    REVERT_PATCH = "REVERT_PATCH"

    # Validation
    VALIDATE = "VALIDATE"
    LINT = "LINT"
    BUILD = "BUILD"

    # Terminal states
    FINALIZE = "FINALIZE"
    ABORT = "ABORT"

    # Idle (no action committed)
    IDLE = "IDLE"


class ActionCategory(Enum):
    """Categories for grouping actions by their nature."""

    EXECUTION = auto()  # Actions that run external tools
    ANALYSIS = auto()  # Actions that analyze data
    MODIFICATION = auto()  # Actions that change code
    VALIDATION = auto()  # Actions that verify changes
    TERMINAL = auto()  # Actions that end the workflow


ACTION_CATEGORIES: Dict[CodingAction, ActionCategory] = {
    CodingAction.RUN_TESTS: ActionCategory.EXECUTION,
    CodingAction.RUN_FOCUSED_TESTS: ActionCategory.EXECUTION,
    CodingAction.ANALYZE_FAILURE: ActionCategory.ANALYSIS,
    CodingAction.ANALYZE_TRACEBACK: ActionCategory.ANALYSIS,
    CodingAction.INSPECT_FILES: ActionCategory.ANALYSIS,
    CodingAction.GENERATE_PATCH: ActionCategory.ANALYSIS,
    CodingAction.APPLY_PATCH: ActionCategory.MODIFICATION,
    CodingAction.REVERT_PATCH: ActionCategory.MODIFICATION,
    CodingAction.VALIDATE: ActionCategory.VALIDATION,
    CodingAction.LINT: ActionCategory.VALIDATION,
    CodingAction.BUILD: ActionCategory.VALIDATION,
    CodingAction.FINALIZE: ActionCategory.TERMINAL,
    CodingAction.ABORT: ActionCategory.TERMINAL,
    CodingAction.IDLE: ActionCategory.TERMINAL,
}


@dataclass
class ActionPayload:
    """Payload for a coding action.

    This is the content that gets committed to the CGW when an action
    is selected. It contains all information needed to execute the action.
    """

    action: CodingAction
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        """Serialize payload for CGW storage."""
        import json

        return json.dumps(
            {
                "action": self.action.value,
                "parameters": self.parameters,
                "context": self.context,
                "metadata": self.metadata,
            }
        ).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "ActionPayload":
        """Deserialize payload from CGW storage."""
        import json

        obj = json.loads(data.decode("utf-8"))
        return cls(
            action=CodingAction(obj["action"]),
            parameters=obj.get("parameters", {}),
            context=obj.get("context", {}),
            metadata=obj.get("metadata", {}),
        )


@dataclass
class ExecutionResult:
    """Result of executing a coding action.

    This is returned by the executor layer after blocking execution.
    It contains all information needed for the next decision cycle.
    """

    action: CodingAction
    success: bool
    output: str = ""
    error: Optional[str] = None
    duration_ms: float = 0.0
    artifacts: Dict[str, Any] = field(default_factory=dict)

    # Test-specific fields
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    failing_tests: List[str] = field(default_factory=list)

    # Patch-specific fields
    patch_applied: bool = False
    files_modified: List[str] = field(default_factory=list)

    def is_terminal(self) -> bool:
        """Check if this result represents a terminal state."""
        return self.action in (CodingAction.FINALIZE, CodingAction.ABORT)


@dataclass
class CycleResult:
    """Result of one complete decision cycle.

    Captures the decision made, the execution result, and metadata
    for replay and auditing.
    """

    cycle_id: int
    action: CodingAction
    payload: ActionPayload
    execution_result: Optional[ExecutionResult]

    # CGW metadata
    slot_id: str = ""
    was_forced: bool = False
    losers: List[str] = field(default_factory=list)

    # Timing
    decision_time_ms: float = 0.0
    execution_time_ms: float = 0.0

    def total_time_ms(self) -> float:
        return self.decision_time_ms + self.execution_time_ms
