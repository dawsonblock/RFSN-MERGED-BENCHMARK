"""
from __future__ import annotations
State tracking for RFSN Planner v5.

Maintains explicit state across proposal attempts for intelligent recovery.
"""

from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID


class HypothesisOutcome(Enum):
    """Outcome of testing a hypothesis."""

    CONFIRMED = "confirmed"  # Proposal accepted, effect observed
    REJECTED_BY_GATE = "rejected_by_gate"  # Gate blocked proposal
    FAILED_EFFECT = "failed_effect"  # Proposal applied but effect didn't match
    CAUSED_REGRESSION = "caused_regression"  # Introduced new failures
    PENDING = "pending"  # Not yet attempted


class GateRejectionType(Enum):
    """Classification of gate rejections."""

    SCHEMA_VIOLATION = "schema_violation"  # Missing or invalid fields
    ORDERING_VIOLATION = "ordering_violation"  # Wrong sequence (e.g., refactor while tests fail)
    BOUNDS_VIOLATION = "bounds_violation"  # Outside allowed scope
    INVARIANT_VIOLATION = "invariant_violation"  # Breaks safety rules
    UNKNOWN = "unknown"


@dataclass
class HypothesisTrial:
    """Record of attempting a hypothesis."""

    proposal_id: UUID
    hypothesis: str
    outcome: HypothesisOutcome
    rejection_reason: str | None = None
    rejection_type: GateRejectionType | None = None
    iteration: int = 0


@dataclass
class StateTracker:
    """
    Tracks planning state across multiple proposal attempts.

    Used by meta-planner to make intelligent decisions about next steps.
    """

    # Reproduction
    repro_command: str | None = None  # Best known command to reproduce failure
    reproduction_confirmed: bool = False

    # Failures
    failing_tests: set[str] = field(default_factory=set)  # Test IDs/nodeids
    exception_types: set[str] = field(default_factory=set)  # Exception class names
    traceback_frames: list[tuple[str, int]] = field(
        default_factory=list
    )  # (file, lineno)

    # Localization
    suspect_files: list[tuple[str, float]] = field(
        default_factory=list
    )  # (path, confidence)
    suspect_symbols: list[tuple[str, str, float]] = field(
        default_factory=list
    )  # (file, symbol, confidence)

    # History
    hypotheses_tried: list[HypothesisTrial] = field(default_factory=list)
    files_modified: set[str] = field(default_factory=set)
    gate_rejections: list[tuple[GateRejectionType, str]] = field(
        default_factory=list
    )  # (type, reason)

    # Budgets
    risk_budget: int = 3  # Max "medium" risk proposals
    risk_spent: int = 0
    iteration_budget: int = 50  # Max iterations before declaring stuck
    current_iteration: int = 0

    # Convergence detection
    consecutive_failures: int = 0  # Same failure pattern
    last_failure_signature: str | None = None
    stuck_threshold: int = 2  # Pivot to localization after N same failures

    def record_hypothesis(
        self,
        proposal_id: UUID,
        hypothesis: str,
        outcome: HypothesisOutcome,
        rejection_reason: str | None = None,
        rejection_type: GateRejectionType | None = None,
    ):
        """Record the outcome of a hypothesis trial."""
        trial = HypothesisTrial(
            proposal_id=proposal_id,
            hypothesis=hypothesis,
            outcome=outcome,
            rejection_reason=rejection_reason,
            rejection_type=rejection_type,
            iteration=self.current_iteration,
        )
        self.hypotheses_tried.append(trial)

        # Update rejection tracking
        if outcome == HypothesisOutcome.REJECTED_BY_GATE and rejection_type:
            self.gate_rejections.append((rejection_type, rejection_reason or ""))

    def record_failure_signature(self, signature: str):
        """Track failure patterns to detect when stuck."""
        if signature == self.last_failure_signature:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 1
            self.last_failure_signature = signature

    def is_stuck(self) -> bool:
        """Determine if planner is stuck in a loop."""
        # Too many iterations
        if self.current_iteration >= self.iteration_budget:
            return True

        # Same failure repeating
        if self.consecutive_failures >= self.stuck_threshold:
            return True

        # Too many gate rejections in a row
        recent_rejections = [
            trial
            for trial in self.hypotheses_tried[-5:]
            if trial.outcome == HypothesisOutcome.REJECTED_BY_GATE
        ]
        if len(recent_rejections) >= 3:
            return True

        return False

    def should_pivot_to_localization(self) -> bool:
        """Determine if should stop patching and gather more evidence."""
        # Stuck on same failure
        if self.consecutive_failures >= self.stuck_threshold:
            return True

        # Multiple patch attempts failed with same symptom
        recent_patch_failures = [
            trial
            for trial in self.hypotheses_tried[-3:]
            if trial.outcome == HypothesisOutcome.FAILED_EFFECT
        ]
        if len(recent_patch_failures) >= 2:
            return True

        return False

    def can_afford_risk(self, risk_level: str) -> bool:
        """Check if we have budget for a medium-risk proposal."""
        if risk_level == "low":
            return True
        if risk_level == "medium":
            return self.risk_spent < self.risk_budget
        return False  # HIGH risk never allowed

    def spend_risk(self, risk_level: str):
        """Consume risk budget."""
        if risk_level == "medium":
            self.risk_spent += 1

    def add_suspect_file(self, path: str, confidence: float):
        """Add a file to the suspect list with confidence score."""
        # Update or add
        self.suspect_files = [
            (p, c) for p, c in self.suspect_files if p != path
        ]  # Remove if exists
        self.suspect_files.append((path, confidence))
        # Sort by confidence descending
        self.suspect_files.sort(key=lambda x: x[1], reverse=True)

    def add_suspect_symbol(self, file: str, symbol: str, confidence: float):
        """Add a symbol (function/class) to suspect list."""
        self.suspect_symbols = [
            (f, s, c) for f, s, c in self.suspect_symbols if not (f == file and s == symbol)
        ]
        self.suspect_symbols.append((file, symbol, confidence))
        self.suspect_symbols.sort(key=lambda x: x[2], reverse=True)

    def get_top_suspect_file(self) -> str | None:
        """Get the most likely suspect file."""
        if not self.suspect_files:
            return None
        return self.suspect_files[0][0]

    def get_top_suspect_symbol(self) -> tuple[str, str] | None:
        """Get the most likely suspect symbol (file, symbol)."""
        if not self.suspect_symbols:
            return None
        return (self.suspect_symbols[0][0], self.suspect_symbols[0][1])

    def increment_iteration(self):
        """Move to next iteration."""
        self.current_iteration += 1

    def recent_gate_rejection_types(self, n: int = 5) -> list[GateRejectionType]:
        """Get recent rejection types for pattern detection."""
        return [r[0] for r in self.gate_rejections[-n:]]

    def has_reproduction(self) -> bool:
        """Check if we have a working reproduction."""
        return self.repro_command is not None and self.reproduction_confirmed

    def to_dict(self) -> dict:
        """Serialize state for persistence."""
        return {
            "repro_command": self.repro_command,
            "reproduction_confirmed": self.reproduction_confirmed,
            "failing_tests": list(self.failing_tests),
            "exception_types": list(self.exception_types),
            "traceback_frames": self.traceback_frames,
            "suspect_files": self.suspect_files,
            "suspect_symbols": self.suspect_symbols,
            "hypotheses_tried": [
                {
                    "proposal_id": str(t.proposal_id),
                    "hypothesis": t.hypothesis,
                    "outcome": t.outcome.value,
                    "rejection_reason": t.rejection_reason,
                    "rejection_type": t.rejection_type.value
                    if t.rejection_type
                    else None,
                    "iteration": t.iteration,
                }
                for t in self.hypotheses_tried
            ],
            "files_modified": list(self.files_modified),
            "gate_rejections": [
                (rt.value, reason) for rt, reason in self.gate_rejections
            ],
            "risk_spent": self.risk_spent,
            "current_iteration": self.current_iteration,
            "consecutive_failures": self.consecutive_failures,
            "last_failure_signature": self.last_failure_signature,
        }
