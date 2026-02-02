"""
from __future__ import annotations
Proposal data structures for RFSN Planner v5.

Defines the mandatory schema for all proposals submitted to the RFSN gate.
"""

from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4


class ProposalIntent(Enum):
    """The high-level intent of a proposal."""

    REPAIR = "repair"  # Fix failing tests
    REFACTOR = "refactor"  # Improve code structure (only when tests pass)
    FEATURE = "feature"  # Add new functionality
    TEST = "test"  # Run tests or verify behavior
    ANALYZE = "analyze"  # Gather evidence without changes


class ActionType(Enum):
    """The type of action being proposed."""

    READ_FILE = "read_file"
    SEARCH_REPO = "search_repo"
    EDIT_FILE = "edit_file"
    ADD_FILE = "add_file"
    DELETE_FILE = "delete_file"
    RUN_TESTS = "run_tests"
    RUN_COMMAND = "run_command"


class RiskLevel(Enum):
    """Risk assessment for the proposed change."""

    LOW = "low"  # Single function, well-tested path
    MEDIUM = "medium"  # Multiple functions or API changes
    # HIGH is not allowed in v5 - gate rejects automatically


class TestExpectation(Enum):
    """Expected test outcome after applying proposal."""

    PASS = "pass"
    FAIL = "fail"
    UNCHANGED = "unchanged"


@dataclass(frozen=True)
class Target:
    """The target of a proposal action."""

    path: str  # Relative path from repo root
    symbol: str | None = None  # Function/class name if applicable

    def __post_init__(self):
        """Validate target path."""
        if not self.path:
            raise ValueError("Target path cannot be empty")
        if self.path.startswith("/") or self.path.startswith(".."):
            raise ValueError("Target path must be relative and within repo")


@dataclass(frozen=True)
class ExpectedEffect:
    """The expected observable effect of a proposal."""

    tests: TestExpectation  # Expected test outcome
    behavior: str  # Observable behavior change

    def __post_init__(self):
        """Validate expected effect."""
        if not self.behavior or not self.behavior.strip():
            raise ValueError("Expected behavior description is required")


@dataclass(frozen=True)
class Proposal:
    """
    A structured proposal for the RFSN gate to accept or reject.

    This is the mandatory schema. All fields must be populated.
    Missing fields will cause immediate gate rejection.
    """

    # Identity
    proposal_id: UUID = field(default_factory=uuid4)

    # Classification
    intent: ProposalIntent = field(default=ProposalIntent.ANALYZE)
    hypothesis: str = field(default="")  # One falsifiable sentence

    # Action
    action_type: ActionType = field(default=ActionType.READ_FILE)
    target: Target = field(default_factory=lambda: Target(path=""))

    # Description
    change_summary: str = field(default="")  # Concise description

    # Expectations
    expected_effect: ExpectedEffect = field(
        default_factory=lambda: ExpectedEffect(
            tests=TestExpectation.UNCHANGED, behavior="No observable change"
        )
    )

    # Risk & Recovery
    risk_level: RiskLevel = field(default=RiskLevel.LOW)
    rollback_plan: str = field(default="")  # How to safely revert

    def __post_init__(self):
        """Validate proposal completeness."""
        errors = []

        # Hypothesis is required and must be specific
        if not self.hypothesis or len(self.hypothesis.strip()) < 10:
            errors.append(
                "Hypothesis must be a specific falsifiable sentence (min 10 chars)"
            )

        # Change summary required for mutations
        if self.action_type in {
            ActionType.EDIT_FILE,
            ActionType.ADD_FILE,
            ActionType.DELETE_FILE,
        }:
            if not self.change_summary or len(self.change_summary.strip()) < 5:
                errors.append(
                    f"Change summary required for {self.action_type.value} (min 5 chars)"
                )

        # Rollback plan required for mutations
        if self.action_type in {
            ActionType.EDIT_FILE,
            ActionType.ADD_FILE,
            ActionType.DELETE_FILE,
        }:
            if not self.rollback_plan or len(self.rollback_plan.strip()) < 10:
                errors.append(
                    f"Rollback plan required for {self.action_type.value} (min 10 chars)"
                )

        # Target path required
        if not self.target.path:
            errors.append("Target path is required")

        if errors:
            raise ValueError(f"Invalid proposal: {'; '.join(errors)}")

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "proposal_id": str(self.proposal_id),
            "intent": self.intent.value,
            "hypothesis": self.hypothesis,
            "action_type": self.action_type.value,
            "target": {
                "path": self.target.path,
                "symbol": self.target.symbol,
            },
            "change_summary": self.change_summary,
            "expected_effect": {
                "tests": self.expected_effect.tests.value,
                "behavior": self.expected_effect.behavior,
            },
            "risk_level": self.risk_level.value,
            "rollback_plan": self.rollback_plan,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Proposal":
        """Create proposal from dictionary."""
        return cls(
            proposal_id=UUID(data["proposal_id"]),
            intent=ProposalIntent(data["intent"]),
            hypothesis=data["hypothesis"],
            action_type=ActionType(data["action_type"]),
            target=Target(
                path=data["target"]["path"],
                symbol=data["target"].get("symbol"),
            ),
            change_summary=data["change_summary"],
            expected_effect=ExpectedEffect(
                tests=TestExpectation(data["expected_effect"]["tests"]),
                behavior=data["expected_effect"]["behavior"],
            ),
            risk_level=RiskLevel(data["risk_level"]),
            rollback_plan=data["rollback_plan"],
        )

    def is_mutation(self) -> bool:
        """Check if this proposal mutates code."""
        return self.action_type in {
            ActionType.EDIT_FILE,
            ActionType.ADD_FILE,
            ActionType.DELETE_FILE,
        }

    def is_test_action(self) -> bool:
        """Check if this proposal runs tests."""
        return self.action_type == ActionType.RUN_TESTS

    def is_read_only(self) -> bool:
        """Check if this proposal only reads/analyzes."""
        return self.action_type in {
            ActionType.READ_FILE,
            ActionType.SEARCH_REPO,
        }
