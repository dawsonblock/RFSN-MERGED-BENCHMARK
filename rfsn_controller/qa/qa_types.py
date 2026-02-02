"""QA system types and schemas.
from __future__ import annotations

Defines the contract for claim-based adversarial QA:
- ClaimType: Types of claims a patch can make
- EvidenceType: Types of evidence to support claims
- Claim, Evidence, Verdict dataclasses
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ClaimType(Enum):
    """Types of claims a patch can make."""

    FUNCTIONAL_FIX = "functional_fix"      # "This patch fixes failing test(s) X"
    NO_REGRESSION = "no_regression"        # "This doesn't break previously passing tests"
    SCOPE_MINIMALITY = "scope_minimality"  # "The change is minimal/targeted"
    SPEC_ALIGNMENT = "spec_alignment"      # "This matches intended behavior"
    SAFETY_COMPLIANCE = "safety_compliance"  # "No forbidden paths/tools; no policy violations"


class EvidenceType(Enum):
    """Types of evidence to support claims."""

    TEST_RESULT = "test_result"        # (command, exit code, failing tests, delta)
    STATIC_CHECK = "static_check"      # (mypy/ruff/flake8 output)
    DELTA_MAP = "delta_map"            # (tests fixed/regressed)
    CODE_REFERENCE = "code_reference"  # (file + line range + symbol)
    POLICY_CHECK = "policy_check"      # (hygiene result, forbidden path scan)
    COVERAGE_SIGNAL = "coverage_signal"  # (touched file imported by tests)


class Verdict(Enum):
    """QA critic verdict for a claim."""

    ACCEPT = "ACCEPT"      # Claim is clearly valid
    CHALLENGE = "CHALLENGE"  # Need specific evidence
    REJECT = "REJECT"      # Claim is likely false


@dataclass
class Claim:
    """A claim made about a patch."""

    id: str
    type: ClaimType
    text: str
    required_evidence: list[EvidenceType] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "text": self.text,
            "required_evidence": [e.value for e in self.required_evidence],
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Claim":
        return cls(
            id=d["id"],
            type=ClaimType(d["type"]),
            text=d["text"],
            required_evidence=[EvidenceType(e) for e in d.get("required_evidence", [])],
        )


@dataclass
class Evidence:
    """Evidence collected to support or refute a claim."""

    type: EvidenceType
    data: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Evidence":
        return cls(
            type=EvidenceType(d["type"]),
            data=d.get("data", {}),
        )


@dataclass
class TestResultEvidence:
    """Evidence from running tests."""

    command: str
    exit_code: int
    failing_tests: list[str] = field(default_factory=list)
    passing_tests: list[str] = field(default_factory=list)
    duration_ms: int = 0

    def to_evidence(self) -> Evidence:
        return Evidence(
            type=EvidenceType.TEST_RESULT,
            data={
                "command": self.command,
                "exit_code": self.exit_code,
                "failing_tests": self.failing_tests,
                "passing_tests": self.passing_tests,
                "duration_ms": self.duration_ms,
            },
        )


@dataclass
class DeltaMapEvidence:
    """Evidence from test delta comparison."""

    fixed: list[str] = field(default_factory=list)      # fail→pass
    regressed: list[str] = field(default_factory=list)  # pass→fail
    still_failing: list[str] = field(default_factory=list)  # fail→fail

    def to_evidence(self) -> Evidence:
        return Evidence(
            type=EvidenceType.DELTA_MAP,
            data={
                "fixed": self.fixed,
                "regressed": self.regressed,
                "still_failing": self.still_failing,
            },
        )

    @property
    def has_regressions(self) -> bool:
        return len(self.regressed) > 0

    @property
    def has_fixes(self) -> bool:
        return len(self.fixed) > 0


@dataclass
class PolicyCheckEvidence:
    """Evidence from hygiene/policy checks."""

    is_valid: bool
    violations: list[str] = field(default_factory=list)
    diff_stats: dict[str, int] = field(default_factory=dict)  # lines_added, lines_removed, files_changed

    def to_evidence(self) -> Evidence:
        return Evidence(
            type=EvidenceType.POLICY_CHECK,
            data={
                "is_valid": self.is_valid,
                "violations": self.violations,
                "diff_stats": self.diff_stats,
            },
        )


@dataclass
class StaticCheckEvidence:
    """Evidence from static analysis tools."""

    tool: str  # mypy, ruff, flake8
    exit_code: int
    issues: list[dict[str, Any]] = field(default_factory=list)

    def to_evidence(self) -> Evidence:
        return Evidence(
            type=EvidenceType.STATIC_CHECK,
            data={
                "tool": self.tool,
                "exit_code": self.exit_code,
                "issues": self.issues,
            },
        )


@dataclass
class ClaimVerdict:
    """Critic's verdict on a single claim."""

    claim_id: str
    verdict: Verdict
    reason: str = ""
    evidence_request: str | None = None  # What evidence is needed for CHALLENGE
    risk_flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "evidence_request": self.evidence_request,
            "risk_flags": self.risk_flags,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ClaimVerdict":
        return cls(
            claim_id=d["claim_id"],
            verdict=Verdict(d["verdict"]),
            reason=d.get("reason", ""),
            evidence_request=d.get("evidence_request"),
            risk_flags=d.get("risk_flags", []),
        )


@dataclass
class QAAttempt:
    """Complete QA evaluation for a patch attempt."""

    attempt_id: str
    claims: list[Claim] = field(default_factory=list)
    verdicts: list[ClaimVerdict] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)

    # Final decision
    accepted: bool = False
    rejection_reason: str | None = None
    escalation_tags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "claims": [c.as_dict() for c in self.claims],
            "verdicts": [v.as_dict() for v in self.verdicts],
            "evidence": [e.as_dict() for e in self.evidence],
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "escalation_tags": self.escalation_tags,
        }

    def get_verdict(self, claim_id: str) -> ClaimVerdict | None:
        """Get verdict for a specific claim."""
        for v in self.verdicts:
            if v.claim_id == claim_id:
                return v
        return None

    def has_rejection(self, claim_type: ClaimType) -> bool:
        """Check if any claim of given type was rejected."""
        for claim in self.claims:
            if claim.type == claim_type:
                verdict = self.get_verdict(claim.id)
                if verdict and verdict.verdict == Verdict.REJECT:
                    return True
        return False

    def has_challenge(self, claim_type: ClaimType) -> bool:
        """Check if any claim of given type was challenged."""
        for claim in self.claims:
            if claim.type == claim_type:
                verdict = self.get_verdict(claim.id)
                if verdict and verdict.verdict == Verdict.CHALLENGE:
                    return True
        return False


# Maximum claims per attempt (hard cap)
MAX_CLAIMS_PER_ATTEMPT = 6

# Default required evidence for each claim type
DEFAULT_REQUIRED_EVIDENCE: dict[ClaimType, list[EvidenceType]] = {
    ClaimType.FUNCTIONAL_FIX: [EvidenceType.TEST_RESULT],
    ClaimType.NO_REGRESSION: [EvidenceType.TEST_RESULT, EvidenceType.DELTA_MAP],
    ClaimType.SCOPE_MINIMALITY: [EvidenceType.POLICY_CHECK],
    ClaimType.SPEC_ALIGNMENT: [EvidenceType.CODE_REFERENCE],
    ClaimType.SAFETY_COMPLIANCE: [EvidenceType.POLICY_CHECK],
}
