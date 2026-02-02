"""QA Gate for patch acceptance decisions.
from __future__ import annotations

The single authority for patch acceptance. Implements decision policy:
- REJECT if functional_fix or no_regression claims fail
- ACCEPT with escalation tags if scope issues but core claims pass
- REJECT if unresolved challenges remain
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from .qa_types import (
    Claim,
    ClaimType,
    ClaimVerdict,
    Evidence,
    QAAttempt,
    Verdict,
)

logger = logging.getLogger(__name__)


@dataclass
class GateDecision:
    """Decision from the QA gate."""

    accepted: bool
    reason: str
    rejection_reasons: list[str] = field(default_factory=list)
    escalation_tags: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "rejection_reasons": self.rejection_reasons,
            "escalation_tags": self.escalation_tags,
            "risk_flags": self.risk_flags,
        }


class QAGate:
    """Decision authority for patch acceptance.
    
    Policy:
    - REJECT if functional_fix claim is REJECT
    - REJECT if no_regression claim is REJECT
    - ACCEPT with escalation if scope_minimality REJECT but core claims pass
    - REJECT if CHALLENGE remains unresolved after evidence
    """

    # Claims that MUST pass for acceptance
    CRITICAL_CLAIMS = {ClaimType.FUNCTIONAL_FIX, ClaimType.NO_REGRESSION}

    # Claims that can be escalated rather than rejected
    ESCALATABLE_CLAIMS = {ClaimType.SCOPE_MINIMALITY}

    def __init__(
        self,
        *,
        strict_mode: bool = False,
        allow_unresolved_challenges: bool = False,
    ):
        """Initialize gate.
        
        Args:
            strict_mode: If True, any REJECT causes rejection.
            allow_unresolved_challenges: If True, unresolved CHALLENGE is not auto-reject.
        """
        self.strict_mode = strict_mode
        self.allow_unresolved_challenges = allow_unresolved_challenges

    def decide(
        self,
        claims: list[Claim],
        verdicts: list[ClaimVerdict],
        evidence: list[Evidence] | None = None,
    ) -> GateDecision:
        """Make acceptance decision based on claims and verdicts.
        
        Args:
            claims: List of claims about the patch.
            verdicts: List of verdicts from critic (and re-evaluations).
            evidence: Collected evidence (for reference).
        
        Returns:
            GateDecision with accept/reject and reasons.
        """
        # Build verdict mapping
        verdict_map = {v.claim_id: v for v in verdicts}

        rejection_reasons: list[str] = []
        escalation_tags: list[str] = []
        risk_flags: list[str] = []

        # Collect all risk flags
        for v in verdicts:
            risk_flags.extend(v.risk_flags)

        # Check critical claims
        for claim in claims:
            verdict = verdict_map.get(claim.id)
            if not verdict:
                continue

            if claim.type in self.CRITICAL_CLAIMS:
                if verdict.verdict == Verdict.REJECT:
                    rejection_reasons.append(
                        f"{claim.type.value} rejected: {verdict.reason}"
                    )
                elif verdict.verdict == Verdict.CHALLENGE:
                    if not self.allow_unresolved_challenges:
                        rejection_reasons.append(
                            f"{claim.type.value} unresolved: {verdict.reason}"
                        )

            elif claim.type in self.ESCALATABLE_CLAIMS:
                if verdict.verdict == Verdict.REJECT:
                    if self.strict_mode:
                        rejection_reasons.append(
                            f"{claim.type.value} rejected: {verdict.reason}"
                        )
                    else:
                        escalation_tags.append(f"{claim.type.value}_escalation")
                        logger.info(
                            "Escalating %s instead of rejecting: %s",
                            claim.type.value,
                            verdict.reason,
                        )

            elif verdict.verdict == Verdict.REJECT:
                risk_flags.append(f"{claim.type.value}_rejected")
                if self.strict_mode:
                    rejection_reasons.append(
                        f"{claim.type.value} rejected: {verdict.reason}"
                    )

        # Make decision
        if rejection_reasons:
            return GateDecision(
                accepted=False,
                reason="Patch rejected due to failed critical claims",
                rejection_reasons=rejection_reasons,
                escalation_tags=escalation_tags,
                risk_flags=risk_flags,
            )

        # Success path
        if escalation_tags:
            return GateDecision(
                accepted=True,
                reason="Patch accepted with escalations",
                rejection_reasons=[],
                escalation_tags=escalation_tags,
                risk_flags=risk_flags,
            )

        return GateDecision(
            accepted=True,
            reason="All claims validated",
            rejection_reasons=[],
            escalation_tags=[],
            risk_flags=risk_flags,
        )

    def evaluate_attempt(
        self,
        attempt: QAAttempt,
    ) -> GateDecision:
        """Evaluate a complete QA attempt.
        
        Args:
            attempt: The QAAttempt with claims, verdicts, evidence.
        
        Returns:
            GateDecision.
        """
        decision = self.decide(
            attempt.claims,
            attempt.verdicts,
            attempt.evidence,
        )

        # Update attempt with decision
        attempt.accepted = decision.accepted
        attempt.rejection_reason = (
            "; ".join(decision.rejection_reasons) if decision.rejection_reasons else None
        )
        attempt.escalation_tags = decision.escalation_tags

        return decision


def create_qa_gate(**kwargs) -> QAGate:
    """Factory function for QAGate."""
    return QAGate(**kwargs)
