"""QA Integration - Bridge between Planner and QA Orchestrator.

Connects planner step verification with QA claim-based validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .schema import ControllerOutcome, FailureCategory, FailureEvidence, Step

if TYPE_CHECKING:
    from ..qa.qa_orchestrator import QAOrchestrator, QAResult
    from ..qa.qa_types import Claim

logger = logging.getLogger(__name__)


@dataclass
class StepQAResult:
    """Result of QA verification for a step."""
    
    step_id: str
    accepted: bool
    rejection_reasons: list[str]
    escalation_tags: list[str]
    should_revise: bool = False
    revision_hints: dict[str, Any] = None
    raw_result: QAResult | None = None
    
    def __post_init__(self):
        if self.revision_hints is None:
            self.revision_hints = {}


class PlannerQABridge:
    """Connects planner steps with QA claim verification."""
    
    def __init__(self, qa_orchestrator: QAOrchestrator | None = None):
        """Initialize the bridge.
        
        Args:
            qa_orchestrator: QA orchestrator instance for claim verification.
        """
        self._qa = qa_orchestrator
    
    @property
    def enabled(self) -> bool:
        """Check if QA integration is enabled."""
        return self._qa is not None
    
    def extract_claims_for_step(
        self,
        step: Step,
        diff: str,
        outcome: ControllerOutcome,
    ) -> list[Claim]:
        """Extract claims from a step outcome for QA evaluation.
        
        Args:
            step: The executed step.
            diff: The diff produced.
            outcome: Controller outcome.
            
        Returns:
            List of claims to verify.
        """
        if not self.enabled:
            logger.debug("QA disabled - skipping claim extraction for step %s", step.step_id)
            return []
        
        from ..qa.qa_types import Claim, ClaimType, EvidenceType
        
        claims = []
        
        # Always claim functional fix if step succeeded
        if outcome.success:
            claims.append(Claim(
                id=f"{step.step_id}-functional",
                type=ClaimType.FUNCTIONAL_FIX,
                text=f"Step '{step.title}' achieves: {step.success_criteria}",
                required_evidence=[EvidenceType.TEST_RESULT],
            ))
        
        # Claim no regression if tests passed
        if outcome.tests_passed:
            claims.append(Claim(
                id=f"{step.step_id}-no-regression",
                type=ClaimType.NO_REGRESSION,
                text="This change doesn't break previously passing tests",
                required_evidence=[EvidenceType.TEST_RESULT, EvidenceType.DELTA_MAP],
            ))
        
        # Claim scope minimality based on diff size
        if diff:
            lines_changed = len(diff.splitlines())
            if lines_changed < 50:
                claims.append(Claim(
                    id=f"{step.step_id}-scope",
                    type=ClaimType.SCOPE_MINIMALITY,
                    text=f"Change is minimal ({lines_changed} lines)",
                    required_evidence=[EvidenceType.POLICY_CHECK],
                ))
        
        # Claim safety compliance if allowed_files were respected
        claims.append(Claim(
            id=f"{step.step_id}-safety",
            type=ClaimType.SAFETY_COMPLIANCE,
            text="Change respects allowed file boundaries",
            required_evidence=[EvidenceType.POLICY_CHECK],
        ))
        
        return claims
    
    def verify_step_outcome(
        self,
        step: Step,
        outcome: ControllerOutcome,
        diff: str,
        test_cmd: str = "",
    ) -> StepQAResult:
        """Run QA verification on a step outcome.
        
        Args:
            step: The executed step.
            outcome: Controller outcome.
            diff: The diff produced.
            test_cmd: Test command for evidence collection.
            
        Returns:
            StepQAResult with verdict and revision hints.
        """
        if not self.enabled:
            return StepQAResult(
                step_id=step.step_id,
                accepted=outcome.success,
                rejection_reasons=[],
                escalation_tags=[],
            )
        
        logger.info("Running QA verification for step %s", step.step_id)
        
        try:
            # Run QA evaluation
            qa_result = self._qa.evaluate_patch(
                diff=diff,
                failing_tests=[],  # Step-level doesn't have pre-failure info
                test_cmd=test_cmd,
                attempt_id=step.step_id,
            )
            
            return StepQAResult(
                step_id=step.step_id,
                accepted=qa_result.accepted,
                rejection_reasons=qa_result.rejection_reasons,
                escalation_tags=qa_result.escalation_tags,
                should_revise=self._should_revise_based_on_qa(qa_result),
                revision_hints=self._get_revision_hints(qa_result),
                raw_result=qa_result,
            )
            
        except Exception as e:
            logger.exception("QA verification failed for step %s: %s", step.step_id, e)
            # QA failed, don't block execution
            return StepQAResult(
                step_id=step.step_id,
                accepted=outcome.success,
                rejection_reasons=[f"QA error: {e!s}"],
                escalation_tags=["qa_error"],
            )
    
    def _should_revise_based_on_qa(self, qa_result: QAResult) -> bool:
        """Determine if step should be revised based on QA result.
        
        Args:
            qa_result: QA evaluation result.
            
        Returns:
            True if revision is warranted.
        """
        from ..qa.qa_types import ClaimType
        
        # Revise if regression detected
        if qa_result.attempt.has_rejection(ClaimType.NO_REGRESSION):
            return True
        
        # Revise if safety violation
        if qa_result.attempt.has_rejection(ClaimType.SAFETY_COMPLIANCE):
            return True
        
        return False
    
    def _get_revision_hints(self, qa_result: QAResult) -> dict[str, Any]:
        """Extract revision hints from QA result.
        
        Args:
            qa_result: QA evaluation result.
            
        Returns:
            Dictionary of hints for revision.
        """
        hints = {}
        
        for verdict in qa_result.attempt.verdicts:
            if verdict.evidence_request:
                hints[verdict.claim_id] = {
                    "reason": verdict.reason,
                    "evidence_needed": verdict.evidence_request,
                    "risk_flags": verdict.risk_flags,
                }
        
        # Add delta map info if available
        for evidence in qa_result.attempt.evidence:
            if evidence.type.value == "delta_map":
                hints["delta_map"] = evidence.data
        
        return hints
    
    def convert_qa_to_failure_evidence(
        self,
        step_result: StepQAResult,
    ) -> FailureEvidence:
        """Convert QA rejection to FailureEvidence for revision.
        
        Args:
            step_result: QA evaluation result wrapper.
            
        Returns:
            FailureEvidence for revision strategies.
        """
        qa_result = step_result.raw_result
        if not qa_result:
            return FailureEvidence(
                category=FailureCategory.UNKNOWN,
                suggestion="QA Rejected (no details)",
            )
        from ..qa.qa_types import ClaimType
        
        # Determine category from rejection type
        category = FailureCategory.UNKNOWN
        if qa_result.attempt.has_rejection(ClaimType.NO_REGRESSION):
            category = FailureCategory.TEST_REGRESSION
        elif qa_result.attempt.has_rejection(ClaimType.SAFETY_COMPLIANCE):
            category = FailureCategory.SANDBOX_VIOLATION
        
        # Extract test info from delta map
        failing_tests = []
        for evidence in qa_result.attempt.evidence:
            if evidence.type.value == "delta_map":
                failing_tests = evidence.data.get("regressed", [])
        
        return FailureEvidence(
            category=category,
            top_failing_tests=failing_tests[:5],
            suggestion="; ".join(qa_result.rejection_reasons[:2]),
        )


class StepClaimGenerator:
    """Generates appropriate claims for different step types."""
    
    STEP_CLAIM_MAP = {
        "analyze": ["spec_alignment"],
        "implement": ["functional_fix", "no_regression", "scope_minimality"],
        "test": ["functional_fix", "no_regression"],
        "verify": ["no_regression"],
        "refactor": ["no_regression", "scope_minimality"],
    }
    
    @classmethod
    def get_claims_for_step_type(cls, step_id: str) -> list[str]:
        """Get claim types appropriate for a step.
        
        Args:
            step_id: Step identifier.
            
        Returns:
            List of claim type names.
        """
        step_lower = step_id.lower()
        
        for step_type, claims in cls.STEP_CLAIM_MAP.items():
            if step_type in step_lower:
                return claims
        
        # Default claims
        return ["functional_fix", "safety_compliance"]
