"""QA Critic for claim evaluation.
from __future__ import annotations

LLM-based structured critique that evaluates claims about a patch.
Outputs ACCEPT/CHALLENGE/REJECT verdicts with evidence requests.

The critic CANNOT propose code. It can only evaluate claims.
"""

import json
import logging
from collections.abc import Callable
from typing import Any

from .qa_types import (
    Claim,
    ClaimType,
    ClaimVerdict,
    Evidence,
    Verdict,
)

logger = logging.getLogger(__name__)


# Structured prompt for QA critic
QA_CRITIC_SYSTEM_PROMPT = """You are a QA critic. Your ONLY job is to evaluate claims about a code patch.

RULES:
1. You CANNOT propose code changes
2. You CANNOT ask for tools you don't have
3. You MUST output valid JSON only
4. For each claim, you MUST output exactly one verdict

VERDICTS:
- ACCEPT: The claim is clearly valid based on available information
- CHALLENGE: Need specific evidence to verify (specify what evidence)
- REJECT: The claim is likely false based on available information

EVIDENCE YOU CAN REQUEST:
- test_result: Run specific tests
- delta_map: Show which tests changed state (fixed/regressed)
- static_check: Run linter/type checker
- policy_check: Check hygiene policies
- code_reference: Show specific file:line ranges"""

QA_CRITIC_USER_TEMPLATE = """Evaluate these claims about a patch:

CLAIMS:
{claims_json}

PATCH SUMMARY:
- Files changed: {files_changed}
- Lines changed: {lines_changed}
- Files touched: {touched_files}

{additional_context}

Output JSON only:
{{"verdicts": [{{"claim_id": "C1", "verdict": "ACCEPT|CHALLENGE|REJECT",
"reason": "...", "evidence_request": "...", "risk_flags": []}}]}}"""


class QACritic:
    """LLM-based critic for claim evaluation.
    
    Produces structured ACCEPT/CHALLENGE/REJECT verdicts.
    """

    def __init__(
        self,
        *,
        llm_call: Callable[[str, str], str] | None = None,
        fallback_to_rules: bool = True,
    ):
        """Initialize critic.
        
        Args:
            llm_call: Function(system_prompt, user_prompt) -> response.
                      If None, uses rule-based fallback.
            fallback_to_rules: Use rules if LLM fails.
        """
        self.llm_call = llm_call
        self.fallback_to_rules = fallback_to_rules

    def critique(
        self,
        claims: list[Claim],
        *,
        patch_summary: dict[str, Any] | None = None,
        additional_context: str = "",
    ) -> list[ClaimVerdict]:
        """Evaluate claims and produce verdicts.
        
        Args:
            claims: Claims to evaluate.
            patch_summary: Summary of the patch (files, lines, etc.).
            additional_context: Additional context for the critic.
        
        Returns:
            List of verdicts for each claim.
        """
        if not claims:
            return []

        patch_summary = patch_summary or {}

        # Try LLM-based critique
        if self.llm_call:
            try:
                verdicts = self._llm_critique(claims, patch_summary, additional_context)
                if verdicts:
                    return verdicts
            except Exception as e:
                logger.warning("LLM critique failed: %s", e)
                if not self.fallback_to_rules:
                    raise

        # Fallback to rule-based critique
        return self._rule_based_critique(claims, patch_summary)

    def _llm_critique(
        self,
        claims: list[Claim],
        patch_summary: dict[str, Any],
        additional_context: str,
    ) -> list[ClaimVerdict]:
        """LLM-based claim evaluation."""
        claims_json = json.dumps([c.as_dict() for c in claims], indent=2)

        user_prompt = QA_CRITIC_USER_TEMPLATE.format(
            claims_json=claims_json,
            files_changed=patch_summary.get("files_changed", "unknown"),
            lines_changed=patch_summary.get("lines_changed", "unknown"),
            touched_files=", ".join(patch_summary.get("touched_files", [])),
            additional_context=additional_context,
        )

        response = self.llm_call(QA_CRITIC_SYSTEM_PROMPT, user_prompt)

        # Parse JSON response
        try:
            # Handle markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            data = json.loads(response.strip())
            verdicts = []
            for v in data.get("verdicts", []):
                verdicts.append(ClaimVerdict.from_dict(v))
            return verdicts
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to parse LLM response: %s", e)
            raise

    def _rule_based_critique(
        self,
        claims: list[Claim],
        patch_summary: dict[str, Any],
    ) -> list[ClaimVerdict]:
        """Rule-based fallback for claim evaluation.
        
        Simple heuristics when LLM is unavailable.
        """
        verdicts = []

        for claim in claims:
            verdict = self._rule_based_verdict(claim, patch_summary)
            verdicts.append(verdict)

        return verdicts

    def _rule_based_verdict(
        self,
        claim: Claim,
        patch_summary: dict[str, Any],
    ) -> ClaimVerdict:
        """Produce rule-based verdict for a single claim."""

        # Default: CHALLENGE with evidence request
        if claim.type == ClaimType.FUNCTIONAL_FIX:
            return ClaimVerdict(
                claim_id=claim.id,
                verdict=Verdict.CHALLENGE,
                reason="Need test execution to verify fix",
                evidence_request="Run tests and provide test_result + delta_map",
            )

        elif claim.type == ClaimType.NO_REGRESSION:
            return ClaimVerdict(
                claim_id=claim.id,
                verdict=Verdict.CHALLENGE,
                reason="Need test execution to verify no regressions",
                evidence_request="Run full test suite and provide delta_map",
            )

        elif claim.type == ClaimType.SAFETY_COMPLIANCE:
            return ClaimVerdict(
                claim_id=claim.id,
                verdict=Verdict.CHALLENGE,
                reason="Need policy check to verify compliance",
                evidence_request="Run policy_check",
            )

        elif claim.type == ClaimType.SCOPE_MINIMALITY:
            lines = patch_summary.get("lines_changed", 0)
            files = patch_summary.get("files_changed", 0)

            if lines > 300 or files > 8:
                return ClaimVerdict(
                    claim_id=claim.id,
                    verdict=Verdict.REJECT,
                    reason=f"Patch too large: {lines} lines, {files} files",
                    risk_flags=["too_broad"],
                )
            elif lines > 150 or files > 5:
                return ClaimVerdict(
                    claim_id=claim.id,
                    verdict=Verdict.CHALLENGE,
                    reason="Patch size is moderate, verify all changes are necessary",
                    evidence_request="Provide diff breakdown",
                )
            else:
                return ClaimVerdict(
                    claim_id=claim.id,
                    verdict=Verdict.ACCEPT,
                    reason=f"Patch is minimal: {lines} lines, {files} files",
                )

        elif claim.type == ClaimType.SPEC_ALIGNMENT:
            return ClaimVerdict(
                claim_id=claim.id,
                verdict=Verdict.CHALLENGE,
                reason="Need to verify alignment with specification",
                evidence_request="Provide code_reference for key changes",
            )

        # Unknown claim type
        return ClaimVerdict(
            claim_id=claim.id,
            verdict=Verdict.CHALLENGE,
            reason="Unknown claim type, need verification",
            evidence_request="Provide relevant evidence",
        )

    def re_evaluate(
        self,
        claim: Claim,
        evidence: list[Evidence],
        original_verdict: ClaimVerdict,
    ) -> ClaimVerdict:
        """Re-evaluate a claim after receiving evidence.
        
        Args:
            claim: The claim being evaluated.
            evidence: Collected evidence.
            original_verdict: Original CHALLENGE verdict.
        
        Returns:
            Updated verdict (ACCEPT or REJECT).
        """
        # Rule-based re-evaluation based on evidence
        for ev in evidence:
            if ev.type.value == "test_result":
                exit_code = ev.data.get("exit_code", 1)
                failing = ev.data.get("failing_tests", [])

                if claim.type == ClaimType.FUNCTIONAL_FIX:
                    if exit_code == 0 or len(failing) == 0:
                        return ClaimVerdict(
                            claim_id=claim.id,
                            verdict=Verdict.ACCEPT,
                            reason="Tests pass after patch",
                        )
                    else:
                        return ClaimVerdict(
                            claim_id=claim.id,
                            verdict=Verdict.REJECT,
                            reason=f"Tests still failing: {failing[:3]}",
                        )

            elif ev.type.value == "delta_map":
                regressed = ev.data.get("regressed", [])
                # fixed = ev.data.get("fixed", [])  # Available but currently unused

                if claim.type == ClaimType.NO_REGRESSION:
                    if regressed:
                        return ClaimVerdict(
                            claim_id=claim.id,
                            verdict=Verdict.REJECT,
                            reason=f"Regressions detected: {regressed[:3]}",
                            risk_flags=["regression"],
                        )
                    else:
                        return ClaimVerdict(
                            claim_id=claim.id,
                            verdict=Verdict.ACCEPT,
                            reason="No regressions detected",
                        )

            elif ev.type.value == "policy_check":
                is_valid = ev.data.get("is_valid", False)
                violations = ev.data.get("violations", [])

                if claim.type == ClaimType.SAFETY_COMPLIANCE:
                    if is_valid:
                        return ClaimVerdict(
                            claim_id=claim.id,
                            verdict=Verdict.ACCEPT,
                            reason="Policy check passed",
                        )
                    else:
                        return ClaimVerdict(
                            claim_id=claim.id,
                            verdict=Verdict.REJECT,
                            reason=f"Policy violations: {violations[:3]}",
                        )

        # No conclusive evidence, remain challenged
        return original_verdict


def create_qa_critic(
    llm_call: Callable[[str, str], str] | None = None,
    **kwargs,
) -> QACritic:
    """Factory function for QACritic.
    
    Args:
        llm_call: LLM call function.
        **kwargs: Additional arguments.
    
    Returns:
        Configured QACritic.
    """
    return QACritic(llm_call=llm_call, **kwargs)
