"""Claim extractor for patch attempts.
from __future__ import annotations

Extracts structured claims from patch context. Rules-based, no LLM.

Rules:
- Always emit: FUNCTIONAL_FIX, NO_REGRESSION, SAFETY_COMPLIANCE
- Emit SCOPE_MINIMALITY if patch ≤ surgical threshold
- Emit SPEC_ALIGNMENT only with issue/requirement text
- Hard cap: 6 claims max
"""

from dataclasses import dataclass

from .qa_types import (
    DEFAULT_REQUIRED_EVIDENCE,
    MAX_CLAIMS_PER_ATTEMPT,
    Claim,
    ClaimType,
)


@dataclass
class PatchContext:
    """Context for claim extraction."""

    # Failure info
    failing_tests: list[str]
    error_signatures: list[str] = None  # type: ignore

    # Patch info
    diff_lines: int = 0
    files_changed: int = 0
    touched_files: list[str] = None  # type: ignore

    # Intent
    is_bugfix: bool = True
    is_feature: bool = False

    # Optional context
    issue_text: str | None = None
    requirement_text: str | None = None

    # Thresholds
    surgical_max_lines: int = 80
    surgical_max_files: int = 3

    def __post_init__(self):
        if self.error_signatures is None:
            self.error_signatures = []
        if self.touched_files is None:
            self.touched_files = []

    @property
    def is_surgical(self) -> bool:
        """Check if patch is within surgical bounds."""
        return (
            self.diff_lines <= self.surgical_max_lines
            and self.files_changed <= self.surgical_max_files
        )

    @property
    def has_spec(self) -> bool:
        """Check if specification context is available."""
        return bool(self.issue_text or self.requirement_text)


class ClaimExtractor:
    """Extracts claims from patch context.
    
    Rule-based extraction, no LLM calls.
    """

    def __init__(
        self,
        *,
        always_emit: list[ClaimType] | None = None,
        max_claims: int = MAX_CLAIMS_PER_ATTEMPT,
    ):
        """Initialize extractor.
        
        Args:
            always_emit: Claim types to always emit. Defaults to core set.
            max_claims: Maximum claims per attempt.
        """
        self.always_emit = always_emit or [
            ClaimType.FUNCTIONAL_FIX,
            ClaimType.NO_REGRESSION,
            ClaimType.SAFETY_COMPLIANCE,
        ]
        self.max_claims = max_claims

    def extract(self, context: PatchContext) -> list[Claim]:
        """Extract claims from patch context.
        
        Args:
            context: The patch context.
        
        Returns:
            List of claims (max self.max_claims).
        """
        claims: list[Claim] = []
        claim_counter = 0

        def add_claim(claim_type: ClaimType, text: str) -> None:
            nonlocal claim_counter
            if len(claims) >= self.max_claims:
                return
            claim_counter += 1
            claims.append(Claim(
                id=f"C{claim_counter}",
                type=claim_type,
                text=text,
                required_evidence=list(DEFAULT_REQUIRED_EVIDENCE.get(claim_type, [])),
            ))

        # Always emit: FUNCTIONAL_FIX
        if ClaimType.FUNCTIONAL_FIX in self.always_emit:
            if context.failing_tests:
                test_list = ", ".join(context.failing_tests[:3])
                if len(context.failing_tests) > 3:
                    test_list += f" (+{len(context.failing_tests) - 3} more)"
                add_claim(
                    ClaimType.FUNCTIONAL_FIX,
                    f"Fixes failing tests: {test_list}",
                )
            else:
                add_claim(
                    ClaimType.FUNCTIONAL_FIX,
                    "Fixes the reported issue",
                )

        # Always emit: NO_REGRESSION
        if ClaimType.NO_REGRESSION in self.always_emit:
            add_claim(
                ClaimType.NO_REGRESSION,
                "No new test failures introduced",
            )

        # Always emit: SAFETY_COMPLIANCE
        if ClaimType.SAFETY_COMPLIANCE in self.always_emit:
            add_claim(
                ClaimType.SAFETY_COMPLIANCE,
                "Patch complies with hygiene policies",
            )

        # Conditional: SCOPE_MINIMALITY
        if context.is_surgical:
            add_claim(
                ClaimType.SCOPE_MINIMALITY,
                f"Minimal change: {context.diff_lines} lines, {context.files_changed} files",
            )

        # Conditional: SPEC_ALIGNMENT (only with spec context)
        if context.has_spec:
            spec_source = "issue" if context.issue_text else "requirement"
            add_claim(
                ClaimType.SPEC_ALIGNMENT,
                f"Aligns with {spec_source} specification",
            )

        return claims[:self.max_claims]

    def extract_from_diff(
        self,
        diff: str,
        failing_tests: list[str],
        *,
        issue_text: str | None = None,
        surgical_max_lines: int = 80,
        surgical_max_files: int = 3,
    ) -> list[Claim]:
        """Convenience method to extract claims from a diff string.
        
        Args:
            diff: The patch diff.
            failing_tests: List of failing test names.
            issue_text: Optional issue/requirement text.
            surgical_max_lines: Max lines for surgical scope.
            surgical_max_files: Max files for surgical scope.
        
        Returns:
            List of claims.
        """
        # Parse diff for stats
        lines = diff.splitlines() if diff else []
        diff_lines = sum(1 for line in lines if line.startswith('+') or line.startswith('-'))

        files = set()
        for line in lines:
            if line.startswith('diff --git'):
                # Extract filename from "diff --git a/foo.py b/foo.py"
                parts = line.split()
                if len(parts) >= 4:
                    files.add(parts[3].lstrip('b/'))

        context = PatchContext(
            failing_tests=failing_tests,
            diff_lines=diff_lines,
            files_changed=len(files),
            touched_files=list(files),
            issue_text=issue_text,
            surgical_max_lines=surgical_max_lines,
            surgical_max_files=surgical_max_files,
        )

        return self.extract(context)


def create_claim_extractor(
    **kwargs,
) -> ClaimExtractor:
    """Factory function for ClaimExtractor.
    
    Args:
        **kwargs: Arguments passed to ClaimExtractor.
    
    Returns:
        Configured ClaimExtractor.
    """
    return ClaimExtractor(**kwargs)
