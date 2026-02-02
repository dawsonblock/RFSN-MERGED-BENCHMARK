"""
from __future__ import annotations
Scoring engine for proposal candidates.

Evaluates multiple candidate proposals and ranks them by quality
without executing any code.
"""

import re
from dataclasses import dataclass

from .proposal import Proposal


@dataclass
class CandidateScore:
    """Score for a proposal candidate."""

    proposal: Proposal
    total_score: float
    file_count_score: float  # Fewer files is better
    traceback_relevance: float  # Directly addresses traceback
    guard_quality: float  # Adds None/boundary/type guards
    api_preservation: float  # Preserves API surface
    refactor_penalty: float  # Avoids broad refactors
    test_narrative_match: float  # Matches failing test description
    details: dict  # Breakdown for debugging

    def __lt__(self, other: "CandidateScore") -> bool:
        """Compare scores (higher is better)."""
        return self.total_score < other.total_score


class ScoringEngine:
    """
    Scores proposal candidates for multi-model selection.

    Uses static analysis only - no execution.
    """

    def __init__(
        self,
        failing_tests: list[str] | None = None,
        traceback_frames: list[tuple[str, int]] | None = None,
        test_narrative: str | None = None,
    ):
        """
        Initialize scoring engine.

        Args:
            failing_tests: List of failing test IDs
            traceback_frames: List of (file, lineno) from traceback
            test_narrative: Description of what test expects
        """
        self.failing_tests = failing_tests or []
        self.traceback_frames = traceback_frames or []
        self.test_narrative = test_narrative or ""

        # Extract key info from traceback
        self.traceback_files = {frame[0] for frame in self.traceback_frames}
        self.traceback_symbols = self._extract_traceback_symbols()

    def _extract_traceback_symbols(self) -> set[str]:
        """Extract function/class names from traceback frames."""
        # This would ideally parse actual traceback text
        # For now, return empty set - can be enhanced
        return set()

    def score_proposal(self, proposal: Proposal) -> CandidateScore:
        """
        Score a single proposal candidate.

        Returns CandidateScore with breakdown.
        """
        details = {}

        # 1. File count score (fewer is better)
        file_count = 1  # Proposals touch one file
        file_count_score = 1.0 / max(file_count, 1.0)
        details["file_count"] = file_count

        # 2. Traceback relevance (directly addresses failing frame)
        traceback_relevance = self._score_traceback_relevance(proposal)
        details["traceback_relevance"] = traceback_relevance

        # 3. Guard quality (adds defensive checks)
        guard_quality = self._score_guard_quality(proposal)
        details["guard_quality"] = guard_quality

        # 4. API preservation (avoids breaking changes)
        api_preservation = self._score_api_preservation(proposal)
        details["api_preservation"] = api_preservation

        # 5. Refactor penalty (avoids broad rewrites)
        refactor_penalty = self._score_refactor_penalty(proposal)
        details["refactor_penalty"] = refactor_penalty

        # 6. Test narrative match
        test_narrative_match = self._score_test_narrative_match(proposal)
        details["test_narrative_match"] = test_narrative_match

        # Weighted total
        total_score = (
            file_count_score * 1.0
            + traceback_relevance * 3.0  # Most important
            + guard_quality * 2.0
            + api_preservation * 1.5
            + refactor_penalty * 1.0  # Negative penalty
            + test_narrative_match * 2.0
        )

        return CandidateScore(
            proposal=proposal,
            total_score=total_score,
            file_count_score=file_count_score,
            traceback_relevance=traceback_relevance,
            guard_quality=guard_quality,
            api_preservation=api_preservation,
            refactor_penalty=refactor_penalty,
            test_narrative_match=test_narrative_match,
            details=details,
        )

    def _score_traceback_relevance(self, proposal: Proposal) -> float:
        """Score how well proposal addresses traceback."""
        if not self.traceback_frames:
            return 0.5  # No traceback info, neutral score

        # Check if proposal targets a file in traceback
        target_file = proposal.target.path
        if target_file in self.traceback_files:
            # Direct hit on traceback file
            score = 1.0

            # Bonus if symbol also matches
            if proposal.target.symbol and proposal.target.symbol in self.traceback_symbols:
                score = 1.5

            return min(score, 1.0)

        return 0.0  # Doesn't address traceback

    def _score_guard_quality(self, proposal: Proposal) -> float:
        """Score presence of defensive checks."""
        if not proposal.is_mutation():
            return 0.5  # Neutral for non-mutations

        # Look for defensive patterns in change_summary
        summary = proposal.change_summary.lower()
        hypothesis = proposal.hypothesis.lower()

        guard_keywords = [
            "none check",
            "null check",
            "guard",
            "boundary",
            "validate",
            "type check",
            "isinstance",
            "if not",
            "raise",
            "default",
        ]

        score = 0.0
        for keyword in guard_keywords:
            if keyword in summary or keyword in hypothesis:
                score += 0.2

        return min(score, 1.0)

    def _score_api_preservation(self, proposal: Proposal) -> float:
        """Score how well proposal preserves existing API."""
        if not proposal.is_mutation():
            return 1.0  # No API changes

        # Look for API-breaking patterns
        breaking_patterns = [
            r"rename.*function",
            r"rename.*class",
            r"change.*signature",
            r"remove.*parameter",
            r"change.*return",
            "breaking change",
            "incompatible",
        ]

        text = (proposal.change_summary + " " + proposal.hypothesis).lower()

        for pattern in breaking_patterns:
            if re.search(pattern, text):
                return 0.0  # Likely breaks API

        # Look for preserving patterns
        preserving_patterns = [
            "backward.?compat",
            "preserve",
            "maintain.*api",
            "add.*guard",
            "internal.*only",
        ]

        for pattern in preserving_patterns:
            if re.search(pattern, text):
                return 1.0  # Explicitly preserves API

        return 0.7  # Default: probably safe

    def _score_refactor_penalty(self, proposal: Proposal) -> float:
        """Penalize broad refactors (negative score)."""
        if not proposal.is_mutation():
            return 0.0  # No penalty

        # Refactor keywords that indicate broad changes
        refactor_keywords = [
            "refactor",
            "reorganize",
            "restructure",
            "rewrite",
            "modernize",
            "clean up",
            "simplify",
            "extract",
            "inline",
            "move.*to",
        ]

        text = (proposal.change_summary + " " + proposal.hypothesis).lower()

        for keyword in refactor_keywords:
            if re.search(keyword, text):
                # Heavy penalty if intent is refactor
                if proposal.intent.value == "refactor":
                    return -2.0
                # Light penalty if accidental refactor language
                return -0.5

        return 0.0  # No penalty

    def _score_test_narrative_match(self, proposal: Proposal) -> float:
        """Score how well proposal matches test failure narrative."""
        if not self.test_narrative:
            return 0.5  # No narrative, neutral

        narrative = self.test_narrative.lower()
        hypothesis = proposal.hypothesis.lower()
        summary = proposal.change_summary.lower()

        # Extract key nouns/verbs from narrative
        # Simple keyword matching (can be enhanced with NLP)
        narrative_words = set(re.findall(r"\b\w{4,}\b", narrative))
        hypothesis_words = set(re.findall(r"\b\w{4,}\b", hypothesis))
        summary_words = set(re.findall(r"\b\w{4,}\b", summary))

        # Jaccard similarity
        proposal_words = hypothesis_words | summary_words
        if not narrative_words or not proposal_words:
            return 0.5

        intersection = narrative_words & proposal_words
        union = narrative_words | proposal_words

        similarity = len(intersection) / len(union) if union else 0.0

        return similarity

    def score_candidates(self, candidates: list[Proposal]) -> list[CandidateScore]:
        """
        Score multiple candidates and return sorted by quality (best first).

        Args:
            candidates: List of proposal candidates

        Returns:
            List of CandidateScore sorted by total_score descending
        """
        scores = [self.score_proposal(proposal) for proposal in candidates]
        scores.sort(reverse=True)  # Highest score first
        return scores

    def select_best(
        self, candidates: list[Proposal], top_n: int = 1
    ) -> list[Proposal]:
        """
        Select the best N candidates.

        Args:
            candidates: List of proposal candidates
            top_n: Number of top candidates to return

        Returns:
            List of top N proposals
        """
        if not candidates:
            return []

        scores = self.score_candidates(candidates)
        return [score.proposal for score in scores[:top_n]]
