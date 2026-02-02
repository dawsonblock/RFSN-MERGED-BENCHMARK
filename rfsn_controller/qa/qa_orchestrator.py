"""QA Orchestrator for unified QA workflow.
from __future__ import annotations

Coordinates claim extraction, critique, evidence collection, and gating
in the controller repair loop.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .claim_extractor import ClaimExtractor, PatchContext
from .evidence_collector import EvidenceCollector
from .qa_critic import QACritic
from .qa_gate import GateDecision, QAGate
from .qa_persistence import QAPersistence
from .qa_types import Evidence, QAAttempt, Verdict

logger = logging.getLogger(__name__)


@dataclass
class QAConfig:
    """Configuration for QA orchestrator."""

    # Thresholds
    surgical_max_lines: int = 80
    surgical_max_files: int = 3

    # Behavior
    strict_mode: bool = False
    allow_unresolved_challenges: bool = False
    persist_outcomes: bool = True

    # Timeouts
    evidence_timeout_ms: int = 60000

    # Persistence
    db_path: str | None = None


class QAOrchestrator:
    """Unified QA workflow for the controller.
    
    Usage in controller:
        qa = QAOrchestrator(...)
        result = qa.evaluate_patch(diff, failing_tests, test_runner)
        if result.decision.accepted:
            apply_patch()
        else:
            # result.decision.rejection_reasons
    """

    def __init__(
        self,
        *,
        config: QAConfig | None = None,
        llm_call: Callable[[str, str], str] | None = None,
        test_runner: Callable[[str], dict[str, Any]] | None = None,
        delta_tracker: Any | None = None,
        hygiene_validator: Callable[[str], dict[str, Any]] | None = None,
        static_checker: Callable[[str], dict[str, Any]] | None = None,
    ):
        """Initialize orchestrator.
        
        Args:
            config: QA configuration.
            llm_call: LLM function for critic (optional).
            test_runner: Test execution function.
            delta_tracker: TestDeltaTracker instance.
            hygiene_validator: Patch hygiene validator.
            static_checker: Static analysis runner.
        """
        self.config = config or QAConfig()

        # Initialize components
        self.extractor = ClaimExtractor()
        self.critic = QACritic(llm_call=llm_call, fallback_to_rules=True)
        self.collector = EvidenceCollector(
            test_runner=test_runner,
            delta_tracker=delta_tracker,
            hygiene_validator=hygiene_validator,
            static_checker=static_checker,
            timeout_ms=self.config.evidence_timeout_ms,
        )
        self.gate = QAGate(
            strict_mode=self.config.strict_mode,
            allow_unresolved_challenges=self.config.allow_unresolved_challenges,
        )

        # Persistence (optional)
        self.persistence: QAPersistence | None = None
        if self.config.persist_outcomes and self.config.db_path:
            self.persistence = QAPersistence(self.config.db_path)
    
    def has_llm_critic(self) -> bool:
        """Check if the critic has an LLM backend (not just rule-based fallback).
        
        Returns:
            True if LLM-based critic is available, False if rule-based only.
        """
        return self.critic.llm_call is not None

    def evaluate_patch(
        self,
        diff: str,
        failing_tests: list[str],
        *,
        test_cmd: str = "",
        issue_text: str | None = None,
        attempt_id: str | None = None,
        failure_signature: str = "",
    ) -> "QAResult":
        """Run complete QA evaluation on a patch.
        
        Args:
            diff: The patch diff.
            failing_tests: List of failing test names.
            test_cmd: Test command for evidence collection.
            issue_text: Optional issue/requirement text.
            attempt_id: Optional attempt identifier.
            failure_signature: Normalized failure signature.
        
        Returns:
            QAResult with decision, claims, verdicts, evidence.
        """
        import uuid
        attempt_id = attempt_id or str(uuid.uuid4())[:8]

        # Parse diff for stats
        diff_stats = self._parse_diff_stats(diff)

        # Step 1: Extract claims
        context = PatchContext(
            failing_tests=failing_tests,
            diff_lines=diff_stats["lines_changed"],
            files_changed=diff_stats["files_changed"],
            touched_files=diff_stats["touched_files"],
            issue_text=issue_text,
            surgical_max_lines=self.config.surgical_max_lines,
            surgical_max_files=self.config.surgical_max_files,
        )
        claims = self.extractor.extract(context)
        logger.debug("Extracted %d claims for attempt %s", len(claims), attempt_id)

        # Step 2: Critique claims
        verdicts = self.critic.critique(
            claims,
            patch_summary={
                "lines_changed": diff_stats["lines_changed"],
                "files_changed": diff_stats["files_changed"],
                "touched_files": diff_stats["touched_files"],
            },
        )

        # Step 3: Collect evidence for challenges
        evidence: list[Evidence] = []
        for verdict in verdicts:
            if verdict.verdict == Verdict.CHALLENGE:
                collected = self.collector.collect_for_verdict(
                    verdict,
                    diff=diff,
                    test_cmd=test_cmd,
                )
                evidence.extend(collected)

                # Re-evaluate with evidence
                claim = next((c for c in claims if c.id == verdict.claim_id), None)
                if claim and collected:
                    new_verdict = self.critic.re_evaluate(claim, collected, verdict)
                    # Update verdict list
                    for i, v in enumerate(verdicts):
                        if v.claim_id == verdict.claim_id:
                            verdicts[i] = new_verdict
                            break

        # Step 4: Gate decision
        decision = self.gate.decide(claims, verdicts, evidence)

        # Step 5: Persist if enabled
        attempt = QAAttempt(
            attempt_id=attempt_id,
            claims=claims,
            verdicts=verdicts,
            evidence=evidence,
            accepted=decision.accepted,
            rejection_reason="; ".join(decision.rejection_reasons) if decision.rejection_reasons else None,
            escalation_tags=decision.escalation_tags,
        )

        if self.persistence:
            self.persistence.record_attempt(
                attempt,
                failure_signature=failure_signature,
                diff_stats=diff_stats,
            )

        return QAResult(
            attempt=attempt,
            decision=decision,
            diff_stats=diff_stats,
        )

    def _parse_diff_stats(self, diff: str) -> dict[str, Any]:
        """Parse diff for statistics."""
        if not diff:
            return {"lines_changed": 0, "files_changed": 0, "touched_files": []}

        lines = diff.splitlines()
        changed_lines = sum(1 for line in lines if line.startswith('+') or line.startswith('-'))

        files = set()
        for line in lines:
            if line.startswith('diff --git'):
                parts = line.split()
                if len(parts) >= 4:
                    files.add(parts[3].lstrip('b/'))

        return {
            "lines_changed": changed_lines,
            "files_changed": len(files),
            "touched_files": list(files),
        }

    def should_escalate_budget(self, result: "QAResult") -> bool:
        """Check if patch budget should be escalated based on QA result.
        
        Args:
            result: QA evaluation result.
        
        Returns:
            True if escalation is warranted.
        """
        # Escalate if scope was the only issue
        if result.decision.accepted and result.decision.escalation_tags:
            return "scope_minimality_escalation" in result.decision.escalation_tags

        # Escalate if rejection reason mentions scope
        for reason in result.decision.rejection_reasons:
            if "scope" in reason.lower() or "too small" in reason.lower():
                return True

        return False

    def close(self) -> None:
        """Clean up resources."""
        if self.persistence:
            self.persistence.close()


@dataclass
class QAResult:
    """Result from QA evaluation."""

    attempt: QAAttempt
    decision: GateDecision
    diff_stats: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.decision.accepted

    @property
    def rejection_reasons(self) -> list[str]:
        return self.decision.rejection_reasons

    @property
    def escalation_tags(self) -> list[str]:
        return self.decision.escalation_tags


def create_qa_orchestrator(**kwargs) -> QAOrchestrator:
    """Factory function for QAOrchestrator."""
    return QAOrchestrator(**kwargs)
