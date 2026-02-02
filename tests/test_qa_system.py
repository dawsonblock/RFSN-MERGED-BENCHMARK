"""Unit tests for QA claim-based verification system."""

import os
import tempfile

import pytest


class TestClaimTypes:
    """Tests for QA types."""

    def test_claim_serialization(self):
        """Claims serialize to dict/JSON."""
        from rfsn_controller.qa import Claim, ClaimType, EvidenceType

        claim = Claim(
            id="C1",
            type=ClaimType.FUNCTIONAL_FIX,
            text="Fixes test_foo",
            required_evidence=[EvidenceType.TEST_RESULT],
        )
        d = claim.as_dict()
        assert d["id"] == "C1"
        assert d["type"] == "functional_fix"
        assert "test_result" in d["required_evidence"]

    def test_claim_deserialization(self):
        """Claims deserialize from dict."""
        from rfsn_controller.qa import Claim, ClaimType

        d = {"id": "C2", "type": "no_regression", "text": "No breaks", "required_evidence": []}
        claim = Claim.from_dict(d)
        assert claim.id == "C2"
        assert claim.type == ClaimType.NO_REGRESSION

    def test_qa_attempt_verdict_lookup(self):
        """QAAttempt can look up verdicts by claim."""
        from rfsn_controller.qa import Claim, ClaimType, ClaimVerdict, QAAttempt, Verdict

        attempt = QAAttempt(
            attempt_id="test-1",
            claims=[Claim("C1", ClaimType.FUNCTIONAL_FIX, "Fix")],
            verdicts=[ClaimVerdict("C1", Verdict.ACCEPT, "Tests pass")],
        )
        v = attempt.get_verdict("C1")
        assert v is not None
        assert v.verdict == Verdict.ACCEPT


class TestClaimExtractor:
    """Tests for claim extraction."""

    def test_always_emits_core_claims(self):
        """Extractor always emits core claim types."""
        from rfsn_controller.qa import ClaimExtractor, ClaimType, PatchContext

        extractor = ClaimExtractor()
        context = PatchContext(failing_tests=["test_a", "test_b"])
        claims = extractor.extract(context)

        types = {c.type for c in claims}
        assert ClaimType.FUNCTIONAL_FIX in types
        assert ClaimType.NO_REGRESSION in types
        assert ClaimType.SAFETY_COMPLIANCE in types

    def test_scope_minimality_for_surgical_patch(self):
        """Scope minimality claim emitted for small patches."""
        from rfsn_controller.qa import ClaimExtractor, ClaimType, PatchContext

        extractor = ClaimExtractor()
        context = PatchContext(
            failing_tests=["test_a"],
            diff_lines=50,
            files_changed=2,
        )
        claims = extractor.extract(context)
        types = {c.type for c in claims}
        assert ClaimType.SCOPE_MINIMALITY in types

    def test_max_claims_enforced(self):
        """Claim count is capped."""
        from rfsn_controller.qa import ClaimExtractor, PatchContext

        extractor = ClaimExtractor(max_claims=3)
        context = PatchContext(
            failing_tests=["test"],
            diff_lines=10,
            files_changed=1,
            issue_text="Fix the bug",
        )
        claims = extractor.extract(context)
        assert len(claims) <= 3


class TestQACritic:
    """Tests for QA critic."""

    def test_rule_based_fallback(self):
        """Critic uses rules when no LLM."""
        from rfsn_controller.qa import Claim, ClaimType, QACritic, Verdict

        critic = QACritic(llm_call=None)
        claims = [Claim("C1", ClaimType.FUNCTIONAL_FIX, "Fix tests")]
        verdicts = critic.critique(claims)

        assert len(verdicts) == 1
        assert verdicts[0].verdict == Verdict.CHALLENGE  # Needs evidence

    def test_scope_minimality_accept(self):
        """Small patch gets ACCEPT for scope."""
        from rfsn_controller.qa import Claim, ClaimType, QACritic, Verdict

        critic = QACritic()
        claims = [Claim("C1", ClaimType.SCOPE_MINIMALITY, "Minimal change")]
        verdicts = critic.critique(
            claims,
            patch_summary={"lines_changed": 30, "files_changed": 1},
        )
        assert verdicts[0].verdict == Verdict.ACCEPT

    def test_scope_minimality_reject(self):
        """Large patch gets REJECT for scope."""
        from rfsn_controller.qa import Claim, ClaimType, QACritic, Verdict

        critic = QACritic()
        claims = [Claim("C1", ClaimType.SCOPE_MINIMALITY, "Minimal change")]
        verdicts = critic.critique(
            claims,
            patch_summary={"lines_changed": 500, "files_changed": 10},
        )
        assert verdicts[0].verdict == Verdict.REJECT


class TestQAGate:
    """Tests for QA gate decisions."""

    def test_accept_all_passed(self):
        """Accept when all claims validated."""
        from rfsn_controller.qa import Claim, ClaimType, ClaimVerdict, QAGate, Verdict

        gate = QAGate()
        claims = [
            Claim("C1", ClaimType.FUNCTIONAL_FIX, "Fix"),
            Claim("C2", ClaimType.NO_REGRESSION, "No breaks"),
        ]
        verdicts = [
            ClaimVerdict("C1", Verdict.ACCEPT, "Pass"),
            ClaimVerdict("C2", Verdict.ACCEPT, "Pass"),
        ]
        decision = gate.decide(claims, verdicts)
        assert decision.accepted

    def test_reject_functional_failure(self):
        """Reject when functional_fix fails."""
        from rfsn_controller.qa import Claim, ClaimType, ClaimVerdict, QAGate, Verdict

        gate = QAGate()
        claims = [Claim("C1", ClaimType.FUNCTIONAL_FIX, "Fix")]
        verdicts = [ClaimVerdict("C1", Verdict.REJECT, "Tests fail")]
        decision = gate.decide(claims, verdicts)
        assert not decision.accepted
        assert "functional_fix rejected" in decision.rejection_reasons[0]

    def test_reject_regression(self):
        """Reject when no_regression fails."""
        from rfsn_controller.qa import Claim, ClaimType, ClaimVerdict, QAGate, Verdict

        gate = QAGate()
        claims = [Claim("C1", ClaimType.NO_REGRESSION, "No breaks")]
        verdicts = [ClaimVerdict("C1", Verdict.REJECT, "Regressions found")]
        decision = gate.decide(claims, verdicts)
        assert not decision.accepted

    def test_escalate_scope(self):
        """Escalate rather than reject for scope issues."""
        from rfsn_controller.qa import Claim, ClaimType, ClaimVerdict, QAGate, Verdict

        gate = QAGate()
        claims = [
            Claim("C1", ClaimType.FUNCTIONAL_FIX, "Fix"),
            Claim("C2", ClaimType.SCOPE_MINIMALITY, "Small"),
        ]
        verdicts = [
            ClaimVerdict("C1", Verdict.ACCEPT, "Pass"),
            ClaimVerdict("C2", Verdict.REJECT, "Too big"),
        ]
        decision = gate.decide(claims, verdicts)
        assert decision.accepted  # Still accepted
        assert "scope_minimality_escalation" in decision.escalation_tags


class TestQAPersistence:
    """Tests for claim outcome persistence."""

    def test_record_and_query(self):
        """Record attempt and query patterns."""
        from rfsn_controller.qa import (
            Claim,
            ClaimType,
            ClaimVerdict,
            QAAttempt,
            QAPersistence,
            Verdict,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "qa.db")
            store = QAPersistence(db_path)

            attempt = QAAttempt(
                attempt_id="test-1",
                claims=[Claim("C1", ClaimType.FUNCTIONAL_FIX, "Fix")],
                verdicts=[ClaimVerdict("C1", Verdict.REJECT, "Failed")],
            )
            store.record_attempt(attempt, failure_signature="AssertionError")

            patterns = store.query_failure_patterns("AssertionError", min_rejections=1)
            assert len(patterns) == 1
            assert patterns[0][0] == "functional_fix"
            store.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
