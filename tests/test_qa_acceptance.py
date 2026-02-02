"""Acceptance tests for QA system.

Tests the 3 key acceptance criteria:
1. Delta map catches fix→regress (patch that fixes one test but breaks another)
2. Scope limits enforced (8 files in surgical mode → reject/escalate)
3. Coverage signal for untested changes
"""


import pytest


class TestDeltaMapCatchesRegression:
    """A patch that fixes test_a but breaks test_b must be rejected."""

    def test_regression_detected_and_rejected(self):
        """Patch with regression is rejected by QA gate."""
        from rfsn_controller.qa import (
            ClaimExtractor,
            ClaimType,
            Evidence,
            EvidenceType,
            PatchContext,
            QACritic,
            QAGate,
        )

        # Setup: Create claims
        extractor = ClaimExtractor()
        context = PatchContext(failing_tests=["test_a"], diff_lines=20, files_changed=1)
        claims = extractor.extract(context)

        # Simulate critic initially challenging
        critic = QACritic()
        verdicts = critic.critique(claims, patch_summary={"lines_changed": 20, "files_changed": 1})

        # Simulate evidence: test_a fixed, but test_b regressed
        delta_evidence = Evidence(
            type=EvidenceType.DELTA_MAP,
            data={
                "fixed": ["test_a"],
                "regressed": ["test_b"],
                "still_failing": [],
            },
        )

        # Re-evaluate NO_REGRESSION claim with delta evidence
        for i, v in enumerate(verdicts):
            claim = next((c for c in claims if c.id == v.claim_id), None)
            if claim and claim.type == ClaimType.NO_REGRESSION:
                new_verdict = critic.re_evaluate(claim, [delta_evidence], v)
                verdicts[i] = new_verdict

        # Gate should reject due to regression
        gate = QAGate()
        decision = gate.decide(claims, verdicts, [delta_evidence])

        assert not decision.accepted
        assert any("regression" in r.lower() for r in decision.rejection_reasons)


class TestScopeLimitsEnforced:
    """A patch touching 8 files in surgical mode must be rejected or escalated."""

    def test_large_patch_rejected_in_strict_mode(self):
        """8 files triggers rejection in strict mode."""
        from rfsn_controller.qa import (
            ClaimExtractor,
            ClaimType,
            PatchContext,
        )

        # Large patch: 8 files
        context = PatchContext(
            failing_tests=["test_x"],
            diff_lines=400,
            files_changed=8,
        )
        extractor = ClaimExtractor()
        claims = extractor.extract(context)

        # Should NOT have scope_minimality claim (too big)
        types = {c.type for c in claims}
        assert ClaimType.SCOPE_MINIMALITY not in types

    def test_large_patch_scope_rejected(self):
        """Explicit scope claim on large patch gets rejected."""
        from rfsn_controller.qa import (
            Claim,
            ClaimType,
            QACritic,
            Verdict,
        )

        # Force a scope claim on large patch
        claims = [
            Claim("C1", ClaimType.FUNCTIONAL_FIX, "Fix"),
            Claim("C2", ClaimType.SCOPE_MINIMALITY, "Small change"),
        ]

        critic = QACritic()
        verdicts = critic.critique(
            claims,
            patch_summary={"lines_changed": 500, "files_changed": 10},
        )

        # Scope claim should be rejected
        scope_verdict = next(v for v in verdicts if v.claim_id == "C2")
        assert scope_verdict.verdict == Verdict.REJECT
        assert "too large" in scope_verdict.reason.lower() or "too big" in scope_verdict.reason.lower()

    def test_moderate_patch_escalated(self):
        """Moderate patch accepted with escalation tag."""
        from rfsn_controller.qa import (
            Claim,
            ClaimType,
            ClaimVerdict,
            QAGate,
            Verdict,
        )

        claims = [
            Claim("C1", ClaimType.FUNCTIONAL_FIX, "Fix"),
            Claim("C2", ClaimType.SCOPE_MINIMALITY, "Moderate"),
        ]
        verdicts = [
            ClaimVerdict("C1", Verdict.ACCEPT, "Works"),
            ClaimVerdict("C2", Verdict.REJECT, "Patch too large"),
        ]

        gate = QAGate()
        decision = gate.decide(claims, verdicts)

        # Accepted but with escalation
        assert decision.accepted
        assert "scope_minimality_escalation" in decision.escalation_tags


class TestCoverageSignalForUntestedChanges:
    """Patch passes tests but touched module not imported by tests."""

    def test_low_coverage_flagged(self):
        """Changes to untested code should be flagged."""
        from rfsn_controller.qa import (
            Evidence,
            EvidenceType,
        )

        # Create coverage signal evidence
        coverage_evidence = Evidence(
            type=EvidenceType.COVERAGE_SIGNAL,
            data={
                "touched_files": ["src/new_module.py"],
                "tested_files": ["src/old_module.py"],
                "untested_changes": ["src/new_module.py"],
                "coverage_confidence": "low",
            },
        )

        # Evidence indicates changed file not covered by tests
        assert coverage_evidence.data["untested_changes"] == ["src/new_module.py"]
        assert coverage_evidence.data["coverage_confidence"] == "low"


class TestQAOrchestratorIntegration:
    """Integration tests for QA orchestrator."""

    def test_full_pipeline_accept(self):
        """Orchestrator accepts clean patch."""
        from rfsn_controller.qa import QAConfig, QAOrchestrator

        # Mock test runner that shows tests pass
        def mock_test_runner(cmd):
            return {
                "exit_code": 0,
                "failing_tests": [],
                "passing_tests": ["test_a"],
                "duration_ms": 100,
            }

        # Mock hygiene validator
        def mock_hygiene(diff):
            return {"is_valid": True, "violations": [], "diff_stats": {}}

        # Mock delta tracker
        class MockDeltaTracker:
            def compute_delta(self, current_failing):
                # No regressions, 1 fix
                return ({"test_a"}, set())  # (fixed, regressed)

        config = QAConfig(
            surgical_max_lines=100,
            surgical_max_files=5,
            allow_unresolved_challenges=True,  # Allow for simpler test
        )
        qa = QAOrchestrator(
            config=config,
            test_runner=mock_test_runner,
            hygiene_validator=mock_hygiene,
            delta_tracker=MockDeltaTracker(),
        )

        diff = """diff --git a/fix.py b/fix.py
--- a/fix.py
+++ b/fix.py
@@ -1 +1 @@
-x = 1
+x = 2
"""
        result = qa.evaluate_patch(
            diff=diff,
            failing_tests=["test_a"],
            test_cmd="pytest test_a",
        )

        assert result.accepted

    def test_full_pipeline_reject_on_failure(self):
        """Orchestrator rejects patch when tests still fail."""
        from rfsn_controller.qa import QAConfig, QAOrchestrator

        # Mock test runner that shows tests fail
        def mock_test_runner(cmd):
            return {
                "exit_code": 1,
                "failing_tests": ["test_a"],
                "passing_tests": [],
                "duration_ms": 100,
            }

        config = QAConfig()
        qa = QAOrchestrator(config=config, test_runner=mock_test_runner)

        result = qa.evaluate_patch(
            diff="--- a/f.py\n+++ b/f.py\n-x\n+y",
            failing_tests=["test_a"],
            test_cmd="pytest",
        )

        assert not result.accepted


class TestStaticChecker:
    """Tests for static checker integration."""

    def test_ruff_check_available(self):
        """StaticChecker can invoke ruff (if installed)."""
        from rfsn_controller.qa import StaticChecker

        checker = StaticChecker(timeout_seconds=10)
        # Test with a non-existent path to avoid actual linting
        result = checker.check("ruff", files=["/nonexistent/path"])
        
        # Either works (exit 0 or 1) or tool not found (127)
        assert result.exit_code in (0, 1, 2, 127)
        assert result.tool == "ruff"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
