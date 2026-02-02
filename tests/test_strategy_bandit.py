"""Unit tests for strategy bandit and negative memory."""

import os
import tempfile

import pytest


class TestStrategyBandit:
    """Tests for Thompson Sampling bandit."""

    def test_select_returns_valid_strategy(self):
        """Selection returns a known strategy."""
        from rfsn_controller.strategy_bandit import StrategyBandit

        bandit = StrategyBandit()
        selected = bandit.select_strategy()
        assert selected in bandit.arms

    def test_update_increases_pulls(self):
        """Updates increment pull count."""
        from rfsn_controller.strategy_bandit import StrategyBandit

        bandit = StrategyBandit()
        strategy = "temp_0.3"
        initial_pulls = bandit.arms[strategy].pulls

        bandit.update(strategy, reward=1.0)
        assert bandit.arms[strategy].pulls == initial_pulls + 1

    def test_success_increases_alpha(self):
        """Success reward increases alpha."""
        from rfsn_controller.strategy_bandit import StrategyBandit

        bandit = StrategyBandit()
        strategy = "temp_0.7"
        initial_alpha = bandit.arms[strategy].alpha

        bandit.update(strategy, reward=1.0)
        assert bandit.arms[strategy].alpha > initial_alpha

    def test_failure_increases_beta(self):
        """Failure reward increases beta."""
        from rfsn_controller.strategy_bandit import StrategyBandit

        bandit = StrategyBandit()
        strategy = "temp_0.0"
        initial_beta = bandit.arms[strategy].beta

        bandit.update(strategy, reward=0.0)
        assert bandit.arms[strategy].beta > initial_beta

    def test_exclude_strategies(self):
        """Excluded strategies not selected."""
        from rfsn_controller.strategy_bandit import StrategyBandit

        bandit = StrategyBandit(strategies=["a", "b", "c"])

        # Run many selections with a, b excluded
        for _ in range(20):
            selected = bandit.select_strategy(exclude={"a", "b"})
            assert selected == "c"


class TestNegativeMemoryStore:
    """Tests for failure tracking store."""

    def test_record_and_retrieve(self):
        """Record a failure and retrieve it."""
        from rfsn_controller.strategy_bandit import (
            FailureFeatures,
            NegativeMemoryStore,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store = NegativeMemoryStore(os.path.join(tmpdir, "neg.db"))

            features = FailureFeatures(
                error_class="AssertionError",
                stack_signature="abc123",
                touched_files=["file.py"],
                test_file="test_file.py",
                error_message_prefix="Expected True",
            )

            store.record_failure(
                features=features,
                strategy="temp_0.7",
                patch_hash="patch123",
                timestamp=100,
            )

            failed = store.get_failed_strategies("AssertionError", min_failures=1)
            assert any(name == "temp_0.7" for name, _ in failed)
            store.close()

    def test_avoidance_threshold(self):
        """Strategy avoided after threshold failures."""
        from rfsn_controller.strategy_bandit import (
            FailureFeatures,
            NegativeMemoryStore,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store = NegativeMemoryStore(os.path.join(tmpdir, "neg.db"))

            features = FailureFeatures(
                error_class="TypeError",
                stack_signature="xyz",
                touched_files=[],
                test_file=None,
                error_message_prefix="",
            )

            # Record 3 failures
            for i in range(3):
                store.record_failure(
                    features=features,
                    strategy="bad_strategy",
                    patch_hash=f"patch{i}",
                    timestamp=i,
                )

            assert store.should_avoid_strategy("TypeError", "bad_strategy", threshold=3)
            assert not store.should_avoid_strategy("TypeError", "good_strategy", threshold=3)
            store.close()


class TestFailureFeatureExtraction:
    """Tests for failure feature extraction."""

    def test_extracts_error_class(self):
        """Extracts Python error class from output."""
        from rfsn_controller.strategy_bandit import extract_failure_features

        features = extract_failure_features(
            stderr="AssertionError: expected 1 but got 2",
            stdout="",
            patch_diff="",
        )
        assert features.error_class == "AssertionError"

    def test_extracts_touched_files(self):
        """Extracts modified files from patch."""
        from rfsn_controller.strategy_bandit import extract_failure_features

        features = extract_failure_features(
            stderr="",
            stdout="",
            patch_diff="""--- a/src/foo.py
+++ b/src/foo.py
@@ -1 +1 @@
-x=1
+x=2
""",
        )
        assert "src/foo.py" in features.touched_files


class TestLearningOrchestrator:
    """Tests for combined bandit + negative memory."""

    def test_select_avoids_bad_strategies(self):
        """Orchestrator avoids strategies with negative history."""
        from rfsn_controller.strategy_bandit import (
            FailureFeatures,
            LearningOrchestrator,
            NegativeMemoryStore,
            StrategyBandit,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            store = NegativeMemoryStore(os.path.join(tmpdir, "neg.db"))
            bandit = StrategyBandit(strategies=["good", "bad", "ugly"])

            # Record failures for "bad" strategy
            features = FailureFeatures(
                error_class="KeyError",
                stack_signature="",
                touched_files=[],
                test_file=None,
                error_message_prefix="",
            )
            for i in range(5):
                store.record_failure(features, "bad", f"patch{i}", i)

            orchestrator = LearningOrchestrator(
                bandit=bandit,
                negative_store=store,
                avoidance_threshold=3,
            )

            # Selection should avoid "bad" for KeyError
            for _ in range(10):
                selected = orchestrator.select_strategy(error_class="KeyError")
                assert selected in ("good", "ugly")

            store.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
