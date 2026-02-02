"""Tests for beam_search module."""

from __future__ import annotations

from typing import Any

import pytest

from rfsn_controller.planner_v2.beam_search import (
    BeamSearchConfig,
    BeamSearcher,
    BeamSearchResult,
    Candidate,
    create_beam_searcher,
    score_test_result,
)


class TestCandidate:
    """Tests for Candidate dataclass."""
    
    def test_is_success_high_score(self) -> None:
        """Test is_success returns True for high score."""
        candidate = Candidate(
            candidate_id="c001",
            patch_diff="test patch",
            score=0.98,
            depth=1,
        )
        assert candidate.is_success is True
    
    def test_is_success_low_score(self) -> None:
        """Test is_success returns False for low score."""
        candidate = Candidate(
            candidate_id="c001",
            patch_diff="test patch",
            score=0.5,
            depth=1,
        )
        assert candidate.is_success is False
    
    def test_is_success_threshold(self) -> None:
        """Test is_success at exact threshold."""
        candidate = Candidate(
            candidate_id="c001",
            patch_diff="test patch",
            score=0.95,
            depth=1,
        )
        assert candidate.is_success is True


class TestBeamSearchConfig:
    """Tests for BeamSearchConfig."""
    
    def test_defaults(self) -> None:
        """Test default configuration values."""
        config = BeamSearchConfig()
        
        assert config.beam_width == 3
        assert config.max_depth == 5
        assert config.score_threshold == 0.95
        assert config.timeout_seconds == 300.0
    
    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = BeamSearchConfig(
            beam_width=5,
            max_depth=10,
            score_threshold=0.9,
        )
        
        assert config.beam_width == 5
        assert config.max_depth == 10
        assert config.score_threshold == 0.9


class TestScoreTestResult:
    """Tests for score_test_result function."""
    
    def test_all_pass(self) -> None:
        """Test scoring when all tests pass."""
        score = score_test_result({"all_pass": True})
        assert score == 1.0
    
    def test_pass_rate(self) -> None:
        """Test scoring with pass rate."""
        score = score_test_result({"pass_rate": 0.5})
        assert score == 0.4  # 0.5 * 0.8
    
    def test_passed_total(self) -> None:
        """Test scoring with passed/total counts."""
        score = score_test_result({"passed": 8, "total": 10})
        assert score == pytest.approx(0.64)  # 0.8 * 0.8
    
    def test_regression_penalty(self) -> None:
        """Test penalty for new failures."""
        score = score_test_result({"all_pass": True, "new_failures": 2})
        assert score == pytest.approx(0.7)  # 1.0 - 0.15*2
    
    def test_diff_penalty(self) -> None:
        """Test penalty for large diffs."""
        score = score_test_result({"all_pass": True}, diff_lines=100)
        assert score == pytest.approx(0.98)  # 1.0 - 0.02*1
    
    def test_focused_test_bonus(self) -> None:
        """Test bonus for focused test passing."""
        score = score_test_result({"pass_rate": 0.5, "focused_test_pass": True})
        assert score == pytest.approx(0.5)  # 0.4 + 0.1
    
    def test_score_clamped(self) -> None:
        """Test score is clamped to [0, 1]."""
        # Huge penalty should clamp to 0
        score = score_test_result({"pass_rate": 0.1, "new_failures": 10})
        assert score == 0.0


class TestCreateBeamSearcher:
    """Tests for create_beam_searcher factory function."""
    
    def test_default_config(self) -> None:
        """Test creating searcher with defaults."""
        searcher = create_beam_searcher()
        
        assert searcher.config.beam_width == 3
        assert searcher.config.max_depth == 5
    
    def test_custom_config(self) -> None:
        """Test creating searcher with custom config."""
        searcher = create_beam_searcher(
            beam_width=5,
            max_depth=3,
            score_threshold=0.9,
            timeout_seconds=60.0,
        )
        
        assert searcher.config.beam_width == 5
        assert searcher.config.max_depth == 3
        assert searcher.config.score_threshold == 0.9
        assert searcher.config.timeout_seconds == 60.0


class TestBeamSearcher:
    """Tests for BeamSearcher class."""
    
    def test_generate_candidate_id(self) -> None:
        """Test candidate ID generation."""
        searcher = create_beam_searcher()
        
        id1 = searcher._generate_candidate_id()
        id2 = searcher._generate_candidate_id()
        
        assert id1 == "c0001"
        assert id2 == "c0002"
    
    def test_combine_patches_both_empty(self) -> None:
        """Test combining empty patches."""
        searcher = create_beam_searcher()
        result = searcher._combine_patches("", "")
        assert result == ""
    
    def test_combine_patches_parent_only(self) -> None:
        """Test combining with only parent patch."""
        searcher = create_beam_searcher()
        result = searcher._combine_patches("parent patch", "")
        assert result == "parent patch"
    
    def test_combine_patches_child_only(self) -> None:
        """Test combining with only child patch."""
        searcher = create_beam_searcher()
        result = searcher._combine_patches("", "child patch")
        assert result == "child patch"
    
    def test_combine_patches_both(self) -> None:
        """Test combining both patches."""
        searcher = create_beam_searcher()
        result = searcher._combine_patches("parent", "child")
        assert result == "parent\nchild"


class TestBeamSearchResult:
    """Tests for BeamSearchResult dataclass."""
    
    def test_success_result(self) -> None:
        """Test successful result."""
        candidate = Candidate(
            candidate_id="c001",
            patch_diff="fix",
            score=0.98,
            depth=2,
        )
        result = BeamSearchResult(
            success=True,
            best_candidate=candidate,
            all_candidates=[candidate],
            search_stats={"total_candidates": 1},
        )
        
        assert result.success is True
        assert result.best_candidate is not None
        assert result.best_candidate.score == 0.98
    
    def test_failure_result(self) -> None:
        """Test failed result."""
        result = BeamSearchResult(
            success=False,
            best_candidate=None,
            all_candidates=[],
            search_stats={"total_candidates": 0},
        )
        
        assert result.success is False
        assert result.best_candidate is None
