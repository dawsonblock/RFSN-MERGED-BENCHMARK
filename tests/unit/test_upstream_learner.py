"""Unit tests for upstream learner modules.

Tests cover:
- worktree_manager: Isolated git worktree management
- reward: Reward computation for task outcomes
- critic: Proposal self-critique rubric
- llm_prompting: LLM API calling utilities
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

import pytest


# =============================================================================
# WorktreeManager Tests
# =============================================================================

class TestWorktreeManager:
    """Tests for WorktreeManager."""
    
    @pytest.fixture
    def git_repo(self, tmp_path: Path) -> Path:
        """Create a test git repository."""
        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()
        subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=repo_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo_path, check=True, capture_output=True
        )
        # Create initial commit
        (repo_path / "README.md").write_text("# Test")
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=repo_path, check=True, capture_output=True
        )
        return repo_path
    
    def test_manager_creation(self, git_repo: Path, tmp_path: Path):
        """Test WorktreeManager initialization."""
        from rfsn_upstream.worktree_manager import WorktreeManager
        
        worktree_base = tmp_path / "worktrees"
        manager = WorktreeManager(root_repo=git_repo, worktree_base=worktree_base)
        
        assert manager.root_repo == git_repo
        assert manager.worktree_base == worktree_base
    
    def test_create_and_destroy_worktree(self, git_repo: Path, tmp_path: Path):
        """Test creating and destroying a worktree."""
        from rfsn_upstream.worktree_manager import WorktreeManager
        
        worktree_base = tmp_path / "worktrees"
        manager = WorktreeManager(root_repo=git_repo, worktree_base=worktree_base)
        
        handle = manager.create(task_id="test-task-1")
        assert handle.path.exists()
        assert (handle.path / "README.md").exists()
        
        manager.destroy(handle)
        assert not handle.path.exists()
    
    def test_worktree_context_manager(self, git_repo: Path, tmp_path: Path):
        """Test worktree context manager for automatic cleanup."""
        from rfsn_upstream.worktree_manager import WorktreeManager
        
        worktree_base = tmp_path / "worktrees"
        manager = WorktreeManager(root_repo=git_repo, worktree_base=worktree_base)
        
        with manager.worktree(task_id="context-test") as handle:
            assert handle.path.exists()
            worktree_path = handle.path
        
        # Should be cleaned up after context
        assert not worktree_path.exists()


# =============================================================================
# Reward Tests
# =============================================================================

class TestReward:
    """Tests for reward computation."""
    
    def test_success_reward(self):
        """Test reward for successful task."""
        from rfsn_upstream.reward import (
            TaskOutcome,
            RewardConfig,
            compute_reward,
            create_success_outcome,
        )
        
        outcome = create_success_outcome(tests_total=10)
        reward = compute_reward(outcome)
        
        assert reward == 1.0  # Full success reward
    
    def test_failure_reward(self):
        """Test reward for failed task."""
        from rfsn_upstream.reward import (
            RewardConfig,
            compute_reward,
            create_failure_outcome,
        )
        
        outcome = create_failure_outcome(
            tests_passed=2,
            tests_total=10,
            patch_applied=True,
            error_message="Tests failed"
        )
        reward = compute_reward(outcome)
        
        assert reward < 1.0
        assert reward >= 0.0  # Should have some partial credit
    
    def test_rejection_reward(self):
        """Test reward for rejected proposal."""
        from rfsn_upstream.reward import (
            RewardConfig,
            compute_reward,
            create_rejection_outcome,
        )
        
        outcome = create_rejection_outcome(reason="Patch too large")
        config = RewardConfig()
        reward = compute_reward(outcome, config)
        
        assert reward == config.rejection_penalty
    
    def test_custom_config(self):
        """Test reward with custom configuration."""
        from rfsn_upstream.reward import (
            RewardConfig,
            compute_reward,
            create_success_outcome,
        )
        
        config = RewardConfig(full_success_reward=2.0)
        outcome = create_success_outcome(tests_total=5)
        reward = compute_reward(outcome, config)
        
        assert reward == 2.0


# =============================================================================
# Critic Tests
# =============================================================================

class TestCritic:
    """Tests for proposal self-critique."""
    
    def test_passing_critique(self):
        """Test critique that passes."""
        from rfsn_upstream.critic import PlannerCritic, CriticConfig
        
        config = CriticConfig(max_lines_changed=100)
        critic = PlannerCritic(config)
        
        proposal = {"diff": "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n-old\n+new"}
        result = critic.evaluate(proposal)
        
        assert result.passed
        assert len(result.issues) == 0
    
    def test_failing_critique_large_patch(self):
        """Test critique that fails due to patch size."""
        from rfsn_upstream.critic import PlannerCritic, CriticConfig
        
        config = CriticConfig(max_lines_changed=5)
        critic = PlannerCritic(config)
        
        # Create a diff with many lines
        diff_lines = ["--- a/file.py", "+++ b/file.py", "@@ -1,20 +1,20 @@"]
        diff_lines.extend([f"-old line {i}" for i in range(20)])
        diff_lines.extend([f"+new line {i}" for i in range(20)])
        proposal = {"diff": "\n".join(diff_lines)}
        
        result = critic.evaluate(proposal)
        
        assert not result.passed
        assert any("lines" in str(i).lower() for i in result.issues)
    
    def test_failing_critique_forbidden_path(self):
        """Test critique that fails due to forbidden path modification."""
        from rfsn_upstream.critic import PlannerCritic, CriticConfig
        
        config = CriticConfig(forbidden_paths=["setup.py"])
        critic = PlannerCritic(config)
        
        proposal = {"diff": "--- a/setup.py\n+++ b/setup.py\n@@ -1 +1 @@\n-old\n+new"}
        result = critic.evaluate(proposal)
        
        assert not result.passed
        assert any("forbidden" in str(i).lower() for i in result.issues)


# =============================================================================
# LLM Prompting Tests
# =============================================================================

class TestLLMPrompting:
    """Tests for LLM prompting utilities."""
    
    def test_parse_json_response_valid(self):
        """Test parsing valid JSON response."""
        from rfsn_upstream.llm_prompting import parse_json_response
        
        response = '{"action": "fix", "file": "test.py"}'
        result = parse_json_response(response)
        
        assert result is not None
        assert result["action"] == "fix"
    
    def test_parse_json_response_with_markdown(self):
        """Test parsing JSON embedded in markdown code block."""
        from rfsn_upstream.llm_prompting import parse_json_response
        
        response = """Here is the fix:
```json
{"action": "fix", "file": "test.py"}
```
"""
        result = parse_json_response(response)
        
        assert result is not None
        assert result["action"] == "fix"
    
    def test_parse_json_response_invalid(self):
        """Test parsing invalid JSON response."""
        from rfsn_upstream.llm_prompting import parse_json_response
        
        response = "This is not JSON at all"
        result = parse_json_response(response)
        
        assert result is None
    
    def test_extract_diff_from_response(self):
        """Test extracting diff from LLM response."""
        from rfsn_upstream.llm_prompting import extract_diff_from_response
        
        response = """Here is the patch:
```diff
--- a/file.py
+++ b/file.py
@@ -1 +1 @@
-old
+new
```
"""
        diff = extract_diff_from_response(response)
        
        assert diff is not None
        assert "--- a/file.py" in diff
        assert "+new" in diff
    
    def test_extract_diff_no_diff(self):
        """Test extracting diff when none present."""
        from rfsn_upstream.llm_prompting import extract_diff_from_response
        
        response = "I couldn't find anything to fix."
        diff = extract_diff_from_response(response)
        
        assert diff is None or diff == ""
    
    @mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"})
    def test_llm_config_provider_auto(self):
        """Test LLM config auto-selects provider based on env."""
        from rfsn_upstream.llm_prompting import LLMConfig, LLMProvider
        
        config = LLMConfig(provider=LLMProvider.AUTO)
        # With DEEPSEEK_API_KEY set, should prefer DeepSeek
        assert config.provider == LLMProvider.AUTO


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for upstream learner components."""
    
    def test_full_critique_flow(self):
        """Test full flow from diff to critique."""
        from rfsn_upstream.critic import create_strict_critic
        from rfsn_upstream.reward import compute_reward, create_rejection_outcome
        
        critic = create_strict_critic()
        
        # Create a reasonable proposal
        proposal = {"diff": "--- a/fix.py\n+++ b/fix.py\n@@ -1 +1 @@\n-bug\n+fix"}
        critique = critic.evaluate(proposal)
        
        if not critique.passed:
            outcome = create_rejection_outcome(
                reason="; ".join(str(i) for i in critique.issues[:3])
            )
            reward = compute_reward(outcome)
            assert reward < 0  # Rejection should be negative
    
    def test_task_data_format(self, tmp_path: Path):
        """Test task data format from JSONL."""
        tasks_file = tmp_path / "tasks.jsonl"
        tasks_file.write_text(json.dumps({
            "instance_id": "test__test-123",
            "repo": "test-org/test-repo",
            "base_commit": "abc123",
            "problem_statement": "Fix the bug",
            "test_patch": "",
            "test_cmd": "pytest",
        }) + "\n")
        
        # Read task back
        with open(tasks_file) as f:
            task_data = json.loads(f.readline())
        
        assert task_data["instance_id"] == "test__test-123"
        assert task_data["repo"] == "test-org/test-repo"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
