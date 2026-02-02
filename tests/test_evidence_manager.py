"""Tests for EvidenceManager module."""

import json
import tempfile
from pathlib import Path

import pytest

from rfsn_controller.evidence_manager import EvidenceEntry, EvidenceManager


class TestEvidenceManager:
    """Test suite for EvidenceManager class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for evidence output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_dir):
        """Create an EvidenceManager instance."""
        return EvidenceManager(output_dir=temp_dir, run_id="test-run-123")

    def test_init(self, temp_dir):
        """Test initialization creates necessary directories."""
        manager = EvidenceManager(output_dir=temp_dir, run_id="test-run-456")
        
        assert manager.output_dir == temp_dir
        assert manager.run_id == "test-run-456"
        assert manager.run_dir.exists()
        assert manager.run_dir == temp_dir / "test-run-456"
        assert len(manager.evidence_log) == 0

    def test_init_without_run_id(self, temp_dir):
        """Test initialization without explicit run_id generates one."""
        manager = EvidenceManager(output_dir=temp_dir)
        
        assert manager.run_id.startswith("run-")
        assert manager.run_dir.exists()

    def test_add_evidence(self, manager):
        """Test adding evidence entries."""
        manager.add_evidence(
            "test_event",
            {"key": "value", "count": 42},
            phase="testing",
            patch_id=1
        )
        
        assert len(manager.evidence_log) == 1
        entry = manager.evidence_log[0]
        assert entry.type == "test_event"
        assert entry.data["key"] == "value"
        assert entry.data["count"] == 42
        assert entry.phase == "testing"
        assert entry.patch_id == 1
        assert entry.timestamp > 0

    def test_add_evidence_persisted(self, manager):
        """Test that evidence is persisted to file."""
        manager.add_evidence(
            "persisted_event",
            {"test": "data"},
            phase="test"
        )
        
        evidence_file = manager.run_dir / "evidence.jsonl"
        assert evidence_file.exists()
        
        # Read and verify content
        with open(evidence_file) as f:
            line = f.readline()
            data = json.loads(line)
            assert data["type"] == "persisted_event"
            assert data["data"]["test"] == "data"
            assert data["phase"] == "test"

    def test_add_llm_call(self, manager):
        """Test adding LLM call evidence."""
        manager.add_llm_call(
            provider="deepseek",
            model="deepseek-chat",
            prompt="Fix the bug",
            response="Apply this patch...",
            tokens=150,
            latency_ms=1234.5,
            cost_usd=0.01,
            phase="patching"
        )
        
        assert len(manager.evidence_log) == 1
        entry = manager.evidence_log[0]
        assert entry.type == "llm_call"
        assert entry.data["provider"] == "deepseek"
        assert entry.data["model"] == "deepseek-chat"
        assert entry.data["tokens"] == 150
        assert entry.data["latency_ms"] == 1234.5
        assert entry.data["cost_usd"] == 0.01
        assert entry.phase == "patching"

    def test_add_llm_call_truncates_long_content(self, manager):
        """Test that long prompts and responses are truncated."""
        long_prompt = "x" * 2000
        long_response = "y" * 3000
        
        manager.add_llm_call(
            provider="test",
            model="test-model",
            prompt=long_prompt,
            response=long_response,
            tokens=100,
            latency_ms=1000
        )
        
        entry = manager.evidence_log[0]
        assert len(entry.data["prompt"]) == 1000  # Truncated
        assert len(entry.data["response"]) == 2000  # Truncated

    def test_add_patch_result(self, manager):
        """Test adding patch result evidence."""
        manager.add_patch_result(
            patch_id=42,
            content="def fix():\n    pass",
            test_passed=True,
            test_output="All tests passed!",
            phase="verification"
        )
        
        assert len(manager.evidence_log) == 1
        entry = manager.evidence_log[0]
        assert entry.type == "patch_result"
        assert entry.patch_id == 42
        assert entry.data["content"] == "def fix():\n    pass"
        assert entry.data["test_passed"] is True
        assert entry.data["test_output"] == "All tests passed!"
        assert entry.phase == "verification"

    def test_add_decision(self, manager):
        """Test adding decision evidence."""
        manager.add_decision(
            decision_type="strategy_selection",
            rationale="Chose greedy approach for speed",
            alternatives=["greedy", "optimal", "heuristic"],
            selected="greedy",
            phase="planning"
        )
        
        assert len(manager.evidence_log) == 1
        entry = manager.evidence_log[0]
        assert entry.type == "decision"
        assert entry.data["decision_type"] == "strategy_selection"
        assert entry.data["selected"] == "greedy"
        assert len(entry.data["alternatives"]) == 3

    def test_add_tool_execution(self, manager):
        """Test adding tool execution evidence."""
        manager.add_tool_execution(
            tool_name="git_diff",
            args={"file": "main.py"},
            result="+ added line\n- removed line",
            success=True,
            phase="localization"
        )
        
        assert len(manager.evidence_log) == 1
        entry = manager.evidence_log[0]
        assert entry.type == "tool_execution"
        assert entry.data["tool_name"] == "git_diff"
        assert entry.data["args"]["file"] == "main.py"
        assert entry.data["success"] is True

    def test_export_summary(self, manager):
        """Test exporting evidence summary."""
        # Add multiple evidence entries
        manager.add_evidence("event1", {"data": 1}, phase="phase1")
        manager.add_evidence("event2", {"data": 2}, phase="phase1")
        manager.add_evidence("event1", {"data": 3}, phase="phase2")
        
        summary_path = manager.export_summary()
        
        assert Path(summary_path).exists()
        assert "summary.json" in summary_path
        
        # Verify summary content
        with open(summary_path) as f:
            summary = json.load(f)
            
        assert summary["run_id"] == "test-run-123"
        assert summary["total_events"] == 3
        assert summary["events_by_type"]["event1"] == 2
        assert summary["events_by_type"]["event2"] == 1
        assert summary["events_by_phase"]["phase1"] == 2
        assert summary["events_by_phase"]["phase2"] == 1
        assert len(summary["evidence"]) == 3

    def test_get_stats(self, manager):
        """Test getting evidence statistics."""
        manager.add_evidence("type_a", {}, phase="phase1")
        manager.add_evidence("type_a", {}, phase="phase2")
        manager.add_evidence("type_b", {}, phase="phase1")
        
        stats = manager.get_stats()
        
        assert stats["run_id"] == "test-run-123"
        assert stats["total_events"] == 3
        assert stats["events_by_type"]["type_a"] == 2
        assert stats["events_by_type"]["type_b"] == 1
        assert stats["events_by_phase"]["phase1"] == 2
        assert stats["events_by_phase"]["phase2"] == 1

    def test_multiple_evidence_entries(self, manager):
        """Test collecting multiple evidence entries."""
        # Add various types of evidence
        manager.add_llm_call("provider1", "model1", "prompt1", "response1", 100, 500)
        manager.add_patch_result(1, "patch1", True, "output1")
        manager.add_decision("type1", "rationale1", ["a", "b"], "a")
        manager.add_tool_execution("tool1", {}, "result1", True)
        
        assert len(manager.evidence_log) == 4
        
        # Verify different types
        types = [e.type for e in manager.evidence_log]
        assert "llm_call" in types
        assert "patch_result" in types
        assert "decision" in types
        assert "tool_execution" in types


class TestEvidenceEntry:
    """Test suite for EvidenceEntry dataclass."""

    def test_evidence_entry_creation(self):
        """Test EvidenceEntry dataclass creation."""
        entry = EvidenceEntry(
            type="test_type",
            data={"key": "value"},
            timestamp=1234567890.0,
            phase="test_phase",
            patch_id=42
        )
        
        assert entry.type == "test_type"
        assert entry.data["key"] == "value"
        assert entry.timestamp == 1234567890.0
        assert entry.phase == "test_phase"
        assert entry.patch_id == 42

    def test_evidence_entry_to_dict(self):
        """Test converting EvidenceEntry to dictionary."""
        entry = EvidenceEntry(
            type="test_type",
            data={"nested": {"key": "value"}},
            timestamp=1234567890.0
        )
        
        entry_dict = entry.to_dict()
        
        assert isinstance(entry_dict, dict)
        assert entry_dict["type"] == "test_type"
        assert entry_dict["data"]["nested"]["key"] == "value"
        assert entry_dict["timestamp"] == 1234567890.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
