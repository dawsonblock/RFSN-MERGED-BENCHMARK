"""
EvidenceManager: Handles collection and export of repair evidence.

This module provides comprehensive evidence collection for autonomous code
repairs, including LLM interactions, patch applications, test results, and
decision traces. Evidence can be exported for debugging, auditing, and
continuous improvement.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .structured_logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvidenceEntry:
    """Single piece of evidence collected during repair."""
    
    type: str
    data: dict[str, Any]
    timestamp: float
    phase: str | None = None
    patch_id: int | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class EvidenceManager:
    """
    Collects and persists evidence during the repair process.
    
    The EvidenceManager maintains a comprehensive log of all repair activities,
    including LLM calls, patch generations, test executions, and decisions made
    by the planner. This evidence can be used for:
    
    - Debugging failed repairs
    - Auditing AI decisions
    - Training and improvement
    - Compliance and accountability
    
    Example:
        >>> manager = EvidenceManager(output_dir=Path("./evidence"))
        >>> manager.add_llm_call(
        ...     provider="deepseek",
        ...     prompt="Fix the bug",
        ...     response="Apply this patch...",
        ...     tokens=150
        ... )
        >>> manager.add_patch_result(
        ...     patch_id=1,
        ...     content="...",
        ...     test_passed=True,
        ...     test_output="All tests passed"
        ... )
        >>> manager.export_summary()
        './evidence/summary.json'
    """

    def __init__(self, output_dir: Path, run_id: str | None = None):
        """
        Initialize the evidence manager.
        
        Args:
            output_dir: Directory to store evidence files
            run_id: Optional run identifier for this repair session
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or f"run-{int(time.time())}"
        self.evidence_log: list[EvidenceEntry] = []
        
        # Create run-specific subdirectory
        self.run_dir = self.output_dir / self.run_id
        self.run_dir.mkdir(exist_ok=True)
        
        logger.info("Evidence manager initialized", run_id=self.run_id, output_dir=str(self.run_dir))

    def add_evidence(
        self, 
        event_type: str, 
        data: dict[str, Any],
        phase: str | None = None,
        patch_id: int | None = None
    ):
        """
        Add a piece of evidence to the log.
        
        Args:
            event_type: Type of event (e.g., 'llm_call', 'patch_applied', 'test_result')
            data: Event data dictionary
            phase: Optional repair phase (e.g., 'reproduce', 'localize', 'patch')
            patch_id: Optional patch identifier
        """
        entry = EvidenceEntry(
            type=event_type,
            data=data,
            timestamp=time.time(),
            phase=phase,
            patch_id=patch_id
        )
        self.evidence_log.append(entry)
        self._persist_entry(entry)
        
        logger.debug("Evidence added", event_type=event_type, phase=phase, patch_id=patch_id)

    def add_llm_call(
        self,
        provider: str,
        model: str,
        prompt: str,
        response: str,
        tokens: int,
        latency_ms: float,
        cost_usd: float = 0.0,
        phase: str | None = None
    ):
        """Add evidence for an LLM API call."""
        self.add_evidence(
            "llm_call",
            {
                "provider": provider,
                "model": model,
                "prompt": prompt[:1000],  # Truncate long prompts
                "response": response[:2000],  # Truncate long responses
                "tokens": tokens,
                "latency_ms": latency_ms,
                "cost_usd": cost_usd
            },
            phase=phase
        )

    def add_patch_result(
        self,
        patch_id: int,
        content: str,
        test_passed: bool,
        test_output: str,
        phase: str | None = None
    ):
        """Add evidence for a patch application and test result."""
        self.add_evidence(
            "patch_result",
            {
                "content": content,
                "test_passed": test_passed,
                "test_output": test_output[:5000]  # Truncate long output
            },
            phase=phase,
            patch_id=patch_id
        )

    def add_decision(
        self,
        decision_type: str,
        rationale: str,
        alternatives: list[str],
        selected: str,
        phase: str | None = None
    ):
        """Add evidence for a planning or strategy decision."""
        self.add_evidence(
            "decision",
            {
                "decision_type": decision_type,
                "rationale": rationale,
                "alternatives": alternatives,
                "selected": selected
            },
            phase=phase
        )

    def add_tool_execution(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: Any,
        success: bool,
        phase: str | None = None
    ):
        """Add evidence for a tool execution."""
        self.add_evidence(
            "tool_execution",
            {
                "tool_name": tool_name,
                "args": args,
                "result": str(result)[:1000],
                "success": success
            },
            phase=phase
        )

    def _persist_entry(self, entry: EvidenceEntry):
        """Append entry to a JSONL file."""
        log_file = self.run_dir / "evidence.jsonl"
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception as e:
            logger.error("Failed to persist evidence entry", error=str(e))

    def export_summary(self) -> str:
        """
        Export a summary of all collected evidence.
        
        Returns:
            Path to the exported summary file
        """
        summary_path = self.run_dir / "summary.json"
        
        try:
            summary = {
                "run_id": self.run_id,
                "start_time": min((e.timestamp for e in self.evidence_log), default=time.time()),
                "end_time": max((e.timestamp for e in self.evidence_log), default=time.time()),
                "total_events": len(self.evidence_log),
                "events_by_type": self._count_by_type(),
                "events_by_phase": self._count_by_phase(),
                "evidence": [e.to_dict() for e in self.evidence_log]
            }
            
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            
            logger.info("Evidence summary exported", path=str(summary_path), events=len(self.evidence_log))
            return str(summary_path)
            
        except Exception as e:
            logger.error("Failed to export evidence summary", error=str(e))
            return ""

    def _count_by_type(self) -> dict[str, int]:
        """Count events by type."""
        counts: dict[str, int] = {}
        for entry in self.evidence_log:
            counts[entry.type] = counts.get(entry.type, 0) + 1
        return counts

    def _count_by_phase(self) -> dict[str, int]:
        """Count events by phase."""
        counts: dict[str, int] = {}
        for entry in self.evidence_log:
            if entry.phase:
                counts[entry.phase] = counts.get(entry.phase, 0) + 1
        return counts

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about collected evidence.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "run_id": self.run_id,
            "total_events": len(self.evidence_log),
            "events_by_type": self._count_by_type(),
            "events_by_phase": self._count_by_phase(),
            "output_dir": str(self.run_dir)
        }
