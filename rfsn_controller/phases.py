"""Pipeline phases for the RFSN controller.
from __future__ import annotations

This module defines the explicit state machine phases for the controller loop.
Each phase represents a distinct stage in the repair pipeline.
"""

from enum import Enum


class Phase(Enum):
    """Explicit pipeline phases for the controller state machine."""

    INGEST = "ingest"
    """Clone/checkout/reset the repository."""

    DETECT = "detect"
    """Detect project type and select appropriate commands."""

    SETUP = "setup"
    """Install dependencies (network ON)."""

    BASELINE = "baseline"
    """Run tests to record initial failure signature (network OFF)."""

    REPAIR_LOOP = "repair_loop"
    """Repeated: gather evidence → propose patch/tool_request → verify candidates → apply winner."""

    FINAL_VERIFY = "final_verify"
    """Run full test suite to confirm fix (network OFF)."""

    EVIDENCE_PACK = "evidence_pack"
    """Export artifacts (winner.diff, before.txt, after.txt, state.json, run.jsonl, files_changed.txt)."""

    BAILOUT = "bailout"
    """Exit due to time/steps exceeded or stall detected."""


class PhaseTransition:
    """Track phase transitions and metadata."""

    def __init__(self, from_phase: Phase | None, to_phase: Phase, reason: str = ""):
        self.from_phase = from_phase
        self.to_phase = to_phase
        self.reason = reason

    def to_dict(self) -> dict:
        return {
            "from": self.from_phase.value if self.from_phase else None,
            "to": self.to_phase.value,
            "reason": self.reason,
        }
