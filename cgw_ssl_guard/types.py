"""Type definitions and dataclasses for the CGW/SSL guard runtime.

This module declares a handful of simple data structures used by the
thalamic gate and CGW runtime.  They are intentionally
straightforward, using Python dataclasses for brevity and clarity.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class SelectionReason(Enum):
    """Reason a signal won the thalamic gate.

    FORCED_OVERRIDE indicates that a forced signal won via the
    structural bypass.  COMPETITION indicates normal competition.  URGENCY
    and SURPRISE indicate a high urgency or surprise in normal
    competition; these are informational hints only.
    """

    FORCED_OVERRIDE = "FORCED_OVERRIDE"
    COMPETITION = "COMPETITION"
    URGENCY = "URGENCY"
    SURPRISE = "SURPRISE"


@dataclass
class Candidate:
    """A normal signal competing for the workspace.

    Attributes:
        slot_id: A unique identifier for this signal; used in event
            emissions and as the basis of the attended content ID.
        source_module: The module that generated this candidate.
        content_payload: Opaque bytes representing the latent or
            command to be committed to the workspace if selected.
        saliency: A scalar in [0,1] representing the relevance of the
            signal; higher is more relevant.
        urgency: A scalar in [0,1] representing time criticality.
        surprise: A scalar in [0,1] representing how unexpected the
            signal is.
    """

    slot_id: str
    source_module: str
    content_payload: bytes
    saliency: float
    urgency: float = 0.0
    surprise: float = 0.0

    def score(self) -> float:
        """Compute a heuristic score used for competition.

        This mixes saliency, urgency and surprise.  Weights are
        adjustable but simple constants work for demonstration.  More
        sophisticated gating could factor in context.
        """
        return 0.5 * self.saliency + 0.3 * self.urgency + 0.2 * self.surprise


@dataclass
class ForcedCandidate:
    """A forced signal that bypasses competition entirely."""

    slot_id: str
    source_module: str
    content_payload: bytes


@dataclass
class SelectionEvent:
    """Event emitted when the thalamic gate selects a winner."""

    cycle_id: int
    slot_id: str
    reason: SelectionReason
    timestamp: float
    forced_queue_size: int
    losers: List[str]
    winner_is_forced: bool

    def total_candidates(self) -> int:
        return 1 + len(self.losers)


@dataclass(frozen=True)
class AttendedContent:
    """The single authoritative slot in the CGW.

    The payload bytes and a SHA256 hash are stored; the hash is used
    for identity checks without exposing the payload.
    """

    slot_id: str
    payload_hash: str
    payload_bytes: bytes
    source_module: str
    timestamp: float


@dataclass
class CausalTrace:
    """Metadata about the winning selection.

    Not part of the attended content; used for introspection.
    """

    winner_reason: SelectionReason
    winner_score: Optional[float]
    losers: List[str] = field(default_factory=list)
    forced_override: bool = False


@dataclass
class SelfModel:
    """Persistent self‑state across cycles.

    This is a placeholder for a more sophisticated self model.  For the
    purposes of the step‑1 runtime, it simply tracks goals, intentions,
    and confidence estimates.
    """

    goals: List[str] = field(default_factory=list)
    active_intentions: List[str] = field(default_factory=list)
    confidence_estimates: Dict[str, float] = field(default_factory=dict)

    def delta_magnitude(self, other: SelfModel) -> float:
        """Compute a crude magnitude of change between self models."""
        goal_diff = set(self.goals) ^ set(other.goals)
        intent_diff = set(self.active_intentions) ^ set(other.active_intentions)
        conf_diff_keys = set(self.confidence_estimates).union(other.confidence_estimates)
        conf_diff = sum(abs(self.confidence_estimates.get(k, 0) - other.confidence_estimates.get(k, 0))
                        for k in conf_diff_keys)
        return float(len(goal_diff) + len(intent_diff) + conf_diff)


@dataclass
class CGWState:
    """Complete workspace state for one moment."""

    cycle_id: int
    timestamp: float
    attended_content: AttendedContent
    causal_trace: CausalTrace
    self_model: SelfModel

    def content_id(self) -> str:
        return self.attended_content.slot_id

    def content_hash(self) -> str:
        return self.attended_content.payload_hash