"""Agent module - unified proposal and execution."""
from .gate_adapter import GateAdapter, GateDecision, get_gate_adapter
from .propose_v2 import (
    propose,
    learn_update,
    get_propose_stats,
    PatchCandidate,
    UpstreamContext,
    build_upstream_context,
)

__all__ = [
    "GateAdapter",
    "GateDecision",
    "get_gate_adapter",
    "propose",
    "learn_update",
    "get_propose_stats",
    "PatchCandidate",
    "UpstreamContext",
    "build_upstream_context",
]
