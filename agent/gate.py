"""Convenience wrapper for kernel PlanGate evaluation.

This module exposes a ``gate`` function that delegates all proposal
validations to the kernel's PlanGate via the ``GateAdapter``. It exists
solely for backward compatibility and convenience; it contains no
independent policy logic. All gating rules live in the PlanGate and
associated policies under ``rfsn_controller/gates`` and ``gate_ext``.

Usage:
    >>> from agent.profiles import Profile
    >>> from agent.types import AgentState, Proposal
    >>> from agent.gate import gate
    >>> decision = gate(profile, state, proposal)
    >>> print(decision.accept, decision.reason)
"""

from __future__ import annotations

from typing import Any, Dict

from agent.types import AgentState, Proposal, GateDecision
from agent.profiles import Profile
from agent.gate_adapter import get_gate_adapter


def gate(profile: Profile, state: AgentState, proposal: Proposal) -> GateDecision:
    """Validate a proposal against profile constraints using PlanGate.

    Converts the agent state and proposal into a simple dictionary format,
    calls the kernel gate via the adapter, and converts the result back
    into a GateDecision. No independent policy logic lives here.

    Args:
        profile: The agent profile (unused here; PlanGate reads config from its own source).
        state: Current agent state.
        proposal: Proposed action from the planner.

    Returns:
        GateDecision: accept/reject with a reason and optional constraints.
    """
    # Build a snapshot for the adapter
    state_snapshot: Dict[str, Any] = {
        "attempt": state.attempt,
        "phase": state.phase.value,
        "budget": {
            "test_runs": state.budget.test_runs,
            "patch_attempts": state.budget.patch_attempts,
        },
        "notes": dict(state.notes),
    }

    prop_dict: Dict[str, Any] = {
        "type": proposal.kind,
        "summary": proposal.rationale,
        "inputs": dict(proposal.inputs),
    }

    diff = proposal.inputs.get("diff") or proposal.inputs.get("patch")
    if isinstance(diff, str) and diff:
        prop_dict["patch_text"] = diff

    adapter = get_gate_adapter()
    decision = adapter.decide(state_snapshot=state_snapshot, proposal=prop_dict)

    return GateDecision(
        accept=bool(decision.allowed),
        reason=str(decision.reason),
        constraints=dict(decision.metadata) if hasattr(decision, "metadata") else {},
    )