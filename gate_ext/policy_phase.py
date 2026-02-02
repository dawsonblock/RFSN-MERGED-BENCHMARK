"""Phase policy - control which proposal kinds are allowed in each phase.

This is DETERMINISTIC. No learning.
"""

from __future__ import annotations

from typing import Tuple

from agent.types import AgentState, Proposal, Phase
from agent.profiles import Profile


def check_phase(profile: Profile, state: AgentState, proposal: Proposal) -> Tuple[bool, str]:
    """Check if proposal kind is allowed in current phase.
    
    Args:
        profile: Profile (not used, but kept for consistency)
        state: Current state
        proposal: Proposed action
        
    Returns:
        (allowed, reason)
        
    Example:
        >>> ok, reason = check_phase(profile, state, proposal)
        >>> if not ok:
        ...     print(f"Rejected: {reason}")
    """
    # Hard allowlist by phase
    allowed = {
        Phase.INGEST: {"inspect"},
        Phase.LOCALIZE: {"inspect", "search"},
        Phase.PLAN: {"inspect", "search"},
        Phase.PATCH_CANDIDATES: {"edit", "inspect", "search"},
        Phase.TEST_STAGE: {"run_tests", "inspect"},
        Phase.DIAGNOSE: {"inspect", "search"},
        Phase.MINIMIZE: {"edit", "inspect"},
        Phase.FINALIZE: {"finalize", "run_tests", "inspect"},
        Phase.DONE: set(),  # No actions allowed
    }
    
    allowed_kinds = allowed.get(state.phase, set())
    
    if proposal.kind not in allowed_kinds:
        return False, f"proposal.kind={proposal.kind} not allowed in phase={state.phase.value}"
    
    return True, "ok"
