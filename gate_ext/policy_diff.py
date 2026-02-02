"""Diff policy - control patch size and risk.

Enforces:
- max_diff_lines
- Risk scoring (large deletions, broad changes)
"""

from __future__ import annotations

from typing import Tuple

from agent.types import AgentState, Proposal
from agent.profiles import Profile


def check_diff(profile: Profile, state: AgentState, proposal: Proposal) -> Tuple[bool, str]:
    """Check diff size and risk.
    
    Args:
        profile: Profile with limits
        state: Current state
        proposal: Proposed action
        
    Returns:
        (allowed, reason)
    """
    if proposal.kind != "edit":
        return True, "ok"
    
    # Check diff size
    diff_text = proposal.inputs.get("diff", "")
    
    if diff_text:
        lines = diff_text.split("\n")
        added = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
        total_changes = added + removed
        
        if total_changes > profile.max_diff_lines:
            return False, (
                f"diff too large: {total_changes} lines > {profile.max_diff_lines}"
            )
        
        # Risk check: large deletions are risky
        if removed > added * 2:
            # More than 2x deletions vs additions
            risk_score = removed / max(added, 1)
            if risk_score > 5.0:
                return False, (
                    f"diff too risky: {removed} deletions vs {added} additions (ratio {risk_score:.1f})"
                )
    
    return True, "ok"
