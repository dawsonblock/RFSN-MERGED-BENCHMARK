"""Test policy - control test modifications and requirements.

Enforces:
- Verified: NEVER allow test file edits
- Verified: Require full suite pass before finalize
"""

from __future__ import annotations

from typing import Tuple

from agent.types import AgentState, Proposal, Phase
from agent.profiles import Profile


def check_tests(profile: Profile, state: AgentState, proposal: Proposal) -> Tuple[bool, str]:
    """Check test-related policies.
    
    Args:
        profile: Profile with policies
        state: Current state
        proposal: Proposed action
        
    Returns:
        (allowed, reason)
    """
    # Verified: NEVER allow test modifications
    if profile.forbid_test_modifications and proposal.kind == "edit":
        files = proposal.inputs.get("files", [])
        
        for f in files:
            # Detect test files
            if "test" in f.lower() or f.startswith("tests/"):
                return False, f"test modification forbidden in profile={profile.name}: {f}"
    
    # Verified: require full suite before finalize
    if state.phase == Phase.FINALIZE and proposal.kind == "finalize":
        if profile.require_full_suite_for_finalize:
            # Check if full suite already passed
            if not bool(state.notes.get("full_suite_passed", False)):
                return False, "full test suite not recorded as passed; cannot finalize"
    
    return True, "ok"
