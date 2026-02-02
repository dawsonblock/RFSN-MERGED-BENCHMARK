"""File policy - control which files can be edited.

Enforces:
- max_files_touched per episode
- Forbid vendor/ and third_party/ edits
- Forbid .github/workflows/ (CI) edits
"""

from __future__ import annotations

from typing import Tuple, Set

from agent.types import AgentState, Proposal
from agent.profiles import Profile


# Patterns that trigger vendor detection
VENDOR_HINTS = ("third_party/", "vendor/", "site-packages/", "node_modules/", "external/")

# Patterns that trigger CI detection
CI_HINTS = (".github/workflows/", ".gitlab-ci.yml", ".circleci/", "Jenkinsfile", ".travis.yml")


def _touched_files_from_proposal(proposal: Proposal) -> Set[str]:
    """Extract files from proposal.
    
    Edit proposals must include "files": ["a.py", ...] in inputs.
    
    Args:
        proposal: Proposal to check
        
    Returns:
        Set of file paths
    """
    if proposal.kind != "edit":
        return set()
    
    files = proposal.inputs.get("files", [])
    return set(map(str, files))


def check_files(profile: Profile, state: AgentState, proposal: Proposal) -> Tuple[bool, str]:
    """Check if file edits are allowed.
    
    Args:
        profile: Profile with limits
        state: Current state
        proposal: Proposed action
        
    Returns:
        (allowed, reason)
    """
    touched = _touched_files_from_proposal(proposal)
    
    if not touched:
        return True, "ok"
    
    # Enforce max touched files across episode
    episode_touched = set(state.touched_files) | touched
    
    if len(episode_touched) > profile.max_files_touched:
        return False, (
            f"max_files_touched exceeded: {len(episode_touched)} > {profile.max_files_touched}"
        )
    
    # Forbid vendor edits (unless profile allows)
    if not profile.allow_vendor_edits:
        for f in touched:
            for hint in VENDOR_HINTS:
                if hint in f:
                    return False, f"vendor edit forbidden: {f}"
    
    # Forbid CI edits (unless profile allows)
    if not profile.allow_ci_edits:
        for f in touched:
            for hint in CI_HINTS:
                if hint in f or f.endswith(hint):
                    return False, f"CI edit forbidden: {f}"
    
    return True, "ok"
