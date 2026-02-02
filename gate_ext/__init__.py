"""Gate extension - wraps core gate with profile-driven policies.

This adds policies ON TOP of the existing gate:
- Phase policy
- File policy
- Test policy
- Diff policy

The core gate remains unchanged.
"""

from __future__ import annotations

from typing import Callable

from agent.types import AgentState, Proposal, GateDecision
from agent.profiles import Profile
from gate_ext.policy_phase import check_phase
from gate_ext.policy_files import check_files
from gate_ext.policy_tests import check_tests
from gate_ext.policy_diff import check_diff

try:
    from rfsn_controller.structured_logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def gate_with_profile(
    profile: Profile,
    state: AgentState,
    proposal: Proposal,
    core_gate_fn: Callable[[AgentState, Proposal], GateDecision] | None = None,
) -> GateDecision:
    """Gate with profile-driven policies.
    
    This wraps the core gate (if provided) with additional policies.
    
    Check order:
    1. Phase policy (must be in right phase)
    2. Core gate (existing RFSN gate, if available)
    3. File policy (file limits, vendor/CI)
    4. Test policy (test modifications, full suite)
    5. Diff policy (size, risk)
    
    Args:
        profile: Profile controlling behavior
        state: Current agent state
        proposal: Proposed action
        core_gate_fn: Optional core gate function
        
    Returns:
        GateDecision (accept/reject with reason)
        
    Example:
        >>> decision = gate_with_profile(profile, state, proposal)
        >>> if decision.accept:
        ...     execute(proposal)
    """
    logger.debug(
        "Gate checking proposal",
        kind=proposal.kind,
        phase=state.phase.value,
        profile=profile.name,
    )
    
    # 1. Phase policy
    ok, why = check_phase(profile, state, proposal)
    if not ok:
        logger.info("Gate reject (phase)", reason=why)
        return GateDecision(False, f"phase_policy: {why}")
    
    # 2. Core gate (existing RFSN gate)
    if core_gate_fn:
        try:
            core = core_gate_fn(state, proposal)
            if not core.accept:
                logger.info("Gate reject (core)", reason=core.reason)
                return core
        except Exception as e:
            logger.error("Core gate error", error=str(e))
            return GateDecision(False, f"core_gate_error: {str(e)}")
    
    # 3. File policy
    ok, why = check_files(profile, state, proposal)
    if not ok:
        logger.info("Gate reject (files)", reason=why)
        return GateDecision(False, f"file_policy: {why}")
    
    # 4. Test policy
    ok, why = check_tests(profile, state, proposal)
    if not ok:
        logger.info("Gate reject (tests)", reason=why)
        return GateDecision(False, f"test_policy: {why}")
    
    # 5. Diff policy
    ok, why = check_diff(profile, state, proposal)
    if not ok:
        logger.info("Gate reject (diff)", reason=why)
        return GateDecision(False, f"diff_policy: {why}")
    
    # All checks passed
    logger.debug("Gate accept")
    return GateDecision(True, "ok", constraints={"profile": profile.name})
