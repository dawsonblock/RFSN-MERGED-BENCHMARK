from gate_ext.semantic_diff import semantic_diff_ok

def gate(state, proposal) -> bool:
    if not proposal.action_allowed:
        return False

    if not state.bounds_ok(proposal):
        return False

    if not semantic_diff_ok(
        proposal.diff_stats,
        state.policy.diff_limits
    ):
        return False

    return True
