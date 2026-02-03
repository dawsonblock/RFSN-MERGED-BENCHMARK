def reward_from_episode(ep: dict) -> float:
    """
    Deterministic reward.
    Gate does NOT see this.
    """

    if not ep["tests_ran"]:
        return -1.0

    if ep["tests_failed"] > 0:
        return -0.5

    if ep["tests_passed"] and ep["patch_applied"]:
        return 1.0

    return 0.0
