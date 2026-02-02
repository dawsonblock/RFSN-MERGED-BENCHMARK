import inspect


def test_simulation_gate_is_advisory_only():
    """SimulationGate must never execute tools, shell, or apply patches.

    This is a *structural invariant* test: it prevents future refactors from
    accidentally turning "mental simulation" into a bypass path.
    """
    from cgw_ssl_guard.coding_agent.sim.simulation_gate import SimulationGate

    src = inspect.getsource(SimulationGate)

    # Hard bans: no subprocess/shell execution in the SimulationGate implementation.
    # NOTE: the module's docstring may mention words like "subprocess".
    # We specifically ban *imports* and *runtime usage* patterns.
    banned = [
        "import subprocess",
        "os.system",
        "Popen",
        "exec(",
        "eval(",
        "GovernedExecutor",
    ]
    for token in banned:
        assert token not in src, f"SimulationGate must not reference `{token}`"


def test_simulation_gate_only_adjusts_candidate_scores():
    from cgw_ssl_guard.coding_agent.action_types import ActionPayload, CodingAction
    from cgw_ssl_guard.coding_agent.proposal_generators import ProposalContext
    from cgw_ssl_guard.coding_agent.sim.simulation_gate import SimulationGate
    from cgw_ssl_guard.event_bus import SimpleEventBus
    from cgw_ssl_guard.types import Candidate

    bus = SimpleEventBus()
    gate = SimulationGate(event_bus=bus)

    # Build a minimal APPLY_PATCH candidate with a diff payload.
    payload = ActionPayload(action=CodingAction.APPLY_PATCH, parameters={"diff": "diff --git a/x b/x\n"})
    c = Candidate(
        slot_id="1",
        source_module="test",
        content_payload=payload.to_bytes(),
        saliency=0.9,
        urgency=0.5,
        surprise=0.0,
    )

    ctx = ProposalContext(
        cycle_id=1,
        last_result=None,
        tests_passing=False,
        failing_tests=["tests/test_x.py::test_y"],
        test_output="",
        current_diff="",
    )

    before_scores = (c.saliency, c.urgency, c.surprise)
    before_payload = c.content_payload
    before_slot = c.slot_id
    gate.adjust_candidates([c], ctx)
    after_scores = (c.saliency, c.urgency, c.surprise)

    # The only allowed side effect is changing the score fields.
    assert c.content_payload == before_payload
    assert c.slot_id == before_slot
    assert all(isinstance(x, float) for x in after_scores)
    # Scores may or may not change depending on heuristics, but must stay bounded.
    assert all(0.0 <= x <= 1.0 for x in after_scores)
    assert all(0.0 <= x <= 1.0 for x in before_scores)
