from planner.plan_graph import validate_transition

def planner_propose(prev_step: str, next_step: str):
    if not validate_transition(prev_step, next_step):
        raise ValueError(f"Illegal planner transition: {prev_step} → {next_step}")
