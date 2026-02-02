"""CLI - Plan visualization for developers.

Simple CLI for printing plan DAG, step statuses, and dependencies.
"""

from __future__ import annotations

from .schema import Plan, PlanState, StepStatus

# Box drawing characters for tree
PIPE = "│"
TEE = "├"
ELBOW = "└"
DASH = "─"


def print_plan_dag(plan: Plan, state: PlanState | None = None) -> str:
    """Print plan DAG with statuses and dependencies.
    
    Args:
        plan: The plan to visualize.
        state: Optional current state.
        
    Returns:
        Formatted string representation.
    
    Example output:
        ┌─ analyze-failure [DONE] ✓
        │  └─> locate-source [DONE] ✓
        │      └─> propose-fix [ACTIVE] ⏳
        │          ├─> verify-focused [PENDING]
        │          └─> verify-regression [PENDING]
        └─> verify-full [PENDING]
    """
    lines = []
    lines.append(f"Plan: {plan.plan_id} (v{plan.version})")
    lines.append(f"Goal: {plan.goal}")
    lines.append("")
    
    # Build dependency mapping
    {s.step_id: s.dependencies for s in plan.steps}
    
    # Find root steps (no dependencies)
    roots = [s for s in plan.steps if not s.dependencies]
    
    # Build child mapping
    children = {s.step_id: [] for s in plan.steps}
    for step in plan.steps:
        for dep_id in step.dependencies:
            if dep_id in children:
                children[dep_id].append(step.step_id)
    
    # Print tree
    visited = set()
    
    def print_step(step_id: str, prefix: str = "", is_last: bool = True):
        if step_id in visited:
            return
        visited.add(step_id)
        
        step = plan.get_step(step_id)
        if not step:
            return
        
        # Status indicator
        status_icons = {
            StepStatus.PENDING: "○",
            StepStatus.ACTIVE: "⏳",
            StepStatus.DONE: "✓",
            StepStatus.FAILED: "✗",
            StepStatus.SKIPPED: "⊘",
            StepStatus.BLOCKED: "⊗",
        }
        icon = status_icons.get(step.status, "?")
        
        # Build line
        connector = ELBOW if is_last else TEE
        line = f"{prefix}{connector}{DASH} {step.step_id} [{step.status.value}] {icon}"
        lines.append(line)
        
        # Print children
        child_ids = children.get(step_id, [])
        for i, child_id in enumerate(child_ids):
            child_prefix = prefix + ("   " if is_last else f"{PIPE}  ")
            print_step(child_id, child_prefix, i == len(child_ids) - 1)
    
    for i, root in enumerate(roots):
        print_step(root.step_id, "", i == len(roots) - 1)
    
    return "\n".join(lines)


def print_plan_summary(plan: Plan, state: PlanState) -> str:
    """Print current plan execution summary.
    
    Args:
        plan: The plan.
        state: Current state.
        
    Returns:
        Summary string.
    """
    total = len(plan.steps)
    done = len(state.completed_steps)
    failed = len(state.failed_steps)
    pending = total - done - failed
    
    lines = [
        f"Plan: {plan.plan_id}",
        f"Goal: {plan.goal}",
        "",
        f"Progress: {done}/{total} steps complete",
        f"  Done:    {done}",
        f"  Failed:  {failed}",
        f"  Pending: {pending}",
        "",
        f"Revisions: {state.revision_count}",
        f"Halted:    {state.halted}",
    ]
    
    if state.halted:
        lines.append(f"Halt reason: {state.halt_reason}")
    
    return "\n".join(lines)


def print_step_detail(plan: Plan, step_id: str) -> str:
    """Print detailed view of a single step.
    
    Args:
        plan: The plan.
        step_id: Step to show.
        
    Returns:
        Detailed step string.
    """
    step = plan.get_step(step_id)
    if not step:
        return f"Step {step_id} not found"
    
    lines = [
        f"Step: {step.step_id}",
        f"Title: {step.title}",
        f"Status: {step.status.value}",
        f"Risk: {step.risk_level.value}",
        "",
        "Intent:",
        f"  {step.intent}",
        "",
        "Success Criteria:",
        f"  {step.success_criteria}",
        "",
        f"Allowed Files: {step.allowed_files}",
        f"Dependencies: {step.dependencies}",
        f"Verify: {step.verify or '(none)'}",
    ]
    
    if step.failure_count > 0:
        lines.append(f"\nFailures: {step.failure_count}")
    
    if step.result:
        lines.append(f"\nResult: {step.result}")
    
    return "\n".join(lines)


def format_plan_for_logging(plan: Plan) -> str:
    """Format plan for structured logging.
    
    Args:
        plan: The plan.
        
    Returns:
        Compact log-friendly string.
    """
    step_strs = []
    for s in plan.steps:
        deps = f" <- {','.join(s.dependencies)}" if s.dependencies else ""
        step_strs.append(f"{s.step_id}[{s.status.value}]{deps}")
    
    return f"Plan({plan.plan_id}, v{plan.version}): " + " → ".join(step_strs)
