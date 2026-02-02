"""Agent loop - serial execution with gate enforcement.

This is the core loop that drives the agent:
1. Propose action (from planner)
2. Gate validates
3. Execute if accepted
4. Log to ledger
5. Update state
6. Repeat until done

Serial authority: ONE proposal at a time, always gated.
"""

from __future__ import annotations

from typing import Callable, Dict, Any
import time

from .types import AgentState, Proposal, GateDecision, ExecResult, LedgerEvent, Phase
from .profiles import Profile
from memory.log import append_event

try:
    from rfsn_controller.structured_logging import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)


def run_episode(
    profile: Profile,
    state: AgentState,
    propose_fn: Callable[[Profile, AgentState], Proposal],
    gate_fn: Callable[[Profile, AgentState, Proposal], GateDecision],
    exec_fn: Callable[[Profile, AgentState, Proposal], ExecResult],
) -> AgentState:
    """Run a complete episode (one SWE-bench task).
    
    This is the CORE loop. Everything flows through here:
    - Planner proposes
    - Gate validates
    - Controller executes
    - Ledger logs
    
    Args:
        profile: Profile controlling behavior (Lite vs Verified)
        state: Current agent state
        propose_fn: Function that generates proposals
        gate_fn: Function that validates proposals
        exec_fn: Function that executes proposals
        
    Returns:
        Final agent state
        
    Example:
        >>> profile = load_profile("profiles/swebench_lite.yaml")
        >>> state = AgentState(...)
        >>> final_state = run_episode(profile, state, propose, gate, execute)
    """
    logger.info(
        "Starting episode",
        task_id=state.task_id,
        profile=profile.name,
        max_rounds=profile.max_rounds,
    )
    
    episode_start = time.time()
    
    while state.phase != Phase.DONE:
        # Budget check
        if state.budget.round_idx >= profile.max_rounds:
            logger.warning(
                "Max rounds reached",
                rounds=state.budget.round_idx,
                max_rounds=profile.max_rounds,
            )
            state.notes["stop_reason"] = "max_rounds"
            state.phase = Phase.DONE
            break
        
        round_start = time.time()
        state.budget.round_idx += 1
        
        logger.info(
            "Round start",
            round=state.budget.round_idx,
            phase=state.phase.value,
        )
        
        # 1. PROPOSE: Get next action from planner
        try:
            proposal = propose_fn(profile, state)
        except Exception as e:
            logger.error("Proposal generation failed", error=str(e))
            # Log error and try to continue
            state.notes["last_error"] = str(e)
            state.phase = Phase.DIAGNOSE
            continue
        
        logger.info(
            "Proposal generated",
            kind=proposal.kind,
            rationale=proposal.rationale[:100],
            evidence_count=len(proposal.evidence),
        )
        
        # 2. GATE: Validate proposal
        decision = gate_fn(profile, state, proposal)
        
        if not decision.accept:
            logger.warning(
                "Gate rejected proposal",
                reason=decision.reason,
                kind=proposal.kind,
            )
            
            # Track consecutive rejects
            reject_count = state.notes.get("consecutive_rejects", 0) + 1
            state.notes["consecutive_rejects"] = reject_count
            
            # Log rejection
            ev = LedgerEvent.now(
                task_id=state.task_id,
                repo_id=state.repo.repo_id,
                phase=state.phase,
                proposal=proposal,
                gate_decision=decision,
                exec_result=ExecResult(status="fail", summary="gate_reject"),
                result={"gate_reject": True, "reason": decision.reason},
            )
            append_event(state, ev)
            
            # Store rejection for planner to see
            state.notes["last_gate_reject"] = decision.reason
            state.notes["last_rejected_proposal"] = {
                "kind": proposal.kind,
                "rationale": proposal.rationale,
            }
            
            # After 2 rejects in PATCH_CANDIDATES, force specific guidance
            if reject_count >= 2 and state.phase == Phase.PATCH_CANDIDATES:
                state.notes["force_source_edit"] = True
                logger.warning("Multiple rejects - forcing source edit guidance")
            
            # Force re-planning
            state.phase = Phase.PLAN if state.phase != Phase.DIAGNOSE else Phase.DIAGNOSE
            continue
        
        # Reset reject counter on success
        state.notes["consecutive_rejects"] = 0
        
        logger.info("Gate accepted proposal")
        
        # 3. EXECUTE: Run the proposal
        try:
            exec_result = exec_fn(profile, state, proposal)
        except Exception as e:
            logger.error("Execution failed", error=str(e))
            exec_result = ExecResult(
                status="fail",
                summary=f"Execution error: {str(e)}",
            )
        
        logger.info(
            "Execution complete",
            status=exec_result.status,
            summary=exec_result.summary[:100],
        )
        
        # 4. UPDATE STATE: Track budgets and touched files
        if proposal.kind == "run_tests":
            state.budget.test_runs += 1
        
        if proposal.kind == "edit":
            state.budget.patch_attempts += 1
            files = proposal.inputs.get("files", [])
            state.touched_files = sorted(set(state.touched_files) | set(files))
            
            # Record outcome for learning
            diff = proposal.inputs.get("diff", "")
            if diff and profile.enable_outcome_learning:
                try:
                    from agent.outcome_learning import record_patch_outcome
                    record_patch_outcome(
                        patch_diff=diff,
                        success=exec_result.status == "ok",
                        task_id=state.task_id,
                    )
                except Exception as e:
                    logger.debug("Outcome learning disabled", reason=str(e))
        
        if proposal.kind in ["inspect", "search", "edit", "run_tests"]:
            state.budget.model_calls += 1
        
        # 5. LOG: Append to ledger
        result_payload: Dict[str, Any] = {
            "exec_status": exec_result.status,
            "round": state.budget.round_idx,
            "round_duration_s": time.time() - round_start,
        }
        
        # Add test results if available
        if "test_result" in exec_result.metrics:
            result_payload["test_result"] = exec_result.metrics["test_result"]
        
        ev = LedgerEvent.now(
            task_id=state.task_id,
            repo_id=state.repo.repo_id,
            phase=state.phase,
            proposal=proposal,
            gate_decision=decision,
            exec_result=exec_result,
            result=result_payload,
        )
        append_event(state, ev)
        
        # 6. ADVANCE: Phase transition (planner can override via state.notes)
        state = _advance_phase(state, proposal, exec_result, profile)
        
        logger.info(
            "Round complete",
            round=state.budget.round_idx,
            next_phase=state.phase.value,
        )
    
    episode_duration = time.time() - episode_start
    
    logger.info(
        "Episode complete",
        task_id=state.task_id,
        rounds=state.budget.round_idx,
        duration_s=round(episode_duration, 2),
        stop_reason=state.notes.get("stop_reason", "unknown"),
    )
    
    return state


def _advance_phase(
    state: AgentState,
    proposal: Proposal,
    exec_result: ExecResult,
    profile: Profile,
) -> AgentState:
    """Advance to next phase based on current phase and execution result.
    
    This is more aggressive about progressing to avoid stuck loops.
    
    Args:
        state: Current state
        proposal: Last proposal
        exec_result: Execution result
        profile: Profile
        
    Returns:
        Updated state
    """
    # Track phase attempts to prevent infinite loops
    phase_key = f"phase_{state.phase.value}_attempts"
    attempts = state.notes.get(phase_key, 0) + 1
    state.notes[phase_key] = attempts
    
    # Planner can force a specific phase
    if "next_phase" in state.notes:
        forced_phase = state.notes.pop("next_phase")
        logger.debug("Planner forced phase", phase=forced_phase)
        state.phase = Phase(forced_phase)
        return state
    
    # Default phase progression (more aggressive)
    if state.phase == Phase.INGEST:
        # Always move to LOCALIZE after first action
        state.phase = Phase.LOCALIZE
    
    elif state.phase == Phase.LOCALIZE:
        # Move to PLAN if we have hits OR if we've tried enough times
        if state.localization_hits or attempts >= 3:
            state.phase = Phase.PLAN
            logger.info("Moving to PLAN phase", 
                       hits=len(state.localization_hits), 
                       attempts=attempts)
    
    elif state.phase == Phase.PLAN:
        # After planning, move to patch (or if we've inspected enough)
        if proposal.kind == "edit" or attempts >= 2:
            state.phase = Phase.PATCH_CANDIDATES
    
    elif state.phase == Phase.PATCH_CANDIDATES:
        if proposal.kind == "edit" and exec_result.status == "ok":
            state.phase = Phase.TEST_STAGE
        elif proposal.kind == "edit" and exec_result.status == "fail":
            # Apply failed - if too many attempts, go to diagnose
            if attempts >= 2:
                state.phase = Phase.DIAGNOSE
            else:
                state.phase = Phase.PLAN
        elif attempts >= 2:
            # Force progress even without edit
            state.phase = Phase.PLAN
    
    elif state.phase == Phase.TEST_STAGE:
        test_result = exec_result.metrics.get("test_result", {})
        if test_result.get("passed", False):
            state.phase = Phase.FINALIZE  # Skip minimize for speed
            state.notes["solved"] = True
        else:
            state.phase = Phase.DIAGNOSE
    
    elif state.phase == Phase.DIAGNOSE:
        # After diagnosis, go back to planning
        state.phase = Phase.PLAN
    
    elif state.phase == Phase.MINIMIZE:
        state.phase = Phase.FINALIZE
    
    elif state.phase == Phase.FINALIZE:
        if proposal.kind == "finalize":
            state.phase = Phase.DONE
            state.notes["stop_reason"] = "finalized"
        elif attempts >= 2:
            # Force done if we've tried to finalize enough
            state.phase = Phase.DONE
            state.notes["stop_reason"] = "forced_finalize"
    
    return state


def check_budgets(state: AgentState, profile: Profile) -> tuple[bool, str]:
    """Check if any budget limits are exceeded.
    
    Args:
        state: Current state
        profile: Profile with limits
        
    Returns:
        (within_budget, reason)
    """
    if state.budget.round_idx >= profile.max_rounds:
        return False, f"max_rounds={profile.max_rounds}"
    
    if state.budget.patch_attempts >= profile.max_patch_attempts:
        return False, f"max_patch_attempts={profile.max_patch_attempts}"
    
    if state.budget.test_runs >= profile.max_test_runs:
        return False, f"max_test_runs={profile.max_test_runs}"
    
    if state.budget.model_calls >= profile.max_model_calls:
        return False, f"max_model_calls={profile.max_model_calls}"
    
    return True, "ok"
