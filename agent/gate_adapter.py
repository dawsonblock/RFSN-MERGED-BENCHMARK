"""Gate adapter - single source of truth gate routing.

This module wraps PlanGate to provide a unified interface for all
evaluation and execution paths. No duplicate 'agent gate' logic allowed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import logging

from rfsn_controller.gates.plan_gate import PlanGate, PlanGateConfig, PlanGateError, StepGateError

logger = logging.getLogger(__name__)


@dataclass
class GateDecision:
    """Result of a gate decision."""
    allowed: bool
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class GateAdapter:
    """
    Single source of truth gate. No duplicate 'agent gate' logic.
    
    All profile/policy knobs must live in PlanGate config.
    This adapter routes everything through the real kernel gate.
    
    Usage:
        adapter = GateAdapter()
        decision = adapter.decide(state_snapshot, proposal)
        if not decision.allowed:
            print(f"Rejected: {decision.reason}")
    """
    
    def __init__(self, config: Optional[PlanGateConfig] = None):
        self.gate = PlanGate(config)
    
    def decide(self, state_snapshot: Dict[str, Any], proposal: Dict[str, Any]) -> GateDecision:
        """
        Make a gate decision for a proposal.
        
        Args:
            state_snapshot: Current state (repo, commit, attempt, etc.)
            proposal: The proposed action with type, patch_text, etc.
            
        Returns:
            GateDecision indicating whether the proposal is allowed
        """
        proposal_type = proposal.get("type", "unknown")
        
        # For patch proposals, validate as a plan with a single step
        if proposal_type == "patch":
            plan = self._wrap_patch_as_plan(proposal, state_snapshot)
        else:
            plan = proposal.get("plan", {"steps": [proposal]})
        
        try:
            self.gate.validate_plan(plan)
            return GateDecision(
                allowed=True,
                reason="APPROVED",
                metadata={
                    "proposal_type": proposal_type,
                    "gate_version": "plan_gate_v1",
                }
            )
        except (PlanGateError, StepGateError) as e:
            logger.warning("Gate rejection: %s", str(e))
            return GateDecision(
                allowed=False,
                reason=str(e),
                metadata={
                    "proposal_type": proposal_type,
                    "error_type": type(e).__name__,
                }
            )
        except Exception as e:
            logger.error("Gate error: %s", str(e))
            return GateDecision(
                allowed=False,
                reason=f"GATE_ERROR: {e}",
                metadata={"error_type": type(e).__name__}
            )
    
    def _wrap_patch_as_plan(self, proposal: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """Wrap a patch proposal as a single-step plan for gate validation."""
        return {
            "plan_id": f"patch_{state.get('attempt', 0)}",
            "steps": [
                {
                    "id": "apply_patch",
                    "step_type": "apply_patch",
                    "description": proposal.get("summary", "Apply patch"),
                    "patch": proposal.get("patch_text", ""),
                    "expected_outcome": "Patch applied successfully",
                }
            ]
        }
    
    def validate_step(self, step: Dict[str, Any]) -> GateDecision:
        """Validate a single step before execution."""
        try:
            self.gate.validate_step(step)
            return GateDecision(allowed=True, reason="APPROVED")
        except StepGateError as e:
            return GateDecision(allowed=False, reason=str(e))
    
    def get_config(self) -> Dict[str, Any]:
        """Get gate configuration for logging."""
        return self.gate.to_dict()


# Singleton for consistent gate usage
_default_adapter: Optional[GateAdapter] = None


def get_gate_adapter() -> GateAdapter:
    """Get the default gate adapter (singleton)."""
    global _default_adapter
    if _default_adapter is None:
        _default_adapter = GateAdapter()
    return _default_adapter
