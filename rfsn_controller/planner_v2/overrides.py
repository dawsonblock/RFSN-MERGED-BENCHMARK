"""Overrides - Runtime plan modification without code changes.

Allows "skip step," "tighten allowlist," "change verify command," 
"halt plan" etc. via JSON file or programmatic interface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import Step


@dataclass
class PlanOverride:
    """Runtime override specification for plan execution."""
    
    # Steps to skip entirely
    skip_steps: set[str] = field(default_factory=set)
    
    # Tighten file allowlists: step_id -> new allowlist
    tighten_allowlists: dict[str, list[str]] = field(default_factory=dict)
    
    # Change verification commands: step_id -> new command
    change_verify_commands: dict[str, str] = field(default_factory=dict)
    
    # Force halt plan
    halt_plan: bool = False
    halt_reason: str = ""
    
    # Force step risk level: step_id -> new risk level
    force_risk_levels: dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "skip_steps": list(self.skip_steps),
            "tighten_allowlists": self.tighten_allowlists,
            "change_verify_commands": self.change_verify_commands,
            "halt_plan": self.halt_plan,
            "halt_reason": self.halt_reason,
            "force_risk_levels": self.force_risk_levels,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> PlanOverride:
        return cls(
            skip_steps=set(data.get("skip_steps", [])),
            tighten_allowlists=data.get("tighten_allowlists", {}),
            change_verify_commands=data.get("change_verify_commands", {}),
            halt_plan=data.get("halt_plan", False),
            halt_reason=data.get("halt_reason", ""),
            force_risk_levels=data.get("force_risk_levels", {}),
        )


class OverrideManager:
    """Manages runtime overrides for plan execution."""
    
    def __init__(self, override_file: Path | None = None):
        """Initialize the override manager.
        
        Args:
            override_file: Optional JSON file containing overrides.
        """
        self.override_file = Path(override_file) if override_file else None
        self._overrides = PlanOverride()
        self._load_overrides()
    
    def _load_overrides(self) -> None:
        """Load overrides from file if present."""
        if self.override_file and self.override_file.exists():
            try:
                data = json.loads(self.override_file.read_text())
                self._overrides = PlanOverride.from_dict(data)
            except (OSError, json.JSONDecodeError):
                pass
    
    def save_overrides(self) -> None:
        """Save current overrides to file."""
        if self.override_file:
            self.override_file.write_text(json.dumps(self._overrides.to_dict(), indent=2))
    
    def should_skip(self, step_id: str) -> bool:
        """Check if step should be skipped.
        
        Args:
            step_id: The step ID.
            
        Returns:
            True if step should be skipped.
        """
        return step_id in self._overrides.skip_steps
    
    def should_halt(self) -> bool:
        """Check if plan should halt.
        
        Returns:
            True if halt requested.
        """
        return self._overrides.halt_plan
    
    def get_halt_reason(self) -> str:
        """Get halt reason if halted."""
        return self._overrides.halt_reason
    
    def apply(self, step: Step) -> Step:
        """Apply overrides to a step before execution.
        
        Args:
            step: The step to modify.
            
        Returns:
            Modified step (original is not mutated).
        """
        from .schema import RiskLevel, Step
        
        # Create a copy with potential modifications
        new_allowed_files = step.allowed_files
        if step.step_id in self._overrides.tighten_allowlists:
            new_allowed_files = self._overrides.tighten_allowlists[step.step_id]
        
        new_verify = step.verify
        if step.step_id in self._overrides.change_verify_commands:
            new_verify = self._overrides.change_verify_commands[step.step_id]
        
        new_risk = step.risk_level
        if step.step_id in self._overrides.force_risk_levels:
            new_risk = RiskLevel(self._overrides.force_risk_levels[step.step_id])
        
        # Return new step with overrides applied
        return Step(
            step_id=step.step_id,
            title=step.title,
            intent=step.intent,
            allowed_files=new_allowed_files,
            success_criteria=step.success_criteria,
            dependencies=step.dependencies,
            inputs=step.inputs,
            verify=new_verify,
            risk_level=new_risk,
            rollback_hint=step.rollback_hint,
            controller_task_spec=step.controller_task_spec,
            status=step.status,
            result=step.result,
            failure_count=step.failure_count,
        )
    
    # Programmatic override methods
    
    def skip_step(self, step_id: str) -> None:
        """Add step to skip list."""
        self._overrides.skip_steps.add(step_id)
    
    def unskip_step(self, step_id: str) -> None:
        """Remove step from skip list."""
        self._overrides.skip_steps.discard(step_id)
    
    def tighten_allowlist(self, step_id: str, files: list[str]) -> None:
        """Tighten file allowlist for a step."""
        self._overrides.tighten_allowlists[step_id] = files
    
    def change_verify(self, step_id: str, command: str) -> None:
        """Change verification command for a step."""
        self._overrides.change_verify_commands[step_id] = command
    
    def force_risk(self, step_id: str, risk: str) -> None:
        """Force risk level for a step."""
        self._overrides.force_risk_levels[step_id] = risk
    
    def request_halt(self, reason: str = "Manual halt requested") -> None:
        """Request plan halt."""
        self._overrides.halt_plan = True
        self._overrides.halt_reason = reason
    
    def clear_halt(self) -> None:
        """Clear halt request."""
        self._overrides.halt_plan = False
        self._overrides.halt_reason = ""
    
    def clear_all(self) -> None:
        """Clear all overrides."""
        self._overrides = PlanOverride()
    
    def get_overrides(self) -> PlanOverride:
        """Get current overrides."""
        return self._overrides
