"""Risk Routing - Risk-aware step constraints.

HIGH-risk steps require stricter constraints:
- Smaller file allowlist
- Mandatory tests
- Smaller diff limits
- Two-phase execution (scaffold first, then modify)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..schema import RiskLevel, Step


@dataclass
class RiskConstraints:
    """Constraints applied based on risk level."""
    
    max_files: int
    max_diff_lines: int
    require_tests: bool
    two_phase_execution: bool
    require_rollback_hint: bool
    max_retry_attempts: int
    
    def to_dict(self) -> dict:
        return {
            "max_files": self.max_files,
            "max_diff_lines": self.max_diff_lines,
            "require_tests": self.require_tests,
            "two_phase_execution": self.two_phase_execution,
            "require_rollback_hint": self.require_rollback_hint,
            "max_retry_attempts": self.max_retry_attempts,
        }


# Default risk profiles
RISK_PROFILES: dict[str, RiskConstraints] = {
    "LOW": RiskConstraints(
        max_files=10,
        max_diff_lines=200,
        require_tests=False,
        two_phase_execution=False,
        require_rollback_hint=False,
        max_retry_attempts=3,
    ),
    "MED": RiskConstraints(
        max_files=5,
        max_diff_lines=100,
        require_tests=True,
        two_phase_execution=False,
        require_rollback_hint=True,
        max_retry_attempts=2,
    ),
    "HIGH": RiskConstraints(
        max_files=2,
        max_diff_lines=50,
        require_tests=True,
        two_phase_execution=True,
        require_rollback_hint=True,
        max_retry_attempts=1,
    ),
}


def get_risk_constraints(risk_level: RiskLevel) -> RiskConstraints:
    """Get constraints for a risk level.
    
    Args:
        risk_level: The RiskLevel enum value.
        
    Returns:
        RiskConstraints for that level.
    """
    return RISK_PROFILES.get(risk_level.value, RISK_PROFILES["LOW"])


def validate_step_against_risk(step: Step) -> list[str]:
    """Validate a step against its risk constraints.
    
    Args:
        step: The step to validate.
        
    Returns:
        List of constraint violations.
    """
    constraints = get_risk_constraints(step.risk_level)
    violations = []
    
    # Check file count
    if len(step.allowed_files) > constraints.max_files:
        violations.append(
            f"Step allows {len(step.allowed_files)} files, "
            f"max for {step.risk_level.value} risk is {constraints.max_files}"
        )
    
    # Check rollback hint requirement
    if constraints.require_rollback_hint and not step.rollback_hint:
        violations.append(
            f"Step has {step.risk_level.value} risk but no rollback_hint"
        )
    
    # Check verification requirement
    if constraints.require_tests and not step.verify:
        violations.append(
            f"Step has {step.risk_level.value} risk but no verify command"
        )
    
    return violations


def should_split_step(step: Step) -> bool:
    """Check if step should use two-phase execution.
    
    Two-phase execution:
    1. First phase: Create scaffold (empty functions, interfaces)
    2. Second phase: Fill in implementation
    
    Args:
        step: The step to check.
        
    Returns:
        True if two-phase execution required.
    """
    constraints = get_risk_constraints(step.risk_level)
    return constraints.two_phase_execution


def get_max_retries(step: Step) -> int:
    """Get maximum retry attempts for a step.
    
    Higher risk = fewer retries (fail fast).
    
    Args:
        step: The step to check.
        
    Returns:
        Maximum retry attempts.
    """
    constraints = get_risk_constraints(step.risk_level)
    return constraints.max_retry_attempts


def get_diff_limit(step: Step) -> int:
    """Get maximum diff lines for a step.
    
    Args:
        step: The step to check.
        
    Returns:
        Maximum diff lines.
    """
    constraints = get_risk_constraints(step.risk_level)
    return constraints.max_diff_lines
