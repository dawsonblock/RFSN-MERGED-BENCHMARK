"""RFSN Planner Self-Critique Rubric.

This module validates plans BEFORE submission to PlanGate.
It implements the hard fail and soft fail conditions from the
RFSN Master Execution Prompt.

Usage:
    from rfsn_controller.gates.self_critique import critique_plan, CritiqueResult
    
    report = critique_plan(plan, gate_config)
    if report.result == CritiqueResult.REJECTED:
        # Regenerate plan
        pass
    elif report.result == CritiqueResult.REWRITE_ADVISED:
        # Consider regenerating
        pass
    else:
        # Submit to gate
        pass
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .plan_gate import DEFAULT_ALLOWED_STEP_TYPES, PlanGateConfig

logger = logging.getLogger(__name__)


class CritiqueResult(Enum):
    """Result of self-critique analysis."""
    
    APPROVED = "APPROVED"           # Plan is safe, minimal, and gate-compatible
    REJECTED = "REJECTED"           # Violates hard constraints (list violations)
    REWRITE_ADVISED = "REWRITE_ADVISED"  # Passes gate, but suboptimal (list risks)


@dataclass
class CritiqueReport:
    """Report from self-critique analysis."""
    
    result: CritiqueResult
    hard_failures: list[str] = field(default_factory=list)
    soft_warnings: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "result": self.result.value,
            "hard_failures": self.hard_failures,
            "soft_warnings": self.soft_warnings,
        }


# =============================================================================
# Hard Fail Conditions (any ONE = reject plan)
# =============================================================================

def _check_structural_correctness(
    plan: dict[str, Any],
    config: PlanGateConfig,
) -> list[str]:
    """Check structural correctness of plan.
    
    A. Structural Correctness Checks:
    1. Does every step have a unique id?
    2. Does every step have a valid type from the allowlist?
    3. Is the dependency graph acyclic?
    4. Is the total number of steps <= the gate's step budget?
    5. Does every step include expected_outcome or success_criteria?
    """
    failures = []
    steps = plan.get("steps", [])
    
    if not isinstance(steps, list):
        failures.append("Steps must be a list")
        return failures
    
    # Check 1: Unique IDs
    seen_ids: set[str] = set()
    for step in steps:
        step_id = step.get("id", step.get("step_id"))
        if not step_id:
            failures.append("Step missing 'id' field")
        elif step_id in seen_ids:
            failures.append(f"Duplicate step id: {step_id}")
        else:
            seen_ids.add(step_id)
    
    # Check 2: Valid step types
    allowed = config.allowed_step_types or DEFAULT_ALLOWED_STEP_TYPES
    for step in steps:
        step_type = step.get("type", step.get("step_type"))
        if step_type not in allowed:
            failures.append(f"Step type '{step_type}' not in allowlist")
    
    # Check 3: Acyclic dependency graph
    if _has_cycles(steps):
        failures.append("Dependency graph contains cycles")
    
    # Check 4: Step budget
    if len(steps) > config.max_steps:
        failures.append(f"Plan exceeds step budget: {len(steps)} > {config.max_steps}")
    
    # Check 5: Expected outcomes
    if config.require_expected_outcomes:
        for step in steps:
            step_id = step.get("id", step.get("step_id", "unknown"))
            if "expected_outcome" not in step and "success_criteria" not in step:
                failures.append(f"Step {step_id} missing expected_outcome or success_criteria")
    
    return failures


def _check_gate_compatibility(
    plan: dict[str, Any],
    config: PlanGateConfig,
) -> list[str]:
    """Check gate compatibility.
    
    B. Gate Compatibility Checks:
    6. Does the plan avoid all non-allowlisted step types?
    7. Does the plan avoid runtime mutation (no step that adds/registers/changes policies)?
    8. Does the plan assume serial execution only?
    9. Does the plan avoid relying on side effects from failed steps?
    """
    failures = []
    steps = plan.get("steps", [])
    
    # Check 6: Already covered in structural correctness
    
    # Check 7: Runtime mutation detection
    mutation_keywords = [
        "register", "add_policy", "change_policy", "modify_gate",
        "add_step_type", "remove_step_type", "bypass", "disable",
    ]
    for step in steps:
        step_id = step.get("id", step.get("step_id", "unknown"))
        step_str = str(step).lower()
        for keyword in mutation_keywords:
            if keyword in step_str:
                failures.append(
                    f"Step {step_id} may attempt runtime mutation: '{keyword}'"
                )
    
    # Check 8: Serial execution (no explicit parallel flags)
    for step in steps:
        step_id = step.get("id", step.get("step_id", "unknown"))
        if step.get("parallel") or step.get("async"):
            failures.append(f"Step {step_id} requests parallel/async execution")
    
    # Check 9: Side effect reliance (steps depending on failed steps)
    # This is validated at execution time, but we can flag suspicious patterns
    # like "fallback" or "on_error" dependencies
    for step in steps:
        step_id = step.get("id", step.get("step_id", "unknown"))
        deps = step.get("depends_on", [])
        if any("error" in str(d).lower() or "fail" in str(d).lower() for d in deps):
            failures.append(
                f"Step {step_id} may rely on failure side effects"
            )
    
    return failures


def _check_command_safety(
    plan: dict[str, Any],
    config: PlanGateConfig,
) -> list[str]:
    """Check command and execution safety.
    
    C. Command & Execution Safety Checks:
    10. Does the plan avoid all shell interpreters (sh, bash, etc.)?
    11. Does the plan avoid all wrapper flags (-c, -lc, -ic, -e)?
    12. Does the plan avoid inline env vars (FOO=bar cmd)?
    13. Does the plan avoid chaining (&&, ||, pipes, redirects)?
    14. Does the plan avoid interpreter string execution (python -c, node -e)?
    15. Are all executions indirect via step types (never raw commands)?
    """
    failures = []
    steps = plan.get("steps", [])
    
    shell_interpreters = {"sh", "bash", "dash", "zsh", "ksh", "csh", "tcsh"}
    wrapper_flags = {"-c", "-lc", "-ic", "-e", "-p"}
    chaining_patterns = ["&&", "||", ";", "|", ">", ">>", "<"]
    inline_env_pattern = r"^[A-Z_][A-Z0-9_]*=\S+\s+\w+"
    inline_env_re = re.compile(inline_env_pattern)
    
    for step in steps:
        step_id = step.get("id", step.get("step_id", "unknown"))
        inputs = step.get("inputs", {})
        
        # Checks 10-14: Scan all input values
        for _key, val in inputs.items():
            if not isinstance(val, str):
                continue
            
            val_lower = val.lower()
            val_split = val.split()
            
            # Check 10: Shell interpreters
            if val_split and val_split[0] in shell_interpreters:
                failures.append(
                    f"Step {step_id} uses shell interpreter: {val_split[0]}"
                )
            
            # Check 11: Wrapper flags
            for flag in wrapper_flags:
                if flag in val_split:
                    failures.append(
                        f"Step {step_id} uses wrapper flag: {flag}"
                    )
            
            # Check 12: Inline env vars
            if inline_env_re.match(val):
                failures.append(
                    f"Step {step_id} uses inline env var: {val[:30]}..."
                )
            
            # Check 13: Chaining
            for pattern in chaining_patterns:
                if pattern in val:
                    failures.append(
                        f"Step {step_id} uses chaining pattern: '{pattern}'"
                    )
            
            # Check 14: Interpreter string execution
            if "python -c" in val_lower or "python3 -c" in val_lower:
                failures.append(f"Step {step_id} uses python -c execution")
            if "node -e" in val_lower or "node -p" in val_lower:
                failures.append(f"Step {step_id} uses node -e execution")
            if "ruby -e" in val_lower or "perl -e" in val_lower:
                failures.append(f"Step {step_id} uses interpreter -e execution")
        
        # Check 15: Raw commands
        if "command" in inputs or "shell" in inputs:
            if not config.allow_shell_in_tools:
                failures.append(
                    f"Step {step_id} attempts raw command execution"
                )
    
    return failures


def _check_path_safety(
    plan: dict[str, Any],
    config: PlanGateConfig,
) -> list[str]:
    """Check file and path safety.
    
    D. File & Path Safety Checks:
    16. Are all paths relative and inside the workspace?
    17. Are forbidden paths avoided (.., .git/hooks, .github/workflows, /etc, /usr)?
    18. Are secrets, credentials, and env files avoided?
    19. Is every file mutation minimal and targeted?
    """
    failures = []
    steps = plan.get("steps", [])
    
    forbidden_patterns = config.forbidden_path_patterns or [
        "..", "/etc", "/usr", "/bin", "/sbin", "/root", "/home",
        ".git/hooks", ".github/workflows",
        ".env", ".secrets", "credentials", "id_rsa", "id_ed25519",
    ]
    
    for step in steps:
        step_id = step.get("id", step.get("step_id", "unknown"))
        inputs = step.get("inputs", {})
        
        # Collect path-like values
        paths: list[str] = []
        for key in ["file", "path", "target_file", "files", "allowed_files"]:
            val = inputs.get(key)
            if isinstance(val, str):
                paths.append(val)
            elif isinstance(val, list):
                paths.extend(v for v in val if isinstance(v, str))
        
        for path_str in paths:
            # Check 16: Absolute paths (outside workspace)
            if path_str.startswith("/") and not path_str.startswith("/workspace"):
                failures.append(
                    f"Step {step_id} uses absolute path: {path_str}"
                )
            
            # Check 17: Forbidden patterns
            for pattern in forbidden_patterns:
                if pattern in path_str:
                    failures.append(
                        f"Step {step_id} accesses forbidden path: '{pattern}' in '{path_str}'"
                    )
    
    return failures


def _check_verification_discipline(
    plan: dict[str, Any],
    config: PlanGateConfig,
) -> list[str]:
    """Check verification discipline.
    
    E. Verification Discipline Checks:
    20. Is verification explicitly planned (run_tests, check_syntax, etc.)?
    21. Does verification happen AFTER mutation steps?
    22. Does the plan halt on failure instead of cascading?
    """
    failures = []
    steps = plan.get("steps", [])
    
    if not config.require_verification:
        return failures
    
    verification_types = {"run_tests", "run_lint", "check_syntax", "validate_types"}
    mutation_types = {"apply_patch", "add_test", "refactor_small", "fix_import", "fix_typing"}
    
    has_verification = False
    has_mutation = False
    mutation_after_verify = False
    
    for step in steps:
        step_type = step.get("type", step.get("step_type"))
        
        if step_type in verification_types:
            has_verification = True
            if has_mutation and not mutation_after_verify:
                # Verification after mutation is good
                pass
        
        if step_type in mutation_types:
            has_mutation = True
            if has_verification:
                mutation_after_verify = True
    
    # Check 20: Verification planned
    if has_mutation and not has_verification:
        failures.append(
            "Plan contains mutations but no verification step"
        )
    
    # Check 21: Verification after mutation
    if mutation_after_verify:
        failures.append(
            "Plan has mutation steps after verification (verify should be last)"
        )
    
    return failures


def _has_cycles(steps: list[dict[str, Any]]) -> bool:
    """Check if dependency graph has cycles using DFS."""
    step_ids = {s.get("id", s.get("step_id", "")) for s in steps}
    deps: dict[str, list[str]] = {}
    
    for step in steps:
        step_id = step.get("id", step.get("step_id", ""))
        deps[step_id] = step.get("depends_on", [])
    
    WHITE, GRAY, BLACK = 0, 1, 2
    colors = {s: WHITE for s in step_ids}
    
    def dfs(node: str) -> bool:
        if node not in colors:
            return False
        colors[node] = GRAY
        for dep in deps.get(node, []):
            if dep in colors:
                if colors[dep] == GRAY:
                    return True  # Cycle
                if colors[dep] == WHITE and dfs(dep):
                    return True
        colors[node] = BLACK
        return False
    
    for step_id in step_ids:
        if colors[step_id] == WHITE:
            if dfs(step_id):
                return True
    return False


# =============================================================================
# Soft Fail Conditions (warnings, but not auto-reject)
# =============================================================================

def _check_soft_conditions(
    plan: dict[str, Any],
    _config: PlanGateConfig,
) -> list[str]:
    """Check soft fail conditions.
    
    If TWO or more apply, the plan should be rewritten:
    - Steps are too coarse (large patches instead of minimal diffs)
    - Multiple unrelated changes in one step
    - Verification scope is too broad or too narrow
    - Reads could be reduced before mutation
    - Replanning is missing after expected failure points
    """
    warnings = []
    steps = plan.get("steps", [])
    
    # Check for coarse patches (large diffs)
    for step in steps:
        step_id = step.get("id", step.get("step_id", "unknown"))
        inputs = step.get("inputs", {})
        
        patch = inputs.get("patch", inputs.get("diff", ""))
        if isinstance(patch, str) and len(patch) > 2000:
            warnings.append(
                f"Step {step_id} has large patch ({len(patch)} chars) - consider splitting"
            )
        
        # Multiple files in one step
        files = inputs.get("files", [])
        if isinstance(files, list) and len(files) > 3:
            warnings.append(
                f"Step {step_id} touches {len(files)} files - consider splitting"
            )
    
    # Check for missing replan capability
    has_replan = any(
        s.get("type", s.get("step_type")) == "replan"
        for s in steps
    )
    if len(steps) > 5 and not has_replan:
        warnings.append(
            "Long plan without 'replan' step - consider adding replan checkpoint"
        )
    
    return warnings


# =============================================================================
# Main Critique Function
# =============================================================================

def critique_plan(
    plan: dict[str, Any],
    config: PlanGateConfig | None = None,
) -> CritiqueReport:
    """Apply self-critique rubric to a plan.
    
    This should be called BEFORE submitting to PlanGate.
    If the plan is REJECTED, it should be regenerated.
    If REWRITE_ADVISED, consider regenerating.
    
    Args:
        plan: The plan dictionary (JSON-like structure).
        config: Optional PlanGateConfig for validation parameters.
        
    Returns:
        CritiqueReport with result and any failures/warnings.
    """
    config = config or PlanGateConfig()
    
    hard_failures: list[str] = []
    
    # Run all hard checks
    hard_failures.extend(_check_structural_correctness(plan, config))
    hard_failures.extend(_check_gate_compatibility(plan, config))
    hard_failures.extend(_check_command_safety(plan, config))
    hard_failures.extend(_check_path_safety(plan, config))
    hard_failures.extend(_check_verification_discipline(plan, config))
    
    # Run soft checks
    soft_warnings = _check_soft_conditions(plan, config)
    
    # Determine result
    if hard_failures:
        result = CritiqueResult.REJECTED
        logger.warning(
            "Plan REJECTED by self-critique: %d hard failures",
            len(hard_failures)
        )
    elif len(soft_warnings) >= 2:
        result = CritiqueResult.REWRITE_ADVISED
        logger.info(
            "Plan REWRITE_ADVISED: %d soft warnings",
            len(soft_warnings)
        )
    else:
        result = CritiqueResult.APPROVED
        logger.info("Plan APPROVED by self-critique")
    
    return CritiqueReport(
        result=result,
        hard_failures=hard_failures,
        soft_warnings=soft_warnings,
    )


def validate_plan_json(plan_json: str) -> CritiqueReport:
    """Validate a plan from JSON string.
    
    Convenience wrapper for critique_plan.
    
    Args:
        plan_json: JSON string containing the plan.
        
    Returns:
        CritiqueReport with validation results.
    """
    try:
        plan = json.loads(plan_json)
    except json.JSONDecodeError as e:
        return CritiqueReport(
            result=CritiqueResult.REJECTED,
            hard_failures=[f"Invalid JSON: {e}"],
            soft_warnings=[],
        )
    
    return critique_plan(plan)
