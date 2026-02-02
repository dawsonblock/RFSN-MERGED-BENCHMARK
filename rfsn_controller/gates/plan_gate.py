"""Plan Gate - Hard Safety Enforcement for Plans.

This module provides the critical safety layer between the planner and controller.
The PlanGate validates plans BEFORE any execution occurs.

HARD SAFETY GUARANTEES:
1. Planner output is DATA (JSON), never executable code
2. Step types must be in an explicit allowlist
3. No raw shell execution (all tools go through registry)
4. File paths must be within workspace allowlist
5. Step budget is enforced
6. Dependency graph must be acyclic

The PlanGate is called:
- Once when a plan is proposed (full validation)
- Before each step execution (re-validation)

Learning systems CANNOT modify this gate.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Pattern to detect inline environment variables (e.g., FOO=bar cmd)
# These bypass normal command parsing and can enable injection
INLINE_ENV_PATTERN = re.compile(r'^[A-Z_][A-Z0-9_]*=\S+\s+\w+')


class PlanGateError(Exception):
    """Raised when a plan violates gate constraints."""
    
    def __init__(self, message: str, step_id: str | None = None):
        self.step_id = step_id
        super().__init__(f"[PlanGate] {message}" + (f" (step: {step_id})" if step_id else ""))


class StepGateError(Exception):
    """Raised when a step violates gate constraints."""
    
    def __init__(self, message: str, step_id: str):
        self.step_id = step_id
        super().__init__(f"[StepGate:{step_id}] {message}")


# Default allowed step types - these are SAFE operations
DEFAULT_ALLOWED_STEP_TYPES: set[str] = {
    # Read-only operations
    "search_repo",
    "read_file",
    "analyze_file",
    "list_directory",
    "grep_search",
    
    # Code modification (through sandboxed tools)
    "apply_patch",
    "add_test",
    "refactor_small",
    "fix_import",
    "fix_typing",
    
    # Verification (read results, don't mutate)
    "run_tests",
    "run_lint",
    "check_syntax",
    "validate_types",
    
    # Coordination (metadata only)
    "wait",
    "checkpoint",
    "replan",
}

# Patterns that indicate shell injection attempts
SHELL_INJECTION_PATTERNS = [
    "$(", "`",  # Command substitution
    "&&", "||", ";",  # Command chaining
    "|",  # Piping (if not in a safe context)
    ">", ">>", "<",  # Redirects
    "rm -rf", "rm -r", "rm -f",
    "chmod", "chown", "sudo",
    "curl", "wget", "nc", "netcat",
    "eval", "exec",
    "python -c", "python3 -c",
    "node -e", "ruby -e", "perl -e",
]

# Dangerous path patterns
FORBIDDEN_PATH_PATTERNS = [
    "..", "/etc", "/usr", "/bin", "/sbin",
    "/root", "/home", "/var", "/tmp",
    ".git/hooks", ".github/workflows",
    "__pycache__", "node_modules",
    ".env", ".secrets", "credentials",
]


@dataclass
class PlanGateConfig:
    """Configuration for PlanGate."""
    
    # Step type constraints
    allowed_step_types: set[str] = field(default_factory=lambda: DEFAULT_ALLOWED_STEP_TYPES.copy())
    max_steps: int = 12
    
    # Path constraints
    workspace_root: Path | None = None
    allowed_path_globs: list[str] = field(default_factory=lambda: ["**/*.py", "**/*.js", "**/*.ts", "**/*.md"])
    forbidden_path_patterns: list[str] = field(default_factory=lambda: FORBIDDEN_PATH_PATTERNS.copy())
    
    # Shell constraints (stricter is safer)
    allow_shell_in_tools: bool = False
    shell_injection_patterns: list[str] = field(default_factory=lambda: SHELL_INJECTION_PATTERNS.copy())
    
    # Validation strictness
    strict_mode: bool = True
    require_expected_outcomes: bool = True
    require_verification: bool = True


class PlanGate:
    """Hard safety gate for plans.
    
    This gate validates plans BEFORE execution and CANNOT be bypassed.
    Learning systems can suggest strategies, but this gate has final authority.
    
    Usage:
        gate = PlanGate(PlanGateConfig(workspace_root=Path("/repo")))
        
        # Validate entire plan before execution
        gate.validate_plan(plan_json)
        
        # Validate each step before execution
        gate.validate_step(step_dict)
    """
    
    def __init__(self, config: PlanGateConfig | None = None):
        """Initialize PlanGate.

        Args:
            config: Gate configuration. Uses defaults if not provided.
        """
        self.config = config or PlanGateConfig()
        # Freeze allowed step types to prevent runtime mutation. Convert to frozenset.
        self.config.allowed_step_types = frozenset(self.config.allowed_step_types)
        self._validated_plan_id: str | None = None
    
    def validate_plan(self, plan: dict[str, Any]) -> bool:
        """Validate an entire plan before execution.
        
        This MUST be called before any step execution.
        
        Args:
            plan: Plan dictionary (from JSON).
            
        Returns:
            True if plan passes all checks.
            
        Raises:
            PlanGateError: If plan violates any constraint.
        """
        logger.info("Validating plan: %s", plan.get("plan_id", "unknown"))
        
        # Check required fields
        if "steps" not in plan:
            raise PlanGateError("Plan missing 'steps' field")
        
        steps = plan["steps"]
        if not isinstance(steps, list):
            raise PlanGateError("Plan 'steps' must be a list")
        
        # Check step budget
        if len(steps) > self.config.max_steps:
            raise PlanGateError(
                f"Plan exceeds step budget: {len(steps)} > {self.config.max_steps}"
            )
        
        if len(steps) == 0:
            raise PlanGateError("Plan has no steps")
        
        # Validate each step
        seen_ids: set[str] = set()
        for step in steps:
            self._validate_step_structure(step, seen_ids)
            seen_ids.add(step.get("id", step.get("step_id", "")))
        
        # Check dependency graph
        self._validate_dependency_graph(steps)
        
        # Record validated plan
        self._validated_plan_id = plan.get("plan_id")
        
        logger.info("Plan validated successfully: %d steps", len(steps))
        return True
    
    def validate_step(self, step: dict[str, Any]) -> bool:
        """Validate a single step before execution.
        
        Called for each step, even after plan validation,
        because state changes can invalidate earlier assumptions.
        
        Args:
            step: Step dictionary.
            
        Returns:
            True if step passes all checks.
            
        Raises:
            StepGateError: If step violates any constraint.
        """
        step_id = step.get("id", step.get("step_id", "unknown"))
        
        # Check step type
        step_type = step.get("type", step.get("step_type"))
        if step_type not in self.config.allowed_step_types:
            raise StepGateError(
                f"Step type '{step_type}' not in allowlist. "
                f"Allowed: {sorted(self.config.allowed_step_types)}",
                step_id
            )
        
        # Check for shell injection in inputs
        self._check_shell_injection(step, step_id)

        # Validate argv safety if present
        self._check_step_argv(step, step_id)

        # Check path safety
        self._check_path_safety(step, step_id)

        return True
    
    def _validate_step_structure(self, step: dict[str, Any], seen_ids: set[str]) -> None:
        """Validate step structure and required fields."""
        step_id = step.get("id", step.get("step_id"))
        
        if not step_id:
            raise PlanGateError("Step missing 'id' field")
        
        if step_id in seen_ids:
            raise PlanGateError(f"Duplicate step id: {step_id}", step_id)
        
        # Check step type
        step_type = step.get("type", step.get("step_type"))
        if not step_type:
            raise PlanGateError("Step missing 'type' field", step_id)
        
        if step_type not in self.config.allowed_step_types:
            raise PlanGateError(
                f"Step type '{step_type}' not in allowlist",
                step_id
            )
        
        # Check for shell injection
        self._check_shell_injection(step, step_id)
        
        # Check path safety
        self._check_path_safety(step, step_id)
        
        # Check for expected outcome
        if self.config.require_expected_outcomes:
            if "expected_outcome" not in step and "success_criteria" not in step:
                if self.config.strict_mode:
                    raise PlanGateError(
                        "Step missing 'expected_outcome' or 'success_criteria'",
                        step_id
                    )
                else:
                    logger.warning("Step %s missing expected outcome", step_id)
    
    def _validate_dependency_graph(self, steps: list[dict[str, Any]]) -> None:
        """Validate that dependency graph is acyclic."""
        # Build adjacency list
        step_ids = {s.get("id", s.get("step_id", "")) for s in steps}
        deps: dict[str, list[str]] = {}
        
        for step in steps:
            step_id = step.get("id", step.get("step_id", ""))
            step_deps = step.get("depends_on", [])
            
            # Validate deps exist
            for dep in step_deps:
                if dep not in step_ids:
                    raise PlanGateError(
                        f"Step depends on unknown step: {dep}",
                        step_id
                    )
            
            deps[step_id] = step_deps
        
        # DFS cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {s: WHITE for s in step_ids}
        
        def dfs(node: str) -> bool:
            colors[node] = GRAY
            for dep in deps.get(node, []):
                if colors[dep] == GRAY:
                    return True  # Cycle detected
                if colors[dep] == WHITE and dfs(dep):
                    return True
            colors[node] = BLACK
            return False
        
        for step_id in step_ids:
            if colors[step_id] == WHITE:
                if dfs(step_id):
                    raise PlanGateError("Dependency graph contains cycles")
    
    def _check_shell_injection(self, step: dict[str, Any], step_id: str) -> None:
        """Check for shell injection patterns in step inputs."""
        inputs = step.get("inputs", {})
        
        # Serialize inputs for pattern matching
        inputs_str = json.dumps(inputs).lower()
        
        for pattern in self.config.shell_injection_patterns:
            if pattern.lower() in inputs_str:
                raise StepGateError(
                    f"Potential shell injection detected: '{pattern}'",
                    step_id
                )
        
        # Check for inline environment variables (FOO=bar cmd)
        # These bypass normal command parsing and can enable injection
        for _key, val in inputs.items():
            if isinstance(val, str) and INLINE_ENV_PATTERN.match(val):
                raise StepGateError(
                    f"Inline env var not allowed: {val[:50]}",
                    step_id
                )
        
        # Check for raw shell key
        if "shell" in inputs and not self.config.allow_shell_in_tools:
            raise StepGateError("Raw shell execution forbidden", step_id)
        
        if "command" in inputs and not self.config.allow_shell_in_tools:
            # Allow only if command is in a safe tool registry
            cmd = inputs.get("command", "")
            if any(p in cmd.lower() for p in self.config.shell_injection_patterns):
                raise StepGateError(
                    f"Unsafe command in inputs: {cmd[:50]}...",
                    step_id
                )
    
    def _check_path_safety(self, step: dict[str, Any], step_id: str) -> None:
        """Check that file paths are within allowed workspace."""
        inputs = step.get("inputs", {})
        
        # Collect all path-like values
        paths_to_check: list[str] = []
        
        for key in ["file", "path", "target_file", "files", "allowed_files"]:
            val = inputs.get(key)
            if isinstance(val, str):
                paths_to_check.append(val)
            elif isinstance(val, list):
                paths_to_check.extend(v for v in val if isinstance(v, str))
        
        for path_str in paths_to_check:
            # Check for forbidden patterns
            for pattern in self.config.forbidden_path_patterns:
                if pattern in path_str:
                    raise StepGateError(
                        f"Forbidden path pattern: '{pattern}' in '{path_str}'",
                        step_id
                    )
            
            # Check path is within workspace
            if self.config.workspace_root:
                try:
                    path = Path(path_str)
                    if path.is_absolute():
                        resolved = path.resolve()
                        if not str(resolved).startswith(str(self.config.workspace_root.resolve())):
                            raise StepGateError(
                                f"Path outside workspace: {path_str}",
                                step_id
                            )
                except Exception as e:
                    logger.warning("Path check failed for %s: %s", path_str, e)

    def _check_step_argv(self, step: dict[str, Any], step_id: str) -> None:
        """Validate argv in a step to prevent shell and exec-string abuse.

        If a step contains an "argv" field, validate that it is a list of strings,
        disallow shell interpreters entirely, and disallow interpreter flags that
        execute arbitrary code strings.
        """
        argv = step.get("argv")
        if argv is None:
            return
        # Ensure argv is a non-empty list of strings
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
            raise StepGateError("Invalid argv (must be list[str])", step_id)
        base_cmd = Path(argv[0]).name
        # Deny shell interpreters regardless of executor config
        if base_cmd in {"sh", "bash", "dash", "zsh", "ksh"}:
            raise StepGateError("Shell interpreters are not allowed", step_id)
        # Disallow interpreter flags that execute code strings
        deny_flags = {
            "python": {"-c"},
            "python3": {"-c"},
            "node": {"-e", "-p"},
            "ruby": {"-e"},
            "perl": {"-e"},
        }.get(base_cmd, set())
        for a in argv[1:]:
            if a in deny_flags:
                raise StepGateError(f"Disallowed flag for {base_cmd}: {a}", step_id)
    
    # register_step_type has been removed to enforce immutability of allowed step types
    
    def get_allowed_step_types(self) -> set[str]:
        """Get the set of allowed step types (read-only)."""
        return self.config.allowed_step_types.copy()
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize gate config for audit logging."""
        return {
            "allowed_step_types": sorted(self.config.allowed_step_types),
            "max_steps": self.config.max_steps,
            "strict_mode": self.config.strict_mode,
            "workspace_root": str(self.config.workspace_root) if self.config.workspace_root else None,
            "allow_shell_in_tools": self.config.allow_shell_in_tools,
        }


def validate_learned_artifact(artifact: dict[str, Any]) -> bool:
    """Check if a learned artifact attempts to modify gates.
    
    This is called when loading learned policies. If the policy
    attempts to change allowlists or budgets, it is REJECTED.
    
    Args:
        artifact: Learned policy artifact.
        
    Returns:
        False if artifact attempts gate modification, True otherwise.
    """
    FORBIDDEN_KEYS = {
        "allowed_step_types",
        "max_steps",
        "forbidden_path_patterns",
        "shell_injection_patterns",
        "workspace_root",
        "allow_shell_in_tools",
        "disable_validation",
        "skip_gate",
        "bypass",
    }
    
    for key in FORBIDDEN_KEYS:
        if key in artifact:
            logger.warning(
                "LEARNED_POLICY_REJECTED: Artifact attempts to modify '%s'",
                key
            )
            return False
    
    return True
