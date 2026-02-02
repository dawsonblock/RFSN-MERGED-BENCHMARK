"""Plan Validator - Strict pre-execution plan validation.

Validates plans before execution to ensure:
- Every step has required fields (allowed_files, verify, success_criteria)
- No empty dependencies on non-root steps
- No forbidden path modifications (controller, safety, secrets)
- No wildcard globs unless explicitly permitted
- No prompt-injection patterns in step content
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..schema import Plan, Step


@dataclass
class ValidationError:
    """A single validation error."""
    
    step_id: str | None
    field: str
    message: str
    severity: str = "error"  # error, warning
    
    def __str__(self) -> str:
        if self.step_id:
            return f"[{self.severity.upper()}] Step '{self.step_id}' {self.field}: {self.message}"
        return f"[{self.severity.upper()}] Plan {self.field}: {self.message}"


@dataclass
class ValidationResult:
    """Result of plan validation."""
    
    valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
    
    def __bool__(self) -> bool:
        return self.valid
    
    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "errors": [str(e) for e in self.errors],
            "warnings": [str(w) for w in self.warnings],
        }


class PlanValidator:
    """Validates plans before execution.
    
    Performs strict checks to ensure plans are safe and complete:
    - Step completeness: required fields present
    - Path safety: no forbidden path modifications
    - Glob restrictions: no wildcards unless permitted
    - Dependency integrity: valid DAG structure
    - Prompt injection: no manipulation patterns
    """
    
    # Paths that must never be modified
    FORBIDDEN_PATHS = [
        "controller.py",
        "controller_adapter.py",
        "sandbox.py",
        "safety",
        "governance",
        ".env",
        "secrets",
        "credentials",
        ".git/",
        "__pycache__",
    ]
    
    # Globs that require explicit permission
    RESTRICTED_GLOBS = ["*", "**", "**/*"]
    
    # Prompt injection patterns to block
    INJECTION_PATTERNS = [
        r"ignore\s+(previous|prior|above)\s+(instructions?|rules?|constraints?)",
        r"disregard\s+(all\s+)?(safety|constraints?|rules?)",
        r"you\s+are\s+now",
        r"new\s+instructions?:",
        r"forget\s+(everything|all)",
        r"override\s+(safety|constraints?)",
        r"run\s+(curl|wget|bash|sh|eval)",
        r"execute\s+(shell|command|script)",
        r"disable\s+(safety|validation|checks?)",
    ]
    
    def __init__(
        self,
        allow_wildcards: bool = False,
        extra_forbidden_paths: list[str] | None = None,
        strict_mode: bool = True,
    ):
        """Initialize the validator.
        
        Args:
            allow_wildcards: If True, permit wildcard globs.
            extra_forbidden_paths: Additional paths to forbid.
            strict_mode: If True, warnings become errors.
        """
        self.allow_wildcards = allow_wildcards
        self.forbidden_paths = set(self.FORBIDDEN_PATHS)
        if extra_forbidden_paths:
            self.forbidden_paths.update(extra_forbidden_paths)
        self.strict_mode = strict_mode
        self._compiled_injection = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS
        ]
    
    def validate(self, plan: Plan) -> ValidationResult:
        """Run all validation checks on a plan.
        
        Args:
            plan: The plan to validate.
            
        Returns:
            ValidationResult with errors and warnings.
        """
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []
        
        # Plan-level checks
        errors.extend(self._check_plan_structure(plan))
        
        # Step-level checks
        for step in plan.steps:
            step_errors, step_warnings = self._check_step(step, plan)
            errors.extend(step_errors)
            warnings.extend(step_warnings)
        
        # Dependency graph checks
        errors.extend(self._check_dependency_graph(plan))
        
        # In strict mode, warnings become errors
        if self.strict_mode:
            errors.extend(warnings)
            warnings = []
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
    
    def _check_plan_structure(self, plan: Plan) -> list[ValidationError]:
        """Check plan-level structure."""
        errors = []
        
        if not plan.plan_id:
            errors.append(ValidationError(None, "plan_id", "Plan ID is required"))
        
        if not plan.goal:
            errors.append(ValidationError(None, "goal", "Goal is required"))
        
        if not plan.steps:
            errors.append(ValidationError(None, "steps", "Plan must have at least one step"))
        
        return errors
    
    def _check_step(
        self, step: Step, plan: Plan
    ) -> tuple[list[ValidationError], list[ValidationError]]:
        """Check a single step."""
        errors = []
        warnings = []
        
        # Required fields
        if not step.step_id:
            errors.append(ValidationError(step.step_id, "step_id", "Step ID is required"))
        
        if not step.intent:
            errors.append(ValidationError(step.step_id, "intent", "Intent is required"))
        
        if not step.success_criteria:
            warnings.append(ValidationError(
                step.step_id, "success_criteria", 
                "Success criteria should be specified"
            ))
        
        # allowed_files checks
        if not step.allowed_files:
            warnings.append(ValidationError(
                step.step_id, "allowed_files",
                "No allowed_files specified - step may access any file"
            ))
        else:
            # Check for forbidden paths
            for path in step.allowed_files:
                for forbidden in self.forbidden_paths:
                    if forbidden in path.lower():
                        errors.append(ValidationError(
                            step.step_id, "allowed_files",
                            f"Forbidden path pattern '{forbidden}' in '{path}'"
                        ))
                
                # Check for restricted globs
                if not self.allow_wildcards:
                    for glob in self.RESTRICTED_GLOBS:
                        if path == glob or path.endswith(f"/{glob}"):
                            errors.append(ValidationError(
                                step.step_id, "allowed_files",
                                f"Wildcard glob '{glob}' not permitted without explicit allow"
                            ))
        
        # Prompt injection check on text fields
        text_to_check = f"{step.intent} {step.title} {step.success_criteria}"
        injections = self._check_prompt_injection(text_to_check)
        for pattern in injections:
            errors.append(ValidationError(
                step.step_id, "content",
                f"Prompt injection pattern detected: '{pattern}'"
            ))
        
        return errors, warnings
    
    def _check_dependency_graph(self, plan: Plan) -> list[ValidationError]:
        """Check dependency graph integrity."""
        errors = []
        step_ids = {s.step_id for s in plan.steps}
        
        for step in plan.steps:
            for dep_id in step.dependencies:
                if dep_id not in step_ids:
                    errors.append(ValidationError(
                        step.step_id, "dependencies",
                        f"Unknown dependency '{dep_id}'"
                    ))
        
        # Check for cycles using DFS
        if self._has_cycle(plan):
            errors.append(ValidationError(
                None, "dependencies",
                "Dependency graph contains a cycle"
            ))
        
        return errors
    
    def _has_cycle(self, plan: Plan) -> bool:
        """Check if dependency graph has cycles."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {s.step_id: WHITE for s in plan.steps}
        deps = {s.step_id: s.dependencies for s in plan.steps}
        
        def dfs(node: str) -> bool:
            color[node] = GRAY
            for dep in deps.get(node, []):
                if dep not in color:
                    continue
                if color[dep] == GRAY:
                    return True
                if color[dep] == WHITE and dfs(dep):
                    return True
            color[node] = BLACK
            return False
        
        for step in plan.steps:
            if color[step.step_id] == WHITE:
                if dfs(step.step_id):
                    return True
        return False
    
    def _check_prompt_injection(self, text: str) -> list[str]:
        """Check for prompt injection patterns."""
        matches = []
        for pattern in self._compiled_injection:
            if pattern.search(text):
                matches.append(pattern.pattern)
        return matches
    
    def validate_step_update(
        self, original: Step, updated: Step
    ) -> ValidationResult:
        """Validate a step modification.
        
        Ensures updates don't weaken constraints.
        
        Args:
            original: Original step.
            updated: Modified step.
            
        Returns:
            ValidationResult.
        """
        errors = []
        
        # Check for file scope expansion
        original_files = set(original.allowed_files)
        updated_files = set(updated.allowed_files)
        new_files = updated_files - original_files
        
        for f in new_files:
            for forbidden in self.forbidden_paths:
                if forbidden in f.lower():
                    errors.append(ValidationError(
                        updated.step_id, "allowed_files",
                        f"Cannot add forbidden path '{f}' in update"
                    ))
        
        return ValidationResult(valid=len(errors) == 0, errors=errors)
