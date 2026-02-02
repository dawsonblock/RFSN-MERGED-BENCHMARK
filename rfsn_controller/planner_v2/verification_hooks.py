"""Verification Hooks - Standard verification recipes.

Provides a library of verification commands that the planner selects from
rather than inventing commands. This ensures consistent, tested verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class VerificationType(Enum):
    """Types of verification."""
    
    PYTEST_SUBSET = "pytest_subset"
    PYTEST_FULL = "pytest_full"
    MYPY_TARGET = "mypy_target"
    RUFF_CHECK = "ruff_check"
    UNIT_BY_TOUCHED = "unit_by_touched"
    BUILD_CHECK = "build_check"
    IMPORT_CHECK = "import_check"
    TYPE_CHECK = "type_check"
    SYNTAX_CHECK = "syntax_check"


@dataclass
class VerificationHook:
    """A verification hook with command template and metadata."""
    
    hook_type: VerificationType
    template: str
    description: str
    is_fast: bool  # Quick check vs full verification
    required_tools: list[str]  # Tools that must be available
    
    def format(self, **kwargs) -> str:
        """Format the template with provided values."""
        return self.template.format(**kwargs)


# Standard verification hooks library
VERIFICATION_HOOKS: dict[VerificationType, VerificationHook] = {
    VerificationType.PYTEST_SUBSET: VerificationHook(
        hook_type=VerificationType.PYTEST_SUBSET,
        template="pytest {files} -x -v --tb=short",
        description="Run pytest on specific files, fail fast",
        is_fast=True,
        required_tools=["pytest"],
    ),
    VerificationType.PYTEST_FULL: VerificationHook(
        hook_type=VerificationType.PYTEST_FULL,
        template="pytest -q",
        description="Run full pytest suite",
        is_fast=False,
        required_tools=["pytest"],
    ),
    VerificationType.MYPY_TARGET: VerificationHook(
        hook_type=VerificationType.MYPY_TARGET,
        template="mypy {files} --ignore-missing-imports",
        description="Type check specific files",
        is_fast=True,
        required_tools=["mypy"],
    ),
    VerificationType.RUFF_CHECK: VerificationHook(
        hook_type=VerificationType.RUFF_CHECK,
        template="ruff check {files}",
        description="Lint specific files with ruff",
        is_fast=True,
        required_tools=["ruff"],
    ),
    VerificationType.UNIT_BY_TOUCHED: VerificationHook(
        hook_type=VerificationType.UNIT_BY_TOUCHED,
        template="pytest {test_files} -x",
        description="Run tests related to touched files",
        is_fast=True,
        required_tools=["pytest"],
    ),
    VerificationType.BUILD_CHECK: VerificationHook(
        hook_type=VerificationType.BUILD_CHECK,
        template="{build_cmd}",
        description="Run build command",
        is_fast=False,
        required_tools=[],
    ),
    VerificationType.IMPORT_CHECK: VerificationHook(
        hook_type=VerificationType.IMPORT_CHECK,
        template="python -c 'import {module}'",
        description="Check module can be imported",
        is_fast=True,
        required_tools=["python"],
    ),
    VerificationType.TYPE_CHECK: VerificationHook(
        hook_type=VerificationType.TYPE_CHECK,
        template="python -m py_compile {files}",
        description="Check syntax of Python files",
        is_fast=True,
        required_tools=["python"],
    ),
    VerificationType.SYNTAX_CHECK: VerificationHook(
        hook_type=VerificationType.SYNTAX_CHECK,
        template="python -m py_compile {files}",
        description="Quick syntax verification",
        is_fast=True,
        required_tools=["python"],
    ),
}


class VerificationHooks:
    """Library of verification commands planner selects from."""
    
    @classmethod
    def get_hook(cls, hook_type: VerificationType) -> VerificationHook:
        """Get a verification hook by type."""
        return VERIFICATION_HOOKS[hook_type]
    
    @classmethod
    def get_fast_hooks(cls) -> list[VerificationHook]:
        """Get all fast verification hooks."""
        return [h for h in VERIFICATION_HOOKS.values() if h.is_fast]
    
    @classmethod
    def get_full_hooks(cls) -> list[VerificationHook]:
        """Get all full verification hooks."""
        return [h for h in VERIFICATION_HOOKS.values() if not h.is_fast]
    
    @classmethod
    def select_for_step(
        cls,
        step_id: str,
        touched_files: list[str],
        test_cmd: str,
        is_milestone: bool = False,
    ) -> str:
        """Select appropriate verification command for a step.
        
        Args:
            step_id: The step identifier.
            touched_files: Files modified in this step.
            test_cmd: Default test command.
            is_milestone: True if this is a milestone step requiring full verification.
            
        Returns:
            Formatted verification command.
        """
        if is_milestone:
            return test_cmd  # Full test suite at milestones
        
        # For non-milestones, use fast subset
        if touched_files:
            # Find related test files
            test_files = cls._find_related_tests(touched_files)
            if test_files:
                hook = VERIFICATION_HOOKS[VerificationType.UNIT_BY_TOUCHED]
                return hook.format(test_files=" ".join(test_files))
        
        # Fallback to fast pytest
        hook = VERIFICATION_HOOKS[VerificationType.PYTEST_SUBSET]
        files = " ".join(touched_files) if touched_files else "."
        return hook.format(files=files)
    
    @staticmethod
    def _find_related_tests(files: list[str]) -> list[str]:
        """Find test files related to source files.
        
        Heuristic: for each src/foo.py, look for test_foo.py.
        """
        test_files = []
        for f in files:
            if f.endswith(".py") and not f.startswith("test_"):
                base = f.rsplit("/", 1)[-1]
                name = base.replace(".py", "")
                test_files.append(f"test_{name}.py")
                test_files.append(f"tests/test_{name}.py")
        return test_files


class TestStrategy:
    """Manages when to run fast vs full tests."""
    
    # Steps that only need fast checks
    FAST_STEPS = [
        "analyze",
        "locate",
        "read",
        "understand",
        "design",
        "parse",
    ]
    
    # Steps that require full test suite
    MILESTONE_STEPS = [
        "implement",
        "add-tests",
        "verify-full",
        "verify-regression",
        "update-docs",
        "finalize",
    ]
    
    @classmethod
    def is_milestone(cls, step_id: str) -> bool:
        """Check if step is a milestone requiring full tests."""
        step_lower = step_id.lower()
        return any(m in step_lower for m in cls.MILESTONE_STEPS)
    
    @classmethod
    def is_fast_step(cls, step_id: str) -> bool:
        """Check if step only needs fast checks."""
        step_lower = step_id.lower()
        return any(f in step_lower for f in cls.FAST_STEPS)
    
    @classmethod
    def get_test_command(
        cls,
        step_id: str,
        full_cmd: str,
        touched_files: list[str] | None = None,
    ) -> str:
        """Get appropriate test command for step.
        
        Args:
            step_id: The step identifier.
            full_cmd: Full test command to use at milestones.
            touched_files: Files modified in this step.
            
        Returns:
            Test command to run.
        """
        if cls.is_milestone(step_id):
            return full_cmd
        
        if cls.is_fast_step(step_id):
            # Very fast: just syntax check
            if touched_files:
                return f"python -m py_compile {' '.join(touched_files)}"
            return "echo 'No verification needed for read-only step'"
        
        # Default: subset tests
        return VerificationHooks.select_for_step(
            step_id=step_id,
            touched_files=touched_files or [],
            test_cmd=full_cmd,
            is_milestone=False,
        )
