"""Enhanced goal types beyond just tests.
from __future__ import annotations

Supports multiple goal types: tests, build, lint, repro, static check, feature.
"""

from dataclasses import dataclass
from enum import Enum

# Default feature subgoals (used by both FeatureGoal and controller)
DEFAULT_FEATURE_SUBGOALS = [
    "scaffold: Create necessary file structure and boilerplate",
    "implement: Write core functionality",
    "tests: Add comprehensive tests",
    "docs: Update documentation",
]


class GoalType(Enum):
    """Types of goals the controller can satisfy."""

    TEST = "test"  # Tests pass
    BUILD = "build"  # Build succeeds
    LINT = "lint"  # Linting passes
    TYPECHECK = "typecheck"  # Type checking passes
    STATIC_CHECK = "static_check"  # Static analysis passes
    REPRO = "repro"  # Repro script exits 0
    CUSTOM = "custom"  # Custom command
    FEATURE = "feature"  # Feature implementation


@dataclass
class Goal:
    """A goal that the controller needs to satisfy."""

    goal_type: GoalType
    command: str
    description: str
    timeout: int = 300
    required: bool = True


@dataclass
class FeatureGoal:
    """A feature implementation goal with acceptance criteria."""

    description: str
    acceptance_criteria: list[str]
    subgoals: list[str] | None = None
    verification_commands: list[str] | None = None
    timeout: int = 600

    def __post_init__(self):
        """Initialize default subgoals if not provided and validate inputs."""
        # Validate description
        if not self.description or not self.description.strip():
            raise ValueError("Feature description cannot be empty")

        # Validate acceptance criteria
        if not self.acceptance_criteria:
            raise ValueError("At least one acceptance criterion is required")

        # Filter empty criteria - create new list to avoid mutating input
        filtered_criteria = [c for c in self.acceptance_criteria if c and c.strip()]
        if not filtered_criteria:
            raise ValueError("All acceptance criteria are empty")
        self.acceptance_criteria = filtered_criteria

        # Set default subgoals if not provided
        if self.subgoals is None:
            self.subgoals = list(DEFAULT_FEATURE_SUBGOALS)  # Create a copy

        # Validate timeout
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")


class GoalFactory:
    """Creates goals based on project type and user preferences."""

    @staticmethod
    def create_test_goal(
        command: str,
        timeout: int = 300,
    ) -> Goal:
        """Create a test goal.

        Args:
            command: Test command to run.
            timeout: Timeout in seconds.

        Returns:
            Goal instance.
        """
        return Goal(
            goal_type=GoalType.TEST,
            command=command,
            description="All tests pass",
            timeout=timeout,
            required=True,
        )

    @staticmethod
    def create_build_goal(
        command: str,
        timeout: int = 300,
        required: bool = True,
    ) -> Goal:
        """Create a build goal.

        Args:
            command: Build command to run.
            timeout: Timeout in seconds.
            required: Whether this goal must pass.

        Returns:
            Goal instance.
        """
        return Goal(
            goal_type=GoalType.BUILD,
            command=command,
            description="Build succeeds",
            timeout=timeout,
            required=required,
        )

    @staticmethod
    def create_lint_goal(
        command: str,
        timeout: int = 120,
        required: bool = False,
    ) -> Goal:
        """Create a lint goal.

        Args:
            command: Lint command to run.
            timeout: Timeout in seconds.
            required: Whether this goal must pass.

        Returns:
            Goal instance.
        """
        return Goal(
            goal_type=GoalType.LINT,
            command=command,
            description="Linting passes",
            timeout=timeout,
            required=required,
        )

    @staticmethod
    def create_typecheck_goal(
        command: str,
        timeout: int = 120,
        required: bool = False,
    ) -> Goal:
        """Create a typecheck goal.

        Args:
            command: Typecheck command to run.
            timeout: Timeout in seconds.
            required: Whether this goal must pass.

        Returns:
            Goal instance.
        """
        return Goal(
            goal_type=GoalType.TYPECHECK,
            command=command,
            description="Type checking passes",
            timeout=timeout,
            required=required,
        )

    @staticmethod
    def create_repro_goal(
        command: str,
        timeout: int = 300,
    ) -> Goal:
        """Create a repro goal.

        Args:
            command: Repro script command.
            timeout: Timeout in seconds.

        Returns:
            Goal instance.
        """
        return Goal(
            goal_type=GoalType.REPRO,
            command=command,
            description="Repro script succeeds",
            timeout=timeout,
            required=True,
        )

    @staticmethod
    def create_verify_goal(
        command: str,
        timeout: int = 300,
        required: bool = False,
    ) -> Goal:
        """Create a verification/smoke test goal.

        Args:
            command: Verification/smoke test command.
            timeout: Timeout in seconds.
            required: Whether this goal must pass.

        Returns:
            Goal instance.
        """
        return Goal(
            goal_type=GoalType.CUSTOM,
            command=command,
            description="Smoke test succeeds",
            timeout=timeout,
            required=required,
        )

    @staticmethod
    def create_custom_goal(
        command: str,
        description: str,
        timeout: int = 300,
        required: bool = True,
    ) -> Goal:
        """Create a custom goal.

        Args:
            command: Custom command to run.
            description: Description of the goal.
            timeout: Timeout in seconds.
            required: Whether this goal must pass.

        Returns:
            Goal instance.
        """
        return Goal(
            goal_type=GoalType.CUSTOM,
            command=command,
            description=description,
            timeout=timeout,
            required=required,
        )

    @staticmethod
    def create_feature_goal(
        description: str,
        acceptance_criteria: list[str],
        verification_commands: list[str] | None = None,
        timeout: int = 600,
    ) -> FeatureGoal:
        """Create a feature implementation goal.

        Args:
            description: Feature description.
            acceptance_criteria: List of acceptance criteria.
            verification_commands: Optional commands to verify feature.
            timeout: Timeout in seconds.

        Returns:
            FeatureGoal instance.
        """
        return FeatureGoal(
            description=description,
            acceptance_criteria=acceptance_criteria,
            verification_commands=verification_commands,
            timeout=timeout,
        )


@dataclass
class GoalSet:
    """A set of goals to satisfy."""

    primary_goal: Goal
    verification_goals: list[Goal]

    def get_all_goals(self) -> list[Goal]:
        """Get all goals (primary + verification).

        Returns:
            List of all goals.
        """
        return [self.primary_goal] + self.verification_goals

    def get_required_goals(self) -> list[Goal]:
        """Get only required goals.

        Returns:
            List of required goals.
        """
        return [g for g in self.get_all_goals() if g.required]


class GoalSetFactory:
    """Creates goal sets based on project type."""

    @staticmethod
    def for_python(
        test_cmd: str = "python -m pytest -q",
        lint_cmd: str | None = None,
        typecheck_cmd: str | None = None,
        repro_cmd: str | None = None,
        verify_cmd: str | None = None,
    ) -> GoalSet:
        """Create goal set for Python project.

        Args:
            test_cmd: Test command.
            lint_cmd: Optional lint command.
            typecheck_cmd: Optional typecheck command.
            repro_cmd: Optional repro command.
            verify_cmd: Optional smoke test command.

        Returns:
            GoalSet instance.
        """
        primary = GoalFactory.create_test_goal(test_cmd)

        verification = []
        if lint_cmd:
            verification.append(GoalFactory.create_lint_goal(lint_cmd, required=False))
        if typecheck_cmd:
            verification.append(GoalFactory.create_typecheck_goal(typecheck_cmd, required=False))
        if repro_cmd:
            verification.append(GoalFactory.create_repro_goal(repro_cmd))
        if verify_cmd:
            verification.append(GoalFactory.create_verify_goal(verify_cmd, required=False))

        return GoalSet(primary_goal=primary, verification_goals=verification)

    @staticmethod
    def for_node(
        test_cmd: str = "npm test",
        build_cmd: str | None = None,
        lint_cmd: str | None = None,
        verify_cmd: str | None = None,
    ) -> GoalSet:
        """Create goal set for Node.js project.

        Args:
            test_cmd: Test command.
            build_cmd: Optional build command.
            lint_cmd: Optional lint command.
            verify_cmd: Optional smoke test command.

        Returns:
            GoalSet instance.
        """
        primary = GoalFactory.create_test_goal(test_cmd)

        verification = []
        if build_cmd:
            verification.append(GoalFactory.create_build_goal(build_cmd, required=False))
        if lint_cmd:
            verification.append(GoalFactory.create_lint_goal(lint_cmd, required=False))
        if verify_cmd:
            verification.append(GoalFactory.create_verify_goal(verify_cmd, required=False))

        return GoalSet(primary_goal=primary, verification_goals=verification)

    @staticmethod
    def for_go(
        test_cmd: str = "go test ./...",
        build_cmd: str | None = None,
        verify_cmd: str | None = None,
    ) -> GoalSet:
        """Create goal set for Go project.

        Args:
            test_cmd: Test command.
            build_cmd: Optional build command.
            verify_cmd: Optional smoke test command.

        Returns:
            GoalSet instance.
        """
        primary = GoalFactory.create_test_goal(test_cmd)

        verification = []
        if build_cmd:
            verification.append(GoalFactory.create_build_goal(build_cmd, required=False))
        if verify_cmd:
            verification.append(GoalFactory.create_verify_goal(verify_cmd, required=False))

        return GoalSet(primary_goal=primary, verification_goals=verification)

    @staticmethod
    def for_rust(
        test_cmd: str = "cargo test",
        build_cmd: str | None = None,
        lint_cmd: str | None = None,
        verify_cmd: str | None = None,
    ) -> GoalSet:
        """Create goal set for Rust project.

        Args:
            test_cmd: Test command.
            build_cmd: Optional build command.
            lint_cmd: Optional lint command.
            verify_cmd: Optional smoke test command.

        Returns:
            GoalSet instance.
        """
        primary = GoalFactory.create_test_goal(test_cmd)

        verification = []
        if build_cmd:
            verification.append(GoalFactory.create_build_goal(build_cmd, required=False))
        if lint_cmd:
            verification.append(GoalFactory.create_lint_goal(lint_cmd, required=False))
        if verify_cmd:
            verification.append(GoalFactory.create_verify_goal(verify_cmd, required=False))

        return GoalSet(primary_goal=primary, verification_goals=verification)

    @staticmethod
    def for_java(
        test_cmd: str = "mvn test",
        build_cmd: str | None = None,
        verify_cmd: str | None = None,
    ) -> GoalSet:
        """Create goal set for Java project.

        Args:
            test_cmd: Test command.
            build_cmd: Optional build command.
            verify_cmd: Optional smoke test command.

        Returns:
            GoalSet instance.
        """
        primary = GoalFactory.create_test_goal(test_cmd)

        verification = []
        if build_cmd:
            verification.append(GoalFactory.create_build_goal(build_cmd, required=False))
        if verify_cmd:
            verification.append(GoalFactory.create_verify_goal(verify_cmd, required=False))

        return GoalSet(primary_goal=primary, verification_goals=verification)

    @staticmethod
    def for_dotnet(
        test_cmd: str = "dotnet test",
        build_cmd: str | None = None,
    ) -> GoalSet:
        """Create goal set for .NET project.

        Args:
            test_cmd: Test command.
            build_cmd: Optional build command.

        Returns:
            GoalSet instance.
        """
        primary = GoalFactory.create_test_goal(test_cmd)

        verification = []
        if build_cmd:
            verification.append(GoalFactory.create_build_goal(build_cmd, required=False))

        return GoalSet(primary_goal=primary, verification_goals=verification)

    @staticmethod
    def for_build_only(
        build_cmd: str,
        lint_cmd: str | None = None,
    ) -> GoalSet:
        """Create goal set for project without tests.

        Args:
            build_cmd: Build command.
            lint_cmd: Optional lint command.

        Returns:
            GoalSet instance.
        """
        primary = GoalFactory.create_build_goal(build_cmd)

        verification = []
        if lint_cmd:
            verification.append(GoalFactory.create_lint_goal(lint_cmd, required=False))

        return GoalSet(primary_goal=primary, verification_goals=verification)
