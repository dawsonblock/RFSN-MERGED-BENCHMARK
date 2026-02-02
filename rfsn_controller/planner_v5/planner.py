"""
from __future__ import annotations
Proposal Planner for RFSN v5.

Generates structured proposals following the strict RFSN protocol.
Never executes code - only outputs proposal JSON.
"""

import re
from uuid import uuid4

from .proposal import (
    ActionType,
    ExpectedEffect,
    Proposal,
    ProposalIntent,
    RiskLevel,
    Target,
    TestExpectation,
)
from .state_tracker import StateTracker


class ProposalPlanner:
    """
    Generates individual proposals based on evidence and state.

    This is the "cautious junior engineer" layer that:
    - Grounds all proposals in evidence
    - Prefers minimal scope
    - Never bundles multiple fixes
    - Expects rejection and recovers cleanly
    """

    def __init__(self, state: StateTracker):
        """
        Initialize planner with state tracker.

        Args:
            state: Current planning state
        """
        self.state = state

    def propose_reproduce(self, test_nodeid: str | None = None) -> Proposal:
        """
        Generate proposal to reproduce the failure.

        This is always the first proposal if we don't have reproduction yet.

        Args:
            test_nodeid: Specific test to run, or None for broader suite

        Returns:
            Proposal to run tests
        """
        if test_nodeid:
            target_path = test_nodeid
            hypothesis = f"Running the specific failing test {test_nodeid} will reproduce the reported error deterministically."
            change_summary = "Run the failing test nodeid to capture traceback and failure context."
        else:
            target_path = "tests/"
            hypothesis = "Running the test suite will identify which tests are failing and capture error details."
            change_summary = "Run test suite to capture all failures and their contexts."

        return Proposal(
            proposal_id=uuid4(),
            intent=ProposalIntent.TEST,
            hypothesis=hypothesis,
            action_type=ActionType.RUN_TESTS,
            target=Target(path=target_path),
            change_summary=change_summary,
            expected_effect=ExpectedEffect(
                tests=TestExpectation.FAIL,
                behavior="Produces traceback, assertion details, and failure context for analysis.",
            ),
            risk_level=RiskLevel.LOW,
            rollback_plan="No changes made; nothing to roll back.",
        )

    def propose_localize_file(self, file_path: str, reason: str) -> Proposal:
        """
        Generate proposal to read a file for localization.

        Args:
            file_path: File to read
            reason: Why this file is relevant

        Returns:
            Proposal to read file
        """
        return Proposal(
            proposal_id=uuid4(),
            intent=ProposalIntent.ANALYZE,
            hypothesis=f"The failure originates in {file_path}; inspecting the code will reveal the fault. {reason}",
            action_type=ActionType.READ_FILE,
            target=Target(path=file_path),
            change_summary=f"Read {file_path} to understand the implementation and identify the fault.",
            expected_effect=ExpectedEffect(
                tests=TestExpectation.UNCHANGED,
                behavior="Provides code context to craft a minimal fix.",
            ),
            risk_level=RiskLevel.LOW,
            rollback_plan="No changes made; nothing to roll back.",
        )

    def propose_search_repo(self, pattern: str, reason: str) -> Proposal:
        """
        Generate proposal to search repository for a pattern.

        Args:
            pattern: Search pattern (function name, error message, etc.)
            reason: Why this search is relevant

        Returns:
            Proposal to search
        """
        return Proposal(
            proposal_id=uuid4(),
            intent=ProposalIntent.ANALYZE,
            hypothesis=f"Searching for '{pattern}' will locate the relevant code. {reason}",
            action_type=ActionType.SEARCH_REPO,
            target=Target(path=".", symbol=pattern),
            change_summary=f"Search repository for pattern '{pattern}' to locate relevant code.",
            expected_effect=ExpectedEffect(
                tests=TestExpectation.UNCHANGED,
                behavior=f"Identifies files and locations containing '{pattern}'.",
            ),
            risk_level=RiskLevel.LOW,
            rollback_plan="No changes made; nothing to roll back.",
        )

    def propose_add_guard(
        self,
        file_path: str,
        symbol: str,
        guard_type: str,
        expected_behavior: str,
    ) -> Proposal:
        """
        Generate proposal to add defensive guard (None check, boundary, etc.).

        Args:
            file_path: File to modify
            symbol: Function/class to guard
            guard_type: Type of guard (none_check, boundary_check, type_check)
            expected_behavior: What should happen with the guard

        Returns:
            Proposal to add guard
        """
        guard_descriptions = {
            "none_check": "Add None-check before accessing attributes",
            "boundary_check": "Add boundary validation for indices/ranges",
            "type_check": "Add type validation with isinstance",
        }

        change_summary = guard_descriptions.get(
            guard_type, f"Add {guard_type} guard to {symbol}"
        )

        return Proposal(
            proposal_id=uuid4(),
            intent=ProposalIntent.REPAIR,
            hypothesis=f"If {symbol} can receive invalid input, adding a {guard_type} will prevent the error and {expected_behavior}.",
            action_type=ActionType.EDIT_FILE,
            target=Target(path=file_path, symbol=symbol),
            change_summary=change_summary,
            expected_effect=ExpectedEffect(
                tests=TestExpectation.PASS,
                behavior=expected_behavior,
            ),
            risk_level=RiskLevel.LOW,
            rollback_plan=f"Revert the guard condition in {symbol} at {file_path}.",
        )

    def propose_fix_logic_error(
        self,
        file_path: str,
        symbol: str,
        error_description: str,
        fix_description: str,
        expected_behavior: str,
    ) -> Proposal:
        """
        Generate proposal to fix a logic error.

        Args:
            file_path: File to modify
            symbol: Function/class with the error
            error_description: What's wrong
            fix_description: What to change
            expected_behavior: Expected outcome

        Returns:
            Proposal to fix logic
        """
        return Proposal(
            proposal_id=uuid4(),
            intent=ProposalIntent.REPAIR,
            hypothesis=f"{error_description}. Fixing this in {symbol} will {expected_behavior}.",
            action_type=ActionType.EDIT_FILE,
            target=Target(path=file_path, symbol=symbol),
            change_summary=fix_description,
            expected_effect=ExpectedEffect(
                tests=TestExpectation.PASS,
                behavior=expected_behavior,
            ),
            risk_level=RiskLevel.LOW,
            rollback_plan=f"Revert logic change in {symbol} at {file_path}.",
        )

    def propose_verify_targeted(
        self, test_nodeid: str, after_fix: str
    ) -> Proposal:
        """
        Generate proposal to verify a targeted test after fix.

        Args:
            test_nodeid: Specific test to run
            after_fix: Description of what was fixed

        Returns:
            Proposal to run test
        """
        return Proposal(
            proposal_id=uuid4(),
            intent=ProposalIntent.TEST,
            hypothesis=f"The targeted test {test_nodeid} will pass if {after_fix}.",
            action_type=ActionType.RUN_TESTS,
            target=Target(path=test_nodeid),
            change_summary="Re-run the specific test to confirm the fix.",
            expected_effect=ExpectedEffect(
                tests=TestExpectation.PASS,
                behavior="Test passes and no new failures appear in this scope.",
            ),
            risk_level=RiskLevel.LOW,
            rollback_plan="If test fails, revert the fix and re-localize.",
        )

    def propose_expand_verification(
        self, test_path: str, scope: str
    ) -> Proposal:
        """
        Generate proposal to expand test verification after local pass.

        Args:
            test_path: Path to test file or directory
            scope: Description of scope (e.g., "related module tests")

        Returns:
            Proposal to run broader tests
        """
        return Proposal(
            proposal_id=uuid4(),
            intent=ProposalIntent.TEST,
            hypothesis=f"Running {scope} will catch regressions introduced by the fix.",
            action_type=ActionType.RUN_TESTS,
            target=Target(path=test_path),
            change_summary=f"Run {scope} to verify no regressions.",
            expected_effect=ExpectedEffect(
                tests=TestExpectation.PASS,
                behavior="No regressions in adjacent behavior.",
            ),
            risk_level=RiskLevel.LOW,
            rollback_plan="If regressions occur, narrow fix behavior or add compatibility layer.",
        )

    def propose_explain_stuck(self, reason: str) -> Proposal:
        """
        Generate proposal explaining why planner is stuck.

        Args:
            reason: Why progress is blocked

        Returns:
            Proposal documenting stuck state
        """
        return Proposal(
            proposal_id=uuid4(),
            intent=ProposalIntent.ANALYZE,
            hypothesis=f"No further safe proposals can be generated. {reason}",
            action_type=ActionType.READ_FILE,
            target=Target(path="."),
            change_summary=f"Document stuck state: {reason}",
            expected_effect=ExpectedEffect(
                tests=TestExpectation.UNCHANGED,
                behavior="Makes explicit that planner cannot proceed safely.",
            ),
            risk_level=RiskLevel.LOW,
            rollback_plan="No changes made; nothing to roll back.",
        )

    def extract_traceback_file(self, traceback_text: str) -> str | None:
        """
        Extract the first project file from traceback.

        Args:
            traceback_text: Full traceback output

        Returns:
            First non-vendor file path, or None
        """
        # Look for lines like:
        #   File "path/to/file.py", line 123, in function_name
        pattern = r'File "([^"]+)", line (\d+), in (\w+)'
        matches = re.findall(pattern, traceback_text)

        for file_path, lineno, func in matches:
            # Skip vendor/stdlib files
            if any(
                skip in file_path
                for skip in ["/site-packages/", "/lib/python", "/usr/lib"]
            ):
                continue
            return file_path

        return None

    def extract_exception_type(self, output: str) -> str | None:
        """
        Extract exception type from test output.

        Args:
            output: Test output text

        Returns:
            Exception class name, or None
        """
        # Look for lines like: AttributeError: ...
        pattern = r"^(\w+Error|Exception): "
        matches = re.findall(pattern, output, re.MULTILINE)
        if matches:
            return matches[0]
        return None

    def parse_pytest_nodeid(self, output: str) -> str | None:
        """
        Extract pytest nodeid from failure output.

        Args:
            output: Pytest output

        Returns:
            Test nodeid like tests/test_x.py::test_y
        """
        # Look for FAILED tests/test_file.py::test_name
        pattern = r"FAILED\s+([\w/._-]+::[\w_-]+)"
        matches = re.findall(pattern, output)
        if matches:
            return matches[0]
        return None
