"""Planner Self-Critique for proposal quality gating.

Implements a deterministic rubric to score proposals before they reach
the kernel. Low-quality proposals are rejected early to save compute.

INVARIANTS:
1. Critique is deterministic (no LLM calls)
2. Rubric checks are configurable but strict
3. Score threshold is configurable (default: 0.5)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# Forbidden paths that should never be modified
IMMUTABLE_PATHS: set[str] = {
    "rfsn_controller/verification_manager.py",
    "rfsn_controller/command_allowlist.py",
    "rfsn_controller/sandbox.py",
    "rfsn_controller/controller.py",
    "rfsn_controller/patch_hygiene.py",
    ".git/",
}


@dataclass
class CritiqueIssue:
    """A single issue found during critique.
    
    Attributes:
        code: Short identifier for the issue type
        severity: Severity level (error, warning, info)
        message: Human-readable description
        location: Optional file/line reference
    """
    code: str
    severity: str  # error, warning, info
    message: str
    location: str | None = None
    
    def __str__(self) -> str:
        loc = f" at {self.location}" if self.location else ""
        return f"[{self.severity.upper()}] {self.code}: {self.message}{loc}"


@dataclass
class Critique:
    """Result of proposal critique.
    
    Attributes:
        score: Quality score in [0.0, 1.0]
        issues: List of issues found
        passed: Whether the proposal passed the threshold
        metadata: Additional critique metadata
    """
    score: float
    issues: list[CritiqueIssue] = field(default_factory=list)
    passed: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def error_count(self) -> int:
        """Count of error-level issues."""
        return sum(1 for i in self.issues if i.severity == "error")
    
    @property
    def warning_count(self) -> int:
        """Count of warning-level issues."""
        return sum(1 for i in self.issues if i.severity == "warning")


@dataclass
class CriticConfig:
    """Configuration for the proposal critic.
    
    Attributes:
        pass_threshold: Minimum score to pass (default: 0.5)
        max_lines_changed: Maximum lines allowed in diff
        max_files_changed: Maximum files allowed in diff
        allow_test_modification: Whether tests can be modified
        check_forbidden_paths: Whether to check immutable paths
    """
    pass_threshold: float = 0.5
    max_lines_changed: int = 200
    max_files_changed: int = 5
    allow_test_modification: bool = False
    check_forbidden_paths: bool = True


class PlannerCritic:
    """Deterministic critic for proposal quality gating.
    
    Evaluates proposals against a rubric of checks before they
    are submitted to the kernel. This saves compute by rejecting
    low-quality proposals early.
    """
    
    def __init__(self, config: CriticConfig | None = None):
        self.config = config or CriticConfig()
    
    def evaluate(self, proposal: dict[str, Any]) -> Critique:
        """Evaluate a proposal against the quality rubric.
        
        Args:
            proposal: The proposal dict containing at minimum:
                - diff: The unified diff string
                - files_changed: List of file paths (optional)
                
        Returns:
            Critique with score, issues, and pass/fail status
        """
        issues: list[CritiqueIssue] = []
        score = 1.0  # Start with perfect score, deduct for issues
        
        diff = proposal.get("diff", "")
        if not diff:
            issues.append(CritiqueIssue(
                code="EMPTY_DIFF",
                severity="error",
                message="Proposal contains no diff",
            ))
            return Critique(score=0.0, issues=issues, passed=False)
        
        # Parse diff to extract metrics
        files_changed, lines_added, lines_removed = self._parse_diff(diff)
        total_lines = lines_added + lines_removed
        
        # Check: Maximum lines changed
        if total_lines > self.config.max_lines_changed:
            issues.append(CritiqueIssue(
                code="TOO_MANY_LINES",
                severity="error",
                message=f"Diff changes {total_lines} lines, max is {self.config.max_lines_changed}",
            ))
            score -= 0.4
        
        # Check: Maximum files changed
        if len(files_changed) > self.config.max_files_changed:
            issues.append(CritiqueIssue(
                code="TOO_MANY_FILES",
                severity="error",
                message=f"Diff changes {len(files_changed)} files, max is {self.config.max_files_changed}",
            ))
            score -= 0.3
        
        # Check: Forbidden paths
        if self.config.check_forbidden_paths:
            for filepath in files_changed:
                norm_path = filepath.replace("\\", "/").lstrip("./")
                for forbidden in IMMUTABLE_PATHS:
                    if norm_path.startswith(forbidden) or norm_path == forbidden.rstrip("/"):
                        issues.append(CritiqueIssue(
                            code="FORBIDDEN_PATH",
                            severity="error",
                            message=f"Cannot modify immutable path: {norm_path}",
                            location=filepath,
                        ))
                        score -= 0.5
        
        # Check: Test file modification
        if not self.config.allow_test_modification:
            for filepath in files_changed:
                if self._is_test_file(filepath):
                    issues.append(CritiqueIssue(
                        code="TEST_MODIFICATION",
                        severity="error",
                        message="Test files cannot be modified in repair mode",
                        location=filepath,
                    ))
                    score -= 0.3
        
        # Check: Debug patterns in diff
        debug_patterns = [
            r'print\(["\']debug',
            r'print\(["\']DEBUG',
            r'pdb\.set_trace',
            r'breakpoint\(\)',
            r'import pdb',
        ]
        for pattern in debug_patterns:
            if re.search(pattern, diff):
                issues.append(CritiqueIssue(
                    code="DEBUG_CODE",
                    severity="warning",
                    message=f"Debug pattern detected: {pattern}",
                ))
                score -= 0.1
        
        # Check: Empty or trivial changes
        if total_lines == 0:
            issues.append(CritiqueIssue(
                code="NO_CHANGES",
                severity="error",
                message="Diff contains no actual changes",
            ))
            score = 0.0
        elif total_lines < 3 and len(files_changed) > 1:
            issues.append(CritiqueIssue(
                code="SUSPICIOUS_SPREAD",
                severity="warning",
                message="Very few changes spread across multiple files",
            ))
            score -= 0.1
        
        # Clamp score to valid range
        score = max(0.0, min(1.0, score))
        passed = score >= self.config.pass_threshold and self._no_critical_errors(issues)
        
        return Critique(
            score=score,
            issues=issues,
            passed=passed,
            metadata={
                "files_changed": list(files_changed),
                "lines_added": lines_added,
                "lines_removed": lines_removed,
                "total_lines": total_lines,
            },
        )
    
    def _parse_diff(self, diff: str) -> tuple[set[str], int, int]:
        """Parse a unified diff to extract metrics.
        
        Returns:
            (files_changed, lines_added, lines_removed)
        """
        files_changed: set[str] = set()
        lines_added = 0
        lines_removed = 0
        
        for line in diff.split("\n"):
            if line.startswith("+++ b/"):
                filepath = line[6:]
                if filepath != "/dev/null":
                    files_changed.add(filepath)
            elif line.startswith("--- a/"):
                filepath = line[6:]
                if filepath != "/dev/null":
                    files_changed.add(filepath)
            elif line.startswith("+") and not line.startswith("+++"):
                lines_added += 1
            elif line.startswith("-") and not line.startswith("---"):
                lines_removed += 1
        
        return files_changed, lines_added, lines_removed
    
    def _is_test_file(self, filepath: str) -> bool:
        """Check if a file is a test file."""
        filename = filepath.lower()
        return (
            filename.startswith("test_")
            or filename.endswith("_test.py")
            or filename.endswith(".test.py")
            or "/test/" in filepath
            or "/tests/" in filepath
        )
    
    def _no_critical_errors(self, issues: list[CritiqueIssue]) -> bool:
        """Check if there are any critical (blocking) errors."""
        critical_codes = {"EMPTY_DIFF", "FORBIDDEN_PATH", "NO_CHANGES"}
        return not any(i.code in critical_codes for i in issues if i.severity == "error")


def create_strict_critic() -> PlannerCritic:
    """Create a strict critic for production use."""
    return PlannerCritic(CriticConfig(
        pass_threshold=0.6,
        max_lines_changed=150,
        max_files_changed=3,
        allow_test_modification=False,
        check_forbidden_paths=True,
    ))


def create_lenient_critic() -> PlannerCritic:
    """Create a lenient critic for development/research."""
    return PlannerCritic(CriticConfig(
        pass_threshold=0.3,
        max_lines_changed=500,
        max_files_changed=15,
        allow_test_modification=True,
        check_forbidden_paths=True,
    ))
