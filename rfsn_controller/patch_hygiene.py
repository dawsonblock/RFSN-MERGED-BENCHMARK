"""Patch hygiene gates for quality control.
from __future__ import annotations

Ensures patches meet quality standards before acceptance:
- Max lines changed
- Max files changed
- Forbidden directories
- Forbidden file patterns
- Test deletion protection
- Skip pattern detection
"""

import re

# Immutable control surfaces that must never be modified by automated patches.
#
# These files form the core safety boundary of the controller and sandbox.
IMMUTABLE_CONTROL_PATHS: set[str] = {
    "rfsn_controller/command_allowlist.py",
    "rfsn_controller/command_normalizer.py",
    "rfsn_controller/apt_whitelist.py",
    "rfsn_controller/url_validation.py",
    "rfsn_controller/sandbox.py",
    "rfsn_controller/tool_manager.py",
    "rfsn_controller/verifier.py",
    "rfsn_controller/patch_hygiene.py",
    "rfsn_controller/controller.py",
    "rfsn_controller/policy.py",
}


class PatchHygieneConfig:
    """Configuration for patch hygiene gates."""

    def __init__(
        self,
        max_lines_changed: int = 200,
        max_files_changed: int = 5,
        allow_test_deletion: bool = False,
        allow_test_modification: bool = False,
        allow_lockfile_changes: bool = False,
        language: str | None = None,
    ):
        self.max_lines_changed = max_lines_changed
        self.max_files_changed = max_files_changed
        # Forbidden dirs and patterns are always strict (non-configurable for security)
        self.forbidden_dirs = self._default_forbidden_dirs()
        self.forbidden_file_patterns = self._default_forbidden_patterns()
        self.allow_test_deletion = allow_test_deletion
        self.allow_test_modification = allow_test_modification
        self.allow_lockfile_changes = allow_lockfile_changes
        self.language = language

    @classmethod
    def for_repair_mode(cls, language: str | None = None) -> "PatchHygieneConfig":
        """Create a strict configuration for repair mode.

        Repair mode requires minimal changes to fix bugs:
        - Max 200 lines changed
        - Max 5 files changed
        - No test deletion
        - No test modification
        - No lockfile changes

        Args:
            language: Optional language for language-specific adjustments.
        """
        return cls(
            max_lines_changed=200,
            max_files_changed=5,
            allow_test_deletion=False,
            allow_test_modification=False,
            allow_lockfile_changes=False,
            language=language,
        )

    @classmethod
    def for_feature_mode(cls, language: str | None = None) -> "PatchHygieneConfig":
        """Create a more permissive configuration for feature mode.

        Feature mode allows larger changes for feature implementation:
        - Max 500 lines changed (allows scaffolding + implementation)
        - Max 15 files changed (allows multi-module features)
        - Allows test creation and modification
        - No test deletion (tests are deliverables)
        - No lockfile changes by default (must be explicitly allowed)

        Language-specific adjustments:
        - Java/C#: +200 lines (boilerplate-heavy)
        - Node.js: +100 lines (config files)

        Args:
            language: Optional language for language-specific adjustments.
        """
        max_lines = 500
        max_files = 15

        # Language-specific adjustments
        if language in ["java", "csharp", "dotnet"]:
            max_lines += 200  # Boilerplate-heavy languages
        elif language in ["node", "javascript", "typescript"]:
            max_lines += 100  # Package.json, config files

        return cls(
            max_lines_changed=max_lines,
            max_files_changed=max_files,
            allow_test_deletion=False,
            allow_test_modification=True,
            allow_lockfile_changes=False,
            language=language,
        )

    @classmethod
    def custom(
        cls,
        max_lines_changed: int,
        max_files_changed: int,
        allow_test_deletion: bool = False,
        allow_test_modification: bool = False,
        allow_lockfile_changes: bool = False,
        language: str | None = None,
    ) -> "PatchHygieneConfig":
        """Create a custom configuration with specific thresholds.

        Note: forbidden_dirs and forbidden_file_patterns are always strict
        and cannot be overridden for security reasons.

        Args:
            max_lines_changed: Maximum lines that can be changed.
            max_files_changed: Maximum files that can be changed.
            allow_test_deletion: Whether to allow test file deletion.
            allow_test_modification: Whether to allow test file modification.
            allow_lockfile_changes: Whether to allow lockfile changes.
            language: Optional language identifier.
        """
        return cls(
            max_lines_changed=max_lines_changed,
            max_files_changed=max_files_changed,
            allow_test_deletion=allow_test_deletion,
            allow_test_modification=allow_test_modification,
            allow_lockfile_changes=allow_lockfile_changes,
            language=language,
        )

    @staticmethod
    def _default_forbidden_dirs() -> set[str]:
        """Default directories that should never be modified."""
        return {
            "vendor/",
            "third_party/",
            "node_modules/",
            ".git/",
            "__pycache__/",
            ".venv/",
            "venv/",
            "env/",
            ".env",
            ".idea/",
            ".vscode/",
            "dist/",
            "build/",
            "target/",
            "bin/",
            "obj/",
        }

    @staticmethod
    def _default_forbidden_patterns() -> set[str]:
        """Default file patterns that should never be modified."""
        return {
            "package-lock.json",
            "yarn.lock",
            "pnpm-lock.yaml",
            "poetry.lock",
            "Pipfile.lock",
            "requirements.lock",
            "*.lock",
            ".env",
            ".env.*",
            "*.key",
            "*.pem",
            "id_rsa",
            "id_ed25519",
            "secrets.yml",
            "config/secrets",
        }


class PatchHygieneResult:
    """Result of patch hygiene validation."""

    def __init__(self, is_valid: bool, violations: list[str]):
        self.is_valid = is_valid
        self.violations = violations

    def __bool__(self):
        return self.is_valid


def validate_patch_hygiene(
    diff: str,
    config: PatchHygieneConfig | None = None,
) -> PatchHygieneResult:
    """Validate a patch against hygiene gates.

    Args:
        diff: The git diff to validate.
        config: Hygiene configuration (uses defaults if None).

    Returns:
        PatchHygieneResult with validation status and violations.
    """
    if config is None:
        config = PatchHygieneConfig()

    violations = []

    # Parse diff to extract changed files and line counts
    files_changed, lines_added, lines_removed = _parse_diff(diff)

    # Check if any immutable control paths were modified
    for filepath in files_changed:
        # Normalize path (remove leading ./ and convert backslashes)
        norm = filepath.replace("\\", "/").lstrip("./")
        if norm in IMMUTABLE_CONTROL_PATHS:
            violations.append(f"Forbidden change to immutable control file: {norm}")

    # Check max files changed
    if len(files_changed) > config.max_files_changed:
        violations.append(
            f"Too many files changed: {len(files_changed)} > {config.max_files_changed}"
        )

    # Check max lines changed
    total_lines = lines_added + lines_removed
    if total_lines > config.max_lines_changed:
        violations.append(f"Too many lines changed: {total_lines} > {config.max_lines_changed}")

    # Check forbidden directories
    for filepath in files_changed:
        for forbidden_dir in config.forbidden_dirs:
            if filepath.startswith(forbidden_dir):
                violations.append(f"Cannot modify files in {forbidden_dir}: {filepath}")

    # Define lockfile patterns
    # Note: This includes both explicit well-known lockfiles and any file ending in .lock
    # This intentionally covers custom lockfiles (e.g., custom-name.lock) to prevent
    # unintended dependency changes. If a .lock file should be modifiable, it should not
    # be named with the .lock extension.
    lockfile_patterns = {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "Pipfile.lock",
        "requirements.lock",
        "Cargo.lock",
        "go.sum",
    }

    # Check forbidden file patterns (excluding lockfiles if allowed)
    for filepath in files_changed:
        filename = filepath.split("/")[-1]

        # Check if this is a lockfile (explicit patterns OR any .lock file)
        is_lockfile = filename in lockfile_patterns or filename.endswith(".lock")

        # If lockfile changes are allowed, skip lockfile pattern checks
        if is_lockfile and config.allow_lockfile_changes:
            continue

        for pattern in config.forbidden_file_patterns:
            if pattern.startswith("*"):
                # Wildcard pattern
                if filename.endswith(pattern[1:]):
                    violations.append(f"Cannot modify file matching pattern {pattern}: {filepath}")
            elif pattern == filename or filepath == pattern:
                violations.append(f"Cannot modify file: {filepath}")

    # Check for test deletion
    if not config.allow_test_deletion:
        # Check if any file was deleted (diff shows +++ /dev/null)
        for line in diff.split("\n"):
            if line.startswith("+++ /dev/null"):
                # File was deleted - check if it was a test file
                # Find the corresponding --- a/ line
                for prev_line in diff.split("\n"):
                    if prev_line.startswith("--- a/"):
                        deleted_file = prev_line[6:]  # Remove '--- a/'
                        if _is_test_file(deleted_file):
                            violations.append(f"Cannot delete test file: {deleted_file}")
                        break

    # Check for test modification (if not allowed)
    if not config.allow_test_modification:
        for filepath in files_changed:
            if _is_test_file(filepath):
                violations.append(f"Cannot modify test file in repair mode: {filepath}")

    # Check for skip patterns in modified files (only if tests are being modified)
    if config.allow_test_modification:
        skip_patterns = [
            r"@pytest\.mark\.skip",
            r"@pytest\.mark\.xfail",
            r"@unittest\.skip",
            r"@unittest\.skipIf",
            r"@unittest\.skipUnless",
        ]

        for filepath in files_changed:
            if _is_test_file(filepath):
                for pattern in skip_patterns:
                    if re.search(pattern, diff):
                        violations.append(f"Test skip pattern detected in {filepath}: {pattern}")

    # Check for debug prints
    debug_patterns = [
        r'print\([\'"]debug',
        r'print\([\'"]DEBUG',
        r'print\([\'"]XXX',
        r"pprint\(",
        r"pdb\.set_trace",
        r"breakpoint\(",
    ]

    for pattern in debug_patterns:
        if re.search(pattern, diff):
            violations.append(f"Debug pattern detected: {pattern}")

    return PatchHygieneResult(len(violations) == 0, violations)


def _parse_diff(diff: str) -> tuple[set[str], int, int]:
    """Parse a git diff to extract changed files and line counts.

    Args:
        diff: The git diff string.

    Returns:
        (files_changed, lines_added, lines_removed) tuple.
    """
    files_changed = set()
    lines_added = 0
    lines_removed = 0

    for line in diff.split("\n"):
        if line.startswith("+++ b/"):
            # Extract file path from +++ b/path
            filepath = line[6:]  # Remove '+++ b/'
            if filepath != "/dev/null":
                files_changed.add(filepath)
        elif line.startswith("--- a/"):
            # Extract file path from --- a/path (for deleted files)
            filepath = line[6:]  # Remove '--- a/'
            # Only add if we haven't seen it from +++ b/ (deleted files)
            if filepath != "/dev/null":
                files_changed.add(filepath)
        elif line.startswith("+") and not line.startswith("+++"):
            lines_added += 1
        elif line.startswith("-") and not line.startswith("---"):
            lines_removed += 1

    return files_changed, lines_added, lines_removed


def _is_test_file(filepath: str) -> bool:
    """Check if a file is a test file.

    Args:
        filepath: The file path.

    Returns:
        True if this is a test file.
    """
    filename = filepath.lower()
    return (
        filename.startswith("test_")
        or filename.endswith("_test.py")
        or filename.endswith("_test.ts")
        or filename.endswith("_test.js")
        or filename.endswith(".test.py")
        or filename.endswith(".test.ts")
        or filename.endswith(".test.js")
        or "/test/" in filepath
        or "/tests/" in filepath
    )
