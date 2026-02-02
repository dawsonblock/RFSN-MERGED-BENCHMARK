"""Diff minimization utilities for patch preprocessing.
from __future__ import annotations

Implements diff shrinking strategies:
- Drop whitespace-only hunks
- Drop import reordering (unless adding new imports)
- Drop comment-only changes (unless docstrings for modified functions)
- Detect speculative edits (changes to files not in error trace)
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MinimizedDiff:
    """Result of diff minimization."""

    original: str
    minimized: str
    dropped_hunks: int
    formatting_only_lines: int
    speculative_files: list[str] = field(default_factory=list)

    @property
    def reduction_ratio(self) -> float:
        """Ratio of size reduction (0.0 = no change, 1.0 = fully removed)."""
        if not self.original:
            return 0.0
        return 1.0 - (len(self.minimized) / len(self.original))


@dataclass
class DiffHunk:
    """Represents a single hunk in a diff."""

    header: str  # @@ -start,count +start,count @@
    lines: list[str]
    file_path: str
    start_line_old: int
    start_line_new: int

    @property
    def is_whitespace_only(self) -> bool:
        """True if hunk only changes whitespace."""
        for line in self.lines:
            if line.startswith("+") or line.startswith("-"):
                content = line[1:].strip()
                if content:  # Non-empty after stripping
                    return False
        return True

    @property
    def is_comment_only(self) -> bool:
        """True if hunk only adds/removes comments."""
        for line in self.lines:
            if line.startswith("+") or line.startswith("-"):
                content = line[1:].strip()
                if content and not content.startswith("#") and not content.startswith("//"):
                    # Check for multi-line comment patterns
                    if not (
                        content.startswith("/*")
                        or content.startswith("*/")
                        or content.startswith("*")
                        or content.startswith('"""')
                        or content.startswith("'''")
                    ):
                        return False
        return True

    @property
    def is_import_only(self) -> bool:
        """True if hunk only changes import statements."""
        for line in self.lines:
            if line.startswith("+") or line.startswith("-"):
                content = line[1:].strip()
                if content:
                    if not (
                        content.startswith("import ")
                        or content.startswith("from ")
                        or content.startswith("#")  # Allow comment lines in import block
                        or content == ""
                    ):
                        return False
        return True

    @property
    def adds_new_imports(self) -> bool:
        """True if hunk adds imports that weren't there before."""
        added_imports = set()
        removed_imports = set()

        for line in self.lines:
            content = line[1:].strip()
            if line.startswith("+") and (
                content.startswith("import ") or content.startswith("from ")
            ):
                added_imports.add(content)
            elif line.startswith("-") and (
                content.startswith("import ") or content.startswith("from ")
            ):
                removed_imports.add(content)

        # Net new imports = added - removed
        return bool(added_imports - removed_imports)


class DiffMinimizer:
    """Shrink diffs by removing non-essential changes."""

    def __init__(
        self,
        *,
        drop_whitespace: bool = True,
        drop_comments: bool = True,
        drop_import_reorder: bool = True,
        trace_files: set[str] | None = None,
    ):
        """Initialize minimizer with filtering options.

        Args:
            drop_whitespace: Drop whitespace-only hunks.
            drop_comments: Drop comment-only hunks.
            drop_import_reorder: Drop import reordering (keeps new imports).
            trace_files: Files mentioned in error trace (for speculative detection).
        """
        self.drop_whitespace = drop_whitespace
        self.drop_comments = drop_comments
        self.drop_import_reorder = drop_import_reorder
        self.trace_files = trace_files or set()

    def minimize(self, diff: str) -> MinimizedDiff:
        """Minimize a diff by removing non-essential hunks.

        Args:
            diff: Git diff string.

        Returns:
            MinimizedDiff with original, minimized, and statistics.
        """
        if not diff:
            return MinimizedDiff(
                original="",
                minimized="",
                dropped_hunks=0,
                formatting_only_lines=0,
            )

        files_and_hunks = self._parse_diff(diff)
        kept_parts: list[str] = []
        dropped_hunks = 0
        formatting_only_lines = 0
        speculative_files: list[str] = []

        for file_path, file_header, hunks in files_and_hunks:
            kept_hunks: list[DiffHunk] = []

            for hunk in hunks:
                should_drop, reason = self._should_drop_hunk(hunk)
                if should_drop:
                    dropped_hunks += 1
                    formatting_only_lines += len(
                        [ln for ln in hunk.lines if ln.startswith("+") or ln.startswith("-")]
                    )
                    logger.debug("Dropping hunk in %s: %s", file_path, reason)
                else:
                    kept_hunks.append(hunk)

            # Track speculative files
            if self.trace_files and file_path not in self.trace_files:
                if kept_hunks:  # Only flag if we're keeping changes
                    speculative_files.append(file_path)

            # Reconstruct file diff if any hunks kept
            if kept_hunks:
                kept_parts.append(file_header)
                for hunk in kept_hunks:
                    kept_parts.append(hunk.header)
                    kept_parts.extend(hunk.lines)

        minimized = "\n".join(kept_parts)
        if minimized and not minimized.endswith("\n"):
            minimized += "\n"

        return MinimizedDiff(
            original=diff,
            minimized=minimized,
            dropped_hunks=dropped_hunks,
            formatting_only_lines=formatting_only_lines,
            speculative_files=speculative_files,
        )

    def _should_drop_hunk(self, hunk: DiffHunk) -> tuple[bool, str]:
        """Determine if a hunk should be dropped.

        Returns:
            Tuple of (should_drop, reason).
        """
        if self.drop_whitespace and hunk.is_whitespace_only:
            return True, "whitespace-only"

        if self.drop_comments and hunk.is_comment_only:
            return True, "comment-only"

        if self.drop_import_reorder and hunk.is_import_only:
            if not hunk.adds_new_imports:
                return True, "import-reorder-only"

        return False, ""

    def _parse_diff(self, diff: str) -> list[tuple[str, str, list[DiffHunk]]]:
        """Parse a git diff into files and hunks.

        Returns:
            List of (file_path, file_header, hunks) tuples.
        """
        result: list[tuple[str, str, list[DiffHunk]]] = []
        lines = diff.split("\n")

        current_file: str | None = None
        current_file_header: list[str] = []
        current_hunks: list[DiffHunk] = []
        current_hunk: DiffHunk | None = None

        hunk_header_pattern = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")

        for line in lines:
            # New file header
            if line.startswith("diff --git"):
                # Save previous file
                if current_file is not None:
                    if current_hunk:
                        current_hunks.append(current_hunk)
                    result.append(
                        (current_file, "\n".join(current_file_header), current_hunks)
                    )

                current_file_header = [line]
                current_hunks = []
                current_hunk = None
                current_file = None

            elif line.startswith("--- a/") or line.startswith("--- /dev/null"):
                current_file_header.append(line)

            elif line.startswith("+++ b/"):
                current_file_header.append(line)
                current_file = line[6:]  # Extract path after '+++ b/'

            elif line.startswith("+++ /dev/null"):
                current_file_header.append(line)
                # File deletion

            elif line.startswith("@@"):
                # Save previous hunk
                if current_hunk:
                    current_hunks.append(current_hunk)

                match = hunk_header_pattern.match(line)
                start_old = int(match.group(1)) if match else 1
                start_new = int(match.group(2)) if match else 1

                current_hunk = DiffHunk(
                    header=line,
                    lines=[],
                    file_path=current_file or "",
                    start_line_old=start_old,
                    start_line_new=start_new,
                )

            elif current_hunk is not None:
                current_hunk.lines.append(line)

            elif current_file_header:
                # Other header lines (index, rename, etc.)
                current_file_header.append(line)

        # Save last file/hunk
        if current_file is not None:
            if current_hunk:
                current_hunks.append(current_hunk)
            result.append((current_file, "\n".join(current_file_header), current_hunks))

        return result

    def split_independent(self, diff: str) -> list[str]:
        """Split a diff into independent per-file diffs.

        Args:
            diff: Git diff string.

        Returns:
            List of single-file diffs.
        """
        files_and_hunks = self._parse_diff(diff)
        result: list[str] = []

        for file_path, file_header, hunks in files_and_hunks:
            if hunks:
                parts = [file_header]
                for hunk in hunks:
                    parts.append(hunk.header)
                    parts.extend(hunk.lines)
                file_diff = "\n".join(parts)
                if not file_diff.endswith("\n"):
                    file_diff += "\n"
                result.append(file_diff)

        return result

    def detect_speculative_edits(self, diff: str) -> list[str]:
        """Detect files changed that aren't in the error trace.

        Args:
            diff: Git diff string.

        Returns:
            List of file paths that appear speculative.
        """
        if not self.trace_files:
            return []

        files_and_hunks = self._parse_diff(diff)
        speculative = []

        for fpath, _, hunks in files_and_hunks:
            if hunks and fpath not in self.trace_files:
                speculative.append(fpath)

        return speculative
