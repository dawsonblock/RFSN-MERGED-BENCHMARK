"""Winner selection by score (not first-pass-wins).
from __future__ import annotations

Implements scoring-based selection for multiple successful patches:
- Smallest lines changed
- Fewest files changed
- Penalty for editing tests (configurable)
- Prefer changes in traceback files
"""

from dataclasses import dataclass


@dataclass
class PatchScore:
    """Score for a patch candidate."""

    diff_hash: str
    diff: str
    lines_changed: int
    files_changed: int
    test_files_edited: int
    traceback_files_edited: int
    total_score: float
    reason: str


def parse_diff_stats(diff: str) -> tuple[int, int, set[str]]:
    """Parse a diff to extract statistics.

    Args:
        diff: The unified diff string.

    Returns:
        Tuple of (lines_changed, files_changed, set_of_files)
    """
    files_changed = set()
    lines_added = 0
    lines_removed = 0

    for line in diff.split("\n"):
        if line.startswith("+++ b/") or line.startswith("--- a/"):
            # Extract file path
            parts = line.split("/")
            if len(parts) > 2:
                filepath = "/".join(parts[2:])
                files_changed.add(filepath)
        elif line.startswith("+") and not line.startswith("+++"):
            lines_added += 1
        elif line.startswith("-") and not line.startswith("---"):
            lines_removed += 1

    lines_changed = lines_added + lines_removed
    return lines_changed, len(files_changed), files_changed


def is_test_file(filepath: str) -> bool:
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


def score_patch(
    diff: str,
    diff_hash: str,
    traceback_files: set[str] | None = None,
    test_edit_penalty: float = 10.0,
    traceback_bonus: float = -5.0,
) -> PatchScore:
    """Score a patch candidate.

    Lower score is better.

    Scoring:
    - Base score = lines_changed + files_changed * 5
    - Penalty for editing test files
    - Bonus for editing traceback files

    Args:
        diff: The unified diff.
        diff_hash: Hash of the diff.
        traceback_files: Set of files mentioned in tracebacks.
        test_edit_penalty: Penalty multiplier for editing test files.
        traceback_bonus: Bonus for editing traceback files.

    Returns:
        PatchScore with computed score.
    """
    if traceback_files is None:
        traceback_files = set()

    lines_changed, files_changed, changed_files = parse_diff_stats(diff)

    # Count test files edited
    test_files_edited = sum(1 for f in changed_files if is_test_file(f))

    # Count traceback files edited
    traceback_files_edited = sum(1 for f in changed_files if f in traceback_files)

    # Base score: lines changed + penalty per file
    base_score = lines_changed + (files_changed * 5)

    # Apply penalties and bonuses
    test_penalty = test_files_edited * test_edit_penalty
    traceback_benefit = traceback_files_edited * traceback_bonus

    total_score = base_score + test_penalty + traceback_benefit

    # Build reason string
    reason_parts = [
        f"{lines_changed} lines changed",
        f"{files_changed} files changed",
    ]
    if test_files_edited > 0:
        reason_parts.append(f"{test_files_edited} test files edited (+{test_penalty})")
    if traceback_files_edited > 0:
        reason_parts.append(f"{traceback_files_edited} traceback files edited ({traceback_bonus})")

    reason = ", ".join(reason_parts)

    return PatchScore(
        diff_hash=diff_hash,
        diff=diff,
        lines_changed=lines_changed,
        files_changed=files_changed,
        test_files_edited=test_files_edited,
        traceback_files_edited=traceback_files_edited,
        total_score=total_score,
        reason=reason,
    )


def select_best_patch(
    candidates: list[tuple[str, float]],  # List of (diff, temperature)
    traceback_files: set[str] | None = None,
    test_edit_penalty: float = 10.0,
    traceback_bonus: float = -5.0,
) -> PatchScore | None:
    """Select the best patch from multiple successful candidates.

    Args:
        candidates: List of (diff, temperature) tuples.
        traceback_files: Set of files mentioned in tracebacks.
        test_edit_penalty: Penalty multiplier for editing test files.
        traceback_bonus: Bonus for editing traceback files.

    Returns:
        PatchScore for the best patch, or None if no candidates.
    """
    if not candidates:
        return None

    # Score all candidates
    scored_patches = []
    for diff, _temp in candidates:
        # Simple hash for diff
        diff_hash = str(hash(diff))
        score = score_patch(
            diff,
            diff_hash,
            traceback_files,
            test_edit_penalty,
            traceback_bonus,
        )
        scored_patches.append(score)

    # Select the one with the lowest score
    best = min(scored_patches, key=lambda s: s.total_score)

    return best


def select_best_patch_from_hashes(
    diff_hashes: dict[str, str],  # Map of diff_hash -> diff
    traceback_files: set[str] | None = None,
    test_edit_penalty: float = 10.0,
    traceback_bonus: float = -5.0,
) -> PatchScore | None:
    """Select the best patch from a map of diff hashes.

    Args:
        diff_hashes: Map of diff_hash -> diff.
        traceback_files: Set of files mentioned in tracebacks.
        test_edit_penalty: Penalty multiplier for editing test files.
        traceback_bonus: Bonus for editing traceback files.

    Returns:
        PatchScore for the best patch, or None if no candidates.
    """
    if not diff_hashes:
        return None

    # Score all candidates
    scored_patches = []
    for diff_hash, diff in diff_hashes.items():
        score = score_patch(
            diff,
            diff_hash,
            traceback_files,
            test_edit_penalty,
            traceback_bonus,
        )
        scored_patches.append(score)

    # Select the one with the lowest score
    best = min(scored_patches, key=lambda s: s.total_score)

    return best
