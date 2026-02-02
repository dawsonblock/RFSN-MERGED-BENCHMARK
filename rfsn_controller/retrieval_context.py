"""Retrieval context builder for prompt grounding.

Uses RepoIndex and failure output to surface relevant files and symbols
for LLM prompts. Produces a concise text block with likely files and
symbols to consider, improving patch quality by focusing the model on
the right code.
"""

from __future__ import annotations

import re

from .repo_index import RepoIndex

_RE_PY_FILE = re.compile(r'File "([^"]+)"')
_RE_PY_FUNC = re.compile(r"in ([A-Za-z_][A-Za-z0-9_]*)")
_RE_ASSERT_HINT = re.compile(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)


def _uniq_preserve(items: list[str]) -> list[str]:
    """Deduplicate while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _normalize_path(path: str) -> str:
    """Normalize absolute-ish paths to repository relative paths.

    Removes common prefixes and leading './' to align with RepoIndex paths.
    """
    path = path.replace("\\", "/")
    for marker in ("/repo/", "/workspace/", "/project/", "/app/"):
        if marker in path:
            return path.split(marker, 1)[1]
    if path.startswith("./"):
        return path[2:]
    return path


def build_retrieval_context(
    repo_index: RepoIndex,
    failure_output: str,
    *,
    max_files: int = 8,
    max_symbols: int = 12,
) -> str:
    """Build a retrieval context block from failure output.

    Extracts file paths and symbol names from stack traces and asserts,
    searches the repository index for matching symbols, and returns a
    formatted text block with likely files, candidate symbols, and top
    symbol hits with file locations.

    Args:
        repo_index: RepoIndex instance for the current repository.
        failure_output: Concatenated stdout/stderr from failing tests.
        max_files: Maximum number of file paths to include.
        max_symbols: Maximum number of symbol hits to include.

    Returns:
        A string containing the retrieval context.
    """
    # Parse file paths from stack traces
    files = [_normalize_path(p) for p in _RE_PY_FILE.findall(failure_output)]
    files = [f for f in files if f and not f.startswith("<")]  # filter out <unknown> etc.
    files = _uniq_preserve(files)[:max_files]

    # Parse candidate symbols from traces and assert statements
    syms = _RE_PY_FUNC.findall(failure_output)
    syms += _RE_ASSERT_HINT.findall(failure_output)
    syms = _uniq_preserve([s for s in syms if len(s) >= 3])

    # Collect top symbol hits from repo index
    sym_hits = []
    for s in syms:
        for hit in repo_index.search_symbols(s):
            sym_hits.append(hit)
            if len(sym_hits) >= max_symbols:
                break
        if len(sym_hits) >= max_symbols:
            break

    # Build retrieval context text
    parts: list[str] = []

    if files:
        parts.append("LIKELY_FILES_FROM_TRACE:")
        for f in files:
            parts.append(f"  - {f}")
        parts.append("")

    if syms:
        parts.append("SYMBOL_CANDIDATES_FROM_TRACE:")
        for s in syms[:max_symbols]:
            parts.append(f"  - {s}")
        parts.append("")

    if sym_hits:
        parts.append("TOP_SYMBOL_HITS_IN_REPO_INDEX:")
        for hit in sym_hits:
            parts.append(f"  - {hit.kind} {hit.name}  ({hit.file}:{hit.line})")
        parts.append("")

    # Suggest next reads: trace files first, then files from symbol hits
    suggested: list[str] = []
    for f in files:
        suggested.append(f)
    for hit in sym_hits:
        if hit.file not in suggested:
            suggested.append(hit.file)
        if len(suggested) >= max_files:
            break
    if suggested:
        parts.append("SUGGESTED_NEXT_READS:")
        for f in suggested[:max_files]:
            parts.append(f"  - {f}")

    return "\n".join(parts).strip()