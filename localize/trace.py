"""Trace-driven localization - parse stack traces to find suspect files.

This is Layer 1 (highest confidence signal).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from localize.types import LocalizationHit

try:
    from rfsn_controller.structured_logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def parse_python_traceback(traceback: str, repo_dir: Path) -> List[LocalizationHit]:
    """Parse Python traceback to extract file locations.
    
    Args:
        traceback: Full traceback text
        repo_dir: Repository root directory
        
    Returns:
        List of localization hits from traceback
        
    Example:
        >>> tb = 'File "main.py", line 42, in foo\\n    x = y.bar()\\nAttributeError'
        >>> hits = parse_python_traceback(tb, Path("/repo"))
    """
    hits = []
    
    # Pattern: File "path/to/file.py", line 42, in function_name
    pattern = r'File "([^"]+)", line (\d+)'
    
    for match in re.finditer(pattern, traceback):
        file_path = match.group(1)
        line_num = int(match.group(2))
        
        # Skip stdlib/site-packages
        if "site-packages" in file_path or "/usr/" in file_path:
            continue
        
        # Make relative to repo
        if file_path.startswith("/"):
            try:
                file_path = str(Path(file_path).relative_to(repo_dir))
            except ValueError:
                # Not in repo
                continue
        
        # Extract surrounding context
        snippet = _extract_snippet(traceback, match.start())
        
        hit = LocalizationHit(
            file=file_path,
            span=(max(1, line_num - 5), line_num + 5),
            score=1.0,  # Highest confidence
            evidence_type="trace",
            evidence_text=snippet,
            why=f"Stack trace points to line {line_num}",
            snippet=snippet,
            confidence=0.95,
        )
        hits.append(hit)
        
        logger.debug(
            "Trace hit",
            file=file_path,
            line=line_num,
        )
    
    return hits


def _extract_snippet(text: str, pos: int, context: int = 100) -> str:
    """Extract snippet around position."""
    start = max(0, pos - context)
    end = min(len(text), pos + context)
    return text[start:end].strip()


def parse_test_failures(test_output: str, repo_dir: Path) -> List[LocalizationHit]:
    """Parse test failure output to find suspects.
    
    Args:
        test_output: Test output (pytest, unittest, etc.)
        repo_dir: Repository root
        
    Returns:
        List of localization hits
    """
    hits = []
    
    # pytest pattern: tests/test_foo.py::test_bar FAILED
    pytest_pattern = r'([\w/]+\.py)::([\w_]+)\s+FAILED'
    
    for match in re.finditer(pytest_pattern, test_output):
        file_path = match.group(1)
        test_name = match.group(2)
        
        hit = LocalizationHit(
            file=file_path,
            span=(1, 1000),  # Whole file
            score=0.8,
            evidence_type="trace",
            evidence_text=f"Test {test_name} failed",
            why=f"Test failure in {test_name}",
            confidence=0.7,
        )
        hits.append(hit)
    
    # Also parse any tracebacks in the output
    hits.extend(parse_python_traceback(test_output, repo_dir))
    
    return hits
