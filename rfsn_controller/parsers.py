"""Helper functions to parse test output and error traces."""
from __future__ import annotations

import hashlib
import re

PYTEST_FAILED_RE = re.compile(r"^FAILED\s+(.+?)$", re.MULTILINE)
TRACE_FILE_RE = re.compile(r'File "([^"]+\.py)"')


def error_signature(stdout: str, stderr: str) -> str:
    """Compute a hash signature of the tail of the combined stdout/stderr.

    Args:
        stdout: Captured standard output.
        stderr: Captured standard error.

    Returns:
        A SHA256 hexdigest of the last 80,000 characters of stdout+stderr.
    """
    blob = (stdout or "") + "\n" + (stderr or "")
    blob = blob[-80_000:]
    return hashlib.sha256(blob.encode("utf-8", errors="ignore")).hexdigest()


def parse_pytest_failures(output: str, limit: int = 20) -> list[str]:
    """Extract a list of failing test identifiers from pytest output.

    Args:
        output: The combined stdout+stderr from a pytest run.
        limit: Maximum number of failing identifiers to return.

    Returns:
        A list of test identifiers (e.g. "path/to/test.py::test_func").
    """
    return PYTEST_FAILED_RE.findall(output or "")[:limit]


def parse_trace_files(output: str, limit: int = 20) -> list[str]:
    """Extract filenames from Python traceback lines in the output.

    Args:
        output: The combined stdout+stderr from a failing run.
        limit: Maximum number of filenames to return.

    Returns:
        A list of file paths referenced in tracebacks.
    """
    out: list[str] = []
    for m in TRACE_FILE_RE.finditer(output or ""):
        out.append(m.group(1))
        if len(out) >= limit:
            break
    return out


def normalize_test_path(failed_id: str) -> str:
    """Normalize a pytest test identifier to just the file path.

    For example, "python_testcases/test_x.py::test_y" becomes
    "python_testcases/test_x.py".
    """
    return failed_id.split("::", 1)[0].strip()
