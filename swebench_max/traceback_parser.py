from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Set


_NODEID_RE = re.compile(r"^(?P<nodeid>[\w\-/\.]+\.py::[\w\-/:\.\[\]]+)$", re.M)
_FILELINE_RE = re.compile(r"^\s*File\s+\"(?P<path>[^\"]+\.py)\"\s*,\s*line\s+(?P<line>\d+)", re.M)
_PYTEST_FAIL_RE = re.compile(r"^(?P<path>[\w\-/\.]+\.py):(?P<line>\d+):\s*(?P<kind>Error|Failed|Failure|Exception)", re.M)
_IMPORT_ERR_RE = re.compile(r"(ModuleNotFoundError|ImportError):\s+No module named ['\"](?P<mod>[^'\"]+)['\"]")
_ATTR_ERR_RE = re.compile(r"AttributeError:\s+module\s+['\"][^'\"]+['\"]\s+has\s+no\s+attribute\s+['\"](?P<attr>[^'\"]+)['\"]")
_NAME_ERR_RE = re.compile(r"NameError:\s+name\s+['\"](?P<name>[^'\"]+)['\"]\s+is\s+not\s+defined")

@dataclass(frozen=True)
class FailureSignals:
    nodeids: List[str]
    paths: List[str]
    missing_modules: List[str]
    keywords: List[str]

def parse_failure_signals(text: str, repo_root: str = ".") -> FailureSignals:
    """
    Extracts:
      - pytest nodeids (tests/test_x.py::TestCls::test_name)
      - python file paths from tracebacks
      - missing modules from ImportError/ModuleNotFoundError
      - a few error keywords for heuristic matching
    """
    if not text:
        return FailureSignals([], [], [], [])

    nodeids: Set[str] = set(m.group("nodeid") for m in _NODEID_RE.finditer(text))

    paths: Set[str] = set()
    for m in _FILELINE_RE.finditer(text):
        p = m.group("path")
        # keep only repo-relative-ish python files
        if p.endswith(".py"):
            paths.add(p)

    for m in _PYTEST_FAIL_RE.finditer(text):
        p = m.group("path")
        if p.endswith(".py"):
            paths.add(p)

    missing: Set[str] = set()
    for m in _IMPORT_ERR_RE.finditer(text):
        missing.add(m.group("mod"))

    keywords: Set[str] = set()
    for m in _ATTR_ERR_RE.finditer(text):
        keywords.add(m.group("attr"))
    for m in _NAME_ERR_RE.finditer(text):
        keywords.add(m.group("name"))

    # keep it small / deterministic
    return FailureSignals(
        nodeids=sorted(nodeids)[:50],
        paths=sorted(paths)[:80],
        missing_modules=sorted(missing)[:20],
        keywords=sorted(keywords)[:30],
    )
