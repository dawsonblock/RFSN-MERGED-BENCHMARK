from __future__ import annotations
import os
from typing import List, Set, Dict, Any

from swebench_max.candidate import DiffStats
from swebench_max.traceback_parser import parse_failure_signals
from swebench_max.import_graph import build_import_graph, reverse_closure


def _tests_under(repo_root: str) -> List[str]:
    td = os.path.join(repo_root, "tests")
    if not os.path.isdir(td):
        return []
    out = []
    for root, _, files in os.walk(td):
        for f in files:
            if f.endswith(".py") and (f.startswith("test") or f.endswith("_test.py")):
                out.append(os.path.relpath(os.path.join(root, f), repo_root).replace("\\", "/"))
    return out


def targeted_tests_v2(
    diff: DiffStats,
    repo_root: str,
    failures_text: str,
    limit: int,
    import_depth: int = 2,
) -> List[str]:
    """
    Deterministic, cheap, higher signal than basename matching.

    Priority order:
      1) Any explicit nodeids in failures (run those exact tests)
      2) Tests that mention failing paths (same directory / substring match)
      3) Import-graph expansion: find modules importing changed/failing modules,
         then pick tests that import/mention them (heuristic)
    """
    cmds: List[str] = []
    seen: Set[str] = set()

    sig = parse_failure_signals(failures_text or "", repo_root=repo_root)

    # 1) exact nodeids
    for nodeid in sig.nodeids:
        cmd = f"pytest -q {nodeid}"
        if cmd not in seen:
            seen.add(cmd)
            cmds.append(cmd)
        if len(cmds) >= limit:
            return cmds[:limit]

    # Candidate paths: from diff + failures
    candidate_paths = set(diff.paths)
    for p in sig.paths:
        # normalize to repo relative if possible
        p = p.replace("\\", "/")
        if p.startswith(repo_root.replace("\\", "/")):
            p = os.path.relpath(p, repo_root).replace("\\", "/")
        if p.endswith(".py"):
            candidate_paths.add(p)

    tests = _tests_under(repo_root)

    # 2) path proximity
    for t in tests:
        for p in list(candidate_paths)[:80]:
            # same dir or filename substring
            if os.path.dirname(t) == os.path.dirname(p) or os.path.basename(p) in t:
                cmd = f"pytest -q {t}"
                if cmd not in seen:
                    seen.add(cmd)
                    cmds.append(cmd)
                break
        if len(cmds) >= limit:
            return cmds[:limit]

    # 3) import graph expansion
    try:
        graph = build_import_graph(repo_root)
    except Exception:
        return cmds[:limit]

    seed_mods: Set[str] = set()
    for p in candidate_paths:
        m = graph.file_to_mod.get(p)
        if m:
            seed_mods.add(m)

    # missing modules from import errors (best-effort seeds)
    for m in sig.missing_modules:
        seed_mods.add(m.split(".")[0])

    closure = reverse_closure(graph, seed_mods, depth=import_depth, cap=1500)

    # Heuristic: pick tests whose file module is in closure OR whose content imports seed
    # Keep deterministic by sorting.
    picked: List[str] = []
    for t in sorted(tests):
        tm = graph.file_to_mod.get(t)
        if tm and (tm in closure):
            picked.append(t)

    # If too few, also match by keyword tokens from failures (AttributeError/NameError)
    if len(picked) < max(5, limit // 2) and sig.keywords:
        kws = set(sig.keywords)
        for t in sorted(tests):
            if t in picked:
                continue
            abspath = os.path.join(repo_root, t)
            try:
                with open(abspath, "r", encoding="utf-8") as fp:
                    src = fp.read(6000)
                if any(k in src for k in kws):
                    picked.append(t)
            except Exception:
                continue

    for t in picked:
        cmd = f"pytest -q {t}"
        if cmd not in seen:
            seen.add(cmd)
            cmds.append(cmd)
        if len(cmds) >= limit:
            break

    return cmds[:limit]
