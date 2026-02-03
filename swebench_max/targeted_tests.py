import os
from typing import List, Set
from swebench_max.candidate import DiffStats

def targeted_tests(diff: DiffStats, repo_root: str, limit: int) -> List[str]:
    """
    Heuristic test selection. Upstream only.
    Looks for tests likely related to touched files.
    """
    test_cmds: List[str] = []
    seen: Set[str] = set()

    # Prefer exact module basename match
    basenames = {os.path.splitext(os.path.basename(p))[0] for p in diff.paths}

    tests_dir = os.path.join(repo_root, "tests")
    if not os.path.isdir(tests_dir):
        return []

    for root, _, files in os.walk(tests_dir):
        for f in files:
            if not f.startswith("test") or not f.endswith(".py"):
                continue
            base = os.path.splitext(f)[0]
            score = 0
            for b in basenames:
                if b and b in base:
                    score += 2
            if score == 0:
                continue

            rel = os.path.relpath(os.path.join(root, f), repo_root)
            cmd = f"pytest -q {rel}"
            if cmd not in seen:
                seen.add(cmd)
                test_cmds.append(cmd)
            if len(test_cmds) >= limit:
                return test_cmds

    return test_cmds[:limit]
