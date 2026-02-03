import re
from typing import List
from swebench_max.candidate import DiffStats

_DIFF_FILE_RE = re.compile(r"^\+\+\+\s+b/(.+)$")

def compute_diff_stats(patch: str) -> DiffStats:
    paths: List[str] = []
    files_changed = 0
    lines_changed = 0

    for line in patch.splitlines():
        m = _DIFF_FILE_RE.match(line)
        if m:
            files_changed += 1
            paths.append(m.group(1))

        if line.startswith("+") and not line.startswith("+++"):
            lines_changed += 1
        elif line.startswith("-") and not line.startswith("---"):
            lines_changed += 1

    return DiffStats(files_changed=files_changed, lines_changed=lines_changed, paths=paths)
