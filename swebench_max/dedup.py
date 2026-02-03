import hashlib
import re
from typing import Set

def normalize_patch(patch: str) -> str:
    # strip timestamps and index lines that vary
    lines = []
    for ln in patch.splitlines():
        if ln.startswith("index "):
            continue
        # normalize file headers
        if ln.startswith("--- ") or ln.startswith("+++ "):
            lines.append(ln.split("\t")[0])
        else:
            lines.append(ln)
    return "\n".join(lines).strip() + "\n"

def patch_hash(patch: str) -> str:
    n = normalize_patch(patch).encode("utf-8", errors="replace")
    return hashlib.sha256(n).hexdigest()

class PatchDeduper:
    def __init__(self):
        self.seen: Set[str] = set()

    def add(self, patch: str) -> bool:
        h = patch_hash(patch)
        if h in self.seen:
            return False
        self.seen.add(h)
        return True
