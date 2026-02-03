from typing import List

RISKY_TOKENS = [
    "subprocess", "os.system", "eval(", "exec(", "pickle", "yaml.load",
    "requests.", "socket", "pty", "shlex", "chmod", "chown"
]

def static_risk_score(patch: str, forbid_prefixes: List[str]) -> float:
    """
    Negative = worse. This is upstream ranking only.
    Gate still enforces hard forbids/limits.
    """
    risk = 0.0
    lower = patch.lower()

    for t in RISKY_TOKENS:
        if t in lower:
            risk -= 1.0

    for pref in forbid_prefixes:
        if f" b/{pref}".lower() in lower or f" a/{pref}".lower() in lower:
            risk -= 10.0

    # Penalize very large patches
    lines = patch.splitlines()
    if len(lines) > 1500:
        risk -= 3.0

    return risk
