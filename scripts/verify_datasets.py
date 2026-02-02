"""Dataset verification script."""
from __future__ import annotations
import os
import sys

REQUIRED_DATASETS = [
    "datasets/swebench_lite.jsonl",
    "datasets/swebench_verified.jsonl",
    "datasets/swebench_full.jsonl",
]


def main() -> int:
    """Verify at least one dataset is present."""
    found = [p for p in REQUIRED_DATASETS if os.path.exists(p)]
    
    if not found:
        print("ERROR: No datasets present under datasets/. Strict mode expects at least one.")
        print("Required (at least one):")
        for p in REQUIRED_DATASETS:
            print(f"  - {p}")
        print("\nTo acquire datasets:")
        print("  python scripts/fetch_swebench.py /path/to/swebench_lite.jsonl")
        return 2
    
    print("OK: found dataset files:")
    for p in found:
        # Count lines
        with open(p, "r") as f:
            count = sum(1 for line in f if line.strip())
        print(f"  - {p} ({count} tasks)")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
