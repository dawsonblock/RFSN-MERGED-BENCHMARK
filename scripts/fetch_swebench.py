"""Copy SWE-bench dataset into datasets/ directory."""
from __future__ import annotations
import os
import shutil
import sys

DEST_DIR = "datasets"


def copy_in(src: str) -> None:
    """Copy a dataset file into the datasets directory."""
    os.makedirs(DEST_DIR, exist_ok=True)
    name = os.path.basename(src)
    dest = os.path.join(DEST_DIR, name)
    shutil.copyfile(src, dest)
    print(f"Copied {src} -> {dest}")
    
    # Count tasks
    with open(dest, "r") as f:
        count = sum(1 for line in f if line.strip())
    print(f"Dataset contains {count} tasks")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/fetch_swebench.py /path/to/swebench_lite.jsonl")
        print("\nThis copies the dataset file into datasets/")
        return 2
    
    src = sys.argv[1]
    if not os.path.exists(src):
        print(f"ERROR: Missing source file: {src}")
        return 2
    
    copy_in(src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
