"""Command-line entrypoint for SWE-bench evaluation (Unified v2).

This wraps the new RFSN Unified Evaluation Harness.
"""
from __future__ import annotations
import sys
from eval.run_v2 import main

if __name__ == "__main__":
    sys.exit(main())
