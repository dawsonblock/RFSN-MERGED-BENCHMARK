"""
Example: Basic SWE-bench Evaluation (Unified v2)

This example shows how to run a basic evaluation using the 
RFSN Unified Evaluation Harness (run_v2).
"""

from pathlib import Path
from eval.run_v2 import run_eval

def main():
    """Run a basic SWE-bench evaluation."""
    print("Starting evaluation (RFSN v2)...")
    
    # Run evaluation
    results = run_eval(
        dataset_name="swebench_lite.jsonl",
        max_tasks=2,      # Small batch
        max_attempts=3,   # Limited attempts
        results_dir="./example_results",
    )
    
    # Print summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed ({(passed/total*100) if total else 0:.1f}%)")
    print(f"{'='*50}")
    
    for result in results:
        status_icon = "✅" if result.passed else "❌"
        # Access enum value correctly
        status_str = result.status.value if hasattr(result.status, "value") else str(result.status)
        print(f"  {status_icon} {result.instance_id}: {status_str} (Attempts: {result.attempts})")
        
        if not result.passed and result.test_output_tail:
            print(f"    Error tail: {result.test_output_tail[:100]}...")

if __name__ == "__main__":
    main()
