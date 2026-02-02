"""Download SWE-bench datasets using Hugging Face datasets library."""
import os
from datasets import load_dataset
import pandas as pd

def main():
    os.makedirs("datasets", exist_ok=True)
    
    # SWE-bench Lite
    print("Downloading SWE-bench Lite (test split)...")
    try:
        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        output_path = "datasets/swebench_lite.jsonl"
        ds.to_json(output_path, orient="records", lines=True)
        print(f"Successfully saved {output_path} ({len(ds)} records)")
    except Exception as e:
        print(f"Failed to download SWE-bench Lite: {e}")

    # SWE-bench Verified (Optional but good to have)
    print("\nDownloading SWE-bench Verified (test split)...")
    try:
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        output_path = "datasets/swebench_verified.jsonl"
        ds.to_json(output_path, orient="records", lines=True)
        print(f"Successfully saved {output_path} ({len(ds)} records)")
    except Exception as e:
        print(f"Failed to download SWE-bench Verified: {e}")

if __name__ == "__main__":
    main()
