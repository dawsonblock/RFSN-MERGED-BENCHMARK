import argparse
import json
import sys

from swebench_max.orchestrator import swebench_max_run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--issue_json", required=True, help="JSON string or @file.json")
    ap.add_argument("--cfg", default="configs/swebench_max.yaml")
    args = ap.parse_args()

    if args.issue_json.startswith("@"):
        with open(args.issue_json[1:], "r", encoding="utf-8") as f:
            issue_json = f.read()
    else:
        issue_json = args.issue_json

    out = swebench_max_run(args.repo, issue_json, args.cfg)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
