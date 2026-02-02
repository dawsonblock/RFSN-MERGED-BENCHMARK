from __future__ import annotations

import argparse
import hashlib
import os
from typing import Optional

from rfsn_controller.audit_chain import append_audit_log_local, build_entry_payload
from rfsn_controller.run_id import make_run_id


def sha256_file(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Repo path (for metadata + git commit)")
    ap.add_argument("--goal", required=True, help="Goal string")
    ap.add_argument("--profile", default=None, help="Profile name (optional)")
    ap.add_argument("--status", default=None, help="Run status (optional)")
    ap.add_argument("--artifacts", default="artifacts", help="Artifacts directory")
    ap.add_argument("--log", default=os.environ.get("RFSN_AUDIT_LOG", "artifacts/audit_log.jsonl"), help="Audit log path")
    ap.add_argument("--run-id", default=None, help="Run id. Default: derived")
    ap.add_argument("--published-to", default=None, help="Publish destination (optional)")

    args = ap.parse_args()

    run_id = args.run_id or make_run_id(repo_path=args.repo, goal=args.goal, profile=args.profile)

    manifest_path = os.path.join(args.artifacts, "manifest.json")
    signature_path = os.path.join(args.artifacts, "signature.txt")

    payload = build_entry_payload(
        run_id=run_id,
        repo=os.path.abspath(args.repo),
        goal=args.goal,
        profile=args.profile,
        status=args.status,
        commit=None,  # run_id already includes commit, but keep field for future
        manifest_sha256=sha256_file(manifest_path),
        signature_sha256=sha256_file(signature_path),
        published_to=args.published_to,
    )

    entry = append_audit_log_local(log_path=args.log, payload=payload)
    print(entry.entry_hash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
