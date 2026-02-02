from __future__ import annotations

import argparse
import os

from rfsn_controller.audit_chain import verify_audit_log_local


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.environ.get("RFSN_AUDIT_LOG", "artifacts/audit_log.jsonl"))
    args = ap.parse_args()

    ok, msg = verify_audit_log_local(args.log)
    if ok:
        print("ok")
        return 0
    print(msg)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
