\
from __future__ import annotations

import argparse
import json
import os
import time

from rfsn_controller.signing import compute_manifest, load_key_from_env, sign_manifest
from rfsn_controller.storage import make_store


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts", help="Artifacts directory to publish")
    ap.add_argument("--run-id", default=None, help="Run id. Default: timestamp")
    ap.add_argument("--publish-backend", choices=["local", "s3"], default=os.environ.get("RFSN_PUBLISH_BACKEND", "local"))
    ap.add_argument("--publish-local-dir", default=os.environ.get("RFSN_PUBLISH_LOCAL_DIR", "artifacts/published"))
    ap.add_argument("--publish-s3-bucket", default=os.environ.get("RFSN_PUBLISH_S3_BUCKET"))
    ap.add_argument("--publish-s3-prefix", default=os.environ.get("RFSN_PUBLISH_S3_PREFIX", "rfsn/runs"))
    ap.add_argument("--sign", action="store_true", default=(os.environ.get("RFSN_SIGN", "1") == "1"))
    ap.add_argument("--signing-key-env", default=os.environ.get("RFSN_SIGNING_KEY_ENV", "RFSN_SIGNING_KEY"))
    args = ap.parse_args()

    artifacts_dir = args.artifacts
    if not os.path.isdir(artifacts_dir):
        raise SystemExit(f"Artifacts dir not found: {artifacts_dir}")

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")

    manifest = compute_manifest(artifacts_dir)
    manifest_path = os.path.join(artifacts_dir, "manifest.json")
    manifest_bytes = json.dumps({"run_id": run_id, "files": manifest.files}, indent=2, sort_keys=True).encode("utf-8")
    with open(manifest_path, "wb") as f:
        f.write(manifest_bytes)

    if args.sign:
        key = load_key_from_env(args.signing_key_env)
        sig = sign_manifest(manifest_bytes, key)
        with open(os.path.join(artifacts_dir, "signature.txt"), "w", encoding="utf-8") as f:
            f.write(sig + "\n")

    if args.publish_backend == "local":
        dest = os.path.join(args.publish_local_dir, run_id)
        store = make_store("local", local_dir=args.publish_local_dir)
        res = store.put_dir(artifacts_dir, dest)
    else:
        store = make_store("s3", s3_bucket=args.publish_s3_bucket, s3_prefix=args.publish_s3_prefix)
        res = store.put_dir(artifacts_dir, run_id)

    print(json.dumps({"run_id": run_id, "published_to": res.destination, "backend": res.backend}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
