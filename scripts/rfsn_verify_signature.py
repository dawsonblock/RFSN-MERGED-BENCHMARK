\
from __future__ import annotations

import argparse
import json
import os

from rfsn_controller.signing import load_key_from_env, verify_manifest, compute_manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--signing-key-env", default=os.environ.get("RFSN_SIGNING_KEY_ENV", "RFSN_SIGNING_KEY"))
    args = ap.parse_args()

    artifacts = args.artifacts
    manifest_path = os.path.join(artifacts, "manifest.json")
    sig_path = os.path.join(artifacts, "signature.txt")

    if not os.path.exists(manifest_path) or not os.path.exists(sig_path):
        raise SystemExit("manifest.json or signature.txt missing")

    with open(manifest_path, "rb") as f:
        manifest_bytes = f.read()
    with open(sig_path, "r", encoding="utf-8") as f:
        sig = f.read().strip()

    key = load_key_from_env(args.signing_key_env)
    ok, expected = verify_manifest(manifest_bytes, key, sig)

    obj = json.loads(manifest_bytes.decode("utf-8"))
    expected_files = obj.get("files", {})
    actual = compute_manifest(artifacts).files

    mismatches = [rel for rel, h in expected_files.items() if actual.get(rel) != h]
    extra = [rel for rel in actual.keys() if rel not in expected_files]

    if ok and not mismatches and not extra:
        print("OK: signature and manifest match artifacts directory")
        return 0

    print("FAILED")
    print("signature_ok:", ok)
    if not ok:
        print("expected_signature:", expected)
    if mismatches:
        print("mismatched_files:", mismatches[:50])
    if extra:
        print("extra_files:", extra[:50])
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
