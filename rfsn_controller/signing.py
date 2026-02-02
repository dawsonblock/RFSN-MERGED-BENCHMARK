\
from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass


@dataclass
class Manifest:
    files: dict[str, str]  # relpath -> sha256 hex


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_manifest(root_dir: str) -> Manifest:
    files: dict[str, str] = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, root_dir)
            if rel in ("manifest.json", "signature.txt"):
                continue
            files[rel] = sha256_file(p)
    return Manifest(files=files)


def sign_manifest(manifest_json_bytes: bytes, key: bytes) -> str:
    return hmac.new(key, manifest_json_bytes, hashlib.sha256).hexdigest()


def verify_manifest(manifest_json_bytes: bytes, key: bytes, signature_hex: str) -> tuple[bool, str]:
    expected = sign_manifest(manifest_json_bytes, key)
    ok = hmac.compare_digest(expected, signature_hex.strip())
    return ok, expected


def load_key_from_env(env_name: str) -> bytes:
    v = os.environ.get(env_name)
    if not v:
        raise RuntimeError(f"Signing key env var not set: {env_name}")
    return v.encode("utf-8")
