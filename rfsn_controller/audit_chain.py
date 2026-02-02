from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _json_canonical(obj: Any) -> bytes:
    # Deterministic JSON for hashing/signing.
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass
class AuditEntry:
    run_id: str
    ts_unix: int
    repo: str
    commit: str | None
    goal: str
    profile: str | None
    status: str | None
    manifest_sha256: str | None
    signature_sha256: str | None
    published_to: str | None
    prev_hash: str
    entry_hash: str


def compute_prev_hash_from_log(log_path: str) -> str:
    if not os.path.exists(log_path):
        return "0" * 64
    # Read last non-empty line
    last = ""
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = line
    if not last:
        return "0" * 64
    try:
        obj = json.loads(last)
        return str(obj.get("entry_hash") or obj.get("hash") or "0" * 64)
    except Exception:
        return "0" * 64


def compute_entry_hash(payload: dict[str, Any], prev_hash: str) -> str:
    # Hash over (prev_hash + canonical(payload))
    base = {"prev_hash": prev_hash, "payload": payload}
    return _sha256_bytes(_json_canonical(base))


def build_entry_payload(
    *,
    run_id: str,
    repo: str,
    goal: str,
    profile: str | None = None,
    status: str | None = None,
    commit: str | None = None,
    manifest_sha256: str | None = None,
    signature_sha256: str | None = None,
    published_to: str | None = None,
    ts_unix: int | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "ts_unix": int(ts_unix if ts_unix is not None else time.time()),
        "repo": repo,
        "commit": commit,
        "goal": goal,
        "profile": profile,
        "status": status,
        "manifest_sha256": manifest_sha256,
        "signature_sha256": signature_sha256,
        "published_to": published_to,
    }


def append_audit_log_local(
    *,
    log_path: str,
    payload: dict[str, Any],
) -> AuditEntry:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    prev_hash = compute_prev_hash_from_log(log_path)
    entry_hash = compute_entry_hash(payload, prev_hash)

    entry_obj = {
        "prev_hash": prev_hash,
        "entry_hash": entry_hash,
        **payload,
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry_obj, ensure_ascii=False) + "\n")

    return AuditEntry(
        run_id=str(payload.get("run_id")),
        ts_unix=int(payload.get("ts_unix")),
        repo=str(payload.get("repo")),
        commit=payload.get("commit"),
        goal=str(payload.get("goal")),
        profile=payload.get("profile"),
        status=payload.get("status"),
        manifest_sha256=payload.get("manifest_sha256"),
        signature_sha256=payload.get("signature_sha256"),
        published_to=payload.get("published_to"),
        prev_hash=prev_hash,
        entry_hash=entry_hash,
    )


def verify_audit_log_local(log_path: str) -> tuple[bool, str]:
    if not os.path.exists(log_path):
        return False, f"Audit log not found: {log_path}"

    prev_expected = "0" * 64
    line_no = 0

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line_no += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                return False, f"Invalid JSON at line {line_no}: {e}"

            prev = str(obj.get("prev_hash", ""))
            h = str(obj.get("entry_hash", ""))

            if prev != prev_expected:
                return False, f"Chain break at line {line_no}: prev_hash={prev} expected={prev_expected}"

            payload = dict(obj)
            payload.pop("prev_hash", None)
            payload.pop("entry_hash", None)

            computed = compute_entry_hash(payload, prev_expected)
            if computed != h:
                return False, f"Hash mismatch at line {line_no}: entry_hash={h} computed={computed}"

            prev_expected = h

    return True, "ok"
