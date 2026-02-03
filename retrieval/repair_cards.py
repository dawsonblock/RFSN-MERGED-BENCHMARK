"""Repair Cards: Storing and Retrieving Successful Fixes.

Stores successful fixes as compact "cards" that can be:
- Retrieved for similar failures (contextual matching)
- Ranked by outcome (win/loss tracking)
- Fed into patcher prompts for learning

This is upstream memory, not a gate bypass.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Any

# Default paths (kept out of gate; pure data)
DEFAULT_PATH = ".rfsn_state/repair_cards/cards.jsonl"
DEFAULT_INDEX = ".rfsn_state/repair_cards/index.json"


_ERR_RE = re.compile(r"\b([A-Za-z_]+Error)\b")
_FILE_RE = re.compile(r'File "([^"]+\.py)"')
_FAIL_TEST_RE = re.compile(r"FAILED\s+([^\s]+)")


def _mkdirp(p: str) -> None:
    """Create directory and parents if needed."""
    os.makedirs(p, exist_ok=True)


def _sha16(s: str) -> str:
    """Return first 16 chars of SHA256 hex digest."""
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _now() -> float:
    """Return current Unix timestamp."""
    return time.time()


def _safe_norm(s: str, limit: int = 2000) -> str:
    """Normalize and truncate string."""
    s = (s or "").strip()
    return s[:limit]


def parse_signals(test_output: str) -> dict[str, str]:
    """Extract error_type, top_file, failing_test from test output."""
    t = test_output or ""
    m = _ERR_RE.search(t)
    err = m.group(1) if m else "UnknownError"

    fm = _FILE_RE.search(t)
    top_file = fm.group(1).replace("\\", "/") if fm else ""

    tm = _FAIL_TEST_RE.search(t)
    failing_test = tm.group(1) if tm else ""

    return {"error_type": err, "top_file": top_file, "failing_test": failing_test}


def repo_fingerprint_from_files(file_blobs: dict[str, str]) -> str:
    """Generate deterministic fingerprint from dependency/config files."""
    keys = [
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
        "requirements.txt",
        "requirements-dev.txt",
        "Pipfile",
        "poetry.lock",
        "tox.ini",
        "pytest.ini",
    ]
    h = hashlib.sha256()
    for k in keys:
        h.update(k.encode("utf-8"))
        h.update((file_blobs.get(k, "") or "").encode("utf-8", errors="ignore")[:250_000])
    return h.hexdigest()[:16]


@dataclass
class RepairCard:
    """A repair card storing a successful fix."""
    card_id: str
    created_ts: float

    repo: str
    repo_fp: str

    bucket: str
    error_type: str
    top_file: str
    failing_test: str

    patch_summary: str
    patch_diff: str

    # outcome stats
    wins: int = 1
    losses: int = 0

    # for similarity search
    key_sig: str = ""


def _card_key_sig(
    repo_fp: str, bucket: str, error_type: str, top_file: str, failing_test: str
) -> str:
    """Generate stable composite key signature."""
    return "|".join([repo_fp, bucket, error_type, top_file[:80], failing_test[:120]])


def _atomic_append_jsonl(path: str, obj: dict[str, Any]) -> None:
    """Append JSON object to JSONL file atomically."""
    _mkdirp(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def _atomic_write(path: str, obj: Any) -> None:
    """Write JSON file atomically using rename."""
    _mkdirp(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def load_index(path: str = DEFAULT_INDEX) -> dict[str, Any]:
    """Load the repair cards index."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"version": 1, "cards": {}}


def save_index(idx: dict[str, Any], path: str = DEFAULT_INDEX) -> None:
    """Save the repair cards index."""
    _atomic_write(path, idx)


def add_card(
    *,
    repo: str,
    repo_fp: str,
    bucket: str,
    test_output: str,
    patch_diff: str,
    patch_summary: str,
    path_jsonl: str = DEFAULT_PATH,
    path_index: str = DEFAULT_INDEX,
) -> RepairCard:
    """Add a new repair card for a successful fix."""
    sig = parse_signals(test_output)
    key_sig = _card_key_sig(
        repo_fp, bucket, sig["error_type"], sig["top_file"], sig["failing_test"]
    )
    card_id = _sha16(key_sig + "|" + patch_diff[:4000] + "|" + str(_now()))

    card = RepairCard(
        card_id=card_id,
        created_ts=_now(),
        repo=_safe_norm(repo, 200),
        repo_fp=_safe_norm(repo_fp, 32),
        bucket=_safe_norm(bucket, 120),
        error_type=_safe_norm(sig["error_type"], 80),
        top_file=_safe_norm(sig["top_file"], 200),
        failing_test=_safe_norm(sig["failing_test"], 200),
        patch_summary=_safe_norm(patch_summary, 240),
        patch_diff=_safe_norm(patch_diff, 250_000),
        key_sig=key_sig,
    )

    _atomic_append_jsonl(path_jsonl, asdict(card))

    idx = load_index(path_index)
    idx["cards"][card_id] = {
        "repo_fp": card.repo_fp,
        "bucket": card.bucket,
        "error_type": card.error_type,
        "top_file": card.top_file,
        "failing_test": card.failing_test,
        "wins": card.wins,
        "losses": card.losses,
        "key_sig": card.key_sig,
        "created_ts": card.created_ts,
        "patch_summary": card.patch_summary,
        "path": path_jsonl,
    }
    save_index(idx, path_index)
    return card


def update_card_outcome(
    card_id: str,
    success: bool,
    path_index: str = DEFAULT_INDEX,
) -> None:
    """Update a card's win/loss statistics based on outcome."""
    idx = load_index(path_index)
    meta = idx["cards"].get(card_id)
    if not meta:
        return
    if success:
        meta["wins"] = int(meta.get("wins", 0)) + 1
    else:
        meta["losses"] = int(meta.get("losses", 0)) + 1
    idx["cards"][card_id] = meta
    save_index(idx, path_index)


def _score(meta: dict[str, Any], query: dict[str, str]) -> float:
    """Score a card against a query based on match criteria."""
    s = 0.0
    if meta.get("repo_fp") == query.get("repo_fp"):
        s += 2.5
    if meta.get("bucket") == query.get("bucket"):
        s += 2.0
    if meta.get("error_type") == query.get("error_type"):
        s += 1.8

    q_file = query.get("top_file", "")
    m_file = meta.get("top_file", "")
    if q_file and m_file and (m_file in q_file or q_file in m_file):
        s += 1.2

    q_test = query.get("failing_test", "")
    if q_test and meta.get("failing_test", "") and (q_test == meta["failing_test"]):
        s += 1.2

    # Apply win rate shaping (Laplace smoothed)
    wins = float(meta.get("wins", 0))
    losses = float(meta.get("losses", 0))
    rate = (wins + 1.0) / (wins + losses + 2.0)
    s *= (0.7 + 0.6 * rate)
    return s


def retrieve_cards(
    *,
    repo_fp: str,
    bucket: str,
    test_output: str,
    k: int = 3,
    path_index: str = DEFAULT_INDEX,
    path_jsonl: str = DEFAULT_PATH,
) -> list[dict[str, Any]]:
    """Retrieve top-k similar repair cards for a failure context."""
    sig = parse_signals(test_output)
    query = {
        "repo_fp": repo_fp,
        "bucket": bucket,
        "error_type": sig["error_type"],
        "top_file": sig["top_file"],
        "failing_test": sig["failing_test"],
    }

    idx = load_index(path_index)
    metas = []
    for cid, meta in idx.get("cards", {}).items():
        meta2 = dict(meta)
        meta2["card_id"] = cid
        meta2["score"] = _score(meta2, query)
        metas.append(meta2)

    metas.sort(key=lambda m: m["score"], reverse=True)
    metas = metas[:k]

    # Load diffs for selected cards
    if not metas:
        return []

    want = set(m["card_id"] for m in metas)
    cards_by_id: dict[str, dict[str, Any]] = {}
    try:
        with open(path_jsonl, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                cid = obj.get("card_id")
                if cid in want:
                    cards_by_id[cid] = obj
    except Exception:
        pass

    out = []
    for m in metas:
        cid = m["card_id"]
        obj = cards_by_id.get(cid, {})
        out.append({
            "card_id": cid,
            "score": float(m["score"]),
            "patch_summary": m.get("patch_summary", ""),
            "bucket": m.get("bucket", ""),
            "error_type": m.get("error_type", ""),
            "top_file": m.get("top_file", ""),
            "failing_test": m.get("failing_test", ""),
            "wins": int(m.get("wins", 0)),
            "losses": int(m.get("losses", 0)),
            "patch_diff": obj.get("patch_diff", ""),
        })
    return out


def format_repair_cards_for_prompt(cards: list[dict], max_chars: int = 6000) -> str:
    """Format repair cards as context for LLM prompt."""
    if not cards:
        return ""
    parts = ["\n## Similar Successful Fixes (Repair Cards)\n"]
    for i, c in enumerate(cards[:3], start=1):
        diff = (c.get("patch_diff") or "").strip()
        if len(diff) > 2500:
            diff = diff[:2500] + "\n... (truncated)"
        parts.append(
            f"\n### Card {i}\n"
            f"- score: {c.get('score', 0):.2f}\n"
            f"- bucket: {c.get('bucket', '')}\n"
            f"- error: {c.get('error_type', '')}\n"
            f"- summary: {c.get('patch_summary', '')}\n"
            f"```diff\n{diff}\n```\n"
        )
    out = "\n".join(parts)
    return out[:max_chars]
