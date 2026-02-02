"""Simple JSONL logger for the RFSN controller."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any

from .clock import Clock


class DataclassJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles dataclass objects."""
    
    def default(self, o: Any) -> Any:
        if is_dataclass(o) and not isinstance(o, type):
            return asdict(o)
        return super().default(o)


def ensure_dir(path: str) -> None:
    """Ensure that a directory exists."""
    os.makedirs(path, exist_ok=True)


def write_jsonl(
    log_dir: str,
    record: dict[str, Any],
    *,
    clock: Clock | None = None,
    ts: float | None = None,
) -> None:
    """Append a record to a JSONL file in the given log directory.

    Args:
        log_dir: Directory where logs should be written.
        record: Arbitrary JSON-serializable dictionary to write.
    """
    ensure_dir(log_dir)
    entry = dict(record)
    if ts is not None:
        entry["ts"] = float(ts)
    elif clock is not None:
        entry["ts"] = float(clock.time())
    else:
        raise ValueError("write_jsonl requires either ts or clock")
    path = os.path.join(log_dir, "run.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, cls=DataclassJSONEncoder) + "\n")
