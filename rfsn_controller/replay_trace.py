import contextlib
import hashlib
import json
from pathlib import Path
from typing import Any

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _canon(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

class TraceWriter:
    """Append-only JSONL trace with hash chaining."""

    def __init__(self, path: str, run_meta: dict[str, Any] | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("a", encoding="utf-8")
        self._last_hash: str | None = None
        if run_meta is not None:
            self.record({"type": "run_meta", "meta": run_meta})

    def record(self, event: dict[str, Any]):
        payload = dict(event)
        prev_hash = self._last_hash
        wrapped = {"prev_hash": prev_hash, "event": payload}
        event_hash = _sha256_bytes(_canon(wrapped))
        out = {**payload, "prev_hash": prev_hash, "event_hash": event_hash}
        self.f.write(json.dumps(out, ensure_ascii=False) + "\n")
        self.f.flush()
        self._last_hash = event_hash

    def close(self):
        with contextlib.suppress(Exception):
            self.f.close()

class TraceReader:
    def __init__(self, path: str, verify_chain: bool = True):
        self.events: list[dict[str, Any]] = []
        self._llm_idx = 0
        last_hash: str | None = None

        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                ev = json.loads(line)
                if verify_chain:
                    prev_hash = ev.get("prev_hash")
                    if prev_hash != last_hash:
                        raise RuntimeError(f"Trace chain broken at line {line_no}: prev_hash mismatch")

                    payload = {k: v for k, v in ev.items() if k not in ("prev_hash", "event_hash")}
                    wrapped = {"prev_hash": prev_hash, "event": payload}
                    expected_hash = _sha256_bytes(_canon(wrapped))
                    if ev.get("event_hash") != expected_hash:
                        raise RuntimeError(f"Trace tampered at line {line_no}: event_hash mismatch")
                    last_hash = ev.get("event_hash")

                self.events.append(ev)

    def next_llm(self) -> list[str]:
        if self._llm_idx >= len(self.events):
            raise RuntimeError("No more recorded events (expected LLM)")

        # In strict mode, we might demand the very next event is LLM.
        # But 'next_llm' implies skipping non-LLM events (like meta/audit) until we find LLM.
        # To remain compatible but robust, we search forward.
        while self._llm_idx < len(self.events):
            ev = self.events[self._llm_idx]
            self._llm_idx += 1
            if ev.get("type") == "llm_candidates":
                return ev.get("candidates", [])
        
        raise RuntimeError("No more recorded LLM events")

    def _verify_next_event(self, expected_type: str, match_payload: dict[str, Any]):
        """
        Scan for next event of expected_type. 
        If strict compliance is needed, we could assert it's the *immediate* next.
        For now, let's implement strict immediate checking for audit events, 
        but allow skipping non-matching types if we want looser coupling?
        
        Actually, for a strict audit, we probably want to consume events in order.
        However, earlier traces might lack these new events. 
        So we should only verify if we find them? 
        
        DECISION: Strict ordered verification. If the trace has extra events we don't know about, that's fine?
        No, let's look for the next event of 'type'.
        """
        while self._llm_idx < len(self.events):
            ev = self.events[self._llm_idx]
            # If we hit an LLM event while looking for a Gate event, that's a mismatch (unless we skip LLM? No).
            # Let's simple scan forward for the type.
            if ev.get("type") == expected_type:
                self._llm_idx += 1
                # Verify payload
                for k, v in match_payload.items():
                    if ev.get(k) != v:
                        raise RuntimeError(
                            f"Replay mismatch for {expected_type} at idx {self._llm_idx}:\n"
                            f"Expected {k}={v}\n"
                            f"Found {k}={ev.get(k)}"
                        )
                return
            
            # If we see a different type, do we skip or fail? 
            # If we want FULL AUDIT, we should probably fail if we see an event we didn't ask for?
            # But the orchestrator might not ask for everything (e.g. meta).
            # Let's skip non-matching events until we find ours or run out.
            self._llm_idx += 1
            
        raise RuntimeError(f"Expected event {expected_type} not found in remaining trace")

    def verify_gate_decision(self, decision: dict[str, Any]):
        # We verify 'allowed' and maybe 'reason'. 
        self._verify_next_event("gate_decision", {
            "allowed": decision.get("allowed"),
            # "reason": decision.get("reason") # Strict reason matching might be brittle to string changes?
        })

    def verify_patch_hash(self, patch_hash: str):
        self._verify_next_event("applied_patch", {"patch_hash": patch_hash})

    def verify_test_result(self, cmd: str, returncode: int, output_hash: str):
        self._verify_next_event("test_result", {
            "cmd": cmd,
            "returncode": returncode,
            "output_hash": output_hash
        })

def make_recording_llm_wrapper(llm_fn, trace: TraceWriter):
    def wrapped(*args, **kwargs):
        cands = llm_fn(*args, **kwargs)
        trace.record({"type": "llm_candidates", "candidates": cands})
        return cands
    return wrapped

def make_replay_llm_wrapper(trace: TraceReader):
    def wrapped(*args, **kwargs):
        return trace.next_llm()
    return wrapped
