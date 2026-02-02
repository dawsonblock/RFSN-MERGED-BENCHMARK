import os
import tempfile
from rfsn_controller.replay_trace import TraceWriter, TraceReader, make_replay_llm_wrapper

def test_trace_hash_chain_and_replay():
    with tempfile.TemporaryDirectory() as td:
        trace_path = os.path.join(td, "trace.jsonl")
        w = TraceWriter(trace_path, run_meta={"seed": 0, "note": "unit"})
        w.record({"type": "llm_candidates", "candidates": ["patch1", "patch2"]})
        w.record({"type": "llm_candidates", "candidates": ["patch3"]})
        w.close()

        r = TraceReader(trace_path, verify_chain=True)
        replay = make_replay_llm_wrapper(r)

        assert replay() == ["patch1", "patch2"]
        assert replay() == ["patch3"]

def test_trace_tamper_detection():
    with tempfile.TemporaryDirectory() as td:
        trace_path = os.path.join(td, "trace.jsonl")
        w = TraceWriter(trace_path)
        w.record({"type": "llm_candidates", "candidates": ["a"]})
        w.close()

        # tamper with content
        with open(trace_path, "r+", encoding="utf-8") as f:
            lines = f.readlines()
            lines[0] = lines[0].replace('"a"', '"b"')
            f.seek(0)
            f.writelines(lines)
            f.truncate()

        try:
            TraceReader(trace_path, verify_chain=True)
            assert False, "Expected tamper detection"
        except RuntimeError:
            pass
