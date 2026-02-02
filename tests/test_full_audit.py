import os
import tempfile
import pytest
from rfsn_controller.replay_trace import TraceWriter, TraceReader

def test_full_audit_lifecycle():
    with tempfile.TemporaryDirectory() as td:
        trace_path = os.path.join(td, "audit_trace.jsonl")
        
        # 1. Record a trace with all new event types
        w = TraceWriter(trace_path, run_meta={"test": "audit"})
        
        # LLM event
        w.record({"type": "llm_candidates", "candidates": ["patch_A"]})
        
        # Gate event
        w.record({"type": "gate_decision", "allowed": True, "reason": "LGTM"})
        
        # Patch event
        w.record({"type": "applied_patch", "patch_hash": "hash_of_patch_A"})
        
        # Test event
        w.record({"type": "test_result", "cmd": "pytest", "returncode": 0, "output_hash": "hash_of_output"})
        
        w.close()
        
        # 2. Verify Replay - Happy Path
        r = TraceReader(trace_path, verify_chain=True)
        
        # LLM
        assert r.next_llm() == ["patch_A"]
        
        # Verify Gate
        r.verify_gate_decision({"allowed": True, "reason": "LGTM"})
        
        # Verify Patch
        r.verify_patch_hash("hash_of_patch_A")
        
        # Verify Test
        r.verify_test_result("pytest", 0, "hash_of_output")
        
def test_audit_tamper_gate():
    with tempfile.TemporaryDirectory() as td:
        trace_path = os.path.join(td, "tamper_gate.jsonl")
        
        # Record
        w = TraceWriter(trace_path)
        w.record({"type": "llm_candidates", "candidates": ["p"]})
        w.record({"type": "gate_decision", "allowed": True})
        w.close()
        
        # Replay with mismatching expectation
        r = TraceReader(trace_path)
        r.next_llm()
        
        # We expect True, but if the simulation says False (or we pass False to verify), it should raise
        # Wait, verify_gate_decision takes the *current execution's* result and compares it to the trace.
        # So if trace says True, and we pass False (current exec rejected), it should raise.
        
        try:
            r.verify_gate_decision({"allowed": False})
            assert False, "Should have raised mismatch"
        except RuntimeError as e:
            assert "Replay mismatch" in str(e)
            assert "Expected allowed=False" in str(e)
            
def test_audit_tamper_patch():
    with tempfile.TemporaryDirectory() as td:
        trace_path = os.path.join(td, "tamper_patch.jsonl")
        w = TraceWriter(trace_path)
        w.record({"type": "llm_candidates", "candidates": ["p"]})
        # Simulate skipping gate for brevity
        w.record({"type": "applied_patch", "patch_hash": "abc"})
        w.close()
        
        r = TraceReader(trace_path)
        r.next_llm()
        
        try:
            r.verify_patch_hash("xyz") # Current execution has different hash
            assert False, "Should have raised mismatch"
        except RuntimeError as e:
            assert "Expected patch_hash=xyz" in str(e)

