
import os
import sys
import tempfile
import asyncio
from pathlib import Path
from dataclasses import dataclass

# Ensure we can import from root
sys.path.append(os.getcwd())

from rfsn_controller.replay_trace import TraceWriter, TraceReader

# Mock LLM Patcher to simulate LLM response without real API calls
# from agent.llm_patcher import patch_generator  # REMOVED - not needed

@dataclass
class MockGate:
    def decide(self, *args, **kwargs):
        print(" [Gate] Deciding: ALLOW")
        return {"action": "allow", "reason": "demo safe"}

async def run_demo():
    print("=== RFSN Full Episode Audit Demo ===\n")
    
    trace_file = Path("demo_audit.trace")
    if trace_file.exists():
        trace_file.unlink()

    print("--- Phase 1: Recording ---")
    
    # 1. Setup Record Environment
    os.environ["RFSN_TRACE_MODE"] = "RECORD"
    os.environ["RFSN_TRACE_FILE"] = str(trace_file)
    

    # We need to manually inject the writer for this standalone script 
    writer = TraceWriter(str(trace_file))
    # writer.__enter__() # TraceWriter is not a context manager
    
    from agent import llm_patcher
    llm_patcher._trace_writer = writer
    
    print(" -> Recording Trace...")
    
    # 1. Gate Decision
    # Matches orchestrator/episode_runner.py structure
    writer.record({
        "type": "gate_decision",
        "allowed": True,
        "reason": "demo pass"
    })
    print("    Recorded: Gate Decision (ALLOW)")
    
    # 2. Patch Applied
    patch_content = "def verify(): pass"
    import hashlib
    patch_hash = hashlib.sha256(patch_content.encode()).hexdigest()
    writer.record({
        "type": "applied_patch",
        "patch_hash": patch_hash
    })
    print(f"    Recorded: Patch Applied (Hash: {patch_hash[:8]}...)")

    # 3. Test Result
    writer.record({
        "type": "test_result",
        "cmd": "pytest",
        "returncode": 0,
        "output_hash": "hash_of_output"
    })
    print("    Recorded: Test Result (Success)")
    
    writer.close()
    print(" -> Trace recorded successfully.\n")

    
    print("--- Phase 2: Replay (Verification) ---")
    
    # 2. Setup Replay Environment
    os.environ["RFSN_TRACE_MODE"] = "REPLAY"
    reader = TraceReader(str(trace_file))
    # reader.__enter__() # TraceReader is not a context manager
    llm_patcher._trace_writer = None
    llm_patcher._trace_reader = reader
    
    print(" -> Verifying Trace...")
    
    try:
        # Verify Gate
        print("    Verifying: Gate Decision...", end=" ")
        reader.verify_gate_decision({"allowed": True, "reason": "demo pass"})
        print("MATCH ✓")
        
        # Verify Patch
        print("    Verifying: Patch Hash...", end=" ")
        reader.verify_patch_hash(patch_hash)
        print("MATCH ✓")
        
        # Verify Test
        print("    Verifying: Test Result...", end=" ")
        reader.verify_test_result("pytest", 0, "hash_of_output")
        print("MATCH ✓")
        
        print("\n=== Demo Completed Successfully: Audit Integrity Verified ===")
        
    except RuntimeError as e:
        print(f"\nFATAL: Verification Failed!\n{e}")
        sys.exit(1)
    
    # Cleanup
    if trace_file.exists():
        trace_file.unlink()

if __name__ == "__main__":
    asyncio.run(run_demo())
