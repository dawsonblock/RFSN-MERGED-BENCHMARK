"""Episode runner - single authority loop for SWE-bench evaluation.

This module implements the correct SWE-bench procedure:
1. Clone repo at base_commit
2. Apply test_patch (must succeed or INVALID)
3. Run baseline tests (expect failure)
4. Iterate:
   - Propose patches via upstream intelligence
   - Gate each proposal
   - Apply and test
   - Accept first passing result
   - Reset between attempts

The gate (PlanGate) is the single source of truth.
All intelligence is upstream.
"""
from __future__ import annotations

import hashlib
import logging
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable


from eval.repo_setup import clone_repo, hard_reset_clean, apply_patch_text, cleanup_workspace
from eval.test_cmd import derive_test_command_for_repo

from agent.gate_adapter import GateAdapter
from agent.propose_v2 import propose as propose_v2, learn_update
from retrieval.failure_index import FailureRecord, FailureIndex
from agent.llm_patcher import get_active_trace_writer, get_active_trace_reader
import re
import hashlib

def _sanitize_output(text: str) -> str:
    """Sanitize output to ensure deterministic hashing."""
    # Remove durations (e.g. "in 0.12s", "(0.01s)", "0.000s")
    text = re.sub(r'in \d+\.\d+s', 'in X.XXs', text)
    text = re.sub(r'\(\d+\.\d+s\)', '(X.XXs)', text)
    
    # Remove temp paths (e.g. rfsn_repo_abcdef123)
    text = re.sub(r'rfsn_repo_[a-zA-Z0-9_]+', '[REPO_PATH]', text)
    
    # Remove absolute paths starting with /tmp or /var
    # (Basic attempt, might need refinement)
    text = re.sub(r'/\S+/rfsn_repo_', '[ROOT]/rfsn_repo_', text)
    
    return text

def _hash_str(s: str) -> str:
    return hashlib.sha256(_sanitize_output(s).encode("utf-8")).hexdigest()

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Result of running a SWE-bench task."""
    passed: bool
    test_output: str
    attempts: int
    invalid: bool = False
    reason: str = ""
    gate_rejections: int = 0
    security_violations: int = 0
    test_delta: int = 0
    runtime: float = 0.0
    patch_size: int = 0
    files_touched: int = 0



def _run_cmd(cmd: list[str], cwd: str, timeout_s: int = 1200) -> tuple[int, str, float]:
    """Run a command and return (returncode, output, runtime)."""
    t0 = time.time()
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return p.returncode, p.stdout, (time.time() - t0)
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT", (time.time() - t0)


def _parse_failure_count(output: str) -> int:
    """Parse number of failed tests from output."""
    if not output:
        return 0
    # Pytest pattern: "3 passed, 1 failed" or "1 failed, 3 passed"
    m = re.search(r"(\d+) failed", output)
    if m:
        return int(m.group(1))
    
    # Unittest pattern: "FAILED (failures=1, errors=2)"
    m = re.search(r"failures=(\d+)", output)
    failures = int(m.group(1)) if m else 0
    m = re.search(r"errors=(\d+)", output)
    errors = int(m.group(1)) if m else 0
    if failures + errors > 0:
        return failures + errors
        
    return 0



def run_one_task(
    task: dict[str, Any],
    repo_url: str,
    llm_patch_fn: Callable,
    max_attempts: int = 6,
    cleanup: bool = True,
    record_callback: Callable[[RunResult], None] | None = None,
) -> RunResult:
    """
    Run a single SWE-bench task with full SWE-bench procedure.
    
    Args:
        task: Task dict with instance_id, repo, base_commit, test_patch, etc.
        repo_url: Git URL to clone
        llm_patch_fn: Function(plan, ctx) -> list[dict] with patch_text, summary
        max_attempts: Maximum patch attempts
        cleanup: Whether to clean up workspace after
        
    Returns:
        RunResult with pass/fail and attempt count
    """
    gate = GateAdapter()
    failure_index = FailureIndex()
    gate = GateAdapter()
    failure_index = FailureIndex()
    gate_rejections = 0
    security_count = 0
    
    # Audit traces
    trace_writer = get_active_trace_writer()
    trace_reader = get_active_trace_reader()
    
    instance_id = task.get("instance_id", "unknown")
    logger.info("Starting task: %s", instance_id)

    # Clone repo
    try:
        ws = clone_repo(repo_url, task["base_commit"])
    except RuntimeError as e:
        logger.error("Failed to clone repo: %s", e)
        return RunResult(passed=False, test_output=str(e), attempts=0, invalid=True, reason="CLONE_FAILED")
    
    try:
        hard_reset_clean(ws)

        # Apply SWE-bench test patch (CRITICAL for valid benchmark)
        test_patch = task.get("test_patch", "") or ""
        if test_patch.strip():
            status = apply_patch_text(ws, test_patch)
            if not status.startswith("APPLIED_OK"):
                logger.error("Test patch failed to apply: %s", status)
                return RunResult(
                    passed=False, 
                    test_output=f"INVALID_TEST_PATCH\n{status}", 
                    attempts=0,
                    invalid=True,
                    reason="TEST_PATCH_FAILED"
                )
            logger.debug("Test patch applied successfully")

        # Determine test command
        cmd = derive_test_command_for_repo(task.get("repo", ""), task.get("hints"))
        logger.debug("Test command: %s", cmd)

        # Baseline run (should fail - the bug still exists)
        code, out, rt = _run_cmd(cmd, cwd=ws.path)
        last_out = out
        last_out_baseline = out

        logger.debug("Baseline test: code=%d, runtime=%.1fs", code, rt)
        baseline_failures = _parse_failure_count(last_out_baseline)

        # Try to fix
        attempts = 0
        for attempt_num in range(max_attempts):
            attempts += 1
            logger.info("Attempt %d/%d for %s", attempts, max_attempts, instance_id)

            # Propose patch candidates using upstream intelligence
            try:
                candidates = propose_v2(task, last_out, llm_patch_fn)
            except Exception as e:
                logger.error("Propose failed: %s", e)
                candidates = []

            if not candidates:
                logger.warning("No candidates generated")
                continue

            # Try each candidate serially (serial authority)
            for cand in candidates:
                # Reset to baseline (base_commit + test_patch) before each candidate
                hard_reset_clean(ws)
                if test_patch.strip():
                    apply_patch_text(ws, test_patch)

                # Gate the proposal
                proposal = {
                    "type": "patch",
                    "summary": cand.summary,
                    "patch_text": cand.patch_text,
                    "metadata": cand.metadata,
                }

                state_snapshot = {
                    "repo": task.get("repo"),
                    "base_commit": task.get("base_commit"),
                    "attempt": attempts,
                    "last_test_output_head": (last_out or "")[:2000],
                    "test_cmd": cmd,
                }

                decision = gate.decide(state_snapshot, proposal)
                
                # AUDIT: Gate Decision
                if trace_writer:
                    trace_writer.record({
                        "type": "gate_decision",
                        "allowed": decision.allowed,
                        "reason": decision.reason
                    })
                if trace_reader:
                    trace_reader.verify_gate_decision({
                        "allowed": decision.allowed,
                        "reason": decision.reason 
                    })

                if not decision.allowed:
                    gate_rejections += 1
                    if "security" in decision.reason.lower() or "violation" in decision.reason.lower():
                        security_count += 1
                    logger.debug("Gate rejected: %s", decision.reason)
                    continue

                # Apply the patch
                status = apply_patch_text(ws, cand.patch_text)
                
                # AUDIT: Patch Application
                patch_hash = _hash_str(cand.patch_text)
                if trace_writer:
                    trace_writer.record({
                        "type": "applied_patch",
                        "patch_hash": patch_hash
                    })
                if trace_reader:
                    trace_reader.verify_patch_hash(patch_hash)

                if not status.startswith("APPLIED_OK"):
                    last_out = f"PATCH_APPLY_FAILED\n{status}"
                    logger.debug("Patch apply failed: %s", status[:100])
                    continue

                # Run tests
                code, out, rt = _run_cmd(cmd, cwd=ws.path)
                last_out = out

                # AUDIT: Test Result
                output_hash = _hash_str(out)
                if trace_writer:
                    trace_writer.record({
                        "type": "test_result",
                        "cmd": cmd,
                        "returncode": code,
                        "output_hash": output_hash
                    })
                if trace_reader:
                    trace_reader.verify_test_result(cmd, code, output_hash)

                passed = (code == 0)
                
                # Update learning
                planner_name = cand.metadata.get("planner", "planner_v1")
                learn_update(planner_name, passed)
                
                # Metrics for learner
                current_failures = _parse_failure_count(out)
                delta = current_failures - baseline_failures
                patch_size = len(cand.patch_text)
                files_touched = cand.patch_text.count("diff --git")
                
                res = RunResult(
                    passed=passed,
                    test_output=out,
                    attempts=attempts,
                    gate_rejections=gate_rejections,
                    security_violations=security_count,
                    test_delta=delta,
                    runtime=rt,
                    patch_size=patch_size,
                    files_touched=files_touched,
                    # We might want to pass error message if failed?
                    reason=cand.summary if not passed else ""
                )
                
                if record_callback:
                    record_callback(res)

                if passed:
                    logger.info("PASS: %s on attempt %d", instance_id, attempts)
                    # Record success pattern for future retrieval
                    failure_index.add(FailureRecord(
                        repo=task.get("repo", "unknown"),
                        signature=(task.get("problem_statement", "") or "")[:2000],
                        patch_summary=cand.summary,
                        metadata={
                            "instance_id": instance_id,
                            "attempt": attempts,
                            "planner": planner_name,
                        },
                    ))
                    
                    return res

        logger.info("FAIL: %s after %d attempts", instance_id, attempts)
        
        # Calculate delta for the LAST attempt
        baseline_failures = _parse_failure_count(last_out_baseline) if 'last_out_baseline' in locals() else 0
        current_failures = _parse_failure_count(last_out)
        delta = current_failures - baseline_failures
        
        return RunResult(
            passed=False, 
            test_output=last_out, 
            attempts=attempts, 
            gate_rejections=gate_rejections, 
            security_violations=security_count,
            test_delta=delta,
            runtime=0.0, # Approximate or last run?
            patch_size=0,
            files_touched=0
        )

    finally:
        if cleanup:
            cleanup_workspace(ws)


def run_batch(
    tasks: list[dict[str, Any]],
    llm_patch_fn: Callable,
    max_attempts: int = 6,
) -> dict[str, RunResult]:
    """
    Run multiple tasks sequentially.
    
    Args:
        tasks: List of task dicts
        llm_patch_fn: Patch generation function
        max_attempts: Max attempts per task
        
    Returns:
        Dict mapping instance_id to RunResult
    """
    results = {}
    for task in tasks:
        instance_id = task.get("instance_id", "unknown")
        repo_url = f"https://github.com/{task['repo']}.git"
        result = run_one_task(task, repo_url, llm_patch_fn, max_attempts)
        results[instance_id] = result
    return results
