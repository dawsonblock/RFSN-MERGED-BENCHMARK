"""Parallel patch evaluation for the RFSN controller.

This module provides utilities for evaluating multiple candidate patches
in parallel using concurrent.futures, significantly reducing the time spent
testing patches from different temperature samples.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass

from .sandbox import DockerResult, Sandbox, apply_patch_in_dir, docker_run, drop_worktree, make_worktree, run_cmd


@dataclass
class PatchResult:
    """Result of evaluating a single patch."""

    diff: str
    diff_hash: str
    ok: bool
    info: str
    temperature: float



def _sanitize_cmd(cmd: str) -> str:
    """Sanitize command for Docker execution."""
    cmd = cmd.strip()
    if cmd.startswith("pytest"):
        return "python -m " + cmd
    return cmd


def _evaluate_single_patch(
    sb: Sandbox,
    diff: str,
    diff_hash: str,
    focus_cmd: str,
    full_cmd: str,
    temperature: float,
    docker_image: str,
    cpu: float,
    mem_mb: int,
    durability_reruns: int = 0,
    unsafe_host_exec: bool = False,
) -> PatchResult:
    """Evaluate a single patch in an isolated worktree.

    Args:
        sb: The sandbox containing the repository.
        diff: The unified diff to apply.
        diff_hash: Hash of the diff for deduplication.
        focus_cmd: Focused test command for quick feedback.
        full_cmd: Full test command for verification.
        temperature: Temperature used to generate this patch.
        unsafe_host_exec: If True, run commands locally instead of in Docker.

    Returns:
        A PatchResult with evaluation outcome.
    """
    import time
    start_time = time.time()
    
    wt = None
    try:
        wt = make_worktree(sb, suffix=diff_hash[:10])
        ap = apply_patch_in_dir(wt, diff)
        if not ap.get("ok"):
            result = PatchResult(
                diff=diff,
                diff_hash=diff_hash,
                ok=False,
                info=f"apply_failed: {ap.get('stderr', '')}{ap.get('stdout', '')}",
                temperature=temperature,
            )
            _track_patch_result(diff, result, time.time() - start_time)
            return result
            _track_patch_result(diff, result, time.time() - start_time)
            return result
        
        # Helper function to run command either locally or in Docker
        def execute_cmd(cmd: str, timeout: int) -> DockerResult:
            if unsafe_host_exec:
                # Run locally using run_cmd
                result = run_cmd(Sandbox(sb.root, wt), cmd, timeout_sec=timeout)
                return DockerResult(
                    ok=result.get('ok', False),
                    exit_code=result.get('exit_code', 1),
                    stdout=result.get('stdout', ''),
                    stderr=result.get('stderr', ''),
                    timed_out=False,
                )
            else:
                return docker_run(
                    Sandbox(sb.root, wt), 
                    cmd, 
                    timeout_sec=timeout,
                    docker_image=docker_image,
                    cpu=cpu,
                    mem_mb=mem_mb
                )
        
        r1 = execute_cmd(focus_cmd, 90)
        if not r1.ok:
            result = PatchResult(
                diff=diff,
                diff_hash=diff_hash,
                ok=False,
                info="focus_failed:\n" + (r1.stdout + r1.stderr),
                temperature=temperature,
            )
            _track_patch_result(diff, result, time.time() - start_time)
            return result
        
        r2 = execute_cmd(full_cmd, 180)
        if r2.ok:
            # Durability reruns: run full suite additional times to detect flakiness
            for k in range(int(durability_reruns or 0)):
                rN = execute_cmd(full_cmd, 180)
                if not rN.ok:
                    result = PatchResult(
                        diff=diff,
                        diff_hash=diff_hash,
                        ok=False,
                        info=f"durability_failed_run_{k+1}:\n" + (rN.stdout + rN.stderr),
                        temperature=temperature,
                    )
                    _track_patch_result(diff, result, time.time() - start_time)
                    return result
            result = PatchResult(
                diff=diff,
                diff_hash=diff_hash,
                ok=True,
                info="PASS",
                temperature=temperature,
            )
            _track_patch_result(diff, result, time.time() - start_time)
            return result
        result = PatchResult(
            diff=diff,
            diff_hash=diff_hash,
            ok=False,
            info="full_failed:\n" + (r2.stdout + r2.stderr),
            temperature=temperature,
        )
        _track_patch_result(diff, result, time.time() - start_time)
        return result
    except Exception as e:
        result = PatchResult(
            diff=diff,
            diff_hash=diff_hash,
            ok=False,
            info=f"exception: {type(e).__name__}: {e!s}",
            temperature=temperature,
        )
        _track_patch_result(diff, result, time.time() - start_time)
        return result
    finally:
        if wt:
            try:
                drop_worktree(sb, wt)
            except Exception:
                pass


def _track_patch_result(diff: str, result: PatchResult, duration_sec: float) -> None:
    """Track patch evaluation with telemetry."""
    try:
        from .telemetry import track_patch_evaluation
        track_patch_evaluation(
            diff=diff,
            model="unknown",  # Model info not available here
            status="pass" if result.ok else "fail",
            duration_sec=duration_sec,
        )
    except ImportError:
        pass  # Telemetry not available


def evaluate_patches_parallel(
    sb: Sandbox,
    patches: list[tuple[str, float]],  # List of (diff, temperature)
    focus_cmd: str,
    full_cmd: str,
    docker_image: str,
    cpu: float,
    mem_mb: int,
    durability_reruns: int = 0,
    max_workers: int = 3,
    unsafe_host_exec: bool = False,
) -> list[PatchResult]:
    """Evaluate multiple patches in parallel using thread pool.

    Args:
        sb: The sandbox containing the repository.
        patches: List of (diff, temperature) tuples to evaluate.
        focus_cmd: Focused test command for quick feedback.
        full_cmd: Full test command for verification.
        max_workers: Maximum number of parallel evaluations.
        unsafe_host_exec: If True, run commands locally instead of in Docker.

    Returns:
        List of PatchResult objects in the same order as input patches.
    """
    import hashlib

    # Pre-compute hashes and create index mapping
    indexed_patches = []
    for idx, (diff, temp) in enumerate(patches):
        diff_hash = hashlib.sha256((diff or "").encode("utf-8", errors="ignore")).hexdigest()
        indexed_patches.append((idx, diff, temp, diff_hash))

    results: list[PatchResult | None] = [None] * len(patches)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all patch evaluations
        future_to_index = {}
        for idx, diff, temp, diff_hash in indexed_patches:
            future = executor.submit(
                _evaluate_single_patch,
                sb,
                diff,
                diff_hash,
                _sanitize_cmd(focus_cmd),
                _sanitize_cmd(full_cmd),
                temp,
                docker_image,
                cpu,
                mem_mb,
                durability_reruns,
                unsafe_host_exec,
            )
            future_to_index[future] = idx

        # Collect results as they complete, preserving order
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                result = future.result()
                results[idx] = result
            except Exception as e:
                # If evaluation itself fails, create a failure result
                _, diff, temp, diff_hash = indexed_patches[idx]
                results[idx] = PatchResult(
                    diff=diff,
                    diff_hash=diff_hash,
                    ok=False,
                    info=f"evaluation_exception: {type(e).__name__}: {e!s}",
                    temperature=temp,
                )

    return [r for r in results if r is not None]


def find_first_successful_patch(results: list[PatchResult]) -> PatchResult | None:
    """Find the first successful patch from evaluation results.

    Args:
        results: List of PatchResult objects.

    Returns:
        The first PatchResult with ok=True, or None if none succeeded.
    """
    for result in results:
        if result.ok:
            return result
    return None
