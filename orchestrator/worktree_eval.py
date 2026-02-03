"""Parallel Worktree Evaluator for Patch Candidates.

Evaluates N patch candidates concurrently using git worktrees:
- Runs targeted tests first
- Promotes only the best candidate to full suite
- Keeps authority serialized (only one candidate can be committed as "final")
- Does not touch the gate (gate still validates per-candidate patch application)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class CandidateResult:
    """Result of evaluating a single patch candidate."""
    idx: int
    ok_apply: bool
    ok_targeted: bool
    ok_full: bool
    runtime_s: float
    targeted_runtime_s: float
    full_runtime_s: float
    targeted_out: str
    full_out: str
    patch_size: int
    files_touched: int
    workdir: str


def _run(cmd: list[str], cwd: str, timeout_s: int = 900) -> tuple[int, str]:
    """Execute a command and return exit code and combined output."""
    p = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    return p.returncode, p.stdout


def _safe_rm(path: str) -> None:
    """Safely remove a directory."""
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def _count_patch_stats(unified_diff: str) -> tuple[int, int]:
    """Count patch lines and files touched from unified diff."""
    if not unified_diff:
        return 0, 0
    lines = unified_diff.splitlines()
    files = sum(1 for ln in lines if ln.startswith("diff --git "))
    patch_lines = sum(
        1 for ln in lines
        if ln and not ln.startswith(("diff --git", "index ", "--- ", "+++ "))
    )
    return patch_lines, files


def _apply_patch_git(workdir: str, unified_diff: str) -> bool:
    """Apply patch using git apply. Returns True on success."""
    if not unified_diff.strip():
        return False
    patch_path = Path(workdir) / f".rfsn_tmp_patch_{uuid.uuid4().hex}.diff"
    patch_path.write_text(unified_diff, encoding="utf-8")

    try:
        rc, _ = _run(
            ["git", "apply", "--whitespace=nowarn", str(patch_path)],
            cwd=workdir, timeout_s=60
        )
        return rc == 0
    finally:
        try:
            patch_path.unlink(missing_ok=True)
        except Exception:
            pass


def _make_worktree(base_repo: str, base_ref: str, parent_dir: str) -> str:
    """Create a detached worktree at base_ref under parent_dir."""
    wt_dir = Path(parent_dir) / f"wt_{uuid.uuid4().hex[:10]}"
    wt_dir.mkdir(parents=True, exist_ok=True)

    rc, out = _run(
        ["git", "worktree", "add", "--detach", str(wt_dir), base_ref],
        cwd=base_repo, timeout_s=120
    )
    if rc != 0:
        raise RuntimeError(f"git worktree add failed:\n{out}")
    return str(wt_dir)


def _remove_worktree(base_repo: str, wt_dir: str) -> None:
    """Remove a git worktree."""
    try:
        _run(
            ["git", "worktree", "remove", "--force", wt_dir],
            cwd=base_repo, timeout_s=60
        )
    except Exception:
        pass
    _safe_rm(wt_dir)


def _run_tests(
    workdir: str,
    targeted_cmd: list[str],
    full_cmd: list[str],
    targeted_timeout_s: int = 420,
    full_timeout_s: int = 1200,
) -> tuple[bool, float, str, bool, float, str]:
    """Run targeted then full tests. Returns (targeted_ok, time, out, full_ok, time, out)."""
    t0 = time.time()
    rc_t, out_t = _run(targeted_cmd, cwd=workdir, timeout_s=targeted_timeout_s)
    t1 = time.time()

    ok_targeted = (rc_t == 0)
    if not ok_targeted:
        return False, (t1 - t0), out_t, False, 0.0, ""

    t2 = time.time()
    rc_f, out_f = _run(full_cmd, cwd=workdir, timeout_s=full_timeout_s)
    t3 = time.time()

    ok_full = (rc_f == 0)
    return True, (t1 - t0), out_t, ok_full, (t3 - t2), out_f


def evaluate_candidates_in_parallel(
    *,
    base_repo: str,
    base_ref: str,
    candidates: list[dict[str, Any]],
    targeted_test_cmd: list[str],
    full_test_cmd: list[str],
    max_workers: int = 4,
    keep_best_worktree: bool = True,
) -> tuple[int | None, list[CandidateResult], str | None]:
    """
    Evaluate patch candidates in parallel using git worktrees.

    Args:
        base_repo: Path to base repository
        base_ref: Git ref to checkout (e.g., "HEAD")
        candidates: List of dicts with "patch" or "diff" key
        targeted_test_cmd: Command for targeted tests
        full_test_cmd: Command for full test suite
        max_workers: Number of parallel workers
        keep_best_worktree: If True, keep the best worktree for further use

    Returns:
        (best_idx, results, best_workdir) where:
        - best_idx: Index of best candidate or None
        - results: List of CandidateResult
        - best_workdir: Path to best worktree if kept, else None

    Selection rule:
        - prefer ok_full
        - then minimal patch_size
        - then minimal full_runtime
    """
    parent = tempfile.mkdtemp(prefix="rfsn_wt_eval_")
    results: list[CandidateResult] = []

    def _one(i: int, patch: str) -> CandidateResult:
        wt = _make_worktree(base_repo, base_ref, parent_dir=parent)
        t0 = time.time()
        ok_apply = _apply_patch_git(wt, patch)
        targeted_ok = False
        full_ok = False
        targ_rt = 0.0
        full_rt = 0.0
        out_t = ""
        out_f = ""

        if ok_apply:
            targeted_ok, targ_rt, out_t, full_ok, full_rt, out_f = _run_tests(
                wt, targeted_test_cmd, full_test_cmd
            )

        patch_size, files_touched = _count_patch_stats(patch)
        total_rt = time.time() - t0

        return CandidateResult(
            idx=i,
            ok_apply=ok_apply,
            ok_targeted=targeted_ok,
            ok_full=full_ok,
            runtime_s=total_rt,
            targeted_runtime_s=targ_rt,
            full_runtime_s=full_rt,
            targeted_out=out_t,
            full_out=out_f,
            patch_size=patch_size,
            files_touched=files_touched,
            workdir=wt,
        )

    # Run candidates in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        for i, c in enumerate(candidates):
            patch = c.get("patch") or c.get("diff") or ""
            futs.append(ex.submit(_one, i, patch))

        for f in as_completed(futs):
            r = f.result()
            results.append(r)

    # Choose best candidate
    ok_full = [r for r in results if r.ok_full]
    if ok_full:
        ok_full.sort(key=lambda r: (r.patch_size, r.full_runtime_s, r.runtime_s))
        best = ok_full[0]
    else:
        ok_t = [r for r in results if r.ok_targeted]
        if ok_t:
            ok_t.sort(key=lambda r: (r.patch_size, r.targeted_runtime_s, r.runtime_s))
            best = ok_t[0]
        else:
            best = None

    best_idx = best.idx if best else None
    best_workdir = best.workdir if (best and keep_best_worktree) else None

    # Cleanup non-best worktrees
    for r in results:
        if best_workdir and r.workdir == best_workdir:
            continue
        _remove_worktree(base_repo, r.workdir)

    # Remove parent dir if no best kept
    if not best_workdir:
        _safe_rm(parent)

    return best_idx, sorted(results, key=lambda x: x.idx), best_workdir
