# rfsn_controller/executor_spine.py
"""Unified governed execution spine.

This is the ONLY place allowed to:
- run subprocess commands
- apply diffs
- mutate the repo

All modes (planner loop, CGW runtime, bridge) must route execution here.

Invariants enforced:
- argv-only subprocess execution (no shell strings)
- global allowlist enforcement
- optional per-profile allowlist enforcement
- patch hygiene + immutable control path protection
- verifier-first (caller chooses when to verify; helper provided)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exec_utils import ExecResult, safe_run
from .patch_hygiene import validate_patch_hygiene
from .sandbox import apply_patch_in_dir


@dataclass
class StepExecResult:
    step_id: str
    ok: bool
    elapsed_ms: int
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    details: dict[str, Any] = None


class GovernedExecutor:
    """Single execution spine for the whole repo."""

    def __init__(
        self,
        repo_dir: str | Path,
        *,
        allowed_commands: set[str] | None = None,
        verify_argv: list[str] | None = None,
        timeout_sec: int = 180,
    ):
        self.repo_dir = str(Path(repo_dir).resolve())
        self.allowed_commands = allowed_commands
        self.verify_argv = verify_argv
        self.timeout_sec = timeout_sec

    # ----------------------------
    # Public API
    # ----------------------------

    def execute_step(self, step: dict[str, Any]) -> StepExecResult:
        """Execute a single step dict produced by planner/CGW.

        Step format (minimum):
          {"id": "...", "type": "<step_type>", ...}

        Supported step types:
          - run_cmd: {"argv": [...], "timeout_sec": int?}
          - run_tests: {"argv": [...]} or uses self.verify_argv
          - apply_patch: {"diff": "unified diff text"}
          - reset_hard: {}
          - read_file: {"path": "relative/path"}
          - grep: {"query": "text", "max_matches": int?}
        """
        start = time.monotonic()
        sid = step.get("id", "unknown")
        stype = step.get("type")

        try:
            if stype == "run_cmd":
                argv = step.get("argv")
                if not isinstance(argv, list) or not argv:
                    return self._fail(sid, start, "run_cmd requires argv: List[str]")
                timeout = int(step.get("timeout_sec", self.timeout_sec))
                r = self._run_argv(argv, timeout_sec=timeout)
                return self._from_exec(sid, start, r)

            if stype == "run_tests":
                argv = step.get("argv") or self.verify_argv
                if not argv:
                    return self._fail(sid, start, "run_tests requires argv or executor.verify_argv")
                timeout = int(step.get("timeout_sec", max(self.timeout_sec, 300)))
                r = self._run_argv(argv, timeout_sec=timeout)
                return self._from_exec(sid, start, r)

            if stype == "apply_patch":
                diff = step.get("diff", "")
                if not diff.strip():
                    return self._fail(sid, start, "apply_patch requires non-empty diff")

                hygiene = validate_patch_hygiene(diff)
                if not hygiene.is_valid:
                    return StepExecResult(
                        step_id=sid,
                        ok=False,
                        elapsed_ms=self._elapsed_ms(start),
                        stdout="",
                        stderr="; ".join(hygiene.violations),
                        exit_code=1,
                        details={"violations": hygiene.violations},
                    )

                pr = apply_patch_in_dir(self.repo_dir, diff)
                ok = bool(pr.get("ok"))
                return StepExecResult(
                    step_id=sid,
                    ok=ok,
                    elapsed_ms=self._elapsed_ms(start),
                    stdout=str(pr.get("stdout", "")),
                    stderr=str(pr.get("stderr", "")),
                    exit_code=int(pr.get("exit_code", 1 if not ok else 0)),
                    details={"applied": ok},
                )

            if stype == "reset_hard":
                r = self._run_argv(["git", "reset", "--hard", "HEAD"], timeout_sec=60)
                return self._from_exec(sid, start, r)

            if stype == "read_file":
                rel = step.get("path")
                if not isinstance(rel, str) or not rel:
                    return self._fail(sid, start, "read_file requires path")
                p = (Path(self.repo_dir) / rel).resolve()
                if not self._is_within_repo(p):
                    return self._fail(sid, start, f"path escapes repo: {rel}")
                if not p.exists() or not p.is_file():
                    return self._fail(sid, start, f"file not found: {rel}")
                data = p.read_text(encoding="utf-8", errors="replace")
                return StepExecResult(
                    step_id=sid,
                    ok=True,
                    elapsed_ms=self._elapsed_ms(start),
                    stdout=data,
                    stderr="",
                    exit_code=0,
                    details={"path": rel, "bytes": len(data.encode("utf-8", "ignore"))},
                )

            if stype == "grep":
                query = step.get("query")
                if not isinstance(query, str) or not query:
                    return self._fail(sid, start, "grep requires query")
                max_matches = int(step.get("max_matches", 200))
                matches = self._grep_repo(query, max_matches=max_matches)
                out = "\n".join(matches)
                return StepExecResult(
                    step_id=sid,
                    ok=True,
                    elapsed_ms=self._elapsed_ms(start),
                    stdout=out,
                    stderr="",
                    exit_code=0,
                    details={"matches": len(matches)},
                )

            return self._fail(sid, start, f"Unknown step type: {stype}")

        except Exception as e:
            return StepExecResult(
                step_id=sid,
                ok=False,
                elapsed_ms=self._elapsed_ms(start),
                stdout="",
                stderr=str(e),
                exit_code=1,
                details={"exception": type(e).__name__},
            )

    def verify(self) -> StepExecResult:
        """Run the executor's configured verifier command."""
        sid = "verify"
        start = time.monotonic()
        if not self.verify_argv:
            return self._fail(sid, start, "No verify_argv configured")
        r = self._run_argv(self.verify_argv, timeout_sec=max(self.timeout_sec, 300))
        return self._from_exec(sid, start, r)

    # ----------------------------
    # Internals
    # ----------------------------

    def _run_argv(self, argv: list[str], *, timeout_sec: int) -> ExecResult:
        return safe_run(
            argv=argv,
            cwd=self.repo_dir,
            timeout_sec=timeout_sec,
            allowed_commands=self.allowed_commands,
            check_global_allowlist=True,
        )

    def _grep_repo(self, query: str, *, max_matches: int) -> list[str]:
        root = Path(self.repo_dir)
        hits: list[str] = []
        # fast-ish grep without calling shell
        for p in root.rglob("*"):
            if len(hits) >= max_matches:
                break
            if p.is_symlink() or not p.is_file():
                continue
            # skip huge/binary-ish files
            try:
                if p.stat().st_size > 512_000:
                    continue
            except Exception:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if query in text:
                rel = str(p.relative_to(root))
                # include first line number hit
                idx = text.find(query)
                line = text[:idx].count("\n") + 1
                hits.append(f"{rel}:{line}: {query}")
        return hits

    def _is_within_repo(self, p: Path) -> bool:
        try:
            p = p.resolve()
            root = Path(self.repo_dir).resolve()
            return root == p or root in p.parents
        except Exception:
            return False

    def _elapsed_ms(self, start: float) -> int:
        return int((time.monotonic() - start) * 1000)

    def _from_exec(self, sid: str, start: float, r: ExecResult) -> StepExecResult:
        return StepExecResult(
            step_id=sid,
            ok=bool(r.ok),
            elapsed_ms=self._elapsed_ms(start),
            stdout=r.stdout,
            stderr=r.stderr,
            exit_code=int(r.exit_code),
            details={"command": r.command, "timed_out": r.timed_out},
        )

    def _fail(self, sid: str, start: float, msg: str) -> StepExecResult:
        return StepExecResult(
            step_id=sid,
            ok=False,
            elapsed_ms=self._elapsed_ms(start),
            stdout="",
            stderr=msg,
            exit_code=1,
            details={},
        )
