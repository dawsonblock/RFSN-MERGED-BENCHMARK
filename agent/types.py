"""Core types and contracts for the SWE-bench agent.

This module defines all the fundamental data structures that glue together:
- Phases (serial state machine)
- Proposals (what planner emits)
- Gate decisions (accept/reject with reasoning)
- Execution results (what happened)
- Ledger events (append-only audit trail)
- Agent state (what the gate sees)

All learning lives OUTSIDE these contracts. The gate is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Literal
import hashlib
import json
import time


# =========================
# Phases (serial state machine)
# =========================

class Phase(str, Enum):
    """Agent phases in a serial state machine.
    
    INGEST: Load problem statement + repo snapshot
    LOCALIZE: Find likely files/regions
    PLAN: Decompose into small steps
    PATCH_CANDIDATES: Generate N candidate patches
    TEST_STAGE: Run staged tests (0→1→2→3)
    DIAGNOSE: Analyze failures, update hypotheses
    MINIMIZE: Shrink successful patch
    FINALIZE: Commit and close
    DONE: Episode complete
    """
    INGEST = "INGEST"
    LOCALIZE = "LOCALIZE"
    PLAN = "PLAN"
    PATCH_CANDIDATES = "PATCH_CANDIDATES"
    TEST_STAGE = "TEST_STAGE"
    DIAGNOSE = "DIAGNOSE"
    MINIMIZE = "MINIMIZE"
    FINALIZE = "FINALIZE"
    DONE = "DONE"


# =========================
# Proposal contract (planner emits exactly this)
# =========================

ProposalKind = Literal["inspect", "search", "edit", "run_tests", "finalize"]


@dataclass(frozen=True)
class Evidence:
    """Evidence supporting a proposal (for citations).
    
    Type can be:
    - trace: from stack trace parsing
    - grep: from ripgrep search
    - embed: from semantic search
    - manual: human-provided
    """
    type: Literal["trace", "grep", "embed", "manual"]
    file: Optional[str] = None
    lines: Optional[Tuple[int, int]] = None  # [start, end)
    snippet: Optional[str] = None
    why: str = ""


@dataclass(frozen=True)
class Proposal:
    """A proposal from the planner to the controller.
    
    This is the ONLY interface between planning and execution.
    Every proposal must pass through the gate.
    
    Example:
        >>> proposal = Proposal(
        ...     kind="edit",
        ...     rationale="Fix AttributeError in dataclass init",
        ...     inputs={"files": ["main.py"], "diff": "..."},
        ...     expected_signal={"type": "pass", "pattern": "test_init"},
        ...     evidence=[
        ...         Evidence(type="trace", file="main.py", lines=(42, 50),
        ...                  why="AttributeError on line 45")
        ...     ]
        ... )
    """
    kind: ProposalKind
    rationale: str  # Short, testable hypothesis
    inputs: Dict[str, Any] = field(default_factory=dict)
    expected_signal: Dict[str, Any] = field(default_factory=dict)
    evidence: List[Evidence] = field(default_factory=list)

    def stable_hash(self) -> str:
        """Compute deterministic hash for deduplication."""
        payload = json.dumps({
            "kind": self.kind,
            "rationale": self.rationale,
            "inputs": self.inputs,
            "expected_signal": self.expected_signal,
            "evidence": [e.__dict__ for e in self.evidence],
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# =========================
# Gate decision
# =========================

@dataclass(frozen=True)
class GateDecision:
    """Gate's decision on a proposal.
    
    The gate is DETERMINISTIC. No learning, no randomness.
    It checks:
    - Phase constraints (can this action happen now?)
    - File constraints (touched files, vendor/CI)
    - Test constraints (can't disable tests, must run stages)
    - Diff constraints (size, risk)
    """
    accept: bool
    reason: str
    constraints: Dict[str, Any] = field(default_factory=dict)


# =========================
# Execution + artifacts
# =========================

@dataclass(frozen=True)
class ExecResult:
    """Result of executing a proposal.
    
    Controller executes the proposal and returns this.
    """
    status: Literal["ok", "fail"]
    summary: str
    artifacts: List[str] = field(default_factory=list)  # Paths to log files, diffs, etc.
    metrics: Dict[str, Any] = field(default_factory=dict)


# =========================
# Test results (structured)
# =========================

@dataclass(frozen=True)
class TestFailure:
    """A single test failure."""
    nodeid: str  # pytest node ID
    message: str
    traceback: Optional[str] = None


@dataclass(frozen=True)
class TestResult:
    """Result of running tests at a specific stage."""
    stage: int  # 0=compile, 1=targeted, 2=subset, 3=full
    command: List[str]
    passed: bool
    duration_s: float
    failures: List[TestFailure] = field(default_factory=list)
    stdout_tail: Optional[str] = None
    stderr_tail: Optional[str] = None


# =========================
# Ledger event (append-only)
# =========================

@dataclass(frozen=True)
class LedgerEvent:
    """Append-only audit trail event.
    
    Every proposal → gate → execution creates one ledger entry.
    This is the ONLY way to learn: read past ledger events.
    """
    ts_unix: float
    task_id: str
    repo_id: str
    phase: Phase
    proposal_hash: str
    proposal: Dict[str, Any]
    gate: Dict[str, Any]
    exec: Dict[str, Any]
    result: Dict[str, Any]

    @staticmethod
    def now(
        task_id: str,
        repo_id: str,
        phase: Phase,
        proposal: Proposal,
        gate_decision: GateDecision,
        exec_result: ExecResult,
        result: Dict[str, Any]
    ) -> "LedgerEvent":
        """Create a ledger event for the current moment."""
        return LedgerEvent(
            ts_unix=time.time(),
            task_id=task_id,
            repo_id=repo_id,
            phase=phase,
            proposal_hash=proposal.stable_hash(),
            proposal={
                "kind": proposal.kind,
                "rationale": proposal.rationale,
                "inputs": proposal.inputs,
                "expected_signal": proposal.expected_signal,
                "evidence": [
                    {
                        "type": e.type,
                        "file": e.file,
                        "lines": e.lines,
                        "snippet": e.snippet,
                        "why": e.why,
                    }
                    for e in proposal.evidence
                ],
            },
            gate=gate_decision.__dict__,
            exec=exec_result.__dict__,
            result=result,
        )


# =========================
# Agent state snapshot (what gate sees)
# =========================

@dataclass
class RepoFingerprint:
    """Stable identifier for a repo + commit."""
    repo_id: str  # org/repo@commit or task workspace hash
    commit_sha: str
    workdir: str  # Path to isolated workspace
    language: str = "python"


@dataclass
class BudgetState:
    """Resource budgets (prevent infinite loops)."""
    max_rounds: int
    round_idx: int = 0
    max_patch_attempts: int = 40
    patch_attempts: int = 0
    max_test_runs: int = 80
    test_runs: int = 0
    max_model_calls: int = 200
    model_calls: int = 0


@dataclass
class AgentState:
    """Complete agent state snapshot.
    
    This is what the gate sees when making decisions.
    Everything here is deterministic and auditable.
    """
    task_id: str
    repo: RepoFingerprint
    phase: Phase
    budget: BudgetState
    
    # Rolling context
    last_failures: List[TestFailure] = field(default_factory=list)
    touched_files: List[str] = field(default_factory=list)
    localization_hits: List[Dict[str, Any]] = field(default_factory=list)
    
    # Scratch space for planner state
    notes: Dict[str, Any] = field(default_factory=dict)
