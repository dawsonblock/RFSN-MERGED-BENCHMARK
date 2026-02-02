"""Ledger logging - append-only audit trail.

Every proposal → gate → execution creates one ledger entry.
This is the ONLY way to learn: read past ledger events.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from agent.types import AgentState, LedgerEvent

# Try to import logger, fall back to basic logging if not available
try:
    from rfsn_controller.structured_logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def append_event(state: AgentState, event: LedgerEvent | dict) -> None:
    """Append a ledger event to the log file.
    
    Events are written as JSONL (one JSON object per line).
    This is append-only - never modify past events.
    
    Args:
        state: Current agent state
        event: Event to log (LedgerEvent or dict)
        
    Example:
        >>> event = LedgerEvent.now(...)
        >>> append_event(state, event)
        >>> # Or with dict:
        >>> append_event(state, {"event": "task_start", "task_id": "..."})
    """
    run_dir = state.notes.get("run_dir")
    if not run_dir:
        # Default: workdir/runs/<task_id>
        run_dir = os.path.join(state.repo.workdir, "runs", state.task_id)
        os.makedirs(run_dir, exist_ok=True)
        state.notes["run_dir"] = run_dir
    
    log_path = Path(run_dir) / "events.jsonl"
    
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            # Handle both LedgerEvent and plain dict
            if isinstance(event, dict):
                # Simple dict event - just add timestamp
                import time
                event_dict = {"ts_unix": time.time(), **event}
            else:
                # Convert LedgerEvent dataclass to dict
                event_dict = {
                    "ts_unix": event.ts_unix,
                    "task_id": event.task_id,
                    "repo_id": event.repo_id,
                    "phase": event.phase.value if hasattr(event.phase, 'value') else str(event.phase),
                    "proposal_hash": event.proposal_hash,
                    "proposal": event.proposal,
                    "gate": event.gate,
                    "exec": event.exec,
                    "result": event.result,
                }
            f.write(json.dumps(event_dict, ensure_ascii=False) + "\n")
        
        task_id = event.get("task_id") if isinstance(event, dict) else event.task_id
        logger.debug("Logged event", task_id=task_id)
        
    except Exception as e:
        task_id = event.get("task_id", "unknown") if isinstance(event, dict) else getattr(event, "task_id", "unknown")
        logger.error("Failed to log event", error=str(e), task_id=task_id)


def read_ledger(run_dir: str) -> list[LedgerEvent]:
    """Read all events from a ledger file.
    
    Args:
        run_dir: Run directory containing events.jsonl
        
    Returns:
        List of ledger events
        
    Example:
        >>> events = read_ledger("runs/task_001")
        >>> for event in events:
        ...     print(event.phase, event.proposal["kind"])
    """
    log_path = Path(run_dir) / "events.jsonl"
    
    if not log_path.exists():
        logger.warning("Ledger file not found", path=str(log_path))
        return []
    
    events = []
    
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    event_dict = json.loads(line)
                    # Reconstruct LedgerEvent (simplified - doesn't restore dataclass)
                    events.append(event_dict)
                except json.JSONDecodeError as e:
                    logger.error(
                        "Failed to parse event",
                        line_num=line_num,
                        error=str(e),
                    )
        
        logger.info("Loaded ledger", path=str(log_path), events=len(events))
        return events
        
    except Exception as e:
        logger.error("Failed to read ledger", error=str(e), path=str(log_path))
        return []


def get_last_n_events(run_dir: str, n: int = 10) -> list[dict]:
    """Get last N events from ledger.
    
    Args:
        run_dir: Run directory
        n: Number of recent events
        
    Returns:
        List of recent events (most recent last)
    """
    events = read_ledger(run_dir)
    return events[-n:] if len(events) > n else events


def filter_events(
    events: list[dict],
    phase: str | None = None,
    proposal_kind: str | None = None,
    gate_accept: bool | None = None,
) -> list[dict]:
    """Filter events by criteria.
    
    Args:
        events: List of events
        phase: Filter by phase
        proposal_kind: Filter by proposal kind
        gate_accept: Filter by gate decision
        
    Returns:
        Filtered events
    """
    filtered = events
    
    if phase:
        filtered = [e for e in filtered if e.get("phase") == phase]
    
    if proposal_kind:
        filtered = [
            e for e in filtered
            if e.get("proposal", {}).get("kind") == proposal_kind
        ]
    
    if gate_accept is not None:
        filtered = [
            e for e in filtered
            if e.get("gate", {}).get("accept") == gate_accept
        ]
    
    return filtered
