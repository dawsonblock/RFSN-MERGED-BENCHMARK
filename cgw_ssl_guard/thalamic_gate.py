"""Thalamic gate implementation with forced override.

The thalamic gate maintains two collections of signals:

* ``forced_queue``: a FIFO queue of ``ForcedCandidate`` objects.  If any
  forced signals are present, the gate will select the oldest forced
  signal without regard to scores.
* ``candidates``: a list of ``Candidate`` objects that compete via a
  weighted sum of saliency, urgency and surprise.

On each call to ``select_winner()``, the gate checks the forced
queue first.  If non‑empty, the forced signal is popped and wins.
Otherwise, the gate scores all candidates and chooses the highest.

The gate emits a ``SelectionEvent`` via an event bus when a winner is
selected, including the list of losers and whether the winner came
from the forced queue.  After selection, the candidate list is
cleared for the next cycle.
"""

from __future__ import annotations

import time
from collections import deque
from typing import List, Optional, Tuple, Union

from .event_bus import SimpleEventBus
from .types import (
    Candidate,
    ForcedCandidate,
    SelectionEvent,
    SelectionReason,
)


class ThalamusGate:
    """Arbitrate between competing signals for CGW admission."""

    def __init__(self, event_bus: SimpleEventBus) -> None:
        self.event_bus = event_bus
        self.forced_queue: deque[ForcedCandidate] = deque()
        self.candidates: List[Candidate] = []
        self.cycle_counter: int = 0
        self.last_selection_time: float = 0.0
        # parameters controlling normal competition
        self.max_candidates_per_cycle: int = 20
        self.competition_cooldown_ms: int = 100  # minimum time between selections

    def inject_forced_signal(self, *, source_module: str, content_payload: bytes, reason: str = "FORCED_OVERRIDE") -> str:
        """Inject a forced signal.  Returns the assigned slot id.

        The slot id is unique per forced injection and is based on the
        current time.  Forced signals bypass normal competition.
        """
        slot_id = f"forced_{int(time.time() * 1e6)}"
        fc = ForcedCandidate(slot_id=slot_id, source_module=source_module, content_payload=content_payload)
        self.forced_queue.append(fc)
        # emit injection event for diagnostics
        self.event_bus.emit("FORCED_INJECTION", {
            "slot_id": slot_id,
            "source": source_module,
            "timestamp": time.time(),
            "queue_depth": len(self.forced_queue)
        })
        return slot_id

    def submit_candidate(self, candidate: Candidate) -> None:
        """Submit a normal candidate for competition."""
        # enforce bounded queue size
        if len(self.candidates) >= self.max_candidates_per_cycle:
            # drop lowest scoring candidate
            self.candidates.sort(key=lambda c: c.score(), reverse=True)
            self.candidates = self.candidates[: self.max_candidates_per_cycle]
        self.candidates.append(candidate)

    def select_winner(self) -> Tuple[Optional[Union[ForcedCandidate, Candidate]], SelectionReason]:
        """Select the next winner based on forced queue or scoring.

        Returns (winner, reason).  ``winner`` may be None if idle.
        """
        self.cycle_counter += 1
        now = time.time()

        # first check forced queue
        if self.forced_queue:
            fc = self.forced_queue.popleft()
            reason = SelectionReason.FORCED_OVERRIDE
            # capture losers (all normal candidates)
            loser_ids = [c.slot_id for c in self.candidates]
            # emit event
            event = SelectionEvent(
                cycle_id=self.cycle_counter,
                slot_id=fc.slot_id,
                reason=reason,
                timestamp=now,
                forced_queue_size=len(self.forced_queue),
                losers=loser_ids,
                winner_is_forced=True,
            )
            self.event_bus.emit("GATE_SELECTION", event)
            # clear normal candidates
            self.candidates.clear()
            return fc, reason

        # no forced signals, normal competition
        if self.candidates:
            # cooldown: ensure a minimum time between selections
            if (now - self.last_selection_time) * 1000 < self.competition_cooldown_ms:
                return None, SelectionReason.COMPETITION
            # sort by score descending
            self.candidates.sort(key=lambda c: c.score(), reverse=True)
            winner = self.candidates[0]
            loser_ids = [c.slot_id for c in self.candidates[1:]]
            # determine reason (informational only)
            if winner.urgency > 0.8:
                reason = SelectionReason.URGENCY
            elif winner.surprise > 0.8:
                reason = SelectionReason.SURPRISE
            else:
                reason = SelectionReason.COMPETITION
            event = SelectionEvent(
                cycle_id=self.cycle_counter,
                slot_id=winner.slot_id,
                reason=reason,
                timestamp=now,
                forced_queue_size=0,
                losers=loser_ids,
                winner_is_forced=False,
            )
            self.event_bus.emit("GATE_SELECTION", event)
            # clear candidates and update last selection time
            self.candidates.clear()
            self.last_selection_time = now
            return winner, reason
        # idle: nothing to select
        return None, SelectionReason.COMPETITION