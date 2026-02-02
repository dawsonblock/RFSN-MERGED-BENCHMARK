"""Implementation of the conscious global workspace state (CGW).

This module holds the concrete representation of the workspace at any
moment in time.  The state is intentionally immutable once created:
each update constructs a new ``CGWState`` object off to the side and
then swaps it into place.  This ensures that no partial updates are
visible and that the ``AttendedContent`` slot remains single and
authoritative.

The runtime also emits ``CGW_COMMIT`` events on the event bus when a
new state is committed.  These events carry minimal identifying
information: cycle id, slot id, reason and whether the winner was
forced.
"""

from __future__ import annotations

import hashlib
import time
from typing import Optional, Tuple, Union

from .event_bus import SimpleEventBus
from .types import (
    AttendedContent,
    Candidate,
    CGWState,
    CausalTrace,
    ForcedCandidate,
    SelectionReason,
    SelfModel,
)


class CGWRuntime:
    """Manage the current CGW state and commit new states atomically."""

    def __init__(self, event_bus: SimpleEventBus) -> None:
        self.event_bus = event_bus
        self.current_state: Optional[CGWState] = None
        self.cycle_counter: int = 0

    def update(self, winner: Union[ForcedCandidate, Candidate], reason: SelectionReason, self_model: SelfModel) -> CGWState:
        """Commit a new CGWState based on the winning signal.

        Args:
            winner: The winning ``Candidate`` or ``ForcedCandidate``.
            reason: The selection reason as returned by the gate.
            self_model: The persistent self model for this cycle.

        Returns:
            The newly committed ``CGWState`` instance.
        """
        self.cycle_counter += 1
        now = time.time()

        # extract payload
        slot_id = winner.slot_id
        payload = winner.content_payload
        source = winner.source_module
        forced = isinstance(winner, ForcedCandidate)
        winner_score = winner.score() if isinstance(winner, Candidate) else None
        payload_hash = hashlib.sha256(payload).hexdigest()
        attended = AttendedContent(
            slot_id=slot_id,
            payload_hash=payload_hash,
            payload_bytes=payload,
            source_module=source,
            timestamp=now,
        )
        trace = CausalTrace(
            winner_reason=reason,
            winner_score=winner_score,
            losers=[],
            forced_override=forced,
        )
        new_state = CGWState(
            cycle_id=self.cycle_counter,
            timestamp=now,
            attended_content=attended,
            causal_trace=trace,
            self_model=self_model,
        )
        # swap the state atomically
        old_state = self.current_state
        self.current_state = new_state
        # emit commit event
        self.event_bus.emit("CGW_COMMIT", {
            "cycle_id": new_state.cycle_id,
            "slot_id": new_state.content_id(),
            "reason": reason.value,
            "forced": forced,
            "timestamp": now,
        })
        return new_state

    def get_current_state(self) -> Optional[CGWState]:
        return self.current_state