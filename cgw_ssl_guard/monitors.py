"""Monitoring utilities for the CGW/SSL guard runtime.

This module contains helper classes that consume events from the
event bus and record simple diagnostics.  They are used in tests to
assert that only one commit occurs per cycle and that events are
emitted in the expected order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SerialityMonitor:
    """Track the number of commits per cycle to detect parallelism.

    Subscribing this monitor to CGW_COMMIT events on the event bus
    allows tests to assert that no cycle has more than one commit.
    """

    commits_per_cycle: Dict[int, int] = field(default_factory=dict)

    def on_commit(self, event: Dict) -> None:
        cycle_id = event.get("cycle_id")
        if cycle_id is not None:
            self.commits_per_cycle[cycle_id] = self.commits_per_cycle.get(cycle_id, 0) + 1