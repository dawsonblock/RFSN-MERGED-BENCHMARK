"""Top‑level package for CGW/SSL guard step‑1 runtime.

This package provides a simple event bus, a thalamic gate with a forced
override queue, a single‑slot conscious global workspace (CGW) state,
and helpers for monitoring and deterministic testing.  The goal of
this runtime is to enforce that exactly one signal wins per cycle and
that forced signals bypass normal competition.

The code here is intentionally minimal: the focus is on making the
control flow easy to audit and reason about, not on performance.  The
gate and runtime emit events whenever a winner is selected or a CGW
state is committed, which can be consumed by monitors or test
harnesses.
"""

from .event_bus import SimpleEventBus
from .thalamic_gate import ThalamusGate, Candidate, ForcedCandidate, SelectionReason
from .cgw_state import CGWRuntime, AttendedContent, CausalTrace, CGWState, SelfModel
from .runtime import Runtime
from .monitors import SerialityMonitor