import os
from typing import Optional
from rfsn_controller.replay_trace import (
    TraceWriter, TraceReader,
    make_recording_llm_wrapper, make_replay_llm_wrapper
)

_RFSN_RECORD_TRACE = os.environ.get("RFSN_RECORD_TRACE")
_RFSN_REPLAY_TRACE = os.environ.get("RFSN_REPLAY_TRACE")

_trace_writer = TraceWriter(_RFSN_RECORD_TRACE, run_meta={"mode": "record"}) if _RFSN_RECORD_TRACE else None
_trace_reader = TraceReader(_RFSN_REPLAY_TRACE, verify_chain=True) if _RFSN_REPLAY_TRACE else None

def get_llm_patch_fn():
    """Factory for patch-candidate generator. Replay avoids any LLM imports/calls."""
    if _trace_reader is not None:
        return make_replay_llm_wrapper(_trace_reader)

    # Lazy import so replay/verification can import this module with LLM disabled.
    from rfsn_controller.llm.llm_router import get_llm_router
    router = get_llm_router()

    # router should expose a candidate generator; fall back conservatively
    base_fn = getattr(router, "generate_patches", None)
    if base_fn is None:
        base_fn = getattr(router, "generate_patches_for_task", None)
    if base_fn is None:
        raise RuntimeError("LLM router missing generate_patches* method")

    if _trace_writer is not None:
        return make_recording_llm_wrapper(base_fn, _trace_writer)
    return base_fn

def get_active_trace_writer() -> Optional[TraceWriter]:
    return _trace_writer

def get_active_trace_reader() -> Optional[TraceReader]:
    return _trace_reader
