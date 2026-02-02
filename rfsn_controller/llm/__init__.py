"""LLM Client Package."""
from __future__ import annotations

from .async_client import generate_patches_parallel
from .deepseek import call_model as call_deepseek
from .ensemble import call_ensemble_sync
from .gemini import call_model as call_gemini

# v0.2.0: Async LLM Pool for parallel operations
try:
    from .async_pool import AsyncLLMPool, LLMRequest, LLMResponse, call_llm_batch

    __all__ = [
        "call_deepseek",
        "call_gemini",
        "call_ensemble_sync",
        "generate_patches_parallel",
        "AsyncLLMPool",
        "LLMRequest",
        "LLMResponse",
        "call_llm_batch",
    ]
except ImportError:
    # async_pool requires httpx[http2]
    __all__ = [
        "call_deepseek",
        "call_gemini",
        "call_ensemble_sync",
        "generate_patches_parallel",
    ]
