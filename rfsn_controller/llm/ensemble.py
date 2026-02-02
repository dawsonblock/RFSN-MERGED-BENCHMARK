from __future__ import annotations
from rfsn_controller.no_llm_guard import assert_llm_allowed
"""Multi-model ensemble for improved patch generation.

This module provides:
1. Parallel calls to multiple LLM providers
2. Response scoring and selection
3. Fallback handling for API failures
"""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

from .async_client import (
    AsyncLLMResponse,
    call_deepseek_cached,
    call_gemini_cached,
)

# ============================================================================
# MODEL REGISTRY
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for a single LLM model."""
    
    name: str
    provider: str  # deepseek, gemini, openai, anthropic
    api_key_env: str  # Environment variable for API key
    base_url: str
    default_temperature: float = 0.0
    max_tokens: int = 4096
    priority: int = 1  # Lower = higher priority
    
    def is_available(self) -> bool:
        """Check if this model's API key is configured."""
        return bool(os.environ.get(self.api_key_env))


# Default model configurations
DEFAULT_MODELS: list[ModelConfig] = [
    ModelConfig(
        name="deepseek-chat",
        provider="deepseek",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        priority=1,
    ),
    ModelConfig(
        name="gemini-2.0-flash",
        provider="gemini",
        api_key_env="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        priority=2,
    ),
    ModelConfig(
        name="gpt-4o-mini",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        priority=3,
    ),
    ModelConfig(
        name="claude-3-haiku-20240307",
        provider="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com/v1",
        priority=4,
    ),
]


def get_available_models() -> list[ModelConfig]:
    """Get list of models that have API keys configured."""
    return [m for m in DEFAULT_MODELS if m.is_available()]


# ============================================================================
# RESPONSE SCORING
# ============================================================================

@dataclass
class ScoredResponse:
    """A response with quality scores."""
    
    response: AsyncLLMResponse
    model: str
    score: float
    scores: dict[str, float] = field(default_factory=dict)
    
    @property
    def patch(self) -> str | None:
        """Extract patch diff if present."""
        data = self.response.to_dict()
        if data.get("mode") == "patch":
            return data.get("diff")
        return None


def score_response(response: AsyncLLMResponse) -> tuple[float, dict[str, float]]:
    """Score an LLM response for quality.
    
    Scoring criteria:
    - Valid JSON structure
    - Contains required fields
    - Patch hygiene (if patch mode)
    - Explanation quality
    
    Returns:
        (total_score, component_scores) tuple.
    """
    scores: dict[str, float] = {}
    
    try:
        data = response.to_dict()
    except Exception:
        return 0.0, {"parse_error": 0.0}
    
    # Check for valid mode
    mode = data.get("mode")
    if mode in ("patch", "tool_request", "feature_summary"):
        scores["valid_mode"] = 1.0
    else:
        scores["valid_mode"] = 0.0
        return sum(scores.values()), scores
    
    # Check for explanation
    why = data.get("why", "")
    if len(why) > 50:
        scores["has_explanation"] = 1.0
    elif len(why) > 10:
        scores["has_explanation"] = 0.5
    else:
        scores["has_explanation"] = 0.0
    
    # Mode-specific scoring
    if mode == "patch":
        diff = data.get("diff", "")
        
        # Has actual diff content
        if diff and len(diff) > 10:
            scores["has_diff"] = 1.0
        else:
            scores["has_diff"] = 0.0
        
        # Diff looks valid (has +/- lines)
        if "@@" in diff and ("+" in diff or "-" in diff):
            scores["valid_diff_format"] = 1.0
        else:
            scores["valid_diff_format"] = 0.0
        
        # Not too large (prefer minimal patches)
        line_count = diff.count("\n")
        if line_count <= 20:
            scores["size_bonus"] = 1.0
        elif line_count <= 50:
            scores["size_bonus"] = 0.5
        else:
            scores["size_bonus"] = 0.0
    
    elif mode == "tool_request":
        requests = data.get("requests", [])
        
        # Has tool requests
        if requests and len(requests) > 0:
            scores["has_requests"] = 1.0
        else:
            scores["has_requests"] = 0.0
        
        # Requests are well-formed
        valid_requests = sum(
            1 for r in requests
            if isinstance(r, dict) and "tool" in r
        )
        if requests:
            scores["valid_requests"] = valid_requests / len(requests)
        else:
            scores["valid_requests"] = 0.0
    
    # Latency bonus (faster is better)
    if response.latency_ms > 0:
        if response.latency_ms < 2000:
            scores["latency_bonus"] = 1.0
        elif response.latency_ms < 5000:
            scores["latency_bonus"] = 0.5
        else:
            scores["latency_bonus"] = 0.0
    
    total = sum(scores.values())
    return total, scores


# ============================================================================
# ENSEMBLE CALLER
# ============================================================================

@dataclass
class EnsembleResult:
    """Result from ensemble model call."""
    
    best: ScoredResponse
    all_responses: list[ScoredResponse]
    failed_models: list[str]
    
    @property
    def consensus(self) -> bool:
        """Check if multiple models agreed on the approach."""
        if len(self.all_responses) < 2:
            return False
        
        modes = [r.response.to_dict().get("mode") for r in self.all_responses]
        return len(set(modes)) == 1


async def call_model_by_provider(
    model: ModelConfig,
    prompt: str,
    temperature: float,
    system_prompt: str | None = None,
) -> AsyncLLMResponse:
    """Call a model based on its provider.
    
    Currently supports DeepSeek, with stubs for other providers.
    """
    if model.provider == "deepseek":
        return await call_deepseek_cached(
            prompt,
            temperature=temperature,
            model=model.name,
            system_prompt=system_prompt,
            use_cache=True,  # Enable caching for speed
        )
    
    elif model.provider == "gemini":
        # Use async cached Gemini call
        return await call_gemini_cached(
            prompt,
            temperature=temperature,
            model=model.name,
            system_prompt=system_prompt,
            use_cache=True,  # Enable caching for speed
        )
    
    else:
        # Stub for other providers
        return AsyncLLMResponse(
            content='{"mode": "error", "error": "Provider not implemented"}',
            model=model.name,
            temperature=temperature,
        )


async def call_ensemble(
    prompt: str,
    *,
    models: list[ModelConfig] | None = None,
    temperature: float = 0.0,
    system_prompt: str | None = None,
    max_models: int = 3,
    timeout: float = 30.0,
) -> EnsembleResult:
    """Call multiple models in parallel and select the best response.
    
    Args:
        prompt: The prompt to send to all models.
        models: List of models to use. Uses available models if None.
        temperature: Base temperature for all models.
        system_prompt: Optional system prompt.
        max_models: Maximum number of models to query.
        timeout: Timeout for each model call.
        
    Returns:
        EnsembleResult with best response and all responses.
    """
    if models is None:
        models = get_available_models()[:max_models]
    
    if not models:
        # No models available, return mock response
        mock_response = AsyncLLMResponse(
            content='{"mode": "tool_request", "requests": [], "why": "No models available"}',
            model="mock",
            temperature=temperature,
        )
        scored = ScoredResponse(
            response=mock_response,
            model="mock",
            score=0.0,
        )
        return EnsembleResult(
            best=scored,
            all_responses=[scored],
            failed_models=[],
        )
    
    # Call all models in parallel
    tasks = [
        asyncio.wait_for(
            call_model_by_provider(model, prompt, temperature, system_prompt),
            timeout=timeout,
        )
        for model in models
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Score responses
    scored_responses: list[ScoredResponse] = []
    failed_models: list[str] = []
    
    for model, result in zip(models, results, strict=False):
        if isinstance(result, Exception):
            failed_models.append(f"{model.name}: {result!s}")
            continue
        
        score, score_breakdown = score_response(result)
        scored_responses.append(ScoredResponse(
            response=result,
            model=model.name,
            score=score,
            scores=score_breakdown,
        ))
    
    if not scored_responses:
        # All models failed
        mock_response = AsyncLLMResponse(
            content='{"mode": "error", "error": "All models failed"}',
            model="error",
            temperature=temperature,
        )
        return EnsembleResult(
            best=ScoredResponse(response=mock_response, model="error", score=0.0),
            all_responses=[],
            failed_models=failed_models,
        )
    
    # Sort by score (highest first)
    scored_responses.sort(key=lambda x: x.score, reverse=True)
    
    return EnsembleResult(
        best=scored_responses[0],
        all_responses=scored_responses,
        failed_models=failed_models,
    )


# ============================================================================
# SYNC WRAPPER
# ============================================================================

def call_ensemble_sync(
    prompt: str,
    temperature: float = 0.0,
    max_models: int = 3,
) -> dict[str, Any]:
    """Synchronous wrapper for ensemble call.
    
    Returns the best response as a dict.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                asyncio.run,
                call_ensemble(prompt, temperature=temperature, max_models=max_models),
            )
            result = future.result()
    else:
        result = loop.run_until_complete(
            call_ensemble(prompt, temperature=temperature, max_models=max_models)
        )
    
    return result.best.response.to_dict()
