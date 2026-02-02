from __future__ import annotations
"""Async LLM client pool for parallel operations.

This module provides connection pooling and batching for async LLM calls,
significantly improving throughput for parallel patch generation.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

try:
    import httpx
except ImportError:
    httpx = None

logger = logging.getLogger(__name__)


@dataclass
class LLMRequest:
    """Single LLM API request."""

    provider: str
    model: str
    messages: list[dict[str, str]]
    temperature: float = 0.0
    max_tokens: int = 4096
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Single LLM API response."""

    content: str
    provider: str
    model: str
    tokens_used: int
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class AsyncLLMPool:
    """Connection pool for async LLM calls.

    Provides:
    - HTTP connection pooling
    - Concurrent request batching
    - Rate limiting
    - Automatic retries

    Example:
        pool = AsyncLLMPool(max_connections=100)
        requests = [LLMRequest(...) for _ in range(10)]
        responses = await pool.call_batch(requests)
    """

    def __init__(
        self,
        max_connections: int = 100,
        max_keepalive: int = 20,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        """Initialize async LLM pool.

        Args:
            max_connections: Maximum concurrent connections
            max_keepalive: Maximum keepalive connections
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts per request
        """
        self.max_retries = max_retries

        if httpx is None:
            logger.warning("httpx not installed, async pool will not work. Install with: pip install httpx")
            self.client = None
        else:
            # Try HTTP/2 if h2 is installed, fallback to HTTP/1.1
            try:
                self.client = httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_connections=max_connections,
                        max_keepalive_connections=max_keepalive,
                    ),
                    timeout=httpx.Timeout(timeout),
                    http2=True,  # Enable HTTP/2 for multiplexing
                )
            except ImportError:
                # h2 not installed, use HTTP/1.1
                self.client = httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_connections=max_connections,
                        max_keepalive_connections=max_keepalive,
                    ),
                    timeout=httpx.Timeout(timeout),
                    http2=False,
                )

        # Rate limiting (simple token bucket)
        self._rate_limit_semaphore = asyncio.Semaphore(max_connections)

    async def call_batch(
        self,
        requests: list[LLMRequest],
        max_concurrent: int | None = None,
    ) -> list[LLMResponse]:
        """Execute multiple LLM calls concurrently.

        Args:
            requests: List of LLM requests to execute
            max_concurrent: Optional limit on concurrent requests

        Returns:
            List of responses (in same order as requests)
        """
        if not self.client:
            logger.error("AsyncLLMPool not initialized (httpx not available)")
            return [
                LLMResponse(
                    content="",
                    provider=req.provider,
                    model=req.model,
                    tokens_used=0,
                    latency_ms=0,
                    error="httpx not installed",
                )
                for req in requests
            ]

        semaphore = asyncio.Semaphore(max_concurrent or len(requests))

        async def call_with_semaphore(req: LLMRequest) -> LLMResponse:
            async with semaphore:
                return await self._call_single(req)

        tasks = [call_with_semaphore(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=False)

    async def _call_single(self, request: LLMRequest) -> LLMResponse:
        """Execute a single LLM API call with retries.

        Args:
            request: LLM request to execute

        Returns:
            LLM response
        """
        import time

        for attempt in range(self.max_retries):
            try:
                async with self._rate_limit_semaphore:
                    start_time = time.time()

                    if request.provider == "deepseek":
                        response = await self._call_deepseek(request)
                    elif request.provider == "gemini":
                        response = await self._call_gemini(request)
                    elif request.provider == "anthropic":
                        response = await self._call_anthropic(request)
                    else:
                        raise ValueError(f"Unknown provider: {request.provider}")

                    latency_ms = (time.time() - start_time) * 1000
                    response.latency_ms = latency_ms
                    return response

            except Exception as e:
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}",
                    exc_info=attempt == self.max_retries - 1,
                )
                if attempt == self.max_retries - 1:
                    return LLMResponse(
                        content="",
                        provider=request.provider,
                        model=request.model,
                        tokens_used=0,
                        latency_ms=0,
                        error=str(e),
                    )
                await asyncio.sleep(2**attempt)  # Exponential backoff

        # Should never reach here
        return LLMResponse(
            content="",
            provider=request.provider,
            model=request.model,
            tokens_used=0,
            latency_ms=0,
            error="Max retries exceeded",
        )

    async def _call_deepseek(self, request: LLMRequest) -> LLMResponse:
        """Call DeepSeek API."""
        import os

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not set")

        response = await self.client.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": request.model,
                "messages": request.messages,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            },
        )
        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            provider="deepseek",
            model=request.model,
            tokens_used=data["usage"]["total_tokens"],
            latency_ms=0,  # Set by caller
            metadata={"finish_reason": data["choices"][0]["finish_reason"]},
        )

    async def _call_gemini(self, request: LLMRequest) -> LLMResponse:
        """Call Google Gemini API."""
        # Placeholder - implement when needed
        raise NotImplementedError("Gemini async client not yet implemented")

    async def _call_anthropic(self, request: LLMRequest) -> LLMResponse:
        """Call Anthropic Claude API."""
        # Placeholder - implement when needed
        raise NotImplementedError("Anthropic async client not yet implemented")

    async def close(self):
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()


# Convenience function
async def call_llm_batch(requests: list[LLMRequest]) -> list[LLMResponse]:
    """Convenience function to call multiple LLMs in parallel.

    Args:
        requests: List of LLM requests

    Returns:
        List of responses

    Example:
        requests = [
            LLMRequest(provider="deepseek", model="deepseek-chat", messages=[...]),
            LLMRequest(provider="deepseek", model="deepseek-chat", messages=[...]),
        ]
        responses = await call_llm_batch(requests)
    """
    async with AsyncLLMPool() as pool:
        return await pool.call_batch(requests)
