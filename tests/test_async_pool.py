"""Tests for async LLM pool."""

import pytest

pytest.importorskip("httpx")


@pytest.mark.asyncio
async def test_async_pool_initialization():
    """Test AsyncLLMPool can be initialized."""
    from rfsn_controller.llm.async_pool import AsyncLLMPool

    pool = AsyncLLMPool(max_connections=10)
    assert pool is not None
    assert pool.client is not None
    assert pool.max_retries == 3
    await pool.close()


@pytest.mark.asyncio
async def test_llm_request_dataclass():
    """Test LLMRequest dataclass."""
    from rfsn_controller.llm.async_pool import LLMRequest

    request = LLMRequest(
        provider="deepseek",
        model="deepseek-chat",
        messages=[{"role": "user", "content": "test"}],
        temperature=0.5,
        max_tokens=100,
    )
    assert request.provider == "deepseek"
    assert request.model == "deepseek-chat"
    assert request.temperature == 0.5
    assert request.max_tokens == 100


@pytest.mark.asyncio
async def test_llm_response_dataclass():
    """Test LLMResponse dataclass."""
    from rfsn_controller.llm.async_pool import LLMResponse

    response = LLMResponse(
        content="test response",
        provider="deepseek",
        model="deepseek-chat",
        tokens_used=50,
        latency_ms=100.5,
    )
    assert response.content == "test response"
    assert response.provider == "deepseek"
    assert response.tokens_used == 50
    assert response.latency_ms == 100.5
    assert response.error is None


@pytest.mark.asyncio
async def test_async_pool_context_manager():
    """Test AsyncLLMPool as async context manager."""
    from rfsn_controller.llm.async_pool import AsyncLLMPool

    async with AsyncLLMPool() as pool:
        assert pool.client is not None

    # Client should be closed after exit
    assert pool.client.is_closed


def test_llm_request_defaults():
    """Test LLMRequest default values."""
    from rfsn_controller.llm.async_pool import LLMRequest

    request = LLMRequest(
        provider="deepseek",
        model="deepseek-chat",
        messages=[],
    )
    assert request.temperature == 0.0
    assert request.max_tokens == 4096
    assert request.metadata == {}
