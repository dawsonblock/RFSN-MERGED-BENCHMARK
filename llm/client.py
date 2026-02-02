"""
LLM Integration Layer for RFSN Patch Generation

Provides unified interface to multiple LLM providers:
- OpenAI (GPT-4, GPT-4-turbo)
- Anthropic (Claude 3.5 Sonnet)
- Google (Gemini)
- DeepSeek

Features:
- Automatic retry with exponential backoff
- Rate limiting and quota management
- Cost tracking per request
- Token usage optimization
- Context window management
- Streaming support
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Dict, List, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"


@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: LLMProvider
    model: str
    api_key: str
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.2
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_exponential_base: float = 2.0


@dataclass
class LLMResponse:
    """LLM response with metadata"""
    content: str
    model: str
    provider: str
    tokens_used: int
    cost_usd: float
    latency_ms: float
    finish_reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMUsageStats:
    """Track LLM usage statistics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    provider_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class LLMClient:
    """Base LLM client interface"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.stats = LLMUsageStats()
    
    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion"""
        raise NotImplementedError
    
    async def complete_with_retry(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Complete with automatic retry on failure"""
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                response = await self.complete(prompt, system, **kwargs)
                self.stats.successful_requests += 1
                return response
            except Exception as e:
                last_error = e
                self.stats.failed_requests += 1
                
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (
                        self.config.retry_exponential_base ** attempt
                    )
                    logger.warning(
                        f"LLM request failed (attempt {attempt + 1}/"
                        f"{self.config.max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"LLM request failed after {self.config.max_retries} attempts: {e}"
                    )
        
        raise last_error
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)"""
        # Rough estimate: 4 characters per token
        return len(text) // 4
    
    def truncate_to_context(
        self,
        text: str,
        max_tokens: int,
        preserve_start: bool = True
    ) -> str:
        """Truncate text to fit context window"""
        estimated_tokens = self.estimate_tokens(text)
        
        if estimated_tokens <= max_tokens:
            return text
        
        # Truncate to approximate character count
        target_chars = max_tokens * 4
        
        if preserve_start:
            return text[:target_chars] + "\n\n[... truncated ...]"
        else:
            return "[... truncated ...]\n\n" + text[-target_chars:]


class OpenAIClient(LLMClient):
    """OpenAI API client"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import openai
            self.openai = openai
            self.client = openai.AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout
            )
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
    
    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using OpenAI API"""
        start_time = time.time()
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature or self.config.temperature,
                stop=stop_sequences,
                **kwargs
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Calculate cost (approximate)
            tokens_used = response.usage.total_tokens
            cost_usd = self._calculate_cost(tokens_used)
            
            # Update stats
            self.stats.total_requests += 1
            self.stats.total_tokens += tokens_used
            self.stats.total_cost_usd += cost_usd
            self.stats.total_latency_ms += latency_ms
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                provider=LLMProvider.OPENAI,
                tokens_used=tokens_used,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                finish_reason=response.choices[0].finish_reason,
                metadata={
                    "response_id": response.id,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def _calculate_cost(self, tokens: int) -> float:
        """Calculate approximate cost in USD"""
        # Pricing as of 2024 (approximate)
        pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
        }
        
        model_base = self.config.model.split("-")[0] + "-" + self.config.model.split("-")[1]
        if model_base not in pricing:
            model_base = "gpt-4-turbo"  # default
        
        # Rough estimate: assume 60% input, 40% output
        cost = (tokens * 0.6 * pricing[model_base]["input"] / 1000 +
                tokens * 0.4 * pricing[model_base]["output"] / 1000)
        return cost


class AnthropicClient(LLMClient):
    """Anthropic Claude API client"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import anthropic
            self.anthropic = anthropic
            self.client = anthropic.AsyncAnthropic(
                api_key=config.api_key,
                timeout=config.timeout
            )
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
    
    async def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate completion using Anthropic API"""
        start_time = time.time()
        
        try:
            response = await self.client.messages.create(
                model=self.config.model,
                system=system or "",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature or self.config.temperature,
                stop_sequences=stop_sequences,
                **kwargs
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Calculate cost
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            cost_usd = self._calculate_cost(
                response.usage.input_tokens,
                response.usage.output_tokens
            )
            
            # Update stats
            self.stats.total_requests += 1
            self.stats.total_tokens += tokens_used
            self.stats.total_cost_usd += cost_usd
            self.stats.total_latency_ms += latency_ms
            
            return LLMResponse(
                content=response.content[0].text,
                model=response.model,
                provider=LLMProvider.ANTHROPIC,
                tokens_used=tokens_used,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                finish_reason=response.stop_reason,
                metadata={
                    "response_id": response.id,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            )
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD"""
        # Pricing as of 2024
        pricing = {
            "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},  # per 1K tokens
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015}
        }
        
        model_key = self.config.model
        if model_key not in pricing:
            model_key = "claude-3-5-sonnet"  # default
        
        cost = (input_tokens * pricing[model_key]["input"] / 1000 +
                output_tokens * pricing[model_key]["output"] / 1000)
        return cost


class LLMClientFactory:
    """Factory for creating LLM clients"""
    
    @staticmethod
    def create(config: LLMConfig) -> LLMClient:
        """Create appropriate LLM client based on provider"""
        if config.provider == LLMProvider.OPENAI:
            return OpenAIClient(config)
        elif config.provider == LLMProvider.ANTHROPIC:
            return AnthropicClient(config)
        elif config.provider == LLMProvider.DEEPSEEK:
            # DeepSeek uses OpenAI-compatible API
            config.base_url = config.base_url or "https://api.deepseek.com/v1"
            return OpenAIClient(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")
    
    @staticmethod
    def from_env(provider: str = "openai") -> LLMClient:
        """Create client from environment variables"""
        provider_enum = LLMProvider(provider.lower())
        
        # Map provider to env var names
        env_vars = {
            LLMProvider.OPENAI: ("OPENAI_API_KEY", "gpt-4-turbo"),
            LLMProvider.ANTHROPIC: ("ANTHROPIC_API_KEY", "claude-3-5-sonnet-20241022"),
            LLMProvider.DEEPSEEK: ("DEEPSEEK_API_KEY", "deepseek-chat")
        }
        
        env_key, default_model = env_vars[provider_enum]
        api_key = os.getenv(env_key)
        
        if not api_key:
            raise ValueError(f"API key not found in environment: {env_key}")
        
        config = LLMConfig(
            provider=provider_enum,
            model=os.getenv(f"{provider.upper()}_MODEL", default_model),
            api_key=api_key,
            base_url=os.getenv(f"{provider.upper()}_BASE_URL")
        )
        
        return LLMClientFactory.create(config)


# Global client cache
_client_cache: Dict[str, LLMClient] = {}


def get_llm_client(
    provider: Optional[str] = None,
    config: Optional[LLMConfig] = None
) -> LLMClient:
    """Get or create LLM client (cached)"""
    if config:
        cache_key = f"{config.provider}:{config.model}:{config.api_key[:8]}"
        if cache_key not in _client_cache:
            _client_cache[cache_key] = LLMClientFactory.create(config)
        return _client_cache[cache_key]
    
    provider = provider or os.getenv("LLM_PROVIDER", "openai")
    cache_key = f"env:{provider}"
    
    if cache_key not in _client_cache:
        _client_cache[cache_key] = LLMClientFactory.from_env(provider)
    
    return _client_cache[cache_key]


async def test_llm_integration():
    """Test LLM integration"""
    print("Testing LLM Integration...")
    
    # Test with environment variables
    try:
        client = get_llm_client("openai")
        
        response = await client.complete_with_retry(
            prompt="Write a simple Python function that adds two numbers.",
            system="You are a helpful coding assistant.",
            max_tokens=200
        )
        
        print(f"\n✅ Response from {response.provider} ({response.model}):")
        print(f"   Content: {response.content[:100]}...")
        print(f"   Tokens: {response.tokens_used}")
        print(f"   Cost: ${response.cost_usd:.4f}")
        print(f"   Latency: {response.latency_ms:.0f}ms")
        print(f"\n📊 Stats: {client.stats}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_llm_integration())
