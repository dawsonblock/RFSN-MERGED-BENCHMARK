"""LLM API Adapter - Real LLM API integration for CGW coding agent.

This module provides real API integrations for:
- DeepSeek (primary)
- OpenAI (fallback)
- Google Gemini (fallback)

The adapters follow the llm_caller signature expected by LLMPatchGenerator
and other LLM consumers: (prompt, model, temperature, max_tokens) -> str
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class LLMAdapterConfig:
    """Configuration for LLM adapters."""
    
    # DeepSeek
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    
    # OpenAI
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"
    
    # Gemini
    gemini_api_key: Optional[str] = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    
    # Common settings
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    
    @classmethod
    def from_env(cls) -> "LLMAdapterConfig":
        """Load configuration from environment variables."""
        return cls(
            deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            gemini_api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
        )


class DeepSeekAdapter:
    """Adapter for DeepSeek API.
    
    DeepSeek is the primary LLM for code generation due to its
    strong performance on coding tasks and cost-effectiveness.
    """
    
    def __init__(self, config: Optional[LLMAdapterConfig] = None):
        self.config = config or LLMAdapterConfig.from_env()
        self._client: Optional[httpx.Client] = None
        
        if not self.config.deepseek_api_key:
            logger.warning("DeepSeek API key not configured")
    
    @property
    def client(self) -> httpx.Client:
        """Lazy-initialize HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.config.deepseek_base_url,
                timeout=self.config.timeout,
                headers={
                    "Authorization": f"Bearer {self.config.deepseek_api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client
    
    def call(
        self,
        prompt: str,
        model: str = "deepseek-chat",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        """Call DeepSeek API.
        
        Args:
            prompt: The prompt text.
            model: Model name (default: deepseek-chat).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            The generated text response.
            
        Raises:
            RuntimeError: If API call fails after retries.
        """
        if not self.config.deepseek_api_key:
            raise RuntimeError("DeepSeek API key not configured")
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.post("/chat/completions", json=payload)
                response.raise_for_status()
                
                data = response.json()
                return data["choices"][0]["message"]["content"]
                
            except httpx.HTTPStatusError as e:
                last_error = e
                logger.warning(f"DeepSeek API error (attempt {attempt + 1}): {e}")
                if e.response.status_code in (429, 500, 502, 503, 504):
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise
            except httpx.RequestError as e:
                last_error = e
                logger.warning(f"DeepSeek request error (attempt {attempt + 1}): {e}")
                time.sleep(self.config.retry_delay * (attempt + 1))
        
        raise RuntimeError(f"DeepSeek API failed after {self.config.max_retries} retries: {last_error}")
    
    def is_available(self) -> bool:
        """Check if adapter is configured and available."""
        return bool(self.config.deepseek_api_key)


class OpenAIAdapter:
    """Adapter for OpenAI API (fallback).
    
    Used as a fallback when DeepSeek is unavailable.
    """
    
    def __init__(self, config: Optional[LLMAdapterConfig] = None):
        self.config = config or LLMAdapterConfig.from_env()
        self._client: Optional[httpx.Client] = None
    
    @property
    def client(self) -> httpx.Client:
        """Lazy-initialize HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.config.openai_base_url,
                timeout=self.config.timeout,
                headers={
                    "Authorization": f"Bearer {self.config.openai_api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client
    
    def call(
        self,
        prompt: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        """Call OpenAI API."""
        if not self.config.openai_api_key:
            raise RuntimeError("OpenAI API key not configured")
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.post("/chat/completions", json=payload)
                response.raise_for_status()
                
                data = response.json()
                return data["choices"][0]["message"]["content"]
                
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                last_error = e
                logger.warning(f"OpenAI API error (attempt {attempt + 1}): {e}")
                time.sleep(self.config.retry_delay * (attempt + 1))
        
        raise RuntimeError(f"OpenAI API failed: {last_error}")
    
    def is_available(self) -> bool:
        """Check if adapter is configured."""
        return bool(self.config.openai_api_key)


class GeminiAdapter:
    """Adapter for Google Gemini API (fallback).
    
    Uses the Gemini API for text generation.
    """
    
    def __init__(self, config: Optional[LLMAdapterConfig] = None):
        self.config = config or LLMAdapterConfig.from_env()
        self._client: Optional[httpx.Client] = None
    
    @property
    def client(self) -> httpx.Client:
        """Lazy-initialize HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.config.gemini_base_url,
                timeout=self.config.timeout,
            )
        return self._client
    
    def call(
        self,
        prompt: str,
        model: str = "gemini-1.5-flash",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        """Call Gemini API."""
        if not self.config.gemini_api_key:
            raise RuntimeError("Gemini API key not configured")
        
        # Gemini uses a different API structure
        url = f"/models/{model}:generateContent"
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.post(
                    url,
                    json=payload,
                    params={"key": self.config.gemini_api_key},
                )
                response.raise_for_status()
                
                data = response.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
                
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                last_error = e
                logger.warning(f"Gemini API error (attempt {attempt + 1}): {e}")
                time.sleep(self.config.retry_delay * (attempt + 1))
        
        raise RuntimeError(f"Gemini API failed: {last_error}")
    
    def is_available(self) -> bool:
        """Check if adapter is configured."""
        return bool(self.config.gemini_api_key)


class LLMRouter:
    """Routes LLM calls to the best available provider.
    
    Implements automatic failover between providers:
    1. DeepSeek (primary - best for code)
    2. OpenAI (fallback)
    3. Gemini (fallback)
    
    Usage:
        router = LLMRouter()
        response = router.call(prompt="Fix this bug", model="deepseek-chat")
    """
    
    def __init__(self, config: Optional[LLMAdapterConfig] = None):
        self.config = config or LLMAdapterConfig.from_env()
        self.deepseek = DeepSeekAdapter(self.config)
        self.openai = OpenAIAdapter(self.config)
        self.gemini = GeminiAdapter(self.config)
        
        # Track usage for logging
        self._call_count = 0
        self._provider_usage: Dict[str, int] = {}
    
    def call(
        self,
        prompt: str,
        model: str = "deepseek-chat",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        """Route call to best available provider.
        
        Tries providers in order of preference until one succeeds.
        
        Args:
            prompt: The prompt text.
            model: Model name hint (provider will use compatible model).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            
        Returns:
            The generated text response.
            
        Raises:
            RuntimeError: If all providers fail.
        """
        self._call_count += 1
        errors = []
        
        # Try DeepSeek first (best for code)
        if self.deepseek.is_available():
            try:
                result = self.deepseek.call(prompt, model, temperature, max_tokens, **kwargs)
                self._track_usage("deepseek")
                return result
            except Exception as e:
                errors.append(f"DeepSeek: {e}")
                logger.warning(f"DeepSeek failed, trying fallback: {e}")
        
        # Fallback to OpenAI
        if self.openai.is_available():
            try:
                # Map model names if needed
                openai_model = "gpt-4o-mini" if "deepseek" in model.lower() else model
                result = self.openai.call(prompt, openai_model, temperature, max_tokens, **kwargs)
                self._track_usage("openai")
                return result
            except Exception as e:
                errors.append(f"OpenAI: {e}")
                logger.warning(f"OpenAI failed, trying fallback: {e}")
        
        # Fallback to Gemini
        if self.gemini.is_available():
            try:
                result = self.gemini.call(prompt, "gemini-1.5-flash", temperature, max_tokens, **kwargs)
                self._track_usage("gemini")
                return result
            except Exception as e:
                errors.append(f"Gemini: {e}")
        
        # All providers failed
        raise RuntimeError(f"All LLM providers failed: {'; '.join(errors)}")
    
    def _track_usage(self, provider: str) -> None:
        """Track provider usage for metrics."""
        self._provider_usage[provider] = self._provider_usage.get(provider, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        return {
            "total_calls": self._call_count,
            "provider_usage": dict(self._provider_usage),
            "available_providers": self.available_providers(),
        }
    
    def available_providers(self) -> List[str]:
        """List available providers."""
        providers = []
        if self.deepseek.is_available():
            providers.append("deepseek")
        if self.openai.is_available():
            providers.append("openai")
        if self.gemini.is_available():
            providers.append("gemini")
        return providers


# Module-level singleton for convenience
_default_router: Optional[LLMRouter] = None


def get_default_router() -> LLMRouter:
    """Get or create the default LLM router."""
    global _default_router
    if _default_router is None:
        _default_router = LLMRouter()
    return _default_router


def create_llm_caller(config: Optional[LLMAdapterConfig] = None) -> Callable[..., str]:
    """Create an llm_caller function for use with LLMPatchGenerator.
    
    This returns a callable with the signature expected by the
    CGW coding agent's LLM integration.
    
    Usage:
        from cgw_ssl_guard.coding_agent.llm_adapter import create_llm_caller
        from cgw_ssl_guard.coding_agent.llm_integration import LLMPatchGenerator
        
        llm_caller = create_llm_caller()
        generator = LLMPatchGenerator(llm_caller=llm_caller)
    """
    router = LLMRouter(config) if config else get_default_router()
    return router.call


def validate_api_keys() -> Dict[str, bool]:
    """Validate which API keys are configured.
    
    Returns:
        Dict mapping provider name to availability.
    """
    config = LLMAdapterConfig.from_env()
    return {
        "deepseek": bool(config.deepseek_api_key),
        "openai": bool(config.openai_api_key),
        "gemini": bool(config.gemini_api_key),
    }
