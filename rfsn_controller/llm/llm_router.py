from __future__ import annotations
from rfsn_controller.no_llm_guard import assert_llm_allowed
"""Canonical LLM routing path for RFSN Controller.

This module provides a single entry point for all LLM calls,
ensuring consistent provider selection, caching, and fallback logic.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from .ensemble import call_ensemble_sync, get_available_models
from .deepseek import call_model as call_deepseek
from .gemini import call_model as call_gemini

logger = logging.getLogger(__name__)

class LLMRouter:
    """
    Routes LLM requests to the appropriate provider.
    
    Supports:
    - Single model calls (DeepSeek, Gemini, OpenAI, Anthropic)
    - Ensemble calls (parallel execution + scoring)
    - Automatic fallback
    """
    
    def __init__(self, default_provider: str = "deepseek"):
        self.default_provider = os.environ.get("RFSN_DEFAULT_LLM_PROVIDER", default_provider)
        
    def call(
        self, 
        prompt: str, 
        temperature: float = 0.0, 
        provider: Optional[str] = None,
        use_ensemble: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Route a call to an LLM provider.
        
        Args:
            prompt: The input prompt.
            temperature: Sampling temperature.
            provider: Explicit provider override.
            use_ensemble: Whether to use multi-model ensemble.
            **kwargs: Additional provider-specific arguments.
            
        Returns:
            Parsed JSON response from the LLM.
        """
        target_provider = provider or self.default_provider
        
        if use_ensemble:
            logger.info("Routing to ensemble")
            return call_ensemble_sync(prompt, temperature=temperature)
            
        if target_provider == "deepseek":
            logger.debug("Routing to DeepSeek")
            return call_deepseek(prompt, temperature=temperature)
            
        if target_provider == "gemini":
            logger.debug("Routing to Gemini")
            return call_gemini(prompt, temperature=temperature)
            
        # Fallback to ensemble if provider is unknown but ensemble is possible
        if get_available_models():
            logger.warning(f"Unknown provider '{target_provider}', falling back to ensemble")
            return call_ensemble_sync(prompt, temperature=temperature)
            
        raise ValueError(f"No available LLM provider for: {target_provider}")

# Singleton instance
_router = LLMRouter()

def get_llm_router() -> LLMRouter:
    """Get the global LLM router instance."""
    return _router

def call_llm(prompt: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to call the default LLM."""
    return _router.call(prompt, **kwargs)
