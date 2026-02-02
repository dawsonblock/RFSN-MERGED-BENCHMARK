"""LLM DeepSeek module - Top-level wrapper for backward compatibility.

This module provides a top-level import path for the DeepSeek LLM functionality.
"""

from __future__ import annotations

from typing import Any

# Module-level state for lazy loading
_sdk_state: dict[str, Any] = {"openai": None}


def _ensure_openai_imported() -> None:
    """Lazily import the OpenAI SDK."""
    if _sdk_state["openai"] is None:
        try:
            import openai
            _sdk_state["openai"] = openai
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI SDK not available. Install with: pip install openai"
            ) from exc


def call_model(prompt: str, **kwargs) -> str:
    """Call the DeepSeek model with the given prompt.
    
    Args:
        prompt: The prompt to send to the model.
        **kwargs: Additional arguments for the API call.
        
    Returns:
        The model's response text.
        
    Raises:
        RuntimeError: If the OpenAI SDK is not installed.
    """
    _ensure_openai_imported()
    
    # Import the actual implementation
    from rfsn_controller.llm.deepseek import call_deepseek
    return call_deepseek(prompt, **kwargs)


def get_client():
    """Get the DeepSeek client (uses OpenAI SDK with DeepSeek base URL).
    
    Returns:
        OpenAI client configured for DeepSeek.
        
    Raises:
        RuntimeError: If the OpenAI SDK is not installed.
    """
    _ensure_openai_imported()
    
    from rfsn_controller.llm.deepseek import get_deepseek_client
    return get_deepseek_client()
