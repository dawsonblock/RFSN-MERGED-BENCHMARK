"""LLM Gemini module - Top-level wrapper for backward compatibility.

This module provides a top-level import path for the Gemini LLM functionality.
"""

from __future__ import annotations

from typing import Any

# Module-level state for lazy loading (avoids global statement)
_sdk_state: dict[str, Any] = {"genai": None, "types": None}


def _ensure_genai_imported() -> None:
    """Lazily import the Google GenAI SDK."""
    if _sdk_state["genai"] is None:
        try:
            from google import genai
            from google.genai import types
            _sdk_state["genai"] = genai
            _sdk_state["types"] = types
        except ImportError as exc:
            raise RuntimeError(
                "Google GenAI SDK not available. Install with: pip install google-genai"
            ) from exc


def call_model(prompt: str, **kwargs) -> str:
    """Call the Gemini model with the given prompt.
    
    Args:
        prompt: The prompt to send to the model.
        **kwargs: Additional arguments for the API call.
        
    Returns:
        The model's response text.
        
    Raises:
        RuntimeError: If the Google GenAI SDK is not installed.
    """
    _ensure_genai_imported()
    
    # Import the actual implementation
    from rfsn_controller.llm.gemini import call_gemini
    return call_gemini(prompt, **kwargs)


def get_client():
    """Get the Gemini client.
    
    Returns:
        Google GenAI client.
        
    Raises:
        RuntimeError: If the Google GenAI SDK is not installed.
    """
    _ensure_genai_imported()
    
    from rfsn_controller.llm.gemini import get_gemini_client
    return get_gemini_client()
