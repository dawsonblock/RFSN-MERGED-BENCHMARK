"""
LLM Integration Package for RFSN

Provides unified interface to multiple LLM providers for patch generation.
"""

from .client import (
    LLMClient,
    LLMConfig,
    LLMProvider,
    LLMResponse,
    LLMUsageStats,
    get_llm_client,
    OpenAIClient,
    AnthropicClient,
    LLMClientFactory
)

from .prompts import (
    PatchPromptTemplates,
    PromptContext,
    build_context_from_localization
)

from .patch_generator import (
    LLMPatchGenerator,
    generate_patches_with_llm
)

__all__ = [
    # Client
    "LLMClient",
    "LLMConfig",
    "LLMProvider",
    "LLMResponse",
    "LLMUsageStats",
    "get_llm_client",
    "OpenAIClient",
    "AnthropicClient",
    "LLMClientFactory",
    # Prompts
    "PatchPromptTemplates",
    "PromptContext",
    "build_context_from_localization",
    # Patch Generator
    "LLMPatchGenerator",
    "generate_patches_with_llm",
]
