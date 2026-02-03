"""LLM Prompting for SWE-bench upstream learner.

Handles LLM API calls with retry logic and strict JSON parsing.
Integrates with existing rfsn_controller/llm/ clients.

INVARIANTS:
1. All LLM calls have timeouts and retry limits
2. JSON responses are strictly validated
3. Supports both DeepSeek and Gemini backends
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LLMProvider(Enum):
    """Supported LLM providers."""
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    AUTO = "auto"  # Auto-select based on available API keys


@dataclass
class LLMConfig:
    """Configuration for LLM calls.
    
    Attributes:
        provider: Which LLM provider to use
        model: Model name (provider-specific)
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries in seconds
    """
    provider: LLMProvider = LLMProvider.AUTO
    model: str = ""  # Empty = use default for provider
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 2.0


@dataclass
class LLMResponse:
    """Structured LLM response.
    
    Attributes:
        content: Raw response text
        parsed: Parsed JSON (if applicable)
        success: Whether the call succeeded
        error: Error message (if failed)
        latency_ms: Response latency in milliseconds
        tokens_used: Approximate token count
    """
    content: str = ""
    parsed: dict[str, Any] | None = None
    success: bool = True
    error: str | None = None
    latency_ms: float = 0.0
    tokens_used: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


def _get_default_provider() -> LLMProvider:
    """Determine the default LLM provider based on available API keys."""
    if os.environ.get("DEEPSEEK_API_KEY"):
        return LLMProvider.DEEPSEEK
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return LLMProvider.GEMINI
    return LLMProvider.DEEPSEEK  # Default fallback


def _get_default_model(provider: LLMProvider) -> str:
    """Get the default model for a provider."""
    if provider == LLMProvider.DEEPSEEK:
        return "deepseek-chat"
    if provider == LLMProvider.GEMINI:
        return "gemini-2.0-flash"
    return "deepseek-chat"


def call_llm(
    prompt: str,
    system_prompt: str = "",
    config: LLMConfig | None = None,
) -> LLMResponse:
    """Call an LLM with retry logic.
    
    Args:
        prompt: The user prompt
        system_prompt: Optional system prompt
        config: LLM configuration
        
    Returns:
        LLMResponse with content and metadata
    """
    if config is None:
        config = LLMConfig()
    
    # Resolve provider
    provider = config.provider
    if provider == LLMProvider.AUTO:
        provider = _get_default_provider()
    
    # Resolve model
    model = config.model or _get_default_model(provider)
    
    start_time = time.time()
    last_error = None
    
    for attempt in range(config.max_retries):
        try:
            if provider == LLMProvider.DEEPSEEK:
                content = _call_deepseek(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
            elif provider == LLMProvider.GEMINI:
                content = _call_gemini(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
            else:
                raise ValueError(f"Unknown provider: {provider}")
            
            latency_ms = (time.time() - start_time) * 1000
            return LLMResponse(
                content=content,
                success=True,
                latency_ms=latency_ms,
                metadata={"provider": provider.value, "model": model, "attempt": attempt + 1},
            )
            
        except Exception as e:
            last_error = str(e)
            if attempt < config.max_retries - 1:
                time.sleep(config.retry_delay * (attempt + 1))  # Exponential backoff
    
    latency_ms = (time.time() - start_time) * 1000
    return LLMResponse(
        content="",
        success=False,
        error=last_error,
        latency_ms=latency_ms,
        metadata={"provider": provider.value, "model": model, "attempts": config.max_retries},
    )


def _call_deepseek(
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call DeepSeek API using direct HTTP methods."""
    return _call_deepseek_direct(prompt, system_prompt, model, temperature, max_tokens)


def _call_deepseek_direct(
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Direct DeepSeek API call without using the rfsn_controller client."""
    import httpx
    
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    response = httpx.post(
        "https://api.deepseek.com/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=120.0,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _call_gemini(
    prompt: str,
    system_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Call Gemini API using existing client."""
    try:
        from rfsn_controller.llm import call_gemini
        
        # Combine system and user prompt for Gemini
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        response = call_gemini(
            prompt=full_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.get("content", "") if isinstance(response, dict) else str(response)
    except ImportError:
        raise RuntimeError("rfsn_controller.llm.call_gemini not available")


def parse_json_response(response: str) -> dict[str, Any] | None:
    """Extract and parse JSON from an LLM response.
    
    Handles common LLM response patterns:
    - Raw JSON
    - JSON in markdown code blocks
    - JSON with surrounding text
    
    Args:
        response: Raw LLM response text
        
    Returns:
        Parsed JSON dict, or None if parsing fails
    """
    if not response:
        return None
    
    # Try direct parse first
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass
    
    # Try extracting from markdown code blocks
    patterns = [
        r"```json\s*([\s\S]*?)\s*```",  # ```json ... ```
        r"```\s*([\s\S]*?)\s*```",       # ``` ... ```
        r"\{[\s\S]*\}",                  # Raw JSON object
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            try:
                json_str = match.group(1) if match.lastindex else match.group(0)
                return json.loads(json_str.strip())
            except (json.JSONDecodeError, IndexError):
                continue
    
    return None


def extract_diff_from_response(response: str) -> str | None:
    """Extract a diff/patch from an LLM response.
    
    Handles common patterns:
    - Raw unified diff
    - Diff in code blocks
    - JSON with 'diff' or 'patch' field
    
    Args:
        response: Raw LLM response text
        
    Returns:
        Unified diff string, or None if not found
    """
    if not response:
        return None
    
    # Try parsing as JSON first
    parsed = parse_json_response(response)
    if parsed:
        for key in ["diff", "patch", "unified_diff"]:
            if key in parsed and isinstance(parsed[key], str):
                return _normalize_diff(parsed[key])
    
    # Look for diff patterns in text
    diff_patterns = [
        r"```diff\s*([\s\S]*?)\s*```",
        r"```patch\s*([\s\S]*?)\s*```",
        r"(---\s+a/.*?\n\+\+\+\s+b/.*?\n@@[\s\S]*?)(?=\n```|$)",
    ]
    
    for pattern in diff_patterns:
        match = re.search(pattern, response)
        if match:
            diff = match.group(1).strip()
            if diff.startswith("---") or diff.startswith("diff --git"):
                return _normalize_diff(diff)
    
    # Check if the whole response looks like a diff
    if response.strip().startswith("---") or response.strip().startswith("diff --git"):
        return _normalize_diff(response.strip())
    
    return None


def _normalize_diff(diff: str) -> str:
    """Normalize a diff to ensure proper unified diff format.
    
    LLMs often output context lines without leading spaces, which
    breaks git apply. This function fixes that.
    
    Args:
        diff: Raw diff string
        
    Returns:
        Normalized diff string
    """
    lines = diff.split('\n')
    result = []
    in_hunk = False
    
    for line in lines:
        # Detect hunk start
        if line.startswith('@@'):
            in_hunk = True
            result.append(line)
            continue
        
        # Headers pass through unchanged
        if line.startswith('---') or line.startswith('+++') or line.startswith('diff '):
            result.append(line)
            continue
        
        # Inside a hunk, lines must start with ' ', '+', '-', or '\'
        if in_hunk:
            if not line:
                # Empty line in diff = context line (should be ' ')
                result.append(' ')
            elif line[0] in (' ', '+', '-', '\\'):
                # Already properly formatted
                result.append(line)
            else:
                # Missing leading space - add it (context line)
                result.append(' ' + line)
        else:
            result.append(line)
    
    return '\n'.join(result)


def build_swebench_prompt(
    variant_template: str,
    problem_statement: str,
    relevant_files: list[dict[str, str]],
    similar_memories: list[dict[str, Any]] | None = None,
) -> str:
    """Build a prompt for SWE-bench task using a variant template.
    
    Args:
        variant_template: Prompt template with placeholders
        problem_statement: The GitHub issue/problem description
        relevant_files: List of {path, content} dicts for context
        similar_memories: Optional list of similar past attempts
        
    Returns:
        Formatted prompt string
    """
    # Build files context
    files_context = ""
    for f in relevant_files:
        path = f.get("path", "unknown")
        content = f.get("content", "")
        files_context += f"\n[{path}]\n{content}\n"
    
    # Build memories context
    memories_context = ""
    if similar_memories:
        memories_context = "\n## Similar Past Attempts:\n"
        for mem in similar_memories[:3]:  # Limit to 3 memories
            outcome = mem.get("outcome", "unknown")
            summary = mem.get("summary", "")
            memories_context += f"- [{outcome}] {summary}\n"
    
    # Apply template substitutions
    prompt = variant_template
    prompt = prompt.replace("{problem_statement}", problem_statement)
    prompt = prompt.replace("{files}", files_context)
    prompt = prompt.replace("{memories}", memories_context)
    prompt = prompt.replace("{relevant_files}", files_context)  # Alias
    
    return prompt
