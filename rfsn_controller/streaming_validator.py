"""Streaming response processor with early validation.

Allows validation of LLM responses as they stream in,
enabling early termination of invalid responses.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamingValidator:
    """Validate LLM responses as they stream in.
    
    Enables early termination if response is clearly invalid,
    saving tokens and time.
    """
    
    # Patterns that indicate a valid patch response
    valid_patterns: list[str] = field(default_factory=lambda: [
        r'"mode"\s*:\s*"patch"',
        r'"mode"\s*:\s*"skip"',
        r'"diff"\s*:',
        r'```diff',
    ])
    
    # Patterns that indicate an invalid/error response
    invalid_patterns: list[str] = field(default_factory=lambda: [
        r'I cannot',
        r"I'm sorry",
        r'As an AI',
        r'I apologize',
        r'"mode"\s*:\s*"error"',
    ])
    
    # Minimum bytes before making validation decision
    min_bytes: int = 100
    
    # Maximum bytes to accumulate before forcing accept
    max_bytes: int = 50000
    
    def __post_init__(self):
        self._valid_compiled = [re.compile(p, re.IGNORECASE) for p in self.valid_patterns]
        self._invalid_compiled = [re.compile(p, re.IGNORECASE) for p in self.invalid_patterns]
    
    def validate_partial(self, content: str) -> bool | None:
        """Check if partial content is valid, invalid, or undetermined.
        
        Args:
            content: Partial response content so far.
            
        Returns:
            True if valid, False if invalid, None if undetermined.
        """
        if len(content) < self.min_bytes:
            return None
        
        # Check for invalid patterns first
        for pattern in self._invalid_compiled:
            if pattern.search(content):
                return False
        
        # Check for valid patterns
        for pattern in self._valid_compiled:
            if pattern.search(content):
                return True
        
        # If we have lots of content and no patterns, probably invalid
        if len(content) > 500:
            # Check for JSON structure at least
            if '{' in content and '"mode"' not in content:
                return False
        
        return None  # Undetermined


async def stream_with_early_validation(
    stream: AsyncIterator[str],
    validator: StreamingValidator | None = None,
    on_valid: Callable[[str], None] | None = None,
    on_invalid: Callable[[str], None] | None = None,
) -> str:
    """Process a stream with early validation.
    
    Args:
        stream: Async iterator of response chunks.
        validator: Optional validator instance.
        on_valid: Callback when response validated as valid.
        on_invalid: Callback when response validated as invalid.
        
    Returns:
        Full accumulated response.
        
    Raises:
        ValueError: If response is determined to be invalid.
    """
    validator = validator or StreamingValidator()
    accumulated = ""
    validated = False
    
    async for chunk in stream:
        accumulated += chunk
        
        if not validated:
            result = validator.validate_partial(accumulated)
            if result is True:
                validated = True
                if on_valid:
                    on_valid(accumulated)
            elif result is False:
                if on_invalid:
                    on_invalid(accumulated)
                raise ValueError(f"Invalid response detected early: {accumulated[:200]}...")
        
        # Safety limit
        if len(accumulated) > validator.max_bytes:
            break
    
    return accumulated


async def stream_first_valid(
    streams: list[AsyncIterator[str]],
    validator: StreamingValidator | None = None,
    timeout: float = 30.0,
) -> str:
    """Race multiple streams and return first valid response.
    
    Args:
        streams: List of async iterators to race.
        validator: Optional validator instance.
        timeout: Maximum time to wait.
        
    Returns:
        First valid response content.
    """
    validator = validator or StreamingValidator()
    
    async def process_stream(stream: AsyncIterator[str], idx: int) -> tuple[int, str]:
        """Process a single stream and return its content."""
        accumulated = ""
        async for chunk in stream:
            accumulated += chunk
            
            # Check validity
            result = validator.validate_partial(accumulated)
            if result is True:
                return idx, accumulated
            elif result is False:
                raise ValueError("Invalid response")
            
            if len(accumulated) > validator.max_bytes:
                break
        
        return idx, accumulated
    
    # Create tasks for all streams
    tasks = [
        asyncio.create_task(process_stream(stream, i))
        for i, stream in enumerate(streams)
    ]
    
    try:
        # Return first completed valid response
        done, pending = await asyncio.wait(
            tasks,
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
        
        # Get result from first done task
        for task in done:
            try:
                idx, content = task.result()
                return content
            except Exception:
                continue
        
        raise ValueError("No valid response from any stream")
    except TimeoutError:
        # Cancel all tasks
        for task in tasks:
            task.cancel()
        raise


def extract_json_streaming(content: str) -> dict[str, Any] | None:
    """Extract JSON from potentially incomplete streaming content.
    
    Handles cases where JSON is wrapped in markdown or incomplete.
    
    Args:
        content: Streaming content so far.
        
    Returns:
        Parsed JSON dict or None if not parseable yet.
    """
    # Try direct parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Try extracting from markdown code block
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try finding JSON object
    brace_start = content.find('{')
    if brace_start >= 0:
        # Find matching close brace
        depth = 0
        for i, c in enumerate(content[brace_start:]):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(content[brace_start:brace_start + i + 1])
                    except json.JSONDecodeError:
                        pass
                    break
    
    return None
