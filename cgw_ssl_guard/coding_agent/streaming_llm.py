"""CGW Streaming LLM Execution.

Adds async streaming capabilities to the LLM adapter:
- Partial token emission as events
- Early termination on safety triggers
- Progress tracking
- Token cost tracking
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, Optional

# Note: Using conditional imports to handle optional async deps
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class StreamingMetrics:
    """Metrics from a streaming LLM call."""
    
    start_time: float = 0.0
    first_token_time: float = 0.0
    end_time: float = 0.0
    tokens_generated: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0
    
    @property
    def time_to_first_token_ms(self) -> float:
        if self.first_token_time and self.start_time:
            return (self.first_token_time - self.start_time) * 1000
        return 0.0
    
    @property
    def total_time_ms(self) -> float:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0
    
    @property
    def tokens_per_second(self) -> float:
        duration = self.end_time - self.start_time
        if duration > 0:
            return self.tokens_generated / duration
        return 0.0
    
    def as_dict(self) -> Dict[str, Any]:
        return {
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "total_time_ms": self.total_time_ms,
            "tokens_generated": self.tokens_generated,
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.total_tokens,
            "tokens_per_second": self.tokens_per_second,
        }


@dataclass
class StreamingConfig:
    """Configuration for streaming LLM calls."""
    
    # Buffer size for token accumulation
    buffer_size: int = 50
    
    # Emit partial results every N tokens
    emit_interval: int = 10
    
    # Maximum tokens to generate
    max_tokens: int = 4096
    
    # Timeout for the entire stream
    timeout: float = 120.0
    
    # Safety patterns that trigger early termination
    safety_patterns: list = None
    
    def __post_init__(self):
        if self.safety_patterns is None:
            self.safety_patterns = [
                "rm -rf /",
                "DROP TABLE",
                "DELETE FROM",
                "sudo rm",
            ]


class StreamingLLMClient:
    """Async streaming LLM client.
    
    Usage:
        client = StreamingLLMClient()
        
        # Async generator usage
        async for chunk in client.stream("Write code"):
            print(chunk, end="", flush=True)
        
        # With callback
        await client.stream_with_callback(
            "Write code",
            on_token=lambda t: print(t, end=""),
            on_complete=lambda m: print(f"\\n[{m.tokens_generated} tokens]")
        )
    """
    
    def __init__(
        self,
        config: Optional[StreamingConfig] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.config = config or StreamingConfig()
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or "https://api.deepseek.com/v1"
        self.model = model or "deepseek-coder"
        self._client: Optional[Any] = None
        self._stop_requested = False
    
    async def _ensure_client(self):
        """Ensure async HTTP client is initialized."""
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is required for streaming. Install with: pip install httpx")
        
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout)
            )
    
    async def stream(
        self,
        prompt: str,
        *,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream tokens from the LLM.
        
        Args:
            prompt: User prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system_prompt: Optional system prompt.
            
        Yields:
            Token strings as they arrive.
        """
        await self._ensure_client()
        self._stop_requested = False
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        request_body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "stream": True,
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        async with self._client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=request_body,
            headers=headers,
        ) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if self._stop_requested:
                    break
                
                if not line or not line.startswith("data: "):
                    continue
                
                data = line[6:]  # Remove "data: " prefix
                if data == "[DONE]":
                    break
                
                try:
                    import json
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    
                    if content:
                        # Check for safety patterns
                        if self._check_safety(content):
                            logger.warning("Safety pattern detected, stopping stream")
                            self._stop_requested = True
                            break
                        
                        yield content
                        
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
    
    async def stream_with_callback(
        self,
        prompt: str,
        *,
        on_token: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[StreamingMetrics], None]] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Stream with callbacks for tokens and completion.
        
        Args:
            prompt: User prompt.
            on_token: Callback for each token.
            on_complete: Callback when streaming completes.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens.
            system_prompt: Optional system prompt.
            
        Returns:
            Complete response text.
        """
        metrics = StreamingMetrics(start_time=time.time())
        buffer = []
        
        async for token in self.stream(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        ):
            if metrics.tokens_generated == 0:
                metrics.first_token_time = time.time()
            
            metrics.tokens_generated += 1
            buffer.append(token)
            
            if on_token:
                on_token(token)
        
        metrics.end_time = time.time()
        
        if on_complete:
            on_complete(metrics)
        
        return "".join(buffer)
    
    def request_stop(self) -> None:
        """Request early termination of the stream."""
        self._stop_requested = True
    
    def _check_safety(self, content: str) -> bool:
        """Check if content contains safety-triggering patterns."""
        content_lower = content.lower()
        for pattern in self.config.safety_patterns:
            if pattern.lower() in content_lower:
                return True
        return False
    
    async def close(self) -> None:
        """Close the async client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# === Sync Wrapper ===

class SyncStreamingWrapper:
    """Synchronous wrapper for streaming LLM calls.
    
    Usage:
        wrapper = SyncStreamingWrapper()
        
        for chunk in wrapper.stream("Write code"):
            print(chunk, end="")
    """
    
    def __init__(
        self,
        config: Optional[StreamingConfig] = None,
        **kwargs,
    ):
        self._async_client = StreamingLLMClient(config=config, **kwargs)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create an event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    
    def stream(
        self,
        prompt: str,
        **kwargs,
    ):
        """Synchronous streaming generator."""
        async def collect():
            chunks = []
            async for chunk in self._async_client.stream(prompt, **kwargs):
                chunks.append(chunk)
                yield chunk
        
        # For truly sync streaming, we need to use a thread
        import queue
        import threading
        
        q: queue.Queue = queue.Queue()
        done_event = threading.Event()
        
        def run_async():
            async def _stream():
                async for chunk in self._async_client.stream(prompt, **kwargs):
                    q.put(chunk)
                q.put(None)  # Signal completion
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_stream())
            finally:
                loop.close()
                done_event.set()
        
        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()
        
        while True:
            try:
                chunk = q.get(timeout=0.1)
                if chunk is None:
                    break
                yield chunk
            except queue.Empty:
                if done_event.is_set() and q.empty():
                    break
    
    def stream_full(
        self,
        prompt: str,
        **kwargs,
    ) -> str:
        """Stream and return full response."""
        chunks = list(self.stream(prompt, **kwargs))
        return "".join(chunks)
    
    def request_stop(self) -> None:
        """Request early termination."""
        self._async_client.request_stop()


# === Event Bus Integration ===

class StreamingEventEmitter:
    """Emits streaming events to the CGW event bus.
    
    Events emitted:
    - LLM_STREAM_START: When streaming begins
    - LLM_STREAM_CHUNK: For each token batch
    - LLM_STREAM_COMPLETE: When streaming ends
    - LLM_STREAM_SAFETY: If safety pattern detected
    """
    
    def __init__(
        self,
        event_bus: Any,
        session_id: str,
        cycle_id: int = 0,
    ):
        self.event_bus = event_bus
        self.session_id = session_id
        self.cycle_id = cycle_id
        self._buffer: list = []
        self._buffer_size = 10
    
    def on_start(self) -> None:
        """Emit stream start event."""
        self.event_bus.emit("LLM_STREAM_START", {
            "session_id": self.session_id,
            "cycle_id": self.cycle_id,
            "timestamp": time.time(),
        })
    
    def on_token(self, token: str) -> None:
        """Handle token and emit in batches."""
        self._buffer.append(token)
        
        if len(self._buffer) >= self._buffer_size:
            self._flush_buffer()
    
    def _flush_buffer(self) -> None:
        """Flush accumulated tokens as an event."""
        if self._buffer:
            self.event_bus.emit("LLM_STREAM_CHUNK", {
                "session_id": self.session_id,
                "cycle_id": self.cycle_id,
                "content": "".join(self._buffer),
                "token_count": len(self._buffer),
                "timestamp": time.time(),
            })
            self._buffer = []
    
    def on_complete(self, metrics: StreamingMetrics) -> None:
        """Emit stream complete event."""
        self._flush_buffer()
        
        self.event_bus.emit("LLM_STREAM_COMPLETE", {
            "session_id": self.session_id,
            "cycle_id": self.cycle_id,
            "metrics": metrics.as_dict(),
            "timestamp": time.time(),
        })
    
    def on_safety_trigger(self, pattern: str) -> None:
        """Emit safety trigger event."""
        self.event_bus.emit("LLM_STREAM_SAFETY", {
            "session_id": self.session_id,
            "cycle_id": self.cycle_id,
            "pattern": pattern,
            "timestamp": time.time(),
        })


# === Utility Functions ===

def create_streaming_caller(
    config: Optional[StreamingConfig] = None,
    **kwargs,
) -> Callable:
    """Create a streaming LLM caller function.
    
    Returns:
        A function that matches the llm_caller signature but streams.
    """
    wrapper = SyncStreamingWrapper(config=config, **kwargs)
    
    def caller(
        prompt: str,
        model: str = "deepseek-coder",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        **kw,
    ) -> str:
        return wrapper.stream_full(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    return caller
