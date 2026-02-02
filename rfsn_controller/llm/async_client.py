from __future__ import annotations
from rfsn_controller.no_llm_guard import assert_llm_allowed
"""Async LLM client with streaming and caching support.

This module provides:
1. Async API calls for parallel LLM invocations
2. Streaming responses for early validation
3. Semantic caching for cost reduction
"""

import asyncio
import hashlib
import json
import os
import sqlite3
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

# Try to import httpx for async HTTP
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore


# ============================================================================
# CONFIGURATION
# ============================================================================

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

DEFAULT_TIMEOUT = 120.0  # 2 minutes

# Connection pool settings for HTTP reuse
CONNECTION_POOL_LIMITS = {
    "max_keepalive_connections": 20,
    "max_connections": 100,
    "keepalive_expiry": 30.0,  # seconds
}

# Global HTTP client pool (reuses connections)
_http_clients: dict[str, Any] = {}

def get_http_client(base_url: str) -> Any:
    """Get or create a pooled HTTP client for the given base URL.
    
    Reusing connections saves ~50-100ms per request by avoiding TLS handshakes.
    """
    if not HTTPX_AVAILABLE:
        return None
    
    if base_url not in _http_clients:
        limits = httpx.Limits(
            max_keepalive_connections=CONNECTION_POOL_LIMITS["max_keepalive_connections"],
            max_connections=CONNECTION_POOL_LIMITS["max_connections"],
            keepalive_expiry=CONNECTION_POOL_LIMITS["keepalive_expiry"],
        )
        _http_clients[base_url] = httpx.AsyncClient(
            base_url=base_url,
            limits=limits,
            timeout=DEFAULT_TIMEOUT,
            http2=True,  # Enable HTTP/2 for multiplexing
        )
    return _http_clients[base_url]



# ============================================================================
# ASYNC LLM CLIENTS
# ============================================================================

@dataclass
class AsyncLLMResponse:
    """Response from an async LLM call."""
    
    content: str
    model: str
    temperature: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Parse content as JSON dict."""
        try:
            return json.loads(self.content)
        except json.JSONDecodeError:
            # If content is empty or invalid, try to reconstruct from dictionary if it was already parsed
            # (The underlying call_model_async might return a dict directly)
            return {"mode": "error", "error": "Invalid JSON", "raw": self.content}


async def call_deepseek_async(
    prompt: str,
    *,
    temperature: float = 0.0,
    model: str = "deepseek-chat",
    system_prompt: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> AsyncLLMResponse:
    """Delegate to llm_deepseek.call_model_async."""
    from .deepseek import call_model_async as ds_call
    
    start = time.time()
    try:
        # Note: system_prompt is currently hardcoded in llm_deepseek but we can adapt if needed
        # The prompt construction happens before this usually.
        result = await ds_call(prompt, temperature)
        content = json.dumps(result)
    except Exception as e:
        content = json.dumps({"mode": "error", "error": str(e)})

    latency_ms = (time.time() - start) * 1000
    
    return AsyncLLMResponse(
        content=content,
        model=model,
        temperature=temperature,
        latency_ms=latency_ms,
    )

async def call_gemini_async(
    prompt: str,
    *,
    temperature: float = 0.0,
    model: str = "gemini-2.0-flash",
    system_prompt: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> AsyncLLMResponse:
    """Delegate to llm_gemini.call_model_async."""
    from .gemini import call_model_async as gem_call
    
    start = time.time()
    try:
        result = await gem_call(prompt, temperature)
        content = json.dumps(result)
    except Exception as e:
        content = json.dumps({"mode": "error", "error": str(e)})

    latency_ms = (time.time() - start) * 1000
    
    return AsyncLLMResponse(
        content=content,
        model=model,
        temperature=temperature,
        latency_ms=latency_ms,
    )


# Global cache instance
_cache_instance: LLMCache | None = None


def get_cache(db_path: str | None = None) -> LLMCache:
    """Get the global LLM cache instance."""
    global _cache_instance
    if _cache_instance is None:
        cache_path = db_path or os.path.join(os.path.dirname(__file__), "llm_cache.db")
        _cache_instance = LLMCache(db_path=cache_path)
    return _cache_instance


async def call_deepseek_cached(
    prompt: str,
    *,
    temperature: float = 0.0,
    model: str = "deepseek-chat",
    system_prompt: str | None = None,
    use_cache: bool = True,
    cache: LLMCache | None = None,
) -> AsyncLLMResponse:
    """Call DeepSeek with caching support."""
    if use_cache:
        cache = cache or get_cache()
        cached = cache.get(prompt, model, temperature)
        if cached is not None:
            return cached
    
    response = await call_deepseek_async(
        prompt,
        temperature=temperature,
        model=model,
        system_prompt=system_prompt,
    )
    
    if use_cache and cache and not response.cached:
        if response.content:
            cache.set(prompt, model, temperature, response.content)
    
    return response

async def call_deepseek_streaming(
    prompt: str,
    *,
    temperature: float = 0.0,
    model: str = "deepseek-chat",
    system_prompt: str | None = None,
) -> AsyncIterator[str]:
    """Delegate to llm_deepseek.call_model_streaming."""
    from .deepseek import call_model_streaming as ds_stream
    
    async for chunk in ds_stream(prompt, temperature):
        yield chunk

async def call_gemini_streaming(
    prompt: str,
    *,
    temperature: float = 0.0,
    model: str = "gemini-2.0-flash",
    system_prompt: str | None = None,
) -> AsyncIterator[str]:
    """Delegate to llm_gemini.call_model_streaming."""
    from .gemini import call_model_streaming as gem_stream
    
    async for chunk in gem_stream(prompt, temperature):
        yield chunk


async def call_gemini_cached(
    prompt: str,
    *,
    temperature: float = 0.0,
    model: str = "gemini-2.0-flash",
    system_prompt: str | None = None,
    use_cache: bool = True,
    cache: LLMCache | None = None,
) -> AsyncLLMResponse:
    """Call Gemini with caching support."""
    if use_cache:
        cache = cache or get_cache()
        cached = cache.get(prompt, model, temperature)
        if cached is not None:
            return cached
    
    response = await call_gemini_async(
        prompt,
        temperature=temperature,
        model=model,
        system_prompt=system_prompt,
    )
    
    if use_cache and cache and not response.cached:
        if response.content: # Only cache valid content
            cache.set(prompt, model, temperature, response.content)
    
    return response


# ============================================================================
# PARALLEL LLM CALLS
# ============================================================================

async def call_parallel(
    prompts: list[tuple[str, float]],  # List of (prompt, temperature)
    *,
    model: str = "deepseek-chat",
    system_prompt: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    use_cache: bool = True,
) -> list[AsyncLLMResponse]:
    """Call LLM with multiple prompts/temperatures in parallel.
    
    Args:
        prompts: List of (prompt, temperature) tuples.
        model: Model to use.
        system_prompt: Optional system prompt.
        timeout: Request timeout.
        use_cache: Whether to use semantic caching.
        
    Returns:
        List of responses in same order as input.
    """
    tasks = []
    for prompt, temp in prompts:
        if "deepseek" in model:
            tasks.append(call_deepseek_cached(
                prompt, 
                temperature=temp, 
                model=model, 
                system_prompt=system_prompt, 
                use_cache=use_cache
            ))
        else:
            tasks.append(call_gemini_cached(
                prompt, 
                temperature=temp, 
                model=model, 
                system_prompt=system_prompt, 
                use_cache=use_cache
            ))
            
    return await asyncio.gather(*tasks, return_exceptions=True)


async def generate_patches_parallel(
    prompt: str,
    *,
    temperatures: list[float] | None = None,
    model: str = "deepseek-chat",
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Generate patches at multiple temperatures in parallel."""
    if temperatures is None:
        temperatures = [0.0, 0.4, 0.8] # Increased variance
    
    prompts = [(prompt, temp) for temp in temperatures]
    responses = await call_parallel(
        prompts,
        model=model,
        system_prompt=system_prompt,
    )
    
    patches = []
    for resp in responses:
        if isinstance(resp, Exception):
            patches.append({"mode": "error", "error": str(resp)})
        elif hasattr(resp, "to_dict"):
             patches.append(resp.to_dict())
        else:
             patches.append({"mode": "error", "error": "Unknown response type"})
    
    return patches


# ============================================================================
# SEMANTIC CACHE
# ============================================================================

@dataclass
class LLMCache:
    """SQLite-based cache for LLM responses.
    
    Uses prompt hashing for exact match caching.
    Future: Add embedding-based semantic similarity.
    """
    
    db_path: str
    max_age_hours: int = 72  # Extended TTL for dev workflows
    max_entries: int = 10000
    
    _conn: sqlite3.Connection | None = field(default=None, repr=False)
    
    def __post_init__(self):
        self._ensure_db()
    
    def _ensure_db(self) -> None:
        """Create database and tables if needed."""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                prompt_hash TEXT PRIMARY KEY,
                prompt TEXT,
                model TEXT,
                temperature REAL,
                response TEXT,
                created_at REAL,
                hit_count INTEGER DEFAULT 0
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_created 
            ON cache(created_at)
        """)
        self._conn.commit()
    
    def _hash_prompt(self, prompt: str, model: str, temperature: float) -> str:
        """Create hash key for cache lookup."""
        key = f"{model}:{temperature:.2f}:{prompt}"
        return hashlib.sha256(key.encode()).hexdigest()[:32]
    
    def get(
        self,
        prompt: str,
        model: str,
        temperature: float,
    ) -> AsyncLLMResponse | None:
        """Look up cached response.
        
        Args:
            prompt: The prompt string.
            model: Model name.
            temperature: Temperature used.
            
        Returns:
            Cached response or None.
        """
        if not self._conn:
            return None
        
        prompt_hash = self._hash_prompt(prompt, model, temperature)
        
        cursor = self._conn.execute(
            """
            SELECT response, created_at FROM cache
            WHERE prompt_hash = ?
            """,
            (prompt_hash,),
        )
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        response_json, created_at = row
        
        # Check age
        age_hours = (time.time() - created_at) / 3600
        if age_hours > self.max_age_hours:
            self._conn.execute(
                "DELETE FROM cache WHERE prompt_hash = ?",
                (prompt_hash,),
            )
            self._conn.commit()
            return None
        
        # Update hit count
        self._conn.execute(
            "UPDATE cache SET hit_count = hit_count + 1 WHERE prompt_hash = ?",
            (prompt_hash,),
        )
        self._conn.commit()
        
        return AsyncLLMResponse(
            content=response_json,
            model=model,
            temperature=temperature,
            cached=True,
        )
    
    def set(
        self,
        prompt: str,
        model: str,
        temperature: float,
        response: str,
    ) -> None:
        """Cache a response.
        
        Args:
            prompt: The prompt string.
            model: Model name.
            temperature: Temperature used.
            response: The response content to cache.
        """
        if not self._conn:
            return
        
        prompt_hash = self._hash_prompt(prompt, model, temperature)
        
        self._conn.execute(
            """
            INSERT OR REPLACE INTO cache 
            (prompt_hash, prompt, model, temperature, response, created_at, hit_count)
            VALUES (?, ?, ?, ?, ?, ?, 0)
            """,
            (prompt_hash, prompt[:1000], model, temperature, response, time.time()),
        )
        self._conn.commit()
        
        # Housekeeping
        self._cleanup()
    
    def _cleanup(self) -> None:
        """Remove old entries if over limit."""
        if not self._conn:
            return
        
        # Delete old entries
        cutoff = time.time() - (self.max_age_hours * 3600)
        self._conn.execute(
            "DELETE FROM cache WHERE created_at < ?",
            (cutoff,),
        )
        
        # Delete excess entries (keep most recent)
        cursor = self._conn.execute("SELECT COUNT(*) FROM cache")
        count = cursor.fetchone()[0]
        
        if count > self.max_entries:
            excess = count - self.max_entries
            self._conn.execute(
                """
                DELETE FROM cache WHERE prompt_hash IN (
                    SELECT prompt_hash FROM cache
                    ORDER BY created_at ASC
                    LIMIT ?
                )
                """,
                (excess,),
            )
        
        self._conn.commit()
    
    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if not self._conn:
            return {"error": "not connected"}
        
        cursor = self._conn.execute(
            """
            SELECT 
                COUNT(*) as entries,
                SUM(hit_count) as total_hits,
                AVG(hit_count) as avg_hits
            FROM cache
            """
        )
        row = cursor.fetchone()
        
        return {
            "entries": row[0],
            "total_hits": row[1] or 0,
            "avg_hits": row[2] or 0.0,
        }





# ============================================================================
# SYNC WRAPPER FOR BACKWARD COMPATIBILITY
# ============================================================================

def call_model_async_sync(
    prompt: str,
    temperature: float = 0.0,
    model: str = "deepseek-chat",
    use_cache: bool = True,
) -> dict[str, Any]:
    """Sync wrapper for async LLM call.
    
    Use this as a drop-in replacement for the existing call_model function.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if loop.is_running():
        # Already in async context, create new loop in thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                asyncio.run,
                call_deepseek_cached(
                    prompt,
                    temperature=temperature,
                    model=model,
                    use_cache=use_cache,
                ),
            )
            response = future.result()
    else:
        response = loop.run_until_complete(
            call_deepseek_cached(
                prompt,
                temperature=temperature,
                model=model,
                use_cache=use_cache,
            )
        )
    
    return response.to_dict()
