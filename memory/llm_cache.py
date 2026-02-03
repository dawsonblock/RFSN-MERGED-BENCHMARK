"""LLM response caching layer for reducing redundant API calls.

Provides both exact-match and semantic similarity caching to avoid
repeated LLM calls for similar or identical prompts.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Try to import embedding support
try:
    from retrieval.advanced_embeddings import embed_code, HAS_TRANSFORMERS
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    HAS_TRANSFORMERS = False


@dataclass
class CacheConfig:
    """Configuration for LLM cache."""
    db_path: str = ".rfsn_state/llm_cache.db"
    max_entries: int = 10000
    ttl_seconds: int = 86400 * 7  # 7 days
    semantic_threshold: float = 0.92  # Similarity threshold for semantic match
    enable_semantic: bool = True  # Use semantic similarity caching


@dataclass
class CacheEntry:
    """A cached LLM response."""
    prompt_hash: str
    prompt_text: str
    response: str
    model: str
    timestamp: float
    tokens_used: int = 0
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self, ttl: int) -> bool:
        return time.time() - self.timestamp > ttl


class LLMCache:
    """
    LLM response cache with exact and semantic matching.
    
    Features:
    - Exact match: Hash-based lookup for identical prompts
    - Semantic match: Embedding similarity for similar prompts
    - SQLite persistence: Survives restarts
    - TTL expiration: Automatic cleanup of old entries
    - Model-aware: Separate caches per model
    """
    
    def __init__(self, config: CacheConfig | None = None):
        self.config = config or CacheConfig()
        self._lock = threading.Lock()
        self._db_path = Path(self.config.db_path)
        self._stats = {
            "hits_exact": 0,
            "hits_semantic": 0,
            "misses": 0,
            "tokens_saved": 0,
        }
        
        # In-memory cache for hot entries
        self._hot_cache: dict[str, CacheEntry] = {}
        
        # Initialize database
        self._init_db()
        self._load_hot_cache()
    
    def _init_db(self) -> None:
        """Initialize SQLite database."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    prompt_hash TEXT PRIMARY KEY,
                    prompt_text TEXT NOT NULL,
                    response TEXT NOT NULL,
                    model TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    tokens_used INTEGER DEFAULT 0,
                    embedding TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_timestamp
                ON cache_entries (model, timestamp)
            """)
            conn.commit()
    
    def _load_hot_cache(self, limit: int = 500) -> None:
        """Load recent entries into hot cache."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute("""
                    SELECT prompt_hash, prompt_text, response, model, 
                           timestamp, tokens_used, embedding, metadata
                    FROM cache_entries
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))
                
                for row in cursor:
                    entry = CacheEntry(
                        prompt_hash=row[0],
                        prompt_text=row[1],
                        response=row[2],
                        model=row[3],
                        timestamp=row[4],
                        tokens_used=row[5] or 0,
                        embedding=json.loads(row[6]) if row[6] else None,
                        metadata=json.loads(row[7]) if row[7] else {},
                    )
                    self._hot_cache[entry.prompt_hash] = entry
                
                logger.debug("Loaded %d entries into hot cache", len(self._hot_cache))
        except Exception as e:
            logger.warning("Failed to load hot cache: %s", e)
    
    def _hash_prompt(self, prompt: str, model: str) -> str:
        """Create a hash key for a prompt + model combination."""
        combined = f"{model}::{prompt}"
        return hashlib.sha256(combined.encode()).hexdigest()[:32]
    
    def get(
        self,
        prompt: str,
        model: str,
        semantic_search: bool = True,
    ) -> tuple[str | None, str]:
        """
        Look up a cached response.
        
        Args:
            prompt: The prompt text
            model: The model identifier
            semantic_search: Whether to try semantic matching
            
        Returns:
            Tuple of (response, match_type) where match_type is:
            - "exact": Exact hash match
            - "semantic": Similar prompt match
            - "miss": No match found
        """
        prompt_hash = self._hash_prompt(prompt, model)
        
        # 1. Check hot cache (exact match)
        with self._lock:
            if prompt_hash in self._hot_cache:
                entry = self._hot_cache[prompt_hash]
                if not entry.is_expired(self.config.ttl_seconds):
                    self._stats["hits_exact"] += 1
                    self._stats["tokens_saved"] += entry.tokens_used
                    return entry.response, "exact"
        
        # 2. Check database (exact match)
        entry = self._get_from_db(prompt_hash)
        if entry and not entry.is_expired(self.config.ttl_seconds):
            self._hot_cache[prompt_hash] = entry
            self._stats["hits_exact"] += 1
            self._stats["tokens_saved"] += entry.tokens_used
            return entry.response, "exact"
        
        # 3. Semantic search (if enabled)
        if (
            semantic_search 
            and self.config.enable_semantic 
            and HAS_EMBEDDINGS 
            and HAS_TRANSFORMERS
        ):
            semantic_result = self._semantic_search(prompt, model)
            if semantic_result:
                self._stats["hits_semantic"] += 1
                self._stats["tokens_saved"] += semantic_result.tokens_used
                return semantic_result.response, "semantic"
        
        # 4. Cache miss
        self._stats["misses"] += 1
        return None, "miss"
    
    def _get_from_db(self, prompt_hash: str) -> CacheEntry | None:
        """Get entry from database."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute("""
                    SELECT prompt_hash, prompt_text, response, model,
                           timestamp, tokens_used, embedding, metadata
                    FROM cache_entries
                    WHERE prompt_hash = ?
                """, (prompt_hash,))
                
                row = cursor.fetchone()
                if row:
                    return CacheEntry(
                        prompt_hash=row[0],
                        prompt_text=row[1],
                        response=row[2],
                        model=row[3],
                        timestamp=row[4],
                        tokens_used=row[5] or 0,
                        embedding=json.loads(row[6]) if row[6] else None,
                        metadata=json.loads(row[7]) if row[7] else {},
                    )
        except Exception as e:
            logger.warning("Database lookup failed: %s", e)
        
        return None
    
    def _semantic_search(
        self,
        prompt: str,
        model: str,
    ) -> CacheEntry | None:
        """Find semantically similar cached prompt."""
        if not HAS_EMBEDDINGS:
            return None
        
        try:
            query_embedding = embed_code(prompt)
            
            best_match: CacheEntry | None = None
            best_similarity = self.config.semantic_threshold
            
            # Search hot cache first
            for entry in self._hot_cache.values():
                if entry.model != model:
                    continue
                if entry.is_expired(self.config.ttl_seconds):
                    continue
                if not entry.embedding:
                    continue
                
                similarity = self._cosine_similarity(
                    query_embedding, entry.embedding
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = entry
            
            if best_match:
                logger.debug(
                    "Semantic cache hit: similarity=%.3f",
                    best_similarity
                )
            
            return best_match
            
        except Exception as e:
            logger.warning("Semantic search failed: %s", e)
            return None
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between vectors."""
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        return dot  # Assumes L2 normalized vectors
    
    def put(
        self,
        prompt: str,
        response: str,
        model: str,
        tokens_used: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Cache a response.
        
        Args:
            prompt: The prompt text
            response: The LLM response
            model: The model identifier
            tokens_used: Number of tokens used
            metadata: Additional metadata
        """
        prompt_hash = self._hash_prompt(prompt, model)
        
        # Generate embedding for semantic search
        embedding = None
        if self.config.enable_semantic and HAS_EMBEDDINGS:
            try:
                embedding = embed_code(prompt)
            except Exception as e:
                logger.warning("Failed to generate embedding: %s", e)
        
        entry = CacheEntry(
            prompt_hash=prompt_hash,
            prompt_text=prompt[:5000],  # Truncate for storage
            response=response,
            model=model,
            timestamp=time.time(),
            tokens_used=tokens_used,
            embedding=embedding,
            metadata=metadata or {},
        )
        
        # Add to hot cache
        with self._lock:
            self._hot_cache[prompt_hash] = entry
        
        # Persist to database
        self._save_to_db(entry)
        
        # Cleanup if needed
        if len(self._hot_cache) > self.config.max_entries:
            self._cleanup()
    
    def _save_to_db(self, entry: CacheEntry) -> None:
        """Save entry to database."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries
                    (prompt_hash, prompt_text, response, model, 
                     timestamp, tokens_used, embedding, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.prompt_hash,
                    entry.prompt_text,
                    entry.response,
                    entry.model,
                    entry.timestamp,
                    entry.tokens_used,
                    json.dumps(entry.embedding) if entry.embedding else None,
                    json.dumps(entry.metadata) if entry.metadata else None,
                ))
                conn.commit()
        except Exception as e:
            logger.warning("Failed to save to database: %s", e)
    
    def _cleanup(self) -> None:
        """Clean up expired entries."""
        cutoff = time.time() - self.config.ttl_seconds
        
        # Clean hot cache
        with self._lock:
            expired = [
                h for h, e in self._hot_cache.items()
                if e.timestamp < cutoff
            ]
            for h in expired:
                del self._hot_cache[h]
        
        # Clean database
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM cache_entries
                    WHERE timestamp < ?
                """, (cutoff,))
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info("Cleaned up %d expired cache entries", cursor.rowcount)
        except Exception as e:
            logger.warning("Cache cleanup failed: %s", e)
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_hits = self._stats["hits_exact"] + self._stats["hits_semantic"]
        total_requests = total_hits + self._stats["misses"]
        
        return {
            "hits_exact": self._stats["hits_exact"],
            "hits_semantic": self._stats["hits_semantic"],
            "misses": self._stats["misses"],
            "hit_rate": total_hits / max(total_requests, 1),
            "tokens_saved": self._stats["tokens_saved"],
            "hot_cache_size": len(self._hot_cache),
        }
    
    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._hot_cache.clear()
        
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("DELETE FROM cache_entries")
                conn.commit()
        except Exception as e:
            logger.warning("Failed to clear cache: %s", e)


# Global cache holder (avoids 'global' statement)
_cache_holder: dict[str, LLMCache] = {}
_cache_lock = threading.Lock()


def get_llm_cache(config: CacheConfig | None = None) -> LLMCache:
    """Get the global LLM cache instance."""
    with _cache_lock:
        if "instance" not in _cache_holder:
            _cache_holder["instance"] = LLMCache(config)
        return _cache_holder["instance"]


def cached_llm_call(
    prompt: str,
    model: str,
    llm_fn: Any,  # Callable that takes prompt and returns response
    tokens_used: int = 0,
    bypass_cache: bool = False,
    **kwargs: Any,
) -> tuple[str, bool]:
    """
    Wrapper for LLM calls with caching.
    
    Args:
        prompt: The prompt text
        model: The model identifier
        llm_fn: Function to call if cache miss
        tokens_used: Estimated tokens (for stats)
        bypass_cache: Skip cache lookup
        **kwargs: Additional arguments for llm_fn
        
    Returns:
        Tuple of (response, was_cached)
    """
    cache = get_llm_cache()
    
    # Check cache first
    if not bypass_cache:
        cached_response, match_type = cache.get(prompt, model)
        if cached_response is not None:
            logger.debug("LLM cache %s hit", match_type)
            return cached_response, True
    
    # Call LLM
    response = llm_fn(prompt, **kwargs)
    
    # Cache result
    cache.put(prompt, response, model, tokens_used=tokens_used)
    
    return response, False


class CachingLLMWrapper:
    """
    Wrapper class to add caching to any LLM client.
    
    Usage:
        original_client = OpenAI()
        cached_client = CachingLLMWrapper(original_client, model="gpt-4")
        
        response = cached_client.generate(prompt)
    """
    
    def __init__(
        self,
        client: Any,
        model: str,
        config: CacheConfig | None = None,
    ):
        self.client = client
        self.model = model
        self.cache = LLMCache(config) if config else get_llm_cache()
    
    def generate(
        self,
        prompt: str,
        bypass_cache: bool = False,
        **kwargs: Any,
    ) -> str:
        """Generate response with caching."""
        # Check cache
        if not bypass_cache:
            cached, match_type = self.cache.get(prompt, self.model)
            if cached:
                return cached
        
        # Call underlying client
        # This is generic - subclass for specific client APIs
        response = self._call_client(prompt, **kwargs)
        
        # Cache result
        self.cache.put(prompt, response, self.model)
        
        return response
    
    def _call_client(self, prompt: str, **kwargs: Any) -> str:
        """Call the underlying LLM client. Override for specific APIs."""
        # Default: assume client is callable
        return self.client(prompt, **kwargs)
    
    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
