"""Unified multi-tier memory system.

Provides a single interface for all memory operations with automatic
tier promotion and fallback.

Tiers:
- L1: In-memory LRU cache (hot, fast, volatile)
- L2: Semantic cache (warm, SQLite-backed, embedding-indexed)  
- L3: Failure index (cold, persistent, pattern-matching)
- L4: Episode database (archive, searchable, cross-task learning)
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from retrieval.failure_index import FailureIndex, FailureRecord
from memory.episode_db import Episode, get_episode_db


@dataclass
class MemoryHit:
    """A result from memory lookup."""
    tier: str           # "L1", "L2", "L3", "L4"
    key: str
    value: Any
    score: float = 1.0  # Similarity score for semantic matches
    source: str = ""    # Additional context
    
    def __repr__(self) -> str:
        return f"MemoryHit({self.tier}, score={self.score:.3f}, source={self.source})"


class LRUCache:
    """Thread-safe LRU cache for L1 tier."""
    
    def __init__(self, maxsize: int = 1000, ttl: float = 3600.0):
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Any | None:
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            timestamp, value = self._cache[key]
            
            # Check TTL
            if time.time() - timestamp > self._ttl:
                del self._cache[key]
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return value
    
    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            elif len(self._cache) >= self._maxsize:
                # Remove oldest
                self._cache.popitem(last=False)
                    
            self._cache[key] = (time.time(), value)
    
    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }


class UnifiedMemory:
    """
    Multi-tier memory manager.
    
    Provides unified access to all memory tiers with automatic
    tier promotion and intelligent fallback.
    """
    
    def __init__(
        self,
        l1_maxsize: int = 1000,
        l1_ttl: float = 3600.0,
        failure_index_path: str = ".rfsn_state/failure_index.jsonl",
        episode_db_path: str = ".rfsn_state/episodes.db",
    ):
        # L1: Fast in-memory cache
        self.l1 = LRUCache(maxsize=l1_maxsize, ttl=l1_ttl)
        
        # L3: Failure patterns
        self.l3 = FailureIndex(path=failure_index_path)
        
        # L4: Episode history
        self.l4 = get_episode_db(episode_db_path)
        
        # Statistics
        self._stats = {"l1": 0, "l3": 0, "l4": 0, "miss": 0}
        self._lock = threading.Lock()
    
    def _hash_query(self, query: str, context: dict[str, Any] | None = None) -> str:
        """Create a hash key for L1 cache."""
        parts = [query]
        if context:
            parts.append(str(sorted(context.items())))
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]
    
    def retrieve(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        repo: str | None = None,
        k: int = 5,
        min_score: float = 0.3,
        tiers: str = "1234",  # Which tiers to check
    ) -> list[MemoryHit]:
        """
        Retrieve from memory using multi-tier lookup.
        
        Args:
            query: Search query (error signature, problem statement, etc.)
            context: Optional context dict for cache key
            repo: Optional repo for biasing results
            k: Max results per tier
            min_score: Minimum similarity score
            tiers: Which tiers to check ("1234" = all)
            
        Returns:
            List of memory hits from various tiers
        """
        results: list[MemoryHit] = []
        
        # L1: Exact match cache
        if "1" in tiers:
            cache_key = self._hash_query(query, context)
            if cached := self.l1.get(cache_key):
                with self._lock:
                    self._stats["l1"] += 1
                results.append(MemoryHit(
                    tier="L1",
                    key=cache_key,
                    value=cached,
                    score=1.0,
                    source="l1_cache",
                ))
        
        # L3: Failure patterns
        if "3" in tiers:
            failures = self.l3.query(query, k=k, repo_bias=repo)
            for rec in failures:
                with self._lock:
                    self._stats["l3"] += 1
                results.append(MemoryHit(
                    tier="L3",
                    key=rec.signature[:50],
                    value=rec,
                    score=0.8,  # Approximate
                    source="failure_index",
                ))
        
        # L4: Episode history
        if "4" in tiers:
            episodes = self.l4.query_similar(
                query,
                k=k,
                repo_bias=repo,
                outcome_filter="pass",  # Prefer successful episodes
            )
            for ep in episodes:
                with self._lock:
                    self._stats["l4"] += 1
                results.append(MemoryHit(
                    tier="L4",
                    key=ep.task_id,
                    value=ep,
                    score=0.7,  # Approximate
                    source="episode_db",
                ))
        
        if not results:
            with self._lock:
                self._stats["miss"] += 1
        
        return results
    
    def store_failure(
        self,
        signature: str,
        patch_summary: str,
        repo: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a failure pattern in L3."""
        rec = FailureRecord(
            repo=repo,
            signature=signature,
            patch_summary=patch_summary,
            metadata=metadata or {},
        )
        self.l3.add(rec)
    
    def store_episode(
        self,
        task_id: str,
        outcome: str,
        repo: str,
        error_signature: str,
        patch_summary: str,
        attempt_number: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> Episode:
        """Store an episode in L4."""
        episode = Episode(
            task_id=task_id,
            timestamp=time.time(),
            outcome=outcome,
            repo=repo,
            error_signature=error_signature,
            patch_summary=patch_summary,
            attempt_number=attempt_number,
            metadata=metadata or {},
        )
        self.l4.add(episode)
        return episode
    
    def cache_result(
        self,
        query: str,
        context: dict[str, Any] | None,
        value: Any,
    ) -> None:
        """Store a result in L1 cache."""
        cache_key = self._hash_query(query, context)
        self.l1.put(cache_key, value)
    
    def get_similar_fixes(
        self,
        signature: str,
        repo: str | None = None,
        k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Get similar fixes from memory (convenience method).
        
        Returns formatted results for injection into LLM context.
        """
        results = []
        
        # Check L3 (failure patterns)
        failures = self.l3.query(signature, k=k, repo_bias=repo)
        for rec in failures:
            results.append({
                "source": "failure_index",
                "repo": rec.repo,
                "signature": rec.signature[:200],
                "fix_summary": rec.patch_summary,
            })
        
        # Check L4 (successful episodes)
        episodes = self.l4.query_similar(
            signature,
            k=k,
            repo_bias=repo,
            outcome_filter="pass",
        )
        for ep in episodes:
            results.append({
                "source": "episode_history",
                "task_id": ep.task_id,
                "repo": ep.repo,
                "fix_summary": ep.patch_summary,
            })
        
        return results[:k]
    
    def get_stats(self) -> dict[str, Any]:
        """Get memory system statistics."""
        with self._lock:
            return {
                "l1": self.l1.get_stats(),
                "l3_size": self.l3.size(),
                "l4_size": self.l4.size(),
                "tier_hits": dict(self._stats),
            }


# Global instance
_unified_memory: UnifiedMemory | None = None
_memory_lock = threading.Lock()


def get_unified_memory() -> UnifiedMemory:
    """Get the global unified memory instance."""
    global _unified_memory
    with _memory_lock:
        if _unified_memory is None:
            _unified_memory = UnifiedMemory()
        return _unified_memory
