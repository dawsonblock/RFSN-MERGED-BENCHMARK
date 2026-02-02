"""Plan Cache - Cache and reuse successful plans.

Caches successful plans for similar goals to avoid redundant decomposition.
Uses goal similarity and repo fingerprinting for cache key matching.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .schema import Plan


@dataclass
class CacheEntry:
    """A cached plan entry."""
    
    goal: str
    goal_hash: str
    context_hash: str
    plan_json: str
    final_status: str
    created_at: str
    access_count: int = 0
    success_rate: float = 1.0
    repo_fingerprint: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "goal_hash": self.goal_hash,
            "context_hash": self.context_hash,
            "plan_json": self.plan_json,
            "final_status": self.final_status,
            "created_at": self.created_at,
            "access_count": self.access_count,
            "success_rate": self.success_rate,
            "repo_fingerprint": self.repo_fingerprint,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheEntry:
        return cls(
            goal=data["goal"],
            goal_hash=data["goal_hash"],
            context_hash=data["context_hash"],
            plan_json=data["plan_json"],
            final_status=data["final_status"],
            created_at=data["created_at"],
            access_count=data.get("access_count", 0),
            success_rate=data.get("success_rate", 1.0),
            repo_fingerprint=data.get("repo_fingerprint", ""),
        )


class PlanCache:
    """Caches successful plans for goal similarity matching."""
    
    def __init__(
        self,
        cache_dir: Path | None = None,
        max_entries: int = 100,
        similarity_threshold: float = 0.85,
    ):
        """Initialize the cache.
        
        Args:
            cache_dir: Directory for persistent cache. None for memory-only.
            max_entries: Maximum cache entries.
            similarity_threshold: Minimum similarity for cache hit.
        """
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._max_entries = max_entries
        self._similarity_threshold = similarity_threshold
        self._memory_cache: dict[str, CacheEntry] = {}
        
        # Load persistent cache
        if self._cache_dir:
            self._load_cache()
    
    def get(
        self,
        goal: str,
        context: dict[str, Any],
        repo_fingerprint: str = "",
    ) -> tuple[Plan, float] | None:
        """Look up a cached plan for a similar goal.
        
        Args:
            goal: The goal to find a plan for.
            context: Execution context.
            repo_fingerprint: Current repo fingerprint for invalidation.
            
        Returns:
            Tuple of (Plan, similarity_score) if found, None otherwise.
        """
        from .schema import Plan
        
        goal_hash = self._hash_goal(goal)
        context_hash = self._hash_context(context)
        
        # Exact match first
        exact_key = f"{goal_hash}:{context_hash}"
        if exact_key in self._memory_cache:
            entry = self._memory_cache[exact_key]
            
            # Check fingerprint if provided
            if repo_fingerprint and entry.repo_fingerprint:
                if repo_fingerprint != entry.repo_fingerprint:
                    # Invalidate stale entry
                    del self._memory_cache[exact_key]
                    return None
            
            entry.access_count += 1
            return Plan.from_json(entry.plan_json), 1.0
        
        # Similarity search
        best_match = None
        best_score = 0.0
        
        for _key, entry in self._memory_cache.items():
            # Check fingerprint first
            if repo_fingerprint and entry.repo_fingerprint:
                if repo_fingerprint != entry.repo_fingerprint:
                    continue  # Skip stale entries
            
            score = self.similarity_score(goal, entry.goal)
            if score > best_score and score >= self._similarity_threshold:
                best_score = score
                best_match = entry
        
        if best_match:
            best_match.access_count += 1
            return Plan.from_json(best_match.plan_json), best_score
        
        return None
    
    def put(
        self,
        goal: str,
        context: dict[str, Any],
        plan: Plan,
        final_status: str,
        repo_fingerprint: str = "",
    ) -> None:
        """Store a plan in the cache.
        
        Args:
            goal: The goal for this plan.
            context: Execution context.
            plan: The plan to cache.
            final_status: Final execution status (success, failed, etc.)
            repo_fingerprint: Repo fingerprint for invalidation.
        """
        # Only cache successful plans
        if final_status not in ("success", "complete"):
            return
        
        goal_hash = self._hash_goal(goal)
        context_hash = self._hash_context(context)
        key = f"{goal_hash}:{context_hash}"
        
        entry = CacheEntry(
            goal=goal,
            goal_hash=goal_hash,
            context_hash=context_hash,
            plan_json=plan.to_json(),
            final_status=final_status,
            created_at=datetime.now(UTC).isoformat(),
            repo_fingerprint=repo_fingerprint,
        )
        
        # Evict if at capacity
        if len(self._memory_cache) >= self._max_entries:
            self._evict_oldest()
        
        self._memory_cache[key] = entry
        
        # Persist
        if self._cache_dir:
            self._save_entry(key, entry)
    
    def invalidate(self, fingerprint: str) -> int:
        """Invalidate all entries with a specific fingerprint.
        
        Args:
            fingerprint: Fingerprint to invalidate.
            
        Returns:
            Number of entries invalidated.
        """
        to_delete = [
            k for k, v in self._memory_cache.items()
            if v.repo_fingerprint == fingerprint
        ]
        
        for key in to_delete:
            del self._memory_cache[key]
            if self._cache_dir:
                path = self._cache_dir / f"{key}.json"
                if path.exists():
                    path.unlink()
        
        return len(to_delete)
    
    def similarity_score(self, goal1: str, goal2: str) -> float:
        """Calculate similarity between two goals.
        
        Uses normalized n-gram Jaccard similarity.
        
        Args:
            goal1: First goal.
            goal2: Second goal.
            
        Returns:
            Similarity score between 0.0 and 1.0.
        """
        # Normalize
        g1 = self._normalize_goal(goal1)
        g2 = self._normalize_goal(goal2)
        
        # Extract n-grams (2-grams and 3-grams)
        ngrams1 = self._extract_ngrams(g1, 2) | self._extract_ngrams(g1, 3)
        ngrams2 = self._extract_ngrams(g2, 2) | self._extract_ngrams(g2, 3)
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = ngrams1 & ngrams2
        union = ngrams1 | ngrams2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _normalize_goal(self, goal: str) -> str:
        """Normalize goal for comparison."""
        # Lowercase
        goal = goal.lower()
        # Remove punctuation
        goal = re.sub(r'[^\w\s]', ' ', goal)
        # Collapse whitespace
        goal = re.sub(r'\s+', ' ', goal).strip()
        return goal
    
    def _extract_ngrams(self, text: str, n: int) -> set:
        """Extract n-grams from text."""
        words = text.split()
        if len(words) < n:
            return {text}
        return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}
    
    def _hash_goal(self, goal: str) -> str:
        """Create hash of goal."""
        normalized = self._normalize_goal(goal)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def _hash_context(self, context: dict[str, Any]) -> str:
        """Create hash of context."""
        # Only hash stable context keys
        stable_keys = ["repo_type", "language", "test_cmd"]
        stable_context = {k: context.get(k, "") for k in stable_keys}
        context_str = json.dumps(stable_context, sort_keys=True)
        return hashlib.sha256(context_str.encode()).hexdigest()[:12]
    
    def _evict_oldest(self) -> None:
        """Evict least recently used entry."""
        if not self._memory_cache:
            return
        
        # Evict entry with lowest access count
        min_key = min(
            self._memory_cache.keys(),
            key=lambda k: self._memory_cache[k].access_count,
        )
        del self._memory_cache[min_key]
        
        if self._cache_dir:
            path = self._cache_dir / f"{min_key}.json"
            if path.exists():
                path.unlink()
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        if not self._cache_dir:
            return
        
        for path in self._cache_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                entry = CacheEntry.from_dict(data)
                key = path.stem
                self._memory_cache[key] = entry
            except (json.JSONDecodeError, KeyError):
                continue
    
    def _save_entry(self, key: str, entry: CacheEntry) -> None:
        """Save entry to disk."""
        if not self._cache_dir:
            return
        
        path = self._cache_dir / f"{key}.json"
        path.write_text(json.dumps(entry.to_dict(), indent=2))
    
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "entries": len(self._memory_cache),
            "max_entries": self._max_entries,
            "total_accesses": sum(e.access_count for e in self._memory_cache.values()),
            "avg_success_rate": (
                sum(e.success_rate for e in self._memory_cache.values()) / len(self._memory_cache)
                if self._memory_cache else 0.0
            ),
        }
    
    def clear(self) -> int:
        """Clear the cache.
        
        Returns:
            Number of entries cleared.
        """
        count = len(self._memory_cache)
        self._memory_cache.clear()
        
        if self._cache_dir:
            for path in self._cache_dir.glob("*.json"):
                path.unlink()
        
        return count
