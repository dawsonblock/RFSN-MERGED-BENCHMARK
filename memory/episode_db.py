"""Persistent episode database for cross-task learning.

Stores structured episode data with embeddings for semantic search,
enabling fast retrieval of similar past experiences.

Uses CodeBERT for high-quality code embeddings when available,
with fallback to hash-based embeddings.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any

# Try to use CodeBERT, fall back to hash embeddings
try:
    from retrieval.advanced_embeddings import embed_code
    
    def _embed(text: str) -> list[float]:
        return embed_code(text)
    
    def _cosine(a: list[float], b: list[float]) -> float:
        """Cosine similarity between vectors."""
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        return dot  # Assumes L2 normalized vectors
    
    _USING_CODEBERT = True
except ImportError:
    from retrieval.embeddings import hash_embed, cosine
    
    def _embed(text: str) -> list[float]:
        return hash_embed(text)
    
    def _cosine(a: list[float], b: list[float]) -> float:
        return cosine(a, b)
    
    _USING_CODEBERT = False



@dataclass
class Episode:
    """A structured record of a task attempt."""
    task_id: str
    timestamp: float
    outcome: str  # "pass", "fail", "error"
    repo: str
    error_signature: str  # Compact error/failure text
    patch_summary: str    # What was attempted
    attempt_number: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "outcome": self.outcome,
            "repo": self.repo,
            "error_signature": self.error_signature,
            "patch_summary": self.patch_summary,
            "attempt_number": self.attempt_number,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Episode:
        return cls(
            task_id=d["task_id"],
            timestamp=d["timestamp"],
            outcome=d["outcome"],
            repo=d.get("repo", ""),
            error_signature=d["error_signature"],
            patch_summary=d["patch_summary"],
            attempt_number=d.get("attempt_number", 1),
            metadata=d.get("metadata", {}),
        )


class EpisodeDB:
    """
    Persistent episode database with semantic search capabilities.
    
    Features:
    - SQLite storage for durability
    - Embedding-based similarity search  
    - Outcome-based filtering
    - Repo-biased retrieval
    """
    
    def __init__(self, path: str = ".rfsn_state/episodes.db"):
        self.path = path
        self._lock = threading.Lock()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        # In-memory embedding cache for similarity search
        self._embeddings: list[tuple[int, list[float]]] = []  # (rowid, embedding)
        self._episodes: dict[int, Episode] = {}
        
        self._init_db()
        self._load_embeddings()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    outcome TEXT NOT NULL,
                    repo TEXT,
                    error_signature TEXT,
                    patch_summary TEXT,
                    attempt_number INTEGER DEFAULT 1,
                    metadata TEXT,
                    embedding BLOB
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_task_id ON episodes(task_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_outcome ON episodes(outcome)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_repo ON episodes(repo)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON episodes(timestamp DESC)")
            conn.commit()
    
    def _load_embeddings(self) -> None:
        """Load embeddings into memory for fast similarity search."""
        with sqlite3.connect(self.path) as conn:
            cursor = conn.execute("""
                SELECT id, task_id, timestamp, outcome, repo, error_signature, 
                       patch_summary, attempt_number, metadata, embedding
                FROM episodes
                ORDER BY timestamp DESC
                LIMIT 10000
            """)
            for row in cursor:
                rowid = row[0]
                episode = Episode(
                    task_id=row[1],
                    timestamp=row[2],
                    outcome=row[3],
                    repo=row[4] or "",
                    error_signature=row[5] or "",
                    patch_summary=row[6] or "",
                    attempt_number=row[7] or 1,
                    metadata=json.loads(row[8]) if row[8] else {},
                )
                self._episodes[rowid] = episode
                
                if row[9]:
                    embedding = json.loads(row[9])
                    self._embeddings.append((rowid, embedding))
    
    def add(self, episode: Episode) -> int:
        """
        Add an episode to the database.
        
        Args:
            episode: The episode to store
            
        Returns:
            The row ID of the inserted episode
        """
        embedding = _embed(episode.error_signature)
        
        with self._lock:
            with sqlite3.connect(self.path) as conn:
                cursor = conn.execute("""
                    INSERT INTO episodes 
                    (task_id, timestamp, outcome, repo, error_signature, 
                     patch_summary, attempt_number, metadata, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    episode.task_id,
                    episode.timestamp,
                    episode.outcome,
                    episode.repo,
                    episode.error_signature,
                    episode.patch_summary,
                    episode.attempt_number,
                    json.dumps(episode.metadata),
                    json.dumps(embedding),
                ))
                conn.commit()
                rowid = cursor.lastrowid
            
            # Update in-memory cache
            self._episodes[rowid] = episode
            self._embeddings.append((rowid, embedding))
            
            return rowid
    
    def query_similar(
        self,
        signature: str,
        k: int = 5,
        outcome_filter: str | None = None,
        repo_bias: str | None = None,
        exclude_task_id: str | None = None,
    ) -> list[Episode]:
        """
        Find similar episodes by error signature.
        
        Args:
            signature: Error signature to match
            k: Number of results
            outcome_filter: Filter by outcome ("pass", "fail", etc.)
            repo_bias: Boost same-repo results
            exclude_task_id: Exclude episodes from this task
            
        Returns:
            Top-k most similar episodes
        """
        if not self._embeddings:
            return []
        
        query_vec = _embed(signature)
        scored: list[tuple[float, int]] = []
        
        for rowid, embedding in self._embeddings:
            episode = self._episodes.get(rowid)
            if not episode:
                continue
                
            # Apply filters
            if outcome_filter and episode.outcome != outcome_filter:
                continue
            if exclude_task_id and episode.task_id == exclude_task_id:
                continue
            
            # Compute similarity
            score = _cosine(query_vec, embedding)
            
            # Repo bias
            if repo_bias and episode.repo == repo_bias:
                score *= 1.2
            
            scored.append((score, rowid))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for _score, rowid in scored[:k]:
            if episode := self._episodes.get(rowid):
                results.append(episode)
        
        return results
    
    def query_by_task(self, task_id: str) -> list[Episode]:
        """Get all episodes for a specific task."""
        return [
            ep for ep in self._episodes.values()
            if ep.task_id == task_id
        ]
    
    def query_recent(
        self,
        limit: int = 20,
        outcome_filter: str | None = None,
    ) -> list[Episode]:
        """Get most recent episodes."""
        episodes = list(self._episodes.values())
        
        if outcome_filter:
            episodes = [ep for ep in episodes if ep.outcome == outcome_filter]
        
        episodes.sort(key=lambda x: x.timestamp, reverse=True)
        return episodes[:limit]
    
    def get_success_rate(self, repo: str | None = None) -> dict[str, float]:
        """Calculate success rates."""
        episodes = list(self._episodes.values())
        
        if repo:
            episodes = [ep for ep in episodes if ep.repo == repo]
        
        if not episodes:
            return {"pass": 0.0, "fail": 0.0, "total": 0}
        
        pass_count = sum(1 for ep in episodes if ep.outcome == "pass")
        fail_count = sum(1 for ep in episodes if ep.outcome == "fail")
        
        return {
            "pass": pass_count / len(episodes),
            "fail": fail_count / len(episodes),
            "total": len(episodes),
        }
    
    def size(self) -> int:
        """Return number of episodes."""
        return len(self._episodes)
    
    def get_repos(self) -> list[str]:
        """Get list of unique repos."""
        return list({ep.repo for ep in self._episodes.values() if ep.repo})


# Global instance
_episode_db: EpisodeDB | None = None
_db_lock = threading.Lock()


def get_episode_db(path: str | None = None) -> EpisodeDB:
    """Get the global episode database instance."""
    global _episode_db
    with _db_lock:
        if _episode_db is None:
            _episode_db = EpisodeDB(path or ".rfsn_state/episodes.db")
        return _episode_db


def record_episode(
    task_id: str,
    outcome: str,
    repo: str,
    error_signature: str,
    patch_summary: str,
    attempt_number: int = 1,
    metadata: dict[str, Any] | None = None,
) -> Episode:
    """
    Convenience function to record an episode.
    
    Returns the created episode.
    """
    db = get_episode_db()
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
    db.add(episode)
    return episode
