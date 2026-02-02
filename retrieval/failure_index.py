"""Failure index - persistent storage for failure patterns and fixes."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
import os
from .embeddings import hash_embed, cosine


@dataclass
class FailureRecord:
    """A record of a failure and how it was fixed."""
    repo: str
    signature: str        # compact failure text or stack excerpt
    patch_summary: str    # what fixed it
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "repo": self.repo,
            "signature": self.signature,
            "patch_summary": self.patch_summary,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FailureRecord":
        return cls(
            repo=d["repo"],
            signature=d["signature"],
            patch_summary=d["patch_summary"],
            metadata=d.get("metadata", {}),
        )


class FailureIndex:
    """
    Persistent index of failure patterns and their fixes.
    
    Uses JSONL storage and hashed embeddings for similarity search.
    No external dependencies (FAISS, numpy, etc.).
    """
    
    def __init__(self, path: str = ".rfsn_state/failure_index.jsonl"):
        self.path = path
        self._vecs: List[List[float]] = []
        self._recs: List[FailureRecord] = []
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self._load()

    def _load(self) -> None:
        """Load existing records from disk."""
        if not os.path.exists(self.path):
            return
        
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    rec = FailureRecord.from_dict(obj)
                    self._recs.append(rec)
                    self._vecs.append(hash_embed(rec.signature))
                except (json.JSONDecodeError, KeyError):
                    continue

    def add(self, rec: FailureRecord) -> None:
        """Add a new failure record."""
        self._recs.append(rec)
        self._vecs.append(hash_embed(rec.signature))
        
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec.to_dict()) + "\n")

    def query(
        self, 
        signature: str, 
        k: int = 5, 
        repo_bias: Optional[str] = None,
    ) -> List[FailureRecord]:
        """
        Query for similar failures.
        
        Args:
            signature: The failure signature to match
            k: Number of results to return
            repo_bias: Optional repo to boost in results
            
        Returns:
            Top-k most similar failure records
        """
        if not self._recs:
            return []
        
        qv = hash_embed(signature)
        scored = []
        
        for v, r in zip(self._vecs, self._recs):
            s = cosine(qv, v)
            # Boost same-repo matches
            if repo_bias and r.repo == repo_bias:
                s *= 1.15
            scored.append((s, r))
        
        scored.sort(key=lambda t: t[0], reverse=True)
        return [r for _, r in scored[:k]]
    
    def size(self) -> int:
        """Return number of records in index."""
        return len(self._recs)
    
    def get_repos(self) -> List[str]:
        """Get list of unique repos in index."""
        return list(set(r.repo for r in self._recs))
