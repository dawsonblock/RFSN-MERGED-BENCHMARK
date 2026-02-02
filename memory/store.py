"""Simple memory store for episode-level state."""
from __future__ import annotations
from typing import Dict, Any, List
import json
import os


class MemoryStore:
    """
    Simple key-value memory store with tag-based retrieval.
    
    Used for episode-level state that doesn't need embedding search.
    For failure patterns, use retrieval.FailureIndex instead.
    """
    
    def __init__(self, path: str = ".rfsn_state/memory.jsonl"):
        self.path = path
        self.records: List[Dict[str, Any]] = []
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self._load()
    
    def _load(self) -> None:
        """Load existing records from disk."""
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    def add(self, record: Dict[str, Any]) -> None:
        """Add a record to memory."""
        self.records.append(record)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def retrieve(self, query: str, field: str = "tag") -> Dict[str, Any]:
        """Retrieve most recent matching record."""
        for r in reversed(self.records):
            if query in str(r.get(field, "")):
                return r
        return {}
    
    def retrieve_all(self, query: str, field: str = "tag", limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve all matching records, most recent first."""
        matches = []
        for r in reversed(self.records):
            if query in str(r.get(field, "")):
                matches.append(r)
                if len(matches) >= limit:
                    break
        return matches
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about memory contents."""
        tags = {}
        for r in self.records:
            tag = r.get("tag", "unknown")
            tags[tag] = tags.get(tag, 0) + 1
        return {
            "total_records": len(self.records),
            "unique_tags": len(tags),
            "records_by_tag": tags,
        }
