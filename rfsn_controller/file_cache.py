"""File content caching with LRU eviction.

Caches frequently accessed files to avoid repeated disk I/O.
Particularly useful for:
- Test files that are read multiple times
- Configuration files
- Frequently referenced source files
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CachedFile:
    """Cached file entry."""
    
    content: str
    path: str
    mtime: float
    size: int
    access_count: int
    last_accessed: float


class FileCache:
    """LRU cache for file contents with staleness checking."""
    
    def __init__(self, max_entries: int = 100, max_size_mb: int = 50):
        """Initialize file cache.
        
        Args:
            max_entries: Maximum number of files to cache
            max_size_mb: Maximum total cache size in MB
        """
        self._cache: OrderedDict[str, CachedFile] = OrderedDict()
        self._max_entries = max_entries
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._current_size_bytes = 0
        
        # Stats
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._stale_reads = 0
    
    def get(self, filepath: str | Path, max_age_seconds: float = 60.0) -> str | None:
        """Get file content from cache or disk.
        
        Args:
            filepath: Path to file
            max_age_seconds: Maximum age before considering cache stale
            
        Returns:
            File content or None if file doesn't exist
        """
        filepath = str(Path(filepath).resolve())
        path_obj = Path(filepath)
        
        # Check if file exists
        if not path_obj.exists():
            return None
        
        current_mtime = path_obj.stat().st_mtime
        now = time.time()
        
        # Check cache
        if filepath in self._cache:
            cached = self._cache[filepath]
            age = now - cached.last_accessed
            
            # Check if cache is stale
            if cached.mtime != current_mtime or age > max_age_seconds:
                # File changed or too old - remove from cache
                self._remove_entry(filepath)
                self._stale_reads += 1
            else:
                # Cache hit
                self._hits += 1
                cached.access_count += 1
                cached.last_accessed = now
                # Move to end (most recently used)
                self._cache.move_to_end(filepath)
                return cached.content
        
        # Cache miss - read from disk
        self._misses += 1
        try:
            content = path_obj.read_text(encoding='utf-8')
        except (UnicodeDecodeError, PermissionError):
            # Can't cache binary or inaccessible files
            return None
        
        # Add to cache
        file_size = len(content.encode('utf-8'))
        self._add_entry(filepath, content, current_mtime, file_size, now)
        
        return content
    
    def _add_entry(
        self,
        filepath: str,
        content: str,
        mtime: float,
        size: int,
        now: float
    ) -> None:
        """Add entry to cache, evicting if necessary."""
        # Evict if at capacity
        while len(self._cache) >= self._max_entries:
            self._evict_lru()
        
        # Evict if over size limit
        while self._current_size_bytes + size > self._max_size_bytes and self._cache:
            self._evict_lru()
        
        # Add new entry
        entry = CachedFile(
            content=content,
            path=filepath,
            mtime=mtime,
            size=size,
            access_count=1,
            last_accessed=now,
        )
        self._cache[filepath] = entry
        self._current_size_bytes += size
    
    def _remove_entry(self, filepath: str) -> None:
        """Remove entry from cache."""
        if filepath in self._cache:
            entry = self._cache.pop(filepath)
            self._current_size_bytes -= entry.size
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._cache:
            filepath, entry = self._cache.popitem(last=False)
            self._current_size_bytes -= entry.size
            self._evictions += 1
    
    def invalidate(self, filepath: str | Path) -> None:
        """Invalidate cached file."""
        filepath = str(Path(filepath).resolve())
        self._remove_entry(filepath)
    
    def clear(self) -> None:
        """Clear all cache."""
        self._cache.clear()
        self._current_size_bytes = 0
        self._evictions += len(self._cache)
    
    def stats(self) -> dict:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'hits': self._hits,
            'misses': self._misses,
            'evictions': self._evictions,
            'stale_reads': self._stale_reads,
            'hit_rate': hit_rate,
            'current_entries': len(self._cache),
            'max_entries': self._max_entries,
            'current_size_mb': self._current_size_bytes / (1024 * 1024),
            'max_size_mb': self._max_size_bytes / (1024 * 1024),
        }


# Global file cache instance
_global_cache: FileCache | None = None


def get_file_cache() -> FileCache:
    """Get global file cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = FileCache()
    return _global_cache


def cached_read_file(filepath: str | Path, max_age_seconds: float = 60.0) -> str | None:
    """Read file with caching.
    
    Args:
        filepath: Path to file
        max_age_seconds: Maximum cache age
        
    Returns:
        File content or None
    """
    cache = get_file_cache()
    return cache.get(filepath, max_age_seconds)
