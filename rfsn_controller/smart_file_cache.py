"""Smart file reading with caching and compression.

This module provides optimized file reading with:
1. LRU caching with TTL
2. Compression for large files
3. Incremental reading for diffs
4. Git-aware file tracking
"""

from __future__ import annotations

import gzip
import hashlib
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

# ============================================================================
# FILE CACHE
# ============================================================================

@dataclass
class CachedFile:
    """A cached file with metadata."""
    
    path: str
    content: str
    size: int
    mtime: float
    hash: str
    compressed: bytes | None = None
    cached_at: float = 0.0
    hits: int = 0
    
    @classmethod
    def from_file(cls, path: str, content: str) -> CachedFile:
        """Create a cached file from content."""
        stat = os.stat(path) if os.path.exists(path) else None
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        cf = cls(
            path=path,
            content=content,
            size=len(content),
            mtime=stat.st_mtime if stat else 0.0,
            hash=content_hash,
            cached_at=time.time(),
        )
        
        # Compress large files
        if len(content) > 10000:
            cf.compressed = gzip.compress(content.encode())
        
        return cf
    
    def get_content(self) -> str:
        """Get file content, decompressing if needed."""
        if self.content:
            return self.content
        if self.compressed:
            self.content = gzip.decompress(self.compressed).decode()
            return self.content
        return ""
    
    def is_stale(self, path: str, max_age: float = 60.0) -> bool:
        """Check if cache entry is stale."""
        if time.time() - self.cached_at > max_age:
            return True
        
        if os.path.exists(path):
            stat = os.stat(path)
            if stat.st_mtime > self.mtime:
                return True
        
        return False


class SmartFileCache:
    """Smart file cache with LRU eviction and staleness checking."""
    
    def __init__(
        self,
        max_size: int = 100,
        max_memory_mb: int = 50,
        default_ttl: float = 60.0,
    ):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.default_ttl = default_ttl
        
        self._cache: dict[str, CachedFile] = {}
        self._access_order: list[str] = []
        self._lock = threading.Lock()
        self._memory_used = 0
    
    def get(self, path: str, max_age: float | None = None) -> str | None:
        """Get a file from cache if available and fresh.
        
        Args:
            path: Absolute path to the file.
            max_age: Maximum age in seconds (uses default if None).
            
        Returns:
            File content or None if not cached/stale.
        """
        max_age = max_age or self.default_ttl
        
        with self._lock:
            if path in self._cache:
                cached = self._cache[path]
                
                if not cached.is_stale(path, max_age):
                    # Update access order (LRU)
                    if path in self._access_order:
                        self._access_order.remove(path)
                    self._access_order.append(path)
                    cached.hits += 1
                    
                    return cached.get_content()
                else:
                    # Remove stale entry
                    self._remove_entry(path)
        
        return None
    
    def put(self, path: str, content: str) -> None:
        """Add a file to the cache.
        
        Args:
            path: Absolute path to the file.
            content: File content.
        """
        cached = CachedFile.from_file(path, content)
        memory_needed = cached.size
        
        with self._lock:
            # Evict if over size limit
            while len(self._cache) >= self.max_size and self._access_order:
                oldest = self._access_order[0]
                self._remove_entry(oldest)
            
            # Evict if over memory limit
            while (
                self._memory_used + memory_needed > self.max_memory_mb * 1024 * 1024
                and self._access_order
            ):
                oldest = self._access_order[0]
                self._remove_entry(oldest)
            
            # Add new entry
            self._cache[path] = cached
            self._access_order.append(path)
            self._memory_used += memory_needed
    
    def _remove_entry(self, path: str) -> None:
        """Remove an entry from cache."""
        if path in self._cache:
            self._memory_used -= self._cache[path].size
            del self._cache[path]
        if path in self._access_order:
            self._access_order.remove(path)
    
    def invalidate(self, path: str) -> None:
        """Invalidate a specific file."""
        with self._lock:
            self._remove_entry(path)
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate files matching a pattern.
        
        Args:
            pattern: Prefix to match against paths.
            
        Returns:
            Number of entries invalidated.
        """
        count = 0
        with self._lock:
            to_remove = [p for p in self._cache if p.startswith(pattern)]
            for path in to_remove:
                self._remove_entry(path)
                count += 1
        return count
    
    def clear(self) -> None:
        """Clear the entire cache."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._memory_used = 0
    
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_hits = sum(c.hits for c in self._cache.values())
            return {
                "entries": len(self._cache),
                "memory_mb": self._memory_used / (1024 * 1024),
                "max_memory_mb": self.max_memory_mb,
                "total_hits": total_hits,
            }


# Global file cache
_file_cache: SmartFileCache | None = None


def get_file_cache() -> SmartFileCache:
    """Get the global file cache."""
    global _file_cache
    if _file_cache is None:
        _file_cache = SmartFileCache()
    return _file_cache


# ============================================================================
# SMART FILE READING
# ============================================================================

def smart_read_file(
    path: str,
    max_bytes: int = 120_000,
    use_cache: bool = True,
) -> str | None:
    """Smart file reading with caching.
    
    Args:
        path: Path to file.
        max_bytes: Maximum bytes to read.
        use_cache: Whether to use caching.
        
    Returns:
        File content or None if error.
    """
    abs_path = os.path.abspath(path)
    
    # Try cache first
    if use_cache:
        cache = get_file_cache()
        cached = cache.get(abs_path)
        if cached is not None:
            return cached[:max_bytes] if len(cached) > max_bytes else cached
    
    # Read from disk
    try:
        with open(abs_path, encoding="utf-8", errors="ignore") as f:
            content = f.read(max_bytes)
        
        # Cache the full content
        if use_cache:
            cache.put(abs_path, content)
        
        return content
    except Exception:
        return None


def smart_read_multiple(
    paths: list[str],
    max_bytes_per_file: int = 60_000,
    use_cache: bool = True,
) -> dict[str, str | None]:
    """Read multiple files efficiently.
    
    Args:
        paths: List of file paths.
        max_bytes_per_file: Maximum bytes per file.
        use_cache: Whether to use caching.
        
    Returns:
        Dict mapping paths to contents (or None if error).
    """
    results = {}
    
    for path in paths:
        results[path] = smart_read_file(path, max_bytes_per_file, use_cache)
    
    return results


# ============================================================================
# GIT-AWARE FILE TRACKING
# ============================================================================

@dataclass
class GitFileTracker:
    """Track file changes via git for smart invalidation."""
    
    repo_dir: str
    
    _known_files: dict[str, str] = field(default_factory=dict)  # path -> hash
    _modified_files: set[str] = field(default_factory=set)
    
    def __post_init__(self):
        self._known_files = {}
        self._modified_files = set()
    
    def scan_status(self) -> set[str]:
        """Get list of modified files from git status.
        
        Returns:
            Set of modified file paths.
        """
        import subprocess
        
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain=v1"],
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                timeout=30, check=False,
            )
            
            if result.returncode != 0:
                return set()
            
            modified = set()
            for line in result.stdout.split("\n"):
                if len(line) > 3:
                    filepath = line[3:].strip()
                    if filepath:
                        modified.add(filepath)
            
            self._modified_files = modified
            return modified
        
        except Exception:
            return set()
    
    def get_file_hash(self, path: str) -> str | None:
        """Get git hash for a file.
        
        Args:
            path: Path to file relative to repo.
            
        Returns:
            Git hash or None.
        """
        import subprocess
        
        try:
            result = subprocess.run(
                ["git", "hash-object", path],
                cwd=self.repo_dir,
                capture_output=True,
                text=True,
                timeout=10, check=False,
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        
        except Exception:
            return None
    
    def invalidate_modified(self, cache: SmartFileCache) -> int:
        """Invalidate cache entries for modified files.
        
        Args:
            cache: The file cache to invalidate.
            
        Returns:
            Number of invalidated entries.
        """
        modified = self.scan_status()
        count = 0
        
        for filepath in modified:
            full_path = os.path.join(self.repo_dir, filepath)
            cache.invalidate(full_path)
            count += 1
        
        return count


# ============================================================================
# DIFF-BASED READING
# ============================================================================

def read_files_from_diff(
    diff: str,
    repo_dir: str,
    context_lines: int = 10,
    max_bytes: int = 60_000,
) -> dict[str, str]:
    """Read only the files mentioned in a diff.
    
    Args:
        diff: The git diff.
        repo_dir: Repository root directory.
        context_lines: Extra lines of context to include.
        max_bytes: Maximum bytes per file.
        
    Returns:
        Dict mapping file paths to contents.
    """
    import re
    
    # Extract file paths from diff
    files = set()
    for line in diff.split("\n"):
        if line.startswith("diff --git"):
            match = re.search(r"diff --git a/(.+?) b/(.+)", line)
            if match:
                files.add(match.group(2))
        elif line.startswith("---"):
            if not line.startswith("--- /dev/null"):
                path = line[4:].strip()
                if path.startswith("a/"):
                    path = path[2:]
                files.add(path)
        elif line.startswith("+++"):
            if not line.startswith("+++ /dev/null"):
                path = line[4:].strip()
                if path.startswith("b/"):
                    path = path[2:]
                files.add(path)
    
    # Read files
    results = {}
    for filepath in files:
        full_path = os.path.join(repo_dir, filepath)
        content = smart_read_file(full_path, max_bytes)
        if content is not None:
            results[filepath] = content
    
    return results


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def clear_all_caches() -> None:
    """Clear all file caches."""
    global _file_cache
    if _file_cache:
        _file_cache.clear()


def get_cache_stats() -> dict[str, Any]:
    """Get statistics for all caches."""
    cache = get_file_cache()
    return {
        "file_cache": cache.stats(),
    }
