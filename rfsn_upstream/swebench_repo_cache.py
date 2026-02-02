"""SWE-bench Repository Cache.

Clones and caches SWE-bench repositories locally for fast access.
Each repository is stored at a specific commit to ensure reproducibility.

INVARIANTS:
1. Cached repos are never modified (read-only after clone)
2. Each (repo, commit) pair is stored independently
3. Cache is persistent across runs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any


@dataclass
class CachedRepo:
    """Handle to a cached repository.
    
    Attributes:
        path: Absolute path to the cached repo
        repo_url: Original repository URL
        commit: Commit SHA the repo is at
        cached_at: Timestamp when cached (ISO format)
    """
    path: Path
    repo_url: str
    commit: str
    cached_at: str = ""


@dataclass
class RepoCacheConfig:
    """Configuration for the repo cache.
    
    Attributes:
        cache_dir: Base directory for cached repos
        shallow: Whether to use shallow clones
        verify_commits: Whether to verify commit after clone
    """
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".rfsn" / "swebench_repos")
    shallow: bool = True
    verify_commits: bool = True


class RepoCache:
    """Manages cached SWE-bench repositories."""
    
    def __init__(self, config: RepoCacheConfig | None = None):
        self.config = config or RepoCacheConfig()
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_index: dict[str, CachedRepo] = {}
        self._load_index()
    
    def _cache_key(self, repo_url: str, commit: str) -> str:
        """Generate a unique cache key for a (repo, commit) pair."""
        # Normalize URL
        url = repo_url.rstrip("/").lower()
        if url.endswith(".git"):
            url = url[:-4]
        # Create hash of url + commit
        key_str = f"{url}@{commit}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def _repo_dir(self, key: str) -> Path:
        """Get the directory path for a cache key."""
        return self.config.cache_dir / key
    
    def _load_index(self) -> None:
        """Load the cache index from disk."""
        index_path = self.config.cache_dir / "index.json"
        if index_path.exists():
            with open(index_path) as f:
                data = json.load(f)
                for key, entry in data.items():
                    self._cache_index[key] = CachedRepo(
                        path=Path(entry["path"]),
                        repo_url=entry["repo_url"],
                        commit=entry["commit"],
                        cached_at=entry.get("cached_at", ""),
                    )
    
    def _save_index(self) -> None:
        """Save the cache index to disk."""
        
        index_path = self.config.cache_dir / "index.json"
        data = {}
        for key, repo in self._cache_index.items():
            data[key] = {
                "path": str(repo.path),
                "repo_url": repo.repo_url,
                "commit": repo.commit,
                "cached_at": repo.cached_at,
            }
        with open(index_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def get(self, repo_url: str, commit: str) -> CachedRepo | None:
        """Get a cached repository if it exists.
        
        Args:
            repo_url: Repository URL
            commit: Commit SHA
            
        Returns:
            CachedRepo if found, None otherwise
        """
        key = self._cache_key(repo_url, commit)
        cached = self._cache_index.get(key)
        if cached and cached.path.exists():
            return cached
        return None
    
    def get_or_clone(self, repo_url: str, commit: str) -> CachedRepo:
        """Get a cached repository, cloning if necessary.
        
        Args:
            repo_url: Repository URL (GitHub)
            commit: Commit SHA to checkout
            
        Returns:
            CachedRepo with path to the repository
            
        Raises:
            RuntimeError: If clone or checkout fails
        """
        # Check cache first
        cached = self.get(repo_url, commit)
        if cached:
            return cached
        
        # Clone fresh
        key = self._cache_key(repo_url, commit)
        target_dir = self._repo_dir(key)
        
        # Clean up any partial clone
        if target_dir.exists():
            shutil.rmtree(target_dir)
        
        # Build clone command
        clone_cmd = ["git", "clone"]
        if self.config.shallow:
            clone_cmd.extend(["--depth", "1", "--no-single-branch"])
        clone_cmd.extend([repo_url, str(target_dir)])
        
        result = subprocess.run(
            clone_cmd,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Clone failed: {result.stderr}")
        
        # Checkout specific commit
        result = subprocess.run(
            ["git", "checkout", commit],
            cwd=target_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            # For shallow clones, we may need to fetch the commit first
            subprocess.run(
                ["git", "fetch", "--depth", "1", "origin", commit],
                cwd=target_dir,
                capture_output=True,
                check=False,
            )
            result = subprocess.run(
                ["git", "checkout", commit],
                cwd=target_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                shutil.rmtree(target_dir)
                raise RuntimeError(f"Checkout failed: {result.stderr}")
        
        # Verify commit if configured
        if self.config.verify_commits:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=target_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            actual_commit = result.stdout.strip()
            if not actual_commit.startswith(commit) and not commit.startswith(actual_commit):
                shutil.rmtree(target_dir)
                raise RuntimeError(f"Commit mismatch: expected {commit}, got {actual_commit}")
        
        # Add to index
        cached = CachedRepo(
            path=target_dir,
            repo_url=repo_url,
            commit=commit,
            cached_at=datetime.now(UTC).isoformat(),
        )
        self._cache_index[key] = cached
        self._save_index()
        
        return cached
    
    def ensure_all(self, tasks: list[dict[str, Any]]) -> dict[str, CachedRepo]:
        """Ensure all repositories for a list of tasks are cached.
        
        Args:
            tasks: List of task dicts with 'repo' and 'base_commit' keys
            
        Returns:
            Dict mapping instance_id -> CachedRepo
        """
        results = {}
        for task in tasks:
            repo_url = task.get("repo", "")
            commit = task.get("base_commit", "HEAD")
            instance_id = task.get("instance_id", "")
            
            if not repo_url:
                continue
            
            try:
                cached = self.get_or_clone(repo_url, commit)
                results[instance_id] = cached
            except Exception as e:
                print(f"Failed to cache {instance_id}: {e}")
        
        return results
    
    def clear(self) -> int:
        """Clear all cached repositories.
        
        Returns:
            Number of repos cleared
        """
        count = len(self._cache_index)
        for repo in self._cache_index.values():
            if repo.path.exists():
                shutil.rmtree(repo.path, ignore_errors=True)
        self._cache_index.clear()
        self._save_index()
        return count
    
    def stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dict with cache stats
        """
        total_size = 0
        for repo in self._cache_index.values():
            if repo.path.exists():
                for path in repo.path.rglob("*"):
                    if path.is_file():
                        total_size += path.stat().st_size
        
        return {
            "repo_count": len(self._cache_index),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.config.cache_dir),
        }


def main() -> None:
    """CLI for managing the repo cache."""
    parser = argparse.ArgumentParser(description="SWE-bench repo cache manager")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Clone command
    clone_parser = subparsers.add_parser("clone", help="Clone a repository")
    clone_parser.add_argument("repo_url", help="Repository URL")
    clone_parser.add_argument("commit", help="Commit SHA")
    
    # Stats command
    subparsers.add_parser("stats", help="Show cache statistics")
    
    # Clear command
    subparsers.add_parser("clear", help="Clear all cached repos")
    
    # Ensure command
    ensure_parser = subparsers.add_parser("ensure", help="Cache all repos from tasks file")
    ensure_parser.add_argument("tasks_file", type=Path, help="Path to tasks.jsonl")
    
    args = parser.parse_args()
    cache = RepoCache()
    
    if args.command == "clone":
        cached = cache.get_or_clone(args.repo_url, args.commit)
        print(f"Cached at: {cached.path}")
    
    elif args.command == "stats":
        stats = cache.stats()
        print(f"Cached repos: {stats['repo_count']}")
        print(f"Total size: {stats['total_size_mb']:.1f} MB")
        print(f"Cache dir: {stats['cache_dir']}")
    
    elif args.command == "clear":
        count = cache.clear()
        print(f"Cleared {count} cached repos")
    
    elif args.command == "ensure":
        tasks = []
        with open(args.tasks_file) as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line))
        results = cache.ensure_all(tasks)
        print(f"Cached {len(results)} repos")


if __name__ == "__main__":
    main()
