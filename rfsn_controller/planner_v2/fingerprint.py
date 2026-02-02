"""Fingerprint - Repo state identification for memory reuse.

Creates unique fingerprints for repo state to ensure memory reuse
is only applied to matching states, preventing overfitting.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class RepoFingerprint:
    """Unique identifier for repo state."""
    
    file_list_hash: str
    lockfile_hashes: dict[str, str] = field(default_factory=dict)
    git_commit: str | None = None
    git_branch: str | None = None
    timestamp: str = ""
    file_count: int = 0
    
    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()
    
    def to_dict(self) -> dict:
        return {
            "file_list_hash": self.file_list_hash,
            "lockfile_hashes": self.lockfile_hashes,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "timestamp": self.timestamp,
            "file_count": self.file_count,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> RepoFingerprint:
        return cls(
            file_list_hash=data["file_list_hash"],
            lockfile_hashes=data.get("lockfile_hashes", {}),
            git_commit=data.get("git_commit"),
            git_branch=data.get("git_branch"),
            timestamp=data.get("timestamp", ""),
            file_count=data.get("file_count", 0),
        )
    
    def matches(self, other: RepoFingerprint, tolerance: float = 0.9) -> bool:
        """Check if fingerprints are similar enough for memory reuse.
        
        Args:
            other: Fingerprint to compare.
            tolerance: Similarity threshold (0.0 to 1.0).
            
        Returns:
            True if fingerprints match within tolerance.
        """
        # Exact match on file list hash is required
        if self.file_list_hash != other.file_list_hash:
            return False
        
        # Check lockfile overlap
        if self.lockfile_hashes and other.lockfile_hashes:
            common_locks = set(self.lockfile_hashes.keys()) & set(other.lockfile_hashes.keys())
            if common_locks:
                matching = sum(
                    1 for k in common_locks
                    if self.lockfile_hashes[k] == other.lockfile_hashes[k]
                )
                match_ratio = matching / len(common_locks)
                if match_ratio < tolerance:
                    return False
        
        return True
    
    def to_hash(self) -> str:
        """Generate a single hash representing this fingerprint."""
        combined = f"{self.file_list_hash}"
        for k in sorted(self.lockfile_hashes.keys()):
            combined += f":{k}={self.lockfile_hashes[k]}"
        if self.git_commit:
            combined += f":commit={self.git_commit}"
        return hashlib.sha256(combined.encode()).hexdigest()[:24]


# Known lockfile names
LOCKFILES = [
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
    "requirements.txt",
    "requirements-dev.txt",
    "Cargo.lock",
    "Gemfile.lock",
    "go.sum",
    "composer.lock",
]


def compute_fingerprint(repo_dir: Path) -> RepoFingerprint:
    """Compute fingerprint for repo state.
    
    Args:
        repo_dir: Path to repository root.
        
    Returns:
        RepoFingerprint for current state.
    """
    repo_path = Path(repo_dir)
    
    # Get file list hash
    files = _get_file_list(repo_path)
    file_list_hash = _hash_file_list(files)
    
    # Get lockfile hashes
    lockfile_hashes = {}
    for lockfile in LOCKFILES:
        lockfile_path = repo_path / lockfile
        if lockfile_path.exists():
            lockfile_hashes[lockfile] = _hash_file(lockfile_path)
    
    # Get git info
    git_commit = _get_git_commit(repo_path)
    git_branch = _get_git_branch(repo_path)
    
    return RepoFingerprint(
        file_list_hash=file_list_hash,
        lockfile_hashes=lockfile_hashes,
        git_commit=git_commit,
        git_branch=git_branch,
        file_count=len(files),
    )


def _get_file_list(repo_path: Path) -> list[str]:
    """Get sorted list of files in repo."""
    files = []
    for f in repo_path.rglob("*"):
        if f.is_file():
            # Exclude common non-source directories
            rel = f.relative_to(repo_path)
            parts = rel.parts
            if any(p.startswith(".") or p in ("node_modules", "__pycache__", "venv", ".git") for p in parts):
                continue
            files.append(str(rel))
    return sorted(files)


def _hash_file_list(files: list[str]) -> str:
    """Hash a list of filenames."""
    combined = "\n".join(files)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def _hash_file(filepath: Path) -> str:
    """Hash file contents."""
    try:
        content = filepath.read_bytes()
        return hashlib.sha256(content).hexdigest()[:16]
    except OSError:
        return "error"


def _get_git_commit(repo_path: Path) -> str | None:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5, check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def _get_git_branch(repo_path: Path) -> str | None:
    """Get current git branch."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5, check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None
