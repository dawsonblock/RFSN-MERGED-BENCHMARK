"""Storage backends for publishing artifacts.

Provides flexible storage backends for publishing build artifacts,
evidence packs, and other outputs to local filesystem or S3.

Supports:
- Local filesystem storage
- AWS S3 storage
- Extensible backend system

Example:
    >>> store = make_store("local", local_dir="/tmp/output")
    >>> result = store.put_dir("/path/to/artifacts", "build-123")
    >>> print(result.destination)
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass


class StorageError(RuntimeError):
    """Exception raised for storage-related errors."""
    pass


@dataclass
class PublishResult:
    """Result of a storage publish operation.
    
    Attributes:
        backend: Storage backend used ("local" or "s3")
        destination: Full path/URL to published artifacts
    """
    backend: str
    destination: str


class BaseStore:
    """Base class for storage backends.
    
    Subclasses must implement put_dir() to handle directory publishing.
    """
    
    def put_dir(self, src_dir: str, dest: str) -> PublishResult:
        """Publish a directory to storage.
        
        Args:
            src_dir: Source directory path to publish
            dest: Destination identifier (path or key prefix)
            
        Returns:
            PublishResult with backend info and destination
            
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError


class LocalStore(BaseStore):
    """Local filesystem storage backend.
    
    Copies directories to the local filesystem, preserving structure
    and file attributes.
    """
    
    def put_dir(self, src_dir: str, dest: str) -> PublishResult:
        """Copy directory to local destination.
        
        Args:
            src_dir: Source directory to copy
            dest: Destination directory path
            
        Returns:
            PublishResult with absolute destination path
        """
        os.makedirs(dest, exist_ok=True)
        for root, dirs, files in os.walk(src_dir):
            rel_root = os.path.relpath(root, src_dir)
            out_root = os.path.join(dest, rel_root) if rel_root != "." else dest
            os.makedirs(out_root, exist_ok=True)
            for d in dirs:
                os.makedirs(os.path.join(out_root, d), exist_ok=True)
            for f in files:
                shutil.copy2(os.path.join(root, f), os.path.join(out_root, f))
        return PublishResult(backend="local", destination=os.path.abspath(dest))


class S3Store(BaseStore):
    """AWS S3 storage backend.
    
    Uploads directories to S3, flattening paths and preserving files.
    Requires boto3 to be installed.
    
    Attributes:
        bucket: S3 bucket name
        prefix: Key prefix (directory path in bucket)
    """
    
    def __init__(self, bucket: str, prefix: str):
        """Initialize S3 store.
        
        Args:
            bucket: S3 bucket name
            prefix: Key prefix for uploaded files
        """
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    def put_dir(self, src_dir: str, dest: str) -> PublishResult:
        """Upload directory to S3.
        
        Args:
            src_dir: Source directory to upload
            dest: Key suffix to append to prefix
            
        Returns:
            PublishResult with S3 URI
            
        Raises:
            StorageError: If boto3 not installed or upload fails
        """
        try:
            import boto3  # type: ignore
        except Exception as e:
            raise StorageError("boto3 is required for S3 publishing. pip install boto3") from e

        s3 = boto3.client("s3")
        base = f"{self.prefix}/{dest}".strip("/")
        for root, _, files in os.walk(src_dir):
            rel_root = os.path.relpath(root, src_dir)
            for f in files:
                local_path = os.path.join(root, f)
                key = f"{base}/{rel_root}/{f}".replace("\\", "/").replace("/./", "/")
                s3.upload_file(local_path, self.bucket, key)
        return PublishResult(backend="s3", destination=f"s3://{self.bucket}/{base}")


def make_store(
    backend: str,
    *,
    local_dir: str | None = None,
    s3_bucket: str | None = None,
    s3_prefix: str | None = None,
) -> BaseStore:
    """Factory function to create storage backend.
    
    Args:
        backend: Backend type ("local" or "s3")
        local_dir: Directory for local backend (required for local)
        s3_bucket: S3 bucket name (required for S3)
        s3_prefix: S3 key prefix (required for S3)
        
    Returns:
        Configured storage backend instance
        
    Raises:
        StorageError: If backend unknown or required args missing
        
    Example:
        >>> store = make_store("local", local_dir="/tmp/artifacts")
        >>> store = make_store("s3", s3_bucket="my-bucket", s3_prefix="builds/")
    """
    backend = (backend or "local").lower()
    if backend == "local":
        if not local_dir:
            raise StorageError("local_dir required for local publishing")
        return LocalStore()
    if backend == "s3":
        if not s3_bucket or not s3_prefix:
            raise StorageError("s3_bucket and s3_prefix required for s3 publishing")
        return S3Store(bucket=s3_bucket, prefix=s3_prefix)
    raise StorageError(f"Unknown publish backend: {backend}")
