"""Batch file operations optimizer.

Batches multiple file operations to reduce I/O overhead.
Useful for reading multiple files, checking multiple paths, etc.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from pathlib import Path


class BatchFileReader:
    """Batch file reading with parallel execution."""
    
    def __init__(self, max_workers: int = 4):
        """Initialize batch reader.
        
        Args:
            max_workers: Maximum parallel workers
        """
        self.max_workers = max_workers
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
    
    def __enter__(self):
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        return self
    
    def __exit__(self, *args):
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
    
    def read_files(self, filepaths: list[str | Path]) -> dict[str, str | None]:
        """Read multiple files in parallel.
        
        Args:
            filepaths: List of file paths to read
            
        Returns:
            Dictionary mapping filepath to content (or None if error)
        """
        if not self._executor:
            raise RuntimeError("BatchFileReader must be used as context manager")
        
        def read_single(filepath: str | Path) -> tuple[str, str | None]:
            """Read single file."""
            filepath = str(filepath)
            try:
                return filepath, Path(filepath).read_text(encoding='utf-8')
            except (FileNotFoundError, PermissionError, UnicodeDecodeError):
                return filepath, None
        
        # Submit all reads
        futures = {
            self._executor.submit(read_single, fp): fp 
            for fp in filepaths
        }
        
        # Collect results
        results = {}
        for future in concurrent.futures.as_completed(futures):
            filepath, content = future.result()
            results[filepath] = content
        
        return results
    
    def file_exists_batch(self, filepaths: list[str | Path]) -> dict[str, bool]:
        """Check if multiple files exist in parallel.
        
        Args:
            filepaths: List of file paths to check
            
        Returns:
            Dictionary mapping filepath to existence boolean
        """
        if not self._executor:
            raise RuntimeError("BatchFileReader must be used as context manager")
        
        def check_exists(filepath: str | Path) -> tuple[str, bool]:
            """Check if file exists."""
            filepath = str(filepath)
            return filepath, Path(filepath).exists()
        
        # Submit all checks
        futures = {
            self._executor.submit(check_exists, fp): fp 
            for fp in filepaths
        }
        
        # Collect results
        results = {}
        for future in concurrent.futures.as_completed(futures):
            filepath, exists = future.result()
            results[filepath] = exists
        
        return results


class AsyncBatchFileReader:
    """Async version of batch file reader."""
    
    def __init__(self, max_concurrent: int = 10):
        """Initialize async batch reader.
        
        Args:
            max_concurrent: Maximum concurrent operations
        """
        self.max_concurrent = max_concurrent
        self._semaphore: asyncio.Semaphore | None = None
    
    async def __aenter__(self):
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self
    
    async def __aexit__(self, *args):
        self._semaphore = None
    
    async def read_files(self, filepaths: list[str | Path]) -> dict[str, str | None]:
        """Read multiple files asynchronously.
        
        Args:
            filepaths: List of file paths to read
            
        Returns:
            Dictionary mapping filepath to content (or None if error)
        """
        if not self._semaphore:
            raise RuntimeError("AsyncBatchFileReader must be used as context manager")
        
        async def read_single(filepath: str | Path) -> tuple[str, str | None]:
            """Read single file with semaphore."""
            async with self._semaphore:
                filepath = str(filepath)
                try:
                    # Use asyncio to run in executor
                    loop = asyncio.get_event_loop()
                    content = await loop.run_in_executor(
                        None,
                        lambda: Path(filepath).read_text(encoding='utf-8')
                    )
                    return filepath, content
                except (FileNotFoundError, PermissionError, UnicodeDecodeError):
                    return filepath, None
        
        # Read all files concurrently
        results_list = await asyncio.gather(*[read_single(fp) for fp in filepaths])
        
        # Convert to dict
        return dict(results_list)
    
    async def file_exists_batch(self, filepaths: list[str | Path]) -> dict[str, bool]:
        """Check if multiple files exist asynchronously.
        
        Args:
            filepaths: List of file paths to check
            
        Returns:
            Dictionary mapping filepath to existence boolean
        """
        if not self._semaphore:
            raise RuntimeError("AsyncBatchFileReader must be used as context manager")
        
        async def check_exists(filepath: str | Path) -> tuple[str, bool]:
            """Check if file exists with semaphore."""
            async with self._semaphore:
                filepath = str(filepath)
                # Use asyncio to run in executor
                loop = asyncio.get_event_loop()
                exists = await loop.run_in_executor(
                    None,
                    lambda: Path(filepath).exists()
                )
                return filepath, exists
        
        # Check all files concurrently
        results_list = await asyncio.gather(*[check_exists(fp) for fp in filepaths])
        
        # Convert to dict
        return dict(results_list)


def batch_read_files(filepaths: list[str | Path], max_workers: int = 4) -> dict[str, str | None]:
    """Convenience function to batch read files.
    
    Args:
        filepaths: List of file paths to read
        max_workers: Maximum parallel workers
        
    Returns:
        Dictionary mapping filepath to content
    """
    with BatchFileReader(max_workers=max_workers) as reader:
        return reader.read_files(filepaths)


async def async_batch_read_files(
    filepaths: list[str | Path],
    max_concurrent: int = 10
) -> dict[str, str | None]:
    """Convenience function to batch read files asynchronously.
    
    Args:
        filepaths: List of file paths to read
        max_concurrent: Maximum concurrent operations
        
    Returns:
        Dictionary mapping filepath to content
    """
    async with AsyncBatchFileReader(max_concurrent=max_concurrent) as reader:
        return await reader.read_files(filepaths)
