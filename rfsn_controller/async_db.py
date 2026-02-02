"""
Async database operations using aiosqlite.

This module provides async wrappers for all database operations,
replacing synchronous sqlite3 calls to eliminate I/O blocking.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite

from .structured_logging import get_logger

logger = get_logger(__name__)


class AsyncConnectionPool:
    """
    Async-compatible database connection pool.
    
    Manages a pool of aiosqlite connections for high-throughput
    async database operations without blocking the event loop.
    
    Example:
        >>> pool = AsyncConnectionPool("cache.db", max_connections=5)
        >>> await pool.initialize()
        >>> async with pool.acquire() as conn:
        ...     async with conn.execute("SELECT * FROM cache") as cursor:
        ...         rows = await cursor.fetchall()
        >>> await pool.close()
    """
    
    def __init__(
        self,
        db_path: str | Path,
        max_connections: int = 5,
        timeout: float = 30.0
    ):
        """
        Initialize the async connection pool.
        
        Args:
            db_path: Path to SQLite database file
            max_connections: Maximum number of pooled connections
            timeout: Connection acquisition timeout in seconds
        """
        self.db_path = Path(db_path)
        self.max_connections = max_connections
        self.timeout = timeout
        self._pool: list[aiosqlite.Connection] = []
        self._in_use: set[aiosqlite.Connection] = set()
        self._initialized = False
        
    async def initialize(self):
        """Initialize the connection pool."""
        if self._initialized:
            return
            
        logger.info(
            "Initializing async connection pool",
            db_path=str(self.db_path),
            max_connections=self.max_connections
        )
        
        # Create initial connections
        for _ in range(self.max_connections):
            conn = await aiosqlite.connect(str(self.db_path), timeout=self.timeout)
            # Enable WAL mode for better concurrency
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA synchronous=NORMAL")
            await conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            self._pool.append(conn)
            
        self._initialized = True
        logger.info("Async connection pool initialized")
    
    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[aiosqlite.Connection, None]:
        """
        Acquire a connection from the pool.
        
        Yields:
            Database connection from pool
            
        Example:
            >>> async with pool.acquire() as conn:
            ...     await conn.execute("INSERT INTO cache VALUES (?, ?)", (key, value))
            ...     await conn.commit()
        """
        if not self._initialized:
            await self.initialize()
        
        # Wait for available connection
        import asyncio
        start_time = asyncio.get_event_loop().time()
        
        while True:
            # Try to get connection from pool
            if self._pool:
                conn = self._pool.pop()
                self._in_use.add(conn)
                try:
                    yield conn
                finally:
                    # Return connection to pool
                    self._in_use.remove(conn)
                    self._pool.append(conn)
                break
            
            # Check timeout
            if asyncio.get_event_loop().time() - start_time > self.timeout:
                raise TimeoutError(
                    f"Could not acquire connection within {self.timeout}s"
                )
            
            # Wait briefly and retry
            await asyncio.sleep(0.01)
    
    async def execute(
        self,
        query: str,
        parameters: tuple[Any, ...] = ()
    ) -> list[tuple[Any, ...]]:
        """
        Execute a query and return all results.
        
        Args:
            query: SQL query to execute
            parameters: Query parameters
            
        Returns:
            List of result tuples
        """
        async with self.acquire() as conn:
            async with conn.execute(query, parameters) as cursor:
                return await cursor.fetchall()
    
    async def execute_many(
        self,
        query: str,
        parameters: list[tuple[Any, ...]]
    ):
        """
        Execute a query with multiple parameter sets.
        
        Args:
            query: SQL query to execute
            parameters: List of parameter tuples
        """
        async with self.acquire() as conn:
            await conn.executemany(query, parameters)
            await conn.commit()
    
    async def close(self):
        """Close all connections in the pool."""
        if not self._initialized:
            return
            
        logger.info("Closing async connection pool")
        
        # Close all connections
        for conn in self._pool:
            await conn.close()
        for conn in self._in_use:
            await conn.close()
            
        self._pool.clear()
        self._in_use.clear()
        self._initialized = False
        
        logger.info("Async connection pool closed")
    
    async def stats(self) -> dict[str, Any]:
        """
        Get pool statistics.
        
        Returns:
            Dictionary with pool stats
        """
        return {
            "max_size": self.max_connections,
            "available": len(self._pool),
            "in_use": len(self._in_use),
            "initialized": self._initialized
        }


class AsyncCache:
    """
    Async-compatible cache using aiosqlite.
    
    Drop-in async replacement for synchronous cache operations.
    """
    
    def __init__(self, pool: AsyncConnectionPool):
        """
        Initialize async cache.
        
        Args:
            pool: AsyncConnectionPool instance
        """
        self.pool = pool
        
    async def get(self, key: str) -> Any | None:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        results = await self.pool.execute(
            "SELECT value FROM cache WHERE key = ? AND expires_at > ?",
            (key, self._now())
        )
        
        if results:
            import json
            return json.loads(results[0][0])
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int = 3600
    ):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds
        """
        import json
        import time
        
        expires_at = time.time() + ttl_seconds
        value_json = json.dumps(value)
        
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO cache (key, value, expires_at)
                VALUES (?, ?, ?)
                """,
                (key, value_json, expires_at)
            )
            await conn.commit()
    
    async def delete(self, key: str):
        """Delete key from cache."""
        async with self.pool.acquire() as conn:
            await conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            await conn.commit()
    
    async def clear_expired(self):
        """Clear all expired entries."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM cache WHERE expires_at <= ?",
                (self._now(),)
            )
            await conn.commit()
    
    def _now(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()
