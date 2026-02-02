"""Database connection pooling for SQLite.

Provides thread-safe connection pooling to improve database performance
by reusing connections instead of creating new ones for each operation.

Example:
    from rfsn_controller.db_pool import get_pool
    
    pool = get_pool("cache.db", pool_size=5)
    
    with pool.connection() as conn:
        cursor = conn.execute("SELECT * FROM cache WHERE key=?", (key,))
        result = cursor.fetchone()
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from queue import Empty, Queue

__all__ = ["SQLiteConnectionPool", "get_pool", "close_all_pools"]


class SQLiteConnectionPool:
    """Thread-safe connection pool for SQLite.
    
    Maintains a pool of reusable database connections to improve performance
    and limit concurrent connections.
    
    Attributes:
        db_path: Path to the SQLite database file
        pool_size: Maximum number of connections in the pool
        timeout: Timeout in seconds for getting a connection
    
    Example:
        pool = SQLiteConnectionPool("cache.db", pool_size=5)
        
        with pool.connection() as conn:
            cursor = conn.execute("SELECT * FROM cache WHERE key=?", (key,))
            result = cursor.fetchone()
    """
    
    def __init__(
        self,
        db_path: str,
        pool_size: int = 5,
        timeout: float = 5.0,
        check_same_thread: bool = False,
    ):
        """Initialize connection pool.
        
        Args:
            db_path: Path to SQLite database
            pool_size: Number of connections to pool
            timeout: Timeout for getting connection from pool (seconds)
            check_same_thread: SQLite check_same_thread parameter
        """
        self.db_path = db_path
        self.pool_size = pool_size
        self.timeout = timeout
        self.check_same_thread = check_same_thread
        
        self._pool: Queue[sqlite3.Connection] = Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._closed = False
        
        # Create initial connections
        for _ in range(pool_size):
            conn = self._create_connection()
            self._pool.put(conn)
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection with optimal settings.
        
        Returns:
            Configured SQLite connection
        """
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=self.check_same_thread,
            timeout=self.timeout,
        )
        
        # Optimize SQLite settings for performance
        conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
        conn.execute("PRAGMA synchronous=NORMAL")  # Balance safety/speed
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        conn.execute("PRAGMA temp_store=MEMORY")  # Temp tables in memory
        
        # Enable dict-like row access
        conn.row_factory = sqlite3.Row
        
        return conn
    
    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Get a connection from the pool.
        
        This is a context manager that automatically returns the connection
        to the pool after use.
        
        Yields:
            Database connection (automatically returned to pool)
            
        Raises:
            RuntimeError: If pool is closed or connection cannot be obtained
        
        Example:
            with pool.connection() as conn:
                cursor = conn.execute("SELECT * FROM users")
                for row in cursor:
                    print(dict(row))
        """
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        
        try:
            conn = self._pool.get(timeout=self.timeout)
        except Empty:
            raise RuntimeError(
                f"Could not get connection from pool (timeout={self.timeout}s). "
                f"Pool size: {self.pool_size}, all connections may be in use."
            )
        
        try:
            yield conn
        finally:
            # Always return connection to pool
            if not self._closed:
                self._pool.put(conn)
    
    def close_all(self) -> None:
        """Close all connections in the pool.
        
        Should be called during application shutdown to properly
        close all database connections.
        
        Example:
            pool.close_all()
        """
        with self._lock:
            if self._closed:
                return
            
            self._closed = True
            
            # Close all connections
            while not self._pool.empty():
                try:
                    conn = self._pool.get_nowait()
                    conn.close()
                except Empty:
                    break
    
    def __enter__(self) -> SQLiteConnectionPool:
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - close all connections."""
        self.close_all()
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SQLiteConnectionPool(db_path={self.db_path!r}, "
            f"pool_size={self.pool_size}, closed={self._closed})"
        )


# =============================================================================
# Global Pool Management
# =============================================================================

# Global registry of pools (per database path)
_pools: dict[str, SQLiteConnectionPool] = {}
_pools_lock = threading.Lock()


def get_pool(
    db_path: str,
    pool_size: int = 5,
    timeout: float = 5.0,
) -> SQLiteConnectionPool:
    """Get or create a connection pool for a database.
    
    This function maintains a global registry of pools, so calling it
    multiple times with the same db_path returns the same pool instance.
    
    Args:
        db_path: Path to database
        pool_size: Pool size (only used when creating new pool)
        timeout: Connection timeout (only used when creating new pool)
        
    Returns:
        Connection pool instance
        
    Example:
        from rfsn_controller.db_pool import get_pool
        
        # Get pool for cache database
        cache_pool = get_pool("~/.cache/rfsn/cache.db", pool_size=5)
        
        with cache_pool.connection() as conn:
            # Use connection
            pass
    """
    with _pools_lock:
        if db_path not in _pools:
            _pools[db_path] = SQLiteConnectionPool(
                db_path,
                pool_size=pool_size,
                timeout=timeout,
            )
        return _pools[db_path]


def close_all_pools() -> None:
    """Close all connection pools.
    
    Should be called during application shutdown to properly close
    all database connections.
    
    Example:
        from rfsn_controller.db_pool import close_all_pools
        
        # At application shutdown
        close_all_pools()
    """
    with _pools_lock:
        for pool in _pools.values():
            pool.close_all()
        _pools.clear()


# =============================================================================
# Utility Functions
# =============================================================================

def execute_with_retry(
    pool: SQLiteConnectionPool,
    query: str,
    params: tuple = (),
    max_retries: int = 3,
) -> list[sqlite3.Row]:
    """Execute a query with automatic retry on database lock.
    
    Args:
        pool: Connection pool
        query: SQL query
        params: Query parameters
        max_retries: Maximum retry attempts
        
    Returns:
        List of result rows
        
    Raises:
        sqlite3.OperationalError: If all retries exhausted
    """
    for attempt in range(max_retries):
        try:
            with pool.connection() as conn:
                cursor = conn.execute(query, params)
                return cursor.fetchall()
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                # Retry after brief pause
                import time
                time.sleep(0.1 * (attempt + 1))
                continue
            raise
    
    raise sqlite3.OperationalError(
        f"Query failed after {max_retries} attempts: {query}"
    )


def get_pool_stats(pool: SQLiteConnectionPool) -> dict:
    """Get statistics about a connection pool.
    
    Args:
        pool: Connection pool to inspect
        
    Returns:
        Dictionary with pool statistics
        
    Example:
        stats = get_pool_stats(pool)
        print(f"Available connections: {stats['available']}")
    """
    return {
        "db_path": pool.db_path,
        "pool_size": pool.pool_size,
        "available": pool._pool.qsize(),
        "in_use": pool.pool_size - pool._pool.qsize(),
        "closed": pool._closed,
    }
