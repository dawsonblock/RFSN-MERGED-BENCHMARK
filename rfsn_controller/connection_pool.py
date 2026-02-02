"""Database connection pooling for RFSN Controller.

Provides efficient database connection management with connection pooling,
reducing overhead and improving performance for database-heavy workloads.

Usage:
    from rfsn_controller.connection_pool import ConnectionPool
    
    # Create pool
    pool = ConnectionPool("cache.db", pool_size=5)
    
    # Use connection
    with pool.get_connection() as conn:
        cursor = conn.execute("SELECT * FROM cache WHERE key=?", (key,))
        result = cursor.fetchone()
"""

from __future__ import annotations

import atexit
import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from queue import Empty, Queue


class ConnectionPool:
    """Thread-safe SQLite connection pool.
    
    Manages a pool of reusable database connections to reduce connection
    overhead and improve performance.
    
    Attributes:
        db_path: Path to SQLite database file
        pool_size: Maximum number of connections in pool
        timeout: Timeout in seconds for getting a connection
    
    Example:
        >>> pool = ConnectionPool("cache.db", pool_size=5)
        >>> with pool.get_connection() as conn:
        ...     result = conn.execute("SELECT * FROM items").fetchall()
    """
    
    def __init__(
        self,
        db_path: str | Path,
        pool_size: int = 5,
        timeout: float = 30.0,
        check_same_thread: bool = False,
        **kwargs
    ):
        """Initialize connection pool.
        
        Args:
            db_path: Path to SQLite database
            pool_size: Maximum connections in pool (default: 5)
            timeout: Timeout for getting connection (default: 30.0s)
            check_same_thread: SQLite thread checking (default: False)
            **kwargs: Additional arguments passed to sqlite3.connect()
            
        Raises:
            ValueError: If pool_size is less than 1
        """
        if pool_size < 1:
            raise ValueError(f"pool_size must be at least 1, got {pool_size}")
        
        self.db_path = str(db_path)
        self.pool_size = pool_size
        self.timeout = timeout
        self.check_same_thread = check_same_thread
        self.kwargs = kwargs
        
        # Thread-safe queue for connection pool
        self.pool: Queue = Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._closed = False
        
        # Pre-populate pool
        for _ in range(pool_size):
            conn = self._create_connection()
            self.pool.put(conn)
        
        # Register cleanup on exit
        atexit.register(self.close_all)
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection.
        
        Returns:
            New SQLite connection
        """
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=self.check_same_thread,
            **self.kwargs
        )
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON")
        # Use WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode = WAL")
        return conn
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection]:
        """Get a connection from the pool.
        
        Yields:
            SQLite connection from pool
            
        Raises:
            RuntimeError: If pool is closed
            Empty: If no connection available within timeout
            
        Example:
            >>> with pool.get_connection() as conn:
            ...     conn.execute("INSERT INTO items VALUES (?, ?)", (1, "test"))
            ...     conn.commit()
        """
        if self._closed:
            raise RuntimeError("Connection pool is closed")
        
        conn: sqlite3.Connection | None = None
        try:
            # Get connection from pool
            conn = self.pool.get(timeout=self.timeout)
            yield conn
        except Empty:
            raise TimeoutError(
                f"Failed to get connection from pool within {self.timeout}s"
            )
        finally:
            # Return connection to pool
            if conn is not None:
                if self._closed:
                    # Pool closed, close connection
                    conn.close()
                else:
                    # Return to pool
                    try:
                        self.pool.put(conn, block=False)
                    except Exception:
                        # Pool full, close connection
                        conn.close()
    
    def close_all(self) -> None:
        """Close all connections in the pool.
        
        Called automatically on program exit via atexit.
        """
        with self._lock:
            if self._closed:
                return
            
            self._closed = True
            
            # Close all connections in pool
            while not self.pool.empty():
                try:
                    conn = self.pool.get(block=False)
                    conn.close()
                except Empty:
                    break
    
    def __enter__(self) -> ConnectionPool:
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - close all connections."""
        self.close_all()
    
    @property
    def is_closed(self) -> bool:
        """Check if pool is closed."""
        return self._closed
    
    @property
    def available_connections(self) -> int:
        """Get number of available connections in pool."""
        return self.pool.qsize()
