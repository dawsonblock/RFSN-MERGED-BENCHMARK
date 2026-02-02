"""Tests for database connection pooling."""

import os
import sqlite3
import tempfile
import threading
import time

import pytest

from rfsn_controller.db_pool import (
    SQLiteConnectionPool,
    close_all_pools,
    execute_with_retry,
    get_pool,
    get_pool_stats,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    # Cleanup
    try:
        os.unlink(db_path)
    except OSError:
        pass


@pytest.fixture
def pool(temp_db):
    """Create a connection pool for testing."""
    pool = SQLiteConnectionPool(temp_db, pool_size=3)
    yield pool
    pool.close_all()


class TestSQLiteConnectionPool:
    """Tests for SQLiteConnectionPool class."""
    
    def test_create_pool(self, temp_db):
        """Test creating a connection pool."""
        pool = SQLiteConnectionPool(temp_db, pool_size=5)
        assert pool.db_path == temp_db
        assert pool.pool_size == 5
        pool.close_all()
    
    def test_get_connection(self, pool):
        """Test getting a connection from the pool."""
        with pool.connection() as conn:
            assert isinstance(conn, sqlite3.Connection)
            cursor = conn.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
    
    def test_connection_returned_to_pool(self, pool):
        """Test that connections are returned to the pool after use."""
        initial_available = pool._pool.qsize()
        
        with pool.connection():
            # Connection is checked out
            assert pool._pool.qsize() == initial_available - 1
        
        # Connection returned to pool
        assert pool._pool.qsize() == initial_available
    
    def test_multiple_connections(self, pool):
        """Test using multiple connections concurrently."""
        connections = []
        
        # Get all connections from pool
        for _ in range(pool.pool_size):
            ctx = pool.connection()
            conn = ctx.__enter__()
            connections.append((ctx, conn))
        
        # Pool should be exhausted
        assert pool._pool.qsize() == 0
        
        # Return all connections
        for ctx, conn in connections:
            ctx.__exit__(None, None, None)
        
        # Pool should be full again
        assert pool._pool.qsize() == pool.pool_size
    
    def test_timeout_when_pool_exhausted(self, temp_db):
        """Test timeout when trying to get connection from exhausted pool."""
        pool = SQLiteConnectionPool(temp_db, pool_size=1, timeout=0.1)
        
        with pool.connection():
            # Pool is exhausted, next get should timeout
            with pytest.raises(RuntimeError, match="Could not get connection"):
                with pool.connection():
                    pass
        
        pool.close_all()
    
    def test_create_table_and_query(self, pool):
        """Test creating a table and querying it."""
        with pool.connection() as conn:
            conn.execute("""
                CREATE TABLE test_users (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    age INTEGER
                )
            """)
            conn.execute("INSERT INTO test_users VALUES (1, 'Alice', 30)")
            conn.commit()
        
        with pool.connection() as conn:
            cursor = conn.execute("SELECT name, age FROM test_users WHERE id=1")
            row = cursor.fetchone()
            assert row["name"] == "Alice"
            assert row["age"] == 30
    
    def test_thread_safety(self, pool, temp_db):
        """Test that pool is thread-safe."""
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                with pool.connection() as conn:
                    # Create table if not exists
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS thread_test (
                            id INTEGER PRIMARY KEY,
                            worker_id INTEGER,
                            timestamp REAL
                        )
                    """)
                    conn.commit()
                    
                    # Insert data
                    conn.execute(
                        "INSERT INTO thread_test (worker_id, timestamp) VALUES (?, ?)",
                        (worker_id, time.time())
                    )
                    conn.commit()
                    
                    results.append(worker_id)
            except Exception as e:
                errors.append(str(e))
        
        # Spawn multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 10
        assert sorted(results) == list(range(10))
    
    def test_close_pool(self, pool):
        """Test closing the connection pool."""
        pool.close_all()
        assert pool._closed is True
        
        # Getting connection from closed pool should raise
        with pytest.raises(RuntimeError, match="Connection pool is closed"):
            with pool.connection():
                pass
    
    def test_context_manager(self, temp_db):
        """Test using pool as a context manager."""
        with SQLiteConnectionPool(temp_db, pool_size=3) as pool:
            with pool.connection() as conn:
                conn.execute("SELECT 1")
        
        # Pool should be closed
        assert pool._closed is True


class TestGlobalPoolManagement:
    """Tests for global pool management functions."""
    
    def test_get_pool(self, temp_db):
        """Test getting a pool from global registry."""
        pool1 = get_pool(temp_db, pool_size=5)
        pool2 = get_pool(temp_db, pool_size=3)  # Should return same pool
        
        assert pool1 is pool2
        assert pool1.pool_size == 5  # Original pool_size preserved
        
        close_all_pools()
    
    def test_get_different_pools(self):
        """Test getting pools for different databases."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db1 = f.name
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db2 = f.name
        
        try:
            pool1 = get_pool(db1)
            pool2 = get_pool(db2)
            
            assert pool1 is not pool2
            assert pool1.db_path == db1
            assert pool2.db_path == db2
        finally:
            close_all_pools()
            os.unlink(db1)
            os.unlink(db2)
    
    def test_close_all_pools(self, temp_db):
        """Test closing all pools."""
        pool1 = get_pool(temp_db)
        
        close_all_pools()
        
        assert pool1._closed is True


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_execute_with_retry(self, pool):
        """Test executing query with retry."""
        # Create test table
        with pool.connection() as conn:
            conn.execute("""
                CREATE TABLE retry_test (
                    id INTEGER PRIMARY KEY,
                    value TEXT
                )
            """)
            conn.execute("INSERT INTO retry_test VALUES (1, 'test')")
            conn.commit()
        
        # Execute query with retry
        results = execute_with_retry(
            pool,
            "SELECT * FROM retry_test WHERE id=?",
            (1,)
        )
        
        assert len(results) == 1
        assert results[0]["value"] == "test"
    
    def test_get_pool_stats(self, pool):
        """Test getting pool statistics."""
        stats = get_pool_stats(pool)
        
        assert stats["db_path"] == pool.db_path
        assert stats["pool_size"] == pool.pool_size
        assert stats["available"] == pool.pool_size
        assert stats["in_use"] == 0
        assert stats["closed"] is False
        
        # Take a connection
        with pool.connection():
            stats = get_pool_stats(pool)
            assert stats["available"] == pool.pool_size - 1
            assert stats["in_use"] == 1


class TestPerformance:
    """Performance tests for connection pooling."""
    
    def test_pooling_performance(self, temp_db):
        """Compare pooled vs non-pooled performance."""
        # Setup
        num_operations = 100
        
        # Create test table
        conn = sqlite3.connect(temp_db)
        conn.execute("CREATE TABLE perf_test (id INTEGER, value TEXT)")
        conn.commit()
        conn.close()
        
        # Test without pooling (create new connection each time)
        start = time.time()
        for i in range(num_operations):
            conn = sqlite3.connect(temp_db)
            conn.execute("INSERT INTO perf_test VALUES (?, ?)", (i, f"value{i}"))
            conn.commit()
            conn.close()
        no_pool_time = time.time() - start
        
        # Clear table
        conn = sqlite3.connect(temp_db)
        conn.execute("DELETE FROM perf_test")
        conn.commit()
        conn.close()
        
        # Test with pooling (reuse connections)
        pool = SQLiteConnectionPool(temp_db, pool_size=5)
        start = time.time()
        for i in range(num_operations):
            with pool.connection() as conn:
                conn.execute("INSERT INTO perf_test VALUES (?, ?)", (i, f"value{i}"))
                conn.commit()
        pool_time = time.time() - start
        pool.close_all()
        
        # Pooling should be faster
        print(f"\nWithout pooling: {no_pool_time:.3f}s")
        print(f"With pooling: {pool_time:.3f}s")
        print(f"Speedup: {no_pool_time / pool_time:.2f}x")
        
        assert pool_time < no_pool_time, "Pooling should be faster"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
