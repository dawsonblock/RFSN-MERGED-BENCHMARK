"""Tests for database connection pooling."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from rfsn_controller.connection_pool import ConnectionPool


class TestConnectionPool:
    """Test connection pool functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        # Create test table
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO test (value) VALUES ('test1'), ('test2')")
        conn.commit()
        conn.close()
        
        yield db_path
        
        # Cleanup
        Path(db_path).unlink(missing_ok=True)
    
    def test_pool_initialization(self, temp_db):
        """Test pool initializes correctly."""
        pool = ConnectionPool(temp_db, pool_size=3)
        
        assert pool.pool_size == 3
        assert not pool.is_closed
        assert pool.available_connections == 3
        
        pool.close_all()
    
    def test_get_connection(self, temp_db):
        """Test getting connection from pool."""
        pool = ConnectionPool(temp_db, pool_size=2)
        
        with pool.get_connection() as conn:
            assert isinstance(conn, sqlite3.Connection)
            result = conn.execute("SELECT COUNT(*) FROM test").fetchone()
            assert result[0] == 2
        
        pool.close_all()
    
    def test_connection_returned_to_pool(self, temp_db):
        """Test connection is returned after use."""
        pool = ConnectionPool(temp_db, pool_size=2)
        
        assert pool.available_connections == 2
        
        with pool.get_connection():
            assert pool.available_connections == 1
        
        # Connection should be returned
        assert pool.available_connections == 2
        
        pool.close_all()
    
    def test_multiple_connections(self, temp_db):
        """Test multiple connections can be used concurrently."""
        pool = ConnectionPool(temp_db, pool_size=3)
        
        with pool.get_connection() as conn1:
            with pool.get_connection() as conn2:
                assert conn1 is not conn2
                assert pool.available_connections == 1
        
        assert pool.available_connections == 3
        
        pool.close_all()
    
    def test_connection_timeout(self, temp_db):
        """Test timeout when pool exhausted."""
        pool = ConnectionPool(temp_db, pool_size=1, timeout=0.1)
        
        with pool.get_connection():
            # Pool exhausted, should timeout
            with pytest.raises(TimeoutError):
                with pool.get_connection():
                    pass
        
        pool.close_all()
    
    def test_wal_mode_enabled(self, temp_db):
        """Test WAL mode is enabled for concurrency."""
        pool = ConnectionPool(temp_db)
        
        with pool.get_connection() as conn:
            result = conn.execute("PRAGMA journal_mode").fetchone()
            assert result[0].lower() == "wal"
        
        pool.close_all()
    
    def test_foreign_keys_enabled(self, temp_db):
        """Test foreign keys are enabled."""
        pool = ConnectionPool(temp_db)
        
        with pool.get_connection() as conn:
            result = conn.execute("PRAGMA foreign_keys").fetchone()
            assert result[0] == 1
        
        pool.close_all()
    
    def test_close_all_connections(self, temp_db):
        """Test all connections are closed."""
        pool = ConnectionPool(temp_db, pool_size=3)
        
        pool.close_all()
        
        assert pool.is_closed
        assert pool.available_connections == 0
        
        # Should raise when trying to get connection
        with pytest.raises(RuntimeError):
            with pool.get_connection():
                pass
    
    def test_context_manager(self, temp_db):
        """Test pool as context manager."""
        with ConnectionPool(temp_db, pool_size=2) as pool:
            assert not pool.is_closed
            
            with pool.get_connection() as conn:
                result = conn.execute("SELECT COUNT(*) FROM test").fetchone()
                assert result[0] == 2
        
        # Pool should be closed after context
        assert pool.is_closed
    
    def test_connection_reuse(self, temp_db):
        """Test connections are reused from pool."""
        pool = ConnectionPool(temp_db, pool_size=1)
        
        # With pool_size=1, there's only one connection in the pool
        # Get and return connection
        with pool.get_connection() as conn1:
            conn1_id = id(conn1)
        
        # Get again - must be same connection since pool_size=1
        with pool.get_connection() as conn2:
            conn2_id = id(conn2)
        
        # Same connection object should be reused
        assert conn1_id == conn2_id
        
        pool.close_all()
    
    def test_custom_kwargs(self, temp_db):
        """Test custom connection kwargs."""
        pool = ConnectionPool(
            temp_db,
            pool_size=1,
            isolation_level="DEFERRED"
        )
        
        with pool.get_connection() as conn:
            # Connection should be configured
            assert conn.isolation_level == "DEFERRED"
        
        pool.close_all()
    
    def test_transaction_handling(self, temp_db):
        """Test transactions work correctly."""
        pool = ConnectionPool(temp_db, pool_size=1)
        
        # Insert in transaction
        with pool.get_connection() as conn:
            conn.execute("INSERT INTO test (value) VALUES ('test3')")
            conn.commit()
        
        # Verify insert
        with pool.get_connection() as conn:
            result = conn.execute("SELECT COUNT(*) FROM test").fetchone()
            assert result[0] == 3
        
        pool.close_all()
    
    def test_rollback_transaction(self, temp_db):
        """Test transaction rollback."""
        pool = ConnectionPool(temp_db, pool_size=1)
        
        # Start transaction but rollback
        with pool.get_connection() as conn:
            conn.execute("INSERT INTO test (value) VALUES ('test_rollback')")
            conn.rollback()
        
        # Verify not inserted
        with pool.get_connection() as conn:
            result = conn.execute("SELECT COUNT(*) FROM test").fetchone()
            assert result[0] == 2  # Original 2 rows
        
        pool.close_all()
    
    def test_exception_during_connection_use(self, temp_db):
        """Test connection returned even if exception occurs."""
        pool = ConnectionPool(temp_db, pool_size=2)
        
        try:
            with pool.get_connection():
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Connection should still be returned
        assert pool.available_connections == 2
        
        pool.close_all()
    
    def test_check_same_thread_disabled(self, temp_db):
        """Test check_same_thread is disabled by default."""
        pool = ConnectionPool(temp_db, check_same_thread=False)
        
        # Should be able to use connection
        # (in real thread scenario, this would matter)
        with pool.get_connection() as conn:
            result = conn.execute("SELECT 1").fetchone()
            assert result[0] == 1
        
        pool.close_all()


class TestConnectionPoolEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_database_path(self):
        """Test with invalid database path."""
        # SQLite will create the file
        with tempfile.NamedTemporaryFile(delete=True) as f:
            db_path = f.name
        
        pool = ConnectionPool(db_path, pool_size=1)
        
        with pool.get_connection() as conn:
            # Should be able to create tables
            conn.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER)")
        
        pool.close_all()
        Path(db_path).unlink(missing_ok=True)
    
    def test_zero_pool_size(self):
        """Test pool with size 0 should fail."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            with pytest.raises(ValueError):
                ConnectionPool(f.name, pool_size=0)
    
    def test_large_pool_size(self):
        """Test pool with large size."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            pool = ConnectionPool(f.name, pool_size=50)
            assert pool.available_connections == 50
            pool.close_all()
