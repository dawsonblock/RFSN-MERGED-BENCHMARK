"""Tests for optimization modules."""

import tempfile
import time
from pathlib import Path

import pytest

from rfsn_controller.batch_file_ops import (
    async_batch_read_files,
    batch_read_files,
)
from rfsn_controller.early_stop_optimizer import (
    EarlyStopConfig,
    EarlyStopOptimizer,
)
from rfsn_controller.file_cache import FileCache


class TestFileCache:
    """Test file caching."""
    
    def test_cache_hit(self):
        """Test cache hit on repeated read."""
        cache = FileCache(max_entries=10)
        
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test content")
            filepath = f.name
        
        try:
            # First read - cache miss
            content1 = cache.get(filepath)
            assert content1 == "test content"
            assert cache.stats()['hits'] == 0
            assert cache.stats()['misses'] == 1
            
            # Second read - cache hit
            content2 = cache.get(filepath)
            assert content2 == "test content"
            assert cache.stats()['hits'] == 1
            assert cache.stats()['misses'] == 1
            
        finally:
            Path(filepath).unlink()
    
    def test_cache_invalidation_on_change(self):
        """Test cache invalidation when file changes."""
        cache = FileCache(max_entries=10)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("original content")
            filepath = f.name
        
        try:
            # Read original
            content1 = cache.get(filepath)
            assert content1 == "original content"
            
            # Modify file
            time.sleep(0.01)  # Ensure mtime changes
            Path(filepath).write_text("updated content")
            
            # Read again - should detect change
            content2 = cache.get(filepath)
            assert content2 == "updated content"
            assert cache.stats()['stale_reads'] == 1
            
        finally:
            Path(filepath).unlink()
    
    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = FileCache(max_entries=2)
        
        files = []
        try:
            # Create 3 temp files
            for i in range(3):
                f = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
                f.write(f"content {i}")
                f.close()
                files.append(f.name)
            
            # Read all 3 - should evict first
            for filepath in files:
                cache.get(filepath)
            
            assert cache.stats()['evictions'] >= 1
            assert cache.stats()['current_entries'] <= 2
            
        finally:
            for filepath in files:
                try:
                    Path(filepath).unlink()
                except:
                    pass


class TestEarlyStopOptimizer:
    """Test early stopping optimizer."""
    
    def test_stop_on_syntax_error(self):
        """Test stopping on syntax error."""
        optimizer = EarlyStopOptimizer()
        
        output = """
        test_example.py F
        
        ================================ FAILURES =================================
        ______________________________ test_foo ___________________________________
        
            def test_foo():
        >       x = 1 +
        E       SyntaxError: invalid syntax
        """
        
        decision = optimizer.should_stop_early(output)
        
        assert decision.should_stop is True
        assert "SyntaxError" in decision.reason or "Syntax error" in decision.reason
        assert decision.error_type == "SyntaxError"
    
    def test_stop_on_import_error(self):
        """Test stopping on import error."""
        optimizer = EarlyStopOptimizer()
        
        output = """
        test_example.py E
        
        =============================== ERRORS ====================================
        _____________________ ERROR collecting test_example.py ____________________
        ImportError: cannot import name 'foo' from 'module'
        """
        
        decision = optimizer.should_stop_early(output)
        
        assert decision.should_stop is True
        assert "ImportError" in decision.reason or "Import error" in decision.reason
        assert decision.error_type == "ImportError"
    
    def test_stop_on_max_failures(self):
        """Test stopping when max failures reached."""
        config = EarlyStopConfig(max_failures=2)
        optimizer = EarlyStopOptimizer(config)
        
        # First failure
        output1 = "FAILED test_1.py::test_a"
        decision1 = optimizer.should_stop_early(output1)
        assert decision1.should_stop is False
        
        # Second failure - should stop
        output2 = "FAILED test_2.py::test_b"
        decision2 = optimizer.should_stop_early(output2)
        assert decision2.should_stop is True
        assert decision2.failures_detected >= 2
    
    def test_disabled_early_stop(self):
        """Test that early stop can be disabled."""
        config = EarlyStopConfig(enabled=False)
        optimizer = EarlyStopOptimizer(config)
        
        output = "SyntaxError: invalid syntax"
        decision = optimizer.should_stop_early(output)
        
        assert decision.should_stop is False
        assert "disabled" in decision.reason.lower()


class TestBatchFileOps:
    """Test batch file operations."""
    
    def test_batch_read_files(self):
        """Test batch file reading."""
        files = []
        try:
            # Create temp files
            for i in range(3):
                f = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
                f.write(f"content {i}")
                f.close()
                files.append(f.name)
            
            # Batch read
            results = batch_read_files(files, max_workers=2)
            
            assert len(results) == 3
            for i, filepath in enumerate(files):
                assert results[filepath] == f"content {i}"
        
        finally:
            for filepath in files:
                try:
                    Path(filepath).unlink()
                except:
                    pass
    
    @pytest.mark.asyncio
    async def test_async_batch_read_files(self):
        """Test async batch file reading."""
        files = []
        try:
            # Create temp files
            for i in range(3):
                f = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
                f.write(f"async content {i}")
                f.close()
                files.append(f.name)
            
            # Async batch read
            results = await async_batch_read_files(files, max_concurrent=2)
            
            assert len(results) == 3
            for i, filepath in enumerate(files):
                assert results[filepath] == f"async content {i}"
        
        finally:
            for filepath in files:
                try:
                    Path(filepath).unlink()
                except:
                    pass
    
    def test_batch_read_missing_files(self):
        """Test batch reading with some missing files."""
        files = ["/nonexistent/file1.txt", "/nonexistent/file2.txt"]
        
        results = batch_read_files(files, max_workers=2)
        
        assert len(results) == 2
        assert all(v is None for v in results.values())


class TestOptimizationIntegration:
    """Test optimizations working together."""
    
    def test_cache_with_batch_ops(self):
        """Test file cache with batch operations."""
        from rfsn_controller.file_cache import get_file_cache
        
        cache = get_file_cache()
        cache.clear()  # Start fresh
        
        files = []
        try:
            # Create temp files
            for i in range(3):
                f = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
                f.write(f"cached content {i}")
                f.close()
                files.append(f.name)
            
            # First batch read - all cache misses
            results1 = batch_read_files(files)
            assert all(v is not None for v in results1.values())
            
            # Clear stats for the individual reads
            cache.clear()
            
            # Re-read one file to warm cache
            for filepath in files:
                _ = cache.get(filepath)
            
            # Read again - should hit cache
            hits_before = cache.stats()['hits']
            for filepath in files:
                cached = cache.get(filepath)
                assert cached is not None
            
            stats = cache.stats()
            assert stats['hits'] > hits_before
            
        finally:
            for filepath in files:
                try:
                    Path(filepath).unlink()
                except:
                    pass
