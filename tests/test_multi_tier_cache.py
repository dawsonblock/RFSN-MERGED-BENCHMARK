"""Tests for multi-tier cache system."""

import tempfile

import pytest

# These tests have timing issues in parallel execution due to SQLite/multiprocess contention
pytestmark = [pytest.mark.timeout(60)]


@pytest.fixture(autouse=True)
def reset_global_cache():
    """Reset global cache before each test to ensure isolation."""
    import rfsn_controller.multi_tier_cache as cache_module
    # Reset global cache
    cache_module._global_cache = None
    yield
    # Cleanup after test
    if cache_module._global_cache is not None:
        cache_module._global_cache.close()
        cache_module._global_cache = None


def test_multi_tier_cache_initialization():
    """Test MultiTierCache can be initialized."""
    from rfsn_controller.multi_tier_cache import MultiTierCache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = MultiTierCache(
            memory_size=100,
            disk_path=f"{tmpdir}/test_cache.db",
        )
        assert cache.memory_size == 100
        assert len(cache._memory_cache) == 0
        cache.close()


def test_cache_put_get():
    """Test basic put/get operations."""
    from rfsn_controller.multi_tier_cache import MultiTierCache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = MultiTierCache(disk_path=f"{tmpdir}/test_cache.db")
        
        # Put a value
        cache.put("test_key", {"data": "value"})
        
        # Get it back
        value = cache.get("test_key")
        assert value == {"data": "value"}
        
        cache.close()


def test_cache_miss():
    """Test cache miss returns None."""
    from rfsn_controller.multi_tier_cache import MultiTierCache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = MultiTierCache(disk_path=f"{tmpdir}/test_cache.db")
        
        value = cache.get("nonexistent_key")
        assert value is None
        
        cache.close()


def test_cache_stats():
    """Test cache statistics tracking."""
    from rfsn_controller.multi_tier_cache import MultiTierCache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = MultiTierCache(disk_path=f"{tmpdir}/test_cache.db")
        
        # Put and get
        cache.put("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.stats()
        assert stats["memory_hits"] == 1
        assert stats["misses"] == 1
        assert stats["memory_size"] == 1
        
        cache.close()


def test_cache_invalidate():
    """Test cache invalidation."""
    from rfsn_controller.multi_tier_cache import MultiTierCache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = MultiTierCache(disk_path=f"{tmpdir}/test_cache.db")
        
        cache.put("test_key", "value")
        assert cache.get("test_key") == "value"
        
        cache.invalidate("test_key")
        assert cache.get("test_key") is None
        
        cache.close()


def test_cache_clear():
    """Test clearing all caches."""
    from rfsn_controller.multi_tier_cache import MultiTierCache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = MultiTierCache(disk_path=f"{tmpdir}/test_cache.db")
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.clear()
        
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        
        cache.close()


def test_cached_decorator():
    """Test @cached decorator."""
    import uuid

    from rfsn_controller.multi_tier_cache import cached

    call_count = [0]
    # Use unique key prefix to avoid disk cache collisions
    unique_prefix = f"test_func_{uuid.uuid4().hex[:8]}"

    @cached(ttl_seconds=3600, key_prefix=unique_prefix)
    def expensive_function(x, y):
        call_count[0] += 1
        return x + y

    # First call
    result1 = expensive_function(1, 2)
    assert result1 == 3
    assert call_count[0] == 1

    # Second call should use cache
    result2 = expensive_function(1, 2)
    assert result2 == 3
    assert call_count[0] == 1  # Not incremented

    # Different args should miss cache
    result3 = expensive_function(2, 3)
    assert result3 == 5
    assert call_count[0] == 2


def test_memory_eviction():
    """Test LRU eviction in memory cache."""
    from rfsn_controller.multi_tier_cache import MultiTierCache

    with tempfile.TemporaryDirectory() as tmpdir:
        cache = MultiTierCache(
            memory_size=2,
            disk_path=f"{tmpdir}/test_cache.db",
        )
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")  # Should evict key1
        
        stats = cache.stats()
        assert stats["memory_evictions"] >= 1
        assert stats["memory_size"] == 2
        
        cache.close()


def test_global_cache():
    """Test global cache singleton."""
    from rfsn_controller.multi_tier_cache import get_global_cache

    cache1 = get_global_cache()
    cache2 = get_global_cache()
    
    assert cache1 is cache2  # Same instance
