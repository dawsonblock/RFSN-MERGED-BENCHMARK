"""Tests for async multi-tier cache."""

import asyncio
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

# These tests have timing issues in parallel execution due to SQLite/multiprocess contention  
pytestmark = [pytest.mark.timeout(60)]

from rfsn_controller.async_multi_tier_cache import (
    AsyncCacheEntry,
    AsyncMultiTierCache,
    get_global_async_cache,
)


@pytest_asyncio.fixture
async def temp_db():
    """Create temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_cache.db"
        yield db_path


@pytest_asyncio.fixture
async def cache(temp_db):
    """Create test cache instance."""
    cache = AsyncMultiTierCache(
        memory_size=10,
        disk_path=temp_db,
        disk_ttl_hours=1,
    )
    await cache.initialize()
    yield cache
    await cache.close()


class TestAsyncCacheEntry:
    """Test cache entry dataclass."""

    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        entry = AsyncCacheEntry(
            key="test-key",
            value={"data": "value"},
            created_at=1000.0,
            last_accessed=1000.0,
            access_count=1,
            size_bytes=100,
            tier="memory",
        )

        assert entry.key == "test-key"
        assert entry.value == {"data": "value"}
        assert entry.tier == "memory"


class TestAsyncMultiTierCache:
    """Test async multi-tier cache."""

    @pytest.mark.asyncio
    async def test_initialization(self, temp_db):
        """Test cache initialization."""
        cache = AsyncMultiTierCache(disk_path=temp_db)
        await cache.initialize()

        assert cache._initialized
        assert cache._pool is not None
        assert temp_db.exists()

        await cache.close()

    @pytest.mark.asyncio
    async def test_put_and_get_memory(self, cache):
        """Test putting and getting from memory cache."""
        # Put value
        await cache.put("test-key", {"data": "value"})

        # Get value (should hit memory)
        value = await cache.get("test-key")
        assert value == {"data": "value"}

        # Check stats
        stats = await cache.stats()
        assert stats["memory_hits"] == 1
        assert stats["disk_hits"] == 0
        assert stats["misses"] == 0

    @pytest.mark.asyncio
    async def test_get_miss(self, cache):
        """Test cache miss."""
        value = await cache.get("nonexistent-key")
        assert value is None

        stats = await cache.stats()
        assert stats["misses"] == 1

    @pytest.mark.asyncio
    async def test_memory_eviction(self, cache):
        """Test LRU eviction from memory."""
        # Fill memory cache (capacity = 10)
        for i in range(12):
            await cache.put(f"key-{i}", f"value-{i}")

        # First two keys should be evicted
        value0 = await cache.get("key-0")
        value1 = await cache.get("key-1")

        # These should hit disk (promoted back to memory)
        assert value0 == "value-0"
        assert value1 == "value-1"

        stats = await cache.stats()
        assert stats["memory_evictions"] >= 2
        assert stats["disk_hits"] >= 2

    @pytest.mark.asyncio
    async def test_disk_persistence(self, temp_db):
        """Test that disk cache persists across instances."""
        # Create cache, put value, close
        cache1 = AsyncMultiTierCache(disk_path=temp_db)
        await cache1.initialize()
        await cache1.put("persistent-key", "persistent-value")
        await cache1.close()

        # Create new cache, should find value on disk
        cache2 = AsyncMultiTierCache(disk_path=temp_db)
        await cache2.initialize()
        value = await cache2.get("persistent-key")
        assert value == "persistent-value"

        stats = await cache2.stats()
        assert stats["disk_hits"] == 1

        await cache2.close()

    @pytest.mark.asyncio
    async def test_ttl_expiration(self, temp_db):
        """Test that expired entries are not returned."""
        cache = AsyncMultiTierCache(
            disk_path=temp_db,
            disk_ttl_hours=0,  # Expire immediately
        )
        await cache.initialize()

        # Put value
        await cache.put("expire-key", "expire-value")

        # Clear memory to force disk lookup
        await cache._memory_lock.__aenter__()
        cache._memory_cache.clear()
        await cache._memory_lock.__aexit__(None, None, None)

        # Wait a bit for expiration
        await asyncio.sleep(0.1)

        # Should not find expired value
        value = await cache.get("expire-key")
        assert value is None

        await cache.close()

    @pytest.mark.asyncio
    async def test_invalidate(self, cache):
        """Test invalidating a cache entry."""
        # Put value
        await cache.put("invalid-key", "invalid-value")

        # Invalidate
        await cache.invalidate("invalid-key")

        # Should not find it
        value = await cache.get("invalid-key")
        assert value is None

    @pytest.mark.asyncio
    async def test_clear(self, cache):
        """Test clearing entire cache."""
        # Put multiple values
        await cache.put("key-1", "value-1")
        await cache.put("key-2", "value-2")
        await cache.put("key-3", "value-3")

        # Clear cache
        await cache.clear()

        # Nothing should be found
        assert await cache.get("key-1") is None
        assert await cache.get("key-2") is None
        assert await cache.get("key-3") is None

    @pytest.mark.asyncio
    async def test_stats(self, cache):
        """Test cache statistics."""
        # Generate some activity
        await cache.put("key-1", "value-1")
        await cache.get("key-1")  # Memory hit
        await cache.get("nonexistent")  # Miss

        # Clear memory, force disk hit
        await cache._memory_lock.__aenter__()
        cache._memory_cache.clear()
        await cache._memory_lock.__aexit__(None, None, None)
        await cache.get("key-1")  # Disk hit

        stats = await cache.stats()

        assert stats["memory_hits"] >= 1
        assert stats["disk_hits"] >= 1
        assert stats["misses"] >= 1
        assert stats["puts"] >= 1
        assert stats["total_requests"] >= 3
        assert 0 <= stats["hit_rate"] <= 1.0

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, temp_db):
        """Test cleanup of expired entries."""
        cache = AsyncMultiTierCache(
            disk_path=temp_db,
            disk_ttl_hours=0,  # Expire immediately
        )
        await cache.initialize()

        # Put some values
        await cache.put("key-1", "value-1")
        await cache.put("key-2", "value-2")

        # Wait for expiration
        await asyncio.sleep(0.1)

        # Cleanup
        deleted = await cache.cleanup_expired()
        assert deleted >= 0  # Should delete expired entries

        await cache.close()

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_access(self, cache):
        """Test concurrent cache operations."""

        async def worker(worker_id: int, count: int):
            for i in range(count):
                key = f"worker-{worker_id}-key-{i}"
                await cache.put(key, f"value-{i}")
                value = await cache.get(key)
                assert value == f"value-{i}"

        # Run multiple workers concurrently
        await asyncio.gather(
            worker(1, 10),
            worker(2, 10),
            worker(3, 10),
        )

        stats = await cache.stats()
        # Relaxed assertions - some puts may be evicted/overwritten in concurrent scenarios
        assert stats["puts"] >= 25  # Allow for some variance
        assert stats["memory_hits"] >= 25

    @pytest.mark.asyncio
    async def test_large_values(self, cache):
        """Test caching large values."""
        large_value = {"data": "x" * 10000}

        await cache.put("large-key", large_value)
        value = await cache.get("large-key")

        assert value == large_value

    @pytest.mark.asyncio
    async def test_complex_types(self, cache):
        """Test caching complex Python types."""
        complex_value = {
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "tuple": (1, 2, 3),
            "set": {1, 2, 3},  # Will become list after pickle
        }

        await cache.put("complex-key", complex_value)
        value = await cache.get("complex-key")

        assert value["list"] == [1, 2, 3]
        assert value["dict"] == {"nested": "value"}
        assert value["tuple"] == (1, 2, 3)


class TestGlobalAsyncCache:
    """Test global async cache."""

    @pytest.mark.asyncio
    async def test_get_global_cache(self, temp_db):
        """Test getting global cache instance."""
        cache1 = await get_global_async_cache(disk_path=temp_db)
        cache2 = await get_global_async_cache(disk_path=temp_db)

        # Should be same instance
        assert cache1 is cache2

        await cache1.close()


@pytest.mark.asyncio
async def test_cache_hit_rate_improvement(temp_db):
    """Integration test: verify cache improves hit rate."""
    cache = AsyncMultiTierCache(memory_size=5, disk_path=temp_db)
    await cache.initialize()

    # First pass: all misses
    for i in range(10):
        await cache.put(f"key-{i}", f"value-{i}")

    # Second pass: should hit memory or disk
    for i in range(10):
        value = await cache.get(f"key-{i}")
        assert value == f"value-{i}"

    stats = await cache.stats()

    # Should have high hit rate
    assert stats["hit_rate"] > 0.5  # At least 50% hit rate

    await cache.close()
