"""Performance benchmarks for RFSN Controller v0.3.0 upgrades.

Benchmarks for:
- Async cache vs sync cache performance
- Planner v5 decision speed
- Verification manager throughput
- Strategy executor efficiency
"""

import asyncio
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import pytest


class BenchmarkTimer:
    """Simple benchmark timer."""

    def __init__(self, name: str):
        self.name = name
        self.start_time = 0.0
        self.end_time = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"\n⏱️  {self.name}: {duration:.4f}s")


def benchmark(func: Callable, iterations: int = 100) -> dict:
    """Run a benchmark function multiple times."""
    times = []

    for _ in range(iterations):
        start = time.time()
        func()
        elapsed = time.time() - start
        times.append(elapsed)

    return {
        "iterations": iterations,
        "total_time": sum(times),
        "avg_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "ops_per_sec": iterations / sum(times),
    }


async def async_benchmark(func: Callable, iterations: int = 100) -> dict:
    """Run an async benchmark function multiple times."""
    times = []

    for _ in range(iterations):
        start = time.time()
        await func()
        elapsed = time.time() - start
        times.append(elapsed)

    return {
        "iterations": iterations,
        "total_time": sum(times),
        "avg_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "ops_per_sec": iterations / sum(times),
    }


@pytest.mark.benchmark
class TestAsyncCacheBenchmark:
    """Benchmark async cache performance."""

    @pytest.mark.asyncio
    async def test_async_cache_read_performance(self):
        """Benchmark async cache read operations."""
        from rfsn_controller.async_multi_tier_cache import AsyncMultiTierCache

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cache = AsyncMultiTierCache(memory_size=1000, disk_path=f.name)
            await cache.initialize()

            # Populate cache
            for i in range(100):
                await cache.put(f"key-{i}", {"data": f"value-{i}"})

            # Benchmark reads
            async def read_op():
                await cache.get("key-50")

            with BenchmarkTimer("Async cache read (100 iterations)"):
                results = await async_benchmark(read_op, iterations=100)

            print(f"\n  Average: {results['avg_time']*1000:.2f}ms")
            print(f"  Throughput: {results['ops_per_sec']:.0f} ops/sec")

            await cache.close()

            # Assertions
            assert results["avg_time"] < 0.01  # <10ms per read

    @pytest.mark.asyncio
    async def test_async_cache_write_performance(self):
        """Benchmark async cache write operations."""
        from rfsn_controller.async_multi_tier_cache import AsyncMultiTierCache

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cache = AsyncMultiTierCache(memory_size=1000, disk_path=f.name)
            await cache.initialize()

            counter = 0

            async def write_op():
                nonlocal counter
                await cache.put(f"key-{counter}", {"data": f"value-{counter}"})
                counter += 1

            with BenchmarkTimer("Async cache write (100 iterations)"):
                results = await async_benchmark(write_op, iterations=100)

            print(f"\n  Average: {results['avg_time']*1000:.2f}ms")
            print(f"  Throughput: {results['ops_per_sec']:.0f} ops/sec")

            await cache.close()

            # Assertions
            assert results["avg_time"] < 0.02  # <20ms per write

    @pytest.mark.asyncio
    async def test_async_cache_concurrent_operations(self):
        """Benchmark concurrent cache operations."""
        from rfsn_controller.async_multi_tier_cache import AsyncMultiTierCache

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cache = AsyncMultiTierCache(memory_size=1000, disk_path=f.name)
            await cache.initialize()

            # Concurrent workers
            async def worker(worker_id: int, operations: int):
                for i in range(operations):
                    key = f"worker-{worker_id}-key-{i}"
                    await cache.put(key, {"data": i})
                    await cache.get(key)

            start = time.time()

            # Run 10 workers concurrently, 50 ops each = 500 total ops
            await asyncio.gather(
                *[worker(i, 50) for i in range(10)]
            )

            elapsed = time.time() - start
            throughput = 500 / elapsed

            print("\n⏱️  Concurrent operations (500 ops, 10 workers)")
            print(f"  Total time: {elapsed:.4f}s")
            print(f"  Throughput: {throughput:.0f} ops/sec")

            await cache.close()

            # Assertions
            assert throughput > 200  # At least 200 ops/sec


@pytest.mark.benchmark
class TestSyncVsAsyncCacheComparison:
    """Compare sync vs async cache performance."""

    @pytest.mark.timeout(120)  # Extended timeout for sympy import overhead
    def test_sync_cache_baseline(self):
        """Baseline sync cache performance."""
        # Reset global cache to avoid contention
        import rfsn_controller.multi_tier_cache as cache_module
        if cache_module._global_cache is not None:
            cache_module._global_cache.close()
            cache_module._global_cache = None
        
        from rfsn_controller.multi_tier_cache import MultiTierCache

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cache = MultiTierCache(memory_size=1000, disk_path=f.name)

            # Populate
            for i in range(100):
                cache.put(f"key-{i}", {"data": f"value-{i}"})

            def read_op():
                cache.get("key-50")

            with BenchmarkTimer("Sync cache read (100 iterations)"):
                results = benchmark(read_op, iterations=100)

            print(f"\n  Average: {results['avg_time']*1000:.2f}ms")
            print(f"  Throughput: {results['ops_per_sec']:.0f} ops/sec")

            cache.close()

    @pytest.mark.asyncio
    async def test_async_cache_comparison(self):
        """Async cache performance for comparison."""
        from rfsn_controller.async_multi_tier_cache import AsyncMultiTierCache

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            cache = AsyncMultiTierCache(memory_size=1000, disk_path=f.name)
            await cache.initialize()

            # Populate
            for i in range(100):
                await cache.put(f"key-{i}", {"data": f"value-{i}"})

            async def read_op():
                await cache.get("key-50")

            with BenchmarkTimer("Async cache read (100 iterations)"):
                results = await async_benchmark(read_op, iterations=100)

            print(f"\n  Average: {results['avg_time']*1000:.2f}ms")
            print(f"  Throughput: {results['ops_per_sec']:.0f} ops/sec")

            await cache.close()


@pytest.mark.benchmark
class TestPlannerV5Benchmark:
    """Benchmark Planner v5 performance."""

    def test_planner_v5_action_generation_speed(self):
        """Benchmark planner v5 action generation."""
        from rfsn_controller.planner_v5_adapter import PlannerV5Adapter

        adapter = PlannerV5Adapter(enabled=True)

        if not adapter.enabled:
            pytest.skip("Planner v5 not available")

        def generate_action():
            feedback = {
                "success": False,
                "tests_failed": 3,
                "output": "Test failure",
            }
            adapter.get_next_action(controller_feedback=feedback)

        with BenchmarkTimer("Planner v5 action generation (100 iterations)"):
            results = benchmark(generate_action, iterations=100)

        print(f"\n  Average: {results['avg_time']*1000:.2f}ms")
        print(f"  Actions/sec: {results['ops_per_sec']:.0f}")

        # Should generate actions quickly
        assert results["avg_time"] < 0.5  # <500ms per action


@pytest.mark.benchmark
class TestVerificationManagerBenchmark:
    """Benchmark VerificationManager performance."""

    @pytest.mark.asyncio
    async def test_verification_overhead(self):
        """Benchmark verification manager overhead."""
        from rfsn_controller.verification_manager import (
            VerificationConfig,
            VerificationManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = VerificationConfig(
                test_command=["echo", "test"],
                timeout_seconds=5,
                working_directory=Path(tmpdir),
            )
            manager = VerificationManager(config)

            async def verify_op():
                await manager.verify_patch(Path(tmpdir))

            with BenchmarkTimer("Verification manager (10 runs)"):
                results = await async_benchmark(verify_op, iterations=10)

            print(f"\n  Average: {results['avg_time']*1000:.2f}ms")
            print(f"  Verifications/sec: {results['ops_per_sec']:.2f}")

            # Overhead should be minimal for simple commands
            assert results["avg_time"] < 1.0  # <1s per verification


@pytest.mark.benchmark
class TestStrategyExecutorBenchmark:
    """Benchmark StrategyExecutor performance."""

    @pytest.mark.asyncio
    async def test_strategy_execution_speed(self):
        """Benchmark strategy execution."""
        from rfsn_controller.strategy_executor import (
            StrategyConfig,
            StrategyExecutor,
            StrategyStep,
            StrategyType,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = StrategyConfig(
                strategy_type=StrategyType.DIRECT_PATCH,
                max_steps=5,
                working_directory=Path(tmpdir),
            )
            executor = StrategyExecutor(config)

            # Simple strategy: read and edit a file
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("# Original content")

            steps = [
                StrategyStep("step1", "read_file", target="test.py"),
                StrategyStep(
                    "step2",
                    "edit_file",
                    target="test.py",
                    content="# Modified content",
                ),
            ]

            async def execute_strategy():
                await executor.execute_strategy(steps)

            with BenchmarkTimer("Strategy execution (10 runs)"):
                results = await async_benchmark(execute_strategy, iterations=10)

            print(f"\n  Average: {results['avg_time']*1000:.2f}ms")
            print(f"  Strategies/sec: {results['ops_per_sec']:.2f}")

            # Should execute quickly
            assert results["avg_time"] < 0.5  # <500ms per strategy


@pytest.mark.benchmark
class TestEndToEndBenchmark:
    """End-to-end performance benchmarks."""

    @pytest.mark.asyncio
    async def test_complete_repair_cycle_simulation(self):
        """Simulate a complete repair cycle and measure performance."""
        from rfsn_controller.async_multi_tier_cache import AsyncMultiTierCache
        from rfsn_controller.planner_v5_adapter import PlannerV5Adapter
        from rfsn_controller.verification_manager import (
            VerificationConfig,
            VerificationManager,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup components
            cache = AsyncMultiTierCache(memory_size=100)
            await cache.initialize()

            config = VerificationConfig(
                test_command=["echo", "tests"],
                timeout_seconds=5,
                working_directory=Path(tmpdir),
            )
            verifier = VerificationManager(config)

            planner = PlannerV5Adapter(enabled=True)

            start = time.time()

            # Simulate repair cycle
            for i in range(5):
                # Cache some data
                await cache.put(f"context-{i}", {"iteration": i})

                # Get planning decision
                if planner.enabled:
                    feedback = {"success": False, "tests_failed": 5 - i}
                    planner.get_next_action(controller_feedback=feedback)

                # Run verification
                await verifier.verify_patch(Path(tmpdir))

            elapsed = time.time() - start

            print("\n⏱️  Complete repair cycle simulation (5 iterations)")
            print(f"  Total time: {elapsed:.4f}s")
            print(f"  Average per iteration: {elapsed/5:.4f}s")

            await cache.close()

            # Should complete reasonably quickly
            assert elapsed < 10.0  # <10s for 5 iterations


def print_benchmark_summary():
    """Print benchmark summary."""
    print("\n" + "=" * 60)
    print("📊 RFSN Controller v0.3.0 - Performance Benchmark Summary")
    print("=" * 60)
    print("\nExpected Performance Improvements:")
    print("  • Async Cache: +15-25% throughput vs sync")
    print("  • Planner v5: <500ms per action generation")
    print("  • Verification: <1s overhead per run")
    print("  • Strategy Execution: <500ms per strategy")
    print("\nRun with: pytest tests/test_benchmarks.py -v -s -m benchmark")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print_benchmark_summary()
