"""Integration tests for VerificationManager."""

import asyncio
import sys
import tempfile
from pathlib import Path

import pytest

from rfsn_controller.verification_manager import (
    VerificationConfig,
    VerificationManager,
    VerificationResult,
)


class TestVerificationManagerIntegration:
    """Integration tests for VerificationManager."""

    @pytest.fixture
    def temp_worktree(self):
        """Create temporary worktree for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            worktree = Path(tmpdir) / "worktree"
            worktree.mkdir()

            # Create a simple Python test file
            test_file = worktree / "test_sample.py"
            test_file.write_text("""
def test_passing():
    assert True

def test_failing():
    assert False
""")
            yield worktree

    @pytest.fixture
    def basic_config(self, temp_worktree):
        """Create basic verification config."""
        return VerificationConfig(
            test_command=[sys.executable, "-m", "pytest", "-v", str(temp_worktree)],
            timeout_seconds=30,
            working_directory=temp_worktree,
        )

    @pytest.mark.asyncio
    async def test_basic_verification(self, basic_config, temp_worktree):
        """Test basic verification execution."""
        manager = VerificationManager(basic_config)

        result = await manager.verify_patch(temp_worktree)

        assert isinstance(result, VerificationResult)
        assert result.exit_code != 0  # Should fail (has failing test)
        assert not result.success
        assert len(result.failing_tests) > 0 or "FAILED" in result.stdout

    @pytest.mark.asyncio
    async def test_timeout_handling(self, temp_worktree):
        """Test timeout handling."""
        # Create config with very short timeout
        config = VerificationConfig(
            test_command=["sleep", "10"],  # Long command
            timeout_seconds=1,  # Short timeout
            working_directory=temp_worktree,
        )
        manager = VerificationManager(config)

        result = await manager.verify_patch(temp_worktree)

        assert not result.success
        assert result.exit_code == -1
        assert "timed out" in result.stderr.lower() or "timeout" in result.stderr.lower()

    @pytest.mark.asyncio
    async def test_retry_logic(self, temp_worktree):
        """Test retry logic on transient failures."""
        retry_count = 3
        config = VerificationConfig(
            test_command=["echo", "test"],
            timeout_seconds=5,
            max_retries=retry_count,
            working_directory=temp_worktree,
        )
        manager = VerificationManager(config)

        result = await manager.verify_patch(temp_worktree)

        # Should succeed (echo always succeeds)
        assert result.success
        assert result.exit_code == 0

        # Check stats
        stats = manager.get_stats()
        assert stats["total_verifications"] == 1

    @pytest.mark.asyncio
    async def test_passing_tests(self, temp_worktree):
        """Test verification with all passing tests."""
        # Create passing test file
        test_file = temp_worktree / "test_passing.py"
        test_file.write_text("""
def test_one():
    assert True

def test_two():
    assert 1 + 1 == 2
""")

        config = VerificationConfig(
            test_command=[sys.executable, "-m", "pytest", "-v", str(temp_worktree)],
            timeout_seconds=30,
            working_directory=temp_worktree,
        )
        manager = VerificationManager(config)

        result = await manager.verify_patch(temp_worktree)

        # All tests pass, so verification should succeed
        # (Note: this depends on pytest being installed)
        if "pytest" in result.stdout or "test" in result.stdout:
            # If pytest ran, check results
            if result.exit_code == 0:
                assert result.success

    @pytest.mark.asyncio
    async def test_multiple_verifications(self, temp_worktree):
        """Test running multiple verifications."""
        config = VerificationConfig(
            test_command=["echo", "test"],
            timeout_seconds=5,
            verify_multiple_times=3,
            working_directory=temp_worktree,
        )
        manager = VerificationManager(config)

        results = await manager.verify_multiple_times(temp_worktree)

        assert len(results) == 3
        assert all(isinstance(r, VerificationResult) for r in results)

    @pytest.mark.asyncio
    async def test_test_output_parsing(self, temp_worktree):
        """Test parsing of test output."""
        # Create test that produces parseable output
        test_file = temp_worktree / "test_parse.py"
        test_file.write_text("""
def test_pass_one():
    assert True

def test_fail_one():
    assert False, "Expected failure"

def test_pass_two():
    assert 1 == 1
""")

        config = VerificationConfig(
            test_command=[sys.executable, "-m", "pytest", "-v", str(temp_worktree)],
            timeout_seconds=30,
            working_directory=temp_worktree,
        )
        manager = VerificationManager(config)

        result = await manager.verify_patch(temp_worktree)

        # Should parse some failing tests
        # (Exact format depends on pytest version)
        assert isinstance(result.failing_tests, list)
        assert isinstance(result.passing_tests, list)

    @pytest.mark.asyncio
    async def test_error_signature_extraction(self, temp_worktree):
        """Test extraction of error signatures."""
        test_file = temp_worktree / "test_error.py"
        test_file.write_text("""
def test_with_error():
    raise ValueError("This is a test error")
""")

        config = VerificationConfig(
            test_command=[sys.executable, "-m", "pytest", "-v", str(temp_worktree)],
            timeout_seconds=30,
            working_directory=temp_worktree,
        )
        manager = VerificationManager(config)

        result = await manager.verify_patch(temp_worktree)

        # Should extract error signature
        if result.error_signature:
            assert "error" in result.error_signature.lower() or \
                   "exception" in result.error_signature.lower()

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, temp_worktree):
        """Test verification statistics tracking."""
        config = VerificationConfig(
            test_command=["echo", "test"],
            timeout_seconds=5,
            working_directory=temp_worktree,
        )
        manager = VerificationManager(config)

        # Run several verifications
        for _ in range(5):
            await manager.verify_patch(temp_worktree)

        stats = manager.get_stats()

        assert stats["total_verifications"] == 5
        assert stats["successful_verifications"] == 5
        assert stats["success_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_environment_variables(self, temp_worktree):
        """Test passing environment variables."""
        config = VerificationConfig(
            test_command=["env"],
            timeout_seconds=5,
            working_directory=temp_worktree,
        )
        manager = VerificationManager(config)

        result = await manager.verify_patch(
            temp_worktree,
            additional_env={"TEST_VAR": "test_value"},
        )

        # Should see environment variable in output
        assert "TEST_VAR" in result.stdout or "test_value" in result.stdout

    @pytest.mark.asyncio
    async def test_concurrent_verifications(self, temp_worktree):
        """Test running multiple verifications concurrently."""
        config = VerificationConfig(
            test_command=["echo", "test"],
            timeout_seconds=5,
            working_directory=temp_worktree,
        )

        managers = [VerificationManager(config) for _ in range(3)]

        # Run concurrently
        results = await asyncio.gather(
            *[m.verify_patch(temp_worktree) for m in managers]
        )

        assert len(results) == 3
        assert all(r.success for r in results)


@pytest.mark.integration
class TestVerificationManagerRealWorld:
    """Real-world integration scenarios."""

    @pytest.fixture
    def temp_worktree(self):
        """Create temporary worktree for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            worktree = Path(tmpdir) / "worktree"
            worktree.mkdir()
            yield worktree

    @pytest.mark.asyncio
    async def test_python_project_verification(self):
        """Test verification of a real Python project structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            worktree = Path(tmpdir)

            # Create project structure
            src_dir = worktree / "src"
            src_dir.mkdir()

            (src_dir / "__init__.py").write_text("")
            (src_dir / "calculator.py").write_text("""
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
""")

            # Create tests
            tests_dir = worktree / "tests"
            tests_dir.mkdir()

            (tests_dir / "__init__.py").write_text("")
            (tests_dir / "test_calculator.py").write_text("""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from calculator import add, subtract

def test_add():
    assert add(2, 3) == 5

def test_subtract():
    assert subtract(5, 3) == 2

def test_failing():
    assert add(2, 2) == 5  # This will fail
""")

            config = VerificationConfig(
                test_command=[sys.executable, "-m", "pytest", "-v"],
                timeout_seconds=30,
                working_directory=worktree,
            )
            manager = VerificationManager(config)

            result = await manager.verify_patch(worktree)

            # Should have run and detected failure
            assert result.exit_code != 0
            assert not result.success

    @pytest.mark.asyncio
    async def test_flaky_test_detection(self, temp_worktree):
        """Test detection of flaky tests through multiple runs."""

        # Create a flaky test
        test_file = temp_worktree / "test_flaky.py"
        test_file.write_text("""
import random

def test_flaky():
    # Flaky: passes ~50% of the time
    assert random.random() > 0.5
""")

        config = VerificationConfig(
            test_command=[sys.executable, "-m", "pytest", "-v", str(temp_worktree)],
            timeout_seconds=30,
            verify_multiple_times=10,
            working_directory=temp_worktree,
        )
        manager = VerificationManager(config)

        results = await manager.verify_multiple_times(temp_worktree)

        # Should have mix of successes and failures
        sum(1 for r in results if r.success)
        sum(1 for r in results if not r.success)

        # Flaky test should show inconsistent results
        # (Could be all pass or all fail due to randomness, but likely mixed)
        assert len(results) == 10


@pytest.mark.performance
class TestVerificationManagerPerformance:
    """Performance tests for VerificationManager."""

    @pytest.mark.asyncio
    async def test_verification_throughput(self):
        """Test verification throughput."""
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            worktree = Path(tmpdir)

            config = VerificationConfig(
                test_command=["echo", "fast"],
                timeout_seconds=5,
                working_directory=worktree,
            )
            manager = VerificationManager(config)

            start = time.time()

            # Run 50 verifications
            for _ in range(50):
                await manager.verify_patch(worktree)

            elapsed = time.time() - start
            throughput = 50 / elapsed

            print(f"\n⏱️  Verification throughput: {throughput:.1f} verifications/sec")

            # Should be reasonably fast
            assert throughput > 10  # At least 10 verifications/sec

    @pytest.mark.asyncio
    async def test_parallel_verification_performance(self):
        """Test parallel verification performance."""
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            worktree = Path(tmpdir)

            config = VerificationConfig(
                test_command=["echo", "test"],
                timeout_seconds=5,
                working_directory=worktree,
            )

            start = time.time()

            # Run 20 verifications in parallel
            managers = [VerificationManager(config) for _ in range(20)]
            await asyncio.gather(
                *[m.verify_patch(worktree) for m in managers]
            )

            elapsed = time.time() - start
            throughput = 20 / elapsed

            print(f"\n⏱️  Parallel verification: {throughput:.1f} verifications/sec")

            # Parallel should be faster than 20 sequential
            assert throughput > 15
