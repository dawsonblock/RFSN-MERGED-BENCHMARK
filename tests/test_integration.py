"""Integration tests for RFSN Controller v0.3.0 upgrades."""

import pytest


class TestPlannerV5Integration:
    """Test Planner v5 integration."""

    def test_planner_v5_import(self):
        """Test that Planner v5 can be imported."""
        from rfsn_controller.planner_v5_adapter import HAS_PLANNER_V5, PlannerV5Adapter
        
        assert HAS_PLANNER_V5 is True
        adapter = PlannerV5Adapter(enabled=True)
        assert adapter.enabled is True

    def test_planner_v5_in_controller(self):
        """Test that Planner v5 is referenced in controller."""
        import inspect

        from rfsn_controller import controller
        
        source = inspect.getsource(controller)
        assert "planner_v5" in source
        assert "PlannerV5Adapter" in source


class TestAsyncDatabaseIntegration:
    """Test async database operations."""

    @pytest.mark.asyncio
    async def test_async_cache_basic(self):
        """Test basic async cache operations."""
        from rfsn_controller.async_multi_tier_cache import AsyncMultiTierCache
        
        cache = AsyncMultiTierCache(memory_size=10)
        await cache.initialize()
        
        # Test put/get
        await cache.put("test", {"value": 123})
        result = await cache.get("test")
        
        assert result == {"value": 123}
        
        await cache.close()


class TestVerificationManager:
    """Test VerificationManager integration."""

    def test_verification_manager_import(self):
        """Test that VerificationManager can be imported."""
        from rfsn_controller.verification_manager import (
            VerificationConfig,
        )
        
        config = VerificationConfig(
            test_command=["pytest", "-q"],
            timeout_seconds=60,
        )
        
        assert config.test_command == ["pytest", "-q"]
        assert config.timeout_seconds == 60


class TestStrategyExecutor:
    """Test StrategyExecutor integration."""

    def test_strategy_executor_import(self):
        """Test that StrategyExecutor can be imported."""
        from rfsn_controller.strategy_executor import (
            StrategyType,
        )
        
        # Check enum values
        assert hasattr(StrategyType, 'INCREMENTAL')
        assert hasattr(StrategyType, 'DIRECT_PATCH')
        assert hasattr(StrategyType, 'ENSEMBLE')


class TestAgentFoundation:
    """Test SWE-bench agent foundation."""

    def test_agent_types_import(self):
        """Test that agent types can be imported."""
        from agent.types import (
            Phase,
        )
        
        # Check Phase enum
        assert hasattr(Phase, 'INGEST')
        assert hasattr(Phase, 'LOCALIZE')
        assert hasattr(Phase, 'PATCH_CANDIDATES')
        
    def test_agent_profiles(self):
        """Test profile loading."""
        from pathlib import Path
        
        # Check that profile files exist
        lite_path = Path("profiles/swebench_lite.yaml")
        verified_path = Path("profiles/swebench_verified.yaml")
        
        assert lite_path.exists(), f"Missing {lite_path}"
        assert verified_path.exists(), f"Missing {verified_path}"

    def test_gate_extensions(self):
        """Test gate extension modules."""
        from gate_ext import gate_with_profile
        from gate_ext.policy_files import check_files
        from gate_ext.policy_phase import check_phase
        from gate_ext.policy_tests import check_tests
        
        # Just check they can be imported
        assert callable(gate_with_profile)
        assert callable(check_phase)
        assert callable(check_files)
        assert callable(check_tests)

    def test_memory_logging(self):
        """Test memory logging module."""
        from memory.log import append_event
        
        assert callable(append_event)


class TestControllerIntegration:
    """Test main controller integration."""

    def test_controller_import(self):
        """Test that controller can be imported."""
        from rfsn_controller.controller import run_controller
        
        assert callable(run_controller)
        
    def test_cli_import(self):
        """Test that CLI can be imported."""
        from rfsn_controller.cli import main
        
        assert callable(main)

    def test_planner_mode_values(self):
        """Test that planner mode accepts v5."""
        from rfsn_controller.config import ControllerConfig
        
        # Should not raise
        config = ControllerConfig(
            github_url="https://github.com/test/repo",
            planner_mode="v5"
        )
        
        assert config.planner_mode == "v5"
