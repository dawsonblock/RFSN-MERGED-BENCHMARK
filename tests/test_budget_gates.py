"""Comprehensive tests for the Budget Gates system (Phase 3).

This module tests:
- Budget class functionality
- State transitions (ACTIVE -> WARNING -> EXCEEDED -> EXHAUSTED)
- Enforcement mechanisms (BudgetExceeded exception)
- Integration with controller, LLM calls, exec_utils, and sandbox
- Edge cases and error handling
- Thread safety
"""

import threading
import time
from pathlib import Path

import pytest

from rfsn_controller.budget import (
    Budget,
    BudgetExceeded,
    BudgetState,
    check_time_budget_global,
    create_budget,
    get_global_budget,
    record_llm_call_global,
    record_subprocess_call_global,
    set_global_budget,
)
from rfsn_controller.config import BudgetConfig, ControllerConfig

# ─────────────────────────────────────────────────────────────────────────────
# Test BudgetState Enum
# ─────────────────────────────────────────────────────────────────────────────

class TestBudgetState:
    """Tests for BudgetState enum."""
    
    def test_state_values(self):
        """Test that all expected states exist."""
        assert BudgetState.ACTIVE.value == "active"
        assert BudgetState.WARNING.value == "warning"
        assert BudgetState.EXCEEDED.value == "exceeded"
        assert BudgetState.EXHAUSTED.value == "exhausted"
    
    def test_state_comparison(self):
        """Test state enum comparisons."""
        assert BudgetState.ACTIVE != BudgetState.WARNING
        assert BudgetState.WARNING != BudgetState.EXCEEDED
        assert BudgetState.EXCEEDED != BudgetState.EXHAUSTED


# ─────────────────────────────────────────────────────────────────────────────
# Test BudgetExceeded Exception
# ─────────────────────────────────────────────────────────────────────────────

class TestBudgetExceededException:
    """Tests for BudgetExceeded exception."""
    
    def test_exception_attributes(self):
        """Test exception has correct attributes."""
        exc = BudgetExceeded(resource="steps", current=10, limit=5)
        assert exc.resource == "steps"
        assert exc.current == 10
        assert exc.limit == 5
        assert "steps" in str(exc)
        assert "10" in str(exc)
        assert "5" in str(exc)
    
    def test_exception_custom_message(self):
        """Test exception with custom message."""
        exc = BudgetExceeded(
            resource="tokens",
            current=1000,
            limit=500,
            message="Custom error: too many tokens"
        )
        assert exc.message == "Custom error: too many tokens"
        assert str(exc) == "Custom error: too many tokens"
    
    def test_exception_repr(self):
        """Test exception repr."""
        exc = BudgetExceeded(resource="llm_calls", current=15, limit=10)
        repr_str = repr(exc)
        assert "BudgetExceeded" in repr_str
        assert "llm_calls" in repr_str
        assert "current=15" in repr_str
        assert "limit=10" in repr_str
    
    def test_exception_is_catchable(self):
        """Test that the exception can be caught."""
        with pytest.raises(BudgetExceeded) as exc_info:
            raise BudgetExceeded(resource="time", current=100, limit=50)
        assert exc_info.value.resource == "time"


# ─────────────────────────────────────────────────────────────────────────────
# Test Budget Class - Basic Functionality
# ─────────────────────────────────────────────────────────────────────────────

class TestBudgetBasic:
    """Basic tests for Budget class."""
    
    def test_create_budget_with_defaults(self):
        """Test budget creation with default values."""
        budget = Budget()
        assert budget.max_steps == 0
        assert budget.max_llm_calls == 0
        assert budget.max_tokens == 0
        assert budget.max_time_seconds == 0
        assert budget.max_subprocess_calls == 0
        assert budget.warning_threshold == 0.8
    
    def test_create_budget_with_limits(self):
        """Test budget creation with specific limits."""
        budget = Budget(
            max_steps=10,
            max_llm_calls=5,
            max_tokens=1000,
            max_time_seconds=60,
            max_subprocess_calls=20,
            warning_threshold=0.75,
        )
        assert budget.max_steps == 10
        assert budget.max_llm_calls == 5
        assert budget.max_tokens == 1000
        assert budget.max_time_seconds == 60
        assert budget.max_subprocess_calls == 20
        assert budget.warning_threshold == 0.75
    
    def test_factory_function(self):
        """Test create_budget factory function."""
        budget = create_budget(
            max_steps=5,
            max_llm_calls=3,
            warning_threshold=0.9,
        )
        assert budget.max_steps == 5
        assert budget.max_llm_calls == 3
        assert budget.warning_threshold == 0.9
    
    def test_initial_counters_are_zero(self):
        """Test that counters start at zero."""
        budget = Budget(max_steps=10)
        assert budget.steps == 0
        assert budget.llm_calls == 0
        assert budget.tokens == 0
        assert budget.subprocess_calls == 0
    
    def test_elapsed_time_tracking(self):
        """Test that elapsed time is tracked."""
        budget = Budget()
        time.sleep(0.1)
        assert budget.elapsed_seconds >= 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Test Budget - Consumption Tracking
# ─────────────────────────────────────────────────────────────────────────────

class TestBudgetConsumption:
    """Tests for budget consumption tracking."""
    
    def test_record_step(self):
        """Test step recording."""
        budget = Budget(max_steps=10)
        budget.record_step()
        assert budget.steps == 1
        budget.record_step(3)
        assert budget.steps == 4
    
    def test_record_llm_call(self):
        """Test LLM call recording."""
        budget = Budget(max_llm_calls=10)
        budget.record_llm_call(tokens=100)
        assert budget.llm_calls == 1
        assert budget.tokens == 100
        budget.record_llm_call(tokens=50)
        assert budget.llm_calls == 2
        assert budget.tokens == 150
    
    def test_record_tokens_only(self):
        """Test token recording without incrementing call count."""
        budget = Budget(max_tokens=1000)
        budget.record_tokens(200)
        assert budget.tokens == 200
        assert budget.llm_calls == 0
    
    def test_record_subprocess_call(self):
        """Test subprocess call recording."""
        budget = Budget(max_subprocess_calls=20)
        budget.record_subprocess_call()
        assert budget.subprocess_calls == 1
        budget.record_subprocess_call(5)
        assert budget.subprocess_calls == 6
    
    def test_remaining_calculations(self):
        """Test remaining resource calculations."""
        budget = Budget(
            max_steps=10,
            max_llm_calls=5,
            max_tokens=1000,
            max_subprocess_calls=20,
        )
        budget.record_step(3)
        budget.record_llm_call(tokens=300)
        budget.record_subprocess_call(5)
        
        assert budget.remaining_steps == 7
        assert budget.remaining_llm_calls == 4
        assert budget.remaining_tokens == 700
        assert budget.remaining_subprocess_calls == 15
    
    def test_remaining_returns_none_for_unlimited(self):
        """Test that remaining is None when limit is 0 (unlimited)."""
        budget = Budget()  # All limits at 0
        assert budget.remaining_steps is None
        assert budget.remaining_llm_calls is None
        assert budget.remaining_tokens is None
        assert budget.remaining_subprocess_calls is None
    
    def test_reset_counters(self):
        """Test that reset clears all counters."""
        budget = Budget(max_steps=10, max_llm_calls=5)
        budget.record_step(5)
        budget.record_llm_call(tokens=100)
        budget.record_subprocess_call(3)
        
        budget.reset()
        
        assert budget.steps == 0
        assert budget.llm_calls == 0
        assert budget.tokens == 0
        assert budget.subprocess_calls == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test Budget - State Transitions
# ─────────────────────────────────────────────────────────────────────────────

class TestBudgetStateTransitions:
    """Tests for budget state transitions."""
    
    def test_initial_state_is_active(self):
        """Test that initial state is ACTIVE."""
        budget = Budget(max_steps=10)
        assert budget.get_state() == BudgetState.ACTIVE
    
    def test_state_transitions_to_warning(self):
        """Test transition to WARNING state at threshold."""
        budget = Budget(max_steps=10, warning_threshold=0.8)
        # At 80% usage (8 steps out of 10)
        budget._steps = 8  # Direct access for testing
        assert budget.get_state() == BudgetState.WARNING
    
    def test_state_transitions_to_exceeded(self):
        """Test transition to EXCEEDED/EXHAUSTED state at limit."""
        budget = Budget(max_steps=10)
        budget._steps = 10
        # When only one limit is set and exceeded, it's EXHAUSTED
        assert budget.get_state() in (BudgetState.EXCEEDED, BudgetState.EXHAUSTED)
    
    def test_state_with_no_limits_stays_active(self):
        """Test that state stays ACTIVE when no limits are set."""
        budget = Budget()  # All limits at 0 (unlimited)
        budget._steps = 1000
        budget._llm_calls = 500
        assert budget.get_state() == BudgetState.ACTIVE
    
    def test_get_resource_states(self):
        """Test getting individual resource states."""
        budget = Budget(
            max_steps=10,
            max_llm_calls=5,
            warning_threshold=0.8,
        )
        budget._steps = 8  # At warning threshold
        budget._llm_calls = 5  # At limit
        
        states = budget.get_resource_states()
        assert states["steps"] == BudgetState.WARNING
        assert states["llm_calls"] == BudgetState.EXCEEDED
        assert states["tokens"] == BudgetState.ACTIVE  # No limit
    
    def test_time_based_warning(self):
        """Test time-based warning state."""
        budget = Budget(max_time_seconds=1.0, warning_threshold=0.8)
        time.sleep(0.85)  # Sleep past 80% threshold
        states = budget.get_resource_states()
        # Should be warning or exceeded depending on exact timing
        assert states.get("time_seconds") in (BudgetState.WARNING, BudgetState.EXCEEDED)


# ─────────────────────────────────────────────────────────────────────────────
# Test Budget - Enforcement
# ─────────────────────────────────────────────────────────────────────────────

class TestBudgetEnforcement:
    """Tests for budget enforcement mechanisms."""
    
    def test_step_limit_enforcement(self):
        """Test that step limit raises exception."""
        budget = Budget(max_steps=3)
        budget.record_step()
        budget.record_step()
        with pytest.raises(BudgetExceeded) as exc_info:
            budget.record_step()
        assert exc_info.value.resource == "steps"
        assert exc_info.value.current == 3
        assert exc_info.value.limit == 3
    
    def test_llm_call_limit_enforcement(self):
        """Test that LLM call limit raises exception."""
        budget = Budget(max_llm_calls=2)
        budget.record_llm_call()
        with pytest.raises(BudgetExceeded) as exc_info:
            budget.record_llm_call()
        assert exc_info.value.resource == "llm_calls"
    
    def test_token_limit_enforcement(self):
        """Test that token limit raises exception."""
        budget = Budget(max_tokens=100)
        budget.record_llm_call(tokens=50)
        with pytest.raises(BudgetExceeded) as exc_info:
            budget.record_llm_call(tokens=60)
        assert exc_info.value.resource == "tokens"
    
    def test_subprocess_limit_enforcement(self):
        """Test that subprocess limit raises exception."""
        budget = Budget(max_subprocess_calls=5)
        budget.record_subprocess_call(4)
        with pytest.raises(BudgetExceeded) as exc_info:
            budget.record_subprocess_call()
        assert exc_info.value.resource == "subprocess_calls"
    
    def test_time_limit_enforcement(self):
        """Test that time limit raises exception."""
        budget = Budget(max_time_seconds=0.1)
        time.sleep(0.15)
        with pytest.raises(BudgetExceeded) as exc_info:
            budget.check_time_budget()
        assert exc_info.value.resource == "time_seconds"
    
    def test_no_enforcement_when_unlimited(self):
        """Test that no exception is raised when limits are 0."""
        budget = Budget()  # All limits at 0 (unlimited)
        # These should all succeed
        for _ in range(100):
            budget.record_step()
            budget.record_llm_call(tokens=1000)
            budget.record_subprocess_call()
        assert budget.steps == 100
        assert budget.llm_calls == 100
    
    def test_is_within_budget(self):
        """Test is_within_budget method."""
        budget = Budget(max_steps=10)
        assert budget.is_within_budget() is True
        budget._steps = 8
        assert budget.is_within_budget() is True  # WARNING is still within budget
        budget._steps = 10
        assert budget.is_within_budget() is False


# ─────────────────────────────────────────────────────────────────────────────
# Test Budget - Callbacks
# ─────────────────────────────────────────────────────────────────────────────

class TestBudgetCallbacks:
    """Tests for budget callback mechanisms."""
    
    def test_warning_callback(self):
        """Test that warning callback is called."""
        budget = Budget(max_steps=10, warning_threshold=0.8)
        callback_calls = []
        
        def on_warning(resource, current, limit):
            callback_calls.append((resource, current, limit))
        
        budget.on_warning(on_warning)
        budget._steps = 7  # Just below threshold
        budget.record_step()  # Now at 8, triggers warning
        
        assert len(callback_calls) == 1
        assert callback_calls[0] == ("steps", 8, 10)
    
    def test_exceeded_callback(self):
        """Test that exceeded callback is called."""
        budget = Budget(max_steps=3)
        callback_calls = []
        
        def on_exceeded(resource, current, limit):
            callback_calls.append((resource, current, limit))
        
        budget.on_exceeded(on_exceeded)
        budget.record_step()
        budget.record_step()
        
        with pytest.raises(BudgetExceeded):
            budget.record_step()
        
        assert len(callback_calls) == 1
        assert callback_calls[0] == ("steps", 3, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Test Budget - Usage Summary
# ─────────────────────────────────────────────────────────────────────────────

class TestBudgetUsageSummary:
    """Tests for budget usage summary."""
    
    def test_get_usage_summary(self):
        """Test getting usage summary."""
        budget = Budget(
            max_steps=10,
            max_llm_calls=5,
            max_tokens=1000,
            max_subprocess_calls=20,
        )
        budget.record_step(3)
        budget.record_llm_call(tokens=200)
        budget.record_subprocess_call(5)
        
        summary = budget.get_usage_summary()
        
        assert summary["steps"]["current"] == 3
        assert summary["steps"]["limit"] == 10
        assert summary["steps"]["remaining"] == 7
        assert summary["steps"]["percentage"] == 30.0
        
        assert summary["llm_calls"]["current"] == 1
        assert summary["llm_calls"]["limit"] == 5
        
        assert summary["tokens"]["current"] == 200
        assert summary["tokens"]["limit"] == 1000
        
        assert summary["subprocess_calls"]["current"] == 5
        assert summary["subprocess_calls"]["limit"] == 20


# ─────────────────────────────────────────────────────────────────────────────
# Test Budget - Thread Safety
# ─────────────────────────────────────────────────────────────────────────────

class TestBudgetThreadSafety:
    """Tests for budget thread safety."""
    
    def test_concurrent_step_recording(self):
        """Test concurrent step recording is thread-safe."""
        budget = Budget(max_steps=1000)
        
        def record_steps():
            for _ in range(100):
                budget.record_step()
        
        threads = [threading.Thread(target=record_steps) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert budget.steps == 500  # 5 threads * 100 steps each
    
    def test_concurrent_llm_recording(self):
        """Test concurrent LLM call recording is thread-safe."""
        budget = Budget(max_llm_calls=1000, max_tokens=100000)
        
        def record_calls():
            for _ in range(50):
                budget.record_llm_call(tokens=10)
        
        threads = [threading.Thread(target=record_calls) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert budget.llm_calls == 250  # 5 threads * 50 calls each
        assert budget.tokens == 2500   # 5 threads * 50 calls * 10 tokens


# ─────────────────────────────────────────────────────────────────────────────
# Test Global Budget
# ─────────────────────────────────────────────────────────────────────────────

class TestGlobalBudget:
    """Tests for global budget functionality."""
    
    def test_set_and_get_global_budget(self):
        """Test setting and getting global budget."""
        budget = Budget(max_steps=10)
        set_global_budget(budget)
        
        retrieved = get_global_budget()
        assert retrieved is budget
        
        # Cleanup
        set_global_budget(None)
    
    def test_get_global_budget_returns_none_when_not_set(self):
        """Test that get returns None when not set."""
        set_global_budget(None)
        assert get_global_budget() is None
    
    def test_record_subprocess_call_global(self):
        """Test global subprocess call recording."""
        budget = Budget(max_subprocess_calls=10)
        set_global_budget(budget)
        
        record_subprocess_call_global()
        assert budget.subprocess_calls == 1
        
        # Cleanup
        set_global_budget(None)
    
    def test_record_subprocess_call_global_no_budget(self):
        """Test global subprocess recording with no budget set."""
        set_global_budget(None)
        # Should not raise
        record_subprocess_call_global()
    
    def test_record_llm_call_global(self):
        """Test global LLM call recording."""
        budget = Budget(max_llm_calls=10)
        set_global_budget(budget)
        
        record_llm_call_global(tokens=100)
        assert budget.llm_calls == 1
        assert budget.tokens == 100
        
        # Cleanup
        set_global_budget(None)
    
    def test_check_time_budget_global(self):
        """Test global time budget checking."""
        budget = Budget(max_time_seconds=0.1)
        set_global_budget(budget)
        
        time.sleep(0.15)
        with pytest.raises(BudgetExceeded):
            check_time_budget_global()
        
        # Cleanup
        set_global_budget(None)


# ─────────────────────────────────────────────────────────────────────────────
# Test BudgetConfig Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestBudgetConfigIntegration:
    """Tests for BudgetConfig and ControllerConfig integration."""
    
    def test_budget_config_defaults(self):
        """Test BudgetConfig default values."""
        config = BudgetConfig()
        assert config.max_steps == 0
        assert config.max_llm_calls == 0
        assert config.max_tokens == 0
        assert config.max_time_seconds == 0
        assert config.max_subprocess_calls == 0
        assert config.warning_threshold == 0.8
    
    def test_budget_config_custom_values(self):
        """Test BudgetConfig with custom values."""
        config = BudgetConfig(
            max_steps=15,
            max_llm_calls=10,
            max_tokens=5000,
            max_time_seconds=300,
            max_subprocess_calls=50,
            warning_threshold=0.9,
        )
        assert config.max_steps == 15
        assert config.max_llm_calls == 10
        assert config.max_tokens == 5000
        assert config.max_time_seconds == 300
        assert config.max_subprocess_calls == 50
        assert config.warning_threshold == 0.9
    
    def test_controller_config_has_budget(self):
        """Test that ControllerConfig includes BudgetConfig."""
        config = ControllerConfig(github_url="https://github.com/test/repo")
        assert hasattr(config, "budget")
        assert isinstance(config.budget, BudgetConfig)
    
    def test_controller_config_with_custom_budget(self):
        """Test ControllerConfig with custom BudgetConfig."""
        budget_config = BudgetConfig(max_steps=20)
        config = ControllerConfig(github_url="https://github.com/test/repo", budget=budget_config)
        assert config.budget.max_steps == 20


# ─────────────────────────────────────────────────────────────────────────────
# Test Context Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestContextBudgetIntegration:
    """Tests for context and budget integration."""
    
    def test_context_has_budget_property(self):
        """Test that ControllerContext has budget property."""
        from rfsn_controller.context import ControllerContext, EventLog
        
        config = ControllerConfig(github_url="https://github.com/test/repo")
        event_log = EventLog(path=Path("/tmp/test_events.jsonl"))
        ctx = ControllerContext(config=config, event_log=event_log)
        
        # Initially None
        assert ctx.budget is None
    
    def test_context_budget_setter(self):
        """Test setting budget on context."""
        from rfsn_controller.context import ControllerContext, EventLog
        
        config = ControllerConfig(github_url="https://github.com/test/repo")
        event_log = EventLog(path=Path("/tmp/test_events.jsonl"))
        ctx = ControllerContext(config=config, event_log=event_log)
        
        budget = Budget(max_steps=10)
        ctx.budget = budget
        
        assert ctx.budget is budget
    
    def test_create_context_initializes_budget(self, tmp_path):
        """Test that create_context initializes budget from config."""
        from rfsn_controller.context import create_context
        
        budget_config = BudgetConfig(max_steps=15, max_llm_calls=10)
        config = ControllerConfig(
            github_url="https://github.com/test/repo",
            budget=budget_config,
            output_dir=str(tmp_path),
        )
        
        ctx = create_context(config)
        
        assert ctx.budget is not None
        assert ctx.budget.max_steps == 15
        assert ctx.budget.max_llm_calls == 10
        
        # Cleanup global budget
        set_global_budget(None)
    
    def test_create_context_no_budget_when_all_zero(self, tmp_path):
        """Test that create_context doesn't create budget when all limits are 0."""
        from rfsn_controller.context import create_context
        
        config = ControllerConfig(
            github_url="https://github.com/test/repo",
            output_dir=str(tmp_path),
        )  # Default budget with all 0
        ctx = create_context(config)
        
        assert ctx.budget is None


# ─────────────────────────────────────────────────────────────────────────────
# Test Integration with exec_utils
# ─────────────────────────────────────────────────────────────────────────────

class TestExecUtilsBudgetIntegration:
    """Tests for exec_utils budget integration."""
    
    def test_safe_run_records_subprocess_call(self, tmp_path):
        """Test that safe_run records subprocess calls in budget."""
        from rfsn_controller.exec_utils import safe_run
        
        budget = Budget(max_subprocess_calls=10)
        set_global_budget(budget)
        
        # Run a simple command
        safe_run(["echo", "test"], cwd=str(tmp_path))
        
        assert budget.subprocess_calls == 1
        
        # Cleanup
        set_global_budget(None)
    
    def test_safe_run_respects_subprocess_limit(self, tmp_path):
        """Test that safe_run respects subprocess call limit."""
        from rfsn_controller.exec_utils import safe_run
        
        budget = Budget(max_subprocess_calls=3)
        set_global_budget(budget)
        
        # First three should succeed (limit is 3, exception raised when reaching limit)
        safe_run(["echo", "test1"], cwd=str(tmp_path))
        safe_run(["echo", "test2"], cwd=str(tmp_path))
        
        # Third should raise (at limit)
        with pytest.raises(BudgetExceeded) as exc_info:
            safe_run(["echo", "test3"], cwd=str(tmp_path))
        
        assert exc_info.value.resource == "subprocess_calls"
        
        # Cleanup
        set_global_budget(None)


# ─────────────────────────────────────────────────────────────────────────────
# Test Integration with sandbox
# ─────────────────────────────────────────────────────────────────────────────

class TestSandboxBudgetIntegration:
    """Tests for sandbox budget integration."""
    
    def test_sandbox_run_records_subprocess_call(self, tmp_path):
        """Test that sandbox _run records subprocess calls in budget."""
        from rfsn_controller.sandbox import _run
        
        budget = Budget(max_subprocess_calls=10)
        set_global_budget(budget)
        
        # Run a simple command (echo should be allowed)
        exit_code, stdout, stderr = _run("echo test", str(tmp_path))
        
        assert budget.subprocess_calls == 1
        
        # Cleanup
        set_global_budget(None)


# ─────────────────────────────────────────────────────────────────────────────
# Test Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

class TestBudgetEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_budget_at_exact_warning_threshold(self):
        """Test budget at exact warning threshold."""
        budget = Budget(max_steps=10, warning_threshold=0.8)
        budget._steps = 8  # Exactly at 80%
        assert budget.get_state() == BudgetState.WARNING
    
    def test_budget_just_below_warning_threshold(self):
        """Test budget just below warning threshold."""
        budget = Budget(max_steps=10, warning_threshold=0.8)
        budget._steps = 7  # Just below 80%
        assert budget.get_state() == BudgetState.ACTIVE
    
    def test_budget_at_exact_limit(self):
        """Test budget at exact limit."""
        budget = Budget(max_steps=10)
        budget._steps = 10
        # When only one limit is set and exceeded, it's EXHAUSTED
        assert budget.get_state() in (BudgetState.EXCEEDED, BudgetState.EXHAUSTED)
    
    def test_budget_over_limit(self):
        """Test budget over limit (shouldn't normally happen but handle gracefully)."""
        budget = Budget(max_steps=10)
        budget._steps = 15  # Over limit
        # When only one limit is set and exceeded, it's EXHAUSTED
        assert budget.get_state() in (BudgetState.EXCEEDED, BudgetState.EXHAUSTED)
    
    def test_zero_warning_threshold(self):
        """Test with warning threshold of 0."""
        budget = Budget(max_steps=10, warning_threshold=0)
        budget._steps = 1
        # Any usage should trigger warning
        assert budget.get_state() == BudgetState.WARNING
    
    def test_one_hundred_percent_warning_threshold(self):
        """Test with warning threshold of 1.0 (100%)."""
        budget = Budget(max_steps=10, warning_threshold=1.0)
        budget._steps = 9
        # Should still be ACTIVE even at 90%
        assert budget.get_state() == BudgetState.ACTIVE
    
    def test_negative_remaining_prevented(self):
        """Test that remaining doesn't go negative."""
        budget = Budget(max_steps=5)
        budget._steps = 10  # Over limit
        assert budget.remaining_steps == 0  # Should be 0, not negative
    
    def test_time_remaining_returns_none_when_unlimited(self):
        """Test that remaining time is None when unlimited."""
        budget = Budget()  # max_time_seconds = 0
        assert budget.remaining_time_seconds is None
    
    def test_time_remaining_calculation(self):
        """Test time remaining calculation."""
        budget = Budget(max_time_seconds=1.0)
        time.sleep(0.3)
        remaining = budget.remaining_time_seconds
        assert remaining is not None
        assert 0.5 <= remaining <= 0.75  # Allow for timing variance


# ─────────────────────────────────────────────────────────────────────────────
# Test config_from_cli_args Budget Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigFromCliArgs:
    """Tests for config_from_cli_args budget integration."""
    
    def test_config_from_cli_args_with_budget(self):
        """Test config_from_cli_args with budget arguments."""
        from rfsn_controller.config import config_from_cli_args
        
        class Args:
            repo = "https://github.com/test/repo"
            ref = None
            feature_mode = "repair"
            test = "pytest"
            steps = 10
            max_steps_without_progress = 5
            sandbox_image = "python:3.11"
            network_access = False
            budget_max_steps = 20
            budget_max_llm_calls = 15
            budget_max_tokens = 5000
            budget_max_time_seconds = 300.0
            budget_max_subprocess_calls = 100
            budget_warning_threshold = 0.75
            learning_db = None
            policy_mode = "off"
            planner_mode = "off"
            repo_index = False
            seed = 42
            model = "deepseek-chat"
            collect_finetuning_data = False
            parallel_patches = False
            enable_llm_cache = False
            no_eval = False
        
        config = config_from_cli_args(Args())
        
        assert config.budget.max_steps == 20
        assert config.budget.max_llm_calls == 15
        assert config.budget.max_tokens == 5000
        assert config.budget.max_time_seconds == 300.0
        assert config.budget.max_subprocess_calls == 100
        assert config.budget.warning_threshold == 0.75
    
    def test_config_from_cli_args_default_budget(self):
        """Test config_from_cli_args with default budget values."""
        from rfsn_controller.config import config_from_cli_args
        
        class Args:
            repo = ""
            ref = None
            feature_mode = "repair"
            test = "pytest"
            steps = 12
            max_steps_without_progress = 10
            sandbox_image = "python:3.11-slim"
            network_access = False
            learning_db = None
            policy_mode = "off"
            planner_mode = "off"
            repo_index = False
            seed = 1337
            model = "deepseek-chat"
            collect_finetuning_data = False
            parallel_patches = False
            enable_llm_cache = False
            no_eval = False
        
        config = config_from_cli_args(Args())
        
        # Budget should use defaults (all 0)
        assert config.budget.max_steps == 0
        assert config.budget.max_llm_calls == 0
        assert config.budget.max_tokens == 0
        assert config.budget.max_time_seconds == 0
        assert config.budget.max_subprocess_calls == 0
        assert config.budget.warning_threshold == 0.8


# Mark all tests as unit tests
pytestmark = pytest.mark.unit
