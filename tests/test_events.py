"""Comprehensive tests for the structured events system.

Tests cover:
- Event and EventType/EventSeverity enums
- EventLogger collection and filtering
- EventStore persistence and retrieval
- EventQuery filtering capabilities
- Integration with existing components (budget, exec_utils, etc.)
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from rfsn_controller.events import (
    Event,
    EventLogger,
    EventQuery,
    EventSeverity,
    EventStore,
    EventType,
    create_event,
    get_global_event_logger,
    log_budget_exceeded_global,
    log_budget_warning_global,
    log_controller_step_global,
    log_error_global,
    log_event_global,
    log_llm_call_global,
    log_security_violation_global,
    log_subprocess_exec_global,
    set_global_event_logger,
)

# =============================================================================
# Test Event Enums
# =============================================================================

class TestEventType:
    """Tests for EventType enum."""
    
    def test_event_type_values(self):
        """EventType has expected values."""
        assert EventType.CONTROLLER_STEP.value == "controller_step"
        assert EventType.LLM_CALL.value == "llm_call"
        assert EventType.BUDGET_WARNING.value == "budget_warning"
        assert EventType.BUDGET_EXCEEDED.value == "budget_exceeded"
        assert EventType.SECURITY_VIOLATION.value == "security_violation"
        assert EventType.SUBPROCESS_EXEC.value == "subprocess_exec"
        assert EventType.FEATURE_REGISTERED.value == "feature_registered"
        assert EventType.ERROR.value == "error"
    
    def test_all_event_types(self):
        """All expected event types exist."""
        types = list(EventType)
        assert len(types) == 8


class TestEventSeverity:
    """Tests for EventSeverity enum."""
    
    def test_severity_values(self):
        """EventSeverity has expected values."""
        assert EventSeverity.DEBUG.value == "debug"
        assert EventSeverity.INFO.value == "info"
        assert EventSeverity.WARNING.value == "warning"
        assert EventSeverity.ERROR.value == "error"
        assert EventSeverity.CRITICAL.value == "critical"
    
    def test_all_severity_levels(self):
        """All expected severity levels exist."""
        levels = list(EventSeverity)
        assert len(levels) == 5


# =============================================================================
# Test Event Dataclass
# =============================================================================

class TestEvent:
    """Tests for Event dataclass."""
    
    def test_create_event(self):
        """Event can be created with required fields."""
        event = Event(
            timestamp="2026-01-20T12:00:00+00:00",
            event_type=EventType.CONTROLLER_STEP,
            source="test",
            data={"step": 1},
        )
        assert event.event_type == EventType.CONTROLLER_STEP
        assert event.source == "test"
        assert event.data == {"step": 1}
        assert event.severity == EventSeverity.INFO
    
    def test_event_with_all_fields(self):
        """Event can be created with all fields."""
        event = Event(
            timestamp="2026-01-20T12:00:00+00:00",
            event_type=EventType.ERROR,
            source="test_module",
            data={"error": "test error"},
            severity=EventSeverity.ERROR,
            event_id="test-id-123",
            run_id="run-456",
            correlation_id="corr-789",
        )
        assert event.event_id == "test-id-123"
        assert event.run_id == "run-456"
        assert event.correlation_id == "corr-789"
    
    def test_event_to_dict(self):
        """Event can be converted to dictionary."""
        event = Event(
            timestamp="2026-01-20T12:00:00+00:00",
            event_type=EventType.LLM_CALL,
            source="llm",
            data={"tokens": 100},
            severity=EventSeverity.INFO,
        )
        d = event.to_dict()
        assert d["event_type"] == "llm_call"
        assert d["source"] == "llm"
        assert d["severity"] == "info"
        assert d["data"]["tokens"] == 100
    
    def test_event_to_json(self):
        """Event can be serialized to JSON."""
        event = Event(
            timestamp="2026-01-20T12:00:00+00:00",
            event_type=EventType.SUBPROCESS_EXEC,
            source="exec",
            data={"command": ["echo", "test"]},
        )
        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["event_type"] == "subprocess_exec"
        assert parsed["data"]["command"] == ["echo", "test"]
    
    def test_event_from_dict(self):
        """Event can be created from dictionary."""
        d = {
            "timestamp": "2026-01-20T12:00:00+00:00",
            "event_type": "budget_warning",
            "source": "budget",
            "data": {"resource": "steps", "current": 80, "limit": 100},
            "severity": "warning",
            "event_id": "test-id",
        }
        event = Event.from_dict(d)
        assert event.event_type == EventType.BUDGET_WARNING
        assert event.severity == EventSeverity.WARNING
        assert event.data["current"] == 80
    
    def test_event_from_json(self):
        """Event can be deserialized from JSON."""
        json_str = '{"timestamp": "2026-01-20T12:00:00+00:00", "event_type": "error", "source": "test", "data": {}, "severity": "error"}'
        event = Event.from_json(json_str)
        assert event.event_type == EventType.ERROR
        assert event.severity == EventSeverity.ERROR
    
    def test_create_event_factory(self):
        """create_event factory creates events with timestamp."""
        event = create_event(
            event_type=EventType.FEATURE_REGISTERED,
            source="feature_manager",
            data={"feature": "test_feature"},
        )
        assert event.timestamp  # Should have a timestamp
        assert event.event_id  # Should have an ID
        assert event.event_type == EventType.FEATURE_REGISTERED


# =============================================================================
# Test EventLogger
# =============================================================================

class TestEventLogger:
    """Tests for EventLogger class."""
    
    def test_create_logger(self):
        """EventLogger can be created."""
        logger = EventLogger()
        assert logger.event_count == 0
        assert logger.run_id is not None
    
    def test_create_logger_with_run_id(self):
        """EventLogger can be created with custom run_id."""
        logger = EventLogger(run_id="custom-run-123")
        assert logger.run_id == "custom-run-123"
    
    def test_log_event(self):
        """Events can be logged."""
        logger = EventLogger()
        event = logger.log(
            EventType.CONTROLLER_STEP,
            "test",
            {"step": 1},
        )
        assert logger.event_count == 1
        assert event.event_type == EventType.CONTROLLER_STEP
    
    def test_log_multiple_events(self):
        """Multiple events can be logged."""
        logger = EventLogger()
        logger.log(EventType.CONTROLLER_STEP, "test", {"step": 1})
        logger.log(EventType.LLM_CALL, "llm", {"model": "test"})
        logger.log(EventType.SUBPROCESS_EXEC, "exec", {"command": ["ls"]})
        assert logger.event_count == 3
    
    def test_log_controller_step(self):
        """Controller step events can be logged."""
        logger = EventLogger()
        event = logger.log_controller_step(step_number=5, phase="REPAIR")
        assert event.event_type == EventType.CONTROLLER_STEP
        assert event.data["step_number"] == 5
        assert event.data["phase"] == "REPAIR"
    
    def test_log_llm_call(self):
        """LLM call events can be logged."""
        logger = EventLogger()
        event = logger.log_llm_call(
            model="gpt-4",
            tokens_prompt=100,
            tokens_completion=50,
            latency_ms=1500.0,
            success=True,
        )
        assert event.event_type == EventType.LLM_CALL
        assert event.data["model"] == "gpt-4"
        assert event.data["tokens_total"] == 150
        assert event.data["success"] is True
    
    def test_log_llm_call_failure(self):
        """Failed LLM call events are logged with ERROR severity."""
        logger = EventLogger()
        event = logger.log_llm_call(
            model="gpt-4",
            tokens_prompt=0,
            tokens_completion=0,
            latency_ms=500.0,
            success=False,
            error="API rate limit",
        )
        assert event.severity == EventSeverity.ERROR
        assert event.data["error"] == "API rate limit"
    
    def test_log_budget_warning(self):
        """Budget warning events can be logged."""
        logger = EventLogger()
        event = logger.log_budget_warning(
            resource="steps",
            current=80,
            limit=100,
            percentage=80.0,
        )
        assert event.event_type == EventType.BUDGET_WARNING
        assert event.severity == EventSeverity.WARNING
        assert event.data["resource"] == "steps"
    
    def test_log_budget_exceeded(self):
        """Budget exceeded events can be logged."""
        logger = EventLogger()
        event = logger.log_budget_exceeded(
            resource="tokens",
            current=10000,
            limit=10000,
        )
        assert event.event_type == EventType.BUDGET_EXCEEDED
        assert event.severity == EventSeverity.CRITICAL
    
    def test_log_security_violation(self):
        """Security violation events can be logged."""
        logger = EventLogger()
        event = logger.log_security_violation(
            violation_type="shell_true",
            file_path="/path/to/file.py",
            line_number=42,
            message="shell=True detected",
            severity="critical",
        )
        assert event.event_type == EventType.SECURITY_VIOLATION
        assert event.severity == EventSeverity.CRITICAL
        assert event.data["violation_type"] == "shell_true"
    
    def test_log_subprocess_exec(self):
        """Subprocess execution events can be logged."""
        logger = EventLogger()
        event = logger.log_subprocess_exec(
            command=["pytest", "-v"],
            exit_code=0,
            success=True,
            duration_ms=2500.0,
            cwd="/project",
        )
        assert event.event_type == EventType.SUBPROCESS_EXEC
        assert event.data["command"] == ["pytest", "-v"]
        assert event.data["success"] is True
    
    def test_log_error(self):
        """Error events can be logged."""
        logger = EventLogger()
        event = logger.log_error(
            source="test_module",
            error_type="ValueError",
            message="Invalid value",
            traceback="Traceback...",
        )
        assert event.event_type == EventType.ERROR
        assert event.severity == EventSeverity.ERROR
        assert event.data["error_type"] == "ValueError"
    
    def test_get_events(self):
        """Events can be retrieved."""
        logger = EventLogger()
        logger.log(EventType.CONTROLLER_STEP, "test", {"step": 1})
        logger.log(EventType.LLM_CALL, "llm", {"model": "test"})
        
        events = logger.events
        assert len(events) == 2
    
    def test_get_events_by_type(self):
        """Events can be filtered by type."""
        logger = EventLogger()
        logger.log(EventType.CONTROLLER_STEP, "test", {"step": 1})
        logger.log(EventType.LLM_CALL, "llm", {"model": "test"})
        logger.log(EventType.CONTROLLER_STEP, "test", {"step": 2})
        
        step_events = logger.get_events_by_type(EventType.CONTROLLER_STEP)
        assert len(step_events) == 2
    
    def test_get_events_by_severity(self):
        """Events can be filtered by severity."""
        logger = EventLogger()
        logger.log(EventType.CONTROLLER_STEP, "test", {}, EventSeverity.INFO)
        logger.log(EventType.BUDGET_WARNING, "budget", {}, EventSeverity.WARNING)
        logger.log(EventType.ERROR, "test", {}, EventSeverity.ERROR)
        
        warning_plus = logger.get_events_by_severity(EventSeverity.WARNING)
        assert len(warning_plus) == 2
    
    def test_clear_events(self):
        """Events can be cleared."""
        logger = EventLogger()
        logger.log(EventType.CONTROLLER_STEP, "test", {})
        logger.log(EventType.LLM_CALL, "llm", {})
        assert logger.event_count == 2
        
        logger.clear()
        assert logger.event_count == 0
    
    def test_max_events_limit(self):
        """Max events limit is enforced."""
        logger = EventLogger(max_events=5)
        for i in range(10):
            logger.log(EventType.CONTROLLER_STEP, "test", {"step": i})
        
        assert logger.event_count == 5
        # Should keep the most recent events
        events = logger.events
        assert events[0].data["step"] == 5
    
    def test_register_callback(self):
        """Callbacks can be registered for events."""
        logger = EventLogger()
        received_events = []
        
        def callback(event):
            received_events.append(event)
        
        logger.register_callback(callback)
        logger.log(EventType.CONTROLLER_STEP, "test", {"step": 1})
        logger.log(EventType.LLM_CALL, "llm", {})
        
        assert len(received_events) == 2
    
    def test_unregister_callback(self):
        """Callbacks can be unregistered."""
        logger = EventLogger()
        received_events = []
        
        def callback(event):
            received_events.append(event)
        
        logger.register_callback(callback)
        logger.log(EventType.CONTROLLER_STEP, "test", {"step": 1})
        
        logger.unregister_callback(callback)
        logger.log(EventType.LLM_CALL, "llm", {})
        
        assert len(received_events) == 1
    
    def test_thread_safety(self):
        """EventLogger is thread-safe."""
        logger = EventLogger(max_events=0)
        errors = []
        
        def log_events(prefix, count):
            try:
                for i in range(count):
                    logger.log(EventType.CONTROLLER_STEP, prefix, {"i": i})
            except Exception as e:
                errors.append(str(e))
        
        threads = [
            threading.Thread(target=log_events, args=(f"thread-{i}", 100))
            for i in range(5)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert logger.event_count == 500


# =============================================================================
# Test EventStore
# =============================================================================

class TestEventStore:
    """Tests for EventStore persistence."""
    
    def test_create_store(self, tmp_path):
        """EventStore can be created."""
        store_path = tmp_path / "events.jsonl"
        store = EventStore(store_path)
        assert store.storage_path == store_path
    
    def test_append_event(self, tmp_path):
        """Events can be appended to store."""
        store_path = tmp_path / "events.jsonl"
        store = EventStore(store_path)
        
        event = create_event(
            EventType.CONTROLLER_STEP,
            "test",
            {"step": 1},
        )
        store.append(event)
        
        assert store.get_event_count() == 1
    
    def test_append_batch(self, tmp_path):
        """Multiple events can be appended at once."""
        store_path = tmp_path / "events.jsonl"
        store = EventStore(store_path)
        
        events = [
            create_event(EventType.CONTROLLER_STEP, "test", {"step": i})
            for i in range(5)
        ]
        store.append_batch(events)
        
        assert store.get_event_count() == 5
    
    def test_read_all(self, tmp_path):
        """All events can be read from store."""
        store_path = tmp_path / "events.jsonl"
        store = EventStore(store_path)
        
        for i in range(3):
            event = create_event(EventType.LLM_CALL, "llm", {"call": i})
            store.append(event)
        
        events = store.read_all()
        assert len(events) == 3
        assert all(e.event_type == EventType.LLM_CALL for e in events)
    
    def test_iter_events(self, tmp_path):
        """Events can be iterated from store."""
        store_path = tmp_path / "events.jsonl"
        store = EventStore(store_path)
        
        for i in range(5):
            event = create_event(EventType.SUBPROCESS_EXEC, "exec", {"i": i})
            store.append(event)
        
        count = 0
        for event in store.iter_events():
            count += 1
            assert event.event_type == EventType.SUBPROCESS_EXEC
        
        assert count == 5
    
    def test_clear(self, tmp_path):
        """Store can be cleared."""
        store_path = tmp_path / "events.jsonl"
        store = EventStore(store_path)
        
        for i in range(3):
            event = create_event(EventType.CONTROLLER_STEP, "test", {"step": i})
            store.append(event)
        
        assert store.get_event_count() == 3
        
        store.clear()
        assert store.get_event_count() == 0
    
    def test_rotate(self, tmp_path):
        """Store can be rotated to keep recent events."""
        store_path = tmp_path / "events.jsonl"
        store = EventStore(store_path)
        
        for i in range(10):
            event = create_event(EventType.CONTROLLER_STEP, "test", {"step": i})
            store.append(event)
        
        removed = store.rotate(max_events=5)
        assert removed == 5
        
        events = store.read_all()
        assert len(events) == 5
        # Should keep most recent (steps 5-9)
        assert events[0].data["step"] == 5
    
    def test_empty_store_read(self, tmp_path):
        """Reading from empty/nonexistent store returns empty list."""
        store_path = tmp_path / "nonexistent.jsonl"
        store = EventStore(store_path)
        
        events = store.read_all()
        assert events == []
    
    def test_persistence(self, tmp_path):
        """Events persist across store instances."""
        store_path = tmp_path / "events.jsonl"
        
        # Write with one instance
        store1 = EventStore(store_path)
        event = create_event(EventType.ERROR, "test", {"msg": "test error"})
        store1.append(event)
        
        # Read with another instance
        store2 = EventStore(store_path)
        events = store2.read_all()
        assert len(events) == 1
        assert events[0].data["msg"] == "test error"


# =============================================================================
# Test EventQuery
# =============================================================================

class TestEventQuery:
    """Tests for EventQuery filtering."""
    
    def test_query_no_filters(self):
        """Query with no filters matches all events."""
        query = EventQuery()
        event = create_event(EventType.CONTROLLER_STEP, "test", {})
        assert query.matches(event)
    
    def test_query_by_event_type(self):
        """Query can filter by event type."""
        query = EventQuery(event_types={EventType.LLM_CALL})
        
        llm_event = create_event(EventType.LLM_CALL, "llm", {})
        step_event = create_event(EventType.CONTROLLER_STEP, "test", {})
        
        assert query.matches(llm_event)
        assert not query.matches(step_event)
    
    def test_query_by_multiple_types(self):
        """Query can filter by multiple event types."""
        query = EventQuery(event_types={EventType.LLM_CALL, EventType.SUBPROCESS_EXEC})
        
        llm_event = create_event(EventType.LLM_CALL, "llm", {})
        exec_event = create_event(EventType.SUBPROCESS_EXEC, "exec", {})
        step_event = create_event(EventType.CONTROLLER_STEP, "test", {})
        
        assert query.matches(llm_event)
        assert query.matches(exec_event)
        assert not query.matches(step_event)
    
    def test_query_by_severity(self):
        """Query can filter by minimum severity."""
        query = EventQuery(min_severity=EventSeverity.WARNING)
        
        info_event = create_event(
            EventType.CONTROLLER_STEP, "test", {}, EventSeverity.INFO
        )
        warning_event = create_event(
            EventType.BUDGET_WARNING, "budget", {}, EventSeverity.WARNING
        )
        error_event = create_event(
            EventType.ERROR, "test", {}, EventSeverity.ERROR
        )
        
        assert not query.matches(info_event)
        assert query.matches(warning_event)
        assert query.matches(error_event)
    
    def test_query_by_source(self):
        """Query can filter by source."""
        query = EventQuery(sources={"llm", "budget"})
        
        llm_event = create_event(EventType.LLM_CALL, "llm", {})
        budget_event = create_event(EventType.BUDGET_WARNING, "budget", {})
        exec_event = create_event(EventType.SUBPROCESS_EXEC, "exec", {})
        
        assert query.matches(llm_event)
        assert query.matches(budget_event)
        assert not query.matches(exec_event)
    
    def test_query_by_time_range(self):
        """Query can filter by time range."""
        query = EventQuery(
            start_time="2026-01-20T10:00:00+00:00",
            end_time="2026-01-20T14:00:00+00:00",
        )
        
        before = Event(
            timestamp="2026-01-20T09:00:00+00:00",
            event_type=EventType.CONTROLLER_STEP,
            source="test",
            data={},
        )
        during = Event(
            timestamp="2026-01-20T12:00:00+00:00",
            event_type=EventType.CONTROLLER_STEP,
            source="test",
            data={},
        )
        after = Event(
            timestamp="2026-01-20T15:00:00+00:00",
            event_type=EventType.CONTROLLER_STEP,
            source="test",
            data={},
        )
        
        assert not query.matches(before)
        assert query.matches(during)
        assert not query.matches(after)
    
    def test_query_by_data_filters(self):
        """Query can filter by data field values."""
        query = EventQuery(data_filters={"model": "gpt-4"})
        
        matching = create_event(EventType.LLM_CALL, "llm", {"model": "gpt-4"})
        non_matching = create_event(EventType.LLM_CALL, "llm", {"model": "claude"})
        
        assert query.matches(matching)
        assert not query.matches(non_matching)
    
    def test_query_filter_list(self):
        """Query can filter a list of events."""
        events = [
            create_event(EventType.CONTROLLER_STEP, "test", {"step": 1}),
            create_event(EventType.LLM_CALL, "llm", {"model": "gpt-4"}),
            create_event(EventType.CONTROLLER_STEP, "test", {"step": 2}),
            create_event(EventType.ERROR, "test", {"error": "test"}),
        ]
        
        query = EventQuery(event_types={EventType.CONTROLLER_STEP})
        filtered = query.filter(events)
        
        assert len(filtered) == 2
        assert all(e.event_type == EventType.CONTROLLER_STEP for e in filtered)
    
    def test_query_with_limit(self):
        """Query respects limit."""
        events = [
            create_event(EventType.CONTROLLER_STEP, "test", {"step": i})
            for i in range(10)
        ]
        
        query = EventQuery(limit=3)
        filtered = query.filter(events)
        
        assert len(filtered) == 3
    
    def test_query_filter_store(self, tmp_path):
        """Query can filter events from store."""
        store_path = tmp_path / "events.jsonl"
        store = EventStore(store_path)
        
        for i in range(5):
            event = create_event(EventType.CONTROLLER_STEP, "test", {"step": i})
            store.append(event)
        for i in range(3):
            event = create_event(EventType.LLM_CALL, "llm", {"call": i})
            store.append(event)
        
        query = EventQuery(event_types={EventType.LLM_CALL})
        filtered = query.filter_store(store)
        
        assert len(filtered) == 3
        assert all(e.event_type == EventType.LLM_CALL for e in filtered)


# =============================================================================
# Test Global Event Logger
# =============================================================================

class TestGlobalEventLogger:
    """Tests for global event logger functionality."""
    
    def test_set_and_get_global_logger(self):
        """Global logger can be set and retrieved."""
        original = get_global_event_logger()
        try:
            logger = EventLogger(run_id="test-run")
            set_global_event_logger(logger)
            
            retrieved = get_global_event_logger()
            assert retrieved is logger
            assert retrieved.run_id == "test-run"
        finally:
            set_global_event_logger(original)
    
    def test_clear_global_logger(self):
        """Global logger can be cleared."""
        original = get_global_event_logger()
        try:
            logger = EventLogger()
            set_global_event_logger(logger)
            
            set_global_event_logger(None)
            assert get_global_event_logger() is None
        finally:
            set_global_event_logger(original)
    
    def test_log_event_global(self):
        """Events can be logged globally."""
        original = get_global_event_logger()
        try:
            logger = EventLogger()
            set_global_event_logger(logger)
            
            event = log_event_global(
                EventType.CONTROLLER_STEP,
                "test",
                {"step": 1},
            )
            
            assert event is not None
            assert logger.event_count == 1
        finally:
            set_global_event_logger(original)
    
    def test_log_event_global_no_logger(self):
        """Global logging returns None when no logger set."""
        original = get_global_event_logger()
        try:
            set_global_event_logger(None)
            
            event = log_event_global(
                EventType.CONTROLLER_STEP,
                "test",
                {"step": 1},
            )
            
            assert event is None
        finally:
            set_global_event_logger(original)
    
    def test_convenience_functions(self):
        """Convenience global logging functions work."""
        original = get_global_event_logger()
        try:
            logger = EventLogger()
            set_global_event_logger(logger)
            
            log_controller_step_global(1, "TEST")
            log_llm_call_global("test", 10, 5, 100.0, True)
            log_budget_warning_global("steps", 8, 10, 80.0)
            log_budget_exceeded_global("tokens", 100, 100)
            log_security_violation_global("shell_true", "/path.py", 10, "test")
            log_subprocess_exec_global(["echo", "hi"], 0, True, 10.0)
            log_error_global("test", "Error", "test error")
            
            assert logger.event_count == 7
        finally:
            set_global_event_logger(original)


# =============================================================================
# Test Integration with Existing Components
# =============================================================================

class TestBudgetIntegration:
    """Tests for budget event integration."""
    
    def test_budget_warning_logs_event(self):
        """Budget warning triggers event logging."""
        original = get_global_event_logger()
        try:
            logger = EventLogger()
            set_global_event_logger(logger)
            
            from rfsn_controller.budget import Budget
            
            budget = Budget(max_steps=10, warning_threshold=0.8)
            
            # Record steps to trigger warning (at 80%)
            for _ in range(8):
                budget.record_step()
            
            # Check if warning event was logged
            warning_events = logger.get_events_by_type(EventType.BUDGET_WARNING)
            assert len(warning_events) >= 1
        finally:
            set_global_event_logger(original)
    
    def test_budget_exceeded_logs_event(self):
        """Budget exceeded triggers event logging."""
        original = get_global_event_logger()
        try:
            logger = EventLogger()
            set_global_event_logger(logger)
            
            from rfsn_controller.budget import Budget, BudgetExceeded
            
            budget = Budget(max_steps=5)
            
            # Record steps to exceed budget
            with pytest.raises(BudgetExceeded):
                for _ in range(10):
                    budget.record_step()
            
            # Check if exceeded event was logged
            exceeded_events = logger.get_events_by_type(EventType.BUDGET_EXCEEDED)
            assert len(exceeded_events) >= 1
        finally:
            set_global_event_logger(original)


class TestExecUtilsIntegration:
    """Tests for exec_utils event integration."""
    
    def test_subprocess_logs_event(self, tmp_path):
        """Subprocess execution logs event."""
        original = get_global_event_logger()
        try:
            logger = EventLogger()
            set_global_event_logger(logger)
            
            from rfsn_controller.exec_utils import safe_run
            
            result = safe_run(
                ["echo", "hello"],
                cwd=str(tmp_path),
                check_global_allowlist=False,
            )
            
            assert result.ok
            
            # Check if subprocess event was logged
            exec_events = logger.get_events_by_type(EventType.SUBPROCESS_EXEC)
            assert len(exec_events) >= 1
            
            event = exec_events[0]
            assert event.data["command"] == ["echo", "hello"]
            assert event.data["success"] is True
        finally:
            set_global_event_logger(original)


class TestEventConfigIntegration:
    """Tests for EventConfig in config.py."""
    
    def test_event_config_defaults(self):
        """EventConfig has sensible defaults."""
        from rfsn_controller.config import EventConfig
        
        config = EventConfig()
        assert config.enabled is True
        assert config.storage_path == ".rfsn/events.jsonl"
        assert config.max_events_memory == 10000
        assert config.persist_events is True
    
    def test_controller_config_has_events(self):
        """ControllerConfig includes EventConfig."""
        from rfsn_controller.config import ControllerConfig
        
        config = ControllerConfig(github_url="https://github.com/test/repo")
        assert hasattr(config, "events")
        assert config.events.enabled is True
    
    def test_config_from_cli_includes_events(self):
        """config_from_cli_args creates EventConfig from CLI args."""
        from rfsn_controller.config import config_from_cli_args
        
        class MockArgs:
            repo = "https://github.com/test/repo"
            ref = None
            events_enabled = True
            events_storage_path = ".rfsn/custom/events.jsonl"
            events_max_memory = 5000
            events_persist = True
        
        config = config_from_cli_args(MockArgs())
        assert config.events.max_events_memory == 5000
        assert config.events.storage_path == ".rfsn/custom/events.jsonl"
        assert config.events.persist_events is True


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance tests for event logging."""
    
    def test_high_volume_logging(self):
        """Logger handles high volume of events."""
        logger = EventLogger(max_events=0)  # Unlimited
        
        start = time.time()
        for i in range(10000):
            logger.log(EventType.CONTROLLER_STEP, "perf_test", {"i": i})
        elapsed = time.time() - start
        
        assert logger.event_count == 10000
        # Should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0
    
    def test_store_write_performance(self, tmp_path):
        """Store handles high volume of writes."""
        store_path = tmp_path / "perf_events.jsonl"
        store = EventStore(store_path)
        
        events = [
            create_event(EventType.SUBPROCESS_EXEC, "perf", {"i": i})
            for i in range(1000)
        ]
        
        start = time.time()
        store.append_batch(events)
        elapsed = time.time() - start
        
        assert store.get_event_count() == 1000
        # Should complete in reasonable time (< 2 seconds)
        assert elapsed < 2.0
