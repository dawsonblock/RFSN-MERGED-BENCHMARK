"""Tests for OpenTelemetry tracing module."""

from unittest.mock import Mock

import pytest

from rfsn_controller.tracing import (
    HAS_OPENTELEMETRY,
    get_tracer,
    init_tracing,
    is_tracing_enabled,
    shutdown_tracing,
    trace_function,
    trace_llm_call,
    trace_proposal,
    trace_span,
    trace_test_execution,
)


class TestTracingInitialization:
    """Test tracing initialization."""
    
    def test_init_without_opentelemetry(self):
        """Test initialization when OpenTelemetry not installed."""
        if not HAS_OPENTELEMETRY:
            result = init_tracing()
            assert result is False
            assert not is_tracing_enabled()
    
    @pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
    def test_init_with_opentelemetry(self):
        """Test successful initialization."""
        result = init_tracing(
            service_name="test-service",
            jaeger_host="localhost",
            jaeger_port=6831
        )
        assert result is True
        assert is_tracing_enabled()
        shutdown_tracing()
    
    @pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
    def test_init_with_console_export(self):
        """Test initialization with console export."""
        result = init_tracing(
            service_name="test-service",
            console_export=True
        )
        assert result is True
        shutdown_tracing()


class TestTracer:
    """Test tracer functionality."""
    
    def test_get_tracer_when_disabled(self):
        """Test getting tracer when tracing disabled."""
        tracer = get_tracer("test")
        assert tracer is not None  # Should return NoOpTracer
    
    @pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
    def test_get_tracer_when_enabled(self):
        """Test getting tracer when enabled."""
        init_tracing()
        tracer = get_tracer("test_module")
        assert tracer is not None
        shutdown_tracing()


class TestTraceSpan:
    """Test trace span context manager."""
    
    def test_trace_span_when_disabled(self):
        """Test span context when tracing disabled."""
        with trace_span("test_operation") as span:
            assert span is None
    
    @pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
    def test_trace_span_when_enabled(self):
        """Test span context when enabled."""
        init_tracing()
        
        with trace_span("test_operation", {"key": "value"}) as span:
            assert span is not None
            # Span operations should not raise
            span.set_attribute("test", "value")
        
        shutdown_tracing()
    
    @pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
    def test_trace_span_with_exception(self):
        """Test span records exceptions."""
        init_tracing()
        
        with pytest.raises(ValueError):
            with trace_span("test_operation"):
                raise ValueError("Test error")
        
        shutdown_tracing()


class TestTraceFunction:
    """Test function tracing decorator."""
    
    def test_trace_function_when_disabled(self):
        """Test decorator when tracing disabled."""
        @trace_function("test_func")
        def test_func(x):
            return x * 2
        
        result = test_func(5)
        assert result == 10
    
    @pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
    def test_trace_function_when_enabled(self):
        """Test decorator when enabled."""
        init_tracing()
        
        @trace_function("test_func")
        def test_func(x):
            return x * 2
        
        result = test_func(5)
        assert result == 10
        
        shutdown_tracing()
    
    @pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
    def test_trace_function_with_attributes(self):
        """Test decorator with static attributes."""
        init_tracing()
        
        @trace_function(attributes={"component": "test"})
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"
        
        shutdown_tracing()
    
    @pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
    def test_trace_function_with_exception(self):
        """Test decorator records exceptions."""
        init_tracing()
        
        @trace_function("test_func")
        def test_func():
            raise RuntimeError("Test error")
        
        with pytest.raises(RuntimeError):
            test_func()
        
        shutdown_tracing()


class TestConvenienceFunctions:
    """Test convenience tracing functions."""
    
    def test_trace_llm_call(self):
        """Test LLM call tracing."""
        # Should not raise even when tracing disabled
        trace_llm_call(
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            duration_ms=1000.5
        )
    
    def test_trace_llm_call_with_error(self):
        """Test LLM call tracing with error."""
        trace_llm_call(
            model="gpt-4",
            prompt_tokens=100,
            completion_tokens=0,
            duration_ms=500.0,
            error="Rate limit exceeded"
        )
    
    def test_trace_proposal(self):
        """Test proposal tracing."""
        trace_proposal(
            proposal_id="prop-123",
            intent="repair",
            action_type="edit_file",
            accepted=True
        )
    
    def test_trace_proposal_rejected(self):
        """Test rejected proposal tracing."""
        trace_proposal(
            proposal_id="prop-124",
            intent="repair",
            action_type="edit_file",
            accepted=False,
            rejection_reason="Schema validation failed"
        )
    
    def test_trace_test_execution(self):
        """Test execution tracing."""
        trace_test_execution(
            test_path="tests/test_module.py::test_function",
            passed=True,
            duration_ms=250.0
        )
    
    def test_trace_test_execution_failed(self):
        """Test failed execution tracing."""
        trace_test_execution(
            test_path="tests/test_module.py::test_function",
            passed=False,
            duration_ms=300.0,
            failures=2,
            errors=1
        )


class TestShutdown:
    """Test tracing shutdown."""
    
    @pytest.mark.skipif(not HAS_OPENTELEMETRY, reason="OpenTelemetry not installed")
    def test_shutdown_when_enabled(self):
        """Test shutdown flushes spans."""
        init_tracing()
        assert is_tracing_enabled()
        
        shutdown_tracing()
        assert not is_tracing_enabled()
    
    def test_shutdown_when_disabled(self):
        """Test shutdown when already disabled."""
        # Should not raise
        shutdown_tracing()
        assert not is_tracing_enabled()


class TestNoOpImplementations:
    """Test no-op implementations."""
    
    def test_noop_tracer(self):
        """Test NoOpTracer."""
        from rfsn_controller.tracing import NoOpTracer
        
        tracer = NoOpTracer()
        with tracer.start_as_current_span("test") as span:
            span.set_attribute("key", "value")
            span.set_status(Mock())
            span.record_exception(Exception("test"))
        # Should not raise
    
    def test_noop_span(self):
        """Test NoOpSpan."""
        from rfsn_controller.tracing import NoOpSpan
        
        span = NoOpSpan()
        span.set_attribute("key", "value")
        span.set_status(Mock())
        span.record_exception(Exception("test"))
        # Should not raise


@pytest.fixture(autouse=True)
def cleanup_tracing():
    """Cleanup tracing after each test."""
    yield
    shutdown_tracing()
