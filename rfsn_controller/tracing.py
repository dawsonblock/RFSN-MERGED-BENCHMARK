"""Distributed tracing for RFSN Controller using OpenTelemetry.

Provides comprehensive distributed tracing capabilities for debugging
and monitoring the RFSN repair pipeline.

Usage:
    from rfsn_controller.tracing import init_tracing, get_tracer, trace_function
    
    # Initialize once at startup
    init_tracing(service_name="rfsn-controller")
    
    # Get tracer for your module
    tracer = get_tracer(__name__)
    
    # Trace a function
    with tracer.start_as_current_span("operation_name") as span:
        span.set_attribute("key", "value")
        do_work()
    
    # Or use decorator
    @trace_function("custom_operation")
    def my_function():
        pass
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.trace import Span, Status, StatusCode
    
    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False
    trace = None  # type: ignore
    TracerProvider = None  # type: ignore
    BatchSpanProcessor = None  # type: ignore
    JaegerExporter = None  # type: ignore
    ConsoleSpanExporter = None  # type: ignore
    Resource = None  # type: ignore
    SERVICE_NAME = None  # type: ignore
    Status = None  # type: ignore
    StatusCode = None  # type: ignore
    Span = None  # type: ignore


# Global tracer provider
_tracer_provider: Any | None = None
_tracing_enabled = False


def init_tracing(
    service_name: str = "rfsn-controller",
    jaeger_host: str = "localhost",
    jaeger_port: int = 6831,
    console_export: bool = False,
) -> bool:
    """Initialize OpenTelemetry tracing.
    
    Args:
        service_name: Name of the service for tracing
        jaeger_host: Jaeger agent hostname
        jaeger_port: Jaeger agent port
        console_export: Also export to console (for debugging)
        
    Returns:
        True if tracing initialized successfully, False otherwise
        
    Example:
        >>> init_tracing(service_name="rfsn-controller")
        >>> # Traces will be sent to Jaeger at localhost:6831
    """
    global _tracer_provider, _tracing_enabled
    
    if not HAS_OPENTELEMETRY:
        print("Warning: OpenTelemetry not installed. Tracing disabled.")
        print("Install with: pip install 'rfsn-controller[observability]'")
        return False
    
    try:
        # Create resource with service name
        resource = Resource(attributes={
            SERVICE_NAME: service_name
        })
        
        # Create tracer provider
        _tracer_provider = TracerProvider(resource=resource)
        
        # Add Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name=jaeger_host,
            agent_port=jaeger_port,
        )
        _tracer_provider.add_span_processor(
            BatchSpanProcessor(jaeger_exporter)
        )
        
        # Optionally add console exporter for debugging
        if console_export:
            console_exporter = ConsoleSpanExporter()
            _tracer_provider.add_span_processor(
                BatchSpanProcessor(console_exporter)
            )
        
        # Set as global tracer provider
        trace.set_tracer_provider(_tracer_provider)
        
        _tracing_enabled = True
        print(f"✓ Tracing initialized: {service_name} → {jaeger_host}:{jaeger_port}")
        return True
        
    except Exception as e:
        print(f"Warning: Failed to initialize tracing: {e}")
        _tracing_enabled = False
        return False


def get_tracer(name: str) -> Any:
    """Get a tracer instance for the given name.
    
    Args:
        name: Tracer name (usually __name__ of module)
        
    Returns:
        Tracer instance or NoOp tracer if tracing disabled
        
    Example:
        >>> tracer = get_tracer(__name__)
        >>> with tracer.start_as_current_span("operation"):
        ...     do_work()
    """
    if not HAS_OPENTELEMETRY or not _tracing_enabled:
        return NoOpTracer()
    
    return trace.get_tracer(name)


def is_tracing_enabled() -> bool:
    """Check if tracing is enabled.
    
    Returns:
        True if tracing is initialized and enabled
    """
    return _tracing_enabled and HAS_OPENTELEMETRY


@contextmanager
def trace_span(
    name: str,
    attributes: dict | None = None,
    tracer_name: str = "rfsn"
):
    """Context manager for creating a trace span.
    
    Args:
        name: Span name
        attributes: Optional span attributes
        tracer_name: Tracer name
        
    Yields:
        Span object or None if tracing disabled
        
    Example:
        >>> with trace_span("database_query", {"query": "SELECT *"}) as span:
        ...     result = db.query()
        ...     if span:
        ...         span.set_attribute("rows", len(result))
    """
    if not is_tracing_enabled():
        yield None
        return
    
    tracer = get_tracer(tracer_name)
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        
        try:
            yield span
        except Exception as e:
            if span:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            raise


def trace_function(
    span_name: str | None = None,
    attributes: dict | None = None
):
    """Decorator to trace a function.
    
    Args:
        span_name: Optional span name (defaults to function name)
        attributes: Optional static attributes
        
    Example:
        >>> @trace_function("process_proposal")
        ... def process(proposal):
        ...     return validate(proposal)
        
        >>> @trace_function(attributes={"component": "planner"})
        ... def generate_proposal():
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        name = span_name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not is_tracing_enabled():
                return func(*args, **kwargs)
            
            tracer = get_tracer(func.__module__)
            with tracer.start_as_current_span(name) as span:
                # Add static attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, str(value))
                
                # Add function info
                span.set_attribute("function", func.__name__)
                span.set_attribute("module", func.__module__)
                
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        return wrapper
    return decorator


class NoOpTracer:
    """No-op tracer for when tracing is disabled."""
    
    @contextmanager
    def start_as_current_span(self, name: str, **kwargs):
        """No-op context manager."""
        yield NoOpSpan()


class NoOpSpan:
    """No-op span for when tracing is disabled."""
    
    def set_attribute(self, key: str, value: Any) -> None:
        """No-op set attribute."""
        pass
    
    def set_status(self, status: Any) -> None:
        """No-op set status."""
        pass
    
    def record_exception(self, exception: Exception) -> None:
        """No-op record exception."""
        pass


# Convenience functions for common tracing patterns

def trace_llm_call(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    duration_ms: float,
    error: str | None = None
):
    """Record LLM call trace.
    
    Args:
        model: LLM model name
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        duration_ms: Call duration in milliseconds
        error: Error message if call failed
    """
    with trace_span("llm.call", {
        "llm.model": model,
        "llm.prompt_tokens": prompt_tokens,
        "llm.completion_tokens": completion_tokens,
        "llm.duration_ms": duration_ms,
        "llm.error": error or "",
    }) as span:
        if span and error:
            span.set_status(Status(StatusCode.ERROR, error))


def trace_proposal(
    proposal_id: str,
    intent: str,
    action_type: str,
    accepted: bool,
    rejection_reason: str | None = None
):
    """Record proposal trace.
    
    Args:
        proposal_id: Unique proposal ID
        intent: Proposal intent
        action_type: Action type
        accepted: Whether proposal was accepted
        rejection_reason: Reason for rejection if not accepted
    """
    with trace_span("proposal.generate", {
        "proposal.id": proposal_id,
        "proposal.intent": intent,
        "proposal.action_type": action_type,
        "proposal.accepted": accepted,
        "proposal.rejection_reason": rejection_reason or "",
    }) as span:
        if span and not accepted:
            span.set_status(Status(StatusCode.ERROR, rejection_reason or "Rejected"))


def trace_test_execution(
    test_path: str,
    passed: bool,
    duration_ms: float,
    failures: int = 0,
    errors: int = 0
):
    """Record test execution trace.
    
    Args:
        test_path: Path to test file/function
        passed: Whether tests passed
        duration_ms: Execution duration in milliseconds
        failures: Number of failures
        errors: Number of errors
    """
    with trace_span("test.execute", {
        "test.path": test_path,
        "test.passed": passed,
        "test.duration_ms": duration_ms,
        "test.failures": failures,
        "test.errors": errors,
    }) as span:
        if span and not passed:
            span.set_status(Status(StatusCode.ERROR, f"{failures} failures, {errors} errors"))


def shutdown_tracing():
    """Shutdown tracing and flush any pending spans."""
    global _tracer_provider, _tracing_enabled
    
    if _tracer_provider and HAS_OPENTELEMETRY:
        try:
            _tracer_provider.shutdown()
            print("✓ Tracing shutdown complete")
        except Exception as e:
            print(f"Warning: Error during tracing shutdown: {e}")
    
    _tracing_enabled = False
    _tracer_provider = None
