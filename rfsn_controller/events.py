"""Structured event logging system for RFSN controller observability.

This module provides comprehensive event tracking for agent operations including:
- Controller steps and state transitions
- LLM calls and token usage
- Budget warnings and exceeded events
- Security violations
- Subprocess executions
- Feature registrations
- Error events

Events are structured, queryable, and persistable for analysis and debugging.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Event Types and Severity
# =============================================================================

class EventType(Enum):
    """Types of events that can be logged."""
    
    CONTROLLER_STEP = "controller_step"
    LLM_CALL = "llm_call"
    BUDGET_WARNING = "budget_warning"
    BUDGET_EXCEEDED = "budget_exceeded"
    SECURITY_VIOLATION = "security_violation"
    SUBPROCESS_EXEC = "subprocess_exec"
    FEATURE_REGISTERED = "feature_registered"
    ERROR = "error"


class EventSeverity(Enum):
    """Severity levels for events."""
    
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# Severity ordering for comparisons (higher index = more severe)
SEVERITY_ORDER = [
    EventSeverity.DEBUG,
    EventSeverity.INFO,
    EventSeverity.WARNING,
    EventSeverity.ERROR,
    EventSeverity.CRITICAL,
]


def _severity_index(severity: EventSeverity) -> int:
    """Get the index of a severity level for comparison."""
    try:
        return SEVERITY_ORDER.index(severity)
    except ValueError:
        return 0


# =============================================================================
# Event Dataclass
# =============================================================================

@dataclass
class Event:
    """Represents a structured event in the system.
    
    Attributes:
        timestamp: ISO 8601 timestamp when the event occurred.
        event_type: Type of the event (EventType enum value).
        source: Module or component that generated the event.
        data: Arbitrary event-specific data dictionary.
        severity: Event severity level (EventSeverity enum value).
        event_id: Unique identifier for the event.
        run_id: Identifier for the current execution run.
        correlation_id: Optional ID to link related events.
    """
    
    timestamp: str
    event_type: EventType
    source: str
    data: dict[str, Any]
    severity: EventSeverity = EventSeverity.INFO
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str | None = None
    correlation_id: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert event to a dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "source": self.source,
            "data": self.data,
            "severity": self.severity.value,
            "event_id": self.event_id,
            "run_id": self.run_id,
            "correlation_id": self.correlation_id,
        }
    
    def to_json(self) -> str:
        """Serialize event to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Event:
        """Create an Event from a dictionary."""
        return cls(
            timestamp=d["timestamp"],
            event_type=EventType(d["event_type"]),
            source=d["source"],
            data=d.get("data", {}),
            severity=EventSeverity(d.get("severity", "info")),
            event_id=d.get("event_id", str(uuid.uuid4())),
            run_id=d.get("run_id"),
            correlation_id=d.get("correlation_id"),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> Event:
        """Deserialize an Event from JSON string."""
        return cls.from_dict(json.loads(json_str))


def create_event(
    event_type: EventType,
    source: str,
    data: dict[str, Any],
    severity: EventSeverity = EventSeverity.INFO,
    run_id: str | None = None,
    correlation_id: str | None = None,
) -> Event:
    """Factory function to create an Event with current timestamp.
    
    Args:
        event_type: Type of the event.
        source: Module or component generating the event.
        data: Event-specific data.
        severity: Event severity level.
        run_id: Current execution run identifier.
        correlation_id: ID to link related events.
        
    Returns:
        A new Event instance.
    """
    return Event(
        timestamp=datetime.now(UTC).isoformat(),
        event_type=event_type,
        source=source,
        data=data,
        severity=severity,
        run_id=run_id,
        correlation_id=correlation_id,
    )


# =============================================================================
# Event Logger
# =============================================================================

class EventLogger:
    """Collects and manages events in memory.
    
    Thread-safe event collection with optional callbacks for real-time
    event processing. Supports filtering and iteration.
    
    Attributes:
        events: List of collected events.
        run_id: Current execution run identifier.
        max_events: Maximum events to retain (0 = unlimited).
    """
    
    def __init__(
        self,
        run_id: str | None = None,
        max_events: int = 10000,
    ):
        """Initialize the event logger.
        
        Args:
            run_id: Identifier for the current run.
            max_events: Maximum events to keep (0 = unlimited).
        """
        self._events: list[Event] = []
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[Event], None]] = []
        self._min_severity: EventSeverity = EventSeverity.DEBUG
        self.run_id = run_id or str(uuid.uuid4())
        self.max_events = max_events
    
    def log(
        self,
        event_type: EventType,
        source: str,
        data: dict[str, Any],
        severity: EventSeverity = EventSeverity.INFO,
        correlation_id: str | None = None,
    ) -> Event:
        """Log a new event.
        
        Args:
            event_type: Type of the event.
            source: Module or component generating the event.
            data: Event-specific data.
            severity: Event severity level.
            correlation_id: ID to link related events.
            
        Returns:
            The created Event instance.
        """
        # Filter by minimum severity using proper ordering
        if _severity_index(severity) < _severity_index(self._min_severity):
            event = create_event(
                event_type=event_type,
                source=source,
                data=data,
                severity=severity,
                run_id=self.run_id,
                correlation_id=correlation_id,
            )
            return event
        
        event = create_event(
            event_type=event_type,
            source=source,
            data=data,
            severity=severity,
            run_id=self.run_id,
            correlation_id=correlation_id,
        )
        
        with self._lock:
            self._events.append(event)
            
            # Enforce max events limit
            if self.max_events > 0 and len(self._events) > self.max_events:
                self._events = self._events[-self.max_events:]
        
        # Call registered callbacks
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.warning(f"Event callback failed: {e}")
        
        return event
    
    def log_controller_step(
        self,
        step_number: int,
        phase: str,
        data: dict[str, Any] | None = None,
    ) -> Event:
        """Log a controller step event.
        
        Args:
            step_number: Current step number.
            phase: Current execution phase.
            data: Additional step data.
            
        Returns:
            The created Event.
        """
        event_data = {
            "step_number": step_number,
            "phase": phase,
            **(data or {}),
        }
        return self.log(
            EventType.CONTROLLER_STEP,
            "controller",
            event_data,
            EventSeverity.INFO,
        )
    
    def log_llm_call(
        self,
        model: str,
        tokens_prompt: int,
        tokens_completion: int,
        latency_ms: float,
        success: bool,
        error: str | None = None,
    ) -> Event:
        """Log an LLM API call event.
        
        Args:
            model: Name of the LLM model.
            tokens_prompt: Prompt token count.
            tokens_completion: Completion token count.
            latency_ms: Call latency in milliseconds.
            success: Whether the call succeeded.
            error: Error message if failed.
            
        Returns:
            The created Event.
        """
        return self.log(
            EventType.LLM_CALL,
            "llm",
            {
                "model": model,
                "tokens_prompt": tokens_prompt,
                "tokens_completion": tokens_completion,
                "tokens_total": tokens_prompt + tokens_completion,
                "latency_ms": latency_ms,
                "success": success,
                "error": error,
            },
            EventSeverity.INFO if success else EventSeverity.ERROR,
        )
    
    def log_budget_warning(
        self,
        resource: str,
        current: int,
        limit: int,
        percentage: float,
    ) -> Event:
        """Log a budget warning event.
        
        Args:
            resource: Name of the resource.
            current: Current consumption.
            limit: Hard limit.
            percentage: Current usage percentage.
            
        Returns:
            The created Event.
        """
        return self.log(
            EventType.BUDGET_WARNING,
            "budget",
            {
                "resource": resource,
                "current": current,
                "limit": limit,
                "percentage": percentage,
            },
            EventSeverity.WARNING,
        )
    
    def log_budget_exceeded(
        self,
        resource: str,
        current: int,
        limit: int,
    ) -> Event:
        """Log a budget exceeded event.
        
        Args:
            resource: Name of the resource.
            current: Current consumption.
            limit: Hard limit.
            
        Returns:
            The created Event.
        """
        return self.log(
            EventType.BUDGET_EXCEEDED,
            "budget",
            {
                "resource": resource,
                "current": current,
                "limit": limit,
            },
            EventSeverity.CRITICAL,
        )
    
    def log_security_violation(
        self,
        violation_type: str,
        file_path: str,
        line_number: int,
        message: str,
        severity: str = "high",
    ) -> Event:
        """Log a security violation event.
        
        Args:
            violation_type: Type of security violation.
            file_path: Path to the file with violation.
            line_number: Line number of violation.
            message: Description of the violation.
            severity: Violation severity level.
            
        Returns:
            The created Event.
        """
        event_severity = EventSeverity.CRITICAL if severity == "critical" else EventSeverity.ERROR
        return self.log(
            EventType.SECURITY_VIOLATION,
            "shell_scanner",
            {
                "violation_type": violation_type,
                "file": file_path,
                "line": line_number,
                "message": message,
                "violation_severity": severity,
            },
            event_severity,
        )
    
    def log_subprocess_exec(
        self,
        command: list[str],
        exit_code: int,
        success: bool,
        duration_ms: float,
        cwd: str | None = None,
    ) -> Event:
        """Log a subprocess execution event.
        
        Args:
            command: Command as argv list.
            exit_code: Process exit code.
            success: Whether execution succeeded.
            duration_ms: Execution duration in milliseconds.
            cwd: Working directory.
            
        Returns:
            The created Event.
        """
        return self.log(
            EventType.SUBPROCESS_EXEC,
            "exec_utils",
            {
                "command": command,
                "exit_code": exit_code,
                "success": success,
                "duration_ms": duration_ms,
                "cwd": cwd,
            },
            EventSeverity.INFO if success else EventSeverity.WARNING,
        )
    
    def log_error(
        self,
        source: str,
        error_type: str,
        message: str,
        traceback: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> Event:
        """Log an error event.
        
        Args:
            source: Module where error occurred.
            error_type: Type/class of the error.
            message: Error message.
            traceback: Full traceback string.
            data: Additional error context.
            
        Returns:
            The created Event.
        """
        return self.log(
            EventType.ERROR,
            source,
            {
                "error_type": error_type,
                "message": message,
                "traceback": traceback,
                **(data or {}),
            },
            EventSeverity.ERROR,
        )
    
    def register_callback(self, callback: Callable[[Event], None]) -> None:
        """Register a callback for new events.
        
        Args:
            callback: Function called with each new event.
        """
        self._callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[Event], None]) -> None:
        """Unregister an event callback.
        
        Args:
            callback: Previously registered callback.
        """
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def set_min_severity(self, severity: EventSeverity) -> None:
        """Set minimum severity for event collection.
        
        Args:
            severity: Minimum severity to log.
        """
        self._min_severity = severity
    
    @property
    def events(self) -> list[Event]:
        """Get a copy of all collected events."""
        with self._lock:
            return list(self._events)
    
    @property
    def event_count(self) -> int:
        """Get the count of collected events."""
        with self._lock:
            return len(self._events)
    
    def clear(self) -> None:
        """Clear all collected events."""
        with self._lock:
            self._events.clear()
    
    def get_events_by_type(self, event_type: EventType) -> list[Event]:
        """Get events filtered by type.
        
        Args:
            event_type: Type to filter by.
            
        Returns:
            List of matching events.
        """
        with self._lock:
            return [e for e in self._events if e.event_type == event_type]
    
    def get_events_by_severity(
        self,
        min_severity: EventSeverity,
    ) -> list[Event]:
        """Get events at or above a severity level.
        
        Args:
            min_severity: Minimum severity level.
            
        Returns:
            List of matching events.
        """
        min_idx = _severity_index(min_severity)
        
        with self._lock:
            return [
                e for e in self._events 
                if _severity_index(e.severity) >= min_idx
            ]


# =============================================================================
# Event Store (Persistence)
# =============================================================================

class EventStore:
    """Persists events to JSON file storage.
    
    Events are stored as JSON Lines (one JSON object per line) for
    efficient append operations and streaming reads.
    
    Attributes:
        storage_path: Path to the events file.
    """
    
    def __init__(self, storage_path: str | Path):
        """Initialize the event store.
        
        Args:
            storage_path: Path to the JSON Lines file.
        """
        self.storage_path = Path(storage_path)
        self._lock = threading.Lock()
        
        # Ensure directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
    
    def append(self, event: Event) -> None:
        """Append an event to storage.
        
        Args:
            event: Event to persist.
        """
        with self._lock:
            with open(self.storage_path, "a") as f:
                f.write(event.to_json() + "\n")
    
    def append_batch(self, events: list[Event]) -> None:
        """Append multiple events to storage.
        
        Args:
            events: List of events to persist.
        """
        with self._lock:
            with open(self.storage_path, "a") as f:
                for event in events:
                    f.write(event.to_json() + "\n")
    
    def read_all(self) -> list[Event]:
        """Read all events from storage.
        
        Returns:
            List of all stored events.
        """
        events = []
        if not self.storage_path.exists():
            return events
        
        with self._lock:
            with open(self.storage_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(Event.from_json(line))
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"Failed to parse event: {e}")
        
        return events
    
    def iter_events(self) -> Iterator[Event]:
        """Iterate over events in storage.
        
        Yields:
            Events one at a time.
        """
        if not self.storage_path.exists():
            return
        
        with open(self.storage_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield Event.from_json(line)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Failed to parse event: {e}")
    
    def clear(self) -> None:
        """Clear all events from storage."""
        with self._lock:
            if self.storage_path.exists():
                self.storage_path.unlink()
    
    def get_event_count(self) -> int:
        """Get the count of stored events.
        
        Returns:
            Number of events in storage.
        """
        if not self.storage_path.exists():
            return 0
        
        count = 0
        with open(self.storage_path) as f:
            for _ in f:
                count += 1
        return count
    
    def rotate(self, max_events: int) -> int:
        """Rotate storage, keeping only recent events.
        
        Args:
            max_events: Maximum events to keep.
            
        Returns:
            Number of events removed.
        """
        events = self.read_all()
        if len(events) <= max_events:
            return 0
        
        removed = len(events) - max_events
        events = events[-max_events:]
        
        with self._lock:
            with open(self.storage_path, "w") as f:
                for event in events:
                    f.write(event.to_json() + "\n")
        
        return removed


# =============================================================================
# Event Query
# =============================================================================

@dataclass
class EventQuery:
    """Query builder for filtering and searching events.
    
    Supports filtering by type, severity, time range, source, and
    arbitrary data field conditions.
    """
    
    event_types: set[EventType] | None = None
    min_severity: EventSeverity | None = None
    sources: set[str] | None = None
    start_time: str | None = None
    end_time: str | None = None
    data_filters: dict[str, Any] = field(default_factory=dict)
    limit: int = 0
    
    def matches(self, event: Event) -> bool:
        """Check if an event matches this query.
        
        Args:
            event: Event to check.
            
        Returns:
            True if event matches all criteria.
        """
        # Check event type
        if self.event_types and event.event_type not in self.event_types:
            return False
        
        # Check severity
        if self.min_severity:
            if _severity_index(event.severity) < _severity_index(self.min_severity):
                return False
        
        # Check source
        if self.sources and event.source not in self.sources:
            return False
        
        # Check time range
        if self.start_time and event.timestamp < self.start_time:
            return False
        if self.end_time and event.timestamp > self.end_time:
            return False
        
        # Check data filters
        for key, value in self.data_filters.items():
            if key not in event.data:
                return False
            if event.data[key] != value:
                return False
        
        return True
    
    def filter(self, events: list[Event]) -> list[Event]:
        """Filter a list of events by this query.
        
        Args:
            events: Events to filter.
            
        Returns:
            List of matching events.
        """
        results = [e for e in events if self.matches(e)]
        if self.limit > 0:
            results = results[:self.limit]
        return results
    
    def filter_store(self, store: EventStore) -> list[Event]:
        """Filter events from a store by this query.
        
        Args:
            store: EventStore to query.
            
        Returns:
            List of matching events.
        """
        results = []
        for event in store.iter_events():
            if self.matches(event):
                results.append(event)
                if self.limit > 0 and len(results) >= self.limit:
                    break
        return results


# =============================================================================
# Global Event Logger (Singleton)
# =============================================================================

_global_event_logger: EventLogger | None = None
_global_event_logger_lock = threading.Lock()


def get_global_event_logger() -> EventLogger | None:
    """Get the global event logger instance.
    
    Returns:
        The global EventLogger or None if not set.
    """
    with _global_event_logger_lock:
        return _global_event_logger


def set_global_event_logger(logger: EventLogger | None) -> None:
    """Set the global event logger instance.
    
    Args:
        logger: EventLogger instance to set globally, or None to clear.
    """
    global _global_event_logger
    with _global_event_logger_lock:
        _global_event_logger = logger


def log_event_global(
    event_type: EventType,
    source: str,
    data: dict[str, Any],
    severity: EventSeverity = EventSeverity.INFO,
    correlation_id: str | None = None,
) -> Event | None:
    """Log an event to the global logger if set.
    
    Args:
        event_type: Type of the event.
        source: Module or component generating the event.
        data: Event-specific data.
        severity: Event severity level.
        correlation_id: ID to link related events.
        
    Returns:
        The created Event or None if no global logger.
    """
    logger = get_global_event_logger()
    if logger is not None:
        return logger.log(event_type, source, data, severity, correlation_id)
    return None


# =============================================================================
# Convenience Functions for Global Logging
# =============================================================================

def log_controller_step_global(
    step_number: int,
    phase: str,
    data: dict[str, Any] | None = None,
) -> Event | None:
    """Log a controller step to the global logger."""
    logger = get_global_event_logger()
    if logger is not None:
        return logger.log_controller_step(step_number, phase, data)
    return None


def log_llm_call_global(
    model: str,
    tokens_prompt: int,
    tokens_completion: int,
    latency_ms: float,
    success: bool,
    error: str | None = None,
) -> Event | None:
    """Log an LLM call to the global logger."""
    logger = get_global_event_logger()
    if logger is not None:
        return logger.log_llm_call(
            model, tokens_prompt, tokens_completion, latency_ms, success, error
        )
    return None


def log_budget_warning_global(
    resource: str,
    current: int,
    limit: int,
    percentage: float,
) -> Event | None:
    """Log a budget warning to the global logger."""
    logger = get_global_event_logger()
    if logger is not None:
        return logger.log_budget_warning(resource, current, limit, percentage)
    return None


def log_budget_exceeded_global(
    resource: str,
    current: int,
    limit: int,
) -> Event | None:
    """Log a budget exceeded event to the global logger."""
    logger = get_global_event_logger()
    if logger is not None:
        return logger.log_budget_exceeded(resource, current, limit)
    return None


def log_security_violation_global(
    violation_type: str,
    file_path: str,
    line_number: int,
    message: str,
    severity: str = "high",
) -> Event | None:
    """Log a security violation to the global logger."""
    logger = get_global_event_logger()
    if logger is not None:
        return logger.log_security_violation(
            violation_type, file_path, line_number, message, severity
        )
    return None


def log_subprocess_exec_global(
    command: list[str],
    exit_code: int,
    success: bool,
    duration_ms: float,
    cwd: str | None = None,
) -> Event | None:
    """Log a subprocess execution to the global logger."""
    logger = get_global_event_logger()
    if logger is not None:
        return logger.log_subprocess_exec(command, exit_code, success, duration_ms, cwd)
    return None


def log_error_global(
    source: str,
    error_type: str,
    message: str,
    traceback: str | None = None,
    data: dict[str, Any] | None = None,
) -> Event | None:
    """Log an error to the global logger."""
    logger = get_global_event_logger()
    if logger is not None:
        return logger.log_error(source, error_type, message, traceback, data)
    return None
