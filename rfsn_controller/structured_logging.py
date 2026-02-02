"""Structured logging with contextvars for request tracing.

Provides request-scoped context tracking and structured logging that
automatically includes trace IDs, user context, and metadata in all log entries.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any

# Context variables for request tracking
request_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
user_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("user", default=None)
session_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("session", default=None)
repo_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("repo", default=None)
phase_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("phase", default=None)


@dataclass
class LogContext:
    """Structured log context that propagates through the call stack."""

    request_id: str | None = None
    user: str | None = None
    session: str | None = None
    repo: str | None = None
    phase: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None and key != "extra":
                result[key] = value
        if self.extra:
            result.update(self.extra)
        return result


class StructuredLogger:
    """Structured logger with context propagation.

    Example:
        logger = StructuredLogger("rfsn.controller")

        # Set context for current request
        with logger.context(request_id="abc123", repo="user/repo"):
            logger.info("Processing patch", patch_id=42, status="success")
            # Output: {"level": "INFO", "message": "Processing patch",
            #          "request_id": "abc123", "repo": "user/repo",
            #          "patch_id": 42, "status": "success", "timestamp": ...}
    """

    def __init__(self, name: str, level: int = logging.INFO):
        """Initialize structured logger.

        Args:
            name: Logger name (typically module name)
            level: Logging level
        """
        self.name = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)

        # Use JSON formatter if not already configured
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(StructuredFormatter())
            self._logger.addHandler(handler)

    def _get_context(self) -> dict[str, Any]:
        """Get current context from contextvars."""
        ctx = {}

        if request_id := request_id_ctx.get():
            ctx["request_id"] = request_id
        if user := user_ctx.get():
            ctx["user"] = user
        if session := session_ctx.get():
            ctx["session"] = session
        if repo := repo_ctx.get():
            ctx["repo"] = repo
        if phase := phase_ctx.get():
            ctx["phase"] = phase

        return ctx

    def _log(self, level: int, message: str, **kwargs: Any) -> None:
        """Internal logging method that adds context."""
        # Merge context with kwargs
        log_data = self._get_context()
        log_data.update(kwargs)

        # Create structured record
        extra = {"structured_data": log_data}
        self._logger.log(level, message, extra=extra)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with context."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message with context."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with context."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message with context."""
        self._log(logging.ERROR, message, **kwargs)

    def exception(self, message: str, exc: Exception | None = None, **kwargs: Any) -> None:
        """Log exception with traceback and context.

        Args:
            message: Error message
            exc: Optional exception to log
            **kwargs: Additional context
        """
        if exc:
            kwargs["exception_type"] = type(exc).__name__
            kwargs["exception_message"] = str(exc)
            kwargs["traceback"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

        self._log(logging.ERROR, message, **kwargs)

    def context(
        self,
        request_id: str | None = None,
        user: str | None = None,
        session: str | None = None,
        repo: str | None = None,
        phase: str | None = None,
        **extra: Any,
    ) -> LogContextManager:
        """Create a context manager for scoped logging context.

        Args:
            request_id: Request/trace ID
            user: User identifier
            session: Session identifier
            repo: Repository name
            phase: Current phase (e.g., "planning", "patching", "testing")
            **extra: Additional context fields

        Example:
            with logger.context(request_id="abc", phase="patching"):
                logger.info("Starting patch")
                do_work()
                logger.info("Patch complete")
        """
        return LogContextManager(
            request_id=request_id,
            user=user,
            session=session,
            repo=repo,
            phase=phase,
            extra=extra,
        )


class LogContextManager:
    """Context manager for scoped logging context."""

    def __init__(
        self,
        request_id: str | None = None,
        user: str | None = None,
        session: str | None = None,
        repo: str | None = None,
        phase: str | None = None,
        extra: dict[str, Any] | None = None,
    ):
        self.request_id = request_id
        self.user = user
        self.session = session
        self.repo = repo
        self.phase = phase
        self.extra = extra or {}

        # Store tokens for cleanup
        self._tokens: list[Any] = []

    def __enter__(self) -> LogContextManager:
        """Enter context and set contextvars."""
        if self.request_id:
            self._tokens.append(request_id_ctx.set(self.request_id))
        if self.user:
            self._tokens.append(user_ctx.set(self.user))
        if self.session:
            self._tokens.append(session_ctx.set(self.session))
        if self.repo:
            self._tokens.append(repo_ctx.set(self.repo))
        if self.phase:
            self._tokens.append(phase_ctx.set(self.phase))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore contextvars."""
        for token in reversed(self._tokens):
            # Reset to previous value
            with contextlib.suppress(Exception):
                token.var.reset(token)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: Log record

        Returns:
            JSON string
        """
        log_entry = {
            "timestamp": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add file/line info in debug mode
        if record.levelno == logging.DEBUG:
            log_entry["file"] = record.pathname
            log_entry["line"] = record.lineno
            log_entry["function"] = record.funcName

        # Add structured data if present
        if hasattr(record, "structured_data"):
            log_entry.update(record.structured_data)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


# Global structured logger instances
_loggers: dict[str, StructuredLogger] = {}


def get_logger(name: str, level: int = logging.INFO) -> StructuredLogger:
    """Get or create a structured logger.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        StructuredLogger instance
    """
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name, level)
    return _loggers[name]


def configure_logging(
    level: int = logging.INFO,
    format_json: bool = True,
) -> None:
    """Configure global logging settings.

    Args:
        level: Global logging level
        format_json: Use JSON formatting
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add new handler
    handler = logging.StreamHandler(sys.stderr)

    if format_json:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    root_logger.addHandler(handler)


# Convenience functions for common operations
def log_performance(func_name: str, duration_ms: float, **kwargs: Any) -> None:
    """Log performance metrics.

    Args:
        func_name: Function name
        duration_ms: Duration in milliseconds
        **kwargs: Additional metrics
    """
    logger = get_logger("rfsn.performance")
    logger.info(
        f"Performance: {func_name}",
        function=func_name,
        duration_ms=duration_ms,
        **kwargs,
    )


def log_llm_call(
    provider: str,
    model: str,
    tokens_used: int,
    latency_ms: float,
    **kwargs: Any,
) -> None:
    """Log LLM API call metrics.

    Args:
        provider: LLM provider (e.g., "deepseek", "gemini")
        model: Model name
        tokens_used: Total tokens consumed
        latency_ms: API call latency
        **kwargs: Additional context
    """
    logger = get_logger("rfsn.llm")
    logger.info(
        f"LLM call: {provider}/{model}",
        provider=provider,
        model=model,
        tokens_used=tokens_used,
        latency_ms=latency_ms,
        **kwargs,
    )


def log_security_event(
    event_type: str,
    severity: str,
    description: str,
    **kwargs: Any,
) -> None:
    """Log security event.

    Args:
        event_type: Type of security event
        severity: Severity level (low, medium, high, critical)
        description: Event description
        **kwargs: Additional context
    """
    logger = get_logger("rfsn.security")
    logger.warning(
        f"Security event: {event_type}",
        event_type=event_type,
        severity=severity,
        description=description,
        **kwargs,
    )
