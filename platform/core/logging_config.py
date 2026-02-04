"""
Structured Logging Configuration - UNLEASH Platform

Provides JSON-structured logging for all platform components with:
- Correlation IDs for distributed request tracing
- Log sampling for high-volume events
- Per-component log levels
- Log aggregation-friendly JSON format

Usage:
    from core.logging_config import get_logger, LogContext

    # Get a component logger
    logger = get_logger("rag_pipeline")

    # Log with context
    with logger.correlation_context(request_id="req-123"):
        logger.info("Processing query", query="search term", results=10)

    # Log with sampling (for high-volume events)
    logger.debug("Cache hit", sample_rate=0.1)  # Only logs 10% of events
"""

from __future__ import annotations

import json
import logging
import random
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union


# =============================================================================
# LOG LEVELS PER COMPONENT
# =============================================================================

class ComponentLogLevel(Enum):
    """Predefined log levels for platform components."""
    # Core components
    RAG_PIPELINE = logging.INFO
    MEMORY_SYSTEM = logging.INFO
    ORCHESTRATOR = logging.INFO

    # Adapters (higher volume, default to WARNING)
    ADAPTER_EXA = logging.WARNING
    ADAPTER_TAVILY = logging.WARNING
    ADAPTER_JINA = logging.WARNING
    ADAPTER_PERPLEXITY = logging.WARNING
    ADAPTER_FIRECRAWL = logging.WARNING
    ADAPTER_LETTA = logging.WARNING
    ADAPTER_COGNEE = logging.WARNING
    ADAPTER_GENERIC = logging.WARNING

    # Research components
    RESEARCH_ENGINE = logging.INFO
    RESEARCH_SWARM = logging.INFO

    # Performance components
    CIRCUIT_BREAKER = logging.WARNING
    RATE_LIMITER = logging.WARNING
    CACHE = logging.WARNING
    CONNECTION_POOL = logging.WARNING

    # Debug/Development
    DEBUG_ALL = logging.DEBUG


# Default component log levels - can be overridden at runtime
COMPONENT_LOG_LEVELS: Dict[str, int] = {
    "rag_pipeline": logging.INFO,
    "memory": logging.INFO,
    "orchestrator": logging.INFO,
    "adapter.exa": logging.WARNING,
    "adapter.tavily": logging.WARNING,
    "adapter.jina": logging.WARNING,
    "adapter.perplexity": logging.WARNING,
    "adapter.firecrawl": logging.WARNING,
    "adapter.letta": logging.WARNING,
    "adapter.cognee": logging.WARNING,
    "adapter": logging.WARNING,
    "research": logging.INFO,
    "circuit_breaker": logging.WARNING,
    "rate_limiter": logging.WARNING,
    "cache": logging.WARNING,
    "pool": logging.WARNING,
}


def set_component_log_level(component: str, level: int) -> None:
    """Set log level for a specific component."""
    COMPONENT_LOG_LEVELS[component] = level


def get_component_log_level(component: str) -> int:
    """Get log level for a component, with fallback to parent levels."""
    if component in COMPONENT_LOG_LEVELS:
        return COMPONENT_LOG_LEVELS[component]

    # Check for parent component (e.g., "adapter.exa" -> "adapter")
    parts = component.split(".")
    while parts:
        parts.pop()
        parent = ".".join(parts)
        if parent in COMPONENT_LOG_LEVELS:
            return COMPONENT_LOG_LEVELS[parent]

    return logging.INFO  # Default


# =============================================================================
# LOG CONTEXT
# =============================================================================

@dataclass
class LogContext:
    """
    Context information for log correlation and tracing.

    Fields are designed for log aggregation tools (ELK, Datadog, etc.)
    """
    # Tracing
    correlation_id: Optional[str] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None

    # Component identification
    component: Optional[str] = None
    operation: Optional[str] = None
    adapter: Optional[str] = None

    # User/session context
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    agent_id: Optional[str] = None

    # Additional context
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for JSON serialization."""
        result = {}
        if self.correlation_id:
            result["correlation_id"] = self.correlation_id
        if self.request_id:
            result["request_id"] = self.request_id
        if self.trace_id:
            result["trace_id"] = self.trace_id
        if self.span_id:
            result["span_id"] = self.span_id
        if self.parent_span_id:
            result["parent_span_id"] = self.parent_span_id
        if self.component:
            result["component"] = self.component
        if self.operation:
            result["operation"] = self.operation
        if self.adapter:
            result["adapter"] = self.adapter
        if self.session_id:
            result["session_id"] = self.session_id
        if self.user_id:
            result["user_id"] = self.user_id
        if self.agent_id:
            result["agent_id"] = self.agent_id
        if self.extra:
            result.update(self.extra)
        return result

    def merge(self, other: "LogContext") -> "LogContext":
        """Merge another context into this one (other takes precedence)."""
        return LogContext(
            correlation_id=other.correlation_id or self.correlation_id,
            request_id=other.request_id or self.request_id,
            trace_id=other.trace_id or self.trace_id,
            span_id=other.span_id or self.span_id,
            parent_span_id=other.parent_span_id or self.parent_span_id,
            component=other.component or self.component,
            operation=other.operation or self.operation,
            adapter=other.adapter or self.adapter,
            session_id=other.session_id or self.session_id,
            user_id=other.user_id or self.user_id,
            agent_id=other.agent_id or self.agent_id,
            extra={**self.extra, **other.extra},
        )


# Thread-local storage for context
_context_storage = threading.local()


def get_current_context() -> Optional[LogContext]:
    """Get the current logging context for this thread."""
    return getattr(_context_storage, "context", None)


def set_current_context(context: LogContext) -> None:
    """Set the current logging context for this thread."""
    _context_storage.context = context


def clear_current_context() -> None:
    """Clear the current logging context."""
    if hasattr(_context_storage, "context"):
        delattr(_context_storage, "context")


# =============================================================================
# JSON LOG FORMATTER
# =============================================================================

class StructuredJSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Output format:
    {
        "timestamp": "2026-02-04T12:00:00.000Z",
        "level": "INFO",
        "component": "rag_pipeline",
        "message": "Query processed",
        "context": {...},
        "data": {...}
    }
    """

    RESERVED_ATTRS = frozenset([
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName",
        "taskName",  # Python 3.12+
    ])

    def __init__(
        self,
        include_timestamp: bool = True,
        include_location: bool = True,
        include_process: bool = False,
        indent: Optional[int] = None,
    ):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_location = include_location
        self.include_process = include_process
        self.indent = indent

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        log_entry: Dict[str, Any] = {}

        # Timestamp (ISO 8601 format)
        if self.include_timestamp:
            log_entry["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Level
        log_entry["level"] = record.levelname

        # Component (from logger name)
        log_entry["component"] = record.name

        # Message
        log_entry["message"] = record.getMessage()

        # Location info
        if self.include_location:
            log_entry["location"] = {
                "file": record.filename,
                "function": record.funcName,
                "line": record.lineno,
            }

        # Process info (optional, for multi-process debugging)
        if self.include_process:
            log_entry["process"] = {
                "id": record.process,
                "name": record.processName,
                "thread": record.thread,
                "thread_name": record.threadName,
            }

        # Context from thread-local storage
        context = get_current_context()
        if context:
            ctx_dict = context.to_dict()
            if ctx_dict:
                log_entry["context"] = ctx_dict

        # Extra data passed to the log call
        extra_data = {}
        for key, value in record.__dict__.items():
            if key not in self.RESERVED_ATTRS and not key.startswith("_"):
                extra_data[key] = self._serialize_value(value)

        if extra_data:
            log_entry["data"] = extra_data

        # Exception info
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        return json.dumps(log_entry, default=str, indent=self.indent)

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for JSON output."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Enum):
            return value.value
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if hasattr(value, "__dict__"):
            return {k: self._serialize_value(v) for k, v in value.__dict__.items()
                    if not k.startswith("_")}
        return str(value)


# =============================================================================
# STRUCTURED LOGGER
# =============================================================================

class StructuredLogger:
    """
    Structured logger with JSON output, context propagation, and sampling.

    Features:
    - JSON-formatted logs for log aggregation
    - Correlation IDs for request tracing
    - Log sampling for high-volume events
    - Per-component log levels
    - Context propagation through context managers
    """

    def __init__(
        self,
        component: str,
        level: Optional[int] = None,
        default_sample_rate: float = 1.0,
    ):
        """
        Initialize a structured logger.

        Args:
            component: Component name (e.g., "rag_pipeline", "adapter.exa")
            level: Log level (defaults to component-specific level)
            default_sample_rate: Default sampling rate (1.0 = log all)
        """
        self.component = component
        self.default_sample_rate = default_sample_rate

        # Create underlying Python logger
        self._logger = logging.getLogger(component)

        # Set level from component config or explicit override
        if level is not None:
            self._logger.setLevel(level)
        else:
            self._logger.setLevel(get_component_log_level(component))

    @contextmanager
    def correlation_context(
        self,
        correlation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        **extra,
    ):
        """
        Context manager for setting correlation context.

        Usage:
            with logger.correlation_context(request_id="req-123"):
                logger.info("Processing")  # Includes request_id in context
        """
        # Generate correlation ID if not provided
        if correlation_id is None and request_id is None:
            correlation_id = f"corr-{uuid.uuid4().hex[:12]}"

        new_context = LogContext(
            correlation_id=correlation_id,
            request_id=request_id,
            trace_id=trace_id,
            span_id=span_id or f"span-{uuid.uuid4().hex[:8]}",
            component=self.component,
            extra=extra,
        )

        # Merge with existing context
        existing = get_current_context()
        if existing:
            new_context = existing.merge(new_context)
            # Track parent span
            if existing.span_id:
                new_context.parent_span_id = existing.span_id

        previous_context = existing
        set_current_context(new_context)

        try:
            yield new_context
        finally:
            if previous_context:
                set_current_context(previous_context)
            else:
                clear_current_context()

    @contextmanager
    def operation_context(self, operation: str, **extra):
        """
        Context manager for tracking an operation.

        Usage:
            with logger.operation_context("search"):
                logger.info("Starting search")
        """
        context = get_current_context() or LogContext(component=self.component)
        new_context = LogContext(
            operation=operation,
            span_id=f"span-{uuid.uuid4().hex[:8]}",
            extra=extra,
        )
        merged = context.merge(new_context)
        merged.parent_span_id = context.span_id

        previous = get_current_context()
        set_current_context(merged)

        start_time = time.time()
        try:
            yield merged
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self._log(
                logging.DEBUG,
                f"Operation {operation} completed",
                duration_ms=duration_ms,
            )
            if previous:
                set_current_context(previous)
            else:
                clear_current_context()

    def _should_sample(self, sample_rate: float) -> bool:
        """Determine if this log should be sampled (recorded)."""
        if sample_rate >= 1.0:
            return True
        return random.random() < sample_rate

    def _log(
        self,
        level: int,
        message: str,
        sample_rate: Optional[float] = None,
        exc_info: bool = False,
        **kwargs,
    ) -> None:
        """Internal logging method."""
        # Check sampling
        rate = sample_rate if sample_rate is not None else self.default_sample_rate
        if not self._should_sample(rate):
            return

        # Check if this level is enabled
        if not self._logger.isEnabledFor(level):
            return

        # Add sample rate to data if not 1.0 (indicates sampling was applied)
        if rate < 1.0:
            kwargs["sample_rate"] = rate

        # Log with extra data
        self._logger.log(level, message, exc_info=exc_info, extra=kwargs)

    def debug(self, message: str, sample_rate: Optional[float] = None, **kwargs) -> None:
        """Log at DEBUG level."""
        self._log(logging.DEBUG, message, sample_rate=sample_rate, **kwargs)

    def info(self, message: str, sample_rate: Optional[float] = None, **kwargs) -> None:
        """Log at INFO level."""
        self._log(logging.INFO, message, sample_rate=sample_rate, **kwargs)

    def warning(self, message: str, sample_rate: Optional[float] = None, **kwargs) -> None:
        """Log at WARNING level."""
        self._log(logging.WARNING, message, sample_rate=sample_rate, **kwargs)

    def error(self, message: str, sample_rate: Optional[float] = None, **kwargs) -> None:
        """Log at ERROR level."""
        self._log(logging.ERROR, message, sample_rate=sample_rate, **kwargs)

    def critical(self, message: str, sample_rate: Optional[float] = None, **kwargs) -> None:
        """Log at CRITICAL level."""
        self._log(logging.CRITICAL, message, sample_rate=sample_rate, **kwargs)

    def exception(self, message: str, sample_rate: Optional[float] = None, **kwargs) -> None:
        """Log an exception at ERROR level with traceback."""
        self._log(logging.ERROR, message, sample_rate=sample_rate, exc_info=True, **kwargs)

    def log_with_timing(self, message: str, start_time: float, **kwargs) -> None:
        """Log a message with elapsed time."""
        duration_ms = (time.time() - start_time) * 1000
        self.info(message, duration_ms=duration_ms, **kwargs)


# =============================================================================
# LOGGING DECORATORS
# =============================================================================

def logged(
    component: Optional[str] = None,
    operation: Optional[str] = None,
    log_args: bool = False,
    log_result: bool = False,
    sample_rate: float = 1.0,
):
    """
    Decorator for automatic logging of function calls.

    Args:
        component: Component name (defaults to module name)
        operation: Operation name (defaults to function name)
        log_args: Whether to log function arguments
        log_result: Whether to log function result
        sample_rate: Sampling rate for logs
    """
    def decorator(func: Callable) -> Callable:
        func_component = component or func.__module__
        func_operation = operation or func.__name__
        logger = get_logger(func_component)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            log_data: Dict[str, Any] = {"operation": func_operation}

            if log_args:
                log_data["args"] = str(args)[:500]
                log_data["kwargs"] = str(kwargs)[:500]

            logger.debug(f"Starting {func_operation}", sample_rate=sample_rate, **log_data)

            try:
                result = await func(*args, **kwargs)

                duration_ms = (time.time() - start_time) * 1000
                success_data = {
                    "operation": func_operation,
                    "duration_ms": duration_ms,
                    "status": "success",
                }
                if log_result:
                    success_data["result"] = str(result)[:500]

                logger.info(f"Completed {func_operation}", sample_rate=sample_rate, **success_data)
                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.exception(
                    f"Failed {func_operation}",
                    sample_rate=1.0,  # Always log errors
                    operation=func_operation,
                    duration_ms=duration_ms,
                    error_type=type(e).__name__,
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            log_data: Dict[str, Any] = {"operation": func_operation}

            if log_args:
                log_data["args"] = str(args)[:500]
                log_data["kwargs"] = str(kwargs)[:500]

            logger.debug(f"Starting {func_operation}", sample_rate=sample_rate, **log_data)

            try:
                result = func(*args, **kwargs)

                duration_ms = (time.time() - start_time) * 1000
                success_data = {
                    "operation": func_operation,
                    "duration_ms": duration_ms,
                    "status": "success",
                }
                if log_result:
                    success_data["result"] = str(result)[:500]

                logger.info(f"Completed {func_operation}", sample_rate=sample_rate, **success_data)
                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.exception(
                    f"Failed {func_operation}",
                    sample_rate=1.0,
                    operation=func_operation,
                    duration_ms=duration_ms,
                    error_type=type(e).__name__,
                )
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# =============================================================================
# LOGGER FACTORY
# =============================================================================

_loggers: Dict[str, StructuredLogger] = {}
_logging_configured = False
_lock = threading.Lock()


def configure_logging(
    level: int = logging.INFO,
    json_format: bool = True,
    include_location: bool = True,
    include_process: bool = False,
    log_file: Optional[str] = None,
    indent: Optional[int] = None,
) -> None:
    """
    Configure the global logging system.

    Args:
        level: Root log level
        json_format: Use JSON format (True for production)
        include_location: Include file/line info in logs
        include_process: Include process/thread info
        log_file: Optional log file path
        indent: JSON indent for pretty printing (None for compact)
    """
    global _logging_configured

    with _lock:
        if _logging_configured:
            return

        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Clear existing handlers
        root_logger.handlers.clear()

        # Create formatter
        if json_format:
            formatter = StructuredJSONFormatter(
                include_location=include_location,
                include_process=include_process,
                indent=indent,
            )
        else:
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
            )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # File handler (optional)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        _logging_configured = True


def get_logger(component: str, **kwargs) -> StructuredLogger:
    """
    Get or create a structured logger for a component.

    Args:
        component: Component name (e.g., "rag_pipeline", "adapter.exa")
        **kwargs: Additional arguments passed to StructuredLogger

    Returns:
        StructuredLogger instance
    """
    # Ensure logging is configured
    if not _logging_configured:
        configure_logging()

    with _lock:
        if component not in _loggers:
            _loggers[component] = StructuredLogger(component, **kwargs)
        return _loggers[component]


def reset_logging() -> None:
    """Reset logging configuration (for testing)."""
    global _logging_configured, _loggers

    with _lock:
        _loggers.clear()
        _logging_configured = False
        clear_current_context()


# =============================================================================
# CORRELATION ID UTILITIES
# =============================================================================

def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return f"corr-{uuid.uuid4().hex[:12]}"


def generate_request_id() -> str:
    """Generate a new request ID."""
    return f"req-{uuid.uuid4().hex[:16]}"


def generate_trace_id() -> str:
    """Generate a new trace ID (compatible with OpenTelemetry format)."""
    return uuid.uuid4().hex


def generate_span_id() -> str:
    """Generate a new span ID."""
    return f"span-{uuid.uuid4().hex[:8]}"


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Core classes
    "StructuredLogger",
    "StructuredJSONFormatter",
    "LogContext",
    # Factory functions
    "get_logger",
    "configure_logging",
    "reset_logging",
    # Context management
    "get_current_context",
    "set_current_context",
    "clear_current_context",
    # Component levels
    "ComponentLogLevel",
    "COMPONENT_LOG_LEVELS",
    "set_component_log_level",
    "get_component_log_level",
    # Decorators
    "logged",
    # Utilities
    "generate_correlation_id",
    "generate_request_id",
    "generate_trace_id",
    "generate_span_id",
]
