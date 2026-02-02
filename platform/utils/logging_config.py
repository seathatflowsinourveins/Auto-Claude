# -*- coding: utf-8 -*-
"""
Unified Logging Configuration for Unleash Platform

Provides structured logging with:
- JSON output for production
- Pretty console output for development
- Unicode-safe formatting for Windows
- Context binding for request tracing
- Performance metrics integration

Based on:
- structlog patterns
- Canonical log lines (Stripe)
- 12-Factor App methodology
"""

import os
import sys
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps

from .encoding import configure_encoding, safe_str, is_windows, supports_unicode

# Configure encoding first
configure_encoding()


# =============================================================================
# Structured Log Entry
# =============================================================================

@dataclass
class LogContext:
    """Thread-local context for log enrichment."""
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    operation: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# Global context (thread-safe via contextvars in production)
_log_context = LogContext()


def get_context() -> LogContext:
    """Get current logging context."""
    return _log_context


def set_context(**kwargs) -> None:
    """Set logging context values."""
    global _log_context
    for key, value in kwargs.items():
        if hasattr(_log_context, key):
            setattr(_log_context, key, value)
        else:
            _log_context.extra[key] = value


def clear_context() -> None:
    """Clear logging context."""
    global _log_context
    _log_context = LogContext()


@contextmanager
def log_context(**kwargs):
    """Context manager for temporary log context."""
    global _log_context
    old_context = LogContext(
        request_id=_log_context.request_id,
        session_id=_log_context.session_id,
        user_id=_log_context.user_id,
        operation=_log_context.operation,
        extra=dict(_log_context.extra)
    )
    set_context(**kwargs)
    try:
        yield
    finally:
        _log_context = old_context


# =============================================================================
# Structured Formatter
# =============================================================================

class StructuredFormatter(logging.Formatter):
    """
    Formatter that outputs structured JSON logs.

    Includes:
    - ISO timestamp
    - Log level
    - Logger name
    - Message
    - Context fields
    - Exception info (if any)
    """

    def format(self, record: logging.LogRecord) -> str:
        # Base log entry
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add context
        ctx = get_context()
        if ctx.request_id:
            log_entry["request_id"] = ctx.request_id
        if ctx.session_id:
            log_entry["session_id"] = ctx.session_id
        if ctx.user_id:
            log_entry["user_id"] = ctx.user_id
        if ctx.operation:
            log_entry["operation"] = ctx.operation
        if ctx.extra:
            log_entry.update(ctx.extra)

        # Add source location
        log_entry["source"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }

        # Add exception if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
            }

        # Add any extra fields from the record
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry, default=str)


class PrettyFormatter(logging.Formatter):
    """
    Formatter for human-readable console output.

    Uses colors when supported, ASCII-safe output on Windows.
    """

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and not is_windows()

    def format(self, record: logging.LogRecord) -> str:
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Level with optional color
        level = record.levelname
        if self.use_colors:
            color = self.COLORS.get(level, "")
            level_str = f"{color}{level:8}{self.RESET}"
        else:
            level_str = f"{level:8}"

        # Logger name (shortened)
        name = record.name
        if len(name) > 20:
            name = "..." + name[-17:]

        # Message
        message = record.getMessage()

        # Make safe for Windows
        if is_windows() and not supports_unicode():
            message = safe_str(message)

        # Context suffix
        ctx = get_context()
        ctx_parts = []
        if ctx.request_id:
            ctx_parts.append(f"req={ctx.request_id[:8]}")
        if ctx.operation:
            ctx_parts.append(f"op={ctx.operation}")
        ctx_str = f" [{', '.join(ctx_parts)}]" if ctx_parts else ""

        # Format line
        line = f"{timestamp} {level_str} {name:20} {message}{ctx_str}"

        # Add exception
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            if is_windows() and not supports_unicode():
                exc_text = safe_str(exc_text)
            line += f"\n{exc_text}"

        return line


# =============================================================================
# Logger Configuration
# =============================================================================

def configure_logging(
    level: int = logging.INFO,
    structured: bool = False,
    log_file: Optional[str] = None,
    include_modules: Optional[List[str]] = None,
) -> None:
    """
    Configure logging for the Unleash platform.

    Args:
        level: Logging level (default: INFO)
        structured: Use JSON structured output (default: False for development)
        log_file: Optional file path for file logging
        include_modules: List of module names to configure
    """
    # Create formatter
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = PrettyFormatter(use_colors=True)

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(StructuredFormatter())  # Always JSON for files
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)

    # Configure specific modules
    modules = include_modules or [
        "core.research_engine",
        "core.ecosystem_orchestrator",
        "core.orchestrator",
        "core.resilience",
        "utils",
    ]
    for module in modules:
        module_logger = logging.getLogger(module)
        module_logger.setLevel(level)


# =============================================================================
# Logging Decorators
# =============================================================================

def log_operation(operation_name: Optional[str] = None, log_args: bool = False):
    """
    Decorator to log function entry/exit with timing.

    Args:
        operation_name: Name for the operation (default: function name)
        log_args: Whether to log function arguments
    """
    def decorator(func: Callable) -> Callable:
        name = operation_name or func.__name__
        logger = logging.getLogger(func.__module__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()

            with log_context(operation=name):
                # Log entry
                if log_args:
                    logger.debug(f"Starting {name}", extra={
                        "extra_fields": {"args": str(args)[:100], "kwargs": str(kwargs)[:100]}
                    })
                else:
                    logger.debug(f"Starting {name}")

                try:
                    result = func(*args, **kwargs)
                    duration = time.perf_counter() - start
                    logger.debug(f"Completed {name} in {duration:.3f}s")
                    return result
                except Exception as e:
                    duration = time.perf_counter() - start
                    logger.error(f"Failed {name} after {duration:.3f}s: {e}")
                    raise

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()

            with log_context(operation=name):
                if log_args:
                    logger.debug(f"Starting {name}", extra={
                        "extra_fields": {"args": str(args)[:100], "kwargs": str(kwargs)[:100]}
                    })
                else:
                    logger.debug(f"Starting {name}")

                try:
                    result = await func(*args, **kwargs)
                    duration = time.perf_counter() - start
                    logger.debug(f"Completed {name} in {duration:.3f}s")
                    return result
                except Exception as e:
                    duration = time.perf_counter() - start
                    logger.error(f"Failed {name} after {duration:.3f}s: {e}")
                    raise

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


def log_error(logger_name: Optional[str] = None):
    """Decorator to log exceptions with full context."""
    def decorator(func: Callable) -> Callable:
        log = logging.getLogger(logger_name or func.__module__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log.exception(f"Exception in {func.__name__}: {e}")
                raise

        return wrapper
    return decorator


# =============================================================================
# Convenience Functions
# =============================================================================

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    return logging.getLogger(name)


# Import asyncio for type checking
import asyncio
import inspect
