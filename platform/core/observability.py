"""
Unleashed Platform - Unified Observability Layer

Integrates logging, metrics, tracing, and alerting into a cohesive observability stack.
Provides structured logging, context propagation, and export capabilities.
"""

import asyncio
import inspect
import json
import sys
import logging
import threading
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from contextlib import contextmanager, asynccontextmanager
from functools import wraps
import traceback
import uuid

# Import core monitoring components
try:
    from .monitoring import (
        MetricRegistry,
        Tracer,
        HealthChecker,
        AlertManager,
        AlertRule,
        AlertSeverity,
        MonitoringDashboard,
        Counter,
        Gauge,
        Histogram,
    )
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


# ============================================================================
# Structured Logging
# ============================================================================

class LogLevel(Enum):
    """Log levels matching Python logging."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class LogContext:
    """Context information attached to log entries."""
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    adapter: Optional[str] = None
    pipeline: Optional[str] = None
    operation: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class StructuredLogFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(self, include_timestamp: bool = True):
        super().__init__()
        self.include_timestamp = include_timestamp

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if self.include_timestamp:
            log_entry["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Add context if available
        if hasattr(record, "context") and record.context:
            ctx = record.context
            if ctx.trace_id:
                log_entry["trace_id"] = ctx.trace_id
            if ctx.span_id:
                log_entry["span_id"] = ctx.span_id
            if ctx.adapter:
                log_entry["adapter"] = ctx.adapter
            if ctx.pipeline:
                log_entry["pipeline"] = ctx.pipeline
            if ctx.operation:
                log_entry["operation"] = ctx.operation
            if ctx.extra:
                log_entry.update(ctx.extra)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }

        return json.dumps(log_entry, default=str)


class ContextualLogger:
    """Logger with automatic context propagation."""

    _context_var: Dict[int, LogContext] = {}
    _lock = threading.Lock()

    def __init__(self, name: str, level: LogLevel = LogLevel.INFO):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level.value)
        self._name = name

    @classmethod
    def set_context(cls, context: LogContext) -> None:
        """Set context for current thread."""
        with cls._lock:
            cls._context_var[threading.get_ident()] = context

    @classmethod
    def get_context(cls) -> Optional[LogContext]:
        """Get context for current thread."""
        with cls._lock:
            return cls._context_var.get(threading.get_ident())

    @classmethod
    def clear_context(cls) -> None:
        """Clear context for current thread."""
        with cls._lock:
            cls._context_var.pop(threading.get_ident(), None)

    @contextmanager
    def with_context(self, **kwargs):
        """Context manager to set logging context."""
        previous = self.get_context()
        new_context = LogContext(**kwargs)
        self.set_context(new_context)
        try:
            yield
        finally:
            if previous:
                self.set_context(previous)
            else:
                self.clear_context()

    def _log(self, level: int, msg: str, *args, **kwargs):
        """Internal log method with context attachment."""
        context = self.get_context()
        extra = kwargs.pop("extra", {})
        extra["context"] = context
        self._logger.log(level, msg, *args, extra=extra, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self._log(logging.CRITICAL, msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        kwargs["exc_info"] = True
        self._log(logging.ERROR, msg, *args, **kwargs)


# ============================================================================
# Unified Observability
# ============================================================================

@dataclass
class ObservabilityConfig:
    """Configuration for the observability layer."""
    # Logging
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "json"  # "json" or "text"
    log_file: Optional[str] = None
    log_to_console: bool = True

    # Metrics
    metrics_enabled: bool = True
    metrics_prefix: str = "unleashed"

    # Tracing
    tracing_enabled: bool = True
    trace_sample_rate: float = 1.0  # 1.0 = 100% sampling

    # Alerting
    alerting_enabled: bool = True
    alert_cooldown_seconds: int = 300

    # Export
    export_endpoint: Optional[str] = None
    export_interval_seconds: int = 60


class Observability:
    """Unified observability layer for the Unleashed platform."""

    _instance: Optional["Observability"] = None
    _lock = threading.Lock()

    def __new__(cls, config: Optional[ObservabilityConfig] = None) -> "Observability":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, config: Optional[ObservabilityConfig] = None):
        if self._initialized:
            return

        self.config = config or ObservabilityConfig()
        self._setup_logging()

        if MONITORING_AVAILABLE:
            self._registry = MetricRegistry()
            self._tracer = Tracer()
            self._health_checker = HealthChecker()
            self._alert_manager = AlertManager()
            self._dashboard = MonitoringDashboard(
                registry=self._registry,
                tracer=self._tracer,
                health_checker=self._health_checker,
                alert_manager=self._alert_manager
            )
        else:
            self._registry = None
            self._tracer = None
            self._health_checker = None
            self._alert_manager = None
            self._dashboard = None

        self._loggers: Dict[str, ContextualLogger] = {}
        self._initialized = True

    def _setup_logging(self) -> None:
        """Configure the logging system."""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.config.log_level.value)

        # Remove existing handlers
        root_logger.handlers.clear()

        # Create formatter
        if self.config.log_format == "json":
            formatter = StructuredLogFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
            )

        # Console handler
        if self.config.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        # File handler
        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

    def get_logger(self, name: str) -> ContextualLogger:
        """Get or create a contextual logger."""
        if name not in self._loggers:
            self._loggers[name] = ContextualLogger(name, self.config.log_level)
        return self._loggers[name]

    # ----------------------------------------
    # Metrics API
    # ----------------------------------------

    def counter(self, name: str, description: str, labels: Optional[List[str]] = None):
        """Create or get a counter metric."""
        if not self._registry:
            return None
        full_name = f"{self.config.metrics_prefix}_{name}"
        return self._registry.counter(full_name, description, labels)

    def gauge(self, name: str, description: str, labels: Optional[List[str]] = None):
        """Create or get a gauge metric."""
        if not self._registry:
            return None
        full_name = f"{self.config.metrics_prefix}_{name}"
        return self._registry.gauge(full_name, description, labels)

    def histogram(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None
    ):
        """Create or get a histogram metric."""
        if not self._registry:
            return None
        full_name = f"{self.config.metrics_prefix}_{name}"
        return self._registry.histogram(full_name, description, labels, buckets)

    def collect_metrics(self) -> Dict[str, Any]:
        """Collect all metrics."""
        if not self._registry:
            return {}
        return self._registry.collect_all()

    # ----------------------------------------
    # Tracing API
    # ----------------------------------------

    @contextmanager
    def trace(self, operation: str, attributes: Optional[Dict[str, Any]] = None):
        """Context manager for tracing operations."""
        if not self._tracer or not self.config.tracing_enabled:
            yield None
            return

        with self._tracer.trace(operation, attributes) as span:
            # Set logging context from span
            ContextualLogger.set_context(LogContext(
                trace_id=span.trace_id,
                span_id=span.span_id,
                operation=operation
            ))
            try:
                yield span
            finally:
                ContextualLogger.clear_context()

    @asynccontextmanager
    async def atrace(self, operation: str, attributes: Optional[Dict[str, Any]] = None):
        """Async context manager for tracing operations."""
        if not self._tracer or not self.config.tracing_enabled:
            yield None
            return

        async with self._tracer.atrace(operation, attributes) as span:
            ContextualLogger.set_context(LogContext(
                trace_id=span.trace_id,
                span_id=span.span_id,
                operation=operation
            ))
            try:
                yield span
            finally:
                ContextualLogger.clear_context()

    def get_traces(self, limit: int = 100) -> List[Any]:
        """Get recent traces."""
        if not self._tracer:
            return []
        return self._tracer.get_traces(limit=limit)

    # ----------------------------------------
    # Health Checks API
    # ----------------------------------------

    def register_health_check(
        self,
        name: str,
        check_fn: Callable[[], Any]
    ) -> None:
        """Register a health check."""
        if self._health_checker:
            self._health_checker.register(name, check_fn)

    async def check_health(self) -> Dict[str, Any]:
        """Run all health checks."""
        if not self._health_checker:
            return {"status": "unknown", "message": "Health checker not available"}
        return await self._health_checker.check_all()

    # ----------------------------------------
    # Alerting API
    # ----------------------------------------

    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        if self._alert_manager:
            self._alert_manager.add_rule(rule)

    async def evaluate_alert(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> List[Any]:
        """Evaluate metric against alert rules."""
        if not self._alert_manager:
            return []
        return await self._alert_manager.evaluate(metric_name, value, labels)

    def get_active_alerts(self) -> List[Any]:
        """Get all active alerts."""
        if not self._alert_manager:
            return []
        return self._alert_manager.get_active_alerts()

    # ----------------------------------------
    # Dashboard API
    # ----------------------------------------

    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        if not self._dashboard:
            return {
                "status": "unavailable",
                "message": "Monitoring components not available"
            }
        return await self._dashboard.get_dashboard_data()

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        if not self._dashboard:
            return ""
        return self._dashboard.export_prometheus()


# ============================================================================
# Decorators
# ============================================================================

def observed(
    operation_name: Optional[str] = None,
    log_args: bool = False,
    log_result: bool = False,
    measure_time: bool = True
):
    """Decorator for full observability on a function."""
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or func.__name__

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            obs = get_observability()
            logger = obs.get_logger(func.__module__)

            # Log entry
            log_data = {"operation": op_name}
            if log_args:
                log_data["args"] = str(args)[:200]
                log_data["kwargs"] = str(kwargs)[:200]
            logger.info(f"Starting {op_name}", extra={"data": log_data})

            # Track with tracing
            async with obs.atrace(op_name) as span:
                start_time = datetime.now(timezone.utc)
                try:
                    result = await func(*args, **kwargs)

                    # Log success
                    duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    log_data = {"operation": op_name, "duration_ms": duration_ms, "status": "success"}
                    if log_result:
                        log_data["result"] = str(result)[:200]
                    logger.info(f"Completed {op_name}", extra={"data": log_data})

                    # Record metric
                    if measure_time and obs._registry:
                        histogram = obs.histogram(
                            "operation_duration_seconds",
                            "Operation duration",
                            ["operation"]
                        )
                        if histogram:
                            histogram.observe(duration_ms / 1000, operation=op_name)

                    return result

                except Exception as e:
                    duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    logger.exception(f"Failed {op_name}", extra={
                        "data": {
                            "operation": op_name,
                            "duration_ms": duration_ms,
                            "error": str(e)
                        }
                    })
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            obs = get_observability()
            logger = obs.get_logger(func.__module__)

            log_data = {"operation": op_name}
            if log_args:
                log_data["args"] = str(args)[:200]
            logger.info(f"Starting {op_name}", extra={"data": log_data})

            with obs.trace(op_name):
                start_time = datetime.now(timezone.utc)
                try:
                    result = func(*args, **kwargs)

                    duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    log_data = {"operation": op_name, "duration_ms": duration_ms, "status": "success"}
                    if log_result:
                        log_data["result"] = str(result)[:200]
                    logger.info(f"Completed {op_name}", extra={"data": log_data})

                    if measure_time and obs._registry:
                        histogram = obs.histogram(
                            "operation_duration_seconds",
                            "Operation duration",
                            ["operation"]
                        )
                        if histogram:
                            histogram.observe(duration_ms / 1000, operation=op_name)

                    return result

                except Exception as e:
                    duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    logger.exception(f"Failed {op_name}", extra={
                        "data": {
                            "operation": op_name,
                            "duration_ms": duration_ms,
                            "error": str(e)
                        }
                    })
                    raise

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# ============================================================================
# Global Instance
# ============================================================================

_observability: Optional[Observability] = None


def get_observability(config: Optional[ObservabilityConfig] = None) -> Observability:
    """Get the global observability instance."""
    global _observability
    if _observability is None:
        _observability = Observability(config)
    return _observability


def configure_observability(config: ObservabilityConfig) -> Observability:
    """Configure and return the global observability instance."""
    global _observability
    _observability = Observability(config)
    return _observability


def reset_observability() -> None:
    """Reset the global observability instance (for testing)."""
    global _observability
    _observability = None


# ============================================================================
# Quick Setup Functions
# ============================================================================

def setup_development_observability() -> Observability:
    """Quick setup for development environment."""
    config = ObservabilityConfig(
        log_level=LogLevel.DEBUG,
        log_format="text",
        log_to_console=True,
        metrics_enabled=True,
        tracing_enabled=True,
        alerting_enabled=False
    )
    return configure_observability(config)


def setup_production_observability(
    log_file: str = "/var/log/unleashed/platform.log",
    export_endpoint: Optional[str] = None
) -> Observability:
    """Quick setup for production environment."""
    config = ObservabilityConfig(
        log_level=LogLevel.INFO,
        log_format="json",
        log_file=log_file,
        log_to_console=False,
        metrics_enabled=True,
        tracing_enabled=True,
        trace_sample_rate=0.1,  # 10% sampling in production
        alerting_enabled=True,
        export_endpoint=export_endpoint
    )
    return configure_observability(config)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Types
    "LogLevel",
    "LogContext",
    "ObservabilityConfig",

    # Logging
    "StructuredLogFormatter",
    "ContextualLogger",

    # Core
    "Observability",
    "get_observability",
    "configure_observability",
    "reset_observability",

    # Decorators
    "observed",

    # Quick Setup
    "setup_development_observability",
    "setup_production_observability",

    # Constants
    "MONITORING_AVAILABLE",
]
