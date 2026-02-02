"""
Unleashed Platform Monitoring & Observability

Comprehensive metrics collection, distributed tracing, health checks,
and performance profiling for SDK operations.
"""

import asyncio
import time
import threading
import functools
import json
from typing import (
    Dict, Any, Optional, List, Callable, TypeVar, Generic,
    Union, Set, Tuple, NamedTuple, Awaitable
)
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from contextlib import contextmanager, asynccontextmanager
from collections import defaultdict
import statistics
import uuid
import logging
import weakref

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================================
# Metric Types
# ============================================================================

class MetricType(Enum):
    """Types of metrics supported."""
    COUNTER = "counter"           # Monotonically increasing value
    GAUGE = "gauge"               # Point-in-time value
    HISTOGRAM = "histogram"       # Distribution of values
    SUMMARY = "summary"           # Statistical summary with quantiles
    TIMER = "timer"               # Duration measurements


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ============================================================================
# Metric Data Structures
# ============================================================================

@dataclass
class MetricValue:
    """A single metric value with metadata."""
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class HistogramBucket:
    """Histogram bucket with upper bound and count."""
    le: float  # Less than or equal (upper bound)
    count: int = 0


@dataclass
class MetricDefinition:
    """Definition of a metric."""
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    quantiles: Optional[List[float]] = None  # For summaries
    unit: Optional[str] = None


@dataclass
class TraceSpan:
    """A span in a distributed trace."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    service_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "ok"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds() * 1000


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: Optional[str] = None
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Alert:
    """An alert notification."""
    id: str
    name: str
    severity: AlertSeverity
    message: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


# ============================================================================
# Metric Collectors
# ============================================================================

class Counter:
    """A monotonically increasing counter metric."""

    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.labels = labels or []
        self._values: Dict[Tuple[str, ...], float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, value: float = 1.0, **labels) -> None:
        """Increment the counter."""
        label_values = self._get_label_values(labels)
        with self._lock:
            self._values[label_values] += value

    def get(self, **labels) -> float:
        """Get current counter value."""
        label_values = self._get_label_values(labels)
        with self._lock:
            return self._values[label_values]

    def _get_label_values(self, labels: Dict[str, str]) -> Tuple[str, ...]:
        return tuple(labels.get(l, "") for l in self.labels)

    def collect(self) -> List[MetricValue]:
        """Collect all metric values."""
        values = []
        with self._lock:
            for label_values, value in self._values.items():
                labels = dict(zip(self.labels, label_values))
                values.append(MetricValue(value=value, labels=labels))
        return values


class Gauge:
    """A point-in-time gauge metric."""

    def __init__(self, name: str, description: str, labels: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.labels = labels or []
        self._values: Dict[Tuple[str, ...], float] = defaultdict(float)
        self._lock = threading.Lock()

    def set(self, value: float, **labels) -> None:
        """Set the gauge value."""
        label_values = self._get_label_values(labels)
        with self._lock:
            self._values[label_values] = value

    def inc(self, value: float = 1.0, **labels) -> None:
        """Increment the gauge."""
        label_values = self._get_label_values(labels)
        with self._lock:
            self._values[label_values] += value

    def dec(self, value: float = 1.0, **labels) -> None:
        """Decrement the gauge."""
        label_values = self._get_label_values(labels)
        with self._lock:
            self._values[label_values] -= value

    def get(self, **labels) -> float:
        """Get current gauge value."""
        label_values = self._get_label_values(labels)
        with self._lock:
            return self._values[label_values]

    def _get_label_values(self, labels: Dict[str, str]) -> Tuple[str, ...]:
        return tuple(labels.get(l, "") for l in self.labels)

    def collect(self) -> List[MetricValue]:
        """Collect all metric values."""
        values = []
        with self._lock:
            for label_values, value in self._values.items():
                labels = dict(zip(self.labels, label_values))
                values.append(MetricValue(value=value, labels=labels))
        return values


class Histogram:
    """A histogram metric for measuring distributions."""

    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None
    ):
        self.name = name
        self.description = description
        self.labels = labels or []
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._counts: Dict[Tuple[str, ...], List[int]] = defaultdict(
            lambda: [0] * len(self.buckets)
        )
        self._sums: Dict[Tuple[str, ...], float] = defaultdict(float)
        self._totals: Dict[Tuple[str, ...], int] = defaultdict(int)
        self._lock = threading.Lock()

    def observe(self, value: float, **labels) -> None:
        """Record an observation."""
        label_values = self._get_label_values(labels)
        with self._lock:
            self._sums[label_values] += value
            self._totals[label_values] += 1
            for i, bucket in enumerate(self.buckets):
                if value <= bucket:
                    self._counts[label_values][i] += 1

    def get_stats(self, **labels) -> Dict[str, Any]:
        """Get histogram statistics."""
        label_values = self._get_label_values(labels)
        with self._lock:
            total = self._totals[label_values]
            sum_val = self._sums[label_values]
            counts = self._counts[label_values].copy()

        return {
            "count": total,
            "sum": sum_val,
            "mean": sum_val / total if total > 0 else 0,
            "buckets": dict(zip(self.buckets, counts))
        }

    def _get_label_values(self, labels: Dict[str, str]) -> Tuple[str, ...]:
        return tuple(labels.get(l, "") for l in self.labels)

    @contextmanager
    def time(self, **labels):
        """Context manager for timing operations."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.observe(time.perf_counter() - start, **labels)


class Summary:
    """A summary metric with quantile calculations."""

    DEFAULT_QUANTILES = [0.5, 0.9, 0.95, 0.99]

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        quantiles: Optional[List[float]] = None,
        max_samples: int = 1000
    ):
        self.name = name
        self.description = description
        self.labels = labels or []
        self.quantiles = quantiles or self.DEFAULT_QUANTILES
        self.max_samples = max_samples
        self._samples: Dict[Tuple[str, ...], List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def observe(self, value: float, **labels) -> None:
        """Record an observation."""
        label_values = self._get_label_values(labels)
        with self._lock:
            samples = self._samples[label_values]
            samples.append(value)
            # Keep only recent samples
            if len(samples) > self.max_samples:
                self._samples[label_values] = samples[-self.max_samples:]

    def get_quantiles(self, **labels) -> Dict[float, float]:
        """Get quantile values."""
        label_values = self._get_label_values(labels)
        with self._lock:
            samples = self._samples[label_values].copy()

        if not samples:
            return {q: 0.0 for q in self.quantiles}

        sorted_samples = sorted(samples)
        result = {}
        for q in self.quantiles:
            idx = int(q * len(sorted_samples))
            idx = min(idx, len(sorted_samples) - 1)
            result[q] = sorted_samples[idx]
        return result

    def get_stats(self, **labels) -> Dict[str, Any]:
        """Get summary statistics."""
        label_values = self._get_label_values(labels)
        with self._lock:
            samples = self._samples[label_values].copy()

        if not samples:
            return {
                "count": 0,
                "sum": 0,
                "mean": 0,
                "min": 0,
                "max": 0,
                "quantiles": {q: 0.0 for q in self.quantiles}
            }

        return {
            "count": len(samples),
            "sum": sum(samples),
            "mean": statistics.mean(samples),
            "min": min(samples),
            "max": max(samples),
            "stddev": statistics.stdev(samples) if len(samples) > 1 else 0,
            "quantiles": self.get_quantiles(**labels)
        }

    def _get_label_values(self, labels: Dict[str, str]) -> Tuple[str, ...]:
        return tuple(labels.get(l, "") for l in self.labels)


# ============================================================================
# Metric Registry
# ============================================================================

class MetricRegistry:
    """Central registry for all metrics."""

    _instance: Optional["MetricRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "MetricRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._summaries: Dict[str, Summary] = {}
        self._registry_lock = threading.Lock()
        self._initialized = True

    def counter(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None
    ) -> Counter:
        """Get or create a counter metric."""
        with self._registry_lock:
            if name not in self._counters:
                self._counters[name] = Counter(name, description, labels)
            return self._counters[name]

    def gauge(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None
    ) -> Gauge:
        """Get or create a gauge metric."""
        with self._registry_lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name, description, labels)
            return self._gauges[name]

    def histogram(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None
    ) -> Histogram:
        """Get or create a histogram metric."""
        with self._registry_lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(name, description, labels, buckets)
            return self._histograms[name]

    def summary(
        self,
        name: str,
        description: str,
        labels: Optional[List[str]] = None,
        quantiles: Optional[List[float]] = None
    ) -> Summary:
        """Get or create a summary metric."""
        with self._registry_lock:
            if name not in self._summaries:
                self._summaries[name] = Summary(name, description, labels, quantiles)
            return self._summaries[name]

    def collect_all(self) -> Dict[str, Any]:
        """Collect all metrics."""
        result = {
            "counters": {},
            "gauges": {},
            "histograms": {},
            "summaries": {},
            "timestamp": datetime.utcnow().isoformat()
        }

        with self._registry_lock:
            for name, counter in self._counters.items():
                result["counters"][name] = [
                    {"value": v.value, "labels": v.labels}
                    for v in counter.collect()
                ]

            for name, gauge in self._gauges.items():
                result["gauges"][name] = [
                    {"value": v.value, "labels": v.labels}
                    for v in gauge.collect()
                ]

            for name, histogram in self._histograms.items():
                result["histograms"][name] = histogram.get_stats()

            for name, summary in self._summaries.items():
                result["summaries"][name] = summary.get_stats()

        return result

    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        with self._registry_lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._summaries.clear()


# ============================================================================
# Distributed Tracing
# ============================================================================

class Tracer:
    """Distributed tracing implementation."""

    _instance: Optional["Tracer"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> "Tracer":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, service_name: str = "unleashed"):
        if self._initialized:
            return

        self.service_name = service_name
        self._active_spans: Dict[str, TraceSpan] = {}
        self._completed_spans: List[TraceSpan] = []
        self._max_completed = 10000
        self._tracer_lock = threading.Lock()
        self._context_var: Dict[int, str] = {}  # thread_id -> span_id
        self._initialized = True

    def start_span(
        self,
        operation_name: str,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> TraceSpan:
        """Start a new trace span."""
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())[:16]

        # Inherit trace_id from parent if available
        if parent_span_id and parent_span_id in self._active_spans:
            parent = self._active_spans[parent_span_id]
            trace_id = parent.trace_id

        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            service_name=self.service_name,
            start_time=datetime.utcnow(),
            attributes=attributes or {}
        )

        with self._tracer_lock:
            self._active_spans[span_id] = span
            self._context_var[threading.get_ident()] = span_id

        return span

    def end_span(
        self,
        span: TraceSpan,
        status: str = "ok",
        error: Optional[Exception] = None
    ) -> None:
        """End a trace span."""
        span.end_time = datetime.utcnow()
        span.status = status

        if error:
            span.status = "error"
            span.attributes["error.type"] = type(error).__name__
            span.attributes["error.message"] = str(error)

        with self._tracer_lock:
            if span.span_id in self._active_spans:
                del self._active_spans[span.span_id]

            self._completed_spans.append(span)

            # Trim old spans
            if len(self._completed_spans) > self._max_completed:
                self._completed_spans = self._completed_spans[-self._max_completed:]

            # Clear context if this was the active span
            thread_id = threading.get_ident()
            if self._context_var.get(thread_id) == span.span_id:
                del self._context_var[thread_id]

    def add_event(
        self,
        span: TraceSpan,
        name: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add an event to a span."""
        span.events.append({
            "name": name,
            "timestamp": datetime.utcnow().isoformat(),
            "attributes": attributes or {}
        })

    def get_current_span(self) -> Optional[TraceSpan]:
        """Get the current active span for this thread."""
        thread_id = threading.get_ident()
        with self._tracer_lock:
            span_id = self._context_var.get(thread_id)
            if span_id:
                return self._active_spans.get(span_id)
        return None

    @contextmanager
    def trace(
        self,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Context manager for tracing operations."""
        parent = self.get_current_span()
        parent_id = parent.span_id if parent else None

        span = self.start_span(operation_name, parent_id, attributes)
        try:
            yield span
            self.end_span(span)
        except Exception as e:
            self.end_span(span, error=e)
            raise

    @asynccontextmanager
    async def atrace(
        self,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Async context manager for tracing operations."""
        parent = self.get_current_span()
        parent_id = parent.span_id if parent else None

        span = self.start_span(operation_name, parent_id, attributes)
        try:
            yield span
            self.end_span(span)
        except Exception as e:
            self.end_span(span, error=e)
            raise

    def get_traces(
        self,
        trace_id: Optional[str] = None,
        operation_name: Optional[str] = None,
        limit: int = 100
    ) -> List[TraceSpan]:
        """Get completed traces with optional filtering."""
        with self._tracer_lock:
            spans = self._completed_spans.copy()

        if trace_id:
            spans = [s for s in spans if s.trace_id == trace_id]
        if operation_name:
            spans = [s for s in spans if s.operation_name == operation_name]

        return spans[-limit:]

    def export_jaeger(self) -> List[Dict[str, Any]]:
        """Export traces in Jaeger-compatible format."""
        with self._tracer_lock:
            spans = self._completed_spans.copy()

        return [
            {
                "traceID": s.trace_id,
                "spanID": s.span_id,
                "parentSpanID": s.parent_span_id,
                "operationName": s.operation_name,
                "serviceName": s.service_name,
                "startTime": int(s.start_time.timestamp() * 1000000),
                "duration": int(s.duration_ms * 1000) if s.duration_ms else 0,
                "tags": [{"key": k, "value": v} for k, v in s.attributes.items()],
                "logs": [
                    {
                        "timestamp": e["timestamp"],
                        "fields": [{"key": k, "value": v} for k, v in e.get("attributes", {}).items()]
                    }
                    for e in s.events
                ]
            }
            for s in spans
        ]


# ============================================================================
# Health Checks
# ============================================================================

class HealthChecker:
    """Health check manager for adapters and services."""

    def __init__(self):
        self._checks: Dict[str, Callable[[], Awaitable[HealthCheckResult]]] = {}
        self._results: Dict[str, HealthCheckResult] = {}
        self._lock = asyncio.Lock()

    def register(
        self,
        name: str,
        check_fn: Callable[[], Awaitable[HealthCheckResult]]
    ) -> None:
        """Register a health check."""
        self._checks[name] = check_fn

    def unregister(self, name: str) -> None:
        """Unregister a health check."""
        self._checks.pop(name, None)
        self._results.pop(name, None)

    async def check(self, name: str) -> HealthCheckResult:
        """Run a specific health check."""
        if name not in self._checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check '{name}' not found"
            )

        check_fn = self._checks[name]
        start = time.perf_counter()

        try:
            result = await asyncio.wait_for(check_fn(), timeout=30.0)
            result.duration_ms = (time.perf_counter() - start) * 1000

            async with self._lock:
                self._results[name] = result

            return result
        except asyncio.TimeoutError:
            result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message="Health check timed out",
                duration_ms=(time.perf_counter() - start) * 1000
            )
            async with self._lock:
                self._results[name] = result
            return result
        except Exception as e:
            result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                duration_ms=(time.perf_counter() - start) * 1000
            )
            async with self._lock:
                self._results[name] = result
            return result

    async def check_all(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks concurrently."""
        tasks = [self.check(name) for name in self._checks]
        await asyncio.gather(*tasks, return_exceptions=True)

        async with self._lock:
            return self._results.copy()

    async def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        results = await self.check_all()

        if not results:
            return HealthStatus.UNKNOWN

        statuses = [r.status for r in results.values()]

        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        if any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY

        return HealthStatus.UNKNOWN

    def get_cached_results(self) -> Dict[str, HealthCheckResult]:
        """Get cached health check results."""
        return self._results.copy()


# ============================================================================
# Alert Manager
# ============================================================================

class AlertRule:
    """Definition of an alert rule."""

    def __init__(
        self,
        name: str,
        metric_name: str,
        condition: Callable[[float], bool],
        severity: AlertSeverity,
        message_template: str,
        cooldown: timedelta = timedelta(minutes=5),
        labels: Optional[Dict[str, str]] = None
    ):
        self.name = name
        self.metric_name = metric_name
        self.condition = condition
        self.severity = severity
        self.message_template = message_template
        self.cooldown = cooldown
        self.labels = labels or {}
        self.last_triggered: Optional[datetime] = None


class AlertManager:
    """Manages alerts and notifications."""

    def __init__(self):
        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._handlers: List[Callable[[Alert], Awaitable[None]]] = []
        self._lock = asyncio.Lock()

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self._rules[rule.name] = rule

    def remove_rule(self, name: str) -> None:
        """Remove an alert rule."""
        self._rules.pop(name, None)

    def add_handler(self, handler: Callable[[Alert], Awaitable[None]]) -> None:
        """Add an alert handler."""
        self._handlers.append(handler)

    async def evaluate(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> List[Alert]:
        """Evaluate metric against alert rules."""
        triggered_alerts = []
        labels = labels or {}

        for rule in self._rules.values():
            if rule.metric_name != metric_name:
                continue

            # Check label matching
            if rule.labels:
                if not all(labels.get(k) == v for k, v in rule.labels.items()):
                    continue

            # Check cooldown
            if rule.last_triggered:
                if datetime.utcnow() - rule.last_triggered < rule.cooldown:
                    continue

            # Evaluate condition
            if rule.condition(value):
                alert = Alert(
                    id=str(uuid.uuid4()),
                    name=rule.name,
                    severity=rule.severity,
                    message=rule.message_template.format(
                        metric=metric_name,
                        value=value,
                        **labels
                    ),
                    metric_name=metric_name,
                    metric_value=value,
                    labels=labels
                )

                rule.last_triggered = datetime.utcnow()
                triggered_alerts.append(alert)

                async with self._lock:
                    self._active_alerts[alert.id] = alert
                    self._alert_history.append(alert)

                # Notify handlers
                for handler in self._handlers:
                    try:
                        await handler(alert)
                    except Exception as e:
                        logger.error(f"Alert handler error: {e}")

        return triggered_alerts

    async def resolve(self, alert_id: str) -> bool:
        """Resolve an active alert."""
        async with self._lock:
            if alert_id in self._active_alerts:
                alert = self._active_alerts.pop(alert_id)
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                return True
        return False

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self._active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        return self._alert_history[-limit:]


# ============================================================================
# Performance Profiler
# ============================================================================

@dataclass
class ProfileResult:
    """Result of a profiling session."""
    name: str
    total_time_ms: float
    call_count: int
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    samples: List[float]


class Profiler:
    """Performance profiler for SDK operations."""

    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self._profiles: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def record(self, name: str, duration_ms: float) -> None:
        """Record a profiling sample."""
        with self._lock:
            samples = self._profiles[name]
            samples.append(duration_ms)
            if len(samples) > self.max_samples:
                self._profiles[name] = samples[-self.max_samples:]

    @contextmanager
    def profile(self, name: str):
        """Context manager for profiling."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self.record(name, duration_ms)

    @asynccontextmanager
    async def aprofile(self, name: str):
        """Async context manager for profiling."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            self.record(name, duration_ms)

    def get_profile(self, name: str) -> Optional[ProfileResult]:
        """Get profiling results for a name."""
        with self._lock:
            samples = self._profiles.get(name, []).copy()

        if not samples:
            return None

        return ProfileResult(
            name=name,
            total_time_ms=sum(samples),
            call_count=len(samples),
            avg_time_ms=statistics.mean(samples),
            min_time_ms=min(samples),
            max_time_ms=max(samples),
            samples=samples
        )

    def get_all_profiles(self) -> Dict[str, ProfileResult]:
        """Get all profiling results."""
        results = {}
        with self._lock:
            names = list(self._profiles.keys())

        for name in names:
            profile = self.get_profile(name)
            if profile:
                results[name] = profile

        return results

    def reset(self, name: Optional[str] = None) -> None:
        """Reset profiling data."""
        with self._lock:
            if name:
                self._profiles.pop(name, None)
            else:
                self._profiles.clear()


# ============================================================================
# Decorators
# ============================================================================

def timed(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to time function execution."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        registry = MetricRegistry()
        histogram = registry.histogram(
            f"{metric_name}_duration_seconds",
            f"Duration of {metric_name}",
            list(labels.keys()) if labels else None
        )
        counter = registry.counter(
            f"{metric_name}_total",
            f"Total calls to {metric_name}",
            list(labels.keys()) if labels else None
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with histogram.time(**(labels or {})):
                counter.inc(**(labels or {}))
                return func(*args, **kwargs)

        return wrapper
    return decorator


def traced(operation_name: Optional[str] = None):
    """Decorator to trace function execution."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        tracer = Tracer()
        op_name = operation_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.trace(op_name):
                return func(*args, **kwargs)

        return wrapper
    return decorator


def profiled(name: Optional[str] = None):
    """Decorator to profile function execution."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        profiler = Profiler()
        profile_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with profiler.profile(profile_name):
                return func(*args, **kwargs)

        return wrapper
    return decorator


# ============================================================================
# Pre-configured SDK Metrics
# ============================================================================

def create_sdk_metrics() -> Dict[str, Any]:
    """Create standard metrics for SDK operations."""
    registry = MetricRegistry()

    return {
        # Adapter metrics
        "adapter_calls": registry.counter(
            "unleashed_adapter_calls_total",
            "Total number of adapter calls",
            ["adapter", "method", "status"]
        ),
        "adapter_latency": registry.histogram(
            "unleashed_adapter_latency_seconds",
            "Adapter call latency in seconds",
            ["adapter", "method"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        ),
        "adapter_errors": registry.counter(
            "unleashed_adapter_errors_total",
            "Total number of adapter errors",
            ["adapter", "error_type"]
        ),

        # Pipeline metrics
        "pipeline_executions": registry.counter(
            "unleashed_pipeline_executions_total",
            "Total number of pipeline executions",
            ["pipeline", "status"]
        ),
        "pipeline_duration": registry.histogram(
            "unleashed_pipeline_duration_seconds",
            "Pipeline execution duration in seconds",
            ["pipeline"],
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0]
        ),

        # Cache metrics
        "cache_hits": registry.counter(
            "unleashed_cache_hits_total",
            "Total number of cache hits",
            ["cache_type"]
        ),
        "cache_misses": registry.counter(
            "unleashed_cache_misses_total",
            "Total number of cache misses",
            ["cache_type"]
        ),
        "cache_size": registry.gauge(
            "unleashed_cache_size_bytes",
            "Current cache size in bytes",
            ["cache_type"]
        ),

        # Resource metrics
        "active_tasks": registry.gauge(
            "unleashed_active_tasks",
            "Number of active async tasks",
            ["task_type"]
        ),
        "queue_depth": registry.gauge(
            "unleashed_queue_depth",
            "Current queue depth",
            ["queue_name"]
        ),

        # Circuit breaker metrics
        "circuit_breaker_state": registry.gauge(
            "unleashed_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=open, 2=half-open)",
            ["adapter"]
        ),
        "circuit_breaker_failures": registry.counter(
            "unleashed_circuit_breaker_failures_total",
            "Total circuit breaker failures",
            ["adapter"]
        ),
    }


# ============================================================================
# Monitoring Dashboard Data
# ============================================================================

class MonitoringDashboard:
    """Aggregates monitoring data for dashboard display."""

    def __init__(
        self,
        registry: Optional[MetricRegistry] = None,
        tracer: Optional[Tracer] = None,
        health_checker: Optional[HealthChecker] = None,
        alert_manager: Optional[AlertManager] = None,
        profiler: Optional[Profiler] = None
    ):
        self.registry = registry or MetricRegistry()
        self.tracer = tracer or Tracer()
        self.health_checker = health_checker or HealthChecker()
        self.alert_manager = alert_manager or AlertManager()
        self.profiler = profiler or Profiler()

    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get all monitoring data for dashboard."""
        health_results = await self.health_checker.check_all()
        overall_health = await self.health_checker.get_overall_status()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "health": {
                "overall": overall_health.value,
                "checks": {
                    name: {
                        "status": r.status.value,
                        "message": r.message,
                        "duration_ms": r.duration_ms
                    }
                    for name, r in health_results.items()
                }
            },
            "metrics": self.registry.collect_all(),
            "traces": {
                "recent": [
                    {
                        "trace_id": s.trace_id,
                        "operation": s.operation_name,
                        "duration_ms": s.duration_ms,
                        "status": s.status
                    }
                    for s in self.tracer.get_traces(limit=20)
                ]
            },
            "alerts": {
                "active": [
                    {
                        "id": a.id,
                        "name": a.name,
                        "severity": a.severity.value,
                        "message": a.message
                    }
                    for a in self.alert_manager.get_active_alerts()
                ],
                "count": len(self.alert_manager.get_active_alerts())
            },
            "profiles": {
                name: {
                    "avg_ms": p.avg_time_ms,
                    "min_ms": p.min_time_ms,
                    "max_ms": p.max_time_ms,
                    "calls": p.call_count
                }
                for name, p in self.profiler.get_all_profiles().items()
            }
        }

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        metrics = self.registry.collect_all()

        # Counters
        for name, values in metrics.get("counters", {}).items():
            lines.append(f"# TYPE {name} counter")
            for v in values:
                label_str = ",".join(f'{k}="{v}"' for k, v in v["labels"].items())
                if label_str:
                    lines.append(f"{name}{{{label_str}}} {v['value']}")
                else:
                    lines.append(f"{name} {v['value']}")

        # Gauges
        for name, values in metrics.get("gauges", {}).items():
            lines.append(f"# TYPE {name} gauge")
            for v in values:
                label_str = ",".join(f'{k}="{v}"' for k, v in v["labels"].items())
                if label_str:
                    lines.append(f"{name}{{{label_str}}} {v['value']}")
                else:
                    lines.append(f"{name} {v['value']}")

        # Histograms (simplified)
        for name, stats in metrics.get("histograms", {}).items():
            lines.append(f"# TYPE {name} histogram")
            lines.append(f"{name}_count {stats['count']}")
            lines.append(f"{name}_sum {stats['sum']}")

        return "\n".join(lines)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Types
    "MetricType",
    "HealthStatus",
    "AlertSeverity",

    # Data structures
    "MetricValue",
    "HistogramBucket",
    "MetricDefinition",
    "TraceSpan",
    "HealthCheckResult",
    "Alert",
    "ProfileResult",

    # Metric collectors
    "Counter",
    "Gauge",
    "Histogram",
    "Summary",

    # Core components
    "MetricRegistry",
    "Tracer",
    "HealthChecker",
    "AlertRule",
    "AlertManager",
    "Profiler",

    # Dashboard
    "MonitoringDashboard",

    # Decorators
    "timed",
    "traced",
    "profiled",

    # Utilities
    "create_sdk_metrics",
]
