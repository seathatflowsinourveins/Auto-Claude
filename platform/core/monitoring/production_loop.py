"""
UNLEASH Production Monitoring Loop

A comprehensive production-ready monitoring system integrating:
- Opik (Comet-ML) for LLM observability
- Langfuse for LLM analytics
- Phoenix (Arize) for LLM tracing
- Prometheus for metrics export
- OpenTelemetry for distributed tracing

Features:
1. Metrics Collection - Request latency (p50/p95/p99), token usage, cache hit rates
2. Distributed Tracing - Span context propagation, cross-agent correlation
3. Alerting - Latency thresholds, error rate spikes, memory pressure
4. Dashboard Export - Prometheus, OTLP, JSON formats
5. Cost Tracking - Token costs by model, API costs by adapter, budget alerts

Integration with existing Ralph production monitoring.

Created: 2026-02-04
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import statistics
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# CONFIGURATION
# =============================================================================

class ExportFormat(Enum):
    """Supported export formats for metrics and traces."""
    PROMETHEUS = "prometheus"
    OTLP = "otlp"
    JSON = "json"
    OPIK = "opik"
    LANGFUSE = "langfuse"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ModelTier(Enum):
    """Model pricing tiers for 3-tier routing."""
    TIER_1_WASM = "tier_1_wasm"      # Agent Booster (WASM) - <1ms, $0
    TIER_2_HAIKU = "tier_2_haiku"    # Haiku - ~500ms, $0.0002
    TIER_3_OPUS = "tier_3_opus"      # Sonnet/Opus - 2-5s, $0.003-0.015


@dataclass
class MonitoringConfig:
    """Configuration for the production monitoring loop."""

    # Feature toggles
    enabled: bool = True
    metrics_enabled: bool = True
    tracing_enabled: bool = True
    alerting_enabled: bool = True
    cost_tracking_enabled: bool = True

    # Sampling
    trace_sample_rate: float = 0.1  # 10% in production
    metrics_sample_rate: float = 1.0  # 100% for metrics

    # Intervals
    metrics_export_interval_s: int = 60
    health_check_interval_s: int = 30
    alert_evaluation_interval_s: int = 10

    # Thresholds
    latency_warning_ms: float = 1000.0
    latency_critical_ms: float = 5000.0
    error_rate_warning: float = 0.05  # 5%
    error_rate_critical: float = 0.10  # 10%
    memory_pressure_warning_mb: float = 1024.0

    # Budget
    daily_budget_usd: float = 100.0
    hourly_budget_usd: float = 10.0
    budget_alert_threshold: float = 0.8  # Alert at 80%

    # Retention
    trace_retention_count: int = 10000
    metrics_retention_minutes: int = 60
    alert_retention_count: int = 1000

    # Exporters
    opik_api_key: Optional[str] = None
    opik_project_name: str = "unleash-platform"
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"
    prometheus_endpoint: Optional[str] = None
    otlp_endpoint: Optional[str] = None

    # Model pricing (per 1M tokens)
    model_pricing: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        # Anthropic
        "claude-opus-4-5": {"input": 15.0, "output": 75.0},
        "claude-sonnet-4": {"input": 3.0, "output": 15.0},
        "claude-haiku-4": {"input": 0.25, "output": 1.25},
        "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
        "claude-3-opus": {"input": 15.0, "output": 75.0},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
        # OpenAI
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "o1": {"input": 15.0, "output": 60.0},
        "o1-mini": {"input": 3.0, "output": 12.0},
        # Default fallback
        "default": {"input": 1.0, "output": 3.0},
    })

    def __post_init__(self):
        # Load from environment if not set
        if not self.opik_api_key:
            self.opik_api_key = os.environ.get("OPIK_API_KEY")
        if not self.langfuse_public_key:
            self.langfuse_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        if not self.langfuse_secret_key:
            self.langfuse_secret_key = os.environ.get("LANGFUSE_SECRET_KEY")


# =============================================================================
# METRICS COLLECTION
# =============================================================================

@dataclass
class LatencyMetrics:
    """Latency metrics with percentiles."""
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    avg: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    count: int = 0

    @classmethod
    def from_samples(cls, samples: List[float]) -> "LatencyMetrics":
        """Calculate latency metrics from samples."""
        if not samples:
            return cls()

        sorted_samples = sorted(samples)
        n = len(sorted_samples)

        return cls(
            p50=sorted_samples[int(n * 0.5)] if n > 0 else 0,
            p90=sorted_samples[int(n * 0.9)] if n > 0 else 0,
            p95=sorted_samples[int(n * 0.95)] if n > 0 else 0,
            p99=sorted_samples[int(n * 0.99)] if n > 0 else 0,
            avg=statistics.mean(sorted_samples) if sorted_samples else 0,
            min_val=min(sorted_samples) if sorted_samples else 0,
            max_val=max(sorted_samples) if sorted_samples else 0,
            count=n,
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "p50": round(self.p50, 2),
            "p90": round(self.p90, 2),
            "p95": round(self.p95, 2),
            "p99": round(self.p99, 2),
            "avg": round(self.avg, 2),
            "min": round(self.min_val, 2),
            "max": round(self.max_val, 2),
            "count": self.count,
        }


@dataclass
class TokenUsage:
    """Token usage tracking."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, input_tokens: int, output_tokens: int, cost_usd: float = 0.0) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens
        self.cost_usd += cost_usd

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 4),
            "size_bytes": self.size_bytes,
        }


@dataclass
class ErrorMetrics:
    """Error tracking by adapter/type."""
    total_errors: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    by_adapter: Dict[str, int] = field(default_factory=dict)
    recent_errors: List[Dict[str, Any]] = field(default_factory=list)

    def record_error(
        self,
        error_type: str,
        adapter: str,
        message: str,
        max_recent: int = 100
    ) -> None:
        self.total_errors += 1
        self.by_type[error_type] = self.by_type.get(error_type, 0) + 1
        self.by_adapter[adapter] = self.by_adapter.get(adapter, 0) + 1

        self.recent_errors.append({
            "type": error_type,
            "adapter": adapter,
            "message": message[:500],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        if len(self.recent_errors) > max_recent:
            self.recent_errors = self.recent_errors[-max_recent:]

    def error_rate(self, total_requests: int) -> float:
        return self.total_errors / total_requests if total_requests > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_errors": self.total_errors,
            "by_type": dict(self.by_type),
            "by_adapter": dict(self.by_adapter),
            "recent_count": len(self.recent_errors),
        }


class MetricsCollector:
    """
    Central metrics collector for the monitoring loop.

    Collects:
    - Request latency (p50, p90, p95, p99)
    - Token usage by model tier
    - Cache hit rates
    - Error rates by adapter
    """

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self._lock = threading.Lock()

        # Latency samples (per adapter/method)
        self._latency_samples: Dict[str, List[float]] = defaultdict(list)

        # Token usage (per model tier)
        self._token_usage: Dict[str, TokenUsage] = defaultdict(TokenUsage)

        # Cache metrics (per cache type)
        self._cache_metrics: Dict[str, CacheMetrics] = defaultdict(CacheMetrics)

        # Error metrics
        self._error_metrics = ErrorMetrics()

        # Request counters
        self._request_counts: Dict[str, int] = defaultdict(int)

        # Timestamps for rolling windows
        self._window_start = datetime.now(timezone.utc)
        self._total_requests = 0

    def record_latency(
        self,
        adapter: str,
        method: str,
        latency_ms: float
    ) -> None:
        """Record a latency observation."""
        with self._lock:
            key = f"{adapter}.{method}"
            self._latency_samples[key].append(latency_ms)
            self._request_counts[key] += 1
            self._total_requests += 1

            # Trim samples for memory efficiency
            max_samples = 10000
            if len(self._latency_samples[key]) > max_samples:
                self._latency_samples[key] = self._latency_samples[key][-max_samples:]

    def record_tokens(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        tier: ModelTier = ModelTier.TIER_3_OPUS
    ) -> float:
        """Record token usage and return cost."""
        pricing = self.config.model_pricing.get(
            model,
            self.config.model_pricing.get("default", {"input": 1.0, "output": 3.0})
        )

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        with self._lock:
            self._token_usage[tier.value].add(input_tokens, output_tokens, total_cost)
            self._token_usage["total"].add(input_tokens, output_tokens, total_cost)

        return total_cost

    def record_cache_hit(self, cache_type: str = "default") -> None:
        """Record a cache hit."""
        with self._lock:
            self._cache_metrics[cache_type].hits += 1

    def record_cache_miss(self, cache_type: str = "default") -> None:
        """Record a cache miss."""
        with self._lock:
            self._cache_metrics[cache_type].misses += 1

    def record_error(
        self,
        error_type: str,
        adapter: str,
        message: str
    ) -> None:
        """Record an error."""
        with self._lock:
            self._error_metrics.record_error(error_type, adapter, message)

    def get_latency_metrics(self, adapter: Optional[str] = None) -> Dict[str, LatencyMetrics]:
        """Get latency metrics, optionally filtered by adapter."""
        with self._lock:
            result = {}
            for key, samples in self._latency_samples.items():
                if adapter is None or key.startswith(adapter):
                    result[key] = LatencyMetrics.from_samples(samples)
            return result

    def get_token_usage(self) -> Dict[str, TokenUsage]:
        """Get token usage by tier."""
        with self._lock:
            return dict(self._token_usage)

    def get_cache_metrics(self) -> Dict[str, CacheMetrics]:
        """Get cache metrics."""
        with self._lock:
            return dict(self._cache_metrics)

    def get_error_metrics(self) -> ErrorMetrics:
        """Get error metrics."""
        with self._lock:
            return self._error_metrics

    def get_error_rate(self) -> float:
        """Get overall error rate."""
        with self._lock:
            return self._error_metrics.error_rate(self._total_requests)

    def get_summary(self) -> Dict[str, Any]:
        """Get a complete metrics summary."""
        with self._lock:
            latencies = {
                k: v.to_dict()
                for k, v in self.get_latency_metrics().items()
            }

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "window_duration_s": (
                    datetime.now(timezone.utc) - self._window_start
                ).total_seconds(),
                "total_requests": self._total_requests,
                "latency": latencies,
                "tokens": {k: v.to_dict() for k, v in self._token_usage.items()},
                "cache": {k: v.to_dict() for k, v in self._cache_metrics.items()},
                "errors": self._error_metrics.to_dict(),
                "error_rate": round(self.get_error_rate(), 4),
            }

    def reset_window(self) -> Dict[str, Any]:
        """Reset the metrics window and return the previous window's data."""
        with self._lock:
            summary = self.get_summary()

            self._latency_samples.clear()
            self._token_usage.clear()
            self._cache_metrics.clear()
            self._error_metrics = ErrorMetrics()
            self._request_counts.clear()
            self._window_start = datetime.now(timezone.utc)
            self._total_requests = 0

            return summary


# =============================================================================
# DISTRIBUTED TRACING
# =============================================================================

@dataclass
class SpanContext:
    """Span context for trace propagation."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)

    def child(self) -> "SpanContext":
        """Create a child span context."""
        return SpanContext(
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4())[:16],
            parent_span_id=self.span_id,
            baggage=dict(self.baggage),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "baggage": dict(self.baggage),
        }


@dataclass
class Span:
    """A distributed trace span."""
    context: SpanContext
    operation_name: str
    service_name: str
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    status: str = "ok"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds() * 1000

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attributes": attributes or {},
        })

    def set_status(self, status: str, message: Optional[str] = None) -> None:
        """Set span status."""
        self.status = status
        if message:
            self.attributes["status.message"] = message

    def finish(self, status: str = "ok") -> None:
        """Mark the span as finished."""
        self.end_time = datetime.now(timezone.utc)
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "operation_name": self.operation_name,
            "service_name": self.service_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
        }


class DistributedTracer:
    """
    Distributed tracing with span context propagation.

    Features:
    - Trace context propagation across agents
    - Async span completion
    - Cross-agent trace correlation
    """

    def __init__(self, config: MonitoringConfig, service_name: str = "unleash"):
        self.config = config
        self.service_name = service_name
        self._lock = threading.Lock()

        # Active spans by thread
        self._active_spans: Dict[int, List[Span]] = defaultdict(list)

        # Completed spans
        self._completed_spans: List[Span] = []

        # Context propagation
        self._current_context: Dict[int, SpanContext] = {}

    def _thread_id(self) -> int:
        return threading.get_ident()

    def start_span(
        self,
        operation_name: str,
        parent_context: Optional[SpanContext] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> Span:
        """Start a new span."""
        thread_id = self._thread_id()

        # Determine parent context
        if parent_context is None:
            parent_context = self._current_context.get(thread_id)

        # Create span context
        if parent_context:
            context = parent_context.child()
        else:
            context = SpanContext(
                trace_id=str(uuid.uuid4()),
                span_id=str(uuid.uuid4())[:16],
            )

        span = Span(
            context=context,
            operation_name=operation_name,
            service_name=self.service_name,
            attributes=attributes or {},
        )

        with self._lock:
            self._active_spans[thread_id].append(span)
            self._current_context[thread_id] = context

        return span

    def end_span(self, span: Span, status: str = "ok") -> None:
        """End a span."""
        span.finish(status)
        thread_id = self._thread_id()

        with self._lock:
            # Remove from active spans
            if thread_id in self._active_spans:
                try:
                    self._active_spans[thread_id].remove(span)
                except ValueError:
                    pass

                # Update current context to parent
                if self._active_spans[thread_id]:
                    self._current_context[thread_id] = self._active_spans[thread_id][-1].context
                else:
                    self._current_context.pop(thread_id, None)

            # Store completed span
            self._completed_spans.append(span)

            # Trim completed spans
            if len(self._completed_spans) > self.config.trace_retention_count:
                self._completed_spans = self._completed_spans[-self.config.trace_retention_count:]

    def get_current_context(self) -> Optional[SpanContext]:
        """Get current span context for this thread."""
        return self._current_context.get(self._thread_id())

    def inject_context(self, headers: Dict[str, str]) -> None:
        """Inject trace context into headers for propagation."""
        context = self.get_current_context()
        if context:
            headers["X-Trace-ID"] = context.trace_id
            headers["X-Span-ID"] = context.span_id
            if context.parent_span_id:
                headers["X-Parent-Span-ID"] = context.parent_span_id
            for key, value in context.baggage.items():
                headers[f"X-Baggage-{key}"] = value

    def extract_context(self, headers: Dict[str, str]) -> Optional[SpanContext]:
        """Extract trace context from headers."""
        trace_id = headers.get("X-Trace-ID")
        span_id = headers.get("X-Span-ID")

        if not trace_id or not span_id:
            return None

        baggage = {}
        for key, value in headers.items():
            if key.startswith("X-Baggage-"):
                baggage[key[10:]] = value

        return SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=headers.get("X-Parent-Span-ID"),
            baggage=baggage,
        )

    @contextmanager
    def trace(
        self,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Context manager for tracing synchronous operations."""
        span = self.start_span(operation_name, attributes=attributes)
        try:
            yield span
            self.end_span(span, "ok")
        except Exception as e:
            span.set_status("error", str(e))
            self.end_span(span, "error")
            raise

    @asynccontextmanager
    async def atrace(
        self,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Async context manager for tracing async operations."""
        span = self.start_span(operation_name, attributes=attributes)
        try:
            yield span
            self.end_span(span, "ok")
        except Exception as e:
            span.set_status("error", str(e))
            self.end_span(span, "error")
            raise

    def get_traces(
        self,
        trace_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Span]:
        """Get completed traces."""
        with self._lock:
            spans = list(self._completed_spans)

        if trace_id:
            spans = [s for s in spans if s.context.trace_id == trace_id]

        return spans[-limit:]

    def get_trace_tree(self, trace_id: str) -> Dict[str, Any]:
        """Get a hierarchical view of a trace."""
        spans = self.get_traces(trace_id=trace_id, limit=1000)

        if not spans:
            return {}

        # Build tree structure
        spans_by_id = {s.context.span_id: s for s in spans}
        root_spans = [s for s in spans if s.context.parent_span_id is None]

        def build_tree(span: Span) -> Dict[str, Any]:
            children = [s for s in spans if s.context.parent_span_id == span.context.span_id]
            return {
                **span.to_dict(),
                "children": [build_tree(c) for c in children],
            }

        return {
            "trace_id": trace_id,
            "roots": [build_tree(s) for s in root_spans],
            "total_spans": len(spans),
        }


# =============================================================================
# ALERTING
# =============================================================================

@dataclass
class AlertRule:
    """Definition of an alert rule."""
    name: str
    metric_name: str
    condition: Callable[[float], bool]
    severity: AlertSeverity
    message_template: str
    cooldown: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    labels: Dict[str, str] = field(default_factory=dict)
    last_triggered: Optional[datetime] = None

    def is_in_cooldown(self) -> bool:
        if self.last_triggered is None:
            return False
        return datetime.now(timezone.utc) - self.last_triggered < self.cooldown


@dataclass
class Alert:
    """An alert instance."""
    id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    metric_name: str
    metric_value: float
    labels: Dict[str, str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def resolve(self) -> None:
        self.resolved = True
        self.resolved_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "message": self.message,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "labels": self.labels,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


class AlertManager:
    """
    Alert manager with configurable rules.

    Monitors:
    - Latency threshold breaches
    - Error rate spikes
    - Memory pressure warnings
    - Budget alerts
    """

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._handlers: List[Callable[[Alert], Awaitable[None]]] = []
        self._lock = asyncio.Lock()

        # Initialize default rules
        self._init_default_rules()

    def _init_default_rules(self) -> None:
        """Initialize default alert rules."""
        # Latency alerts
        self.add_rule(AlertRule(
            name="latency_warning",
            metric_name="request_latency_ms",
            condition=lambda v: v > self.config.latency_warning_ms,
            severity=AlertSeverity.WARNING,
            message_template="Request latency {value}ms exceeds warning threshold ({threshold}ms)",
            cooldown=timedelta(minutes=2),
        ))

        self.add_rule(AlertRule(
            name="latency_critical",
            metric_name="request_latency_ms",
            condition=lambda v: v > self.config.latency_critical_ms,
            severity=AlertSeverity.CRITICAL,
            message_template="Request latency {value}ms exceeds critical threshold ({threshold}ms)",
            cooldown=timedelta(minutes=1),
        ))

        # Error rate alerts
        self.add_rule(AlertRule(
            name="error_rate_warning",
            metric_name="error_rate",
            condition=lambda v: v > self.config.error_rate_warning,
            severity=AlertSeverity.WARNING,
            message_template="Error rate {value:.1%} exceeds warning threshold ({threshold:.1%})",
            cooldown=timedelta(minutes=5),
        ))

        self.add_rule(AlertRule(
            name="error_rate_critical",
            metric_name="error_rate",
            condition=lambda v: v > self.config.error_rate_critical,
            severity=AlertSeverity.CRITICAL,
            message_template="Error rate {value:.1%} exceeds critical threshold ({threshold:.1%})",
            cooldown=timedelta(minutes=2),
        ))

        # Budget alerts
        self.add_rule(AlertRule(
            name="budget_warning",
            metric_name="daily_spend_usd",
            condition=lambda v: v > self.config.daily_budget_usd * self.config.budget_alert_threshold,
            severity=AlertSeverity.WARNING,
            message_template="Daily spend ${value:.2f} approaching budget limit (${threshold:.2f})",
            cooldown=timedelta(hours=1),
        ))

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self._rules[rule.name] = rule

    def remove_rule(self, name: str) -> None:
        """Remove an alert rule."""
        self._rules.pop(name, None)

    def add_handler(self, handler: Callable[[Alert], Awaitable[None]]) -> None:
        """Add an alert handler callback."""
        self._handlers.append(handler)

    async def evaluate(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> List[Alert]:
        """Evaluate metric against all rules."""
        triggered = []
        labels = labels or {}

        for rule in self._rules.values():
            if rule.metric_name != metric_name:
                continue

            # Check label matching
            if rule.labels and not all(
                labels.get(k) == v for k, v in rule.labels.items()
            ):
                continue

            # Check cooldown
            if rule.is_in_cooldown():
                continue

            # Evaluate condition
            if rule.condition(value):
                alert = Alert(
                    id=str(uuid.uuid4()),
                    rule_name=rule.name,
                    severity=rule.severity,
                    message=rule.message_template.format(
                        value=value,
                        threshold=self._get_threshold(rule),
                        **labels,
                    ),
                    metric_name=metric_name,
                    metric_value=value,
                    labels=labels,
                )

                rule.last_triggered = datetime.now(timezone.utc)
                triggered.append(alert)

                async with self._lock:
                    self._active_alerts[alert.id] = alert
                    self._alert_history.append(alert)

                    # Trim history
                    if len(self._alert_history) > self.config.alert_retention_count:
                        self._alert_history = self._alert_history[-self.config.alert_retention_count:]

                # Notify handlers
                for handler in self._handlers:
                    try:
                        await handler(alert)
                    except Exception as e:
                        logger.error(f"Alert handler failed: {e}")

        return triggered

    def _get_threshold(self, rule: AlertRule) -> float:
        """Extract threshold from rule for message formatting."""
        # Try to extract from condition (this is a simplification)
        if "latency" in rule.name:
            if "critical" in rule.name:
                return self.config.latency_critical_ms
            return self.config.latency_warning_ms
        if "error_rate" in rule.name:
            if "critical" in rule.name:
                return self.config.error_rate_critical
            return self.config.error_rate_warning
        if "budget" in rule.name:
            return self.config.daily_budget_usd * self.config.budget_alert_threshold
        return 0.0

    async def resolve(self, alert_id: str) -> bool:
        """Resolve an active alert."""
        async with self._lock:
            if alert_id in self._active_alerts:
                alert = self._active_alerts.pop(alert_id)
                alert.resolve()
                return True
        return False

    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return list(self._active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        return self._alert_history[-limit:]


# =============================================================================
# COST TRACKING
# =============================================================================

@dataclass
class CostEntry:
    """A single cost entry."""
    model: str
    tier: ModelTier
    adapter: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "tier": self.tier.value,
            "adapter": self.adapter,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "timestamp": self.timestamp.isoformat(),
        }


class CostTracker:
    """
    Token and API cost tracking.

    Tracks:
    - Token costs by model
    - API call costs by adapter
    - Budget alerts
    """

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self._entries: List[CostEntry] = []
        self._by_model: Dict[str, float] = defaultdict(float)
        self._by_adapter: Dict[str, float] = defaultdict(float)
        self._by_tier: Dict[str, float] = defaultdict(float)
        self._hourly_costs: Dict[str, float] = defaultdict(float)  # hour_key -> cost
        self._daily_costs: Dict[str, float] = defaultdict(float)   # day_key -> cost
        self._lock = threading.Lock()

    def _hour_key(self, dt: Optional[datetime] = None) -> str:
        dt = dt or datetime.now(timezone.utc)
        return dt.strftime("%Y-%m-%d-%H")

    def _day_key(self, dt: Optional[datetime] = None) -> str:
        dt = dt or datetime.now(timezone.utc)
        return dt.strftime("%Y-%m-%d")

    def record_cost(
        self,
        model: str,
        adapter: str,
        input_tokens: int,
        output_tokens: int,
        tier: ModelTier = ModelTier.TIER_3_OPUS
    ) -> CostEntry:
        """Record a cost entry and return the entry with calculated cost."""
        pricing = self.config.model_pricing.get(
            model,
            self.config.model_pricing.get("default", {"input": 1.0, "output": 3.0})
        )

        cost = (
            (input_tokens / 1_000_000) * pricing["input"] +
            (output_tokens / 1_000_000) * pricing["output"]
        )

        entry = CostEntry(
            model=model,
            tier=tier,
            adapter=adapter,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

        with self._lock:
            self._entries.append(entry)
            self._by_model[model] += cost
            self._by_adapter[adapter] += cost
            self._by_tier[tier.value] += cost
            self._hourly_costs[self._hour_key()] += cost
            self._daily_costs[self._day_key()] += cost

            # Trim entries for memory efficiency
            max_entries = 100000
            if len(self._entries) > max_entries:
                self._entries = self._entries[-max_entries:]

        return entry

    def get_total_cost(self) -> float:
        """Get total cost across all entries."""
        with self._lock:
            return sum(self._by_model.values())

    def get_hourly_cost(self, hour_key: Optional[str] = None) -> float:
        """Get cost for a specific hour (current hour by default)."""
        hour_key = hour_key or self._hour_key()
        with self._lock:
            return self._hourly_costs.get(hour_key, 0.0)

    def get_daily_cost(self, day_key: Optional[str] = None) -> float:
        """Get cost for a specific day (current day by default)."""
        day_key = day_key or self._day_key()
        with self._lock:
            return self._daily_costs.get(day_key, 0.0)

    def get_cost_by_model(self) -> Dict[str, float]:
        """Get costs grouped by model."""
        with self._lock:
            return dict(self._by_model)

    def get_cost_by_adapter(self) -> Dict[str, float]:
        """Get costs grouped by adapter."""
        with self._lock:
            return dict(self._by_adapter)

    def get_cost_by_tier(self) -> Dict[str, float]:
        """Get costs grouped by model tier."""
        with self._lock:
            return dict(self._by_tier)

    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        daily_cost = self.get_daily_cost()
        hourly_cost = self.get_hourly_cost()

        return {
            "daily_cost_usd": round(daily_cost, 4),
            "daily_budget_usd": self.config.daily_budget_usd,
            "daily_percent_used": round(daily_cost / self.config.daily_budget_usd * 100, 2),
            "daily_remaining_usd": round(self.config.daily_budget_usd - daily_cost, 4),
            "hourly_cost_usd": round(hourly_cost, 4),
            "hourly_budget_usd": self.config.hourly_budget_usd,
            "hourly_percent_used": round(hourly_cost / self.config.hourly_budget_usd * 100, 2),
            "total_cost_usd": round(self.get_total_cost(), 4),
            "is_daily_warning": daily_cost >= self.config.daily_budget_usd * self.config.budget_alert_threshold,
            "is_hourly_warning": hourly_cost >= self.config.hourly_budget_usd * self.config.budget_alert_threshold,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get a complete cost summary."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_cost_usd": round(self.get_total_cost(), 4),
            "by_model": {k: round(v, 4) for k, v in self._by_model.items()},
            "by_adapter": {k: round(v, 4) for k, v in self._by_adapter.items()},
            "by_tier": {k: round(v, 4) for k, v in self._by_tier.items()},
            "budget_status": self.get_budget_status(),
            "entry_count": len(self._entries),
        }


# =============================================================================
# EXPORTERS
# =============================================================================

class Exporter(ABC):
    """Base class for metrics/trace exporters."""

    @abstractmethod
    async def export_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Export metrics. Returns True on success."""
        pass

    @abstractmethod
    async def export_traces(self, traces: List[Span]) -> bool:
        """Export traces. Returns True on success."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check exporter health."""
        pass


class OpikExporter(Exporter):
    """
    Opik (Comet-ML) exporter for LLM observability.

    Features:
    - Deep LLM tracing
    - LLM-as-a-judge evaluations
    - Experiment tracking
    """

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self._client = None
        self._initialized = False

    async def _ensure_initialized(self) -> bool:
        if self._initialized:
            return self._client is not None

        try:
            import opik

            if self.config.opik_api_key:
                opik.configure(
                    api_key=self.config.opik_api_key,
                    workspace="default",
                    force=True,
                )
                self._client = opik
                logger.info("Opik exporter initialized")
            else:
                logger.warning("Opik API key not configured")
                self._client = None

            self._initialized = True
            return self._client is not None
        except ImportError:
            logger.warning("Opik package not installed")
            self._initialized = True
            return False
        except Exception as e:
            logger.error(f"Opik initialization failed: {e}")
            self._initialized = True
            return False

    async def export_metrics(self, metrics: Dict[str, Any]) -> bool:
        if not await self._ensure_initialized():
            return False

        try:
            # Opik focuses on traces/evaluations rather than raw metrics
            # Log key metrics as experiment parameters
            if hasattr(self._client, "log_metrics"):
                self._client.log_metrics({
                    "total_requests": metrics.get("total_requests", 0),
                    "error_rate": metrics.get("error_rate", 0),
                })
            return True
        except Exception as e:
            logger.error(f"Opik metrics export failed: {e}")
            return False

    async def export_traces(self, traces: List[Span]) -> bool:
        if not await self._ensure_initialized():
            return False

        try:
            for span in traces:
                if hasattr(self._client, "track"):
                    # Use Opik's track context manager style
                    with self._client.track(
                        name=span.operation_name,
                        project_name=self.config.opik_project_name,
                    ) as opik_span:
                        if hasattr(opik_span, "set_attribute"):
                            opik_span.set_attribute("trace_id", span.context.trace_id)
                            opik_span.set_attribute("span_id", span.context.span_id)
                            opik_span.set_attribute("duration_ms", span.duration_ms)
                            opik_span.set_attribute("status", span.status)
                            for k, v in span.attributes.items():
                                opik_span.set_attribute(k, str(v))
            return True
        except Exception as e:
            logger.error(f"Opik trace export failed: {e}")
            return False

    async def health_check(self) -> bool:
        return await self._ensure_initialized()


class LangfuseExporter(Exporter):
    """
    Langfuse exporter for LLM analytics.

    Features:
    - Trace ingestion
    - Prompt management
    - Cost tracking
    - Evaluations
    """

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self._client = None
        self._initialized = False

    async def _ensure_initialized(self) -> bool:
        if self._initialized:
            return self._client is not None

        try:
            from langfuse import Langfuse

            if self.config.langfuse_public_key and self.config.langfuse_secret_key:
                self._client = Langfuse(
                    public_key=self.config.langfuse_public_key,
                    secret_key=self.config.langfuse_secret_key,
                    host=self.config.langfuse_host,
                )
                logger.info("Langfuse exporter initialized")
            else:
                logger.warning("Langfuse keys not configured")
                self._client = None

            self._initialized = True
            return self._client is not None
        except ImportError:
            logger.warning("Langfuse package not installed")
            self._initialized = True
            return False
        except Exception as e:
            logger.error(f"Langfuse initialization failed: {e}")
            self._initialized = True
            return False

    async def export_metrics(self, metrics: Dict[str, Any]) -> bool:
        if not await self._ensure_initialized():
            return False

        try:
            # Langfuse uses scores for metrics
            if self._client:
                self._client.score(
                    name="system_metrics",
                    value=1.0,
                    comment=json.dumps(metrics),
                )
            return True
        except Exception as e:
            logger.error(f"Langfuse metrics export failed: {e}")
            return False

    async def export_traces(self, traces: List[Span]) -> bool:
        if not await self._ensure_initialized():
            return False

        try:
            for span in traces:
                trace = self._client.trace(
                    id=span.context.trace_id,
                    name=span.operation_name,
                    metadata={
                        "span_id": span.context.span_id,
                        "parent_span_id": span.context.parent_span_id,
                        "service": span.service_name,
                        **span.attributes,
                    },
                )

                # Create generation span
                trace.generation(
                    name=span.operation_name,
                    start_time=span.start_time,
                    end_time=span.end_time,
                    metadata=span.attributes,
                )

            # Flush to ensure delivery
            self._client.flush()
            return True
        except Exception as e:
            logger.error(f"Langfuse trace export failed: {e}")
            return False

    async def health_check(self) -> bool:
        if not await self._ensure_initialized():
            return False

        try:
            self._client.flush()
            return True
        except Exception:
            return False


class PrometheusExporter(Exporter):
    """
    Prometheus metrics exporter.

    Exports metrics in Prometheus text format for scraping.
    """

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self._metrics_cache: str = ""
        self._last_export = datetime.now(timezone.utc)

    async def export_metrics(self, metrics: Dict[str, Any]) -> bool:
        try:
            lines = []
            timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)

            # Total requests counter
            lines.append("# TYPE unleash_requests_total counter")
            lines.append(f"unleash_requests_total {metrics.get('total_requests', 0)}")

            # Error rate gauge
            lines.append("# TYPE unleash_error_rate gauge")
            lines.append(f"unleash_error_rate {metrics.get('error_rate', 0)}")

            # Latency histograms
            for key, latency in metrics.get("latency", {}).items():
                safe_key = key.replace(".", "_")
                lines.append(f"# TYPE unleash_latency_{safe_key}_ms summary")
                lines.append(f'unleash_latency_{safe_key}_ms{{quantile="0.5"}} {latency.get("p50", 0)}')
                lines.append(f'unleash_latency_{safe_key}_ms{{quantile="0.9"}} {latency.get("p90", 0)}')
                lines.append(f'unleash_latency_{safe_key}_ms{{quantile="0.95"}} {latency.get("p95", 0)}')
                lines.append(f'unleash_latency_{safe_key}_ms{{quantile="0.99"}} {latency.get("p99", 0)}')
                lines.append(f"unleash_latency_{safe_key}_ms_sum {latency.get('avg', 0) * latency.get('count', 0)}")
                lines.append(f"unleash_latency_{safe_key}_ms_count {latency.get('count', 0)}")

            # Token usage
            for tier, usage in metrics.get("tokens", {}).items():
                safe_tier = tier.replace(".", "_")
                lines.append(f"# TYPE unleash_tokens_{safe_tier}_total counter")
                lines.append(f"unleash_tokens_{safe_tier}_input {usage.get('input_tokens', 0)}")
                lines.append(f"unleash_tokens_{safe_tier}_output {usage.get('output_tokens', 0)}")
                lines.append(f"unleash_tokens_{safe_tier}_cost_usd {usage.get('cost_usd', 0)}")

            # Cache metrics
            for cache_type, cache in metrics.get("cache", {}).items():
                safe_type = cache_type.replace(".", "_")
                lines.append(f"# TYPE unleash_cache_{safe_type}_hits_total counter")
                lines.append(f"unleash_cache_{safe_type}_hits_total {cache.get('hits', 0)}")
                lines.append(f"unleash_cache_{safe_type}_misses_total {cache.get('misses', 0)}")
                lines.append(f"unleash_cache_{safe_type}_hit_rate {cache.get('hit_rate', 0)}")

            # Errors
            errors = metrics.get("errors", {})
            lines.append("# TYPE unleash_errors_total counter")
            lines.append(f"unleash_errors_total {errors.get('total_errors', 0)}")

            for error_type, count in errors.get("by_type", {}).items():
                safe_type = error_type.replace(".", "_").replace("-", "_")
                lines.append(f'unleash_errors_by_type{{type="{safe_type}"}} {count}')

            for adapter, count in errors.get("by_adapter", {}).items():
                safe_adapter = adapter.replace(".", "_").replace("-", "_")
                lines.append(f'unleash_errors_by_adapter{{adapter="{safe_adapter}"}} {count}')

            self._metrics_cache = "\n".join(lines)
            self._last_export = datetime.now(timezone.utc)
            return True
        except Exception as e:
            logger.error(f"Prometheus export failed: {e}")
            return False

    async def export_traces(self, traces: List[Span]) -> bool:
        # Prometheus doesn't support traces natively
        # We export trace counts as metrics
        trace_count = len(traces)
        try:
            self._metrics_cache += f"\n# TYPE unleash_traces_exported counter\nunleash_traces_exported {trace_count}"
            return True
        except Exception as e:
            logger.error(f"Prometheus trace count export failed: {e}")
            return False

    async def health_check(self) -> bool:
        return True

    def get_metrics_text(self) -> str:
        """Get Prometheus-formatted metrics text."""
        return self._metrics_cache


class OTLPExporter(Exporter):
    """
    OpenTelemetry Protocol (OTLP) exporter.

    Exports traces in OTLP format for OpenTelemetry collectors.
    """

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self._exporter = None
        self._initialized = False

    async def _ensure_initialized(self) -> bool:
        if self._initialized:
            return self._exporter is not None

        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry import trace

            if self.config.otlp_endpoint:
                self._exporter = OTLPSpanExporter(endpoint=self.config.otlp_endpoint)

                provider = TracerProvider()
                processor = BatchSpanProcessor(self._exporter)
                provider.add_span_processor(processor)
                trace.set_tracer_provider(provider)

                logger.info(f"OTLP exporter initialized: {self.config.otlp_endpoint}")
            else:
                logger.warning("OTLP endpoint not configured")
                self._exporter = None

            self._initialized = True
            return self._exporter is not None
        except ImportError:
            logger.warning("OpenTelemetry packages not installed")
            self._initialized = True
            return False
        except Exception as e:
            logger.error(f"OTLP initialization failed: {e}")
            self._initialized = True
            return False

    async def export_metrics(self, metrics: Dict[str, Any]) -> bool:
        # OTLP metrics export would require opentelemetry-metrics SDK
        # For now, we focus on trace export
        return True

    async def export_traces(self, traces: List[Span]) -> bool:
        if not await self._ensure_initialized():
            return False

        try:
            from opentelemetry import trace as otel_trace

            tracer = otel_trace.get_tracer("unleash")

            for span in traces:
                with tracer.start_as_current_span(
                    span.operation_name,
                    attributes=span.attributes,
                ) as otel_span:
                    otel_span.set_attribute("trace_id", span.context.trace_id)
                    otel_span.set_attribute("span_id", span.context.span_id)
                    otel_span.set_attribute("duration_ms", span.duration_ms or 0)

                    for event in span.events:
                        otel_span.add_event(
                            event["name"],
                            attributes=event.get("attributes", {}),
                        )

                    if span.status == "error":
                        otel_span.set_status(
                            otel_trace.StatusCode.ERROR,
                            span.attributes.get("status.message", ""),
                        )

            return True
        except Exception as e:
            logger.error(f"OTLP trace export failed: {e}")
            return False

    async def health_check(self) -> bool:
        return await self._ensure_initialized()


class JSONExporter(Exporter):
    """
    JSON exporter for custom dashboards.

    Exports metrics and traces as JSON for flexible consumption.
    """

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self._metrics_json: Dict[str, Any] = {}
        self._traces_json: List[Dict[str, Any]] = []

    async def export_metrics(self, metrics: Dict[str, Any]) -> bool:
        try:
            self._metrics_json = {
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "data": metrics,
            }
            return True
        except Exception as e:
            logger.error(f"JSON metrics export failed: {e}")
            return False

    async def export_traces(self, traces: List[Span]) -> bool:
        try:
            self._traces_json = [span.to_dict() for span in traces]
            return True
        except Exception as e:
            logger.error(f"JSON trace export failed: {e}")
            return False

    async def health_check(self) -> bool:
        return True

    def get_metrics_json(self) -> Dict[str, Any]:
        """Get JSON-formatted metrics."""
        return self._metrics_json

    def get_traces_json(self) -> List[Dict[str, Any]]:
        """Get JSON-formatted traces."""
        return self._traces_json


# =============================================================================
# PRODUCTION MONITORING LOOP
# =============================================================================

class ProductionMonitoringLoop:
    """
    Main production monitoring loop integrating all components.

    Provides:
    1. Metrics Collection - Request latency, token usage, cache hit rates, error rates
    2. Distributed Tracing - Span context propagation, cross-agent correlation
    3. Alerting - Configurable thresholds and notifications
    4. Dashboard Export - Prometheus, OTLP, JSON, Opik, Langfuse
    5. Cost Tracking - Token costs, API costs, budget alerts
    """

    def __init__(
        self,
        config: Optional[MonitoringConfig] = None,
        service_name: str = "unleash"
    ):
        self.config = config or MonitoringConfig()
        self.service_name = service_name

        # Initialize components
        self.metrics = MetricsCollector(self.config)
        self.tracer = DistributedTracer(self.config, service_name)
        self.alerts = AlertManager(self.config)
        self.costs = CostTracker(self.config)

        # Initialize exporters
        self._exporters: Dict[ExportFormat, Exporter] = {}
        self._init_exporters()

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._running = False

        # Integration hooks for Ralph
        self._on_iteration_start: Optional[Callable[[int], Awaitable[None]]] = None
        self._on_iteration_end: Optional[Callable[[int, Dict[str, Any]], Awaitable[None]]] = None

    def _init_exporters(self) -> None:
        """Initialize configured exporters."""
        # Always include Prometheus and JSON
        self._exporters[ExportFormat.PROMETHEUS] = PrometheusExporter(self.config)
        self._exporters[ExportFormat.JSON] = JSONExporter(self.config)

        # Optional exporters based on config
        if self.config.opik_api_key:
            self._exporters[ExportFormat.OPIK] = OpikExporter(self.config)

        if self.config.langfuse_public_key and self.config.langfuse_secret_key:
            self._exporters[ExportFormat.LANGFUSE] = LangfuseExporter(self.config)

        if self.config.otlp_endpoint:
            self._exporters[ExportFormat.OTLP] = OTLPExporter(self.config)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def record_request(
        self,
        adapter: str,
        method: str,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: Optional[str] = None,
        tier: ModelTier = ModelTier.TIER_3_OPUS,
        error: Optional[str] = None,
        cache_hit: bool = False
    ) -> None:
        """
        Record a complete request with all metrics.

        This is the primary entry point for recording monitoring data.
        """
        # Latency
        self.metrics.record_latency(adapter, method, latency_ms)

        # Tokens and cost
        if input_tokens or output_tokens:
            model = model or "default"
            self.metrics.record_tokens(model, input_tokens, output_tokens, tier)
            self.costs.record_cost(model, adapter, input_tokens, output_tokens, tier)

        # Cache
        if cache_hit:
            self.metrics.record_cache_hit(adapter)
        else:
            self.metrics.record_cache_miss(adapter)

        # Error
        if error:
            self.metrics.record_error(type(error).__name__ if isinstance(error, Exception) else "Error", adapter, str(error))

    @contextmanager
    def trace_request(
        self,
        operation: str,
        adapter: Optional[str] = None,
        method: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Context manager for tracing and timing a request."""
        start_time = time.perf_counter()
        span = self.tracer.start_span(operation, attributes={
            "adapter": adapter or "unknown",
            "method": method or "unknown",
            **(attributes or {}),
        })

        try:
            yield span
            status = "ok"
        except Exception as e:
            status = "error"
            span.set_status("error", str(e))
            raise
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000
            span.attributes["latency_ms"] = latency_ms
            self.tracer.end_span(span, status)

            if adapter and method:
                self.metrics.record_latency(adapter, method, latency_ms)

    @asynccontextmanager
    async def atrace_request(
        self,
        operation: str,
        adapter: Optional[str] = None,
        method: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Async context manager for tracing and timing a request."""
        start_time = time.perf_counter()
        span = self.tracer.start_span(operation, attributes={
            "adapter": adapter or "unknown",
            "method": method or "unknown",
            **(attributes or {}),
        })

        try:
            yield span
            status = "ok"
        except Exception as e:
            status = "error"
            span.set_status("error", str(e))
            raise
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000
            span.attributes["latency_ms"] = latency_ms
            self.tracer.end_span(span, status)

            if adapter and method:
                self.metrics.record_latency(adapter, method, latency_ms)

    async def evaluate_alerts(self) -> List[Alert]:
        """Evaluate current metrics against alert rules."""
        triggered = []

        # Latency alerts (use p95 as representative)
        latencies = self.metrics.get_latency_metrics()
        for key, metrics in latencies.items():
            alerts = await self.alerts.evaluate(
                "request_latency_ms",
                metrics.p95,
                {"endpoint": key},
            )
            triggered.extend(alerts)

        # Error rate alert
        error_rate = self.metrics.get_error_rate()
        alerts = await self.alerts.evaluate("error_rate", error_rate)
        triggered.extend(alerts)

        # Budget alerts
        budget = self.costs.get_budget_status()
        alerts = await self.alerts.evaluate(
            "daily_spend_usd",
            budget["daily_cost_usd"],
        )
        triggered.extend(alerts)

        return triggered

    async def export_all(self) -> Dict[ExportFormat, bool]:
        """Export metrics and traces to all configured exporters."""
        results = {}

        metrics_summary = self.metrics.get_summary()
        recent_traces = self.tracer.get_traces(limit=100)

        for format_type, exporter in self._exporters.items():
            try:
                metrics_ok = await exporter.export_metrics(metrics_summary)
                traces_ok = await exporter.export_traces(recent_traces)
                results[format_type] = metrics_ok and traces_ok
            except Exception as e:
                logger.error(f"Export to {format_type.value} failed: {e}")
                results[format_type] = False

        return results

    # -------------------------------------------------------------------------
    # Background Loop
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background monitoring loop."""
        if self._running:
            return

        self._running = True

        # Start export task
        self._background_tasks.append(
            asyncio.create_task(self._export_loop())
        )

        # Start alert evaluation task
        self._background_tasks.append(
            asyncio.create_task(self._alert_loop())
        )

        logger.info("Production monitoring loop started")

    async def stop(self) -> None:
        """Stop the background monitoring loop."""
        self._running = False

        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._background_tasks.clear()
        logger.info("Production monitoring loop stopped")

    async def _export_loop(self) -> None:
        """Background loop for periodic metric export."""
        while self._running:
            try:
                await asyncio.sleep(self.config.metrics_export_interval_s)
                await self.export_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Export loop error: {e}")

    async def _alert_loop(self) -> None:
        """Background loop for alert evaluation."""
        while self._running:
            try:
                await asyncio.sleep(self.config.alert_evaluation_interval_s)
                triggered = await self.evaluate_alerts()

                for alert in triggered:
                    logger.warning(
                        f"ALERT [{alert.severity.value.upper()}]: {alert.message}"
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Alert loop error: {e}")

    # -------------------------------------------------------------------------
    # Dashboard Data
    # -------------------------------------------------------------------------

    def get_prometheus_metrics(self) -> str:
        """Get Prometheus-formatted metrics."""
        exporter = self._exporters.get(ExportFormat.PROMETHEUS)
        if isinstance(exporter, PrometheusExporter):
            return exporter.get_metrics_text()
        return ""

    def get_json_metrics(self) -> Dict[str, Any]:
        """Get JSON-formatted metrics."""
        exporter = self._exporters.get(ExportFormat.JSON)
        if isinstance(exporter, JSONExporter):
            return exporter.get_metrics_json()
        return {}

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": self.service_name,
            "metrics": self.metrics.get_summary(),
            "costs": self.costs.get_summary(),
            "alerts": {
                "active": [a.to_dict() for a in self.alerts.get_active_alerts()],
                "recent": [a.to_dict() for a in self.alerts.get_alert_history(20)],
            },
            "traces": {
                "recent": [s.to_dict() for s in self.tracer.get_traces(limit=20)],
            },
            "exporters": {
                fmt.value: "enabled" for fmt in self._exporters.keys()
            },
        }

    async def health_check(self) -> Dict[str, Any]:
        """Run health checks on all components."""
        exporter_health = {}
        for fmt, exporter in self._exporters.items():
            try:
                exporter_health[fmt.value] = await exporter.health_check()
            except Exception as e:
                exporter_health[fmt.value] = False
                logger.error(f"Health check failed for {fmt.value}: {e}")

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "healthy" if all(exporter_health.values()) else "degraded",
            "components": {
                "metrics_collector": True,
                "tracer": True,
                "alert_manager": True,
                "cost_tracker": True,
                "exporters": exporter_health,
            },
            "loop_running": self._running,
        }


# =============================================================================
# RALPH INTEGRATION
# =============================================================================

class RalphMonitoringIntegration:
    """
    Integration layer between ProductionMonitoringLoop and Ralph production monitoring.

    Bridges the gap between the advanced monitoring loop and the existing
    Ralph production infrastructure.
    """

    def __init__(
        self,
        monitoring_loop: ProductionMonitoringLoop,
        ralph_monitor: Optional[Any] = None  # RalphProductionMonitor from production_monitoring.py
    ):
        self.loop = monitoring_loop
        self.ralph_monitor = ralph_monitor
        self._iteration_count = 0

    async def on_iteration_start(
        self,
        iteration: int,
        task: str,
        strategy: str
    ) -> str:
        """Called at the start of each Ralph Loop iteration."""
        self._iteration_count = iteration

        span = self.loop.tracer.start_span(
            f"ralph_iteration_{iteration}",
            attributes={
                "iteration": iteration,
                "task": task[:200],
                "strategy": strategy,
            },
        )

        # Propagate to Ralph monitor if available
        if self.ralph_monitor:
            return await self.ralph_monitor.on_iteration_start(iteration, task, strategy)

        return span.context.trace_id

    async def on_iteration_complete(
        self,
        trace_id: str,
        iteration: int,
        fitness: float,
        latency_ms: float,
        strategy: str,
        improved: bool,
        metadata: Dict[str, Any]
    ) -> List[Any]:
        """Called at the end of each Ralph Loop iteration."""
        # Record to monitoring loop
        self.loop.record_request(
            adapter="ralph",
            method=f"iteration_{strategy}",
            latency_ms=latency_ms,
            input_tokens=metadata.get("input_tokens", 0),
            output_tokens=metadata.get("output_tokens", 0),
            model=metadata.get("model", "claude-sonnet-4"),
        )

        # Evaluate alerts
        await self.loop.evaluate_alerts()

        # Propagate to Ralph monitor if available
        if self.ralph_monitor:
            return await self.ralph_monitor.on_iteration_complete(
                trace_id, iteration, fitness, latency_ms, strategy, improved, metadata
            )

        return []


# =============================================================================
# FACTORY AND UTILITIES
# =============================================================================

def create_production_loop(
    service_name: str = "unleash",
    opik_enabled: bool = True,
    langfuse_enabled: bool = True,
    **config_kwargs
) -> ProductionMonitoringLoop:
    """
    Factory function to create a configured production monitoring loop.

    Args:
        service_name: Name of the service for tracing
        opik_enabled: Enable Opik exporter if API key is configured
        langfuse_enabled: Enable Langfuse exporter if keys are configured
        **config_kwargs: Additional MonitoringConfig parameters

    Returns:
        Configured ProductionMonitoringLoop instance
    """
    config = MonitoringConfig(**config_kwargs)

    # Disable exporters if explicitly disabled
    if not opik_enabled:
        config.opik_api_key = None
    if not langfuse_enabled:
        config.langfuse_public_key = None
        config.langfuse_secret_key = None

    return ProductionMonitoringLoop(config, service_name)


async def quick_health_check() -> Dict[str, Any]:
    """Quick health check for the monitoring system."""
    loop = create_production_loop()
    return await loop.health_check()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "MonitoringConfig",
    "ExportFormat",
    "AlertSeverity",
    "ModelTier",

    # Metrics
    "MetricsCollector",
    "LatencyMetrics",
    "TokenUsage",
    "CacheMetrics",
    "ErrorMetrics",

    # Tracing
    "DistributedTracer",
    "SpanContext",
    "Span",

    # Alerting
    "AlertManager",
    "AlertRule",
    "Alert",

    # Cost Tracking
    "CostTracker",
    "CostEntry",

    # Exporters
    "Exporter",
    "OpikExporter",
    "LangfuseExporter",
    "PrometheusExporter",
    "OTLPExporter",
    "JSONExporter",

    # Main Loop
    "ProductionMonitoringLoop",

    # Ralph Integration
    "RalphMonitoringIntegration",

    # Factory
    "create_production_loop",
    "quick_health_check",
]
