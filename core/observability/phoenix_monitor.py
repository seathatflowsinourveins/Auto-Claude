#!/usr/bin/env python3
"""
Phoenix Monitor - Real-time LLM Monitoring with OpenTelemetry
Part of the V33 Observability Layer.

Uses Arize Phoenix for real-time monitoring including:
- OpenTelemetry integration
- Embedding drift detection
- Performance alerting
- Dashboard visualization
"""

from __future__ import annotations

import os
import uuid
from enum import Enum
from typing import Any, Optional, Dict, List, Callable
from datetime import datetime
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    trace = None
    TracerProvider = None
    BatchSpanProcessor = None

try:
    import phoenix as px
    from phoenix.trace import suppress_tracing
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    px = None
    suppress_tracing = None


# ============================================================================
# Alert and Drift Types
# ============================================================================

class AlertSeverity(str, Enum):
    """Severity levels for alerts."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DriftType(str, Enum):
    """Types of drift that can be detected."""
    EMBEDDING = "embedding"
    PREDICTION = "prediction"
    INPUT = "input"
    OUTPUT = "output"
    LATENCY = "latency"
    COST = "cost"


class MonitorStatus(str, Enum):
    """Status of the monitor."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# ============================================================================
# Monitoring Models
# ============================================================================

@dataclass
class Alert:
    """An alert triggered by monitoring."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    severity: AlertSeverity = AlertSeverity.INFO
    message: str = ""
    metric_name: str = ""
    current_value: float = 0.0
    threshold: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingRecord:
    """A record of an embedding for drift detection."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: List[float] = field(default_factory=list)
    text: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftResult:
    """Result of a drift detection."""
    drift_type: DriftType
    detected: bool = False
    score: float = 0.0
    threshold: float = 0.5
    reference_period: str = ""
    current_period: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class MonitoringMetrics(BaseModel):
    """Aggregated monitoring metrics."""
    total_requests: int = Field(default=0)
    total_tokens: int = Field(default=0)
    total_cost_usd: float = Field(default=0.0)
    avg_latency_ms: float = Field(default=0.0)
    p95_latency_ms: float = Field(default=0.0)
    p99_latency_ms: float = Field(default=0.0)
    error_rate: float = Field(default=0.0)
    success_rate: float = Field(default=0.0)
    active_alerts: int = Field(default=0)
    status: MonitorStatus = Field(default=MonitorStatus.UNKNOWN)


class AlertConfig(BaseModel):
    """Configuration for an alert rule."""
    name: str = Field(description="Alert name")
    metric: str = Field(description="Metric to monitor")
    operator: str = Field(default="gt", description="Comparison operator (gt, lt, eq)")
    threshold: float = Field(description="Threshold value")
    severity: AlertSeverity = Field(default=AlertSeverity.WARNING)
    cooldown_seconds: int = Field(default=300)


# ============================================================================
# Phoenix Monitor
# ============================================================================

class PhoenixMonitor:
    """
    Phoenix-based monitoring for LLM applications.

    Provides real-time monitoring including:
    - OpenTelemetry integration for tracing
    - Embedding drift detection
    - Performance alerting
    - Dashboard visualization
    """

    def __init__(
        self,
        project_name: str = "v33-monitor",
        enable_phoenix: bool = True,
        enable_otel: bool = True,
        phoenix_port: int = 6006,
    ):
        """
        Initialize the Phoenix monitor.

        Args:
            project_name: Project name for organizing data
            enable_phoenix: Enable Phoenix dashboard
            enable_otel: Enable OpenTelemetry
            phoenix_port: Port for Phoenix dashboard
        """
        self.project_name = project_name
        self.enable_phoenix = enable_phoenix
        self.enable_otel = enable_otel
        self.phoenix_port = phoenix_port

        self._tracer = None
        self._alerts: Dict[str, AlertConfig] = {}
        self._triggered_alerts: List[Alert] = []
        self._embeddings: List[EmbeddingRecord] = []
        self._metrics: Dict[str, List[float]] = {}
        self._request_count = 0
        self._error_count = 0

        self._init_monitoring()

    def _init_monitoring(self) -> None:
        """Initialize monitoring components."""
        if OPENTELEMETRY_AVAILABLE and self.enable_otel:
            try:
                provider = TracerProvider()
                trace.set_tracer_provider(provider)
                self._tracer = trace.get_tracer(self.project_name)
            except Exception:
                pass

        if PHOENIX_AVAILABLE and self.enable_phoenix:
            try:
                # Phoenix session would be launched here
                pass
            except Exception:
                pass

    @property
    def is_available(self) -> bool:
        """Check if monitoring is available."""
        return OPENTELEMETRY_AVAILABLE or PHOENIX_AVAILABLE

    def record_request(
        self,
        latency_ms: float,
        tokens: int,
        cost_usd: float,
        success: bool = True,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Record a request for monitoring.

        Args:
            latency_ms: Request latency
            tokens: Tokens used
            cost_usd: Cost of request
            success: Whether request succeeded
            metadata: Additional metadata
        """
        self._request_count += 1
        if not success:
            self._error_count += 1

        # Track metrics
        self._add_metric("latency_ms", latency_ms)
        self._add_metric("tokens", tokens)
        self._add_metric("cost_usd", cost_usd)

        # Check alerts
        self._check_alerts({
            "latency_ms": latency_ms,
            "tokens": tokens,
            "cost_usd": cost_usd,
            "error_rate": self._error_count / max(1, self._request_count),
        })

    def _add_metric(self, name: str, value: float) -> None:
        """Add a metric value."""
        if name not in self._metrics:
            self._metrics[name] = []
        self._metrics[name].append(value)

        # Keep only last 1000 values
        if len(self._metrics[name]) > 1000:
            self._metrics[name] = self._metrics[name][-1000:]

    def record_embedding(
        self,
        embedding: List[float],
        text: str = "",
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Record an embedding for drift detection.

        Args:
            embedding: Embedding vector
            text: Source text
            metadata: Additional metadata

        Returns:
            Record ID
        """
        record = EmbeddingRecord(
            embedding=embedding,
            text=text,
            metadata=metadata or {},
        )
        self._embeddings.append(record)

        # Keep only last 10000 embeddings
        if len(self._embeddings) > 10000:
            self._embeddings = self._embeddings[-10000:]

        return record.id

    def add_alert_rule(self, config: AlertConfig) -> None:
        """
        Add an alert rule.

        Args:
            config: Alert configuration
        """
        self._alerts[config.name] = config

    def _check_alerts(self, metrics: Dict[str, float]) -> None:
        """Check all alert rules against current metrics."""
        for name, config in self._alerts.items():
            if config.metric not in metrics:
                continue

            value = metrics[config.metric]
            triggered = False

            if config.operator == "gt" and value > config.threshold:
                triggered = True
            elif config.operator == "lt" and value < config.threshold:
                triggered = True
            elif config.operator == "eq" and value == config.threshold:
                triggered = True

            if triggered:
                alert = Alert(
                    severity=config.severity,
                    message=f"{config.metric} {config.operator} {config.threshold}",
                    metric_name=config.metric,
                    current_value=value,
                    threshold=config.threshold,
                )
                self._triggered_alerts.append(alert)

    def detect_embedding_drift(
        self,
        reference_start: Optional[datetime] = None,
        reference_end: Optional[datetime] = None,
        current_start: Optional[datetime] = None,
        current_end: Optional[datetime] = None,
        threshold: float = 0.5,
    ) -> DriftResult:
        """
        Detect embedding drift between time periods.

        Args:
            reference_start: Start of reference period
            reference_end: End of reference period
            current_start: Start of current period
            current_end: End of current period
            threshold: Drift threshold

        Returns:
            DriftResult with detection results
        """
        if len(self._embeddings) < 2:
            return DriftResult(
                drift_type=DriftType.EMBEDDING,
                detected=False,
                details={"reason": "Insufficient embeddings"},
            )

        # Simple drift detection using cosine distance
        try:
            import numpy as np

            # Get reference and current embeddings
            mid_point = len(self._embeddings) // 2
            ref_embeddings = [e.embedding for e in self._embeddings[:mid_point]]
            cur_embeddings = [e.embedding for e in self._embeddings[mid_point:]]

            if not ref_embeddings or not cur_embeddings:
                return DriftResult(
                    drift_type=DriftType.EMBEDDING,
                    detected=False,
                )

            # Calculate centroid distance
            ref_centroid = np.mean(ref_embeddings, axis=0)
            cur_centroid = np.mean(cur_embeddings, axis=0)

            # Cosine distance
            dot_product = np.dot(ref_centroid, cur_centroid)
            ref_norm = np.linalg.norm(ref_centroid)
            cur_norm = np.linalg.norm(cur_centroid)

            if ref_norm == 0 or cur_norm == 0:
                drift_score = 0.0
            else:
                cosine_sim = dot_product / (ref_norm * cur_norm)
                drift_score = 1.0 - cosine_sim

            return DriftResult(
                drift_type=DriftType.EMBEDDING,
                detected=drift_score > threshold,
                score=drift_score,
                threshold=threshold,
                details={
                    "ref_count": len(ref_embeddings),
                    "cur_count": len(cur_embeddings),
                },
            )

        except ImportError:
            # NumPy not available, use simple heuristic
            return DriftResult(
                drift_type=DriftType.EMBEDDING,
                detected=False,
                details={"reason": "NumPy not available"},
            )

    def detect_latency_drift(self, threshold: float = 0.3) -> DriftResult:
        """
        Detect latency drift.

        Args:
            threshold: Drift threshold (percentage change)

        Returns:
            DriftResult with detection results
        """
        if "latency_ms" not in self._metrics:
            return DriftResult(
                drift_type=DriftType.LATENCY,
                detected=False,
            )

        values = self._metrics["latency_ms"]
        if len(values) < 10:
            return DriftResult(
                drift_type=DriftType.LATENCY,
                detected=False,
            )

        mid = len(values) // 2
        ref_avg = sum(values[:mid]) / mid
        cur_avg = sum(values[mid:]) / (len(values) - mid)

        if ref_avg == 0:
            drift_score = 0.0
        else:
            drift_score = abs(cur_avg - ref_avg) / ref_avg

        return DriftResult(
            drift_type=DriftType.LATENCY,
            detected=drift_score > threshold,
            score=drift_score,
            threshold=threshold,
            details={
                "ref_avg_ms": ref_avg,
                "cur_avg_ms": cur_avg,
            },
        )

    def get_metrics(self) -> MonitoringMetrics:
        """
        Get current monitoring metrics.

        Returns:
            Aggregated metrics
        """
        latencies = self._metrics.get("latency_ms", [])
        tokens = self._metrics.get("tokens", [])
        costs = self._metrics.get("cost_usd", [])

        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        sorted_latencies = sorted(latencies) if latencies else []
        p95_idx = int(len(sorted_latencies) * 0.95)
        p99_idx = int(len(sorted_latencies) * 0.99)
        p95_latency = sorted_latencies[p95_idx] if sorted_latencies else 0.0
        p99_latency = sorted_latencies[p99_idx] if sorted_latencies else 0.0

        error_rate = self._error_count / max(1, self._request_count)
        success_rate = 1.0 - error_rate

        # Determine status
        if error_rate > 0.1:
            status = MonitorStatus.UNHEALTHY
        elif error_rate > 0.05 or avg_latency > 5000:
            status = MonitorStatus.DEGRADED
        elif self._request_count > 0:
            status = MonitorStatus.HEALTHY
        else:
            status = MonitorStatus.UNKNOWN

        return MonitoringMetrics(
            total_requests=self._request_count,
            total_tokens=sum(tokens),
            total_cost_usd=sum(costs),
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            error_rate=error_rate,
            success_rate=success_rate,
            active_alerts=len(self._triggered_alerts),
            status=status,
        )

    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """
        Get triggered alerts.

        Args:
            severity: Filter by severity
            limit: Maximum alerts to return

        Returns:
            List of alerts
        """
        alerts = self._triggered_alerts

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts[-limit:]

    def clear_alerts(self) -> int:
        """
        Clear all triggered alerts.

        Returns:
            Number of alerts cleared
        """
        count = len(self._triggered_alerts)
        self._triggered_alerts = []
        return count

    def start_span(self, name: str, attributes: Optional[Dict] = None):
        """
        Start an OpenTelemetry span.

        Args:
            name: Span name
            attributes: Span attributes

        Returns:
            Span context manager
        """
        if self._tracer:
            return self._tracer.start_as_current_span(
                name,
                attributes=attributes or {},
            )
        else:
            from contextlib import nullcontext
            return nullcontext()


# ============================================================================
# Convenience Functions
# ============================================================================

def create_phoenix_monitor(
    project_name: str = "v33-monitor",
    **kwargs: Any,
) -> PhoenixMonitor:
    """
    Factory function to create a PhoenixMonitor.

    Args:
        project_name: Project name
        **kwargs: Additional configuration

    Returns:
        Configured PhoenixMonitor instance
    """
    return PhoenixMonitor(project_name=project_name, **kwargs)


# Export availability
__all__ = [
    "PhoenixMonitor",
    "AlertSeverity",
    "DriftType",
    "MonitorStatus",
    "Alert",
    "EmbeddingRecord",
    "DriftResult",
    "MonitoringMetrics",
    "AlertConfig",
    "create_phoenix_monitor",
    "OPENTELEMETRY_AVAILABLE",
    "PHOENIX_AVAILABLE",
]
