#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "prometheus-client>=0.19.0",
# ]
# ///
"""
Platform Metrics - Prometheus Metrics Exporter

Exposes platform health and performance metrics in Prometheus format.

Metrics:
- uap_component_health: Health status of each component (0=unhealthy, 1=healthy)
- uap_component_latency_ms: Latency of health checks in milliseconds
- uap_circuit_breaker_state: Circuit breaker state (0=CLOSED, 1=HALF_OPEN, 2=OPEN)
- uap_circuit_breaker_failures: Number of failures recorded
- uap_health_check_total: Total number of health checks performed
- uap_health_check_errors_total: Total number of health check errors

Usage:
    # Start metrics server standalone
    uv run metrics.py --port 9090

    # Or import and use programmatically
    from metrics import MetricsExporter
    exporter = MetricsExporter()
    exporter.record_health_check("qdrant", healthy=True, latency_ms=45.2)
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from enum import IntEnum
from http.server import HTTPServer
from typing import Dict, Optional

from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.core import CollectorRegistry


class CircuitState(IntEnum):
    """Circuit breaker states as integers for Prometheus."""
    CLOSED = 0
    HALF_OPEN = 1
    OPEN = 2


@dataclass
class ComponentMetrics:
    """Metrics for a single component."""
    healthy: bool = False
    latency_ms: float = 0.0
    circuit_state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_check_time: float = 0.0


class MetricsExporter:
    """
    Prometheus metrics exporter for the Ultimate Autonomous Platform.

    Collects and exposes metrics about:
    - Component health status
    - Health check latencies
    - Circuit breaker states
    - Error counts
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()

        # Health metrics
        self.component_health = Gauge(
            'uap_component_health',
            'Health status of platform components (1=healthy, 0=unhealthy)',
            ['component'],
            registry=self.registry
        )

        self.component_latency = Gauge(
            'uap_component_latency_ms',
            'Health check latency in milliseconds',
            ['component'],
            registry=self.registry
        )

        # Circuit breaker metrics
        self.circuit_state = Gauge(
            'uap_circuit_breaker_state',
            'Circuit breaker state (0=CLOSED, 1=HALF_OPEN, 2=OPEN)',
            ['component'],
            registry=self.registry
        )

        self.circuit_failures = Gauge(
            'uap_circuit_breaker_failures',
            'Number of recorded failures in circuit breaker',
            ['component'],
            registry=self.registry
        )

        # Counter metrics
        self.health_checks_total = Counter(
            'uap_health_check_total',
            'Total number of health checks performed',
            ['component'],
            registry=self.registry
        )

        self.health_errors_total = Counter(
            'uap_health_check_errors_total',
            'Total number of health check errors',
            ['component'],
            registry=self.registry
        )

        # Histogram for latency distribution
        self.latency_histogram = Histogram(
            'uap_health_check_latency_seconds',
            'Health check latency distribution',
            ['component'],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry
        )

        # Internal state
        self._component_metrics: Dict[str, ComponentMetrics] = {}

    def record_health_check(
        self,
        component: str,
        healthy: bool,
        latency_ms: float,
        circuit_state: str = "CLOSED",
        failure_count: int = 0
    ) -> None:
        """
        Record a health check result.

        Args:
            component: Component name (e.g., "qdrant", "neo4j")
            healthy: Whether the component is healthy
            latency_ms: Health check latency in milliseconds
            circuit_state: Circuit breaker state string
            failure_count: Number of failures in circuit breaker
        """
        # Update gauges
        self.component_health.labels(component=component).set(1 if healthy else 0)
        self.component_latency.labels(component=component).set(latency_ms)

        # Map circuit state string to enum
        state_map = {
            "CLOSED": CircuitState.CLOSED,
            "HALF_OPEN": CircuitState.HALF_OPEN,
            "OPEN": CircuitState.OPEN
        }
        state = state_map.get(circuit_state, CircuitState.CLOSED)
        self.circuit_state.labels(component=component).set(state.value)
        self.circuit_failures.labels(component=component).set(failure_count)

        # Increment counters
        self.health_checks_total.labels(component=component).inc()
        if not healthy:
            self.health_errors_total.labels(component=component).inc()

        # Record latency histogram (convert ms to seconds)
        self.latency_histogram.labels(component=component).observe(latency_ms / 1000.0)

        # Store internal state
        self._component_metrics[component] = ComponentMetrics(
            healthy=healthy,
            latency_ms=latency_ms,
            circuit_state=state,
            failure_count=failure_count,
            last_check_time=time.time()
        )

    def get_metrics(self) -> bytes:
        """Generate Prometheus metrics output."""
        return generate_latest(self.registry)

    def get_content_type(self) -> str:
        """Get Prometheus content type for HTTP response."""
        return CONTENT_TYPE_LATEST


class MetricsHandler:
    """HTTP request handler for Prometheus metrics endpoint."""

    def __init__(self, exporter: MetricsExporter):
        self.exporter = exporter

    def handle_request(self, environ, start_response):
        """WSGI-style request handler."""
        if environ['PATH_INFO'] == '/metrics':
            output = self.exporter.get_metrics()
            status = '200 OK'
            headers = [
                ('Content-Type', self.exporter.get_content_type()),
                ('Content-Length', str(len(output)))
            ]
            start_response(status, headers)
            return [output]
        else:
            # Health check endpoint
            status = '200 OK'
            body = b'OK'
            headers = [
                ('Content-Type', 'text/plain'),
                ('Content-Length', str(len(body)))
            ]
            start_response(status, headers)
            return [body]


def create_metrics_app(exporter: MetricsExporter):
    """Create WSGI application for metrics server."""
    handler = MetricsHandler(exporter)
    return handler.handle_request


# Singleton instance for global access
_global_exporter: Optional[MetricsExporter] = None


def get_metrics_exporter() -> MetricsExporter:
    """Get or create global metrics exporter instance."""
    global _global_exporter
    if _global_exporter is None:
        _global_exporter = MetricsExporter()
    return _global_exporter


def main():
    """Run standalone metrics server."""
    parser = argparse.ArgumentParser(description="Platform Metrics Server")
    parser.add_argument("--port", type=int, default=9090, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    exporter = get_metrics_exporter()

    # Add some sample data for testing
    exporter.record_health_check("qdrant", True, 45.2, "CLOSED", 0)
    exporter.record_health_check("neo4j", True, 128.5, "CLOSED", 0)
    exporter.record_health_check("letta", True, 67.3, "CLOSED", 0)
    exporter.record_health_check("auto_claude", False, 0, "CLOSED", 0)

    print(f"[>>] Starting metrics server on {args.host}:{args.port}")
    print(f"[OK] Metrics available at http://{args.host}:{args.port}/metrics")

    # Use prometheus_client's built-in HTTP server
    from prometheus_client import start_http_server
    start_http_server(args.port, addr=args.host, registry=exporter.registry)

    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[OK] Metrics server stopped")


if __name__ == "__main__":
    main()
