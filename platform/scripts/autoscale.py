#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "prometheus-client>=0.19.0",
# ]
# ///
"""
Platform Autoscaling - Kubernetes HPA Custom Metrics

Provides custom metrics for Kubernetes Horizontal Pod Autoscaler (HPA).
Exposes metrics in Prometheus format for the metrics-server adapter.

Metrics for HPA:
- uap_request_rate: Current request rate (per second)
- uap_queue_depth: Pending tasks in queue
- uap_cpu_utilization: CPU utilization percentage
- uap_memory_utilization: Memory utilization percentage
- uap_concurrent_requests: Current concurrent requests

Usage:
    from autoscale import AutoscaleMetrics, get_autoscale_metrics

    metrics = get_autoscale_metrics()
    metrics.record_request()
    metrics.set_queue_depth(42)

    # Get HPA decision metrics
    decisions = metrics.get_hpa_metrics()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional
import threading
import json

from prometheus_client import Counter, Gauge, Summary, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.core import CollectorRegistry


@dataclass
class AutoscaleConfig:
    """Configuration for autoscaling metrics."""
    # HPA thresholds
    target_request_rate: float = 100.0  # requests per second
    target_queue_depth: int = 50  # pending tasks
    target_cpu_percent: float = 70.0
    target_memory_percent: float = 80.0
    target_concurrent: int = 100

    # Scale bounds
    min_replicas: int = 1
    max_replicas: int = 10

    # Cooldown periods
    scale_up_cooldown_seconds: float = 60.0
    scale_down_cooldown_seconds: float = 300.0


@dataclass
class ScaleRecommendation:
    """Autoscaling recommendation."""
    action: str  # "scale_up", "scale_down", "no_change"
    current_replicas: int
    recommended_replicas: int
    reason: str
    metrics: Dict[str, float]
    timestamp: float = field(default_factory=time.time)


class AutoscaleMetrics:
    """
    Collects and exposes metrics for Kubernetes HPA.

    Provides both Prometheus-format metrics and JSON API for
    custom metrics server integration.
    """

    def __init__(
        self,
        config: Optional[AutoscaleConfig] = None,
        registry: Optional[CollectorRegistry] = None
    ):
        self.config = config or AutoscaleConfig()
        self.registry = registry or CollectorRegistry()

        # Request metrics
        self._request_counter = Counter(
            'uap_requests_total',
            'Total number of requests',
            registry=self.registry
        )

        self._request_rate = Gauge(
            'uap_request_rate',
            'Current request rate (per second)',
            registry=self.registry
        )

        self._concurrent_requests = Gauge(
            'uap_concurrent_requests',
            'Current concurrent requests',
            registry=self.registry
        )

        # Queue metrics
        self._queue_depth = Gauge(
            'uap_queue_depth',
            'Number of pending tasks in queue',
            registry=self.registry
        )

        # Resource metrics
        self._cpu_utilization = Gauge(
            'uap_cpu_utilization_percent',
            'CPU utilization percentage',
            registry=self.registry
        )

        self._memory_utilization = Gauge(
            'uap_memory_utilization_percent',
            'Memory utilization percentage',
            registry=self.registry
        )

        # Latency metrics
        self._request_latency = Summary(
            'uap_request_latency_seconds',
            'Request latency in seconds',
            registry=self.registry
        )

        # HPA decision metrics
        self._scale_recommendation = Gauge(
            'uap_scale_recommendation',
            'Recommended replica count',
            registry=self.registry
        )

        # Internal state
        self._request_times: List[float] = []
        self._request_window = 60.0  # 1 minute window for rate calculation
        self._current_concurrent = 0
        self._lock = threading.Lock()

        # Scale tracking
        self._last_scale_up_time = 0.0
        self._last_scale_down_time = 0.0
        self._current_replicas = 1

    def record_request(self) -> None:
        """Record a new request."""
        with self._lock:
            now = time.time()
            self._request_times.append(now)
            self._request_counter.inc()

            # Clean old entries
            cutoff = now - self._request_window
            self._request_times = [t for t in self._request_times if t > cutoff]

            # Update rate
            rate = len(self._request_times) / self._request_window
            self._request_rate.set(rate)

    def enter_request(self) -> None:
        """Mark entering a request (for concurrent tracking)."""
        with self._lock:
            self._current_concurrent += 1
            self._concurrent_requests.set(self._current_concurrent)

    def exit_request(self, latency_seconds: float) -> None:
        """Mark exiting a request."""
        with self._lock:
            self._current_concurrent = max(0, self._current_concurrent - 1)
            self._concurrent_requests.set(self._current_concurrent)
            self._request_latency.observe(latency_seconds)

    def set_queue_depth(self, depth: int) -> None:
        """Set the current queue depth."""
        self._queue_depth.set(depth)

    def set_cpu_utilization(self, percent: float) -> None:
        """Set CPU utilization percentage."""
        self._cpu_utilization.set(percent)

    def set_memory_utilization(self, percent: float) -> None:
        """Set memory utilization percentage."""
        self._memory_utilization.set(percent)

    def set_replicas(self, count: int) -> None:
        """Set current replica count."""
        self._current_replicas = count

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metric values."""
        with self._lock:
            cutoff = time.time() - self._request_window
            recent_requests = [t for t in self._request_times if t > cutoff]
            rate = len(recent_requests) / self._request_window

        return {
            "request_rate": rate,
            "concurrent_requests": self._current_concurrent,
            "queue_depth": float(self._queue_depth._value._value),
            "cpu_percent": float(self._cpu_utilization._value._value),
            "memory_percent": float(self._memory_utilization._value._value),
        }

    def get_hpa_metrics(self) -> Dict[str, Any]:
        """
        Get metrics formatted for Kubernetes custom metrics API.

        Returns format compatible with metrics-server adapter.
        """
        metrics = self.get_current_metrics()

        return {
            "items": [
                {
                    "metricName": "request_rate",
                    "metricValue": f"{metrics['request_rate']:.2f}",
                    "target": f"{self.config.target_request_rate}",
                    "currentUtilization": int(metrics['request_rate'] / self.config.target_request_rate * 100)
                },
                {
                    "metricName": "queue_depth",
                    "metricValue": f"{metrics['queue_depth']:.0f}",
                    "target": f"{self.config.target_queue_depth}",
                    "currentUtilization": int(metrics['queue_depth'] / self.config.target_queue_depth * 100)
                },
                {
                    "metricName": "concurrent_requests",
                    "metricValue": f"{metrics['concurrent_requests']:.0f}",
                    "target": f"{self.config.target_concurrent}",
                    "currentUtilization": int(metrics['concurrent_requests'] / self.config.target_concurrent * 100)
                }
            ]
        }

    def calculate_scale_recommendation(self) -> ScaleRecommendation:
        """
        Calculate scaling recommendation based on current metrics.

        Uses multiple metrics with weighted scoring to determine
        optimal replica count.
        """
        metrics = self.get_current_metrics()
        now = time.time()

        # Calculate utilization ratios
        request_util = metrics["request_rate"] / self.config.target_request_rate
        queue_util = metrics["queue_depth"] / self.config.target_queue_depth
        concurrent_util = metrics["concurrent_requests"] / self.config.target_concurrent

        # Take maximum utilization
        max_util = max(request_util, queue_util, concurrent_util)

        # Calculate recommended replicas
        recommended = int(self._current_replicas * max_util)
        recommended = max(self.config.min_replicas, min(self.config.max_replicas, recommended))

        # Determine action
        if recommended > self._current_replicas:
            if now - self._last_scale_up_time < self.config.scale_up_cooldown_seconds:
                action = "cooldown"
                reason = f"Scale up cooldown ({self.config.scale_up_cooldown_seconds}s)"
            else:
                action = "scale_up"
                reason = f"High utilization: {max_util:.1%}"
        elif recommended < self._current_replicas:
            if now - self._last_scale_down_time < self.config.scale_down_cooldown_seconds:
                action = "cooldown"
                reason = f"Scale down cooldown ({self.config.scale_down_cooldown_seconds}s)"
            else:
                action = "scale_down"
                reason = f"Low utilization: {max_util:.1%}"
        else:
            action = "no_change"
            reason = f"Utilization at target: {max_util:.1%}"

        recommendation = ScaleRecommendation(
            action=action,
            current_replicas=self._current_replicas,
            recommended_replicas=recommended,
            reason=reason,
            metrics=metrics
        )

        # Update prometheus metric
        self._scale_recommendation.set(recommended)

        return recommendation

    def apply_scale_decision(self, new_replicas: int) -> None:
        """Record that a scale decision was applied."""
        now = time.time()
        if new_replicas > self._current_replicas:
            self._last_scale_up_time = now
        elif new_replicas < self._current_replicas:
            self._last_scale_down_time = now

        self._current_replicas = new_replicas

    def get_prometheus_metrics(self) -> bytes:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry)

    def get_content_type(self) -> str:
        """Get Prometheus content type."""
        return CONTENT_TYPE_LATEST


# Global instance
_global_metrics: Optional[AutoscaleMetrics] = None


def get_autoscale_metrics() -> AutoscaleMetrics:
    """Get or create global autoscale metrics."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = AutoscaleMetrics()
    return _global_metrics


def serve_autoscale_metrics(port: int = 9091) -> None:
    """
    Start HTTP server for autoscale metrics.

    Endpoints:
    - /metrics: Prometheus format
    - /hpa: HPA custom metrics JSON
    - /recommend: Scaling recommendation
    """
    metrics = get_autoscale_metrics()

    class MetricsHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            if self.path == "/metrics":
                output = metrics.get_prometheus_metrics()
                self.send_response(200)
                self.send_header("Content-Type", metrics.get_content_type())
                self.end_headers()
                self.wfile.write(output)

            elif self.path == "/hpa":
                hpa_data = metrics.get_hpa_metrics()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(hpa_data, indent=2).encode())

            elif self.path == "/recommend":
                recommendation = metrics.calculate_scale_recommendation()
                data = {
                    "action": recommendation.action,
                    "current_replicas": recommendation.current_replicas,
                    "recommended_replicas": recommendation.recommended_replicas,
                    "reason": recommendation.reason,
                    "metrics": recommendation.metrics,
                    "timestamp": recommendation.timestamp
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data, indent=2).encode())

            else:
                self.send_response(404)
                self.end_headers()

    server = HTTPServer(("0.0.0.0", port), MetricsHandler)
    print(f"[OK] Autoscale metrics server on port {port}")
    print(f"     GET /metrics    - Prometheus format")
    print(f"     GET /hpa        - HPA custom metrics")
    print(f"     GET /recommend  - Scale recommendation")
    server.serve_forever()


def main():
    """Demo autoscaling metrics."""
    import random

    print("=" * 60)
    print("AUTOSCALING METRICS DEMO")
    print("=" * 60)
    print()

    metrics = get_autoscale_metrics()
    metrics.set_replicas(3)

    # Simulate load
    print("[>>] Simulating request load...")
    for i in range(50):
        metrics.record_request()
        if i % 5 == 0:
            metrics.enter_request()
        if i % 7 == 0:
            metrics.exit_request(random.uniform(0.01, 0.5))

    # Set queue depth
    metrics.set_queue_depth(35)
    metrics.set_cpu_utilization(65.5)
    metrics.set_memory_utilization(72.3)

    # Get current metrics
    print("\n[>>] Current Metrics:")
    current = metrics.get_current_metrics()
    for name, value in current.items():
        print(f"  {name}: {value:.2f}")

    # Get HPA metrics
    print("\n[>>] HPA Metrics:")
    hpa = metrics.get_hpa_metrics()
    for item in hpa["items"]:
        print(f"  {item['metricName']}: {item['metricValue']} / {item['target']} ({item['currentUtilization']}%)")

    # Get scale recommendation
    print("\n[>>] Scale Recommendation:")
    rec = metrics.calculate_scale_recommendation()
    print(f"  Action: {rec.action}")
    print(f"  Current: {rec.current_replicas} replicas")
    print(f"  Recommended: {rec.recommended_replicas} replicas")
    print(f"  Reason: {rec.reason}")

    print("\n[OK] Autoscaling demo complete")
    print("\n[>>] To start metrics server: python autoscale.py serve")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 9091
        serve_autoscale_metrics(port)
    else:
        main()
