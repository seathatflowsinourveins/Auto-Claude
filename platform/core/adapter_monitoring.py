"""
Unleashed Platform - Adapter Monitoring Integration

Pre-configured monitoring, health checks, and metrics for SDK adapters.
"""

import asyncio
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import logging

from .monitoring import (
    MetricRegistry,
    Tracer,
    HealthChecker,
    AlertManager,
    AlertRule,
    AlertSeverity,
    HealthStatus,
    HealthCheckResult,
    Profiler,
    MonitoringDashboard,
    create_sdk_metrics,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Adapter Status Types
# ============================================================================

class AdapterState(Enum):
    """Operational state of an adapter."""
    UNKNOWN = "unknown"
    INITIALIZING = "initializing"
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


@dataclass
class AdapterHealth:
    """Health information for a single adapter."""
    name: str
    state: AdapterState
    available: bool
    initialized: bool
    last_check: datetime
    last_success: Optional[datetime] = None
    last_error: Optional[str] = None
    error_count: int = 0
    success_count: int = 0
    avg_latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Adapter Health Checks
# ============================================================================

class AdapterHealthChecker:
    """Health checker specifically for SDK adapters."""

    def __init__(self):
        self._adapters: Dict[str, Dict[str, Any]] = {}
        self._health_data: Dict[str, AdapterHealth] = {}
        self._checker = HealthChecker()
        self._registry = MetricRegistry()
        self._metrics = create_sdk_metrics()

    def register_adapter(
        self,
        name: str,
        adapter_instance: Any,
        health_check_fn: Optional[Callable[[], Awaitable[bool]]] = None,
        timeout: float = 10.0
    ) -> None:
        """Register an adapter for health monitoring."""
        self._adapters[name] = {
            "instance": adapter_instance,
            "health_check_fn": health_check_fn,
            "timeout": timeout
        }

        # Initialize health data
        self._health_data[name] = AdapterHealth(
            name=name,
            state=AdapterState.UNKNOWN,
            available=False,
            initialized=False,
            last_check=datetime.now(timezone.utc)
        )

        # Register health check with main checker
        async def check_adapter() -> HealthCheckResult:
            return await self._check_adapter_health(name)

        self._checker.register(f"adapter:{name}", check_adapter)

    async def _check_adapter_health(self, name: str) -> HealthCheckResult:
        """Perform health check for a specific adapter."""
        if name not in self._adapters:
            return HealthCheckResult(
                name=f"adapter:{name}",
                status=HealthStatus.UNKNOWN,
                message=f"Adapter {name} not registered"
            )

        adapter_info = self._adapters[name]
        adapter = adapter_info["instance"]
        health_check_fn = adapter_info["health_check_fn"]
        timeout = adapter_info["timeout"]

        start_time = datetime.now(timezone.utc)
        health_data = self._health_data[name]

        try:
            # Check if adapter has built-in health check
            if health_check_fn:
                is_healthy = await asyncio.wait_for(
                    health_check_fn(),
                    timeout=timeout
                )
            elif hasattr(adapter, "health_check"):
                is_healthy = await asyncio.wait_for(
                    adapter.health_check(),
                    timeout=timeout
                )
            elif hasattr(adapter, "ping"):
                is_healthy = await asyncio.wait_for(
                    adapter.ping(),
                    timeout=timeout
                )
            else:
                # Assume healthy if no check available
                is_healthy = True

            duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            if is_healthy:
                health_data.state = AdapterState.READY
                health_data.available = True
                health_data.last_success = datetime.now(timezone.utc)
                health_data.success_count += 1
                health_data.avg_latency_ms = (
                    (health_data.avg_latency_ms * (health_data.success_count - 1) + duration)
                    / health_data.success_count
                )

                # Update metrics
                self._metrics["adapter_calls"].inc(
                    adapter=name, method="health_check", status="success"
                )

                return HealthCheckResult(
                    name=f"adapter:{name}",
                    status=HealthStatus.HEALTHY,
                    message="Adapter is healthy",
                    duration_ms=duration,
                    details={
                        "success_count": health_data.success_count,
                        "avg_latency_ms": health_data.avg_latency_ms
                    }
                )
            else:
                health_data.state = AdapterState.DEGRADED
                health_data.error_count += 1

                self._metrics["adapter_errors"].inc(
                    adapter=name, error_type="health_check_failed"
                )

                return HealthCheckResult(
                    name=f"adapter:{name}",
                    status=HealthStatus.DEGRADED,
                    message="Adapter health check returned false",
                    duration_ms=duration
                )

        except asyncio.TimeoutError:
            health_data.state = AdapterState.UNAVAILABLE
            health_data.error_count += 1
            health_data.last_error = "Health check timed out"

            self._metrics["adapter_errors"].inc(
                adapter=name, error_type="timeout"
            )

            return HealthCheckResult(
                name=f"adapter:{name}",
                status=HealthStatus.UNHEALTHY,
                message="Health check timed out",
                duration_ms=timeout * 1000
            )

        except Exception as e:
            health_data.state = AdapterState.ERROR
            health_data.error_count += 1
            health_data.last_error = str(e)

            self._metrics["adapter_errors"].inc(
                adapter=name, error_type=type(e).__name__
            )

            return HealthCheckResult(
                name=f"adapter:{name}",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                duration_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )

        finally:
            health_data.last_check = datetime.now(timezone.utc)

    async def check_all(self) -> Dict[str, AdapterHealth]:
        """Check health of all registered adapters."""
        tasks = [
            self._check_adapter_health(name)
            for name in self._adapters
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        return self._health_data.copy()

    def get_adapter_health(self, name: str) -> Optional[AdapterHealth]:
        """Get health data for a specific adapter."""
        return self._health_data.get(name)

    def get_all_health(self) -> Dict[str, AdapterHealth]:
        """Get health data for all adapters."""
        return self._health_data.copy()

    async def get_overall_status(self) -> AdapterState:
        """Get overall adapter health status."""
        if not self._health_data:
            return AdapterState.UNKNOWN

        states = [h.state for h in self._health_data.values()]

        if all(s == AdapterState.READY for s in states):
            return AdapterState.READY
        if any(s == AdapterState.ERROR for s in states):
            return AdapterState.ERROR
        if any(s == AdapterState.UNAVAILABLE for s in states):
            return AdapterState.UNAVAILABLE
        if any(s == AdapterState.DEGRADED for s in states):
            return AdapterState.DEGRADED

        return AdapterState.UNKNOWN


# ============================================================================
# Adapter Metrics Collector
# ============================================================================

class AdapterMetricsCollector:
    """Collects and aggregates metrics from SDK adapters."""

    def __init__(self):
        self._registry = MetricRegistry()
        self._tracer = Tracer()
        self._profiler = Profiler()

        # Pre-create standard adapter metrics
        self._call_counter = self._registry.counter(
            "unleashed_adapter_calls_total",
            "Total number of adapter method calls",
            ["adapter", "method", "status"]
        )
        self._latency_histogram = self._registry.histogram(
            "unleashed_adapter_latency_seconds",
            "Adapter method latency in seconds",
            ["adapter", "method"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
        )
        self._error_counter = self._registry.counter(
            "unleashed_adapter_errors_total",
            "Total number of adapter errors",
            ["adapter", "error_type"]
        )
        self._active_calls = self._registry.gauge(
            "unleashed_adapter_active_calls",
            "Number of active adapter calls",
            ["adapter"]
        )

    def record_call_start(self, adapter: str, method: str) -> None:
        """Record the start of an adapter call."""
        self._active_calls.inc(adapter=adapter)

    def record_call_end(
        self,
        adapter: str,
        method: str,
        duration: float,
        success: bool,
        error_type: Optional[str] = None
    ) -> None:
        """Record the end of an adapter call."""
        self._active_calls.dec(adapter=adapter)

        status = "success" if success else "error"
        self._call_counter.inc(adapter=adapter, method=method, status=status)
        self._latency_histogram.observe(duration, adapter=adapter, method=method)

        if not success and error_type:
            self._error_counter.inc(adapter=adapter, error_type=error_type)

        # Record in profiler
        self._profiler.record(f"{adapter}.{method}", duration * 1000)

    def get_adapter_stats(self, adapter: str) -> Dict[str, Any]:
        """Get statistics for a specific adapter."""
        profile = self._profiler.get_profile(f"{adapter}.*")

        return {
            "adapter": adapter,
            "call_count": self._call_counter.get(adapter=adapter, method="", status="success"),
            "error_count": self._call_counter.get(adapter=adapter, method="", status="error"),
            "active_calls": self._active_calls.get(adapter=adapter),
            "latency": self._latency_histogram.get_stats(adapter=adapter, method=""),
            "profile": profile.__dict__ if profile else None
        }

    def get_all_stats(self) -> Dict[str, Any]:
        """Get all adapter statistics."""
        return {
            "metrics": self._registry.collect_all(),
            "profiles": {
                name: p.__dict__
                for name, p in self._profiler.get_all_profiles().items()
            }
        }


# ============================================================================
# Adapter Alert Rules
# ============================================================================

def create_adapter_alert_rules() -> List[AlertRule]:
    """Create standard alert rules for adapter monitoring."""
    return [
        AlertRule(
            name="adapter_high_error_rate",
            metric_name="unleashed_adapter_errors_total",
            condition=lambda v: v > 10,  # More than 10 errors
            severity=AlertSeverity.ERROR,
            message_template="Adapter {adapter} has high error rate: {value} errors",
            cooldown=timedelta(minutes=5)
        ),
        AlertRule(
            name="adapter_high_latency",
            metric_name="unleashed_adapter_latency_seconds",
            condition=lambda v: v > 5.0,  # More than 5 seconds
            severity=AlertSeverity.WARNING,
            message_template="Adapter {adapter} has high latency: {value}s",
            cooldown=timedelta(minutes=2)
        ),
        AlertRule(
            name="adapter_unavailable",
            metric_name="unleashed_adapter_health_status",
            condition=lambda v: v == 0,  # 0 = unavailable
            severity=AlertSeverity.CRITICAL,
            message_template="Adapter {adapter} is unavailable",
            cooldown=timedelta(minutes=1)
        ),
        AlertRule(
            name="adapter_circuit_open",
            metric_name="unleashed_circuit_breaker_state",
            condition=lambda v: v == 1,  # 1 = open
            severity=AlertSeverity.WARNING,
            message_template="Circuit breaker for {adapter} is open",
            cooldown=timedelta(minutes=5)
        ),
    ]


# ============================================================================
# Integrated Adapter Monitor
# ============================================================================

class AdapterMonitor:
    """Integrated monitoring system for all SDK adapters."""

    def __init__(self):
        self._health_checker = AdapterHealthChecker()
        self._metrics_collector = AdapterMetricsCollector()
        self._alert_manager = AlertManager()
        self._tracer = Tracer()
        self._dashboard = MonitoringDashboard(
            health_checker=self._health_checker._checker,
            alert_manager=self._alert_manager,
            profiler=self._metrics_collector._profiler
        )

        # Register standard alert rules
        for rule in create_adapter_alert_rules():
            self._alert_manager.add_rule(rule)

    def register_adapter(
        self,
        name: str,
        adapter_instance: Any,
        health_check_fn: Optional[Callable[[], Awaitable[bool]]] = None
    ) -> None:
        """Register an adapter for monitoring."""
        self._health_checker.register_adapter(name, adapter_instance, health_check_fn)

    async def track_call(
        self,
        adapter: str,
        method: str,
        func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> Any:
        """Track an adapter method call with full observability."""
        self._metrics_collector.record_call_start(adapter, method)

        async with self._tracer.atrace(
            f"{adapter}.{method}",
            attributes={"adapter": adapter, "method": method}
        ) as span:
            start_time = datetime.now(timezone.utc)
            try:
                result = await func(*args, **kwargs)

                duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                self._metrics_collector.record_call_end(
                    adapter, method, duration, success=True
                )

                # Check for latency alerts
                await self._alert_manager.evaluate(
                    "unleashed_adapter_latency_seconds",
                    duration,
                    {"adapter": adapter, "method": method}
                )

                return result

            except Exception as e:
                duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                self._metrics_collector.record_call_end(
                    adapter, method, duration,
                    success=False,
                    error_type=type(e).__name__
                )

                # Check for error alerts
                await self._alert_manager.evaluate(
                    "unleashed_adapter_errors_total",
                    1,
                    {"adapter": adapter, "error_type": type(e).__name__}
                )

                span.attributes["error"] = str(e)
                raise

    async def check_health(self) -> Dict[str, AdapterHealth]:
        """Check health of all registered adapters."""
        return await self._health_checker.check_all()

    def get_metrics(self) -> Dict[str, Any]:
        """Get all adapter metrics."""
        return self._metrics_collector.get_all_stats()

    def get_active_alerts(self) -> List[Any]:
        """Get all active alerts."""
        return self._alert_manager.get_active_alerts()

    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        health = await self._health_checker.check_all()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "adapters": {
                name: {
                    "state": h.state.value,
                    "available": h.available,
                    "error_count": h.error_count,
                    "success_count": h.success_count,
                    "avg_latency_ms": h.avg_latency_ms,
                    "last_check": h.last_check.isoformat() if h.last_check else None,
                    "last_error": h.last_error
                }
                for name, h in health.items()
            },
            "metrics": self._metrics_collector.get_all_stats(),
            "alerts": [
                {
                    "name": a.name,
                    "severity": a.severity.value,
                    "message": a.message,
                    "timestamp": a.timestamp.isoformat()
                }
                for a in self._alert_manager.get_active_alerts()
            ],
            "traces": self._tracer.get_traces(limit=50)
        }


# ============================================================================
# Global Monitor Instance
# ============================================================================

_adapter_monitor: Optional[AdapterMonitor] = None


def get_adapter_monitor() -> AdapterMonitor:
    """Get the global adapter monitor instance."""
    global _adapter_monitor
    if _adapter_monitor is None:
        _adapter_monitor = AdapterMonitor()
    return _adapter_monitor


def reset_adapter_monitor() -> None:
    """Reset the global adapter monitor (for testing)."""
    global _adapter_monitor
    _adapter_monitor = None


# ============================================================================
# Decorator for Automatic Tracking
# ============================================================================

def monitored_adapter_call(adapter_name: str, method_name: Optional[str] = None):
    """Decorator to automatically track adapter method calls."""
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        async def wrapper(*args, **kwargs):
            monitor = get_adapter_monitor()
            method = method_name or func.__name__
            return await monitor.track_call(adapter_name, method, func, *args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Types
    "AdapterState",
    "AdapterHealth",

    # Health Checker
    "AdapterHealthChecker",

    # Metrics Collector
    "AdapterMetricsCollector",

    # Alert Rules
    "create_adapter_alert_rules",

    # Integrated Monitor
    "AdapterMonitor",
    "get_adapter_monitor",
    "reset_adapter_monitor",

    # Decorator
    "monitored_adapter_call",
]
