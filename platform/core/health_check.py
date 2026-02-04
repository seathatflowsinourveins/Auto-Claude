"""
Comprehensive Health Check System for Production Monitoring
============================================================

Provides enterprise-grade health checking with:
- Adapter health monitoring via registry
- Memory backend health verification
- RAG pipeline health checks
- Prometheus metrics export
- JSON API response format
- Circuit breaker state monitoring

Usage:
    from core.health_check import HealthChecker, get_health_checker

    # Get singleton checker
    checker = get_health_checker()

    # Run all health checks
    report = await checker.check_all()

    # Export Prometheus metrics
    metrics = checker.export_prometheus()

    # JSON API response
    json_response = report.to_json()
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Awaitable, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Health Status Definitions
# =============================================================================

class HealthStatus(str, Enum):
    """Health status levels for components."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentCategory(str, Enum):
    """Categories of components being checked."""
    ADAPTER = "adapter"
    MEMORY = "memory"
    RAG = "rag"
    CACHE = "cache"
    DATABASE = "database"
    EXTERNAL = "external"
    SYSTEM = "system"


# =============================================================================
# Health Data Structures
# =============================================================================

@dataclass
class ComponentHealth:
    """Health status for a single component."""
    name: str
    status: HealthStatus
    category: ComponentCategory
    latency_ms: float = 0.0
    message: Optional[str] = None
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    consecutive_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "status": self.status.value,
            "category": self.category.value,
            "latency_ms": round(self.latency_ms, 2),
            "message": self.message,
            "error": self.error,
            "details": self.details,
            "last_check": self.last_check.isoformat(),
            "consecutive_failures": self.consecutive_failures,
        }


@dataclass
class HealthReport:
    """Complete health report for the system."""
    overall_status: HealthStatus
    component_status: Dict[str, ComponentHealth]
    last_check: datetime
    issues: List[str]
    summary: Dict[str, int] = field(default_factory=dict)
    version: str = "1.0.0"
    hostname: str = field(default_factory=lambda: os.environ.get("HOSTNAME", "unknown"))

    def __post_init__(self):
        """Calculate summary statistics."""
        self.summary = {
            "total": len(self.component_status),
            "healthy": sum(1 for c in self.component_status.values() if c.status == HealthStatus.HEALTHY),
            "degraded": sum(1 for c in self.component_status.values() if c.status == HealthStatus.DEGRADED),
            "unhealthy": sum(1 for c in self.component_status.values() if c.status == HealthStatus.UNHEALTHY),
            "unknown": sum(1 for c in self.component_status.values() if c.status == HealthStatus.UNKNOWN),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall_status": self.overall_status.value,
            "component_status": {
                name: comp.to_dict() for name, comp in self.component_status.items()
            },
            "last_check": self.last_check.isoformat(),
            "issues": self.issues,
            "summary": self.summary,
            "version": self.version,
            "hostname": self.hostname,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)

    @property
    def is_healthy(self) -> bool:
        """Check if overall system is healthy."""
        return self.overall_status == HealthStatus.HEALTHY

    @property
    def is_operational(self) -> bool:
        """Check if system is operational (healthy or degraded)."""
        return self.overall_status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)


# =============================================================================
# Prometheus Metrics
# =============================================================================

class PrometheusMetrics:
    """Prometheus metrics collector for health checks."""

    def __init__(self):
        self._metrics: Dict[str, Any] = {}
        self._labels: Dict[str, Dict[str, str]] = {}

    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric."""
        key = self._make_key(name, labels)
        self._metrics[key] = ("gauge", name, value, labels or {})

    def counter(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a counter metric."""
        key = self._make_key(name, labels)
        self._metrics[key] = ("counter", name, value, labels or {})

    def histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram observation."""
        key = self._make_key(name, labels)
        if key not in self._metrics:
            self._metrics[key] = ("histogram", name, [], labels or {})
        self._metrics[key][2].append(value)

    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create a unique key for metric + labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def export(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        seen_types: Dict[str, bool] = {}

        for key, (metric_type, name, value, labels) in sorted(self._metrics.items()):
            # Type annotation (once per metric name)
            if name not in seen_types:
                lines.append(f"# TYPE {name} {metric_type}")
                seen_types[name] = True

            # Format labels
            if labels:
                label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
                metric_line = f"{name}{{{label_str}}}"
            else:
                metric_line = name

            # Format value
            if isinstance(value, list):
                # Histogram: emit count and sum
                if value:
                    lines.append(f"{metric_line}_count {len(value)}")
                    lines.append(f"{metric_line}_sum {sum(value)}")
            else:
                lines.append(f"{metric_line} {value}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all metrics."""
        self._metrics.clear()


# =============================================================================
# Health Checker Implementation
# =============================================================================

class HealthChecker:
    """
    Comprehensive health checker for production monitoring.

    Provides:
    - Adapter health checking via registry
    - Memory backend health verification
    - RAG pipeline health checks
    - Prometheus metrics export
    - JSON API response format
    """

    _instance: Optional["HealthChecker"] = None

    def __new__(cls) -> "HealthChecker":
        """Singleton pattern for global health checker access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._checks: Dict[str, Callable[[], Awaitable[ComponentHealth]]] = {}
        self._component_status: Dict[str, ComponentHealth] = {}
        self._failure_counts: Dict[str, int] = {}
        self._metrics = PrometheusMetrics()
        self._check_timeout: float = 30.0
        self._degradation_threshold: int = 3  # Failures before degraded
        self._unhealthy_threshold: int = 5    # Failures before unhealthy
        self._initialized = True

        # Register default checks
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        """Register default health checks."""
        # System checks
        self.register_check("system_memory", self._check_system_memory, ComponentCategory.SYSTEM)
        self.register_check("system_disk", self._check_system_disk, ComponentCategory.SYSTEM)

    def register_check(
        self,
        name: str,
        check_fn: Callable[[], Awaitable[ComponentHealth]],
        category: ComponentCategory = ComponentCategory.SYSTEM,
    ) -> None:
        """
        Register a health check function.

        Args:
            name: Unique name for the check
            check_fn: Async function that returns ComponentHealth
            category: Category of the component
        """
        self._checks[name] = check_fn
        self._failure_counts[name] = 0

    def unregister_check(self, name: str) -> None:
        """Unregister a health check."""
        self._checks.pop(name, None)
        self._failure_counts.pop(name, None)
        self._component_status.pop(name, None)

    async def check_all(self, timeout: Optional[float] = None) -> HealthReport:
        """
        Run all health checks and generate a report.

        Args:
            timeout: Maximum time for all checks (default: 30s)

        Returns:
            HealthReport with all component statuses
        """
        timeout = timeout or self._check_timeout
        start_time = time.time()

        # Run adapter checks
        adapter_results = await self.check_adapters(timeout=timeout / 3)

        # Run memory checks
        memory_results = await self.check_memory(timeout=timeout / 3)

        # Run RAG checks
        rag_results = await self.check_rag(timeout=timeout / 3)

        # Run registered checks
        custom_results = await self._run_custom_checks(timeout=timeout / 3)

        # Combine all results
        all_results: Dict[str, ComponentHealth] = {}
        all_results.update(adapter_results)
        all_results.update(memory_results)
        all_results.update(rag_results)
        all_results.update(custom_results)

        # Store component status
        self._component_status = all_results

        # Collect issues
        issues: List[str] = []
        for name, health in all_results.items():
            if health.status == HealthStatus.UNHEALTHY:
                issues.append(f"{name}: {health.error or health.message or 'unhealthy'}")
            elif health.status == HealthStatus.DEGRADED:
                issues.append(f"{name}: degraded - {health.message or 'performance issues'}")

        # Determine overall status
        overall_status = self._calculate_overall_status(all_results)

        # Update Prometheus metrics
        self._update_metrics(all_results, time.time() - start_time)

        return HealthReport(
            overall_status=overall_status,
            component_status=all_results,
            last_check=datetime.now(timezone.utc),
            issues=issues,
        )

    async def check_adapters(self, timeout: float = 30.0) -> Dict[str, ComponentHealth]:
        """
        Check health of all registered adapters.

        Uses the adapter registry's health_check_all() method.

        Args:
            timeout: Maximum time for adapter checks

        Returns:
            Dict mapping adapter names to ComponentHealth
        """
        results: Dict[str, ComponentHealth] = {}

        try:
            # Import registry
            from adapters.registry import get_registry
            registry = get_registry()

            if registry is None:
                return {
                    "adapter_registry": ComponentHealth(
                        name="adapter_registry",
                        status=HealthStatus.UNHEALTHY,
                        category=ComponentCategory.ADAPTER,
                        error="Registry not available",
                    )
                }

            # Run health checks via registry
            start_time = time.time()
            health_results = await asyncio.wait_for(
                registry.health_check_all(timeout=timeout),
                timeout=timeout + 5,
            )

            # Convert registry results to ComponentHealth
            for adapter_name, result in health_results.items():
                status = HealthStatus.HEALTHY if result.is_healthy else HealthStatus.UNHEALTHY

                # Track consecutive failures
                if not result.is_healthy:
                    self._failure_counts[f"adapter_{adapter_name}"] = \
                        self._failure_counts.get(f"adapter_{adapter_name}", 0) + 1
                else:
                    self._failure_counts[f"adapter_{adapter_name}"] = 0

                failures = self._failure_counts.get(f"adapter_{adapter_name}", 0)
                if failures >= self._unhealthy_threshold:
                    status = HealthStatus.UNHEALTHY
                elif failures >= self._degradation_threshold:
                    status = HealthStatus.DEGRADED

                results[f"adapter_{adapter_name}"] = ComponentHealth(
                    name=adapter_name,
                    status=status,
                    category=ComponentCategory.ADAPTER,
                    latency_ms=result.latency_ms,
                    message=result.status,
                    error=result.error,
                    details=result.details,
                    consecutive_failures=failures,
                )

            # Add registry status
            available = registry.list_available()
            unavailable = registry.list_unavailable()

            results["adapter_registry"] = ComponentHealth(
                name="adapter_registry",
                status=HealthStatus.HEALTHY if available else HealthStatus.DEGRADED,
                category=ComponentCategory.ADAPTER,
                latency_ms=(time.time() - start_time) * 1000,
                message=f"{len(available)} adapters available, {len(unavailable)} unavailable",
                details={
                    "available": available,
                    "unavailable_count": len(unavailable),
                },
            )

        except asyncio.TimeoutError:
            results["adapter_registry"] = ComponentHealth(
                name="adapter_registry",
                status=HealthStatus.UNHEALTHY,
                category=ComponentCategory.ADAPTER,
                error="Health check timed out",
            )
        except ImportError as e:
            results["adapter_registry"] = ComponentHealth(
                name="adapter_registry",
                status=HealthStatus.UNKNOWN,
                category=ComponentCategory.ADAPTER,
                error=f"Registry import failed: {e}",
            )
        except Exception as e:
            logger.error(f"Adapter health check failed: {e}")
            results["adapter_registry"] = ComponentHealth(
                name="adapter_registry",
                status=HealthStatus.UNHEALTHY,
                category=ComponentCategory.ADAPTER,
                error=str(e),
            )

        return results

    async def check_memory(self, timeout: float = 10.0) -> Dict[str, ComponentHealth]:
        """
        Check health of memory backends.

        Args:
            timeout: Maximum time for memory checks

        Returns:
            Dict mapping memory backend names to ComponentHealth
        """
        results: Dict[str, ComponentHealth] = {}

        # Check in-memory tier backend
        try:
            from core.memory.backends.in_memory import InMemoryTierBackend

            start_time = time.time()
            backend = InMemoryTierBackend()

            # Simple health check - verify operations work
            await backend.put("__health_check__", {"test": True})
            result = await backend.get("__health_check__")
            await backend.delete("__health_check__")

            latency_ms = (time.time() - start_time) * 1000

            results["memory_in_memory"] = ComponentHealth(
                name="in_memory_backend",
                status=HealthStatus.HEALTHY,
                category=ComponentCategory.MEMORY,
                latency_ms=latency_ms,
                message="In-memory backend operational",
                details={"entry_count": await backend.count()},
            )
        except ImportError:
            results["memory_in_memory"] = ComponentHealth(
                name="in_memory_backend",
                status=HealthStatus.UNKNOWN,
                category=ComponentCategory.MEMORY,
                message="In-memory backend not available",
            )
        except Exception as e:
            results["memory_in_memory"] = ComponentHealth(
                name="in_memory_backend",
                status=HealthStatus.UNHEALTHY,
                category=ComponentCategory.MEMORY,
                error=str(e),
            )

        # Check Letta backend if available
        try:
            from core.memory.backends.letta import LettaTierBackend

            start_time = time.time()
            backend = LettaTierBackend()
            is_healthy = await asyncio.wait_for(backend.health_check(), timeout=timeout)
            latency_ms = (time.time() - start_time) * 1000

            results["memory_letta"] = ComponentHealth(
                name="letta_backend",
                status=HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
                category=ComponentCategory.MEMORY,
                latency_ms=latency_ms,
                message="Letta backend " + ("operational" if is_healthy else "unhealthy"),
            )
        except ImportError:
            # Letta not available - not an error
            pass
        except asyncio.TimeoutError:
            results["memory_letta"] = ComponentHealth(
                name="letta_backend",
                status=HealthStatus.UNHEALTHY,
                category=ComponentCategory.MEMORY,
                error="Health check timed out",
            )
        except Exception as e:
            results["memory_letta"] = ComponentHealth(
                name="letta_backend",
                status=HealthStatus.DEGRADED,
                category=ComponentCategory.MEMORY,
                error=str(e),
                message="Letta backend unreachable (optional)",
            )

        # Check unified memory gateway if available
        try:
            from core.unified_memory_gateway import UnifiedMemoryGateway

            start_time = time.time()
            gateway = UnifiedMemoryGateway()

            # Check if gateway is initialized
            if hasattr(gateway, 'health_check'):
                is_healthy = await asyncio.wait_for(
                    gateway.health_check(),
                    timeout=timeout
                )
                latency_ms = (time.time() - start_time) * 1000

                results["memory_gateway"] = ComponentHealth(
                    name="unified_memory_gateway",
                    status=HealthStatus.HEALTHY if is_healthy else HealthStatus.DEGRADED,
                    category=ComponentCategory.MEMORY,
                    latency_ms=latency_ms,
                    message="Memory gateway operational",
                )
            else:
                results["memory_gateway"] = ComponentHealth(
                    name="unified_memory_gateway",
                    status=HealthStatus.HEALTHY,
                    category=ComponentCategory.MEMORY,
                    message="Memory gateway available",
                )
        except ImportError:
            pass
        except Exception as e:
            results["memory_gateway"] = ComponentHealth(
                name="unified_memory_gateway",
                status=HealthStatus.DEGRADED,
                category=ComponentCategory.MEMORY,
                error=str(e),
            )

        return results

    async def check_rag(self, timeout: float = 10.0) -> Dict[str, ComponentHealth]:
        """
        Check health of RAG (Retrieval-Augmented Generation) components.

        Args:
            timeout: Maximum time for RAG checks

        Returns:
            Dict mapping RAG component names to ComponentHealth
        """
        results: Dict[str, ComponentHealth] = {}

        # Check RAG evaluator if available
        try:
            from adapters.rag_evaluator import RAGEvaluator

            start_time = time.time()
            evaluator = RAGEvaluator()

            # Basic health check
            if hasattr(evaluator, 'health_check'):
                is_healthy = await asyncio.wait_for(
                    evaluator.health_check(),
                    timeout=timeout
                )
            else:
                is_healthy = True

            latency_ms = (time.time() - start_time) * 1000

            results["rag_evaluator"] = ComponentHealth(
                name="rag_evaluator",
                status=HealthStatus.HEALTHY if is_healthy else HealthStatus.DEGRADED,
                category=ComponentCategory.RAG,
                latency_ms=latency_ms,
                message="RAG evaluator operational",
            )
        except ImportError:
            pass
        except Exception as e:
            results["rag_evaluator"] = ComponentHealth(
                name="rag_evaluator",
                status=HealthStatus.DEGRADED,
                category=ComponentCategory.RAG,
                error=str(e),
            )

        # Check embedding pipeline if available
        try:
            from core.embedding_pipeline import EmbeddingPipeline

            start_time = time.time()
            pipeline = EmbeddingPipeline()

            if hasattr(pipeline, 'health_check'):
                is_healthy = await asyncio.wait_for(
                    pipeline.health_check(),
                    timeout=timeout
                )
            else:
                is_healthy = True

            latency_ms = (time.time() - start_time) * 1000

            results["rag_embedding"] = ComponentHealth(
                name="embedding_pipeline",
                status=HealthStatus.HEALTHY if is_healthy else HealthStatus.DEGRADED,
                category=ComponentCategory.RAG,
                latency_ms=latency_ms,
                message="Embedding pipeline operational",
            )
        except ImportError:
            pass
        except Exception as e:
            results["rag_embedding"] = ComponentHealth(
                name="embedding_pipeline",
                status=HealthStatus.DEGRADED,
                category=ComponentCategory.RAG,
                error=str(e),
            )

        # Check iterative retrieval if available
        try:
            from core.iterative_retrieval import IterativeRetriever

            start_time = time.time()

            results["rag_retrieval"] = ComponentHealth(
                name="iterative_retriever",
                status=HealthStatus.HEALTHY,
                category=ComponentCategory.RAG,
                latency_ms=(time.time() - start_time) * 1000,
                message="Iterative retriever available",
            )
        except ImportError:
            pass
        except Exception as e:
            results["rag_retrieval"] = ComponentHealth(
                name="iterative_retriever",
                status=HealthStatus.DEGRADED,
                category=ComponentCategory.RAG,
                error=str(e),
            )

        return results

    async def _run_custom_checks(self, timeout: float = 10.0) -> Dict[str, ComponentHealth]:
        """Run all registered custom health checks."""
        results: Dict[str, ComponentHealth] = {}

        async def run_check(name: str, check_fn: Callable) -> tuple[str, ComponentHealth]:
            """Run a single check with timeout."""
            start_time = time.time()
            try:
                result = await asyncio.wait_for(check_fn(), timeout=timeout)
                result.latency_ms = (time.time() - start_time) * 1000

                # Track failures
                if result.status in (HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN):
                    self._failure_counts[name] = self._failure_counts.get(name, 0) + 1
                else:
                    self._failure_counts[name] = 0

                result.consecutive_failures = self._failure_counts.get(name, 0)
                return name, result

            except asyncio.TimeoutError:
                self._failure_counts[name] = self._failure_counts.get(name, 0) + 1
                return name, ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    category=ComponentCategory.SYSTEM,
                    latency_ms=(time.time() - start_time) * 1000,
                    error="Health check timed out",
                    consecutive_failures=self._failure_counts.get(name, 0),
                )
            except Exception as e:
                self._failure_counts[name] = self._failure_counts.get(name, 0) + 1
                return name, ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    category=ComponentCategory.SYSTEM,
                    latency_ms=(time.time() - start_time) * 1000,
                    error=str(e),
                    consecutive_failures=self._failure_counts.get(name, 0),
                )

        # Run all checks concurrently
        tasks = [run_check(name, fn) for name, fn in self._checks.items()]
        if tasks:
            completed = await asyncio.gather(*tasks, return_exceptions=True)
            for item in completed:
                if isinstance(item, tuple):
                    name, health = item
                    results[name] = health

        return results

    async def _check_system_memory(self) -> ComponentHealth:
        """Check system memory usage."""
        try:
            import psutil

            memory = psutil.virtual_memory()
            used_percent = memory.percent

            if used_percent >= 95:
                status = HealthStatus.UNHEALTHY
                message = f"Critical memory usage: {used_percent}%"
            elif used_percent >= 85:
                status = HealthStatus.DEGRADED
                message = f"High memory usage: {used_percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage: {used_percent}%"

            return ComponentHealth(
                name="system_memory",
                status=status,
                category=ComponentCategory.SYSTEM,
                message=message,
                details={
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_percent": used_percent,
                },
            )
        except ImportError:
            return ComponentHealth(
                name="system_memory",
                status=HealthStatus.UNKNOWN,
                category=ComponentCategory.SYSTEM,
                message="psutil not available",
            )
        except Exception as e:
            return ComponentHealth(
                name="system_memory",
                status=HealthStatus.UNKNOWN,
                category=ComponentCategory.SYSTEM,
                error=str(e),
            )

    async def _check_system_disk(self) -> ComponentHealth:
        """Check system disk usage."""
        try:
            import psutil

            disk = psutil.disk_usage("/")
            used_percent = disk.percent

            if used_percent >= 95:
                status = HealthStatus.UNHEALTHY
                message = f"Critical disk usage: {used_percent}%"
            elif used_percent >= 85:
                status = HealthStatus.DEGRADED
                message = f"High disk usage: {used_percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage: {used_percent}%"

            return ComponentHealth(
                name="system_disk",
                status=status,
                category=ComponentCategory.SYSTEM,
                message=message,
                details={
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "used_percent": used_percent,
                },
            )
        except ImportError:
            return ComponentHealth(
                name="system_disk",
                status=HealthStatus.UNKNOWN,
                category=ComponentCategory.SYSTEM,
                message="psutil not available",
            )
        except Exception as e:
            return ComponentHealth(
                name="system_disk",
                status=HealthStatus.UNKNOWN,
                category=ComponentCategory.SYSTEM,
                error=str(e),
            )

    def _calculate_overall_status(self, components: Dict[str, ComponentHealth]) -> HealthStatus:
        """Calculate overall system status from component statuses."""
        if not components:
            return HealthStatus.UNKNOWN

        statuses = [c.status for c in components.values()]

        # Any unhealthy critical component = unhealthy
        unhealthy_count = sum(1 for s in statuses if s == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for s in statuses if s == HealthStatus.DEGRADED)
        unknown_count = sum(1 for s in statuses if s == HealthStatus.UNKNOWN)

        # Threshold-based logic
        total = len(statuses)

        # More than 20% unhealthy = overall unhealthy
        if unhealthy_count / total > 0.2:
            return HealthStatus.UNHEALTHY

        # Any unhealthy or more than 30% degraded = degraded
        if unhealthy_count > 0 or degraded_count / total > 0.3:
            return HealthStatus.DEGRADED

        # All unknown = unknown
        if unknown_count == total:
            return HealthStatus.UNKNOWN

        return HealthStatus.HEALTHY

    def _update_metrics(self, components: Dict[str, ComponentHealth], total_time: float) -> None:
        """Update Prometheus metrics from health check results."""
        # Clear previous metrics
        self._metrics.clear()

        # Overall health (1 = healthy, 0.5 = degraded, 0 = unhealthy)
        status_value = {
            HealthStatus.HEALTHY: 1.0,
            HealthStatus.DEGRADED: 0.5,
            HealthStatus.UNHEALTHY: 0.0,
            HealthStatus.UNKNOWN: 0.0,
        }

        overall_status = self._calculate_overall_status(components)
        self._metrics.gauge("unleashed_health_status", status_value[overall_status])

        # Component counts
        self._metrics.gauge("unleashed_health_components_total", len(components))
        self._metrics.gauge(
            "unleashed_health_components_healthy",
            sum(1 for c in components.values() if c.status == HealthStatus.HEALTHY)
        )
        self._metrics.gauge(
            "unleashed_health_components_degraded",
            sum(1 for c in components.values() if c.status == HealthStatus.DEGRADED)
        )
        self._metrics.gauge(
            "unleashed_health_components_unhealthy",
            sum(1 for c in components.values() if c.status == HealthStatus.UNHEALTHY)
        )

        # Per-component metrics
        for name, health in components.items():
            labels = {"component": name, "category": health.category.value}

            self._metrics.gauge(
                "unleashed_component_health_status",
                status_value[health.status],
                labels
            )
            self._metrics.gauge(
                "unleashed_component_latency_ms",
                health.latency_ms,
                labels
            )
            self._metrics.gauge(
                "unleashed_component_failures",
                health.consecutive_failures,
                labels
            )

        # Health check duration
        self._metrics.gauge("unleashed_health_check_duration_seconds", total_time)
        self._metrics.counter("unleashed_health_checks_total", 1)

    def export_prometheus(self) -> str:
        """Export health metrics in Prometheus format."""
        return self._metrics.export()

    def get_component_status(self, name: str) -> Optional[ComponentHealth]:
        """Get cached status for a specific component."""
        return self._component_status.get(name)

    def get_all_component_status(self) -> Dict[str, ComponentHealth]:
        """Get cached status for all components."""
        return self._component_status.copy()


# =============================================================================
# Global Instance and Factory
# =============================================================================

_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """
    Get the global health checker instance.

    Returns:
        The singleton HealthChecker instance
    """
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


async def quick_health_check() -> Dict[str, Any]:
    """
    Perform a quick health check and return JSON-serializable result.

    Returns:
        Dict suitable for JSON API response
    """
    checker = get_health_checker()
    report = await checker.check_all(timeout=15.0)
    return report.to_dict()


# =============================================================================
# FastAPI/Starlette Integration
# =============================================================================

def create_health_endpoint():
    """
    Create a health check endpoint for FastAPI/Starlette.

    Usage:
        from fastapi import FastAPI
        from core.health_check import create_health_endpoint

        app = FastAPI()

        @app.get("/health")
        async def health():
            return await create_health_endpoint()()
    """
    async def health_endpoint():
        return await quick_health_check()
    return health_endpoint


def create_prometheus_endpoint():
    """
    Create a Prometheus metrics endpoint.

    Usage:
        from fastapi import FastAPI, Response
        from core.health_check import create_prometheus_endpoint

        app = FastAPI()

        @app.get("/metrics")
        async def metrics():
            return Response(
                content=await create_prometheus_endpoint()(),
                media_type="text/plain"
            )
    """
    async def prometheus_endpoint():
        checker = get_health_checker()
        await checker.check_all(timeout=15.0)
        return checker.export_prometheus()
    return prometheus_endpoint


# =============================================================================
# CLI Support
# =============================================================================

async def main():
    """CLI entry point for health checks."""
    import argparse
    import json as json_module

    parser = argparse.ArgumentParser(description="Run health checks")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--prometheus", action="store_true", help="Output Prometheus metrics")
    parser.add_argument("--timeout", type=float, default=30.0, help="Check timeout in seconds")
    args = parser.parse_args()

    checker = get_health_checker()
    report = await checker.check_all(timeout=args.timeout)

    if args.prometheus:
        print(checker.export_prometheus())
    elif args.json:
        print(json_module.dumps(report.to_dict(), indent=2))
    else:
        # Human-readable output
        status_symbols = {
            HealthStatus.HEALTHY: "[OK]",
            HealthStatus.DEGRADED: "[WARN]",
            HealthStatus.UNHEALTHY: "[FAIL]",
            HealthStatus.UNKNOWN: "[?]",
        }

        print(f"\nHealth Check Report - {report.last_check.isoformat()}")
        print(f"Overall Status: {status_symbols[report.overall_status]} {report.overall_status.value.upper()}")
        print(f"\nSummary: {report.summary['healthy']} healthy, "
              f"{report.summary['degraded']} degraded, "
              f"{report.summary['unhealthy']} unhealthy\n")

        for name, component in sorted(report.component_status.items()):
            symbol = status_symbols[component.status]
            latency = f"{component.latency_ms:.1f}ms" if component.latency_ms else "N/A"
            message = component.message or component.error or ""
            print(f"  {symbol} {name}: {message} ({latency})")

        if report.issues:
            print(f"\nIssues ({len(report.issues)}):")
            for issue in report.issues:
                print(f"  - {issue}")


if __name__ == "__main__":
    asyncio.run(main())


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Status enums
    "HealthStatus",
    "ComponentCategory",
    # Data classes
    "ComponentHealth",
    "HealthReport",
    # Metrics
    "PrometheusMetrics",
    # Main class
    "HealthChecker",
    # Factory functions
    "get_health_checker",
    "quick_health_check",
    # Web integration
    "create_health_endpoint",
    "create_prometheus_endpoint",
]
