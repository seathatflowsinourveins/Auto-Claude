"""
Unit Tests for Health Check System
==================================

Tests for the comprehensive health check system.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


# =============================================================================
# Test HealthStatus and ComponentCategory
# =============================================================================

def test_health_status_values():
    """Test HealthStatus enum values."""
    from core.health_check import HealthStatus

    assert HealthStatus.HEALTHY.value == "healthy"
    assert HealthStatus.DEGRADED.value == "degraded"
    assert HealthStatus.UNHEALTHY.value == "unhealthy"
    assert HealthStatus.UNKNOWN.value == "unknown"


def test_component_category_values():
    """Test ComponentCategory enum values."""
    from core.health_check import ComponentCategory

    assert ComponentCategory.ADAPTER.value == "adapter"
    assert ComponentCategory.MEMORY.value == "memory"
    assert ComponentCategory.RAG.value == "rag"
    assert ComponentCategory.CACHE.value == "cache"
    assert ComponentCategory.DATABASE.value == "database"
    assert ComponentCategory.EXTERNAL.value == "external"
    assert ComponentCategory.SYSTEM.value == "system"


# =============================================================================
# Test ComponentHealth
# =============================================================================

def test_component_health_creation():
    """Test ComponentHealth dataclass creation."""
    from core.health_check import ComponentHealth, HealthStatus, ComponentCategory

    health = ComponentHealth(
        name="test_component",
        status=HealthStatus.HEALTHY,
        category=ComponentCategory.ADAPTER,
        latency_ms=15.5,
        message="All good",
    )

    assert health.name == "test_component"
    assert health.status == HealthStatus.HEALTHY
    assert health.category == ComponentCategory.ADAPTER
    assert health.latency_ms == 15.5
    assert health.message == "All good"
    assert health.error is None
    assert health.consecutive_failures == 0


def test_component_health_to_dict():
    """Test ComponentHealth JSON serialization."""
    from core.health_check import ComponentHealth, HealthStatus, ComponentCategory

    health = ComponentHealth(
        name="test_component",
        status=HealthStatus.DEGRADED,
        category=ComponentCategory.MEMORY,
        latency_ms=25.123456,
        message="Slow response",
        details={"queue_depth": 100},
    )

    result = health.to_dict()

    assert result["name"] == "test_component"
    assert result["status"] == "degraded"
    assert result["category"] == "memory"
    assert result["latency_ms"] == 25.12  # Rounded to 2 decimals
    assert result["message"] == "Slow response"
    assert result["details"]["queue_depth"] == 100
    assert "last_check" in result


# =============================================================================
# Test HealthReport
# =============================================================================

def test_health_report_creation():
    """Test HealthReport dataclass creation and summary calculation."""
    from core.health_check import (
        HealthReport, ComponentHealth, HealthStatus, ComponentCategory
    )

    components = {
        "adapter_exa": ComponentHealth(
            name="exa",
            status=HealthStatus.HEALTHY,
            category=ComponentCategory.ADAPTER,
        ),
        "adapter_tavily": ComponentHealth(
            name="tavily",
            status=HealthStatus.DEGRADED,
            category=ComponentCategory.ADAPTER,
        ),
        "memory_letta": ComponentHealth(
            name="letta",
            status=HealthStatus.UNHEALTHY,
            category=ComponentCategory.MEMORY,
            error="Connection refused",
        ),
    }

    report = HealthReport(
        overall_status=HealthStatus.DEGRADED,
        component_status=components,
        last_check=datetime.now(timezone.utc),
        issues=["memory_letta: Connection refused"],
    )

    assert report.overall_status == HealthStatus.DEGRADED
    assert len(report.component_status) == 3
    assert report.summary["total"] == 3
    assert report.summary["healthy"] == 1
    assert report.summary["degraded"] == 1
    assert report.summary["unhealthy"] == 1
    assert report.summary["unknown"] == 0


def test_health_report_to_json():
    """Test HealthReport JSON conversion."""
    from core.health_check import (
        HealthReport, ComponentHealth, HealthStatus, ComponentCategory
    )
    import json

    components = {
        "test": ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY,
            category=ComponentCategory.SYSTEM,
        ),
    }

    report = HealthReport(
        overall_status=HealthStatus.HEALTHY,
        component_status=components,
        last_check=datetime.now(timezone.utc),
        issues=[],
    )

    json_str = report.to_json()
    parsed = json.loads(json_str)

    assert parsed["overall_status"] == "healthy"
    assert "test" in parsed["component_status"]
    assert parsed["summary"]["total"] == 1


def test_health_report_is_healthy():
    """Test HealthReport is_healthy property."""
    from core.health_check import HealthReport, HealthStatus

    healthy_report = HealthReport(
        overall_status=HealthStatus.HEALTHY,
        component_status={},
        last_check=datetime.now(timezone.utc),
        issues=[],
    )
    assert healthy_report.is_healthy is True

    degraded_report = HealthReport(
        overall_status=HealthStatus.DEGRADED,
        component_status={},
        last_check=datetime.now(timezone.utc),
        issues=[],
    )
    assert degraded_report.is_healthy is False


def test_health_report_is_operational():
    """Test HealthReport is_operational property."""
    from core.health_check import HealthReport, HealthStatus

    healthy_report = HealthReport(
        overall_status=HealthStatus.HEALTHY,
        component_status={},
        last_check=datetime.now(timezone.utc),
        issues=[],
    )
    assert healthy_report.is_operational is True

    degraded_report = HealthReport(
        overall_status=HealthStatus.DEGRADED,
        component_status={},
        last_check=datetime.now(timezone.utc),
        issues=[],
    )
    assert degraded_report.is_operational is True

    unhealthy_report = HealthReport(
        overall_status=HealthStatus.UNHEALTHY,
        component_status={},
        last_check=datetime.now(timezone.utc),
        issues=[],
    )
    assert unhealthy_report.is_operational is False


# =============================================================================
# Test PrometheusMetrics
# =============================================================================

def test_prometheus_metrics_gauge():
    """Test PrometheusMetrics gauge functionality."""
    from core.health_check import PrometheusMetrics

    metrics = PrometheusMetrics()
    metrics.gauge("test_gauge", 42.5)

    output = metrics.export()
    assert "# TYPE test_gauge gauge" in output
    assert "test_gauge 42.5" in output


def test_prometheus_metrics_with_labels():
    """Test PrometheusMetrics with labels."""
    from core.health_check import PrometheusMetrics

    metrics = PrometheusMetrics()
    metrics.gauge("test_metric", 100, {"component": "exa", "category": "adapter"})

    output = metrics.export()
    assert 'test_metric{category="adapter",component="exa"} 100' in output


def test_prometheus_metrics_counter():
    """Test PrometheusMetrics counter functionality."""
    from core.health_check import PrometheusMetrics

    metrics = PrometheusMetrics()
    metrics.counter("test_counter", 5)

    output = metrics.export()
    assert "# TYPE test_counter counter" in output
    assert "test_counter 5" in output


def test_prometheus_metrics_clear():
    """Test PrometheusMetrics clear functionality."""
    from core.health_check import PrometheusMetrics

    metrics = PrometheusMetrics()
    metrics.gauge("test", 1)
    metrics.clear()

    output = metrics.export()
    assert output == ""


# =============================================================================
# Test HealthChecker
# =============================================================================

def test_health_checker_singleton():
    """Test HealthChecker singleton pattern."""
    from core.health_check import HealthChecker

    # Reset singleton for test
    HealthChecker._instance = None

    checker1 = HealthChecker()
    checker2 = HealthChecker()

    assert checker1 is checker2


def test_health_checker_register_check():
    """Test registering custom health checks."""
    from core.health_check import (
        HealthChecker, ComponentHealth, HealthStatus, ComponentCategory
    )

    # Reset singleton for test
    HealthChecker._instance = None
    checker = HealthChecker()

    async def custom_check():
        return ComponentHealth(
            name="custom",
            status=HealthStatus.HEALTHY,
            category=ComponentCategory.EXTERNAL,
        )

    checker.register_check("custom_service", custom_check, ComponentCategory.EXTERNAL)

    assert "custom_service" in checker._checks


def test_health_checker_unregister_check():
    """Test unregistering health checks."""
    from core.health_check import (
        HealthChecker, ComponentHealth, HealthStatus, ComponentCategory
    )

    # Reset singleton for test
    HealthChecker._instance = None
    checker = HealthChecker()

    async def custom_check():
        return ComponentHealth(
            name="custom",
            status=HealthStatus.HEALTHY,
            category=ComponentCategory.EXTERNAL,
        )

    checker.register_check("to_remove", custom_check)
    assert "to_remove" in checker._checks

    checker.unregister_check("to_remove")
    assert "to_remove" not in checker._checks


@pytest.mark.asyncio
async def test_health_checker_check_all():
    """Test running all health checks."""
    from core.health_check import HealthChecker, HealthStatus

    # Reset singleton for test
    HealthChecker._instance = None
    checker = HealthChecker()

    # Mock the adapter registry check to return predictable results
    with patch.object(checker, 'check_adapters', new_callable=AsyncMock) as mock_adapters, \
         patch.object(checker, 'check_memory', new_callable=AsyncMock) as mock_memory, \
         patch.object(checker, 'check_rag', new_callable=AsyncMock) as mock_rag:

        mock_adapters.return_value = {}
        mock_memory.return_value = {}
        mock_rag.return_value = {}

        report = await checker.check_all(timeout=5.0)

        assert report is not None
        assert report.overall_status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
            HealthStatus.UNKNOWN,
        ]


@pytest.mark.asyncio
async def test_health_checker_calculate_overall_status():
    """Test overall status calculation logic."""
    from core.health_check import (
        HealthChecker, ComponentHealth, HealthStatus, ComponentCategory
    )

    # Reset singleton for test
    HealthChecker._instance = None
    checker = HealthChecker()

    # All healthy
    all_healthy = {
        "a": ComponentHealth(name="a", status=HealthStatus.HEALTHY, category=ComponentCategory.SYSTEM),
        "b": ComponentHealth(name="b", status=HealthStatus.HEALTHY, category=ComponentCategory.SYSTEM),
    }
    assert checker._calculate_overall_status(all_healthy) == HealthStatus.HEALTHY

    # Some degraded
    some_degraded = {
        "a": ComponentHealth(name="a", status=HealthStatus.HEALTHY, category=ComponentCategory.SYSTEM),
        "b": ComponentHealth(name="b", status=HealthStatus.DEGRADED, category=ComponentCategory.SYSTEM),
    }
    # Less than 30% degraded, no unhealthy = healthy
    # But degraded > 30% threshold test
    many_degraded = {
        "a": ComponentHealth(name="a", status=HealthStatus.DEGRADED, category=ComponentCategory.SYSTEM),
        "b": ComponentHealth(name="b", status=HealthStatus.DEGRADED, category=ComponentCategory.SYSTEM),
        "c": ComponentHealth(name="c", status=HealthStatus.HEALTHY, category=ComponentCategory.SYSTEM),
    }
    # 66% degraded > 30% threshold
    assert checker._calculate_overall_status(many_degraded) == HealthStatus.DEGRADED

    # Any unhealthy = degraded (unless > 20%)
    some_unhealthy = {
        "a": ComponentHealth(name="a", status=HealthStatus.HEALTHY, category=ComponentCategory.SYSTEM),
        "b": ComponentHealth(name="b", status=HealthStatus.UNHEALTHY, category=ComponentCategory.SYSTEM),
        "c": ComponentHealth(name="c", status=HealthStatus.HEALTHY, category=ComponentCategory.SYSTEM),
        "d": ComponentHealth(name="d", status=HealthStatus.HEALTHY, category=ComponentCategory.SYSTEM),
        "e": ComponentHealth(name="e", status=HealthStatus.HEALTHY, category=ComponentCategory.SYSTEM),
    }
    # 20% unhealthy = degraded
    assert checker._calculate_overall_status(some_unhealthy) == HealthStatus.DEGRADED

    # > 20% unhealthy = unhealthy
    mostly_unhealthy = {
        "a": ComponentHealth(name="a", status=HealthStatus.UNHEALTHY, category=ComponentCategory.SYSTEM),
        "b": ComponentHealth(name="b", status=HealthStatus.UNHEALTHY, category=ComponentCategory.SYSTEM),
        "c": ComponentHealth(name="c", status=HealthStatus.HEALTHY, category=ComponentCategory.SYSTEM),
    }
    # 66% unhealthy > 20%
    assert checker._calculate_overall_status(mostly_unhealthy) == HealthStatus.UNHEALTHY


def test_health_checker_export_prometheus():
    """Test Prometheus export functionality."""
    from core.health_check import HealthChecker

    # Reset singleton for test
    HealthChecker._instance = None
    checker = HealthChecker()

    # Add some metrics manually
    checker._metrics.gauge("test_metric", 42)

    output = checker.export_prometheus()
    assert "test_metric" in output


# =============================================================================
# Test Factory Functions
# =============================================================================

def test_get_health_checker():
    """Test get_health_checker factory function."""
    from core.health_check import get_health_checker, HealthChecker

    # Reset both singletons
    HealthChecker._instance = None
    import core.health_check as module
    module._health_checker = None

    checker = get_health_checker()
    assert checker is not None
    assert isinstance(checker, HealthChecker)


@pytest.mark.asyncio
async def test_quick_health_check():
    """Test quick_health_check function."""
    from core.health_check import quick_health_check, HealthChecker
    import core.health_check as module

    # Reset singletons
    HealthChecker._instance = None
    module._health_checker = None

    result = await quick_health_check()

    assert isinstance(result, dict)
    assert "overall_status" in result
    assert "component_status" in result
    assert "summary" in result


# =============================================================================
# Test Web Integration
# =============================================================================

def test_create_health_endpoint():
    """Test health endpoint factory."""
    from core.health_check import create_health_endpoint

    endpoint = create_health_endpoint()
    assert callable(endpoint)


def test_create_prometheus_endpoint():
    """Test Prometheus endpoint factory."""
    from core.health_check import create_prometheus_endpoint

    endpoint = create_prometheus_endpoint()
    assert callable(endpoint)


# =============================================================================
# Test Exports
# =============================================================================

def test_module_exports():
    """Test that all expected symbols are exported."""
    from core.health_check import (
        HealthStatus,
        ComponentCategory,
        ComponentHealth,
        HealthReport,
        PrometheusMetrics,
        HealthChecker,
        get_health_checker,
        quick_health_check,
        create_health_endpoint,
        create_prometheus_endpoint,
    )

    # All imports should succeed
    assert HealthStatus is not None
    assert ComponentCategory is not None
    assert ComponentHealth is not None
    assert HealthReport is not None
    assert PrometheusMetrics is not None
    assert HealthChecker is not None
    assert get_health_checker is not None
    assert quick_health_check is not None
    assert create_health_endpoint is not None
    assert create_prometheus_endpoint is not None
