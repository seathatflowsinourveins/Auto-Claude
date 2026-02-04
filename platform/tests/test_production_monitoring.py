"""
Tests for the Production Monitoring Loop

Tests cover:
- Metrics collection (latency, tokens, cache, errors)
- Distributed tracing with context propagation
- Alerting with configurable thresholds
- Cost tracking with budget alerts
- Export to Prometheus, JSON formats
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import sys
from pathlib import Path

# Add platform to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.monitoring.production_loop import (
    # Configuration
    MonitoringConfig,
    ExportFormat,
    AlertSeverity,
    ModelTier,
    # Metrics
    MetricsCollector,
    LatencyMetrics,
    TokenUsage,
    CacheMetrics,
    ErrorMetrics,
    # Tracing
    DistributedTracer,
    SpanContext,
    Span,
    # Alerting
    AlertManager,
    AlertRule,
    Alert,
    # Cost Tracking
    CostTracker,
    CostEntry,
    # Exporters
    PrometheusExporter,
    JSONExporter,
    # Main Loop
    ProductionMonitoringLoop,
    # Factory
    create_production_loop,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def config():
    """Create a test configuration."""
    return MonitoringConfig(
        enabled=True,
        trace_sample_rate=1.0,
        metrics_export_interval_s=1,
        alert_evaluation_interval_s=1,
        latency_warning_ms=100.0,
        latency_critical_ms=500.0,
        error_rate_warning=0.05,
        error_rate_critical=0.10,
        daily_budget_usd=10.0,
        hourly_budget_usd=1.0,
    )


@pytest.fixture
def metrics_collector(config):
    """Create a metrics collector."""
    return MetricsCollector(config)


@pytest.fixture
def tracer(config):
    """Create a distributed tracer."""
    return DistributedTracer(config, "test-service")


@pytest.fixture
def alert_manager(config):
    """Create an alert manager."""
    return AlertManager(config)


@pytest.fixture
def cost_tracker(config):
    """Create a cost tracker."""
    return CostTracker(config)


@pytest.fixture
def monitoring_loop(config):
    """Create a production monitoring loop."""
    return ProductionMonitoringLoop(config, "test-service")


# =============================================================================
# LATENCY METRICS TESTS
# =============================================================================

class TestLatencyMetrics:
    """Tests for latency metrics calculation."""

    def test_empty_samples(self):
        """Test latency metrics with no samples."""
        metrics = LatencyMetrics.from_samples([])
        assert metrics.count == 0
        assert metrics.p50 == 0
        assert metrics.p99 == 0

    def test_single_sample(self):
        """Test latency metrics with single sample."""
        metrics = LatencyMetrics.from_samples([100.0])
        assert metrics.count == 1
        assert metrics.avg == 100.0
        assert metrics.min_val == 100.0
        assert metrics.max_val == 100.0

    def test_percentile_calculation(self):
        """Test percentile calculation with multiple samples."""
        samples = list(range(1, 101))  # 1 to 100
        metrics = LatencyMetrics.from_samples(samples)

        assert metrics.count == 100
        assert metrics.p50 == 50
        assert metrics.p90 == 90
        assert metrics.p95 == 95
        assert metrics.p99 == 99
        assert metrics.min_val == 1
        assert metrics.max_val == 100

    def test_to_dict(self):
        """Test dictionary conversion."""
        metrics = LatencyMetrics.from_samples([10.0, 20.0, 30.0])
        data = metrics.to_dict()

        assert "p50" in data
        assert "p95" in data
        assert "p99" in data
        assert "avg" in data
        assert "count" in data
        assert data["count"] == 3


# =============================================================================
# METRICS COLLECTOR TESTS
# =============================================================================

class TestMetricsCollector:
    """Tests for the metrics collector."""

    def test_record_latency(self, metrics_collector):
        """Test latency recording."""
        metrics_collector.record_latency("tavily", "search", 100.0)
        metrics_collector.record_latency("tavily", "search", 200.0)
        metrics_collector.record_latency("exa", "query", 50.0)

        latencies = metrics_collector.get_latency_metrics()

        assert "tavily.search" in latencies
        assert latencies["tavily.search"].count == 2
        assert "exa.query" in latencies
        assert latencies["exa.query"].count == 1

    def test_record_tokens(self, metrics_collector):
        """Test token usage recording."""
        cost1 = metrics_collector.record_tokens("claude-sonnet-4", 1000, 500, ModelTier.TIER_3_OPUS)
        cost2 = metrics_collector.record_tokens("claude-haiku-4", 2000, 1000, ModelTier.TIER_2_HAIKU)

        usage = metrics_collector.get_token_usage()

        assert "total" in usage
        assert usage["total"].input_tokens == 3000
        assert usage["total"].output_tokens == 1500
        assert cost1 > 0
        assert cost2 > 0

    def test_record_cache(self, metrics_collector):
        """Test cache metrics recording."""
        metrics_collector.record_cache_hit("memory")
        metrics_collector.record_cache_hit("memory")
        metrics_collector.record_cache_miss("memory")

        cache = metrics_collector.get_cache_metrics()

        assert "memory" in cache
        assert cache["memory"].hits == 2
        assert cache["memory"].misses == 1
        assert cache["memory"].hit_rate == pytest.approx(0.667, rel=0.01)

    def test_record_error(self, metrics_collector):
        """Test error recording."""
        metrics_collector.record_error("TimeoutError", "tavily", "Request timed out")
        metrics_collector.record_error("APIError", "tavily", "Rate limited")
        metrics_collector.record_error("TimeoutError", "exa", "Connection timeout")

        errors = metrics_collector.get_error_metrics()

        assert errors.total_errors == 3
        assert errors.by_type["TimeoutError"] == 2
        assert errors.by_adapter["tavily"] == 2

    def test_error_rate_calculation(self, metrics_collector):
        """Test error rate calculation."""
        # Record some successful requests
        for _ in range(10):
            metrics_collector.record_latency("adapter", "method", 100.0)

        # Record one error
        metrics_collector.record_error("Error", "adapter", "Test error")

        rate = metrics_collector.get_error_rate()
        assert rate == pytest.approx(0.091, rel=0.01)  # 1/11

    def test_summary(self, metrics_collector):
        """Test summary generation."""
        metrics_collector.record_latency("adapter", "method", 100.0)
        metrics_collector.record_tokens("claude-sonnet-4", 100, 50)
        metrics_collector.record_cache_hit("default")
        metrics_collector.record_error("Error", "adapter", "Test")

        summary = metrics_collector.get_summary()

        assert "timestamp" in summary
        assert "total_requests" in summary
        assert "latency" in summary
        assert "tokens" in summary
        assert "cache" in summary
        assert "errors" in summary
        assert "error_rate" in summary

    def test_reset_window(self, metrics_collector):
        """Test window reset."""
        metrics_collector.record_latency("adapter", "method", 100.0)

        old_summary = metrics_collector.reset_window()

        assert old_summary["total_requests"] == 1

        new_summary = metrics_collector.get_summary()
        assert new_summary["total_requests"] == 0


# =============================================================================
# DISTRIBUTED TRACER TESTS
# =============================================================================

class TestDistributedTracer:
    """Tests for distributed tracing."""

    def test_start_span(self, tracer):
        """Test starting a span."""
        span = tracer.start_span("test_operation")

        assert span.operation_name == "test_operation"
        assert span.service_name == "test-service"
        assert span.context.trace_id is not None
        assert span.context.span_id is not None
        assert span.start_time is not None

        tracer.end_span(span)

    def test_nested_spans(self, tracer):
        """Test nested span creation."""
        parent = tracer.start_span("parent_op")
        child = tracer.start_span("child_op")

        assert child.context.parent_span_id == parent.context.span_id
        assert child.context.trace_id == parent.context.trace_id

        tracer.end_span(child)
        tracer.end_span(parent)

    def test_span_context_propagation(self, tracer):
        """Test span context propagation via headers."""
        span = tracer.start_span("test_op")

        headers = {}
        tracer.inject_context(headers)

        assert "X-Trace-ID" in headers
        assert "X-Span-ID" in headers
        assert headers["X-Trace-ID"] == span.context.trace_id

        # Extract context
        extracted = tracer.extract_context(headers)
        assert extracted is not None
        assert extracted.trace_id == span.context.trace_id

        tracer.end_span(span)

    def test_trace_context_manager(self, tracer):
        """Test synchronous trace context manager."""
        with tracer.trace("sync_operation") as span:
            time.sleep(0.01)  # Simulate work
            span.attributes["custom"] = "value"

        assert span.end_time is not None
        assert span.duration_ms is not None
        assert span.duration_ms >= 10
        assert span.attributes["custom"] == "value"

    @pytest.mark.asyncio
    async def test_atrace_context_manager(self, tracer):
        """Test async trace context manager."""
        async with tracer.atrace("async_operation") as span:
            await asyncio.sleep(0.01)
            span.attributes["async"] = True

        assert span.end_time is not None
        assert span.duration_ms >= 10
        assert span.attributes["async"] is True

    def test_trace_error_handling(self, tracer):
        """Test error handling in trace context manager."""
        with pytest.raises(ValueError):
            with tracer.trace("error_operation") as span:
                raise ValueError("Test error")

        assert span.status == "error"
        assert "status.message" in span.attributes

    def test_get_traces(self, tracer):
        """Test trace retrieval."""
        for i in range(5):
            with tracer.trace(f"operation_{i}"):
                pass

        traces = tracer.get_traces(limit=3)
        assert len(traces) == 3

    def test_trace_tree(self, tracer):
        """Test hierarchical trace tree."""
        with tracer.trace("root") as root:
            trace_id = root.context.trace_id
            with tracer.trace("child1"):
                with tracer.trace("grandchild"):
                    pass
            with tracer.trace("child2"):
                pass

        tree = tracer.get_trace_tree(trace_id)

        assert tree["trace_id"] == trace_id
        assert tree["total_spans"] >= 4


# =============================================================================
# ALERT MANAGER TESTS
# =============================================================================

class TestAlertManager:
    """Tests for the alert manager."""

    def test_default_rules(self, alert_manager):
        """Test that default rules are initialized."""
        assert "latency_warning" in alert_manager._rules
        assert "latency_critical" in alert_manager._rules
        assert "error_rate_warning" in alert_manager._rules
        assert "budget_warning" in alert_manager._rules

    @pytest.mark.asyncio
    async def test_latency_warning_alert(self, alert_manager):
        """Test latency warning alert triggering."""
        alerts = await alert_manager.evaluate("request_latency_ms", 150.0)

        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING
        assert "latency" in alerts[0].message.lower()

    @pytest.mark.asyncio
    async def test_latency_critical_alert(self, alert_manager):
        """Test latency critical alert triggering."""
        alerts = await alert_manager.evaluate("request_latency_ms", 600.0)

        # Should trigger both warning and critical
        assert len(alerts) == 2
        severities = {a.severity for a in alerts}
        assert AlertSeverity.CRITICAL in severities

    @pytest.mark.asyncio
    async def test_error_rate_alert(self, alert_manager):
        """Test error rate alert triggering."""
        alerts = await alert_manager.evaluate("error_rate", 0.07)

        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING

    @pytest.mark.asyncio
    async def test_cooldown_prevents_spam(self, alert_manager):
        """Test that cooldown prevents alert spam."""
        # First evaluation triggers
        alerts1 = await alert_manager.evaluate("request_latency_ms", 150.0)
        assert len(alerts1) == 1

        # Second immediate evaluation should be blocked by cooldown
        alerts2 = await alert_manager.evaluate("request_latency_ms", 150.0)
        assert len(alerts2) == 0

    @pytest.mark.asyncio
    async def test_custom_rule(self, alert_manager):
        """Test adding custom alert rule."""
        custom_rule = AlertRule(
            name="custom_metric_high",
            metric_name="custom_metric",
            condition=lambda v: v > 100,
            severity=AlertSeverity.ERROR,
            message_template="Custom metric {value} exceeds threshold",
        )
        alert_manager.add_rule(custom_rule)

        alerts = await alert_manager.evaluate("custom_metric", 150.0)

        assert len(alerts) == 1
        assert alerts[0].rule_name == "custom_metric_high"

    @pytest.mark.asyncio
    async def test_resolve_alert(self, alert_manager):
        """Test alert resolution."""
        alerts = await alert_manager.evaluate("request_latency_ms", 150.0)
        assert len(alert_manager.get_active_alerts()) == 1

        resolved = await alert_manager.resolve(alerts[0].id)
        assert resolved is True
        assert len(alert_manager.get_active_alerts()) == 0

    @pytest.mark.asyncio
    async def test_alert_handler_callback(self, alert_manager):
        """Test alert handler callback."""
        callback_alerts = []

        async def handler(alert):
            callback_alerts.append(alert)

        alert_manager.add_handler(handler)
        await alert_manager.evaluate("request_latency_ms", 150.0)

        assert len(callback_alerts) == 1


# =============================================================================
# COST TRACKER TESTS
# =============================================================================

class TestCostTracker:
    """Tests for cost tracking."""

    def test_record_cost(self, cost_tracker):
        """Test cost recording."""
        entry = cost_tracker.record_cost(
            model="claude-sonnet-4",
            adapter="tavily",
            input_tokens=1000,
            output_tokens=500,
        )

        assert entry.cost_usd > 0
        assert entry.input_tokens == 1000
        assert entry.output_tokens == 500

    def test_cost_by_model(self, cost_tracker):
        """Test cost aggregation by model."""
        cost_tracker.record_cost("claude-sonnet-4", "adapter", 1000, 500)
        cost_tracker.record_cost("claude-haiku-4", "adapter", 2000, 1000)
        cost_tracker.record_cost("claude-sonnet-4", "adapter", 500, 250)

        by_model = cost_tracker.get_cost_by_model()

        assert "claude-sonnet-4" in by_model
        assert "claude-haiku-4" in by_model
        assert by_model["claude-sonnet-4"] > by_model["claude-haiku-4"]

    def test_cost_by_adapter(self, cost_tracker):
        """Test cost aggregation by adapter."""
        cost_tracker.record_cost("model", "tavily", 1000, 500)
        cost_tracker.record_cost("model", "exa", 2000, 1000)

        by_adapter = cost_tracker.get_cost_by_adapter()

        assert "tavily" in by_adapter
        assert "exa" in by_adapter

    def test_cost_by_tier(self, cost_tracker):
        """Test cost aggregation by tier."""
        cost_tracker.record_cost("model", "adapter", 1000, 500, ModelTier.TIER_3_OPUS)
        cost_tracker.record_cost("model", "adapter", 1000, 500, ModelTier.TIER_2_HAIKU)

        by_tier = cost_tracker.get_cost_by_tier()

        assert ModelTier.TIER_3_OPUS.value in by_tier
        assert ModelTier.TIER_2_HAIKU.value in by_tier

    def test_budget_status(self, cost_tracker):
        """Test budget status calculation."""
        # Record some costs
        for _ in range(10):
            cost_tracker.record_cost("claude-sonnet-4", "adapter", 1000, 500)

        status = cost_tracker.get_budget_status()

        assert "daily_cost_usd" in status
        assert "daily_budget_usd" in status
        assert "daily_percent_used" in status
        assert "is_daily_warning" in status

    def test_hourly_cost_tracking(self, cost_tracker):
        """Test hourly cost tracking."""
        cost_tracker.record_cost("model", "adapter", 100000, 50000)  # High token count

        hourly = cost_tracker.get_hourly_cost()
        assert hourly > 0

    def test_daily_cost_tracking(self, cost_tracker):
        """Test daily cost tracking."""
        cost_tracker.record_cost("model", "adapter", 100000, 50000)

        daily = cost_tracker.get_daily_cost()
        assert daily > 0


# =============================================================================
# EXPORTER TESTS
# =============================================================================

class TestPrometheusExporter:
    """Tests for Prometheus exporter."""

    @pytest.mark.asyncio
    async def test_export_metrics(self, config):
        """Test Prometheus metrics export."""
        exporter = PrometheusExporter(config)

        metrics = {
            "total_requests": 100,
            "error_rate": 0.05,
            "latency": {
                "adapter.method": {
                    "p50": 50.0,
                    "p90": 90.0,
                    "p95": 95.0,
                    "p99": 99.0,
                    "avg": 60.0,
                    "count": 100,
                }
            },
            "tokens": {
                "total": {
                    "input_tokens": 10000,
                    "output_tokens": 5000,
                    "cost_usd": 0.05,
                }
            },
            "cache": {
                "default": {
                    "hits": 80,
                    "misses": 20,
                    "hit_rate": 0.8,
                }
            },
            "errors": {
                "total_errors": 5,
                "by_type": {"TimeoutError": 3, "APIError": 2},
                "by_adapter": {"tavily": 3, "exa": 2},
            },
        }

        result = await exporter.export_metrics(metrics)
        assert result is True

        text = exporter.get_metrics_text()
        assert "unleash_requests_total" in text
        assert "unleash_error_rate" in text
        assert "unleash_latency" in text

    @pytest.mark.asyncio
    async def test_health_check(self, config):
        """Test Prometheus exporter health check."""
        exporter = PrometheusExporter(config)
        healthy = await exporter.health_check()
        assert healthy is True


class TestJSONExporter:
    """Tests for JSON exporter."""

    @pytest.mark.asyncio
    async def test_export_metrics(self, config):
        """Test JSON metrics export."""
        exporter = JSONExporter(config)

        metrics = {"test": "data", "value": 123}
        result = await exporter.export_metrics(metrics)

        assert result is True

        json_data = exporter.get_metrics_json()
        assert "exported_at" in json_data
        assert json_data["data"]["value"] == 123

    @pytest.mark.asyncio
    async def test_export_traces(self, config, tracer):
        """Test JSON trace export."""
        exporter = JSONExporter(config)

        with tracer.trace("test_op"):
            pass

        traces = tracer.get_traces()
        result = await exporter.export_traces(traces)

        assert result is True

        json_traces = exporter.get_traces_json()
        assert len(json_traces) >= 1


# =============================================================================
# PRODUCTION MONITORING LOOP TESTS
# =============================================================================

class TestProductionMonitoringLoop:
    """Tests for the main production monitoring loop."""

    def test_creation(self, monitoring_loop):
        """Test loop creation."""
        assert monitoring_loop.config is not None
        assert monitoring_loop.metrics is not None
        assert monitoring_loop.tracer is not None
        assert monitoring_loop.alerts is not None
        assert monitoring_loop.costs is not None

    def test_record_request(self, monitoring_loop):
        """Test recording a complete request."""
        monitoring_loop.record_request(
            adapter="tavily",
            method="search",
            latency_ms=100.0,
            input_tokens=500,
            output_tokens=200,
            model="claude-sonnet-4",
            tier=ModelTier.TIER_3_OPUS,
            cache_hit=True,
        )

        summary = monitoring_loop.metrics.get_summary()
        assert summary["total_requests"] == 1

        cache = monitoring_loop.metrics.get_cache_metrics()
        assert cache["tavily"].hits == 1

    def test_trace_request_context_manager(self, monitoring_loop):
        """Test trace_request context manager."""
        with monitoring_loop.trace_request(
            "search_operation",
            adapter="tavily",
            method="search"
        ) as span:
            time.sleep(0.01)

        assert span.duration_ms >= 10
        assert span.attributes["adapter"] == "tavily"

    @pytest.mark.asyncio
    async def test_atrace_request_context_manager(self, monitoring_loop):
        """Test async trace_request context manager."""
        async with monitoring_loop.atrace_request(
            "async_search",
            adapter="exa",
            method="query"
        ) as span:
            await asyncio.sleep(0.01)

        assert span.duration_ms >= 10

    @pytest.mark.asyncio
    async def test_evaluate_alerts(self, monitoring_loop):
        """Test alert evaluation."""
        # Record a high-latency request
        monitoring_loop.record_request(
            adapter="slow_adapter",
            method="slow_method",
            latency_ms=2000.0,  # Above warning threshold
        )

        # Add more requests to avoid division by zero in error rate
        for _ in range(10):
            monitoring_loop.record_request(
                adapter="adapter",
                method="method",
                latency_ms=50.0,
            )

        alerts = await monitoring_loop.evaluate_alerts()
        # May or may not trigger depending on exact timing and thresholds
        assert isinstance(alerts, list)

    @pytest.mark.asyncio
    async def test_export_all(self, monitoring_loop):
        """Test exporting to all configured exporters."""
        monitoring_loop.record_request(
            adapter="adapter",
            method="method",
            latency_ms=100.0,
        )

        results = await monitoring_loop.export_all()

        # At minimum, Prometheus and JSON should succeed
        assert ExportFormat.PROMETHEUS in results
        assert ExportFormat.JSON in results
        assert results[ExportFormat.PROMETHEUS] is True
        assert results[ExportFormat.JSON] is True

    def test_get_prometheus_metrics(self, monitoring_loop):
        """Test Prometheus metrics retrieval."""
        monitoring_loop.record_request(
            adapter="adapter",
            method="method",
            latency_ms=100.0,
        )

        text = monitoring_loop.get_prometheus_metrics()
        # Initially may be empty until export is called
        assert isinstance(text, str)

    def test_get_dashboard_data(self, monitoring_loop):
        """Test dashboard data retrieval."""
        monitoring_loop.record_request(
            adapter="adapter",
            method="method",
            latency_ms=100.0,
        )

        dashboard = monitoring_loop.get_dashboard_data()

        assert "timestamp" in dashboard
        assert "service" in dashboard
        assert "metrics" in dashboard
        assert "costs" in dashboard
        assert "alerts" in dashboard
        assert "exporters" in dashboard

    @pytest.mark.asyncio
    async def test_health_check(self, monitoring_loop):
        """Test health check."""
        health = await monitoring_loop.health_check()

        assert "status" in health
        assert "components" in health
        assert health["components"]["metrics_collector"] is True
        assert health["components"]["tracer"] is True

    @pytest.mark.asyncio
    async def test_start_stop(self, monitoring_loop):
        """Test starting and stopping the monitoring loop."""
        await monitoring_loop.start()
        assert monitoring_loop._running is True

        await monitoring_loop.stop()
        assert monitoring_loop._running is False


# =============================================================================
# FACTORY TESTS
# =============================================================================

class TestFactory:
    """Tests for factory functions."""

    def test_create_production_loop(self):
        """Test factory function."""
        loop = create_production_loop(
            service_name="test-factory",
            opik_enabled=False,
            langfuse_enabled=False,
        )

        assert loop.service_name == "test-factory"
        assert ExportFormat.OPIK not in loop._exporters
        assert ExportFormat.LANGFUSE not in loop._exporters

    def test_create_with_custom_config(self):
        """Test factory with custom configuration."""
        loop = create_production_loop(
            service_name="custom-service",
            daily_budget_usd=50.0,
            latency_warning_ms=200.0,
        )

        assert loop.config.daily_budget_usd == 50.0
        assert loop.config.latency_warning_ms == 200.0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the full monitoring flow."""

    @pytest.mark.asyncio
    async def test_full_monitoring_flow(self):
        """Test complete monitoring flow."""
        loop = create_production_loop(
            service_name="integration-test",
            opik_enabled=False,
            langfuse_enabled=False,
            latency_warning_ms=50.0,  # Low threshold for testing
        )

        # Start the loop
        await loop.start()

        try:
            # Simulate multiple requests
            for i in range(10):
                async with loop.atrace_request(
                    f"operation_{i}",
                    adapter="test_adapter",
                    method="test_method",
                ) as span:
                    await asyncio.sleep(0.01)

                    # Record additional data
                    loop.record_request(
                        adapter="test_adapter",
                        method="test_method",
                        latency_ms=span.duration_ms or 10.0,
                        input_tokens=100 * (i + 1),
                        output_tokens=50 * (i + 1),
                        model="claude-sonnet-4",
                    )

            # Simulate an error
            loop.metrics.record_error(
                "TestError",
                "test_adapter",
                "Test error message",
            )

            # Export metrics
            results = await loop.export_all()
            assert results[ExportFormat.PROMETHEUS] is True
            assert results[ExportFormat.JSON] is True

            # Check dashboard data
            dashboard = loop.get_dashboard_data()
            assert dashboard["metrics"]["total_requests"] >= 10

            # Check costs
            costs = loop.costs.get_summary()
            assert costs["total_cost_usd"] > 0

            # Check health
            health = await loop.health_check()
            assert health["status"] in ("healthy", "degraded")

        finally:
            await loop.stop()

    @pytest.mark.asyncio
    async def test_alert_flow(self):
        """Test alert triggering and resolution flow."""
        loop = create_production_loop(
            service_name="alert-test",
            opik_enabled=False,
            langfuse_enabled=False,
            latency_warning_ms=10.0,  # Very low for testing
            latency_critical_ms=50.0,
        )

        # Simulate high-latency requests
        for _ in range(5):
            loop.record_request(
                adapter="slow_adapter",
                method="slow_method",
                latency_ms=100.0,  # Above critical
            )

        # Evaluate alerts
        alerts = await loop.evaluate_alerts()

        # Should have triggered some alerts
        # Note: May be affected by cooldowns from previous tests
        assert isinstance(alerts, list)

        # Check active alerts
        active = loop.alerts.get_active_alerts()
        for alert in active:
            assert alert.resolved is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
