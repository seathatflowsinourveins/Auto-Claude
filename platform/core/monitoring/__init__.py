"""
UNLEASH Production Monitoring Module

Comprehensive production monitoring with:
- Opik (Comet-ML) LLM observability
- Langfuse LLM analytics
- Phoenix (Arize) tracing
- Prometheus metrics export
- OpenTelemetry (OTLP) support

Usage:
    from core.monitoring import create_production_loop, ProductionMonitoringLoop

    # Create and start the monitoring loop
    loop = create_production_loop(service_name="my-service")
    await loop.start()

    # Record requests
    loop.record_request(
        adapter="tavily",
        method="search",
        latency_ms=250.0,
        input_tokens=100,
        output_tokens=500,
        model="claude-sonnet-4",
    )

    # Or use context managers
    async with loop.atrace_request("search", adapter="tavily") as span:
        result = await adapter.search(query)
        span.attributes["result_count"] = len(result)

    # Get dashboard data
    dashboard = loop.get_dashboard_data()

    # Get Prometheus metrics
    prometheus_text = loop.get_prometheus_metrics()

    # Stop when done
    await loop.stop()
"""

from .production_loop import (
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
    Exporter,
    OpikExporter,
    LangfuseExporter,
    PrometheusExporter,
    OTLPExporter,
    JSONExporter,

    # Main Loop
    ProductionMonitoringLoop,

    # Ralph Integration
    RalphMonitoringIntegration,

    # Factory
    create_production_loop,
    quick_health_check,
)

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

__version__ = "1.0.0"
