#!/usr/bin/env python3
"""
Phoenix Compatibility Layer for V35 Validation.

This module re-exports PhoenixMonitor as PhoenixCompat for consistency
with the V35 validation script's import expectations.

Usage:
    from core.observability.phoenix_compat import PhoenixCompat, PHOENIX_COMPAT_AVAILABLE

    if PHOENIX_COMPAT_AVAILABLE:
        monitor = PhoenixCompat()
        monitor.record_request(latency_ms=100, tokens=500, cost_usd=0.01)
"""

from .phoenix_monitor import (
    PhoenixMonitor,
    AlertSeverity,
    DriftType,
    MonitorStatus,
    Alert,
    EmbeddingRecord,
    DriftResult,
    MonitoringMetrics,
    AlertConfig,
    create_phoenix_monitor,
    OPENTELEMETRY_AVAILABLE,
    PHOENIX_AVAILABLE,
)

# V35 Compat alias
PhoenixCompat = PhoenixMonitor

# Compatibility flag for V35 validation
PHOENIX_COMPAT_AVAILABLE = True

__all__ = [
    # V35 compat alias
    "PhoenixCompat",
    "PHOENIX_COMPAT_AVAILABLE",
    # Original exports
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
