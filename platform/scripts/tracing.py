#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "opentelemetry-api>=1.22.0",
#     "opentelemetry-sdk>=1.22.0",
#     "opentelemetry-exporter-otlp>=1.22.0",
#     "opentelemetry-instrumentation>=0.43b0",
# ]
# ///
"""
Platform Tracing - OpenTelemetry Distributed Tracing

Provides distributed tracing for the Ultimate Autonomous Platform.
Traces can be exported to Jaeger, Zipkin, or any OTLP-compatible backend.

Features:
- Automatic span context propagation
- Custom span attributes for platform operations
- Integration with circuit breakers and bulkheads
- Sampling strategies for high-volume operations

Usage:
    from tracing import get_tracer, trace_operation

    tracer = get_tracer()

    with tracer.start_as_current_span("health_check") as span:
        span.set_attribute("component", "qdrant")
        # ... perform operation
        span.set_attribute("latency_ms", 45.2)
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# Check if OTLP exporter is available (optional dependency)
try:
    import opentelemetry.exporter.otlp.proto.grpc.trace_exporter  # noqa: F401
    OTLP_AVAILABLE = True
except ImportError:
    OTLP_AVAILABLE = False


# Type variable for generic function decorator
F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class TracingConfig:
    """Configuration for distributed tracing."""
    service_name: str = "ultimate-autonomous-platform"
    service_version: str = "1.0.0"
    environment: str = "development"
    otlp_endpoint: Optional[str] = None
    console_export: bool = True
    sampling_rate: float = 1.0  # 1.0 = 100% sampling


class TracingManager:
    """
    Manages OpenTelemetry tracing for the platform.

    Supports multiple exporters:
    - Console (for development/debugging)
    - OTLP (for Jaeger, Zipkin, or other backends)
    """

    _instance: Optional['TracingManager'] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if TracingManager._initialized:
            return
        TracingManager._initialized = True

        self._tracer_provider: Optional[TracerProvider] = None
        self._tracer: Optional[trace.Tracer] = None
        self._propagator = TraceContextTextMapPropagator()
        self._config: Optional[TracingConfig] = None

    def initialize(self, config: Optional[TracingConfig] = None) -> None:
        """
        Initialize the tracing system.

        Args:
            config: Tracing configuration. If None, uses defaults.
        """
        if self._tracer_provider is not None:
            return  # Already initialized

        self._config = config or TracingConfig()

        # Create resource with service information
        resource = Resource.create({
            "service.name": self._config.service_name,
            "service.version": self._config.service_version,
            "deployment.environment": self._config.environment,
        })

        # Create tracer provider
        self._tracer_provider = TracerProvider(resource=resource)

        # Add console exporter if enabled
        if self._config.console_export:
            console_exporter = ConsoleSpanExporter()
            self._tracer_provider.add_span_processor(
                BatchSpanProcessor(console_exporter)
            )

        # Add OTLP exporter if endpoint configured and available
        if self._config.otlp_endpoint and OTLP_AVAILABLE:
            # Import is safe here because OTLP_AVAILABLE guards the check
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as OTLPExporter
            otlp_exporter = OTLPExporter(endpoint=self._config.otlp_endpoint)
            self._tracer_provider.add_span_processor(
                BatchSpanProcessor(otlp_exporter)
            )

        # Set as global tracer provider
        trace.set_tracer_provider(self._tracer_provider)

        # Create tracer
        self._tracer = trace.get_tracer(
            self._config.service_name,
            self._config.service_version
        )

    def get_tracer(self) -> trace.Tracer:
        """Get the configured tracer."""
        if self._tracer is None:
            self.initialize()
        # After initialize(), _tracer is guaranteed to be set
        assert self._tracer is not None
        return self._tracer

    def shutdown(self) -> None:
        """Shutdown the tracing system and flush pending spans."""
        if self._tracer_provider:
            self._tracer_provider.shutdown()

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        kind: trace.SpanKind = trace.SpanKind.INTERNAL
    ):
        """
        Context manager for creating a span.

        Args:
            name: Span name
            attributes: Initial span attributes
            kind: Span kind (INTERNAL, CLIENT, SERVER, PRODUCER, CONSUMER)

        Yields:
            The active span
        """
        tracer = self.get_tracer()
        with tracer.start_as_current_span(name, kind=kind) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def inject_context(self, carrier: Dict[str, str]) -> None:
        """Inject trace context into a carrier (e.g., HTTP headers)."""
        self._propagator.inject(carrier)

    def extract_context(self, carrier: Dict[str, str]):
        """Extract trace context from a carrier."""
        return self._propagator.extract(carrier)


def get_tracing_manager() -> TracingManager:
    """Get the singleton tracing manager."""
    return TracingManager()


def get_tracer() -> trace.Tracer:
    """Get the configured tracer."""
    return get_tracing_manager().get_tracer()


def trace_operation(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL
) -> Callable[[F], F]:
    """
    Decorator to trace a function.

    Args:
        name: Span name (defaults to function name)
        attributes: Static span attributes
        kind: Span kind

    Example:
        @trace_operation(attributes={"component": "health_check"})
        async def check_qdrant():
            ...
    """
    def decorator(func: F) -> F:
        span_name = name or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            manager = get_tracing_manager()
            with manager.span(span_name, attributes, kind) as span:
                # Add function arguments as attributes (limited)
                if args:
                    span.set_attribute("args_count", len(args))
                if kwargs:
                    span.set_attribute("kwargs_keys", ",".join(kwargs.keys()))

                result = await func(*args, **kwargs)
                return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            manager = get_tracing_manager()
            with manager.span(span_name, attributes, kind) as span:
                if args:
                    span.set_attribute("args_count", len(args))
                if kwargs:
                    span.set_attribute("kwargs_keys", ",".join(kwargs.keys()))

                result = func(*args, **kwargs)
                return result

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


# Convenience functions for common span types

def trace_health_check(component: str):
    """Decorator for health check operations."""
    return trace_operation(
        name=f"health_check.{component}",
        attributes={"component": component, "operation": "health_check"},
        kind=trace.SpanKind.CLIENT
    )


def trace_swarm_task(task_type: str):
    """Decorator for swarm task operations."""
    return trace_operation(
        name=f"swarm.{task_type}",
        attributes={"task_type": task_type, "operation": "swarm_task"},
        kind=trace.SpanKind.INTERNAL
    )


def trace_knowledge_graph(operation: str):
    """Decorator for knowledge graph operations."""
    return trace_operation(
        name=f"knowledge_graph.{operation}",
        attributes={"operation": operation, "subsystem": "knowledge_graph"},
        kind=trace.SpanKind.CLIENT
    )


# Platform-specific span helpers

class PlatformSpans:
    """Helper class for creating platform-specific spans."""

    @staticmethod
    @contextmanager
    def circuit_breaker_call(component: str, state: str):
        """Trace a circuit breaker protected call."""
        manager = get_tracing_manager()
        with manager.span(
            f"circuit_breaker.{component}",
            attributes={
                "component": component,
                "circuit_state": state,
                "pattern": "circuit_breaker"
            }
        ) as span:
            yield span

    @staticmethod
    @contextmanager
    def bulkhead_acquire(name: str, max_concurrent: int, current: int):
        """Trace a bulkhead acquisition."""
        manager = get_tracing_manager()
        with manager.span(
            f"bulkhead.{name}",
            attributes={
                "bulkhead_name": name,
                "max_concurrent": max_concurrent,
                "current_concurrent": current,
                "pattern": "bulkhead"
            }
        ) as span:
            yield span

    @staticmethod
    @contextmanager
    def retry_attempt(operation: str, attempt: int, max_attempts: int):
        """Trace a retry attempt."""
        manager = get_tracing_manager()
        with manager.span(
            f"retry.{operation}",
            attributes={
                "operation": operation,
                "attempt": attempt,
                "max_attempts": max_attempts,
                "pattern": "retry"
            }
        ) as span:
            yield span


def main():
    """Demo the tracing system."""
    import time

    # Initialize with console output
    config = TracingConfig(
        service_name="uap-tracing-demo",
        console_export=True,
        environment="development"
    )

    manager = get_tracing_manager()
    manager.initialize(config)

    print("[>>] Starting tracing demonstration...")
    print()

    # Demo basic span
    with manager.span("demo_operation", {"demo": True}) as span:
        span.set_attribute("step", 1)
        time.sleep(0.1)
        span.set_attribute("step", 2)
        span.set_attribute("completed", True)

    # Demo nested spans
    with manager.span("parent_operation") as parent:
        parent.set_attribute("level", 0)

        with manager.span("child_operation_1") as child1:
            child1.set_attribute("level", 1)
            time.sleep(0.05)

        with manager.span("child_operation_2") as child2:
            child2.set_attribute("level", 1)
            time.sleep(0.05)

    # Demo error handling
    try:
        with manager.span("error_operation") as span:
            span.set_attribute("will_fail", True)
            raise ValueError("Demo error")
    except ValueError:
        pass  # Expected

    # Demo platform-specific spans
    with PlatformSpans.circuit_breaker_call("qdrant", "CLOSED") as span:
        span.set_attribute("latency_ms", 45.2)
        time.sleep(0.05)

    with PlatformSpans.bulkhead_acquire("health_check", 10, 3) as span:
        span.set_attribute("acquired", True)
        time.sleep(0.05)

    print()
    print("[OK] Tracing demonstration complete")
    print("[>>] In production, spans would be exported to Jaeger/Zipkin/OTLP")

    # Cleanup
    manager.shutdown()


if __name__ == "__main__":
    main()
