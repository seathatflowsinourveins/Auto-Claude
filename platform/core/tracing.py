"""
OpenTelemetry Distributed Tracing - Unleashed Platform
=======================================================

Production-grade distributed tracing with OpenTelemetry compatibility.
Provides trace context propagation, span creation, and multi-format export.

Features:
- TracingManager: Central tracing coordination with singleton pattern
- Trace context propagation: W3C Trace Context standard support
- Span creation for key operations: RAG pipeline, adapters, memory
- Integration with structured logging: Correlation IDs flow through
- Export formats: OTLP, Jaeger, Zipkin

Based on OpenTelemetry Specification: https://opentelemetry.io/docs/specs/

Usage:
    from core.tracing import (
        TracingManager,
        get_tracer,
        traced,
        traced_async,
        inject_context,
        extract_context,
    )

    # Get the global tracer
    tracer = get_tracer(service_name="rag-pipeline")

    # Use as decorator
    @traced("rag.retrieve")
    async def retrieve_documents(query: str):
        ...

    # Use as context manager
    async with tracer.start_span("adapter.exa.search") as span:
        span.set_attribute("query", query)
        span.add_event("search_started")
        result = await exa.search(query)
        span.add_event("search_completed", {"result_count": len(result)})

    # Context propagation for distributed systems
    headers = inject_context()  # Add to outgoing HTTP headers
    context = extract_context(incoming_headers)  # Extract from incoming
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Span Status and Kind
# =============================================================================


class SpanKind(Enum):
    """OpenTelemetry SpanKind values."""

    INTERNAL = "internal"  # Default, internal operation
    SERVER = "server"  # Server-side of RPC
    CLIENT = "client"  # Client-side of RPC
    PRODUCER = "producer"  # Message producer
    CONSUMER = "consumer"  # Message consumer


class SpanStatus(Enum):
    """OpenTelemetry SpanStatus values."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


# =============================================================================
# Trace Context (W3C Trace Context Standard)
# =============================================================================


@dataclass
class TraceContext:
    """W3C Trace Context implementation.

    Format: {version}-{trace_id}-{span_id}-{trace_flags}
    Example: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01
    """

    trace_id: str
    span_id: str
    trace_flags: int = 1  # 1 = sampled
    version: str = "00"

    @classmethod
    def generate(cls) -> "TraceContext":
        """Generate a new trace context."""
        return cls(
            trace_id=uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16],
        )

    @classmethod
    def from_traceparent(cls, traceparent: str) -> Optional["TraceContext"]:
        """Parse W3C traceparent header."""
        if not traceparent:
            return None
        try:
            parts = traceparent.split("-")
            if len(parts) != 4:
                return None
            return cls(
                version=parts[0],
                trace_id=parts[1],
                span_id=parts[2],
                trace_flags=int(parts[3], 16),
            )
        except (ValueError, IndexError):
            return None

    def to_traceparent(self) -> str:
        """Serialize to W3C traceparent header."""
        return f"{self.version}-{self.trace_id}-{self.span_id}-{self.trace_flags:02x}"

    def child_context(self) -> "TraceContext":
        """Create a child context (same trace, new span)."""
        return TraceContext(
            trace_id=self.trace_id,
            span_id=uuid.uuid4().hex[:16],
            trace_flags=self.trace_flags,
            version=self.version,
        )

    @property
    def is_sampled(self) -> bool:
        """Check if trace is sampled."""
        return bool(self.trace_flags & 0x01)


@dataclass
class Baggage:
    """W3C Baggage implementation for context propagation."""

    items: Dict[str, str] = field(default_factory=dict)

    def set(self, key: str, value: str) -> None:
        """Set a baggage item."""
        self.items[key] = value

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a baggage item."""
        return self.items.get(key, default)

    def to_header(self) -> str:
        """Serialize to W3C baggage header."""
        return ",".join(f"{k}={v}" for k, v in self.items.items())

    @classmethod
    def from_header(cls, header: str) -> "Baggage":
        """Parse W3C baggage header."""
        items = {}
        if header:
            for item in header.split(","):
                if "=" in item:
                    key, value = item.strip().split("=", 1)
                    items[key] = value
        return cls(items=items)


# =============================================================================
# Span Event and Link
# =============================================================================


@dataclass
class SpanEvent:
    """An event within a span."""

    name: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "attributes": self.attributes,
        }


@dataclass
class SpanLink:
    """A link to another span (for async operations)."""

    trace_id: str
    span_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "attributes": self.attributes,
        }


# =============================================================================
# Span
# =============================================================================


@dataclass
class Span:
    """A single span in a trace.

    Represents a unit of work with timing, attributes, and relationships.
    """

    name: str
    context: TraceContext
    parent_span_id: Optional[str] = None
    kind: SpanKind = SpanKind.INTERNAL
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    status: SpanStatus = SpanStatus.UNSET
    status_message: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    links: List[SpanLink] = field(default_factory=list)
    service_name: str = "unleashed"
    dropped_attributes_count: int = 0
    dropped_events_count: int = 0
    dropped_links_count: int = 0

    # Limits (OpenTelemetry defaults)
    MAX_ATTRIBUTES = 128
    MAX_EVENTS = 128
    MAX_LINKS = 128
    MAX_ATTRIBUTE_VALUE_LENGTH = 16384

    @property
    def trace_id(self) -> str:
        """Get trace ID."""
        return self.context.trace_id

    @property
    def span_id(self) -> str:
        """Get span ID."""
        return self.context.span_id

    @property
    def duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds() * 1000

    @property
    def duration_ns(self) -> Optional[int]:
        """Get span duration in nanoseconds (OTLP format)."""
        if self.end_time is None:
            return None
        return int((self.end_time - self.start_time).total_seconds() * 1e9)

    def set_attribute(self, key: str, value: Any) -> "Span":
        """Set a span attribute."""
        if len(self.attributes) >= self.MAX_ATTRIBUTES:
            self.dropped_attributes_count += 1
            return self
        # Truncate long string values
        if isinstance(value, str) and len(value) > self.MAX_ATTRIBUTE_VALUE_LENGTH:
            value = value[: self.MAX_ATTRIBUTE_VALUE_LENGTH]
        self.attributes[key] = value
        return self

    def set_attributes(self, attributes: Dict[str, Any]) -> "Span":
        """Set multiple span attributes."""
        for key, value in attributes.items():
            self.set_attribute(key, value)
        return self

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> "Span":
        """Add an event to the span."""
        if len(self.events) >= self.MAX_EVENTS:
            self.dropped_events_count += 1
            return self
        event = SpanEvent(
            name=name,
            timestamp=timestamp or datetime.now(timezone.utc),
            attributes=attributes or {},
        )
        self.events.append(event)
        return self

    def add_link(
        self,
        trace_id: str,
        span_id: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "Span":
        """Add a link to another span."""
        if len(self.links) >= self.MAX_LINKS:
            self.dropped_links_count += 1
            return self
        link = SpanLink(
            trace_id=trace_id,
            span_id=span_id,
            attributes=attributes or {},
        )
        self.links.append(link)
        return self

    def set_status(
        self, status: SpanStatus, message: Optional[str] = None
    ) -> "Span":
        """Set span status."""
        self.status = status
        self.status_message = message
        return self

    def record_exception(
        self,
        exception: BaseException,
        attributes: Optional[Dict[str, Any]] = None,
        escaped: bool = False,
    ) -> "Span":
        """Record an exception as an event."""
        exc_attributes = {
            "exception.type": type(exception).__name__,
            "exception.message": str(exception),
            "exception.escaped": escaped,
        }
        if attributes:
            exc_attributes.update(attributes)
        self.add_event("exception", exc_attributes)
        self.set_status(SpanStatus.ERROR, str(exception))
        return self

    def end(self, end_time: Optional[datetime] = None) -> None:
        """End the span."""
        if self.end_time is None:
            self.end_time = end_time or datetime.now(timezone.utc)

    def is_recording(self) -> bool:
        """Check if span is still recording."""
        return self.end_time is None

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "kind": self.kind.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "events": [e.to_dict() for e in self.events],
            "links": [link.to_dict() for link in self.links],
            "service_name": self.service_name,
        }

    def to_otlp(self) -> Dict[str, Any]:
        """Convert span to OTLP format."""

        def to_otlp_value(value: Any) -> Dict[str, Any]:
            """Convert a value to OTLP attribute format."""
            if isinstance(value, bool):
                return {"boolValue": value}
            elif isinstance(value, int):
                return {"intValue": str(value)}
            elif isinstance(value, float):
                return {"doubleValue": value}
            elif isinstance(value, (list, tuple)):
                return {
                    "arrayValue": {
                        "values": [to_otlp_value(v) for v in value]
                    }
                }
            else:
                return {"stringValue": str(value)}

        def to_otlp_attributes(attrs: Dict[str, Any]) -> List[Dict[str, Any]]:
            """Convert attributes to OTLP format."""
            return [
                {"key": k, "value": to_otlp_value(v)} for k, v in attrs.items()
            ]

        return {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "parentSpanId": self.parent_span_id or "",
            "name": self.name,
            "kind": {
                SpanKind.INTERNAL: 1,
                SpanKind.SERVER: 2,
                SpanKind.CLIENT: 3,
                SpanKind.PRODUCER: 4,
                SpanKind.CONSUMER: 5,
            }.get(self.kind, 1),
            "startTimeUnixNano": str(int(self.start_time.timestamp() * 1e9)),
            "endTimeUnixNano": (
                str(int(self.end_time.timestamp() * 1e9)) if self.end_time else ""
            ),
            "attributes": to_otlp_attributes(self.attributes),
            "events": [
                {
                    "timeUnixNano": str(int(e.timestamp.timestamp() * 1e9)),
                    "name": e.name,
                    "attributes": to_otlp_attributes(e.attributes),
                }
                for e in self.events
            ],
            "links": [
                {
                    "traceId": link.trace_id,
                    "spanId": link.span_id,
                    "attributes": to_otlp_attributes(link.attributes),
                }
                for link in self.links
            ],
            "status": {
                "code": {
                    SpanStatus.UNSET: 0,
                    SpanStatus.OK: 1,
                    SpanStatus.ERROR: 2,
                }.get(self.status, 0),
                "message": self.status_message or "",
            },
            "droppedAttributesCount": self.dropped_attributes_count,
            "droppedEventsCount": self.dropped_events_count,
            "droppedLinksCount": self.dropped_links_count,
        }

    def to_jaeger(self) -> Dict[str, Any]:
        """Convert span to Jaeger format."""
        tags = [
            {"key": k, "type": "string", "value": str(v)}
            for k, v in self.attributes.items()
        ]
        logs = [
            {
                "timestamp": int(e.timestamp.timestamp() * 1e6),
                "fields": [
                    {"key": k, "type": "string", "value": str(v)}
                    for k, v in e.attributes.items()
                ]
                + [{"key": "event", "type": "string", "value": e.name}],
            }
            for e in self.events
        ]
        return {
            "traceID": self.trace_id,
            "spanID": self.span_id,
            "parentSpanID": self.parent_span_id or "",
            "operationName": self.name,
            "references": [
                {
                    "refType": "FOLLOWS_FROM",
                    "traceID": link.trace_id,
                    "spanID": link.span_id,
                }
                for link in self.links
            ],
            "startTime": int(self.start_time.timestamp() * 1e6),
            "duration": int(self.duration_ms * 1000) if self.duration_ms else 0,
            "tags": tags,
            "logs": logs,
            "processID": "p1",
            "warnings": [] if self.status != SpanStatus.ERROR else [self.status_message],
        }

    def to_zipkin(self) -> Dict[str, Any]:
        """Convert span to Zipkin format."""
        kind_map = {
            SpanKind.CLIENT: "CLIENT",
            SpanKind.SERVER: "SERVER",
            SpanKind.PRODUCER: "PRODUCER",
            SpanKind.CONSUMER: "CONSUMER",
        }
        return {
            "traceId": self.trace_id,
            "id": self.span_id,
            "parentId": self.parent_span_id,
            "name": self.name,
            "kind": kind_map.get(self.kind),
            "timestamp": int(self.start_time.timestamp() * 1e6),
            "duration": int(self.duration_ms * 1000) if self.duration_ms else None,
            "localEndpoint": {"serviceName": self.service_name},
            "tags": {k: str(v) for k, v in self.attributes.items()},
            "annotations": [
                {
                    "timestamp": int(e.timestamp.timestamp() * 1e6),
                    "value": e.name,
                }
                for e in self.events
            ],
        }


# =============================================================================
# Span Processor (Base Class)
# =============================================================================


class SpanProcessor(ABC):
    """Base class for span processors."""

    @abstractmethod
    def on_start(self, span: Span, parent_context: Optional[TraceContext]) -> None:
        """Called when a span starts."""
        pass

    @abstractmethod
    def on_end(self, span: Span) -> None:
        """Called when a span ends."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the processor."""
        pass

    @abstractmethod
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any buffered spans."""
        pass


class SimpleSpanProcessor(SpanProcessor):
    """Synchronous span processor that exports immediately."""

    def __init__(self, exporter: "SpanExporter"):
        self._exporter = exporter

    def on_start(self, span: Span, parent_context: Optional[TraceContext]) -> None:
        pass

    def on_end(self, span: Span) -> None:
        self._exporter.export([span])

    def shutdown(self) -> None:
        self._exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class BatchSpanProcessor(SpanProcessor):
    """Batching span processor for efficient export."""

    def __init__(
        self,
        exporter: "SpanExporter",
        max_queue_size: int = 2048,
        max_export_batch_size: int = 512,
        export_timeout_millis: int = 30000,
        schedule_delay_millis: int = 5000,
    ):
        self._exporter = exporter
        self._max_queue_size = max_queue_size
        self._max_export_batch_size = max_export_batch_size
        self._export_timeout_millis = export_timeout_millis
        self._schedule_delay_millis = schedule_delay_millis

        self._queue: List[Span] = []
        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._export_thread = threading.Thread(target=self._export_loop, daemon=True)
        self._export_thread.start()

    def on_start(self, span: Span, parent_context: Optional[TraceContext]) -> None:
        pass

    def on_end(self, span: Span) -> None:
        with self._lock:
            if len(self._queue) < self._max_queue_size:
                self._queue.append(span)
            else:
                logger.warning("Span queue full, dropping span")

    def _export_loop(self) -> None:
        """Background export loop."""
        while not self._shutdown.is_set():
            self._shutdown.wait(self._schedule_delay_millis / 1000)
            self._do_export()

    def _do_export(self) -> None:
        """Export batched spans."""
        with self._lock:
            if not self._queue:
                return
            batch = self._queue[: self._max_export_batch_size]
            self._queue = self._queue[self._max_export_batch_size :]

        if batch:
            try:
                self._exporter.export(batch)
            except Exception as e:
                logger.error(f"Failed to export spans: {e}")

    def shutdown(self) -> None:
        self._shutdown.set()
        self._do_export()  # Final export
        self._exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        self._do_export()
        return True


# =============================================================================
# Span Exporter (Base Class and Implementations)
# =============================================================================


class SpanExporter(ABC):
    """Base class for span exporters."""

    @abstractmethod
    def export(self, spans: List[Span]) -> bool:
        """Export spans. Returns True on success."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass


class ConsoleSpanExporter(SpanExporter):
    """Export spans to console (for development)."""

    def __init__(self, pretty: bool = True):
        self._pretty = pretty

    def export(self, spans: List[Span]) -> bool:
        for span in spans:
            data = span.to_dict()
            if self._pretty:
                print(json.dumps(data, indent=2, default=str))
            else:
                print(json.dumps(data, default=str))
        return True

    def shutdown(self) -> None:
        pass


class InMemorySpanExporter(SpanExporter):
    """Export spans to memory (for testing)."""

    def __init__(self, max_spans: int = 10000):
        self._spans: List[Span] = []
        self._max_spans = max_spans
        self._lock = threading.Lock()

    def export(self, spans: List[Span]) -> bool:
        with self._lock:
            self._spans.extend(spans)
            if len(self._spans) > self._max_spans:
                self._spans = self._spans[-self._max_spans :]
        return True

    def get_spans(self) -> List[Span]:
        """Get all exported spans."""
        with self._lock:
            return list(self._spans)

    def clear(self) -> None:
        """Clear exported spans."""
        with self._lock:
            self._spans.clear()

    def shutdown(self) -> None:
        pass


class OTLPSpanExporter(SpanExporter):
    """Export spans in OTLP format via HTTP."""

    def __init__(
        self,
        endpoint: str = "http://localhost:4318/v1/traces",
        headers: Optional[Dict[str, str]] = None,
        timeout_seconds: int = 10,
    ):
        self._endpoint = endpoint
        self._headers = headers or {}
        self._headers.setdefault("Content-Type", "application/json")
        self._timeout = timeout_seconds

        # Try to import httpx for async HTTP
        try:
            import httpx

            self._client = httpx.Client(timeout=self._timeout)
            self._http_available = True
        except ImportError:
            self._client = None
            self._http_available = False
            logger.warning("httpx not available, OTLP export will be skipped")

    def export(self, spans: List[Span]) -> bool:
        if not self._http_available or not spans:
            return True

        # Build OTLP payload
        payload = {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {
                                "key": "service.name",
                                "value": {"stringValue": spans[0].service_name},
                            }
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "unleashed.tracing", "version": "1.0.0"},
                            "spans": [span.to_otlp() for span in spans],
                        }
                    ],
                }
            ]
        }

        try:
            response = self._client.post(
                self._endpoint,
                json=payload,
                headers=self._headers,
            )
            return response.status_code in (200, 202)
        except Exception as e:
            logger.error(f"OTLP export failed: {e}")
            return False

    def shutdown(self) -> None:
        if self._client:
            self._client.close()


class JaegerSpanExporter(SpanExporter):
    """Export spans to Jaeger via HTTP."""

    def __init__(
        self,
        endpoint: str = "http://localhost:14268/api/traces",
        service_name: str = "unleashed",
        timeout_seconds: int = 10,
    ):
        self._endpoint = endpoint
        self._service_name = service_name
        self._timeout = timeout_seconds

        try:
            import httpx

            self._client = httpx.Client(timeout=self._timeout)
            self._http_available = True
        except ImportError:
            self._client = None
            self._http_available = False

    def export(self, spans: List[Span]) -> bool:
        if not self._http_available or not spans:
            return True

        # Build Jaeger payload
        payload = {
            "data": [
                {
                    "traceID": spans[0].trace_id,
                    "spans": [span.to_jaeger() for span in spans],
                    "processes": {
                        "p1": {
                            "serviceName": self._service_name,
                            "tags": [],
                        }
                    },
                }
            ]
        }

        try:
            response = self._client.post(
                self._endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            return response.status_code in (200, 202)
        except Exception as e:
            logger.error(f"Jaeger export failed: {e}")
            return False

    def shutdown(self) -> None:
        if self._client:
            self._client.close()


class ZipkinSpanExporter(SpanExporter):
    """Export spans to Zipkin via HTTP."""

    def __init__(
        self,
        endpoint: str = "http://localhost:9411/api/v2/spans",
        timeout_seconds: int = 10,
    ):
        self._endpoint = endpoint
        self._timeout = timeout_seconds

        try:
            import httpx

            self._client = httpx.Client(timeout=self._timeout)
            self._http_available = True
        except ImportError:
            self._client = None
            self._http_available = False

    def export(self, spans: List[Span]) -> bool:
        if not self._http_available or not spans:
            return True

        payload = [span.to_zipkin() for span in spans]

        try:
            response = self._client.post(
                self._endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            return response.status_code in (200, 202)
        except Exception as e:
            logger.error(f"Zipkin export failed: {e}")
            return False

    def shutdown(self) -> None:
        if self._client:
            self._client.close()


# =============================================================================
# Sampler
# =============================================================================


class Sampler(ABC):
    """Base class for sampling strategies."""

    @abstractmethod
    def should_sample(
        self,
        trace_id: str,
        name: str,
        kind: SpanKind,
        attributes: Dict[str, Any],
        parent_context: Optional[TraceContext],
    ) -> Tuple[bool, Dict[str, Any]]:
        """Determine if a span should be sampled.

        Returns: (should_sample, attributes_to_add)
        """
        pass


class AlwaysOnSampler(Sampler):
    """Sample all traces."""

    def should_sample(
        self,
        trace_id: str,
        name: str,
        kind: SpanKind,
        attributes: Dict[str, Any],
        parent_context: Optional[TraceContext],
    ) -> Tuple[bool, Dict[str, Any]]:
        return True, {}


class AlwaysOffSampler(Sampler):
    """Sample no traces."""

    def should_sample(
        self,
        trace_id: str,
        name: str,
        kind: SpanKind,
        attributes: Dict[str, Any],
        parent_context: Optional[TraceContext],
    ) -> Tuple[bool, Dict[str, Any]]:
        return False, {}


class TraceIdRatioSampler(Sampler):
    """Sample traces based on trace ID ratio."""

    def __init__(self, ratio: float = 1.0):
        self._ratio = max(0.0, min(1.0, ratio))
        self._upper_bound = int(self._ratio * (2**64 - 1))

    def should_sample(
        self,
        trace_id: str,
        name: str,
        kind: SpanKind,
        attributes: Dict[str, Any],
        parent_context: Optional[TraceContext],
    ) -> Tuple[bool, Dict[str, Any]]:
        # Use first 16 hex chars of trace_id
        trace_id_int = int(trace_id[:16], 16)
        return trace_id_int < self._upper_bound, {}


class ParentBasedSampler(Sampler):
    """Sample based on parent decision."""

    def __init__(
        self,
        root: Sampler,
        remote_parent_sampled: Optional[Sampler] = None,
        remote_parent_not_sampled: Optional[Sampler] = None,
        local_parent_sampled: Optional[Sampler] = None,
        local_parent_not_sampled: Optional[Sampler] = None,
    ):
        self._root = root
        self._remote_parent_sampled = remote_parent_sampled or AlwaysOnSampler()
        self._remote_parent_not_sampled = (
            remote_parent_not_sampled or AlwaysOffSampler()
        )
        self._local_parent_sampled = local_parent_sampled or AlwaysOnSampler()
        self._local_parent_not_sampled = (
            local_parent_not_sampled or AlwaysOffSampler()
        )

    def should_sample(
        self,
        trace_id: str,
        name: str,
        kind: SpanKind,
        attributes: Dict[str, Any],
        parent_context: Optional[TraceContext],
    ) -> Tuple[bool, Dict[str, Any]]:
        if parent_context is None:
            return self._root.should_sample(
                trace_id, name, kind, attributes, parent_context
            )

        if parent_context.is_sampled:
            return self._local_parent_sampled.should_sample(
                trace_id, name, kind, attributes, parent_context
            )
        else:
            return self._local_parent_not_sampled.should_sample(
                trace_id, name, kind, attributes, parent_context
            )


# =============================================================================
# Tracer
# =============================================================================


class Tracer:
    """OpenTelemetry-compatible tracer."""

    def __init__(
        self,
        service_name: str = "unleashed",
        sampler: Optional[Sampler] = None,
        processors: Optional[List[SpanProcessor]] = None,
    ):
        self.service_name = service_name
        self._sampler = sampler or AlwaysOnSampler()
        self._processors = processors or []
        self._context_stack: Dict[int, List[TraceContext]] = defaultdict(list)
        self._lock = threading.Lock()

    def add_processor(self, processor: SpanProcessor) -> None:
        """Add a span processor."""
        self._processors.append(processor)

    def get_current_span(self) -> Optional[Span]:
        """Get the current active span for this thread."""
        thread_id = threading.get_ident()
        with self._lock:
            stack = self._context_stack.get(thread_id, [])
            if stack:
                # Return a minimal span representation
                ctx = stack[-1]
                return Span(
                    name="current",
                    context=ctx,
                    service_name=self.service_name,
                )
        return None

    def get_current_context(self) -> Optional[TraceContext]:
        """Get the current trace context for this thread."""
        thread_id = threading.get_ident()
        with self._lock:
            stack = self._context_stack.get(thread_id, [])
            return stack[-1] if stack else None

    def _push_context(self, context: TraceContext) -> None:
        """Push context to stack."""
        thread_id = threading.get_ident()
        with self._lock:
            self._context_stack[thread_id].append(context)

    def _pop_context(self) -> Optional[TraceContext]:
        """Pop context from stack."""
        thread_id = threading.get_ident()
        with self._lock:
            stack = self._context_stack.get(thread_id, [])
            return stack.pop() if stack else None

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        links: Optional[List[SpanLink]] = None,
        parent: Optional[TraceContext] = None,
    ) -> Span:
        """Start a new span."""
        parent_context = parent or self.get_current_context()

        # Create trace context
        if parent_context:
            context = parent_context.child_context()
            parent_span_id = parent_context.span_id
        else:
            context = TraceContext.generate()
            parent_span_id = None

        # Check sampling
        should_sample, sample_attrs = self._sampler.should_sample(
            context.trace_id, name, kind, attributes or {}, parent_context
        )
        if not should_sample:
            context.trace_flags = 0

        # Create span
        span = Span(
            name=name,
            context=context,
            parent_span_id=parent_span_id,
            kind=kind,
            service_name=self.service_name,
        )

        # Add attributes
        if attributes:
            span.set_attributes(attributes)
        if sample_attrs:
            span.set_attributes(sample_attrs)
        if links:
            for link in links:
                span.links.append(link)

        # Push to context stack
        self._push_context(context)

        # Notify processors
        for processor in self._processors:
            try:
                processor.on_start(span, parent_context)
            except Exception as e:
                logger.error(f"Processor on_start failed: {e}")

        return span

    def end_span(self, span: Span) -> None:
        """End a span."""
        span.end()
        self._pop_context()

        # Notify processors
        for processor in self._processors:
            try:
                processor.on_end(span)
            except Exception as e:
                logger.error(f"Processor on_end failed: {e}")

    @contextmanager
    def trace(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Span, None, None]:
        """Context manager for tracing operations."""
        span = self.start_span(name, kind, attributes)
        try:
            yield span
            if span.status == SpanStatus.UNSET:
                span.set_status(SpanStatus.OK)
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            self.end_span(span)

    @asynccontextmanager
    async def atrace(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """Async context manager for tracing operations."""
        span = self.start_span(name, kind, attributes)
        try:
            yield span
            if span.status == SpanStatus.UNSET:
                span.set_status(SpanStatus.OK)
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            self.end_span(span)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush all processors to export buffered spans."""
        for processor in self._processors:
            try:
                processor.force_flush(timeout_millis)
            except Exception as e:
                logger.error(f"Processor flush failed: {e}")
                return False
        return True

    def shutdown(self) -> None:
        """Shutdown all processors."""
        for processor in self._processors:
            try:
                processor.shutdown()
            except Exception as e:
                logger.error(f"Processor shutdown failed: {e}")


# =============================================================================
# Tracing Manager (Singleton)
# =============================================================================


@dataclass
class TracingConfig:
    """Configuration for the tracing manager."""

    service_name: str = "unleashed"
    enabled: bool = True
    sample_rate: float = 1.0  # 1.0 = 100%

    # Export configuration
    export_otlp: bool = False
    otlp_endpoint: str = "http://localhost:4318/v1/traces"
    otlp_headers: Optional[Dict[str, str]] = None

    export_jaeger: bool = False
    jaeger_endpoint: str = "http://localhost:14268/api/traces"

    export_zipkin: bool = False
    zipkin_endpoint: str = "http://localhost:9411/api/v2/spans"

    export_console: bool = False  # For development
    export_memory: bool = True  # For testing/introspection

    # Batch processor configuration
    use_batch_processor: bool = True
    batch_max_queue_size: int = 2048
    batch_max_export_size: int = 512
    batch_schedule_delay_ms: int = 5000


class TracingManager:
    """Central manager for distributed tracing.

    Singleton pattern ensures consistent tracing across the platform.
    """

    _instance: Optional["TracingManager"] = None
    _lock = threading.Lock()

    def __new__(cls, config: Optional[TracingConfig] = None) -> "TracingManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, config: Optional[TracingConfig] = None):
        if self._initialized:
            return

        self.config = config or TracingConfig()
        self._tracer: Optional[Tracer] = None
        self._memory_exporter: Optional[InMemorySpanExporter] = None
        self._setup_tracer()
        self._initialized = True

    def _setup_tracer(self) -> None:
        """Setup tracer with configured exporters."""
        if not self.config.enabled:
            return

        # Create sampler
        if self.config.sample_rate >= 1.0:
            sampler = AlwaysOnSampler()
        elif self.config.sample_rate <= 0.0:
            sampler = AlwaysOffSampler()
        else:
            sampler = ParentBasedSampler(
                root=TraceIdRatioSampler(self.config.sample_rate)
            )

        # Create tracer
        self._tracer = Tracer(
            service_name=self.config.service_name,
            sampler=sampler,
        )

        # Create exporters and processors
        exporters: List[SpanExporter] = []

        if self.config.export_memory:
            self._memory_exporter = InMemorySpanExporter()
            exporters.append(self._memory_exporter)

        if self.config.export_console:
            exporters.append(ConsoleSpanExporter())

        if self.config.export_otlp:
            exporters.append(
                OTLPSpanExporter(
                    endpoint=self.config.otlp_endpoint,
                    headers=self.config.otlp_headers,
                )
            )

        if self.config.export_jaeger:
            exporters.append(
                JaegerSpanExporter(
                    endpoint=self.config.jaeger_endpoint,
                    service_name=self.config.service_name,
                )
            )

        if self.config.export_zipkin:
            exporters.append(ZipkinSpanExporter(endpoint=self.config.zipkin_endpoint))

        # Add processors
        for exporter in exporters:
            if self.config.use_batch_processor:
                processor = BatchSpanProcessor(
                    exporter=exporter,
                    max_queue_size=self.config.batch_max_queue_size,
                    max_export_batch_size=self.config.batch_max_export_size,
                    schedule_delay_millis=self.config.batch_schedule_delay_ms,
                )
            else:
                processor = SimpleSpanProcessor(exporter)
            self._tracer.add_processor(processor)

    @property
    def tracer(self) -> Optional[Tracer]:
        """Get the tracer instance."""
        return self._tracer

    def get_tracer(self, name: Optional[str] = None) -> Tracer:
        """Get a tracer, creating if necessary."""
        if self._tracer is None:
            self._setup_tracer()
        return self._tracer

    def get_current_context(self) -> Optional[TraceContext]:
        """Get current trace context."""
        if self._tracer:
            return self._tracer.get_current_context()
        return None

    def get_spans(self, limit: int = 100) -> List[Span]:
        """Get exported spans from memory."""
        # Force flush any batch processors to ensure spans are exported
        if self._tracer:
            self._tracer.force_flush()
        if self._memory_exporter:
            spans = self._memory_exporter.get_spans()
            return spans[-limit:]
        return []

    def get_traces(
        self,
        trace_id: Optional[str] = None,
        operation_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get traces with optional filtering."""
        spans = self.get_spans(limit * 10)  # Get more, then filter

        if trace_id:
            spans = [s for s in spans if s.trace_id == trace_id]
        if operation_name:
            spans = [s for s in spans if s.name == operation_name]

        # Group by trace_id
        traces: Dict[str, List[Span]] = defaultdict(list)
        for span in spans:
            traces[span.trace_id].append(span)

        result = []
        for tid, trace_spans in list(traces.items())[:limit]:
            result.append(
                {
                    "trace_id": tid,
                    "spans": [s.to_dict() for s in trace_spans],
                    "duration_ms": sum(
                        s.duration_ms or 0 for s in trace_spans if not s.parent_span_id
                    ),
                }
            )

        return result

    def clear_spans(self) -> None:
        """Clear exported spans."""
        if self._memory_exporter:
            self._memory_exporter.clear()

    def shutdown(self) -> None:
        """Shutdown the tracing manager."""
        if self._tracer:
            self._tracer.shutdown()

    def export_otlp(self, spans: Optional[List[Span]] = None) -> List[Dict[str, Any]]:
        """Export spans in OTLP format."""
        if spans is None:
            spans = self.get_spans()
        return [span.to_otlp() for span in spans]

    def export_jaeger(self, spans: Optional[List[Span]] = None) -> List[Dict[str, Any]]:
        """Export spans in Jaeger format."""
        if spans is None:
            spans = self.get_spans()
        return [span.to_jaeger() for span in spans]

    def export_zipkin(self, spans: Optional[List[Span]] = None) -> List[Dict[str, Any]]:
        """Export spans in Zipkin format."""
        if spans is None:
            spans = self.get_spans()
        return [span.to_zipkin() for span in spans]


# =============================================================================
# Context Propagation
# =============================================================================


def inject_context(
    headers: Optional[Dict[str, str]] = None,
    baggage: Optional[Baggage] = None,
) -> Dict[str, str]:
    """Inject trace context into headers for outgoing requests.

    Returns headers dict with traceparent and optionally tracestate/baggage.
    """
    headers = headers or {}
    manager = get_tracing_manager()

    context = manager.get_current_context()
    if context:
        headers["traceparent"] = context.to_traceparent()

    if baggage:
        headers["baggage"] = baggage.to_header()

    return headers


def extract_context(headers: Dict[str, str]) -> Optional[TraceContext]:
    """Extract trace context from incoming request headers."""
    traceparent = headers.get("traceparent") or headers.get("Traceparent")
    return TraceContext.from_traceparent(traceparent)


def extract_baggage(headers: Dict[str, str]) -> Baggage:
    """Extract baggage from incoming request headers."""
    baggage_header = headers.get("baggage") or headers.get("Baggage") or ""
    return Baggage.from_header(baggage_header)


# =============================================================================
# Decorators
# =============================================================================


def traced(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
):
    """Decorator for tracing synchronous functions."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        span_name = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            manager = get_tracing_manager()
            tracer = manager.get_tracer()
            if tracer is None:
                return func(*args, **kwargs)

            with tracer.trace(span_name, kind, attributes) as span:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


def traced_async(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None,
):
    """Decorator for tracing async functions."""

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        span_name = name or func.__name__

        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            manager = get_tracing_manager()
            tracer = manager.get_tracer()
            if tracer is None:
                return await func(*args, **kwargs)

            async with tracer.atrace(span_name, kind, attributes) as span:
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


# =============================================================================
# Integration Helpers
# =============================================================================


class TracedOperation:
    """Context manager for tracing operations with logging integration."""

    def __init__(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        log_entry: bool = True,
        log_exit: bool = True,
    ):
        self.name = name
        self.kind = kind
        self.attributes = attributes or {}
        self.log_entry = log_entry
        self.log_exit = log_exit
        self._span: Optional[Span] = None
        self._tracer: Optional[Tracer] = None
        self._start_time: float = 0

    def __enter__(self) -> Span:
        manager = get_tracing_manager()
        self._tracer = manager.get_tracer()

        if self._tracer:
            self._span = self._tracer.start_span(self.name, self.kind, self.attributes)

            if self.log_entry:
                logger.debug(
                    f"Starting {self.name}",
                    extra={
                        "trace_id": self._span.trace_id,
                        "span_id": self._span.span_id,
                        "operation": self.name,
                    },
                )
        else:
            # Create a dummy span for consistent API
            self._span = Span(
                name=self.name,
                context=TraceContext.generate(),
                kind=self.kind,
            )

        self._start_time = time.perf_counter()
        return self._span

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self._start_time) * 1000

        if self._span:
            if exc_val:
                self._span.record_exception(exc_val)
            elif self._span.status == SpanStatus.UNSET:
                self._span.set_status(SpanStatus.OK)

            if self._tracer:
                self._tracer.end_span(self._span)

            if self.log_exit:
                log_extra = {
                    "trace_id": self._span.trace_id,
                    "span_id": self._span.span_id,
                    "operation": self.name,
                    "duration_ms": duration_ms,
                    "status": self._span.status.value,
                }
                if exc_val:
                    logger.error(f"Failed {self.name}", extra=log_extra)
                else:
                    logger.debug(f"Completed {self.name}", extra=log_extra)

        return False  # Don't suppress exceptions

    async def __aenter__(self) -> Span:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)


# =============================================================================
# RAG Pipeline Tracing Helpers
# =============================================================================


def trace_rag_retrieve(query: str, attributes: Optional[Dict[str, Any]] = None):
    """Create a traced operation for RAG retrieval."""
    attrs = {"rag.query": query[:500], "rag.stage": "retrieve"}
    if attributes:
        attrs.update(attributes)
    return TracedOperation("rag.retrieve", SpanKind.INTERNAL, attrs)


def trace_rag_rerank(
    query: str, num_documents: int, attributes: Optional[Dict[str, Any]] = None
):
    """Create a traced operation for RAG reranking."""
    attrs = {
        "rag.query": query[:500],
        "rag.stage": "rerank",
        "rag.num_documents": num_documents,
    }
    if attributes:
        attrs.update(attributes)
    return TracedOperation("rag.rerank", SpanKind.INTERNAL, attrs)


def trace_rag_generate(query: str, attributes: Optional[Dict[str, Any]] = None):
    """Create a traced operation for RAG generation."""
    attrs = {"rag.query": query[:500], "rag.stage": "generate"}
    if attributes:
        attrs.update(attributes)
    return TracedOperation("rag.generate", SpanKind.INTERNAL, attrs)


def trace_adapter_call(
    adapter_name: str, operation: str, attributes: Optional[Dict[str, Any]] = None
):
    """Create a traced operation for adapter calls."""
    attrs = {"adapter.name": adapter_name, "adapter.operation": operation}
    if attributes:
        attrs.update(attributes)
    return TracedOperation(f"adapter.{adapter_name}.{operation}", SpanKind.CLIENT, attrs)


def trace_memory_operation(
    operation: str, tier: str, attributes: Optional[Dict[str, Any]] = None
):
    """Create a traced operation for memory operations."""
    attrs = {"memory.operation": operation, "memory.tier": tier}
    if attributes:
        attrs.update(attributes)
    return TracedOperation(f"memory.{operation}", SpanKind.INTERNAL, attrs)


# =============================================================================
# Global Instance Management
# =============================================================================

_tracing_manager: Optional[TracingManager] = None
_manager_lock = threading.Lock()


def get_tracing_manager(config: Optional[TracingConfig] = None) -> TracingManager:
    """Get the global tracing manager instance."""
    global _tracing_manager
    with _manager_lock:
        if _tracing_manager is None:
            _tracing_manager = TracingManager(config)
        return _tracing_manager


def get_tracer(
    service_name: Optional[str] = None,
    config: Optional[TracingConfig] = None,
) -> Tracer:
    """Get a tracer instance."""
    if config is None and service_name:
        config = TracingConfig(service_name=service_name)
    manager = get_tracing_manager(config)
    return manager.get_tracer()


def configure_tracing(config: TracingConfig) -> TracingManager:
    """Configure and return the global tracing manager."""
    global _tracing_manager
    with _manager_lock:
        if _tracing_manager:
            _tracing_manager.shutdown()
        _tracing_manager = TracingManager(config)
        return _tracing_manager


def reset_tracing() -> None:
    """Reset the global tracing manager (for testing)."""
    global _tracing_manager
    with _manager_lock:
        if _tracing_manager:
            _tracing_manager.shutdown()
        _tracing_manager = None
        # Reset the singleton class instance so next creation starts fresh
        TracingManager._instance = None


# =============================================================================
# Quick Setup Functions
# =============================================================================


def setup_development_tracing(service_name: str = "unleashed-dev") -> TracingManager:
    """Quick setup for development environment."""
    config = TracingConfig(
        service_name=service_name,
        enabled=True,
        sample_rate=1.0,
        export_console=True,
        export_memory=True,
        use_batch_processor=False,  # Immediate export for dev
    )
    return configure_tracing(config)


def setup_production_tracing(
    service_name: str = "unleashed",
    otlp_endpoint: Optional[str] = None,
    jaeger_endpoint: Optional[str] = None,
    zipkin_endpoint: Optional[str] = None,
    sample_rate: float = 0.1,
) -> TracingManager:
    """Quick setup for production environment."""
    config = TracingConfig(
        service_name=service_name,
        enabled=True,
        sample_rate=sample_rate,
        export_console=False,
        export_memory=True,
        export_otlp=otlp_endpoint is not None,
        otlp_endpoint=otlp_endpoint or "http://localhost:4318/v1/traces",
        export_jaeger=jaeger_endpoint is not None,
        jaeger_endpoint=jaeger_endpoint or "http://localhost:14268/api/traces",
        export_zipkin=zipkin_endpoint is not None,
        zipkin_endpoint=zipkin_endpoint or "http://localhost:9411/api/v2/spans",
        use_batch_processor=True,
    )
    return configure_tracing(config)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Core types
    "SpanKind",
    "SpanStatus",
    "TraceContext",
    "Baggage",
    "SpanEvent",
    "SpanLink",
    "Span",
    # Processors
    "SpanProcessor",
    "SimpleSpanProcessor",
    "BatchSpanProcessor",
    # Exporters
    "SpanExporter",
    "ConsoleSpanExporter",
    "InMemorySpanExporter",
    "OTLPSpanExporter",
    "JaegerSpanExporter",
    "ZipkinSpanExporter",
    # Samplers
    "Sampler",
    "AlwaysOnSampler",
    "AlwaysOffSampler",
    "TraceIdRatioSampler",
    "ParentBasedSampler",
    # Tracer
    "Tracer",
    # Manager
    "TracingConfig",
    "TracingManager",
    # Context propagation
    "inject_context",
    "extract_context",
    "extract_baggage",
    # Decorators
    "traced",
    "traced_async",
    # Integration helpers
    "TracedOperation",
    "trace_rag_retrieve",
    "trace_rag_rerank",
    "trace_rag_generate",
    "trace_adapter_call",
    "trace_memory_operation",
    # Global functions
    "get_tracing_manager",
    "get_tracer",
    "configure_tracing",
    "reset_tracing",
    # Quick setup
    "setup_development_tracing",
    "setup_production_tracing",
]
