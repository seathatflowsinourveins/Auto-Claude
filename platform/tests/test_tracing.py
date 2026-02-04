"""
Tests for OpenTelemetry Distributed Tracing Module.

Verifies:
- TracingManager singleton and configuration
- Span creation and lifecycle
- Context propagation (W3C Trace Context)
- Export formats: OTLP, Jaeger, Zipkin
- Samplers
- Decorators and integration helpers
"""

import asyncio
import json
import threading
import time
import unittest
from datetime import datetime, timezone
from typing import Dict, List, Any

# Import tracing module
from core.tracing import (
    # Core types
    SpanKind,
    SpanStatus,
    TraceContext,
    Baggage,
    SpanEvent,
    SpanLink,
    Span,
    # Processors
    SpanProcessor,
    SimpleSpanProcessor,
    BatchSpanProcessor,
    # Exporters
    SpanExporter,
    ConsoleSpanExporter,
    InMemorySpanExporter,
    OTLPSpanExporter,
    JaegerSpanExporter,
    ZipkinSpanExporter,
    # Samplers
    Sampler,
    AlwaysOnSampler,
    AlwaysOffSampler,
    TraceIdRatioSampler,
    ParentBasedSampler,
    # Tracer
    Tracer,
    # Manager
    TracingConfig,
    TracingManager,
    # Context propagation
    inject_context,
    extract_context,
    extract_baggage,
    # Decorators
    traced,
    traced_async,
    # Integration helpers
    TracedOperation,
    trace_rag_retrieve,
    trace_rag_rerank,
    trace_rag_generate,
    trace_adapter_call,
    trace_memory_operation,
    # Global functions
    get_tracing_manager,
    get_tracer,
    configure_tracing,
    reset_tracing,
    # Quick setup
    setup_development_tracing,
    setup_production_tracing,
)


class TestTraceContext(unittest.TestCase):
    """Test TraceContext W3C implementation."""

    def test_generate(self):
        """Test context generation."""
        ctx = TraceContext.generate()
        self.assertEqual(len(ctx.trace_id), 32)
        self.assertEqual(len(ctx.span_id), 16)
        self.assertEqual(ctx.version, "00")
        self.assertTrue(ctx.is_sampled)

    def test_to_traceparent(self):
        """Test traceparent header serialization."""
        ctx = TraceContext(
            trace_id="0af7651916cd43dd8448eb211c80319c",
            span_id="b7ad6b7169203331",
            trace_flags=1,
        )
        traceparent = ctx.to_traceparent()
        self.assertEqual(traceparent, "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01")

    def test_from_traceparent(self):
        """Test traceparent header parsing."""
        traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        ctx = TraceContext.from_traceparent(traceparent)
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx.trace_id, "0af7651916cd43dd8448eb211c80319c")
        self.assertEqual(ctx.span_id, "b7ad6b7169203331")
        self.assertEqual(ctx.trace_flags, 1)
        self.assertTrue(ctx.is_sampled)

    def test_from_traceparent_invalid(self):
        """Test invalid traceparent handling."""
        self.assertIsNone(TraceContext.from_traceparent("invalid"))
        self.assertIsNone(TraceContext.from_traceparent(""))
        self.assertIsNone(TraceContext.from_traceparent(None))
        self.assertIsNone(TraceContext.from_traceparent("00-invalid"))

    def test_child_context(self):
        """Test child context creation."""
        parent = TraceContext.generate()
        child = parent.child_context()

        # Same trace ID
        self.assertEqual(child.trace_id, parent.trace_id)
        # Different span ID
        self.assertNotEqual(child.span_id, parent.span_id)
        # Same flags
        self.assertEqual(child.trace_flags, parent.trace_flags)


class TestBaggage(unittest.TestCase):
    """Test W3C Baggage implementation."""

    def test_set_and_get(self):
        """Test baggage item operations."""
        baggage = Baggage()
        baggage.set("user_id", "123")
        baggage.set("session", "abc")

        self.assertEqual(baggage.get("user_id"), "123")
        self.assertEqual(baggage.get("session"), "abc")
        self.assertIsNone(baggage.get("missing"))
        self.assertEqual(baggage.get("missing", "default"), "default")

    def test_to_header(self):
        """Test baggage header serialization."""
        baggage = Baggage(items={"user_id": "123", "session": "abc"})
        header = baggage.to_header()
        self.assertIn("user_id=123", header)
        self.assertIn("session=abc", header)

    def test_from_header(self):
        """Test baggage header parsing."""
        baggage = Baggage.from_header("user_id=123,session=abc")
        self.assertEqual(baggage.get("user_id"), "123")
        self.assertEqual(baggage.get("session"), "abc")


class TestSpan(unittest.TestCase):
    """Test Span functionality."""

    def setUp(self):
        """Set up test span."""
        self.ctx = TraceContext.generate()
        self.span = Span(
            name="test-operation",
            context=self.ctx,
            service_name="test-service",
        )

    def test_span_creation(self):
        """Test span creation."""
        self.assertEqual(self.span.name, "test-operation")
        self.assertEqual(self.span.service_name, "test-service")
        self.assertEqual(self.span.trace_id, self.ctx.trace_id)
        self.assertEqual(self.span.span_id, self.ctx.span_id)
        self.assertTrue(self.span.is_recording())
        self.assertEqual(self.span.status, SpanStatus.UNSET)

    def test_set_attributes(self):
        """Test attribute setting."""
        self.span.set_attribute("key1", "value1")
        self.span.set_attribute("key2", 42)
        self.span.set_attribute("key3", True)

        self.assertEqual(self.span.attributes["key1"], "value1")
        self.assertEqual(self.span.attributes["key2"], 42)
        self.assertEqual(self.span.attributes["key3"], True)

    def test_set_attributes_bulk(self):
        """Test bulk attribute setting."""
        self.span.set_attributes({
            "a": 1,
            "b": 2,
            "c": 3,
        })
        self.assertEqual(len(self.span.attributes), 3)

    def test_attribute_limits(self):
        """Test attribute limits."""
        # Add max attributes
        for i in range(Span.MAX_ATTRIBUTES + 10):
            self.span.set_attribute(f"key_{i}", f"value_{i}")

        self.assertEqual(len(self.span.attributes), Span.MAX_ATTRIBUTES)
        self.assertGreater(self.span.dropped_attributes_count, 0)

    def test_add_event(self):
        """Test event addition."""
        self.span.add_event("started", {"detail": "test"})
        self.span.add_event("completed")

        self.assertEqual(len(self.span.events), 2)
        self.assertEqual(self.span.events[0].name, "started")
        self.assertEqual(self.span.events[0].attributes["detail"], "test")

    def test_add_link(self):
        """Test link addition."""
        other_ctx = TraceContext.generate()
        self.span.add_link(other_ctx.trace_id, other_ctx.span_id, {"reason": "related"})

        self.assertEqual(len(self.span.links), 1)
        self.assertEqual(self.span.links[0].trace_id, other_ctx.trace_id)

    def test_set_status(self):
        """Test status setting."""
        self.span.set_status(SpanStatus.OK)
        self.assertEqual(self.span.status, SpanStatus.OK)

        self.span.set_status(SpanStatus.ERROR, "Something went wrong")
        self.assertEqual(self.span.status, SpanStatus.ERROR)
        self.assertEqual(self.span.status_message, "Something went wrong")

    def test_record_exception(self):
        """Test exception recording."""
        try:
            raise ValueError("test error")
        except ValueError as e:
            self.span.record_exception(e)

        self.assertEqual(self.span.status, SpanStatus.ERROR)
        self.assertEqual(len(self.span.events), 1)
        self.assertEqual(self.span.events[0].name, "exception")
        self.assertEqual(self.span.events[0].attributes["exception.type"], "ValueError")

    def test_span_end(self):
        """Test span ending."""
        self.assertIsNone(self.span.end_time)
        self.assertIsNone(self.span.duration_ms)

        time.sleep(0.01)  # 10ms
        self.span.end()

        self.assertIsNotNone(self.span.end_time)
        self.assertIsNotNone(self.span.duration_ms)
        self.assertGreaterEqual(self.span.duration_ms, 10)
        self.assertFalse(self.span.is_recording())

    def test_to_dict(self):
        """Test span serialization."""
        self.span.set_attribute("key", "value")
        self.span.add_event("test")
        self.span.end()

        data = self.span.to_dict()

        self.assertEqual(data["name"], "test-operation")
        self.assertEqual(data["trace_id"], self.ctx.trace_id)
        self.assertIn("key", data["attributes"])
        self.assertEqual(len(data["events"]), 1)

    def test_to_otlp(self):
        """Test OTLP format export."""
        self.span.set_attribute("key", "value")
        self.span.end()

        otlp = self.span.to_otlp()

        self.assertEqual(otlp["traceId"], self.ctx.trace_id)
        self.assertEqual(otlp["spanId"], self.ctx.span_id)
        self.assertEqual(otlp["name"], "test-operation")
        self.assertIsInstance(otlp["attributes"], list)

    def test_to_jaeger(self):
        """Test Jaeger format export."""
        self.span.set_attribute("key", "value")
        self.span.end()

        jaeger = self.span.to_jaeger()

        self.assertEqual(jaeger["traceID"], self.ctx.trace_id)
        self.assertEqual(jaeger["spanID"], self.ctx.span_id)
        self.assertEqual(jaeger["operationName"], "test-operation")
        self.assertIsInstance(jaeger["tags"], list)

    def test_to_zipkin(self):
        """Test Zipkin format export."""
        self.span.set_attribute("key", "value")
        self.span.end()

        zipkin = self.span.to_zipkin()

        self.assertEqual(zipkin["traceId"], self.ctx.trace_id)
        self.assertEqual(zipkin["id"], self.ctx.span_id)
        self.assertEqual(zipkin["name"], "test-operation")
        self.assertIn("key", zipkin["tags"])


class TestSamplers(unittest.TestCase):
    """Test sampling strategies."""

    def test_always_on_sampler(self):
        """Test AlwaysOnSampler."""
        sampler = AlwaysOnSampler()
        should_sample, attrs = sampler.should_sample(
            "abc123", "test", SpanKind.INTERNAL, {}, None
        )
        self.assertTrue(should_sample)

    def test_always_off_sampler(self):
        """Test AlwaysOffSampler."""
        sampler = AlwaysOffSampler()
        should_sample, attrs = sampler.should_sample(
            "abc123", "test", SpanKind.INTERNAL, {}, None
        )
        self.assertFalse(should_sample)

    def test_trace_id_ratio_sampler(self):
        """Test TraceIdRatioSampler."""
        # 100% sampling
        sampler = TraceIdRatioSampler(1.0)
        should_sample, _ = sampler.should_sample(
            "abc123", "test", SpanKind.INTERNAL, {}, None
        )
        self.assertTrue(should_sample)

        # 0% sampling
        sampler = TraceIdRatioSampler(0.0)
        should_sample, _ = sampler.should_sample(
            "abc123", "test", SpanKind.INTERNAL, {}, None
        )
        self.assertFalse(should_sample)

    def test_parent_based_sampler(self):
        """Test ParentBasedSampler."""
        sampler = ParentBasedSampler(root=TraceIdRatioSampler(0.5))

        # With sampled parent - should sample
        parent_ctx = TraceContext.generate()  # sampled by default
        should_sample, _ = sampler.should_sample(
            "abc123", "test", SpanKind.INTERNAL, {}, parent_ctx
        )
        self.assertTrue(should_sample)

        # With unsampled parent - should not sample
        parent_ctx.trace_flags = 0
        should_sample, _ = sampler.should_sample(
            "abc123", "test", SpanKind.INTERNAL, {}, parent_ctx
        )
        self.assertFalse(should_sample)


class TestTracer(unittest.TestCase):
    """Test Tracer functionality."""

    def setUp(self):
        """Set up test tracer."""
        self.exporter = InMemorySpanExporter()
        self.processor = SimpleSpanProcessor(self.exporter)
        self.tracer = Tracer(service_name="test-service")
        self.tracer.add_processor(self.processor)

    def tearDown(self):
        """Clean up."""
        self.tracer.shutdown()

    def test_start_and_end_span(self):
        """Test basic span lifecycle."""
        span = self.tracer.start_span("test-op")
        self.assertTrue(span.is_recording())

        self.tracer.end_span(span)
        self.assertFalse(span.is_recording())

        spans = self.exporter.get_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].name, "test-op")

    def test_trace_context_manager(self):
        """Test trace context manager."""
        with self.tracer.trace("operation-1") as span:
            span.set_attribute("key", "value")

        spans = self.exporter.get_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].status, SpanStatus.OK)

    def test_trace_context_manager_with_error(self):
        """Test trace context manager with exception."""
        try:
            with self.tracer.trace("failing-op") as span:
                raise RuntimeError("test error")
        except RuntimeError:
            pass

        spans = self.exporter.get_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].status, SpanStatus.ERROR)

    def test_nested_spans(self):
        """Test nested span creation."""
        with self.tracer.trace("parent") as parent:
            with self.tracer.trace("child") as child:
                self.assertEqual(child.trace_id, parent.trace_id)
                self.assertEqual(child.parent_span_id, parent.span_id)

        spans = self.exporter.get_spans()
        self.assertEqual(len(spans), 2)

    def test_async_trace(self):
        """Test async trace context manager."""
        async def async_operation():
            async with self.tracer.atrace("async-op") as span:
                span.set_attribute("async", True)
                await asyncio.sleep(0.01)

        asyncio.run(async_operation())

        spans = self.exporter.get_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].attributes["async"], True)


class TestTracingManager(unittest.TestCase):
    """Test TracingManager singleton."""

    def setUp(self):
        """Reset tracing before each test."""
        reset_tracing()

    def tearDown(self):
        """Reset tracing after each test."""
        reset_tracing()

    def test_singleton(self):
        """Test singleton pattern."""
        manager1 = get_tracing_manager()
        manager2 = get_tracing_manager()
        self.assertIs(manager1, manager2)

    def test_configure(self):
        """Test configuration."""
        config = TracingConfig(
            service_name="my-service",
            sample_rate=0.5,
        )
        manager = configure_tracing(config)

        self.assertEqual(manager.config.service_name, "my-service")
        self.assertEqual(manager.config.sample_rate, 0.5)

    def test_get_spans(self):
        """Test span retrieval."""
        manager = get_tracing_manager()
        tracer = manager.get_tracer()

        with tracer.trace("test-1"):
            pass
        with tracer.trace("test-2"):
            pass

        spans = manager.get_spans()
        self.assertEqual(len(spans), 2)

    def test_export_formats(self):
        """Test export format methods."""
        manager = get_tracing_manager()
        tracer = manager.get_tracer()

        with tracer.trace("test"):
            pass

        otlp = manager.export_otlp()
        self.assertEqual(len(otlp), 1)
        self.assertIn("traceId", otlp[0])

        jaeger = manager.export_jaeger()
        self.assertEqual(len(jaeger), 1)
        self.assertIn("traceID", jaeger[0])

        zipkin = manager.export_zipkin()
        self.assertEqual(len(zipkin), 1)
        self.assertIn("traceId", zipkin[0])


class TestContextPropagation(unittest.TestCase):
    """Test context propagation functions."""

    def setUp(self):
        """Reset tracing before each test."""
        reset_tracing()

    def tearDown(self):
        """Reset tracing after each test."""
        reset_tracing()

    def test_inject_and_extract(self):
        """Test context injection and extraction."""
        manager = get_tracing_manager()
        tracer = manager.get_tracer()

        with tracer.trace("parent"):
            headers = inject_context()
            self.assertIn("traceparent", headers)

            # Simulate receiving in another service
            extracted = extract_context(headers)
            self.assertIsNotNone(extracted)
            self.assertTrue(extracted.is_sampled)

    def test_inject_with_baggage(self):
        """Test injection with baggage."""
        baggage = Baggage()
        baggage.set("user_id", "123")

        headers = inject_context(baggage=baggage)
        self.assertIn("baggage", headers)
        self.assertIn("user_id=123", headers["baggage"])

    def test_extract_baggage(self):
        """Test baggage extraction."""
        headers = {"baggage": "user_id=123,session=abc"}
        baggage = extract_baggage(headers)

        self.assertEqual(baggage.get("user_id"), "123")
        self.assertEqual(baggage.get("session"), "abc")


class TestDecorators(unittest.TestCase):
    """Test tracing decorators."""

    def setUp(self):
        """Reset tracing before each test."""
        reset_tracing()

    def tearDown(self):
        """Reset tracing after each test."""
        reset_tracing()

    def test_traced_decorator(self):
        """Test @traced decorator."""
        @traced("sync-function")
        def my_function():
            return "result"

        result = my_function()

        self.assertEqual(result, "result")
        manager = get_tracing_manager()
        spans = manager.get_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].name, "sync-function")

    def test_traced_async_decorator(self):
        """Test @traced_async decorator."""
        @traced_async("async-function")
        async def my_async_function():
            await asyncio.sleep(0.01)
            return "async-result"

        result = asyncio.run(my_async_function())

        self.assertEqual(result, "async-result")
        manager = get_tracing_manager()
        spans = manager.get_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].name, "async-function")

    def test_traced_decorator_with_exception(self):
        """Test decorator with exception."""
        @traced("failing-function")
        def failing_function():
            raise ValueError("test error")

        with self.assertRaises(ValueError):
            failing_function()

        manager = get_tracing_manager()
        spans = manager.get_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].status, SpanStatus.ERROR)


class TestIntegrationHelpers(unittest.TestCase):
    """Test integration helper functions."""

    def setUp(self):
        """Reset tracing before each test."""
        reset_tracing()

    def tearDown(self):
        """Reset tracing after each test."""
        reset_tracing()

    def test_traced_operation(self):
        """Test TracedOperation context manager."""
        with TracedOperation("test-op", attributes={"key": "value"}) as span:
            span.add_event("started")

        manager = get_tracing_manager()
        spans = manager.get_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].attributes["key"], "value")

    def test_trace_rag_retrieve(self):
        """Test RAG retrieve helper."""
        with trace_rag_retrieve("test query") as span:
            span.add_event("retrieved")

        manager = get_tracing_manager()
        spans = manager.get_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].name, "rag.retrieve")
        self.assertIn("rag.query", spans[0].attributes)

    def test_trace_adapter_call(self):
        """Test adapter call helper."""
        with trace_adapter_call("exa", "search") as span:
            span.set_attribute("result_count", 10)

        manager = get_tracing_manager()
        spans = manager.get_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].name, "adapter.exa.search")
        self.assertEqual(spans[0].kind, SpanKind.CLIENT)

    def test_trace_memory_operation(self):
        """Test memory operation helper."""
        with trace_memory_operation("store", "archival") as span:
            span.set_attribute("key", "test-key")

        manager = get_tracing_manager()
        spans = manager.get_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].name, "memory.store")


class TestQuickSetup(unittest.TestCase):
    """Test quick setup functions."""

    def tearDown(self):
        """Reset tracing after each test."""
        reset_tracing()

    def test_development_setup(self):
        """Test development setup."""
        manager = setup_development_tracing()

        self.assertEqual(manager.config.service_name, "unleashed-dev")
        self.assertEqual(manager.config.sample_rate, 1.0)
        self.assertTrue(manager.config.export_console)
        self.assertFalse(manager.config.use_batch_processor)

    def test_production_setup(self):
        """Test production setup."""
        manager = setup_production_tracing(
            service_name="prod-service",
            sample_rate=0.1,
        )

        self.assertEqual(manager.config.service_name, "prod-service")
        self.assertEqual(manager.config.sample_rate, 0.1)
        self.assertFalse(manager.config.export_console)
        self.assertTrue(manager.config.use_batch_processor)


if __name__ == "__main__":
    unittest.main()
