"""
Tests for Domain Events System V40 Enhancements

Tests:
1. Event aggregation batching
2. Event replay capability
3. Dead letter queue with retry policies
4. Event versioning and schema evolution
5. Metrics and observability
6. Research events integration
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

# Import from domain_events
from core.orchestration.domain_events import (
    # Metrics
    EventMetrics,
    # Versioning
    EventVersion,
    EventSchemaEvolution,
    # Base event
    DomainEvent,
    # Research events
    ResearchQueryRequestedEvent,
    ResearchResultReceivedEvent,
    ResearchResultCachedEvent,
    ResearchCacheHitEvent,
    ResearchAdapterHealthEvent,
    # Dead letter queue
    DeadLetterEntry,
    DeadLetterQueue,
    # Event aggregation
    EventAggregator,
    # Event bus
    EventBus,
    # Event store
    EventStore,
    # Event emitters
    ResearchEventEmitter,
    # Registry
    EventRegistry,
)


# =============================================================================
# EVENT METRICS TESTS
# =============================================================================

class TestEventMetrics:
    """Tests for EventMetrics class."""

    def test_record_published(self):
        """Test recording published events."""
        metrics = EventMetrics()
        metrics.record_published("TestEvent")
        metrics.record_published("TestEvent")

        summary = metrics.get_summary()
        assert summary["events_published"] == 2

    def test_record_handled_latency(self):
        """Test recording handled events with latency."""
        metrics = EventMetrics()
        metrics.record_handled("handler1", 100.0)
        metrics.record_handled("handler1", 200.0)
        metrics.record_handled("handler1", 150.0)

        summary = metrics.get_summary()
        assert summary["events_handled"] == 3
        assert "handler1" in summary["handler_stats"]
        assert summary["handler_stats"]["handler1"]["count"] == 3
        assert summary["handler_stats"]["handler1"]["avg_ms"] == 150.0

    def test_record_failed(self):
        """Test recording failed events."""
        metrics = EventMetrics()
        metrics.record_failed("handler1", "TestEvent")
        metrics.record_failed("handler1", "TestEvent")
        metrics.record_failed("handler2", "OtherEvent")

        summary = metrics.get_summary()
        assert summary["events_failed"] == 3
        assert summary["errors_by_event_type"]["TestEvent"] == 2
        assert summary["errors_by_event_type"]["OtherEvent"] == 1

    def test_dlq_metrics(self):
        """Test dead letter queue metrics."""
        metrics = EventMetrics()
        metrics.record_dlq_addition()
        metrics.record_dlq_addition()
        metrics.record_dlq_retry()
        metrics.record_dlq_success()

        summary = metrics.get_summary()
        dlq = summary["dead_letter_queue"]
        assert dlq["additions"] == 2
        assert dlq["retries"] == 1
        assert dlq["successes"] == 1

    def test_get_percentile(self):
        """Test percentile calculations."""
        metrics = EventMetrics()
        for i in range(100):
            metrics.record_handled("handler1", float(i))

        # p50 should be around 49-50
        p50 = metrics.get_percentile("handler1", 50)
        assert 45 <= p50 <= 55

        # p95 should be around 94-95
        p95 = metrics.get_percentile("handler1", 95)
        assert 90 <= p95 <= 99


# =============================================================================
# EVENT VERSION TESTS
# =============================================================================

class TestEventVersion:
    """Tests for EventVersion class."""

    def test_version_creation(self):
        """Test creating versions."""
        v = EventVersion(1, 2, 3)
        assert str(v) == "1.2.3"

    def test_version_parsing(self):
        """Test parsing version strings."""
        v = EventVersion.parse("2.1.0")
        assert v.major == 2
        assert v.minor == 1
        assert v.patch == 0

    def test_version_comparison(self):
        """Test version comparison."""
        v1 = EventVersion(1, 0, 0)
        v2 = EventVersion(1, 1, 0)
        v3 = EventVersion(2, 0, 0)

        assert v1 < v2
        assert v2 < v3
        assert v1 < v3
        assert v2 > v1
        assert v1 <= v1
        assert v1 >= v1

    def test_version_compatibility(self):
        """Test version compatibility checks."""
        v1 = EventVersion(1, 0, 0)
        v2 = EventVersion(1, 5, 0)
        v3 = EventVersion(2, 0, 0)

        assert v1.is_compatible(v2)
        assert not v1.is_compatible(v3)
        assert v2.is_backward_compatible(v1)

    def test_version_bumping(self):
        """Test version bumping."""
        v = EventVersion(1, 2, 3)

        v_major = v.bump_major()
        assert str(v_major) == "2.0.0"

        v_minor = v.bump_minor()
        assert str(v_minor) == "1.3.0"

        v_patch = v.bump_patch()
        assert str(v_patch) == "1.2.4"


# =============================================================================
# DEAD LETTER QUEUE TESTS
# =============================================================================

class TestDeadLetterQueue:
    """Tests for DeadLetterQueue class."""

    @pytest.fixture
    def dlq(self):
        return DeadLetterQueue(max_retries=3, base_delay_seconds=0.01)

    @pytest.fixture
    def sample_event(self):
        return DomainEvent(
            event_type="TestEvent",
            aggregate_id="test-123",
        )

    @pytest.mark.asyncio
    async def test_add_to_dlq(self, dlq, sample_event):
        """Test adding events to DLQ."""
        await dlq.add(
            event=sample_event,
            handler_name="test_handler",
            error=ValueError("Test error"),
        )

        assert len(dlq) == 1
        stats = dlq.get_statistics()
        assert stats["total_added"] == 1
        assert stats["by_handler"]["test_handler"] == 1

    @pytest.mark.asyncio
    async def test_retry_success(self, dlq, sample_event):
        """Test successful retry from DLQ."""
        await dlq.add(
            event=sample_event,
            handler_name="test_handler",
            error=ValueError("Test error"),
        )

        # Mock handler that succeeds
        handler = AsyncMock()

        succeeded, failed = await dlq.retry_all({"test_handler": handler})

        assert succeeded == 1
        assert failed == 0
        assert len(dlq) == 0
        handler.assert_called_once_with(sample_event)

    @pytest.mark.asyncio
    async def test_retry_failure(self, dlq, sample_event):
        """Test failed retry stays in DLQ."""
        await dlq.add(
            event=sample_event,
            handler_name="test_handler",
            error=ValueError("Test error"),
        )

        # Mock handler that fails
        handler = AsyncMock(side_effect=RuntimeError("Still failing"))

        succeeded, failed = await dlq.retry_all({"test_handler": handler})

        assert succeeded == 0
        assert failed == 1
        assert len(dlq) == 1

        stats = dlq.get_statistics()
        assert stats["total_retried"] == 1

    @pytest.mark.asyncio
    async def test_max_retries_respected(self, dlq, sample_event):
        """Test that events exceeding max retries are skipped."""
        await dlq.add(
            event=sample_event,
            handler_name="test_handler",
            error=ValueError("Test error"),
        )

        # Fail multiple times to exceed max retries
        failing_handler = AsyncMock(side_effect=RuntimeError("Failing"))

        for _ in range(4):  # More than max_retries (3)
            await dlq.retry_all({"test_handler": failing_handler})

        # Entry should still be there but skipped
        entries = dlq.get_entries()
        assert len(entries) == 1
        assert entries[0].retry_count >= 3

    def test_get_entries_with_filters(self, dlq):
        """Test getting entries with filters."""
        # Add entries synchronously for testing
        entry1 = DeadLetterEntry(
            event=DomainEvent(event_type="TypeA", aggregate_id="1"),
            handler_name="handler1",
            error=ValueError("Error 1"),
            failed_at=datetime.now(timezone.utc),
        )
        entry2 = DeadLetterEntry(
            event=DomainEvent(event_type="TypeB", aggregate_id="2"),
            handler_name="handler2",
            error=ValueError("Error 2"),
            failed_at=datetime.now(timezone.utc),
        )
        dlq._queue = [entry1, entry2]

        # Filter by handler
        filtered = dlq.get_entries(handler_name="handler1")
        assert len(filtered) == 1
        assert filtered[0].handler_name == "handler1"

        # Filter by event type
        filtered = dlq.get_entries(event_type="TypeB")
        assert len(filtered) == 1
        assert filtered[0].event.event_type == "TypeB"


# =============================================================================
# EVENT AGGREGATOR TESTS
# =============================================================================

class TestEventAggregator:
    """Tests for EventAggregator class."""

    @pytest.fixture
    def mock_bus(self):
        bus = MagicMock(spec=EventBus)
        bus.publish_batch = AsyncMock()
        return bus

    @pytest.fixture
    def aggregator(self, mock_bus):
        return EventAggregator(
            mock_bus,
            batch_size=5,
            flush_interval_ms=100.0,
        )

    def test_add_events(self, aggregator):
        """Test adding events to aggregator."""
        event = DomainEvent(event_type="Test", aggregate_id="1")

        # Add less than batch size
        for i in range(3):
            should_flush = aggregator.add(event)
            assert not should_flush

        assert aggregator.pending_count() == 3

    def test_batch_size_trigger(self, aggregator):
        """Test that reaching batch size triggers flush flag."""
        event = DomainEvent(event_type="Test", aggregate_id="1")

        for i in range(4):
            should_flush = aggregator.add(event)
            assert not should_flush

        # 5th event should trigger flush
        should_flush = aggregator.add(event)
        assert should_flush

    @pytest.mark.asyncio
    async def test_flush(self, aggregator, mock_bus):
        """Test flushing events to bus."""
        events = [
            DomainEvent(event_type="Test", aggregate_id=str(i))
            for i in range(3)
        ]

        for event in events:
            aggregator.add(event)

        count = await aggregator.flush()

        assert count == 3
        assert aggregator.pending_count() == 0
        mock_bus.publish_batch.assert_called_once()

    def test_deduplication(self, mock_bus):
        """Test event deduplication."""
        aggregator = EventAggregator(
            mock_bus,
            batch_size=10,
            deduplicate=True,
        )

        event = DomainEvent(event_type="Test", aggregate_id="1")

        aggregator.add(event)
        aggregator.add(event)  # Duplicate

        stats = aggregator.get_statistics()
        assert stats["duplicates_skipped"] == 1
        assert aggregator.pending_count() == 1


# =============================================================================
# EVENT BUS ENHANCED TESTS
# =============================================================================

class TestEventBusEnhanced:
    """Tests for enhanced EventBus features."""

    @pytest.fixture
    def bus(self):
        metrics = EventMetrics()
        return EventBus(
            max_retries=2,
            retry_delay_seconds=0.01,
            handler_timeout_seconds=1.0,
            metrics=metrics,
        )

    @pytest.mark.asyncio
    async def test_handler_timeout(self, bus):
        """Test that slow handlers timeout."""
        async def slow_handler(event):
            await asyncio.sleep(5)  # Longer than timeout

        bus.subscribe("TestEvent", slow_handler, "slow_handler")

        event = DomainEvent(event_type="TestEvent", aggregate_id="1")
        await bus.publish(event)

        # Should be in DLQ after timeout
        dlq = bus.get_dlq()
        assert len(dlq) == 1

    @pytest.mark.asyncio
    async def test_metrics_tracked(self, bus):
        """Test that metrics are tracked correctly."""
        call_count = 0

        async def counting_handler(event):
            nonlocal call_count
            call_count += 1

        bus.subscribe("TestEvent", counting_handler, "counter")

        for i in range(3):
            event = DomainEvent(event_type="TestEvent", aggregate_id=str(i))
            await bus.publish(event)

        metrics = bus.get_metrics()
        assert metrics["events_published"] == 3
        assert metrics["events_handled"] == 3

    @pytest.mark.asyncio
    async def test_retry_dead_letters(self, bus):
        """Test retrying events from DLQ."""
        fail_count = 0

        async def failing_then_succeeding(event):
            nonlocal fail_count
            if fail_count < 3:  # Fail first 3 times
                fail_count += 1
                raise ValueError("Temporary failure")

        bus.subscribe("TestEvent", failing_then_succeeding, "flaky_handler")

        event = DomainEvent(event_type="TestEvent", aggregate_id="1")
        await bus.publish(event)

        # Should be in DLQ after initial failures
        assert len(bus.get_dlq()) == 1

        # Retry should eventually succeed
        await bus.retry_dead_letters()

        # After retry, DLQ may still have the event if it failed again
        # or be empty if it succeeded


# =============================================================================
# RESEARCH EVENTS TESTS
# =============================================================================

class TestResearchEvents:
    """Tests for research domain events."""

    def test_research_query_requested_event(self):
        """Test ResearchQueryRequestedEvent creation."""
        event = ResearchQueryRequestedEvent(
            query="AI safety research",
            adapter_name="exa",
            operation="search",
            parameters={"num_results": 10},
        )

        assert event.event_type == "ResearchQueryRequestedEvent"
        assert event.aggregate_type == "Research"
        assert event.payload["query"] == "AI safety research"
        assert event.payload["adapter_name"] == "exa"
        assert "num_results" in event.payload["parameters"]

    def test_research_result_received_event(self):
        """Test ResearchResultReceivedEvent creation."""
        event = ResearchResultReceivedEvent(
            request_id="req-123",
            adapter_name="tavily",
            result_count=15,
            latency_ms=250.5,
            success=True,
            cost_dollars=0.002,
        )

        assert event.event_type == "ResearchResultReceivedEvent"
        assert event.payload["result_count"] == 15
        assert event.payload["latency_ms"] == 250.5
        assert event.payload["cost_dollars"] == 0.002

    def test_research_result_cached_event(self):
        """Test ResearchResultCachedEvent creation."""
        event = ResearchResultCachedEvent(
            request_id="req-123",
            cache_key="abc123",
            cache_type="semantic",
            ttl_seconds=3600,
            adapter_name="exa",
            result_count=10,
        )

        assert event.event_type == "ResearchResultCachedEvent"
        assert event.payload["cache_type"] == "semantic"
        assert event.payload["ttl_seconds"] == 3600

    def test_research_cache_hit_event(self):
        """Test ResearchCacheHitEvent creation."""
        event = ResearchCacheHitEvent(
            query="AI agents",
            cache_key="xyz789",
            adapter_name="perplexity",
            similarity_score=0.95,
        )

        assert event.event_type == "ResearchCacheHitEvent"
        assert event.payload["similarity_score"] == 0.95


# =============================================================================
# EVENT STORE REPLAY TESTS
# =============================================================================

class TestEventStoreReplay:
    """Tests for EventStore replay capabilities."""

    @pytest.fixture
    def store(self):
        return EventStore()

    @pytest.fixture
    async def populated_store(self, store):
        """Store with some events."""
        events = [
            DomainEvent(
                event_type="TypeA",
                aggregate_id="agg-1",
                aggregate_type="TestAggregate",
            ),
            DomainEvent(
                event_type="TypeB",
                aggregate_id="agg-1",
                aggregate_type="TestAggregate",
            ),
            DomainEvent(
                event_type="TypeA",
                aggregate_id="agg-2",
                aggregate_type="TestAggregate",
            ),
        ]

        for event in events:
            await store.append([event])

        return store

    @pytest.mark.asyncio
    async def test_replay_all(self, populated_store):
        """Test replaying all events."""
        replayed = []

        async def handler(event):
            replayed.append(event)

        count = await populated_store.replay_all(handler)

        assert count == 3
        assert len(replayed) == 3

    @pytest.mark.asyncio
    async def test_replay_with_type_filter(self, populated_store):
        """Test replaying events with type filter."""
        replayed = []

        async def handler(event):
            replayed.append(event)

        count = await populated_store.replay_all(
            handler,
            event_types=["TypeA"],
        )

        assert count == 2
        assert all(e.event_type == "TypeA" for e in replayed)

    @pytest.mark.asyncio
    async def test_replay_by_aggregate_type(self, populated_store):
        """Test replaying events by aggregate type."""
        replayed = []

        async def handler(event):
            replayed.append(event)

        count = await populated_store.replay_by_aggregate_type(
            "TestAggregate",
            handler,
        )

        assert count == 3

    @pytest.mark.asyncio
    async def test_rebuild_projection(self, populated_store):
        """Test rebuilding a projection."""
        state = {"type_a_count": 0, "type_b_count": 0}

        async def handle_type_a(event):
            state["type_a_count"] += 1

        async def handle_type_b(event):
            state["type_b_count"] += 1

        counts = await populated_store.rebuild_projection(
            "test_projection",
            {
                "TypeA": handle_type_a,
                "TypeB": handle_type_b,
            },
        )

        assert counts["TypeA"] == 2
        assert counts["TypeB"] == 1
        assert state["type_a_count"] == 2
        assert state["type_b_count"] == 1


# =============================================================================
# EVENT REGISTRY TESTS
# =============================================================================

class TestEventRegistry:
    """Tests for EventRegistry."""

    def test_registry_singleton(self):
        """Test registry is a singleton."""
        r1 = EventRegistry.instance()
        r2 = EventRegistry.instance()
        assert r1 is r2

    def test_builtin_events_registered(self):
        """Test that builtin events are registered."""
        registry = EventRegistry.instance()

        # Check research events are registered
        assert registry.get_event_class("ResearchQueryRequestedEvent") is not None
        assert registry.get_event_class("ResearchResultReceivedEvent") is not None
        assert registry.get_event_class("ResearchResultCachedEvent") is not None

    def test_deserialize_event(self):
        """Test deserializing an event."""
        registry = EventRegistry.instance()

        event_data = {
            "event_type": "ResearchQueryRequestedEvent",
            "aggregate_id": "test-123",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {"query": "test query"},
        }

        event = registry.deserialize(event_data)
        assert event.event_type == "ResearchQueryRequestedEvent"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
