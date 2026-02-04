"""
Unit tests for Domain Events System - V39 Architecture

Tests cover:
- Event creation and serialization
- Event bus publish/subscribe
- Event store (event sourcing)
- Transactional outbox
- Memory integration
"""

from __future__ import annotations

import asyncio
import json
import pytest
from datetime import datetime, timezone
from typing import List

from core.orchestration.domain_events import (
    # Versioning
    EventVersion,
    # Base event
    DomainEvent,
    # Memory events
    MemoryStoredEvent,
    MemoryRetrievedEvent,
    MemoryEvictedEvent,
    MemoryPromotedEvent,
    # Session events
    SessionStartedEvent,
    SessionEndedEvent,
    SessionCheckpointEvent,
    # Flow events
    FlowStartedEvent,
    FlowCompletedEvent,
    FlowFailedEvent,
    # Event bus
    EventBus,
    # Event sourcing
    EventStore,
    EventSourcedAggregate,
    ConcurrencyError,
    # Outbox pattern
    OutboxMessage,
    TransactionalOutbox,
    # Memory integration
    MemoryEventEmitter,
    # Registry
    EventRegistry,
)


# =============================================================================
# EVENT VERSION TESTS
# =============================================================================

class TestEventVersion:
    """Tests for EventVersion."""

    def test_version_string(self):
        """Test version string representation."""
        v = EventVersion(1, 2, 3)
        assert str(v) == "1.2.3"

    def test_version_equality(self):
        """Test version equality."""
        v1 = EventVersion(1, 0, 0)
        v2 = EventVersion(1, 0, 0)
        v3 = EventVersion(1, 1, 0)

        assert v1 == v2
        assert v1 != v3

    def test_version_compatibility(self):
        """Test version compatibility check."""
        v1 = EventVersion(1, 0, 0)
        v2 = EventVersion(1, 5, 3)
        v3 = EventVersion(2, 0, 0)

        assert v1.is_compatible(v2)
        assert not v1.is_compatible(v3)

    def test_version_parse(self):
        """Test version parsing."""
        v = EventVersion.parse("2.3.4")
        assert v.major == 2
        assert v.minor == 3
        assert v.patch == 4


# =============================================================================
# DOMAIN EVENT TESTS
# =============================================================================

class TestDomainEvent:
    """Tests for DomainEvent base class."""

    def test_event_creation(self):
        """Test basic event creation."""
        event = DomainEvent(
            aggregate_id="agg-123",
            aggregate_type="TestAggregate",
        )

        assert event.event_type == "DomainEvent"
        assert event.aggregate_id == "agg-123"
        assert event.aggregate_type == "TestAggregate"
        assert event.event_id is not None
        assert event.version == 1
        assert event.timestamp is not None

    def test_event_serialization(self):
        """Test event serialization to dict."""
        event = DomainEvent(
            event_id="evt-001",
            aggregate_id="agg-123",
            aggregate_type="TestAggregate",
            payload={"key": "value"},
            metadata={"user": "test"},
        )

        data = event.to_dict()

        assert data["event_id"] == "evt-001"
        assert data["aggregate_id"] == "agg-123"
        assert data["payload"] == {"key": "value"}
        assert data["metadata"] == {"user": "test"}

    def test_event_deserialization(self):
        """Test event deserialization from dict."""
        data = {
            "event_id": "evt-001",
            "event_type": "DomainEvent",
            "aggregate_id": "agg-123",
            "aggregate_type": "TestAggregate",
            "version": 2,
            "timestamp": "2024-01-01T12:00:00+00:00",
            "payload": {"key": "value"},
            "metadata": {},
            "schema_version": "1.0.0",
        }

        event = DomainEvent.from_dict(data)

        assert event.event_id == "evt-001"
        assert event.aggregate_id == "agg-123"
        assert event.version == 2

    def test_event_with_metadata(self):
        """Test adding metadata to event."""
        event = DomainEvent(
            aggregate_id="agg-123",
            metadata={"original": "value"},
        )

        new_event = event.with_metadata(added="new_value")

        assert new_event.metadata["original"] == "value"
        assert new_event.metadata["added"] == "new_value"
        # Original event unchanged
        assert "added" not in event.metadata


# =============================================================================
# MEMORY EVENT TESTS
# =============================================================================

class TestMemoryEvents:
    """Tests for memory domain events."""

    def test_memory_stored_event(self):
        """Test MemoryStoredEvent creation."""
        event = MemoryStoredEvent(
            aggregate_id="session-123",
            memory_id="mem-456",
            content_preview="User prefers dark mode settings",
            memory_type="preference",
        )

        assert event.event_type == "MemoryStoredEvent"
        assert event.aggregate_type == "Memory"
        assert event.memory_id == "mem-456"
        assert event.payload["memory_type"] == "preference"
        assert event.payload["content_preview"] == "User prefers dark mode settings"

    def test_memory_retrieved_event(self):
        """Test MemoryRetrievedEvent creation."""
        event = MemoryRetrievedEvent(
            aggregate_id="session-123",
            memory_id="mem-456",
            query="user preferences",
            hit_count=5,
        )

        assert event.event_type == "MemoryRetrievedEvent"
        assert event.payload["hit_count"] == 5

    def test_memory_evicted_event(self):
        """Test MemoryEvictedEvent creation."""
        event = MemoryEvictedEvent(
            aggregate_id="session-123",
            memory_id="mem-456",
            eviction_reason="ttl_expired",
            tier="archival",
        )

        assert event.event_type == "MemoryEvictedEvent"
        assert event.payload["eviction_reason"] == "ttl_expired"

    def test_memory_promoted_event(self):
        """Test MemoryPromotedEvent creation."""
        event = MemoryPromotedEvent(
            aggregate_id="session-123",
            memory_id="mem-456",
            from_tier="recall_memory",
            to_tier="core_memory",
            reason="frequency",
        )

        assert event.event_type == "MemoryPromotedEvent"
        assert event.payload["from_tier"] == "recall_memory"
        assert event.payload["to_tier"] == "core_memory"


# =============================================================================
# SESSION EVENT TESTS
# =============================================================================

class TestSessionEvents:
    """Tests for session domain events."""

    def test_session_started_event(self):
        """Test SessionStartedEvent creation."""
        event = SessionStartedEvent(
            session_id="sess-001",
            project_path="/home/user/project",
            agent_type="coder",
        )

        assert event.event_type == "SessionStartedEvent"
        assert event.aggregate_id == "sess-001"
        assert event.payload["project_path"] == "/home/user/project"

    def test_session_ended_event(self):
        """Test SessionEndedEvent creation."""
        event = SessionEndedEvent(
            session_id="sess-001",
            duration_seconds=3600.0,
            exit_reason="completed",
        )

        assert event.event_type == "SessionEndedEvent"
        assert event.payload["duration_seconds"] == 3600.0

    def test_session_checkpoint_event(self):
        """Test SessionCheckpointEvent creation."""
        event = SessionCheckpointEvent(
            session_id="sess-001",
            checkpoint_id="cp-001",
            state_summary="Completed authentication module",
        )

        assert event.event_type == "SessionCheckpointEvent"
        assert event.payload["checkpoint_id"] == "cp-001"


# =============================================================================
# FLOW EVENT TESTS
# =============================================================================

class TestFlowEvents:
    """Tests for flow domain events."""

    def test_flow_started_event(self):
        """Test FlowStartedEvent creation."""
        event = FlowStartedEvent(
            flow_id="flow-001",
            flow_type="research",
            input_summary="Searching for AI trends",
        )

        assert event.event_type == "FlowStartedEvent"
        assert event.aggregate_id == "flow-001"

    def test_flow_completed_event(self):
        """Test FlowCompletedEvent creation."""
        event = FlowCompletedEvent(
            flow_id="flow-001",
            flow_type="research",
            duration_seconds=120.5,
            steps_completed=5,
        )

        assert event.event_type == "FlowCompletedEvent"
        assert event.payload["steps_completed"] == 5

    def test_flow_failed_event(self):
        """Test FlowFailedEvent creation."""
        event = FlowFailedEvent(
            flow_id="flow-001",
            flow_type="research",
            error_message="Rate limit exceeded",
            failed_step=3,
        )

        assert event.event_type == "FlowFailedEvent"
        assert event.payload["failed_step"] == 3


# =============================================================================
# EVENT BUS TESTS
# =============================================================================

class TestEventBus:
    """Tests for EventBus."""

    @pytest.fixture
    def event_bus(self):
        """Create fresh event bus."""
        return EventBus(max_retries=2, retry_delay_seconds=0.01)

    @pytest.mark.asyncio
    async def test_publish_to_subscriber(self, event_bus):
        """Test basic publish/subscribe."""
        received_events: List[DomainEvent] = []

        async def handler(event: DomainEvent):
            received_events.append(event)

        event_bus.subscribe("MemoryStoredEvent", handler)

        event = MemoryStoredEvent(
            aggregate_id="session-123",
            memory_id="mem-001",
            content_preview="Test content",
            memory_type="fact",
        )

        await event_bus.publish(event)

        assert len(received_events) == 1
        assert received_events[0].memory_id == "mem-001"

    @pytest.mark.asyncio
    async def test_wildcard_subscription(self, event_bus):
        """Test wildcard subscription receives all events."""
        received_events: List[DomainEvent] = []

        async def handler(event: DomainEvent):
            received_events.append(event)

        event_bus.subscribe("*", handler)

        await event_bus.publish(MemoryStoredEvent(
            aggregate_id="agg-1", memory_id="m1"
        ))
        await event_bus.publish(SessionStartedEvent(
            session_id="s1", project_path="/test"
        ))

        assert len(received_events) == 2

    @pytest.mark.asyncio
    async def test_multiple_handlers(self, event_bus):
        """Test multiple handlers for same event type."""
        results = []

        async def handler1(event: DomainEvent):
            results.append("handler1")

        async def handler2(event: DomainEvent):
            results.append("handler2")

        event_bus.subscribe("MemoryStoredEvent", handler1)
        event_bus.subscribe("MemoryStoredEvent", handler2)

        await event_bus.publish(MemoryStoredEvent(
            aggregate_id="agg-1", memory_id="m1"
        ))

        assert "handler1" in results
        assert "handler2" in results

    @pytest.mark.asyncio
    async def test_handler_retry(self, event_bus):
        """Test handler retry on failure."""
        attempts = []

        async def failing_handler(event: DomainEvent):
            attempts.append(1)
            if len(attempts) < 2:
                raise RuntimeError("Transient error")

        event_bus.subscribe("MemoryStoredEvent", failing_handler)

        await event_bus.publish(MemoryStoredEvent(
            aggregate_id="agg-1", memory_id="m1"
        ))

        assert len(attempts) == 2  # First failure + retry success

    @pytest.mark.asyncio
    async def test_dead_letter_queue(self, event_bus):
        """Test dead letter queue for failed events."""
        async def always_fails(event: DomainEvent):
            raise RuntimeError("Permanent error")

        event_bus.subscribe("MemoryStoredEvent", always_fails)

        await event_bus.publish(MemoryStoredEvent(
            aggregate_id="agg-1", memory_id="m1"
        ))

        dlq = event_bus.get_dead_letter_queue()
        assert len(dlq) == 1
        assert dlq[0][0].memory_id == "m1"

    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_bus):
        """Test handler unsubscription."""
        received = []

        async def handler(event: DomainEvent):
            received.append(event)

        event_bus.subscribe("MemoryStoredEvent", handler)

        # First publish should be received
        await event_bus.publish(MemoryStoredEvent(
            aggregate_id="agg-1", memory_id="m1"
        ))
        assert len(received) == 1

        # Unsubscribe
        result = event_bus.unsubscribe("MemoryStoredEvent", handler)
        assert result is True

        # Second publish should not be received
        await event_bus.publish(MemoryStoredEvent(
            aggregate_id="agg-2", memory_id="m2"
        ))
        assert len(received) == 1  # Still 1

    @pytest.mark.asyncio
    async def test_batch_publish(self, event_bus):
        """Test batch event publishing."""
        received = []

        async def handler(event: DomainEvent):
            received.append(event)

        event_bus.subscribe("*", handler)

        events = [
            MemoryStoredEvent(aggregate_id="a1", memory_id=f"m{i}")
            for i in range(5)
        ]

        await event_bus.publish_batch(events)

        assert len(received) == 5

    def test_metrics(self, event_bus):
        """Test event bus metrics."""
        metrics = event_bus.get_metrics()

        assert "events_published" in metrics
        assert "events_handled" in metrics
        assert "registered_handlers" in metrics


# =============================================================================
# EVENT STORE TESTS
# =============================================================================

class TestEventStore:
    """Tests for EventStore (event sourcing)."""

    @pytest.fixture
    def event_store(self):
        """Create fresh event store."""
        return EventStore()

    @pytest.mark.asyncio
    async def test_append_events(self, event_store):
        """Test appending events to store."""
        events = [
            MemoryStoredEvent(aggregate_id="agg-1", memory_id="m1"),
            MemoryStoredEvent(aggregate_id="agg-1", memory_id="m2"),
        ]

        await event_store.append(events)

        stored = await event_store.get_events("agg-1")
        assert len(stored) == 2
        assert stored[0].version == 1
        assert stored[1].version == 2

    @pytest.mark.asyncio
    async def test_get_events_after_version(self, event_store):
        """Test getting events after specific version."""
        events = [
            MemoryStoredEvent(aggregate_id="agg-1", memory_id=f"m{i}")
            for i in range(5)
        ]
        await event_store.append(events)

        after_v2 = await event_store.get_events("agg-1", after_version=2)
        assert len(after_v2) == 3
        assert after_v2[0].version == 3

    @pytest.mark.asyncio
    async def test_replay_events(self, event_store):
        """Test event replay."""
        events = [
            MemoryStoredEvent(aggregate_id="agg-1", memory_id=f"m{i}")
            for i in range(3)
        ]
        await event_store.append(events)

        replayed = []
        def handler(event: DomainEvent):
            replayed.append(event)

        count = await event_store.replay("agg-1", handler)

        assert count == 3
        assert len(replayed) == 3

    @pytest.mark.asyncio
    async def test_optimistic_concurrency(self, event_store):
        """Test optimistic concurrency check."""
        # Append first event
        await event_store.append([
            MemoryStoredEvent(aggregate_id="agg-1", memory_id="m1")
        ])

        # Try to append with wrong expected version
        with pytest.raises(ConcurrencyError):
            await event_store.append(
                [MemoryStoredEvent(aggregate_id="agg-1", memory_id="m2")],
                expected_version=0,  # Should be 1
            )

    @pytest.mark.asyncio
    async def test_snapshots(self, event_store):
        """Test snapshot save/load."""
        await event_store.save_snapshot(
            aggregate_id="agg-1",
            version=5,
            snapshot_data={"state": "active", "count": 10},
        )

        snapshot = await event_store.get_snapshot("agg-1")
        assert snapshot is not None
        version, data = snapshot
        assert version == 5
        assert data["count"] == 10

    @pytest.mark.asyncio
    async def test_stream_version(self, event_store):
        """Test getting stream version."""
        await event_store.append([
            MemoryStoredEvent(aggregate_id="agg-1", memory_id=f"m{i}")
            for i in range(3)
        ])

        version = await event_store.get_stream_version("agg-1")
        assert version == 3

    @pytest.mark.asyncio
    async def test_stream_exists(self, event_store):
        """Test stream existence check."""
        assert not await event_store.stream_exists("agg-1")

        await event_store.append([
            MemoryStoredEvent(aggregate_id="agg-1", memory_id="m1")
        ])

        assert await event_store.stream_exists("agg-1")

    def test_statistics(self, event_store):
        """Test event store statistics."""
        stats = event_store.get_statistics()

        assert "total_events" in stats
        assert "stream_count" in stats


# =============================================================================
# TRANSACTIONAL OUTBOX TESTS
# =============================================================================

class TestTransactionalOutbox:
    """Tests for TransactionalOutbox."""

    @pytest.fixture
    def outbox(self):
        """Create fresh outbox."""
        return TransactionalOutbox()

    @pytest.mark.asyncio
    async def test_add_event(self, outbox):
        """Test adding event to outbox."""
        event = MemoryStoredEvent(
            aggregate_id="agg-1",
            memory_id="m1",
        )

        message_id = await outbox.add(event)

        assert message_id is not None
        pending = await outbox.get_pending()
        assert len(pending) == 1

    @pytest.mark.asyncio
    async def test_mark_processed(self, outbox):
        """Test marking message as processed."""
        event = MemoryStoredEvent(aggregate_id="agg-1", memory_id="m1")
        message_id = await outbox.add(event)

        await outbox.mark_processed(message_id)

        pending = await outbox.get_pending()
        assert len(pending) == 0

    @pytest.mark.asyncio
    async def test_process_batch(self, outbox):
        """Test batch processing."""
        # Add multiple events
        for i in range(5):
            await outbox.add(MemoryStoredEvent(
                aggregate_id=f"agg-{i}",
                memory_id=f"m{i}",
            ))

        published = []

        async def publisher(event: DomainEvent):
            published.append(event)

        processed, failed = await outbox.process_batch(publisher)

        assert processed == 5
        assert failed == 0
        assert len(published) == 5

    @pytest.mark.asyncio
    async def test_retry_tracking(self, outbox):
        """Test retry count tracking on failure."""
        event = MemoryStoredEvent(aggregate_id="agg-1", memory_id="m1")
        message_id = await outbox.add(event)

        # Mark as failed
        await outbox.mark_processed(message_id, error="Network error")

        pending = await outbox.get_pending()
        assert len(pending) == 1
        assert pending[0].retry_count == 1
        assert pending[0].last_error == "Network error"

    @pytest.mark.asyncio
    async def test_cleanup(self, outbox):
        """Test cleanup of old processed messages."""
        import asyncio
        event = MemoryStoredEvent(aggregate_id="agg-1", memory_id="m1")
        message_id = await outbox.add(event)
        await outbox.mark_processed(message_id)

        # Need to wait briefly for age to be > 0
        await asyncio.sleep(0.001)

        # With 0 age threshold, should remove if any time has passed
        removed = await outbox.cleanup(older_than_seconds=0)

        assert removed == 1

    def test_statistics(self, outbox):
        """Test outbox statistics."""
        stats = outbox.get_statistics()

        assert "total_messages" in stats
        assert "processed" in stats
        assert "pending" in stats


# =============================================================================
# EVENT BUS + OUTBOX INTEGRATION TESTS
# =============================================================================

class TestEventBusOutboxIntegration:
    """Tests for EventBus and TransactionalOutbox integration."""

    @pytest.mark.asyncio
    async def test_publish_and_persist(self):
        """Test publish_and_persist pattern."""
        event_bus = EventBus()
        outbox = TransactionalOutbox()

        received = []

        async def handler(event: DomainEvent):
            received.append(event)

        event_bus.subscribe("MemoryStoredEvent", handler)

        event = MemoryStoredEvent(
            aggregate_id="agg-1",
            memory_id="m1",
        )

        await event_bus.publish_and_persist(event, outbox)

        # Should be delivered locally
        assert len(received) == 1

        # Should be persisted in outbox
        pending = await outbox.get_pending()
        assert len(pending) == 1


# =============================================================================
# EVENT-SOURCED AGGREGATE TESTS
# =============================================================================

class TestEventSourcedAggregate:
    """Tests for EventSourcedAggregate base class."""

    class CounterAggregate(EventSourcedAggregate):
        """Test aggregate that counts events."""

        def __init__(self, aggregate_id: str):
            super().__init__(aggregate_id)
            self.count = 0

        def apply_MemoryStoredEvent(self, event: MemoryStoredEvent):
            self.count += 1

        def increment(self):
            self.raise_event(MemoryStoredEvent(
                aggregate_id=self.aggregate_id,
                memory_id=f"m{self.count + 1}",
            ))

        def create_snapshot(self):
            return {"count": self.count}

        def restore_snapshot(self, snapshot):
            self.count = snapshot["count"]

    @pytest.mark.asyncio
    async def test_aggregate_event_application(self):
        """Test aggregate applies events correctly."""
        agg = self.CounterAggregate("counter-1")

        agg.increment()
        agg.increment()
        agg.increment()

        assert agg.count == 3
        assert agg.version == 3
        assert len(agg.uncommitted_events) == 3

    @pytest.mark.asyncio
    async def test_aggregate_save_and_load(self):
        """Test aggregate persistence."""
        store = EventStore()

        # Create and save aggregate
        agg1 = self.CounterAggregate("counter-1")
        agg1.increment()
        agg1.increment()
        await agg1.save(store)

        # Load from store
        agg2 = await self.CounterAggregate.load("counter-1", store)

        assert agg2.count == 2
        assert agg2.version == 2

    @pytest.mark.asyncio
    async def test_aggregate_with_snapshot(self):
        """Test aggregate with snapshot recovery."""
        store = EventStore()

        # Create aggregate with many events
        agg1 = self.CounterAggregate("counter-1")
        for _ in range(10):
            agg1.increment()
        await agg1.save(store)

        # Save snapshot at version 5
        await store.save_snapshot("counter-1", 5, {"count": 5})

        # Load should use snapshot + replay remaining events
        agg2 = await self.CounterAggregate.load("counter-1", store)

        assert agg2.count == 10
        assert agg2.version == 10


# =============================================================================
# MEMORY EVENT EMITTER TESTS
# =============================================================================

class TestMemoryEventEmitter:
    """Tests for MemoryEventEmitter."""

    @pytest.mark.asyncio
    async def test_emit_memory_stored(self):
        """Test emitting memory stored event."""
        event_bus = EventBus()
        emitter = MemoryEventEmitter(event_bus)

        received = []

        async def handler(event: MemoryStoredEvent):
            received.append(event)

        event_bus.subscribe("MemoryStoredEvent", handler)

        await emitter.emit_memory_stored(
            memory_id="m1",
            content="Test content",
            memory_type="fact",
            session_id="s1",
        )

        assert len(received) == 1
        assert received[0].memory_id == "m1"

    @pytest.mark.asyncio
    async def test_emit_memory_evicted(self):
        """Test emitting memory evicted event."""
        event_bus = EventBus()
        emitter = MemoryEventEmitter(event_bus)

        received = []

        async def handler(event: MemoryEvictedEvent):
            received.append(event)

        event_bus.subscribe("MemoryEvictedEvent", handler)

        await emitter.emit_memory_evicted(
            memory_id="m1",
            reason="capacity",
            tier="main_context",
        )

        assert len(received) == 1
        assert received[0].eviction_reason == "capacity"


# =============================================================================
# EVENT REGISTRY TESTS
# =============================================================================

class TestEventRegistry:
    """Tests for EventRegistry."""

    def test_singleton_instance(self):
        """Test registry singleton."""
        reg1 = EventRegistry.instance()
        reg2 = EventRegistry.instance()

        assert reg1 is reg2

    def test_builtin_events_registered(self):
        """Test builtin events are registered."""
        registry = EventRegistry.instance()

        assert registry.get_event_class("MemoryStoredEvent") is MemoryStoredEvent
        assert registry.get_event_class("SessionStartedEvent") is SessionStartedEvent

    def test_deserialize_event(self):
        """Test event deserialization via registry."""
        registry = EventRegistry.instance()

        data = {
            "event_id": "evt-001",
            "event_type": "MemoryStoredEvent",
            "aggregate_id": "agg-1",
            "aggregate_type": "Memory",
            "version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {"memory_id": "m1"},
            "metadata": {},
            "schema_version": "1.0.0",
        }

        event = registry.deserialize(data)

        # Should return proper event type
        assert isinstance(event, DomainEvent)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestDomainEventsIntegration:
    """Integration tests for the complete domain events system."""

    @pytest.mark.asyncio
    async def test_full_event_flow(self):
        """Test complete event flow: create -> publish -> store -> replay."""
        # Setup
        event_bus = EventBus()
        event_store = EventStore()
        outbox = TransactionalOutbox()

        published_events = []
        stored_events = []

        async def local_handler(event: DomainEvent):
            published_events.append(event)
            await event_store.append([event])

        event_bus.subscribe("*", local_handler)

        # Create session
        session_started = SessionStartedEvent(
            session_id="session-001",
            project_path="/test/project",
        )
        await event_bus.publish_and_persist(session_started, outbox)

        # Store memory
        memory_stored = MemoryStoredEvent(
            aggregate_id="session-001",
            memory_id="mem-001",
            content_preview="Important fact",
            memory_type="fact",
        )
        await event_bus.publish_and_persist(memory_stored, outbox)

        # End session
        session_ended = SessionEndedEvent(
            session_id="session-001",
            duration_seconds=120.0,
        )
        await event_bus.publish_and_persist(session_ended, outbox)

        # Verify events were published
        assert len(published_events) == 3

        # Verify events are in store
        all_events = await event_store.get_all_events()
        assert len(all_events) == 3

        # Verify events are in outbox
        outbox_stats = outbox.get_statistics()
        assert outbox_stats["pending"] == 3

        # Replay events
        replayed = []
        def replay_handler(event: DomainEvent):
            replayed.append(event)

        # Session events stored under session-001
        count = await event_store.replay("session-001", replay_handler)

        # Memory event also stored under session-001 as aggregate_id
        assert count >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
