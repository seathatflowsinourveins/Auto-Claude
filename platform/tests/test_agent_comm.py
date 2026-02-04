"""
Tests for Agent-to-Agent Communication Module

Comprehensive test suite covering:
1. Message Types and serialization
2. Memory-Based Inbox/Outbox
3. Reference-Based Large Outputs
4. Broadcast Support
5. Message Queue (FIFO, priority, acknowledgments)
6. Swarm Integration

Run with: pytest platform/tests/test_agent_comm.py -v
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

from core.orchestration.agent_comm import (
    LARGE_PAYLOAD_THRESHOLD,
    AgentCommSystem,
    AgentMessage,
    BroadcastSentEvent,
    InboxStats,
    MessageAcknowledgedEvent,
    MessageAcknowledgment,
    MessagePriority,
    MessageReceivedEvent,
    MessageSentEvent,
    MessageStatus,
    MessageType,
    SQLiteMessageStore,
    TopicSubscription,
    create_agent_comm_system,
)
from core.orchestration.domain_events import EventBus


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_messages.db"


@pytest.fixture
def message_store(temp_db_path):
    """Create a message store with temporary database."""
    store = SQLiteMessageStore(db_path=temp_db_path)
    yield store
    store.close()


@pytest.fixture
def event_bus():
    """Create an event bus for testing."""
    return EventBus()


@pytest.fixture
async def comm_system(temp_db_path, event_bus):
    """Create a communication system for testing."""
    comm = await create_agent_comm_system(
        db_path=temp_db_path,
        event_bus=event_bus,
    )
    yield comm
    await comm.stop()


# =============================================================================
# MESSAGE TYPE TESTS
# =============================================================================


class TestAgentMessage:
    """Tests for AgentMessage data class."""

    def test_message_creation(self):
        """Test basic message creation."""
        msg = AgentMessage(
            sender_id="agent-1",
            receiver_id="agent-2",
            message_type=MessageType.TASK,
            payload={"task": "test task"},
        )

        assert msg.sender_id == "agent-1"
        assert msg.receiver_id == "agent-2"
        assert msg.message_type == MessageType.TASK
        assert msg.payload == {"task": "test task"}
        assert msg.id.startswith("msg-")
        assert msg.status == MessageStatus.PENDING
        assert not msg.is_broadcast

    def test_message_with_priority(self):
        """Test message with different priorities."""
        low = AgentMessage(priority=MessagePriority.LOW)
        normal = AgentMessage(priority=MessagePriority.NORMAL)
        high = AgentMessage(priority=MessagePriority.HIGH)
        critical = AgentMessage(priority=MessagePriority.CRITICAL)

        assert low.priority.value < normal.priority.value
        assert normal.priority.value < high.priority.value
        assert high.priority.value < critical.priority.value

    def test_broadcast_message(self):
        """Test broadcast message detection."""
        broadcast = AgentMessage(
            sender_id="agent-1",
            receiver_id="broadcast",
            topic="status_updates",
        )

        assert broadcast.is_broadcast
        assert broadcast.topic == "status_updates"

    def test_message_expiration(self):
        """Test message TTL and expiration."""
        # Non-expiring message
        msg1 = AgentMessage(ttl_seconds=0)
        assert not msg1.is_expired

        # Expired message (simulate with past timestamp)
        msg2 = AgentMessage(
            ttl_seconds=1,
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        assert msg2.is_expired

    def test_message_reference(self):
        """Test reference detection for large payloads."""
        # No reference
        msg1 = AgentMessage(payload={"small": "data"})
        assert not msg1.has_reference

        # With reference
        msg2 = AgentMessage(
            payload={"_reference": "memory://payloads/123"},
            reference="memory://payloads/123",
        )
        assert msg2.has_reference

    def test_message_serialization(self):
        """Test message to_dict and from_dict."""
        original = AgentMessage(
            sender_id="agent-1",
            receiver_id="agent-2",
            message_type=MessageType.RESULT,
            payload={"result": "success", "data": [1, 2, 3]},
            priority=MessagePriority.HIGH,
            correlation_id="corr-123",
            topic="results",
            metadata={"key": "value"},
        )

        # Convert to dict and back
        data = original.to_dict()
        restored = AgentMessage.from_dict(data)

        assert restored.id == original.id
        assert restored.sender_id == original.sender_id
        assert restored.receiver_id == original.receiver_id
        assert restored.message_type == original.message_type
        assert restored.payload == original.payload
        assert restored.priority == original.priority
        assert restored.correlation_id == original.correlation_id
        assert restored.topic == original.topic
        assert restored.metadata == original.metadata


class TestMessageAcknowledgment:
    """Tests for MessageAcknowledgment."""

    def test_successful_ack(self):
        """Test successful acknowledgment."""
        ack = MessageAcknowledgment(
            message_id="msg-123",
            agent_id="agent-1",
            success=True,
        )

        assert ack.message_id == "msg-123"
        assert ack.agent_id == "agent-1"
        assert ack.success is True
        assert ack.error is None

    def test_failed_ack(self):
        """Test failed acknowledgment with error."""
        ack = MessageAcknowledgment(
            message_id="msg-123",
            agent_id="agent-1",
            success=False,
            error="Processing failed: timeout",
        )

        assert not ack.success
        assert "timeout" in ack.error


# =============================================================================
# MESSAGE STORE TESTS
# =============================================================================


class TestSQLiteMessageStore:
    """Tests for SQLiteMessageStore."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve_message(self, message_store):
        """Test storing and retrieving a message."""
        msg = AgentMessage(
            sender_id="agent-1",
            receiver_id="agent-2",
            message_type=MessageType.TASK,
            payload={"task": "test"},
        )

        # Store
        result = await message_store.store_message(msg)
        assert result is True

        # Retrieve
        messages = await message_store.get_messages_for_agent("agent-2")
        assert len(messages) == 1
        assert messages[0].id == msg.id
        assert messages[0].sender_id == "agent-1"
        assert messages[0].payload == {"task": "test"}

    @pytest.mark.asyncio
    async def test_message_status_updates(self, message_store):
        """Test updating message status."""
        msg = AgentMessage(
            sender_id="agent-1",
            receiver_id="agent-2",
        )
        await message_store.store_message(msg)

        # Update to delivered
        await message_store.update_message_status(msg.id, MessageStatus.DELIVERED)
        messages = await message_store.get_messages_for_agent(
            "agent-2", status=MessageStatus.DELIVERED
        )
        assert len(messages) == 1

        # Update to read
        await message_store.update_message_status(msg.id, MessageStatus.READ)
        messages = await message_store.get_messages_for_agent(
            "agent-2", status=MessageStatus.DELIVERED
        )
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_message_priority_ordering(self, message_store):
        """Test messages are ordered by priority."""
        # Send messages with different priorities
        low = AgentMessage(
            sender_id="agent-1",
            receiver_id="agent-2",
            priority=MessagePriority.LOW,
            payload={"priority": "low"},
        )
        high = AgentMessage(
            sender_id="agent-1",
            receiver_id="agent-2",
            priority=MessagePriority.HIGH,
            payload={"priority": "high"},
        )
        normal = AgentMessage(
            sender_id="agent-1",
            receiver_id="agent-2",
            priority=MessagePriority.NORMAL,
            payload={"priority": "normal"},
        )

        await message_store.store_message(low)
        await message_store.store_message(high)
        await message_store.store_message(normal)

        # Retrieve - should be ordered by priority DESC
        messages = await message_store.get_messages_for_agent("agent-2")
        priorities = [m.priority for m in messages]

        assert priorities[0] == MessagePriority.HIGH
        assert priorities[1] == MessagePriority.NORMAL
        assert priorities[2] == MessagePriority.LOW

    @pytest.mark.asyncio
    async def test_filter_by_message_type(self, message_store):
        """Test filtering messages by type."""
        task = AgentMessage(
            sender_id="agent-1",
            receiver_id="agent-2",
            message_type=MessageType.TASK,
        )
        result = AgentMessage(
            sender_id="agent-1",
            receiver_id="agent-2",
            message_type=MessageType.RESULT,
        )

        await message_store.store_message(task)
        await message_store.store_message(result)

        # Filter by TASK
        tasks = await message_store.get_messages_for_agent(
            "agent-2", message_type=MessageType.TASK
        )
        assert len(tasks) == 1
        assert tasks[0].message_type == MessageType.TASK

        # Filter by RESULT
        results = await message_store.get_messages_for_agent(
            "agent-2", message_type=MessageType.RESULT
        )
        assert len(results) == 1
        assert results[0].message_type == MessageType.RESULT

    @pytest.mark.asyncio
    async def test_large_payload_storage(self, message_store):
        """Test storing and retrieving large payloads."""
        # Create payload larger than threshold
        large_data = {"data": "x" * (LARGE_PAYLOAD_THRESHOLD + 1000)}
        payload_json = json.dumps(large_data)

        # Store
        reference = await message_store.store_large_payload(payload_json)
        assert reference.startswith("memory://")

        # Retrieve
        retrieved = await message_store.get_large_payload(reference)
        assert retrieved == payload_json
        assert json.loads(retrieved) == large_data

    @pytest.mark.asyncio
    async def test_topic_subscriptions(self, message_store):
        """Test topic subscription management."""
        # Subscribe
        result = await message_store.subscribe_to_topic("agent-1", "status_updates")
        assert result is True

        result = await message_store.subscribe_to_topic("agent-2", "status_updates")
        assert result is True

        result = await message_store.subscribe_to_topic("agent-1", "alerts")
        assert result is True

        # Get subscribers
        subs = await message_store.get_topic_subscribers("status_updates")
        assert len(subs) == 2
        assert "agent-1" in subs
        assert "agent-2" in subs

        # Unsubscribe
        result = await message_store.unsubscribe_from_topic("agent-1", "status_updates")
        assert result is True

        subs = await message_store.get_topic_subscribers("status_updates")
        assert len(subs) == 1
        assert "agent-2" in subs

    @pytest.mark.asyncio
    async def test_inbox_stats(self, message_store):
        """Test inbox statistics."""
        agent_id = "agent-stats"

        # Create various messages
        for i in range(5):
            await message_store.store_message(
                AgentMessage(
                    sender_id="sender",
                    receiver_id=agent_id,
                    message_type=MessageType.TASK if i < 3 else MessageType.QUERY,
                    priority=MessagePriority.HIGH if i == 0 else MessagePriority.NORMAL,
                )
            )

        stats = await message_store.get_inbox_stats(agent_id)

        assert stats.agent_id == agent_id
        assert stats.total_messages == 5
        assert stats.pending_messages == 5
        assert stats.by_type.get("task", 0) == 3
        assert stats.by_type.get("query", 0) == 2

    @pytest.mark.asyncio
    async def test_acknowledgment_storage(self, message_store):
        """Test storing acknowledgments."""
        ack = MessageAcknowledgment(
            message_id="msg-123",
            agent_id="agent-1",
            success=True,
        )

        result = await message_store.store_acknowledgment(ack)
        assert result is True

    @pytest.mark.asyncio
    async def test_cleanup_expired_messages(self, message_store):
        """Test cleaning up expired messages."""
        # Create expired message
        expired = AgentMessage(
            sender_id="agent-1",
            receiver_id="agent-2",
            ttl_seconds=1,
            timestamp=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        await message_store.store_message(expired)

        # Create valid message
        valid = AgentMessage(
            sender_id="agent-1",
            receiver_id="agent-2",
            ttl_seconds=3600,
        )
        await message_store.store_message(valid)

        # Cleanup
        deleted = await message_store.cleanup_expired_messages()
        assert deleted >= 1

        # Verify valid message still exists
        messages = await message_store.get_messages_for_agent("agent-2")
        # Only valid message should remain (expired filtered out on retrieval)
        valid_ids = [m.id for m in messages if not m.is_expired]
        assert valid.id in valid_ids


# =============================================================================
# COMMUNICATION SYSTEM TESTS
# =============================================================================


class TestAgentCommSystem:
    """Tests for AgentCommSystem."""

    @pytest.mark.asyncio
    async def test_send_and_receive_message(self, comm_system):
        """Test basic message send and receive."""
        # Send
        msg = await comm_system.send_message(
            sender_id="coordinator",
            receiver_id="researcher",
            message_type=MessageType.TASK,
            payload={"task": "Research OAuth patterns"},
        )

        assert msg.id is not None
        assert msg.sender_id == "coordinator"
        assert msg.receiver_id == "researcher"

        # Receive
        messages = await comm_system.receive_messages("researcher")
        assert len(messages) == 1
        assert messages[0].id == msg.id
        assert messages[0].payload["task"] == "Research OAuth patterns"

    @pytest.mark.asyncio
    async def test_send_task(self, comm_system):
        """Test convenience method for sending tasks."""
        msg = await comm_system.send_task(
            sender_id="coordinator",
            receiver_id="coder",
            task_description="Implement authentication",
            task_context={"tech_stack": ["Python", "FastAPI"]},
        )

        assert msg.message_type == MessageType.TASK
        assert msg.payload["task"] == "Implement authentication"
        assert msg.payload["context"]["tech_stack"] == ["Python", "FastAPI"]

    @pytest.mark.asyncio
    async def test_send_result(self, comm_system):
        """Test convenience method for sending results."""
        msg = await comm_system.send_result(
            sender_id="researcher",
            receiver_id="coordinator",
            task_id="task-123",
            result={"findings": ["OAuth 2.0", "JWT", "PKCE"]},
            success=True,
        )

        assert msg.message_type == MessageType.RESULT
        assert msg.correlation_id == "task-123"
        assert msg.payload["success"] is True
        assert len(msg.payload["result"]["findings"]) == 3

    @pytest.mark.asyncio
    async def test_send_query(self, comm_system):
        """Test convenience method for sending queries."""
        msg = await comm_system.send_query(
            sender_id="coder",
            receiver_id="researcher",
            query="What are the best OAuth libraries for Python?",
        )

        assert msg.message_type == MessageType.QUERY
        assert "OAuth libraries" in msg.payload["query"]

    @pytest.mark.asyncio
    async def test_large_payload_handling(self, comm_system):
        """Test automatic handling of large payloads."""
        # Create large payload
        large_result = {
            "research_output": "x" * (LARGE_PAYLOAD_THRESHOLD + 5000),
            "findings": ["finding " * 1000 for _ in range(10)],
        }

        msg = await comm_system.send_result(
            sender_id="researcher",
            receiver_id="coordinator",
            task_id="task-large",
            result=large_result,
        )

        # Message should have reference
        assert msg.reference is not None
        assert msg.reference.startswith("memory://")
        assert "_reference" in msg.payload
        assert "_summary" in msg.payload

        # Fetch full payload
        full_payload = await comm_system.fetch_payload(msg)
        assert full_payload["result"] == large_result

    @pytest.mark.asyncio
    async def test_broadcast_message(self, comm_system):
        """Test broadcasting messages to topics."""
        # Subscribe agents to topic
        await comm_system.subscribe_to_topic("agent-1", "status_updates")
        await comm_system.subscribe_to_topic("agent-2", "status_updates")
        await comm_system.subscribe_to_topic("agent-3", "other_topic")

        # Broadcast
        msg = await comm_system.broadcast(
            sender_id="coordinator",
            topic="status_updates",
            payload={"status": "research_complete", "progress": 0.75},
        )

        assert msg.is_broadcast
        assert msg.topic == "status_updates"

        # Check subscribers receive
        messages_1 = await comm_system.receive_messages("agent-1")
        messages_2 = await comm_system.receive_messages("agent-2")
        messages_3 = await comm_system.receive_messages("agent-3")

        # agent-1 and agent-2 subscribed to status_updates
        assert any(m.topic == "status_updates" for m in messages_1)
        assert any(m.topic == "status_updates" for m in messages_2)
        # agent-3 not subscribed
        assert not any(m.topic == "status_updates" for m in messages_3)

    @pytest.mark.asyncio
    async def test_message_acknowledgment(self, comm_system):
        """Test message acknowledgment flow."""
        # Send message
        msg = await comm_system.send_message(
            sender_id="coordinator",
            receiver_id="worker",
            message_type=MessageType.TASK,
            payload={"task": "process data"},
        )

        # Receive
        messages = await comm_system.receive_messages("worker")
        received = messages[0]

        # Acknowledge
        await comm_system.acknowledge(
            message_id=received.id,
            agent_id="worker",
            success=True,
        )

        # Verify metrics
        assert comm_system.metrics["acknowledgments"] >= 1

    @pytest.mark.asyncio
    async def test_message_handler_registration(self, comm_system):
        """Test registering and processing message handlers."""
        processed_tasks = []

        async def task_handler(msg: AgentMessage):
            processed_tasks.append(msg.payload["task"])

        # Register handler
        comm_system.register_handler(MessageType.TASK, task_handler)

        # Send tasks
        await comm_system.send_task("coordinator", "processor", "Task 1")
        await comm_system.send_task("coordinator", "processor", "Task 2")

        # Process messages
        count = await comm_system.process_messages("processor")

        assert count == 2
        assert "Task 1" in processed_tasks
        assert "Task 2" in processed_tasks

    @pytest.mark.asyncio
    async def test_agent_registration(self, comm_system):
        """Test agent registration and unregistration."""
        # Register agent
        comm_system.register_agent("test-agent", topics=["updates", "alerts"])

        assert "test-agent" in comm_system._registered_agents

        # Unregister
        comm_system.unregister_agent("test-agent")
        assert "test-agent" not in comm_system._registered_agents

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, comm_system):
        """Test that metrics are tracked correctly."""
        # Send messages
        await comm_system.send_message(
            sender_id="a1", receiver_id="a2",
            message_type=MessageType.TASK, payload={}
        )
        await comm_system.send_message(
            sender_id="a1", receiver_id="a2",
            message_type=MessageType.TASK, payload={}
        )

        # Broadcast
        await comm_system.broadcast(
            sender_id="a1", topic="test", payload={}
        )

        # Receive
        await comm_system.receive_messages("a2")

        metrics = comm_system.get_metrics()

        assert metrics["messages_sent"] >= 2
        assert metrics["broadcasts_sent"] >= 1
        assert metrics["messages_received"] >= 2

    @pytest.mark.asyncio
    async def test_inbox_stats(self, comm_system):
        """Test getting inbox statistics."""
        agent_id = "stats-agent"

        # Send various messages
        for _ in range(3):
            await comm_system.send_task("coordinator", agent_id, "Task")
        for _ in range(2):
            await comm_system.send_query("other", agent_id, "Query?")

        stats = await comm_system.get_inbox_stats(agent_id)

        assert isinstance(stats, InboxStats)
        assert stats.agent_id == agent_id
        assert stats.total_messages == 5


# =============================================================================
# EVENT EMISSION TESTS
# =============================================================================


class TestEventEmission:
    """Tests for event emission during communication."""

    @pytest.mark.asyncio
    async def test_message_sent_event(self, comm_system, event_bus):
        """Test that MessageSentEvent is emitted."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        event_bus.subscribe("MessageSentEvent", handler)

        await comm_system.send_message(
            sender_id="a1",
            receiver_id="a2",
            message_type=MessageType.TASK,
            payload={"test": True},
        )

        # Allow event to propagate
        await asyncio.sleep(0.1)

        assert len(received_events) >= 1
        assert isinstance(received_events[0], MessageSentEvent)
        assert received_events[0].sender_id == "a1"
        assert received_events[0].receiver_id == "a2"

    @pytest.mark.asyncio
    async def test_message_received_event(self, comm_system, event_bus):
        """Test that MessageReceivedEvent is emitted."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        event_bus.subscribe("MessageReceivedEvent", handler)

        await comm_system.send_message(
            sender_id="a1",
            receiver_id="a2",
            message_type=MessageType.TASK,
            payload={},
        )

        await comm_system.receive_messages("a2")

        await asyncio.sleep(0.1)

        assert len(received_events) >= 1
        assert isinstance(received_events[0], MessageReceivedEvent)

    @pytest.mark.asyncio
    async def test_broadcast_sent_event(self, comm_system, event_bus):
        """Test that BroadcastSentEvent is emitted."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        event_bus.subscribe("BroadcastSentEvent", handler)

        await comm_system.subscribe_to_topic("agent-1", "test_topic")
        await comm_system.subscribe_to_topic("agent-2", "test_topic")

        await comm_system.broadcast(
            sender_id="coordinator",
            topic="test_topic",
            payload={"status": "active"},
        )

        await asyncio.sleep(0.1)

        assert len(received_events) >= 1
        event = received_events[0]
        assert isinstance(event, BroadcastSentEvent)
        assert event.topic == "test_topic"
        assert event.recipient_count == 2

    @pytest.mark.asyncio
    async def test_acknowledgment_event(self, comm_system, event_bus):
        """Test that MessageAcknowledgedEvent is emitted."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        event_bus.subscribe("MessageAcknowledgedEvent", handler)

        msg = await comm_system.send_message(
            sender_id="a1",
            receiver_id="a2",
            message_type=MessageType.TASK,
            payload={},
        )

        await comm_system.acknowledge(
            message_id=msg.id,
            agent_id="a2",
            success=True,
        )

        await asyncio.sleep(0.1)

        assert len(received_events) >= 1
        event = received_events[0]
        assert isinstance(event, MessageAcknowledgedEvent)
        assert event.message_id == msg.id
        assert event.success is True


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for complete communication flows."""

    @pytest.mark.asyncio
    async def test_task_result_flow(self, comm_system):
        """Test complete task assignment and result flow."""
        # Coordinator sends task
        task_msg = await comm_system.send_task(
            sender_id="coordinator",
            receiver_id="researcher",
            task_description="Research authentication patterns",
            task_context={"domain": "security"},
        )

        # Researcher receives task
        inbox = await comm_system.receive_messages("researcher")
        assert len(inbox) == 1
        task = inbox[0]
        assert task.message_type == MessageType.TASK

        # Researcher acknowledges
        await comm_system.acknowledge(task.id, "researcher", success=True)

        # Researcher sends result
        result_msg = await comm_system.send_result(
            sender_id="researcher",
            receiver_id="coordinator",
            task_id=task.correlation_id or task.id,
            result={
                "patterns": ["OAuth2", "OIDC", "SAML"],
                "recommendation": "OAuth2 with PKCE",
            },
            success=True,
        )

        # Coordinator receives result
        results = await comm_system.receive_messages(
            "coordinator", message_type=MessageType.RESULT
        )
        assert len(results) == 1
        assert results[0].payload["success"] is True

    @pytest.mark.asyncio
    async def test_query_response_flow(self, comm_system):
        """Test query and response pattern."""
        # Agent sends query
        query = await comm_system.send_query(
            sender_id="coder",
            receiver_id="architect",
            query="What database should we use for caching?",
            context={"requirements": ["fast", "distributed"]},
        )

        # Architect receives and responds
        queries = await comm_system.receive_messages(
            "architect", message_type=MessageType.QUERY
        )
        assert len(queries) == 1

        # Send response
        response = await comm_system.send_message(
            sender_id="architect",
            receiver_id="coder",
            message_type=MessageType.RESPONSE,
            payload={
                "answer": "Redis is recommended for distributed caching",
                "alternatives": ["Memcached", "Hazelcast"],
            },
            reply_to=query.id,
        )

        # Coder receives response
        responses = await comm_system.receive_messages(
            "coder", message_type=MessageType.RESPONSE
        )
        assert len(responses) == 1
        assert responses[0].reply_to == query.id

    @pytest.mark.asyncio
    async def test_broadcast_coordination_flow(self, comm_system):
        """Test broadcast-based swarm coordination."""
        # Register workers
        workers = ["worker-1", "worker-2", "worker-3"]
        for w in workers:
            comm_system.register_agent(w, topics=["task_distribution"])

        # Coordinator broadcasts task availability
        await comm_system.broadcast(
            sender_id="coordinator",
            topic="task_distribution",
            payload={
                "event": "new_tasks_available",
                "count": 10,
                "priority": "high",
            },
        )

        # All workers should receive
        for w in workers:
            messages = await comm_system.receive_messages(w)
            broadcasts = [m for m in messages if m.topic == "task_distribution"]
            assert len(broadcasts) >= 1

    @pytest.mark.asyncio
    async def test_large_research_output_flow(self, comm_system):
        """Test handling large research outputs with references."""
        # Simulate large research output
        large_output = {
            "analysis": "Comprehensive security analysis " * 2000,
            "findings": [f"Finding {i}: " + "detail " * 500 for i in range(20)],
            "recommendations": ["rec " * 200 for _ in range(10)],
            "code_samples": ["code " * 300 for _ in range(15)],
        }

        # Send result
        result = await comm_system.send_result(
            sender_id="researcher",
            receiver_id="synthesizer",
            task_id="research-task-1",
            result=large_output,
        )

        # Should be stored as reference
        assert result.reference is not None
        assert "_reference" in result.payload

        # Synthesizer receives summary
        messages = await comm_system.receive_messages("synthesizer")
        msg = messages[0]

        # Summary should be small
        assert "_summary" in msg.payload
        assert len(json.dumps(msg.payload)) < LARGE_PAYLOAD_THRESHOLD

        # Fetch full payload when needed
        full_payload = await comm_system.fetch_payload(msg)
        assert full_payload["result"]["analysis"] == large_output["analysis"]
        assert len(full_payload["result"]["findings"]) == 20

    @pytest.mark.asyncio
    async def test_concurrent_message_handling(self, comm_system):
        """Test handling concurrent message sends and receives."""
        sender_count = 5
        messages_per_sender = 10

        async def send_messages(sender_id: str):
            for i in range(messages_per_sender):
                await comm_system.send_message(
                    sender_id=sender_id,
                    receiver_id="collector",
                    message_type=MessageType.STATUS,
                    payload={"sender": sender_id, "index": i},
                )

        # Send concurrently
        await asyncio.gather(*[
            send_messages(f"sender-{i}") for i in range(sender_count)
        ])

        # Receive all
        all_messages = await comm_system.receive_messages(
            "collector", limit=sender_count * messages_per_sender
        )

        assert len(all_messages) == sender_count * messages_per_sender

        # Verify all senders represented
        senders = set(m.payload["sender"] for m in all_messages)
        assert len(senders) == sender_count


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
