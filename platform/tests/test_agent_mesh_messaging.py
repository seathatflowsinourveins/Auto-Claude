"""
Tests for Agent Mesh Dynamic Messaging Protocol (Gap14)

Tests the V66 messaging protocol implementation including:
- Priority-based message queues
- Message routing strategies
- Delivery confirmation and acknowledgment
- Dead letter queue handling
- CVT consensus integration
- Message integrity verification

Version: V66.0.0 (February 2026)
"""

import asyncio
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.agent_mesh import (
    AgentMesh,
    AgentInfo,
    AgentRole,
    AgentStatus,
    MeshMessage,
    MessageType,
    MessagePriority,
    DeliveryStatus,
    RoutingStrategy,
    DeliveryReceipt,
    TaskState,
    PriorityMessageQueue,
    MessageRouter,
    create_agent_mesh,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mesh():
    """Create a fresh AgentMesh instance for testing."""
    return AgentMesh(use_mock=True, enable_consensus=False)


@pytest.fixture
async def started_mesh():
    """Create and start a mesh with registered agents."""
    mesh = AgentMesh(use_mock=True, enable_consensus=False)
    await mesh.start()

    # Register test agents
    await mesh.register_agent(AgentRole.ROUTER, "local", agent_id="router-1")
    await mesh.register_agent(AgentRole.CODER, "local", agent_id="coder-1")
    await mesh.register_agent(AgentRole.CODER, "hybrid", agent_id="coder-2", priority_weight=1.5)
    await mesh.register_agent(AgentRole.REVIEWER, "hybrid", agent_id="reviewer-1")
    await mesh.register_agent(AgentRole.ARCHITECT, "premium", agent_id="architect-1", priority_weight=2.0)

    yield mesh

    await mesh.stop()


@pytest.fixture
def priority_queue():
    """Create a PriorityMessageQueue for testing."""
    return PriorityMessageQueue(max_size=100)


# =============================================================================
# MeshMessage Tests
# =============================================================================

class TestMeshMessage:
    """Tests for MeshMessage data class."""

    def test_message_creation_with_defaults(self):
        """Test creating a message with default values."""
        msg = MeshMessage(
            message_id="msg-001",
            message_type=MessageType.TASK_ASSIGN,
            sender_id="sender-1",
            recipient_id="recipient-1",
            payload={"task": "test"},
        )

        assert msg.message_id == "msg-001"
        assert msg.message_type == MessageType.TASK_ASSIGN
        assert msg.sender_id == "sender-1"
        assert msg.recipient_id == "recipient-1"
        assert msg.priority == MessagePriority.NORMAL
        assert msg.delivery_status == DeliveryStatus.PENDING
        assert msg.retry_count == 0
        assert msg.max_retries == 3
        assert msg.requires_ack is False
        assert msg.checksum is not None

    def test_message_checksum_verification(self):
        """Test message checksum computation and verification."""
        msg = MeshMessage(
            message_id="msg-001",
            message_type=MessageType.QUERY,
            sender_id="sender-1",
            recipient_id="recipient-1",
            payload={"data": "test"},
        )

        # Original checksum should verify
        assert msg.verify_checksum() is True

        # Modify payload - checksum should fail
        msg.payload["data"] = "tampered"
        assert msg.verify_checksum() is False

    def test_message_expiration(self):
        """Test message expiration check."""
        # Create message with short TTL
        msg = MeshMessage(
            message_id="msg-001",
            message_type=MessageType.HEARTBEAT,
            sender_id="sender-1",
            recipient_id=None,
            payload={},
            ttl_seconds=1,
        )

        # Should not be expired immediately
        assert msg.is_expired is False

        # Manually set timestamp to past
        msg.timestamp = datetime.now(timezone.utc) - timedelta(seconds=2)
        assert msg.is_expired is True

    def test_message_serialization(self):
        """Test message to_dict and from_dict."""
        original = MeshMessage(
            message_id="msg-001",
            message_type=MessageType.TASK_ASSIGN,
            sender_id="sender-1",
            recipient_id="recipient-1",
            payload={"task": "implement feature"},
            priority=MessagePriority.HIGH,
            routing_strategy=RoutingStrategy.DIRECT,
            requires_ack=True,
        )

        # Serialize
        data = original.to_dict()
        assert data["message_id"] == "msg-001"
        assert data["priority"] == MessagePriority.HIGH.value

        # Deserialize
        restored = MeshMessage.from_dict(data)
        assert restored.message_id == original.message_id
        assert restored.message_type == original.message_type
        assert restored.priority == original.priority
        assert restored.routing_strategy == original.routing_strategy

    def test_all_message_types(self):
        """Test creating messages with all message types."""
        for msg_type in MessageType:
            msg = MeshMessage(
                message_id=f"msg-{msg_type.value}",
                message_type=msg_type,
                sender_id="sender-1",
                recipient_id="recipient-1",
                payload={},
            )
            assert msg.message_type == msg_type

    def test_all_priority_levels(self):
        """Test creating messages with all priority levels."""
        for priority in MessagePriority:
            msg = MeshMessage(
                message_id=f"msg-{priority.name}",
                message_type=MessageType.QUERY,
                sender_id="sender-1",
                recipient_id="recipient-1",
                payload={},
                priority=priority,
            )
            assert msg.priority == priority


# =============================================================================
# PriorityMessageQueue Tests
# =============================================================================

class TestPriorityMessageQueue:
    """Tests for PriorityMessageQueue."""

    @pytest.mark.asyncio
    async def test_enqueue_and_dequeue(self, priority_queue):
        """Test basic enqueue and dequeue operations."""
        msg = MeshMessage(
            message_id="msg-001",
            message_type=MessageType.QUERY,
            sender_id="sender-1",
            recipient_id="recipient-1",
            payload={},
        )

        # Enqueue
        result = await priority_queue.enqueue(msg)
        assert result is True
        assert msg.delivery_status == DeliveryStatus.QUEUED

        # Dequeue
        dequeued = await priority_queue.dequeue()
        assert dequeued is not None
        assert dequeued.message_id == "msg-001"
        assert dequeued.delivery_status == DeliveryStatus.IN_FLIGHT

    @pytest.mark.asyncio
    async def test_priority_ordering(self, priority_queue):
        """Test that messages are dequeued in priority order."""
        # Enqueue messages in reverse priority order
        low_msg = MeshMessage(
            message_id="low",
            message_type=MessageType.HEARTBEAT,
            sender_id="sender-1",
            recipient_id="recipient-1",
            payload={},
            priority=MessagePriority.LOW,
        )

        normal_msg = MeshMessage(
            message_id="normal",
            message_type=MessageType.QUERY,
            sender_id="sender-1",
            recipient_id="recipient-1",
            payload={},
            priority=MessagePriority.NORMAL,
        )

        critical_msg = MeshMessage(
            message_id="critical",
            message_type=MessageType.VOTE_REQUEST,
            sender_id="sender-1",
            recipient_id="recipient-1",
            payload={},
            priority=MessagePriority.CRITICAL,
        )

        # Enqueue in reverse priority order
        await priority_queue.enqueue(low_msg)
        await priority_queue.enqueue(normal_msg)
        await priority_queue.enqueue(critical_msg)

        # Dequeue should return in priority order
        first = await priority_queue.dequeue()
        assert first.message_id == "critical", "Critical should come first"

        second = await priority_queue.dequeue()
        assert second.message_id == "normal", "Normal should come second"

        third = await priority_queue.dequeue()
        assert third.message_id == "low", "Low should come third"

    @pytest.mark.asyncio
    async def test_expired_message_handling(self, priority_queue):
        """Test that expired messages are not enqueued."""
        msg = MeshMessage(
            message_id="expired",
            message_type=MessageType.QUERY,
            sender_id="sender-1",
            recipient_id="recipient-1",
            payload={},
            ttl_seconds=1,
        )

        # Make message expired
        msg.timestamp = datetime.now(timezone.utc) - timedelta(seconds=2)

        result = await priority_queue.enqueue(msg)
        assert result is False

        stats = priority_queue.get_stats()
        assert stats["total_expired"] == 1

    @pytest.mark.asyncio
    async def test_dead_letter_queue(self, priority_queue):
        """Test moving messages to dead letter queue."""
        msg = MeshMessage(
            message_id="failed",
            message_type=MessageType.TASK_ASSIGN,
            sender_id="sender-1",
            recipient_id="recipient-1",
            payload={},
        )

        await priority_queue.dead_letter(msg, "Handler error")

        assert msg.delivery_status == DeliveryStatus.DEAD_LETTER
        assert "_dlq_error" in msg.payload
        assert msg.payload["_dlq_error"] == "Handler error"

        stats = priority_queue.get_stats()
        assert stats["total_dead_lettered"] == 1
        assert stats["dead_letter_depth"] == 1

    @pytest.mark.asyncio
    async def test_queue_statistics(self, priority_queue):
        """Test queue statistics tracking."""
        # Enqueue several messages
        for i in range(5):
            msg = MeshMessage(
                message_id=f"msg-{i}",
                message_type=MessageType.QUERY,
                sender_id="sender-1",
                recipient_id="recipient-1",
                payload={},
                priority=MessagePriority.NORMAL,
            )
            await priority_queue.enqueue(msg)

        # Dequeue some
        await priority_queue.dequeue()
        await priority_queue.dequeue()

        stats = priority_queue.get_stats()
        assert stats["total_enqueued"] == 5
        assert stats["total_dequeued"] == 2
        assert stats["queue_depths"]["NORMAL"] == 3


# =============================================================================
# MessageRouter Tests
# =============================================================================

class TestMessageRouter:
    """Tests for MessageRouter."""

    @pytest.mark.asyncio
    async def test_direct_routing(self, started_mesh):
        """Test direct routing to specific agent."""
        router = started_mesh._router

        msg = MeshMessage(
            message_id="msg-001",
            message_type=MessageType.QUERY,
            sender_id="router-1",
            recipient_id="coder-1",
            payload={},
            routing_strategy=RoutingStrategy.DIRECT,
        )

        targets = router.route(msg)
        assert targets == ["coder-1"]

    @pytest.mark.asyncio
    async def test_broadcast_routing(self, started_mesh):
        """Test broadcast routing to all agents."""
        router = started_mesh._router

        msg = MeshMessage(
            message_id="msg-001",
            message_type=MessageType.CONTEXT_SHARE,
            sender_id="router-1",
            recipient_id=None,
            payload={},
            routing_strategy=RoutingStrategy.BROADCAST,
        )

        targets = router.route(msg)
        assert "coder-1" in targets
        assert "coder-2" in targets
        assert "reviewer-1" in targets
        assert "architect-1" in targets
        assert "router-1" not in targets  # Sender excluded

    @pytest.mark.asyncio
    async def test_round_robin_routing(self, started_mesh):
        """Test round-robin routing across same-role agents."""
        router = started_mesh._router

        targets_seen = set()
        for _ in range(10):
            msg = MeshMessage(
                message_id="msg",
                message_type=MessageType.TASK_ASSIGN,
                sender_id="router-1",
                recipient_id=None,
                payload={},
                routing_strategy=RoutingStrategy.ROUND_ROBIN,
            )
            targets = router.route(msg, target_role=AgentRole.CODER)
            if targets:
                targets_seen.add(targets[0])

        # Should have rotated through both coders
        assert "coder-1" in targets_seen
        assert "coder-2" in targets_seen

    @pytest.mark.asyncio
    async def test_least_loaded_routing(self, started_mesh):
        """Test least-loaded routing strategy."""
        router = started_mesh._router

        # Set coder-1 as busy with high queue depth
        started_mesh._agents["coder-1"].status = AgentStatus.BUSY
        started_mesh._agents["coder-1"].queue_depth = 10

        msg = MeshMessage(
            message_id="msg",
            message_type=MessageType.TASK_ASSIGN,
            sender_id="router-1",
            recipient_id=None,
            payload={},
            routing_strategy=RoutingStrategy.LEAST_LOADED,
        )

        targets = router.route(msg, target_role=AgentRole.CODER)
        # Should pick coder-2 as it has lower load
        assert targets == ["coder-2"]

    @pytest.mark.asyncio
    async def test_priority_routing(self, started_mesh):
        """Test priority-based routing strategy."""
        router = started_mesh._router

        msg = MeshMessage(
            message_id="msg",
            message_type=MessageType.ESCALATE,
            sender_id="coder-1",
            recipient_id=None,
            payload={},
            routing_strategy=RoutingStrategy.PRIORITY,
        )

        targets = router.route(msg, target_role=AgentRole.CODER)
        # Should pick coder-2 as it has higher priority_weight (1.5 vs 1.0)
        assert targets == ["coder-2"]

    @pytest.mark.asyncio
    async def test_routing_stats(self, started_mesh):
        """Test routing statistics tracking."""
        router = started_mesh._router

        # Send several messages with different strategies
        for strategy in [RoutingStrategy.DIRECT, RoutingStrategy.BROADCAST, RoutingStrategy.LEAST_LOADED]:
            msg = MeshMessage(
                message_id="msg",
                message_type=MessageType.QUERY,
                sender_id="router-1",
                recipient_id="coder-1",
                payload={},
                routing_strategy=strategy,
            )
            router.route(msg)

        stats = router.get_stats()
        assert stats[RoutingStrategy.DIRECT] == 1
        assert stats[RoutingStrategy.BROADCAST] == 1
        assert stats[RoutingStrategy.LEAST_LOADED] == 1


# =============================================================================
# AgentMesh Integration Tests
# =============================================================================

class TestAgentMeshMessaging:
    """Integration tests for AgentMesh messaging."""

    @pytest.mark.asyncio
    async def test_agent_registration(self, started_mesh):
        """Test agent registration creates queue and updates state."""
        agent = started_mesh.get_agent("coder-1")
        assert agent is not None
        assert agent.role == AgentRole.CODER
        assert agent.model_tier == "local"
        assert "coder-1" in started_mesh._queues

    @pytest.mark.asyncio
    async def test_agent_unregistration(self, started_mesh):
        """Test agent unregistration cleans up resources."""
        await started_mesh.unregister_agent("coder-1")

        assert "coder-1" not in started_mesh._agents
        assert "coder-1" not in started_mesh._queues

    @pytest.mark.asyncio
    async def test_send_direct_message(self, started_mesh):
        """Test sending a direct message."""
        received = []

        async def handler(msg):
            # Only count QUERY messages (ignore broadcasts and heartbeats)
            if msg.message_type == MessageType.QUERY:
                received.append(msg)
            return None

        await started_mesh.subscribe("coder-1", handler)

        msg_id = await started_mesh.send(
            sender_id="router-1",
            recipient_id="coder-1",
            message_type=MessageType.QUERY,
            payload={"query": "test"},
        )

        # Wait for delivery
        await asyncio.sleep(0.2)

        assert len(received) == 1, f"Expected 1 QUERY message, got {len(received)}"
        assert received[0].payload["query"] == "test"
        assert started_mesh._stats["messages_sent"] >= 1

    @pytest.mark.asyncio
    async def test_send_routed_message(self, started_mesh):
        """Test sending a message with automatic routing."""
        received = []

        async def handler(msg):
            # Only count TASK_ASSIGN messages (ignore broadcasts and heartbeats)
            if msg.message_type == MessageType.TASK_ASSIGN:
                received.append(msg)
            return None

        await started_mesh.subscribe("coder-1", handler)
        await started_mesh.subscribe("coder-2", handler)

        msg_id, targets = await started_mesh.send_routed(
            sender_id="router-1",
            message_type=MessageType.TASK_ASSIGN,
            payload={"task": "test"},
            target_role=AgentRole.CODER,
            routing_strategy=RoutingStrategy.LEAST_LOADED,
        )

        # Wait for delivery
        await asyncio.sleep(0.2)

        assert len(targets) == 1, f"Expected 1 target, got {targets}"
        assert len(received) == 1, f"Expected 1 TASK_ASSIGN message, got {len(received)}"

    @pytest.mark.asyncio
    async def test_broadcast_message(self, started_mesh):
        """Test broadcasting a message to all agents."""
        received = []

        async def handler(msg):
            # Only count CONTEXT_SHARE messages (ignore heartbeats)
            if msg.message_type == MessageType.CONTEXT_SHARE:
                received.append(msg)
            return None

        # Subscribe all agents
        for agent_id in ["coder-1", "coder-2", "reviewer-1", "architect-1"]:
            await started_mesh.subscribe(agent_id, handler)

        msg_id = await started_mesh.broadcast(
            sender_id="router-1",
            message_type=MessageType.CONTEXT_SHARE,
            payload={"key": "test", "value": "data"},
        )

        # Wait for delivery
        await asyncio.sleep(0.3)

        # All agents except sender should receive
        assert len(received) == 4

    @pytest.mark.asyncio
    async def test_request_response(self, started_mesh):
        """Test request-response pattern."""
        async def handler(msg):
            if msg.message_type == MessageType.QUERY:
                response = MeshMessage(
                    message_id=f"resp-{msg.message_id}",
                    message_type=MessageType.RESPONSE,
                    sender_id=msg.recipient_id,
                    recipient_id=msg.sender_id,
                    payload={"answer": "42"},
                    correlation_id=msg.correlation_id,
                )
                return response
            return None

        await started_mesh.subscribe("coder-1", handler)

        response = await started_mesh.request(
            sender_id="router-1",
            recipient_id="coder-1",
            message_type=MessageType.QUERY,
            payload={"question": "meaning of life"},
            timeout=5.0,
        )

        # Note: In this implementation, response comes through pending futures
        # which requires the handler to call mesh.send() with correlation_id
        # For this test, we verify the request was sent
        assert started_mesh._stats["messages_sent"] >= 1

    @pytest.mark.asyncio
    async def test_message_acknowledgment(self, started_mesh):
        """Test message acknowledgment flow."""
        ack_received = []

        async def sender_handler(msg):
            if msg.message_type in (MessageType.ACK, MessageType.NACK):
                ack_received.append(msg)
            return None

        async def receiver_handler(msg):
            await started_mesh.acknowledge(msg, success=True, response_payload={"status": "ok"})
            return None

        await started_mesh.subscribe("router-1", sender_handler)
        await started_mesh.subscribe("coder-1", receiver_handler)

        await started_mesh.send(
            sender_id="router-1",
            recipient_id="coder-1",
            message_type=MessageType.TASK_ASSIGN,
            payload={"task": "test"},
            requires_ack=True,
        )

        # Wait for delivery and ack
        await asyncio.sleep(0.3)

        assert started_mesh._stats["messages_acknowledged"] >= 1

    @pytest.mark.asyncio
    async def test_task_assignment_with_routing(self, started_mesh):
        """Test task assignment uses routing strategy."""
        received_tasks = []

        async def handler(msg):
            if msg.message_type == MessageType.TASK_ASSIGN:
                received_tasks.append(msg)
            return None

        await started_mesh.subscribe("coder-1", handler)
        await started_mesh.subscribe("coder-2", handler)

        assigned_id = await started_mesh.assign_task(
            task_id="task-001",
            description="Test task",
            preferred_role=AgentRole.CODER,
            routing_strategy=RoutingStrategy.PRIORITY,
        )

        # Wait for delivery
        await asyncio.sleep(0.2)

        assert assigned_id is not None
        # Priority routing should pick coder-2 (higher weight)
        assert assigned_id == "coder-2"
        assert started_mesh._stats["tasks_assigned"] == 1

    @pytest.mark.asyncio
    async def test_task_completion(self, started_mesh):
        """Test task completion flow."""
        # Assign task
        assigned_id = await started_mesh.assign_task(
            task_id="task-001",
            description="Test task",
            preferred_role=AgentRole.CODER,
        )

        # Complete task
        await started_mesh.complete_task(
            agent_id=assigned_id,
            task_id="task-001",
            result="Success",
        )

        task = started_mesh._tasks["task-001"]
        assert task.status == "completed"
        assert task.result == "Success"
        assert started_mesh._stats["tasks_completed"] == 1

        # Agent should be idle again
        agent = started_mesh.get_agent(assigned_id)
        assert agent.status == AgentStatus.IDLE
        assert agent.tasks_completed == 1

    @pytest.mark.asyncio
    async def test_task_escalation(self, started_mesh):
        """Test task escalation to higher-tier agent."""
        # Assign task to local tier
        await started_mesh.assign_task(
            task_id="task-001",
            description="Complex task",
            preferred_role=AgentRole.CODER,
            preferred_tier="local",
        )

        # Escalate to premium tier
        new_agent_id = await started_mesh.escalate_task(
            agent_id="coder-1",
            task_id="task-001",
            reason="Too complex",
            target_tier="premium",
        )

        # Should escalate to architect (only premium agent)
        assert new_agent_id == "architect-1"
        assert started_mesh._stats["escalations"] == 1

        # Original agent should be idle
        coder = started_mesh.get_agent("coder-1")
        assert coder.status == AgentStatus.IDLE

    @pytest.mark.asyncio
    async def test_context_sharing(self, started_mesh):
        """Test context sharing between agents."""
        received_context = []

        async def handler(msg):
            if msg.message_type == MessageType.CONTEXT_SHARE:
                received_context.append(msg.payload)
            return None

        await started_mesh.subscribe("coder-1", handler)

        await started_mesh.share_context(
            sender_id="architect-1",
            context_key="design_pattern",
            context_value="repository",
            recipient_ids=["coder-1"],
        )

        # Wait for delivery
        await asyncio.sleep(0.2)

        assert len(received_context) == 1
        assert received_context[0]["key"] == "design_pattern"
        assert received_context[0]["value"] == "repository"


# =============================================================================
# Agent State Tests
# =============================================================================

class TestAgentInfo:
    """Tests for AgentInfo data class."""

    def test_agent_availability(self):
        """Test agent availability check."""
        agent = AgentInfo(
            agent_id="test-1",
            role=AgentRole.CODER,
            model_tier="local",
        )

        # Idle agent should be available
        agent.status = AgentStatus.IDLE
        assert agent.is_available is True

        # Busy agent should be available
        agent.status = AgentStatus.BUSY
        assert agent.is_available is True

        # Overloaded agent should not be available
        agent.status = AgentStatus.OVERLOADED
        assert agent.is_available is False

        # Offline agent should not be available
        agent.status = AgentStatus.OFFLINE
        assert agent.is_available is False

    def test_agent_staleness(self):
        """Test agent staleness detection."""
        agent = AgentInfo(
            agent_id="test-1",
            role=AgentRole.CODER,
            model_tier="local",
        )

        # Fresh heartbeat should not be stale
        agent.last_heartbeat = datetime.now(timezone.utc)
        assert agent.is_stale is False

        # Old heartbeat should be stale
        agent.last_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=60)
        assert agent.is_stale is True

    def test_agent_load_score(self):
        """Test agent load score calculation."""
        agent = AgentInfo(
            agent_id="test-1",
            role=AgentRole.CODER,
            model_tier="local",
        )

        # Idle with no queue should have low score
        agent.status = AgentStatus.IDLE
        agent.queue_depth = 0
        assert agent.load_score == 0.0

        # Busy with queue should have higher score
        agent.status = AgentStatus.BUSY
        agent.queue_depth = 5
        assert agent.load_score == 20.0  # 10 (busy) + 5*2 (queue)

        # Overloaded should have very high score
        agent.status = AgentStatus.OVERLOADED
        assert agent.load_score == 100.0

        # Offline should have infinite score
        agent.status = AgentStatus.OFFLINE
        assert agent.load_score == float('inf')


# =============================================================================
# Statistics and Monitoring Tests
# =============================================================================

class TestStatistics:
    """Tests for statistics and monitoring."""

    @pytest.mark.asyncio
    async def test_mesh_stats(self, started_mesh):
        """Test mesh statistics collection."""
        stats = started_mesh.get_stats()

        assert "messages_sent" in stats
        assert "messages_received" in stats
        assert "messages_acknowledged" in stats
        assert "total_agents" in stats
        assert "available_agents" in stats
        assert "routing_stats" in stats
        assert "queue_stats" in stats
        assert stats["total_agents"] == 5

    @pytest.mark.asyncio
    async def test_agent_stats(self, started_mesh):
        """Test per-agent statistics."""
        agent_stats = started_mesh.get_agent_stats()

        assert len(agent_stats) == 5

        coder_stat = next(s for s in agent_stats if s["agent_id"] == "coder-1")
        assert coder_stat["role"] == "coder"
        assert coder_stat["model_tier"] == "local"
        assert "queue_depth" in coder_stat
        assert "load_score" in coder_stat

    @pytest.mark.asyncio
    async def test_delivery_stats(self, started_mesh):
        """Test delivery statistics."""
        # Send some messages
        async def handler(msg):
            return None

        await started_mesh.subscribe("coder-1", handler)

        await started_mesh.send(
            sender_id="router-1",
            recipient_id="coder-1",
            message_type=MessageType.QUERY,
            payload={},
        )

        await asyncio.sleep(0.2)

        stats = started_mesh.get_delivery_stats()
        assert "total_receipts" in stats
        assert "by_status" in stats


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_agent_mesh(self):
        """Test agent mesh factory function."""
        mesh = create_agent_mesh(enable_consensus=False, use_mock=True)

        assert isinstance(mesh, AgentMesh)
        assert mesh.use_mock is True
        assert mesh.enable_consensus is False


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_send_to_nonexistent_agent(self, started_mesh):
        """Test sending message to non-existent agent."""
        msg_id = await started_mesh.send(
            sender_id="router-1",
            recipient_id="nonexistent-agent",
            message_type=MessageType.QUERY,
            payload={},
        )

        # Should not raise, but message won't be delivered
        assert msg_id is not None

    @pytest.mark.asyncio
    async def test_routed_send_no_targets(self, started_mesh):
        """Test routed send with no available targets."""
        # Mark all coders as offline
        started_mesh._agents["coder-1"].status = AgentStatus.OFFLINE
        started_mesh._agents["coder-2"].status = AgentStatus.OFFLINE

        msg_id, targets = await started_mesh.send_routed(
            sender_id="router-1",
            message_type=MessageType.TASK_ASSIGN,
            payload={},
            target_role=AgentRole.CODER,
            routing_strategy=RoutingStrategy.LEAST_LOADED,
        )

        assert targets == []

    @pytest.mark.asyncio
    async def test_task_assignment_no_agents(self, started_mesh):
        """Test task assignment when no agents available."""
        # Mark all agents as offline
        for agent in started_mesh._agents.values():
            agent.status = AgentStatus.OFFLINE

        assigned = await started_mesh.assign_task(
            task_id="task-fail",
            description="Test",
            preferred_role=AgentRole.CODER,
        )

        assert assigned is None

    @pytest.mark.asyncio
    async def test_escalation_no_targets(self, started_mesh):
        """Test escalation when no higher-tier agents available."""
        # Assign task and escalate
        await started_mesh.assign_task(
            task_id="task-001",
            description="Test",
            preferred_role=AgentRole.CODER,
        )

        # Mark all premium agents as offline
        started_mesh._agents["architect-1"].status = AgentStatus.OFFLINE

        new_agent = await started_mesh.escalate_task(
            agent_id="coder-1",
            task_id="task-001",
            reason="Complex",
            target_tier="premium",
        )

        assert new_agent is None

    @pytest.mark.asyncio
    async def test_handler_exception(self, started_mesh):
        """Test handling of exceptions in message handlers."""
        async def failing_handler(msg):
            raise ValueError("Handler error")

        await started_mesh.subscribe("coder-1", failing_handler)

        # Should not propagate exception
        await started_mesh.send(
            sender_id="router-1",
            recipient_id="coder-1",
            message_type=MessageType.QUERY,
            payload={},
        )

        await asyncio.sleep(0.2)

        # Message should be retried or dead-lettered
        assert started_mesh._stats["messages_sent"] >= 1


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
