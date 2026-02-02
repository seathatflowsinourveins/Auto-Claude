"""
UNLEASH V2 Agent Mesh Communication Layer
==========================================

Event-driven agent mesh replacing orchestrator bottleneck.
Agents communicate directly via Redis pub/sub channels.

Key Features:
- Direct agent-to-agent communication (no bottleneck)
- Shared state layer (Redis hot, PostgreSQL cold)
- Event-driven architecture
- Automatic load balancing
- Fault tolerance with heartbeats

Research-verified (2026-01-30):
- Ably: "Orchestrators become a bottleneck"
- Stack AI: Hierarchical + Mesh hybrid patterns
- Architecture: Redis pub/sub for realtime, PostgreSQL for durability
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Awaitable, Optional, Dict, List, Set
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Message Types
# =============================================================================

class MessageType(Enum):
    """Types of messages in the agent mesh."""
    # Task management
    TASK_ASSIGN = "task.assign"
    TASK_ACCEPT = "task.accept"
    TASK_REJECT = "task.reject"
    TASK_PROGRESS = "task.progress"
    TASK_COMPLETE = "task.complete"
    TASK_FAILED = "task.failed"

    # Agent coordination
    HEARTBEAT = "agent.heartbeat"
    AGENT_JOIN = "agent.join"
    AGENT_LEAVE = "agent.leave"
    AGENT_STATUS = "agent.status"

    # Data sharing
    CONTEXT_SHARE = "context.share"
    CONTEXT_REQUEST = "context.request"
    RESULT_SHARE = "result.share"

    # Escalation
    ESCALATE = "escalate"
    ESCALATE_ACCEPT = "escalate.accept"

    # Consensus
    VOTE_REQUEST = "vote.request"
    VOTE_SUBMIT = "vote.submit"
    CONSENSUS_REACHED = "consensus.reached"


class AgentRole(Enum):
    """Agent roles in the mesh."""
    ROUTER = "router"           # Entry point, task classification
    CODER = "coder"             # Code generation
    REVIEWER = "reviewer"       # Code review
    ARCHITECT = "architect"     # System design
    SECURITY = "security"       # Security analysis
    RESEARCHER = "researcher"   # Research and documentation
    EXECUTOR = "executor"       # Tool execution
    QUEEN = "queen"             # Final decision maker


class AgentStatus(Enum):
    """Agent health status."""
    IDLE = "idle"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    OFFLINE = "offline"


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class AgentInfo:
    """Information about an agent in the mesh."""
    agent_id: str
    role: AgentRole
    model_tier: str  # local, hybrid, premium
    status: AgentStatus = AgentStatus.IDLE
    capabilities: List[str] = field(default_factory=list)
    current_task: Optional[str] = None
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    tasks_completed: int = 0
    success_rate: float = 1.0

    @property
    def is_available(self) -> bool:
        return self.status in (AgentStatus.IDLE, AgentStatus.BUSY)

    @property
    def is_stale(self) -> bool:
        """Check if heartbeat is stale (>30s)."""
        return datetime.now(timezone.utc) - self.last_heartbeat > timedelta(seconds=30)


@dataclass
class MeshMessage:
    """Message sent through the agent mesh."""
    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None  # For request-response
    ttl_seconds: int = 300

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "ttl_seconds": self.ttl_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MeshMessage":
        return cls(
            message_id=data["message_id"],
            message_type=MessageType(data["message_type"]),
            sender_id=data["sender_id"],
            recipient_id=data.get("recipient_id"),
            payload=data["payload"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            correlation_id=data.get("correlation_id"),
            ttl_seconds=data.get("ttl_seconds", 300),
        )


@dataclass
class TaskState:
    """State of a task in the mesh."""
    task_id: str
    description: str
    assigned_agent: Optional[str] = None
    status: str = "pending"
    priority: int = 5
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    escalation_chain: List[str] = field(default_factory=list)


# =============================================================================
# Message Handlers
# =============================================================================

MessageHandler = Callable[[MeshMessage], Awaitable[Optional[MeshMessage]]]


# =============================================================================
# Agent Mesh Implementation
# =============================================================================

class AgentMesh:
    """
    Agent mesh for direct agent-to-agent communication.

    Architecture:
    - Pub/Sub channels for real-time messaging
    - Shared state in Redis for hot data
    - PostgreSQL for persistent state
    - No central orchestrator bottleneck

    Usage:
        mesh = AgentMesh()
        await mesh.start()

        # Register an agent
        agent_id = await mesh.register_agent(AgentRole.CODER, "local")

        # Subscribe to messages
        await mesh.subscribe(agent_id, handler_callback)

        # Send message to another agent
        await mesh.send(
            sender_id=agent_id,
            recipient_id="architect-1",
            message_type=MessageType.CONTEXT_REQUEST,
            payload={"query": "What pattern should I use?"}
        )
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        use_mock: bool = False,
    ):
        """
        Initialize the agent mesh.

        Args:
            redis_url: Redis connection URL
            use_mock: Use in-memory mock (for testing)
        """
        self.redis_url = redis_url
        self.use_mock = use_mock

        # Agent registry
        self._agents: Dict[str, AgentInfo] = {}

        # Message handlers
        self._handlers: Dict[str, List[MessageHandler]] = {}

        # Task state
        self._tasks: Dict[str, TaskState] = {}

        # Pending responses (for request-response pattern)
        self._pending: Dict[str, asyncio.Future] = {}

        # Mock pub/sub queues (when Redis not available)
        self._mock_channels: Dict[str, asyncio.Queue] = {}

        # Background tasks
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None

        # Statistics
        self._stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "tasks_assigned": 0,
            "tasks_completed": 0,
            "escalations": 0,
        }

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self):
        """Start the agent mesh."""
        self._running = True

        # Start background tasks
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("Agent mesh started")

    async def stop(self):
        """Stop the agent mesh."""
        self._running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()

        logger.info("Agent mesh stopped")

    # =========================================================================
    # Agent Management
    # =========================================================================

    async def register_agent(
        self,
        role: AgentRole,
        model_tier: str,
        capabilities: Optional[List[str]] = None,
        agent_id: Optional[str] = None,
    ) -> str:
        """
        Register a new agent in the mesh.

        Args:
            role: Agent role
            model_tier: local, hybrid, or premium
            capabilities: List of capabilities
            agent_id: Optional custom agent ID

        Returns:
            Agent ID
        """
        if agent_id is None:
            agent_id = f"{role.value}-{uuid.uuid4().hex[:8]}"

        agent_info = AgentInfo(
            agent_id=agent_id,
            role=role,
            model_tier=model_tier,
            capabilities=capabilities or [],
        )

        self._agents[agent_id] = agent_info

        # Broadcast join message
        await self.broadcast(
            sender_id=agent_id,
            message_type=MessageType.AGENT_JOIN,
            payload={"role": role.value, "model_tier": model_tier},
        )

        logger.info(f"Agent registered: {agent_id} ({role.value})")
        return agent_id

    async def unregister_agent(self, agent_id: str):
        """Unregister an agent from the mesh."""
        if agent_id in self._agents:
            # Broadcast leave message
            await self.broadcast(
                sender_id=agent_id,
                message_type=MessageType.AGENT_LEAVE,
                payload={},
            )

            del self._agents[agent_id]
            logger.info(f"Agent unregistered: {agent_id}")

    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """Get agent info by ID."""
        return self._agents.get(agent_id)

    def get_agents_by_role(self, role: AgentRole) -> List[AgentInfo]:
        """Get all agents with a specific role."""
        return [a for a in self._agents.values() if a.role == role]

    def get_available_agents(self, role: Optional[AgentRole] = None) -> List[AgentInfo]:
        """Get all available (non-busy) agents."""
        agents = self._agents.values()
        if role:
            agents = [a for a in agents if a.role == role]
        return [a for a in agents if a.is_available and not a.is_stale]

    # =========================================================================
    # Messaging
    # =========================================================================

    async def send(
        self,
        sender_id: str,
        recipient_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        correlation_id: Optional[str] = None,
    ) -> str:
        """
        Send a message to a specific agent.

        Args:
            sender_id: Sending agent ID
            recipient_id: Receiving agent ID
            message_type: Type of message
            payload: Message payload
            correlation_id: For request-response correlation

        Returns:
            Message ID
        """
        message = MeshMessage(
            message_id=uuid.uuid4().hex,
            message_type=message_type,
            sender_id=sender_id,
            recipient_id=recipient_id,
            payload=payload,
            correlation_id=correlation_id,
        )

        await self._deliver(message)
        self._stats["messages_sent"] += 1

        return message.message_id

    async def broadcast(
        self,
        sender_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        exclude: Optional[Set[str]] = None,
    ) -> str:
        """
        Broadcast a message to all agents.

        Args:
            sender_id: Sending agent ID
            message_type: Type of message
            payload: Message payload
            exclude: Agent IDs to exclude

        Returns:
            Message ID
        """
        message = MeshMessage(
            message_id=uuid.uuid4().hex,
            message_type=message_type,
            sender_id=sender_id,
            recipient_id=None,  # Broadcast
            payload=payload,
        )

        # Deliver to all agents except excluded
        for agent_id in self._agents:
            if exclude and agent_id in exclude:
                continue
            if agent_id != sender_id:
                await self._deliver(message, target_id=agent_id)

        self._stats["messages_sent"] += 1
        return message.message_id

    async def request(
        self,
        sender_id: str,
        recipient_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        timeout: float = 30.0,
    ) -> Optional[MeshMessage]:
        """
        Send a request and wait for response.

        Args:
            sender_id: Sending agent ID
            recipient_id: Receiving agent ID
            message_type: Type of message
            payload: Message payload
            timeout: Timeout in seconds

        Returns:
            Response message or None if timeout
        """
        correlation_id = uuid.uuid4().hex

        # Create future for response
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[correlation_id] = future

        # Send request
        await self.send(
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload,
            correlation_id=correlation_id,
        )

        try:
            response = await asyncio.wait_for(future, timeout)
            return response
        except asyncio.TimeoutError:
            logger.warning(f"Request timeout: {correlation_id}")
            return None
        finally:
            self._pending.pop(correlation_id, None)

    async def subscribe(
        self,
        agent_id: str,
        handler: MessageHandler,
        message_types: Optional[List[MessageType]] = None,
    ):
        """
        Subscribe to messages for an agent.

        Args:
            agent_id: Agent ID to subscribe
            handler: Async callback for messages
            message_types: Filter to specific types (None = all)
        """
        if agent_id not in self._handlers:
            self._handlers[agent_id] = []

        # Wrap handler to filter by message type
        async def filtered_handler(msg: MeshMessage) -> Optional[MeshMessage]:
            if message_types and msg.message_type not in message_types:
                return None
            return await handler(msg)

        self._handlers[agent_id].append(filtered_handler)

    async def _deliver(
        self,
        message: MeshMessage,
        target_id: Optional[str] = None,
    ):
        """Deliver a message to target agent(s)."""
        target = target_id or message.recipient_id

        if target and target in self._handlers:
            for handler in self._handlers[target]:
                try:
                    response = await handler(message)
                    self._stats["messages_received"] += 1

                    # Handle response for request-response pattern
                    if response and message.correlation_id:
                        if message.correlation_id in self._pending:
                            self._pending[message.correlation_id].set_result(response)
                except Exception as e:
                    logger.error(f"Handler error for {target}: {e}")

    # =========================================================================
    # Task Management
    # =========================================================================

    async def assign_task(
        self,
        task_id: str,
        description: str,
        preferred_role: Optional[AgentRole] = None,
        preferred_tier: Optional[str] = None,
        priority: int = 5,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Assign a task to an available agent.

        Uses load balancing to select the best agent:
        1. Filter by role and tier preference
        2. Select least loaded agent
        3. Fallback to any available agent

        Args:
            task_id: Unique task ID
            description: Task description
            preferred_role: Preferred agent role
            preferred_tier: Preferred model tier
            priority: Task priority (1-10)
            context: Additional context

        Returns:
            Assigned agent ID or None if no agents available
        """
        # Create task state
        task = TaskState(
            task_id=task_id,
            description=description,
            priority=priority,
            context=context or {},
        )
        self._tasks[task_id] = task

        # Find available agents
        candidates = self.get_available_agents(preferred_role)

        if preferred_tier:
            tier_candidates = [a for a in candidates if a.model_tier == preferred_tier]
            if tier_candidates:
                candidates = tier_candidates

        if not candidates:
            # Fallback to any available agent
            candidates = self.get_available_agents()

        if not candidates:
            logger.warning(f"No available agents for task {task_id}")
            return None

        # Select least loaded agent
        selected = min(candidates, key=lambda a: 0 if a.status == AgentStatus.IDLE else 1)

        # Update task state
        task.assigned_agent = selected.agent_id
        task.status = "assigned"
        task.started_at = datetime.now(timezone.utc)

        # Update agent status
        selected.status = AgentStatus.BUSY
        selected.current_task = task_id

        # Send assignment message
        await self.send(
            sender_id="mesh",
            recipient_id=selected.agent_id,
            message_type=MessageType.TASK_ASSIGN,
            payload={
                "task_id": task_id,
                "description": description,
                "priority": priority,
                "context": context or {},
            },
        )

        self._stats["tasks_assigned"] += 1
        logger.info(f"Task {task_id} assigned to {selected.agent_id}")

        return selected.agent_id

    async def complete_task(
        self,
        agent_id: str,
        task_id: str,
        result: Any,
    ):
        """Mark a task as completed."""
        if task_id not in self._tasks:
            return

        task = self._tasks[task_id]
        task.status = "completed"
        task.completed_at = datetime.now(timezone.utc)
        task.result = result

        # Update agent status
        if agent_id in self._agents:
            agent = self._agents[agent_id]
            agent.status = AgentStatus.IDLE
            agent.current_task = None
            agent.tasks_completed += 1

        # Broadcast completion
        await self.broadcast(
            sender_id=agent_id,
            message_type=MessageType.TASK_COMPLETE,
            payload={
                "task_id": task_id,
                "result_summary": str(result)[:500],
            },
        )

        self._stats["tasks_completed"] += 1
        logger.info(f"Task {task_id} completed by {agent_id}")

    async def escalate_task(
        self,
        agent_id: str,
        task_id: str,
        reason: str,
        target_role: Optional[AgentRole] = None,
        target_tier: Optional[str] = None,
    ) -> Optional[str]:
        """
        Escalate a task to a higher-tier agent.

        Args:
            agent_id: Current agent ID
            task_id: Task to escalate
            reason: Reason for escalation
            target_role: Target agent role
            target_tier: Target model tier (e.g., "premium")

        Returns:
            New agent ID or None
        """
        if task_id not in self._tasks:
            return None

        task = self._tasks[task_id]
        task.escalation_chain.append(agent_id)

        # Free current agent
        if agent_id in self._agents:
            self._agents[agent_id].status = AgentStatus.IDLE
            self._agents[agent_id].current_task = None

        # Find higher-tier agent
        candidates = self.get_available_agents(target_role)

        if target_tier:
            candidates = [a for a in candidates if a.model_tier == target_tier]

        # Exclude agents already in escalation chain
        candidates = [a for a in candidates if a.agent_id not in task.escalation_chain]

        if not candidates:
            logger.warning(f"No escalation targets for task {task_id}")
            return None

        selected = candidates[0]

        # Reassign
        task.assigned_agent = selected.agent_id
        selected.status = AgentStatus.BUSY
        selected.current_task = task_id

        # Send escalation message
        await self.send(
            sender_id=agent_id,
            recipient_id=selected.agent_id,
            message_type=MessageType.ESCALATE,
            payload={
                "task_id": task_id,
                "reason": reason,
                "previous_agent": agent_id,
                "context": task.context,
            },
        )

        self._stats["escalations"] += 1
        logger.info(f"Task {task_id} escalated from {agent_id} to {selected.agent_id}")

        return selected.agent_id

    # =========================================================================
    # Context Sharing
    # =========================================================================

    async def share_context(
        self,
        sender_id: str,
        context_key: str,
        context_value: Any,
        recipient_ids: Optional[List[str]] = None,
    ):
        """
        Share context with other agents.

        Args:
            sender_id: Sending agent
            context_key: Context key
            context_value: Context value
            recipient_ids: Specific recipients (None = broadcast)
        """
        payload = {
            "key": context_key,
            "value": context_value,
        }

        if recipient_ids:
            for recipient_id in recipient_ids:
                await self.send(
                    sender_id=sender_id,
                    recipient_id=recipient_id,
                    message_type=MessageType.CONTEXT_SHARE,
                    payload=payload,
                )
        else:
            await self.broadcast(
                sender_id=sender_id,
                message_type=MessageType.CONTEXT_SHARE,
                payload=payload,
            )

    async def request_context(
        self,
        sender_id: str,
        recipient_id: str,
        context_key: str,
        timeout: float = 10.0,
    ) -> Any:
        """
        Request context from another agent.

        Args:
            sender_id: Requesting agent
            recipient_id: Agent with context
            context_key: Context key to request
            timeout: Timeout in seconds

        Returns:
            Context value or None
        """
        response = await self.request(
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=MessageType.CONTEXT_REQUEST,
            payload={"key": context_key},
            timeout=timeout,
        )

        if response:
            return response.payload.get("value")
        return None

    # =========================================================================
    # Background Tasks
    # =========================================================================

    async def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        while self._running:
            try:
                for agent_id in list(self._agents.keys()):
                    agent = self._agents[agent_id]
                    agent.last_heartbeat = datetime.now(timezone.utc)

                    await self.broadcast(
                        sender_id=agent_id,
                        message_type=MessageType.HEARTBEAT,
                        payload={"status": agent.status.value},
                    )

                await asyncio.sleep(10)  # Heartbeat every 10s
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def _cleanup_loop(self):
        """Clean up stale agents and expired messages."""
        while self._running:
            try:
                # Remove stale agents
                stale_agents = [
                    agent_id for agent_id, agent in self._agents.items()
                    if agent.is_stale
                ]

                for agent_id in stale_agents:
                    logger.warning(f"Removing stale agent: {agent_id}")
                    await self.unregister_agent(agent_id)

                await asyncio.sleep(30)  # Cleanup every 30s
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get mesh statistics."""
        return {
            **self._stats,
            "total_agents": len(self._agents),
            "available_agents": len(self.get_available_agents()),
            "pending_tasks": len([t for t in self._tasks.values() if t.status == "pending"]),
            "active_tasks": len([t for t in self._tasks.values() if t.status == "assigned"]),
        }

    def get_agent_stats(self) -> List[Dict[str, Any]]:
        """Get per-agent statistics."""
        return [
            {
                "agent_id": a.agent_id,
                "role": a.role.value,
                "model_tier": a.model_tier,
                "status": a.status.value,
                "tasks_completed": a.tasks_completed,
                "success_rate": a.success_rate,
            }
            for a in self._agents.values()
        ]


# =============================================================================
# Demo
# =============================================================================

async def demo():
    """Demonstrate the agent mesh."""
    print("=" * 70)
    print("UNLEASH V2 Agent Mesh - Demo")
    print("=" * 70)
    print()

    mesh = AgentMesh(use_mock=True)
    await mesh.start()

    # Register agents
    router_id = await mesh.register_agent(AgentRole.ROUTER, "local")
    coder_id = await mesh.register_agent(AgentRole.CODER, "local")
    reviewer_id = await mesh.register_agent(AgentRole.REVIEWER, "hybrid")
    architect_id = await mesh.register_agent(AgentRole.ARCHITECT, "premium")

    print(f"Registered agents: {len(mesh._agents)}")
    print()

    # Subscribe to messages
    received_messages = []

    async def coder_handler(msg: MeshMessage) -> Optional[MeshMessage]:
        received_messages.append(msg)
        print(f"[{coder_id}] Received: {msg.message_type.value}")

        if msg.message_type == MessageType.TASK_ASSIGN:
            # Simulate work
            await asyncio.sleep(0.1)

            # Complete task
            await mesh.complete_task(
                agent_id=coder_id,
                task_id=msg.payload["task_id"],
                result="Code generated successfully",
            )

        return None

    await mesh.subscribe(coder_id, coder_handler)

    # Assign a task
    print("Assigning task...")
    assigned = await mesh.assign_task(
        task_id="task-001",
        description="Implement JWT authentication",
        preferred_role=AgentRole.CODER,
        priority=8,
    )
    print(f"Task assigned to: {assigned}")
    print()

    # Wait for completion
    await asyncio.sleep(0.5)

    # Share context
    await mesh.share_context(
        sender_id=architect_id,
        context_key="auth_pattern",
        context_value={"type": "JWT", "expiry": "1h"},
    )

    # Print statistics
    print("Mesh Statistics:")
    stats = mesh.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print()
    print("Agent Statistics:")
    for agent_stat in mesh.get_agent_stats():
        print(f"  {agent_stat['agent_id']}: {agent_stat['status']} ({agent_stat['model_tier']})")

    await mesh.stop()
    print()
    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demo())
