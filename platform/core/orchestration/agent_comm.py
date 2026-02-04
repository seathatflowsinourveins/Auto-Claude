"""
Agent-to-Agent Communication Module - V40 Architecture

Memory-based agent-to-agent communication system enabling UNLEASH agents to
coordinate, share results, and maintain synchronized state across distributed
execution contexts.

Features:
1. Message Types - TASK, RESULT, QUERY, RESPONSE with typed payloads
2. Memory-Based Inbox/Outbox - SQLite-backed persistent message queues
3. Reference-Based Large Outputs - Lazy loading for payloads > 8KB
4. Broadcast Support - Topic-based pub/sub for agent coordination
5. Message Queue - FIFO with priority support and acknowledgments
6. Swarm Integration - Works with ExtremeSwarmController, hierarchical topology, event bus

Architecture:
    +------------------------------------------------------------------+
    |                    Agent Communication Layer                       |
    +------------------------------------------------------------------+
    |                                                                   |
    |   Agent A                                                         |
    |      |                                                            |
    |      +---> [Outbox] ---> MessageStore (SQLite)                   |
    |                               |                                   |
    |                               v                                   |
    |   Agent B <--- [Inbox] <--- Routing Layer                        |
    |                               |                                   |
    |                               v                                   |
    |   Broadcast <--- [Topics] <--- Topic Subscriptions               |
    |                                                                   |
    |   Large Payloads:                                                 |
    |      payload > 8KB --> memory:// reference                       |
    |      receiver fetches via reference                              |
    |                                                                   |
    +------------------------------------------------------------------+

Usage:
    from core.orchestration.agent_comm import (
        AgentCommSystem,
        AgentMessage,
        MessageType,
        create_agent_comm_system,
    )

    # Create communication system
    comm = await create_agent_comm_system()

    # Send a task to another agent
    await comm.send_message(
        sender_id="coordinator-1",
        receiver_id="researcher-1",
        message_type=MessageType.TASK,
        payload={"task": "Research OAuth patterns", "context": {...}}
    )

    # Receive messages
    messages = await comm.receive_messages("researcher-1")

    # Send result with large payload (auto-referenced)
    await comm.send_result(
        sender_id="researcher-1",
        receiver_id="coordinator-1",
        task_id="task-123",
        result=large_research_output  # Auto-stored if > 8KB
    )

    # Broadcast to all agents
    await comm.broadcast(
        sender_id="coordinator-1",
        topic="status_update",
        payload={"status": "research_complete", "progress": 0.5}
    )

Version: V1.0.0 (February 2026)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from .domain_events import DomainEvent, EventBus
from .extreme_swarm import (
    AgentMetadata,
    ExtremeSwarmController,
    SwarmEvent,
    SwarmRole,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Payload size threshold for reference-based storage (8KB)
LARGE_PAYLOAD_THRESHOLD = 8 * 1024

# Default message TTL (1 hour)
DEFAULT_MESSAGE_TTL_SECONDS = 60 * 60

# Maximum messages to fetch at once
MAX_BATCH_SIZE = 100

# Cleanup interval for old messages
CLEANUP_INTERVAL_SECONDS = 300


# =============================================================================
# ENUMS
# =============================================================================


class MessageType(str, Enum):
    """Types of agent-to-agent messages."""

    TASK = "task"           # Task assignment from coordinator
    RESULT = "result"       # Task completion result
    QUERY = "query"         # Information request
    RESPONSE = "response"   # Response to query
    STATUS = "status"       # Status update
    HEARTBEAT = "heartbeat" # Liveness check
    BROADCAST = "broadcast" # Broadcast to all/topic
    ACK = "ack"            # Acknowledgment


class MessageStatus(str, Enum):
    """Message delivery status."""

    PENDING = "pending"       # Waiting in queue
    DELIVERED = "delivered"   # Delivered to inbox
    READ = "read"            # Read by receiver
    ACKNOWLEDGED = "acknowledged"  # Explicitly acknowledged
    FAILED = "failed"        # Delivery failed
    EXPIRED = "expired"      # TTL exceeded


class MessagePriority(int, Enum):
    """Message priority levels."""

    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class AgentMessage:
    """
    A message between agents in the UNLEASH swarm.

    Attributes:
        id: Unique message identifier
        sender_id: ID of the sending agent
        receiver_id: ID of receiving agent or "broadcast" for all
        message_type: Type of message (TASK, RESULT, QUERY, etc.)
        payload: Message content (dict or reference)
        reference: Optional memory:// reference for large payloads
        timestamp: Message creation time
        priority: Message priority (1-10)
        correlation_id: ID linking related messages (e.g., task -> result)
        reply_to: Message ID this is replying to
        ttl_seconds: Time-to-live before expiration
        status: Current message status
        topic: Topic for broadcast messages
        metadata: Additional message metadata
    """

    id: str = field(default_factory=lambda: f"msg-{uuid.uuid4().hex[:12]}")
    sender_id: str = ""
    receiver_id: str = ""  # "broadcast" for broadcast messages
    message_type: MessageType = MessageType.TASK
    payload: Dict[str, Any] = field(default_factory=dict)
    reference: Optional[str] = None  # memory:// reference for large payloads
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl_seconds: int = DEFAULT_MESSAGE_TTL_SECONDS
    status: MessageStatus = MessageStatus.PENDING
    topic: Optional[str] = None  # For broadcast messages
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_broadcast(self) -> bool:
        """Check if this is a broadcast message."""
        return self.receiver_id == "broadcast" or self.topic is not None

    @property
    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.ttl_seconds <= 0:
            return False  # No expiration
        age = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        return age > self.ttl_seconds

    @property
    def has_reference(self) -> bool:
        """Check if payload is stored as reference."""
        return self.reference is not None and self.reference.startswith("memory://")

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for storage."""
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.value,
            "payload": json.dumps(self.payload) if self.payload else "{}",
            "reference": self.reference,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "ttl_seconds": self.ttl_seconds,
            "status": self.status.value,
            "topic": self.topic,
            "metadata": json.dumps(self.metadata) if self.metadata else "{}",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create message from dictionary."""
        return cls(
            id=data["id"],
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            message_type=MessageType(data["message_type"]),
            payload=json.loads(data["payload"]) if isinstance(data["payload"], str) else data["payload"],
            reference=data.get("reference"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"],
            priority=MessagePriority(data["priority"]) if isinstance(data["priority"], int) else MessagePriority.NORMAL,
            correlation_id=data.get("correlation_id"),
            reply_to=data.get("reply_to"),
            ttl_seconds=data.get("ttl_seconds", DEFAULT_MESSAGE_TTL_SECONDS),
            status=MessageStatus(data["status"]) if data.get("status") else MessageStatus.PENDING,
            topic=data.get("topic"),
            metadata=json.loads(data["metadata"]) if isinstance(data.get("metadata"), str) else data.get("metadata", {}),
        )


@dataclass
class MessageAcknowledgment:
    """Acknowledgment for a received message."""

    message_id: str
    agent_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    success: bool = True
    error: Optional[str] = None


@dataclass
class TopicSubscription:
    """A subscription to a broadcast topic."""

    agent_id: str
    topic: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    filter_fn: Optional[Callable[[AgentMessage], bool]] = None


@dataclass
class InboxStats:
    """Statistics for an agent's inbox."""

    agent_id: str
    total_messages: int
    unread_messages: int
    pending_messages: int
    oldest_message_age_seconds: float
    by_type: Dict[str, int] = field(default_factory=dict)
    by_priority: Dict[int, int] = field(default_factory=dict)


# =============================================================================
# EVENTS
# =============================================================================


@dataclass
class MessageSentEvent(SwarmEvent):
    """Event emitted when a message is sent."""

    message_id: str = ""
    sender_id: str = ""
    receiver_id: str = ""
    message_type: str = ""


@dataclass
class MessageReceivedEvent(SwarmEvent):
    """Event emitted when a message is received."""

    message_id: str = ""
    receiver_id: str = ""
    sender_id: str = ""
    message_type: str = ""


@dataclass
class MessageAcknowledgedEvent(SwarmEvent):
    """Event emitted when a message is acknowledged."""

    message_id: str = ""
    agent_id: str = ""
    success: bool = True


@dataclass
class BroadcastSentEvent(SwarmEvent):
    """Event emitted when a broadcast is sent."""

    message_id: str = ""
    sender_id: str = ""
    topic: str = ""
    recipient_count: int = 0


# =============================================================================
# SQLITE MESSAGE STORE
# =============================================================================


MESSAGES_SCHEMA = """
-- Messages table for inbox/outbox storage
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    sender_id TEXT NOT NULL,
    receiver_id TEXT NOT NULL,
    message_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    reference TEXT,
    timestamp TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 5,
    correlation_id TEXT,
    reply_to TEXT,
    ttl_seconds INTEGER NOT NULL DEFAULT 3600,
    status TEXT NOT NULL DEFAULT 'pending',
    topic TEXT,
    metadata TEXT,
    delivered_at TEXT,
    read_at TEXT,
    acknowledged_at TEXT
);

-- Large payload storage for reference-based messages
CREATE TABLE IF NOT EXISTS large_payloads (
    reference TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT
);

-- Topic subscriptions
CREATE TABLE IF NOT EXISTS topic_subscriptions (
    agent_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (agent_id, topic)
);

-- Acknowledgments
CREATE TABLE IF NOT EXISTS acknowledgments (
    message_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    success INTEGER NOT NULL DEFAULT 1,
    error TEXT,
    PRIMARY KEY (message_id, agent_id)
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_messages_receiver ON messages(receiver_id, status);
CREATE INDEX IF NOT EXISTS idx_messages_sender ON messages(sender_id);
CREATE INDEX IF NOT EXISTS idx_messages_type ON messages(message_type);
CREATE INDEX IF NOT EXISTS idx_messages_topic ON messages(topic) WHERE topic IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_messages_correlation ON messages(correlation_id) WHERE correlation_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_messages_priority ON messages(priority DESC);
CREATE INDEX IF NOT EXISTS idx_payloads_created ON large_payloads(created_at);
"""


class SQLiteMessageStore:
    """
    SQLite-based message store for agent communication.

    Provides persistent storage for:
    - Agent inboxes and outboxes
    - Large payload references
    - Topic subscriptions
    - Message acknowledgments
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
    ):
        """
        Initialize the message store.

        Args:
            db_path: Path to SQLite database. Defaults to ~/.unleash/messages.db
        """
        self.db_path = db_path or Path.home() / ".unleash" / "messages.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()
        self._init_db()

        logger.info(f"Message store initialized at {self.db_path}")

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript(MESSAGES_SCHEMA)
            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get database connection with context management."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")

        try:
            yield self._connection
        except Exception:
            self._connection.rollback()
            raise

    async def store_message(self, message: AgentMessage) -> bool:
        """
        Store a message in the database.

        Args:
            message: The message to store

        Returns:
            True if stored successfully
        """
        async with self._lock:
            with self._get_connection() as conn:
                data = message.to_dict()
                conn.execute(
                    """
                    INSERT INTO messages (
                        id, sender_id, receiver_id, message_type, payload,
                        reference, timestamp, priority, correlation_id,
                        reply_to, ttl_seconds, status, topic, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        data["id"],
                        data["sender_id"],
                        data["receiver_id"],
                        data["message_type"],
                        data["payload"],
                        data["reference"],
                        data["timestamp"],
                        data["priority"],
                        data["correlation_id"],
                        data["reply_to"],
                        data["ttl_seconds"],
                        data["status"],
                        data["topic"],
                        data["metadata"],
                    ),
                )
                conn.commit()
                return True

    async def get_messages_for_agent(
        self,
        agent_id: str,
        status: Optional[MessageStatus] = None,
        message_type: Optional[MessageType] = None,
        limit: int = MAX_BATCH_SIZE,
        include_broadcasts: bool = True,
    ) -> List[AgentMessage]:
        """
        Get messages for an agent.

        Args:
            agent_id: The agent's ID
            status: Filter by status (optional)
            message_type: Filter by type (optional)
            limit: Maximum messages to return
            include_broadcasts: Include broadcast messages

        Returns:
            List of messages ordered by priority DESC, timestamp ASC
        """
        async with self._lock:
            with self._get_connection() as conn:
                query = "SELECT * FROM messages WHERE receiver_id = ?"
                params: List[Any] = [agent_id]

                if include_broadcasts:
                    # Get subscribed topics
                    topics_cursor = conn.execute(
                        "SELECT topic FROM topic_subscriptions WHERE agent_id = ?",
                        (agent_id,),
                    )
                    topics = [row["topic"] for row in topics_cursor.fetchall()]

                    if topics:
                        topic_placeholders = ",".join("?" * len(topics))
                        query = f"""
                            SELECT * FROM messages
                            WHERE receiver_id = ?
                               OR (receiver_id = 'broadcast' AND topic IN ({topic_placeholders}))
                        """
                        params.extend(topics)

                if status:
                    query += " AND status = ?"
                    params.append(status.value)

                if message_type:
                    query += " AND message_type = ?"
                    params.append(message_type.value)

                query += " ORDER BY priority DESC, timestamp ASC LIMIT ?"
                params.append(limit)

                cursor = conn.execute(query, params)
                messages = []
                for row in cursor.fetchall():
                    msg = AgentMessage.from_dict(dict(row))
                    if not msg.is_expired:
                        messages.append(msg)

                return messages

    async def update_message_status(
        self,
        message_id: str,
        status: MessageStatus,
    ) -> bool:
        """Update message status."""
        async with self._lock:
            with self._get_connection() as conn:
                now = datetime.now(timezone.utc).isoformat()

                # Set appropriate timestamp field
                timestamp_field = {
                    MessageStatus.DELIVERED: "delivered_at",
                    MessageStatus.READ: "read_at",
                    MessageStatus.ACKNOWLEDGED: "acknowledged_at",
                }.get(status)

                if timestamp_field:
                    conn.execute(
                        f"UPDATE messages SET status = ?, {timestamp_field} = ? WHERE id = ?",
                        (status.value, now, message_id),
                    )
                else:
                    conn.execute(
                        "UPDATE messages SET status = ? WHERE id = ?",
                        (status.value, message_id),
                    )

                conn.commit()
                return conn.total_changes > 0

    async def store_large_payload(
        self,
        payload: str,
    ) -> str:
        """
        Store a large payload and return reference.

        Args:
            payload: The payload content

        Returns:
            memory:// reference to the stored payload
        """
        reference = f"memory://agent_messages/payloads/{uuid.uuid4().hex}"
        size_bytes = len(payload.encode("utf-8"))
        now = datetime.now(timezone.utc).isoformat()

        async with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO large_payloads (reference, payload, created_at, size_bytes)
                    VALUES (?, ?, ?, ?)
                    """,
                    (reference, payload, now, size_bytes),
                )
                conn.commit()

        return reference

    async def get_large_payload(self, reference: str) -> Optional[str]:
        """
        Retrieve a large payload by reference.

        Args:
            reference: The memory:// reference

        Returns:
            The payload content or None if not found
        """
        async with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT payload FROM large_payloads WHERE reference = ?",
                    (reference,),
                )
                row = cursor.fetchone()
                if row:
                    # Update access tracking
                    now = datetime.now(timezone.utc).isoformat()
                    conn.execute(
                        """
                        UPDATE large_payloads
                        SET access_count = access_count + 1, last_accessed = ?
                        WHERE reference = ?
                        """,
                        (now, reference),
                    )
                    conn.commit()
                    return row["payload"]
                return None

    async def subscribe_to_topic(self, agent_id: str, topic: str) -> bool:
        """Subscribe an agent to a broadcast topic."""
        async with self._lock:
            with self._get_connection() as conn:
                now = datetime.now(timezone.utc).isoformat()
                try:
                    conn.execute(
                        "INSERT OR REPLACE INTO topic_subscriptions VALUES (?, ?, ?)",
                        (agent_id, topic, now),
                    )
                    conn.commit()
                    return True
                except Exception as e:
                    logger.error(f"Failed to subscribe {agent_id} to {topic}: {e}")
                    return False

    async def unsubscribe_from_topic(self, agent_id: str, topic: str) -> bool:
        """Unsubscribe an agent from a topic."""
        async with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    "DELETE FROM topic_subscriptions WHERE agent_id = ? AND topic = ?",
                    (agent_id, topic),
                )
                conn.commit()
                return conn.total_changes > 0

    async def get_topic_subscribers(self, topic: str) -> List[str]:
        """Get all subscribers for a topic."""
        async with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT agent_id FROM topic_subscriptions WHERE topic = ?",
                    (topic,),
                )
                return [row["agent_id"] for row in cursor.fetchall()]

    async def store_acknowledgment(self, ack: MessageAcknowledgment) -> bool:
        """Store a message acknowledgment."""
        async with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO acknowledgments
                    (message_id, agent_id, timestamp, success, error)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        ack.message_id,
                        ack.agent_id,
                        ack.timestamp.isoformat(),
                        1 if ack.success else 0,
                        ack.error,
                    ),
                )
                conn.commit()
                return True

    async def get_inbox_stats(self, agent_id: str) -> InboxStats:
        """Get inbox statistics for an agent."""
        async with self._lock:
            with self._get_connection() as conn:
                # Total and unread counts
                cursor = conn.execute(
                    """
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN status IN ('pending', 'delivered') THEN 1 ELSE 0 END) as unread,
                        SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                        MIN(timestamp) as oldest
                    FROM messages
                    WHERE receiver_id = ?
                    """,
                    (agent_id,),
                )
                row = cursor.fetchone()

                # Calculate oldest message age
                oldest_age = 0.0
                if row["oldest"]:
                    oldest_time = datetime.fromisoformat(row["oldest"])
                    oldest_age = (datetime.now(timezone.utc) - oldest_time).total_seconds()

                # By type
                type_cursor = conn.execute(
                    """
                    SELECT message_type, COUNT(*) as count
                    FROM messages WHERE receiver_id = ?
                    GROUP BY message_type
                    """,
                    (agent_id,),
                )
                by_type = {r["message_type"]: r["count"] for r in type_cursor.fetchall()}

                # By priority
                priority_cursor = conn.execute(
                    """
                    SELECT priority, COUNT(*) as count
                    FROM messages WHERE receiver_id = ?
                    GROUP BY priority
                    """,
                    (agent_id,),
                )
                by_priority = {r["priority"]: r["count"] for r in priority_cursor.fetchall()}

                return InboxStats(
                    agent_id=agent_id,
                    total_messages=row["total"] or 0,
                    unread_messages=row["unread"] or 0,
                    pending_messages=row["pending"] or 0,
                    oldest_message_age_seconds=oldest_age,
                    by_type=by_type,
                    by_priority=by_priority,
                )

    async def cleanup_expired_messages(self) -> int:
        """
        Remove expired messages and old payloads.

        Returns:
            Number of messages cleaned up
        """
        async with self._lock:
            with self._get_connection() as conn:
                now = datetime.now(timezone.utc)

                # Delete expired messages
                cursor = conn.execute(
                    """
                    DELETE FROM messages
                    WHERE datetime(timestamp, '+' || ttl_seconds || ' seconds') < datetime(?)
                    """,
                    (now.isoformat(),),
                )
                deleted = cursor.rowcount

                # Delete orphaned payloads (older than 24 hours with no recent access)
                cutoff = (now - timedelta(hours=24)).isoformat()
                conn.execute(
                    """
                    DELETE FROM large_payloads
                    WHERE created_at < ?
                      AND (last_accessed IS NULL OR last_accessed < ?)
                    """,
                    (cutoff, cutoff),
                )

                conn.commit()
                return deleted

    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None


# =============================================================================
# AGENT COMMUNICATION SYSTEM
# =============================================================================


class AgentCommSystem:
    """
    Main agent-to-agent communication system.

    Provides:
    - Message sending with automatic large payload handling
    - Inbox management with priority queuing
    - Broadcast/topic-based messaging
    - Integration with ExtremeSwarmController
    - Event emission for observability
    """

    def __init__(
        self,
        store: Optional[SQLiteMessageStore] = None,
        event_bus: Optional[EventBus] = None,
        swarm_controller: Optional[ExtremeSwarmController] = None,
        large_payload_threshold: int = LARGE_PAYLOAD_THRESHOLD,
    ):
        """
        Initialize the communication system.

        Args:
            store: Message store instance (created if not provided)
            event_bus: Event bus for emitting communication events
            swarm_controller: Optional swarm controller for agent discovery
            large_payload_threshold: Threshold for storing payloads as references
        """
        self.store = store or SQLiteMessageStore()
        self.event_bus = event_bus or EventBus()
        self.swarm_controller = swarm_controller
        self.large_payload_threshold = large_payload_threshold

        # Track registered agents
        self._registered_agents: Set[str] = set()

        # Message handlers by type
        self._handlers: Dict[str, List[Callable[[AgentMessage], Awaitable[None]]]] = {}

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        # Metrics
        self.metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "broadcasts_sent": 0,
            "large_payloads_stored": 0,
            "acknowledgments": 0,
            "cleanup_runs": 0,
        }

    async def start(self) -> None:
        """Start the communication system."""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Agent communication system started")

    async def stop(self) -> None:
        """Stop the communication system."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        self.store.close()
        logger.info("Agent communication system stopped")

    def register_agent(self, agent_id: str, topics: Optional[List[str]] = None) -> None:
        """
        Register an agent with the communication system.

        Args:
            agent_id: The agent's unique identifier
            topics: Optional list of topics to subscribe to
        """
        self._registered_agents.add(agent_id)

        if topics:
            for topic in topics:
                asyncio.create_task(self.subscribe_to_topic(agent_id, topic))

        logger.debug(f"Agent {agent_id} registered with comm system")

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the communication system."""
        self._registered_agents.discard(agent_id)
        logger.debug(f"Agent {agent_id} unregistered from comm system")

    async def send_message(
        self,
        sender_id: str,
        receiver_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None,
        reply_to: Optional[str] = None,
        ttl_seconds: int = DEFAULT_MESSAGE_TTL_SECONDS,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentMessage:
        """
        Send a message to another agent.

        Args:
            sender_id: Sender agent ID
            receiver_id: Receiver agent ID
            message_type: Type of message
            payload: Message content
            priority: Message priority
            correlation_id: Correlation ID for related messages
            reply_to: Message ID being replied to
            ttl_seconds: Message time-to-live
            metadata: Additional metadata

        Returns:
            The sent message with ID
        """
        # Check payload size for reference-based storage
        payload_json = json.dumps(payload)
        reference = None

        if len(payload_json.encode("utf-8")) > self.large_payload_threshold:
            # Store as reference
            reference = await self.store.store_large_payload(payload_json)
            # Store summary in payload
            payload = {
                "_reference": reference,
                "_summary": self._generate_payload_summary(payload),
                "_size_bytes": len(payload_json.encode("utf-8")),
            }
            self.metrics["large_payloads_stored"] += 1

        message = AgentMessage(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            payload=payload,
            reference=reference,
            priority=priority,
            correlation_id=correlation_id,
            reply_to=reply_to,
            ttl_seconds=ttl_seconds,
            metadata=metadata or {},
        )

        await self.store.store_message(message)
        self.metrics["messages_sent"] += 1

        # Emit event
        await self.event_bus.publish(
            MessageSentEvent(
                aggregate_id=message.id,
                source_agent=sender_id,
                message_id=message.id,
                sender_id=sender_id,
                receiver_id=receiver_id,
                message_type=message_type.value,
            )
        )

        logger.debug(
            f"Message sent: {message.id} from {sender_id} to {receiver_id} "
            f"(type={message_type.value}, priority={priority.value})"
        )

        return message

    async def send_task(
        self,
        sender_id: str,
        receiver_id: str,
        task_description: str,
        task_context: Optional[Dict[str, Any]] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        ttl_seconds: int = DEFAULT_MESSAGE_TTL_SECONDS,
    ) -> AgentMessage:
        """
        Send a task assignment to another agent.

        Convenience method for sending TASK type messages.
        """
        return await self.send_message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=MessageType.TASK,
            payload={
                "task": task_description,
                "context": task_context or {},
            },
            priority=priority,
            ttl_seconds=ttl_seconds,
        )

    async def send_result(
        self,
        sender_id: str,
        receiver_id: str,
        task_id: str,
        result: Any,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentMessage:
        """
        Send a task result to another agent.

        Convenience method for sending RESULT type messages.
        Large results are automatically stored as references.
        """
        payload = {
            "task_id": task_id,
            "success": success,
            "result": result,
        }
        if error:
            payload["error"] = error

        return await self.send_message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=MessageType.RESULT,
            payload=payload,
            correlation_id=task_id,
            priority=MessagePriority.HIGH,  # Results are usually important
            metadata=metadata,
        )

    async def send_query(
        self,
        sender_id: str,
        receiver_id: str,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentMessage:
        """
        Send a query to another agent.

        Convenience method for sending QUERY type messages.
        """
        return await self.send_message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=MessageType.QUERY,
            payload={
                "query": query,
                "context": context or {},
            },
            priority=MessagePriority.NORMAL,
        )

    async def broadcast(
        self,
        sender_id: str,
        topic: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> AgentMessage:
        """
        Broadcast a message to all subscribers of a topic.

        Args:
            sender_id: Sender agent ID
            topic: Broadcast topic
            payload: Message content
            priority: Message priority

        Returns:
            The broadcast message
        """
        message = AgentMessage(
            sender_id=sender_id,
            receiver_id="broadcast",
            message_type=MessageType.BROADCAST,
            payload=payload,
            topic=topic,
            priority=priority,
        )

        await self.store.store_message(message)

        # Get subscriber count for metrics
        subscribers = await self.store.get_topic_subscribers(topic)
        self.metrics["broadcasts_sent"] += 1

        # Emit event
        await self.event_bus.publish(
            BroadcastSentEvent(
                aggregate_id=message.id,
                source_agent=sender_id,
                message_id=message.id,
                sender_id=sender_id,
                topic=topic,
                recipient_count=len(subscribers),
            )
        )

        logger.debug(
            f"Broadcast sent: {message.id} from {sender_id} to topic '{topic}' "
            f"({len(subscribers)} subscribers)"
        )

        return message

    async def receive_messages(
        self,
        agent_id: str,
        status: Optional[MessageStatus] = None,
        message_type: Optional[MessageType] = None,
        limit: int = MAX_BATCH_SIZE,
        auto_mark_delivered: bool = True,
    ) -> List[AgentMessage]:
        """
        Receive messages for an agent.

        Args:
            agent_id: The agent's ID
            status: Filter by status (default: PENDING)
            message_type: Filter by message type
            limit: Maximum messages to fetch
            auto_mark_delivered: Automatically mark as delivered

        Returns:
            List of messages ordered by priority and timestamp
        """
        messages = await self.store.get_messages_for_agent(
            agent_id=agent_id,
            status=status or MessageStatus.PENDING,
            message_type=message_type,
            limit=limit,
        )

        if auto_mark_delivered:
            for msg in messages:
                await self.store.update_message_status(msg.id, MessageStatus.DELIVERED)

        self.metrics["messages_received"] += len(messages)

        # Emit events
        for msg in messages:
            await self.event_bus.publish(
                MessageReceivedEvent(
                    aggregate_id=msg.id,
                    source_agent=msg.sender_id,
                    message_id=msg.id,
                    receiver_id=agent_id,
                    sender_id=msg.sender_id,
                    message_type=msg.message_type.value,
                )
            )

        return messages

    async def fetch_payload(self, message: AgentMessage) -> Dict[str, Any]:
        """
        Fetch the full payload for a message.

        If the message has a reference, retrieves the full payload.
        Otherwise returns the inline payload.

        Args:
            message: The message to fetch payload for

        Returns:
            The full payload dictionary
        """
        if message.has_reference:
            payload_json = await self.store.get_large_payload(message.reference)
            if payload_json:
                return json.loads(payload_json)

        return message.payload

    async def acknowledge(
        self,
        message_id: str,
        agent_id: str,
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """
        Acknowledge receipt of a message.

        Args:
            message_id: The message ID to acknowledge
            agent_id: The acknowledging agent's ID
            success: Whether processing was successful
            error: Error message if not successful
        """
        ack = MessageAcknowledgment(
            message_id=message_id,
            agent_id=agent_id,
            success=success,
            error=error,
        )

        await self.store.store_acknowledgment(ack)
        await self.store.update_message_status(message_id, MessageStatus.ACKNOWLEDGED)

        self.metrics["acknowledgments"] += 1

        # Emit event
        await self.event_bus.publish(
            MessageAcknowledgedEvent(
                aggregate_id=message_id,
                source_agent=agent_id,
                message_id=message_id,
                agent_id=agent_id,
                success=success,
            )
        )

    async def subscribe_to_topic(self, agent_id: str, topic: str) -> bool:
        """Subscribe an agent to a broadcast topic."""
        return await self.store.subscribe_to_topic(agent_id, topic)

    async def unsubscribe_from_topic(self, agent_id: str, topic: str) -> bool:
        """Unsubscribe an agent from a topic."""
        return await self.store.unsubscribe_from_topic(agent_id, topic)

    async def get_inbox_stats(self, agent_id: str) -> InboxStats:
        """Get inbox statistics for an agent."""
        return await self.store.get_inbox_stats(agent_id)

    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable[[AgentMessage], Awaitable[None]],
    ) -> None:
        """
        Register a handler for a message type.

        Args:
            message_type: The message type to handle
            handler: Async handler function
        """
        type_key = message_type.value
        if type_key not in self._handlers:
            self._handlers[type_key] = []
        self._handlers[type_key].append(handler)

    async def process_messages(
        self,
        agent_id: str,
        limit: int = MAX_BATCH_SIZE,
    ) -> int:
        """
        Process messages for an agent using registered handlers.

        Args:
            agent_id: The agent's ID
            limit: Maximum messages to process

        Returns:
            Number of messages processed
        """
        messages = await self.receive_messages(agent_id, limit=limit)
        processed = 0

        for msg in messages:
            handlers = self._handlers.get(msg.message_type.value, [])
            for handler in handlers:
                try:
                    await handler(msg)
                except Exception as e:
                    logger.error(
                        f"Handler error for message {msg.id}: {e}",
                        exc_info=True,
                    )

            # Mark as read after processing
            await self.store.update_message_status(msg.id, MessageStatus.READ)
            processed += 1

        return processed

    def _generate_payload_summary(self, payload: Dict[str, Any]) -> str:
        """Generate a summary of a large payload."""
        # Extract key information for summary
        summary_parts = []

        if "task" in payload:
            summary_parts.append(f"Task: {str(payload['task'])[:100]}")
        if "result" in payload:
            result_str = str(payload["result"])
            summary_parts.append(f"Result: {result_str[:100]}...")
        if "query" in payload:
            summary_parts.append(f"Query: {str(payload['query'])[:100]}")

        return "; ".join(summary_parts) if summary_parts else "Large payload stored as reference"

    async def _cleanup_loop(self) -> None:
        """Background task for cleaning up expired messages."""
        while self._running:
            try:
                await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
                deleted = await self.store.cleanup_expired_messages()
                self.metrics["cleanup_runs"] += 1
                if deleted > 0:
                    logger.debug(f"Cleaned up {deleted} expired messages")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    # =========================================================================
    # SWARM INTEGRATION
    # =========================================================================

    async def integrate_with_swarm(
        self,
        controller: ExtremeSwarmController,
    ) -> None:
        """
        Integrate with an ExtremeSwarmController.

        Sets up automatic agent registration and event handling.
        """
        self.swarm_controller = controller

        # Register existing agents
        if controller.topology.queen:
            self.register_agent(controller.topology.queen.agent_id)

        for coord in controller.topology.sub_coordinators.values():
            self.register_agent(coord.agent_id)

        for worker in controller.topology.workers.values():
            self.register_agent(worker.agent_id)

        # Subscribe to swarm events for auto-registration
        from .extreme_swarm import AgentJoinedEvent, AgentLeftEvent

        async def on_agent_joined(event: AgentJoinedEvent) -> None:
            self.register_agent(event.agent_id)

        async def on_agent_left(event: AgentLeftEvent) -> None:
            self.unregister_agent(event.agent_id)

        controller.event_coordinator.subscribe("AgentJoinedEvent", on_agent_joined)
        controller.event_coordinator.subscribe("AgentLeftEvent", on_agent_left)

        logger.info(f"Integrated with swarm {controller.swarm_id}")

    async def send_to_coordinator(
        self,
        sender_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
    ) -> Optional[AgentMessage]:
        """
        Send a message to the swarm coordinator (queen).

        Args:
            sender_id: Sender agent ID
            message_type: Message type
            payload: Message payload

        Returns:
            Sent message or None if no controller
        """
        if not self.swarm_controller or not self.swarm_controller.topology.queen:
            logger.warning("No swarm controller or queen available")
            return None

        queen_id = self.swarm_controller.topology.queen.agent_id
        return await self.send_message(
            sender_id=sender_id,
            receiver_id=queen_id,
            message_type=message_type,
            payload=payload,
            priority=MessagePriority.HIGH,
        )

    async def send_to_workers(
        self,
        sender_id: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        region: Optional[str] = None,
    ) -> List[AgentMessage]:
        """
        Send a message to all workers (optionally in a region).

        Args:
            sender_id: Sender agent ID
            message_type: Message type
            payload: Message payload
            region: Optional region filter

        Returns:
            List of sent messages
        """
        if not self.swarm_controller:
            logger.warning("No swarm controller available")
            return []

        messages = []
        for worker_id, worker in self.swarm_controller.topology.workers.items():
            if region and worker.region != region:
                continue

            msg = await self.send_message(
                sender_id=sender_id,
                receiver_id=worker_id,
                message_type=message_type,
                payload=payload,
            )
            messages.append(msg)

        return messages

    def get_metrics(self) -> Dict[str, Any]:
        """Get communication system metrics."""
        return {
            **self.metrics,
            "registered_agents": len(self._registered_agents),
            "handler_types": list(self._handlers.keys()),
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


async def create_agent_comm_system(
    db_path: Optional[Path] = None,
    event_bus: Optional[EventBus] = None,
    swarm_controller: Optional[ExtremeSwarmController] = None,
) -> AgentCommSystem:
    """
    Factory function to create an agent communication system.

    Args:
        db_path: Path to SQLite database
        event_bus: Event bus for events
        swarm_controller: Optional swarm controller

    Returns:
        Configured and started AgentCommSystem
    """
    store = SQLiteMessageStore(db_path=db_path)
    comm = AgentCommSystem(
        store=store,
        event_bus=event_bus or EventBus(),
        swarm_controller=swarm_controller,
    )
    await comm.start()

    if swarm_controller:
        await comm.integrate_with_swarm(swarm_controller)

    return comm


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "LARGE_PAYLOAD_THRESHOLD",
    "DEFAULT_MESSAGE_TTL_SECONDS",
    "MAX_BATCH_SIZE",
    # Enums
    "MessageType",
    "MessageStatus",
    "MessagePriority",
    # Data classes
    "AgentMessage",
    "MessageAcknowledgment",
    "TopicSubscription",
    "InboxStats",
    # Events
    "MessageSentEvent",
    "MessageReceivedEvent",
    "MessageAcknowledgedEvent",
    "BroadcastSentEvent",
    # Store
    "SQLiteMessageStore",
    # Main system
    "AgentCommSystem",
    # Factory
    "create_agent_comm_system",
]
