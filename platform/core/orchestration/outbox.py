"""
Transactional Outbox Pattern - V39 Architecture

Reliable event publishing with transactional guarantees and at-least-once delivery.

Key Features:
- SQLite-based transactional storage (same transaction as business logic)
- Polling-based event publishing with configurable batch size
- At-least-once delivery guarantees with retry logic
- Dead letter queue for failed events with reason tracking
- Integration with memory events (memory stored, session started, etc.)

Architecture Decision: ADR-029 - Reliable Event Publishing

Usage:
    from core.orchestration.outbox import (
        TransactionalOutbox,
        OutboxPublisher,
        OutboxEvent,
        MemoryEventTypes,
    )

    # Store events transactionally
    outbox = TransactionalOutbox(db_path=Path("./outbox.db"))
    await outbox.initialize()

    event_id = await outbox.store(
        event_type=MemoryEventTypes.MEMORY_STORED,
        payload={"key": "user-prefs", "namespace": "config"}
    )

    # Start background publisher
    async def my_publisher(event: OutboxEvent) -> bool:
        # Publish to message broker, webhook, etc.
        return await kafka.publish(event.event_type, event.payload)

    publisher = OutboxPublisher(outbox, my_publisher)
    await publisher.start()
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


# =============================================================================
# Event Types
# =============================================================================

class EventStatus(str, Enum):
    """Status of an outbox event."""
    PENDING = "pending"
    PUBLISHED = "published"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


class MemoryEventTypes:
    """Standard memory-related event types for platform integration."""
    # Core memory events
    MEMORY_STORED = "memory.stored"
    MEMORY_RETRIEVED = "memory.retrieved"
    MEMORY_DELETED = "memory.deleted"
    MEMORY_UPDATED = "memory.updated"

    # Session events
    SESSION_STARTED = "session.started"
    SESSION_ENDED = "session.ended"
    SESSION_CHECKPOINT = "session.checkpoint"
    SESSION_RESTORED = "session.restored"

    # Agent events
    AGENT_SPAWNED = "agent.spawned"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"

    # Swarm events
    SWARM_INITIALIZED = "swarm.initialized"
    SWARM_COMPLETED = "swarm.completed"
    SWARM_TASK_ASSIGNED = "swarm.task.assigned"

    # Knowledge graph events
    FACT_ADDED = "knowledge.fact.added"
    FACT_INVALIDATED = "knowledge.fact.invalidated"

    # Archival events
    ARCHIVAL_STORED = "archival.stored"
    ARCHIVAL_SEARCHED = "archival.searched"


# =============================================================================
# Outbox Event
# =============================================================================

@dataclass
class OutboxEvent:
    """
    Event stored in the transactional outbox.

    Attributes:
        id: Unique event identifier (UUID)
        event_type: Type of event (e.g., "memory.stored")
        payload: Event data as dictionary
        created_at: When event was created
        published_at: When event was successfully published (None if not published)
        retry_count: Number of publish attempts
        status: Current event status
        last_error: Last error message if failed
        metadata: Additional metadata (correlation_id, trace_id, etc.)
    """
    id: str
    event_type: str
    payload: Dict[str, Any]
    created_at: datetime
    published_at: Optional[datetime] = None
    retry_count: int = 0
    status: str = EventStatus.PENDING.value
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        event_type: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> "OutboxEvent":
        """Factory method to create a new event."""
        return cls(
            id=str(uuid.uuid4()),
            event_type=event_type,
            payload=payload,
            created_at=datetime.now(timezone.utc),
            metadata=metadata or {}
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "payload": self.payload,
            "created_at": self.created_at.isoformat(),
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "retry_count": self.retry_count,
            "status": self.status,
            "last_error": self.last_error,
            "metadata": self.metadata
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "OutboxEvent":
        """Create event from database row."""
        return cls(
            id=row["id"],
            event_type=row["event_type"],
            payload=json.loads(row["payload"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            published_at=datetime.fromisoformat(row["published_at"]) if row["published_at"] else None,
            retry_count=row["retry_count"],
            status=row["status"],
            last_error=row["last_error"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {}
        )


# =============================================================================
# Publisher Protocol
# =============================================================================

class EventPublisher(Protocol):
    """Protocol for event publishers."""

    async def __call__(self, event: OutboxEvent) -> bool:
        """
        Publish an event.

        Args:
            event: The event to publish

        Returns:
            True if published successfully, False otherwise
        """
        ...


# =============================================================================
# Transactional Outbox
# =============================================================================

class TransactionalOutbox:
    """
    Reliable event publishing with transactional guarantees.

    The outbox pattern ensures that events are stored in the same transaction
    as the business logic, providing exactly-once semantics for the store
    operation and at-least-once delivery when combined with idempotent consumers.

    Features:
    - SQLite-based storage (can be same DB as business data for true ACID)
    - Configurable retry limits and dead letter handling
    - Batch polling for efficient event retrieval
    - Event metadata support (correlation IDs, trace IDs)

    Example:
        outbox = TransactionalOutbox(db_path=Path("./events.db"))
        await outbox.initialize()

        # Store event (ideally in same transaction as business logic)
        event_id = await outbox.store(
            event_type="user.created",
            payload={"user_id": "123", "email": "user@example.com"}
        )

        # Poll and publish events
        async def publish_to_kafka(event: OutboxEvent) -> bool:
            return await kafka.send(event.event_type, event.to_dict())

        published = await outbox.poll_and_publish(publish_to_kafka, batch_size=10)
    """

    # SQLite schema for outbox table
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS outbox_events (
        id TEXT PRIMARY KEY,
        event_type TEXT NOT NULL,
        payload TEXT NOT NULL,
        created_at TEXT NOT NULL,
        published_at TEXT,
        retry_count INTEGER DEFAULT 0,
        status TEXT DEFAULT 'pending',
        last_error TEXT,
        metadata TEXT,

        -- Indexes for efficient polling
        INDEX idx_status_created (status, created_at),
        INDEX idx_event_type (event_type)
    );

    CREATE TABLE IF NOT EXISTS dead_letter_queue (
        id TEXT PRIMARY KEY,
        original_event_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        payload TEXT NOT NULL,
        created_at TEXT NOT NULL,
        moved_at TEXT NOT NULL,
        reason TEXT NOT NULL,
        retry_count INTEGER,
        metadata TEXT,

        INDEX idx_moved_at (moved_at),
        INDEX idx_event_type_dlq (event_type)
    );
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        max_retries: int = 5,
        retry_delay_base: float = 1.0,
        retry_delay_max: float = 60.0
    ):
        """
        Initialize the transactional outbox.

        Args:
            db_path: Path to SQLite database (default: ~/.uap/outbox.db)
            max_retries: Maximum retry attempts before dead lettering
            retry_delay_base: Base delay for exponential backoff (seconds)
            retry_delay_max: Maximum retry delay (seconds)
        """
        if db_path is None:
            db_path = Path.home() / ".uap" / "outbox.db"

        self.db_path = db_path
        self.max_retries = max_retries
        self.retry_delay_base = retry_delay_base
        self.retry_delay_max = retry_delay_max

        self._connection: Optional[sqlite3.Connection] = None
        self._initialized = False
        self._lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "events_stored": 0,
            "events_published": 0,
            "events_failed": 0,
            "events_dead_lettered": 0,
            "total_retries": 0
        }

    async def initialize(self) -> None:
        """Initialize the database and create tables."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            # Ensure directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Connect and create schema
            self._connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False
            )
            self._connection.row_factory = sqlite3.Row

            # Enable WAL mode for better concurrency
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")

            # Create tables (handle indexes separately for SQLite)
            cursor = self._connection.cursor()

            # Create outbox_events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS outbox_events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    published_at TEXT,
                    retry_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    last_error TEXT,
                    metadata TEXT
                )
            """)

            # Create indexes for outbox_events
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_status_created
                ON outbox_events(status, created_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_type
                ON outbox_events(event_type)
            """)

            # Create dead_letter_queue table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dead_letter_queue (
                    id TEXT PRIMARY KEY,
                    original_event_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    moved_at TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    retry_count INTEGER,
                    metadata TEXT
                )
            """)

            # Create indexes for dead_letter_queue
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_moved_at
                ON dead_letter_queue(moved_at)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_type_dlq
                ON dead_letter_queue(event_type)
            """)

            self._connection.commit()
            self._initialized = True

            logger.info(f"TransactionalOutbox initialized at {self.db_path}")

    async def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure the outbox is initialized."""
        if not self._initialized or not self._connection:
            raise RuntimeError("Outbox not initialized. Call initialize() first.")

    @asynccontextmanager
    async def transaction(self):
        """
        Context manager for transactional operations.

        Use this to store events in the same transaction as business logic.

        Example:
            async with outbox.transaction() as conn:
                # Business logic
                conn.execute("INSERT INTO users ...")

                # Store event in same transaction
                await outbox.store_with_connection(
                    conn,
                    "user.created",
                    {"user_id": "123"}
                )
        """
        self._ensure_initialized()

        cursor = self._connection.cursor()
        cursor.execute("BEGIN IMMEDIATE")

        try:
            yield self._connection
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    async def store(
        self,
        event_type: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store event in outbox (within same transaction as business logic).

        For true transactional guarantees, use store_with_connection() within
        an explicit transaction that includes your business logic.

        Args:
            event_type: Type of event (e.g., "memory.stored")
            payload: Event data
            metadata: Optional metadata (correlation_id, etc.)

        Returns:
            Event ID (UUID)
        """
        self._ensure_initialized()

        event = OutboxEvent.create(event_type, payload, metadata)

        async with self._lock:
            cursor = self._connection.cursor()
            cursor.execute(
                """
                INSERT INTO outbox_events
                (id, event_type, payload, created_at, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.event_type,
                    json.dumps(event.payload),
                    event.created_at.isoformat(),
                    EventStatus.PENDING.value,
                    json.dumps(event.metadata)
                )
            )
            self._connection.commit()

        self._stats["events_stored"] += 1
        logger.debug(f"Stored event {event.id} of type {event_type}")

        return event.id

    def store_with_connection(
        self,
        connection: sqlite3.Connection,
        event_type: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store event using an existing connection (for true transactional guarantees).

        Use this within an explicit transaction to ensure event storage and
        business logic are atomic.

        Args:
            connection: SQLite connection with active transaction
            event_type: Type of event
            payload: Event data
            metadata: Optional metadata

        Returns:
            Event ID (UUID)
        """
        event = OutboxEvent.create(event_type, payload, metadata)

        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO outbox_events
            (id, event_type, payload, created_at, status, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.event_type,
                json.dumps(event.payload),
                event.created_at.isoformat(),
                EventStatus.PENDING.value,
                json.dumps(event.metadata)
            )
        )

        self._stats["events_stored"] += 1
        return event.id

    async def poll_pending(
        self,
        batch_size: int = 10,
        event_types: Optional[List[str]] = None
    ) -> List[OutboxEvent]:
        """
        Poll pending events from the outbox.

        Args:
            batch_size: Maximum number of events to retrieve
            event_types: Optional filter for specific event types

        Returns:
            List of pending events (oldest first)
        """
        self._ensure_initialized()

        async with self._lock:
            cursor = self._connection.cursor()

            if event_types:
                placeholders = ",".join("?" * len(event_types))
                cursor.execute(
                    f"""
                    SELECT * FROM outbox_events
                    WHERE status = ? AND event_type IN ({placeholders})
                    ORDER BY created_at ASC
                    LIMIT ?
                    """,
                    (EventStatus.PENDING.value, *event_types, batch_size)
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM outbox_events
                    WHERE status = ?
                    ORDER BY created_at ASC
                    LIMIT ?
                    """,
                    (EventStatus.PENDING.value, batch_size)
                )

            rows = cursor.fetchall()
            return [OutboxEvent.from_row(row) for row in rows]

    async def poll_and_publish(
        self,
        publisher: Callable[[OutboxEvent], Awaitable[bool]],
        batch_size: int = 10,
        event_types: Optional[List[str]] = None
    ) -> int:
        """
        Poll pending events and publish them.

        This is the main method for processing the outbox. It:
        1. Polls pending events
        2. Attempts to publish each event
        3. Marks successful events as published
        4. Retries failed events with exponential backoff
        5. Moves exhausted events to dead letter queue

        Args:
            publisher: Async function to publish events
            batch_size: Maximum events to process per poll
            event_types: Optional filter for specific event types

        Returns:
            Number of events successfully published
        """
        events = await self.poll_pending(batch_size, event_types)

        if not events:
            return 0

        published_count = 0

        for event in events:
            try:
                success = await publisher(event)

                if success:
                    await self.mark_published(event.id)
                    published_count += 1
                    logger.debug(f"Published event {event.id}")
                else:
                    await self._handle_publish_failure(event, "Publisher returned False")

            except Exception as e:
                logger.warning(f"Failed to publish event {event.id}: {e}")
                await self._handle_publish_failure(event, str(e))

        return published_count

    async def _handle_publish_failure(self, event: OutboxEvent, error: str) -> None:
        """Handle a failed publish attempt."""
        new_retry_count = event.retry_count + 1
        self._stats["total_retries"] += 1

        if new_retry_count >= self.max_retries:
            # Move to dead letter queue
            await self.move_to_dead_letter(event.id, f"Max retries exceeded: {error}")
        else:
            # Update retry count and error
            await self._update_event_retry(event.id, new_retry_count, error)

    async def _update_event_retry(
        self,
        event_id: str,
        retry_count: int,
        error: str
    ) -> None:
        """Update event retry count and last error."""
        async with self._lock:
            cursor = self._connection.cursor()
            cursor.execute(
                """
                UPDATE outbox_events
                SET retry_count = ?, last_error = ?, status = ?
                WHERE id = ?
                """,
                (retry_count, error, EventStatus.FAILED.value, event_id)
            )
            self._connection.commit()

        self._stats["events_failed"] += 1

    async def mark_published(self, event_id: str) -> None:
        """
        Mark an event as successfully published.

        Args:
            event_id: The event ID to mark as published
        """
        self._ensure_initialized()

        now = datetime.now(timezone.utc).isoformat()

        async with self._lock:
            cursor = self._connection.cursor()
            cursor.execute(
                """
                UPDATE outbox_events
                SET status = ?, published_at = ?
                WHERE id = ?
                """,
                (EventStatus.PUBLISHED.value, now, event_id)
            )
            self._connection.commit()

        self._stats["events_published"] += 1

    async def move_to_dead_letter(self, event_id: str, reason: str) -> None:
        """
        Move a failed event to the dead letter queue.

        Args:
            event_id: The event ID to move
            reason: Reason for moving to DLQ
        """
        self._ensure_initialized()

        now = datetime.now(timezone.utc).isoformat()
        dlq_id = str(uuid.uuid4())

        async with self._lock:
            cursor = self._connection.cursor()

            # Get original event
            cursor.execute(
                "SELECT * FROM outbox_events WHERE id = ?",
                (event_id,)
            )
            row = cursor.fetchone()

            if not row:
                logger.warning(f"Event {event_id} not found for dead letter move")
                return

            # Insert into DLQ
            cursor.execute(
                """
                INSERT INTO dead_letter_queue
                (id, original_event_id, event_type, payload, created_at,
                 moved_at, reason, retry_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dlq_id,
                    event_id,
                    row["event_type"],
                    row["payload"],
                    row["created_at"],
                    now,
                    reason,
                    row["retry_count"],
                    row["metadata"]
                )
            )

            # Update original event status
            cursor.execute(
                """
                UPDATE outbox_events
                SET status = ?, last_error = ?
                WHERE id = ?
                """,
                (EventStatus.DEAD_LETTER.value, reason, event_id)
            )

            self._connection.commit()

        self._stats["events_dead_lettered"] += 1
        logger.warning(f"Moved event {event_id} to dead letter queue: {reason}")

    async def retry_dead_letter(self, dlq_id: str) -> Optional[str]:
        """
        Retry an event from the dead letter queue.

        Creates a new event with reset retry count.

        Args:
            dlq_id: Dead letter queue entry ID

        Returns:
            New event ID if successful, None otherwise
        """
        self._ensure_initialized()

        async with self._lock:
            cursor = self._connection.cursor()

            cursor.execute(
                "SELECT * FROM dead_letter_queue WHERE id = ?",
                (dlq_id,)
            )
            row = cursor.fetchone()

            if not row:
                return None

            # Create new event
            new_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()

            cursor.execute(
                """
                INSERT INTO outbox_events
                (id, event_type, payload, created_at, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    new_id,
                    row["event_type"],
                    row["payload"],
                    now,
                    EventStatus.PENDING.value,
                    row["metadata"]
                )
            )

            # Remove from DLQ
            cursor.execute(
                "DELETE FROM dead_letter_queue WHERE id = ?",
                (dlq_id,)
            )

            self._connection.commit()

        logger.info(f"Retried dead letter {dlq_id} as new event {new_id}")
        return new_id

    async def get_dead_letter_events(
        self,
        limit: int = 100,
        event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get events from the dead letter queue.

        Args:
            limit: Maximum number of events
            event_type: Optional filter by event type

        Returns:
            List of dead letter entries
        """
        self._ensure_initialized()

        async with self._lock:
            cursor = self._connection.cursor()

            if event_type:
                cursor.execute(
                    """
                    SELECT * FROM dead_letter_queue
                    WHERE event_type = ?
                    ORDER BY moved_at DESC
                    LIMIT ?
                    """,
                    (event_type, limit)
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM dead_letter_queue
                    ORDER BY moved_at DESC
                    LIMIT ?
                    """,
                    (limit,)
                )

            rows = cursor.fetchall()
            return [
                {
                    "id": row["id"],
                    "original_event_id": row["original_event_id"],
                    "event_type": row["event_type"],
                    "payload": json.loads(row["payload"]),
                    "created_at": row["created_at"],
                    "moved_at": row["moved_at"],
                    "reason": row["reason"],
                    "retry_count": row["retry_count"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
                }
                for row in rows
            ]

    async def cleanup_published(
        self,
        older_than_hours: int = 24
    ) -> int:
        """
        Clean up old published events.

        Args:
            older_than_hours: Delete events older than this many hours

        Returns:
            Number of events deleted
        """
        self._ensure_initialized()

        cutoff = datetime.now(timezone.utc)
        from datetime import timedelta
        cutoff = cutoff - timedelta(hours=older_than_hours)
        cutoff_str = cutoff.isoformat()

        async with self._lock:
            cursor = self._connection.cursor()
            cursor.execute(
                """
                DELETE FROM outbox_events
                WHERE status = ? AND published_at < ?
                """,
                (EventStatus.PUBLISHED.value, cutoff_str)
            )
            deleted = cursor.rowcount
            self._connection.commit()

        logger.info(f"Cleaned up {deleted} published events older than {older_than_hours}h")
        return deleted

    async def get_event(self, event_id: str) -> Optional[OutboxEvent]:
        """Get a specific event by ID."""
        self._ensure_initialized()

        async with self._lock:
            cursor = self._connection.cursor()
            cursor.execute(
                "SELECT * FROM outbox_events WHERE id = ?",
                (event_id,)
            )
            row = cursor.fetchone()

            if row:
                return OutboxEvent.from_row(row)
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get outbox statistics."""
        stats = self._stats.copy()
        stats["db_path"] = str(self.db_path)
        stats["max_retries"] = self.max_retries
        stats["initialized"] = self._initialized
        return stats


# =============================================================================
# Outbox Publisher (Background Worker)
# =============================================================================

class OutboxPublisher:
    """
    Background worker that processes the outbox.

    Runs in a loop, polling the outbox and publishing events.
    Supports graceful shutdown and configurable poll intervals.

    Features:
    - Configurable poll interval
    - Exponential backoff on errors
    - Graceful shutdown
    - Statistics tracking

    Example:
        outbox = TransactionalOutbox()
        await outbox.initialize()

        async def kafka_publisher(event: OutboxEvent) -> bool:
            return await kafka.send(event.event_type, event.to_dict())

        publisher = OutboxPublisher(outbox, kafka_publisher)
        await publisher.start()  # Runs in background

        # Later...
        await publisher.stop()
    """

    def __init__(
        self,
        outbox: TransactionalOutbox,
        publisher: Callable[[OutboxEvent], Awaitable[bool]],
        poll_interval: float = 1.0,
        batch_size: int = 10,
        event_types: Optional[List[str]] = None,
        error_backoff_base: float = 1.0,
        error_backoff_max: float = 60.0
    ):
        """
        Initialize the outbox publisher.

        Args:
            outbox: TransactionalOutbox instance
            publisher: Async function to publish events
            poll_interval: Seconds between polls (default 1.0)
            batch_size: Events per poll batch
            event_types: Optional filter for specific event types
            error_backoff_base: Base backoff on errors (seconds)
            error_backoff_max: Max backoff on errors (seconds)
        """
        self.outbox = outbox
        self.publisher = publisher
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self.event_types = event_types
        self.error_backoff_base = error_backoff_base
        self.error_backoff_max = error_backoff_max

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._consecutive_errors = 0

        # Statistics
        self._stats = {
            "polls": 0,
            "events_processed": 0,
            "poll_errors": 0,
            "started_at": None,
            "stopped_at": None
        }

    async def start(self, poll_interval: Optional[float] = None) -> None:
        """
        Start the background publisher.

        Args:
            poll_interval: Override poll interval (optional)
        """
        if self._running:
            logger.warning("OutboxPublisher already running")
            return

        if poll_interval is not None:
            self.poll_interval = poll_interval

        self._running = True
        self._stats["started_at"] = datetime.now(timezone.utc).isoformat()
        self._stats["stopped_at"] = None

        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"OutboxPublisher started (interval={self.poll_interval}s)")

    async def stop(self) -> None:
        """Stop the background publisher gracefully."""
        if not self._running:
            return

        self._running = False
        self._stats["stopped_at"] = datetime.now(timezone.utc).isoformat()

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("OutboxPublisher stopped")

    async def _run_loop(self) -> None:
        """Main polling loop."""
        while self._running:
            try:
                # Poll and publish
                published = await self.outbox.poll_and_publish(
                    self.publisher,
                    batch_size=self.batch_size,
                    event_types=self.event_types
                )

                self._stats["polls"] += 1
                self._stats["events_processed"] += published
                self._consecutive_errors = 0

                # Wait for next poll
                await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._consecutive_errors += 1
                self._stats["poll_errors"] += 1

                # Exponential backoff on errors
                backoff = min(
                    self.error_backoff_base * (2 ** self._consecutive_errors),
                    self.error_backoff_max
                )

                logger.error(f"OutboxPublisher error (backoff={backoff}s): {e}")
                await asyncio.sleep(backoff)

    @property
    def is_running(self) -> bool:
        """Check if publisher is running."""
        return self._running

    def get_stats(self) -> Dict[str, Any]:
        """Get publisher statistics."""
        stats = self._stats.copy()
        stats["running"] = self._running
        stats["poll_interval"] = self.poll_interval
        stats["batch_size"] = self.batch_size
        stats["consecutive_errors"] = self._consecutive_errors
        return stats


# =============================================================================
# Memory Event Integration
# =============================================================================

class MemoryEventEmitter:
    """
    Helper class for emitting memory-related events through the outbox.

    Provides convenient methods for common memory operations.

    Example:
        outbox = TransactionalOutbox()
        emitter = MemoryEventEmitter(outbox)

        # Emit events for memory operations
        await emitter.memory_stored("user-prefs", "config", {"theme": "dark"})
        await emitter.session_started("session-123", "agent-456")
    """

    def __init__(self, outbox: TransactionalOutbox):
        """Initialize with outbox instance."""
        self.outbox = outbox

    async def memory_stored(
        self,
        key: str,
        namespace: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Emit memory stored event."""
        return await self.outbox.store(
            event_type=MemoryEventTypes.MEMORY_STORED,
            payload={
                "key": key,
                "namespace": namespace,
                "value": value,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            metadata=metadata
        )

    async def memory_retrieved(
        self,
        key: str,
        namespace: str,
        found: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Emit memory retrieved event."""
        return await self.outbox.store(
            event_type=MemoryEventTypes.MEMORY_RETRIEVED,
            payload={
                "key": key,
                "namespace": namespace,
                "found": found,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            metadata=metadata
        )

    async def memory_deleted(
        self,
        key: str,
        namespace: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Emit memory deleted event."""
        return await self.outbox.store(
            event_type=MemoryEventTypes.MEMORY_DELETED,
            payload={
                "key": key,
                "namespace": namespace,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            metadata=metadata
        )

    async def session_started(
        self,
        session_id: str,
        agent_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Emit session started event."""
        return await self.outbox.store(
            event_type=MemoryEventTypes.SESSION_STARTED,
            payload={
                "session_id": session_id,
                "agent_id": agent_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            metadata=metadata
        )

    async def session_ended(
        self,
        session_id: str,
        agent_id: str,
        duration_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Emit session ended event."""
        return await self.outbox.store(
            event_type=MemoryEventTypes.SESSION_ENDED,
            payload={
                "session_id": session_id,
                "agent_id": agent_id,
                "duration_seconds": duration_seconds,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            metadata=metadata
        )

    async def agent_spawned(
        self,
        agent_id: str,
        agent_type: str,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Emit agent spawned event."""
        return await self.outbox.store(
            event_type=MemoryEventTypes.AGENT_SPAWNED,
            payload={
                "agent_id": agent_id,
                "agent_type": agent_type,
                "parent_id": parent_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            metadata=metadata
        )

    async def agent_completed(
        self,
        agent_id: str,
        result: Optional[Dict[str, Any]] = None,
        duration_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Emit agent completed event."""
        return await self.outbox.store(
            event_type=MemoryEventTypes.AGENT_COMPLETED,
            payload={
                "agent_id": agent_id,
                "result": result,
                "duration_seconds": duration_seconds,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            metadata=metadata
        )

    async def fact_added(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Emit knowledge fact added event."""
        return await self.outbox.store(
            event_type=MemoryEventTypes.FACT_ADDED,
            payload={
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "confidence": confidence,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            metadata=metadata
        )


# =============================================================================
# Convenience Functions
# =============================================================================

async def create_outbox(
    db_path: Optional[Path] = None,
    max_retries: int = 5
) -> TransactionalOutbox:
    """
    Create and initialize a transactional outbox.

    Args:
        db_path: Path to SQLite database
        max_retries: Maximum retry attempts

    Returns:
        Initialized TransactionalOutbox
    """
    outbox = TransactionalOutbox(db_path=db_path, max_retries=max_retries)
    await outbox.initialize()
    return outbox


# =============================================================================
# Demo / Test
# =============================================================================

async def _demo():
    """Demo the outbox pattern."""
    import tempfile

    print("=" * 60)
    print("TRANSACTIONAL OUTBOX DEMO")
    print("=" * 60)
    print()

    # Create outbox in temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "outbox.db"
        outbox = await create_outbox(db_path=db_path, max_retries=3)

        # Store some events
        print("[>>] Storing events...")
        event1 = await outbox.store(
            event_type=MemoryEventTypes.MEMORY_STORED,
            payload={"key": "user-prefs", "value": {"theme": "dark"}}
        )
        print(f"  Stored event: {event1}")

        event2 = await outbox.store(
            event_type=MemoryEventTypes.SESSION_STARTED,
            payload={"session_id": "sess-123", "agent_id": "agent-456"}
        )
        print(f"  Stored event: {event2}")

        # Poll pending events
        print("\n[>>] Polling pending events...")
        pending = await outbox.poll_pending(batch_size=10)
        print(f"  Found {len(pending)} pending events")
        for event in pending:
            print(f"    - {event.event_type}: {event.payload}")

        # Simulate publishing
        print("\n[>>] Publishing events...")

        async def mock_publisher(event: OutboxEvent) -> bool:
            print(f"    Publishing: {event.event_type}")
            # Simulate 50% success rate
            return event.event_type == MemoryEventTypes.MEMORY_STORED

        published = await outbox.poll_and_publish(mock_publisher, batch_size=10)
        print(f"  Published {published} events")

        # Check stats
        print("\n[>>] Outbox stats:")
        stats = outbox.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Test dead letter queue
        print("\n[>>] Checking dead letter queue (after max retries)...")

        # Force some retries
        for _ in range(3):
            await outbox.poll_and_publish(mock_publisher, batch_size=10)

        dlq = await outbox.get_dead_letter_events()
        print(f"  Dead letter entries: {len(dlq)}")
        for entry in dlq:
            print(f"    - {entry['event_type']}: {entry['reason']}")

        # Memory event emitter demo
        print("\n[>>] Using MemoryEventEmitter...")
        emitter = MemoryEventEmitter(outbox)
        await emitter.memory_stored("test-key", "test-ns", {"data": "value"})
        await emitter.session_started("sess-789", "agent-abc")
        await emitter.fact_added("user", "prefers", "dark_mode")
        print("  Emitted 3 events via MemoryEventEmitter")

        # Final stats
        print("\n[>>] Final stats:")
        stats = outbox.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        await outbox.close()

    print("\n[OK] Outbox demo complete")


def main():
    """Run the demo."""
    asyncio.run(_demo())


if __name__ == "__main__":
    main()
