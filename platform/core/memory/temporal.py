"""
Bi-Temporal Memory Model - V40 Architecture (Zep Research)

This module implements a bi-temporal memory model inspired by Zep's research on
time-aware memory systems. Bi-temporal modeling tracks two independent time dimensions:

1. Transaction Time (t_transaction): When a fact was recorded in the system
2. Valid Time (t_valid_from, t_valid_to): When the fact was/is true in reality

This enables powerful temporal queries:
- As-of queries: What did we know at time T?
- Valid-time queries: What was true at time T?
- Bi-temporal queries: What did we know about time T2 at time T1?

Key Features:
- Full bi-temporal support with transaction and valid time tracking
- Fact invalidation with historical version preservation
- Supersedes relationship tracking for fact evolution
- Temporal indexing with B-tree indexes on temporal columns
- Efficient range queries and temporal aggregation
- Backward compatible with existing MemoryEntry (defaults to current time)

Usage:
    from core.memory.temporal import (
        BiTemporalMemory,
        TemporalMemoryEntry,
        create_bitemporal_memory,
    )

    # Create instance
    memory = await create_bitemporal_memory(db_path="/path/to/memory.db")

    # Store a fact with valid time
    entry = await memory.store(
        content="User prefers dark mode",
        valid_from=datetime(2024, 1, 15, tzinfo=timezone.utc),
    )

    # Query what we knew at a specific time
    results = await memory.query_as_of("dark mode", as_of_time=last_month)

    # Query what was true at a specific time
    results = await memory.query_valid_at("preferences", valid_time=yesterday)

    # Full bi-temporal query
    results = await memory.query_bitemporal(
        "preferences",
        transaction_time=yesterday,  # What we knew yesterday...
        valid_time=last_week,        # ...about what was true last week
    )

    # Invalidate a fact (set valid_to)
    await memory.invalidate(entry.id, reason="User changed preference")
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .backends.base import (
    MemoryEntry,
    MemoryNamespace,
    MemoryPriority,
    MemoryTier,
    TierBackend,
    generate_memory_id,
)

logger = logging.getLogger(__name__)


# =============================================================================
# TEMPORAL DATA CLASSES
# =============================================================================


@dataclass
class TemporalMemoryEntry(MemoryEntry):
    """
    Extended memory entry with bi-temporal fields.

    Bi-temporal dimensions:
    - t_created: When the fact was first ingested into memory (immutable)
    - t_valid_from: When the fact became true in reality
    - t_valid_to: When the fact stopped being true (None = still valid)
    - t_transaction: When this version was recorded (for audit trail)

    Relationship tracking:
    - supersedes_id: ID of the entry this one supersedes
    - superseded_by_id: ID of the entry that supersedes this one
    - version: Version number for this fact chain
    """

    # Bi-temporal fields
    t_created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    t_valid_from: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    t_valid_to: Optional[datetime] = None
    t_transaction: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Relationship tracking
    supersedes_id: Optional[str] = None
    superseded_by_id: Optional[str] = None
    version: int = 1

    # Invalidation metadata
    invalidation_reason: Optional[str] = None
    invalidated_at: Optional[datetime] = None

    @property
    def is_currently_valid(self) -> bool:
        """Check if the fact is currently valid (not invalidated)."""
        now = datetime.now(timezone.utc)
        if self.t_valid_to is not None and self.t_valid_to <= now:
            return False
        return self.t_valid_from <= now

    @property
    def is_superseded(self) -> bool:
        """Check if this entry has been superseded by another."""
        return self.superseded_by_id is not None

    def was_valid_at(self, point_in_time: datetime) -> bool:
        """Check if this fact was valid at a specific point in time."""
        if point_in_time < self.t_valid_from:
            return False
        if self.t_valid_to is not None and point_in_time >= self.t_valid_to:
            return False
        return True

    def was_known_at(self, point_in_time: datetime) -> bool:
        """Check if this fact was known (recorded) at a specific point in time."""
        return self.t_transaction <= point_in_time


@dataclass
class TemporalSearchResult:
    """Result from a temporal memory search."""

    entry: TemporalMemoryEntry
    score: float
    temporal_relevance: float  # How relevant based on time proximity
    match_type: str  # exact, semantic, fuzzy, temporal


@dataclass
class TemporalAggregation:
    """Aggregated temporal data for analysis."""

    time_bucket: datetime
    entry_count: int
    avg_validity_duration_hours: float
    invalidation_count: int
    supersession_count: int


# =============================================================================
# SQLITE SCHEMA FOR BI-TEMPORAL MEMORY
# =============================================================================

BITEMPORAL_SCHEMA_VERSION = 1

BITEMPORAL_SCHEMA_SQL = """
-- Schema version tracking for bi-temporal
CREATE TABLE IF NOT EXISTS bitemporal_schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

-- Main bi-temporal memories table
CREATE TABLE IF NOT EXISTS temporal_memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    memory_type TEXT NOT NULL DEFAULT 'fact',
    tier TEXT NOT NULL DEFAULT 'core_memory',
    namespace TEXT,
    priority TEXT NOT NULL DEFAULT 'normal',
    importance REAL NOT NULL DEFAULT 0.5,

    -- Bi-temporal timestamps (core of bi-temporal model)
    t_created TEXT NOT NULL,        -- When ingested into memory (immutable)
    t_valid_from TEXT NOT NULL,     -- When fact became true in reality
    t_valid_to TEXT,                -- When fact stopped being true (NULL = still valid)
    t_transaction TEXT NOT NULL,    -- When this version was recorded

    -- Access tracking
    last_accessed TEXT,
    access_count INTEGER NOT NULL DEFAULT 0,

    -- Relationship tracking for fact evolution
    supersedes_id TEXT,             -- ID of entry this supersedes
    superseded_by_id TEXT,          -- ID of entry that supersedes this
    version INTEGER NOT NULL DEFAULT 1,

    -- Invalidation tracking
    invalidation_reason TEXT,
    invalidated_at TEXT,

    -- Session tracking
    session_id TEXT,

    -- Metadata
    tags TEXT,  -- JSON array
    metadata TEXT,  -- JSON object
    embedding BLOB,  -- Binary embedding vector

    -- Foreign keys
    FOREIGN KEY (supersedes_id) REFERENCES temporal_memories(id),
    FOREIGN KEY (superseded_by_id) REFERENCES temporal_memories(id)
);

-- Full-text search for temporal memories
CREATE VIRTUAL TABLE IF NOT EXISTS temporal_memories_fts USING fts5(
    id,
    content,
    tags,
    tokenize='porter unicode61'
);

-- Temporal fact chains table (tracks evolution of facts)
CREATE TABLE IF NOT EXISTS fact_chains (
    chain_id TEXT PRIMARY KEY,
    root_entry_id TEXT NOT NULL,
    current_entry_id TEXT NOT NULL,
    chain_length INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    topic TEXT,  -- High-level topic for the chain

    FOREIGN KEY (root_entry_id) REFERENCES temporal_memories(id),
    FOREIGN KEY (current_entry_id) REFERENCES temporal_memories(id)
);

-- B-tree indexes on temporal columns for efficient range queries
CREATE INDEX IF NOT EXISTS idx_temporal_t_created ON temporal_memories(t_created);
CREATE INDEX IF NOT EXISTS idx_temporal_t_valid_from ON temporal_memories(t_valid_from);
CREATE INDEX IF NOT EXISTS idx_temporal_t_valid_to ON temporal_memories(t_valid_to);
CREATE INDEX IF NOT EXISTS idx_temporal_t_transaction ON temporal_memories(t_transaction);
CREATE INDEX IF NOT EXISTS idx_temporal_valid_range ON temporal_memories(t_valid_from, t_valid_to);

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_temporal_as_of ON temporal_memories(t_transaction, t_valid_from);
CREATE INDEX IF NOT EXISTS idx_temporal_type_valid ON temporal_memories(memory_type, t_valid_from, t_valid_to);
CREATE INDEX IF NOT EXISTS idx_temporal_namespace_valid ON temporal_memories(namespace, t_valid_from, t_valid_to);

-- Indexes for relationship queries
CREATE INDEX IF NOT EXISTS idx_temporal_supersedes ON temporal_memories(supersedes_id);
CREATE INDEX IF NOT EXISTS idx_temporal_superseded_by ON temporal_memories(superseded_by_id);
CREATE INDEX IF NOT EXISTS idx_temporal_version ON temporal_memories(version DESC);

-- Other useful indexes
CREATE INDEX IF NOT EXISTS idx_temporal_importance ON temporal_memories(importance DESC);
CREATE INDEX IF NOT EXISTS idx_temporal_type ON temporal_memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_temporal_namespace ON temporal_memories(namespace);
"""

BITEMPORAL_TRIGGERS_SQL = """
-- Sync FTS on insert
CREATE TRIGGER IF NOT EXISTS temporal_memories_ai AFTER INSERT ON temporal_memories BEGIN
    INSERT INTO temporal_memories_fts(id, content, tags)
    VALUES (NEW.id, NEW.content, NEW.tags);
END;

-- Sync FTS on update
CREATE TRIGGER IF NOT EXISTS temporal_memories_au AFTER UPDATE ON temporal_memories BEGIN
    UPDATE temporal_memories_fts SET content = NEW.content, tags = NEW.tags WHERE id = NEW.id;
END;

-- Sync FTS on delete
CREATE TRIGGER IF NOT EXISTS temporal_memories_ad AFTER DELETE ON temporal_memories BEGIN
    DELETE FROM temporal_memories_fts WHERE id = OLD.id;
END;
"""


# =============================================================================
# BI-TEMPORAL MEMORY BACKEND
# =============================================================================


class BiTemporalMemory(TierBackend[TemporalMemoryEntry]):
    """
    Bi-temporal memory backend with SQLite storage.

    Implements the bi-temporal model from Zep research, tracking both
    transaction time (when recorded) and valid time (when true in reality).

    Key capabilities:
    - Store facts with valid time ranges
    - Query what was known at a specific time (as-of queries)
    - Query what was true at a specific time (valid-time queries)
    - Full bi-temporal queries combining both dimensions
    - Fact invalidation with audit trail
    - Supersedes relationship tracking
    - Efficient temporal range queries via B-tree indexes
    """

    def __init__(
        self,
        tier: MemoryTier = MemoryTier.CORE_MEMORY,
        db_path: Optional[Path] = None,
        embedding_provider: Optional[Callable[[str], List[float]]] = None,
    ) -> None:
        """
        Initialize bi-temporal memory backend.

        Args:
            tier: Default memory tier for stored entries
            db_path: Path to SQLite database. Defaults to ~/.claude/memory/temporal.db
            embedding_provider: Optional function to generate embeddings
        """
        self.tier = tier
        self.db_path = db_path or Path.home() / ".claude" / "memory" / "temporal.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.embedding_provider = embedding_provider
        self._connection: Optional[sqlite3.Connection] = None

        # Initialize database
        self._init_db()

        logger.info(f"Bi-temporal memory initialized at {self.db_path}")

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._get_connection() as conn:
            # Check schema version
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='bitemporal_schema_version'"
            )
            if not cursor.fetchone():
                # Fresh database - create schema
                conn.executescript(BITEMPORAL_SCHEMA_SQL)
                conn.executescript(BITEMPORAL_TRIGGERS_SQL)
                conn.execute(
                    "INSERT INTO bitemporal_schema_version (version, applied_at) VALUES (?, ?)",
                    (BITEMPORAL_SCHEMA_VERSION, datetime.now(timezone.utc).isoformat()),
                )
                conn.commit()
                logger.info(
                    "Created bi-temporal database with schema v%d",
                    BITEMPORAL_SCHEMA_VERSION,
                )
            else:
                # Check for migrations
                cursor = conn.execute(
                    "SELECT MAX(version) FROM bitemporal_schema_version"
                )
                current_version = cursor.fetchone()[0] or 0
                if current_version < BITEMPORAL_SCHEMA_VERSION:
                    self._migrate(conn, current_version, BITEMPORAL_SCHEMA_VERSION)

    def _migrate(
        self, conn: sqlite3.Connection, from_version: int, to_version: int
    ) -> None:
        """Run schema migrations."""
        logger.info("Migrating bi-temporal database from v%d to v%d", from_version, to_version)
        # Add migration logic here as schema evolves
        conn.execute(
            "INSERT INTO bitemporal_schema_version (version, applied_at) VALUES (?, ?)",
            (to_version, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get database connection with context management."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self.db_path), check_same_thread=False, timeout=30.0
            )
            self._connection.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")

        try:
            yield self._connection
        except Exception:
            self._connection.rollback()
            raise

    def _content_hash(self, content: str) -> str:
        """Generate content hash for deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _row_to_entry(self, row: sqlite3.Row) -> TemporalMemoryEntry:
        """Convert a database row to TemporalMemoryEntry."""

        def parse_datetime(val: Optional[str]) -> Optional[datetime]:
            if val is None:
                return None
            try:
                return datetime.fromisoformat(val)
            except (ValueError, TypeError):
                return None

        return TemporalMemoryEntry(
            id=row["id"],
            content=row["content"],
            tier=MemoryTier(row["tier"]) if row["tier"] else self.tier,
            priority=(
                MemoryPriority(row["priority"])
                if row["priority"]
                else MemoryPriority.NORMAL
            ),
            namespace=(
                MemoryNamespace(row["namespace"]) if row["namespace"] else None
            ),
            content_type=row["memory_type"] or "fact",
            # Bi-temporal fields
            t_created=parse_datetime(row["t_created"]) or datetime.now(timezone.utc),
            t_valid_from=parse_datetime(row["t_valid_from"])
            or datetime.now(timezone.utc),
            t_valid_to=parse_datetime(row["t_valid_to"]),
            t_transaction=parse_datetime(row["t_transaction"])
            or datetime.now(timezone.utc),
            # Access tracking
            last_accessed=parse_datetime(row["last_accessed"]),
            access_count=row["access_count"] or 0,
            # Relationships
            supersedes_id=row["supersedes_id"],
            superseded_by_id=row["superseded_by_id"],
            version=row["version"] or 1,
            # Invalidation
            invalidation_reason=row["invalidation_reason"],
            invalidated_at=parse_datetime(row["invalidated_at"]),
            # Metadata
            tags=json.loads(row["tags"]) if row["tags"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
        )

    # =========================================================================
    # TierBackend Interface Implementation
    # =========================================================================

    async def get(self, key: str) -> Optional[TemporalMemoryEntry]:
        """Get memory entry by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM temporal_memories WHERE id = ?", (key,)
            )
            row = cursor.fetchone()
            if row:
                # Update access tracking
                conn.execute(
                    "UPDATE temporal_memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                    (datetime.now(timezone.utc).isoformat(), key),
                )
                conn.commit()
                return self._row_to_entry(row)
        return None

    async def put(self, key: str, value: TemporalMemoryEntry) -> None:
        """Store memory entry with bi-temporal tracking."""
        now = datetime.now(timezone.utc)
        content_hash = self._content_hash(value.content)

        # Generate embedding if provider available
        embedding_blob = None
        if self.embedding_provider and not value.embedding:
            try:
                embedding = self.embedding_provider(value.content)
                if embedding:
                    value.embedding = embedding
                    embedding_blob = json.dumps(embedding)
            except Exception as e:
                logger.warning("Failed to generate embedding: %s", e)
        elif value.embedding:
            embedding_blob = json.dumps(value.embedding)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO temporal_memories (
                    id, content, content_hash, memory_type, tier, namespace, priority,
                    importance, t_created, t_valid_from, t_valid_to, t_transaction,
                    last_accessed, access_count, supersedes_id, superseded_by_id,
                    version, invalidation_reason, invalidated_at, session_id,
                    tags, metadata, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    key or value.id,
                    value.content,
                    content_hash,
                    value.content_type,
                    value.tier.value if value.tier else self.tier.value,
                    value.namespace.value if value.namespace else None,
                    (
                        value.priority.value
                        if value.priority
                        else MemoryPriority.NORMAL.value
                    ),
                    value.metadata.get("importance", 0.5),
                    value.t_created.isoformat(),
                    value.t_valid_from.isoformat(),
                    value.t_valid_to.isoformat() if value.t_valid_to else None,
                    value.t_transaction.isoformat(),
                    value.last_accessed.isoformat() if value.last_accessed else None,
                    value.access_count,
                    value.supersedes_id,
                    value.superseded_by_id,
                    value.version,
                    value.invalidation_reason,
                    value.invalidated_at.isoformat() if value.invalidated_at else None,
                    value.metadata.get("session_id"),
                    json.dumps(value.tags) if value.tags else None,
                    json.dumps(value.metadata) if value.metadata else None,
                    embedding_blob,
                ),
            )
            conn.commit()

    async def delete(self, key: str) -> bool:
        """Delete memory entry by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM temporal_memories WHERE id = ?", (key,)
            )
            conn.commit()
            return cursor.rowcount > 0

    async def search(self, query: str, limit: int = 10) -> List[TemporalMemoryEntry]:
        """Search memories using full-text search (only currently valid facts)."""
        return await self.search_valid_now(query, limit)

    async def list_all(self) -> List[TemporalMemoryEntry]:
        """List all entries (including historical)."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM temporal_memories ORDER BY t_transaction DESC"
            )
            return [self._row_to_entry(row) for row in cursor.fetchall()]

    async def count(self) -> int:
        """Get total entry count (including historical)."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM temporal_memories")
            return cursor.fetchone()[0]

    # =========================================================================
    # Bi-Temporal Store Operations
    # =========================================================================

    async def store(
        self,
        content: str,
        valid_from: Optional[datetime] = None,
        valid_to: Optional[datetime] = None,
        memory_type: str = "fact",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        namespace: Optional[MemoryNamespace] = None,
        supersedes_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TemporalMemoryEntry:
        """
        Store a fact with bi-temporal tracking.

        Args:
            content: The fact content
            valid_from: When the fact became true (defaults to now)
            valid_to: When the fact stopped being true (None = still valid)
            memory_type: Type of memory (fact, decision, learning, preference)
            importance: Importance score 0.0-1.0
            tags: Optional tags for categorization
            namespace: Memory namespace for organization
            supersedes_id: ID of entry this supersedes (for fact evolution)
            metadata: Additional metadata

        Returns:
            The stored TemporalMemoryEntry
        """
        now = datetime.now(timezone.utc)

        # Generate ID
        memory_id = generate_memory_id(content, f"temp_{memory_type}_")

        # Handle supersession
        version = 1
        if supersedes_id:
            # Get the superseded entry
            superseded = await self.get(supersedes_id)
            if superseded:
                version = superseded.version + 1
                # Mark the old entry as superseded
                await self._mark_superseded(supersedes_id, memory_id)

        entry = TemporalMemoryEntry(
            id=memory_id,
            content=content,
            tier=self.tier,
            priority=MemoryPriority.HIGH if importance > 0.7 else MemoryPriority.NORMAL,
            content_type=memory_type,
            namespace=namespace,
            # Bi-temporal fields
            t_created=now,
            t_valid_from=valid_from or now,
            t_valid_to=valid_to,
            t_transaction=now,
            # Relationships
            supersedes_id=supersedes_id,
            version=version,
            # Metadata
            tags=tags or [],
            metadata={**(metadata or {}), "importance": importance},
        )

        await self.put(memory_id, entry)
        return entry

    async def _mark_superseded(self, entry_id: str, superseded_by_id: str) -> None:
        """Mark an entry as superseded by another."""
        now = datetime.now(timezone.utc)
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE temporal_memories
                SET superseded_by_id = ?, t_valid_to = ?
                WHERE id = ?
            """,
                (superseded_by_id, now.isoformat(), entry_id),
            )
            conn.commit()

    # =========================================================================
    # Bi-Temporal Query Operations
    # =========================================================================

    async def query_as_of(
        self,
        query: str,
        as_of_time: datetime,
        limit: int = 10,
        include_superseded: bool = False,
    ) -> List[TemporalSearchResult]:
        """
        As-of query: What did we know at time T?

        Returns facts that were recorded (t_transaction) before or at as_of_time.
        This lets you "go back in time" to see the state of knowledge at a past point.

        Args:
            query: Search query text
            as_of_time: The point in time to query knowledge state
            limit: Maximum results to return
            include_superseded: Whether to include superseded entries

        Returns:
            List of TemporalSearchResult ordered by relevance
        """
        results: List[TemporalSearchResult] = []

        with self._get_connection() as conn:
            # Build query based on whether to include superseded
            superseded_clause = (
                "" if include_superseded else "AND superseded_by_id IS NULL"
            )

            try:
                # FTS search with transaction time filter
                cursor = conn.execute(
                    f"""
                    SELECT m.*, bm25(temporal_memories_fts) as score
                    FROM temporal_memories m
                    JOIN temporal_memories_fts fts ON m.id = fts.id
                    WHERE temporal_memories_fts MATCH ?
                    AND m.t_transaction <= ?
                    {superseded_clause}
                    ORDER BY score
                    LIMIT ?
                """,
                    (query, as_of_time.isoformat(), limit * 2),
                )

                for row in cursor.fetchall():
                    entry = self._row_to_entry(row)
                    score = -row["score"]  # BM25 scores are negative
                    # Calculate temporal relevance (closer to as_of_time = higher)
                    time_diff = (as_of_time - entry.t_transaction).total_seconds()
                    temporal_relevance = max(0, 1 - (time_diff / (86400 * 30)))  # 30 day decay

                    results.append(
                        TemporalSearchResult(
                            entry=entry,
                            score=score,
                            temporal_relevance=temporal_relevance,
                            match_type="fts_as_of",
                        )
                    )

            except sqlite3.OperationalError:
                # Fallback to LIKE search
                cursor = conn.execute(
                    f"""
                    SELECT * FROM temporal_memories
                    WHERE content LIKE ?
                    AND t_transaction <= ?
                    {superseded_clause}
                    ORDER BY t_transaction DESC
                    LIMIT ?
                """,
                    (f"%{query}%", as_of_time.isoformat(), limit),
                )

                for row in cursor.fetchall():
                    entry = self._row_to_entry(row)
                    results.append(
                        TemporalSearchResult(
                            entry=entry,
                            score=0.5,
                            temporal_relevance=0.5,
                            match_type="like_as_of",
                        )
                    )

        # Sort by combined score
        results.sort(key=lambda x: x.score * x.temporal_relevance, reverse=True)
        return results[:limit]

    async def query_valid_at(
        self,
        query: str,
        valid_time: datetime,
        limit: int = 10,
    ) -> List[TemporalSearchResult]:
        """
        Valid-time query: What was true at time T?

        Returns facts that were valid (t_valid_from <= T < t_valid_to) at valid_time.
        This lets you query historical truth regardless of when it was recorded.

        Args:
            query: Search query text
            valid_time: The point in time to check validity
            limit: Maximum results to return

        Returns:
            List of TemporalSearchResult ordered by relevance
        """
        results: List[TemporalSearchResult] = []

        with self._get_connection() as conn:
            try:
                # FTS search with validity time filter
                cursor = conn.execute(
                    """
                    SELECT m.*, bm25(temporal_memories_fts) as score
                    FROM temporal_memories m
                    JOIN temporal_memories_fts fts ON m.id = fts.id
                    WHERE temporal_memories_fts MATCH ?
                    AND m.t_valid_from <= ?
                    AND (m.t_valid_to IS NULL OR m.t_valid_to > ?)
                    ORDER BY score
                    LIMIT ?
                """,
                    (query, valid_time.isoformat(), valid_time.isoformat(), limit * 2),
                )

                for row in cursor.fetchall():
                    entry = self._row_to_entry(row)
                    score = -row["score"]

                    # Calculate temporal relevance (how long has this been valid?)
                    valid_duration = (valid_time - entry.t_valid_from).total_seconds()
                    temporal_relevance = min(1.0, valid_duration / (86400 * 7))  # 7 day scale

                    results.append(
                        TemporalSearchResult(
                            entry=entry,
                            score=score,
                            temporal_relevance=temporal_relevance,
                            match_type="fts_valid_at",
                        )
                    )

            except sqlite3.OperationalError:
                # Fallback to LIKE search
                cursor = conn.execute(
                    """
                    SELECT * FROM temporal_memories
                    WHERE content LIKE ?
                    AND t_valid_from <= ?
                    AND (t_valid_to IS NULL OR t_valid_to > ?)
                    ORDER BY t_valid_from DESC
                    LIMIT ?
                """,
                    (f"%{query}%", valid_time.isoformat(), valid_time.isoformat(), limit),
                )

                for row in cursor.fetchall():
                    entry = self._row_to_entry(row)
                    results.append(
                        TemporalSearchResult(
                            entry=entry,
                            score=0.5,
                            temporal_relevance=0.5,
                            match_type="like_valid_at",
                        )
                    )

        results.sort(key=lambda x: x.score * x.temporal_relevance, reverse=True)
        return results[:limit]

    async def query_bitemporal(
        self,
        query: str,
        transaction_time: datetime,
        valid_time: datetime,
        limit: int = 10,
    ) -> List[TemporalSearchResult]:
        """
        Full bi-temporal query: What did we know about time T2 at time T1?

        This is the most powerful query, combining both dimensions:
        - transaction_time (T1): When the knowledge was recorded
        - valid_time (T2): When the fact was supposed to be true

        Example: "What did we know yesterday about last week's preferences?"

        Args:
            query: Search query text
            transaction_time: The point in time of knowledge state (T1)
            valid_time: The point in time being queried about (T2)
            limit: Maximum results to return

        Returns:
            List of TemporalSearchResult ordered by relevance
        """
        results: List[TemporalSearchResult] = []

        with self._get_connection() as conn:
            try:
                # FTS with both temporal filters
                cursor = conn.execute(
                    """
                    SELECT m.*, bm25(temporal_memories_fts) as score
                    FROM temporal_memories m
                    JOIN temporal_memories_fts fts ON m.id = fts.id
                    WHERE temporal_memories_fts MATCH ?
                    AND m.t_transaction <= ?
                    AND m.t_valid_from <= ?
                    AND (m.t_valid_to IS NULL OR m.t_valid_to > ?)
                    ORDER BY score
                    LIMIT ?
                """,
                    (
                        query,
                        transaction_time.isoformat(),
                        valid_time.isoformat(),
                        valid_time.isoformat(),
                        limit * 2,
                    ),
                )

                for row in cursor.fetchall():
                    entry = self._row_to_entry(row)
                    score = -row["score"]

                    # Combined temporal relevance
                    tx_diff = (transaction_time - entry.t_transaction).total_seconds()
                    tx_relevance = max(0, 1 - (tx_diff / (86400 * 30)))

                    valid_duration = (valid_time - entry.t_valid_from).total_seconds()
                    valid_relevance = min(1.0, valid_duration / (86400 * 7))

                    temporal_relevance = (tx_relevance + valid_relevance) / 2

                    results.append(
                        TemporalSearchResult(
                            entry=entry,
                            score=score,
                            temporal_relevance=temporal_relevance,
                            match_type="fts_bitemporal",
                        )
                    )

            except sqlite3.OperationalError:
                # Fallback to LIKE search
                cursor = conn.execute(
                    """
                    SELECT * FROM temporal_memories
                    WHERE content LIKE ?
                    AND t_transaction <= ?
                    AND t_valid_from <= ?
                    AND (t_valid_to IS NULL OR t_valid_to > ?)
                    ORDER BY t_transaction DESC
                    LIMIT ?
                """,
                    (
                        f"%{query}%",
                        transaction_time.isoformat(),
                        valid_time.isoformat(),
                        valid_time.isoformat(),
                        limit,
                    ),
                )

                for row in cursor.fetchall():
                    entry = self._row_to_entry(row)
                    results.append(
                        TemporalSearchResult(
                            entry=entry,
                            score=0.5,
                            temporal_relevance=0.5,
                            match_type="like_bitemporal",
                        )
                    )

        results.sort(key=lambda x: x.score * x.temporal_relevance, reverse=True)
        return results[:limit]

    async def search_valid_now(
        self, query: str, limit: int = 10
    ) -> List[TemporalMemoryEntry]:
        """
        Search only currently valid facts (convenience method).

        Args:
            query: Search query text
            limit: Maximum results to return

        Returns:
            List of currently valid TemporalMemoryEntry
        """
        now = datetime.now(timezone.utc)
        results = await self.query_valid_at(query, now, limit)
        return [r.entry for r in results]

    # =========================================================================
    # Fact Invalidation
    # =========================================================================

    async def invalidate(
        self,
        entry_id: str,
        reason: Optional[str] = None,
        valid_to: Optional[datetime] = None,
    ) -> bool:
        """
        Invalidate a fact (mark as no longer valid).

        This sets t_valid_to to mark when the fact stopped being true,
        while preserving the historical record for audit purposes.

        Args:
            entry_id: ID of the entry to invalidate
            reason: Reason for invalidation
            valid_to: When the fact stopped being true (defaults to now)

        Returns:
            True if entry was invalidated, False if not found
        """
        now = datetime.now(timezone.utc)
        valid_to = valid_to or now

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE temporal_memories
                SET t_valid_to = ?,
                    invalidation_reason = ?,
                    invalidated_at = ?
                WHERE id = ?
                AND t_valid_to IS NULL
            """,
                (
                    valid_to.isoformat(),
                    reason,
                    now.isoformat(),
                    entry_id,
                ),
            )
            conn.commit()
            return cursor.rowcount > 0

    async def get_invalidated(
        self, limit: int = 50, since: Optional[datetime] = None
    ) -> List[TemporalMemoryEntry]:
        """
        Get invalidated facts for audit/review.

        Args:
            limit: Maximum results
            since: Only get facts invalidated after this time

        Returns:
            List of invalidated TemporalMemoryEntry
        """
        with self._get_connection() as conn:
            if since:
                cursor = conn.execute(
                    """
                    SELECT * FROM temporal_memories
                    WHERE invalidated_at IS NOT NULL
                    AND invalidated_at >= ?
                    ORDER BY invalidated_at DESC
                    LIMIT ?
                """,
                    (since.isoformat(), limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM temporal_memories
                    WHERE invalidated_at IS NOT NULL
                    ORDER BY invalidated_at DESC
                    LIMIT ?
                """,
                    (limit,),
                )
            return [self._row_to_entry(row) for row in cursor.fetchall()]

    # =========================================================================
    # Supersession & Fact Evolution
    # =========================================================================

    async def supersede(
        self,
        old_entry_id: str,
        new_content: str,
        valid_from: Optional[datetime] = None,
        reason: Optional[str] = None,
        **kwargs,
    ) -> TemporalMemoryEntry:
        """
        Create a new fact that supersedes an existing one.

        This is used when a fact evolves or is corrected. The old fact
        is marked as superseded and its valid_to is set, while the new
        fact tracks the relationship.

        Args:
            old_entry_id: ID of the entry being superseded
            new_content: Content for the new entry
            valid_from: When the new fact became true (defaults to now)
            reason: Reason for the supersession
            **kwargs: Additional arguments passed to store()

        Returns:
            The new TemporalMemoryEntry that supersedes the old one
        """
        # Invalidate the old entry
        await self.invalidate(old_entry_id, reason=reason, valid_to=valid_from)

        # Store the new entry with supersedes relationship
        return await self.store(
            content=new_content,
            valid_from=valid_from,
            supersedes_id=old_entry_id,
            metadata={**kwargs.get("metadata", {}), "supersession_reason": reason},
            **{k: v for k, v in kwargs.items() if k != "metadata"},
        )

    async def get_fact_history(self, entry_id: str) -> List[TemporalMemoryEntry]:
        """
        Get the full history of a fact (all versions).

        Traces the supersedes chain backward and superseded_by chain forward
        to reconstruct the complete evolution of a fact.

        Args:
            entry_id: ID of any entry in the chain

        Returns:
            List of TemporalMemoryEntry in chronological order (oldest first)
        """
        history: List[TemporalMemoryEntry] = []
        seen_ids: set = set()

        # First, get the entry
        entry = await self.get(entry_id)
        if not entry:
            return []

        # Trace backward through supersedes chain
        current_id = entry.supersedes_id
        backward_chain: List[TemporalMemoryEntry] = []
        while current_id and current_id not in seen_ids:
            seen_ids.add(current_id)
            prev_entry = await self.get(current_id)
            if prev_entry:
                backward_chain.append(prev_entry)
                current_id = prev_entry.supersedes_id
            else:
                break

        # Reverse to get oldest first
        history.extend(reversed(backward_chain))

        # Add the starting entry
        history.append(entry)
        seen_ids.add(entry_id)

        # Trace forward through superseded_by chain
        current_id = entry.superseded_by_id
        while current_id and current_id not in seen_ids:
            seen_ids.add(current_id)
            next_entry = await self.get(current_id)
            if next_entry:
                history.append(next_entry)
                current_id = next_entry.superseded_by_id
            else:
                break

        return history

    async def get_current_version(self, entry_id: str) -> Optional[TemporalMemoryEntry]:
        """
        Get the current (non-superseded) version of a fact chain.

        Args:
            entry_id: ID of any entry in the chain

        Returns:
            The current TemporalMemoryEntry or None if not found
        """
        history = await self.get_fact_history(entry_id)
        if not history:
            return None
        # Return the last non-superseded entry
        for entry in reversed(history):
            if not entry.is_superseded:
                return entry
        return history[-1]  # Return latest even if superseded

    # =========================================================================
    # Temporal Aggregation & Analytics
    # =========================================================================

    async def get_temporal_aggregation(
        self,
        bucket_size_hours: int = 24,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[TemporalAggregation]:
        """
        Get temporal aggregation statistics.

        Args:
            bucket_size_hours: Size of time buckets in hours
            start_time: Start of analysis period
            end_time: End of analysis period

        Returns:
            List of TemporalAggregation for each time bucket
        """
        now = datetime.now(timezone.utc)
        end_time = end_time or now
        start_time = start_time or datetime(
            end_time.year, end_time.month, end_time.day, tzinfo=timezone.utc
        ) - timedelta(days=30)

        aggregations: List[TemporalAggregation] = []

        with self._get_connection() as conn:
            # Calculate stats per bucket
            current = start_time
            while current < end_time:
                bucket_end = current + timedelta(hours=bucket_size_hours)

                # Count entries created in this bucket
                cursor = conn.execute(
                    """
                    SELECT
                        COUNT(*) as entry_count,
                        AVG(
                            CASE
                                WHEN t_valid_to IS NOT NULL
                                THEN (julianday(t_valid_to) - julianday(t_valid_from)) * 24
                                ELSE NULL
                            END
                        ) as avg_validity_hours,
                        SUM(CASE WHEN invalidated_at IS NOT NULL THEN 1 ELSE 0 END) as invalidation_count,
                        SUM(CASE WHEN superseded_by_id IS NOT NULL THEN 1 ELSE 0 END) as supersession_count
                    FROM temporal_memories
                    WHERE t_created >= ? AND t_created < ?
                """,
                    (current.isoformat(), bucket_end.isoformat()),
                )

                row = cursor.fetchone()
                aggregations.append(
                    TemporalAggregation(
                        time_bucket=current,
                        entry_count=row["entry_count"] or 0,
                        avg_validity_duration_hours=row["avg_validity_hours"] or 0,
                        invalidation_count=row["invalidation_count"] or 0,
                        supersession_count=row["supersession_count"] or 0,
                    )
                )

                current = bucket_end

        return aggregations

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive bi-temporal memory statistics."""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM temporal_memories").fetchone()[0]
            currently_valid = conn.execute(
                """
                SELECT COUNT(*) FROM temporal_memories
                WHERE t_valid_to IS NULL AND superseded_by_id IS NULL
            """
            ).fetchone()[0]
            invalidated = conn.execute(
                "SELECT COUNT(*) FROM temporal_memories WHERE invalidated_at IS NOT NULL"
            ).fetchone()[0]
            superseded = conn.execute(
                "SELECT COUNT(*) FROM temporal_memories WHERE superseded_by_id IS NOT NULL"
            ).fetchone()[0]

            by_type = dict(
                conn.execute(
                    "SELECT memory_type, COUNT(*) FROM temporal_memories GROUP BY memory_type"
                ).fetchall()
            )

            return {
                "total_entries": total,
                "currently_valid": currently_valid,
                "invalidated": invalidated,
                "superseded": superseded,
                "by_type": by_type,
                "storage_path": str(self.db_path),
                "db_size_bytes": (
                    self.db_path.stat().st_size if self.db_path.exists() else 0
                ),
            }

    # =========================================================================
    # Migration Support
    # =========================================================================

    async def migrate_from_memory_entry(self, entry: MemoryEntry) -> TemporalMemoryEntry:
        """
        Migrate a standard MemoryEntry to TemporalMemoryEntry.

        For backward compatibility, this creates a temporal entry with
        default temporal values (created_at for all timestamps).

        Args:
            entry: Standard MemoryEntry to migrate

        Returns:
            The created TemporalMemoryEntry
        """
        now = datetime.now(timezone.utc)
        created_at = entry.created_at or now

        temporal_entry = TemporalMemoryEntry(
            id=entry.id,
            content=entry.content,
            tier=entry.tier,
            priority=entry.priority,
            namespace=entry.namespace,
            content_type=entry.content_type,
            # Use created_at for temporal defaults (backward compatible)
            t_created=created_at,
            t_valid_from=created_at,
            t_valid_to=None,
            t_transaction=now,  # Migration transaction time
            # Preserve metadata
            last_accessed=entry.last_accessed,
            access_count=entry.access_count,
            tags=entry.tags,
            metadata={**entry.metadata, "migrated_from": "MemoryEntry"},
            embedding=entry.embedding,
        )

        await self.put(temporal_entry.id, temporal_entry)
        return temporal_entry

    async def bulk_migrate(
        self, entries: List[MemoryEntry]
    ) -> Tuple[int, int]:
        """
        Bulk migrate MemoryEntry instances to temporal format.

        Args:
            entries: List of MemoryEntry to migrate

        Returns:
            Tuple of (success_count, failure_count)
        """
        success = 0
        failures = 0

        for entry in entries:
            try:
                await self.migrate_from_memory_entry(entry)
                success += 1
            except Exception as e:
                logger.warning(f"Failed to migrate entry {entry.id}: {e}")
                failures += 1

        return success, failures

    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None


# =============================================================================
# IMPORT TIMEDELTA FOR AGGREGATION
# =============================================================================

from datetime import timedelta


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

_bitemporal_instance: Optional[BiTemporalMemory] = None


async def create_bitemporal_memory(
    db_path: Optional[Path] = None,
    embedding_provider: Optional[Callable[[str], List[float]]] = None,
) -> BiTemporalMemory:
    """
    Create or get a BiTemporalMemory instance.

    Args:
        db_path: Path to SQLite database
        embedding_provider: Optional embedding function

    Returns:
        Configured BiTemporalMemory instance
    """
    global _bitemporal_instance
    if _bitemporal_instance is None:
        _bitemporal_instance = BiTemporalMemory(
            db_path=db_path, embedding_provider=embedding_provider
        )
    return _bitemporal_instance


def get_bitemporal_memory() -> Optional[BiTemporalMemory]:
    """Get the current BiTemporalMemory instance if initialized."""
    return _bitemporal_instance


def reset_bitemporal_memory() -> None:
    """Reset the singleton instance (for testing)."""
    global _bitemporal_instance
    if _bitemporal_instance:
        _bitemporal_instance.close()
    _bitemporal_instance = None


__all__ = [
    # Main classes
    "BiTemporalMemory",
    "TemporalMemoryEntry",
    "TemporalSearchResult",
    "TemporalAggregation",
    # Factory functions
    "create_bitemporal_memory",
    "get_bitemporal_memory",
    "reset_bitemporal_memory",
    # Schema constants
    "BITEMPORAL_SCHEMA_VERSION",
]
