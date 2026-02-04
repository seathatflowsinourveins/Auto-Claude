"""
SQLite Backend - Real Persistent Cross-Session Memory

This module provides REAL file-based persistence for cross-session memory
using SQLite for fast queries and JSON export for portability.

Features:
- SQLite database for structured queries
- Full-text search with FTS5
- Embedding-based semantic search (via external providers)
- JSON export/import for portability
- Session tracking with metadata
- Memory consolidation and deduplication

Storage:
    ~/.claude/memory/
        memory.db          # SQLite database
        embeddings.json    # Cached embeddings
        sessions/          # Session snapshots
        exports/           # Portable exports

V36 Architecture: Implements TierBackend interface for memory tiers integration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import (
    MemoryEntry,
    MemoryLayer,
    MemoryNamespace,
    MemoryPriority,
    MemoryTier,
    TierBackend,
    TTL_CONFIG,
    generate_memory_id,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SQLite Schema
# =============================================================================

SCHEMA_VERSION = 2

SCHEMA_SQL = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

-- Main memories table
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    memory_type TEXT NOT NULL DEFAULT 'context',
    tier TEXT NOT NULL DEFAULT 'main_context',
    namespace TEXT,
    priority TEXT NOT NULL DEFAULT 'normal',
    importance REAL NOT NULL DEFAULT 0.5,

    -- Timestamps
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_accessed TEXT,
    valid_from TEXT,
    valid_to TEXT,

    -- Access tracking
    access_count INTEGER NOT NULL DEFAULT 0,

    -- Session tracking
    session_id TEXT,

    -- Metadata
    tags TEXT,  -- JSON array
    metadata TEXT,  -- JSON object
    embedding BLOB,  -- Binary embedding vector

    -- V40: Forgetting curve support
    strength REAL NOT NULL DEFAULT 1.0,
    decay_rate REAL NOT NULL DEFAULT 0.15,
    last_reinforced TEXT,
    reinforcement_count INTEGER NOT NULL DEFAULT 0,

    -- Indexes
    UNIQUE(content_hash)
);

-- Full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    id,
    content,
    tags,
    tokenize='porter unicode61'
);

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    task_summary TEXT,
    project_path TEXT,
    memories_created INTEGER NOT NULL DEFAULT 0,
    decisions TEXT,  -- JSON array
    learnings TEXT,  -- JSON array
    metadata TEXT  -- JSON object
);

-- Index for common queries
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);

-- V40: Forgetting curve indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_memories_strength ON memories(strength);
CREATE INDEX IF NOT EXISTS idx_memories_reinforced ON memories(last_reinforced);
"""

TRIGGER_SQL = """
-- Sync FTS on insert
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(id, content, tags)
    VALUES (NEW.id, NEW.content, NEW.tags);
END;

-- Sync FTS on update
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    UPDATE memories_fts SET content = NEW.content, tags = NEW.tags WHERE id = NEW.id;
END;

-- Sync FTS on delete
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    DELETE FROM memories_fts WHERE id = OLD.id;
END;
"""


# =============================================================================
# SQLite Persistent Backend
# =============================================================================

class SQLiteTierBackend(TierBackend[MemoryEntry]):
    """
    SQLite-based persistent storage for cross-session memory.

    Provides:
    - Real file persistence (survives restarts)
    - Full-text search with FTS5
    - Semantic search with embeddings
    - Session tracking
    - Memory consolidation

    Storage location: ~/.claude/memory/memory.db
    """

    def __init__(
        self,
        tier: MemoryTier = MemoryTier.ARCHIVAL_MEMORY,
        db_path: Optional[Path] = None,
        embedding_provider: Optional[Callable[[str], List[float]]] = None
    ) -> None:
        """Initialize SQLite backend.

        Args:
            tier: Memory tier this backend serves
            db_path: Path to SQLite database. Defaults to ~/.claude/memory/memory.db
            embedding_provider: Optional function to generate embeddings for semantic search
        """
        self.tier = tier
        self.db_path = db_path or Path.home() / ".claude" / "memory" / "memory.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.embedding_provider = embedding_provider
        self._connection: Optional[sqlite3.Connection] = None

        # Initialize database
        self._init_db()

        logger.info(f"SQLite backend initialized at {self.db_path}")

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._get_connection() as conn:
            # Check schema version
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            )
            if not cursor.fetchone():
                # Fresh database - create schema
                conn.executescript(SCHEMA_SQL)
                conn.executescript(TRIGGER_SQL)
                conn.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (SCHEMA_VERSION, datetime.now(timezone.utc).isoformat())
                )
                conn.commit()
                logger.info("Created new SQLite database with schema v%d", SCHEMA_VERSION)
            else:
                # Check for migrations
                cursor = conn.execute("SELECT MAX(version) FROM schema_version")
                current_version = cursor.fetchone()[0] or 0
                if current_version < SCHEMA_VERSION:
                    self._migrate(conn, current_version, SCHEMA_VERSION)

    def _migrate(self, conn: sqlite3.Connection, from_version: int, to_version: int) -> None:
        """Run schema migrations."""
        logger.info("Migrating database from v%d to v%d", from_version, to_version)

        # V1 -> V2: Add forgetting curve columns
        if from_version < 2 and to_version >= 2:
            migration_statements = [
                "ALTER TABLE memories ADD COLUMN strength REAL NOT NULL DEFAULT 1.0",
                "ALTER TABLE memories ADD COLUMN decay_rate REAL NOT NULL DEFAULT 0.15",
                "ALTER TABLE memories ADD COLUMN last_reinforced TEXT",
                "ALTER TABLE memories ADD COLUMN reinforcement_count INTEGER NOT NULL DEFAULT 0",
                "CREATE INDEX IF NOT EXISTS idx_memories_strength ON memories(strength)",
                "CREATE INDEX IF NOT EXISTS idx_memories_reinforced ON memories(last_reinforced)",
            ]
            for stmt in migration_statements:
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    # Ignore "column already exists" or similar errors
                    if "duplicate column name" not in str(e).lower():
                        logger.warning(f"Migration statement failed (may be OK): {stmt} - {e}")

            logger.info("Applied V2 migration: forgetting curve support")

        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (to_version, datetime.now(timezone.utc).isoformat())
        )
        conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get database connection with context management."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0
            )
            self._connection.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")

        try:
            yield self._connection
        except Exception as e:
            self._connection.rollback()
            raise

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        """Convert a database row to MemoryEntry."""
        # Safely get forgetting curve fields (may not exist in older schemas)
        row_keys = row.keys() if hasattr(row, 'keys') else []
        strength = row["strength"] if "strength" in row_keys else 1.0
        decay_rate = row["decay_rate"] if "decay_rate" in row_keys else 0.15
        last_reinforced = None
        if "last_reinforced" in row_keys and row["last_reinforced"]:
            last_reinforced = datetime.fromisoformat(row["last_reinforced"])
        reinforcement_count = row["reinforcement_count"] if "reinforcement_count" in row_keys else 0

        return MemoryEntry(
            id=row["id"],
            content=row["content"],
            tier=MemoryTier(row["tier"]) if row["tier"] else self.tier,
            priority=MemoryPriority(row["priority"]) if row["priority"] else MemoryPriority.NORMAL,
            namespace=MemoryNamespace(row["namespace"]) if row["namespace"] else None,
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(timezone.utc),
            last_accessed=datetime.fromisoformat(row["last_accessed"]) if row["last_accessed"] else None,
            access_count=row["access_count"] or 0,
            tags=json.loads(row["tags"]) if row["tags"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            embedding=list(row["embedding"]) if row["embedding"] else None,
            # V40: Forgetting curve fields
            strength=strength if strength is not None else 1.0,
            decay_rate=decay_rate if decay_rate is not None else 0.15,
            last_reinforced=last_reinforced,
            reinforcement_count=reinforcement_count if reinforcement_count is not None else 0,
        )

    def _content_hash(self, content: str) -> str:
        """Generate content hash for deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def get(self, key: str, reinforce: bool = True) -> Optional[MemoryEntry]:
        """Get memory entry by ID.

        Args:
            key: Memory ID
            reinforce: Whether to reinforce memory strength on access

        Returns:
            MemoryEntry if found, None otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM memories WHERE id = ?",
                (key,)
            )
            row = cursor.fetchone()
            if row:
                entry = self._row_to_entry(row)
                now = datetime.now(timezone.utc).isoformat()

                if reinforce:
                    # Reinforce memory on access
                    new_strength = entry.reinforce(access_type="recall")
                    conn.execute(
                        """UPDATE memories SET
                            access_count = access_count + 1,
                            last_accessed = ?,
                            strength = ?,
                            last_reinforced = ?,
                            reinforcement_count = reinforcement_count + 1
                        WHERE id = ?""",
                        (now, new_strength, now, key)
                    )
                else:
                    # Just update access tracking
                    conn.execute(
                        "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                        (now, key)
                    )
                conn.commit()
                return entry
        return None

    async def put(self, key: str, value: MemoryEntry) -> None:
        """Store memory entry."""
        now = datetime.now(timezone.utc).isoformat()
        content_hash = self._content_hash(value.content)

        # Generate embedding if provider available
        embedding_blob = None
        if self.embedding_provider and not value.embedding:
            try:
                embedding = self.embedding_provider(value.content)
                if embedding:
                    value.embedding = embedding
                    embedding_blob = json.dumps(embedding).encode()
            except Exception as e:
                logger.warning("Failed to generate embedding: %s", e)
        elif value.embedding:
            embedding_blob = json.dumps(value.embedding).encode()

        with self._get_connection() as conn:
            # Use INSERT OR REPLACE for upsert
            conn.execute("""
                INSERT OR REPLACE INTO memories (
                    id, content, content_hash, memory_type, tier, namespace, priority,
                    importance, created_at, updated_at, last_accessed, valid_from, valid_to,
                    access_count, session_id, tags, metadata, embedding,
                    strength, decay_rate, last_reinforced, reinforcement_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                key or value.id,
                value.content,
                content_hash,
                value.content_type,
                value.tier.value if value.tier else self.tier.value,
                value.namespace.value if value.namespace else None,
                value.priority.value if value.priority else MemoryPriority.NORMAL.value,
                value.metadata.get("importance", 0.5) if value.metadata else 0.5,
                value.created_at.isoformat() if value.created_at else now,
                now,
                value.last_accessed.isoformat() if value.last_accessed else None,
                now,
                None,  # valid_to
                value.access_count,
                value.metadata.get("session_id") if value.metadata else None,
                json.dumps(value.tags) if value.tags else None,
                json.dumps(value.metadata) if value.metadata else None,
                embedding_blob,
                # V40: Forgetting curve fields
                value.strength if hasattr(value, 'strength') else 1.0,
                value.decay_rate if hasattr(value, 'decay_rate') else 0.15,
                value.last_reinforced.isoformat() if hasattr(value, 'last_reinforced') and value.last_reinforced else None,
                value.reinforcement_count if hasattr(value, 'reinforcement_count') else 0,
            ))
            conn.commit()

    async def delete(self, key: str) -> bool:
        """Delete memory entry by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (key,))
            conn.commit()
            return cursor.rowcount > 0

    async def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Search memories using full-text search and optional semantic search."""
        results: List[Tuple[MemoryEntry, float]] = []

        with self._get_connection() as conn:
            # Full-text search
            try:
                cursor = conn.execute("""
                    SELECT m.*, bm25(memories_fts) as score
                    FROM memories m
                    JOIN memories_fts fts ON m.id = fts.id
                    WHERE memories_fts MATCH ?
                    ORDER BY score
                    LIMIT ?
                """, (query, limit * 2))  # Get more for potential dedup

                for row in cursor.fetchall():
                    entry = self._row_to_entry(row)
                    score = -row["score"]  # BM25 scores are negative, lower is better
                    results.append((entry, score))
            except sqlite3.OperationalError:
                # FTS may not have data yet, fall back to LIKE
                cursor = conn.execute("""
                    SELECT * FROM memories
                    WHERE content LIKE ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (f"%{query}%", limit))

                for row in cursor.fetchall():
                    entry = self._row_to_entry(row)
                    results.append((entry, 0.5))

        # If we have embeddings and provider, do semantic search too
        if self.embedding_provider:
            try:
                query_embedding = self.embedding_provider(query)
                if query_embedding:
                    semantic_results = await self._semantic_search(query_embedding, limit)
                    # Merge with FTS results
                    existing_ids = {e.id for e, _ in results}
                    for entry, score in semantic_results:
                        if entry.id not in existing_ids:
                            results.append((entry, score * 0.8))  # Slight penalty for semantic-only
            except Exception as e:
                logger.debug("Semantic search failed: %s", e)

        # Sort by score and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in results[:limit]]

    async def _semantic_search(
        self,
        query_embedding: List[float],
        limit: int = 10
    ) -> List[Tuple[MemoryEntry, float]]:
        """Search using cosine similarity with embeddings."""
        results = []

        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM memories WHERE embedding IS NOT NULL"
            )

            for row in cursor.fetchall():
                stored_embedding = json.loads(row["embedding"]) if row["embedding"] else None
                if stored_embedding:
                    similarity = self._cosine_similarity(query_embedding, stored_embedding)
                    if similarity > 0.5:  # Threshold
                        entry = self._row_to_entry(row)
                        results.append((entry, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    async def list_all(self) -> List[MemoryEntry]:
        """List all entries."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM memories ORDER BY created_at DESC"
            )
            return [self._row_to_entry(row) for row in cursor.fetchall()]

    async def count(self) -> int:
        """Get entry count."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM memories")
            return cursor.fetchone()[0]

    # =========================================================================
    # Cross-Session Memory API
    # =========================================================================

    async def store_memory(
        self,
        content: str,
        memory_type: str = "context",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        namespace: Optional[MemoryNamespace] = None
    ) -> str:
        """Store a memory with full metadata.

        Args:
            content: The memory content
            memory_type: Type: fact, decision, learning, context, task
            importance: 0.0 to 1.0 importance score
            tags: Optional tags for categorization
            session_id: Optional session ID for tracking
            namespace: Memory namespace for TTL management

        Returns:
            Memory ID
        """
        memory_id = generate_memory_id(content, f"{memory_type}_")

        entry = MemoryEntry(
            id=memory_id,
            content=content,
            tier=self.tier,
            priority=MemoryPriority.HIGH if importance > 0.7 else MemoryPriority.NORMAL,
            content_type=memory_type,
            namespace=namespace,
            tags=tags or [],
            metadata={
                "importance": importance,
                "session_id": session_id,
                "memory_type": memory_type
            }
        )

        await self.put(memory_id, entry)
        return memory_id

    async def get_by_type(self, memory_type: str, limit: int = 20) -> List[MemoryEntry]:
        """Get memories by type."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM memories WHERE memory_type = ? ORDER BY created_at DESC LIMIT ?",
                (memory_type, limit)
            )
            return [self._row_to_entry(row) for row in cursor.fetchall()]

    async def get_decisions(self, limit: int = 20) -> List[MemoryEntry]:
        """Get recent decisions."""
        return await self.get_by_type("decision", limit)

    async def get_learnings(self, limit: int = 20) -> List[MemoryEntry]:
        """Get recent learnings."""
        return await self.get_by_type("learning", limit)

    async def get_facts(self, limit: int = 20) -> List[MemoryEntry]:
        """Get recent facts."""
        return await self.get_by_type("fact", limit)

    async def get_session_context(self, max_tokens: int = 4000) -> str:
        """Generate context summary for session handoff.

        Returns a formatted string with:
        - Recent sessions
        - Key decisions
        - Important learnings
        - Critical facts
        """
        parts = []

        # Recent sessions
        sessions = await self.get_recent_sessions(5)
        if sessions:
            parts.append("## Recent Sessions")
            for s in sessions:
                parts.append(f"- **{s.get('started_at', '')[:10]}**: {s.get('task_summary', 'No summary')}")

        # Key decisions
        decisions = await self.get_decisions(10)
        if decisions:
            parts.append("\n## Key Decisions")
            for d in decisions:
                parts.append(f"- [{d.created_at.isoformat()[:10]}] {d.content[:200]}")

        # Learnings
        learnings = await self.get_learnings(10)
        if learnings:
            parts.append("\n## Learnings")
            for l in learnings:
                parts.append(f"- {l.content[:200]}")

        # Important facts
        facts = await self.get_high_importance(10)
        if facts:
            parts.append("\n## Important Facts")
            for f in facts:
                parts.append(f"- {f.content[:200]}")

        context = "\n".join(parts)

        # Truncate if needed (rough: 4 chars per token)
        max_chars = max_tokens * 4
        if len(context) > max_chars:
            context = context[:max_chars] + "\n\n[Context truncated...]"

        return context

    async def get_high_importance(self, limit: int = 20) -> List[MemoryEntry]:
        """Get high-importance memories."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM memories WHERE importance >= 0.7 AND valid_to IS NULL ORDER BY importance DESC, created_at DESC LIMIT ?",
                (limit,)
            )
            return [self._row_to_entry(row) for row in cursor.fetchall()]

    # =========================================================================
    # Session Management
    # =========================================================================

    async def start_session(self, task_summary: str = "", project_path: str = "") -> str:
        """Start a new session.

        Returns:
            Session ID
        """
        session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:12]
        now = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO sessions (id, started_at, task_summary, project_path, memories_created)
                VALUES (?, ?, ?, ?, 0)
            """, (session_id, now, task_summary, project_path))
            conn.commit()

        logger.info(f"Started session: {session_id}")
        return session_id

    async def end_session(self, session_id: str, summary: Optional[str] = None) -> None:
        """End a session."""
        now = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            if summary:
                conn.execute(
                    "UPDATE sessions SET ended_at = ?, task_summary = ? WHERE id = ?",
                    (now, summary, session_id)
                )
            else:
                conn.execute(
                    "UPDATE sessions SET ended_at = ? WHERE id = ?",
                    (now, session_id)
                )
            conn.commit()

        logger.info(f"Ended session: {session_id}")

    async def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent sessions."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?",
                (limit,)
            )
            sessions = []
            for row in cursor.fetchall():
                sessions.append({
                    "id": row["id"],
                    "started_at": row["started_at"],
                    "ended_at": row["ended_at"],
                    "task_summary": row["task_summary"],
                    "project_path": row["project_path"],
                    "memories_created": row["memories_created"],
                    "decisions": json.loads(row["decisions"]) if row["decisions"] else [],
                    "learnings": json.loads(row["learnings"]) if row["learnings"] else [],
                })
            return sessions

    # =========================================================================
    # Export/Import
    # =========================================================================

    async def export_to_json(self, output_path: Optional[Path] = None) -> Path:
        """Export all memories to JSON for portability.

        Returns:
            Path to the exported file
        """
        if output_path is None:
            exports_dir = self.db_path.parent / "exports"
            exports_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = exports_dir / f"memory_export_{timestamp}.json"

        memories = await self.list_all()
        sessions = await self.get_recent_sessions(100)

        export_data = {
            "version": SCHEMA_VERSION,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "memories": [
                {
                    "id": m.id,
                    "content": m.content,
                    "tier": m.tier.value if m.tier else None,
                    "priority": m.priority.value if m.priority else None,
                    "namespace": m.namespace.value if m.namespace else None,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                    "access_count": m.access_count,
                    "tags": m.tags,
                    "metadata": m.metadata,
                }
                for m in memories
            ],
            "sessions": sessions,
            "stats": {
                "total_memories": len(memories),
                "total_sessions": len(sessions),
            }
        }

        output_path.write_text(json.dumps(export_data, indent=2), encoding="utf-8")
        logger.info(f"Exported {len(memories)} memories to {output_path}")
        return output_path

    async def import_from_json(self, input_path: Path) -> int:
        """Import memories from JSON export.

        Returns:
            Number of memories imported
        """
        data = json.loads(input_path.read_text(encoding="utf-8"))
        imported = 0

        for memory_data in data.get("memories", []):
            entry = MemoryEntry(
                id=memory_data["id"],
                content=memory_data["content"],
                tier=MemoryTier(memory_data["tier"]) if memory_data.get("tier") else self.tier,
                priority=MemoryPriority(memory_data["priority"]) if memory_data.get("priority") else MemoryPriority.NORMAL,
                namespace=MemoryNamespace(memory_data["namespace"]) if memory_data.get("namespace") else None,
                tags=memory_data.get("tags", []),
                metadata=memory_data.get("metadata", {}),
            )

            try:
                await self.put(entry.id, entry)
                imported += 1
            except Exception as e:
                logger.warning(f"Failed to import memory {entry.id}: {e}")

        logger.info(f"Imported {imported} memories from {input_path}")
        return imported

    # =========================================================================
    # V40: Forgetting Curve Support
    # =========================================================================

    async def get_weak_memories(
        self,
        threshold: float = 0.1,
        limit: int = 100
    ) -> List[MemoryEntry]:
        """Get memories with strength below threshold.

        Args:
            threshold: Strength threshold (0.0-1.0)
            limit: Maximum entries to return

        Returns:
            List of weak memories sorted by strength ascending
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM memories WHERE strength < ? ORDER BY strength ASC LIMIT ?",
                (threshold, limit)
            )
            return [self._row_to_entry(row) for row in cursor.fetchall()]

    async def get_memories_for_review(
        self,
        limit: int = 20
    ) -> List[MemoryEntry]:
        """Get memories that would benefit from review (optimal spaced repetition timing).

        Returns memories where enough time has passed since last reinforcement
        based on their reinforcement count.

        Args:
            limit: Maximum entries to return

        Returns:
            List of memories due for review
        """
        now = datetime.now(timezone.utc)
        results: List[MemoryEntry] = []

        # Get all memories with reinforcement data
        entries = await self.list_all()

        for entry in entries:
            if entry.last_reinforced is None:
                # Never reinforced, due for review
                results.append(entry)
                continue

            # Calculate expected interval based on reinforcement count
            optimal_intervals = [1, 3, 7, 14, 30, 60]
            idx = min(entry.reinforcement_count, len(optimal_intervals) - 1)
            expected_days = optimal_intervals[idx]

            # Check if time has elapsed
            ref_time = entry.last_reinforced
            if ref_time.tzinfo is None:
                ref_time = ref_time.replace(tzinfo=timezone.utc)

            days_elapsed = (now - ref_time).total_seconds() / 86400

            # Due for review if we're past 80% of expected interval
            if days_elapsed >= expected_days * 0.8:
                results.append(entry)

        # Sort by days overdue and return top limit
        def sort_key(e):
            if e.last_reinforced is None:
                return float('inf')
            ref = e.last_reinforced
            if ref.tzinfo is None:
                ref = ref.replace(tzinfo=timezone.utc)
            return (now - ref).total_seconds()

        results.sort(key=sort_key, reverse=True)
        return results[:limit]

    async def archive_weak_memories(
        self,
        threshold: float = 0.1
    ) -> Tuple[int, int]:
        """Archive memories below strength threshold.

        Moves weak memories to archived state instead of deleting them.

        Args:
            threshold: Strength threshold below which to archive

        Returns:
            Tuple of (archived_count, total_weak_count)
        """
        weak_memories = await self.get_weak_memories(threshold, limit=1000)
        archived = 0
        now = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            for entry in weak_memories:
                try:
                    # Mark as archived in metadata
                    metadata = entry.metadata or {}
                    metadata['archived_at'] = now
                    metadata['archived_strength'] = entry.calculate_current_strength()

                    # Set valid_to to mark as archived (soft delete)
                    conn.execute(
                        "UPDATE memories SET valid_to = ?, metadata = ? WHERE id = ?",
                        (now, json.dumps(metadata), entry.id)
                    )
                    archived += 1
                except Exception as e:
                    logger.warning(f"Failed to archive memory {entry.id}: {e}")

            conn.commit()

        logger.info(f"Archived {archived}/{len(weak_memories)} weak memories")
        return archived, len(weak_memories)

    async def delete_archived_memories(
        self,
        days_old: int = 30
    ) -> int:
        """Permanently delete archived memories older than specified days.

        Args:
            days_old: Days since archival after which to delete

        Returns:
            Number of memories deleted
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_old)
        cutoff_str = cutoff.isoformat()

        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE valid_to IS NOT NULL AND valid_to < ?",
                (cutoff_str,)
            )
            conn.commit()
            deleted = cursor.rowcount

        logger.info(f"Deleted {deleted} archived memories older than {days_old} days")
        return deleted

    async def update_all_strengths(self) -> Dict[str, Any]:
        """Recalculate current strength for all memories.

        Returns:
            Statistics about the update
        """
        entries = await self.list_all()
        total = len(entries)
        weak_count = 0
        very_weak_count = 0

        with self._get_connection() as conn:
            for entry in entries:
                current_strength = entry.calculate_current_strength()

                if current_strength < 0.1:
                    weak_count += 1
                if current_strength < 0.01:
                    very_weak_count += 1

                # Note: We don't update stored strength here as it represents
                # the strength at last reinforcement, not decayed strength.
                # The decay is calculated dynamically via calculate_current_strength()

            conn.commit()

        return {
            "total_memories": total,
            "weak_memories": weak_count,
            "very_weak_memories": very_weak_count,
            "weak_threshold": 0.1,
            "very_weak_threshold": 0.01,
        }

    # =========================================================================
    # Statistics
    # =========================================================================

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            by_type = dict(conn.execute(
                "SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type"
            ).fetchall())
            by_namespace = dict(conn.execute(
                "SELECT namespace, COUNT(*) FROM memories WHERE namespace IS NOT NULL GROUP BY namespace"
            ).fetchall())
            sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

            # V40: Add strength statistics
            strength_stats = conn.execute("""
                SELECT
                    AVG(strength) as avg_strength,
                    MIN(strength) as min_strength,
                    MAX(strength) as max_strength,
                    SUM(CASE WHEN strength < 0.1 THEN 1 ELSE 0 END) as weak_count,
                    SUM(CASE WHEN strength < 0.01 THEN 1 ELSE 0 END) as very_weak_count,
                    AVG(reinforcement_count) as avg_reinforcements
                FROM memories WHERE valid_to IS NULL
            """).fetchone()

            return {
                "total_memories": total,
                "memories_by_type": by_type,
                "memories_by_namespace": by_namespace,
                "total_sessions": sessions,
                "storage_path": str(self.db_path),
                "db_size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
                # V40: Forgetting curve stats
                "strength_stats": {
                    "average": strength_stats[0] if strength_stats else None,
                    "minimum": strength_stats[1] if strength_stats else None,
                    "maximum": strength_stats[2] if strength_stats else None,
                    "weak_count": strength_stats[3] if strength_stats else 0,
                    "very_weak_count": strength_stats[4] if strength_stats else 0,
                    "avg_reinforcements": strength_stats[5] if strength_stats else 0,
                },
            }

    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None


# =============================================================================
# Singleton Instance
# =============================================================================

_sqlite_backend: Optional[SQLiteTierBackend] = None


def get_sqlite_backend(
    embedding_provider: Optional[Callable[[str], List[float]]] = None
) -> SQLiteTierBackend:
    """Get or create the singleton SQLite backend."""
    global _sqlite_backend
    if _sqlite_backend is None:
        _sqlite_backend = SQLiteTierBackend(
            tier=MemoryTier.ARCHIVAL_MEMORY,
            embedding_provider=embedding_provider
        )
    return _sqlite_backend


__all__ = [
    "SQLiteTierBackend",
    "get_sqlite_backend",
    "SCHEMA_VERSION",
]
