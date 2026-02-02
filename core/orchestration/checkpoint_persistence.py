#!/usr/bin/env python3
"""
Checkpoint Persistence Layer for Orchestration (V33.11).

Provides SQLite-based persistent storage for:
- Orchestration run history and results
- Workflow execution metrics
- Ralph iteration state
- LangGraph checkpoint integration
- Session continuity data

This module enables crash recovery, session resume, and historical analysis
of orchestration activities.

Example:
    >>> from core.orchestration.checkpoint_persistence import (
    ...     create_checkpoint_store,
    ...     CheckpointConfig,
    ... )
    >>>
    >>> store = await create_checkpoint_store()
    >>> await store.save_run(run_id="run_123", task="my_task", result={...})
    >>> runs = await store.list_runs(limit=10)
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import structlog

# =============================================================================
# Logging Setup
# =============================================================================

log = structlog.get_logger(__name__)

# =============================================================================
# Availability Flags
# =============================================================================

# Check for aiosqlite availability
try:
    import aiosqlite
    AIOSQLITE_AVAILABLE = True
except ImportError:
    AIOSQLITE_AVAILABLE = False
    aiosqlite = None  # type: ignore[assignment]

# Check for LangGraph checkpoint availability
try:
    from langgraph.checkpoint.base import BaseCheckpointSaver as _BaseCheckpointSaver
    LANGGRAPH_CHECKPOINT_AVAILABLE = True
    del _BaseCheckpointSaver  # Only checking availability
except ImportError:
    LANGGRAPH_CHECKPOINT_AVAILABLE = False

# Combined availability flag
CHECKPOINT_PERSISTENCE_AVAILABLE = AIOSQLITE_AVAILABLE


# =============================================================================
# Enums and Types
# =============================================================================

class RunStatus(str, Enum):
    """Status of an orchestration run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CheckpointType(str, Enum):
    """Type of checkpoint data."""

    ORCHESTRATION_RUN = "orchestration_run"
    RALPH_ITERATION = "ralph_iteration"
    WORKFLOW_STATE = "workflow_state"
    AGENT_STATE = "agent_state"
    METRICS = "metrics"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CheckpointConfig:
    """Configuration for checkpoint persistence."""

    # Database path (default: ~/.unleash/checkpoints.db)
    db_path: str = field(
        default_factory=lambda: str(
            Path.home() / ".unleash" / "checkpoints.db"
        )
    )

    # Whether to enable WAL mode for better concurrency
    wal_mode: bool = True

    # Auto-vacuum to keep database compact
    auto_vacuum: bool = True

    # Maximum number of runs to retain (0 = unlimited)
    max_runs_retained: int = 1000

    # Maximum age of runs in days (0 = unlimited)
    max_run_age_days: int = 30

    # Enable automatic cleanup on startup
    auto_cleanup: bool = True

    # Connection timeout in seconds
    timeout: float = 30.0


@dataclass
class OrchestrationRun:
    """Represents an orchestration run record."""

    run_id: str
    task: str
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None

    # Run details
    agents: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    # Metrics
    duration_ms: Optional[float] = None
    token_usage: Optional[Dict[str, int]] = None
    cost_usd: Optional[float] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Checkpoint reference
    checkpoint_id: Optional[str] = None
    parent_run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "run_id": self.run_id,
            "task": self.task,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "agents": self.agents,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "token_usage": self.token_usage,
            "cost_usd": self.cost_usd,
            "metadata": self.metadata,
            "tags": self.tags,
            "checkpoint_id": self.checkpoint_id,
            "parent_run_id": self.parent_run_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrchestrationRun":
        """Create from dictionary."""
        return cls(
            run_id=data["run_id"],
            task=data["task"],
            status=RunStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data.get("completed_at")
                else None
            ),
            agents=data.get("agents", []),
            result=data.get("result"),
            error=data.get("error"),
            duration_ms=data.get("duration_ms"),
            token_usage=data.get("token_usage"),
            cost_usd=data.get("cost_usd"),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            checkpoint_id=data.get("checkpoint_id"),
            parent_run_id=data.get("parent_run_id"),
        )


@dataclass
class RalphIterationState:
    """State of a Ralph Loop iteration."""

    iteration_id: str
    iteration_number: int
    status: RunStatus
    created_at: datetime

    # Iteration details
    task: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    # Metrics
    duration_ms: Optional[float] = None
    improvements: List[str] = field(default_factory=list)

    # Session reference
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "iteration_id": self.iteration_id,
            "iteration_number": self.iteration_number,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "task": self.task,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "improvements": self.improvements,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RalphIterationState":
        """Create from dictionary."""
        return cls(
            iteration_id=data["iteration_id"],
            iteration_number=data["iteration_number"],
            status=RunStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            task=data.get("task"),
            result=data.get("result"),
            error=data.get("error"),
            duration_ms=data.get("duration_ms"),
            improvements=data.get("improvements", []),
            session_id=data.get("session_id"),
        )


@dataclass
class ExecutionMetrics:
    """Aggregated execution metrics."""

    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0

    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0

    total_tokens: int = 0
    total_cost_usd: float = 0.0

    # Time range
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None


# =============================================================================
# SQL Schema
# =============================================================================

_SCHEMA_SQL = """
-- Enable WAL mode for better concurrency
PRAGMA journal_mode=WAL;

-- Orchestration runs table
CREATE TABLE IF NOT EXISTS orchestration_runs (
    run_id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    agents TEXT,  -- JSON array
    result TEXT,  -- JSON object
    error TEXT,
    duration_ms REAL,
    token_usage TEXT,  -- JSON object
    cost_usd REAL,
    metadata TEXT,  -- JSON object
    tags TEXT,  -- JSON array
    checkpoint_id TEXT,
    parent_run_id TEXT,
    FOREIGN KEY (parent_run_id) REFERENCES orchestration_runs(run_id)
);

-- Ralph iterations table
CREATE TABLE IF NOT EXISTS ralph_iterations (
    iteration_id TEXT PRIMARY KEY,
    iteration_number INTEGER NOT NULL,
    session_id TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    task TEXT,
    result TEXT,  -- JSON object
    error TEXT,
    duration_ms REAL,
    improvements TEXT  -- JSON array
);

-- Checkpoints table (for generic state storage)
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    checkpoint_type TEXT NOT NULL,
    thread_id TEXT,
    namespace TEXT DEFAULT '',
    parent_id TEXT,
    created_at TEXT NOT NULL,
    state TEXT NOT NULL,  -- JSON object
    metadata TEXT,  -- JSON object
    FOREIGN KEY (parent_id) REFERENCES checkpoints(checkpoint_id)
);

-- Execution metrics table (aggregated)
CREATE TABLE IF NOT EXISTS execution_metrics (
    metric_id TEXT PRIMARY KEY,
    period TEXT NOT NULL,  -- 'hourly', 'daily', 'weekly'
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    total_runs INTEGER DEFAULT 0,
    successful_runs INTEGER DEFAULT 0,
    failed_runs INTEGER DEFAULT 0,
    total_duration_ms REAL DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    total_cost_usd REAL DEFAULT 0,
    created_at TEXT NOT NULL
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_runs_status ON orchestration_runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_created ON orchestration_runs(created_at);
CREATE INDEX IF NOT EXISTS idx_runs_task ON orchestration_runs(task);
CREATE INDEX IF NOT EXISTS idx_ralph_session ON ralph_iterations(session_id);
CREATE INDEX IF NOT EXISTS idx_ralph_number ON ralph_iterations(iteration_number);
CREATE INDEX IF NOT EXISTS idx_checkpoints_type ON checkpoints(checkpoint_type);
CREATE INDEX IF NOT EXISTS idx_checkpoints_thread ON checkpoints(thread_id);
CREATE INDEX IF NOT EXISTS idx_metrics_period ON execution_metrics(period, period_start);
"""


# =============================================================================
# Checkpoint Store Implementation
# =============================================================================

class CheckpointStore:
    """
    SQLite-based checkpoint persistence store.

    Provides persistent storage for orchestration runs, Ralph iterations,
    generic checkpoints, and execution metrics.

    Example:
        >>> store = CheckpointStore()
        >>> await store.initialize()
        >>>
        >>> # Save a run
        >>> run = await store.create_run("my_task", agents=["ralph"])
        >>> await store.complete_run(run.run_id, result={"success": True})
        >>>
        >>> # Query runs
        >>> recent = await store.list_runs(limit=10)
    """

    def __init__(self, config: Optional[CheckpointConfig] = None) -> None:
        """Initialize checkpoint store."""
        self.config = config or CheckpointConfig()
        self._conn: Optional[Any] = None  # aiosqlite.Connection
        self._lock = asyncio.Lock()
        self._is_setup = False
        self._initialized = False

        log.debug(
            "checkpoint_store_created",
            db_path=self.config.db_path,
        )

    @property
    def is_initialized(self) -> bool:
        """Check if store is initialized."""
        return self._initialized

    async def initialize(self) -> bool:
        """
        Initialize the checkpoint store.

        Creates database file, tables, and indexes.

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        if not AIOSQLITE_AVAILABLE:
            log.error("aiosqlite_not_available")
            raise RuntimeError(
                "aiosqlite is required for checkpoint persistence. "
                "Install with: pip install aiosqlite"
            )

        try:
            # Ensure directory exists
            db_dir = Path(self.config.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)

            # Connect to database (aiosqlite availability checked above)
            assert aiosqlite is not None  # Type narrowing
            self._conn = await aiosqlite.connect(
                self.config.db_path,
                timeout=self.config.timeout,
            )

            # Setup schema
            await self._setup_schema()

            # Run auto cleanup if enabled
            if self.config.auto_cleanup:
                await self._cleanup_old_runs()

            self._initialized = True
            log.info(
                "checkpoint_store_initialized",
                db_path=self.config.db_path,
            )
            return True

        except Exception as e:
            log.error("checkpoint_store_init_failed", error=str(e))
            raise

    async def close(self) -> None:
        """Close database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            self._initialized = False
            log.debug("checkpoint_store_closed")

    async def _setup_schema(self) -> None:
        """Create database schema."""
        if self._is_setup or not self._conn:
            return

        async with self._lock:
            await self._conn.executescript(_SCHEMA_SQL)
            await self._conn.commit()
            self._is_setup = True

    async def _cleanup_old_runs(self) -> int:
        """
        Clean up old runs based on retention settings.

        Returns:
            Number of runs deleted
        """
        if not self._conn:
            return 0

        deleted = 0

        async with self._lock:
            # Delete by count limit
            if self.config.max_runs_retained > 0:
                cursor = await self._conn.execute(
                    """
                    DELETE FROM orchestration_runs
                    WHERE run_id NOT IN (
                        SELECT run_id FROM orchestration_runs
                        ORDER BY created_at DESC
                        LIMIT ?
                    )
                    """,
                    (self.config.max_runs_retained,),
                )
                deleted += cursor.rowcount

            # Delete by age
            if self.config.max_run_age_days > 0:
                cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.max_run_age_days)
                cursor = await self._conn.execute(
                    """
                    DELETE FROM orchestration_runs
                    WHERE created_at < ?
                    """,
                    (cutoff.isoformat(),),
                )
                deleted += cursor.rowcount

            await self._conn.commit()

        if deleted > 0:
            log.info("cleanup_old_runs", deleted=deleted)

        return deleted

    # =========================================================================
    # Orchestration Runs
    # =========================================================================

    async def create_run(
        self,
        task: str,
        *,
        run_id: Optional[str] = None,
        agents: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        parent_run_id: Optional[str] = None,
    ) -> OrchestrationRun:
        """
        Create a new orchestration run record.

        Args:
            task: Task description
            run_id: Optional custom run ID
            agents: List of agents involved
            metadata: Additional metadata
            tags: Tags for categorization
            parent_run_id: Parent run for nested runs

        Returns:
            Created OrchestrationRun
        """
        if not self._conn:
            raise RuntimeError("Store not initialized")

        now = datetime.now(timezone.utc)
        run = OrchestrationRun(
            run_id=run_id or f"run_{uuid.uuid4().hex[:12]}",
            task=task,
            status=RunStatus.PENDING,
            created_at=now,
            updated_at=now,
            agents=agents or [],
            metadata=metadata or {},
            tags=tags or [],
            parent_run_id=parent_run_id,
        )

        async with self._lock:
            await self._conn.execute(
                """
                INSERT INTO orchestration_runs (
                    run_id, task, status, created_at, updated_at,
                    agents, metadata, tags, parent_run_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.task,
                    run.status.value,
                    run.created_at.isoformat(),
                    run.updated_at.isoformat(),
                    json.dumps(run.agents),
                    json.dumps(run.metadata),
                    json.dumps(run.tags),
                    run.parent_run_id,
                ),
            )
            await self._conn.commit()

        log.debug("run_created", run_id=run.run_id, task=task)
        return run

    async def update_run_status(
        self,
        run_id: str,
        status: RunStatus,
    ) -> None:
        """Update run status."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        now = datetime.now(timezone.utc)

        async with self._lock:
            await self._conn.execute(
                """
                UPDATE orchestration_runs
                SET status = ?, updated_at = ?
                WHERE run_id = ?
                """,
                (status.value, now.isoformat(), run_id),
            )
            await self._conn.commit()

        log.debug("run_status_updated", run_id=run_id, status=status.value)

    async def complete_run(
        self,
        run_id: str,
        *,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        duration_ms: Optional[float] = None,
        token_usage: Optional[Dict[str, int]] = None,
        cost_usd: Optional[float] = None,
    ) -> None:
        """
        Mark a run as completed (success or failure).

        Args:
            run_id: Run ID to complete
            result: Result data (for success)
            error: Error message (for failure)
            duration_ms: Total duration in milliseconds
            token_usage: Token usage breakdown
            cost_usd: Estimated cost in USD
        """
        if not self._conn:
            raise RuntimeError("Store not initialized")

        now = datetime.now(timezone.utc)
        status = RunStatus.FAILED if error else RunStatus.COMPLETED

        async with self._lock:
            await self._conn.execute(
                """
                UPDATE orchestration_runs
                SET status = ?, updated_at = ?, completed_at = ?,
                    result = ?, error = ?, duration_ms = ?,
                    token_usage = ?, cost_usd = ?
                WHERE run_id = ?
                """,
                (
                    status.value,
                    now.isoformat(),
                    now.isoformat(),
                    json.dumps(result) if result else None,
                    error,
                    duration_ms,
                    json.dumps(token_usage) if token_usage else None,
                    cost_usd,
                    run_id,
                ),
            )
            await self._conn.commit()

        log.debug(
            "run_completed",
            run_id=run_id,
            status=status.value,
            duration_ms=duration_ms,
        )

    async def get_run(self, run_id: str) -> Optional[OrchestrationRun]:
        """Get a run by ID."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        async with self._lock:
            cursor = await self._conn.execute(
                """
                SELECT run_id, task, status, created_at, updated_at, completed_at,
                       agents, result, error, duration_ms, token_usage, cost_usd,
                       metadata, tags, checkpoint_id, parent_run_id
                FROM orchestration_runs
                WHERE run_id = ?
                """,
                (run_id,),
            )
            row = await cursor.fetchone()

        if not row:
            return None

        return self._row_to_run(row)

    async def list_runs(
        self,
        *,
        status: Optional[RunStatus] = None,
        task_contains: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[OrchestrationRun]:
        """
        List orchestration runs with filtering.

        Args:
            status: Filter by status
            task_contains: Filter by task containing string
            tags: Filter by tags (any match)
            limit: Maximum results
            offset: Offset for pagination

        Returns:
            List of matching runs
        """
        if not self._conn:
            raise RuntimeError("Store not initialized")

        query = """
            SELECT run_id, task, status, created_at, updated_at, completed_at,
                   agents, result, error, duration_ms, token_usage, cost_usd,
                   metadata, tags, checkpoint_id, parent_run_id
            FROM orchestration_runs
            WHERE 1=1
        """
        params: List[Any] = []

        if status:
            query += " AND status = ?"
            params.append(status.value)

        if task_contains:
            query += " AND task LIKE ?"
            params.append(f"%{task_contains}%")

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        async with self._lock:
            cursor = await self._conn.execute(query, params)
            rows = await cursor.fetchall()

        runs = [self._row_to_run(row) for row in rows]

        # Filter by tags if specified (JSON array filtering)
        if tags:
            runs = [
                r for r in runs
                if any(t in r.tags for t in tags)
            ]

        return runs

    def _row_to_run(self, row: tuple) -> OrchestrationRun:
        """Convert database row to OrchestrationRun."""
        (
            run_id, task, status, created_at, updated_at, completed_at,
            agents, result, error, duration_ms, token_usage, cost_usd,
            metadata, tags, checkpoint_id, parent_run_id
        ) = row

        return OrchestrationRun(
            run_id=run_id,
            task=task,
            status=RunStatus(status),
            created_at=datetime.fromisoformat(created_at),
            updated_at=datetime.fromisoformat(updated_at),
            completed_at=(
                datetime.fromisoformat(completed_at)
                if completed_at else None
            ),
            agents=json.loads(agents) if agents else [],
            result=json.loads(result) if result else None,
            error=error,
            duration_ms=duration_ms,
            token_usage=json.loads(token_usage) if token_usage else None,
            cost_usd=cost_usd,
            metadata=json.loads(metadata) if metadata else {},
            tags=json.loads(tags) if tags else [],
            checkpoint_id=checkpoint_id,
            parent_run_id=parent_run_id,
        )

    # =========================================================================
    # Ralph Iterations
    # =========================================================================

    async def save_ralph_iteration(
        self,
        iteration_number: int,
        *,
        iteration_id: Optional[str] = None,
        session_id: Optional[str] = None,
        task: Optional[str] = None,
        status: RunStatus = RunStatus.RUNNING,
    ) -> RalphIterationState:
        """
        Save a Ralph iteration state.

        Args:
            iteration_number: Iteration number
            iteration_id: Optional custom ID
            session_id: Session ID for grouping
            task: Task being executed
            status: Current status

        Returns:
            Created RalphIterationState
        """
        if not self._conn:
            raise RuntimeError("Store not initialized")

        now = datetime.now(timezone.utc)
        iteration = RalphIterationState(
            iteration_id=iteration_id or f"ralph_{uuid.uuid4().hex[:8]}",
            iteration_number=iteration_number,
            status=status,
            created_at=now,
            task=task,
            session_id=session_id,
        )

        async with self._lock:
            await self._conn.execute(
                """
                INSERT OR REPLACE INTO ralph_iterations (
                    iteration_id, iteration_number, session_id, status,
                    created_at, task
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    iteration.iteration_id,
                    iteration.iteration_number,
                    iteration.session_id,
                    iteration.status.value,
                    iteration.created_at.isoformat(),
                    iteration.task,
                ),
            )
            await self._conn.commit()

        log.debug(
            "ralph_iteration_saved",
            iteration_id=iteration.iteration_id,
            number=iteration_number,
        )
        return iteration

    async def complete_ralph_iteration(
        self,
        iteration_id: str,
        *,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        duration_ms: Optional[float] = None,
        improvements: Optional[List[str]] = None,
    ) -> None:
        """Complete a Ralph iteration."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        status = RunStatus.FAILED if error else RunStatus.COMPLETED

        async with self._lock:
            await self._conn.execute(
                """
                UPDATE ralph_iterations
                SET status = ?, result = ?, error = ?,
                    duration_ms = ?, improvements = ?
                WHERE iteration_id = ?
                """,
                (
                    status.value,
                    json.dumps(result) if result else None,
                    error,
                    duration_ms,
                    json.dumps(improvements or []),
                    iteration_id,
                ),
            )
            await self._conn.commit()

        log.debug(
            "ralph_iteration_completed",
            iteration_id=iteration_id,
            status=status.value,
        )

    async def get_ralph_iterations(
        self,
        session_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[RalphIterationState]:
        """Get Ralph iterations, optionally filtered by session."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        if session_id:
            query = """
                SELECT iteration_id, iteration_number, session_id, status,
                       created_at, task, result, error, duration_ms, improvements
                FROM ralph_iterations
                WHERE session_id = ?
                ORDER BY iteration_number DESC
                LIMIT ?
            """
            params = (session_id, limit)
        else:
            query = """
                SELECT iteration_id, iteration_number, session_id, status,
                       created_at, task, result, error, duration_ms, improvements
                FROM ralph_iterations
                ORDER BY created_at DESC
                LIMIT ?
            """
            params = (limit,)

        async with self._lock:
            cursor = await self._conn.execute(query, params)
            rows = await cursor.fetchall()

        return [self._row_to_ralph_iteration(row) for row in rows]

    def _row_to_ralph_iteration(self, row: tuple) -> RalphIterationState:
        """Convert database row to RalphIterationState."""
        (
            iteration_id, iteration_number, session_id, status,
            created_at, task, result, error, duration_ms, improvements
        ) = row

        return RalphIterationState(
            iteration_id=iteration_id,
            iteration_number=iteration_number,
            status=RunStatus(status),
            created_at=datetime.fromisoformat(created_at),
            task=task,
            result=json.loads(result) if result else None,
            error=error,
            duration_ms=duration_ms,
            improvements=json.loads(improvements) if improvements else [],
            session_id=session_id,
        )

    # =========================================================================
    # Generic Checkpoints
    # =========================================================================

    async def save_checkpoint(
        self,
        checkpoint_type: CheckpointType,
        state: Dict[str, Any],
        *,
        checkpoint_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        namespace: str = "",
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a generic checkpoint.

        Args:
            checkpoint_type: Type of checkpoint
            state: State data to persist
            checkpoint_id: Optional custom ID
            thread_id: Thread/workflow ID
            namespace: Namespace for grouping
            parent_id: Parent checkpoint for chains
            metadata: Additional metadata

        Returns:
            Checkpoint ID
        """
        if not self._conn:
            raise RuntimeError("Store not initialized")

        checkpoint_id = checkpoint_id or f"cp_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        async with self._lock:
            await self._conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints (
                    checkpoint_id, checkpoint_type, thread_id, namespace,
                    parent_id, created_at, state, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    checkpoint_type.value,
                    thread_id,
                    namespace,
                    parent_id,
                    now.isoformat(),
                    json.dumps(state),
                    json.dumps(metadata or {}),
                ),
            )
            await self._conn.commit()

        log.debug(
            "checkpoint_saved",
            checkpoint_id=checkpoint_id,
            type=checkpoint_type.value,
        )
        return checkpoint_id

    async def get_checkpoint(
        self,
        checkpoint_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a checkpoint by ID."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        async with self._lock:
            cursor = await self._conn.execute(
                """
                SELECT checkpoint_id, checkpoint_type, thread_id, namespace,
                       parent_id, created_at, state, metadata
                FROM checkpoints
                WHERE checkpoint_id = ?
                """,
                (checkpoint_id,),
            )
            row = await cursor.fetchone()

        if not row:
            return None

        return {
            "checkpoint_id": row[0],
            "checkpoint_type": row[1],
            "thread_id": row[2],
            "namespace": row[3],
            "parent_id": row[4],
            "created_at": row[5],
            "state": json.loads(row[6]),
            "metadata": json.loads(row[7]) if row[7] else {},
        }

    async def get_latest_checkpoint(
        self,
        thread_id: str,
        checkpoint_type: Optional[CheckpointType] = None,
        namespace: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Get the latest checkpoint for a thread."""
        if not self._conn:
            raise RuntimeError("Store not initialized")

        if checkpoint_type:
            query = """
                SELECT checkpoint_id, checkpoint_type, thread_id, namespace,
                       parent_id, created_at, state, metadata
                FROM checkpoints
                WHERE thread_id = ? AND checkpoint_type = ? AND namespace = ?
                ORDER BY created_at DESC
                LIMIT 1
            """
            params = (thread_id, checkpoint_type.value, namespace)
        else:
            query = """
                SELECT checkpoint_id, checkpoint_type, thread_id, namespace,
                       parent_id, created_at, state, metadata
                FROM checkpoints
                WHERE thread_id = ? AND namespace = ?
                ORDER BY created_at DESC
                LIMIT 1
            """
            params = (thread_id, namespace)

        async with self._lock:
            cursor = await self._conn.execute(query, params)
            row = await cursor.fetchone()

        if not row:
            return None

        return {
            "checkpoint_id": row[0],
            "checkpoint_type": row[1],
            "thread_id": row[2],
            "namespace": row[3],
            "parent_id": row[4],
            "created_at": row[5],
            "state": json.loads(row[6]),
            "metadata": json.loads(row[7]) if row[7] else {},
        }

    # =========================================================================
    # Metrics
    # =========================================================================

    async def get_metrics(
        self,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> ExecutionMetrics:
        """
        Get aggregated execution metrics.

        Args:
            period_start: Start of period (default: all time)
            period_end: End of period (default: now)

        Returns:
            ExecutionMetrics with aggregated data
        """
        if not self._conn:
            raise RuntimeError("Store not initialized")

        query = """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as success,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(COALESCE(duration_ms, 0)) as total_duration,
                SUM(COALESCE(cost_usd, 0)) as total_cost
            FROM orchestration_runs
            WHERE 1=1
        """
        params: List[Any] = []

        if period_start:
            query += " AND created_at >= ?"
            params.append(period_start.isoformat())

        if period_end:
            query += " AND created_at <= ?"
            params.append(period_end.isoformat())

        async with self._lock:
            cursor = await self._conn.execute(query, params)
            row = await cursor.fetchone()

        total, success, failed, total_duration, total_cost = row

        return ExecutionMetrics(
            total_runs=total or 0,
            successful_runs=success or 0,
            failed_runs=failed or 0,
            total_duration_ms=total_duration or 0.0,
            avg_duration_ms=(
                (total_duration / total) if total and total_duration else 0.0
            ),
            total_cost_usd=total_cost or 0.0,
            period_start=period_start,
            period_end=period_end,
        )

    # =========================================================================
    # Context Manager
    # =========================================================================

    async def __aenter__(self) -> "CheckpointStore":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(
        self,
        _exc_type: Any,
        _exc_val: Any,
        _exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.close()


# =============================================================================
# Factory Functions
# =============================================================================

async def create_checkpoint_store(
    config: Optional[CheckpointConfig] = None,
) -> CheckpointStore:
    """
    Create and initialize a checkpoint store.

    Args:
        config: Optional configuration

    Returns:
        Initialized CheckpointStore

    Example:
        >>> store = await create_checkpoint_store()
        >>> run = await store.create_run("my_task")
    """
    store = CheckpointStore(config)
    await store.initialize()
    return store


def get_checkpoint_store_sync(
    config: Optional[CheckpointConfig] = None,
) -> CheckpointStore:
    """
    Get a checkpoint store synchronously (uninitialized).

    Note: Must call initialize() before use.

    Args:
        config: Optional configuration

    Returns:
        Uninitialized CheckpointStore
    """
    return CheckpointStore(config)


@asynccontextmanager
async def checkpoint_store_context(
    config: Optional[CheckpointConfig] = None,
) -> AsyncIterator[CheckpointStore]:
    """
    Context manager for checkpoint store.

    Example:
        >>> async with checkpoint_store_context() as store:
        ...     run = await store.create_run("my_task")
    """
    store = CheckpointStore(config)
    try:
        await store.initialize()
        yield store
    finally:
        await store.close()


# =============================================================================
# Global Singleton (Optional)
# =============================================================================

_global_store: Optional[CheckpointStore] = None
_global_lock = asyncio.Lock()


async def get_global_checkpoint_store() -> CheckpointStore:
    """
    Get or create the global checkpoint store singleton.

    Returns:
        The global CheckpointStore instance
    """
    global _global_store

    async with _global_lock:
        if _global_store is None:
            _global_store = await create_checkpoint_store()
        return _global_store


# =============================================================================
# Export
# =============================================================================

__all__ = [
    # Availability
    "CHECKPOINT_PERSISTENCE_AVAILABLE",
    "AIOSQLITE_AVAILABLE",
    "LANGGRAPH_CHECKPOINT_AVAILABLE",
    # Enums
    "RunStatus",
    "CheckpointType",
    # Config
    "CheckpointConfig",
    # Data classes
    "OrchestrationRun",
    "RalphIterationState",
    "ExecutionMetrics",
    # Store
    "CheckpointStore",
    # Factory functions
    "create_checkpoint_store",
    "get_checkpoint_store_sync",
    "checkpoint_store_context",
    "get_global_checkpoint_store",
]
