"""
Saga Orchestration Pattern - V36 Architecture

Implements the Saga pattern for distributed transactions across multiple SDK adapters.
Provides automatic compensation (rollback) on failure with state persistence for crash recovery.

Architecture Decision: ADR-029 - Saga Pattern for Distributed Transactions
Features:
- Step-based saga execution with compensation actions
- Async execution with configurable timeouts
- SQLite state persistence for crash recovery
- Idempotency keys for safe retries
- Prometheus-style metrics tracking
- Integration with SDK Registry for multi-adapter operations

Usage:
    from core.orchestration.saga import (
        SagaOrchestrator, Saga, SagaStep, SagaResult
    )

    # Define saga steps
    saga = Saga(
        saga_id="order-processing-001",
        steps=[
            SagaStep(
                name="reserve_inventory",
                execute=reserve_inventory,
                compensate=release_inventory,
                timeout_seconds=10.0
            ),
            SagaStep(
                name="charge_payment",
                execute=charge_payment,
                compensate=refund_payment,
                timeout_seconds=30.0
            ),
            SagaStep(
                name="confirm_order",
                execute=confirm_order,
                compensate=cancel_order,
                timeout_seconds=5.0
            ),
        ]
    )

    # Execute saga
    orchestrator = SagaOrchestrator(db_path="sagas.db")
    result = await orchestrator.execute(saga)

    if result.success:
        print(f"Saga completed: {result.completed_steps}")
    else:
        print(f"Saga failed at step {result.failed_at_step}: {result.error}")
        print(f"Compensated steps: {result.compensated_steps}")
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union

from .base import AdapterResult, SDKAdapter
from .sdk_registry import SDKRegistry, get_registry
from .infrastructure.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SagaState(Enum):
    """Saga lifecycle states."""
    PENDING = auto()         # Created but not started
    RUNNING = auto()         # Currently executing steps
    COMPENSATING = auto()    # Running compensation after failure
    COMPLETED = auto()       # Successfully completed all steps
    FAILED = auto()          # Failed and compensation completed
    PARTIALLY_FAILED = auto()  # Compensation also failed


class StepState(Enum):
    """Individual step states."""
    PENDING = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    FAILED = auto()
    COMPENSATING = auto()
    COMPENSATED = auto()
    COMPENSATION_FAILED = auto()


@dataclass
class SagaStep:
    """
    A single step in a saga transaction.

    Attributes:
        name: Unique name for this step within the saga
        execute: Async function to execute the step
        compensate: Async function to compensate/rollback the step
        timeout_seconds: Maximum time for step execution
        retry_count: Number of retries on failure
        retry_delay_seconds: Delay between retries
        idempotency_key: Optional key for safe retries (auto-generated if not provided)
    """
    name: str
    execute: Callable[[Dict[str, Any]], Awaitable[Any]]
    compensate: Callable[[Dict[str, Any], Any], Awaitable[None]]
    timeout_seconds: float = 30.0
    retry_count: int = 0
    retry_delay_seconds: float = 1.0
    idempotency_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.idempotency_key is None:
            self.idempotency_key = str(uuid.uuid4())


@dataclass
class StepResult:
    """Result of executing a single step."""
    step_name: str
    state: StepState
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retry_attempts: int = 0
    idempotency_key: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        """Calculate step duration in milliseconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            "step_name": self.step_name,
            "state": self.state.name,
            "result": self.result if self._is_serializable(self.result) else str(self.result),
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "retry_attempts": self.retry_attempts,
            "idempotency_key": self.idempotency_key,
            "duration_ms": self.duration_ms,
        }

    @staticmethod
    def _is_serializable(obj: Any) -> bool:
        """Check if object is JSON serializable."""
        try:
            json.dumps(obj)
            return True
        except (TypeError, ValueError):
            return False


@dataclass
class Saga:
    """
    A saga representing a distributed transaction.

    Attributes:
        saga_id: Unique identifier for this saga instance
        steps: Ordered list of saga steps
        context: Shared context passed between steps
        metadata: Additional saga metadata
    """
    saga_id: str
    steps: List[SagaStep]
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.saga_id:
            self.saga_id = str(uuid.uuid4())

    def generate_idempotency_key(self) -> str:
        """Generate a deterministic idempotency key for the saga."""
        content = json.dumps({
            "saga_id": self.saga_id,
            "steps": [s.name for s in self.steps],
            "context": self.context,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class SagaResult:
    """
    Result of saga execution.

    Attributes:
        saga_id: The saga identifier
        success: Whether the saga completed successfully
        state: Final saga state
        completed_steps: List of successfully completed steps
        failed_at_step: Index of the step that failed (if any)
        compensated_steps: List of steps that were compensated
        error: Error message if failed
        step_results: Detailed results for each step
        started_at: Saga start timestamp
        completed_at: Saga completion timestamp
    """
    saga_id: str
    success: bool
    state: SagaState
    completed_steps: List[str] = field(default_factory=list)
    failed_at_step: Optional[int] = None
    compensated_steps: List[str] = field(default_factory=list)
    error: Optional[str] = None
    step_results: List[StepResult] = field(default_factory=list)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    @property
    def duration_ms(self) -> float:
        """Calculate total saga duration in milliseconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            "saga_id": self.saga_id,
            "success": self.success,
            "state": self.state.name,
            "completed_steps": self.completed_steps,
            "failed_at_step": self.failed_at_step,
            "compensated_steps": self.compensated_steps,
            "error": self.error,
            "step_results": [sr.to_dict() for sr in self.step_results],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
        }


class SagaStateStore:
    """
    SQLite-based state persistence for saga recovery.

    Provides durability for saga state, enabling recovery after crashes.
    Uses WAL mode for better concurrent performance.
    """

    def __init__(self, db_path: str = "sagas.db"):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;

                CREATE TABLE IF NOT EXISTS sagas (
                    saga_id TEXT PRIMARY KEY,
                    idempotency_key TEXT UNIQUE,
                    state TEXT NOT NULL,
                    context TEXT,
                    metadata TEXT,
                    started_at REAL,
                    updated_at REAL,
                    completed_at REAL,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS saga_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    saga_id TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    step_name TEXT NOT NULL,
                    state TEXT NOT NULL,
                    result TEXT,
                    error TEXT,
                    started_at REAL,
                    completed_at REAL,
                    retry_attempts INTEGER DEFAULT 0,
                    idempotency_key TEXT,
                    FOREIGN KEY (saga_id) REFERENCES sagas(saga_id),
                    UNIQUE(saga_id, step_index)
                );

                CREATE TABLE IF NOT EXISTS idempotency_keys (
                    key TEXT PRIMARY KEY,
                    saga_id TEXT NOT NULL,
                    step_name TEXT,
                    result TEXT,
                    created_at REAL NOT NULL,
                    expires_at REAL
                );

                CREATE INDEX IF NOT EXISTS idx_sagas_state ON sagas(state);
                CREATE INDEX IF NOT EXISTS idx_sagas_idempotency ON sagas(idempotency_key);
                CREATE INDEX IF NOT EXISTS idx_steps_saga ON saga_steps(saga_id);
                CREATE INDEX IF NOT EXISTS idx_idempotency_expires ON idempotency_keys(expires_at);
            """)

    @asynccontextmanager
    async def _get_async_connection(self):
        """Get async database connection (runs in thread pool)."""
        loop = asyncio.get_event_loop()
        conn = await loop.run_in_executor(None, self._get_connection_sync)
        try:
            yield conn
        finally:
            await loop.run_in_executor(None, conn.close)

    def _get_connection_sync(self) -> sqlite3.Connection:
        """Get synchronous database connection."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection (context manager)."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    async def save_saga(
        self,
        saga: Saga,
        state: SagaState,
        error: Optional[str] = None,
        started_at: Optional[float] = None,
        completed_at: Optional[float] = None,
    ) -> None:
        """Save saga state to database."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._save_saga_sync,
            saga, state, error, started_at, completed_at
        )

    def _save_saga_sync(
        self,
        saga: Saga,
        state: SagaState,
        error: Optional[str],
        started_at: Optional[float],
        completed_at: Optional[float],
    ) -> None:
        """Synchronous saga save."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO sagas
                    (saga_id, idempotency_key, state, context, metadata, started_at, updated_at, completed_at, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    saga.saga_id,
                    saga.generate_idempotency_key(),
                    state.name,
                    json.dumps(saga.context),
                    json.dumps(saga.metadata),
                    started_at,
                    time.time(),
                    completed_at,
                    error,
                ))
                conn.commit()

    async def save_step(
        self,
        saga_id: str,
        step_index: int,
        step_result: StepResult,
    ) -> None:
        """Save step state to database."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._save_step_sync,
            saga_id, step_index, step_result
        )

    def _save_step_sync(
        self,
        saga_id: str,
        step_index: int,
        step_result: StepResult,
    ) -> None:
        """Synchronous step save."""
        with self._lock:
            with self._get_connection() as conn:
                result_json = None
                if step_result.result is not None:
                    try:
                        result_json = json.dumps(step_result.result)
                    except (TypeError, ValueError):
                        result_json = json.dumps(str(step_result.result))

                conn.execute("""
                    INSERT OR REPLACE INTO saga_steps
                    (saga_id, step_index, step_name, state, result, error, started_at, completed_at, retry_attempts, idempotency_key)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    saga_id,
                    step_index,
                    step_result.step_name,
                    step_result.state.name,
                    result_json,
                    step_result.error,
                    step_result.started_at,
                    step_result.completed_at,
                    step_result.retry_attempts,
                    step_result.idempotency_key,
                ))
                conn.commit()

    async def get_saga(self, saga_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve saga state from database."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_saga_sync, saga_id)

    def _get_saga_sync(self, saga_id: str) -> Optional[Dict[str, Any]]:
        """Synchronous saga retrieval."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM sagas WHERE saga_id = ?",
                (saga_id,)
            ).fetchone()

            if not row:
                return None

            steps = conn.execute(
                "SELECT * FROM saga_steps WHERE saga_id = ? ORDER BY step_index",
                (saga_id,)
            ).fetchall()

            return {
                "saga_id": row["saga_id"],
                "idempotency_key": row["idempotency_key"],
                "state": row["state"],
                "context": json.loads(row["context"]) if row["context"] else {},
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
                "error": row["error"],
                "steps": [dict(s) for s in steps],
            }

    async def check_idempotency(
        self,
        key: str,
    ) -> Optional[Dict[str, Any]]:
        """Check if an idempotency key exists and return cached result."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._check_idempotency_sync, key)

    def _check_idempotency_sync(self, key: str) -> Optional[Dict[str, Any]]:
        """Synchronous idempotency check."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM idempotency_keys WHERE key = ? AND (expires_at IS NULL OR expires_at > ?)",
                (key, time.time())
            ).fetchone()

            if row:
                return {
                    "saga_id": row["saga_id"],
                    "step_name": row["step_name"],
                    "result": json.loads(row["result"]) if row["result"] else None,
                }
            return None

    async def store_idempotency(
        self,
        key: str,
        saga_id: str,
        step_name: Optional[str],
        result: Any,
        ttl_seconds: float = 86400.0,
    ) -> None:
        """Store an idempotency key with result."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._store_idempotency_sync,
            key, saga_id, step_name, result, ttl_seconds
        )

    def _store_idempotency_sync(
        self,
        key: str,
        saga_id: str,
        step_name: Optional[str],
        result: Any,
        ttl_seconds: float,
    ) -> None:
        """Synchronous idempotency storage."""
        with self._lock:
            with self._get_connection() as conn:
                result_json = None
                if result is not None:
                    try:
                        result_json = json.dumps(result)
                    except (TypeError, ValueError):
                        result_json = json.dumps(str(result))

                conn.execute("""
                    INSERT OR REPLACE INTO idempotency_keys
                    (key, saga_id, step_name, result, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    key,
                    saga_id,
                    step_name,
                    result_json,
                    time.time(),
                    time.time() + ttl_seconds,
                ))
                conn.commit()

    async def get_incomplete_sagas(self) -> List[Dict[str, Any]]:
        """Get all incomplete sagas for recovery."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_incomplete_sync)

    def _get_incomplete_sync(self) -> List[Dict[str, Any]]:
        """Synchronous incomplete saga retrieval."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM sagas WHERE state IN (?, ?)",
                (SagaState.RUNNING.name, SagaState.COMPENSATING.name)
            ).fetchall()

            return [dict(r) for r in rows]

    async def cleanup_expired(self) -> int:
        """Remove expired idempotency keys."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._cleanup_expired_sync)

    def _cleanup_expired_sync(self) -> int:
        """Synchronous cleanup."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM idempotency_keys WHERE expires_at < ?",
                    (time.time(),)
                )
                conn.commit()
                return cursor.rowcount


@dataclass
class SagaMetrics:
    """Metrics tracking for saga orchestration."""
    total_sagas: int = 0
    successful_sagas: int = 0
    failed_sagas: int = 0
    compensation_count: int = 0
    total_steps_executed: int = 0
    total_steps_compensated: int = 0
    total_duration_ms: float = 0.0
    step_durations: Dict[str, List[float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def success_rate(self) -> float:
        """Calculate saga success rate."""
        if self.total_sagas == 0:
            return 0.0
        return self.successful_sagas / self.total_sagas

    @property
    def compensation_rate(self) -> float:
        """Calculate compensation rate."""
        if self.total_sagas == 0:
            return 0.0
        return self.compensation_count / self.total_sagas

    @property
    def avg_duration_ms(self) -> float:
        """Calculate average saga duration."""
        if self.total_sagas == 0:
            return 0.0
        return self.total_duration_ms / self.total_sagas

    def record_saga_complete(self, success: bool, duration_ms: float, compensated: bool) -> None:
        """Record saga completion."""
        with self._lock:
            self.total_sagas += 1
            self.total_duration_ms += duration_ms
            if success:
                self.successful_sagas += 1
            else:
                self.failed_sagas += 1
            if compensated:
                self.compensation_count += 1

    def record_step(self, step_name: str, duration_ms: float, compensated: bool = False) -> None:
        """Record step execution."""
        with self._lock:
            if compensated:
                self.total_steps_compensated += 1
            else:
                self.total_steps_executed += 1

            if step_name not in self.step_durations:
                self.step_durations[step_name] = []
            self.step_durations[step_name].append(duration_ms)

            # Keep only last 1000 samples per step
            if len(self.step_durations[step_name]) > 1000:
                self.step_durations[step_name] = self.step_durations[step_name][-1000:]

    def get_step_stats(self, step_name: str) -> Dict[str, float]:
        """Get statistics for a specific step."""
        durations = self.step_durations.get(step_name, [])
        if not durations:
            return {"count": 0, "avg_ms": 0, "min_ms": 0, "max_ms": 0, "p95_ms": 0}

        sorted_durations = sorted(durations)
        p95_idx = int(len(sorted_durations) * 0.95)

        return {
            "count": len(durations),
            "avg_ms": sum(durations) / len(durations),
            "min_ms": min(durations),
            "max_ms": max(durations),
            "p95_ms": sorted_durations[min(p95_idx, len(sorted_durations) - 1)],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            "total_sagas": self.total_sagas,
            "successful_sagas": self.successful_sagas,
            "failed_sagas": self.failed_sagas,
            "compensation_count": self.compensation_count,
            "success_rate": round(self.success_rate * 100, 2),
            "compensation_rate": round(self.compensation_rate * 100, 2),
            "avg_duration_ms": round(self.avg_duration_ms, 2),
            "total_steps_executed": self.total_steps_executed,
            "total_steps_compensated": self.total_steps_compensated,
            "step_stats": {
                name: self.get_step_stats(name)
                for name in self.step_durations
            },
        }


class SagaOrchestrator:
    """
    Orchestrates distributed transactions using the Saga pattern.

    Features:
    - Step-based execution with automatic rollback
    - State persistence for crash recovery
    - Idempotency key support for safe retries
    - Integration with SDK Registry
    - Comprehensive metrics tracking

    Example:
        orchestrator = SagaOrchestrator(db_path="sagas.db")

        # Define a saga for multi-adapter operation
        saga = Saga(
            saga_id="multi-search-001",
            steps=[
                SagaStep(
                    name="search_memory",
                    execute=search_memory_adapter,
                    compensate=rollback_memory_search,
                ),
                SagaStep(
                    name="search_knowledge",
                    execute=search_knowledge_adapter,
                    compensate=rollback_knowledge_search,
                ),
            ],
            context={"query": "authentication patterns"},
        )

        result = await orchestrator.execute(saga)
    """

    def __init__(
        self,
        db_path: str = "sagas.db",
        registry: Optional[SDKRegistry] = None,
        enable_persistence: bool = True,
        idempotency_ttl_seconds: float = 86400.0,
    ):
        """
        Initialize the saga orchestrator.

        Args:
            db_path: Path to SQLite database for state persistence
            registry: SDK Registry instance (uses global if not provided)
            enable_persistence: Whether to persist state to database
            idempotency_ttl_seconds: TTL for idempotency keys
        """
        self._registry = registry or get_registry()
        self._enable_persistence = enable_persistence
        self._idempotency_ttl = idempotency_ttl_seconds
        self._metrics = SagaMetrics()

        if enable_persistence:
            self._store = SagaStateStore(db_path)
        else:
            self._store = None

        logger.info(f"SagaOrchestrator initialized (persistence={enable_persistence})")

    @property
    def metrics(self) -> SagaMetrics:
        """Get metrics tracker."""
        return self._metrics

    async def execute(
        self,
        saga: Saga,
        skip_idempotency_check: bool = False,
    ) -> SagaResult:
        """
        Execute a saga with automatic compensation on failure.

        Args:
            saga: The saga to execute
            skip_idempotency_check: Skip checking for existing saga with same idempotency key

        Returns:
            SagaResult with execution details
        """
        started_at = time.time()
        idempotency_key = saga.generate_idempotency_key()

        # Check idempotency
        if not skip_idempotency_check and self._store:
            cached = await self._store.check_idempotency(idempotency_key)
            if cached:
                logger.info(f"[Saga:{saga.saga_id}] Returning cached result (idempotency)")
                existing = await self._store.get_saga(cached["saga_id"])
                if existing:
                    return SagaResult(
                        saga_id=existing["saga_id"],
                        success=existing["state"] == SagaState.COMPLETED.name,
                        state=SagaState[existing["state"]],
                        completed_steps=[],
                        error=existing.get("error"),
                        started_at=existing.get("started_at"),
                        completed_at=existing.get("completed_at"),
                    )

        # Initialize result tracking
        result = SagaResult(
            saga_id=saga.saga_id,
            success=False,
            state=SagaState.RUNNING,
            started_at=started_at,
        )

        # Persist initial state
        if self._store:
            await self._store.save_saga(saga, SagaState.RUNNING, started_at=started_at)

        completed_step_results: List[StepResult] = []
        failed_at_step: Optional[int] = None

        try:
            # Execute each step
            for step_index, step in enumerate(saga.steps):
                step_result = await self._execute_step(
                    saga, step, step_index, saga.context
                )
                result.step_results.append(step_result)

                if step_result.state == StepState.COMPLETED:
                    completed_step_results.append(step_result)
                    result.completed_steps.append(step.name)

                    # Update context with step result if available
                    if step_result.result is not None:
                        saga.context[f"_result_{step.name}"] = step_result.result
                else:
                    # Step failed
                    failed_at_step = step_index
                    result.failed_at_step = step_index
                    result.error = step_result.error
                    result.state = SagaState.COMPENSATING

                    if self._store:
                        await self._store.save_saga(
                            saga, SagaState.COMPENSATING, error=step_result.error
                        )

                    logger.warning(
                        f"[Saga:{saga.saga_id}] Step {step.name} failed: {step_result.error}"
                    )
                    break

            # If all steps completed
            if failed_at_step is None:
                result.success = True
                result.state = SagaState.COMPLETED
                result.completed_at = time.time()

                if self._store:
                    await self._store.save_saga(
                        saga, SagaState.COMPLETED, completed_at=result.completed_at
                    )
                    await self._store.store_idempotency(
                        idempotency_key,
                        saga.saga_id,
                        None,
                        result.to_dict(),
                        self._idempotency_ttl,
                    )

                logger.info(f"[Saga:{saga.saga_id}] Completed successfully")

            else:
                # Run compensation
                await self.compensate(saga, completed_step_results, result)

        except Exception as e:
            logger.error(f"[Saga:{saga.saga_id}] Unexpected error: {e}")
            result.error = str(e)
            result.state = SagaState.FAILED
            result.completed_at = time.time()

            if self._store:
                await self._store.save_saga(
                    saga, SagaState.FAILED, error=str(e), completed_at=result.completed_at
                )

        # Record metrics
        self._metrics.record_saga_complete(
            success=result.success,
            duration_ms=result.duration_ms,
            compensated=len(result.compensated_steps) > 0,
        )

        return result

    async def compensate(
        self,
        saga: Saga,
        completed_step_results: List[StepResult],
        result: SagaResult,
    ) -> None:
        """
        Run compensation for all completed steps in reverse order.

        Args:
            saga: The saga being compensated
            completed_step_results: Results of completed steps to compensate
            result: The saga result being updated
        """
        logger.info(
            f"[Saga:{saga.saga_id}] Starting compensation for {len(completed_step_results)} steps"
        )

        compensation_failures: List[str] = []

        # Compensate in reverse order
        for step_result in reversed(completed_step_results):
            # Find the corresponding step definition
            step = next((s for s in saga.steps if s.name == step_result.step_name), None)
            if not step:
                logger.error(f"[Saga:{saga.saga_id}] Step {step_result.step_name} not found for compensation")
                continue

            try:
                comp_started = time.time()
                await asyncio.wait_for(
                    step.compensate(saga.context, step_result.result),
                    timeout=step.timeout_seconds,
                )
                comp_duration = (time.time() - comp_started) * 1000

                result.compensated_steps.append(step.name)
                self._metrics.record_step(step.name, comp_duration, compensated=True)

                logger.info(f"[Saga:{saga.saga_id}] Compensated step: {step.name}")

                if self._store:
                    await self._store.save_step(
                        saga.saga_id,
                        saga.steps.index(step),
                        StepResult(
                            step_name=step.name,
                            state=StepState.COMPENSATED,
                            started_at=comp_started,
                            completed_at=time.time(),
                        ),
                    )

            except asyncio.TimeoutError:
                error_msg = f"Compensation timeout for step {step.name}"
                logger.error(f"[Saga:{saga.saga_id}] {error_msg}")
                compensation_failures.append(error_msg)

            except Exception as e:
                error_msg = f"Compensation failed for step {step.name}: {e}"
                logger.error(f"[Saga:{saga.saga_id}] {error_msg}")
                compensation_failures.append(error_msg)

        # Determine final state
        if compensation_failures:
            result.state = SagaState.PARTIALLY_FAILED
            result.error = f"{result.error}; Compensation errors: {'; '.join(compensation_failures)}"
        else:
            result.state = SagaState.FAILED

        result.completed_at = time.time()

        if self._store:
            await self._store.save_saga(
                saga,
                result.state,
                error=result.error,
                completed_at=result.completed_at,
            )

    async def _execute_step(
        self,
        saga: Saga,
        step: SagaStep,
        step_index: int,
        context: Dict[str, Any],
    ) -> StepResult:
        """Execute a single saga step with retry logic."""
        step_result = StepResult(
            step_name=step.name,
            state=StepState.EXECUTING,
            started_at=time.time(),
            idempotency_key=step.idempotency_key,
        )

        # Check step idempotency
        if self._store and step.idempotency_key:
            cached = await self._store.check_idempotency(step.idempotency_key)
            if cached:
                logger.info(f"[Saga:{saga.saga_id}] Step {step.name} returning cached result")
                step_result.state = StepState.COMPLETED
                step_result.result = cached.get("result")
                step_result.completed_at = time.time()
                return step_result

        # Persist step start
        if self._store:
            await self._store.save_step(saga.saga_id, step_index, step_result)

        # Execute with retries
        last_error: Optional[Exception] = None

        for attempt in range(step.retry_count + 1):
            step_result.retry_attempts = attempt

            try:
                result = await asyncio.wait_for(
                    step.execute(context),
                    timeout=step.timeout_seconds,
                )

                step_result.state = StepState.COMPLETED
                step_result.result = result
                step_result.completed_at = time.time()

                # Record metrics
                self._metrics.record_step(step.name, step_result.duration_ms)

                # Store idempotency
                if self._store and step.idempotency_key:
                    await self._store.store_idempotency(
                        step.idempotency_key,
                        saga.saga_id,
                        step.name,
                        result,
                        self._idempotency_ttl,
                    )

                # Persist step completion
                if self._store:
                    await self._store.save_step(saga.saga_id, step_index, step_result)

                logger.debug(
                    f"[Saga:{saga.saga_id}] Step {step.name} completed in {step_result.duration_ms:.2f}ms"
                )
                return step_result

            except asyncio.TimeoutError:
                last_error = TimeoutError(f"Step {step.name} timed out after {step.timeout_seconds}s")
                logger.warning(
                    f"[Saga:{saga.saga_id}] Step {step.name} timeout (attempt {attempt + 1}/{step.retry_count + 1})"
                )

            except Exception as e:
                last_error = e
                logger.warning(
                    f"[Saga:{saga.saga_id}] Step {step.name} failed (attempt {attempt + 1}/{step.retry_count + 1}): {e}"
                )

            # Wait before retry
            if attempt < step.retry_count:
                await asyncio.sleep(step.retry_delay_seconds * (2 ** attempt))  # Exponential backoff

        # All retries exhausted
        step_result.state = StepState.FAILED
        step_result.error = str(last_error)
        step_result.completed_at = time.time()

        if self._store:
            await self._store.save_step(saga.saga_id, step_index, step_result)

        return step_result

    async def recover_incomplete(self) -> List[SagaResult]:
        """
        Recover and complete any incomplete sagas after restart.

        Returns:
            List of recovery results
        """
        if not self._store:
            return []

        results: List[SagaResult] = []
        incomplete = await self._store.get_incomplete_sagas()

        for saga_data in incomplete:
            logger.info(f"[Recovery] Found incomplete saga: {saga_data['saga_id']}")

            # For now, mark as failed (full recovery would require step definitions)
            result = SagaResult(
                saga_id=saga_data["saga_id"],
                success=False,
                state=SagaState.FAILED,
                error="Recovered after crash - marked as failed",
                started_at=saga_data.get("started_at"),
                completed_at=time.time(),
            )

            await self._store.save_saga(
                Saga(saga_id=saga_data["saga_id"], steps=[]),
                SagaState.FAILED,
                error="Recovered after crash",
                completed_at=time.time(),
            )

            results.append(result)

        return results

    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics."""
        return self._metrics.to_dict()


# ============================================================================
# Example: Multi-Adapter Saga Operations
# ============================================================================

async def create_multi_adapter_saga(
    registry: SDKRegistry,
    query: str,
    adapters: List[str],
) -> Saga:
    """
    Create a saga for querying multiple SDK adapters.

    Example usage for researching across memory and knowledge layers.

    Args:
        registry: The SDK registry
        query: Search query
        adapters: List of adapter names to query

    Returns:
        Configured Saga ready for execution
    """
    steps: List[SagaStep] = []

    for adapter_name in adapters:
        # Create execute function
        async def execute_search(
            ctx: Dict[str, Any],
            name: str = adapter_name,
        ) -> Dict[str, Any]:
            adapter = await registry.get(name)
            if not adapter:
                raise ValueError(f"Adapter {name} not found")

            result = await adapter.execute("search", query=ctx.get("query", ""))
            return {
                "adapter": name,
                "success": result.success,
                "data": result.data,
                "latency_ms": result.latency_ms,
            }

        # Create compensate function (for search, just log - no actual rollback needed)
        async def compensate_search(
            ctx: Dict[str, Any],
            result: Any,
            name: str = adapter_name,
        ) -> None:
            logger.info(f"[Compensate] Rolling back search on {name} (no-op for read operations)")

        steps.append(SagaStep(
            name=f"search_{adapter_name}",
            execute=execute_search,
            compensate=compensate_search,
            timeout_seconds=30.0,
            metadata={"adapter": adapter_name},
        ))

    return Saga(
        saga_id=f"multi-search-{uuid.uuid4().hex[:8]}",
        steps=steps,
        context={"query": query},
        metadata={"adapters": adapters, "query": query},
    )


async def example_order_processing_saga() -> Saga:
    """
    Example: Order processing saga with inventory, payment, and confirmation.

    Demonstrates a typical e-commerce distributed transaction pattern.
    """

    async def reserve_inventory(ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Reserve inventory for the order."""
        order_id = ctx.get("order_id")
        items = ctx.get("items", [])
        # Simulate inventory reservation
        await asyncio.sleep(0.1)
        return {
            "reservation_id": f"res-{order_id}",
            "items_reserved": len(items),
            "timestamp": time.time(),
        }

    async def release_inventory(ctx: Dict[str, Any], result: Any) -> None:
        """Release reserved inventory."""
        reservation_id = result.get("reservation_id") if result else None
        logger.info(f"Releasing inventory reservation: {reservation_id}")
        await asyncio.sleep(0.05)

    async def charge_payment(ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Charge customer payment."""
        order_id = ctx.get("order_id")
        amount = ctx.get("amount", 0)
        # Simulate payment processing
        await asyncio.sleep(0.2)
        return {
            "transaction_id": f"txn-{order_id}",
            "amount_charged": amount,
            "timestamp": time.time(),
        }

    async def refund_payment(ctx: Dict[str, Any], result: Any) -> None:
        """Refund charged payment."""
        transaction_id = result.get("transaction_id") if result else None
        logger.info(f"Refunding transaction: {transaction_id}")
        await asyncio.sleep(0.1)

    async def confirm_order(ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Confirm the order."""
        order_id = ctx.get("order_id")
        # Simulate order confirmation
        await asyncio.sleep(0.05)
        return {
            "confirmation_number": f"conf-{order_id}",
            "status": "confirmed",
            "timestamp": time.time(),
        }

    async def cancel_order(ctx: Dict[str, Any], result: Any) -> None:
        """Cancel confirmed order."""
        confirmation = result.get("confirmation_number") if result else None
        logger.info(f"Cancelling order: {confirmation}")
        await asyncio.sleep(0.05)

    return Saga(
        saga_id=f"order-{uuid.uuid4().hex[:8]}",
        steps=[
            SagaStep(
                name="reserve_inventory",
                execute=reserve_inventory,
                compensate=release_inventory,
                timeout_seconds=10.0,
            ),
            SagaStep(
                name="charge_payment",
                execute=charge_payment,
                compensate=refund_payment,
                timeout_seconds=30.0,
            ),
            SagaStep(
                name="confirm_order",
                execute=confirm_order,
                compensate=cancel_order,
                timeout_seconds=5.0,
            ),
        ],
        context={
            "order_id": f"ord-{uuid.uuid4().hex[:8]}",
            "items": ["item1", "item2", "item3"],
            "amount": 99.99,
        },
    )


# Export public interface
__all__ = [
    "SagaState",
    "StepState",
    "SagaStep",
    "StepResult",
    "Saga",
    "SagaResult",
    "SagaStateStore",
    "SagaMetrics",
    "SagaOrchestrator",
    "create_multi_adapter_saga",
    "example_order_processing_saga",
]
