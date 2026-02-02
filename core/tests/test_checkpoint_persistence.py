#!/usr/bin/env python3
"""
Tests for the Checkpoint Persistence Layer (V33.11).

Tests verify SQLite-based persistence for orchestration runs and Ralph iterations.
Tests are skipped if aiosqlite is not available.

Run with: pytest core/tests/test_checkpoint_persistence.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Test: Import Availability
# =============================================================================

class TestImportAvailability:
    """Test that checkpoint persistence modules can be imported."""

    def test_checkpoint_module_importable(self):
        """Verify checkpoint_persistence module imports correctly."""
        from core.orchestration import checkpoint_persistence
        assert checkpoint_persistence is not None

    def test_aiosqlite_availability_flag(self):
        """Verify AIOSQLITE_AVAILABLE flag is boolean."""
        from core.orchestration.checkpoint_persistence import AIOSQLITE_AVAILABLE
        assert isinstance(AIOSQLITE_AVAILABLE, bool)

    def test_checkpoint_persistence_availability_flag(self):
        """Verify CHECKPOINT_PERSISTENCE_AVAILABLE flag is boolean."""
        from core.orchestration.checkpoint_persistence import CHECKPOINT_PERSISTENCE_AVAILABLE
        assert isinstance(CHECKPOINT_PERSISTENCE_AVAILABLE, bool)

    def test_langgraph_checkpoint_availability_flag(self):
        """Verify LANGGRAPH_CHECKPOINT_AVAILABLE flag is boolean."""
        from core.orchestration.checkpoint_persistence import LANGGRAPH_CHECKPOINT_AVAILABLE
        assert isinstance(LANGGRAPH_CHECKPOINT_AVAILABLE, bool)

    def test_orchestration_exports_checkpoint_persistence(self):
        """Verify orchestration __init__ exports checkpoint persistence symbols."""
        from core.orchestration import (
            CHECKPOINT_PERSISTENCE_AVAILABLE,
            AIOSQLITE_AVAILABLE,
            LANGGRAPH_CHECKPOINT_AVAILABLE,
        )
        assert isinstance(CHECKPOINT_PERSISTENCE_AVAILABLE, bool)
        assert isinstance(AIOSQLITE_AVAILABLE, bool)
        assert isinstance(LANGGRAPH_CHECKPOINT_AVAILABLE, bool)


# =============================================================================
# Test: Configuration Classes
# =============================================================================

class TestCheckpointConfig:
    """Test CheckpointConfig dataclass structure."""

    def test_default_config(self):
        """Test CheckpointConfig with defaults."""
        from core.orchestration.checkpoint_persistence import CheckpointConfig

        config = CheckpointConfig()
        # Default is ~/.unleash/checkpoints.db
        assert ".unleash" in config.db_path or "checkpoints.db" in config.db_path
        assert config.wal_mode is True
        assert config.auto_vacuum is True
        assert config.timeout == 30.0

    def test_custom_config(self):
        """Test CheckpointConfig with custom values."""
        from core.orchestration.checkpoint_persistence import CheckpointConfig

        config = CheckpointConfig(
            db_path="/tmp/test.db",
            wal_mode=False,
            auto_vacuum=False,
            timeout=60.0,
            max_runs_retained=500,
        )
        assert config.db_path == "/tmp/test.db"
        assert config.wal_mode is False
        assert config.auto_vacuum is False
        assert config.timeout == 60.0
        assert config.max_runs_retained == 500


class TestRunStatus:
    """Test RunStatus enum."""

    def test_run_status_values(self):
        """Test run status enum values."""
        from core.orchestration.checkpoint_persistence import RunStatus

        assert RunStatus.PENDING.value == "pending"
        assert RunStatus.RUNNING.value == "running"
        assert RunStatus.COMPLETED.value == "completed"
        assert RunStatus.FAILED.value == "failed"
        assert RunStatus.CANCELLED.value == "cancelled"


class TestCheckpointType:
    """Test CheckpointType enum."""

    def test_checkpoint_type_values(self):
        """Test checkpoint type enum values."""
        from core.orchestration.checkpoint_persistence import CheckpointType

        assert CheckpointType.ORCHESTRATION_RUN.value == "orchestration_run"
        assert CheckpointType.RALPH_ITERATION.value == "ralph_iteration"
        assert CheckpointType.WORKFLOW_STATE.value == "workflow_state"
        assert CheckpointType.AGENT_STATE.value == "agent_state"
        assert CheckpointType.METRICS.value == "metrics"


# =============================================================================
# Test: Data Classes
# =============================================================================

class TestOrchestrationRun:
    """Test OrchestrationRun data class."""

    def test_orchestration_run_creation(self):
        """Test creating OrchestrationRun."""
        from core.orchestration.checkpoint_persistence import (
            OrchestrationRun,
            RunStatus,
        )

        now = datetime.now(timezone.utc)
        run = OrchestrationRun(
            run_id="test_run_123",
            task="Test task description",
            status=RunStatus.RUNNING,
            created_at=now,
            updated_at=now,
            agents=["agent1", "agent2"],
        )
        assert run.run_id == "test_run_123"
        assert run.task == "Test task description"
        assert run.status == RunStatus.RUNNING
        assert run.created_at is not None
        assert run.agents == ["agent1", "agent2"]

    def test_orchestration_run_to_dict(self):
        """Test OrchestrationRun serialization."""
        from core.orchestration.checkpoint_persistence import (
            OrchestrationRun,
            RunStatus,
        )

        now = datetime.now(timezone.utc)
        run = OrchestrationRun(
            run_id="test_run_123",
            task="Test task",
            status=RunStatus.COMPLETED,
            created_at=now,
            updated_at=now,
            agents=["ralph"],
        )
        data = run.to_dict()
        assert data["run_id"] == "test_run_123"
        assert data["task"] == "Test task"
        assert data["status"] == "completed"
        assert data["agents"] == ["ralph"]

    def test_orchestration_run_from_dict(self):
        """Test OrchestrationRun deserialization."""
        from core.orchestration.checkpoint_persistence import (
            OrchestrationRun,
            RunStatus,
        )

        data = {
            "run_id": "test_run_456",
            "task": "Another task",
            "status": "failed",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:01:00+00:00",
            "agents": ["agent1"],
            "result": {"outcome": "test"},
            "error": "Test error",
        }
        run = OrchestrationRun.from_dict(data)
        assert run.run_id == "test_run_456"
        assert run.task == "Another task"
        assert run.status == RunStatus.FAILED
        assert run.error == "Test error"


class TestRalphIterationState:
    """Test RalphIterationState data class."""

    def test_ralph_iteration_creation(self):
        """Test creating RalphIterationState."""
        from core.orchestration.checkpoint_persistence import (
            RalphIterationState,
            RunStatus,
        )

        now = datetime.now(timezone.utc)
        state = RalphIterationState(
            iteration_id="iter_001",
            iteration_number=1,
            status=RunStatus.RUNNING,
            created_at=now,
            session_id="session_abc",
            improvements=["improvement_1"],
        )
        assert state.iteration_id == "iter_001"
        assert state.iteration_number == 1
        assert state.status == RunStatus.RUNNING
        assert state.session_id == "session_abc"
        assert state.improvements == ["improvement_1"]

    def test_ralph_iteration_to_dict(self):
        """Test RalphIterationState serialization."""
        from core.orchestration.checkpoint_persistence import (
            RalphIterationState,
            RunStatus,
        )

        now = datetime.now(timezone.utc)
        state = RalphIterationState(
            iteration_id="iter_002",
            iteration_number=5,
            status=RunStatus.COMPLETED,
            created_at=now,
            improvements=["discovery_1", "discovery_2"],
        )
        data = state.to_dict()
        assert data["iteration_id"] == "iter_002"
        assert data["iteration_number"] == 5
        assert data["status"] == "completed"
        assert data["improvements"] == ["discovery_1", "discovery_2"]


class TestExecutionMetrics:
    """Test ExecutionMetrics data class."""

    def test_execution_metrics_creation(self):
        """Test creating ExecutionMetrics."""
        from core.orchestration.checkpoint_persistence import ExecutionMetrics

        metrics = ExecutionMetrics(
            total_runs=100,
            successful_runs=80,
            failed_runs=15,
            avg_duration_ms=45.5,
        )
        assert metrics.total_runs == 100
        assert metrics.successful_runs == 80
        assert metrics.failed_runs == 15
        assert metrics.avg_duration_ms == 45.5

    def test_execution_metrics_defaults(self):
        """Test ExecutionMetrics default values."""
        from core.orchestration.checkpoint_persistence import ExecutionMetrics

        metrics = ExecutionMetrics()
        assert metrics.total_runs == 0
        assert metrics.successful_runs == 0
        assert metrics.failed_runs == 0
        assert metrics.avg_duration_ms == 0.0


# =============================================================================
# Test: CheckpointStore Class (requires aiosqlite)
# =============================================================================

# Skip tests if aiosqlite is not available
try:
    import aiosqlite
    _has_aiosqlite = True
except ImportError:
    _has_aiosqlite = False


@pytest.mark.skipif(
    not _has_aiosqlite,
    reason="aiosqlite not available"
)
class TestCheckpointStoreBasic:
    """Test CheckpointStore basic functionality."""

    @pytest.fixture
    def temp_db_path(self, tmp_path: Path) -> str:
        """Create a temporary database path."""
        return str(tmp_path / "test_checkpoints.db")

    def test_checkpoint_store_creation(self, temp_db_path: str):
        """Test creating CheckpointStore."""
        from core.orchestration.checkpoint_persistence import (
            CheckpointStore,
            CheckpointConfig,
        )

        config = CheckpointConfig(db_path=temp_db_path)
        store = CheckpointStore(config=config)
        assert store is not None
        assert store.config == config
        assert store.is_initialized is False

    def test_checkpoint_store_default_config(self):
        """Test CheckpointStore with default config."""
        from core.orchestration.checkpoint_persistence import CheckpointStore

        store = CheckpointStore()
        # Default config has ~/.unleash/checkpoints.db
        assert "checkpoints.db" in store.config.db_path


@pytest.mark.skipif(
    not _has_aiosqlite,
    reason="aiosqlite not available"
)
class TestCheckpointStoreOperations:
    """Test CheckpointStore async operations."""

    @pytest.fixture
    def temp_db_path(self, tmp_path: Path) -> str:
        """Create a temporary database path."""
        return str(tmp_path / "test_checkpoints.db")

    @pytest.mark.asyncio
    async def test_initialize_creates_tables(self, temp_db_path: str):
        """Test initialization creates database tables."""
        from core.orchestration.checkpoint_persistence import (
            CheckpointStore,
            CheckpointConfig,
        )

        config = CheckpointConfig(db_path=temp_db_path)
        store = CheckpointStore(config=config)

        await store.initialize()
        assert store.is_initialized is True

        # Verify tables exist
        async with aiosqlite.connect(temp_db_path) as db:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in await cursor.fetchall()]

        assert "orchestration_runs" in tables
        assert "ralph_iterations" in tables
        assert "checkpoints" in tables

        await store.close()

    @pytest.mark.asyncio
    async def test_save_and_get_orchestration_run(self, temp_db_path: str):
        """Test saving and retrieving orchestration runs."""
        from core.orchestration.checkpoint_persistence import (
            CheckpointStore,
            CheckpointConfig,
            OrchestrationRun,
            RunStatus,
        )

        config = CheckpointConfig(db_path=temp_db_path)
        store = CheckpointStore(config=config)
        await store.initialize()

        # Create a run using factory method
        run = await store.create_run(
            task="Test orchestration task",
            run_id="test_run_001",
            agents=["claude_flow"],
        )

        # Retrieve it
        retrieved = await store.get_run("test_run_001")
        assert retrieved is not None
        assert retrieved.run_id == "test_run_001"
        assert retrieved.task == "Test orchestration task"
        assert retrieved.status == RunStatus.PENDING  # Default status from factory

        await store.close()

    @pytest.mark.asyncio
    async def test_update_run_status(self, temp_db_path: str):
        """Test updating run status."""
        from core.orchestration.checkpoint_persistence import (
            CheckpointStore,
            CheckpointConfig,
            OrchestrationRun,
            RunStatus,
        )

        config = CheckpointConfig(db_path=temp_db_path)
        store = CheckpointStore(config=config)
        await store.initialize()

        # Create a run using factory method
        await store.create_run(
            task="Status update test",
            run_id="test_run_002",
        )

        # Update status (no result parameter in the implementation)
        await store.update_run_status(
            "test_run_002",
            RunStatus.COMPLETED,
        )

        # Verify update
        retrieved = await store.get_run("test_run_002")
        assert retrieved is not None
        assert retrieved.status == RunStatus.COMPLETED

        await store.close()

    @pytest.mark.asyncio
    async def test_save_and_get_ralph_iteration(self, temp_db_path: str):
        """Test saving and retrieving Ralph iteration states."""
        from core.orchestration.checkpoint_persistence import (
            CheckpointStore,
            CheckpointConfig,
            RalphIterationState,
            RunStatus,
        )

        config = CheckpointConfig(db_path=temp_db_path)
        store = CheckpointStore(config=config)
        await store.initialize()

        # Save an iteration using factory method
        state = await store.save_ralph_iteration(
            iteration_number=1,
            iteration_id="iter_001",
            session_id="session_abc",
            task="test_task",
            status=RunStatus.RUNNING,
        )

        # Retrieve it - use get_ralph_iterations and filter by ID
        iterations = await store.get_ralph_iterations(session_id="session_abc")
        retrieved = next((it for it in iterations if it.iteration_id == "iter_001"), None)
        assert retrieved is not None
        assert retrieved.iteration_id == "iter_001"
        assert retrieved.session_id == "session_abc"
        assert retrieved.iteration_number == 1

        await store.close()

    @pytest.mark.asyncio
    async def test_list_runs(self, temp_db_path: str):
        """Test listing runs."""
        from core.orchestration.checkpoint_persistence import (
            CheckpointStore,
            CheckpointConfig,
            OrchestrationRun,
            RunStatus,
        )

        config = CheckpointConfig(db_path=temp_db_path)
        store = CheckpointStore(config=config)
        await store.initialize()

        # Create multiple runs using factory method
        for i in range(5):
            await store.create_run(
                task=f"Task {i}",
                run_id=f"run_{i}",
            )

        # List runs
        runs = await store.list_runs(limit=10)
        assert len(runs) == 5

        await store.close()

    @pytest.mark.asyncio
    async def test_save_and_get_checkpoint(self, temp_db_path: str):
        """Test saving and retrieving generic checkpoints."""
        from core.orchestration.checkpoint_persistence import (
            CheckpointStore,
            CheckpointConfig,
            CheckpointType,
        )

        config = CheckpointConfig(db_path=temp_db_path)
        store = CheckpointStore(config=config)
        await store.initialize()

        # Save a checkpoint using correct signature: (checkpoint_type, state, *, checkpoint_id=...)
        checkpoint_data = {
            "state": "active",
            "progress": 0.5,
            "intermediate_results": [1, 2, 3],
        }
        await store.save_checkpoint(
            CheckpointType.WORKFLOW_STATE,
            checkpoint_data,
            checkpoint_id="ckpt_001",
        )

        # Retrieve it - get_checkpoint returns full record with state as a field
        retrieved = await store.get_checkpoint("ckpt_001")
        assert retrieved is not None
        assert retrieved["checkpoint_id"] == "ckpt_001"
        assert retrieved["checkpoint_type"] == CheckpointType.WORKFLOW_STATE.value
        # The state field contains the saved checkpoint_data
        state = retrieved["state"]
        assert state["state"] == "active"
        assert state["progress"] == 0.5
        assert state["intermediate_results"] == [1, 2, 3]

        await store.close()

    @pytest.mark.asyncio
    async def test_get_metrics(self, temp_db_path: str):
        """Test getting execution metrics."""
        from core.orchestration.checkpoint_persistence import (
            CheckpointStore,
            CheckpointConfig,
            OrchestrationRun,
            RunStatus,
        )

        config = CheckpointConfig(db_path=temp_db_path)
        store = CheckpointStore(config=config)
        await store.initialize()

        # Create runs and update their statuses
        statuses = [
            RunStatus.COMPLETED,
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        ]
        for i, status in enumerate(statuses):
            await store.create_run(
                task=f"Metrics task {i}",
                run_id=f"metrics_run_{i}",
            )
            await store.update_run_status(f"metrics_run_{i}", status)

        # Get metrics
        metrics = await store.get_metrics()
        assert metrics.total_runs == 4
        assert metrics.successful_runs == 2
        assert metrics.failed_runs == 1

        await store.close()

    @pytest.mark.asyncio
    async def test_context_manager_usage(self, temp_db_path: str):
        """Test using CheckpointStore as async context manager."""
        from core.orchestration.checkpoint_persistence import (
            CheckpointStore,
            CheckpointConfig,
            OrchestrationRun,
            RunStatus,
        )

        config = CheckpointConfig(db_path=temp_db_path)

        async with CheckpointStore(config=config) as store:
            assert store.is_initialized is True

            await store.create_run(
                task="Context manager test",
                run_id="ctx_run_001",
            )

            retrieved = await store.get_run("ctx_run_001")
            assert retrieved is not None

        # After exiting context, connection should be closed
        assert store._conn is None


# =============================================================================
# Test: Factory Functions
# =============================================================================

@pytest.mark.skipif(
    not _has_aiosqlite,
    reason="aiosqlite not available"
)
class TestFactoryFunctions:
    """Test factory functions."""

    @pytest.fixture
    def temp_db_path(self, tmp_path: Path) -> str:
        """Create a temporary database path."""
        return str(tmp_path / "test_checkpoints.db")

    def test_get_checkpoint_store_sync(self, temp_db_path: str):
        """Test synchronous factory function."""
        from core.orchestration.checkpoint_persistence import (
            get_checkpoint_store_sync,
            CheckpointConfig,
        )

        config = CheckpointConfig(db_path=temp_db_path)
        store = get_checkpoint_store_sync(config=config)
        assert store is not None
        assert store.is_initialized is False

    @pytest.mark.asyncio
    async def test_create_checkpoint_store(self, temp_db_path: str):
        """Test async factory function."""
        from core.orchestration.checkpoint_persistence import (
            create_checkpoint_store,
            CheckpointConfig,
        )

        config = CheckpointConfig(db_path=temp_db_path)
        store = await create_checkpoint_store(config=config)
        assert store is not None
        assert store.is_initialized is True

        await store.close()


# =============================================================================
# Test: Session Recovery
# =============================================================================

@pytest.mark.skipif(
    not _has_aiosqlite,
    reason="aiosqlite not available"
)
class TestSessionRecovery:
    """Test session recovery functionality."""

    @pytest.fixture
    def temp_db_path(self, tmp_path: Path) -> str:
        """Create a temporary database path."""
        return str(tmp_path / "test_checkpoints.db")

    @pytest.mark.asyncio
    async def test_get_session_iterations(self, temp_db_path: str):
        """Test retrieving all iterations for a session."""
        from core.orchestration.checkpoint_persistence import (
            CheckpointStore,
            CheckpointConfig,
            RalphIterationState,
            RunStatus,
        )

        config = CheckpointConfig(db_path=temp_db_path)
        async with CheckpointStore(config=config) as store:
            # Save multiple iterations for same session using factory method
            for i in range(3):
                await store.save_ralph_iteration(
                    iteration_number=i + 1,
                    iteration_id=f"iter_{i}",
                    session_id="session_abc",
                    status=RunStatus.COMPLETED,
                )

            # Save iteration for different session
            await store.save_ralph_iteration(
                iteration_number=1,
                iteration_id="iter_other",
                session_id="session_xyz",
                status=RunStatus.COMPLETED,
            )

            # Get iterations for specific session
            iterations = await store.get_ralph_iterations(session_id="session_abc")
            assert len(iterations) == 3
            assert all(it.session_id == "session_abc" for it in iterations)

    @pytest.mark.asyncio
    async def test_get_latest_iteration(self, temp_db_path: str):
        """Test getting the latest iteration for a session."""
        from core.orchestration.checkpoint_persistence import (
            CheckpointStore,
            CheckpointConfig,
            RalphIterationState,
            RunStatus,
        )

        config = CheckpointConfig(db_path=temp_db_path)
        async with CheckpointStore(config=config) as store:
            # Save iterations out of order using factory method
            for i in [2, 1, 3]:
                await store.save_ralph_iteration(
                    iteration_number=i,
                    iteration_id=f"iter_{i}",
                    session_id="session_abc",
                    status=RunStatus.COMPLETED,
                )

            # Get latest - sort by iteration_number descending and take first
            iterations = await store.get_ralph_iterations(session_id="session_abc")
            sorted_iterations = sorted(iterations, key=lambda x: x.iteration_number, reverse=True)
            latest = sorted_iterations[0] if sorted_iterations else None
            assert latest is not None
            assert latest.iteration_number == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
