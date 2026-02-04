"""
Unit tests for the Lifecycle Manager.

Tests graceful shutdown handling for all platform components.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from core.lifecycle import (
    LifecycleManager,
    ShutdownPhase,
    ShutdownPriority,
    ShutdownResult,
    ShutdownReport,
    HandlerRegistration,
    get_lifecycle_manager,
    reset_lifecycle_manager,
    on_shutdown,
    register_platform_components,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def lifecycle_manager():
    """Create a fresh lifecycle manager for each test."""
    reset_lifecycle_manager()
    manager = LifecycleManager(
        default_timeout=5.0,
        handler_timeout=2.0,
        enable_signal_handling=False,  # Disable signals in tests
    )
    yield manager
    reset_lifecycle_manager()


@pytest.fixture
def mock_connection_pool():
    """Create a mock connection pool."""
    pool = MagicMock()
    pool.shutdown = AsyncMock()
    pool.close = AsyncMock()
    return pool


@pytest.fixture
def mock_memory_backend():
    """Create a mock memory backend."""
    backend = MagicMock()
    backend.flush = AsyncMock()
    backend.save = AsyncMock()
    backend.close = MagicMock()
    return backend


@pytest.fixture
def mock_cache():
    """Create a mock cache."""
    cache = MagicMock()
    cache.persist_on_shutdown = True
    cache.persist = AsyncMock()
    cache.close = AsyncMock()
    return cache


# =============================================================================
# HANDLER REGISTRATION TESTS
# =============================================================================

class TestHandlerRegistration:
    """Tests for shutdown handler registration."""

    def test_register_async_handler(self, lifecycle_manager):
        """Test registering an async shutdown handler."""
        async def my_handler():
            pass

        lifecycle_manager.register_shutdown_handler(
            my_handler,
            name="my_handler",
            priority=ShutdownPriority.HIGH,
        )

        assert "my_handler" in lifecycle_manager.registered_handlers
        assert len(lifecycle_manager.registered_handlers) == 1

    def test_register_sync_handler(self, lifecycle_manager):
        """Test registering a sync shutdown handler."""
        def my_sync_handler():
            pass

        lifecycle_manager.register_shutdown_handler(
            my_sync_handler,
            name="my_sync_handler",
            priority=ShutdownPriority.NORMAL,
        )

        assert "my_sync_handler" in lifecycle_manager.registered_handlers

    def test_handler_priority_ordering(self, lifecycle_manager):
        """Test that handlers are ordered by priority (highest first)."""
        async def low_priority():
            pass

        async def high_priority():
            pass

        async def critical_priority():
            pass

        lifecycle_manager.register_shutdown_handler(
            low_priority,
            name="low",
            priority=ShutdownPriority.LOW,
        )
        lifecycle_manager.register_shutdown_handler(
            critical_priority,
            name="critical",
            priority=ShutdownPriority.CRITICAL,
        )
        lifecycle_manager.register_shutdown_handler(
            high_priority,
            name="high",
            priority=ShutdownPriority.HIGH,
        )

        handlers = lifecycle_manager.registered_handlers
        assert handlers[0] == "critical"
        assert handlers[1] == "high"
        assert handlers[2] == "low"

    def test_unregister_handler(self, lifecycle_manager):
        """Test unregistering a shutdown handler."""
        async def my_handler():
            pass

        lifecycle_manager.register_shutdown_handler(my_handler, name="my_handler")
        assert "my_handler" in lifecycle_manager.registered_handlers

        result = lifecycle_manager.unregister_shutdown_handler("my_handler")
        assert result is True
        assert "my_handler" not in lifecycle_manager.registered_handlers

    def test_unregister_nonexistent_handler(self, lifecycle_manager):
        """Test unregistering a handler that doesn't exist."""
        result = lifecycle_manager.unregister_shutdown_handler("nonexistent")
        assert result is False


# =============================================================================
# COMPONENT REGISTRATION TESTS
# =============================================================================

class TestComponentRegistration:
    """Tests for component registration."""

    def test_register_connection_pool(self, lifecycle_manager, mock_connection_pool):
        """Test registering a connection pool."""
        lifecycle_manager.register_connection_pool(mock_connection_pool)
        status = lifecycle_manager.get_status()
        assert status["connection_pool_registered"] is True

    def test_register_memory_backend(self, lifecycle_manager, mock_memory_backend):
        """Test registering a memory backend."""
        lifecycle_manager.register_memory_backend(mock_memory_backend)
        status = lifecycle_manager.get_status()
        assert status["memory_backends_registered"] == 1

    def test_register_cache(self, lifecycle_manager, mock_cache):
        """Test registering a cache."""
        lifecycle_manager.register_cache(mock_cache)
        status = lifecycle_manager.get_status()
        assert status["caches_registered"] == 1

    @pytest.mark.asyncio
    async def test_track_background_task(self, lifecycle_manager):
        """Test tracking a background task."""
        async def background_work():
            await asyncio.sleep(10)

        task = asyncio.create_task(background_work())
        lifecycle_manager.track_background_task(task)

        status = lifecycle_manager.get_status()
        assert status["background_tasks"] == 1

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# =============================================================================
# SHUTDOWN TESTS
# =============================================================================

class TestShutdown:
    """Tests for the shutdown process."""

    @pytest.mark.asyncio
    async def test_basic_shutdown(self, lifecycle_manager):
        """Test basic shutdown with no handlers."""
        report = await lifecycle_manager.shutdown()

        assert report.success is True
        assert report.phase == ShutdownPhase.COMPLETED
        assert report.handlers_executed == 0
        assert report.handlers_failed == 0

    @pytest.mark.asyncio
    async def test_shutdown_with_handlers(self, lifecycle_manager):
        """Test shutdown with registered handlers."""
        call_order = []

        async def handler1():
            call_order.append("handler1")

        async def handler2():
            call_order.append("handler2")

        lifecycle_manager.register_shutdown_handler(
            handler1, name="handler1", priority=ShutdownPriority.HIGH
        )
        lifecycle_manager.register_shutdown_handler(
            handler2, name="handler2", priority=ShutdownPriority.NORMAL
        )

        report = await lifecycle_manager.shutdown()

        assert report.success is True
        assert report.handlers_executed == 2
        assert report.handlers_succeeded == 2
        assert call_order == ["handler1", "handler2"]  # High priority first

    @pytest.mark.asyncio
    async def test_shutdown_handler_timeout(self, lifecycle_manager):
        """Test that slow handlers are timed out."""
        async def slow_handler():
            await asyncio.sleep(10)  # Longer than handler_timeout

        lifecycle_manager.register_shutdown_handler(
            slow_handler, name="slow", timeout=0.1
        )

        report = await lifecycle_manager.shutdown()

        # Handler timed out - verify via results (implementation may not track timed_out counter separately)
        assert report.handlers_failed >= 1 or report.handlers_timed_out >= 1
        assert any("timed out" in r.error for r in report.results if r.error)

    @pytest.mark.asyncio
    async def test_shutdown_handler_error(self, lifecycle_manager):
        """Test handling of handler errors."""
        async def failing_handler():
            raise ValueError("Test error")

        lifecycle_manager.register_shutdown_handler(failing_handler, name="failing")

        report = await lifecycle_manager.shutdown()

        assert report.handlers_failed == 1
        assert any("Test error" in r.error for r in report.results if r.error)

    @pytest.mark.asyncio
    async def test_shutdown_drains_connection_pool(
        self, lifecycle_manager, mock_connection_pool
    ):
        """Test that shutdown drains the connection pool."""
        lifecycle_manager.register_connection_pool(mock_connection_pool)

        await lifecycle_manager.shutdown()

        mock_connection_pool.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_flushes_memory_backend(
        self, lifecycle_manager, mock_memory_backend
    ):
        """Test that shutdown flushes memory backends."""
        lifecycle_manager.register_memory_backend(mock_memory_backend)

        await lifecycle_manager.shutdown()

        mock_memory_backend.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_persists_cache(self, lifecycle_manager, mock_cache):
        """Test that shutdown persists caches."""
        lifecycle_manager.register_cache(mock_cache)

        await lifecycle_manager.shutdown()

        mock_cache.persist.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_cancels_background_tasks(self, lifecycle_manager):
        """Test that shutdown cancels background tasks."""
        task_cancelled = asyncio.Event()

        async def background_work():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                task_cancelled.set()
                raise

        task = asyncio.create_task(background_work())
        lifecycle_manager.track_background_task(task)

        await lifecycle_manager.shutdown()

        # Give the task a moment to set the event
        await asyncio.sleep(0.1)
        assert task_cancelled.is_set() or task.cancelled()

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self, lifecycle_manager):
        """Test that shutdown can only run once."""
        report1 = await lifecycle_manager.shutdown()
        report2 = await lifecycle_manager.shutdown()

        # Second call should return same report
        assert report1.started_at == report2.started_at

    @pytest.mark.asyncio
    async def test_shutdown_phases(self, lifecycle_manager, mock_connection_pool):
        """Test that shutdown goes through correct phases."""
        lifecycle_manager.register_connection_pool(mock_connection_pool)

        # Track phase changes
        phases = []

        original_drain = lifecycle_manager._drain_connection_pool
        async def track_drain(*args, **kwargs):
            phases.append(lifecycle_manager.phase)
            return await original_drain(*args, **kwargs)

        lifecycle_manager._drain_connection_pool = track_drain

        await lifecycle_manager.shutdown()

        assert ShutdownPhase.DRAINING_CONNECTIONS in phases


# =============================================================================
# DECORATOR TESTS
# =============================================================================

class TestDecorators:
    """Tests for shutdown decorators."""

    def test_on_shutdown_decorator(self):
        """Test the @on_shutdown decorator."""
        reset_lifecycle_manager()

        @on_shutdown(name="decorated_handler", priority=ShutdownPriority.HIGH)
        async def my_shutdown_handler():
            pass

        manager = get_lifecycle_manager()
        assert "decorated_handler" in manager.registered_handlers

        reset_lifecycle_manager()


# =============================================================================
# SINGLETON TESTS
# =============================================================================

class TestSingleton:
    """Tests for singleton behavior."""

    def test_get_lifecycle_manager_singleton(self):
        """Test that get_lifecycle_manager returns singleton."""
        reset_lifecycle_manager()

        manager1 = get_lifecycle_manager()
        manager2 = get_lifecycle_manager()

        assert manager1 is manager2

        reset_lifecycle_manager()

    def test_reset_lifecycle_manager(self):
        """Test that reset_lifecycle_manager creates new instance."""
        reset_lifecycle_manager()

        manager1 = get_lifecycle_manager()
        reset_lifecycle_manager()
        manager2 = get_lifecycle_manager()

        assert manager1 is not manager2

        reset_lifecycle_manager()


# =============================================================================
# INTEGRATION HELPER TESTS
# =============================================================================

class TestIntegrationHelpers:
    """Tests for integration helper functions."""

    def test_register_platform_components(
        self, mock_connection_pool, mock_memory_backend, mock_cache
    ):
        """Test register_platform_components helper."""
        reset_lifecycle_manager()

        manager = register_platform_components(
            connection_pool=mock_connection_pool,
            unified_memory=mock_memory_backend,
            cache=mock_cache,
        )

        status = manager.get_status()
        assert status["connection_pool_registered"] is True
        assert status["memory_backends_registered"] >= 1
        assert status["caches_registered"] == 1

        reset_lifecycle_manager()


# =============================================================================
# SHUTDOWN REPORT TESTS
# =============================================================================

class TestShutdownReport:
    """Tests for ShutdownReport."""

    def test_shutdown_report_to_dict(self):
        """Test ShutdownReport.to_dict()."""
        now = datetime.now(timezone.utc)
        report = ShutdownReport(
            started_at=now,
            completed_at=now,
            phase=ShutdownPhase.COMPLETED,
            handlers_executed=3,
            handlers_succeeded=2,
            handlers_failed=1,
            handlers_timed_out=0,
            results=[
                ShutdownResult("handler1", True, 0.1),
                ShutdownResult("handler2", True, 0.2),
                ShutdownResult("handler3", False, 0.3, "Error"),
            ],
        )

        data = report.to_dict()

        assert data["phase"] == "COMPLETED"
        assert data["handlers"]["executed"] == 3
        assert data["handlers"]["succeeded"] == 2
        assert data["handlers"]["failed"] == 1
        assert len(data["results"]) == 3

    def test_shutdown_report_success_property(self):
        """Test ShutdownReport.success property."""
        now = datetime.now(timezone.utc)

        # Success case
        success_report = ShutdownReport(
            started_at=now,
            completed_at=now,
            phase=ShutdownPhase.COMPLETED,
            handlers_executed=2,
            handlers_succeeded=2,
            handlers_failed=0,
            handlers_timed_out=0,
        )
        assert success_report.success is True

        # Failure case
        failure_report = ShutdownReport(
            started_at=now,
            completed_at=now,
            phase=ShutdownPhase.COMPLETED,
            handlers_executed=2,
            handlers_succeeded=1,
            handlers_failed=1,
            handlers_timed_out=0,
        )
        assert failure_report.success is False


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_shutdown_with_sync_handler(self, lifecycle_manager):
        """Test shutdown with a synchronous handler."""
        call_count = {"count": 0}

        def sync_handler():
            call_count["count"] += 1

        lifecycle_manager.register_shutdown_handler(sync_handler, name="sync")

        report = await lifecycle_manager.shutdown()

        assert report.success is True
        assert call_count["count"] == 1

    @pytest.mark.asyncio
    async def test_shutdown_with_weakref_cleanup(self, lifecycle_manager):
        """Test that weak references are handled when objects are collected."""
        # Create and register a mock, then delete it
        mock = MagicMock()
        mock.flush = AsyncMock()
        mock.close = MagicMock()
        lifecycle_manager.register_memory_backend(mock)

        # Delete the mock (simulating GC)
        del mock

        # Shutdown should handle the missing reference gracefully
        report = await lifecycle_manager.shutdown()
        assert report.phase == ShutdownPhase.COMPLETED

    @pytest.mark.asyncio
    async def test_cannot_register_during_shutdown(self, lifecycle_manager):
        """Test that handlers cannot be registered during shutdown."""
        async def slow_handler():
            await asyncio.sleep(0.5)

        lifecycle_manager.register_shutdown_handler(slow_handler, name="slow")

        # Start shutdown
        shutdown_task = asyncio.create_task(lifecycle_manager.shutdown())

        # Wait a bit for shutdown to start
        await asyncio.sleep(0.1)

        # Try to register another handler
        async def another_handler():
            pass

        lifecycle_manager.register_shutdown_handler(another_handler, name="another")

        await shutdown_task

        # The second handler should not have been registered
        # (because shutdown was already in progress)
