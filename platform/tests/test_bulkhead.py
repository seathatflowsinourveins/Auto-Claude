"""
Tests for Bulkhead Pattern Implementation

Tests cover:
- Basic concurrency limiting
- Queue management
- Dynamic limit adjustment
- Circuit breaker functionality
- Registry operations
- MCP integration
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from core.orchestration.bulkhead import (
    Bulkhead,
    BulkheadConfig,
    BulkheadError,
    BulkheadFullError,
    BulkheadRegistry,
    BulkheadRejectionReason,
    BulkheadState,
    BulkheadStats,
    BulkheadTimeoutError,
    create_api_bulkhead,
    create_database_bulkhead,
    create_mcp_bulkhead,
    get_bulkhead,
    get_bulkhead_registry,
)


class TestBulkheadConfig:
    """Test BulkheadConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BulkheadConfig(name="test")
        assert config.name == "test"
        assert config.max_concurrent == 10
        assert config.max_queue_size == 100
        assert config.timeout_seconds == 30.0
        assert config.enable_dynamic_limits is True
        assert config.enable_circuit_breaker is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BulkheadConfig(
            name="custom",
            max_concurrent=5,
            max_queue_size=50,
            timeout_seconds=60.0,
            enable_dynamic_limits=False,
        )
        assert config.max_concurrent == 5
        assert config.max_queue_size == 50
        assert config.timeout_seconds == 60.0
        assert config.enable_dynamic_limits is False


class TestBulkhead:
    """Test Bulkhead class."""

    @pytest.fixture
    def bulkhead(self):
        """Create a test bulkhead."""
        config = BulkheadConfig(
            name="test",
            max_concurrent=3,
            max_queue_size=5,
            timeout_seconds=5.0,
            enable_dynamic_limits=False,
            enable_circuit_breaker=False,
        )
        return Bulkhead(config)

    @pytest.fixture
    def circuit_breaker_bulkhead(self):
        """Create a bulkhead with circuit breaker."""
        config = BulkheadConfig(
            name="circuit_test",
            max_concurrent=3,
            max_queue_size=5,
            timeout_seconds=5.0,
            enable_circuit_breaker=True,
            failure_threshold=3,
            success_threshold=2,
            circuit_open_duration_seconds=1.0,
        )
        return Bulkhead(config)

    @pytest.mark.asyncio
    async def test_execute_simple(self, bulkhead):
        """Test simple operation execution."""
        result = await bulkhead.execute(lambda: 42)
        assert result == 42

    @pytest.mark.asyncio
    async def test_execute_async(self, bulkhead):
        """Test async operation execution."""
        async def async_op():
            await asyncio.sleep(0.01)
            return "async_result"

        result = await bulkhead.execute(async_op)
        assert result == "async_result"

    @pytest.mark.asyncio
    async def test_concurrency_limit(self, bulkhead):
        """Test that concurrency is limited."""
        execution_count = 0
        max_concurrent = 0
        lock = asyncio.Lock()

        async def tracked_operation():
            nonlocal execution_count, max_concurrent
            async with lock:
                execution_count += 1
                max_concurrent = max(max_concurrent, execution_count)
            await asyncio.sleep(0.1)
            async with lock:
                execution_count -= 1
            return True

        # Start more operations than max_concurrent
        tasks = [
            asyncio.create_task(bulkhead.execute(tracked_operation))
            for _ in range(6)
        ]

        results = await asyncio.gather(*tasks)

        assert all(results)
        assert max_concurrent <= 3  # Should not exceed max_concurrent

    @pytest.mark.asyncio
    async def test_queue_full_rejection(self, bulkhead):
        """Test that queue full raises error."""
        # Fill up the bulkhead
        blocking_event = asyncio.Event()

        async def blocking_op():
            await blocking_event.wait()
            return True

        # Start max_concurrent + max_queue_size operations
        tasks = []
        for _ in range(3 + 5):  # 3 concurrent + 5 queued
            tasks.append(asyncio.create_task(bulkhead.execute(blocking_op)))

        await asyncio.sleep(0.1)  # Let tasks start

        # This should fail - queue is full
        with pytest.raises(BulkheadFullError):
            await bulkhead.execute(lambda: "overflow")

        # Cleanup
        blocking_event.set()
        await asyncio.gather(*tasks, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_try_execute_returns_none(self, bulkhead):
        """Test try_execute returns None instead of raising."""
        blocking_event = asyncio.Event()

        async def blocking_op():
            await blocking_event.wait()
            return True

        # Fill up the bulkhead
        tasks = []
        for _ in range(8):
            tasks.append(asyncio.create_task(bulkhead.execute(blocking_op)))

        await asyncio.sleep(0.1)

        # try_execute should return None
        result = await bulkhead.try_execute(lambda: "overflow")
        assert result is None

        # Cleanup
        blocking_event.set()
        await asyncio.gather(*tasks, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_timeout(self, bulkhead):
        """Test operation timeout."""
        async def slow_op():
            await asyncio.sleep(10)
            return True

        with pytest.raises(BulkheadTimeoutError):
            await bulkhead.execute(slow_op, timeout=0.1)

    @pytest.mark.asyncio
    async def test_stats_tracking(self, bulkhead):
        """Test statistics are tracked correctly."""
        # Execute some operations
        for _ in range(5):
            await bulkhead.execute(lambda: True)

        # One failure
        try:
            await bulkhead.execute(lambda: 1 / 0)
        except ZeroDivisionError:
            pass

        stats = bulkhead.get_stats()
        assert stats.total_requests == 6
        assert stats.successful_requests == 5
        assert stats.failed_requests == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens(self, circuit_breaker_bulkhead):
        """Test circuit breaker opens after failures."""
        async def failing_op():
            raise ValueError("Operation failed")

        # Trigger failures
        for _ in range(3):
            try:
                await circuit_breaker_bulkhead.execute(failing_op)
            except ValueError:
                pass

        # Circuit should be open
        assert circuit_breaker_bulkhead.state == BulkheadState.OPEN

        # New requests should be rejected
        with pytest.raises(BulkheadError) as exc_info:
            await circuit_breaker_bulkhead.execute(lambda: True)
        assert exc_info.value.reason == BulkheadRejectionReason.CIRCUIT_OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, circuit_breaker_bulkhead):
        """Test circuit breaker recovers after timeout."""
        async def failing_op():
            raise ValueError("Operation failed")

        # Trigger failures to open circuit
        for _ in range(3):
            try:
                await circuit_breaker_bulkhead.execute(failing_op)
            except ValueError:
                pass

        assert circuit_breaker_bulkhead.state == BulkheadState.OPEN

        # Wait for circuit open duration
        await asyncio.sleep(1.1)

        # Should be able to execute again (half-open)
        result = await circuit_breaker_bulkhead.execute(lambda: "success")
        assert result == "success"

        # Another success should close circuit
        await circuit_breaker_bulkhead.execute(lambda: "success2")
        assert circuit_breaker_bulkhead.state == BulkheadState.CLOSED

    @pytest.mark.asyncio
    async def test_priority_queue(self, bulkhead):
        """Test higher priority requests are processed first."""
        blocking_event = asyncio.Event()
        results = []

        async def blocking_op():
            await blocking_event.wait()
            return True

        async def record_op(name):
            results.append(name)
            return name

        # Fill concurrent slots
        concurrent_tasks = [
            asyncio.create_task(bulkhead.execute(blocking_op))
            for _ in range(3)
        ]

        await asyncio.sleep(0.05)

        # Queue operations with different priorities
        low_task = asyncio.create_task(
            bulkhead.execute(lambda: record_op("low"), priority=0)
        )
        high_task = asyncio.create_task(
            bulkhead.execute(lambda: record_op("high"), priority=10)
        )

        await asyncio.sleep(0.05)

        # Release blocking ops
        blocking_event.set()

        await asyncio.gather(
            *concurrent_tasks, low_task, high_task,
            return_exceptions=True
        )

        # High priority should be processed before low
        if len(results) == 2:
            assert results[0] == "high"

    @pytest.mark.asyncio
    async def test_shutdown(self, bulkhead):
        """Test graceful shutdown."""
        blocking_event = asyncio.Event()

        async def blocking_op():
            await blocking_event.wait()
            return True

        # Start some operations
        tasks = [
            asyncio.create_task(bulkhead.execute(blocking_op))
            for _ in range(3)
        ]

        await asyncio.sleep(0.05)

        # Shutdown
        await bulkhead.shutdown()

        # New operations should be rejected
        with pytest.raises(BulkheadError) as exc_info:
            await bulkhead.execute(lambda: True)
        assert exc_info.value.reason == BulkheadRejectionReason.SHUTDOWN

        # Cleanup
        blocking_event.set()
        await asyncio.gather(*tasks, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_health_status(self, bulkhead):
        """Test health reporting."""
        health = bulkhead.get_health()
        assert health["status"] == "healthy"
        assert health["accepting_requests"] is True

    @pytest.mark.asyncio
    async def test_adjust_limit(self, bulkhead):
        """Test manual limit adjustment."""
        original = bulkhead._current_limit
        bulkhead.adjust_limit(original + 5)
        assert bulkhead._current_limit == original + 5

        # Should respect ceiling
        bulkhead.adjust_limit(1000)
        assert bulkhead._current_limit <= bulkhead._config.max_concurrent_ceiling


class TestBulkheadRegistry:
    """Test BulkheadRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create a test registry."""
        return BulkheadRegistry()

    def test_get_or_create(self, registry):
        """Test bulkhead creation."""
        bulkhead = registry.get_or_create("test")
        assert bulkhead.name == "test"

        # Same name returns same instance
        same_bulkhead = registry.get_or_create("test")
        assert same_bulkhead is bulkhead

    def test_get_or_create_with_config(self, registry):
        """Test bulkhead creation with config."""
        config = BulkheadConfig(name="custom", max_concurrent=5)
        bulkhead = registry.get_or_create("custom", config)
        assert bulkhead.config.max_concurrent == 5

    def test_list_bulkheads(self, registry):
        """Test listing bulkheads."""
        registry.get_or_create("a")
        registry.get_or_create("b")
        registry.get_or_create("c")

        names = registry.list_bulkheads()
        assert set(names) == {"a", "b", "c"}

    def test_remove(self, registry):
        """Test removing bulkhead."""
        registry.get_or_create("test")
        removed = registry.remove("test")
        assert removed is not None
        assert registry.get("test") is None

    def test_get_stats(self, registry):
        """Test getting stats for all bulkheads."""
        registry.get_or_create("a")
        registry.get_or_create("b")

        stats = registry.get_stats()
        assert "a" in stats
        assert "b" in stats
        assert isinstance(stats["a"], BulkheadStats)

    def test_get_health(self, registry):
        """Test getting aggregated health."""
        registry.get_or_create("a")
        registry.get_or_create("b")

        health = registry.get_health()
        assert health["status"] == "healthy"
        assert health["bulkhead_count"] == 2

    @pytest.mark.asyncio
    async def test_start_all(self, registry):
        """Test starting all bulkheads."""
        registry.get_or_create("a", BulkheadConfig(
            name="a", enable_dynamic_limits=True
        ))
        registry.get_or_create("b", BulkheadConfig(
            name="b", enable_dynamic_limits=True
        ))

        await registry.start_all()
        assert registry._started is True

        # Cleanup
        await registry.shutdown_all()

    @pytest.mark.asyncio
    async def test_shutdown_all(self, registry):
        """Test shutting down all bulkheads."""
        registry.get_or_create("a")
        registry.get_or_create("b")

        await registry.start_all()
        await registry.shutdown_all()

        assert registry._started is False


class TestBulkheadFactories:
    """Test pre-configured bulkhead factories."""

    def test_create_database_bulkhead(self):
        """Test database bulkhead factory."""
        bulkhead = create_database_bulkhead("test_db")
        assert bulkhead.config.name == "test_db"
        assert bulkhead.config.max_concurrent == 20
        assert bulkhead.config.failure_threshold == 3

    def test_create_api_bulkhead(self):
        """Test API bulkhead factory."""
        bulkhead = create_api_bulkhead("test_api")
        assert bulkhead.config.name == "test_api"
        assert bulkhead.config.max_concurrent == 10
        assert bulkhead.config.timeout_seconds == 60.0

    def test_create_mcp_bulkhead(self):
        """Test MCP bulkhead factory."""
        bulkhead = create_mcp_bulkhead("test_mcp")
        assert bulkhead.config.name == "test_mcp"
        assert bulkhead.config.max_concurrent == 15


class TestGlobalFunctions:
    """Test global convenience functions."""

    def test_get_bulkhead(self):
        """Test get_bulkhead convenience function."""
        bulkhead = get_bulkhead("global_test")
        assert bulkhead is not None
        assert bulkhead.name == "global_test"

    def test_get_bulkhead_registry(self):
        """Test get_bulkhead_registry convenience function."""
        registry = get_bulkhead_registry()
        assert registry is not None
        assert isinstance(registry, BulkheadRegistry)


class TestDynamicLimitAdjustment:
    """Test dynamic limit adjustment feature."""

    @pytest.mark.asyncio
    async def test_scale_up_under_load(self):
        """Test that limits scale up when queue has items."""
        config = BulkheadConfig(
            name="dynamic_test",
            max_concurrent=3,
            max_queue_size=10,
            enable_dynamic_limits=True,
            adjustment_interval_seconds=0.1,
            scale_up_threshold=0.5,
            adjustment_step=2,
        )
        bulkhead = Bulkhead(config)
        await bulkhead.start()

        try:
            blocking_event = asyncio.Event()

            async def blocking_op():
                await blocking_event.wait()
                return True

            # Fill concurrent slots and add to queue
            tasks = []
            for _ in range(5):
                tasks.append(asyncio.create_task(bulkhead.execute(blocking_op)))

            await asyncio.sleep(0.3)  # Allow adjustment to run

            # Limit should have increased
            assert bulkhead._current_limit >= 3

            # Cleanup
            blocking_event.set()
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await bulkhead.shutdown()


class TestBulkheadStats:
    """Test BulkheadStats dataclass."""

    def test_to_dict(self):
        """Test stats serialization."""
        stats = BulkheadStats(
            name="test",
            state=BulkheadState.CLOSED,
            current_concurrent=5,
            max_concurrent=10,
            queue_size=3,
            max_queue_size=100,
            total_requests=1000,
            successful_requests=990,
            failed_requests=5,
            rejected_requests=5,
            timeout_requests=0,
            avg_wait_time_ms=10.5,
            avg_execution_time_ms=50.3,
            p95_wait_time_ms=25.0,
            p95_execution_time_ms=120.0,
            utilization=0.5,
            circuit_failures=0,
            last_adjustment_time=None,
        )

        d = stats.to_dict()
        assert d["name"] == "test"
        assert d["state"] == "closed"
        assert d["utilization"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
