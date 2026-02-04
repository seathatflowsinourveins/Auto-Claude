"""
Unit Tests for Cache Warming System - V39 Architecture

Tests the cache warming functionality including:
- WarmingStrategy registration and execution
- CacheWarmer background loop
- Priority-based warming order
- Usage pattern tracking for predictive warming
- Pre-built warming strategies
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.orchestration.cache_warming import (
    CacheWarmer,
    WarmingStrategy,
    WarmingResult,
    WarmingPriority,
    WarmingStatus,
    UsagePattern,
    create_default_warmer,
    create_minimal_warmer,
    get_cache_warmer,
    set_cache_warmer,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def warmer():
    """Create a fresh CacheWarmer for each test."""
    return CacheWarmer(max_concurrent=2, background_interval=0.1)


@pytest.fixture
def mock_warmup_fn():
    """Create a mock async warmup function."""
    return AsyncMock()


@pytest.fixture
def slow_warmup_fn():
    """Create a slow warmup function for timeout tests."""
    async def slow():
        await asyncio.sleep(10.0)
    return slow


@pytest.fixture
def failing_warmup_fn():
    """Create a failing warmup function."""
    async def fail():
        raise ValueError("Intentional failure")
    return fail


# =============================================================================
# WarmingStrategy Tests
# =============================================================================

class TestWarmingStrategy:
    """Tests for WarmingStrategy dataclass."""

    def test_strategy_creation(self, mock_warmup_fn):
        """Test basic strategy creation."""
        strategy = WarmingStrategy(
            name="test_strategy",
            priority=WarmingPriority.HIGH,
            warmup_fn=mock_warmup_fn,
            interval_seconds=300.0,
        )

        assert strategy.name == "test_strategy"
        assert strategy.priority == WarmingPriority.HIGH
        assert strategy.interval_seconds == 300.0
        assert strategy.enabled is True
        assert strategy.last_warmed is None
        assert strategy.success_count == 0
        assert strategy.failure_count == 0

    def test_needs_warming_never_warmed(self, mock_warmup_fn):
        """Test needs_warming when never warmed."""
        strategy = WarmingStrategy(
            name="test",
            priority=1,
            warmup_fn=mock_warmup_fn,
        )

        assert strategy.needs_warming is True

    def test_needs_warming_recently_warmed(self, mock_warmup_fn):
        """Test needs_warming when recently warmed."""
        strategy = WarmingStrategy(
            name="test",
            priority=1,
            warmup_fn=mock_warmup_fn,
            interval_seconds=300.0,
            last_warmed=datetime.now(timezone.utc),
        )

        assert strategy.needs_warming is False

    def test_needs_warming_expired(self, mock_warmup_fn):
        """Test needs_warming when interval elapsed."""
        strategy = WarmingStrategy(
            name="test",
            priority=1,
            warmup_fn=mock_warmup_fn,
            interval_seconds=1.0,
            last_warmed=datetime.now(timezone.utc) - timedelta(seconds=2),
        )

        assert strategy.needs_warming is True

    def test_needs_warming_disabled(self, mock_warmup_fn):
        """Test needs_warming when disabled."""
        strategy = WarmingStrategy(
            name="test",
            priority=1,
            warmup_fn=mock_warmup_fn,
            enabled=False,
        )

        assert strategy.needs_warming is False

    def test_needs_warming_one_time_done(self, mock_warmup_fn):
        """Test one-time strategy doesn't need re-warming."""
        strategy = WarmingStrategy(
            name="test",
            priority=1,
            warmup_fn=mock_warmup_fn,
            interval_seconds=0,  # One-time only
            last_warmed=datetime.now(timezone.utc) - timedelta(hours=1),
        )

        assert strategy.needs_warming is False

    def test_avg_duration_empty(self, mock_warmup_fn):
        """Test avg_duration_ms with no executions."""
        strategy = WarmingStrategy(
            name="test",
            priority=1,
            warmup_fn=mock_warmup_fn,
        )

        assert strategy.avg_duration_ms == 0.0

    def test_success_rate(self, mock_warmup_fn):
        """Test success rate calculation."""
        strategy = WarmingStrategy(
            name="test",
            priority=1,
            warmup_fn=mock_warmup_fn,
        )
        strategy.success_count = 7
        strategy.failure_count = 3
        strategy.total_duration_ms = 1000.0

        assert strategy.success_rate == 70.0
        assert strategy.avg_duration_ms == 100.0


# =============================================================================
# UsagePattern Tests
# =============================================================================

class TestUsagePattern:
    """Tests for UsagePattern dataclass."""

    def test_pattern_creation(self):
        """Test basic pattern creation."""
        pattern = UsagePattern(key="test_key")

        assert pattern.key == "test_key"
        assert pattern.access_count == 0
        assert pattern.last_accessed is None
        assert pattern.avg_interval_seconds == 0.0

    def test_record_single_access(self):
        """Test recording a single access."""
        pattern = UsagePattern(key="test")
        pattern.record_access()

        assert pattern.access_count == 1
        assert pattern.last_accessed is not None

    def test_record_multiple_accesses(self):
        """Test recording multiple accesses."""
        pattern = UsagePattern(key="test")
        pattern.record_access()
        pattern.record_access()
        pattern.record_access()

        assert pattern.access_count == 3

    def test_predicted_next_access_no_data(self):
        """Test prediction with no data."""
        pattern = UsagePattern(key="test")

        assert pattern.predicted_next_access is None

    def test_predicted_next_access_with_data(self):
        """Test prediction with access history."""
        pattern = UsagePattern(key="test")
        pattern.last_accessed = datetime.now(timezone.utc)
        pattern.avg_interval_seconds = 60.0

        predicted = pattern.predicted_next_access
        assert predicted is not None


# =============================================================================
# CacheWarmer Tests
# =============================================================================

class TestCacheWarmer:
    """Tests for CacheWarmer class."""

    def test_warmer_creation(self, warmer):
        """Test CacheWarmer initialization."""
        assert warmer._running is False
        assert len(warmer._strategies) == 0
        assert warmer._max_concurrent == 2

    def test_register_strategy(self, warmer, mock_warmup_fn):
        """Test strategy registration."""
        strategy = WarmingStrategy(
            name="test_strategy",
            priority=WarmingPriority.HIGH,
            warmup_fn=mock_warmup_fn,
        )

        warmer.register(strategy)

        assert "test_strategy" in warmer._strategies
        assert warmer.get_strategy("test_strategy") == strategy

    def test_unregister_strategy(self, warmer, mock_warmup_fn):
        """Test strategy unregistration."""
        strategy = WarmingStrategy(
            name="test_strategy",
            priority=1,
            warmup_fn=mock_warmup_fn,
        )

        warmer.register(strategy)
        result = warmer.unregister("test_strategy")

        assert result is True
        assert "test_strategy" not in warmer._strategies

    def test_unregister_nonexistent(self, warmer):
        """Test unregistering non-existent strategy."""
        result = warmer.unregister("nonexistent")
        assert result is False

    def test_list_strategies_sorted(self, warmer, mock_warmup_fn):
        """Test strategies are listed in priority order."""
        warmer.register(WarmingStrategy(
            name="low",
            priority=WarmingPriority.LOW,
            warmup_fn=mock_warmup_fn,
        ))
        warmer.register(WarmingStrategy(
            name="critical",
            priority=WarmingPriority.CRITICAL,
            warmup_fn=mock_warmup_fn,
        ))
        warmer.register(WarmingStrategy(
            name="normal",
            priority=WarmingPriority.NORMAL,
            warmup_fn=mock_warmup_fn,
        ))

        strategies = warmer.list_strategies()

        assert strategies[0].name == "critical"
        assert strategies[1].name == "normal"
        assert strategies[2].name == "low"

    @pytest.mark.asyncio
    async def test_warm_strategy_success(self, warmer, mock_warmup_fn):
        """Test successful strategy warming."""
        strategy = WarmingStrategy(
            name="test_strategy",
            priority=1,
            warmup_fn=mock_warmup_fn,
        )
        warmer.register(strategy)

        result = await warmer.warm_strategy("test_strategy")

        assert result.status == WarmingStatus.SUCCESS
        assert result.strategy_name == "test_strategy"
        # Mock functions return instantly, so duration may be 0 or very small
        assert result.duration_ms >= 0
        assert strategy.success_count == 1
        assert strategy.last_warmed is not None
        mock_warmup_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_warm_strategy_not_found(self, warmer):
        """Test warming non-existent strategy."""
        result = await warmer.warm_strategy("nonexistent")

        assert result.status == WarmingStatus.FAILED
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_warm_strategy_failure(self, warmer, failing_warmup_fn):
        """Test failed strategy warming."""
        strategy = WarmingStrategy(
            name="failing_strategy",
            priority=1,
            warmup_fn=failing_warmup_fn,
        )
        warmer.register(strategy)

        result = await warmer.warm_strategy("failing_strategy")

        assert result.status == WarmingStatus.FAILED
        assert "Intentional failure" in result.error
        assert strategy.failure_count == 1

    @pytest.mark.asyncio
    async def test_warm_strategy_timeout(self, warmer, slow_warmup_fn):
        """Test strategy timeout."""
        strategy = WarmingStrategy(
            name="slow_strategy",
            priority=1,
            warmup_fn=slow_warmup_fn,
            timeout_seconds=0.1,
        )
        warmer.register(strategy)

        result = await warmer.warm_strategy("slow_strategy")

        assert result.status == WarmingStatus.TIMEOUT
        assert strategy.failure_count == 1

    @pytest.mark.asyncio
    async def test_warm_all(self, warmer, mock_warmup_fn):
        """Test warming all strategies."""
        for i in range(3):
            warmer.register(WarmingStrategy(
                name=f"strategy_{i}",
                priority=i + 1,
                warmup_fn=mock_warmup_fn,
            ))

        results = await warmer.warm_all()

        assert len(results) == 3
        assert all(r.status == WarmingStatus.SUCCESS for r in results.values())

    @pytest.mark.asyncio
    async def test_warm_all_force(self, warmer, mock_warmup_fn):
        """Test forced warming of all strategies."""
        strategy = WarmingStrategy(
            name="recently_warmed",
            priority=1,
            warmup_fn=mock_warmup_fn,
            interval_seconds=300.0,
            last_warmed=datetime.now(timezone.utc),  # Just warmed
        )
        warmer.register(strategy)

        # Without force, should skip
        results_no_force = await warmer.warm_all(force=False)
        assert len(results_no_force) == 0

        # With force, should warm
        results_force = await warmer.warm_all(force=True)
        assert len(results_force) == 1

    @pytest.mark.asyncio
    async def test_warm_with_dependencies(self, warmer, mock_warmup_fn):
        """Test warming strategies with dependencies."""
        warmer.register(WarmingStrategy(
            name="dependency",
            priority=WarmingPriority.CRITICAL,
            warmup_fn=mock_warmup_fn,
        ))
        warmer.register(WarmingStrategy(
            name="dependent",
            priority=WarmingPriority.HIGH,
            warmup_fn=mock_warmup_fn,
            dependencies=["dependency"],
        ))

        results = await warmer.warm_all()

        # Both should succeed
        assert results["dependency"].status == WarmingStatus.SUCCESS
        assert results["dependent"].status == WarmingStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_background_warming_start_stop(self, warmer):
        """Test starting and stopping background warming."""
        await warmer.start_background_warming()
        assert warmer._running is True
        assert warmer._background_task is not None

        await warmer.stop()
        assert warmer._running is False
        assert warmer._background_task is None

    @pytest.mark.asyncio
    async def test_background_warming_executes(self, warmer, mock_warmup_fn):
        """Test that background warming executes strategies."""
        warmer.register(WarmingStrategy(
            name="bg_test",
            priority=1,
            warmup_fn=mock_warmup_fn,
            interval_seconds=0.1,
        ))

        await warmer.start_background_warming()
        await asyncio.sleep(0.3)  # Let a few cycles run
        await warmer.stop()

        # Should have been called at least once
        assert mock_warmup_fn.call_count >= 1

    def test_record_usage(self, warmer):
        """Test usage pattern recording."""
        warmer.record_usage("test_key")
        warmer.record_usage("test_key")
        warmer.record_usage("test_key")

        assert "test_key" in warmer._usage_patterns
        assert warmer._usage_patterns["test_key"].access_count == 3

    def test_get_stats(self, warmer, mock_warmup_fn):
        """Test statistics gathering."""
        warmer.register(WarmingStrategy(
            name="test",
            priority=1,
            warmup_fn=mock_warmup_fn,
        ))

        stats = warmer.get_stats()

        assert stats["running"] is False
        assert stats["total_strategies"] == 1
        assert stats["strategies_warmed"] == 0
        assert "test" in stats["strategies"]


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_default_warmer(self):
        """Test default warmer creation."""
        warmer = create_default_warmer(
            mcp_servers=["exa", "jina"],
            common_queries=["test query"],
            session_id="test_session",
        )

        assert len(warmer._strategies) >= 1

    def test_create_minimal_warmer(self):
        """Test minimal warmer creation."""
        warmer = create_minimal_warmer(mcp_servers=["exa"])

        assert len(warmer._strategies) == 1
        assert "mcp_connections" in warmer._strategies


# =============================================================================
# Singleton Tests
# =============================================================================

class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_cache_warmer(self):
        """Test getting global warmer."""
        warmer1 = get_cache_warmer()
        warmer2 = get_cache_warmer()

        assert warmer1 is warmer2

    def test_set_cache_warmer(self):
        """Test setting global warmer."""
        custom_warmer = CacheWarmer()
        set_cache_warmer(custom_warmer)

        assert get_cache_warmer() is custom_warmer


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for cache warming with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_concurrent_warming_limit(self, warmer, mock_warmup_fn):
        """Test that concurrent warming is limited."""
        concurrent_count = 0
        max_concurrent = 0

        async def track_concurrent():
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.1)
            concurrent_count -= 1

        for i in range(5):
            warmer.register(WarmingStrategy(
                name=f"strategy_{i}",
                priority=WarmingPriority.NORMAL,
                warmup_fn=track_concurrent,
            ))

        await warmer.warm_all()

        assert max_concurrent <= warmer._max_concurrent

    @pytest.mark.asyncio
    async def test_priority_execution_order(self, warmer):
        """Test strategies execute in priority order within groups."""
        execution_order: List[str] = []

        async def record_execution(name: str):
            execution_order.append(name)

        warmer.register(WarmingStrategy(
            name="low",
            priority=WarmingPriority.LOW,
            warmup_fn=lambda: record_execution("low"),
        ))
        warmer.register(WarmingStrategy(
            name="critical",
            priority=WarmingPriority.CRITICAL,
            warmup_fn=lambda: record_execution("critical"),
        ))
        warmer.register(WarmingStrategy(
            name="high",
            priority=WarmingPriority.HIGH,
            warmup_fn=lambda: record_execution("high"),
        ))

        await warmer.warm_all()

        # Critical should be before high, high before low
        assert execution_order.index("critical") < execution_order.index("high")
        assert execution_order.index("high") < execution_order.index("low")

    @pytest.mark.asyncio
    async def test_results_history(self, warmer, mock_warmup_fn):
        """Test results are recorded in history."""
        warmer.register(WarmingStrategy(
            name="test",
            priority=1,
            warmup_fn=mock_warmup_fn,
        ))

        await warmer.warm_strategy("test")
        await warmer.warm_strategy("test")

        assert len(warmer._results_history) == 2
