"""
Unit Tests for Adaptive Timeout Module
======================================

Tests the adaptive timeout handler functionality including:
- Latency recording and percentile calculation
- Dynamic timeout adjustment
- Warmup behavior
- Bounds enforcement
- Alert callbacks
"""

import asyncio
import pytest
import time
from datetime import datetime
from typing import List, Tuple

# Import the module under test
from core.adaptive_timeout import (
    AdaptiveTimeout,
    TimeoutProfile,
    LatencyStats,
    MeasurementContext,
    get_adaptive_timeout,
    get_adaptive_timeout_sync,
    get_all_timeout_stats,
    reset_all_timeouts,
    AdaptiveTimeoutMiddleware,
    TimeoutAlertManager,
    get_timeout_alert_manager,
    calculate_percentile,
    DEFAULT_PROFILES,
)


class TestCalculatePercentile:
    """Tests for percentile calculation."""

    def test_empty_list(self):
        """Empty list returns 0."""
        assert calculate_percentile([], 50) == 0.0

    def test_single_value(self):
        """Single value returns itself for any percentile."""
        assert calculate_percentile([100.0], 50) == 100.0
        assert calculate_percentile([100.0], 99) == 100.0

    def test_p50_median(self):
        """P50 should return median."""
        values = sorted([10, 20, 30, 40, 50])
        assert calculate_percentile(values, 50) == 30.0

    def test_p99_high_percentile(self):
        """P99 should return near-maximum value."""
        values = sorted(list(range(1, 101)))  # 1-100
        p99 = calculate_percentile(values, 99)
        assert 98 <= p99 <= 100

    def test_interpolation(self):
        """Percentiles should interpolate between values."""
        values = [0.0, 100.0]
        assert calculate_percentile(values, 50) == 50.0
        assert calculate_percentile(values, 25) == 25.0
        assert calculate_percentile(values, 75) == 75.0


class TestTimeoutProfile:
    """Tests for TimeoutProfile configuration."""

    def test_default_values(self):
        """Default profile has sensible values."""
        profile = TimeoutProfile()
        assert profile.min_timeout_ms == 1000.0
        assert profile.max_timeout_ms == 300000.0
        assert profile.default_timeout_ms == 30000.0
        assert profile.percentile_target == 99.0
        assert profile.multiplier == 1.5

    def test_custom_values(self):
        """Custom profile values are respected."""
        profile = TimeoutProfile(
            min_timeout_ms=500.0,
            max_timeout_ms=60000.0,
            default_timeout_ms=5000.0,
            percentile_target=95.0,
            multiplier=2.0,
        )
        assert profile.min_timeout_ms == 500.0
        assert profile.max_timeout_ms == 60000.0
        assert profile.default_timeout_ms == 5000.0
        assert profile.percentile_target == 95.0
        assert profile.multiplier == 2.0


class TestAdaptiveTimeout:
    """Tests for AdaptiveTimeout class."""

    def test_initialization(self):
        """Handler initializes with default timeout."""
        handler = AdaptiveTimeout("test_operation")
        assert handler.operation_name == "test_operation"
        assert handler.get_timeout() == handler.profile.default_timeout_ms

    def test_record_latency(self):
        """Recording latency updates statistics."""
        handler = AdaptiveTimeout("test_operation")
        handler.record_latency(100.0)
        handler.record_latency(200.0)
        handler.record_latency(300.0)

        stats = handler.get_stats()
        assert stats.sample_count == 3
        assert stats.min_ms == 100.0
        assert stats.max_ms == 300.0
        assert stats.mean_ms == 200.0

    def test_warmup_period(self):
        """Timeout is not adjusted during warmup period."""
        profile = TimeoutProfile(
            warmup_samples=5,
            default_timeout_ms=10000.0,
        )
        handler = AdaptiveTimeout("test_operation", profile)

        # Record fewer samples than warmup threshold
        for _ in range(4):
            handler.record_latency(100.0)

        # Should still be at default
        assert not handler.is_warmed_up()
        assert handler.get_timeout() == 10000.0

    def test_adaptive_adjustment(self):
        """Timeout adjusts after warmup based on p99."""
        profile = TimeoutProfile(
            warmup_samples=5,
            default_timeout_ms=10000.0,
            min_timeout_ms=100.0,
            max_timeout_ms=100000.0,
            percentile_target=99.0,
            multiplier=1.5,
            adjustment_threshold=0.05,  # 5% threshold
            cooldown_seconds=0.0,  # No cooldown for testing
        )
        handler = AdaptiveTimeout("test_operation", profile)

        # Record consistent latencies
        for i in range(20):
            handler.record_latency(100.0)

        # Should now be warmed up and adjusted
        assert handler.is_warmed_up()
        # Expected: p99 of [100.0] * 20 = 100.0
        # Timeout = 100.0 * 1.5 = 150.0
        assert handler.get_timeout() == pytest.approx(150.0, rel=0.1)

    def test_min_bound_enforced(self):
        """Timeout does not go below minimum bound."""
        profile = TimeoutProfile(
            warmup_samples=2,
            min_timeout_ms=1000.0,
            cooldown_seconds=0.0,
        )
        handler = AdaptiveTimeout("test_operation", profile)

        # Record very fast latencies
        for _ in range(10):
            handler.record_latency(10.0)

        # Should be clamped to minimum
        assert handler.get_timeout() >= 1000.0

    def test_max_bound_enforced(self):
        """Timeout does not exceed maximum bound."""
        profile = TimeoutProfile(
            warmup_samples=2,
            max_timeout_ms=5000.0,
            cooldown_seconds=0.0,
        )
        handler = AdaptiveTimeout("test_operation", profile)

        # Record very slow latencies
        for _ in range(10):
            handler.record_latency(10000.0)

        # Should be clamped to maximum
        assert handler.get_timeout() <= 5000.0

    def test_reset(self):
        """Reset returns handler to initial state."""
        handler = AdaptiveTimeout("test_operation")

        # Record some latencies
        for _ in range(20):
            handler.record_latency(500.0)

        # Reset
        handler.reset()

        stats = handler.get_stats()
        assert stats.sample_count == 0
        assert stats.timeout_adjustments == 0
        assert handler.get_timeout() == handler.profile.default_timeout_ms

    def test_get_timeout_seconds(self):
        """get_timeout_seconds returns value in seconds."""
        profile = TimeoutProfile(default_timeout_ms=5000.0)
        handler = AdaptiveTimeout("test_operation", profile)
        assert handler.get_timeout_seconds() == 5.0

    def test_alert_callback(self):
        """Alert callback is fired on timeout adjustment."""
        alerts: List[Tuple[str, float, float]] = []

        def callback(name, old, new):
            alerts.append((name, old, new))

        profile = TimeoutProfile(
            warmup_samples=2,
            default_timeout_ms=10000.0,
            cooldown_seconds=0.0,
            adjustment_threshold=0.05,
        )
        handler = AdaptiveTimeout("test_operation", profile)
        handler.add_alert_callback(callback)

        # Record latencies to trigger adjustment
        for _ in range(10):
            handler.record_latency(100.0)

        # Should have triggered at least one alert
        assert len(alerts) >= 1
        assert alerts[0][0] == "test_operation"

    def test_stats_to_dict(self):
        """Stats can be serialized to dictionary."""
        handler = AdaptiveTimeout("test_operation")
        handler.record_latency(100.0)
        handler.record_latency(200.0)

        stats = handler.get_stats()
        stats_dict = stats.to_dict()

        assert "sample_count" in stats_dict
        assert "p99_ms" in stats_dict
        assert "current_timeout_ms" in stats_dict
        assert stats_dict["sample_count"] == 2


class TestMeasureContextManager:
    """Tests for the measure() context manager."""

    @pytest.mark.asyncio
    async def test_measure_records_latency(self):
        """Measure context manager records latency."""
        handler = AdaptiveTimeout("test_operation")

        async with handler.measure() as ctx:
            await asyncio.sleep(0.01)  # 10ms

        assert ctx.latency_ms >= 10.0
        assert handler.get_stats().sample_count == 1

    @pytest.mark.asyncio
    async def test_measure_on_success(self):
        """Latency is recorded on successful operation."""
        handler = AdaptiveTimeout("test_operation")

        async with handler.measure():
            pass  # Successful operation

        assert handler.get_stats().sample_count == 1

    @pytest.mark.asyncio
    async def test_measure_on_exception(self):
        """Latency is recorded even on exception."""
        handler = AdaptiveTimeout("test_operation")

        with pytest.raises(ValueError):
            async with handler.measure():
                raise ValueError("Test error")

        # Latency should still be recorded
        assert handler.get_stats().sample_count == 1


class TestGlobalRegistry:
    """Tests for the global timeout registry."""

    @pytest.mark.asyncio
    async def test_get_adaptive_timeout(self):
        """get_adaptive_timeout creates and caches handlers."""
        reset_all_timeouts()

        handler1 = await get_adaptive_timeout("registry_test_1")
        handler2 = await get_adaptive_timeout("registry_test_1")

        # Should be same instance
        assert handler1 is handler2

    def test_get_adaptive_timeout_sync(self):
        """Sync version creates handlers."""
        reset_all_timeouts()

        handler = get_adaptive_timeout_sync("registry_test_sync")
        assert handler.operation_name == "registry_test_sync"

    def test_get_all_timeout_stats(self):
        """get_all_timeout_stats returns stats for all handlers."""
        reset_all_timeouts()

        handler1 = get_adaptive_timeout_sync("stats_test_1")
        handler2 = get_adaptive_timeout_sync("stats_test_2")

        handler1.record_latency(100.0)
        handler2.record_latency(200.0)

        all_stats = get_all_timeout_stats()

        assert "stats_test_1" in all_stats
        assert "stats_test_2" in all_stats
        assert all_stats["stats_test_1"]["sample_count"] == 1
        assert all_stats["stats_test_2"]["sample_count"] == 1

    def test_reset_all_timeouts(self):
        """reset_all_timeouts resets all handlers."""
        reset_all_timeouts()

        handler = get_adaptive_timeout_sync("reset_test")
        handler.record_latency(100.0)

        count = reset_all_timeouts()
        assert count >= 1

        # Stats should be reset
        assert handler.get_stats().sample_count == 0


class TestAdaptiveTimeoutMiddleware:
    """Tests for AdaptiveTimeoutMiddleware."""

    def test_initialization(self):
        """Middleware initializes correctly."""
        middleware = AdaptiveTimeoutMiddleware("test_adapter")
        assert middleware._adapter_name == "test_adapter"

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self):
        """Execute wraps function with timeout tracking."""
        middleware = AdaptiveTimeoutMiddleware("test_adapter")

        async def my_operation():
            await asyncio.sleep(0.01)
            return "result"

        result = await middleware.execute("my_method", my_operation)
        assert result == "result"

        stats = middleware.get_all_stats()
        assert "test_adapter_my_method" in stats
        assert stats["test_adapter_my_method"]["sample_count"] == 1

    @pytest.mark.asyncio
    async def test_execute_timeout_override(self):
        """Timeout override is respected."""
        middleware = AdaptiveTimeoutMiddleware("test_adapter")

        async def slow_operation():
            await asyncio.sleep(10.0)  # Would timeout

        with pytest.raises(asyncio.TimeoutError):
            await middleware.execute(
                "slow_method",
                slow_operation,
                timeout_override=0.01  # 10ms timeout
            )

    def test_wrap_decorator(self):
        """wrap decorator works correctly."""
        middleware = AdaptiveTimeoutMiddleware("test_adapter")

        @middleware.wrap("decorated_method")
        async def decorated_function():
            return "decorated"

        assert decorated_function.__name__ == "decorated_function"


class TestDefaultProfiles:
    """Tests for default timeout profiles."""

    def test_exa_profiles_exist(self):
        """Exa adapter profiles are defined."""
        assert "exa_search" in DEFAULT_PROFILES
        assert "exa_fast" in DEFAULT_PROFILES
        assert "exa_deep" in DEFAULT_PROFILES

    def test_research_adapter_profiles(self):
        """Research adapter profiles have appropriate defaults."""
        assert "tavily_search" in DEFAULT_PROFILES
        assert "perplexity_research" in DEFAULT_PROFILES
        assert "jina_search" in DEFAULT_PROFILES
        assert "firecrawl_scrape" in DEFAULT_PROFILES

    def test_memory_profiles(self):
        """Memory operation profiles are defined."""
        assert "memory_store" in DEFAULT_PROFILES
        assert "memory_retrieve" in DEFAULT_PROFILES
        assert "memory_search" in DEFAULT_PROFILES

    def test_default_fallback(self):
        """Default fallback profile exists."""
        assert "_default" in DEFAULT_PROFILES


class TestTimeoutAlertManager:
    """Tests for TimeoutAlertManager."""

    def test_alert_callback_registration(self):
        """Alert callbacks can be registered."""
        manager = TimeoutAlertManager()
        alerts = []

        manager.register_callback(lambda alert: alerts.append(alert))
        manager.on_timeout_adjusted("test_op", 1000.0, 2000.0)

        assert len(alerts) == 1
        assert alerts[0]["operation"] == "test_op"
        assert alerts[0]["direction"] == "increased"

    def test_severity_levels(self):
        """Alert severity is calculated correctly."""
        manager = TimeoutAlertManager()
        alerts = []

        manager.register_callback(lambda alert: alerts.append(alert))

        # Small change - info
        manager.on_timeout_adjusted("test", 1000.0, 1100.0)
        assert alerts[-1]["severity"] == "info"

        # 25% change - warning
        manager.on_timeout_adjusted("test", 1000.0, 1300.0)
        assert alerts[-1]["severity"] == "warning"

        # 50% change - error
        manager.on_timeout_adjusted("test", 1000.0, 1600.0)
        assert alerts[-1]["severity"] == "error"

        # 100% change - critical
        manager.on_timeout_adjusted("test", 1000.0, 2100.0)
        assert alerts[-1]["severity"] == "critical"

    def test_global_alert_manager(self):
        """Global alert manager is accessible."""
        manager = get_timeout_alert_manager()
        assert manager is not None
        assert isinstance(manager, TimeoutAlertManager)


class TestLatencyStats:
    """Tests for LatencyStats dataclass."""

    def test_default_values(self):
        """LatencyStats has sensible defaults."""
        stats = LatencyStats()
        assert stats.sample_count == 0
        assert stats.min_ms == 0.0
        assert stats.is_warmed_up == False

    def test_to_dict_serialization(self):
        """LatencyStats serializes to dict correctly."""
        stats = LatencyStats(
            sample_count=100,
            min_ms=10.0,
            max_ms=1000.0,
            mean_ms=100.0,
            p50_ms=50.0,
            p90_ms=200.0,
            p95_ms=400.0,
            p99_ms=800.0,
            current_timeout_ms=1200.0,
            is_warmed_up=True,
        )

        result = stats.to_dict()

        assert result["sample_count"] == 100
        assert result["min_ms"] == 10.0
        assert result["p99_ms"] == 800.0
        assert result["is_warmed_up"] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
