"""
Tests for Error Recovery Module

Tests cover:
- Error classification (transient vs permanent)
- Retry strategy with backoff
- Fallback chain execution
- Partial result handling
- Error aggregation and reporting
- Memory fallback
- Cache recovery
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

# Import from the module under test
from core.error_recovery import (
    ErrorType,
    RecoveryAction,
    ClassifiedError,
    ErrorClassifier,
    RetryStrategy,
    FallbackChainStrategy,
    CacheRecoveryStrategy,
    DegradedResponseStrategy,
    MemoryFallbackStrategy,
    PartialResult,
    PartialResultHandler,
    ErrorPattern,
    ErrorAggregator,
    RecoveryConfig,
    RecoveryResult,
    ErrorRecovery,
    with_recovery,
    create_recovery,
    get_error_recovery,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def classifier():
    """Create an error classifier."""
    return ErrorClassifier()


@pytest.fixture
def config():
    """Create a recovery config."""
    return RecoveryConfig(
        max_retries=3,
        base_delay=0.01,  # Fast for tests
        max_delay=0.1,
        enable_fallback=True,
        enable_cache_recovery=True,
        enable_partial_results=True,
    )


@pytest.fixture
def recovery(config):
    """Create an error recovery instance."""
    return ErrorRecovery(config=config)


@pytest.fixture
def aggregator():
    """Create an error aggregator."""
    return ErrorAggregator(max_errors=100, pattern_window_minutes=60)


# =============================================================================
# ERROR CLASSIFICATION TESTS
# =============================================================================

class TestErrorClassifier:
    """Tests for ErrorClassifier."""

    def test_classify_timeout_error(self, classifier):
        """Timeout errors should be transient."""
        error = TimeoutError("Connection timed out")
        classified = classifier.classify(error)

        assert classified.error_type == ErrorType.TRANSIENT
        assert classified.is_retriable is True
        assert classified.recommended_action == RecoveryAction.RETRY

    def test_classify_connection_error(self, classifier):
        """Connection errors should be transient."""
        error = ConnectionError("Connection refused")
        classified = classifier.classify(error)

        assert classified.error_type == ErrorType.TRANSIENT
        assert classified.is_retriable is True

    def test_classify_value_error(self, classifier):
        """Value errors should be permanent."""
        error = ValueError("Invalid input")
        classified = classifier.classify(error)

        assert classified.error_type == ErrorType.PERMANENT
        assert classified.is_retriable is False
        assert classified.recommended_action == RecoveryAction.FAIL

    def test_classify_http_429(self, classifier):
        """HTTP 429 should be rate limited."""
        error = MagicMock()
        error.status_code = 429
        error.response = MagicMock()
        error.response.headers = {"Retry-After": "30"}

        classified = classifier.classify(error)

        assert classified.error_type == ErrorType.RATE_LIMITED
        assert classified.is_retriable is True
        assert classified.retry_after_seconds == 30.0

    def test_classify_http_500(self, classifier):
        """HTTP 500 should be transient."""
        error = MagicMock()
        error.status_code = 500

        classified = classifier.classify(error)

        assert classified.error_type == ErrorType.TRANSIENT
        assert classified.is_retriable is True

    def test_classify_http_401(self, classifier):
        """HTTP 401 should be permanent."""
        error = MagicMock()
        error.status_code = 401

        classified = classifier.classify(error)

        assert classified.error_type == ErrorType.PERMANENT
        assert classified.is_retriable is False

    def test_classify_unknown_error(self, classifier):
        """Unknown errors should default to unknown type."""
        error = Exception("Some random error")
        classified = classifier.classify(error)

        assert classified.error_type == ErrorType.UNKNOWN
        assert classified.is_retriable is True  # Conservative default

    def test_classify_with_keyword_timeout(self, classifier):
        """Error messages with 'timeout' should be transient."""
        error = Exception("The operation timed out")
        classified = classifier.classify(error)

        assert classified.error_type == ErrorType.TRANSIENT

    def test_classify_with_keyword_unauthorized(self, classifier):
        """Error messages with 'unauthorized' should be permanent."""
        error = Exception("Unauthorized access")
        classified = classifier.classify(error)

        assert classified.error_type == ErrorType.PERMANENT


# =============================================================================
# RETRY STRATEGY TESTS
# =============================================================================

class TestRetryStrategy:
    """Tests for RetryStrategy."""

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self):
        """Should retry on transient errors."""
        strategy = RetryStrategy(max_retries=3, base_delay=0.01, max_delay=0.1)

        classified = ClassifiedError(
            original_error=TimeoutError("timeout"),
            error_type=ErrorType.TRANSIENT,
            recommended_action=RecoveryAction.RETRY,
            is_retriable=True,
        )

        # Should be able to recover
        assert await strategy.can_recover(classified) is True

        # Create a mock operation that succeeds on retry
        attempts = [0]
        async def operation():
            attempts[0] += 1
            if attempts[0] < 2:
                raise TimeoutError("timeout")
            return "success"

        result = await strategy.recover(
            classified, operation, (), {}
        )
        assert result == "success"

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Should fail after max retries."""
        strategy = RetryStrategy(max_retries=2, base_delay=0.01, max_delay=0.1)

        classified = ClassifiedError(
            original_error=TimeoutError("timeout"),
            error_type=ErrorType.TRANSIENT,
            recommended_action=RecoveryAction.RETRY,
            is_retriable=True,
        )

        # Exhaust retries
        strategy._current_attempt = 2

        assert await strategy.can_recover(classified) is False

    def test_reset_clears_attempts(self):
        """Reset should clear attempt counter."""
        strategy = RetryStrategy(max_retries=3)
        strategy._current_attempt = 3
        strategy.reset()
        assert strategy._current_attempt == 0


# =============================================================================
# FALLBACK CHAIN TESTS
# =============================================================================

class TestFallbackChainStrategy:
    """Tests for FallbackChainStrategy."""

    @pytest.mark.asyncio
    async def test_fallback_chain_success(self):
        """Should use fallback when primary fails."""
        async def fallback1(*args, **kwargs):
            return "fallback1_result"

        async def fallback2(*args, **kwargs):
            return "fallback2_result"

        strategy = FallbackChainStrategy(
            fallbacks=[fallback1, fallback2],
            adapter_names=["backup1", "backup2"],
        )

        classified = ClassifiedError(
            original_error=Exception("primary failed"),
            error_type=ErrorType.TRANSIENT,
            recommended_action=RecoveryAction.FALLBACK,
            is_retriable=True,
        )

        assert await strategy.can_recover(classified) is True

        result = await strategy.recover(
            classified,
            AsyncMock(side_effect=Exception("primary")),
            (),
            {},
        )
        assert result == "fallback1_result"

    @pytest.mark.asyncio
    async def test_fallback_chain_exhausted(self):
        """Should report no recovery when all fallbacks used."""
        strategy = FallbackChainStrategy(
            fallbacks=[],
            adapter_names=[],
        )

        classified = ClassifiedError(
            original_error=Exception("failed"),
            error_type=ErrorType.TRANSIENT,
            recommended_action=RecoveryAction.FALLBACK,
            is_retriable=True,
        )

        assert await strategy.can_recover(classified) is False


# =============================================================================
# CACHE RECOVERY TESTS
# =============================================================================

class TestCacheRecoveryStrategy:
    """Tests for CacheRecoveryStrategy."""

    @pytest.mark.asyncio
    async def test_cache_hit_recovery(self):
        """Should return cached result on error."""
        strategy = CacheRecoveryStrategy(max_age_seconds=3600)

        # Store a result
        args = ("query",)
        kwargs = {"limit": 10}
        strategy.store(args, kwargs, {"result": "cached_data"})

        classified = ClassifiedError(
            original_error=Exception("service unavailable"),
            error_type=ErrorType.TRANSIENT,
            recommended_action=RecoveryAction.CACHE,
            is_retriable=True,
            context={"args": args, "kwargs": kwargs},
        )

        assert await strategy.can_recover(classified) is True

        result = await strategy.recover(
            classified,
            AsyncMock(),
            args,
            kwargs,
        )
        assert result == {"result": "cached_data"}

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Should not recover on cache miss."""
        strategy = CacheRecoveryStrategy(max_age_seconds=3600)

        classified = ClassifiedError(
            original_error=Exception("service unavailable"),
            error_type=ErrorType.TRANSIENT,
            recommended_action=RecoveryAction.CACHE,
            is_retriable=True,
            context={"args": (), "kwargs": {}},
        )

        assert await strategy.can_recover(classified) is False


# =============================================================================
# MEMORY FALLBACK TESTS
# =============================================================================

class TestMemoryFallbackStrategy:
    """Tests for MemoryFallbackStrategy."""

    @pytest.mark.asyncio
    async def test_memory_read_fallback(self):
        """Should read from fallback storage."""
        strategy = MemoryFallbackStrategy()
        strategy._fallback_storage["test_key"] = "test_value"

        classified = ClassifiedError(
            original_error=Exception("Redis connection failed"),
            error_type=ErrorType.TRANSIENT,
            recommended_action=RecoveryAction.RETRY,
            is_retriable=True,
            context={"operation_type": "memory_read"},
        )

        assert await strategy.can_recover(classified) is True

        result = await strategy.recover(
            classified,
            AsyncMock(),
            (),
            {"key": "test_key"},
        )
        assert result == "test_value"

    @pytest.mark.asyncio
    async def test_memory_write_fallback(self):
        """Should write to fallback storage."""
        strategy = MemoryFallbackStrategy()

        classified = ClassifiedError(
            original_error=Exception("Redis connection failed"),
            error_type=ErrorType.TRANSIENT,
            recommended_action=RecoveryAction.RETRY,
            is_retriable=True,
            context={"operation_type": "memory_write"},
        )

        assert await strategy.can_recover(classified) is True

        await strategy.recover(
            classified,
            AsyncMock(),
            (),
            {"key": "new_key", "value": "new_value"},
        )

        assert strategy._fallback_storage["new_key"] == "new_value"


# =============================================================================
# PARTIAL RESULT TESTS
# =============================================================================

class TestPartialResultHandler:
    """Tests for PartialResultHandler."""

    def test_partial_success_accepted(self):
        """Should accept partial results above threshold."""
        handler = PartialResultHandler(min_success_rate=0.5, fail_on_empty=True)
        handler.set_total_requested(4)

        handler.add_result("result1")
        handler.add_result("result2")
        handler.add_result("result3")
        handler.add_error(ClassifiedError(
            original_error=Exception("failed"),
            error_type=ErrorType.TRANSIENT,
            recommended_action=RecoveryAction.RETRY,
            is_retriable=True,
        ))

        assert handler.should_accept_partial() is True

        partial = handler.get_partial_result()
        assert partial.success_rate == 0.75
        assert len(partial.results) == 3
        assert len(partial.errors) == 1

    def test_partial_success_rejected(self):
        """Should reject partial results below threshold."""
        handler = PartialResultHandler(min_success_rate=0.5, fail_on_empty=True)
        handler.set_total_requested(4)

        handler.add_result("result1")
        handler.add_error(ClassifiedError(
            original_error=Exception("failed"),
            error_type=ErrorType.TRANSIENT,
            recommended_action=RecoveryAction.RETRY,
            is_retriable=True,
        ))
        handler.add_error(ClassifiedError(
            original_error=Exception("failed"),
            error_type=ErrorType.TRANSIENT,
            recommended_action=RecoveryAction.RETRY,
            is_retriable=True,
        ))
        handler.add_error(ClassifiedError(
            original_error=Exception("failed"),
            error_type=ErrorType.TRANSIENT,
            recommended_action=RecoveryAction.RETRY,
            is_retriable=True,
        ))

        assert handler.should_accept_partial() is False

    def test_empty_result_rejected(self):
        """Should reject empty results when fail_on_empty is True."""
        handler = PartialResultHandler(min_success_rate=0.0, fail_on_empty=True)
        handler.set_total_requested(4)

        assert handler.should_accept_partial() is False


# =============================================================================
# ERROR AGGREGATION TESTS
# =============================================================================

class TestErrorAggregator:
    """Tests for ErrorAggregator."""

    @pytest.mark.asyncio
    async def test_record_error(self, aggregator):
        """Should record errors and update patterns."""
        error = ClassifiedError(
            original_error=TimeoutError("timeout"),
            error_type=ErrorType.TRANSIENT,
            recommended_action=RecoveryAction.RETRY,
            is_retriable=True,
        )

        await aggregator.record(error, "test_operation")

        stats = await aggregator.get_statistics()
        assert stats["total_errors"] == 1
        assert ErrorType.TRANSIENT.value in stats["by_type"]

    @pytest.mark.asyncio
    async def test_error_pattern_detection(self, aggregator):
        """Should detect recurring error patterns."""
        # Record multiple similar errors
        for _ in range(5):
            error = ClassifiedError(
                original_error=ConnectionError("connection refused"),
                error_type=ErrorType.TRANSIENT,
                recommended_action=RecoveryAction.RETRY,
                is_retriable=True,
            )
            await aggregator.record(error, "api_call")

        patterns = await aggregator.get_error_patterns(min_count=3)
        assert len(patterns) >= 1
        assert patterns[0].count >= 5

    @pytest.mark.asyncio
    async def test_health_assessment(self, aggregator):
        """Should provide health assessment."""
        # Record some errors
        for _ in range(5):
            error = ClassifiedError(
                original_error=Exception("test error"),
                error_type=ErrorType.UNKNOWN,
                recommended_action=RecoveryAction.RETRY,
                is_retriable=True,
            )
            await aggregator.record(error, "operation")

        health = await aggregator.get_health_assessment()
        assert "health_level" in health
        assert "errors_per_hour" in health
        assert "recommendations" in health


# =============================================================================
# ERROR RECOVERY INTEGRATION TESTS
# =============================================================================

class TestErrorRecovery:
    """Tests for ErrorRecovery main class."""

    @pytest.mark.asyncio
    async def test_successful_execution(self, recovery):
        """Should return success for successful operations."""
        async def successful_operation():
            return "success"

        result = await recovery.execute(successful_operation)

        assert result.success is True
        assert result.result == "success"
        assert result.attempts == 1
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self, recovery):
        """Should retry on transient errors and eventually succeed."""
        attempts = [0]

        async def flaky_operation():
            attempts[0] += 1
            if attempts[0] < 2:
                raise ConnectionError("connection failed")
            return "success after retry"

        result = await recovery.execute(flaky_operation)

        assert result.success is True
        assert result.result == "success after retry"

    @pytest.mark.asyncio
    async def test_fail_on_permanent_error(self, recovery):
        """Should fail immediately on permanent errors."""
        async def invalid_operation():
            raise ValueError("invalid input")

        result = await recovery.execute(invalid_operation)

        assert result.success is False
        assert len(result.errors) > 0
        assert result.errors[0].error_type == ErrorType.PERMANENT

    @pytest.mark.asyncio
    async def test_fallback_chain_execution(self, recovery):
        """Should use fallbacks when primary fails."""
        async def primary():
            raise Exception("primary failed")

        async def fallback1():
            return "fallback1 result"

        result = await recovery.execute_with_fallback(
            primary=primary,
            fallbacks=[fallback1],
            fallback_names=["backup"],
        )

        assert result.success is True
        assert result.result == "fallback1 result"
        assert "fallback:backup" in (result.recovery_used or "")

    @pytest.mark.asyncio
    async def test_batch_partial_results(self, recovery):
        """Should handle partial batch results."""
        items = [1, 2, 3, 4, 5]

        async def process_item(item):
            if item == 3:
                raise Exception("failed on item 3")
            return item * 2

        result = await recovery.execute_batch(
            items=items,
            process_fn=process_item,
            concurrency=2,
        )

        # 4 out of 5 succeeded, which is > 50% threshold
        assert result.success is True
        assert result.partial_result is not None
        assert result.partial_result.total_succeeded == 4
        assert result.partial_result.total_requested == 5

    @pytest.mark.asyncio
    async def test_health_status(self, recovery):
        """Should provide health status."""
        # Generate some errors
        async def failing_operation():
            raise TimeoutError("timeout")

        for _ in range(3):
            await recovery.execute(failing_operation)

        health = await recovery.get_health_status()
        assert "health_level" in health


# =============================================================================
# DECORATOR TESTS
# =============================================================================

class TestWithRecoveryDecorator:
    """Tests for @with_recovery decorator."""

    @pytest.mark.asyncio
    async def test_decorator_success(self):
        """Decorated function should return RecoveryResult."""
        @with_recovery(config=RecoveryConfig(max_retries=1, base_delay=0.01))
        async def my_operation():
            return "result"

        result = await my_operation()

        assert result.success is True
        assert result.result == "result"

    @pytest.mark.asyncio
    async def test_decorator_with_fallback(self):
        """Decorator should use fallbacks."""
        async def fallback_fn():
            return "fallback result"

        @with_recovery(
            config=RecoveryConfig(max_retries=1, base_delay=0.01),
            fallbacks=[fallback_fn],
        )
        async def failing_operation():
            raise Exception("primary failed")

        result = await failing_operation()

        assert result.success is True
        assert result.result == "fallback result"


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_recovery(self):
        """Should create ErrorRecovery with config."""
        config = RecoveryConfig(max_retries=5)
        recovery = create_recovery(config)

        assert recovery.config.max_retries == 5

    def test_get_global_recovery(self):
        """Should return global singleton."""
        recovery1 = get_error_recovery()
        recovery2 = get_error_recovery()

        # Should be same instance (singleton)
        assert recovery1 is recovery2


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
