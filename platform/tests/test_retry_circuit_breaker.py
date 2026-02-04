"""
Tests for Retry with Circuit Breaker Integration
=================================================

Tests the exponential backoff with jitter and circuit breaker
integration in platform/adapters/retry.py

Run with: pytest platform/tests/test_retry_circuit_breaker.py -v
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from adapters.retry import (
    RetryConfig,
    RetryWithCircuitBreakerConfig,
    calculate_delay,
    is_retryable_exception,
    retry_async,
    retry_sync,
    with_retry,
    with_retry_and_circuit_breaker,
    retry_async_with_circuit_breaker,
    RetryableClient,
    ResilientClient,
    http_request_with_retry,
    http_request_with_retry_and_circuit_breaker,
    RETRYABLE_STATUS_CODES,
    RETRYABLE_EXCEPTIONS,
    CIRCUIT_BREAKER_AVAILABLE,
)


class TestCalculateDelay:
    """Tests for calculate_delay function."""

    def test_exponential_backoff_basic(self):
        """Test basic exponential backoff calculation."""
        # attempt 0: base_delay * 2^0 = 1.0
        delay = calculate_delay(0, base_delay=1.0, max_delay=60.0, jitter=0.0)
        assert delay == 1.0

        # attempt 1: base_delay * 2^1 = 2.0
        delay = calculate_delay(1, base_delay=1.0, max_delay=60.0, jitter=0.0)
        assert delay == 2.0

        # attempt 2: base_delay * 2^2 = 4.0
        delay = calculate_delay(2, base_delay=1.0, max_delay=60.0, jitter=0.0)
        assert delay == 4.0

        # attempt 3: base_delay * 2^3 = 8.0
        delay = calculate_delay(3, base_delay=1.0, max_delay=60.0, jitter=0.0)
        assert delay == 8.0

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        # With high attempt, delay should not exceed max
        delay = calculate_delay(10, base_delay=1.0, max_delay=30.0, jitter=0.0)
        assert delay == 30.0

    def test_jitter_applied(self):
        """Test that jitter adds randomness to delay."""
        delays = set()
        for _ in range(100):
            delay = calculate_delay(2, base_delay=1.0, max_delay=60.0, jitter=0.5)
            delays.add(round(delay, 2))

        # With jitter=0.5 and base delay 4.0 (2^2), expect range [2.0, 6.0]
        assert len(delays) > 1  # Multiple different values
        assert all(2.0 <= d <= 6.0 for d in delays)

    def test_jitter_range(self):
        """Test jitter produces values within expected range."""
        # With jitter=0.5, delay should be in [delay*(1-0.5), delay*(1+0.5)]
        # For attempt 0, base=1.0: range [0.5, 1.5]
        for _ in range(100):
            delay = calculate_delay(0, base_delay=1.0, max_delay=60.0, jitter=0.5)
            assert 0.5 <= delay <= 1.5

    def test_custom_exponential_base(self):
        """Test custom exponential base."""
        # With base 3: 1.0 * 3^2 = 9.0
        delay = calculate_delay(2, base_delay=1.0, max_delay=60.0, exponential_base=3.0, jitter=0.0)
        assert delay == 9.0


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter == 0.5
        assert 429 in config.retry_on_status_codes
        assert 503 in config.retry_on_status_codes
        assert ConnectionError in config.retry_on_exceptions
        assert TimeoutError in config.retry_on_exceptions

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RetryConfig(
            max_retries=5,
            base_delay=0.5,
            max_delay=30.0,
            jitter=0.3,
        )
        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.jitter == 0.3


class TestRetryWithCircuitBreakerConfig:
    """Tests for RetryWithCircuitBreakerConfig dataclass."""

    def test_inherits_retry_config(self):
        """Test that it inherits from RetryConfig."""
        config = RetryWithCircuitBreakerConfig(max_retries=5, base_delay=0.5)
        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert hasattr(config, 'enable_circuit_breaker')

    def test_circuit_breaker_defaults(self):
        """Test circuit breaker default values."""
        config = RetryWithCircuitBreakerConfig()
        assert config.enable_circuit_breaker is True
        assert config.circuit_breaker_name is None
        assert config.circuit_breaker_instance is None
        assert config.fail_fast_on_circuit_open is True

    def test_custom_circuit_breaker_config(self):
        """Test custom circuit breaker configuration."""
        config = RetryWithCircuitBreakerConfig(
            max_retries=3,
            circuit_breaker_name="test_adapter",
            enable_circuit_breaker=True,
            fail_fast_on_circuit_open=False,
        )
        assert config.circuit_breaker_name == "test_adapter"
        assert config.fail_fast_on_circuit_open is False


class TestIsRetryableException:
    """Tests for is_retryable_exception function."""

    def test_connection_error(self):
        """Test ConnectionError is retryable."""
        assert is_retryable_exception(ConnectionError("test"))

    def test_timeout_error(self):
        """Test TimeoutError is retryable."""
        assert is_retryable_exception(TimeoutError("test"))

    def test_value_error_not_retryable(self):
        """Test ValueError is not retryable by default."""
        assert not is_retryable_exception(ValueError("test"))

    def test_custom_retryable_exceptions(self):
        """Test custom retryable exceptions."""
        assert is_retryable_exception(
            ValueError("test"),
            retry_on_exceptions=(ValueError,),
        )


class TestRetryAsync:
    """Tests for retry_async function."""

    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        """Test successful call with no retries needed."""
        mock_func = AsyncMock(return_value="success")
        config = RetryConfig(max_retries=3)

        result = await retry_async(mock_func, config=config)

        assert result == "success"
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_connection_error(self):
        """Test retry on ConnectionError."""
        mock_func = AsyncMock(side_effect=[ConnectionError("fail"), "success"])
        config = RetryConfig(max_retries=3, base_delay=0.01)

        result = await retry_async(mock_func, config=config)

        assert result == "success"
        assert mock_func.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """Test that retries are exhausted after max_retries."""
        mock_func = AsyncMock(side_effect=ConnectionError("always fail"))
        config = RetryConfig(max_retries=2, base_delay=0.01)

        with pytest.raises(ConnectionError):
            await retry_async(mock_func, config=config)

        assert mock_func.call_count == 3  # initial + 2 retries

    @pytest.mark.asyncio
    async def test_non_retryable_exception_raised_immediately(self):
        """Test non-retryable exception is raised without retry."""
        mock_func = AsyncMock(side_effect=ValueError("not retryable"))
        config = RetryConfig(max_retries=3)

        with pytest.raises(ValueError):
            await retry_async(mock_func, config=config)

        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_on_retry_callback(self):
        """Test on_retry callback is called."""
        retry_calls = []

        def on_retry(attempt, exc, delay):
            retry_calls.append((attempt, type(exc).__name__, delay))

        mock_func = AsyncMock(side_effect=[ConnectionError("fail"), "success"])
        config = RetryConfig(max_retries=3, base_delay=0.01, on_retry=on_retry)

        await retry_async(mock_func, config=config)

        assert len(retry_calls) == 1
        assert retry_calls[0][0] == 1
        assert retry_calls[0][1] == "ConnectionError"


class TestRetrySync:
    """Tests for retry_sync function."""

    def test_success_no_retry(self):
        """Test successful call with no retries needed."""
        mock_func = MagicMock(return_value="success")
        config = RetryConfig(max_retries=3)

        result = retry_sync(mock_func, config=config)

        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_on_connection_error(self):
        """Test retry on ConnectionError."""
        mock_func = MagicMock(side_effect=[ConnectionError("fail"), "success"])
        config = RetryConfig(max_retries=3, base_delay=0.01)

        result = retry_sync(mock_func, config=config)

        assert result == "success"
        assert mock_func.call_count == 2


class TestWithRetryDecorator:
    """Tests for @with_retry decorator."""

    @pytest.mark.asyncio
    async def test_decorator_on_async_function(self):
        """Test decorator works on async function."""
        call_count = 0

        @with_retry(max_retries=2, base_delay=0.01)
        async def my_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("fail")
            return "success"

        result = await my_func()
        assert result == "success"
        assert call_count == 2

    def test_decorator_on_sync_function(self):
        """Test decorator works on sync function."""
        call_count = 0

        @with_retry(max_retries=2, base_delay=0.01)
        def my_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("fail")
            return "success"

        result = my_func()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_decorator_without_parentheses(self):
        """Test decorator works without parentheses."""
        call_count = 0

        @with_retry
        async def my_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await my_func()
        assert result == "success"


class TestWithRetryAndCircuitBreaker:
    """Tests for @with_retry_and_circuit_breaker decorator."""

    @pytest.mark.asyncio
    async def test_basic_functionality(self):
        """Test basic decorator functionality."""
        call_count = 0

        @with_retry_and_circuit_breaker(
            max_retries=2,
            base_delay=0.01,
            enable_circuit_breaker=False,  # Disable for test
        )
        async def my_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("fail")
            return "success"

        result = await my_func()
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_with_circuit_breaker_name(self):
        """Test decorator with circuit breaker name."""
        @with_retry_and_circuit_breaker(
            max_retries=1,
            base_delay=0.01,
            circuit_breaker_name="test_adapter",
        )
        async def my_func():
            return "success"

        result = await my_func()
        assert result == "success"


class TestRetryableClientMixin:
    """Tests for RetryableClient mixin."""

    @pytest.mark.asyncio
    async def test_retry_call(self):
        """Test retry_call method."""
        class TestAdapter(RetryableClient):
            def __init__(self):
                self.init_retry(max_retries=2, base_delay=0.01)

        adapter = TestAdapter()
        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("fail")
            return "success"

        result = await adapter.retry_call(flaky_func)
        assert result == "success"
        assert call_count == 2


class TestResilientClientMixin:
    """Tests for ResilientClient mixin."""

    @pytest.mark.asyncio
    async def test_resilient_call(self):
        """Test resilient_call method."""
        class TestAdapter(ResilientClient):
            def __init__(self):
                self.init_resilience(
                    max_retries=2,
                    base_delay=0.01,
                    circuit_breaker_name="test_adapter",
                    enable_circuit_breaker=False,  # Disable for test
                )

        adapter = TestAdapter()
        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("fail")
            return "success"

        result = await adapter.resilient_call(flaky_func)
        assert result == "success"
        assert call_count == 2


class TestRetryableStatusCodes:
    """Tests for retryable status codes."""

    def test_contains_rate_limit(self):
        """Test 429 (rate limit) is retryable."""
        assert 429 in RETRYABLE_STATUS_CODES

    def test_contains_server_errors(self):
        """Test 5xx errors are retryable."""
        assert 500 in RETRYABLE_STATUS_CODES
        assert 502 in RETRYABLE_STATUS_CODES
        assert 503 in RETRYABLE_STATUS_CODES
        assert 504 in RETRYABLE_STATUS_CODES

    def test_contains_cloudflare_errors(self):
        """Test Cloudflare error codes are retryable."""
        assert 520 in RETRYABLE_STATUS_CODES
        assert 521 in RETRYABLE_STATUS_CODES
        assert 522 in RETRYABLE_STATUS_CODES


class TestExponentialBackoffTiming:
    """Tests to verify exponential backoff timing behavior."""

    @pytest.mark.asyncio
    async def test_delay_increases_exponentially(self):
        """Test that delay increases between retries."""
        delays = []

        def on_retry(attempt, exc, delay):
            delays.append(delay)

        mock_func = AsyncMock(side_effect=[
            ConnectionError("1"),
            ConnectionError("2"),
            ConnectionError("3"),
            "success"
        ])
        config = RetryConfig(
            max_retries=3,
            base_delay=0.01,
            jitter=0.0,  # No jitter for predictable test
            on_retry=on_retry,
        )

        await retry_async(mock_func, config=config)

        # With no jitter: 0.01, 0.02, 0.04
        assert len(delays) == 3
        assert delays[0] < delays[1] < delays[2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
