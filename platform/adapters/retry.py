"""
Exponential Backoff Retry Utilities for Research Adapters
==========================================================

Provides robust retry logic with:
- Exponential backoff with jitter
- Configurable retry conditions
- Support for both sync and async functions
- Transient error detection (429, 500-599, ConnectionError, Timeout)

Usage:
    from adapters.retry import with_retry, RetryConfig

    # As a decorator
    @with_retry(max_retries=3, base_delay=1.0)
    async def my_api_call():
        ...

    # With custom config
    config = RetryConfig(max_retries=5, base_delay=0.5, max_delay=60.0)
    @with_retry(config=config)
    async def my_api_call():
        ...

    # Programmatic use
    result = await retry_async(my_func, args, kwargs, config=config)
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")


# HTTP status codes that indicate transient errors worth retrying
RETRYABLE_STATUS_CODES: Set[int] = {
    429,  # Too Many Requests (rate limit)
    500,  # Internal Server Error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
    520,  # Cloudflare: Unknown Error
    521,  # Cloudflare: Web Server Is Down
    522,  # Cloudflare: Connection Timed Out
    523,  # Cloudflare: Origin Is Unreachable
    524,  # Cloudflare: A Timeout Occurred
}

# Exception types that indicate transient errors worth retrying
RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    """Maximum number of retry attempts (not including the initial attempt)."""

    base_delay: float = 1.0
    """Base delay in seconds for exponential backoff."""

    max_delay: float = 60.0
    """Maximum delay in seconds between retries."""

    exponential_base: float = 2.0
    """Base for exponential backoff calculation."""

    jitter: float = 0.5
    """Jitter factor (0-1) to randomize delays and prevent thundering herd."""

    retry_on_status_codes: Set[int] = field(default_factory=lambda: RETRYABLE_STATUS_CODES.copy())
    """HTTP status codes that trigger a retry."""

    retry_on_exceptions: Tuple[Type[Exception], ...] = RETRYABLE_EXCEPTIONS
    """Exception types that trigger a retry."""

    on_retry: Optional[Callable[[int, Exception, float], None]] = None
    """Optional callback called before each retry: (attempt, exception, delay)."""


def calculate_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    exponential_base: float = 2.0,
    jitter: float = 0.5,
) -> float:
    """
    Calculate delay with exponential backoff and jitter.

    Args:
        attempt: Current retry attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Jitter factor (0-1) for randomization

    Returns:
        Delay in seconds with jitter applied
    """
    # Exponential backoff: base_delay * (exponential_base ^ attempt)
    delay = base_delay * (exponential_base ** attempt)

    # Cap at max delay
    delay = min(delay, max_delay)

    # Apply jitter: delay * (1 - jitter + random(0, 2*jitter))
    # This gives a range of [delay * (1-jitter), delay * (1+jitter)]
    if jitter > 0:
        jitter_range = delay * jitter
        delay = delay - jitter_range + (random.random() * 2 * jitter_range)

    return max(0.0, delay)


def is_retryable_exception(
    exc: Exception,
    retry_on_exceptions: Tuple[Type[Exception], ...] = RETRYABLE_EXCEPTIONS,
    retry_on_status_codes: Set[int] = RETRYABLE_STATUS_CODES,
) -> bool:
    """
    Check if an exception indicates a transient error worth retrying.

    Args:
        exc: The exception to check
        retry_on_exceptions: Tuple of exception types to retry on
        retry_on_status_codes: Set of HTTP status codes to retry on

    Returns:
        True if the exception is retryable
    """
    # Direct exception type match
    if isinstance(exc, retry_on_exceptions):
        return True

    # Check for HTTP status errors with retryable status codes
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in retry_on_status_codes

    # Check for generic HTTP errors that might wrap status codes
    if hasattr(exc, "status_code"):
        return getattr(exc, "status_code") in retry_on_status_codes

    if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
        return exc.response.status_code in retry_on_status_codes

    return False


async def retry_async(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    config: Optional[RetryConfig] = None,
    **kwargs: Any,
) -> T:
    """
    Execute an async function with retry logic.

    Args:
        func: Async function to execute
        *args: Positional arguments for the function
        config: Retry configuration (uses defaults if not provided)
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function call

    Raises:
        The last exception if all retries are exhausted
    """
    if config is None:
        config = RetryConfig()

    last_exception: Optional[Exception] = None

    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)

        except Exception as exc:
            last_exception = exc

            # Check if this is a retryable error
            if not is_retryable_exception(
                exc,
                config.retry_on_exceptions,
                config.retry_on_status_codes,
            ):
                logger.debug(f"Non-retryable exception: {type(exc).__name__}: {exc}")
                raise

            # Check if we have retries left
            if attempt >= config.max_retries:
                logger.warning(
                    f"All {config.max_retries} retries exhausted for {getattr(func, '__name__', repr(func))}. "
                    f"Last error: {type(exc).__name__}: {exc}"
                )
                raise

            # Calculate delay
            delay = calculate_delay(
                attempt,
                config.base_delay,
                config.max_delay,
                config.exponential_base,
                config.jitter,
            )

            logger.info(
                f"Retry {attempt + 1}/{config.max_retries} for {getattr(func, '__name__', repr(func))} "
                f"after {type(exc).__name__}: {exc}. Waiting {delay:.2f}s..."
            )

            # Call optional callback
            if config.on_retry:
                try:
                    config.on_retry(attempt + 1, exc, delay)
                except Exception as callback_exc:
                    logger.warning(f"Retry callback raised exception: {callback_exc}")

            # Wait before retry
            await asyncio.sleep(delay)

    # This should not be reached, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError(f"Unexpected state in retry_async for {getattr(func, '__name__', repr(func))}")


def retry_sync(
    func: Callable[..., T],
    *args: Any,
    config: Optional[RetryConfig] = None,
    **kwargs: Any,
) -> T:
    """
    Execute a sync function with retry logic.

    Args:
        func: Sync function to execute
        *args: Positional arguments for the function
        config: Retry configuration (uses defaults if not provided)
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function call

    Raises:
        The last exception if all retries are exhausted
    """
    if config is None:
        config = RetryConfig()

    last_exception: Optional[Exception] = None

    for attempt in range(config.max_retries + 1):
        try:
            return func(*args, **kwargs)

        except Exception as exc:
            last_exception = exc

            # Check if this is a retryable error
            if not is_retryable_exception(
                exc,
                config.retry_on_exceptions,
                config.retry_on_status_codes,
            ):
                logger.debug(f"Non-retryable exception: {type(exc).__name__}: {exc}")
                raise

            # Check if we have retries left
            if attempt >= config.max_retries:
                logger.warning(
                    f"All {config.max_retries} retries exhausted for {getattr(func, '__name__', repr(func))}. "
                    f"Last error: {type(exc).__name__}: {exc}"
                )
                raise

            # Calculate delay
            delay = calculate_delay(
                attempt,
                config.base_delay,
                config.max_delay,
                config.exponential_base,
                config.jitter,
            )

            logger.info(
                f"Retry {attempt + 1}/{config.max_retries} for {getattr(func, '__name__', repr(func))} "
                f"after {type(exc).__name__}: {exc}. Waiting {delay:.2f}s..."
            )

            # Call optional callback
            if config.on_retry:
                try:
                    config.on_retry(attempt + 1, exc, delay)
                except Exception as callback_exc:
                    logger.warning(f"Retry callback raised exception: {callback_exc}")

            # Wait before retry
            time.sleep(delay)

    # This should not be reached, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError(f"Unexpected state in retry_sync for {getattr(func, '__name__', repr(func))}")


def with_retry(
    func: Optional[Callable[..., T]] = None,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: float = 0.5,
    retry_on_status_codes: Optional[Set[int]] = None,
    retry_on_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    config: Optional[RetryConfig] = None,
) -> Union[Callable[..., T], Callable[[Callable[..., T]], Callable[..., T]]]:
    """
    Decorator to add retry logic to a function.

    Can be used with or without arguments:

        @with_retry
        async def my_func():
            ...

        @with_retry(max_retries=5, base_delay=0.5)
        async def my_func():
            ...

        @with_retry(config=RetryConfig(...))
        async def my_func():
            ...

    Args:
        func: The function to decorate (when used without parentheses)
        max_retries: Maximum retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Exponential backoff base
        jitter: Jitter factor (0-1)
        retry_on_status_codes: HTTP status codes to retry on
        retry_on_exceptions: Exception types to retry on
        on_retry: Optional callback for retry events
        config: Full RetryConfig (overrides individual params)

    Returns:
        Decorated function with retry logic
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        # Build config from parameters or use provided config
        if config is not None:
            retry_config = config
        else:
            retry_config = RetryConfig(
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                jitter=jitter,
                retry_on_status_codes=(
                    retry_on_status_codes
                    if retry_on_status_codes is not None
                    else RETRYABLE_STATUS_CODES.copy()
                ),
                retry_on_exceptions=(
                    retry_on_exceptions
                    if retry_on_exceptions is not None
                    else RETRYABLE_EXCEPTIONS
                ),
                on_retry=on_retry,
            )

        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                return await retry_async(fn, *args, config=retry_config, **kwargs)

            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                return retry_sync(fn, *args, config=retry_config, **kwargs)

            return sync_wrapper

    # Handle both @with_retry and @with_retry() usage
    if func is not None:
        return decorator(func)
    return decorator


class RetryableClient:
    """
    Mixin class that adds retry capabilities to adapter classes.

    Usage:
        class MyAdapter(SDKAdapter, RetryableClient):
            def __init__(self):
                super().__init__()
                self.init_retry(max_retries=3, base_delay=1.0)

            async def _my_api_call(self, ...):
                return await self.retry_call(self._actual_api_call, ...)
    """

    _retry_config: RetryConfig

    def init_retry(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """Initialize retry configuration."""
        self._retry_config = RetryConfig(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            jitter=jitter,
            **kwargs,
        )

    async def retry_call(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute an async function with retry logic using this client's config."""
        config = getattr(self, "_retry_config", RetryConfig())
        return await retry_async(func, *args, config=config, **kwargs)

    def retry_call_sync(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute a sync function with retry logic using this client's config."""
        config = getattr(self, "_retry_config", RetryConfig())
        return retry_sync(func, *args, config=config, **kwargs)


# Convenience function for wrapping httpx calls
async def http_request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    config: Optional[RetryConfig] = None,
    **kwargs: Any,
) -> httpx.Response:
    """
    Make an HTTP request with retry logic.

    Args:
        client: httpx.AsyncClient instance
        method: HTTP method (GET, POST, etc.)
        url: Request URL
        config: Retry configuration
        **kwargs: Additional arguments for the request

    Returns:
        httpx.Response

    Raises:
        Last exception if all retries exhausted
    """
    async def make_request() -> httpx.Response:
        response = await client.request(method, url, **kwargs)
        response.raise_for_status()
        return response

    return await retry_async(make_request, config=config)


# =============================================================================
# Circuit Breaker Integration
# =============================================================================

# Import circuit breaker if available
try:
    from .circuit_breaker_manager import adapter_circuit_breaker, get_adapter_circuit_manager
    from core.resilience import CircuitBreaker, CircuitOpenError
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False
    CircuitBreaker = None
    CircuitOpenError = Exception

    def adapter_circuit_breaker(name: str):
        """Dummy circuit breaker when not available."""

        class DummyBreaker:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        return DummyBreaker()

    def get_adapter_circuit_manager():
        return None


@dataclass
class RetryWithCircuitBreakerConfig(RetryConfig):
    """
    Extended retry configuration with circuit breaker integration.

    Combines exponential backoff retry logic with circuit breaker
    pattern for comprehensive resilience handling.
    """

    enable_circuit_breaker: bool = True
    """Enable circuit breaker integration."""

    circuit_breaker_name: Optional[str] = None
    """Name for circuit breaker (used with adapter_circuit_breaker)."""

    circuit_breaker_instance: Optional[Any] = None
    """Direct circuit breaker instance (overrides name-based lookup)."""

    fail_fast_on_circuit_open: bool = True
    """If True, immediately fail when circuit is open without retrying."""


async def retry_async_with_circuit_breaker(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    config: Optional[RetryWithCircuitBreakerConfig] = None,
    **kwargs: Any,
) -> T:
    """
    Execute an async function with retry logic and circuit breaker protection.

    This combines exponential backoff retry with circuit breaker pattern:
    1. First checks if circuit is open (fails fast if enabled)
    2. Executes the function with retry logic
    3. Records success/failure to circuit breaker

    Args:
        func: Async function to execute
        *args: Positional arguments for the function
        config: Combined retry and circuit breaker configuration
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function call

    Raises:
        CircuitOpenError: If circuit is open and fail_fast_on_circuit_open is True
        The last exception if all retries are exhausted
    """
    if config is None:
        config = RetryWithCircuitBreakerConfig()

    # Get circuit breaker instance
    circuit_breaker = None
    if CIRCUIT_BREAKER_AVAILABLE and config.enable_circuit_breaker:
        if config.circuit_breaker_instance is not None:
            circuit_breaker = config.circuit_breaker_instance
        elif config.circuit_breaker_name:
            circuit_breaker = adapter_circuit_breaker(config.circuit_breaker_name)

    # Execute with or without circuit breaker
    if circuit_breaker is not None:
        async with circuit_breaker:
            return await retry_async(func, *args, config=config, **kwargs)
    else:
        return await retry_async(func, *args, config=config, **kwargs)


def with_retry_and_circuit_breaker(
    func: Optional[Callable[..., T]] = None,
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: float = 0.5,
    retry_on_status_codes: Optional[Set[int]] = None,
    retry_on_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    circuit_breaker_name: Optional[str] = None,
    enable_circuit_breaker: bool = True,
    fail_fast_on_circuit_open: bool = True,
) -> Union[Callable[..., T], Callable[[Callable[..., T]], Callable[..., T]]]:
    """
    Decorator combining retry logic with circuit breaker protection.

    This is the recommended decorator for adapter methods as it provides
    comprehensive resilience:
    - Exponential backoff with jitter to handle transient failures
    - Circuit breaker to prevent cascade failures
    - Proper integration with the adapter circuit breaker manager

    Usage:
        @with_retry_and_circuit_breaker(
            max_retries=3,
            circuit_breaker_name="exa_adapter"
        )
        async def my_api_call():
            ...

        # Or with custom config
        @with_retry_and_circuit_breaker(
            max_retries=5,
            base_delay=0.5,
            circuit_breaker_name="tavily_adapter",
            fail_fast_on_circuit_open=True
        )
        async def another_api_call():
            ...

    Args:
        func: The function to decorate
        max_retries: Maximum retry attempts (default: 3)
        base_delay: Base delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        exponential_base: Exponential backoff base (default: 2.0)
        jitter: Jitter factor 0-1 (default: 0.5)
        retry_on_status_codes: HTTP status codes to retry on
        retry_on_exceptions: Exception types to retry on
        on_retry: Optional callback for retry events
        circuit_breaker_name: Name for adapter circuit breaker
        enable_circuit_breaker: Enable circuit breaker (default: True)
        fail_fast_on_circuit_open: Fail immediately if circuit open

    Returns:
        Decorated function with retry and circuit breaker logic
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        # Build config
        config = RetryWithCircuitBreakerConfig(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            jitter=jitter,
            retry_on_status_codes=(
                retry_on_status_codes
                if retry_on_status_codes is not None
                else RETRYABLE_STATUS_CODES.copy()
            ),
            retry_on_exceptions=(
                retry_on_exceptions
                if retry_on_exceptions is not None
                else RETRYABLE_EXCEPTIONS
            ),
            on_retry=on_retry,
            enable_circuit_breaker=enable_circuit_breaker,
            circuit_breaker_name=circuit_breaker_name,
            fail_fast_on_circuit_open=fail_fast_on_circuit_open,
        )

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                return await retry_async_with_circuit_breaker(
                    fn, *args, config=config, **kwargs
                )

            return async_wrapper
        else:
            # For sync functions, use basic retry without circuit breaker
            # (Circuit breaker is async-only in our implementation)
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> T:
                return retry_sync(fn, *args, config=config, **kwargs)

            return sync_wrapper

    # Handle both @decorator and @decorator() usage
    if func is not None:
        return decorator(func)
    return decorator


class ResilientClient(RetryableClient):
    """
    Enhanced mixin class with both retry and circuit breaker capabilities.

    Extends RetryableClient to add circuit breaker protection for
    comprehensive resilience handling.

    Usage:
        class MyAdapter(SDKAdapter, ResilientClient):
            def __init__(self):
                super().__init__()
                self.init_resilience(
                    max_retries=3,
                    base_delay=1.0,
                    circuit_breaker_name="my_adapter"
                )

            async def _my_api_call(self, ...):
                return await self.resilient_call(self._actual_api_call, ...)
    """

    _circuit_breaker_name: Optional[str]
    _enable_circuit_breaker: bool

    def init_resilience(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: float = 0.5,
        circuit_breaker_name: Optional[str] = None,
        enable_circuit_breaker: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize resilience configuration with retry and circuit breaker.

        Args:
            max_retries: Maximum retry attempts
            base_delay: Base delay for exponential backoff
            max_delay: Maximum delay between retries
            jitter: Jitter factor (0-1) for randomization
            circuit_breaker_name: Name for adapter circuit breaker
            enable_circuit_breaker: Enable circuit breaker protection
            **kwargs: Additional arguments for RetryConfig
        """
        # Initialize parent retry config
        self.init_retry(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            jitter=jitter,
            **kwargs,
        )

        # Store circuit breaker settings
        self._circuit_breaker_name = circuit_breaker_name
        self._enable_circuit_breaker = enable_circuit_breaker

    async def resilient_call(
        self,
        func: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute an async function with retry and circuit breaker protection.

        This is the primary method for making resilient API calls within
        adapters that use this mixin.

        Args:
            func: Async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function call
        """
        config = RetryWithCircuitBreakerConfig(
            max_retries=self._retry_config.max_retries,
            base_delay=self._retry_config.base_delay,
            max_delay=self._retry_config.max_delay,
            jitter=self._retry_config.jitter,
            retry_on_status_codes=self._retry_config.retry_on_status_codes,
            retry_on_exceptions=self._retry_config.retry_on_exceptions,
            on_retry=self._retry_config.on_retry,
            enable_circuit_breaker=self._enable_circuit_breaker,
            circuit_breaker_name=self._circuit_breaker_name,
        )
        return await retry_async_with_circuit_breaker(
            func, *args, config=config, **kwargs
        )


# Convenience function for wrapping httpx calls with circuit breaker
async def http_request_with_retry_and_circuit_breaker(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    config: Optional[RetryWithCircuitBreakerConfig] = None,
    **kwargs: Any,
) -> httpx.Response:
    """
    Make an HTTP request with retry and circuit breaker protection.

    This is a convenience function for making resilient HTTP requests
    with both retry logic and circuit breaker protection.

    Args:
        client: httpx.AsyncClient instance
        method: HTTP method (GET, POST, etc.)
        url: Request URL
        config: Combined retry and circuit breaker configuration
        **kwargs: Additional arguments for the request

    Returns:
        httpx.Response

    Raises:
        CircuitOpenError: If circuit is open
        Last exception if all retries exhausted
    """

    async def make_request() -> httpx.Response:
        response = await client.request(method, url, **kwargs)
        response.raise_for_status()
        return response

    return await retry_async_with_circuit_breaker(make_request, config=config)


# Export public API
__all__ = [
    # Core retry types and functions
    "RetryConfig",
    "calculate_delay",
    "is_retryable_exception",
    "retry_async",
    "retry_sync",
    "with_retry",
    "RetryableClient",
    "http_request_with_retry",
    "RETRYABLE_STATUS_CODES",
    "RETRYABLE_EXCEPTIONS",
    # Circuit breaker integration
    "CIRCUIT_BREAKER_AVAILABLE",
    "RetryWithCircuitBreakerConfig",
    "retry_async_with_circuit_breaker",
    "with_retry_and_circuit_breaker",
    "ResilientClient",
    "http_request_with_retry_and_circuit_breaker",
]
