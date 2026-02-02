# -*- coding: utf-8 -*-
"""
Resilient SDK Wrappers

Provides resilient wrappers around SDK calls with:
- Automatic retry with exponential backoff
- Circuit breaker for failure detection
- Rate limiting
- Proper error categorization
- Structured logging and telemetry

Usage:
    from core.resilient_sdk import ResilientResearchEngine

    engine = ResilientResearchEngine()
    result = await engine.search("query")  # Auto-retries on transient failures
"""

import asyncio
import inspect
import logging
import time
from typing import Any, Dict, List, Optional, Callable, TypeVar, Union
from functools import wraps
from dataclasses import dataclass

# Import our utilities
from utils import (
    configure_logging,
    get_logger,
    log_context,
    set_context,
    # Errors
    UnleashError,
    ExaError,
    FirecrawlError,
    NetworkError,
    TimeoutError as UnleashTimeoutError,
    RateLimitError,
    ServiceUnavailableError,
    AuthenticationError,
    ValidationError,
    wrap_exception,
    ErrorAggregator,
    RETRIABLE_EXCEPTIONS,
    NON_RETRIABLE_EXCEPTIONS,
    # Status helpers
    ok, fail, warn, safe_print,
)

# Import resilience patterns
from .resilience import (
    CircuitBreaker,
    CircuitOpenError,
    RetryPolicy,
    RetryStrategy,
    RateLimiter,
    RateLimitExceeded,
    ResilienceHandler,
    ResilienceConfig,
    TelemetryCollector,
    HealthChecker,
    HealthCheck,
    HealthStatus,
)

logger = get_logger(__name__)

T = TypeVar("T")


# =============================================================================
# SDK-Specific Error Handlers
# =============================================================================

def categorize_sdk_error(exc: Exception, sdk_name: str) -> UnleashError:
    """
    Categorize an SDK exception into our error hierarchy.

    Args:
        exc: Original exception from SDK
        sdk_name: Name of the SDK (exa, firecrawl, etc.)

    Returns:
        Appropriate UnleashError subclass
    """
    error_str = str(exc).lower()
    exc_type = type(exc).__name__.lower()

    # Already one of our errors
    if isinstance(exc, UnleashError):
        return exc

    # Network/connection errors
    if any(term in error_str for term in ["connection", "network", "socket", "dns"]):
        return NetworkError(str(exc), source=sdk_name, cause=exc)

    # Timeout errors
    if any(term in error_str for term in ["timeout", "timed out", "deadline"]):
        return UnleashTimeoutError(str(exc), source=sdk_name, cause=exc)

    # Rate limiting
    if any(term in error_str for term in ["rate limit", "429", "too many requests", "throttl"]):
        # Try to extract retry-after
        retry_after = None
        if "retry" in error_str and "after" in error_str:
            try:
                import re
                match = re.search(r'(\d+)\s*(?:s|sec|seconds?)', error_str)
                if match:
                    retry_after = float(match.group(1))
            except:
                pass
        return RateLimitError(str(exc), source=sdk_name, cause=exc, retry_after=retry_after)

    # Authentication errors
    if any(term in error_str for term in ["unauthorized", "401", "api key", "invalid key", "auth"]):
        return AuthenticationError(str(exc), source=sdk_name, cause=exc)

    # Service unavailable
    if any(term in error_str for term in ["503", "502", "unavailable", "overloaded", "maintenance"]):
        return ServiceUnavailableError(str(exc), source=sdk_name, cause=exc)

    # Validation errors
    if any(term in error_str for term in ["invalid", "validation", "400", "bad request", "malformed"]):
        return ValidationError(str(exc), source=sdk_name, cause=exc)

    # SDK-specific default
    if sdk_name == "exa":
        return ExaError(str(exc), cause=exc)
    elif sdk_name == "firecrawl":
        return FirecrawlError(str(exc), cause=exc)
    else:
        return wrap_exception(exc, source=sdk_name)


# =============================================================================
# Resilient Call Wrapper
# =============================================================================

@dataclass
class ResilientCallConfig:
    """Configuration for resilient SDK calls."""
    # Retry settings
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL

    # Circuit breaker settings
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: float = 30.0

    # Rate limiting
    rate_limit_per_second: float = 10.0
    rate_bucket_size: int = 50

    # Timeouts
    operation_timeout: float = 60.0


class ResilientSDKCaller:
    """
    Wraps SDK calls with resilience patterns.

    Provides:
    - Automatic retry with exponential backoff
    - Circuit breaker to fail fast
    - Rate limiting
    - Error categorization
    - Telemetry collection
    """

    def __init__(
        self,
        sdk_name: str,
        config: Optional[ResilientCallConfig] = None,
    ):
        """
        Initialize resilient caller for an SDK.

        Args:
            sdk_name: Name of the SDK (for error categorization)
            config: Resilience configuration
        """
        self.sdk_name = sdk_name
        self.config = config or ResilientCallConfig()

        # Initialize resilience components
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.circuit_failure_threshold,
            recovery_timeout=self.config.circuit_recovery_timeout,
            excluded_exceptions={ValidationError, AuthenticationError},
        )

        self.retry_policy = RetryPolicy(
            max_retries=self.config.max_retries,
            base_delay=self.config.base_delay,
            max_delay=self.config.max_delay,
            strategy=self.config.retry_strategy,
            jitter=True,
            retriable_exceptions=RETRIABLE_EXCEPTIONS,
            non_retriable_exceptions=NON_RETRIABLE_EXCEPTIONS,
        )

        self.rate_limiter = RateLimiter(
            tokens_per_second=self.config.rate_limit_per_second,
            bucket_size=self.config.rate_bucket_size,
        )

        self.telemetry = TelemetryCollector(service_name=f"unleash-{sdk_name}")

        # Health tracking
        self._last_success_time: Optional[float] = None
        self._consecutive_failures: int = 0

    async def call(
        self,
        func: Callable[..., T],
        *args,
        operation_name: Optional[str] = None,
        skip_circuit: bool = False,
        skip_rate_limit: bool = False,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> T:
        """
        Execute an SDK call with full resilience.

        Args:
            func: Function to call (sync or async)
            *args: Positional arguments
            operation_name: Name for logging/telemetry
            skip_circuit: Skip circuit breaker check
            skip_rate_limit: Skip rate limiting
            timeout: Override operation timeout
            **kwargs: Keyword arguments

        Returns:
            Result from the function

        Raises:
            UnleashError: Categorized error on failure
            CircuitOpenError: Circuit breaker is open
            RateLimitExceeded: Rate limit exceeded
        """
        op_name = operation_name or func.__name__
        timeout_val = timeout or self.config.operation_timeout

        # Rate limiting
        if not skip_rate_limit:
            if not await self.rate_limiter.acquire():
                self.telemetry.record_counter(
                    "rate_limited",
                    labels={"sdk": self.sdk_name, "operation": op_name},
                )
                raise RateLimitError(
                    f"Rate limit exceeded for {self.sdk_name}.{op_name}",
                    source=self.sdk_name,
                )

        # Execute with circuit breaker and retry
        with self.telemetry.span(op_name, attributes={"sdk": self.sdk_name}) as span:
            start_time = time.time()

            try:
                if skip_circuit:
                    result = await self._execute_with_retry(func, args, kwargs, op_name, timeout_val)
                else:
                    async with self.circuit_breaker:
                        result = await self._execute_with_retry(func, args, kwargs, op_name, timeout_val)

                # Record success
                duration = time.time() - start_time
                self._last_success_time = time.time()
                self._consecutive_failures = 0

                self.telemetry.record_counter(
                    "operation_success",
                    labels={"sdk": self.sdk_name, "operation": op_name},
                )
                self.telemetry.record_histogram(
                    "operation_duration_ms",
                    duration * 1000,
                    labels={"sdk": self.sdk_name, "operation": op_name},
                )

                span.attributes["duration_ms"] = duration * 1000
                span.attributes["retries"] = len(self.retry_policy.attempts) - 1

                logger.debug(f"{self.sdk_name}.{op_name} succeeded in {duration:.2f}s")
                return result

            except CircuitOpenError:
                self.telemetry.record_counter(
                    "circuit_open",
                    labels={"sdk": self.sdk_name, "operation": op_name},
                )
                raise

            except Exception as e:
                duration = time.time() - start_time
                self._consecutive_failures += 1

                # Categorize the error
                categorized = categorize_sdk_error(e, self.sdk_name)
                categorized.operation = op_name

                self.telemetry.record_counter(
                    "operation_failure",
                    labels={
                        "sdk": self.sdk_name,
                        "operation": op_name,
                        "error_code": categorized.code,
                    },
                )

                span.attributes["error"] = categorized.code
                span.attributes["duration_ms"] = duration * 1000

                logger.warning(
                    f"{self.sdk_name}.{op_name} failed after {duration:.2f}s: {categorized.code}"
                )
                raise categorized from e

    async def _execute_with_retry(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        op_name: str,
        timeout: float,
    ) -> Any:
        """Execute function with retry logic."""

        async def wrapped():
            try:
                # Handle both sync and async functions
                if inspect.iscoroutinefunction(func):
                    return await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=timeout
                    )
                else:
                    # Run sync function in executor with timeout
                    loop = asyncio.get_event_loop()
                    return await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: func(*args, **kwargs)),
                        timeout=timeout
                    )
            except asyncio.TimeoutError:
                raise UnleashTimeoutError(
                    f"Operation {op_name} timed out after {timeout}s",
                    source=self.sdk_name,
                )
            except Exception as e:
                # Re-categorize for retry decision
                raise categorize_sdk_error(e, self.sdk_name) from e

        return await self.retry_policy.execute(wrapped)

    def get_health(self) -> HealthCheck:
        """Get health status of this SDK connection."""
        if self.circuit_breaker.state.value == "open":
            return HealthCheck(
                name=self.sdk_name,
                status=HealthStatus.UNHEALTHY,
                message="Circuit breaker is open",
                metadata={
                    "circuit_state": self.circuit_breaker.state.value,
                    "consecutive_failures": self._consecutive_failures,
                },
            )

        if self._consecutive_failures >= 3:
            return HealthCheck(
                name=self.sdk_name,
                status=HealthStatus.DEGRADED,
                message=f"{self._consecutive_failures} consecutive failures",
                metadata={
                    "circuit_state": self.circuit_breaker.state.value,
                    "consecutive_failures": self._consecutive_failures,
                },
            )

        return HealthCheck(
            name=self.sdk_name,
            status=HealthStatus.HEALTHY,
            metadata={
                "circuit_state": self.circuit_breaker.state.value,
                "last_success": self._last_success_time,
            },
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "sdk": self.sdk_name,
            "circuit_breaker": {
                "state": self.circuit_breaker.state.value,
                "stats": {
                    "total_calls": self.circuit_breaker.stats.total_calls,
                    "failed_calls": self.circuit_breaker.stats.failed_calls,
                    "rejected_calls": self.circuit_breaker.stats.rejected_calls,
                    "failure_rate": self.circuit_breaker.stats.failure_rate,
                },
            },
            "retry_policy": self.retry_policy.get_stats(),
            "rate_limiter": {
                "available_tokens": self.rate_limiter.available_tokens,
                "total_requests": self.rate_limiter.stats.total_requests,
                "rejection_rate": self.rate_limiter.stats.rejection_rate,
            },
            "health": self.get_health().status.value,
            "telemetry": self.telemetry.get_all_metrics(),
        }


# =============================================================================
# Resilient Research Engine
# =============================================================================

class ResilientResearchEngine:
    """
    Research engine with full resilience capabilities.

    Wraps the base ResearchEngine with:
    - Automatic retry on transient failures
    - Circuit breakers per SDK
    - Rate limiting
    - Proper error categorization
    - Health monitoring
    - Telemetry collection
    """

    def __init__(
        self,
        firecrawl_config: Optional[ResilientCallConfig] = None,
        exa_config: Optional[ResilientCallConfig] = None,
    ):
        """Initialize with optional per-SDK configuration."""
        # Import research engine
        from .research_engine import ResearchEngine

        # Get base engine
        self.engine = ResearchEngine.get_instance()

        # Create resilient callers for each SDK
        self.firecrawl_caller = ResilientSDKCaller(
            "firecrawl",
            config=firecrawl_config or ResilientCallConfig(
                rate_limit_per_second=5.0,  # Firecrawl has stricter limits
                max_retries=3,
            ),
        )

        self.exa_caller = ResilientSDKCaller(
            "exa",
            config=exa_config or ResilientCallConfig(
                rate_limit_per_second=10.0,
                max_retries=3,
            ),
        )

        # Health checker
        self.health_checker = HealthChecker()
        self.health_checker.register("firecrawl", lambda: self.firecrawl_caller.get_health())
        self.health_checker.register("exa", lambda: self.exa_caller.get_health())

    # =========================================================================
    # Firecrawl Methods (Resilient)
    # =========================================================================

    async def scrape(self, url: str, **kwargs) -> Dict[str, Any]:
        """Resilient scrape with retry and circuit breaker."""
        return await self.firecrawl_caller.call(
            self.engine.scrape,
            url,
            operation_name="scrape",
            **kwargs,
        )

    async def crawl(self, url: str, **kwargs) -> Dict[str, Any]:
        """Resilient crawl with retry and circuit breaker."""
        return await self.firecrawl_caller.call(
            self.engine.crawl,
            url,
            operation_name="crawl",
            timeout=300.0,  # Crawls can be slow
            **kwargs,
        )

    async def batch_scrape(self, urls: List[str], **kwargs) -> Dict[str, Any]:
        """Resilient batch scrape."""
        return await self.firecrawl_caller.call(
            self.engine.batch_scrape,
            urls,
            operation_name="batch_scrape",
            timeout=300.0,
            **kwargs,
        )

    async def map_site(self, url: str, **kwargs) -> Dict[str, Any]:
        """Resilient site mapping."""
        return await self.firecrawl_caller.call(
            self.engine.map_site,
            url,
            operation_name="map_site",
            **kwargs,
        )

    async def firecrawl_search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Resilient Firecrawl search."""
        return await self.firecrawl_caller.call(
            self.engine.firecrawl_search,
            query,
            operation_name="firecrawl_search",
            **kwargs,
        )

    # =========================================================================
    # Exa Methods (Resilient)
    # =========================================================================

    async def search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Resilient Exa search."""
        return await self.exa_caller.call(
            self.engine.exa_search,
            query,
            operation_name="exa_search",
            **kwargs,
        )

    async def search_and_contents(self, query: str, **kwargs) -> Dict[str, Any]:
        """Resilient search with content extraction."""
        return await self.exa_caller.call(
            self.engine.exa_search_and_contents,
            query,
            operation_name="exa_search_and_contents",
            **kwargs,
        )

    async def find_similar(self, url: str, **kwargs) -> Dict[str, Any]:
        """Resilient similarity search."""
        return await self.exa_caller.call(
            self.engine.exa_find_similar,
            url,
            operation_name="exa_find_similar",
            **kwargs,
        )

    async def get_contents(self, ids_or_urls: List[str], **kwargs) -> Dict[str, Any]:
        """Resilient content extraction."""
        return await self.exa_caller.call(
            self.engine.exa_get_contents,
            ids_or_urls,
            operation_name="exa_get_contents",
            **kwargs,
        )

    async def answer(self, query: str, **kwargs) -> Dict[str, Any]:
        """Resilient answer API."""
        return await self.exa_caller.call(
            self.engine.exa_answer,
            query,
            operation_name="exa_answer",
            **kwargs,
        )

    # =========================================================================
    # Health & Stats
    # =========================================================================

    async def health_check(self) -> Dict[str, HealthCheck]:
        """Check health of all SDK connections."""
        return await self.health_checker.check_all()

    def get_overall_health(self) -> HealthStatus:
        """Get overall health status."""
        return self.health_checker.get_overall_status()

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all SDKs."""
        return {
            "firecrawl": self.firecrawl_caller.get_stats(),
            "exa": self.exa_caller.get_stats(),
            "overall_health": self.get_overall_health().value,
        }

    def reset_circuits(self):
        """Reset all circuit breakers."""
        self.firecrawl_caller.circuit_breaker.reset()
        self.exa_caller.circuit_breaker.reset()
        logger.info("All circuit breakers reset")


# =============================================================================
# Convenience Functions
# =============================================================================

def get_resilient_engine() -> ResilientResearchEngine:
    """Get or create resilient research engine singleton."""
    if not hasattr(get_resilient_engine, "_instance"):
        get_resilient_engine._instance = ResilientResearchEngine()
    return get_resilient_engine._instance


async def resilient_search(query: str, **kwargs) -> Dict[str, Any]:
    """Convenience function for resilient Exa search."""
    engine = get_resilient_engine()
    return await engine.search(query, **kwargs)


async def resilient_scrape(url: str, **kwargs) -> Dict[str, Any]:
    """Convenience function for resilient Firecrawl scrape."""
    engine = get_resilient_engine()
    return await engine.scrape(url, **kwargs)
