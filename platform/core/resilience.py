#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pydantic>=2.0.0",
# ]
# ///
"""
UAP Resilience Module - Production-grade error handling and observability.

Implements battle-tested patterns for production systems:
- Circuit Breaker: Fail-fast to prevent cascade failures
- Retry Policy: Exponential backoff with jitter
- Rate Limiter: Token bucket for throughput control
- Backpressure: Load shedding when overwhelmed
- Health Checks: Component health aggregation
- Telemetry: Metrics, spans, structured logging

Based on:
- Netflix Hystrix patterns
- AWS Well-Architected Framework
- Site Reliability Engineering (Google SRE book)
- Temporal.io resilience patterns

Usage:
    from resilience import CircuitBreaker, RetryPolicy, RateLimiter

    # Circuit breaker for external service
    breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

    # Retry with exponential backoff
    policy = RetryPolicy(max_retries=3, base_delay=1.0, max_delay=30.0)

    # Rate limiting
    limiter = RateLimiter(tokens_per_second=10.0, bucket_size=100)
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitState(str, Enum):
    """Circuit breaker states following the standard pattern."""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Failing fast, requests rejected
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitStats:
    """Statistics for circuit breaker monitoring."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    state_changes: int = 0
    time_in_open: float = 0.0

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 1.0
        return self.successful_calls / self.total_calls


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Prevents cascade failures by failing fast when a service is unhealthy.

    States:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Service is failing, reject requests immediately
    - HALF_OPEN: Testing recovery, allow limited requests

    Example:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

        async with breaker:
            result = await external_service_call()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        excluded_exceptions: Optional[Set[type]] = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            success_threshold: Successes in half-open to close circuit
            recovery_timeout: Seconds before attempting recovery
            half_open_max_calls: Max concurrent calls in half-open state
            excluded_exceptions: Exceptions that don't count as failures
        """
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.excluded_exceptions = excluded_exceptions or set()

        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._last_state_change = time.time()
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    @property
    def stats(self) -> CircuitStats:
        """Circuit statistics."""
        return self._stats

    def _should_attempt_reset(self) -> bool:
        """Check if we should try to reset from OPEN state."""
        if self._state != CircuitState.OPEN:
            return False
        elapsed = time.time() - self._last_state_change
        return elapsed >= self.recovery_timeout

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        if old_state == new_state:
            return

        # Track time in open state
        if old_state == CircuitState.OPEN:
            self._stats.time_in_open += time.time() - self._last_state_change

        self._state = new_state
        self._last_state_change = time.time()
        self._stats.state_changes += 1

        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._stats.consecutive_successes = 0

        logger.info(
            f"Circuit breaker state change: {old_state.value} -> {new_state.value}"
        )

    async def __aenter__(self) -> "CircuitBreaker":
        """Enter async context - check if request should proceed."""
        async with self._lock:
            # Check for state transition
            if self._should_attempt_reset():
                self._transition_to(CircuitState.HALF_OPEN)

            if self._state == CircuitState.OPEN:
                self._stats.rejected_calls += 1
                raise CircuitOpenError(
                    f"Circuit is OPEN, {self.recovery_timeout - (time.time() - self._last_state_change):.1f}s until retry"
                )

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    self._stats.rejected_calls += 1
                    raise CircuitOpenError("Circuit is HALF_OPEN, max test calls reached")
                self._half_open_calls += 1

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        """Exit async context - record result."""
        del exc_val, exc_tb  # Unused but required by protocol
        async with self._lock:
            self._stats.total_calls += 1

            if exc_type is None:
                # Success
                self._record_success()
            elif exc_type in self.excluded_exceptions:
                # Excluded exception - don't count as failure
                pass
            else:
                # Failure
                self._record_failure()

        return False  # Don't suppress exceptions

    def _record_success(self) -> None:
        """Record a successful call."""
        self._stats.successful_calls += 1
        self._stats.last_success_time = time.time()
        self._stats.consecutive_failures = 0
        self._stats.consecutive_successes += 1

        if self._state == CircuitState.HALF_OPEN:
            if self._stats.consecutive_successes >= self.success_threshold:
                self._transition_to(CircuitState.CLOSED)

    def _record_failure(self) -> None:
        """Record a failed call."""
        self._stats.failed_calls += 1
        self._stats.last_failure_time = time.time()
        self._stats.consecutive_successes = 0
        self._stats.consecutive_failures += 1

        if self._state == CircuitState.CLOSED:
            if self._stats.consecutive_failures >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open reopens the circuit
            self._transition_to(CircuitState.OPEN)

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._last_state_change = time.time()
        self._half_open_calls = 0


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# =============================================================================
# Retry Policy
# =============================================================================

class RetryStrategy(str, Enum):
    """Retry backoff strategies."""
    FIXED = "fixed"              # Same delay each retry
    LINEAR = "linear"            # Delay increases linearly
    EXPONENTIAL = "exponential"  # Delay doubles each retry
    DECORRELATED_JITTER = "decorrelated_jitter"  # AWS-style jitter


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""
    attempt_number: int
    delay_before: float
    exception: Optional[Exception] = None
    duration: float = 0.0
    succeeded: bool = False


class RetryPolicy:
    """
    Configurable retry policy with multiple backoff strategies.

    Supports:
    - Exponential backoff with optional jitter
    - Maximum delay capping
    - Retriable exception filtering
    - Retry budget tracking

    Example:
        policy = RetryPolicy(
            max_retries=3,
            base_delay=1.0,
            max_delay=30.0,
            strategy=RetryStrategy.EXPONENTIAL,
            jitter=True,
        )

        result = await policy.execute(async_function, arg1, arg2)
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        jitter: bool = True,
        jitter_factor: float = 0.5,
        retriable_exceptions: Optional[Set[type]] = None,
        non_retriable_exceptions: Optional[Set[type]] = None,
    ):
        """
        Initialize retry policy.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay between retries
            strategy: Backoff strategy to use
            jitter: Add randomness to prevent thundering herd
            jitter_factor: Amount of jitter (0.0-1.0)
            retriable_exceptions: Only retry these exceptions (default: all)
            non_retriable_exceptions: Never retry these exceptions
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.strategy = strategy
        self.jitter = jitter
        self.jitter_factor = jitter_factor
        self.retriable_exceptions = retriable_exceptions
        self.non_retriable_exceptions = non_retriable_exceptions or set()

        self._attempts: List[RetryAttempt] = []
        self._total_retries = 0
        self._successful_first_attempts = 0
        self._last_delay = base_delay  # For decorrelated jitter

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number."""
        if self.strategy == RetryStrategy.FIXED:
            delay = self.base_delay
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.base_delay * (attempt + 1)
        elif self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.base_delay * (2 ** attempt)
        elif self.strategy == RetryStrategy.DECORRELATED_JITTER:
            # AWS-style decorrelated jitter
            delay = min(self.max_delay, random.uniform(self.base_delay, self._last_delay * 3))
            self._last_delay = delay
            return delay
        else:
            delay = self.base_delay

        # Apply jitter
        if self.jitter and self.strategy != RetryStrategy.DECORRELATED_JITTER:
            jitter_range = delay * self.jitter_factor
            delay = delay + random.uniform(-jitter_range, jitter_range)

        # Cap at max delay
        return min(max(0, delay), self.max_delay)

    def is_retriable(self, exception: Exception) -> bool:
        """Check if an exception should be retried."""
        exc_type = type(exception)

        # Check non-retriable first
        if exc_type in self.non_retriable_exceptions:
            return False

        # If retriable set is specified, only retry those
        if self.retriable_exceptions:
            return exc_type in self.retriable_exceptions

        # Default: retry all
        return True

    async def execute(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from successful function call

        Raises:
            Last exception if all retries exhausted
        """
        self._attempts = []
        last_exception: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            delay = self.calculate_delay(attempt) if attempt > 0 else 0.0

            attempt_info = RetryAttempt(
                attempt_number=attempt,
                delay_before=delay,
            )

            if delay > 0:
                logger.debug(f"Retry attempt {attempt}, waiting {delay:.2f}s")
                await asyncio.sleep(delay)

            start_time = time.time()
            try:
                if inspect.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                attempt_info.duration = time.time() - start_time
                attempt_info.succeeded = True
                self._attempts.append(attempt_info)

                if attempt == 0:
                    self._successful_first_attempts += 1
                else:
                    self._total_retries += attempt

                return result

            except Exception as e:
                attempt_info.duration = time.time() - start_time
                attempt_info.exception = e
                self._attempts.append(attempt_info)
                last_exception = e

                if not self.is_retriable(e):
                    logger.warning(f"Non-retriable exception: {type(e).__name__}")
                    raise

                if attempt < self.max_retries:
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {e}"
                    )

        # All retries exhausted
        self._total_retries += self.max_retries
        logger.error(f"All {self.max_retries + 1} attempts failed")
        raise last_exception  # type: ignore

    @property
    def attempts(self) -> List[RetryAttempt]:
        """Get retry attempts from last execution."""
        return self._attempts

    def get_stats(self) -> Dict[str, Any]:
        """Get retry statistics."""
        return {
            "total_retries": self._total_retries,
            "successful_first_attempts": self._successful_first_attempts,
            "max_retries_configured": self.max_retries,
            "strategy": self.strategy.value,
        }


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimitStrategy(str, Enum):
    """Rate limiting algorithms."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"


@dataclass
class RateLimitStats:
    """Rate limiter statistics."""
    total_requests: int = 0
    allowed_requests: int = 0
    rejected_requests: int = 0
    current_tokens: float = 0.0
    last_refill: float = 0.0

    @property
    def rejection_rate(self) -> float:
        """Calculate rejection rate."""
        if self.total_requests == 0:
            return 0.0
        return self.rejected_requests / self.total_requests


class RateLimiter:
    """
    Rate limiter using token bucket algorithm.

    Controls throughput by limiting requests per time unit.
    Tokens are consumed for each request and refilled over time.

    Example:
        limiter = RateLimiter(tokens_per_second=10.0, bucket_size=100)

        if await limiter.acquire():
            # Process request
            pass
        else:
            # Rate limited
            raise RateLimitExceeded()
    """

    def __init__(
        self,
        tokens_per_second: float = 10.0,
        bucket_size: int = 100,
        initial_tokens: Optional[float] = None,
    ):
        """
        Initialize rate limiter.

        Args:
            tokens_per_second: Token refill rate
            bucket_size: Maximum tokens in bucket
            initial_tokens: Starting tokens (default: bucket_size)
        """
        self.tokens_per_second = tokens_per_second
        self.bucket_size = bucket_size
        self._tokens = float(initial_tokens if initial_tokens is not None else bucket_size)
        self._last_refill = time.time()
        self._stats = RateLimitStats(
            current_tokens=self._tokens,
            last_refill=self._last_refill,
        )
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill
        tokens_to_add = elapsed * self.tokens_per_second
        self._tokens = min(self.bucket_size, self._tokens + tokens_to_add)
        self._last_refill = now
        self._stats.current_tokens = self._tokens
        self._stats.last_refill = now

    async def acquire(self, tokens: float = 1.0, wait: bool = False) -> bool:
        """
        Attempt to acquire tokens.

        Args:
            tokens: Number of tokens to acquire
            wait: If True, wait until tokens available

        Returns:
            True if tokens acquired, False if rate limited
        """
        async with self._lock:
            self._refill()
            self._stats.total_requests += 1

            if self._tokens >= tokens:
                self._tokens -= tokens
                self._stats.current_tokens = self._tokens
                self._stats.allowed_requests += 1
                return True

            if not wait:
                self._stats.rejected_requests += 1
                return False

        # Wait for tokens
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self._stats.current_tokens = self._tokens
                    self._stats.allowed_requests += 1
                    return True

            # Calculate wait time
            tokens_needed = tokens - self._tokens
            wait_time = tokens_needed / self.tokens_per_second
            await asyncio.sleep(min(wait_time, 0.1))

    async def __aenter__(self) -> "RateLimiter":
        """Acquire token on context entry."""
        acquired = await self.acquire(wait=True)
        if not acquired:
            raise RateLimitExceeded("Could not acquire rate limit token")
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        """No-op on exit."""
        del exc_type, exc_val, exc_tb  # Required by protocol but unused

    @property
    def stats(self) -> RateLimitStats:
        """Get rate limiter statistics."""
        return self._stats

    @property
    def available_tokens(self) -> float:
        """Get current available tokens."""
        return self._tokens


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    pass


# =============================================================================
# Backpressure & Load Shedding
# =============================================================================

class LoadLevel(str, Enum):
    """System load levels for backpressure."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"
    OVERLOADED = "overloaded"


@dataclass
class BackpressureConfig:
    """Configuration for backpressure handling."""
    queue_size_threshold: int = 100
    latency_threshold_ms: float = 1000.0
    cpu_threshold: float = 0.8
    memory_threshold: float = 0.85
    shed_percentage_elevated: float = 0.1
    shed_percentage_high: float = 0.3
    shed_percentage_critical: float = 0.5
    shed_percentage_overloaded: float = 0.8


class BackpressureManager:
    """
    Manages backpressure and load shedding.

    Monitors system load and sheds traffic when overwhelmed
    to prevent complete system failure.

    Example:
        manager = BackpressureManager(config)

        if manager.should_accept_request(priority=1):
            process_request()
        else:
            return_503_service_unavailable()
    """

    def __init__(self, config: Optional[BackpressureConfig] = None):
        """Initialize backpressure manager."""
        self.config = config or BackpressureConfig()
        self._current_load = LoadLevel.NORMAL
        self._queue_depth = 0
        self._recent_latencies: deque[float] = deque(maxlen=100)
        self._requests_accepted = 0
        self._requests_shed = 0
        self._lock = asyncio.Lock()

    def update_metrics(
        self,
        queue_depth: Optional[int] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Update load metrics."""
        if queue_depth is not None:
            self._queue_depth = queue_depth
        if latency_ms is not None:
            self._recent_latencies.append(latency_ms)

        self._calculate_load_level()

    def _calculate_load_level(self) -> None:
        """Calculate current load level from metrics."""
        # Check queue depth
        queue_ratio = self._queue_depth / self.config.queue_size_threshold

        # Check latency (if we have samples)
        latency_ratio = 0.0
        if self._recent_latencies:
            avg_latency = sum(self._recent_latencies) / len(self._recent_latencies)
            latency_ratio = avg_latency / self.config.latency_threshold_ms

        # Use worst of the metrics
        load_ratio = max(queue_ratio, latency_ratio)

        if load_ratio >= 2.0:
            self._current_load = LoadLevel.OVERLOADED
        elif load_ratio >= 1.5:
            self._current_load = LoadLevel.CRITICAL
        elif load_ratio >= 1.0:
            self._current_load = LoadLevel.HIGH
        elif load_ratio >= 0.7:
            self._current_load = LoadLevel.ELEVATED
        else:
            self._current_load = LoadLevel.NORMAL

    def _get_shed_probability(self) -> float:
        """Get probability of shedding a request."""
        if self._current_load == LoadLevel.NORMAL:
            return 0.0
        elif self._current_load == LoadLevel.ELEVATED:
            return self.config.shed_percentage_elevated
        elif self._current_load == LoadLevel.HIGH:
            return self.config.shed_percentage_high
        elif self._current_load == LoadLevel.CRITICAL:
            return self.config.shed_percentage_critical
        else:  # OVERLOADED
            return self.config.shed_percentage_overloaded

    def should_accept_request(self, priority: int = 0) -> bool:
        """
        Determine if a request should be accepted.

        Args:
            priority: Request priority (higher = more important)

        Returns:
            True if request should be processed
        """
        shed_prob = self._get_shed_probability()

        # Adjust for priority (high priority reduces shed probability)
        adjusted_prob = shed_prob * (1.0 / (1.0 + priority * 0.2))

        if random.random() < adjusted_prob:
            self._requests_shed += 1
            return False

        self._requests_accepted += 1
        return True

    @property
    def load_level(self) -> LoadLevel:
        """Current load level."""
        return self._current_load

    def get_stats(self) -> Dict[str, Any]:
        """Get backpressure statistics."""
        return {
            "load_level": self._current_load.value,
            "queue_depth": self._queue_depth,
            "avg_latency_ms": (
                sum(self._recent_latencies) / len(self._recent_latencies)
                if self._recent_latencies else 0.0
            ),
            "requests_accepted": self._requests_accepted,
            "requests_shed": self._requests_shed,
            "shed_rate": (
                self._requests_shed / (self._requests_accepted + self._requests_shed)
                if (self._requests_accepted + self._requests_shed) > 0 else 0.0
            ),
        }


# =============================================================================
# Health Checks
# =============================================================================

class HealthStatus(str, Enum):
    """Component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: Optional[str] = None
    latency_ms: Optional[float] = None
    last_check: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """
    Aggregates health checks from multiple components.

    Provides overall system health status based on component health.

    Example:
        checker = HealthChecker()
        checker.register("database", db_health_check)
        checker.register("cache", cache_health_check)

        overall_health = await checker.check_all()
    """

    def __init__(self):
        """Initialize health checker."""
        self._checks: Dict[str, Callable[[], Union[HealthCheck, Any]]] = {}
        self._last_results: Dict[str, HealthCheck] = {}
        self._check_interval = 30.0  # seconds

    def register(
        self,
        name: str,
        check_func: Callable[[], Union[HealthCheck, Any]],
    ) -> None:
        """Register a health check function."""
        self._checks[name] = check_func

    def unregister(self, name: str) -> None:
        """Unregister a health check."""
        self._checks.pop(name, None)
        self._last_results.pop(name, None)

    async def check_one(self, name: str) -> HealthCheck:
        """Run a single health check."""
        if name not in self._checks:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNKNOWN,
                message="Health check not registered",
            )

        check_func = self._checks[name]
        start = time.time()

        try:
            if inspect.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()

            latency_ms = (time.time() - start) * 1000

            if isinstance(result, HealthCheck):
                result.latency_ms = latency_ms
                result.last_check = time.time()
                self._last_results[name] = result
                return result

            # Convert boolean or other result
            status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
            check = HealthCheck(
                name=name,
                status=status,
                latency_ms=latency_ms,
            )
            self._last_results[name] = check
            return check

        except Exception as e:
            check = HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=(time.time() - start) * 1000,
            )
            self._last_results[name] = check
            return check

    async def check_all(self) -> Dict[str, HealthCheck]:
        """Run all health checks concurrently."""
        tasks = [self.check_one(name) for name in self._checks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        result_dict = {}
        for name, result in zip(self._checks.keys(), results):
            if isinstance(result, Exception):
                result_dict[name] = HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(result),
                )
            else:
                result_dict[name] = result

        return result_dict

    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        if not self._last_results:
            return HealthStatus.UNKNOWN

        statuses = [r.status for r in self._last_results.values()]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNKNOWN

    @property
    def last_results(self) -> Dict[str, HealthCheck]:
        """Get last health check results."""
        return self._last_results


# =============================================================================
# Telemetry & Observability
# =============================================================================

class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """A single metric value."""
    name: str
    type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    unit: Optional[str] = None


@dataclass
class Span:
    """A tracing span for distributed tracing."""
    trace_id: str
    span_id: str
    parent_id: Optional[str]
    operation: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "ok"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000


class TelemetryCollector:
    """
    Collects metrics, traces, and logs for observability.

    Provides a unified interface for:
    - Metrics: Counters, gauges, histograms
    - Tracing: Distributed trace spans
    - Structured logging: Contextual log events

    Example:
        telemetry = TelemetryCollector()

        # Record a metric
        telemetry.record_counter("api_requests", 1, {"endpoint": "/users"})

        # Create a span
        with telemetry.span("process_request") as span:
            span.set_attribute("user_id", "123")
            # ... do work ...
    """

    def __init__(self, service_name: str = "uap"):
        """Initialize telemetry collector."""
        self.service_name = service_name
        self._metrics: Dict[str, List[Metric]] = {}
        self._spans: List[Span] = []
        self._active_spans: Dict[str, Span] = {}
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._max_retained_metrics = 10000
        self._max_retained_spans = 1000

    def record_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a counter metric (monotonically increasing)."""
        key = self._metric_key(name, labels or {})
        self._counters[key] = self._counters.get(key, 0) + value

        metric = Metric(
            name=name,
            type=MetricType.COUNTER,
            value=self._counters[key],
            labels=labels or {},
        )
        self._store_metric(metric)

    def record_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a gauge metric (can go up or down)."""
        key = self._metric_key(name, labels or {})
        self._gauges[key] = value

        metric = Metric(
            name=name,
            type=MetricType.GAUGE,
            value=value,
            labels=labels or {},
        )
        self._store_metric(metric)

    def record_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a histogram observation."""
        key = self._metric_key(name, labels or {})
        if key not in self._histograms:
            self._histograms[key] = []
        self._histograms[key].append(value)

        # Keep only recent observations
        if len(self._histograms[key]) > 1000:
            self._histograms[key] = self._histograms[key][-1000:]

        metric = Metric(
            name=name,
            type=MetricType.HISTOGRAM,
            value=value,
            labels=labels or {},
        )
        self._store_metric(metric)

    def _metric_key(self, name: str, labels: Dict[str, str]) -> str:
        """Create a unique key for a metric with labels."""
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}:{label_str}"

    def _store_metric(self, metric: Metric) -> None:
        """Store a metric."""
        if metric.name not in self._metrics:
            self._metrics[metric.name] = []
        self._metrics[metric.name].append(metric)

        # Prune old metrics
        if len(self._metrics[metric.name]) > self._max_retained_metrics:
            self._metrics[metric.name] = self._metrics[metric.name][-self._max_retained_metrics:]

    def start_span(
        self,
        operation: str,
        parent_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """Start a new tracing span."""
        import uuid

        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())[:16]

        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_id=parent_id,
            operation=operation,
            start_time=time.time(),
            attributes=attributes or {},
        )

        self._active_spans[span_id] = span
        return span

    def end_span(self, span: Span, status: str = "ok") -> None:
        """End a tracing span."""
        span.end_time = time.time()
        span.status = status

        self._active_spans.pop(span.span_id, None)
        self._spans.append(span)

        # Prune old spans
        if len(self._spans) > self._max_retained_spans:
            self._spans = self._spans[-self._max_retained_spans:]

    class SpanContext:
        """Context manager for spans."""

        def __init__(self, collector: "TelemetryCollector", span: Span):
            self._collector = collector
            self._span = span

        def __enter__(self) -> Span:
            return self._span

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
            del exc_tb  # Required by protocol but unused
            status = "error" if exc_type else "ok"
            if exc_val:
                self._span.events.append({
                    "name": "exception",
                    "timestamp": time.time(),
                    "attributes": {
                        "exception.type": exc_type.__name__ if exc_type else None,
                        "exception.message": str(exc_val),
                    },
                })
            self._collector.end_span(self._span, status)
            return False

    def span(
        self,
        operation: str,
        parent_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> SpanContext:
        """Create a span context manager."""
        span = self.start_span(operation, parent_id, attributes)
        return self.SpanContext(self, span)

    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a histogram."""
        values: List[float] = []
        for key, vals in self._histograms.items():
            if key.startswith(f"{name}:") or key == name:
                values.extend(vals)

        if not values:
            return {}

        sorted_vals = sorted(values)
        n = len(sorted_vals)

        return {
            "count": n,
            "min": sorted_vals[0],
            "max": sorted_vals[-1],
            "mean": sum(sorted_vals) / n,
            "p50": sorted_vals[int(n * 0.5)],
            "p90": sorted_vals[int(n * 0.9)],
            "p99": sorted_vals[min(int(n * 0.99), n - 1)],
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metric values."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                name: self.get_histogram_stats(name)
                for name in set(k.split(":")[0] for k in self._histograms.keys())
            },
        }

    def get_recent_spans(self, limit: int = 100) -> List[Span]:
        """Get recent completed spans."""
        return self._spans[-limit:]


# =============================================================================
# Composite Resilience Handler
# =============================================================================

class ResilienceConfig(BaseModel):
    """Configuration for composite resilience handler."""
    # Circuit breaker
    circuit_failure_threshold: int = Field(default=5, ge=1)
    circuit_success_threshold: int = Field(default=3, ge=1)
    circuit_recovery_timeout: float = Field(default=30.0, gt=0)

    # Retry
    retry_max_attempts: int = Field(default=3, ge=0)
    retry_base_delay: float = Field(default=1.0, gt=0)
    retry_max_delay: float = Field(default=60.0, gt=0)
    retry_jitter: bool = True

    # Rate limiting
    rate_tokens_per_second: float = Field(default=10.0, gt=0)
    rate_bucket_size: int = Field(default=100, ge=1)

    # Backpressure
    enable_backpressure: bool = True
    backpressure_queue_threshold: int = Field(default=100, ge=1)


class ResilienceHandler:
    """
    Composite handler combining all resilience patterns.

    Provides a unified interface for:
    - Circuit breaking
    - Retry with backoff
    - Rate limiting
    - Backpressure
    - Health checking
    - Telemetry

    Example:
        handler = ResilienceHandler(config)

        result = await handler.execute(
            "api_call",
            async_function,
            arg1, arg2,
        )
    """

    def __init__(self, config: Optional[ResilienceConfig] = None):
        """Initialize resilience handler."""
        self.config = config or ResilienceConfig()

        # Initialize components
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.circuit_failure_threshold,
            success_threshold=self.config.circuit_success_threshold,
            recovery_timeout=self.config.circuit_recovery_timeout,
        )

        self.retry_policy = RetryPolicy(
            max_retries=self.config.retry_max_attempts,
            base_delay=self.config.retry_base_delay,
            max_delay=self.config.retry_max_delay,
            jitter=self.config.retry_jitter,
        )

        self.rate_limiter = RateLimiter(
            tokens_per_second=self.config.rate_tokens_per_second,
            bucket_size=self.config.rate_bucket_size,
        )

        self.backpressure = BackpressureManager(
            BackpressureConfig(
                queue_size_threshold=self.config.backpressure_queue_threshold,
            )
        ) if self.config.enable_backpressure else None

        self.health_checker = HealthChecker()
        self.telemetry = TelemetryCollector()

    async def execute(
        self,
        operation_name: str,
        func: Callable[..., Any],
        *args: Any,
        priority: int = 0,
        skip_circuit: bool = False,
        skip_rate_limit: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Execute an operation with full resilience handling.

        Args:
            operation_name: Name for telemetry/logging
            func: Function to execute
            *args: Positional arguments
            priority: Request priority for backpressure
            skip_circuit: Skip circuit breaker
            skip_rate_limit: Skip rate limiting
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        # Backpressure check
        if self.backpressure and not self.backpressure.should_accept_request(priority):
            self.telemetry.record_counter(
                "request_shed",
                labels={"operation": operation_name},
            )
            raise BackpressureError("Request shed due to high load")

        # Rate limiting
        if not skip_rate_limit:
            if not await self.rate_limiter.acquire():
                self.telemetry.record_counter(
                    "rate_limited",
                    labels={"operation": operation_name},
                )
                raise RateLimitExceeded("Rate limit exceeded")

        # Execute with circuit breaker and retry
        with self.telemetry.span(operation_name) as span:
            try:
                if skip_circuit:
                    exec_result = await self.retry_policy.execute(func, *args, **kwargs)
                else:
                    async with self.circuit_breaker:
                        exec_result = await self.retry_policy.execute(func, *args, **kwargs)

                self.telemetry.record_counter(
                    "operation_success",
                    labels={"operation": operation_name},
                )
                span.attributes["retries"] = len(self.retry_policy.attempts) - 1
                return exec_result  # type: ignore[possibly-undefined]

            except CircuitOpenError:
                self.telemetry.record_counter(
                    "circuit_open",
                    labels={"operation": operation_name},
                )
                raise
            except Exception as e:
                self.telemetry.record_counter(
                    "operation_failure",
                    labels={"operation": operation_name, "error": type(e).__name__},
                )
                raise

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive resilience statistics."""
        stats = {
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
            "health": self.health_checker.get_overall_status().value,
            "telemetry": self.telemetry.get_all_metrics(),
        }

        if self.backpressure:
            stats["backpressure"] = self.backpressure.get_stats()

        return stats


class BackpressureError(Exception):
    """Raised when request is shed due to backpressure."""
    pass


# =============================================================================
# Factory Functions
# =============================================================================

def create_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
) -> CircuitBreaker:
    """Create a circuit breaker with common defaults."""
    return CircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
    )


def create_retry_policy(
    max_retries: int = 3,
    base_delay: float = 1.0,
    exponential: bool = True,
) -> RetryPolicy:
    """Create a retry policy with common defaults."""
    return RetryPolicy(
        max_retries=max_retries,
        base_delay=base_delay,
        strategy=RetryStrategy.EXPONENTIAL if exponential else RetryStrategy.FIXED,
        jitter=True,
    )


def create_rate_limiter(
    requests_per_second: float = 10.0,
    burst_size: int = 100,
) -> RateLimiter:
    """Create a rate limiter with common defaults."""
    return RateLimiter(
        tokens_per_second=requests_per_second,
        bucket_size=burst_size,
    )


def create_resilience_handler(
    config: Optional[ResilienceConfig] = None,
) -> ResilienceHandler:
    """Create a resilience handler with all patterns."""
    return ResilienceHandler(config)


def create_telemetry(service_name: str = "uap") -> TelemetryCollector:
    """Create a telemetry collector."""
    return TelemetryCollector(service_name)
