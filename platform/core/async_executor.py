"""
Async Executor for Unleashed Platform

Provides high-performance async execution patterns for SDK operations:
- Parallel execution with semaphore-based rate limiting
- Batch processing with configurable concurrency
- Circuit breaker for fault tolerance
- Retry logic with exponential backoff
- Task queuing and prioritization

These utilities enable efficient orchestration of multiple SDK
adapters while respecting API rate limits and handling failures gracefully.

V43 Enhancement: Opik observability tracing for circuit breaker state changes.
"""

import asyncio
import time
import functools
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging

logger = logging.getLogger(__name__)

# V43: Opik tracing for circuit breaker observability
V43_OPIK_AVAILABLE = False
_opik_module: Any = None
try:
    import opik as _opik_module  # type: ignore
    V43_OPIK_AVAILABLE = True
except ImportError:
    pass

T = TypeVar("T")
R = TypeVar("R")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class TaskPriority(Enum):
    """Task priority levels for queue ordering."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class ExecutionResult(Generic[T]):
    """Result from an async execution."""
    success: bool
    value: Optional[T] = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult(Generic[T]):
    """Result from batch execution."""
    results: List[ExecutionResult[T]]
    total_time: float
    success_count: int
    failure_count: int

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5      # Failures before opening
    recovery_timeout: float = 30.0  # Seconds before half-open
    half_open_max_calls: int = 3    # Test calls in half-open state
    success_threshold: int = 2      # Successes to close from half-open


class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance.

    Prevents cascading failures by:
    1. Tracking failure rate
    2. Opening circuit when threshold exceeded
    3. Allowing test requests after recovery timeout
    4. Closing circuit when service recovers

    V43 Enhancement: Opik observability tracing for state transitions.
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        self._lock = asyncio.Lock()
        # V43: Metrics for observability
        self._total_requests = 0
        self._total_failures = 0
        self._total_rejections = 0  # Requests rejected due to open circuit
        self._state_transitions: List[Dict[str, Any]] = []

    def _trace_state_change(self, from_state: CircuitState, to_state: CircuitState, reason: str = ""):
        """V43: Record state transition for observability."""
        transition = {
            "timestamp": time.time(),
            "from_state": from_state.value,
            "to_state": to_state.value,
            "reason": reason,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
        }
        self._state_transitions.append(transition)
        # Keep only last 50 transitions
        if len(self._state_transitions) > 50:
            self._state_transitions = self._state_transitions[-50:]

        # V43: Opik tracing if available
        if V43_OPIK_AVAILABLE and _opik_module:
            try:
                _opik_module.log_metadata({
                    "circuit_breaker": self.name,
                    "event": "state_transition",
                    "from_state": from_state.value,
                    "to_state": to_state.value,
                    "reason": reason,
                    "failure_count": self.failure_count,
                    "total_requests": self._total_requests,
                    "total_rejections": self._total_rejections,
                })
            except Exception:
                pass  # Opik not in active context

    def get_metrics(self) -> Dict[str, Any]:
        """V43: Get circuit breaker metrics for observability."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self._total_requests,
            "total_failures": self._total_failures,
            "total_rejections": self._total_rejections,
            "last_failure_time": self.last_failure_time,
            "recent_transitions": self._state_transitions[-10:],
        }

    async def can_execute(self) -> bool:
        """Check if execution is allowed."""
        async with self._lock:
            self._total_requests += 1

            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self.last_failure_time:
                    elapsed = time.time() - self.last_failure_time
                    if elapsed >= self.config.recovery_timeout:
                        old_state = self.state
                        self.state = CircuitState.HALF_OPEN
                        self.half_open_calls = 0
                        logger.info(f"Circuit {self.name} transitioning to HALF_OPEN")
                        # V43: Trace state transition
                        self._trace_state_change(old_state, self.state, f"Recovery timeout elapsed ({elapsed:.1f}s)")
                        return True
                # V43: Track rejection
                self._total_rejections += 1
                return False

            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls < self.config.half_open_max_calls:
                    self.half_open_calls += 1
                    return True
                # V43: Track rejection
                self._total_rejections += 1
                return False

            return False

    async def record_success(self):
        """Record a successful execution."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    old_state = self.state
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info(f"Circuit {self.name} closed (recovered)")
                    # V43: Trace state transition
                    self._trace_state_change(old_state, self.state, "Recovered after success threshold")
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = max(0, self.failure_count - 1)

    async def record_failure(self, error: Exception):
        """Record a failed execution."""
        async with self._lock:
            self.failure_count += 1
            self._total_failures += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                # Immediate transition back to open
                old_state = self.state
                self.state = CircuitState.OPEN
                self.success_count = 0
                logger.warning(f"Circuit {self.name} reopened due to failure: {error}")
                # V43: Trace state transition
                self._trace_state_change(old_state, self.state, f"Failure in half-open: {type(error).__name__}")

            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    old_state = self.state
                    self.state = CircuitState.OPEN
                    logger.warning(
                        f"Circuit {self.name} opened after {self.failure_count} failures"
                    )
                    # V43: Trace state transition
                    self._trace_state_change(old_state, self.state, f"Threshold exceeded: {self.failure_count} failures")


class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Allows smooth rate limiting with bursting capability.
    """

    def __init__(
        self,
        rate: float,           # Tokens per second
        max_tokens: int = 10,  # Maximum burst capacity
    ):
        self.rate = rate
        self.max_tokens = max_tokens
        self.tokens = max_tokens
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens, waiting if necessary.

        Returns wait time in seconds.
        """
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update

            # Refill tokens
            self.tokens = min(
                self.max_tokens,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0

            # Calculate wait time
            needed = tokens - self.tokens
            wait_time = needed / self.rate

            await asyncio.sleep(wait_time)

            self.tokens = 0
            self.last_update = time.time()

            return wait_time


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: float = 0.1  # Random factor to prevent thundering herd


class AsyncExecutor:
    """
    High-performance async executor for SDK operations.

    Features:
    - Parallel execution with configurable concurrency
    - Rate limiting per adapter
    - Circuit breakers for fault tolerance
    - Automatic retries with backoff
    - Priority-based task queuing
    """

    def __init__(
        self,
        max_concurrency: int = 10,
        default_rate_limit: float = 10.0,  # Requests per second
    ):
        """
        Initialize async executor.

        Args:
            max_concurrency: Maximum parallel executions
            default_rate_limit: Default rate limit for adapters
        """
        self.max_concurrency = max_concurrency
        self.default_rate_limit = default_rate_limit
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._rate_limiters: Dict[str, RateLimiter] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._retry_config = RetryConfig()

        # Metrics
        self._total_executions = 0
        self._total_successes = 0
        self._total_failures = 0

    def set_rate_limit(self, adapter_name: str, rate: float, burst: int = 10):
        """Set rate limit for a specific adapter."""
        self._rate_limiters[adapter_name] = RateLimiter(rate, burst)

    def set_circuit_breaker(
        self,
        adapter_name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        """Configure circuit breaker for an adapter."""
        self._circuit_breakers[adapter_name] = CircuitBreaker(adapter_name, config)

    def set_retry_config(self, config: RetryConfig):
        """Set retry configuration."""
        self._retry_config = config

    async def _get_rate_limiter(self, adapter_name: str) -> RateLimiter:
        """Get or create rate limiter for adapter."""
        if adapter_name not in self._rate_limiters:
            self._rate_limiters[adapter_name] = RateLimiter(self.default_rate_limit)
        return self._rate_limiters[adapter_name]

    async def _execute_with_retry(
        self,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args,
        adapter_name: str = "default",
        **kwargs,
    ) -> ExecutionResult[T]:
        """
        Execute function with retry logic.

        Args:
            func: Async function to execute
            adapter_name: Name of adapter (for circuit breaker)
            *args, **kwargs: Arguments to pass to function

        Returns:
            ExecutionResult with value or error
        """
        start_time = time.time()
        retries = 0
        last_error: Optional[Exception] = None

        # Check circuit breaker
        circuit = self._circuit_breakers.get(adapter_name)
        if circuit and not await circuit.can_execute():
            return ExecutionResult(
                success=False,
                error=Exception(f"Circuit breaker open for {adapter_name}"),
                metadata={"circuit_state": "open"},
            )

        # Rate limiting
        rate_limiter = await self._get_rate_limiter(adapter_name)
        await rate_limiter.acquire()

        while retries <= self._retry_config.max_retries:
            try:
                async with self._semaphore:
                    result = await func(*args, **kwargs)

                # Record success
                if circuit:
                    await circuit.record_success()

                self._total_executions += 1
                self._total_successes += 1

                return ExecutionResult(
                    success=True,
                    value=result,
                    execution_time=time.time() - start_time,
                    retries=retries,
                )

            except Exception as e:
                last_error = e
                retries += 1

                if retries <= self._retry_config.max_retries:
                    # Calculate delay with exponential backoff
                    delay = min(
                        self._retry_config.initial_delay * (
                            self._retry_config.exponential_base ** (retries - 1)
                        ),
                        self._retry_config.max_delay,
                    )
                    # Add jitter
                    delay *= (1 + self._retry_config.jitter * (2 * asyncio.get_event_loop().time() % 1 - 0.5))

                    logger.warning(
                        f"Retry {retries}/{self._retry_config.max_retries} "
                        f"for {adapter_name} after {delay:.2f}s: {e}"
                    )
                    await asyncio.sleep(delay)

        # Record failure
        if circuit:
            await circuit.record_failure(last_error)

        self._total_executions += 1
        self._total_failures += 1

        return ExecutionResult(
            success=False,
            error=last_error,
            execution_time=time.time() - start_time,
            retries=retries - 1,
        )

    async def execute(
        self,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args,
        adapter_name: str = "default",
        **kwargs,
    ) -> ExecutionResult[T]:
        """
        Execute a single async function with all protections.

        Args:
            func: Async function to execute
            adapter_name: Name of adapter for rate limiting/circuit breaker
            *args, **kwargs: Arguments to pass to function

        Returns:
            ExecutionResult with outcome
        """
        return await self._execute_with_retry(
            func, *args, adapter_name=adapter_name, **kwargs
        )

    async def execute_batch(
        self,
        tasks: List[Callable[..., Coroutine[Any, Any, T]]],
        adapter_name: str = "default",
        fail_fast: bool = False,
    ) -> BatchResult[T]:
        """
        Execute multiple tasks in parallel.

        Args:
            tasks: List of async callables
            adapter_name: Adapter name for all tasks
            fail_fast: Stop on first failure if True

        Returns:
            BatchResult with all outcomes
        """
        start_time = time.time()
        results: List[ExecutionResult[T]] = []

        if fail_fast:
            # Execute one at a time, stop on failure
            for task in tasks:
                result = await self.execute(task, adapter_name=adapter_name)
                results.append(result)
                if not result.success:
                    break
        else:
            # Execute all in parallel
            async def wrapped_task(t):
                return await self.execute(t, adapter_name=adapter_name)

            results = await asyncio.gather(
                *[wrapped_task(task) for task in tasks]
            )

        success_count = sum(1 for r in results if r.success)
        failure_count = len(results) - success_count

        return BatchResult(
            results=results,
            total_time=time.time() - start_time,
            success_count=success_count,
            failure_count=failure_count,
        )

    async def execute_map(
        self,
        func: Callable[[T], Coroutine[Any, Any, R]],
        items: List[T],
        adapter_name: str = "default",
        chunk_size: Optional[int] = None,
    ) -> List[ExecutionResult[R]]:
        """
        Map a function over items with controlled concurrency.

        Args:
            func: Async function to apply to each item
            items: List of items to process
            adapter_name: Adapter name
            chunk_size: Process in chunks (None = all parallel)

        Returns:
            List of results in same order as items
        """
        if chunk_size is None:
            chunk_size = len(items)

        all_results: List[ExecutionResult[R]] = []

        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            tasks = [functools.partial(func, item) for item in chunk]

            chunk_results = await asyncio.gather(
                *[self.execute(task, adapter_name=adapter_name) for task in tasks]
            )
            all_results.extend(chunk_results)

        return all_results

    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics."""
        return {
            "total_executions": self._total_executions,
            "total_successes": self._total_successes,
            "total_failures": self._total_failures,
            "success_rate": (
                self._total_successes / self._total_executions
                if self._total_executions > 0 else 0.0
            ),
            "circuit_breakers": {
                name: cb.state.value
                for name, cb in self._circuit_breakers.items()
            },
        }


class TaskQueue:
    """
    Priority-based async task queue.

    Enables ordered processing of tasks based on priority.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._queues: Dict[TaskPriority, deque] = {
            priority: deque() for priority in TaskPriority
        }
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Event()

    async def put(
        self,
        task: Callable[..., Coroutine],
        priority: TaskPriority = TaskPriority.NORMAL,
        *args,
        **kwargs,
    ):
        """Add task to queue with priority."""
        async with self._lock:
            total_size = sum(len(q) for q in self._queues.values())
            if total_size >= self.max_size:
                raise asyncio.QueueFull(f"Queue full ({self.max_size} tasks)")

            self._queues[priority].append((task, args, kwargs))
            self._not_empty.set()

    async def get(self) -> tuple:
        """Get highest priority task."""
        while True:
            async with self._lock:
                # Check queues in priority order (highest first)
                for priority in reversed(TaskPriority):
                    if self._queues[priority]:
                        task, args, kwargs = self._queues[priority].popleft()

                        # Clear event if all queues empty
                        if all(len(q) == 0 for q in self._queues.values()):
                            self._not_empty.clear()

                        return task, args, kwargs

            # Wait for new tasks
            await self._not_empty.wait()

    def size(self) -> int:
        """Get total queue size."""
        return sum(len(q) for q in self._queues.values())

    async def process(
        self,
        executor: AsyncExecutor,
        adapter_name: str = "default",
        num_workers: int = 5,
    ):
        """
        Process queue with multiple workers.

        Args:
            executor: AsyncExecutor to use
            adapter_name: Adapter name for execution
            num_workers: Number of concurrent workers
        """
        async def worker():
            while True:
                task, args, kwargs = await self.get()
                await executor.execute(
                    task, *args, adapter_name=adapter_name, **kwargs
                )

        workers = [asyncio.create_task(worker()) for _ in range(num_workers)]
        await asyncio.gather(*workers)


# Convenience decorator for retryable functions
def retryable(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
):
    """
    Decorator to make an async function retryable.

    Usage:
        @retryable(max_retries=5, initial_delay=0.5)
        async def my_api_call():
            ...
    """
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_error: Optional[Exception] = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        delay = initial_delay * (exponential_base ** attempt)
                        await asyncio.sleep(delay)

            raise last_error

        return wrapper
    return decorator


# Convenience decorator for rate-limited functions
def rate_limited(
    rate: float = 10.0,
    burst: int = 10,
):
    """
    Decorator to rate limit an async function.

    Usage:
        @rate_limited(rate=5.0, burst=10)
        async def my_api_call():
            ...
    """
    limiter = RateLimiter(rate, burst)

    def decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            await limiter.acquire()
            return await func(*args, **kwargs)

        return wrapper
    return decorator


def get_executor(**kwargs) -> AsyncExecutor:
    """Get configured async executor."""
    return AsyncExecutor(**kwargs)
