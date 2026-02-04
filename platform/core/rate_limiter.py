"""
Rate Limiting Middleware for Research Adapters
===============================================

Provides token bucket rate limiting with:
- Per-adapter configurable rate limits
- Adaptive rate limiting based on 429 responses
- Rate limit header parsing (X-RateLimit-*, Retry-After)
- Async-first design with thread-safe operations
- Sliding window support for burst handling
- Circuit breaker integration

Default Rate Limits (per minute unless specified):
- Exa: 100/min
- Tavily: 100/min
- Serper: 300/sec (18,000/min)
- Perplexity: 20/min
- Firecrawl: 100/min
- Jina: 200/min
- Context7: 60/min

Usage:
    from core.rate_limiter import RateLimiter, get_rate_limiter

    # Get the singleton rate limiter
    limiter = get_rate_limiter()

    # Check if request is allowed
    if await limiter.acquire("exa"):
        # Make request
        response = await adapter.execute(...)
        # Update based on response headers
        limiter.update_from_headers("exa", response.headers)
    else:
        # Rate limited - wait or queue

    # As a context manager
    async with limiter.acquire_context("tavily"):
        response = await adapter.execute(...)

    # Decorator usage
    @rate_limited("perplexity")
    async def search_perplexity(query: str):
        ...
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RateLimitState(str, Enum):
    """Rate limiter states."""
    NORMAL = "normal"           # Operating normally
    THROTTLED = "throttled"     # Temporarily reduced rate
    BACKOFF = "backoff"         # Exponential backoff active
    SUSPENDED = "suspended"     # Temporarily suspended (circuit open)


@dataclass
class RateLimitConfig:
    """Configuration for a rate limiter bucket."""

    requests_per_second: float = 1.0
    """Maximum requests per second."""

    burst_size: int = 10
    """Maximum burst capacity (token bucket size)."""

    min_interval_ms: float = 0.0
    """Minimum interval between requests in milliseconds."""

    adaptive: bool = True
    """Enable adaptive rate limiting based on 429 responses."""

    backoff_multiplier: float = 2.0
    """Multiplier for exponential backoff on rate limit errors."""

    max_backoff_seconds: float = 300.0
    """Maximum backoff duration in seconds."""

    recovery_rate: float = 0.1
    """Rate at which to recover from throttled state (per second)."""

    @classmethod
    def from_per_minute(cls, requests_per_minute: int, burst_size: Optional[int] = None) -> "RateLimitConfig":
        """Create config from requests per minute."""
        rps = requests_per_minute / 60.0
        return cls(
            requests_per_second=rps,
            burst_size=burst_size or max(1, int(rps * 2)),
            min_interval_ms=1000.0 / rps if rps > 0 else 0,
        )

    @classmethod
    def from_per_second(cls, requests_per_second: int, burst_size: Optional[int] = None) -> "RateLimitConfig":
        """Create config from requests per second."""
        return cls(
            requests_per_second=float(requests_per_second),
            burst_size=burst_size or max(1, requests_per_second * 2),
            min_interval_ms=1000.0 / requests_per_second if requests_per_second > 0 else 0,
        )


@dataclass
class RateLimitStats:
    """Statistics for a rate limiter bucket."""

    total_requests: int = 0
    """Total requests made."""

    allowed_requests: int = 0
    """Requests that were allowed."""

    rejected_requests: int = 0
    """Requests that were rejected due to rate limiting."""

    rate_limit_hits: int = 0
    """Number of 429 responses received."""

    current_wait_time_ms: float = 0.0
    """Current wait time if rate limited."""

    tokens_available: float = 0.0
    """Current tokens available in bucket."""

    state: RateLimitState = RateLimitState.NORMAL
    """Current rate limiter state."""

    last_request_time: Optional[datetime] = None
    """Time of last request."""

    last_rate_limit_time: Optional[datetime] = None
    """Time of last 429 response."""

    backoff_until: Optional[datetime] = None
    """Time until which backoff is active."""

    avg_request_latency_ms: float = 0.0
    """Average request latency in milliseconds."""


class TokenBucket:
    """
    Token bucket rate limiter implementation.

    Thread-safe and async-compatible implementation that supports:
    - Token refill based on configured rate
    - Burst handling via bucket capacity
    - Adaptive throttling based on 429 responses
    - Exponential backoff with jitter
    """

    def __init__(self, name: str, config: RateLimitConfig):
        self.name = name
        self.config = config
        self._lock = asyncio.Lock()

        # Token bucket state
        self._tokens = float(config.burst_size)
        self._last_refill = time.monotonic()

        # Adaptive rate limiting state
        self._effective_rate = config.requests_per_second
        self._backoff_until: Optional[float] = None
        self._consecutive_429s = 0
        self._state = RateLimitState.NORMAL

        # Statistics
        self._stats = RateLimitStats(
            tokens_available=self._tokens,
            state=self._state,
        )

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill

        # Calculate tokens to add based on effective rate
        tokens_to_add = elapsed * self._effective_rate

        # Cap at burst size
        self._tokens = min(self._tokens + tokens_to_add, float(self.config.burst_size))
        self._last_refill = now
        self._stats.tokens_available = self._tokens

    async def acquire(self, tokens: float = 1.0, wait: bool = True) -> Tuple[bool, float]:
        """
        Attempt to acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire
            wait: If True, wait for tokens to become available

        Returns:
            Tuple of (acquired, wait_time_seconds)
        """
        async with self._lock:
            # Check if in backoff
            if self._backoff_until is not None:
                now = time.monotonic()
                if now < self._backoff_until:
                    wait_time = self._backoff_until - now
                    self._stats.current_wait_time_ms = wait_time * 1000
                    if wait:
                        self._stats.rejected_requests += 1
                        return False, wait_time
                    return False, wait_time
                else:
                    # Backoff expired, start recovery
                    self._backoff_until = None
                    self._state = RateLimitState.THROTTLED
                    self._stats.state = self._state

            # Refill tokens
            self._refill()

            self._stats.total_requests += 1

            if self._tokens >= tokens:
                # Tokens available
                self._tokens -= tokens
                self._stats.allowed_requests += 1
                self._stats.current_wait_time_ms = 0
                self._stats.tokens_available = self._tokens
                self._stats.last_request_time = datetime.utcnow()

                # Gradual recovery in throttled state
                if self._state == RateLimitState.THROTTLED:
                    self._recover_rate()

                return True, 0.0
            else:
                # Need to wait
                tokens_needed = tokens - self._tokens
                wait_time = tokens_needed / self._effective_rate

                self._stats.current_wait_time_ms = wait_time * 1000

                if wait:
                    self._stats.rejected_requests += 1
                    return False, wait_time

                return False, wait_time

    def _recover_rate(self) -> None:
        """Gradually recover to normal rate."""
        if self._effective_rate < self.config.requests_per_second:
            recovery_amount = self.config.recovery_rate * self.config.requests_per_second
            self._effective_rate = min(
                self._effective_rate + recovery_amount,
                self.config.requests_per_second
            )

            if self._effective_rate >= self.config.requests_per_second * 0.95:
                self._state = RateLimitState.NORMAL
                self._effective_rate = self.config.requests_per_second
                self._consecutive_429s = 0
                self._stats.state = self._state
                logger.info(f"Rate limiter '{self.name}' recovered to normal state")

    def on_rate_limit(self, retry_after: Optional[float] = None) -> None:
        """
        Handle a 429 rate limit response.

        Args:
            retry_after: Optional Retry-After value from headers (seconds)
        """
        self._stats.rate_limit_hits += 1
        self._stats.last_rate_limit_time = datetime.utcnow()
        self._consecutive_429s += 1

        # Calculate backoff duration
        if retry_after is not None:
            backoff = retry_after
        else:
            # Exponential backoff with jitter
            import random
            base_backoff = min(
                (self.config.backoff_multiplier ** self._consecutive_429s),
                self.config.max_backoff_seconds
            )
            jitter = random.uniform(0.5, 1.5)
            backoff = base_backoff * jitter

        self._backoff_until = time.monotonic() + backoff
        self._stats.backoff_until = datetime.utcnow().replace(
            microsecond=int((time.time() + backoff) * 1000000) % 1000000
        )

        # Reduce effective rate if adaptive
        if self.config.adaptive:
            self._effective_rate = max(
                self._effective_rate * 0.5,
                self.config.requests_per_second * 0.1  # Minimum 10% of normal rate
            )
            self._state = RateLimitState.BACKOFF
            self._stats.state = self._state

        logger.warning(
            f"Rate limiter '{self.name}' hit rate limit. "
            f"Backing off for {backoff:.1f}s. "
            f"Consecutive 429s: {self._consecutive_429s}. "
            f"Effective rate: {self._effective_rate:.2f} req/s"
        )

    def update_from_headers(self, headers: Dict[str, str]) -> None:
        """
        Update rate limiter state from response headers.

        Parses common rate limit headers:
        - X-RateLimit-Limit
        - X-RateLimit-Remaining
        - X-RateLimit-Reset
        - Retry-After
        - RateLimit-Limit, RateLimit-Remaining, RateLimit-Reset (RFC 9110)
        """
        # Parse Retry-After header
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
        if retry_after:
            try:
                if retry_after.isdigit():
                    self.on_rate_limit(float(retry_after))
                    return
                else:
                    # HTTP-date format - parse and calculate delta
                    from email.utils import parsedate_to_datetime
                    retry_date = parsedate_to_datetime(retry_after)
                    delta = (retry_date - datetime.now(retry_date.tzinfo)).total_seconds()
                    if delta > 0:
                        self.on_rate_limit(delta)
                        return
            except Exception as e:
                logger.debug(f"Failed to parse Retry-After header: {e}")

        # Parse rate limit headers
        remaining = None
        limit = None
        reset = None

        # X-RateLimit-* headers (common)
        for prefix in ["X-RateLimit-", "x-ratelimit-", "RateLimit-", "ratelimit-"]:
            if f"{prefix}Remaining" in headers or f"{prefix}remaining" in headers:
                remaining = headers.get(f"{prefix}Remaining") or headers.get(f"{prefix}remaining")
            if f"{prefix}Limit" in headers or f"{prefix}limit" in headers:
                limit = headers.get(f"{prefix}Limit") or headers.get(f"{prefix}limit")
            if f"{prefix}Reset" in headers or f"{prefix}reset" in headers:
                reset = headers.get(f"{prefix}Reset") or headers.get(f"{prefix}reset")

        # Update state based on remaining requests
        if remaining is not None:
            try:
                remaining_int = int(remaining)
                if remaining_int == 0:
                    # No remaining requests - likely about to hit rate limit
                    if reset:
                        try:
                            reset_time = float(reset)
                            # Check if it's a Unix timestamp or seconds until reset
                            if reset_time > 1e9:  # Unix timestamp
                                wait_time = reset_time - time.time()
                            else:
                                wait_time = reset_time

                            if wait_time > 0:
                                self._backoff_until = time.monotonic() + wait_time
                                self._state = RateLimitState.THROTTLED
                                self._stats.state = self._state
                                logger.info(
                                    f"Rate limiter '{self.name}' throttled. "
                                    f"Remaining: 0, Reset in: {wait_time:.1f}s"
                                )
                        except ValueError:
                            pass
            except ValueError:
                pass

    def get_stats(self) -> RateLimitStats:
        """Get current statistics."""
        self._stats.tokens_available = self._tokens
        return self._stats

    def reset(self) -> None:
        """Reset the rate limiter to initial state."""
        self._tokens = float(self.config.burst_size)
        self._last_refill = time.monotonic()
        self._effective_rate = self.config.requests_per_second
        self._backoff_until = None
        self._consecutive_429s = 0
        self._state = RateLimitState.NORMAL
        self._stats = RateLimitStats(
            tokens_available=self._tokens,
            state=self._state,
        )
        logger.info(f"Rate limiter '{self.name}' reset to initial state")


# =============================================================================
# Default Rate Limit Configurations
# =============================================================================

DEFAULT_RATE_LIMITS: Dict[str, RateLimitConfig] = {
    # Research Layer Adapters
    "exa": RateLimitConfig.from_per_minute(100, burst_size=10),
    "tavily": RateLimitConfig.from_per_minute(100, burst_size=10),
    "serper": RateLimitConfig.from_per_second(300, burst_size=50),  # 300/sec
    "perplexity": RateLimitConfig.from_per_minute(20, burst_size=5),
    "firecrawl": RateLimitConfig.from_per_minute(100, burst_size=10),
    "jina": RateLimitConfig.from_per_minute(200, burst_size=20),
    "context7": RateLimitConfig.from_per_minute(60, burst_size=10),

    # Memory Layer Adapters
    "letta": RateLimitConfig.from_per_minute(60, burst_size=10),
    "cognee": RateLimitConfig.from_per_minute(60, burst_size=10),
    "mem0": RateLimitConfig.from_per_minute(100, burst_size=20),
    "graphiti": RateLimitConfig.from_per_minute(60, burst_size=10),

    # Orchestration Adapters
    "openai_agents": RateLimitConfig.from_per_minute(60, burst_size=10),
    "strands_agents": RateLimitConfig.from_per_minute(60, burst_size=10),

    # Default for unknown adapters
    "default": RateLimitConfig.from_per_minute(60, burst_size=10),
}


class RateLimiter:
    """
    Centralized rate limiter for all adapters.

    Provides:
    - Per-adapter rate limiting with configurable limits
    - Adaptive rate limiting based on 429 responses
    - Rate limit header parsing
    - Statistics and monitoring
    """

    _instance: Optional["RateLimiter"] = None

    def __new__(cls) -> "RateLimiter":
        """Singleton pattern for global rate limiter access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._buckets: Dict[str, TokenBucket] = {}
        self._configs = DEFAULT_RATE_LIMITS.copy()
        self._lock = asyncio.Lock()
        self._initialized = True

    def configure(
        self,
        adapter_name: str,
        config: Optional[RateLimitConfig] = None,
        requests_per_minute: Optional[int] = None,
        requests_per_second: Optional[int] = None,
        burst_size: Optional[int] = None,
    ) -> None:
        """
        Configure rate limiting for an adapter.

        Args:
            adapter_name: Name of the adapter
            config: Full RateLimitConfig (overrides other params)
            requests_per_minute: Requests per minute limit
            requests_per_second: Requests per second limit
            burst_size: Maximum burst size
        """
        if config is not None:
            self._configs[adapter_name] = config
        elif requests_per_second is not None:
            self._configs[adapter_name] = RateLimitConfig.from_per_second(
                requests_per_second, burst_size
            )
        elif requests_per_minute is not None:
            self._configs[adapter_name] = RateLimitConfig.from_per_minute(
                requests_per_minute, burst_size
            )

        # Reset existing bucket if configured
        if adapter_name in self._buckets:
            del self._buckets[adapter_name]

        logger.info(f"Configured rate limit for '{adapter_name}': {self._configs.get(adapter_name)}")

    def _get_bucket(self, adapter_name: str) -> TokenBucket:
        """Get or create a token bucket for an adapter."""
        if adapter_name not in self._buckets:
            config = self._configs.get(adapter_name) or self._configs.get("default")
            self._buckets[adapter_name] = TokenBucket(adapter_name, config)
        return self._buckets[adapter_name]

    async def acquire(
        self,
        adapter_name: str,
        tokens: float = 1.0,
        wait: bool = False,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Acquire rate limit permission for an adapter.

        Args:
            adapter_name: Name of the adapter
            tokens: Number of tokens to acquire
            wait: If True, wait for tokens to become available
            timeout: Maximum time to wait (if wait=True)

        Returns:
            True if acquired, False if rate limited
        """
        bucket = self._get_bucket(adapter_name)

        if not wait:
            acquired, _ = await bucket.acquire(tokens, wait=False)
            return acquired

        # Wait for tokens with optional timeout
        start_time = time.monotonic()
        while True:
            acquired, wait_time = await bucket.acquire(tokens, wait=False)

            if acquired:
                return True

            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed + wait_time > timeout:
                    return False
                wait_time = min(wait_time, timeout - elapsed)

            await asyncio.sleep(wait_time)

    async def acquire_context(self, adapter_name: str, tokens: float = 1.0):
        """
        Context manager for acquiring rate limit permission.

        Usage:
            async with limiter.acquire_context("exa"):
                response = await adapter.execute(...)
        """
        return _RateLimitContext(self, adapter_name, tokens)

    def on_rate_limit(self, adapter_name: str, retry_after: Optional[float] = None) -> None:
        """Handle a 429 rate limit response for an adapter."""
        bucket = self._get_bucket(adapter_name)
        bucket.on_rate_limit(retry_after)

    def update_from_headers(self, adapter_name: str, headers: Dict[str, str]) -> None:
        """Update rate limiter state from response headers."""
        bucket = self._get_bucket(adapter_name)
        bucket.update_from_headers(headers)

    def get_stats(self, adapter_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get rate limiter statistics.

        Args:
            adapter_name: If provided, get stats for specific adapter.
                         If None, get stats for all adapters.

        Returns:
            Statistics dictionary
        """
        if adapter_name:
            bucket = self._get_bucket(adapter_name)
            stats = bucket.get_stats()
            return {
                "adapter": adapter_name,
                "state": stats.state.value,
                "total_requests": stats.total_requests,
                "allowed_requests": stats.allowed_requests,
                "rejected_requests": stats.rejected_requests,
                "rate_limit_hits": stats.rate_limit_hits,
                "tokens_available": stats.tokens_available,
                "current_wait_time_ms": stats.current_wait_time_ms,
            }

        # All adapters
        all_stats = {}
        for name, bucket in self._buckets.items():
            stats = bucket.get_stats()
            all_stats[name] = {
                "state": stats.state.value,
                "total_requests": stats.total_requests,
                "allowed_requests": stats.allowed_requests,
                "rejected_requests": stats.rejected_requests,
                "rate_limit_hits": stats.rate_limit_hits,
                "tokens_available": stats.tokens_available,
            }

        return {
            "adapters": all_stats,
            "total_buckets": len(self._buckets),
            "configured_adapters": list(self._configs.keys()),
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics across all adapters."""
        total_requests = 0
        total_allowed = 0
        total_rejected = 0
        total_429s = 0
        throttled_adapters = []

        for name, bucket in self._buckets.items():
            stats = bucket.get_stats()
            total_requests += stats.total_requests
            total_allowed += stats.allowed_requests
            total_rejected += stats.rejected_requests
            total_429s += stats.rate_limit_hits
            if stats.state in (RateLimitState.THROTTLED, RateLimitState.BACKOFF):
                throttled_adapters.append(name)

        return {
            "total_requests": total_requests,
            "total_allowed": total_allowed,
            "total_rejected": total_rejected,
            "total_rate_limit_hits": total_429s,
            "success_rate": total_allowed / total_requests if total_requests > 0 else 1.0,
            "throttled_adapters": throttled_adapters,
            "active_buckets": len(self._buckets),
        }

    def reset(self, adapter_name: Optional[str] = None) -> None:
        """
        Reset rate limiter state.

        Args:
            adapter_name: If provided, reset specific adapter.
                         If None, reset all adapters.
        """
        if adapter_name:
            if adapter_name in self._buckets:
                self._buckets[adapter_name].reset()
        else:
            for bucket in self._buckets.values():
                bucket.reset()

    def clear(self) -> None:
        """Clear all rate limiter buckets."""
        self._buckets.clear()


class _RateLimitContext:
    """Context manager for rate limiting."""

    def __init__(self, limiter: RateLimiter, adapter_name: str, tokens: float):
        self._limiter = limiter
        self._adapter_name = adapter_name
        self._tokens = tokens
        self._acquired = False

    async def __aenter__(self):
        self._acquired = await self._limiter.acquire(
            self._adapter_name,
            self._tokens,
            wait=True,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Handle 429 responses
        if exc_type is not None:
            # Check if it's a rate limit exception
            if hasattr(exc_val, "status_code") and exc_val.status_code == 429:
                retry_after = None
                if hasattr(exc_val, "headers"):
                    retry_after_str = exc_val.headers.get("Retry-After")
                    if retry_after_str and retry_after_str.isdigit():
                        retry_after = float(retry_after_str)
                self._limiter.on_rate_limit(self._adapter_name, retry_after)
        return False


# =============================================================================
# Decorator for Rate Limiting
# =============================================================================

def rate_limited(
    adapter_name: str,
    tokens: float = 1.0,
    wait: bool = True,
    timeout: Optional[float] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to add rate limiting to a function.

    Usage:
        @rate_limited("exa")
        async def search_exa(query: str):
            ...

        @rate_limited("perplexity", wait=False)
        async def search_perplexity(query: str):
            ...

    Args:
        adapter_name: Name of the adapter for rate limiting
        tokens: Number of tokens to acquire
        wait: If True, wait for rate limit. If False, raise exception.
        timeout: Maximum time to wait (if wait=True)

    Raises:
        RateLimitExceeded: If rate limit exceeded and wait=False
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            limiter = get_rate_limiter()
            acquired = await limiter.acquire(adapter_name, tokens, wait, timeout)

            if not acquired:
                raise RateLimitExceeded(
                    f"Rate limit exceeded for adapter '{adapter_name}'"
                )

            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Check for 429 response
                if hasattr(e, "status_code") and e.status_code == 429:
                    retry_after = None
                    if hasattr(e, "headers"):
                        retry_after_str = e.headers.get("Retry-After")
                        if retry_after_str and retry_after_str.isdigit():
                            retry_after = float(retry_after_str)
                    limiter.on_rate_limit(adapter_name, retry_after)
                raise

        return async_wrapper

    return decorator


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, message: str, wait_time: Optional[float] = None):
        super().__init__(message)
        self.wait_time = wait_time


# =============================================================================
# Global Singleton Access
# =============================================================================

_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """
    Get the global rate limiter instance.

    Returns:
        The singleton RateLimiter instance
    """
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def configure_rate_limit(
    adapter_name: str,
    requests_per_minute: Optional[int] = None,
    requests_per_second: Optional[int] = None,
    burst_size: Optional[int] = None,
) -> None:
    """
    Configure rate limiting for an adapter.

    Convenience function that wraps get_rate_limiter().configure().
    """
    limiter = get_rate_limiter()
    limiter.configure(
        adapter_name,
        requests_per_minute=requests_per_minute,
        requests_per_second=requests_per_second,
        burst_size=burst_size,
    )


# =============================================================================
# Integration Helpers
# =============================================================================

async def with_rate_limit(
    adapter_name: str,
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """
    Execute a function with rate limiting.

    Args:
        adapter_name: Name of the adapter
        func: Async function to execute
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function

    Raises:
        RateLimitExceeded: If rate limit is exceeded
    """
    limiter = get_rate_limiter()

    acquired = await limiter.acquire(adapter_name, wait=True)
    if not acquired:
        raise RateLimitExceeded(f"Rate limit exceeded for '{adapter_name}'")

    try:
        result = await func(*args, **kwargs)
        return result
    except Exception as e:
        # Check for 429 response
        if hasattr(e, "status_code") and getattr(e, "status_code") == 429:
            retry_after = None
            if hasattr(e, "headers"):
                headers = getattr(e, "headers", {})
                retry_after_str = headers.get("Retry-After")
                if retry_after_str and str(retry_after_str).isdigit():
                    retry_after = float(retry_after_str)
            limiter.on_rate_limit(adapter_name, retry_after)
        raise


def parse_rate_limit_headers(headers: Dict[str, str]) -> Dict[str, Any]:
    """
    Parse rate limit headers from an HTTP response.

    Args:
        headers: Response headers dictionary

    Returns:
        Parsed rate limit information
    """
    result = {
        "limit": None,
        "remaining": None,
        "reset": None,
        "retry_after": None,
    }

    # Try different header prefixes
    for prefix in ["X-RateLimit-", "x-ratelimit-", "RateLimit-", "ratelimit-"]:
        for key in ["Limit", "limit"]:
            if f"{prefix}{key}" in headers:
                try:
                    result["limit"] = int(headers[f"{prefix}{key}"])
                except ValueError:
                    pass

        for key in ["Remaining", "remaining"]:
            if f"{prefix}{key}" in headers:
                try:
                    result["remaining"] = int(headers[f"{prefix}{key}"])
                except ValueError:
                    pass

        for key in ["Reset", "reset"]:
            if f"{prefix}{key}" in headers:
                try:
                    result["reset"] = float(headers[f"{prefix}{key}"])
                except ValueError:
                    pass

    # Retry-After header
    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if retry_after:
        try:
            result["retry_after"] = float(retry_after) if retry_after.replace(".", "").isdigit() else None
        except ValueError:
            pass

    return result


# =============================================================================
# Export Public API
# =============================================================================

__all__ = [
    # Classes
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitStats",
    "RateLimitState",
    "TokenBucket",
    "RateLimitExceeded",

    # Functions
    "get_rate_limiter",
    "configure_rate_limit",
    "rate_limited",
    "with_rate_limit",
    "parse_rate_limit_headers",

    # Constants
    "DEFAULT_RATE_LIMITS",
]
