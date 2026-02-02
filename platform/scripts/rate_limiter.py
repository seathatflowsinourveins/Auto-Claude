#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Platform Rate Limiter - Token Bucket and Sliding Window Implementations

Provides rate limiting for the Ultimate Autonomous Platform APIs.
Supports multiple strategies to prevent resource exhaustion.

Algorithms:
- Token Bucket: Smooth rate limiting with burst capacity
- Sliding Window: Precise rate limiting with time window
- Adaptive: Dynamically adjusts based on system load

Usage:
    from rate_limiter import TokenBucketLimiter, RateLimitExceeded

    limiter = TokenBucketLimiter(rate=100, capacity=200)  # 100/sec, 200 burst

    async with limiter.acquire():
        # ... perform rate-limited operation
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[float] = None
    ):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class RateLimiterStats:
    """Statistics for a rate limiter."""
    total_requests: int = 0
    accepted_requests: int = 0
    rejected_requests: int = 0
    current_rate: float = 0.0
    last_request_time: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        """Calculate acceptance rate."""
        if self.total_requests == 0:
            return 1.0
        return self.accepted_requests / self.total_requests


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""

    @abstractmethod
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Attempt to acquire tokens.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if acquired, False otherwise
        """
        pass

    @abstractmethod
    def get_stats(self) -> RateLimiterStats:
        """Get rate limiter statistics."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the rate limiter."""
        pass


@dataclass
class TokenBucketLimiter(RateLimiter):
    """
    Token Bucket rate limiter.

    Allows bursts up to capacity while maintaining average rate.
    Tokens are replenished at a constant rate.

    Args:
        rate: Tokens per second to add to bucket
        capacity: Maximum bucket capacity (burst size)
        initial_tokens: Initial tokens (defaults to capacity)
    """
    rate: float  # tokens per second
    capacity: float  # max tokens
    initial_tokens: Optional[float] = None

    # Internal state
    _tokens: float = field(init=False)
    _last_update: float = field(init=False)
    _stats: RateLimiterStats = field(default_factory=RateLimiterStats, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self):
        self._tokens = self.initial_tokens if self.initial_tokens is not None else self.capacity
        self._last_update = time.monotonic()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_update = now

    async def acquire(self, tokens: int = 1) -> bool:
        """
        Attempt to acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens acquired, False if insufficient tokens
        """
        async with self._lock:
            self._refill()
            self._stats.total_requests += 1
            self._stats.last_request_time = time.time()

            if self._tokens >= tokens:
                self._tokens -= tokens
                self._stats.accepted_requests += 1
                self._stats.current_rate = self._tokens / self.capacity
                return True
            else:
                self._stats.rejected_requests += 1
                return False

    async def acquire_or_raise(self, tokens: int = 1) -> None:
        """
        Acquire tokens or raise RateLimitExceeded.

        Args:
            tokens: Number of tokens to acquire

        Raises:
            RateLimitExceeded: If tokens cannot be acquired
        """
        if not await self.acquire(tokens):
            # Calculate retry_after based on when tokens will be available
            wait_time = (tokens - self._tokens) / self.rate
            raise RateLimitExceeded(
                f"Rate limit exceeded. Retry after {wait_time:.2f}s",
                retry_after=wait_time
            )

    async def __aenter__(self):
        """Context manager entry - acquire one token or raise."""
        await self.acquire_or_raise(1)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass

    def get_stats(self) -> RateLimiterStats:
        """Get current statistics."""
        return self._stats

    def reset(self) -> None:
        """Reset the limiter to initial state."""
        self._tokens = self.initial_tokens if self.initial_tokens is not None else self.capacity
        self._last_update = time.monotonic()
        self._stats = RateLimiterStats()


@dataclass
class SlidingWindowLimiter(RateLimiter):
    """
    Sliding Window rate limiter.

    Counts requests within a sliding time window.
    More precise than token bucket but requires more memory.

    Args:
        max_requests: Maximum requests allowed in window
        window_seconds: Window size in seconds
    """
    max_requests: int
    window_seconds: float

    # Internal state
    _requests: deque = field(default_factory=deque, init=False)
    _stats: RateLimiterStats = field(default_factory=RateLimiterStats, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def _cleanup_old_requests(self) -> None:
        """Remove requests outside the current window."""
        cutoff = time.monotonic() - self.window_seconds
        while self._requests and self._requests[0] < cutoff:
            self._requests.popleft()

    async def acquire(self, tokens: int = 1) -> bool:
        """
        Attempt to record a request.

        Args:
            tokens: Number of request slots to acquire

        Returns:
            True if request allowed, False otherwise
        """
        async with self._lock:
            self._cleanup_old_requests()
            self._stats.total_requests += 1
            self._stats.last_request_time = time.time()

            if len(self._requests) + tokens <= self.max_requests:
                now = time.monotonic()
                for _ in range(tokens):
                    self._requests.append(now)
                self._stats.accepted_requests += 1
                self._stats.current_rate = len(self._requests) / self.max_requests
                return True
            else:
                self._stats.rejected_requests += 1
                return False

    async def acquire_or_raise(self, tokens: int = 1) -> None:
        """Acquire or raise RateLimitExceeded."""
        if not await self.acquire(tokens):
            # Estimate when oldest request will expire
            if self._requests:
                oldest = self._requests[0]
                retry_after = self.window_seconds - (time.monotonic() - oldest)
            else:
                retry_after = self.window_seconds

            raise RateLimitExceeded(
                f"Rate limit exceeded ({self.max_requests} requests per {self.window_seconds}s)",
                retry_after=max(0, retry_after)
            )

    async def __aenter__(self):
        await self.acquire_or_raise(1)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_stats(self) -> RateLimiterStats:
        return self._stats

    def reset(self) -> None:
        self._requests.clear()
        self._stats = RateLimiterStats()


@dataclass
class AdaptiveRateLimiter(RateLimiter):
    """
    Adaptive rate limiter that adjusts based on system response.

    Automatically backs off when errors increase and increases
    throughput when system is healthy.

    Args:
        initial_rate: Starting tokens per second
        min_rate: Minimum rate (floor)
        max_rate: Maximum rate (ceiling)
        capacity: Bucket capacity
        backoff_factor: Rate reduction on error (0.5 = halve)
        recovery_factor: Rate increase on success (1.1 = 10% increase)
    """
    initial_rate: float
    min_rate: float
    max_rate: float
    capacity: float
    backoff_factor: float = 0.5
    recovery_factor: float = 1.1

    # Internal state
    _current_rate: float = field(init=False)
    _bucket: TokenBucketLimiter = field(init=False)
    _consecutive_errors: int = field(default=0, init=False)
    _consecutive_successes: int = field(default=0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self):
        self._current_rate = self.initial_rate
        self._bucket = TokenBucketLimiter(
            rate=self._current_rate,
            capacity=self.capacity
        )

    def _adjust_rate(self, success: bool) -> None:
        """Adjust rate based on operation outcome."""
        if success:
            self._consecutive_errors = 0
            self._consecutive_successes += 1

            # Increase rate after consecutive successes
            if self._consecutive_successes >= 10:
                new_rate = min(self.max_rate, self._current_rate * self.recovery_factor)
                if new_rate != self._current_rate:
                    self._current_rate = new_rate
                    self._bucket.rate = new_rate
                    self._consecutive_successes = 0
        else:
            self._consecutive_successes = 0
            self._consecutive_errors += 1

            # Back off after errors
            if self._consecutive_errors >= 3:
                new_rate = max(self.min_rate, self._current_rate * self.backoff_factor)
                if new_rate != self._current_rate:
                    self._current_rate = new_rate
                    self._bucket.rate = new_rate
                    self._consecutive_errors = 0

    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from adaptive bucket."""
        return await self._bucket.acquire(tokens)

    def record_success(self) -> None:
        """Record a successful operation (for rate adjustment)."""
        self._adjust_rate(True)

    def record_error(self) -> None:
        """Record a failed operation (triggers backoff)."""
        self._adjust_rate(False)

    @property
    def current_rate(self) -> float:
        """Get current rate."""
        return self._current_rate

    def get_stats(self) -> RateLimiterStats:
        stats = self._bucket.get_stats()
        stats.current_rate = self._current_rate / self.max_rate
        return stats

    def reset(self) -> None:
        self._current_rate = self.initial_rate
        self._bucket = TokenBucketLimiter(
            rate=self._current_rate,
            capacity=self.capacity
        )
        self._consecutive_errors = 0
        self._consecutive_successes = 0


class RateLimiterRegistry:
    """
    Registry for managing multiple rate limiters.

    Useful for per-endpoint or per-client rate limiting.
    """

    def __init__(self):
        self._limiters: Dict[str, RateLimiter] = {}
        self._lock = asyncio.Lock()

    def register(self, name: str, limiter: RateLimiter) -> None:
        """Register a rate limiter."""
        self._limiters[name] = limiter

    def get(self, name: str) -> Optional[RateLimiter]:
        """Get a rate limiter by name."""
        return self._limiters.get(name)

    async def acquire(self, name: str, tokens: int = 1) -> bool:
        """Acquire tokens from a named limiter."""
        limiter = self._limiters.get(name)
        if limiter is None:
            return True  # No limiter = unlimited
        return await limiter.acquire(tokens)

    def get_all_stats(self) -> Dict[str, RateLimiterStats]:
        """Get stats for all limiters."""
        return {name: limiter.get_stats() for name, limiter in self._limiters.items()}


# Global registry instance
_global_registry: Optional[RateLimiterRegistry] = None


def get_rate_limiter_registry() -> RateLimiterRegistry:
    """Get or create global rate limiter registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = RateLimiterRegistry()
    return _global_registry


async def main():
    """Demo the rate limiting system."""

    print("=" * 60)
    print("RATE LIMITER DEMONSTRATION")
    print("=" * 60)
    print()

    # Demo 1: Token Bucket
    print("[>>] Token Bucket Limiter (10/sec, burst 20)")
    print("-" * 40)

    bucket = TokenBucketLimiter(rate=10, capacity=20)

    # Simulate burst
    for i in range(25):
        acquired = await bucket.acquire()
        status = "[OK]" if acquired else "[LIMIT]"
        print(f"  Request {i+1:2d}: {status}")
        if i == 19:
            print("  --- burst capacity reached ---")

    stats = bucket.get_stats()
    print(f"\n  Stats: {stats.accepted_requests}/{stats.total_requests} accepted ({stats.acceptance_rate:.1%})")
    print()

    # Demo 2: Sliding Window
    print("[>>] Sliding Window Limiter (5 requests per 2 seconds)")
    print("-" * 40)

    window = SlidingWindowLimiter(max_requests=5, window_seconds=2.0)

    for i in range(8):
        acquired = await window.acquire()
        status = "[OK]" if acquired else "[LIMIT]"
        print(f"  Request {i+1}: {status}")

    print("  Waiting for window to slide...")
    await asyncio.sleep(2.1)

    for i in range(3):
        acquired = await window.acquire()
        status = "[OK]" if acquired else "[LIMIT]"
        print(f"  Request {i+9}: {status}")

    print()

    # Demo 3: Adaptive Rate Limiter
    print("[>>] Adaptive Rate Limiter (adjusts based on errors)")
    print("-" * 40)

    adaptive = AdaptiveRateLimiter(
        initial_rate=10,
        min_rate=1,
        max_rate=100,
        capacity=20
    )

    print(f"  Initial rate: {adaptive.current_rate:.1f}/sec")

    # Simulate errors to trigger backoff
    print("  Simulating errors...")
    for _ in range(5):
        adaptive.record_error()

    print(f"  After errors: {adaptive.current_rate:.1f}/sec")

    # Simulate recovery
    print("  Simulating successful requests...")
    for _ in range(15):
        adaptive.record_success()

    print(f"  After recovery: {adaptive.current_rate:.1f}/sec")
    print()

    # Demo 4: Registry
    print("[>>] Rate Limiter Registry")
    print("-" * 40)

    registry = get_rate_limiter_registry()
    registry.register("api", TokenBucketLimiter(rate=100, capacity=200))
    registry.register("database", SlidingWindowLimiter(max_requests=50, window_seconds=1.0))
    registry.register("external", TokenBucketLimiter(rate=10, capacity=10))

    for name in ["api", "database", "external", "unknown"]:
        acquired = await registry.acquire(name)
        status = "[OK]" if acquired else "[LIMIT]"
        print(f"  {name}: {status}")

    print()
    print("[OK] Rate limiter demonstration complete")


if __name__ == "__main__":
    asyncio.run(main())
