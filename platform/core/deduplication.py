"""
Request Deduplication Module
============================

Prevents duplicate API calls through in-flight request tracking and TTL-based
response caching. Thread-safe implementation using asyncio locks.

Key Features:
- Hash-based request fingerprinting (SHA-256)
- In-flight request tracking (returns same future for duplicate)
- TTL-based cache for recent responses
- Metrics for deduplication hit rate
- Thread-safe implementation

Performance Target: Eliminate 30%+ duplicate calls in batch scenarios.

Usage:
    from core.deduplication import RequestDeduplicator, get_deduplicator

    # Get singleton instance
    deduplicator = get_deduplicator()

    # Use as context manager or decorator
    async with deduplicator.deduplicate("search", query="test") as result:
        if result.is_cached:
            return result.value
        # Execute actual request
        response = await make_api_call()
        return response

    # Or as decorator
    @deduplicated("exa_adapter")
    async def search(query: str) -> dict:
        ...

    # Get metrics
    stats = deduplicator.get_stats()
    print(f"Dedup hit rate: {stats['hit_rate']:.2%}")
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from collections import OrderedDict
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DeduplicationStrategy(str, Enum):
    """Deduplication strategy options."""

    EXACT = "exact"
    """Exact match on all parameters."""

    CONTENT_HASH = "content_hash"
    """Hash-based match on normalized content."""

    SEMANTIC = "semantic"
    """Semantic similarity match (requires embeddings)."""


@dataclass
class DeduplicationConfig:
    """Configuration for request deduplication."""

    enabled: bool = True
    """Enable/disable deduplication globally."""

    default_ttl_seconds: float = 60.0
    """Default TTL for cached responses (seconds)."""

    max_cache_size: int = 1000
    """Maximum number of cached responses."""

    max_in_flight: int = 100
    """Maximum number of in-flight requests to track."""

    strategy: DeduplicationStrategy = DeduplicationStrategy.CONTENT_HASH
    """Deduplication strategy."""

    track_metrics: bool = True
    """Enable metrics collection."""

    cleanup_interval_seconds: float = 30.0
    """Interval for automatic cleanup of expired entries."""


@dataclass
class DeduplicationStats:
    """Statistics for deduplication performance."""

    total_requests: int = 0
    """Total number of requests processed."""

    cache_hits: int = 0
    """Requests served from cache."""

    in_flight_hits: int = 0
    """Requests that joined an in-flight request."""

    cache_misses: int = 0
    """Requests that resulted in new API calls."""

    evictions: int = 0
    """Number of cache evictions."""

    errors: int = 0
    """Number of deduplication errors."""

    total_bytes_saved: int = 0
    """Estimated bytes saved by deduplication."""

    avg_response_size: float = 0.0
    """Average response size in bytes."""

    @property
    def hit_rate(self) -> float:
        """Calculate overall deduplication hit rate."""
        total = self.cache_hits + self.in_flight_hits + self.cache_misses
        if total == 0:
            return 0.0
        return (self.cache_hits + self.in_flight_hits) / total

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache-only hit rate."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    @property
    def in_flight_hit_rate(self) -> float:
        """Calculate in-flight hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.in_flight_hits / self.total_requests

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "in_flight_hits": self.in_flight_hits,
            "cache_misses": self.cache_misses,
            "evictions": self.evictions,
            "errors": self.errors,
            "hit_rate": self.hit_rate,
            "cache_hit_rate": self.cache_hit_rate,
            "in_flight_hit_rate": self.in_flight_hit_rate,
            "total_bytes_saved": self.total_bytes_saved,
            "avg_response_size": self.avg_response_size,
        }


@dataclass
class CachedResponse(Generic[T]):
    """A cached response with metadata."""

    value: T
    """The cached response value."""

    fingerprint: str
    """Request fingerprint used for matching."""

    created_at: float
    """Unix timestamp when cached."""

    ttl: float
    """Time-to-live in seconds."""

    access_count: int = 0
    """Number of times this cache entry was accessed."""

    last_accessed: float = field(default_factory=time.time)
    """Unix timestamp of last access."""

    size_bytes: int = 0
    """Estimated size of cached response in bytes."""

    adapter_name: Optional[str] = None
    """Name of adapter that generated this response."""

    operation: Optional[str] = None
    """Operation name (e.g., 'search', 'extract')."""

    def is_expired(self) -> bool:
        """Check if the cached response has expired."""
        return time.time() - self.created_at > self.ttl

    def touch(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()


@dataclass
class DeduplicationResult(Generic[T]):
    """Result of a deduplication check."""

    is_cached: bool
    """True if response was found in cache."""

    is_in_flight: bool
    """True if an identical request is in flight."""

    value: Optional[T] = None
    """Cached value if is_cached is True."""

    future: Optional[asyncio.Future] = None
    """Future to await if is_in_flight is True."""

    fingerprint: str = ""
    """Request fingerprint."""


class RequestDeduplicator:
    """
    Thread-safe request deduplication with in-flight tracking.

    Provides two levels of deduplication:
    1. Cache-based: Returns cached responses for recent identical requests
    2. In-flight: Returns same future for concurrent identical requests

    Thread-safe via asyncio.Lock for all state mutations.

    Example:
        deduplicator = RequestDeduplicator()

        # Method 1: Context manager
        async with deduplicator.deduplicate("search", query="test") as result:
            if result.is_cached:
                return result.value
            if result.is_in_flight:
                return await result.future
            # Execute actual request
            response = await api_call()
            return response

        # Method 2: Wrapper
        response = await deduplicator.execute(
            lambda: api_call(query="test"),
            operation="search",
            query="test"
        )
    """

    def __init__(self, config: Optional[DeduplicationConfig] = None):
        """
        Initialize the deduplicator.

        Args:
            config: Configuration options (uses defaults if not provided)
        """
        self.config = config or DeduplicationConfig()

        # Response cache: fingerprint -> CachedResponse
        self._cache: OrderedDict[str, CachedResponse] = OrderedDict()

        # In-flight requests: fingerprint -> (Future, creation_time)
        self._in_flight: Dict[str, Tuple[asyncio.Future, float]] = {}

        # Lock for thread-safe access
        self._lock = asyncio.Lock()

        # Statistics
        self._stats = DeduplicationStats()

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        logger.debug(
            f"RequestDeduplicator initialized with TTL={self.config.default_ttl_seconds}s, "
            f"max_cache={self.config.max_cache_size}"
        )

    async def start(self) -> None:
        """Start the background cleanup task."""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.debug("RequestDeduplicator cleanup task started")

    async def stop(self) -> None:
        """Stop the background cleanup task."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        logger.debug("RequestDeduplicator cleanup task stopped")

    async def _cleanup_loop(self) -> None:
        """Background task for periodic cleanup of expired entries."""
        while self._running:
            try:
                await asyncio.sleep(self.config.cleanup_interval_seconds)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Error in deduplication cleanup: {e}")

    async def _cleanup_expired(self) -> int:
        """Remove expired entries from cache and stale in-flight requests."""
        cleaned = 0
        async with self._lock:
            now = time.time()

            # Clean expired cache entries
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
                cleaned += 1
                self._stats.evictions += 1

            # Clean stale in-flight requests (older than 5 minutes)
            stale_keys = [
                key for key, (_, created) in self._in_flight.items()
                if now - created > 300  # 5 minute timeout
            ]
            for key in stale_keys:
                future, _ = self._in_flight.pop(key)
                if not future.done():
                    future.cancel()
                cleaned += 1

        if cleaned > 0:
            logger.debug(f"Cleaned {cleaned} expired/stale deduplication entries")

        return cleaned

    def _generate_fingerprint(
        self,
        operation: str,
        adapter_name: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a unique fingerprint for a request.

        Uses SHA-256 hash of normalized request parameters.

        Args:
            operation: Operation name (e.g., 'search', 'extract')
            adapter_name: Optional adapter name for namespacing
            **kwargs: Request parameters

        Returns:
            Hex string fingerprint
        """
        # Normalize kwargs for consistent hashing
        normalized = self._normalize_params(kwargs)

        # Build fingerprint components
        components = {
            "operation": operation,
            "adapter": adapter_name or "default",
            "params": normalized,
        }

        # Serialize to JSON for hashing
        try:
            serialized = json.dumps(components, sort_keys=True, default=str)
        except (TypeError, ValueError):
            # Fallback for non-serializable content
            serialized = f"{operation}:{adapter_name}:{str(kwargs)}"

        # Generate SHA-256 hash
        return hashlib.sha256(serialized.encode()).hexdigest()[:32]

    def _normalize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize parameters for consistent fingerprinting.

        Handles:
        - Sorting dict keys
        - Converting sets to sorted lists
        - Removing None values
        - Normalizing strings (lowercase, strip)
        """
        normalized = {}

        for key, value in sorted(params.items()):
            if value is None:
                continue

            if isinstance(value, dict):
                normalized[key] = self._normalize_params(value)
            elif isinstance(value, (list, tuple)):
                normalized[key] = [
                    self._normalize_params(v) if isinstance(v, dict) else v
                    for v in value
                ]
            elif isinstance(value, set):
                normalized[key] = sorted(list(value))
            elif isinstance(value, str):
                # Normalize strings for queries
                normalized[key] = value.strip().lower()
            else:
                normalized[key] = value

        return normalized

    def _estimate_size(self, value: Any) -> int:
        """Estimate the size of a value in bytes."""
        try:
            return len(json.dumps(value, default=str).encode())
        except (TypeError, ValueError):
            return len(str(value).encode())

    async def check(
        self,
        operation: str,
        adapter_name: Optional[str] = None,
        **kwargs: Any,
    ) -> DeduplicationResult:
        """
        Check if a request can be deduplicated.

        Args:
            operation: Operation name
            adapter_name: Optional adapter name
            **kwargs: Request parameters

        Returns:
            DeduplicationResult indicating cache hit, in-flight, or miss
        """
        if not self.config.enabled:
            return DeduplicationResult(
                is_cached=False,
                is_in_flight=False,
                fingerprint=""
            )

        fingerprint = self._generate_fingerprint(operation, adapter_name, **kwargs)

        async with self._lock:
            self._stats.total_requests += 1

            # Check cache first
            if fingerprint in self._cache:
                entry = self._cache[fingerprint]
                if not entry.is_expired():
                    entry.touch()
                    # Move to end for LRU behavior
                    self._cache.move_to_end(fingerprint)
                    self._stats.cache_hits += 1
                    self._stats.total_bytes_saved += entry.size_bytes

                    logger.debug(
                        f"Cache hit for {operation}:{fingerprint[:8]} "
                        f"(accessed {entry.access_count} times)"
                    )

                    return DeduplicationResult(
                        is_cached=True,
                        is_in_flight=False,
                        value=entry.value,
                        fingerprint=fingerprint
                    )
                else:
                    # Expired, remove it
                    del self._cache[fingerprint]
                    self._stats.evictions += 1

            # Check in-flight requests
            if fingerprint in self._in_flight:
                future, _ = self._in_flight[fingerprint]
                if not future.done():
                    self._stats.in_flight_hits += 1

                    logger.debug(
                        f"In-flight hit for {operation}:{fingerprint[:8]}"
                    )

                    return DeduplicationResult(
                        is_cached=False,
                        is_in_flight=True,
                        future=future,
                        fingerprint=fingerprint
                    )
                else:
                    # Future is done but not yet cached, clean up
                    del self._in_flight[fingerprint]

            self._stats.cache_misses += 1
            return DeduplicationResult(
                is_cached=False,
                is_in_flight=False,
                fingerprint=fingerprint
            )

    async def register_in_flight(
        self,
        fingerprint: str,
    ) -> asyncio.Future:
        """
        Register a request as in-flight.

        Args:
            fingerprint: Request fingerprint

        Returns:
            Future that will be resolved when the request completes
        """
        async with self._lock:
            if fingerprint in self._in_flight:
                return self._in_flight[fingerprint][0]

            # Check if we're at capacity
            if len(self._in_flight) >= self.config.max_in_flight:
                # Remove oldest in-flight request
                oldest_key = next(iter(self._in_flight))
                old_future, _ = self._in_flight.pop(oldest_key)
                if not old_future.done():
                    old_future.cancel()

            future: asyncio.Future = asyncio.get_event_loop().create_future()
            self._in_flight[fingerprint] = (future, time.time())

            return future

    async def complete_request(
        self,
        fingerprint: str,
        result: T,
        ttl: Optional[float] = None,
        adapter_name: Optional[str] = None,
        operation: Optional[str] = None,
    ) -> None:
        """
        Complete an in-flight request and cache the result.

        Args:
            fingerprint: Request fingerprint
            result: Response to cache
            ttl: Optional custom TTL (uses default if not specified)
            adapter_name: Adapter name for metadata
            operation: Operation name for metadata
        """
        effective_ttl = ttl if ttl is not None else self.config.default_ttl_seconds
        size_bytes = self._estimate_size(result)

        async with self._lock:
            # Complete the future
            if fingerprint in self._in_flight:
                future, _ = self._in_flight.pop(fingerprint)
                if not future.done():
                    future.set_result(result)

            # Add to cache
            # Evict if at capacity (LRU)
            while len(self._cache) >= self.config.max_cache_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats.evictions += 1

            self._cache[fingerprint] = CachedResponse(
                value=result,
                fingerprint=fingerprint,
                created_at=time.time(),
                ttl=effective_ttl,
                size_bytes=size_bytes,
                adapter_name=adapter_name,
                operation=operation,
            )

            # Update average response size
            total_size = sum(e.size_bytes for e in self._cache.values())
            self._stats.avg_response_size = total_size / len(self._cache)

            logger.debug(
                f"Cached response for {operation}:{fingerprint[:8]} "
                f"(size={size_bytes}b, ttl={effective_ttl}s)"
            )

    async def fail_request(
        self,
        fingerprint: str,
        error: Exception,
    ) -> None:
        """
        Mark an in-flight request as failed.

        Args:
            fingerprint: Request fingerprint
            error: The exception that occurred
        """
        async with self._lock:
            if fingerprint in self._in_flight:
                future, _ = self._in_flight.pop(fingerprint)
                if not future.done():
                    future.set_exception(error)
            self._stats.errors += 1

    @asynccontextmanager
    async def deduplicate(
        self,
        operation: str,
        adapter_name: Optional[str] = None,
        ttl: Optional[float] = None,
        **kwargs: Any,
    ):
        """
        Context manager for request deduplication.

        Example:
            async with deduplicator.deduplicate("search", query="test") as result:
                if result.is_cached:
                    yield result.value
                    return
                if result.is_in_flight:
                    yield await result.future
                    return
                # Execute actual request
                response = await make_api_call()
                yield response

        Args:
            operation: Operation name
            adapter_name: Optional adapter name
            ttl: Optional custom TTL
            **kwargs: Request parameters

        Yields:
            DeduplicationResult
        """
        result = await self.check(operation, adapter_name, **kwargs)

        if result.is_cached or result.is_in_flight:
            yield result
            return

        # Register as in-flight
        future = await self.register_in_flight(result.fingerprint)

        try:
            # Let caller execute the request
            yield result

            # Note: Caller must call complete_request() with the result

        except Exception as e:
            await self.fail_request(result.fingerprint, e)
            raise

    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        operation: str,
        adapter_name: Optional[str] = None,
        ttl: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        """
        Execute a function with deduplication.

        This is a convenience method that handles the full deduplication flow.

        Args:
            func: Async function to execute
            operation: Operation name
            adapter_name: Optional adapter name
            ttl: Optional custom TTL
            **kwargs: Function arguments (also used for fingerprinting)

        Returns:
            Result from cache, in-flight request, or new execution
        """
        result = await self.check(operation, adapter_name, **kwargs)

        if result.is_cached:
            logger.debug(f"Returning cached result for {operation}")
            return result.value

        if result.is_in_flight:
            logger.debug(f"Waiting for in-flight request for {operation}")
            return await result.future

        # Register as in-flight
        await self.register_in_flight(result.fingerprint)

        try:
            # Execute the function
            response = await func(**kwargs)

            # Cache the result
            await self.complete_request(
                result.fingerprint,
                response,
                ttl=ttl,
                adapter_name=adapter_name,
                operation=operation,
            )

            return response

        except Exception as e:
            await self.fail_request(result.fingerprint, e)
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        return {
            **self._stats.to_dict(),
            "cache_size": len(self._cache),
            "in_flight_count": len(self._in_flight),
            "config": {
                "enabled": self.config.enabled,
                "default_ttl": self.config.default_ttl_seconds,
                "max_cache_size": self.config.max_cache_size,
                "max_in_flight": self.config.max_in_flight,
            },
        }

    async def clear(self) -> int:
        """Clear all cached responses and in-flight requests."""
        async with self._lock:
            count = len(self._cache) + len(self._in_flight)

            # Cancel all in-flight futures
            for future, _ in self._in_flight.values():
                if not future.done():
                    future.cancel()

            self._cache.clear()
            self._in_flight.clear()

            logger.info(f"Cleared {count} deduplication entries")
            return count

    async def invalidate(
        self,
        operation: Optional[str] = None,
        adapter_name: Optional[str] = None,
    ) -> int:
        """
        Invalidate cached entries matching criteria.

        Args:
            operation: Optional operation name to match
            adapter_name: Optional adapter name to match

        Returns:
            Number of entries invalidated
        """
        async with self._lock:
            invalidated = 0

            keys_to_remove = []
            for key, entry in self._cache.items():
                if operation and entry.operation != operation:
                    continue
                if adapter_name and entry.adapter_name != adapter_name:
                    continue
                keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._cache[key]
                invalidated += 1

            if invalidated > 0:
                logger.debug(
                    f"Invalidated {invalidated} cache entries "
                    f"(operation={operation}, adapter={adapter_name})"
                )

            return invalidated


# =============================================================================
# Singleton Instance and Factory Functions
# =============================================================================

_deduplicator: Optional[RequestDeduplicator] = None
_deduplicator_lock = asyncio.Lock()


async def get_deduplicator(
    config: Optional[DeduplicationConfig] = None,
) -> RequestDeduplicator:
    """
    Get the global request deduplicator instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        The singleton RequestDeduplicator instance
    """
    global _deduplicator

    async with _deduplicator_lock:
        if _deduplicator is None:
            _deduplicator = RequestDeduplicator(config)
            await _deduplicator.start()
        return _deduplicator


def get_deduplicator_sync(
    config: Optional[DeduplicationConfig] = None,
) -> RequestDeduplicator:
    """
    Get the global request deduplicator instance (sync version).

    Note: This does not start the cleanup task. Call start() manually
    if using in an async context later.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        The singleton RequestDeduplicator instance
    """
    global _deduplicator

    if _deduplicator is None:
        _deduplicator = RequestDeduplicator(config)
    return _deduplicator


async def reset_deduplicator() -> None:
    """Reset the global deduplicator instance."""
    global _deduplicator

    async with _deduplicator_lock:
        if _deduplicator is not None:
            await _deduplicator.stop()
            await _deduplicator.clear()
            _deduplicator = None


# =============================================================================
# Decorator for Easy Integration
# =============================================================================

def deduplicated(
    adapter_name: str,
    operation: Optional[str] = None,
    ttl: Optional[float] = None,
    config: Optional[DeduplicationConfig] = None,
):
    """
    Decorator to add deduplication to an async function.

    Example:
        @deduplicated("exa_adapter", operation="search")
        async def search(query: str, max_results: int = 10) -> dict:
            return await client.search(query, max_results=max_results)

    Args:
        adapter_name: Name of the adapter (for metrics/namespacing)
        operation: Operation name (defaults to function name)
        ttl: Optional custom TTL for cached responses
        config: Optional custom deduplication config

    Returns:
        Decorated function with deduplication
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        op_name = operation or func.__name__

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            # Get deduplicator
            deduplicator = get_deduplicator_sync(config)

            # Execute with deduplication
            return await deduplicator.execute(
                func,
                operation=op_name,
                adapter_name=adapter_name,
                ttl=ttl,
                **kwargs,
            )

        return wrapper

    return decorator


# =============================================================================
# Adapter Mixin for Easy Integration
# =============================================================================

class DeduplicatedAdapter:
    """
    Mixin class that adds deduplication capabilities to adapters.

    Usage:
        class MyAdapter(SDKAdapter, DeduplicatedAdapter):
            def __init__(self):
                super().__init__()
                self.init_deduplication(
                    adapter_name="my_adapter",
                    default_ttl=60.0
                )

            async def search(self, query: str) -> dict:
                return await self.deduplicated_call(
                    self._do_search,
                    operation="search",
                    query=query
                )

            async def _do_search(self, query: str) -> dict:
                # Actual API call
                return await self.client.search(query)
    """

    _dedup_adapter_name: str
    _dedup_default_ttl: float
    _dedup_enabled: bool

    def init_deduplication(
        self,
        adapter_name: str,
        default_ttl: float = 60.0,
        enabled: bool = True,
    ) -> None:
        """
        Initialize deduplication for this adapter.

        Args:
            adapter_name: Name of the adapter
            default_ttl: Default TTL for cached responses
            enabled: Enable/disable deduplication
        """
        self._dedup_adapter_name = adapter_name
        self._dedup_default_ttl = default_ttl
        self._dedup_enabled = enabled

    async def deduplicated_call(
        self,
        func: Callable[..., Awaitable[T]],
        operation: str,
        ttl: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        """
        Execute a function with deduplication.

        Args:
            func: Async function to execute
            operation: Operation name
            ttl: Optional custom TTL
            **kwargs: Function arguments

        Returns:
            Result from cache, in-flight, or new execution
        """
        if not self._dedup_enabled:
            return await func(**kwargs)

        deduplicator = get_deduplicator_sync()

        return await deduplicator.execute(
            func,
            operation=operation,
            adapter_name=self._dedup_adapter_name,
            ttl=ttl if ttl is not None else self._dedup_default_ttl,
            **kwargs,
        )


# =============================================================================
# Export Public API
# =============================================================================

__all__ = [
    # Configuration
    "DeduplicationConfig",
    "DeduplicationStrategy",
    # Statistics
    "DeduplicationStats",
    # Core types
    "CachedResponse",
    "DeduplicationResult",
    # Main class
    "RequestDeduplicator",
    # Factory functions
    "get_deduplicator",
    "get_deduplicator_sync",
    "reset_deduplicator",
    # Decorator
    "deduplicated",
    # Mixin
    "DeduplicatedAdapter",
]
