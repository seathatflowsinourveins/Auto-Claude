"""
Caching Layer for Unleashed Platform

Multi-backend caching system supporting:
- In-memory cache (LRU with TTL)
- Redis cache (distributed)
- File-based cache (persistent)
- Semantic cache (embedding-based similarity)

Enables:
- Response caching for LLM calls
- Embedding caching for vector operations
- Research result caching
- Pipeline state persistence
"""

import asyncio
import inspect
import hashlib
import json
import pickle
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic, Callable, Union
from dataclasses import dataclass, field
from collections import OrderedDict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# V45: Circuit Breaker for Cache Operations
# =============================================================================

from enum import Enum as _CBEnum
from datetime import datetime as _cb_datetime, timezone as _cb_timezone


class _CacheCircuitState(_CBEnum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class _CacheCircuitBreaker:
    """
    V45: Circuit breaker for Redis cache operations.

    Prevents cascade failures when Redis is unavailable.
    """
    name: str = "redis_cache"
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0
    half_open_max_requests: int = 3

    state: _CacheCircuitState = field(default=_CacheCircuitState.CLOSED)
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    half_open_requests: int = 0

    def can_execute(self) -> bool:
        """Check if the circuit allows execution."""
        if self.state == _CacheCircuitState.CLOSED:
            return True

        if self.state == _CacheCircuitState.OPEN:
            if self.last_failure_time:
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.recovery_timeout_seconds:
                    self.state = _CacheCircuitState.HALF_OPEN
                    self.half_open_requests = 0
                    logger.info(f"[V45] Circuit {self.name} transitioning to HALF_OPEN")
                    return True
            return False

        if self.state == _CacheCircuitState.HALF_OPEN:
            if self.half_open_requests < self.half_open_max_requests:
                self.half_open_requests += 1
                return True
            return False

        return True

    def record_success(self) -> None:
        """Record a successful operation."""
        if self.state == _CacheCircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 3:
                self.state = _CacheCircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"[V45] Circuit {self.name} CLOSED (recovered)")
        elif self.state == _CacheCircuitState.CLOSED:
            if self.failure_count > 0:
                self.failure_count -= 1

    def record_failure(self) -> None:
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == _CacheCircuitState.HALF_OPEN:
            self.state = _CacheCircuitState.OPEN
            logger.warning(f"[V45] Circuit {self.name} OPEN (half-open failed)")
        elif self.state == _CacheCircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self.state = _CacheCircuitState.OPEN
                logger.warning(f"[V45] Circuit {self.name} OPEN (threshold reached)")


@dataclass
class CacheEntry(Generic[T]):
    """A single cache entry with metadata."""
    key: str
    value: T
    created_at: float
    ttl: Optional[float] = None  # Time to live in seconds
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CacheBackend(ABC, Generic[T]):
    """Abstract base class for cache backends."""

    @abstractmethod
    async def get(self, key: str) -> Optional[T]:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        value: T,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Set value in cache."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    async def clear(self) -> int:
        """Clear all entries, return count cleared."""
        pass

    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        pass


class MemoryCache(CacheBackend[T]):
    """
    In-memory LRU cache with TTL support.

    V13 Enhancement: Adaptive TTL - extends TTL on cache hits for stable data.
    Thread-safe with automatic eviction of expired entries.
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[float] = 3600,  # 1 hour default
        adaptive_ttl: bool = True,  # V13: Enable adaptive TTL
        max_ttl: Optional[float] = 14400,  # V13: 4 hour max TTL
        ttl_growth_factor: float = 1.2,  # V13: 20% TTL increase on hit
    ):
        """
        Initialize memory cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds (None = no expiry)
            adaptive_ttl: V13 - Enable adaptive TTL on cache hits
            max_ttl: V13 - Maximum TTL after growth (None = unlimited)
            ttl_growth_factor: V13 - TTL multiplier on each hit (default: 1.2 = 20%)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        # V13: Adaptive TTL configuration
        self.adaptive_ttl = adaptive_ttl
        self.max_ttl = max_ttl
        self.ttl_growth_factor = ttl_growth_factor
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._stats = CacheStats(max_size=max_size)

    async def get(self, key: str) -> Optional[T]:
        """Get value from cache.

        V13: Adaptive TTL - extends TTL on cache hits for frequently accessed data.
        Expected improvement: 3-5x effective cache lifetime for stable data.
        """
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                self._stats.size -= 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.update_access()

            # V13: Adaptive TTL - extend TTL on cache hit
            if self.adaptive_ttl and entry.ttl is not None:
                new_ttl = entry.ttl * self.ttl_growth_factor
                if self.max_ttl is not None:
                    new_ttl = min(new_ttl, self.max_ttl)
                entry.ttl = new_ttl

            self._stats.hits += 1
            return entry.value

    async def set(
        self,
        key: str,
        value: T,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Set value in cache."""
        async with self._lock:
            # Use default TTL if not specified
            effective_ttl = ttl if ttl is not None else self.default_ttl

            # Remove existing entry if present
            if key in self._cache:
                del self._cache[key]
            else:
                self._stats.size += 1

            # Evict oldest entries if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats.evictions += 1
                self._stats.size -= 1

            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                ttl=effective_ttl,
                metadata=metadata or {},
            )
            self._cache[key] = entry

            return True

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.size -= 1
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired():
                del self._cache[key]
                self._stats.size -= 1
                return False
            return True

    async def clear(self) -> int:
        """Clear all entries."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats.size = 0
            return count

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return CacheStats(
            hits=self._stats.hits,
            misses=self._stats.misses,
            evictions=self._stats.evictions,
            size=self._stats.size,
            max_size=self.max_size,
        )

    async def cleanup_expired(self) -> int:
        """Remove all expired entries."""
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
                self._stats.evictions += 1
                self._stats.size -= 1
            return len(expired_keys)


class FileCache(CacheBackend[T]):
    """
    File-based persistent cache.

    Stores entries as serialized files for persistence across restarts.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        default_ttl: Optional[float] = 86400,  # 24 hours
        serializer: str = "pickle",  # "pickle" or "json"
    ):
        """
        Initialize file cache.

        Args:
            cache_dir: Directory to store cache files
            default_ttl: Default TTL in seconds
            serializer: Serialization format
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self.serializer = serializer
        self._stats = CacheStats()
        self._lock = asyncio.Lock()

    def _key_to_path(self, key: str) -> Path:
        """Convert cache key to file path."""
        # Use hash to avoid filesystem issues with special characters
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def _serialize(self, entry: CacheEntry) -> bytes:
        """Serialize cache entry."""
        if self.serializer == "json":
            data = {
                "key": entry.key,
                "value": entry.value,
                "created_at": entry.created_at,
                "ttl": entry.ttl,
                "access_count": entry.access_count,
                "metadata": entry.metadata,
            }
            return json.dumps(data).encode()
        else:
            return pickle.dumps(entry)

    def _deserialize(self, data: bytes) -> CacheEntry:
        """Deserialize cache entry."""
        if self.serializer == "json":
            obj = json.loads(data.decode())
            return CacheEntry(
                key=obj["key"],
                value=obj["value"],
                created_at=obj["created_at"],
                ttl=obj["ttl"],
                access_count=obj.get("access_count", 0),
                metadata=obj.get("metadata", {}),
            )
        else:
            return pickle.loads(data)

    async def get(self, key: str) -> Optional[T]:
        """Get value from file cache."""
        async with self._lock:
            path = self._key_to_path(key)

            if not path.exists():
                self._stats.misses += 1
                return None

            try:
                data = path.read_bytes()
                entry = self._deserialize(data)

                if entry.is_expired():
                    path.unlink()
                    self._stats.misses += 1
                    self._stats.evictions += 1
                    return None

                # Update access stats
                entry.update_access()
                path.write_bytes(self._serialize(entry))

                self._stats.hits += 1
                return entry.value

            except Exception as e:
                logger.error(f"File cache read error for {key}: {e}")
                self._stats.misses += 1
                return None

    async def set(
        self,
        key: str,
        value: T,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Set value in file cache."""
        async with self._lock:
            path = self._key_to_path(key)

            effective_ttl = ttl if ttl is not None else self.default_ttl

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                ttl=effective_ttl,
                metadata=metadata or {},
            )

            try:
                path.write_bytes(self._serialize(entry))
                return True
            except Exception as e:
                logger.error(f"File cache write error for {key}: {e}")
                return False

    async def delete(self, key: str) -> bool:
        """Delete value from file cache."""
        async with self._lock:
            path = self._key_to_path(key)
            if path.exists():
                path.unlink()
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        result = await self.get(key)
        return result is not None

    async def clear(self) -> int:
        """Clear all cache files."""
        async with self._lock:
            count = 0
            for path in self.cache_dir.glob("*.cache"):
                try:
                    path.unlink()
                    count += 1
                except Exception as e:
                    logger.error(f"Error deleting cache file {path}: {e}")
            return count

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        # Count files for current size
        size = len(list(self.cache_dir.glob("*.cache")))
        return CacheStats(
            hits=self._stats.hits,
            misses=self._stats.misses,
            evictions=self._stats.evictions,
            size=size,
            max_size=0,  # No limit for file cache
        )


class RedisCache(CacheBackend[T]):
    """
    Redis-based distributed cache.

    V45: Environment-configurable URL + circuit breaker protection.
    Requires redis-py async support.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        prefix: str = "unleashed:",
        default_ttl: Optional[int] = 3600,
    ):
        """
        Initialize Redis cache.

        Args:
            redis_url: Redis connection URL (defaults to REDIS_URL env var)
            prefix: Key prefix for namespacing
            default_ttl: Default TTL in seconds
        """
        import os
        # V45 FIX: Environment-configurable Redis URL
        self.redis_url = redis_url or os.environ.get(
            "REDIS_URL", "redis://localhost:6379"
        )
        self.prefix = prefix
        self.default_ttl = default_ttl
        self._client = None
        self._stats = CacheStats()
        # V45: Circuit breaker for Redis operations
        self._circuit_breaker = _CacheCircuitBreaker(name="redis_cache")

    async def _get_client(self):
        """Get or create Redis client with circuit breaker check."""
        # V45: Check circuit breaker first
        if not self._circuit_breaker.can_execute():
            logger.warning("[V45] Redis circuit breaker OPEN - skipping operation")
            return None

        if self._client is None:
            try:
                import redis.asyncio as redis
                self._client = redis.from_url(self.redis_url)
                self._circuit_breaker.record_success()
            except ImportError:
                raise ImportError("redis package required: pip install redis")
            except Exception as e:
                self._circuit_breaker.record_failure()
                logger.error(f"[V45] Redis client init failed: {e}")
                return None
        return self._client

    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.prefix}{key}"

    async def get(self, key: str) -> Optional[T]:
        """Get value from Redis. V45: Protected by circuit breaker."""
        try:
            client = await self._get_client()
            # V45: Circuit breaker may block client creation
            if client is None:
                self._stats.misses += 1
                return None

            data = await client.get(self._make_key(key))

            if data is None:
                self._stats.misses += 1
                return None

            self._stats.hits += 1
            self._circuit_breaker.record_success()  # V45
            return pickle.loads(data)

        except Exception as e:
            self._circuit_breaker.record_failure()  # V45
            logger.error(f"Redis get error for {key}: {e}")
            self._stats.misses += 1
            return None

    async def set(
        self,
        key: str,
        value: T,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Set value in Redis. V45: Protected by circuit breaker."""
        try:
            client = await self._get_client()
            # V45: Circuit breaker may block client creation
            if client is None:
                return False

            effective_ttl = int(ttl) if ttl else self.default_ttl

            data = pickle.dumps(value)

            if effective_ttl:
                await client.setex(
                    self._make_key(key),
                    effective_ttl,
                    data,
                )
            else:
                await client.set(self._make_key(key), data)

            self._circuit_breaker.record_success()  # V45
            return True

        except Exception as e:
            self._circuit_breaker.record_failure()  # V45
            logger.error(f"Redis set error for {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete value from Redis. V45: Protected by circuit breaker."""
        try:
            client = await self._get_client()
            if client is None:
                return False
            result = await client.delete(self._make_key(key))
            self._circuit_breaker.record_success()  # V45
            return result > 0
        except Exception as e:
            self._circuit_breaker.record_failure()  # V45
            logger.error(f"Redis delete error for {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis. V45: Protected by circuit breaker."""
        try:
            client = await self._get_client()
            if client is None:
                return False
            result = await client.exists(self._make_key(key)) > 0
            self._circuit_breaker.record_success()  # V45
            return result
        except Exception as e:
            self._circuit_breaker.record_failure()  # V45
            logger.error(f"Redis exists error for {key}: {e}")
            return False

    async def clear(self) -> int:
        """Clear all keys with prefix."""
        try:
            client = await self._get_client()
            keys = []
            async for key in client.scan_iter(f"{self.prefix}*"):
                keys.append(key)

            if keys:
                return await client.delete(*keys)
            return 0

        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return 0

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats


class SemanticCache(CacheBackend[T]):
    """
    Embedding-based semantic cache.

    Returns cached results for semantically similar queries,
    useful for LLM response caching where exact key matching
    would miss many cache hits.
    """

    def __init__(
        self,
        embedding_fn: Callable[[str], List[float]],
        similarity_threshold: float = 0.95,
        max_size: int = 1000,
        default_ttl: Optional[float] = 3600,
    ):
        """
        Initialize semantic cache.

        Args:
            embedding_fn: Function to generate embeddings
            similarity_threshold: Minimum similarity for cache hit
            max_size: Maximum number of entries
            default_ttl: Default TTL in seconds
        """
        self.embedding_fn = embedding_fn
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._entries: List[tuple] = []  # (key, embedding, entry)
        self._lock = asyncio.Lock()
        self._stats = CacheStats(max_size=max_size)

    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float],
    ) -> float:
        """Calculate cosine similarity between vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _find_similar(
        self,
        query_embedding: List[float],
    ) -> Optional[CacheEntry[T]]:
        """Find most similar non-expired entry."""
        best_match = None
        best_similarity = 0.0

        now = time.time()
        expired_indices = []

        for i, (key, embedding, entry) in enumerate(self._entries):
            # Check expiration
            if entry.is_expired():
                expired_indices.append(i)
                continue

            similarity = self._cosine_similarity(query_embedding, embedding)
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = entry

        # Remove expired entries
        for i in reversed(expired_indices):
            del self._entries[i]
            self._stats.evictions += 1
            self._stats.size -= 1

        return best_match

    async def get(self, key: str) -> Optional[T]:
        """Get semantically similar value."""
        async with self._lock:
            # Generate embedding for query
            try:
                embedding = self.embedding_fn(key)
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                self._stats.misses += 1
                return None

            # Find similar entry
            entry = self._find_similar(embedding)

            if entry is None:
                self._stats.misses += 1
                return None

            entry.update_access()
            self._stats.hits += 1

            return entry.value

    async def set(
        self,
        key: str,
        value: T,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Set value with semantic key."""
        async with self._lock:
            try:
                embedding = self.embedding_fn(key)
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                return False

            effective_ttl = ttl if ttl is not None else self.default_ttl

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                ttl=effective_ttl,
                metadata=metadata or {},
            )

            # Evict if at capacity (remove least recently accessed)
            while len(self._entries) >= self.max_size:
                # Sort by last_accessed, remove oldest
                self._entries.sort(key=lambda x: x[2].last_accessed)
                del self._entries[0]
                self._stats.evictions += 1
                self._stats.size -= 1

            self._entries.append((key, embedding, entry))
            self._stats.size += 1

            return True

    async def delete(self, key: str) -> bool:
        """Delete entry by exact key."""
        async with self._lock:
            for i, (k, _, _) in enumerate(self._entries):
                if k == key:
                    del self._entries[i]
                    self._stats.size -= 1
                    return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if similar key exists."""
        result = await self.get(key)
        return result is not None

    async def clear(self) -> int:
        """Clear all entries."""
        async with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._stats.size = 0
            return count

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return CacheStats(
            hits=self._stats.hits,
            misses=self._stats.misses,
            evictions=self._stats.evictions,
            size=self._stats.size,
            max_size=self.max_size,
        )


class TieredCache(CacheBackend[T]):
    """
    Multi-tier cache with fallback.

    Checks faster caches first (memory), falls back to slower (file/redis).
    Automatically populates faster tiers on cache hits from slower tiers.
    """

    def __init__(self, backends: List[CacheBackend[T]]):
        """
        Initialize tiered cache.

        Args:
            backends: List of backends, fastest first
        """
        self.backends = backends
        self._stats = CacheStats()

    async def get(self, key: str) -> Optional[T]:
        """Get value, checking each tier."""
        for i, backend in enumerate(self.backends):
            value = await backend.get(key)

            if value is not None:
                # Populate faster tiers
                for faster_backend in self.backends[:i]:
                    await faster_backend.set(key, value)

                self._stats.hits += 1
                return value

        self._stats.misses += 1
        return None

    async def set(
        self,
        key: str,
        value: T,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Set value in all tiers."""
        success = True
        for backend in self.backends:
            if not await backend.set(key, value, ttl, metadata):
                success = False
        return success

    async def delete(self, key: str) -> bool:
        """Delete from all tiers."""
        deleted = False
        for backend in self.backends:
            if await backend.delete(key):
                deleted = True
        return deleted

    async def exists(self, key: str) -> bool:
        """Check if exists in any tier."""
        for backend in self.backends:
            if await backend.exists(key):
                return True
        return False

    async def clear(self) -> int:
        """Clear all tiers."""
        total = 0
        for backend in self.backends:
            total += await backend.clear()
        return total

    def get_stats(self) -> CacheStats:
        """Get aggregated statistics."""
        return self._stats


# Cache decorator for easy function caching
def cached(
    cache: CacheBackend,
    key_fn: Optional[Callable[..., str]] = None,
    ttl: Optional[float] = None,
):
    """
    Decorator to cache function results.

    Usage:
        @cached(cache=memory_cache, ttl=3600)
        async def expensive_function(arg1, arg2):
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                key = f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"

            # Check cache
            cached_value = await cache.get(key)
            if cached_value is not None:
                return cached_value

            # Execute function
            if inspect.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Cache result
            await cache.set(key, result, ttl=ttl)

            return result

        return wrapper
    return decorator


def get_memory_cache(**kwargs) -> MemoryCache:
    """Get configured memory cache."""
    return MemoryCache(**kwargs)


def get_file_cache(cache_dir: str, **kwargs) -> FileCache:
    """Get configured file cache."""
    return FileCache(cache_dir, **kwargs)


def get_tiered_cache(
    memory_size: int = 1000,
    file_dir: Optional[str] = None,
    redis_url: Optional[str] = None,
) -> TieredCache:
    """
    Get configured tiered cache.

    Combines memory + optional file + optional Redis backends.
    """
    backends = [MemoryCache(max_size=memory_size)]

    if file_dir:
        backends.append(FileCache(file_dir))

    if redis_url:
        backends.append(RedisCache(redis_url))

    return TieredCache(backends)
