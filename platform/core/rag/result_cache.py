"""
Intelligent Query Result Cache for RAG Pipeline

This module provides a two-level caching system for RAG query results:
- L1: Exact query match (hash-based, O(1) lookup)
- L2: Semantic similarity match (embedding-based, >0.95 similarity threshold)

Features:
- LRU eviction with configurable max size
- Configurable TTL per query type
- Memory budget enforcement
- Multiple cache invalidation strategies
- Thread-safe operations

Architecture:
    +--------------------------------------------------+
    |                  ResultCache                      |
    |--------------------------------------------------|
    | L1 Cache (ExactMatchCache)                       |
    |   - Hash-based lookup, O(1)                      |
    |   - LRU eviction                                 |
    |   - Per-entry TTL                                |
    |--------------------------------------------------|
    | L2 Cache (SemanticCache)                         |
    |   - Embedding-based similarity                   |
    |   - Threshold: >0.95 for cache hit               |
    |   - Vector search with cosine similarity         |
    |--------------------------------------------------|
    | Memory Budget Manager                            |
    |   - Tracks memory usage                          |
    |   - Triggers eviction when budget exceeded       |
    +--------------------------------------------------+

Usage:
    from core.rag.result_cache import ResultCache, ResultCacheConfig

    config = ResultCacheConfig(
        max_entries=5000,
        memory_budget_mb=256,
        default_ttl_seconds=3600,
        semantic_threshold=0.95,
    )

    cache = ResultCache(config=config, embedding_provider=my_embedder)

    # Cache a result
    cache.put(
        query="What is RAG?",
        result=pipeline_result,
        query_type=QueryType.FACTUAL,
    )

    # Get cached result (tries L1, then L2)
    cached = cache.get("What is RAG?")
    if cached:
        print(f"Cache hit! Level: {cached.cache_level}")

    # Invalidate by pattern
    cache.invalidate_by_pattern("RAG*")

    # Invalidate by age
    cache.invalidate_by_age(max_age_seconds=3600)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

# Type variable for cached values
T = TypeVar("T")


# =============================================================================
# PROTOCOLS
# =============================================================================


class EmbeddingProviderProtocol(Protocol):
    """Protocol for embedding providers."""

    def encode(self, texts: List[str]) -> Any:
        """Encode texts to embeddings. Returns numpy array or list of lists."""
        ...


class PipelineResultProtocol(Protocol):
    """Protocol for pipeline results."""

    response: str
    confidence: float
    contexts_used: List[str]


# =============================================================================
# ENUMS
# =============================================================================


class CacheLevel(str, Enum):
    """Cache level where hit occurred."""

    L1_EXACT = "l1_exact"
    L2_SEMANTIC = "l2_semantic"
    MISS = "miss"


class InvalidationStrategy(str, Enum):
    """Cache invalidation strategies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time-To-Live based
    FIFO = "fifo"  # First In First Out
    SIZE = "size"  # Size-based (evict largest)


class QueryTypeTTL(str, Enum):
    """Query types for TTL configuration."""

    FACTUAL = "factual"
    RESEARCH = "research"
    CODE = "code"
    NEWS = "news"
    GENERAL = "general"


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class ResultCacheConfig:
    """Configuration for the result cache.

    Attributes:
        max_entries: Maximum number of entries in L1 cache (default: 5000)
        memory_budget_mb: Maximum memory budget in MB (default: 256)
        default_ttl_seconds: Default TTL for cache entries (default: 3600)
        semantic_threshold: Similarity threshold for L2 cache hits (default: 0.95)
        enable_l2_cache: Enable semantic L2 cache (default: True)
        l2_max_entries: Maximum entries in L2 cache (default: 1000)
        eviction_strategy: Strategy for cache eviction (default: LRU)
        ttl_per_query_type: Custom TTL per query type
        enable_compression: Enable compression for large entries (default: False)
        compression_threshold_bytes: Size threshold to trigger compression (default: 10KB)
        track_metrics: Enable metrics tracking (default: True)
    """

    max_entries: int = 5000
    memory_budget_mb: float = 256.0
    default_ttl_seconds: int = 3600
    semantic_threshold: float = 0.95
    enable_l2_cache: bool = True
    l2_max_entries: int = 1000
    eviction_strategy: InvalidationStrategy = InvalidationStrategy.LRU
    ttl_per_query_type: Dict[str, int] = field(default_factory=lambda: {
        QueryTypeTTL.FACTUAL.value: 7200,    # 2 hours - facts change slowly
        QueryTypeTTL.RESEARCH.value: 3600,   # 1 hour - research evolves
        QueryTypeTTL.CODE.value: 1800,       # 30 min - code changes frequently
        QueryTypeTTL.NEWS.value: 300,        # 5 min - news is time-sensitive
        QueryTypeTTL.GENERAL.value: 3600,    # 1 hour - default
    })
    enable_compression: bool = False
    compression_threshold_bytes: int = 10240  # 10KB
    track_metrics: bool = True


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class CacheEntry(Generic[T]):
    """A cached result with metadata."""

    key: str
    query: str
    value: T
    query_hash: str
    embedding: Optional[List[float]] = None
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    ttl_seconds: int = 3600
    access_count: int = 0
    size_bytes: int = 0
    query_type: str = "general"
    compressed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() > self.created_at + self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Get entry age in seconds."""
        return time.time() - self.created_at

    @property
    def time_until_expiry(self) -> float:
        """Get time until expiry in seconds."""
        return max(0, (self.created_at + self.ttl_seconds) - time.time())

    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheHit(Generic[T]):
    """Result of a cache lookup."""

    entry: CacheEntry[T]
    cache_level: CacheLevel
    similarity_score: float = 1.0  # 1.0 for exact match
    lookup_time_ms: float = 0.0

    @property
    def value(self) -> T:
        """Get the cached value."""
        return self.entry.value


@dataclass
class CacheStats:
    """Cache statistics."""

    l1_entries: int = 0
    l2_entries: int = 0
    l1_hits: int = 0
    l2_hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_used_bytes: int = 0
    memory_budget_bytes: int = 0
    avg_l1_lookup_ms: float = 0.0
    avg_l2_lookup_ms: float = 0.0
    oldest_entry_age_seconds: float = 0.0
    newest_entry_age_seconds: float = 0.0

    @property
    def total_entries(self) -> int:
        return self.l1_entries + self.l2_entries

    @property
    def total_hits(self) -> int:
        return self.l1_hits + self.l2_hits

    @property
    def total_requests(self) -> int:
        return self.total_hits + self.misses

    @property
    def hit_rate(self) -> float:
        total = self.total_requests
        return self.total_hits / total if total > 0 else 0.0

    @property
    def l1_hit_rate(self) -> float:
        total = self.total_requests
        return self.l1_hits / total if total > 0 else 0.0

    @property
    def l2_hit_rate(self) -> float:
        total = self.total_requests
        return self.l2_hits / total if total > 0 else 0.0

    @property
    def memory_utilization(self) -> float:
        if self.memory_budget_bytes == 0:
            return 0.0
        return self.memory_used_bytes / self.memory_budget_bytes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "l1_entries": self.l1_entries,
            "l2_entries": self.l2_entries,
            "total_entries": self.total_entries,
            "l1_hits": self.l1_hits,
            "l2_hits": self.l2_hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "l1_hit_rate": self.l1_hit_rate,
            "l2_hit_rate": self.l2_hit_rate,
            "evictions": self.evictions,
            "memory_used_mb": self.memory_used_bytes / (1024 * 1024),
            "memory_budget_mb": self.memory_budget_bytes / (1024 * 1024),
            "memory_utilization": self.memory_utilization,
            "avg_l1_lookup_ms": self.avg_l1_lookup_ms,
            "avg_l2_lookup_ms": self.avg_l2_lookup_ms,
        }


# =============================================================================
# L1 CACHE: EXACT MATCH (HASH-BASED)
# =============================================================================


class ExactMatchCache(Generic[T]):
    """L1 Cache with exact query matching using hash-based lookup.

    Provides O(1) lookup time for exact query matches with LRU eviction.
    """

    def __init__(
        self,
        max_entries: int = 5000,
        eviction_strategy: InvalidationStrategy = InvalidationStrategy.LRU,
    ):
        """Initialize exact match cache.

        Args:
            max_entries: Maximum number of entries
            eviction_strategy: Eviction strategy to use
        """
        self.max_entries = max_entries
        self.eviction_strategy = eviction_strategy
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lookup_times: List[float] = []

    def _make_key(self, query: str) -> str:
        """Generate cache key from query."""
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[CacheEntry[T]]:
        """Get entry by exact query match.

        Args:
            query: The query string

        Returns:
            CacheEntry if found and not expired, None otherwise
        """
        start_time = time.time()
        key = self._make_key(query)

        with self._lock:
            entry = self._cache.get(key)

            lookup_time = (time.time() - start_time) * 1000
            self._lookup_times.append(lookup_time)
            if len(self._lookup_times) > 1000:
                self._lookup_times = self._lookup_times[-1000:]

            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end for LRU
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1
            return entry

    def put(
        self,
        query: str,
        value: T,
        ttl_seconds: int = 3600,
        query_type: str = "general",
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CacheEntry[T]:
        """Store entry in cache.

        Args:
            query: The query string
            value: Value to cache
            ttl_seconds: Time-to-live in seconds
            query_type: Type of query for TTL lookup
            embedding: Optional embedding for L2 cache
            metadata: Optional metadata

        Returns:
            The created cache entry
        """
        key = self._make_key(query)

        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self.max_entries:
                self._evict_one()

            entry = CacheEntry(
                key=key,
                query=query,
                value=value,
                query_hash=key,
                embedding=embedding,
                ttl_seconds=ttl_seconds,
                query_type=query_type,
                size_bytes=self._estimate_size(value),
                metadata=metadata or {},
            )

            self._cache[key] = entry
            self._cache.move_to_end(key)
            return entry

    def _evict_one(self) -> Optional[str]:
        """Evict one entry based on strategy.

        Returns:
            Key of evicted entry or None
        """
        if not self._cache:
            return None

        self._evictions += 1

        if self.eviction_strategy == InvalidationStrategy.LRU:
            # Remove least recently used (first item)
            key, _ = self._cache.popitem(last=False)
            return key
        elif self.eviction_strategy == InvalidationStrategy.LFU:
            # Remove least frequently used
            min_key = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)
            del self._cache[min_key]
            return min_key
        elif self.eviction_strategy == InvalidationStrategy.TTL:
            # Remove entry closest to expiry
            min_key = min(self._cache.keys(), key=lambda k: self._cache[k].time_until_expiry)
            del self._cache[min_key]
            return min_key
        elif self.eviction_strategy == InvalidationStrategy.FIFO:
            # Remove oldest entry
            min_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
            del self._cache[min_key]
            return min_key
        elif self.eviction_strategy == InvalidationStrategy.SIZE:
            # Remove largest entry
            max_key = max(self._cache.keys(), key=lambda k: self._cache[k].size_bytes)
            del self._cache[max_key]
            return max_key
        else:
            # Default to LRU
            key, _ = self._cache.popitem(last=False)
            return key

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            return sys.getsizeof(value)
        except TypeError:
            # Fallback for complex objects
            return len(str(value))

    def delete(self, query: str) -> bool:
        """Delete entry by query.

        Args:
            query: The query string

        Returns:
            True if entry was deleted, False otherwise
        """
        key = self._make_key(query)
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> int:
        """Clear all entries.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def invalidate_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)

    def invalidate_by_pattern(self, pattern: str) -> int:
        """Remove entries matching a pattern.

        Args:
            pattern: Pattern with * wildcards

        Returns:
            Number of entries removed
        """
        import fnmatch

        with self._lock:
            matching_keys = [
                key for key, entry in self._cache.items()
                if fnmatch.fnmatch(entry.query.lower(), pattern.lower())
            ]
            for key in matching_keys:
                del self._cache[key]
            return len(matching_keys)

    def invalidate_by_age(self, max_age_seconds: float) -> int:
        """Remove entries older than specified age.

        Args:
            max_age_seconds: Maximum age in seconds

        Returns:
            Number of entries removed
        """
        with self._lock:
            cutoff = time.time() - max_age_seconds
            old_keys = [
                key for key, entry in self._cache.items()
                if entry.created_at < cutoff
            ]
            for key in old_keys:
                del self._cache[key]
            return len(old_keys)

    def invalidate_by_query_type(self, query_type: str) -> int:
        """Remove entries of a specific query type.

        Args:
            query_type: Query type to invalidate

        Returns:
            Number of entries removed
        """
        with self._lock:
            matching_keys = [
                key for key, entry in self._cache.items()
                if entry.query_type == query_type
            ]
            for key in matching_keys:
                del self._cache[key]
            return len(matching_keys)

    def get_all_entries(self) -> List[CacheEntry[T]]:
        """Get all cache entries.

        Returns:
            List of all entries
        """
        with self._lock:
            return list(self._cache.values())

    @property
    def size(self) -> int:
        """Get number of entries."""
        return len(self._cache)

    @property
    def memory_used_bytes(self) -> int:
        """Get estimated memory usage."""
        with self._lock:
            return sum(entry.size_bytes for entry in self._cache.values())

    @property
    def avg_lookup_time_ms(self) -> float:
        """Get average lookup time in milliseconds."""
        if not self._lookup_times:
            return 0.0
        return sum(self._lookup_times) / len(self._lookup_times)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "entries": self.size,
            "max_entries": self.max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0.0,
            "memory_used_bytes": self.memory_used_bytes,
            "avg_lookup_time_ms": self.avg_lookup_time_ms,
        }


# =============================================================================
# L2 CACHE: SEMANTIC SIMILARITY (EMBEDDING-BASED)
# =============================================================================


class SemanticCache(Generic[T]):
    """L2 Cache with semantic similarity matching using embeddings.

    Provides cache hits for semantically similar queries (>0.95 threshold).
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProviderProtocol] = None,
        max_entries: int = 1000,
        similarity_threshold: float = 0.95,
    ):
        """Initialize semantic cache.

        Args:
            embedding_provider: Provider for query embeddings
            max_entries: Maximum number of entries
            similarity_threshold: Minimum similarity for cache hit (default 0.95)
        """
        self.embedding_provider = embedding_provider
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self._entries: List[CacheEntry[T]] = []
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._lookup_times: List[float] = []

        # Try to use numpy for vector operations
        try:
            import numpy as np
            self._np = np
            self._use_numpy = True
        except ImportError:
            self._np = None
            self._use_numpy = False

    def _compute_embedding(self, query: str) -> Optional[List[float]]:
        """Compute embedding for query."""
        if self.embedding_provider is None:
            return None

        try:
            result = self.embedding_provider.encode([query])
            if self._use_numpy and hasattr(result, "tolist"):
                return result[0].tolist()
            elif isinstance(result, list) and len(result) > 0:
                if hasattr(result[0], "tolist"):
                    return result[0].tolist()
                return result[0]
            return None
        except Exception as e:
            logger.warning(f"Failed to compute embedding: {e}")
            return None

    def _cosine_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        if self._use_numpy:
            a = self._np.array(embedding1)
            b = self._np.array(embedding2)
            norm_a = self._np.linalg.norm(a)
            norm_b = self._np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(self._np.dot(a, b) / (norm_a * norm_b))
        else:
            # Pure Python fallback
            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
            norm1 = sum(a * a for a in embedding1) ** 0.5
            norm2 = sum(b * b for b in embedding2) ** 0.5
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)

    def get(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
    ) -> Optional[Tuple[CacheEntry[T], float]]:
        """Get entry by semantic similarity.

        Args:
            query: The query string
            query_embedding: Pre-computed embedding (optional)

        Returns:
            Tuple of (CacheEntry, similarity_score) if found, None otherwise
        """
        start_time = time.time()

        # Get or compute embedding
        if query_embedding is None:
            query_embedding = self._compute_embedding(query)

        if query_embedding is None:
            self._misses += 1
            return None

        with self._lock:
            best_entry: Optional[CacheEntry[T]] = None
            best_similarity: float = 0.0

            # Find most similar non-expired entry
            for entry in self._entries:
                if entry.is_expired:
                    continue

                if entry.embedding is None:
                    continue

                similarity = self._cosine_similarity(query_embedding, entry.embedding)

                if similarity >= self.similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_entry = entry

            lookup_time = (time.time() - start_time) * 1000
            self._lookup_times.append(lookup_time)
            if len(self._lookup_times) > 1000:
                self._lookup_times = self._lookup_times[-1000:]

            if best_entry is not None:
                best_entry.touch()
                self._hits += 1
                return (best_entry, best_similarity)

            self._misses += 1
            return None

    def put(self, entry: CacheEntry[T]) -> None:
        """Store entry in semantic cache.

        Args:
            entry: Cache entry with embedding
        """
        if entry.embedding is None:
            entry.embedding = self._compute_embedding(entry.query)

        if entry.embedding is None:
            logger.debug("Cannot store entry without embedding")
            return

        with self._lock:
            # Evict if at capacity (remove oldest)
            while len(self._entries) >= self.max_entries:
                self._entries.sort(key=lambda e: e.last_accessed)
                self._entries.pop(0)

            self._entries.append(entry)

    def delete_by_key(self, key: str) -> bool:
        """Delete entry by key.

        Args:
            key: The entry key

        Returns:
            True if entry was deleted
        """
        with self._lock:
            for i, entry in enumerate(self._entries):
                if entry.key == key:
                    self._entries.pop(i)
                    return True
            return False

    def clear(self) -> int:
        """Clear all entries.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            return count

    def invalidate_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if not e.is_expired]
            return before - len(self._entries)

    @property
    def size(self) -> int:
        """Get number of entries."""
        return len(self._entries)

    @property
    def avg_lookup_time_ms(self) -> float:
        """Get average lookup time in milliseconds."""
        if not self._lookup_times:
            return 0.0
        return sum(self._lookup_times) / len(self._lookup_times)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "entries": self.size,
            "max_entries": self.max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0.0,
            "similarity_threshold": self.similarity_threshold,
            "avg_lookup_time_ms": self.avg_lookup_time_ms,
        }


# =============================================================================
# MEMORY BUDGET MANAGER
# =============================================================================


class MemoryBudgetManager:
    """Manages memory budget for cache system."""

    def __init__(self, budget_mb: float = 256.0):
        """Initialize memory budget manager.

        Args:
            budget_mb: Memory budget in megabytes
        """
        self.budget_bytes = int(budget_mb * 1024 * 1024)
        self._current_bytes = 0
        self._lock = threading.RLock()

    def allocate(self, size_bytes: int) -> bool:
        """Try to allocate memory.

        Args:
            size_bytes: Size to allocate

        Returns:
            True if allocation succeeded
        """
        with self._lock:
            if self._current_bytes + size_bytes <= self.budget_bytes:
                self._current_bytes += size_bytes
                return True
            return False

    def release(self, size_bytes: int) -> None:
        """Release allocated memory.

        Args:
            size_bytes: Size to release
        """
        with self._lock:
            self._current_bytes = max(0, self._current_bytes - size_bytes)

    def set_used(self, size_bytes: int) -> None:
        """Set current usage directly.

        Args:
            size_bytes: Current usage
        """
        with self._lock:
            self._current_bytes = min(size_bytes, self.budget_bytes)

    @property
    def available_bytes(self) -> int:
        """Get available memory in bytes."""
        return max(0, self.budget_bytes - self._current_bytes)

    @property
    def used_bytes(self) -> int:
        """Get used memory in bytes."""
        return self._current_bytes

    @property
    def utilization(self) -> float:
        """Get memory utilization (0-1)."""
        if self.budget_bytes == 0:
            return 0.0
        return self._current_bytes / self.budget_bytes

    def needs_eviction(self, threshold: float = 0.9) -> bool:
        """Check if eviction is needed.

        Args:
            threshold: Utilization threshold (default 0.9)

        Returns:
            True if utilization exceeds threshold
        """
        return self.utilization >= threshold


# =============================================================================
# MAIN RESULT CACHE
# =============================================================================


class ResultCache(Generic[T]):
    """
    Intelligent two-level cache for RAG pipeline results.

    Provides L1 (exact match) and L2 (semantic similarity) caching with:
    - LRU eviction
    - Configurable TTL per query type
    - Memory budget enforcement
    - Multiple invalidation strategies

    Example:
        >>> cache = ResultCache(
        ...     config=ResultCacheConfig(max_entries=5000),
        ...     embedding_provider=my_embedder,
        ... )
        >>> cache.put("What is RAG?", result, query_type="factual")
        >>> hit = cache.get("What is RAG?")
        >>> if hit:
        ...     print(f"Cache hit at {hit.cache_level.value}")
    """

    def __init__(
        self,
        config: Optional[ResultCacheConfig] = None,
        embedding_provider: Optional[EmbeddingProviderProtocol] = None,
    ):
        """Initialize result cache.

        Args:
            config: Cache configuration
            embedding_provider: Provider for query embeddings (required for L2)
        """
        self.config = config or ResultCacheConfig()
        self.embedding_provider = embedding_provider

        # Initialize L1 (exact match) cache
        self._l1_cache: ExactMatchCache[T] = ExactMatchCache(
            max_entries=self.config.max_entries,
            eviction_strategy=self.config.eviction_strategy,
        )

        # Initialize L2 (semantic) cache if enabled
        self._l2_cache: Optional[SemanticCache[T]] = None
        if self.config.enable_l2_cache and embedding_provider is not None:
            self._l2_cache = SemanticCache(
                embedding_provider=embedding_provider,
                max_entries=self.config.l2_max_entries,
                similarity_threshold=self.config.semantic_threshold,
            )

        # Initialize memory budget manager
        self._memory_manager = MemoryBudgetManager(
            budget_mb=self.config.memory_budget_mb
        )

        # Metrics
        self._total_gets = 0
        self._total_puts = 0

        logger.info(
            f"ResultCache initialized: L1={self.config.max_entries} entries, "
            f"L2={'enabled' if self._l2_cache else 'disabled'}, "
            f"budget={self.config.memory_budget_mb}MB"
        )

    def _get_ttl(self, query_type: str) -> int:
        """Get TTL for query type.

        Args:
            query_type: Query type string

        Returns:
            TTL in seconds
        """
        return self.config.ttl_per_query_type.get(
            query_type,
            self.config.default_ttl_seconds
        )

    def get(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
    ) -> Optional[CacheHit[T]]:
        """Get cached result for query.

        Tries L1 (exact match) first, then L2 (semantic) if enabled.

        Args:
            query: The query string
            query_embedding: Pre-computed embedding (optional, for L2)

        Returns:
            CacheHit if found, None otherwise
        """
        self._total_gets += 1
        start_time = time.time()

        # Try L1 cache first (exact match)
        l1_entry = self._l1_cache.get(query)
        if l1_entry is not None:
            lookup_time = (time.time() - start_time) * 1000
            return CacheHit(
                entry=l1_entry,
                cache_level=CacheLevel.L1_EXACT,
                similarity_score=1.0,
                lookup_time_ms=lookup_time,
            )

        # Try L2 cache (semantic match)
        if self._l2_cache is not None:
            l2_result = self._l2_cache.get(query, query_embedding)
            if l2_result is not None:
                entry, similarity = l2_result
                lookup_time = (time.time() - start_time) * 1000
                return CacheHit(
                    entry=entry,
                    cache_level=CacheLevel.L2_SEMANTIC,
                    similarity_score=similarity,
                    lookup_time_ms=lookup_time,
                )

        return None

    def put(
        self,
        query: str,
        value: T,
        query_type: str = "general",
        ttl_seconds: Optional[int] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CacheEntry[T]:
        """Store result in cache.

        Args:
            query: The query string
            value: Value to cache
            query_type: Type of query for TTL lookup
            ttl_seconds: Optional custom TTL (overrides query type TTL)
            embedding: Pre-computed embedding (optional)
            metadata: Optional metadata

        Returns:
            The created cache entry
        """
        self._total_puts += 1

        # Determine TTL
        ttl = ttl_seconds if ttl_seconds is not None else self._get_ttl(query_type)

        # Compute embedding if needed and not provided
        if embedding is None and self._l2_cache is not None:
            embedding = self._l2_cache._compute_embedding(query)

        # Check memory budget
        self._enforce_memory_budget()

        # Store in L1 cache
        entry = self._l1_cache.put(
            query=query,
            value=value,
            ttl_seconds=ttl,
            query_type=query_type,
            embedding=embedding,
            metadata=metadata,
        )

        # Store in L2 cache if enabled
        if self._l2_cache is not None and embedding is not None:
            self._l2_cache.put(entry)

        # Update memory tracking
        self._memory_manager.set_used(
            self._l1_cache.memory_used_bytes
        )

        return entry

    def delete(self, query: str) -> bool:
        """Delete entry from both cache levels.

        Args:
            query: The query string

        Returns:
            True if entry was deleted from any level
        """
        l1_deleted = self._l1_cache.delete(query)

        l2_deleted = False
        if self._l2_cache is not None:
            key = self._l1_cache._make_key(query)
            l2_deleted = self._l2_cache.delete_by_key(key)

        return l1_deleted or l2_deleted

    def clear(self) -> Dict[str, int]:
        """Clear all cache levels.

        Returns:
            Dict with counts per level
        """
        result = {"l1": self._l1_cache.clear(), "l2": 0}

        if self._l2_cache is not None:
            result["l2"] = self._l2_cache.clear()

        self._memory_manager.set_used(0)

        logger.info(f"Cache cleared: L1={result['l1']}, L2={result['l2']}")
        return result

    def _enforce_memory_budget(self) -> int:
        """Enforce memory budget by evicting entries if needed.

        Returns:
            Number of entries evicted
        """
        evicted = 0

        while self._memory_manager.needs_eviction(threshold=0.95):
            # Evict from L1
            key = self._l1_cache._evict_one()
            if key is None:
                break

            evicted += 1

            # Also remove from L2
            if self._l2_cache is not None:
                self._l2_cache.delete_by_key(key)

            # Update memory tracking
            self._memory_manager.set_used(
                self._l1_cache.memory_used_bytes
            )

        return evicted

    # =========================================================================
    # INVALIDATION STRATEGIES
    # =========================================================================

    def invalidate_expired(self) -> Dict[str, int]:
        """Remove all expired entries from both levels.

        Returns:
            Dict with counts per level
        """
        result = {"l1": self._l1_cache.invalidate_expired(), "l2": 0}

        if self._l2_cache is not None:
            result["l2"] = self._l2_cache.invalidate_expired()

        self._memory_manager.set_used(self._l1_cache.memory_used_bytes)

        logger.debug(f"Invalidated expired: L1={result['l1']}, L2={result['l2']}")
        return result

    def invalidate_by_pattern(self, pattern: str) -> int:
        """Remove entries matching a pattern (supports * wildcards).

        Args:
            pattern: Pattern with * wildcards (e.g., "RAG*", "*authentication*")

        Returns:
            Number of entries removed
        """
        count = self._l1_cache.invalidate_by_pattern(pattern)
        self._memory_manager.set_used(self._l1_cache.memory_used_bytes)
        return count

    def invalidate_by_age(self, max_age_seconds: float) -> int:
        """Remove entries older than specified age.

        Args:
            max_age_seconds: Maximum age in seconds

        Returns:
            Number of entries removed
        """
        count = self._l1_cache.invalidate_by_age(max_age_seconds)
        self._memory_manager.set_used(self._l1_cache.memory_used_bytes)
        return count

    def invalidate_by_query_type(self, query_type: str) -> int:
        """Remove entries of a specific query type.

        Args:
            query_type: Query type to invalidate (e.g., "news", "code")

        Returns:
            Number of entries removed
        """
        count = self._l1_cache.invalidate_by_query_type(query_type)
        self._memory_manager.set_used(self._l1_cache.memory_used_bytes)
        return count

    def invalidate_all_for_topic(
        self,
        topic_keywords: List[str],
        threshold: float = 0.8,
    ) -> int:
        """Invalidate entries related to specific topics.

        Args:
            topic_keywords: Keywords to match
            threshold: Minimum keyword match ratio

        Returns:
            Number of entries removed
        """
        count = 0
        entries = self._l1_cache.get_all_entries()

        for entry in entries:
            query_lower = entry.query.lower()
            matches = sum(1 for kw in topic_keywords if kw.lower() in query_lower)
            if matches / len(topic_keywords) >= threshold:
                if self._l1_cache.delete(entry.query):
                    count += 1

        self._memory_manager.set_used(self._l1_cache.memory_used_bytes)
        return count

    # =========================================================================
    # STATISTICS AND MONITORING
    # =========================================================================

    @property
    def stats(self) -> CacheStats:
        """Get comprehensive cache statistics."""
        l1_stats = self._l1_cache.stats
        l2_stats = self._l2_cache.stats if self._l2_cache else {}

        # Calculate age metrics
        entries = self._l1_cache.get_all_entries()
        oldest_age = 0.0
        newest_age = float("inf")
        if entries:
            oldest_age = max(e.age_seconds for e in entries)
            newest_age = min(e.age_seconds for e in entries)

        return CacheStats(
            l1_entries=l1_stats.get("entries", 0),
            l2_entries=l2_stats.get("entries", 0),
            l1_hits=l1_stats.get("hits", 0),
            l2_hits=l2_stats.get("hits", 0),
            misses=l1_stats.get("misses", 0),
            evictions=l1_stats.get("evictions", 0),
            memory_used_bytes=self._memory_manager.used_bytes,
            memory_budget_bytes=self._memory_manager.budget_bytes,
            avg_l1_lookup_ms=l1_stats.get("avg_lookup_time_ms", 0.0),
            avg_l2_lookup_ms=l2_stats.get("avg_lookup_time_ms", 0.0),
            oldest_entry_age_seconds=oldest_age,
            newest_entry_age_seconds=newest_age if newest_age != float("inf") else 0.0,
        )

    def get_hot_queries(self, top_k: int = 10) -> List[Tuple[str, int]]:
        """Get most frequently accessed queries.

        Args:
            top_k: Number of queries to return

        Returns:
            List of (query, access_count) tuples
        """
        entries = self._l1_cache.get_all_entries()
        sorted_entries = sorted(entries, key=lambda e: e.access_count, reverse=True)
        return [(e.query, e.access_count) for e in sorted_entries[:top_k]]

    def get_expiring_soon(
        self,
        threshold_seconds: float = 300,
    ) -> List[Tuple[str, float]]:
        """Get entries that will expire soon.

        Args:
            threshold_seconds: Time threshold in seconds

        Returns:
            List of (query, time_until_expiry) tuples
        """
        entries = self._l1_cache.get_all_entries()
        expiring = [
            (e.query, e.time_until_expiry)
            for e in entries
            if 0 < e.time_until_expiry <= threshold_seconds
        ]
        return sorted(expiring, key=lambda x: x[1])


# =============================================================================
# INTEGRATION WITH RAG PIPELINE
# =============================================================================


class CachedRAGPipelineMixin:
    """
    Mixin to add result caching to RAG pipelines.

    Add this mixin to your pipeline class to enable automatic
    result caching with intelligent invalidation.

    Example:
        class MyPipeline(CachedRAGPipelineMixin, RAGPipeline):
            pass

        pipeline = MyPipeline(...)
        pipeline.init_result_cache(config=ResultCacheConfig())
        result = await pipeline.run("query")  # Automatically cached
    """

    _result_cache: Optional[ResultCache] = None
    _cache_enabled: bool = False

    def init_result_cache(
        self,
        config: Optional[ResultCacheConfig] = None,
        embedding_provider: Optional[EmbeddingProviderProtocol] = None,
    ) -> None:
        """Initialize result caching for this pipeline.

        Args:
            config: Cache configuration
            embedding_provider: Embedding provider for L2 cache
        """
        self._result_cache = ResultCache(
            config=config,
            embedding_provider=embedding_provider,
        )
        self._cache_enabled = True

    def get_cached_result(self, query: str) -> Optional[CacheHit]:
        """Get cached result for query.

        Args:
            query: The query string

        Returns:
            CacheHit if found
        """
        if self._result_cache and self._cache_enabled:
            return self._result_cache.get(query)
        return None

    def cache_result(
        self,
        query: str,
        result: Any,
        query_type: str = "general",
    ) -> None:
        """Cache a pipeline result.

        Args:
            query: The query string
            result: Result to cache
            query_type: Type of query
        """
        if self._result_cache and self._cache_enabled:
            self._result_cache.put(query, result, query_type=query_type)

    def disable_cache(self) -> None:
        """Disable result caching."""
        self._cache_enabled = False

    def enable_cache(self) -> None:
        """Enable result caching."""
        self._cache_enabled = True

    @property
    def cache_stats(self) -> Optional[CacheStats]:
        """Get cache statistics."""
        if self._result_cache:
            return self._result_cache.stats
        return None


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_result_cache(
    max_entries: int = 5000,
    memory_budget_mb: float = 256.0,
    default_ttl_seconds: int = 3600,
    semantic_threshold: float = 0.95,
    enable_l2_cache: bool = True,
    embedding_provider: Optional[EmbeddingProviderProtocol] = None,
    **kwargs,
) -> ResultCache:
    """Factory function to create a configured ResultCache.

    Args:
        max_entries: Maximum L1 cache entries
        memory_budget_mb: Memory budget in MB
        default_ttl_seconds: Default TTL
        semantic_threshold: L2 similarity threshold
        enable_l2_cache: Enable semantic caching
        embedding_provider: Embedding provider for L2
        **kwargs: Additional config options

    Returns:
        Configured ResultCache instance
    """
    config = ResultCacheConfig(
        max_entries=max_entries,
        memory_budget_mb=memory_budget_mb,
        default_ttl_seconds=default_ttl_seconds,
        semantic_threshold=semantic_threshold,
        enable_l2_cache=enable_l2_cache,
        **kwargs,
    )

    return ResultCache(
        config=config,
        embedding_provider=embedding_provider,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main class
    "ResultCache",
    "CachedRAGPipelineMixin",
    # Configuration
    "ResultCacheConfig",
    # Data structures
    "CacheEntry",
    "CacheHit",
    "CacheStats",
    # Enums
    "CacheLevel",
    "InvalidationStrategy",
    "QueryTypeTTL",
    # Cache implementations
    "ExactMatchCache",
    "SemanticCache",
    "MemoryBudgetManager",
    # Factory
    "create_result_cache",
    # Protocols
    "EmbeddingProviderProtocol",
    "PipelineResultProtocol",
]
