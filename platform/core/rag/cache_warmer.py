"""
Cache Warming for RAG Pipeline - Pre-populate and Refresh Frequently Accessed Content

This module provides intelligent cache warming capabilities for RAG pipelines:
- Identifies frequently accessed query patterns from history
- Pre-populates caches on startup for faster initial responses
- Background refresh of expiring entries before TTL
- Integration with RAGPipeline and RerankerCache

Architecture:
    +------------------------------------------+
    |            CacheWarmer                   |
    |------------------------------------------|
    | - Pattern Analyzer (access frequency)    |
    | - Startup Warmer (pre-population)        |
    | - Background Refresh (TTL management)    |
    | - Metrics Collector (hit rates, timing)  |
    +------------------------------------------+
              |
              v
    +------------------------------------------+
    |       RAGPipeline / RerankerCache        |
    +------------------------------------------+

Usage:
    from core.rag.cache_warmer import CacheWarmer, CacheWarmerConfig

    config = CacheWarmerConfig(
        max_warm_queries=100,
        refresh_threshold_pct=0.8,  # Refresh when 80% of TTL elapsed
        enable_background_refresh=True,
    )

    warmer = CacheWarmer(
        pipeline=my_pipeline,
        config=config,
    )

    # Warm cache on startup
    await warmer.warm_startup()

    # Start background refresh task
    warmer.start_background_refresh()

    # Record query access for pattern learning
    warmer.record_access("What is RAG?")

    # Manual refresh of specific queries
    await warmer.refresh_queries(["common query 1", "common query 2"])
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from heapq import nlargest
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PROTOCOLS
# =============================================================================

class CacheProtocol(Protocol):
    """Protocol for cache implementations."""

    def get(self, key: str) -> Optional[Any]:
        """Get cached value by key."""
        ...

    def put(self, key: str, value: Any) -> None:
        """Store value in cache."""
        ...

    def clear(self) -> None:
        """Clear all cache entries."""
        ...

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        ...


class PipelineProtocol(Protocol):
    """Protocol for RAG pipeline implementations."""

    async def run(self, query: str, **kwargs) -> Any:
        """Run query through pipeline."""
        ...


class RetrieverProtocol(Protocol):
    """Protocol for retriever implementations."""

    async def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Retrieve documents for query."""
        ...


# =============================================================================
# CONFIGURATION
# =============================================================================

class WarmingStrategy(str, Enum):
    """Cache warming strategies."""
    FREQUENCY = "frequency"  # Warm most frequently accessed queries
    RECENCY = "recency"  # Warm most recently accessed queries
    HYBRID = "hybrid"  # Combination of frequency and recency
    CUSTOM = "custom"  # Custom list of queries


@dataclass
class CacheWarmerConfig:
    """Configuration for cache warming.

    Attributes:
        max_warm_queries: Maximum queries to warm on startup (default: 100)
        refresh_threshold_pct: Refresh when this % of TTL elapsed (default: 0.8)
        enable_background_refresh: Enable automatic background refresh (default: True)
        refresh_interval_seconds: Interval between refresh cycles (default: 60)
        min_access_count: Minimum access count to consider for warming (default: 2)
        recency_weight: Weight for recency in hybrid strategy (default: 0.3)
        frequency_weight: Weight for frequency in hybrid strategy (default: 0.7)
        strategy: Warming strategy to use (default: HYBRID)
        batch_size: Batch size for concurrent warming (default: 10)
        max_history_size: Maximum size of access history (default: 10000)
        warmup_timeout_seconds: Timeout for each warm query (default: 30)
        enable_metrics: Enable detailed metrics collection (default: True)
    """
    max_warm_queries: int = 100
    refresh_threshold_pct: float = 0.8
    enable_background_refresh: bool = True
    refresh_interval_seconds: int = 60
    min_access_count: int = 2
    recency_weight: float = 0.3
    frequency_weight: float = 0.7
    strategy: WarmingStrategy = WarmingStrategy.HYBRID
    batch_size: int = 10
    max_history_size: int = 10000
    warmup_timeout_seconds: float = 30.0
    enable_metrics: bool = True


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class QueryAccess:
    """Record of a query access."""
    query: str
    timestamp: float
    query_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.query_hash:
            self.query_hash = hashlib.md5(self.query.encode()).hexdigest()[:12]


@dataclass
class QueryPattern:
    """Pattern for frequently accessed queries."""
    query: str
    query_hash: str
    access_count: int
    last_access: float
    first_access: float
    avg_latency_ms: float = 0.0
    score: float = 0.0  # Computed priority score

    @property
    def access_frequency(self) -> float:
        """Calculate access frequency (accesses per hour)."""
        duration_hours = max((self.last_access - self.first_access) / 3600, 0.1)
        return self.access_count / duration_hours


@dataclass
class WarmingResult:
    """Result of a cache warming operation."""
    query: str
    success: bool
    latency_ms: float
    cache_hit: bool = False
    error: Optional[str] = None


@dataclass
class WarmingStats:
    """Statistics for cache warming operations."""
    total_warmed: int = 0
    successful: int = 0
    failed: int = 0
    cache_hits: int = 0
    total_latency_ms: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    queries_warmed: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Get success rate."""
        return self.successful / self.total_warmed if self.total_warmed > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Get average latency."""
        return self.total_latency_ms / self.successful if self.successful > 0 else 0.0

    @property
    def duration_ms(self) -> float:
        """Get total duration."""
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_warmed": self.total_warmed,
            "successful": self.successful,
            "failed": self.failed,
            "cache_hits": self.cache_hits,
            "success_rate": self.success_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "duration_ms": self.duration_ms,
        }


# =============================================================================
# PATTERN ANALYZER
# =============================================================================

class PatternAnalyzer:
    """Analyzes query access patterns to identify frequently accessed content."""

    def __init__(self, config: CacheWarmerConfig):
        self.config = config
        self._access_history: List[QueryAccess] = []
        self._pattern_cache: Dict[str, QueryPattern] = {}
        self._latency_samples: Dict[str, List[float]] = defaultdict(list)

    def record_access(
        self,
        query: str,
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a query access for pattern learning.

        Args:
            query: The query that was accessed
            latency_ms: Optional latency of the query execution
            metadata: Optional metadata about the access
        """
        now = time.time()
        access = QueryAccess(
            query=query,
            timestamp=now,
            metadata=metadata or {}
        )

        self._access_history.append(access)

        # Update pattern cache
        if access.query_hash in self._pattern_cache:
            pattern = self._pattern_cache[access.query_hash]
            pattern.access_count += 1
            pattern.last_access = now
        else:
            self._pattern_cache[access.query_hash] = QueryPattern(
                query=query,
                query_hash=access.query_hash,
                access_count=1,
                last_access=now,
                first_access=now,
            )

        # Record latency sample
        if latency_ms > 0:
            samples = self._latency_samples[access.query_hash]
            samples.append(latency_ms)
            # Keep only last 100 samples
            if len(samples) > 100:
                self._latency_samples[access.query_hash] = samples[-100:]

            # Update average latency
            pattern = self._pattern_cache[access.query_hash]
            pattern.avg_latency_ms = sum(samples) / len(samples)

        # Trim history if needed
        if len(self._access_history) > self.config.max_history_size:
            self._trim_history()

    def _trim_history(self) -> None:
        """Trim access history to max size."""
        # Keep most recent entries
        self._access_history = self._access_history[-self.config.max_history_size:]

        # Rebuild pattern cache from remaining history
        seen_hashes: Set[str] = {a.query_hash for a in self._access_history}

        # Remove patterns not in current history
        for qhash in list(self._pattern_cache.keys()):
            if qhash not in seen_hashes:
                del self._pattern_cache[qhash]
                self._latency_samples.pop(qhash, None)

    def get_top_patterns(
        self,
        n: int,
        strategy: Optional[WarmingStrategy] = None
    ) -> List[QueryPattern]:
        """Get top N query patterns based on strategy.

        Args:
            n: Number of patterns to return
            strategy: Warming strategy (defaults to config strategy)

        Returns:
            List of top query patterns
        """
        strategy = strategy or self.config.strategy
        patterns = list(self._pattern_cache.values())

        # Filter by minimum access count
        patterns = [
            p for p in patterns
            if p.access_count >= self.config.min_access_count
        ]

        if not patterns:
            return []

        now = time.time()

        # Score patterns based on strategy
        for pattern in patterns:
            if strategy == WarmingStrategy.FREQUENCY:
                pattern.score = pattern.access_count
            elif strategy == WarmingStrategy.RECENCY:
                # Decay based on time since last access
                hours_since_access = (now - pattern.last_access) / 3600
                pattern.score = 1.0 / (1.0 + hours_since_access)
            elif strategy == WarmingStrategy.HYBRID:
                # Combine frequency and recency
                hours_since_access = (now - pattern.last_access) / 3600
                recency_score = 1.0 / (1.0 + hours_since_access)
                frequency_score = pattern.access_frequency
                pattern.score = (
                    self.config.frequency_weight * frequency_score +
                    self.config.recency_weight * recency_score
                )
            else:
                pattern.score = pattern.access_count

        # Sort by score descending
        patterns.sort(key=lambda p: p.score, reverse=True)

        return patterns[:n]

    def get_expiring_queries(
        self,
        cache_ttl_seconds: int,
        threshold_pct: float
    ) -> List[QueryPattern]:
        """Get queries that will expire soon based on cache TTL.

        Args:
            cache_ttl_seconds: Cache TTL in seconds
            threshold_pct: Threshold percentage (0.0-1.0)

        Returns:
            List of patterns that need refresh
        """
        now = time.time()
        threshold_seconds = cache_ttl_seconds * threshold_pct
        expiring: List[QueryPattern] = []

        for pattern in self._pattern_cache.values():
            time_since_access = now - pattern.last_access
            if time_since_access >= threshold_seconds:
                expiring.append(pattern)

        # Sort by access count (prioritize frequently accessed)
        expiring.sort(key=lambda p: p.access_count, reverse=True)

        return expiring

    def clear_history(self) -> None:
        """Clear all access history."""
        self._access_history.clear()
        self._pattern_cache.clear()
        self._latency_samples.clear()

    @property
    def stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "total_accesses": len(self._access_history),
            "unique_patterns": len(self._pattern_cache),
            "avg_accesses_per_pattern": (
                len(self._access_history) / len(self._pattern_cache)
                if self._pattern_cache else 0
            ),
        }


# =============================================================================
# CACHE WARMER
# =============================================================================

class CacheWarmer:
    """
    Intelligent cache warming for RAG pipelines.

    Pre-populates and refreshes cache entries based on access patterns
    to improve response times for frequently accessed content.

    Example:
        >>> config = CacheWarmerConfig(max_warm_queries=50)
        >>> warmer = CacheWarmer(pipeline=my_pipeline, config=config)
        >>>
        >>> # Warm on startup
        >>> stats = await warmer.warm_startup()
        >>> print(f"Warmed {stats.successful} queries")
        >>>
        >>> # Start background refresh
        >>> warmer.start_background_refresh()
        >>>
        >>> # Record accesses for pattern learning
        >>> warmer.record_access("common query", latency_ms=150)
    """

    def __init__(
        self,
        pipeline: Optional[PipelineProtocol] = None,
        retrievers: Optional[List[RetrieverProtocol]] = None,
        cache: Optional[CacheProtocol] = None,
        config: Optional[CacheWarmerConfig] = None,
        custom_queries: Optional[List[str]] = None,
        cache_ttl_seconds: int = 300,
    ):
        """Initialize cache warmer.

        Args:
            pipeline: RAG pipeline to warm cache for
            retrievers: Optional list of retrievers to warm
            cache: Optional cache implementation to warm
            config: Warming configuration
            custom_queries: Custom list of queries to warm
            cache_ttl_seconds: Cache TTL for refresh calculations
        """
        self.pipeline = pipeline
        self.retrievers = retrievers or []
        self.cache = cache
        self.config = config or CacheWarmerConfig()
        self.custom_queries = custom_queries or []
        self.cache_ttl_seconds = cache_ttl_seconds

        self.analyzer = PatternAnalyzer(self.config)
        self._background_task: Optional[asyncio.Task] = None
        self._running = False

        # Metrics
        self._total_warm_count = 0
        self._total_refresh_count = 0
        self._last_warm_stats: Optional[WarmingStats] = None

    def record_access(
        self,
        query: str,
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a query access for pattern learning.

        Args:
            query: The query that was accessed
            latency_ms: Optional latency of the query execution
            metadata: Optional metadata about the access
        """
        self.analyzer.record_access(query, latency_ms, metadata)

    async def warm_startup(
        self,
        additional_queries: Optional[List[str]] = None
    ) -> WarmingStats:
        """Warm cache on startup with frequently accessed queries.

        Args:
            additional_queries: Optional additional queries to warm

        Returns:
            WarmingStats with results
        """
        stats = WarmingStats(start_time=time.time())
        queries_to_warm: List[str] = []

        # Get queries based on strategy
        if self.config.strategy == WarmingStrategy.CUSTOM:
            queries_to_warm = self.custom_queries[:self.config.max_warm_queries]
        else:
            patterns = self.analyzer.get_top_patterns(
                self.config.max_warm_queries,
                self.config.strategy
            )
            queries_to_warm = [p.query for p in patterns]

        # Add additional queries
        if additional_queries:
            queries_to_warm.extend(additional_queries)

        # Add custom queries if not already custom strategy
        if self.config.strategy != WarmingStrategy.CUSTOM and self.custom_queries:
            queries_to_warm.extend(self.custom_queries)

        # Deduplicate while preserving order
        seen: Set[str] = set()
        unique_queries: List[str] = []
        for q in queries_to_warm:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)

        # Limit to max
        unique_queries = unique_queries[:self.config.max_warm_queries]

        logger.info(f"Warming cache with {len(unique_queries)} queries")

        # Warm in batches
        results = await self._warm_batch(unique_queries)

        for result in results:
            stats.total_warmed += 1
            if result.success:
                stats.successful += 1
                stats.total_latency_ms += result.latency_ms
                stats.queries_warmed.append(result.query)
            else:
                stats.failed += 1
            if result.cache_hit:
                stats.cache_hits += 1

        stats.end_time = time.time()
        self._last_warm_stats = stats
        self._total_warm_count += stats.successful

        logger.info(
            f"Cache warming complete: {stats.successful}/{stats.total_warmed} "
            f"successful in {stats.duration_ms:.0f}ms"
        )

        return stats

    async def _warm_batch(self, queries: List[str]) -> List[WarmingResult]:
        """Warm a batch of queries concurrently.

        Args:
            queries: List of queries to warm

        Returns:
            List of warming results
        """
        results: List[WarmingResult] = []

        for i in range(0, len(queries), self.config.batch_size):
            batch = queries[i:i + self.config.batch_size]
            batch_tasks = [self._warm_single(q) for q in batch]

            batch_results = await asyncio.gather(
                *batch_tasks,
                return_exceptions=True
            )

            for query, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    results.append(WarmingResult(
                        query=query,
                        success=False,
                        latency_ms=0,
                        error=str(result)
                    ))
                else:
                    results.append(result)

        return results

    async def _warm_single(self, query: str) -> WarmingResult:
        """Warm a single query.

        Args:
            query: Query to warm

        Returns:
            WarmingResult
        """
        start_time = time.time()

        try:
            # Try pipeline first
            if self.pipeline:
                await asyncio.wait_for(
                    self.pipeline.run(query),
                    timeout=self.config.warmup_timeout_seconds
                )
                latency_ms = (time.time() - start_time) * 1000
                return WarmingResult(
                    query=query,
                    success=True,
                    latency_ms=latency_ms,
                    cache_hit=False  # First run is always a miss
                )

            # Fall back to retrievers
            for retriever in self.retrievers:
                await asyncio.wait_for(
                    retriever.retrieve(query, top_k=5),
                    timeout=self.config.warmup_timeout_seconds
                )

            latency_ms = (time.time() - start_time) * 1000
            return WarmingResult(
                query=query,
                success=True,
                latency_ms=latency_ms,
                cache_hit=False
            )

        except asyncio.TimeoutError:
            return WarmingResult(
                query=query,
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error="Timeout"
            )
        except Exception as e:
            logger.warning(f"Failed to warm query '{query[:50]}...': {e}")
            return WarmingResult(
                query=query,
                success=False,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )

    async def refresh_expiring(self) -> WarmingStats:
        """Refresh cache entries that are about to expire.

        Returns:
            WarmingStats with refresh results
        """
        stats = WarmingStats(start_time=time.time())

        # Get expiring queries
        expiring = self.analyzer.get_expiring_queries(
            self.cache_ttl_seconds,
            self.config.refresh_threshold_pct
        )

        if not expiring:
            stats.end_time = time.time()
            return stats

        queries = [p.query for p in expiring[:self.config.max_warm_queries]]

        logger.debug(f"Refreshing {len(queries)} expiring cache entries")

        results = await self._warm_batch(queries)

        for result in results:
            stats.total_warmed += 1
            if result.success:
                stats.successful += 1
                stats.total_latency_ms += result.latency_ms
                stats.queries_warmed.append(result.query)
            else:
                stats.failed += 1

        stats.end_time = time.time()
        self._total_refresh_count += stats.successful

        return stats

    async def refresh_queries(self, queries: List[str]) -> WarmingStats:
        """Manually refresh specific queries.

        Args:
            queries: List of queries to refresh

        Returns:
            WarmingStats with results
        """
        stats = WarmingStats(start_time=time.time())

        results = await self._warm_batch(queries)

        for result in results:
            stats.total_warmed += 1
            if result.success:
                stats.successful += 1
                stats.total_latency_ms += result.latency_ms
                stats.queries_warmed.append(result.query)
            else:
                stats.failed += 1

        stats.end_time = time.time()
        self._total_refresh_count += stats.successful

        return stats

    def start_background_refresh(self) -> None:
        """Start background refresh task."""
        if not self.config.enable_background_refresh:
            logger.warning("Background refresh is disabled in config")
            return

        if self._running:
            logger.warning("Background refresh already running")
            return

        self._running = True
        self._background_task = asyncio.create_task(self._background_refresh_loop())
        logger.info("Started background cache refresh task")

    def stop_background_refresh(self) -> None:
        """Stop background refresh task."""
        self._running = False
        if self._background_task:
            self._background_task.cancel()
            self._background_task = None
            logger.info("Stopped background cache refresh task")

    async def _background_refresh_loop(self) -> None:
        """Background loop for refreshing expiring entries."""
        while self._running:
            try:
                await asyncio.sleep(self.config.refresh_interval_seconds)

                if not self._running:
                    break

                stats = await self.refresh_expiring()

                if stats.successful > 0:
                    logger.debug(
                        f"Background refresh: {stats.successful} entries "
                        f"in {stats.duration_ms:.0f}ms"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background refresh error: {e}")
                await asyncio.sleep(5)  # Brief pause before retry

    def get_warming_candidates(self, n: int = 10) -> List[QueryPattern]:
        """Get top N candidates for cache warming.

        Args:
            n: Number of candidates to return

        Returns:
            List of query patterns
        """
        return self.analyzer.get_top_patterns(n)

    def add_custom_queries(self, queries: List[str]) -> None:
        """Add custom queries to warm.

        Args:
            queries: List of queries to add
        """
        self.custom_queries.extend(queries)

    def clear_custom_queries(self) -> None:
        """Clear custom query list."""
        self.custom_queries.clear()

    def clear_history(self) -> None:
        """Clear access history."""
        self.analyzer.clear_history()

    @property
    def is_running(self) -> bool:
        """Check if background refresh is running."""
        return self._running

    @property
    def stats(self) -> Dict[str, Any]:
        """Get warmer statistics."""
        return {
            "total_warmed": self._total_warm_count,
            "total_refreshed": self._total_refresh_count,
            "background_running": self._running,
            "analyzer_stats": self.analyzer.stats,
            "last_warm_stats": (
                self._last_warm_stats.to_dict()
                if self._last_warm_stats else None
            ),
            "config": {
                "max_warm_queries": self.config.max_warm_queries,
                "refresh_threshold_pct": self.config.refresh_threshold_pct,
                "strategy": self.config.strategy.value,
            }
        }


# =============================================================================
# PIPELINE INTEGRATION
# =============================================================================

class CacheWarmingMixin:
    """
    Mixin to add cache warming capabilities to RAG pipelines.

    Add this mixin to your pipeline class to enable automatic
    cache warming and access tracking.

    Example:
        class MyPipeline(CacheWarmingMixin, RAGPipeline):
            pass

        pipeline = MyPipeline(...)
        await pipeline.init_cache_warming()
        result = await pipeline.run("query")  # Automatically tracked
    """

    _cache_warmer: Optional[CacheWarmer] = None
    _cache_warming_initialized: bool = False

    async def init_cache_warming(
        self,
        config: Optional[CacheWarmerConfig] = None,
        custom_queries: Optional[List[str]] = None,
        warm_on_init: bool = True,
        start_background_refresh: bool = True,
    ) -> WarmingStats:
        """Initialize cache warming for this pipeline.

        Args:
            config: Warming configuration
            custom_queries: Custom queries to warm
            warm_on_init: Whether to warm cache immediately
            start_background_refresh: Whether to start background refresh

        Returns:
            WarmingStats if warm_on_init, else empty stats
        """
        self._cache_warmer = CacheWarmer(
            pipeline=self,  # type: ignore
            config=config,
            custom_queries=custom_queries,
        )
        self._cache_warming_initialized = True

        stats = WarmingStats()

        if warm_on_init:
            stats = await self._cache_warmer.warm_startup()

        if start_background_refresh:
            self._cache_warmer.start_background_refresh()

        return stats

    def track_query_access(
        self,
        query: str,
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track a query access for pattern learning.

        Args:
            query: The query that was accessed
            latency_ms: Latency of the query
            metadata: Optional metadata
        """
        if self._cache_warmer:
            self._cache_warmer.record_access(query, latency_ms, metadata)

    def stop_cache_warming(self) -> None:
        """Stop cache warming and background refresh."""
        if self._cache_warmer:
            self._cache_warmer.stop_background_refresh()
            self._cache_warming_initialized = False

    @property
    def cache_warming_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache warming statistics."""
        if self._cache_warmer:
            return self._cache_warmer.stats
        return None


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_cache_warmer(
    pipeline: Optional[PipelineProtocol] = None,
    retrievers: Optional[List[RetrieverProtocol]] = None,
    max_warm_queries: int = 100,
    enable_background_refresh: bool = True,
    refresh_interval_seconds: int = 60,
    strategy: WarmingStrategy = WarmingStrategy.HYBRID,
    custom_queries: Optional[List[str]] = None,
    **kwargs
) -> CacheWarmer:
    """Factory function to create a configured CacheWarmer.

    Args:
        pipeline: RAG pipeline to warm
        retrievers: List of retrievers to warm
        max_warm_queries: Maximum queries to warm on startup
        enable_background_refresh: Enable automatic background refresh
        refresh_interval_seconds: Interval between refresh cycles
        strategy: Warming strategy to use
        custom_queries: Custom list of queries to warm
        **kwargs: Additional config options

    Returns:
        Configured CacheWarmer instance
    """
    config = CacheWarmerConfig(
        max_warm_queries=max_warm_queries,
        enable_background_refresh=enable_background_refresh,
        refresh_interval_seconds=refresh_interval_seconds,
        strategy=strategy,
        **kwargs
    )

    return CacheWarmer(
        pipeline=pipeline,
        retrievers=retrievers,
        config=config,
        custom_queries=custom_queries,
    )


async def warm_pipeline_cache(
    pipeline: PipelineProtocol,
    queries: List[str],
    batch_size: int = 10,
    timeout_seconds: float = 30.0,
) -> WarmingStats:
    """Convenience function to warm a pipeline with specific queries.

    Args:
        pipeline: Pipeline to warm
        queries: Queries to warm
        batch_size: Batch size for concurrent warming
        timeout_seconds: Timeout per query

    Returns:
        WarmingStats with results
    """
    config = CacheWarmerConfig(
        batch_size=batch_size,
        warmup_timeout_seconds=timeout_seconds,
        strategy=WarmingStrategy.CUSTOM,
    )

    warmer = CacheWarmer(
        pipeline=pipeline,
        config=config,
        custom_queries=queries,
    )

    return await warmer.warm_startup()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main classes
    "CacheWarmer",
    "PatternAnalyzer",
    "CacheWarmingMixin",
    # Configuration
    "CacheWarmerConfig",
    "WarmingStrategy",
    # Data structures
    "QueryAccess",
    "QueryPattern",
    "WarmingResult",
    "WarmingStats",
    # Factory functions
    "create_cache_warmer",
    "warm_pipeline_cache",
    # Protocols
    "CacheProtocol",
    "PipelineProtocol",
    "RetrieverProtocol",
]
