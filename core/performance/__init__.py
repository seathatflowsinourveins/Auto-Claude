#!/usr/bin/env python3
"""
V33 Performance Layer - Phase 8 CLI Integration & Performance Optimization.

Provides production-grade performance optimizations:
- HTTP/2 connection pooling
- Thread-safe caching with TTL
- Request deduplication
- Async batch processing
- Lazy SDK loading
- Performance profiling
"""

from .optimizer import (
    # Core classes
    HTTPConnectionPool,
    LRUCache,
    RedisCache,
    CacheManager,
    RequestDeduplicator,
    BatchProcessor,
    LazyLoader,
    Profiler,
    PerformanceManager,
    # Data classes
    CacheEntry,
    TimingRecord,
    # Convenience functions
    get_performance_manager,
    cached,
    timed,
)

__all__ = [
    # Core classes
    "HTTPConnectionPool",
    "LRUCache",
    "RedisCache",
    "CacheManager",
    "RequestDeduplicator",
    "BatchProcessor",
    "LazyLoader",
    "Profiler",
    "PerformanceManager",
    # Data classes
    "CacheEntry",
    "TimingRecord",
    # Convenience functions
    "get_performance_manager",
    "cached",
    "timed",
]
