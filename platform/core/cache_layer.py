"""
Cache Layer - Research Pipeline Caching System.

Provides multi-level caching for the research pipeline:
- Memory cache: Fast, in-process caching with LRU eviction
- Disk cache: Persistent caching across sessions
- TTL support: Time-based expiration
- Compression: Optional compression for large content

Caches:
- Search results (by query hash)
- Scraped content (by URL hash)
- Crawled pages (by URL + params hash)
- API responses (generic)
"""

import os
import sys
import json
import time
import hashlib
import gzip
import logging
from typing import Any, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import OrderedDict
from abc import ABC, abstractmethod
import threading

logger = logging.getLogger(__name__)

# =============================================================================
# Module Availability Flag
# =============================================================================

CACHE_AVAILABLE = True  # Cache layer is always available as core module

# =============================================================================
# Cache Configuration
# =============================================================================

@dataclass
class CacheConfig:
    """Configuration for cache layer."""
    # Memory cache settings
    memory_max_items: int = 1000
    memory_enabled: bool = True

    # Disk cache settings
    disk_enabled: bool = True
    disk_cache_dir: Optional[str] = None  # Default: ~/.cache/unleash-research
    disk_compress: bool = True
    disk_max_size_mb: int = 500

    # TTL settings (in seconds)
    search_ttl: int = 3600  # 1 hour for search results
    scrape_ttl: int = 86400  # 24 hours for scraped content
    crawl_ttl: int = 86400  # 24 hours for crawled content
    api_ttl: int = 1800  # 30 minutes for API responses

    # Cache key prefix
    key_prefix: str = "unleash"

    def __post_init__(self):
        if self.disk_cache_dir is None:
            self.disk_cache_dir = str(Path.home() / ".cache" / "unleash-research")


# =============================================================================
# Cache Entry Data Model
# =============================================================================

@dataclass
class CacheEntry:
    """A cached item with metadata."""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    ttl: int = 3600
    hit_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    compressed: bool = False
    source: str = "memory"

    @property
    def is_expired(self) -> bool:
        return time.time() > self.created_at + self.ttl

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    def touch(self):
        """Update access time and hit count."""
        self.last_accessed = time.time()
        self.hit_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at,
            "ttl": self.ttl,
            "hit_count": self.hit_count,
            "last_accessed": self.last_accessed,
            "compressed": self.compressed,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        return cls(**data)


# =============================================================================
# Abstract Cache Backend
# =============================================================================

class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get a cache entry by key."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set a cache entry."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        pass

    @abstractmethod
    def clear(self) -> int:
        """Clear all cache entries, return count deleted."""
        pass

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


# =============================================================================
# Memory Cache Backend (LRU)
# =============================================================================

class MemoryCache(CacheBackend):
    """
    In-memory LRU cache with thread safety.

    Uses OrderedDict for O(1) access and LRU eviction.
    """

    def __init__(self, max_items: int = 1000):
        self.max_items = max_items
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[CacheEntry]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1
            return entry

    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        with self._lock:
            # Evict LRU items if at capacity
            while len(self._cache) >= self.max_items:
                self._cache.popitem(last=False)

            entry = CacheEntry(
                key=key,
                value=value,
                ttl=ttl,
                source="memory",
            )
            self._cache[key] = entry
            self._cache.move_to_end(key)
            return True

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> int:
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "type": "memory",
                "items": len(self._cache),
                "max_items": self.max_items,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }


# =============================================================================
# Disk Cache Backend
# =============================================================================

class DiskCache(CacheBackend):
    """
    Disk-based cache with optional compression.

    Stores each entry as a separate JSON file for simplicity.
    """

    def __init__(
        self,
        cache_dir: str,
        compress: bool = True,
        max_size_mb: int = 500,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.compress = compress
        self.max_size_mb = max_size_mb
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def _key_to_path(self, key: str) -> Path:
        """Convert cache key to file path."""
        # V37: Use SHA-256 for safe filesystem name (CVE-UNLEASH-002 remediation)
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        ext = ".json.gz" if self.compress else ".json"
        return self.cache_dir / f"{key_hash}{ext}"

    def get(self, key: str) -> Optional[CacheEntry]:
        path = self._key_to_path(key)

        with self._lock:
            if not path.exists():
                self._misses += 1
                return None

            try:
                if self.compress:
                    with gzip.open(path, "rt", encoding="utf-8") as f:
                        data = json.load(f)
                else:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                entry = CacheEntry.from_dict(data)

                if entry.is_expired:
                    path.unlink(missing_ok=True)
                    self._misses += 1
                    return None

                entry.touch()
                entry.source = "disk"
                self._hits += 1
                return entry

            except Exception as e:
                logger.debug(f"Failed to read cache entry {key}: {e}")
                self._misses += 1
                return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        path = self._key_to_path(key)

        with self._lock:
            try:
                entry = CacheEntry(
                    key=key,
                    value=value,
                    ttl=ttl,
                    compressed=self.compress,
                    source="disk",
                )

                if self.compress:
                    with gzip.open(path, "wt", encoding="utf-8") as f:
                        json.dump(entry.to_dict(), f)
                else:
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(entry.to_dict(), f)

                return True

            except Exception as e:
                logger.debug(f"Failed to write cache entry {key}: {e}")
                return False

    def delete(self, key: str) -> bool:
        path = self._key_to_path(key)

        with self._lock:
            if path.exists():
                path.unlink()
                return True
            return False

    def clear(self) -> int:
        with self._lock:
            count = 0
            for path in self.cache_dir.glob("*.json*"):
                try:
                    path.unlink()
                    count += 1
                except Exception:
                    pass
            return count

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            files = list(self.cache_dir.glob("*.json*"))
            total_size = sum(f.stat().st_size for f in files if f.exists())
            total = self._hits + self._misses

            return {
                "type": "disk",
                "items": len(files),
                "size_mb": total_size / (1024 * 1024),
                "max_size_mb": self.max_size_mb,
                "compressed": self.compress,
                "cache_dir": str(self.cache_dir),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }

    def cleanup_expired(self) -> int:
        """Remove expired entries from disk."""
        removed = 0
        with self._lock:
            for path in self.cache_dir.glob("*.json*"):
                try:
                    if self.compress:
                        with gzip.open(path, "rt", encoding="utf-8") as f:
                            data = json.load(f)
                    else:
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                    entry = CacheEntry.from_dict(data)
                    if entry.is_expired:
                        path.unlink()
                        removed += 1
                except Exception:
                    pass
        return removed


# =============================================================================
# Research Cache (Main Interface)
# =============================================================================

class ResearchCache:
    """
    Multi-level cache for research pipeline operations.

    Provides a unified interface with:
    - L1: Memory cache (fast, limited size)
    - L2: Disk cache (persistent, larger)
    - Specialized methods for search, scrape, crawl caching
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()

        # Initialize backends
        self._memory: Optional[MemoryCache] = None
        self._disk: Optional[DiskCache] = None

        if self.config.memory_enabled:
            self._memory = MemoryCache(max_items=self.config.memory_max_items)

        if self.config.disk_enabled and self.config.disk_cache_dir:
            self._disk = DiskCache(
                cache_dir=self.config.disk_cache_dir,
                compress=self.config.disk_compress,
                max_size_mb=self.config.disk_max_size_mb,
            )

        logger.info(f"[CACHE] ResearchCache initialized (memory={self.config.memory_enabled}, disk={self.config.disk_enabled})")

    def _make_key(self, prefix: str, *parts) -> str:
        """Create a cache key from parts."""
        key_data = ":".join(str(p) for p in parts)
        key_hash = hashlib.sha256(key_data.encode()).hexdigest()[:16]
        return f"{self.config.key_prefix}:{prefix}:{key_hash}"

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (L1 then L2)."""
        # Try memory first
        if self._memory:
            entry = self._memory.get(key)
            if entry:
                return entry.value

        # Try disk
        if self._disk:
            entry = self._disk.get(key)
            if entry:
                # Promote to memory cache
                if self._memory:
                    self._memory.set(key, entry.value, entry.ttl)
                return entry.value

        return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache (both L1 and L2)."""
        success = False

        if self._memory:
            success = self._memory.set(key, value, ttl) or success

        if self._disk:
            success = self._disk.set(key, value, ttl) or success

        return success

    def delete(self, key: str) -> bool:
        """Delete from both cache levels."""
        deleted = False

        if self._memory:
            deleted = self._memory.delete(key) or deleted

        if self._disk:
            deleted = self._disk.delete(key) or deleted

        return deleted

    # -------------------------------------------------------------------------
    # Specialized Cache Methods
    # -------------------------------------------------------------------------

    def cache_search(self, query: str, result: Dict[str, Any]) -> bool:
        """Cache a search result."""
        key = self._make_key("search", query.lower().strip())
        return self.set(key, result, ttl=self.config.search_ttl)

    def get_search(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached search result."""
        key = self._make_key("search", query.lower().strip())
        return self.get(key)

    def cache_scrape(self, url: str, content: Dict[str, Any]) -> bool:
        """Cache scraped content for a URL."""
        key = self._make_key("scrape", url)
        return self.set(key, content, ttl=self.config.scrape_ttl)

    def get_scrape(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached scraped content."""
        key = self._make_key("scrape", url)
        return self.get(key)

    def cache_crawl(self, url: str, params: Dict[str, Any], content: Dict[str, Any]) -> bool:
        """Cache crawled content with parameters."""
        params_str = json.dumps(params, sort_keys=True)
        key = self._make_key("crawl", url, params_str)
        return self.set(key, content, ttl=self.config.crawl_ttl)

    def get_crawl(self, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached crawled content."""
        params_str = json.dumps(params, sort_keys=True)
        key = self._make_key("crawl", url, params_str)
        return self.get(key)

    def cache_api(self, endpoint: str, params: Dict[str, Any], result: Any) -> bool:
        """Cache generic API response."""
        params_str = json.dumps(params, sort_keys=True)
        key = self._make_key("api", endpoint, params_str)
        return self.set(key, result, ttl=self.config.api_ttl)

    def get_api(self, endpoint: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached API response."""
        params_str = json.dumps(params, sort_keys=True)
        key = self._make_key("api", endpoint, params_str)
        return self.get(key)

    # -------------------------------------------------------------------------
    # Cache Decorator
    # -------------------------------------------------------------------------

    def cached(
        self,
        key_func: Optional[Callable] = None,
        ttl: Optional[int] = None,
        prefix: str = "func",
    ):
        """
        Decorator for caching function results.

        Args:
            key_func: Function to generate cache key from args/kwargs
            ttl: Time-to-live in seconds
            prefix: Key prefix

        Example:
            @cache.cached(ttl=3600)
            async def expensive_operation(query):
                ...
        """
        def decorator(func):
            import functools
            import asyncio
            import inspect

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    key_parts = [str(a) for a in args] + [f"{k}={v}" for k, v in sorted(kwargs.items())]
                    cache_key = self._make_key(prefix, func.__name__, *key_parts)

                # Check cache
                cached = self.get(cache_key)
                if cached is not None:
                    logger.debug(f"[CACHE HIT] {func.__name__}")
                    return cached

                # Execute function
                result = await func(*args, **kwargs)

                # Store in cache
                cache_ttl = ttl or self.config.api_ttl
                self.set(cache_key, result, ttl=cache_ttl)

                return result

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    key_parts = [str(a) for a in args] + [f"{k}={v}" for k, v in sorted(kwargs.items())]
                    cache_key = self._make_key(prefix, func.__name__, *key_parts)

                # Check cache
                cached = self.get(cache_key)
                if cached is not None:
                    logger.debug(f"[CACHE HIT] {func.__name__}")
                    return cached

                # Execute function
                result = func(*args, **kwargs)

                # Store in cache
                cache_ttl = ttl or self.config.api_ttl
                self.set(cache_key, result, ttl=cache_ttl)

                return result

            if inspect.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper

        return decorator

    # -------------------------------------------------------------------------
    # Management
    # -------------------------------------------------------------------------

    def clear(self) -> Dict[str, int]:
        """Clear all caches."""
        result = {"memory": 0, "disk": 0}

        if self._memory:
            result["memory"] = self._memory.clear()

        if self._disk:
            result["disk"] = self._disk.clear()

        logger.info(f"[CACHE] Cleared: memory={result['memory']}, disk={result['disk']}")
        return result

    def cleanup(self) -> Dict[str, int]:
        """Remove expired entries."""
        result = {"memory": 0, "disk": 0}

        # Memory cache auto-cleans on access, but we can force it
        if self._memory:
            with self._memory._lock:
                expired_keys = [
                    k for k, v in self._memory._cache.items()
                    if v.is_expired
                ]
                for k in expired_keys:
                    del self._memory._cache[k]
                result["memory"] = len(expired_keys)

        if self._disk:
            result["disk"] = self._disk.cleanup_expired()

        logger.info(f"[CACHE] Cleanup: memory={result['memory']}, disk={result['disk']} expired entries removed")
        return result

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "config": asdict(self.config),
            "memory": self._memory.stats() if self._memory else None,
            "disk": self._disk.stats() if self._disk else None,
        }


# =============================================================================
# Singleton Instance
# =============================================================================

_research_cache: Optional[ResearchCache] = None


def get_cache(config: Optional[CacheConfig] = None) -> ResearchCache:
    """Get or create the global research cache instance."""
    global _research_cache
    if _research_cache is None:
        _research_cache = ResearchCache(config)
    return _research_cache


def clear_cache() -> Dict[str, int]:
    """Clear the global cache."""
    return get_cache().clear()


def cache_stats() -> Dict[str, Any]:
    """Get global cache statistics."""
    return get_cache().stats()


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Availability
    "CACHE_AVAILABLE",

    # Config
    "CacheConfig",
    "CacheEntry",

    # Backends
    "CacheBackend",
    "MemoryCache",
    "DiskCache",

    # Main Cache
    "ResearchCache",

    # Convenience
    "get_cache",
    "clear_cache",
    "cache_stats",
]
