"""
MCP Tool Result Cache - V38 Architecture (ADR-028)

Semantic caching for MCP tool results using embedding similarity.
Reduces redundant API calls and latency for repeated research patterns.

Expected Gains:
- 40-60% reduction in redundant API calls
- 90%+ cache hit rate for repeated research queries
- $500-2000/month API cost savings

Usage:
    cache = MCPToolCache(embedding_layer)
    await cache.initialize()

    # Get or execute
    result = await cache.get_or_execute(
        server="exa",
        tool="search",
        args={"query": "LangGraph patterns"},
        executor=lambda: exa.search("LangGraph patterns"),
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached tool result."""
    key: str
    server: str
    tool: str
    args_hash: str
    args_embedding: Optional[List[float]]
    result: Any
    created_at: float
    ttl_seconds: float
    hit_count: int = 0
    last_hit_at: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() - self.created_at > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Age of entry in seconds."""
        return time.time() - self.created_at

    def record_hit(self) -> None:
        """Record a cache hit."""
        self.hit_count += 1
        self.last_hit_at = time.time()


@dataclass
class CacheStats:
    """Statistics for the cache."""
    total_requests: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_entries: int = 0
    total_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        """Overall cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return (self.exact_hits + self.semantic_hits) / self.total_requests

    @property
    def semantic_hit_rate(self) -> float:
        """Semantic (similarity-based) hit rate."""
        hits = self.exact_hits + self.semantic_hits
        if hits == 0:
            return 0.0
        return self.semantic_hits / hits


class MCPToolCache:
    """
    Semantic caching for MCP tool results.

    Uses embedding similarity to cache semantically similar queries,
    reducing API costs and latency for repeated research patterns.

    Cache Strategies:
    - Exact match: Hash-based lookup (O(1))
    - Semantic match: Embedding similarity (O(n) but cached)

    Eviction Policy:
    - LRU (Least Recently Used) when at capacity
    - TTL-based expiration
    """

    # Tools that benefit from semantic caching
    CACHEABLE_TOOLS = {
        "exa": ["search", "search_and_contents", "code_search"],
        "tavily": ["search", "extract"],
        "perplexity": ["search", "ask", "research"],
        "firecrawl": ["scrape"],
        "jina": ["read_url", "search_web"],
        "context7": ["query-docs"],
        "serper": ["google_search"],
    }

    # Tools that should NOT be cached (dynamic/stateful)
    UNCACHEABLE_TOOLS = {
        "github": ["*"],  # All GitHub tools are stateful
        "git": ["*"],
        "sqlite": ["*"],
        "filesystem": ["*"],
    }

    # Default TTLs by tool type (seconds)
    DEFAULT_TTLS = {
        "search": 3600,      # 1 hour for search results
        "research": 7200,    # 2 hours for research
        "scrape": 86400,     # 24 hours for scraped content
        "docs": 604800,      # 1 week for documentation
        "default": 3600,
    }

    def __init__(
        self,
        embedding_layer: Optional[Any] = None,
        similarity_threshold: float = 0.92,
        max_cache_size: int = 10000,
        default_ttl_seconds: float = 3600.0,
        semantic_cache_enabled: bool = True,
    ):
        """
        Initialize the MCP tool cache.

        Args:
            embedding_layer: EmbeddingLayer for semantic matching (optional)
            similarity_threshold: Minimum similarity for semantic match (0-1)
            max_cache_size: Maximum number of cached entries
            default_ttl_seconds: Default TTL for cached entries
            semantic_cache_enabled: Enable semantic (embedding-based) caching
        """
        self._embedding_layer = embedding_layer
        self._similarity_threshold = similarity_threshold
        self._max_cache_size = max_cache_size
        self._default_ttl = default_ttl_seconds
        self._semantic_enabled = semantic_cache_enabled and embedding_layer is not None

        # Cache storage
        self._exact_cache: Dict[str, CacheEntry] = {}
        self._semantic_cache: List[CacheEntry] = []  # For similarity search
        self._stats = CacheStats()
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the cache (and embedding layer if needed)."""
        if self._initialized:
            return

        if self._semantic_enabled and self._embedding_layer:
            try:
                await self._embedding_layer.initialize()
            except Exception as e:
                logger.warning(f"[MCP_CACHE] Embedding init failed, semantic caching disabled: {e}")
                self._semantic_enabled = False

        self._initialized = True
        logger.info(
            f"[MCP_CACHE] Initialized (semantic={self._semantic_enabled}, "
            f"threshold={self._similarity_threshold}, max_size={self._max_cache_size})"
        )

    async def get_or_execute(
        self,
        server: str,
        tool: str,
        args: Dict[str, Any],
        executor: Callable[[], Any],
        ttl_seconds: Optional[float] = None,
        force_refresh: bool = False,
    ) -> Any:
        """
        Get cached result or execute and cache.

        Args:
            server: MCP server name
            tool: Tool name
            args: Tool arguments
            executor: Async function to execute if not cached
            ttl_seconds: TTL override
            force_refresh: Force execution even if cached

        Returns:
            Tool result (from cache or executor)
        """
        self._stats.total_requests += 1

        # Check if tool is cacheable
        if not self._is_cacheable(server, tool):
            return await executor()

        # Compute cache key
        cache_key = self._compute_cache_key(server, tool, args)

        if not force_refresh:
            # Try exact cache match
            cached = await self._get_exact(cache_key)
            if cached is not None:
                self._stats.exact_hits += 1
                logger.debug(f"[MCP_CACHE] Exact hit for {server}:{tool}")
                return cached

            # Try semantic cache match
            if self._semantic_enabled:
                cached = await self._get_semantic(server, tool, args)
                if cached is not None:
                    self._stats.semantic_hits += 1
                    logger.debug(f"[MCP_CACHE] Semantic hit for {server}:{tool}")
                    return cached

        # Cache miss - execute
        self._stats.misses += 1
        result = await executor()

        # Cache the result
        ttl = ttl_seconds or self._get_ttl(tool)
        await self._cache_result(server, tool, args, cache_key, result, ttl)

        return result

    async def invalidate(
        self,
        server: Optional[str] = None,
        tool: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Invalidate cache entries matching criteria.

        Args:
            server: Server name filter (None = all)
            tool: Tool name filter (None = all)
            args: Arguments filter (None = all)

        Returns:
            Number of entries invalidated
        """
        async with self._lock:
            to_remove = []

            for key, entry in self._exact_cache.items():
                if server and entry.server != server:
                    continue
                if tool and entry.tool != tool:
                    continue
                if args:
                    args_hash = self._hash_args(args)
                    if entry.args_hash != args_hash:
                        continue
                to_remove.append(key)

            for key in to_remove:
                del self._exact_cache[key]

            # Also remove from semantic cache
            self._semantic_cache = [
                e for e in self._semantic_cache
                if not (
                    (server is None or e.server == server) and
                    (tool is None or e.tool == tool)
                )
            ]

            logger.info(f"[MCP_CACHE] Invalidated {len(to_remove)} entries")
            return len(to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_requests": self._stats.total_requests,
            "exact_hits": self._stats.exact_hits,
            "semantic_hits": self._stats.semantic_hits,
            "misses": self._stats.misses,
            "hit_rate": round(self._stats.hit_rate * 100, 2),
            "semantic_hit_rate": round(self._stats.semantic_hit_rate * 100, 2),
            "total_entries": len(self._exact_cache),
            "semantic_entries": len(self._semantic_cache),
            "evictions": self._stats.evictions,
            "semantic_enabled": self._semantic_enabled,
            "similarity_threshold": self._similarity_threshold,
            "max_size": self._max_cache_size,
        }

    def _is_cacheable(self, server: str, tool: str) -> bool:
        """Check if a tool call is cacheable."""
        # Check uncacheable list
        if server in self.UNCACHEABLE_TOOLS:
            uncacheable = self.UNCACHEABLE_TOOLS[server]
            if "*" in uncacheable or tool in uncacheable:
                return False

        # Check cacheable list (if specified, must be in list)
        if server in self.CACHEABLE_TOOLS:
            return tool in self.CACHEABLE_TOOLS[server]

        # Default: cache
        return True

    def _compute_cache_key(self, server: str, tool: str, args: Dict[str, Any]) -> str:
        """Compute cache key from server, tool, and args."""
        args_hash = self._hash_args(args)
        return f"{server}:{tool}:{args_hash}"

    def _hash_args(self, args: Dict[str, Any]) -> str:
        """Hash arguments for cache key."""
        # Sort keys for consistent hashing
        sorted_args = json.dumps(args, sort_keys=True, default=str)
        return hashlib.sha256(sorted_args.encode()).hexdigest()[:16]

    def _get_ttl(self, tool: str) -> float:
        """Get TTL for a tool type."""
        for key, ttl in self.DEFAULT_TTLS.items():
            if key in tool.lower():
                return ttl
        return self.DEFAULT_TTLS["default"]

    async def _get_exact(self, cache_key: str) -> Optional[Any]:
        """Get exact cache match."""
        async with self._lock:
            entry = self._exact_cache.get(cache_key)
            if entry and not entry.is_expired:
                entry.record_hit()
                return entry.result
            elif entry:
                # Expired - remove
                del self._exact_cache[cache_key]
            return None

    async def _get_semantic(
        self,
        server: str,
        tool: str,
        args: Dict[str, Any]
    ) -> Optional[Any]:
        """Get semantic cache match using embedding similarity."""
        if not self._embedding_layer:
            return None

        # Get embedding for query args
        query_text = self._args_to_text(args)
        if not query_text:
            return None

        try:
            query_embedding = await self._embedding_layer.embed(
                [query_text],
                input_type="query"
            )
            if not query_embedding.embeddings:
                return None

            query_vec = query_embedding.embeddings[0]

            # Search semantic cache
            best_match: Optional[CacheEntry] = None
            best_similarity = 0.0

            async with self._lock:
                for entry in self._semantic_cache:
                    # Filter by server and tool
                    if entry.server != server or entry.tool != tool:
                        continue

                    # Skip expired
                    if entry.is_expired:
                        continue

                    # Skip if no embedding
                    if not entry.args_embedding:
                        continue

                    # Compute similarity
                    similarity = self._cosine_similarity(query_vec, entry.args_embedding)
                    if similarity >= self._similarity_threshold and similarity > best_similarity:
                        best_match = entry
                        best_similarity = similarity

                if best_match:
                    best_match.record_hit()
                    logger.debug(
                        f"[MCP_CACHE] Semantic match: similarity={best_similarity:.3f}"
                    )
                    return best_match.result

        except Exception as e:
            logger.warning(f"[MCP_CACHE] Semantic lookup failed: {e}")

        return None

    async def _cache_result(
        self,
        server: str,
        tool: str,
        args: Dict[str, Any],
        cache_key: str,
        result: Any,
        ttl_seconds: float,
    ) -> None:
        """Cache a tool result."""
        # Compute embedding for semantic cache
        args_embedding = None
        if self._semantic_enabled and self._embedding_layer:
            try:
                query_text = self._args_to_text(args)
                if query_text:
                    embedding_result = await self._embedding_layer.embed(
                        [query_text],
                        input_type="query"
                    )
                    if embedding_result.embeddings:
                        args_embedding = embedding_result.embeddings[0]
            except Exception as e:
                logger.debug(f"[MCP_CACHE] Embedding failed for cache: {e}")

        entry = CacheEntry(
            key=cache_key,
            server=server,
            tool=tool,
            args_hash=self._hash_args(args),
            args_embedding=args_embedding,
            result=result,
            created_at=time.time(),
            ttl_seconds=ttl_seconds,
        )

        async with self._lock:
            # Evict if at capacity
            if len(self._exact_cache) >= self._max_cache_size:
                await self._evict_lru()

            # Add to exact cache
            self._exact_cache[cache_key] = entry

            # Add to semantic cache if we have embedding
            if args_embedding:
                self._semantic_cache.append(entry)
                # Limit semantic cache size
                if len(self._semantic_cache) > self._max_cache_size:
                    self._semantic_cache = self._semantic_cache[-self._max_cache_size:]

    async def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        if not self._exact_cache:
            return

        # Sort by last hit time
        sorted_entries = sorted(
            self._exact_cache.items(),
            key=lambda x: x[1].last_hit_at
        )

        # Evict oldest 10%
        evict_count = max(1, len(sorted_entries) // 10)
        for key, _ in sorted_entries[:evict_count]:
            del self._exact_cache[key]
            self._stats.evictions += 1

    def _args_to_text(self, args: Dict[str, Any]) -> str:
        """Convert args to searchable text."""
        # Extract common query-like fields
        text_parts = []
        for key in ["query", "q", "input", "text", "question", "search"]:
            if key in args:
                text_parts.append(str(args[key]))

        # Include all string values if no query fields
        if not text_parts:
            for value in args.values():
                if isinstance(value, str) and len(value) > 5:
                    text_parts.append(value)

        return " ".join(text_parts)

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b) or len(a) == 0:
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)


# Global cache instance
_cache: Optional[MCPToolCache] = None


def get_mcp_cache(embedding_layer: Optional[Any] = None) -> MCPToolCache:
    """Get global MCP tool cache."""
    global _cache
    if _cache is None:
        _cache = MCPToolCache(embedding_layer=embedding_layer)
    return _cache


async def cached_tool_call(
    server: str,
    tool: str,
    args: Dict[str, Any],
    executor: Callable[[], Any],
    ttl_seconds: Optional[float] = None,
    embedding_layer: Optional[Any] = None,
) -> Any:
    """
    Convenience function for cached tool calls.

    Usage:
        result = await cached_tool_call(
            server="exa",
            tool="search",
            args={"query": "LangGraph"},
            executor=lambda: exa_client.search("LangGraph"),
        )
    """
    cache = get_mcp_cache(embedding_layer)
    if not cache._initialized:
        await cache.initialize()

    return await cache.get_or_execute(
        server=server,
        tool=tool,
        args=args,
        executor=executor,
        ttl_seconds=ttl_seconds,
    )
