#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pydantic>=2.0.0",
#     "httpx>=0.25.0",
#     "numpy>=1.24.0",
# ]
# ///
"""
Advanced Memory System - Letta Integration, Semantic Search, Consolidation

Enhances the base memory system with:
- Letta Cloud integration for persistent long-term memory
- Semantic search using vector embeddings
- Memory consolidation (summarization, compression, pruning)
- Cross-session memory continuity

Based on:
- Letta SDK: https://docs.letta.com/guides/agents/memory
- MemGPT: Memory Management for LLMs
- Graphiti: Temporal knowledge graphs

Usage:
    from advanced_memory import AdvancedMemorySystem, LettaClient

    # Create with Letta integration
    memory = AdvancedMemorySystem(
        agent_id="agent-001",
        letta_api_key="sk-...",
        enable_semantic_search=True,
    )

    # Store with embeddings
    await memory.store_semantic("conversation-123", content, metadata)

    # Search semantically
    results = await memory.search_semantic("user preferences", limit=5)

    # Consolidate old memories
    await memory.consolidate(older_than_days=7)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections import OrderedDict

# Module logger
logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, cast
import httpx

# V121: Circuit Breaker for API failure resilience
from .resilience import CircuitBreaker, CircuitOpenError

# V122: Observability integration for metrics
try:
    from .observability import get_observability
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False


# =============================================================================
# V122: Memory System Metrics
# =============================================================================

class MemoryMetrics:
    """V122: Centralized metrics collector for memory system operations.

    Tracks:
    - Embedding operations (calls, latency, tokens, errors)
    - Cache performance (hits, misses, hit rate, evictions)
    - Circuit breaker state and transitions
    - Semantic search operations
    - Memory consolidation operations
    """

    _instance: Optional["MemoryMetrics"] = None

    def __new__(cls) -> "MemoryMetrics":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True

        # Internal tracking counters (for get_all_stats and testing)
        self._internal_embed_calls = 0
        self._internal_embed_errors = 0
        self._internal_embed_tokens = 0
        self._internal_embed_cache_hits = 0
        self._internal_embed_cache_misses = 0
        self._internal_embed_latencies: list = []
        self._internal_search_calls = 0
        self._internal_search_latencies: list = []
        self._internal_cache_size = 0
        self._internal_cache_ttl_evictions = 0
        self._internal_cache_lru_evictions = 0
        self._internal_circuit_state = "closed"

        if not OBSERVABILITY_AVAILABLE:
            self._embed_calls = None
            self._embed_latency = None
            self._embed_tokens = None
            self._embed_errors = None
            self._cache_hits = None
            self._cache_misses = None
            self._cache_size = None
            self._cache_evictions = None
            self._circuit_state = None
            self._circuit_transitions = None
            self._search_calls = None
            self._search_latency = None
            self._consolidation_calls = None
            return

        obs = get_observability()

        # Embedding metrics
        self._embed_calls = obs.counter(
            "memory_embed_calls_total",
            "Total embedding API calls",
            ["provider", "model", "cache_status"]
        )
        self._embed_latency = obs.histogram(
            "memory_embed_latency_seconds",
            "Embedding operation latency",
            ["provider", "model"],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )
        self._embed_tokens = obs.counter(
            "memory_embed_tokens_total",
            "Total tokens used for embeddings",
            ["provider", "model"]
        )
        self._embed_errors = obs.counter(
            "memory_embed_errors_total",
            "Total embedding errors",
            ["provider", "model", "error_type"]
        )

        # Cache metrics
        self._cache_hits = obs.counter(
            "memory_cache_hits_total",
            "Embedding cache hits",
            ["model"]
        )
        self._cache_misses = obs.counter(
            "memory_cache_misses_total",
            "Embedding cache misses",
            ["model"]
        )
        self._cache_size = obs.gauge(
            "memory_cache_size",
            "Current embedding cache size",
            []
        )
        self._cache_evictions = obs.counter(
            "memory_cache_evictions_total",
            "Embedding cache evictions (LRU or TTL)",
            ["reason"]
        )

        # Circuit breaker metrics
        self._circuit_state = obs.gauge(
            "memory_circuit_state",
            "Circuit breaker state (0=closed, 1=half_open, 2=open)",
            ["provider"]
        )
        self._circuit_transitions = obs.counter(
            "memory_circuit_transitions_total",
            "Circuit breaker state transitions",
            ["provider", "from_state", "to_state"]
        )

        # Semantic search metrics
        self._search_calls = obs.counter(
            "memory_search_calls_total",
            "Semantic search operations",
            ["index"]
        )
        self._search_latency = obs.histogram(
            "memory_search_latency_seconds",
            "Semantic search latency",
            ["index"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25]
        )

        # Memory consolidation metrics
        self._consolidation_calls = obs.counter(
            "memory_consolidation_calls_total",
            "Memory consolidation operations",
            ["strategy"]
        )

    def record_embed_call(
        self,
        provider: str,
        model: str,
        cache_hit: bool,
        latency_seconds: float,
        tokens_used: int = 0,
    ) -> None:
        """Record an embedding operation."""
        # Internal tracking (always runs)
        self._internal_embed_calls += 1
        self._internal_embed_latencies.append(latency_seconds)
        self._internal_embed_tokens += tokens_used
        if cache_hit:
            self._internal_embed_cache_hits += 1
        else:
            self._internal_embed_cache_misses += 1

        if not OBSERVABILITY_AVAILABLE:
            return

        cache_status = "hit" if cache_hit else "miss"
        if self._embed_calls:
            self._embed_calls.inc(provider=provider, model=model, cache_status=cache_status)
        if self._embed_latency and not cache_hit:
            self._embed_latency.observe(latency_seconds, provider=provider, model=model)
        if self._embed_tokens and tokens_used > 0:
            self._embed_tokens.inc(tokens_used, provider=provider, model=model)

        # Also track cache separately
        if cache_hit:
            if self._cache_hits:
                self._cache_hits.inc(model=model)
        else:
            if self._cache_misses:
                self._cache_misses.inc(model=model)

    def record_embed_error(
        self,
        provider: str,
        model: str,
        error_type: str,
    ) -> None:
        """Record an embedding error."""
        self._internal_embed_errors += 1
        if self._embed_errors:
            self._embed_errors.inc(provider=provider, model=model, error_type=error_type)

    def update_cache_size(self, size: int) -> None:
        """Update current cache size."""
        self._internal_cache_size = size
        if self._cache_size:
            self._cache_size.set(size)

    def record_cache_eviction(self, reason: str) -> None:
        """Record a cache eviction (LRU or TTL)."""
        if reason == "ttl":
            self._internal_cache_ttl_evictions += 1
        elif reason == "lru":
            self._internal_cache_lru_evictions += 1
        if self._cache_evictions:
            self._cache_evictions.inc(reason=reason)

    def update_circuit_state(self, provider: str, state: int) -> None:
        """Update circuit breaker state (0=closed, 1=half_open, 2=open)."""
        if self._circuit_state:
            self._circuit_state.set(state, provider=provider)

    def record_circuit_transition(
        self,
        provider: str,
        from_state: str,
        to_state: str,
    ) -> None:
        """Record a circuit breaker state transition."""
        if self._circuit_transitions:
            self._circuit_transitions.inc(
                provider=provider, from_state=from_state, to_state=to_state
            )

    def record_search(self, index: str, latency_seconds: float) -> None:
        """Record a semantic search operation."""
        self._internal_search_calls += 1
        self._internal_search_latencies.append(latency_seconds)
        if self._search_calls:
            self._search_calls.inc(index=index)
        if self._search_latency:
            self._search_latency.observe(latency_seconds, index=index)

    def record_consolidation(self, strategy: str) -> None:
        """Record a memory consolidation operation."""
        if self._consolidation_calls:
            self._consolidation_calls.inc(strategy=strategy)

    # Public property accessors for internal counters
    @property
    def embed_calls(self) -> int:
        return self._internal_embed_calls

    @property
    def embed_errors(self) -> int:
        return self._internal_embed_errors

    @property
    def embed_tokens_total(self) -> int:
        return self._internal_embed_tokens

    @property
    def embed_cache_hits(self) -> int:
        return self._internal_embed_cache_hits

    @property
    def embed_cache_misses(self) -> int:
        return self._internal_embed_cache_misses

    @property
    def embed_latencies(self) -> list:
        return self._internal_embed_latencies

    @property
    def search_calls(self) -> int:
        return self._internal_search_calls

    @property
    def search_latencies(self) -> list:
        return self._internal_search_latencies

    @property
    def cache_ttl_evictions(self) -> int:
        return self._internal_cache_ttl_evictions

    @property
    def cache_lru_evictions(self) -> int:
        return self._internal_cache_lru_evictions

    @property
    def circuit_state(self) -> str:
        return self._internal_circuit_state

    def _compute_percentile(self, latencies: list, p: float) -> float:
        """Compute latency percentile in milliseconds."""
        if not latencies:
            return 0.0
        sorted_l = sorted(latencies)
        idx = int(len(sorted_l) * p / 100.0)
        idx = min(idx, len(sorted_l) - 1)
        return sorted_l[idx] * 1000.0  # convert to ms

    def get_all_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics."""
        stats = {
            "embedding": {
                "calls": self._internal_embed_calls,
                "errors": self._internal_embed_errors,
                "tokens_total": self._internal_embed_tokens,
                "cache_hits": self._internal_embed_cache_hits,
                "cache_misses": self._internal_embed_cache_misses,
                "latency_p50_ms": self._compute_percentile(self._internal_embed_latencies, 50),
                "latency_p95_ms": self._compute_percentile(self._internal_embed_latencies, 95),
                "latency_p99_ms": self._compute_percentile(self._internal_embed_latencies, 99),
            },
            "cache": {
                **(_embedding_cache.stats if _embedding_cache else {}),
                "size": self._internal_cache_size,
                "ttl_evictions": self._internal_cache_ttl_evictions,
                "lru_evictions": self._internal_cache_lru_evictions,
            },
            "circuit_breaker": {
                "state": self._internal_circuit_state,
            },
            "search": {
                "calls": self._internal_search_calls,
                "latency_p50_ms": self._compute_percentile(self._internal_search_latencies, 50),
                "latency_p95_ms": self._compute_percentile(self._internal_search_latencies, 95),
            },
        }

        # Add circuit breaker stats if OpenAIEmbeddingProvider is available
        try:
            stats["circuit_breaker"]["openai"] = OpenAIEmbeddingProvider.get_circuit_stats()
        except Exception as e:
            stats["circuit_breaker"]["openai_error"] = str(e)

        return stats

    def reset(self) -> None:
        """Reset all internal counters (for testing)."""
        self._internal_embed_calls = 0
        self._internal_embed_errors = 0
        self._internal_embed_tokens = 0
        self._internal_embed_cache_hits = 0
        self._internal_embed_cache_misses = 0
        self._internal_embed_latencies = []
        self._internal_search_calls = 0
        self._internal_search_latencies = []
        self._internal_cache_size = 0
        self._internal_cache_ttl_evictions = 0
        self._internal_cache_lru_evictions = 0
        self._internal_circuit_state = "closed"


# Global metrics instance (V122)
_memory_metrics = MemoryMetrics()


# =============================================================================
# V120: Embedding Cache with TTL
# =============================================================================

class EmbeddingCache:
    """V120: LRU cache for embeddings with TTL support.

    Prevents redundant embedding API calls for the same text.
    Thread-safe using OrderedDict for LRU eviction.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 3600.0,  # 1 hour default
    ):
        self._cache: OrderedDict[str, Tuple[List[float], float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    def _make_key(self, text: str, model: str) -> str:
        """Create cache key from text and model."""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Get cached embedding if exists and not expired."""
        key = self._make_key(text, model)
        if key not in self._cache:
            self._misses += 1
            return None

        embedding, timestamp = self._cache[key]

        # Check TTL
        if time.time() - timestamp > self._ttl:
            del self._cache[key]
            self._misses += 1
            # V122: Record TTL eviction
            _memory_metrics.record_cache_eviction("ttl")
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        return embedding

    def set(self, text: str, model: str, embedding: List[float]) -> None:
        """Cache an embedding."""
        key = self._make_key(text, model)

        # Evict LRU items if at capacity
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
            # V122: Record LRU eviction
            _memory_metrics.record_cache_eviction("lru")

        self._cache[key] = (embedding, time.time())
        # V122: Update cache size metric
        _memory_metrics.update_cache_size(len(self._cache))

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "size": len(self._cache),
            "max_size": self._max_size,
        }

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0


# Global embedding cache (V120: shared across providers)
_embedding_cache = EmbeddingCache(max_size=2000, ttl_seconds=3600.0)

from .memory import (
    MemorySystem,
    CoreMemory,
    ArchivalMemory,
    TemporalGraph,
)


# =============================================================================
# Embedding Generation
# =============================================================================

class EmbeddingModel(str, Enum):
    """V123: Supported embedding models with multi-provider support."""
    # Local models (sentence-transformers)
    LOCAL_MINILM = "all-MiniLM-L6-v2"  # 384 dims, local, fast
    LOCAL_MPNET = "all-mpnet-base-v2"  # 768 dims, local, quality
    LOCAL_BGE_SMALL = "BAAI/bge-small-en-v1.5"  # 384 dims, local, optimized
    LOCAL_BGE_BASE = "BAAI/bge-base-en-v1.5"  # 768 dims, local, balanced

    # OpenAI models
    OPENAI_ADA = "text-embedding-ada-002"  # 1536 dims
    OPENAI_3_SMALL = "text-embedding-3-small"  # 1536 dims
    OPENAI_3_LARGE = "text-embedding-3-large"  # 3072 dims

    # Voyage AI models (V123)
    VOYAGE_CODE_3 = "voyage-code-3"  # 1024 dims, optimized for code
    VOYAGE_3_LARGE = "voyage-3-large"  # 1024 dims (configurable), best quality
    VOYAGE_3_5 = "voyage-3.5"  # 1024 dims, general purpose
    VOYAGE_3_5_LITE = "voyage-3.5-lite"  # 1024 dims, cost-effective
    VOYAGE_2 = "voyage-2"  # 1024 dims, legacy

    # Letta
    LETTA_DEFAULT = "letta-default"  # Letta's default


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""
    text: str
    embedding: List[float]
    model: str
    dimensions: int
    tokens_used: int = 0


class EmbeddingProvider(ABC):
    """Abstract embedding provider."""

    @abstractmethod
    async def embed(self, text: str) -> EmbeddingResult:
        """Generate embedding for text."""
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts."""
        pass


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding provider using simple hash-based embeddings.

    For development/testing. Production should use sentence-transformers
    or API-based providers.
    """

    def __init__(self, dimensions: int = 384):
        self._dimensions = dimensions

    async def embed(self, text: str) -> EmbeddingResult:
        """V122: Generate pseudo-embedding from text hash with metrics."""
        start_time = time.time()  # V122

        # Create deterministic embedding from text hash
        text_hash = hashlib.sha256(text.encode()).digest()

        # Expand hash to desired dimensions
        embedding = []
        for i in range(self._dimensions):
            byte_idx = i % len(text_hash)
            # Normalize to [-1, 1]
            value = (text_hash[byte_idx] - 128) / 128.0
            embedding.append(value)

        latency = time.time() - start_time  # V122

        # V122: Record embed call
        _memory_metrics.record_embed_call(
            provider="local", model="local-hash", cache_hit=False,
            latency_seconds=latency, tokens_used=0
        )

        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model="local-hash",
            dimensions=self._dimensions,
        )

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """V119+V122: Generate embeddings for batch with metrics."""
        # Use asyncio.gather for parallel embedding generation
        tasks = [self.embed(text) for text in texts]
        return await asyncio.gather(*tasks)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider with connection pooling (V118), caching (V120), and circuit breaker (V121)."""

    # V118: Shared client for connection pooling
    _shared_client: Optional[httpx.AsyncClient] = None

    # V121: Circuit breaker for API resilience (shared across all instances)
    _circuit_breaker: CircuitBreaker = CircuitBreaker(
        failure_threshold=5,       # Open after 5 consecutive failures
        success_threshold=2,       # Close after 2 successes in half-open
        recovery_timeout=30.0,     # Try again after 30 seconds
        half_open_max_calls=3,     # Allow 3 test calls in half-open
    )

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }.get(model, 1536)

    def _get_client(self) -> httpx.AsyncClient:
        """V118: Get shared client with connection pooling."""
        if OpenAIEmbeddingProvider._shared_client is None:
            OpenAIEmbeddingProvider._shared_client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                timeout=httpx.Timeout(30.0),
            )
        return OpenAIEmbeddingProvider._shared_client

    async def embed(self, text: str) -> EmbeddingResult:
        """V120+V121+V122: Generate embedding via OpenAI API with caching, circuit breaker, and metrics."""
        # V120: Check cache first
        cached = _embedding_cache.get(text, self._model)
        if cached is not None:
            # V122: Record cache hit
            _memory_metrics.record_embed_call(
                provider="openai", model=self._model, cache_hit=True,
                latency_seconds=0.0, tokens_used=0
            )
            return EmbeddingResult(
                text=text,
                embedding=cached,
                model=self._model,
                dimensions=len(cached),
                tokens_used=0,  # No tokens used for cache hit
            )

        # Cache miss - call API with circuit breaker (V121)
        client = self._get_client()
        start_time = time.time()  # V122: Track latency

        try:
            async with self._circuit_breaker:
                response = await client.post(
                    f"{self._base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"input": text, "model": self._model},
                )

                if response.status_code != 200:
                    raise RuntimeError(f"Embedding API error: {response.text}")

                data = response.json()
        except CircuitOpenError:
            # V122: Record circuit open error
            _memory_metrics.record_embed_error(
                provider="openai", model=self._model, error_type="circuit_open"
            )
            raise
        except Exception as e:
            # V122: Record other errors
            _memory_metrics.record_embed_error(
                provider="openai", model=self._model, error_type=type(e).__name__
            )
            raise

        embedding = data["data"][0]["embedding"]
        tokens_used = data.get("usage", {}).get("total_tokens", 0)
        latency = time.time() - start_time  # V122

        # V122: Record successful API call
        _memory_metrics.record_embed_call(
            provider="openai", model=self._model, cache_hit=False,
            latency_seconds=latency, tokens_used=tokens_used
        )

        # V120: Cache the result
        _embedding_cache.set(text, self._model, embedding)

        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model=self._model,
            dimensions=len(embedding),
            tokens_used=tokens_used,
        )

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """V120+V121+V122: Generate embeddings for batch with caching, circuit breaker, and metrics."""
        if not texts:
            return []

        # V120: Check cache for all texts first
        results: List[Optional[EmbeddingResult]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []
        cache_hits = 0  # V122: Track cache hits

        for i, text in enumerate(texts):
            cached = _embedding_cache.get(text, self._model)
            if cached is not None:
                results[i] = EmbeddingResult(
                    text=text,
                    embedding=cached,
                    model=self._model,
                    dimensions=len(cached),
                    tokens_used=0,
                )
                cache_hits += 1  # V122
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # V122: Record all cache hits
        for _ in range(cache_hits):
            _memory_metrics.record_embed_call(
                provider="openai", model=self._model, cache_hit=True,
                latency_seconds=0.0, tokens_used=0
            )

        # If all cached, return early
        if not uncached_texts:
            return cast(List[EmbeddingResult], results)

        # Make API call only for uncached texts with circuit breaker (V121)
        client = self._get_client()
        start_time = time.time()  # V122: Track latency

        try:
            async with self._circuit_breaker:
                response = await client.post(
                    f"{self._base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"input": uncached_texts, "model": self._model},
                )

                if response.status_code != 200:
                    raise RuntimeError(f"Embedding API error: {response.text}")

                data = response.json()
        except CircuitOpenError:
            # V122: Record circuit open errors for each uncached text
            for _ in uncached_texts:
                _memory_metrics.record_embed_error(
                    provider="openai", model=self._model, error_type="circuit_open"
                )
            raise
        except Exception as e:
            # V122: Record other errors
            for _ in uncached_texts:
                _memory_metrics.record_embed_error(
                    provider="openai", model=self._model, error_type=type(e).__name__
                )
            raise

        total_tokens = data.get("usage", {}).get("total_tokens", 0)
        tokens_per_text = total_tokens // len(uncached_texts) if uncached_texts else 0
        latency = time.time() - start_time  # V122

        # V122: Record API call (once for batch, latency per item)
        latency_per_item = latency / len(uncached_texts)
        for _ in uncached_texts:
            _memory_metrics.record_embed_call(
                provider="openai", model=self._model, cache_hit=False,
                latency_seconds=latency_per_item, tokens_used=tokens_per_text
            )

        # Process API results and cache them
        for j, item in enumerate(data["data"]):
            original_idx = uncached_indices[j]
            text = uncached_texts[j]
            embedding = item["embedding"]

            # V120: Cache the new embedding
            _embedding_cache.set(text, self._model, embedding)

            results[original_idx] = EmbeddingResult(
                text=text,
                embedding=embedding,
                model=self._model,
                dimensions=len(embedding),
                tokens_used=tokens_per_text,
            )

        return cast(List[EmbeddingResult], results)

    @classmethod
    def get_circuit_stats(cls) -> Dict[str, Any]:
        """V121: Get circuit breaker statistics for monitoring."""
        stats = cls._circuit_breaker.stats
        return {
            "state": cls._circuit_breaker.state.value,
            "total_calls": stats.total_calls,
            "successful_calls": stats.successful_calls,
            "failed_calls": stats.failed_calls,
            "rejected_calls": stats.rejected_calls,
            "failure_rate": stats.failure_rate,
            "consecutive_failures": stats.consecutive_failures,
            "time_in_open": stats.time_in_open,
        }


# =============================================================================
# V123: Voyage AI Embedding Provider
# =============================================================================

class VoyageEmbeddingProvider(EmbeddingProvider):
    """
    V123: Voyage AI embedding provider for high-quality code and general embeddings.

    Supports models:
    - voyage-code-3: Optimized for code retrieval (recommended for code)
    - voyage-3.5: Best general-purpose and multilingual
    - voyage-3.5-lite: Cost-effective with good quality
    - voyage-3-large: Highest quality with configurable dimensions

    Features:
    - Connection pooling (V118 pattern)
    - Embedding cache integration (V120)
    - Circuit breaker resilience (V121)
    - Comprehensive metrics (V122)
    """

    # V118: Shared client for connection pooling
    _shared_client: Optional[httpx.AsyncClient] = None

    # V121: Circuit breaker for API resilience
    _circuit_breaker: CircuitBreaker = CircuitBreaker(
        failure_threshold=5,
        success_threshold=2,
        recovery_timeout=30.0,
        half_open_max_calls=3,
    )

    # Model dimensions lookup
    _MODEL_DIMENSIONS = {
        "voyage-code-3": 1024,
        "voyage-3-large": 1024,  # Default, supports 256, 512, 2048
        "voyage-3.5": 1024,
        "voyage-3.5-lite": 1024,
        "voyage-2": 1024,
        "voyage-finance-2": 1024,
        "voyage-law-2": 1024,
        "voyage-multilingual-2": 1024,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-code-3",
        input_type: str = "document",  # "document" or "query"
    ):
        self._api_key = api_key
        self._model = model
        self._input_type = input_type
        self._dimensions = self._MODEL_DIMENSIONS.get(model, 1024)
        self._base_url = "https://api.voyageai.com/v1"

    def _get_client(self) -> httpx.AsyncClient:
        """V118: Get shared client with connection pooling."""
        if VoyageEmbeddingProvider._shared_client is None:
            VoyageEmbeddingProvider._shared_client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                timeout=httpx.Timeout(60.0),  # Voyage can be slower for large batches
            )
        return VoyageEmbeddingProvider._shared_client

    async def embed(self, text: str) -> EmbeddingResult:
        """V120+V121+V122: Generate embedding via Voyage AI with caching, circuit breaker, metrics."""
        # V120: Check cache first
        cached = _embedding_cache.get(text, self._model)
        if cached is not None:
            _memory_metrics.record_embed_call(
                provider="voyage", model=self._model, cache_hit=True,
                latency_seconds=0.0, tokens_used=0
            )
            return EmbeddingResult(
                text=text,
                embedding=cached,
                model=self._model,
                dimensions=len(cached),
                tokens_used=0,
            )

        # Cache miss - call API with circuit breaker
        client = self._get_client()
        start_time = time.time()

        try:
            async with self._circuit_breaker:
                response = await client.post(
                    f"{self._base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "input": [text],
                        "model": self._model,
                        "input_type": self._input_type,
                    },
                )

                if response.status_code != 200:
                    raise RuntimeError(f"Voyage API error: {response.text}")

                data = response.json()
                embedding = data["data"][0]["embedding"]
                tokens_used = data.get("usage", {}).get("total_tokens", 0)

            latency = time.time() - start_time

            # V120: Cache the result
            _embedding_cache.set(text, self._model, embedding)

            # V122: Record success
            _memory_metrics.record_embed_call(
                provider="voyage", model=self._model, cache_hit=False,
                latency_seconds=latency, tokens_used=tokens_used
            )

            return EmbeddingResult(
                text=text,
                embedding=embedding,
                model=self._model,
                dimensions=len(embedding),
                tokens_used=tokens_used,
            )

        except CircuitOpenError:
            _memory_metrics.record_embed_error("voyage", self._model, "circuit_open")
            raise
        except Exception as e:
            _memory_metrics.record_embed_error("voyage", self._model, str(type(e).__name__))
            raise

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """V119+V120+V121+V122: Batch embedding with Voyage AI."""
        results: List[EmbeddingResult] = []
        uncached_texts: List[str] = []
        uncached_indices: List[int] = []

        # V120: Check cache for each text
        for i, text in enumerate(texts):
            cached = _embedding_cache.get(text, self._model)
            if cached is not None:
                _memory_metrics.record_embed_call(
                    provider="voyage", model=self._model, cache_hit=True,
                    latency_seconds=0.0, tokens_used=0
                )
                results.append(EmbeddingResult(
                    text=text,
                    embedding=cached,
                    model=self._model,
                    dimensions=len(cached),
                    tokens_used=0,
                ))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                results.append(None)  # type: ignore

        if not uncached_texts:
            return results

        # Batch call for uncached texts
        client = self._get_client()
        start_time = time.time()

        try:
            async with self._circuit_breaker:
                # Voyage supports up to 128 texts per batch
                batch_size = 128
                all_embeddings: List[List[float]] = []
                total_tokens = 0

                for i in range(0, len(uncached_texts), batch_size):
                    batch = uncached_texts[i:i + batch_size]
                    response = await client.post(
                        f"{self._base_url}/embeddings",
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "input": batch,
                            "model": self._model,
                            "input_type": self._input_type,
                        },
                    )

                    if response.status_code != 200:
                        raise RuntimeError(f"Voyage API error: {response.text}")

                    data = response.json()
                    batch_embeddings = [d["embedding"] for d in data["data"]]
                    all_embeddings.extend(batch_embeddings)
                    total_tokens += data.get("usage", {}).get("total_tokens", 0)

            latency = time.time() - start_time

            # Fill in results and cache
            for idx, (text, embedding) in enumerate(zip(uncached_texts, all_embeddings)):
                original_idx = uncached_indices[idx]
                _embedding_cache.set(text, self._model, embedding)

                _memory_metrics.record_embed_call(
                    provider="voyage", model=self._model, cache_hit=False,
                    latency_seconds=latency / len(uncached_texts), tokens_used=total_tokens // len(uncached_texts)
                )

                results[original_idx] = EmbeddingResult(
                    text=text,
                    embedding=embedding,
                    model=self._model,
                    dimensions=len(embedding),
                    tokens_used=total_tokens // len(uncached_texts),
                )

            return results

        except CircuitOpenError:
            _memory_metrics.record_embed_error("voyage", self._model, "circuit_open")
            raise
        except Exception as e:
            _memory_metrics.record_embed_error("voyage", self._model, str(type(e).__name__))
            raise

    @classmethod
    def get_circuit_stats(cls) -> Dict[str, Any]:
        """V121: Get circuit breaker statistics."""
        stats = cls._circuit_breaker.stats
        return {
            "state": cls._circuit_breaker.state.value,
            "total_calls": stats.total_calls,
            "successful_calls": stats.successful_calls,
            "failed_calls": stats.failed_calls,
            "rejected_calls": stats.rejected_calls,
            "failure_rate": stats.failure_rate,
        }


# =============================================================================
# V123: Sentence-Transformers Local Embedding Provider
# =============================================================================

class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """
    V123: Local embedding provider using sentence-transformers.

    High-quality local embeddings without API calls. Supports models:
    - all-MiniLM-L6-v2: Fast, 384 dims (default)
    - all-mpnet-base-v2: Quality, 768 dims
    - BAAI/bge-small-en-v1.5: Optimized for retrieval, 384 dims
    - BAAI/bge-base-en-v1.5: Balanced, 768 dims

    Features:
    - Zero API cost
    - Low latency (local inference)
    - Embedding cache integration (V120)
    - Comprehensive metrics (V122)
    """

    # Lazy-loaded model cache (class-level for reuse)
    _models: Dict[str, Any] = {}

    # Model dimensions lookup
    _MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
    }

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self._model_name = model
        self._dimensions = self._MODEL_DIMENSIONS.get(model, 384)
        self._model = None  # Lazy load

    def _get_model(self):
        """Lazy load sentence-transformers model."""
        if self._model is None:
            # Check class-level cache first
            if self._model_name in SentenceTransformerEmbeddingProvider._models:
                self._model = SentenceTransformerEmbeddingProvider._models[self._model_name]
            else:
                try:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(self._model_name)
                    SentenceTransformerEmbeddingProvider._models[self._model_name] = self._model
                except ImportError:
                    raise RuntimeError(
                        "sentence-transformers not installed. "
                        "Install with: pip install sentence-transformers"
                    )
        return self._model

    async def embed(self, text: str) -> EmbeddingResult:
        """V120+V122: Generate embedding locally with caching and metrics."""
        # V120: Check cache first
        cached = _embedding_cache.get(text, self._model_name)
        if cached is not None:
            _memory_metrics.record_embed_call(
                provider="sentence-transformers", model=self._model_name, cache_hit=True,
                latency_seconds=0.0, tokens_used=0
            )
            return EmbeddingResult(
                text=text,
                embedding=cached,
                model=self._model_name,
                dimensions=len(cached),
                tokens_used=0,
            )

        # Cache miss - generate locally
        start_time = time.time()

        model = self._get_model()
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: model.encode(text, convert_to_numpy=True).tolist()
        )

        latency = time.time() - start_time

        # V120: Cache the result
        _embedding_cache.set(text, self._model_name, embedding)

        # V122: Record the call
        _memory_metrics.record_embed_call(
            provider="sentence-transformers", model=self._model_name, cache_hit=False,
            latency_seconds=latency, tokens_used=0  # Local, no token cost
        )

        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model=self._model_name,
            dimensions=len(embedding),
            tokens_used=0,
        )

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """V119+V120+V122: Batch embedding with sentence-transformers."""
        results: List[EmbeddingResult] = []
        uncached_texts: List[str] = []
        uncached_indices: List[int] = []

        # V120: Check cache for each text
        for i, text in enumerate(texts):
            cached = _embedding_cache.get(text, self._model_name)
            if cached is not None:
                _memory_metrics.record_embed_call(
                    provider="sentence-transformers", model=self._model_name, cache_hit=True,
                    latency_seconds=0.0, tokens_used=0
                )
                results.append(EmbeddingResult(
                    text=text,
                    embedding=cached,
                    model=self._model_name,
                    dimensions=len(cached),
                    tokens_used=0,
                ))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                results.append(None)  # type: ignore

        if not uncached_texts:
            return results

        # Batch encode for uncached texts
        start_time = time.time()

        model = self._get_model()
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(uncached_texts, convert_to_numpy=True).tolist()
        )

        latency = time.time() - start_time

        # Fill in results and cache
        for idx, (text, embedding) in enumerate(zip(uncached_texts, embeddings)):
            original_idx = uncached_indices[idx]
            _embedding_cache.set(text, self._model_name, embedding)

            _memory_metrics.record_embed_call(
                provider="sentence-transformers", model=self._model_name, cache_hit=False,
                latency_seconds=latency / len(uncached_texts), tokens_used=0
            )

            results[original_idx] = EmbeddingResult(
                text=text,
                embedding=embedding,
                model=self._model_name,
                dimensions=len(embedding),
                tokens_used=0,
            )

        return results


# =============================================================================
# V123: Unified Embedding Provider Factory
# =============================================================================

def create_embedding_provider(
    model: str = "all-MiniLM-L6-v2",
    api_key: Optional[str] = None,
    provider_type: Optional[str] = None,
) -> EmbeddingProvider:
    """
    V123: Factory function to create the appropriate embedding provider.

    Args:
        model: Model name (e.g., "voyage-code-3", "all-MiniLM-L6-v2", "text-embedding-3-small")
        api_key: API key for cloud providers (Voyage AI, OpenAI)
        provider_type: Override provider detection ("voyage", "openai", "local")

    Returns:
        Configured EmbeddingProvider instance

    Example:
        # Voyage AI (best for code)
        provider = create_embedding_provider("voyage-code-3", api_key=os.environ["VOYAGE_API_KEY"])

        # OpenAI
        provider = create_embedding_provider("text-embedding-3-small", api_key=os.environ["OPENAI_API_KEY"])

        # Local (no API key needed)
        provider = create_embedding_provider("all-MiniLM-L6-v2")
    """
    # Auto-detect provider from model name
    if provider_type is None:
        if model.startswith("voyage-"):
            provider_type = "voyage"
        elif model.startswith("text-embedding-"):
            provider_type = "openai"
        elif model in ("all-MiniLM-L6-v2", "all-mpnet-base-v2") or model.startswith("BAAI/"):
            provider_type = "local"
        else:
            # Default to local hash-based for unknown models
            provider_type = "local-hash"

    if provider_type == "voyage":
        if not api_key:
            import os
            api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("VOYAGE_API_KEY required for Voyage AI models")
        return VoyageEmbeddingProvider(api_key=api_key, model=model)

    elif provider_type == "openai":
        if not api_key:
            import os
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI models")
        return OpenAIEmbeddingProvider(api_key=api_key, model=model)

    elif provider_type == "local":
        return SentenceTransformerEmbeddingProvider(model=model)

    else:
        # Fallback to simple local provider
        return LocalEmbeddingProvider()


# =============================================================================
# V124: Intelligent Model Routing
# =============================================================================

class ContentType(str, Enum):
    """V124: Content types for intelligent model selection."""
    CODE = "code"           # Source code, function definitions, classes
    TEXT = "text"           # Natural language, documentation, comments
    MULTILINGUAL = "multilingual"  # Non-English or mixed language content
    MIXED = "mixed"         # Combination of code and text
    UNKNOWN = "unknown"     # Cannot determine


# Code detection patterns
_CODE_PATTERNS = [
    r'\bdef\s+\w+\s*\(',           # Python function
    r'\bclass\s+\w+',              # Class definition
    r'\bfunction\s+\w+\s*\(',      # JavaScript function
    r'\bconst\s+\w+\s*=',          # JS/TS const
    r'\blet\s+\w+\s*=',            # JS/TS let
    r'\bimport\s+[\w{},\s]+from',  # ES6 import
    r'\bfrom\s+\w+\s+import',      # Python import
    r'\breturn\s+',                # Return statement
    r';\s*$',                      # Semicolon line ending
    r'\bif\s*\(.+\)\s*{',          # If block
    r'\basync\s+def\b',            # Async function
    r'\bawait\s+\w+',              # Await expression
    r'\b(int|str|float|bool|void|string|number)\b',  # Type hints
    r'->\s*\w+:',                  # Python return type
    r'::\w+',                      # Rust/C++ namespace
]

# Non-English patterns for multilingual detection
_NON_ENGLISH_RANGES = [
    (0x0400, 0x04FF),  # Cyrillic
    (0x4E00, 0x9FFF),  # CJK
    (0x3040, 0x30FF),  # Japanese
    (0xAC00, 0xD7AF),  # Korean
    (0x0600, 0x06FF),  # Arabic
    (0x0590, 0x05FF),  # Hebrew
]


def detect_content_type(text: str) -> ContentType:
    """
    V124: Detect the type of content for optimal model selection.

    Args:
        text: Input text to analyze

    Returns:
        ContentType indicating the dominant content type
    """
    if not text or len(text.strip()) < 5:
        return ContentType.UNKNOWN

    text_lower = text.lower()

    # Check for code patterns - categorize by strength
    code_matches = 0
    strong_matches = 0

    # Strong patterns that alone indicate code
    strong_patterns = [
        r'\bdef\s+\w+\s*\(',           # Python function
        r'\bclass\s+\w+',              # Class definition
        r'\bfunction\s+\w+\s*\(',      # JavaScript function
        r'\bimport\s+[\w{},\s]+from',  # ES6 import
        r'\bfrom\s+\w+\s+import',      # Python import
        r'\basync\s+def\b',            # Async function
    ]

    for pattern in _CODE_PATTERNS:
        if re.search(pattern, text, re.MULTILINE):
            code_matches += 1
            if pattern in strong_patterns:
                strong_matches += 1

    # Code indicator: any strong pattern OR multiple weak patterns with hints
    is_code = (
        strong_matches >= 1 or  # Any strong code pattern is definitive
        code_matches >= 3 or    # Multiple weak patterns
        (code_matches >= 1 and (
            '```' in text or
            text.count('    ') >= 2 or  # 2+ indentation blocks
            text.count('\t') >= 1 or    # Any tab indentation
            (':\n' in text or '{\n' in text)  # Block structure
        ))
    )

    # Check for non-English characters
    non_english_chars = 0
    total_alpha = 0
    for char in text:
        if char.isalpha():
            total_alpha += 1
            code_point = ord(char)
            for start, end in _NON_ENGLISH_RANGES:
                if start <= code_point <= end:
                    non_english_chars += 1
                    break

    # Multilingual if significant portion of alphabetic chars are non-English
    is_multilingual = total_alpha > 0 and non_english_chars > total_alpha * 0.3  # >30% of letters

    # Classify
    if is_code and is_multilingual:
        return ContentType.MIXED
    elif is_code:
        return ContentType.CODE
    elif is_multilingual:
        return ContentType.MULTILINGUAL
    else:
        return ContentType.TEXT


# =============================================================================
# V126: Hybrid Routing with Confidence Scoring
# =============================================================================

@dataclass
class ContentConfidence:
    """
    V126: Content detection result with confidence scores.

    Provides per-type confidence scores for more nuanced routing decisions.
    """
    content_type: ContentType
    confidence: float  # Overall confidence (0.0 - 1.0)
    code_score: float  # Code pattern strength
    multilingual_score: float  # Multilingual character ratio
    text_score: float  # Natural text indicators

    def __post_init__(self):
        # Clamp values
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.code_score = max(0.0, min(1.0, self.code_score))
        self.multilingual_score = max(0.0, min(1.0, self.multilingual_score))
        self.text_score = max(0.0, min(1.0, self.text_score))


# V126: Enhanced strong code patterns with weights
_V126_STRONG_CODE_PATTERNS = [
    (r'\bdef\s+\w+\s*\(', 0.9),           # Python function
    (r'\bclass\s+\w+[\s(:]*', 0.9),       # Class definition
    (r'\bfunction\s+\w+\s*\(', 0.9),      # JavaScript function
    (r'\bconst\s+\w+\s*=\s*\(', 0.85),    # Arrow function
    (r'\bimport\s+[\w{},\s]+\s+from', 0.85),  # ES6 import
    (r'\bfrom\s+\w+\s+import', 0.85),     # Python import
    (r'\basync\s+(def|function)\b', 0.9), # Async function
    (r'\bfn\s+\w+\s*\(', 0.9),            # Rust function
    (r'\bpub\s+(fn|struct|enum)\b', 0.9), # Rust public
    (r'#\[derive\(', 0.95),               # Rust derive macro
    (r'impl\s+\w+\s+for\s+\w+', 0.95),    # Rust impl
]

# V126: Weak code patterns (need multiple matches)
_V126_WEAK_CODE_PATTERNS = [
    (r'\breturn\s+', 0.3),
    (r';\s*$', 0.2),
    (r'\{[\s\n]*\}', 0.2),
    (r'\bif\s*\(.+\)', 0.3),
    (r'\bfor\s*\(.+\)', 0.3),
    (r'\bwhile\s*\(.+\)', 0.3),
    (r'->\s*\w+', 0.4),                   # Type annotation
    (r':\s*(int|str|float|bool|None|string|number|any)\b', 0.4),
]

# V126: Text indicators (natural language patterns)
_V126_TEXT_PATTERNS = [
    (r'\b(the|a|an|is|are|was|were|be|been|being)\b', 0.3),
    (r'\b(I|you|we|they|he|she|it)\b', 0.2),
    (r'\b(this|that|these|those)\b', 0.2),
    (r'\.\s+[A-Z]', 0.4),                 # Sentence boundary
    (r'\?$', 0.3),                        # Question
    (r'!\s*$', 0.2),                      # Exclamation
    (r',\s+(and|but|or)\s+', 0.3),        # Conjunctions
]


def detect_content_type_v126(text: str) -> ContentConfidence:
    """
    V126: Hybrid content detection with multi-signal fusion.

    Improves upon V124 by:
    1. Weighted pattern matching for code detection
    2. Lower thresholds for MIXED content when both signals present
    3. Natural language indicators for TEXT detection
    4. Confidence scoring for better routing decisions

    Args:
        text: Input text to analyze

    Returns:
        ContentConfidence with type and confidence scores
    """
    if not text or len(text.strip()) < 5:
        return ContentConfidence(
            content_type=ContentType.UNKNOWN,
            confidence=1.0,
            code_score=0.0,
            multilingual_score=0.0,
            text_score=0.0,
        )

    # =========================================================================
    # Signal 1: Code pattern analysis (weighted)
    # =========================================================================
    code_score = 0.0

    # Check strong patterns (any match is significant)
    for pattern, weight in _V126_STRONG_CODE_PATTERNS:
        if re.search(pattern, text, re.MULTILINE):
            code_score = max(code_score, weight)
            break  # Strong pattern found, use its weight

    # Accumulate weak patterns
    weak_score = 0.0
    weak_count = 0
    for pattern, weight in _V126_WEAK_CODE_PATTERNS:
        if re.search(pattern, text, re.MULTILINE):
            weak_score += weight
            weak_count += 1

    # Weak patterns contribute up to 0.6 if many present
    if weak_count >= 2:
        code_score = max(code_score, min(0.6, weak_score))

    # Structural hints boost code score
    if '```' in text:
        code_score = max(code_score, 0.7)
    if text.count('    ') >= 3 or text.count('\t') >= 2:
        code_score = min(1.0, code_score + 0.2)
    if '{\n' in text or ':\n' in text:
        code_score = min(1.0, code_score + 0.1)

    # =========================================================================
    # Signal 2: Multilingual character analysis
    # =========================================================================
    non_english_chars = 0
    total_alpha = 0

    for char in text:
        if char.isalpha():
            total_alpha += 1
            code_point = ord(char)
            for start, end in _NON_ENGLISH_RANGES:
                if start <= code_point <= end:
                    non_english_chars += 1
                    break

    # Calculate multilingual score with gradual scaling
    if total_alpha > 0:
        ratio = non_english_chars / total_alpha
        # V126: More gradual scoring - even 10% non-English is notable
        if ratio >= 0.5:
            multilingual_score = 1.0
        elif ratio >= 0.3:
            multilingual_score = 0.8
        elif ratio >= 0.15:
            multilingual_score = 0.6  # V126: Lower threshold
        elif ratio >= 0.05:
            multilingual_score = 0.3  # V126: Very low still counts
        else:
            multilingual_score = 0.0
    else:
        multilingual_score = 0.0

    # =========================================================================
    # Signal 3: Natural text indicators
    # =========================================================================
    text_score = 0.0
    text_matches = 0

    for pattern, weight in _V126_TEXT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            text_score += weight
            text_matches += 1

    # Cap at 1.0 and require multiple matches
    text_score = min(1.0, text_score) if text_matches >= 2 else text_score * 0.5

    # =========================================================================
    # V126: Hybrid Classification with improved MIXED detection
    # =========================================================================

    # Decision thresholds
    CODE_THRESHOLD = 0.5
    MULTILINGUAL_THRESHOLD = 0.3  # V126: Lower for MIXED detection
    TEXT_THRESHOLD = 0.4

    # MIXED: Both code AND multilingual signals present
    # V126: Use lower thresholds when both are present
    if code_score >= CODE_THRESHOLD * 0.8 and multilingual_score >= 0.15:
        # Clear MIXED case
        content_type = ContentType.MIXED
        confidence = min(code_score, multilingual_score + 0.3)
    elif code_score >= CODE_THRESHOLD:
        content_type = ContentType.CODE
        confidence = code_score
    elif multilingual_score >= MULTILINGUAL_THRESHOLD:
        content_type = ContentType.MULTILINGUAL
        confidence = multilingual_score
    elif text_score >= TEXT_THRESHOLD or (text_matches >= 1 and code_score < 0.3):
        content_type = ContentType.TEXT
        confidence = max(text_score, 0.5)  # Default confidence for text
    else:
        # Fallback: if some code patterns but not definitive
        if code_score >= 0.2:
            content_type = ContentType.CODE
            confidence = code_score
        else:
            content_type = ContentType.TEXT
            confidence = 0.4  # Low confidence

    return ContentConfidence(
        content_type=content_type,
        confidence=confidence,
        code_score=code_score,
        multilingual_score=multilingual_score,
        text_score=text_score,
    )


def detect_content_type_hybrid(text: str) -> ContentType:
    """
    V126: Hybrid detection that returns just the ContentType.

    Drop-in replacement for detect_content_type() with improved accuracy.

    Args:
        text: Input text to analyze

    Returns:
        ContentType (same as V124 for compatibility)
    """
    result = detect_content_type_v126(text)
    return result.content_type


class EmbeddingRouter:
    """
    V124: Intelligent embedding router that selects optimal model per content.

    Maintains multiple providers and routes requests based on content analysis.
    """

    # Optimal model recommendations per content type
    MODEL_RECOMMENDATIONS = {
        ContentType.CODE: [
            ("voyage-code-3", "voyage"),           # Best for code
            ("BAAI/bge-base-en-v1.5", "local"),   # Good local alternative
            ("all-mpnet-base-v2", "local"),       # Fallback local
        ],
        ContentType.TEXT: [
            ("voyage-3.5", "voyage"),             # Best general
            ("text-embedding-3-small", "openai"), # Good OpenAI option
            ("all-mpnet-base-v2", "local"),       # Free local
        ],
        ContentType.MULTILINGUAL: [
            ("voyage-3.5", "voyage"),             # Multilingual support
            ("text-embedding-3-large", "openai"), # OpenAI large
            ("all-MiniLM-L6-v2", "local"),        # Basic local
        ],
        ContentType.MIXED: [
            ("voyage-code-3", "voyage"),          # Code priority
            ("all-mpnet-base-v2", "local"),       # Balanced local
        ],
        ContentType.UNKNOWN: [
            ("all-MiniLM-L6-v2", "local"),        # Safe default
        ],
    }

    def __init__(
        self,
        voyage_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        prefer_local: bool = False,
        track_metrics: bool = True,
    ):
        """
        Initialize the embedding router.

        Args:
            voyage_api_key: API key for Voyage AI
            openai_api_key: API key for OpenAI
            prefer_local: If True, prefer local models even when API available
            track_metrics: Whether to track routing decisions
        """
        import os
        self._voyage_key = voyage_api_key or os.environ.get("VOYAGE_API_KEY")
        self._openai_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self._prefer_local = prefer_local
        self._track_metrics = track_metrics

        # Provider cache
        self._providers: Dict[str, EmbeddingProvider] = {}

        # Routing metrics
        self._routing_stats = {
            "total_routed": 0,
            "by_content_type": {ct.value: 0 for ct in ContentType},
            "by_provider": {"voyage": 0, "openai": 0, "local": 0},
        }

    def _get_provider(self, model: str, provider_type: str) -> Optional[EmbeddingProvider]:
        """Get or create a provider."""
        cache_key = f"{provider_type}:{model}"

        if cache_key in self._providers:
            return self._providers[cache_key]

        try:
            if provider_type == "voyage" and self._voyage_key and not self._prefer_local:
                provider = create_embedding_provider(model, api_key=self._voyage_key)
            elif provider_type == "openai" and self._openai_key and not self._prefer_local:
                provider = create_embedding_provider(model, api_key=self._openai_key)
            elif provider_type == "local":
                provider = create_embedding_provider(model, provider_type="local")
            else:
                return None

            self._providers[cache_key] = provider
            return provider

        except Exception as e:
            logger.warning(f"V124: Failed to create provider {cache_key}: {e}")
            return None

    def select_provider(self, content_type: ContentType) -> Tuple[EmbeddingProvider, str]:
        """
        Select the best available provider for the content type.

        Returns:
            Tuple of (provider, model_name)
        """
        recommendations = self.MODEL_RECOMMENDATIONS.get(
            content_type,
            self.MODEL_RECOMMENDATIONS[ContentType.UNKNOWN]
        )

        for model, provider_type in recommendations:
            provider = self._get_provider(model, provider_type)
            if provider is not None:
                return provider, model

        # Ultimate fallback - simple local
        return LocalEmbeddingProvider(), "local-hash"

    async def embed(self, text: str, force_type: Optional[ContentType] = None) -> EmbeddingResult:
        """
        Embed text with automatic model selection.

        Args:
            text: Text to embed
            force_type: Override automatic content detection

        Returns:
            EmbeddingResult from optimal provider
        """
        # Detect content type
        content_type = force_type or detect_content_type(text)

        # Select provider
        provider, model = self.select_provider(content_type)

        # Track metrics
        if self._track_metrics:
            self._routing_stats["total_routed"] += 1
            self._routing_stats["by_content_type"][content_type.value] += 1
            provider_name = "local" if isinstance(provider, (LocalEmbeddingProvider, SentenceTransformerEmbeddingProvider)) else \
                           "voyage" if isinstance(provider, VoyageEmbeddingProvider) else "openai"
            self._routing_stats["by_provider"][provider_name] += 1

        # Embed
        return await provider.embed(text)

    async def embed_batch(
        self,
        texts: List[str],
        force_type: Optional[ContentType] = None,
    ) -> List[EmbeddingResult]:
        """
        Batch embed with automatic routing.

        For efficiency, groups texts by detected type and routes accordingly.
        """
        if not texts:
            return []

        if force_type:
            # All same type - use single provider
            provider, _ = self.select_provider(force_type)
            return await provider.embed_batch(texts)

        # Group by content type
        typed_texts: Dict[ContentType, List[Tuple[int, str]]] = {}
        for i, text in enumerate(texts):
            ct = detect_content_type(text)
            if ct not in typed_texts:
                typed_texts[ct] = []
            typed_texts[ct].append((i, text))

        # Embed each group
        results: List[Optional[EmbeddingResult]] = [None] * len(texts)

        for content_type, indexed_texts in typed_texts.items():
            provider, _ = self.select_provider(content_type)
            batch_texts = [t for _, t in indexed_texts]
            batch_results = await provider.embed_batch(batch_texts)

            for (orig_idx, _), result in zip(indexed_texts, batch_results):
                results[orig_idx] = result

            # Update metrics
            if self._track_metrics:
                self._routing_stats["total_routed"] += len(batch_texts)
                self._routing_stats["by_content_type"][content_type.value] += len(batch_texts)

        return [r for r in results if r is not None]

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            **self._routing_stats,
            "available_providers": {
                "voyage": self._voyage_key is not None and not self._prefer_local,
                "openai": self._openai_key is not None and not self._prefer_local,
                "local": True,
            },
            "cached_providers": list(self._providers.keys()),
        }

    def reset_stats(self) -> None:
        """Reset routing statistics."""
        self._routing_stats = {
            "total_routed": 0,
            "by_content_type": {ct.value: 0 for ct in ContentType},
            "by_provider": {"voyage": 0, "openai": 0, "local": 0},
        }


def create_embedding_router(
    prefer_local: bool = False,
    voyage_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> EmbeddingRouter:
    """
    V124: Create an intelligent embedding router.

    The router automatically selects the optimal embedding model based on
    content analysis:
    - Code → voyage-code-3 (or local BGE)
    - Text → voyage-3.5 or text-embedding-3-small
    - Multilingual → voyage-3.5

    Args:
        prefer_local: Force local models even when API keys available
        voyage_api_key: Voyage AI API key (or from env)
        openai_api_key: OpenAI API key (or from env)

    Returns:
        Configured EmbeddingRouter

    Example:
        router = create_embedding_router()

        # Automatic routing
        code_result = await router.embed("def fibonacci(n): return n if n < 2 else...")
        text_result = await router.embed("The user prefers dark mode")

        # Force specific type
        result = await router.embed(text, force_type=ContentType.CODE)

        # Check routing stats
        print(router.get_routing_stats())
    """
    return EmbeddingRouter(
        voyage_api_key=voyage_api_key,
        openai_api_key=openai_api_key,
        prefer_local=prefer_local,
    )


# =============================================================================
# Semantic Search
# =============================================================================

@dataclass
class SemanticEntry:
    """Entry with semantic embedding."""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    accessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    importance: float = 1.0


@dataclass
class SearchResult:
    """Semantic search result."""
    entry: SemanticEntry
    score: float  # Similarity score [0, 1]
    rank: int


class SemanticIndex:
    """
    In-memory semantic search index.

    For production, use Qdrant, Pinecone, or similar vector DB.
    """

    def __init__(self, embedding_provider: EmbeddingProvider):
        self._provider = embedding_provider
        self._entries: Dict[str, SemanticEntry] = {}

    async def add(
        self,
        id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 1.0,
    ) -> SemanticEntry:
        """Add entry to index with embedding."""
        result = await self._provider.embed(content)

        entry = SemanticEntry(
            id=id,
            content=content,
            embedding=result.embedding,
            metadata=metadata or {},
            importance=importance,
        )
        self._entries[id] = entry
        return entry

    async def add_batch(
        self,
        items: List[tuple[str, str, Optional[Dict[str, Any]], float]],
    ) -> List[SemanticEntry]:
        """V119: Add multiple entries with batch embedding for efficiency.

        Args:
            items: List of (id, content, metadata, importance) tuples

        Returns:
            List of created SemanticEntry objects
        """
        if not items:
            return []

        # Extract contents for batch embedding
        contents = [item[1] for item in items]

        # Use batch embedding (V119: parallel processing)
        results = await self._provider.embed_batch(contents)

        # Create entries
        entries = []
        for (id, content, metadata, importance), result in zip(items, results):
            entry = SemanticEntry(
                id=id,
                content=content,
                embedding=result.embedding,
                metadata=metadata or {},
                importance=importance,
            )
            self._entries[id] = entry
            entries.append(entry)

        return entries

    async def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """V122: Search entries by semantic similarity with metrics."""
        start_time = time.time()  # V122

        if not self._entries:
            # V122: Record empty search
            _memory_metrics.record_search("semantic_index", time.time() - start_time)
            return []

        # Embed query
        query_result = await self._provider.embed(query)
        query_embedding = query_result.embedding

        # Calculate similarities
        scored = []
        for entry in self._entries.values():
            # Apply metadata filters
            if filters:
                match = all(
                    entry.metadata.get(k) == v
                    for k, v in filters.items()
                )
                if not match:
                    continue

            # Cosine similarity
            score = self._cosine_similarity(query_embedding, entry.embedding)

            # Apply importance weight
            weighted_score = score * entry.importance

            if weighted_score >= min_score:
                scored.append((entry, weighted_score))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Build results
        results = []
        for rank, (entry, score) in enumerate(scored[:limit]):
            # Update access stats
            entry.accessed_at = datetime.now(timezone.utc)
            entry.access_count += 1

            results.append(SearchResult(
                entry=entry,
                score=score,
                rank=rank + 1,
            ))

        # V122: Record search latency
        _memory_metrics.record_search("semantic_index", time.time() - start_time)

        return results

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def get(self, id: str) -> Optional[SemanticEntry]:
        """Get entry by ID."""
        return self._entries.get(id)

    def delete(self, id: str) -> bool:
        """Delete entry by ID."""
        if id in self._entries:
            del self._entries[id]
            return True
        return False

    def count(self) -> int:
        """Get entry count."""
        return len(self._entries)

    def export(self) -> List[Dict[str, Any]]:
        """Export all entries for persistence."""
        return [
            {
                "id": e.id,
                "content": e.content,
                "embedding": e.embedding,
                "metadata": e.metadata,
                "created_at": e.created_at.isoformat(),
                "accessed_at": e.accessed_at.isoformat(),
                "access_count": e.access_count,
                "importance": e.importance,
            }
            for e in self._entries.values()
        ]

    def import_entries(self, entries: List[Dict[str, Any]]) -> int:
        """Import entries from export."""
        count = 0
        for data in entries:
            entry = SemanticEntry(
                id=data["id"],
                content=data["content"],
                embedding=data["embedding"],
                metadata=data.get("metadata", {}),
                created_at=datetime.fromisoformat(data["created_at"]),
                accessed_at=datetime.fromisoformat(data.get("accessed_at", data["created_at"])),
                access_count=data.get("access_count", 0),
                importance=data.get("importance", 1.0),
            )
            self._entries[entry.id] = entry
            count += 1
        return count


# =============================================================================
# Memory Consolidation
# =============================================================================

class ConsolidationStrategy(str, Enum):
    """Memory consolidation strategies."""
    SUMMARIZE = "summarize"  # Summarize old entries
    COMPRESS = "compress"  # Compress similar entries
    PRUNE = "prune"  # Remove low-importance entries
    HIERARCHICAL = "hierarchical"  # Build summary hierarchy


@dataclass
class ConsolidationResult:
    """Result of memory consolidation."""
    strategy: ConsolidationStrategy
    entries_processed: int
    entries_removed: int
    entries_created: int
    tokens_saved: int
    duration_ms: float


class MemoryConsolidator:
    """
    Consolidates memories to manage context window and storage.

    Strategies:
    - Summarize: Create summaries of old conversations
    - Compress: Merge similar memories
    - Prune: Remove rarely accessed, low-importance entries
    - Hierarchical: Build summary trees (day -> week -> month)
    """

    def __init__(
        self,
        semantic_index: SemanticIndex,
        summarizer: Optional[Callable[[str], str]] = None,
    ):
        self._index = semantic_index
        self._summarizer = summarizer or self._default_summarizer

    def _default_summarizer(self, text: str) -> str:
        """Default summarizer - truncates to first 500 chars."""
        if len(text) <= 500:
            return text
        return text[:497] + "..."

    async def consolidate(
        self,
        strategy: ConsolidationStrategy,
        older_than: Optional[timedelta] = None,
        min_access_count: int = 0,
        max_entries: Optional[int] = None,
    ) -> ConsolidationResult:
        """
        Consolidate memories using specified strategy.

        Args:
            strategy: Consolidation strategy to use
            older_than: Only consolidate entries older than this
            min_access_count: Only consolidate entries with fewer accesses
            max_entries: Maximum entries to consolidate
        """
        start_time = time.time()

        # Find candidates for consolidation
        candidates = self._find_candidates(older_than, min_access_count)
        if max_entries:
            candidates = candidates[:max_entries]

        if strategy == ConsolidationStrategy.SUMMARIZE:
            result = await self._summarize(candidates)
        elif strategy == ConsolidationStrategy.COMPRESS:
            result = await self._compress(candidates)
        elif strategy == ConsolidationStrategy.PRUNE:
            result = await self._prune(candidates)
        elif strategy == ConsolidationStrategy.HIERARCHICAL:
            result = await self._hierarchical(candidates)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        result.duration_ms = (time.time() - start_time) * 1000
        return result

    def _find_candidates(
        self,
        older_than: Optional[timedelta],
        min_access_count: int,
    ) -> List[SemanticEntry]:
        """Find entries that are candidates for consolidation."""
        candidates = []
        now = datetime.now(timezone.utc)

        for entry in self._index._entries.values():
            # Check age
            if older_than:
                age = now - entry.created_at
                if age < older_than:
                    continue

            # Check access count
            if entry.access_count > min_access_count:
                continue

            candidates.append(entry)

        # Sort by age (oldest first)
        candidates.sort(key=lambda e: e.created_at)
        return candidates

    async def _summarize(self, candidates: List[SemanticEntry]) -> ConsolidationResult:
        """Summarize entries into more compact form."""
        entries_removed = 0
        entries_created = 0
        tokens_saved = 0

        # Group by day
        by_day: Dict[str, List[SemanticEntry]] = {}
        for entry in candidates:
            day_key = entry.created_at.strftime("%Y-%m-%d")
            if day_key not in by_day:
                by_day[day_key] = []
            by_day[day_key].append(entry)

        # Summarize each day
        for day_key, entries in by_day.items():
            if len(entries) < 2:
                continue  # Not worth summarizing single entries

            # Combine content
            combined = "\n---\n".join(e.content for e in entries)
            original_tokens = len(combined) // 4

            # Summarize
            summary = self._summarizer(combined)
            summary_tokens = len(summary) // 4

            # Create summary entry
            await self._index.add(
                id=f"summary-{day_key}",
                content=summary,
                metadata={
                    "type": "summary",
                    "source_date": day_key,
                    "source_count": len(entries),
                },
                importance=sum(e.importance for e in entries) / len(entries),
            )
            entries_created += 1

            # Remove original entries
            for entry in entries:
                self._index.delete(entry.id)
                entries_removed += 1

            tokens_saved += original_tokens - summary_tokens

        return ConsolidationResult(
            strategy=ConsolidationStrategy.SUMMARIZE,
            entries_processed=len(candidates),
            entries_removed=entries_removed,
            entries_created=entries_created,
            tokens_saved=tokens_saved,
            duration_ms=0,
        )

    async def _compress(self, candidates: List[SemanticEntry]) -> ConsolidationResult:
        """Compress similar entries by merging."""
        entries_removed = 0
        entries_created = 0
        tokens_saved = 0

        # Find similar pairs (simple approach: same metadata type)
        by_type: Dict[str, List[SemanticEntry]] = {}
        for entry in candidates:
            entry_type = entry.metadata.get("type", "default")
            if entry_type not in by_type:
                by_type[entry_type] = []
            by_type[entry_type].append(entry)

        # Merge entries of same type
        for entry_type, entries in by_type.items():
            if len(entries) < 2:
                continue

            # Merge into single entry
            combined_content = "\n".join(e.content for e in entries)
            original_tokens = sum(len(e.content) // 4 for e in entries)

            # Create merged entry
            await self._index.add(
                id=f"merged-{entry_type}-{int(time.time())}",
                content=combined_content,
                metadata={
                    "type": entry_type,
                    "merged_count": len(entries),
                },
                importance=max(e.importance for e in entries),
            )
            entries_created += 1

            # Remove originals
            for entry in entries:
                self._index.delete(entry.id)
                entries_removed += 1

            # Calculate savings (no summarization, just deduplication overhead)
            merged_tokens = len(combined_content) // 4
            tokens_saved += original_tokens - merged_tokens

        return ConsolidationResult(
            strategy=ConsolidationStrategy.COMPRESS,
            entries_processed=len(candidates),
            entries_removed=entries_removed,
            entries_created=entries_created,
            tokens_saved=tokens_saved,
            duration_ms=0,
        )

    async def _prune(self, candidates: List[SemanticEntry]) -> ConsolidationResult:
        """Prune low-value entries."""
        entries_removed = 0
        tokens_saved = 0

        # Sort by importance (lowest first)
        sorted_candidates = sorted(candidates, key=lambda e: e.importance)

        # Remove bottom 50%
        to_remove = sorted_candidates[:len(sorted_candidates) // 2]

        for entry in to_remove:
            tokens_saved += len(entry.content) // 4
            self._index.delete(entry.id)
            entries_removed += 1

        return ConsolidationResult(
            strategy=ConsolidationStrategy.PRUNE,
            entries_processed=len(candidates),
            entries_removed=entries_removed,
            entries_created=0,
            tokens_saved=tokens_saved,
            duration_ms=0,
        )

    async def _hierarchical(self, candidates: List[SemanticEntry]) -> ConsolidationResult:
        """Build hierarchical summary structure."""
        # Group by week
        by_week: Dict[str, List[SemanticEntry]] = {}
        for entry in candidates:
            week_key = entry.created_at.strftime("%Y-W%W")
            if week_key not in by_week:
                by_week[week_key] = []
            by_week[week_key].append(entry)

        entries_removed = 0
        entries_created = 0
        tokens_saved = 0

        for week_key, entries in by_week.items():
            if len(entries) < 3:
                continue

            # Create week summary
            combined = "\n".join(e.content[:200] for e in entries)
            original_tokens = sum(len(e.content) // 4 for e in entries)

            summary = self._summarizer(combined)

            await self._index.add(
                id=f"week-summary-{week_key}",
                content=summary,
                metadata={
                    "type": "week_summary",
                    "week": week_key,
                    "entry_count": len(entries),
                },
                importance=1.5,  # Higher importance for summaries
            )
            entries_created += 1

            # Keep originals but reduce importance
            for entry in entries:
                entry.importance *= 0.5

            tokens_saved += original_tokens - len(summary) // 4

        return ConsolidationResult(
            strategy=ConsolidationStrategy.HIERARCHICAL,
            entries_processed=len(candidates),
            entries_removed=entries_removed,
            entries_created=entries_created,
            tokens_saved=tokens_saved,
            duration_ms=0,
        )


# =============================================================================
# Letta Cloud Integration
# =============================================================================

class LettaClient:
    """
    Client for Letta Cloud API.

    Provides:
    - Agent memory management
    - Passage storage and retrieval
    - Memory block synchronization
    """

    BASE_URL = "https://api.letta.com/v1"

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self._api_key = api_key
        self._base_url = base_url or self.BASE_URL
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
        return self._client

    async def create_agent(
        self,
        name: str,
        system_prompt: str = "",
        memory_blocks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Create a new Letta agent."""
        client = await self._get_client()

        payload: Dict[str, Any] = {
            "name": name,
            "system_prompt": system_prompt,
        }

        if memory_blocks:
            payload["memory_blocks"] = memory_blocks

        response = await client.post("/agents", json=payload)
        response.raise_for_status()
        return response.json()

    async def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent by ID."""
        client = await self._get_client()
        response = await client.get(f"/agents/{agent_id}")

        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    async def update_memory_block(
        self,
        agent_id: str,
        block_label: str,
        content: str,
    ) -> bool:
        """Update a memory block for an agent."""
        client = await self._get_client()

        response = await client.patch(
            f"/agents/{agent_id}/blocks/{block_label}",
            json={"content": content},
        )
        return response.status_code == 200

    async def get_memory_blocks(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get all memory blocks for an agent."""
        client = await self._get_client()
        response = await client.get(f"/agents/{agent_id}/blocks")
        response.raise_for_status()
        return response.json()

    async def add_passage(
        self,
        agent_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add a passage to agent's archival memory."""
        client = await self._get_client()

        payload: Dict[str, Any] = {"content": content}
        if metadata:
            payload["metadata"] = metadata

        response = await client.post(
            f"/agents/{agent_id}/passages",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def search_passages(
        self,
        agent_id: str,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search passages in agent's archival memory."""
        client = await self._get_client()

        response = await client.get(
            f"/agents/{agent_id}/passages",
            params={"query": query, "limit": limit},
        )
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# =============================================================================
# Advanced Memory System
# =============================================================================

class AdvancedMemorySystem:
    """
    Enhanced memory system with Letta integration and semantic search.

    Features:
    - Three-tier memory (Core, Archival, Temporal) from base system
    - Semantic search via embeddings
    - Memory consolidation strategies
    - Optional Letta Cloud sync for persistence
    """

    def __init__(
        self,
        agent_id: str,
        letta_api_key: Optional[str] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        enable_semantic_search: bool = True,
        storage_path: Optional[Path] = None,
    ):
        self.agent_id = agent_id

        # Base memory system
        self._base = MemorySystem(agent_id=agent_id)

        # Embedding provider
        self._embedding_provider = embedding_provider or LocalEmbeddingProvider()

        # Semantic index
        self._semantic_index = SemanticIndex(self._embedding_provider) if enable_semantic_search else None

        # Consolidator
        self._consolidator = MemoryConsolidator(self._semantic_index) if self._semantic_index else None

        # Letta client
        self._letta = LettaClient(letta_api_key) if letta_api_key else None
        self._letta_agent_id: Optional[str] = None

        # Storage
        self._storage_path = storage_path

    @property
    def core(self) -> CoreMemory:
        """Access core memory."""
        return self._base.core

    @property
    def archival(self) -> ArchivalMemory:
        """Access archival memory."""
        return self._base.archival

    @property
    def temporal(self) -> TemporalGraph:
        """Access temporal graph."""
        return self._base.temporal

    async def store_semantic(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 1.0,
    ) -> Optional[SemanticEntry]:
        """Store content with semantic embedding."""
        if not self._semantic_index:
            return None

        entry_id = f"{self.agent_id}-{int(time.time() * 1000)}"
        entry = await self._semantic_index.add(
            id=entry_id,
            content=content,
            metadata=metadata,
            importance=importance,
        )

        # Sync to Letta if enabled
        if self._letta and self._letta_agent_id:
            await self._letta.add_passage(
                self._letta_agent_id,
                content,
                metadata,
            )

        return entry

    async def search_semantic(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search memories semantically."""
        if not self._semantic_index:
            return []

        return await self._semantic_index.search(
            query=query,
            limit=limit,
            min_score=min_score,
            filters=filters,
        )

    async def consolidate(
        self,
        strategy: ConsolidationStrategy = ConsolidationStrategy.SUMMARIZE,
        older_than_days: int = 7,
    ) -> Optional[ConsolidationResult]:
        """Consolidate old memories."""
        if not self._consolidator:
            return None

        return await self._consolidator.consolidate(
            strategy=strategy,
            older_than=timedelta(days=older_than_days),
        )

    async def sync_to_letta(self) -> bool:
        """Sync memory state to Letta Cloud."""
        if not self._letta:
            return False

        # Create agent if needed
        if not self._letta_agent_id:
            result = await self._letta.create_agent(
                name=f"uap-{self.agent_id}",
                memory_blocks=[
                    {"label": label, "content": block.content}
                    for label, block in self.core.blocks.items()
                ],
            )
            self._letta_agent_id = result.get("id")

        # Sync memory blocks
        agent_id = self._letta_agent_id
        if not agent_id:
            return False

        for label, block in self.core.blocks.items():
            await self._letta.update_memory_block(
                agent_id,
                label,
                block.content,
            )

        return True

    async def sync_from_letta(self) -> bool:
        """Sync memory state from Letta Cloud."""
        if not self._letta or not self._letta_agent_id:
            return False

        blocks = await self._letta.get_memory_blocks(self._letta_agent_id)
        for block in blocks:
            self.core.update(block["label"], block["content"])

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        stats = {
            "agent_id": self.agent_id,
            "core_blocks": len(self.core.blocks),
            "core_tokens": self.core.total_tokens(),
            "letta_enabled": self._letta is not None,
            "letta_agent_id": self._letta_agent_id,
        }

        if self._semantic_index:
            stats["semantic_entries"] = self._semantic_index.count()

        return stats

    async def save(self, path: Optional[Path] = None) -> bool:
        """Save memory state to disk."""
        save_path = path or self._storage_path
        if not save_path:
            return False

        save_path.mkdir(parents=True, exist_ok=True)

        # Save core memory
        core_file = save_path / "core_memory.json"
        with open(core_file, "w") as f:
            json.dump(self.core.export(), f, indent=2)

        # Save semantic index
        if self._semantic_index:
            semantic_file = save_path / "semantic_index.json"
            with open(semantic_file, "w") as f:
                json.dump(self._semantic_index.export(), f, indent=2)

        return True

    async def load(self, path: Optional[Path] = None) -> bool:
        """Load memory state from disk."""
        load_path = path or self._storage_path
        if not load_path or not load_path.exists():
            return False

        # Load core memory
        core_file = load_path / "core_memory.json"
        if core_file.exists():
            with open(core_file) as f:
                self.core.import_state(json.load(f))

        # Load semantic index
        if self._semantic_index:
            semantic_file = load_path / "semantic_index.json"
            if semantic_file.exists():
                with open(semantic_file) as f:
                    self._semantic_index.import_entries(json.load(f))

        return True

    async def close(self) -> None:
        """Clean up resources."""
        if self._letta:
            await self._letta.close()


# =============================================================================
# Factory Functions
# =============================================================================

def create_advanced_memory(
    agent_id: str,
    letta_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    storage_path: Optional[Path] = None,
) -> AdvancedMemorySystem:
    """
    Create an advanced memory system.

    Args:
        agent_id: Unique agent identifier
        letta_api_key: Optional Letta Cloud API key
        openai_api_key: Optional OpenAI API key for embeddings
        storage_path: Optional path for local storage
    """
    embedding_provider = None
    if openai_api_key:
        embedding_provider = OpenAIEmbeddingProvider(openai_api_key)
    else:
        embedding_provider = LocalEmbeddingProvider()

    return AdvancedMemorySystem(
        agent_id=agent_id,
        letta_api_key=letta_api_key,
        embedding_provider=embedding_provider,
        enable_semantic_search=True,
        storage_path=storage_path,
    )


def create_consolidator(
    semantic_index: SemanticIndex,
    summarizer: Optional[Callable[[str], str]] = None,
) -> MemoryConsolidator:
    """Create a memory consolidator."""
    return MemoryConsolidator(semantic_index, summarizer)


def get_memory_stats() -> Dict[str, Any]:
    """
    V122: Get comprehensive memory system statistics.

    Returns a dictionary containing metrics from all memory subsystems:
    - Embedding operations (calls, latency, tokens, errors)
    - Cache performance (hits, misses, evictions, hit rate)
    - Circuit breaker state and transitions
    - Semantic search operations
    - Memory consolidation metrics

    Example:
        >>> stats = get_memory_stats()
        >>> print(f"Cache hit rate: {stats['cache']['hit_rate']:.1%}")
        >>> print(f"Embedding latency p50: {stats['embedding']['latency_p50_ms']:.1f}ms")
        >>> print(f"Circuit state: {stats['circuit_breaker']['state']}")
    """
    return _memory_metrics.get_all_stats()


def get_embedding_cache_stats() -> Dict[str, Any]:
    """
    V120+V122: Get embedding cache statistics.

    Returns:
        Dictionary with cache stats including:
        - size: Current number of cached embeddings
        - max_size: Maximum cache capacity
        - hits: Number of cache hits
        - misses: Number of cache misses
        - hit_rate: Ratio of hits to total requests
        - ttl_evictions: Entries expired by TTL
        - lru_evictions: Entries evicted by LRU policy
    """
    cache_stats = _embedding_cache.stats
    metrics_stats = _memory_metrics.get_all_stats()

    return {
        **cache_stats,
        "ttl_evictions": metrics_stats.get("cache", {}).get("ttl_evictions", 0),
        "lru_evictions": metrics_stats.get("cache", {}).get("lru_evictions", 0),
    }


def reset_memory_metrics() -> None:
    """
    V122: Reset all memory metrics counters.

    Useful for benchmarking or starting fresh measurement periods.
    Note: This resets metrics only, not the actual cache contents.
    """
    _memory_metrics.reset()


# =============================================================================
# Demo
# =============================================================================

async def demo():
    """Demo advanced memory system."""
    print("=" * 60)
    print("ADVANCED MEMORY SYSTEM DEMO")
    print("=" * 60)
    print()

    # Create memory system
    memory = create_advanced_memory(
        agent_id="demo-agent",
        storage_path=Path("./demo_memory"),
    )

    # Store some memories
    print("[>>] Storing semantic memories...")
    await memory.store_semantic(
        "User prefers dark mode and concise responses",
        metadata={"type": "preference"},
        importance=1.5,
    )
    await memory.store_semantic(
        "User is working on a Python project with FastAPI",
        metadata={"type": "context"},
        importance=1.2,
    )
    await memory.store_semantic(
        "User asked about database optimization yesterday",
        metadata={"type": "history"},
        importance=0.8,
    )

    # Search semantically
    print("\n[>>] Searching for 'user preferences'...")
    results = await memory.search_semantic("user preferences", limit=2)
    for r in results:
        print(f"  [{r.rank}] Score: {r.score:.3f} - {r.entry.content[:50]}...")

    # Update core memory
    print("\n[>>] Updating core memory...")
    memory.core.update("task_state", "Currently testing advanced memory features")

    # Get stats
    print("\n[>>] Memory Statistics:")
    stats = memory.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Save to disk
    print("\n[>>] Saving memory state...")
    await memory.save()
    print("  Saved to ./demo_memory/")

    await memory.close()
    print("\n[OK] Advanced memory demo complete")


def main():
    """Run demo."""
    asyncio.run(demo())


if __name__ == "__main__":
    main()
