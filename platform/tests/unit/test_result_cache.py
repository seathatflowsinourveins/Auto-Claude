"""
Unit tests for RAG Result Cache

Tests the two-level caching system:
- L1: Exact query match (hash-based)
- L2: Semantic similarity (embedding-based)
"""

import pytest
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Import cache components
from core.rag.result_cache import (
    ResultCache,
    ResultCacheConfig,
    CacheEntry,
    CacheHit,
    CacheStats,
    CacheLevel,
    InvalidationStrategy,
    QueryTypeTTL,
    ExactMatchCache,
    SemanticCache,
    MemoryBudgetManager,
    create_result_cache,
)


# =============================================================================
# FIXTURES
# =============================================================================

@dataclass
class MockPipelineResult:
    """Mock pipeline result for testing."""
    response: str
    confidence: float
    contexts_used: List[str]


class MockEmbeddingProvider:
    """Mock embedding provider that returns deterministic embeddings."""

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self._cache: Dict[str, List[float]] = {}

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings based on text content."""
        results = []
        for text in texts:
            if text not in self._cache:
                # Generate a deterministic embedding based on text hash
                import hashlib
                text_hash = hashlib.md5(text.encode()).digest()
                embedding = [
                    (b / 255.0) - 0.5
                    for b in (text_hash * (self.embedding_dim // 16 + 1))[:self.embedding_dim]
                ]
                self._cache[text] = embedding
            results.append(self._cache[text])
        return results


@pytest.fixture
def config():
    """Default test configuration."""
    return ResultCacheConfig(
        max_entries=100,
        memory_budget_mb=10.0,
        default_ttl_seconds=60,
        semantic_threshold=0.95,
        enable_l2_cache=True,
        l2_max_entries=50,
    )


@pytest.fixture
def embedding_provider():
    """Mock embedding provider."""
    return MockEmbeddingProvider()


@pytest.fixture
def result_cache(config, embedding_provider):
    """Result cache with mock embedding provider."""
    return ResultCache(config=config, embedding_provider=embedding_provider)


@pytest.fixture
def l1_cache():
    """Standalone L1 exact match cache."""
    return ExactMatchCache(max_entries=100)


# =============================================================================
# L1 EXACT MATCH CACHE TESTS
# =============================================================================

class TestExactMatchCache:
    """Tests for L1 exact match cache."""

    def test_put_and_get(self, l1_cache):
        """Test basic put and get operations."""
        result = MockPipelineResult("Test response", 0.9, ["context1"])
        l1_cache.put("What is RAG?", result, ttl_seconds=3600)

        entry = l1_cache.get("What is RAG?")
        assert entry is not None
        assert entry.value.response == "Test response"
        assert entry.value.confidence == 0.9

    def test_exact_match_only(self, l1_cache):
        """Test that L1 cache only matches exact queries."""
        result = MockPipelineResult("Response", 0.8, [])
        l1_cache.put("What is RAG?", result)

        # Exact match should work
        assert l1_cache.get("What is RAG?") is not None

        # Similar but different queries should not match
        assert l1_cache.get("What is RAG") is None  # Missing ?
        assert l1_cache.get("what is rag?") is not None  # Case insensitive

    def test_ttl_expiration(self, l1_cache):
        """Test that entries expire after TTL."""
        result = MockPipelineResult("Response", 0.8, [])
        l1_cache.put("Query", result, ttl_seconds=1)

        # Should be available immediately
        assert l1_cache.get("Query") is not None

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert l1_cache.get("Query") is None

    def test_lru_eviction(self):
        """Test LRU eviction when at capacity."""
        cache = ExactMatchCache(max_entries=3)

        # Fill cache
        cache.put("Query1", "Result1")
        cache.put("Query2", "Result2")
        cache.put("Query3", "Result3")

        # Access Query1 to make it recently used
        cache.get("Query1")

        # Add new entry, should evict Query2 (LRU)
        cache.put("Query4", "Result4")

        assert cache.get("Query1") is not None
        assert cache.get("Query2") is None  # Evicted
        assert cache.get("Query3") is not None
        assert cache.get("Query4") is not None

    def test_delete(self, l1_cache):
        """Test entry deletion."""
        l1_cache.put("Query", "Result")
        assert l1_cache.get("Query") is not None

        assert l1_cache.delete("Query") is True
        assert l1_cache.get("Query") is None
        assert l1_cache.delete("Query") is False  # Already deleted

    def test_invalidate_by_pattern(self, l1_cache):
        """Test pattern-based invalidation."""
        l1_cache.put("What is RAG?", "Result1")
        l1_cache.put("What is LLM?", "Result2")
        l1_cache.put("How does search work?", "Result3")

        # Invalidate all "What is" queries
        count = l1_cache.invalidate_by_pattern("What is *")
        assert count == 2

        assert l1_cache.get("What is RAG?") is None
        assert l1_cache.get("What is LLM?") is None
        assert l1_cache.get("How does search work?") is not None

    def test_invalidate_by_age(self, l1_cache):
        """Test age-based invalidation."""
        l1_cache.put("Old query", "Result1")
        time.sleep(0.5)
        l1_cache.put("New query", "Result2")

        # Invalidate entries older than 0.3 seconds
        count = l1_cache.invalidate_by_age(0.3)
        assert count == 1

        assert l1_cache.get("Old query") is None
        assert l1_cache.get("New query") is not None

    def test_stats(self, l1_cache):
        """Test cache statistics."""
        l1_cache.put("Query1", "Result1")
        l1_cache.put("Query2", "Result2")

        l1_cache.get("Query1")  # Hit
        l1_cache.get("Query1")  # Hit
        l1_cache.get("NonExistent")  # Miss

        stats = l1_cache.stats
        assert stats["entries"] == 2
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2/3)


# =============================================================================
# L2 SEMANTIC CACHE TESTS
# =============================================================================

class TestSemanticCache:
    """Tests for L2 semantic similarity cache."""

    def test_semantic_match(self, embedding_provider):
        """Test semantic similarity matching."""
        cache = SemanticCache(
            embedding_provider=embedding_provider,
            max_entries=100,
            similarity_threshold=0.5,  # Low threshold for testing
        )

        # Store entry
        entry = CacheEntry(
            key="test",
            query="What is retrieval augmented generation?",
            value="RAG is a technique...",
            query_hash="test",
            ttl_seconds=3600,
        )
        cache.put(entry)

        # Same query should match
        result = cache.get("What is retrieval augmented generation?")
        assert result is not None
        entry, similarity = result
        assert similarity >= 0.5

    def test_no_embedding_provider(self):
        """Test cache behavior without embedding provider."""
        cache = SemanticCache(
            embedding_provider=None,
            max_entries=100,
        )

        # Without provider, get should return None
        result = cache.get("Any query")
        assert result is None


# =============================================================================
# RESULT CACHE (COMBINED L1/L2) TESTS
# =============================================================================

class TestResultCache:
    """Tests for combined L1/L2 result cache."""

    def test_l1_hit(self, result_cache):
        """Test L1 cache hit for exact match."""
        result = MockPipelineResult("Response", 0.9, ["ctx"])
        result_cache.put("What is RAG?", result, query_type="factual")

        hit = result_cache.get("What is RAG?")
        assert hit is not None
        assert hit.cache_level == CacheLevel.L1_EXACT
        assert hit.similarity_score == 1.0
        assert hit.value.response == "Response"

    def test_cache_miss(self, result_cache):
        """Test cache miss."""
        hit = result_cache.get("Nonexistent query")
        assert hit is None

    def test_ttl_per_query_type(self, result_cache):
        """Test different TTL for different query types."""
        result = MockPipelineResult("Response", 0.9, [])

        # Store with different query types
        result_cache.put("News query", result, query_type="news")
        result_cache.put("Factual query", result, query_type="factual")

        # Check entries have different TTLs
        stats = result_cache.stats
        assert stats.l1_entries == 2

    def test_memory_budget_enforcement(self):
        """Test memory budget enforcement."""
        config = ResultCacheConfig(
            max_entries=1000,
            memory_budget_mb=0.001,  # Very small budget
        )
        cache = ResultCache(config=config)

        # Store many entries
        for i in range(100):
            cache.put(f"Query {i}", f"Result {i}" * 100)

        # Memory should be under budget
        stats = cache.stats
        assert stats.memory_used_bytes <= config.memory_budget_mb * 1024 * 1024 * 1.1

    def test_invalidation_strategies(self, result_cache):
        """Test various invalidation strategies."""
        # Store entries
        result_cache.put("RAG query 1", "Result1", query_type="factual")
        result_cache.put("RAG query 2", "Result2", query_type="research")
        result_cache.put("LLM query 1", "Result3", query_type="factual")

        # Invalidate by pattern
        count = result_cache.invalidate_by_pattern("RAG *")
        assert count == 2

        # Remaining entry should still exist
        assert result_cache.get("LLM query 1") is not None

    def test_clear(self, result_cache):
        """Test cache clearing."""
        result_cache.put("Query1", "Result1")
        result_cache.put("Query2", "Result2")

        result = result_cache.clear()
        assert result["l1"] >= 2

        assert result_cache.get("Query1") is None
        assert result_cache.get("Query2") is None

    def test_stats(self, result_cache):
        """Test comprehensive statistics."""
        result_cache.put("Query1", "Result1")
        result_cache.put("Query2", "Result2")

        result_cache.get("Query1")  # Hit
        result_cache.get("NonExistent")  # Miss

        stats = result_cache.stats
        assert stats.l1_entries >= 2
        assert stats.l1_hits == 1
        assert stats.misses >= 1
        assert stats.hit_rate > 0

    def test_hot_queries(self, result_cache):
        """Test hot queries tracking."""
        result_cache.put("Popular", "Result1")
        result_cache.put("Unpopular", "Result2")

        # Access popular query multiple times
        for _ in range(5):
            result_cache.get("Popular")

        hot = result_cache.get_hot_queries(top_k=2)
        assert len(hot) == 2
        assert hot[0][0] == "Popular"
        assert hot[0][1] >= 5


# =============================================================================
# MEMORY BUDGET MANAGER TESTS
# =============================================================================

class TestMemoryBudgetManager:
    """Tests for memory budget management."""

    def test_allocation(self):
        """Test memory allocation."""
        manager = MemoryBudgetManager(budget_mb=1.0)

        # Should allocate small amounts
        assert manager.allocate(1000) is True
        assert manager.used_bytes == 1000

        # Should refuse over-budget allocation
        assert manager.allocate(manager.budget_bytes) is False

    def test_release(self):
        """Test memory release."""
        manager = MemoryBudgetManager(budget_mb=1.0)
        manager.allocate(1000)
        manager.release(500)
        assert manager.used_bytes == 500

    def test_utilization(self):
        """Test utilization calculation."""
        manager = MemoryBudgetManager(budget_mb=1.0)
        manager.allocate(manager.budget_bytes // 2)
        assert manager.utilization == pytest.approx(0.5)

    def test_needs_eviction(self):
        """Test eviction threshold."""
        manager = MemoryBudgetManager(budget_mb=1.0)
        manager.set_used(int(manager.budget_bytes * 0.95))
        assert manager.needs_eviction(threshold=0.9) is True
        assert manager.needs_eviction(threshold=0.99) is False


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestCreateResultCache:
    """Tests for factory function."""

    def test_create_with_defaults(self):
        """Test creating cache with default settings."""
        cache = create_result_cache()
        assert cache is not None
        assert cache.config.max_entries == 5000

    def test_create_with_custom_config(self):
        """Test creating cache with custom settings."""
        cache = create_result_cache(
            max_entries=1000,
            memory_budget_mb=128.0,
            semantic_threshold=0.90,
        )
        assert cache.config.max_entries == 1000
        assert cache.config.memory_budget_mb == 128.0
        assert cache.config.semantic_threshold == 0.90


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestCacheIntegration:
    """Integration tests for cache with pipeline."""

    def test_cache_pipeline_result(self, result_cache):
        """Test caching actual pipeline-like results."""
        result = MockPipelineResult(
            response="RAG is a technique that combines retrieval with generation.",
            confidence=0.85,
            contexts_used=["Context 1", "Context 2"],
        )

        # Cache the result
        result_cache.put(
            query="What is RAG?",
            value=result,
            query_type="factual",
            metadata={"strategy": "basic"},
        )

        # Retrieve and verify
        hit = result_cache.get("What is RAG?")
        assert hit is not None
        assert hit.value.response == result.response
        assert hit.value.confidence == result.confidence
        assert hit.cache_level == CacheLevel.L1_EXACT

    def test_query_type_ttl_mapping(self, result_cache):
        """Test that query types map to correct TTLs."""
        config = result_cache.config

        # Verify TTL mappings
        assert config.ttl_per_query_type.get("news", 0) < config.ttl_per_query_type.get("factual", 0)
        assert config.ttl_per_query_type.get("code", 0) < config.ttl_per_query_type.get("factual", 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
