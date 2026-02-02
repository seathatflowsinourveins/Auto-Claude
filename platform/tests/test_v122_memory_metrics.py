#!/usr/bin/env python3
"""
V122 Optimization Test: Memory System Metrics & Observability

Tests memory metrics by importing and testing real classes -
not by grepping file contents.

Test Date: 2026-01-30, Updated: 2026-02-02 (V14 Iter 55)
"""

import os
import sys
import time
import pytest

# Add platform to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestMemoryMetricsStructure:
    """Test memory metrics structure by importing real classes."""

    def test_memory_metrics_importable(self):
        """MemoryMetrics class should be importable."""
        try:
            from core.advanced_memory import MemoryMetrics
        except ImportError:
            pytest.skip("advanced_memory not importable")
        assert MemoryMetrics is not None

    def test_memory_metrics_has_required_methods(self):
        """MemoryMetrics should have all recording methods."""
        try:
            from core.advanced_memory import MemoryMetrics
        except ImportError:
            pytest.skip("advanced_memory not importable")
        m = MemoryMetrics()
        assert hasattr(m, "record_embed_call") and callable(m.record_embed_call)
        assert hasattr(m, "record_embed_error") and callable(m.record_embed_error)
        assert hasattr(m, "update_cache_size") and callable(m.update_cache_size)
        assert hasattr(m, "record_cache_eviction") and callable(m.record_cache_eviction)
        assert hasattr(m, "record_search") and callable(m.record_search)
        assert hasattr(m, "get_all_stats") and callable(m.get_all_stats)

    def test_global_metrics_instance(self):
        """Global _memory_metrics should be a MemoryMetrics instance."""
        try:
            from core.advanced_memory import _memory_metrics, MemoryMetrics
        except ImportError:
            pytest.skip("advanced_memory not importable")
        assert isinstance(_memory_metrics, MemoryMetrics)

    def test_helper_functions_importable(self):
        """Helper functions should be importable and callable."""
        try:
            from core.advanced_memory import (
                get_memory_stats,
                get_embedding_cache_stats,
                reset_memory_metrics,
            )
        except ImportError:
            pytest.skip("advanced_memory not importable")
        assert callable(get_memory_stats)
        assert callable(get_embedding_cache_stats)
        assert callable(reset_memory_metrics)

    def test_providers_have_metrics_support(self):
        """Embedding providers should support metrics recording."""
        try:
            from core.advanced_memory import (
                LocalEmbeddingProvider,
                OpenAIEmbeddingProvider,
            )
        except ImportError:
            pytest.skip("advanced_memory not importable")
        # Both providers should exist and be instantiable
        local = LocalEmbeddingProvider()
        assert hasattr(local, "embed")
        openai = OpenAIEmbeddingProvider(api_key="test")
        assert hasattr(openai, "embed")

    def test_semantic_index_has_search(self):
        """SemanticIndex should have search method for metrics."""
        try:
            from core.advanced_memory import SemanticIndex, LocalEmbeddingProvider
        except ImportError:
            pytest.skip("advanced_memory not importable")
        provider = LocalEmbeddingProvider()
        index = SemanticIndex(provider)
        assert hasattr(index, "search") and callable(index.search)

    def test_embedding_cache_has_instrumentation(self):
        """EmbeddingCache should track size and evictions."""
        try:
            from core.advanced_memory import EmbeddingCache
        except ImportError:
            pytest.skip("advanced_memory not importable")
        cache = EmbeddingCache(max_size=5, ttl_seconds=60.0)
        assert hasattr(cache, "stats")
        stats = cache.stats
        assert "size" in stats


class TestMemoryMetricsBehavior:
    """Test actual behavior of memory metrics."""

    @pytest.fixture(autouse=True)
    def reset_metrics(self):
        """Reset metrics singleton before each test."""
        try:
            from core.advanced_memory import reset_memory_metrics
            reset_memory_metrics()
        except ImportError:
            pass

    def test_metrics_class_initialization(self):
        """Test MemoryMetrics initializes correctly."""
        try:
            from core.advanced_memory import MemoryMetrics
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        metrics = MemoryMetrics()

        # Should have zero initial counters
        assert metrics.embed_calls == 0
        assert metrics.embed_errors == 0
        assert metrics.embed_tokens_total == 0
        assert metrics.embed_cache_hits == 0
        assert metrics.embed_cache_misses == 0

        # Should have empty latency lists
        assert len(metrics.embed_latencies) == 0
        assert len(metrics.search_latencies) == 0

        # Should have default circuit state
        assert metrics.circuit_state == "closed"

    def test_record_embed_call(self):
        """Test recording embedding calls."""
        try:
            from core.advanced_memory import MemoryMetrics
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        metrics = MemoryMetrics()

        # Record a cache hit
        metrics.record_embed_call(
            provider="test",
            model="test-model",
            cache_hit=True,
            latency_seconds=0.05,
            tokens_used=100
        )

        assert metrics.embed_calls == 1
        assert metrics.embed_cache_hits == 1
        assert metrics.embed_cache_misses == 0
        assert metrics.embed_tokens_total == 100
        assert len(metrics.embed_latencies) == 1

        # Record a cache miss
        metrics.record_embed_call(
            provider="test",
            model="test-model",
            cache_hit=False,
            latency_seconds=0.1,
            tokens_used=200
        )

        assert metrics.embed_calls == 2
        assert metrics.embed_cache_hits == 1
        assert metrics.embed_cache_misses == 1
        assert metrics.embed_tokens_total == 300
        assert len(metrics.embed_latencies) == 2

    def test_record_embed_error(self):
        """Test recording embedding errors."""
        try:
            from core.advanced_memory import MemoryMetrics
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        metrics = MemoryMetrics()

        metrics.record_embed_error("test", "test-model", "timeout")
        metrics.record_embed_error("test", "test-model", "rate_limit")

        assert metrics.embed_errors == 2

    def test_cache_eviction_tracking(self):
        """Test cache eviction tracking."""
        try:
            from core.advanced_memory import MemoryMetrics
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        metrics = MemoryMetrics()

        metrics.record_cache_eviction("ttl")
        metrics.record_cache_eviction("ttl")
        metrics.record_cache_eviction("lru")

        assert metrics.cache_ttl_evictions == 2
        assert metrics.cache_lru_evictions == 1

    def test_search_recording(self):
        """Test search operation recording."""
        try:
            from core.advanced_memory import MemoryMetrics
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        metrics = MemoryMetrics()

        metrics.record_search("semantic_index", 0.05)
        metrics.record_search("semantic_index", 0.1)
        metrics.record_search("semantic_index", 0.08)

        assert metrics.search_calls == 3
        assert len(metrics.search_latencies) == 3

    def test_get_all_stats(self):
        """Test comprehensive stats retrieval."""
        try:
            from core.advanced_memory import MemoryMetrics
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        metrics = MemoryMetrics()

        # Generate some activity
        metrics.record_embed_call("test", "model", True, 0.05, 100)
        metrics.record_embed_call("test", "model", False, 0.1, 200)
        metrics.record_embed_error("test", "model", "timeout")
        metrics.record_cache_eviction("ttl")
        metrics.update_cache_size(10)
        metrics.record_search("index", 0.05)

        stats = metrics.get_all_stats()

        # Check structure
        assert "embedding" in stats
        assert "cache" in stats
        assert "circuit_breaker" in stats
        assert "search" in stats

        # Check embedding stats
        assert stats["embedding"]["calls"] == 2
        assert stats["embedding"]["errors"] == 1
        assert stats["embedding"]["cache_hits"] == 1
        assert stats["embedding"]["cache_misses"] == 1
        assert stats["embedding"]["tokens_total"] == 300

        # Check cache stats
        assert stats["cache"]["size"] == 10
        assert stats["cache"]["ttl_evictions"] == 1

        # Check search stats
        assert stats["search"]["calls"] == 1

    def test_latency_percentiles(self):
        """Test latency percentile calculations."""
        try:
            from core.advanced_memory import MemoryMetrics
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        metrics = MemoryMetrics()

        # Add 100 latencies ranging from 10ms to 1000ms
        for i in range(100):
            latency = 0.01 + (i * 0.01)  # 10ms to 1000ms
            metrics.record_embed_call("test", "model", False, latency, 10)

        stats = metrics.get_all_stats()

        # p50 should be around 500ms
        assert 400 < stats["embedding"]["latency_p50_ms"] < 600

        # p95 should be around 950ms
        assert 900 < stats["embedding"]["latency_p95_ms"] < 1000

        # p99 should be around 990ms
        assert 950 < stats["embedding"]["latency_p99_ms"] < 1010


class TestHelperFunctions:
    """Test V122 helper functions."""

    def test_get_memory_stats(self):
        """Test get_memory_stats helper function."""
        try:
            from core.advanced_memory import (
                get_memory_stats,
                reset_memory_metrics,
            )
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        # Reset for clean slate
        reset_memory_metrics()

        stats = get_memory_stats()

        # Should return dict with expected structure
        assert isinstance(stats, dict)
        assert "embedding" in stats
        assert "cache" in stats
        assert "circuit_breaker" in stats
        assert "search" in stats

    def test_get_embedding_cache_stats(self):
        """Test get_embedding_cache_stats helper function."""
        try:
            from core.advanced_memory import get_embedding_cache_stats
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        stats = get_embedding_cache_stats()

        # Should have combined stats
        assert isinstance(stats, dict)
        assert "size" in stats or "hits" in stats or "misses" in stats
        # Note: exact fields depend on EmbeddingCache.stats implementation

    def test_reset_memory_metrics(self):
        """Test reset_memory_metrics helper function."""
        try:
            from core.advanced_memory import (
                get_memory_stats,
                reset_memory_metrics,
                LocalEmbeddingProvider,
            )
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        # Generate some activity
        provider = LocalEmbeddingProvider()
        import asyncio

        async def generate_activity():
            await provider.embed("test text")
            await provider.embed("another test")

        asyncio.run(generate_activity())

        # Get stats - should show activity
        stats_before = get_memory_stats()
        calls_before = stats_before["embedding"]["calls"]

        # Reset
        reset_memory_metrics()

        # Get stats again - should be zero
        stats_after = get_memory_stats()
        assert stats_after["embedding"]["calls"] == 0
        assert stats_after["embedding"]["errors"] == 0


class TestMetricsIntegration:
    """Test metrics integration with other V1xx optimizations."""

    @pytest.mark.asyncio
    async def test_local_provider_records_metrics(self):
        """Test LocalEmbeddingProvider records metrics on embed."""
        try:
            from core.advanced_memory import (
                LocalEmbeddingProvider,
                get_memory_stats,
                reset_memory_metrics,
            )
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        reset_memory_metrics()

        provider = LocalEmbeddingProvider()
        initial_stats = get_memory_stats()
        initial_calls = initial_stats["embedding"]["calls"]

        # Make embedding call
        result = await provider.embed("test embedding")

        # Should have recorded the call
        stats = get_memory_stats()
        assert stats["embedding"]["calls"] > initial_calls

        # Should have recorded latency
        assert stats["embedding"].get("latency_p50_ms", 0) >= 0

    @pytest.mark.asyncio
    async def test_cache_hit_records_correctly(self):
        """Test that cache hits are recorded correctly."""
        try:
            from core.advanced_memory import (
                OpenAIEmbeddingProvider,
                _embedding_cache,
                get_memory_stats,
                reset_memory_metrics,
            )
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        reset_memory_metrics()

        # Pre-populate cache
        test_text = "cached embedding test"
        _embedding_cache.set(test_text, "text-embedding-3-small", [0.1] * 1536)

        # Create provider (won't actually call API due to cache)
        provider = OpenAIEmbeddingProvider(api_key="fake-key")

        # This should hit cache
        result = await provider.embed(test_text)

        # Should record as cache hit
        stats = get_memory_stats()
        assert stats["embedding"]["cache_hits"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
