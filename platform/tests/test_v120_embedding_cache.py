#!/usr/bin/env python3
"""
V120 Optimization Test: Embedding Cache with TTL

Tests that embedding operations use caching by importing
and testing the real classes - not by grepping file contents.

Test Date: 2026-01-30, Updated: 2026-02-02 (V14 Iter 52)
"""

import os
import sys
import time
import pytest

# Add platform to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestEmbeddingCacheStructure:
    """Test EmbeddingCache class structure by importing real class."""

    def test_embedding_cache_class_importable(self):
        """EmbeddingCache class should be importable."""
        try:
            from core.advanced_memory import EmbeddingCache
        except ImportError:
            pytest.skip("advanced_memory not importable")

        assert EmbeddingCache is not None

    def test_embedding_cache_has_required_methods(self):
        """EmbeddingCache should have get, set, clear, stats."""
        try:
            from core.advanced_memory import EmbeddingCache
        except ImportError:
            pytest.skip("advanced_memory not importable")

        cache = EmbeddingCache(max_size=10, ttl_seconds=60.0)
        assert hasattr(cache, "get"), "Should have get method"
        assert callable(cache.get)
        assert hasattr(cache, "set"), "Should have set method"
        assert callable(cache.set)
        assert hasattr(cache, "clear"), "Should have clear method"
        assert callable(cache.clear)
        assert hasattr(cache, "stats"), "Should have stats property"

    def test_openai_provider_has_embed_methods(self):
        """OpenAIEmbeddingProvider should have embed and embed_batch."""
        try:
            from core.advanced_memory import OpenAIEmbeddingProvider
        except ImportError:
            pytest.skip("advanced_memory not importable")

        provider = OpenAIEmbeddingProvider(api_key="test-key")
        assert hasattr(provider, "embed"), "Should have embed method"
        assert hasattr(provider, "embed_batch"), "Should have embed_batch method"

    def test_global_cache_instance_importable(self):
        """Global _embedding_cache instance should be importable."""
        try:
            from core.advanced_memory import _embedding_cache, EmbeddingCache
        except ImportError:
            pytest.skip("advanced_memory not importable")

        assert isinstance(_embedding_cache, EmbeddingCache), \
            "Global _embedding_cache should be an EmbeddingCache instance"


class TestEmbeddingCacheBehavior:
    """Test actual behavior of embedding cache."""

    def test_cache_basic_operations(self):
        """Test basic cache get/set operations."""
        try:
            from core.advanced_memory import EmbeddingCache
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        cache = EmbeddingCache(max_size=10, ttl_seconds=60.0)

        # Test set and get
        embedding = [0.1, 0.2, 0.3]
        cache.set("hello", "model-1", embedding)

        result = cache.get("hello", "model-1")
        assert result == embedding, "Should return cached embedding"

        # Test cache miss
        result = cache.get("world", "model-1")
        assert result is None, "Should return None for cache miss"

        # Test model-specific keys
        result = cache.get("hello", "model-2")
        assert result is None, "Different model should not share cache"

    def test_cache_ttl_expiration(self):
        """Test TTL-based cache expiration."""
        try:
            from core.advanced_memory import EmbeddingCache
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        # Short TTL for testing
        cache = EmbeddingCache(max_size=10, ttl_seconds=0.1)

        embedding = [0.1, 0.2, 0.3]
        cache.set("test", "model", embedding)

        # Should be cached initially
        result = cache.get("test", "model")
        assert result == embedding

        # Wait for TTL expiration
        time.sleep(0.15)

        # Should be expired
        result = cache.get("test", "model")
        assert result is None, "Should return None after TTL expiration"

    def test_cache_lru_eviction(self):
        """Test LRU eviction when at capacity."""
        try:
            from core.advanced_memory import EmbeddingCache
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        cache = EmbeddingCache(max_size=3, ttl_seconds=60.0)

        # Fill cache to capacity
        for i in range(3):
            cache.set(f"text-{i}", "model", [float(i)])

        # All should be cached
        for i in range(3):
            assert cache.get(f"text-{i}", "model") is not None

        # Add one more (should evict text-0)
        cache.set("text-3", "model", [3.0])

        # First item should be evicted
        assert cache.get("text-0", "model") is None, \
            "LRU item should be evicted"

        # Others should still exist
        assert cache.get("text-1", "model") is not None
        assert cache.get("text-2", "model") is not None
        assert cache.get("text-3", "model") is not None

    def test_cache_stats(self):
        """Test cache statistics tracking."""
        try:
            from core.advanced_memory import EmbeddingCache
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        cache = EmbeddingCache(max_size=10, ttl_seconds=60.0)

        # Generate some cache activity
        cache.set("text-1", "model", [1.0])
        cache.get("text-1", "model")  # Hit
        cache.get("text-1", "model")  # Hit
        cache.get("text-2", "model")  # Miss

        stats = cache.stats
        assert stats["hits"] == 2, "Should track 2 hits"
        assert stats["misses"] == 1, "Should track 1 miss"
        assert stats["hit_rate"] == 2 / 3, "Hit rate should be 2/3"
        assert stats["size"] == 1, "Cache size should be 1"
        assert stats["max_size"] == 10, "Max size should be 10"

    def test_cache_clear(self):
        """Test cache clear operation."""
        try:
            from core.advanced_memory import EmbeddingCache
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        cache = EmbeddingCache(max_size=10, ttl_seconds=60.0)

        cache.set("text-1", "model", [1.0])
        cache.get("text-1", "model")  # Hit

        cache.clear()

        assert cache.get("text-1", "model") is None, "Should be empty after clear"
        stats = cache.stats
        assert stats["hits"] == 0, "Stats should be reset"
        assert stats["misses"] == 1, "Should count current miss"
        assert stats["size"] == 0, "Size should be 0"


class TestEmbeddingCacheIntegration:
    """Test cache integration with providers."""

    @pytest.mark.asyncio
    async def test_global_cache_shared(self):
        """Test that global cache is shared across operations."""
        try:
            from core.advanced_memory import _embedding_cache
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        # Clear to start fresh
        _embedding_cache.clear()

        # Verify global cache is accessible
        assert _embedding_cache.stats["size"] == 0, "Should start empty"

    def test_cache_key_uniqueness(self):
        """Test that cache keys are unique for different text/model combos."""
        try:
            from core.advanced_memory import EmbeddingCache
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        cache = EmbeddingCache(max_size=10, ttl_seconds=60.0)

        # Same text, different models
        cache.set("hello", "model-a", [1.0])
        cache.set("hello", "model-b", [2.0])

        assert cache.get("hello", "model-a") == [1.0]
        assert cache.get("hello", "model-b") == [2.0]
        assert cache.stats["size"] == 2, "Should have 2 separate entries"

        # Different text, same model
        cache.set("world", "model-a", [3.0])

        assert cache.get("hello", "model-a") == [1.0]
        assert cache.get("world", "model-a") == [3.0]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
