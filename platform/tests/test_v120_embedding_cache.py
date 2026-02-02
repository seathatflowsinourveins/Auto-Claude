#!/usr/bin/env python3
"""
V120 Optimization Test: Embedding Cache with TTL

This test validates that embedding operations use caching:
1. EmbeddingCache class exists with LRU + TTL
2. OpenAIEmbeddingProvider uses cache for embed()
3. OpenAIEmbeddingProvider uses cache for embed_batch()
4. Cache statistics are tracked correctly

Expected Gains:
- API calls reduced by cache hit rate
- Latency: ~0ms for cache hits vs ~100ms for API calls

Test Date: 2026-01-30
"""

import os
import re
import sys
import time
import pytest

# Add platform to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestEmbeddingCachePatterns:
    """Test suite for embedding cache pattern verification."""

    def test_embedding_cache_class_exists(self):
        """Verify EmbeddingCache class exists with required methods."""
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "core", "advanced_memory.py"
        )

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Should have EmbeddingCache class
        assert "class EmbeddingCache:" in content, \
            "EmbeddingCache class should exist"

        # Find EmbeddingCache class section
        cache_start = content.find("class EmbeddingCache:")
        cache_end = content.find("class ", cache_start + 1)
        if cache_end == -1:
            cache_end = len(content)
        cache_section = content[cache_start:cache_end]

        # Should have LRU with OrderedDict
        assert "OrderedDict" in cache_section, \
            "EmbeddingCache should use OrderedDict for LRU"

        # Should have TTL support
        assert "ttl" in cache_section.lower(), \
            "EmbeddingCache should support TTL"

        # Should have get method
        assert "def get(" in cache_section, \
            "EmbeddingCache should have get method"

        # Should have set method
        assert "def set(" in cache_section, \
            "EmbeddingCache should have set method"

        # Should have stats property
        assert "def stats" in cache_section or "@property" in cache_section, \
            "EmbeddingCache should have stats property"

    def test_openai_provider_uses_cache_in_embed(self):
        """Verify OpenAIEmbeddingProvider.embed uses cache."""
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "core", "advanced_memory.py"
        )

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find OpenAIEmbeddingProvider class section
        provider_start = content.find("class OpenAIEmbeddingProvider")
        provider_end = content.find("\nclass ", provider_start + 1)
        if provider_end == -1:
            provider_end = len(content)
        provider_section = content[provider_start:provider_end]

        # Find embed method
        embed_match = re.search(r"async def embed\(self.*?\n(?:\s{8}.*\n)*", provider_section)
        assert embed_match, "OpenAIEmbeddingProvider should have embed method"
        embed_method = embed_match.group(0)

        # Should check cache
        assert "_embedding_cache.get" in embed_method, \
            "embed method should check cache with _embedding_cache.get"

        # Should set cache
        assert "_embedding_cache.set" in provider_section, \
            "embed method should cache results with _embedding_cache.set"

    def test_openai_provider_uses_cache_in_embed_batch(self):
        """Verify OpenAIEmbeddingProvider.embed_batch uses cache."""
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "core", "advanced_memory.py"
        )

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find OpenAIEmbeddingProvider class section
        provider_start = content.find("class OpenAIEmbeddingProvider")
        provider_end = content.find("\nclass ", provider_start + 1)
        if provider_end == -1:
            provider_end = len(content)
        provider_section = content[provider_start:provider_end]

        # Find embed_batch method
        batch_match = re.search(r"async def embed_batch\(self.*?\n(?:\s{8}.*\n)*", provider_section)
        assert batch_match, "OpenAIEmbeddingProvider should have embed_batch method"
        batch_method = batch_match.group(0)

        # Should check cache for each text
        assert "_embedding_cache.get" in batch_method, \
            "embed_batch should check cache with _embedding_cache.get"

        # Should set cache for new embeddings
        assert "_embedding_cache.set" in batch_method, \
            "embed_batch should cache results with _embedding_cache.set"

        # Should track uncached texts
        assert "uncached" in batch_method.lower(), \
            "embed_batch should track uncached texts separately"

    def test_global_cache_instance(self):
        """Verify global _embedding_cache instance exists."""
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "core", "advanced_memory.py"
        )

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Should have global cache instance
        assert "_embedding_cache = EmbeddingCache(" in content, \
            "Global _embedding_cache instance should exist"


class TestEmbeddingCacheBehavior:
    """Test actual behavior of embedding cache."""

    def test_cache_basic_operations(self):
        """Test basic cache get/set operations."""
        try:
            from platform.core.advanced_memory import EmbeddingCache
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
            from platform.core.advanced_memory import EmbeddingCache
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
            from platform.core.advanced_memory import EmbeddingCache
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
            from platform.core.advanced_memory import EmbeddingCache
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
            from platform.core.advanced_memory import EmbeddingCache
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
            from platform.core.advanced_memory import _embedding_cache
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        # Clear to start fresh
        _embedding_cache.clear()

        # Verify global cache is accessible
        assert _embedding_cache.stats["size"] == 0, "Should start empty"

    def test_cache_key_uniqueness(self):
        """Test that cache keys are unique for different text/model combos."""
        try:
            from platform.core.advanced_memory import EmbeddingCache
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
