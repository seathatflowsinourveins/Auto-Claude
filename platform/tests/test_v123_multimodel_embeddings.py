#!/usr/bin/env python3
"""
V123 Optimization Test: Multi-Model Embedding Support

Tests multi-model embedding by importing and testing real classes -
not by grepping file contents.

Test Date: 2026-01-30, Updated: 2026-02-02 (V14 Iter 55)
"""

import os
import sys
import pytest

# Add platform to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestMultiModelStructure:
    """Test multi-model embedding structure by importing real classes."""

    def test_embedding_model_enum_importable(self):
        """EmbeddingModel enum should be importable with Voyage and local models."""
        try:
            from core.advanced_memory import EmbeddingModel
        except ImportError:
            pytest.skip("advanced_memory not importable")

        assert hasattr(EmbeddingModel, "VOYAGE_CODE_3"), "Should have voyage-code-3"
        assert hasattr(EmbeddingModel, "VOYAGE_3_5"), "Should have voyage-3.5"
        assert hasattr(EmbeddingModel, "LOCAL_MINILM"), "Should have MiniLM"

    def test_voyage_provider_importable(self):
        """VoyageEmbeddingProvider should be importable with required methods."""
        try:
            from core.advanced_memory import VoyageEmbeddingProvider
        except ImportError:
            pytest.skip("advanced_memory not importable")

        provider = VoyageEmbeddingProvider(api_key="test", model="voyage-code-3")
        assert hasattr(provider, "embed")
        assert hasattr(provider, "embed_batch")
        assert hasattr(VoyageEmbeddingProvider, "_circuit_breaker")
        assert hasattr(VoyageEmbeddingProvider, "get_circuit_stats")

    def test_sentence_transformer_provider_importable(self):
        """SentenceTransformerEmbeddingProvider should be importable."""
        try:
            from core.advanced_memory import SentenceTransformerEmbeddingProvider
        except ImportError:
            pytest.skip("advanced_memory not importable")

        assert SentenceTransformerEmbeddingProvider is not None

    def test_create_embedding_provider_factory(self):
        """create_embedding_provider factory should be importable and callable."""
        try:
            from core.advanced_memory import create_embedding_provider
        except ImportError:
            pytest.skip("advanced_memory not importable")

        assert callable(create_embedding_provider)


class TestSentenceTransformerProvider:
    """Test SentenceTransformerEmbeddingProvider behavior."""

    @pytest.mark.asyncio
    async def test_local_provider_embed(self):
        """Test local sentence-transformers embedding."""
        try:
            from core.advanced_memory import (
                SentenceTransformerEmbeddingProvider,
                reset_memory_metrics,
            )
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        reset_memory_metrics()

        try:
            provider = SentenceTransformerEmbeddingProvider("all-MiniLM-L6-v2")
            result = await provider.embed("This is a test sentence for embedding.")

            # Should have correct dimensions
            assert len(result.embedding) == 384, f"Expected 384 dims, got {len(result.embedding)}"
            assert result.model == "all-MiniLM-L6-v2"
            assert result.tokens_used == 0  # Local, no token cost
            assert result.dimensions == 384

        except RuntimeError as e:
            if "sentence-transformers not installed" in str(e):
                pytest.skip("sentence-transformers not available")
            raise

    @pytest.mark.asyncio
    async def test_local_provider_batch(self):
        """Test local batch embedding."""
        try:
            from core.advanced_memory import SentenceTransformerEmbeddingProvider
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        try:
            provider = SentenceTransformerEmbeddingProvider("all-MiniLM-L6-v2")

            texts = [
                "First test sentence.",
                "Second test sentence.",
                "Third test sentence.",
            ]
            results = await provider.embed_batch(texts)

            assert len(results) == 3
            for r in results:
                assert len(r.embedding) == 384
                assert r.tokens_used == 0

        except RuntimeError as e:
            if "sentence-transformers not installed" in str(e):
                pytest.skip("sentence-transformers not available")
            raise

    @pytest.mark.asyncio
    async def test_local_provider_caching(self):
        """Test that local provider uses cache."""
        try:
            from core.advanced_memory import (
                SentenceTransformerEmbeddingProvider,
                _embedding_cache,
                get_memory_stats,
                reset_memory_metrics,
            )
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        reset_memory_metrics()
        _embedding_cache.clear()

        try:
            provider = SentenceTransformerEmbeddingProvider("all-MiniLM-L6-v2")

            # First call - cache miss
            result1 = await provider.embed("Cached test sentence.")

            # Second call - should hit cache
            result2 = await provider.embed("Cached test sentence.")

            # Embeddings should be identical
            assert result1.embedding == result2.embedding

            # Check metrics show cache activity
            stats = get_memory_stats()
            cache_stats = stats.get("cache", stats.get("embedding", {}))
            assert cache_stats.get("cache_hits", cache_stats.get("hits", 0)) >= 1, \
                f"Should have recorded cache hit, got stats: {stats}"

        except RuntimeError as e:
            if "sentence-transformers not installed" in str(e):
                pytest.skip("sentence-transformers not available")
            raise


class TestFactoryFunction:
    """Test create_embedding_provider factory."""

    def test_factory_auto_detects_voyage(self):
        """Test factory auto-detects Voyage models."""
        try:
            from core.advanced_memory import (
                create_embedding_provider,
                VoyageEmbeddingProvider,
            )
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        # With key (env or explicit), should create VoyageEmbeddingProvider
        provider = create_embedding_provider("voyage-code-3", api_key="fake-key")
        assert isinstance(provider, VoyageEmbeddingProvider)

    def test_factory_auto_detects_openai(self):
        """Test factory auto-detects OpenAI models."""
        try:
            from core.advanced_memory import (
                create_embedding_provider,
                OpenAIEmbeddingProvider,
            )
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        # With explicit key, should create OpenAIEmbeddingProvider
        provider = create_embedding_provider("text-embedding-3-small", api_key="fake-key")
        assert isinstance(provider, OpenAIEmbeddingProvider)

        # With fake key, should create OpenAIEmbeddingProvider
        provider = create_embedding_provider("text-embedding-3-small", api_key="fake-key")
        assert isinstance(provider, OpenAIEmbeddingProvider)

    def test_factory_auto_detects_local(self):
        """Test factory auto-detects local models."""
        try:
            from core.advanced_memory import (
                create_embedding_provider,
                SentenceTransformerEmbeddingProvider,
            )
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        # Should not require API key for local models
        provider = create_embedding_provider("all-MiniLM-L6-v2")
        assert isinstance(provider, SentenceTransformerEmbeddingProvider)

        # Also test BAAI models
        provider = create_embedding_provider("BAAI/bge-small-en-v1.5")
        assert isinstance(provider, SentenceTransformerEmbeddingProvider)

    def test_factory_provider_override(self):
        """Test factory respects provider_type override."""
        try:
            from core.advanced_memory import (
                create_embedding_provider,
                LocalEmbeddingProvider,
            )
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        # Force local-hash even for unknown model
        provider = create_embedding_provider(
            "unknown-model",
            provider_type="local-hash"
        )
        assert isinstance(provider, LocalEmbeddingProvider)


class TestVoyageProvider:
    """Test VoyageEmbeddingProvider (without real API calls)."""

    def test_voyage_provider_initialization(self):
        """Test Voyage provider initializes correctly."""
        try:
            from core.advanced_memory import VoyageEmbeddingProvider
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        provider = VoyageEmbeddingProvider(api_key="fake-key", model="voyage-code-3")
        assert provider._model == "voyage-code-3"
        assert provider._dimensions == 1024

    def test_voyage_provider_circuit_breaker(self):
        """Test Voyage provider has circuit breaker."""
        try:
            from core.advanced_memory import VoyageEmbeddingProvider
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        # Should have circuit breaker as class variable
        assert hasattr(VoyageEmbeddingProvider, "_circuit_breaker")
        assert hasattr(VoyageEmbeddingProvider, "get_circuit_stats")

        stats = VoyageEmbeddingProvider.get_circuit_stats()
        assert "state" in stats
        assert "total_calls" in stats


class TestProviderIntegration:
    """Test provider integration with memory system."""

    @pytest.mark.asyncio
    async def test_providers_record_metrics(self):
        """Test that all providers record metrics correctly."""
        try:
            from core.advanced_memory import (
                LocalEmbeddingProvider,
                get_memory_stats,
                reset_memory_metrics,
            )
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        reset_memory_metrics()

        # Test local provider
        provider = LocalEmbeddingProvider()
        await provider.embed("Test text")

        stats = get_memory_stats()
        assert stats["embedding"]["calls"] >= 1, "Should record local provider calls"

    @pytest.mark.asyncio
    async def test_sentence_transformer_records_metrics(self):
        """Test sentence-transformers records metrics."""
        try:
            from core.advanced_memory import (
                SentenceTransformerEmbeddingProvider,
                get_memory_stats,
                reset_memory_metrics,
            )
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        reset_memory_metrics()

        try:
            provider = SentenceTransformerEmbeddingProvider("all-MiniLM-L6-v2")
            await provider.embed("Test sentence")

            stats = get_memory_stats()
            assert stats["embedding"]["calls"] >= 1

        except RuntimeError as e:
            if "sentence-transformers not installed" in str(e):
                pytest.skip("sentence-transformers not available")
            raise


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
