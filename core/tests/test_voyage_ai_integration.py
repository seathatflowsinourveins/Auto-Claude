"""
Real Voyage AI Integration Tests - ITERATION 6.

These tests use the REAL Voyage AI API (not mocks) to verify:
- API connectivity and authentication
- Document embedding (input_type="document")
- Query embedding (input_type="query")
- End-to-end semantic search pipeline

NOTE: Free tier has 3 RPM limit. Tests are designed to minimize API calls.
Tests are split into:
- Fast tests: No API calls (availability, configuration)
- Slow tests: API calls with rate limiting (marked with @pytest.mark.slow)

Run fast tests only: pytest -m "not slow"
Run all tests: pytest --slow (will take ~3 minutes due to rate limits)
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import numpy as np
import pytest


# =============================================================================
# Constants
# =============================================================================

VOYAGE_API_KEY = "pa-KCpYL_zzmvoPK1dM6tN5kdCD8e6qnAndC-dSTlCuzK4"
RATE_LIMIT_DELAY = 21.0  # 21 seconds between API calls (3 RPM limit)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def voyage_api_key() -> str:
    """Get the Voyage AI API key."""
    return VOYAGE_API_KEY


@pytest.fixture
def sample_documents() -> list[str]:
    """Sample documents for embedding tests."""
    return [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Neural networks are inspired by the human brain.",
    ]


@pytest.fixture
def sample_code_snippets() -> list[str]:
    """Sample code snippets for code embedding tests."""
    return [
        "def hello_world():\n    print('Hello, World!')\n    return 42",
        "async function fetchData(url) {\n    return await fetch(url);\n}",
    ]


# =============================================================================
# Fast Tests (No API Calls)
# =============================================================================

class TestVoyageAIAvailability:
    """Test Voyage AI SDK availability and configuration."""

    def test_voyageai_installed(self):
        """Verify voyageai package is installed."""
        import voyageai
        assert hasattr(voyageai, "Client")
        assert hasattr(voyageai, "__version__")

    def test_embedding_layer_importable(self):
        """Verify embedding layer imports correctly."""
        from core.orchestration.embedding_layer import (
            EmbeddingLayer,
            EmbeddingConfig,
            EmbeddingResult,
            EmbeddingModel,
            InputType,
            VOYAGE_AVAILABLE,
        )
        assert VOYAGE_AVAILABLE is True
        assert EmbeddingLayer is not None
        assert EmbeddingConfig is not None

    def test_embedding_models_defined(self):
        """Verify all embedding models are defined."""
        from core.orchestration.embedding_layer import EmbeddingModel

        assert EmbeddingModel.VOYAGE_3_LARGE.value == "voyage-3-large"
        assert EmbeddingModel.VOYAGE_CODE_3.value == "voyage-code-3"
        assert EmbeddingModel.VOYAGE_3.value == "voyage-3"
        assert EmbeddingModel.VOYAGE_3_LITE.value == "voyage-3-lite"

    def test_model_dimensions(self):
        """Verify model dimension properties."""
        from core.orchestration.embedding_layer import EmbeddingModel

        assert EmbeddingModel.VOYAGE_3_LARGE.dimension == 1024
        assert EmbeddingModel.VOYAGE_CODE_3.dimension == 2048
        assert EmbeddingModel.VOYAGE_3.dimension == 1024
        assert EmbeddingModel.VOYAGE_3_LITE.dimension == 512

    def test_config_defaults(self):
        """Test EmbeddingConfig default values."""
        from core.orchestration.embedding_layer import EmbeddingConfig

        config = EmbeddingConfig()
        assert config.model == "voyage-3-large"
        assert config.batch_size == 128
        assert config.max_retries == 3
        assert config.cache_enabled is True

    def test_input_type_enum(self):
        """Test InputType enum values."""
        from core.orchestration.embedding_layer import InputType

        assert InputType.DOCUMENT.value == "document"
        assert InputType.QUERY.value == "query"

    def test_factory_function_creates_layer(self):
        """Test create_embedding_layer factory."""
        from core.orchestration.embedding_layer import create_embedding_layer

        layer = create_embedding_layer(
            model="voyage-3",
            api_key="test-key",
            cache_enabled=False,
        )
        assert layer.config.model == "voyage-3"
        assert layer.config.api_key == "test-key"
        assert layer.config.cache_enabled is False
        assert layer.is_initialized is False

    def test_code_detection_heuristics(self):
        """Test the code detection logic."""
        from core.orchestration.embedding_layer import EmbeddingLayer

        layer = EmbeddingLayer()

        # Should detect as code
        code_texts = [
            "def hello():\n    return 'world'",
            "function test() { return 42; }",
        ]
        assert layer._looks_like_code(code_texts) is True

        # Should NOT detect as code
        normal_texts = [
            "Hello, this is a normal sentence.",
            "The weather is nice today.",
        ]
        assert layer._looks_like_code(normal_texts) is False


class TestCacheLogic:
    """Test caching logic without API calls."""

    def test_cache_key_generation(self):
        """Test cache key is deterministic."""
        from core.orchestration.embedding_layer import EmbeddingLayer

        layer = EmbeddingLayer()

        key1 = layer._get_cache_key("test", "voyage-3", "document")
        key2 = layer._get_cache_key("test", "voyage-3", "document")
        key3 = layer._get_cache_key("test", "voyage-3", "query")  # Different input type

        assert key1 == key2
        assert key1 != key3

    def test_cache_check_with_empty_cache(self):
        """Test cache check returns all texts when cache is empty."""
        from core.orchestration.embedding_layer import EmbeddingLayer, EmbeddingConfig

        layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True))
        texts = ["doc1", "doc2", "doc3"]

        uncached, indices, cached = layer._check_cache(texts, "voyage-3", "document")

        assert uncached == texts
        assert indices == [0, 1, 2]
        assert cached == {}

    def test_cache_disabled(self):
        """Test cache bypass when disabled."""
        from core.orchestration.embedding_layer import EmbeddingLayer, EmbeddingConfig

        layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=False))

        # Manually add to cache (simulating previous call)
        layer._cache["some_key"] = [1.0, 2.0, 3.0]

        # Should still return all texts as uncached
        texts = ["doc1"]
        uncached, indices, cached = layer._check_cache(texts, "voyage-3", "document")

        assert uncached == texts
        assert cached == {}

    def test_cache_update(self):
        """Test cache update adds entries."""
        from core.orchestration.embedding_layer import EmbeddingLayer, EmbeddingConfig

        layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True))
        texts = ["doc1", "doc2"]
        embeddings = [[1.0, 2.0], [3.0, 4.0]]

        layer._update_cache(texts, embeddings, "voyage-3", "document")

        assert layer.get_stats()["cache_size"] == 2

    def test_cache_clear(self):
        """Test cache clearing."""
        from core.orchestration.embedding_layer import EmbeddingLayer, EmbeddingConfig

        layer = EmbeddingLayer(EmbeddingConfig(cache_enabled=True))
        layer._cache = {"key1": [1.0], "key2": [2.0]}

        cleared = layer.clear_cache()

        assert cleared == 2
        assert layer.get_stats()["cache_size"] == 0


class TestLayerStats:
    """Test statistics without API calls."""

    def test_stats_before_init(self):
        """Test stats on uninitialized layer."""
        from core.orchestration.embedding_layer import EmbeddingLayer

        layer = EmbeddingLayer()
        stats = layer.get_stats()

        assert stats["initialized"] is False
        assert stats["model"] == "voyage-3-large"
        assert stats["cache_size"] == 0
        assert stats["voyage_available"] is True


# =============================================================================
# Slow Tests (Real API Calls - Rate Limited)
# =============================================================================

@pytest.mark.slow
class TestRealAPIConnection:
    """Test real connection to Voyage AI API."""

    @pytest.mark.asyncio
    async def test_api_connection_and_embedding(self, voyage_api_key):
        """
        Comprehensive API test - tests connection, initialization, and embedding.
        This is a single consolidated test to minimize API calls.
        """
        import voyageai
        from core.orchestration.embedding_layer import (
            create_embedding_layer,
            EmbeddingModel,
        )

        # Test 1: Direct API connection
        client = voyageai.Client(api_key=voyage_api_key)
        result = client.embed(
            texts=["Connection test"],
            model="voyage-3",
            input_type="document",
            truncation=True,
        )
        assert result.embeddings is not None
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 1024

        # Test 2: EmbeddingLayer initialization
        layer = create_embedding_layer(
            model=EmbeddingModel.VOYAGE_3.value,
            api_key=voyage_api_key,
        )
        await layer.initialize()
        assert layer.is_initialized is True


@pytest.mark.slow
class TestRealEmbeddings:
    """Test real embeddings with rate limiting between tests."""

    @pytest.mark.asyncio
    async def test_document_and_query_embeddings(self, voyage_api_key, sample_documents):
        """
        Comprehensive embedding test covering documents, queries, and search.
        Single test to minimize API calls within rate limits.
        """
        from core.orchestration.embedding_layer import (
            create_embedding_layer,
            InputType,
        )

        layer = create_embedding_layer(
            model="voyage-3",
            api_key=voyage_api_key,
            cache_enabled=True,
        )
        await layer.initialize()

        # Test 1: Embed documents
        doc_result = await layer.embed_documents(sample_documents)
        assert doc_result.count == len(sample_documents)
        assert doc_result.dimension == 1024
        assert doc_result.input_type == "document"

        # Rate limit pause
        print("\n  [Rate limit] Waiting for next API call...")
        await asyncio.sleep(RATE_LIMIT_DELAY)

        # Test 2: Embed query
        query = "What is machine learning?"
        query_embedding = await layer.embed_query(query)
        assert len(query_embedding) == 1024

        # Test 3: Verify embeddings are normalized
        doc_emb = np.array(doc_result.embeddings[0])
        doc_norm = np.linalg.norm(doc_emb)
        assert 0.99 < doc_norm < 1.01, f"Doc not normalized: {doc_norm}"

        query_emb = np.array(query_embedding)
        query_norm = np.linalg.norm(query_emb)
        assert 0.99 < query_norm < 1.01, f"Query not normalized: {query_norm}"

        # Test 4: Semantic similarity - ML query should match ML document
        similarities = []
        for emb in doc_result.embeddings:
            sim = np.dot(query_emb, np.array(emb))
            similarities.append(sim)

        best_idx = np.argmax(similarities)
        # First document is about machine learning
        assert best_idx == 0, f"Expected ML doc (idx 0) to rank highest, got {best_idx}"

        # Test 5: Caching works
        cached_result = await layer.embed_documents(sample_documents[:1])
        assert cached_result.cached_count == 1


@pytest.mark.slow
class TestRealCodeEmbedding:
    """Test code embedding with voyage-code-3 model."""

    @pytest.mark.asyncio
    async def test_code_embedding_with_auto_detect(self, voyage_api_key, sample_code_snippets):
        """Test code embeddings with auto-detection and voyage-code-3 model."""
        from core.orchestration.embedding_layer import create_embedding_layer

        layer = create_embedding_layer(
            model="voyage-3",  # Start with text model
            api_key=voyage_api_key,
        )
        await layer.initialize()

        # Test 1: Auto-detect should switch to code model
        result = await layer.embed(
            sample_code_snippets,
            detect_code=True,
        )

        assert result.model == "voyage-code-3"
        assert result.dimension == 2048
        assert result.count == len(sample_code_snippets)


@pytest.mark.slow
class TestRealErrorHandling:
    """Test error handling with real API."""

    @pytest.mark.asyncio
    async def test_invalid_api_key_raises_error(self):
        """Test that invalid API key raises appropriate error."""
        from core.orchestration.embedding_layer import create_embedding_layer

        layer = create_embedding_layer(
            model="voyage-3",
            api_key="invalid-key-12345",
        )

        with pytest.raises(Exception) as excinfo:
            await layer.initialize()

        error_msg = str(excinfo.value).lower()
        assert "invalid" in error_msg or "api" in error_msg or "401" in error_msg

    @pytest.mark.asyncio
    async def test_empty_input_handling(self, voyage_api_key):
        """Test handling of empty input."""
        from core.orchestration.embedding_layer import create_embedding_layer

        layer = create_embedding_layer(
            model="voyage-3",
            api_key=voyage_api_key,
        )
        await layer.initialize()

        result = await layer.embed_documents([])

        assert result.count == 0
        assert result.embeddings == []


# =============================================================================
# Integration Summary Test
# =============================================================================

@pytest.mark.slow
class TestVoyageIntegrationSummary:
    """Final integration summary test."""

    @pytest.mark.asyncio
    async def test_full_semantic_search_pipeline(self, voyage_api_key):
        """
        Complete end-to-end semantic search test.
        This simulates a real use case: indexing documents and finding relevant ones.
        """
        from core.orchestration.embedding_layer import create_embedding_layer

        layer = create_embedding_layer(
            model="voyage-3",
            api_key=voyage_api_key,
            cache_enabled=True,
        )
        await layer.initialize()

        # Documents to index
        documents = [
            "The quick brown fox jumps over the lazy dog.",
            "Python is a versatile programming language.",
            "Machine learning enables computers to learn from data.",
            "Neural networks are computational models inspired by brains.",
            "Natural language processing handles text and speech.",
        ]

        # Index documents
        doc_result = await layer.embed_documents(documents)
        doc_embeddings = [np.array(e) for e in doc_result.embeddings]

        # Wait for rate limit
        print("\n  [Rate limit] Waiting before query...")
        await asyncio.sleep(RATE_LIMIT_DELAY)

        # Search query
        query = "How do computers learn patterns from examples?"
        query_embedding = np.array(await layer.embed_query(query))

        # Calculate similarities
        similarities = []
        for i, doc_emb in enumerate(doc_embeddings):
            sim = np.dot(query_embedding, doc_emb)
            similarities.append((i, sim, documents[i]))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Top result should be about machine learning
        top_idx, top_sim, top_doc = similarities[0]
        print(f"\n  Query: '{query}'")
        print(f"  Top match ({top_sim:.3f}): '{top_doc}'")

        assert "machine learning" in top_doc.lower() or "neural" in top_doc.lower(), (
            f"Expected ML-related doc to rank first, got: {top_doc}"
        )

        # Stats verification
        stats = layer.get_stats()
        assert stats["initialized"] is True
        assert stats["cache_size"] >= 5  # At least the 5 documents cached


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    # Run fast tests by default
    pytest.main([__file__, "-v", "-m", "not slow"])
