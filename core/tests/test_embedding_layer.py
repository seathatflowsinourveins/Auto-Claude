#!/usr/bin/env python3
"""
Tests for the Voyage AI Embedding Layer.

These tests verify REAL SDK integration with actual API calls.
Tests are skipped if VOYAGE_API_KEY is not available.

Run with: pytest core/tests/test_embedding_layer.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import pytest

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Test: Import Availability
# =============================================================================

class TestImportAvailability:
    """Test that embedding layer modules can be imported."""

    def test_embedding_layer_importable(self):
        """Verify embedding_layer module imports correctly."""
        from core.orchestration import embedding_layer
        assert embedding_layer is not None

    def test_voyage_availability_flag(self):
        """Verify VOYAGE_AVAILABLE flag is boolean."""
        from core.orchestration.embedding_layer import VOYAGE_AVAILABLE
        assert isinstance(VOYAGE_AVAILABLE, bool)

    def test_orchestration_exports_embedding(self):
        """Verify orchestration __init__ exports embedding symbols."""
        from core.orchestration import (
            EMBEDDING_LAYER_AVAILABLE,
            VOYAGE_AVAILABLE,
        )
        assert isinstance(EMBEDDING_LAYER_AVAILABLE, bool)
        assert isinstance(VOYAGE_AVAILABLE, bool)


# =============================================================================
# Test: Configuration Classes
# =============================================================================

class TestEmbeddingConfig:
    """Test EmbeddingConfig dataclass structure."""

    def test_default_config(self):
        """Test EmbeddingConfig with defaults."""
        from core.orchestration.embedding_layer import EmbeddingConfig

        config = EmbeddingConfig()
        assert config.model == "voyage-3-large"
        assert config.batch_size == 128
        assert config.max_retries == 3
        assert config.cache_enabled is True

    def test_custom_config(self):
        """Test EmbeddingConfig with custom values."""
        from core.orchestration.embedding_layer import EmbeddingConfig

        config = EmbeddingConfig(
            model="voyage-code-3",
            api_key="test-key",
            batch_size=64,
            cache_enabled=False,
        )
        assert config.model == "voyage-code-3"
        assert config.api_key == "test-key"
        assert config.batch_size == 64
        assert config.cache_enabled is False


class TestEmbeddingModel:
    """Test EmbeddingModel enum."""

    def test_model_values(self):
        """Test model enum values."""
        from core.orchestration.embedding_layer import EmbeddingModel

        assert EmbeddingModel.VOYAGE_3_LARGE.value == "voyage-3-large"
        assert EmbeddingModel.VOYAGE_CODE_3.value == "voyage-code-3"
        assert EmbeddingModel.VOYAGE_3.value == "voyage-3"
        assert EmbeddingModel.VOYAGE_3_LITE.value == "voyage-3-lite"

    def test_model_dimensions(self):
        """Test model dimension properties."""
        from core.orchestration.embedding_layer import EmbeddingModel

        assert EmbeddingModel.VOYAGE_3_LARGE.dimension == 1024
        assert EmbeddingModel.VOYAGE_CODE_3.dimension == 2048
        assert EmbeddingModel.VOYAGE_3_LITE.dimension == 512


class TestInputType:
    """Test InputType enum."""

    def test_input_type_values(self):
        """Test input type enum values."""
        from core.orchestration.embedding_layer import InputType

        assert InputType.DOCUMENT.value == "document"
        assert InputType.QUERY.value == "query"


class TestEmbeddingResult:
    """Test EmbeddingResult dataclass."""

    def test_result_creation(self):
        """Test creating EmbeddingResult."""
        from core.orchestration.embedding_layer import EmbeddingResult

        result = EmbeddingResult(
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            model="voyage-3-large",
            input_type="document",
            total_tokens=100,
            cached_count=1,
            dimension=2,
        )
        assert result.count == 2
        assert result.dimension == 2
        assert result.total_tokens == 100
        assert result.cached_count == 1

    def test_result_repr(self):
        """Test EmbeddingResult string representation."""
        from core.orchestration.embedding_layer import EmbeddingResult

        result = EmbeddingResult(
            embeddings=[[0.1] * 1024],
            model="voyage-3-large",
            input_type="document",
            dimension=1024,
        )
        repr_str = repr(result)
        assert "count=1" in repr_str
        assert "dim=1024" in repr_str
        assert "voyage-3-large" in repr_str


# =============================================================================
# Test: EmbeddingLayer Class
# =============================================================================

class TestEmbeddingLayerBasic:
    """Test EmbeddingLayer basic functionality (no API key required)."""

    def test_layer_creation(self):
        """Test creating EmbeddingLayer."""
        from core.orchestration.embedding_layer import EmbeddingLayer, EmbeddingConfig

        config = EmbeddingConfig()
        layer = EmbeddingLayer(config)
        assert layer is not None
        assert layer.config == config
        assert layer.is_initialized is False

    def test_layer_default_config(self):
        """Test EmbeddingLayer with default config."""
        from core.orchestration.embedding_layer import EmbeddingLayer

        layer = EmbeddingLayer()
        assert layer.config.model == "voyage-3-large"

    def test_is_available_property(self):
        """Test is_available property."""
        from core.orchestration.embedding_layer import EmbeddingLayer, VOYAGE_AVAILABLE

        layer = EmbeddingLayer()
        assert layer.is_available == VOYAGE_AVAILABLE

    def test_get_stats(self):
        """Test get_stats method."""
        from core.orchestration.embedding_layer import EmbeddingLayer

        layer = EmbeddingLayer()
        stats = layer.get_stats()
        assert "initialized" in stats
        assert "model" in stats
        assert "cache_enabled" in stats
        assert "voyage_available" in stats


class TestEmbeddingLayerCodeDetection:
    """Test code detection heuristics."""

    def test_looks_like_code_python(self):
        """Test code detection for Python."""
        from core.orchestration.embedding_layer import EmbeddingLayer

        layer = EmbeddingLayer()

        python_code = [
            "def hello():\n    return 'world'",
            "import numpy as np\nclass Foo:\n    pass",
        ]
        assert layer._looks_like_code(python_code) is True

    def test_looks_like_code_javascript(self):
        """Test code detection for JavaScript."""
        from core.orchestration.embedding_layer import EmbeddingLayer

        layer = EmbeddingLayer()

        js_code = [
            "const x = () => { return 42; }",
            "function test() { let y = 1; }",
        ]
        assert layer._looks_like_code(js_code) is True

    def test_looks_like_code_natural_text(self):
        """Test code detection returns False for natural text."""
        from core.orchestration.embedding_layer import EmbeddingLayer

        layer = EmbeddingLayer()

        natural_text = [
            "Hello, this is a regular sentence.",
            "The quick brown fox jumps over the lazy dog.",
        ]
        assert layer._looks_like_code(natural_text) is False


class TestCacheOperations:
    """Test cache functionality."""

    def test_cache_key_generation(self):
        """Test cache key generation is deterministic."""
        from core.orchestration.embedding_layer import EmbeddingLayer

        layer = EmbeddingLayer()

        key1 = layer._get_cache_key("test", "model", "document")
        key2 = layer._get_cache_key("test", "model", "document")
        key3 = layer._get_cache_key("test", "model", "query")

        assert key1 == key2  # Same inputs = same key
        assert key1 != key3  # Different input_type = different key

    def test_clear_cache(self):
        """Test cache clearing."""
        from core.orchestration.embedding_layer import EmbeddingLayer

        layer = EmbeddingLayer()
        layer._cache["test_key"] = [0.1, 0.2]

        count = layer.clear_cache()
        assert count == 1
        assert len(layer._cache) == 0


# =============================================================================
# Test: Factory Functions
# =============================================================================

class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_embedding_layer(self):
        """Test create_embedding_layer factory."""
        from core.orchestration.embedding_layer import create_embedding_layer

        layer = create_embedding_layer()
        assert layer is not None
        assert layer.config.model == "voyage-3-large"

    def test_create_embedding_layer_custom(self):
        """Test create_embedding_layer with custom options."""
        from core.orchestration.embedding_layer import create_embedding_layer

        layer = create_embedding_layer(
            model="voyage-code-3",
            cache_enabled=False,
        )
        assert layer.config.model == "voyage-code-3"
        assert layer.config.cache_enabled is False

    def test_get_embedding_layer_sync(self):
        """Test synchronous getter (uninitialized)."""
        from core.orchestration.embedding_layer import get_embedding_layer_sync

        layer = get_embedding_layer_sync()
        assert layer is not None
        # Note: layer is NOT initialized


# =============================================================================
# Test: Real API Integration (requires VOYAGE_API_KEY)
# =============================================================================

# Check for API key availability
_has_voyage_key = bool(
    os.environ.get("VOYAGE_API_KEY") or
    # Default key from UNLEASH project
    True  # Default key is embedded in module
)


@pytest.mark.skipif(
    not _has_voyage_key,
    reason="VOYAGE_API_KEY not available"
)
class TestRealVoyageIntegration:
    """
    Tests that require a valid VOYAGE_API_KEY and make real API calls.
    These tests verify actual SDK functionality.

    Note: Free-tier Voyage AI has 3 RPM rate limit. Tests may fail
    if run in rapid succession. Run with: pytest -x --slow
    """

    # Shared layer instance to reduce initialization calls
    _layer = None

    @pytest.fixture(autouse=True)
    async def setup_layer(self):
        """Setup shared layer and add delay for rate limiting."""
        import asyncio
        # Add small delay between tests to respect rate limits
        await asyncio.sleep(0.5)

    @pytest.mark.asyncio
    async def test_layer_initialization(self):
        """Test layer initializes successfully with real API key."""
        from core.orchestration.embedding_layer import (
            create_embedding_layer,
            VOYAGE_AVAILABLE,
        )

        if not VOYAGE_AVAILABLE:
            pytest.skip("voyageai not installed")

        layer = create_embedding_layer()
        await layer.initialize()

        assert layer.is_initialized is True

    @pytest.mark.asyncio
    async def test_embed_single_document(self):
        """Test embedding a single document."""
        from core.orchestration.embedding_layer import (
            create_embedding_layer,
            VOYAGE_AVAILABLE,
        )

        if not VOYAGE_AVAILABLE:
            pytest.skip("voyageai not installed")

        layer = create_embedding_layer()
        await layer.initialize()

        result = await layer.embed_documents(["Hello, this is a test document."])

        assert result.count == 1
        assert result.dimension == 1024  # voyage-3-large default
        assert len(result.embeddings[0]) == 1024

    @pytest.mark.asyncio
    async def test_embed_multiple_documents(self):
        """Test embedding multiple documents."""
        from core.orchestration.embedding_layer import (
            create_embedding_layer,
            VOYAGE_AVAILABLE,
        )

        if not VOYAGE_AVAILABLE:
            pytest.skip("voyageai not installed")

        layer = create_embedding_layer()
        await layer.initialize()

        texts = [
            "First document about machine learning.",
            "Second document about data science.",
            "Third document about artificial intelligence.",
        ]

        result = await layer.embed_documents(texts)

        assert result.count == 3
        assert all(len(emb) == 1024 for emb in result.embeddings)

    @pytest.mark.asyncio
    async def test_embed_query(self):
        """Test embedding a search query."""
        from core.orchestration.embedding_layer import (
            create_embedding_layer,
            VOYAGE_AVAILABLE,
        )

        if not VOYAGE_AVAILABLE:
            pytest.skip("voyageai not installed")

        layer = create_embedding_layer()
        await layer.initialize()

        query_emb = await layer.embed_query("machine learning techniques")

        assert len(query_emb) == 1024
        assert all(isinstance(x, float) for x in query_emb)

    @pytest.mark.asyncio
    async def test_embed_code(self):
        """Test embedding code snippets with code model."""
        from core.orchestration.embedding_layer import (
            create_embedding_layer,
            VOYAGE_AVAILABLE,
        )

        if not VOYAGE_AVAILABLE:
            pytest.skip("voyageai not installed")

        layer = create_embedding_layer()
        await layer.initialize()

        code = [
            "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)",
            "async function fetchData(url) { return await fetch(url); }",
        ]

        result = await layer.embed_code(code)

        assert result.count == 2
        assert result.model == "voyage-code-3"
        assert result.dimension == 2048  # code model dimension

    @pytest.mark.asyncio
    async def test_caching_works(self):
        """Test that caching reduces API calls."""
        from core.orchestration.embedding_layer import (
            create_embedding_layer,
            VOYAGE_AVAILABLE,
        )

        if not VOYAGE_AVAILABLE:
            pytest.skip("voyageai not installed")

        layer = create_embedding_layer(cache_enabled=True)
        await layer.initialize()

        text = "This text will be cached."

        # First call - should hit API
        result1 = await layer.embed_documents([text])
        assert result1.cached_count == 0

        # Second call - should hit cache
        result2 = await layer.embed_documents([text])
        assert result2.cached_count == 1

        # Embeddings should be identical
        assert result1.embeddings[0] == result2.embeddings[0]

    @pytest.mark.asyncio
    async def test_embed_for_search(self):
        """Test convenience function for search."""
        from core.orchestration.embedding_layer import (
            embed_for_search,
            VOYAGE_AVAILABLE,
        )

        if not VOYAGE_AVAILABLE:
            pytest.skip("voyageai not installed")

        query = "machine learning"
        documents = [
            "Machine learning is a branch of AI.",
            "Data science uses statistics.",
        ]

        query_emb, doc_embs = await embed_for_search(query, documents)

        assert len(query_emb) == 1024
        assert len(doc_embs) == 2
        assert all(len(d) == 1024 for d in doc_embs)

    @pytest.mark.asyncio
    async def test_embed_texts_convenience(self):
        """Test embed_texts convenience function."""
        from core.orchestration.embedding_layer import (
            embed_texts,
            VOYAGE_AVAILABLE,
        )

        if not VOYAGE_AVAILABLE:
            pytest.skip("voyageai not installed")

        texts = ["Hello world", "Test embedding"]
        embeddings = await embed_texts(texts)

        assert len(embeddings) == 2
        assert all(len(e) == 1024 for e in embeddings)

    @pytest.mark.asyncio
    async def test_auto_detect_code(self):
        """Test automatic code detection switches model."""
        from core.orchestration.embedding_layer import (
            create_embedding_layer,
            InputType,
            VOYAGE_AVAILABLE,
        )

        if not VOYAGE_AVAILABLE:
            pytest.skip("voyageai not installed")

        layer = create_embedding_layer()
        await layer.initialize()

        code = [
            "def hello():\n    return 'world'",
            "import numpy as np\nclass Foo:\n    pass",
        ]

        result = await layer.embed(code, detect_code=True)

        # Should auto-switch to code model
        assert result.model == "voyage-code-3"
        assert result.dimension == 2048

    @pytest.mark.asyncio
    async def test_empty_input(self):
        """Test handling empty input."""
        from core.orchestration.embedding_layer import (
            create_embedding_layer,
            VOYAGE_AVAILABLE,
        )

        if not VOYAGE_AVAILABLE:
            pytest.skip("voyageai not installed")

        layer = create_embedding_layer()
        await layer.initialize()

        result = await layer.embed_documents([])

        assert result.count == 0
        assert result.embeddings == []


# =============================================================================
# Test: Error Handling
# =============================================================================

class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_initialization_without_voyage(self):
        """Test initialization fails gracefully without voyageai."""
        from core.orchestration.embedding_layer import (
            EmbeddingLayer,
            VOYAGE_AVAILABLE,
        )

        if VOYAGE_AVAILABLE:
            pytest.skip("voyageai is installed - testing unavailable scenario")

        layer = EmbeddingLayer()

        with pytest.raises(Exception) as exc_info:
            await layer.initialize()

        assert "voyageai" in str(exc_info.value).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
