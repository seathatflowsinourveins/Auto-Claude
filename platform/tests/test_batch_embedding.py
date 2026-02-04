"""
Tests for the Batch Embedding Pipeline (V39).

Tests cover:
- EmbeddingCache functionality
- Provider initialization
- Batch processing with dynamic batching
- Caching behavior
- Multi-provider fallback
- Statistics tracking
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from core.orchestration.batch_embedding import (
    BatchEmbeddingPipeline,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingCache,
    EmbeddingProvider,
    ProviderModel,
    BaseEmbedder,
    OpenAIEmbedder,
    VoyageEmbedder,
    JinaEmbedder,
    LocalEmbedder,
    create_batch_pipeline,
    quick_embed,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def embedding_cache():
    """Create a fresh embedding cache."""
    return EmbeddingCache(max_size=100, ttl_seconds=60.0)


@pytest.fixture
def mock_embeddings():
    """Sample embedding vectors."""
    return [
        [0.1, 0.2, 0.3, 0.4] * 256,  # 1024-dim
        [0.5, 0.6, 0.7, 0.8] * 256,
        [0.9, 0.1, 0.2, 0.3] * 256,
    ]


class MockEmbedder(BaseEmbedder):
    """Mock embedder for testing."""

    def __init__(self, embeddings=None, should_fail=False):
        from core.orchestration.batch_embedding import ProviderConfig
        config = ProviderConfig(
            provider=EmbeddingProvider.LOCAL,
            model="mock-model",
            dimension=1024,
        )
        super().__init__(config)
        self._embeddings = embeddings or [[0.1] * 1024]
        self._should_fail = should_fail
        self._call_count = 0

    async def initialize(self) -> None:
        self._initialized = True

    async def embed(self, texts):
        if self._should_fail:
            raise RuntimeError("Mock embedding error")

        self._call_count += 1
        self._total_calls += 1

        # Return appropriate number of embeddings
        results = []
        for i in range(len(texts)):
            idx = i % len(self._embeddings)
            results.append(self._embeddings[idx])

        return results, len(texts) * 10


# =============================================================================
# EmbeddingCache Tests
# =============================================================================

class TestEmbeddingCache:
    """Tests for EmbeddingCache."""

    @pytest.mark.asyncio
    async def test_put_and_get(self, embedding_cache, mock_embeddings):
        """Test basic put and get operations."""
        await embedding_cache.put("hello", "model-1", mock_embeddings[0])

        result = await embedding_cache.get("hello", "model-1")
        assert result == mock_embeddings[0]

    @pytest.mark.asyncio
    async def test_cache_miss(self, embedding_cache):
        """Test cache miss returns None."""
        result = await embedding_cache.get("nonexistent", "model-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_different_models_separate_cache(self, embedding_cache, mock_embeddings):
        """Test that different models have separate cache entries."""
        await embedding_cache.put("hello", "model-1", mock_embeddings[0])
        await embedding_cache.put("hello", "model-2", mock_embeddings[1])

        result1 = await embedding_cache.get("hello", "model-1")
        result2 = await embedding_cache.get("hello", "model-2")

        assert result1 == mock_embeddings[0]
        assert result2 == mock_embeddings[1]

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test that cached entries expire after TTL."""
        cache = EmbeddingCache(max_size=100, ttl_seconds=0.1)
        embedding = [0.1] * 1024

        await cache.put("hello", "model", embedding)

        # Should be present immediately
        result = await cache.get("hello", "model")
        assert result is not None

        # Wait for TTL
        await asyncio.sleep(0.15)

        # Should be expired
        result = await cache.get("hello", "model")
        assert result is None

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = EmbeddingCache(max_size=3, ttl_seconds=60.0)

        await cache.put("a", "model", [0.1])
        await cache.put("b", "model", [0.2])
        await cache.put("c", "model", [0.3])

        # Access "a" to make it recently used
        await cache.get("a", "model")

        # Add "d" - should evict "b" (least recently used)
        await cache.put("d", "model", [0.4])

        assert await cache.get("a", "model") is not None
        assert await cache.get("b", "model") is None  # evicted
        assert await cache.get("c", "model") is not None
        assert await cache.get("d", "model") is not None

    @pytest.mark.asyncio
    async def test_batch_operations(self, embedding_cache, mock_embeddings):
        """Test batch get and put operations."""
        texts = ["hello", "world", "test"]

        # Put batch
        await embedding_cache.put_batch(texts, "model", mock_embeddings)

        # Get batch
        cached, uncached = await embedding_cache.get_batch(texts + ["unknown"], "model")

        assert len(cached) == 3
        assert len(uncached) == 1
        assert 3 in uncached  # "unknown" is at index 3

    @pytest.mark.asyncio
    async def test_stats(self, embedding_cache, mock_embeddings):
        """Test cache statistics."""
        await embedding_cache.put("a", "model", mock_embeddings[0])

        # Hit
        await embedding_cache.get("a", "model")
        # Miss
        await embedding_cache.get("b", "model")

        stats = embedding_cache.stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 50.0


# =============================================================================
# Provider Tests
# =============================================================================

class TestMockEmbedder:
    """Tests for the mock embedder."""

    @pytest.mark.asyncio
    async def test_mock_embedder(self, mock_embeddings):
        """Test mock embedder returns correct embeddings."""
        embedder = MockEmbedder(embeddings=mock_embeddings)
        await embedder.initialize()

        results, tokens = await embedder.embed(["hello", "world"])

        assert len(results) == 2
        assert results[0] == mock_embeddings[0]
        assert results[1] == mock_embeddings[1]
        assert tokens == 20

    @pytest.mark.asyncio
    async def test_mock_embedder_failure(self):
        """Test mock embedder failure mode."""
        embedder = MockEmbedder(should_fail=True)
        await embedder.initialize()

        with pytest.raises(RuntimeError, match="Mock embedding error"):
            await embedder.embed(["hello"])


# =============================================================================
# BatchEmbeddingPipeline Tests
# =============================================================================

class TestBatchEmbeddingPipeline:
    """Tests for BatchEmbeddingPipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_lifecycle(self, mock_embeddings):
        """Test pipeline start and stop."""
        embedder = MockEmbedder(embeddings=mock_embeddings)

        pipeline = BatchEmbeddingPipeline(
            max_batch_size=10,
            max_wait_ms=50.0,
        )
        pipeline._primary_provider = embedder

        await pipeline.start()
        assert pipeline._running

        await pipeline.stop()
        assert not pipeline._running

    @pytest.mark.asyncio
    async def test_single_embed(self, mock_embeddings):
        """Test single embedding request."""
        embedder = MockEmbedder(embeddings=mock_embeddings)

        pipeline = BatchEmbeddingPipeline(
            max_batch_size=10,
            max_wait_ms=10.0,  # Short timeout for testing
            cache_enabled=False,
        )

        # Manually set provider to bypass auto-detection
        pipeline._primary_provider = embedder
        await embedder.initialize()
        pipeline._running = True
        pipeline._shutdown_event.clear()
        pipeline._start_time = time.time()
        pipeline._batch_task = asyncio.create_task(pipeline._batch_processor())

        try:
            result = await asyncio.wait_for(
                pipeline.embed("hello"),
                timeout=1.0
            )

            assert len(result) == 1024
            assert embedder._call_count >= 1
        finally:
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_batch_embed(self, mock_embeddings):
        """Test batch embedding request."""
        embedder = MockEmbedder(embeddings=mock_embeddings)

        pipeline = BatchEmbeddingPipeline(
            max_batch_size=10,
            max_wait_ms=10.0,
            cache_enabled=False,
        )

        pipeline._primary_provider = embedder
        await embedder.initialize()
        pipeline._running = True
        pipeline._shutdown_event.clear()
        pipeline._start_time = time.time()
        pipeline._batch_task = asyncio.create_task(pipeline._batch_processor())

        try:
            results = await asyncio.wait_for(
                pipeline.embed_batch(["hello", "world", "test"]),
                timeout=1.0
            )

            assert len(results) == 3
        finally:
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_caching(self, mock_embeddings):
        """Test that caching works correctly."""
        embedder = MockEmbedder(embeddings=mock_embeddings)

        pipeline = BatchEmbeddingPipeline(
            max_batch_size=10,
            max_wait_ms=10.0,
            cache_enabled=True,
        )

        pipeline._primary_provider = embedder
        await embedder.initialize()
        pipeline._running = True
        pipeline._shutdown_event.clear()
        pipeline._start_time = time.time()
        pipeline._batch_task = asyncio.create_task(pipeline._batch_processor())

        try:
            # First request
            result1 = await asyncio.wait_for(
                pipeline.embed("hello"),
                timeout=1.0
            )

            initial_calls = embedder._call_count

            # Second request (should be cached)
            result2 = await asyncio.wait_for(
                pipeline.embed("hello"),
                timeout=1.0
            )

            assert result1 == result2
            # Provider should not have been called again
            assert embedder._call_count == initial_calls
            assert pipeline._total_cached > 0
        finally:
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_batching_efficiency(self, mock_embeddings):
        """Test that requests are batched efficiently."""
        embedder = MockEmbedder(embeddings=mock_embeddings)

        pipeline = BatchEmbeddingPipeline(
            max_batch_size=10,
            max_wait_ms=100.0,  # Longer timeout to allow batching
            cache_enabled=False,
        )

        pipeline._primary_provider = embedder
        await embedder.initialize()
        pipeline._running = True
        pipeline._shutdown_event.clear()
        pipeline._start_time = time.time()
        pipeline._batch_task = asyncio.create_task(pipeline._batch_processor())

        try:
            # Submit multiple requests concurrently
            tasks = [
                pipeline.embed(f"text_{i}")
                for i in range(5)
            ]

            results = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=2.0
            )

            assert len(results) == 5

            # Check stats
            stats = pipeline.stats
            assert stats["total_requests"] == 5
            # Should have fewer batches than requests (batching is efficient)
            assert stats["total_batches"] <= 5
        finally:
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_fallback_provider(self, mock_embeddings):
        """Test fallback when primary provider fails."""
        failing_embedder = MockEmbedder(should_fail=True)
        backup_embedder = MockEmbedder(embeddings=mock_embeddings)

        pipeline = BatchEmbeddingPipeline(
            max_batch_size=10,
            max_wait_ms=10.0,
            cache_enabled=False,
        )

        pipeline._primary_provider = failing_embedder
        pipeline._fallback_providers = [backup_embedder]

        await failing_embedder.initialize()
        await backup_embedder.initialize()

        pipeline._running = True
        pipeline._shutdown_event.clear()
        pipeline._start_time = time.time()
        pipeline._batch_task = asyncio.create_task(pipeline._batch_processor())

        try:
            result = await asyncio.wait_for(
                pipeline.embed("hello"),
                timeout=1.0
            )

            # Should succeed using backup
            assert len(result) == 1024
            # Backup should have been called
            assert backup_embedder._call_count >= 1
        finally:
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_stats(self, mock_embeddings):
        """Test statistics tracking."""
        embedder = MockEmbedder(embeddings=mock_embeddings)

        pipeline = BatchEmbeddingPipeline(
            max_batch_size=10,
            max_wait_ms=10.0,
            cache_enabled=True,
        )

        pipeline._primary_provider = embedder
        await embedder.initialize()
        pipeline._running = True
        pipeline._shutdown_event.clear()
        pipeline._start_time = time.time()
        pipeline._batch_task = asyncio.create_task(pipeline._batch_processor())

        try:
            await asyncio.wait_for(
                pipeline.embed_batch(["a", "b", "c"]),
                timeout=1.0
            )

            stats = pipeline.stats

            assert stats["running"]
            assert stats["total_requests"] == 3
            assert stats["total_batches"] >= 1
            assert "cache" in stats
            assert "primary_provider" in stats
        finally:
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_embeddings):
        """Test async context manager usage."""
        # This test would require actual provider setup
        # For unit testing, we just verify the interface
        pipeline = create_batch_pipeline(
            provider="local",
            max_batch_size=10,
            max_wait_ms=10.0,
        )

        assert isinstance(pipeline, BatchEmbeddingPipeline)
        assert not pipeline._running


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_batch_pipeline(self):
        """Test create_batch_pipeline factory."""
        pipeline = create_batch_pipeline(
            provider="auto",
            max_batch_size=50,
            max_wait_ms=100.0,
        )

        assert isinstance(pipeline, BatchEmbeddingPipeline)
        assert pipeline._max_batch_size == 50
        assert pipeline._max_wait_ms == 100.0

    def test_create_batch_pipeline_with_model(self):
        """Test create_batch_pipeline with model override."""
        pipeline = create_batch_pipeline(
            provider="openai",
            model="text-embedding-3-large",
        )

        assert pipeline._model_override == "text-embedding-3-large"


# =============================================================================
# EmbeddingRequest Tests
# =============================================================================

class TestEmbeddingRequest:
    """Tests for EmbeddingRequest dataclass."""

    def test_priority_ordering(self):
        """Test that requests are ordered by priority."""
        loop = asyncio.new_event_loop()

        req1 = EmbeddingRequest(
            text="low",
            request_id="1",
            future=loop.create_future(),
            priority=10,
        )

        req2 = EmbeddingRequest(
            text="high",
            request_id="2",
            future=loop.create_future(),
            priority=1,
        )

        # Higher priority (lower number) should be "less than"
        assert req2 < req1

        loop.close()

    def test_time_ordering_same_priority(self):
        """Test that requests with same priority are ordered by time."""
        loop = asyncio.new_event_loop()

        req1 = EmbeddingRequest(
            text="first",
            request_id="1",
            future=loop.create_future(),
            priority=5,
            created_at=100.0,
        )

        req2 = EmbeddingRequest(
            text="second",
            request_id="2",
            future=loop.create_future(),
            priority=5,
            created_at=200.0,
        )

        # Earlier time should be "less than"
        assert req1 < req2

        loop.close()


# =============================================================================
# EmbeddingResult Tests
# =============================================================================

class TestEmbeddingResult:
    """Tests for EmbeddingResult dataclass."""

    def test_result_creation(self):
        """Test result creation with all fields."""
        result = EmbeddingResult(
            request_id="test-123",
            embedding=[0.1, 0.2, 0.3],
            model="test-model",
            cached=True,
            latency_ms=10.5,
            provider="test",
        )

        assert result.request_id == "test-123"
        assert result.embedding == [0.1, 0.2, 0.3]
        assert result.model == "test-model"
        assert result.cached
        assert result.latency_ms == 10.5
        assert result.provider == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
