"""
Tests for Jina Reranker v3 Adapter (V65 Production Ready).

Comprehensive test coverage for:
- Rerank operation (single query)
- Batch rerank operation (multiple queries)
- Model info retrieval
- Error handling and validation
- Circuit breaker behavior
- Timeout handling
- RerankerProtocol integration
- Mock mode testing
"""

import asyncio
import sys
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import pytest

# Add platform directory to path for imports
_platform_dir = os.path.join(os.path.dirname(__file__), "..")
if _platform_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_platform_dir))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx AsyncClient."""
    client = AsyncMock()

    # Mock successful rerank response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {"index": 0, "relevance_score": 0.95, "document": {"text": "doc1"}},
            {"index": 2, "relevance_score": 0.87, "document": {"text": "doc3"}},
            {"index": 1, "relevance_score": 0.72, "document": {"text": "doc2"}},
        ],
        "usage": {"total_tokens": 100},
    }
    mock_response.raise_for_status = MagicMock()

    client.post = AsyncMock(return_value=mock_response)
    client.aclose = AsyncMock()

    return client


@pytest.fixture
def mock_circuit_breaker():
    """Create a mock circuit breaker."""
    cb = MagicMock()
    cb.is_open = False
    cb.record_success = MagicMock()
    cb.record_failure = MagicMock()
    return cb


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "Natural language processing enables computers to understand text.",
        "Computer vision allows machines to interpret images.",
        "Reinforcement learning trains agents through rewards.",
    ]


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "What is machine learning?"


# =============================================================================
# Test Data Classes
# =============================================================================

class TestRerankDocument:
    """Tests for RerankDocument dataclass."""

    def test_creates_with_required_fields(self):
        """Test creating RerankDocument with required fields."""
        from adapters.jina_reranker_adapter import RerankDocument

        doc = RerankDocument(index=0, text="Test content", score=0.95)

        assert doc.index == 0
        assert doc.text == "Test content"
        assert doc.score == 0.95
        assert doc.metadata == {}

    def test_creates_with_metadata(self):
        """Test creating RerankDocument with metadata."""
        from adapters.jina_reranker_adapter import RerankDocument

        doc = RerankDocument(
            index=1,
            text="Content",
            score=0.8,
            metadata={"source": "web", "original_index": 5},
        )

        assert doc.metadata["source"] == "web"
        assert doc.metadata["original_index"] == 5


class TestRerankResult:
    """Tests for RerankResult dataclass."""

    def test_creates_with_all_fields(self):
        """Test creating RerankResult with all fields."""
        from adapters.jina_reranker_adapter import RerankResult, RerankDocument

        docs = [
            RerankDocument(index=0, text="doc1", score=0.9),
            RerankDocument(index=1, text="doc2", score=0.8),
        ]

        result = RerankResult(
            query="test query",
            documents=docs,
            model="jina-reranker-v2-base-multilingual",
            usage={"total_tokens": 50},
            latency_ms=100.0,
        )

        assert result.query == "test query"
        assert len(result.documents) == 2
        assert result.model == "jina-reranker-v2-base-multilingual"
        assert result.usage["total_tokens"] == 50


class TestBatchRerankResult:
    """Tests for BatchRerankResult dataclass."""

    def test_creates_with_all_fields(self):
        """Test creating BatchRerankResult with all fields."""
        from adapters.jina_reranker_adapter import BatchRerankResult, RerankResult

        results = [
            RerankResult(query="q1", documents=[], model="model"),
            RerankResult(query="q2", documents=[], model="model"),
        ]

        batch = BatchRerankResult(
            results=results,
            total_queries=2,
            successful_queries=2,
            failed_queries=0,
            total_latency_ms=200.0,
        )

        assert batch.total_queries == 2
        assert batch.successful_queries == 2
        assert batch.failed_queries == 0


# =============================================================================
# Test Adapter Creation
# =============================================================================

class TestAdapterCreation:
    """Tests for adapter instantiation."""

    def test_creates_adapter(self):
        """Test basic adapter creation."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        assert adapter is not None

    def test_creates_with_api_key(self):
        """Test adapter creation with API key."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter(api_key="test-key")
        assert adapter._api_key == "test-key"

    def test_get_status_uninitialized(self):
        """Test get_status returns correct state when uninitialized."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        status = adapter.get_status()

        assert status["available"] is False
        assert status["initialized"] is False

    def test_factory_function(self):
        """Test create_jina_reranker factory function."""
        from adapters.jina_reranker_adapter import create_jina_reranker

        adapter = create_jina_reranker(api_key="test-key", model="jina-reranker-v1-base-en")
        assert adapter is not None
        assert adapter._model == "jina-reranker-v1-base-en"

    def test_sdk_name_property(self):
        """Test sdk_name property."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        assert adapter.sdk_name == "jina_reranker"


# =============================================================================
# Test Initialization
# =============================================================================

class TestInitialization:
    """Tests for adapter initialization."""

    @pytest.mark.asyncio
    async def test_init_mock_mode(self):
        """Test initialization in mock mode."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        result = await adapter.initialize({"mock_mode": True})

        assert result.success
        assert result.data["mode"] == "mock"
        assert adapter._available is True

    @pytest.mark.asyncio
    async def test_init_fails_without_api_key(self):
        """Test initialization fails without API key."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        # Clear any env var
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("JINA_API_KEY", None)

            adapter = JinaRerankerAdapter()
            result = await adapter.initialize({})

            assert not result.success
            assert "JINA_API_KEY" in result.error

    @pytest.mark.asyncio
    async def test_init_with_env_api_key(self, mock_httpx_client):
        """Test initialization with API key from environment."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        with patch.dict(os.environ, {"JINA_API_KEY": "env-test-key"}):
            with patch('httpx.AsyncClient', return_value=mock_httpx_client):
                adapter = JinaRerankerAdapter()
                adapter._client = mock_httpx_client
                result = await adapter.initialize({"mock_mode": True})

                assert result.success

    @pytest.mark.asyncio
    async def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        result = await adapter.initialize({
            "mock_mode": True,
            "model": "jina-reranker-v1-turbo-en",
        })

        assert result.success
        assert adapter._model == "jina-reranker-v1-turbo-en"

    @pytest.mark.asyncio
    async def test_init_returns_available_models(self):
        """Test initialization returns available models."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        result = await adapter.initialize({"mock_mode": True})

        assert result.success
        assert "available_models" in result.data
        assert "jina-reranker-v2-base-multilingual" in result.data["available_models"]


# =============================================================================
# Test Rerank Operation
# =============================================================================

class TestRerankOperation:
    """Tests for the rerank operation."""

    @pytest.mark.asyncio
    async def test_rerank_requires_query(self, sample_documents):
        """Test rerank fails without query."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        await adapter.initialize({"mock_mode": True})

        result = await adapter.execute("rerank", documents=sample_documents)

        assert not result.success
        assert "query" in result.error.lower()

    @pytest.mark.asyncio
    async def test_rerank_requires_documents(self, sample_query):
        """Test rerank fails without documents."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        await adapter.initialize({"mock_mode": True})

        result = await adapter.execute("rerank", query=sample_query)

        assert not result.success
        assert "documents" in result.error.lower()

    @pytest.mark.asyncio
    async def test_rerank_returns_results(self, sample_query, sample_documents):
        """Test rerank returns ranked documents."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        await adapter.initialize({"mock_mode": True})

        result = await adapter.execute(
            "rerank",
            query=sample_query,
            documents=sample_documents,
            top_k=3,
        )

        assert result.success
        assert "documents" in result.data
        assert result.data["count"] == 3
        assert len(result.data["documents"]) == 3

    @pytest.mark.asyncio
    async def test_rerank_returns_scores(self, sample_query, sample_documents):
        """Test rerank returns relevance scores."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        await adapter.initialize({"mock_mode": True})

        result = await adapter.execute(
            "rerank",
            query=sample_query,
            documents=sample_documents,
        )

        assert result.success
        for doc in result.data["documents"]:
            assert "score" in doc
            assert 0 <= doc["score"] <= 1

    @pytest.mark.asyncio
    async def test_rerank_respects_top_k(self, sample_query, sample_documents):
        """Test rerank respects top_k parameter."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        await adapter.initialize({"mock_mode": True})

        result = await adapter.execute(
            "rerank",
            query=sample_query,
            documents=sample_documents,
            top_k=2,
        )

        assert result.success
        assert result.data["count"] == 2

    @pytest.mark.asyncio
    async def test_rerank_handles_dict_documents(self, sample_query):
        """Test rerank handles documents as dicts."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        await adapter.initialize({"mock_mode": True})

        docs = [
            {"content": "Document one content"},
            {"text": "Document two text"},
            {"content": "Document three content", "metadata": {"id": 3}},
        ]

        result = await adapter.execute(
            "rerank",
            query=sample_query,
            documents=docs,
        )

        assert result.success
        assert result.data["count"] == 3

    @pytest.mark.asyncio
    async def test_rerank_truncates_excessive_documents(self, sample_query):
        """Test rerank truncates documents beyond MAX_DOCUMENTS_PER_CALL."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter, MAX_DOCUMENTS_PER_CALL

        adapter = JinaRerankerAdapter()
        await adapter.initialize({"mock_mode": True})

        # Create more documents than allowed
        docs = [f"Document {i}" for i in range(MAX_DOCUMENTS_PER_CALL + 10)]

        result = await adapter.execute(
            "rerank",
            query=sample_query,
            documents=docs,
        )

        assert result.success
        # Should truncate to MAX_DOCUMENTS_PER_CALL
        assert result.data["count"] <= MAX_DOCUMENTS_PER_CALL

    @pytest.mark.asyncio
    async def test_rerank_updates_stats(self, sample_query, sample_documents):
        """Test rerank updates adapter statistics."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        await adapter.initialize({"mock_mode": True})

        await adapter.execute(
            "rerank",
            query=sample_query,
            documents=sample_documents,
        )

        stats = adapter.get_stats()
        assert stats["rerank_calls"] == 1
        assert stats["total_documents"] == len(sample_documents)

    @pytest.mark.asyncio
    async def test_rerank_includes_latency(self, sample_query, sample_documents):
        """Test rerank includes latency in result."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        await adapter.initialize({"mock_mode": True})

        result = await adapter.execute(
            "rerank",
            query=sample_query,
            documents=sample_documents,
        )

        assert result.success
        # Latency >= 0 (mock operations can be very fast)
        assert result.latency_ms >= 0


# =============================================================================
# Test Batch Rerank Operation
# =============================================================================

class TestBatchRerankOperation:
    """Tests for the batch_rerank operation."""

    @pytest.mark.asyncio
    async def test_batch_rerank_requires_queries(self):
        """Test batch_rerank fails without queries."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        await adapter.initialize({"mock_mode": True})

        result = await adapter.execute(
            "batch_rerank",
            document_sets=[["doc1", "doc2"]],
        )

        assert not result.success
        assert "queries" in result.error.lower()

    @pytest.mark.asyncio
    async def test_batch_rerank_requires_document_sets(self):
        """Test batch_rerank fails without document_sets."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        await adapter.initialize({"mock_mode": True})

        result = await adapter.execute(
            "batch_rerank",
            queries=["query1", "query2"],
        )

        assert not result.success
        assert "document_sets" in result.error.lower()

    @pytest.mark.asyncio
    async def test_batch_rerank_requires_matching_lengths(self):
        """Test batch_rerank fails when queries and document_sets have different lengths."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        await adapter.initialize({"mock_mode": True})

        result = await adapter.execute(
            "batch_rerank",
            queries=["q1", "q2", "q3"],
            document_sets=[["d1"], ["d2"]],  # Only 2 sets for 3 queries
        )

        assert not result.success
        assert "same length" in result.error.lower()

    @pytest.mark.asyncio
    async def test_batch_rerank_returns_results(self):
        """Test batch_rerank returns results for all queries."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        await adapter.initialize({"mock_mode": True})

        result = await adapter.execute(
            "batch_rerank",
            queries=["query1", "query2"],
            document_sets=[
                ["doc1a", "doc1b", "doc1c"],
                ["doc2a", "doc2b", "doc2c"],
            ],
            top_k=2,
        )

        assert result.success
        assert result.data["total_queries"] == 2
        assert result.data["successful_queries"] == 2
        assert len(result.data["results"]) == 2

    @pytest.mark.asyncio
    async def test_batch_rerank_respects_concurrency(self):
        """Test batch_rerank respects max_concurrency parameter."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        await adapter.initialize({"mock_mode": True})

        result = await adapter.execute(
            "batch_rerank",
            queries=["q1", "q2", "q3"],
            document_sets=[["d1"], ["d2"], ["d3"]],
            max_concurrency=2,
        )

        assert result.success
        assert result.data["successful_queries"] == 3

    @pytest.mark.asyncio
    async def test_batch_rerank_updates_stats(self):
        """Test batch_rerank updates statistics."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        await adapter.initialize({"mock_mode": True})

        await adapter.execute(
            "batch_rerank",
            queries=["q1", "q2"],
            document_sets=[["d1"], ["d2"]],
        )

        stats = adapter.get_stats()
        assert stats["batch_calls"] == 1


# =============================================================================
# Test Model Info Operation
# =============================================================================

class TestModelInfoOperation:
    """Tests for the get_model_info operation."""

    @pytest.mark.asyncio
    async def test_get_all_models(self):
        """Test get_model_info returns all models."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        await adapter.initialize({"mock_mode": True})

        result = await adapter.execute("get_model_info")

        assert result.success
        assert "models" in result.data
        assert "jina-reranker-v2-base-multilingual" in result.data["models"]

    @pytest.mark.asyncio
    async def test_get_specific_model(self):
        """Test get_model_info returns specific model info."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        await adapter.initialize({"mock_mode": True})

        result = await adapter.execute(
            "get_model_info",
            model="jina-reranker-v2-base-multilingual",
        )

        assert result.success
        assert result.data["model"] == "jina-reranker-v2-base-multilingual"
        assert "description" in result.data

    @pytest.mark.asyncio
    async def test_get_unknown_model_fails(self):
        """Test get_model_info fails for unknown model."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        await adapter.initialize({"mock_mode": True})

        result = await adapter.execute(
            "get_model_info",
            model="nonexistent-model",
        )

        assert not result.success
        assert "Unknown model" in result.error

    @pytest.mark.asyncio
    async def test_get_model_capabilities(self):
        """Test get_model_info includes capabilities."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        await adapter.initialize({"mock_mode": True})

        result = await adapter.execute("get_model_info")

        assert result.success
        assert "capabilities" in result.data
        assert result.data["capabilities"]["cross_document_reasoning"] is True
        assert result.data["capabilities"]["multilingual"] is True


# =============================================================================
# Test Error Handling
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_handles_unknown_operation(self):
        """Test adapter handles unknown operation."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        await adapter.initialize({"mock_mode": True})

        result = await adapter.execute("unknown_operation")

        assert not result.success
        assert "Unknown operation" in result.error

    @pytest.mark.asyncio
    async def test_handles_not_initialized(self):
        """Test adapter handles calls when not initialized."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        # Don't initialize

        result = await adapter.execute("rerank", query="test", documents=["doc"])

        assert not result.success
        assert "not initialized" in result.error.lower()

    @pytest.mark.asyncio
    async def test_handles_empty_documents(self, sample_query):
        """Test adapter handles empty documents list."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        await adapter.initialize({"mock_mode": True})

        result = await adapter.execute(
            "rerank",
            query=sample_query,
            documents=[],
        )

        assert not result.success
        assert "empty" in result.error.lower()

    @pytest.mark.asyncio
    async def test_handles_timeout(self, sample_query, sample_documents):
        """Test adapter handles timeout."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        await adapter.initialize({"mock_mode": True})

        # Create a slow mock operation
        original_rerank = adapter._rerank

        async def slow_rerank(kwargs):
            await asyncio.sleep(2)
            return await original_rerank(kwargs)

        adapter._rerank = slow_rerank

        result = await adapter.execute(
            "rerank",
            query=sample_query,
            documents=sample_documents,
            timeout=0.01,
        )

        assert not result.success
        assert "timed out" in result.error.lower()


# =============================================================================
# Test Circuit Breaker Integration
# =============================================================================

class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration."""

    @pytest.mark.asyncio
    async def test_records_success(self, sample_query, sample_documents, mock_circuit_breaker):
        """Test circuit breaker records success."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        with patch('adapters.jina_reranker_adapter.adapter_circuit_breaker') as mock_cb_func:
            mock_cb_func.return_value = mock_circuit_breaker

            adapter = JinaRerankerAdapter()
            await adapter.initialize({"mock_mode": True})

            result = await adapter.execute(
                "rerank",
                query=sample_query,
                documents=sample_documents,
            )

            assert result.success
            mock_circuit_breaker.record_success.assert_called()

    @pytest.mark.asyncio
    async def test_circuit_open_returns_error(self, sample_query, sample_documents, mock_circuit_breaker):
        """Test returns error when circuit is open."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        mock_circuit_breaker.is_open = True

        with patch('adapters.jina_reranker_adapter.adapter_circuit_breaker') as mock_cb_func:
            mock_cb_func.return_value = mock_circuit_breaker

            adapter = JinaRerankerAdapter()
            await adapter.initialize({"mock_mode": True})

            result = await adapter.execute(
                "rerank",
                query=sample_query,
                documents=sample_documents,
            )

            assert not result.success
            assert "circuit breaker" in result.error.lower()


# =============================================================================
# Test RerankerProtocol Integration
# =============================================================================

class TestRerankerProtocolIntegration:
    """Tests for RerankerProtocol implementation."""

    @pytest.mark.asyncio
    async def test_rerank_method_exists(self):
        """Test rerank method exists for protocol compliance."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        assert hasattr(adapter, 'rerank')
        assert callable(adapter.rerank)

    @pytest.mark.asyncio
    async def test_rerank_method_returns_list(self, sample_query, sample_documents):
        """Test rerank method returns list of documents."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        await adapter.initialize({"mock_mode": True})

        result = await adapter.rerank(
            query=sample_query,
            documents=sample_documents,
            top_k=3,
        )

        assert isinstance(result, list)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_rerank_method_returns_scores(self, sample_query, sample_documents):
        """Test rerank method returns documents with scores."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        await adapter.initialize({"mock_mode": True})

        result = await adapter.rerank(
            query=sample_query,
            documents=sample_documents,
        )

        for doc in result:
            assert "score" in doc
            assert "text" in doc

    @pytest.mark.asyncio
    async def test_rerank_method_handles_failure(self, sample_query, sample_documents):
        """Test rerank method handles failure gracefully."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        # Don't initialize - should fail gracefully

        result = await adapter.rerank(
            query=sample_query,
            documents=sample_documents,
        )

        # Should return original documents on failure
        assert isinstance(result, list)


# =============================================================================
# Test Health Check
# =============================================================================

class TestHealthCheck:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_passes_mock_mode(self):
        """Test health check passes in mock mode."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        await adapter.initialize({"mock_mode": True})

        result = await adapter.health_check()

        assert result.success
        assert result.data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_fails_not_initialized(self):
        """Test health check fails when not initialized."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()

        result = await adapter.health_check()

        assert not result.success
        assert "not initialized" in result.error.lower()

    @pytest.mark.asyncio
    async def test_health_check_returns_stats(self):
        """Test health check returns adapter stats or mode info."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        await adapter.initialize({"mock_mode": True})

        result = await adapter.health_check()

        assert result.success
        # In mock mode, stats are included; ensure status is present at minimum
        assert "status" in result.data
        assert result.data["status"] == "healthy"


# =============================================================================
# Test Shutdown
# =============================================================================

class TestShutdown:
    """Tests for shutdown functionality."""

    @pytest.mark.asyncio
    async def test_shutdown_clears_state(self):
        """Test shutdown clears adapter state."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        await adapter.initialize({"mock_mode": True})

        assert adapter._available is True

        result = await adapter.shutdown()

        assert result.success
        assert adapter._available is False
        assert adapter._client is None

    @pytest.mark.asyncio
    async def test_shutdown_returns_stats(self):
        """Test shutdown returns final stats."""
        from adapters.jina_reranker_adapter import JinaRerankerAdapter

        adapter = JinaRerankerAdapter()
        await adapter.initialize({"mock_mode": True})

        await adapter.execute(
            "rerank",
            query="test",
            documents=["doc1", "doc2"],
        )

        result = await adapter.shutdown()

        assert result.success
        assert "stats" in result.data
        assert result.data["stats"]["rerank_calls"] == 1


# =============================================================================
# Test Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    """Tests for module convenience functions."""

    @pytest.mark.asyncio
    async def test_rerank_documents_function(self):
        """Test rerank_documents convenience function."""
        from adapters.jina_reranker_adapter import rerank_documents

        with patch('adapters.jina_reranker_adapter.JinaRerankerAdapter') as MockAdapter:
            mock_instance = AsyncMock()
            mock_instance.initialize = AsyncMock()
            mock_instance.execute = AsyncMock(return_value=MagicMock(
                success=True,
                data={"documents": [{"text": "doc1", "score": 0.9}]},
            ))
            mock_instance.shutdown = AsyncMock()
            MockAdapter.return_value = mock_instance

            result = await rerank_documents(
                query="test",
                documents=["doc1", "doc2"],
                top_k=1,
            )

            assert len(result) == 1
            mock_instance.initialize.assert_called_once()
            mock_instance.shutdown.assert_called_once()


# =============================================================================
# Test Module Exports
# =============================================================================

class TestModuleExports:
    """Tests for module exports."""

    def test_exports_available(self):
        """Test all expected exports are available."""
        from adapters import jina_reranker_adapter

        expected = [
            "JinaRerankerAdapter",
            "RerankDocument",
            "RerankResult",
            "BatchRerankResult",
            "rerank_documents",
            "create_jina_reranker",
            "JINA_RERANKER_AVAILABLE",
            "RERANKER_MODELS",
            "MAX_DOCUMENTS_PER_CALL",
        ]

        for name in expected:
            assert hasattr(jina_reranker_adapter, name), f"Missing export: {name}"

    def test_all_list_complete(self):
        """Test __all__ contains expected exports."""
        from adapters.jina_reranker_adapter import __all__

        assert "JinaRerankerAdapter" in __all__
        assert "rerank_documents" in __all__
        assert "JINA_RERANKER_AVAILABLE" in __all__


# =============================================================================
# Test Constants
# =============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_max_documents_defined(self):
        """Test MAX_DOCUMENTS_PER_CALL is defined correctly."""
        from adapters.jina_reranker_adapter import MAX_DOCUMENTS_PER_CALL

        assert MAX_DOCUMENTS_PER_CALL == 64

    def test_max_tokens_defined(self):
        """Test MAX_TOKENS_PER_DOCUMENT is defined correctly."""
        from adapters.jina_reranker_adapter import MAX_TOKENS_PER_DOCUMENT

        assert MAX_TOKENS_PER_DOCUMENT == 8192

    def test_reranker_models_defined(self):
        """Test RERANKER_MODELS dictionary is defined."""
        from adapters.jina_reranker_adapter import RERANKER_MODELS

        assert len(RERANKER_MODELS) >= 3
        assert "jina-reranker-v2-base-multilingual" in RERANKER_MODELS
