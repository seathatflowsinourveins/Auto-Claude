"""
Tests for ColBERT/RAGatouille Retriever Integration

Tests cover:
- ColBERTRetriever initialization and configuration
- Retrieval with ColBERT late interaction
- Reranking functionality
- Fallback to dense retrieval
- Integration with RRF fusion
- Pipeline integration

Version: V1.0.0 (2026-02-04)
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict, List


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def mock_ragatouille():
    """Mock RAGatouille library for testing without installation."""
    with patch.dict("sys.modules", {"ragatouille": MagicMock()}):
        mock_model = MagicMock()
        mock_model.search.return_value = [
            {"content": "Document 1 content about RAG", "score": 0.95, "document_id": "1"},
            {"content": "Document 2 about retrieval", "score": 0.87, "document_id": "2"},
            {"content": "Document 3 on ColBERT", "score": 0.82, "document_id": "3"},
        ]
        mock_model.rerank.return_value = [
            {"content": "Reranked document 1", "score": 0.92, "document_id": "1"},
            {"content": "Reranked document 2", "score": 0.85, "document_id": "2"},
        ]
        yield mock_model


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {"content": "RAG combines retrieval with generation for better answers.", "id": "1"},
        {"content": "ColBERT uses late interaction for token-level matching.", "id": "2"},
        {"content": "Dense retrieval encodes queries and documents into vectors.", "id": "3"},
        {"content": "Reranking improves initial retrieval results.", "id": "4"},
        {"content": "RRF fusion combines multiple ranked lists.", "id": "5"},
    ]


@pytest.fixture
def colbert_config():
    """Standard ColBERT configuration for tests."""
    from core.rag.colbert_retriever import ColBERTConfig
    return ColBERTConfig(
        index_path=".ragatouille",
        model_name="colbert-ir/colbertv2.0",
        index_name="test_index",
        fallback_enabled=True,
    )


# =============================================================================
# UNIT TESTS - ColBERTConfig
# =============================================================================

class TestColBERTConfig:
    """Tests for ColBERTConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from core.rag.colbert_retriever import ColBERTConfig

        config = ColBERTConfig()

        assert config.index_path == ".ragatouille"
        assert config.model_name == "colbert-ir/colbertv2.0"
        assert config.index_name == "default"
        assert config.n_probe == 10
        assert config.doc_maxlen == 300
        assert config.query_maxlen == 32
        assert config.use_gpu is False
        assert config.fallback_enabled is True

    def test_custom_config(self):
        """Test custom configuration values."""
        from core.rag.colbert_retriever import ColBERTConfig

        config = ColBERTConfig(
            index_path="/custom/path",
            model_name="custom-model",
            index_name="my_index",
            use_gpu=True,
        )

        assert config.index_path == "/custom/path"
        assert config.model_name == "custom-model"
        assert config.index_name == "my_index"
        assert config.use_gpu is True


# =============================================================================
# UNIT TESTS - ColBERTDocument
# =============================================================================

class TestColBERTDocument:
    """Tests for ColBERTDocument dataclass."""

    def test_document_creation(self):
        """Test creating a ColBERTDocument."""
        from core.rag.colbert_retriever import ColBERTDocument

        doc = ColBERTDocument(
            id="test-1",
            content="Test content",
            score=0.95,
            rank=1,
            metadata={"source": "test"},
        )

        assert doc.id == "test-1"
        assert doc.content == "Test content"
        assert doc.score == 0.95
        assert doc.rank == 1
        assert doc.metadata == {"source": "test"}
        assert doc.token_scores is None

    def test_document_with_token_scores(self):
        """Test document with token-level scores."""
        from core.rag.colbert_retriever import ColBERTDocument

        doc = ColBERTDocument(
            id="test-2",
            content="Token level test",
            score=0.88,
            rank=2,
            token_scores=[0.9, 0.85, 0.92],
        )

        assert doc.token_scores == [0.9, 0.85, 0.92]


# =============================================================================
# UNIT TESTS - ColBERTRetriever
# =============================================================================

class TestColBERTRetriever:
    """Tests for ColBERTRetriever class."""

    def test_retriever_initialization(self, colbert_config):
        """Test retriever initialization with config."""
        from core.rag.colbert_retriever import ColBERTRetriever

        retriever = ColBERTRetriever(config=colbert_config)

        assert retriever.config == colbert_config
        assert retriever.name == "colbert"
        assert retriever._model is None
        assert retriever._index_loaded is False

    def test_retriever_name_property(self):
        """Test retriever name for pipeline identification."""
        from core.rag.colbert_retriever import ColBERTRetriever

        retriever = ColBERTRetriever()
        assert retriever.name == "colbert"

    def test_availability_check_without_ragatouille(self):
        """Test availability check when RAGatouille not installed."""
        from core.rag.colbert_retriever import ColBERTRetriever

        # Reset class-level cache
        ColBERTRetriever._ragatouille_available = None

        retriever = ColBERTRetriever()
        # Will be False since RAGatouille is not installed in test env
        # Just verify it returns a boolean
        result = retriever.is_available
        assert isinstance(result, bool)

    def test_diagnostics(self, colbert_config):
        """Test diagnostics output."""
        from core.rag.colbert_retriever import ColBERTRetriever

        retriever = ColBERTRetriever(config=colbert_config)
        diag = retriever.get_diagnostics()

        assert diag["name"] == "colbert"
        assert diag["model_name"] == "colbert-ir/colbertv2.0"
        assert diag["index_name"] == "test_index"
        assert diag["fallback_enabled"] is True
        assert "ragatouille_available" in diag
        assert "model_loaded" in diag
        assert "index_loaded" in diag


# =============================================================================
# UNIT TESTS - Fallback Retrieval
# =============================================================================

class TestFallbackRetrieval:
    """Tests for fallback dense retrieval."""

    def test_set_fallback_documents(self, colbert_config):
        """Test setting fallback documents and embeddings."""
        from core.rag.colbert_retriever import ColBERTRetriever

        retriever = ColBERTRetriever(config=colbert_config)

        documents = {
            "doc1": "Content of document 1",
            "doc2": "Content of document 2",
        }
        embeddings = {
            "doc1": [0.1, 0.2, 0.3],
            "doc2": [0.4, 0.5, 0.6],
        }

        retriever.set_fallback_documents(documents, embeddings)

        assert retriever._fallback_documents == documents
        assert retriever._fallback_index == embeddings

    def test_cosine_similarity(self, colbert_config):
        """Test cosine similarity calculation."""
        from core.rag.colbert_retriever import ColBERTRetriever

        retriever = ColBERTRetriever(config=colbert_config)

        # Identical vectors
        sim1 = retriever._cosine_similarity([1.0, 0.0], [1.0, 0.0])
        assert abs(sim1 - 1.0) < 0.001

        # Orthogonal vectors
        sim2 = retriever._cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(sim2 - 0.0) < 0.001

        # 45 degree angle
        sim3 = retriever._cosine_similarity([1.0, 0.0], [1.0, 1.0])
        assert 0.7 < sim3 < 0.72  # Should be ~0.707

    def test_passthrough_rerank(self, colbert_config, sample_documents):
        """Test passthrough reranking when ColBERT unavailable."""
        from core.rag.colbert_retriever import ColBERTRetriever

        retriever = ColBERTRetriever(config=colbert_config)

        results = retriever._passthrough_rerank(sample_documents, top_k=3)

        assert len(results) == 3
        assert results[0].rank == 1
        assert results[1].rank == 2
        assert results[2].rank == 3
        assert results[0].score > results[1].score  # Reciprocal rank ordering
        assert all(r.metadata.get("passthrough") is True for r in results)


# =============================================================================
# UNIT TESTS - ColBERTReranker
# =============================================================================

class TestColBERTReranker:
    """Tests for standalone ColBERTReranker."""

    def test_reranker_initialization(self):
        """Test reranker initialization."""
        from core.rag.colbert_retriever import ColBERTReranker

        reranker = ColBERTReranker()
        assert reranker.model_name == "colbert-ir/colbertv2.0"

    def test_reranker_custom_model(self):
        """Test reranker with custom model."""
        from core.rag.colbert_retriever import ColBERTReranker

        reranker = ColBERTReranker(model_name="custom-colbert-model")
        assert reranker.model_name == "custom-colbert-model"


# =============================================================================
# UNIT TESTS - Factory Functions
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_colbert_retriever(self):
        """Test create_colbert_retriever factory."""
        from core.rag.colbert_retriever import create_colbert_retriever

        retriever = create_colbert_retriever(
            model_name="test-model",
            fallback_enabled=False,
        )

        assert retriever.config.model_name == "test-model"
        assert retriever.config.fallback_enabled is False

    def test_create_colbert_reranker(self):
        """Test create_colbert_reranker factory."""
        from core.rag.colbert_retriever import create_colbert_reranker

        reranker = create_colbert_reranker(model_name="test-reranker")
        assert reranker.model_name == "test-reranker"


# =============================================================================
# INTEGRATION TESTS - Pipeline
# =============================================================================

class TestPipelineIntegration:
    """Tests for RAGPipeline integration."""

    def test_colbert_strategy_type_exists(self):
        """Test COLBERT strategy type is available."""
        from core.rag.pipeline import StrategyType

        assert hasattr(StrategyType, "COLBERT")
        assert StrategyType.COLBERT.value == "colbert"

    def test_create_pipeline_with_colbert(self):
        """Test creating pipeline with ColBERT retriever."""
        from core.rag.pipeline import create_pipeline
        from core.rag.colbert_retriever import ColBERTRetriever

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value="Test response")

        # Create ColBERT retriever
        colbert = ColBERTRetriever()

        # Create pipeline with ColBERT
        pipeline = create_pipeline(
            llm=mock_llm,
            retrievers=[colbert],
            colbert_retriever=colbert,
        )

        assert pipeline is not None
        assert "colbert" in pipeline.rag_implementations
        assert colbert in pipeline.retrievers


# =============================================================================
# INTEGRATION TESTS - RRF Fusion
# =============================================================================

class TestRRFFusionIntegration:
    """Tests for RRF fusion compatibility."""

    @pytest.mark.asyncio
    async def test_colbert_results_compatible_with_rrf(self, colbert_config):
        """Test ColBERT results can be used with RRF fusion."""
        from core.rag.colbert_retriever import ColBERTRetriever, ColBERTDocument
        from core.rag.pipeline import RRFFusion, RetrievedDocument

        # Create mock ColBERT results
        colbert_results = [
            ColBERTDocument(id="1", content="ColBERT result 1", score=0.95, rank=1),
            ColBERTDocument(id="2", content="ColBERT result 2", score=0.87, rank=2),
        ]

        # Convert to RetrievedDocument for RRF
        retrieved_docs = [
            RetrievedDocument(
                content=doc.content,
                score=doc.score,
                source="colbert",
                metadata=doc.metadata,
            )
            for doc in colbert_results
        ]

        # Create another result list (mock dense retrieval)
        dense_results = [
            RetrievedDocument(content="Dense result 1", score=0.88, source="dense"),
            RetrievedDocument(content="Dense result 2", score=0.82, source="dense"),
        ]

        # Apply RRF fusion
        fusion = RRFFusion(k=60)
        fused = fusion.fuse([retrieved_docs, dense_results])

        assert len(fused) > 0
        # Check that ColBERT results are included
        colbert_contents = {doc.content for doc in colbert_results}
        fused_contents = {doc.content for doc in fused}
        assert colbert_contents & fused_contents  # At least one match


# =============================================================================
# ASYNC TESTS
# =============================================================================

class TestAsyncOperations:
    """Tests for async operations."""

    @pytest.mark.asyncio
    async def test_retrieve_returns_list(self, colbert_config):
        """Test retrieve returns a list of documents."""
        from core.rag.colbert_retriever import ColBERTRetriever

        retriever = ColBERTRetriever(config=colbert_config)

        # Without loaded index, should return empty or fallback
        results = await retriever.retrieve("test query", top_k=5)

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_rerank_returns_list(self, colbert_config, sample_documents):
        """Test rerank returns a list of ColBERTDocument."""
        from core.rag.colbert_retriever import ColBERTRetriever

        retriever = ColBERTRetriever(config=colbert_config)

        # Rerank should use passthrough when ColBERT unavailable
        results = await retriever.rerank("test query", sample_documents, top_k=3)

        assert isinstance(results, list)
        assert len(results) <= 3


# =============================================================================
# MODULE EXPORTS TEST
# =============================================================================

class TestModuleExports:
    """Test module exports are accessible."""

    def test_exports_from_colbert_module(self):
        """Test all expected exports from colbert_retriever module."""
        from core.rag import colbert_retriever

        expected_exports = [
            "ColBERTConfig",
            "ColBERTDocument",
            "ColBERTResult",
            "ColBERTRetriever",
            "ColBERTReranker",
            "create_colbert_retriever",
            "create_colbert_reranker",
        ]

        for export in expected_exports:
            assert hasattr(colbert_retriever, export), f"Missing export: {export}"

    def test_exports_from_rag_init(self):
        """Test exports accessible from core.rag."""
        from core.rag import (
            ColBERTConfig,
            ColBERTDocument,
            ColBERTRetriever,
            ColBERTReranker,
            create_colbert_retriever,
            create_colbert_reranker,
        )

        # Just verify imports work
        assert ColBERTConfig is not None
        assert ColBERTDocument is not None
        assert ColBERTRetriever is not None
        assert ColBERTReranker is not None
        assert create_colbert_retriever is not None
        assert create_colbert_reranker is not None


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
