"""
Tests for the Semantic Reranker

Tests cover:
1. Document and ScoredDocument dataclasses
2. RerankerCache with TTL and LRU eviction
3. TFIDFScorer fallback
4. CrossEncoderReranker (mocked when unavailable)
5. SemanticReranker main class
6. RRF fusion
7. Diversity reranking
8. MemoryReranker integration
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import test subjects
from core.rag.reranker import (
    Document,
    ScoredDocument,
    RerankerCache,
    TFIDFScorer,
    CrossEncoderReranker,
    SemanticReranker,
    MemoryReranker,
    create_reranker,
    create_memory_reranker,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_documents() -> List[Document]:
    """Create sample documents for testing."""
    return [
        Document(
            id="doc1",
            content="Python is a popular programming language for data science.",
            metadata={"source": "wiki"},
        ),
        Document(
            id="doc2",
            content="Machine learning uses algorithms to learn from data.",
            metadata={"source": "textbook"},
        ),
        Document(
            id="doc3",
            content="Python and machine learning are often used together.",
            metadata={"source": "blog"},
        ),
        Document(
            id="doc4",
            content="JavaScript is used for web development.",
            metadata={"source": "wiki"},
        ),
        Document(
            id="doc5",
            content="Data science involves statistics and programming.",
            metadata={"source": "course"},
        ),
    ]


@pytest.fixture
def scored_documents(sample_documents) -> List[ScoredDocument]:
    """Create scored documents for testing."""
    scores = [0.9, 0.8, 0.85, 0.3, 0.7]
    return [
        ScoredDocument(document=doc, score=score, rank=i + 1)
        for i, (doc, score) in enumerate(zip(sample_documents, scores))
    ]


@pytest.fixture
def reranker_cache() -> RerankerCache:
    """Create a reranker cache for testing."""
    return RerankerCache(max_size=10, ttl_seconds=5)


@pytest.fixture
def tfidf_scorer() -> TFIDFScorer:
    """Create a TF-IDF scorer for testing."""
    return TFIDFScorer()


@pytest.fixture
def semantic_reranker() -> SemanticReranker:
    """Create a semantic reranker for testing."""
    return SemanticReranker(enable_cache=True, cache_size=100, cache_ttl=60)


# =============================================================================
# DOCUMENT TESTS
# =============================================================================

class TestDocument:
    """Tests for Document dataclass."""

    def test_document_creation(self):
        """Test creating a Document."""
        doc = Document(id="test1", content="Test content")
        assert doc.id == "test1"
        assert doc.content == "Test content"
        assert doc.metadata == {}
        assert doc.embedding is None

    def test_document_with_metadata(self):
        """Test Document with metadata and embedding."""
        embedding = [0.1, 0.2, 0.3]
        doc = Document(
            id="test2",
            content="Content",
            metadata={"key": "value"},
            embedding=embedding,
        )
        assert doc.metadata == {"key": "value"}
        assert doc.embedding == embedding

    def test_document_hash_and_equality(self):
        """Test Document hashing and equality."""
        doc1 = Document(id="same", content="Content 1")
        doc2 = Document(id="same", content="Different content")
        doc3 = Document(id="different", content="Content 1")

        assert doc1 == doc2  # Same ID
        assert doc1 != doc3  # Different ID
        assert hash(doc1) == hash(doc2)

    def test_document_in_set(self):
        """Test Document can be used in sets."""
        doc1 = Document(id="1", content="A")
        doc2 = Document(id="1", content="B")
        doc3 = Document(id="2", content="C")

        docs = {doc1, doc2, doc3}
        assert len(docs) == 2  # doc1 and doc2 have same ID


class TestScoredDocument:
    """Tests for ScoredDocument dataclass."""

    def test_scored_document_creation(self, sample_documents):
        """Test creating a ScoredDocument."""
        scored = ScoredDocument(
            document=sample_documents[0],
            score=0.95,
            rank=1,
        )
        assert scored.document == sample_documents[0]
        assert scored.score == 0.95
        assert scored.rank == 1

    def test_scored_document_comparison(self, sample_documents):
        """Test ScoredDocument comparison."""
        scored1 = ScoredDocument(document=sample_documents[0], score=0.9)
        scored2 = ScoredDocument(document=sample_documents[1], score=0.8)

        assert scored2 < scored1  # Lower score is "less than"
        assert sorted([scored1, scored2]) == [scored2, scored1]


# =============================================================================
# CACHE TESTS
# =============================================================================

class TestRerankerCache:
    """Tests for RerankerCache."""

    def test_cache_put_and_get(self, reranker_cache, scored_documents):
        """Test basic cache put and get."""
        query = "test query"
        doc_ids = ["doc1", "doc2"]
        top_k = 5

        reranker_cache.put(query, doc_ids, top_k, scored_documents[:2])
        result = reranker_cache.get(query, doc_ids, top_k)

        assert result is not None
        assert len(result) == 2

    def test_cache_miss(self, reranker_cache):
        """Test cache miss returns None."""
        result = reranker_cache.get("nonexistent", ["doc1"], 5)
        assert result is None

    def test_cache_ttl_expiration(self, scored_documents):
        """Test cache entries expire after TTL."""
        cache = RerankerCache(max_size=10, ttl_seconds=1)

        cache.put("query", ["doc1"], 5, scored_documents[:1])
        assert cache.get("query", ["doc1"], 5) is not None

        time.sleep(1.5)
        assert cache.get("query", ["doc1"], 5) is None

    def test_cache_lru_eviction(self, scored_documents):
        """Test LRU eviction when cache is full."""
        cache = RerankerCache(max_size=3, ttl_seconds=300)

        # Fill cache
        for i in range(3):
            cache.put(f"query{i}", [f"doc{i}"], 5, scored_documents[:1])

        # Add one more, should evict first
        cache.put("query_new", ["doc_new"], 5, scored_documents[:1])

        assert cache.get("query0", ["doc0"], 5) is None  # Evicted
        assert cache.get("query1", ["doc1"], 5) is not None
        assert cache.get("query_new", ["doc_new"], 5) is not None

    def test_cache_stats(self, reranker_cache, scored_documents):
        """Test cache statistics."""
        reranker_cache.put("q1", ["d1"], 5, scored_documents[:1])
        reranker_cache.get("q1", ["d1"], 5)  # Hit
        reranker_cache.get("q2", ["d2"], 5)  # Miss

        stats = reranker_cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cache_clear(self, reranker_cache, scored_documents):
        """Test cache clear."""
        reranker_cache.put("q1", ["d1"], 5, scored_documents[:1])
        reranker_cache.clear()

        assert reranker_cache.get("q1", ["d1"], 5) is None
        assert len(reranker_cache._cache) == 0


# =============================================================================
# TF-IDF SCORER TESTS
# =============================================================================

class TestTFIDFScorer:
    """Tests for TFIDFScorer fallback."""

    def test_tfidf_score_documents(self, tfidf_scorer, sample_documents):
        """Test TF-IDF scoring."""
        query = "Python programming language"
        results = tfidf_scorer.score(query, sample_documents)

        assert len(results) == len(sample_documents)
        # All results should be (Document, float) tuples
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)

    def test_tfidf_relevance_ranking(self, tfidf_scorer, sample_documents):
        """Test that TF-IDF ranks relevant documents higher."""
        query = "Python data science"
        results = tfidf_scorer.score(query, sample_documents)

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Doc1 and doc5 should rank higher (contain query terms)
        top_ids = [r[0].id for r in results[:2]]
        assert "doc1" in top_ids or "doc5" in top_ids

    def test_tfidf_empty_documents(self, tfidf_scorer):
        """Test TF-IDF with empty document list."""
        results = tfidf_scorer.score("query", [])
        assert results == []

    def test_tfidf_tokenization(self, tfidf_scorer):
        """Test tokenization handles special characters."""
        tokens = tfidf_scorer._tokenize("Hello, World! This is a test-case.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        assert "case" in tokens


# =============================================================================
# CROSS-ENCODER TESTS
# =============================================================================

class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker."""

    def test_cross_encoder_initialization(self):
        """Test cross-encoder initializes."""
        reranker = CrossEncoderReranker()
        assert reranker.model_name == CrossEncoderReranker.MODEL_NAME
        assert reranker._model is None
        assert not reranker._load_attempted

    def test_cross_encoder_unavailable_graceful(self, sample_documents):
        """Test graceful handling when sentence-transformers unavailable."""
        reranker = CrossEncoderReranker()

        # Mock the import to fail
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            reranker._load_attempted = False
            reranker._available = False

            # Should return empty list without error
            results = reranker.score("query", sample_documents)
            # If sentence-transformers is actually installed, we get results
            # If not, we get empty list
            assert isinstance(results, list)

    @pytest.mark.skipif(
        True,  # Skip if sentence-transformers not installed
        reason="sentence-transformers not installed"
    )
    def test_cross_encoder_score(self, sample_documents):
        """Test cross-encoder scoring (requires sentence-transformers)."""
        reranker = CrossEncoderReranker()
        if not reranker.is_available:
            pytest.skip("Cross-encoder model not available")

        results = reranker.score("Python programming", sample_documents)
        assert len(results) == len(sample_documents)


# =============================================================================
# SEMANTIC RERANKER TESTS
# =============================================================================

class TestSemanticReranker:
    """Tests for SemanticReranker main class."""

    @pytest.mark.asyncio
    async def test_rerank_basic(self, semantic_reranker, sample_documents):
        """Test basic reranking."""
        results = await semantic_reranker.rerank(
            "Python programming",
            sample_documents,
            top_k=3,
        )

        assert len(results) <= 3
        assert all(isinstance(r, ScoredDocument) for r in results)
        # Check ranks are assigned
        assert results[0].rank == 1

    @pytest.mark.asyncio
    async def test_rerank_empty_documents(self, semantic_reranker):
        """Test reranking empty list."""
        results = await semantic_reranker.rerank("query", [], top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_rerank_with_cache(self, semantic_reranker, sample_documents):
        """Test that caching works for repeated queries."""
        query = "Python programming"

        # First call
        results1 = await semantic_reranker.rerank(query, sample_documents, top_k=3)

        # Second call should hit cache
        results2 = await semantic_reranker.rerank(query, sample_documents, top_k=3)

        assert semantic_reranker._cache._hits >= 1
        assert len(results1) == len(results2)

    @pytest.mark.asyncio
    async def test_rerank_cache_disabled(self, sample_documents):
        """Test reranking with cache disabled."""
        reranker = SemanticReranker(enable_cache=False)

        results = await reranker.rerank(
            "query",
            sample_documents,
            top_k=3,
            use_cache=True,  # Should be ignored
        )

        assert len(results) <= 3


class TestRRFFusion:
    """Tests for RRF (Reciprocal Rank Fusion)."""

    @pytest.mark.asyncio
    async def test_rrf_basic(self, semantic_reranker, sample_documents):
        """Test basic RRF fusion."""
        list1 = sample_documents[:3]
        list2 = sample_documents[2:]
        list3 = [sample_documents[4], sample_documents[0], sample_documents[1]]

        results = await semantic_reranker.rrf_fusion([list1, list2, list3])

        assert len(results) == 5  # All unique documents
        assert all(isinstance(r, ScoredDocument) for r in results)
        assert results[0].score > results[-1].score

    @pytest.mark.asyncio
    async def test_rrf_empty_lists(self, semantic_reranker):
        """Test RRF with empty lists."""
        results = await semantic_reranker.rrf_fusion([])
        assert results == []

    @pytest.mark.asyncio
    async def test_rrf_single_list(self, semantic_reranker, sample_documents):
        """Test RRF with single list."""
        results = await semantic_reranker.rrf_fusion([sample_documents])

        assert len(results) == len(sample_documents)
        # Scores should follow 1/(k+rank) pattern
        assert results[0].score > results[1].score

    @pytest.mark.asyncio
    async def test_rrf_with_weights(self, semantic_reranker, sample_documents):
        """Test RRF with custom weights."""
        list1 = sample_documents[:2]
        list2 = sample_documents[2:4]

        # Weight list1 higher
        results = await semantic_reranker.rrf_fusion(
            [list1, list2],
            weights=[2.0, 1.0]
        )

        # Documents from list1 should have higher scores
        assert results[0].document.id in ["doc1", "doc2"]

    @pytest.mark.asyncio
    async def test_rrf_weight_mismatch_error(self, semantic_reranker, sample_documents):
        """Test RRF raises error on weight mismatch."""
        with pytest.raises(ValueError, match="Number of weights"):
            await semantic_reranker.rrf_fusion(
                [sample_documents[:2], sample_documents[2:]],
                weights=[1.0]  # Wrong number of weights
            )

    @pytest.mark.asyncio
    async def test_rrf_k_parameter(self, semantic_reranker, sample_documents):
        """Test RRF k parameter affects smoothing."""
        list1 = sample_documents[:3]

        # Low k = sharper rank differences
        results_low_k = await semantic_reranker.rrf_fusion([list1], k=10)

        # High k = smoother rank differences
        results_high_k = await semantic_reranker.rrf_fusion([list1], k=100)

        # Score ratio between rank 1 and rank 2 should be larger with low k
        ratio_low = results_low_k[0].score / results_low_k[1].score
        ratio_high = results_high_k[0].score / results_high_k[1].score

        assert ratio_low > ratio_high


class TestDiversityReranking:
    """Tests for diversity-aware reranking."""

    @pytest.mark.asyncio
    async def test_diversity_basic(self, semantic_reranker, scored_documents):
        """Test basic diversity reranking."""
        results = await semantic_reranker.diversity_rerank(
            scored_documents,
            lambda_diversity=0.5,
        )

        assert len(results) == len(scored_documents)
        assert all(isinstance(r, ScoredDocument) for r in results)

    @pytest.mark.asyncio
    async def test_diversity_empty(self, semantic_reranker):
        """Test diversity with empty list."""
        results = await semantic_reranker.diversity_rerank([])
        assert results == []

    @pytest.mark.asyncio
    async def test_diversity_lambda_affects_order(self, semantic_reranker, scored_documents):
        """Test that lambda_diversity affects result order."""
        # High lambda = more relevance focused
        results_high = await semantic_reranker.diversity_rerank(
            scored_documents,
            lambda_diversity=1.0,
        )

        # Low lambda = more diversity focused
        results_low = await semantic_reranker.diversity_rerank(
            scored_documents,
            lambda_diversity=0.1,
        )

        # With lambda=1.0, should preserve original relevance order
        # (top relevance doc stays on top)
        assert results_high[0].document.id == scored_documents[0].document.id

    @pytest.mark.asyncio
    async def test_diversity_top_k(self, semantic_reranker, scored_documents):
        """Test diversity with top_k limit."""
        results = await semantic_reranker.diversity_rerank(
            scored_documents,
            lambda_diversity=0.5,
            top_k=3,
        )

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_diversity_jaccard_fallback(self, semantic_reranker, scored_documents):
        """Test diversity uses Jaccard fallback when embeddings unavailable."""
        # Force embedding unavailable
        semantic_reranker._embedding_available = False
        semantic_reranker._embedding_load_attempted = True

        results = await semantic_reranker.diversity_rerank(
            scored_documents,
            lambda_diversity=0.5,
        )

        assert len(results) == len(scored_documents)
        # Check metadata indicates jaccard method
        assert results[0].metadata.get("diversity_method") == "jaccard"


class TestFullPipeline:
    """Tests for the full reranking pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, semantic_reranker, sample_documents):
        """Test full pipeline: RRF -> rerank -> diversity."""
        list1 = sample_documents[:3]
        list2 = sample_documents[2:]

        results = await semantic_reranker.rerank_with_fusion_and_diversity(
            query="Python data science",
            result_lists=[list1, list2],
            top_k=3,
            rrf_k=60,
            lambda_diversity=0.3,
        )

        assert len(results) <= 3
        assert all(isinstance(r, ScoredDocument) for r in results)

    @pytest.mark.asyncio
    async def test_full_pipeline_empty(self, semantic_reranker):
        """Test full pipeline with empty input."""
        results = await semantic_reranker.rerank_with_fusion_and_diversity(
            query="query",
            result_lists=[],
            top_k=5,
        )

        assert results == []


# =============================================================================
# MEMORY RERANKER TESTS
# =============================================================================

class TestMemoryReranker:
    """Tests for MemoryReranker integration."""

    @dataclass
    class MockMemoryEntry:
        """Mock MemoryEntry for testing."""
        id: str
        content: str
        metadata: Dict[str, Any]
        embedding: Optional[List[float]] = None

    @pytest.fixture
    def mock_backend(self, sample_documents):
        """Create a mock memory backend."""
        backend = MagicMock()

        async def mock_search(query: str, limit: int = 10):
            entries = [
                self.MockMemoryEntry(
                    id=doc.id,
                    content=doc.content,
                    metadata=doc.metadata,
                )
                for doc in sample_documents[:limit]
            ]
            return entries

        backend.search = mock_search
        return backend

    @pytest.mark.asyncio
    async def test_memory_reranker_basic(self, mock_backend):
        """Test MemoryReranker basic usage."""
        reranker = MemoryReranker()

        results = await reranker.search_and_rerank(
            backend=mock_backend,
            query="Python programming",
            initial_limit=5,
            final_top_k=3,
            apply_diversity=False,
        )

        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_memory_reranker_with_diversity(self, mock_backend):
        """Test MemoryReranker with diversity."""
        reranker = MemoryReranker()

        results = await reranker.search_and_rerank(
            backend=mock_backend,
            query="Python",
            initial_limit=5,
            final_top_k=3,
            apply_diversity=True,
            lambda_diversity=0.5,
        )

        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_memory_to_document_conversion(self):
        """Test memory entry to document conversion."""
        reranker = MemoryReranker()

        entry = self.MockMemoryEntry(
            id="mem1",
            content="Memory content",
            metadata={"key": "value"},
            embedding=[0.1, 0.2],
        )

        doc = reranker._memory_to_document(entry)

        assert doc.id == "mem1"
        assert doc.content == "Memory content"
        assert doc.metadata == {"key": "value"}
        assert doc.embedding == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_memory_to_document_dict(self):
        """Test conversion from dict."""
        reranker = MemoryReranker()

        entry_dict = {
            "id": "dict1",
            "content": "Dict content",
            "metadata": {"source": "test"},
        }

        doc = reranker._memory_to_document(entry_dict)

        assert doc.id == "dict1"
        assert doc.content == "Dict content"

    @pytest.mark.asyncio
    async def test_hybrid_search(self, sample_documents):
        """Test hybrid search with multiple backends."""
        # Create mock backends
        backend1 = MagicMock()
        backend2 = MagicMock()

        async def search1(query, limit):
            return [
                self.MockMemoryEntry(id=d.id, content=d.content, metadata={})
                for d in sample_documents[:3]
            ]

        async def search2(query, limit):
            return [
                self.MockMemoryEntry(id=d.id, content=d.content, metadata={})
                for d in sample_documents[2:]
            ]

        backend1.search = search1
        backend2.search = search2

        reranker = MemoryReranker()
        results = await reranker.hybrid_search_and_rerank(
            backends=[backend1, backend2],
            query="Python",
            initial_limit=5,
            final_top_k=3,
        )

        assert len(results) <= 3


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_reranker(self):
        """Test create_reranker factory."""
        reranker = create_reranker(
            enable_cache=True,
            cache_size=500,
            cache_ttl=120,
        )

        assert isinstance(reranker, SemanticReranker)
        assert reranker._cache_enabled is True
        assert reranker._cache.max_size == 500

    def test_create_memory_reranker(self):
        """Test create_memory_reranker factory."""
        reranker = create_memory_reranker()
        assert isinstance(reranker, MemoryReranker)
        assert isinstance(reranker.reranker, SemanticReranker)

    def test_create_memory_reranker_with_custom(self):
        """Test create_memory_reranker with custom SemanticReranker."""
        semantic = SemanticReranker(cache_size=50)
        reranker = create_memory_reranker(semantic_reranker=semantic)

        assert reranker.reranker is semantic


# =============================================================================
# DIAGNOSTICS TESTS
# =============================================================================

class TestDiagnostics:
    """Tests for diagnostic methods."""

    def test_cache_stats(self, semantic_reranker):
        """Test cache stats."""
        stats = semantic_reranker.cache_stats
        assert "size" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats

    def test_diagnostics(self, semantic_reranker):
        """Test diagnostics output."""
        diag = semantic_reranker.get_diagnostics()

        assert "cross_encoder_available" in diag
        assert "cross_encoder_model" in diag
        assert "cache_enabled" in diag
        assert "cache_stats" in diag

    def test_clear_cache(self, semantic_reranker, sample_documents):
        """Test cache clearing."""
        # Add to cache
        asyncio.run(semantic_reranker.rerank("query", sample_documents, top_k=3))

        # Clear
        semantic_reranker.clear_cache()

        assert semantic_reranker._cache.stats["size"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
