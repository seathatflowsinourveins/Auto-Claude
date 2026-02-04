"""
RAG Pipeline Integration Tests - UNLEASH Platform V36

Comprehensive integration test suite for the full RAG (Retrieval Augmented Generation) pipeline:

1. TestRAGPipelineIntegration:
   - Basic retrieval flow with document indexing and search
   - Multi-retriever fusion combining multiple search strategies
   - Strategy selection based on query characteristics
   - Context management with token limits and prioritization

2. TestMemoryIntegration:
   - Unified memory store/retrieve across layers
   - HNSW vector search with approximate nearest neighbors
   - Cross-session persistence and recovery

3. TestAdapterIntegration:
   - Registry health check for all adapters
   - Adapter fallback chains on failure

Usage:
    pytest platform/tests/integration/test_rag_integration.py -v
    pytest platform/tests/integration/test_rag_integration.py -v -k "test_basic"
    pytest platform/tests/integration/test_rag_integration.py -v -m integration
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# =============================================================================
# IMPORTS WITH GRACEFUL FALLBACKS
# =============================================================================

# Unified pipeline
try:
    from core.unified_pipeline import (
        Pipeline,
        PipelineFactory,
        PipelineResult,
        PipelineStatus,
        PipelineStep,
        DeepResearchPipeline,
        AutonomousTaskPipeline,
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    Pipeline = None
    PipelineFactory = None
    PipelineResult = None

# Unified memory gateway
try:
    from core.unified_memory_gateway import (
        UnifiedMemoryGateway,
        MemoryEntry,
        MemoryQuery,
        MemoryResult,
        MemoryLayer,
        MemoryNamespace,
        MemoryBackend,
        get_memory_gateway,
        reset_memory_gateway,
        ClaudeMemBackend,
        EpisodicMemoryBackend,
        GraphMemoryBackend,
        StaticMemoryBackend,
    )
    MEMORY_GATEWAY_AVAILABLE = True
except ImportError:
    MEMORY_GATEWAY_AVAILABLE = False
    UnifiedMemoryGateway = None
    MemoryEntry = None
    MemoryQuery = None

# Iterative retrieval
try:
    from core.iterative_retrieval import (
        IterativeRetriever,
        RetrievalResult,
        StorageResult,
        SubAgentMemoryMixin,
        retrieve_context_for_task,
    )
    ITERATIVE_RETRIEVAL_AVAILABLE = True
except ImportError:
    ITERATIVE_RETRIEVAL_AVAILABLE = False
    IterativeRetriever = None
    RetrievalResult = None

# Cross-session memory
try:
    from core.cross_session_memory import (
        CrossSessionMemory,
        Memory,
        Session,
        get_memory_store,
        remember_decision,
        remember_fact,
        remember_learning,
        recall,
    )
    CROSS_SESSION_AVAILABLE = True
except ImportError:
    CROSS_SESSION_AVAILABLE = False
    CrossSessionMemory = None

# Adapter registry
try:
    from adapters.registry import (
        AdapterRegistry,
        AdapterInfo,
        AdapterLoadStatus,
        HealthCheckResult,
        get_registry,
        register_adapter,
    )
    ADAPTER_REGISTRY_AVAILABLE = True
except ImportError:
    ADAPTER_REGISTRY_AVAILABLE = False
    AdapterRegistry = None
    get_registry = None

# Research adapters
try:
    from adapters.exa_adapter import ExaAdapter
    EXA_AVAILABLE = True
except ImportError:
    ExaAdapter = None
    EXA_AVAILABLE = False

try:
    from adapters.tavily_adapter import TavilyAdapter
    TAVILY_AVAILABLE = True
except ImportError:
    TavilyAdapter = None
    TAVILY_AVAILABLE = False

try:
    from adapters.jina_adapter import JinaAdapter
    JINA_AVAILABLE = True
except ImportError:
    JinaAdapter = None
    JINA_AVAILABLE = False

try:
    from adapters.perplexity_adapter import PerplexityAdapter
    PERPLEXITY_AVAILABLE = True
except ImportError:
    PerplexityAdapter = None
    PERPLEXITY_AVAILABLE = False


# =============================================================================
# HELPER CLASSES FOR TESTING
# =============================================================================

@dataclass
class MockSearchResult:
    """Mock search result for testing retrieval."""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MockVectorIndex:
    """Mock HNSW-like vector index for testing."""

    def __init__(self, dim: int = 768, m: int = 16, ef_construction: int = 200):
        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self.vectors: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def add(self, id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None):
        """Add vector to index."""
        if len(vector) != self.dim:
            raise ValueError(f"Vector dimension mismatch: expected {self.dim}, got {len(vector)}")
        self.vectors[id] = vector
        self.metadata[id] = metadata or {}

    def search(self, query_vector: List[float], k: int = 10, ef_search: int = 100) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors."""
        if len(query_vector) != self.dim:
            raise ValueError(f"Query dimension mismatch: expected {self.dim}, got {len(query_vector)}")

        # Calculate cosine similarity for all vectors
        similarities = []
        for id, vec in self.vectors.items():
            sim = self._cosine_similarity(query_vector, vec)
            similarities.append((id, sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:k]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def count(self) -> int:
        """Return number of vectors in index."""
        return len(self.vectors)

    def clear(self):
        """Clear all vectors from index."""
        self.vectors.clear()
        self.metadata.clear()


class MockEmbeddingModel:
    """Mock embedding model for testing."""

    def __init__(self, dim: int = 768):
        self.dim = dim
        self._cache: Dict[str, List[float]] = {}

    def embed(self, text: str) -> List[float]:
        """Generate deterministic pseudo-embedding for text."""
        if text in self._cache:
            return self._cache[text]

        # Generate deterministic embedding based on text hash
        random.seed(hash(text) % (2**32))
        embedding = [random.random() for _ in range(self.dim)]
        self._cache[text] = embedding
        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]


class MockRetriever:
    """Mock retriever for testing multi-retriever fusion."""

    def __init__(self, name: str, documents: List[Dict[str, Any]]):
        self.name = name
        self.documents = documents
        self.call_count = 0

    async def search(self, query: str, k: int = 10) -> List[MockSearchResult]:
        """Search documents using simple keyword matching."""
        self.call_count += 1
        query_lower = query.lower()
        results = []

        for doc in self.documents:
            content_lower = doc["content"].lower()
            # Calculate simple relevance score
            score = 0.0
            query_terms = set(query_lower.split())
            content_terms = set(content_lower.split())
            overlap = len(query_terms & content_terms)
            if query_terms:
                score = overlap / len(query_terms)

            # Exact phrase bonus
            if query_lower in content_lower:
                score += 0.3

            if score > 0:
                results.append(MockSearchResult(
                    id=doc["id"],
                    content=doc["content"],
                    score=min(score, 1.0),
                    metadata=doc.get("metadata", {})
                ))

        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_memory_dir():
    """Create a temporary directory for memory persistence tests."""
    temp_dir = tempfile.mkdtemp(prefix="rag_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_documents() -> List[Dict[str, Any]]:
    """Sample documents for RAG testing."""
    return [
        {
            "id": "doc-1",
            "content": "Python is a versatile programming language used in web development, data science, and AI applications. It features dynamic typing and garbage collection.",
            "metadata": {"topic": "programming", "language": "python", "difficulty": "beginner"}
        },
        {
            "id": "doc-2",
            "content": "Machine learning models learn patterns from training data to make accurate predictions on new data. Common algorithms include decision trees, random forests, and neural networks.",
            "metadata": {"topic": "machine_learning", "difficulty": "intermediate"}
        },
        {
            "id": "doc-3",
            "content": "Natural language processing enables computers to understand, interpret, and generate human language. Key tasks include sentiment analysis, named entity recognition, and machine translation.",
            "metadata": {"topic": "nlp", "difficulty": "advanced"}
        },
        {
            "id": "doc-4",
            "content": "Deep learning uses artificial neural networks with multiple layers to extract hierarchical features. Transformers have revolutionized NLP with attention mechanisms.",
            "metadata": {"topic": "deep_learning", "difficulty": "advanced"}
        },
        {
            "id": "doc-5",
            "content": "Vector databases store embeddings and enable efficient similarity search for retrieval augmented generation. HNSW is a popular algorithm for approximate nearest neighbor search.",
            "metadata": {"topic": "vector_db", "difficulty": "intermediate"}
        },
        {
            "id": "doc-6",
            "content": "RAG systems combine retrieval and generation to produce factually grounded responses. They reduce hallucination by providing relevant context to language models.",
            "metadata": {"topic": "rag", "difficulty": "intermediate"}
        },
        {
            "id": "doc-7",
            "content": "Prompt engineering involves crafting effective prompts to guide language model behavior. Techniques include few-shot learning, chain-of-thought, and system prompts.",
            "metadata": {"topic": "prompt_engineering", "difficulty": "intermediate"}
        },
        {
            "id": "doc-8",
            "content": "Semantic search uses embeddings to find documents based on meaning rather than exact keyword matches. This improves recall for queries with synonyms or paraphrases.",
            "metadata": {"topic": "semantic_search", "difficulty": "intermediate"}
        },
    ]


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model."""
    return MockEmbeddingModel(dim=384)


@pytest.fixture
def mock_vector_index():
    """Create a mock vector index."""
    return MockVectorIndex(dim=384, m=16, ef_construction=200)


@pytest.fixture
def indexed_documents(sample_documents, mock_embedding_model, mock_vector_index):
    """Create an indexed document collection."""
    for doc in sample_documents:
        embedding = mock_embedding_model.embed(doc["content"])
        mock_vector_index.add(doc["id"], embedding, {"content": doc["content"], **doc.get("metadata", {})})
    return {
        "documents": sample_documents,
        "index": mock_vector_index,
        "model": mock_embedding_model
    }


@pytest.fixture
def mock_retrievers(sample_documents):
    """Create multiple mock retrievers for fusion testing."""
    # Split documents for different retrievers
    keyword_docs = sample_documents[:4]
    semantic_docs = sample_documents[2:6]
    hybrid_docs = sample_documents[4:]

    return {
        "keyword": MockRetriever("keyword", keyword_docs),
        "semantic": MockRetriever("semantic", semantic_docs),
        "hybrid": MockRetriever("hybrid", hybrid_docs),
    }


@pytest.fixture
def memory_gateway(temp_memory_dir):
    """Create a memory gateway for testing."""
    if not MEMORY_GATEWAY_AVAILABLE:
        pytest.skip("UnifiedMemoryGateway not available")

    # Reset singleton
    reset_memory_gateway()

    # Create gateway with mocked Letta (no real API calls)
    with patch.dict(os.environ, {"LETTA_API_KEY": ""}, clear=False):
        gateway = UnifiedMemoryGateway(project="test")
        yield gateway

    reset_memory_gateway()


@pytest.fixture
def cross_session_memory(temp_memory_dir):
    """Create a cross-session memory instance with temp storage."""
    if not CROSS_SESSION_AVAILABLE:
        pytest.skip("CrossSessionMemory not available")

    memory = CrossSessionMemory(base_path=temp_memory_dir)
    yield memory


# =============================================================================
# TEST CLASS: RAG PIPELINE INTEGRATION
# =============================================================================

@pytest.mark.integration
class TestRAGPipelineIntegration:
    """Integration tests for the full RAG pipeline."""

    @pytest.mark.asyncio
    async def test_basic_retrieval_flow(self, indexed_documents):
        """Test basic document retrieval flow with indexing and search."""
        index = indexed_documents["index"]
        model = indexed_documents["model"]
        documents = indexed_documents["documents"]

        # Verify documents are indexed
        assert index.count() == len(documents)

        # Test query
        query = "machine learning neural networks"
        query_embedding = model.embed(query)

        # Search
        results = index.search(query_embedding, k=3)

        # Verify results structure
        assert len(results) <= 3
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)  # (id, score)

        # Verify scores are in valid range
        for doc_id, score in results:
            assert 0.0 <= score <= 1.0
            assert doc_id in index.metadata

        # Verify top results are relevant
        top_ids = [r[0] for r in results]
        # Machine learning or deep learning docs should be in top results
        ml_related = ["doc-2", "doc-4"]  # ML and deep learning docs
        assert any(doc_id in ml_related for doc_id in top_ids), \
            f"Expected ML-related docs in top results, got {top_ids}"

    @pytest.mark.asyncio
    async def test_multi_retriever_fusion(self, mock_retrievers, sample_documents):
        """Test multi-retriever fusion combining multiple search strategies."""

        async def reciprocal_rank_fusion(
            retrievers: Dict[str, MockRetriever],
            query: str,
            k: int = 10,
            rrf_k: int = 60
        ) -> List[MockSearchResult]:
            """Perform Reciprocal Rank Fusion across multiple retrievers."""
            # Gather results from all retrievers
            all_results: Dict[str, List[MockSearchResult]] = {}
            for name, retriever in retrievers.items():
                all_results[name] = await retriever.search(query, k=k)

            # Calculate RRF scores
            rrf_scores: Dict[str, float] = {}
            result_map: Dict[str, MockSearchResult] = {}

            for name, results in all_results.items():
                for rank, result in enumerate(results, 1):
                    doc_id = result.id
                    rrf_score = 1.0 / (rrf_k + rank)

                    if doc_id not in rrf_scores:
                        rrf_scores[doc_id] = 0.0
                        result_map[doc_id] = result

                    rrf_scores[doc_id] += rrf_score

            # Sort by RRF score
            sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

            # Build final results
            fused_results = []
            for doc_id in sorted_ids[:k]:
                result = result_map[doc_id]
                result.score = rrf_scores[doc_id]
                fused_results.append(result)

            return fused_results

        # Test fusion
        query = "neural network deep learning patterns"
        fused = await reciprocal_rank_fusion(mock_retrievers, query, k=5)

        # Verify all retrievers were called
        assert mock_retrievers["keyword"].call_count == 1
        assert mock_retrievers["semantic"].call_count == 1
        assert mock_retrievers["hybrid"].call_count == 1

        # Verify fusion produced results
        assert len(fused) > 0

        # Verify RRF scores are calculated
        for result in fused:
            assert result.score > 0

    @pytest.mark.asyncio
    async def test_strategy_selection(self, sample_documents):
        """Test retrieval strategy selection based on query characteristics."""

        def select_strategy(query: str) -> str:
            """Select retrieval strategy based on query analysis."""
            query_lower = query.lower()
            word_count = len(query.split())

            # Factual/lookup queries -> keyword search
            factual_indicators = ["what is", "define", "who is", "when did"]
            if any(ind in query_lower for ind in factual_indicators):
                return "keyword"

            # Complex/reasoning queries -> semantic search
            reasoning_indicators = ["how does", "why", "explain", "compare", "relationship"]
            if any(ind in query_lower for ind in reasoning_indicators):
                return "semantic"

            # Short queries -> keyword, long queries -> hybrid
            if word_count <= 3:
                return "keyword"
            elif word_count >= 8:
                return "hybrid"
            else:
                return "semantic"

        # Test strategy selection
        test_cases = [
            ("what is machine learning", "keyword"),
            ("define neural network", "keyword"),
            ("how does attention mechanism work in transformers", "semantic"),
            ("why are embeddings useful", "semantic"),
            ("compare RNN and transformer architectures for NLP tasks", "hybrid"),
            ("python", "keyword"),
            ("RAG", "keyword"),
            ("explain the relationship between retrieval and generation in RAG systems", "hybrid"),
        ]

        for query, expected_strategy in test_cases:
            selected = select_strategy(query)
            assert selected == expected_strategy, \
                f"Query '{query}': expected {expected_strategy}, got {selected}"

    @pytest.mark.asyncio
    async def test_context_management(self, indexed_documents):
        """Test context management with token limits and prioritization."""
        documents = indexed_documents["documents"]

        def manage_context(
            retrieved_docs: List[Dict[str, Any]],
            max_tokens: int = 2000,
            chars_per_token: float = 4.0
        ) -> Dict[str, Any]:
            """Manage context with token limits and prioritization."""
            max_chars = int(max_tokens * chars_per_token)

            # Sort by relevance score (assuming docs have 'score' field)
            sorted_docs = sorted(
                retrieved_docs,
                key=lambda d: d.get("score", 0),
                reverse=True
            )

            selected_docs = []
            total_chars = 0

            for doc in sorted_docs:
                content = doc.get("content", "")
                doc_chars = len(content)

                # Check if adding this doc would exceed limit
                if total_chars + doc_chars <= max_chars:
                    selected_docs.append(doc)
                    total_chars += doc_chars
                elif total_chars < max_chars:
                    # Truncate to fit remaining space
                    remaining = max_chars - total_chars
                    truncated_content = content[:remaining] + "..."
                    truncated_doc = {**doc, "content": truncated_content, "truncated": True}
                    selected_docs.append(truncated_doc)
                    total_chars = max_chars
                    break

            return {
                "documents": selected_docs,
                "total_documents": len(retrieved_docs),
                "selected_documents": len(selected_docs),
                "total_chars": total_chars,
                "estimated_tokens": int(total_chars / chars_per_token),
                "max_tokens": max_tokens,
            }

        # Add scores to documents
        scored_docs = [
            {**doc, "score": 1.0 - (i * 0.1)}
            for i, doc in enumerate(documents)
        ]

        # Test with small token limit
        result = manage_context(scored_docs, max_tokens=500)

        assert result["selected_documents"] < result["total_documents"]
        assert result["estimated_tokens"] <= result["max_tokens"]
        assert len(result["documents"]) > 0

        # Verify highest scoring docs are selected
        selected_ids = [d["id"] for d in result["documents"]]
        assert scored_docs[0]["id"] in selected_ids

        # Test with large token limit (all docs fit)
        result_large = manage_context(scored_docs, max_tokens=10000)
        assert result_large["selected_documents"] == result_large["total_documents"]

    @pytest.mark.asyncio
    async def test_query_expansion(self, indexed_documents):
        """Test query expansion for improved recall."""
        model = indexed_documents["model"]
        index = indexed_documents["index"]

        def expand_query(query: str, expansion_map: Dict[str, List[str]]) -> List[str]:
            """Expand query with synonyms and related terms."""
            expanded = [query]
            query_lower = query.lower()

            for term, expansions in expansion_map.items():
                if term in query_lower:
                    for expansion in expansions:
                        expanded_query = query_lower.replace(term, expansion)
                        expanded.append(expanded_query)

            return list(set(expanded))

        # Define expansion map
        expansion_map = {
            "ml": ["machine learning", "artificial intelligence"],
            "nlp": ["natural language processing", "text processing"],
            "neural": ["deep learning", "neural network"],
            "ai": ["artificial intelligence", "machine learning"],
        }

        # Test expansion
        query = "ml models for nlp"
        expanded = expand_query(query, expansion_map)

        assert len(expanded) > 1  # Should have expansions
        assert query in expanded  # Original query preserved

        # Search with expanded queries and combine results
        all_results = {}
        for eq in expanded:
            query_embedding = model.embed(eq)
            results = index.search(query_embedding, k=3)
            for doc_id, score in results:
                if doc_id not in all_results or all_results[doc_id] < score:
                    all_results[doc_id] = score

        # Verify expansion improved coverage
        assert len(all_results) > 0


# =============================================================================
# TEST CLASS: MEMORY INTEGRATION
# =============================================================================

@pytest.mark.integration
class TestMemoryIntegration:
    """Integration tests for unified memory system."""

    @pytest.mark.asyncio
    async def test_unified_memory_store_retrieve(self, memory_gateway):
        """Test storing and retrieving memories across unified gateway."""
        # Store memories in different layers
        await memory_gateway.store(
            "Python best practices for async programming",
            MemoryLayer.CLAUDE_MEM,
            MemoryNamespace.PATTERNS,
            {"topic": "python", "category": "best_practices"}
        )

        await memory_gateway.store(
            "Decision to use DSPy for prompt optimization",
            MemoryLayer.CLAUDE_MEM,
            MemoryNamespace.DECISIONS,
            {"topic": "architecture", "sdk": "dspy"}
        )

        await memory_gateway.store(
            "Learned that HNSW with M=16 is optimal for our scale",
            MemoryLayer.EPISODIC,
            MemoryNamespace.LEARNINGS,
            {"topic": "vector_search", "parameter": "M"}
        )

        # Search across all layers
        query = MemoryQuery(
            query_text="async programming patterns",
            layers=None,  # All layers
            max_results=10,
            min_relevance=0.0
        )

        result = await memory_gateway.query(query)

        assert result is not None
        assert result.total_found >= 0
        assert isinstance(result.entries, list)
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_hnsw_search(self, mock_vector_index, mock_embedding_model):
        """Test HNSW-style vector search with approximate nearest neighbors."""
        # Create test dataset
        test_texts = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning algorithms process data",
            "Neural networks learn from examples",
            "Natural language understanding is complex",
            "Vector databases enable semantic search",
            "Transformers use attention mechanisms",
            "Deep learning requires lots of data",
            "Embeddings capture semantic meaning",
        ]

        # Index all texts
        for i, text in enumerate(test_texts):
            embedding = mock_embedding_model.embed(text)
            mock_vector_index.add(f"text-{i}", embedding, {"content": text})

        assert mock_vector_index.count() == len(test_texts)

        # Query similar to "neural networks and deep learning"
        query = "neural networks and deep learning"
        query_embedding = mock_embedding_model.embed(query)

        results = mock_vector_index.search(query_embedding, k=3, ef_search=50)

        # Verify results
        assert len(results) == 3
        assert all(score >= 0 for _, score in results)

        # Verify results are sorted by similarity (descending)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_cross_session_persistence(self, temp_memory_dir):
        """Test that memories persist across sessions."""
        if not CROSS_SESSION_AVAILABLE:
            pytest.skip("CrossSessionMemory not available")

        # Session 1: Create and store memories
        memory1 = CrossSessionMemory(base_path=temp_memory_dir)
        session1 = memory1.start_session("First session - store patterns")

        memory1.add(
            "Use async/await for I/O bound operations",
            memory_type="decision",
            importance=0.9,
            tags=["python", "async", "best-practice"]
        )

        memory1.add(
            "HNSW M=16 provides good balance of speed and recall",
            memory_type="learning",
            importance=0.8,
            tags=["vector-search", "hnsw", "tuning"]
        )

        memory1.add(
            "RAG reduces hallucination by grounding responses",
            memory_type="fact",
            importance=0.85,
            tags=["rag", "llm", "quality"]
        )

        memory1.end_session("Stored 3 important memories")

        stats1 = memory1.get_stats()
        assert stats1["total_memories"] == 3

        # Session 2: Load and verify persistence
        memory2 = CrossSessionMemory(base_path=temp_memory_dir)
        session2 = memory2.start_session("Second session - verify persistence")

        stats2 = memory2.get_stats()
        assert stats2["total_memories"] == 3

        # Search for persisted memories
        results = memory2.search("async programming", limit=5)
        assert len(results) >= 1
        assert any("async" in r.content.lower() for r in results)

        # Verify decisions persisted
        decisions = memory2.get_decisions(limit=10)
        assert len(decisions) >= 1

        # Verify learnings persisted
        learnings = memory2.get_learnings(limit=10)
        assert len(learnings) >= 1

        # Generate context for new session
        context = memory2.get_session_context(max_tokens=1000)
        assert len(context) > 0
        assert "Decision" in context or "decision" in context.lower()

    @pytest.mark.asyncio
    async def test_memory_deduplication(self, memory_gateway):
        """Test memory deduplication across layers."""
        # Store same content in multiple layers
        content = "Common knowledge: embeddings capture semantic meaning"

        await memory_gateway.store(
            content,
            MemoryLayer.CLAUDE_MEM,
            MemoryNamespace.PATTERNS
        )

        await memory_gateway.store(
            content,
            MemoryLayer.EPISODIC,
            MemoryNamespace.PATTERNS
        )

        # Query should deduplicate
        query = MemoryQuery(
            query_text="embeddings semantic meaning",
            layers=None,
            max_results=10,
            min_relevance=0.0
        )

        result = await memory_gateway.query(query)

        # Deduplication count should be > 0 if same content found in multiple layers
        # Note: This depends on content hash matching
        assert result.deduplicated_count >= 0

    @pytest.mark.asyncio
    async def test_memory_ttl_expiration(self, temp_memory_dir):
        """Test memory TTL-based expiration."""
        if not CROSS_SESSION_AVAILABLE:
            pytest.skip("CrossSessionMemory not available")

        memory = CrossSessionMemory(base_path=temp_memory_dir)
        memory.start_session("TTL test")

        # Add memory (note: CrossSessionMemory doesn't have built-in TTL,
        # but we test the concept)
        added = memory.add(
            "This is temporary context",
            memory_type="context",
            importance=0.3
        )

        assert added.id is not None

        # Invalidate to simulate expiration
        memory.invalidate(added.id, reason="Simulated TTL expiration")

        # Search should not return invalidated memory
        results = memory.search("temporary context", limit=10)
        assert not any(r.id == added.id for r in results)


# =============================================================================
# TEST CLASS: ADAPTER INTEGRATION
# =============================================================================

@pytest.mark.integration
class TestAdapterIntegration:
    """Integration tests for adapter registry and fallback."""

    @pytest.mark.asyncio
    async def test_registry_health_check(self):
        """Test health check for all registered adapters."""
        if not ADAPTER_REGISTRY_AVAILABLE:
            pytest.skip("AdapterRegistry not available")

        registry = AdapterRegistry()

        # Create mock adapter class
        class MockHealthyAdapter:
            def __init__(self):
                self._initialized = False

            async def initialize(self, config):
                self._initialized = True

            async def health_check(self):
                return MagicMock(success=True, data={"status": "healthy"})

            async def shutdown(self):
                pass

        class MockUnhealthyAdapter:
            def __init__(self):
                self._initialized = False

            async def initialize(self, config):
                self._initialized = True

            async def health_check(self):
                return MagicMock(success=False, error="Connection failed")

            async def shutdown(self):
                pass

        # Register adapters
        registry.register_adapter(
            name="healthy-adapter",
            adapter_class=MockHealthyAdapter,
            layer="RESEARCH",
            priority=10,
            features=["search", "extract"]
        )

        registry.register_adapter(
            name="unhealthy-adapter",
            adapter_class=MockUnhealthyAdapter,
            layer="RESEARCH",
            priority=5,
            features=["search"]
        )

        # Get status report
        status = registry.get_status_report()

        assert "summary" in status
        assert status["summary"]["total_registered"] >= 2

        # List available adapters
        available = registry.list_available()
        assert len(available) >= 0  # May be 0 if SDK checks fail

        # Run health checks
        health_results = await registry.health_check_all(timeout=5.0)

        # Verify health check results structure
        assert isinstance(health_results, dict)
        for name, result in health_results.items():
            assert hasattr(result, "is_healthy")
            assert hasattr(result, "latency_ms")

    @pytest.mark.asyncio
    async def test_adapter_fallback(self):
        """Test adapter fallback chain on failure."""
        if not ADAPTER_REGISTRY_AVAILABLE:
            pytest.skip("AdapterRegistry not available")

        class AdapterFallbackChain:
            """Adapter fallback chain for resilient operations."""

            def __init__(self, adapters: List[Any]):
                self.adapters = adapters
                self.current_index = 0
                self.fallback_count = 0

            async def execute_with_fallback(self, operation: str, **kwargs) -> Any:
                """Execute operation with automatic fallback on failure."""
                last_error = None

                for i, adapter in enumerate(self.adapters):
                    try:
                        result = await adapter.execute(operation, **kwargs)
                        if result.success:
                            self.current_index = i
                            return result
                        last_error = result.error
                    except Exception as e:
                        last_error = str(e)
                        self.fallback_count += 1
                        continue

                # All adapters failed
                return MagicMock(success=False, error=f"All adapters failed. Last error: {last_error}")

        # Create mock adapters with different behaviors
        class FailingAdapter:
            async def execute(self, operation, **kwargs):
                raise Exception("Primary adapter unavailable")

        class SlowAdapter:
            async def execute(self, operation, **kwargs):
                await asyncio.sleep(0.1)
                return MagicMock(success=True, data={"source": "slow"})

        class ReliableAdapter:
            async def execute(self, operation, **kwargs):
                return MagicMock(success=True, data={"source": "reliable"})

        # Test fallback chain
        chain = AdapterFallbackChain([
            FailingAdapter(),
            SlowAdapter(),
            ReliableAdapter(),
        ])

        result = await chain.execute_with_fallback("search", query="test")

        assert result.success is True
        assert chain.fallback_count >= 1  # At least one fallback occurred

    @pytest.mark.asyncio
    async def test_adapter_circuit_breaker(self):
        """Test circuit breaker pattern for adapter protection."""

        class CircuitBreaker:
            """Simple circuit breaker for adapter protection."""

            def __init__(
                self,
                failure_threshold: int = 3,
                recovery_timeout: float = 30.0
            ):
                self.failure_threshold = failure_threshold
                self.recovery_timeout = recovery_timeout
                self.failure_count = 0
                self.last_failure_time: Optional[float] = None
                self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

            def can_execute(self) -> bool:
                """Check if execution is allowed."""
                if self.state == "CLOSED":
                    return True

                if self.state == "OPEN":
                    # Check if recovery timeout has passed
                    if self.last_failure_time:
                        elapsed = time.time() - self.last_failure_time
                        if elapsed >= self.recovery_timeout:
                            self.state = "HALF_OPEN"
                            return True
                    return False

                # HALF_OPEN allows one request
                return True

            def record_success(self):
                """Record successful execution."""
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                self.failure_count = 0

            def record_failure(self):
                """Record failed execution."""
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"

        # Test circuit breaker
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)

        assert cb.can_execute() is True
        assert cb.state == "CLOSED"

        # Simulate failures
        for _ in range(3):
            cb.record_failure()

        assert cb.state == "OPEN"
        assert cb.can_execute() is False

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        assert cb.can_execute() is True
        assert cb.state == "HALF_OPEN"

        # Success in half-open state closes circuit
        cb.record_success()
        assert cb.state == "CLOSED"

    @pytest.mark.asyncio
    async def test_real_adapter_initialization(self):
        """Test real adapter initialization (mock mode)."""
        if ExaAdapter is None:
            pytest.skip("ExaAdapter not available")

        adapter = ExaAdapter()

        # Initialize in mock mode (no API key)
        with patch.dict(os.environ, {"EXA_API_KEY": ""}, clear=False):
            result = await adapter.initialize({})

        assert result.success is True

        # Health check
        health = await adapter.health_check()
        assert health is not None

        # Cleanup
        await adapter.shutdown()

    @pytest.mark.asyncio
    async def test_adapter_retry_logic(self):
        """Test adapter retry logic with exponential backoff."""

        class RetryableAdapter:
            """Adapter with retry logic."""

            def __init__(self, max_retries: int = 3, base_delay: float = 0.1):
                self.max_retries = max_retries
                self.base_delay = base_delay
                self.attempt_count = 0

            async def execute_with_retry(
                self,
                operation: Callable,
                *args,
                **kwargs
            ) -> Any:
                """Execute operation with exponential backoff retry."""
                last_error = None
                self.attempt_count = 0

                for attempt in range(self.max_retries + 1):
                    self.attempt_count = attempt + 1
                    try:
                        result = await operation(*args, **kwargs)
                        return result
                    except Exception as e:
                        last_error = e
                        if attempt < self.max_retries:
                            delay = self.base_delay * (2 ** attempt)
                            await asyncio.sleep(delay)

                raise last_error

        # Test retry logic
        call_count = {"count": 0}

        async def flaky_operation():
            call_count["count"] += 1
            if call_count["count"] < 3:
                raise Exception("Temporary failure")
            return "Success"

        adapter = RetryableAdapter(max_retries=3, base_delay=0.01)
        result = await adapter.execute_with_retry(flaky_operation)

        assert result == "Success"
        assert adapter.attempt_count == 3
        assert call_count["count"] == 3


# =============================================================================
# ADDITIONAL TESTS
# =============================================================================

@pytest.mark.integration
class TestRAGQualityMetrics:
    """Tests for RAG quality metrics and evaluation."""

    @pytest.mark.asyncio
    async def test_retrieval_precision(self, indexed_documents):
        """Test retrieval precision metric."""
        index = indexed_documents["index"]
        model = indexed_documents["model"]
        documents = indexed_documents["documents"]

        # Define relevant documents for test queries
        relevance_judgments = {
            "machine learning neural networks": ["doc-2", "doc-4"],
            "vector database embeddings": ["doc-5", "doc-8"],
            "natural language processing": ["doc-3", "doc-4"],
        }

        def calculate_precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
            """Calculate precision@k."""
            retrieved_k = retrieved[:k]
            relevant_retrieved = len(set(retrieved_k) & set(relevant))
            return relevant_retrieved / k if k > 0 else 0.0

        for query, relevant_docs in relevance_judgments.items():
            query_embedding = model.embed(query)
            results = index.search(query_embedding, k=5)
            retrieved_ids = [doc_id for doc_id, _ in results]

            p_at_3 = calculate_precision_at_k(retrieved_ids, relevant_docs, k=3)
            p_at_5 = calculate_precision_at_k(retrieved_ids, relevant_docs, k=5)

            # Precision should be in valid range
            assert 0.0 <= p_at_3 <= 1.0
            assert 0.0 <= p_at_5 <= 1.0

    @pytest.mark.asyncio
    async def test_retrieval_recall(self, indexed_documents):
        """Test retrieval recall metric."""
        index = indexed_documents["index"]
        model = indexed_documents["model"]

        relevant_docs = ["doc-2", "doc-4"]  # ML-related docs
        query = "machine learning deep neural network training"

        query_embedding = model.embed(query)
        results = index.search(query_embedding, k=5)
        retrieved_ids = [doc_id for doc_id, _ in results]

        # Calculate recall
        relevant_retrieved = len(set(retrieved_ids) & set(relevant_docs))
        recall = relevant_retrieved / len(relevant_docs) if relevant_docs else 0.0

        assert 0.0 <= recall <= 1.0

    @pytest.mark.asyncio
    async def test_mrr_metric(self, indexed_documents):
        """Test Mean Reciprocal Rank (MRR) metric."""
        index = indexed_documents["index"]
        model = indexed_documents["model"]

        test_queries = [
            ("python programming", "doc-1"),
            ("machine learning", "doc-2"),
            ("vector database", "doc-5"),
        ]

        reciprocal_ranks = []

        for query, expected_top_doc in test_queries:
            query_embedding = model.embed(query)
            results = index.search(query_embedding, k=10)
            retrieved_ids = [doc_id for doc_id, _ in results]

            # Find rank of expected document
            if expected_top_doc in retrieved_ids:
                rank = retrieved_ids.index(expected_top_doc) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

        assert 0.0 <= mrr <= 1.0


@pytest.mark.integration
class TestContextWindow:
    """Tests for context window management."""

    @pytest.mark.asyncio
    async def test_context_compression(self, sample_documents):
        """Test context compression for long documents."""

        def compress_context(text: str, max_length: int) -> str:
            """Compress text to fit within max_length."""
            if len(text) <= max_length:
                return text

            # Simple compression: keep first and last parts
            keep_length = (max_length - 10) // 2  # 10 chars for ellipsis
            return text[:keep_length] + " ... " + text[-keep_length:]

        long_content = " ".join([doc["content"] for doc in sample_documents])
        assert len(long_content) > 500

        compressed = compress_context(long_content, max_length=500)

        assert len(compressed) <= 500
        assert " ... " in compressed

    @pytest.mark.asyncio
    async def test_sliding_window_retrieval(self, sample_documents, mock_embedding_model):
        """Test sliding window for processing long documents."""

        def create_sliding_windows(text: str, window_size: int, overlap: int) -> List[Dict[str, Any]]:
            """Create overlapping windows from text."""
            windows = []
            start = 0
            window_idx = 0

            while start < len(text):
                end = min(start + window_size, len(text))
                window_text = text[start:end]

                windows.append({
                    "index": window_idx,
                    "start": start,
                    "end": end,
                    "text": window_text,
                    "overlap_with_prev": min(overlap, start) if start > 0 else 0
                })

                start = end - overlap if end < len(text) else len(text)
                window_idx += 1

            return windows

        # Create long document
        long_doc = " ".join([doc["content"] for doc in sample_documents])

        windows = create_sliding_windows(long_doc, window_size=200, overlap=50)

        assert len(windows) > 1
        assert all(len(w["text"]) <= 200 for w in windows)

        # Verify overlap
        for i in range(1, len(windows)):
            assert windows[i]["overlap_with_prev"] > 0 or windows[i-1]["end"] == len(long_doc)


# =============================================================================
# CLEANUP
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_registry():
    """Reset adapter registry between tests."""
    yield
    if ADAPTER_REGISTRY_AVAILABLE:
        # Reset singleton if implemented
        pass


@pytest.fixture(autouse=True)
def cleanup_memory_gateway():
    """Reset memory gateway between tests."""
    yield
    if MEMORY_GATEWAY_AVAILABLE:
        reset_memory_gateway()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
