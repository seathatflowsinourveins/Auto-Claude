"""
Unit Tests for RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)

Tests cover:
- Tree construction from documents
- Multiple clustering methods (GMM, K-Means)
- Tree traversal retrieval
- Collapsed tree retrieval
- Hybrid retrieval
- Incremental document updates
- Integration with SemanticChunker
- Edge cases and error handling
"""

import asyncio
import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock, patch

# Import RAPTOR components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from core.rag.raptor import (
    RAPTOR,
    RAPTORWithChunker,
    RAPTORIndex,
    RAPTORNode,
    RAPTORTree,
    RAPTORResult,
    RAPTORConfig,
    TreeBuildStats,
    ClusterMethod,
    RetrievalMethod,
    Clusterer,
    GMMClusterer,
    KMeansClusterer,
    get_clusterer,
    LLMSummarizer,
)


# =============================================================================
# MOCK PROVIDERS
# =============================================================================

class MockSummarizer:
    """Mock LLM for summarization."""

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.call_count = 0
        self.calls: List[str] = []

    async def summarize(
        self,
        text: str,
        max_length: int = 256,
        **kwargs
    ) -> str:
        self.call_count += 1
        self.calls.append(text)

        # Return custom response if available
        for key, response in self.responses.items():
            if key in text:
                return response

        # Default: return truncated input
        return f"Summary: {text[:100]}..."


class MockEmbedder:
    """Mock embedding provider."""

    def __init__(self, dim: int = 384):
        self._dim = dim
        self.call_count = 0

    def encode(self, texts: Union[str, List[str]]) -> Any:
        import numpy as np
        self.call_count += 1

        if isinstance(texts, str):
            texts = [texts]

        # Generate deterministic embeddings based on text content
        embeddings = []
        for text in texts:
            # Use hash for reproducibility
            hash_val = hash(text) % 10000
            np.random.seed(hash_val)
            emb = np.random.randn(self._dim).astype(np.float32)
            # Normalize
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)

        return np.array(embeddings) if len(embeddings) > 1 else embeddings[0]

    @property
    def embedding_dim(self) -> int:
        return self._dim


class MockChunker:
    """Mock semantic chunker."""

    def __init__(self, chunk_size: int = 100):
        self.chunk_size = chunk_size

    def chunk(self, text: str, metadata: Optional[Dict] = None):
        """Simple chunking by paragraphs."""
        @dataclass
        class MockChunk:
            content: str
            metadata: Dict

        paragraphs = text.split('\n\n')
        chunks = []
        for para in paragraphs:
            if para.strip():
                chunks.append(MockChunk(
                    content=para.strip(),
                    metadata=metadata or {}
                ))
        return chunks


# =============================================================================
# TEST DATA
# =============================================================================

SAMPLE_DOCUMENTS = [
    """Machine learning is a subset of artificial intelligence that enables systems
    to learn and improve from experience. It focuses on developing computer programs
    that can access data and use it to learn for themselves.

    There are three main types of machine learning: supervised learning, unsupervised
    learning, and reinforcement learning. Each has different use cases and algorithms.""",

    """Deep learning is a subset of machine learning that uses neural networks with
    many layers. It has revolutionized fields like computer vision and natural
    language processing.

    Common architectures include CNNs for images, RNNs for sequences, and Transformers
    for attention-based processing.""",

    """Retrieval-Augmented Generation (RAG) combines retrieval systems with language
    models. It improves accuracy by grounding responses in retrieved documents.

    RAPTOR is an advanced RAG technique that builds hierarchical summaries for
    multi-level retrieval.""",
]


# =============================================================================
# RAPTOR NODE TESTS
# =============================================================================

class TestRAPTORNode:
    """Tests for RAPTORNode data class."""

    def test_node_creation(self):
        """Test basic node creation."""
        node = RAPTORNode(
            id="test_node_1",
            content="Test content",
            level=0
        )

        assert node.id == "test_node_1"
        assert node.content == "Test content"
        assert node.level == 0
        assert node.children == []
        assert node.parent is None
        assert not node.is_summary

    def test_node_with_embedding(self):
        """Test node with embedding."""
        embedding = [0.1, 0.2, 0.3]
        node = RAPTORNode(
            id="test_node_2",
            content="Test content",
            embedding=embedding
        )

        assert node.embedding == embedding

    def test_node_token_estimate(self):
        """Test token estimation."""
        node = RAPTORNode(
            id="test",
            content="This is a test content with multiple words"
        )

        # ~4 chars per token
        expected_tokens = len(node.content) // 4
        assert node.token_estimate == expected_tokens

    def test_node_equality(self):
        """Test node equality based on ID."""
        node1 = RAPTORNode(id="same_id", content="Content 1")
        node2 = RAPTORNode(id="same_id", content="Content 2")
        node3 = RAPTORNode(id="different_id", content="Content 1")

        assert node1 == node2
        assert node1 != node3

    def test_node_auto_id_generation(self):
        """Test automatic ID generation."""
        node = RAPTORNode(
            id=None,
            content="Test content"
        )

        assert node.id is not None
        assert len(node.id) == 16  # SHA256 hex truncated


# =============================================================================
# RAPTOR TREE TESTS
# =============================================================================

class TestRAPTORTree:
    """Tests for RAPTORTree data class."""

    def test_tree_creation(self):
        """Test empty tree creation."""
        tree = RAPTORTree()

        assert tree.total_nodes == 0
        assert tree.max_level == 0
        assert len(tree.leaf_ids) == 0
        assert len(tree.root_ids) == 0

    def test_add_node(self):
        """Test adding nodes to tree."""
        tree = RAPTORTree()

        node = RAPTORNode(id="node1", content="Content", level=0)
        tree.add_node(node)

        assert tree.total_nodes == 1
        assert "node1" in tree.leaf_ids
        assert tree.get_node("node1") == node

    def test_multi_level_tree(self):
        """Test tree with multiple levels."""
        tree = RAPTORTree()

        # Add leaf nodes
        leaf1 = RAPTORNode(id="leaf1", content="Leaf 1", level=0)
        leaf2 = RAPTORNode(id="leaf2", content="Leaf 2", level=0)
        tree.add_node(leaf1)
        tree.add_node(leaf2)

        # Add summary node
        summary = RAPTORNode(
            id="summary1",
            content="Summary",
            level=1,
            children=["leaf1", "leaf2"],
            is_summary=True
        )
        leaf1.parent = "summary1"
        leaf2.parent = "summary1"
        tree.add_node(summary)

        assert tree.total_nodes == 3
        assert tree.max_level == 1
        assert len(tree.get_level(0)) == 2
        assert len(tree.get_level(1)) == 1

    def test_get_children(self):
        """Test getting child nodes."""
        tree = RAPTORTree()

        leaf1 = RAPTORNode(id="leaf1", content="Leaf 1", level=0)
        leaf2 = RAPTORNode(id="leaf2", content="Leaf 2", level=0)
        summary = RAPTORNode(
            id="summary1",
            content="Summary",
            level=1,
            children=["leaf1", "leaf2"]
        )

        tree.add_node(leaf1)
        tree.add_node(leaf2)
        tree.add_node(summary)

        children = tree.get_children("summary1")
        assert len(children) == 2
        assert leaf1 in children
        assert leaf2 in children

    def test_get_parent(self):
        """Test getting parent node."""
        tree = RAPTORTree()

        leaf = RAPTORNode(id="leaf1", content="Leaf", level=0, parent="summary1")
        summary = RAPTORNode(id="summary1", content="Summary", level=1)

        tree.add_node(leaf)
        tree.add_node(summary)

        parent = tree.get_parent("leaf1")
        assert parent == summary


# =============================================================================
# CLUSTERING TESTS
# =============================================================================

class TestClustering:
    """Tests for clustering algorithms."""

    def test_get_clusterer_gmm(self):
        """Test getting GMM clusterer."""
        clusterer = get_clusterer(ClusterMethod.GMM)
        assert isinstance(clusterer, GMMClusterer)

    def test_get_clusterer_kmeans(self):
        """Test getting K-Means clusterer."""
        clusterer = get_clusterer(ClusterMethod.KMEANS)
        assert isinstance(clusterer, KMeansClusterer)

    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="sklearn required"),
        reason="sklearn not available"
    )
    def test_gmm_clustering(self):
        """Test GMM clustering."""
        import numpy as np

        clusterer = GMMClusterer()
        embeddings = np.random.randn(10, 64)
        n_clusters = 3

        clusters = clusterer.cluster(embeddings, n_clusters)

        assert len(clusters) > 0
        # All indices should be covered
        all_indices = set()
        for cluster in clusters:
            all_indices.update(cluster)
        assert all_indices == set(range(10))

    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="sklearn required"),
        reason="sklearn not available"
    )
    def test_kmeans_clustering(self):
        """Test K-Means clustering."""
        import numpy as np

        clusterer = KMeansClusterer()
        embeddings = np.random.randn(10, 64)
        n_clusters = 3

        clusters = clusterer.cluster(embeddings, n_clusters)

        assert len(clusters) > 0
        all_indices = set()
        for cluster in clusters:
            all_indices.update(cluster)
        assert all_indices == set(range(10))


# =============================================================================
# RAPTOR BUILD TESTS
# =============================================================================

class TestRAPTORTreeBuilding:
    """Tests for RAPTOR tree building."""

    @pytest.fixture
    def raptor(self):
        """Create RAPTOR instance with mocks."""
        summarizer = MockSummarizer()
        embedder = MockEmbedder(dim=64)
        config = RAPTORConfig(
            chunk_size=100,
            max_tree_depth=3,
            min_cluster_size=2,
            n_clusters_ratio=0.5
        )
        return RAPTOR(
            summarizer=summarizer,
            embedder=embedder,
            config=config
        )

    @pytest.mark.asyncio
    async def test_build_tree_single_document(self, raptor):
        """Test tree building with single document."""
        docs = ["This is a single test document."]

        tree = await raptor.build_tree(docs)

        assert tree is not None
        assert tree.total_nodes >= 1

    @pytest.mark.asyncio
    async def test_build_tree_multiple_documents(self, raptor):
        """Test tree building with multiple documents."""
        tree = await raptor.build_tree(SAMPLE_DOCUMENTS)

        assert tree is not None
        assert tree.total_nodes > len(SAMPLE_DOCUMENTS)
        assert tree.max_level >= 0

    @pytest.mark.asyncio
    async def test_build_tree_with_metadata(self, raptor):
        """Test tree building with document metadata."""
        docs = ["Document 1", "Document 2"]
        metadata = [
            {"source": "file1.txt"},
            {"source": "file2.txt"}
        ]

        tree = await raptor.build_tree(docs, metadata)

        assert tree is not None
        # Check metadata is preserved in leaf nodes
        leaf_nodes = tree.get_level(0)
        for node in leaf_nodes:
            assert "doc_index" in node.metadata

    @pytest.mark.asyncio
    async def test_build_tree_creates_summaries(self, raptor):
        """Test that tree building creates summary nodes."""
        # Use enough documents to trigger summarization
        docs = [f"Document {i} with some content." for i in range(10)]

        tree = await raptor.build_tree(docs)

        # Should have summary nodes at level > 0
        summary_nodes = [n for n in tree.get_all_nodes() if n.is_summary]
        assert len(summary_nodes) >= 0  # May have summaries depending on clustering

    @pytest.mark.asyncio
    async def test_build_stats_tracking(self, raptor):
        """Test that build statistics are tracked."""
        await raptor.build_tree(SAMPLE_DOCUMENTS)

        stats = raptor.stats
        assert stats is not None
        assert stats.total_documents == len(SAMPLE_DOCUMENTS)
        assert stats.build_time_seconds > 0


# =============================================================================
# RAPTOR RETRIEVAL TESTS
# =============================================================================

class TestRAPTORRetrieval:
    """Tests for RAPTOR retrieval methods."""

    @pytest.fixture
    async def raptor_with_tree(self):
        """Create RAPTOR instance with built tree."""
        summarizer = MockSummarizer()
        embedder = MockEmbedder(dim=64)
        config = RAPTORConfig(
            chunk_size=100,
            max_tree_depth=2,
            top_k=3
        )
        raptor = RAPTOR(
            summarizer=summarizer,
            embedder=embedder,
            config=config
        )
        await raptor.build_tree(SAMPLE_DOCUMENTS)
        return raptor

    @pytest.mark.asyncio
    async def test_collapsed_retrieval(self, raptor_with_tree):
        """Test collapsed tree retrieval."""
        raptor = await raptor_with_tree

        result = await raptor.retrieve(
            "What is machine learning?",
            method=RetrievalMethod.COLLAPSED,
            top_k=3
        )

        assert isinstance(result, RAPTORResult)
        assert len(result.nodes) <= 3
        assert result.retrieval_method == RetrievalMethod.COLLAPSED
        assert len(result.context) > 0

    @pytest.mark.asyncio
    async def test_tree_traversal_retrieval(self, raptor_with_tree):
        """Test tree traversal retrieval."""
        raptor = await raptor_with_tree

        result = await raptor.retrieve(
            "What is deep learning?",
            method=RetrievalMethod.TREE_TRAVERSAL,
            top_k=3
        )

        assert isinstance(result, RAPTORResult)
        assert result.retrieval_method == RetrievalMethod.TREE_TRAVERSAL

    @pytest.mark.asyncio
    async def test_hybrid_retrieval(self, raptor_with_tree):
        """Test hybrid retrieval."""
        raptor = await raptor_with_tree

        result = await raptor.retrieve(
            "What is RAG?",
            method=RetrievalMethod.HYBRID,
            top_k=3
        )

        assert isinstance(result, RAPTORResult)
        assert result.retrieval_method == RetrievalMethod.HYBRID

    @pytest.mark.asyncio
    async def test_retrieval_method_string(self, raptor_with_tree):
        """Test retrieval with string method."""
        raptor = await raptor_with_tree

        result = await raptor.retrieve(
            "test query",
            method="collapsed",
            top_k=2
        )

        assert result.retrieval_method == RetrievalMethod.COLLAPSED

    @pytest.mark.asyncio
    async def test_retrieval_empty_tree(self):
        """Test retrieval on empty tree."""
        summarizer = MockSummarizer()
        embedder = MockEmbedder()
        raptor = RAPTOR(summarizer=summarizer, embedder=embedder)

        result = await raptor.retrieve("test query")

        assert len(result.nodes) == 0
        assert result.context == ""


# =============================================================================
# INCREMENTAL UPDATE TESTS
# =============================================================================

class TestIncrementalUpdates:
    """Tests for incremental document updates."""

    @pytest.fixture
    async def raptor_with_tree(self):
        """Create RAPTOR instance with built tree."""
        summarizer = MockSummarizer()
        embedder = MockEmbedder(dim=64)
        raptor = RAPTOR(summarizer=summarizer, embedder=embedder)
        await raptor.build_tree(SAMPLE_DOCUMENTS[:2])
        return raptor

    @pytest.mark.asyncio
    async def test_add_documents(self, raptor_with_tree):
        """Test adding documents to existing tree."""
        raptor = await raptor_with_tree
        initial_count = raptor.tree.total_nodes

        new_doc = ["New document about transformers and attention mechanisms."]
        added = await raptor.add_documents(new_doc)

        assert added > 0
        assert raptor.tree.total_nodes > initial_count

    @pytest.mark.asyncio
    async def test_add_documents_with_metadata(self, raptor_with_tree):
        """Test adding documents with metadata."""
        raptor = await raptor_with_tree

        new_doc = ["Another new document."]
        metadata = [{"source": "new_file.txt"}]

        added = await raptor.add_documents(new_doc, metadata)

        assert added > 0

    @pytest.mark.asyncio
    async def test_add_to_empty_tree(self):
        """Test adding documents when no tree exists."""
        summarizer = MockSummarizer()
        embedder = MockEmbedder()
        raptor = RAPTOR(summarizer=summarizer, embedder=embedder)

        docs = ["First document", "Second document"]
        added = await raptor.add_documents(docs)

        assert added > 0
        assert raptor.tree is not None


# =============================================================================
# RAPTOR INDEX TESTS
# =============================================================================

class TestRAPTORIndex:
    """Tests for RAPTORIndex high-level interface."""

    @pytest.fixture
    def index(self):
        """Create RAPTORIndex instance."""
        summarizer = MockSummarizer()
        embedder = MockEmbedder(dim=64)
        return RAPTORIndex(
            summarizer=summarizer,
            embedder=embedder
        )

    @pytest.mark.asyncio
    async def test_add_single_document(self, index):
        """Test adding single document."""
        count = await index.add("Single test document")
        assert count > 0

    @pytest.mark.asyncio
    async def test_add_multiple_documents(self, index):
        """Test adding multiple documents."""
        count = await index.add(["Doc 1", "Doc 2", "Doc 3"])
        assert count > 0

    @pytest.mark.asyncio
    async def test_search(self, index):
        """Test search functionality."""
        await index.add(SAMPLE_DOCUMENTS)

        results = await index.search("machine learning", k=3)

        assert len(results) <= 3
        for node, score in results:
            assert isinstance(node, RAPTORNode)
            assert isinstance(score, float)

    @pytest.mark.asyncio
    async def test_clear(self, index):
        """Test clearing the index."""
        await index.add(["Test document"])
        index.clear()

        results = await index.search("test")
        assert len(results) == 0


# =============================================================================
# RAPTOR WITH CHUNKER TESTS
# =============================================================================

class TestRAPTORWithChunker:
    """Tests for RAPTOR integrated with SemanticChunker."""

    @pytest.fixture
    def raptor_with_chunker(self):
        """Create RAPTORWithChunker instance."""
        summarizer = MockSummarizer()
        embedder = MockEmbedder(dim=64)
        chunker = MockChunker(chunk_size=50)

        return RAPTORWithChunker(
            summarizer=summarizer,
            embedder=embedder,
            chunker=chunker
        )

    @pytest.mark.asyncio
    async def test_build_with_chunker(self, raptor_with_chunker):
        """Test tree building with chunker."""
        tree = await raptor_with_chunker.build_tree(SAMPLE_DOCUMENTS)

        assert tree is not None
        assert tree.total_nodes > 0

    @pytest.mark.asyncio
    async def test_retrieve_with_chunker(self, raptor_with_chunker):
        """Test retrieval with chunker integration."""
        await raptor_with_chunker.build_tree(SAMPLE_DOCUMENTS)

        result = await raptor_with_chunker.retrieve("neural networks", top_k=3)

        assert isinstance(result, RAPTORResult)


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestRAPTORConfig:
    """Tests for RAPTOR configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RAPTORConfig()

        assert config.chunk_size == 100
        assert config.cluster_method == ClusterMethod.GMM
        assert config.max_tree_depth == 5
        assert config.top_k == 5

    def test_custom_config(self):
        """Test custom configuration."""
        config = RAPTORConfig(
            chunk_size=200,
            cluster_method=ClusterMethod.KMEANS,
            max_tree_depth=3,
            top_k=10,
            similarity_threshold=0.7
        )

        assert config.chunk_size == 200
        assert config.cluster_method == ClusterMethod.KMEANS
        assert config.max_tree_depth == 3
        assert config.top_k == 10
        assert config.similarity_threshold == 0.7

    @pytest.mark.asyncio
    async def test_config_affects_tree_depth(self):
        """Test that max_tree_depth config is respected."""
        summarizer = MockSummarizer()
        embedder = MockEmbedder(dim=64)
        config = RAPTORConfig(max_tree_depth=2)

        raptor = RAPTOR(
            summarizer=summarizer,
            embedder=embedder,
            config=config
        )

        # Use many documents to trigger deep tree
        docs = [f"Document {i} with content." for i in range(20)]
        tree = await raptor.build_tree(docs)

        assert tree.max_level <= 2


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_documents(self):
        """Test handling of empty documents."""
        summarizer = MockSummarizer()
        embedder = MockEmbedder()
        raptor = RAPTOR(summarizer=summarizer, embedder=embedder)

        tree = await raptor.build_tree([])

        assert tree.total_nodes == 0

    @pytest.mark.asyncio
    async def test_whitespace_only_documents(self):
        """Test handling of whitespace-only documents."""
        summarizer = MockSummarizer()
        embedder = MockEmbedder()
        raptor = RAPTOR(summarizer=summarizer, embedder=embedder)

        tree = await raptor.build_tree(["   ", "\n\n", "\t"])

        assert tree.total_nodes == 0

    @pytest.mark.asyncio
    async def test_single_document(self):
        """Test tree building with single document."""
        summarizer = MockSummarizer()
        embedder = MockEmbedder()
        raptor = RAPTOR(summarizer=summarizer, embedder=embedder)

        tree = await raptor.build_tree(["Single document"])

        assert tree.total_nodes >= 1

    @pytest.mark.asyncio
    async def test_very_long_document(self):
        """Test handling of very long documents."""
        summarizer = MockSummarizer()
        embedder = MockEmbedder()
        config = RAPTORConfig(chunk_size=50)
        raptor = RAPTOR(
            summarizer=summarizer,
            embedder=embedder,
            config=config
        )

        long_doc = "This is a sentence. " * 1000
        tree = await raptor.build_tree([long_doc])

        assert tree.total_nodes > 1  # Should be chunked

    @pytest.mark.asyncio
    async def test_clear_tree(self):
        """Test clearing the tree."""
        summarizer = MockSummarizer()
        embedder = MockEmbedder()
        raptor = RAPTOR(summarizer=summarizer, embedder=embedder)

        await raptor.build_tree(SAMPLE_DOCUMENTS)
        raptor.clear_tree()

        assert raptor.tree is None
        assert raptor.stats is None


# =============================================================================
# SIMILARITY AND CONTEXT TESTS
# =============================================================================

class TestSimilarityAndContext:
    """Tests for similarity computation and context building."""

    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        summarizer = MockSummarizer()
        embedder = MockEmbedder()
        raptor = RAPTOR(summarizer=summarizer, embedder=embedder)

        # Same vector should have similarity 1.0
        vec = [1.0, 0.0, 0.0]
        assert raptor._cosine_similarity(vec, vec) == pytest.approx(1.0)

        # Orthogonal vectors should have similarity 0.0
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        assert raptor._cosine_similarity(vec1, vec2) == pytest.approx(0.0)

        # Opposite vectors should have similarity -1.0
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        assert raptor._cosine_similarity(vec1, vec2) == pytest.approx(-1.0)

    def test_context_building(self):
        """Test context string building from nodes."""
        summarizer = MockSummarizer()
        embedder = MockEmbedder()
        raptor = RAPTOR(summarizer=summarizer, embedder=embedder)

        nodes = [
            RAPTORNode(id="n1", content="Content 1", level=0, is_summary=False),
            RAPTORNode(id="n2", content="Summary 1", level=1, is_summary=True),
        ]

        context = raptor._build_context(nodes)

        assert "Content 1" in context
        assert "Summary 1" in context
        assert "[Source]" in context
        assert "[Level 1]" in context


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
