#!/usr/bin/env python3
"""
Tests for Contextual Retrieval Module

Tests Anthropic's contextual retrieval pattern implementation including:
- Document analysis and context extraction
- LLM and rule-based context generation
- Cache functionality (memory and disk)
- SemanticChunker integration
- Embedding generation with context
"""

import asyncio
import hashlib
import os
import shutil
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import the module under test
from core.rag.contextual_retrieval import (
    ContextualRetriever,
    ContextualSemanticChunker,
    ContextualRetrievalConfig,
    ContextGenerationStrategy,
    ContextualizedChunk,
    DocumentContext,
    ContextualizationStats,
    ContextCache,
    LLMContextGenerator,
    RuleBasedContextGenerator,
    ContextPrompts,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_document():
    """Sample document for testing."""
    return """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems
to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning

In supervised learning, the algorithm learns from labeled training data.
The model makes predictions based on the input features and compares them
to the known correct outputs.

```python
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model
```

### Unsupervised Learning

Unsupervised learning works with unlabeled data. The algorithm tries to
find hidden patterns or intrinsic structures in the input data.

## Applications

Machine learning has numerous applications:
- Healthcare: Disease diagnosis
- Finance: Fraud detection
- Transportation: Self-driving cars
"""


@pytest.fixture
def sample_chunk():
    """Sample chunk content."""
    return """### Supervised Learning

In supervised learning, the algorithm learns from labeled training data.
The model makes predictions based on the input features and compares them
to the known correct outputs.

```python
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model
```"""


@pytest.fixture
def mock_llm():
    """Mock LLM provider."""
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value="This chunk discusses supervised learning techniques.")
    return llm


@pytest.fixture
def mock_embedding_provider():
    """Mock embedding provider."""
    provider = MagicMock()
    provider.embed = MagicMock(return_value=[0.1] * 384)
    provider.embed_batch = MagicMock(return_value=[[0.1] * 384, [0.2] * 384])
    return provider


@pytest.fixture
def temp_cache_dir():
    """Temporary directory for cache testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def config(temp_cache_dir):
    """Test configuration."""
    return ContextualRetrievalConfig(
        strategy=ContextGenerationStrategy.HYBRID,
        max_context_tokens=100,
        batch_size=2,
        cache_enabled=True,
        cache_dir=temp_cache_dir,
        memory_cache_size=100,
        parallel_requests=2,
        timeout_seconds=10.0,
    )


# =============================================================================
# CONTEXT CACHE TESTS
# =============================================================================

class TestContextCache:
    """Tests for ContextCache class."""

    def test_cache_init(self, temp_cache_dir):
        """Test cache initialization."""
        cache = ContextCache(cache_dir=temp_cache_dir, memory_size=100)
        assert cache.cache_dir.exists() is False  # Not created until first use
        cache._init_db()
        assert cache.cache_dir.exists()
        cache.close()

    def test_cache_put_get(self, temp_cache_dir):
        """Test storing and retrieving from cache."""
        cache = ContextCache(cache_dir=temp_cache_dir, memory_size=100)

        chunk_content = "This is a test chunk about machine learning."
        doc_id = "doc-123"
        context = "This chunk discusses ML concepts."

        # Store
        cache.put(chunk_content, doc_id, context)

        # Retrieve
        retrieved = cache.get(chunk_content, doc_id)
        assert retrieved == context

        cache.close()

    def test_cache_miss(self, temp_cache_dir):
        """Test cache miss behavior."""
        cache = ContextCache(cache_dir=temp_cache_dir, memory_size=100)

        result = cache.get("nonexistent chunk", "doc-999")
        assert result is None

        cache.close()

    def test_cache_memory_eviction(self, temp_cache_dir):
        """Test memory cache eviction."""
        cache = ContextCache(cache_dir=temp_cache_dir, memory_size=3)

        # Fill cache beyond capacity
        for i in range(5):
            cache.put(f"chunk_{i}", "doc", f"context_{i}")

        # Only 3 should be in memory
        assert len(cache._memory_cache) <= 3

        # But all should be retrievable from disk
        for i in range(5):
            result = cache.get(f"chunk_{i}", "doc")
            assert result == f"context_{i}"

        cache.close()

    def test_cache_invalidate_document(self, temp_cache_dir):
        """Test document invalidation."""
        cache = ContextCache(cache_dir=temp_cache_dir, memory_size=100)

        # Add entries for multiple documents
        cache.put("chunk1", "doc-A", "context1")
        cache.put("chunk2", "doc-A", "context2")
        cache.put("chunk3", "doc-B", "context3")

        # Invalidate doc-A
        count = cache.invalidate_document("doc-A")
        assert count == 2

        # doc-A entries should be gone
        assert cache.get("chunk1", "doc-A") is None
        assert cache.get("chunk2", "doc-A") is None

        # doc-B entry should remain
        assert cache.get("chunk3", "doc-B") == "context3"

        cache.close()

    def test_cache_stats(self, temp_cache_dir):
        """Test cache statistics."""
        cache = ContextCache(cache_dir=temp_cache_dir, memory_size=100)

        cache.put("chunk1", "doc1", "context1")
        cache.put("chunk2", "doc1", "context2")

        stats = cache.get_stats()
        assert stats["memory_entries"] == 2
        assert stats["disk_entries"] == 2
        assert stats["memory_capacity"] == 100

        cache.close()


# =============================================================================
# RULE-BASED CONTEXT GENERATOR TESTS
# =============================================================================

class TestRuleBasedContextGenerator:
    """Tests for RuleBasedContextGenerator."""

    def test_extract_header_context(self, sample_document, sample_chunk):
        """Test context extraction with headers."""
        generator = RuleBasedContextGenerator()
        doc_context = DocumentContext(
            doc_id="doc-1",
            title="ML Guide",
            topics=["machine learning", "AI"],
            doc_type="tutorial"
        )

        context = generator.extract_context(sample_chunk, sample_document, doc_context)

        assert "ML Guide" in context or "tutorial" in context
        assert len(context) > 0

    def test_extract_code_context(self):
        """Test context extraction from code."""
        generator = RuleBasedContextGenerator()
        doc_context = DocumentContext(doc_id="doc-1", title="Code Examples")

        code_chunk = """
def calculate_metrics(y_true, y_pred):
    '''Calculate classification metrics.'''
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy
"""
        document = f"# Metrics Module\n\n{code_chunk}"

        context = generator.extract_context(code_chunk, document, doc_context)

        # Should mention function
        assert "function" in context.lower() or "calculate_metrics" in context

    def test_extract_minimal_context(self):
        """Test context with minimal information."""
        generator = RuleBasedContextGenerator()
        doc_context = DocumentContext(doc_id="doc-1", title="Untitled")

        context = generator.extract_context("Just some text.", "Just some text.", doc_context)

        # Should still produce something
        assert len(context) > 0
        assert "Untitled" in context


# =============================================================================
# LLM CONTEXT GENERATOR TESTS
# =============================================================================

class TestLLMContextGenerator:
    """Tests for LLMContextGenerator."""

    @pytest.mark.asyncio
    async def test_analyze_document(self, mock_llm, config):
        """Test document analysis."""
        mock_llm.generate.return_value = '{"summary": "ML tutorial", "topics": ["ML", "AI"], "doc_type": "tutorial"}'

        generator = LLMContextGenerator(mock_llm, config)
        result = await generator.analyze_document("Some document content", "ML Guide")

        assert result.title == "ML Guide"
        assert "ML" in result.topics or "AI" in result.topics

    @pytest.mark.asyncio
    async def test_generate_chunk_context(self, mock_llm, config, sample_document, sample_chunk):
        """Test chunk context generation."""
        expected_context = "This chunk explains supervised learning with a code example."
        mock_llm.generate.return_value = expected_context

        generator = LLMContextGenerator(mock_llm, config)
        doc_context = DocumentContext(doc_id="doc-1", title="ML Guide")

        context = await generator.generate_chunk_context(
            sample_chunk, sample_document, doc_context
        )

        assert context == expected_context
        mock_llm.generate.assert_called()

    @pytest.mark.asyncio
    async def test_chunk_context_timeout(self, mock_llm, config):
        """Test timeout handling."""
        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(20)
            return "Too slow"

        mock_llm.generate = slow_generate

        generator = LLMContextGenerator(mock_llm, config)
        generator.config.timeout_seconds = 0.1

        doc_context = DocumentContext(doc_id="doc-1", title="Test")

        with pytest.raises(asyncio.TimeoutError):
            await generator.generate_chunk_context(
                "Test chunk", "Test document", doc_context
            )


# =============================================================================
# CONTEXTUAL RETRIEVER TESTS
# =============================================================================

class TestContextualRetriever:
    """Tests for ContextualRetriever class."""

    @pytest.mark.asyncio
    async def test_contextualize_document_hybrid(
        self, mock_llm, mock_embedding_provider, config, sample_document
    ):
        """Test full document contextualization with hybrid strategy."""
        mock_llm.generate.side_effect = [
            '{"summary": "ML tutorial", "topics": ["ML"], "doc_type": "tutorial"}',
            "Context for chunk 1",
            "Context for chunk 2",
            "Context for chunk 3",
            "Context for chunk 4",
            "Context for chunk 5",
        ]

        retriever = ContextualRetriever(
            llm=mock_llm,
            embedding_provider=mock_embedding_provider,
            config=config
        )

        chunks = await retriever.contextualize_document(
            document=sample_document,
            doc_title="ML Guide",
            doc_metadata={"source": "test"}
        )

        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, ContextualizedChunk)
            assert chunk.prepended_context
            assert chunk.contextualized_content
            assert chunk.original_content

        retriever.close()

    @pytest.mark.asyncio
    async def test_contextualize_document_rule_based(
        self, mock_embedding_provider, temp_cache_dir, sample_document
    ):
        """Test document contextualization with rule-based strategy."""
        config = ContextualRetrievalConfig(
            strategy=ContextGenerationStrategy.RULE_BASED,
            cache_dir=temp_cache_dir,
        )

        retriever = ContextualRetriever(
            llm=None,  # No LLM
            embedding_provider=mock_embedding_provider,
            config=config
        )

        chunks = await retriever.contextualize_document(
            document=sample_document,
            doc_title="ML Guide"
        )

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.prepended_context
            # Rule-based should mention the title
            assert "ML Guide" in chunk.prepended_context or "document" in chunk.prepended_context.lower()

        retriever.close()

    @pytest.mark.asyncio
    async def test_cache_hit_behavior(
        self, mock_llm, mock_embedding_provider, config, sample_document
    ):
        """Test that cache hits avoid LLM calls."""
        mock_llm.generate.side_effect = [
            '{"summary": "ML tutorial", "topics": ["ML"], "doc_type": "tutorial"}',
            "Context from LLM",
        ] * 10  # Enough for multiple calls

        retriever = ContextualRetriever(
            llm=mock_llm,
            embedding_provider=mock_embedding_provider,
            config=config
        )

        # First call - should use LLM
        chunks1 = await retriever.contextualize_document(
            document=sample_document,
            doc_title="ML Guide"
        )

        initial_call_count = mock_llm.generate.call_count

        # Second call with same document - should use cache
        chunks2 = await retriever.contextualize_document(
            document=sample_document,
            doc_title="ML Guide"
        )

        # LLM should not be called again for chunks (only for document analysis)
        assert mock_llm.generate.call_count <= initial_call_count + 1

        # Same number of chunks
        assert len(chunks1) == len(chunks2)

        # Second call should have cache hits
        cache_hits = sum(1 for c in chunks2 if c.cache_hit)
        assert cache_hits > 0

        retriever.close()

    @pytest.mark.asyncio
    async def test_contextualize_single_chunk(
        self, mock_llm, mock_embedding_provider, config, sample_document, sample_chunk
    ):
        """Test single chunk contextualization."""
        mock_llm.generate.side_effect = [
            '{"summary": "ML tutorial", "topics": ["ML"], "doc_type": "tutorial"}',
            "Context for supervised learning chunk",
        ]

        retriever = ContextualRetriever(
            llm=mock_llm,
            embedding_provider=mock_embedding_provider,
            config=config
        )

        chunk = await retriever.contextualize_chunk(
            chunk_content=sample_chunk,
            document=sample_document,
            doc_title="ML Guide"
        )

        assert isinstance(chunk, ContextualizedChunk)
        assert "supervised learning" in chunk.prepended_context.lower() or chunk.prepended_context

        retriever.close()

    def test_invalidate_cache(self, mock_llm, config):
        """Test cache invalidation."""
        retriever = ContextualRetriever(
            llm=mock_llm,
            config=config
        )

        # Pre-populate cache
        retriever._cache.put("chunk1", "doc-123", "context1")
        retriever._cache.put("chunk2", "doc-123", "context2")

        # Invalidate
        count = retriever.invalidate_cache("doc-123")
        assert count == 2

        # Verify invalidation
        assert retriever._cache.get("chunk1", "doc-123") is None

        retriever.close()

    def test_get_cache_stats(self, mock_llm, config):
        """Test cache statistics retrieval."""
        retriever = ContextualRetriever(
            llm=mock_llm,
            config=config
        )

        stats = retriever.get_cache_stats()
        assert "memory_entries" in stats
        assert "memory_capacity" in stats

        retriever.close()


# =============================================================================
# CONTEXTUAL SEMANTIC CHUNKER TESTS
# =============================================================================

class TestContextualSemanticChunker:
    """Tests for ContextualSemanticChunker integration."""

    @pytest.mark.asyncio
    async def test_chunk_with_context(
        self, mock_llm, mock_embedding_provider, temp_cache_dir, sample_document
    ):
        """Test integrated chunking with context."""
        mock_llm.generate.side_effect = [
            '{"summary": "ML tutorial", "topics": ["ML"], "doc_type": "tutorial"}',
        ] + ["Context for chunk"] * 20

        config = ContextualRetrievalConfig(
            cache_dir=temp_cache_dir,
            strategy=ContextGenerationStrategy.HYBRID,
        )

        chunker = ContextualSemanticChunker(
            llm=mock_llm,
            embedding_provider=mock_embedding_provider,
            chunker_config={"max_chunk_size": 200, "min_chunk_size": 50},
            retrieval_config=config
        )

        chunks = await chunker.chunk_with_context(
            text=sample_document,
            doc_title="ML Guide"
        )

        assert len(chunks) > 0

        stats = chunker.get_stats(chunks)
        assert stats["total_chunks"] == len(chunks)
        assert stats["avg_context_length"] > 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for contextual retrieval."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, mock_llm, mock_embedding_provider, temp_cache_dir):
        """Test complete pipeline from document to contextualized embeddings."""
        document = """# API Reference

## Authentication

All API requests require authentication using an API key.
Pass the key in the Authorization header.

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" https://api.example.com/v1/users
```

## Users Endpoint

### GET /users

Returns a list of all users.

### POST /users

Creates a new user. Requires name and email fields.
"""
        mock_llm.generate.side_effect = [
            '{"summary": "API Reference docs", "topics": ["API", "REST"], "doc_type": "api_docs"}',
        ] + ["Context about API " + str(i) for i in range(10)]

        config = ContextualRetrievalConfig(
            cache_dir=temp_cache_dir,
            strategy=ContextGenerationStrategy.HYBRID,
            max_context_tokens=50,
        )

        retriever = ContextualRetriever(
            llm=mock_llm,
            embedding_provider=mock_embedding_provider,
            config=config
        )

        # Process document
        chunks = await retriever.contextualize_document(
            document=document,
            doc_title="API Reference",
            doc_metadata={"version": "1.0", "type": "api"}
        )

        # Verify results
        assert len(chunks) > 0

        for chunk in chunks:
            # Each chunk should have context
            assert chunk.prepended_context
            # Context should be prepended
            assert chunk.contextualized_content.startswith(chunk.prepended_context)
            # Embeddings should be generated
            assert chunk.contextualized_embedding is not None
            # Document context should be set
            assert chunk.document_context is not None
            assert chunk.document_context.title == "API Reference"

        retriever.close()


# =============================================================================
# PROMPT TESTS
# =============================================================================

class TestContextPrompts:
    """Tests for prompt templates."""

    def test_document_analysis_prompt(self):
        """Test document analysis prompt formatting."""
        prompt = ContextPrompts.DOCUMENT_ANALYSIS.format(
            document_preview="Sample document content"
        )
        assert "Sample document content" in prompt
        assert "summary" in prompt.lower()
        assert "topics" in prompt.lower()

    def test_chunk_context_prompt(self):
        """Test chunk context prompt formatting."""
        prompt = ContextPrompts.CHUNK_CONTEXT.format(
            document_content="Full document here",
            chunk_content="Specific chunk content"
        )
        assert "Full document here" in prompt
        assert "Specific chunk content" in prompt
        assert "<document>" in prompt
        assert "<chunk>" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
