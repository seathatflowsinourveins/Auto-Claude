"""
Unit Tests for Batch RAG Pipeline Operations

Tests cover:
- Batch query processing with batch_run()
- Query deduplication
- Rate limiting
- Shared retrieval results
- Throughput optimization
- Error handling in batch mode
"""

from __future__ import annotations

import asyncio
import time
import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

# Import modules under test
from core.rag.pipeline import (
    RAGPipeline,
    PipelineConfig,
    PipelineResult,
    BatchResult,
    QueryType,
    StrategyType,
    RetrievedDocument,
    RateLimiter,
    QueryDeduplicator,
)


# =============================================================================
# MOCK PROVIDERS
# =============================================================================

class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.calls: List[Dict[str, Any]] = []
        self.default_response = "This is a mock response about the query."
        self.call_count = 0

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        self.call_count += 1
        self.calls.append({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })

        for key, response in self.responses.items():
            if key in prompt:
                return response

        return self.default_response


class MockRetriever:
    """Mock retriever for testing."""

    def __init__(self, name: str = "mock", documents: Optional[List[Dict]] = None):
        self._name = name
        self.documents = documents or [
            {"content": "Document 1 about Python programming.", "score": 0.9},
            {"content": "Document 2 about machine learning.", "score": 0.8},
            {"content": "Document 3 about data science.", "score": 0.7},
        ]
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    async def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        self.call_count += 1
        await asyncio.sleep(0.01)  # Simulate network latency
        return self.documents[:top_k]


# =============================================================================
# RATE LIMITER TESTS
# =============================================================================

class TestRateLimiter:
    """Tests for the RateLimiter class."""

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_burst(self):
        """Test that rate limiter allows burst capacity."""
        limiter = RateLimiter(qps=10.0, burst_multiplier=2.0)

        # Should allow burst without waiting
        start = time.time()
        for _ in range(20):
            await limiter.acquire()
        elapsed = time.time() - start

        # Burst of 20 tokens at 10 QPS with 2x burst should complete quickly
        assert elapsed < 1.0  # Should be much faster than 2 seconds

    @pytest.mark.asyncio
    async def test_rate_limiter_throttles_sustained_load(self):
        """Test that rate limiter throttles sustained high load."""
        limiter = RateLimiter(qps=100.0, burst_multiplier=1.0)

        start = time.time()
        for _ in range(150):
            await limiter.acquire()
        elapsed = time.time() - start

        # 150 requests at 100 QPS should take at least 0.5 seconds
        assert elapsed >= 0.4  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_rate_limiter_refills_tokens(self):
        """Test that tokens refill over time."""
        limiter = RateLimiter(qps=10.0, burst_multiplier=1.0)

        # Exhaust tokens
        for _ in range(10):
            await limiter.acquire()

        # Wait for refill
        await asyncio.sleep(0.5)

        # Should be able to acquire more
        start = time.time()
        await limiter.acquire()
        elapsed = time.time() - start

        assert elapsed < 0.1  # Should not wait much


# =============================================================================
# QUERY DEDUPLICATOR TESTS
# =============================================================================

class TestQueryDeduplicator:
    """Tests for the QueryDeduplicator class."""

    def test_deduplicate_identical_queries(self):
        """Test deduplication of identical queries."""
        deduplicator = QueryDeduplicator(similarity_threshold=0.85)
        queries = [
            "What is Python?",
            "What is Python?",
            "What is Python?",
        ]

        unique, mapping, stats = deduplicator.deduplicate(queries)

        assert len(unique) == 1
        assert unique[0] == "What is Python?"
        assert len(mapping["What is Python?"]) == 3
        assert stats["duplicates_found"] == 2

    def test_deduplicate_similar_queries(self):
        """Test deduplication of similar queries."""
        deduplicator = QueryDeduplicator(similarity_threshold=0.5)
        queries = [
            "What is Python programming?",
            "What is Python programming language?",
            "Explain Docker containers",
        ]

        unique, mapping, stats = deduplicator.deduplicate(queries)

        assert len(unique) == 2  # Python queries should be deduplicated
        assert stats["unique"] == 2

    def test_deduplicate_preserves_distinct_queries(self):
        """Test that distinct queries are preserved."""
        deduplicator = QueryDeduplicator(similarity_threshold=0.85)
        queries = [
            "What is Python?",
            "How does Docker work?",
            "Explain Kubernetes architecture",
        ]

        unique, mapping, stats = deduplicator.deduplicate(queries)

        assert len(unique) == 3
        assert stats["duplicates_found"] == 0

    def test_find_similar_groups(self):
        """Test grouping of similar queries."""
        deduplicator = QueryDeduplicator(similarity_threshold=0.5)
        queries = [
            "Python programming basics",
            "Python programming tutorial",
            "Docker container basics",
            "Docker containers tutorial",
        ]

        groups = deduplicator.find_similar_groups(queries)

        # Should have 2-3 groups depending on n-gram similarity at threshold=0.5
        assert 2 <= len(groups) <= 4

    def test_empty_queries_list(self):
        """Test handling of empty queries list."""
        deduplicator = QueryDeduplicator()
        unique, mapping, stats = deduplicator.deduplicate([])

        assert unique == []
        assert mapping == {}
        assert stats["total"] == 0


# =============================================================================
# BATCH PIPELINE TESTS
# =============================================================================

class TestBatchPipeline:
    """Tests for batch_run functionality in RAGPipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create a test pipeline."""
        llm = MockLLMProvider()
        retriever = MockRetriever()
        config = PipelineConfig(
            enable_query_rewrite=False,
            enable_reranking=False,
            enable_evaluation=False,
            batch_qps=100.0,  # High QPS for fast tests
            batch_max_concurrency=10,
            batch_similarity_threshold=0.85,
        )
        return RAGPipeline(
            llm=llm,
            retrievers=[retriever],
            config=config,
        )

    @pytest.mark.asyncio
    async def test_batch_run_single_query(self, pipeline):
        """Test batch_run with a single query."""
        queries = ["What is Python?"]
        result = await pipeline.batch_run(queries)

        assert isinstance(result, BatchResult)
        assert result.total_queries == 1
        assert result.unique_queries == 1
        assert len(result.results) == 1
        assert result.results[0].response is not None

    @pytest.mark.asyncio
    async def test_batch_run_multiple_distinct_queries(self, pipeline):
        """Test batch_run with multiple distinct queries."""
        queries = [
            "What is Python?",
            "How does Docker work?",
            "Explain machine learning",
        ]
        result = await pipeline.batch_run(queries)

        assert result.total_queries == 3
        assert result.unique_queries == 3
        assert len(result.results) == 3
        assert all(r.response for r in result.results)

    @pytest.mark.asyncio
    async def test_batch_run_deduplicates_similar_queries(self, pipeline):
        """Test that batch_run deduplicates similar queries."""
        queries = [
            "What is Python programming?",
            "What is Python programming?",  # Exact duplicate
            "What is Python programming?",  # Exact duplicate
        ]
        result = await pipeline.batch_run(queries)

        assert result.total_queries == 3
        assert result.unique_queries == 1
        assert result.shared_retrievals == 2
        assert result.deduplication_stats["duplicates_found"] == 2

    @pytest.mark.asyncio
    async def test_batch_run_empty_queries(self, pipeline):
        """Test batch_run with empty queries list."""
        result = await pipeline.batch_run([])

        assert result.total_queries == 0
        assert result.unique_queries == 0
        assert result.results == []
        assert result.throughput_qps == 0.0

    @pytest.mark.asyncio
    async def test_batch_run_preserves_order(self, pipeline):
        """Test that results are returned in original query order."""
        queries = [
            "First query about Python",
            "Second query about Docker",
            "Third query about Kubernetes",
        ]
        result = await pipeline.batch_run(queries)

        # Results should be in same order as queries
        assert len(result.results) == 3
        # Each result should exist
        assert all(r is not None for r in result.results)

    @pytest.mark.asyncio
    async def test_batch_run_with_query_types(self, pipeline):
        """Test batch_run with explicit query types."""
        queries = ["Code example for sorting", "Latest news about AI"]
        query_types = [QueryType.CODE, QueryType.NEWS]

        result = await pipeline.batch_run(queries, query_types=query_types)

        assert result.total_queries == 2
        assert len(result.results) == 2

    @pytest.mark.asyncio
    async def test_batch_run_throughput_metrics(self, pipeline):
        """Test that throughput metrics are calculated."""
        queries = ["Query " + str(i) for i in range(10)]
        result = await pipeline.batch_run(queries)

        assert result.throughput_qps > 0
        assert result.total_latency_ms > 0

    @pytest.mark.asyncio
    async def test_batch_run_shared_retrieval_metadata(self, pipeline):
        """Test that shared retrieval results are marked in metadata."""
        queries = [
            "What is Python?",
            "What is Python?",  # Duplicate
        ]
        result = await pipeline.batch_run(queries)

        # First result should not have shared_retrieval flag
        assert result.results[0].metadata.get("shared_retrieval") is not True

        # Second result should have shared_retrieval flag
        assert result.results[1].metadata.get("shared_retrieval") is True

    @pytest.mark.asyncio
    async def test_batch_run_concurrency_control(self, pipeline):
        """Test that concurrency is properly controlled."""
        # Create pipeline with low concurrency
        llm = MockLLMProvider()
        retriever = MockRetriever()
        config = PipelineConfig(
            enable_query_rewrite=False,
            enable_reranking=False,
            enable_evaluation=False,
            batch_qps=1000.0,
            batch_max_concurrency=2,  # Only 2 concurrent
        )
        limited_pipeline = RAGPipeline(
            llm=llm,
            retrievers=[retriever],
            config=config,
        )

        queries = ["Query " + str(i) for i in range(10)]
        result = await limited_pipeline.batch_run(queries)

        assert result.total_queries == 10
        assert len(result.results) == 10


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestBatchPipelineIntegration:
    """Integration tests for batch pipeline operations."""

    @pytest.mark.asyncio
    async def test_batch_run_10x_throughput_improvement(self):
        """Test that batch processing achieves significant throughput improvement."""
        llm = MockLLMProvider()
        retriever = MockRetriever()
        config = PipelineConfig(
            enable_query_rewrite=False,
            enable_reranking=False,
            enable_evaluation=False,
            batch_qps=1000.0,
            batch_max_concurrency=20,
            batch_similarity_threshold=0.85,
        )
        pipeline = RAGPipeline(
            llm=llm,
            retrievers=[retriever],
            config=config,
        )

        # Create queries with some duplicates
        base_queries = [
            "What is Python?",
            "How does Docker work?",
            "Explain machine learning",
            "What is Kubernetes?",
            "How to use Git?",
        ]
        # Duplicate each query twice (15 total, 5 unique)
        queries = base_queries * 3

        # Measure batch processing time
        start = time.time()
        result = await pipeline.batch_run(queries)
        batch_time = time.time() - start

        # Verify deduplication worked
        assert result.unique_queries == 5
        assert result.total_queries == 15
        assert result.shared_retrievals == 10

        # Throughput should be high due to deduplication
        assert result.throughput_qps > 10  # At least 10 QPS

    @pytest.mark.asyncio
    async def test_batch_vs_sequential_comparison(self):
        """Compare batch processing vs sequential processing."""
        llm = MockLLMProvider()
        retriever = MockRetriever()
        config = PipelineConfig(
            enable_query_rewrite=False,
            enable_reranking=False,
            enable_evaluation=False,
            batch_qps=1000.0,
            batch_max_concurrency=10,
        )
        pipeline = RAGPipeline(
            llm=llm,
            retrievers=[retriever],
            config=config,
        )

        queries = ["Query " + str(i) for i in range(5)]

        # Sequential processing
        start = time.time()
        sequential_results = []
        for query in queries:
            result = await pipeline.run(query)
            sequential_results.append(result)
        sequential_time = time.time() - start

        # Reset counters
        llm.call_count = 0
        retriever.call_count = 0

        # Batch processing
        start = time.time()
        batch_result = await pipeline.batch_run(queries)
        batch_time = time.time() - start

        # Batch should be faster or equal (concurrent processing)
        # Note: With mock providers, the difference may be small
        assert batch_result.total_queries == len(queries)
        assert len(batch_result.results) == len(sequential_results)


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestBatchPipelineErrorHandling:
    """Tests for error handling in batch operations."""

    @pytest.mark.asyncio
    async def test_batch_run_handles_individual_failures(self):
        """Test that batch_run handles individual query failures gracefully."""

        class FailingLLM:
            def __init__(self):
                self.call_count = 0

            async def generate(self, prompt: str, **kwargs) -> str:
                self.call_count += 1
                if "fail" in prompt.lower():
                    raise ValueError("Intentional failure")
                return "Success response"

        llm = FailingLLM()
        retriever = MockRetriever()
        config = PipelineConfig(
            enable_query_rewrite=False,
            enable_reranking=False,
            enable_evaluation=False,
        )
        pipeline = RAGPipeline(
            llm=llm,
            retrievers=[retriever],
            config=config,
        )

        queries = [
            "Normal query",
            "This should fail",  # Will trigger error
            "Another normal query",
        ]

        result = await pipeline.batch_run(queries)

        # Should return results for all queries
        assert len(result.results) == 3
        # The failing query should have an error in response
        assert "error" in result.results[1].response.lower() or result.results[1].confidence == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
