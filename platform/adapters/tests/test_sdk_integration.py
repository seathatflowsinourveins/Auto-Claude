"""
SDK Integration Test Suite (V1.0)

Tests cross-compatibility of all SDK adapters:
- Letta-Voyage integration
- DSPy-Voyage retriever
- Opik tracing
- Temporal workflow activities

Tests verify:
1. Graceful degradation when SDKs unavailable
2. Correct data flow between adapters
3. Async/sync compatibility
4. Error handling and recovery
"""

import asyncio
import pytest
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_embedding_layer():
    """Mock EmbeddingLayer for tests without Voyage."""
    layer = AsyncMock()
    layer.embed = AsyncMock(return_value=MagicMock(
        embeddings=[[0.1] * 128 for _ in range(3)]
    ))
    layer.semantic_search = AsyncMock(return_value=[
        (0, 0.95, "doc1"),
        (1, 0.85, "doc2"),
    ])
    layer.hybrid_search = AsyncMock(return_value=[
        (0, 0.92, "doc1"),
        (1, 0.88, "doc2"),
    ])
    return layer


@pytest.fixture
def mock_qdrant_store():
    """Mock QdrantVectorStore for tests."""
    store = MagicMock()
    store.search = AsyncMock(return_value=[
        MagicMock(id="mem1", score=0.9, payload={"preview": "test memory"}),
    ])
    return store


@pytest.fixture
def sample_corpus():
    """Sample corpus for retriever tests."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning enables computers to learn from data.",
        "Natural language processing handles text understanding.",
        "Deep learning uses neural networks with many layers.",
        "Transformers revolutionized sequence-to-sequence modeling.",
    ]


@pytest.fixture
def sample_interactions():
    """Sample interactions for learning session tests."""
    return [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI..."},
        {"role": "user", "content": "How does it work?"},
    ]


# =============================================================================
# Availability Tests
# =============================================================================

class TestSDKAvailability:
    """Test SDK availability detection."""

    def test_letta_availability_flag_exists(self):
        """Letta availability flag should exist."""
        from adapters.letta_voyage_adapter import LETTA_AVAILABLE, VOYAGE_AVAILABLE
        assert isinstance(LETTA_AVAILABLE, bool)
        assert isinstance(VOYAGE_AVAILABLE, bool)

    def test_dspy_availability_flag_exists(self):
        """DSPy availability flag should exist."""
        from adapters.dspy_voyage_retriever import DSPY_AVAILABLE, VOYAGE_AVAILABLE
        assert isinstance(DSPY_AVAILABLE, bool)
        assert isinstance(VOYAGE_AVAILABLE, bool)

    def test_opik_availability_flag_exists(self):
        """Opik availability flag should exist."""
        from adapters.opik_tracing_adapter import OPIK_AVAILABLE, ANTHROPIC_AVAILABLE
        assert isinstance(OPIK_AVAILABLE, bool)
        assert isinstance(ANTHROPIC_AVAILABLE, bool)

    def test_temporal_availability_flag_exists(self):
        """Temporal availability flag should exist."""
        from adapters.temporal_workflow_activities import TEMPORAL_AVAILABLE
        assert isinstance(TEMPORAL_AVAILABLE, bool)


# =============================================================================
# Letta-Voyage Integration Tests
# =============================================================================

class TestLettaVoyageAdapter:
    """Test Letta-Voyage integration."""

    def test_context_snippet_creation(self):
        """Test ContextSnippet data class."""
        from adapters.letta_voyage_adapter import ContextSnippet

        snippet = ContextSnippet(
            id="test1",
            content="Test content",
            score=0.9,
            metadata={"key": "value"},
            source="voyage_memory",
        )

        assert snippet.id == "test1"
        assert snippet.score == 0.9
        assert "test1" in snippet.to_xml()
        assert "0.900" in snippet.to_xml()

    def test_memory_block_creation(self):
        """Test MemoryBlock data class."""
        from adapters.letta_voyage_adapter import MemoryBlock

        block = MemoryBlock(
            name="test_block",
            content="Test content for memory block",
            block_type="archival",
        )

        xml = block.to_xml()
        assert "test_block" in xml
        assert "archival" in xml
        assert "Test content" in xml

    def test_learning_session_tracking(self):
        """Test LearningSession interaction tracking."""
        from adapters.letta_voyage_adapter import LearningSession
        import time

        session = LearningSession(
            session_id="test-session",
            agent_name="test-agent",
            start_time=time.time(),
        )

        session.add_interaction("user", "Hello")
        session.add_interaction("assistant", "Hi there!")

        assert len(session.interactions) == 2
        summary = session.get_summary()
        assert "user: Hello" in summary
        assert "assistant: Hi" in summary

    @pytest.mark.asyncio
    async def test_voyage_context_client_format_blocks(self, mock_embedding_layer):
        """Test VoyageContextClient formatting."""
        from adapters.letta_voyage_adapter import VoyageContextClient, ContextSnippet

        # Create mock adapter
        mock_adapter = MagicMock()

        client = VoyageContextClient(mock_adapter)

        snippets = [
            ContextSnippet(id="1", content="Content 1", score=0.9),
            ContextSnippet(id="2", content="Content 2", score=0.8),
        ]

        formatted = client.format_as_blocks(snippets)

        assert "retrieved_context" in formatted
        assert 'count="2"' in formatted
        assert "Content 1" in formatted
        assert "Content 2" in formatted

    def test_empty_snippets_format(self):
        """Test formatting with no snippets."""
        from adapters.letta_voyage_adapter import VoyageContextClient

        mock_adapter = MagicMock()
        client = VoyageContextClient(mock_adapter)

        formatted = client.format_as_blocks([])
        assert formatted == ""


# =============================================================================
# DSPy-Voyage Retriever Tests
# =============================================================================

class TestDSPyVoyageRetriever:
    """Test DSPy-Voyage retriever integration."""

    def test_retrieved_passage_creation(self):
        """Test RetrievedPassage data class."""
        from adapters.dspy_voyage_retriever import RetrievedPassage

        passage = RetrievedPassage(
            text="Test passage content",
            score=0.95,
            index=0,
            metadata={"source": "test"},
        )

        assert passage.text == "Test passage content"
        assert passage.score == 0.95
        assert str(passage) == "Test passage content"

    def test_retriever_config_defaults(self):
        """Test RetrieverConfig default values."""
        from adapters.dspy_voyage_retriever import RetrieverConfig

        config = RetrieverConfig()

        assert config.model == "voyage-4-large"
        assert config.top_k == 5
        assert config.use_hybrid is True
        assert config.cache_enabled is True

    def test_retriever_config_custom(self):
        """Test RetrieverConfig custom values."""
        from adapters.dspy_voyage_retriever import RetrieverConfig

        config = RetrieverConfig(
            model="voyage-3-large",
            top_k=10,
            use_mmr=True,
            use_hybrid=False,
        )

        assert config.model == "voyage-3-large"
        assert config.top_k == 10
        assert config.use_mmr is True
        assert config.use_hybrid is False

    def test_voyage_embedder_callable(self):
        """Test VoyageEmbedder is callable for DSPy compatibility."""
        from adapters.dspy_voyage_retriever import VoyageEmbedder

        embedder = VoyageEmbedder(model="voyage-4-large")

        # Verify it's callable
        assert callable(embedder)
        assert hasattr(embedder, "embed_sync")
        assert hasattr(embedder, "__call__")


# =============================================================================
# Opik Tracing Tests
# =============================================================================

class TestOpikTracing:
    """Test Opik tracing integration."""

    def test_trace_metadata_creation(self):
        """Test TraceMetadata data class."""
        from adapters.opik_tracing_adapter import TraceMetadata

        metadata = TraceMetadata(
            sdk_name="voyage",
            operation="embed",
            project="unleash",
            tags=["production", "v1"],
        )

        assert metadata.sdk_name == "voyage"
        assert metadata.operation == "embed"
        assert "production" in metadata.tags

    def test_metric_result_creation(self):
        """Test MetricResult data class."""
        from adapters.opik_tracing_adapter import MetricResult

        result = MetricResult(
            name="test_metric",
            score=0.85,
            reason="Test passed",
            metadata={"key": "value"},
        )

        assert result.name == "test_metric"
        assert result.score == 0.85
        assert result.reason == "Test passed"

    def test_voyage_embedding_metric_empty_docs(self):
        """Test VoyageEmbeddingMetric with no documents."""
        from adapters.opik_tracing_adapter import VoyageEmbeddingMetric

        metric = VoyageEmbeddingMetric()
        result = metric.score(
            query="test query",
            retrieved_docs=[],
            scores=[],
        )

        assert result.score == 0.0
        assert "No documents" in result.reason

    def test_voyage_embedding_metric_scoring(self):
        """Test VoyageEmbeddingMetric scoring logic."""
        from adapters.opik_tracing_adapter import VoyageEmbeddingMetric

        metric = VoyageEmbeddingMetric()
        result = metric.score(
            query="machine learning",
            retrieved_docs=[
                "Machine learning is a subset of AI",
                "Learning algorithms process data",
            ],
            scores=[0.9, 0.8],
        )

        assert result.score > 0.5
        assert result.name == "voyage_embedding_quality"
        assert "avg_relevance" in result.metadata

    def test_letta_memory_metric(self):
        """Test LettaMemoryMetric scoring."""
        from adapters.opik_tracing_adapter import LettaMemoryMetric

        metric = LettaMemoryMetric()
        result = metric.score(
            query="test query",
            retrieved_memories=[
                {"score": 0.8},
                {"score": 0.7},
            ],
            context_used=True,
            session_interactions=5,
        )

        assert result.score > 0.7
        assert "2 results" in result.reason

    def test_dspy_optimization_metric(self):
        """Test DSPyOptimizationMetric scoring."""
        from adapters.opik_tracing_adapter import DSPyOptimizationMetric

        metric = DSPyOptimizationMetric()
        result = metric.score(
            original_score=0.6,
            optimized_score=0.85,
            iterations=50,
        )

        assert result.score > 0
        assert "improvement" in result.reason.lower()
        assert result.metadata["improvement"] == 0.25

    def test_opik_tracer_status(self):
        """Test OpikTracer status reporting."""
        from adapters.opik_tracing_adapter import OpikTracer

        tracer = OpikTracer(project_name="test-project")
        status = tracer.get_status()

        assert "opik_available" in status
        assert "configured" in status
        assert status["project"] == "test-project"

    def test_track_sdk_operation_decorator(self):
        """Test track_sdk_operation decorator works."""
        from adapters.opik_tracing_adapter import track_sdk_operation

        @track_sdk_operation("test", "operation")
        def sync_function(x: int) -> int:
            return x * 2

        @track_sdk_operation("test", "async_operation")
        async def async_function(x: int) -> int:
            return x * 2

        # Verify decorators don't break functions
        assert sync_function(5) == 10

        # Test async
        result = asyncio.run(async_function(5))
        assert result == 10


# =============================================================================
# Temporal Workflow Activity Tests
# =============================================================================

class TestTemporalWorkflowActivities:
    """Test Temporal workflow activities."""

    def test_activity_result_creation(self):
        """Test ActivityResult data class."""
        from adapters.temporal_workflow_activities import ActivityResult

        result = ActivityResult(
            success=True,
            data={"key": "value"},
            duration_ms=100.5,
            metadata={"operation": "embed"},
        )

        assert result.success is True
        assert result.data["key"] == "value"
        assert result.duration_ms == 100.5

    def test_embedding_activity_input(self):
        """Test EmbeddingActivityInput data class."""
        from adapters.temporal_workflow_activities import EmbeddingActivityInput

        input_data = EmbeddingActivityInput(
            texts=["text1", "text2"],
            model="voyage-4-large",
            input_type="document",
        )

        assert len(input_data.texts) == 2
        assert input_data.model == "voyage-4-large"

    def test_search_activity_input(self):
        """Test SearchActivityInput data class."""
        from adapters.temporal_workflow_activities import SearchActivityInput

        input_data = SearchActivityInput(
            query="test query",
            top_k=10,
            use_hybrid=True,
        )

        assert input_data.query == "test query"
        assert input_data.top_k == 10

    def test_memory_activity_input(self):
        """Test MemoryActivityInput data class."""
        from adapters.temporal_workflow_activities import MemoryActivityInput

        input_data = MemoryActivityInput(
            query="recall memories",
            top_k=5,
            memory_types=["memory", "skills"],
        )

        assert "memory" in input_data.memory_types
        assert input_data.min_score == 0.3

    def test_retry_policies_exist(self):
        """Test retry policies are defined."""
        from adapters.temporal_workflow_activities import (
            STANDARD_RETRY,
            CRITICAL_RETRY,
            LIGHT_RETRY,
            TEMPORAL_AVAILABLE,
        )

        if TEMPORAL_AVAILABLE:
            assert STANDARD_RETRY is not None
            assert CRITICAL_RETRY is not None
            assert LIGHT_RETRY is not None


# =============================================================================
# Cross-SDK Integration Tests
# =============================================================================

class TestCrossSDKIntegration:
    """Test integration between multiple SDKs."""

    def test_data_type_compatibility(self):
        """Test data types are compatible across SDKs."""
        from adapters.letta_voyage_adapter import ContextSnippet
        from adapters.dspy_voyage_retriever import RetrievedPassage
        from adapters.opik_tracing_adapter import MetricResult

        # Create instances
        snippet = ContextSnippet(id="1", content="test", score=0.9)
        passage = RetrievedPassage(text="test", score=0.9, index=0)
        metric = MetricResult(name="test", score=0.9)

        # All have score attribute
        assert hasattr(snippet, "score")
        assert hasattr(passage, "score")
        assert hasattr(metric, "score")

        # All are serializable (have simple types)
        assert isinstance(snippet.score, float)
        assert isinstance(passage.score, float)
        assert isinstance(metric.score, float)

    def test_opik_metrics_for_all_sdks(self):
        """Test Opik has metrics for all SDKs."""
        from adapters.opik_tracing_adapter import (
            VoyageEmbeddingMetric,
            LettaMemoryMetric,
            DSPyOptimizationMetric,
        )

        # All metrics have name and score method
        voyage_metric = VoyageEmbeddingMetric()
        letta_metric = LettaMemoryMetric()
        dspy_metric = DSPyOptimizationMetric()

        assert hasattr(voyage_metric, "name")
        assert hasattr(voyage_metric, "score")
        assert hasattr(letta_metric, "name")
        assert hasattr(letta_metric, "score")
        assert hasattr(dspy_metric, "name")
        assert hasattr(dspy_metric, "score")

    def test_adapter_registration_pattern(self):
        """Test all adapters use the same registration pattern."""
        from adapters import get_adapter_status

        status = get_adapter_status()

        # All adapters should be registered
        expected_adapters = [
            "letta_voyage",
            "dspy_voyage",
            "opik_tracing",
            "temporal_workflow",
        ]

        for adapter_name in expected_adapters:
            assert adapter_name in status, f"Missing adapter: {adapter_name}"


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling across SDKs."""

    def test_letta_adapter_not_initialized_error(self):
        """Test LettaVoyageAdapter raises when not initialized."""
        from adapters.letta_voyage_adapter import LettaVoyageAdapter

        adapter = LettaVoyageAdapter()

        with pytest.raises(RuntimeError, match="not initialized"):
            adapter.learning("test")

    def test_voyage_embedder_not_initialized_error(self):
        """Test VoyageEmbedder raises when not initialized."""
        from adapters.dspy_voyage_retriever import VoyageEmbedder

        embedder = VoyageEmbedder()

        with pytest.raises(RuntimeError, match="not initialized"):
            asyncio.run(embedder.embed(["test"]))

    def test_activity_result_error_state(self):
        """Test ActivityResult properly represents errors."""
        from adapters.temporal_workflow_activities import ActivityResult

        result = ActivityResult(
            success=False,
            error="Test error message",
            duration_ms=50.0,
        )

        assert result.success is False
        assert result.error == "Test error message"
        assert result.data is None


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Basic performance tests."""

    def test_snippet_xml_generation_speed(self):
        """Test XML generation is fast enough."""
        from adapters.letta_voyage_adapter import ContextSnippet
        import time

        snippets = [
            ContextSnippet(id=str(i), content=f"Content {i}", score=0.9)
            for i in range(100)
        ]

        start = time.time()
        for s in snippets:
            _ = s.to_xml()
        duration = time.time() - start

        # Should complete 100 XML generations in under 100ms
        assert duration < 0.1, f"XML generation too slow: {duration:.3f}s"

    def test_metric_scoring_speed(self):
        """Test metric scoring is fast."""
        from adapters.opik_tracing_adapter import VoyageEmbeddingMetric
        import time

        metric = VoyageEmbeddingMetric()
        docs = [f"Document {i} content" for i in range(50)]
        scores = [0.9 - i * 0.01 for i in range(50)]

        start = time.time()
        for _ in range(100):
            _ = metric.score(
                query="test query",
                retrieved_docs=docs,
                scores=scores,
            )
        duration = time.time() - start

        # 100 metric evaluations in under 100ms
        assert duration < 0.1, f"Metric scoring too slow: {duration:.3f}s"


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Test factory functions for creating adapters."""

    def test_letta_factory_function(self):
        """Test create_letta_voyage_adapter factory."""
        from adapters.letta_voyage_adapter import create_letta_voyage_adapter

        adapter = create_letta_voyage_adapter(
            qdrant_url="localhost:6333",
            auto_store=False,
        )

        assert adapter is not None
        assert adapter._auto_store is False

    def test_opik_get_tracer(self):
        """Test get_tracer singleton pattern."""
        from adapters.opik_tracing_adapter import get_tracer

        tracer1 = get_tracer()
        tracer2 = get_tracer()

        # Should return same instance
        assert tracer1 is tracer2


# =============================================================================
# Module Export Tests
# =============================================================================

class TestModuleExports:
    """Test module exports are complete."""

    def test_letta_exports(self):
        """Test letta_voyage_adapter exports."""
        from adapters.letta_voyage_adapter import __all__

        expected = [
            "LettaVoyageAdapter",
            "VoyageContextClient",
            "ContextSnippet",
            "MemoryBlock",
        ]
        for name in expected:
            assert name in __all__, f"Missing export: {name}"

    def test_dspy_exports(self):
        """Test dspy_voyage_retriever exports."""
        from adapters.dspy_voyage_retriever import __all__

        expected = [
            "VoyageEmbedder",
            "VoyageRetriever",
            "RetrieverConfig",
            "RetrievedPassage",
        ]
        for name in expected:
            assert name in __all__, f"Missing export: {name}"

    def test_opik_exports(self):
        """Test opik_tracing_adapter exports."""
        from adapters.opik_tracing_adapter import __all__

        expected = [
            "OpikTracer",
            "VoyageEmbeddingMetric",
            "track_sdk_operation",
            "get_tracer",
        ]
        for name in expected:
            assert name in __all__, f"Missing export: {name}"

    def test_temporal_exports(self):
        """Test temporal_workflow_activities exports."""
        from adapters.temporal_workflow_activities import __all__

        expected = [
            "ActivityResult",
            "RAGPipelineWorkflow",
            "LearningSessionWorkflow",
            "create_worker",
        ]
        for name in expected:
            assert name in __all__, f"Missing export: {name}"
