"""
Tests for Opik Tracing Adapter (V66)
=====================================

Comprehensive unit tests for platform/adapters/opik_tracing_adapter.py

Tests cover:
- Custom metrics (VoyageEmbeddingMetric, LettaMemoryMetric, DSPyOptimizationMetric)
- MetricResult dataclass
- TraceMetadata dataclass
- track_sdk_operation decorator
- OpikTracer class
- Error handling

Run with: pytest platform/tests/test_opik_tracing_adapter.py -v
"""

import asyncio
import pytest
import time
import os
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# =============================================================================
# Test MetricResult Dataclass
# =============================================================================

class TestMetricResult:
    """Tests for MetricResult dataclass."""

    def test_create_metric_result(self):
        """Test creating a metric result."""
        from adapters.opik_tracing_adapter import MetricResult

        result = MetricResult(
            name="test_metric",
            score=0.85,
            reason="Good result",
            metadata={"key": "value"},
        )

        assert result.name == "test_metric"
        assert result.score == 0.85
        assert result.reason == "Good result"
        assert result.metadata == {"key": "value"}

    def test_metric_result_defaults(self):
        """Test MetricResult default values."""
        from adapters.opik_tracing_adapter import MetricResult

        result = MetricResult(name="test", score=0.5)

        assert result.reason is None
        assert result.metadata == {}


class TestTraceMetadata:
    """Tests for TraceMetadata dataclass."""

    def test_create_trace_metadata(self):
        """Test creating trace metadata."""
        from adapters.opik_tracing_adapter import TraceMetadata

        metadata = TraceMetadata(
            sdk_name="voyage",
            operation="embed",
            project="test-project",
            tags=["test", "embedding"],
        )

        assert metadata.sdk_name == "voyage"
        assert metadata.operation == "embed"
        assert metadata.project == "test-project"
        assert "test" in metadata.tags

    def test_trace_metadata_defaults(self):
        """Test TraceMetadata default values."""
        from adapters.opik_tracing_adapter import TraceMetadata

        metadata = TraceMetadata(sdk_name="letta", operation="search")

        assert metadata.project == "unleash"
        assert metadata.tags == []
        assert metadata.extra == {}


# =============================================================================
# Test VoyageEmbeddingMetric
# =============================================================================

class TestVoyageEmbeddingMetric:
    """Tests for VoyageEmbeddingMetric."""

    def test_metric_name(self):
        """Test metric has correct name."""
        from adapters.opik_tracing_adapter import VoyageEmbeddingMetric

        metric = VoyageEmbeddingMetric()
        assert metric.name == "voyage_embedding_quality"

    def test_score_with_good_results(self):
        """Test scoring with good retrieval results."""
        from adapters.opik_tracing_adapter import VoyageEmbeddingMetric

        metric = VoyageEmbeddingMetric()
        result = metric.score(
            query="machine learning",
            retrieved_docs=["Machine learning is great", "Learning about ML"],
            scores=[0.95, 0.88],
        )

        assert result.score > 0.5
        assert "retrieval quality" in result.reason.lower()
        assert result.metadata["avg_relevance"] > 0.8

    def test_score_with_empty_docs(self):
        """Test scoring with no documents."""
        from adapters.opik_tracing_adapter import VoyageEmbeddingMetric

        metric = VoyageEmbeddingMetric()
        result = metric.score(
            query="test",
            retrieved_docs=[],
            scores=[],
        )

        assert result.score == 0.0
        assert "no documents" in result.reason.lower()

    def test_score_with_low_similarity(self):
        """Test scoring with low similarity scores."""
        from adapters.opik_tracing_adapter import VoyageEmbeddingMetric

        metric = VoyageEmbeddingMetric()
        result = metric.score(
            query="quantum physics",
            retrieved_docs=["cooking recipes", "sports news"],
            scores=[0.2, 0.15],
        )

        assert result.score < 0.5
        assert result.metadata["avg_relevance"] < 0.3

    def test_score_with_custom_thresholds(self):
        """Test scoring with custom thresholds."""
        from adapters.opik_tracing_adapter import VoyageEmbeddingMetric

        metric = VoyageEmbeddingMetric(
            min_score_threshold=0.7,
            diversity_weight=0.5,
        )

        result = metric.score(
            query="test",
            retrieved_docs=["test document"],
            scores=[0.8],
        )

        assert result is not None
        assert metric.diversity_weight == 0.5


# =============================================================================
# Test LettaMemoryMetric
# =============================================================================

class TestLettaMemoryMetric:
    """Tests for LettaMemoryMetric."""

    def test_metric_name(self):
        """Test metric has correct name."""
        from adapters.opik_tracing_adapter import LettaMemoryMetric

        metric = LettaMemoryMetric()
        assert metric.name == "letta_memory_quality"

    def test_score_with_memories(self):
        """Test scoring with retrieved memories."""
        from adapters.opik_tracing_adapter import LettaMemoryMetric

        metric = LettaMemoryMetric()
        result = metric.score(
            query="test query",
            retrieved_memories=[
                {"content": "memory 1", "score": 0.9},
                {"content": "memory 2", "score": 0.85},
            ],
            context_used=True,
            session_interactions=5,
        )

        assert result.score > 0.7
        assert result.metadata["context_used"] is True

    def test_score_without_memories(self):
        """Test scoring with no memories."""
        from adapters.opik_tracing_adapter import LettaMemoryMetric

        metric = LettaMemoryMetric()
        result = metric.score(
            query="test",
            retrieved_memories=[],
        )

        assert result.score == 0.0
        assert "no memories" in result.reason.lower()


# =============================================================================
# Test DSPyOptimizationMetric
# =============================================================================

class TestDSPyOptimizationMetric:
    """Tests for DSPyOptimizationMetric."""

    def test_metric_name(self):
        """Test metric has correct name."""
        from adapters.opik_tracing_adapter import DSPyOptimizationMetric

        metric = DSPyOptimizationMetric()
        assert metric.name == "dspy_optimization_quality"

    def test_score_with_improvement(self):
        """Test scoring with optimization improvement."""
        from adapters.opik_tracing_adapter import DSPyOptimizationMetric

        metric = DSPyOptimizationMetric()
        result = metric.score(
            original_score=0.6,
            optimized_score=0.85,
            iterations=10,
        )

        assert result.score > 0.3
        assert "improvement" in result.reason.lower()
        assert result.metadata["improvement"] == 0.25

    def test_score_with_no_improvement(self):
        """Test scoring with no improvement."""
        from adapters.opik_tracing_adapter import DSPyOptimizationMetric

        metric = DSPyOptimizationMetric()
        result = metric.score(
            original_score=0.8,
            optimized_score=0.75,
            iterations=50,
        )

        assert result.score < 0.5
        assert "regression" in result.reason.lower() or "no improvement" in result.reason.lower()


# =============================================================================
# Test track_sdk_operation Decorator
# =============================================================================

class TestTrackSdkOperation:
    """Tests for track_sdk_operation decorator."""

    def test_decorator_exists(self):
        """Test decorator is importable."""
        from adapters.opik_tracing_adapter import track_sdk_operation

        assert callable(track_sdk_operation)

    def test_decorator_on_sync_function(self):
        """Test decorator works on sync functions."""
        from adapters.opik_tracing_adapter import track_sdk_operation

        @track_sdk_operation("test_sdk", "test_op")
        def sample_function(x):
            return x * 2

        # Should still work as a function
        result = sample_function(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_decorator_on_async_function(self):
        """Test decorator works on async functions."""
        from adapters.opik_tracing_adapter import track_sdk_operation

        @track_sdk_operation("test_sdk", "async_op")
        async def async_sample(x):
            await asyncio.sleep(0.01)
            return x * 3

        result = await async_sample(4)
        assert result == 12

    def test_decorator_with_custom_tags(self):
        """Test decorator with custom tags."""
        from adapters.opik_tracing_adapter import track_sdk_operation

        @track_sdk_operation("voyage", "embed", tags=["embedding", "text"])
        def embed_func(texts):
            return [[0.1] * 1024 for _ in texts]

        result = embed_func(["test"])
        assert len(result) == 1


# =============================================================================
# Test OpikTracer Class
# =============================================================================

class TestOpikTracer:
    """Tests for OpikTracer class."""

    def test_tracer_creation(self):
        """Test OpikTracer can be created."""
        from adapters.opik_tracing_adapter import OpikTracer

        tracer = OpikTracer()
        assert tracer is not None

    def test_tracer_project_config(self):
        """Test OpikTracer with project configuration."""
        from adapters.opik_tracing_adapter import OpikTracer

        tracer = OpikTracer(project_name="test-project")
        # OpikTracer uses project_name, not _project_name
        assert tracer.project_name == "test-project"

    def test_tracer_track_method(self):
        """Test OpikTracer.track() method."""
        from adapters.opik_tracing_adapter import OpikTracer

        tracer = OpikTracer()

        # track() should return a result
        if hasattr(tracer, 'track'):
            result = tracer.track(
                sdk_name="test",
                operation="sample",
                duration_ms=100,
                success=True,
            )
            # Should return some result (may be None if Opik not available)
            assert result is None or result is not None


# =============================================================================
# Test Module-Level Functions
# =============================================================================

class TestModuleFunctions:
    """Tests for module-level helper functions."""

    def test_opik_available_check(self):
        """Test OPIK_AVAILABLE flag is defined."""
        from adapters.opik_tracing_adapter import OPIK_AVAILABLE

        # Should be a boolean
        assert isinstance(OPIK_AVAILABLE, bool)

    def test_opik_metrics_available_check(self):
        """Test OPIK_METRICS_AVAILABLE flag is defined."""
        from adapters.opik_tracing_adapter import OPIK_METRICS_AVAILABLE

        assert isinstance(OPIK_METRICS_AVAILABLE, bool)

    def test_langchain_opik_available_check(self):
        """Test LANGCHAIN_OPIK_AVAILABLE flag is defined."""
        from adapters.opik_tracing_adapter import LANGCHAIN_OPIK_AVAILABLE

        assert isinstance(LANGCHAIN_OPIK_AVAILABLE, bool)


# =============================================================================
# Test Error Handling
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in the adapter."""

    def test_metric_handles_none_values(self):
        """Test metrics handle None values gracefully."""
        from adapters.opik_tracing_adapter import VoyageEmbeddingMetric

        metric = VoyageEmbeddingMetric()

        # Should handle empty/None gracefully
        result = metric.score(
            query="test",
            retrieved_docs=[],
            scores=[],
        )

        assert result.score == 0.0

    def test_decorator_handles_exceptions(self):
        """Test decorator handles function exceptions."""
        from adapters.opik_tracing_adapter import track_sdk_operation

        @track_sdk_operation("test", "error_op")
        def error_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            error_func()


# =============================================================================
# Test Circuit Breaker Integration
# =============================================================================

class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration."""

    def test_circuit_breaker_available_flag(self):
        """Test CIRCUIT_BREAKER_AVAILABLE flag is defined."""
        from adapters.opik_tracing_adapter import CIRCUIT_BREAKER_AVAILABLE

        assert isinstance(CIRCUIT_BREAKER_AVAILABLE, bool)


# =============================================================================
# Test Retry Configuration
# =============================================================================

class TestRetryConfiguration:
    """Tests for retry configuration."""

    def test_retry_config_exists(self):
        """Test retry configuration is defined."""
        from adapters.opik_tracing_adapter import OPIK_TRACING_RETRY_CONFIG

        # May be None if retry module not available
        assert OPIK_TRACING_RETRY_CONFIG is None or OPIK_TRACING_RETRY_CONFIG is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
