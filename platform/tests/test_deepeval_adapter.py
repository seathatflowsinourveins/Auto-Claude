"""
Test Suite for DeepEval Adapter (V65 Gap11)

Comprehensive tests for the DeepEval evaluation adapter covering:
- Core evaluation functionality
- Hallucination detection
- Heuristic fallback
- Circuit breaker behavior
- Pipeline integration (EvaluatorProtocol)

Total: 35+ tests
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

# Import the adapter
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.deepeval_adapter import (
    DeepEvalAdapter,
    DeepEvalMetricConfig,
    HeuristicDeepEvalEvaluator,
    DEEPEVAL_AVAILABLE,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def adapter():
    """Create a DeepEvalAdapter instance."""
    return DeepEvalAdapter()


@pytest.fixture
def initialized_adapter():
    """Create an initialized DeepEvalAdapter instance."""
    adapter = DeepEvalAdapter()
    asyncio.get_event_loop().run_until_complete(adapter.initialize({}))
    return adapter


@pytest.fixture
def heuristic_evaluator():
    """Create a HeuristicDeepEvalEvaluator instance."""
    return HeuristicDeepEvalEvaluator()


@pytest.fixture
def sample_data():
    """Sample data for evaluation tests."""
    return {
        "input": "What is machine learning?",
        "actual_output": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "retrieval_context": [
            "Machine learning is a subset of artificial intelligence.",
            "ML systems learn patterns from data without explicit programming.",
            "Deep learning is a type of machine learning using neural networks.",
        ],
        "expected_output": "Machine learning is a branch of AI that allows computers to learn from data.",
    }


# =============================================================================
# BASIC ADAPTER TESTS
# =============================================================================

class TestDeepEvalAdapterBasic:
    """Basic adapter functionality tests."""

    def test_adapter_creation(self, adapter):
        """Test adapter can be created."""
        assert adapter is not None
        assert adapter.sdk_name == "deepeval"
        assert adapter.layer == 5  # OBSERVABILITY

    def test_adapter_available_property(self, adapter):
        """Test available property reflects SDK status."""
        assert isinstance(adapter.available, bool)

    @pytest.mark.asyncio
    async def test_initialize_default_config(self, adapter):
        """Test initialization with default config."""
        result = await adapter.initialize({})
        assert result.success is True
        assert "model" in result.data
        assert "thresholds" in result.data
        assert result.data["thresholds"]["faithfulness"] == 0.7
        assert result.data["thresholds"]["relevancy"] == 0.7
        assert result.data["thresholds"]["hallucination"] == 0.3

    @pytest.mark.asyncio
    async def test_initialize_custom_config(self, adapter):
        """Test initialization with custom config."""
        config = {
            "faithfulness_threshold": 0.8,
            "relevancy_threshold": 0.8,
            "hallucination_threshold": 0.2,
            "use_claude": True,
            "model": "claude-3-5-sonnet-20241022",
        }
        result = await adapter.initialize(config)
        assert result.success is True
        assert result.data["use_claude"] is True
        assert result.data["thresholds"]["faithfulness"] == 0.8

    @pytest.mark.asyncio
    async def test_health_check(self, initialized_adapter):
        """Test health check returns valid data."""
        result = await initialized_adapter.health_check()
        assert result.success is True
        assert "status" in result.data
        assert "deepeval_available" in result.data
        assert "heuristic_fallback" in result.data
        assert result.data["heuristic_fallback"] is True

    @pytest.mark.asyncio
    async def test_shutdown(self, initialized_adapter):
        """Test adapter shutdown."""
        result = await initialized_adapter.shutdown()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_get_stats(self, initialized_adapter):
        """Test statistics retrieval."""
        result = await initialized_adapter.execute("get_stats")
        assert result.success is True
        assert "call_count" in result.data
        assert "success_count" in result.data
        assert "error_count" in result.data
        assert "avg_latency_ms" in result.data
        assert "deepeval_native" in result.data

    @pytest.mark.asyncio
    async def test_unknown_operation(self, initialized_adapter):
        """Test unknown operation returns error."""
        result = await initialized_adapter.execute("unknown_operation")
        assert result.success is False
        assert "Unknown operation" in result.error


# =============================================================================
# HEURISTIC EVALUATOR TESTS
# =============================================================================

class TestHeuristicDeepEvalEvaluator:
    """Tests for the heuristic fallback evaluator."""

    def test_evaluator_creation(self, heuristic_evaluator):
        """Test heuristic evaluator can be created."""
        assert heuristic_evaluator is not None
        assert heuristic_evaluator.config is not None

    def test_evaluate_basic(self, heuristic_evaluator, sample_data):
        """Test basic evaluation."""
        result = heuristic_evaluator.evaluate(
            input_text=sample_data["input"],
            actual_output=sample_data["actual_output"],
            retrieval_context=sample_data["retrieval_context"],
        )
        assert "scores" in result
        assert "passed" in result
        assert "method" in result
        assert result["method"] == "heuristic"
        assert "faithfulness" in result["scores"]
        assert "answer_relevancy" in result["scores"]
        assert "hallucination" in result["scores"]

    def test_evaluate_with_expected_output(self, heuristic_evaluator, sample_data):
        """Test evaluation with expected output for recall."""
        result = heuristic_evaluator.evaluate(
            input_text=sample_data["input"],
            actual_output=sample_data["actual_output"],
            retrieval_context=sample_data["retrieval_context"],
            expected_output=sample_data["expected_output"],
        )
        assert "contextual_recall" in result["scores"]
        assert result["scores"]["contextual_recall"] > 0.0

    def test_evaluate_empty_output(self, heuristic_evaluator, sample_data):
        """Test evaluation with empty output."""
        result = heuristic_evaluator.evaluate(
            input_text=sample_data["input"],
            actual_output="",
            retrieval_context=sample_data["retrieval_context"],
        )
        assert result["scores"]["faithfulness"] == 0.0

    def test_evaluate_empty_context(self, heuristic_evaluator, sample_data):
        """Test evaluation with empty context."""
        result = heuristic_evaluator.evaluate(
            input_text=sample_data["input"],
            actual_output=sample_data["actual_output"],
            retrieval_context=[],
        )
        assert result["scores"]["faithfulness"] == 0.0
        assert result["scores"]["hallucination"] == 1.0  # All ungrounded

    def test_hallucination_low_when_grounded(self, heuristic_evaluator):
        """Test low hallucination when output matches context."""
        result = heuristic_evaluator.evaluate(
            input_text="What is the capital of France?",
            actual_output="Paris is the capital of France.",
            retrieval_context=["Paris is the capital of France."],
        )
        # Should have low hallucination since answer is grounded
        assert result["scores"]["hallucination"] < 0.5

    def test_hallucination_high_when_ungrounded(self, heuristic_evaluator):
        """Test high hallucination when output differs from context."""
        result = heuristic_evaluator.evaluate(
            input_text="What is the capital of France?",
            actual_output="London is a beautiful city in England with red buses.",
            retrieval_context=["Paris is the capital of France."],
        )
        assert result["scores"]["hallucination"] > 0.5

    def test_detect_hallucination_basic(self, heuristic_evaluator):
        """Test hallucination detection method."""
        result = heuristic_evaluator.detect_hallucination(
            actual_output="Paris is the capital of France.",
            retrieval_context=["Paris is the capital of France."],
        )
        assert "hallucination_score" in result
        assert "is_hallucinated" in result
        assert result["method"] == "heuristic"

    def test_detect_hallucination_empty_output(self, heuristic_evaluator):
        """Test hallucination detection with empty output."""
        result = heuristic_evaluator.detect_hallucination(
            actual_output="",
            retrieval_context=["Some context"],
        )
        assert result["hallucination_score"] == 0.0
        assert result["is_hallucinated"] is False

    def test_detect_hallucination_ungrounded(self, heuristic_evaluator):
        """Test hallucination detection with ungrounded content."""
        result = heuristic_evaluator.detect_hallucination(
            actual_output="The moon is made of cheese and unicorns live there.",
            retrieval_context=["The Moon is Earth's natural satellite."],
        )
        assert result["is_hallucinated"] is True
        assert result["hallucination_score"] > 0.5

    def test_scores_in_valid_range(self, heuristic_evaluator, sample_data):
        """Test all scores are in valid 0-1 range."""
        result = heuristic_evaluator.evaluate(
            input_text=sample_data["input"],
            actual_output=sample_data["actual_output"],
            retrieval_context=sample_data["retrieval_context"],
            expected_output=sample_data["expected_output"],
        )
        for metric, score in result["scores"].items():
            assert 0.0 <= score <= 1.0, f"{metric} score {score} out of range"


# =============================================================================
# ADAPTER EVALUATION TESTS
# =============================================================================

class TestDeepEvalAdapterEvaluation:
    """Tests for adapter evaluation functionality."""

    @pytest.mark.asyncio
    async def test_evaluate_basic(self, initialized_adapter, sample_data):
        """Test basic evaluation through adapter."""
        result = await initialized_adapter.execute(
            "evaluate",
            input=sample_data["input"],
            actual_output=sample_data["actual_output"],
            retrieval_context=sample_data["retrieval_context"],
        )
        assert result.success is True
        assert "scores" in result.data
        assert "passed" in result.data
        assert "method" in result.data

    @pytest.mark.asyncio
    async def test_evaluate_with_expected_output(self, initialized_adapter, sample_data):
        """Test evaluation with expected output."""
        result = await initialized_adapter.execute(
            "evaluate",
            input=sample_data["input"],
            actual_output=sample_data["actual_output"],
            retrieval_context=sample_data["retrieval_context"],
            expected_output=sample_data["expected_output"],
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_detect_hallucination(self, initialized_adapter, sample_data):
        """Test hallucination detection."""
        result = await initialized_adapter.execute(
            "detect_hallucination",
            actual_output=sample_data["actual_output"],
            retrieval_context=sample_data["retrieval_context"],
        )
        assert result.success is True
        assert "hallucination_score" in result.data
        assert "is_hallucinated" in result.data

    @pytest.mark.asyncio
    async def test_detect_hallucination_direct_method(self, initialized_adapter, sample_data):
        """Test direct detect_hallucination method."""
        result = await initialized_adapter.detect_hallucination(
            answer=sample_data["actual_output"],
            contexts=sample_data["retrieval_context"],
        )
        assert isinstance(result, dict)
        assert "hallucination_score" in result or "error" in result

    @pytest.mark.asyncio
    async def test_evaluate_single_protocol(self, initialized_adapter, sample_data):
        """Test evaluate_single for EvaluatorProtocol compatibility."""
        result = await initialized_adapter.execute(
            "evaluate_single",
            question=sample_data["input"],
            contexts=sample_data["retrieval_context"],
            answer=sample_data["actual_output"],
            ground_truth=sample_data["expected_output"],
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_evaluate_single_direct_method(self, initialized_adapter, sample_data):
        """Test direct evaluate_single method."""
        result = await initialized_adapter.evaluate_single(
            question=sample_data["input"],
            contexts=sample_data["retrieval_context"],
            answer=sample_data["actual_output"],
            ground_truth=sample_data["expected_output"],
        )
        assert isinstance(result, dict)
        assert "scores" in result or "error" in result

    @pytest.mark.asyncio
    async def test_evaluate_batch(self, initialized_adapter, sample_data):
        """Test batch evaluation."""
        test_cases = [
            {
                "input": sample_data["input"],
                "actual_output": sample_data["actual_output"],
                "retrieval_context": sample_data["retrieval_context"],
            },
            {
                "input": "What is AI?",
                "actual_output": "AI is artificial intelligence.",
                "retrieval_context": ["AI stands for artificial intelligence."],
            },
        ]
        result = await initialized_adapter.execute(
            "evaluate_batch",
            test_cases=test_cases,
        )
        assert result.success is True
        assert "results" in result.data
        assert "total" in result.data
        assert result.data["total"] == 2


# =============================================================================
# CIRCUIT BREAKER TESTS
# =============================================================================

class TestDeepEvalAdapterCircuitBreaker:
    """Tests for circuit breaker behavior."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_fallback(self, sample_data):
        """Test heuristic fallback when circuit is open."""
        adapter = DeepEvalAdapter()
        await adapter.initialize({})

        with patch("adapters.deepeval_adapter.adapter_circuit_breaker") as mock_cb:
            mock_instance = MagicMock()
            mock_instance.is_open = True
            mock_cb.return_value = mock_instance

            result = await adapter.execute(
                "evaluate",
                input=sample_data["input"],
                actual_output=sample_data["actual_output"],
                retrieval_context=sample_data["retrieval_context"],
            )

            assert result.success is True
            assert result.data["method"] == "heuristic"

    @pytest.mark.asyncio
    async def test_circuit_breaker_hallucination_fallback(self, sample_data):
        """Test hallucination detection fallback when circuit is open."""
        adapter = DeepEvalAdapter()
        await adapter.initialize({})

        with patch("adapters.deepeval_adapter.adapter_circuit_breaker") as mock_cb:
            mock_instance = MagicMock()
            mock_instance.is_open = True
            mock_cb.return_value = mock_instance

            result = await adapter.execute(
                "detect_hallucination",
                actual_output=sample_data["actual_output"],
                retrieval_context=sample_data["retrieval_context"],
            )

            assert result.success is True
            assert result.data["method"] == "heuristic"


# =============================================================================
# METRIC CONFIG TESTS
# =============================================================================

class TestDeepEvalMetricConfig:
    """Tests for DeepEvalMetricConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DeepEvalMetricConfig()
        assert config.faithfulness_threshold == 0.7
        assert config.relevancy_threshold == 0.7
        assert config.hallucination_threshold == 0.3
        assert config.precision_threshold == 0.5
        assert config.recall_threshold == 0.5
        assert config.use_claude is False
        assert config.model == "gpt-4o-mini"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DeepEvalMetricConfig(
            faithfulness_threshold=0.9,
            hallucination_threshold=0.1,
            use_claude=True,
            model="claude-3-5-sonnet-20241022",
        )
        assert config.faithfulness_threshold == 0.9
        assert config.hallucination_threshold == 0.1
        assert config.use_claude is True
        assert config.model == "claude-3-5-sonnet-20241022"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestDeepEvalAdapterEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    async def test_empty_input(self, initialized_adapter):
        """Test with empty input."""
        result = await initialized_adapter.execute(
            "evaluate",
            input="",
            actual_output="Some answer",
            retrieval_context=["Some context"],
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_empty_output(self, initialized_adapter):
        """Test with empty output."""
        result = await initialized_adapter.execute(
            "evaluate",
            input="What is X?",
            actual_output="",
            retrieval_context=["Context about X"],
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_empty_context(self, initialized_adapter):
        """Test with empty context list."""
        result = await initialized_adapter.execute(
            "evaluate",
            input="What is X?",
            actual_output="X is something",
            retrieval_context=[],
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_none_context(self, initialized_adapter):
        """Test with None context."""
        result = await initialized_adapter.execute(
            "evaluate",
            input="What is X?",
            actual_output="X is something",
            retrieval_context=None,
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_batch_empty_list(self, initialized_adapter):
        """Test batch with empty list."""
        result = await initialized_adapter.execute(
            "evaluate_batch",
            test_cases=[],
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_batch_none(self, initialized_adapter):
        """Test batch with None."""
        result = await initialized_adapter.execute(
            "evaluate_batch",
            test_cases=None,
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_very_long_text(self, initialized_adapter):
        """Test with very long text."""
        long_text = "word " * 1000
        result = await initialized_adapter.execute(
            "evaluate",
            input=long_text,
            actual_output=long_text,
            retrieval_context=[long_text],
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_special_characters(self, initialized_adapter):
        """Test with special characters."""
        result = await initialized_adapter.execute(
            "evaluate",
            input="What is @#$%^&*()?",
            actual_output="It's a test with !@#$%",
            retrieval_context=["Context with special chars: <>&\"'"],
        )
        assert result.success is True


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestDeepEvalAdapterIntegration:
    """Integration tests."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, sample_data):
        """Test full evaluation workflow."""
        adapter = DeepEvalAdapter()

        # Initialize
        init_result = await adapter.initialize({
            "faithfulness_threshold": 0.6,
            "relevancy_threshold": 0.6,
            "hallucination_threshold": 0.4,
        })
        assert init_result.success is True

        # Evaluate
        eval_result = await adapter.execute(
            "evaluate",
            input=sample_data["input"],
            actual_output=sample_data["actual_output"],
            retrieval_context=sample_data["retrieval_context"],
        )
        assert eval_result.success is True

        # Detect hallucination
        hall_result = await adapter.execute(
            "detect_hallucination",
            actual_output=sample_data["actual_output"],
            retrieval_context=sample_data["retrieval_context"],
        )
        assert hall_result.success is True

        # Get stats
        stats_result = await adapter.execute("get_stats")
        assert stats_result.success is True
        assert stats_result.data["call_count"] >= 2

        # Health check
        health_result = await adapter.health_check()
        assert health_result.success is True

        # Shutdown
        shutdown_result = await adapter.shutdown()
        assert shutdown_result.success is True

    @pytest.mark.asyncio
    async def test_multiple_evaluations(self, initialized_adapter, sample_data):
        """Test multiple evaluations in sequence."""
        for i in range(3):
            result = await initialized_adapter.execute(
                "evaluate",
                input=f"{sample_data['input']} {i}",
                actual_output=sample_data["actual_output"],
                retrieval_context=sample_data["retrieval_context"],
            )
            assert result.success is True

        stats = await initialized_adapter.execute("get_stats")
        assert stats.data["call_count"] >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
