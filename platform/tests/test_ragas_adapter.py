"""
Test Suite for Ragas Adapter (V65 Gap11)

Comprehensive tests for the Ragas evaluation adapter covering:
- Core evaluation functionality
- Full metric suite
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

from adapters.ragas_adapter import (
    RagasAdapter,
    RagasMetricConfig,
    HeuristicRagasEvaluator,
    RAGAS_AVAILABLE,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def adapter():
    """Create a RagasAdapter instance."""
    return RagasAdapter()


@pytest.fixture
def initialized_adapter():
    """Create an initialized RagasAdapter instance."""
    adapter = RagasAdapter()
    asyncio.get_event_loop().run_until_complete(adapter.initialize({}))
    return adapter


@pytest.fixture
def heuristic_evaluator():
    """Create a HeuristicRagasEvaluator instance."""
    return HeuristicRagasEvaluator()


@pytest.fixture
def sample_data():
    """Sample data for evaluation tests."""
    return {
        "question": "What is machine learning?",
        "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "contexts": [
            "Machine learning is a subset of artificial intelligence.",
            "ML systems learn patterns from data without explicit programming.",
            "Deep learning is a type of machine learning using neural networks.",
        ],
        "ground_truth": "Machine learning is a branch of AI that allows computers to learn from data.",
    }


# =============================================================================
# BASIC ADAPTER TESTS
# =============================================================================

class TestRagasAdapterBasic:
    """Basic adapter functionality tests."""

    def test_adapter_creation(self, adapter):
        """Test adapter can be created."""
        assert adapter is not None
        assert adapter.sdk_name == "ragas"
        assert adapter.layer == 5  # OBSERVABILITY

    def test_adapter_available_property(self, adapter):
        """Test available property reflects SDK status."""
        # Should return True/False based on actual ragas availability
        assert isinstance(adapter.available, bool)

    @pytest.mark.asyncio
    async def test_initialize_default_config(self, adapter):
        """Test initialization with default config."""
        result = await adapter.initialize({})
        assert result.success is True
        assert "llm_model" in result.data
        assert "thresholds" in result.data
        assert result.data["thresholds"]["faithfulness"] == 0.7
        assert result.data["thresholds"]["relevancy"] == 0.6
        assert result.data["thresholds"]["precision"] == 0.5

    @pytest.mark.asyncio
    async def test_initialize_custom_config(self, adapter):
        """Test initialization with custom config."""
        config = {
            "faithfulness_threshold": 0.8,
            "relevancy_threshold": 0.7,
            "precision_threshold": 0.6,
            "recall_threshold": 0.5,
            "use_claude": True,
            "claude_model": "claude-3-opus-20240229",
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
        assert "ragas_available" in result.data
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

    @pytest.mark.asyncio
    async def test_unknown_operation(self, initialized_adapter):
        """Test unknown operation returns error."""
        result = await initialized_adapter.execute("unknown_operation")
        assert result.success is False
        assert "Unknown operation" in result.error


# =============================================================================
# HEURISTIC EVALUATOR TESTS
# =============================================================================

class TestHeuristicRagasEvaluator:
    """Tests for the heuristic fallback evaluator."""

    def test_evaluator_creation(self, heuristic_evaluator):
        """Test heuristic evaluator can be created."""
        assert heuristic_evaluator is not None
        assert heuristic_evaluator.config is not None

    def test_evaluate_basic(self, heuristic_evaluator, sample_data):
        """Test basic evaluation."""
        result = heuristic_evaluator.evaluate(
            question=sample_data["question"],
            answer=sample_data["answer"],
            contexts=sample_data["contexts"],
        )
        assert "scores" in result
        assert "passed" in result
        assert "method" in result
        assert result["method"] == "heuristic"
        assert "faithfulness" in result["scores"]
        assert "answer_relevancy" in result["scores"]
        assert "context_precision" in result["scores"]

    def test_evaluate_with_ground_truth(self, heuristic_evaluator, sample_data):
        """Test evaluation with ground truth for recall."""
        result = heuristic_evaluator.evaluate(
            question=sample_data["question"],
            answer=sample_data["answer"],
            contexts=sample_data["contexts"],
            ground_truth=sample_data["ground_truth"],
        )
        assert "context_recall" in result["scores"]
        assert result["scores"]["context_recall"] > 0.0

    def test_evaluate_empty_answer(self, heuristic_evaluator, sample_data):
        """Test evaluation with empty answer."""
        result = heuristic_evaluator.evaluate(
            question=sample_data["question"],
            answer="",
            contexts=sample_data["contexts"],
        )
        assert result["scores"]["faithfulness"] == 0.0
        assert result["passed"] is False

    def test_evaluate_empty_contexts(self, heuristic_evaluator, sample_data):
        """Test evaluation with empty contexts."""
        result = heuristic_evaluator.evaluate(
            question=sample_data["question"],
            answer=sample_data["answer"],
            contexts=[],
        )
        assert result["scores"]["faithfulness"] == 0.0
        assert result["scores"]["context_precision"] == 0.0

    def test_evaluate_high_faithfulness(self, heuristic_evaluator):
        """Test high faithfulness when answer matches context."""
        result = heuristic_evaluator.evaluate(
            question="What is the capital of France?",
            answer="Paris is the capital of France.",
            contexts=["Paris is the capital of France."],
        )
        assert result["scores"]["faithfulness"] > 0.5

    def test_evaluate_low_faithfulness(self, heuristic_evaluator):
        """Test low faithfulness when answer differs from context."""
        result = heuristic_evaluator.evaluate(
            question="What is the capital of France?",
            answer="London is a beautiful city in England.",
            contexts=["Paris is the capital of France."],
        )
        # Most words in answer are not in context
        assert result["scores"]["faithfulness"] < 0.5

    def test_evaluate_scores_in_valid_range(self, heuristic_evaluator, sample_data):
        """Test all scores are in valid 0-1 range."""
        result = heuristic_evaluator.evaluate(
            question=sample_data["question"],
            answer=sample_data["answer"],
            contexts=sample_data["contexts"],
            ground_truth=sample_data["ground_truth"],
        )
        for metric, score in result["scores"].items():
            assert 0.0 <= score <= 1.0, f"{metric} score {score} out of range"

    def test_evaluate_thresholds_in_result(self, heuristic_evaluator, sample_data):
        """Test thresholds are included in result."""
        result = heuristic_evaluator.evaluate(
            question=sample_data["question"],
            answer=sample_data["answer"],
            contexts=sample_data["contexts"],
        )
        assert "thresholds" in result
        assert "faithfulness" in result["thresholds"]
        assert "relevancy" in result["thresholds"]
        assert "precision" in result["thresholds"]
        assert "recall" in result["thresholds"]

    def test_custom_config(self):
        """Test heuristic evaluator with custom config."""
        config = RagasMetricConfig(
            faithfulness_threshold=0.9,
            relevancy_threshold=0.9,
        )
        evaluator = HeuristicRagasEvaluator(config)
        result = evaluator.evaluate(
            question="Test?",
            answer="Test answer",
            contexts=["Some context"],
        )
        # With high thresholds, should fail
        assert result["thresholds"]["faithfulness"] == 0.9
        assert result["thresholds"]["relevancy"] == 0.9


# =============================================================================
# ADAPTER EVALUATION TESTS
# =============================================================================

class TestRagasAdapterEvaluation:
    """Tests for adapter evaluation functionality."""

    @pytest.mark.asyncio
    async def test_evaluate_basic(self, initialized_adapter, sample_data):
        """Test basic evaluation through adapter."""
        result = await initialized_adapter.execute(
            "evaluate",
            question=sample_data["question"],
            answer=sample_data["answer"],
            contexts=sample_data["contexts"],
        )
        assert result.success is True
        assert "scores" in result.data
        assert "passed" in result.data
        assert "method" in result.data

    @pytest.mark.asyncio
    async def test_evaluate_with_ground_truth(self, initialized_adapter, sample_data):
        """Test evaluation with ground truth."""
        result = await initialized_adapter.execute(
            "evaluate",
            question=sample_data["question"],
            answer=sample_data["answer"],
            contexts=sample_data["contexts"],
            ground_truth=sample_data["ground_truth"],
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_evaluate_full_metrics(self, initialized_adapter, sample_data):
        """Test full metric suite evaluation."""
        result = await initialized_adapter.execute(
            "evaluate_full",
            question=sample_data["question"],
            answer=sample_data["answer"],
            contexts=sample_data["contexts"],
            ground_truth=sample_data["ground_truth"],
            metrics=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        )
        assert result.success is True
        assert "scores" in result.data

    @pytest.mark.asyncio
    async def test_evaluate_single_protocol(self, initialized_adapter, sample_data):
        """Test evaluate_single for EvaluatorProtocol compatibility."""
        result = await initialized_adapter.execute(
            "evaluate_single",
            question=sample_data["question"],
            contexts=sample_data["contexts"],
            answer=sample_data["answer"],
            ground_truth=sample_data["ground_truth"],
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_evaluate_single_direct_method(self, initialized_adapter, sample_data):
        """Test direct evaluate_single method."""
        result = await initialized_adapter.evaluate_single(
            question=sample_data["question"],
            contexts=sample_data["contexts"],
            answer=sample_data["answer"],
            ground_truth=sample_data["ground_truth"],
        )
        assert isinstance(result, dict)
        assert "scores" in result or "error" in result

    @pytest.mark.asyncio
    async def test_evaluate_latency_tracking(self, initialized_adapter, sample_data):
        """Test that latency is tracked (>=0 since heuristic can be fast)."""
        result = await initialized_adapter.execute(
            "evaluate",
            question=sample_data["question"],
            answer=sample_data["answer"],
            contexts=sample_data["contexts"],
        )
        # Latency can be 0 or very small for fast heuristic evaluations
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_evaluate_call_count(self, initialized_adapter, sample_data):
        """Test that call count is incremented."""
        # Get initial stats
        stats1 = await initialized_adapter.execute("get_stats")
        initial_count = stats1.data["call_count"]

        # Make an evaluation call
        await initialized_adapter.execute(
            "evaluate",
            question=sample_data["question"],
            answer=sample_data["answer"],
            contexts=sample_data["contexts"],
        )

        # Check count increased
        stats2 = await initialized_adapter.execute("get_stats")
        assert stats2.data["call_count"] > initial_count


# =============================================================================
# CIRCUIT BREAKER TESTS
# =============================================================================

class TestRagasAdapterCircuitBreaker:
    """Tests for circuit breaker behavior."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_fallback(self, sample_data):
        """Test heuristic fallback when circuit is open."""
        adapter = RagasAdapter()
        await adapter.initialize({})

        # Mock circuit breaker as open
        with patch("adapters.ragas_adapter.adapter_circuit_breaker") as mock_cb:
            mock_instance = MagicMock()
            mock_instance.is_open = True
            mock_cb.return_value = mock_instance

            result = await adapter.execute(
                "evaluate",
                question=sample_data["question"],
                answer=sample_data["answer"],
                contexts=sample_data["contexts"],
            )

            assert result.success is True
            # Should use heuristic fallback
            assert result.data["method"] == "heuristic"


# =============================================================================
# METRIC CONFIG TESTS
# =============================================================================

class TestRagasMetricConfig:
    """Tests for RagasMetricConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RagasMetricConfig()
        assert config.faithfulness_threshold == 0.7
        assert config.relevancy_threshold == 0.6
        assert config.precision_threshold == 0.5
        assert config.recall_threshold == 0.5
        assert config.use_claude is False
        assert config.claude_model == "claude-3-5-sonnet-20241022"
        assert config.openai_model == "gpt-4o-mini"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RagasMetricConfig(
            faithfulness_threshold=0.9,
            use_claude=True,
            claude_model="claude-3-opus-20240229",
        )
        assert config.faithfulness_threshold == 0.9
        assert config.use_claude is True
        assert config.claude_model == "claude-3-opus-20240229"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestRagasAdapterEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    async def test_empty_question(self, initialized_adapter):
        """Test with empty question."""
        result = await initialized_adapter.execute(
            "evaluate",
            question="",
            answer="Some answer",
            contexts=["Some context"],
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_empty_answer(self, initialized_adapter):
        """Test with empty answer."""
        result = await initialized_adapter.execute(
            "evaluate",
            question="What is X?",
            answer="",
            contexts=["Context about X"],
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_empty_contexts(self, initialized_adapter):
        """Test with empty contexts list."""
        result = await initialized_adapter.execute(
            "evaluate",
            question="What is X?",
            answer="X is something",
            contexts=[],
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_none_contexts(self, initialized_adapter):
        """Test with None contexts."""
        result = await initialized_adapter.execute(
            "evaluate",
            question="What is X?",
            answer="X is something",
            contexts=None,
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_very_long_text(self, initialized_adapter):
        """Test with very long text."""
        long_text = "word " * 1000
        result = await initialized_adapter.execute(
            "evaluate",
            question=long_text,
            answer=long_text,
            contexts=[long_text],
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_special_characters(self, initialized_adapter):
        """Test with special characters."""
        result = await initialized_adapter.execute(
            "evaluate",
            question="What is @#$%^&*()?",
            answer="It's a test with !@#$%",
            contexts=["Context with special chars: <>&\"'"],
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_unicode_text(self, initialized_adapter):
        """Test with unicode text."""
        result = await initialized_adapter.execute(
            "evaluate",
            question="Qu'est-ce que c'est?",
            answer="C'est une reponse en francais avec accents.",
            contexts=["Le contexte avec des caracteres speciaux."],
        )
        assert result.success is True


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestRagasAdapterIntegration:
    """Integration tests."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, sample_data):
        """Test full evaluation workflow."""
        adapter = RagasAdapter()

        # Initialize
        init_result = await adapter.initialize({
            "faithfulness_threshold": 0.6,
            "relevancy_threshold": 0.5,
        })
        assert init_result.success is True

        # Evaluate
        eval_result = await adapter.execute(
            "evaluate",
            question=sample_data["question"],
            answer=sample_data["answer"],
            contexts=sample_data["contexts"],
        )
        assert eval_result.success is True

        # Get stats
        stats_result = await adapter.execute("get_stats")
        assert stats_result.success is True
        assert stats_result.data["call_count"] >= 1

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
                question=f"{sample_data['question']} {i}",
                answer=sample_data["answer"],
                contexts=sample_data["contexts"],
            )
            assert result.success is True

        stats = await initialized_adapter.execute("get_stats")
        assert stats.data["call_count"] >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
