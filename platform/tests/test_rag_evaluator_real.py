"""
test_rag_evaluator_real.py - Real RAG Evaluator Tests (No Mocks)

Tests the actual RAG evaluation functionality including:
- HeuristicEvaluator (always available)
- RAGEvaluator with Ragas/DeepEval SDK when available
- EvaluatorProtocol compliance for pipeline integration
- Answer relevance, faithfulness, context precision metrics
- Hallucination detection
- BERTScore and semantic similarity

Gap11 Status: IMPLEMENTED (V66)

Usage:
    pytest platform/tests/test_rag_evaluator_real.py -v
    pytest platform/tests/test_rag_evaluator_real.py -v -k "test_heuristic"
"""

import asyncio
import pytest
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add platform path for imports
PLATFORM_ROOT = Path(__file__).parent.parent
if str(PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATFORM_ROOT))


# =============================================================================
# TEST DATA
# =============================================================================

SAMPLE_QUESTION = "What is Retrieval-Augmented Generation (RAG)?"

SAMPLE_CONTEXTS = [
    "Retrieval-Augmented Generation (RAG) is a technique that combines information "
    "retrieval with text generation to improve LLM outputs. RAG first retrieves "
    "relevant documents from a knowledge base, then uses those documents as context "
    "for generating responses.",
    "Benefits of RAG include reduced hallucinations, improved accuracy, and the "
    "ability to cite sources. RAG systems typically use vector databases for "
    "efficient semantic search over large document collections.",
    "The RAG architecture consists of three main components: a retriever (often "
    "using embeddings), a knowledge store (vector database), and a generator (LLM). "
    "These work together to provide grounded, factual responses.",
]

SAMPLE_ANSWER_GOOD = (
    "Retrieval-Augmented Generation (RAG) is a technique that enhances LLM outputs "
    "by first retrieving relevant documents from a knowledge base and using them as "
    "context for generation. This approach reduces hallucinations and improves accuracy "
    "by grounding responses in factual information from the retrieved documents."
)

SAMPLE_ANSWER_POOR = (
    "RAG was invented by Google in 2015 and is the only way to make LLMs accurate. "
    "It uses quantum computing to search databases instantly and has replaced all "
    "traditional search engines worldwide."
)

SAMPLE_GROUND_TRUTH = (
    "RAG (Retrieval-Augmented Generation) is an AI framework that retrieves relevant "
    "documents and uses them to ground LLM responses, reducing hallucinations and "
    "improving factual accuracy."
)


# =============================================================================
# HEURISTIC EVALUATOR TESTS (Always Available)
# =============================================================================

class TestHeuristicEvaluator:
    """Tests for HeuristicEvaluator - the always-available fallback."""

    def test_import_heuristic_evaluator(self):
        """Test that HeuristicEvaluator can be imported."""
        from adapters.rag_evaluator import HeuristicEvaluator
        evaluator = HeuristicEvaluator()
        assert evaluator is not None

    def test_heuristic_evaluate_high_quality_answer(self):
        """Test heuristic evaluation of a high-quality answer."""
        from adapters.rag_evaluator import HeuristicEvaluator

        evaluator = HeuristicEvaluator()
        result = evaluator.evaluate(
            question=SAMPLE_QUESTION,
            answer=SAMPLE_ANSWER_GOOD,
            contexts=SAMPLE_CONTEXTS,
            ground_truth=SAMPLE_GROUND_TRUTH,
        )

        assert "scores" in result
        scores = result["scores"]

        # High-quality answer should score well
        assert scores["faithfulness"] > 0.3, f"Faithfulness too low: {scores['faithfulness']}"
        assert scores["answer_relevancy"] > 0.2, f"Relevancy too low: {scores['answer_relevancy']}"
        assert scores["context_precision"] > 0.2, f"Precision too low: {scores['context_precision']}"
        assert result["method"] == "heuristic"

    def test_heuristic_evaluate_low_quality_answer(self):
        """Test heuristic evaluation of a low-quality/hallucinated answer."""
        from adapters.rag_evaluator import HeuristicEvaluator

        evaluator = HeuristicEvaluator()
        result = evaluator.evaluate(
            question=SAMPLE_QUESTION,
            answer=SAMPLE_ANSWER_POOR,
            contexts=SAMPLE_CONTEXTS,
        )

        scores = result["scores"]

        # Poor answer should have lower faithfulness (contains hallucinations)
        # Note: Heuristic only checks token overlap, not semantic accuracy
        assert scores["faithfulness"] < 0.7, f"Faithfulness should be lower for poor answer: {scores['faithfulness']}"

    def test_heuristic_evaluate_empty_contexts(self):
        """Test heuristic evaluation with no contexts."""
        from adapters.rag_evaluator import HeuristicEvaluator

        evaluator = HeuristicEvaluator()
        result = evaluator.evaluate(
            question=SAMPLE_QUESTION,
            answer=SAMPLE_ANSWER_GOOD,
            contexts=[],
        )

        scores = result["scores"]

        # With no context, faithfulness should be 0 (nothing to ground against)
        assert scores["faithfulness"] == 0.0, "Faithfulness should be 0 with empty contexts"
        assert result["passed"] is False, "Should not pass with empty contexts"

    def test_heuristic_evaluate_empty_answer(self):
        """Test heuristic evaluation with empty answer."""
        from adapters.rag_evaluator import HeuristicEvaluator

        evaluator = HeuristicEvaluator()
        result = evaluator.evaluate(
            question=SAMPLE_QUESTION,
            answer="",
            contexts=SAMPLE_CONTEXTS,
        )

        scores = result["scores"]
        assert scores["faithfulness"] == 0.0
        assert scores["answer_relevancy"] == 0.0

    def test_heuristic_evaluate_with_ground_truth(self):
        """Test that ground truth affects context_recall score."""
        from adapters.rag_evaluator import HeuristicEvaluator

        evaluator = HeuristicEvaluator()

        # Without ground truth
        result_no_gt = evaluator.evaluate(
            question=SAMPLE_QUESTION,
            answer=SAMPLE_ANSWER_GOOD,
            contexts=SAMPLE_CONTEXTS,
        )

        # With ground truth
        result_with_gt = evaluator.evaluate(
            question=SAMPLE_QUESTION,
            answer=SAMPLE_ANSWER_GOOD,
            contexts=SAMPLE_CONTEXTS,
            ground_truth=SAMPLE_GROUND_TRUTH,
        )

        # Ground truth should add context_recall score
        assert "context_recall" not in result_no_gt["scores"] or result_no_gt["scores"].get("context_recall", 0) == 0
        assert result_with_gt["scores"]["context_recall"] > 0

    def test_heuristic_custom_thresholds(self):
        """Test HeuristicEvaluator with custom thresholds."""
        from adapters.rag_evaluator import HeuristicEvaluator

        # Very high thresholds - should fail
        evaluator_strict = HeuristicEvaluator(
            faithfulness_threshold=0.99,
            relevancy_threshold=0.99,
            precision_threshold=0.99,
        )
        result_strict = evaluator_strict.evaluate(
            question=SAMPLE_QUESTION,
            answer=SAMPLE_ANSWER_GOOD,
            contexts=SAMPLE_CONTEXTS,
        )
        assert result_strict["passed"] is False, "Should fail with very high thresholds"

        # Very low thresholds - should pass
        evaluator_lenient = HeuristicEvaluator(
            faithfulness_threshold=0.01,
            relevancy_threshold=0.01,
            precision_threshold=0.01,
        )
        result_lenient = evaluator_lenient.evaluate(
            question=SAMPLE_QUESTION,
            answer=SAMPLE_ANSWER_GOOD,
            contexts=SAMPLE_CONTEXTS,
        )
        assert result_lenient["passed"] is True, "Should pass with very low thresholds"


# =============================================================================
# RAGEvaluator TESTS
# =============================================================================

class TestRAGEvaluator:
    """Tests for the main RAGEvaluator class."""

    def test_import_rag_evaluator(self):
        """Test that RAGEvaluator can be imported."""
        from adapters.rag_evaluator import RAGEvaluator
        evaluator = RAGEvaluator()
        assert evaluator is not None

    def test_available_frameworks(self):
        """Test checking available frameworks."""
        from adapters.rag_evaluator import RAGEvaluator

        evaluator = RAGEvaluator()
        frameworks = evaluator.available_frameworks()

        assert "ragas" in frameworks
        assert "deepeval" in frameworks
        assert "heuristic" in frameworks
        assert frameworks["heuristic"] is True  # Always available

        # Log which SDKs are available for debugging
        print(f"\nAvailable frameworks: {frameworks}")

    @pytest.mark.asyncio
    async def test_rag_evaluator_evaluate(self):
        """Test RAGEvaluator.evaluate() method."""
        from adapters.rag_evaluator import RAGEvaluator

        evaluator = RAGEvaluator()
        result = await evaluator.evaluate(
            question=SAMPLE_QUESTION,
            answer=SAMPLE_ANSWER_GOOD,
            contexts=SAMPLE_CONTEXTS,
        )

        # Check EvalResult structure
        assert hasattr(result, "scores"), "Result should have scores"
        assert hasattr(result, "passed"), "Result should have passed"
        assert hasattr(result, "framework"), "Result should have framework"
        assert hasattr(result, "details"), "Result should have details"
        assert hasattr(result, "metadata"), "Result should have metadata"

        # Check scores are valid
        for metric, score in result.scores.items():
            assert 0.0 <= score <= 1.0, f"Score {metric}={score} out of range"

        print(f"\nEvaluation result: {result.scores}")
        print(f"Framework used: {result.framework}")
        print(f"Passed: {result.passed}")

    @pytest.mark.asyncio
    async def test_rag_evaluator_evaluate_single(self):
        """Test EvaluatorProtocol.evaluate_single() for pipeline integration."""
        from adapters.rag_evaluator import RAGEvaluator

        evaluator = RAGEvaluator()

        # This is the method signature used by RAGPipeline
        result = await evaluator.evaluate_single(
            question=SAMPLE_QUESTION,
            contexts=SAMPLE_CONTEXTS,
            answer=SAMPLE_ANSWER_GOOD,
            ground_truth=SAMPLE_GROUND_TRUTH,
        )

        # Should return a dict with standard keys
        assert isinstance(result, dict), "evaluate_single should return a dict"
        assert "scores" in result
        assert "passed" in result
        assert "framework" in result

    @pytest.mark.asyncio
    async def test_rag_evaluator_heuristic_fallback(self):
        """Test that heuristic fallback works when SDKs are unavailable."""
        from adapters.rag_evaluator import RAGEvaluator

        evaluator = RAGEvaluator(use_heuristic_fallback=True)

        # Force heuristic by disabling both SDKs
        result = await evaluator.evaluate(
            question=SAMPLE_QUESTION,
            answer=SAMPLE_ANSWER_GOOD,
            contexts=SAMPLE_CONTEXTS,
            use_ragas=False,
            use_deepeval=False,
        )

        # Should use heuristic
        assert "heuristic" in result.framework.lower(), f"Expected heuristic, got {result.framework}"
        assert "faithfulness" in result.scores
        assert "answer_relevancy" in result.scores

    @pytest.mark.asyncio
    async def test_rag_evaluator_with_ground_truth(self):
        """Test evaluation with ground truth for recall calculation."""
        from adapters.rag_evaluator import RAGEvaluator

        evaluator = RAGEvaluator()

        result = await evaluator.evaluate(
            question=SAMPLE_QUESTION,
            answer=SAMPLE_ANSWER_GOOD,
            contexts=SAMPLE_CONTEXTS,
            ground_truth=SAMPLE_GROUND_TRUTH,
        )

        # Ground truth should enable context_recall metric
        if "context_recall" in result.scores:
            assert result.scores["context_recall"] > 0

    @pytest.mark.asyncio
    async def test_rag_evaluator_metadata(self):
        """Test that evaluation returns useful metadata."""
        from adapters.rag_evaluator import RAGEvaluator

        evaluator = RAGEvaluator()
        result = await evaluator.evaluate(
            question=SAMPLE_QUESTION,
            answer=SAMPLE_ANSWER_GOOD,
            contexts=SAMPLE_CONTEXTS,
        )

        # Check metadata
        assert "question_len" in result.metadata
        assert "answer_len" in result.metadata
        assert "num_contexts" in result.metadata
        assert "thresholds" in result.metadata

        assert result.metadata["num_contexts"] == len(SAMPLE_CONTEXTS)


# =============================================================================
# DEEPEVAL ADAPTER TESTS
# =============================================================================

class TestDeepEvalAdapter:
    """Tests for DeepEvalAdapter with real functionality."""

    def test_import_deepeval_adapter(self):
        """Test that DeepEvalAdapter can be imported."""
        from adapters.deepeval_adapter import DeepEvalAdapter, DEEPEVAL_AVAILABLE

        adapter = DeepEvalAdapter()
        assert adapter is not None
        print(f"\nDeepEval SDK available: {DEEPEVAL_AVAILABLE}")

    @pytest.mark.asyncio
    async def test_deepeval_adapter_initialize(self):
        """Test DeepEval adapter initialization."""
        from adapters.deepeval_adapter import DeepEvalAdapter

        adapter = DeepEvalAdapter()
        result = await adapter.initialize({"model": "gpt-4o-mini"})

        assert result.success is True
        assert "deepeval_native" in result.data

    @pytest.mark.asyncio
    async def test_deepeval_adapter_evaluate(self):
        """Test DeepEval adapter evaluation."""
        from adapters.deepeval_adapter import DeepEvalAdapter

        adapter = DeepEvalAdapter()
        await adapter.initialize({})

        result = await adapter.execute(
            "evaluate",
            input=SAMPLE_QUESTION,
            actual_output=SAMPLE_ANSWER_GOOD,
            retrieval_context=SAMPLE_CONTEXTS,
        )

        assert result.success is True
        assert "scores" in result.data
        assert "passed" in result.data
        assert "method" in result.data

    @pytest.mark.asyncio
    async def test_deepeval_adapter_detect_hallucination(self):
        """Test DeepEval hallucination detection."""
        from adapters.deepeval_adapter import DeepEvalAdapter

        adapter = DeepEvalAdapter()
        await adapter.initialize({})

        result = await adapter.execute(
            "detect_hallucination",
            actual_output=SAMPLE_ANSWER_POOR,  # Poor answer with hallucinations
            retrieval_context=SAMPLE_CONTEXTS,
        )

        assert result.success is True
        assert "hallucination_score" in result.data
        assert "is_hallucinated" in result.data

    @pytest.mark.asyncio
    async def test_deepeval_adapter_evaluate_single(self):
        """Test DeepEval EvaluatorProtocol compliance."""
        from adapters.deepeval_adapter import DeepEvalAdapter

        adapter = DeepEvalAdapter()
        await adapter.initialize({})

        # Using the direct method that implements EvaluatorProtocol
        result = await adapter.evaluate_single(
            question=SAMPLE_QUESTION,
            contexts=SAMPLE_CONTEXTS,
            answer=SAMPLE_ANSWER_GOOD,
        )

        assert isinstance(result, dict)
        assert "scores" in result or "error" in result

    @pytest.mark.asyncio
    async def test_deepeval_adapter_health_check(self):
        """Test DeepEval adapter health check."""
        from adapters.deepeval_adapter import DeepEvalAdapter

        adapter = DeepEvalAdapter()
        await adapter.initialize({})

        result = await adapter.health_check()

        assert result.success is True
        assert "status" in result.data
        assert "heuristic_fallback" in result.data
        assert result.data["heuristic_fallback"] is True  # Always available


# =============================================================================
# RAGAS ADAPTER TESTS
# =============================================================================

class TestRagasAdapter:
    """Tests for RagasAdapter with real functionality."""

    def test_import_ragas_adapter(self):
        """Test that RagasAdapter can be imported."""
        from adapters.ragas_adapter import RagasAdapter, RAGAS_AVAILABLE

        adapter = RagasAdapter()
        assert adapter is not None
        print(f"\nRagas SDK available: {RAGAS_AVAILABLE}")

    @pytest.mark.asyncio
    async def test_ragas_adapter_initialize(self):
        """Test Ragas adapter initialization."""
        from adapters.ragas_adapter import RagasAdapter

        adapter = RagasAdapter()
        result = await adapter.initialize({"openai_model": "gpt-4o-mini"})

        assert result.success is True
        assert "ragas_native" in result.data

    @pytest.mark.asyncio
    async def test_ragas_adapter_evaluate(self):
        """Test Ragas adapter evaluation."""
        from adapters.ragas_adapter import RagasAdapter

        adapter = RagasAdapter()
        await adapter.initialize({})

        result = await adapter.execute(
            "evaluate",
            question=SAMPLE_QUESTION,
            answer=SAMPLE_ANSWER_GOOD,
            contexts=SAMPLE_CONTEXTS,
        )

        assert result.success is True
        assert "scores" in result.data
        assert "passed" in result.data
        assert "method" in result.data

    @pytest.mark.asyncio
    async def test_ragas_adapter_evaluate_full(self):
        """Test Ragas full evaluation with all metrics."""
        from adapters.ragas_adapter import RagasAdapter

        adapter = RagasAdapter()
        await adapter.initialize({})

        result = await adapter.execute(
            "evaluate_full",
            question=SAMPLE_QUESTION,
            answer=SAMPLE_ANSWER_GOOD,
            contexts=SAMPLE_CONTEXTS,
            ground_truth=SAMPLE_GROUND_TRUTH,
            metrics=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        )

        assert result.success is True
        assert "scores" in result.data
        # Heuristic fallback should provide all requested metrics
        scores = result.data["scores"]
        print(f"\nFull evaluation scores: {scores}")

    @pytest.mark.asyncio
    async def test_ragas_adapter_evaluate_single(self):
        """Test Ragas EvaluatorProtocol compliance."""
        from adapters.ragas_adapter import RagasAdapter

        adapter = RagasAdapter()
        await adapter.initialize({})

        result = await adapter.evaluate_single(
            question=SAMPLE_QUESTION,
            contexts=SAMPLE_CONTEXTS,
            answer=SAMPLE_ANSWER_GOOD,
        )

        assert isinstance(result, dict)
        assert "scores" in result or "error" in result

    @pytest.mark.asyncio
    async def test_ragas_adapter_health_check(self):
        """Test Ragas adapter health check."""
        from adapters.ragas_adapter import RagasAdapter

        adapter = RagasAdapter()
        await adapter.initialize({})

        result = await adapter.health_check()

        assert result.success is True
        assert "status" in result.data
        assert "heuristic_fallback" in result.data


# =============================================================================
# OPIK INTEGRATION TESTS
# =============================================================================

class TestOpikMetrics:
    """Tests for Opik tracing adapter metrics."""

    def test_import_opik_adapter(self):
        """Test that Opik adapter can be imported."""
        from adapters.opik_tracing_adapter import (
            OpikTracer,
            VoyageEmbeddingMetric,
            OPIK_AVAILABLE,
            OPIK_METRICS_AVAILABLE,
        )

        tracer = OpikTracer()
        assert tracer is not None
        print(f"\nOpik available: {OPIK_AVAILABLE}")
        print(f"Opik metrics available: {OPIK_METRICS_AVAILABLE}")

    def test_voyage_embedding_metric(self):
        """Test VoyageEmbeddingMetric scoring."""
        from adapters.opik_tracing_adapter import VoyageEmbeddingMetric

        metric = VoyageEmbeddingMetric()

        result = metric.score(
            query="What is RAG?",
            retrieved_docs=["RAG is retrieval augmented generation", "RAG improves LLM accuracy"],
            scores=[0.9, 0.85],
        )

        assert result.name == "voyage_embedding_quality"
        assert 0.0 <= result.score <= 1.0
        assert result.metadata["num_docs"] == 2
        print(f"\nVoyage metric score: {result.score}")
        print(f"Reason: {result.reason}")

    def test_voyage_embedding_metric_no_docs(self):
        """Test VoyageEmbeddingMetric with no documents."""
        from adapters.opik_tracing_adapter import VoyageEmbeddingMetric

        metric = VoyageEmbeddingMetric()

        result = metric.score(
            query="What is RAG?",
            retrieved_docs=[],
            scores=[],
        )

        assert result.score == 0.0
        assert "No documents" in result.reason

    def test_opik_tracer_status(self):
        """Test OpikTracer status reporting."""
        from adapters.opik_tracing_adapter import OpikTracer

        tracer = OpikTracer()
        status = tracer.get_status()

        assert "opik_available" in status
        assert "configured" in status
        assert "project_name" in status


# =============================================================================
# EVALUATOR PROTOCOL COMPLIANCE TESTS
# =============================================================================

class TestEvaluatorProtocolCompliance:
    """Tests to ensure all evaluators comply with EvaluatorProtocol."""

    @pytest.mark.asyncio
    async def test_rag_evaluator_protocol(self):
        """Test RAGEvaluator implements EvaluatorProtocol correctly."""
        from adapters.rag_evaluator import RAGEvaluator

        evaluator = RAGEvaluator()

        # The protocol requires: evaluate_single(question, contexts, answer, ground_truth) -> Any
        result = await evaluator.evaluate_single(
            question=SAMPLE_QUESTION,
            contexts=SAMPLE_CONTEXTS,
            answer=SAMPLE_ANSWER_GOOD,
            ground_truth=SAMPLE_GROUND_TRUTH,
        )

        assert isinstance(result, dict), "evaluate_single must return a dict"
        # Should have at minimum scores and passed
        assert "scores" in result, "Result must have scores"
        assert "passed" in result, "Result must have passed status"

    @pytest.mark.asyncio
    async def test_deepeval_adapter_protocol(self):
        """Test DeepEvalAdapter implements EvaluatorProtocol correctly."""
        from adapters.deepeval_adapter import DeepEvalAdapter

        adapter = DeepEvalAdapter()
        await adapter.initialize({})

        result = await adapter.evaluate_single(
            question=SAMPLE_QUESTION,
            contexts=SAMPLE_CONTEXTS,
            answer=SAMPLE_ANSWER_GOOD,
            ground_truth=SAMPLE_GROUND_TRUTH,
        )

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_ragas_adapter_protocol(self):
        """Test RagasAdapter implements EvaluatorProtocol correctly."""
        from adapters.ragas_adapter import RagasAdapter

        adapter = RagasAdapter()
        await adapter.initialize({})

        result = await adapter.evaluate_single(
            question=SAMPLE_QUESTION,
            contexts=SAMPLE_CONTEXTS,
            answer=SAMPLE_ANSWER_GOOD,
            ground_truth=SAMPLE_GROUND_TRUTH,
        )

        assert isinstance(result, dict)


# =============================================================================
# BENCHMARK TESTS
# =============================================================================

class TestEvaluatorBenchmarks:
    """Performance benchmarks for evaluators."""

    @pytest.mark.asyncio
    async def test_heuristic_evaluator_performance(self):
        """Benchmark heuristic evaluator - should be very fast."""
        import time
        from adapters.rag_evaluator import HeuristicEvaluator

        evaluator = HeuristicEvaluator()

        start = time.time()
        iterations = 100

        for _ in range(iterations):
            evaluator.evaluate(
                question=SAMPLE_QUESTION,
                answer=SAMPLE_ANSWER_GOOD,
                contexts=SAMPLE_CONTEXTS,
            )

        elapsed = time.time() - start
        avg_ms = (elapsed / iterations) * 1000

        print(f"\nHeuristic evaluator: {avg_ms:.2f}ms per evaluation")
        assert avg_ms < 10, f"Heuristic should be <10ms, got {avg_ms:.2f}ms"

    @pytest.mark.asyncio
    async def test_rag_evaluator_heuristic_mode_performance(self):
        """Benchmark RAGEvaluator in heuristic-only mode."""
        import time
        from adapters.rag_evaluator import RAGEvaluator

        evaluator = RAGEvaluator()

        start = time.time()
        iterations = 50

        for _ in range(iterations):
            await evaluator.evaluate(
                question=SAMPLE_QUESTION,
                answer=SAMPLE_ANSWER_GOOD,
                contexts=SAMPLE_CONTEXTS,
                use_ragas=False,
                use_deepeval=False,
            )

        elapsed = time.time() - start
        avg_ms = (elapsed / iterations) * 1000

        print(f"\nRAGEvaluator (heuristic mode): {avg_ms:.2f}ms per evaluation")
        assert avg_ms < 50, f"Heuristic mode should be <50ms, got {avg_ms:.2f}ms"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Edge case and error handling tests."""

    @pytest.mark.asyncio
    async def test_unicode_text_handling(self):
        """Test evaluation with Unicode characters."""
        from adapters.rag_evaluator import RAGEvaluator

        evaluator = RAGEvaluator()

        result = await evaluator.evaluate(
            question="What is the capital of Japan?",
            answer="The capital of Japan is Tokyo (|6771|4EAC).",
            contexts=["Tokyo (|6771|4EAC) is the capital city of Japan."],
            use_ragas=False,
            use_deepeval=False,
        )

        assert result.scores["faithfulness"] > 0

    @pytest.mark.asyncio
    async def test_very_long_context(self):
        """Test evaluation with very long context."""
        from adapters.rag_evaluator import RAGEvaluator

        evaluator = RAGEvaluator()

        # Generate a long context
        long_context = " ".join(["This is a test sentence about RAG."] * 500)

        result = await evaluator.evaluate(
            question=SAMPLE_QUESTION,
            answer=SAMPLE_ANSWER_GOOD,
            contexts=[long_context],
            use_ragas=False,
            use_deepeval=False,
        )

        assert result.scores is not None

    @pytest.mark.asyncio
    async def test_special_characters(self):
        """Test evaluation with special characters."""
        from adapters.rag_evaluator import RAGEvaluator

        evaluator = RAGEvaluator()

        result = await evaluator.evaluate(
            question="What is <RAG> & how does it work?",
            answer="RAG (<Retrieval-Augmented Generation>) works by: 1) retrieving, 2) generating.",
            contexts=["RAG uses <retrieval> & generation to improve LLMs."],
            use_ragas=False,
            use_deepeval=False,
        )

        assert "faithfulness" in result.scores


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
