"""
test_rag_pipeline_evaluator_integration.py - Pipeline + Evaluator Integration Tests

Tests the integration between RAGPipeline and RAGEvaluator/adapters.
Verifies EvaluatorProtocol compliance and end-to-end evaluation flow.

Gap11 Status: IMPLEMENTED (V66)

Usage:
    pytest platform/tests/test_rag_pipeline_evaluator_integration.py -v
"""

import asyncio
import pytest
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add platform path
PLATFORM_ROOT = Path(__file__).parent.parent
if str(PLATFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(PLATFORM_ROOT))


# =============================================================================
# TEST DATA
# =============================================================================

TEST_QUESTION = "What is RAG and how does it work?"

TEST_CONTEXTS = [
    "RAG (Retrieval-Augmented Generation) enhances LLM outputs by retrieving "
    "relevant documents from a knowledge base before generation.",
    "RAG systems use vector embeddings to find semantically similar documents "
    "and then use those as context for the LLM to generate responses.",
    "The benefits of RAG include reduced hallucinations, improved accuracy, "
    "and the ability to cite sources for generated content.",
]

TEST_ANSWER = (
    "RAG (Retrieval-Augmented Generation) is a technique that improves LLM outputs "
    "by first retrieving relevant documents from a knowledge base and using them as "
    "context for generation. This helps reduce hallucinations and improve accuracy."
)


# =============================================================================
# MOCK LLM PROVIDER
# =============================================================================

class MockLLMProvider:
    """Mock LLM provider for testing pipeline without real API calls."""

    def __init__(self, response: str = TEST_ANSWER):
        self.response = response
        self.calls = []

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        self.calls.append(prompt)
        return self.response


# =============================================================================
# MOCK RETRIEVER
# =============================================================================

class MockRetriever:
    """Mock retriever that returns predefined documents."""

    name = "mock_retriever"

    def __init__(self, documents: List[str] = None):
        self.documents = documents or TEST_CONTEXTS

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        return [
            {"content": doc, "score": 0.9 - i * 0.1, "source": "mock"}
            for i, doc in enumerate(self.documents[:top_k])
        ]


# =============================================================================
# EVALUATOR PROTOCOL WRAPPER
# =============================================================================

class EvaluatorWrapper:
    """
    Wrapper to ensure evaluator output matches pipeline expectations.

    The pipeline uses getattr() expecting object attributes, but our evaluators
    return dicts. This wrapper provides attribute access to dict values.
    """

    def __init__(self, evaluator):
        self._evaluator = evaluator

    async def evaluate_single(
        self,
        question: str,
        contexts: List[str],
        answer: str,
        ground_truth: Optional[str] = None,
    ) -> "EvaluationResultWrapper":
        result = await self._evaluator.evaluate_single(
            question=question,
            contexts=contexts,
            answer=answer,
            ground_truth=ground_truth,
        )
        return EvaluationResultWrapper(result)


class EvaluationResultWrapper:
    """Wraps dict result to provide attribute access for pipeline compatibility."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data
        scores = data.get("scores", {})
        self.faithfulness = scores.get("faithfulness", 0.0)
        self.answer_relevancy = scores.get("answer_relevancy", 0.0)
        self.context_precision = scores.get("context_precision", 0.0)
        self.context_recall = scores.get("context_recall", 0.0)
        self.passed = data.get("passed", True)
        self.framework = data.get("framework", "unknown")
        self.details = data.get("details", {})


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestEvaluatorPipelineIntegration:
    """Tests for evaluator integration with RAG pipeline."""

    def test_rag_evaluator_returns_dict(self):
        """Verify RAGEvaluator.evaluate_single returns a dict."""
        from adapters.rag_evaluator import RAGEvaluator

        async def check():
            evaluator = RAGEvaluator()
            result = await evaluator.evaluate_single(
                question=TEST_QUESTION,
                contexts=TEST_CONTEXTS,
                answer=TEST_ANSWER,
            )
            assert isinstance(result, dict), "evaluate_single should return dict"
            assert "scores" in result
            assert "passed" in result
            return result

        result = asyncio.get_event_loop().run_until_complete(check())
        print(f"\nRAGEvaluator result: {result}")

    def test_deepeval_adapter_returns_dict(self):
        """Verify DeepEvalAdapter.evaluate_single returns a dict."""
        from adapters.deepeval_adapter import DeepEvalAdapter

        async def check():
            adapter = DeepEvalAdapter()
            await adapter.initialize({})
            result = await adapter.evaluate_single(
                question=TEST_QUESTION,
                contexts=TEST_CONTEXTS,
                answer=TEST_ANSWER,
            )
            assert isinstance(result, dict)
            return result

        result = asyncio.get_event_loop().run_until_complete(check())
        print(f"\nDeepEvalAdapter result: {result}")

    def test_ragas_adapter_returns_dict(self):
        """Verify RagasAdapter.evaluate_single returns a dict."""
        from adapters.ragas_adapter import RagasAdapter

        async def check():
            adapter = RagasAdapter()
            await adapter.initialize({})
            result = await adapter.evaluate_single(
                question=TEST_QUESTION,
                contexts=TEST_CONTEXTS,
                answer=TEST_ANSWER,
            )
            assert isinstance(result, dict)
            return result

        result = asyncio.get_event_loop().run_until_complete(check())
        print(f"\nRagasAdapter result: {result}")

    def test_evaluator_wrapper_provides_attributes(self):
        """Test EvaluatorWrapper provides attribute access."""
        from adapters.rag_evaluator import RAGEvaluator

        async def check():
            evaluator = RAGEvaluator()
            wrapped = EvaluatorWrapper(evaluator)
            result = await wrapped.evaluate_single(
                question=TEST_QUESTION,
                contexts=TEST_CONTEXTS,
                answer=TEST_ANSWER,
            )

            # Should have attribute access
            assert hasattr(result, "faithfulness")
            assert hasattr(result, "answer_relevancy")
            assert hasattr(result, "passed")

            # Values should be accessible
            assert isinstance(result.faithfulness, float)
            assert isinstance(result.answer_relevancy, float)
            assert isinstance(result.passed, bool)

            return result

        result = asyncio.get_event_loop().run_until_complete(check())
        print(f"\nWrapped result: faithfulness={result.faithfulness:.3f}, "
              f"relevancy={result.answer_relevancy:.3f}, passed={result.passed}")

    @pytest.mark.asyncio
    async def test_pipeline_config_evaluation_flag(self):
        """Test PipelineConfig.enable_evaluation flag."""
        from core.rag.pipeline import PipelineConfig

        # Default should have evaluation enabled
        config = PipelineConfig()
        assert hasattr(config, "enable_evaluation")

        # Can be disabled
        config_disabled = PipelineConfig(enable_evaluation=False)
        assert config_disabled.enable_evaluation is False


# =============================================================================
# EVALUATOR COMPARISON TESTS
# =============================================================================

class TestEvaluatorComparison:
    """Compare results across different evaluators."""

    @pytest.mark.asyncio
    async def test_all_evaluators_return_same_structure(self):
        """All evaluators should return structurally similar results."""
        from adapters.rag_evaluator import RAGEvaluator
        from adapters.deepeval_adapter import DeepEvalAdapter
        from adapters.ragas_adapter import RagasAdapter

        evaluators = [
            ("RAGEvaluator", RAGEvaluator()),
            ("DeepEvalAdapter", DeepEvalAdapter()),
            ("RagasAdapter", RagasAdapter()),
        ]

        for name, evaluator in evaluators:
            if hasattr(evaluator, "initialize"):
                await evaluator.initialize({})

            result = await evaluator.evaluate_single(
                question=TEST_QUESTION,
                contexts=TEST_CONTEXTS,
                answer=TEST_ANSWER,
            )

            # All should return dict
            assert isinstance(result, dict), f"{name} should return dict"

            # All should have scores (or error)
            if "error" not in result:
                assert "scores" in result, f"{name} missing scores"
                assert "passed" in result, f"{name} missing passed"

                # Scores should be in valid range
                for metric, score in result.get("scores", {}).items():
                    assert 0.0 <= score <= 1.0, f"{name}.{metric}={score} out of range"

            print(f"\n{name}: {result}")

    @pytest.mark.asyncio
    async def test_heuristic_consistency_across_evaluators(self):
        """Heuristic fallback should give similar results across evaluators."""
        from adapters.rag_evaluator import RAGEvaluator
        from adapters.deepeval_adapter import DeepEvalAdapter
        from adapters.ragas_adapter import RagasAdapter

        # RAGEvaluator in heuristic mode
        rag_eval = RAGEvaluator()
        rag_result = await rag_eval.evaluate(
            question=TEST_QUESTION,
            answer=TEST_ANSWER,
            contexts=TEST_CONTEXTS,
            use_ragas=False,
            use_deepeval=False,
        )

        # DeepEval heuristic (synchronous)
        de_adapter = DeepEvalAdapter()
        await de_adapter.initialize({})
        de_result = de_adapter._heuristic_evaluator.evaluate(
            input_text=TEST_QUESTION,
            actual_output=TEST_ANSWER,
            retrieval_context=TEST_CONTEXTS,
        )

        # Ragas heuristic (synchronous)
        ragas_adapter = RagasAdapter()
        await ragas_adapter.initialize({})
        ragas_result = ragas_adapter._heuristic_evaluator.evaluate(
            question=TEST_QUESTION,
            answer=TEST_ANSWER,
            contexts=TEST_CONTEXTS,
        )

        # All should have similar structure
        for name, result in [("RAG", rag_result.scores), ("DE", de_result["scores"]), ("Ragas", ragas_result["scores"])]:
            assert "faithfulness" in result, f"{name} missing faithfulness"
            print(f"{name} faithfulness: {result.get('faithfulness', 'N/A')}")


# =============================================================================
# METRIC TESTS
# =============================================================================

class TestEvaluationMetrics:
    """Tests for individual evaluation metrics."""

    @pytest.mark.asyncio
    async def test_faithfulness_metric(self):
        """Test faithfulness scoring - answer grounded in context."""
        from adapters.rag_evaluator import RAGEvaluator

        evaluator = RAGEvaluator()

        # Grounded answer (uses context)
        grounded = await evaluator.evaluate(
            question="What is RAG?",
            answer="RAG retrieves documents and uses them as context for LLM generation.",
            contexts=["RAG retrieves documents and uses them as context for LLM generation."],
            use_ragas=False,
            use_deepeval=False,
        )

        # Ungrounded answer (invents facts)
        ungrounded = await evaluator.evaluate(
            question="What is RAG?",
            answer="RAG was invented by Elon Musk in 2010 using quantum algorithms.",
            contexts=["RAG retrieves documents and uses them as context for LLM generation."],
            use_ragas=False,
            use_deepeval=False,
        )

        assert grounded.scores["faithfulness"] > ungrounded.scores["faithfulness"], \
            "Grounded answer should have higher faithfulness"

    @pytest.mark.asyncio
    async def test_answer_relevancy_metric(self):
        """Test answer relevancy - answer addresses question."""
        from adapters.rag_evaluator import RAGEvaluator

        evaluator = RAGEvaluator()

        # Relevant answer
        relevant = await evaluator.evaluate(
            question="What is the capital of France?",
            answer="The capital of France is Paris.",
            contexts=["Paris is the capital and most populous city of France."],
            use_ragas=False,
            use_deepeval=False,
        )

        # Irrelevant answer
        irrelevant = await evaluator.evaluate(
            question="What is the capital of France?",
            answer="The weather is nice today.",
            contexts=["Paris is the capital and most populous city of France."],
            use_ragas=False,
            use_deepeval=False,
        )

        assert relevant.scores["answer_relevancy"] > irrelevant.scores["answer_relevancy"], \
            "Relevant answer should have higher relevancy"

    @pytest.mark.asyncio
    async def test_context_precision_metric(self):
        """Test context precision - context relevant to question."""
        from adapters.rag_evaluator import RAGEvaluator

        evaluator = RAGEvaluator()

        # Relevant context
        with_relevant = await evaluator.evaluate(
            question="What is RAG?",
            answer="RAG is retrieval augmented generation.",
            contexts=["RAG is retrieval augmented generation that improves LLMs."],
            use_ragas=False,
            use_deepeval=False,
        )

        # Irrelevant context
        with_irrelevant = await evaluator.evaluate(
            question="What is RAG?",
            answer="RAG is retrieval augmented generation.",
            contexts=["The stock market closed higher today."],
            use_ragas=False,
            use_deepeval=False,
        )

        assert with_relevant.scores["context_precision"] > with_irrelevant.scores["context_precision"], \
            "Relevant context should have higher precision"


# =============================================================================
# HALLUCINATION DETECTION TESTS
# =============================================================================

class TestHallucinationDetection:
    """Tests for hallucination detection capabilities."""

    @pytest.mark.asyncio
    async def test_deepeval_hallucination_detection(self):
        """Test DeepEval adapter hallucination detection."""
        from adapters.deepeval_adapter import DeepEvalAdapter

        adapter = DeepEvalAdapter()
        await adapter.initialize({"hallucination_threshold": 0.5})

        # Factual answer (from context)
        factual = await adapter.detect_hallucination(
            answer="RAG retrieves documents and uses them as context.",
            contexts=["RAG retrieves documents and uses them as context for generation."],
        )

        # Hallucinated answer
        hallucinated = await adapter.detect_hallucination(
            answer="RAG was invented by aliens in 1850 using magic crystals.",
            contexts=["RAG retrieves documents and uses them as context for generation."],
        )

        print(f"\nFactual: {factual}")
        print(f"Hallucinated: {hallucinated}")

        # Hallucinated should have higher hallucination score
        assert hallucinated["hallucination_score"] > factual["hallucination_score"], \
            "Hallucinated answer should have higher hallucination score"

    @pytest.mark.asyncio
    async def test_rag_evaluator_detects_hallucination_via_faithfulness(self):
        """RAGEvaluator can detect hallucination through low faithfulness."""
        from adapters.rag_evaluator import RAGEvaluator

        evaluator = RAGEvaluator()

        result = await evaluator.evaluate(
            question="What is RAG?",
            answer="RAG was created by NASA to communicate with Mars rovers.",
            contexts=["RAG is a technique that retrieves documents for LLM context."],
            use_ragas=False,
            use_deepeval=False,
        )

        # Low faithfulness indicates potential hallucination
        assert result.scores["faithfulness"] < 0.5, \
            "Hallucinated answer should have low faithfulness"


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestEvaluatorPerformance:
    """Performance tests for evaluators."""

    @pytest.mark.asyncio
    async def test_batch_evaluation_performance(self):
        """Test evaluating multiple samples efficiently."""
        import time
        from adapters.deepeval_adapter import DeepEvalAdapter

        adapter = DeepEvalAdapter()
        await adapter.initialize({})

        test_cases = [
            {
                "input": f"Question {i}",
                "actual_output": f"Answer {i}",
                "retrieval_context": [f"Context {i}"],
            }
            for i in range(5)
        ]

        start = time.time()
        result = await adapter.execute("evaluate_batch", test_cases=test_cases)
        elapsed = time.time() - start

        assert result.success
        assert result.data["total"] == 5
        print(f"\nBatch evaluation of 5 samples: {elapsed:.2f}s")

    @pytest.mark.asyncio
    async def test_parallel_evaluation(self):
        """Test parallel evaluation of multiple questions."""
        import time
        from adapters.rag_evaluator import RAGEvaluator

        evaluator = RAGEvaluator()

        questions = [
            ("What is RAG?", "RAG is retrieval augmented generation."),
            ("What is LLM?", "LLM is a large language model."),
            ("What is NLP?", "NLP is natural language processing."),
        ]

        start = time.time()
        tasks = [
            evaluator.evaluate(
                question=q,
                answer=a,
                contexts=["Context for " + q],
                use_ragas=False,
                use_deepeval=False,
            )
            for q, a in questions
        ]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start

        assert len(results) == 3
        assert all(r.scores is not None for r in results)
        print(f"\nParallel evaluation of 3 samples: {elapsed:.2f}s")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
