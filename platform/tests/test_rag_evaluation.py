"""
test_rag_evaluation.py - Unit tests for RAG evaluation metrics.

Tests the Ragas-style evaluation module including:
- Context metrics (precision, recall, relevancy)
- Answer metrics (relevancy, correctness, faithfulness)
- Hallucination detection
- Ranking metrics (NDCG, MRR, F1)
- Quick evaluator (non-LLM based)

Usage:
    pytest platform/tests/test_rag_evaluation.py -v
    pytest platform/tests/test_rag_evaluation.py -v -k "test_ranking"
"""

import asyncio
import pytest
from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch


# =============================================================================
# MOCK LLM PROVIDER
# =============================================================================

class MockLLMProvider:
    """Mock LLM provider for testing without real API calls."""

    def __init__(self, responses: Optional[dict] = None):
        self.responses = responses or {}
        self.call_count = 0
        self.call_history: List[str] = []

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        self.call_count += 1
        self.call_history.append(prompt)

        # Return predefined response if available
        for key, response in self.responses.items():
            if key.lower() in prompt.lower():
                return response

        # Default responses based on prompt content
        if "extract" in prompt.lower() and "claim" in prompt.lower():
            return """CLAIM 1: RAG stands for Retrieval-Augmented Generation
CLAIM 2: RAG improves LLM accuracy by grounding responses in retrieved context
CLAIM 3: RAG reduces hallucinations"""

        if "verify" in prompt.lower() or "supported" in prompt.lower():
            return """VERDICT: SUPPORTED
CONFIDENCE: 0.85
REASONING: The claim is directly stated in the context.
SUPPORTING_TEXT: "RAG is a technique that combines retrieval with generation" """

        if "relevance" in prompt.lower():
            return """RELEVANCE: 0.85
IS_RELEVANT: YES
REASONING: The context directly addresses the question about RAG."""

        if "answer" in prompt.lower() and "relevancy" in prompt.lower():
            return """RELEVANCE: 0.9
ADDRESSES_QUESTION: FULLY
REASONING: The answer directly explains what RAG is."""

        if "correctness" in prompt.lower():
            return """CORRECTNESS: 0.85
FACTUAL_OVERLAP: HIGH
CONTRADICTIONS: None
REASONING: The answer matches the ground truth semantically."""

        if "hallucination" in prompt.lower():
            return """HALLUCINATION_DETECTED: NO
HALLUCINATED_CLAIMS: None
HALLUCINATION_SEVERITY: NONE
REASONING: All claims are supported by the context."""

        if "precision" in prompt.lower():
            return """PRECISION_SCORE: 0.8
RANKING_QUALITY: GOOD
REASONING: Relevant contexts are ranked higher."""

        if "recall" in prompt.lower():
            return """RECALL_SCORE: 0.75
COVERAGE: PARTIAL
MISSING_INFORMATION: Some implementation details
REASONING: Most key facts are covered."""

        return "Default response"


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def sample_question():
    return "What is RAG?"


@pytest.fixture
def sample_contexts():
    return [
        "RAG stands for Retrieval-Augmented Generation. It is a technique that combines "
        "information retrieval with text generation to improve LLM outputs.",
        "RAG systems first retrieve relevant documents from a knowledge base, then use "
        "those documents as context for generating responses.",
        "Benefits of RAG include reduced hallucinations, improved accuracy, and the ability "
        "to cite sources for generated content.",
    ]


@pytest.fixture
def sample_answer():
    return (
        "RAG, or Retrieval-Augmented Generation, is a technique that enhances LLM outputs "
        "by first retrieving relevant information from a knowledge base and using it as "
        "context for generation. This helps reduce hallucinations and improve accuracy."
    )


@pytest.fixture
def sample_ground_truth():
    return (
        "RAG (Retrieval-Augmented Generation) is an AI framework that retrieves relevant "
        "documents and uses them to ground LLM responses, reducing hallucinations and "
        "improving factual accuracy."
    )


# =============================================================================
# RANKING METRICS TESTS (NO LLM NEEDED)
# =============================================================================

class TestRankingMetrics:
    """Tests for ranking metrics calculations."""

    def test_dcg_at_k_basic(self):
        """Test DCG calculation with simple relevance scores."""
        from core.rag.evaluation import RankingMetrics

        # Perfect ranking: [1.0, 0.8, 0.6]
        relevances = [1.0, 0.8, 0.6]
        dcg = RankingMetrics.dcg_at_k(relevances, k=3)

        # DCG = 1.0 + 0.8/log2(3) + 0.6/log2(4)
        expected = 1.0 + 0.8 / 1.585 + 0.6 / 2.0
        assert abs(dcg - expected) < 0.01

    def test_dcg_at_k_empty(self):
        """Test DCG with empty relevances."""
        from core.rag.evaluation import RankingMetrics

        dcg = RankingMetrics.dcg_at_k([], k=5)
        assert dcg == 0.0

    def test_ndcg_perfect_ranking(self):
        """Test NDCG with perfect ranking."""
        from core.rag.evaluation import RankingMetrics

        # Perfect ranking (already sorted descending)
        relevances = [1.0, 0.8, 0.6, 0.4]
        ndcg = RankingMetrics.ndcg_at_k(relevances, k=4)
        assert ndcg == 1.0  # Perfect = 1.0

    def test_ndcg_imperfect_ranking(self):
        """Test NDCG with imperfect ranking."""
        from core.rag.evaluation import RankingMetrics

        # Imperfect ranking (relevant doc at position 3)
        relevances = [0.2, 0.3, 1.0, 0.4]
        ndcg = RankingMetrics.ndcg_at_k(relevances, k=4)
        assert 0.0 < ndcg < 1.0

    def test_mrr_first_relevant(self):
        """Test MRR when first result is relevant."""
        from core.rag.evaluation import RankingMetrics

        relevances = [0.8, 0.3, 0.2]  # First is relevant (>0.5)
        mrr = RankingMetrics.mrr(relevances)
        assert mrr == 1.0

    def test_mrr_second_relevant(self):
        """Test MRR when second result is first relevant."""
        from core.rag.evaluation import RankingMetrics

        relevances = [0.2, 0.8, 0.9]  # Second is first relevant
        mrr = RankingMetrics.mrr(relevances)
        assert mrr == 0.5

    def test_mrr_no_relevant(self):
        """Test MRR when no results are relevant."""
        from core.rag.evaluation import RankingMetrics

        relevances = [0.1, 0.2, 0.3]  # None >= 0.5
        mrr = RankingMetrics.mrr(relevances)
        assert mrr == 0.0

    def test_precision_at_k(self):
        """Test precision at k calculation."""
        from core.rag.evaluation import RankingMetrics

        relevances = [0.8, 0.7, 0.3, 0.9, 0.1]
        # At k=3: 2 relevant (0.8, 0.7) out of 3
        precision = RankingMetrics.precision_at_k(relevances, k=3)
        assert abs(precision - 2/3) < 0.01

    def test_recall_at_k(self):
        """Test recall at k calculation."""
        from core.rag.evaluation import RankingMetrics

        relevances = [0.8, 0.7, 0.3, 0.9, 0.1]
        # 3 total relevant, at k=3: 2 found
        recall = RankingMetrics.recall_at_k(relevances, k=3, total_relevant=3)
        assert abs(recall - 2/3) < 0.01

    def test_f1_score(self):
        """Test F1 score calculation."""
        from core.rag.evaluation import RankingMetrics

        precision = 0.8
        recall = 0.6
        f1 = RankingMetrics.f1_score(precision, recall)
        expected = 2 * (0.8 * 0.6) / (0.8 + 0.6)
        assert abs(f1 - expected) < 0.01

    def test_f1_score_zero(self):
        """Test F1 score when both are zero."""
        from core.rag.evaluation import RankingMetrics

        f1 = RankingMetrics.f1_score(0.0, 0.0)
        assert f1 == 0.0

    def test_token_overlap_f1_identical(self):
        """Test token overlap F1 with identical texts."""
        from core.rag.evaluation import RankingMetrics

        text = "RAG is a retrieval augmented generation technique"
        f1 = RankingMetrics.token_overlap_f1(text, text)
        assert f1 == 1.0

    def test_token_overlap_f1_partial(self):
        """Test token overlap F1 with partial overlap."""
        from core.rag.evaluation import RankingMetrics

        answer = "RAG improves LLM accuracy"
        ground_truth = "RAG enhances model accuracy and reduces errors"
        f1 = RankingMetrics.token_overlap_f1(answer, ground_truth)
        assert 0.0 < f1 < 1.0


# =============================================================================
# QUICK EVALUATOR TESTS (NO LLM NEEDED)
# =============================================================================

class TestQuickEvaluator:
    """Tests for QuickEvaluator (heuristic-based evaluation)."""

    def test_faithfulness_high_overlap(self):
        """Test faithfulness with high token overlap."""
        from core.rag.evaluation import QuickEvaluator

        evaluator = QuickEvaluator()
        answer = "RAG combines retrieval with generation for better results"
        contexts = ["RAG combines retrieval with generation to improve LLM outputs"]

        score = evaluator.evaluate_faithfulness(answer, contexts)
        assert score > 0.5

    def test_faithfulness_low_overlap(self):
        """Test faithfulness with low token overlap."""
        from core.rag.evaluation import QuickEvaluator

        evaluator = QuickEvaluator()
        answer = "Machine learning models require training data"
        contexts = ["RAG is a retrieval augmented generation technique"]

        score = evaluator.evaluate_faithfulness(answer, contexts)
        assert score < 0.5

    def test_faithfulness_empty_contexts(self):
        """Test faithfulness with empty contexts."""
        from core.rag.evaluation import QuickEvaluator

        evaluator = QuickEvaluator()
        score = evaluator.evaluate_faithfulness("any answer", [])
        assert score == 0.0

    def test_answer_relevancy(self):
        """Test answer relevancy scoring."""
        from core.rag.evaluation import QuickEvaluator

        evaluator = QuickEvaluator()
        question = "What is RAG architecture?"
        answer = "RAG architecture combines retrieval systems with generation models"

        score = evaluator.evaluate_answer_relevancy(question, answer)
        assert score > 0.3

    def test_context_relevancy(self):
        """Test context relevancy scoring."""
        from core.rag.evaluation import QuickEvaluator

        evaluator = QuickEvaluator()
        question = "How does RAG reduce hallucinations?"
        contexts = [
            "RAG reduces hallucinations by grounding responses in retrieved facts",
            "The weather today is sunny and warm",
        ]

        score = evaluator.evaluate_context_relevancy(question, contexts)
        assert 0.0 < score < 1.0

    def test_evaluate_all(self):
        """Test full quick evaluation."""
        from core.rag.evaluation import QuickEvaluator

        evaluator = QuickEvaluator()
        scores = evaluator.evaluate_all(
            question="What is RAG?",
            contexts=["RAG is retrieval augmented generation"],
            answer="RAG stands for retrieval augmented generation"
        )

        assert "faithfulness" in scores
        assert "answer_relevancy" in scores
        assert "context_relevancy" in scores
        assert all(0.0 <= v <= 1.0 for v in scores.values())


# =============================================================================
# CLAIMS EXTRACTION TESTS
# =============================================================================

class TestClaimsExtractor:
    """Tests for claims extraction."""

    @pytest.mark.asyncio
    async def test_extract_claims(self, mock_llm):
        """Test extracting claims from answer."""
        from core.rag.evaluation import ClaimsExtractor

        extractor = ClaimsExtractor(mock_llm, max_claims=5)
        claims = await extractor.extract(
            "RAG stands for Retrieval-Augmented Generation. "
            "It improves LLM accuracy. It reduces hallucinations."
        )

        assert len(claims) > 0
        assert all(hasattr(c, 'text') for c in claims)
        assert all(hasattr(c, 'confidence') for c in claims)

    @pytest.mark.asyncio
    async def test_extract_claims_fallback(self):
        """Test fallback extraction on LLM error."""
        from core.rag.evaluation import ClaimsExtractor

        # Mock LLM that raises error
        class ErrorLLM:
            async def generate(self, *args, **kwargs):
                raise Exception("API Error")

        extractor = ClaimsExtractor(ErrorLLM(), max_claims=5)
        claims = await extractor.extract(
            "This is a sentence. This is another sentence."
        )

        # Should use fallback sentence splitting
        assert len(claims) >= 1


# =============================================================================
# CLAIMS VERIFIER TESTS
# =============================================================================

class TestClaimsVerifier:
    """Tests for claims verification."""

    @pytest.mark.asyncio
    async def test_verify_supported_claim(self, mock_llm):
        """Test verifying a supported claim."""
        from core.rag.evaluation import ClaimsVerifier, Claim

        verifier = ClaimsVerifier(mock_llm)
        claim = Claim(
            text="RAG combines retrieval with generation",
            source_sentence="RAG combines retrieval with generation"
        )
        contexts = ["RAG is a technique that combines retrieval with generation"]

        verified = await verifier.verify(claim, contexts)

        assert verified.is_supported is True
        assert verified.confidence > 0.5

    @pytest.mark.asyncio
    async def test_verify_batch(self, mock_llm):
        """Test batch verification of claims."""
        from core.rag.evaluation import ClaimsVerifier, Claim

        verifier = ClaimsVerifier(mock_llm)
        claims = [
            Claim(text="Claim 1", source_sentence="Claim 1"),
            Claim(text="Claim 2", source_sentence="Claim 2"),
        ]
        contexts = ["Context supporting the claims"]

        verified = await verifier.verify_batch(claims, contexts, batch_size=2)

        assert len(verified) == 2


# =============================================================================
# FAITHFULNESS CALCULATOR TESTS
# =============================================================================

class TestFaithfulnessCalculator:
    """Tests for faithfulness calculation."""

    @pytest.mark.asyncio
    async def test_calculate_faithfulness(self, mock_llm):
        """Test faithfulness score calculation."""
        from core.rag.evaluation import FaithfulnessCalculator, EvaluationConfig

        config = EvaluationConfig()
        calculator = FaithfulnessCalculator(mock_llm, config)

        score, claims = await calculator.calculate(
            answer="RAG improves accuracy by using retrieved context",
            contexts=["RAG retrieves documents to improve generation accuracy"]
        )

        assert 0.0 <= score <= 1.0
        assert isinstance(claims, list)


# =============================================================================
# CONTEXT RELEVANCE CALCULATOR TESTS
# =============================================================================

class TestContextRelevanceCalculator:
    """Tests for context relevance calculation."""

    @pytest.mark.asyncio
    async def test_calculate_single(self, mock_llm):
        """Test single context relevance calculation."""
        from core.rag.evaluation import ContextRelevanceCalculator

        calculator = ContextRelevanceCalculator(mock_llm)

        relevance = await calculator.calculate_single(
            question="What is RAG?",
            context="RAG is retrieval augmented generation"
        )

        assert 0.0 <= relevance.relevance_score <= 1.0
        assert isinstance(relevance.is_relevant, bool)

    @pytest.mark.asyncio
    async def test_calculate_batch(self, mock_llm):
        """Test batch context relevance calculation."""
        from core.rag.evaluation import ContextRelevanceCalculator

        calculator = ContextRelevanceCalculator(mock_llm)

        relevances = await calculator.calculate_batch(
            question="What is RAG?",
            contexts=["Context 1", "Context 2", "Context 3"]
        )

        assert len(relevances) == 3
        assert all(0.0 <= r.relevance_score <= 1.0 for r in relevances)


# =============================================================================
# HALLUCINATION DETECTOR TESTS
# =============================================================================

class TestHallucinationDetector:
    """Tests for hallucination detection."""

    @pytest.mark.asyncio
    async def test_detect_no_hallucination(self, mock_llm):
        """Test detection when no hallucination present."""
        from core.rag.evaluation import HallucinationDetector

        detector = HallucinationDetector(mock_llm)

        score, claims, reasoning = await detector.detect(
            question="What is RAG?",
            answer="RAG is retrieval augmented generation",
            contexts=["RAG stands for retrieval augmented generation"]
        )

        assert score < 0.5  # Low hallucination score
        assert isinstance(claims, list)

    @pytest.mark.asyncio
    async def test_detect_with_hallucination(self):
        """Test detection with hallucination present."""
        from core.rag.evaluation import HallucinationDetector

        # Mock LLM that reports hallucination
        class HallucinationLLM:
            async def generate(self, *args, **kwargs):
                return """HALLUCINATION_DETECTED: YES
HALLUCINATED_CLAIMS: The answer claims RAG was invented in 2015, but this is not in the context
HALLUCINATION_SEVERITY: MEDIUM
REASONING: The date claim is fabricated"""

        detector = HallucinationDetector(HallucinationLLM())

        score, claims, reasoning = await detector.detect(
            question="What is RAG?",
            answer="RAG was invented in 2015",
            contexts=["RAG is a technique for improving LLMs"]
        )

        assert score > 0.3  # Higher hallucination score
        assert len(claims) > 0


# =============================================================================
# RAG EVALUATOR INTEGRATION TESTS
# =============================================================================

class TestRAGEvaluator:
    """Integration tests for RAGEvaluator."""

    @pytest.mark.asyncio
    async def test_evaluate_single_sample(
        self,
        mock_llm,
        sample_question,
        sample_contexts,
        sample_answer
    ):
        """Test evaluating a single sample."""
        from core.rag.evaluation import RAGEvaluator, EvaluationConfig

        config = EvaluationConfig(
            faithfulness_threshold=0.6,
            answer_relevancy_threshold=0.6,
        )
        evaluator = RAGEvaluator(llm=mock_llm, config=config)

        sample = await evaluator.evaluate_single(
            question=sample_question,
            contexts=sample_contexts,
            answer=sample_answer
        )

        assert 0.0 <= sample.faithfulness <= 1.0
        assert 0.0 <= sample.answer_relevancy <= 1.0
        assert 0.0 <= sample.context_precision <= 1.0
        assert hasattr(sample, 'passed')

    @pytest.mark.asyncio
    async def test_evaluate_batch(
        self,
        mock_llm,
        sample_question,
        sample_contexts,
        sample_answer,
        sample_ground_truth
    ):
        """Test evaluating multiple samples."""
        from core.rag.evaluation import RAGEvaluator

        evaluator = RAGEvaluator(llm=mock_llm)

        results = await evaluator.evaluate(
            questions=[sample_question, sample_question],
            retrieved_contexts=[sample_contexts, sample_contexts],
            generated_answers=[sample_answer, sample_answer],
            ground_truth=[sample_ground_truth, sample_ground_truth]
        )

        assert results.total_samples == 2
        assert len(results.samples) == 2
        assert 0.0 <= results.faithfulness <= 1.0
        assert 0.0 <= results.answer_relevancy <= 1.0
        assert hasattr(results, 'recommendations')

    @pytest.mark.asyncio
    async def test_evaluate_generates_recommendations(self, mock_llm):
        """Test that evaluation generates recommendations."""
        from core.rag.evaluation import RAGEvaluator, EvaluationConfig

        config = EvaluationConfig(
            faithfulness_threshold=0.99,  # Set high to trigger recommendation
        )
        evaluator = RAGEvaluator(llm=mock_llm, config=config)

        results = await evaluator.evaluate(
            questions=["What is RAG?"],
            retrieved_contexts=[["RAG is retrieval augmented generation"]],
            generated_answers=["RAG improves LLMs"]
        )

        assert len(results.recommendations) > 0

    @pytest.mark.asyncio
    async def test_evaluate_with_relevance_labels(self, mock_llm):
        """Test evaluation with provided relevance labels."""
        from core.rag.evaluation import RAGEvaluator, EvaluationConfig

        config = EvaluationConfig(ndcg_k=3)
        evaluator = RAGEvaluator(llm=mock_llm, config=config)

        results = await evaluator.evaluate(
            questions=["What is RAG?"],
            retrieved_contexts=[["Ctx1", "Ctx2", "Ctx3"]],
            generated_answers=["Answer"],
            relevance_labels=[[0.9, 0.6, 0.3]]
        )

        assert results.ndcg_at_k > 0
        assert results.mrr > 0

    def test_result_to_dict(self, mock_llm):
        """Test EvaluationResult serialization."""
        from core.rag.evaluation import EvaluationResult

        result = EvaluationResult(
            faithfulness=0.85,
            answer_relevancy=0.9,
            context_precision=0.8,
            total_samples=1,
            overall_passed=True
        )

        d = result.to_dict()

        assert "scores" in d
        assert "summary" in d
        assert d["scores"]["faithfulness"] == 0.85
        assert d["summary"]["total_samples"] == 1


# =============================================================================
# DATA STRUCTURE TESTS
# =============================================================================

class TestDataStructures:
    """Tests for data structures."""

    def test_claim_dataclass(self):
        """Test Claim dataclass."""
        from core.rag.evaluation import Claim

        claim = Claim(
            text="Test claim",
            source_sentence="Source sentence",
            confidence=0.9,
            is_supported=True
        )

        assert claim.text == "Test claim"
        assert claim.confidence == 0.9
        assert claim.is_supported is True

    def test_context_relevance_dataclass(self):
        """Test ContextRelevance dataclass."""
        from core.rag.evaluation import ContextRelevance

        cr = ContextRelevance(
            context="Test context",
            relevance_score=0.85,
            is_relevant=True,
            reasoning="Highly relevant"
        )

        assert cr.relevance_score == 0.85
        assert cr.is_relevant is True

    def test_evaluation_config_defaults(self):
        """Test EvaluationConfig default values."""
        from core.rag.evaluation import EvaluationConfig

        config = EvaluationConfig()

        assert config.faithfulness_threshold == 0.7
        assert config.answer_relevancy_threshold == 0.7
        assert config.hallucination_threshold == 0.3
        assert config.ndcg_k == 10

    def test_sample_evaluation_failed_metrics(self):
        """Test SampleEvaluation failed_metrics tracking."""
        from core.rag.evaluation import SampleEvaluation

        sample = SampleEvaluation(
            question="Q",
            answer="A",
            contexts=["C"],
            faithfulness=0.5,  # Below default threshold
            passed=False,
            failed_metrics=["faithfulness"]
        )

        assert "faithfulness" in sample.failed_metrics
        assert sample.passed is False


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_evaluator(self, mock_llm):
        """Test create_evaluator factory function."""
        from core.rag.evaluation import create_evaluator

        evaluator = create_evaluator(
            llm=mock_llm,
            faithfulness_threshold=0.8,
            ndcg_k=5
        )

        assert evaluator.config.faithfulness_threshold == 0.8
        assert evaluator.config.ndcg_k == 5

    def test_create_evaluator_with_config(self, mock_llm):
        """Test create_evaluator with explicit config."""
        from core.rag.evaluation import create_evaluator, EvaluationConfig

        config = EvaluationConfig(
            faithfulness_threshold=0.9,
            batch_size=10
        )
        evaluator = create_evaluator(llm=mock_llm, config=config)

        assert evaluator.config.faithfulness_threshold == 0.9
        assert evaluator.config.batch_size == 10


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_evaluate_mismatched_lengths(self, mock_llm):
        """Test that mismatched input lengths raise error."""
        from core.rag.evaluation import RAGEvaluator

        evaluator = RAGEvaluator(llm=mock_llm)

        with pytest.raises(ValueError):
            await evaluator.evaluate(
                questions=["Q1", "Q2"],
                retrieved_contexts=[["C1"]],  # Mismatched length
                generated_answers=["A1", "A2"]
            )

    @pytest.mark.asyncio
    async def test_graceful_llm_failure(self):
        """Test graceful handling of LLM failures."""
        from core.rag.evaluation import RAGEvaluator

        class FailingLLM:
            async def generate(self, *args, **kwargs):
                raise Exception("API Error")

        evaluator = RAGEvaluator(llm=FailingLLM())

        # Should not raise, should return default/partial results
        result = await evaluator.evaluate_single(
            question="What is RAG?",
            contexts=["RAG context"],
            answer="RAG answer"
        )

        # Metrics should have some default values
        assert hasattr(result, 'faithfulness')


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
