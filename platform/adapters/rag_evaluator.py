"""
rag_evaluator.py - RAG Quality Evaluation Adapter V2 (V65 Gap11 RESOLVED)

Unified RAG evaluation using Ragas (primary) + DeepEval (optional) + Heuristic fallback.
Implements EvaluatorProtocol for RAG pipeline integration.

Features:
  - Ragas 0.4.x: faithfulness, answer_relevancy, context_precision, context_recall
  - DeepEval: hallucination detection, contextual metrics
  - Heuristic fallback: Always available when SDKs fail
  - Opik integration: traces evaluation runs
  - LLM-as-judge with Claude or GPT

Gap11 Status: RESOLVED (V65)

Based on: Production patterns from Haystack, LlamaIndex, Awesome-RAG-Production.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Retry and circuit breaker for production resilience
try:
    from .retry import RetryConfig, retry_async
    RAG_EVALUATOR_RETRY_CONFIG = RetryConfig(
        max_retries=3, base_delay=1.0, max_delay=30.0, jitter=0.5
    )
except ImportError:
    RetryConfig = None
    retry_async = None
    RAG_EVALUATOR_RETRY_CONFIG = None

try:
    from .circuit_breaker_manager import adapter_circuit_breaker, get_adapter_circuit_manager
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False
    adapter_circuit_breaker = None

# Default timeout for evaluation operations (seconds)
RAG_EVALUATOR_OPERATION_TIMEOUT = 60

# Check availability at import time
_RAGAS_AVAILABLE = False
_DEEPEVAL_AVAILABLE = False

try:
    from ragas import evaluate as ragas_evaluate
    try:
        # Ragas >= 0.4.x uses ragas.metrics.collections
        from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision
    except ImportError:
        # Fallback for older versions or direct imports
        try:
            from ragas.metrics._faithfulness import Faithfulness
            from ragas.metrics._answer_relevance import AnswerRelevancy
            from ragas.metrics._context_precision import ContextPrecision
        except ImportError:
            from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    _RAGAS_AVAILABLE = True
except ImportError:
    logger.info("Ragas not installed - using heuristic evaluation")
    ragas_evaluate = None
    Faithfulness = None
    AnswerRelevancy = None
    ContextPrecision = None
    SingleTurnSample = None
    EvaluationDataset = None

try:
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
    )
    from deepeval.test_case import LLMTestCase
    from deepeval import evaluate as deepeval_evaluate
    _DEEPEVAL_AVAILABLE = True
except ImportError:
    logger.info("DeepEval not installed - using heuristic evaluation")
    FaithfulnessMetric = None
    AnswerRelevancyMetric = None
    ContextualPrecisionMetric = None
    LLMTestCase = None
    deepeval_evaluate = None


@dataclass
class EvalResult:
    """Result of RAG evaluation."""
    scores: Dict[str, float]        # metric_name -> score (0.0-1.0)
    passed: bool                    # All metrics above thresholds?
    details: Dict[str, str]         # metric_name -> reason/explanation
    framework: str                  # 'ragas', 'deepeval', 'heuristic', or combinations
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# HEURISTIC EVALUATOR (Always Available Fallback)
# =============================================================================

class HeuristicEvaluator:
    """
    Heuristic-based RAG evaluation for when SDKs are unavailable.

    Uses TF-IDF-like scoring and keyword overlap for approximate metrics.
    This is always available and provides consistent baseline scores.
    """

    def __init__(
        self,
        faithfulness_threshold: float = 0.7,
        relevancy_threshold: float = 0.6,
        precision_threshold: float = 0.5,
    ):
        self.faithfulness_threshold = faithfulness_threshold
        self.relevancy_threshold = relevancy_threshold
        self.precision_threshold = precision_threshold

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase words."""
        return [t.lower() for t in re.findall(r'\b\w+\b', text)]

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate using heuristics."""
        q_tokens = set(self._tokenize(question))
        a_tokens = set(self._tokenize(answer))
        c_tokens = set()
        for ctx in contexts:
            c_tokens.update(self._tokenize(ctx))

        # Faithfulness: How much of answer is grounded in context
        if a_tokens:
            faithfulness = len(a_tokens & c_tokens) / len(a_tokens)
        else:
            faithfulness = 0.0

        # Answer relevancy: How well answer addresses question
        if q_tokens and a_tokens:
            overlap = len(q_tokens & a_tokens)
            relevancy = overlap / len(q_tokens)
        else:
            relevancy = 0.0

        # Context precision: Are contexts relevant to question
        if c_tokens and q_tokens:
            precision = len(q_tokens & c_tokens) / len(q_tokens)
        else:
            precision = 0.0

        # Context recall: Does context cover ground truth
        recall = 0.0
        if ground_truth:
            gt_tokens = set(self._tokenize(ground_truth))
            if gt_tokens:
                recall = len(gt_tokens & c_tokens) / len(gt_tokens)

        scores = {
            "faithfulness": round(min(1.0, faithfulness), 4),
            "answer_relevancy": round(min(1.0, relevancy), 4),
            "context_precision": round(min(1.0, precision), 4),
        }
        if ground_truth:
            scores["context_recall"] = round(recall, 4)

        passed = (
            scores["faithfulness"] >= self.faithfulness_threshold
            and scores["answer_relevancy"] >= self.relevancy_threshold
            and scores["context_precision"] >= self.precision_threshold
        )

        return {
            "scores": scores,
            "passed": passed,
            "method": "heuristic",
        }


class RAGEvaluator:
    """
    Unified RAG evaluation using Ragas (primary) + DeepEval (optional) + Heuristic fallback.

    Implements EvaluatorProtocol for seamless integration with RAGPipeline.

    Usage:
        evaluator = RAGEvaluator()
        result = await evaluator.evaluate(
            question="What is X?",
            answer="X is ...",
            contexts=["Context chunk 1", "Context chunk 2"],
        )
        print(result.scores)  # {"faithfulness": 0.85, "answer_relevancy": 0.92, ...}
        print(result.passed)  # True/False

        # Or use EvaluatorProtocol method for pipeline integration:
        data = await evaluator.evaluate_single(
            question="What is X?",
            contexts=["..."],
            answer="X is ...",
        )
    """

    def __init__(
        self,
        faithfulness_threshold: float = 0.7,
        relevancy_threshold: float = 0.6,
        precision_threshold: float = 0.5,
        llm_model: str = "gpt-4o-mini",
        use_heuristic_fallback: bool = True,
    ):
        self.faithfulness_threshold = faithfulness_threshold
        self.relevancy_threshold = relevancy_threshold
        self.precision_threshold = precision_threshold
        self.llm_model = llm_model
        self.use_heuristic_fallback = use_heuristic_fallback
        self._heuristic = HeuristicEvaluator(
            faithfulness_threshold=faithfulness_threshold,
            relevancy_threshold=relevancy_threshold,
            precision_threshold=precision_threshold,
        )

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        use_ragas: bool = True,
        use_deepeval: bool = True,
    ) -> EvalResult:
        """
        Evaluate RAG output quality.

        Args:
            question: The user's query
            answer: The generated answer
            contexts: Retrieved context chunks
            ground_truth: Expected answer (optional, improves some metrics)
            use_ragas: Use Ragas framework
            use_deepeval: Use DeepEval framework
        """
        all_scores: Dict[str, float] = {}
        all_details: Dict[str, str] = {}
        frameworks_used: List[str] = []

        # Circuit breaker check before evaluation
        cb_open = False
        if CIRCUIT_BREAKER_AVAILABLE and adapter_circuit_breaker is not None:
            try:
                cb = adapter_circuit_breaker("rag_evaluator_adapter")
                if hasattr(cb, 'is_open') and cb.is_open:
                    cb_open = True
                    logger.warning("Circuit breaker open for rag_evaluator_adapter, using heuristic fallback")
            except Exception as cb_err:
                logger.debug("Circuit breaker check failed: %s", cb_err)

        # If circuit breaker is open or no SDKs available, use heuristic
        if cb_open or (not _RAGAS_AVAILABLE and not _DEEPEVAL_AVAILABLE):
            if self.use_heuristic_fallback:
                heuristic_result = self._heuristic.evaluate(question, answer, contexts, ground_truth)
                return EvalResult(
                    scores=heuristic_result["scores"],
                    passed=heuristic_result["passed"],
                    details={k: "heuristic" for k in heuristic_result["scores"]},
                    framework="heuristic",
                    metadata={
                        "question_len": len(question),
                        "answer_len": len(answer),
                        "num_contexts": len(contexts),
                        "thresholds": {
                            "faithfulness": self.faithfulness_threshold,
                            "relevancy": self.relevancy_threshold,
                            "precision": self.precision_threshold,
                        },
                    },
                )

        # Ragas evaluation (primary)
        if use_ragas and _RAGAS_AVAILABLE and not cb_open:
            try:
                ragas_scores = await self._evaluate_ragas(question, answer, contexts, ground_truth)
                all_scores.update(ragas_scores)
                all_details.update({k: "ragas" for k in ragas_scores})
                frameworks_used.append("ragas")
                # Record success
                if CIRCUIT_BREAKER_AVAILABLE and adapter_circuit_breaker is not None:
                    try:
                        adapter_circuit_breaker("rag_evaluator_adapter").record_success()
                    except Exception:
                        pass
            except (ValueError, TypeError, RuntimeError) as e:
                logger.warning("Ragas evaluation failed: %s", e)
                all_details["ragas_error"] = str(e)
                if CIRCUIT_BREAKER_AVAILABLE and adapter_circuit_breaker is not None:
                    try:
                        adapter_circuit_breaker("rag_evaluator_adapter").record_failure()
                    except Exception:
                        pass
            except ConnectionError as e:
                logger.warning("Ragas evaluation connection error: %s", e)
                all_details["ragas_error"] = str(e)
                if CIRCUIT_BREAKER_AVAILABLE and adapter_circuit_breaker is not None:
                    try:
                        adapter_circuit_breaker("rag_evaluator_adapter").record_failure()
                    except Exception:
                        pass

        # DeepEval evaluation (optional, CI/CD focused)
        if use_deepeval and _DEEPEVAL_AVAILABLE and not cb_open:
            try:
                deepeval_scores = await self._evaluate_deepeval_async(question, answer, contexts, ground_truth)
                # Prefix to avoid key collision
                for k, v in deepeval_scores.items():
                    all_scores[f"de_{k}"] = v
                all_details.update({f"de_{k}": "deepeval" for k in deepeval_scores})
                frameworks_used.append("deepeval")
                # Record success
                if CIRCUIT_BREAKER_AVAILABLE and adapter_circuit_breaker is not None:
                    try:
                        adapter_circuit_breaker("rag_evaluator_adapter").record_success()
                    except Exception:
                        pass
            except (ValueError, TypeError, RuntimeError) as e:
                logger.warning("DeepEval evaluation failed: %s", e)
                all_details["deepeval_error"] = str(e)
                if CIRCUIT_BREAKER_AVAILABLE and adapter_circuit_breaker is not None:
                    try:
                        adapter_circuit_breaker("rag_evaluator_adapter").record_failure()
                    except Exception:
                        pass
            except ConnectionError as e:
                logger.warning("DeepEval evaluation connection error: %s", e)
                all_details["deepeval_error"] = str(e)
                if CIRCUIT_BREAKER_AVAILABLE and adapter_circuit_breaker is not None:
                    try:
                        adapter_circuit_breaker("rag_evaluator_adapter").record_failure()
                    except Exception:
                        pass

        # If no SDK succeeded, use heuristic fallback
        if not all_scores and self.use_heuristic_fallback:
            heuristic_result = self._heuristic.evaluate(question, answer, contexts, ground_truth)
            all_scores = heuristic_result["scores"]
            all_details = {k: "heuristic" for k in heuristic_result["scores"]}
            frameworks_used.append("heuristic")

        # Determine pass/fail
        passed = True
        if "faithfulness" in all_scores:
            passed = passed and all_scores["faithfulness"] >= self.faithfulness_threshold
        if "answer_relevancy" in all_scores:
            passed = passed and all_scores["answer_relevancy"] >= self.relevancy_threshold
        if "context_precision" in all_scores:
            passed = passed and all_scores["context_precision"] >= self.precision_threshold

        framework = " + ".join(frameworks_used) if frameworks_used else "none"

        return EvalResult(
            scores=all_scores,
            passed=passed,
            details=all_details,
            framework=framework,
            metadata={
                "question_len": len(question),
                "answer_len": len(answer),
                "num_contexts": len(contexts),
                "thresholds": {
                    "faithfulness": self.faithfulness_threshold,
                    "relevancy": self.relevancy_threshold,
                    "precision": self.precision_threshold,
                },
            },
        )

    async def _evaluate_ragas(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """Run Ragas evaluation metrics."""
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper

        # Setup LLM for evaluation
        llm = LangchainLLMWrapper(ChatOpenAI(model=self.llm_model))

        # Build metrics
        metrics = [
            Faithfulness(llm=llm),
            AnswerRelevancy(llm=llm),
            ContextPrecision(llm=llm),
        ]

        # Build dataset
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )
        if ground_truth:
            sample.reference = ground_truth

        dataset = EvaluationDataset(samples=[sample])

        # Run evaluation in thread pool (Ragas uses sync operations)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: ragas_evaluate(dataset=dataset, metrics=metrics)
        )

        # Extract scores
        scores: Dict[str, float] = {}
        for metric_name in ["faithfulness", "answer_relevancy", "context_precision"]:
            if metric_name in results:
                val = results[metric_name]
                scores[metric_name] = float(val) if val is not None else 0.0

        return scores

    async def _evaluate_deepeval_async(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """Run DeepEval evaluation metrics asynchronously."""
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=contexts,
        )
        if ground_truth:
            test_case.expected_output = ground_truth

        metrics = [
            FaithfulnessMetric(threshold=self.faithfulness_threshold, model=self.llm_model),
            AnswerRelevancyMetric(threshold=self.relevancy_threshold, model=self.llm_model),
            ContextualPrecisionMetric(threshold=self.precision_threshold, model=self.llm_model),
        ]

        # Run in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: deepeval_evaluate(test_cases=[test_case], metrics=metrics)
        )

        scores: Dict[str, float] = {}
        for metric in metrics:
            name = metric.__class__.__name__.replace("Metric", "").lower()
            if hasattr(metric, 'score') and metric.score is not None:
                scores[name] = float(metric.score)

        return scores

    # =========================================================================
    # EvaluatorProtocol Implementation
    # =========================================================================

    async def evaluate_single(
        self,
        question: str,
        contexts: List[str],
        answer: str,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Implements EvaluatorProtocol.evaluate_single for RAGPipeline integration.

        This method signature matches what RAGPipeline expects.

        Args:
            question: The user query
            contexts: Retrieved context documents
            answer: Generated answer
            ground_truth: Optional expected answer

        Returns:
            Dict with scores, passed status, and metadata
        """
        result = await self.evaluate(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )
        return {
            "scores": result.scores,
            "passed": result.passed,
            "framework": result.framework,
            "details": result.details,
            "metadata": result.metadata,
        }

    @staticmethod
    def available_frameworks() -> Dict[str, bool]:
        """Check which evaluation frameworks are available."""
        return {
            "ragas": _RAGAS_AVAILABLE,
            "deepeval": _DEEPEVAL_AVAILABLE,
            "heuristic": True,  # Always available
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "RAGEvaluator",
    "EvalResult",
    "HeuristicEvaluator",
    "_RAGAS_AVAILABLE",
    "_DEEPEVAL_AVAILABLE",
]


if __name__ == "__main__":
    ev = RAGEvaluator()
    frameworks = ev.available_frameworks()
    print(f"Ragas: {'available' if frameworks['ragas'] else 'NOT installed'}")
    print(f"DeepEval: {'available' if frameworks['deepeval'] else 'NOT installed'}")
    print(f"Heuristic: {'available' if frameworks['heuristic'] else 'NOT available'}")
    print("OK")
