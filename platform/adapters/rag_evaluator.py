"""
rag_evaluator.py - RAG Quality Evaluation Adapter V1
Wraps Ragas (primary) + DeepEval (optional) for RAG pipeline quality measurement.

Uses:
  - Ragas 0.4.3: faithfulness, answer_relevancy, context_precision
  - DeepEval (optional): CI/CD pytest integration, hallucination detection
  - Opik integration: traces evaluation runs (via existing opik_integration.py)

Based on: Production patterns from Haystack, LlamaIndex, Awesome-RAG-Production.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Check availability at import time
_RAGAS_AVAILABLE = False
_DEEPEVAL_AVAILABLE = False

try:
    from ragas import evaluate as ragas_evaluate
    try:
        # Ragas >= 0.4.x uses ragas.metrics.collections
        from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision
    except ImportError:
        # Fallback for older versions
        from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    _RAGAS_AVAILABLE = True
except ImportError:
    logger.info("Ragas not installed — Ragas evaluation disabled")

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
    logger.info("DeepEval not installed — DeepEval evaluation disabled")


@dataclass
class EvalResult:
    """Result of RAG evaluation."""
    scores: dict[str, float]        # metric_name -> score (0.0-1.0)
    passed: bool                    # All metrics above thresholds?
    details: dict[str, str]         # metric_name -> reason/explanation
    framework: str                  # 'ragas', 'deepeval', or 'both'
    metadata: dict = field(default_factory=dict)


class RAGEvaluator:
    """
    Unified RAG evaluation using Ragas (primary) + DeepEval (optional).

    Usage:
        evaluator = RAGEvaluator()
        result = await evaluator.evaluate(
            question="What is X?",
            answer="X is ...",
            contexts=["Context chunk 1", "Context chunk 2"],
        )
        print(result.scores)  # {"faithfulness": 0.85, "answer_relevancy": 0.92, ...}
        print(result.passed)  # True/False
    """

    def __init__(
        self,
        faithfulness_threshold: float = 0.7,
        relevancy_threshold: float = 0.6,
        precision_threshold: float = 0.5,
        llm_model: str = "gpt-4o-mini",
    ):
        self.faithfulness_threshold = faithfulness_threshold
        self.relevancy_threshold = relevancy_threshold
        self.precision_threshold = precision_threshold
        self.llm_model = llm_model

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
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
        all_scores = {}
        all_details = {}
        frameworks_used = []

        # Ragas evaluation (primary)
        if use_ragas and _RAGAS_AVAILABLE:
            try:
                ragas_scores = await self._evaluate_ragas(question, answer, contexts, ground_truth)
                all_scores.update(ragas_scores)
                all_details.update({k: "ragas" for k in ragas_scores})
                frameworks_used.append("ragas")
            except Exception as e:
                logger.warning(f"Ragas evaluation failed: {e}")
                all_details["ragas_error"] = str(e)

        # DeepEval evaluation (optional, CI/CD focused)
        if use_deepeval and _DEEPEVAL_AVAILABLE:
            try:
                deepeval_scores = self._evaluate_deepeval(question, answer, contexts, ground_truth)
                # Prefix to avoid key collision
                for k, v in deepeval_scores.items():
                    all_scores[f"de_{k}"] = v
                all_details.update({f"de_{k}": "deepeval" for k in deepeval_scores})
                frameworks_used.append("deepeval")
            except Exception as e:
                logger.warning(f"DeepEval evaluation failed: {e}")
                all_details["deepeval_error"] = str(e)

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
        contexts: list[str],
        ground_truth: Optional[str] = None,
    ) -> dict[str, float]:
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

        # Run evaluation
        results = ragas_evaluate(dataset=dataset, metrics=metrics)

        # Extract scores
        scores = {}
        for metric_name in ["faithfulness", "answer_relevancy", "context_precision"]:
            if metric_name in results:
                val = results[metric_name]
                scores[metric_name] = float(val) if val is not None else 0.0

        return scores

    def _evaluate_deepeval(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: Optional[str] = None,
    ) -> dict[str, float]:
        """Run DeepEval evaluation metrics."""
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

        results = deepeval_evaluate(test_cases=[test_case], metrics=metrics)

        scores = {}
        for metric in metrics:
            name = metric.__class__.__name__.replace("Metric", "").lower()
            if hasattr(metric, 'score') and metric.score is not None:
                scores[name] = float(metric.score)

        return scores

    @staticmethod
    def available_frameworks() -> dict[str, bool]:
        """Check which evaluation frameworks are available."""
        return {
            "ragas": _RAGAS_AVAILABLE,
            "deepeval": _DEEPEVAL_AVAILABLE,
        }


if __name__ == "__main__":
    ev = RAGEvaluator()
    frameworks = ev.available_frameworks()
    print(f"Ragas: {'available' if frameworks['ragas'] else 'NOT installed'}")
    print(f"DeepEval: {'available' if frameworks['deepeval'] else 'NOT installed'}")
    print("OK")
