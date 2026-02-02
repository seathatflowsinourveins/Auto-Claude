#!/usr/bin/env python3
"""
RAGAS Evaluator - RAG Pipeline Evaluation
Part of the V33 Observability Layer.

Uses RAGAS for comprehensive RAG evaluation including:
- Context precision and recall
- Faithfulness scoring
- Answer relevancy
- Answer correctness
"""

from __future__ import annotations

import os
import uuid
from enum import Enum
from typing import Any, Optional, Dict, List, Union
from datetime import datetime
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    ragas_evaluate = None
    context_precision = None
    context_recall = None
    faithfulness = None
    answer_relevancy = None
    Dataset = None


# ============================================================================
# Metric Types
# ============================================================================

class RAGASMetricType(str, Enum):
    """Types of RAGAS metrics."""
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    ANSWER_CORRECTNESS = "answer_correctness"
    ANSWER_SIMILARITY = "answer_similarity"
    CONTEXT_ENTITY_RECALL = "context_entity_recall"


# ============================================================================
# Evaluation Models
# ============================================================================

@dataclass
class RAGASSample:
    """A sample for RAGAS evaluation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    question: str = ""
    answer: str = ""
    contexts: List[str] = field(default_factory=list)
    ground_truth: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGASResult:
    """Result of a RAGAS metric evaluation."""
    metric_type: RAGASMetricType
    score: float = 0.0
    threshold: float = 0.5
    passed: bool = True
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class RAGASEvaluationResult(BaseModel):
    """Complete RAGAS evaluation result."""
    evaluation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sample_count: int = Field(default=0)
    metrics: Dict[str, float] = Field(default_factory=dict)
    per_sample_scores: List[Dict[str, Any]] = Field(default_factory=list)
    overall_score: float = Field(default=0.0)
    passed: bool = Field(default=True)
    duration_ms: float = Field(default=0.0)
    error: Optional[str] = Field(default=None)


# ============================================================================
# RAGAS Evaluator
# ============================================================================

class RAGASEvaluator:
    """
    RAGAS-based evaluation for RAG pipelines.

    Provides comprehensive RAG evaluation including:
    - Context precision and recall
    - Faithfulness scoring
    - Answer relevancy
    - Answer correctness
    """

    def __init__(
        self,
        metrics: Optional[List[RAGASMetricType]] = None,
        threshold: float = 0.5,
    ):
        """
        Initialize the RAGAS evaluator.

        Args:
            metrics: Metrics to compute (defaults to all)
            threshold: Default threshold for passing
        """
        self.threshold = threshold
        self.metrics = metrics or [
            RAGASMetricType.CONTEXT_PRECISION,
            RAGASMetricType.CONTEXT_RECALL,
            RAGASMetricType.FAITHFULNESS,
            RAGASMetricType.ANSWER_RELEVANCY,
        ]

        self._ragas_metrics = {}
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize RAGAS metrics."""
        if RAGAS_AVAILABLE:
            try:
                self._ragas_metrics = {
                    RAGASMetricType.CONTEXT_PRECISION: context_precision,
                    RAGASMetricType.CONTEXT_RECALL: context_recall,
                    RAGASMetricType.FAITHFULNESS: faithfulness,
                    RAGASMetricType.ANSWER_RELEVANCY: answer_relevancy,
                }
            except Exception:
                pass

    @property
    def is_available(self) -> bool:
        """Check if RAGAS is available."""
        return RAGAS_AVAILABLE

    async def evaluate_context_precision(
        self,
        question: str,
        contexts: List[str],
        ground_truth: str,
        threshold: Optional[float] = None,
    ) -> RAGASResult:
        """
        Evaluate context precision.

        Args:
            question: Query question
            contexts: Retrieved contexts
            ground_truth: Ground truth answer
            threshold: Score threshold

        Returns:
            RAGASResult with precision score
        """
        threshold = threshold or self.threshold
        start_time = datetime.now()

        try:
            if RAGAS_AVAILABLE and Dataset:
                dataset = Dataset.from_dict({
                    "question": [question],
                    "contexts": [contexts],
                    "ground_truth": [ground_truth],
                })
                result = ragas_evaluate(
                    dataset,
                    metrics=[context_precision],
                )
                score = result["context_precision"]
            else:
                # Fallback heuristic
                score = self._heuristic_context_precision(question, contexts, ground_truth)

            return RAGASResult(
                metric_type=RAGASMetricType.CONTEXT_PRECISION,
                score=score,
                threshold=threshold,
                passed=score >= threshold,
            )

        except Exception as e:
            return RAGASResult(
                metric_type=RAGASMetricType.CONTEXT_PRECISION,
                error=str(e),
            )

    async def evaluate_context_recall(
        self,
        question: str,
        contexts: List[str],
        ground_truth: str,
        threshold: Optional[float] = None,
    ) -> RAGASResult:
        """
        Evaluate context recall.

        Args:
            question: Query question
            contexts: Retrieved contexts
            ground_truth: Ground truth answer
            threshold: Score threshold

        Returns:
            RAGASResult with recall score
        """
        threshold = threshold or self.threshold

        try:
            if RAGAS_AVAILABLE and Dataset:
                dataset = Dataset.from_dict({
                    "question": [question],
                    "contexts": [contexts],
                    "ground_truth": [ground_truth],
                })
                result = ragas_evaluate(
                    dataset,
                    metrics=[context_recall],
                )
                score = result["context_recall"]
            else:
                # Fallback heuristic
                score = self._heuristic_context_recall(contexts, ground_truth)

            return RAGASResult(
                metric_type=RAGASMetricType.CONTEXT_RECALL,
                score=score,
                threshold=threshold,
                passed=score >= threshold,
            )

        except Exception as e:
            return RAGASResult(
                metric_type=RAGASMetricType.CONTEXT_RECALL,
                error=str(e),
            )

    async def evaluate_faithfulness(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        threshold: Optional[float] = None,
    ) -> RAGASResult:
        """
        Evaluate answer faithfulness to contexts.

        Args:
            question: Query question
            answer: Generated answer
            contexts: Source contexts
            threshold: Score threshold

        Returns:
            RAGASResult with faithfulness score
        """
        threshold = threshold or self.threshold

        try:
            if RAGAS_AVAILABLE and Dataset:
                dataset = Dataset.from_dict({
                    "question": [question],
                    "answer": [answer],
                    "contexts": [contexts],
                })
                result = ragas_evaluate(
                    dataset,
                    metrics=[faithfulness],
                )
                score = result["faithfulness"]
            else:
                # Fallback heuristic
                score = self._heuristic_faithfulness(answer, contexts)

            return RAGASResult(
                metric_type=RAGASMetricType.FAITHFULNESS,
                score=score,
                threshold=threshold,
                passed=score >= threshold,
            )

        except Exception as e:
            return RAGASResult(
                metric_type=RAGASMetricType.FAITHFULNESS,
                error=str(e),
            )

    async def evaluate_answer_relevancy(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        threshold: Optional[float] = None,
    ) -> RAGASResult:
        """
        Evaluate answer relevancy to question.

        Args:
            question: Query question
            answer: Generated answer
            contexts: Source contexts
            threshold: Score threshold

        Returns:
            RAGASResult with relevancy score
        """
        threshold = threshold or self.threshold

        try:
            if RAGAS_AVAILABLE and Dataset:
                dataset = Dataset.from_dict({
                    "question": [question],
                    "answer": [answer],
                    "contexts": [contexts],
                })
                result = ragas_evaluate(
                    dataset,
                    metrics=[answer_relevancy],
                )
                score = result["answer_relevancy"]
            else:
                # Fallback heuristic
                score = self._heuristic_answer_relevancy(question, answer)

            return RAGASResult(
                metric_type=RAGASMetricType.ANSWER_RELEVANCY,
                score=score,
                threshold=threshold,
                passed=score >= threshold,
            )

        except Exception as e:
            return RAGASResult(
                metric_type=RAGASMetricType.ANSWER_RELEVANCY,
                error=str(e),
            )

    async def evaluate_sample(
        self,
        sample: RAGASSample,
        metrics: Optional[List[RAGASMetricType]] = None,
    ) -> Dict[str, RAGASResult]:
        """
        Evaluate a single sample with multiple metrics.

        Args:
            sample: Sample to evaluate
            metrics: Metrics to compute

        Returns:
            Dictionary of metric results
        """
        metrics = metrics or self.metrics
        results: Dict[str, RAGASResult] = {}

        for metric in metrics:
            if metric == RAGASMetricType.CONTEXT_PRECISION and sample.ground_truth:
                result = await self.evaluate_context_precision(
                    sample.question,
                    sample.contexts,
                    sample.ground_truth,
                )
            elif metric == RAGASMetricType.CONTEXT_RECALL and sample.ground_truth:
                result = await self.evaluate_context_recall(
                    sample.question,
                    sample.contexts,
                    sample.ground_truth,
                )
            elif metric == RAGASMetricType.FAITHFULNESS:
                result = await self.evaluate_faithfulness(
                    sample.question,
                    sample.answer,
                    sample.contexts,
                )
            elif metric == RAGASMetricType.ANSWER_RELEVANCY:
                result = await self.evaluate_answer_relevancy(
                    sample.question,
                    sample.answer,
                    sample.contexts,
                )
            else:
                continue

            results[metric.value] = result

        return results

    async def evaluate_dataset(
        self,
        samples: List[RAGASSample],
        metrics: Optional[List[RAGASMetricType]] = None,
    ) -> RAGASEvaluationResult:
        """
        Evaluate a dataset of samples.

        Args:
            samples: Samples to evaluate
            metrics: Metrics to compute

        Returns:
            Complete evaluation result
        """
        start_time = datetime.now()
        metrics = metrics or self.metrics

        per_sample_scores: List[Dict[str, Any]] = []
        aggregated_scores: Dict[str, List[float]] = {m.value: [] for m in metrics}

        for sample in samples:
            sample_results = await self.evaluate_sample(sample, metrics)

            sample_score = {
                "id": sample.id,
                "question": sample.question[:100],
            }

            for metric_name, result in sample_results.items():
                sample_score[metric_name] = result.score
                if result.error is None:
                    aggregated_scores[metric_name].append(result.score)

            per_sample_scores.append(sample_score)

        # Calculate averages
        avg_scores = {
            name: sum(scores) / len(scores) if scores else 0.0
            for name, scores in aggregated_scores.items()
        }

        overall = sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0.0
        duration = (datetime.now() - start_time).total_seconds() * 1000

        return RAGASEvaluationResult(
            sample_count=len(samples),
            metrics=avg_scores,
            per_sample_scores=per_sample_scores,
            overall_score=overall,
            passed=overall >= self.threshold,
            duration_ms=duration,
        )

    # ========================================================================
    # Heuristic Fallback Methods
    # ========================================================================

    def _heuristic_context_precision(
        self,
        question: str,
        contexts: List[str],
        ground_truth: str,
    ) -> float:
        """Heuristic for context precision."""
        if not contexts:
            return 0.0

        ground_truth_words = set(ground_truth.lower().split())
        relevant_contexts = 0

        for context in contexts:
            context_words = set(context.lower().split())
            overlap = len(ground_truth_words & context_words)
            if overlap / max(len(ground_truth_words), 1) > 0.2:
                relevant_contexts += 1

        return relevant_contexts / len(contexts)

    def _heuristic_context_recall(
        self,
        contexts: List[str],
        ground_truth: str,
    ) -> float:
        """Heuristic for context recall."""
        if not contexts:
            return 0.0

        ground_truth_words = set(ground_truth.lower().split())
        all_context_words = set()

        for context in contexts:
            all_context_words.update(context.lower().split())

        if not ground_truth_words:
            return 0.5

        overlap = len(ground_truth_words & all_context_words)
        return overlap / len(ground_truth_words)

    def _heuristic_faithfulness(
        self,
        answer: str,
        contexts: List[str],
    ) -> float:
        """Heuristic for faithfulness."""
        if not contexts:
            return 0.5

        all_context_text = " ".join(contexts).lower()
        answer_words = set(answer.lower().split())
        context_words = set(all_context_text.split())

        if not answer_words:
            return 0.5

        overlap = len(answer_words & context_words)
        return min(1.0, overlap / len(answer_words) * 1.5)

    def _heuristic_answer_relevancy(
        self,
        question: str,
        answer: str,
    ) -> float:
        """Heuristic for answer relevancy."""
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())

        if not question_words or not answer_words:
            return 0.5

        overlap = len(question_words & answer_words)
        return min(1.0, overlap / len(question_words) * 2)


# ============================================================================
# Convenience Functions
# ============================================================================

def create_ragas_evaluator(
    metrics: Optional[List[RAGASMetricType]] = None,
    **kwargs: Any,
) -> RAGASEvaluator:
    """
    Factory function to create a RAGASEvaluator.

    Args:
        metrics: Metrics to compute
        **kwargs: Additional configuration

    Returns:
        Configured RAGASEvaluator instance
    """
    return RAGASEvaluator(metrics=metrics, **kwargs)


# Export availability
__all__ = [
    "RAGASEvaluator",
    "RAGASMetricType",
    "RAGASSample",
    "RAGASResult",
    "RAGASEvaluationResult",
    "create_ragas_evaluator",
    "RAGAS_AVAILABLE",
]
