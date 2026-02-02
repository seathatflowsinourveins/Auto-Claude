#!/usr/bin/env python3
"""
Opik Evaluator - LLM Evaluation and A/B Testing
Part of the V33 Observability Layer.

Uses Opik for comprehensive LLM evaluation including:
- Hallucination detection
- Answer relevance scoring
- Coherence and fluency metrics
- A/B testing experiments
"""

from __future__ import annotations

import os
import uuid
from enum import Enum
from typing import Any, Optional, Dict, List, Callable
from datetime import datetime
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

try:
    import opik
    from opik import track, opik_context
    from opik.evaluation.metrics import (
        Hallucination,
        AnswerRelevance,
    )
    OPIK_AVAILABLE = True
except ImportError:
    OPIK_AVAILABLE = False
    opik = None
    track = None
    opik_context = None


# ============================================================================
# Metric Types
# ============================================================================

class MetricType(str, Enum):
    """Types of evaluation metrics."""
    HALLUCINATION = "hallucination"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    FLUENCY = "fluency"
    FAITHFULNESS = "faithfulness"
    TOXICITY = "toxicity"
    BIAS = "bias"
    CUSTOM = "custom"


class EvaluationStatus(str, Enum):
    """Status of an evaluation run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================================
# Evaluation Models
# ============================================================================

@dataclass
class EvaluationResult:
    """Result of a single evaluation."""
    metric_type: MetricType
    score: float = 0.0
    passed: bool = True
    threshold: float = 0.7
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment."""
    name: str
    description: str = ""
    variants: List[str] = field(default_factory=lambda: ["control", "treatment"])
    metrics: List[MetricType] = field(default_factory=list)
    sample_size: int = 100
    confidence_level: float = 0.95


class ExperimentResult(BaseModel):
    """Result of an A/B experiment."""
    experiment_id: str = Field(description="Unique experiment identifier")
    name: str = Field(description="Experiment name")
    status: EvaluationStatus = Field(default=EvaluationStatus.PENDING)
    variants: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    winner: Optional[str] = Field(default=None)
    statistical_significance: float = Field(default=0.0)
    sample_count: int = Field(default=0)
    error: Optional[str] = Field(default=None)


class EvaluationSample(BaseModel):
    """A sample for evaluation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input: str = Field(description="Input prompt")
    output: str = Field(description="Model output")
    context: Optional[str] = Field(default=None, description="Context for RAG")
    reference: Optional[str] = Field(default=None, description="Reference answer")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Opik Evaluator
# ============================================================================

class OpikEvaluator:
    """
    Opik-based evaluation for LLM applications.

    Provides comprehensive evaluation including:
    - Hallucination detection
    - Answer relevance scoring
    - Coherence and fluency metrics
    - A/B testing experiments
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_name: str = "v33-evaluation",
        workspace: Optional[str] = None,
    ):
        """
        Initialize the Opik evaluator.

        Args:
            api_key: Opik API key (or OPIK_API_KEY env var)
            project_name: Project name for organizing evaluations
            workspace: Optional workspace
        """
        self.api_key = api_key or os.getenv("OPIK_API_KEY")
        self.project_name = project_name
        self.workspace = workspace

        self._client = None
        self._metrics: Dict[MetricType, Any] = {}
        self._experiments: Dict[str, ExperimentConfig] = {}

        if OPIK_AVAILABLE and self.api_key:
            try:
                opik.configure(api_key=self.api_key)
                self._init_metrics()
            except Exception:
                pass

    def _init_metrics(self) -> None:
        """Initialize built-in metrics."""
        if OPIK_AVAILABLE:
            try:
                self._metrics[MetricType.HALLUCINATION] = Hallucination()
                self._metrics[MetricType.RELEVANCE] = AnswerRelevance()
            except Exception:
                pass

    @property
    def is_available(self) -> bool:
        """Check if Opik is available and configured."""
        return OPIK_AVAILABLE and self.api_key is not None

    async def evaluate_hallucination(
        self,
        output: str,
        context: str,
        threshold: float = 0.5,
    ) -> EvaluationResult:
        """
        Evaluate output for hallucination.

        Args:
            output: Model output to evaluate
            context: Context that should ground the output
            threshold: Score threshold for passing

        Returns:
            EvaluationResult with hallucination score
        """
        start_time = datetime.now()

        try:
            if OPIK_AVAILABLE and MetricType.HALLUCINATION in self._metrics:
                metric = self._metrics[MetricType.HALLUCINATION]
                score = metric.score(output=output, context=context)
                score_value = score.value if hasattr(score, 'value') else float(score)
            else:
                # Fallback heuristic
                score_value = self._heuristic_hallucination(output, context)

            latency = (datetime.now() - start_time).total_seconds() * 1000

            return EvaluationResult(
                metric_type=MetricType.HALLUCINATION,
                score=score_value,
                passed=score_value <= threshold,  # Lower is better for hallucination
                threshold=threshold,
                latency_ms=latency,
            )

        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds() * 1000
            return EvaluationResult(
                metric_type=MetricType.HALLUCINATION,
                score=0.0,
                passed=False,
                threshold=threshold,
                error=str(e),
                latency_ms=latency,
            )

    async def evaluate_relevance(
        self,
        input: str,
        output: str,
        threshold: float = 0.7,
    ) -> EvaluationResult:
        """
        Evaluate output relevance to input.

        Args:
            input: Input prompt
            output: Model output to evaluate
            threshold: Score threshold for passing

        Returns:
            EvaluationResult with relevance score
        """
        start_time = datetime.now()

        try:
            if OPIK_AVAILABLE and MetricType.RELEVANCE in self._metrics:
                metric = self._metrics[MetricType.RELEVANCE]
                score = metric.score(input=input, output=output)
                score_value = score.value if hasattr(score, 'value') else float(score)
            else:
                # Fallback heuristic
                score_value = self._heuristic_relevance(input, output)

            latency = (datetime.now() - start_time).total_seconds() * 1000

            return EvaluationResult(
                metric_type=MetricType.RELEVANCE,
                score=score_value,
                passed=score_value >= threshold,
                threshold=threshold,
                latency_ms=latency,
            )

        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds() * 1000
            return EvaluationResult(
                metric_type=MetricType.RELEVANCE,
                score=0.0,
                passed=False,
                threshold=threshold,
                error=str(e),
                latency_ms=latency,
            )

    async def evaluate_coherence(
        self,
        output: str,
        threshold: float = 0.7,
    ) -> EvaluationResult:
        """
        Evaluate output coherence.

        Args:
            output: Model output to evaluate
            threshold: Score threshold for passing

        Returns:
            EvaluationResult with coherence score
        """
        start_time = datetime.now()

        try:
            # Use heuristic coherence scoring
            score_value = self._heuristic_coherence(output)
            latency = (datetime.now() - start_time).total_seconds() * 1000

            return EvaluationResult(
                metric_type=MetricType.COHERENCE,
                score=score_value,
                passed=score_value >= threshold,
                threshold=threshold,
                latency_ms=latency,
            )

        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds() * 1000
            return EvaluationResult(
                metric_type=MetricType.COHERENCE,
                score=0.0,
                passed=False,
                threshold=threshold,
                error=str(e),
                latency_ms=latency,
            )

    async def evaluate_fluency(
        self,
        output: str,
        threshold: float = 0.7,
    ) -> EvaluationResult:
        """
        Evaluate output fluency.

        Args:
            output: Model output to evaluate
            threshold: Score threshold for passing

        Returns:
            EvaluationResult with fluency score
        """
        start_time = datetime.now()

        try:
            score_value = self._heuristic_fluency(output)
            latency = (datetime.now() - start_time).total_seconds() * 1000

            return EvaluationResult(
                metric_type=MetricType.FLUENCY,
                score=score_value,
                passed=score_value >= threshold,
                threshold=threshold,
                latency_ms=latency,
            )

        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds() * 1000
            return EvaluationResult(
                metric_type=MetricType.FLUENCY,
                score=0.0,
                passed=False,
                threshold=threshold,
                error=str(e),
                latency_ms=latency,
            )

    async def run_evaluation(
        self,
        samples: List[EvaluationSample],
        metrics: List[MetricType],
    ) -> Dict[str, List[EvaluationResult]]:
        """
        Run evaluation on multiple samples.

        Args:
            samples: List of samples to evaluate
            metrics: Metrics to compute

        Returns:
            Dictionary mapping sample IDs to results
        """
        results: Dict[str, List[EvaluationResult]] = {}

        for sample in samples:
            sample_results = []

            for metric in metrics:
                if metric == MetricType.HALLUCINATION and sample.context:
                    result = await self.evaluate_hallucination(
                        sample.output, sample.context
                    )
                elif metric == MetricType.RELEVANCE:
                    result = await self.evaluate_relevance(
                        sample.input, sample.output
                    )
                elif metric == MetricType.COHERENCE:
                    result = await self.evaluate_coherence(sample.output)
                elif metric == MetricType.FLUENCY:
                    result = await self.evaluate_fluency(sample.output)
                else:
                    continue

                sample_results.append(result)

            results[sample.id] = sample_results

        return results

    def create_experiment(self, config: ExperimentConfig) -> str:
        """
        Create a new A/B experiment.

        Args:
            config: Experiment configuration

        Returns:
            Experiment ID
        """
        experiment_id = str(uuid.uuid4())
        self._experiments[experiment_id] = config
        return experiment_id

    async def run_experiment(
        self,
        experiment_id: str,
        variant_samples: Dict[str, List[EvaluationSample]],
    ) -> ExperimentResult:
        """
        Run an A/B experiment.

        Args:
            experiment_id: Experiment ID
            variant_samples: Samples per variant

        Returns:
            ExperimentResult with comparison
        """
        if experiment_id not in self._experiments:
            return ExperimentResult(
                experiment_id=experiment_id,
                name="unknown",
                status=EvaluationStatus.FAILED,
                error="Experiment not found",
            )

        config = self._experiments[experiment_id]
        variant_scores: Dict[str, Dict[str, float]] = {}
        total_samples = 0

        for variant, samples in variant_samples.items():
            results = await self.run_evaluation(samples, config.metrics)
            total_samples += len(samples)

            # Aggregate scores per metric
            metric_scores: Dict[str, List[float]] = {}
            for sample_results in results.values():
                for result in sample_results:
                    metric_name = result.metric_type.value
                    if metric_name not in metric_scores:
                        metric_scores[metric_name] = []
                    metric_scores[metric_name].append(result.score)

            # Average scores
            variant_scores[variant] = {
                metric: sum(scores) / len(scores) if scores else 0.0
                for metric, scores in metric_scores.items()
            }

        # Determine winner (simple comparison for now)
        winner = None
        if len(variant_scores) >= 2:
            variants = list(variant_scores.keys())
            avg_scores = {
                v: sum(variant_scores[v].values()) / len(variant_scores[v])
                if variant_scores[v] else 0.0
                for v in variants
            }
            winner = max(avg_scores, key=lambda k: avg_scores[k])

        return ExperimentResult(
            experiment_id=experiment_id,
            name=config.name,
            status=EvaluationStatus.COMPLETED,
            variants=variant_scores,
            winner=winner,
            sample_count=total_samples,
        )

    # ========================================================================
    # Heuristic Fallback Methods
    # ========================================================================

    def _heuristic_hallucination(self, output: str, context: str) -> float:
        """Simple heuristic for hallucination detection."""
        # Check overlap between output and context
        output_words = set(output.lower().split())
        context_words = set(context.lower().split())

        if not output_words:
            return 0.5

        overlap = len(output_words & context_words)
        overlap_ratio = overlap / len(output_words)

        # Higher overlap = lower hallucination
        return 1.0 - min(1.0, overlap_ratio * 1.5)

    def _heuristic_relevance(self, input: str, output: str) -> float:
        """Simple heuristic for relevance scoring."""
        input_words = set(input.lower().split())
        output_words = set(output.lower().split())

        if not input_words or not output_words:
            return 0.5

        overlap = len(input_words & output_words)
        return min(1.0, overlap / len(input_words) * 2)

    def _heuristic_coherence(self, output: str) -> float:
        """Simple heuristic for coherence scoring."""
        sentences = output.split('.')
        if len(sentences) < 2:
            return 0.7

        # Check for repeated content (indicates incoherence)
        unique_sentences = set(s.strip().lower() for s in sentences if s.strip())
        repetition_ratio = len(unique_sentences) / len(sentences)

        return min(1.0, repetition_ratio * 1.2)

    def _heuristic_fluency(self, output: str) -> float:
        """Simple heuristic for fluency scoring."""
        words = output.split()
        if not words:
            return 0.0

        # Check average word length (proxy for vocabulary)
        avg_word_length = sum(len(w) for w in words) / len(words)

        # Check sentence structure
        sentences = output.split('.')
        has_structure = len(sentences) > 1 or output.endswith('.')

        score = 0.5
        if 4 <= avg_word_length <= 8:
            score += 0.25
        if has_structure:
            score += 0.25

        return min(1.0, score)


# ============================================================================
# Decorator for automatic tracking
# ============================================================================

def evaluated(
    metrics: Optional[List[MetricType]] = None,
    project: str = "v33-evaluation",
):
    """
    Decorator for automatic evaluation tracking.

    Args:
        metrics: Metrics to compute
        project: Project name
    """
    def decorator(func: Callable) -> Callable:
        if OPIK_AVAILABLE and track:
            return track(
                name=func.__name__,
                project_name=project,
            )(func)
        return func
    return decorator


# ============================================================================
# Convenience Functions
# ============================================================================

def create_opik_evaluator(
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> OpikEvaluator:
    """
    Factory function to create an OpikEvaluator.

    Args:
        api_key: Optional API key
        **kwargs: Additional configuration

    Returns:
        Configured OpikEvaluator instance
    """
    return OpikEvaluator(api_key=api_key, **kwargs)


# Export availability
__all__ = [
    "OpikEvaluator",
    "MetricType",
    "EvaluationStatus",
    "EvaluationResult",
    "ExperimentConfig",
    "ExperimentResult",
    "EvaluationSample",
    "evaluated",
    "create_opik_evaluator",
    "OPIK_AVAILABLE",
]
