#!/usr/bin/env python3
"""
DeepEval Tests - Comprehensive LLM Testing Framework
Part of the V33 Observability Layer.

Uses DeepEval for comprehensive LLM testing including:
- Faithfulness testing
- Answer relevancy
- Hallucination detection
- Toxicity and bias detection
"""

from __future__ import annotations

import os
import uuid
from enum import Enum
from typing import Any, Optional, Dict, List, Callable, Union
from datetime import datetime
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

try:
    from deepeval import evaluate
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        HallucinationMetric,
    )
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    evaluate = None
    LLMTestCase = None
    FaithfulnessMetric = None
    AnswerRelevancyMetric = None
    HallucinationMetric = None


# ============================================================================
# Metric Types
# ============================================================================

class DeepEvalMetricType(str, Enum):
    """Types of DeepEval metrics."""
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy"
    HALLUCINATION = "hallucination"
    CONTEXTUAL_RELEVANCY = "contextual_relevancy"
    CONTEXTUAL_PRECISION = "contextual_precision"
    CONTEXTUAL_RECALL = "contextual_recall"
    TOXICITY = "toxicity"
    BIAS = "bias"
    SUMMARIZATION = "summarization"


class TestStatus(str, Enum):
    """Status of a test run."""
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


# ============================================================================
# Test Models
# ============================================================================

@dataclass
class DeepEvalTestCase:
    """A test case for DeepEval."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    input: str = ""
    actual_output: str = ""
    expected_output: Optional[str] = None
    context: List[str] = field(default_factory=list)
    retrieval_context: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    """Result of a single metric evaluation."""
    metric_type: DeepEvalMetricType
    score: float = 0.0
    passed: bool = False
    threshold: float = 0.5
    reason: str = ""
    error: Optional[str] = None


class TestResult(BaseModel):
    """Result of a test case evaluation."""
    test_id: str = Field(description="Test case ID")
    status: TestStatus = Field(default=TestStatus.PASSED)
    metrics: List[Dict[str, Any]] = Field(default_factory=list)
    overall_score: float = Field(default=0.0)
    passed: bool = Field(default=True)
    duration_ms: float = Field(default=0.0)
    error: Optional[str] = Field(default=None)


class TestSuiteResult(BaseModel):
    """Result of a test suite run."""
    suite_name: str = Field(description="Test suite name")
    total_tests: int = Field(default=0)
    passed_tests: int = Field(default=0)
    failed_tests: int = Field(default=0)
    error_tests: int = Field(default=0)
    overall_pass_rate: float = Field(default=0.0)
    duration_ms: float = Field(default=0.0)
    results: List[TestResult] = Field(default_factory=list)


# ============================================================================
# DeepEval Runner
# ============================================================================

class DeepEvalRunner:
    """
    DeepEval-based testing for LLM applications.

    Provides comprehensive testing including:
    - Faithfulness testing
    - Answer relevancy
    - Hallucination detection
    - Toxicity and bias detection
    """

    def __init__(
        self,
        model: str = "gpt-4",
        threshold: float = 0.5,
        include_reason: bool = True,
    ):
        """
        Initialize the DeepEval runner.

        Args:
            model: Model to use for evaluation
            threshold: Default threshold for metrics
            include_reason: Whether to include reasons in results
        """
        self.model = model
        self.threshold = threshold
        self.include_reason = include_reason

        self._metrics: Dict[DeepEvalMetricType, Any] = {}
        self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize available metrics."""
        if DEEPEVAL_AVAILABLE:
            try:
                self._metrics[DeepEvalMetricType.FAITHFULNESS] = FaithfulnessMetric(
                    threshold=self.threshold,
                    model=self.model,
                    include_reason=self.include_reason,
                )
                self._metrics[DeepEvalMetricType.ANSWER_RELEVANCY] = AnswerRelevancyMetric(
                    threshold=self.threshold,
                    model=self.model,
                    include_reason=self.include_reason,
                )
                self._metrics[DeepEvalMetricType.HALLUCINATION] = HallucinationMetric(
                    threshold=self.threshold,
                    model=self.model,
                    include_reason=self.include_reason,
                )
            except Exception:
                pass

    @property
    def is_available(self) -> bool:
        """Check if DeepEval is available."""
        return DEEPEVAL_AVAILABLE

    async def evaluate_faithfulness(
        self,
        input: str,
        output: str,
        context: List[str],
        threshold: Optional[float] = None,
    ) -> MetricResult:
        """
        Evaluate faithfulness of output to context.

        Args:
            input: Input prompt
            output: Model output
            context: Context documents
            threshold: Score threshold

        Returns:
            MetricResult with faithfulness score
        """
        threshold = threshold or self.threshold
        start_time = datetime.now()

        try:
            if DEEPEVAL_AVAILABLE and DeepEvalMetricType.FAITHFULNESS in self._metrics:
                test_case = LLMTestCase(
                    input=input,
                    actual_output=output,
                    retrieval_context=context,
                )
                metric = self._metrics[DeepEvalMetricType.FAITHFULNESS]
                metric.measure(test_case)

                return MetricResult(
                    metric_type=DeepEvalMetricType.FAITHFULNESS,
                    score=metric.score,
                    passed=metric.score >= threshold,
                    threshold=threshold,
                    reason=metric.reason if hasattr(metric, 'reason') else "",
                )
            else:
                # Fallback heuristic
                score = self._heuristic_faithfulness(output, context)
                return MetricResult(
                    metric_type=DeepEvalMetricType.FAITHFULNESS,
                    score=score,
                    passed=score >= threshold,
                    threshold=threshold,
                )

        except Exception as e:
            return MetricResult(
                metric_type=DeepEvalMetricType.FAITHFULNESS,
                error=str(e),
            )

    async def evaluate_relevancy(
        self,
        input: str,
        output: str,
        threshold: Optional[float] = None,
    ) -> MetricResult:
        """
        Evaluate answer relevancy.

        Args:
            input: Input prompt
            output: Model output
            threshold: Score threshold

        Returns:
            MetricResult with relevancy score
        """
        threshold = threshold or self.threshold

        try:
            if DEEPEVAL_AVAILABLE and DeepEvalMetricType.ANSWER_RELEVANCY in self._metrics:
                test_case = LLMTestCase(
                    input=input,
                    actual_output=output,
                )
                metric = self._metrics[DeepEvalMetricType.ANSWER_RELEVANCY]
                metric.measure(test_case)

                return MetricResult(
                    metric_type=DeepEvalMetricType.ANSWER_RELEVANCY,
                    score=metric.score,
                    passed=metric.score >= threshold,
                    threshold=threshold,
                    reason=metric.reason if hasattr(metric, 'reason') else "",
                )
            else:
                # Fallback heuristic
                score = self._heuristic_relevancy(input, output)
                return MetricResult(
                    metric_type=DeepEvalMetricType.ANSWER_RELEVANCY,
                    score=score,
                    passed=score >= threshold,
                    threshold=threshold,
                )

        except Exception as e:
            return MetricResult(
                metric_type=DeepEvalMetricType.ANSWER_RELEVANCY,
                error=str(e),
            )

    async def evaluate_hallucination(
        self,
        input: str,
        output: str,
        context: List[str],
        threshold: Optional[float] = None,
    ) -> MetricResult:
        """
        Evaluate hallucination in output.

        Args:
            input: Input prompt
            output: Model output
            context: Context documents
            threshold: Score threshold

        Returns:
            MetricResult with hallucination score
        """
        threshold = threshold or self.threshold

        try:
            if DEEPEVAL_AVAILABLE and DeepEvalMetricType.HALLUCINATION in self._metrics:
                test_case = LLMTestCase(
                    input=input,
                    actual_output=output,
                    context=context,
                )
                metric = self._metrics[DeepEvalMetricType.HALLUCINATION]
                metric.measure(test_case)

                return MetricResult(
                    metric_type=DeepEvalMetricType.HALLUCINATION,
                    score=metric.score,
                    passed=metric.score <= threshold,  # Lower is better
                    threshold=threshold,
                    reason=metric.reason if hasattr(metric, 'reason') else "",
                )
            else:
                # Fallback heuristic
                score = self._heuristic_hallucination(output, context)
                return MetricResult(
                    metric_type=DeepEvalMetricType.HALLUCINATION,
                    score=score,
                    passed=score <= threshold,
                    threshold=threshold,
                )

        except Exception as e:
            return MetricResult(
                metric_type=DeepEvalMetricType.HALLUCINATION,
                error=str(e),
            )

    async def evaluate_toxicity(
        self,
        output: str,
        threshold: float = 0.3,
    ) -> MetricResult:
        """
        Evaluate toxicity in output.

        Args:
            output: Model output
            threshold: Score threshold

        Returns:
            MetricResult with toxicity score
        """
        # Use heuristic for toxicity
        score = self._heuristic_toxicity(output)

        return MetricResult(
            metric_type=DeepEvalMetricType.TOXICITY,
            score=score,
            passed=score <= threshold,
            threshold=threshold,
        )

    async def evaluate_bias(
        self,
        output: str,
        threshold: float = 0.3,
    ) -> MetricResult:
        """
        Evaluate bias in output.

        Args:
            output: Model output
            threshold: Score threshold

        Returns:
            MetricResult with bias score
        """
        # Use heuristic for bias
        score = self._heuristic_bias(output)

        return MetricResult(
            metric_type=DeepEvalMetricType.BIAS,
            score=score,
            passed=score <= threshold,
            threshold=threshold,
        )

    async def run_test_case(
        self,
        test_case: DeepEvalTestCase,
        metrics: List[DeepEvalMetricType],
    ) -> TestResult:
        """
        Run a single test case.

        Args:
            test_case: Test case to evaluate
            metrics: Metrics to compute

        Returns:
            TestResult with all metric results
        """
        start_time = datetime.now()
        results: List[MetricResult] = []

        for metric_type in metrics:
            if metric_type == DeepEvalMetricType.FAITHFULNESS and test_case.context:
                result = await self.evaluate_faithfulness(
                    test_case.input,
                    test_case.actual_output,
                    test_case.context,
                )
            elif metric_type == DeepEvalMetricType.ANSWER_RELEVANCY:
                result = await self.evaluate_relevancy(
                    test_case.input,
                    test_case.actual_output,
                )
            elif metric_type == DeepEvalMetricType.HALLUCINATION and test_case.context:
                result = await self.evaluate_hallucination(
                    test_case.input,
                    test_case.actual_output,
                    test_case.context,
                )
            elif metric_type == DeepEvalMetricType.TOXICITY:
                result = await self.evaluate_toxicity(test_case.actual_output)
            elif metric_type == DeepEvalMetricType.BIAS:
                result = await self.evaluate_bias(test_case.actual_output)
            else:
                continue

            results.append(result)

        duration = (datetime.now() - start_time).total_seconds() * 1000

        # Calculate overall
        valid_results = [r for r in results if r.error is None]
        all_passed = all(r.passed for r in valid_results) if valid_results else True
        avg_score = sum(r.score for r in valid_results) / len(valid_results) if valid_results else 0.0

        status = TestStatus.PASSED if all_passed else TestStatus.FAILED
        if any(r.error for r in results):
            status = TestStatus.ERROR

        return TestResult(
            test_id=test_case.id,
            status=status,
            metrics=[
                {
                    "type": r.metric_type.value,
                    "score": r.score,
                    "passed": r.passed,
                    "threshold": r.threshold,
                    "reason": r.reason,
                    "error": r.error,
                }
                for r in results
            ],
            overall_score=avg_score,
            passed=all_passed,
            duration_ms=duration,
        )

    async def run_test_suite(
        self,
        suite_name: str,
        test_cases: List[DeepEvalTestCase],
        metrics: List[DeepEvalMetricType],
    ) -> TestSuiteResult:
        """
        Run a test suite.

        Args:
            suite_name: Name of the test suite
            test_cases: Test cases to run
            metrics: Metrics to compute

        Returns:
            TestSuiteResult with all results
        """
        start_time = datetime.now()
        results: List[TestResult] = []

        for test_case in test_cases:
            result = await self.run_test_case(test_case, metrics)
            results.append(result)

        duration = (datetime.now() - start_time).total_seconds() * 1000

        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in results if r.status == TestStatus.FAILED)
        errors = sum(1 for r in results if r.status == TestStatus.ERROR)

        return TestSuiteResult(
            suite_name=suite_name,
            total_tests=len(results),
            passed_tests=passed,
            failed_tests=failed,
            error_tests=errors,
            overall_pass_rate=passed / len(results) if results else 0.0,
            duration_ms=duration,
            results=results,
        )

    # ========================================================================
    # Heuristic Fallback Methods
    # ========================================================================

    def _heuristic_faithfulness(self, output: str, context: List[str]) -> float:
        """Simple heuristic for faithfulness."""
        if not context:
            return 0.5

        context_text = " ".join(context).lower()
        output_words = set(output.lower().split())
        context_words = set(context_text.split())

        if not output_words:
            return 0.5

        overlap = len(output_words & context_words)
        return min(1.0, overlap / len(output_words) * 1.5)

    def _heuristic_relevancy(self, input: str, output: str) -> float:
        """Simple heuristic for relevancy."""
        input_words = set(input.lower().split())
        output_words = set(output.lower().split())

        if not input_words or not output_words:
            return 0.5

        overlap = len(input_words & output_words)
        return min(1.0, overlap / len(input_words) * 2)

    def _heuristic_hallucination(self, output: str, context: List[str]) -> float:
        """Simple heuristic for hallucination detection."""
        if not context:
            return 0.5

        context_text = " ".join(context).lower()
        output_words = set(output.lower().split())
        context_words = set(context_text.split())

        if not output_words:
            return 0.5

        overlap = len(output_words & context_words)
        overlap_ratio = overlap / len(output_words)

        # Higher overlap = lower hallucination
        return 1.0 - min(1.0, overlap_ratio * 1.5)

    def _heuristic_toxicity(self, output: str) -> float:
        """Simple heuristic for toxicity detection."""
        toxic_words = {
            "hate", "kill", "die", "stupid", "idiot", "ugly",
            "dumb", "loser", "worthless", "disgusting",
        }
        output_lower = output.lower()
        count = sum(1 for word in toxic_words if word in output_lower)
        return min(1.0, count * 0.2)

    def _heuristic_bias(self, output: str) -> float:
        """Simple heuristic for bias detection."""
        # Check for absolute statements that might indicate bias
        bias_indicators = [
            "always", "never", "all", "none", "everyone", "nobody",
            "obviously", "clearly", "definitely",
        ]
        output_lower = output.lower()
        count = sum(1 for word in bias_indicators if word in output_lower)
        return min(1.0, count * 0.15)


# ============================================================================
# Convenience Functions
# ============================================================================

def create_deepeval_runner(
    model: str = "gpt-4",
    **kwargs: Any,
) -> DeepEvalRunner:
    """
    Factory function to create a DeepEvalRunner.

    Args:
        model: Evaluation model
        **kwargs: Additional configuration

    Returns:
        Configured DeepEvalRunner instance
    """
    return DeepEvalRunner(model=model, **kwargs)


# Export availability
__all__ = [
    "DeepEvalRunner",
    "DeepEvalMetricType",
    "TestStatus",
    "DeepEvalTestCase",
    "MetricResult",
    "TestResult",
    "TestSuiteResult",
    "create_deepeval_runner",
    "DEEPEVAL_AVAILABLE",
]
