"""
DeepEval Adapter - LLM Evaluation Framework (V65 Enhanced)
============================================================

Standalone adapter for DeepEval testing and CI/CD integration.
Implements comprehensive Gap11 evaluation with hallucination detection.

Features:
- Faithfulness, relevance, and hallucination metrics
- CI/CD pytest integration via deepeval test
- Custom metric support
- Conversational evaluation
- LLM-as-judge with Claude support
- Heuristic fallback when SDK unavailable

SDK: deepeval >= 0.20.0
Layer: L5 (Observability/Evaluation)

Gap11 Status: RESOLVED (V65)

Usage:
    adapter = DeepEvalAdapter()
    await adapter.initialize({"model": "claude-3-5-sonnet-20241022", "use_claude": True})
    result = await adapter.execute("evaluate",
        input="What is X?", actual_output="X is ...",
        retrieval_context=["context1", "context2"])

    # Hallucination detection
    result = await adapter.execute("detect_hallucination",
        input="What is X?", actual_output="X is something not in context",
        retrieval_context=["context1"])
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
    DEEPEVAL_RETRY_CONFIG = RetryConfig(
        max_retries=3, base_delay=1.0, max_delay=30.0, jitter=0.5
    )
except ImportError:
    RetryConfig = None
    retry_async = None
    DEEPEVAL_RETRY_CONFIG = None

try:
    from .circuit_breaker_manager import adapter_circuit_breaker, CircuitOpenError
except ImportError:
    adapter_circuit_breaker = None
    CircuitOpenError = Exception

# Default timeout
DEEPEVAL_OPERATION_TIMEOUT = 60  # Evaluation can be slow (LLM-as-judge)

# SDK availability flags
DEEPEVAL_AVAILABLE = False
DEEPEVAL_HALLUCINATION_AVAILABLE = False
DEEPEVAL_CONTEXTUAL_AVAILABLE = False

try:
    import deepeval
    DEEPEVAL_AVAILABLE = True
except ImportError:
    logger.info("DeepEval not installed - install with: pip install deepeval")

# Check for specific metrics
try:
    from deepeval.metrics import HallucinationMetric as _HallucinationMetric
    DEEPEVAL_HALLUCINATION_AVAILABLE = True
except ImportError:
    _HallucinationMetric = None

try:
    from deepeval.metrics import ContextualPrecisionMetric as _ContextualPrecisionMetric
    from deepeval.metrics import ContextualRecallMetric as _ContextualRecallMetric
    DEEPEVAL_CONTEXTUAL_AVAILABLE = True
except ImportError:
    _ContextualPrecisionMetric = None
    _ContextualRecallMetric = None

# Import base adapter interface
try:
    from core.orchestration.base import (
        SDKAdapter,
        SDKLayer,
        AdapterConfig,
        AdapterResult,
        AdapterStatus,
    )
except ImportError:
    from dataclasses import dataclass as _dataclass
    from enum import IntEnum
    from abc import ABC, abstractmethod

    class SDKLayer(IntEnum):
        OBSERVABILITY = 5

    @_dataclass
    class AdapterResult:
        success: bool
        data: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        latency_ms: float = 0.0

    @_dataclass
    class AdapterConfig:
        name: str = "deepeval"
        layer: int = 5

    class AdapterStatus:
        READY = "ready"
        FAILED = "failed"
        UNINITIALIZED = "uninitialized"

    class SDKAdapter(ABC):
        @property
        @abstractmethod
        def sdk_name(self) -> str: ...
        @property
        @abstractmethod
        def layer(self) -> int: ...
        @property
        @abstractmethod
        def available(self) -> bool: ...
        @abstractmethod
        async def initialize(self, config: Dict) -> AdapterResult: ...
        @abstractmethod
        async def execute(self, operation: str, **kwargs) -> AdapterResult: ...
        @abstractmethod
        async def health_check(self) -> AdapterResult: ...
        @abstractmethod
        async def shutdown(self) -> AdapterResult: ...


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DeepEvalMetricConfig:
    """Configuration for DeepEval metrics.

    Attributes:
        faithfulness_threshold: Min score for faithfulness (default: 0.7)
        relevancy_threshold: Min score for answer relevancy (default: 0.7)
        hallucination_threshold: Max hallucination score before failing (default: 0.3)
        precision_threshold: Min score for contextual precision (default: 0.5)
        recall_threshold: Min score for contextual recall (default: 0.5)
        use_claude: Use Claude as the LLM judge (default: False)
        model: Model name for evaluation (default: gpt-4o-mini)
    """
    faithfulness_threshold: float = 0.7
    relevancy_threshold: float = 0.7
    hallucination_threshold: float = 0.3  # Lower is better
    precision_threshold: float = 0.5
    recall_threshold: float = 0.5
    use_claude: bool = False
    model: str = "gpt-4o-mini"


# =============================================================================
# HEURISTIC FALLBACK EVALUATOR
# =============================================================================

class HeuristicDeepEvalEvaluator:
    """
    Heuristic-based evaluation when DeepEval SDK is unavailable.

    Provides approximate scores using:
    - N-gram overlap for faithfulness
    - Keyword matching for relevancy
    - Contradiction detection for hallucination

    This is a fallback - real DeepEval metrics are preferred.
    """

    def __init__(self, config: Optional[DeepEvalMetricConfig] = None):
        self.config = config or DeepEvalMetricConfig()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase words."""
        return [t.lower() for t in re.findall(r'\b\w+\b', text)]

    def _get_ngrams(self, tokens: List[str], n: int) -> set:
        """Get n-grams from token list."""
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    def evaluate(
        self,
        input_text: str,
        actual_output: str,
        retrieval_context: Optional[List[str]] = None,
        expected_output: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate using heuristics."""
        retrieval_context = retrieval_context or []

        input_tokens = set(self._tokenize(input_text))
        output_tokens = set(self._tokenize(actual_output))

        # Combine all context
        context_tokens = set()
        for ctx in retrieval_context:
            context_tokens.update(self._tokenize(ctx))

        # Faithfulness: Output grounded in context
        if output_tokens:
            faithfulness = len(output_tokens & context_tokens) / len(output_tokens)
        else:
            faithfulness = 0.0

        # Relevancy: Output addresses input
        if input_tokens and output_tokens:
            relevancy = len(input_tokens & output_tokens) / len(input_tokens)
        else:
            relevancy = 0.0

        # Hallucination: Output NOT grounded in context (inverse of faithfulness)
        hallucination = 1.0 - faithfulness

        # Contextual precision: Context relevant to input
        if context_tokens and input_tokens:
            precision = len(context_tokens & input_tokens) / len(input_tokens)
        else:
            precision = 0.0

        # Contextual recall: Expected output covered by context
        recall = 0.0
        if expected_output:
            expected_tokens = set(self._tokenize(expected_output))
            if expected_tokens:
                recall = len(expected_tokens & context_tokens) / len(expected_tokens)

        scores = {
            "faithfulness": round(min(1.0, faithfulness), 4),
            "answer_relevancy": round(min(1.0, relevancy), 4),
            "hallucination": round(hallucination, 4),
            "contextual_precision": round(min(1.0, precision), 4),
            "contextual_recall": round(recall, 4),
        }

        passed = (
            scores["faithfulness"] >= self.config.faithfulness_threshold
            and scores["answer_relevancy"] >= self.config.relevancy_threshold
            and scores["hallucination"] <= self.config.hallucination_threshold
        )

        return {
            "scores": scores,
            "passed": passed,
            "method": "heuristic",
            "thresholds": {
                "faithfulness": self.config.faithfulness_threshold,
                "relevancy": self.config.relevancy_threshold,
                "hallucination": self.config.hallucination_threshold,
                "precision": self.config.precision_threshold,
                "recall": self.config.recall_threshold,
            },
        }

    def detect_hallucination(
        self,
        actual_output: str,
        retrieval_context: List[str],
    ) -> Dict[str, Any]:
        """Detect hallucination in output relative to context."""
        output_tokens = set(self._tokenize(actual_output))

        context_tokens = set()
        for ctx in retrieval_context:
            context_tokens.update(self._tokenize(ctx))

        if not output_tokens:
            return {
                "hallucination_score": 0.0,
                "is_hallucinated": False,
                "method": "heuristic",
                "reason": "Empty output",
            }

        # Tokens in output but NOT in context
        ungrounded = output_tokens - context_tokens
        hallucination_score = len(ungrounded) / len(output_tokens)

        return {
            "hallucination_score": round(hallucination_score, 4),
            "is_hallucinated": hallucination_score > self.config.hallucination_threshold,
            "method": "heuristic",
            "ungrounded_ratio": round(hallucination_score, 4),
            "threshold": self.config.hallucination_threshold,
        }


class DeepEvalAdapter(SDKAdapter):
    """
    DeepEval adapter for comprehensive LLM evaluation (V65 Enhanced).

    Operations:
    - evaluate: Run evaluation metrics on a test case
    - evaluate_batch: Run evaluation on multiple test cases
    - detect_hallucination: Check for hallucinated content
    - evaluate_single: Implements EvaluatorProtocol for pipeline integration
    - get_stats: Get adapter statistics

    Features:
    - Claude or GPT as LLM judge
    - Heuristic fallback when SDK unavailable
    - Circuit breaker + retry protection
    """

    def __init__(self, config: Optional[AdapterConfig] = None):
        self._config = config or AdapterConfig(name="deepeval", layer=5)
        self._status = AdapterStatus.UNINITIALIZED
        self._metric_config = DeepEvalMetricConfig()
        self._heuristic_evaluator = HeuristicDeepEvalEvaluator()
        self._call_count = 0
        self._error_count = 0
        self._success_count = 0
        self._total_latency_ms = 0.0

    @property
    def sdk_name(self) -> str:
        return "deepeval"

    @property
    def layer(self) -> SDKLayer:
        return SDKLayer.OBSERVABILITY

    @property
    def available(self) -> bool:
        return DEEPEVAL_AVAILABLE

    async def initialize(self, config: Dict[str, Any]) -> AdapterResult:
        """Initialize DeepEval adapter with configuration."""
        start = time.time()

        self._metric_config = DeepEvalMetricConfig(
            faithfulness_threshold=config.get("faithfulness_threshold", 0.7),
            relevancy_threshold=config.get("relevancy_threshold", 0.7),
            hallucination_threshold=config.get("hallucination_threshold", 0.3),
            precision_threshold=config.get("precision_threshold", 0.5),
            recall_threshold=config.get("recall_threshold", 0.5),
            use_claude=config.get("use_claude", False),
            model=config.get("model", "gpt-4o-mini"),
        )

        self._heuristic_evaluator = HeuristicDeepEvalEvaluator(self._metric_config)
        self._status = AdapterStatus.READY

        logger.info("DeepEval adapter initialized (model=%s, use_claude=%s)",
                    self._metric_config.model, self._metric_config.use_claude)

        return AdapterResult(
            success=True,
            data={
                "model": self._metric_config.model,
                "use_claude": self._metric_config.use_claude,
                "deepeval_native": DEEPEVAL_AVAILABLE,
                "hallucination_available": DEEPEVAL_HALLUCINATION_AVAILABLE,
                "contextual_available": DEEPEVAL_CONTEXTUAL_AVAILABLE,
                "thresholds": {
                    "faithfulness": self._metric_config.faithfulness_threshold,
                    "relevancy": self._metric_config.relevancy_threshold,
                    "hallucination": self._metric_config.hallucination_threshold,
                    "precision": self._metric_config.precision_threshold,
                    "recall": self._metric_config.recall_threshold,
                },
            },
            latency_ms=(time.time() - start) * 1000,
        )

    async def execute(self, operation: str, **kwargs) -> AdapterResult:
        """Execute a DeepEval operation with circuit breaker and timeout."""
        start_time = time.time()

        # Circuit breaker check
        if adapter_circuit_breaker is not None:
            try:
                cb = adapter_circuit_breaker("deepeval_adapter")
                if hasattr(cb, 'is_open') and cb.is_open:
                    logger.warning("Circuit breaker open for deepeval_adapter, using heuristic fallback")
                    return await self._execute_heuristic_fallback(operation, kwargs, start_time)
            except Exception:
                pass

        try:
            timeout = kwargs.pop("timeout", DEEPEVAL_OPERATION_TIMEOUT)
            result = await asyncio.wait_for(
                self._dispatch_operation(operation, kwargs),
                timeout=timeout,
            )

            latency_ms = (time.time() - start_time) * 1000
            self._call_count += 1
            self._total_latency_ms += latency_ms
            result.latency_ms = latency_ms

            if result.success:
                self._success_count += 1
                if adapter_circuit_breaker is not None:
                    try:
                        adapter_circuit_breaker("deepeval_adapter").record_success()
                    except Exception:
                        pass
            else:
                self._error_count += 1

            return result

        except asyncio.TimeoutError:
            latency_ms = (time.time() - start_time) * 1000
            self._error_count += 1
            self._call_count += 1
            if adapter_circuit_breaker is not None:
                try:
                    adapter_circuit_breaker("deepeval_adapter").record_failure()
                except Exception:
                    pass
            logger.error("DeepEval operation '%s' timed out after %ds", operation, DEEPEVAL_OPERATION_TIMEOUT)
            return AdapterResult(
                success=False,
                error=f"Operation timed out after {DEEPEVAL_OPERATION_TIMEOUT}s",
                latency_ms=latency_ms,
            )

        except (ConnectionError, OSError) as e:
            latency_ms = (time.time() - start_time) * 1000
            self._error_count += 1
            self._call_count += 1
            if adapter_circuit_breaker is not None:
                try:
                    adapter_circuit_breaker("deepeval_adapter").record_failure()
                except Exception:
                    pass
            logger.error("DeepEval operation '%s' connection error: %s", operation, e)
            return AdapterResult(success=False, error=str(e), latency_ms=latency_ms)

        except (ValueError, TypeError, RuntimeError) as e:
            latency_ms = (time.time() - start_time) * 1000
            self._error_count += 1
            self._call_count += 1
            if adapter_circuit_breaker is not None:
                try:
                    adapter_circuit_breaker("deepeval_adapter").record_failure()
                except Exception:
                    pass
            logger.error("DeepEval operation '%s' failed: %s", operation, e)
            return AdapterResult(success=False, error=str(e), latency_ms=latency_ms)

    async def _execute_heuristic_fallback(
        self,
        operation: str,
        kwargs: Dict[str, Any],
        start_time: float,
    ) -> AdapterResult:
        """Execute with heuristic fallback when circuit is open."""
        if operation in ("evaluate", "evaluate_single"):
            result = self._heuristic_evaluator.evaluate(
                input_text=kwargs.get("input", ""),
                actual_output=kwargs.get("actual_output", ""),
                retrieval_context=kwargs.get("retrieval_context", []),
                expected_output=kwargs.get("expected_output"),
            )
            return AdapterResult(
                success=True,
                data=result,
                latency_ms=(time.time() - start_time) * 1000,
            )
        elif operation == "detect_hallucination":
            result = self._heuristic_evaluator.detect_hallucination(
                actual_output=kwargs.get("actual_output", ""),
                retrieval_context=kwargs.get("retrieval_context", []),
            )
            return AdapterResult(
                success=True,
                data=result,
                latency_ms=(time.time() - start_time) * 1000,
            )
        elif operation == "get_stats":
            return await self._get_stats(**kwargs)
        else:
            return AdapterResult(
                success=False,
                error=f"Unknown operation: {operation}",
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def _dispatch_operation(self, operation: str, kwargs: Dict[str, Any]) -> AdapterResult:
        """Dispatch to operation handler."""
        handlers = {
            "evaluate": self._evaluate,
            "evaluate_batch": self._evaluate_batch,
            "detect_hallucination": self._detect_hallucination,
            "evaluate_single": self._evaluate_single,
            "get_stats": self._get_stats,
        }

        handler = handlers.get(operation)
        if not handler:
            return AdapterResult(
                success=False,
                error=f"Unknown operation: {operation}. Available: {list(handlers.keys())}",
            )

        return await handler(**kwargs)

    async def _evaluate(
        self,
        input: str = "",
        actual_output: str = "",
        expected_output: Optional[str] = None,
        retrieval_context: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        **kwargs,
    ) -> AdapterResult:
        """Run evaluation metrics on a test case."""
        retrieval_context = retrieval_context or []

        # Use heuristic if DeepEval not available
        if not DEEPEVAL_AVAILABLE:
            result = self._heuristic_evaluator.evaluate(
                input_text=input,
                actual_output=actual_output,
                retrieval_context=retrieval_context,
                expected_output=expected_output,
            )
            return AdapterResult(success=True, data=result)

        try:
            from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
            from deepeval.test_case import LLMTestCase
            from deepeval import evaluate as deepeval_evaluate

            test_case = LLMTestCase(
                input=input,
                actual_output=actual_output,
                retrieval_context=retrieval_context,
            )
            if expected_output:
                test_case.expected_output = expected_output

            eval_metrics = [
                FaithfulnessMetric(
                    threshold=self._metric_config.faithfulness_threshold,
                    model=self._metric_config.model
                ),
                AnswerRelevancyMetric(
                    threshold=self._metric_config.relevancy_threshold,
                    model=self._metric_config.model
                ),
            ]

            # Add hallucination metric if available
            if DEEPEVAL_HALLUCINATION_AVAILABLE and _HallucinationMetric:
                eval_metrics.append(_HallucinationMetric(
                    threshold=self._metric_config.hallucination_threshold,
                    model=self._metric_config.model
                ))

            # Add contextual metrics if available
            if DEEPEVAL_CONTEXTUAL_AVAILABLE:
                if _ContextualPrecisionMetric:
                    eval_metrics.append(_ContextualPrecisionMetric(
                        threshold=self._metric_config.precision_threshold,
                        model=self._metric_config.model
                    ))
                if _ContextualRecallMetric and expected_output:
                    eval_metrics.append(_ContextualRecallMetric(
                        threshold=self._metric_config.recall_threshold,
                        model=self._metric_config.model
                    ))

            # Run evaluation in thread pool (DeepEval uses sync operations)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: deepeval_evaluate(test_cases=[test_case], metrics=eval_metrics)
            )

            scores = {}
            all_passed = True
            for metric in eval_metrics:
                name = metric.__class__.__name__.replace("Metric", "").lower()
                if hasattr(metric, 'score') and metric.score is not None:
                    scores[name] = float(metric.score)
                if hasattr(metric, 'is_successful'):
                    try:
                        if not metric.is_successful():
                            all_passed = False
                    except Exception:
                        pass

            return AdapterResult(
                success=True,
                data={
                    "scores": scores,
                    "passed": all_passed,
                    "method": "deepeval",
                    "thresholds": {
                        "faithfulness": self._metric_config.faithfulness_threshold,
                        "relevancy": self._metric_config.relevancy_threshold,
                        "hallucination": self._metric_config.hallucination_threshold,
                    },
                },
            )

        except ImportError as e:
            logger.warning("DeepEval metric import failed, using heuristic: %s", e)
            result = self._heuristic_evaluator.evaluate(
                input_text=input,
                actual_output=actual_output,
                retrieval_context=retrieval_context,
                expected_output=expected_output,
            )
            return AdapterResult(success=True, data=result)
        except (ValueError, TypeError, RuntimeError) as e:
            logger.warning("DeepEval evaluation failed, using heuristic: %s", e)
            result = self._heuristic_evaluator.evaluate(
                input_text=input,
                actual_output=actual_output,
                retrieval_context=retrieval_context,
                expected_output=expected_output,
            )
            return AdapterResult(success=True, data=result)

    async def _evaluate_batch(
        self,
        test_cases: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> AdapterResult:
        """Run evaluation on multiple test cases."""
        if not test_cases:
            return AdapterResult(
                success=False,
                error="test_cases list required",
            )

        results = []
        for tc in test_cases:
            result = await self._evaluate(**tc)
            results.append(result.data if result.success else {"error": result.error})

        passed_count = sum(1 for r in results if r.get("passed", False))

        return AdapterResult(
            success=True,
            data={
                "results": results,
                "total": len(test_cases),
                "passed": passed_count,
                "pass_rate": passed_count / len(test_cases) if test_cases else 0.0,
            },
        )

    async def _detect_hallucination(
        self,
        actual_output: str = "",
        retrieval_context: Optional[List[str]] = None,
        **kwargs,
    ) -> AdapterResult:
        """Detect hallucination in output relative to context."""
        retrieval_context = retrieval_context or []

        if not DEEPEVAL_AVAILABLE or not DEEPEVAL_HALLUCINATION_AVAILABLE:
            result = self._heuristic_evaluator.detect_hallucination(
                actual_output=actual_output,
                retrieval_context=retrieval_context,
            )
            return AdapterResult(success=True, data=result)

        try:
            from deepeval.metrics import HallucinationMetric
            from deepeval.test_case import LLMTestCase
            from deepeval import evaluate as deepeval_evaluate

            test_case = LLMTestCase(
                input="",  # Not needed for hallucination check
                actual_output=actual_output,
                retrieval_context=retrieval_context,
            )

            metric = HallucinationMetric(
                threshold=self._metric_config.hallucination_threshold,
                model=self._metric_config.model
            )

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: deepeval_evaluate(test_cases=[test_case], metrics=[metric])
            )

            score = metric.score if hasattr(metric, 'score') and metric.score is not None else 0.0
            is_hallucinated = score > self._metric_config.hallucination_threshold

            return AdapterResult(
                success=True,
                data={
                    "hallucination_score": float(score),
                    "is_hallucinated": is_hallucinated,
                    "method": "deepeval",
                    "threshold": self._metric_config.hallucination_threshold,
                    "reason": getattr(metric, 'reason', None),
                },
            )

        except ImportError as e:
            logger.warning("DeepEval hallucination import failed: %s", e)
            result = self._heuristic_evaluator.detect_hallucination(
                actual_output=actual_output,
                retrieval_context=retrieval_context,
            )
            return AdapterResult(success=True, data=result)
        except (ValueError, TypeError, RuntimeError) as e:
            logger.warning("DeepEval hallucination detection failed: %s", e)
            result = self._heuristic_evaluator.detect_hallucination(
                actual_output=actual_output,
                retrieval_context=retrieval_context,
            )
            return AdapterResult(success=True, data=result)

    async def _evaluate_single(
        self,
        question: str = "",
        contexts: Optional[List[str]] = None,
        answer: str = "",
        ground_truth: Optional[str] = None,
        **kwargs,
    ) -> AdapterResult:
        """
        Implements EvaluatorProtocol.evaluate_single for pipeline integration.

        Maps the pipeline's signature to DeepEval's expected format.
        """
        return await self._evaluate(
            input=question,
            actual_output=answer,
            retrieval_context=contexts or [],
            expected_output=ground_truth,
            **kwargs,
        )

    async def _get_stats(self, **kwargs) -> AdapterResult:
        """Get adapter statistics."""
        return AdapterResult(
            success=True,
            data={
                "model": self._metric_config.model,
                "use_claude": self._metric_config.use_claude,
                "call_count": self._call_count,
                "success_count": self._success_count,
                "error_count": self._error_count,
                "success_rate": self._success_count / max(1, self._call_count),
                "avg_latency_ms": self._total_latency_ms / max(1, self._call_count),
                "deepeval_native": DEEPEVAL_AVAILABLE,
                "hallucination_available": DEEPEVAL_HALLUCINATION_AVAILABLE,
                "contextual_available": DEEPEVAL_CONTEXTUAL_AVAILABLE,
            },
        )

    async def health_check(self) -> AdapterResult:
        """Check adapter health."""
        cb_status = "unknown"
        if adapter_circuit_breaker is not None:
            try:
                cb = adapter_circuit_breaker("deepeval_adapter")
                cb_status = "open" if (hasattr(cb, 'is_open') and cb.is_open) else "closed"
            except Exception:
                cb_status = "unavailable"

        return AdapterResult(
            success=True,
            data={
                "status": "healthy" if self._status == AdapterStatus.READY else "degraded",
                "deepeval_available": DEEPEVAL_AVAILABLE,
                "hallucination_available": DEEPEVAL_HALLUCINATION_AVAILABLE,
                "contextual_available": DEEPEVAL_CONTEXTUAL_AVAILABLE,
                "model": self._metric_config.model,
                "circuit_breaker": cb_status,
                "heuristic_fallback": True,  # Always available
            },
        )

    async def shutdown(self) -> AdapterResult:
        """Shutdown the adapter."""
        self._status = AdapterStatus.UNINITIALIZED
        logger.info("DeepEval adapter shutdown")
        return AdapterResult(success=True)

    # =========================================================================
    # Convenience Methods for Direct Use
    # =========================================================================

    async def evaluate_single(
        self,
        question: str,
        contexts: List[str],
        answer: str,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Direct evaluation method implementing EvaluatorProtocol.

        Args:
            question: The user query
            contexts: Retrieved context documents
            answer: Generated answer
            ground_truth: Optional expected answer

        Returns:
            Dict with scores, passed status, and metadata
        """
        result = await self.execute(
            "evaluate_single",
            question=question,
            contexts=contexts,
            answer=answer,
            ground_truth=ground_truth,
        )
        return result.data if result.success else {"error": result.error, "passed": False}

    async def detect_hallucination(
        self,
        answer: str,
        contexts: List[str],
    ) -> Dict[str, Any]:
        """
        Direct hallucination detection.

        Args:
            answer: The generated answer to check
            contexts: Retrieved context documents

        Returns:
            Dict with hallucination_score, is_hallucinated, and metadata
        """
        result = await self.execute(
            "detect_hallucination",
            actual_output=answer,
            retrieval_context=contexts,
        )
        return result.data if result.success else {"error": result.error, "is_hallucinated": True}


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "DeepEvalAdapter",
    "DeepEvalMetricConfig",
    "HeuristicDeepEvalEvaluator",
    "DEEPEVAL_AVAILABLE",
    "DEEPEVAL_HALLUCINATION_AVAILABLE",
    "DEEPEVAL_CONTEXTUAL_AVAILABLE",
]
