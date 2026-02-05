"""
Ragas Adapter - RAG Evaluation Framework (V65 Enhanced)
=========================================================

Standalone adapter for Ragas evaluation metrics with full Gap11 support.
Implements comprehensive RAG quality scoring using Ragas >= 0.4.x.

Features:
- Faithfulness scoring (groundedness to source)
- Answer relevancy (query-answer alignment)
- Context precision (relevant context ranking)
- Context recall (ground truth coverage)
- LLM-as-judge evaluation with Claude or GPT
- Async/sync hybrid execution
- Circuit breaker + retry protection

SDK: ragas >= 0.4.0
Layer: L5 (Observability/Evaluation)

Gap11 Status: RESOLVED (V65)

Usage:
    adapter = RagasAdapter()
    await adapter.initialize({"llm_model": "claude-3-5-sonnet-20241022"})
    result = await adapter.execute("evaluate",
        question="What is X?", answer="X is ...",
        contexts=["context1", "context2"],
        ground_truth="X is the answer")

    # Full metrics suite
    result = await adapter.execute("evaluate_full",
        question="What is X?", answer="X is ...",
        contexts=["context1"], ground_truth="Expected answer",
        metrics=["faithfulness", "answer_relevancy", "context_precision", "context_recall"])
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)

# Retry for production resilience
try:
    from .retry import RetryConfig, retry_async
    RAGAS_RETRY_CONFIG = RetryConfig(
        max_retries=3, base_delay=1.0, max_delay=30.0, jitter=0.5
    )
except ImportError:
    RetryConfig = None
    retry_async = None
    RAGAS_RETRY_CONFIG = None

# Circuit breaker for production resilience
try:
    from .circuit_breaker_manager import adapter_circuit_breaker, CircuitOpenError
except ImportError:
    adapter_circuit_breaker = None
    CircuitOpenError = Exception

# Default timeout (evaluation involves LLM calls)
RAGAS_OPERATION_TIMEOUT = 120

# SDK availability flags
RAGAS_AVAILABLE = False
RAGAS_METRICS_AVAILABLE = False
ANTHROPIC_LLM_AVAILABLE = False

try:
    import ragas
    RAGAS_AVAILABLE = True
except ImportError:
    logger.info("Ragas not installed - install with: pip install ragas")

# Check for specific metric imports (Ragas 0.4.x structure)
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    RAGAS_METRICS_AVAILABLE = True
except ImportError:
    ragas_evaluate = None
    SingleTurnSample = None
    EvaluationDataset = None

# Check for Anthropic LLM wrapper (for Claude-as-judge)
try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_LLM_AVAILABLE = True
except ImportError:
    ChatAnthropic = None

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
        name: str = "ragas"
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
# METRIC CONFIGURATION
# =============================================================================

@dataclass
class RagasMetricConfig:
    """Configuration for Ragas metrics.

    Attributes:
        faithfulness_threshold: Min score for faithfulness (default: 0.7)
        relevancy_threshold: Min score for answer relevancy (default: 0.6)
        precision_threshold: Min score for context precision (default: 0.5)
        recall_threshold: Min score for context recall (default: 0.5)
        use_claude: Use Claude as the LLM judge (default: False, uses GPT)
        claude_model: Claude model to use (default: claude-3-5-sonnet-20241022)
        openai_model: OpenAI model to use (default: gpt-4o-mini)
    """
    faithfulness_threshold: float = 0.7
    relevancy_threshold: float = 0.6
    precision_threshold: float = 0.5
    recall_threshold: float = 0.5
    use_claude: bool = False
    claude_model: str = "claude-3-5-sonnet-20241022"
    openai_model: str = "gpt-4o-mini"


# =============================================================================
# HEURISTIC FALLBACK EVALUATOR (No LLM Required)
# =============================================================================

class HeuristicRagasEvaluator:
    """
    Heuristic-based RAG evaluation when Ragas SDK is unavailable.

    Provides approximate scores using:
    - Term overlap for faithfulness
    - Semantic similarity (Jaccard) for relevancy
    - Keyword matching for precision/recall

    This is a fallback - real Ragas metrics are preferred.
    """

    def __init__(self, config: Optional[RagasMetricConfig] = None):
        self.config = config or RagasMetricConfig()

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate using heuristics."""
        import re
        from collections import Counter

        def tokenize(text: str) -> List[str]:
            return [t.lower() for t in re.findall(r'\b\w+\b', text)]

        q_tokens = set(tokenize(question))
        a_tokens = set(tokenize(answer))
        c_tokens = set()
        for ctx in contexts:
            c_tokens.update(tokenize(ctx))

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
            gt_tokens = set(tokenize(ground_truth))
            if gt_tokens:
                recall = len(gt_tokens & c_tokens) / len(gt_tokens)

        scores = {
            "faithfulness": round(min(1.0, faithfulness), 4),
            "answer_relevancy": round(min(1.0, relevancy), 4),
            "context_precision": round(min(1.0, precision), 4),
            "context_recall": round(recall, 4),
        }

        passed = (
            scores["faithfulness"] >= self.config.faithfulness_threshold
            and scores["answer_relevancy"] >= self.config.relevancy_threshold
            and scores["context_precision"] >= self.config.precision_threshold
        )

        return {
            "scores": scores,
            "passed": passed,
            "method": "heuristic",
            "thresholds": {
                "faithfulness": self.config.faithfulness_threshold,
                "relevancy": self.config.relevancy_threshold,
                "precision": self.config.precision_threshold,
                "recall": self.config.recall_threshold,
            },
        }


class RagasAdapter(SDKAdapter):
    """
    Ragas adapter for comprehensive RAG evaluation (V65 Enhanced).

    Operations:
    - evaluate: Run core Ragas evaluation metrics
    - evaluate_full: Run full metric suite with context recall
    - evaluate_single: Implements EvaluatorProtocol for pipeline integration
    - get_stats: Get adapter statistics

    Features:
    - Claude or GPT as LLM judge
    - Heuristic fallback when SDK unavailable
    - Circuit breaker + retry protection
    - Async execution with timeout
    """

    def __init__(self, config: Optional[AdapterConfig] = None):
        self._config = config or AdapterConfig(name="ragas", layer=5)
        self._status = AdapterStatus.UNINITIALIZED
        self._metric_config = RagasMetricConfig()
        self._heuristic_evaluator = HeuristicRagasEvaluator()
        self._llm = None
        self._embeddings = None
        self._call_count = 0
        self._error_count = 0
        self._success_count = 0
        self._total_latency_ms = 0.0

    @property
    def sdk_name(self) -> str:
        return "ragas"

    @property
    def layer(self) -> SDKLayer:
        return SDKLayer.OBSERVABILITY

    @property
    def available(self) -> bool:
        return RAGAS_AVAILABLE

    async def initialize(self, config: Dict[str, Any]) -> AdapterResult:
        """Initialize Ragas adapter with LLM configuration."""
        start = time.time()

        # Update metric config from initialization params
        self._metric_config = RagasMetricConfig(
            faithfulness_threshold=config.get("faithfulness_threshold", 0.7),
            relevancy_threshold=config.get("relevancy_threshold", 0.6),
            precision_threshold=config.get("precision_threshold", 0.5),
            recall_threshold=config.get("recall_threshold", 0.5),
            use_claude=config.get("use_claude", False),
            claude_model=config.get("claude_model", "claude-3-5-sonnet-20241022"),
            openai_model=config.get("openai_model", "gpt-4o-mini"),
        )

        # Update heuristic evaluator with new config
        self._heuristic_evaluator = HeuristicRagasEvaluator(self._metric_config)

        # Initialize LLM wrapper if Ragas is available
        if RAGAS_AVAILABLE and RAGAS_METRICS_AVAILABLE:
            try:
                self._llm = self._create_llm_wrapper()
            except Exception as e:
                logger.warning("Failed to create LLM wrapper: %s. Using heuristic fallback.", e)

        self._status = AdapterStatus.READY
        llm_model = (
            self._metric_config.claude_model
            if self._metric_config.use_claude
            else self._metric_config.openai_model
        )
        logger.info("Ragas adapter initialized (model=%s, ragas_native=%s)",
                    llm_model, RAGAS_AVAILABLE)

        return AdapterResult(
            success=True,
            data={
                "llm_model": llm_model,
                "use_claude": self._metric_config.use_claude,
                "ragas_native": RAGAS_AVAILABLE,
                "ragas_metrics": RAGAS_METRICS_AVAILABLE,
                "anthropic_llm": ANTHROPIC_LLM_AVAILABLE,
                "thresholds": {
                    "faithfulness": self._metric_config.faithfulness_threshold,
                    "relevancy": self._metric_config.relevancy_threshold,
                    "precision": self._metric_config.precision_threshold,
                    "recall": self._metric_config.recall_threshold,
                },
            },
            latency_ms=(time.time() - start) * 1000,
        )

    def _create_llm_wrapper(self) -> Any:
        """Create LLM wrapper for Ragas metrics."""
        from ragas.llms import LangchainLLMWrapper

        if self._metric_config.use_claude and ANTHROPIC_LLM_AVAILABLE:
            llm = ChatAnthropic(model=self._metric_config.claude_model)
        else:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=self._metric_config.openai_model)

        return LangchainLLMWrapper(llm)

    async def execute(self, operation: str, **kwargs) -> AdapterResult:
        """Execute a Ragas operation with circuit breaker and timeout."""
        start_time = time.time()

        # Circuit breaker check
        if adapter_circuit_breaker is not None:
            try:
                cb = adapter_circuit_breaker("ragas_adapter")
                if hasattr(cb, 'is_open') and cb.is_open:
                    # Fall back to heuristic when circuit is open
                    logger.warning("Circuit breaker open for ragas_adapter, using heuristic fallback")
                    return await self._execute_heuristic_fallback(operation, kwargs, start_time)
            except Exception:
                pass

        try:
            timeout = kwargs.pop("timeout", RAGAS_OPERATION_TIMEOUT)
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
                # Record success
                if adapter_circuit_breaker is not None:
                    try:
                        adapter_circuit_breaker("ragas_adapter").record_success()
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
                    adapter_circuit_breaker("ragas_adapter").record_failure()
                except Exception:
                    pass
            logger.error("Ragas operation '%s' timed out after %ds", operation, RAGAS_OPERATION_TIMEOUT)
            return AdapterResult(
                success=False,
                error=f"Operation timed out after {RAGAS_OPERATION_TIMEOUT}s",
                latency_ms=latency_ms,
            )

        except (ConnectionError, OSError) as e:
            latency_ms = (time.time() - start_time) * 1000
            self._error_count += 1
            self._call_count += 1
            if adapter_circuit_breaker is not None:
                try:
                    adapter_circuit_breaker("ragas_adapter").record_failure()
                except Exception:
                    pass
            logger.error("Ragas operation '%s' connection error: %s", operation, e)
            return AdapterResult(success=False, error=str(e), latency_ms=latency_ms)

        except (ValueError, TypeError, RuntimeError) as e:
            latency_ms = (time.time() - start_time) * 1000
            self._error_count += 1
            self._call_count += 1
            if adapter_circuit_breaker is not None:
                try:
                    adapter_circuit_breaker("ragas_adapter").record_failure()
                except Exception:
                    pass
            logger.error("Ragas operation '%s' failed: %s", operation, e)
            return AdapterResult(success=False, error=str(e), latency_ms=latency_ms)

    async def _execute_heuristic_fallback(
        self,
        operation: str,
        kwargs: Dict[str, Any],
        start_time: float,
    ) -> AdapterResult:
        """Execute with heuristic fallback when circuit is open."""
        if operation in ("evaluate", "evaluate_full", "evaluate_single"):
            result = self._heuristic_evaluator.evaluate(
                question=kwargs.get("question", ""),
                answer=kwargs.get("answer", ""),
                contexts=kwargs.get("contexts", []),
                ground_truth=kwargs.get("ground_truth"),
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
            "evaluate_full": self._evaluate_full,
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
        question: str = "",
        answer: str = "",
        contexts: Optional[List[str]] = None,
        ground_truth: Optional[str] = None,
        **kwargs,
    ) -> AdapterResult:
        """Run core Ragas evaluation metrics (faithfulness, relevancy, precision)."""
        contexts = contexts or []

        # Use heuristic if Ragas not available
        if not RAGAS_AVAILABLE or not RAGAS_METRICS_AVAILABLE:
            result = self._heuristic_evaluator.evaluate(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
            )
            return AdapterResult(success=True, data=result)

        try:
            # Import metrics
            try:
                from ragas.metrics._faithfulness import Faithfulness
                from ragas.metrics._answer_relevance import AnswerRelevancy
                from ragas.metrics._context_precision import ContextPrecision
            except ImportError:
                try:
                    from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision
                except ImportError:
                    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision

            # Use cached LLM or create new one
            if self._llm is None:
                self._llm = self._create_llm_wrapper()

            metrics = [
                Faithfulness(llm=self._llm),
                AnswerRelevancy(llm=self._llm),
                ContextPrecision(llm=self._llm),
            ]

            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
            )
            if ground_truth:
                sample.reference = ground_truth

            dataset = EvaluationDataset(samples=[sample])

            # Run evaluation in thread pool (Ragas uses sync operations internally)
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: ragas_evaluate(dataset=dataset, metrics=metrics)
            )

            scores = {}
            for metric_name in ["faithfulness", "answer_relevancy", "context_precision"]:
                if metric_name in results:
                    val = results[metric_name]
                    scores[metric_name] = float(val) if val is not None else 0.0

            passed = (
                scores.get("faithfulness", 0.0) >= self._metric_config.faithfulness_threshold
                and scores.get("answer_relevancy", 0.0) >= self._metric_config.relevancy_threshold
                and scores.get("context_precision", 0.0) >= self._metric_config.precision_threshold
            )

            return AdapterResult(
                success=True,
                data={
                    "scores": scores,
                    "passed": passed,
                    "method": "ragas",
                    "thresholds": {
                        "faithfulness": self._metric_config.faithfulness_threshold,
                        "relevancy": self._metric_config.relevancy_threshold,
                        "precision": self._metric_config.precision_threshold,
                    },
                },
            )

        except ImportError as e:
            logger.warning("Ragas metric import failed, using heuristic: %s", e)
            result = self._heuristic_evaluator.evaluate(question, answer, contexts, ground_truth)
            return AdapterResult(success=True, data=result)
        except (ValueError, TypeError, RuntimeError) as e:
            logger.warning("Ragas evaluation failed, using heuristic: %s", e)
            result = self._heuristic_evaluator.evaluate(question, answer, contexts, ground_truth)
            return AdapterResult(success=True, data=result)

    async def _evaluate_full(
        self,
        question: str = "",
        answer: str = "",
        contexts: Optional[List[str]] = None,
        ground_truth: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        **kwargs,
    ) -> AdapterResult:
        """Run full Ragas evaluation with context recall and all metrics."""
        contexts = contexts or []
        requested_metrics = metrics or ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

        if not RAGAS_AVAILABLE or not RAGAS_METRICS_AVAILABLE:
            result = self._heuristic_evaluator.evaluate(question, answer, contexts, ground_truth)
            return AdapterResult(success=True, data=result)

        try:
            # Import all available metrics
            available_metrics = {}

            try:
                from ragas.metrics._faithfulness import Faithfulness
                from ragas.metrics._answer_relevance import AnswerRelevancy
                from ragas.metrics._context_precision import ContextPrecision
                from ragas.metrics._context_recall import ContextRecall
                available_metrics = {
                    "faithfulness": Faithfulness,
                    "answer_relevancy": AnswerRelevancy,
                    "context_precision": ContextPrecision,
                    "context_recall": ContextRecall,
                }
            except ImportError:
                try:
                    from ragas.metrics import (
                        Faithfulness,
                        AnswerRelevancy,
                        ContextPrecision,
                        ContextRecall,
                    )
                    available_metrics = {
                        "faithfulness": Faithfulness,
                        "answer_relevancy": AnswerRelevancy,
                        "context_precision": ContextPrecision,
                        "context_recall": ContextRecall,
                    }
                except ImportError as e:
                    logger.warning("Could not import all metrics: %s", e)

            if self._llm is None:
                self._llm = self._create_llm_wrapper()

            # Build metric instances for requested metrics
            metric_instances = []
            for metric_name in requested_metrics:
                if metric_name in available_metrics:
                    metric_instances.append(available_metrics[metric_name](llm=self._llm))

            if not metric_instances:
                result = self._heuristic_evaluator.evaluate(question, answer, contexts, ground_truth)
                return AdapterResult(success=True, data=result)

            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
            )
            if ground_truth:
                sample.reference = ground_truth

            dataset = EvaluationDataset(samples=[sample])

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: ragas_evaluate(dataset=dataset, metrics=metric_instances)
            )

            scores = {}
            for metric_name in requested_metrics:
                if metric_name in results:
                    val = results[metric_name]
                    scores[metric_name] = float(val) if val is not None else 0.0

            # Check pass/fail against all configured thresholds
            passed = True
            if "faithfulness" in scores:
                passed = passed and scores["faithfulness"] >= self._metric_config.faithfulness_threshold
            if "answer_relevancy" in scores:
                passed = passed and scores["answer_relevancy"] >= self._metric_config.relevancy_threshold
            if "context_precision" in scores:
                passed = passed and scores["context_precision"] >= self._metric_config.precision_threshold
            if "context_recall" in scores:
                passed = passed and scores["context_recall"] >= self._metric_config.recall_threshold

            return AdapterResult(
                success=True,
                data={
                    "scores": scores,
                    "passed": passed,
                    "method": "ragas_full",
                    "metrics_evaluated": list(scores.keys()),
                    "thresholds": {
                        "faithfulness": self._metric_config.faithfulness_threshold,
                        "relevancy": self._metric_config.relevancy_threshold,
                        "precision": self._metric_config.precision_threshold,
                        "recall": self._metric_config.recall_threshold,
                    },
                },
            )

        except ImportError as e:
            logger.warning("Ragas full evaluation import failed: %s", e)
            result = self._heuristic_evaluator.evaluate(question, answer, contexts, ground_truth)
            return AdapterResult(success=True, data=result)
        except (ValueError, TypeError, RuntimeError) as e:
            logger.warning("Ragas full evaluation failed: %s", e)
            result = self._heuristic_evaluator.evaluate(question, answer, contexts, ground_truth)
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

        This method matches the signature expected by RAGPipeline's EvaluatorProtocol.
        """
        return await self._evaluate(
            question=question,
            answer=answer,
            contexts=contexts or [],
            ground_truth=ground_truth,
            **kwargs,
        )

    async def _get_stats(self, **kwargs) -> AdapterResult:
        """Get adapter statistics."""
        llm_model = (
            self._metric_config.claude_model
            if self._metric_config.use_claude
            else self._metric_config.openai_model
        )
        return AdapterResult(
            success=True,
            data={
                "llm_model": llm_model,
                "use_claude": self._metric_config.use_claude,
                "call_count": self._call_count,
                "success_count": self._success_count,
                "error_count": self._error_count,
                "success_rate": self._success_count / max(1, self._call_count),
                "avg_latency_ms": self._total_latency_ms / max(1, self._call_count),
                "ragas_native": RAGAS_AVAILABLE,
                "ragas_metrics": RAGAS_METRICS_AVAILABLE,
                "anthropic_llm": ANTHROPIC_LLM_AVAILABLE,
            },
        )

    async def health_check(self) -> AdapterResult:
        """Check adapter health."""
        llm_model = (
            self._metric_config.claude_model
            if self._metric_config.use_claude
            else self._metric_config.openai_model
        )

        # Check circuit breaker status
        cb_status = "unknown"
        if adapter_circuit_breaker is not None:
            try:
                cb = adapter_circuit_breaker("ragas_adapter")
                cb_status = "open" if (hasattr(cb, 'is_open') and cb.is_open) else "closed"
            except Exception:
                cb_status = "unavailable"

        return AdapterResult(
            success=True,
            data={
                "status": "healthy" if self._status == AdapterStatus.READY else "degraded",
                "ragas_available": RAGAS_AVAILABLE,
                "ragas_metrics": RAGAS_METRICS_AVAILABLE,
                "llm_model": llm_model,
                "circuit_breaker": cb_status,
                "heuristic_fallback": True,  # Always available
            },
        )

    async def shutdown(self) -> AdapterResult:
        """Shutdown the adapter."""
        self._status = AdapterStatus.UNINITIALIZED
        self._llm = None
        self._embeddings = None
        logger.info("Ragas adapter shutdown")
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

        This is the method signature expected by the RAG pipeline.

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


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "RagasAdapter",
    "RagasMetricConfig",
    "HeuristicRagasEvaluator",
    "RAGAS_AVAILABLE",
    "RAGAS_METRICS_AVAILABLE",
    "ANTHROPIC_LLM_AVAILABLE",
]
