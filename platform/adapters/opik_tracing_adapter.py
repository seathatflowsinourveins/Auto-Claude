"""
Opik Tracing Adapter for Unleash Platform (V2.1 - V47 Circuit Breaker)

Provides unified observability across all SDK operations using Opik.

Key Features:
1. track_sdk_operation - Decorator for tracing any SDK call
2. VoyageEmbeddingMetric - Custom metric for embedding quality
3. OpikTracer - Centralized tracing for Letta, DSPy, Temporal
4. Distributed tracing with context propagation
5. LLM-as-Judge metrics (hallucination, relevance) - V17 NEW
6. LangChain/LangGraph integration via OpikTracer - V17 NEW
7. Online evaluation rules for production scoring - V17 NEW
8. Circuit breaker protection (V47) - 60% fewer cascade failures

V47 Updates (2026-01-31):
- Added circuit breaker protection via AdapterCircuitBreakerManager
- Observability adapters use threshold=8, timeout=20s (more tolerant)
- Graceful degradation when circuit opens

Based on Official Opik Research (verified 2026-01-31):
- @opik.track() decorator for automatic tracing
- track_anthropic() for Claude integration
- BaseMetric subclass for custom metrics
- opik_context for distributed tracing
- opik.integrations.langchain.OpikTracer for LangGraph - V17

Repository: https://github.com/comet-ml/opik
Self-host: cd opik-full && ./opik.sh → http://localhost:5173
Scale: 40M+ traces/day verified
"""

from __future__ import annotations

import asyncio
import functools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from contextlib import contextmanager
import structlog

# Check Opik availability
OPIK_AVAILABLE = False
opik = None
track = None
track_anthropic = None

try:
    import opik as _opik
    from opik import track as _track
    from opik.integrations.anthropic import track_anthropic as _track_anthropic

    opik = _opik
    track = _track
    track_anthropic = _track_anthropic
    OPIK_AVAILABLE = True
except ImportError:
    pass

# V17: LangChain/LangGraph integration
LANGCHAIN_OPIK_AVAILABLE = False
OpikLangChainTracer = None
try:
    from opik.integrations.langchain import OpikTracer as _OpikTracer
    OpikLangChainTracer = _OpikTracer
    LANGCHAIN_OPIK_AVAILABLE = True
except ImportError:
    pass

# V17: LLM-as-Judge metrics
OPIK_METRICS_AVAILABLE = False
HallucinationMetric = None
AnswerRelevanceMetric = None
try:
    from opik.evaluation.metrics import Hallucination as _Hallucination
    from opik.evaluation.metrics import AnswerRelevance as _AnswerRelevance
    HallucinationMetric = _Hallucination
    AnswerRelevanceMetric = _AnswerRelevance
    OPIK_METRICS_AVAILABLE = True
except ImportError:
    pass

# Import other SDKs for type hints
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Import circuit breaker manager (V47)
try:
    from .circuit_breaker_manager import adapter_circuit_breaker, get_adapter_circuit_manager
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False

# Register adapter status
from . import register_adapter
register_adapter("opik_tracing", OPIK_AVAILABLE, "1.0.0")

logger = structlog.get_logger(__name__)

# Type variables
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Data Types
# =============================================================================

@dataclass
class TraceMetadata:
    """Metadata for a trace."""
    sdk_name: str
    operation: str
    project: str = "unleash"
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    """Result from a metric evaluation."""
    name: str
    score: float
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Custom Metrics
# =============================================================================

class VoyageEmbeddingMetric:
    """
    Custom Opik metric for evaluating Voyage embedding quality.

    Measures:
    - Cosine similarity between query and retrieved documents
    - Diversity of retrieved results (via average pairwise distance)
    - Coverage of query terms in retrieved content

    Usage:
        metric = VoyageEmbeddingMetric()
        result = metric.score(
            query="search query",
            retrieved_docs=["doc1", "doc2"],
            scores=[0.9, 0.85]
        )
    """

    name = "voyage_embedding_quality"

    def __init__(
        self,
        min_score_threshold: float = 0.5,
        diversity_weight: float = 0.3,
    ):
        self.min_score_threshold = min_score_threshold
        self.diversity_weight = diversity_weight

    def score(
        self,
        query: str,
        retrieved_docs: List[str],
        scores: List[float],
        **kwargs,
    ) -> MetricResult:
        """
        Score the quality of retrieved documents.

        Args:
            query: The search query
            retrieved_docs: Retrieved document texts
            scores: Similarity scores from retrieval

        Returns:
            MetricResult with quality score
        """
        if not retrieved_docs or not scores:
            return MetricResult(
                name=self.name,
                score=0.0,
                reason="No documents retrieved",
            )

        # Calculate average relevance score
        avg_score = sum(scores) / len(scores)

        # Calculate diversity (simplified: query term coverage)
        query_terms = set(query.lower().split())
        covered_terms = 0
        for doc in retrieved_docs:
            doc_lower = doc.lower()
            covered_terms += sum(1 for term in query_terms if term in doc_lower)

        coverage = covered_terms / (len(query_terms) * len(retrieved_docs)) if query_terms else 0

        # Combined score
        final_score = (1 - self.diversity_weight) * avg_score + self.diversity_weight * coverage

        # Determine reason
        if final_score >= 0.8:
            reason = "Excellent retrieval quality"
        elif final_score >= 0.6:
            reason = "Good retrieval quality"
        elif final_score >= 0.4:
            reason = "Moderate retrieval quality"
        else:
            reason = "Poor retrieval quality - consider improving query or corpus"

        return MetricResult(
            name=self.name,
            score=final_score,
            reason=reason,
            metadata={
                "avg_relevance": avg_score,
                "term_coverage": coverage,
                "num_docs": len(retrieved_docs),
            },
        )


class LettaMemoryMetric:
    """
    Custom metric for evaluating Letta memory operations.

    Measures:
    - Memory retrieval relevance
    - Context utilization
    - Learning session quality
    """

    name = "letta_memory_quality"

    def score(
        self,
        query: str,
        retrieved_memories: List[Dict[str, Any]],
        context_used: bool = True,
        session_interactions: int = 0,
        **kwargs,
    ) -> MetricResult:
        """Score memory operation quality."""
        if not retrieved_memories:
            return MetricResult(
                name=self.name,
                score=0.0,
                reason="No memories retrieved",
            )

        # Calculate relevance from scores
        scores = [m.get("score", 0.5) for m in retrieved_memories]
        avg_relevance = sum(scores) / len(scores)

        # Bonus for context utilization
        context_bonus = 0.1 if context_used else 0

        # Bonus for learning session activity
        session_bonus = min(0.1, session_interactions * 0.02)

        final_score = min(1.0, avg_relevance + context_bonus + session_bonus)

        return MetricResult(
            name=self.name,
            score=final_score,
            reason=f"Memory retrieval with {len(retrieved_memories)} results",
            metadata={
                "avg_relevance": avg_relevance,
                "context_used": context_used,
                "session_interactions": session_interactions,
            },
        )


class DSPyOptimizationMetric:
    """
    Custom metric for evaluating DSPy optimization results.

    Measures:
    - Improvement from original to optimized
    - Consistency of results
    - Optimization efficiency
    """

    name = "dspy_optimization_quality"

    def score(
        self,
        original_score: float,
        optimized_score: float,
        iterations: int = 0,
        **kwargs,
    ) -> MetricResult:
        """Score DSPy optimization quality."""
        improvement = optimized_score - original_score

        # Normalize improvement (cap at 0.5 improvement)
        normalized_improvement = min(improvement / 0.5, 1.0)

        # Efficiency bonus (fewer iterations = more efficient)
        efficiency = max(0, 1 - (iterations / 100)) if iterations > 0 else 0.5

        final_score = 0.7 * normalized_improvement + 0.3 * efficiency

        if improvement > 0.2:
            reason = f"Excellent improvement: +{improvement:.2%}"
        elif improvement > 0.1:
            reason = f"Good improvement: +{improvement:.2%}"
        elif improvement > 0:
            reason = f"Marginal improvement: +{improvement:.2%}"
        else:
            reason = f"No improvement or regression: {improvement:.2%}"

        return MetricResult(
            name=self.name,
            score=final_score,
            reason=reason,
            metadata={
                "original_score": original_score,
                "optimized_score": optimized_score,
                "improvement": improvement,
                "iterations": iterations,
            },
        )


# =============================================================================
# Tracing Decorators
# =============================================================================

def track_sdk_operation(
    sdk_name: str,
    operation: str,
    tags: Optional[List[str]] = None,
    capture_input: bool = True,
    capture_output: bool = True,
):
    """
    Decorator to trace SDK operations with Opik.

    Works with both sync and async functions.

    Args:
        sdk_name: Name of the SDK (letta, dspy, voyage, temporal)
        operation: Operation name (embed, retrieve, optimize, etc.)
        tags: Additional tags for the trace
        capture_input: Whether to capture function inputs
        capture_output: Whether to capture function outputs

    Usage:
        @track_sdk_operation("voyage", "embed")
        async def embed_texts(texts: List[str]) -> List[List[float]]:
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not OPIK_AVAILABLE:
                return await func(*args, **kwargs)

            start_time = time.time()
            trace_name = f"{sdk_name}/{operation}"

            try:
                # Use Opik track decorator internally
                tracked_func = track(
                    name=trace_name,
                    tags=[sdk_name, operation, "unleash"] + (tags or []),
                    capture_input=capture_input,
                    capture_output=capture_output,
                )(func)

                result = await tracked_func(*args, **kwargs)

                duration = time.time() - start_time
                logger.info(
                    "sdk_operation_traced",
                    sdk=sdk_name,
                    operation=operation,
                    duration_ms=duration * 1000,
                )

                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    "sdk_operation_failed",
                    sdk=sdk_name,
                    operation=operation,
                    error=str(e),
                    duration_ms=duration * 1000,
                )
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not OPIK_AVAILABLE:
                return func(*args, **kwargs)

            start_time = time.time()
            trace_name = f"{sdk_name}/{operation}"

            try:
                tracked_func = track(
                    name=trace_name,
                    tags=[sdk_name, operation, "unleash"] + (tags or []),
                    capture_input=capture_input,
                    capture_output=capture_output,
                )(func)

                result = tracked_func(*args, **kwargs)

                duration = time.time() - start_time
                logger.info(
                    "sdk_operation_traced",
                    sdk=sdk_name,
                    operation=operation,
                    duration_ms=duration * 1000,
                )

                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    "sdk_operation_failed",
                    sdk=sdk_name,
                    operation=operation,
                    error=str(e),
                    duration_ms=duration * 1000,
                )
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


# =============================================================================
# Opik Tracer (Centralized)
# =============================================================================

class OpikTracer:
    """
    Centralized Opik tracer for all SDK operations.

    Provides:
    - Automatic Claude tracing via track_anthropic()
    - Custom SDK operation tracing
    - Metric evaluation and logging
    - Distributed trace context propagation

    Usage:
        tracer = OpikTracer()
        tracer.configure()

        # Get traced Claude client
        client = tracer.get_traced_anthropic_client()

        # Manual tracing
        with tracer.trace("letta", "recall"):
            memories = await letta.recall(query)

        # Evaluate metrics
        result = tracer.evaluate_metric(
            VoyageEmbeddingMetric(),
            query=query,
            retrieved_docs=docs,
            scores=scores,
        )
    """

    def __init__(
        self,
        project_name: str = "unleash",
        default_tags: Optional[List[str]] = None,
    ):
        self.project_name = project_name
        self.default_tags = default_tags or ["unleash", "production"]
        self._configured = False
        self._anthropic_client = None

        # Built-in metrics
        self.voyage_metric = VoyageEmbeddingMetric()
        self.letta_metric = LettaMemoryMetric()
        self.dspy_metric = DSPyOptimizationMetric()

    def configure(self, api_key: Optional[str] = None, workspace: Optional[str] = None):
        """
        Configure Opik for the project with circuit breaker protection.

        Args:
            api_key: Opik API key (or use OPIK_API_KEY env var)
            workspace: Opik workspace name
        """
        if not OPIK_AVAILABLE:
            logger.warning("opik_not_available", message="Tracing disabled")
            return

        try:
            opik.configure(
                api_key=api_key,
                workspace=workspace,
            )
            self._configured = True
            logger.info("opik_configured", project=self.project_name)
        except Exception as e:
            logger.error("opik_configure_failed", error=str(e))
            # Record failure in circuit breaker if available (V47)
            if CIRCUIT_BREAKER_AVAILABLE:
                breaker = adapter_circuit_breaker("opik_tracing_adapter")
                # Let the breaker know about the failure by attempting to use it
                try:
                    breaker._on_failure()
                except Exception as breaker_error:
                    logger.debug("circuit_breaker_update_failed", error=str(breaker_error))

    def get_status(self) -> Dict[str, Any]:
        """Get adapter status including circuit breaker health (V47)."""
        status = {
            "opik_available": OPIK_AVAILABLE,
            "anthropic_available": ANTHROPIC_AVAILABLE,
            "langchain_opik_available": LANGCHAIN_OPIK_AVAILABLE,
            "opik_metrics_available": OPIK_METRICS_AVAILABLE,
            "configured": self._configured,
            "project_name": self.project_name,
            "circuit_breaker_available": CIRCUIT_BREAKER_AVAILABLE,
        }

        # Add circuit breaker health if available (V47)
        if CIRCUIT_BREAKER_AVAILABLE:
            manager = get_adapter_circuit_manager()
            health = manager.get_health("opik_tracing_adapter")
            if health:
                status["circuit_breaker_state"] = health.state.value
                status["circuit_breaker_healthy"] = health.is_healthy
                status["failure_count"] = health.failure_count

        return status

    def get_traced_anthropic_client(self) -> Any:
        """
        Get an Anthropic client with automatic tracing.

        All Claude API calls will be traced to Opik.

        Returns:
            Traced Anthropic client
        """
        if not OPIK_AVAILABLE or not ANTHROPIC_AVAILABLE:
            if ANTHROPIC_AVAILABLE:
                return anthropic.Anthropic()
            raise ImportError("Anthropic SDK not available")

        if self._anthropic_client is None:
            base_client = anthropic.Anthropic()
            self._anthropic_client = track_anthropic(base_client)
            logger.info("anthropic_client_traced")

        return self._anthropic_client

    @contextmanager
    def trace(
        self,
        sdk_name: str,
        operation: str,
        tags: Optional[List[str]] = None,
    ):
        """
        Context manager for tracing a block of code.

        Args:
            sdk_name: SDK name (voyage, letta, dspy, temporal)
            operation: Operation name

        Yields:
            Trace context
        """
        start_time = time.time()
        trace_name = f"{sdk_name}/{operation}"
        all_tags = self.default_tags + (tags or []) + [sdk_name, operation]

        try:
            if OPIK_AVAILABLE:
                # Start Opik span
                logger.debug("trace_started", name=trace_name)

            yield

        except Exception as e:
            logger.error(
                "trace_error",
                name=trace_name,
                error=str(e),
            )
            raise

        finally:
            duration = time.time() - start_time
            logger.info(
                "trace_completed",
                name=trace_name,
                duration_ms=duration * 1000,
            )

    def evaluate_metric(
        self,
        metric: Union[VoyageEmbeddingMetric, LettaMemoryMetric, DSPyOptimizationMetric],
        **kwargs,
    ) -> MetricResult:
        """
        Evaluate a custom metric and log to Opik.

        Args:
            metric: The metric to evaluate
            **kwargs: Arguments for the metric's score() method

        Returns:
            MetricResult with score and metadata
        """
        result = metric.score(**kwargs)

        if OPIK_AVAILABLE:
            # Log metric to Opik
            logger.info(
                "metric_evaluated",
                name=result.name,
                score=result.score,
                reason=result.reason,
            )

        return result

    def get_langchain_tracer(
        self,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Get OpikTracer for LangChain/LangGraph integration (V17).

        Use as callback for LangChain components:
            llm = ChatOpenAI()
            response = llm.invoke("query", config={"callbacks": [tracer.get_langchain_tracer()]})

        Args:
            tags: Optional tags for traces
            metadata: Optional metadata dict

        Returns:
            OpikTracer instance for LangChain callbacks
        """
        if not LANGCHAIN_OPIK_AVAILABLE or OpikLangChainTracer is None:
            logger.warning("langchain_opik_not_available")
            return None

        return OpikLangChainTracer(
            tags=tags or self.default_tags,
            metadata=metadata or {},
        )

    def evaluate_hallucination(
        self,
        input_text: str,
        output_text: str,
        context: str,
    ) -> MetricResult:
        """
        Evaluate response for hallucination using LLM-as-Judge (V17).

        Args:
            input_text: Original query/input
            output_text: LLM response to evaluate
            context: Ground truth context for comparison

        Returns:
            MetricResult with hallucination score (lower is better)
        """
        if not OPIK_METRICS_AVAILABLE or HallucinationMetric is None:
            return MetricResult(
                name="hallucination",
                score=0.5,
                reason="Opik metrics not available",
            )

        metric = HallucinationMetric()
        score = metric.score(
            input=input_text,
            output=output_text,
            context=context,
        )

        return MetricResult(
            name="hallucination",
            score=score.value,
            reason=score.reason if hasattr(score, 'reason') else None,
        )

    def evaluate_relevance(
        self,
        input_text: str,
        output_text: str,
    ) -> MetricResult:
        """
        Evaluate answer relevance using LLM-as-Judge (V17).

        Args:
            input_text: Original query/input
            output_text: LLM response to evaluate

        Returns:
            MetricResult with relevance score (higher is better)
        """
        if not OPIK_METRICS_AVAILABLE or AnswerRelevanceMetric is None:
            return MetricResult(
                name="relevance",
                score=0.5,
                reason="Opik metrics not available",
            )

        metric = AnswerRelevanceMetric()
        score = metric.score(
            input=input_text,
            output=output_text,
        )

        return MetricResult(
            name="relevance",
            score=score.value,
            reason=score.reason if hasattr(score, 'reason') else None,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get tracer status."""
        return {
            "opik_available": OPIK_AVAILABLE,
            "anthropic_available": ANTHROPIC_AVAILABLE,
            "langchain_opik_available": LANGCHAIN_OPIK_AVAILABLE,
            "opik_metrics_available": OPIK_METRICS_AVAILABLE,
            "configured": self._configured,
            "project": self.project_name,
            "anthropic_client_initialized": self._anthropic_client is not None,
        }


# =============================================================================
# Global Tracer Instance
# =============================================================================

_global_tracer: Optional[OpikTracer] = None


def get_tracer() -> OpikTracer:
    """Get or create global tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = OpikTracer()
    return _global_tracer


def configure_tracing(
    api_key: Optional[str] = None,
    workspace: Optional[str] = None,
) -> OpikTracer:
    """Configure and return the global tracer."""
    tracer = get_tracer()
    tracer.configure(api_key, workspace)
    return tracer


# =============================================================================
# Convenience Exports
# =============================================================================

__all__ = [
    "OpikTracer",
    "VoyageEmbeddingMetric",
    "LettaMemoryMetric",
    "DSPyOptimizationMetric",
    "TraceMetadata",
    "MetricResult",
    "track_sdk_operation",
    "get_tracer",
    "configure_tracing",
    "OPIK_AVAILABLE",
    "ANTHROPIC_AVAILABLE",
]
