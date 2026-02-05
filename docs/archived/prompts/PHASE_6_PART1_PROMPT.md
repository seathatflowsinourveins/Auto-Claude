# Phase 6 Part 1: Observability Layer (Tracing & Monitoring)

## Overview
Layer 5 of V33 Architecture - Observability Layer
Part 1 covers: langfuse, opik, arize-phoenix

## Pre-Flight
```bash
python -c "from core.structured import StructuredOutputFactory; print('Phase 5 OK')"
```

## Install Dependencies
```bash
pip install langfuse opik arize-phoenix opentelemetry-sdk opentelemetry-api opentelemetry-exporter-otlp
```

## Step 1: Create Directory
```bash
mkdir -p core/observability
```

## Step 2: Create Langfuse Tracer

Create `core/observability/langfuse_tracer.py`:

```python
"""
Langfuse Tracer - LLM Observability and Tracing
Provides comprehensive tracing for LLM generations with cost tracking.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, Generator, List, Optional, TypeVar, Union
from uuid import uuid4

from pydantic import BaseModel, Field

# Conditional import for langfuse
try:
    from langfuse import Langfuse
    from langfuse.client import StatefulSpanClient, StatefulTraceClient
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None
    StatefulSpanClient = None
    StatefulTraceClient = None


class TraceLevel(str, Enum):
    """Trace verbosity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class FeedbackType(str, Enum):
    """Types of feedback for traces."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"
    COMMENT = "comment"
    CORRECTION = "correction"


class GenerationMetadata(BaseModel):
    """Metadata for a generation trace."""
    model: str = Field(..., description="Model identifier")
    prompt_tokens: int = Field(0, ge=0)
    completion_tokens: int = Field(0, ge=0)
    total_tokens: int = Field(0, ge=0)
    latency_ms: float = Field(0, ge=0)
    cost_usd: Optional[float] = Field(None, ge=0)
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    
    class Config:
        extra = "allow"


class TraceMetadata(BaseModel):
    """Metadata for a trace span."""
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    span_id: str = Field(default_factory=lambda: str(uuid4()))
    parent_span_id: Optional[str] = None
    name: str = Field(..., description="Span name")
    level: TraceLevel = TraceLevel.INFO
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"


class FeedbackRecord(BaseModel):
    """Feedback record for a trace."""
    trace_id: str
    feedback_type: FeedbackType
    value: Union[int, float, str, bool]
    comment: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


@dataclass
class LangfuseConfig:
    """Configuration for Langfuse client."""
    public_key: Optional[str] = None
    secret_key: Optional[str] = None
    host: str = "https://cloud.langfuse.com"
    release: Optional[str] = None
    debug: bool = False
    threads: int = 1
    flush_at: int = 15
    flush_interval: float = 0.5
    max_retries: int = 3
    timeout: int = 20
    enabled: bool = True
    
    # Cost configuration per model (USD per 1K tokens)
    model_costs: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
    })


class LangfuseTracer:
    """
    Langfuse tracer for LLM observability.
    
    Provides tracing, cost tracking, and feedback collection for LLM operations.
    Integrates with the LLM Gateway for automatic instrumentation.
    """
    
    def __init__(self, config: Optional[LangfuseConfig] = None):
        """Initialize the Langfuse tracer."""
        self.config = config or LangfuseConfig()
        self._client: Optional[Langfuse] = None
        self._active_traces: Dict[str, Any] = {}
        self._active_spans: Dict[str, Any] = {}
        self._generation_count = 0
        self._total_cost = 0.0
        
        if self.config.enabled and LANGFUSE_AVAILABLE:
            self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the Langfuse client."""
        if not LANGFUSE_AVAILABLE:
            raise ImportError("langfuse package not installed. Run: pip install langfuse")
        
        self._client = Langfuse(
            public_key=self.config.public_key,
            secret_key=self.config.secret_key,
            host=self.config.host,
            release=self.config.release,
            debug=self.config.debug,
            threads=self.config.threads,
            flush_at=self.config.flush_at,
            flush_interval=self.config.flush_interval,
            max_retries=self.config.max_retries,
            timeout=self.config.timeout,
        )
    
    @property
    def client(self) -> Optional[Langfuse]:
        """Get the Langfuse client."""
        return self._client
    
    @property
    def is_enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self.config.enabled and self._client is not None
    
    def calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """
        Calculate the cost of a generation.
        
        Args:
            model: Model identifier
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            
        Returns:
            Cost in USD
        """
        # Normalize model name for lookup
        model_lower = model.lower()
        model_key = None
        
        for key in self.config.model_costs:
            if key.lower() in model_lower or model_lower in key.lower():
                model_key = key
                break
        
        if not model_key:
            # Default cost estimation
            return (prompt_tokens * 0.001 + completion_tokens * 0.002) / 1000
        
        costs = self.config.model_costs[model_key]
        input_cost = (prompt_tokens / 1000) * costs.get("input", 0.001)
        output_cost = (completion_tokens / 1000) * costs.get("output", 0.002)
        
        return input_cost + output_cost
    
    def trace_generation(
        self,
        name: str,
        model: str,
        input_messages: List[Dict[str, Any]],
        output: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        level: TraceLevel = TraceLevel.INFO,
        **kwargs: Any,
    ) -> GenerationMetadata:
        """
        Trace an LLM generation.
        
        Args:
            name: Generation name/identifier
            model: Model used for generation
            input_messages: Input messages/prompt
            output: Generated output
            prompt_tokens: Input token count
            completion_tokens: Output token count
            latency_ms: Generation latency in milliseconds
            trace_id: Optional trace ID for grouping
            parent_span_id: Optional parent span ID
            user_id: Optional user identifier
            session_id: Optional session identifier
            metadata: Additional metadata
            tags: Optional tags for filtering
            level: Trace level
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationMetadata with cost and token information
        """
        total_tokens = prompt_tokens + completion_tokens
        cost = self.calculate_cost(model, prompt_tokens, completion_tokens)
        
        self._generation_count += 1
        self._total_cost += cost
        
        gen_metadata = GenerationMetadata(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            **kwargs,
        )
        
        if not self.is_enabled:
            return gen_metadata
        
        # Create or get trace
        trace_id = trace_id or str(uuid4())
        
        if trace_id not in self._active_traces:
            trace = self._client.trace(
                id=trace_id,
                name=name,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {},
                tags=tags or [],
            )
            self._active_traces[trace_id] = trace
        else:
            trace = self._active_traces[trace_id]
        
        # Create generation
        generation = trace.generation(
            name=name,
            model=model,
            input=input_messages,
            output=output,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            metadata={
                **(metadata or {}),
                "cost_usd": cost,
                "latency_ms": latency_ms,
                "level": level.value,
                **kwargs,
            },
        )
        
        return gen_metadata
    
    @contextmanager
    def trace_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        level: TraceLevel = TraceLevel.INFO,
    ) -> Generator[TraceMetadata, None, None]:
        """
        Context manager for tracing a span.
        
        Args:
            name: Span name
            trace_id: Optional trace ID
            parent_span_id: Optional parent span ID
            metadata: Additional metadata
            tags: Tags for filtering
            level: Trace level
            
        Yields:
            TraceMetadata for the span
        """
        trace_id = trace_id or str(uuid4())
        span_id = str(uuid4())
        start_time = datetime.utcnow()
        
        trace_meta = TraceMetadata(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=name,
            level=level,
            start_time=start_time,
            tags=tags or [],
            metadata=metadata or {},
        )
        
        span = None
        if self.is_enabled:
            # Get or create trace
            if trace_id not in self._active_traces:
                trace = self._client.trace(
                    id=trace_id,
                    name=name,
                    metadata=metadata or {},
                    tags=tags or [],
                )
                self._active_traces[trace_id] = trace
            else:
                trace = self._active_traces[trace_id]
            
            # Create span
            span = trace.span(
                id=span_id,
                name=name,
                metadata=metadata or {},
            )
            self._active_spans[span_id] = span
        
        try:
            yield trace_meta
        except Exception as e:
            trace_meta.metadata["error"] = str(e)
            trace_meta.level = TraceLevel.ERROR
            raise
        finally:
            end_time = datetime.utcnow()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            trace_meta.end_time = end_time
            trace_meta.duration_ms = duration_ms
            
            if span:
                span.end(metadata={"duration_ms": duration_ms})
                self._active_spans.pop(span_id, None)
    
    def record_feedback(
        self,
        trace_id: str,
        feedback_type: FeedbackType,
        value: Union[int, float, str, bool],
        comment: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> FeedbackRecord:
        """
        Record feedback for a trace.
        
        Args:
            trace_id: Trace identifier
            feedback_type: Type of feedback
            value: Feedback value
            comment: Optional comment
            user_id: Optional user identifier
            
        Returns:
            FeedbackRecord
        """
        record = FeedbackRecord(
            trace_id=trace_id,
            feedback_type=feedback_type,
            value=value,
            comment=comment,
            user_id=user_id,
        )
        
        if self.is_enabled:
            # Map feedback type to Langfuse score
            score_name = feedback_type.value
            
            if feedback_type == FeedbackType.THUMBS_UP:
                score_value = 1
            elif feedback_type == FeedbackType.THUMBS_DOWN:
                score_value = 0
            elif feedback_type == FeedbackType.RATING:
                score_value = float(value) if isinstance(value, (int, float)) else 0.5
            else:
                score_value = 1 if value else 0
            
            self._client.score(
                trace_id=trace_id,
                name=score_name,
                value=score_value,
                comment=comment,
            )
        
        return record
    
    async def trace_generation_async(
        self,
        name: str,
        model: str,
        input_messages: List[Dict[str, Any]],
        output: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        **kwargs: Any,
    ) -> GenerationMetadata:
        """Async version of trace_generation."""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.trace_generation(
                name=name,
                model=model,
                input_messages=input_messages,
                output=output,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                **kwargs,
            ),
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracing statistics."""
        return {
            "generation_count": self._generation_count,
            "total_cost_usd": round(self._total_cost, 6),
            "active_traces": len(self._active_traces),
            "active_spans": len(self._active_spans),
            "enabled": self.is_enabled,
        }
    
    def flush(self) -> None:
        """Flush pending traces to Langfuse."""
        if self._client:
            self._client.flush()
    
    def shutdown(self) -> None:
        """Shutdown the tracer and flush pending data."""
        self.flush()
        self._active_traces.clear()
        self._active_spans.clear()
        if self._client:
            self._client.shutdown()
            self._client = None


# Factory function for easy instantiation
def create_langfuse_tracer(
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    **kwargs: Any,
) -> LangfuseTracer:
    """
    Create a LangfuseTracer instance.
    
    Args:
        public_key: Langfuse public key
        secret_key: Langfuse secret key
        **kwargs: Additional configuration options
        
    Returns:
        Configured LangfuseTracer
    """
    config = LangfuseConfig(
        public_key=public_key,
        secret_key=secret_key,
        **kwargs,
    )
    return LangfuseTracer(config)


__all__ = [
    "LangfuseConfig",
    "LangfuseTracer",
    "TraceLevel",
    "FeedbackType",
    "GenerationMetadata",
    "TraceMetadata",
    "FeedbackRecord",
    "create_langfuse_tracer",
]
```

## Step 3: Create Opik Evaluator

Create `core/observability/opik_evaluator.py`:

```python
"""
Opik Evaluator - LLM Evaluation and Experimentation
Provides metrics, evaluation, and A/B testing capabilities.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
from uuid import uuid4

from pydantic import BaseModel, Field, validator

# Conditional import for opik
try:
    import opik
    from opik import Opik
    from opik.evaluation import evaluate
    from opik.evaluation.metrics import base_metric, Hallucination, AnswerRelevance
    OPIK_AVAILABLE = True
except ImportError:
    OPIK_AVAILABLE = False
    opik = None
    Opik = None


class MetricType(str, Enum):
    """Types of evaluation metrics."""
    HALLUCINATION = "hallucination"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    FLUENCY = "fluency"
    ACCURACY = "accuracy"
    TOXICITY = "toxicity"
    CUSTOM = "custom"


class ExperimentStatus(str, Enum):
    """Status of an experiment."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvaluationResult(BaseModel):
    """Result of a single evaluation."""
    eval_id: str = Field(default_factory=lambda: str(uuid4()))
    metric_name: str
    metric_type: MetricType
    score: float = Field(..., ge=0.0, le=1.0)
    passed: bool = True
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @validator("passed", pre=True, always=True)
    def compute_passed(cls, v, values):
        threshold = values.get("metadata", {}).get("threshold", 0.5)
        return values.get("score", 0) >= threshold


class BatchEvaluationResult(BaseModel):
    """Result of batch evaluation."""
    batch_id: str = Field(default_factory=lambda: str(uuid4()))
    results: List[EvaluationResult] = Field(default_factory=list)
    total_count: int = 0
    passed_count: int = 0
    failed_count: int = 0
    average_score: float = 0.0
    min_score: float = 1.0
    max_score: float = 0.0
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def update_statistics(self) -> None:
        """Update aggregate statistics."""
        if not self.results:
            return
        
        self.total_count = len(self.results)
        self.passed_count = sum(1 for r in self.results if r.passed)
        self.failed_count = self.total_count - self.passed_count
        
        scores = [r.score for r in self.results]
        self.average_score = sum(scores) / len(scores) if scores else 0.0
        self.min_score = min(scores) if scores else 0.0
        self.max_score = max(scores) if scores else 0.0


class ExperimentConfig(BaseModel):
    """Configuration for an A/B experiment."""
    experiment_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: Optional[str] = None
    variant_a: str = Field(..., description="Control variant identifier")
    variant_b: str = Field(..., description="Treatment variant identifier")
    metrics: List[MetricType] = Field(default_factory=list)
    sample_size: int = Field(100, ge=1)
    confidence_level: float = Field(0.95, ge=0.5, le=0.99)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ExperimentResult(BaseModel):
    """Result of an A/B experiment comparison."""
    experiment_id: str
    variant_a_results: BatchEvaluationResult
    variant_b_results: BatchEvaluationResult
    winner: Optional[str] = None
    confidence: float = 0.0
    p_value: float = 1.0
    effect_size: float = 0.0
    statistically_significant: bool = False
    recommendation: str = ""


class MetricDefinition(BaseModel):
    """Definition of a custom metric."""
    name: str
    metric_type: MetricType
    description: Optional[str] = None
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    weight: float = Field(1.0, ge=0.0)
    evaluator_prompt: Optional[str] = None
    aggregation: str = Field("mean", pattern="^(mean|median|min|max|sum)$")


@dataclass
class OpikConfig:
    """Configuration for Opik client."""
    api_key: Optional[str] = None
    workspace: Optional[str] = None
    host: str = "https://www.comet.com/opik/api"
    project_name: str = "default"
    enabled: bool = True
    cache_results: bool = True
    cache_ttl_seconds: int = 3600
    max_workers: int = 4
    timeout: int = 30
    retry_attempts: int = 3
    default_model: str = "gpt-4o"


class OpikEvaluator:
    """
    Opik evaluator for LLM evaluation and experimentation.
    
    Provides metrics evaluation, batch processing, and A/B testing
    capabilities for LLM outputs.
    """
    
    def __init__(self, config: Optional[OpikConfig] = None):
        """Initialize the Opik evaluator."""
        self.config = config or OpikConfig()
        self._client: Optional[Opik] = None
        self._metrics: Dict[str, MetricDefinition] = {}
        self._experiments: Dict[str, ExperimentConfig] = {}
        self._results_cache: Dict[str, EvaluationResult] = {}
        self._evaluation_count = 0
        
        if self.config.enabled and OPIK_AVAILABLE:
            self._initialize_client()
        
        self._register_default_metrics()
    
    def _initialize_client(self) -> None:
        """Initialize the Opik client."""
        if not OPIK_AVAILABLE:
            raise ImportError("opik package not installed. Run: pip install opik")
        
        self._client = Opik(
            api_key=self.config.api_key,
            workspace=self.config.workspace,
            host=self.config.host,
        )
    
    def _register_default_metrics(self) -> None:
        """Register default evaluation metrics."""
        default_metrics = [
            MetricDefinition(
                name="hallucination",
                metric_type=MetricType.HALLUCINATION,
                description="Detects hallucinated content not grounded in context",
                threshold=0.3,
                weight=1.5,
            ),
            MetricDefinition(
                name="relevance",
                metric_type=MetricType.RELEVANCE,
                description="Measures answer relevance to the question",
                threshold=0.7,
                weight=1.0,
            ),
            MetricDefinition(
                name="coherence",
                metric_type=MetricType.COHERENCE,
                description="Evaluates logical flow and consistency",
                threshold=0.6,
                weight=1.0,
            ),
            MetricDefinition(
                name="fluency",
                metric_type=MetricType.FLUENCY,
                description="Assesses grammatical correctness and readability",
                threshold=0.7,
                weight=0.8,
            ),
        ]
        
        for metric in default_metrics:
            self._metrics[metric.name] = metric
    
    @property
    def client(self) -> Optional[Opik]:
        """Get the Opik client."""
        return self._client
    
    @property
    def is_enabled(self) -> bool:
        """Check if evaluation is enabled."""
        return self.config.enabled
    
    def create_metric(
        self,
        name: str,
        metric_type: MetricType = MetricType.CUSTOM,
        description: Optional[str] = None,
        threshold: float = 0.5,
        weight: float = 1.0,
        evaluator_prompt: Optional[str] = None,
        aggregation: str = "mean",
    ) -> MetricDefinition:
        """
        Create a custom evaluation metric.
        
        Args:
            name: Metric name
            metric_type: Type of metric
            description: Metric description
            threshold: Pass/fail threshold
            weight: Weight for aggregation
            evaluator_prompt: Custom LLM prompt for evaluation
            aggregation: Aggregation method for batch evaluation
            
        Returns:
            MetricDefinition
        """
        metric = MetricDefinition(
            name=name,
            metric_type=metric_type,
            description=description,
            threshold=threshold,
            weight=weight,
            evaluator_prompt=evaluator_prompt,
            aggregation=aggregation,
        )
        
        self._metrics[name] = metric
        return metric
    
    def evaluate_single(
        self,
        input_text: str,
        output_text: str,
        metric_name: str,
        context: Optional[str] = None,
        expected_output: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """
        Evaluate a single input-output pair.
        
        Args:
            input_text: Input/question text
            output_text: Generated output
            metric_name: Name of metric to use
            context: Optional context for grounding
            expected_output: Optional expected/reference output
            metadata: Additional metadata
            
        Returns:
            EvaluationResult
        """
        if metric_name not in self._metrics:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        metric = self._metrics[metric_name]
        self._evaluation_count += 1
        
        # Check cache
        cache_key = f"{metric_name}:{hash(input_text + output_text)}"
        if self.config.cache_results and cache_key in self._results_cache:
            return self._results_cache[cache_key]
        
        # Perform evaluation based on metric type
        score, reasoning = self._compute_metric_score(
            metric=metric,
            input_text=input_text,
            output_text=output_text,
            context=context,
            expected_output=expected_output,
        )
        
        result = EvaluationResult(
            metric_name=metric_name,
            metric_type=metric.metric_type,
            score=score,
            reasoning=reasoning,
            metadata={
                **(metadata or {}),
                "threshold": metric.threshold,
                "weight": metric.weight,
            },
        )
        
        # Update cache
        if self.config.cache_results:
            self._results_cache[cache_key] = result
        
        return result
    
    def _compute_metric_score(
        self,
        metric: MetricDefinition,
        input_text: str,
        output_text: str,
        context: Optional[str] = None,
        expected_output: Optional[str] = None,
    ) -> Tuple[float, Optional[str]]:
        """Compute the score for a metric."""
        # Use Opik built-in metrics if available
        if self.is_enabled and OPIK_AVAILABLE and self._client:
            try:
                if metric.metric_type == MetricType.HALLUCINATION:
                    opik_metric = Hallucination()
                    result = opik_metric.score(
                        input=input_text,
                        output=output_text,
                        context=[context] if context else [],
                    )
                    # Invert hallucination score (lower is better)
                    return 1.0 - result.value, result.reason
                
                elif metric.metric_type == MetricType.RELEVANCE:
                    opik_metric = AnswerRelevance()
                    result = opik_metric.score(
                        input=input_text,
                        output=output_text,
                        context=[context] if context else [],
                    )
                    return result.value, result.reason
                
            except Exception as e:
                # Fall through to heuristic evaluation
                pass
        
        # Heuristic evaluation fallback
        return self._heuristic_evaluation(
            metric=metric,
            input_text=input_text,
            output_text=output_text,
            context=context,
            expected_output=expected_output,
        )
    
    def _heuristic_evaluation(
        self,
        metric: MetricDefinition,
        input_text: str,
        output_text: str,
        context: Optional[str] = None,
        expected_output: Optional[str] = None,
    ) -> Tuple[float, str]:
        """Fallback heuristic evaluation."""
        score = 0.5
        reasoning = "Heuristic evaluation"
        
        if metric.metric_type == MetricType.HALLUCINATION:
            # Check if output contains information not in context
            if context:
                output_words = set(output_text.lower().split())
                context_words = set(context.lower().split())
                input_words = set(input_text.lower().split())
                
                grounded = len(output_words & (context_words | input_words))
                total = len(output_words) if output_words else 1
                score = grounded / total
                reasoning = f"Grounding ratio: {grounded}/{total}"
            else:
                score = 0.8
                reasoning = "No context provided for hallucination check"
        
        elif metric.metric_type == MetricType.RELEVANCE:
            # Check keyword overlap with input
            input_words = set(input_text.lower().split())
            output_words = set(output_text.lower().split())
            
            overlap = len(input_words & output_words)
            total = len(input_words) if input_words else 1
            score = min(1.0, overlap / total + 0.3)  # Base relevance
            reasoning = f"Keyword overlap: {overlap}/{total}"
        
        elif metric.metric_type == MetricType.COHERENCE:
            # Basic coherence check
            sentences = output_text.split(".")
            score = min(1.0, 0.5 + len(sentences) * 0.1)
            reasoning = f"Sentence count: {len(sentences)}"
        
        elif metric.metric_type == MetricType.FLUENCY:
            # Basic fluency check
            words = output_text.split()
            avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
            score = 0.8 if 3 < avg_word_len < 10 else 0.5
            reasoning = f"Average word length: {avg_word_len:.1f}"
        
        elif metric.metric_type == MetricType.ACCURACY:
            # Compare with expected output if available
            if expected_output:
                expected_words = set(expected_output.lower().split())
                output_words = set(output_text.lower().split())
                
                overlap = len(expected_words & output_words)
                total = len(expected_words) if expected_words else 1
                score = overlap / total
                reasoning = f"Match with expected: {overlap}/{total}"
            else:
                score = 0.5
                reasoning = "No expected output for accuracy check"
        
        return max(0.0, min(1.0, score)), reasoning
    
    async def evaluate_single_async(
        self,
        input_text: str,
        output_text: str,
        metric_name: str,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Async version of evaluate_single."""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.evaluate_single(
                input_text=input_text,
                output_text=output_text,
                metric_name=metric_name,
                **kwargs,
            ),
        )
    
    def evaluate_batch(
        self,
        samples: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None,
        context_key: str = "context",
        input_key: str = "input",
        output_key: str = "output",
        expected_key: str = "expected",
    ) -> BatchEvaluationResult:
        """
        Evaluate a batch of samples.
        
        Args:
            samples: List of sample dictionaries
            metrics: Metrics to evaluate (all if None)
            context_key: Key for context in samples
            input_key: Key for input in samples
            output_key: Key for output in samples
            expected_key: Key for expected output in samples
            
        Returns:
            BatchEvaluationResult
        """
        import time
        start_time = time.time()
        
        metrics = metrics or list(self._metrics.keys())
        batch_result = BatchEvaluationResult()
        
        for sample in samples:
            input_text = sample.get(input_key, "")
            output_text = sample.get(output_key, "")
            context = sample.get(context_key)
            expected = sample.get(expected_key)
            
            for metric_name in metrics:
                try:
                    result = self.evaluate_single(
                        input_text=input_text,
                        output_text=output_text,
                        metric_name=metric_name,
                        context=context,
                        expected_output=expected,
                        metadata={"sample_id": sample.get("id", str(uuid4()))},
                    )
                    batch_result.results.append(result)
                except Exception as e:
                    # Record failed evaluation
                    batch_result.results.append(EvaluationResult(
                        metric_name=metric_name,
                        metric_type=MetricType.CUSTOM,
                        score=0.0,
                        passed=False,
                        reasoning=f"Evaluation error: {str(e)}",
                    ))
        
        batch_result.duration_ms = (time.time() - start_time) * 1000
        batch_result.update_statistics()
        
        return batch_result
    
    async def evaluate_batch_async(
        self,
        samples: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> BatchEvaluationResult:
        """Async version of evaluate_batch."""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.evaluate_batch(samples=samples, **kwargs),
        )
    
    def compare_experiments(
        self,
        config: ExperimentConfig,
        variant_a_samples: List[Dict[str, Any]],
        variant_b_samples: List[Dict[str, Any]],
    ) -> ExperimentResult:
        """
        Compare two variants in an A/B experiment.
        
        Args:
            config: Experiment configuration
            variant_a_samples: Samples from variant A (control)
            variant_b_samples: Samples from variant B (treatment)
            
        Returns:
            ExperimentResult with statistical analysis
        """
        self._experiments[config.experiment_id] = config
        
        # Evaluate both variants
        metrics = [m.value for m in config.metrics] if config.metrics else None
        
        variant_a_results = self.evaluate_batch(
            samples=variant_a_samples[:config.sample_size],
            metrics=metrics,
        )
        
        variant_b_results = self.evaluate_batch(
            samples=variant_b_samples[:config.sample_size],
            metrics=metrics,
        )
        
        # Statistical analysis
        effect_size = variant_b_results.average_score - variant_a_results.average_score
        
        # Compute p-value using simple t-test approximation
        a_scores = [r.score for r in variant_a_results.results]
        b_scores = [r.score for r in variant_b_results.results]
        
        p_value = self._compute_p_value(a_scores, b_scores)
        statistically_significant = p_value < (1 - config.confidence_level)
        
        # Determine winner
        winner = None
        if statistically_significant:
            if effect_size > 0:
                winner = config.variant_b
            elif effect_size < 0:
                winner = config.variant_a
        
        # Generate recommendation
        if winner:
            recommendation = f"Recommend {winner} with {config.confidence_level*100:.0f}% confidence"
        elif statistically_significant:
            recommendation = "Results are equivalent within significance threshold"
        else:
            recommendation = f"Insufficient evidence (p={p_value:.3f}). Consider larger sample size"
        
        return ExperimentResult(
            experiment_id=config.experiment_id,
            variant_a_results=variant_a_results,
            variant_b_results=variant_b_results,
            winner=winner,
            confidence=config.confidence_level,
            p_value=p_value,
            effect_size=effect_size,
            statistically_significant=statistically_significant,
            recommendation=recommendation,
        )
    
    def _compute_p_value(
        self,
        a_scores: List[float],
        b_scores: List[float],
    ) -> float:
        """Compute p-value using Welch's t-test approximation."""
        import math
        
        if not a_scores or not b_scores:
            return 1.0
        
        n_a, n_b = len(a_scores), len(b_scores)
        mean_a = sum(a_scores) / n_a
        mean_b = sum(b_scores) / n_b
        
        var_a = sum((x - mean_a) ** 2 for x in a_scores) / (n_a - 1) if n_a > 1 else 0.001
        var_b = sum((x - mean_b) ** 2 for x in b_scores) / (n_b - 1) if n_b > 1 else 0.001
        
        se = math.sqrt(var_a / n_a + var_b / n_b)
        if se == 0:
            return 1.0
        
        t_stat = abs(mean_a - mean_b) / se
        
        # Approximate p-value (simplified)
        df = ((var_a / n_a + var_b / n_b) ** 2 /
              ((var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)))
        
        # Simple approximation using normal distribution
        p_value = 2 * (1 - self._normal_cdf(t_stat))
        
        return max(0.0, min(1.0, p_value))
    
    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF."""
        import math
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def export_results(
        self,
        results: Union[EvaluationResult, BatchEvaluationResult, ExperimentResult],
        output_path: Union[str, Path],
        format: str = "json",
    ) -> Path:
        """
        Export evaluation results.
        
        Args:
            results: Results to export
            output_path: Output file path
            format: Export format (json, csv)
            
        Returns:
            Path to exported file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_path, "w") as f:
                json.dump(results.model_dump(), f, indent=2, default=str)
        
        elif format == "csv":
            import csv
            
            if isinstance(results, BatchEvaluationResult):
                rows = [r.model_dump() for r in results.results]
            elif isinstance(results, EvaluationResult):
                rows = [results.model_dump()]
            else:
                rows = [results.model_dump()]
            
            if rows:
                with open(output_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
        
        return output_path
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        return {
            "evaluation_count": self._evaluation_count,
            "registered_metrics": list(self._metrics.keys()),
            "active_experiments": list(self._experiments.keys()),
            "cache_size": len(self._results_cache),
            "enabled": self.is_enabled,
        }
    
    def clear_cache(self) -> None:
        """Clear the results cache."""
        self._results_cache.clear()


# Factory function
def create_opik_evaluator(
    api_key: Optional[str] = None,
    workspace: Optional[str] = None,
    **kwargs: Any,
) -> OpikEvaluator:
    """
    Create an OpikEvaluator instance.
    
    Args:
        api_key: Opik API key
        workspace: Opik workspace
        **kwargs: Additional configuration options
        
    Returns:
        Configured OpikEvaluator
    """
    config = OpikConfig(
        api_key=api_key,
        workspace=workspace,
        **kwargs,
    )
    return OpikEvaluator(config)


__all__ = [
    "OpikConfig",
    "OpikEvaluator",
    "MetricType",
    "MetricDefinition",
    "EvaluationResult",
    "BatchEvaluationResult",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentStatus",
    "create_opik_evaluator",
]
```

## Step 4: Create Phoenix Monitor

Create `core/observability/phoenix_monitor.py`:

```python
"""
Phoenix Monitor - Real-time LLM Monitoring with OpenTelemetry
Provides embedding monitoring, drift detection, and alerting.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field, validator

# Conditional imports for Phoenix and OpenTelemetry
try:
    import phoenix as px
    from phoenix.trace import SpanExporter
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    px = None
    SpanExporter = None

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    TracerProvider = None


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DriftType(str, Enum):
    """Types of drift detection."""
    EMBEDDING = "embedding"
    OUTPUT = "output"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    COST = "cost"


class MetricUnit(str, Enum):
    """Units for metrics."""
    COUNT = "count"
    MILLISECONDS = "ms"
    SECONDS = "s"
    PERCENTAGE = "percent"
    USD = "usd"
    TOKENS = "tokens"


class EmbeddingRecord(BaseModel):
    """Record for embedding data."""
    record_id: str = Field(default_factory=lambda: str(uuid4()))
    vector: List[float]
    text: Optional[str] = None
    label: Optional[str] = None
    prediction: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator("vector")
    def validate_vector(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Vector cannot be empty")
        return v


class DriftResult(BaseModel):
    """Result of drift detection."""
    drift_type: DriftType
    detected: bool = False
    score: float = Field(0.0, ge=0.0, le=1.0)
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    baseline_value: Optional[float] = None
    current_value: Optional[float] = None
    change_percentage: Optional[float] = None
    window_start: datetime = Field(default_factory=datetime.utcnow)
    window_end: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AlertConfig(BaseModel):
    """Configuration for an alert."""
    alert_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    metric_name: str
    condition: str = Field(..., description="Condition like 'gt', 'lt', 'eq', 'drift'")
    threshold: float
    severity: AlertSeverity = AlertSeverity.WARNING
    cooldown_seconds: int = Field(300, ge=0)
    enabled: bool = True
    callback: Optional[str] = None  # Callback function name
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AlertEvent(BaseModel):
    """An alert event."""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    alert_id: str
    alert_name: str
    severity: AlertSeverity
    message: str
    metric_value: float
    threshold: float
    triggered_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False


class MonitoringMetric(BaseModel):
    """A monitoring metric."""
    name: str
    value: float
    unit: MetricUnit
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class PhoenixConfig:
    """Configuration for Phoenix monitor."""
    host: str = "127.0.0.1"
    port: int = 6006
    project_name: str = "default"
    enabled: bool = True
    auto_start_server: bool = True
    
    # OpenTelemetry configuration
    otel_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "llm-observability"
    otel_enabled: bool = True
    
    # Drift detection configuration
    drift_window_hours: int = 24
    drift_threshold: float = 0.2
    drift_check_interval_seconds: int = 300
    
    # Embedding configuration
    embedding_dimension: int = 1536
    max_embeddings_stored: int = 10000
    
    # Alert configuration
    alert_cooldown_seconds: int = 300
    max_alerts_stored: int = 1000


class PhoenixMonitor:
    """
    Phoenix monitor for real-time LLM observability.
    
    Provides embedding monitoring, drift detection, and alerting
    with OpenTelemetry integration.
    """
    
    def __init__(self, config: Optional[PhoenixConfig] = None):
        """Initialize the Phoenix monitor."""
        self.config = config or PhoenixConfig()
        self._session: Optional[Any] = None
        self._tracer: Optional[Any] = None
        self._tracer_provider: Optional[Any] = None
        
        # Storage
        self._embeddings: List[EmbeddingRecord] = []
        self._metrics: Dict[str, List[MonitoringMetric]] = {}
        self._alerts: Dict[str, AlertConfig] = {}
        self._alert_events: List[AlertEvent] = []
        self._last_alert_times: Dict[str, datetime] = {}
        
        # Drift baselines
        self._baselines: Dict[str, Any] = {}
        
        # Background tasks
        self._running = False
        self._drift_check_thread: Optional[threading.Thread] = None
        self._alert_callbacks: Dict[str, Callable] = {}
        
        if self.config.enabled:
            self._initialize()
    
    def _initialize(self) -> None:
        """Initialize Phoenix and OpenTelemetry."""
        if self.config.otel_enabled and OTEL_AVAILABLE:
            self._setup_opentelemetry()
    
    def _setup_opentelemetry(self) -> None:
        """Set up OpenTelemetry tracing."""
        resource = Resource.create({
            "service.name": self.config.otel_service_name,
            "service.version": "1.0.0",
        })
        
        self._tracer_provider = TracerProvider(resource=resource)
        
        # Add OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=self.config.otel_endpoint,
            insecure=True,
        )
        span_processor = BatchSpanProcessor(otlp_exporter)
        self._tracer_provider.add_span_processor(span_processor)
        
        # Add Phoenix exporter if available
        if PHOENIX_AVAILABLE:
            phoenix_exporter = SpanExporter()
            phoenix_processor = BatchSpanProcessor(phoenix_exporter)
            self._tracer_provider.add_span_processor(phoenix_processor)
        
        trace.set_tracer_provider(self._tracer_provider)
        self._tracer = trace.get_tracer(self.config.otel_service_name)
    
    def start_server(self) -> str:
        """
        Start the Phoenix server.
        
        Returns:
            URL of the Phoenix UI
        """
        if not PHOENIX_AVAILABLE:
            raise ImportError("phoenix package not installed. Run: pip install arize-phoenix")
        
        self._session = px.launch_app(
            host=self.config.host,
            port=self.config.port,
        )
        
        return f"http://{self.config.host}:{self.config.port}"
    
    def log_embedding(
        self,
        vector: Union[List[float], np.ndarray],
        text: Optional[str] = None,
        label: Optional[str] = None,
        prediction: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingRecord:
        """
        Log an embedding for monitoring.
        
        Args:
            vector: Embedding vector
            text: Source text
            label: Ground truth label
            prediction: Model prediction
            metadata: Additional metadata
            
        Returns:
            EmbeddingRecord
        """
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        
        record = EmbeddingRecord(
            vector=vector,
            text=text,
            label=label,
            prediction=prediction,
            metadata=metadata or {},
        )
        
        self._embeddings.append(record)
        
        # Enforce storage limit
        if len(self._embeddings) > self.config.max_embeddings_stored:
            self._embeddings = self._embeddings[-self.config.max_embeddings_stored:]
        
        # Log to Phoenix if available
        if PHOENIX_AVAILABLE and self._session:
            try:
                # Phoenix embedding logging would go here
                pass
            except Exception:
                pass
        
        return record
    
    def log_metric(
        self,
        name: str,
        value: float,
        unit: MetricUnit = MetricUnit.COUNT,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MonitoringMetric:
        """
        Log a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Metric unit
            tags: Tags for filtering
            metadata: Additional metadata
            
        Returns:
            MonitoringMetric
        """
        metric = MonitoringMetric(
            name=name,
            value=value,
            unit=unit,
            tags=tags or {},
            metadata=metadata or {},
        )
        
        if name not in self._metrics:
            self._metrics[name] = []
        
        self._metrics[name].append(metric)
        
        # Enforce storage limit
        if len(self._metrics[name]) > 10000:
            self._metrics[name] = self._metrics[name][-10000:]
        
        # Check alerts
        self._check_metric_alerts(metric)
        
        # Create OpenTelemetry span
        if self._tracer:
            with self._tracer.start_as_current_span(f"metric.{name}") as span:
                span.set_attribute("metric.name", name)
                span.set_attribute("metric.value", value)
                span.set_attribute("metric.unit", unit.value)
                for tag_key, tag_value in (tags or {}).items():
                    span.set_attribute(f"tag.{tag_key}", tag_value)
        
        return metric
    
    def detect_drift(
        self,
        drift_type: DriftType = DriftType.EMBEDDING,
        window_hours: Optional[int] = None,
    ) -> DriftResult:
        """
        Detect drift in monitored data.
        
        Args:
            drift_type: Type of drift to detect
            window_hours: Time window for comparison
            
        Returns:
            DriftResult
        """
        window_hours = window_hours or self.config.drift_window_hours
        window_start = datetime.utcnow() - timedelta(hours=window_hours)
        
        if drift_type == DriftType.EMBEDDING:
            return self._detect_embedding_drift(window_start)
        elif drift_type == DriftType.LATENCY:
            return self._detect_metric_drift("latency", window_start)
        elif drift_type == DriftType.ERROR_RATE:
            return self._detect_metric_drift("error_rate", window_start)
        elif drift_type == DriftType.COST:
            return self._detect_metric_drift("cost", window_start)
        else:
            return DriftResult(
                drift_type=drift_type,
                detected=False,
                score=0.0,
                threshold=self.config.drift_threshold,
            )
    
    def _detect_embedding_drift(self, window_start: datetime) -> DriftResult:
        """Detect drift in embeddings."""
        # Split embeddings into baseline and current
        baseline = [e for e in self._embeddings if e.timestamp < window_start]
        current = [e for e in self._embeddings if e.timestamp >= window_start]
        
        if not baseline or not current:
            return DriftResult(
                drift_type=DriftType.EMBEDDING,
                detected=False,
                score=0.0,
                threshold=self.config.drift_threshold,
                window_start=window_start,
                window_end=datetime.utcnow(),
                metadata={"reason": "Insufficient data"},
            )
        
        # Compute centroid distance
        baseline_vectors = np.array([e.vector for e in baseline])
        current_vectors = np.array([e.vector for e in current])
        
        baseline_centroid = np.mean(baseline_vectors, axis=0)
        current_centroid = np.mean(current_vectors, axis=0)
        
        # Cosine distance
        dot_product = np.dot(baseline_centroid, current_centroid)
        norm_product = np.linalg.norm(baseline_centroid) * np.linalg.norm(current_centroid)
        
        if norm_product == 0:
            cosine_similarity = 0.0
        else:
            cosine_similarity = dot_product / norm_product
        
        drift_score = 1.0 - cosine_similarity
        detected = drift_score > self.config.drift_threshold
        
        result = DriftResult(
            drift_type=DriftType.EMBEDDING,
            detected=detected,
            score=float(drift_score),
            threshold=self.config.drift_threshold,
            window_start=window_start,
            window_end=datetime.utcnow(),
            metadata={
                "baseline_count": len(baseline),
                "current_count": len(current),
                "cosine_similarity": float(cosine_similarity),
            },
        )
        
        if detected:
            self._trigger_drift_alert(result)
        
        return result
    
    def _detect_metric_drift(
        self,
        metric_name: str,
        window_start: datetime,
    ) -> DriftResult:
        """Detect drift in a metric."""
        if metric_name not in self._metrics:
            return DriftResult(
                drift_type=DriftType.OUTPUT,
                detected=False,
                score=0.0,
                threshold=self.config.drift_threshold,
            )
        
        metrics = self._metrics[metric_name]
        baseline = [m for m in metrics if m.timestamp < window_start]
        current = [m for m in metrics if m.timestamp >= window_start]
        
        if not baseline or not current:
            return DriftResult(
                drift_type=DriftType.OUTPUT,
                detected=False,
                score=0.0,
                threshold=self.config.drift_threshold,
            )
        
        baseline_mean = sum(m.value for m in baseline) / len(baseline)
        current_mean = sum(m.value for m in current) / len(current)
        
        if baseline_mean == 0:
            change_pct = 1.0 if current_mean != 0 else 0.0
        else:
            change_pct = abs(current_mean - baseline_mean) / baseline_mean
        
        drift_score = min(1.0, change_pct)
        detected = drift_score > self.config.drift_threshold
        
        return DriftResult(
            drift_type=DriftType.OUTPUT,
            detected=detected,
            score=drift_score,
            threshold=self.config.drift_threshold,
            baseline_value=baseline_mean,
            current_value=current_mean,
            change_percentage=change_pct * 100,
            window_start=window_start,
            window_end=datetime.utcnow(),
        )
    
    def set_alert(
        self,
        name: str,
        metric_name: str,
        condition: str,
        threshold: float,
        severity: AlertSeverity = AlertSeverity.WARNING,
        cooldown_seconds: Optional[int] = None,
        callback: Optional[Callable[[AlertEvent], None]] = None,
    ) -> AlertConfig:
        """
        Set up an alert.
        
        Args:
            name: Alert name
            metric_name: Metric to monitor
            condition: Condition (gt, lt, eq, drift)
            threshold: Threshold value
            severity: Alert severity
            cooldown_seconds: Cooldown between alerts
            callback: Callback function for alerts
            
        Returns:
            AlertConfig
        """
        alert = AlertConfig(
            name=name,
            metric_name=metric_name,
            condition=condition,
            threshold=threshold,
            severity=severity,
            cooldown_seconds=cooldown_seconds or self.config.alert_cooldown_seconds,
        )
        
        self._alerts[alert.alert_id] = alert
        
        if callback:
            self._alert_callbacks[alert.alert_id] = callback
        
        return alert
    
    def _check_metric_alerts(self, metric: MonitoringMetric) -> None:
        """Check if a metric triggers any alerts."""
        for alert_id, alert in self._alerts.items():
            if not alert.enabled or alert.metric_name != metric.name:
                continue
            
            # Check cooldown
            last_alert_time = self._last_alert_times.get(alert_id)
            if last_alert_time:
                cooldown_end = last_alert_time + timedelta(seconds=alert.cooldown_seconds)
                if datetime.utcnow() < cooldown_end:
                    continue
            
            # Check condition
            triggered = False
            if alert.condition == "gt" and metric.value > alert.threshold:
                triggered = True
            elif alert.condition == "lt" and metric.value < alert.threshold:
                triggered = True
            elif alert.condition == "eq" and metric.value == alert.threshold:
                triggered = True
            
            if triggered:
                self._fire_alert(alert, metric)
    
    def _fire_alert(self, alert: AlertConfig, metric: MonitoringMetric) -> None:
        """Fire an alert."""
        event = AlertEvent(
            alert_id=alert.alert_id,
            alert_name=alert.name,
            severity=alert.severity,
            message=f"Alert '{alert.name}': {metric.name} {alert.condition} {alert.threshold} (value: {metric.value})",
            metric_value=metric.value,
            threshold=alert.threshold,
        )
        
        self._alert_events.append(event)
        self._last_alert_times[alert.alert_id] = datetime.utcnow()
        
        # Enforce storage limit
        if len(self._alert_events) > self.config.max_alerts_stored:
            self._alert_events = self._alert_events[-self.config.max_alerts_stored:]
        
        # Execute callback
        if alert.alert_id in self._alert_callbacks:
            try:
                self._alert_callbacks[alert.alert_id](event)
            except Exception:
                pass
    
    def _trigger_drift_alert(self, drift_result: DriftResult) -> None:
        """Trigger alert for detected drift."""
        event = AlertEvent(
            alert_id="drift_detection",
            alert_name=f"Drift Detected: {drift_result.drift_type.value}",
            severity=AlertSeverity.WARNING,
            message=f"Drift detected in {drift_result.drift_type.value}: score={drift_result.score:.3f}",
            metric_value=drift_result.score,
            threshold=drift_result.threshold,
        )
        self._alert_events.append(event)
    
    def get_metrics(
        self,
        metric_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> List[MonitoringMetric]:
        """
        Get metrics with optional filtering.
        
        Args:
            metric_name: Filter by metric name
            start_time: Filter by start time
            end_time: Filter by end time
            tags: Filter by tags
            
        Returns:
            List of matching metrics
        """
        results = []
        
        metrics_to_search = self._metrics.get(metric_name, []) if metric_name else \
            [m for metrics in self._metrics.values() for m in metrics]
        
        for metric in metrics_to_search:
            if start_time and metric.timestamp < start_time:
                continue
            if end_time and metric.timestamp > end_time:
                continue
            if tags:
                if not all(metric.tags.get(k) == v for k, v in tags.items()):
                    continue
            
            results.append(metric)
        
        return results
    
    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        acknowledged: Optional[bool] = None,
        limit: int = 100,
    ) -> List[AlertEvent]:
        """
        Get alert events with optional filtering.
        
        Args:
            severity: Filter by severity
            acknowledged: Filter by acknowledged status
            limit: Maximum number of events
            
        Returns:
            List of alert events
        """
        results = []
        
        for event in reversed(self._alert_events):
            if severity and event.severity != severity:
                continue
            if acknowledged is not None and event.acknowledged != acknowledged:
                continue
            
            results.append(event)
            
            if len(results) >= limit:
                break
        
        return results
    
    def acknowledge_alert(self, event_id: str) -> bool:
        """Acknowledge an alert event."""
        for event in self._alert_events:
            if event.event_id == event_id:
                event.acknowledged = True
                return True
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            "embeddings_stored": len(self._embeddings),
            "metric_names": list(self._metrics.keys()),
            "total_metrics": sum(len(m) for m in self._metrics.values()),
            "active_alerts": len(self._alerts),
            "alert_events": len(self._alert_events),
            "unacknowledged_alerts": sum(1 for e in self._alert_events if not e.acknowledged),
            "enabled": self.config.enabled,
            "otel_enabled": self.config.otel_enabled and OTEL_AVAILABLE,
            "phoenix_available": PHOENIX_AVAILABLE,
        }
    
    def close(self) -> None:
        """Shutdown the monitor."""
        self._running = False
        
        if self._drift_check_thread and self._drift_check_thread.is_alive():
            self._drift_check_thread.join(timeout=5)
        
        if self._tracer_provider:
            self._tracer_provider.shutdown()
        
        self._session = None


# Factory function
def create_phoenix_monitor(
    host: str = "127.0.0.1",
    port: int = 6006,
    otel_endpoint: Optional[str] = None,
    **kwargs: Any,
) -> PhoenixMonitor:
    """
    Create a PhoenixMonitor instance.
    
    Args:
        host: Phoenix server host
        port: Phoenix server port
        otel_endpoint: OpenTelemetry endpoint
        **kwargs: Additional configuration options
        
    Returns:
        Configured PhoenixMonitor
    """
    config = PhoenixConfig(
        host=host,
        port=port,
        otel_endpoint=otel_endpoint or "http://localhost:4317",
        **kwargs,
    )
    return PhoenixMonitor(config)


__all__ = [
    "PhoenixConfig",
    "PhoenixMonitor",
    "AlertSeverity",
    "DriftType",
    "MetricUnit",
    "EmbeddingRecord",
    "DriftResult",
    "AlertConfig",
    "AlertEvent",
    "MonitoringMetric",
    "create_phoenix_monitor",
]
```

## Step 5: Create Package Init

Create `core/observability/__init__.py`:

```python
"""
Observability Layer - Phase 6 Part 1
Provides tracing, evaluation, and monitoring for LLM operations.
"""

from core.observability.langfuse_tracer import (
    LangfuseConfig,
    LangfuseTracer,
    TraceLevel,
    FeedbackType,
    GenerationMetadata,
    TraceMetadata,
    FeedbackRecord,
    create_langfuse_tracer,
)

from core.observability.opik_evaluator import (
    OpikConfig,
    OpikEvaluator,
    MetricType,
    MetricDefinition,
    EvaluationResult,
    BatchEvaluationResult,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    create_opik_evaluator,
)

from core.observability.phoenix_monitor import (
    PhoenixConfig,
    PhoenixMonitor,
    AlertSeverity,
    DriftType,
    MetricUnit,
    EmbeddingRecord,
    DriftResult,
    AlertConfig,
    AlertEvent,
    MonitoringMetric,
    create_phoenix_monitor,
)

__all__ = [
    # Langfuse
    "LangfuseConfig",
    "LangfuseTracer",
    "TraceLevel",
    "FeedbackType",
    "GenerationMetadata",
    "TraceMetadata",
    "FeedbackRecord",
    "create_langfuse_tracer",
    # Opik
    "OpikConfig",
    "OpikEvaluator",
    "MetricType",
    "MetricDefinition",
    "EvaluationResult",
    "BatchEvaluationResult",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentStatus",
    "create_opik_evaluator",
    # Phoenix
    "PhoenixConfig",
    "PhoenixMonitor",
    "AlertSeverity",
    "DriftType",
    "MetricUnit",
    "EmbeddingRecord",
    "DriftResult",
    "AlertConfig",
    "AlertEvent",
    "MonitoringMetric",
    "create_phoenix_monitor",
]
```

## Validation Part 1

```python
# Test imports
from core.observability.langfuse_tracer import LangfuseTracer
from core.observability.opik_evaluator import OpikEvaluator
from core.observability.phoenix_monitor import PhoenixMonitor
print("Part 1 imports OK")

# Test instantiation
tracer = LangfuseTracer()
evaluator = OpikEvaluator()
monitor = PhoenixMonitor()
print("Part 1 instantiation OK")

# Test basic operations
print(f"Tracer stats: {tracer.get_statistics()}")
print(f"Evaluator stats: {evaluator.get_statistics()}")
print(f"Monitor stats: {monitor.get_statistics()}")
```

## Integration Example

```python
"""Example integration with LLM Gateway."""
import asyncio
from core.observability import (
    LangfuseTracer,
    OpikEvaluator,
    PhoenixMonitor,
    TraceLevel,
    MetricType,
)

async def main():
    # Initialize observability components
    tracer = LangfuseTracer()
    evaluator = OpikEvaluator()
    monitor = PhoenixMonitor()
    
    # Trace a generation
    with tracer.trace_span("complete_task", tags=["demo"]) as span:
        # Simulate LLM call
        input_messages = [{"role": "user", "content": "Hello!"}]
        output = "Hello! How can I help you today?"
        
        # Record generation
        gen_meta = tracer.trace_generation(
            name="greeting",
            model="gpt-4o",
            input_messages=input_messages,
            output=output,
            prompt_tokens=10,
            completion_tokens=15,
            latency_ms=150.0,
            trace_id=span.trace_id,
        )
        
        print(f"Generation cost: ${gen_meta.cost_usd:.6f}")
        
        # Evaluate output
        result = evaluator.evaluate_single(
            input_text="Hello!",
            output_text=output,
            metric_name="relevance",
        )
        print(f"Relevance score: {result.score:.2f}")
        
        # Log metrics
        monitor.log_metric("latency", 150.0, unit=MetricUnit.MILLISECONDS)
        monitor.log_metric("tokens", 25, unit=MetricUnit.TOKENS)
    
    # Get statistics
    print(f"\nTracer: {tracer.get_statistics()}")
    print(f"Evaluator: {evaluator.get_statistics()}")
    print(f"Monitor: {monitor.get_statistics()}")
    
    # Cleanup
    tracer.shutdown()
    monitor.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Next: Part 2

Continue with Part 2 for: deepeval, ragas, promptfoo

Part 2 will add:
- DeepEval for comprehensive LLM testing
- RAGAS for RAG evaluation metrics
- Promptfoo for prompt testing and red-teaming
