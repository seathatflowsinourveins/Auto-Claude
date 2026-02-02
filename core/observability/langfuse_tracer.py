#!/usr/bin/env python3
"""
Langfuse Tracer - LLM Tracing and Cost Tracking
Part of the V33 Observability Layer.

Uses Langfuse for comprehensive LLM tracing with:
- Trace spans and hierarchical tracking
- Generation metadata and cost calculation
- User feedback collection
- Session management
"""

from __future__ import annotations

import os
import uuid
from enum import Enum
from typing import Any, Optional, Dict, List, Callable
from datetime import datetime
from dataclasses import dataclass, field
from contextlib import contextmanager

from pydantic import BaseModel, Field

try:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None
    observe = None
    langfuse_context = None


# ============================================================================
# Trace Configuration
# ============================================================================

class TraceLevel(str, Enum):
    """Trace verbosity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class FeedbackType(str, Enum):
    """Types of user feedback."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"
    COMMENT = "comment"
    CORRECTION = "correction"


@dataclass
class GenerationMetadata:
    """Metadata for a generation event."""
    model: str = "claude-sonnet-4-20250514"
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    temperature: float = 0.7
    max_tokens: int = 4096
    stop_reason: Optional[str] = None


@dataclass
class TraceSpan:
    """A span within a trace."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    level: TraceLevel = TraceLevel.INFO
    parent_id: Optional[str] = None
    input: Optional[str] = None
    output: Optional[str] = None


class TraceResult(BaseModel):
    """Result of a trace operation."""
    trace_id: str = Field(description="Unique trace identifier")
    success: bool = Field(default=True)
    spans: List[str] = Field(default_factory=list)
    total_latency_ms: float = Field(default=0.0)
    total_cost_usd: float = Field(default=0.0)
    error: Optional[str] = Field(default=None)


# ============================================================================
# Langfuse Tracer
# ============================================================================

class LangfuseTracer:
    """
    Langfuse-based tracing for LLM applications.

    Provides comprehensive observability including:
    - Hierarchical trace spans
    - Generation tracking with cost calculation
    - User feedback collection
    - Session and user management
    """

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: str = "https://cloud.langfuse.com",
        debug: bool = False,
    ):
        """
        Initialize the Langfuse tracer.

        Args:
            public_key: Langfuse public key (or LANGFUSE_PUBLIC_KEY env var)
            secret_key: Langfuse secret key (or LANGFUSE_SECRET_KEY env var)
            host: Langfuse host URL
            debug: Enable debug mode
        """
        self.public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.host = host
        self.debug = debug

        self._client = None
        self._current_trace = None
        self._spans: Dict[str, TraceSpan] = {}

        if LANGFUSE_AVAILABLE and self.public_key and self.secret_key:
            try:
                self._client = Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host,
                    debug=self.debug,
                )
            except Exception:
                pass

    @property
    def is_available(self) -> bool:
        """Check if Langfuse is available and configured."""
        return self._client is not None

    def create_trace(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Create a new trace.

        Args:
            name: Trace name
            user_id: Optional user identifier
            session_id: Optional session identifier
            metadata: Optional metadata
            tags: Optional tags

        Returns:
            Trace ID
        """
        trace_id = str(uuid.uuid4())

        if self._client:
            try:
                self._current_trace = self._client.trace(
                    id=trace_id,
                    name=name,
                    user_id=user_id,
                    session_id=session_id,
                    metadata=metadata or {},
                    tags=tags or [],
                )
            except Exception:
                pass

        return trace_id

    def start_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        input: Optional[str] = None,
        metadata: Optional[Dict] = None,
        level: TraceLevel = TraceLevel.INFO,
    ) -> str:
        """
        Start a new span within a trace.

        Args:
            name: Span name
            trace_id: Parent trace ID
            parent_span_id: Parent span ID (for nested spans)
            input: Span input
            metadata: Span metadata
            level: Trace level

        Returns:
            Span ID
        """
        span = TraceSpan(
            name=name,
            parent_id=parent_span_id,
            input=input,
            metadata=metadata or {},
            level=level,
        )
        self._spans[span.id] = span

        if self._client and self._current_trace:
            try:
                self._current_trace.span(
                    id=span.id,
                    name=name,
                    input=input,
                    metadata=metadata or {},
                    level=level.value,
                )
            except Exception:
                pass

        return span.id

    def end_span(
        self,
        span_id: str,
        output: Optional[str] = None,
        status_code: int = 200,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        End a span.

        Args:
            span_id: Span ID to end
            output: Span output
            status_code: HTTP-like status code
            metadata: Additional metadata
        """
        if span_id in self._spans:
            span = self._spans[span_id]
            span.end_time = datetime.now()
            span.output = output
            if metadata:
                span.metadata.update(metadata)

        if self._client:
            try:
                # Langfuse handles span ending internally
                pass
            except Exception:
                pass

    def log_generation(
        self,
        name: str,
        prompt: str,
        completion: str,
        metadata: GenerationMetadata,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ) -> str:
        """
        Log an LLM generation event.

        Args:
            name: Generation name
            prompt: Input prompt
            completion: Model completion
            metadata: Generation metadata
            trace_id: Parent trace ID
            span_id: Parent span ID

        Returns:
            Generation ID
        """
        generation_id = str(uuid.uuid4())

        if self._client and self._current_trace:
            try:
                self._current_trace.generation(
                    id=generation_id,
                    name=name,
                    input=prompt,
                    output=completion,
                    model=metadata.model,
                    model_parameters={
                        "temperature": metadata.temperature,
                        "max_tokens": metadata.max_tokens,
                    },
                    usage={
                        "input": metadata.input_tokens,
                        "output": metadata.output_tokens,
                        "total": metadata.total_tokens,
                    },
                    metadata={
                        "latency_ms": metadata.latency_ms,
                        "cost_usd": metadata.cost_usd,
                        "stop_reason": metadata.stop_reason,
                    },
                )
            except Exception:
                pass

        return generation_id

    def log_feedback(
        self,
        trace_id: str,
        feedback_type: FeedbackType,
        value: Any,
        comment: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Log user feedback for a trace.

        Args:
            trace_id: Trace to attach feedback to
            feedback_type: Type of feedback
            value: Feedback value
            comment: Optional comment
            user_id: User providing feedback

        Returns:
            Success status
        """
        if self._client:
            try:
                score_name = feedback_type.value
                score_value = 1 if feedback_type == FeedbackType.THUMBS_UP else (
                    0 if feedback_type == FeedbackType.THUMBS_DOWN else value
                )

                self._client.score(
                    trace_id=trace_id,
                    name=score_name,
                    value=score_value,
                    comment=comment,
                )
                return True
            except Exception:
                return False
        return False

    @contextmanager
    def trace_context(
        self,
        name: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """
        Context manager for tracing.

        Args:
            name: Trace name
            user_id: Optional user ID
            session_id: Optional session ID

        Yields:
            Trace ID
        """
        trace_id = self.create_trace(name, user_id, session_id)
        start_time = datetime.now()
        error = None

        try:
            yield trace_id
        except Exception as e:
            error = str(e)
            raise
        finally:
            latency = (datetime.now() - start_time).total_seconds() * 1000
            if self._client:
                try:
                    self._client.flush()
                except Exception:
                    pass

    def flush(self) -> None:
        """Flush pending events to Langfuse."""
        if self._client:
            try:
                self._client.flush()
            except Exception:
                pass

    def shutdown(self) -> None:
        """Shutdown the tracer and flush remaining events."""
        self.flush()
        if self._client:
            try:
                self._client.shutdown()
            except Exception:
                pass


# ============================================================================
# Decorator for automatic tracing
# ============================================================================

def traced(
    name: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
):
    """
    Decorator for automatic function tracing.

    Args:
        name: Optional custom name (defaults to function name)
        capture_input: Whether to capture function input
        capture_output: Whether to capture function output
    """
    def decorator(func: Callable) -> Callable:
        if LANGFUSE_AVAILABLE and observe:
            return observe(
                name=name or func.__name__,
                capture_input=capture_input,
                capture_output=capture_output,
            )(func)
        return func
    return decorator


# ============================================================================
# Convenience Functions
# ============================================================================

def create_langfuse_tracer(
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    **kwargs: Any,
) -> LangfuseTracer:
    """
    Factory function to create a LangfuseTracer.

    Args:
        public_key: Optional public key
        secret_key: Optional secret key
        **kwargs: Additional configuration

    Returns:
        Configured LangfuseTracer instance
    """
    return LangfuseTracer(
        public_key=public_key,
        secret_key=secret_key,
        **kwargs,
    )


# Export availability
__all__ = [
    "LangfuseTracer",
    "TraceLevel",
    "FeedbackType",
    "GenerationMetadata",
    "TraceSpan",
    "TraceResult",
    "traced",
    "create_langfuse_tracer",
    "LANGFUSE_AVAILABLE",
]
