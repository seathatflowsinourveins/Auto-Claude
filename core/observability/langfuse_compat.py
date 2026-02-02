#!/usr/bin/env python3
"""
Langfuse Compatibility Layer for Python 3.14+
Part of V34 Architecture - Phase 10 Fix.

This module provides a direct HTTP API implementation for Langfuse tracing,
bypassing the SDK's Pydantic V1 dependencies that break on Python 3.14+.

The Langfuse SDK uses pydantic.v1 internally, which calls ForwardRef._evaluate()
- a method that's deprecated in Python 3.14 and removed in 3.16. This module
implements the essential Langfuse functionality using direct HTTP calls.

Usage:
    from core.observability.langfuse_compat import (
        LangfuseCompat,
        create_trace,
        create_span,
        create_generation,
    )

    # Initialize client
    client = LangfuseCompat()

    # Create a trace
    trace = client.create_trace(name="my-trace")

    # Create a span within the trace
    span = client.create_span(
        trace_id=trace.id,
        name="my-span",
        input={"query": "hello"}
    )

    # End the span
    client.end_span(span.id, output={"response": "world"})
"""

from __future__ import annotations

import os
import json
import time
import uuid
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
from queue import Queue, Empty
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# =============================================================================
# CONFIGURATION
# =============================================================================

LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")

# API endpoints
API_BASE = f"{LANGFUSE_HOST}/api/public"
INGESTION_ENDPOINT = f"{API_BASE}/ingestion"

# Logger
logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class LangfuseCompatError(Exception):
    """Base exception for Langfuse compatibility layer."""
    pass


class LangfuseConfigError(LangfuseCompatError):
    """Raised when Langfuse is not properly configured."""

    def __init__(self, missing_keys: List[str]):
        self.missing_keys = missing_keys
        msg = f"""
============================================================
LANGFUSE CONFIGURATION ERROR
============================================================

Missing environment variables:
  {', '.join(missing_keys)}

Required configuration:
  LANGFUSE_PUBLIC_KEY=pk-lf-...
  LANGFUSE_SECRET_KEY=sk-lf-...
  LANGFUSE_HOST=https://cloud.langfuse.com  (optional)

Get your API keys at: https://cloud.langfuse.com
============================================================
"""
        super().__init__(msg)


class LangfuseAPIError(LangfuseCompatError):
    """Raised when API call fails."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"Langfuse API error ({status_code}): {message}")


# =============================================================================
# ENUMS
# =============================================================================

class ObservationType(str, Enum):
    TRACE = "trace-create"
    SPAN = "span-create"
    SPAN_UPDATE = "span-update"
    GENERATION = "generation-create"
    GENERATION_UPDATE = "generation-update"
    EVENT = "event-create"
    SCORE = "score-create"


class SpanLevel(str, Enum):
    DEBUG = "DEBUG"
    DEFAULT = "DEFAULT"
    WARNING = "WARNING"
    ERROR = "ERROR"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TraceData:
    """Represents a Langfuse trace."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    input: Optional[Any] = None
    output: Optional[Any] = None
    timestamp: Optional[str] = None
    release: Optional[str] = None
    version: Optional[str] = None
    public: bool = False

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class SpanData:
    """Represents a Langfuse span."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: Optional[str] = None
    parent_observation_id: Optional[str] = None
    name: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    level: SpanLevel = SpanLevel.DEFAULT
    status_message: Optional[str] = None
    input: Optional[Any] = None
    output: Optional[Any] = None
    version: Optional[str] = None

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now(timezone.utc).isoformat()


@dataclass
class GenerationData:
    """Represents a Langfuse generation (LLM call)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: Optional[str] = None
    parent_observation_id: Optional[str] = None
    name: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    completion_start_time: Optional[str] = None
    model: Optional[str] = None
    model_parameters: Optional[Dict[str, Any]] = None
    input: Optional[Any] = None
    output: Optional[Any] = None
    usage: Optional[Dict[str, int]] = None  # prompt_tokens, completion_tokens, total_tokens
    metadata: Optional[Dict[str, Any]] = None
    level: SpanLevel = SpanLevel.DEFAULT
    status_message: Optional[str] = None
    version: Optional[str] = None

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now(timezone.utc).isoformat()


@dataclass
class ScoreData:
    """Represents a Langfuse score."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = ""
    observation_id: Optional[str] = None
    name: str = ""
    value: Union[float, int, str] = 0.0
    comment: Optional[str] = None
    data_type: str = "NUMERIC"  # NUMERIC, BOOLEAN, CATEGORICAL


@dataclass
class EventData:
    """Represents a Langfuse event."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: Optional[str] = None
    parent_observation_id: Optional[str] = None
    name: Optional[str] = None
    start_time: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    level: SpanLevel = SpanLevel.DEFAULT
    status_message: Optional[str] = None
    input: Optional[Any] = None
    output: Optional[Any] = None
    version: Optional[str] = None

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now(timezone.utc).isoformat()


# =============================================================================
# HTTP CLIENT
# =============================================================================

def _make_request(
    endpoint: str,
    method: str = "POST",
    data: Optional[Dict] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Make an HTTP request to Langfuse API.

    Uses basic auth with public/secret keys.
    """
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        missing = []
        if not LANGFUSE_PUBLIC_KEY:
            missing.append("LANGFUSE_PUBLIC_KEY")
        if not LANGFUSE_SECRET_KEY:
            missing.append("LANGFUSE_SECRET_KEY")
        raise LangfuseConfigError(missing)

    import base64
    auth_string = f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}"
    auth_bytes = base64.b64encode(auth_string.encode("utf-8")).decode("utf-8")

    headers = {
        "Authorization": f"Basic {auth_bytes}",
        "Content-Type": "application/json",
        "User-Agent": "unleash-langfuse-compat/1.0"
    }

    body = json.dumps(data).encode("utf-8") if data else None

    request = Request(endpoint, data=body, headers=headers, method=method)

    try:
        with urlopen(request, timeout=timeout) as response:
            response_data = response.read().decode("utf-8")
            return json.loads(response_data) if response_data else {}
    except HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else str(e)
        raise LangfuseAPIError(e.code, error_body)
    except URLError as e:
        raise LangfuseCompatError(f"Network error: {e.reason}")


def _clean_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Remove None values from a dictionary."""
    return {k: v for k, v in d.items() if v is not None}


# =============================================================================
# BATCH INGESTION QUEUE
# =============================================================================

class IngestionQueue:
    """
    Background queue for batching Langfuse ingestion events.

    Collects events and flushes them in batches to reduce API calls.
    """

    def __init__(self, batch_size: int = 10, flush_interval: float = 5.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._queue: Queue = Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the background flush thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background flush thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)

    def enqueue(self, observation_type: ObservationType, body: Dict[str, Any]):
        """Add an event to the queue."""
        event = {
            "id": str(uuid.uuid4()),
            "type": observation_type.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "body": body
        }
        self._queue.put(event)

        # Flush immediately if batch is full
        if self._queue.qsize() >= self.batch_size:
            self._flush()

    def _flush_loop(self):
        """Background loop that flushes events periodically."""
        while self._running:
            time.sleep(self.flush_interval)
            self._flush()

    def _flush(self):
        """Flush all queued events to Langfuse."""
        events = []
        while True:
            try:
                event = self._queue.get_nowait()
                events.append(event)
            except Empty:
                break

        if not events:
            return

        try:
            _make_request(INGESTION_ENDPOINT, data={"batch": events})
            logger.debug(f"Flushed {len(events)} events to Langfuse")
        except Exception as e:
            logger.error(f"Failed to flush events to Langfuse: {e}")
            # Re-queue failed events
            for event in events:
                self._queue.put(event)

    def flush_sync(self):
        """Synchronously flush all events."""
        self._flush()


# =============================================================================
# LANGFUSE COMPAT CLIENT
# =============================================================================

class LangfuseCompat:
    """
    Langfuse-compatible client using direct HTTP API.

    This replaces the official Langfuse SDK to avoid Pydantic V1 dependencies
    that break on Python 3.14+.
    """

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        flush_at: int = 10,
        flush_interval: float = 5.0,
        enabled: bool = True
    ):
        global LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST, API_BASE, INGESTION_ENDPOINT

        if public_key:
            LANGFUSE_PUBLIC_KEY = public_key
        if secret_key:
            LANGFUSE_SECRET_KEY = secret_key
        if host:
            LANGFUSE_HOST = host
            API_BASE = f"{LANGFUSE_HOST}/api/public"
            INGESTION_ENDPOINT = f"{API_BASE}/ingestion"

        self.enabled = enabled
        self._queue = IngestionQueue(batch_size=flush_at, flush_interval=flush_interval)

        if enabled:
            self._queue.start()

    def shutdown(self):
        """Flush remaining events and stop background thread."""
        self._queue.flush_sync()
        self._queue.stop()

    def flush(self):
        """Flush all pending events synchronously."""
        self._queue.flush_sync()

    # =========================================================================
    # TRACE OPERATIONS
    # =========================================================================

    def create_trace(
        self,
        name: Optional[str] = None,
        id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        public: bool = False,
        **kwargs
    ) -> TraceData:
        """Create a new trace."""
        trace = TraceData(
            id=id or str(uuid.uuid4()),
            name=name,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            tags=tags,
            input=input,
            output=output,
            public=public
        )

        if self.enabled:
            body = _clean_dict(asdict(trace))
            self._queue.enqueue(ObservationType.TRACE, body)

        return trace

    def update_trace(
        self,
        trace_id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        **kwargs
    ):
        """Update an existing trace."""
        if not self.enabled:
            return

        body = _clean_dict({
            "id": trace_id,
            "name": name,
            "metadata": metadata,
            "input": input,
            "output": output
        })
        self._queue.enqueue(ObservationType.TRACE, body)

    # =========================================================================
    # SPAN OPERATIONS
    # =========================================================================

    def create_span(
        self,
        trace_id: str,
        name: Optional[str] = None,
        id: Optional[str] = None,
        parent_observation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        level: SpanLevel = SpanLevel.DEFAULT,
        input: Optional[Any] = None,
        **kwargs
    ) -> SpanData:
        """Create a new span within a trace."""
        span = SpanData(
            id=id or str(uuid.uuid4()),
            trace_id=trace_id,
            parent_observation_id=parent_observation_id,
            name=name,
            metadata=metadata,
            level=level,
            input=input
        )

        if self.enabled:
            body = _clean_dict(asdict(span))
            body["level"] = span.level.value
            self._queue.enqueue(ObservationType.SPAN, body)

        return span

    def end_span(
        self,
        span_id: str,
        output: Optional[Any] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """End a span with optional output."""
        if not self.enabled:
            return

        body = _clean_dict({
            "id": span_id,
            "end_time": datetime.now(timezone.utc).isoformat(),
            "output": output,
            "level": level.value if level else None,
            "status_message": status_message,
            "metadata": metadata
        })
        self._queue.enqueue(ObservationType.SPAN_UPDATE, body)

    # =========================================================================
    # GENERATION OPERATIONS
    # =========================================================================

    def create_generation(
        self,
        trace_id: str,
        name: Optional[str] = None,
        id: Optional[str] = None,
        parent_observation_id: Optional[str] = None,
        model: Optional[str] = None,
        model_parameters: Optional[Dict[str, Any]] = None,
        input: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> GenerationData:
        """Create a new generation (LLM call) within a trace."""
        generation = GenerationData(
            id=id or str(uuid.uuid4()),
            trace_id=trace_id,
            parent_observation_id=parent_observation_id,
            name=name,
            model=model,
            model_parameters=model_parameters,
            input=input,
            metadata=metadata
        )

        if self.enabled:
            body = _clean_dict(asdict(generation))
            body["level"] = generation.level.value
            self._queue.enqueue(ObservationType.GENERATION, body)

        return generation

    def end_generation(
        self,
        generation_id: str,
        output: Optional[Any] = None,
        usage: Optional[Dict[str, int]] = None,
        level: Optional[SpanLevel] = None,
        status_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """End a generation with output and usage stats."""
        if not self.enabled:
            return

        body = _clean_dict({
            "id": generation_id,
            "end_time": datetime.now(timezone.utc).isoformat(),
            "output": output,
            "usage": usage,
            "level": level.value if level else None,
            "status_message": status_message,
            "metadata": metadata
        })
        self._queue.enqueue(ObservationType.GENERATION_UPDATE, body)

    # =========================================================================
    # EVENT OPERATIONS
    # =========================================================================

    def create_event(
        self,
        trace_id: str,
        name: Optional[str] = None,
        id: Optional[str] = None,
        parent_observation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        level: SpanLevel = SpanLevel.DEFAULT,
        input: Optional[Any] = None,
        output: Optional[Any] = None,
        **kwargs
    ) -> EventData:
        """Create a new event within a trace."""
        event = EventData(
            id=id or str(uuid.uuid4()),
            trace_id=trace_id,
            parent_observation_id=parent_observation_id,
            name=name,
            metadata=metadata,
            level=level,
            input=input,
            output=output
        )

        if self.enabled:
            body = _clean_dict(asdict(event))
            body["level"] = event.level.value
            self._queue.enqueue(ObservationType.EVENT, body)

        return event

    # =========================================================================
    # SCORE OPERATIONS
    # =========================================================================

    def score(
        self,
        trace_id: str,
        name: str,
        value: Union[float, int, str],
        id: Optional[str] = None,
        observation_id: Optional[str] = None,
        comment: Optional[str] = None,
        data_type: str = "NUMERIC",
        **kwargs
    ) -> ScoreData:
        """Create a score for a trace or observation."""
        score_data = ScoreData(
            id=id or str(uuid.uuid4()),
            trace_id=trace_id,
            observation_id=observation_id,
            name=name,
            value=value,
            comment=comment,
            data_type=data_type
        )

        if self.enabled:
            body = _clean_dict(asdict(score_data))
            self._queue.enqueue(ObservationType.SCORE, body)

        return score_data


# =============================================================================
# DECORATOR FOR TRACING
# =============================================================================

def observe(
    name: Optional[str] = None,
    as_type: str = "span",  # "span" or "generation"
    capture_input: bool = True,
    capture_output: bool = True,
    transform_input: Optional[Any] = None,  # Callable[[Any], Any]
    transform_output: Optional[Any] = None,  # Callable[[Any], Any]
):
    """
    Decorator for automatic function tracing.

    Similar to @langfuse.observe() but using the compat layer.

    Usage:
        @observe(name="my-function")
        def my_function(x, y):
            return x + y
    """
    def decorator(func):
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create client
            client = _get_default_client()
            if client is None or not client.enabled:
                return func(*args, **kwargs)

            # Determine observation name
            obs_name = name or func.__name__

            # Get current trace context (simplified - no context propagation)
            trace = client.create_trace(name=f"auto-{obs_name}")

            # Capture input
            input_data = None
            if capture_input:
                input_data = {"args": args, "kwargs": kwargs}
                if transform_input:
                    input_data = transform_input(input_data)

            # Create observation
            if as_type == "generation":
                obs = client.create_generation(
                    trace_id=trace.id,
                    name=obs_name,
                    input=input_data
                )
            else:
                obs = client.create_span(
                    trace_id=trace.id,
                    name=obs_name,
                    input=input_data
                )

            # Execute function
            try:
                result = func(*args, **kwargs)

                # Capture output
                output_data = result if capture_output else None
                if capture_output and transform_output:
                    output_data = transform_output(output_data)

                # End observation
                if as_type == "generation":
                    client.end_generation(obs.id, output=output_data)
                else:
                    client.end_span(obs.id, output=output_data)

                return result

            except Exception as e:
                # Record error
                if as_type == "generation":
                    client.end_generation(
                        obs.id,
                        level=SpanLevel.ERROR,
                        status_message=str(e)
                    )
                else:
                    client.end_span(
                        obs.id,
                        level=SpanLevel.ERROR,
                        status_message=str(e)
                    )
                raise

        return wrapper
    return decorator


# =============================================================================
# DEFAULT CLIENT SINGLETON
# =============================================================================

_default_client: Optional[LangfuseCompat] = None
_client_lock = threading.Lock()


def _get_default_client() -> Optional[LangfuseCompat]:
    """Get or create the default client singleton."""
    global _default_client
    with _client_lock:
        if _default_client is None:
            # Only create if configured
            if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
                _default_client = LangfuseCompat()
        return _default_client


def get_client() -> LangfuseCompat:
    """Get the default Langfuse client."""
    client = _get_default_client()
    if client is None:
        raise LangfuseConfigError(["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"])
    return client


def init_langfuse(
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    host: Optional[str] = None,
    **kwargs
) -> LangfuseCompat:
    """Initialize the default Langfuse client."""
    global _default_client
    with _client_lock:
        _default_client = LangfuseCompat(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            **kwargs
        )
        return _default_client


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_trace(**kwargs) -> TraceData:
    """Create a trace using the default client."""
    return get_client().create_trace(**kwargs)


def create_span(trace_id: str, **kwargs) -> SpanData:
    """Create a span using the default client."""
    return get_client().create_span(trace_id=trace_id, **kwargs)


def create_generation(trace_id: str, **kwargs) -> GenerationData:
    """Create a generation using the default client."""
    return get_client().create_generation(trace_id=trace_id, **kwargs)


def flush():
    """Flush the default client."""
    client = _get_default_client()
    if client:
        client.flush()


# =============================================================================
# EXPORTS
# =============================================================================

# Compatibility flag for V35 validation
LANGFUSE_COMPAT_AVAILABLE = True

__all__ = [
    # Exceptions
    "LangfuseCompatError",
    "LangfuseConfigError",
    "LangfuseAPIError",
    # Enums
    "ObservationType",
    "SpanLevel",
    # Data classes
    "TraceData",
    "SpanData",
    "GenerationData",
    "ScoreData",
    "EventData",
    # Client
    "LangfuseCompat",
    "get_client",
    "init_langfuse",
    # Convenience
    "create_trace",
    "create_span",
    "create_generation",
    "flush",
    # Decorator
    "observe",
    # Compat flag
    "LANGFUSE_COMPAT_AVAILABLE",
]
