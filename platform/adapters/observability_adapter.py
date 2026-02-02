"""
UNLEASH L5 Observability Layer Adapter
======================================

Unified observability adapter supporting multiple backends:
- Langfuse (OTEL-based, @observe decorator)
- Arize Phoenix (OpenInference instrumentors)

Verified against official docs 2026-01-30:
- Context7: /llmstxt/langfuse_llms_txt (1025 snippets)
- Context7: /arize-ai/phoenix (3783 snippets)
- Exa deep search for production patterns
"""

from typing import Optional, Callable, Any, Union
from functools import wraps
from enum import Enum
from dataclasses import dataclass
import logging
import os

logger = logging.getLogger(__name__)


class ObservabilityBackend(Enum):
    """Supported observability backends"""
    LANGFUSE = "langfuse"
    PHOENIX = "phoenix"
    BOTH = "both"  # Dual-write for migration
    NONE = "none"  # Disabled


class SpanKind(Enum):
    """OpenInference span kinds for semantic categorization"""
    CHAIN = "chain"
    LLM = "llm"
    RETRIEVER = "retriever"
    EMBEDDING = "embedding"
    TOOL = "tool"
    AGENT = "agent"
    RERANKER = "reranker"


@dataclass
class TraceContext:
    """Context for distributed tracing"""
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[dict] = None


class UnifiedObserver:
    """
    Unified observability layer for UNLEASH platform.

    Supports multiple backends with automatic instrumentation
    and manual span creation for custom logic.

    Example:
        observer = UnifiedObserver(
            backend=ObservabilityBackend.PHOENIX,
            project_name="unleash-platform"
        )

        @observer.trace(name="sdk-call", span_kind=SpanKind.LLM)
        def call_llm(prompt: str):
            return openai.chat.completions.create(...)
    """

    def __init__(
        self,
        backend: ObservabilityBackend = ObservabilityBackend.PHOENIX,
        project_name: str = "unleash-app",
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_export: bool = True,
        auto_instrument: bool = True,
    ):
        self.backend = backend
        self.project_name = project_name
        self.endpoint = endpoint
        self.api_key = api_key
        self.batch_export = batch_export
        self.auto_instrument = auto_instrument

        self._langfuse_client = None
        self._phoenix_tracer = None
        self._initialized = False

        if backend != ObservabilityBackend.NONE:
            self._init_backends()

    def _init_backends(self) -> None:
        """Initialize selected observability backends"""
        try:
            if self.backend in (ObservabilityBackend.LANGFUSE, ObservabilityBackend.BOTH):
                self._init_langfuse()

            if self.backend in (ObservabilityBackend.PHOENIX, ObservabilityBackend.BOTH):
                self._init_phoenix()

            self._initialized = True
            logger.info(f"Observability initialized: {self.backend.value}")

        except ImportError as e:
            logger.warning(f"Observability backend not available: {e}")
            self.backend = ObservabilityBackend.NONE

    def _init_langfuse(self) -> None:
        """Initialize Langfuse backend"""
        from langfuse import get_client

        # Configure from environment or params
        if self.api_key:
            os.environ.setdefault("LANGFUSE_SECRET_KEY", self.api_key)
        if self.endpoint:
            os.environ.setdefault("LANGFUSE_HOST", self.endpoint)

        self._langfuse_client = get_client()
        logger.info("Langfuse observability initialized")

    def _init_phoenix(self) -> None:
        """Initialize Arize Phoenix backend"""
        from phoenix.otel import register

        # Configure endpoint
        endpoint = self.endpoint or os.environ.get(
            "PHOENIX_COLLECTOR_ENDPOINT",
            "http://localhost:6006"
        )

        # Set API key for cloud
        if self.api_key:
            os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={self.api_key}"

        provider = register(
            project_name=self.project_name,
            endpoint=endpoint,
            auto_instrument=self.auto_instrument,
            batch=self.batch_export,
        )

        self._phoenix_tracer = provider.get_tracer(f"unleash.{self.project_name}")
        logger.info(f"Phoenix observability initialized: {endpoint}")

    def trace(
        self,
        name: Optional[str] = None,
        span_kind: Union[SpanKind, str] = SpanKind.CHAIN,
        capture_input: bool = True,
        capture_output: bool = True,
    ) -> Callable:
        """
        Universal decorator for tracing functions.

        Args:
            name: Span name (defaults to function name)
            span_kind: Type of span for semantic categorization
            capture_input: Whether to log function inputs
            capture_output: Whether to log function outputs

        Returns:
            Decorated function with tracing
        """
        if isinstance(span_kind, str):
            span_kind = SpanKind(span_kind)

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                span_name = name or func.__name__

                if not self._initialized:
                    return func(*args, **kwargs)

                # Langfuse tracing path
                if self.backend == ObservabilityBackend.LANGFUSE:
                    return self._trace_langfuse(
                        func, span_name, span_kind,
                        capture_input, capture_output,
                        *args, **kwargs
                    )

                # Phoenix tracing path
                if self.backend == ObservabilityBackend.PHOENIX:
                    return self._trace_phoenix(
                        func, span_name, span_kind,
                        capture_input, capture_output,
                        *args, **kwargs
                    )

                # Dual-write path
                if self.backend == ObservabilityBackend.BOTH:
                    return self._trace_both(
                        func, span_name, span_kind,
                        capture_input, capture_output,
                        *args, **kwargs
                    )

                return func(*args, **kwargs)
            return wrapper
        return decorator

    def _trace_langfuse(
        self, func, span_name, span_kind, capture_input, capture_output, *args, **kwargs
    ):
        """Execute with Langfuse tracing"""
        from langfuse import observe

        as_type = "generation" if span_kind == SpanKind.LLM else None
        traced_func = observe(name=span_name, as_type=as_type)(func)
        return traced_func(*args, **kwargs)

    def _trace_phoenix(
        self, func, span_name, span_kind, capture_input, capture_output, *args, **kwargs
    ):
        """Execute with Phoenix tracing"""
        from opentelemetry.trace import Status, StatusCode

        with self._phoenix_tracer.start_as_current_span(
            span_name,
            openinference_span_kind=span_kind.value
        ) as span:
            try:
                if capture_input and args:
                    span.set_attribute("input.value", str(args[0])[:1000])

                result = func(*args, **kwargs)

                if capture_output:
                    span.set_attribute("output.value", str(result)[:1000])
                span.set_status(Status(StatusCode.OK))

                return result

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attribute("error.message", str(e))
                span.set_attribute("error.type", type(e).__name__)
                raise

    def _trace_both(
        self, func, span_name, span_kind, capture_input, capture_output, *args, **kwargs
    ):
        """Execute with both Langfuse and Phoenix tracing (dual-write)"""
        from langfuse import observe
        from opentelemetry.trace import Status, StatusCode

        # Langfuse wrapper
        as_type = "generation" if span_kind == SpanKind.LLM else None
        langfuse_wrapped = observe(name=span_name, as_type=as_type)(func)

        # Phoenix span wrapper
        with self._phoenix_tracer.start_as_current_span(
            span_name,
            openinference_span_kind=span_kind.value
        ) as span:
            try:
                if capture_input and args:
                    span.set_attribute("input.value", str(args[0])[:1000])

                result = langfuse_wrapped(*args, **kwargs)

                if capture_output:
                    span.set_attribute("output.value", str(result)[:1000])
                span.set_status(Status(StatusCode.OK))

                return result

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    def create_trace_id(self, seed: Optional[str] = None) -> str:
        """
        Create a trace ID, optionally deterministic from seed.

        Args:
            seed: Optional seed for deterministic ID generation

        Returns:
            32-character hexadecimal trace ID
        """
        if self._langfuse_client and seed:
            return self._langfuse_client.create_trace_id(seed=seed)

        import uuid
        return uuid.uuid4().hex

    def update_trace_metadata(
        self,
        input_data: Any = None,
        output_data: Any = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Update current trace with metadata.

        Args:
            input_data: Input to record
            output_data: Output to record
            metadata: Additional metadata dict
        """
        if self._langfuse_client:
            from langfuse import get_client
            client = get_client()
            client.update_current_trace(
                input=input_data,
                output=output_data,
                metadata=metadata,
            )

    def instrument_openai(self) -> None:
        """Manually instrument OpenAI SDK"""
        if self._phoenix_tracer:
            from openinference.instrumentation.openai import OpenAIInstrumentor
            OpenAIInstrumentor().instrument()
            logger.info("OpenAI instrumented for Phoenix")

    def instrument_langchain(self) -> None:
        """Manually instrument LangChain"""
        if self._phoenix_tracer:
            from openinference.instrumentation.langchain import LangChainInstrumentor
            LangChainInstrumentor().instrument()
            logger.info("LangChain instrumented for Phoenix")

    def instrument_llamaindex(self) -> None:
        """Manually instrument LlamaIndex"""
        if self._phoenix_tracer:
            from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
            LlamaIndexInstrumentor().instrument()
            logger.info("LlamaIndex instrumented for Phoenix")


# Singleton instance for global access
_default_observer: Optional[UnifiedObserver] = None


def get_observer() -> UnifiedObserver:
    """Get the default observer instance"""
    global _default_observer
    if _default_observer is None:
        _default_observer = UnifiedObserver()
    return _default_observer


def configure_observability(
    backend: ObservabilityBackend = ObservabilityBackend.PHOENIX,
    project_name: str = "unleash-app",
    **kwargs
) -> UnifiedObserver:
    """
    Configure global observability settings.

    Args:
        backend: Which observability backend to use
        project_name: Project name for grouping traces
        **kwargs: Additional configuration options

    Returns:
        Configured UnifiedObserver instance
    """
    global _default_observer
    _default_observer = UnifiedObserver(
        backend=backend,
        project_name=project_name,
        **kwargs
    )
    return _default_observer


# Convenience decorator using default observer
def trace(
    name: Optional[str] = None,
    span_kind: Union[SpanKind, str] = SpanKind.CHAIN,
    **kwargs
) -> Callable:
    """
    Trace decorator using the default observer.

    Example:
        @trace(name="my-operation", span_kind=SpanKind.LLM)
        def call_llm(prompt):
            return openai.chat.completions.create(...)
    """
    return get_observer().trace(name=name, span_kind=span_kind, **kwargs)
