"""
Opik Integration for LLM Observability - V14

Provides deep LLM tracing, experiment tracking, and evaluation capabilities
integrated with Comet's Opik platform.

Features:
- LLM trace logging with full input/output capture
- Token usage and cost tracking
- Evaluation metrics integration
- Seamless integration with existing observability layer

API Key: Configured via OPIK_API_KEY environment variable
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import os
import time
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar
from uuid import uuid4

logger = logging.getLogger(__name__)

# Type variable for generic decorator
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Opik Client Wrapper
# =============================================================================

class OpikClientStatus(Enum):
    """Status of Opik client connection."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class OpikConfig:
    """Configuration for Opik integration."""
    api_key: Optional[str] = None
    workspace: str = "default"
    project_name: str = "unleash-platform"
    enabled: bool = True
    log_prompts: bool = True
    log_responses: bool = True
    log_tokens: bool = True
    log_costs: bool = True
    sample_rate: float = 1.0  # 1.0 = 100% sampling

    def __post_init__(self):
        # Load API key from environment if not provided
        if not self.api_key:
            self.api_key = os.environ.get("OPIK_API_KEY")


@dataclass
class LLMTrace:
    """Represents a single LLM interaction trace."""
    trace_id: str
    span_id: str
    model: str
    provider: str
    operation: str
    input_text: Optional[str] = None
    output_text: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    status: str = "success"
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "model": self.model,
            "provider": self.provider,
            "operation": self.operation,
            "input_text": self.input_text[:500] if self.input_text else None,  # Truncate
            "output_text": self.output_text[:500] if self.output_text else None,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "status": self.status,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class OpikClient:
    """
    Wrapper around Opik SDK for LLM observability.

    Provides:
    - Trace logging for LLM calls
    - Experiment tracking
    - Evaluation metrics
    - Cost tracking
    """

    _instance: Optional["OpikClient"] = None

    def __new__(cls, config: Optional[OpikConfig] = None) -> "OpikClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[OpikConfig] = None):
        if self._initialized:
            return

        self.config = config or OpikConfig()
        self.status = OpikClientStatus.DISCONNECTED
        self._opik = None
        self._traces: List[LLMTrace] = []
        self._trace_buffer: List[Dict[str, Any]] = []
        self._buffer_size = 100

        # Token pricing (USD per 1K tokens) - V14 rates
        self._pricing: Dict[str, Dict[str, float]] = {
            "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "claude-opus-4-5": {"input": 0.015, "output": 0.075},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "o1": {"input": 0.015, "output": 0.060},
            "default": {"input": 0.001, "output": 0.002},
        }

        self._connect()
        self._initialized = True

    def _connect(self) -> None:
        """Initialize Opik connection."""
        if not self.config.enabled or not self.config.api_key:
            logger.warning("Opik disabled or API key not configured")
            self.status = OpikClientStatus.DISCONNECTED
            return

        try:
            import opik

            # Configure Opik with force=True to avoid interactive prompts
            opik.configure(
                api_key=self.config.api_key,
                workspace=self.config.workspace,
                force=True,  # Skip interactive workspace confirmation
            )

            self._opik = opik
            self.status = OpikClientStatus.CONNECTED
            logger.info(f"Opik connected to workspace: {self.config.workspace}")

        except ImportError:
            logger.warning("Opik package not installed. Run: pip install opik")
            self.status = OpikClientStatus.DISCONNECTED
        except Exception as e:
            logger.error(f"Failed to connect to Opik: {e}")
            self.status = OpikClientStatus.ERROR

    @property
    def is_connected(self) -> bool:
        return self.status == OpikClientStatus.CONNECTED

    # -------------------------------------------------------------------------
    # Trace Logging
    # -------------------------------------------------------------------------

    def log_trace(self, trace: LLMTrace) -> None:
        """Log an LLM trace to Opik."""
        self._traces.append(trace)

        if not self.is_connected:
            # Buffer for later
            self._trace_buffer.append(trace.to_dict())
            if len(self._trace_buffer) > self._buffer_size:
                self._trace_buffer.pop(0)
            return

        try:
            # Log to Opik using their trace API
            if self._opik and hasattr(self._opik, "track"):
                # Opik 1.9+ uses track() as a decorator or context manager
                tracker = self._opik.track(
                    name=trace.operation,
                    project_name=self.config.project_name,
                    metadata=trace.metadata,
                )
                # Check if tracker is a context manager
                if hasattr(tracker, "__enter__") and hasattr(tracker, "__exit__"):
                    with tracker as span:  # type: ignore[union-attr]
                        if hasattr(span, "set_attribute"):
                            span.set_attribute("model", trace.model)
                            span.set_attribute("provider", trace.provider)
                            span.set_attribute("tokens.input", trace.input_tokens)
                            span.set_attribute("tokens.output", trace.output_tokens)
                            span.set_attribute("tokens.total", trace.total_tokens)
                            span.set_attribute("cost_usd", trace.cost_usd)
                            span.set_attribute("latency_ms", trace.latency_ms)

                            if trace.input_text and self.config.log_prompts:
                                span.set_attribute("input", trace.input_text[:1000])

                            if trace.output_text and self.config.log_responses:
                                span.set_attribute("output", trace.output_text[:1000])

                            if trace.error:
                                span.set_attribute("error", trace.error)
                                span.set_attribute("status", "error")
                else:
                    # Fallback: just log to internal buffer
                    logger.debug(f"Opik tracker not a context manager, buffering trace")

        except Exception as e:
            logger.warning(f"Failed to log trace to Opik: {e}")

    def create_trace(
        self,
        model: str,
        provider: str,
        operation: str,
        input_text: Optional[str] = None,
        output_text: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LLMTrace:
        """Create and log an LLM trace."""
        trace_id = str(uuid4())
        span_id = str(uuid4())

        # Calculate cost
        cost = self._calculate_cost(model, input_tokens, output_tokens)

        trace = LLMTrace(
            trace_id=trace_id,
            span_id=span_id,
            model=model,
            provider=provider,
            operation=operation,
            input_text=input_text,
            output_text=output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            status="error" if error else "success",
            error=error,
            metadata=metadata or {},
        )

        self.log_trace(trace)
        return trace

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD for token usage."""
        # Find matching pricing
        pricing = self._pricing.get("default")
        for model_key, prices in self._pricing.items():
            if model_key in model.lower():
                pricing = prices
                break

        if pricing:
            input_cost = (input_tokens / 1000) * pricing["input"]
            output_cost = (output_tokens / 1000) * pricing["output"]
            return round(input_cost + output_cost, 6)

        return 0.0

    # -------------------------------------------------------------------------
    # Context Managers
    # -------------------------------------------------------------------------

    @contextmanager
    def trace_llm(
        self,
        model: str,
        provider: str,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for tracing LLM calls."""
        start_time = time.perf_counter()
        result = {
            "input_text": None,
            "output_text": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "error": None,
        }

        try:
            yield result
        except Exception as e:
            result["error"] = str(e)
            raise
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000

            self.create_trace(
                model=model,
                provider=provider,
                operation=operation,
                input_text=result["input_text"],
                output_text=result["output_text"],
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],
                latency_ms=latency_ms,
                error=result["error"],
                metadata=metadata,
            )

    @asynccontextmanager
    async def atrace_llm(
        self,
        model: str,
        provider: str,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Async context manager for tracing LLM calls."""
        start_time = time.perf_counter()
        result = {
            "input_text": None,
            "output_text": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "error": None,
        }

        try:
            yield result
        except Exception as e:
            result["error"] = str(e)
            raise
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000

            self.create_trace(
                model=model,
                provider=provider,
                operation=operation,
                input_text=result["input_text"],
                output_text=result["output_text"],
                input_tokens=result["input_tokens"],
                output_tokens=result["output_tokens"],
                latency_ms=latency_ms,
                error=result["error"],
                metadata=metadata,
            )

    # -------------------------------------------------------------------------
    # Metrics & Statistics
    # -------------------------------------------------------------------------

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics for current session."""
        if not self._traces:
            return {
                "total_traces": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "avg_latency_ms": 0.0,
                "error_rate": 0.0,
            }

        total_tokens = sum(t.total_tokens for t in self._traces)
        total_cost = sum(t.cost_usd for t in self._traces)
        total_latency = sum(t.latency_ms for t in self._traces)
        errors = sum(1 for t in self._traces if t.error)

        return {
            "total_traces": len(self._traces),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 4),
            "avg_latency_ms": round(total_latency / len(self._traces), 2),
            "error_rate": round(errors / len(self._traces), 4),
            "by_model": self._stats_by_model(),
            "by_operation": self._stats_by_operation(),
        }

    def _stats_by_model(self) -> Dict[str, Dict[str, Any]]:
        """Get stats grouped by model."""
        by_model: Dict[str, List[LLMTrace]] = {}
        for trace in self._traces:
            if trace.model not in by_model:
                by_model[trace.model] = []
            by_model[trace.model].append(trace)

        return {
            model: {
                "count": len(traces),
                "tokens": sum(t.total_tokens for t in traces),
                "cost_usd": round(sum(t.cost_usd for t in traces), 4),
            }
            for model, traces in by_model.items()
        }

    def _stats_by_operation(self) -> Dict[str, int]:
        """Get trace count by operation."""
        by_op: Dict[str, int] = {}
        for trace in self._traces:
            by_op[trace.operation] = by_op.get(trace.operation, 0) + 1
        return by_op

    def flush_buffer(self) -> int:
        """Flush buffered traces to Opik."""
        if not self.is_connected or not self._trace_buffer:
            return 0

        flushed = 0
        while self._trace_buffer:
            trace_data = self._trace_buffer.pop(0)
            try:
                # Re-create trace from buffer
                trace = LLMTrace(
                    trace_id=trace_data["trace_id"],
                    span_id=trace_data["span_id"],
                    model=trace_data["model"],
                    provider=trace_data["provider"],
                    operation=trace_data["operation"],
                    input_tokens=trace_data.get("input_tokens", 0),
                    output_tokens=trace_data.get("output_tokens", 0),
                    total_tokens=trace_data.get("total_tokens", 0),
                    cost_usd=trace_data.get("cost_usd", 0.0),
                    latency_ms=trace_data.get("latency_ms", 0.0),
                )
                self.log_trace(trace)
                flushed += 1
            except Exception as e:
                logger.warning(f"Failed to flush trace: {e}")

        return flushed


# =============================================================================
# Decorators
# =============================================================================

def trace_llm_call(
    model: str,
    provider: str = "anthropic",
    operation: Optional[str] = None,
):
    """
    Decorator to trace LLM calls.

    Usage:
        @trace_llm_call(model="claude-3-5-sonnet", provider="anthropic")
        async def my_llm_function(prompt: str) -> str:
            ...
    """
    def decorator(func: F) -> F:
        op_name = operation or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            client = get_opik_client()

            async with client.atrace_llm(model, provider, op_name) as ctx:
                # Try to capture input
                if args and isinstance(args[0], str):
                    ctx["input_text"] = args[0]
                elif "prompt" in kwargs:
                    ctx["input_text"] = str(kwargs["prompt"])
                elif "messages" in kwargs:
                    ctx["input_text"] = json.dumps(kwargs["messages"])[:500]

                result = await func(*args, **kwargs)

                # Try to capture output
                if isinstance(result, str):
                    ctx["output_text"] = result
                elif hasattr(result, "content"):
                    ctx["output_text"] = str(result.content)

                # Try to capture token usage
                if not isinstance(result, str) and hasattr(result, "usage"):
                    usage = getattr(result, "usage", None)
                    if usage:
                        ctx["input_tokens"] = getattr(usage, "input_tokens", 0)
                        ctx["output_tokens"] = getattr(usage, "output_tokens", 0)

                return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            client = get_opik_client()

            with client.trace_llm(model, provider, op_name) as ctx:
                if args and isinstance(args[0], str):
                    ctx["input_text"] = args[0]
                elif "prompt" in kwargs:
                    ctx["input_text"] = str(kwargs["prompt"])

                result = func(*args, **kwargs)

                if isinstance(result, str):
                    ctx["output_text"] = result
                elif hasattr(result, "content"):
                    ctx["output_text"] = str(result.content)

                if not isinstance(result, str) and hasattr(result, "usage"):
                    usage = getattr(result, "usage", None)
                    if usage:
                        ctx["input_tokens"] = getattr(usage, "input_tokens", 0)
                        ctx["output_tokens"] = getattr(usage, "output_tokens", 0)

                return result

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


# =============================================================================
# Integration with Letta
# =============================================================================

class LettaOpikIntegration:
    """
    Integrates Opik tracing with Letta agent calls.

    Automatically captures:
    - Agent messages (create, stream)
    - Memory operations (blocks, passages)
    - Tool calls
    """

    def __init__(self, opik_client: Optional[OpikClient] = None):
        self.opik = opik_client or get_opik_client()
        self._wrapped_methods: set = set()

    def wrap_letta_client(self, letta_client: Any) -> Any:
        """
        Wrap a Letta client to automatically trace all operations.

        Usage:
            from letta_client import Letta
            from platform.core.opik_integration import LettaOpikIntegration

            client = Letta(api_key=..., base_url=...)
            integration = LettaOpikIntegration()
            client = integration.wrap_letta_client(client)
        """
        # Wrap messages.create
        if hasattr(letta_client, "agents") and hasattr(letta_client.agents, "messages"):
            original_create = letta_client.agents.messages.create

            @functools.wraps(original_create)
            def traced_create(agent_id: str, messages: list, **kwargs):
                start_time = time.perf_counter()
                error = None
                response = None

                try:
                    # Letta SDK requires keyword arguments
                    response = original_create(agent_id=agent_id, messages=messages, **kwargs)
                    return response
                except Exception as e:
                    error = str(e)
                    raise
                finally:
                    latency_ms = (time.perf_counter() - start_time) * 1000

                    # Extract info from response
                    output_text = None
                    if response and hasattr(response, "messages"):
                        for msg in response.messages:
                            if getattr(msg, "message_type", "") == "assistant_message":
                                output_text = getattr(msg, "content", "")
                                break

                    self.opik.create_trace(
                        model="letta-agent",
                        provider="letta",
                        operation="agent.messages.create",
                        input_text=json.dumps(messages)[:500] if messages else None,
                        output_text=output_text,
                        latency_ms=latency_ms,
                        error=error,
                        metadata={"agent_id": agent_id},
                    )

            letta_client.agents.messages.create = traced_create
            self._wrapped_methods.add("messages.create")

        return letta_client

    def trace_letta_operation(
        self,
        operation: str,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for tracing custom Letta operations."""
        return self.opik.trace_llm(
            model="letta-agent",
            provider="letta",
            operation=operation,
            metadata={"agent_id": agent_id, **(metadata or {})},
        )


# =============================================================================
# Global Instance
# =============================================================================

_opik_client: Optional[OpikClient] = None


def get_opik_client(config: Optional[OpikConfig] = None) -> OpikClient:
    """Get the global Opik client instance."""
    global _opik_client
    if _opik_client is None:
        _opik_client = OpikClient(config)
    return _opik_client


def configure_opik(
    api_key: Optional[str] = None,
    workspace: str = "default",
    project_name: str = "unleash-platform",
    **kwargs,
) -> OpikClient:
    """Configure and return the Opik client."""
    global _opik_client
    config = OpikConfig(
        api_key=api_key,
        workspace=workspace,
        project_name=project_name,
        **kwargs,
    )
    _opik_client = OpikClient(config)
    return _opik_client


def reset_opik() -> None:
    """Reset the global Opik client (for testing)."""
    global _opik_client
    _opik_client = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Config
    "OpikConfig",
    "OpikClientStatus",

    # Client
    "OpikClient",
    "get_opik_client",
    "configure_opik",
    "reset_opik",

    # Tracing
    "LLMTrace",
    "trace_llm_call",

    # Integrations
    "LettaOpikIntegration",
]
