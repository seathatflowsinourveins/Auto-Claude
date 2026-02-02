#!/usr/bin/env python3
"""
UNLEASH Code Intelligence Observability
=======================================
Part of the Unified Code Intelligence Architecture (5-Layer, 2026)

Provides metrics tracking and optional Opik integration for code intelligence ops.

Features:
- Track embedding generation latency and throughput
- Track semantic search operations
- Pipeline cost estimation
- Dashboard for metrics visibility

Usage:
    from core.code_intel_observability import (
        trace_embedding,
        trace_search,
        CodeIntelMetrics,
    )

    @trace_embedding
    def embed_code(code: str) -> list[float]:
        ...

Requirements:
    pip install structlog
    pip install opik  # Optional, for cloud tracing
"""

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, Callable, TypeVar

import structlog

# Configure logging
log = structlog.get_logger()

# Type variable for decorators
F = TypeVar("F", bound=Callable[..., Any])

# Constants
VOYAGE_COST_PER_1K_TOKENS = 0.0001  # Approximate Voyage-code-3 cost


@dataclass
class CodeIntelMetrics:
    """Metrics for code intelligence operations."""
    embeddings_generated: int = 0
    searches_performed: int = 0
    total_tokens_embedded: int = 0
    total_search_latency_ms: float = 0
    total_embedding_latency_ms: float = 0
    errors: int = 0
    start_time: datetime = field(default_factory=datetime.now)

    @property
    def avg_search_latency_ms(self) -> float:
        if self.searches_performed == 0:
            return 0
        return self.total_search_latency_ms / self.searches_performed

    @property
    def avg_embedding_latency_ms(self) -> float:
        if self.embeddings_generated == 0:
            return 0
        return self.total_embedding_latency_ms / self.embeddings_generated

    @property
    def estimated_cost_usd(self) -> float:
        return (self.total_tokens_embedded / 1000) * VOYAGE_COST_PER_1K_TOKENS


# Global metrics instance
_metrics = CodeIntelMetrics()


def get_metrics() -> CodeIntelMetrics:
    """Get the global metrics instance."""
    return _metrics


def reset_metrics() -> None:
    """Reset the global metrics instance."""
    global _metrics
    _metrics = CodeIntelMetrics()


def trace_embedding(func: F) -> F:
    """
    Decorator to trace embedding generation operations.

    Tracks:
    - Execution time
    - Token count (estimated)
    - Errors
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        error = None
        result = None

        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            error = e
            _metrics.errors += 1
            raise
        finally:
            elapsed_ms = (time.time() - start) * 1000
            _metrics.total_embedding_latency_ms += elapsed_ms
            _metrics.embeddings_generated += 1

            # Estimate tokens from input
            if args:
                input_text = args[0] if isinstance(args[0], str) else str(args[0])
                estimated_tokens = len(input_text.split())
                _metrics.total_tokens_embedded += estimated_tokens

            log.debug(
                "embedding_traced",
                latency_ms=round(elapsed_ms, 1),
                error=str(error) if error else None,
            )

    return wrapper  # type: ignore


def trace_search(func: F) -> F:
    """
    Decorator to trace semantic search operations.

    Tracks:
    - Query execution time
    - Result count
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        error = None
        result = None

        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            error = e
            _metrics.errors += 1
            raise
        finally:
            elapsed_ms = (time.time() - start) * 1000
            _metrics.total_search_latency_ms += elapsed_ms
            _metrics.searches_performed += 1

            # Extract query from args
            search_query = args[0] if args else kwargs.get("query", "unknown")

            log.debug(
                "search_traced",
                query=str(search_query)[:50],
                latency_ms=round(elapsed_ms, 1),
                results=len(result) if result else 0,
            )

    return wrapper  # type: ignore


def trace_pipeline(name: str) -> Callable[[F], F]:
    """
    Decorator factory for tracing pipeline operations.

    Usage:
        @trace_pipeline("embedding_pipeline")
        def run_pipeline():
            ...
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            error = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error = e
                raise
            finally:
                elapsed_s = time.time() - start

                log.info(
                    "pipeline_traced",
                    name=name,
                    duration_s=round(elapsed_s, 2),
                    embeddings=_metrics.embeddings_generated,
                    searches=_metrics.searches_performed,
                    errors=_metrics.errors,
                    error=str(error) if error else None,
                )

        return wrapper  # type: ignore
    return decorator


class CodeIntelDashboard:
    """
    Dashboard for code intelligence metrics.

    Provides real-time visibility into embedding pipeline and search performance.
    """

    def __init__(self):
        self.metrics = get_metrics()

    def summary(self) -> dict[str, Any]:
        """Get metrics summary."""
        return {
            "embeddings": {
                "count": self.metrics.embeddings_generated,
                "avg_latency_ms": round(self.metrics.avg_embedding_latency_ms, 1),
                "total_tokens": self.metrics.total_tokens_embedded,
            },
            "searches": {
                "count": self.metrics.searches_performed,
                "avg_latency_ms": round(self.metrics.avg_search_latency_ms, 1),
            },
            "errors": self.metrics.errors,
            "estimated_cost_usd": round(self.metrics.estimated_cost_usd, 4),
            "uptime_s": round((datetime.now() - self.metrics.start_time).total_seconds(), 0),
        }

    def print_summary(self):
        """Print metrics summary to console."""
        s = self.summary()
        print("\n" + "=" * 50)
        print("CODE INTELLIGENCE METRICS")
        print("=" * 50)
        print(f"Embeddings: {s['embeddings']['count']}")
        print(f"  Avg latency: {s['embeddings']['avg_latency_ms']}ms")
        print(f"  Total tokens: {s['embeddings']['total_tokens']}")
        print(f"Searches: {s['searches']['count']}")
        print(f"  Avg latency: {s['searches']['avg_latency_ms']}ms")
        print(f"Errors: {s['errors']}")
        print(f"Est. cost: ${s['estimated_cost_usd']}")
        print(f"Uptime: {s['uptime_s']}s")
        print("=" * 50)


def setup_opik(project_name: str = "unleash-code-intelligence") -> bool:
    """
    Configure Opik for AI observability (optional).

    Returns True if Opik was successfully configured.

    Usage with Opik decorated functions:
        import opik

        @opik.track(name="embed_code")
        def embed_code(code: str) -> list[float]:
            ...
    """
    try:
        import opik  # type: ignore[import-untyped]

        # Configure project via environment
        os.environ.setdefault("OPIK_PROJECT_NAME", project_name)

        # Try to configure (will use API key from env or local config)
        opik.configure()  # type: ignore[attr-defined]
        log.info("opik_configured", project=project_name)
        return True
    except ImportError:
        log.info("opik_not_installed", hint="pip install opik")
        return False
    except Exception as e:
        log.debug("opik_config_skipped", reason=str(e))
        return False


if __name__ == "__main__":
    # Demo usage
    print("Code Intelligence Observability Demo")
    print("-" * 40)

    # Setup Opik (optional)
    opik_available = setup_opik()
    print(f"Opik available: {opik_available}")

    # Simulate some operations
    @trace_embedding
    def mock_embed(text: str) -> list[float]:
        time.sleep(0.1)  # Simulate API call
        _ = text  # Use the parameter
        return [0.1] * 1024

    @trace_search
    def mock_search(query: str) -> list[dict]:
        time.sleep(0.05)  # Simulate search
        _ = query  # Use the parameter
        return [{"score": 0.9, "content": "test"}]

    # Run operations
    for i in range(5):
        mock_embed(f"def function_{i}(): pass")
        mock_search(f"query {i}")

    # Print dashboard
    dashboard = CodeIntelDashboard()
    dashboard.print_summary()
