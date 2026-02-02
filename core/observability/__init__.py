#!/usr/bin/env python3
"""
Observability Layer - No Stubs, Explicit Failures Only
Part of V33 Architecture (Layer 4) - Phase 9 Production Fix.

This module provides observability integrations. Each SDK must be:
1. Explicitly installed and configured
2. Raise clear errors if unavailable
3. No silent fallbacks or stub patterns

Available SDKs:
- langfuse: LLM tracing and cost tracking
- opik: Evaluation and experiment tracking
- phoenix: Real-time monitoring with OpenTelemetry
- deepeval: Comprehensive LLM testing
- ragas: RAG pipeline evaluation
- logfire: Structured logging (Pydantic)

Usage:
    from core.observability import (
        get_langfuse_tracer,
        get_opik_client,
        SDKNotAvailableError,
    )

    try:
        tracer = get_langfuse_tracer()
    except SDKNotAvailableError as e:
        print(f"Install SDK: {e.install_cmd}")
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# CUSTOM EXCEPTIONS - Explicit Failure Pattern
# =============================================================================

class SDKNotAvailableError(Exception):
    """Raised when required SDK is not installed."""

    def __init__(self, sdk_name: str, install_cmd: str, docs_url: str = None):
        self.sdk_name = sdk_name
        self.install_cmd = install_cmd
        self.docs_url = docs_url
        msg = f"""
============================================================
SDK NOT AVAILABLE: {sdk_name}
============================================================

Install with:
  {install_cmd}

Documentation:
  {docs_url or f'https://docs.unleash.dev/sdks/{sdk_name}'}

This error is INTENTIONAL. Unleash does not use silent fallbacks.
============================================================
"""
        super().__init__(msg)


class SDKConfigurationError(Exception):
    """Raised when SDK is installed but misconfigured."""

    def __init__(self, sdk_name: str, missing_config: List[str], example: str = None):
        self.sdk_name = sdk_name
        self.missing_config = missing_config
        config_list = '\n  '.join(f"- {c}" for c in missing_config)
        msg = f"""
============================================================
SDK CONFIGURATION ERROR: {sdk_name}
============================================================

Missing configuration:
  {config_list}

Example .env configuration:
{example or 'See documentation for configuration examples.'}
============================================================
"""
        super().__init__(msg)


# =============================================================================
# SDK AVAILABILITY CHECKS - Import-time validation
# =============================================================================

# Langfuse - Tracing
LANGFUSE_AVAILABLE = False
LANGFUSE_ERROR = None
try:
    from langfuse import Langfuse
    from langfuse.decorators import observe as langfuse_observe
    LANGFUSE_AVAILABLE = True
except Exception as e:
    # Catch ALL exceptions - including pydantic compatibility errors on Python 3.14+
    LANGFUSE_ERROR = str(e)

# Phoenix/Arize - Evaluation
PHOENIX_AVAILABLE = False
PHOENIX_ERROR = None
try:
    import phoenix as px
    PHOENIX_AVAILABLE = True
except Exception as e:
    PHOENIX_ERROR = str(e)

# Opik - Experiment Tracking
OPIK_AVAILABLE = False
OPIK_ERROR = None
try:
    import opik
    from opik import track as opik_track, Opik as OpikClient
    OPIK_AVAILABLE = True
except Exception as e:
    OPIK_ERROR = str(e)

# DeepEval - LLM Evaluation
DEEPEVAL_AVAILABLE = False
DEEPEVAL_ERROR = None
try:
    from deepeval import evaluate as deepeval_evaluate
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except Exception as e:
    DEEPEVAL_ERROR = str(e)

# RAGAS - RAG Evaluation
RAGAS_AVAILABLE = False
RAGAS_ERROR = None
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    RAGAS_AVAILABLE = True
except Exception as e:
    RAGAS_ERROR = str(e)

# Logfire - Structured Logging
LOGFIRE_AVAILABLE = False
LOGFIRE_ERROR = None
try:
    import logfire
    LOGFIRE_AVAILABLE = True
except Exception as e:
    LOGFIRE_ERROR = str(e)

# OpenTelemetry - Tracing
OPENTELEMETRY_AVAILABLE = False
OPENTELEMETRY_ERROR = None
try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    OPENTELEMETRY_AVAILABLE = True
except Exception as e:
    OPENTELEMETRY_ERROR = str(e)


# =============================================================================
# OBSERVABILITY LAYER AVAILABILITY
# =============================================================================
# Layer is available if at least one SDK is available
OBSERVABILITY_AVAILABLE = any([
    LANGFUSE_AVAILABLE,
    PHOENIX_AVAILABLE,
    OPIK_AVAILABLE,
    DEEPEVAL_AVAILABLE,
    RAGAS_AVAILABLE,
    LOGFIRE_AVAILABLE,
    OPENTELEMETRY_AVAILABLE,
])


# =============================================================================
# TRACING PROVIDERS - No Stubs
# =============================================================================

def get_langfuse_tracer() -> "Langfuse":
    """Get Langfuse tracer. Raises explicit error if unavailable."""
    if not LANGFUSE_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="langfuse",
            install_cmd="pip install langfuse>=2.0.0",
            docs_url="https://langfuse.com/docs/sdk/python"
        )

    # Check configuration
    required_config = []
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        required_config.append("LANGFUSE_PUBLIC_KEY")
    if not os.getenv("LANGFUSE_SECRET_KEY"):
        required_config.append("LANGFUSE_SECRET_KEY")

    if required_config:
        raise SDKConfigurationError(
            sdk_name="langfuse",
            missing_config=required_config,
            example="""
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com  # Optional
"""
        )

    return Langfuse()


def get_langfuse_observe():
    """Get Langfuse observe decorator. Raises explicit error if unavailable."""
    if not LANGFUSE_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="langfuse",
            install_cmd="pip install langfuse>=2.0.0",
            docs_url="https://langfuse.com/docs/sdk/python"
        )
    return langfuse_observe


def get_phoenix_client() -> "px.Client":
    """Get Phoenix client. Raises explicit error if unavailable."""
    if not PHOENIX_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="phoenix (arize-phoenix)",
            install_cmd="pip install arize-phoenix>=4.0.0",
            docs_url="https://docs.arize.com/phoenix"
        )

    return px.Client()


def get_opik_client() -> "OpikClient":
    """Get Opik client. Raises explicit error if unavailable."""
    if not OPIK_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="opik",
            install_cmd="pip install opik>=1.0.0",
            docs_url="https://www.comet.com/docs/opik/"
        )

    api_key = os.getenv("OPIK_API_KEY")
    if not api_key:
        raise SDKConfigurationError(
            sdk_name="opik",
            missing_config=["OPIK_API_KEY"],
            example="OPIK_API_KEY=your-api-key-here"
        )

    return OpikClient(api_key=api_key)


def get_opik_track():
    """Get Opik track decorator. Raises explicit error if unavailable."""
    if not OPIK_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="opik",
            install_cmd="pip install opik>=1.0.0",
            docs_url="https://www.comet.com/docs/opik/"
        )
    return opik_track


def get_deepeval_evaluator():
    """Get DeepEval evaluate function. Raises explicit error if unavailable."""
    if not DEEPEVAL_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="deepeval",
            install_cmd="pip install deepeval>=1.0.0",
            docs_url="https://docs.confident-ai.com/"
        )

    return deepeval_evaluate


def get_deepeval_metrics():
    """Get DeepEval metrics. Raises explicit error if unavailable."""
    if not DEEPEVAL_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="deepeval",
            install_cmd="pip install deepeval>=1.0.0",
            docs_url="https://docs.confident-ai.com/"
        )

    return {
        "AnswerRelevancyMetric": AnswerRelevancyMetric,
        "FaithfulnessMetric": FaithfulnessMetric,
        "LLMTestCase": LLMTestCase,
    }


def get_ragas_evaluator():
    """Get RAGAS evaluate function. Raises explicit error if unavailable."""
    if not RAGAS_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="ragas",
            install_cmd="pip install ragas>=0.2.0",
            docs_url="https://docs.ragas.io/"
        )

    return ragas_evaluate


def get_ragas_metrics():
    """Get RAGAS metrics. Raises explicit error if unavailable."""
    if not RAGAS_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="ragas",
            install_cmd="pip install ragas>=0.2.0",
            docs_url="https://docs.ragas.io/"
        )

    return {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }


def get_logfire_logger() -> "logfire":
    """Get Logfire logger. Raises explicit error if unavailable."""
    if not LOGFIRE_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="logfire",
            install_cmd="pip install logfire>=0.30.0",
            docs_url="https://logfire.pydantic.dev/docs/"
        )

    token = os.getenv("LOGFIRE_TOKEN")
    if not token:
        raise SDKConfigurationError(
            sdk_name="logfire",
            missing_config=["LOGFIRE_TOKEN"],
            example="LOGFIRE_TOKEN=your-token-here"
        )

    logfire.configure()
    return logfire


def get_opentelemetry_tracer(service_name: str = "unleash") -> "otel_trace.Tracer":
    """Get OpenTelemetry tracer. Raises explicit error if unavailable."""
    if not OPENTELEMETRY_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="opentelemetry",
            install_cmd="pip install opentelemetry-api opentelemetry-sdk>=1.20.0",
            docs_url="https://opentelemetry.io/docs/instrumentation/python/"
        )

    provider = TracerProvider()
    otel_trace.set_tracer_provider(provider)
    return otel_trace.get_tracer(service_name)


# =============================================================================
# SDK STATUS REPORT - For diagnostics
# =============================================================================

@dataclass
class SDKStatus:
    """Status of a single SDK."""
    name: str
    available: bool
    configured: bool
    error: Optional[str] = None
    install_cmd: str = ""


def get_observability_status() -> Dict[str, SDKStatus]:
    """Get status of all observability SDKs."""
    return {
        "langfuse": SDKStatus(
            name="langfuse",
            available=LANGFUSE_AVAILABLE,
            configured=bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")),
            error=LANGFUSE_ERROR,
            install_cmd="pip install langfuse>=2.0.0"
        ),
        "phoenix": SDKStatus(
            name="phoenix",
            available=PHOENIX_AVAILABLE,
            configured=PHOENIX_AVAILABLE,  # Phoenix works without config
            error=PHOENIX_ERROR,
            install_cmd="pip install arize-phoenix>=4.0.0"
        ),
        "opik": SDKStatus(
            name="opik",
            available=OPIK_AVAILABLE,
            configured=bool(os.getenv("OPIK_API_KEY")),
            error=OPIK_ERROR,
            install_cmd="pip install opik>=1.0.0"
        ),
        "deepeval": SDKStatus(
            name="deepeval",
            available=DEEPEVAL_AVAILABLE,
            configured=DEEPEVAL_AVAILABLE,  # DeepEval works locally
            error=DEEPEVAL_ERROR,
            install_cmd="pip install deepeval>=1.0.0"
        ),
        "ragas": SDKStatus(
            name="ragas",
            available=RAGAS_AVAILABLE,
            configured=RAGAS_AVAILABLE,
            error=RAGAS_ERROR,
            install_cmd="pip install ragas>=0.2.0"
        ),
        "logfire": SDKStatus(
            name="logfire",
            available=LOGFIRE_AVAILABLE,
            configured=bool(os.getenv("LOGFIRE_TOKEN")),
            error=LOGFIRE_ERROR,
            install_cmd="pip install logfire>=0.30.0"
        ),
        "opentelemetry": SDKStatus(
            name="opentelemetry",
            available=OPENTELEMETRY_AVAILABLE,
            configured=OPENTELEMETRY_AVAILABLE,
            error=OPENTELEMETRY_ERROR,
            install_cmd="pip install opentelemetry-api opentelemetry-sdk>=1.20.0"
        ),
    }


def get_available_sdks() -> Dict[str, bool]:
    """Get availability status of all observability SDKs."""
    return {
        "langfuse": LANGFUSE_AVAILABLE,
        "phoenix": PHOENIX_AVAILABLE,
        "opik": OPIK_AVAILABLE,
        "deepeval": DEEPEVAL_AVAILABLE,
        "ragas": RAGAS_AVAILABLE,
        "logfire": LOGFIRE_AVAILABLE,
        "opentelemetry": OPENTELEMETRY_AVAILABLE,
    }


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__ = [
    # Exceptions
    "SDKNotAvailableError",
    "SDKConfigurationError",
    # Layer availability (for unified access)
    "OBSERVABILITY_AVAILABLE",
    # Individual SDK availability flags (for conditional logic in tests)
    "LANGFUSE_AVAILABLE",
    "PHOENIX_AVAILABLE",
    "OPIK_AVAILABLE",
    "DEEPEVAL_AVAILABLE",
    "RAGAS_AVAILABLE",
    "LOGFIRE_AVAILABLE",
    "OPENTELEMETRY_AVAILABLE",
    # Getter functions (raise on unavailable)
    "get_langfuse_tracer",
    "get_langfuse_observe",
    "get_phoenix_client",
    "get_opik_client",
    "get_opik_track",
    "get_deepeval_evaluator",
    "get_deepeval_metrics",
    "get_ragas_evaluator",
    "get_ragas_metrics",
    "get_logfire_logger",
    "get_opentelemetry_tracer",
    # Status reporting
    "get_observability_status",
    "get_available_sdks",
    "SDKStatus",
]
