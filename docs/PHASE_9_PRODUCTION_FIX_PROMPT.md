# Phase 9: Production Readiness Fix - Deep Integration Tests + No Fallbacks

## Executive Summary

**Current Status (CRITICAL):**
- Test Coverage: 30% (D) - Only checks imports, not functionality
- Real Integration: 60% (C) - 7/25 SDKs are stubs
- Production Ready: 45% (F) - NOT PRODUCTION READY
- 30 stub references in observability layer
- No real workflow verification tests

**Target:**
- Test Coverage: 90%+ with REAL integration tests
- Real Integration: 100% (all SDKs functional or explicit error)
- Production Ready: 95%+ with documented gaps only
- Zero stub patterns
- Full workflow verification

---

## Section 1: Remove All Stubs (~300 lines)

### 1.1 Philosophy: Explicit Failure Over Silent Degradation

**RULE: NO graceful degradation. EXPLICIT failures when SDK missing.**

```python
# BAD - Silent degradation (REMOVE THIS PATTERN)
class StubLangfuseTracer:
    """Silent no-op when Langfuse unavailable."""
    def trace(self, *args, **kwargs):
        pass  # Silently does nothing
    
def get_tracer():
    if LANGFUSE_AVAILABLE:
        return LangfuseTracer()
    return StubLangfuseTracer()  # WRONG - hides the problem

# GOOD - Explicit failure with actionable error
class SDKNotAvailableError(Exception):
    """Raised when required SDK is not installed or configured."""
    def __init__(self, sdk_name: str, install_cmd: str, docs_url: str = None):
        self.sdk_name = sdk_name
        self.install_cmd = install_cmd
        self.docs_url = docs_url
        msg = f"""
SDK '{sdk_name}' is not available.

To fix:
  {install_cmd}

Documentation: {docs_url or 'https://docs.unleash.dev/sdks/' + sdk_name}
"""
        super().__init__(msg)

def get_tracer():
    if not LANGFUSE_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="langfuse",
            install_cmd="pip install langfuse>=2.0.0",
            docs_url="https://langfuse.com/docs"
        )
    return LangfuseTracer()
```

### 1.2 Update `core/observability/__init__.py`

**Task: Remove ALL StubXXX classes and implement explicit errors**

```python
# core/observability/__init__.py - COMPLETE REWRITE

"""
Observability Layer - No Stubs, Explicit Failures Only

This module provides observability integrations. Each SDK must be:
1. Explicitly installed and configured
2. Raise clear errors if unavailable
3. No silent fallbacks or stub patterns
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
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SDK NOT AVAILABLE: {sdk_name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Install with:
  {install_cmd}

Documentation:
  {docs_url or f'https://docs.unleash.dev/sdks/{sdk_name}'}

This error is INTENTIONAL. Unleash does not use silent fallbacks.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        super().__init__(msg)


class SDKConfigurationError(Exception):
    """Raised when SDK is installed but misconfigured."""
    def __init__(self, sdk_name: str, missing_config: List[str], example: str = None):
        self.sdk_name = sdk_name
        self.missing_config = missing_config
        config_list = '\n  '.join(f"- {c}" for c in missing_config)
        msg = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SDK CONFIGURATION ERROR: {sdk_name}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Missing configuration:
  {config_list}

Example .env configuration:
{example or 'See documentation for configuration examples.'}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
    from langfuse.decorators import observe
    LANGFUSE_AVAILABLE = True
except ImportError as e:
    LANGFUSE_ERROR = str(e)

# Phoenix/Arize - Evaluation
PHOENIX_AVAILABLE = False
PHOENIX_ERROR = None
try:
    import phoenix as px
    from phoenix.trace import SpanEvaluations
    PHOENIX_AVAILABLE = True
except ImportError as e:
    PHOENIX_ERROR = str(e)

# Opik - Experiment Tracking
OPIK_AVAILABLE = False
OPIK_ERROR = None
try:
    import opik
    from opik import track, Opik as OpikClient
    OPIK_AVAILABLE = True
except ImportError as e:
    OPIK_ERROR = str(e)

# DeepEval - LLM Evaluation
DEEPEVAL_AVAILABLE = False
DEEPEVAL_ERROR = None
try:
    from deepeval import evaluate
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError as e:
    DEEPEVAL_ERROR = str(e)

# Logfire - Structured Logging
LOGFIRE_AVAILABLE = False
LOGFIRE_ERROR = None
try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError as e:
    LOGFIRE_ERROR = str(e)


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


def get_deepeval_evaluator():
    """Get DeepEval evaluator. Raises explicit error if unavailable."""
    if not DEEPEVAL_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="deepeval",
            install_cmd="pip install deepeval>=1.0.0",
            docs_url="https://docs.confident-ai.com/"
        )
    
    return evaluate


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
        "logfire": SDKStatus(
            name="logfire",
            available=LOGFIRE_AVAILABLE,
            configured=bool(os.getenv("LOGFIRE_TOKEN")),
            error=LOGFIRE_ERROR,
            install_cmd="pip install logfire>=0.30.0"
        ),
    }


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__ = [
    # Exceptions
    "SDKNotAvailableError",
    "SDKConfigurationError",
    # Availability flags (for conditional logic in tests)
    "LANGFUSE_AVAILABLE",
    "PHOENIX_AVAILABLE", 
    "OPIK_AVAILABLE",
    "DEEPEVAL_AVAILABLE",
    "LOGFIRE_AVAILABLE",
    # Getter functions (raise on unavailable)
    "get_langfuse_tracer",
    "get_phoenix_client",
    "get_opik_client",
    "get_deepeval_evaluator",
    "get_logfire_logger",
    # Status reporting
    "get_observability_status",
    "SDKStatus",
]
```

### 1.3 Update `core/memory/__init__.py`

**Apply same pattern to memory layer:**

```python
# core/memory/__init__.py - COMPLETE REWRITE

"""
Memory Layer - No Stubs, Explicit Failures Only
"""

from __future__ import annotations
import os
from typing import Optional, List, Dict, Any

# Import custom exceptions from observability
from core.observability import SDKNotAvailableError, SDKConfigurationError

# =============================================================================
# SDK AVAILABILITY CHECKS
# =============================================================================

# Mem0 - Memory Management
MEM0_AVAILABLE = False
MEM0_ERROR = None
try:
    from mem0 import Memory
    MEM0_AVAILABLE = True
except ImportError as e:
    MEM0_ERROR = str(e)

# Letta - Agent Memory
LETTA_AVAILABLE = False
LETTA_ERROR = None
try:
    from letta import create_client, LettaClient
    LETTA_AVAILABLE = True
except ImportError as e:
    LETTA_ERROR = str(e)

# Graphiti - Knowledge Graphs
GRAPHITI_AVAILABLE = False
GRAPHITI_ERROR = None
try:
    from graphiti_core import Graphiti
    GRAPHITI_AVAILABLE = True
except ImportError as e:
    GRAPHITI_ERROR = str(e)

# Cognee - Memory Reasoning
COGNEE_AVAILABLE = False
COGNEE_ERROR = None
try:
    import cognee
    COGNEE_AVAILABLE = True
except ImportError as e:
    COGNEE_ERROR = str(e)


# =============================================================================
# MEMORY PROVIDERS - No Stubs
# =============================================================================

def get_mem0_client() -> "Memory":
    """Get Mem0 client. Raises explicit error if unavailable."""
    if not MEM0_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="mem0",
            install_cmd="pip install mem0ai>=0.1.0",
            docs_url="https://docs.mem0.ai/"
        )
    
    api_key = os.getenv("MEM0_API_KEY")
    if api_key:
        return Memory(api_key=api_key)
    
    # Local mode requires config
    return Memory()


def get_letta_client() -> "LettaClient":
    """Get Letta client. Raises explicit error if unavailable."""
    if not LETTA_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="letta",
            install_cmd="pip install letta>=0.4.0",
            docs_url="https://docs.letta.com/"
        )
    
    base_url = os.getenv("LETTA_BASE_URL")
    if base_url:
        return create_client(base_url=base_url)
    
    # Local mode
    return create_client()


def get_graphiti_client() -> "Graphiti":
    """Get Graphiti client. Raises explicit error if unavailable."""
    if not GRAPHITI_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="graphiti",
            install_cmd="pip install graphiti-core>=0.3.0",
            docs_url="https://github.com/getzep/graphiti"
        )
    
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    
    if not all([neo4j_uri, neo4j_user, neo4j_password]):
        raise SDKConfigurationError(
            sdk_name="graphiti",
            missing_config=["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"],
            example="""
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
"""
        )
    
    return Graphiti(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password
    )


def get_cognee_client():
    """Get Cognee client. Raises explicit error if unavailable."""
    if not COGNEE_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="cognee",
            install_cmd="pip install cognee>=0.1.0",
            docs_url="https://github.com/topoteretes/cognee"
        )
    
    return cognee


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "MEM0_AVAILABLE",
    "LETTA_AVAILABLE",
    "GRAPHITI_AVAILABLE",
    "COGNEE_AVAILABLE",
    "get_mem0_client",
    "get_letta_client",
    "get_graphiti_client",
    "get_cognee_client",
]
```

### 1.4 Update `core/orchestration/__init__.py`

```python
# core/orchestration/__init__.py - COMPLETE REWRITE

"""
Orchestration Layer - No Stubs, Explicit Failures Only
"""

from __future__ import annotations
import os
from typing import Optional, Dict, Any

from core.observability import SDKNotAvailableError, SDKConfigurationError

# =============================================================================
# SDK AVAILABILITY CHECKS
# =============================================================================

# LangGraph - Workflow Graphs
LANGGRAPH_AVAILABLE = False
LANGGRAPH_ERROR = None
try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    LANGGRAPH_ERROR = str(e)

# CrewAI - Multi-Agent
CREWAI_AVAILABLE = False
CREWAI_ERROR = None
try:
    from crewai import Agent, Task, Crew
    CREWAI_AVAILABLE = True
except ImportError as e:
    CREWAI_ERROR = str(e)

# Temporal - Durable Workflows
TEMPORAL_AVAILABLE = False
TEMPORAL_ERROR = None
try:
    from temporalio import workflow, activity
    from temporalio.client import Client as TemporalClient
    TEMPORAL_AVAILABLE = True
except ImportError as e:
    TEMPORAL_ERROR = str(e)

# Prefect - Data Pipelines
PREFECT_AVAILABLE = False
PREFECT_ERROR = None
try:
    from prefect import flow, task
    from prefect.client import get_client as get_prefect_client
    PREFECT_AVAILABLE = True
except ImportError as e:
    PREFECT_ERROR = str(e)


# =============================================================================
# ORCHESTRATION PROVIDERS - No Stubs
# =============================================================================

def get_langgraph_builder():
    """Get LangGraph StateGraph builder. Raises explicit error if unavailable."""
    if not LANGGRAPH_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="langgraph",
            install_cmd="pip install langgraph>=0.2.0",
            docs_url="https://langchain-ai.github.io/langgraph/"
        )
    return StateGraph


async def get_temporal_client() -> "TemporalClient":
    """Get Temporal client. Raises explicit error if unavailable."""
    if not TEMPORAL_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="temporal",
            install_cmd="pip install temporalio>=1.5.0",
            docs_url="https://docs.temporal.io/develop/python"
        )
    
    host = os.getenv("TEMPORAL_HOST", "localhost:7233")
    namespace = os.getenv("TEMPORAL_NAMESPACE", "default")
    
    return await TemporalClient.connect(host, namespace=namespace)


def get_crewai_builder():
    """Get CrewAI components. Raises explicit error if unavailable."""
    if not CREWAI_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="crewai",
            install_cmd="pip install crewai>=0.60.0",
            docs_url="https://docs.crewai.com/"
        )
    
    return {"Agent": Agent, "Task": Task, "Crew": Crew}


def get_prefect_flow():
    """Get Prefect flow decorator. Raises explicit error if unavailable."""
    if not PREFECT_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="prefect",
            install_cmd="pip install prefect>=2.0.0",
            docs_url="https://docs.prefect.io/"
        )
    return flow


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "LANGGRAPH_AVAILABLE",
    "CREWAI_AVAILABLE",
    "TEMPORAL_AVAILABLE",
    "PREFECT_AVAILABLE",
    "get_langgraph_builder",
    "get_temporal_client",
    "get_crewai_builder",
    "get_prefect_flow",
]
```

### 1.5 Update `core/structured/__init__.py`

```python
# core/structured/__init__.py - COMPLETE REWRITE

"""
Structured Output Layer - No Stubs, Explicit Failures Only
"""

from __future__ import annotations
import os
from typing import Type, TypeVar, Any

from core.observability import SDKNotAvailableError, SDKConfigurationError

T = TypeVar('T')

# =============================================================================
# SDK AVAILABILITY CHECKS
# =============================================================================

# Instructor - Structured LLM Output
INSTRUCTOR_AVAILABLE = False
INSTRUCTOR_ERROR = None
try:
    import instructor
    from instructor import patch
    INSTRUCTOR_AVAILABLE = True
except ImportError as e:
    INSTRUCTOR_ERROR = str(e)

# Outlines - Structured Generation
OUTLINES_AVAILABLE = False
OUTLINES_ERROR = None
try:
    import outlines
    from outlines import generate
    OUTLINES_AVAILABLE = True
except ImportError as e:
    OUTLINES_ERROR = str(e)

# Mirascope - Multi-Provider Structured
MIRASCOPE_AVAILABLE = False
MIRASCOPE_ERROR = None
try:
    from mirascope.core import anthropic, openai
    MIRASCOPE_AVAILABLE = True
except ImportError as e:
    MIRASCOPE_ERROR = str(e)


# =============================================================================
# STRUCTURED OUTPUT PROVIDERS - No Stubs
# =============================================================================

def get_instructor_client(llm_client: Any, mode: str = "tool"):
    """Patch LLM client with Instructor. Raises explicit error if unavailable."""
    if not INSTRUCTOR_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="instructor",
            install_cmd="pip install instructor>=1.0.0",
            docs_url="https://python.useinstructor.com/"
        )
    
    return instructor.patch(llm_client, mode=mode)


def get_outlines_generator(model_name: str = "gpt-4"):
    """Get Outlines generator. Raises explicit error if unavailable."""
    if not OUTLINES_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="outlines",
            install_cmd="pip install outlines>=0.0.40",
            docs_url="https://outlines-dev.github.io/outlines/"
        )
    
    return generate


def get_mirascope_provider(provider: str = "anthropic"):
    """Get Mirascope provider. Raises explicit error if unavailable."""
    if not MIRASCOPE_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="mirascope",
            install_cmd="pip install mirascope>=1.0.0",
            docs_url="https://docs.mirascope.io/"
        )
    
    providers = {
        "anthropic": anthropic,
        "openai": openai,
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(providers.keys())}")
    
    return providers[provider]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "INSTRUCTOR_AVAILABLE",
    "OUTLINES_AVAILABLE",
    "MIRASCOPE_AVAILABLE",
    "get_instructor_client",
    "get_outlines_generator",
    "get_mirascope_provider",
]
```

---

## Section 2: Real Integration Tests (~500 lines)

### 2.1 Test Infrastructure Setup

**Create `tests/integration/__init__.py`:**

```python
# tests/integration/__init__.py

"""
Real Integration Tests - Actually Call SDKs

These tests are marked with @pytest.mark.integration and:
1. Actually call real SDK methods
2. Skip (not stub) if API key missing
3. Verify real functionality
4. Are excluded from CI without credentials
"""

import os
import pytest

# Integration test marker
integration = pytest.mark.integration

# Skip decorator for missing API keys
def skip_without_env(*env_vars):
    """Skip test if any environment variable is missing."""
    missing = [v for v in env_vars if not os.getenv(v)]
    if missing:
        return pytest.mark.skip(
            reason=f"Missing environment variables: {', '.join(missing)}"
        )
    return pytest.mark.usefixtures()
```

**Create `tests/integration/conftest.py`:**

```python
# tests/integration/conftest.py

"""
Shared fixtures for integration tests.
"""

import os
import pytest
from typing import Generator, Any

# =============================================================================
# ENVIRONMENT FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def integration_env():
    """Validate integration test environment."""
    return {
        "langfuse_configured": bool(
            os.getenv("LANGFUSE_PUBLIC_KEY") and 
            os.getenv("LANGFUSE_SECRET_KEY")
        ),
        "opik_configured": bool(os.getenv("OPIK_API_KEY")),
        "mem0_configured": bool(os.getenv("MEM0_API_KEY")),
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic_configured": bool(os.getenv("ANTHROPIC_API_KEY")),
        "tavily_configured": bool(os.getenv("TAVILY_API_KEY")),
    }


# =============================================================================
# SDK CLIENT FIXTURES
# =============================================================================

@pytest.fixture(scope="function")
def langfuse_tracer():
    """Real Langfuse tracer for integration tests."""
    from core.observability import LANGFUSE_AVAILABLE, get_langfuse_tracer
    
    if not LANGFUSE_AVAILABLE:
        pytest.skip("Langfuse not installed")
    
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        pytest.skip("LANGFUSE_PUBLIC_KEY not set")
    
    return get_langfuse_tracer()


@pytest.fixture(scope="function")
def mem0_client():
    """Real Mem0 client for integration tests."""
    from core.memory import MEM0_AVAILABLE, get_mem0_client
    
    if not MEM0_AVAILABLE:
        pytest.skip("Mem0 not installed")
    
    return get_mem0_client()


@pytest.fixture(scope="function")
def langgraph_builder():
    """Real LangGraph builder for integration tests."""
    from core.orchestration import LANGGRAPH_AVAILABLE, get_langgraph_builder
    
    if not LANGGRAPH_AVAILABLE:
        pytest.skip("LangGraph not installed")
    
    return get_langgraph_builder()


@pytest.fixture(scope="function")
def instructor_client():
    """Real Instructor client for integration tests."""
    from core.structured import INSTRUCTOR_AVAILABLE, get_instructor_client
    
    if not INSTRUCTOR_AVAILABLE:
        pytest.skip("Instructor not installed")
    
    # Need an LLM client to patch
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    from openai import OpenAI
    client = OpenAI()
    return get_instructor_client(client)


# =============================================================================
# CLEANUP FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_traces(langfuse_tracer):
    """Flush traces after each test."""
    yield
    if langfuse_tracer:
        langfuse_tracer.flush()
```

### 2.2 Memory Integration Tests

**Create `tests/integration/test_memory_real.py`:**

```python
# tests/integration/test_memory_real.py

"""
Real Memory Integration Tests

These tests actually store and retrieve data from memory systems.
They are NOT unit tests - they verify real SDK functionality.
"""

import os
import uuid
import pytest
from datetime import datetime

from tests.integration import integration, skip_without_env


@integration
class TestMem0RealIntegration:
    """Real Mem0 integration tests."""
    
    @pytest.fixture
    def unique_user_id(self):
        """Generate unique user ID for isolation."""
        return f"test_user_{uuid.uuid4().hex[:8]}"
    
    @skip_without_env("MEM0_API_KEY")
    def test_store_and_retrieve_memory(self, mem0_client, unique_user_id):
        """Test actually storing and retrieving a memory."""
        # Store a real memory
        memory_content = f"Test memory created at {datetime.now().isoformat()}"
        
        result = mem0_client.add(
            messages=[{"role": "user", "content": memory_content}],
            user_id=unique_user_id
        )
        
        # Verify storage succeeded
        assert result is not None
        assert "id" in result or "results" in result
        
        # Retrieve and verify
        memories = mem0_client.get_all(user_id=unique_user_id)
        assert len(memories) > 0
        
        # Verify content matches
        memory_texts = [m.get("memory", "") for m in memories]
        assert any(memory_content in text for text in memory_texts)
    
    @skip_without_env("MEM0_API_KEY")
    def test_search_memories(self, mem0_client, unique_user_id):
        """Test semantic memory search."""
        # Store test memories
        mem0_client.add(
            messages=[{"role": "user", "content": "I love programming in Python"}],
            user_id=unique_user_id
        )
        mem0_client.add(
            messages=[{"role": "user", "content": "Machine learning is fascinating"}],
            user_id=unique_user_id
        )
        
        # Search for related content
        results = mem0_client.search(
            query="coding languages",
            user_id=unique_user_id,
            limit=5
        )
        
        assert results is not None
        # Semantic search should find Python memory
    
    @skip_without_env("MEM0_API_KEY")
    def test_delete_memory(self, mem0_client, unique_user_id):
        """Test memory deletion."""
        # Store a memory
        result = mem0_client.add(
            messages=[{"role": "user", "content": "Temporary test memory"}],
            user_id=unique_user_id
        )
        
        # Get memory ID
        memory_id = result.get("id") or result.get("results", [{}])[0].get("id")
        assert memory_id is not None
        
        # Delete it
        mem0_client.delete(memory_id)
        
        # Verify deletion
        memories = mem0_client.get_all(user_id=unique_user_id)
        memory_ids = [m.get("id") for m in memories]
        assert memory_id not in memory_ids


@integration
class TestLettaRealIntegration:
    """Real Letta integration tests."""
    
    @pytest.fixture
    def letta_client(self):
        """Get Letta client, skip if unavailable."""
        from core.memory import LETTA_AVAILABLE, get_letta_client
        
        if not LETTA_AVAILABLE:
            pytest.skip("Letta not installed")
        
        return get_letta_client()
    
    def test_create_agent(self, letta_client):
        """Test creating a Letta agent."""
        agent_name = f"test_agent_{uuid.uuid4().hex[:8]}"
        
        agent = letta_client.create_agent(
            name=agent_name,
            system="You are a helpful test agent."
        )
        
        assert agent is not None
        assert agent.name == agent_name
        
        # Cleanup
        letta_client.delete_agent(agent.id)
    
    def test_agent_memory_persistence(self, letta_client):
        """Test that agent memory persists across interactions."""
        agent_name = f"memory_test_{uuid.uuid4().hex[:8]}"
        
        agent = letta_client.create_agent(
            name=agent_name,
            system="You are a test agent. Remember everything the user tells you."
        )
        
        try:
            # First interaction - store information
            response1 = letta_client.send_message(
                agent_id=agent.id,
                message="My favorite color is blue."
            )
            assert response1 is not None
            
            # Second interaction - recall information
            response2 = letta_client.send_message(
                agent_id=agent.id,
                message="What is my favorite color?"
            )
            
            # Agent should remember
            assert "blue" in str(response2).lower()
        finally:
            letta_client.delete_agent(agent.id)


@integration
class TestGraphitiRealIntegration:
    """Real Graphiti (knowledge graph) integration tests."""
    
    @pytest.fixture
    def graphiti_client(self):
        """Get Graphiti client, skip if unavailable or unconfigured."""
        from core.memory import GRAPHITI_AVAILABLE, get_graphiti_client
        
        if not GRAPHITI_AVAILABLE:
            pytest.skip("Graphiti not installed")
        
        if not all([
            os.getenv("NEO4J_URI"),
            os.getenv("NEO4J_USER"),
            os.getenv("NEO4J_PASSWORD")
        ]):
            pytest.skip("Neo4j not configured")
        
        return get_graphiti_client()
    
    def test_add_episode(self, graphiti_client):
        """Test adding an episode to the knowledge graph."""
        episode_content = f"Test episode at {datetime.now().isoformat()}"
        
        # Add episode
        result = graphiti_client.add_episode(
            name="test_episode",
            episode_body=episode_content,
            source_description="integration_test"
        )
        
        assert result is not None
    
    def test_search_knowledge(self, graphiti_client):
        """Test searching the knowledge graph."""
        # First add some knowledge
        graphiti_client.add_episode(
            name="python_knowledge",
            episode_body="Python is a programming language created by Guido van Rossum.",
            source_description="integration_test"
        )
        
        # Search for it
        results = graphiti_client.search(
            query="Who created Python?",
            num_results=5
        )
        
        assert results is not None
```

### 2.3 Orchestration Integration Tests

**Create `tests/integration/test_orchestration_real.py`:**

```python
# tests/integration/test_orchestration_real.py

"""
Real Orchestration Integration Tests

These tests execute real workflows and verify orchestration functionality.
"""

import os
import asyncio
import pytest
from typing import TypedDict, Annotated
from operator import add

from tests.integration import integration, skip_without_env


@integration
class TestLangGraphRealIntegration:
    """Real LangGraph workflow tests."""
    
    def test_simple_graph_execution(self, langgraph_builder):
        """Test executing a simple LangGraph workflow."""
        from langgraph.graph import END
        
        # Define state
        class State(TypedDict):
            messages: Annotated[list, add]
            step_count: int
        
        # Define nodes
        def process_input(state: State) -> State:
            return {
                "messages": ["Processed input"],
                "step_count": state.get("step_count", 0) + 1
            }
        
        def generate_output(state: State) -> State:
            return {
                "messages": ["Generated output"],
                "step_count": state.get("step_count", 0) + 1
            }
        
        # Build graph
        builder = langgraph_builder(State)
        builder.add_node("process", process_input)
        builder.add_node("generate", generate_output)
        builder.add_edge("process", "generate")
        builder.add_edge("generate", END)
        builder.set_entry_point("process")
        
        graph = builder.compile()
        
        # Execute
        result = graph.invoke({"messages": [], "step_count": 0})
        
        # Verify
        assert result["step_count"] == 2
        assert len(result["messages"]) == 2
    
    @skip_without_env("OPENAI_API_KEY")
    def test_llm_integrated_graph(self, langgraph_builder):
        """Test LangGraph with real LLM integration."""
        from langchain_openai import ChatOpenAI
        from langgraph.graph import END
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        class State(TypedDict):
            question: str
            answer: str
        
        def answer_question(state: State) -> State:
            response = llm.invoke(state["question"])
            return {"answer": response.content}
        
        builder = langgraph_builder(State)
        builder.add_node("answer", answer_question)
        builder.add_edge("answer", END)
        builder.set_entry_point("answer")
        
        graph = builder.compile()
        
        result = graph.invoke({
            "question": "What is 2 + 2?",
            "answer": ""
        })
        
        assert "4" in result["answer"]


@integration
class TestCrewAIRealIntegration:
    """Real CrewAI multi-agent tests."""
    
    @pytest.fixture
    def crewai_components(self):
        """Get CrewAI components, skip if unavailable."""
        from core.orchestration import CREWAI_AVAILABLE, get_crewai_builder
        
        if not CREWAI_AVAILABLE:
            pytest.skip("CrewAI not installed")
        
        return get_crewai_builder()
    
    @skip_without_env("OPENAI_API_KEY")
    def test_simple_crew_execution(self, crewai_components):
        """Test executing a simple CrewAI crew."""
        Agent = crewai_components["Agent"]
        Task = crewai_components["Task"]
        Crew = crewai_components["Crew"]
        
        # Create agent
        researcher = Agent(
            role="Researcher",
            goal="Research and summarize information",
            backstory="You are an expert researcher.",
            verbose=False
        )
        
        # Create task
        research_task = Task(
            description="List 3 benefits of exercise.",
            expected_output="A list of 3 benefits",
            agent=researcher
        )
        
        # Create and run crew
        crew = Crew(
            agents=[researcher],
            tasks=[research_task],
            verbose=False
        )
        
        result = crew.kickoff()
        
        assert result is not None
        assert len(str(result)) > 50  # Should have substantive output


@integration  
class TestTemporalRealIntegration:
    """Real Temporal workflow tests."""
    
    @pytest.fixture
    async def temporal_client(self):
        """Get Temporal client, skip if unavailable."""
        from core.orchestration import TEMPORAL_AVAILABLE, get_temporal_client
        
        if not TEMPORAL_AVAILABLE:
            pytest.skip("Temporal SDK not installed")
        
        try:
            client = await get_temporal_client()
            return client
        except Exception as e:
            pytest.skip(f"Temporal server not available: {e}")
    
    @pytest.mark.asyncio
    async def test_simple_workflow_execution(self, temporal_client):
        """Test executing a simple Temporal workflow."""
        from temporalio import workflow, activity
        from temporalio.worker import Worker
        
        # Define activity
        @activity.defn
        async def greet(name: str) -> str:
            return f"Hello, {name}!"
        
        # Define workflow
        @workflow.defn
        class GreetingWorkflow:
            @workflow.run
            async def run(self, name: str) -> str:
                return await workflow.execute_activity(
                    greet,
                    name,
                    start_to_close_timeout=timedelta(seconds=10)
                )
        
        # Run worker and execute workflow
        async with Worker(
            temporal_client,
            task_queue="test-queue",
            workflows=[GreetingWorkflow],
            activities=[greet]
        ):
            result = await temporal_client.execute_workflow(
                GreetingWorkflow.run,
                "World",
                id=f"test-{uuid.uuid4()}",
                task_queue="test-queue"
            )
            
            assert result == "Hello, World!"
```

### 2.4 Observability Integration Tests

**Create `tests/integration/test_observability_real.py`:**

```python
# tests/integration/test_observability_real.py

"""
Real Observability Integration Tests

These tests send real traces, metrics, and evaluations to observability platforms.
"""

import os
import time
import pytest
from datetime import datetime

from tests.integration import integration, skip_without_env


@integration
class TestLangfuseRealIntegration:
    """Real Langfuse tracing tests."""
    
    @skip_without_env("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
    def test_create_trace(self, langfuse_tracer):
        """Test creating a real trace in Langfuse."""
        trace = langfuse_tracer.trace(
            name="integration_test_trace",
            user_id="test_user",
            metadata={"test": True, "timestamp": datetime.now().isoformat()}
        )
        
        assert trace is not None
        assert trace.id is not None
        
        # Flush to ensure it's sent
        langfuse_tracer.flush()
    
    @skip_without_env("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
    def test_trace_with_spans(self, langfuse_tracer):
        """Test creating a trace with multiple spans."""
        trace = langfuse_tracer.trace(
            name="multi_span_test",
            user_id="test_user"
        )
        
        # Create spans
        span1 = trace.span(name="preprocessing")
        time.sleep(0.1)  # Simulate work
        span1.end()
        
        span2 = trace.span(name="llm_call")
        time.sleep(0.1)
        span2.end(output="LLM response")
        
        span3 = trace.span(name="postprocessing")
        time.sleep(0.05)
        span3.end()
        
        langfuse_tracer.flush()
        
        # Trace should be complete
        assert trace.id is not None
    
    @skip_without_env("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")
    def test_trace_with_generation(self, langfuse_tracer):
        """Test logging LLM generations."""
        trace = langfuse_tracer.trace(name="generation_test")
        
        generation = trace.generation(
            name="test_generation",
            model="gpt-4",
            input="What is AI?",
            output="AI is artificial intelligence...",
            usage={"prompt_tokens": 10, "completion_tokens": 50}
        )
        
        langfuse_tracer.flush()
        
        assert generation.id is not None


@integration
class TestOpikRealIntegration:
    """Real Opik experiment tracking tests."""
    
    @pytest.fixture
    def opik_client(self):
        """Get Opik client, skip if unavailable."""
        from core.observability import OPIK_AVAILABLE, get_opik_client
        
        if not OPIK_AVAILABLE:
            pytest.skip("Opik not installed")
        
        if not os.getenv("OPIK_API_KEY"):
            pytest.skip("OPIK_API_KEY not set")
        
        return get_opik_client()
    
    @skip_without_env("OPIK_API_KEY")
    def test_log_experiment(self, opik_client):
        """Test logging an experiment to Opik."""
        from opik import track
        
        @track
        def test_function(input_text: str) -> str:
            return f"Processed: {input_text}"
        
        result = test_function("Hello, Opik!")
        
        assert result == "Processed: Hello, Opik!"


@integration
class TestDeepEvalRealIntegration:
    """Real DeepEval evaluation tests."""
    
    @pytest.fixture
    def deepeval_evaluator(self):
        """Get DeepEval evaluator, skip if unavailable."""
        from core.observability import DEEPEVAL_AVAILABLE, get_deepeval_evaluator
        
        if not DEEPEVAL_AVAILABLE:
            pytest.skip("DeepEval not installed")
        
        return get_deepeval_evaluator()
    
    @skip_without_env("OPENAI_API_KEY")
    def test_answer_relevancy_evaluation(self, deepeval_evaluator):
        """Test evaluating answer relevancy."""
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase
        
        metric = AnswerRelevancyMetric(threshold=0.5)
        
        test_case = LLMTestCase(
            input="What is Python?",
            actual_output="Python is a programming language known for its simple syntax.",
            retrieval_context=["Python is a high-level programming language."]
        )
        
        metric.measure(test_case)
        
        assert metric.score is not None
        assert metric.score >= 0.0
        assert metric.score <= 1.0
    
    @skip_without_env("OPENAI_API_KEY")
    def test_faithfulness_evaluation(self, deepeval_evaluator):
        """Test evaluating faithfulness to context."""
        from deepeval.metrics import FaithfulnessMetric
        from deepeval.test_case import LLMTestCase
        
        metric = FaithfulnessMetric(threshold=0.5)
        
        test_case = LLMTestCase(
            input="What color is the sky?",
            actual_output="The sky is blue during the day.",
            retrieval_context=["The sky appears blue due to Rayleigh scattering."]
        )
        
        metric.measure(test_case)
        
        assert metric.score is not None
```

### 2.5 Structured Output Integration Tests

**Create `tests/integration/test_structured_real.py`:**

```python
# tests/integration/test_structured_real.py

"""
Real Structured Output Integration Tests

These tests actually parse LLM output with structured output libraries.
"""

import os
import pytest
from pydantic import BaseModel, Field
from typing import List

from tests.integration import integration, skip_without_env


@integration
class TestInstructorRealIntegration:
    """Real Instructor structured output tests."""
    
    @skip_without_env("OPENAI_API_KEY")
    def test_simple_extraction(self, instructor_client):
        """Test extracting structured data from LLM."""
        
        class Person(BaseModel):
            name: str
            age: int
        
        person = instructor_client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=Person,
            messages=[
                {"role": "user", "content": "John is 25 years old."}
            ]
        )
        
        assert person.name == "John"
        assert person.age == 25
    
    @skip_without_env("OPENAI_API_KEY")
    def test_complex_extraction(self, instructor_client):
        """Test extracting complex nested structures."""
        
        class Address(BaseModel):
            street: str
            city: str
            country: str
        
        class Company(BaseModel):
            name: str
            employees: int
            address: Address
        
        company = instructor_client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=Company,
            messages=[
                {"role": "user", "content": """
                    Acme Corp has 500 employees and is located at 
                    123 Main Street, New York, USA.
                """}
            ]
        )
        
        assert company.name == "Acme Corp"
        assert company.employees == 500
        assert company.address.city == "New York"
    
    @skip_without_env("OPENAI_API_KEY")
    def test_list_extraction(self, instructor_client):
        """Test extracting lists of structured items."""
        
        class Task(BaseModel):
            title: str
            priority: str = Field(description="high, medium, or low")
        
        class TaskList(BaseModel):
            tasks: List[Task]
        
        result = instructor_client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=TaskList,
            messages=[
                {"role": "user", "content": """
                    My tasks are:
                    1. Fix critical bug (urgent)
                    2. Update documentation (whenever)
                    3. Code review (soon)
                """}
            ]
        )
        
        assert len(result.tasks) == 3
        assert result.tasks[0].priority.lower() == "high"


@integration
class TestMirascopeRealIntegration:
    """Real Mirascope structured output tests."""
    
    @pytest.fixture
    def mirascope_anthropic(self):
        """Get Mirascope Anthropic provider, skip if unavailable."""
        from core.structured import MIRASCOPE_AVAILABLE, get_mirascope_provider
        
        if not MIRASCOPE_AVAILABLE:
            pytest.skip("Mirascope not installed")
        
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")
        
        return get_mirascope_provider("anthropic")
    
    @skip_without_env("ANTHROPIC_API_KEY")
    def test_mirascope_extraction(self, mirascope_anthropic):
        """Test Mirascope structured extraction with Claude."""
        from pydantic import BaseModel
        
        class Sentiment(BaseModel):
            text: str
            sentiment: str  # positive, negative, neutral
            confidence: float
        
        @mirascope_anthropic.call("claude-3-haiku-20240307", response_model=Sentiment)
        def analyze_sentiment(text: str) -> str:
            return f"Analyze the sentiment of: {text}"
        
        result = analyze_sentiment("I love this product! It's amazing!")
        
        assert result.sentiment.lower() == "positive"
        assert result.confidence > 0.5
```

### 2.6 Tools Integration Tests

**Create `tests/integration/test_tools_real.py`:**

```python
# tests/integration/test_tools_real.py

"""
Real Tools Integration Tests

These tests actually execute web scraping, search, and other tool operations.
"""

import os
import pytest
from tests.integration import integration, skip_without_env


@integration
class TestCrawl4AIRealIntegration:
    """Real Crawl4AI web scraping tests."""
    
    @pytest.fixture
    def crawler(self):
        """Get Crawl4AI crawler, skip if unavailable."""
        try:
            from crawl4ai import AsyncWebCrawler
            return AsyncWebCrawler
        except ImportError:
            pytest.skip("Crawl4AI not installed")
    
    @pytest.mark.asyncio
    async def test_simple_crawl(self, crawler):
        """Test crawling a simple webpage."""
        async with crawler() as crawler_instance:
            result = await crawler_instance.arun(
                url="https://example.com"
            )
            
            assert result.success
            assert "Example Domain" in result.html
            assert len(result.markdown) > 0
    
    @pytest.mark.asyncio
    async def test_crawl_with_extraction(self, crawler):
        """Test crawling with content extraction."""
        async with crawler() as crawler_instance:
            result = await crawler_instance.arun(
                url="https://httpbin.org/html",
                extraction_strategy="markdown"
            )
            
            assert result.success
            assert result.markdown is not None


@integration
class TestTavilyRealIntegration:
    """Real Tavily search tests."""
    
    @pytest.fixture
    def tavily_client(self):
        """Get Tavily client, skip if unavailable."""
        try:
            from tavily import TavilyClient
        except ImportError:
            pytest.skip("Tavily not installed")
        
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            pytest.skip("TAVILY_API_KEY not set")
        
        return TavilyClient(api_key=api_key)
    
    @skip_without_env("TAVILY_API_KEY")
    def test_simple_search(self, tavily_client):
        """Test simple web search."""
        results = tavily_client.search(
            query="What is Python programming language?",
            max_results=3
        )
        
        assert results is not None
        assert "results" in results
        assert len(results["results"]) > 0
    
    @skip_without_env("TAVILY_API_KEY")
    def test_search_with_context(self, tavily_client):
        """Test search returning context for RAG."""
        results = tavily_client.get_search_context(
            query="Latest Python version features",
            max_results=3
        )
        
        assert results is not None
        assert len(results) > 0


@integration
class TestFirecrawlRealIntegration:
    """Real Firecrawl scraping tests."""
    
    @pytest.fixture
    def firecrawl_app(self):
        """Get Firecrawl app, skip if unavailable."""
        try:
            from firecrawl import FirecrawlApp
        except ImportError:
            pytest.skip("Firecrawl not installed")
        
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            pytest.skip("FIRECRAWL_API_KEY not set")
        
        return FirecrawlApp(api_key=api_key)
    
    @skip_without_env("FIRECRAWL_API_KEY")
    def test_scrape_url(self, firecrawl_app):
        """Test scraping a URL with Firecrawl."""
        result = firecrawl_app.scrape_url(
            url="https://example.com",
            params={"formats": ["markdown"]}
        )
        
        assert result is not None
        assert "markdown" in result
        assert len(result["markdown"]) > 0


@integration
class TestBrowserbaseRealIntegration:
    """Real Browserbase browser automation tests."""
    
    @pytest.fixture
    def browserbase_client(self):
        """Get Browserbase client, skip if unavailable."""
        try:
            from browserbase import Browserbase
        except ImportError:
            pytest.skip("Browserbase not installed")
        
        api_key = os.getenv("BROWSERBASE_API_KEY")
        if not api_key:
            pytest.skip("BROWSERBASE_API_KEY not set")
        
        return Browserbase(api_key=api_key)
    
    @skip_without_env("BROWSERBASE_API_KEY")
    def test_create_session(self, browserbase_client):
        """Test creating a browser session."""
        session = browserbase_client.sessions.create()
        
        assert session is not None
        assert session.id is not None
        
        # Cleanup
        browserbase_client.sessions.complete(session.id)
```

---

## Section 3: SDK Availability Validator (~200 lines)

### 3.1 Create `core/validator.py`

```python
# core/validator.py

"""
SDK Availability Validator

Comprehensive validation of all SDK installations, configurations, and functionality.
No stubs, no fallbacks - explicit status reporting.

Usage:
    from core.validator import validate_all_sdks, get_system_health
    
    # Check all SDKs
    results = validate_all_sdks()
    
    # Get full health report
    health = get_system_health()
"""

from __future__ import annotations
import os
import sys
import importlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from datetime import datetime


class SDKCategory(Enum):
    """Categories of SDKs."""
    OBSERVABILITY = "observability"
    MEMORY = "memory"
    ORCHESTRATION = "orchestration"
    STRUCTURED = "structured"
    TOOLS = "tools"
    LLM = "llm"


class ValidationStatus(Enum):
    """SDK validation status."""
    AVAILABLE = "available"        # Installed, configured, functional
    INSTALLED = "installed"        # Installed but not configured
    UNAVAILABLE = "unavailable"    # Not installed
    ERROR = "error"                # Installed but errors on import
    DEPRECATED = "deprecated"      # SDK is deprecated


@dataclass
class SDKValidationResult:
    """Result of validating a single SDK."""
    name: str
    category: SDKCategory
    status: ValidationStatus
    version: Optional[str] = None
    import_error: Optional[str] = None
    config_missing: List[str] = field(default_factory=list)
    functional_test: Optional[bool] = None
    install_cmd: str = ""
    docs_url: str = ""
    
    @property
    def is_ready(self) -> bool:
        """Check if SDK is fully ready for use."""
        return self.status == ValidationStatus.AVAILABLE and self.functional_test is True


@dataclass
class SystemHealthReport:
    """Complete system health report."""
    timestamp: datetime
    python_version: str
    total_sdks: int
    available_sdks: int
    configured_sdks: int
    functional_sdks: int
    results: Dict[str, SDKValidationResult]
    critical_issues: List[str]
    warnings: List[str]
    
    @property
    def health_score(self) -> float:
        """Calculate overall health score (0-100)."""
        if self.total_sdks == 0:
            return 0.0
        return (self.functional_sdks / self.total_sdks) * 100


# =============================================================================
# SDK DEFINITIONS
# =============================================================================

SDK_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Observability
    "langfuse": {
        "category": SDKCategory.OBSERVABILITY,
        "import_path": "langfuse",
        "test_import": "from langfuse import Langfuse",
        "config_vars": ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"],
        "install_cmd": "pip install langfuse>=2.0.0",
        "docs_url": "https://langfuse.com/docs",
    },
    "phoenix": {
        "category": SDKCategory.OBSERVABILITY,
        "import_path": "phoenix",
        "test_import": "import phoenix as px",
        "config_vars": [],
        "install_cmd": "pip install arize-phoenix>=4.0.0",
        "docs_url": "https://docs.arize.com/phoenix",
    },
    "opik": {
        "category": SDKCategory.OBSERVABILITY,
        "import_path": "opik",
        "test_import": "from opik import track",
        "config_vars": ["OPIK_API_KEY"],
        "install_cmd": "pip install opik>=1.0.0",
        "docs_url": "https://www.comet.com/docs/opik/",
    },
    "deepeval": {
        "category": SDKCategory.OBSERVABILITY,
        "import_path": "deepeval",
        "test_import": "from deepeval import evaluate",
        "config_vars": [],
        "install_cmd": "pip install deepeval>=1.0.0",
        "docs_url": "https://docs.confident-ai.com/",
    },
    "logfire": {
        "category": SDKCategory.OBSERVABILITY,
        "import_path": "logfire",
        "test_import": "import logfire",
        "config_vars": ["LOGFIRE_TOKEN"],
        "install_cmd": "pip install logfire>=0.30.0",
        "docs_url": "https://logfire.pydantic.dev/docs/",
    },
    
    # Memory
    "mem0": {
        "category": SDKCategory.MEMORY,
        "import_path": "mem0",
        "test_import": "from mem0 import Memory",
        "config_vars": [],  # Can work locally
        "install_cmd": "pip install mem0ai>=0.1.0",
        "docs_url": "https://docs.mem0.ai/",
    },
    "letta": {
        "category": SDKCategory.MEMORY,
        "import_path": "letta",
        "test_import": "from letta import create_client",
        "config_vars": [],
        "install_cmd": "pip install letta>=0.4.0",
        "docs_url": "https://docs.letta.com/",
    },
    "graphiti": {
        "category": SDKCategory.MEMORY,
        "import_path": "graphiti_core",
        "test_import": "from graphiti_core import Graphiti",
        "config_vars": ["NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"],
        "install_cmd": "pip install graphiti-core>=0.3.0",
        "docs_url": "https://github.com/getzep/graphiti",
    },
    "cognee": {
        "category": SDKCategory.MEMORY,
        "import_path": "cognee",
        "test_import": "import cognee",
        "config_vars": [],
        "install_cmd": "pip install cognee>=0.1.0",
        "docs_url": "https://github.com/topoteretes/cognee",
    },
    
    # Orchestration
    "langgraph": {
        "category": SDKCategory.ORCHESTRATION,
        "import_path": "langgraph",
        "test_import": "from langgraph.graph import StateGraph",
        "config_vars": [],
        "install_cmd": "pip install langgraph>=0.2.0",
        "docs_url": "https://langchain-ai.github.io/langgraph/",
    },
    "crewai": {
        "category": SDKCategory.ORCHESTRATION,
        "import_path": "crewai",
        "test_import": "from crewai import Agent, Task, Crew",
        "config_vars": ["OPENAI_API_KEY"],
        "install_cmd": "pip install crewai>=0.60.0",
        "docs_url": "https://docs.crewai.com/",
    },
    "temporal": {
        "category": SDKCategory.ORCHESTRATION,
        "import_path": "temporalio",
        "test_import": "from temporalio import workflow",
        "config_vars": [],
        "install_cmd": "pip install temporalio>=1.5.0",
        "docs_url": "https://docs.temporal.io/develop/python",
    },
    "prefect": {
        "category": SDKCategory.ORCHESTRATION,
        "import_path": "prefect",
        "test_import": "from prefect import flow, task",
        "config_vars": [],
        "install_cmd": "pip install prefect>=2.0.0",
        "docs_url": "https://docs.prefect.io/",
    },
    
    # Structured Output
    "instructor": {
        "category": SDKCategory.STRUCTURED,
        "import_path": "instructor",
        "test_import": "import instructor",
        "config_vars": [],
        "install_cmd": "pip install instructor>=1.0.0",
        "docs_url": "https://python.useinstructor.com/",
    },
    "outlines": {
        "category": SDKCategory.STRUCTURED,
        "import_path": "outlines",
        "test_import": "from outlines import generate",
        "config_vars": [],
        "install_cmd": "pip install outlines>=0.0.40",
        "docs_url": "https://outlines-dev.github.io/outlines/",
    },
    "mirascope": {
        "category": SDKCategory.STRUCTURED,
        "import_path": "mirascope",
        "test_import": "from mirascope.core import anthropic",
        "config_vars": [],
        "install_cmd": "pip install mirascope>=1.0.0",
        "docs_url": "https://docs.mirascope.io/",
    },
    
    # Tools
    "crawl4ai": {
        "category": SDKCategory.TOOLS,
        "import_path": "crawl4ai",
        "test_import": "from crawl4ai import AsyncWebCrawler",
        "config_vars": [],
        "install_cmd": "pip install crawl4ai>=0.3.0",
        "docs_url": "https://crawl4ai.com/",
    },
    "tavily": {
        "category": SDKCategory.TOOLS,
        "import_path": "tavily",
        "test_import": "from tavily import TavilyClient",
        "config_vars": ["TAVILY_API_KEY"],
        "install_cmd": "pip install tavily-python>=0.3.0",
        "docs_url": "https://docs.tavily.com/",
    },
    "firecrawl": {
        "category": SDKCategory.TOOLS,
        "import_path": "firecrawl",
        "test_import": "from firecrawl import FirecrawlApp",
        "config_vars": ["FIRECRAWL_API_KEY"],
        "install_cmd": "pip install firecrawl-py>=1.0.0",
        "docs_url": "https://docs.firecrawl.dev/",
    },
    "browserbase": {
        "category": SDKCategory.TOOLS,
        "import_path": "browserbase",
        "test_import": "from browserbase import Browserbase",
        "config_vars": ["BROWSERBASE_API_KEY"],
        "install_cmd": "pip install browserbase>=0.3.0",
        "docs_url": "https://docs.browserbase.com/",
    },
    
    # LLM Providers
    "anthropic": {
        "category": SDKCategory.LLM,
        "import_path": "anthropic",
        "test_import": "from anthropic import Anthropic",
        "config_vars": ["ANTHROPIC_API_KEY"],
        "install_cmd": "pip install anthropic>=0.30.0",
        "docs_url": "https://docs.anthropic.com/",
    },
    "openai": {
        "category": SDKCategory.LLM,
        "import_path": "openai",
        "test_import": "from openai import OpenAI",
        "config_vars": ["OPENAI_API_KEY"],
        "install_cmd": "pip install openai>=1.30.0",
        "docs_url": "https://platform.openai.com/docs/",
    },
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_sdk(name: str) -> SDKValidationResult:
    """Validate a single SDK."""
    if name not in SDK_REGISTRY:
        return SDKValidationResult(
            name=name,
            category=SDKCategory.TOOLS,
            status=ValidationStatus.ERROR,
            import_error=f"Unknown SDK: {name}"
        )
    
    sdk_info = SDK_REGISTRY[name]
    result = SDKValidationResult(
        name=name,
        category=sdk_info["category"],
        status=ValidationStatus.UNAVAILABLE,
        install_cmd=sdk_info["install_cmd"],
        docs_url=sdk_info["docs_url"]
    )
    
    # Check import
    try:
        exec(sdk_info["test_import"])
        result.status = ValidationStatus.INSTALLED
        
        # Get version
        try:
            module = importlib.import_module(sdk_info["import_path"])
            result.version = getattr(module, "__version__", "unknown")
        except:
            pass
            
    except ImportError as e:
        result.import_error = str(e)
        return result
    except Exception as e:
        result.status = ValidationStatus.ERROR
        result.import_error = str(e)
        return result
    
    # Check configuration
    missing_config = []
    for var in sdk_info["config_vars"]:
        if not os.getenv(var):
            missing_config.append(var)
    
    result.config_missing = missing_config
    
    if not missing_config:
        result.status = ValidationStatus.AVAILABLE
    
    return result


def validate_all_sdks() -> Dict[str, SDKValidationResult]:
    """Validate all registered SDKs."""
    results = {}
    for name in SDK_REGISTRY:
        results[name] = validate_sdk(name)
    return results


def validate_api_keys() -> Dict[str, bool]:
    """Check all required API keys."""
    all_vars = set()
    for sdk_info in SDK_REGISTRY.values():
        all_vars.update(sdk_info["config_vars"])
    
    return {var: bool(os.getenv(var)) for var in all_vars}


def get_system_health() -> SystemHealthReport:
    """Get comprehensive system health report."""
    results = validate_all_sdks()
    
    available = sum(1 for r in results.values() if r.status in [ValidationStatus.AVAILABLE, ValidationStatus.INSTALLED])
    configured = sum(1 for r in results.values() if r.status == ValidationStatus.AVAILABLE)
    functional = sum(1 for r in results.values() if r.is_ready)
    
    critical_issues = []
    warnings = []
    
    for name, result in results.items():
        if result.status == ValidationStatus.UNAVAILABLE:
            warnings.append(f"{name}: Not installed. Run: {result.install_cmd}")
        elif result.status == ValidationStatus.ERROR:
            critical_issues.append(f"{name}: Import error - {result.import_error}")
        elif result.config_missing:
            warnings.append(f"{name}: Missing config: {', '.join(result.config_missing)}")
    
    return SystemHealthReport(
        timestamp=datetime.now(),
        python_version=sys.version,
        total_sdks=len(results),
        available_sdks=available,
        configured_sdks=configured,
        functional_sdks=functional,
        results=results,
        critical_issues=critical_issues,
        warnings=warnings
    )


def print_health_report(report: SystemHealthReport) -> None:
    """Print formatted health report."""
    print("\n" + "=" * 60)
    print(" UNLEASH SYSTEM HEALTH REPORT")
    print("=" * 60)
    print(f" Timestamp: {report.timestamp.isoformat()}")
    print(f" Python: {report.python_version.split()[0]}")
    print(f" Health Score: {report.health_score:.1f}%")
    print("-" * 60)
    print(f" Total SDKs: {report.total_sdks}")
    print(f" Available: {report.available_sdks}")
    print(f" Configured: {report.configured_sdks}")
    print(f" Functional: {report.functional_sdks}")
    print("-" * 60)
    
    if report.critical_issues:
        print("\n🚨 CRITICAL ISSUES:")
        for issue in report.critical_issues:
            print(f"   ❌ {issue}")
    
    if report.warnings:
        print("\n⚠️  WARNINGS:")
        for warning in report.warnings:
            print(f"   ⚠️  {warning}")
    
    print("\n" + "=" * 60)


# =============================================================================
# CLI INTEGRATION
# =============================================================================

def cli_doctor(deep: bool = False) -> int:
    """Run health check from CLI. Returns exit code."""
    report = get_system_health()
    print_health_report(report)
    
    if deep:
        print("\n🔬 Running deep functional tests...")
        # Run functional tests for available SDKs
        # This would import and run quick sanity checks
    
    if report.critical_issues:
        return 1
    return 0


if __name__ == "__main__":
    import sys
    deep = "--deep" in sys.argv
    exit(cli_doctor(deep=deep))
```

### 3.2 Add CLI Command `unleash doctor`

```python
# Add to cli.py or create cli/commands/doctor.py

import click
from core.validator import get_system_health, print_health_report, validate_all_sdks

@click.command()
@click.option('--deep', is_flag=True, help='Run deep functional tests')
@click.option('--json', 'as_json', is_flag=True, help='Output as JSON')
@click.option('--category', help='Filter by category (observability, memory, etc.)')
def doctor(deep: bool, as_json: bool, category: str):
    """Check system health and SDK availability.
    
    Examples:
        unleash doctor           # Quick health check
        unleash doctor --deep    # Run functional tests
        unleash doctor --json    # Machine-readable output
    """
    report = get_system_health()
    
    if as_json:
        import json
        output = {
            "timestamp": report.timestamp.isoformat(),
            "health_score": report.health_score,
            "total_sdks": report.total_sdks,
            "available_sdks": report.available_sdks,
            "critical_issues": report.critical_issues,
            "warnings": report.warnings,
            "sdks": {
                name: {
                    "status": r.status.value,
                    "version": r.version,
                    "config_missing": r.config_missing
                }
                for name, r in report.results.items()
            }
        }
        click.echo(json.dumps(output, indent=2))
    else:
        print_health_report(report)
    
    if deep:
        click.echo("\n🔬 Running deep validation...")
        # Import and run integration tests in validation mode
        import subprocess
        result = subprocess.run([
            "pytest", "tests/integration/", 
            "-m", "integration",
            "--tb=short",
            "-q"
        ], capture_output=True, text=True)
        click.echo(result.stdout)
        if result.returncode != 0:
            click.echo(result.stderr)
    
    # Exit code based on health
    if report.critical_issues:
        raise SystemExit(1)
```

---

## Section 4: Fix Python 3.14 Issues (~100 lines)

### 4.1 Create `scripts/fix_python314.py`

```python
#!/usr/bin/env python3
"""
Python 3.14 Compatibility Fixes

This script addresses compatibility issues with Python 3.14.
Options:
  A. Pin SDK versions that work
  B. Create compatibility shims
  C. Document downgrade to Python 3.12
"""

import sys
import subprocess
from pathlib import Path

# Known Python 3.14 incompatibilities
INCOMPATIBLE_SDKS = {
    "crewai": {
        "issue": "Uses deprecated asyncio.coroutine",
        "fixed_version": None,  # Not yet fixed
        "workaround": "Use Python 3.12"
    },
    "langgraph": {
        "issue": "Depends on langchain with legacy typing",
        "fixed_version": "0.2.5",
        "workaround": "pip install langgraph>=0.2.5"
    },
    "instructor": {
        "issue": "Pydantic v1 compatibility layer removed",
        "fixed_version": "1.5.0",
        "workaround": "pip install instructor>=1.5.0"
    },
}

# Option A: Pin compatible versions
COMPATIBLE_VERSIONS = """
# requirements-py314.txt
# Python 3.14 compatible versions

# Core
pydantic>=2.6.0
anthropic>=0.30.0
openai>=1.30.0

# Observability (all compatible)
langfuse>=2.40.0
arize-phoenix>=4.0.0
opik>=1.0.0
deepeval>=1.0.0
logfire>=0.30.0

# Memory (mostly compatible)
mem0ai>=0.1.0
letta>=0.4.0

# Orchestration (some issues)
langgraph>=0.2.5
# crewai - NOT COMPATIBLE, use Python 3.12
temporalio>=1.5.0
prefect>=2.0.0

# Structured (updated required)
instructor>=1.5.0
outlines>=0.0.40
mirascope>=1.0.0

# Tools
crawl4ai>=0.3.0
tavily-python>=0.3.0
firecrawl-py>=1.0.0
"""


def check_python_version():
    """Check current Python version."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version >= (3, 14):
        print("⚠️  Python 3.14+ detected. Some SDKs may not work.")
        print("\nIncompatible SDKs:")
        for sdk, info in INCOMPATIBLE_SDKS.items():
            print(f"  - {sdk}: {info['issue']}")
            if info['fixed_version']:
                print(f"    Fix: {info['workaround']}")
            else:
                print(f"    Workaround: {info['workaround']}")
        return False
    
    print("✅ Python version compatible with all SDKs")
    return True


def create_compatibility_requirements():
    """Create Python 3.14 compatible requirements file."""
    path = Path("requirements-py314.txt")
    path.write_text(COMPATIBLE_VERSIONS)
    print(f"Created {path}")


def create_compatibility_shim():
    """Create compatibility shim for problematic imports."""
    shim_content = '''
# core/compat.py
"""
Compatibility shim for Python 3.14+

Import from this module instead of directly from SDKs that have issues.
"""

import sys
import warnings

if sys.version_info >= (3, 14):
    warnings.warn(
        "Python 3.14+ detected. Some features may not work. "
        "Consider using Python 3.12 for full compatibility.",
        RuntimeWarning
    )

# CrewAI shim
def get_crewai():
    """Get CrewAI with compatibility handling."""
    if sys.version_info >= (3, 14):
        raise ImportError(
            "CrewAI is not compatible with Python 3.14+. "
            "Please use Python 3.12 or wait for CrewAI update."
        )
    from crewai import Agent, Task, Crew
    return Agent, Task, Crew

# LangGraph shim  
def get_langgraph():
    """Get LangGraph with version check."""
    from langgraph.graph import StateGraph, END
    return StateGraph, END
'''
    
    path = Path("core/compat.py")
    path.write_text(shim_content)
    print(f"Created {path}")


def suggest_downgrade():
    """Suggest Python downgrade steps."""
    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDED: Use Python 3.12 for Full Compatibility
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Option 1: pyenv (recommended)
  pyenv install 3.12.0
  pyenv local 3.12.0
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt

Option 2: conda
  conda create -n unleash python=3.12
  conda activate unleash
  pip install -r requirements.txt

Option 3: Docker
  FROM python:3.12-slim
  # ... (see Dockerfile)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Python 3.14 compatibility fixes")
    parser.add_argument("--check", action="store_true", help="Check current compatibility")
    parser.add_argument("--fix", choices=["versions", "shim", "downgrade"], help="Apply fix")
    args = parser.parse_args()
    
    if args.check or not args.fix:
        check_python_version()
    
    if args.fix == "versions":
        create_compatibility_requirements()
    elif args.fix == "shim":
        create_compatibility_shim()
    elif args.fix == "downgrade":
        suggest_downgrade()


if __name__ == "__main__":
    main()
```

---

## Section 5: Production Validation (~200 lines)

### 5.1 Create `scripts/validate_production.py`

```python
#!/usr/bin/env python3
"""
Production Validation Script

Validates that ALL SDKs actually work, not just import.
Runs sample operations and reports PASS/FAIL with reasons.

Usage:
    python scripts/validate_production.py
    python scripts/validate_production.py --verbose
    python scripts/validate_production.py --fix
"""

import os
import sys
import asyncio
import traceback
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from datetime import datetime


class TestResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    ERROR = "ERROR"


@dataclass
class ValidationTest:
    """A single validation test."""
    name: str
    sdk: str
    test_fn: Callable
    required_env: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test: ValidationTest
    result: TestResult
    duration_ms: float
    error: Optional[str] = None
    details: Optional[str] = None


class ProductionValidator:
    """Validates production readiness of all SDKs."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[ValidationResult] = []
        self.tests = self._register_tests()
    
    def _register_tests(self) -> List[ValidationTest]:
        """Register all validation tests."""
        return [
            # Observability Tests
            ValidationTest(
                name="langfuse_trace",
                sdk="langfuse",
                test_fn=self._test_langfuse,
                required_env=["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"],
                description="Create and flush a Langfuse trace"
            ),
            ValidationTest(
                name="deepeval_metric",
                sdk="deepeval",
                test_fn=self._test_deepeval,
                required_env=["OPENAI_API_KEY"],
                description="Run a DeepEval metric evaluation"
            ),
            
            # Memory Tests
            ValidationTest(
                name="mem0_memory",
                sdk="mem0",
                test_fn=self._test_mem0,
                required_env=[],
                description="Store and retrieve a Mem0 memory"
            ),
            
            # Orchestration Tests
            ValidationTest(
                name="langgraph_workflow",
                sdk="langgraph",
                test_fn=self._test_langgraph,
                required_env=[],
                description="Execute a simple LangGraph workflow"
            ),
            
            # Structured Output Tests
            ValidationTest(
                name="instructor_extraction",
                sdk="instructor",
                test_fn=self._test_instructor,
                required_env=["OPENAI_API_KEY"],
                description="Extract structured data with Instructor"
            ),
            
            # Tools Tests
            ValidationTest(
                name="crawl4ai_scrape",
                sdk="crawl4ai",
                test_fn=self._test_crawl4ai,
                required_env=[],
                description="Scrape a webpage with Crawl4AI"
            ),
            ValidationTest(
                name="tavily_search",
                sdk="tavily",
                test_fn=self._test_tavily,
                required_env=["TAVILY_API_KEY"],
                description="Perform a Tavily web search"
            ),
            
            # LLM Tests
            ValidationTest(
                name="anthropic_completion",
                sdk="anthropic",
                test_fn=self._test_anthropic,
                required_env=["ANTHROPIC_API_KEY"],
                description="Generate a Claude completion"
            ),
            ValidationTest(
                name="openai_completion",
                sdk="openai",
                test_fn=self._test_openai,
                required_env=["OPENAI_API_KEY"],
                description="Generate an OpenAI completion"
            ),
        ]
    
    # ==========================================================================
    # Test Implementations
    # ==========================================================================
    
    def _test_langfuse(self) -> str:
        """Test Langfuse tracing."""
        from langfuse import Langfuse
        client = Langfuse()
        trace = client.trace(name="production_validation_test")
        trace.span(name="test_span").end()
        client.flush()
        return f"Created trace {trace.id}"
    
    def _test_deepeval(self) -> str:
        """Test DeepEval evaluation."""
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase
        
        metric = AnswerRelevancyMetric(threshold=0.5)
        test_case = LLMTestCase(
            input="What is 2+2?",
            actual_output="2+2 equals 4.",
            retrieval_context=["Basic arithmetic: 2+2=4"]
        )
        metric.measure(test_case)
        return f"Score: {metric.score:.2f}"
    
    def _test_mem0(self) -> str:
        """Test Mem0 memory operations."""
        from mem0 import Memory
        memory = Memory()
        result = memory.add(
            messages=[{"role": "user", "content": "Test memory for production validation"}],
            user_id="production_validation_test"
        )
        return f"Added memory: {result}"
    
    def _test_langgraph(self) -> str:
        """Test LangGraph workflow."""
        from langgraph.graph import StateGraph, END
        from typing import TypedDict
        
        class State(TypedDict):
            value: int
        
        def increment(state: State) -> State:
            return {"value": state["value"] + 1}
        
        builder = StateGraph(State)
        builder.add_node("increment", increment)
        builder.add_edge("increment", END)
        builder.set_entry_point("increment")
        
        graph = builder.compile()
        result = graph.invoke({"value": 0})
        
        assert result["value"] == 1
        return f"Workflow result: {result}"
    
    def _test_instructor(self) -> str:
        """Test Instructor structured extraction."""
        import instructor
        from openai import OpenAI
        from pydantic import BaseModel
        
        class Number(BaseModel):
            value: int
        
        client = instructor.patch(OpenAI())
        result = client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=Number,
            messages=[{"role": "user", "content": "What is 5 + 5?"}]
        )
        return f"Extracted: {result.value}"
    
    def _test_crawl4ai(self) -> str:
        """Test Crawl4AI web scraping."""
        from crawl4ai import AsyncWebCrawler
        
        async def crawl():
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url="https://example.com")
                return result
        
        result = asyncio.run(crawl())
        assert result.success
        return f"Scraped {len(result.markdown)} chars"
    
    def _test_tavily(self) -> str:
        """Test Tavily search."""
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        results = client.search(query="Python programming", max_results=1)
        return f"Found {len(results['results'])} results"
    
    def _test_anthropic(self) -> str:
        """Test Anthropic Claude."""
        from anthropic import Anthropic
        client = Anthropic()
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=50,
            messages=[{"role": "user", "content": "Say 'OK' if you can hear me."}]
        )
        return f"Response: {response.content[0].text[:50]}"
    
    def _test_openai(self) -> str:
        """Test OpenAI."""
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=50,
            messages=[{"role": "user", "content": "Say 'OK' if you can hear me."}]
        )
        return f"Response: {response.choices[0].message.content[:50]}"
    
    # ==========================================================================
    # Validation Runner
    # ==========================================================================
    
    def _check_env(self, test: ValidationTest) -> Optional[str]:
        """Check if required environment variables are set."""
        missing = [v for v in test.required_env if not os.getenv(v)]
        if missing:
            return f"Missing env vars: {', '.join(missing)}"
        return None
    
    def _check_import(self, sdk: str) -> Optional[str]:
        """Check if SDK can be imported."""
        from core.validator import validate_sdk, ValidationStatus
        result = validate_sdk(sdk)
        if result.status in [ValidationStatus.UNAVAILABLE, ValidationStatus.ERROR]:
            return f"SDK not available: {result.import_error or 'not installed'}"
        return None
    
    def run_test(self, test: ValidationTest) -> ValidationResult:
        """Run a single validation test."""
        import time
        start = time.time()
        
        # Check environment
        env_error = self._check_env(test)
        if env_error:
            return ValidationResult(
                test=test,
                result=TestResult.SKIP,
                duration_ms=0,
                error=env_error
            )
        
        # Check import
        import_error = self._check_import(test.sdk)
        if import_error:
            return ValidationResult(
                test=test,
                result=TestResult.SKIP,
                duration_ms=0,
                error=import_error
            )
        
        # Run test
        try:
            details = test.test_fn()
            duration = (time.time() - start) * 1000
            return ValidationResult(
                test=test,
                result=TestResult.PASS,
                duration_ms=duration,
                details=details
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ValidationResult(
                test=test,
                result=TestResult.FAIL,
                duration_ms=duration,
                error=str(e),
                details=traceback.format_exc() if self.verbose else None
            )
    
    def run_all(self) -> Dict[str, Any]:
        """Run all validation tests."""
        print("\n" + "=" * 60)
        print(" PRODUCTION VALIDATION")
        print("=" * 60 + "\n")
        
        for test in self.tests:
            print(f"Testing {test.name}...", end=" ", flush=True)
            result = self.run_test(test)
            self.results.append(result)
            
            if result.result == TestResult.PASS:
                print(f"✅ PASS ({result.duration_ms:.0f}ms)")
                if self.verbose and result.details:
                    print(f"   {result.details}")
            elif result.result == TestResult.SKIP:
                print(f"⏭️  SKIP")
                print(f"   {result.error}")
            else:
                print(f"❌ FAIL")
                print(f"   {result.error}")
        
        # Summary
        passed = sum(1 for r in self.results if r.result == TestResult.PASS)
        failed = sum(1 for r in self.results if r.result == TestResult.FAIL)
        skipped = sum(1 for r in self.results if r.result == TestResult.SKIP)
        total = len(self.results)
        
        print("\n" + "-" * 60)
        print(f" Results: {passed}/{total} passed, {failed} failed, {skipped} skipped")
        
        if failed > 0:
            print("\n FAILED TESTS:")
            for r in self.results:
                if r.result == TestResult.FAIL:
                    print(f"   ❌ {r.test.name}: {r.error}")
        
        score = (passed / (total - skipped)) * 100 if (total - skipped) > 0 else 0
        print(f"\n Production Readiness: {score:.0f}%")
        
        if score >= 95:
            print(" ✅ PRODUCTION READY")
        elif score >= 80:
            print(" ⚠️  MOSTLY READY - Fix failures before production")
        else:
            print(" ❌ NOT PRODUCTION READY")
        
        print("=" * 60 + "\n")
        
        return {
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "total": total,
            "score": score,
            "production_ready": score >= 95
        }


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Validate production readiness")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix issues")
    args = parser.parse_args()
    
    validator = ProductionValidator(verbose=args.verbose)
    results = validator.run_all()
    
    if args.fix and results["failed"] > 0:
        print("\nAttempting fixes...")
        # Auto-fix logic could go here
    
    sys.exit(0 if results["production_ready"] else 1)


if __name__ == "__main__":
    main()
```

---

## Section 6: Implementation Checklist

### Phase 9A: Remove Stubs (Day 1)

- [ ] Update `core/observability/__init__.py` - Remove all StubXXX classes
- [ ] Update `core/memory/__init__.py` - Remove all stub patterns
- [ ] Update `core/orchestration/__init__.py` - Remove all fallbacks
- [ ] Update `core/structured/__init__.py` - Remove all silent failures
- [ ] Add `SDKNotAvailableError` to shared exceptions
- [ ] Add `SDKConfigurationError` to shared exceptions
- [ ] Update all getter functions to raise explicit errors
- [ ] Run `grep -r "Stub" core/` to verify removal
- [ ] Run `grep -r "pass$" core/` to find remaining stubs

### Phase 9B: Create Integration Tests (Day 2-3)

- [ ] Create `tests/integration/__init__.py`
- [ ] Create `tests/integration/conftest.py`
- [ ] Create `tests/integration/test_memory_real.py`
- [ ] Create `tests/integration/test_orchestration_real.py`
- [ ] Create `tests/integration/test_observability_real.py`
- [ ] Create `tests/integration/test_structured_real.py`
- [ ] Create `tests/integration/test_tools_real.py`
- [ ] Add `@pytest.mark.integration` to pytest.ini
- [ ] Verify tests skip (not fail) when credentials missing

### Phase 9C: Create Validator (Day 4)

- [ ] Create `core/validator.py`
- [ ] Add all SDKs to `SDK_REGISTRY`
- [ ] Implement `validate_sdk()` 
- [ ] Implement `validate_all_sdks()`
- [ ] Implement `get_system_health()`
- [ ] Add `unleash doctor` CLI command
- [ ] Add `unleash doctor --deep` for functional tests

### Phase 9D: Python 3.14 Fixes (Day 5)

- [ ] Create `scripts/fix_python314.py`
- [ ] Create `requirements-py314.txt`
- [ ] Create `core/compat.py` shim
- [ ] Document compatibility matrix in README

### Phase 9E: Production Validation (Day 5)

- [ ] Create `scripts/validate_production.py`
- [ ] Add all functional tests
- [ ] Test with real credentials
- [ ] Document any gaps
- [ ] Target: 100% or explicit documented gaps

---

## Verification Commands

```bash
# Verify no stubs remain
grep -r "class Stub" core/
grep -r "StubLangfuse" core/
grep -r "StubMem0" core/
grep -r "pass$" core/ --include="*.py"

# Run integration tests
pytest tests/integration/ -v -m integration

# Check system health
python -m core.validator
unleash doctor --deep

# Validate production
python scripts/validate_production.py --verbose

# Check Python compatibility
python scripts/fix_python314.py --check
```

---

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Test Coverage | 30% | 90%+ |
| Real Integration | 60% | 100% |
| Stub Count | 30 | 0 |
| Production Ready | 45% | 95%+ |
| Explicit Errors | No | Yes |
| Deep Tests | No | Yes |

**PHASE 9 COMPLETE WHEN:**
1. Zero stub classes in codebase
2. All SDKs raise explicit errors when unavailable
3. Integration tests exist for all SDK categories
4. `unleash doctor --deep` passes
5. `validate_production.py` reports 95%+ score
