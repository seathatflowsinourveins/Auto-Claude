"""
Unleashed Platform SDK Adapters - V40 Architecture
===================================================

This module provides unified adapters for integrating best-in-class SDKs.
All adapters follow a consistent interface pattern with:
- Lazy loading for optimal performance
- Explicit ImportError handling with clear messages
- Version checking for SDK dependencies
- Health check methods for monitoring
- Centralized registry for adapter management

Adapter Categories:
-------------------

Research Layer (V36.1):
- ExaAdapter: Neural semantic search with auto-prompts
- TavilyAdapter: AI-optimized web search
- PerplexityAdapter: Sonar conversational search
- JinaAdapter: Document reading and embedding
- FirecrawlAdapter: Web scraping and crawling
- Context7Adapter: SDK documentation search
- SerperAdapter: Google SERP results

Memory Layer:
- LettaAdapter: Letta memory management
- CogneeAdapter/CogneeV36Adapter: Cognee knowledge graphs
- Mem0Adapter: Mem0 memory backend
- GraphitiAdapter: Temporal knowledge graphs

Orchestration:
- DSPyAdapter: Optimization framework
- LangGraphAdapter: State graph orchestration
- OpenAIAgentsAdapter: OpenAI Agents SDK
- StrandsAgentsAdapter: AWS Strands Agents
- A2AProtocolAdapter: Google A2A Protocol

Code & Reasoning:
- AiderAdapter: AI pair programming
- LLMReasonersAdapter: Reasoning framework
- TextGradAdapter: Text optimization

Observability:
- OpikTracer: Distributed tracing
- BraintrustAdapter: Evaluation and logging
- PortkeyGatewayAdapter: LLM gateway

Workflows:
- TemporalSDKClient: Durable workflow execution

Utilities:
- RetryConfig: Retry configuration
- with_retry: Retry decorator
- AdapterRegistry: Centralized adapter management

Usage:
    # Using the registry (recommended)
    from adapters.registry import get_registry

    registry = get_registry()
    available = registry.list_available()
    health = await registry.health_check_all()

    # Direct adapter access
    from platform.adapters import ExaAdapter, TavilyAdapter

    adapter = ExaAdapter()
    await adapter.initialize({"api_key": "..."})
    result = await adapter.execute("search", query="...")
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional, Type, List

logger = logging.getLogger(__name__)

# =============================================================================
# Registry Import (Must be first)
# =============================================================================

try:
    from .registry import (
        AdapterRegistry,
        AdapterInfo,
        AdapterLoadStatus,
        HealthCheckResult,
        get_registry,
        register_adapter as registry_register,
    )
    REGISTRY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Adapter registry unavailable: {e}")
    REGISTRY_AVAILABLE = False
    AdapterRegistry = None
    AdapterInfo = None
    AdapterLoadStatus = None
    HealthCheckResult = None

    def get_registry():
        return None

    def registry_register(*args, **kwargs):
        pass


# =============================================================================
# Adapter availability tracking (legacy compatibility)
# =============================================================================

ADAPTER_STATUS: Dict[str, Dict[str, Any]] = {}


def register_adapter(name: str, available: bool, version: Optional[str] = None):
    """Register an adapter's availability status (legacy API)."""
    ADAPTER_STATUS[name] = {
        "available": available,
        "version": version,
        "initialized": False,
    }


def get_adapter_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all registered adapters (legacy API)."""
    return ADAPTER_STATUS.copy()


# =============================================================================
# SDK Version Checking Utilities
# =============================================================================

def check_sdk_version(
    package_name: str,
    min_version: Optional[str] = None,
    import_name: Optional[str] = None,
) -> tuple[bool, Optional[str], Optional[str]]:
    """
    Check if an SDK package is installed and meets version requirements.

    Args:
        package_name: PyPI package name (e.g., "exa-py")
        min_version: Minimum required version (e.g., "0.1.0")
        import_name: Python import name if different from package name

    Returns:
        Tuple of (available, version, error_message)
    """
    import importlib

    # Map package names to import names
    import_map = {
        "exa-py": "exa_py",
        "tavily-python": "tavily",
        "firecrawl-py": "firecrawl",
        "jina": "jina",
        "cognee": "cognee",
        "mem0ai": "mem0",
        "dspy-ai": "dspy",
        "langgraph": "langgraph",
        "httpx": "httpx",
    }

    actual_import = import_name or import_map.get(package_name, package_name.replace("-", "_"))

    try:
        module = importlib.import_module(actual_import)
        version = getattr(module, "__version__", None)

        # Try alternative version attributes
        if version is None:
            for attr in ["VERSION", "version", "__VERSION__"]:
                version = getattr(module, attr, None)
                if version:
                    break

        if min_version and version:
            # Simple version comparison
            def parse_version(v: str) -> tuple:
                try:
                    return tuple(int(x) for x in v.split(".")[:3] if x.isdigit() or x[0].isdigit())
                except (ValueError, IndexError):
                    return (0, 0, 0)

            if parse_version(str(version)) < parse_version(min_version):
                return (
                    False,
                    str(version),
                    f"Version {version} < required {min_version}. Upgrade with: pip install --upgrade {package_name}",
                )

        return True, str(version) if version else "unknown", None

    except ImportError as e:
        install_cmd = f"pip install {package_name}"
        if min_version:
            install_cmd += f">={min_version}"
        return (
            False,
            None,
            f"{package_name} not installed. Install with: {install_cmd}",
        )
    except Exception as e:
        return False, None, f"Error checking {package_name}: {e}"


# =============================================================================
# Adapter Loading with Explicit Error Handling
# =============================================================================

def _safe_import_adapter(
    module_path: str,
    class_name: str,
    availability_var: str,
    adapter_name: str,
    sdk_package: Optional[str] = None,
    min_version: Optional[str] = None,
) -> tuple[Optional[Type], bool, Optional[str]]:
    """
    Safely import an adapter with explicit error handling.

    Args:
        module_path: Relative module path (e.g., ".exa_adapter")
        class_name: Class name to import
        availability_var: Variable name for availability flag
        adapter_name: Human-readable adapter name
        sdk_package: Optional SDK package to check
        min_version: Minimum SDK version required

    Returns:
        Tuple of (adapter_class, is_available, error_message)
    """
    import importlib

    # Check SDK first if specified
    if sdk_package:
        available, version, error = check_sdk_version(sdk_package, min_version)
        if not available:
            logger.info(f"Adapter '{adapter_name}' unavailable: {error}")
            register_adapter(adapter_name, False, None)
            return None, False, error

    try:
        module = importlib.import_module(module_path, package=__name__)
        adapter_class = getattr(module, class_name)
        is_available = getattr(module, availability_var, True)

        if is_available:
            version = None
            if sdk_package:
                _, version, _ = check_sdk_version(sdk_package)
            register_adapter(adapter_name, True, version)
            logger.debug(f"Adapter '{adapter_name}' loaded successfully")
            return adapter_class, True, None
        else:
            error = f"SDK dependency not available (check {availability_var})"
            register_adapter(adapter_name, False, None)
            return adapter_class, False, error

    except ImportError as e:
        error = f"Import failed: {e}"
        logger.warning(f"Adapter '{adapter_name}' import error: {error}")
        register_adapter(adapter_name, False, None)
        return None, False, error

    except AttributeError as e:
        error = f"Class '{class_name}' not found in module"
        logger.error(f"Adapter '{adapter_name}' error: {error}")
        register_adapter(adapter_name, False, None)
        return None, False, error

    except Exception as e:
        error = f"Unexpected error: {e}"
        logger.error(f"Adapter '{adapter_name}' error: {error}")
        register_adapter(adapter_name, False, None)
        return None, False, error


# =============================================================================
# Factory Functions with Lazy Loading
# =============================================================================

def get_dspy_adapter():
    """Get DSPy adapter if available."""
    try:
        from .dspy_adapter import DSPyAdapter, DSPY_AVAILABLE
        if DSPY_AVAILABLE:
            return DSPyAdapter
        logger.info("DSPy adapter: SDK not available (pip install dspy-ai)")
        return None
    except ImportError as e:
        logger.info(f"DSPy adapter unavailable: {e}")
        return None


def get_langgraph_adapter():
    """Get LangGraph adapter if available."""
    try:
        from .langgraph_adapter import LangGraphAdapter, LANGGRAPH_AVAILABLE
        if LANGGRAPH_AVAILABLE:
            return LangGraphAdapter
        logger.info("LangGraph adapter: SDK not available (pip install langgraph)")
        return None
    except ImportError as e:
        logger.info(f"LangGraph adapter unavailable: {e}")
        return None


def get_mem0_adapter():
    """Get Mem0 adapter if available."""
    try:
        from .mem0_adapter import Mem0Adapter, MEM0_AVAILABLE
        if MEM0_AVAILABLE:
            return Mem0Adapter
        logger.info("Mem0 adapter: SDK not available (pip install mem0ai)")
        return None
    except ImportError as e:
        logger.info(f"Mem0 adapter unavailable: {e}")
        return None


def get_llm_reasoners_adapter():
    """Get llm-reasoners adapter if available."""
    try:
        from .llm_reasoners_adapter import LLMReasonersAdapter, LLM_REASONERS_AVAILABLE
        if LLM_REASONERS_AVAILABLE:
            return LLMReasonersAdapter
        logger.info("LLM Reasoners adapter: SDK not available (pip install llm-reasoners)")
        return None
    except ImportError as e:
        logger.info(f"LLM Reasoners adapter unavailable: {e}")
        return None


def get_evoagentx_adapter():
    """Get EvoAgentX adapter if available.

    Note: EvoAgentX adapter is deprecated - functionality moved to
    quality_diversity_adapter.py. This function returns None for
    backwards compatibility.
    """
    # DEPRECATED: evoagentx_adapter.py does not exist
    # Use quality_diversity_adapter.py instead for evolutionary algorithms
    logger.debug("EvoAgentX adapter deprecated - use quality_diversity_adapter instead")
    return None


def get_letta_adapter():
    """Get V36 Letta adapter if available."""
    try:
        from .letta_adapter import LettaAdapter
        return LettaAdapter
    except ImportError as e:
        logger.info(f"Letta adapter unavailable: {e}")
        return None


def get_textgrad_adapter():
    """Get TextGrad adapter if available."""
    try:
        from .textgrad_adapter import TextGradAdapter, TEXTGRAD_AVAILABLE
        if TEXTGRAD_AVAILABLE:
            return TextGradAdapter
        logger.info("TextGrad adapter: SDK not available (pip install textgrad)")
        return None
    except ImportError as e:
        logger.info(f"TextGrad adapter unavailable: {e}")
        return None


def get_aider_adapter():
    """Get Aider code assistant adapter if available."""
    try:
        from .aider_adapter import AiderAdapter, AIDER_AVAILABLE
        if AIDER_AVAILABLE:
            return AiderAdapter
        logger.info("Aider adapter: SDK not available (pip install aider-chat)")
        return None
    except ImportError as e:
        logger.info(f"Aider adapter unavailable: {e}")
        return None


# =============================================================================
# V1.0 SDK Integration Adapters (Voyage AI Foundation)
# =============================================================================

def get_letta_voyage_adapter():
    """Get Letta-Voyage integration adapter if available."""
    try:
        from .letta_voyage_adapter import LettaVoyageAdapter, LETTA_AVAILABLE, VOYAGE_AVAILABLE
        if LETTA_AVAILABLE and VOYAGE_AVAILABLE:
            return LettaVoyageAdapter
        missing = []
        if not LETTA_AVAILABLE:
            missing.append("letta")
        if not VOYAGE_AVAILABLE:
            missing.append("voyageai")
        logger.info(f"Letta-Voyage adapter: Missing SDKs: {', '.join(missing)}")
        return None
    except ImportError as e:
        logger.info(f"Letta-Voyage adapter unavailable: {e}")
        return None


def get_dspy_voyage_retriever():
    """Get DSPy-Voyage retriever if available."""
    try:
        from .dspy_voyage_retriever import VoyageRetriever, DSPY_AVAILABLE, VOYAGE_AVAILABLE
        if DSPY_AVAILABLE and VOYAGE_AVAILABLE:
            return VoyageRetriever
        missing = []
        if not DSPY_AVAILABLE:
            missing.append("dspy-ai")
        if not VOYAGE_AVAILABLE:
            missing.append("voyageai")
        logger.info(f"DSPy-Voyage retriever: Missing SDKs: {', '.join(missing)}")
        return None
    except ImportError as e:
        logger.info(f"DSPy-Voyage retriever unavailable: {e}")
        return None


def get_opik_tracer():
    """Get Opik tracer if available."""
    try:
        from .opik_tracing_adapter import OpikTracer, OPIK_AVAILABLE
        if OPIK_AVAILABLE:
            return OpikTracer
        logger.info("Opik tracer: SDK not available (pip install opik)")
        return None
    except ImportError as e:
        logger.info(f"Opik tracer unavailable: {e}")
        return None


def get_temporal_sdk_client():
    """Get Temporal SDK client if available."""
    try:
        from .temporal_workflow_activities import TemporalSDKClient, TEMPORAL_AVAILABLE
        if TEMPORAL_AVAILABLE:
            return TemporalSDKClient
        logger.info("Temporal SDK: Not available (pip install temporalio)")
        return None
    except ImportError as e:
        logger.info(f"Temporal SDK unavailable: {e}")
        return None


# =============================================================================
# V36 Architecture Adapters (New SDK Integrations)
# =============================================================================

def get_openai_agents_adapter():
    """Get OpenAI Agents SDK adapter if available."""
    try:
        from .openai_agents_adapter import OpenAIAgentsAdapter, SWARM_AVAILABLE
        if SWARM_AVAILABLE:
            return OpenAIAgentsAdapter
        logger.info("OpenAI Agents adapter: SDK not available")
        return None
    except ImportError as e:
        logger.info(f"OpenAI Agents adapter unavailable: {e}")
        return None


def get_cognee_adapter():
    """Get Cognee adapter if available."""
    try:
        from .cognee_adapter import CogneeAdapter, COGNEE_AVAILABLE
        if COGNEE_AVAILABLE:
            return CogneeAdapter
        logger.info("Cognee adapter: SDK not available (pip install cognee)")
    except ImportError:
        pass

    # Try V36 adapter as fallback
    try:
        from .cognee_v36_adapter import CogneeV36Adapter, COGNEE_AVAILABLE
        if COGNEE_AVAILABLE:
            return CogneeV36Adapter
        logger.info("Cognee V36 adapter: SDK not available (pip install cognee)")
    except ImportError as e:
        logger.info(f"Cognee adapters unavailable: {e}")

    return None


def get_mcp_apps_adapter():
    """Get MCP Apps adapter if available."""
    try:
        from .mcp_apps_adapter import MCPAppsAdapter
        return MCPAppsAdapter
    except ImportError as e:
        logger.info(f"MCP Apps adapter unavailable: {e}")
        return None


def get_graphiti_adapter():
    """Get Graphiti adapter if available."""
    try:
        from .graphiti_adapter import GraphitiAdapter, GRAPHITI_AVAILABLE
        if GRAPHITI_AVAILABLE:
            return GraphitiAdapter
        logger.info("Graphiti adapter: SDK not available (pip install graphiti-core)")
        return None
    except ImportError as e:
        logger.info(f"Graphiti adapter unavailable: {e}")
        return None


def get_strands_agents_adapter():
    """Get AWS Strands Agents adapter if available."""
    try:
        from .strands_agents_adapter import StrandsAgentsAdapter, STRANDS_AVAILABLE
        if STRANDS_AVAILABLE:
            return StrandsAgentsAdapter
        logger.info("Strands Agents adapter: SDK not available (pip install strands-agents)")
        return None
    except ImportError as e:
        logger.info(f"Strands Agents adapter unavailable: {e}")
        return None


def get_a2a_protocol_adapter():
    """Get Google A2A Protocol adapter if available."""
    try:
        from .a2a_protocol_adapter import A2AProtocolAdapter, A2A_AVAILABLE
        if A2A_AVAILABLE:
            return A2AProtocolAdapter
        logger.info("A2A Protocol adapter: SDK not available")
        return None
    except ImportError as e:
        logger.info(f"A2A Protocol adapter unavailable: {e}")
        return None


def get_ragflow_adapter():
    """Get RAGFlow adapter if available."""
    try:
        from .ragflow_adapter import RAGFlowAdapter, RAGFLOW_AVAILABLE
        if RAGFLOW_AVAILABLE:
            return RAGFlowAdapter
        logger.info("RAGFlow adapter: SDK not available")
        return None
    except ImportError as e:
        logger.info(f"RAGFlow adapter unavailable: {e}")
        return None


def get_simplemem_adapter():
    """Get SimpleMem adapter if available."""
    try:
        from .simplemem_adapter import SimpleMemAdapter, SIMPLEMEM_AVAILABLE
        if SIMPLEMEM_AVAILABLE:
            return SimpleMemAdapter
        logger.info("SimpleMem adapter: SDK not available")
        return None
    except ImportError as e:
        logger.info(f"SimpleMem adapter unavailable: {e}")
        return None


def get_ragatouille_adapter():
    """Get RAGatouille adapter if available."""
    try:
        from .ragatouille_adapter import RAGatouilleAdapter, RAGATOUILLE_AVAILABLE
        if RAGATOUILLE_AVAILABLE:
            return RAGatouilleAdapter
        logger.info("RAGatouille adapter: SDK not available (pip install ragatouille)")
        return None
    except ImportError as e:
        logger.info(f"RAGatouille adapter unavailable: {e}")
        return None


def get_braintrust_adapter():
    """Get Braintrust adapter if available."""
    try:
        from .braintrust_adapter import BraintrustAdapter, BRAINTRUST_AVAILABLE
        if BRAINTRUST_AVAILABLE:
            return BraintrustAdapter
        logger.info("Braintrust adapter: SDK not available (pip install braintrust)")
        return None
    except ImportError as e:
        logger.info(f"Braintrust adapter unavailable: {e}")
        return None


def get_portkey_gateway_adapter():
    """Get Portkey Gateway adapter if available."""
    try:
        from .portkey_gateway_adapter import PortkeyGatewayAdapter, PORTKEY_AVAILABLE
        if PORTKEY_AVAILABLE:
            return PortkeyGatewayAdapter
        logger.info("Portkey Gateway adapter: SDK not available (pip install portkey-ai)")
        return None
    except ImportError as e:
        logger.info(f"Portkey Gateway adapter unavailable: {e}")
        return None


# =============================================================================
# V36.1 Research Layer Adapters - Unleash Claude's Potential
# =============================================================================

def get_exa_adapter():
    """Get Exa neural search adapter if available."""
    try:
        from .exa_adapter import ExaAdapter, EXA_AVAILABLE
        if EXA_AVAILABLE:
            return ExaAdapter
        logger.info("Exa adapter: SDK not available (pip install exa-py)")
        return None
    except ImportError as e:
        logger.info(f"Exa adapter unavailable: {e}")
        return None


def get_tavily_adapter():
    """Get Tavily AI search adapter if available."""
    try:
        from .tavily_adapter import TavilyAdapter, TAVILY_AVAILABLE
        if TAVILY_AVAILABLE:
            return TavilyAdapter
        logger.info("Tavily adapter: SDK not available (pip install tavily-python)")
        return None
    except ImportError as e:
        logger.info(f"Tavily adapter unavailable: {e}")
        return None


def get_jina_adapter():
    """Get Jina Reader adapter if available."""
    try:
        from .jina_adapter import JinaAdapter
        return JinaAdapter
    except ImportError as e:
        logger.info(f"Jina adapter unavailable: {e}")
        return None


def get_perplexity_adapter():
    """Get Perplexity Sonar adapter if available."""
    try:
        from .perplexity_adapter import PerplexityAdapter
        return PerplexityAdapter
    except ImportError as e:
        logger.info(f"Perplexity adapter unavailable: {e}")
        return None


def get_firecrawl_adapter():
    """Get Firecrawl web scraping adapter if available."""
    try:
        from .firecrawl_adapter import FirecrawlAdapter, FIRECRAWL_AVAILABLE
        if FIRECRAWL_AVAILABLE:
            return FirecrawlAdapter
        logger.info("Firecrawl adapter: SDK not available (pip install firecrawl-py)")
        return None
    except ImportError as e:
        logger.info(f"Firecrawl adapter unavailable: {e}")
        return None


def get_context7_adapter():
    """Get Context7 SDK documentation adapter if available."""
    try:
        from .context7_adapter import Context7Adapter
        return Context7Adapter
    except ImportError as e:
        logger.info(f"Context7 adapter unavailable: {e}")
        return None


def get_serper_adapter():
    """Get Serper Google SERP adapter if available."""
    try:
        from .serper_adapter import SerperAdapter, SERPER_AVAILABLE
        if SERPER_AVAILABLE:
            return SerperAdapter
        logger.info("Serper adapter: API key required (SERPER_API_KEY)")
        return None
    except ImportError as e:
        logger.info(f"Serper adapter unavailable: {e}")
        return None


# =============================================================================
# Retry Utilities for Transient Error Handling
# =============================================================================

def get_retry_config():
    """Get RetryConfig class for configuring retry behavior."""
    try:
        from .retry import RetryConfig
        return RetryConfig
    except ImportError as e:
        logger.warning(f"RetryConfig unavailable: {e}")
        return None


def get_retry_decorator():
    """Get with_retry decorator for adding retry logic to functions."""
    try:
        from .retry import with_retry
        return with_retry
    except ImportError as e:
        logger.warning(f"with_retry unavailable: {e}")
        return None


def get_http_connection_pool():
    """Get HTTPConnectionPool class for creating connection pools."""
    try:
        from .http_pool import HTTPConnectionPool
        return HTTPConnectionPool
    except ImportError as e:
        logger.warning(f"HTTPConnectionPool unavailable: {e}")
        return None


def get_http_pool_stats():
    """
    Get statistics for all shared HTTP connection pools.

    Returns dict with metrics by base URL including:
    - total_requests: Total requests made
    - successful_requests: Successful request count
    - avg_wait_time_ms: Average connection wait time
    - connection_reuse_count: Times connections were reused

    Example:
        stats = get_http_pool_stats()
        for url, data in stats.items():
            print(f"{url}: {data['metrics']['total_requests']} requests")
    """
    try:
        from .http_pool import get_pool_stats
        return get_pool_stats()
    except ImportError as e:
        logger.warning(f"get_pool_stats unavailable: {e}")
        return {}


# =============================================================================
# Public API Exports
# =============================================================================

__all__ = [
    # === Registry (New) ===
    "AdapterRegistry",
    "AdapterInfo",
    "AdapterLoadStatus",
    "HealthCheckResult",
    "get_registry",

    # === Core Registration (Legacy) ===
    "ADAPTER_STATUS",
    "register_adapter",
    "get_adapter_status",

    # === Utilities ===
    "check_sdk_version",

    # === Factory Functions - Original Adapters ===
    "get_dspy_adapter",
    "get_langgraph_adapter",
    "get_mem0_adapter",
    "get_llm_reasoners_adapter",
    "get_evoagentx_adapter",  # Deprecated, returns None
    "get_textgrad_adapter",
    "get_aider_adapter",

    # === Factory Functions - V1.0 SDK Integration (Voyage Foundation) ===
    "get_letta_voyage_adapter",
    "get_dspy_voyage_retriever",
    "get_opik_tracer",
    "get_temporal_sdk_client",

    # === Factory Functions - V36 Architecture (Letta) ===
    "get_letta_adapter",

    # === Factory Functions - V36 Architecture (P0 Critical) ===
    "get_openai_agents_adapter",
    "get_cognee_adapter",
    "get_mcp_apps_adapter",

    # === Factory Functions - V36 Architecture (P1 Important) ===
    "get_graphiti_adapter",
    "get_strands_agents_adapter",
    "get_a2a_protocol_adapter",
    "get_ragflow_adapter",

    # === Factory Functions - V36 Architecture (P2 Specialized) ===
    "get_simplemem_adapter",
    "get_ragatouille_adapter",
    "get_braintrust_adapter",
    "get_portkey_gateway_adapter",

    # === Factory Functions - V36.1 Research Layer ===
    "get_exa_adapter",
    "get_tavily_adapter",
    "get_jina_adapter",
    "get_perplexity_adapter",
    "get_firecrawl_adapter",
    "get_context7_adapter",
    "get_serper_adapter",

    # === Retry Utilities ===
    "get_retry_config",
    "get_retry_decorator",

    # === HTTP Connection Pool ===
    "get_http_connection_pool",
    "get_http_pool_stats",

    # === Direct Adapter Class Exports (Lazy Loaded) ===
    # Research Layer Adapters
    "ExaAdapter",
    "TavilyAdapter",
    "PerplexityAdapter",
    "JinaAdapter",
    "FirecrawlAdapter",
    "Context7Adapter",
    "SerperAdapter",
    # Memory Layer Adapters
    "LettaAdapter",
    "CogneeAdapter",
    "CogneeV36Adapter",
    # Retry utilities
    "RetryConfig",
    "with_retry",
    "RetryWithCircuitBreakerConfig",
    "with_retry_and_circuit_breaker",
    "retry_async_with_circuit_breaker",
    "ResilientClient",
    "http_request_with_retry_and_circuit_breaker",
    # HTTP connection pool
    "HTTPConnectionPool",
    "PoolConfig",
    "PoolMetrics",
    "get_shared_pool",
    "get_pool_stats",
]


# =============================================================================
# Lazy Loading for Direct Adapter Imports
# =============================================================================

# Module mapping for lazy loading
_LAZY_IMPORTS = {
    # Research Layer Adapters
    "ExaAdapter": (".exa_adapter", "ExaAdapter"),
    "TavilyAdapter": (".tavily_adapter", "TavilyAdapter"),
    "PerplexityAdapter": (".perplexity_adapter", "PerplexityAdapter"),
    "JinaAdapter": (".jina_adapter", "JinaAdapter"),
    "FirecrawlAdapter": (".firecrawl_adapter", "FirecrawlAdapter"),
    "Context7Adapter": (".context7_adapter", "Context7Adapter"),
    "SerperAdapter": (".serper_adapter", "SerperAdapter"),
    # Memory Layer Adapters
    "LettaAdapter": (".letta_adapter", "LettaAdapter"),
    "CogneeAdapter": (".cognee_adapter", "CogneeAdapter"),
    "CogneeV36Adapter": (".cognee_v36_adapter", "CogneeV36Adapter"),
    # Retry utilities
    "RetryConfig": (".retry", "RetryConfig"),
    "with_retry": (".retry", "with_retry"),
    "RetryWithCircuitBreakerConfig": (".retry", "RetryWithCircuitBreakerConfig"),
    "with_retry_and_circuit_breaker": (".retry", "with_retry_and_circuit_breaker"),
    "retry_async_with_circuit_breaker": (".retry", "retry_async_with_circuit_breaker"),
    "ResilientClient": (".retry", "ResilientClient"),
    "http_request_with_retry_and_circuit_breaker": (".retry", "http_request_with_retry_and_circuit_breaker"),
    # HTTP connection pool
    "HTTPConnectionPool": (".http_pool", "HTTPConnectionPool"),
    "PoolConfig": (".http_pool", "PoolConfig"),
    "PoolMetrics": (".http_pool", "PoolMetrics"),
    "get_shared_pool": (".http_pool", "get_shared_pool"),
    "get_pool_stats": (".http_pool", "get_pool_stats"),
    # Adaptive timeout
    "AdaptiveTimeout": ("..core.adaptive_timeout", "AdaptiveTimeout"),
    "TimeoutProfile": ("..core.adaptive_timeout", "TimeoutProfile"),
    "LatencyStats": ("..core.adaptive_timeout", "LatencyStats"),
    "AdaptiveTimeoutMiddleware": ("..core.adaptive_timeout", "AdaptiveTimeoutMiddleware"),
    "TimeoutAlertManager": ("..core.adaptive_timeout", "TimeoutAlertManager"),
    "get_adaptive_timeout": ("..core.adaptive_timeout", "get_adaptive_timeout"),
    "get_adaptive_timeout_sync": ("..core.adaptive_timeout", "get_adaptive_timeout_sync"),
    "get_all_timeout_stats": ("..core.adaptive_timeout", "get_all_timeout_stats"),
    "reset_all_timeouts": ("..core.adaptive_timeout", "reset_all_timeouts"),
    "get_timeout_alert_manager": ("..core.adaptive_timeout", "get_timeout_alert_manager"),
}

_loaded_cache: Dict[str, Any] = {}


def __getattr__(name: str) -> Any:
    """Lazy load adapter classes on first access with informative error messages."""
    if name in _loaded_cache:
        return _loaded_cache[name]

    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        try:
            import importlib
            module = importlib.import_module(module_path, package=__name__)
            value = getattr(module, attr_name)
            _loaded_cache[name] = value
            return value
        except ImportError as e:
            # Provide helpful error message
            raise ImportError(
                f"Cannot import {name}: {e}\n"
                f"This adapter may have missing SDK dependencies. "
                f"Check the adapter module for required packages."
            ) from e
        except AttributeError as e:
            raise AttributeError(
                f"Cannot import {name}: class not found in module {module_path}\n"
                f"The adapter module may be misconfigured."
            ) from e

    raise AttributeError(f"module 'platform.adapters' has no attribute '{name}'")


def __dir__():
    """Return all available names for tab completion."""
    return list(__all__)


# =============================================================================
# Registry Auto-Population
# =============================================================================

def _populate_registry():
    """
    Populate the adapter registry with all known adapters.

    This runs on module import to register all adapters with the
    centralized registry for health monitoring and status tracking.
    """
    if not REGISTRY_AVAILABLE:
        return

    registry = get_registry()
    if registry is None:
        return

    # Define all adapters with their metadata
    _ADAPTER_DEFINITIONS = [
        # Research Layer
        ("exa", ".exa_adapter", "ExaAdapter", "EXA_AVAILABLE", "exa-py", None, "RESEARCH", 25, ["search", "find_similar", "answer", "research"]),
        ("tavily", ".tavily_adapter", "TavilyAdapter", "TAVILY_AVAILABLE", "tavily-python", None, "RESEARCH", 24, ["search", "research", "extract", "qna"]),
        ("perplexity", ".perplexity_adapter", "PerplexityAdapter", None, None, None, "RESEARCH", 22, ["chat", "reasoning", "deep_research"]),
        ("jina", ".jina_adapter", "JinaAdapter", None, None, None, "RESEARCH", 23, ["read", "search", "embed", "rerank"]),
        ("firecrawl", ".firecrawl_adapter", "FirecrawlAdapter", "FIRECRAWL_AVAILABLE", "firecrawl-py", None, "RESEARCH", 23, ["scrape", "crawl", "map", "search"]),
        ("context7", ".context7_adapter", "Context7Adapter", None, None, None, "RESEARCH", 20, ["search_docs"]),
        ("serper", ".serper_adapter", "SerperAdapter", "SERPER_AVAILABLE", None, None, "RESEARCH", 21, ["search", "knowledge_graph", "images", "news", "videos", "places", "maps", "shopping", "scholar", "patents", "autocomplete"]),

        # Memory Layer
        ("letta", ".letta_adapter", "LettaAdapter", None, None, None, "MEMORY", 22, ["create_agent", "send_message"]),
        ("cognee", ".cognee_adapter", "CogneeAdapter", "COGNEE_AVAILABLE", "cognee", None, "KNOWLEDGE", 22, ["ingest", "search", "search_multi"]),
        ("mem0", ".mem0_adapter", "Mem0Adapter", "MEM0_AVAILABLE", "mem0ai", None, "MEMORY", 20, ["add", "search", "get"]),
        ("graphiti", ".graphiti_adapter", "GraphitiAdapter", "GRAPHITI_AVAILABLE", "graphiti-core", None, "KNOWLEDGE", 21, ["add_episode", "search"]),

        # Orchestration
        ("dspy", ".dspy_adapter", "DSPyAdapter", "DSPY_AVAILABLE", "dspy-ai", None, "ORCHESTRATION", 25, ["optimize", "predict"]),
        ("langgraph", ".langgraph_adapter", "LangGraphAdapter", "LANGGRAPH_AVAILABLE", "langgraph", None, "ORCHESTRATION", 24, ["compile", "invoke"]),
        ("openai_agents", ".openai_agents_adapter", "OpenAIAgentsAdapter", "SWARM_AVAILABLE", None, None, "ORCHESTRATION", 23, ["run_agent"]),
        ("strands_agents", ".strands_agents_adapter", "StrandsAgentsAdapter", "STRANDS_AVAILABLE", "strands-agents", None, "ORCHESTRATION", 21, ["invoke"]),
        ("a2a_protocol", ".a2a_protocol_adapter", "A2AProtocolAdapter", "A2A_AVAILABLE", None, None, "ORCHESTRATION", 20, ["send_task"]),

        # Code & Reasoning
        ("aider", ".aider_adapter", "AiderAdapter", "AIDER_AVAILABLE", "aider-chat", None, "CODE", 22, ["edit", "ask"]),
        ("llm_reasoners", ".llm_reasoners_adapter", "LLMReasonersAdapter", "LLM_REASONERS_AVAILABLE", "llm-reasoners", None, "REASONING", 21, ["reason"]),
        ("textgrad", ".textgrad_adapter", "TextGradAdapter", "TEXTGRAD_AVAILABLE", "textgrad", None, "OPTIMIZATION", 20, ["optimize"]),

        # Observability
        ("opik", ".opik_tracing_adapter", "OpikTracer", "OPIK_AVAILABLE", "opik", None, "OBSERVABILITY", 22, ["trace", "log"]),
        ("braintrust", ".braintrust_adapter", "BraintrustAdapter", "BRAINTRUST_AVAILABLE", "braintrust", None, "OBSERVABILITY", 21, ["eval", "log"]),
        ("portkey", ".portkey_gateway_adapter", "PortkeyGatewayAdapter", "PORTKEY_AVAILABLE", "portkey-ai", None, "GATEWAY", 20, ["route", "fallback"]),

        # Workflows
        ("temporal", ".temporal_workflow_activities", "TemporalSDKClient", "TEMPORAL_AVAILABLE", "temporalio", None, "WORKFLOW", 23, ["start_workflow", "execute_activity"]),

        # Specialized
        ("ragflow", ".ragflow_adapter", "RAGFlowAdapter", "RAGFLOW_AVAILABLE", None, None, "RAG", 20, ["query", "index"]),
        ("simplemem", ".simplemem_adapter", "SimpleMemAdapter", "SIMPLEMEM_AVAILABLE", None, None, "MEMORY", 18, ["store", "retrieve"]),
        ("ragatouille", ".ragatouille_adapter", "RAGatouilleAdapter", "RAGATOUILLE_AVAILABLE", "ragatouille", None, "RAG", 19, ["index", "search"]),
    ]

    for adapter_def in _ADAPTER_DEFINITIONS:
        name, module_path, class_name, avail_var, sdk_pkg, min_ver, layer, priority, features = adapter_def

        try:
            import importlib
            module = importlib.import_module(module_path, package=__name__)
            adapter_class = getattr(module, class_name, None)

            # Check availability flag if specified
            is_available = True
            if avail_var:
                is_available = getattr(module, avail_var, False)

            if adapter_class and is_available:
                registry.register_adapter(
                    name=name,
                    adapter_class=adapter_class,
                    sdk_name=sdk_pkg,
                    min_version=min_ver,
                    layer=layer,
                    priority=priority,
                    features=features,
                )
            elif adapter_class:
                # Class exists but SDK not available
                registry.register_adapter(
                    name=name,
                    adapter_class=None,
                    sdk_name=sdk_pkg,
                    min_version=min_ver,
                    layer=layer,
                    priority=priority,
                    features=features,
                )

        except ImportError:
            # Silently skip - adapter module not available
            pass
        except Exception as e:
            logger.debug(f"Error registering adapter '{name}': {e}")


# Run auto-population on import
_populate_registry()


# =============================================================================
# Startup Status Logging
# =============================================================================

def log_adapter_status(level: int = logging.INFO) -> None:
    """
    Log the status of all adapters at startup.

    Call this from your application startup to see which adapters
    are available and which have missing dependencies.
    """
    if REGISTRY_AVAILABLE:
        registry = get_registry()
        if registry:
            registry.log_startup_status(level)
    else:
        # Fallback to basic status
        available = [name for name, info in ADAPTER_STATUS.items() if info.get("available")]
        unavailable = [name for name, info in ADAPTER_STATUS.items() if not info.get("available")]
        logger.log(level, f"Adapters: {len(available)} available, {len(unavailable)} unavailable")
        if available:
            logger.log(level, f"  Available: {', '.join(available)}")
        if unavailable:
            logger.log(level, f"  Unavailable: {', '.join(unavailable)}")


async def health_check_all_adapters(timeout: float = 30.0) -> Dict[str, Any]:
    """
    Run health checks on all available adapters.

    Args:
        timeout: Maximum time for all health checks

    Returns:
        Dict with health check results for each adapter
    """
    if REGISTRY_AVAILABLE:
        registry = get_registry()
        if registry:
            results = await registry.health_check_all(timeout=timeout)
            return {
                name: {
                    "healthy": r.is_healthy,
                    "status": r.status,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                }
                for name, r in results.items()
            }

    # Fallback - no health checks available
    return {"error": "Registry not available"}


# =============================================================================
# Circuit Breaker Status
# =============================================================================

def get_circuit_breaker_status() -> Dict[str, Any]:
    """
    Get circuit breaker status for all research adapters.

    Returns:
        Dict with circuit breaker state for each adapter:
        - state: CLOSED, OPEN, or HALF_OPEN
        - failure_count: Number of consecutive failures
        - last_failure_time: Timestamp of last failure
        - is_healthy: True if circuit is not OPEN
    """
    try:
        from .circuit_breaker_manager import get_adapter_circuit_manager
        manager = get_adapter_circuit_manager()
        if manager:
            health_data = manager.get_all_health()
            return {
                name: {
                    "state": health.state.value if health else "unknown",
                    "failure_count": health.failure_count if health else 0,
                    "success_count": health.success_count if health else 0,
                    "is_healthy": health.is_healthy if health else True,
                    "last_failure_time": str(health.last_failure_time) if health and health.last_failure_time else None,
                }
                for name, health in health_data.items()
            }
    except ImportError:
        pass
    return {"error": "Circuit breaker manager not available"}


def get_circuit_breaker_summary() -> Dict[str, Any]:
    """
    Get summary statistics for all circuit breakers.

    Returns:
        Dict with aggregated stats:
        - total_adapters: Number of adapters with circuit breakers
        - healthy_count: Number of healthy circuits
        - unhealthy_count: Number of open circuits
        - health_percentage: Percentage of healthy circuits
        - unhealthy_adapters: List of adapter names with open circuits
    """
    try:
        from .circuit_breaker_manager import get_adapter_circuit_manager
        manager = get_adapter_circuit_manager()
        if manager:
            return manager.get_stats_summary()
    except ImportError:
        pass
    return {"error": "Circuit breaker manager not available"}


def reset_circuit_breaker(adapter_name: str) -> bool:
    """
    Force reset a circuit breaker for an adapter.

    Args:
        adapter_name: Name of the adapter (e.g., "exa_adapter")

    Returns:
        True if reset successful, False otherwise
    """
    try:
        from .circuit_breaker_manager import get_adapter_circuit_manager
        manager = get_adapter_circuit_manager()
        if manager:
            return manager.reset_breaker(adapter_name)
    except ImportError:
        pass
    return False


def reset_all_circuit_breakers() -> int:
    """
    Reset all circuit breakers to CLOSED state.

    Returns:
        Number of circuit breakers reset
    """
    try:
        from .circuit_breaker_manager import get_adapter_circuit_manager
        manager = get_adapter_circuit_manager()
        if manager:
            return manager.reset_all()
    except ImportError:
        pass
    return 0


# Add to exports
__all__.extend([
    "log_adapter_status",
    "health_check_all_adapters",
    "get_circuit_breaker_status",
    "get_circuit_breaker_summary",
    "reset_circuit_breaker",
    "reset_all_circuit_breakers",
])


# =============================================================================
# Rate Limiter Integration
# =============================================================================

try:
    from ..core.rate_limiter import (
        RateLimiter,
        RateLimitConfig,
        RateLimitStats,
        RateLimitState,
        RateLimitExceeded,
        get_rate_limiter,
        configure_rate_limit,
        rate_limited,
        with_rate_limit,
        parse_rate_limit_headers,
        DEFAULT_RATE_LIMITS,
    )
    RATE_LIMITER_AVAILABLE = True
except ImportError:
    RATE_LIMITER_AVAILABLE = False
    RateLimiter = None
    RateLimitConfig = None
    RateLimitStats = None
    RateLimitState = None
    RateLimitExceeded = None
    get_rate_limiter = None
    configure_rate_limit = None
    rate_limited = None
    with_rate_limit = None
    parse_rate_limit_headers = None
    DEFAULT_RATE_LIMITS = None


def get_rate_limit_status() -> Dict[str, Any]:
    """
    Get rate limit status for all adapters.

    Returns:
        Dict with rate limit state for each adapter:
        - state: NORMAL, THROTTLED, BACKOFF, or SUSPENDED
        - total_requests: Total requests made
        - allowed_requests: Requests allowed
        - rejected_requests: Requests rejected
        - tokens_available: Current tokens in bucket
    """
    if not RATE_LIMITER_AVAILABLE:
        return {"error": "Rate limiter not available"}

    limiter = get_rate_limiter()
    return limiter.get_stats()


def get_rate_limit_summary() -> Dict[str, Any]:
    """
    Get summary statistics for all rate limiters.

    Returns:
        Dict with aggregated stats:
        - total_requests: Total requests across all adapters
        - total_allowed: Total allowed requests
        - total_rejected: Total rejected requests
        - success_rate: Percentage of allowed requests
        - throttled_adapters: List of throttled adapter names
    """
    if not RATE_LIMITER_AVAILABLE:
        return {"error": "Rate limiter not available"}

    limiter = get_rate_limiter()
    return limiter.get_summary()


def reset_rate_limiter(adapter_name: str = None) -> bool:
    """
    Reset rate limiter for an adapter or all adapters.

    Args:
        adapter_name: Name of adapter to reset. If None, resets all.

    Returns:
        True if reset successful
    """
    if not RATE_LIMITER_AVAILABLE:
        return False

    limiter = get_rate_limiter()
    limiter.reset(adapter_name)
    return True


# Add rate limiter exports to __all__
__all__.extend([
    # Rate limiter classes
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitStats",
    "RateLimitState",
    "RateLimitExceeded",
    # Rate limiter functions
    "get_rate_limiter",
    "configure_rate_limit",
    "rate_limited",
    "with_rate_limit",
    "parse_rate_limit_headers",
    # Rate limiter status functions
    "get_rate_limit_status",
    "get_rate_limit_summary",
    "reset_rate_limiter",
    # Constants
    "DEFAULT_RATE_LIMITS",
    "RATE_LIMITER_AVAILABLE",
])


# =============================================================================
# Request Deduplication Integration
# =============================================================================

try:
    from ..core.deduplication import (
        DeduplicationConfig,
        DeduplicationStrategy,
        DeduplicationStats,
        RequestDeduplicator,
        get_deduplicator,
        get_deduplicator_sync,
        reset_deduplicator,
        deduplicated,
        DeduplicatedAdapter,
    )
    DEDUPLICATION_AVAILABLE = True
except ImportError:
    DEDUPLICATION_AVAILABLE = False
    DeduplicationConfig = None
    DeduplicationStrategy = None
    DeduplicationStats = None
    RequestDeduplicator = None
    get_deduplicator = None
    get_deduplicator_sync = None
    reset_deduplicator = None
    deduplicated = None
    DeduplicatedAdapter = None


def get_deduplication_status() -> Dict[str, Any]:
    """
    Get deduplication statistics for all adapters.

    Returns:
        Dict with deduplication metrics:
        - total_requests: Total requests processed
        - cache_hits: Requests served from cache
        - in_flight_hits: Requests that joined in-flight
        - hit_rate: Overall deduplication hit rate
        - bytes_saved: Estimated bytes saved
    """
    if not DEDUPLICATION_AVAILABLE:
        return {"error": "Deduplication not available"}

    deduplicator = get_deduplicator_sync()
    return deduplicator.get_stats()


async def clear_deduplication_cache() -> int:
    """
    Clear the deduplication cache.

    Returns:
        Number of entries cleared
    """
    if not DEDUPLICATION_AVAILABLE:
        return 0

    deduplicator = get_deduplicator_sync()
    return await deduplicator.clear()


async def invalidate_adapter_cache(adapter_name: str) -> int:
    """
    Invalidate cached entries for a specific adapter.

    Args:
        adapter_name: Name of the adapter

    Returns:
        Number of entries invalidated
    """
    if not DEDUPLICATION_AVAILABLE:
        return 0

    deduplicator = get_deduplicator_sync()
    return await deduplicator.invalidate(adapter_name=adapter_name)


# Add deduplication exports to __all__
__all__.extend([
    # Deduplication classes
    "DeduplicationConfig",
    "DeduplicationStrategy",
    "DeduplicationStats",
    "RequestDeduplicator",
    "DeduplicatedAdapter",
    # Deduplication functions
    "get_deduplicator",
    "get_deduplicator_sync",
    "reset_deduplicator",
    "deduplicated",
    # Deduplication status functions
    "get_deduplication_status",
    "clear_deduplication_cache",
    "invalidate_adapter_cache",
    # Constants
    "DEDUPLICATION_AVAILABLE",
])


# =============================================================================
# Adaptive Timeout Integration
# =============================================================================

try:
    from ..core.adaptive_timeout import (
        AdaptiveTimeout,
        TimeoutProfile,
        LatencyStats,
        get_adaptive_timeout,
        get_adaptive_timeout_sync,
        get_all_timeout_stats,
        reset_all_timeouts,
        AdaptiveTimeoutMiddleware,
        TimeoutAlertManager,
        get_timeout_alert_manager,
        DEFAULT_PROFILES as TIMEOUT_PROFILES,
    )
    ADAPTIVE_TIMEOUT_AVAILABLE = True
except ImportError:
    ADAPTIVE_TIMEOUT_AVAILABLE = False
    AdaptiveTimeout = None
    TimeoutProfile = None
    LatencyStats = None
    get_adaptive_timeout = None
    get_adaptive_timeout_sync = None
    get_all_timeout_stats = None
    reset_all_timeouts = None
    AdaptiveTimeoutMiddleware = None
    TimeoutAlertManager = None
    get_timeout_alert_manager = None
    TIMEOUT_PROFILES = None


def get_adaptive_timeout_status() -> Dict[str, Any]:
    """
    Get adaptive timeout statistics for all tracked operations.

    Returns:
        Dict with timeout stats for each operation:
        - sample_count: Number of latency samples
        - p99_ms: 99th percentile latency
        - current_timeout_ms: Current calculated timeout
        - is_warmed_up: Whether enough samples collected
    """
    if not ADAPTIVE_TIMEOUT_AVAILABLE:
        return {"error": "Adaptive timeout not available"}

    return get_all_timeout_stats()


def get_adaptive_timeout_summary() -> Dict[str, Any]:
    """
    Get summary of adaptive timeout behavior.

    Returns:
        Dict with aggregated stats:
        - total_operations: Number of tracked operations
        - warmed_up_count: Operations with enough samples
        - total_adjustments: Total timeout adjustments made
        - avg_p99_ms: Average p99 latency across operations
    """
    if not ADAPTIVE_TIMEOUT_AVAILABLE:
        return {"error": "Adaptive timeout not available"}

    all_stats = get_all_timeout_stats()
    if not all_stats:
        return {
            "total_operations": 0,
            "warmed_up_count": 0,
            "total_adjustments": 0,
            "avg_p99_ms": 0.0,
        }

    warmed_up = [s for s in all_stats.values() if s.get("is_warmed_up", False)]
    total_adjustments = sum(s.get("timeout_adjustments", 0) for s in all_stats.values())
    p99_values = [s.get("p99_ms", 0) for s in warmed_up if s.get("p99_ms", 0) > 0]

    return {
        "total_operations": len(all_stats),
        "warmed_up_count": len(warmed_up),
        "total_adjustments": total_adjustments,
        "avg_p99_ms": sum(p99_values) / len(p99_values) if p99_values else 0.0,
        "operations": list(all_stats.keys()),
    }


def reset_adaptive_timeout(operation_name: str = None) -> int:
    """
    Reset adaptive timeout statistics.

    Args:
        operation_name: Specific operation to reset. If None, resets all.

    Returns:
        Number of operations reset
    """
    if not ADAPTIVE_TIMEOUT_AVAILABLE:
        return 0

    if operation_name:
        handler = get_adaptive_timeout_sync(operation_name)
        if handler:
            handler.reset()
            return 1
        return 0
    else:
        return reset_all_timeouts()


# Add adaptive timeout exports to __all__
__all__.extend([
    # Adaptive timeout classes
    "AdaptiveTimeout",
    "TimeoutProfile",
    "LatencyStats",
    "AdaptiveTimeoutMiddleware",
    "TimeoutAlertManager",
    # Adaptive timeout functions
    "get_adaptive_timeout",
    "get_adaptive_timeout_sync",
    "get_all_timeout_stats",
    "reset_all_timeouts",
    "get_timeout_alert_manager",
    # Adaptive timeout status functions
    "get_adaptive_timeout_status",
    "get_adaptive_timeout_summary",
    "reset_adaptive_timeout",
    # Constants
    "TIMEOUT_PROFILES",
    "ADAPTIVE_TIMEOUT_AVAILABLE",
])
