"""
Unleashed Platform SDK Adapters

This module provides unified adapters for integrating best-in-class SDKs:
- DSPy (Optimization) + Voyage Retriever
- LangGraph (Orchestration)
- Mem0/Letta/Graphiti (Memory) + Voyage Integration
- llm-reasoners (Reasoning)
- Exa/Firecrawl/Crawl4AI (Research)
- Aider/Serena (Code)
- EvoAgentX (Self-Improvement)
- Opik (Tracing/Observability)
- Temporal (Durable Workflows)

Each adapter provides a consistent interface while leveraging SDK-specific features.
The V1.0 SDK Integration wires all adapters to the Voyage AI embedding foundation.
"""

from typing import Optional, Dict, Any

# Adapter availability tracking
ADAPTER_STATUS: Dict[str, Dict[str, Any]] = {}


def register_adapter(name: str, available: bool, version: Optional[str] = None):
    """Register an adapter's availability status."""
    ADAPTER_STATUS[name] = {
        "available": available,
        "version": version,
        "initialized": False,
    }


def get_adapter_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all registered adapters."""
    return ADAPTER_STATUS.copy()


# Lazy imports with availability checks
def get_dspy_adapter():
    """Get DSPy adapter if available."""
    try:
        from .dspy_adapter import DSPyAdapter, DSPY_AVAILABLE
        return DSPyAdapter if DSPY_AVAILABLE else None
    except ImportError:
        return None


def get_langgraph_adapter():
    """Get LangGraph adapter if available."""
    try:
        from .langgraph_adapter import LangGraphAdapter, LANGGRAPH_AVAILABLE
        return LangGraphAdapter if LANGGRAPH_AVAILABLE else None
    except ImportError:
        return None


def get_mem0_adapter():
    """Get Mem0 adapter if available."""
    try:
        from .mem0_adapter import Mem0Adapter, MEM0_AVAILABLE
        return Mem0Adapter if MEM0_AVAILABLE else None
    except ImportError:
        return None


def get_llm_reasoners_adapter():
    """Get llm-reasoners adapter if available."""
    try:
        from .llm_reasoners_adapter import LLMReasonersAdapter, LLM_REASONERS_AVAILABLE
        return LLMReasonersAdapter if LLM_REASONERS_AVAILABLE else None
    except ImportError:
        return None


def get_evoagentx_adapter():
    """Get EvoAgentX adapter if available.

    Note: EvoAgentX adapter is deprecated - functionality moved to
    quality_diversity_adapter.py. This function returns None for
    backwards compatibility.
    """
    # DEPRECATED: evoagentx_adapter.py does not exist
    # Use quality_diversity_adapter.py instead for evolutionary algorithms
    return None


def get_letta_adapter():
    """Get V36 Letta adapter if available."""
    try:
        from .letta_adapter import LettaAdapter
        return LettaAdapter
    except ImportError:
        return None


def get_textgrad_adapter():
    """Get TextGrad adapter if available."""
    try:
        from .textgrad_adapter import TextGradAdapter, TEXTGRAD_AVAILABLE
        return TextGradAdapter if TEXTGRAD_AVAILABLE else None
    except ImportError:
        return None


def get_aider_adapter():
    """Get Aider code assistant adapter if available."""
    try:
        from .aider_adapter import AiderAdapter, AIDER_AVAILABLE
        return AiderAdapter if AIDER_AVAILABLE else None
    except ImportError:
        return None


# =============================================================================
# V1.0 SDK Integration Adapters (Voyage AI Foundation)
# =============================================================================

def get_letta_voyage_adapter():
    """Get Letta-Voyage integration adapter if available."""
    try:
        from .letta_voyage_adapter import LettaVoyageAdapter, LETTA_AVAILABLE, VOYAGE_AVAILABLE
        return LettaVoyageAdapter if (LETTA_AVAILABLE and VOYAGE_AVAILABLE) else None
    except ImportError:
        return None


def get_dspy_voyage_retriever():
    """Get DSPy-Voyage retriever if available."""
    try:
        from .dspy_voyage_retriever import VoyageRetriever, DSPY_AVAILABLE, VOYAGE_AVAILABLE
        return VoyageRetriever if (DSPY_AVAILABLE and VOYAGE_AVAILABLE) else None
    except ImportError:
        return None


def get_opik_tracer():
    """Get Opik tracer if available."""
    try:
        from .opik_tracing_adapter import OpikTracer, OPIK_AVAILABLE
        return OpikTracer if OPIK_AVAILABLE else None
    except ImportError:
        return None


def get_temporal_sdk_client():
    """Get Temporal SDK client if available."""
    try:
        from .temporal_workflow_activities import TemporalSDKClient, TEMPORAL_AVAILABLE
        return TemporalSDKClient if TEMPORAL_AVAILABLE else None
    except ImportError:
        return None


# =============================================================================
# V36 Architecture Adapters (New SDK Integrations)
# =============================================================================

def get_openai_agents_adapter():
    """Get OpenAI Agents SDK adapter if available."""
    try:
        from .openai_agents_adapter import OpenAIAgentsAdapter, SWARM_AVAILABLE
        return OpenAIAgentsAdapter
    except ImportError:
        return None


def get_cognee_adapter():
    """Get Cognee V36 adapter if available."""
    try:
        from .cognee_v36_adapter import CogneeV36Adapter, COGNEE_AVAILABLE
        return CogneeV36Adapter
    except ImportError:
        return None


def get_mcp_apps_adapter():
    """Get MCP Apps adapter if available."""
    try:
        from .mcp_apps_adapter import MCPAppsAdapter
        return MCPAppsAdapter
    except ImportError:
        return None


def get_graphiti_adapter():
    """Get Graphiti adapter if available."""
    try:
        from .graphiti_adapter import GraphitiAdapter, GRAPHITI_AVAILABLE
        return GraphitiAdapter
    except ImportError:
        return None


def get_strands_agents_adapter():
    """Get AWS Strands Agents adapter if available."""
    try:
        from .strands_agents_adapter import StrandsAgentsAdapter, STRANDS_AVAILABLE
        return StrandsAgentsAdapter
    except ImportError:
        return None


def get_a2a_protocol_adapter():
    """Get Google A2A Protocol adapter if available."""
    try:
        from .a2a_protocol_adapter import A2AProtocolAdapter, A2A_AVAILABLE
        return A2AProtocolAdapter
    except ImportError:
        return None


def get_ragflow_adapter():
    """Get RAGFlow adapter if available."""
    try:
        from .ragflow_adapter import RAGFlowAdapter, RAGFLOW_AVAILABLE
        return RAGFlowAdapter
    except ImportError:
        return None


def get_simplemem_adapter():
    """Get SimpleMem adapter if available."""
    try:
        from .simplemem_adapter import SimpleMemAdapter, SIMPLEMEM_AVAILABLE
        return SimpleMemAdapter
    except ImportError:
        return None


def get_ragatouille_adapter():
    """Get RAGatouille adapter if available."""
    try:
        from .ragatouille_adapter import RAGatouilleAdapter, RAGATOUILLE_AVAILABLE
        return RAGatouilleAdapter
    except ImportError:
        return None


def get_braintrust_adapter():
    """Get Braintrust adapter if available."""
    try:
        from .braintrust_adapter import BraintrustAdapter, BRAINTRUST_AVAILABLE
        return BraintrustAdapter
    except ImportError:
        return None


def get_portkey_gateway_adapter():
    """Get Portkey Gateway adapter if available."""
    try:
        from .portkey_gateway_adapter import PortkeyGatewayAdapter, PORTKEY_AVAILABLE
        return PortkeyGatewayAdapter
    except ImportError:
        return None


__all__ = [
    # Core registration
    "ADAPTER_STATUS",
    "register_adapter",
    "get_adapter_status",
    # Original adapters
    "get_dspy_adapter",
    "get_langgraph_adapter",
    "get_mem0_adapter",
    "get_llm_reasoners_adapter",
    "get_evoagentx_adapter",  # Deprecated, returns None
    "get_textgrad_adapter",
    "get_aider_adapter",
    # V1.0 SDK Integration adapters (Voyage foundation)
    "get_letta_voyage_adapter",
    "get_dspy_voyage_retriever",
    "get_opik_tracer",
    "get_temporal_sdk_client",
    # V36 Architecture adapters - Letta
    "get_letta_adapter",
    # V36 Architecture adapters - P0 Critical
    "get_openai_agents_adapter",
    "get_cognee_adapter",
    "get_mcp_apps_adapter",
    # V36 Architecture adapters - P1 Important
    "get_graphiti_adapter",
    "get_strands_agents_adapter",
    "get_a2a_protocol_adapter",
    "get_ragflow_adapter",
    # V36 Architecture adapters - P2 Specialized
    "get_simplemem_adapter",
    "get_ragatouille_adapter",
    "get_braintrust_adapter",
    "get_portkey_gateway_adapter",
]


# Auto-register adapters on import (populates ADAPTER_STATUS)
def _auto_register():
    """Import adapters to trigger their register_adapter() calls."""
    _adapter_modules = [
        ("dspy_adapter", "DSPyAdapter", "DSPY_AVAILABLE"),
        ("langgraph_adapter", "LangGraphAdapter", "LANGGRAPH_AVAILABLE"),
        ("mem0_adapter", "Mem0Adapter", "MEM0_AVAILABLE"),
        ("llm_reasoners_adapter", "LLMReasonersAdapter", "LLM_REASONERS_AVAILABLE"),
    ]
    for mod_name, _, _ in _adapter_modules:
        try:
            __import__(f"adapters.{mod_name}")
        except (ImportError, Exception):
            pass


_auto_register()
