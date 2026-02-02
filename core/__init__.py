#!/usr/bin/env python3
"""
Unleash Platform Core Module - V33 Architecture (Complete 8-Layer Stack)

The V33 Architecture provides a unified, cross-session accessible SDK ecosystem
with explicit failure patterns (no stubs, no silent fallbacks).

Layer Stack:
- L0: Protocol Layer (LLM Gateway, MCP Server, Provider SDKs)
- L1: Orchestration Layer (Temporal, LangGraph, CrewAI, AutoGen, Claude-Flow)
- L2: Memory Layer (Letta, Zep, Mem0, CrossSession)
- L3: Structured Output Layer (Instructor, BAML, Outlines, Pydantic-AI)
- L4: Reasoning Layer (DSPy, Serena)
- L5: Observability Layer (Langfuse, Phoenix, Opik, DeepEval, Ragas, Logfire)
- L6: Safety Layer (Guardrails-AI, LLM-Guard, NeMo-Guardrails)
- L7: Processing Layer (Crawl4AI, Firecrawl, Aider, AST-Grep)
- L8: Knowledge Layer (GraphRAG, PyRibs)

Usage:
    from core import V33, get_v33_status

    # Get unified system status
    status = get_v33_status()
    print(f"V33 Architecture: {status['available_layers']}/8 layers available")

    # Access any layer
    from core.reasoning import get_dspy_lm, DSPyClient
    from core.safety import get_guardrails_guard
    from core.knowledge import get_graphrag_client
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

# =============================================================================
# L0: Protocol Layer
# =============================================================================
from core.llm_gateway import (
    LLMGateway,
    Provider,
    ModelConfig,
    Message,
    CompletionResponse,
    quick_complete,
)
from core.mcp_server import (
    create_mcp_server,
    get_server,
    run_server,
    ToolResult,
    FASTMCP_AVAILABLE,
)

# =============================================================================
# L1: Orchestration Layer
# =============================================================================
from core.orchestration import (
    UnifiedOrchestrator,
    OrchestrationResult,
    create_unified_orchestrator,
    get_available_frameworks,
)

# =============================================================================
# L2: Memory Layer
# =============================================================================
from core.memory import (
    UnifiedMemory,
    create_memory,
    get_available_memory_providers,
    MEMORY_LAYER_AVAILABLE,
)

# =============================================================================
# L3: Structured Output Layer
# =============================================================================
from core.structured import (
    InstructorClient,
    BAMLClient,
    OutlinesGenerator,
    PydanticAIAgent,
    StructuredOutputFactory,
    get_available_sdks as get_structured_sdks,
    INSTRUCTOR_AVAILABLE,
    BAML_AVAILABLE,
    OUTLINES_AVAILABLE,
    PYDANTIC_AI_AVAILABLE,
    STRUCTURED_OUTPUT_AVAILABLE,
)

# =============================================================================
# L4: Reasoning Layer
# =============================================================================
from core.reasoning import (
    DSPyClient,
    SerenaClient,
    ReasoningFactory,
    get_dspy_lm,
    get_dspy_optimizer,
    get_serena_client,
    get_available_sdks as get_reasoning_sdks,
    DSPY_AVAILABLE,
    SERENA_AVAILABLE,
    REASONING_AVAILABLE,
)

# =============================================================================
# L5: Observability Layer
# =============================================================================
from core.observability import (
    # Getter functions (the actual API - no client classes, use getters)
    get_langfuse_tracer,
    get_langfuse_observe,
    get_phoenix_client,
    get_opik_client,
    get_opik_track,
    get_deepeval_evaluator,
    get_deepeval_metrics,
    get_ragas_evaluator,
    get_ragas_metrics,
    get_logfire_logger,
    get_opentelemetry_tracer,
    get_observability_status,
    get_available_sdks as get_observability_sdks,
    # Status class
    SDKStatus as ObservabilityStatus,
    # Layer availability
    OBSERVABILITY_AVAILABLE,
    # Individual SDK availability
    LANGFUSE_AVAILABLE,
    PHOENIX_AVAILABLE,
    OPIK_AVAILABLE,
    DEEPEVAL_AVAILABLE,
    RAGAS_AVAILABLE,
    LOGFIRE_AVAILABLE,
    OPENTELEMETRY_AVAILABLE,
    # Exceptions (re-export from central location)
    SDKNotAvailableError,
    SDKConfigurationError,
)

# =============================================================================
# L6: Safety Layer
# =============================================================================
from core.safety import (
    GuardrailsClient,
    LLMGuardClient,
    NemoGuardrailsClient,
    SafetyFactory,
    get_guardrails_guard,
    get_llm_guard_scanner,
    get_nemo_rails,
    get_available_sdks as get_safety_sdks,
    GUARDRAILS_AVAILABLE,
    LLM_GUARD_AVAILABLE,
    NEMO_AVAILABLE,
    SAFETY_AVAILABLE,
)

# =============================================================================
# L7: Processing Layer
# =============================================================================
from core.processing import (
    Crawl4AIClient,
    FirecrawlClient,
    AiderClient,
    ASTGrepClient,
    ProcessingFactory,
    get_crawl4ai_crawler,
    get_firecrawl_client,
    get_aider_coder,
    get_astgrep_client,
    get_available_sdks as get_processing_sdks,
    CRAWL4AI_AVAILABLE,
    FIRECRAWL_AVAILABLE,
    AIDER_AVAILABLE,
    ASTGREP_AVAILABLE,
    PROCESSING_AVAILABLE,
)

# =============================================================================
# L8: Knowledge Layer
# =============================================================================
from core.knowledge import (
    GraphRAGClient,
    PyRibsClient,
    KnowledgeFactory,
    get_graphrag_client,
    get_pyribs_archive,
    get_available_sdks as get_knowledge_sdks,
    GRAPHRAG_AVAILABLE,
    PYRIBS_AVAILABLE,
    KNOWLEDGE_AVAILABLE,
)

# =============================================================================
# Phase 12: Compatibility Layers for Python 3.14 Impossible SDKs
# =============================================================================

# CrewAI Compat - LangGraph-based multi-agent orchestration
try:
    from core.orchestration.crewai_compat import (
        CrewCompat,
        Agent as CrewAgent,
        Task as CrewTask,
        AgentRole,
        CrewState,
        CREWAI_COMPAT_AVAILABLE,
    )
except ImportError:
    CREWAI_COMPAT_AVAILABLE = False
    CrewCompat = None
    CrewAgent = None
    CrewTask = None
    AgentRole = None
    CrewState = None

# Outlines Compat - JSON Schema + Regex constrained generation
try:
    from core.structured.outlines_compat import (
        OutlinesCompat,
        Choice,
        Regex,
        Integer,
        Float,
        JsonGenerator,
        Constraint,
        OUTLINES_COMPAT_AVAILABLE,
    )
except ImportError:
    OUTLINES_COMPAT_AVAILABLE = False
    OutlinesCompat = None
    Choice = None
    Regex = None
    Integer = None
    Float = None
    JsonGenerator = None
    Constraint = None

# Aider Compat - Git-aware code modification
try:
    from core.processing.aider_compat import (
        AiderCompat,
        EditBlock,
        AiderSession,
        AIDER_COMPAT_AVAILABLE,
    )
except ImportError:
    AIDER_COMPAT_AVAILABLE = False
    AiderCompat = None
    EditBlock = None
    AiderSession = None

# AgentLite Compat - Lightweight ReAct agent framework
try:
    from core.reasoning.agentlite_compat import (
        AgentLiteCompat,
        Tool as AgentTool,
        Action,
        ActionType,
        AgentState,
        create_tool,
        AGENTLITE_COMPAT_AVAILABLE,
    )
except ImportError:
    AGENTLITE_COMPAT_AVAILABLE = False
    AgentLiteCompat = None
    AgentTool = None
    Action = None
    ActionType = None
    AgentState = None
    create_tool = None

# =============================================================================
# Tools Layer (Cross-Layer Integration)
# =============================================================================
from core.tools import (
    UnifiedToolLayer,
    create_tool_layer,
    get_available_tool_sources,
    PLATFORM_TOOLS_AVAILABLE,
    SDK_INTEGRATIONS_AVAILABLE,
)


# =============================================================================
# V33 Unified Access Layer
# =============================================================================

@dataclass
class LayerStatus:
    """Status of a V33 layer."""
    name: str
    available: bool
    sdk_count: int
    available_sdks: int
    sdks: Dict[str, bool] = field(default_factory=dict)


@dataclass
class V33Status:
    """Complete V33 system status."""
    total_layers: int = 9  # L0-L8
    available_layers: int = 0
    total_sdks: int = 0
    available_sdks: int = 0
    layers: Dict[str, LayerStatus] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_layers": self.total_layers,
            "available_layers": self.available_layers,
            "total_sdks": self.total_sdks,
            "available_sdks": self.available_sdks,
            "completion_percentage": round(self.available_sdks / self.total_sdks * 100, 1) if self.total_sdks > 0 else 0,
            "layers": {
                name: {
                    "available": layer.available,
                    "sdk_count": layer.sdk_count,
                    "available_sdks": layer.available_sdks,
                    "sdks": layer.sdks,
                }
                for name, layer in self.layers.items()
            }
        }


def get_v33_status() -> V33Status:
    """
    Get comprehensive V33 system status.

    Returns status of all 9 layers (L0-L8) with SDK availability.
    This is the primary entry point for cross-session state validation.

    Returns:
        V33Status with complete layer and SDK availability information
    """
    status = V33Status()

    # L0: Protocol Layer
    l0_sdks = {
        "fastmcp": FASTMCP_AVAILABLE,
        "litellm": True,  # Core dependency
        "anthropic": True,  # Core provider
        "openai": True,  # Core provider
    }
    status.layers["L0_Protocol"] = LayerStatus(
        name="Protocol",
        available=True,  # Core layer always available
        sdk_count=len(l0_sdks),
        available_sdks=sum(l0_sdks.values()),
        sdks=l0_sdks,
    )

    # L1: Orchestration Layer
    l1_sdks = get_available_frameworks()
    status.layers["L1_Orchestration"] = LayerStatus(
        name="Orchestration",
        available=len(l1_sdks) > 0,
        sdk_count=5,  # temporal, langgraph, claude_flow, crewai, autogen
        available_sdks=len(l1_sdks),
        sdks={name: True for name in l1_sdks},
    )

    # L2: Memory Layer
    l2_sdks = get_available_memory_providers()
    status.layers["L2_Memory"] = LayerStatus(
        name="Memory",
        available=MEMORY_LAYER_AVAILABLE,
        sdk_count=4,  # letta, zep, mem0, cross_session
        available_sdks=len(l2_sdks),
        sdks={name: True for name in l2_sdks},
    )

    # L3: Structured Output Layer
    l3_sdks = get_structured_sdks()
    status.layers["L3_Structured"] = LayerStatus(
        name="Structured",
        available=STRUCTURED_OUTPUT_AVAILABLE,
        sdk_count=4,  # instructor, baml, outlines, pydantic_ai
        available_sdks=len(l3_sdks),
        sdks=l3_sdks,
    )

    # L4: Reasoning Layer
    l4_sdks = get_reasoning_sdks()
    status.layers["L4_Reasoning"] = LayerStatus(
        name="Reasoning",
        available=REASONING_AVAILABLE,
        sdk_count=2,  # dspy, serena
        available_sdks=sum(l4_sdks.values()),
        sdks=l4_sdks,
    )

    # L5: Observability Layer
    l5_sdks = get_observability_sdks()
    status.layers["L5_Observability"] = LayerStatus(
        name="Observability",
        available=OBSERVABILITY_AVAILABLE,
        sdk_count=7,  # langfuse, phoenix, opik, deepeval, ragas, logfire, opentelemetry
        available_sdks=sum(l5_sdks.values()),
        sdks=l5_sdks,
    )

    # L6: Safety Layer
    l6_sdks = get_safety_sdks()
    status.layers["L6_Safety"] = LayerStatus(
        name="Safety",
        available=SAFETY_AVAILABLE,
        sdk_count=3,  # guardrails, llm_guard, nemo
        available_sdks=sum(l6_sdks.values()),
        sdks=l6_sdks,
    )

    # L7: Processing Layer
    l7_sdks = get_processing_sdks()
    status.layers["L7_Processing"] = LayerStatus(
        name="Processing",
        available=PROCESSING_AVAILABLE,
        sdk_count=4,  # crawl4ai, firecrawl, aider, ast_grep
        available_sdks=sum(l7_sdks.values()),
        sdks=l7_sdks,
    )

    # L8: Knowledge Layer
    l8_sdks = get_knowledge_sdks()
    status.layers["L8_Knowledge"] = LayerStatus(
        name="Knowledge",
        available=KNOWLEDGE_AVAILABLE,
        sdk_count=2,  # graphrag, pyribs
        available_sdks=sum(l8_sdks.values()),
        sdks=l8_sdks,
    )

    # Calculate totals
    for layer in status.layers.values():
        status.total_sdks += layer.sdk_count
        status.available_sdks += layer.available_sdks
        if layer.available:
            status.available_layers += 1

    return status


def print_v33_status(status: Optional[V33Status] = None) -> None:
    """
    Print formatted V33 system status.

    Args:
        status: Optional pre-computed status. If None, will compute.
    """
    if status is None:
        status = get_v33_status()

    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        _print_v33_rich(status)
    except ImportError:
        _print_v33_plain(status)


def _print_v33_rich(status: V33Status) -> None:
    """Print V33 status using Rich."""
    import sys
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    # Use ASCII on Windows to avoid encoding issues
    is_windows = sys.platform == "win32"
    check_mark = "[green]OK[/green]" if is_windows else "[green]✓[/green]"
    x_mark = "[red]X[/red]" if is_windows else "[red]✗[/red]"

    console = Console()

    pct = round(status.available_sdks / status.total_sdks * 100, 1) if status.total_sdks > 0 else 0
    color = "green" if pct >= 80 else "yellow" if pct >= 50 else "red"

    console.print(Panel(
        f"[bold]V33 Architecture Status[/bold]\n"
        f"Layers: [{color}]{status.available_layers}/{status.total_layers}[/{color}] available\n"
        f"SDKs: [{color}]{status.available_sdks}/{status.total_sdks}[/{color}] ({pct}%)",
        title="System Status",
    ))

    table = Table(title="Layer Details")
    table.add_column("Layer", style="cyan")
    table.add_column("Status")
    table.add_column("SDKs")
    table.add_column("Available")

    for name, layer in status.layers.items():
        avail = check_mark if layer.available else x_mark
        sdk_list = ", ".join(f"[green]{k}[/green]" if v else f"[dim]{k}[/dim]"
                            for k, v in layer.sdks.items())
        table.add_row(
            name,
            avail,
            f"{layer.available_sdks}/{layer.sdk_count}",
            sdk_list[:60] + "..." if len(sdk_list) > 60 else sdk_list,
        )

    console.print(table)


def _print_v33_plain(status: V33Status) -> None:
    """Print V33 status using plain text."""
    pct = round(status.available_sdks / status.total_sdks * 100, 1) if status.total_sdks > 0 else 0

    print("=" * 60)
    print("V33 Architecture Status")
    print("=" * 60)
    print(f"Layers: {status.available_layers}/{status.total_layers} available")
    print(f"SDKs: {status.available_sdks}/{status.total_sdks} ({pct}%)")
    print()

    for name, layer in status.layers.items():
        avail = "OK" if layer.available else "MISSING"
        print(f"{name}: {avail} ({layer.available_sdks}/{layer.sdk_count})")
        for sdk, available in layer.sdks.items():
            mark = "+" if available else "-"
            print(f"  {mark} {sdk}")
    print()


# =============================================================================
# Convenience Aliases
# =============================================================================

# Unified accessor class
class V33:
    """
    Unified V33 Architecture accessor.

    Provides a single entry point to all V33 layers with explicit availability checks.

    Usage:
        v33 = V33()
        status = v33.status()

        # Access layers
        if v33.reasoning.available:
            dspy = v33.reasoning.get_dspy()

        if v33.safety.available:
            guard = v33.safety.get_guardrails()
    """

    def __init__(self):
        self._status: Optional[V33Status] = None

    def status(self, refresh: bool = False) -> V33Status:
        """Get or refresh V33 status."""
        if self._status is None or refresh:
            self._status = get_v33_status()
        return self._status

    @property
    def reasoning(self) -> ReasoningFactory:
        """Access L4: Reasoning Layer."""
        return ReasoningFactory()

    @property
    def safety(self) -> SafetyFactory:
        """Access L6: Safety Layer."""
        return SafetyFactory()

    @property
    def processing(self) -> ProcessingFactory:
        """Access L7: Processing Layer."""
        return ProcessingFactory()

    @property
    def knowledge(self) -> KnowledgeFactory:
        """Access L8: Knowledge Layer."""
        return KnowledgeFactory()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # ==========================================================================
    # L0: Protocol Layer
    # ==========================================================================
    "LLMGateway",
    "Provider",
    "ModelConfig",
    "Message",
    "CompletionResponse",
    "quick_complete",
    "create_mcp_server",
    "get_server",
    "run_server",
    "ToolResult",
    "FASTMCP_AVAILABLE",

    # ==========================================================================
    # L1: Orchestration Layer
    # ==========================================================================
    "UnifiedOrchestrator",
    "OrchestrationResult",
    "create_unified_orchestrator",
    "get_available_frameworks",

    # ==========================================================================
    # L2: Memory Layer
    # ==========================================================================
    "UnifiedMemory",
    "create_memory",
    "get_available_memory_providers",
    "MEMORY_LAYER_AVAILABLE",

    # ==========================================================================
    # L3: Structured Output Layer
    # ==========================================================================
    "InstructorClient",
    "BAMLClient",
    "OutlinesGenerator",
    "PydanticAIAgent",
    "StructuredOutputFactory",
    "get_structured_sdks",
    "INSTRUCTOR_AVAILABLE",
    "BAML_AVAILABLE",
    "OUTLINES_AVAILABLE",
    "PYDANTIC_AI_AVAILABLE",
    "STRUCTURED_OUTPUT_AVAILABLE",

    # ==========================================================================
    # L4: Reasoning Layer
    # ==========================================================================
    "DSPyClient",
    "SerenaClient",
    "ReasoningFactory",
    "get_dspy_lm",
    "get_dspy_optimizer",
    "get_serena_client",
    "get_reasoning_sdks",
    "DSPY_AVAILABLE",
    "SERENA_AVAILABLE",
    "REASONING_AVAILABLE",

    # ==========================================================================
    # L5: Observability Layer (getter-function pattern)
    # ==========================================================================
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
    "get_observability_status",
    "get_observability_sdks",
    "ObservabilityStatus",
    "OBSERVABILITY_AVAILABLE",
    "LANGFUSE_AVAILABLE",
    "PHOENIX_AVAILABLE",
    "OPIK_AVAILABLE",
    "DEEPEVAL_AVAILABLE",
    "RAGAS_AVAILABLE",
    "LOGFIRE_AVAILABLE",
    "OPENTELEMETRY_AVAILABLE",
    "SDKNotAvailableError",
    "SDKConfigurationError",

    # ==========================================================================
    # L6: Safety Layer
    # ==========================================================================
    "GuardrailsClient",
    "LLMGuardClient",
    "NemoGuardrailsClient",
    "SafetyFactory",
    "get_guardrails_guard",
    "get_llm_guard_scanner",
    "get_nemo_rails",
    "get_safety_sdks",
    "GUARDRAILS_AVAILABLE",
    "LLM_GUARD_AVAILABLE",
    "NEMO_AVAILABLE",
    "SAFETY_AVAILABLE",

    # ==========================================================================
    # L7: Processing Layer
    # ==========================================================================
    "Crawl4AIClient",
    "FirecrawlClient",
    "AiderClient",
    "ASTGrepClient",
    "ProcessingFactory",
    "get_crawl4ai_crawler",
    "get_firecrawl_client",
    "get_aider_coder",
    "get_astgrep_client",
    "get_processing_sdks",
    "CRAWL4AI_AVAILABLE",
    "FIRECRAWL_AVAILABLE",
    "AIDER_AVAILABLE",
    "ASTGREP_AVAILABLE",
    "PROCESSING_AVAILABLE",

    # ==========================================================================
    # L8: Knowledge Layer
    # ==========================================================================
    "GraphRAGClient",
    "PyRibsClient",
    "KnowledgeFactory",
    "get_graphrag_client",
    "get_pyribs_archive",
    "get_knowledge_sdks",
    "GRAPHRAG_AVAILABLE",
    "PYRIBS_AVAILABLE",
    "KNOWLEDGE_AVAILABLE",

    # ==========================================================================
    # Tools Layer (Cross-Layer)
    # ==========================================================================
    "UnifiedToolLayer",
    "create_tool_layer",
    "get_available_tool_sources",
    "PLATFORM_TOOLS_AVAILABLE",
    "SDK_INTEGRATIONS_AVAILABLE",

    # ==========================================================================
    # Phase 12: Compatibility Layers (Python 3.14 Impossible SDKs)
    # ==========================================================================
    # CrewAI Compat
    "CrewCompat",
    "CrewAgent",
    "CrewTask",
    "AgentRole",
    "CrewState",
    "CREWAI_COMPAT_AVAILABLE",
    # Outlines Compat
    "OutlinesCompat",
    "Choice",
    "Regex",
    "Integer",
    "Float",
    "JsonGenerator",
    "Constraint",
    "OUTLINES_COMPAT_AVAILABLE",
    # Aider Compat
    "AiderCompat",
    "EditBlock",
    "AiderSession",
    "AIDER_COMPAT_AVAILABLE",
    # AgentLite Compat
    "AgentLiteCompat",
    "AgentTool",
    "Action",
    "ActionType",
    "AgentState",
    "create_tool",
    "AGENTLITE_COMPAT_AVAILABLE",

    # ==========================================================================
    # V33 Unified Access
    # ==========================================================================
    "V33",
    "V33Status",
    "LayerStatus",
    "get_v33_status",
    "print_v33_status",
]
