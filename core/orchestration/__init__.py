#!/usr/bin/env python3
"""
Orchestration Layer - Unified Multi-Agent System Interface (Phase 9 Production Fix)

This module provides a unified interface to multiple orchestration frameworks:
- Temporal: Durable workflows with fault tolerance
- LangGraph: Graph-based multi-agent workflows
- Claude Flow: Native Claude-optimized agent orchestration
- CrewAI: Role-based team orchestration
- AutoGen: Conversational multi-agent systems

NO STUBS - EXPLICIT FAILURES ONLY:
- get_temporal_orchestrator() raises SDKNotAvailableError if not installed
- get_langgraph_orchestrator() raises SDKNotAvailableError if not installed
- get_claude_flow_orchestrator() raises SDKNotAvailableError if not installed
- get_crewai_manager() raises SDKNotAvailableError if not installed
- get_autogen_orchestrator() raises SDKNotAvailableError if not installed

Usage:
    from core.orchestration import UnifiedOrchestrator, SDKNotAvailableError

    orchestrator = UnifiedOrchestrator()

    # For direct SDK access (raises on unavailable):
    try:
        langgraph = get_langgraph_orchestrator()
    except SDKNotAvailableError as e:
        print(f"Install SDK: {e.install_cmd}")

    # Or use the best available option
    result = await orchestrator.run(task="...", prefer="claude_flow")
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional, Literal
from dataclasses import dataclass, field

import structlog

# Import exceptions from observability layer
from core.observability import (
    SDKNotAvailableError,
    SDKConfigurationError,
)

logger = structlog.get_logger(__name__)

# Import all orchestration modules with availability flags
from .temporal_workflows import (
    TEMPORAL_AVAILABLE,
    WorkflowInput,
    WorkflowResult,
)

# New SDK-integrated layers (V33.8)
try:
    from .agent_sdk_layer import (
        create_agent,
        run_agent_loop,
        Agent,
        AgentConfig as SDKAgentConfig,
        AgentResult,
        CLAUDE_AGENT_SDK_AVAILABLE,
        ANTHROPIC_AVAILABLE,
    )
    AGENT_SDK_LAYER_AVAILABLE = True
except ImportError:
    AGENT_SDK_LAYER_AVAILABLE = False
    logger.debug("agent_sdk_layer_not_available")

try:
    from .langgraph_layer import (
        load_pipeline,
        execute_pipeline,
        Pipeline,
        PipelineConfig,
        PipelineBuilder,
        LANGGRAPH_AVAILABLE as LANGGRAPH_PIPELINE_AVAILABLE,
    )
    LANGGRAPH_LAYER_AVAILABLE = True
except ImportError:
    LANGGRAPH_LAYER_AVAILABLE = False
    logger.debug("langgraph_layer_not_available")

try:
    from .workflow_runner import (
        get_workflow,
        execute_workflow,
        Workflow,
        WorkflowDefinition,
        WorkflowStep,
        WorkflowResult as WorkflowRunnerResult,
        WorkflowType,
        ExecutionMode,
        register_workflow,
        load_workflow_from_yaml,
    )
    WORKFLOW_RUNNER_AVAILABLE = True
except ImportError:
    WORKFLOW_RUNNER_AVAILABLE = False
    logger.debug("workflow_runner_not_available")

# V33.9: Embedding Layer (Voyage AI)
try:
    from .embedding_layer import (
        embed_texts,
        embed_for_search,
        create_embedding_layer,
        get_embedding_layer,
        EmbeddingLayer,
        EmbeddingConfig,
        EmbeddingResult,
        EmbeddingModel,
        InputType as EmbeddingInputType,
        VOYAGE_AVAILABLE,
    )
    EMBEDDING_LAYER_AVAILABLE = True
except ImportError:
    EMBEDDING_LAYER_AVAILABLE = False
    VOYAGE_AVAILABLE = False
    logger.debug("embedding_layer_not_available")

# V33.10: Orchestration Observability (Opik)
try:
    from .orchestration_observability import (
        create_observable_orchestrator,
        get_observable_orchestrator_sync,
        ObservableOrchestrator,
        ObservabilityConfig,
        TraceLevel,
        traced,
        trace_orchestration,
        OPIK_AVAILABLE,
        EVALUATOR_AVAILABLE,
        ORCHESTRATION_OBSERVABILITY_AVAILABLE,
    )
except ImportError:
    ORCHESTRATION_OBSERVABILITY_AVAILABLE = False
    OPIK_AVAILABLE = False
    EVALUATOR_AVAILABLE = False
    logger.debug("orchestration_observability_not_available")

# V33.11: Checkpoint Persistence (SQLite)
try:
    from .checkpoint_persistence import (
        # Factory functions
        create_checkpoint_store,
        get_checkpoint_store_sync,
        get_global_checkpoint_store,
        # Main class
        CheckpointStore,
        CheckpointConfig,
        # Data classes
        OrchestrationRun,
        RalphIterationState,
        ExecutionMetrics,
        # Enums
        RunStatus,
        CheckpointType,
        # Availability flags
        CHECKPOINT_PERSISTENCE_AVAILABLE,
        AIOSQLITE_AVAILABLE,
        LANGGRAPH_CHECKPOINT_AVAILABLE,
    )
except ImportError:
    CHECKPOINT_PERSISTENCE_AVAILABLE = False
    AIOSQLITE_AVAILABLE = False
    LANGGRAPH_CHECKPOINT_AVAILABLE = False
    logger.debug("checkpoint_persistence_not_available")

from .langgraph_agents import (
    LANGGRAPH_AVAILABLE,
    GraphConfig,
    GraphResult,
)

from .claude_flow import (
    CLAUDE_FLOW_AVAILABLE,
    AgentRole,
    FlowConfig,
    FlowResult,
    AgentConfig as ClaudeAgentConfig,
)

from .crew_manager import (
    CREWAI_AVAILABLE,
    AgentSpec,
    TaskSpec,
    CrewSpec,
    CrewResult,
)

from .autogen_agents import (
    AUTOGEN_AVAILABLE,
    LLMConfig,
    AgentConfig as AutoGenAgentConfig,
    GroupChatConfig,
    ConversationResult,
)

# Conditional imports for orchestrators
if TEMPORAL_AVAILABLE:
    from .temporal_workflows import TemporalOrchestrator, create_orchestrator as create_temporal

if LANGGRAPH_AVAILABLE:
    from .langgraph_agents import (
        LangGraphOrchestrator,
        MultiAgentGraph,
        create_orchestrator as create_langgraph,
    )

if CLAUDE_FLOW_AVAILABLE:
    from .claude_flow import (
        ClaudeFlow,
        ClaudeAgent,
        ClaudeFlowOrchestrator,
        create_orchestrator as create_claude_flow,
        create_default_flow,
    )

if CREWAI_AVAILABLE:
    from .crew_manager import (
        CrewManager,
        ManagedCrew,
        create_manager as create_crew_manager,
        create_agent_spec,
        create_task_spec,
    )

if AUTOGEN_AVAILABLE:
    from .autogen_agents import (
        AutoGenOrchestrator,
        AutoGenConversation,
        create_orchestrator as create_autogen,
        create_default_conversation,
    )


# Framework preference type
FrameworkType = Literal["temporal", "langgraph", "claude_flow", "crewai", "autogen", "auto"]


# =============================================================================
# EXPLICIT GETTER FUNCTIONS - No Stubs, Raise on Unavailable
# =============================================================================

def get_temporal_orchestrator(address: str = "localhost:7233") -> "TemporalOrchestrator":
    """
    Get Temporal orchestrator. Raises explicit error if unavailable.

    Args:
        address: Temporal server address

    Returns:
        Configured TemporalOrchestrator instance

    Raises:
        SDKNotAvailableError: If temporalio is not installed
        SDKConfigurationError: If required config is missing
    """
    if not TEMPORAL_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="temporalio",
            install_cmd="pip install temporalio>=1.5.0",
            docs_url="https://docs.temporal.io/dev-guide/python"
        )

    return create_temporal(address)


def get_langgraph_orchestrator() -> "LangGraphOrchestrator":
    """
    Get LangGraph orchestrator. Raises explicit error if unavailable.

    Returns:
        Configured LangGraphOrchestrator instance

    Raises:
        SDKNotAvailableError: If langgraph is not installed
    """
    if not LANGGRAPH_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="langgraph",
            install_cmd="pip install langgraph>=0.2.0 langchain-core>=0.2.0",
            docs_url="https://python.langchain.com/docs/langgraph"
        )

    return create_langgraph()


def get_claude_flow_orchestrator() -> "ClaudeFlowOrchestrator":
    """
    Get Claude Flow orchestrator. Raises explicit error if unavailable.

    Returns:
        Configured ClaudeFlowOrchestrator instance

    Raises:
        SDKNotAvailableError: If anthropic is not installed
        SDKConfigurationError: If ANTHROPIC_API_KEY not configured
    """
    if not CLAUDE_FLOW_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="anthropic",
            install_cmd="pip install anthropic>=0.30.0",
            docs_url="https://docs.anthropic.com/claude/reference"
        )

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SDKConfigurationError(
            sdk_name="claude_flow",
            missing_config=["ANTHROPIC_API_KEY"],
            example="""
ANTHROPIC_API_KEY=sk-ant-...
"""
        )

    return create_claude_flow()


def get_crewai_manager() -> "CrewManager":
    """
    Get CrewAI manager. Raises explicit error if unavailable.

    Returns:
        Configured CrewManager instance

    Raises:
        SDKNotAvailableError: If crewai is not installed
    """
    if not CREWAI_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="crewai",
            install_cmd="pip install crewai>=0.30.0",
            docs_url="https://docs.crewai.com/"
        )

    return create_crew_manager()


def get_autogen_orchestrator() -> "AutoGenOrchestrator":
    """
    Get AutoGen orchestrator. Raises explicit error if unavailable.

    Returns:
        Configured AutoGenOrchestrator instance

    Raises:
        SDKNotAvailableError: If pyautogen is not installed
    """
    if not AUTOGEN_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="pyautogen",
            install_cmd="pip install pyautogen>=0.2.0",
            docs_url="https://microsoft.github.io/autogen/"
        )

    return create_autogen()


@dataclass
class OrchestrationResult:
    """Unified result from any orchestration framework."""
    success: bool
    framework: str
    output: Any = None
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedOrchestrator:
    """
    Unified orchestrator providing access to all available frameworks.

    Automatically detects which frameworks are available and provides
    a consistent interface for running multi-agent workflows.

    V33 Integration:
    - Integrates with UnifiedMemory for cross-session memory
    - Integrates with UnifiedToolLayer for tool access
    - Provides context enrichment from memory before tasks
    - Records task results to memory for future sessions

    Usage:
        orchestrator = UnifiedOrchestrator()

        # Check availability
        print(f"Available: {orchestrator.available_frameworks}")

        # Run with preferred framework
        result = await orchestrator.run(
            task="Analyze data and generate report",
            prefer="claude_flow",
        )

        # Or run with specific framework
        result = await orchestrator.run_claude_flow(
            task="Create implementation plan",
        )

        # V33: With memory and tools integration
        from core.memory import create_memory
        from core.tools import create_tool_layer

        orchestrator = UnifiedOrchestrator(
            memory=create_memory(),
            tools=create_tool_layer(),
        )
        result = await orchestrator.run(
            task="...",
            use_memory=True,  # Enriches context from past executions
            record_result=True,  # Stores result for future sessions
        )
    """

    _temporal: Any = field(default=None)
    _langgraph: Any = field(default=None)
    _claude_flow: Any = field(default=None)
    _crew_manager: Any = field(default=None)
    _autogen: Any = field(default=None)

    # V33 Integration: Memory and Tools layers
    _memory: Any = field(default=None)
    _tools: Any = field(default=None)
    _initialized: bool = field(default=False)

    @property
    def temporal_available(self) -> bool:
        """Check if Temporal is available."""
        return TEMPORAL_AVAILABLE

    @property
    def langgraph_available(self) -> bool:
        """Check if LangGraph is available."""
        return LANGGRAPH_AVAILABLE

    @property
    def claude_flow_available(self) -> bool:
        """Check if Claude Flow is available."""
        return CLAUDE_FLOW_AVAILABLE

    @property
    def crewai_available(self) -> bool:
        """Check if CrewAI is available."""
        return CREWAI_AVAILABLE

    @property
    def autogen_available(self) -> bool:
        """Check if AutoGen is available."""
        return AUTOGEN_AVAILABLE

    @property
    def available_frameworks(self) -> list[str]:
        """List all available frameworks."""
        frameworks = []
        if self.temporal_available:
            frameworks.append("temporal")
        if self.langgraph_available:
            frameworks.append("langgraph")
        if self.claude_flow_available:
            frameworks.append("claude_flow")
        if self.crewai_available:
            frameworks.append("crewai")
        if self.autogen_available:
            frameworks.append("autogen")
        return frameworks

    # V33 Integration Properties

    @property
    def memory_available(self) -> bool:
        """Check if memory layer is attached."""
        return self._memory is not None

    @property
    def tools_available(self) -> bool:
        """Check if tools layer is attached."""
        return self._tools is not None

    async def initialize(self) -> None:
        """Initialize memory and tools layers if attached."""
        if self._initialized:
            return

        if self._memory and hasattr(self._memory, 'initialize'):
            await self._memory.initialize()
            logger.info("orchestrator_memory_initialized")

        if self._tools and hasattr(self._tools, 'initialize'):
            await self._tools.initialize()
            logger.info("orchestrator_tools_initialized")

        self._initialized = True

    async def _enrich_context_from_memory(
        self,
        task: str,
        context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Enrich context with relevant memories for the task."""
        enriched = dict(context) if context else {}

        if not self._memory:
            return enriched

        try:
            # Search for relevant past executions
            results = await self._memory.search(task, limit=5)
            if results:
                enriched["_memory_context"] = [
                    {
                        "content": r.content,
                        "score": r.score,
                        "metadata": r.metadata,
                    }
                    for r in results
                ]
                logger.info(
                    "context_enriched_from_memory",
                    memories_found=len(results),
                )
        except Exception as e:
            logger.warning("memory_enrichment_failed", error=str(e))

        return enriched

    async def _record_result_to_memory(
        self,
        task: str,
        result: OrchestrationResult,
    ) -> None:
        """Record execution result to memory for future sessions."""
        if not self._memory:
            return

        try:
            content = f"Task: {task}\nFramework: {result.framework}\n"
            if result.success:
                content += f"Result: {str(result.output)[:500]}"
            else:
                content += f"Error: {result.error}"

            await self._memory.store(
                content=content,
                metadata={
                    "type": "orchestration_result",
                    "framework": result.framework,
                    "success": result.success,
                    "task_summary": task[:100],
                },
                importance=0.7 if result.success else 0.5,
            )
            logger.info("result_recorded_to_memory", framework=result.framework)
        except Exception as e:
            logger.warning("memory_recording_failed", error=str(e))

    def attach_memory(self, memory: Any) -> "UnifiedOrchestrator":
        """Attach a memory layer (e.g., UnifiedMemory)."""
        self._memory = memory
        self._initialized = False
        return self

    def attach_tools(self, tools: Any) -> "UnifiedOrchestrator":
        """Attach a tools layer (e.g., UnifiedToolLayer)."""
        self._tools = tools
        self._initialized = False
        return self

    def get_tools_for_framework(self) -> list[dict[str, Any]]:
        """Get tool schemas formatted for the current framework."""
        if not self._tools:
            return []

        try:
            return self._tools.get_schemas_for_llm(format="anthropic")
        except Exception as e:
            logger.warning("get_tools_failed", error=str(e))
            return []

    def _select_framework(self, prefer: FrameworkType) -> str:
        """Select the best available framework based on preference."""
        if prefer != "auto" and prefer in self.available_frameworks:
            return prefer

        # Priority order for auto selection
        priority = ["claude_flow", "langgraph", "crewai", "autogen", "temporal"]

        for framework in priority:
            if framework in self.available_frameworks:
                return framework

        raise RuntimeError("No orchestration frameworks available")

    async def run(
        self,
        task: str,
        prefer: FrameworkType = "auto",
        context: Optional[dict[str, Any]] = None,
        use_memory: bool = False,
        record_result: bool = False,
        **kwargs,
    ) -> OrchestrationResult:
        """
        Run a task using the preferred or best available framework.

        Args:
            task: The task to execute
            prefer: Preferred framework ("auto" selects best available)
            context: Additional context for the task
            use_memory: V33 - Enrich context with relevant past memories
            record_result: V33 - Store execution result for future sessions
            **kwargs: Framework-specific options

        Returns:
            OrchestrationResult with unified output
        """
        # V33: Initialize layers if needed
        if (use_memory or record_result) and not self._initialized:
            await self.initialize()

        # V33: Enrich context from memory
        if use_memory and self._memory:
            context = await self._enrich_context_from_memory(task, context)

        framework = self._select_framework(prefer)

        logger.info(
            "orchestration_run",
            framework=framework,
            task=task[:50],
            use_memory=use_memory,
        )

        result: OrchestrationResult

        try:
            if framework == "claude_flow":
                result = await self._run_claude_flow(task, context, **kwargs)
            elif framework == "langgraph":
                result = await self._run_langgraph(task, context, **kwargs)
            elif framework == "crewai":
                result = await self._run_crewai(task, context, **kwargs)
            elif framework == "autogen":
                result = await self._run_autogen(task, context, **kwargs)
            elif framework == "temporal":
                result = await self._run_temporal(task, context, **kwargs)
            else:
                raise ValueError(f"Unknown framework: {framework}")

        except Exception as e:
            logger.error("orchestration_failed", framework=framework, error=str(e))
            result = OrchestrationResult(
                success=False,
                framework=framework,
                error=str(e),
            )

        # V33: Record result to memory
        if record_result and self._memory:
            await self._record_result_to_memory(task, result)

        return result

    async def _run_claude_flow(
        self,
        task: str,
        context: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> OrchestrationResult:
        """Run task using Claude Flow."""
        if not self._claude_flow:
            self._claude_flow = create_claude_flow()

        flow = self._claude_flow.create_flow(
            name=kwargs.get("flow_name", "default"),
        )

        result = await self._claude_flow.run(flow, task, context)

        return OrchestrationResult(
            success=result.success,
            framework="claude_flow",
            output=result.final_output,
            error=result.error,
            metadata={
                "total_steps": result.total_steps,
                "duration_seconds": result.duration_seconds,
            },
        )

    async def _run_langgraph(
        self,
        task: str,
        context: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> OrchestrationResult:
        """Run task using LangGraph."""
        if not self._langgraph:
            self._langgraph = create_langgraph()

        graph = self._langgraph.create_graph(
            name=kwargs.get("graph_name", "default"),
            max_steps=kwargs.get("max_steps", 10),
        )

        result = await self._langgraph.run(graph, task, context)

        return OrchestrationResult(
            success=result.success,
            framework="langgraph",
            output=result.result,
            error=result.error,
            metadata={
                "steps_taken": result.steps_taken,
                "final_agent": result.final_agent,
            },
        )

    async def _run_crewai(
        self,
        task: str,
        context: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> OrchestrationResult:
        """Run task using CrewAI."""
        if not self._crew_manager:
            self._crew_manager = create_crew_manager()

        crew = self._crew_manager.create_default_crew(
            name=kwargs.get("crew_name", "default"),
        )

        inputs = {"task": task}
        if context:
            inputs.update(context)

        result = await self._crew_manager.run(crew, inputs)

        return OrchestrationResult(
            success=result.success,
            framework="crewai",
            output=result.output,
            error=result.error,
            metadata={
                "task_outputs": result.task_outputs,
            },
        )

    async def _run_autogen(
        self,
        task: str,
        context: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> OrchestrationResult:
        """Run task using AutoGen."""
        if not self._autogen:
            self._autogen = create_autogen()

        conv = self._autogen.create_coding_conversation(
            name=kwargs.get("conversation_name", "default"),
        )

        result = await self._autogen.run(
            conv,
            sender="executor",
            receiver="coder",
            message=task,
        )

        return OrchestrationResult(
            success=result.success,
            framework="autogen",
            output=result.last_message,
            error=result.error,
            metadata={
                "chat_history_length": len(result.chat_history),
            },
        )

    async def _run_temporal(
        self,
        task: str,
        context: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> OrchestrationResult:
        """Run task using Temporal."""
        # Temporal requires async initialization
        if not self._temporal:
            address = kwargs.get("temporal_address", "localhost:7233")
            self._temporal = await create_temporal(address)

        result = await self._temporal.run_workflow(
            task=task,
            context=context,
            max_steps=kwargs.get("max_steps", 10),
        )

        return OrchestrationResult(
            success=result.success,
            framework="temporal",
            output=result.result,
            error=result.error,
            metadata={
                "steps_taken": result.steps_taken,
                "total_duration_ms": result.total_duration_ms,
            },
        )

    # Convenience methods for direct framework access

    async def run_claude_flow(
        self,
        task: str,
        context: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> OrchestrationResult:
        """Directly run using Claude Flow. Raises SDKNotAvailableError if unavailable."""
        if not self.claude_flow_available:
            raise SDKNotAvailableError(
                sdk_name="anthropic",
                install_cmd="pip install anthropic>=0.30.0",
                docs_url="https://docs.anthropic.com/claude/reference"
            )
        return await self._run_claude_flow(task, context, **kwargs)

    async def run_langgraph(
        self,
        task: str,
        context: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> OrchestrationResult:
        """Directly run using LangGraph. Raises SDKNotAvailableError if unavailable."""
        if not self.langgraph_available:
            raise SDKNotAvailableError(
                sdk_name="langgraph",
                install_cmd="pip install langgraph>=0.2.0 langchain-core>=0.2.0",
                docs_url="https://python.langchain.com/docs/langgraph"
            )
        return await self._run_langgraph(task, context, **kwargs)

    async def run_crewai(
        self,
        task: str,
        context: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> OrchestrationResult:
        """Directly run using CrewAI. Raises SDKNotAvailableError if unavailable."""
        if not self.crewai_available:
            raise SDKNotAvailableError(
                sdk_name="crewai",
                install_cmd="pip install crewai>=0.30.0",
                docs_url="https://docs.crewai.com/"
            )
        return await self._run_crewai(task, context, **kwargs)

    async def run_autogen(
        self,
        task: str,
        context: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> OrchestrationResult:
        """Directly run using AutoGen. Raises SDKNotAvailableError if unavailable."""
        if not self.autogen_available:
            raise SDKNotAvailableError(
                sdk_name="pyautogen",
                install_cmd="pip install pyautogen>=0.2.0",
                docs_url="https://microsoft.github.io/autogen/"
            )
        return await self._run_autogen(task, context, **kwargs)

    async def run_temporal(
        self,
        task: str,
        context: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> OrchestrationResult:
        """Directly run using Temporal. Raises SDKNotAvailableError if unavailable."""
        if not self.temporal_available:
            raise SDKNotAvailableError(
                sdk_name="temporalio",
                install_cmd="pip install temporalio>=1.5.0",
                docs_url="https://docs.temporal.io/dev-guide/python"
            )
        return await self._run_temporal(task, context, **kwargs)


def create_unified_orchestrator(
    memory: Any = None,
    tools: Any = None,
) -> UnifiedOrchestrator:
    """
    Factory function to create a unified orchestrator.

    V33 Integration:
        from core.memory import create_memory
        from core.tools import create_tool_layer

        orchestrator = create_unified_orchestrator(
            memory=create_memory(),
            tools=create_tool_layer(),
        )

    Args:
        memory: Optional UnifiedMemory instance for cross-session memory
        tools: Optional UnifiedToolLayer instance for tool access

    Returns:
        Configured UnifiedOrchestrator
    """
    orchestrator = UnifiedOrchestrator()

    if memory:
        orchestrator.attach_memory(memory)

    if tools:
        orchestrator.attach_tools(tools)

    return orchestrator


def get_available_frameworks() -> dict[str, bool]:
    """Get availability status of all frameworks."""
    return {
        "temporal": TEMPORAL_AVAILABLE,
        "langgraph": LANGGRAPH_AVAILABLE,
        "claude_flow": CLAUDE_FLOW_AVAILABLE,
        "crewai": CREWAI_AVAILABLE,
        "autogen": AUTOGEN_AVAILABLE,
    }


def get_v33_integration_status() -> dict[str, Any]:
    """
    Get V33 integration status for memory, tools, and frameworks.

    Returns:
        Dictionary with availability status for all V33 components
    """
    status: dict[str, Any] = {
        "frameworks": get_available_frameworks(),
        "memory_layer": False,
        "tools_layer": False,
    }

    # Check memory layer availability
    try:
        from core.memory import MEMORY_LAYER_AVAILABLE
        status["memory_layer"] = MEMORY_LAYER_AVAILABLE
    except ImportError:
        pass

    # Check tools layer availability
    try:
        from core.tools import PLATFORM_TOOLS_AVAILABLE, SDK_INTEGRATIONS_AVAILABLE
        status["tools_layer"] = PLATFORM_TOOLS_AVAILABLE or SDK_INTEGRATIONS_AVAILABLE
        status["platform_tools"] = PLATFORM_TOOLS_AVAILABLE
        status["sdk_integrations"] = SDK_INTEGRATIONS_AVAILABLE
    except ImportError:
        pass

    return status


# Export all public symbols
__all__ = [
    # Exceptions (re-exported from observability)
    "SDKNotAvailableError",
    "SDKConfigurationError",

    # Availability flags
    "TEMPORAL_AVAILABLE",
    "LANGGRAPH_AVAILABLE",
    "CLAUDE_FLOW_AVAILABLE",
    "CREWAI_AVAILABLE",
    "AUTOGEN_AVAILABLE",

    # New SDK-integrated layers (V33.8+)
    "AGENT_SDK_LAYER_AVAILABLE",
    "LANGGRAPH_LAYER_AVAILABLE",
    "WORKFLOW_RUNNER_AVAILABLE",
    "EMBEDDING_LAYER_AVAILABLE",
    "VOYAGE_AVAILABLE",

    # Agent SDK Layer
    "create_agent",
    "run_agent_loop",
    "Agent",
    "SDKAgentConfig",
    "AgentResult",

    # LangGraph Pipeline Layer
    "load_pipeline",
    "execute_pipeline",
    "Pipeline",
    "PipelineConfig",
    "PipelineBuilder",

    # Workflow Runner
    "get_workflow",
    "execute_workflow",
    "Workflow",
    "WorkflowDefinition",
    "WorkflowStep",
    "WorkflowRunnerResult",
    "WorkflowType",
    "ExecutionMode",
    "register_workflow",
    "load_workflow_from_yaml",

    # Embedding Layer (V33.9)
    "embed_texts",
    "embed_for_search",
    "create_embedding_layer",
    "get_embedding_layer",
    "EmbeddingLayer",
    "EmbeddingConfig",
    "EmbeddingResult",
    "EmbeddingModel",
    "EmbeddingInputType",

    # Orchestration Observability (V33.10)
    "ORCHESTRATION_OBSERVABILITY_AVAILABLE",
    "OPIK_AVAILABLE",
    "EVALUATOR_AVAILABLE",
    "create_observable_orchestrator",
    "get_observable_orchestrator_sync",
    "ObservableOrchestrator",
    "ObservabilityConfig",
    "TraceLevel",
    "traced",
    "trace_orchestration",

    # Checkpoint Persistence (V33.11)
    "CHECKPOINT_PERSISTENCE_AVAILABLE",
    "AIOSQLITE_AVAILABLE",
    "LANGGRAPH_CHECKPOINT_AVAILABLE",
    "create_checkpoint_store",
    "get_checkpoint_store_sync",
    "get_global_checkpoint_store",
    "CheckpointStore",
    "CheckpointConfig",
    "OrchestrationRun",
    "RalphIterationState",
    "ExecutionMetrics",
    "RunStatus",
    "CheckpointType",

    # Getter functions (raise on unavailable)
    "get_temporal_orchestrator",
    "get_langgraph_orchestrator",
    "get_claude_flow_orchestrator",
    "get_crewai_manager",
    "get_autogen_orchestrator",

    # Unified interface
    "UnifiedOrchestrator",
    "OrchestrationResult",
    "create_unified_orchestrator",
    "get_available_frameworks",
    "get_v33_integration_status",
    "FrameworkType",

    # Result types
    "WorkflowResult",
    "GraphResult",
    "FlowResult",
    "CrewResult",
    "ConversationResult",

    # Config types
    "WorkflowInput",
    "GraphConfig",
    "FlowConfig",
    "CrewSpec",
    "AgentSpec",
    "TaskSpec",
    "LLMConfig",
    "GroupChatConfig",
    "ClaudeAgentConfig",
    "AutoGenAgentConfig",
    "AgentRole",
]


if __name__ == "__main__":
    async def main():
        """Test the unified orchestrator."""
        print("Orchestration Layer Status")
        print("=" * 40)

        status = get_available_frameworks()
        for framework, available in status.items():
            symbol = "✓" if available else "✗"
            print(f"  {symbol} {framework}")

        print()

        orchestrator = create_unified_orchestrator()
        print(f"Available: {orchestrator.available_frameworks}")

        if orchestrator.available_frameworks:
            print(f"\nRunning test task with {orchestrator.available_frameworks[0]}...")
            result = await orchestrator.run(
                task="Create a simple hello world function",
                prefer="auto",
            )
            print(f"Success: {result.success}")
            print(f"Framework: {result.framework}")
            if result.output:
                print(f"Output: {str(result.output)[:200]}...")

    asyncio.run(main())
