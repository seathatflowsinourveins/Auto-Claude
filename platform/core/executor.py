"""
UAP Agent Executor - Unified orchestration combining all core modules.

Implements the ReAct (Reason + Act) pattern with extended thinking support:
    Thought → Action → Observation → Update → (loop)

Architecture based on 2026 agentic patterns:
    Intent Router → Agent Orchestrator → (Planner, Executor, Reflector, Memory) → Output

Components integrated:
    - AgentHarness: Context window management and shift handoffs
    - MemorySystem: Three-tier memory (Core, Archival, Temporal)
    - CooperationManager: Multi-agent coordination
    - MCPServerManager: Dynamic tool orchestration
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass  # Used by dataclass-style models
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
)

from pydantic import BaseModel, Field

# Internal imports (relative)
from .harness import AgentHarness, ContextPillar, ContextPriority, ShiftHandoff
from .memory import MemorySystem
from .cooperation import CooperationManager
from .mcp_manager import MCPServerManager, ToolSchema

# Research engine integration (optional, loaded lazily)
try:
    from .research_engine import ResearchEngine, get_engine as get_research_engine
    RESEARCH_ENGINE_AVAILABLE = True
except ImportError:
    RESEARCH_ENGINE_AVAILABLE = False
    ResearchEngine = None
    get_research_engine = None

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class ExecutionPhase(str, Enum):
    """Phases in the ReAct execution loop."""

    THINK = "think"          # Reasoning phase (extended thinking)
    PLAN = "plan"            # Planning phase (decompose task)
    ACT = "act"              # Action phase (tool execution)
    OBSERVE = "observe"      # Observation phase (process results)
    REFLECT = "reflect"      # Reflection phase (self-critique)
    UPDATE = "update"        # Update phase (memory/state update)
    COMPLETE = "complete"    # Task completion


class TaskStatus(str, Enum):
    """Status of an executor task."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ThinkingMode(str, Enum):
    """Extended thinking configuration modes."""

    DISABLED = "disabled"          # No extended thinking
    BUDGET_LIMITED = "budget"      # Token budget for thinking
    INTERLEAVED = "interleaved"    # Think between actions
    ULTRATHINK = "ultrathink"      # Maximum 128K tokens


# =============================================================================
# Data Models
# =============================================================================

class ThinkingConfig(BaseModel):
    """Configuration for extended thinking."""

    mode: ThinkingMode = ThinkingMode.INTERLEAVED
    budget_tokens: int = Field(default=16000, ge=1000, le=128000)
    show_thinking: bool = False  # Hidden by default (Anthropic pattern)
    temperature: float = Field(default=1.0, ge=0.0, le=1.0)


class ActionRequest(BaseModel):
    """Request for a tool action."""

    tool_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float = 30.0
    retry_count: int = 0
    max_retries: int = 3


class ActionResult(BaseModel):
    """Result from a tool action."""

    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    retry_count: int = 0


class ThoughtRecord(BaseModel):
    """Record of a reasoning step."""

    phase: ExecutionPhase
    content: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    timestamp: float = Field(default_factory=time.time)
    token_count: int = 0


class ReflectionRecord(BaseModel):
    """Record of a self-reflection."""

    observation: str
    analysis: str
    improvements: List[str] = Field(default_factory=list)
    should_retry: bool = False
    confidence_delta: float = 0.0


class ExecutorState(BaseModel):
    """Complete state of the executor for persistence."""

    task_id: str
    task_description: str
    status: TaskStatus
    current_phase: ExecutionPhase
    thoughts: List[ThoughtRecord] = Field(default_factory=list)
    actions: List[ActionResult] = Field(default_factory=list)
    reflections: List[ReflectionRecord] = Field(default_factory=list)
    iteration_count: int = 0
    max_iterations: int = 50
    started_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None
    final_result: Any = None
    error_message: Optional[str] = None


# =============================================================================
# Protocols (Interface Definitions)
# =============================================================================

class ToolExecutor(Protocol):
    """Protocol for tool execution backends."""

    async def execute(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        timeout: float,
    ) -> ActionResult:
        """Execute a tool and return the result."""
        ...


class ReasoningEngine(Protocol):
    """Protocol for the reasoning/thinking component."""

    async def think(
        self,
        context: str,
        question: str,
        config: ThinkingConfig,
    ) -> ThoughtRecord:
        """Generate a thought/reasoning step."""
        ...

    async def plan(
        self,
        task: str,
        context: str,
        available_tools: List[ToolSchema],
    ) -> List[str]:
        """Decompose a task into steps."""
        ...

    async def reflect(
        self,
        action: ActionResult,
        expected: str,
        context: str,
    ) -> ReflectionRecord:
        """Reflect on an action's outcome."""
        ...


# =============================================================================
# Default Implementations
# =============================================================================

class MCPToolExecutor:
    """Tool executor using MCP servers."""

    def __init__(self, mcp_manager: MCPServerManager):
        self._mcp = mcp_manager
        self._tool_cache: Dict[str, str] = {}  # tool_name -> server_name

    def _get_server_for_tool(self, tool_name: str) -> Optional[str]:
        """Find which server provides a tool."""
        if tool_name in self._tool_cache:
            return self._tool_cache[tool_name]

        server = self._mcp.get_tool_server(tool_name)
        if server:
            self._tool_cache[tool_name] = server
        return server

    async def execute(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        timeout: float,
    ) -> ActionResult:
        """Execute a tool via MCP."""
        start_time = time.time()

        server = self._get_server_for_tool(tool_name)
        if not server:
            return ActionResult(
                tool_name=tool_name,
                success=False,
                error=f"No server found for tool: {tool_name}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        try:
            # In a real implementation, this would call the MCP server
            # For now, we return a placeholder result
            result = {
                "server": server,
                "tool": tool_name,
                "parameters": parameters,
                "status": "executed",
            }

            return ActionResult(
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        except asyncio.TimeoutError:
            return ActionResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool execution timed out after {timeout}s",
                execution_time_ms=timeout * 1000,
            )
        except Exception as e:
            return ActionResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )


class SimpleReasoningEngine:
    """Simple reasoning engine (placeholder for LLM integration)."""

    def __init__(self, thinking_config: ThinkingConfig):
        self._config = thinking_config

    async def think(
        self,
        context: str,
        question: str,
        config: ThinkingConfig,
    ) -> ThoughtRecord:
        """Generate a reasoning step."""
        # In production, this would call Claude with extended thinking
        # Placeholder references context and config for LLM call
        _ = (context, config)  # Will be used when LLM integration is added
        thought_content = f"Analyzing: {question[:100]}..."

        return ThoughtRecord(
            phase=ExecutionPhase.THINK,
            content=thought_content,
            confidence=0.8,
            token_count=len(thought_content.split()) * 2,  # Rough estimate
        )

    async def plan(
        self,
        task: str,
        context: str,
        available_tools: List[ToolSchema],
    ) -> List[str]:
        """Decompose task into steps."""
        # Simple placeholder - would use LLM in production
        _ = context  # Will be used when LLM integration is added
        tool_names = [t.name for t in available_tools[:5]]
        return [
            f"Analyze task requirements: {task[:50]}",
            f"Available tools: {', '.join(tool_names)}",
            "Execute primary action",
            "Verify results",
            "Report completion",
        ]

    async def reflect(
        self,
        action: ActionResult,
        expected: str,
        context: str,
    ) -> ReflectionRecord:
        """Reflect on action outcome."""
        # Will be used when LLM integration is added
        _ = (expected, context)
        if action.success:
            return ReflectionRecord(
                observation=f"Action {action.tool_name} succeeded",
                analysis="Result matches expectations",
                improvements=[],
                should_retry=False,
                confidence_delta=0.05,
            )
        else:
            return ReflectionRecord(
                observation=f"Action {action.tool_name} failed: {action.error}",
                analysis="Need to investigate failure cause",
                improvements=["Check parameters", "Verify tool availability"],
                should_retry=action.retry_count < 3,
                confidence_delta=-0.1,
            )


# =============================================================================
# Main Executor Class
# =============================================================================

class AgentExecutor:
    """
    Unified Agent Executor implementing the ReAct pattern.

    Combines:
        - AgentHarness for context management
        - MemorySystem for persistent memory
        - CooperationManager for multi-agent coordination
        - MCPServerManager for tool execution

    Execution Flow:
        1. THINK: Reason about the task (extended thinking)
        2. PLAN: Decompose into actionable steps
        3. ACT: Execute tools via MCP
        4. OBSERVE: Process and validate results
        5. REFLECT: Self-critique and adjust
        6. UPDATE: Update memory and state
        7. Loop until COMPLETE or max iterations
    """

    def __init__(
        self,
        harness: AgentHarness,
        memory: MemorySystem,
        cooperation: CooperationManager,
        mcp_manager: MCPServerManager,
        thinking_config: Optional[ThinkingConfig] = None,
        max_iterations: int = 50,
        enable_auto_research: bool = True,
    ):
        self._harness = harness
        self._memory = memory
        self._cooperation = cooperation
        self._mcp = mcp_manager

        self._thinking_config = thinking_config or ThinkingConfig()
        self._max_iterations = max_iterations

        # Execution components
        self._tool_executor = MCPToolExecutor(mcp_manager)
        self._reasoning = SimpleReasoningEngine(self._thinking_config)

        # Research engine integration
        self._research_engine = None
        self._auto_research_enabled = enable_auto_research
        if enable_auto_research and RESEARCH_ENGINE_AVAILABLE and get_research_engine:
            try:
                self._research_engine = get_research_engine()
                logger.info("[EXECUTOR] Research engine integrated")
            except Exception as e:
                logger.warning(f"[EXECUTOR] Could not load research engine: {e}")

        # State tracking
        self._current_state: Optional[ExecutorState] = None
        self._running = False
        self._pause_requested = False

        # Callbacks
        self._on_phase_change: Optional[Callable[[ExecutionPhase], None]] = None
        self._on_action: Optional[Callable[[ActionResult], None]] = None
        self._on_thought: Optional[Callable[[ThoughtRecord], None]] = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def state(self) -> Optional[ExecutorState]:
        """Get current execution state."""
        return self._current_state

    @property
    def is_running(self) -> bool:
        """Check if executor is running."""
        return self._running

    @property
    def available_tools(self) -> List[ToolSchema]:
        """Get all available tools from MCP servers."""
        return self._mcp.get_all_tools()

    # -------------------------------------------------------------------------
    # Callback Registration
    # -------------------------------------------------------------------------

    def on_phase_change(self, callback: Callable[[ExecutionPhase], None]) -> None:
        """Register callback for phase changes."""
        self._on_phase_change = callback

    def on_action(self, callback: Callable[[ActionResult], None]) -> None:
        """Register callback for action completions."""
        self._on_action = callback

    def on_thought(self, callback: Callable[[ThoughtRecord], None]) -> None:
        """Register callback for thinking steps."""
        self._on_thought = callback

    # -------------------------------------------------------------------------
    # Research Methods
    # -------------------------------------------------------------------------

    @property
    def has_research(self) -> bool:
        """Check if research engine is available."""
        return self._research_engine is not None

    async def research(
        self,
        query: str,
        deep: bool = False,
        sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Perform research on a query.

        Args:
            query: Research query
            deep: Use deep research (multiple sources, synthesis)
            sources: Preferred sources ('exa', 'firecrawl', 'both')

        Returns:
            Research results with sources and content
        """
        if not self._research_engine:
            return {"error": "Research engine not available", "success": False}

        results: Dict[str, Any] = {"query": query, "sources": [], "success": False}

        try:
            # Exa search (primary for semantic search)
            if not sources or "exa" in sources or "both" in sources:
                exa_result = self._research_engine.exa_search(query, num_results=5)
                if exa_result.get("success"):
                    results["exa"] = exa_result.get("results", [])
                    results["sources"].append("exa")

            # Firecrawl search (for web content)
            if sources and ("firecrawl" in sources or "both" in sources):
                fc_result = self._research_engine.firecrawl_search(query, limit=5)
                if fc_result.get("success"):
                    results["firecrawl"] = fc_result.get("data", [])
                    results["sources"].append("firecrawl")

            # Deep research with Exa answer
            if deep and hasattr(self._research_engine, "exa_answer"):
                answer_result = self._research_engine.exa_answer(query)
                if answer_result.get("success"):
                    results["answer"] = answer_result.get("answer")
                    results["citations"] = answer_result.get("citations", [])

            results["success"] = bool(results.get("sources"))
            return results

        except Exception as e:
            logger.error(f"Research error: {e}")
            return {"error": str(e), "success": False}

    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """Scrape content from a URL."""
        if not self._research_engine:
            return {"error": "Research engine not available", "success": False}
        return self._research_engine.scrape(url)

    async def auto_research_if_needed(self, task: str) -> Optional[Dict[str, Any]]:
        """
        Automatically trigger research if task seems to need external info.

        Keywords that trigger auto-research:
        - "research", "find out", "what is", "how does", "explain"
        - "latest", "current", "recent", "2024", "2025", "2026"
        - "compare", "analyze", "investigate"
        """
        if not self._auto_research_enabled or not self._research_engine:
            return None

        triggers = [
            "research", "find out", "find information", "what is", "how does",
            "explain", "latest", "current", "recent", "2024", "2025", "2026",
            "compare", "analyze", "investigate", "look up", "search for"
        ]

        task_lower = task.lower()
        if any(trigger in task_lower for trigger in triggers):
            logger.info(f"[AUTO-RESEARCH] Triggered for: {task[:50]}...")
            return await self.research(task, deep=True)

        return None

    # -------------------------------------------------------------------------
    # Core Execution Methods
    # -------------------------------------------------------------------------

    async def execute(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
    ) -> ExecutorState:
        """
        Execute a task using the ReAct loop.

        Args:
            task: The task description to execute
            context: Optional additional context
            task_id: Optional task ID (generated if not provided)

        Returns:
            ExecutorState with final results
        """
        # Initialize state
        self._current_state = ExecutorState(
            task_id=task_id or str(uuid.uuid4()),
            task_description=task,
            status=TaskStatus.RUNNING,
            current_phase=ExecutionPhase.THINK,
            max_iterations=self._max_iterations,
        )

        self._running = True
        self._pause_requested = False

        # Initialize harness with task
        self._harness.begin_task(task, self._current_state.task_id)

        # Add initial context
        if context:
            for key, value in context.items():
                self._harness.add_to_context(
                    key=key,
                    content=str(value),
                    pillar=ContextPillar.KNOWLEDGE,
                    priority=ContextPriority.MEDIUM,
                )

        # Store task in memory (serialize dict to JSON string)
        self._memory.core.update(
            f"task:{self._current_state.task_id}",
            json.dumps({"description": task, "status": "running"}),
        )

        try:
            # Execute ReAct loop
            while self._should_continue():
                await self._execute_iteration()

            # Mark completion
            self._current_state.status = TaskStatus.COMPLETED
            self._current_state.completed_at = time.time()

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            self._current_state.status = TaskStatus.FAILED
            self._current_state.error_message = str(e)
            self._current_state.completed_at = time.time()

        finally:
            self._running = False

            # Archive to memory (async operation, schedule in background)
            try:
                asyncio.create_task(
                    self._memory.archival.store(
                        content=json.dumps(self._current_state.model_dump()),
                        category="execution",
                        metadata={"task_id": self._current_state.task_id},
                    )
                )
            except Exception:
                pass  # Best-effort archival

        return self._current_state

    def _should_continue(self) -> bool:
        """Check if execution should continue."""
        if not self._running:
            return False

        if self._pause_requested:
            return False

        state = self._current_state
        if not state:
            return False

        if state.current_phase == ExecutionPhase.COMPLETE:
            return False

        if state.iteration_count >= state.max_iterations:
            logger.warning(f"Max iterations ({state.max_iterations}) reached")
            return False

        return True

    async def _execute_iteration(self) -> None:
        """Execute one iteration of the ReAct loop."""
        state = self._current_state
        if not state:
            return

        state.iteration_count += 1
        logger.info(f"Iteration {state.iteration_count}: Phase {state.current_phase}")

        # Execute current phase
        if state.current_phase == ExecutionPhase.THINK:
            await self._phase_think()
        elif state.current_phase == ExecutionPhase.PLAN:
            await self._phase_plan()
        elif state.current_phase == ExecutionPhase.ACT:
            await self._phase_act()
        elif state.current_phase == ExecutionPhase.OBSERVE:
            await self._phase_observe()
        elif state.current_phase == ExecutionPhase.REFLECT:
            await self._phase_reflect()
        elif state.current_phase == ExecutionPhase.UPDATE:
            await self._phase_update()

    async def _phase_think(self) -> None:
        """THINK phase: Reason about the current state."""
        state = self._current_state
        if not state:
            return

        self._notify_phase_change(ExecutionPhase.THINK)

        # Get context summary
        context = self._harness.context.get_context_summary()

        # Generate thought
        thought = await self._reasoning.think(
            context=context,
            question=state.task_description,
            config=self._thinking_config,
        )

        state.thoughts.append(thought)
        self._notify_thought(thought)

        # Record in harness
        self._harness.record_decision(
            decision="Completed thinking phase",
            rationale=thought.content,
            alternatives=None,
        )

        # Transition to PLAN
        state.current_phase = ExecutionPhase.PLAN

    async def _phase_plan(self) -> None:
        """PLAN phase: Decompose task into steps."""
        state = self._current_state
        if not state:
            return

        self._notify_phase_change(ExecutionPhase.PLAN)

        # Generate plan
        context = self._harness.context.get_context_summary()
        steps = await self._reasoning.plan(
            task=state.task_description,
            context=context,
            available_tools=self.available_tools,
        )

        # Store plan as thought
        plan_thought = ThoughtRecord(
            phase=ExecutionPhase.PLAN,
            content=f"Plan steps: {'; '.join(steps)}",
            confidence=0.75,
        )
        state.thoughts.append(plan_thought)
        self._notify_thought(plan_thought)

        # Add plan to context
        self._harness.add_to_context(
            key="current_plan",
            content="\n".join(f"- {step}" for step in steps),
            pillar=ContextPillar.INSTRUCTIONS,
            priority=ContextPriority.HIGH,
        )

        # Transition to ACT
        state.current_phase = ExecutionPhase.ACT

    async def _phase_act(self) -> None:
        """ACT phase: Execute tools."""
        state = self._current_state
        if not state:
            return

        self._notify_phase_change(ExecutionPhase.ACT)

        # For demo, execute a simple action
        # In production, this would parse the plan and execute appropriate tools
        tools = self.available_tools
        if tools:
            action_request = ActionRequest(
                tool_name=tools[0].name,
                parameters={},
            )

            result = await self._tool_executor.execute(
                tool_name=action_request.tool_name,
                parameters=action_request.parameters,
                timeout=action_request.timeout_seconds,
            )

            state.actions.append(result)
            self._notify_action(result)

            # Record in harness
            self._harness.record_action(
                action=f"Executed tool: {result.tool_name}",
                result=str(result.result) if result.success else result.error or "Unknown error",
                success=result.success,
            )

        # Transition to OBSERVE
        state.current_phase = ExecutionPhase.OBSERVE

    async def _phase_observe(self) -> None:
        """OBSERVE phase: Process action results."""
        state = self._current_state
        if not state:
            return

        self._notify_phase_change(ExecutionPhase.OBSERVE)

        # Get latest action result
        if state.actions:
            latest_action = state.actions[-1]

            observation = ThoughtRecord(
                phase=ExecutionPhase.OBSERVE,
                content=f"Observed: {latest_action.tool_name} -> {'success' if latest_action.success else 'failed'}",
                confidence=0.9 if latest_action.success else 0.5,
            )
            state.thoughts.append(observation)
            self._notify_thought(observation)

        # Transition to REFLECT
        state.current_phase = ExecutionPhase.REFLECT

    async def _phase_reflect(self) -> None:
        """REFLECT phase: Self-critique and adjust."""
        state = self._current_state
        if not state:
            return

        self._notify_phase_change(ExecutionPhase.REFLECT)

        # Reflect on latest action
        if state.actions:
            latest_action = state.actions[-1]

            reflection = await self._reasoning.reflect(
                action=latest_action,
                expected="Task completion",
                context=self._harness.context.get_context_summary(),
            )

            state.reflections.append(reflection)

            # Record reflection
            self._harness.record_decision(
                decision="Reflection complete",
                rationale=reflection.analysis,
                alternatives=reflection.improvements if reflection.improvements else None,
            )

            # Determine if task is complete (simplified logic)
            if latest_action.success and state.iteration_count >= 3:
                state.current_phase = ExecutionPhase.COMPLETE
                state.final_result = latest_action.result
                return

        # Transition to UPDATE
        state.current_phase = ExecutionPhase.UPDATE

    async def _phase_update(self) -> None:
        """UPDATE phase: Update memory and state."""
        state = self._current_state
        if not state:
            return

        self._notify_phase_change(ExecutionPhase.UPDATE)

        # Update memory with current state
        self._memory.core.update(
            f"task:{state.task_id}:iteration:{state.iteration_count}",
            json.dumps({
                "phase": state.current_phase.value,
                "thoughts_count": len(state.thoughts),
                "actions_count": len(state.actions),
            }),
        )

        # Check if we should continue or complete
        if state.iteration_count >= self._max_iterations // 2:
            # For demo, complete after half max iterations
            state.current_phase = ExecutionPhase.COMPLETE
            state.final_result = {
                "iterations": state.iteration_count,
                "actions": len(state.actions),
                "thoughts": len(state.thoughts),
            }
        else:
            # Continue loop
            state.current_phase = ExecutionPhase.THINK

    # -------------------------------------------------------------------------
    # Control Methods
    # -------------------------------------------------------------------------

    def pause(self) -> None:
        """Request execution pause."""
        self._pause_requested = True
        if self._current_state:
            self._current_state.status = TaskStatus.PAUSED

    def resume(self) -> None:
        """Resume paused execution."""
        self._pause_requested = False
        if self._current_state:
            self._current_state.status = TaskStatus.RUNNING

    def cancel(self) -> None:
        """Cancel execution."""
        self._running = False
        if self._current_state:
            self._current_state.status = TaskStatus.CANCELLED
            self._current_state.completed_at = time.time()

    # -------------------------------------------------------------------------
    # Handoff Methods
    # -------------------------------------------------------------------------

    def create_handoff(self) -> ShiftHandoff:
        """Create a shift handoff for session continuity."""
        return self._harness.create_shift_handoff()

    @classmethod
    def from_handoff(
        cls,
        handoff: ShiftHandoff,
        memory: MemorySystem,
        cooperation: CooperationManager,
        mcp_manager: MCPServerManager,
        thinking_config: Optional[ThinkingConfig] = None,
    ) -> "AgentExecutor":
        """Create executor from a previous handoff."""
        harness = AgentHarness.from_handoff(handoff)

        return cls(
            harness=harness,
            memory=memory,
            cooperation=cooperation,
            mcp_manager=mcp_manager,
            thinking_config=thinking_config,
        )

    # -------------------------------------------------------------------------
    # Notification Methods
    # -------------------------------------------------------------------------

    def _notify_phase_change(self, phase: ExecutionPhase) -> None:
        """Notify phase change callback."""
        if self._on_phase_change:
            self._on_phase_change(phase)

    def _notify_action(self, result: ActionResult) -> None:
        """Notify action callback."""
        if self._on_action:
            self._on_action(result)

    def _notify_thought(self, thought: ThoughtRecord) -> None:
        """Notify thought callback."""
        if self._on_thought:
            self._on_thought(thought)


# =============================================================================
# Factory Function
# =============================================================================

def create_executor(
    agent_id: Optional[str] = None,
    max_tokens: int = 100000,
    storage_path: Optional[Path] = None,
    thinking_mode: ThinkingMode = ThinkingMode.INTERLEAVED,
    thinking_budget: int = 16000,
) -> AgentExecutor:
    """
    Factory function to create a fully configured AgentExecutor.

    Args:
        agent_id: Unique identifier for this agent (auto-generated if None)
        max_tokens: Maximum context window size
        storage_path: Path for persistent storage
        thinking_mode: Extended thinking mode
        thinking_budget: Token budget for thinking

    Returns:
        Configured AgentExecutor instance
    """
    # Generate agent ID if not provided
    if agent_id is None:
        agent_id = f"executor-{uuid.uuid4().hex[:8]}"

    # Create components
    harness = AgentHarness(max_tokens=max_tokens, storage_path=storage_path)
    memory = MemorySystem(agent_id=agent_id, storage_base=storage_path)
    cooperation = CooperationManager(storage_path=storage_path)
    mcp_manager = MCPServerManager()

    thinking_config = ThinkingConfig(
        mode=thinking_mode,
        budget_tokens=thinking_budget,
    )

    return AgentExecutor(
        harness=harness,
        memory=memory,
        cooperation=cooperation,
        mcp_manager=mcp_manager,
        thinking_config=thinking_config,
    )


# =============================================================================
# Demo
# =============================================================================

async def demo():
    """Demonstrate the AgentExecutor."""
    print("=" * 60)
    print("UAP Agent Executor Demo")
    print("=" * 60)

    # Create executor
    executor = create_executor(
        thinking_mode=ThinkingMode.INTERLEAVED,
        thinking_budget=8000,
    )

    # Register callbacks
    executor.on_phase_change(lambda p: print(f"  Phase: {p.value}"))
    executor.on_action(lambda a: print(f"  Action: {a.tool_name} -> {'OK' if a.success else 'FAIL'}"))
    executor.on_thought(lambda t: print(f"  Thought: {t.content[:50]}..."))

    # Execute a task
    print("\n[Starting Task Execution]")
    result = await executor.execute(
        task="Analyze the codebase structure and identify key modules",
        context={"project": "UAP", "focus": "architecture"},
    )

    # Print results
    print("\n[Execution Complete]")
    print(f"  Status: {result.status.value}")
    print(f"  Iterations: {result.iteration_count}")
    print(f"  Thoughts: {len(result.thoughts)}")
    print(f"  Actions: {len(result.actions)}")
    print(f"  Reflections: {len(result.reflections)}")

    if result.final_result:
        print(f"  Result: {result.final_result}")

    # Create handoff
    handoff = executor.create_handoff()
    print(f"\n[Handoff Created]")
    print(f"  Task ID: {handoff.task_id}")
    print(f"  Completed steps: {len(handoff.completed_steps)}")
    print(f"  Remaining steps: {len(handoff.remaining_steps)}")


if __name__ == "__main__":
    asyncio.run(demo())
