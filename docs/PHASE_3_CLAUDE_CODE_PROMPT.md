# Phase 3: Orchestration Layer Setup

**Layer 1 - 5 Core Orchestration SDKs**

Execute this prompt in Claude Code CLI to implement the Orchestration Layer.

---

## Pre-Flight Validation

Before beginning, verify Phase 1 & 2 are complete:

```bash
# Verify Phase 2 validation passes
cd /path/to/unleash
python scripts/validate_phase2.py

# Expected: "Phase 2 Validation PASSED"
```

**Required Checks:**
- [ ] LLM Gateway operational (`core/llm_gateway.py`)
- [ ] MCP Server functional (`core/mcp_server.py`)
- [ ] Providers working (`core/providers/`)
- [ ] All Phase 2 SDKs installed

---

## Step 1: Create Orchestration Directory Structure

```bash
mkdir -p core/orchestration
```

Create the directory with proper Python package structure.

---

## Step 2: Create Temporal Workflows

Create `core/orchestration/temporal_workflows.py`:

```python
#!/usr/bin/env python3
"""
Temporal Workflow Orchestration
Durable, fault-tolerant workflow execution for AI agents.
"""

from __future__ import annotations

import asyncio
import os
from datetime import timedelta
from typing import Any, Optional
from dataclasses import dataclass, field

import structlog
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment
load_dotenv()

logger = structlog.get_logger(__name__)

# Import Temporal SDK
try:
    from temporalio import workflow, activity
    from temporalio.client import Client
    from temporalio.worker import Worker
    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False
    logger.warning("temporalio not available - install with: pip install temporalio")


# ============================================
# Workflow State Models
# ============================================

class WorkflowInput(BaseModel):
    """Input for orchestration workflows."""
    task_id: str
    task_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=3600, ge=1)


class WorkflowOutput(BaseModel):
    """Output from orchestration workflows."""
    task_id: str
    status: str  # "completed", "failed", "cancelled"
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: int = 0


class LLMTaskInput(BaseModel):
    """Input for LLM completion tasks."""
    prompt: str
    system: str = "You are a helpful assistant."
    model: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 4096


# ============================================
# Activities (Atomic Work Units)
# ============================================

if TEMPORAL_AVAILABLE:

    @activity.defn
    async def llm_complete_activity(task_input: dict) -> dict:
        """
        Execute LLM completion via the gateway.
        
        This activity wraps the LLM Gateway for use in Temporal workflows.
        """
        logger.info("activity_started", activity="llm_complete", task=task_input.get("task_id"))
        
        try:
            from core.llm_gateway import LLMGateway, Message, ModelConfig, Provider
            
            gateway = LLMGateway()
            input_data = LLMTaskInput(**task_input)
            
            messages = [
                Message(role="system", content=input_data.system),
                Message(role="user", content=input_data.prompt),
            ]
            
            provider = Provider.OPENAI if "gpt" in input_data.model.lower() else Provider.ANTHROPIC
            config = ModelConfig(
                provider=provider,
                model_id=input_data.model,
                max_tokens=input_data.max_tokens,
            )
            
            response = await gateway.complete(messages, config)
            
            logger.info("activity_completed", activity="llm_complete")
            return {
                "content": response.content,
                "model": response.model,
                "usage": response.usage,
            }
            
        except Exception as e:
            logger.error("activity_failed", activity="llm_complete", error=str(e))
            raise


    @activity.defn
    async def analyze_task_activity(task_input: dict) -> dict:
        """
        Analyze a task and determine execution strategy.
        """
        logger.info("activity_started", activity="analyze_task")
        
        task_type = task_input.get("task_type", "generic")
        payload = task_input.get("payload", {})
        
        # Simple task analysis logic
        analysis = {
            "task_type": task_type,
            "complexity": "medium",
            "estimated_steps": 1,
            "requires_tools": False,
        }
        
        # Determine complexity based on payload
        if "multi_step" in payload or "chain" in payload:
            analysis["complexity"] = "high"
            analysis["estimated_steps"] = payload.get("steps", 3)
        
        if "tools" in payload:
            analysis["requires_tools"] = True
        
        logger.info("activity_completed", activity="analyze_task", analysis=analysis)
        return analysis


    @activity.defn
    async def execute_tool_activity(tool_name: str, tool_input: dict) -> dict:
        """
        Execute a tool via the MCP server.
        """
        logger.info("activity_started", activity="execute_tool", tool=tool_name)
        
        try:
            # Import MCP tools
            from core.mcp_server import create_mcp_server
            
            # Note: In production, you'd call the tool through the MCP protocol
            # This is a simplified version for workflow integration
            result = {
                "tool": tool_name,
                "status": "executed",
                "output": f"Tool {tool_name} executed with input: {tool_input}",
            }
            
            logger.info("activity_completed", activity="execute_tool", tool=tool_name)
            return result
            
        except Exception as e:
            logger.error("activity_failed", activity="execute_tool", tool=tool_name, error=str(e))
            raise


# ============================================
# Workflows (Orchestration Logic)
# ============================================

if TEMPORAL_AVAILABLE:

    @workflow.defn
    class SimpleAgentWorkflow:
        """
        Simple single-agent workflow.
        
        Executes a single LLM completion task with optional tool use.
        """
        
        @workflow.run
        async def run(self, input_data: dict) -> dict:
            """Execute the simple agent workflow."""
            workflow_input = WorkflowInput(**input_data)
            
            logger.info(
                "workflow_started",
                workflow="SimpleAgentWorkflow",
                task_id=workflow_input.task_id,
            )
            
            start_time = workflow.now()
            
            try:
                # Analyze the task
                analysis = await workflow.execute_activity(
                    analyze_task_activity,
                    input_data,
                    start_to_close_timeout=timedelta(seconds=30),
                )
                
                # Execute LLM completion
                llm_input = {
                    "prompt": workflow_input.payload.get("prompt", "Hello"),
                    "system": workflow_input.payload.get("system", "You are a helpful assistant."),
                    "model": workflow_input.config.get("model", "claude-3-5-sonnet-20241022"),
                }
                
                result = await workflow.execute_activity(
                    llm_complete_activity,
                    llm_input,
                    start_to_close_timeout=timedelta(seconds=workflow_input.timeout_seconds),
                )
                
                execution_time = int((workflow.now() - start_time).total_seconds() * 1000)
                
                output = WorkflowOutput(
                    task_id=workflow_input.task_id,
                    status="completed",
                    result=result,
                    execution_time_ms=execution_time,
                )
                
                logger.info(
                    "workflow_completed",
                    workflow="SimpleAgentWorkflow",
                    task_id=workflow_input.task_id,
                    execution_time_ms=execution_time,
                )
                
                return output.model_dump()
                
            except Exception as e:
                logger.error(
                    "workflow_failed",
                    workflow="SimpleAgentWorkflow",
                    task_id=workflow_input.task_id,
                    error=str(e),
                )
                
                return WorkflowOutput(
                    task_id=workflow_input.task_id,
                    status="failed",
                    error=str(e),
                ).model_dump()


    @workflow.defn
    class ChainedAgentWorkflow:
        """
        Multi-step chained agent workflow.
        
        Executes a sequence of LLM calls, passing outputs to inputs.
        """
        
        @workflow.run
        async def run(self, input_data: dict) -> dict:
            """Execute the chained agent workflow."""
            workflow_input = WorkflowInput(**input_data)
            
            logger.info(
                "workflow_started",
                workflow="ChainedAgentWorkflow",
                task_id=workflow_input.task_id,
            )
            
            start_time = workflow.now()
            steps = workflow_input.payload.get("steps", [])
            results = []
            
            try:
                current_context = workflow_input.payload.get("initial_context", "")
                
                for i, step in enumerate(steps):
                    # Build prompt with context from previous steps
                    prompt = step.get("prompt", "")
                    if current_context:
                        prompt = f"Context from previous step:\n{current_context}\n\nTask:\n{prompt}"
                    
                    llm_input = {
                        "prompt": prompt,
                        "system": step.get("system", "You are a helpful assistant."),
                        "model": step.get("model", "claude-3-5-sonnet-20241022"),
                    }
                    
                    result = await workflow.execute_activity(
                        llm_complete_activity,
                        llm_input,
                        start_to_close_timeout=timedelta(seconds=120),
                    )
                    
                    results.append({
                        "step": i + 1,
                        "result": result,
                    })
                    
                    # Update context for next step
                    current_context = result.get("content", "")
                
                execution_time = int((workflow.now() - start_time).total_seconds() * 1000)
                
                output = WorkflowOutput(
                    task_id=workflow_input.task_id,
                    status="completed",
                    result={"steps": results, "final_output": current_context},
                    execution_time_ms=execution_time,
                )
                
                logger.info(
                    "workflow_completed",
                    workflow="ChainedAgentWorkflow",
                    task_id=workflow_input.task_id,
                    steps_executed=len(results),
                )
                
                return output.model_dump()
                
            except Exception as e:
                logger.error(
                    "workflow_failed",
                    workflow="ChainedAgentWorkflow",
                    task_id=workflow_input.task_id,
                    error=str(e),
                )
                
                return WorkflowOutput(
                    task_id=workflow_input.task_id,
                    status="failed",
                    error=str(e),
                    result={"completed_steps": results},
                ).model_dump()


# ============================================
# Temporal Client Manager
# ============================================

@dataclass
class TemporalOrchestrator:
    """
    Manager for Temporal workflow orchestration.
    
    Provides high-level interface for starting and managing workflows.
    """
    
    temporal_host: str = "localhost:7233"
    namespace: str = "default"
    task_queue: str = "unleash-agents"
    _client: Optional[Any] = field(default=None, init=False)
    
    async def connect(self) -> None:
        """Connect to Temporal server."""
        if not TEMPORAL_AVAILABLE:
            raise ImportError("temporalio not installed")
        
        self._client = await Client.connect(self.temporal_host, namespace=self.namespace)
        logger.info("temporal_connected", host=self.temporal_host, namespace=self.namespace)
    
    async def disconnect(self) -> None:
        """Disconnect from Temporal server."""
        self._client = None
        logger.info("temporal_disconnected")
    
    async def start_simple_agent(
        self,
        task_id: str,
        prompt: str,
        system: str = "You are a helpful assistant.",
        model: str = "claude-3-5-sonnet-20241022",
    ) -> str:
        """
        Start a simple single-agent workflow.
        
        Returns the workflow ID for status tracking.
        """
        if not self._client:
            await self.connect()
        
        workflow_input = WorkflowInput(
            task_id=task_id,
            task_type="simple_agent",
            payload={"prompt": prompt, "system": system},
            config={"model": model},
        )
        
        handle = await self._client.start_workflow(
            SimpleAgentWorkflow.run,
            workflow_input.model_dump(),
            id=f"simple-{task_id}",
            task_queue=self.task_queue,
        )
        
        logger.info("workflow_started", workflow_id=handle.id, task_id=task_id)
        return handle.id
    
    async def start_chained_agent(
        self,
        task_id: str,
        steps: list[dict],
        initial_context: str = "",
    ) -> str:
        """
        Start a chained multi-step agent workflow.
        
        Args:
            task_id: Unique task identifier
            steps: List of step configs with prompt, system, model
            initial_context: Initial context to seed the chain
        
        Returns the workflow ID.
        """
        if not self._client:
            await self.connect()
        
        workflow_input = WorkflowInput(
            task_id=task_id,
            task_type="chained_agent",
            payload={"steps": steps, "initial_context": initial_context},
        )
        
        handle = await self._client.start_workflow(
            ChainedAgentWorkflow.run,
            workflow_input.model_dump(),
            id=f"chained-{task_id}",
            task_queue=self.task_queue,
        )
        
        logger.info("workflow_started", workflow_id=handle.id, task_id=task_id)
        return handle.id
    
    async def get_workflow_result(self, workflow_id: str) -> dict:
        """Get the result of a completed workflow."""
        if not self._client:
            await self.connect()
        
        handle = self._client.get_workflow_handle(workflow_id)
        result = await handle.result()
        return result
    
    async def cancel_workflow(self, workflow_id: str) -> None:
        """Cancel a running workflow."""
        if not self._client:
            await self.connect()
        
        handle = self._client.get_workflow_handle(workflow_id)
        await handle.cancel()
        logger.info("workflow_cancelled", workflow_id=workflow_id)


async def run_temporal_worker() -> None:
    """Run the Temporal worker to process workflows."""
    if not TEMPORAL_AVAILABLE:
        raise ImportError("temporalio not installed")
    
    client = await Client.connect("localhost:7233")
    
    worker = Worker(
        client,
        task_queue="unleash-agents",
        workflows=[SimpleAgentWorkflow, ChainedAgentWorkflow],
        activities=[llm_complete_activity, analyze_task_activity, execute_tool_activity],
    )
    
    logger.info("temporal_worker_starting", task_queue="unleash-agents")
    await worker.run()


if __name__ == "__main__":
    import asyncio
    
    async def main():
        """Test Temporal orchestration."""
        if not TEMPORAL_AVAILABLE:
            print("Temporal SDK not installed. Install with: pip install temporalio")
            return
        
        print("Temporal Workflows Module")
        print("-" * 40)
        print("Available workflows:")
        print("  - SimpleAgentWorkflow")
        print("  - ChainedAgentWorkflow")
        print("\nTo run worker: python -m core.orchestration.temporal_workflows")
        print("\nNote: Requires Temporal server running on localhost:7233")
    
    asyncio.run(main())
```

---

## Step 3: Create LangGraph Agents

Create `core/orchestration/langgraph_agents.py`:

```python
#!/usr/bin/env python3
"""
LangGraph Multi-Agent Orchestration
Graph-based agent workflows using LangChain's LangGraph.
"""

from __future__ import annotations

import asyncio
import operator
from typing import Any, Annotated, Optional, TypedDict
from dataclasses import dataclass, field
from enum import Enum

import structlog
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment
load_dotenv()

logger = structlog.get_logger(__name__)

# Import LangGraph
try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("langgraph not available - install with: pip install langgraph langchain-core")


# ============================================
# State Definitions
# ============================================

class AgentState(TypedDict):
    """State for LangGraph agent workflows."""
    messages: Annotated[list[BaseMessage], operator.add]
    current_agent: str
    task_complete: bool
    iteration: int
    max_iterations: int
    results: dict[str, Any]


class AgentRole(str, Enum):
    """Pre-defined agent roles."""
    PLANNER = "planner"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    COORDINATOR = "coordinator"


class AgentConfig(BaseModel):
    """Configuration for a LangGraph agent."""
    name: str
    role: AgentRole
    system_prompt: str
    model: str = "claude-3-5-sonnet-20241022"
    max_tokens: int = 4096


# ============================================
# Agent Nodes
# ============================================

def create_llm_node(config: AgentConfig):
    """Create an LLM node that uses the unified gateway."""
    
    async def llm_node(state: AgentState) -> dict:
        """Execute LLM completion for this agent."""
        logger.info("agent_node_executing", agent=config.name, role=config.role.value)
        
        try:
            from core.llm_gateway import LLMGateway, Message, ModelConfig, Provider
            
            gateway = LLMGateway()
            
            # Convert LangChain messages to gateway format
            messages = [Message(role="system", content=config.system_prompt)]
            
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    messages.append(Message(role="user", content=msg.content))
                elif isinstance(msg, AIMessage):
                    messages.append(Message(role="assistant", content=msg.content))
            
            # Determine provider
            provider = Provider.OPENAI if "gpt" in config.model.lower() else Provider.ANTHROPIC
            
            model_config = ModelConfig(
                provider=provider,
                model_id=config.model,
                max_tokens=config.max_tokens,
            )
            
            response = await gateway.complete(messages, model_config)
            
            logger.info("agent_node_completed", agent=config.name)
            
            return {
                "messages": [AIMessage(content=response.content, name=config.name)],
                "iteration": state["iteration"] + 1,
            }
            
        except Exception as e:
            logger.error("agent_node_failed", agent=config.name, error=str(e))
            return {
                "messages": [AIMessage(content=f"Error: {str(e)}", name=config.name)],
                "iteration": state["iteration"] + 1,
            }
    
    return llm_node


def create_router_node():
    """Create a router node that determines next agent."""
    
    def router(state: AgentState) -> str:
        """Route to next agent or end."""
        if state["task_complete"]:
            return END
        
        if state["iteration"] >= state["max_iterations"]:
            return END
        
        # Simple round-robin for demo - in production, use LLM to decide
        agents = ["planner", "executor", "reviewer"]
        current_idx = agents.index(state["current_agent"]) if state["current_agent"] in agents else -1
        next_idx = (current_idx + 1) % len(agents)
        
        return agents[next_idx]
    
    return router


# ============================================
# Graph Builder
# ============================================

@dataclass
class LangGraphOrchestrator:
    """
    LangGraph-based multi-agent orchestrator.
    
    Creates and manages graph-based agent workflows.
    """
    
    agents: dict[str, AgentConfig] = field(default_factory=dict)
    _graph: Optional[Any] = field(default=None, init=False)
    
    def add_agent(self, config: AgentConfig) -> None:
        """Add an agent to the orchestrator."""
        self.agents[config.name] = config
        logger.info("agent_added", name=config.name, role=config.role.value)
    
    def build_sequential_graph(self, agent_sequence: list[str]) -> Any:
        """
        Build a sequential agent graph.
        
        Agents execute in the specified order.
        """
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("langgraph not installed")
        
        builder = StateGraph(AgentState)
        
        # Add nodes for each agent
        for agent_name in agent_sequence:
            if agent_name not in self.agents:
                raise ValueError(f"Agent {agent_name} not found")
            
            config = self.agents[agent_name]
            builder.add_node(agent_name, create_llm_node(config))
        
        # Set entry point
        builder.set_entry_point(agent_sequence[0])
        
        # Add edges in sequence
        for i in range(len(agent_sequence) - 1):
            builder.add_edge(agent_sequence[i], agent_sequence[i + 1])
        
        # Final agent goes to END
        builder.add_edge(agent_sequence[-1], END)
        
        self._graph = builder.compile()
        logger.info("graph_built", type="sequential", agents=agent_sequence)
        
        return self._graph
    
    def build_cyclic_graph(
        self,
        agents: list[str],
        should_continue_fn: Optional[Any] = None,
    ) -> Any:
        """
        Build a cyclic agent graph with conditional routing.
        
        Agents can loop until a condition is met.
        """
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("langgraph not installed")
        
        builder = StateGraph(AgentState)
        
        # Add nodes
        for agent_name in agents:
            if agent_name not in self.agents:
                raise ValueError(f"Agent {agent_name} not found")
            
            config = self.agents[agent_name]
            builder.add_node(agent_name, create_llm_node(config))
        
        # Add router node
        def default_should_continue(state: AgentState) -> str:
            if state["task_complete"] or state["iteration"] >= state["max_iterations"]:
                return END
            return agents[(state["iteration"]) % len(agents)]
        
        router_fn = should_continue_fn or default_should_continue
        
        builder.set_entry_point(agents[0])
        
        # Add conditional edges from each agent to router
        for agent_name in agents:
            builder.add_conditional_edges(
                agent_name,
                router_fn,
                {name: name for name in agents} | {END: END},
            )
        
        self._graph = builder.compile()
        logger.info("graph_built", type="cyclic", agents=agents)
        
        return self._graph
    
    async def run(
        self,
        initial_message: str,
        max_iterations: int = 10,
        config: Optional[dict] = None,
    ) -> dict:
        """
        Execute the agent graph.
        
        Args:
            initial_message: Starting user message
            max_iterations: Maximum agent iterations
            config: Optional runtime config
        
        Returns:
            Final state with results
        """
        if not self._graph:
            raise RuntimeError("Graph not built - call build_* first")
        
        initial_state: AgentState = {
            "messages": [HumanMessage(content=initial_message)],
            "current_agent": "",
            "task_complete": False,
            "iteration": 0,
            "max_iterations": max_iterations,
            "results": {},
        }
        
        logger.info("graph_execution_starting", initial_message=initial_message[:50])
        
        # Execute graph
        final_state = await self._graph.ainvoke(initial_state, config)
        
        logger.info(
            "graph_execution_completed",
            iterations=final_state["iteration"],
            message_count=len(final_state["messages"]),
        )
        
        return final_state


# ============================================
# Pre-built Graph Templates
# ============================================

def create_planning_execution_graph() -> LangGraphOrchestrator:
    """
    Create a standard planning-execution-review graph.
    
    Three agents work together:
    1. Planner - breaks down tasks
    2. Executor - performs actions
    3. Reviewer - validates results
    """
    orchestrator = LangGraphOrchestrator()
    
    orchestrator.add_agent(AgentConfig(
        name="planner",
        role=AgentRole.PLANNER,
        system_prompt="""You are a task planner. Your job is to:
1. Analyze the given task
2. Break it down into clear, actionable steps
3. Identify any dependencies between steps
4. Output a structured plan

Be concise and specific. Format your plan as a numbered list.""",
    ))
    
    orchestrator.add_agent(AgentConfig(
        name="executor",
        role=AgentRole.EXECUTOR,
        system_prompt="""You are a task executor. Your job is to:
1. Follow the plan provided by the planner
2. Execute each step thoroughly
3. Document your actions and results
4. Flag any issues encountered

Work step by step and be thorough.""",
    ))
    
    orchestrator.add_agent(AgentConfig(
        name="reviewer",
        role=AgentRole.REVIEWER,
        system_prompt="""You are a quality reviewer. Your job is to:
1. Review the execution results
2. Check for completeness and accuracy
3. Identify any gaps or improvements
4. Provide final assessment

Be critical but constructive. Summarize the overall outcome.""",
    ))
    
    # Build sequential graph
    orchestrator.build_sequential_graph(["planner", "executor", "reviewer"])
    
    return orchestrator


def create_debate_graph() -> LangGraphOrchestrator:
    """
    Create a debate-style graph where agents discuss.
    
    Two agents with opposing viewpoints discuss a topic.
    """
    orchestrator = LangGraphOrchestrator()
    
    orchestrator.add_agent(AgentConfig(
        name="advocate",
        role=AgentRole.EXECUTOR,
        system_prompt="""You are an advocate who supports the proposition. Your job is to:
1. Present strong arguments in favor
2. Provide evidence and reasoning
3. Address counterarguments
4. Maintain a logical, persuasive tone

Build on previous points in the conversation.""",
    ))
    
    orchestrator.add_agent(AgentConfig(
        name="critic",
        role=AgentRole.REVIEWER,
        system_prompt="""You are a critic who challenges the proposition. Your job is to:
1. Present strong counterarguments
2. Question assumptions and evidence
3. Identify weaknesses in the advocate's position
4. Maintain a logical, analytical tone

Engage with specific points made by the advocate.""",
    ))
    
    orchestrator.add_agent(AgentConfig(
        name="moderator",
        role=AgentRole.COORDINATOR,
        system_prompt="""You are a debate moderator. Your job is to:
1. Summarize key points from both sides
2. Identify areas of agreement and disagreement
3. Determine if the debate should continue or conclude
4. If concluding, provide a balanced summary

Set task_complete to True when debate has reached a natural conclusion.""",
    ))
    
    # Build cyclic graph
    def debate_router(state: AgentState) -> str:
        if state["task_complete"] or state["iteration"] >= state["max_iterations"]:
            return END
        
        # Rotate through: advocate -> critic -> moderator
        rotation = ["advocate", "critic", "moderator"]
        idx = state["iteration"] % 3
        return rotation[idx]
    
    orchestrator.build_cyclic_graph(
        ["advocate", "critic", "moderator"],
        debate_router,
    )
    
    return orchestrator


if __name__ == "__main__":
    import asyncio
    
    async def main():
        """Test LangGraph orchestration."""
        if not LANGGRAPH_AVAILABLE:
            print("LangGraph not installed. Install with: pip install langgraph langchain-core")
            return
        
        print("LangGraph Agents Module")
        print("-" * 40)
        print("Available templates:")
        print("  - create_planning_execution_graph()")
        print("  - create_debate_graph()")
        print("\nExample usage:")
        print("  orchestrator = create_planning_execution_graph()")
        print("  result = await orchestrator.run('Build a REST API')")
    
    asyncio.run(main())
```

---

## Step 4: Create Claude Flow

Create `core/orchestration/claude_flow.py`:

```python
#!/usr/bin/env python3
"""
Claude Flow - Claude Native Multi-Agent Orchestration
Native Claude orchestration using tool use for multi-agent patterns.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

import structlog
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment
load_dotenv()

logger = structlog.get_logger(__name__)


# ============================================
# State Models
# ============================================

class FlowState(BaseModel):
    """State for Claude Flow orchestration."""
    flow_id: str
    current_step: int = 0
    messages: list[dict] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    results: list[dict] = Field(default_factory=list)
    status: str = "running"  # running, completed, failed
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class AgentProfile(BaseModel):
    """Profile for a Claude agent in the flow."""
    name: str
    persona: str
    capabilities: list[str] = Field(default_factory=list)
    model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.7


class FlowStep(BaseModel):
    """A step in the flow execution."""
    agent: str
    instruction: str
    tools: list[str] = Field(default_factory=list)
    pass_context: bool = True
    output_key: Optional[str] = None


# ============================================
# Native Claude Orchestrator
# ============================================

@dataclass
class ClaudeFlowOrchestrator:
    """
    Native Claude multi-agent orchestrator.
    
    Uses Claude's tool use capability to coordinate multiple
    agent personas without external frameworks.
    """
    
    agents: dict[str, AgentProfile] = field(default_factory=dict)
    flows: dict[str, list[FlowStep]] = field(default_factory=dict)
    tools: dict[str, Callable] = field(default_factory=dict)
    
    def register_agent(self, profile: AgentProfile) -> None:
        """Register an agent profile."""
        self.agents[profile.name] = profile
        logger.info("agent_registered", name=profile.name)
    
    def register_tool(self, name: str, func: Callable, description: str = "") -> None:
        """Register a tool for agent use."""
        self.tools[name] = func
        logger.info("tool_registered", name=name)
    
    def define_flow(self, flow_name: str, steps: list[FlowStep]) -> None:
        """Define a multi-step flow."""
        # Validate agents exist
        for step in steps:
            if step.agent not in self.agents:
                raise ValueError(f"Agent {step.agent} not registered")
        
        self.flows[flow_name] = steps
        logger.info("flow_defined", name=flow_name, steps=len(steps))
    
    async def _execute_agent_step(
        self,
        agent: AgentProfile,
        instruction: str,
        context: dict[str, Any],
        available_tools: list[str],
    ) -> dict[str, Any]:
        """Execute a single agent step."""
        from core.llm_gateway import LLMGateway, Message, ModelConfig, Provider
        
        gateway = LLMGateway()
        
        # Build system prompt with persona
        system_prompt = f"""You are {agent.name}.

{agent.persona}

Your capabilities include: {', '.join(agent.capabilities)}

{f"Available context: {json.dumps(context, indent=2)}" if context else ""}

Respond thoughtfully and stay in character."""
        
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=instruction),
        ]
        
        provider = Provider.OPENAI if "gpt" in agent.model.lower() else Provider.ANTHROPIC
        config = ModelConfig(
            provider=provider,
            model_id=agent.model,
            temperature=agent.temperature,
        )
        
        response = await gateway.complete(messages, config)
        
        return {
            "agent": agent.name,
            "response": response.content,
            "model": response.model,
            "usage": response.usage,
        }
    
    async def run_flow(
        self,
        flow_name: str,
        initial_input: str,
        initial_context: Optional[dict] = None,
    ) -> FlowState:
        """
        Execute a defined flow.
        
        Args:
            flow_name: Name of the flow to execute
            initial_input: Starting input/task
            initial_context: Optional initial context
        
        Returns:
            Final flow state with all results
        """
        if flow_name not in self.flows:
            raise ValueError(f"Flow {flow_name} not defined")
        
        steps = self.flows[flow_name]
        
        state = FlowState(
            flow_id=f"{flow_name}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            context=initial_context or {},
        )
        
        state.messages.append({
            "role": "user",
            "content": initial_input,
        })
        
        logger.info("flow_started", flow_id=state.flow_id, flow_name=flow_name)
        
        try:
            for i, step in enumerate(steps):
                state.current_step = i
                
                agent = self.agents[step.agent]
                
                # Build instruction with context if needed
                instruction = step.instruction
                if step.pass_context and state.results:
                    last_result = state.results[-1]
                    instruction = f"Previous result from {last_result['agent']}:\n{last_result['response']}\n\nYour task: {instruction}"
                elif "{input}" in instruction:
                    instruction = instruction.replace("{input}", initial_input)
                
                logger.info(
                    "step_executing",
                    flow_id=state.flow_id,
                    step=i,
                    agent=agent.name,
                )
                
                result = await self._execute_agent_step(
                    agent=agent,
                    instruction=instruction,
                    context=state.context if step.pass_context else {},
                    available_tools=step.tools,
                )
                
                state.results.append(result)
                state.messages.append({
                    "role": "assistant",
                    "name": agent.name,
                    "content": result["response"],
                })
                
                # Store output in context if key specified
                if step.output_key:
                    state.context[step.output_key] = result["response"]
            
            state.status = "completed"
            logger.info("flow_completed", flow_id=state.flow_id, steps=len(steps))
            
        except Exception as e:
            state.status = "failed"
            state.context["error"] = str(e)
            logger.error("flow_failed", flow_id=state.flow_id, error=str(e))
        
        return state
    
    async def run_conversation(
        self,
        agents: list[str],
        topic: str,
        turns: int = 5,
    ) -> FlowState:
        """
        Run a multi-agent conversation.
        
        Agents take turns discussing a topic.
        """
        state = FlowState(
            flow_id=f"conversation-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        )
        
        state.messages.append({
            "role": "user",
            "content": f"Topic for discussion: {topic}",
        })
        
        logger.info("conversation_started", flow_id=state.flow_id, agents=agents)
        
        try:
            for turn in range(turns):
                for agent_name in agents:
                    if agent_name not in self.agents:
                        raise ValueError(f"Agent {agent_name} not registered")
                    
                    agent = self.agents[agent_name]
                    
                    # Build conversation context
                    conv_history = "\n".join([
                        f"{m.get('name', m['role'])}: {m['content']}"
                        for m in state.messages
                    ])
                    
                    instruction = f"""The conversation so far:
{conv_history}

Continue the conversation naturally. Add your perspective on the topic.
Keep your response focused and under 200 words."""
                    
                    result = await self._execute_agent_step(
                        agent=agent,
                        instruction=instruction,
                        context={},
                        available_tools=[],
                    )
                    
                    state.results.append(result)
                    state.messages.append({
                        "role": "assistant",
                        "name": agent.name,
                        "content": result["response"],
                    })
                
                state.current_step = turn + 1
            
            state.status = "completed"
            logger.info("conversation_completed", flow_id=state.flow_id, turns=turns)
            
        except Exception as e:
            state.status = "failed"
            state.context["error"] = str(e)
            logger.error("conversation_failed", flow_id=state.flow_id, error=str(e))
        
        return state


# ============================================
# Pre-built Flow Templates
# ============================================

def create_research_flow() -> ClaudeFlowOrchestrator:
    """Create a research-focused flow with specialized agents."""
    orchestrator = ClaudeFlowOrchestrator()
    
    # Register agents
    orchestrator.register_agent(AgentProfile(
        name="researcher",
        persona="You are a thorough researcher who excels at finding and synthesizing information. You approach topics systematically and cite your reasoning.",
        capabilities=["deep analysis", "information synthesis", "fact-checking"],
    ))
    
    orchestrator.register_agent(AgentProfile(
        name="analyst",
        persona="You are a critical analyst who evaluates information for accuracy, bias, and completeness. You identify gaps and inconsistencies.",
        capabilities=["critical analysis", "bias detection", "gap identification"],
    ))
    
    orchestrator.register_agent(AgentProfile(
        name="writer",
        persona="You are a skilled technical writer who transforms complex research into clear, actionable content. You focus on clarity and structure.",
        capabilities=["technical writing", "summarization", "document structure"],
    ))
    
    # Define research flow
    orchestrator.define_flow("research", [
        FlowStep(
            agent="researcher",
            instruction="Research the following topic thoroughly: {input}",
            output_key="research",
        ),
        FlowStep(
            agent="analyst",
            instruction="Analyze the research for completeness, accuracy, and any gaps that need addressing.",
            output_key="analysis",
        ),
        FlowStep(
            agent="writer",
            instruction="Create a well-structured summary combining the research and analysis. Include key findings, conclusions, and recommendations.",
            output_key="summary",
        ),
    ])
    
    return orchestrator


def create_code_review_flow() -> ClaudeFlowOrchestrator:
    """Create a code review flow with specialized agents."""
    orchestrator = ClaudeFlowOrchestrator()
    
    orchestrator.register_agent(AgentProfile(
        name="security_auditor",
        persona="You are a security expert who identifies vulnerabilities, unsafe patterns, and potential exploits in code.",
        capabilities=["security analysis", "vulnerability detection", "threat modeling"],
    ))
    
    orchestrator.register_agent(AgentProfile(
        name="performance_analyst",
        persona="You are a performance expert who identifies inefficiencies, memory issues, and optimization opportunities.",
        capabilities=["performance analysis", "optimization", "complexity analysis"],
    ))
    
    orchestrator.register_agent(AgentProfile(
        name="code_reviewer",
        persona="You are a senior developer who reviews code for best practices, maintainability, and adherence to standards.",
        capabilities=["code review", "best practices", "design patterns"],
    ))
    
    orchestrator.define_flow("code_review", [
        FlowStep(
            agent="security_auditor",
            instruction="Perform a security audit of the following code: {input}",
            output_key="security_findings",
        ),
        FlowStep(
            agent="performance_analyst",
            instruction="Analyze the code for performance issues and optimization opportunities.",
            output_key="performance_findings",
        ),
        FlowStep(
            agent="code_reviewer",
            instruction="Provide a comprehensive code review incorporating the security and performance findings. Give actionable recommendations.",
            output_key="review_summary",
        ),
    ])
    
    return orchestrator


if __name__ == "__main__":
    import asyncio
    
    async def main():
        """Test Claude Flow orchestration."""
        print("Claude Flow Module")
        print("-" * 40)
        print("Available templates:")
        print("  - create_research_flow()")
        print("  - create_code_review_flow()")
        print("\nExample usage:")
        print("  orchestrator = create_research_flow()")
        print("  result = await orchestrator.run_flow('research', 'AI safety')")
    
    asyncio.run(main())
```

---

## Step 5: Create CrewAI Manager

Create `core/orchestration/crew_manager.py`:

```python
#!/usr/bin/env python3
"""
CrewAI Team Management
Role-based agent teams for collaborative AI workflows.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional
from dataclasses import dataclass, field

import structlog
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment
load_dotenv()

logger = structlog.get_logger(__name__)

# Import CrewAI
try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    logger.warning("crewai not available - install with: pip install crewai")


# ============================================
# Custom Tools for CrewAI
# ============================================

if CREWAI_AVAILABLE:
    
    class LLMGatewayTool(BaseTool):
        """Tool that uses the unified LLM Gateway."""
        name: str = "llm_complete"
        description: str = "Execute an LLM completion using the unified gateway"
        
        def _run(self, prompt: str) -> str:
            """Synchronous execution."""
            try:
                from core.llm_gateway import LLMGateway, Message
                
                gateway = LLMGateway()
                response = gateway.complete_sync(
                    messages=[Message(role="user", content=prompt)]
                )
                return response.content
            except Exception as e:
                return f"Error: {str(e)}"


# ============================================
# Configuration Models
# ============================================

class CrewMemberConfig(BaseModel):
    """Configuration for a crew member."""
    role: str
    goal: str
    backstory: str
    tools: list[str] = Field(default_factory=list)
    verbose: bool = True
    allow_delegation: bool = False
    llm_model: str = "claude-3-5-sonnet-20241022"


class CrewTaskConfig(BaseModel):
    """Configuration for a crew task."""
    description: str
    expected_output: str
    agent_role: str  # Maps to CrewMemberConfig.role
    context_from: list[str] = Field(default_factory=list)  # Task descriptions for context


class CrewConfig(BaseModel):
    """Configuration for entire crew."""
    name: str
    process: str = "sequential"  # sequential or hierarchical
    verbose: bool = True
    members: list[CrewMemberConfig]
    tasks: list[CrewTaskConfig]


# ============================================
# Crew Manager
# ============================================

@dataclass
class CrewManager:
    """
    Manager for CrewAI team orchestration.
    
    Creates and manages role-based agent crews.
    """
    
    crews: dict[str, Any] = field(default_factory=dict)
    agents: dict[str, Any] = field(default_factory=dict)
    custom_tools: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize with default tools."""
        if CREWAI_AVAILABLE:
            self.custom_tools["llm_gateway"] = LLMGatewayTool()
    
    def _get_llm(self, model: str):
        """Get LLM configuration for CrewAI."""
        # CrewAI uses langchain LLMs - we configure via env vars
        # The actual LLM is configured through OPENAI_API_KEY or ANTHROPIC_API_KEY
        return None  # Uses default from environment
    
    def _create_agent(self, config: CrewMemberConfig) -> Any:
        """Create a CrewAI agent from config."""
        if not CREWAI_AVAILABLE:
            raise ImportError("crewai not installed")
        
        # Collect tools for this agent
        agent_tools = []
        for tool_name in config.tools:
            if tool_name in self.custom_tools:
                agent_tools.append(self.custom_tools[tool_name])
        
        agent = Agent(
            role=config.role,
            goal=config.goal,
            backstory=config.backstory,
            tools=agent_tools,
            verbose=config.verbose,
            allow_delegation=config.allow_delegation,
        )
        
        self.agents[config.role] = agent
        logger.info("agent_created", role=config.role)
        
        return agent
    
    def _create_task(self, config: CrewTaskConfig, context_tasks: list = None) -> Any:
        """Create a CrewAI task from config."""
        if not CREWAI_AVAILABLE:
            raise ImportError("crewai not installed")
        
        if config.agent_role not in self.agents:
            raise ValueError(f"Agent with role {config.agent_role} not found")
        
        agent = self.agents[config.agent_role]
        
        task = Task(
            description=config.description,
            expected_output=config.expected_output,
            agent=agent,
            context=context_tasks or [],
        )
        
        logger.info("task_created", description=config.description[:50], agent=config.agent_role)
        
        return task
    
    def create_crew(self, config: CrewConfig) -> Any:
        """
        Create a complete crew from configuration.
        
        Args:
            config: Complete crew configuration
        
        Returns:
            Configured CrewAI Crew object
        """
        if not CREWAI_AVAILABLE:
            raise ImportError("crewai not installed")
        
        # Create all agents first
        for member_config in config.members:
            self._create_agent(member_config)
        
        # Create tasks with context linking
        tasks = []
        task_map = {}  # Map description to task for context
        
        for task_config in config.tasks:
            # Collect context tasks
            context_tasks = [
                task_map[desc]
                for desc in task_config.context_from
                if desc in task_map
            ]
            
            task = self._create_task(task_config, context_tasks)
            tasks.append(task)
            task_map[task_config.description] = task
        
        # Determine process type
        process = Process.sequential if config.process == "sequential" else Process.hierarchical
        
        # Create the crew
        crew = Crew(
            agents=list(self.agents.values()),
            tasks=tasks,
            process=process,
            verbose=config.verbose,
        )
        
        self.crews[config.name] = crew
        logger.info("crew_created", name=config.name, members=len(config.members), tasks=len(tasks))
        
        return crew
    
    def run_crew(self, crew_name: str, inputs: Optional[dict] = None) -> Any:
        """
        Execute a crew's workflow.
        
        Args:
            crew_name: Name of the crew to run
            inputs: Optional inputs for the crew
        
        Returns:
            Crew execution result
        """
        if crew_name not in self.crews:
            raise ValueError(f"Crew {crew_name} not found")
        
        crew = self.crews[crew_name]
        
        logger.info("crew_starting", name=crew_name)
        result = crew.kickoff(inputs=inputs or {})
        logger.info("crew_completed", name=crew_name)
        
        return result


# ============================================
# Pre-built Crew Templates
# ============================================

def create_content_crew() -> tuple[CrewManager, str]:
    """
    Create a content creation crew.
    
    Returns manager and crew name.
    """
    manager = CrewManager()
    
    config = CrewConfig(
        name="content_creators",
        process="sequential",
        members=[
            CrewMemberConfig(
                role="Content Researcher",
                goal="Research topics thoroughly and gather key information",
                backstory="You are an expert researcher with years of experience finding and synthesizing information from multiple sources.",
            ),
            CrewMemberConfig(
                role="Content Writer",
                goal="Write engaging, well-structured content based on research",
                backstory="You are a professional writer skilled at transforming research into compelling narratives.",
            ),
            CrewMemberConfig(
                role="Content Editor",
                goal="Polish content for clarity, grammar, and engagement",
                backstory="You are a meticulous editor with an eye for detail and a passion for perfect prose.",
            ),
        ],
        tasks=[
            CrewTaskConfig(
                description="Research the topic: {topic}. Gather key facts, statistics, and insights.",
                expected_output="A comprehensive research document with organized findings",
                agent_role="Content Researcher",
            ),
            CrewTaskConfig(
                description="Write a compelling article based on the research findings.",
                expected_output="A well-structured article draft with clear sections",
                agent_role="Content Writer",
                context_from=["Research the topic: {topic}. Gather key facts, statistics, and insights."],
            ),
            CrewTaskConfig(
                description="Edit the article for clarity, grammar, and engagement. Ensure it flows well.",
                expected_output="A polished, publication-ready article",
                agent_role="Content Editor",
                context_from=["Write a compelling article based on the research findings."],
            ),
        ],
    )
    
    manager.create_crew(config)
    return manager, "content_creators"


def create_development_crew() -> tuple[CrewManager, str]:
    """
    Create a software development crew.
    
    Returns manager and crew name.
    """
    manager = CrewManager()
    
    config = CrewConfig(
        name="dev_team",
        process="sequential",
        members=[
            CrewMemberConfig(
                role="Technical Architect",
                goal="Design robust, scalable software architectures",
                backstory="You are a senior architect with experience designing systems that scale to millions of users.",
                tools=["llm_gateway"],
            ),
            CrewMemberConfig(
                role="Senior Developer",
                goal="Implement clean, efficient code following best practices",
                backstory="You are a 10x developer who writes elegant, maintainable code.",
                tools=["llm_gateway"],
            ),
            CrewMemberConfig(
                role="QA Engineer",
                goal="Ensure code quality through thorough testing and review",
                backstory="You are a quality-obsessed engineer who catches bugs before they reach production.",
            ),
        ],
        tasks=[
            CrewTaskConfig(
                description="Design the architecture for: {feature}. Include components, interfaces, and data flow.",
                expected_output="A detailed architecture document with diagrams and specifications",
                agent_role="Technical Architect",
            ),
            CrewTaskConfig(
                description="Implement the feature based on the architecture. Write clean, documented code.",
                expected_output="Working code implementation with inline documentation",
                agent_role="Senior Developer",
                context_from=["Design the architecture for: {feature}. Include components, interfaces, and data flow."],
            ),
            CrewTaskConfig(
                description="Review the implementation. Write test cases and identify potential issues.",
                expected_output="Test suite and code review report with recommendations",
                agent_role="QA Engineer",
                context_from=["Implement the feature based on the architecture. Write clean, documented code."],
            ),
        ],
    )
    
    manager.create_crew(config)
    return manager, "dev_team"


if __name__ == "__main__":
    print("CrewAI Manager Module")
    print("-" * 40)
    
    if not CREWAI_AVAILABLE:
        print("CrewAI not installed. Install with: pip install crewai")
    else:
        print("Available templates:")
        print("  - create_content_crew()")
        print("  - create_development_crew()")
        print("\nExample usage:")
        print("  manager, crew_name = create_content_crew()")
        print("  result = manager.run_crew(crew_name, {'topic': 'AI agents'})")
```

---

## Step 6: Create AutoGen Agents

Create `core/orchestration/autogen_agents.py`:

```python
#!/usr/bin/env python3
"""
AutoGen Conversational Agents
Microsoft's framework for multi-agent conversations.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Union
from dataclasses import dataclass, field

import structlog
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment
load_dotenv()

logger = structlog.get_logger(__name__)

# Import AutoGen
try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    logger.warning("autogen not available - install with: pip install pyautogen")


# ============================================
# Configuration Models
# ============================================

class AutoGenAgentConfig(BaseModel):
    """Configuration for an AutoGen agent."""
    name: str
    system_message: str
    is_user_proxy: bool = False
    human_input_mode: str = "NEVER"  # ALWAYS, TERMINATE, NEVER
    max_consecutive_auto_reply: int = 10
    code_execution_config: Optional[dict] = None


class AutoGenGroupConfig(BaseModel):
    """Configuration for AutoGen group chat."""
    name: str
    agents: list[str]  # Agent names
    max_round: int = 12
    admin_name: str = "Admin"
    speaker_selection_method: str = "auto"  # auto, round_robin, random, manual


class LLMConfig(BaseModel):
    """LLM configuration for AutoGen."""
    model: str = "claude-3-5-sonnet-20241022"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_type: str = "anthropic"  # openai, anthropic, azure
    temperature: float = 0.7
    max_tokens: int = 4096


# ============================================
# AutoGen Manager
# ============================================

@dataclass
class AutoGenManager:
    """
    Manager for AutoGen multi-agent orchestration.
    
    Creates and manages conversational agent groups.
    """
    
    agents: dict[str, Any] = field(default_factory=dict)
    groups: dict[str, Any] = field(default_factory=dict)
    llm_config: Optional[dict] = field(default=None)
    
    def __post_init__(self):
        """Initialize LLM configuration."""
        self._setup_llm_config()
    
    def _setup_llm_config(self) -> None:
        """Configure LLM for AutoGen agents."""
        import os
        
        # Try Anthropic first, then OpenAI
        if os.getenv("ANTHROPIC_API_KEY"):
            self.llm_config = {
                "config_list": [{
                    "model": "claude-3-5-sonnet-20241022",
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                    "api_type": "anthropic",
                }],
                "temperature": 0.7,
            }
            logger.info("autogen_configured", provider="anthropic")
        elif os.getenv("OPENAI_API_KEY"):
            self.llm_config = {
                "config_list": [{
                    "model": "gpt-4o",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                }],
                "temperature": 0.7,
            }
            logger.info("autogen_configured", provider="openai")
        else:
            logger.warning("no_api_key_configured")
            self.llm_config = None
    
    def create_agent(self, config: AutoGenAgentConfig) -> Any:
        """
        Create an AutoGen agent from configuration.
        
        Args:
            config: Agent configuration
        
        Returns:
            AutoGen agent instance
        """
        if not AUTOGEN_AVAILABLE:
            raise ImportError("autogen not installed")
        
        if config.is_user_proxy:
            agent = UserProxyAgent(
                name=config.name,
                system_message=config.system_message,
                human_input_mode=config.human_input_mode,
                max_consecutive_auto_reply=config.max_consecutive_auto_reply,
                code_execution_config=config.code_execution_config or False,
            )
        else:
            agent = AssistantAgent(
                name=config.name,
                system_message=config.system_message,
                llm_config=self.llm_config,
                max_consecutive_auto_reply=config.max_consecutive_auto_reply,
            )
        
        self.agents[config.name] = agent
        logger.info("agent_created", name=config.name, type="user_proxy" if config.is_user_proxy else "assistant")
        
        return agent
    
    def create_group(self, config: AutoGenGroupConfig) -> Any:
        """
        Create an AutoGen group chat.
        
        Args:
            config: Group configuration
        
        Returns:
            GroupChatManager instance
        """
        if not AUTOGEN_AVAILABLE:
            raise ImportError("autogen not installed")
        
        # Collect agents for the group
        group_agents = []
        for agent_name in config.agents:
            if agent_name not in self.agents:
                raise ValueError(f"Agent {agent_name} not found")
            group_agents.append(self.agents[agent_name])
        
        # Create group chat
        group_chat = GroupChat(
            agents=group_agents,
            messages=[],
            max_round=config.max_round,
            admin_name=config.admin_name,
            speaker_selection_method=config.speaker_selection_method,
        )
        
        # Create manager
        manager = GroupChatManager(
            groupchat=group_chat,
            llm_config=self.llm_config,
        )
        
        self.groups[config.name] = {
            "chat": group_chat,
            "manager": manager,
        }
        
        logger.info("group_created", name=config.name, agents=config.agents)
        
        return manager
    
    def run_two_agent_chat(
        self,
        initiator_name: str,
        responder_name: str,
        message: str,
        max_turns: int = 10,
    ) -> list[dict]:
        """
        Run a two-agent conversation.
        
        Args:
            initiator_name: Name of the agent starting the conversation
            responder_name: Name of the responding agent
            message: Initial message
            max_turns: Maximum conversation turns
        
        Returns:
            List of conversation messages
        """
        if initiator_name not in self.agents:
            raise ValueError(f"Agent {initiator_name} not found")
        if responder_name not in self.agents:
            raise ValueError(f"Agent {responder_name} not found")
        
        initiator = self.agents[initiator_name]
        responder = self.agents[responder_name]
        
        logger.info("two_agent_chat_starting", initiator=initiator_name, responder=responder_name)
        
        # Start conversation
        initiator.initiate_chat(
            responder,
            message=message,
            max_turns=max_turns,
        )
        
        # Collect messages
        messages = responder.chat_messages.get(initiator, [])
        
        logger.info("two_agent_chat_completed", message_count=len(messages))
        
        return messages
    
    def run_group_chat(
        self,
        group_name: str,
        message: str,
        sender_name: Optional[str] = None,
    ) -> list[dict]:
        """
        Run a group conversation.
        
        Args:
            group_name: Name of the group
            message: Initial message
            sender_name: Name of the agent initiating (defaults to first agent)
        
        Returns:
            List of conversation messages
        """
        if group_name not in self.groups:
            raise ValueError(f"Group {group_name} not found")
        
        group_info = self.groups[group_name]
        manager = group_info["manager"]
        chat = group_info["chat"]
        
        # Determine sender
        if sender_name:
            if sender_name not in self.agents:
                raise ValueError(f"Agent {sender_name} not found")
            sender = self.agents[sender_name]
        else:
            sender = chat.agents[0]
        
        logger.info("group_chat_starting", group=group_name, sender=sender.name)
        
        # Start group chat
        sender.initiate_chat(
            manager,
            message=message,
        )
        
        # Return messages from the group chat
        messages = chat.messages
        
        logger.info("group_chat_completed", group=group_name, message_count=len(messages))
        
        return messages


# ============================================
# Pre-built Agent Templates
# ============================================

def create_coding_pair() -> AutoGenManager:
    """
    Create a coding pair (programmer + reviewer).
    
    Returns configured AutoGenManager.
    """
    manager = AutoGenManager()
    
    # Create programmer agent
    manager.create_agent(AutoGenAgentConfig(
        name="programmer",
        system_message="""You are an expert programmer. Write clean, efficient, well-documented code.
When given a task:
1. Understand the requirements
2. Plan your approach
3. Write the code with comments
4. Explain your implementation

Always consider edge cases and error handling.""",
    ))
    
    # Create reviewer agent
    manager.create_agent(AutoGenAgentConfig(
        name="reviewer",
        system_message="""You are a code reviewer. Your job is to:
1. Review code for correctness and best practices
2. Identify potential bugs or issues
3. Suggest improvements
4. Approve code when it meets standards

Be constructive but thorough. Say 'APPROVED' when code is ready.""",
    ))
    
    # Create user proxy for termination
    manager.create_agent(AutoGenAgentConfig(
        name="user",
        system_message="You represent the user.",
        is_user_proxy=True,
        human_input_mode="NEVER",
    ))
    
    return manager


def create_research_team() -> tuple[AutoGenManager, str]:
    """
    Create a research team with group chat.
    
    Returns manager and group name.
    """
    manager = AutoGenManager()
    
    # Create researcher
    manager.create_agent(AutoGenAgentConfig(
        name="researcher",
        system_message="""You are a research specialist. Your job is to:
1. Gather information on topics
2. Synthesize findings from multiple angles
3. Present research clearly
4. Identify gaps in knowledge""",
    ))
    
    # Create analyst
    manager.create_agent(AutoGenAgentConfig(
        name="analyst",
        system_message="""You are a data analyst. Your job is to:
1. Analyze research findings
2. Identify patterns and trends
3. Draw data-driven conclusions
4. Challenge assumptions with evidence""",
    ))
    
    # Create writer
    manager.create_agent(AutoGenAgentConfig(
        name="writer",
        system_message="""You are a technical writer. Your job is to:
1. Compile research and analysis into clear reports
2. Structure information logically
3. Make complex topics accessible
4. Create executive summaries""",
    ))
    
    # Create coordinator
    manager.create_agent(AutoGenAgentConfig(
        name="coordinator",
        system_message="""You are the team coordinator. Your job is to:
1. Ensure the team stays on topic
2. Summarize progress
3. Identify when objectives are met
4. Say 'TASK COMPLETE' when the research is finished""",
        is_user_proxy=True,
        human_input_mode="NEVER",
    ))
    
    # Create group
    group_config = AutoGenGroupConfig(
        name="research_team",
        agents=["researcher", "analyst", "writer", "coordinator"],
        max_round=15,
        speaker_selection_method="auto",
    )
    
    manager.create_group(group_config)
    
    return manager, "research_team"


def create_debate_team() -> tuple[AutoGenManager, str]:
    """
    Create a debate team for exploring topics.
    
    Returns manager and group name.
    """
    manager = AutoGenManager()
    
    # Create advocate
    manager.create_agent(AutoGenAgentConfig(
        name="advocate",
        system_message="""You argue IN FAVOR of propositions. Your role is to:
1. Present strong supporting arguments
2. Provide evidence and examples
3. Counter opposing arguments
4. Maintain logical consistency""",
    ))
    
    # Create opponent
    manager.create_agent(AutoGenAgentConfig(
        name="opponent",
        system_message="""You argue AGAINST propositions. Your role is to:
1. Present strong counterarguments
2. Question assumptions and evidence
3. Identify weaknesses in supporting arguments
4. Play devil's advocate constructively""",
    ))
    
    # Create judge
    manager.create_agent(AutoGenAgentConfig(
        name="judge",
        system_message="""You are an impartial judge. Your role is to:
1. Evaluate arguments from both sides
2. Identify the strongest points
3. Determine the winner based on logic and evidence
4. Provide a balanced summary

Say 'DEBATE CONCLUDED' when you've reached a verdict.""",
        is_user_proxy=True,
        human_input_mode="NEVER",
    ))
    
    # Create group
    group_config = AutoGenGroupConfig(
        name="debate_team",
        agents=["advocate", "opponent", "judge"],
        max_round=10,
        speaker_selection_method="round_robin",
    )
    
    manager.create_group(group_config)
    
    return manager, "debate_team"


if __name__ == "__main__":
    print("AutoGen Agents Module")
    print("-" * 40)
    
    if not AUTOGEN_AVAILABLE:
        print("AutoGen not installed. Install with: pip install pyautogen")
    else:
        print("Available templates:")
        print("  - create_coding_pair()")
        print("  - create_research_team()")
        print("  - create_debate_team()")
        print("\nExample usage:")
        print("  manager = create_coding_pair()")
        print("  messages = manager.run_two_agent_chat('programmer', 'reviewer', 'Write a function to...')")
```

---

## Step 7: Create Unified Interface

Create `core/orchestration/__init__.py`:

```python
#!/usr/bin/env python3
"""
Unified Orchestration Interface
Layer 1 - Multi-Agent Orchestration across 5 frameworks.
"""

from __future__ import annotations

from typing import Any, Optional
from enum import Enum
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


# ============================================
# Framework Availability
# ============================================

class OrchestrationFramework(str, Enum):
    """Available orchestration frameworks."""
    TEMPORAL = "temporal"
    LANGGRAPH = "langgraph"
    CLAUDE_FLOW = "claude_flow"
    CREWAI = "crewai"
    AUTOGEN = "autogen"


def get_available_frameworks() -> dict[str, bool]:
    """Check which frameworks are available."""
    availability = {}
    
    try:
        from temporalio import workflow
        availability["temporal"] = True
    except ImportError:
        availability["temporal"] = False
    
    try:
        from langgraph.graph import StateGraph
        availability["langgraph"] = True
    except ImportError:
        availability["langgraph"] = False
    
    # Claude Flow is always available (native implementation)
    availability["claude_flow"] = True
    
    try:
        from crewai import Crew
        availability["crewai"] = True
    except ImportError:
        availability["crewai"] = False
    
    try:
        import autogen
        availability["autogen"] = True
    except ImportError:
        availability["autogen"] = False
    
    return availability


# ============================================
# Import Framework Components
# ============================================

# Temporal
try:
    from core.orchestration.temporal_workflows import (
        TemporalOrchestrator,
        WorkflowInput,
        WorkflowOutput,
        SimpleAgentWorkflow,
        ChainedAgentWorkflow,
        run_temporal_worker,
        TEMPORAL_AVAILABLE,
    )
except ImportError:
    TEMPORAL_AVAILABLE = False
    TemporalOrchestrator = None
    WorkflowInput = None
    WorkflowOutput = None
    SimpleAgentWorkflow = None
    ChainedAgentWorkflow = None
    run_temporal_worker = None

# LangGraph
try:
    from core.orchestration.langgraph_agents import (
        LangGraphOrchestrator,
        AgentConfig,
        AgentRole,
        AgentState,
        create_planning_execution_graph,
        create_debate_graph,
        LANGGRAPH_AVAILABLE,
    )
except ImportError:
    LANGGRAPH_AVAILABLE = False
    LangGraphOrchestrator = None
    AgentConfig = None
    AgentRole = None
    AgentState = None
    create_planning_execution_graph = None
    create_debate_graph = None

# Claude Flow
try:
    from core.orchestration.claude_flow import (
        ClaudeFlowOrchestrator,
        FlowState,
        AgentProfile,
        FlowStep,
        create_research_flow,
        create_code_review_flow,
    )
    CLAUDE_FLOW_AVAILABLE = True
except ImportError:
    CLAUDE_FLOW_AVAILABLE = False
    ClaudeFlowOrchestrator = None
    FlowState = None
    AgentProfile = None
    FlowStep = None
    create_research_flow = None
    create_code_review_flow = None

# CrewAI
try:
    from core.orchestration.crew_manager import (
        CrewManager,
        CrewConfig,
        CrewMemberConfig,
        CrewTaskConfig,
        create_content_crew,
        create_development_crew,
        CREWAI_AVAILABLE,
    )
except ImportError:
    CREWAI_AVAILABLE = False
    CrewManager = None
    CrewConfig = None
    CrewMemberConfig = None
    CrewTaskConfig = None
    create_content_crew = None
    create_development_crew = None

# AutoGen
try:
    from core.orchestration.autogen_agents import (
        AutoGenManager,
        AutoGenAgentConfig,
        AutoGenGroupConfig,
        create_coding_pair,
        create_research_team,
        create_debate_team,
        AUTOGEN_AVAILABLE,
    )
except ImportError:
    AUTOGEN_AVAILABLE = False
    AutoGenManager = None
    AutoGenAgentConfig = None
    AutoGenGroupConfig = None
    create_coding_pair = None
    create_research_team = None
    create_debate_team = None


# ============================================
# Unified Orchestration Interface
# ============================================

@dataclass
class UnifiedOrchestrator:
    """
    Unified interface for all orchestration frameworks.
    
    Provides a consistent API to work with any supported framework.
    """
    
    def get_status(self) -> dict[str, Any]:
        """Get status of all orchestration frameworks."""
        return {
            "available_frameworks": get_available_frameworks(),
            "temporal": {
                "available": TEMPORAL_AVAILABLE,
                "features": ["durable workflows", "activity retries", "timers"],
            },
            "langgraph": {
                "available": LANGGRAPH_AVAILABLE,
                "features": ["graph-based agents", "state management", "conditional routing"],
            },
            "claude_flow": {
                "available": CLAUDE_FLOW_AVAILABLE,
                "features": ["native Claude", "multi-agent", "conversation flows"],
            },
            "crewai": {
                "available": CREWAI_AVAILABLE,
                "features": ["role-based teams", "task delegation", "sequential/hierarchical"],
            },
            "autogen": {
                "available": AUTOGEN_AVAILABLE,
                "features": ["conversational agents", "group chat", "code execution"],
            },
        }
    
    def get_orchestrator(self, framework: OrchestrationFramework) -> Any:
        """
        Get an orchestrator for the specified framework.
        
        Args:
            framework: Which framework to use
        
        Returns:
            Framework-specific orchestrator instance
        """
        if framework == OrchestrationFramework.TEMPORAL:
            if not TEMPORAL_AVAILABLE:
                raise ImportError("temporalio not installed")
            return TemporalOrchestrator()
        
        elif framework == OrchestrationFramework.LANGGRAPH:
            if not LANGGRAPH_AVAILABLE:
                raise ImportError("langgraph not installed")
            return LangGraphOrchestrator()
        
        elif framework == OrchestrationFramework.CLAUDE_FLOW:
            if not CLAUDE_FLOW_AVAILABLE:
                raise ImportError("claude_flow not available")
            return ClaudeFlowOrchestrator()
        
        elif framework == OrchestrationFramework.CREWAI:
            if not CREWAI_AVAILABLE:
                raise ImportError("crewai not installed")
            return CrewManager()
        
        elif framework == OrchestrationFramework.AUTOGEN:
            if not AUTOGEN_AVAILABLE:
                raise ImportError("autogen not installed")
            return AutoGenManager()
        
        else:
            raise ValueError(f"Unknown framework: {framework}")
    
    def recommend_framework(self, use_case: str) -> OrchestrationFramework:
        """
        Recommend a framework based on use case.
        
        Args:
            use_case: Description of the use case
        
        Returns:
            Recommended framework
        """
        use_case_lower = use_case.lower()
        
        # Check for keywords
        if any(kw in use_case_lower for kw in ["durable", "reliable", "retry", "long-running"]):
            if TEMPORAL_AVAILABLE:
                return OrchestrationFramework.TEMPORAL
        
        if any(kw in use_case_lower for kw in ["graph", "state machine", "conditional", "branching"]):
            if LANGGRAPH_AVAILABLE:
                return OrchestrationFramework.LANGGRAPH
        
        if any(kw in use_case_lower for kw in ["conversation", "chat", "discuss", "debate"]):
            if AUTOGEN_AVAILABLE:
                return OrchestrationFramework.AUTOGEN
        
        if any(kw in use_case_lower for kw in ["team", "role", "crew", "collaborate"]):
            if CREWAI_AVAILABLE:
                return OrchestrationFramework.CREWAI
        
        # Default to Claude Flow (always available)
        return OrchestrationFramework.CLAUDE_FLOW


# ============================================
# Module Exports
# ============================================

__all__ = [
    # Enums
    "OrchestrationFramework",
    
    # Availability checks
    "get_available_frameworks",
    "TEMPORAL_AVAILABLE",
    "LANGGRAPH_AVAILABLE",
    "CLAUDE_FLOW_AVAILABLE",
    "CREWAI_AVAILABLE",
    "AUTOGEN_AVAILABLE",
    
    # Unified interface
    "UnifiedOrchestrator",
    
    # Temporal
    "TemporalOrchestrator",
    "WorkflowInput",
    "WorkflowOutput",
    "SimpleAgentWorkflow",
    "ChainedAgentWorkflow",
    "run_temporal_worker",
    
    # LangGraph
    "LangGraphOrchestrator",
    "AgentConfig",
    "AgentRole",
    "AgentState",
    "create_planning_execution_graph",
    "create_debate_graph",
    
    # Claude Flow
    "ClaudeFlowOrchestrator",
    "FlowState",
    "AgentProfile",
    "FlowStep",
    "create_research_flow",
    "create_code_review_flow",
    
    # CrewAI
    "CrewManager",
    "CrewConfig",
    "CrewMemberConfig",
    "CrewTaskConfig",
    "create_content_crew",
    "create_development_crew",
    
    # AutoGen
    "AutoGenManager",
    "AutoGenAgentConfig",
    "AutoGenGroupConfig",
    "create_coding_pair",
    "create_research_team",
    "create_debate_team",
]


if __name__ == "__main__":
    print("Orchestration Layer Status")
    print("=" * 50)
    
    orchestrator = UnifiedOrchestrator()
    status = orchestrator.get_status()
    
    for framework, available in status["available_frameworks"].items():
        status_str = "✓" if available else "✗"
        info = status[framework]
        print(f"\n{status_str} {framework.upper()}")
        if available:
            print(f"  Features: {', '.join(info['features'])}")
        else:
            print(f"  Status: Not installed")
```

---

## Step 8: Create Validation Script

Create `scripts/validate_phase3.py`:

```python
#!/usr/bin/env python3
"""
Phase 3 Orchestration Layer Validation Script
Validates all Layer 1 orchestration components.
"""

import os
import sys
from pathlib import Path
from typing import Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_sdk_import(name: str, package: str) -> Tuple[bool, str]:
    """Check if an SDK can be imported."""
    try:
        __import__(package)
        return True, f"{name} importable"
    except ImportError as e:
        return False, f"{name} not installed (optional)"


def check_module_exists(module_path: str, description: str) -> Tuple[bool, str]:
    """Check if a module file exists."""
    full_path = project_root / module_path.replace(".", "/")
    py_path = full_path.with_suffix(".py")
    
    if py_path.exists():
        return True, f"{description} exists"
    elif full_path.is_dir() and (full_path / "__init__.py").exists():
        return True, f"{description} exists"
    return False, f"{description} not found"


def check_orchestration_imports() -> Tuple[bool, str]:
    """Check if orchestration module imports work."""
    try:
        from core.orchestration import (
            UnifiedOrchestrator,
            OrchestrationFramework,
            get_available_frameworks,
        )
        return True, "Core imports work"
    except ImportError as e:
        return False, f"Import error: {e}"


def check_framework_availability() -> dict:
    """Check which frameworks are available."""
    try:
        from core.orchestration import get_available_frameworks
        return get_available_frameworks()
    except:
        return {}


def check_claude_flow() -> Tuple[bool, str]:
    """Check Claude Flow (always available)."""
    try:
        from core.orchestration.claude_flow import (
            ClaudeFlowOrchestrator,
            create_research_flow,
        )
        return True, "Claude Flow operational"
    except ImportError as e:
        return False, f"Claude Flow error: {e}"


def main():
    """Run all Phase 3 validation checks."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    # Force UTF-8 for Windows console compatibility
    import io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    console = Console(force_terminal=True)
    console.print(Panel.fit(
        "[bold blue]Phase 3: Orchestration Layer Validation[/bold blue]\n"
        "[dim]Layer 1 - 5 Orchestration SDKs[/dim]",
        border_style="blue"
    ))
    console.print()

    # SDK Checks
    sdk_table = Table(title="SDK Availability", show_header=True, header_style="bold magenta")
    sdk_table.add_column("SDK", style="cyan")
    sdk_table.add_column("Status", style="green")
    sdk_table.add_column("Details")

    sdk_checks = [
        ("Temporal", "temporalio"),
        ("LangGraph", "langgraph"),
        ("LangChain Core", "langchain_core"),
        ("CrewAI", "crewai"),
        ("AutoGen", "autogen"),
    ]

    sdk_available_count = 0
    for name, package in sdk_checks:
        passed, details = check_sdk_import(name, package)
        status = "[green]PASS[/green]" if passed else "[yellow]SKIP[/yellow]"
        sdk_table.add_row(name, status, details)
        if passed:
            sdk_available_count += 1

    console.print(sdk_table)
    console.print()

    # Module Checks
    module_table = Table(title="Orchestration Modules", show_header=True, header_style="bold magenta")
    module_table.add_column("Module", style="cyan")
    module_table.add_column("Status", style="green")
    module_table.add_column("Details")

    module_checks = [
        ("core/orchestration/__init__.py", "Unified Interface"),
        ("core/orchestration/temporal_workflows.py", "Temporal Workflows"),
        ("core/orchestration/langgraph_agents.py", "LangGraph Agents"),
        ("core/orchestration/claude_flow.py", "Claude Flow"),
        ("core/orchestration/crew_manager.py", "CrewAI Manager"),
        ("core/orchestration/autogen_agents.py", "AutoGen Agents"),
    ]

    modules_passed = True
    for module, description in module_checks:
        passed, details = check_module_exists(module, description)
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        module_table.add_row(module, status, details)
        if not passed:
            modules_passed = False

    console.print(module_table)
    console.print()

    # Functional Checks
    func_table = Table(title="Functional Validation", show_header=True, header_style="bold magenta")
    func_table.add_column("Check", style="cyan")
    func_table.add_column("Status", style="green")
    func_table.add_column("Details")

    func_checks = [
        ("Orchestration Imports", check_orchestration_imports),
        ("Claude Flow", check_claude_flow),
    ]

    func_passed = True
    for name, check_fn in func_checks:
        passed, details = check_fn()
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        func_table.add_row(name, status, details)
        if not passed:
            func_passed = False

    console.print(func_table)
    console.print()

    # Framework Availability Summary
    frameworks = check_framework_availability()
    if frameworks:
        fw_table = Table(title="Framework Status", show_header=True, header_style="bold cyan")
        fw_table.add_column("Framework", style="cyan")
        fw_table.add_column("Available", style="green")
        
        for fw, available in frameworks.items():
            status = "[green]Yes[/green]" if available else "[yellow]No[/yellow]"
            fw_table.add_row(fw, status)
        
        console.print(fw_table)
        console.print()

    # Summary
    # Claude Flow is always available, so minimum 1 framework
    all_required_passed = modules_passed and func_passed

    if all_required_passed:
        console.print(Panel.fit(
            "[bold green]Phase 3 Validation PASSED[/bold green]\n\n"
            f"Orchestration Layer (Layer 1) is operational.\n"
            f"- {sdk_available_count}/5 optional SDKs installed\n"
            f"- All module files created\n"
            f"- Claude Flow (native) always available\n\n"
            "[dim]Ready for Phase 4: Memory Layer[/dim]",
            border_style="green"
        ))
        return 0
    else:
        console.print(Panel.fit(
            "[bold red]Phase 3 Validation FAILED[/bold red]\n\n"
            "Some required checks did not pass.\n"
            "Please review the tables above and fix issues.",
            border_style="red"
        ))
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

---

## Success Criteria

After execution, verify:

- [ ] `core/orchestration/` directory exists
- [ ] `core/orchestration/__init__.py` - Unified interface
- [ ] `core/orchestration/temporal_workflows.py` - Temporal workflows
- [ ] `core/orchestration/langgraph_agents.py` - LangGraph agents
- [ ] `core/orchestration/claude_flow.py` - Claude Flow (always works)
- [ ] `core/orchestration/crew_manager.py` - CrewAI manager
- [ ] `core/orchestration/autogen_agents.py` - AutoGen agents
- [ ] `scripts/validate_phase3.py` - Validation script
- [ ] `python scripts/validate_phase3.py` passes

## Rollback

If issues occur:

```bash
# Remove orchestration directory
rm -rf core/orchestration

# Remove validation script
rm scripts/validate_phase3.py
```

## Notes

- **Claude Flow is always available** - Uses native LLM Gateway, no external deps
- Other frameworks are **optional** but recommended for production
- Each orchestrator integrates with `core/llm_gateway.py`
- All frameworks support async/await patterns
- Pre-built templates provide quick-start options

---

## Installation Commands

```bash
# Install all Layer 1 SDKs
pip install temporalio langgraph langchain-core crewai pyautogen

# Or install individually as needed
pip install temporalio      # Temporal workflows
pip install langgraph       # LangGraph agents
pip install langchain-core  # Required for LangGraph
pip install crewai          # CrewAI teams
pip install pyautogen       # AutoGen conversations
```

---

**End of Phase 3 Prompt**
