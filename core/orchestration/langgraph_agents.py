#!/usr/bin/env python3
"""
LangGraph Agent Orchestration
Graph-based multi-agent workflows with state management.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Annotated, Literal, Optional, TypedDict, Sequence
from dataclasses import dataclass, field

import structlog
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment
load_dotenv()

logger = structlog.get_logger(__name__)

# Import LangGraph SDK
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("langgraph not available - install with: pip install langgraph langchain-core")


class AgentState(TypedDict):
    """State shared between agents in the graph."""
    messages: Sequence[BaseMessage]
    current_agent: str
    task: str
    context: dict[str, Any]
    step_count: int
    max_steps: int
    final_result: Optional[str]
    error: Optional[str]


class GraphConfig(BaseModel):
    """Configuration for a LangGraph agent workflow."""
    name: str = Field(..., description="Name of the graph")
    max_steps: int = Field(default=10, ge=1, le=100)
    checkpoint: bool = Field(default=True)
    agents: list[str] = Field(default_factory=lambda: ["planner", "executor", "reviewer"])


class GraphResult(BaseModel):
    """Result from a LangGraph execution."""
    success: bool
    result: Optional[str] = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    steps_taken: int = 0
    final_agent: str = ""
    error: Optional[str] = None


def add_message(left: Sequence[BaseMessage], right: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    """Reducer function to append messages."""
    return list(left) + list(right)


if LANGGRAPH_AVAILABLE:
    class MultiAgentGraph:
        """
        Multi-agent graph workflow using LangGraph.

        Implements a Planner -> Executor -> Reviewer pattern with
        conditional routing based on agent outputs.

        Usage:
            graph = MultiAgentGraph()
            result = await graph.run("Analyze this data and create a report")
        """

        def __init__(
            self,
            config: Optional[GraphConfig] = None,
        ) -> None:
            """Initialize the multi-agent graph."""
            self.config = config or GraphConfig(name="default")
            self.checkpointer = MemorySaver() if self.config.checkpoint else None
            self.graph = self._build_graph()
            self.compiled = self.graph.compile(checkpointer=self.checkpointer)

            logger.info("langgraph_initialized", name=self.config.name)

        def _build_graph(self) -> StateGraph:
            """Build the agent graph structure."""
            graph = StateGraph(AgentState)

            # Add nodes for each agent
            graph.add_node("planner", self._planner_node)
            graph.add_node("executor", self._executor_node)
            graph.add_node("reviewer", self._reviewer_node)

            # Set entry point
            graph.set_entry_point("planner")

            # Add edges with conditional routing
            graph.add_conditional_edges(
                "planner",
                self._route_after_planner,
                {
                    "executor": "executor",
                    "end": END,
                }
            )

            graph.add_conditional_edges(
                "executor",
                self._route_after_executor,
                {
                    "reviewer": "reviewer",
                    "planner": "planner",
                    "end": END,
                }
            )

            graph.add_conditional_edges(
                "reviewer",
                self._route_after_reviewer,
                {
                    "executor": "executor",
                    "end": END,
                }
            )

            return graph

        async def _call_llm(self, prompt: str, context: dict[str, Any]) -> str:
            """Call LLM for agent reasoning."""
            try:
                from core.llm_gateway import LLMGateway, Message
                gateway = LLMGateway()

                messages = [
                    Message(role="user", content=prompt),
                ]

                response = await gateway.complete(messages)
                return response.content

            except Exception as e:
                logger.warning("llm_call_fallback", error=str(e))
                return f"Simulated response for: {prompt[:50]}"

        async def _planner_node(self, state: AgentState) -> dict[str, Any]:
            """Planner agent: breaks down tasks into steps."""
            logger.info("planner_executing", task=state["task"][:50])

            prompt = f"""You are a planning agent. Break down this task into clear steps:

Task: {state['task']}
Context: {state['context']}
Previous messages: {len(state['messages'])}

Provide a clear plan. If the task is complete, say "TASK_COMPLETE".
"""
            response = await self._call_llm(prompt, state["context"])

            return {
                "messages": [AIMessage(content=f"[Planner] {response}")],
                "current_agent": "planner",
                "step_count": state["step_count"] + 1,
            }

        async def _executor_node(self, state: AgentState) -> dict[str, Any]:
            """Executor agent: carries out planned steps."""
            logger.info("executor_executing", step=state["step_count"])

            last_message = state["messages"][-1].content if state["messages"] else ""

            prompt = f"""You are an executor agent. Execute the next step based on the plan:

Task: {state['task']}
Current Plan: {last_message}
Step: {state['step_count']}

Execute the next action. If all steps are done, say "EXECUTION_COMPLETE".
"""
            response = await self._call_llm(prompt, state["context"])

            return {
                "messages": [AIMessage(content=f"[Executor] {response}")],
                "current_agent": "executor",
                "step_count": state["step_count"] + 1,
            }

        async def _reviewer_node(self, state: AgentState) -> dict[str, Any]:
            """Reviewer agent: validates execution results."""
            logger.info("reviewer_executing", step=state["step_count"])

            execution_messages = [
                m.content for m in state["messages"]
                if hasattr(m, "content") and "[Executor]" in str(m.content)
            ]

            prompt = f"""You are a reviewer agent. Review the execution results:

Task: {state['task']}
Execution Results: {execution_messages[-3:] if execution_messages else 'None'}

Evaluate the results. If satisfactory, say "REVIEW_PASSED".
If more work needed, say "NEEDS_REVISION" and explain what.
"""
            response = await self._call_llm(prompt, state["context"])

            final_result = None
            if "REVIEW_PASSED" in response or "TASK_COMPLETE" in response:
                final_result = response

            return {
                "messages": [AIMessage(content=f"[Reviewer] {response}")],
                "current_agent": "reviewer",
                "step_count": state["step_count"] + 1,
                "final_result": final_result,
            }

        def _route_after_planner(self, state: AgentState) -> Literal["executor", "end"]:
            """Route after planner node."""
            if state["step_count"] >= state["max_steps"]:
                return "end"

            last_message = state["messages"][-1].content if state["messages"] else ""
            if "TASK_COMPLETE" in str(last_message):
                return "end"

            return "executor"

        def _route_after_executor(self, state: AgentState) -> Literal["reviewer", "planner", "end"]:
            """Route after executor node."""
            if state["step_count"] >= state["max_steps"]:
                return "end"

            last_message = state["messages"][-1].content if state["messages"] else ""

            if "EXECUTION_COMPLETE" in str(last_message):
                return "reviewer"
            elif "ERROR" in str(last_message).upper():
                return "planner"

            return "reviewer"

        def _route_after_reviewer(self, state: AgentState) -> Literal["executor", "end"]:
            """Route after reviewer node."""
            if state["step_count"] >= state["max_steps"]:
                return "end"

            if state.get("final_result"):
                return "end"

            last_message = state["messages"][-1].content if state["messages"] else ""
            if "NEEDS_REVISION" in str(last_message):
                return "executor"

            return "end"

        async def run(
            self,
            task: str,
            context: Optional[dict[str, Any]] = None,
            max_steps: Optional[int] = None,
            thread_id: Optional[str] = None,
        ) -> GraphResult:
            """
            Run the multi-agent graph.

            Args:
                task: The task to execute
                context: Additional context
                max_steps: Override max steps
                thread_id: Thread ID for checkpointing

            Returns:
                GraphResult with execution details
            """
            import uuid

            initial_state: AgentState = {
                "messages": [HumanMessage(content=task)],
                "current_agent": "start",
                "task": task,
                "context": context or {},
                "step_count": 0,
                "max_steps": max_steps or self.config.max_steps,
                "final_result": None,
                "error": None,
            }

            config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}

            logger.info("langgraph_run_starting", task=task[:50])

            try:
                final_state = await self.compiled.ainvoke(initial_state, config)

                messages = [
                    {"role": "ai", "content": str(m.content)}
                    for m in final_state.get("messages", [])
                ]

                return GraphResult(
                    success=True,
                    result=final_state.get("final_result"),
                    messages=messages,
                    steps_taken=final_state.get("step_count", 0),
                    final_agent=final_state.get("current_agent", ""),
                )

            except Exception as e:
                logger.error("langgraph_run_failed", error=str(e))
                return GraphResult(
                    success=False,
                    error=str(e),
                )

        async def stream(
            self,
            task: str,
            context: Optional[dict[str, Any]] = None,
        ):
            """Stream graph execution events."""
            import uuid

            initial_state: AgentState = {
                "messages": [HumanMessage(content=task)],
                "current_agent": "start",
                "task": task,
                "context": context or {},
                "step_count": 0,
                "max_steps": self.config.max_steps,
                "final_result": None,
                "error": None,
            }

            config = {"configurable": {"thread_id": str(uuid.uuid4())}}

            async for event in self.compiled.astream(initial_state, config):
                yield event


@dataclass
class LangGraphOrchestrator:
    """
    High-level orchestrator for LangGraph workflows.

    Provides a simple interface for creating and running graph-based
    multi-agent workflows.

    Usage:
        orchestrator = LangGraphOrchestrator()
        graph = orchestrator.create_graph("my-workflow")
        result = await orchestrator.run(graph, "Analyze this data")
    """

    graphs: dict[str, Any] = field(default_factory=dict)

    def create_graph(
        self,
        name: str,
        max_steps: int = 10,
        checkpoint: bool = True,
    ) -> Any:
        """Create a new multi-agent graph."""
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("langgraph not installed")

        config = GraphConfig(
            name=name,
            max_steps=max_steps,
            checkpoint=checkpoint,
        )

        graph = MultiAgentGraph(config)
        self.graphs[name] = graph

        logger.info("graph_created", name=name)
        return graph

    async def run(
        self,
        graph: Any,
        task: str,
        context: Optional[dict[str, Any]] = None,
    ) -> GraphResult:
        """Run a graph with the given task."""
        return await graph.run(task, context)

    def get_graph(self, name: str) -> Optional[Any]:
        """Get a graph by name."""
        return self.graphs.get(name)

    def list_graphs(self) -> list[str]:
        """List all created graphs."""
        return list(self.graphs.keys())


def create_orchestrator() -> LangGraphOrchestrator:
    """Factory function to create a LangGraph orchestrator."""
    return LangGraphOrchestrator()


if __name__ == "__main__":
    async def main():
        """Test the LangGraph orchestrator."""
        print(f"LangGraph available: {LANGGRAPH_AVAILABLE}")

        if LANGGRAPH_AVAILABLE:
            try:
                orchestrator = create_orchestrator()
                graph = orchestrator.create_graph("test-workflow", max_steps=5)

                result = await orchestrator.run(
                    graph,
                    "Create a simple Python function that calculates factorial",
                )

                print(f"Success: {result.success}")
                print(f"Steps taken: {result.steps_taken}")
                print(f"Final agent: {result.final_agent}")

            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Install langgraph to use this module")

    asyncio.run(main())
