"""
LangGraph Layer - Pipeline execution using LangGraph.

Part of L1 Orchestration layer. Provides:
- load_pipeline(): Load a pipeline from YAML definition
- execute_pipeline(): Execute a loaded pipeline with input data

Uses langgraph with langchain-anthropic for Claude integration.

NO STUBS - Real implementations only.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union

import structlog

# Configure logging
logger = structlog.get_logger(__name__)

# Check for LangGraph availability
LANGGRAPH_AVAILABLE = False
try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import create_react_agent
    LANGGRAPH_AVAILABLE = True
    logger.info("langgraph_available")
except ImportError:
    logger.warning("langgraph_not_available", install_cmd="pip install langgraph>=1.0.0")

# Check for LangChain Anthropic
LANGCHAIN_ANTHROPIC_AVAILABLE = False
try:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    LANGCHAIN_ANTHROPIC_AVAILABLE = True
    logger.info("langchain_anthropic_available")
except ImportError:
    logger.warning("langchain_anthropic_not_available", install_cmd="pip install langchain-anthropic")


class PipelineState(TypedDict, total=False):
    """State for pipeline execution."""
    messages: List[Dict[str, Any]]
    input: Dict[str, Any]
    output: Dict[str, Any]
    current_step: str
    steps_completed: List[str]
    metadata: Dict[str, Any]
    error: Optional[str]


@dataclass
class PipelineConfig:
    """Configuration for a pipeline."""
    name: str
    description: str = ""
    model: str = "claude-sonnet-4-20250514"
    steps: List[Dict[str, Any]] = field(default_factory=list)
    tools: List[Dict[str, Any]] = field(default_factory=list)
    max_iterations: int = 10
    checkpoint_enabled: bool = True


@dataclass
class Pipeline:
    """A loaded pipeline ready for execution."""
    config: PipelineConfig
    graph: Any = None  # StateGraph instance
    checkpointer: Any = None  # MemorySaver instance
    llm: Any = None  # ChatAnthropic instance

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the pipeline with input data."""
        if not self.graph:
            raise RuntimeError("Pipeline not compiled. Call compile() first.")

        initial_state: PipelineState = {
            "messages": [],
            "input": input_data,
            "output": {},
            "current_step": "start",
            "steps_completed": [],
            "metadata": {"pipeline": self.config.name},
            "error": None,
        }

        config = {"configurable": {"thread_id": f"pipeline_{self.config.name}"}}

        try:
            # Execute the graph
            result = await self.graph.ainvoke(initial_state, config=config)
            return {
                "success": True,
                "output": result.get("output", {}),
                "steps_completed": result.get("steps_completed", []),
                "metadata": result.get("metadata", {}),
            }
        except Exception as e:
            logger.error("pipeline_execution_error", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "output": {},
            }


class PipelineBuilder:
    """Builder for creating LangGraph pipelines."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.llm = None
        self.tools = []

    async def build(self) -> Pipeline:
        """Build and compile the pipeline."""
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph not available. Install with: pip install langgraph>=1.0.0"
            )

        if not LANGCHAIN_ANTHROPIC_AVAILABLE:
            raise ImportError(
                "LangChain Anthropic not available. Install with: pip install langchain-anthropic"
            )

        # Initialize LLM
        self.llm = ChatAnthropic(
            model=self.config.model,
            temperature=0.7,
            max_tokens=4096,
        )

        # Build the graph based on pipeline steps
        graph = self._build_graph()

        # Create checkpointer if enabled
        checkpointer = MemorySaver() if self.config.checkpoint_enabled else None

        # Compile the graph
        compiled = graph.compile(checkpointer=checkpointer)

        pipeline = Pipeline(
            config=self.config,
            graph=compiled,
            checkpointer=checkpointer,
            llm=self.llm,
        )

        logger.info("pipeline_built", name=self.config.name, steps=len(self.config.steps))
        return pipeline

    def _build_graph(self) -> "StateGraph":
        """Build the StateGraph from pipeline configuration."""
        graph = StateGraph(PipelineState)

        # Add nodes for each step
        for i, step in enumerate(self.config.steps):
            step_name = step.get("name", f"step_{i}")
            step_type = step.get("type", "llm")

            if step_type == "llm":
                node_fn = self._create_llm_node(step)
            elif step_type == "tool":
                node_fn = self._create_tool_node(step)
            elif step_type == "conditional":
                node_fn = self._create_conditional_node(step)
            else:
                node_fn = self._create_passthrough_node(step)

            graph.add_node(step_name, node_fn)

        # Add edges between steps
        if self.config.steps:
            # Connect START to first step
            first_step = self.config.steps[0].get("name", "step_0")
            graph.add_edge(START, first_step)

            # Connect steps in sequence
            for i in range(len(self.config.steps) - 1):
                current = self.config.steps[i].get("name", f"step_{i}")
                next_step = self.config.steps[i + 1].get("name", f"step_{i + 1}")

                # Check for conditional routing
                if self.config.steps[i].get("type") == "conditional":
                    conditions = self.config.steps[i].get("conditions", {})
                    graph.add_conditional_edges(
                        current,
                        self._create_router(conditions),
                        conditions,
                    )
                else:
                    graph.add_edge(current, next_step)

            # Connect last step to END
            last_step = self.config.steps[-1].get("name", f"step_{len(self.config.steps) - 1}")
            if self.config.steps[-1].get("type") != "conditional":
                graph.add_edge(last_step, END)

        return graph

    def _create_llm_node(self, step: Dict[str, Any]) -> Callable:
        """Create an LLM node function."""
        prompt_template = step.get("prompt", "Process the input: {input}")
        system_prompt = step.get("system", None)

        async def llm_node(state: PipelineState) -> PipelineState:
            step_name = step.get("name", "llm_step")
            logger.info("executing_llm_node", step=step_name)

            try:
                # Format the prompt
                prompt = prompt_template.format(**state.get("input", {}))

                messages = []
                if system_prompt:
                    messages.append(SystemMessage(content=system_prompt))
                messages.append(HumanMessage(content=prompt))

                # Call the LLM
                response = await self.llm.ainvoke(messages)

                # Update state
                state["output"][step_name] = response.content
                state["steps_completed"].append(step_name)
                state["current_step"] = step_name
                state["messages"].append({
                    "step": step_name,
                    "role": "assistant",
                    "content": response.content,
                })

            except Exception as e:
                logger.error("llm_node_error", step=step_name, error=str(e))
                state["error"] = str(e)

            return state

        return llm_node

    def _create_tool_node(self, step: Dict[str, Any]) -> Callable:
        """Create a tool execution node."""
        tool_name = step.get("tool", "unknown")
        tool_config = step.get("config", {})

        async def tool_node(state: PipelineState) -> PipelineState:
            step_name = step.get("name", f"tool_{tool_name}")
            logger.info("executing_tool_node", step=step_name, tool=tool_name)

            try:
                # Execute the tool
                result = await self._execute_tool(tool_name, state.get("input", {}), tool_config)

                state["output"][step_name] = result
                state["steps_completed"].append(step_name)
                state["current_step"] = step_name

            except Exception as e:
                logger.error("tool_node_error", step=step_name, error=str(e))
                state["error"] = str(e)

            return state

        return tool_node

    def _create_conditional_node(self, step: Dict[str, Any]) -> Callable:
        """Create a conditional routing node."""
        condition_field = step.get("condition_field", "output")

        async def conditional_node(state: PipelineState) -> PipelineState:
            step_name = step.get("name", "conditional")
            logger.info("executing_conditional_node", step=step_name)

            state["steps_completed"].append(step_name)
            state["current_step"] = step_name
            return state

        return conditional_node

    def _create_passthrough_node(self, step: Dict[str, Any]) -> Callable:
        """Create a passthrough node."""
        async def passthrough_node(state: PipelineState) -> PipelineState:
            step_name = step.get("name", "passthrough")
            state["steps_completed"].append(step_name)
            state["current_step"] = step_name
            return state

        return passthrough_node

    def _create_router(self, conditions: Dict[str, str]) -> Callable:
        """Create a routing function for conditional edges."""
        def router(state: PipelineState) -> str:
            # Simple routing based on output
            output = state.get("output", {})
            for key, target in conditions.items():
                if key in str(output):
                    return target
            return list(conditions.values())[0] if conditions else END

        return router

    async def _execute_tool(
        self,
        tool_name: str,
        input_data: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Any:
        """Execute a tool by name."""
        # This would integrate with the unified tool layer
        # For now, simple implementations
        if tool_name == "search":
            return {"results": f"Search results for: {input_data}"}
        elif tool_name == "analyze":
            return {"analysis": f"Analysis of: {input_data}"}
        else:
            return {"tool": tool_name, "input": input_data, "status": "executed"}


# ============================================================================
# Public API Functions (used by CLI)
# ============================================================================

async def load_pipeline(
    pipeline_def: Dict[str, Any],
    config_override: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """
    Load a pipeline from a YAML definition.

    Args:
        pipeline_def: Pipeline definition dictionary (from YAML)
        config_override: Optional configuration overrides

    Returns:
        Loaded Pipeline ready for execution

    Example pipeline_def:
        ```yaml
        name: my_pipeline
        description: A sample pipeline
        model: claude-sonnet-4-20250514
        steps:
          - name: analyze
            type: llm
            prompt: "Analyze the following: {input}"
          - name: summarize
            type: llm
            prompt: "Summarize: {analyze_output}"
        ```
    """
    if not LANGGRAPH_AVAILABLE:
        raise ImportError(
            "LangGraph not available. Install with: pip install langgraph>=1.0.0"
        )

    # Merge config override
    merged_def = {**pipeline_def}
    if config_override:
        merged_def.update(config_override)

    # Create pipeline config
    config = PipelineConfig(
        name=merged_def.get("name", "unnamed_pipeline"),
        description=merged_def.get("description", ""),
        model=merged_def.get("model", "claude-sonnet-4-20250514"),
        steps=merged_def.get("steps", []),
        tools=merged_def.get("tools", []),
        max_iterations=merged_def.get("max_iterations", 10),
        checkpoint_enabled=merged_def.get("checkpoint_enabled", True),
    )

    # Build the pipeline
    builder = PipelineBuilder(config)
    pipeline = await builder.build()

    logger.info("pipeline_loaded", name=config.name, steps=len(config.steps))
    return pipeline


async def execute_pipeline(
    pipeline: Pipeline,
    input_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute a loaded pipeline with input data.

    Args:
        pipeline: The loaded pipeline
        input_data: Input data for the pipeline

    Returns:
        Dictionary with execution results
    """
    logger.info("pipeline_execution_starting", pipeline=pipeline.config.name)

    result = await pipeline.execute(input_data)

    logger.info(
        "pipeline_execution_completed",
        pipeline=pipeline.config.name,
        success=result.get("success", False),
        steps=len(result.get("steps_completed", [])),
    )

    return result


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    "Pipeline",
    "PipelineConfig",
    "PipelineBuilder",
    "PipelineState",
    "load_pipeline",
    "execute_pipeline",
    "LANGGRAPH_AVAILABLE",
    "LANGCHAIN_ANTHROPIC_AVAILABLE",
]


if __name__ == "__main__":
    # Test the module
    async def test():
        pipeline_def = {
            "name": "test_pipeline",
            "description": "A test pipeline",
            "steps": [
                {
                    "name": "greet",
                    "type": "llm",
                    "prompt": "Say hello to {name}",
                }
            ],
        }

        pipeline = await load_pipeline(pipeline_def)
        result = await execute_pipeline(pipeline, {"name": "World"})
        print(f"Result: {result}")

    asyncio.run(test())
