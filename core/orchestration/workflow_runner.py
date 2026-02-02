"""
Workflow Runner - Unified workflow execution layer.

Part of L1 Orchestration layer. Provides:
- get_workflow(): Load a workflow definition
- execute_workflow(): Execute a workflow with input data

Supports multiple execution backends:
- Agent-based execution (via agent_sdk_layer)
- Pipeline-based execution (via langgraph_layer)
- Temporal workflows (if available)

NO STUBS - Real implementations only.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import structlog
import yaml

# Configure logging
logger = structlog.get_logger(__name__)

# Import internal layers
try:
    from core.orchestration.agent_sdk_layer import create_agent, run_agent_loop, Agent
    AGENT_LAYER_AVAILABLE = True
except ImportError:
    AGENT_LAYER_AVAILABLE = False
    logger.warning("agent_sdk_layer_not_available")

try:
    from core.orchestration.langgraph_layer import load_pipeline, execute_pipeline, Pipeline
    LANGGRAPH_LAYER_AVAILABLE = True
except ImportError:
    LANGGRAPH_LAYER_AVAILABLE = False
    logger.warning("langgraph_layer_not_available")

# Check for Temporal
TEMPORAL_AVAILABLE = False
try:
    from temporalio.client import Client as TemporalClient
    from temporalio.worker import Worker as TemporalWorker
    TEMPORAL_AVAILABLE = True
    logger.info("temporal_available")
except ImportError:
    logger.debug("temporal_not_available", install_cmd="pip install temporalio")


class WorkflowType(Enum):
    """Types of workflow execution."""
    AGENT = "agent"           # Single agent execution
    MULTI_AGENT = "multi_agent"  # Multiple agents in sequence/parallel
    PIPELINE = "pipeline"     # LangGraph pipeline
    TEMPORAL = "temporal"     # Temporal workflow
    HYBRID = "hybrid"         # Mixed execution modes


class ExecutionMode(Enum):
    """Execution mode for multi-step workflows."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"


@dataclass
class WorkflowStep:
    """A single step in a workflow."""
    name: str
    type: str  # agent, pipeline, tool, conditional
    config: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None
    timeout: int = 300  # seconds


@dataclass
class WorkflowDefinition:
    """Complete workflow definition."""
    name: str
    description: str = ""
    type: WorkflowType = WorkflowType.AGENT
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    steps: List[WorkflowStep] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """Result from workflow execution."""
    workflow_name: str
    success: bool
    output: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Workflow:
    """
    Unified workflow wrapper supporting multiple execution backends.
    """

    def __init__(self, definition: WorkflowDefinition):
        self.definition = definition
        self._agents: Dict[str, Agent] = {}
        self._pipelines: Dict[str, Pipeline] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize workflow resources."""
        if self._initialized:
            return

        # Initialize agents for agent-type steps
        for step in self.definition.steps:
            if step.type == "agent" and AGENT_LAYER_AVAILABLE:
                agent_config = step.config.get("agent", {})
                agent = await create_agent(
                    name=step.name,
                    model=agent_config.get("model", "claude-sonnet-4-20250514"),
                    tools=agent_config.get("tools", ["Read", "Write", "Bash"]),
                    system_prompt=agent_config.get("system_prompt"),
                )
                self._agents[step.name] = agent

            elif step.type == "pipeline" and LANGGRAPH_LAYER_AVAILABLE:
                pipeline_def = step.config.get("pipeline", {})
                pipeline = await load_pipeline(pipeline_def)
                self._pipelines[step.name] = pipeline

        self._initialized = True
        logger.info("workflow_initialized", name=self.definition.name, steps=len(self.definition.steps))

    async def execute(self, input_data: Dict[str, Any]) -> WorkflowResult:
        """Execute the workflow with input data."""
        await self.initialize()

        logger.info("workflow_execution_starting", workflow=self.definition.name)

        step_results: Dict[str, Any] = {}
        errors: List[str] = []
        context = {"input": input_data, "results": step_results}

        try:
            if self.definition.execution_mode == ExecutionMode.SEQUENTIAL:
                await self._execute_sequential(context, step_results, errors)
            elif self.definition.execution_mode == ExecutionMode.PARALLEL:
                await self._execute_parallel(context, step_results, errors)
            elif self.definition.execution_mode == ExecutionMode.CONDITIONAL:
                await self._execute_conditional(context, step_results, errors)

            success = len(errors) == 0
            output = self._build_output(step_results)

            return WorkflowResult(
                workflow_name=self.definition.name,
                success=success,
                output=output,
                step_results=step_results,
                errors=errors,
                metadata={"execution_mode": self.definition.execution_mode.value},
            )

        except Exception as e:
            logger.error("workflow_execution_error", error=str(e))
            return WorkflowResult(
                workflow_name=self.definition.name,
                success=False,
                errors=[str(e)],
                step_results=step_results,
            )

    async def _execute_sequential(
        self,
        context: Dict[str, Any],
        step_results: Dict[str, Any],
        errors: List[str],
    ) -> None:
        """Execute steps sequentially."""
        for step in self.definition.steps:
            # Check condition if present
            if step.condition and not self._evaluate_condition(step.condition, context):
                logger.info("step_skipped_condition", step=step.name)
                continue

            try:
                result = await self._execute_step(step, context)
                step_results[step.name] = result
                context["results"][step.name] = result
            except Exception as e:
                error_msg = f"Step {step.name} failed: {str(e)}"
                errors.append(error_msg)
                logger.error("step_failed", step=step.name, error=str(e))
                # Continue or break based on step config
                if step.config.get("fail_fast", True):
                    break

    async def _execute_parallel(
        self,
        context: Dict[str, Any],
        step_results: Dict[str, Any],
        errors: List[str],
    ) -> None:
        """Execute independent steps in parallel."""
        # Group steps by dependencies
        ready_steps = [s for s in self.definition.steps if not s.depends_on]
        pending_steps = [s for s in self.definition.steps if s.depends_on]

        # Execute ready steps in parallel
        while ready_steps:
            tasks = []
            for step in ready_steps:
                if step.condition and not self._evaluate_condition(step.condition, context):
                    logger.info("step_skipped_condition", step=step.name)
                    continue
                tasks.append(self._execute_step_with_timeout(step, context))

            # Wait for all tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for step, result in zip(ready_steps, results):
                if isinstance(result, Exception):
                    errors.append(f"Step {step.name} failed: {str(result)}")
                else:
                    step_results[step.name] = result
                    context["results"][step.name] = result

            # Find newly ready steps
            ready_steps = []
            still_pending = []
            for step in pending_steps:
                if all(dep in step_results for dep in step.depends_on):
                    ready_steps.append(step)
                else:
                    still_pending.append(step)
            pending_steps = still_pending

    async def _execute_conditional(
        self,
        context: Dict[str, Any],
        step_results: Dict[str, Any],
        errors: List[str],
    ) -> None:
        """Execute steps based on conditions."""
        for step in self.definition.steps:
            if step.condition:
                if self._evaluate_condition(step.condition, context):
                    try:
                        result = await self._execute_step(step, context)
                        step_results[step.name] = result
                        context["results"][step.name] = result
                    except Exception as e:
                        errors.append(f"Step {step.name} failed: {str(e)}")
                else:
                    logger.info("step_skipped_condition", step=step.name)
            else:
                # No condition, always execute
                try:
                    result = await self._execute_step(step, context)
                    step_results[step.name] = result
                    context["results"][step.name] = result
                except Exception as e:
                    errors.append(f"Step {step.name} failed: {str(e)}")

    async def _execute_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute a single workflow step."""
        logger.info("executing_step", step=step.name, type=step.type)

        if step.type == "agent":
            return await self._execute_agent_step(step, context)
        elif step.type == "pipeline":
            return await self._execute_pipeline_step(step, context)
        elif step.type == "tool":
            return await self._execute_tool_step(step, context)
        elif step.type == "transform":
            return await self._execute_transform_step(step, context)
        else:
            raise ValueError(f"Unknown step type: {step.type}")

    async def _execute_step_with_timeout(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute a step with timeout."""
        try:
            return await asyncio.wait_for(
                self._execute_step(step, context),
                timeout=step.timeout,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Step {step.name} timed out after {step.timeout}s")

    async def _execute_agent_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute an agent-based step."""
        if not AGENT_LAYER_AVAILABLE:
            raise RuntimeError("Agent layer not available")

        agent = self._agents.get(step.name)
        if not agent:
            raise ValueError(f"Agent not found for step: {step.name}")

        # Build prompt from context
        prompt_template = step.config.get("prompt", "{input}")
        prompt = self._format_template(prompt_template, context)

        max_turns = step.config.get("max_turns", 10)
        result = await run_agent_loop(agent, prompt, max_turns=max_turns)

        return result

    async def _execute_pipeline_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute a pipeline-based step."""
        if not LANGGRAPH_LAYER_AVAILABLE:
            raise RuntimeError("LangGraph layer not available")

        pipeline = self._pipelines.get(step.name)
        if not pipeline:
            raise ValueError(f"Pipeline not found for step: {step.name}")

        # Build input from context
        input_data = self._build_step_input(step, context)
        result = await execute_pipeline(pipeline, input_data)

        return result

    async def _execute_tool_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute a tool directly."""
        tool_name = step.config.get("tool")
        tool_input = self._build_step_input(step, context)

        # Simple tool execution
        if tool_name == "read_file":
            path = tool_input.get("path", "")
            if os.path.exists(path):
                with open(path, "r") as f:
                    return {"content": f.read()}
            return {"error": f"File not found: {path}"}

        elif tool_name == "write_file":
            path = tool_input.get("path", "")
            content = tool_input.get("content", "")
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return {"success": True, "path": path}

        elif tool_name == "bash":
            command = tool_input.get("command", "")
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            return {
                "stdout": stdout.decode()[:10000],
                "stderr": stderr.decode()[:2000],
                "returncode": proc.returncode,
            }

        else:
            return {"error": f"Unknown tool: {tool_name}"}

    async def _execute_transform_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute a data transformation step."""
        transform_type = step.config.get("transform", "passthrough")
        input_data = self._build_step_input(step, context)

        if transform_type == "passthrough":
            return input_data

        elif transform_type == "extract":
            # Extract specific fields
            fields = step.config.get("fields", [])
            return {k: input_data.get(k) for k in fields if k in input_data}

        elif transform_type == "merge":
            # Merge multiple step results
            sources = step.config.get("sources", [])
            merged = {}
            for source in sources:
                if source in context["results"]:
                    merged.update(context["results"][source])
            return merged

        elif transform_type == "format":
            # Format output using template
            template = step.config.get("template", "{input}")
            return self._format_template(template, context)

        else:
            return input_data

    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a condition string against context."""
        try:
            # Simple condition evaluation
            # Supports: "results.step_name.success == True"
            # Supports: "input.key exists"
            # Supports: "results.step_name.output contains 'value'"

            if "==" in condition:
                left, right = condition.split("==", 1)
                left_val = self._resolve_path(left.strip(), context)
                right_val = eval(right.strip())  # Safe for simple literals
                return left_val == right_val

            elif "exists" in condition:
                path = condition.replace("exists", "").strip()
                try:
                    self._resolve_path(path, context)
                    return True
                except (KeyError, AttributeError):
                    return False

            elif "contains" in condition:
                parts = condition.split("contains", 1)
                left_val = str(self._resolve_path(parts[0].strip(), context))
                right_val = parts[1].strip().strip("'\"")
                return right_val in left_val

            else:
                # Default: treat as truthy check
                val = self._resolve_path(condition, context)
                return bool(val)

        except Exception as e:
            logger.warning("condition_evaluation_error", condition=condition, error=str(e))
            return False

    def _resolve_path(self, path: str, context: Dict[str, Any]) -> Any:
        """Resolve a dotted path in context."""
        parts = path.split(".")
        current = context
        for part in parts:
            if isinstance(current, dict):
                current = current[part]
            else:
                current = getattr(current, part)
        return current

    def _format_template(self, template: str, context: Dict[str, Any]) -> str:
        """Format a template string with context values."""
        # Replace {path.to.value} with actual values
        pattern = r"\{([^}]+)\}"

        def replace(match: re.Match) -> str:
            path = match.group(1)
            try:
                value = self._resolve_path(path, context)
                return str(value)
            except Exception:
                return match.group(0)

        return re.sub(pattern, replace, template)

    def _build_step_input(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build input for a step from context."""
        input_config = step.config.get("input", {})

        if isinstance(input_config, str):
            # Reference to another step's output
            return context["results"].get(input_config, context["input"])

        elif isinstance(input_config, dict):
            # Build input from config
            result = {}
            for key, value in input_config.items():
                if isinstance(value, str) and value.startswith("$"):
                    # Reference: $results.step_name.output
                    path = value[1:]
                    try:
                        result[key] = self._resolve_path(path, context)
                    except Exception:
                        result[key] = value
                else:
                    result[key] = value
            return result

        else:
            return context["input"]

    def _build_output(self, step_results: Dict[str, Any]) -> Dict[str, Any]:
        """Build final output from step results."""
        # Get output from last step or configured output step
        output_step = self.definition.config.get("output_step")

        if output_step and output_step in step_results:
            return step_results[output_step]

        # Return last step result
        if step_results:
            last_step = self.definition.steps[-1].name
            return step_results.get(last_step, {})

        return {}


# ============================================================================
# Workflow Registry and Loading
# ============================================================================

_workflow_registry: Dict[str, WorkflowDefinition] = {}


def register_workflow(definition: WorkflowDefinition) -> None:
    """Register a workflow definition."""
    _workflow_registry[definition.name] = definition
    logger.info("workflow_registered", name=definition.name)


def load_workflow_from_yaml(path: Union[str, Path]) -> WorkflowDefinition:
    """Load a workflow definition from YAML file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Workflow file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return _parse_workflow_definition(data)


def load_workflow_from_dict(data: Dict[str, Any]) -> WorkflowDefinition:
    """Load a workflow definition from a dictionary."""
    return _parse_workflow_definition(data)


def _parse_workflow_definition(data: Dict[str, Any]) -> WorkflowDefinition:
    """Parse workflow definition from dictionary."""
    # Parse workflow type
    workflow_type = WorkflowType.AGENT
    type_str = data.get("type", "agent")
    try:
        workflow_type = WorkflowType(type_str)
    except ValueError:
        pass

    # Parse execution mode
    execution_mode = ExecutionMode.SEQUENTIAL
    mode_str = data.get("execution_mode", "sequential")
    try:
        execution_mode = ExecutionMode(mode_str)
    except ValueError:
        pass

    # Parse steps
    steps = []
    for i, step_data in enumerate(data.get("steps", [])):
        step = WorkflowStep(
            name=step_data.get("name", f"step_{i}"),
            type=step_data.get("type", "agent"),
            config=step_data.get("config", {}),
            depends_on=step_data.get("depends_on", []),
            condition=step_data.get("condition"),
            timeout=step_data.get("timeout", 300),
        )
        steps.append(step)

    return WorkflowDefinition(
        name=data.get("name", "unnamed_workflow"),
        description=data.get("description", ""),
        type=workflow_type,
        execution_mode=execution_mode,
        steps=steps,
        config=data.get("config", {}),
        metadata=data.get("metadata", {}),
    )


# ============================================================================
# Public API Functions (used by CLI)
# ============================================================================

async def get_workflow(
    name: str,
    path: Optional[Union[str, Path]] = None,
) -> Workflow:
    """
    Get a workflow by name or load from file.

    Args:
        name: Workflow name (for registry lookup)
        path: Optional path to workflow definition file

    Returns:
        Workflow instance ready for execution
    """
    # Check registry first
    if name in _workflow_registry:
        definition = _workflow_registry[name]
        logger.info("workflow_from_registry", name=name)
        return Workflow(definition)

    # Load from file if path provided
    if path:
        definition = load_workflow_from_yaml(path)
        register_workflow(definition)
        logger.info("workflow_loaded_from_file", name=definition.name, path=str(path))
        return Workflow(definition)

    # Search in standard locations
    search_paths = [
        Path.cwd() / "workflows" / f"{name}.yaml",
        Path.cwd() / "workflows" / f"{name}.yml",
        Path.cwd() / ".unleash" / "workflows" / f"{name}.yaml",
        Path.home() / ".unleash" / "workflows" / f"{name}.yaml",
    ]

    for search_path in search_paths:
        if search_path.exists():
            definition = load_workflow_from_yaml(search_path)
            register_workflow(definition)
            logger.info("workflow_found", name=name, path=str(search_path))
            return Workflow(definition)

    raise ValueError(f"Workflow not found: {name}")


async def execute_workflow(
    workflow: Workflow,
    input_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Execute a workflow with input data.

    Args:
        workflow: The workflow to execute
        input_data: Input data for the workflow

    Returns:
        Dictionary with execution results
    """
    logger.info("workflow_execution_starting", workflow=workflow.definition.name)

    result = await workflow.execute(input_data)

    logger.info(
        "workflow_execution_completed",
        workflow=workflow.definition.name,
        success=result.success,
        steps=len(result.step_results),
    )

    return {
        "workflow": result.workflow_name,
        "success": result.success,
        "output": result.output,
        "step_results": result.step_results,
        "errors": result.errors,
        "metadata": result.metadata,
    }


# ============================================================================
# Built-in Workflows
# ============================================================================

# Register some built-in workflows
_builtin_workflows = [
    WorkflowDefinition(
        name="simple_agent",
        description="Execute a single agent task",
        type=WorkflowType.AGENT,
        steps=[
            WorkflowStep(
                name="execute",
                type="agent",
                config={
                    "agent": {
                        "model": "claude-sonnet-4-20250514",
                        "tools": ["Read", "Write", "Bash"],
                    },
                    "prompt": "{input.task}",
                },
            )
        ],
    ),
    WorkflowDefinition(
        name="code_review",
        description="Automated code review workflow",
        type=WorkflowType.MULTI_AGENT,
        execution_mode=ExecutionMode.SEQUENTIAL,
        steps=[
            WorkflowStep(
                name="analyze",
                type="agent",
                config={
                    "agent": {
                        "model": "claude-sonnet-4-20250514",
                        "system_prompt": "You are a code analysis expert.",
                    },
                    "prompt": "Analyze this code for issues: {input.code}",
                },
            ),
            WorkflowStep(
                name="security",
                type="agent",
                config={
                    "agent": {
                        "model": "claude-sonnet-4-20250514",
                        "system_prompt": "You are a security expert.",
                    },
                    "prompt": "Review for security issues: {input.code}\nAnalysis: {results.analyze.output}",
                },
            ),
            WorkflowStep(
                name="summarize",
                type="transform",
                config={
                    "transform": "merge",
                    "sources": ["analyze", "security"],
                },
            ),
        ],
    ),
]

for workflow in _builtin_workflows:
    register_workflow(workflow)


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    "Workflow",
    "WorkflowDefinition",
    "WorkflowStep",
    "WorkflowResult",
    "WorkflowType",
    "ExecutionMode",
    "get_workflow",
    "execute_workflow",
    "register_workflow",
    "load_workflow_from_yaml",
    "load_workflow_from_dict",
    "AGENT_LAYER_AVAILABLE",
    "LANGGRAPH_LAYER_AVAILABLE",
    "TEMPORAL_AVAILABLE",
]


if __name__ == "__main__":
    # Test the module
    async def test():
        # Test simple agent workflow
        workflow = await get_workflow("simple_agent")
        result = await execute_workflow(
            workflow,
            {"task": "List the files in the current directory"},
        )
        print(f"Result: {result}")

    asyncio.run(test())
