#!/usr/bin/env python3
"""
Integration tests for the UNLEASH orchestration layer.

Tests the real SDK integration (agent_sdk_layer, langgraph_layer, workflow_runner)
instead of the previous `await asyncio.sleep(0.1)` stubs.

Run with: pytest core/tests/test_orchestration_layer.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Test: Import Availability
# =============================================================================

class TestImportAvailability:
    """Test that all orchestration modules can be imported."""

    def test_agent_sdk_layer_importable(self):
        """Verify agent_sdk_layer module imports correctly."""
        from core.orchestration import agent_sdk_layer
        assert agent_sdk_layer is not None

    def test_langgraph_layer_importable(self):
        """Verify langgraph_layer module imports correctly."""
        from core.orchestration import langgraph_layer
        assert langgraph_layer is not None

    def test_workflow_runner_importable(self):
        """Verify workflow_runner module imports correctly."""
        from core.orchestration import workflow_runner
        assert workflow_runner is not None

    def test_orchestrator_importable(self):
        """Verify main orchestrator imports correctly."""
        from core import orchestrator
        assert orchestrator is not None

    def test_orchestration_init_exports(self):
        """Verify orchestration __init__ exports all required symbols."""
        from core.orchestration import (
            AGENT_SDK_LAYER_AVAILABLE,
            LANGGRAPH_LAYER_AVAILABLE,
            WORKFLOW_RUNNER_AVAILABLE,
        )
        # These should be booleans indicating SDK availability
        assert isinstance(AGENT_SDK_LAYER_AVAILABLE, bool)
        assert isinstance(LANGGRAPH_LAYER_AVAILABLE, bool)
        assert isinstance(WORKFLOW_RUNNER_AVAILABLE, bool)


# =============================================================================
# Test: Agent SDK Layer
# =============================================================================

class TestAgentSDKLayer:
    """Test agent_sdk_layer module functionality."""

    def test_availability_flags(self):
        """Test that availability flags are properly set."""
        from core.orchestration.agent_sdk_layer import ANTHROPIC_AVAILABLE
        # ANTHROPIC_AVAILABLE should be boolean
        assert isinstance(ANTHROPIC_AVAILABLE, bool)

    def test_agent_config_dataclass(self):
        """Test AgentConfig dataclass structure."""
        from core.orchestration.agent_sdk_layer import AgentConfig

        config = AgentConfig(
            name="test-agent",
            model="claude-sonnet-4-20250514",
            tools=["Read", "Write"],
            system_prompt="Test prompt",
        )
        assert config.name == "test-agent"
        assert config.model == "claude-sonnet-4-20250514"
        assert config.tools == ["Read", "Write"]
        assert config.system_prompt == "Test prompt"

    def test_agent_result_dataclass(self):
        """Test AgentResult dataclass structure."""
        from core.orchestration.agent_sdk_layer import AgentResult

        result = AgentResult(
            output="Test output",
            tool_calls=[{"name": "Read"}],
            success=True,
        )
        assert result.output == "Test output"
        assert result.tool_calls == [{"name": "Read"}]
        assert result.success is True

    def test_agent_class_exists(self):
        """Test Agent class exists with correct attributes."""
        from core.orchestration.agent_sdk_layer import Agent, AgentConfig

        config = AgentConfig(name="test", model="claude-sonnet-4-20250514")
        agent = Agent(config)
        assert hasattr(agent, "config")
        assert hasattr(agent, "_client")
        assert hasattr(agent, "_initialized")
        assert agent._initialized is False

    def test_create_agent_function_exists(self):
        """Test create_agent async function exists and is callable."""
        from core.orchestration.agent_sdk_layer import create_agent
        import inspect
        assert inspect.iscoroutinefunction(create_agent)

    def test_run_agent_loop_function_exists(self):
        """Test run_agent_loop async function exists and is callable."""
        from core.orchestration.agent_sdk_layer import run_agent_loop
        import inspect
        assert inspect.iscoroutinefunction(run_agent_loop)


# =============================================================================
# Test: LangGraph Layer
# =============================================================================

class TestLangGraphLayer:
    """Test langgraph_layer module functionality."""

    def test_availability_flag(self):
        """Test that LANGGRAPH_AVAILABLE flag is set."""
        from core.orchestration.langgraph_layer import LANGGRAPH_AVAILABLE
        assert isinstance(LANGGRAPH_AVAILABLE, bool)

    def test_pipeline_config_dataclass(self):
        """Test PipelineConfig dataclass structure."""
        from core.orchestration.langgraph_layer import PipelineConfig

        config = PipelineConfig(
            name="test-pipeline",
            description="Test description",
            model="claude-sonnet-4-20250514",
            checkpoint_enabled=True,
        )
        assert config.name == "test-pipeline"
        assert config.description == "Test description"
        assert config.model == "claude-sonnet-4-20250514"
        assert config.checkpoint_enabled is True
        assert config.max_iterations == 10  # default value

    def test_pipeline_dataclass_defaults(self):
        """Test Pipeline dataclass has expected default values."""
        from core.orchestration.langgraph_layer import PipelineConfig

        config = PipelineConfig(name="minimal")
        assert config.description == ""
        assert config.steps == []
        assert config.tools == []
        assert config.max_iterations == 10
        assert config.checkpoint_enabled is True

    def test_pipeline_class_exists(self):
        """Test Pipeline class exists with correct attributes."""
        from core.orchestration.langgraph_layer import Pipeline, PipelineConfig

        config = PipelineConfig(name="test")
        pipeline = Pipeline(config=config)
        assert hasattr(pipeline, "config")
        assert hasattr(pipeline, "graph")
        assert hasattr(pipeline, "checkpointer")
        assert hasattr(pipeline, "llm")

    def test_load_pipeline_function_exists(self):
        """Test load_pipeline async function exists."""
        from core.orchestration.langgraph_layer import load_pipeline
        import inspect
        assert inspect.iscoroutinefunction(load_pipeline)

    def test_execute_pipeline_function_exists(self):
        """Test execute_pipeline async function exists."""
        from core.orchestration.langgraph_layer import execute_pipeline
        import inspect
        assert inspect.iscoroutinefunction(execute_pipeline)


# =============================================================================
# Test: Workflow Runner
# =============================================================================

class TestWorkflowRunner:
    """Test workflow_runner module functionality."""

    def test_workflow_type_enum(self):
        """Test WorkflowType enum values."""
        from core.orchestration.workflow_runner import WorkflowType

        assert WorkflowType.AGENT.value == "agent"
        assert WorkflowType.MULTI_AGENT.value == "multi_agent"
        assert WorkflowType.PIPELINE.value == "pipeline"
        assert WorkflowType.TEMPORAL.value == "temporal"
        assert WorkflowType.HYBRID.value == "hybrid"

    def test_execution_mode_enum(self):
        """Test ExecutionMode enum values."""
        from core.orchestration.workflow_runner import ExecutionMode

        assert ExecutionMode.SEQUENTIAL.value == "sequential"
        assert ExecutionMode.PARALLEL.value == "parallel"
        assert ExecutionMode.CONDITIONAL.value == "conditional"

    def test_workflow_definition_dataclass(self):
        """Test WorkflowDefinition dataclass structure."""
        from core.orchestration.workflow_runner import WorkflowDefinition, WorkflowType, ExecutionMode

        definition = WorkflowDefinition(
            name="test-workflow",
            type=WorkflowType.AGENT,
            execution_mode=ExecutionMode.SEQUENTIAL,
        )
        assert definition.name == "test-workflow"
        assert definition.type == WorkflowType.AGENT
        assert definition.execution_mode == ExecutionMode.SEQUENTIAL

    def test_workflow_result_dataclass(self):
        """Test WorkflowResult dataclass structure."""
        from core.orchestration.workflow_runner import WorkflowResult

        result = WorkflowResult(
            workflow_name="test-workflow",
            success=True,
            output={"data": "test"},
        )
        assert result.workflow_name == "test-workflow"
        assert result.success is True
        assert result.output == {"data": "test"}

    def test_get_workflow_function_exists(self):
        """Test get_workflow async function exists."""
        from core.orchestration.workflow_runner import get_workflow
        import inspect
        assert inspect.iscoroutinefunction(get_workflow)

    def test_execute_workflow_function_exists(self):
        """Test execute_workflow async function exists."""
        from core.orchestration.workflow_runner import execute_workflow
        import inspect
        assert inspect.iscoroutinefunction(execute_workflow)


# =============================================================================
# Test: Core Orchestrator Integration
# =============================================================================

class TestCoreOrchestrator:
    """Test the main CoreOrchestrator with SDK integration."""

    def test_agent_role_enum(self):
        """Test AgentRole enum values."""
        from core.orchestrator import AgentRole

        assert AgentRole.ORCHESTRATOR.value == "orchestrator"
        assert AgentRole.WORKER.value == "worker"
        assert AgentRole.SPECIALIST.value == "specialist"
        assert AgentRole.MONITOR.value == "monitor"
        assert AgentRole.MEMORY.value == "memory"

    def test_task_status_enum(self):
        """Test TaskStatus enum values."""
        from core.orchestrator import TaskStatus

        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"

    def test_agent_config_dataclass(self):
        """Test AgentConfig dataclass with new SDK fields."""
        from core.orchestrator import AgentConfig, AgentRole

        config = AgentConfig(
            name="test-agent",
            role=AgentRole.WORKER,
            capabilities=["code_review"],
            model="claude-sonnet-4-20250514",
            tools=["Read", "Write", "Bash"],
            system_prompt="You are a test agent.",
        )
        assert config.name == "test-agent"
        assert config.role == AgentRole.WORKER
        assert config.model == "claude-sonnet-4-20250514"
        assert config.tools == ["Read", "Write", "Bash"]
        assert config.system_prompt == "You are a test agent."
        assert config._agent_instance is None  # Not yet initialized

    def test_task_result_dataclass(self):
        """Test TaskResult dataclass structure."""
        from core.orchestrator import TaskResult, TaskStatus

        result = TaskResult(
            task_id="task_123",
            status=TaskStatus.COMPLETED,
            output={"message": "Success"},
            duration_ms=1500.5,
        )
        assert result.task_id == "task_123"
        assert result.status == TaskStatus.COMPLETED
        assert result.output == {"message": "Success"}
        assert result.duration_ms == 1500.5

    def test_orchestrator_instantiation(self):
        """Test CoreOrchestrator can be instantiated."""
        from core.orchestrator import CoreOrchestrator

        orchestrator = CoreOrchestrator()
        assert orchestrator is not None
        assert orchestrator._initialized is False
        assert len(orchestrator.agents) == 0  # Not yet registered

    def test_sdk_availability_flag(self):
        """Test AGENT_SDK_AVAILABLE flag is set in orchestrator."""
        from core.orchestrator import AGENT_SDK_AVAILABLE
        assert isinstance(AGENT_SDK_AVAILABLE, bool)

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self):
        """Test orchestrator initialization registers agents."""
        from core.orchestrator import CoreOrchestrator

        orchestrator = CoreOrchestrator()
        await orchestrator.initialize()

        # Check agents were registered
        assert len(orchestrator.agents) > 0
        assert "ralph" in orchestrator.agents
        assert "validator" in orchestrator.agents
        assert "memory-agent" in orchestrator.agents
        assert "health-monitor" in orchestrator.agents

        # Check each agent has the new SDK fields
        for name, config in orchestrator.agents.items():
            assert config.model is not None
            assert config.tools is not None
            assert config.system_prompt is not None

    @pytest.mark.asyncio
    async def test_orchestrator_get_status(self):
        """Test get_status returns proper structure."""
        from core.orchestrator import CoreOrchestrator

        orchestrator = CoreOrchestrator()
        await orchestrator.initialize()

        status = orchestrator.get_status()
        assert "initialized" in status
        assert "registered_agents" in status
        assert "active_tasks" in status
        assert "paths" in status
        assert isinstance(status["registered_agents"], list)

    @pytest.mark.asyncio
    async def test_agent_selection_for_validation(self):
        """Test agent auto-selection for validation tasks."""
        from core.orchestrator import CoreOrchestrator

        orchestrator = CoreOrchestrator()
        await orchestrator.initialize()

        # Test agent selection
        selected = orchestrator._select_agents_for_task("validate hooks")
        assert "validator" in selected

    @pytest.mark.asyncio
    async def test_agent_selection_for_health(self):
        """Test agent auto-selection for health tasks."""
        from core.orchestrator import CoreOrchestrator

        orchestrator = CoreOrchestrator()
        await orchestrator.initialize()

        selected = orchestrator._select_agents_for_task("check health")
        assert "health-monitor" in selected

    @pytest.mark.asyncio
    async def test_agent_selection_for_memory(self):
        """Test agent auto-selection for memory tasks."""
        from core.orchestrator import CoreOrchestrator

        orchestrator = CoreOrchestrator()
        await orchestrator.initialize()

        selected = orchestrator._select_agents_for_task("persist data to memory")
        assert "memory-agent" in selected

    @pytest.mark.asyncio
    async def test_build_agent_prompt(self):
        """Test _build_agent_prompt generates role-specific prompts."""
        from core.orchestrator import CoreOrchestrator

        orchestrator = CoreOrchestrator()
        await orchestrator.initialize()

        ralph_config = orchestrator.agents["ralph"]
        prompt = orchestrator._build_agent_prompt("test task", ralph_config)

        assert "orchestration agent" in prompt.lower()
        assert "test task" in prompt


# =============================================================================
# Test: SDK Integration (requires API key)
# =============================================================================

@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)
class TestSDKIntegrationWithAPIKey:
    """
    Tests that require a valid ANTHROPIC_API_KEY.
    These are skipped in CI environments without the key.
    """

    @pytest.mark.asyncio
    async def test_create_agent_with_api_key(self):
        """Test creating an agent with valid API key."""
        from core.orchestration.agent_sdk_layer import create_agent

        agent = await create_agent(
            name="integration-test-agent",
            model="claude-sonnet-4-20250514",
            tools=["Read"],
        )
        assert agent is not None
        # Agent should have config.name set
        assert agent.config.name == "integration-test-agent"

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_run_simple_agent_task(self):
        """Test running a simple task through the agent (uses API)."""
        from core.orchestration.agent_sdk_layer import create_agent, run_agent_loop

        agent = await create_agent(
            name="test-runner",
            model="claude-sonnet-4-20250514",
            tools=["Read"],
            system_prompt="You are a test agent. Respond briefly.",
        )

        # This will make an actual API call
        result = await run_agent_loop(
            agent=agent,
            prompt="Say 'test successful' and nothing else.",
            max_turns=1,
        )

        assert result is not None
        # Result should have success or output fields
        assert isinstance(result, dict)


# =============================================================================
# Test: End-to-End Coordination (fallback mode)
# =============================================================================

class TestCoordinationFallback:
    """Test coordination in fallback mode (no API key required)."""

    @pytest.mark.asyncio
    async def test_coordinate_agents_fallback(self):
        """Test coordinate_agents works in fallback mode."""
        from core.orchestrator import CoreOrchestrator, TaskStatus

        orchestrator = CoreOrchestrator()
        await orchestrator.initialize()

        # This should work even without API key (fallback mode)
        result = await orchestrator.coordinate_agents(
            task="test coordination",
            agents=["ralph"],
        )

        assert result is not None
        assert result.task_id.startswith("task_")
        # In fallback mode, should still return a result
        assert result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
        assert result.duration_ms > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
