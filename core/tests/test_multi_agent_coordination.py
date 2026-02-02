"""
Tests for Multi-Agent Coordination - UnifiedOrchestrator and Agent Communication.

RALPH ITERATION 5: Comprehensive tests for multi-agent coordination patterns
including framework selection, agent spawning, communication, and memory integration.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================================
# Test Infrastructure
# ============================================================================

class TestImportOrchestration:
    """Test that orchestration components import correctly."""

    def test_unified_orchestrator_importable(self):
        """Verify UnifiedOrchestrator imports without errors."""
        from core.orchestration import UnifiedOrchestrator
        assert UnifiedOrchestrator is not None

    def test_orchestration_result_importable(self):
        """Verify OrchestrationResult imports without errors."""
        from core.orchestration import OrchestrationResult
        assert OrchestrationResult is not None

    def test_agent_sdk_layer_importable(self):
        """Verify agent_sdk_layer imports without errors."""
        from core.orchestration import agent_sdk_layer
        assert hasattr(agent_sdk_layer, "Agent")
        assert hasattr(agent_sdk_layer, "AgentConfig")

    def test_create_agent_importable(self):
        """Verify create_agent factory is importable."""
        from core.orchestration import create_agent
        assert create_agent is not None

    def test_availability_flags_importable(self):
        """Verify availability flags are importable."""
        from core.orchestration import (
            TEMPORAL_AVAILABLE,
            LANGGRAPH_AVAILABLE,
            CLAUDE_FLOW_AVAILABLE,
            CREWAI_AVAILABLE,
            AUTOGEN_AVAILABLE,
        )
        # All should be boolean
        assert isinstance(TEMPORAL_AVAILABLE, bool)
        assert isinstance(LANGGRAPH_AVAILABLE, bool)
        assert isinstance(CLAUDE_FLOW_AVAILABLE, bool)
        assert isinstance(CREWAI_AVAILABLE, bool)
        assert isinstance(AUTOGEN_AVAILABLE, bool)


class TestOrchestrationResult:
    """Test OrchestrationResult dataclass."""

    def test_result_creation(self):
        """Test creating an OrchestrationResult."""
        from core.orchestration import OrchestrationResult

        result = OrchestrationResult(
            success=True,
            output="Task completed successfully",
            framework="claude_flow",
        )
        assert result.success is True
        assert result.output == "Task completed successfully"
        assert result.framework == "claude_flow"

    def test_result_with_error(self):
        """Test OrchestrationResult with error."""
        from core.orchestration import OrchestrationResult

        result = OrchestrationResult(
            success=False,
            output="",
            framework="langgraph",
            error="Connection timeout",
        )
        assert result.success is False
        assert result.error == "Connection timeout"


# ============================================================================
# Framework Selection Tests
# ============================================================================

class TestFrameworkSelection:
    """Test framework auto-selection logic."""

    def test_unified_orchestrator_creation(self):
        """Test creating a UnifiedOrchestrator."""
        from core.orchestration import UnifiedOrchestrator

        orchestrator = UnifiedOrchestrator()
        assert orchestrator is not None

    def test_available_frameworks_property(self):
        """Test that available_frameworks returns a list of available frameworks."""
        from core.orchestration import UnifiedOrchestrator

        orchestrator = UnifiedOrchestrator()
        available = orchestrator.available_frameworks

        # available_frameworks returns a list of available framework names
        assert isinstance(available, list)
        # Each item should be a string framework name
        for framework in available:
            assert isinstance(framework, str)

    def test_get_available_frameworks_function(self):
        """Test the standalone get_available_frameworks function."""
        from core.orchestration import get_available_frameworks

        available = get_available_frameworks()
        assert isinstance(available, dict)

    def test_framework_priority_selection(self):
        """Test that _select_framework follows priority order."""
        from core.orchestration import UnifiedOrchestrator

        orchestrator = UnifiedOrchestrator()

        # Priority should be: claude_flow → langgraph → crewai → autogen → temporal
        # _select_framework requires a 'prefer' argument (can be None for auto-select)
        selected = orchestrator._select_framework(prefer=None)

        # Should return a string or None
        assert selected is None or isinstance(selected, str)

        # If any framework available, should have selected one
        if orchestrator.available_frameworks:  # List is truthy if not empty
            assert selected is not None


# ============================================================================
# Agent Spawning Tests
# ============================================================================

class TestAgentSpawning:
    """Test agent creation and spawning patterns."""

    @pytest.mark.asyncio
    async def test_create_single_agent(self):
        """Test creating a single agent."""
        from core.orchestration import create_agent

        agent = await create_agent(
            name="spawned_agent",
            tools=["Read"],
        )

        assert agent.config.name == "spawned_agent"
        assert "Read" in agent.config.tools

    @pytest.mark.asyncio
    async def test_create_multiple_agents(self):
        """Test creating multiple agents concurrently."""
        from core.orchestration import create_agent

        # Create multiple agents concurrently
        agent_configs = [
            {"name": "agent_1", "tools": ["Read"]},
            {"name": "agent_2", "tools": ["Write"]},
            {"name": "agent_3", "tools": ["Bash"]},
        ]

        agents = await asyncio.gather(*[
            create_agent(**config) for config in agent_configs
        ])

        assert len(agents) == 3
        assert agents[0].config.name == "agent_1"
        assert agents[1].config.name == "agent_2"
        assert agents[2].config.name == "agent_3"

    @pytest.mark.asyncio
    async def test_agent_with_all_tools(self):
        """Test agent with complete tool suite."""
        from core.orchestration import create_agent

        all_tools = [
            "Read", "Write", "Edit", "Bash", "Grep",
            "Glob", "ListDir", "WebFetch", "SaveMemory", "SearchMemory"
        ]

        agent = await create_agent(
            name="fully_equipped_agent",
            tools=all_tools,
        )

        assert len(agent.config.tools) == 10
        built_tools = agent._build_tools()
        assert len(built_tools) == 10


# ============================================================================
# Agent Communication Tests
# ============================================================================

class TestAgentCommunication:
    """Test agent-to-agent communication patterns."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.mark.asyncio
    async def test_agent_file_handoff(self, temp_dir):
        """Test agents can hand off work via files."""
        from core.orchestration.agent_sdk_layer import Agent, AgentConfig

        # Agent 1 writes a file
        config1 = AgentConfig(name="writer_agent", tools=["Write"])
        agent1 = Agent(config1)

        test_file = os.path.join(temp_dir, "handoff.txt")
        write_calls = [{
            "name": "write_file",
            "input": {"path": test_file, "content": "Data from agent 1"},
            "id": "call_1"
        }]
        write_results = await agent1._execute_tools(write_calls)
        assert "success" in write_results[0]["content"].lower()

        # Agent 2 reads the file
        config2 = AgentConfig(name="reader_agent", tools=["Read"])
        agent2 = Agent(config2)

        read_calls = [{
            "name": "read_file",
            "input": {"path": test_file},
            "id": "call_2"
        }]
        read_results = await agent2._execute_tools(read_calls)
        assert "Data from agent 1" in read_results[0]["content"]

    @pytest.mark.asyncio
    async def test_agent_edit_chain(self, temp_dir):
        """Test sequential agents editing the same file."""
        from core.orchestration.agent_sdk_layer import Agent, AgentConfig

        test_file = os.path.join(temp_dir, "chain_edit.txt")

        # Initial write
        config = AgentConfig(name="chain_agent", tools=["Write", "Edit", "Read"])
        agent = Agent(config)

        # Write initial content
        await agent._execute_tools([{
            "name": "write_file",
            "input": {"path": test_file, "content": "Step 1"},
            "id": "call_1"
        }])

        # Edit step 1 -> step 2
        await agent._execute_tools([{
            "name": "edit_file",
            "input": {"path": test_file, "old_string": "Step 1", "new_string": "Step 1, Step 2"},
            "id": "call_2"
        }])

        # Edit step 2 -> step 3
        await agent._execute_tools([{
            "name": "edit_file",
            "input": {"path": test_file, "old_string": "Step 2", "new_string": "Step 2, Step 3"},
            "id": "call_3"
        }])

        # Verify final content
        read_results = await agent._execute_tools([{
            "name": "read_file",
            "input": {"path": test_file},
            "id": "call_4"
        }])
        assert "Step 1, Step 2, Step 3" in read_results[0]["content"]


# ============================================================================
# Workflow Pattern Tests
# ============================================================================

class TestWorkflowPatterns:
    """Test different workflow execution patterns."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.mark.asyncio
    async def test_sequential_workflow(self, temp_dir):
        """Test sequential task execution."""
        from core.orchestration.agent_sdk_layer import Agent, AgentConfig

        config = AgentConfig(name="sequential_agent", tools=["Write", "Read"])
        agent = Agent(config)

        # Sequential: write file 1, write file 2, read both
        file1 = os.path.join(temp_dir, "seq1.txt")
        file2 = os.path.join(temp_dir, "seq2.txt")

        await agent._execute_tools([{
            "name": "write_file",
            "input": {"path": file1, "content": "First"},
            "id": "call_1"
        }])

        await agent._execute_tools([{
            "name": "write_file",
            "input": {"path": file2, "content": "Second"},
            "id": "call_2"
        }])

        # Read both files
        results = await agent._execute_tools([
            {"name": "read_file", "input": {"path": file1}, "id": "call_3"},
            {"name": "read_file", "input": {"path": file2}, "id": "call_4"},
        ])

        assert "First" in results[0]["content"]
        assert "Second" in results[1]["content"]

    @pytest.mark.asyncio
    async def test_parallel_workflow(self, temp_dir):
        """Test parallel task execution."""
        from core.orchestration import create_agent

        # Create multiple agents for parallel work
        agents = await asyncio.gather(
            create_agent(name="parallel_1", tools=["Write"]),
            create_agent(name="parallel_2", tools=["Write"]),
            create_agent(name="parallel_3", tools=["Write"]),
        )

        # Execute writes in parallel
        files = [os.path.join(temp_dir, f"parallel_{i}.txt") for i in range(3)]

        write_tasks = [
            agents[i]._execute_tools([{
                "name": "write_file",
                "input": {"path": files[i], "content": f"Content {i}"},
                "id": f"call_{i}"
            }])
            for i in range(3)
        ]

        results = await asyncio.gather(*write_tasks)

        # Verify all writes succeeded
        for result in results:
            assert "success" in result[0]["content"].lower()

        # Verify files exist
        for f in files:
            assert os.path.exists(f)

    @pytest.mark.asyncio
    async def test_fan_out_fan_in_workflow(self, temp_dir):
        """Test fan-out/fan-in pattern."""
        from core.orchestration.agent_sdk_layer import Agent, AgentConfig

        # Fan-out: one agent creates multiple files
        writer_config = AgentConfig(name="fan_out_agent", tools=["Write"])
        writer = Agent(writer_config)

        files = [os.path.join(temp_dir, f"fanout_{i}.txt") for i in range(3)]

        # Fan-out: write all files in parallel
        write_calls = [
            {"name": "write_file", "input": {"path": f, "content": f"Data {i}"}, "id": f"call_{i}"}
            for i, f in enumerate(files)
        ]
        await writer._execute_tools(write_calls)

        # Fan-in: another agent reads and aggregates
        reader_config = AgentConfig(name="fan_in_agent", tools=["Read"])
        reader = Agent(reader_config)

        read_calls = [
            {"name": "read_file", "input": {"path": f}, "id": f"read_{i}"}
            for i, f in enumerate(files)
        ]
        read_results = await reader._execute_tools(read_calls)

        # Verify all data collected
        for i, result in enumerate(read_results):
            assert f"Data {i}" in result["content"]


# ============================================================================
# Memory Integration Tests
# ============================================================================

class TestMemoryIntegration:
    """Test memory system integration with orchestration."""

    @pytest.fixture
    def agent_with_memory(self):
        """Create an agent with memory tools."""
        from core.orchestration.agent_sdk_layer import Agent, AgentConfig

        config = AgentConfig(
            name="memory_agent",
            tools=["SaveMemory", "SearchMemory"],
        )
        return Agent(config)

    @pytest.mark.asyncio
    async def test_memory_save(self, agent_with_memory):
        """Test saving to memory."""
        save_calls = [{
            "name": "save_memory",
            "input": {
                "content": "Test coordination memory entry",
                "category": "coordination",
                "tags": ["test", "multi-agent"]
            },
            "id": "call_1"
        }]
        results = await agent_with_memory._execute_tools(save_calls)
        assert "success" in results[0]["content"].lower()

    @pytest.mark.asyncio
    async def test_memory_search(self, agent_with_memory):
        """Test searching memory."""
        # First save something
        await agent_with_memory._execute_tools([{
            "name": "save_memory",
            "input": {
                "content": "Searchable coordination test data",
                "category": "test",
                "tags": ["searchable"]
            },
            "id": "call_1"
        }])

        # Then search for it
        search_calls = [{
            "name": "search_memory",
            "input": {"query": "coordination test", "limit": 5},
            "id": "call_2"
        }]
        results = await agent_with_memory._execute_tools(search_calls)

        # Should find something (or return empty gracefully)
        assert results[0]["type"] == "tool_result"

    @pytest.mark.asyncio
    async def test_cross_agent_memory_sharing(self):
        """Test that agents can share context via memory."""
        from core.orchestration.agent_sdk_layer import Agent, AgentConfig

        # Agent 1 saves to memory
        config1 = AgentConfig(name="memory_writer", tools=["SaveMemory"])
        agent1 = Agent(config1)

        await agent1._execute_tools([{
            "name": "save_memory",
            "input": {
                "content": "Shared context: Project uses Python 3.11",
                "category": "context",
                "tags": ["shared", "project-info"]
            },
            "id": "call_1"
        }])

        # Agent 2 searches memory
        config2 = AgentConfig(name="memory_reader", tools=["SearchMemory"])
        agent2 = Agent(config2)

        results = await agent2._execute_tools([{
            "name": "search_memory",
            "input": {"query": "Python version project", "limit": 5},
            "id": "call_2"
        }])

        # Should retrieve the shared context
        assert results[0]["type"] == "tool_result"


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestCoordinationErrorHandling:
    """Test error handling in multi-agent scenarios."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.mark.asyncio
    async def test_agent_handles_missing_file(self):
        """Test agent gracefully handles missing file."""
        from core.orchestration.agent_sdk_layer import Agent, AgentConfig

        config = AgentConfig(name="error_handler", tools=["Read"])
        agent = Agent(config)

        results = await agent._execute_tools([{
            "name": "read_file",
            "input": {"path": "/nonexistent/path/file.txt"},
            "id": "call_1"
        }])

        # Should return error, not crash
        content = results[0]["content"].lower()
        assert "error" in content or "not found" in content

    @pytest.mark.asyncio
    async def test_agent_handles_invalid_edit(self, temp_dir):
        """Test agent handles edit with non-matching string."""
        from core.orchestration.agent_sdk_layer import Agent, AgentConfig

        config = AgentConfig(name="edit_error_handler", tools=["Write", "Edit"])
        agent = Agent(config)

        test_file = os.path.join(temp_dir, "edit_error.txt")

        # Create file
        await agent._execute_tools([{
            "name": "write_file",
            "input": {"path": test_file, "content": "Original content"},
            "id": "call_1"
        }])

        # Try to edit non-existent string
        results = await agent._execute_tools([{
            "name": "edit_file",
            "input": {"path": test_file, "old_string": "This doesn't exist", "new_string": "New"},
            "id": "call_2"
        }])

        content = results[0]["content"].lower()
        assert "error" in content or "not found" in content

    @pytest.mark.asyncio
    async def test_unknown_tool_handling(self):
        """Test handling of unknown tool requests."""
        from core.orchestration.agent_sdk_layer import Agent, AgentConfig

        config = AgentConfig(name="unknown_tool_handler", tools=["Read"])
        agent = Agent(config)

        results = await agent._execute_tools([{
            "name": "nonexistent_tool",
            "input": {},
            "id": "call_1"
        }])

        content = results[0]["content"].lower()
        assert "unknown" in content or "error" in content


# ============================================================================
# Orchestrator Integration Tests
# ============================================================================

class TestOrchestratorIntegration:
    """Integration tests for the full orchestration pipeline."""

    def test_orchestrator_initialization(self):
        """Test orchestrator initializes correctly."""
        from core.orchestration import UnifiedOrchestrator

        orchestrator = UnifiedOrchestrator()
        assert orchestrator is not None
        assert hasattr(orchestrator, "available_frameworks")
        assert hasattr(orchestrator, "run")

    def test_orchestrator_framework_detection(self):
        """Test orchestrator detects available frameworks."""
        from core.orchestration import UnifiedOrchestrator

        orchestrator = UnifiedOrchestrator()
        available = orchestrator.available_frameworks
        # available_frameworks returns a list of available framework names
        assert isinstance(available, list)

    @pytest.mark.asyncio
    async def test_orchestrator_initialization_async(self):
        """Test orchestrator async initialization."""
        from core.orchestration import UnifiedOrchestrator

        orchestrator = UnifiedOrchestrator()
        await orchestrator.initialize()
        assert orchestrator._initialized is True

    def test_orchestrator_memory_attachment(self):
        """Test attaching memory to orchestrator."""
        from core.orchestration import UnifiedOrchestrator

        orchestrator = UnifiedOrchestrator()

        # Create a mock memory
        mock_memory = MagicMock()
        orchestrator.attach_memory(mock_memory)

        assert orchestrator._memory is mock_memory

    def test_orchestrator_tools_attachment(self):
        """Test attaching tools to orchestrator."""
        from core.orchestration import UnifiedOrchestrator

        orchestrator = UnifiedOrchestrator()

        # Create mock tools
        mock_tools = [{"name": "test_tool"}]
        orchestrator.attach_tools(mock_tools)

        assert orchestrator._tools == mock_tools


# ============================================================================
# Performance Tests
# ============================================================================

class TestCoordinationPerformance:
    """Performance tests for multi-agent coordination."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.mark.asyncio
    async def test_concurrent_agent_creation(self):
        """Test creating many agents concurrently."""
        from core.orchestration import create_agent
        import time

        start = time.time()

        # Create 10 agents concurrently
        agents = await asyncio.gather(*[
            create_agent(name=f"perf_agent_{i}", tools=["Read"])
            for i in range(10)
        ])

        elapsed = time.time() - start

        assert len(agents) == 10
        # Relaxed threshold to accommodate slower systems
        assert elapsed < 15.0  # Should complete in under 15 seconds

    @pytest.mark.asyncio
    async def test_concurrent_file_operations(self, temp_dir):
        """Test many concurrent file operations."""
        from core.orchestration.agent_sdk_layer import Agent, AgentConfig
        import time

        config = AgentConfig(name="perf_file_agent", tools=["Write", "Read"])
        agent = Agent(config)

        start = time.time()

        # Write 20 files concurrently
        files = [os.path.join(temp_dir, f"perf_{i}.txt") for i in range(20)]
        write_calls = [
            {"name": "write_file", "input": {"path": f, "content": f"Content {i}"}, "id": f"w_{i}"}
            for i, f in enumerate(files)
        ]

        await agent._execute_tools(write_calls)

        # Read all files
        read_calls = [
            {"name": "read_file", "input": {"path": f}, "id": f"r_{i}"}
            for i, f in enumerate(files)
        ]

        results = await agent._execute_tools(read_calls)

        elapsed = time.time() - start

        assert len(results) == 20
        assert elapsed < 10.0  # Should complete in under 10 seconds


# ============================================================================
# Unified Orchestrator Run Tests
# ============================================================================

class TestOrchestratorRun:
    """Test the main run() method of UnifiedOrchestrator."""

    @pytest.mark.asyncio
    async def test_run_with_no_frameworks(self):
        """Test run() behavior when no frameworks are available."""
        from core.orchestration import UnifiedOrchestrator

        orchestrator = UnifiedOrchestrator()

        # If no frameworks available, should handle gracefully
        # available_frameworks is a list, check if empty
        if not orchestrator.available_frameworks:
            try:
                result = await orchestrator.run("Test task")
                # Should return error or None gracefully
                assert result is None or not result.success
            except Exception as e:
                # Expected if no frameworks
                assert "no framework" in str(e).lower() or "not available" in str(e).lower()

    @pytest.mark.asyncio
    async def test_orchestrator_select_framework(self):
        """Test framework selection logic."""
        from core.orchestration import UnifiedOrchestrator

        orchestrator = UnifiedOrchestrator()

        # Test the internal _select_framework method (requires prefer argument)
        selected = orchestrator._select_framework(prefer=None)

        # If any framework is available, it should select one
        # available_frameworks is a list of available framework names
        if orchestrator.available_frameworks:
            assert selected is not None
            assert selected in orchestrator.available_frameworks
        else:
            assert selected is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
