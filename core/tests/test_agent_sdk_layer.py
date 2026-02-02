"""
Tests for Agent SDK Layer - Tool definitions and execution.

Tests the expanded tool suite including:
- Read/Write/Edit file operations
- Bash command execution
- Grep/Glob file search
- Web fetch
- Memory operations
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest


class TestImportAvailability:
    """Test that agent_sdk_layer imports correctly."""

    def test_agent_sdk_layer_importable(self):
        """Verify module imports without errors."""
        from core.orchestration import agent_sdk_layer
        assert hasattr(agent_sdk_layer, "Agent")
        assert hasattr(agent_sdk_layer, "AgentConfig")
        assert hasattr(agent_sdk_layer, "AgentResult")

    def test_availability_flags(self):
        """Check SDK availability flags are set."""
        from core.orchestration.agent_sdk_layer import (
            CLAUDE_AGENT_SDK_AVAILABLE,
            ANTHROPIC_AVAILABLE,
        )
        # At least one should be available for tests to work
        assert CLAUDE_AGENT_SDK_AVAILABLE or ANTHROPIC_AVAILABLE


class TestAgentConfig:
    """Test AgentConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from core.orchestration.agent_sdk_layer import AgentConfig

        config = AgentConfig(name="test_agent")
        assert config.name == "test_agent"
        assert config.model == "claude-sonnet-4-20250514"
        assert config.tools == []
        assert config.max_tokens == 4096
        assert config.temperature == 0.7

    def test_custom_config(self):
        """Test custom configuration values."""
        from core.orchestration.agent_sdk_layer import AgentConfig

        config = AgentConfig(
            name="custom_agent",
            model="claude-opus-4-5-20251101",
            tools=["Read", "Write", "Bash"],
            max_tokens=8192,
            temperature=0.5,
        )
        assert config.name == "custom_agent"
        assert config.model == "claude-opus-4-5-20251101"
        assert config.tools == ["Read", "Write", "Bash"]


class TestToolDefinitions:
    """Test tool definition building."""

    def test_build_basic_tools(self):
        """Test building basic Read/Write/Bash tools."""
        from core.orchestration.agent_sdk_layer import Agent, AgentConfig

        config = AgentConfig(name="test", tools=["Read", "Write", "Bash"])
        agent = Agent(config)
        tools = agent._build_tools()

        assert len(tools) == 3
        tool_names = [t["name"] for t in tools]
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "execute_bash" in tool_names

    def test_build_all_tools(self):
        """Test building all available tools."""
        from core.orchestration.agent_sdk_layer import Agent, AgentConfig

        all_tools = [
            "Read", "Write", "Edit", "Bash", "Grep",
            "Glob", "ListDir", "WebFetch", "SaveMemory", "SearchMemory"
        ]
        config = AgentConfig(name="test", tools=all_tools)
        agent = Agent(config)
        tools = agent._build_tools()

        assert len(tools) == 10
        tool_names = [t["name"] for t in tools]
        assert "read_file" in tool_names
        assert "edit_file" in tool_names
        assert "grep_files" in tool_names
        assert "glob_files" in tool_names
        assert "list_directory" in tool_names
        assert "web_fetch" in tool_names
        assert "save_memory" in tool_names
        assert "search_memory" in tool_names

    def test_tool_schema_structure(self):
        """Test that tool schemas have required structure."""
        from core.orchestration.agent_sdk_layer import Agent, AgentConfig

        config = AgentConfig(name="test", tools=["Read"])
        agent = Agent(config)
        tools = agent._build_tools()

        assert len(tools) == 1
        tool = tools[0]
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"
        assert "properties" in tool["input_schema"]

    def test_unknown_tool_warning(self):
        """Test that unknown tools are handled gracefully."""
        from core.orchestration.agent_sdk_layer import Agent, AgentConfig

        config = AgentConfig(name="test", tools=["Read", "UnknownTool"])
        agent = Agent(config)
        tools = agent._build_tools()

        # Should only build the known tool
        assert len(tools) == 1
        assert tools[0]["name"] == "read_file"


class TestToolExecution:
    """Test tool execution methods."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def agent(self):
        """Create an agent with all tools."""
        from core.orchestration.agent_sdk_layer import Agent, AgentConfig

        all_tools = [
            "Read", "Write", "Edit", "Bash", "Grep",
            "Glob", "ListDir", "SaveMemory", "SearchMemory"
        ]
        config = AgentConfig(name="test_agent", tools=all_tools)
        return Agent(config)

    @pytest.mark.asyncio
    async def test_read_file(self, agent, temp_dir):
        """Test reading a file."""
        # Create test file
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("Hello, World!")

        # Execute read tool
        tool_calls = [{"name": "read_file", "input": {"path": test_file}, "id": "call_1"}]
        results = await agent._execute_tools(tool_calls)

        assert len(results) == 1
        assert results[0]["type"] == "tool_result"
        assert "Hello, World!" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_read_file_not_found(self, agent, temp_dir):
        """Test reading a non-existent file."""
        tool_calls = [{"name": "read_file", "input": {"path": "/nonexistent/file.txt"}, "id": "call_1"}]
        results = await agent._execute_tools(tool_calls)

        assert len(results) == 1
        assert "error" in results[0]["content"].lower() or "not found" in results[0]["content"].lower()

    @pytest.mark.asyncio
    async def test_write_file(self, agent, temp_dir):
        """Test writing a file."""
        test_file = os.path.join(temp_dir, "output.txt")

        tool_calls = [{"name": "write_file", "input": {"path": test_file, "content": "Test content"}, "id": "call_1"}]
        results = await agent._execute_tools(tool_calls)

        assert len(results) == 1
        assert "success" in results[0]["content"].lower()

        # Verify file was written
        with open(test_file) as f:
            assert f.read() == "Test content"

    @pytest.mark.asyncio
    async def test_edit_file(self, agent, temp_dir):
        """Test editing a file with search/replace."""
        test_file = os.path.join(temp_dir, "edit_test.txt")
        with open(test_file, "w") as f:
            f.write("Hello, World!")

        tool_calls = [{
            "name": "edit_file",
            "input": {"path": test_file, "old_string": "World", "new_string": "Universe"},
            "id": "call_1"
        }]
        results = await agent._execute_tools(tool_calls)

        assert len(results) == 1
        assert "success" in results[0]["content"].lower()

        # Verify edit was applied
        with open(test_file) as f:
            assert f.read() == "Hello, Universe!"

    @pytest.mark.asyncio
    async def test_edit_file_string_not_found(self, agent, temp_dir):
        """Test editing with non-existent string."""
        test_file = os.path.join(temp_dir, "edit_test2.txt")
        with open(test_file, "w") as f:
            f.write("Hello, World!")

        tool_calls = [{
            "name": "edit_file",
            "input": {"path": test_file, "old_string": "NotFound", "new_string": "New"},
            "id": "call_1"
        }]
        results = await agent._execute_tools(tool_calls)

        assert len(results) == 1
        assert "error" in results[0]["content"].lower() or "not found" in results[0]["content"].lower()

    @pytest.mark.asyncio
    async def test_execute_bash(self, agent):
        """Test bash command execution."""
        # Use a simple cross-platform command (no timeout to avoid context issues)
        tool_calls = [{"name": "execute_bash", "input": {"command": "echo Hello", "timeout": 30}, "id": "call_1"}]

        # Run within an asyncio task to allow wait_for to work
        async def run_tool():
            return await agent._execute_tools(tool_calls)

        results = await asyncio.create_task(run_tool())

        assert len(results) == 1
        # Check for Hello or success (bash on Windows uses different output)
        content = results[0]["content"]
        assert "Hello" in content or "stdout" in content

    @pytest.mark.asyncio
    async def test_grep_files(self, agent, temp_dir):
        """Test grep/search in files."""
        # Create test files
        test_file1 = os.path.join(temp_dir, "test1.py")
        test_file2 = os.path.join(temp_dir, "test2.py")
        with open(test_file1, "w") as f:
            f.write("def hello():\n    print('hello')\n")
        with open(test_file2, "w") as f:
            f.write("def goodbye():\n    print('goodbye')\n")

        tool_calls = [{
            "name": "grep_files",
            "input": {"pattern": "hello", "path": temp_dir, "file_pattern": "*.py"},
            "id": "call_1"
        }]
        results = await agent._execute_tools(tool_calls)

        assert len(results) == 1
        assert "hello" in results[0]["content"].lower()
        assert "test1.py" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_glob_files(self, agent, temp_dir):
        """Test glob file finding."""
        # Create test files
        os.makedirs(os.path.join(temp_dir, "subdir"))
        for name in ["a.py", "b.py", "c.txt"]:
            Path(os.path.join(temp_dir, name)).touch()
        Path(os.path.join(temp_dir, "subdir", "d.py")).touch()

        tool_calls = [{
            "name": "glob_files",
            "input": {"pattern": "**/*.py", "path": temp_dir},
            "id": "call_1"
        }]
        results = await agent._execute_tools(tool_calls)

        assert len(results) == 1
        content = results[0]["content"]
        assert "a.py" in content
        assert "b.py" in content
        assert "d.py" in content
        assert "c.txt" not in content  # txt files should be excluded

    @pytest.mark.asyncio
    async def test_list_directory(self, agent, temp_dir):
        """Test directory listing."""
        # Create test structure
        os.makedirs(os.path.join(temp_dir, "subdir"))
        Path(os.path.join(temp_dir, "file.txt")).touch()
        Path(os.path.join(temp_dir, "subdir", "nested.txt")).touch()

        tool_calls = [{
            "name": "list_directory",
            "input": {"path": temp_dir},
            "id": "call_1"
        }]
        results = await agent._execute_tools(tool_calls)

        assert len(results) == 1
        content = results[0]["content"]
        assert "file.txt" in content
        assert "subdir" in content

    @pytest.mark.asyncio
    async def test_save_and_search_memory(self, agent):
        """Test memory save and search."""
        # Save to memory
        save_calls = [{
            "name": "save_memory",
            "input": {
                "content": "Test memory entry for unit testing",
                "category": "learning",
                "tags": ["test", "unit"]
            },
            "id": "call_1"
        }]
        save_results = await agent._execute_tools(save_calls)
        assert "success" in save_results[0]["content"].lower()

        # Search memory
        search_calls = [{
            "name": "search_memory",
            "input": {"query": "unit testing", "limit": 5},
            "id": "call_2"
        }]
        search_results = await agent._execute_tools(search_calls)
        assert "test memory entry" in search_results[0]["content"].lower()

    @pytest.mark.asyncio
    async def test_unknown_tool_execution(self, agent):
        """Test handling of unknown tool in execution."""
        tool_calls = [{"name": "unknown_tool", "input": {}, "id": "call_1"}]
        results = await agent._execute_tools(tool_calls)

        assert len(results) == 1
        assert "unknown" in results[0]["content"].lower() or "error" in results[0]["content"].lower()


class TestAgentCreation:
    """Test agent creation functions."""

    @pytest.mark.asyncio
    async def test_create_agent(self):
        """Test create_agent factory function."""
        from core.orchestration.agent_sdk_layer import create_agent

        agent = await create_agent(
            name="factory_agent",
            model="claude-sonnet-4-20250514",
            tools=["Read", "Write"],
        )

        assert agent.config.name == "factory_agent"
        assert agent.config.tools == ["Read", "Write"]


class TestAgentResult:
    """Test AgentResult dataclass."""

    def test_agent_result_creation(self):
        """Test creating AgentResult."""
        from core.orchestration.agent_sdk_layer import AgentResult

        # AgentResult fields: output, tool_calls, messages, metadata, success, error
        result = AgentResult(
            output="Task completed successfully",
            tool_calls=[{"name": "read_file", "input": {"path": "test.txt"}}],
            messages=[{"role": "assistant", "content": "Done"}],
            success=True,
        )

        assert result.success is True
        assert result.output == "Task completed successfully"
        assert len(result.tool_calls) == 1
        assert len(result.messages) == 1
        assert result.error is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
