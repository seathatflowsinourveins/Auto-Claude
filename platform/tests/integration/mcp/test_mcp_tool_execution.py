"""
MCP Tool Execution Tests

Tests for MCP tool execution including:
- Sync and async tool execution
- Parameter handling
- Error handling
- Context passing
- Timeout handling
"""

import pytest
import asyncio
from typing import Dict, Any

# Import MCP components
try:
    from hooks.hook_utils import (
        FastMCPServer,
        FastMCPTool,
        MCPContext,
        LifespanContext,
    )
except ImportError:
    pytest.skip("MCP utilities not available", allow_module_level=True)


class TestMCPToolExecution:
    """Tests for MCP tool execution."""

    @pytest.fixture
    def server(self):
        """Create a FastMCP server for testing."""
        return FastMCPServer("test-server", version="1.0.0")

    @pytest.mark.asyncio
    async def test_execute_sync_tool(self, server):
        """Execute a synchronous tool."""
        @server.tool()
        def add(a: int, b: int) -> int:
            return a + b

        result = await server.call_tool("add", {"a": 5, "b": 3})
        assert result == 8

    @pytest.mark.asyncio
    async def test_execute_async_tool(self, server):
        """Execute an asynchronous tool."""
        @server.tool()
        async def async_multiply(x: int, y: int) -> int:
            await asyncio.sleep(0.01)
            return x * y

        result = await server.call_tool("async_multiply", {"x": 4, "y": 7})
        assert result == 28

    @pytest.mark.asyncio
    async def test_execute_with_string_parameters(self, server):
        """Execute tool with string parameters."""
        @server.tool()
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        result = await server.call_tool("greet", {"name": "World"})
        assert result == "Hello, World!"

    @pytest.mark.asyncio
    async def test_execute_with_dict_parameter(self, server):
        """Execute tool with dictionary parameter."""
        @server.tool()
        def process_config(config: dict) -> str:
            return f"Keys: {list(config.keys())}"

        result = await server.call_tool("process_config", {
            "config": {"key1": "value1", "key2": "value2"}
        })
        assert "key1" in result
        assert "key2" in result

    @pytest.mark.asyncio
    async def test_execute_with_list_parameter(self, server):
        """Execute tool with list parameter."""
        @server.tool()
        def sum_list(numbers: list) -> int:
            return sum(numbers)

        result = await server.call_tool("sum_list", {"numbers": [1, 2, 3, 4, 5]})
        assert result == 15

    @pytest.mark.asyncio
    async def test_execute_with_optional_parameter(self, server):
        """Execute tool with optional parameter."""
        @server.tool()
        def greet_optional(name: str, prefix: str = "Hello") -> str:
            return f"{prefix}, {name}!"

        # With default
        result1 = await server.call_tool("greet_optional", {"name": "World"})
        assert result1 == "Hello, World!"

        # With custom value
        result2 = await server.call_tool("greet_optional", {"name": "World", "prefix": "Hi"})
        assert result2 == "Hi, World!"


class TestMCPToolErrorHandling:
    """Tests for MCP tool error handling."""

    @pytest.fixture
    def server(self):
        return FastMCPServer("test-server")

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, server):
        """Executing unknown tool should raise error."""
        with pytest.raises(ValueError, match="Unknown tool"):
            await server.call_tool("nonexistent", {})

    @pytest.mark.asyncio
    async def test_execute_with_missing_required_param(self, server):
        """Missing required parameter should raise error."""
        @server.tool()
        def required_param(value: str) -> str:
            return value

        with pytest.raises((TypeError, KeyError)):
            await server.call_tool("required_param", {})

    @pytest.mark.asyncio
    async def test_tool_that_raises_exception(self, server):
        """Tool that raises exception should propagate."""
        @server.tool()
        def raise_error() -> None:
            raise RuntimeError("Intentional error")

        with pytest.raises(RuntimeError, match="Intentional error"):
            await server.call_tool("raise_error", {})


class TestFastMCPToolDefinition:
    """Tests for FastMCPTool definition."""

    def test_tool_creation(self):
        """Create a tool definition."""
        tool = FastMCPTool(
            name="test_tool",
            description="A test tool",
            handler=lambda x: x * 2
        )

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"

    def test_tool_schema_generation(self):
        """Tool should generate valid schema."""
        tool = FastMCPTool(
            name="calculate",
            description="Perform calculation",
            handler=lambda a, b: a + b
        )

        schema = tool.to_schema()
        assert schema["name"] == "calculate"
        assert schema["description"] == "Perform calculation"
        assert "inputSchema" in schema

    @pytest.mark.asyncio
    async def test_tool_execute_sync(self):
        """Execute sync handler through tool."""
        tool = FastMCPTool(
            name="double",
            description="Double a value",
            handler=lambda x: x * 2
        )

        result = await tool.execute({"x": 5})
        assert result == 10


class TestMCPContext:
    """Tests for MCP context handling."""

    def test_context_creation(self):
        """Create MCP context."""
        server = FastMCPServer("test")
        context = MCPContext(request_id="req-123", server=server)

        assert context.request_id == "req-123"
        assert context.server == server

    def test_context_with_lifespan(self):
        """Context should access lifespan state."""
        server = FastMCPServer("test")
        lifespan = LifespanContext(server=server)
        lifespan.set("db_connection", "mock_connection")

        context = MCPContext(
            request_id="req-456",
            server=server,
            lifespan=lifespan
        )

        assert context.get_lifespan_state("db_connection") == "mock_connection"

    def test_context_lifespan_state_missing(self):
        """Missing lifespan state should return None."""
        server = FastMCPServer("test")
        context = MCPContext(request_id="req-789", server=server)

        assert context.get_lifespan_state("nonexistent") is None


class TestLifespanContext:
    """Tests for lifespan context management."""

    def test_lifespan_set_get(self):
        """Set and get lifespan state."""
        server = FastMCPServer("test")
        lifespan = LifespanContext(server=server)

        lifespan.set("key", "value")
        assert lifespan.get("key") == "value"

    def test_lifespan_get_default(self):
        """Get with default for missing key."""
        server = FastMCPServer("test")
        lifespan = LifespanContext(server=server)

        assert lifespan.get("missing", "default") == "default"

    @pytest.mark.asyncio
    async def test_lifespan_context_manager(self):
        """Lifespan as async context manager."""
        server = FastMCPServer("test")
        lifespan = LifespanContext(server=server)

        async with lifespan as ctx:
            ctx.set("temp_key", "temp_value")
            assert ctx.get("temp_key") == "temp_value"

        # State should be cleared after exit
        assert lifespan.get("temp_key") is None


class TestMCPToolWithContext:
    """Tests for tools that use context."""

    @pytest.fixture
    def server(self):
        return FastMCPServer("test-server")

    @pytest.mark.asyncio
    async def test_tool_accesses_lifespan_state(self, server):
        """Tool should be able to access lifespan state."""
        lifespan = LifespanContext(server=server)
        lifespan.set("counter", 0)

        # Track call count in lifespan state
        @server.tool()
        def increment_counter() -> int:
            current = lifespan.get("counter", 0)
            lifespan.set("counter", current + 1)
            return lifespan.get("counter")

        result1 = await server.call_tool("increment_counter", {})
        assert result1 == 1

        result2 = await server.call_tool("increment_counter", {})
        assert result2 == 2


class TestMCPToolRegistration:
    """Tests for tool registration."""

    @pytest.fixture
    def server(self):
        return FastMCPServer("test-server")

    def test_tool_decorator_registers(self, server):
        """Tool decorator should register tool."""
        @server.tool()
        def my_tool() -> str:
            return "result"

        assert "my_tool" in server.list_tools()

    def test_tool_with_custom_name(self, server):
        """Tool with custom name should register correctly."""
        @server.tool(name="custom_name", description="Custom tool")
        def original_name() -> str:
            return "result"

        assert "custom_name" in server.list_tools()
        assert server._tools["custom_name"].description == "Custom tool"

    def test_multiple_tools_register(self, server):
        """Multiple tools should all register."""
        @server.tool()
        def tool_a() -> str:
            return "a"

        @server.tool()
        def tool_b() -> str:
            return "b"

        @server.tool()
        def tool_c() -> str:
            return "c"

        tools = server.list_tools()
        assert "tool_a" in tools
        assert "tool_b" in tools
        assert "tool_c" in tools


class TestMCPConcurrentExecution:
    """Tests for concurrent tool execution."""

    @pytest.fixture
    def server(self):
        return FastMCPServer("test-server")

    @pytest.mark.asyncio
    async def test_concurrent_tool_calls(self, server):
        """Multiple concurrent tool calls should work."""
        @server.tool()
        async def slow_add(a: int, b: int) -> int:
            await asyncio.sleep(0.05)
            return a + b

        # Run multiple calls concurrently
        tasks = [
            server.call_tool("slow_add", {"a": i, "b": i})
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)
        assert results == [0, 2, 4, 6, 8]

    @pytest.mark.asyncio
    async def test_different_tools_concurrent(self, server):
        """Different tools should execute concurrently."""
        @server.tool()
        async def tool_1() -> str:
            await asyncio.sleep(0.02)
            return "result_1"

        @server.tool()
        async def tool_2() -> str:
            await asyncio.sleep(0.02)
            return "result_2"

        @server.tool()
        async def tool_3() -> str:
            await asyncio.sleep(0.02)
            return "result_3"

        results = await asyncio.gather(
            server.call_tool("tool_1", {}),
            server.call_tool("tool_2", {}),
            server.call_tool("tool_3", {}),
        )

        assert results == ["result_1", "result_2", "result_3"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
