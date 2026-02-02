#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pydantic>=2.0.0",
#     "httpx>=0.25.0",
# ]
# ///
"""
MCP Server Manager - Dynamic Server Loading and Management

Manages Model Context Protocol (MCP) servers with:
- Dynamic discovery and loading of servers
- Health monitoring and auto-recovery
- Tool capability registration
- Server lifecycle management

Based on MCP specification 2025-06-18 patterns:
- Tools are model-controlled
- Resources provide context
- Prompts define workflows

Reference: https://modelcontextprotocol.io/specification/2025-06-18

Usage:
    from mcp_manager import MCPServerManager, ServerConfig

    manager = MCPServerManager()

    # Register a server
    manager.register_server(ServerConfig(
        name="memory",
        transport="stdio",
        command=["python", "-m", "memory_server"]
    ))

    # Start all servers
    await manager.start_all()

    # Get available tools
    tools = manager.get_all_tools()
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field


class TransportType(str, Enum):
    """MCP transport types."""
    STDIO = "stdio"              # Local process communication
    HTTP_SSE = "http+sse"        # HTTP with SSE (legacy)
    STREAMABLE_HTTP = "streamable-http"  # Modern HTTP transport


class ServerStatus(str, Enum):
    """Server lifecycle status."""
    REGISTERED = "registered"    # Config registered but not started
    STARTING = "starting"        # In process of starting
    RUNNING = "running"          # Active and healthy
    UNHEALTHY = "unhealthy"      # Running but failing health checks
    STOPPED = "stopped"          # Gracefully stopped
    FAILED = "failed"            # Failed to start or crashed


class ToolSchema(BaseModel):
    """Schema for an MCP tool."""
    name: str
    description: str
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    server_name: str = ""


class ResourceSchema(BaseModel):
    """Schema for an MCP resource."""
    uri: str
    name: str
    description: str = ""
    mime_type: str = "application/json"
    server_name: str = ""


class PromptSchema(BaseModel):
    """Schema for an MCP prompt."""
    name: str
    description: str = ""
    arguments: List[Dict[str, Any]] = Field(default_factory=list)
    server_name: str = ""


class ServerConfig(BaseModel):
    """Configuration for an MCP server."""
    name: str
    transport: TransportType = TransportType.STDIO

    # For stdio transport
    command: List[str] = Field(default_factory=list)
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)

    # For HTTP transports
    url: str = ""
    headers: Dict[str, str] = Field(default_factory=dict)

    # Settings
    auto_start: bool = True
    health_check_interval: float = 30.0
    restart_on_failure: bool = True
    max_restart_attempts: int = 3
    # V43: Timeout configuration per MCP spec best practices
    init_timeout: float = 30.0          # Timeout for initialize
    tool_call_timeout: float = 90.0     # Timeout for tool calls
    resource_read_timeout: float = 60.0  # Timeout for resource reads

    # Capabilities (discovered after connection)
    tools: List[ToolSchema] = Field(default_factory=list)
    resources: List[ResourceSchema] = Field(default_factory=list)
    prompts: List[PromptSchema] = Field(default_factory=list)


@dataclass
class ServerInstance:
    """Runtime instance of an MCP server."""
    config: ServerConfig
    status: ServerStatus = ServerStatus.REGISTERED
    process: Optional[subprocess.Popen] = None
    started_at: Optional[float] = None
    last_health_check: Optional[float] = None
    restart_count: int = 0
    error_message: str = ""


class MCPServerManager:
    """
    Manages MCP servers with dynamic loading and health monitoring.

    Features:
    - Register servers from config or discovery
    - Start/stop servers on demand
    - Monitor health and auto-restart
    - Aggregate tools from all servers
    - Route tool calls to appropriate server
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self._servers: Dict[str, ServerInstance] = {}
        self._tool_registry: Dict[str, str] = {}  # tool_name -> server_name
        self._health_task: Optional[asyncio.Task] = None
        self._callbacks: Dict[str, List[Callable]] = {
            "server_started": [],
            "server_stopped": [],
            "server_failed": [],
            "tool_registered": [],
        }

    # V43: Async context manager for proper resource management
    async def __aenter__(self) -> "MCPServerManager":
        """Start all auto-start servers on context entry."""
        await self.start_all()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop all servers and cleanup on context exit."""
        await self.stop_all()
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

    def register_server(self, config: ServerConfig) -> None:
        """Register a new MCP server configuration."""
        if config.name in self._servers:
            raise ValueError(f"Server '{config.name}' already registered")

        self._servers[config.name] = ServerInstance(config=config)

        # Pre-register known tools
        for tool in config.tools:
            tool.server_name = config.name
            self._tool_registry[tool.name] = config.name

    def unregister_server(self, name: str) -> bool:
        """Unregister a server (stops if running)."""
        if name not in self._servers:
            return False

        instance = self._servers[name]
        if instance.status == ServerStatus.RUNNING:
            self._stop_server_sync(name)

        # Remove tools from registry
        tools_to_remove = [
            tool_name for tool_name, server_name in self._tool_registry.items()
            if server_name == name
        ]
        for tool_name in tools_to_remove:
            del self._tool_registry[tool_name]

        del self._servers[name]
        return True

    def _stop_server_sync(self, name: str) -> None:
        """Synchronously stop a server."""
        instance = self._servers.get(name)
        if instance and instance.process:
            instance.process.terminate()
            try:
                instance.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                instance.process.kill()
            instance.process = None
            instance.status = ServerStatus.STOPPED

    async def start_server(self, name: str) -> bool:
        """Start a specific MCP server."""
        if name not in self._servers:
            return False

        instance = self._servers[name]
        if instance.status == ServerStatus.RUNNING:
            return True

        instance.status = ServerStatus.STARTING

        try:
            if instance.config.transport == TransportType.STDIO:
                # SEC-003 FIX: Validate command arguments to prevent injection
                cmd = instance.config.command + instance.config.args

                # Validate no shell metacharacters in command args
                dangerous_chars = set(';&|`$(){}[]<>\\"\'\n\r')
                for arg in cmd:
                    if any(c in arg for c in dangerous_chars):
                        raise ValueError(
                            f"SEC-003: Potentially dangerous characters in command argument: {arg!r}. "
                            "Command arguments must not contain shell metacharacters."
                        )

                env = {**dict(__import__('os').environ), **instance.config.env}

                instance.process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env
                )

                # V43: Wait for startup with configurable timeout
                init_timeout = instance.config.init_timeout
                start_time = time.time()
                while time.time() - start_time < init_timeout:
                    await asyncio.sleep(0.5)
                    if instance.process.poll() is not None:
                        stderr_output = ""
                        if instance.process.stderr:
                            stderr_output = instance.process.stderr.read().decode("utf-8", errors="replace")[:500]
                        raise RuntimeError(
                            f"Process exited with code {instance.process.returncode}. "
                            f"Stderr: {stderr_output}"
                        )
                    # Check if server is responsive (basic check - process still running)
                    if instance.process.poll() is None:
                        break  # Server appears to be running
                else:
                    # Timeout waiting for server to start
                    raise RuntimeError(f"Server '{name}' failed to start within {init_timeout}s timeout")

            elif instance.config.transport in (TransportType.HTTP_SSE, TransportType.STREAMABLE_HTTP):
                # For HTTP servers, just verify connectivity
                # In a real implementation, would establish connection here
                pass

            instance.status = ServerStatus.RUNNING
            instance.started_at = time.time()
            instance.last_health_check = time.time()

            # Discover capabilities
            await self._discover_capabilities(name)

            # Fire callbacks
            for callback in self._callbacks["server_started"]:
                callback(name)

            return True

        except Exception as e:
            instance.status = ServerStatus.FAILED
            instance.error_message = str(e)

            for callback in self._callbacks["server_failed"]:
                callback(name, str(e))

            return False

    async def stop_server(self, name: str) -> bool:
        """Stop a specific MCP server."""
        if name not in self._servers:
            return False

        instance = self._servers[name]
        if instance.status != ServerStatus.RUNNING:
            return True

        self._stop_server_sync(name)

        for callback in self._callbacks["server_stopped"]:
            callback(name)

        return True

    async def start_all(self) -> Dict[str, bool]:
        """Start all registered servers with auto_start=True."""
        results = {}
        for name, instance in self._servers.items():
            if instance.config.auto_start:
                results[name] = await self.start_server(name)
        return results

    async def stop_all(self) -> None:
        """Stop all running servers."""
        if self._health_task:
            self._health_task.cancel()
            self._health_task = None

        for name in list(self._servers.keys()):
            await self.stop_server(name)

    async def _discover_capabilities(self, name: str) -> None:
        """Discover server capabilities (tools, resources, prompts)."""
        instance = self._servers.get(name)
        if not instance:
            return

        # In a real implementation, would send tools/list, resources/list, prompts/list
        # For now, just register any pre-configured tools
        for tool in instance.config.tools:
            tool.server_name = name
            self._tool_registry[tool.name] = name

            for callback in self._callbacks["tool_registered"]:
                callback(tool.name, name)

    async def health_check(self, name: str) -> bool:
        """Check health of a specific server."""
        instance = self._servers.get(name)
        if not instance:
            return False

        if instance.status != ServerStatus.RUNNING:
            return False

        try:
            if instance.config.transport == TransportType.STDIO:
                # Check if process is still running
                if instance.process and instance.process.poll() is not None:
                    instance.status = ServerStatus.FAILED
                    instance.error_message = f"Process exited with code {instance.process.returncode}"
                    return False

            instance.last_health_check = time.time()
            return True

        except Exception as e:
            instance.status = ServerStatus.UNHEALTHY
            instance.error_message = str(e)
            return False

    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop."""
        while True:
            for name, instance in self._servers.items():
                if instance.status == ServerStatus.RUNNING:
                    healthy = await self.health_check(name)

                    if not healthy and instance.config.restart_on_failure:
                        if instance.restart_count < instance.config.max_restart_attempts:
                            instance.restart_count += 1
                            await self.start_server(name)

            await asyncio.sleep(10.0)  # Check every 10 seconds

    def start_health_monitor(self) -> None:
        """Start the background health monitor."""
        if self._health_task is None:
            self._health_task = asyncio.create_task(self._health_monitor_loop())

    def get_all_tools(self) -> List[ToolSchema]:
        """Get all registered tools from all servers."""
        tools = []
        for instance in self._servers.values():
            for tool in instance.config.tools:
                tools.append(tool)
        return tools

    def get_tool_server(self, tool_name: str) -> Optional[str]:
        """Get which server provides a specific tool."""
        return self._tool_registry.get(tool_name)

    def get_server_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific server."""
        instance = self._servers.get(name)
        if not instance:
            return None

        return {
            "name": name,
            "status": instance.status.value,
            "transport": instance.config.transport.value,
            "started_at": instance.started_at,
            "last_health_check": instance.last_health_check,
            "restart_count": instance.restart_count,
            "error": instance.error_message,
            "tools": [t.name for t in instance.config.tools],
            "resources": [r.name for r in instance.config.resources],
            "prompts": [p.name for p in instance.config.prompts],
        }

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all servers."""
        result: Dict[str, Dict[str, Any]] = {}
        for name in self._servers:
            status = self.get_server_status(name)
            if status is not None:
                result[name] = status
        return result

    def on(self, event: str, callback: Callable) -> None:
        """Register an event callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    @classmethod
    def from_config_file(cls, config_path: Path) -> "MCPServerManager":
        """Create manager from a .mcp.json config file."""
        manager = cls(config_path=config_path)

        if config_path.exists():
            config_data = json.loads(config_path.read_text())

            # Parse mcpServers section
            servers = config_data.get("mcpServers", {})
            for name, server_config in servers.items():
                transport = server_config.get("transport", "stdio")

                config = ServerConfig(
                    name=name,
                    transport=TransportType(transport) if transport in [t.value for t in TransportType] else TransportType.STDIO,
                    command=server_config.get("command", []),
                    args=server_config.get("args", []),
                    env=server_config.get("env", {}),
                    url=server_config.get("url", ""),
                    auto_start=server_config.get("autoStart", True),
                )

                manager.register_server(config)

        return manager

    def export_config(self) -> Dict[str, Any]:
        """Export current configuration as .mcp.json format."""
        servers = {}
        for name, instance in self._servers.items():
            config = instance.config
            server_config: Dict[str, Any] = {}

            if config.transport == TransportType.STDIO:
                server_config["command"] = config.command
                if config.args:
                    server_config["args"] = config.args
                if config.env:
                    server_config["env"] = config.env
            else:
                server_config["transport"] = config.transport.value
                server_config["url"] = config.url
                if config.headers:
                    server_config["headers"] = config.headers

            servers[name] = server_config

        return {"mcpServers": servers}


def main():
    """Demo MCP server manager."""
    print("=" * 60)
    print("MCP SERVER MANAGER DEMO")
    print("=" * 60)
    print()

    # Create manager
    manager = MCPServerManager()

    # Register some example servers
    manager.register_server(ServerConfig(
        name="memory",
        transport=TransportType.STDIO,
        command=["python", "-m", "memory_server"],
        tools=[
            ToolSchema(name="memory_store", description="Store a memory"),
            ToolSchema(name="memory_recall", description="Recall a memory"),
            ToolSchema(name="memory_search", description="Search memories"),
        ]
    ))

    manager.register_server(ServerConfig(
        name="filesystem",
        transport=TransportType.STDIO,
        command=["npx", "-y", "@anthropic/mcp-server-filesystem"],
        tools=[
            ToolSchema(name="read_file", description="Read a file"),
            ToolSchema(name="write_file", description="Write a file"),
            ToolSchema(name="list_directory", description="List directory contents"),
        ]
    ))

    manager.register_server(ServerConfig(
        name="github",
        transport=TransportType.STREAMABLE_HTTP,
        url="https://api.github.com/mcp",
        auto_start=False,  # Requires auth
        tools=[
            ToolSchema(name="create_issue", description="Create GitHub issue"),
            ToolSchema(name="list_repos", description="List repositories"),
        ]
    ))

    # Get all tools
    print("[>>] Registered Tools:")
    tools = manager.get_all_tools()
    for tool in tools:
        server = manager.get_tool_server(tool.name)
        print(f"  {tool.name} ({server}): {tool.description}")

    # Get all status
    print("\n[>>] Server Status:")
    for name, status in manager.get_all_status().items():
        if status:
            print(f"  {name}:")
            print(f"    Status: {status['status']}")
            print(f"    Transport: {status['transport']}")
            print(f"    Tools: {len(status['tools'])}")

    # Export config
    print("\n[>>] Exported Config:")
    config = manager.export_config()
    print(json.dumps(config, indent=2))

    print("\n[OK] MCP Server Manager demo complete")


if __name__ == "__main__":
    main()
