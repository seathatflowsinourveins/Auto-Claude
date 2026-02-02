#!/usr/bin/env python3
"""
MCP Server with FastMCP
Provides tools and resources via Model Context Protocol.
"""

from __future__ import annotations

import os
import sys
import json
from typing import Any, Optional
from datetime import datetime

import structlog
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment
load_dotenv()

logger = structlog.get_logger(__name__)

# Import fastmcp - the Pythonic MCP framework
try:
    from fastmcp import FastMCP, Context
    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    logger.warning("fastmcp not available - install with: pip install fastmcp")


class ToolResult(BaseModel):
    """Standardized tool execution result."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


def create_mcp_server(name: str = "unleash") -> "FastMCP":
    """
    Create and configure the MCP server.

    Args:
        name: Server name/identifier

    Returns:
        Configured FastMCP server instance
    """
    if not FASTMCP_AVAILABLE:
        raise ImportError("fastmcp required - install with: pip install fastmcp")

    # Initialize MCP server
    mcp = FastMCP(name)

    # ============================================
    # Tool: Platform Status
    # ============================================
    @mcp.tool()
    async def platform_status(ctx: Context) -> dict[str, Any]:
        """
        Get current platform status and health information.

        Returns system status, loaded SDKs, and configuration state.
        """
        logger.info("tool_called", tool="platform_status")

        status = {
            "platform": "unleash",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "environment": {
                "python_version": sys.version,
                "debug_mode": os.getenv("DEBUG_MODE", "false"),
            },
            "sdks": {
                "protocol_layer": {
                    "mcp": "available",
                    "fastmcp": "available" if FASTMCP_AVAILABLE else "not_installed",
                    "litellm": "available",
                    "anthropic": "available" if os.getenv("ANTHROPIC_API_KEY") else "not_configured",
                    "openai": "available" if os.getenv("OPENAI_API_KEY") else "not_configured",
                }
            },
            "status": "operational",
        }

        return status

    # ============================================
    # Tool: LLM Complete
    # ============================================
    @mcp.tool()
    async def llm_complete(
        ctx: Context,
        prompt: str,
        system: str = "You are a helpful assistant.",
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """
        Complete a prompt using the LLM Gateway.

        Args:
            prompt: The user prompt to complete
            system: System message for the LLM
            model: Model identifier to use
            max_tokens: Maximum tokens in response

        Returns:
            Completion result with content and metadata
        """
        logger.info("tool_called", tool="llm_complete", model=model)

        try:
            # Import gateway here to avoid circular imports
            from core.llm_gateway import LLMGateway, Message, ModelConfig, Provider

            gateway = LLMGateway()

            messages = [
                Message(role="system", content=system),
                Message(role="user", content=prompt),
            ]

            # Determine provider from model name
            provider = Provider.OPENAI if "gpt" in model.lower() else Provider.ANTHROPIC

            config = ModelConfig(
                provider=provider,
                model_id=model,
                max_tokens=max_tokens,
            )

            response = await gateway.complete(messages, config)

            return ToolResult(
                success=True,
                data={
                    "content": response.content,
                    "model": response.model,
                    "provider": response.provider.value,
                    "usage": response.usage,
                },
            ).model_dump()

        except Exception as e:
            logger.error("llm_complete_failed", error=str(e))
            return ToolResult(
                success=False,
                error=str(e),
            ).model_dump()

    # ============================================
    # Tool: Read File
    # ============================================
    @mcp.tool()
    async def read_file(ctx: Context, path: str) -> dict[str, Any]:
        """
        Read contents of a file in the workspace.

        Args:
            path: Relative or absolute path to file

        Returns:
            File contents or error
        """
        logger.info("tool_called", tool="read_file", path=path)

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            return ToolResult(
                success=True,
                data={
                    "path": path,
                    "content": content,
                    "size": len(content),
                    "lines": content.count("\n") + 1,
                },
            ).model_dump()

        except FileNotFoundError:
            return ToolResult(
                success=False,
                error=f"File not found: {path}",
            ).model_dump()
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
            ).model_dump()

    # ============================================
    # Tool: Write File
    # ============================================
    @mcp.tool()
    async def write_file(
        ctx: Context,
        path: str,
        content: str,
        create_dirs: bool = True,
    ) -> dict[str, Any]:
        """
        Write content to a file.

        Args:
            path: Path to write to
            content: Content to write
            create_dirs: Create parent directories if needed

        Returns:
            Write result with file info
        """
        logger.info("tool_called", tool="write_file", path=path)

        try:
            if create_dirs:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            return ToolResult(
                success=True,
                data={
                    "path": path,
                    "bytes_written": len(content.encode("utf-8")),
                },
            ).model_dump()

        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
            ).model_dump()

    # ============================================
    # Tool: List Directory
    # ============================================
    @mcp.tool()
    async def list_directory(
        ctx: Context,
        path: str = ".",
        recursive: bool = False,
    ) -> dict[str, Any]:
        """
        List contents of a directory.

        Args:
            path: Directory path
            recursive: Include subdirectories

        Returns:
            List of files and directories
        """
        logger.info("tool_called", tool="list_directory", path=path)

        try:
            items = []

            if recursive:
                for root, dirs, files in os.walk(path):
                    for d in dirs:
                        items.append({
                            "path": os.path.join(root, d),
                            "type": "directory",
                        })
                    for f in files:
                        full_path = os.path.join(root, f)
                        items.append({
                            "path": full_path,
                            "type": "file",
                            "size": os.path.getsize(full_path),
                        })
            else:
                for item in os.listdir(path):
                    full_path = os.path.join(path, item)
                    items.append({
                        "path": full_path,
                        "type": "directory" if os.path.isdir(full_path) else "file",
                        "size": os.path.getsize(full_path) if os.path.isfile(full_path) else None,
                    })

            return ToolResult(
                success=True,
                data={
                    "path": path,
                    "count": len(items),
                    "items": items,
                },
            ).model_dump()

        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
            ).model_dump()

    # ============================================
    # Tool: Execute Python
    # ============================================
    @mcp.tool()
    async def execute_python(ctx: Context, code: str) -> dict[str, Any]:
        """
        Execute Python code in a sandboxed environment.

        Args:
            code: Python code to execute

        Returns:
            Execution result or error
        """
        logger.info("tool_called", tool="execute_python")

        try:
            # Create a restricted globals dict
            restricted_globals: dict[str, Any] = {
                "__builtins__": {
                    "print": print,
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "list": list,
                    "dict": dict,
                    "tuple": tuple,
                    "set": set,
                    "range": range,
                    "enumerate": enumerate,
                    "zip": zip,
                    "map": map,
                    "filter": filter,
                    "sorted": sorted,
                    "sum": sum,
                    "min": min,
                    "max": max,
                    "abs": abs,
                    "round": round,
                    "isinstance": isinstance,
                    "type": type,
                    "True": True,
                    "False": False,
                    "None": None,
                },
            }

            # Capture output
            import io

            stdout_capture = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = stdout_capture

            try:
                exec(code, restricted_globals)
                output = stdout_capture.getvalue()
            finally:
                sys.stdout = old_stdout

            return ToolResult(
                success=True,
                data={
                    "output": output,
                    "globals": list(restricted_globals.keys()),
                },
            ).model_dump()

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
            ).model_dump()

    # ============================================
    # Resource: Configuration
    # ============================================
    @mcp.resource("config://platform")
    async def get_platform_config() -> str:
        """Get platform configuration as JSON."""
        config = {
            "name": "unleash",
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "features": {
                "memory": os.getenv("ENABLE_MEMORY", "true") == "true",
                "guardrails": os.getenv("ENABLE_GUARDRAILS", "true") == "true",
                "observability": os.getenv("ENABLE_OBSERVABILITY", "true") == "true",
            },
            "paths": {
                "sdk_base": os.getenv("SDK_BASE_PATH", "./sdks"),
                "stack": os.getenv("STACK_PATH", "./stack"),
                "core": os.getenv("CORE_PATH", "./core"),
                "platform": os.getenv("PLATFORM_PATH", "./platform"),
            },
        }
        return json.dumps(config, indent=2)

    # ============================================
    # Resource: SDK List
    # ============================================
    @mcp.resource("sdks://list")
    async def get_sdk_list() -> str:
        """Get list of available SDKs."""
        sdks = {
            "layer_0_protocol": [
                {"name": "mcp-python-sdk", "status": "available"},
                {"name": "fastmcp", "status": "available" if FASTMCP_AVAILABLE else "not_installed"},
                {"name": "litellm", "status": "available"},
                {"name": "anthropic", "status": "configured" if os.getenv("ANTHROPIC_API_KEY") else "not_configured"},
                {"name": "openai", "status": "configured" if os.getenv("OPENAI_API_KEY") else "not_configured"},
            ],
            "layer_1_orchestration": [
                {"name": "temporal", "status": "pending_phase_3"},
                {"name": "langgraph", "status": "pending_phase_3"},
                {"name": "crewai", "status": "pending_phase_3"},
            ],
            "layer_2_memory": [
                {"name": "letta", "status": "pending_phase_4"},
                {"name": "zep", "status": "pending_phase_4"},
                {"name": "mem0", "status": "pending_phase_4"},
            ],
        }
        return json.dumps(sdks, indent=2)

    logger.info("mcp_server_created", name=name, tools=5, resources=2)
    return mcp


# Global server instance
_server: Optional["FastMCP"] = None


def get_server() -> "FastMCP":
    """Get or create the global MCP server instance."""
    global _server
    if _server is None:
        _server = create_mcp_server()
    return _server


def run_server(transport: str = "stdio") -> None:
    """
    Run the MCP server.

    Args:
        transport: Transport type ("stdio" or "sse")
    """
    server = get_server()
    logger.info("mcp_server_starting", transport=transport)
    server.run(transport=transport)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unleash MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport protocol",
    )

    args = parser.parse_args()
    run_server(args.transport)
