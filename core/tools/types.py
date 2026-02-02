#!/usr/bin/env python3
"""
Tool Layer Types - V33 Architecture.
Shared types for the unified tool layer.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field


class ToolCategory(str, Enum):
    """Categories of tools for organization and discovery."""

    FILE_SYSTEM = "file_system"        # Read, write, edit files
    SHELL = "shell"                    # Command execution
    SEARCH = "search"                  # Web search, code search
    MEMORY = "memory"                  # State persistence
    NETWORK = "network"                # HTTP, WebSocket
    DATABASE = "database"              # SQL, NoSQL
    CREATIVE = "creative"              # Image, video, audio
    ANALYSIS = "analysis"              # Data processing
    COMMUNICATION = "communication"    # Notifications, messaging
    UTILITY = "utility"                # Time, calculator, etc.
    MCP = "mcp"                        # MCP server tools
    SDK = "sdk"                        # SDK integrations


class ToolPermission(str, Enum):
    """Permission levels for tools."""

    READ_ONLY = "read_only"            # Only read operations
    READ_WRITE = "read_write"          # Read and write
    EXECUTE = "execute"                # Can execute code/commands
    NETWORK = "network"                # Can make network requests
    ELEVATED = "elevated"              # Requires explicit approval


class ToolStatus(str, Enum):
    """Status of a tool in the registry."""

    AVAILABLE = "available"            # Ready to use
    LOADING = "loading"                # Being loaded
    DISABLED = "disabled"              # Explicitly disabled
    ERROR = "error"                    # Failed to load
    DEPRECATED = "deprecated"          # Will be removed


class ToolSource(str, Enum):
    """Source of a tool."""

    BUILTIN = "builtin"                # Built-in core tools
    MCP = "mcp"                        # MCP server
    SDK = "sdk"                        # SDK integration
    PLUGIN = "plugin"                  # Plugin system
    CUSTOM = "custom"                  # User-defined


class ToolParameter(BaseModel):
    """Schema for a tool parameter."""

    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = False
    default: Optional[Any] = None
    enum: Optional[list[Any]] = None
    items: Optional[dict[str, Any]] = None  # For arrays


class ToolSchema(BaseModel):
    """Complete schema for a tool."""

    name: str
    description: str
    parameters: list[ToolParameter] = Field(default_factory=list)
    returns: Optional[str] = None
    examples: list[dict[str, Any]] = Field(default_factory=list)

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties: dict[str, dict[str, Any]] = {}
        required: list[str] = []

        for param in self.parameters:
            prop: dict[str, Any] = {"type": param.type, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            if param.items:
                prop["items"] = param.items
            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic tool format."""
        properties: dict[str, dict[str, Any]] = {}
        required: list[str] = []

        for param in self.parameters:
            prop: dict[str, Any] = {"type": param.type, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            if param.items:
                prop["items"] = param.items
            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class ToolInfo(BaseModel):
    """Complete information about a registered tool."""

    schema_: ToolSchema = Field(alias="schema")
    category: ToolCategory = ToolCategory.UTILITY
    permission: ToolPermission = ToolPermission.READ_ONLY
    status: ToolStatus = ToolStatus.AVAILABLE
    source: ToolSource = ToolSource.BUILTIN
    source_name: str = ""  # e.g., "mcp:github", "sdk:graphrag"
    version: str = "1.0.0"
    tags: list[str] = Field(default_factory=list)
    usage_count: int = 0
    last_used: Optional[datetime] = None
    error_message: Optional[str] = None

    class Config:
        populate_by_name = True


@dataclass
class ToolResult:
    """Result from executing a tool."""

    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolConfig:
    """Configuration for the tool layer."""

    # Permission settings
    allowed_permissions: set[ToolPermission] = field(
        default_factory=lambda: {ToolPermission.READ_ONLY}
    )
    blocked_tools: set[str] = field(default_factory=set)

    # Discovery settings
    enable_mcp: bool = True
    enable_sdk: bool = True
    enable_plugins: bool = True

    # Caching
    cache_tool_results: bool = False
    cache_ttl_seconds: int = 300

    # Logging
    log_executions: bool = True
    log_level: str = "INFO"
