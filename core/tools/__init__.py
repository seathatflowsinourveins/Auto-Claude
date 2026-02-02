#!/usr/bin/env python3
"""
Unified Tool Layer - V33 Architecture.
Provides a unified interface to tool management and execution.

This module unifies:
- Platform tool registry (file system, shell, search tools)
- MCP server tools (GitHub, database, creative tools)
- SDK integrations (GraphRAG, LlamaIndex, DSPy)

Usage:
    from core.tools import UnifiedToolLayer, create_tool_layer

    tools = create_tool_layer()

    # Search for tools
    results = await tools.search("read file")

    # Execute a tool
    result = await tools.execute("Read", {"file_path": "/path/to/file"})

    # Get tool schemas for LLM
    schemas = tools.get_schemas_for_llm(format="anthropic")
"""

from __future__ import annotations

import sys
import asyncio
from pathlib import Path
from typing import Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

import structlog

# Import types
from .types import (
    ToolCategory,
    ToolPermission,
    ToolStatus,
    ToolSource,
    ToolParameter,
    ToolSchema,
    ToolInfo,
    ToolResult,
    ToolConfig,
)

logger = structlog.get_logger(__name__)

# Use importlib to avoid conflict with Python's standard library 'platform' module
import importlib.util

_platform_path = Path(__file__).parent.parent.parent / "platform"

def _load_module_from_path(module_name: str, file_path: Path):
    """Load a module directly from file path to avoid namespace conflicts.

    Note: On Python 3.14+, we must register the module in sys.modules BEFORE
    executing it, otherwise the dataclass decorator fails when trying to
    access the module's __dict__ through sys.modules.
    """
    import sys as _sys
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    # Register module BEFORE execution (required for Python 3.14+ dataclasses)
    _sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        # Clean up on failure
        _sys.modules.pop(module_name, None)
        raise
    return module

# Try to import platform tool registry
PLATFORM_TOOLS_AVAILABLE = False
PlatformToolRegistry = None
create_tool_registry = None
PlatformToolCategory = None
PlatformToolPermission = None
PlatformToolStatus = None
PlatformToolSchema = None
PlatformToolInfo = None
PlatformToolResult = None

try:
    _tool_registry_path = _platform_path / "core" / "tool_registry.py"
    if _tool_registry_path.exists():
        _tool_registry_module = _load_module_from_path("platform_tool_registry", _tool_registry_path)
        PlatformToolRegistry = _tool_registry_module.ToolRegistry
        create_tool_registry = _tool_registry_module.create_tool_registry
        PlatformToolCategory = _tool_registry_module.ToolCategory
        PlatformToolPermission = _tool_registry_module.ToolPermission
        PlatformToolStatus = _tool_registry_module.ToolStatus
        PlatformToolSchema = _tool_registry_module.ToolSchema
        PlatformToolInfo = _tool_registry_module.ToolInfo
        PlatformToolResult = _tool_registry_module.ToolResult
        PLATFORM_TOOLS_AVAILABLE = True
        logger.info("Platform tool registry loaded via importlib")
    else:
        logger.warning(f"Platform tool registry not found at {_tool_registry_path}")
except Exception as e:
    logger.warning(f"Platform tool registry not available: {e}")

# Try to import SDK integrations
SDK_INTEGRATIONS_AVAILABLE = False
GRAPHRAG_AVAILABLE = False
LIGHTRAG_AVAILABLE = False
LLAMAINDEX_AVAILABLE = False
DSPY_AVAILABLE = False

try:
    _sdk_integrations_path = _platform_path / "core" / "sdk_integrations.py"
    if _sdk_integrations_path.exists():
        _sdk_module = _load_module_from_path("platform_sdk_integrations", _sdk_integrations_path)
        GRAPHRAG_AVAILABLE = getattr(_sdk_module, "GRAPHRAG_AVAILABLE", False)
        LIGHTRAG_AVAILABLE = getattr(_sdk_module, "LIGHTRAG_AVAILABLE", False)
        LLAMAINDEX_AVAILABLE = getattr(_sdk_module, "LLAMAINDEX_AVAILABLE", False)
        DSPY_AVAILABLE = getattr(_sdk_module, "DSPY_AVAILABLE", False)
        SDK_INTEGRATIONS_AVAILABLE = True
        logger.info("SDK integrations loaded via importlib")
    else:
        logger.warning(f"SDK integrations not found at {_sdk_integrations_path}")
except Exception as e:
    logger.warning(f"SDK integrations not available: {e}")


@dataclass
class UnifiedToolLayer:
    """
    Unified Tool Layer - Single interface to all tool backends.

    Provides:
    - Tool registration and discovery
    - Multi-source tool search
    - Permission-aware tool execution
    - LLM-compatible schema generation
    - Usage tracking and analytics

    Usage:
        tools = UnifiedToolLayer()
        await tools.initialize()

        # Search for tools
        results = await tools.search("file")

        # Execute a tool
        result = await tools.execute("Read", {"file_path": "test.py"})
    """

    config: ToolConfig = field(default_factory=ToolConfig)
    _registry: Optional[Any] = field(default=None)
    _custom_handlers: dict[str, Callable] = field(default_factory=dict)
    _initialized: bool = field(default=False)

    async def initialize(self) -> None:
        """Initialize the tool layer."""
        if self._initialized:
            return

        logger.info("unified_tool_layer_initializing")

        # Initialize platform registry if available
        if PLATFORM_TOOLS_AVAILABLE:
            try:
                self._registry = create_tool_registry(include_builtins=True)
                logger.info(
                    "platform_registry_initialized",
                    tools=len(self._registry._tools),
                )
            except Exception as e:
                logger.error("platform_registry_failed", error=str(e))

        # Register SDK tools if available
        if SDK_INTEGRATIONS_AVAILABLE and self.config.enable_sdk:
            await self._register_sdk_tools()

        self._initialized = True
        logger.info(
            "unified_tool_layer_ready",
            platform=PLATFORM_TOOLS_AVAILABLE,
            sdk=SDK_INTEGRATIONS_AVAILABLE,
        )

    async def _register_sdk_tools(self) -> None:
        """Register tools from SDK integrations."""
        if not self._registry:
            return

        # GraphRAG tools
        if GRAPHRAG_AVAILABLE:
            self._registry.register(
                schema=PlatformToolSchema(
                    name="graphrag_query",
                    description="Query a GraphRAG knowledge graph for complex relationships.",
                    parameters=[
                        {"name": "query", "type": "string", "description": "Natural language query", "required": True},
                        {"name": "search_type", "type": "string", "description": "local or global", "default": "local"},
                    ],
                ),
                category=PlatformToolCategory.ANALYSIS,
                permission=PlatformToolPermission.READ_ONLY,
                source="sdk:graphrag",
                tags=["rag", "knowledge-graph", "analysis"],
            )
            logger.info("registered_sdk_tool", tool="graphrag_query")

        # LightRAG tools
        if LIGHTRAG_AVAILABLE:
            self._registry.register(
                schema=PlatformToolSchema(
                    name="lightrag_query",
                    description="Query a LightRAG index for fast retrieval.",
                    parameters=[
                        {"name": "query", "type": "string", "description": "Search query", "required": True},
                        {"name": "mode", "type": "string", "description": "naive, local, global, or hybrid", "default": "hybrid"},
                    ],
                ),
                category=PlatformToolCategory.SEARCH,
                permission=PlatformToolPermission.READ_ONLY,
                source="sdk:lightrag",
                tags=["rag", "search", "retrieval"],
            )
            logger.info("registered_sdk_tool", tool="lightrag_query")

        # LlamaIndex tools
        if LLAMAINDEX_AVAILABLE:
            self._registry.register(
                schema=PlatformToolSchema(
                    name="llamaindex_query",
                    description="Query a LlamaIndex vector store.",
                    parameters=[
                        {"name": "query", "type": "string", "description": "Search query", "required": True},
                        {"name": "top_k", "type": "number", "description": "Number of results", "default": 5},
                    ],
                ),
                category=PlatformToolCategory.SEARCH,
                permission=PlatformToolPermission.READ_ONLY,
                source="sdk:llamaindex",
                tags=["rag", "vector", "retrieval"],
            )
            logger.info("registered_sdk_tool", tool="llamaindex_query")

    def register(
        self,
        schema: ToolSchema,
        handler: Optional[Callable] = None,
        category: ToolCategory = ToolCategory.UTILITY,
        permission: ToolPermission = ToolPermission.READ_ONLY,
        source: ToolSource = ToolSource.CUSTOM,
        tags: Optional[list[str]] = None,
    ) -> bool:
        """
        Register a custom tool.

        Args:
            schema: Tool schema definition
            handler: Optional handler function
            category: Tool category
            permission: Required permission level
            source: Tool source
            tags: Searchable tags

        Returns:
            True if registered successfully
        """
        if handler:
            self._custom_handlers[schema.name] = handler

        if self._registry and PLATFORM_TOOLS_AVAILABLE:
            # Convert to platform types
            platform_schema = PlatformToolSchema(
                name=schema.name,
                description=schema.description,
                parameters=[
                    {"name": p.name, "type": p.type, "description": p.description, "required": p.required}
                    for p in schema.parameters
                ],
            )
            platform_category = PlatformToolCategory(category.value)
            platform_permission = PlatformToolPermission(permission.value)

            return self._registry.register(
                schema=platform_schema,
                category=platform_category,
                permission=platform_permission,
                source=f"{source.value}:{schema.name}",
                handler=handler,
                tags=tags,
            )

        logger.info("registered_custom_tool", tool=schema.name)
        return True

    async def search(
        self,
        query: str,
        categories: Optional[list[ToolCategory]] = None,
        max_results: int = 10,
    ) -> list[ToolInfo]:
        """
        Search for tools matching a query.

        Args:
            query: Search query
            categories: Filter by categories
            max_results: Maximum results

        Returns:
            List of matching tools
        """
        await self.initialize()

        if self._registry and PLATFORM_TOOLS_AVAILABLE:
            platform_categories = None
            if categories:
                platform_categories = [PlatformToolCategory(c.value) for c in categories]

            results = self._registry.search(
                query=query,
                categories=platform_categories,
                max_results=max_results,
            )

            # Convert to our types
            return [
                ToolInfo(
                    schema=ToolSchema(
                        name=r.schema_.name,
                        description=r.schema_.description,
                        parameters=[
                            ToolParameter(
                                name=p.name if hasattr(p, "name") else p.get("name", ""),
                                type=p.type if hasattr(p, "type") else p.get("type", "string"),
                                description=p.description if hasattr(p, "description") else p.get("description", ""),
                                required=p.required if hasattr(p, "required") else p.get("required", False),
                            )
                            for p in r.schema_.parameters
                        ],
                    ),
                    category=ToolCategory(r.category.value),
                    permission=ToolPermission(r.permission.value),
                    status=ToolStatus(r.status.value),
                    source=ToolSource.BUILTIN if r.source == "builtin" else ToolSource.CUSTOM,
                    source_name=r.source,
                    tags=r.tags,
                    usage_count=r.usage_count,
                )
                for r in results
            ]

        return []

    async def recommend_for_task(
        self,
        task_description: str,
        max_tools: int = 5,
    ) -> list[ToolInfo]:
        """
        Recommend tools for a given task.

        Args:
            task_description: Natural language task description
            max_tools: Maximum tools to recommend

        Returns:
            List of recommended tools
        """
        await self.initialize()

        if self._registry and PLATFORM_TOOLS_AVAILABLE:
            results = self._registry.recommend_for_task(
                task_description=task_description,
                max_tools=max_tools,
            )

            # Convert to our types (similar to search)
            return [
                ToolInfo(
                    schema=ToolSchema(
                        name=r.schema_.name,
                        description=r.schema_.description,
                    ),
                    category=ToolCategory(r.category.value),
                    permission=ToolPermission(r.permission.value),
                    status=ToolStatus(r.status.value),
                    source_name=r.source,
                )
                for r in results
            ]

        return []

    async def execute(
        self,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> ToolResult:
        """
        Execute a tool with given parameters.

        Args:
            tool_name: Name of the tool
            parameters: Tool parameters

        Returns:
            ToolResult with output or error
        """
        await self.initialize()

        # Check custom handlers first
        if tool_name in self._custom_handlers:
            import time
            start = time.time()

            try:
                handler = self._custom_handlers[tool_name]
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(**parameters)
                else:
                    result = handler(**parameters)

                return ToolResult(
                    success=True,
                    output=result,
                    duration_ms=(time.time() - start) * 1000,
                )
            except Exception as e:
                return ToolResult(
                    success=False,
                    error=str(e),
                    duration_ms=(time.time() - start) * 1000,
                )

        # Try platform registry
        if self._registry and PLATFORM_TOOLS_AVAILABLE:
            platform_result = await self._registry.execute(tool_name, parameters)
            return ToolResult(
                success=platform_result.success,
                output=platform_result.output,
                error=platform_result.error,
                duration_ms=platform_result.duration_ms,
                metadata=platform_result.metadata,
            )

        return ToolResult(
            success=False,
            error=f"Tool not found: {tool_name}",
        )

    def get_schemas_for_llm(
        self,
        format: str = "anthropic",
    ) -> list[dict[str, Any]]:
        """
        Get tool schemas in LLM-compatible format.

        Args:
            format: "anthropic" or "openai"

        Returns:
            List of tool schemas
        """
        if self._registry and PLATFORM_TOOLS_AVAILABLE:
            return self._registry.get_schemas_for_llm(format=format)
        return []

    def allow_permission(self, permission: ToolPermission) -> None:
        """Allow a permission level."""
        self.config.allowed_permissions.add(permission)
        if self._registry:
            self._registry.allow_permission(PlatformToolPermission(permission.value))

    def revoke_permission(self, permission: ToolPermission) -> None:
        """Revoke a permission level."""
        self.config.allowed_permissions.discard(permission)
        if self._registry:
            self._registry.revoke_permission(PlatformToolPermission(permission.value))

    def block_tool(self, tool_name: str) -> None:
        """Block a specific tool."""
        self.config.blocked_tools.add(tool_name)
        if self._registry:
            self._registry.block_tool(tool_name)

    def unblock_tool(self, tool_name: str) -> None:
        """Unblock a specific tool."""
        self.config.blocked_tools.discard(tool_name)
        if self._registry:
            self._registry.unblock_tool(tool_name)

    def get_stats(self) -> dict[str, Any]:
        """Get tool layer statistics."""
        stats = {
            "platform_available": PLATFORM_TOOLS_AVAILABLE,
            "sdk_available": SDK_INTEGRATIONS_AVAILABLE,
            "custom_handlers": len(self._custom_handlers),
            "sdk_status": {
                "graphrag": GRAPHRAG_AVAILABLE,
                "lightrag": LIGHTRAG_AVAILABLE,
                "llamaindex": LLAMAINDEX_AVAILABLE,
                "dspy": DSPY_AVAILABLE,
            },
        }

        if self._registry and PLATFORM_TOOLS_AVAILABLE:
            registry_stats = self._registry.get_stats()
            stats.update(registry_stats)

        return stats

    def list_tools(
        self,
        category: Optional[str] = None,
        source: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        List all available tools with optional filtering.
        
        This method provides CLI compatibility for listing tools.
        
        Args:
            category: Filter by tool category
            source: Filter by source (builtin, mcp, custom, sdk)
            
        Returns:
            List of tool information dictionaries
        """
        tools = []
        
        # Get tools from platform registry
        if self._registry and PLATFORM_TOOLS_AVAILABLE:
            try:
                registry_tools = self._registry.list_all()
                for t in registry_tools:
                    tool_info = {
                        "name": t.schema_.name if hasattr(t, "schema_") else str(t),
                        "category": t.schema_.category if hasattr(t, "schema_") and hasattr(t.schema_, "category") else "general",
                        "source": "builtin",
                        "description": t.schema_.description if hasattr(t, "schema_") else "",
                    }
                    
                    # Apply filters
                    if category and tool_info["category"] != category:
                        continue
                    if source and tool_info["source"] != source:
                        continue
                        
                    tools.append(tool_info)
            except Exception as e:
                logger.warning("registry_list_failed", error=str(e))
        
        # Add SDK-integrated tools
        if SDK_INTEGRATIONS_AVAILABLE:
            sdk_tools = [
                ("graphrag", GRAPHRAG_AVAILABLE, "GraphRAG knowledge retrieval"),
                ("lightrag", LIGHTRAG_AVAILABLE, "LightRAG fast retrieval"),
                ("llamaindex", LLAMAINDEX_AVAILABLE, "LlamaIndex document processing"),
                ("dspy", DSPY_AVAILABLE, "DSPy structured generation"),
            ]
            for name, available, desc in sdk_tools:
                if available:
                    tool_info = {
                        "name": name,
                        "category": "sdk",
                        "source": "sdk",
                        "description": desc,
                    }
                    if source and source != "sdk":
                        continue
                    if category and category != "sdk":
                        continue
                    tools.append(tool_info)
        
        # Add custom handlers
        for name in self._custom_handlers.keys():
            tool_info = {
                "name": name,
                "category": "custom",
                "source": "custom",
                "description": "Custom tool handler",
            }
            if source and source != "custom":
                continue
            if category and category != "custom":
                continue
            tools.append(tool_info)
        
        return tools

    @property
    def available_tools(self) -> list[str]:
        """List names of available tools."""
        tools = list(self._custom_handlers.keys())
        if self._registry and PLATFORM_TOOLS_AVAILABLE:
            tools.extend(t.schema_.name for t in self._registry.list_all())
        return list(set(tools))


def create_tool_layer(
    config: Optional[ToolConfig] = None,
    enable_sdk: bool = True,
) -> UnifiedToolLayer:
    """
    Factory function to create a unified tool layer.

    Args:
        config: Optional configuration
        enable_sdk: Enable SDK integrations

    Returns:
        Configured UnifiedToolLayer instance
    """
    if config is None:
        config = ToolConfig(enable_sdk=enable_sdk)

    return UnifiedToolLayer(config=config)


def get_available_tool_sources() -> dict[str, bool]:
    """Get availability status of all tool sources."""
    return {
        "platform": PLATFORM_TOOLS_AVAILABLE,
        "sdk": SDK_INTEGRATIONS_AVAILABLE,
        "graphrag": GRAPHRAG_AVAILABLE,
        "lightrag": LIGHTRAG_AVAILABLE,
        "llamaindex": LLAMAINDEX_AVAILABLE,
        "dspy": DSPY_AVAILABLE,
    }


# Singleton instance for CLI compatibility
_tool_registry_instance: Optional[UnifiedToolLayer] = None


def get_tool_registry() -> UnifiedToolLayer:
    """
    Get the singleton tool registry instance.
    
    This provides CLI compatibility by returning a cached tool layer.
    The instance is created on first access and reused for subsequent calls.
    
    Returns:
        UnifiedToolLayer: The singleton tool registry
    """
    global _tool_registry_instance
    if _tool_registry_instance is None:
        _tool_registry_instance = create_tool_layer()
    return _tool_registry_instance


# Export all public symbols
__all__ = [
    # Core classes
    "UnifiedToolLayer",

    # Types
    "ToolCategory",
    "ToolPermission",
    "ToolStatus",
    "ToolSource",
    "ToolParameter",
    "ToolSchema",
    "ToolInfo",
    "ToolResult",
    "ToolConfig",

    # Factory functions
    "create_tool_layer",
    "get_available_tool_sources",
    "get_tool_registry",

    # Availability flags
    "PLATFORM_TOOLS_AVAILABLE",
    "SDK_INTEGRATIONS_AVAILABLE",
    "GRAPHRAG_AVAILABLE",
    "LIGHTRAG_AVAILABLE",
    "LLAMAINDEX_AVAILABLE",
    "DSPY_AVAILABLE",
]


if __name__ == "__main__":
    async def main():
        """Test the unified tool layer."""
        print("Tool Layer Status")
        print("=" * 40)

        status = get_available_tool_sources()
        for source, available in status.items():
            symbol = "[+]" if available else "[X]"
            print(f"  {symbol} {source}")

        print()

        tools = create_tool_layer()
        await tools.initialize()

        print(f"Available tools: {len(tools.available_tools)}")

        # Test search
        results = await tools.search("read")
        print(f"Search 'read': {len(results)} results")

        # Test stats
        stats = tools.get_stats()
        print(f"Stats: {stats}")

    asyncio.run(main())
