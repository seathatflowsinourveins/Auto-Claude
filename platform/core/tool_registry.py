"""
UAP Tool Registry - Dynamic tool loading and management.

Implements the MCP tool registry pattern:
- Centralized catalog of available tools
- Dynamic tool discovery and loading
- Tool schema validation
- Permission and capability management
- Tool search and recommendation

Based on MCP Registry patterns (registry.modelcontextprotocol.io).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Enums
# =============================================================================

class ToolCategory(str, Enum):
    """Categories of tools."""

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


# =============================================================================
# Data Models
# =============================================================================

class ToolParameter(BaseModel):
    """Schema for a tool parameter."""

    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = False
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None
    items: Optional[Dict[str, Any]] = None  # For arrays


class ToolSchema(BaseModel):
    """Complete schema for a tool."""

    name: str
    description: str
    parameters: List[ToolParameter] = Field(default_factory=list)
    returns: Optional[str] = None
    examples: List[Dict[str, Any]] = Field(default_factory=list)

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties: Dict[str, Dict[str, Any]] = {}
        required: List[str] = []

        for param in self.parameters:
            prop: Dict[str, Any] = {"type": param.type, "description": param.description}
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

    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic tool format."""
        properties: Dict[str, Dict[str, Any]] = {}
        required: List[str] = []

        for param in self.parameters:
            prop: Dict[str, Any] = {"type": param.type, "description": param.description}
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

    model_config = ConfigDict(populate_by_name=True)

    schema_: ToolSchema = Field(alias="schema")
    category: ToolCategory = ToolCategory.UTILITY
    permission: ToolPermission = ToolPermission.READ_ONLY
    status: ToolStatus = ToolStatus.AVAILABLE
    source: str = "builtin"  # "builtin", "mcp:<server>", "plugin:<name>"
    version: str = "1.0.0"
    tags: List[str] = Field(default_factory=list)
    usage_count: int = 0
    last_used: Optional[datetime] = None
    error_message: Optional[str] = None


# =============================================================================
# Tool Executor Interface
# =============================================================================

@dataclass
class ToolResult:
    """Result from executing a tool."""

    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolExecutor:
    """Interface for tool execution."""

    async def execute(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
    ) -> ToolResult:
        """Execute a tool with given parameters."""
        _ = (tool_name, parameters)  # Interface method - subclasses implement
        raise NotImplementedError


class LocalToolExecutor(ToolExecutor):
    """Executes locally registered tools."""

    def __init__(self):
        self._handlers: Dict[str, Callable] = {}

    def register_handler(
        self,
        tool_name: str,
        handler: Callable,
    ) -> None:
        """Register a handler function for a tool."""
        self._handlers[tool_name] = handler

    async def execute(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
    ) -> ToolResult:
        """Execute a tool."""
        import time

        start = time.time()

        handler = self._handlers.get(tool_name)
        if not handler:
            return ToolResult(
                success=False,
                error=f"No handler for tool: {tool_name}",
            )

        try:
            if inspect.iscoroutinefunction(handler):
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


# =============================================================================
# Tool Registry
# =============================================================================

class ToolRegistry:
    """
    Central registry for tool discovery and management.

    Implements the MCP Registry pattern with:
    - Dynamic tool registration
    - Tool search and recommendation
    - Permission management
    - Usage tracking

    V13 OPTIMIZATION: Inverted indices for O(1) candidate lookups
    - category_index: category → tool names
    - tag_index: tag → tool names
    - keyword_index: keyword → tool names
    Expected improvement: 40-60% search latency reduction for large registries
    """

    def __init__(self):
        self._tools: Dict[str, ToolInfo] = {}
        self._executors: Dict[str, ToolExecutor] = {}
        self._local_executor = LocalToolExecutor()
        self._allowed_permissions: Set[ToolPermission] = {
            ToolPermission.READ_ONLY,
        }
        self._blocked_tools: Set[str] = set()

        # V13: Inverted indices for O(1) candidate lookups
        self._category_index: Dict[ToolCategory, Set[str]] = {cat: set() for cat in ToolCategory}
        self._tag_index: Dict[str, Set[str]] = {}  # tag_lower → tool_names
        self._keyword_index: Dict[str, Set[str]] = {}  # keyword_lower → tool_names

    # -------------------------------------------------------------------------
    # V13: Index Management Helpers
    # -------------------------------------------------------------------------

    def _extract_keywords(self, schema: ToolSchema) -> Set[str]:
        """
        V13: Extract searchable keywords from tool schema.

        Extracts from:
        - Tool name (split by underscore, camelCase)
        - Description words (3+ chars)
        - Parameter names
        """
        keywords: Set[str] = set()

        # From name: split by underscore and camelCase
        name_parts = re.split(r'[_\s]', schema.name.lower())
        for part in name_parts:
            if len(part) >= 3:
                keywords.add(part)
        # CamelCase splitting
        camel_parts = re.findall(r'[A-Z]?[a-z]{3,}', schema.name)
        for part in camel_parts:
            keywords.add(part.lower())

        # From description: extract words 3+ chars
        desc_words = re.findall(r'\b\w{3,}\b', schema.description.lower())
        keywords.update(desc_words)

        # From parameter names
        for param in schema.parameters:
            if len(param.name) >= 3:
                keywords.add(param.name.lower())

        return keywords

    def _index_tool(self, tool_name: str, tool_info: ToolInfo) -> None:
        """V13: Add tool to all inverted indices."""
        # Category index
        self._category_index[tool_info.category].add(tool_name)

        # Tag index
        for tag in tool_info.tags:
            tag_lower = tag.lower()
            if tag_lower not in self._tag_index:
                self._tag_index[tag_lower] = set()
            self._tag_index[tag_lower].add(tool_name)

        # Keyword index
        keywords = self._extract_keywords(tool_info.schema_)
        for keyword in keywords:
            if keyword not in self._keyword_index:
                self._keyword_index[keyword] = set()
            self._keyword_index[keyword].add(tool_name)

    def _unindex_tool(self, tool_name: str, tool_info: ToolInfo) -> None:
        """V13: Remove tool from all inverted indices."""
        # Category index
        self._category_index[tool_info.category].discard(tool_name)

        # Tag index
        for tag in tool_info.tags:
            tag_lower = tag.lower()
            if tag_lower in self._tag_index:
                self._tag_index[tag_lower].discard(tool_name)

        # Keyword index
        keywords = self._extract_keywords(tool_info.schema_)
        for keyword in keywords:
            if keyword in self._keyword_index:
                self._keyword_index[keyword].discard(tool_name)

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def register(
        self,
        schema: ToolSchema,
        category: ToolCategory = ToolCategory.UTILITY,
        permission: ToolPermission = ToolPermission.READ_ONLY,
        source: str = "builtin",
        handler: Optional[Callable] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """
        Register a tool in the registry.

        Args:
            schema: Tool schema definition
            category: Tool category
            permission: Required permission level
            source: Where the tool comes from
            handler: Optional local handler function
            tags: Searchable tags

        Returns:
            True if registered successfully
        """
        tool_info = ToolInfo(
            schema=schema,
            category=category,
            permission=permission,
            source=source,
            tags=tags or [],
        )

        self._tools[schema.name] = tool_info

        # V13: Index the tool for O(1) lookups
        self._index_tool(schema.name, tool_info)

        if handler:
            self._local_executor.register_handler(schema.name, handler)
            self._executors[schema.name] = self._local_executor

        logger.info(f"Registered tool: {schema.name} (source={source})")
        return True

    def register_mcp_tool(
        self,
        schema: ToolSchema,
        server_name: str,
        executor: ToolExecutor,
    ) -> bool:
        """Register a tool from an MCP server."""
        tool_info = ToolInfo(
            schema=schema,
            category=self._infer_category(schema),
            permission=self._infer_permission(schema),
            source=f"mcp:{server_name}",
        )

        self._tools[schema.name] = tool_info
        self._executors[schema.name] = executor

        # V13: Index the tool for O(1) lookups
        self._index_tool(schema.name, tool_info)

        logger.info(f"Registered MCP tool: {schema.name} from {server_name}")
        return True

    def unregister(self, tool_name: str) -> bool:
        """Unregister a tool."""
        if tool_name in self._tools:
            tool_info = self._tools[tool_name]

            # V13: Remove from indices before deleting
            self._unindex_tool(tool_name, tool_info)

            del self._tools[tool_name]
            self._executors.pop(tool_name, None)
            return True
        return False

    # -------------------------------------------------------------------------
    # Discovery
    # -------------------------------------------------------------------------

    def search(
        self,
        query: str,
        categories: Optional[List[ToolCategory]] = None,
        max_results: int = 10,
    ) -> List[ToolInfo]:
        """
        Search for tools matching a query.

        V13 OPTIMIZATION: Uses inverted indices for O(1) candidate lookups
        - Old: O(n) linear scan through ALL tools
        - New: O(k) where k = number of candidate matches (~10-20% of n)
        - Expected improvement: 40-60% latency reduction for large registries

        Args:
            query: Search query
            categories: Filter by categories
            max_results: Maximum results to return

        Returns:
            List of matching tools sorted by relevance
        """
        query_lower = query.lower()

        # V13: Build candidate set from inverted indices (O(1) lookups)
        candidates: Set[str] = set()

        # Extract query words for keyword index lookup
        query_words = set(re.findall(r'\b\w{3,}\b', query_lower))

        # Gather candidates from keyword index
        for word in query_words:
            if word in self._keyword_index:
                candidates.update(self._keyword_index[word])

        # Also check tag index
        for word in query_words:
            if word in self._tag_index:
                candidates.update(self._tag_index[word])

        # If category filter specified, intersect with category index
        if categories:
            category_candidates: Set[str] = set()
            for cat in categories:
                category_candidates.update(self._category_index[cat])
            if candidates:
                candidates &= category_candidates
            else:
                candidates = category_candidates

        # Fallback: if no candidates from index, use all tools (preserves original behavior)
        if not candidates:
            candidates = set(self._tools.keys())

        # Score only the candidates (not all tools)
        results: List[tuple[ToolInfo, float]] = []

        for tool_name in candidates:
            tool = self._tools.get(tool_name)
            if not tool:
                continue

            if tool.status in (ToolStatus.DISABLED, ToolStatus.ERROR):
                continue

            if categories and tool.category not in categories:
                continue

            relevance = self._calculate_relevance(query_lower, tool)
            if relevance > 0:
                results.append((tool, relevance))

        results.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in results[:max_results]]

    def _calculate_relevance(self, query: str, tool: ToolInfo) -> float:
        """Calculate search relevance for a tool."""
        scores = []
        schema = tool.schema_

        # Name match (highest weight)
        if query in schema.name.lower():
            scores.append(1.0)
        elif schema.name.lower() in query:
            scores.append(0.9)

        # Description match
        if query in schema.description.lower():
            scores.append(0.7)

        # Tag match
        for tag in tool.tags:
            if query in tag.lower():
                scores.append(0.6)

        # Parameter name match
        for param in schema.parameters:
            if query in param.name.lower():
                scores.append(0.4)

        return max(scores) if scores else 0.0

    def recommend_for_task(
        self,
        task_description: str,
        max_tools: int = 5,
    ) -> List[ToolInfo]:
        """
        Recommend tools for a given task.

        Uses keyword extraction and category inference.
        """
        # Extract keywords
        keywords = set(re.findall(r'\b\w{3,}\b', task_description.lower()))

        # Map keywords to categories
        category_keywords = {
            ToolCategory.FILE_SYSTEM: {"file", "read", "write", "edit", "create", "delete"},
            ToolCategory.SHELL: {"run", "execute", "command", "bash", "terminal"},
            ToolCategory.SEARCH: {"search", "find", "lookup", "query", "google"},
            ToolCategory.MEMORY: {"remember", "save", "load", "store", "recall"},
            ToolCategory.NETWORK: {"fetch", "http", "api", "request", "download"},
            ToolCategory.DATABASE: {"database", "sql", "query", "table", "insert"},
            ToolCategory.CREATIVE: {"image", "video", "audio", "generate", "render"},
        }

        # Score categories
        category_scores: Dict[ToolCategory, int] = {}
        for cat, cat_keywords in category_keywords.items():
            overlap = len(keywords & cat_keywords)
            if overlap > 0:
                category_scores[cat] = overlap

        # Find matching tools
        results: List[tuple[ToolInfo, float]] = []

        for tool in self._tools.values():
            if tool.status != ToolStatus.AVAILABLE:
                continue

            score = 0.0

            # Category bonus
            if tool.category in category_scores:
                score += category_scores[tool.category] * 0.3

            # Keyword match in description
            desc_lower = tool.schema_.description.lower()
            for kw in keywords:
                if kw in desc_lower:
                    score += 0.2

            if score > 0:
                results.append((tool, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in results[:max_tools]]

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    async def execute(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
    ) -> ToolResult:
        """
        Execute a tool with given parameters.

        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters to pass to the tool

        Returns:
            ToolResult with output or error
        """
        # Check if tool exists
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool not found: {tool_name}",
            )

        # Check if blocked
        if tool_name in self._blocked_tools:
            return ToolResult(
                success=False,
                error=f"Tool is blocked: {tool_name}",
            )

        # Check permissions
        if tool.permission not in self._allowed_permissions:
            return ToolResult(
                success=False,
                error=f"Permission denied: {tool.permission.value} required",
            )

        # Check status
        if tool.status != ToolStatus.AVAILABLE:
            return ToolResult(
                success=False,
                error=f"Tool not available: {tool.status.value}",
            )

        # Get executor
        executor = self._executors.get(tool_name)
        if not executor:
            return ToolResult(
                success=False,
                error=f"No executor for tool: {tool_name}",
            )

        # Execute
        result = await executor.execute(tool_name, parameters)

        # Update usage stats
        tool.usage_count += 1
        tool.last_used = datetime.now()

        return result

    # -------------------------------------------------------------------------
    # Permission Management
    # -------------------------------------------------------------------------

    def allow_permission(self, permission: ToolPermission) -> None:
        """Allow a permission level."""
        self._allowed_permissions.add(permission)
        logger.info(f"Allowed permission: {permission.value}")

    def revoke_permission(self, permission: ToolPermission) -> None:
        """Revoke a permission level."""
        self._allowed_permissions.discard(permission)
        logger.info(f"Revoked permission: {permission.value}")

    def block_tool(self, tool_name: str) -> None:
        """Block a specific tool."""
        self._blocked_tools.add(tool_name)

    def unblock_tool(self, tool_name: str) -> None:
        """Unblock a specific tool."""
        self._blocked_tools.discard(tool_name)

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def _infer_category(self, schema: ToolSchema) -> ToolCategory:
        """Infer category from tool schema."""
        name_lower = schema.name.lower()
        desc_lower = schema.description.lower()

        if any(w in name_lower for w in ["read", "write", "file", "glob"]):
            return ToolCategory.FILE_SYSTEM
        if any(w in name_lower for w in ["bash", "shell", "exec"]):
            return ToolCategory.SHELL
        if any(w in name_lower for w in ["search", "grep", "find"]):
            return ToolCategory.SEARCH
        if any(w in name_lower for w in ["memory", "store", "recall"]):
            return ToolCategory.MEMORY
        if any(w in desc_lower for w in ["network", "http", "fetch", "api"]):
            return ToolCategory.NETWORK

        return ToolCategory.UTILITY

    def _infer_permission(self, schema: ToolSchema) -> ToolPermission:
        """Infer permission from tool schema."""
        name_lower = schema.name.lower()

        if any(w in name_lower for w in ["write", "edit", "create", "delete"]):
            return ToolPermission.READ_WRITE
        if any(w in name_lower for w in ["bash", "exec", "run"]):
            return ToolPermission.EXECUTE
        if any(w in name_lower for w in ["fetch", "http", "api"]):
            return ToolPermission.NETWORK

        return ToolPermission.READ_ONLY

    def get(self, tool_name: str) -> Optional[ToolInfo]:
        """Get tool info by name."""
        return self._tools.get(tool_name)

    def list_all(self) -> List[ToolInfo]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_by_category(
        self,
        category: ToolCategory,
    ) -> List[ToolInfo]:
        """List tools in a category."""
        return [t for t in self._tools.values() if t.category == category]

    def get_schemas_for_llm(
        self,
        format: str = "anthropic",
    ) -> List[Dict[str, Any]]:
        """
        Get tool schemas in LLM-compatible format.

        Args:
            format: "anthropic" or "openai"

        Returns:
            List of tool schemas
        """
        schemas = []

        for tool in self._tools.values():
            if tool.status != ToolStatus.AVAILABLE:
                continue
            if tool.permission not in self._allowed_permissions:
                continue

            if format == "anthropic":
                schemas.append(tool.schema_.to_anthropic_format())
            else:
                schemas.append(tool.schema_.to_openai_format())

        return schemas

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        by_category: Dict[str, int] = {}
        by_source: Dict[str, int] = {}
        by_status: Dict[str, int] = {}

        for tool in self._tools.values():
            cat = tool.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

            src = tool.source.split(":")[0]
            by_source[src] = by_source.get(src, 0) + 1

            stat = tool.status.value
            by_status[stat] = by_status.get(stat, 0) + 1

        return {
            "total_tools": len(self._tools),
            "allowed_permissions": [p.value for p in self._allowed_permissions],
            "blocked_tools": len(self._blocked_tools),
            "by_category": by_category,
            "by_source": by_source,
            "by_status": by_status,
        }


# =============================================================================
# Built-in Tools
# =============================================================================

def _read_file_handler(file_path: str, offset: int = 0, limit: int = 2000) -> str:
    """Built-in file read handler."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    lines = path.read_text(encoding="utf-8").split("\n")
    selected = lines[offset:offset + limit]
    return "\n".join(f"{i + offset + 1}→{line}" for i, line in enumerate(selected))


def _list_dir_handler(path: str, recursive: bool = False) -> List[str]:
    """Built-in directory listing handler."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if recursive:
        return [str(f.relative_to(p)) for f in p.rglob("*") if f.is_file()]
    return [f.name for f in p.iterdir()]


BUILTIN_TOOLS: List[tuple[ToolSchema, ToolCategory, ToolPermission, Callable]] = [
    (
        ToolSchema(
            name="Read",
            description="Read contents of a file. Returns numbered lines.",
            parameters=[
                ToolParameter(name="file_path", type="string", description="Path to file", required=True),
                ToolParameter(name="offset", type="number", description="Line offset", default=0),
                ToolParameter(name="limit", type="number", description="Max lines", default=2000),
            ],
        ),
        ToolCategory.FILE_SYSTEM,
        ToolPermission.READ_ONLY,
        _read_file_handler,
    ),
    (
        ToolSchema(
            name="ListDir",
            description="List files and directories in a path.",
            parameters=[
                ToolParameter(name="path", type="string", description="Directory path", required=True),
                ToolParameter(name="recursive", type="boolean", description="Include subdirs", default=False),
            ],
        ),
        ToolCategory.FILE_SYSTEM,
        ToolPermission.READ_ONLY,
        _list_dir_handler,
    ),
]


# =============================================================================
# Convenience Functions
# =============================================================================

def create_tool_registry(include_builtins: bool = True) -> ToolRegistry:
    """
    Factory function to create a configured ToolRegistry.

    Args:
        include_builtins: Whether to include built-in tools

    Returns:
        Configured ToolRegistry
    """
    registry = ToolRegistry()

    if include_builtins:
        for schema, category, permission, handler in BUILTIN_TOOLS:
            registry.register(
                schema=schema,
                category=category,
                permission=permission,
                source="builtin",
                handler=handler,
            )

    return registry


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate the tool registry."""
    print("=" * 60)
    print("UAP Tool Registry Demo")
    print("=" * 60)

    # Create registry
    registry = create_tool_registry(include_builtins=True)
    print(f"\nCreated registry with {len(registry._tools)} built-in tools")

    # Register a custom tool
    print("\n[Register Custom Tool]")
    registry.register(
        schema=ToolSchema(
            name="SearchWeb",
            description="Search the web using a query string.",
            parameters=[
                ToolParameter(name="query", type="string", description="Search query", required=True),
                ToolParameter(name="num_results", type="number", description="Max results", default=10),
            ],
        ),
        category=ToolCategory.SEARCH,
        permission=ToolPermission.NETWORK,
        tags=["web", "search", "google"],
    )
    print("  Registered: SearchWeb")

    # Search tools
    print("\n[Tool Search]")
    results = registry.search("read file")
    for tool in results:
        print(f"  - {tool.schema_.name}: {tool.schema_.description[:40]}...")

    # Recommend tools for task
    print("\n[Tool Recommendation]")
    task = "I need to read some configuration files and search the web"
    recommended = registry.recommend_for_task(task)
    print(f"  Task: '{task[:50]}...'")
    for tool in recommended:
        print(f"  -> {tool.schema_.name} ({tool.category.value})")

    # Get schemas for LLM
    print("\n[LLM Tool Schemas]")
    schemas = registry.get_schemas_for_llm(format="anthropic")
    print(f"  Generated {len(schemas)} tool schemas for Anthropic format")

    # Stats
    print("\n[Registry Stats]")
    stats = registry.get_stats()
    print(f"  Total: {stats['total_tools']}")
    print(f"  By category: {stats['by_category']}")
    print(f"  Allowed permissions: {stats['allowed_permissions']}")


if __name__ == "__main__":
    demo()
