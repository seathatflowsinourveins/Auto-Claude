"""
MCP (Model Context Protocol) Optimization Layer

This module provides enhanced MCP capabilities:
- Connection pooling for reduced latency
- Semantic caching for tool results
- Fallback chains for reliability
- Parallel execution coordination

V38 Architecture - MCP Optimization (ADR-028)

Usage:
    from core.mcp import (
        get_mcp_pool,
        get_mcp_cache,
        get_fallback_manager,
        cached_tool_call,
        execute_with_fallback,
    )

    # Connection pooling
    pool = get_mcp_pool()
    async with pool.connection("exa") as conn:
        result = await conn.call_tool("search", {"query": "..."})

    # Cached tool calls
    result = await cached_tool_call(
        server="exa",
        tool="search",
        args={"query": "LangGraph"},
        executor=lambda: exa.search("LangGraph"),
    )

    # Fallback execution
    result = await execute_with_fallback(
        chain="research",
        tool="search",
        args={"query": "AI patterns"},
    )
"""

from typing import Dict, Any, Optional, List

# Module availability tracking
MCP_OPTIMIZATIONS_AVAILABLE = True

# Connection pooling
from .connection_pool import (
    MCPConnectionPool,
    MCPConnection,
    ServerHealth,
    get_mcp_pool,
    warmup_connections,
)

# Tool caching
from .tool_cache import (
    MCPToolCache,
    CacheEntry,
    CacheStats,
    get_mcp_cache,
    cached_tool_call,
)

# Fallback chains
from .fallback_chain import (
    FallbackChainManager,
    FallbackChain,
    FallbackStrategy,
    ChainMember,
    ChainResult,
    get_fallback_manager,
    execute_with_fallback,
)

# Bulkhead integration (V38 - ADR-029)
from .bulkhead_integration import (
    BulkheadMCPPool,
    MCPBulkheadConfig,
    MCPBulkheadStats,
    get_bulkhead_mcp_pool,
    initialize_bulkhead_mcp_pool,
)

# Tasks primitive (MCP long-running operations)
from .tasks import (
    Task,
    TaskStatus,
    TaskPriority,
    TaskMetadata,
    TaskProgress,
    TaskCreateRequest,
    TaskListRequest,
    TaskManager,
    get_task_manager,
    initialize_task_manager,
)

__all__ = [
    # Availability
    "MCP_OPTIMIZATIONS_AVAILABLE",
    # Connection pooling
    "MCPConnectionPool",
    "MCPConnection",
    "ServerHealth",
    "get_mcp_pool",
    "warmup_connections",
    # Tool caching
    "MCPToolCache",
    "CacheEntry",
    "CacheStats",
    "get_mcp_cache",
    "cached_tool_call",
    # Fallback chains
    "FallbackChainManager",
    "FallbackChain",
    "FallbackStrategy",
    "ChainMember",
    "ChainResult",
    "get_fallback_manager",
    "execute_with_fallback",
    # Bulkhead integration
    "BulkheadMCPPool",
    "MCPBulkheadConfig",
    "MCPBulkheadStats",
    "get_bulkhead_mcp_pool",
    "initialize_bulkhead_mcp_pool",
    # Tasks primitive
    "Task",
    "TaskStatus",
    "TaskPriority",
    "TaskMetadata",
    "TaskProgress",
    "TaskCreateRequest",
    "TaskListRequest",
    "TaskManager",
    "get_task_manager",
    "initialize_task_manager",
]
