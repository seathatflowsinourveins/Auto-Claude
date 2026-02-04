"""
MCP Connection Pool Bulkhead Integration - V38 Architecture

Integrates bulkhead pattern with MCP connection pool for enhanced
resource isolation and fault tolerance.

This module provides:
- Per-server bulkhead isolation
- Combined connection pool + bulkhead management
- Automatic bulkhead creation for new servers
- Unified health and metrics reporting

Usage:
    pool = BulkheadMCPPool()
    await pool.initialize()

    # Execute operation with bulkhead protection
    async with pool.protected_connection("exa") as conn:
        result = await conn.call_tool("search", query="...")

    # Or use direct execute
    result = await pool.execute(
        "exa",
        lambda conn: conn.call_tool("search", query="...")
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar

from core.mcp.connection_pool import (
    MCPConnection,
    MCPConnectionPool,
    ServerHealth,
    get_mcp_pool,
)
from core.orchestration.bulkhead import (
    Bulkhead,
    BulkheadConfig,
    BulkheadError,
    BulkheadRegistry,
    BulkheadStats,
    get_bulkhead_registry,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class MCPBulkheadConfig:
    """Configuration for MCP server bulkhead."""
    server_name: str
    max_concurrent: int = 10
    max_queue_size: int = 100
    timeout_seconds: float = 30.0
    enable_dynamic_limits: bool = True
    enable_circuit_breaker: bool = True
    failure_threshold: int = 5


@dataclass
class MCPBulkheadStats:
    """Combined statistics for MCP connection pool and bulkhead."""
    server_name: str
    # Connection pool stats
    pooled_connections: int
    active_connections: int
    max_connections: int
    # Bulkhead stats
    bulkhead_concurrent: int
    bulkhead_max_concurrent: int
    bulkhead_queue_size: int
    bulkhead_utilization: float
    # Health stats
    is_healthy: bool
    error_rate: float
    avg_latency_ms: float
    # Combined stats
    total_requests: int
    successful_requests: int
    rejected_requests: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "server_name": self.server_name,
            "pooled_connections": self.pooled_connections,
            "active_connections": self.active_connections,
            "max_connections": self.max_connections,
            "bulkhead_concurrent": self.bulkhead_concurrent,
            "bulkhead_max_concurrent": self.bulkhead_max_concurrent,
            "bulkhead_queue_size": self.bulkhead_queue_size,
            "bulkhead_utilization": round(self.bulkhead_utilization, 3),
            "is_healthy": self.is_healthy,
            "error_rate": round(self.error_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "rejected_requests": self.rejected_requests,
        }


class BulkheadMCPPool:
    """
    MCP Connection Pool with integrated bulkhead protection.

    Combines connection pooling with bulkhead pattern for:
    - Connection reuse and warm connections
    - Concurrency limiting per server
    - Fault isolation between servers
    - Circuit breaker protection
    - Unified health monitoring

    Example:
        pool = BulkheadMCPPool()
        await pool.initialize()

        # Use context manager
        async with pool.protected_connection("exa") as conn:
            result = await conn.call_tool("search", query="test")

        # Use execute helper
        result = await pool.execute(
            "tavily",
            lambda conn: conn.call_tool("search", query="test")
        )

        # Get unified stats
        stats = pool.get_stats()
    """

    def __init__(
        self,
        connection_pool: Optional[MCPConnectionPool] = None,
        bulkhead_registry: Optional[BulkheadRegistry] = None,
        default_bulkhead_config: Optional[MCPBulkheadConfig] = None,
    ):
        """
        Initialize the bulkhead-protected MCP pool.

        Args:
            connection_pool: Existing connection pool (creates new if None)
            bulkhead_registry: Existing bulkhead registry (creates new if None)
            default_bulkhead_config: Default config for auto-created bulkheads
        """
        self._connection_pool = connection_pool or get_mcp_pool()
        self._bulkhead_registry = bulkhead_registry or get_bulkhead_registry()
        self._default_config = default_bulkhead_config

        # Server-specific configs
        self._server_configs: Dict[str, MCPBulkheadConfig] = {}

        # Track bulkhead names for MCP servers
        self._server_bulkheads: Dict[str, str] = {}

        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the pool and start bulkheads."""
        if self._initialized:
            return

        await self._connection_pool.initialize()
        await self._bulkhead_registry.start_all()
        self._initialized = True
        logger.info("[BULKHEAD_MCP] Initialized bulkhead-protected MCP pool")

    async def shutdown(self) -> None:
        """Shutdown pool and bulkheads."""
        await self._bulkhead_registry.shutdown_all()
        await self._connection_pool.shutdown()
        self._initialized = False
        logger.info("[BULKHEAD_MCP] Shutdown bulkhead-protected MCP pool")

    def configure_server(self, config: MCPBulkheadConfig) -> None:
        """
        Configure bulkhead settings for a specific server.

        Args:
            config: Server-specific bulkhead configuration
        """
        self._server_configs[config.server_name] = config
        logger.info(
            f"[BULKHEAD_MCP] Configured server '{config.server_name}': "
            f"max_concurrent={config.max_concurrent}"
        )

    def _get_bulkhead(self, server_name: str) -> Bulkhead:
        """Get or create bulkhead for server."""
        bulkhead_name = f"mcp_{server_name}"

        if bulkhead_name not in self._server_bulkheads:
            # Get server-specific or default config
            server_config = self._server_configs.get(server_name)

            if server_config:
                config = BulkheadConfig(
                    name=bulkhead_name,
                    max_concurrent=server_config.max_concurrent,
                    max_queue_size=server_config.max_queue_size,
                    timeout_seconds=server_config.timeout_seconds,
                    enable_dynamic_limits=server_config.enable_dynamic_limits,
                    enable_circuit_breaker=server_config.enable_circuit_breaker,
                    failure_threshold=server_config.failure_threshold,
                )
            elif self._default_config:
                config = BulkheadConfig(
                    name=bulkhead_name,
                    max_concurrent=self._default_config.max_concurrent,
                    max_queue_size=self._default_config.max_queue_size,
                    timeout_seconds=self._default_config.timeout_seconds,
                    enable_dynamic_limits=self._default_config.enable_dynamic_limits,
                    enable_circuit_breaker=self._default_config.enable_circuit_breaker,
                    failure_threshold=self._default_config.failure_threshold,
                )
            else:
                # Use defaults for MCP servers
                config = BulkheadConfig(
                    name=bulkhead_name,
                    max_concurrent=10,
                    max_queue_size=100,
                    timeout_seconds=45.0,
                    enable_dynamic_limits=True,
                    enable_circuit_breaker=True,
                    failure_threshold=5,
                    circuit_open_duration_seconds=45.0,
                )

            self._server_bulkheads[server_name] = bulkhead_name
            logger.debug(f"[BULKHEAD_MCP] Created bulkhead for server '{server_name}'")

        return self._bulkhead_registry.get_or_create(
            self._server_bulkheads[server_name],
            None  # Config already set above
        )

    @asynccontextmanager
    async def protected_connection(
        self,
        server_name: str,
        timeout: Optional[float] = None,
        priority: int = 0
    ):
        """
        Get a connection protected by bulkhead.

        Usage:
            async with pool.protected_connection("exa") as conn:
                result = await conn.call_tool(...)

        Args:
            server_name: Name of MCP server
            timeout: Operation timeout
            priority: Request priority (higher = higher priority)

        Yields:
            MCPConnection instance
        """
        bulkhead = self._get_bulkhead(server_name)

        async def acquire_connection() -> MCPConnection:
            conn = await self._connection_pool.acquire(
                server_name,
                timeout=timeout or bulkhead.config.timeout_seconds
            )
            if conn is None:
                raise ConnectionError(f"Failed to acquire connection to {server_name}")
            return conn

        # Acquire connection within bulkhead
        conn = await bulkhead.execute(acquire_connection, timeout, priority)

        success = True
        start_time = time.time()
        try:
            yield conn
        except Exception:
            success = False
            raise
        finally:
            latency_ms = (time.time() - start_time) * 1000
            # Release connection back to pool
            await self._connection_pool.release(server_name, conn, success)

    async def execute(
        self,
        server_name: str,
        operation: Callable[[MCPConnection], Awaitable[T]],
        timeout: Optional[float] = None,
        priority: int = 0
    ) -> T:
        """
        Execute operation on MCP server with bulkhead protection.

        Args:
            server_name: Name of MCP server
            operation: Async callable that receives connection
            timeout: Operation timeout
            priority: Request priority

        Returns:
            Operation result
        """
        async with self.protected_connection(server_name, timeout, priority) as conn:
            return await operation(conn)

    async def try_execute(
        self,
        server_name: str,
        operation: Callable[[MCPConnection], Awaitable[T]],
        timeout: Optional[float] = None,
        priority: int = 0,
        default: T = None
    ) -> T:
        """
        Try to execute operation, return default if bulkhead rejects.

        Args:
            server_name: Name of MCP server
            operation: Async callable that receives connection
            timeout: Operation timeout
            priority: Request priority
            default: Default value if operation cannot execute

        Returns:
            Operation result or default
        """
        bulkhead = self._get_bulkhead(server_name)

        async def execute_with_connection() -> T:
            conn = await self._connection_pool.acquire(
                server_name,
                timeout=timeout or bulkhead.config.timeout_seconds
            )
            if conn is None:
                raise ConnectionError(f"Failed to acquire connection to {server_name}")

            try:
                return await operation(conn)
            finally:
                await self._connection_pool.release(server_name, conn, True)

        result = await bulkhead.try_execute(execute_with_connection, timeout, priority)
        return result if result is not None else default

    def get_server_stats(self, server_name: str) -> Optional[MCPBulkheadStats]:
        """Get combined stats for a server."""
        # Get pool stats
        pool_stats = self._connection_pool.get_stats()
        server_pool_stats = pool_stats.get("servers", {}).get(server_name)

        if not server_pool_stats:
            return None

        # Get bulkhead stats
        bulkhead = self._get_bulkhead(server_name)
        bh_stats = bulkhead.get_stats()

        # Get health
        health = self._connection_pool.get_health(server_name)

        return MCPBulkheadStats(
            server_name=server_name,
            pooled_connections=server_pool_stats.get("pooled", 0),
            active_connections=server_pool_stats.get("active", 0),
            max_connections=self._connection_pool.max_connections_per_server,
            bulkhead_concurrent=bh_stats.current_concurrent,
            bulkhead_max_concurrent=bh_stats.max_concurrent,
            bulkhead_queue_size=bh_stats.queue_size,
            bulkhead_utilization=bh_stats.utilization,
            is_healthy=health.is_healthy if health else True,
            error_rate=health.error_rate if health else 0.0,
            avg_latency_ms=health.avg_latency_ms if health else 0.0,
            total_requests=bh_stats.total_requests,
            successful_requests=bh_stats.successful_requests,
            rejected_requests=bh_stats.rejected_requests,
        )

    def get_stats(self) -> Dict[str, MCPBulkheadStats]:
        """Get stats for all servers."""
        stats = {}

        # Get all known servers
        pool_stats = self._connection_pool.get_stats()
        servers = set(pool_stats.get("servers", {}).keys())
        servers.update(self._server_bulkheads.keys())

        for server_name in servers:
            server_stats = self.get_server_stats(server_name)
            if server_stats:
                stats[server_name] = server_stats

        return stats

    def get_stats_dict(self) -> Dict[str, Dict[str, Any]]:
        """Get stats as dictionaries for serialization."""
        return {name: stats.to_dict() for name, stats in self.get_stats().items()}

    def get_health(self) -> Dict[str, Any]:
        """Get combined health status."""
        pool_health = self._connection_pool.get_all_health()
        bulkhead_health = self._bulkhead_registry.get_health()

        # Aggregate health
        servers_health = {}
        all_servers = set(pool_health.keys()) | set(self._server_bulkheads.keys())

        unhealthy_count = 0
        degraded_count = 0

        for server_name in all_servers:
            server_pool = pool_health.get(server_name)
            bulkhead = self._get_bulkhead(server_name)
            bh_health = bulkhead.get_health()

            pool_healthy = server_pool.is_healthy if server_pool else True
            bh_healthy = bh_health["status"] in ("healthy", "stressed")

            if not pool_healthy or not bh_healthy:
                if bh_health["status"] == "unhealthy":
                    unhealthy_count += 1
                else:
                    degraded_count += 1

            servers_health[server_name] = {
                "pool_healthy": pool_healthy,
                "bulkhead_status": bh_health["status"],
                "circuit_open": bh_health.get("circuit_open", False),
                "utilization": bh_health.get("utilization", 0),
            }

        # Overall status
        if unhealthy_count > 0:
            overall = "unhealthy"
        elif degraded_count > 0:
            overall = "degraded"
        else:
            overall = "healthy"

        return {
            "status": overall,
            "initialized": self._initialized,
            "server_count": len(all_servers),
            "unhealthy_count": unhealthy_count,
            "degraded_count": degraded_count,
            "servers": servers_health,
        }

    def adjust_server_limit(self, server_name: str, new_limit: int) -> None:
        """Manually adjust bulkhead limit for a server."""
        bulkhead = self._get_bulkhead(server_name)
        bulkhead.adjust_limit(new_limit)


# Global instance
_bulkhead_mcp_pool: Optional[BulkheadMCPPool] = None


def get_bulkhead_mcp_pool() -> BulkheadMCPPool:
    """Get global bulkhead-protected MCP pool."""
    global _bulkhead_mcp_pool
    if _bulkhead_mcp_pool is None:
        _bulkhead_mcp_pool = BulkheadMCPPool()
    return _bulkhead_mcp_pool


async def initialize_bulkhead_mcp_pool(
    server_configs: Optional[List[MCPBulkheadConfig]] = None
) -> BulkheadMCPPool:
    """
    Initialize global bulkhead MCP pool with optional server configs.

    Args:
        server_configs: Optional list of server-specific configurations

    Returns:
        Initialized BulkheadMCPPool
    """
    pool = get_bulkhead_mcp_pool()

    if server_configs:
        for config in server_configs:
            pool.configure_server(config)

    await pool.initialize()
    return pool
