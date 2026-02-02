#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pydantic>=2.0.0",
#     "httpx>=0.25.0",
#     "aiofiles>=23.0.0",
# ]
# ///
"""
MCP Discovery - Dynamic Server Discovery and Capability Negotiation

Enhances MCP server management with:
- Dynamic discovery from MCP Registry (registry.modelcontextprotocol.io)
- Tool capability negotiation via MCP protocol
- Connection pooling and resource management
- Server health scoring and intelligent routing

Based on MCP specification 2025-06-18:
- tools/list for discovering available tools
- resources/list for discovering resources
- prompts/list for discovering prompts
- initialize for capability exchange

Reference: https://modelcontextprotocol.io/specification/2025-06-18

Usage:
    from mcp_discovery import MCPDiscovery, RegistryClient, ConnectionPool

    # Discover servers from registry
    registry = RegistryClient()
    servers = await registry.search("memory")

    # Create discovery-enabled manager
    discovery = MCPDiscovery()
    await discovery.discover_and_register("memory-server")
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
import httpx

from .mcp_manager import (
    MCPServerManager,
    ServerConfig,
    ServerStatus,
    ToolSchema,
    ResourceSchema,
    PromptSchema,
    TransportType,
)


# =============================================================================
# Registry Client - Discover servers from MCP Registry
# =============================================================================

class RegistryEntry(BaseModel):
    """Entry from the MCP Registry."""
    name: str
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    license: str = ""
    repository: str = ""
    homepage: str = ""

    # Installation
    package_name: str = ""  # npm or pip package
    package_manager: str = "npm"  # npm, pip, cargo
    command: List[str] = Field(default_factory=list)

    # Capabilities (from registry metadata)
    capabilities: List[str] = Field(default_factory=list)
    tool_count: int = 0
    resource_count: int = 0
    prompt_count: int = 0

    # Ratings
    downloads: int = 0
    stars: int = 0
    verified: bool = False

    # Timestamps
    published_at: str = ""
    updated_at: str = ""


class RegistrySearchResult(BaseModel):
    """Search results from registry."""
    entries: List[RegistryEntry] = Field(default_factory=list)
    total_count: int = 0
    page: int = 1
    per_page: int = 20


class RegistryClient:
    """
    Client for the MCP Registry (registry.modelcontextprotocol.io).

    Discovers MCP servers from the central registry with:
    - Search by name, capability, or description
    - Filter by verification status, downloads, stars
    - Cached results to reduce API calls
    """

    REGISTRY_URL = "https://registry.modelcontextprotocol.io/api/v1"

    # V13: Adaptive TTL Configuration
    BASE_TTL = 3600       # 1 hour base TTL
    MIN_TTL = 900         # 15 minutes minimum
    MAX_TTL = 14400       # 4 hours maximum
    TTL_GROWTH_FACTOR = 1.2  # 20% increase on each hit
    TTL_DECAY_FACTOR = 0.8   # 20% decrease on miss after expiry

    def __init__(self, cache_dir: Optional[Path] = None):
        self._cache_dir = cache_dir or Path.home() / ".cache" / "mcp-registry"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        # V13: Enhanced cache structure with adaptive TTL tracking
        # Structure: {key: (timestamp, value, current_ttl, hit_count)}
        self._cache: Dict[str, Tuple[float, Any, float, int]] = {}
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers={"Accept": "application/json"},
            )
        return self._client

    def _cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate cache key for request."""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.sha256(f"{endpoint}:{param_str}".encode()).hexdigest()[:16]

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached value if not expired.

        V13: Adaptive TTL - increases TTL on hits for frequently accessed data.
        Expected improvement: 3-5x cache hit rate for stable data.
        """
        if key in self._cache:
            timestamp, value, current_ttl, hit_count = self._cache[key]
            if time.time() - timestamp < current_ttl:
                # Cache hit - increase TTL for next time (adaptive)
                new_ttl = min(current_ttl * self.TTL_GROWTH_FACTOR, self.MAX_TTL)
                self._cache[key] = (timestamp, value, new_ttl, hit_count + 1)
                return value
            # Expired - remove entry (will be refreshed with decayed TTL)
            del self._cache[key]
        return None

    def _set_cached(self, key: str, value: Any, from_miss: bool = False) -> None:
        """Cache a value with adaptive TTL.

        V13: If refreshing after expiry (from_miss=True), use decayed TTL.
        If first cache or hit refresh, use base TTL.
        """
        if from_miss and key in self._cache:
            # Decayed TTL for frequently-expiring entries
            _, _, old_ttl, _ = self._cache[key]
            new_ttl = max(old_ttl * self.TTL_DECAY_FACTOR, self.MIN_TTL)
        else:
            # Fresh entry or stable pattern - use base TTL
            new_ttl = self.BASE_TTL
        self._cache[key] = (time.time(), value, new_ttl, 0)

    async def search(
        self,
        query: str = "",
        capability: str = "",
        verified_only: bool = False,
        min_downloads: int = 0,
        page: int = 1,
        per_page: int = 20,
    ) -> RegistrySearchResult:
        """
        Search the MCP Registry for servers.

        Args:
            query: Search term (name, description)
            capability: Filter by capability (e.g., "memory", "filesystem")
            verified_only: Only return verified servers
            min_downloads: Minimum download count
            page: Page number
            per_page: Results per page

        Returns:
            RegistrySearchResult with matching entries
        """
        params = {
            "q": query,
            "capability": capability,
            "verified": verified_only,
            "min_downloads": min_downloads,
            "page": page,
            "per_page": per_page,
        }

        cache_key = self._cache_key("search", params)
        cached = self._get_cached(cache_key)
        if cached:
            return RegistrySearchResult(**cached)

        try:
            client = await self._get_client()
            response = await client.get(f"{self.REGISTRY_URL}/servers", params=params)

            if response.status_code == 200:
                data = response.json()
                self._set_cached(cache_key, data)
                return RegistrySearchResult(**data)
            else:
                # Return empty result on error
                return RegistrySearchResult()

        except Exception:
            # Fall back to local cache or empty result
            return RegistrySearchResult()

    async def get_server(self, name: str) -> Optional[RegistryEntry]:
        """Get a specific server by name."""
        cache_key = self._cache_key("server", {"name": name})
        cached = self._get_cached(cache_key)
        if cached:
            return RegistryEntry(**cached)

        try:
            client = await self._get_client()
            response = await client.get(f"{self.REGISTRY_URL}/servers/{name}")

            if response.status_code == 200:
                data = response.json()
                self._set_cached(cache_key, data)
                return RegistryEntry(**data)

        except Exception:
            pass

        return None

    async def get_popular(self, limit: int = 10) -> List[RegistryEntry]:
        """Get most popular servers by downloads."""
        result = await self.search(per_page=limit)
        return sorted(result.entries, key=lambda e: e.downloads, reverse=True)[:limit]

    async def get_by_capability(self, capability: str) -> List[RegistryEntry]:
        """Get servers that provide a specific capability."""
        result = await self.search(capability=capability, per_page=50)
        return result.entries

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# =============================================================================
# Capability Negotiation - Query servers for capabilities
# =============================================================================

class MCPProtocolVersion(str, Enum):
    """Supported MCP protocol versions."""
    V_2024_11 = "2024-11-05"
    V_2025_03 = "2025-03-26"
    V_2025_06 = "2025-06-18"  # Latest


class ServerCapabilities(BaseModel):
    """Server capabilities discovered via initialization."""
    protocol_version: str = MCPProtocolVersion.V_2025_06.value
    server_name: str = ""
    server_version: str = ""

    # Capability flags
    supports_tools: bool = True
    supports_resources: bool = True
    supports_prompts: bool = True
    supports_logging: bool = False
    supports_sampling: bool = False

    # Discovered capabilities
    tools: List[ToolSchema] = Field(default_factory=list)
    resources: List[ResourceSchema] = Field(default_factory=list)
    prompts: List[PromptSchema] = Field(default_factory=list)

    # Metadata
    discovered_at: float = 0.0
    discovery_duration_ms: float = 0.0


class CapabilityNegotiator:
    """
    Negotiates capabilities with MCP servers.

    Uses MCP protocol methods:
    - initialize: Exchange protocol version and capabilities
    - tools/list: Discover available tools
    - resources/list: Discover available resources
    - prompts/list: Discover available prompts
    """

    def __init__(self, timeout: float = 30.0):
        self._timeout = timeout
        self._negotiation_cache: Dict[str, ServerCapabilities] = {}

    async def negotiate(
        self,
        server_name: str,
        transport: TransportType,
        process: Any = None,
        url: str = "",
    ) -> ServerCapabilities:
        """
        Negotiate capabilities with a server.

        Args:
            server_name: Server identifier
            transport: Transport type (stdio, http, etc.)
            process: Process handle for stdio transport
            url: URL for HTTP transports

        Returns:
            ServerCapabilities discovered from server
        """
        start_time = time.time()
        capabilities = ServerCapabilities(server_name=server_name)

        try:
            if transport == TransportType.STDIO:
                capabilities = await self._negotiate_stdio(server_name, process)
            elif transport in (TransportType.HTTP_SSE, TransportType.STREAMABLE_HTTP):
                capabilities = await self._negotiate_http(server_name, url)

        except Exception:
            # Return minimal capabilities on error
            capabilities.discovered_at = time.time()
            capabilities.discovery_duration_ms = (time.time() - start_time) * 1000

        capabilities.discovered_at = time.time()
        capabilities.discovery_duration_ms = (time.time() - start_time) * 1000
        self._negotiation_cache[server_name] = capabilities

        return capabilities

    async def _negotiate_stdio(
        self,
        server_name: str,
        process: Any,
    ) -> ServerCapabilities:
        """Negotiate with stdio server using JSON-RPC."""
        capabilities = ServerCapabilities(server_name=server_name)

        if not process or not process.stdin or not process.stdout:
            return capabilities

        # Send initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": MCPProtocolVersion.V_2025_06.value,
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {},
                },
                "clientInfo": {
                    "name": "UAP-Core",
                    "version": "1.0.0",
                },
            },
        }

        try:
            # Write request
            request_data = json.dumps(init_request) + "\n"
            process.stdin.write(request_data.encode())
            process.stdin.flush()

            # Read response (with timeout)
            response_line = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, process.stdout.readline
                ),
                timeout=self._timeout,
            )

            if response_line:
                response = json.loads(response_line.decode())
                if "result" in response:
                    result = response["result"]
                    capabilities.protocol_version = result.get("protocolVersion", capabilities.protocol_version)
                    capabilities.server_version = result.get("serverInfo", {}).get("version", "")

                    caps = result.get("capabilities", {})
                    capabilities.supports_tools = "tools" in caps
                    capabilities.supports_resources = "resources" in caps
                    capabilities.supports_prompts = "prompts" in caps
                    capabilities.supports_logging = "logging" in caps
                    capabilities.supports_sampling = "sampling" in caps

            # Query tools
            if capabilities.supports_tools:
                capabilities.tools = await self._query_stdio_tools(process)

            # Query resources
            if capabilities.supports_resources:
                capabilities.resources = await self._query_stdio_resources(process)

            # Query prompts
            if capabilities.supports_prompts:
                capabilities.prompts = await self._query_stdio_prompts(process)

        except asyncio.TimeoutError:
            pass
        except Exception:
            pass

        return capabilities

    async def _query_stdio_tools(self, process: Any) -> List[ToolSchema]:
        """Query tools from stdio server."""
        tools = []

        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }

        try:
            request_data = json.dumps(request) + "\n"
            process.stdin.write(request_data.encode())
            process.stdin.flush()

            response_line = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, process.stdout.readline
                ),
                timeout=self._timeout,
            )

            if response_line:
                response = json.loads(response_line.decode())
                if "result" in response:
                    for tool_data in response["result"].get("tools", []):
                        tools.append(ToolSchema(
                            name=tool_data.get("name", ""),
                            description=tool_data.get("description", ""),
                            input_schema=tool_data.get("inputSchema", {}),
                        ))

        except Exception:
            pass

        return tools

    async def _query_stdio_resources(self, process: Any) -> List[ResourceSchema]:
        """Query resources from stdio server."""
        resources = []

        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/list",
            "params": {},
        }

        try:
            request_data = json.dumps(request) + "\n"
            process.stdin.write(request_data.encode())
            process.stdin.flush()

            response_line = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, process.stdout.readline
                ),
                timeout=self._timeout,
            )

            if response_line:
                response = json.loads(response_line.decode())
                if "result" in response:
                    for res_data in response["result"].get("resources", []):
                        resources.append(ResourceSchema(
                            uri=res_data.get("uri", ""),
                            name=res_data.get("name", ""),
                            description=res_data.get("description", ""),
                            mime_type=res_data.get("mimeType", "application/json"),
                        ))

        except Exception:
            pass

        return resources

    async def _query_stdio_prompts(self, process: Any) -> List[PromptSchema]:
        """Query prompts from stdio server."""
        prompts = []

        request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "prompts/list",
            "params": {},
        }

        try:
            request_data = json.dumps(request) + "\n"
            process.stdin.write(request_data.encode())
            process.stdin.flush()

            response_line = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, process.stdout.readline
                ),
                timeout=self._timeout,
            )

            if response_line:
                response = json.loads(response_line.decode())
                if "result" in response:
                    for prompt_data in response["result"].get("prompts", []):
                        prompts.append(PromptSchema(
                            name=prompt_data.get("name", ""),
                            description=prompt_data.get("description", ""),
                            arguments=prompt_data.get("arguments", []),
                        ))

        except Exception:
            pass

        return prompts

    async def _negotiate_http(
        self,
        server_name: str,
        url: str,
    ) -> ServerCapabilities:
        """Negotiate with HTTP server."""
        capabilities = ServerCapabilities(server_name=server_name)

        if not url:
            return capabilities

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            # Send initialize request
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": MCPProtocolVersion.V_2025_06.value,
                    "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                    "clientInfo": {"name": "UAP-Core", "version": "1.0.0"},
                },
            }

            try:
                response = await client.post(
                    url,
                    json=init_request,
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code == 200:
                    result = response.json().get("result", {})
                    capabilities.protocol_version = result.get("protocolVersion", capabilities.protocol_version)
                    capabilities.server_version = result.get("serverInfo", {}).get("version", "")

                    caps = result.get("capabilities", {})
                    capabilities.supports_tools = "tools" in caps
                    capabilities.supports_resources = "resources" in caps
                    capabilities.supports_prompts = "prompts" in caps

                # Query tools
                if capabilities.supports_tools:
                    tools_response = await client.post(
                        url,
                        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
                    )
                    if tools_response.status_code == 200:
                        for tool_data in tools_response.json().get("result", {}).get("tools", []):
                            capabilities.tools.append(ToolSchema(
                                name=tool_data.get("name", ""),
                                description=tool_data.get("description", ""),
                                input_schema=tool_data.get("inputSchema", {}),
                            ))

                # Query resources
                if capabilities.supports_resources:
                    res_response = await client.post(
                        url,
                        json={"jsonrpc": "2.0", "id": 3, "method": "resources/list", "params": {}},
                    )
                    if res_response.status_code == 200:
                        for res_data in res_response.json().get("result", {}).get("resources", []):
                            capabilities.resources.append(ResourceSchema(
                                uri=res_data.get("uri", ""),
                                name=res_data.get("name", ""),
                                description=res_data.get("description", ""),
                            ))

                # Query prompts
                if capabilities.supports_prompts:
                    prompts_response = await client.post(
                        url,
                        json={"jsonrpc": "2.0", "id": 4, "method": "prompts/list", "params": {}},
                    )
                    if prompts_response.status_code == 200:
                        for prompt_data in prompts_response.json().get("result", {}).get("prompts", []):
                            capabilities.prompts.append(PromptSchema(
                                name=prompt_data.get("name", ""),
                                description=prompt_data.get("description", ""),
                                arguments=prompt_data.get("arguments", []),
                            ))

            except Exception:
                pass

        return capabilities

    def get_cached(self, server_name: str) -> Optional[ServerCapabilities]:
        """Get cached capabilities for a server."""
        return self._negotiation_cache.get(server_name)


# =============================================================================
# Connection Pool - Manage pooled connections
# =============================================================================

@dataclass
class PooledConnection:
    """A pooled connection to an MCP server."""
    server_name: str
    transport: TransportType
    process: Any = None
    client: Optional[httpx.AsyncClient] = None
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    request_count: int = 0
    error_count: int = 0
    is_healthy: bool = True


class ConnectionPool:
    """
    Connection pool for MCP servers.

    Features:
    - Reuse connections to reduce latency
    - Automatic cleanup of idle connections
    - Health-based connection selection
    - Maximum connections per server
    """

    def __init__(
        self,
        max_connections_per_server: int = 5,
        idle_timeout: float = 300.0,  # 5 minutes
        max_lifetime: float = 3600.0,  # 1 hour
    ):
        self._max_per_server = max_connections_per_server
        self._idle_timeout = idle_timeout
        self._max_lifetime = max_lifetime

        self._pools: Dict[str, List[PooledConnection]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    async def acquire(
        self,
        server_name: str,
        transport: TransportType,
        factory: Callable[[], Any],
    ) -> PooledConnection:
        """
        Acquire a connection from the pool or create a new one.

        Args:
            server_name: Server to connect to
            transport: Transport type
            factory: Factory function to create new connection

        Returns:
            PooledConnection ready for use
        """
        async with self._lock:
            pool = self._pools[server_name]

            # Find healthy idle connection
            for conn in pool:
                if conn.is_healthy and self._is_valid(conn):
                    conn.last_used = time.time()
                    conn.request_count += 1
                    return conn

            # Create new connection if under limit
            if len(pool) < self._max_per_server:
                resource = await asyncio.get_event_loop().run_in_executor(
                    None, factory
                )

                conn = PooledConnection(
                    server_name=server_name,
                    transport=transport,
                    process=resource if transport == TransportType.STDIO else None,
                    client=resource if transport != TransportType.STDIO else None,
                )
                pool.append(conn)
                return conn

            # Wait for available connection
            for _ in range(10):  # Retry loop
                await asyncio.sleep(0.1)
                for conn in pool:
                    if conn.is_healthy and self._is_valid(conn):
                        conn.last_used = time.time()
                        conn.request_count += 1
                        return conn

            # Force create if all else fails
            resource = await asyncio.get_event_loop().run_in_executor(
                None, factory
            )
            conn = PooledConnection(
                server_name=server_name,
                transport=transport,
                process=resource if transport == TransportType.STDIO else None,
                client=resource if transport != TransportType.STDIO else None,
            )
            pool.append(conn)
            return conn

    def release(self, conn: PooledConnection, error: bool = False) -> None:
        """Release a connection back to the pool."""
        conn.last_used = time.time()
        if error:
            conn.error_count += 1
            if conn.error_count >= 3:
                conn.is_healthy = False

    def _is_valid(self, conn: PooledConnection) -> bool:
        """Check if connection is still valid."""
        now = time.time()

        # Check idle timeout
        if now - conn.last_used > self._idle_timeout:
            return False

        # Check max lifetime
        if now - conn.created_at > self._max_lifetime:
            return False

        # Check health
        if not conn.is_healthy:
            return False

        # Check process for stdio
        if conn.transport == TransportType.STDIO:
            if conn.process and conn.process.poll() is not None:
                return False

        return True

    async def cleanup(self) -> int:
        """Remove stale connections. Returns count of removed connections."""
        removed = 0
        async with self._lock:
            for server_name in list(self._pools.keys()):
                pool = self._pools[server_name]
                valid = []

                for conn in pool:
                    if self._is_valid(conn):
                        valid.append(conn)
                    else:
                        removed += 1
                        await self._close_connection(conn)

                self._pools[server_name] = valid

                if not valid:
                    del self._pools[server_name]

        return removed

    async def _close_connection(self, conn: PooledConnection) -> None:
        """Close a connection."""
        try:
            if conn.process:
                conn.process.terminate()
                try:
                    conn.process.wait(timeout=2.0)
                except Exception:
                    conn.process.kill()

            if conn.client:
                await conn.client.aclose()
        except Exception:
            pass

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while True:
            await asyncio.sleep(60.0)  # Every minute
            await self.cleanup()

    def start_cleanup_loop(self) -> None:
        """Start background cleanup."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def close_all(self) -> None:
        """Close all connections."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None

        async with self._lock:
            for pool in self._pools.values():
                for conn in pool:
                    await self._close_connection(conn)
            self._pools.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        stats = {
            "total_connections": 0,
            "healthy_connections": 0,
            "servers": {},
        }

        for server_name, pool in self._pools.items():
            healthy = sum(1 for c in pool if c.is_healthy)
            stats["total_connections"] += len(pool)
            stats["healthy_connections"] += healthy
            stats["servers"][server_name] = {
                "total": len(pool),
                "healthy": healthy,
                "total_requests": sum(c.request_count for c in pool),
                "total_errors": sum(c.error_count for c in pool),
            }

        return stats


# =============================================================================
# MCPDiscovery - Enhanced MCP Manager with Discovery
# =============================================================================

class MCPDiscovery:
    """
    Enhanced MCP server management with dynamic discovery.

    Combines:
    - Registry-based server discovery
    - Capability negotiation
    - Connection pooling
    - Health-based routing
    """

    def __init__(
        self,
        base_manager: Optional[MCPServerManager] = None,
        enable_registry: bool = True,
        enable_pooling: bool = True,
    ):
        self._manager = base_manager or MCPServerManager()
        self._registry = RegistryClient() if enable_registry else None
        self._negotiator = CapabilityNegotiator()
        self._pool = ConnectionPool() if enable_pooling else None

        self._capabilities: Dict[str, ServerCapabilities] = {}
        self._server_scores: Dict[str, float] = {}  # Health/performance scores

    @property
    def manager(self) -> MCPServerManager:
        """Get underlying server manager."""
        return self._manager

    async def discover_servers(
        self,
        query: str = "",
        capability: str = "",
        auto_register: bool = False,
    ) -> List[RegistryEntry]:
        """
        Discover servers from the MCP Registry.

        Args:
            query: Search term
            capability: Filter by capability
            auto_register: Automatically register discovered servers

        Returns:
            List of discovered servers
        """
        if not self._registry:
            return []

        result = await self._registry.search(query=query, capability=capability)

        if auto_register:
            for entry in result.entries:
                await self.register_from_registry(entry)

        return result.entries

    async def register_from_registry(self, entry: RegistryEntry) -> bool:
        """
        Register a server from a registry entry.

        Args:
            entry: Registry entry to register

        Returns:
            True if registration successful
        """
        try:
            # Build command based on package manager
            if entry.package_manager == "npm":
                command = ["npx", "-y", entry.package_name] if entry.package_name else entry.command
            elif entry.package_manager == "pip":
                command = ["python", "-m", entry.package_name] if entry.package_name else entry.command
            else:
                command = entry.command

            if not command:
                return False

            config = ServerConfig(
                name=entry.name,
                transport=TransportType.STDIO,
                command=command,
                auto_start=False,  # Don't auto-start discovered servers
            )

            self._manager.register_server(config)
            return True

        except Exception:
            return False

    async def negotiate_capabilities(self, server_name: str) -> Optional[ServerCapabilities]:
        """
        Negotiate capabilities with a server.

        Args:
            server_name: Server to negotiate with

        Returns:
            Discovered capabilities or None
        """
        status = self._manager.get_server_status(server_name)
        if not status or status["status"] != ServerStatus.RUNNING.value:
            return None

        instance = self._manager._servers.get(server_name)
        if not instance:
            return None

        capabilities = await self._negotiator.negotiate(
            server_name=server_name,
            transport=instance.config.transport,
            process=instance.process,
            url=instance.config.url,
        )

        self._capabilities[server_name] = capabilities

        # Update server config with discovered capabilities
        instance.config.tools = capabilities.tools
        instance.config.resources = capabilities.resources
        instance.config.prompts = capabilities.prompts

        # Update tool registry
        for tool in capabilities.tools:
            tool.server_name = server_name
            self._manager._tool_registry[tool.name] = server_name

        return capabilities

    async def start_with_negotiation(self, server_name: str) -> bool:
        """
        Start a server and negotiate capabilities.

        Args:
            server_name: Server to start

        Returns:
            True if started and negotiated successfully
        """
        started = await self._manager.start_server(server_name)
        if not started:
            return False

        capabilities = await self.negotiate_capabilities(server_name)
        return capabilities is not None

    async def get_pooled_connection(self, server_name: str) -> Optional[PooledConnection]:
        """
        Get a pooled connection to a server.

        Args:
            server_name: Server to connect to

        Returns:
            PooledConnection or None
        """
        if not self._pool:
            return None

        instance = self._manager._servers.get(server_name)
        if not instance:
            return None

        def factory():
            # For stdio, we can't really pool processes
            # For HTTP, we pool the client
            if instance.config.transport != TransportType.STDIO:
                return httpx.AsyncClient(timeout=30.0)
            return None

        return await self._pool.acquire(
            server_name=server_name,
            transport=instance.config.transport,
            factory=factory,
        )

    def update_server_score(self, server_name: str, latency_ms: float, success: bool) -> None:
        """
        Update health/performance score for a server.

        Args:
            server_name: Server to update
            latency_ms: Request latency
            success: Whether request succeeded
        """
        current = self._server_scores.get(server_name, 100.0)

        # Simple exponential moving average
        if success:
            # Score based on latency (lower is better)
            latency_score = max(0, 100 - latency_ms / 10)
            new_score = current * 0.9 + latency_score * 0.1
        else:
            # Penalize failures
            new_score = current * 0.7

        self._server_scores[server_name] = max(0, min(100, new_score))

    def get_best_server_for_tool(self, tool_name: str) -> Optional[str]:
        """
        Get the best server for a tool based on health scores.

        Args:
            tool_name: Tool to find server for

        Returns:
            Server name or None
        """
        # Find all servers that provide this tool
        candidates = []
        for server_name, capabilities in self._capabilities.items():
            if any(t.name == tool_name for t in capabilities.tools):
                score = self._server_scores.get(server_name, 100.0)
                candidates.append((server_name, score))

        if not candidates:
            # Fall back to basic registry
            return self._manager.get_tool_server(tool_name)

        # Return highest scored server
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def get_all_capabilities(self) -> Dict[str, ServerCapabilities]:
        """Get all discovered capabilities."""
        return self._capabilities.copy()

    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get discovery and pool statistics."""
        stats = {
            "servers_registered": len(self._manager._servers),
            "servers_with_capabilities": len(self._capabilities),
            "total_tools_discovered": sum(
                len(c.tools) for c in self._capabilities.values()
            ),
            "total_resources_discovered": sum(
                len(c.resources) for c in self._capabilities.values()
            ),
            "total_prompts_discovered": sum(
                len(c.prompts) for c in self._capabilities.values()
            ),
            "server_scores": self._server_scores.copy(),
        }

        if self._pool:
            stats["pool"] = self._pool.get_stats()

        return stats

    async def close(self) -> None:
        """Clean up resources."""
        if self._registry:
            await self._registry.close()
        if self._pool:
            await self._pool.close_all()
        await self._manager.stop_all()


# =============================================================================
# Factory Functions
# =============================================================================

def create_discovery_manager(
    enable_registry: bool = True,
    enable_pooling: bool = True,
    config_path: Optional[Path] = None,
) -> MCPDiscovery:
    """
    Create an MCP discovery manager.

    Args:
        enable_registry: Enable registry-based discovery
        enable_pooling: Enable connection pooling
        config_path: Path to .mcp.json config file

    Returns:
        Configured MCPDiscovery instance
    """
    base_manager = MCPServerManager.from_config_file(config_path) if config_path else MCPServerManager()

    return MCPDiscovery(
        base_manager=base_manager,
        enable_registry=enable_registry,
        enable_pooling=enable_pooling,
    )


def create_registry_client(cache_dir: Optional[Path] = None) -> RegistryClient:
    """Create a registry client for server discovery."""
    return RegistryClient(cache_dir=cache_dir)


def create_connection_pool(
    max_per_server: int = 5,
    idle_timeout: float = 300.0,
) -> ConnectionPool:
    """Create a connection pool."""
    return ConnectionPool(
        max_connections_per_server=max_per_server,
        idle_timeout=idle_timeout,
    )


# =============================================================================
# Demo
# =============================================================================

def main():
    """Demo MCP Discovery."""
    print("=" * 60)
    print("MCP DISCOVERY MODULE DEMO")
    print("=" * 60)
    print()

    # Create discovery manager
    discovery = MCPDiscovery(enable_registry=False)  # Registry may not be accessible

    # Register some servers manually
    discovery.manager.register_server(ServerConfig(
        name="memory",
        transport=TransportType.STDIO,
        command=["python", "-m", "memory_server"],
        tools=[
            ToolSchema(name="memory_store", description="Store a memory"),
            ToolSchema(name="memory_recall", description="Recall a memory"),
        ]
    ))

    discovery.manager.register_server(ServerConfig(
        name="filesystem",
        transport=TransportType.STDIO,
        command=["npx", "-y", "@anthropic/mcp-server-filesystem"],
        tools=[
            ToolSchema(name="read_file", description="Read a file"),
            ToolSchema(name="write_file", description="Write a file"),
        ]
    ))

    # Simulate capability discovery
    discovery._capabilities["memory"] = ServerCapabilities(
        server_name="memory",
        tools=[
            ToolSchema(name="memory_store", description="Store a memory"),
            ToolSchema(name="memory_recall", description="Recall a memory"),
            ToolSchema(name="memory_search", description="Search memories"),  # Discovered!
        ],
        supports_tools=True,
        supports_resources=True,
    )

    discovery._server_scores["memory"] = 95.0
    discovery._server_scores["filesystem"] = 88.0

    # Get stats
    print("[>>] Discovery Stats:")
    stats = discovery.get_discovery_stats()
    print(f"  Servers registered: {stats['servers_registered']}")
    print(f"  Servers with capabilities: {stats['servers_with_capabilities']}")
    print(f"  Total tools discovered: {stats['total_tools_discovered']}")

    print("\n[>>] Server Scores:")
    for name, score in stats["server_scores"].items():
        print(f"  {name}: {score:.1f}/100")

    print("\n[>>] Discovered Capabilities:")
    for name, caps in discovery.get_all_capabilities().items():
        print(f"  {name}:")
        print(f"    Protocol: {caps.protocol_version}")
        print(f"    Tools: {[t.name for t in caps.tools]}")
        print(f"    Supports Resources: {caps.supports_resources}")

    # Test best server selection
    print("\n[>>] Best Server for 'memory_store':")
    best = discovery.get_best_server_for_tool("memory_store")
    print(f"  {best}")

    print("\n[OK] MCP Discovery demo complete")


if __name__ == "__main__":
    main()
