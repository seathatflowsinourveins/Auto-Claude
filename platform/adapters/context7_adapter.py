"""
Context7 SDK Documentation Adapter - Up-to-date Library Docs (MCP-Only)
=======================================================================

Context7 provides real-time SDK and library documentation lookup,
ensuring you always have the latest API information.

V40 MIGRATION: REST API (api.context7.com) has been deprecated.
This adapter now uses MCP-based communication exclusively.

Features:
- resolve-library-id: Get library ID for documentation queries
- get-library-docs: Query specific library documentation with version support
- Topic-based filtering for focused documentation retrieval
- Pagination support (pages 1-10) for comprehensive coverage
- Supports all major libraries (React, LangChain, DSPy, etc.)
- Real-time updates from official sources

MCP Endpoints:
- MCP Remote: https://mcp.context7.com/mcp (for HTTP transport)
- MCP Local: npx -y @upstash/context7-mcp@2.1.1 (for stdio transport)

MCP Tools (v2.1.1):
1. resolve-library-id:
   - libraryName (required): Name of library to search
   - Returns: Context7-compatible library ID (e.g., /mongodb/docs, /vercel/next.js)

2. get-library-docs:
   - context7CompatibleLibraryID (required): Exact library ID from resolve step
   - topic (optional): Focus docs on specific topic (e.g., "routing", "hooks")
   - tokens (optional, default 10000): Maximum tokens to return

Best Practices:
- Use library ID directly in prompts: "use library /supabase/supabase for API"
- Add topic filter for focused results: topic="authentication"
- Cache resolved library IDs - they rarely change

Documentation: https://context7.com/docs
Package: @upstash/context7-mcp v2.1.1

Usage:
    adapter = Context7Adapter()
    await adapter.initialize({})

    # Resolve library first
    result = await adapter.execute("resolve", library_name="langchain")
    # Returns: {"library_id": "/langchain/langchain", "name": "langchain", ...}

    # Then query docs with topic filter
    result = await adapter.execute("query", library_id="/langchain/langchain",
                                    query="StateGraph", topic="graphs")

    # Multi-library query
    result = await adapter.execute("multi_query",
                                    libraries=["langchain", "langgraph"],
                                    query="agent orchestration")
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Literal, Optional, Tuple
from enum import Enum

import httpx

logger = logging.getLogger(__name__)

# Retry utilities
try:
    from .retry import RetryConfig, with_retry, retry_async, http_request_with_retry
except ImportError:
    # Fallback for standalone testing
    RetryConfig = None
    with_retry = lambda f=None, **kw: (lambda fn: fn) if f is None else f
    retry_async = None
    http_request_with_retry = None

# SDK Layer imports
try:
    from core.orchestration.base import SDKAdapter, AdapterResult, AdapterStatus, SDKLayer, register_adapter
except ImportError:
    try:
        from core.orchestration.base import SDKAdapter, AdapterResult, AdapterStatus, SDKLayer, register_adapter
    except ImportError:
        # Minimal fallback definitions for standalone use
        from enum import IntEnum
        from abc import ABC, abstractmethod

        class SDKLayer(IntEnum):
            KNOWLEDGE = 8

        class AdapterStatus(Enum):
            UNINITIALIZED = "uninitialized"
            READY = "ready"
            FAILED = "failed"
            ERROR = "error"

        @dataclass
        class AdapterResult:
            success: bool
            data: Optional[Dict[str, Any]] = None
            error: Optional[str] = None
            latency_ms: float = 0.0
            cached: bool = False
            metadata: Dict[str, Any] = field(default_factory=dict)
            timestamp: datetime = field(default_factory=datetime.utcnow)

        class SDKAdapter(ABC):
            @property
            @abstractmethod
            def sdk_name(self) -> str: ...
            @abstractmethod
            async def initialize(self, config: Dict) -> AdapterResult: ...
            @abstractmethod
            async def execute(self, operation: str, **kwargs) -> AdapterResult: ...
            @abstractmethod
            async def shutdown(self) -> None: ...

        def register_adapter(name, layer, priority=0):
            def decorator(cls):
                return cls
            return decorator


# Context7 MCP endpoints (V40: REST API deprecated, MCP-only)
# api.context7.com no longer resolves - use MCP exclusively
CONTEXT7_MCP_REMOTE_URL = "https://mcp.context7.com/mcp"
CONTEXT7_MCP_PACKAGE = "@upstash/context7-mcp@2.1.1"

# MCP tool names
MCP_TOOL_RESOLVE_LIBRARY = "resolve-library-id"
MCP_TOOL_GET_LIBRARY_DOCS = "get-library-docs"

# Common library ID mappings (Context7 uses GitHub-style paths)
# Priority libraries are resolved locally for performance
LIBRARY_ID_MAPPINGS = {
    # AI/ML Frameworks (Priority 1 - Most commonly used)
    "langchain": "/langchain-ai/langchain",
    "langgraph": "/langchain-ai/langgraph",
    "dspy": "/stanfordnlp/dspy",
    "openai": "/openai/openai-python",
    "anthropic": "/anthropics/anthropic-sdk-python",
    "claude-sdk": "/anthropics/anthropic-sdk-python",
    "claude": "/anthropics/anthropic-sdk-python",
    "letta": "/letta-ai/letta",
    "crewai": "/joaomdmoura/crewai",
    "autogen": "/microsoft/autogen",

    # Web Frameworks (Priority 2)
    "react": "/facebook/react",
    "nextjs": "/vercel/next.js",
    "next": "/vercel/next.js",
    "next.js": "/vercel/next.js",
    "fastapi": "/tiangolo/fastapi",
    "django": "/django/django",
    "flask": "/pallets/flask",
    "express": "/expressjs/express",
    "vue": "/vuejs/vue",
    "angular": "/angular/angular",
    "svelte": "/sveltejs/svelte",
    "astro": "/withastro/astro",
    "remix": "/remix-run/remix",
    "nuxt": "/nuxt/nuxt",

    # ML/Data Science (Priority 3)
    "pytorch": "/pytorch/pytorch",
    "torch": "/pytorch/pytorch",
    "huggingface": "/huggingface/transformers",
    "transformers": "/huggingface/transformers",
    "tensorflow": "/tensorflow/tensorflow",
    "keras": "/keras-team/keras",
    "numpy": "/numpy/numpy",
    "pandas": "/pandas-dev/pandas",
    "scikit-learn": "/scikit-learn/scikit-learn",
    "sklearn": "/scikit-learn/scikit-learn",
    "jax": "/google/jax",
    "pydantic": "/pydantic/pydantic",

    # Database/Backend (Priority 4)
    "prisma": "/prisma/prisma",
    "supabase": "/supabase/supabase",
    "firebase": "/firebase/firebase-js-sdk",
    "mongodb": "/mongodb/docs",
    "postgres": "/postgres/postgres",
    "redis": "/redis/redis",
    "sqlalchemy": "/sqlalchemy/sqlalchemy",
    "drizzle": "/drizzle-team/drizzle-orm",

    # Cloud/Infrastructure (Priority 5)
    "cloudflare": "/cloudflare/workers-sdk",
    "aws-cdk": "/aws/aws-cdk",
    "terraform": "/hashicorp/terraform",
    "vercel": "/vercel/vercel",
    "docker": "/docker/docs",
    "kubernetes": "/kubernetes/kubernetes",

    # Utilities (Priority 6)
    "typescript": "/microsoft/TypeScript",
    "python": "/python/cpython",
    "requests": "/psf/requests",
    "axios": "/axios/axios",
    "zod": "/colinhacks/zod",
    "trpc": "/trpc/trpc",
    "tanstack-query": "/TanStack/query",
    "react-query": "/TanStack/query",
    "playwright": "/microsoft/playwright",
    "vitest": "/vitest-dev/vitest",
    "jest": "/jestjs/jest",
    "tailwind": "/tailwindlabs/tailwindcss",
    "tailwindcss": "/tailwindlabs/tailwindcss",

    # MCP/Agent Tools (Priority 7 - Special focus)
    "mcp": "/modelcontextprotocol/docs",
    "model-context-protocol": "/modelcontextprotocol/docs",
    "claude-flow": "/ruvnet/claude-flow",
    "exa": "/exa-labs/exa-docs",
    "tavily": "/tavily-ai/tavily-python",
}

# Priority whitelist for auto-resolution (skip API call for these)
PRIORITY_LIBRARIES = {
    "langchain", "langgraph", "openai", "anthropic", "claude",
    "react", "nextjs", "next", "fastapi", "typescript",
    "pytorch", "transformers", "prisma", "supabase",
    "dspy", "letta", "crewai", "mcp", "pydantic",
}

# Version pinning for stable documentation access
VERSION_PINS = {
    "langchain": "0.3",
    "langgraph": "0.2",
    "openai": "1.x",
    "anthropic": "0.40",
    "nextjs": "15",
    "react": "19",
    "fastapi": "0.115",
    "pydantic": "2.x",
}


@dataclass
class Context7Library:
    """Resolved library information from Context7 API."""
    library_id: str  # e.g., "/langchain-ai/langchain"
    name: str
    description: Optional[str] = None
    trust_score: Optional[float] = None
    benchmark_score: Optional[float] = None
    total_snippets: Optional[int] = None
    versions: list = field(default_factory=list)
    topics: list = field(default_factory=list)  # Available documentation topics


@dataclass
class Context7DocsResult:
    """Documentation result from Context7 query."""
    library_id: str
    query: str
    topic: Optional[str] = None
    page: int = 1
    content: str = ""
    code_examples: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    has_more_pages: bool = False


@register_adapter("context7", SDKLayer.KNOWLEDGE, priority=24)
class Context7Adapter(SDKAdapter):
    """
    Context7 SDK documentation adapter using MCP protocol (V40 migration).

    V40 CHANGE: api.context7.com has been deprecated. This adapter now uses
    MCP-based communication exclusively via mcp.context7.com or local npx.

    Operations:
        - resolve: Resolve library name to Context7 library ID
        - query: Query library documentation with topic and pagination
        - get_docs: Combined resolve + query for convenience
        - search: Search across all libraries (via resolve)
        - multi_query: Query multiple libraries in parallel
        - get_code_examples: Extract code examples with syntax highlighting

    MCP Tool Mapping:
        - resolve -> resolve-library-id (libraryName)
        - query -> get-library-docs (context7CompatibleLibraryID, topic, tokens)
    """

    OPERATIONS = {
        "resolve": "_resolve_library",
        "query": "_query_docs",
        "get_docs": "_get_docs",
        "search": "_search_libraries",
        "multi_query": "_multi_library_query",
        "get_code_examples": "_get_code_examples",
    }

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize Context7 adapter.

        Args:
            api_key: Optional API key for higher rate limits.
                     Can also be set via CONTEXT7_API_KEY env var.
            transport: Transport type - "stdio" (local npx) or "http" (remote MCP)
            mcp_timeout: Timeout for MCP operations (default 30s)
        """
        self._api_key = api_key
        self._status = AdapterStatus.UNINITIALIZED
        self._library_cache: Dict[str, Context7Library] = {}
        self._docs_cache: Dict[str, Context7DocsResult] = {}
        self._mock_mode = False
        self._use_priority_whitelist = kwargs.get("use_priority_whitelist", True)
        self._default_page_size = kwargs.get("default_page_size", 1)
        self._max_pages = kwargs.get("max_pages", 10)

        # MCP transport configuration (V40)
        self._transport = kwargs.get("transport", "http")  # "stdio" or "http"
        self._mcp_timeout = kwargs.get("mcp_timeout", 30.0)
        self._mcp_process: Optional[subprocess.Popen] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._request_id = 0

        self._stats = {
            "queries": 0,
            "resolves": 0,
            "cache_hits": 0,
            "mcp_calls": 0,
            "avg_latency_ms": 0.0,
            "retries": 0,
            "priority_hits": 0,  # Resolved from whitelist
            "multi_queries": 0,
            "transport": self._transport,
        }
        # Retry configuration for transient errors
        self._retry_config = RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=60.0,
            jitter=0.5,
        ) if RetryConfig else None

    @property
    def sdk_name(self) -> str:
        return "context7"

    @property
    def layer(self) -> SDKLayer:
        return SDKLayer.KNOWLEDGE

    @property
    def available(self) -> bool:
        return self._status == AdapterStatus.READY

    def _normalize_library_id(self, library_name: str) -> str:
        """
        Normalize library name to Context7 format.

        Context7 uses GitHub-style paths like "/langchain-ai/langchain".

        Args:
            library_name: Raw library name (e.g., "langchain", "react")

        Returns:
            Normalized library ID (e.g., "/langchain-ai/langchain")
        """
        # Already in correct format
        if library_name.startswith("/"):
            return library_name

        # Check known mappings
        normalized = library_name.lower().strip()
        if normalized in LIBRARY_ID_MAPPINGS:
            return LIBRARY_ID_MAPPINGS[normalized]

        # Default format: /{name}/{name}
        clean_name = normalized.replace(" ", "-").replace("_", "-")
        return f"/{clean_name}/{clean_name}"

    def _is_priority_library(self, library_name: str) -> bool:
        """Check if library is in priority whitelist for fast resolution."""
        normalized = library_name.lower().strip()
        return normalized in PRIORITY_LIBRARIES

    def _get_pinned_version(self, library_name: str) -> Optional[str]:
        """Get pinned version for stable documentation access."""
        normalized = library_name.lower().strip()
        return VERSION_PINS.get(normalized)

    def _generate_cache_key(self, library_id: str, query: str, topic: Optional[str], page: int) -> str:
        """Generate cache key for documentation queries."""
        return f"{library_id}:{query}:{topic or 'none'}:{page}"

    async def initialize(self, config: Dict[str, Any]) -> AdapterResult:
        """
        Initialize Context7 adapter with MCP connection (V40 migration).

        V40: REST API (api.context7.com) deprecated. Now uses MCP exclusively.

        Args:
            config: Configuration dict with optional keys:
                - api_key: Optional API key for rate limits
                - mock_mode: Enable mock mode for testing
                - transport: "http" (remote MCP) or "stdio" (local npx)
                - mcp_timeout: Timeout for MCP operations
        """
        start = time.time()

        # Get configuration
        self._api_key = config.get("api_key") or self._api_key or os.environ.get("CONTEXT7_API_KEY")
        self._mock_mode = config.get("mock_mode", False)
        self._transport = config.get("transport", self._transport)
        self._mcp_timeout = config.get("mcp_timeout", self._mcp_timeout)

        # Mock mode bypasses MCP connection
        if self._mock_mode:
            self._status = AdapterStatus.READY
            return AdapterResult(
                success=True,
                data={
                    "status": "ready",
                    "mode": "mock",
                    "transport": "mock",
                    "known_libraries": list(LIBRARY_ID_MAPPINGS.keys()),
                },
                latency_ms=(time.time() - start) * 1000,
            )

        # Initialize MCP connection based on transport type
        try:
            if self._transport == "stdio":
                # Local MCP server via npx
                await self._init_stdio_transport()
            else:
                # Remote MCP via HTTP
                await self._init_http_transport()

            # Verify MCP connectivity with a health check
            health_result = await self._mcp_health_check()
            if not health_result:
                # Fall back to priority whitelist mode
                self._status = AdapterStatus.READY
                return AdapterResult(
                    success=True,
                    data={
                        "status": "ready",
                        "mode": "limited",
                        "transport": self._transport,
                        "warning": "MCP connectivity limited. Using cached library mappings.",
                        "known_libraries": list(LIBRARY_ID_MAPPINGS.keys()),
                    },
                    latency_ms=(time.time() - start) * 1000,
                )

            self._status = AdapterStatus.READY
            return AdapterResult(
                success=True,
                data={
                    "status": "ready",
                    "mode": "mcp",
                    "transport": self._transport,
                    "api_key_set": bool(self._api_key),
                    "known_libraries": list(LIBRARY_ID_MAPPINGS.keys()),
                    "mcp_tools": [MCP_TOOL_RESOLVE_LIBRARY, MCP_TOOL_GET_LIBRARY_DOCS],
                },
                latency_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            logger.warning("MCP initialization failed: %s. Falling back to whitelist mode.", e)
            self._status = AdapterStatus.READY
            return AdapterResult(
                success=True,
                data={
                    "status": "ready",
                    "mode": "limited",
                    "transport": self._transport,
                    "warning": f"MCP initialization failed: {e}. Using cached library mappings.",
                    "known_libraries": list(LIBRARY_ID_MAPPINGS.keys()),
                },
                latency_ms=(time.time() - start) * 1000,
            )

    async def _init_stdio_transport(self) -> None:
        """Initialize stdio transport using local npx MCP server."""
        if self._mcp_process is not None:
            return

        # Build command - use npx.cmd on Windows
        npx_cmd = "npx.cmd" if os.name == "nt" else "npx"
        cmd = [npx_cmd, "-y", CONTEXT7_MCP_PACKAGE]

        # Set environment
        env = dict(os.environ)
        if self._api_key:
            env["CONTEXT7_API_KEY"] = self._api_key

        self._mcp_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Send MCP initialize request
        await self._mcp_initialize()

    async def _init_http_transport(self) -> None:
        """Initialize HTTP transport for remote MCP server."""
        if self._http_client is not None:
            return

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        self._http_client = httpx.AsyncClient(
            timeout=self._mcp_timeout,
            headers=headers,
        )

    async def _mcp_initialize(self) -> bool:
        """Send MCP initialize handshake."""
        init_request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "clientInfo": {
                    "name": "UNLEASH-Context7-Adapter",
                    "version": "2.0.0",
                },
            },
        }

        try:
            response = await self._send_mcp_request(init_request)
            if response and "result" in response:
                # Send initialized notification
                await self._send_mcp_notification({
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                })
                return True
        except Exception as e:
            logger.debug("MCP initialize failed: %s", e)

        return False

    async def _mcp_health_check(self) -> bool:
        """Check MCP endpoint health by testing tool availability."""
        try:
            # Try to list tools
            list_request = {
                "jsonrpc": "2.0",
                "id": self._next_request_id(),
                "method": "tools/list",
                "params": {},
            }
            response = await self._send_mcp_request(list_request)

            if response and "result" in response:
                tools = response["result"].get("tools", [])
                tool_names = [t.get("name") for t in tools]
                # Check if expected tools are available
                return MCP_TOOL_RESOLVE_LIBRARY in tool_names or MCP_TOOL_GET_LIBRARY_DOCS in tool_names

        except Exception as e:
            logger.debug("MCP health check failed: %s", e)

        return False

    def _next_request_id(self) -> int:
        """Get next JSON-RPC request ID."""
        self._request_id += 1
        return self._request_id

    async def _send_mcp_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send MCP JSON-RPC request and get response."""
        self._stats["mcp_calls"] += 1

        if self._transport == "stdio":
            return await self._send_stdio_request(request)
        else:
            return await self._send_http_request(request)

    async def _send_mcp_notification(self, notification: Dict[str, Any]) -> None:
        """Send MCP notification (no response expected)."""
        if self._transport == "stdio":
            await self._send_stdio_notification(notification)
        # HTTP transport doesn't need notifications

    async def _send_stdio_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send request via stdio transport."""
        if not self._mcp_process or not self._mcp_process.stdin or not self._mcp_process.stdout:
            return None

        try:
            # Write request
            request_data = json.dumps(request) + "\n"
            self._mcp_process.stdin.write(request_data.encode())
            self._mcp_process.stdin.flush()

            # Read response with timeout
            loop = asyncio.get_event_loop()
            response_line = await asyncio.wait_for(
                loop.run_in_executor(None, self._mcp_process.stdout.readline),
                timeout=self._mcp_timeout,
            )

            if response_line:
                return json.loads(response_line.decode())

        except asyncio.TimeoutError:
            logger.debug("MCP stdio request timed out")
        except Exception as e:
            logger.debug("MCP stdio request failed: %s", e)

        return None

    async def _send_stdio_notification(self, notification: Dict[str, Any]) -> None:
        """Send notification via stdio transport."""
        if not self._mcp_process or not self._mcp_process.stdin:
            return

        try:
            notification_data = json.dumps(notification) + "\n"
            self._mcp_process.stdin.write(notification_data.encode())
            self._mcp_process.stdin.flush()
        except Exception as e:
            logger.debug("MCP stdio notification failed: %s", e)

    async def _send_http_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send request via HTTP transport to remote MCP server."""
        if not self._http_client:
            return None

        try:
            response = await self._http_client.post(
                CONTEXT7_MCP_REMOTE_URL,
                json=request,
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.debug("MCP HTTP request failed: %d", response.status_code)

        except Exception as e:
            logger.debug("MCP HTTP request failed: %s", e)

        return None

    async def _call_mcp_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Call an MCP tool and return the result.

        Args:
            tool_name: Name of MCP tool to call
            arguments: Tool arguments

        Returns:
            Tuple of (success, result_data, error_message)
        """
        request = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }

        response = await self._send_mcp_request(request)

        if not response:
            return False, None, "No response from MCP server"

        if "error" in response:
            error = response["error"]
            return False, None, f"MCP error: {error.get('message', str(error))}"

        if "result" in response:
            result = response["result"]
            # MCP tool results come in content array
            if isinstance(result, dict) and "content" in result:
                content = result["content"]
                if isinstance(content, list) and content:
                    # Extract text content
                    for item in content:
                        if item.get("type") == "text":
                            text = item.get("text", "")
                            # Check if text contains MCP error message
                            if text.startswith("MCP error") or "Tool" in text and "not found" in text:
                                return False, None, text
                            # Try to parse as JSON
                            try:
                                return True, json.loads(text), None
                            except json.JSONDecodeError:
                                return True, {"text": text}, None
                    return True, {"content": content}, None
                # Check for isError flag in content
                if result.get("isError"):
                    error_text = str(content[0].get("text", content) if isinstance(content, list) else content)
                    return False, None, error_text
            return True, result, None

        return False, None, "Invalid MCP response format"

    async def execute(self, operation: str, **kwargs) -> AdapterResult:
        """Execute Context7 operation."""
        if self._status != AdapterStatus.READY:
            return AdapterResult(
                success=False,
                error="Context7 adapter not initialized. Call initialize() first.",
            )

        if operation not in self.OPERATIONS:
            return AdapterResult(
                success=False,
                error=f"Unknown operation: {operation}. Available: {list(self.OPERATIONS.keys())}",
            )

        method_name = self.OPERATIONS[operation]
        method = getattr(self, method_name)
        return await method(**kwargs)

    async def _resolve_library(
        self,
        library_name: str,
        query: str = "",
        **kwargs,
    ) -> AdapterResult:
        """
        Resolve library name to Context7 library ID using MCP.

        V40: Now uses MCP tool 'resolve-library-id' instead of REST API.

        Args:
            library_name: The name of the library to resolve (e.g., "langchain")
            query: Optional query (not used in MCP, kept for compatibility)

        Returns:
            AdapterResult with library_id, name, description, trust_score, etc.
        """
        start = time.time()
        self._stats["resolves"] += 1

        if not library_name:
            return AdapterResult(
                success=False,
                error="library_name is required",
                latency_ms=(time.time() - start) * 1000,
            )

        # Check cache first
        cache_key = library_name.lower()
        if cache_key in self._library_cache:
            self._stats["cache_hits"] += 1
            lib = self._library_cache[cache_key]
            return AdapterResult(
                success=True,
                data={
                    "library_id": lib.library_id,
                    "name": lib.name,
                    "description": lib.description,
                    "trust_score": lib.trust_score,
                    "benchmark_score": lib.benchmark_score,
                    "total_snippets": lib.total_snippets,
                    "versions": lib.versions,
                    "topics": lib.topics,
                },
                cached=True,
                latency_ms=(time.time() - start) * 1000,
            )

        # Fast path: Use priority whitelist for known libraries
        if self._use_priority_whitelist and self._is_priority_library(library_name):
            self._stats["priority_hits"] += 1
            normalized_id = self._normalize_library_id(library_name)
            lib = Context7Library(
                library_id=normalized_id,
                name=library_name,
                description=f"Priority-resolved documentation for {library_name}",
                trust_score=1.0,  # Trusted whitelist entry
            )
            self._library_cache[cache_key] = lib
            return AdapterResult(
                success=True,
                data={
                    "library_id": lib.library_id,
                    "name": lib.name,
                    "description": lib.description,
                    "trust_score": lib.trust_score,
                    "pinned_version": self._get_pinned_version(library_name),
                },
                latency_ms=(time.time() - start) * 1000,
                metadata={"priority_resolved": True},
            )

        # Mock mode returns synthetic data
        if self._mock_mode:
            return self._mock_resolve(library_name, start)

        # Call MCP tool: resolve-library-id
        try:
            success, data, error = await self._call_mcp_tool(
                MCP_TOOL_RESOLVE_LIBRARY,
                {"libraryName": library_name},
            )

            if not success:
                # Fall back to normalized ID on MCP error
                normalized_id = self._normalize_library_id(library_name)
                return AdapterResult(
                    success=True,
                    data={
                        "library_id": normalized_id,
                        "name": library_name,
                        "description": f"MCP resolution failed. Using normalized ID.",
                    },
                    latency_ms=(time.time() - start) * 1000,
                    metadata={"fallback": True, "error": error},
                )

            # Parse MCP response
            # Context7 MCP returns library ID in various formats
            library_id = None
            description = None

            if isinstance(data, dict):
                # Check for direct library_id field
                library_id = data.get("libraryId") or data.get("library_id") or data.get("id")
                description = data.get("description")

                # Check for text content that contains the library ID
                if not library_id and "text" in data:
                    text = data["text"]
                    # Parse text that might contain library ID (e.g., "/langchain-ai/langchain")
                    if text.startswith("/"):
                        library_id = text.split()[0] if " " in text else text
                    elif "/" in text:
                        # Extract path from text
                        for part in text.split():
                            if part.startswith("/"):
                                library_id = part
                                break

            if not library_id:
                # Final fallback to normalized ID
                library_id = self._normalize_library_id(library_name)

            lib = Context7Library(
                library_id=library_id,
                name=library_name,
                description=description or f"Documentation for {library_name}",
                trust_score=0.9,  # MCP-resolved
            )

            # Cache the result
            self._library_cache[cache_key] = lib

            return AdapterResult(
                success=True,
                data={
                    "library_id": lib.library_id,
                    "name": lib.name,
                    "description": lib.description,
                    "trust_score": lib.trust_score,
                },
                latency_ms=(time.time() - start) * 1000,
                metadata={"mcp_resolved": True},
            )

        except Exception as e:
            # Use fallback for any errors
            normalized_id = self._normalize_library_id(library_name)
            return AdapterResult(
                success=True,
                data={
                    "library_id": normalized_id,
                    "name": library_name,
                    "description": f"Error resolving library: {e}. Using normalized ID.",
                },
                latency_ms=(time.time() - start) * 1000,
                metadata={"fallback": True, "error": str(e)},
            )

    async def _query_docs(
        self,
        library_id: str,
        query: str,
        tokens: int = 10000,
        format: Literal["json", "txt"] = "json",
        version: Optional[str] = None,
        topic: Optional[str] = None,
        page: int = 1,
        **kwargs,
    ) -> AdapterResult:
        """
        Query documentation for a library using MCP.

        V40: Now uses MCP tool 'get-library-docs' instead of REST API.

        Maps to MCP tool: get-library-docs

        Args:
            library_id: Context7 library ID (e.g., "/langchain-ai/langchain")
            query: The question or task to get relevant documentation for
            tokens: Max tokens to return (default 10,000)
            format: Response format (kept for compatibility, MCP returns text)
            version: Optional specific version to query (not in MCP spec)
            topic: Optional topic focus (e.g., "routing", "hooks", "authentication")
            page: Pagination (not in current MCP spec, kept for compatibility)

        Returns:
            AdapterResult with context, snippets, and metadata

        Best Practices:
            - Use topic filter for focused results: topic="authentication"
            - Cache resolved library IDs - they rarely change
        """
        start = time.time()
        self._stats["queries"] += 1

        if not library_id:
            return AdapterResult(
                success=False,
                error="library_id is required. Use 'resolve' operation first.",
                latency_ms=(time.time() - start) * 1000,
            )

        # Normalize library ID if needed
        if not library_id.startswith("/"):
            library_id = self._normalize_library_id(library_id)

        # Check docs cache (using query as part of key since MCP doesn't have pagination)
        cache_key = self._generate_cache_key(library_id, query or "general", topic, 1)
        if cache_key in self._docs_cache:
            self._stats["cache_hits"] += 1
            cached_result = self._docs_cache[cache_key]
            return AdapterResult(
                success=True,
                data={
                    "context": cached_result.content,
                    "code_examples": cached_result.code_examples,
                    "library_id": cached_result.library_id,
                    "query": cached_result.query,
                    "topic": cached_result.topic,
                    "page": cached_result.page,
                    "has_more_pages": cached_result.has_more_pages,
                },
                cached=True,
                latency_ms=0.1,
            )

        # Enforce minimum token limit
        tokens = max(5000, tokens)

        # Mock mode returns synthetic data
        if self._mock_mode:
            return self._mock_query(library_id, query or "general", start)

        # Build MCP tool arguments
        # MCP tool: get-library-docs
        # Args: context7CompatibleLibraryID (required), topic (optional), tokens (optional)
        mcp_args: Dict[str, Any] = {
            "context7CompatibleLibraryID": library_id,
        }
        if topic:
            mcp_args["topic"] = topic
        if tokens != 10000:
            mcp_args["tokens"] = tokens

        # Call MCP tool: get-library-docs
        try:
            success, data, error = await self._call_mcp_tool(
                MCP_TOOL_GET_LIBRARY_DOCS,
                mcp_args,
            )

            # Update latency stats
            latency = (time.time() - start) * 1000
            self._stats["avg_latency_ms"] = (
                (self._stats["avg_latency_ms"] * (self._stats["queries"] - 1) + latency)
                / self._stats["queries"]
            )

            if not success:
                return AdapterResult(
                    success=False,
                    error=f"MCP query failed: {error}",
                    latency_ms=latency,
                )

            # Parse MCP response
            context = ""
            snippets = []
            code_examples = []

            if isinstance(data, dict):
                # Handle different response formats
                if "text" in data:
                    context = data["text"]
                elif "context" in data:
                    context = data["context"]
                elif "content" in data:
                    content = data["content"]
                    if isinstance(content, str):
                        context = content
                    elif isinstance(content, list):
                        context = "\n".join(
                            item.get("text", str(item))
                            for item in content
                            if isinstance(item, dict)
                        )

                # Extract code examples from context
                if context:
                    code_examples = self._extract_code_examples(context)

                snippets = data.get("snippets", [])

            # Cache the result
            docs_result = Context7DocsResult(
                library_id=library_id,
                query=query or "general",
                topic=topic,
                page=1,
                content=context,
                code_examples=code_examples,
                has_more_pages=False,
            )
            self._docs_cache[cache_key] = docs_result

            return AdapterResult(
                success=True,
                data={
                    "context": context,
                    "snippets": snippets,
                    "code_examples": code_examples,
                    "metadata": {"mcp_transport": self._transport},
                    "library_id": library_id,
                    "query": query,
                    "topic": topic,
                    "tokens_requested": tokens,
                    "version": version,
                },
                latency_ms=latency,
            )

        except Exception as e:
            return AdapterResult(
                success=False,
                error=f"MCP request failed: {e}",
                latency_ms=(time.time() - start) * 1000,
            )

    def _extract_code_examples(self, context: str) -> list:
        """Extract code examples from markdown context."""
        import re
        code_examples = []
        # Match markdown code blocks
        code_pattern = r"```(\w*)\n(.*?)```"
        matches = re.findall(code_pattern, context, re.DOTALL)
        for i, (lang, code) in enumerate(matches[:10]):  # Limit to 10 examples
            code_examples.append({
                "language": lang or "text",
                "code": code.strip(),
                "index": i,
            })
        return code_examples

    async def _get_docs(
        self,
        library_name: str,
        query: str,
        tokens: int = 10000,
        version: Optional[str] = None,
        topic: Optional[str] = None,
        page: int = 1,
        **kwargs,
    ) -> AdapterResult:
        """
        Convenience method: Combined resolve + query in one call.

        Args:
            library_name: Library name to resolve and query
            query: The question or task to get documentation for
            tokens: Max tokens to return
            version: Optional specific version (or uses pinned version)
            topic: Optional topic filter (e.g., "routing", "hooks")
            page: Pagination 1-10

        Returns:
            AdapterResult with documentation context
        """
        start = time.time()

        # Step 1: Resolve library
        resolve_result = await self._resolve_library(library_name, query)
        if not resolve_result.success:
            return resolve_result

        library_id = resolve_result.data.get("library_id")
        if not library_id:
            return AdapterResult(
                success=False,
                error=f"Could not resolve library: {library_name}",
                latency_ms=(time.time() - start) * 1000,
            )

        # Use pinned version if not specified
        if not version:
            version = self._get_pinned_version(library_name)

        # Step 2: Query docs with topic and pagination
        query_result = await self._query_docs(
            library_id=library_id,
            query=query,
            tokens=tokens,
            version=version,
            topic=topic,
            page=page,
        )

        # Add resolve info to result
        if query_result.success and query_result.data:
            query_result.data["resolved_from"] = library_name
            query_result.data["resolve_info"] = resolve_result.data
            query_result.data["pinned_version"] = version

        return query_result

    async def _multi_library_query(
        self,
        libraries: list,
        query: str,
        topic: Optional[str] = None,
        tokens_per_library: int = 5000,
        **kwargs,
    ) -> AdapterResult:
        """
        Query multiple libraries in parallel for comprehensive documentation.

        Args:
            libraries: List of library names to query
            query: The question or task to get documentation for
            topic: Optional topic filter applied to all libraries
            tokens_per_library: Max tokens per library (default 5000)

        Returns:
            AdapterResult with combined documentation from all libraries
        """
        start = time.time()
        self._stats["multi_queries"] += 1

        if not libraries:
            return AdapterResult(
                success=False,
                error="libraries list is required",
                latency_ms=(time.time() - start) * 1000,
            )

        # Query all libraries in parallel
        tasks = [
            self._get_docs(
                library_name=lib,
                query=query,
                tokens=tokens_per_library,
                topic=topic,
            )
            for lib in libraries
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        combined_context = []
        all_snippets = []
        successful_libraries = []
        failed_libraries = []

        for lib, result in zip(libraries, results):
            if isinstance(result, Exception):
                failed_libraries.append({"library": lib, "error": str(result)})
            elif result.success:
                successful_libraries.append(lib)
                if result.data.get("context"):
                    combined_context.append(f"## {lib}\n{result.data['context']}")
                if result.data.get("snippets"):
                    all_snippets.extend(result.data["snippets"])
            else:
                failed_libraries.append({"library": lib, "error": result.error})

        return AdapterResult(
            success=len(successful_libraries) > 0,
            data={
                "combined_context": "\n\n".join(combined_context),
                "snippets": all_snippets,
                "successful_libraries": successful_libraries,
                "failed_libraries": failed_libraries,
                "query": query,
                "topic": topic,
            },
            latency_ms=(time.time() - start) * 1000,
            metadata={"parallel_queries": len(libraries)},
        )

    async def _get_code_examples(
        self,
        library_id: str,
        query: str,
        language: str = "python",
        max_examples: int = 5,
        **kwargs,
    ) -> AdapterResult:
        """
        Extract code examples with syntax highlighting info.

        Args:
            library_id: Context7 library ID
            query: Query to find relevant code examples
            language: Programming language filter (python, javascript, typescript, etc.)
            max_examples: Maximum number of examples to return

        Returns:
            AdapterResult with extracted code examples
        """
        start = time.time()

        # Query docs with code-focused topic
        result = await self._query_docs(
            library_id=library_id,
            query=f"{query} code example {language}",
            topic="examples",
            tokens=15000,  # Request more tokens for code
        )

        if not result.success:
            return result

        # Extract code blocks from context
        context = result.data.get("context", "")
        snippets = result.data.get("snippets", [])

        code_examples = []

        # Parse code blocks from markdown (```language ... ```)
        code_pattern = rf"```(?:{language}|{language[:2]}|)\n(.*?)```"
        matches = re.findall(code_pattern, context, re.DOTALL | re.IGNORECASE)

        for i, code in enumerate(matches[:max_examples]):
            code_examples.append({
                "language": language,
                "code": code.strip(),
                "source": "context",
                "index": i,
            })

        # Also extract from snippets
        for snippet in snippets[:max_examples - len(code_examples)]:
            content = snippet.get("content", "")
            if "```" in content or "def " in content or "function " in content:
                code_examples.append({
                    "language": language,
                    "code": content,
                    "source": "snippet",
                    "title": snippet.get("title", ""),
                })

        return AdapterResult(
            success=True,
            data={
                "code_examples": code_examples,
                "total_found": len(code_examples),
                "library_id": library_id,
                "query": query,
                "language": language,
            },
            latency_ms=(time.time() - start) * 1000,
        )

    async def _search_libraries(
        self,
        query: str,
        limit: int = 10,
        **kwargs,
    ) -> AdapterResult:
        """
        Search across all libraries using local whitelist.

        V40: MCP doesn't have a search API, so this uses the local
        LIBRARY_ID_MAPPINGS whitelist for fuzzy matching.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            AdapterResult with matching libraries
        """
        start = time.time()
        self._stats["queries"] += 1

        if not query:
            return AdapterResult(
                success=False,
                error="query is required",
                latency_ms=(time.time() - start) * 1000,
            )

        if self._mock_mode:
            return self._mock_search(query, limit, start)

        # Search in local whitelist (MCP doesn't have search API)
        query_lower = query.lower()
        results = []

        for name, library_id in LIBRARY_ID_MAPPINGS.items():
            # Simple fuzzy matching
            if query_lower in name or name in query_lower:
                results.append({
                    "library_id": library_id,
                    "name": name,
                    "description": f"Documentation for {name}",
                    "trust_score": 0.95 if name in PRIORITY_LIBRARIES else 0.8,
                })

        # Sort by relevance (exact match first, then priority libraries)
        results.sort(key=lambda x: (
            x["name"] != query_lower,  # Exact match first
            x["name"] not in PRIORITY_LIBRARIES,  # Priority libraries second
            x["name"],  # Alphabetical
        ))

        return AdapterResult(
            success=True,
            data={
                "results": results[:limit],
                "total": len(results),
                "query": query,
                "source": "whitelist",  # V40: Indicate source is local whitelist
            },
            latency_ms=(time.time() - start) * 1000,
            metadata={"whitelist_search": True},
        )

    # Mock methods for testing
    def _mock_resolve(self, library_name: str, start: float) -> AdapterResult:
        """Return mock data for resolve operation."""
        normalized_id = self._normalize_library_id(library_name)
        return AdapterResult(
            success=True,
            data={
                "library_id": normalized_id,
                "name": library_name,
                "description": f"Mock documentation for {library_name}",
                "trust_score": 0.95,
                "benchmark_score": 0.88,
                "total_snippets": 1500,
                "versions": ["latest", "v1.0", "v0.9"],
            },
            latency_ms=(time.time() - start) * 1000,
            metadata={"mock": True},
        )

    def _mock_query(self, library_id: str, query: str, start: float) -> AdapterResult:
        """Return mock data for query operation."""
        return AdapterResult(
            success=True,
            data={
                "context": f"Mock documentation context for {library_id} regarding: {query}",
                "snippets": [
                    {
                        "title": f"Mock snippet for {query}",
                        "content": f"This is mock content for {library_id}",
                        "source": "mock",
                    }
                ],
                "metadata": {"mock": True},
                "library_id": library_id,
                "query": query,
            },
            latency_ms=(time.time() - start) * 1000,
            metadata={"mock": True},
        )

    def _mock_search(self, query: str, limit: int, start: float) -> AdapterResult:
        """Return mock data for search operation."""
        results = [
            {
                "library_id": self._normalize_library_id(name),
                "name": name,
                "description": f"Documentation for {name}",
                "trust_score": 0.9,
            }
            for name in list(LIBRARY_ID_MAPPINGS.keys())[:limit]
        ]
        return AdapterResult(
            success=True,
            data={
                "results": results,
                "total": len(results),
                "query": query,
            },
            latency_ms=(time.time() - start) * 1000,
            metadata={"mock": True},
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return dict(self._stats)

    async def shutdown(self) -> AdapterResult:
        """Cleanup MCP resources."""
        # Cleanup stdio transport
        if self._mcp_process:
            try:
                self._mcp_process.terminate()
                self._mcp_process.wait(timeout=5.0)
            except Exception:
                try:
                    self._mcp_process.kill()
                except Exception:
                    pass
            self._mcp_process = None

        # Cleanup HTTP transport
        if self._http_client:
            try:
                await self._http_client.aclose()
            except Exception:
                pass
            self._http_client = None

        self._status = AdapterStatus.UNINITIALIZED
        return AdapterResult(success=True, data={"stats": self._stats})


# Convenience functions
async def query_sdk_docs(library: str, query: str, version: Optional[str] = None) -> AdapterResult:
    """
    Quick helper to query SDK documentation.

    Args:
        library: Library name (e.g., "langchain", "react")
        query: Documentation query
        version: Optional specific version

    Returns:
        AdapterResult with documentation
    """
    adapter = Context7Adapter()
    await adapter.initialize({})

    result = await adapter.execute(
        "get_docs",
        library_name=library,
        query=query,
        version=version,
    )

    await adapter.shutdown()
    return result


async def resolve_library(library_name: str) -> AdapterResult:
    """
    Quick helper to resolve a library name to Context7 ID.

    Args:
        library_name: Library name to resolve

    Returns:
        AdapterResult with library_id and metadata
    """
    adapter = Context7Adapter()
    await adapter.initialize({})

    result = await adapter.execute("resolve", library_name=library_name)

    await adapter.shutdown()
    return result
