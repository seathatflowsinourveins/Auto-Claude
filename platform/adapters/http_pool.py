"""
HTTP Connection Pool for Research Adapters
============================================

Provides shared HTTP connection pooling across all adapters to reduce
connection overhead and improve performance.

Features:
- Configurable pool size (default: 10 connections)
- Connection timeout management (default: 30s)
- Keep-alive support for connection reuse
- Pool metrics (active connections, wait time, requests)
- Singleton pool management per base URL
- Thread-safe and async-safe design
- Adaptive timeout integration for dynamic timeout adjustment

Performance Target: 50% reduction in connection overhead through
connection reuse and efficient pooling.

Usage:
    from adapters.http_pool import (
        HTTPConnectionPool,
        get_shared_pool,
        PoolMetrics,
    )

    # Get shared pool for an API
    pool = get_shared_pool("https://api.example.com")

    # Make requests through the pool
    response = await pool.get("/endpoint")
    response = await pool.post("/endpoint", json={"key": "value"})

    # Get pool metrics
    metrics = pool.get_metrics()
    print(f"Active connections: {metrics.active_connections}")
    print(f"Total requests: {metrics.total_requests}")
    print(f"Avg wait time: {metrics.avg_wait_time_ms}ms")

    # Make request with adaptive timeout
    response = await pool.get("/endpoint", use_adaptive_timeout=True)

    # Cleanup
    await pool.close()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Union
from weakref import WeakValueDictionary

import httpx

# Adaptive timeout integration
try:
    from ..core.adaptive_timeout import (
        AdaptiveTimeout,
        get_adaptive_timeout_sync,
        TimeoutProfile,
    )
    ADAPTIVE_TIMEOUT_AVAILABLE = True
except ImportError:
    ADAPTIVE_TIMEOUT_AVAILABLE = False
    AdaptiveTimeout = None
    TimeoutProfile = None

    def get_adaptive_timeout_sync(name, profile=None):
        return None

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """Configuration for HTTP connection pool."""

    max_connections: int = 10
    """Maximum number of connections in the pool."""

    max_keepalive_connections: int = 5
    """Maximum number of keep-alive connections to maintain."""

    keepalive_expiry: float = 30.0
    """Time in seconds before keep-alive connections expire."""

    connect_timeout: float = 30.0
    """Timeout in seconds for establishing a connection."""

    read_timeout: float = 60.0
    """Timeout in seconds for reading response data."""

    write_timeout: float = 30.0
    """Timeout in seconds for sending request data."""

    pool_timeout: float = 30.0
    """Timeout in seconds for acquiring a connection from the pool."""

    retries: int = 0
    """Number of connection retries (handled separately by retry module)."""

    http2: bool = False
    """Enable HTTP/2 support for multiplexing."""

    follow_redirects: bool = True
    """Follow HTTP redirects automatically."""

    verify_ssl: bool = True
    """Verify SSL certificates."""

    default_headers: Dict[str, str] = field(default_factory=dict)
    """Default headers to include in all requests."""

    enable_adaptive_timeout: bool = True
    """Enable adaptive timeout based on historical latency."""

    adaptive_timeout_operation: Optional[str] = None
    """Operation name for adaptive timeout tracking (derived from base_url if not set)."""


@dataclass
class PoolMetrics:
    """Metrics for connection pool monitoring."""

    total_requests: int = 0
    """Total number of requests made through the pool."""

    successful_requests: int = 0
    """Number of successful requests."""

    failed_requests: int = 0
    """Number of failed requests."""

    active_connections: int = 0
    """Current number of active connections."""

    total_wait_time_ms: float = 0.0
    """Cumulative wait time for acquiring connections."""

    avg_wait_time_ms: float = 0.0
    """Average wait time for acquiring connections."""

    max_wait_time_ms: float = 0.0
    """Maximum wait time for acquiring a connection."""

    total_request_time_ms: float = 0.0
    """Cumulative request execution time."""

    avg_request_time_ms: float = 0.0
    """Average request execution time."""

    connection_reuse_count: int = 0
    """Number of times connections were reused."""

    created_at: datetime = field(default_factory=datetime.utcnow)
    """When the pool was created."""

    last_request_at: Optional[datetime] = None
    """Timestamp of the last request."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "active_connections": self.active_connections,
            "avg_wait_time_ms": round(self.avg_wait_time_ms, 2),
            "max_wait_time_ms": round(self.max_wait_time_ms, 2),
            "avg_request_time_ms": round(self.avg_request_time_ms, 2),
            "connection_reuse_count": self.connection_reuse_count,
            "success_rate": (
                round(self.successful_requests / self.total_requests * 100, 2)
                if self.total_requests > 0 else 0.0
            ),
            "created_at": self.created_at.isoformat(),
            "last_request_at": (
                self.last_request_at.isoformat()
                if self.last_request_at else None
            ),
        }


class HTTPConnectionPool:
    """
    HTTP connection pool with keep-alive support and metrics tracking.

    Provides efficient connection reuse across multiple requests to the
    same base URL, reducing connection overhead and improving performance.

    Features:
    - Connection pooling with configurable limits
    - Keep-alive support for connection reuse
    - Detailed metrics tracking
    - Thread-safe and async-safe operations
    - Automatic cleanup on context exit

    Example:
        async with HTTPConnectionPool("https://api.example.com") as pool:
            response = await pool.get("/users")
            data = response.json()
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        config: Optional[PoolConfig] = None,
        **kwargs: Any,
    ):
        """
        Initialize HTTP connection pool.

        Args:
            base_url: Base URL for all requests (optional)
            config: Pool configuration (uses defaults if not provided)
            **kwargs: Override config parameters directly
        """
        self._config = config or PoolConfig()

        # Apply kwargs overrides
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

        self._base_url = base_url
        self._client: Optional[httpx.AsyncClient] = None
        self._metrics = PoolMetrics()
        self._lock = asyncio.Lock()
        self._closed = False
        self._request_count = 0  # For tracking reuse

        # Initialize adaptive timeout if available and enabled
        self._adaptive_timeout: Optional[AdaptiveTimeout] = None
        if ADAPTIVE_TIMEOUT_AVAILABLE and self._config.enable_adaptive_timeout:
            operation_name = self._config.adaptive_timeout_operation
            if not operation_name and base_url:
                # Derive operation name from base URL
                operation_name = self._derive_operation_name(base_url)
            if operation_name:
                self._adaptive_timeout = get_adaptive_timeout_sync(
                    f"http_{operation_name}"
                )

    def _derive_operation_name(self, url: str) -> str:
        """Derive an operation name from a URL for adaptive timeout tracking."""
        # Extract domain from URL
        if "://" in url:
            url = url.split("://", 1)[1]
        if "/" in url:
            url = url.split("/", 1)[0]
        if ":" in url:
            url = url.split(":", 1)[0]

        # Map common domains to operation names
        domain_map = {
            "api.exa.ai": "exa",
            "api.tavily.com": "tavily",
            "api.perplexity.ai": "perplexity",
            "api.jina.ai": "jina",
            "api.firecrawl.dev": "firecrawl",
            "api.serper.dev": "serper",
        }

        for domain, name in domain_map.items():
            if domain in url:
                return name

        # Use sanitized domain as fallback
        return url.replace(".", "_").replace("-", "_")

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure the client is initialized."""
        if self._client is None or self._closed:
            async with self._lock:
                if self._client is None or self._closed:
                    self._client = self._create_client()
                    self._closed = False
        return self._client

    def _create_client(self) -> httpx.AsyncClient:
        """Create a new httpx client with pooling configuration."""
        # Configure connection limits
        limits = httpx.Limits(
            max_connections=self._config.max_connections,
            max_keepalive_connections=self._config.max_keepalive_connections,
            keepalive_expiry=self._config.keepalive_expiry,
        )

        # Configure timeouts
        timeout = httpx.Timeout(
            connect=self._config.connect_timeout,
            read=self._config.read_timeout,
            write=self._config.write_timeout,
            pool=self._config.pool_timeout,
        )

        # Build transport with retry support
        transport = httpx.AsyncHTTPTransport(
            retries=self._config.retries,
            http2=self._config.http2,
            verify=self._config.verify_ssl,
        )

        # Create client
        client = httpx.AsyncClient(
            base_url=self._base_url or "",
            limits=limits,
            timeout=timeout,
            transport=transport,
            follow_redirects=self._config.follow_redirects,
            headers=self._config.default_headers,
        )

        logger.debug(
            f"Created HTTP connection pool: base_url={self._base_url}, "
            f"max_connections={self._config.max_connections}, "
            f"keepalive={self._config.max_keepalive_connections}"
        )

        return client

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Any] = None,
        data: Optional[Any] = None,
        content: Optional[bytes] = None,
        timeout: Optional[float] = None,
        use_adaptive_timeout: bool = False,
        **kwargs: Any,
    ) -> httpx.Response:
        """
        Make an HTTP request through the connection pool.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            url: Request URL (relative if base_url is set)
            headers: Request headers
            params: Query parameters
            json: JSON body (auto-serialized)
            data: Form data
            content: Raw bytes content
            timeout: Request timeout override (seconds)
            use_adaptive_timeout: Use adaptive timeout based on historical latency
            **kwargs: Additional httpx request arguments

        Returns:
            httpx.Response object

        Raises:
            httpx.HTTPError: On request failure
        """
        wait_start = time.time()
        client = await self._ensure_client()
        wait_time = (time.time() - wait_start) * 1000

        # Track metrics
        self._metrics.total_requests += 1
        self._metrics.total_wait_time_ms += wait_time
        self._metrics.max_wait_time_ms = max(
            self._metrics.max_wait_time_ms, wait_time
        )
        if self._metrics.total_requests > 0:
            self._metrics.avg_wait_time_ms = (
                self._metrics.total_wait_time_ms / self._metrics.total_requests
            )

        # Track connection reuse
        self._request_count += 1
        if self._request_count > 1:
            self._metrics.connection_reuse_count += 1

        # Determine timeout to use
        effective_timeout = timeout
        if effective_timeout is None and use_adaptive_timeout and self._adaptive_timeout:
            # Use adaptive timeout (convert from ms to seconds)
            effective_timeout = self._adaptive_timeout.get_timeout_seconds()

        # Build request kwargs
        request_kwargs: Dict[str, Any] = {}
        if headers:
            request_kwargs["headers"] = headers
        if params:
            request_kwargs["params"] = params
        if json is not None:
            request_kwargs["json"] = json
        if data is not None:
            request_kwargs["data"] = data
        if content is not None:
            request_kwargs["content"] = content
        if effective_timeout is not None:
            request_kwargs["timeout"] = effective_timeout
        request_kwargs.update(kwargs)

        request_start = time.time()
        self._metrics.active_connections += 1

        try:
            response = await client.request(method, url, **request_kwargs)
            self._metrics.successful_requests += 1
            self._metrics.last_request_at = datetime.utcnow()
            return response

        except Exception:
            self._metrics.failed_requests += 1
            raise

        finally:
            request_time = (time.time() - request_start) * 1000
            self._metrics.total_request_time_ms += request_time
            if self._metrics.total_requests > 0:
                self._metrics.avg_request_time_ms = (
                    self._metrics.total_request_time_ms / self._metrics.total_requests
                )
            self._metrics.active_connections = max(
                0, self._metrics.active_connections - 1
            )

            # Record latency for adaptive timeout
            if self._adaptive_timeout:
                self._adaptive_timeout.record_latency(request_time)

    async def get(
        self,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make a GET request."""
        return await self.request("GET", url, **kwargs)

    async def post(
        self,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make a POST request."""
        return await self.request("POST", url, **kwargs)

    async def put(
        self,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make a PUT request."""
        return await self.request("PUT", url, **kwargs)

    async def patch(
        self,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make a PATCH request."""
        return await self.request("PATCH", url, **kwargs)

    async def delete(
        self,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make a DELETE request."""
        return await self.request("DELETE", url, **kwargs)

    async def head(
        self,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make a HEAD request."""
        return await self.request("HEAD", url, **kwargs)

    async def options(
        self,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make an OPTIONS request."""
        return await self.request("OPTIONS", url, **kwargs)

    def get_metrics(self) -> PoolMetrics:
        """Get current pool metrics."""
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset pool metrics to initial state."""
        self._metrics = PoolMetrics()
        self._request_count = 0

    def get_adaptive_timeout_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get adaptive timeout statistics if enabled.

        Returns:
            Dictionary with latency percentiles and timeout info, or None if disabled
        """
        if self._adaptive_timeout:
            return self._adaptive_timeout.get_stats().to_dict()
        return None

    def get_current_adaptive_timeout(self) -> Optional[float]:
        """
        Get current adaptive timeout in seconds.

        Returns:
            Current timeout in seconds, or None if adaptive timeout is disabled
        """
        if self._adaptive_timeout:
            return self._adaptive_timeout.get_timeout_seconds()
        return None

    @property
    def adaptive_timeout(self) -> Optional[AdaptiveTimeout]:
        """Get the adaptive timeout handler if available."""
        return self._adaptive_timeout

    @property
    def base_url(self) -> Optional[str]:
        """Get the base URL."""
        return self._base_url

    @property
    def config(self) -> PoolConfig:
        """Get the pool configuration."""
        return self._config

    @property
    def is_closed(self) -> bool:
        """Check if the pool is closed."""
        return self._closed

    async def close(self) -> None:
        """Close the connection pool and release resources."""
        async with self._lock:
            if self._client and not self._closed:
                await self._client.aclose()
                self._client = None
                self._closed = True
                logger.debug(f"Closed HTTP connection pool: {self._base_url}")

    async def shutdown(self, drain_timeout: float = 10.0) -> None:
        """
        Gracefully shutdown the connection pool.

        Waits for active connections to complete (up to drain_timeout),
        then closes the pool. This ensures no data loss during shutdown.

        Args:
            drain_timeout: Maximum time to wait for active connections to complete
        """
        logger.info(f"Shutting down HTTP pool: {self._base_url} (active={self._metrics.active_connections})")

        # Wait for active connections to complete
        start = time.time()
        while self._metrics.active_connections > 0:
            if time.time() - start > drain_timeout:
                logger.warning(
                    f"Pool drain timeout: {self._metrics.active_connections} connections still active"
                )
                break
            await asyncio.sleep(0.1)

        # Close the pool
        await self.close()

        drain_time = time.time() - start
        logger.info(f"HTTP pool shutdown complete: {self._base_url} (drain_time={drain_time:.2f}s)")

    async def __aenter__(self) -> "HTTPConnectionPool":
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()


# Global pool registry for shared pools
_pool_registry: Dict[str, HTTPConnectionPool] = {}
_registry_lock = asyncio.Lock()


async def get_shared_pool(
    base_url: str,
    config: Optional[PoolConfig] = None,
    **kwargs: Any,
) -> HTTPConnectionPool:
    """
    Get or create a shared connection pool for a base URL.

    Pools are cached and reused for the same base URL, ensuring
    efficient connection sharing across adapters.

    Args:
        base_url: Base URL for the pool
        config: Pool configuration (only used if creating new pool)
        **kwargs: Override config parameters

    Returns:
        HTTPConnectionPool instance

    Example:
        pool = await get_shared_pool("https://api.exa.ai")
        response = await pool.get("/search")
    """
    # Normalize URL for cache key
    cache_key = base_url.rstrip("/")

    if cache_key in _pool_registry:
        pool = _pool_registry[cache_key]
        if not pool.is_closed:
            return pool

    async with _registry_lock:
        # Double-check after acquiring lock
        if cache_key in _pool_registry:
            pool = _pool_registry[cache_key]
            if not pool.is_closed:
                return pool

        # Create new pool
        pool = HTTPConnectionPool(base_url, config, **kwargs)
        _pool_registry[cache_key] = pool
        logger.debug(f"Created shared pool for: {base_url}")

    return pool


def get_shared_pool_sync(
    base_url: str,
    config: Optional[PoolConfig] = None,
    **kwargs: Any,
) -> HTTPConnectionPool:
    """
    Synchronous version of get_shared_pool.

    Creates a pool without initializing the client. The client
    will be created on first request.

    Args:
        base_url: Base URL for the pool
        config: Pool configuration
        **kwargs: Override config parameters

    Returns:
        HTTPConnectionPool instance (client not yet created)
    """
    cache_key = base_url.rstrip("/")

    if cache_key in _pool_registry:
        pool = _pool_registry[cache_key]
        if not pool.is_closed:
            return pool

    pool = HTTPConnectionPool(base_url, config, **kwargs)
    _pool_registry[cache_key] = pool

    return pool


async def close_all_pools() -> None:
    """Close all shared connection pools."""
    async with _registry_lock:
        for pool in _pool_registry.values():
            try:
                await pool.close()
            except Exception as e:
                logger.warning(f"Error closing pool: {e}")
        _pool_registry.clear()
        logger.debug("Closed all shared connection pools")


async def shutdown_all_pools(drain_timeout: float = 10.0) -> None:
    """
    Gracefully shutdown all shared connection pools.

    Drains active connections before closing each pool.
    This should be called during graceful shutdown to ensure
    no requests are dropped.

    Args:
        drain_timeout: Maximum time to wait for connections to drain per pool
    """
    logger.info(f"Shutting down {len(_pool_registry)} connection pools...")

    async with _registry_lock:
        # Shutdown pools in parallel
        shutdown_tasks = []
        for base_url, pool in _pool_registry.items():
            if not pool.is_closed:
                shutdown_tasks.append(pool.shutdown(drain_timeout))

        if shutdown_tasks:
            results = await asyncio.gather(*shutdown_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Pool shutdown error: {result}")

        _pool_registry.clear()
        logger.info("All connection pools shutdown complete")


def get_pool_stats() -> Dict[str, Any]:
    """
    Get statistics for all shared pools.

    Returns:
        Dictionary with pool statistics by base URL
    """
    stats = {}
    for base_url, pool in _pool_registry.items():
        metrics = pool.get_metrics()
        pool_stats = {
            "is_closed": pool.is_closed,
            "metrics": metrics.to_dict(),
            "config": {
                "max_connections": pool.config.max_connections,
                "max_keepalive_connections": pool.config.max_keepalive_connections,
                "connect_timeout": pool.config.connect_timeout,
            },
        }

        # Include adaptive timeout stats if available
        adaptive_stats = pool.get_adaptive_timeout_stats()
        if adaptive_stats:
            pool_stats["adaptive_timeout"] = adaptive_stats

        stats[base_url] = pool_stats
    return stats


# Default pool configurations for common services
DEFAULT_POOL_CONFIGS: Dict[str, PoolConfig] = {
    "exa": PoolConfig(
        max_connections=10,
        max_keepalive_connections=5,
        connect_timeout=30.0,
        read_timeout=60.0,
    ),
    "tavily": PoolConfig(
        max_connections=10,
        max_keepalive_connections=5,
        connect_timeout=30.0,
        read_timeout=120.0,  # Research can take longer
    ),
    "perplexity": PoolConfig(
        max_connections=10,
        max_keepalive_connections=5,
        connect_timeout=30.0,
        read_timeout=300.0,  # Deep research timeout
    ),
    "jina": PoolConfig(
        max_connections=10,
        max_keepalive_connections=5,
        connect_timeout=30.0,
        read_timeout=120.0,
    ),
    "firecrawl": PoolConfig(
        max_connections=10,
        max_keepalive_connections=5,
        connect_timeout=30.0,
        read_timeout=120.0,  # Crawling can take longer
    ),
    "default": PoolConfig(
        max_connections=10,
        max_keepalive_connections=5,
        connect_timeout=30.0,
        read_timeout=60.0,
    ),
}


def get_config_for_service(service: str) -> PoolConfig:
    """
    Get the default pool configuration for a service.

    Args:
        service: Service name (exa, tavily, perplexity, jina, etc.)

    Returns:
        PoolConfig for the service
    """
    return DEFAULT_POOL_CONFIGS.get(service.lower(), DEFAULT_POOL_CONFIGS["default"])


# Export public API
__all__ = [
    "HTTPConnectionPool",
    "PoolConfig",
    "PoolMetrics",
    "get_shared_pool",
    "get_shared_pool_sync",
    "close_all_pools",
    "shutdown_all_pools",
    "get_pool_stats",
    "get_config_for_service",
    "DEFAULT_POOL_CONFIGS",
    "ADAPTIVE_TIMEOUT_AVAILABLE",
]
