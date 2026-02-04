"""
Firecrawl AI Web Scraping Adapter - Turn Websites Into LLM-Ready Data
======================================================================

Firecrawl transforms any website into clean, structured data optimized
for LLM consumption. Built for AI agents that need web content.

Key Features:
- scrape: Single page extraction with JavaScript rendering
- crawl: Multi-page crawling with depth/breadth control
- batch_scrape: Parallel batch URL processing
- map: URL discovery without content extraction
- search: Web search with optional result scraping
- extract: AI-powered structured data extraction from URLs
- agent: AI-powered autonomous data gathering agent
- Actions: Interact with pages before scraping (click, type, scroll)

Official Docs: https://docs.firecrawl.dev/
GitHub: https://github.com/mendableai/firecrawl
MCP Server: firecrawl-mcp

Pricing:
- Free: 500 credits/month
- Hobby: 3,000 credits/month ($19)
- Standard: 100,000 credits/month ($99)
- Scale: 500,000 credits/month ($399)

Usage:
    adapter = FirecrawlAdapter()
    await adapter.initialize({"api_key": "fc-xxx"})

    # Single page scrape
    result = await adapter.execute("scrape", url="https://example.com")

    # Crawl entire site
    result = await adapter.execute("crawl", url="https://example.com", max_depth=3)

    # Batch scrape multiple URLs
    result = await adapter.execute("batch_scrape", urls=["url1", "url2", "url3"])

    # Web search
    result = await adapter.execute("search", query="AI agents", num_results=10)

    # Extract structured data
    result = await adapter.execute("extract", urls=["url1"], prompt="Extract product info")

    # AI agent for autonomous gathering
    result = await adapter.execute("agent", url="https://example.com", objective="Find pricing")
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

# Retry utilities
try:
    from .retry import RetryConfig, with_retry, retry_async, retry_sync
except ImportError:
    # Fallback for standalone testing
    RetryConfig = None
    with_retry = lambda f=None, **kw: (lambda fn: fn) if f is None else f
    retry_async = None
    retry_sync = None

# Circuit breaker imports
try:
    from .circuit_breaker_manager import adapter_circuit_breaker, get_adapter_circuit_manager
    from core.resilience import CircuitOpenError
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False
    CircuitOpenError = Exception
    def adapter_circuit_breaker(name):
        class DummyBreaker:
            async def __aenter__(self): return self
            async def __aexit__(self, *args): return False
        return DummyBreaker()
    def get_adapter_circuit_manager():
        return None

# SDK imports
try:
    from core.orchestration.base import SDKAdapter, AdapterResult, AdapterStatus, SDKLayer, register_adapter
except ImportError:
    try:
        from core.orchestration.base import SDKAdapter, AdapterResult, AdapterStatus, SDKLayer, register_adapter
    except ImportError:
        from enum import IntEnum
        from abc import ABC, abstractmethod

        class SDKLayer(IntEnum):
            RESEARCH = 8

        class AdapterStatus(Enum):
            UNINITIALIZED = "uninitialized"
            READY = "ready"
            DEGRADED = "degraded"
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


# Firecrawl SDK
FIRECRAWL_AVAILABLE = False
try:
    from firecrawl import FirecrawlApp, AsyncFirecrawlApp
    FIRECRAWL_AVAILABLE = True
except ImportError:
    FirecrawlApp = None
    AsyncFirecrawlApp = None


class FirecrawlFormat(str, Enum):
    """Output format options."""
    MARKDOWN = "markdown"
    HTML = "html"
    RAW_HTML = "rawHtml"
    LINKS = "links"
    SCREENSHOT = "screenshot"
    SCREENSHOT_FULL = "screenshot@fullPage"


class FirecrawlActionType(str, Enum):
    """Supported page interaction actions."""
    WAIT = "wait"           # Wait for duration (ms)
    CLICK = "click"         # Click element by selector
    TYPE = "write"          # Type text into element
    PRESS = "press"         # Press keyboard key
    SCROLL = "scroll"       # Scroll page (x, y)
    SCREENSHOT = "screenshot"  # Take screenshot at point
    SCRAPE = "scrape"       # Scrape at current state


@dataclass
class FirecrawlAction:
    """Page interaction action."""
    type: FirecrawlActionType
    selector: Optional[str] = None      # CSS selector for click/type
    text: Optional[str] = None          # Text for type action
    milliseconds: Optional[int] = None  # Wait duration
    key: Optional[str] = None           # Keyboard key for press
    direction: Optional[str] = None     # Scroll direction (up/down)
    amount: Optional[int] = None        # Scroll amount (pixels)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API format."""
        result = {"type": self.type.value}
        if self.selector:
            result["selector"] = self.selector
        if self.text:
            result["text"] = self.text
        if self.milliseconds:
            result["milliseconds"] = self.milliseconds
        if self.key:
            result["key"] = self.key
        if self.direction:
            result["direction"] = self.direction
        if self.amount:
            result["amount"] = self.amount
        return result


@register_adapter("firecrawl", SDKLayer.RESEARCH, priority=23)
class FirecrawlAdapter(SDKAdapter):
    """
    Firecrawl AI web scraping adapter - SDK v2.0+ compliant.

    Operations (17 total):

    Core Scraping:
        - scrape: Single page extraction with JS rendering, actions support
        - crawl: Multi-page crawling with depth/breadth control
        - batch_scrape: Parallel batch URL scraping (up to 1000 URLs)
        - map: URL discovery without content extraction

    Crawl Job Management:
        - check_crawl: Check async crawl status
        - cancel_crawl: Cancel ongoing crawl
        - get_crawl_status_page: Paginate large crawl results

    Batch Scrape Management:
        - get_batch_scrape_status: Check batch scrape job status
        - get_batch_scrape_status_page: Paginate batch results

    Search:
        - search: Web search with optional result scraping

    Extract (LLM-powered):
        - extract: AI-powered structured data extraction (sync)
        - start_extract: Start async extract job
        - get_extract_status: Check extract job status

    Agent (AI Autonomous):
        - agent: AI-powered autonomous data gathering (sync)
        - start_agent: Start async agent job
        - check_agent_status: Check agent job status

    SDK Version: firecrawl-py >= 2.0.0
    """

    def __init__(self):
        self._app: Optional[FirecrawlApp] = None
        self._async_app: Optional[AsyncFirecrawlApp] = None
        self._status = AdapterStatus.UNINITIALIZED
        self._config: Dict[str, Any] = {}
        self._stats = {
            "scrapes": 0,
            "crawls": 0,
            "batch_scrapes": 0,
            "maps": 0,
            "searches": 0,
            "extracts": 0,
            "agents": 0,
            "pages_processed": 0,
            "avg_latency_ms": 0.0,
            "retries": 0,
        }
        self._active_crawls: Dict[str, Dict[str, Any]] = {}
        self._active_agents: Dict[str, Dict[str, Any]] = {}
        # Retry configuration for transient errors
        self._retry_config = RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=60.0,
            jitter=0.5,
        ) if RetryConfig else None

    @property
    def sdk_name(self) -> str:
        return "firecrawl"

    @property
    def layer(self) -> SDKLayer:
        return SDKLayer.RESEARCH

    @property
    def available(self) -> bool:
        return FIRECRAWL_AVAILABLE

    async def initialize(self, config: Dict[str, Any]) -> AdapterResult:
        """Initialize Firecrawl client."""
        start = time.time()

        if not FIRECRAWL_AVAILABLE:
            self._status = AdapterStatus.ERROR
            return AdapterResult(
                success=False,
                error="Firecrawl SDK not installed. Run: pip install firecrawl-py",
                latency_ms=(time.time() - start) * 1000,
            )

        try:
            api_key = config.get("api_key") or os.environ.get("FIRECRAWL_API_KEY")

            if not api_key:
                self._status = AdapterStatus.DEGRADED
                return AdapterResult(
                    success=True,
                    data={"status": "degraded", "reason": "No API key - mock mode"},
                    latency_ms=(time.time() - start) * 1000,
                )

            # Initialize both sync and async clients
            self._app = FirecrawlApp(api_key=api_key)

            # Async client if available
            try:
                self._async_app = AsyncFirecrawlApp(api_key=api_key)
            except Exception:
                self._async_app = None  # Fall back to sync

            self._config = config
            self._status = AdapterStatus.READY

            return AdapterResult(
                success=True,
                data={
                    "status": "ready",
                    "sdk_version": "2.0+",
                    "features": [
                        # Core scraping
                        "scrape", "crawl", "batch_scrape", "map",
                        # Crawl management
                        "check_crawl", "cancel_crawl", "get_crawl_status_page",
                        # Batch management
                        "get_batch_scrape_status", "get_batch_scrape_status_page",
                        # Search
                        "search",
                        # Extract (LLM)
                        "extract", "start_extract", "get_extract_status",
                        # Agent (AI)
                        "agent", "start_agent", "check_agent_status",
                        # Actions
                        "actions",
                    ],
                    "formats": [f.value for f in FirecrawlFormat],
                    "actions": [a.value for a in FirecrawlActionType],
                    "async_available": self._async_app is not None,
                },
                latency_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            self._status = AdapterStatus.ERROR
            return AdapterResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start) * 1000,
            )

    async def execute(self, operation: str, **kwargs) -> AdapterResult:
        """Execute Firecrawl operation with circuit breaker protection."""
        start = time.time()

        operations = {
            # Core scraping operations
            "scrape": self._scrape,
            "crawl": self._crawl,
            "batch_scrape": self._batch_scrape,
            "map": self._map,
            # Crawl job management
            "check_crawl": self._check_crawl,
            "cancel_crawl": self._cancel_crawl,
            "get_crawl_status_page": self._get_crawl_status_page,
            # Batch scrape job management
            "get_batch_scrape_status": self._get_batch_scrape_status,
            "get_batch_scrape_status_page": self._get_batch_scrape_status_page,
            # Search operation
            "search": self._search,
            # Extract operations (LLM-powered)
            "extract": self._extract,
            "start_extract": self._start_extract,
            "get_extract_status": self._get_extract_status,
            # Agent operations (AI autonomous)
            "agent": self._agent,
            "start_agent": self._start_agent,
            "check_agent_status": self._check_agent_status,
        }

        if operation not in operations:
            return AdapterResult(
                success=False,
                error=f"Unknown operation: {operation}. Valid: {list(operations.keys())}",
                latency_ms=(time.time() - start) * 1000,
            )

        # Execute with circuit breaker protection
        try:
            async with adapter_circuit_breaker("firecrawl_adapter"):
                result = await operations[operation](**kwargs)
                result.latency_ms = (time.time() - start) * 1000
                self._update_avg_latency(result.latency_ms)
                return result
        except CircuitOpenError as e:
            # Circuit is open - return fallback response
            return AdapterResult(
                success=False,
                error=f"Circuit breaker open for firecrawl_adapter: {e}",
                latency_ms=(time.time() - start) * 1000,
                metadata={"circuit_breaker": "open", "adapter": "firecrawl"},
            )
        except Exception as e:
            return AdapterResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start) * 1000,
            )

    async def _scrape(
        self,
        url: str,
        formats: Optional[List[str]] = None,
        only_main_content: bool = True,
        include_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        headers: Optional[Dict[str, str]] = None,
        wait_for: Optional[int] = None,
        timeout: int = 30000,
        actions: Optional[List[Dict[str, Any]]] = None,
        location: Optional[Dict[str, Any]] = None,
        mobile: bool = False,
        skip_tls_verification: bool = False,
        remove_base64_images: bool = False,
        # New parameters
        max_age: Optional[int] = None,
        store_in_cache: bool = True,
        proxy: Optional[str] = None,
        enhanced: bool = False,
        **kwargs,
    ) -> AdapterResult:
        """
        Scrape a single URL with optional actions.

        Args:
            url: Target URL to scrape
            formats: Output formats (markdown, html, rawHtml, links, screenshot)
            only_main_content: Extract only main content (no headers/footers)
            include_tags: Only include these HTML tags
            exclude_tags: Exclude these HTML tags
            headers: Custom HTTP headers
            wait_for: Wait for element selector (ms)
            timeout: Request timeout in milliseconds
            actions: Page interactions before scraping
            location: Geolocation {country: str, languages: [str]}
            mobile: Use mobile viewport
            skip_tls_verification: Skip TLS verification
            remove_base64_images: Remove base64 images from output
            max_age: Cache freshness in hours (None = always fresh)
            store_in_cache: Whether to cache the result (default True)
            proxy: Proxy server URL to use
            enhanced: Enhanced mode for complex/dynamic sites
        """
        self._stats["scrapes"] += 1

        if not self._app:
            return AdapterResult(
                success=True,
                data={
                    "url": url,
                    "markdown": f"Mock scraped content from {url}",
                    "metadata": {"title": "Mock Title"},
                    "mock": True,
                }
            )

        # Build scrape parameters
        params = {
            "url": url,
            "formats": formats or ["markdown"],
            "onlyMainContent": only_main_content,
            "timeout": timeout,
        }

        if include_tags:
            params["includeTags"] = include_tags
        if exclude_tags:
            params["excludeTags"] = exclude_tags
        if headers:
            params["headers"] = headers
        if wait_for:
            params["waitFor"] = wait_for
        if actions:
            params["actions"] = actions
        if location:
            params["location"] = location
        if mobile:
            params["mobile"] = True
        if skip_tls_verification:
            params["skipTlsVerification"] = True
        if remove_base64_images:
            params["removeBase64Images"] = True
        # New parameters
        if max_age is not None:
            params["maxAge"] = max_age
        if not store_in_cache:
            params["storeInCache"] = False
        if proxy:
            params["proxy"] = proxy
        if enhanced:
            params["enhanced"] = True

        # Execute in thread pool with retry logic
        async def _do_scrape():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return await loop.run_in_executor(
                None,
                lambda: self._app.scrape(**params)
            )

        if retry_async and self._retry_config:
            def _on_retry(attempt, exc, delay):
                self._stats["retries"] += 1
            config = RetryConfig(
                max_retries=self._retry_config.max_retries,
                base_delay=self._retry_config.base_delay,
                max_delay=self._retry_config.max_delay,
                jitter=self._retry_config.jitter,
                on_retry=_on_retry,
            )
            response = await retry_async(_do_scrape, config=config)
        else:
            response = await _do_scrape()

        self._stats["pages_processed"] += 1

        return AdapterResult(
            success=True,
            data={
                "url": url,
                "markdown": response.get("markdown", ""),
                "html": response.get("html", ""),
                "raw_html": response.get("rawHtml", ""),
                "links": response.get("links", []),
                "screenshot": response.get("screenshot"),
                "metadata": response.get("metadata", {}),
                "actions_executed": len(actions) if actions else 0,
            }
        )

    async def _crawl(
        self,
        url: str,
        max_depth: int = 2,
        limit: int = 100,
        include_paths: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        ignore_sitemap: bool = False,
        allow_external_links: bool = False,
        allow_backward_links: bool = False,
        poll_interval: int = 5,
        async_mode: bool = True,
        webhook: Optional[str] = None,
        **kwargs,
    ) -> AdapterResult:
        """
        Crawl website starting from URL.

        Args:
            url: Starting URL for crawl
            max_depth: Maximum depth from start URL (0-10)
            limit: Maximum pages to crawl
            include_paths: Only crawl paths matching these patterns
            exclude_paths: Skip paths matching these patterns
            ignore_sitemap: Don't use sitemap for URL discovery
            allow_external_links: Follow links to other domains
            allow_backward_links: Allow links going up in hierarchy
            poll_interval: Seconds between status checks (async)
            async_mode: Return immediately with crawl ID
            webhook: Webhook URL for completion notification
        """
        self._stats["crawls"] += 1

        if not self._app:
            return AdapterResult(
                success=True,
                data={
                    "crawl_id": "mock-crawl-123",
                    "status": "completed",
                    "pages": [{"url": url, "content": "Mock content"}],
                    "mock": True,
                }
            )

        # Build crawl parameters
        params = {
            "url": url,
            "maxDepth": min(max_depth, 10),
            "limit": limit,
            "ignoreSitemap": ignore_sitemap,
            "allowExternalLinks": allow_external_links,
            "allowBackwardLinks": allow_backward_links,
        }

        if include_paths:
            params["includePaths"] = include_paths
        if exclude_paths:
            params["excludePaths"] = exclude_paths
        if webhook:
            params["webhook"] = webhook

        # Execute crawl with retry logic
        def _on_retry(attempt, exc, delay):
            self._stats["retries"] += 1

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if async_mode:
            # Start async crawl and return ID
            async def _do_async_crawl():
                return await loop.run_in_executor(
                    None,
                    lambda: self._app.async_crawl(**params)
                )

            if retry_async and self._retry_config:
                config = RetryConfig(
                    max_retries=self._retry_config.max_retries,
                    base_delay=self._retry_config.base_delay,
                    max_delay=self._retry_config.max_delay,
                    jitter=self._retry_config.jitter,
                    on_retry=_on_retry,
                )
                response = await retry_async(_do_async_crawl, config=config)
            else:
                response = await _do_async_crawl()

            crawl_id = response.get("id", response.get("jobId"))

            # Track active crawl
            self._active_crawls[crawl_id] = {
                "url": url,
                "started_at": datetime.utcnow(),
                "params": params,
            }

            return AdapterResult(
                success=True,
                data={
                    "crawl_id": crawl_id,
                    "status": "crawling",
                    "message": f"Crawl started. Use check_crawl('{crawl_id}') to monitor progress.",
                }
            )
        else:
            # Synchronous crawl - wait for completion
            async def _do_crawl():
                return await loop.run_in_executor(
                    None,
                    lambda: self._app.crawl(**params)
                )

            if retry_async and self._retry_config:
                config = RetryConfig(
                    max_retries=self._retry_config.max_retries,
                    base_delay=self._retry_config.base_delay,
                    max_delay=self._retry_config.max_delay,
                    jitter=self._retry_config.jitter,
                    on_retry=_on_retry,
                )
                response = await retry_async(_do_crawl, config=config)
            else:
                response = await _do_crawl()

            pages = response.get("data", [])
            self._stats["pages_processed"] += len(pages)

            return AdapterResult(
                success=True,
                data={
                    "status": "completed",
                    "total_pages": len(pages),
                    "pages": pages,
                    "url": url,
                }
            )

    async def _batch_scrape(
        self,
        urls: List[str],
        formats: Optional[List[str]] = None,
        only_main_content: bool = True,
        **kwargs,
    ) -> AdapterResult:
        """
        Batch scrape multiple URLs in parallel.

        Args:
            urls: List of URLs to scrape (up to 1000)
            formats: Output formats for all pages
            only_main_content: Extract only main content
        """
        self._stats["batch_scrapes"] += 1

        if not self._app:
            return AdapterResult(
                success=True,
                data={
                    "batch_id": "mock-batch-123",
                    "status": "completed",
                    "results": [{"url": u, "content": f"Mock: {u}"} for u in urls],
                    "mock": True,
                }
            )

        # Firecrawl batch API
        params = {
            "urls": urls[:1000],  # API limit
            "formats": formats or ["markdown"],
            "onlyMainContent": only_main_content,
        }

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            # Try async batch if available
            response = await loop.run_in_executor(
                None,
                lambda: self._app.batch_scrape(**params)
            )

            batch_id = response.get("id", response.get("jobId"))

            return AdapterResult(
                success=True,
                data={
                    "batch_id": batch_id,
                    "status": "processing",
                    "total_urls": len(urls),
                    "message": f"Batch started. Check status with check_crawl('{batch_id}')",
                }
            )
        except AttributeError:
            # Fall back to sequential scraping
            results = []
            for url in urls[:100]:  # Limit for sequential
                try:
                    r = await loop.run_in_executor(
                        None,
                        lambda u=url: self._app.scrape(url=u, formats=params["formats"])
                    )
                    results.append({"url": url, "success": True, "data": r})
                except Exception as e:
                    results.append({"url": url, "success": False, "error": str(e)})

            self._stats["pages_processed"] += len(results)

            return AdapterResult(
                success=True,
                data={
                    "status": "completed",
                    "total": len(results),
                    "successful": sum(1 for r in results if r["success"]),
                    "results": results,
                }
            )

    async def _map(
        self,
        url: str,
        search: Optional[str] = None,
        ignore_sitemap: bool = False,
        include_subdomains: bool = False,
        limit: int = 5000,
        **kwargs,
    ) -> AdapterResult:
        """
        Discover URLs on a website without extracting content.

        Args:
            url: Starting URL for discovery
            search: Filter URLs containing this text
            ignore_sitemap: Don't use sitemap
            include_subdomains: Include subdomain URLs
            limit: Maximum URLs to return
        """
        self._stats["maps"] += 1

        if not self._app:
            return AdapterResult(
                success=True,
                data={
                    "base_url": url,
                    "urls": [url, f"{url}/page1", f"{url}/page2"],
                    "total": 3,
                    "mock": True,
                }
            )

        params = {
            "url": url,
            "ignoreSitemap": ignore_sitemap,
            "includeSubdomains": include_subdomains,
            "limit": limit,
        }

        if search:
            params["search"] = search

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        response = await loop.run_in_executor(
            None,
            lambda: self._app.map(**params)
        )

        urls = response.get("links", response.get("urls", []))

        return AdapterResult(
            success=True,
            data={
                "base_url": url,
                "urls": urls,
                "total": len(urls),
                "search_filter": search,
            }
        )

    async def _check_crawl(
        self,
        crawl_id: str,
        **kwargs,
    ) -> AdapterResult:
        """Check status of async crawl or batch job."""
        if not self._app:
            return AdapterResult(
                success=True,
                data={
                    "crawl_id": crawl_id,
                    "status": "completed",
                    "progress": 100,
                    "mock": True,
                }
            )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        response = await loop.run_in_executor(
            None,
            lambda: self._app.check_crawl_status(crawl_id)
        )

        status = response.get("status", "unknown")

        if status == "completed":
            # Get results
            data = response.get("data", [])
            self._stats["pages_processed"] += len(data)

            # Remove from active crawls
            self._active_crawls.pop(crawl_id, None)

            return AdapterResult(
                success=True,
                data={
                    "crawl_id": crawl_id,
                    "status": "completed",
                    "total_pages": len(data),
                    "pages": data,
                }
            )
        else:
            return AdapterResult(
                success=True,
                data={
                    "crawl_id": crawl_id,
                    "status": status,
                    "completed": response.get("completed", 0),
                    "total": response.get("total", 0),
                    "credits_used": response.get("creditsUsed", 0),
                }
            )

    async def _cancel_crawl(
        self,
        crawl_id: str,
        **kwargs,
    ) -> AdapterResult:
        """Cancel an ongoing crawl."""
        if not self._app:
            return AdapterResult(
                success=True,
                data={"crawl_id": crawl_id, "cancelled": True, "mock": True}
            )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            await loop.run_in_executor(
                None,
                lambda: self._app.cancel_crawl(crawl_id)
            )

            self._active_crawls.pop(crawl_id, None)

            return AdapterResult(
                success=True,
                data={"crawl_id": crawl_id, "cancelled": True}
            )
        except Exception as e:
            return AdapterResult(
                success=False,
                error=f"Failed to cancel crawl: {e}"
            )

    # =========================================================================
    # NEW OPERATIONS: search, extract, agent, batch_scrape_status
    # =========================================================================

    async def _search(
        self,
        query: str,
        num_results: int = 10,
        scrape_options: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AdapterResult:
        """
        Perform web search with optional scraping of results.

        Args:
            query: Search query
            num_results: Number of results to return
            scrape_options: Options for scraping search results
        """
        self._stats["searches"] += 1

        if not self._app:
            return self._mock_search(query)

        try:
            params: Dict[str, Any] = {
                "query": query,
                "limit": num_results,
            }
            if scrape_options:
                params["scrapeOptions"] = scrape_options

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            response = await loop.run_in_executor(
                None,
                lambda: self._app.search(**params)
            )

            results = response.get("data", [])

            return AdapterResult(
                success=True,
                data={
                    "results": results,
                    "query": query,
                },
                metadata={"num_results": len(results)},
            )
        except Exception as e:
            return AdapterResult(
                success=False,
                data=None,
                error=str(e),
            )

    def _mock_search(self, query: str) -> AdapterResult:
        """Mock search response for testing without API key."""
        return AdapterResult(
            success=True,
            data={
                "results": [
                    {
                        "title": f"Mock Result 1 for: {query}",
                        "url": "https://example.com/result1",
                        "description": "Mock search result description",
                    },
                    {
                        "title": f"Mock Result 2 for: {query}",
                        "url": "https://example.com/result2",
                        "description": "Another mock search result",
                    },
                ],
                "query": query,
                "mock": True,
            },
            metadata={"num_results": 2},
        )

    async def _extract(
        self,
        urls: List[str],
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AdapterResult:
        """
        Extract structured data from URLs using AI.

        Args:
            urls: List of URLs to extract from
            prompt: Extraction prompt describing what to extract
            schema: Optional JSON schema for structured output
        """
        self._stats["extracts"] += 1

        if not self._app:
            return self._mock_extract(urls, prompt)

        try:
            params: Dict[str, Any] = {
                "urls": urls,
                "prompt": prompt,
            }
            if schema:
                params["schema"] = schema

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            response = await loop.run_in_executor(
                None,
                lambda: self._app.extract(**params)
            )

            return AdapterResult(
                success=True,
                data={
                    "extracted": response.get("data", []),
                    "urls": urls,
                },
            )
        except Exception as e:
            return AdapterResult(
                success=False,
                data=None,
                error=str(e),
            )

    def _mock_extract(self, urls: List[str], prompt: str) -> AdapterResult:
        """Mock extract response for testing without API key."""
        return AdapterResult(
            success=True,
            data={
                "extracted": [
                    {"url": url, "data": {"mock_field": f"Extracted from {url}"}}
                    for url in urls
                ],
                "urls": urls,
                "prompt": prompt,
                "mock": True,
            },
        )

    async def _agent(
        self,
        url: str,
        objective: str,
        max_steps: int = 10,
        **kwargs,
    ) -> AdapterResult:
        """
        AI-powered autonomous data gathering agent.

        Args:
            url: Starting URL
            objective: What the agent should accomplish
            max_steps: Maximum steps for the agent
        """
        self._stats["agents"] += 1

        if not self._app:
            return self._mock_agent(url, objective)

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            response = await loop.run_in_executor(
                None,
                lambda: self._app.agent(
                    url=url,
                    objective=objective,
                    maxSteps=max_steps,
                )
            )

            return AdapterResult(
                success=True,
                data={
                    "agent_id": response.get("id"),
                    "status": response.get("status"),
                    "result": response.get("data"),
                },
            )
        except Exception as e:
            return AdapterResult(
                success=False,
                data=None,
                error=str(e),
            )

    def _mock_agent(self, url: str, objective: str) -> AdapterResult:
        """Mock agent response for testing without API key."""
        return AdapterResult(
            success=True,
            data={
                "agent_id": "mock-agent-123",
                "status": "completed",
                "result": {
                    "url": url,
                    "objective": objective,
                    "findings": "Mock agent findings based on objective",
                },
                "mock": True,
            },
        )

    async def _start_agent(
        self,
        url: str,
        objective: str,
        max_steps: int = 10,
        **kwargs,
    ) -> AdapterResult:
        """
        Start async agent job.

        Args:
            url: Starting URL
            objective: What the agent should accomplish
            max_steps: Maximum steps for the agent
        """
        self._stats["agents"] += 1

        if not self._app:
            agent_id = "mock-agent-async-123"
            self._active_agents[agent_id] = {
                "url": url,
                "objective": objective,
                "started_at": time.time(),
            }
            return AdapterResult(
                success=True,
                data={"agent_id": agent_id, "status": "started", "mock": True},
            )

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            response = await loop.run_in_executor(
                None,
                lambda: self._app.start_agent(
                    url=url,
                    objective=objective,
                    maxSteps=max_steps,
                )
            )

            agent_id = response.get("id")
            self._active_agents[agent_id] = {
                "url": url,
                "objective": objective,
                "started_at": time.time(),
            }

            return AdapterResult(
                success=True,
                data={"agent_id": agent_id, "status": "started"},
            )
        except Exception as e:
            return AdapterResult(
                success=False,
                data=None,
                error=str(e),
            )

    async def _check_agent_status(
        self,
        agent_id: str,
        **kwargs,
    ) -> AdapterResult:
        """
        Check status of async agent job.

        Args:
            agent_id: The agent job ID to check
        """
        if not self._app:
            return AdapterResult(
                success=True,
                data={
                    "agent_id": agent_id,
                    "status": "completed",
                    "result": {"mock": True, "findings": "Mock agent completed"},
                    "steps_completed": 5,
                    "mock": True,
                },
            )

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            response = await loop.run_in_executor(
                None,
                lambda: self._app.check_agent_status(agent_id)
            )

            status = response.get("status")

            # Clean up if completed
            if status == "completed":
                self._active_agents.pop(agent_id, None)

            return AdapterResult(
                success=True,
                data={
                    "agent_id": agent_id,
                    "status": status,
                    "result": response.get("data"),
                    "steps_completed": response.get("stepsCompleted"),
                },
            )
        except Exception as e:
            return AdapterResult(
                success=False,
                data=None,
                error=str(e),
            )

    async def _get_batch_scrape_status(
        self,
        batch_id: str,
        **kwargs,
    ) -> AdapterResult:
        """
        Get status of a batch scrape job.

        Args:
            batch_id: The batch scrape job ID to check
        """
        if not self._app:
            return AdapterResult(
                success=True,
                data={
                    "batch_id": batch_id,
                    "status": "completed",
                    "completed": 10,
                    "total": 10,
                    "data": [{"url": f"https://example.com/{i}", "content": f"Mock {i}"} for i in range(10)],
                    "mock": True,
                },
            )

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            response = await loop.run_in_executor(
                None,
                lambda: self._app.get_batch_scrape_status(batch_id)
            )

            return AdapterResult(
                success=True,
                data={
                    "batch_id": batch_id,
                    "status": response.get("status"),
                    "completed": response.get("completed"),
                    "total": response.get("total"),
                    "data": response.get("data", []),
                    "next": response.get("next"),  # Pagination URL
                },
            )
        except Exception as e:
            return AdapterResult(
                success=False,
                data=None,
                error=str(e),
            )

    async def _get_batch_scrape_status_page(
        self,
        next_url: str,
        **kwargs,
    ) -> AdapterResult:
        """
        Get next page of batch scrape results using pagination URL.

        Args:
            next_url: The opaque next URL from previous status response
        """
        if not self._app:
            return AdapterResult(
                success=True,
                data={"data": [], "next": None, "mock": True},
            )

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            response = await loop.run_in_executor(
                None,
                lambda: self._app.get_batch_scrape_status_page(next_url)
            )

            return AdapterResult(
                success=True,
                data={
                    "data": response.get("data", []),
                    "next": response.get("next"),
                },
            )
        except Exception as e:
            return AdapterResult(
                success=False,
                data=None,
                error=str(e),
            )

    async def _get_crawl_status_page(
        self,
        next_url: str,
        **kwargs,
    ) -> AdapterResult:
        """
        Get next page of crawl results using pagination URL.

        Args:
            next_url: The opaque next URL from previous status response
        """
        if not self._app:
            return AdapterResult(
                success=True,
                data={"data": [], "next": None, "mock": True},
            )

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            response = await loop.run_in_executor(
                None,
                lambda: self._app.get_crawl_status_page(next_url)
            )

            return AdapterResult(
                success=True,
                data={
                    "data": response.get("data", []),
                    "next": response.get("next"),
                },
            )
        except Exception as e:
            return AdapterResult(
                success=False,
                data=None,
                error=str(e),
            )

    async def _start_extract(
        self,
        urls: List[str],
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        enable_web_search: bool = False,
        **kwargs,
    ) -> AdapterResult:
        """
        Start async extract job (non-blocking).

        Args:
            urls: List of URLs to extract from (supports wildcards like example.com/*)
            prompt: Natural language description of data to extract
            schema: Optional JSON schema for structured output
            enable_web_search: Allow crawling beyond specified domains
        """
        self._stats["extracts"] += 1

        if not self._app:
            return AdapterResult(
                success=True,
                data={"job_id": "mock-extract-123", "status": "processing", "mock": True},
            )

        try:
            params: Dict[str, Any] = {
                "urls": urls,
                "prompt": prompt,
            }
            if schema:
                params["schema"] = schema
            if enable_web_search:
                params["enableWebSearch"] = True

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            response = await loop.run_in_executor(
                None,
                lambda: self._app.start_extract(**params)
            )

            return AdapterResult(
                success=True,
                data={
                    "job_id": response.get("id"),
                    "status": "processing",
                },
            )
        except Exception as e:
            return AdapterResult(
                success=False,
                data=None,
                error=str(e),
            )

    async def _get_extract_status(
        self,
        job_id: str,
        **kwargs,
    ) -> AdapterResult:
        """
        Check status of async extract job.

        Args:
            job_id: The extract job ID to check
        """
        if not self._app:
            return AdapterResult(
                success=True,
                data={
                    "job_id": job_id,
                    "status": "completed",
                    "data": [{"mock_field": "extracted data"}],
                    "mock": True,
                },
            )

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            response = await loop.run_in_executor(
                None,
                lambda: self._app.get_extract_status(job_id)
            )

            return AdapterResult(
                success=True,
                data={
                    "job_id": job_id,
                    "status": response.get("status"),
                    "data": response.get("data", []),
                },
            )
        except Exception as e:
            return AdapterResult(
                success=False,
                data=None,
                error=str(e),
            )

    # =========================================================================
    # END SDK V2 OPERATIONS
    # =========================================================================

    def _update_avg_latency(self, latency: float):
        """Update rolling average latency."""
        total_ops = sum([
            self._stats["scrapes"],
            self._stats["crawls"],
            self._stats["batch_scrapes"],
            self._stats["maps"],
            self._stats["searches"],
            self._stats["extracts"],
            self._stats["agents"],
        ])
        if total_ops > 0:
            self._stats["avg_latency_ms"] = (
                (self._stats["avg_latency_ms"] * (total_ops - 1) + latency)
                / total_ops
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return {
            **self._stats,
            "active_crawls": len(self._active_crawls),
            "active_agents": len(self._active_agents),
        }

    async def health_check(self) -> AdapterResult:
        """Check Firecrawl API health."""
        if not FIRECRAWL_AVAILABLE:
            return AdapterResult(success=False, error="SDK not installed")

        if not self._app:
            return AdapterResult(
                success=True,
                data={"status": "degraded", "reason": "No API key"}
            )

        try:
            # Simple scrape test
            result = await self._scrape("https://example.com")
            return AdapterResult(
                success=True,
                data={"status": "healthy", "stats": self.get_stats()}
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def shutdown(self) -> AdapterResult:
        """Cleanup resources."""
        # Cancel active crawls
        for crawl_id in list(self._active_crawls.keys()):
            try:
                await self._cancel_crawl(crawl_id)
            except Exception:
                pass

        # Note: Active agents cannot be cancelled through the API
        # They will complete on their own

        self._app = None
        self._async_app = None
        self._status = AdapterStatus.UNINITIALIZED
        self._active_crawls.clear()
        self._active_agents.clear()

        return AdapterResult(success=True, data={"stats": self.get_stats()})


# Helper functions
def create_action(
    action_type: str,
    selector: Optional[str] = None,
    text: Optional[str] = None,
    milliseconds: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Create a Firecrawl action for page interaction.

    Examples:
        create_action("wait", milliseconds=2000)
        create_action("click", selector="#submit-btn")
        create_action("write", selector="#search", text="AI agents")
        create_action("scroll", direction="down", amount=500)
    """
    action = FirecrawlAction(
        type=FirecrawlActionType(action_type),
        selector=selector,
        text=text,
        milliseconds=milliseconds,
        **{k: v for k, v in kwargs.items() if v is not None}
    )
    return action.to_dict()


def get_firecrawl_adapter() -> type[FirecrawlAdapter]:
    """Get the Firecrawl adapter class."""
    return FirecrawlAdapter


if __name__ == "__main__":
    async def test():
        adapter = FirecrawlAdapter()
        await adapter.initialize({})

        # Test scrape
        result = await adapter.execute(
            "scrape",
            url="https://example.com",
            formats=["markdown"]
        )
        print(f"Scrape result: {result}")

        # Test map
        result = await adapter.execute(
            "map",
            url="https://example.com"
        )
        print(f"Map result: {result}")

        await adapter.shutdown()

    asyncio.run(test())
