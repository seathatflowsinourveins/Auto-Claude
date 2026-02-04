"""
Tavily AI Search Adapter - Search Engine for AI Agents
=======================================================

Tavily is the first search engine built specifically for AI agents,
optimizing search for LLMs with structured outputs and citations.

Key Features:
- search: AI-optimized search with 20 sites per call
- research: Native deep research endpoint (Agent-in-a-Box)
- get_research: Retrieve async research results
- extract: Structured data extraction from URLs
- crawl: Multi-page website crawling
- map: Website URL discovery
- qna: Quick question answering
- context: LLM-optimized context generation

Official Docs: https://docs.tavily.com/
GitHub: https://github.com/tavily-ai
Python SDK: https://github.com/tavily-ai/tavily-python

API Tiers:
- Free: 1000 credits/month
- Basic: 10,000 credits/month
- Pro: 100,000 credits/month
- Enterprise: Custom

Usage:
    adapter = TavilyAdapter()
    await adapter.initialize({"api_key": "tvly-xxx"})

    # Standard search
    result = await adapter.execute("search", query="LangChain agents")

    # Deep research (native endpoint)
    result = await adapter.execute("research",
        input="distributed systems",
        model="pro",  # "mini", "pro", or "auto"
        citation_format="apa",  # "numbered", "mla", "apa", "chicago"
        stream=False
    )

    # Async research with polling
    result = await adapter.execute("research",
        input="AI safety",
        async_mode=True
    )
    # Later: await adapter.execute("get_research", request_id=result.data["request_id"])
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, AsyncGenerator, Union, Literal, Sequence

# Structured logging
try:
    from ..core.logging_config import get_logger, generate_correlation_id
    _logger = get_logger("adapter.tavily")
except ImportError:
    import logging
    _logger = logging.getLogger(__name__)
    generate_correlation_id = lambda: "corr-fallback"

# Retry utilities
try:
    from .retry import RetryConfig, with_retry, retry_async
except ImportError:
    # Fallback for standalone testing
    RetryConfig = None
    with_retry = lambda f=None, **kw: (lambda fn: fn) if f is None else f
    retry_async = None

# HTTP connection pool (for future HTTP-based operations)
try:
    from .http_pool import (
        HTTPConnectionPool,
        get_shared_pool_sync,
        get_config_for_service,
        PoolMetrics,
    )
    HTTP_POOL_AVAILABLE = True
except ImportError:
    HTTPConnectionPool = None
    get_shared_pool_sync = None
    get_config_for_service = None
    PoolMetrics = None
    HTTP_POOL_AVAILABLE = False

# Circuit breaker imports
try:
    from .circuit_breaker_manager import adapter_circuit_breaker, get_adapter_circuit_manager
    from ..core.resilience import CircuitOpenError
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

try:
    from ..core.orchestration.base import SDKAdapter, AdapterResult, AdapterStatus, SDKLayer, register_adapter
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


# Tavily SDK
TAVILY_AVAILABLE = False
try:
    from tavily import TavilyClient, AsyncTavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TavilyClient = None
    AsyncTavilyClient = None


class TavilySearchDepth(str, Enum):
    """Search depth options."""
    ULTRA_FAST = "ultra-fast"  # Fastest, lower relevance
    FAST = "fast"              # Very fast, moderate relevance
    BASIC = "basic"            # Quick search (1 credit)
    ADVANCED = "advanced"      # More thorough (2 credits)


class TavilyTopic(str, Enum):
    """Topic categories for search optimization."""
    GENERAL = "general"   # General-purpose (country param available)
    NEWS = "news"         # Real-time updates, current events
    FINANCE = "finance"   # Financial and stock market data


class TavilyResearchModel(str, Enum):
    """Research model options per official API docs."""
    MINI = "mini"    # Faster, lighter research
    PRO = "pro"      # Full deep research
    AUTO = "auto"    # Automatic model selection based on query


class TavilyCitationFormat(str, Enum):
    """Citation format options per official API docs."""
    NUMBERED = "numbered"   # Default numbered citations
    MLA = "mla"             # MLA citation format
    APA = "apa"             # APA citation format
    CHICAGO = "chicago"     # Chicago citation format


@register_adapter("tavily", SDKLayer.RESEARCH, priority=24)
class TavilyAdapter(SDKAdapter):
    """
    Tavily AI search adapter - built for AI agents.

    Operations:
        - search: Standard AI-optimized search
        - research: Native deep research endpoint
        - get_research: Retrieve async research results
        - extract: Extract structured data from URLs
        - qna: Quick question-answering search
        - context: Get LLM-optimized context
        - map: Map website structure
        - crawl: Crawl and extract website content
    """

    def __init__(self):
        self._client: Optional[TavilyClient] = None
        self._async_client: Optional[AsyncTavilyClient] = None
        self._status = AdapterStatus.UNINITIALIZED
        self._config: Dict[str, Any] = {}
        self._stats = {
            "searches": 0,
            "research_queries": 0,
            "extractions": 0,
            "qna_queries": 0,
            "maps": 0,
            "crawls": 0,
            "total_results": 0,
            "avg_latency_ms": 0.0,
            "retries": 0,
        }
        self._pending_research: Dict[str, Dict[str, Any]] = {}
        # Retry configuration for transient errors
        self._retry_config = RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=60.0,
            jitter=0.5,
        ) if RetryConfig else None

    @property
    def sdk_name(self) -> str:
        return "tavily"

    @property
    def layer(self) -> SDKLayer:
        return SDKLayer.RESEARCH

    @property
    def available(self) -> bool:
        return TAVILY_AVAILABLE

    async def initialize(self, config: Dict[str, Any]) -> AdapterResult:
        """Initialize Tavily client."""
        start = time.time()
        _logger.info("Initializing Tavily adapter", adapter="tavily")

        if not TAVILY_AVAILABLE:
            self._status = AdapterStatus.ERROR
            _logger.error("Tavily SDK not installed", adapter="tavily")
            return AdapterResult(
                success=False,
                error="Tavily SDK not installed. Run: pip install tavily-python",
                latency_ms=(time.time() - start) * 1000,
            )

        try:
            api_key = config.get("api_key") or os.environ.get("TAVILY_API_KEY")

            if not api_key:
                self._status = AdapterStatus.DEGRADED
                _logger.warning("No API key provided, running in mock mode", adapter="tavily")
                return AdapterResult(
                    success=True,
                    data={"status": "degraded", "reason": "No API key - mock mode"},
                    latency_ms=(time.time() - start) * 1000,
                )

            self._client = TavilyClient(api_key=api_key)
            self._async_client = AsyncTavilyClient(api_key=api_key)
            self._config = config
            self._status = AdapterStatus.READY

            _logger.info("Tavily adapter initialized successfully", adapter="tavily", status="ready")
            return AdapterResult(
                success=True,
                data={
                    "status": "ready",
                    "features": [
                        "search", "research", "get_research", "extract",
                        "qna", "context", "map", "crawl"
                    ],
                    "search_depths": [d.value for d in TavilySearchDepth],
                    "topics": [t.value for t in TavilyTopic],
                    "research_models": [m.value for m in TavilyResearchModel],
                },
                latency_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            self._status = AdapterStatus.ERROR
            _logger.exception("Failed to initialize Tavily adapter", adapter="tavily", error_type=type(e).__name__)
            return AdapterResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start) * 1000,
            )

    async def execute(self, operation: str, **kwargs) -> AdapterResult:
        """Execute Tavily operations with circuit breaker protection."""
        start = time.time()
        correlation_id = generate_correlation_id()

        operations = {
            "search": self._search,
            "research": self._research,
            "get_research": self._get_research,
            "extract": self._extract,
            "qna": self._qna,
            "context": self._get_context,
            "map": self._map,
            "crawl": self._crawl,
        }

        if operation not in operations:
            _logger.warning(
                "Unknown operation requested",
                adapter="tavily",
                operation=operation,
                correlation_id=correlation_id,
            )
            return AdapterResult(
                success=False,
                error=f"Unknown operation: {operation}. Valid: {list(operations.keys())}",
                latency_ms=(time.time() - start) * 1000,
            )

        # Log operation start (sampled for high-volume)
        _logger.debug(
            "Executing Tavily operation",
            adapter="tavily",
            operation=operation,
            correlation_id=correlation_id,
            sample_rate=0.1,
        )

        # Execute with circuit breaker protection
        try:
            async with adapter_circuit_breaker("tavily_adapter"):
                result = await operations[operation](**kwargs)
                result.latency_ms = (time.time() - start) * 1000
                self._update_avg_latency(result.latency_ms)

                # Log successful completion
                _logger.info(
                    "Tavily operation completed",
                    adapter="tavily",
                    operation=operation,
                    correlation_id=correlation_id,
                    duration_ms=result.latency_ms,
                    success=result.success,
                    sample_rate=0.5,
                )
                return result
        except CircuitOpenError as e:
            latency_ms = (time.time() - start) * 1000
            _logger.warning(
                "Circuit breaker open",
                adapter="tavily",
                operation=operation,
                correlation_id=correlation_id,
                duration_ms=latency_ms,
            )
            # Circuit is open - return fallback response
            return AdapterResult(
                success=False,
                error=f"Circuit breaker open for tavily_adapter: {e}",
                latency_ms=latency_ms,
                metadata={"circuit_breaker": "open", "adapter": "tavily", "correlation_id": correlation_id},
            )
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            _logger.exception(
                "Tavily operation failed",
                adapter="tavily",
                operation=operation,
                correlation_id=correlation_id,
                duration_ms=latency_ms,
                error_type=type(e).__name__,
            )
            return AdapterResult(
                success=False,
                error=str(e),
                latency_ms=latency_ms,
                metadata={"correlation_id": correlation_id},
            )

    async def _search(
        self,
        query: str,
        search_depth: Optional[Literal["basic", "advanced", "fast", "ultra-fast"]] = None,
        topic: Optional[Literal["general", "news", "finance"]] = None,
        time_range: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: Optional[int] = None,
        max_results: int = 10,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        include_answer: Union[bool, Literal["basic", "advanced"]] = True,
        include_raw_content: Union[bool, Literal["markdown", "text"]] = False,
        include_images: bool = False,
        country: Optional[str] = None,
        auto_parameters: bool = False,
        include_favicon: bool = False,
        include_usage: bool = False,
        timeout: float = 60,
        **kwargs,
    ) -> AdapterResult:
        """
        Execute AI-optimized search.

        Args:
            query: Search query
            search_depth: "ultra-fast", "fast", "basic", or "advanced"
            topic: "general", "news", or "finance"
            time_range: "day", "week", "month", or "year" (d/w/m/y)
            start_date: Start date for search range (YYYY-MM-DD)
            end_date: End date for search range (YYYY-MM-DD)
            days: Number of days back to search
            max_results: Maximum results (1-20)
            include_domains: Only search these domains (up to 300)
            exclude_domains: Exclude these domains (up to 150)
            include_answer: Include AI-generated answer (bool, "basic", or "advanced")
            include_raw_content: Include raw content ("markdown" or "text", or bool)
            include_images: Include image results
            country: 2-letter country code (190+ supported, general topic only)
            auto_parameters: Let Tavily automatically optimize parameters
            include_favicon: Include favicon URLs in results
            include_usage: Include API usage information in response
            timeout: Request timeout in seconds
        """
        self._stats["searches"] += 1

        if not self._async_client:
            return AdapterResult(
                success=True,
                data={
                    "answer": f"Mock answer for: {query}",
                    "results": [
                        {
                            "title": f"Mock result: {query}",
                            "url": "https://example.com",
                            "content": "Mock content",
                            "score": 0.95,
                        }
                    ],
                    "mock": True,
                }
            )

        # Build search parameters
        search_params = {
            "query": query,
            "max_results": min(max_results, 20),
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "include_images": include_images,
        }

        # Optional parameters
        if search_depth:
            search_params["search_depth"] = search_depth
        if topic:
            search_params["topic"] = topic
        if include_domains:
            search_params["include_domains"] = include_domains[:300]
        if exclude_domains:
            search_params["exclude_domains"] = exclude_domains[:150]
        if time_range:
            search_params["time_range"] = time_range
        if start_date:
            search_params["start_date"] = start_date
        if end_date:
            search_params["end_date"] = end_date
        if days is not None:
            search_params["days"] = days
        if country and (topic == "general" or topic is None):
            search_params["country"] = country
        if auto_parameters:
            search_params["auto_parameters"] = True
        if include_favicon:
            search_params["include_favicon"] = True
        if include_usage:
            search_params["include_usage"] = True
        if timeout != 60:
            search_params["timeout"] = timeout

        # Execute search with retry logic
        async def _do_search():
            return await self._async_client.search(**search_params)

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
            response = await retry_async(_do_search, config=config)
        else:
            response = await _do_search()

        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": r.get("score", 0.0),
                "published_date": r.get("published_date"),
            })

        self._stats["total_results"] += len(results)

        return AdapterResult(
            success=True,
            data={
                "answer": response.get("answer"),
                "results": results,
                "count": len(results),
                "images": response.get("images", []) if include_images else [],
                "response_time": response.get("response_time"),
                "query": query,
            }
        )

    async def _research(
        self,
        input: str = None,
        query: str = None,  # Backward compatibility - deprecated, use 'input'
        model: Optional[Literal["mini", "pro", "auto"]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        citation_format: Literal["numbered", "mla", "apa", "chicago"] = "numbered",
        max_sources: int = 20,
        async_mode: bool = False,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> AdapterResult:
        """
        Execute native deep research using Tavily's /research endpoint.

        This is the "Agent-in-a-Box" feature that performs:
        - Multi-step iterative research
        - Reasoning over accumulated data
        - Source deduplication
        - Citation formatting
        - Structured JSON outputs (with output_schema)

        Args:
            input: Research topic or question (required)
            model: "mini", "pro", or "auto" for automatic selection
            output_schema: JSON schema for structured output
            stream: Enable streaming response (returns generator)
            citation_format: "numbered", "mla", "apa", or "chicago"
            max_sources: Maximum unique sources to include
            async_mode: Return immediately with request_id for polling
            timeout: Request timeout in seconds
        """
        # Backward compatibility: 'query' param deprecated, use 'input'
        if query and not input:
            logger.warning("Tavily research: 'query' param is deprecated, use 'input' instead")
            input = query

        if not input:
            return AdapterResult(
                success=False,
                error="Research requires 'input' parameter (the research topic)"
            )

        self._stats["research_queries"] += 1

        if not self._async_client:
            return AdapterResult(
                success=True,
                data={
                    "report": f"Mock research report for: {input}",
                    "sources": [],
                    "citations": [],
                    "mock": True,
                }
            )

        # Build research parameters - use 'input' as the API parameter name
        research_params = {
            "input": input,
        }

        # Add optional parameters if SDK supports them
        if model:
            research_params["model"] = model
        if citation_format:
            research_params["citation_format"] = citation_format
        if max_sources:
            research_params["max_sources"] = max_sources
        if output_schema:
            research_params["output_schema"] = output_schema
        if timeout:
            research_params["timeout"] = timeout

        try:
            # Try native research() method first
            if hasattr(self._async_client, 'research'):
                if async_mode:
                    # Async research with request_id
                    response = await self._async_client.research(**research_params, async_mode=True)
                    request_id = response.get("request_id", response.get("id"))

                    self._pending_research[request_id] = {
                        "input": input,
                        "started_at": datetime.utcnow(),
                        "params": research_params,
                    }

                    return AdapterResult(
                        success=True,
                        data={
                            "request_id": request_id,
                            "status": "processing",
                            "message": f"Research started. Use get_research('{request_id}') to retrieve results.",
                        }
                    )

                elif stream:
                    # Streaming research (returns generator metadata)
                    # Caller must handle streaming separately
                    return AdapterResult(
                        success=True,
                        data={
                            "streaming": True,
                            "message": "Use research_stream() for streaming results",
                            "params": research_params,
                        }
                    )

                else:
                    # Synchronous research
                    response = await self._async_client.research(**research_params)

                    return AdapterResult(
                        success=True,
                        data={
                            "report": response.get("report", response.get("content", "")),
                            "sources": response.get("sources", []),
                            "citations": response.get("citations", []),
                            "model": model,
                            "input": input,
                        }
                    )

            else:
                # Fallback: simulate research with multiple advanced searches
                return await self._research_fallback(input, max_sources, **kwargs)

        except AttributeError:
            # SDK doesn't have research() - use fallback
            return await self._research_fallback(input, max_sources, **kwargs)

        except Exception as e:
            # If research endpoint fails, try fallback
            if "research" in str(e).lower() or "endpoint" in str(e).lower():
                return await self._research_fallback(input, max_sources, **kwargs)
            raise

    async def _research_fallback(
        self,
        input: str,
        max_sources: int = 20,
        max_iterations: int = 3,
        **kwargs,
    ) -> AdapterResult:
        """
        Fallback research using multiple search iterations.
        Used when native /research endpoint is unavailable.
        """
        all_results = []
        seen_urls = set()
        answers = []

        # Iterative search with query refinement
        search_queries = [
            input,
            f"{input} detailed analysis",
            f"{input} examples implementation",
        ][:max_iterations]

        for i, q in enumerate(search_queries):
            response = await self._async_client.search(
                query=q,
                search_depth="advanced",
                max_results=10,
                include_answer=True,
            )

            # Collect answer
            if response.get("answer"):
                answers.append(response["answer"])

            # Deduplicate results
            for r in response.get("results", []):
                url = r.get("url", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append({
                        "title": r.get("title", ""),
                        "url": url,
                        "content": r.get("content", ""),
                        "score": r.get("score", 0.0),
                    })

            if len(all_results) >= max_sources:
                break

        # Synthesize report from answers
        report = "\n\n".join(answers) if answers else f"Research findings for: {input}"

        return AdapterResult(
            success=True,
            data={
                "report": report,
                "sources": all_results[:max_sources],
                "citations": [{"index": i+1, "url": r["url"]} for i, r in enumerate(all_results[:max_sources])],
                "iterations": len(search_queries),
                "fallback": True,
                "input": input,
            }
        )

    async def _get_research(
        self,
        request_id: str,
        **kwargs,
    ) -> AdapterResult:
        """
        Retrieve results of async research by request_id.

        Args:
            request_id: The request ID from async research call
        """
        if not self._async_client:
            return AdapterResult(
                success=True,
                data={
                    "request_id": request_id,
                    "status": "completed",
                    "report": "Mock research report",
                    "mock": True,
                }
            )

        try:
            if hasattr(self._async_client, 'get_research'):
                response = await self._async_client.get_research(request_id)

                status = response.get("status", "unknown")

                if status == "completed":
                    # Remove from pending
                    self._pending_research.pop(request_id, None)

                    return AdapterResult(
                        success=True,
                        data={
                            "request_id": request_id,
                            "status": "completed",
                            "report": response.get("report", response.get("content", "")),
                            "sources": response.get("sources", []),
                            "citations": response.get("citations", []),
                        }
                    )
                else:
                    return AdapterResult(
                        success=True,
                        data={
                            "request_id": request_id,
                            "status": status,
                            "progress": response.get("progress", 0),
                        }
                    )

            else:
                # SDK doesn't support get_research
                return AdapterResult(
                    success=False,
                    error="get_research not supported by SDK version"
                )

        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _extract(
        self,
        urls: List[str],
        include_images: bool = False,
        extract_depth: Optional[Literal["basic", "advanced"]] = None,
        format: Optional[Literal["markdown", "text"]] = None,
        query: Optional[str] = None,
        chunks_per_source: Optional[int] = None,
        include_favicon: bool = False,
        include_usage: bool = False,
        **kwargs,
    ) -> AdapterResult:
        """
        Extract structured content from URLs.

        Args:
            urls: List of URLs to extract (up to 20)
            include_images: Include image extraction
            extract_depth: "basic" or "advanced" extraction depth
            format: Output format - "markdown" or "text"
            query: Optional query to filter/focus extraction
            chunks_per_source: Number of chunks per source URL
            include_favicon: Include favicon URLs in results
            include_usage: Include API usage information in response
        """
        self._stats["extractions"] += len(urls)

        if not self._async_client:
            return AdapterResult(
                success=True,
                data={
                    "results": [{"url": url, "content": f"Mock content from {url}"} for url in urls],
                    "mock": True,
                }
            )

        # Build extract parameters
        extract_params = {
            "urls": urls[:20],
            "include_images": include_images,
        }

        if extract_depth:
            extract_params["extract_depth"] = extract_depth
        if format:
            extract_params["format"] = format
        if query:
            extract_params["query"] = query
        if chunks_per_source is not None:
            extract_params["chunks_per_source"] = chunks_per_source
        if include_favicon:
            extract_params["include_favicon"] = True
        if include_usage:
            extract_params["include_usage"] = True

        response = await self._async_client.extract(**extract_params)

        return AdapterResult(
            success=True,
            data={
                "results": response.get("results", []),
                "failed_urls": response.get("failed_results", []),
            }
        )

    async def _qna(
        self,
        query: str,
        search_depth: str = "advanced",
        **kwargs,
    ) -> AdapterResult:
        """
        Quick question-answering search.

        Args:
            query: Question to answer
            search_depth: Search depth for finding answer
        """
        self._stats["qna_queries"] += 1

        if not self._async_client:
            return AdapterResult(
                success=True,
                data={"answer": f"Mock answer for: {query}", "mock": True}
            )

        response = await self._async_client.qna_search(
            query=query,
            search_depth=search_depth,
        )

        return AdapterResult(
            success=True,
            data={
                "answer": response,
                "query": query,
            }
        )

    async def _get_context(
        self,
        query: str,
        max_tokens: int = 4000,
        search_depth: str = "basic",
        topic: str = "general",
        **kwargs,
    ) -> AdapterResult:
        """
        Get LLM-optimized context for a query.

        Args:
            query: Context query
            max_tokens: Maximum tokens in context
            search_depth: Search depth
            topic: Topic category
        """
        if not self._async_client:
            return AdapterResult(
                success=True,
                data={"context": f"Mock context for: {query}", "mock": True}
            )

        response = await self._async_client.get_search_context(
            query=query,
            max_tokens=max_tokens,
            search_depth=search_depth,
            topic=topic,
        )

        return AdapterResult(
            success=True,
            data={
                "context": response,
                "query": query,
                "max_tokens": max_tokens,
            }
        )

    async def _map(
        self,
        url: str,
        max_depth: int = 1,
        max_breadth: int = 20,
        limit: int = 50,
        instructions: Optional[str] = None,
        allow_external: bool = True,
        select_paths: Optional[Sequence[str]] = None,
        select_domains: Optional[Sequence[str]] = None,
        exclude_paths: Optional[Sequence[str]] = None,
        exclude_domains: Optional[Sequence[str]] = None,
        include_favicon: bool = False,
        include_usage: bool = False,
        chunks_per_source: Optional[int] = None,
        **kwargs,
    ) -> AdapterResult:
        """
        Map website structure without extracting content.

        Args:
            url: Root URL for mapping
            max_depth: 1-5 levels from base URL
            max_breadth: 1-500 links per page level
            limit: Total pages to process (max 50)
            instructions: Natural language guidance for mapping
            allow_external: Include external links
            select_paths: Only include URLs matching these paths
            select_domains: Only include URLs from these domains
            exclude_paths: Exclude URLs matching these paths
            exclude_domains: Exclude URLs from these domains
            include_favicon: Include favicon URLs in results
            include_usage: Include API usage information in response
            chunks_per_source: Number of chunks per source URL
        """
        self._stats["maps"] += 1

        if not self._async_client:
            return AdapterResult(
                success=True,
                data={
                    "base_url": url,
                    "results": [url, f"{url}/page1", f"{url}/page2"],
                    "mock": True,
                }
            )

        try:
            map_params = {
                "url": url,
                "max_depth": min(max_depth, 5),
                "max_breadth": min(max_breadth, 500),
                "limit": min(limit, 50),
                "allow_external": allow_external,
            }
            if instructions:
                map_params["instructions"] = instructions
            if select_paths:
                map_params["select_paths"] = list(select_paths)
            if select_domains:
                map_params["select_domains"] = list(select_domains)
            if exclude_paths:
                map_params["exclude_paths"] = list(exclude_paths)
            if exclude_domains:
                map_params["exclude_domains"] = list(exclude_domains)
            if include_favicon:
                map_params["include_favicon"] = True
            if include_usage:
                map_params["include_usage"] = True
            if chunks_per_source is not None:
                map_params["chunks_per_source"] = chunks_per_source

            response = await self._async_client.map(**map_params)

            return AdapterResult(
                success=True,
                data={
                    "base_url": url,
                    "results": response.get("results", []),
                    "total_pages": len(response.get("results", [])),
                    "response_time": response.get("response_time"),
                }
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _crawl(
        self,
        url: str,
        max_depth: int = 1,
        max_breadth: int = 20,
        limit: int = 50,
        instructions: Optional[str] = None,
        extract_depth: Optional[Literal["basic", "advanced"]] = None,
        format: Optional[Literal["markdown", "text"]] = None,
        allow_external: bool = True,
        include_images: bool = False,
        select_paths: Optional[Sequence[str]] = None,
        select_domains: Optional[Sequence[str]] = None,
        exclude_paths: Optional[Sequence[str]] = None,
        exclude_domains: Optional[Sequence[str]] = None,
        include_favicon: bool = False,
        include_usage: bool = False,
        chunks_per_source: Optional[int] = None,
        **kwargs,
    ) -> AdapterResult:
        """
        Crawl and extract content from website.

        Args:
            url: Root URL for crawl
            max_depth: 1-5 levels from base URL
            max_breadth: 1-500 links per page level
            limit: Total pages to process
            instructions: Natural language guidance
            extract_depth: "basic" or "advanced"
            format: "markdown" or "text"
            allow_external: Follow external links
            include_images: Extract images
            select_paths: Only include URLs matching these paths
            select_domains: Only include URLs from these domains
            exclude_paths: Exclude URLs matching these paths
            exclude_domains: Exclude URLs from these domains
            include_favicon: Include favicon URLs in results
            include_usage: Include API usage information in response
            chunks_per_source: Number of chunks per source URL
        """
        self._stats["crawls"] += 1

        if not self._async_client:
            return AdapterResult(
                success=True,
                data={
                    "base_url": url,
                    "results": [{"url": url, "content": "Mock crawled content"}],
                    "mock": True,
                }
            )

        try:
            crawl_params = {
                "url": url,
                "max_depth": min(max_depth, 5),
                "max_breadth": min(max_breadth, 500),
                "limit": min(limit, 50),
                "allow_external": allow_external,
                "include_images": include_images,
            }
            if instructions:
                crawl_params["instructions"] = instructions
            if extract_depth:
                crawl_params["extract_depth"] = extract_depth
            if format:
                crawl_params["format"] = format
            if select_paths:
                crawl_params["select_paths"] = list(select_paths)
            if select_domains:
                crawl_params["select_domains"] = list(select_domains)
            if exclude_paths:
                crawl_params["exclude_paths"] = list(exclude_paths)
            if exclude_domains:
                crawl_params["exclude_domains"] = list(exclude_domains)
            if include_favicon:
                crawl_params["include_favicon"] = True
            if include_usage:
                crawl_params["include_usage"] = True
            if chunks_per_source is not None:
                crawl_params["chunks_per_source"] = chunks_per_source

            response = await self._async_client.crawl(**crawl_params)

            self._stats["total_results"] += len(response.get("results", []))

            return AdapterResult(
                success=True,
                data={
                    "base_url": url,
                    "results": response.get("results", []),
                    "total_pages": len(response.get("results", [])),
                    "failed_results": response.get("failed_results", []),
                    "response_time": response.get("response_time"),
                }
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    def _update_avg_latency(self, latency: float):
        """Update rolling average latency."""
        total_ops = sum([
            self._stats["searches"],
            self._stats["research_queries"],
            self._stats["extractions"],
            self._stats["qna_queries"],
            self._stats["maps"],
            self._stats["crawls"],
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
            "pending_research": len(self._pending_research),
        }

    async def health_check(self) -> AdapterResult:
        """Check Tavily API health."""
        if not TAVILY_AVAILABLE:
            return AdapterResult(success=False, error="SDK not installed")

        if not self._async_client:
            return AdapterResult(
                success=True,
                data={"status": "degraded", "reason": "No API key"}
            )

        try:
            result = await self._search("test", max_results=1)
            return AdapterResult(
                success=True,
                data={"status": "healthy", "stats": self.get_stats()}
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def shutdown(self) -> AdapterResult:
        """Cleanup resources."""
        self._client = None
        self._async_client = None
        self._status = AdapterStatus.UNINITIALIZED
        self._pending_research.clear()
        return AdapterResult(success=True, data={"stats": self.get_stats()})


def get_tavily_adapter() -> type[TavilyAdapter]:
    """Get the Tavily adapter class."""
    return TavilyAdapter


if __name__ == "__main__":
    async def test():
        adapter = TavilyAdapter()
        await adapter.initialize({})

        # Test search
        result = await adapter.execute("search", query="LangChain agents patterns")
        print(f"Search result: {result}")

        # Test research (using 'input' parameter, not 'query')
        result = await adapter.execute("research", input="AI agent architectures", model="pro")
        print(f"Research result: {result}")

        await adapter.shutdown()

    asyncio.run(test())
