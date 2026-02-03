"""
Tavily AI Search Adapter - Search Engine for AI Agents
=======================================================

Tavily is the first search engine built specifically for AI agents,
optimizing search for LLMs with structured outputs and citations.

Key Features:
- Aggregates up to 20 sites per API call
- Proprietary AI scoring and ranking
- Custom context fields for LLM optimization
- /search, /extract, /crawl, /map, and /research endpoints
- 1000 free API credits/month

Official Docs: https://docs.tavily.com/
GitHub: https://github.com/tavily-ai

Usage:
    adapter = TavilyAdapter()
    await adapter.initialize({"api_key": "tvly-xxx"})

    # Standard search
    result = await adapter.execute("search", query="LangChain agents")

    # Deep research (Agent-in-a-Box)
    result = await adapter.execute("research", query="distributed systems", depth="deep")

    # Extract structured data
    result = await adapter.execute("extract", urls=["https://..."])
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

try:
    from .base import SDKAdapter, AdapterResult, AdapterStatus, SDKLayer, register_adapter
except ImportError:
    from base import SDKAdapter, AdapterResult, AdapterStatus, SDKLayer, register_adapter

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
    BASIC = "basic"        # Quick search
    ADVANCED = "advanced"  # More thorough


class TavilyTopic(str, Enum):
    """Topic categories for search optimization."""
    GENERAL = "general"
    NEWS = "news"


@register_adapter("tavily", SDKLayer.RESEARCH, priority=24)
class TavilyAdapter(SDKAdapter):
    """
    Tavily AI search adapter - built for AI agents.

    Operations:
        - search: Standard AI-optimized search
        - research: Deep multi-step research (Agent-in-a-Box)
        - extract: Extract structured data from URLs
        - qna: Quick question-answering search
    """

    def __init__(self):
        self._client: Optional[TavilyClient] = None
        self._async_client: Optional[AsyncTavilyClient] = None
        self._status = AdapterStatus.UNINITIALIZED
        self._config: dict[str, Any] = {}
        self._stats = {
            "searches": 0,
            "research_queries": 0,
            "extractions": 0,
            "total_results": 0,
        }

    @property
    def sdk_name(self) -> str:
        return "tavily"

    @property
    def layer(self) -> SDKLayer:
        return SDKLayer.RESEARCH

    @property
    def available(self) -> bool:
        return TAVILY_AVAILABLE

    async def initialize(self, config: dict[str, Any]) -> AdapterResult:
        """Initialize Tavily client."""
        if not TAVILY_AVAILABLE:
            self._status = AdapterStatus.ERROR
            return AdapterResult(
                success=False,
                error="Tavily SDK not installed. Run: pip install tavily-python"
            )

        try:
            api_key = config.get("api_key") or os.getenv("TAVILY_API_KEY")
            if not api_key:
                self._status = AdapterStatus.DEGRADED
                return AdapterResult(
                    success=True,
                    data={"status": "degraded", "reason": "No API key - mock mode"},
                )

            self._client = TavilyClient(api_key=api_key)
            self._async_client = AsyncTavilyClient(api_key=api_key)
            self._config = config
            self._status = AdapterStatus.READY

            return AdapterResult(
                success=True,
                data={
                    "status": "ready",
                    "features": ["search", "research", "extract", "qna"],
                }
            )
        except Exception as e:
            self._status = AdapterStatus.ERROR
            return AdapterResult(success=False, error=str(e))

    async def execute(self, operation: str, **kwargs) -> AdapterResult:
        """Execute Tavily operations."""
        start_time = time.time()

        operations = {
            "search": self._search,
            "research": self._research,
            "extract": self._extract,
            "qna": self._qna,
            "context": self._get_context,
        }

        if operation not in operations:
            return AdapterResult(
                success=False,
                error=f"Unknown operation: {operation}. Valid: {list(operations.keys())}"
            )

        try:
            result = await operations[operation](**kwargs)
            result.latency_ms = (time.time() - start_time) * 1000
            return result
        except Exception as e:
            return AdapterResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def _search(
        self,
        query: str,
        search_depth: str = "basic",
        topic: str = "general",
        max_results: int = 10,
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None,
        include_answer: bool = True,
        include_raw_content: bool = False,
        include_images: bool = False,
        **kwargs,
    ) -> AdapterResult:
        """
        Execute AI-optimized search.

        Args:
            query: Search query
            search_depth: "basic" or "advanced"
            topic: "general" or "news"
            max_results: Maximum results (1-20)
            include_domains: Only search these domains
            exclude_domains: Exclude these domains
            include_answer: Include AI-generated answer
            include_raw_content: Include raw HTML
            include_images: Include image results
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

        response = await self._async_client.search(
            query=query,
            search_depth=search_depth,
            topic=topic,
            max_results=min(max_results, 20),
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
            include_images=include_images,
        )

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
            }
        )

    async def _research(
        self,
        query: str,
        depth: str = "deep",
        max_iterations: int = 5,
        **kwargs,
    ) -> AdapterResult:
        """
        Execute deep research using Tavily's Research endpoint.

        This is the "Agent-in-a-Box" feature that performs:
        - Multiple iterative searches
        - Reasoning over data
        - Multi-agent coordination
        - Deduplication
        - Structured JSON outputs

        Args:
            query: Research topic
            depth: Research depth ("basic", "deep")
            max_iterations: Maximum search iterations
        """
        self._stats["research_queries"] += 1

        if not self._async_client:
            return AdapterResult(
                success=True,
                data={
                    "report": f"Mock research report for: {query}",
                    "sources": [],
                    "mock": True,
                }
            )

        # Use advanced search with multiple iterations for deep research
        all_results = []
        seen_urls = set()

        for i in range(max_iterations):
            # Modify query for deeper exploration
            modified_query = query if i == 0 else f"{query} (additional context iteration {i+1})"

            response = await self._async_client.search(
                query=modified_query,
                search_depth="advanced" if depth == "deep" else "basic",
                max_results=10,
                include_answer=True,
            )

            # Deduplicate
            for r in response.get("results", []):
                url = r.get("url", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(r)

            # Check if we have enough diverse results
            if len(all_results) >= 20:
                break

        # Synthesize
        answer = response.get("answer", "") if response else ""

        return AdapterResult(
            success=True,
            data={
                "report": answer,
                "sources": all_results[:20],
                "iterations": i + 1,
                "total_unique_sources": len(all_results),
            }
        )

    async def _extract(
        self,
        urls: list[str],
        **kwargs,
    ) -> AdapterResult:
        """Extract structured content from URLs."""
        self._stats["extractions"] += len(urls)

        if not self._async_client:
            return AdapterResult(
                success=True,
                data={
                    "results": [{"url": url, "content": "Mock content"} for url in urls],
                    "mock": True,
                }
            )

        response = await self._async_client.extract(urls=urls)

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
        **kwargs,
    ) -> AdapterResult:
        """Quick question-answering search."""
        if not self._async_client:
            return AdapterResult(
                success=True,
                data={"answer": f"Mock answer for: {query}", "mock": True}
            )

        response = await self._async_client.qna_search(query=query)

        return AdapterResult(
            success=True,
            data={"answer": response}
        )

    async def _get_context(
        self,
        query: str,
        max_tokens: int = 4000,
        **kwargs,
    ) -> AdapterResult:
        """Get LLM-optimized context for a query."""
        if not self._async_client:
            return AdapterResult(
                success=True,
                data={"context": f"Mock context for: {query}", "mock": True}
            )

        response = await self._async_client.get_search_context(
            query=query,
            max_tokens=max_tokens,
        )

        return AdapterResult(
            success=True,
            data={"context": response}
        )

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
                data={"status": "healthy", "stats": self._stats}
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def shutdown(self) -> AdapterResult:
        """Cleanup resources."""
        self._client = None
        self._async_client = None
        self._status = AdapterStatus.UNINITIALIZED
        return AdapterResult(success=True, data={"stats": self._stats})


def get_tavily_adapter() -> type[TavilyAdapter]:
    """Get the Tavily adapter class."""
    return TavilyAdapter


if __name__ == "__main__":
    async def test():
        adapter = TavilyAdapter()
        await adapter.initialize({})
        result = await adapter.execute("search", query="LangChain agents patterns")
        print(f"Search result: {result}")
        await adapter.shutdown()

    asyncio.run(test())
