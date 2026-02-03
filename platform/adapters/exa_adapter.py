"""
Exa AI Search Adapter - Neural Search for UNLEASH
==================================================

Exa is the first search engine optimized for AI - uses neural/embeddings-based
search rather than keyword matching.

Exa 2.0 Features (2026):
- Exa Fast: <350ms P50 latency, 30% faster than competitors
- Exa Auto: Higher quality default with intelligent mode selection
- Exa Deep: Agentic retrieval with multi-step search (3.5s P50)

Official Docs: https://docs.exa.ai/
GitHub: https://github.com/exa-labs/exa-py

Usage:
    adapter = ExaAdapter()
    await adapter.initialize({"api_key": "exa-xxx"})

    # Fast search (<350ms)
    result = await adapter.execute("search", query="LangGraph patterns", type="fast")

    # Deep agentic search (highest quality)
    result = await adapter.execute("search", query="distributed consensus", type="deep")

    # Get contents from URLs
    result = await adapter.execute("get_contents", urls=["https://..."])
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

# SDK Layer imports
try:
    from ..core.orchestration.base import SDKAdapter, AdapterResult, AdapterStatus, SDKLayer, register_adapter
except ImportError:
    try:
        from core.orchestration.base import SDKAdapter, AdapterResult, AdapterStatus, SDKLayer, register_adapter
    except ImportError:
        # Minimal fallback definitions for standalone use
        from enum import Enum, IntEnum
        from dataclasses import dataclass, field
        from datetime import datetime
        from typing import Dict, Any, Optional
        from abc import ABC, abstractmethod

        class SDKLayer(IntEnum):
            RESEARCH = 8

        class AdapterStatus(Enum):
            UNINITIALIZED = "uninitialized"
            READY = "ready"
            FAILED = "failed"

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

# Exa SDK
EXA_AVAILABLE = False
try:
    from exa_py import Exa
    EXA_AVAILABLE = True
except ImportError:
    Exa = None


class ExaSearchType(str, Enum):
    """Exa search types with different speed/quality tradeoffs."""
    FAST = "fast"          # <350ms, streamlined models
    AUTO = "auto"          # Default, intelligent hybrid
    NEURAL = "neural"      # Pure embeddings-based
    DEEP = "deep"          # Agentic retrieval, highest quality (3.5s)


@dataclass
class ExaSearchResult:
    """Individual search result from Exa."""
    title: str
    url: str
    text: str = ""
    highlights: list[str] = field(default_factory=list)
    score: float = 0.0
    published_date: Optional[str] = None
    author: Optional[str] = None


@register_adapter("exa", SDKLayer.RESEARCH, priority=25)
class ExaAdapter(SDKAdapter):
    """
    Exa AI neural search adapter.

    Operations:
        - search: Neural/semantic search with type selection
        - get_contents: Extract content from URLs
        - find_similar: Find similar content to a URL
        - search_and_contents: Combined search + content extraction
    """

    def __init__(self):
        self._client: Optional[Exa] = None
        self._status = AdapterStatus.UNINITIALIZED
        self._config: dict[str, Any] = {}
        self._stats = {
            "searches": 0,
            "contents_fetched": 0,
            "total_results": 0,
            "avg_latency_ms": 0.0,
        }

    @property
    def sdk_name(self) -> str:
        return "exa"

    @property
    def layer(self) -> SDKLayer:
        return SDKLayer.RESEARCH

    @property
    def available(self) -> bool:
        return EXA_AVAILABLE

    async def initialize(self, config: dict[str, Any]) -> AdapterResult:
        """Initialize Exa client with API key."""
        if not EXA_AVAILABLE:
            self._status = AdapterStatus.ERROR
            return AdapterResult(
                success=False,
                error="Exa SDK not installed. Run: pip install exa-py"
            )

        try:
            api_key = config.get("api_key") or os.getenv("EXA_API_KEY")
            if not api_key:
                self._status = AdapterStatus.DEGRADED
                return AdapterResult(
                    success=True,
                    data={"status": "degraded", "reason": "No API key - mock mode"},
                )

            self._client = Exa(api_key=api_key)
            self._config = config
            self._status = AdapterStatus.READY

            return AdapterResult(
                success=True,
                data={
                    "status": "ready",
                    "features": ["fast", "auto", "neural", "deep"],
                    "version": "2.0",
                }
            )
        except Exception as e:
            self._status = AdapterStatus.ERROR
            return AdapterResult(success=False, error=str(e))

    async def execute(self, operation: str, **kwargs) -> AdapterResult:
        """Execute Exa operations."""
        start_time = time.time()

        operations = {
            "search": self._search,
            "get_contents": self._get_contents,
            "find_similar": self._find_similar,
            "search_and_contents": self._search_and_contents,
            "answer": self._answer,
            "research": self._research,
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
        type: str = "auto",
        num_results: int = 10,
        include_domains: Optional[list[str]] = None,
        exclude_domains: Optional[list[str]] = None,
        start_published_date: Optional[str] = None,
        end_published_date: Optional[str] = None,
        use_autoprompt: bool = True,
        category: Optional[str] = None,
        livecrawl: str = "fallback",
        include_text: Optional[list[str]] = None,
        exclude_text: Optional[list[str]] = None,
        moderation: bool = False,
        user_location: Optional[str] = None,
        **kwargs,
    ) -> AdapterResult:
        """
        Execute neural search with full Exa 2.0+ capabilities.

        Args:
            query: Search query
            type: Search type:
                - "auto" (default): Intelligent hybrid selection
                - "neural": Semantic/embedding-based search
                - "keyword": Traditional BM25-style search
                - "fast": <350ms latency optimized
                - "deep": Exhaustive multi-step agentic search
            num_results: Number of results (max 100)
            include_domains: Only search these domains (max 1200, supports paths)
            exclude_domains: Exclude these domains (max 1200)
            start_published_date: Filter by date (YYYY-MM-DD)
            end_published_date: Filter by date
            use_autoprompt: Let Exa optimize the query
            category: Focus on: "company", "research paper", "news", "tweet",
                      "personal site", "financial report", "people", "pdf"
            livecrawl: "never", "fallback", "preferred", "always", "auto"
            include_text: Strings that must appear in results (max 1 string, 5 words)
            exclude_text: Strings to exclude from results
            moderation: Filter unsafe content
            user_location: Two-letter ISO country code for localization
        """
        self._stats["searches"] += 1

        if not self._client:
            # Mock mode for testing
            return AdapterResult(
                success=True,
                data={
                    "results": [
                        {
                            "title": f"Mock result for: {query}",
                            "url": "https://example.com/mock",
                            "text": f"This is a mock result for query: {query}",
                            "score": 0.95,
                        }
                    ],
                    "type": type,
                    "mock": True,
                }
            )

        # Build search parameters
        search_params = {
            "query": query,
            "num_results": min(num_results, 100),
        }

        # use_autoprompt is deprecated in newer Exa SDK versions
        # Only add if the SDK version supports it
        try:
            import inspect
            sig = inspect.signature(self._client.search)
            if 'use_autoprompt' in sig.parameters:
                search_params["use_autoprompt"] = use_autoprompt
        except Exception:
            pass  # Skip if we can't introspect

        # Map type to Exa search type
        if type == "fast":
            search_params["type"] = "keyword"  # Fast mode
        elif type == "neural":
            search_params["type"] = "neural"
        elif type == "deep":
            # Deep mode uses auto with more results and processing
            search_params["type"] = "auto"
            search_params["num_results"] = min(num_results * 2, 100)
        else:
            search_params["type"] = "auto"

        # Add filters
        if include_domains:
            search_params["include_domains"] = include_domains[:1200]
        if exclude_domains:
            search_params["exclude_domains"] = exclude_domains[:1200]
        if start_published_date:
            search_params["start_published_date"] = start_published_date
        if end_published_date:
            search_params["end_published_date"] = end_published_date
        if category:
            # Note: company/people categories don't support date filtering
            search_params["category"] = category
        if include_text:
            search_params["include_text"] = include_text[:1]  # Max 1 string, 5 words
        if exclude_text:
            search_params["exclude_text"] = exclude_text
        if moderation:
            search_params["moderation"] = True
        if user_location:
            search_params["user_location"] = user_location

        # Execute search
        response = self._client.search(**search_params)

        # Parse results
        results = []
        for r in response.results:
            results.append({
                "title": r.title,
                "url": r.url,
                "text": getattr(r, "text", ""),
                "highlights": getattr(r, "highlights", []),
                "score": getattr(r, "score", 0.0),
                "published_date": getattr(r, "published_date", None),
                "author": getattr(r, "author", None),
            })

        self._stats["total_results"] += len(results)

        return AdapterResult(
            success=True,
            data={
                "results": results,
                "count": len(results),
                "type": type,
                "autoprompt_used": use_autoprompt,
            }
        )

    async def _get_contents(
        self,
        urls: list[str],
        text: bool = True,
        highlights: bool = False,
        summary: bool = False,
        subpages: int = 0,
        livecrawl: str = "fallback",
        **kwargs,
    ) -> AdapterResult:
        """
        Get contents from URLs.

        Args:
            urls: List of URLs to fetch
            text: Include full text
            highlights: Include highlights
            summary: Generate AI summary
            subpages: Number of subpages to crawl (0-10)
            livecrawl: "never", "fallback", "preferred", "always", "auto"
        """
        self._stats["contents_fetched"] += len(urls)

        if not self._client:
            return AdapterResult(
                success=True,
                data={
                    "contents": [{"url": url, "text": "Mock content"} for url in urls],
                    "mock": True,
                }
            )

        # Build content options
        contents_options = {}
        if text:
            contents_options["text"] = True
        if highlights:
            contents_options["highlights"] = {"num_sentences": 3}
        if summary:
            contents_options["summary"] = True
        if subpages > 0:
            contents_options["subpages"] = min(subpages, 10)
        if livecrawl and livecrawl != "fallback":
            contents_options["livecrawl"] = livecrawl

        response = self._client.get_contents(urls, **contents_options)

        contents = []
        for r in response.results:
            contents.append({
                "url": r.url,
                "title": getattr(r, "title", ""),
                "text": getattr(r, "text", ""),
                "highlights": getattr(r, "highlights", []),
            })

        return AdapterResult(
            success=True,
            data={
                "contents": contents,
                "count": len(contents),
            }
        )

    async def _find_similar(
        self,
        url: str,
        num_results: int = 10,
        exclude_source_domain: bool = True,
        **kwargs,
    ) -> AdapterResult:
        """Find content similar to a given URL."""
        if not self._client:
            return AdapterResult(
                success=True,
                data={"results": [], "mock": True}
            )

        response = self._client.find_similar(
            url=url,
            num_results=num_results,
            exclude_source_domain=exclude_source_domain,
        )

        results = []
        for r in response.results:
            results.append({
                "title": r.title,
                "url": r.url,
                "score": getattr(r, "score", 0.0),
            })

        return AdapterResult(
            success=True,
            data={
                "results": results,
                "source_url": url,
            }
        )

    async def _search_and_contents(
        self,
        query: str,
        num_results: int = 5,
        type: str = "auto",
        **kwargs,
    ) -> AdapterResult:
        """Combined search and content extraction."""
        if not self._client:
            return AdapterResult(
                success=True,
                data={"results": [], "mock": True}
            )

        response = self._client.search_and_contents(
            query=query,
            num_results=num_results,
            type=type if type != "deep" else "auto",
            text=True,
            highlights={"num_sentences": 3},
        )

        results = []
        for r in response.results:
            results.append({
                "title": r.title,
                "url": r.url,
                "text": getattr(r, "text", ""),
                "highlights": getattr(r, "highlights", []),
                "score": getattr(r, "score", 0.0),
            })

        return AdapterResult(
            success=True,
            data={
                "results": results,
                "count": len(results),
            }
        )

    async def _answer(
        self,
        query: str,
        **kwargs,
    ) -> AdapterResult:
        """
        Ask a question and get an AI-generated answer with citations.

        Args:
            query: The question to answer
        """
        self._stats["searches"] += 1

        if not self._client:
            return AdapterResult(
                success=True,
                data={
                    "answer": f"Mock answer for: {query}",
                    "citations": [],
                    "mock": True,
                }
            )

        try:
            response = self._client.answer(query)
            return AdapterResult(
                success=True,
                data={
                    "answer": getattr(response, "answer", ""),
                    "citations": getattr(response, "citations", []),
                }
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _research(
        self,
        instructions: str,
        output_schema: dict = None,
        **kwargs,
    ) -> AdapterResult:
        """
        Perform complex research with structured output.

        Args:
            instructions: Research instructions
            output_schema: JSON schema for structured output
        """
        self._stats["searches"] += 1

        if not self._client:
            return AdapterResult(
                success=True,
                data={
                    "result": f"Mock research for: {instructions}",
                    "mock": True,
                }
            )

        try:
            # Check if research API is available
            if hasattr(self._client, 'research'):
                params = {"instructions": instructions}
                if output_schema:
                    params["output_schema"] = output_schema
                response = self._client.research.create(**params)
                return AdapterResult(
                    success=True,
                    data={
                        "result": response,
                    }
                )
            else:
                # Fallback to search + answer
                search_result = await self._search(instructions, num_results=5, include_contents=True)
                if search_result.success:
                    answer_result = await self._answer(instructions)
                    return AdapterResult(
                        success=True,
                        data={
                            "result": answer_result.data.get("answer", ""),
                            "sources": search_result.data.get("results", []),
                            "fallback": True,
                        }
                    )
                return search_result
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def health_check(self) -> AdapterResult:
        """Check Exa API health."""
        if not EXA_AVAILABLE:
            return AdapterResult(success=False, error="SDK not installed")

        if not self._client:
            return AdapterResult(
                success=True,
                data={"status": "degraded", "reason": "No API key"}
            )

        try:
            # Quick test search
            result = await self._search("test", num_results=1)
            return AdapterResult(
                success=True,
                data={
                    "status": "healthy",
                    "stats": self._stats,
                }
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def shutdown(self) -> AdapterResult:
        """Cleanup resources."""
        self._client = None
        self._status = AdapterStatus.UNINITIALIZED
        return AdapterResult(success=True, data={"stats": self._stats})


# =============================================================================
# Factory
# =============================================================================

def get_exa_adapter() -> type[ExaAdapter]:
    """Get the Exa adapter class."""
    return ExaAdapter


if __name__ == "__main__":
    async def test():
        adapter = ExaAdapter()
        await adapter.initialize({})
        result = await adapter.execute("search", query="LangGraph StateGraph patterns")
        print(f"Search result: {result}")
        await adapter.shutdown()

    asyncio.run(test())
