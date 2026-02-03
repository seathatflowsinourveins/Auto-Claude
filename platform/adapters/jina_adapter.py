"""
Jina AI Reader Adapter - URL to Markdown Conversion
====================================================

Jina Reader converts any URL to LLM-friendly markdown with a simple
prefix: r.jina.ai/https://example.com

Features:
- ReaderLM-v2: 1.5B model for HTML to markdown (512K context)
- Supports 29 languages
- PDF support
- Web search via s.jina.ai
- Image captions
- Embeddings and reranking

Official Docs: https://jina.ai/reader/
GitHub: https://github.com/jina-ai/reader
MCP Server: https://github.com/jina-ai/MCP

Usage:
    adapter = JinaAdapter()
    await adapter.initialize({"api_key": "jina_xxx"})

    # Read URL as markdown
    result = await adapter.execute("read", url="https://docs.python.org")

    # Web search
    result = await adapter.execute("search", query="LangGraph patterns")

    # Get embeddings
    result = await adapter.execute("embed", texts=["Hello world"])
"""

from __future__ import annotations

import asyncio
import os
import time
import httpx
from typing import Any, Optional

try:
    from .base import SDKAdapter, AdapterResult, AdapterStatus, SDKLayer, register_adapter
except ImportError:
    from base import SDKAdapter, AdapterResult, AdapterStatus, SDKLayer, register_adapter


# Jina endpoints
JINA_READER_URL = "https://r.jina.ai"
JINA_SEARCH_URL = "https://s.jina.ai"
JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"
JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"


@register_adapter("jina", SDKLayer.RESEARCH, priority=23)
class JinaAdapter(SDKAdapter):
    """
    Jina AI adapter for reading, searching, and embedding.

    Operations:
        - read: Convert URL to markdown
        - search: Web search via s.jina.ai
        - embed: Generate embeddings
        - rerank: Rerank documents
    """

    def __init__(self):
        self._api_key: Optional[str] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._status = AdapterStatus.UNINITIALIZED
        self._config: dict[str, Any] = {}
        self._stats = {
            "reads": 0,
            "searches": 0,
            "embeddings": 0,
            "reranks": 0,
        }

    @property
    def sdk_name(self) -> str:
        return "jina"

    @property
    def layer(self) -> SDKLayer:
        return SDKLayer.RESEARCH

    @property
    def available(self) -> bool:
        return True  # Uses HTTP, always available

    async def initialize(self, config: dict[str, Any]) -> AdapterResult:
        """Initialize Jina client."""
        try:
            self._api_key = config.get("api_key") or os.getenv("JINA_API_KEY")
            self._client = httpx.AsyncClient(timeout=60.0)
            self._config = config
            self._status = AdapterStatus.READY

            return AdapterResult(
                success=True,
                data={
                    "status": "ready",
                    "features": ["read", "search", "embed", "rerank"],
                    "api_key_provided": bool(self._api_key),
                }
            )
        except Exception as e:
            self._status = AdapterStatus.ERROR
            return AdapterResult(success=False, error=str(e))

    async def execute(self, operation: str, **kwargs) -> AdapterResult:
        """Execute Jina operations."""
        start_time = time.time()

        operations = {
            "read": self._read_url,
            "search": self._search,
            "embed": self._embed,
            "rerank": self._rerank,
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

    async def _read_url(
        self,
        url: str,
        respond_with: str = "markdown",
        wait_for_selector: Optional[str] = None,
        bypass_cache: bool = False,
        with_images: bool = False,
        **kwargs,
    ) -> AdapterResult:
        """
        Read URL and convert to markdown.

        Args:
            url: URL to read
            respond_with: "markdown", "html", "text", or "screenshot"
            wait_for_selector: CSS selector to wait for
            bypass_cache: Bypass Jina's cache
            with_images: Include image descriptions
        """
        self._stats["reads"] += 1

        if not self._client:
            return AdapterResult(
                success=False,
                error="Client not initialized"
            )

        # Build headers
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if respond_with != "markdown":
            headers["X-Respond-With"] = respond_with
        if wait_for_selector:
            headers["X-Wait-For-Selector"] = wait_for_selector
        if bypass_cache:
            headers["X-No-Cache"] = "true"
        if with_images:
            headers["X-With-Generated-Alt"] = "true"

        # Make request
        reader_url = f"{JINA_READER_URL}/{url}"
        response = await self._client.get(reader_url, headers=headers)

        if response.status_code != 200:
            return AdapterResult(
                success=False,
                error=f"Jina Reader error: {response.status_code} - {response.text[:500]}"
            )

        return AdapterResult(
            success=True,
            data={
                "content": response.text,
                "url": url,
                "format": respond_with,
                "content_length": len(response.text),
            }
        )

    async def _search(
        self,
        query: str,
        **kwargs,
    ) -> AdapterResult:
        """
        Web search via Jina s.jina.ai.

        Args:
            query: Search query
        """
        self._stats["searches"] += 1

        if not self._client:
            return AdapterResult(success=False, error="Client not initialized")

        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        # URL-encode the query
        search_url = f"{JINA_SEARCH_URL}/{query.replace(' ', '+')}"
        response = await self._client.get(search_url, headers=headers)

        if response.status_code != 200:
            return AdapterResult(
                success=False,
                error=f"Jina Search error: {response.status_code}"
            )

        return AdapterResult(
            success=True,
            data={
                "content": response.text,
                "query": query,
            }
        )

    async def _embed(
        self,
        texts: list[str],
        model: str = "jina-embeddings-v3",
        task: str = "text-matching",
        dimensions: int = 1024,
        **kwargs,
    ) -> AdapterResult:
        """
        Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            model: Embedding model
            task: Task type (text-matching, retrieval.query, etc.)
            dimensions: Output dimensions
        """
        self._stats["embeddings"] += len(texts)

        if not self._client or not self._api_key:
            return AdapterResult(
                success=False,
                error="API key required for embeddings"
            )

        response = await self._client.post(
            JINA_EMBED_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "task": task,
                "dimensions": dimensions,
                "input": texts,
            }
        )

        if response.status_code != 200:
            return AdapterResult(
                success=False,
                error=f"Jina Embed error: {response.status_code}"
            )

        data = response.json()
        embeddings = [item["embedding"] for item in data.get("data", [])]

        return AdapterResult(
            success=True,
            data={
                "embeddings": embeddings,
                "model": model,
                "dimensions": dimensions,
                "usage": data.get("usage", {}),
            }
        )

    async def _rerank(
        self,
        query: str,
        documents: list[str],
        model: str = "jina-reranker-v2-base-multilingual",
        top_n: int = 5,
        **kwargs,
    ) -> AdapterResult:
        """
        Rerank documents by relevance to query.

        Args:
            query: Query string
            documents: List of documents to rerank
            model: Reranker model
            top_n: Number of top results to return
        """
        self._stats["reranks"] += 1

        if not self._client or not self._api_key:
            return AdapterResult(
                success=False,
                error="API key required for reranking"
            )

        response = await self._client.post(
            JINA_RERANK_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
            }
        )

        if response.status_code != 200:
            return AdapterResult(
                success=False,
                error=f"Jina Rerank error: {response.status_code}"
            )

        data = response.json()

        return AdapterResult(
            success=True,
            data={
                "results": data.get("results", []),
                "model": model,
                "usage": data.get("usage", {}),
            }
        )

    async def health_check(self) -> AdapterResult:
        """Check Jina API health."""
        try:
            # Quick test read
            result = await self._read_url("https://example.com")
            return AdapterResult(
                success=True,
                data={"status": "healthy", "stats": self._stats}
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def shutdown(self) -> AdapterResult:
        """Cleanup resources."""
        if self._client:
            await self._client.aclose()
        self._client = None
        self._status = AdapterStatus.UNINITIALIZED
        return AdapterResult(success=True, data={"stats": self._stats})


def get_jina_adapter() -> type[JinaAdapter]:
    """Get the Jina adapter class."""
    return JinaAdapter


if __name__ == "__main__":
    async def test():
        adapter = JinaAdapter()
        await adapter.initialize({})
        result = await adapter.execute("read", url="https://example.com")
        print(f"Read result: {result.data.get('content', '')[:500]}")
        await adapter.shutdown()

    asyncio.run(test())
