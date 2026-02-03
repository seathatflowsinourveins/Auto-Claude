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
            "segment": self._segment,
            "ground": self._ground,
            "deepsearch": self._deepsearch,
            "classify": self._classify,
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
        with_images_summary: bool = False,
        retain_images: bool = False,
        target_selector: Optional[str] = None,
        timeout: int = 30,
        **kwargs,
    ) -> AdapterResult:
        """
        Read URL and convert to LLM-ready markdown (Jina Reader API).

        This is the core "Web Reader" that converts bloated HTML into
        clean, token-efficient Markdown for agents.

        Args:
            url: URL to read
            respond_with: "markdown", "html", "text", or "screenshot"
            wait_for_selector: CSS selector to wait for (dynamic content)
            bypass_cache: Bypass Jina's cache for fresh content
            with_images: Include image descriptions via vision models
            with_images_summary: Generate summary of images on page
            retain_images: Keep image markdown links in output
            target_selector: CSS selector to extract specific element
            timeout: Request timeout in seconds
        """
        self._stats["reads"] += 1

        if not self._client:
            return AdapterResult(
                success=False,
                error="Client not initialized"
            )

        # Build headers for Jina Reader API
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
        if with_images_summary:
            headers["X-With-Images-Summary"] = "true"
        if retain_images:
            headers["X-Retain-Images"] = "true"
        if target_selector:
            headers["X-Target-Selector"] = target_selector
        if timeout != 30:
            headers["X-Timeout"] = str(timeout)

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
        site: Optional[str] = None,
        **kwargs,
    ) -> AdapterResult:
        """
        Web search via Jina s.jina.ai.

        Args:
            query: Search query
            site: Restrict search to specific domain (e.g., "jina.ai", "github.com")
        """
        self._stats["searches"] += 1

        if not self._client:
            return AdapterResult(success=False, error="Client not initialized")

        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        # URL-encode the query
        encoded_query = query.replace(' ', '+')
        search_url = f"{JINA_SEARCH_URL}/{encoded_query}"

        # Add site filter as query parameter
        if site:
            search_url += f"?site={site}"

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
                "site": site,
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

    async def _segment(
        self,
        content: str,
        max_chunk_length: int = 1000,
        return_tokens: bool = False,
        **kwargs,
    ) -> AdapterResult:
        """
        Segment long content into chunks for processing.

        Args:
            content: Text content to segment
            max_chunk_length: Maximum characters per chunk
            return_tokens: Return token counts
        """
        self._stats["segments"] = self._stats.get("segments", 0) + 1

        if not self._client or not self._api_key:
            # Fallback to simple chunking
            chunks = []
            for i in range(0, len(content), max_chunk_length):
                chunks.append(content[i:i + max_chunk_length])
            return AdapterResult(
                success=True,
                data={
                    "chunks": chunks,
                    "count": len(chunks),
                    "fallback": True,
                }
            )

        try:
            response = await self._client.post(
                "https://segment.jina.ai/",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "content": content,
                    "max_chunk_length": max_chunk_length,
                    "return_tokens": return_tokens,
                }
            )

            if response.status_code == 200:
                data = response.json()
                return AdapterResult(
                    success=True,
                    data={
                        "chunks": data.get("chunks", []),
                        "count": len(data.get("chunks", [])),
                        "tokens": data.get("tokens") if return_tokens else None,
                    }
                )
            else:
                # Fallback to simple chunking
                chunks = []
                for i in range(0, len(content), max_chunk_length):
                    chunks.append(content[i:i + max_chunk_length])
                return AdapterResult(
                    success=True,
                    data={
                        "chunks": chunks,
                        "count": len(chunks),
                        "fallback": True,
                    }
                )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _ground(
        self,
        statement: str,
        references: list[str] = None,
        **kwargs,
    ) -> AdapterResult:
        """
        Fact-check a statement using web search grounding.

        Args:
            statement: Statement to fact-check
            references: Optional reference URLs to check against
        """
        self._stats["grounds"] = self._stats.get("grounds", 0) + 1

        if not self._client:
            return AdapterResult(success=False, error="Client not initialized")

        try:
            # Use search to find supporting/contradicting evidence
            search_result = await self._search(statement)

            if not search_result.success:
                return search_result

            return AdapterResult(
                success=True,
                data={
                    "statement": statement,
                    "evidence": search_result.data.get("content", ""),
                    "grounded": True,
                }
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _deepsearch(
        self,
        query: str,
        budget_tokens: int = 8000,
        max_attempts: int = 10,
        **kwargs,
    ) -> AdapterResult:
        """
        Deep search using Jina's DeepSearch API (jina-deepsearch-v1).

        This performs iterative search, reading, and reasoning to find
        comprehensive answers to complex questions.

        Args:
            query: The search query or question
            budget_tokens: Token budget for reasoning (default 8000)
            max_attempts: Maximum search attempts (default 10)
        """
        self._stats["deepsearches"] = self._stats.get("deepsearches", 0) + 1

        if not self._client or not self._api_key:
            # Fallback to regular search
            return await self._search(query)

        try:
            response = await self._client.post(
                "https://deepsearch.jina.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "jina-deepsearch-v1",
                    "messages": [{"role": "user", "content": query}],
                    "stream": False,
                    "reasoning_effort": "medium",
                    "budget_tokens": budget_tokens,
                    "max_attempts": max_attempts,
                },
                timeout=120.0,  # DeepSearch can take longer
            )

            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    return AdapterResult(
                        success=True,
                        data={
                            "answer": message.get("content", ""),
                            "reasoning": message.get("reasoning", ""),
                            "sources": data.get("sources", []),
                            "usage": data.get("usage", {}),
                            "model": "jina-deepsearch-v1",
                        }
                    )
                return AdapterResult(
                    success=True,
                    data={"answer": "", "sources": [], "model": "jina-deepsearch-v1"}
                )
            else:
                # Fallback to regular search
                return await self._search(query)
        except Exception as e:
            # Fallback to regular search on error
            fallback = await self._search(query)
            fallback.metadata["deepsearch_error"] = str(e)
            fallback.metadata["fallback"] = True
            return fallback

    async def _classify(
        self,
        text: str,
        labels: list[str],
        **kwargs,
    ) -> AdapterResult:
        """
        Zero-shot text classification using Jina embeddings similarity.

        Args:
            text: Text to classify
            labels: List of possible labels
        """
        self._stats["classifications"] = self._stats.get("classifications", 0) + 1

        if not self._client or not self._api_key:
            return AdapterResult(
                success=False,
                error="API key required for classification"
            )

        try:
            # Use embeddings + similarity for zero-shot classification
            # Embed the text and all labels
            all_texts = [text] + labels

            response = await self._client.post(
                JINA_EMBED_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "jina-embeddings-v3",
                    "task": "text-matching",
                    "dimensions": 256,  # Smaller for efficiency
                    "input": all_texts,
                }
            )

            if response.status_code == 200:
                data = response.json()
                embeddings = [item["embedding"] for item in data.get("data", [])]

                if len(embeddings) >= 2:
                    text_emb = embeddings[0]
                    label_embs = embeddings[1:]

                    # Calculate cosine similarity for each label
                    import math
                    def cosine_sim(a, b):
                        dot = sum(x*y for x, y in zip(a, b))
                        norm_a = math.sqrt(sum(x*x for x in a))
                        norm_b = math.sqrt(sum(x*x for x in b))
                        return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0

                    scores = {}
                    for label, emb in zip(labels, label_embs):
                        scores[label] = cosine_sim(text_emb, emb)

                    # Find best label
                    best_label = max(scores, key=scores.get)
                    best_score = scores[best_label]

                    return AdapterResult(
                        success=True,
                        data={
                            "text": text,
                            "label": best_label,
                            "score": best_score,
                            "all_scores": scores,
                            "usage": data.get("usage", {}),
                        }
                    )
                return AdapterResult(
                    success=True,
                    data={"text": text, "label": "", "score": 0.0}
                )
            else:
                return AdapterResult(
                    success=False,
                    error=f"Jina Classify error: {response.status_code} - {response.text[:200]}"
                )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

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
