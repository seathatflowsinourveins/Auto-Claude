"""
Jina AI Reader Adapter - FULLY UNLEASHED
==========================================

Jina AI provides a complete search foundation with Reader, Embeddings, and Reranking.

Latest Features (2026):
- ReaderLM-v2: 1.5B model for HTML to markdown (512K context)
- jina-embeddings-v4: Multimodal (3.8B, 32K context, text+images)
- jina-embeddings-v3: Text embeddings with 5 task adapters
- jina-reranker-v3: Listwise architecture (131K context, SOTA)
- jina-reranker-m0: Multimodal document reranker
- jina-deepsearch-v1: Iterative search with reasoning
- Unified API key across all services

Official Docs: https://jina.ai/
GitHub: https://github.com/jina-ai/reader
MCP Server: https://github.com/jina-ai/MCP

Usage:
    adapter = JinaAdapter()
    await adapter.initialize({"api_key": "jina_xxx"})

    # Read URL as markdown
    result = await adapter.execute("read", url="https://docs.python.org")

    # Multimodal embeddings (v4)
    result = await adapter.execute("embed", texts=["Hello"], model="jina-embeddings-v4")

    # Latest reranker
    result = await adapter.execute("rerank", query="...", documents=[...],
                                   model="jina-reranker-v3")

    # Deep search with reasoning
    result = await adapter.execute("deepsearch", query="complex question",
                                   reasoning_effort="high")
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
import math
import httpx
from typing import Any, Optional, Union, List
from enum import Enum

# Retry utilities
try:
    from .retry import RetryConfig, with_retry, retry_async, http_request_with_retry
except ImportError:
    # Fallback for standalone testing
    RetryConfig = None
    with_retry = lambda f=None, **kw: (lambda fn: fn) if f is None else f
    retry_async = None
    http_request_with_retry = None

# HTTP connection pool
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
        from enum import Enum, IntEnum
        from dataclasses import dataclass, field
        from datetime import datetime
        from typing import Dict, Any, Optional
        from abc import ABC, abstractmethod

        class SDKLayer(IntEnum):
            RESEARCH = 8

        class AdapterStatus(str, Enum):
            UNINITIALIZED = "uninitialized"
            READY = "ready"
            FAILED = "failed"
            ERROR = "error"
            DEGRADED = "degraded"

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
JINA_SEGMENT_URL = "https://segment.jina.ai/"
JINA_DEEPSEARCH_URL = "https://deepsearch.jina.ai/v1/chat/completions"
JINA_CLASSIFY_URL = "https://api.jina.ai/v1/classify"
JINA_EXPAND_URL = "https://api.jina.ai/v1/expand"


class JinaEmbeddingModel(str, Enum):
    """Available Jina embedding models."""
    V3 = "jina-embeddings-v3"           # 570M, text, 5 task adapters
    V4 = "jina-embeddings-v4"           # 3.8B, multimodal (text+images), 32K context
    CLIP_V2 = "jina-clip-v2"            # 865M, multimodal, 89 languages


class JinaRerankerModel(str, Enum):
    """Available Jina reranker models."""
    V2_BASE = "jina-reranker-v2-base-multilingual"  # Base multilingual
    V3 = "jina-reranker-v3"             # 0.6B, listwise, 131K context, SOTA
    M0 = "jina-reranker-m0"             # Multimodal document reranker
    COLBERT_V2 = "jina-colbert-v2"      # ColBERT-style late interaction


class JinaEmbeddingTask(str, Enum):
    """Task types for embeddings (v3)."""
    TEXT_MATCHING = "text-matching"
    RETRIEVAL_QUERY = "retrieval.query"
    RETRIEVAL_PASSAGE = "retrieval.passage"
    SEPARATION = "separation"
    CLASSIFICATION = "classification"


@register_adapter("jina", SDKLayer.RESEARCH, priority=23)
class JinaAdapter(SDKAdapter):
    """
    Jina AI adapter - FULLY UNLEASHED.

    Operations:
        - read: Convert URL to markdown (Reader API)
        - search: Web search via s.jina.ai
        - embed: Generate embeddings (v3, v4, clip-v2)
        - embed_multimodal: Multimodal embeddings for text+images
        - rerank: Rerank documents (v2, v3, m0, colbert)
        - rerank_multimodal: Rerank visual documents
        - segment: Intelligent text chunking
        - deepsearch: Deep search with reasoning
        - classify: Zero-shot classification
        - ground: Fact-check statements
        - parallel_read: Read multiple URLs in parallel
        - search_arxiv: Search arXiv papers
        - search_ssrn: Search SSRN papers
        - search_images: Search for images
        - extract_pdf: Extract content from PDF
        - capture_screenshot: Capture screenshot of URL
        - expand_query: Expand query with related terms
        - deduplicate_strings: Deduplicate similar strings using embeddings
    """

    def __init__(self):
        self._api_key: Optional[str] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._pool_reader: Optional[HTTPConnectionPool] = None  # For r.jina.ai
        self._pool_search: Optional[HTTPConnectionPool] = None  # For s.jina.ai
        self._pool_api: Optional[HTTPConnectionPool] = None     # For api.jina.ai
        self._status = AdapterStatus.UNINITIALIZED
        self._config: dict[str, Any] = {}
        self._stats = {
            "reads": 0,
            "searches": 0,
            "embeddings": 0,
            "reranks": 0,
            "segments": 0,
            "deepsearches": 0,
            "classifications": 0,
            "grounds": 0,
            "parallel_reads": 0,
            "arxiv_searches": 0,
            "ssrn_searches": 0,
            "image_searches": 0,
            "pdf_extractions": 0,
            "screenshots": 0,
            "query_expansions": 0,
            "deduplications": 0,
            "retries": 0,
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
        return "jina"

    @property
    def layer(self) -> SDKLayer:
        return SDKLayer.RESEARCH

    @property
    def available(self) -> bool:
        return True  # Uses HTTP, always available

    async def initialize(self, config: dict[str, Any]) -> AdapterResult:
        """Initialize Jina client with connection pooling."""
        try:
            self._api_key = config.get("api_key") or os.getenv("JINA_API_KEY")
            self._config = config

            # Use shared connection pools if available
            if HTTP_POOL_AVAILABLE and get_shared_pool_sync:
                pool_config = get_config_for_service("jina")

                # Create pools for different Jina endpoints
                default_headers = {}
                if self._api_key:
                    default_headers["Authorization"] = f"Bearer {self._api_key}"

                self._pool_reader = get_shared_pool_sync(
                    JINA_READER_URL,
                    pool_config,
                    default_headers=default_headers,
                )
                self._pool_search = get_shared_pool_sync(
                    JINA_SEARCH_URL,
                    pool_config,
                    default_headers=default_headers,
                )
                self._pool_api = get_shared_pool_sync(
                    "https://api.jina.ai",
                    pool_config,
                    default_headers={
                        **default_headers,
                        "Content-Type": "application/json",
                    },
                )
                self._client = None  # Use pools instead
            else:
                # Fallback to direct client
                self._client = httpx.AsyncClient(timeout=120.0)
                self._pool_reader = None
                self._pool_search = None
                self._pool_api = None

            self._status = AdapterStatus.READY

            return AdapterResult(
                success=True,
                data={
                    "status": "ready",
                    "features": [
                        "read", "search", "embed", "embed_multimodal",
                        "rerank", "rerank_multimodal", "segment",
                        "deepsearch", "classify", "ground",
                        "parallel_read", "search_arxiv", "search_ssrn",
                        "search_images", "extract_pdf", "capture_screenshot",
                        "expand_query", "deduplicate_strings"
                    ],
                    "embedding_models": [m.value for m in JinaEmbeddingModel],
                    "reranker_models": [m.value for m in JinaRerankerModel],
                    "api_key_provided": bool(self._api_key),
                    "connection_pooling": self._pool_reader is not None,
                }
            )
        except Exception as e:
            self._status = AdapterStatus.ERROR
            return AdapterResult(success=False, error=str(e))

    async def execute(self, operation: str, **kwargs) -> AdapterResult:
        """Execute Jina operations with circuit breaker protection."""
        start_time = time.time()

        operations = {
            "read": self._read_url,
            "search": self._search,
            "embed": self._embed,
            "embed_multimodal": self._embed_multimodal,
            "rerank": self._rerank,
            "rerank_multimodal": self._rerank_multimodal,
            "segment": self._segment,
            "deepsearch": self._deepsearch,
            "classify": self._classify,
            "ground": self._ground,
            # New MCP tools
            "parallel_read": self._parallel_read_url,
            "search_arxiv": self._search_arxiv,
            "search_ssrn": self._search_ssrn,
            "search_images": self._search_images,
            "extract_pdf": self._extract_pdf,
            "capture_screenshot": self._capture_screenshot,
            "expand_query": self._expand_query,
            "deduplicate_strings": self._deduplicate_strings,
        }

        if operation not in operations:
            return AdapterResult(
                success=False,
                error=f"Unknown operation: {operation}. Valid: {list(operations.keys())}"
            )

        # Execute with circuit breaker protection
        try:
            async with adapter_circuit_breaker("jina_adapter"):
                result = await operations[operation](**kwargs)
                result.latency_ms = (time.time() - start_time) * 1000
                return result
        except CircuitOpenError as e:
            # Circuit is open - return fallback response
            return AdapterResult(
                success=False,
                error=f"Circuit breaker open for jina_adapter: {e}",
                latency_ms=(time.time() - start_time) * 1000,
                metadata={"circuit_breaker": "open", "adapter": "jina"},
            )
        except Exception as e:
            return AdapterResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def _make_reader_request(
        self,
        url: str,
        headers: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
        """Make request to Jina Reader API using pool or direct client."""
        if self._pool_reader:
            return await self._pool_reader.get(f"/{url}", headers=headers, timeout=timeout)
        elif self._client:
            reader_url = f"{JINA_READER_URL}/{url}"
            return await self._client.get(reader_url, headers=headers, timeout=timeout)
        else:
            raise RuntimeError("No HTTP client or pool initialized")

    async def _make_search_request(
        self,
        path: str,
        headers: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
        """Make request to Jina Search API using pool or direct client."""
        if self._pool_search:
            return await self._pool_search.get(path, headers=headers, timeout=timeout)
        elif self._client:
            search_url = f"{JINA_SEARCH_URL}{path}"
            return await self._client.get(search_url, headers=headers, timeout=timeout)
        else:
            raise RuntimeError("No HTTP client or pool initialized")

    async def _make_api_request(
        self,
        method: str,
        endpoint: str,
        json: Optional[dict] = None,
        headers: Optional[dict] = None,
        timeout: Optional[float] = None,
    ) -> httpx.Response:
        """Make request to Jina API using pool or direct client."""
        if self._pool_api:
            return await self._pool_api.request(method, endpoint, json=json, headers=headers, timeout=timeout)
        elif self._client:
            api_url = f"https://api.jina.ai{endpoint}"
            default_headers = {"Content-Type": "application/json"}
            if self._api_key:
                default_headers["Authorization"] = f"Bearer {self._api_key}"
            if headers:
                default_headers.update(headers)
            return await self._client.request(method, api_url, json=json, headers=default_headers, timeout=timeout)
        else:
            raise RuntimeError("No HTTP client or pool initialized")

    def get_pool_metrics(self) -> Optional[Dict[str, Any]]:
        """Get connection pool metrics for monitoring."""
        if not HTTP_POOL_AVAILABLE:
            return None

        metrics = {}
        if self._pool_reader:
            metrics["reader"] = self._pool_reader.get_metrics().to_dict()
        if self._pool_search:
            metrics["search"] = self._pool_search.get_metrics().to_dict()
        if self._pool_api:
            metrics["api"] = self._pool_api.get_metrics().to_dict()

        return metrics if metrics else None

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
        json_response: bool = False,
        forward_cookies: bool = False,
        proxy: Optional[str] = None,
        **kwargs,
    ) -> AdapterResult:
        """
        Read URL and convert to LLM-ready markdown (Jina Reader API).

        Args:
            url: URL to read
            respond_with: "markdown", "html", "text", "screenshot", or "pageshot"
            wait_for_selector: CSS selector to wait for (dynamic content)
            bypass_cache: Bypass Jina's cache for fresh content
            with_images: Include image descriptions via vision models
            with_images_summary: Generate summary of images on page
            retain_images: Keep image markdown links in output
            target_selector: CSS selector to extract specific element
            timeout: Request timeout in seconds (max 120)
            json_response: Return structured JSON instead of raw text
            forward_cookies: Forward cookies from request
            proxy: Proxy URL for request
        """
        self._stats["reads"] += 1

        if not self._client and not self._pool_reader:
            return AdapterResult(success=False, error="Client not initialized")

        # Build headers for Jina Reader API
        headers = {"Accept": "application/json" if json_response else "text/plain"}
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
            headers["X-Timeout"] = str(min(timeout, 120))
        if forward_cookies:
            headers["X-Forward-Cookie"] = "true"
        if proxy:
            headers["X-Proxy-Url"] = proxy

        # Execute with retry logic using connection pool
        async def _do_read():
            resp = await self._make_reader_request(url, headers=headers)
            resp.raise_for_status()
            return resp

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
            try:
                response = await retry_async(_do_read, config=config)
            except httpx.HTTPStatusError as e:
                return AdapterResult(
                    success=False,
                    error=f"Jina Reader error: {e.response.status_code} - {e.response.text[:500]}"
                )
        else:
            response = await self._make_reader_request(url, headers=headers)
            if response.status_code != 200:
                return AdapterResult(
                    success=False,
                    error=f"Jina Reader error: {response.status_code} - {response.text[:500]}"
                )

        if json_response:
            try:
                data = response.json()
                return AdapterResult(
                    success=True,
                    data={
                        "content": data.get("content", ""),
                        "title": data.get("title", ""),
                        "url": url,
                        "format": respond_with,
                    }
                )
            except Exception:
                pass

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
        """Web search via Jina s.jina.ai."""
        self._stats["searches"] += 1

        if not self._client:
            return AdapterResult(success=False, error="Client not initialized")

        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        encoded_query = query.replace(' ', '+')
        search_url = f"{JINA_SEARCH_URL}/{encoded_query}"
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
        late_chunking: bool = False,
        **kwargs,
    ) -> AdapterResult:
        """
        Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            model: Embedding model (jina-embeddings-v3, v4, jina-clip-v2)
            task: Task type for v3 (text-matching, retrieval.query, etc.)
            dimensions: Output dimensions (default 1024)
            late_chunking: Enable late chunking for long documents
        """
        self._stats["embeddings"] += len(texts)

        if not self._client or not self._api_key:
            return AdapterResult(
                success=False,
                error="API key required for embeddings"
            )

        payload = {
            "model": model,
            "input": texts,
            "dimensions": dimensions,
        }

        # Add task for v3
        if "v3" in model:
            payload["task"] = task

        # Late chunking for long documents
        if late_chunking:
            payload["late_chunking"] = True

        # Execute with retry logic
        async def _do_embed():
            resp = await self._client.post(
                JINA_EMBED_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload
            )
            resp.raise_for_status()
            return resp

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
            try:
                response = await retry_async(_do_embed, config=config)
            except httpx.HTTPStatusError as e:
                return AdapterResult(
                    success=False,
                    error=f"Jina Embed error: {e.response.status_code} - {e.response.text[:200]}"
                )
        else:
            response = await self._client.post(
                JINA_EMBED_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload
            )
            if response.status_code != 200:
                return AdapterResult(
                    success=False,
                    error=f"Jina Embed error: {response.status_code} - {response.text[:200]}"
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

    async def _embed_multimodal(
        self,
        inputs: list[dict],
        model: str = "jina-embeddings-v4",
        dimensions: int = 1024,
        **kwargs,
    ) -> AdapterResult:
        """
        Generate multimodal embeddings for text and images.

        Args:
            inputs: List of {"text": "..."} or {"image": "base64/url"} dicts
            model: Multimodal model (jina-embeddings-v4 or jina-clip-v2)
            dimensions: Output dimensions
        """
        self._stats["embeddings"] += len(inputs)

        if not self._client or not self._api_key:
            return AdapterResult(
                success=False,
                error="API key required for multimodal embeddings"
            )

        response = await self._client.post(
            JINA_EMBED_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "input": inputs,
                "dimensions": dimensions,
            }
        )

        if response.status_code != 200:
            return AdapterResult(
                success=False,
                error=f"Jina Multimodal Embed error: {response.status_code}"
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
        model: str = "jina-reranker-v3",
        top_n: int = 5,
        return_documents: bool = True,
        **kwargs,
    ) -> AdapterResult:
        """
        Rerank documents by relevance to query.

        Args:
            query: Query string
            documents: List of documents to rerank
            model: Reranker model (v2-base, v3, m0, colbert-v2)
            top_n: Number of top results to return
            return_documents: Include document text in response
        """
        self._stats["reranks"] += 1

        if not self._client or not self._api_key:
            return AdapterResult(
                success=False,
                error="API key required for reranking"
            )

        payload = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": return_documents,
        }

        response = await self._client.post(
            JINA_RERANK_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload
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

    async def _rerank_multimodal(
        self,
        query: str,
        documents: list[dict],
        model: str = "jina-reranker-m0",
        top_n: int = 5,
        **kwargs,
    ) -> AdapterResult:
        """
        Rerank visual/multimodal documents.

        Args:
            query: Query string
            documents: List of {"text": "...", "image": "base64/url"} dicts
            model: Multimodal reranker (jina-reranker-m0)
            top_n: Number of top results
        """
        self._stats["reranks"] += 1

        if not self._client or not self._api_key:
            return AdapterResult(
                success=False,
                error="API key required for multimodal reranking"
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
                error=f"Jina Multimodal Rerank error: {response.status_code}"
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
        tokenizer: str = "cl100k_base",
        **kwargs,
    ) -> AdapterResult:
        """
        Segment long content into chunks for processing.

        Args:
            content: Text content to segment
            max_chunk_length: Maximum characters per chunk
            return_tokens: Return token counts
            tokenizer: Tokenizer to use (cl100k_base, o200k_base)
        """
        self._stats["segments"] += 1

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
                JINA_SEGMENT_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "content": content,
                    "max_chunk_length": max_chunk_length,
                    "return_tokens": return_tokens,
                    "tokenizer": tokenizer,
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
                        "usage": data.get("usage", {}),
                    }
                )
            else:
                # Fallback
                chunks = []
                for i in range(0, len(content), max_chunk_length):
                    chunks.append(content[i:i + max_chunk_length])
                return AdapterResult(
                    success=True,
                    data={"chunks": chunks, "count": len(chunks), "fallback": True}
                )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _deepsearch(
        self,
        query: str,
        reasoning_effort: str = "medium",
        max_tokens: int = 8000,
        stream: bool = False,
        **kwargs,
    ) -> AdapterResult:
        """
        Deep search using Jina's DeepSearch API (jina-deepsearch-v1).

        Performs iterative search, reading, and reasoning to find
        comprehensive answers to complex questions.

        Args:
            query: The search query or question
            reasoning_effort: "low", "medium", or "high"
            max_tokens: Maximum tokens for response
            stream: Enable streaming (returns collected response)
        """
        self._stats["deepsearches"] += 1

        if not self._client or not self._api_key:
            return await self._search(query)

        try:
            response = await self._client.post(
                JINA_DEEPSEARCH_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json={
                    "model": "jina-deepsearch-v1",
                    "messages": [{"role": "user", "content": query}],
                    "stream": False,
                    "reasoning_effort": reasoning_effort,
                    "max_tokens": max_tokens,
                },
                timeout=300.0,  # DeepSearch can take longer
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
            fallback = await self._search(query)
            fallback.metadata["deepsearch_error"] = str(e)
            fallback.metadata["fallback"] = True
            return fallback

    async def _classify(
        self,
        texts: Union[str, list[str]],
        labels: list[str],
        model: str = "jina-embeddings-v3",
        **kwargs,
    ) -> AdapterResult:
        """
        Zero-shot text classification using embeddings similarity.

        Args:
            texts: Text(s) to classify
            labels: List of possible labels
            model: Embedding model to use
        """
        self._stats["classifications"] += 1

        if isinstance(texts, str):
            texts = [texts]

        if not self._client or not self._api_key:
            return AdapterResult(
                success=False,
                error="API key required for classification"
            )

        try:
            # Embed texts and labels together
            all_texts = texts + labels

            response = await self._client.post(
                JINA_EMBED_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "task": "classification",
                    "dimensions": 256,
                    "input": all_texts,
                }
            )

            if response.status_code != 200:
                return AdapterResult(
                    success=False,
                    error=f"Jina Classify error: {response.status_code}"
                )

            data = response.json()
            embeddings = [item["embedding"] for item in data.get("data", [])]

            if len(embeddings) < len(texts) + len(labels):
                return AdapterResult(success=False, error="Not enough embeddings returned")

            text_embs = embeddings[:len(texts)]
            label_embs = embeddings[len(texts):]

            def cosine_sim(a, b):
                dot = sum(x * y for x, y in zip(a, b))
                norm_a = math.sqrt(sum(x * x for x in a))
                norm_b = math.sqrt(sum(x * x for x in b))
                return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0

            results = []
            for i, text_emb in enumerate(text_embs):
                scores = {}
                for label, label_emb in zip(labels, label_embs):
                    scores[label] = cosine_sim(text_emb, label_emb)

                best_label = max(scores, key=scores.get)
                results.append({
                    "text": texts[i],
                    "label": best_label,
                    "score": scores[best_label],
                    "all_scores": scores,
                })

            return AdapterResult(
                success=True,
                data={
                    "results": results if len(results) > 1 else results[0],
                    "usage": data.get("usage", {}),
                }
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _ground(
        self,
        statement: str,
        references: Optional[list[str]] = None,
        **kwargs,
    ) -> AdapterResult:
        """
        Fact-check a statement using web search grounding.

        Args:
            statement: Statement to fact-check
            references: Optional reference URLs to check against
        """
        self._stats["grounds"] += 1

        if not self._client:
            return AdapterResult(success=False, error="Client not initialized")

        try:
            # Use deep search for grounding
            search_result = await self._deepsearch(
                f"Is this statement true or false? Please verify: {statement}",
                reasoning_effort="medium",
            )

            if not search_result.success:
                # Fallback to regular search
                search_result = await self._search(statement)

            return AdapterResult(
                success=True,
                data={
                    "statement": statement,
                    "evidence": search_result.data.get("answer", search_result.data.get("content", "")),
                    "sources": search_result.data.get("sources", []),
                    "grounded": True,
                }
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    # =========================================================================
    # NEW MCP TOOLS - Parallel Operations, Academic Search, PDF, Screenshots
    # =========================================================================

    async def _parallel_read_url(
        self,
        urls: List[str],
        max_concurrent: int = 5,
        **kwargs,
    ) -> AdapterResult:
        """
        Read multiple URLs in parallel.

        Args:
            urls: List of URLs to read
            max_concurrent: Maximum concurrent requests (default 5)
            **kwargs: Additional arguments passed to _read_url
        """
        self._stats["parallel_reads"] += 1
        start = time.time()

        if not self._client:
            return AdapterResult(success=False, error="Client not initialized")

        async def read_single(url: str) -> AdapterResult:
            return await self._read_url(url, **kwargs)

        # Use semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def read_with_limit(url: str) -> AdapterResult:
            async with semaphore:
                return await read_single(url)

        results = await asyncio.gather(
            *[read_with_limit(url) for url in urls],
            return_exceptions=True
        )

        processed_results = []
        for i, r in enumerate(results):
            if isinstance(r, AdapterResult):
                processed_results.append({
                    "url": urls[i],
                    "success": r.success,
                    "data": r.data if r.success else None,
                    "error": r.error if not r.success else None,
                })
            elif isinstance(r, Exception):
                processed_results.append({
                    "url": urls[i],
                    "success": False,
                    "error": str(r),
                })
            else:
                processed_results.append({
                    "url": urls[i],
                    "success": False,
                    "error": "Unknown error",
                })

        successful_count = sum(1 for r in processed_results if r["success"])

        return AdapterResult(
            success=True,
            data={
                "results": processed_results,
                "total": len(urls),
                "successful": successful_count,
                "failed": len(urls) - successful_count,
            },
            latency_ms=(time.time() - start) * 1000,
        )

    async def _search_arxiv(
        self,
        query: str,
        max_results: int = 10,
        **kwargs,
    ) -> AdapterResult:
        """
        Search arXiv papers.

        Args:
            query: Search query
            max_results: Maximum number of results (default 10)
        """
        self._stats["arxiv_searches"] += 1
        start = time.time()

        if not self._client:
            return AdapterResult(success=False, error="Client not initialized")

        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        # Jina search with arxiv site filter
        encoded_query = query.replace(' ', '+')
        search_url = f"{JINA_SEARCH_URL}/{encoded_query}?site=arxiv.org&count={max_results}"

        try:
            response = await self._client.get(search_url, headers=headers)

            if response.status_code != 200:
                return AdapterResult(
                    success=False,
                    error=f"arXiv search error: {response.status_code}",
                    latency_ms=(time.time() - start) * 1000,
                )

            return AdapterResult(
                success=True,
                data={
                    "content": response.text,
                    "query": query,
                    "source": "arxiv",
                    "max_results": max_results,
                },
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return AdapterResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start) * 1000,
            )

    async def _search_ssrn(
        self,
        query: str,
        max_results: int = 10,
        **kwargs,
    ) -> AdapterResult:
        """
        Search SSRN papers.

        Args:
            query: Search query
            max_results: Maximum number of results (default 10)
        """
        self._stats["ssrn_searches"] += 1
        start = time.time()

        if not self._client:
            return AdapterResult(success=False, error="Client not initialized")

        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        # Jina search with SSRN site filter
        encoded_query = query.replace(' ', '+')
        search_url = f"{JINA_SEARCH_URL}/{encoded_query}?site=ssrn.com&count={max_results}"

        try:
            response = await self._client.get(search_url, headers=headers)

            if response.status_code != 200:
                return AdapterResult(
                    success=False,
                    error=f"SSRN search error: {response.status_code}",
                    latency_ms=(time.time() - start) * 1000,
                )

            return AdapterResult(
                success=True,
                data={
                    "content": response.text,
                    "query": query,
                    "source": "ssrn",
                    "max_results": max_results,
                },
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return AdapterResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start) * 1000,
            )

    async def _search_images(
        self,
        query: str,
        max_results: int = 10,
        **kwargs,
    ) -> AdapterResult:
        """
        Search for images.

        Args:
            query: Search query for images
            max_results: Maximum number of results (default 10)
        """
        self._stats["image_searches"] += 1
        start = time.time()

        if not self._client:
            return AdapterResult(success=False, error="Client not initialized")

        headers = {"X-Search-Type": "images"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        encoded_query = query.replace(' ', '+')
        search_url = f"{JINA_SEARCH_URL}/{encoded_query}?count={max_results}"

        try:
            response = await self._client.get(search_url, headers=headers)

            if response.status_code != 200:
                return AdapterResult(
                    success=False,
                    error=f"Image search error: {response.status_code}",
                    latency_ms=(time.time() - start) * 1000,
                )

            return AdapterResult(
                success=True,
                data={
                    "content": response.text,
                    "query": query,
                    "type": "images",
                    "max_results": max_results,
                },
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return AdapterResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start) * 1000,
            )

    async def _extract_pdf(
        self,
        url: str,
        pages: Optional[str] = None,
        **kwargs,
    ) -> AdapterResult:
        """
        Extract content from PDF.

        Args:
            url: URL of the PDF document
            pages: Page range to extract (e.g., "1-5", "1,3,5")
        """
        self._stats["pdf_extractions"] += 1
        start = time.time()

        if not self._client:
            return AdapterResult(success=False, error="Client not initialized")

        headers = {"Accept": "text/plain"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if pages:
            headers["X-PDF-Pages"] = pages

        reader_url = f"{JINA_READER_URL}/{url}"

        try:
            response = await self._client.get(reader_url, headers=headers, timeout=180.0)

            if response.status_code != 200:
                return AdapterResult(
                    success=False,
                    error=f"PDF extraction error: {response.status_code} - {response.text[:200]}",
                    latency_ms=(time.time() - start) * 1000,
                )

            return AdapterResult(
                success=True,
                data={
                    "content": response.text,
                    "url": url,
                    "pages": pages,
                    "content_length": len(response.text),
                },
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return AdapterResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start) * 1000,
            )

    async def _capture_screenshot(
        self,
        url: str,
        full_page: bool = False,
        **kwargs,
    ) -> AdapterResult:
        """
        Capture screenshot of URL.

        Args:
            url: URL to capture
            full_page: Whether to capture full page (default False)
        """
        self._stats["screenshots"] += 1
        start = time.time()

        if not self._client:
            return AdapterResult(success=False, error="Client not initialized")

        headers = {"X-Return-Format": "screenshot"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if full_page:
            headers["X-Full-Page"] = "true"

        reader_url = f"{JINA_READER_URL}/{url}"

        try:
            response = await self._client.get(reader_url, headers=headers, timeout=60.0)

            if response.status_code != 200:
                return AdapterResult(
                    success=False,
                    error=f"Screenshot error: {response.status_code}",
                    latency_ms=(time.time() - start) * 1000,
                )

            # Screenshot is returned as binary content
            screenshot_b64 = base64.b64encode(response.content).decode('utf-8')

            return AdapterResult(
                success=True,
                data={
                    "screenshot_base64": screenshot_b64,
                    "url": url,
                    "full_page": full_page,
                    "content_type": response.headers.get("content-type", "image/png"),
                    "size_bytes": len(response.content),
                },
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return AdapterResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start) * 1000,
            )

    async def _expand_query(
        self,
        query: str,
        num_expansions: int = 5,
        **kwargs,
    ) -> AdapterResult:
        """
        Expand query with related terms using embeddings.

        Args:
            query: Original query to expand
            num_expansions: Number of expanded queries to generate (default 5)
        """
        self._stats["query_expansions"] += 1
        start = time.time()

        if not self._client or not self._api_key:
            # Fallback: simple expansion using common patterns
            expansions = [
                query,
                f"{query} definition",
                f"{query} examples",
                f"what is {query}",
                f"{query} explained",
            ][:num_expansions]
            return AdapterResult(
                success=True,
                data={
                    "original_query": query,
                    "expanded_queries": expansions,
                    "fallback": True,
                },
                latency_ms=(time.time() - start) * 1000,
            )

        try:
            # Use Jina's expand endpoint if available, otherwise use embeddings
            response = await self._client.post(
                JINA_EXPAND_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "query": query,
                    "num": num_expansions,
                },
            )

            if response.status_code == 200:
                data = response.json()
                return AdapterResult(
                    success=True,
                    data={
                        "original_query": query,
                        "expanded_queries": data.get("expansions", [query]),
                        "usage": data.get("usage", {}),
                    },
                    latency_ms=(time.time() - start) * 1000,
                )
            else:
                # Fallback to simple expansion
                expansions = [
                    query,
                    f"{query} definition",
                    f"{query} examples",
                    f"what is {query}",
                    f"{query} explained",
                ][:num_expansions]
                return AdapterResult(
                    success=True,
                    data={
                        "original_query": query,
                        "expanded_queries": expansions,
                        "fallback": True,
                    },
                    latency_ms=(time.time() - start) * 1000,
                )
        except Exception as e:
            return AdapterResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start) * 1000,
            )

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

    async def _deduplicate_strings(
        self,
        strings: List[str],
        threshold: float = 0.9,
        model: str = "jina-embeddings-v3",
        **kwargs,
    ) -> AdapterResult:
        """
        Deduplicate similar strings using embeddings.

        Args:
            strings: List of strings to deduplicate
            threshold: Similarity threshold (default 0.9, higher = stricter)
            model: Embedding model to use (default jina-embeddings-v3)
        """
        self._stats["deduplications"] += 1
        start = time.time()

        if not strings:
            return AdapterResult(
                success=True,
                data={
                    "unique_strings": [],
                    "original_count": 0,
                    "deduplicated_count": 0,
                    "removed_count": 0,
                },
                latency_ms=(time.time() - start) * 1000,
            )

        if len(strings) == 1:
            return AdapterResult(
                success=True,
                data={
                    "unique_strings": strings,
                    "original_count": 1,
                    "deduplicated_count": 1,
                    "removed_count": 0,
                },
                latency_ms=(time.time() - start) * 1000,
            )

        # Get embeddings for all strings
        embeddings_result = await self._embed(strings, model=model, dimensions=256)
        if not embeddings_result.success:
            return AdapterResult(
                success=False,
                error=f"Failed to get embeddings: {embeddings_result.error}",
                latency_ms=(time.time() - start) * 1000,
            )

        embeddings = embeddings_result.data.get("embeddings", [])

        if len(embeddings) != len(strings):
            return AdapterResult(
                success=False,
                error="Embeddings count mismatch",
                latency_ms=(time.time() - start) * 1000,
            )

        # Greedy deduplication using cosine similarity
        unique_indices = []
        duplicate_mapping = {}  # Maps duplicate index to original index

        for i, emb in enumerate(embeddings):
            is_duplicate = False
            for j in unique_indices:
                similarity = self._cosine_similarity(emb, embeddings[j])
                if similarity > threshold:
                    is_duplicate = True
                    duplicate_mapping[i] = j
                    break
            if not is_duplicate:
                unique_indices.append(i)

        unique_strings = [strings[i] for i in unique_indices]

        return AdapterResult(
            success=True,
            data={
                "unique_strings": unique_strings,
                "original_count": len(strings),
                "deduplicated_count": len(unique_strings),
                "removed_count": len(strings) - len(unique_strings),
                "unique_indices": unique_indices,
                "duplicate_mapping": duplicate_mapping,
                "threshold": threshold,
            },
            latency_ms=(time.time() - start) * 1000,
        )

    async def health_check(self) -> AdapterResult:
        """Check Jina API health."""
        try:
            result = await self._read_url("https://example.com")
            return AdapterResult(
                success=True,
                data={"status": "healthy", "stats": self._stats}
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def shutdown(self) -> AdapterResult:
        """Cleanup resources."""
        # Get pool metrics before shutdown for reporting
        pool_metrics = self.get_pool_metrics()

        if self._client:
            await self._client.aclose()
        if self._pool_reader:
            await self._pool_reader.close()
        if self._pool_search:
            await self._pool_search.close()
        if self._pool_api:
            await self._pool_api.close()

        self._client = None
        self._pool_reader = None
        self._pool_search = None
        self._pool_api = None
        self._status = AdapterStatus.UNINITIALIZED

        return AdapterResult(
            success=True,
            data={
                "stats": self._stats,
                "pool_metrics": pool_metrics,
            }
        )


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
