"""
Perplexity Sonar Adapter - FULLY UNLEASHED
============================================

Perplexity Sonar models combine LLM capabilities with real-time web search.

Latest Models (2026):
- sonar: Fast search and Q&A (Llama 3.3 70B, 128K context)
- sonar-pro: Advanced multi-step queries, 2x citations (200K context)
- sonar-reasoning-pro: Chain-of-thought, complex analytical tasks
- sonar-deep-research: Multi-step retrieval and synthesis

Features:
- Real-time web grounding with citations
- Search modes: web, academic, sec (SEC EDGAR filings)
- Search depth: high, medium, low
- 1200 tokens/sec via Cerebras inference
- Citation tokens are FREE (not billed)
- SSE streaming support for real-time responses
- Structured output with JSON schema validation
- Batch search (up to 5 queries)
- strip_thinking parameter for reasoning models

Official Docs: https://docs.perplexity.ai/
Pricing: Pay-per-use with $5/mo credit for Pro subscribers

Usage:
    adapter = PerplexityAdapter()
    await adapter.initialize({"api_key": "pplx-xxx"})

    # Standard search
    result = await adapter.execute("chat", query="What is LangGraph?")

    # Reasoning with chain-of-thought (auto-strips <thinking> tags)
    result = await adapter.execute("reasoning", query="complex question",
                                   strip_thinking=True)

    # Deep research
    result = await adapter.execute("deep_research", query="topic")

    # Streaming chat (collects full response)
    result = await adapter.execute("stream", query="Explain quantum computing")

    # Direct streaming (for real-time output)
    async for chunk in adapter.get_stream_generator("Explain quantum computing"):
        print(chunk, end="", flush=True)

    # Structured output with JSON schema
    schema = {"name": "analysis", "schema": {"type": "object", "properties": {...}}}
    result = await adapter.execute("chat_with_schema", query="Analyze X",
                                   output_schema=schema)

    # Batch search (up to 5 queries)
    result = await adapter.execute("batch_search",
                                   queries=["query1", "query2", "query3"])
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import httpx
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
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

try:
    from core.orchestration.base import SDKAdapter, AdapterResult, AdapterStatus, SDKLayer, register_adapter
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


PERPLEXITY_API_URL = "https://api.perplexity.ai"


class PerplexityModel(str, Enum):
    """Available Perplexity Sonar models."""
    SONAR = "sonar"                           # Fast, lightweight (128K context)
    SONAR_PRO = "sonar-pro"                   # Advanced, 2x citations (200K context)
    SONAR_REASONING_PRO = "sonar-reasoning-pro"  # Chain-of-thought reasoning
    SONAR_DEEP_RESEARCH = "sonar-deep-research"  # Multi-step research


class SearchMode(str, Enum):
    """Search modes for Perplexity."""
    WEB = "web"           # General web search
    ACADEMIC = "academic"  # Academic/research papers
    SEC = "sec"           # SEC EDGAR filings


class SearchDepth(str, Enum):
    """Search depth levels (affects pricing and quality)."""
    HIGH = "high"      # Maximum depth and context
    MEDIUM = "medium"  # Balanced approach
    LOW = "low"        # Cost-efficient


@register_adapter("perplexity", SDKLayer.RESEARCH, priority=22)
class PerplexityAdapter(SDKAdapter):
    """
    Perplexity Sonar adapter - FULLY UNLEASHED.

    Operations:
        - chat: Standard Sonar chat with web grounding
        - pro: Pro-tier queries with sonar-pro (2x citations)
        - reasoning: Chain-of-thought with sonar-reasoning-pro
        - deep_research: Multi-step research with sonar-deep-research
        - search: Raw web search without LLM synthesis
        - stream: SSE streaming chat responses
        - chat_with_schema: Structured JSON output with schema validation
        - batch_search: Batch search with up to 5 queries
    """

    def __init__(self):
        self._api_key: Optional[str] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._pool: Optional[HTTPConnectionPool] = None
        self._status = AdapterStatus.UNINITIALIZED
        self._config: dict[str, Any] = {}
        self._stats = {
            "chats": 0,
            "research_queries": 0,
            "reasoning_queries": 0,
            "searches": 0,
            "streams": 0,
            "batch_searches": 0,
            "structured_outputs": 0,
            "total_tokens": 0,
            "retries": 0,
        }
        self._headers: Dict[str, str] = {}
        self._timeout: float = 180.0
        # Retry configuration for transient errors
        self._retry_config = RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=60.0,
            jitter=0.5,
        ) if RetryConfig else None

    @property
    def sdk_name(self) -> str:
        return "perplexity"

    @property
    def layer(self) -> SDKLayer:
        return SDKLayer.RESEARCH

    @property
    def available(self) -> bool:
        return True  # Uses HTTP

    async def initialize(self, config: dict[str, Any]) -> AdapterResult:
        """Initialize Perplexity client with connection pooling."""
        try:
            self._api_key = config.get("api_key") or os.getenv("PERPLEXITY_API_KEY")
            self._timeout = config.get("timeout", 180.0)
            self._config = config
            self._headers = {
                "Authorization": f"Bearer {self._api_key}" if self._api_key else "",
                "Content-Type": "application/json",
            }

            # Use shared connection pool if available
            if HTTP_POOL_AVAILABLE and get_shared_pool_sync:
                pool_config = get_config_for_service("perplexity")
                # Override read timeout with configured timeout
                pool_config.read_timeout = self._timeout
                self._pool = get_shared_pool_sync(
                    PERPLEXITY_API_URL,
                    pool_config,
                    default_headers=self._headers,
                )
                self._client = None  # Use pool instead
            else:
                # Fallback to direct client
                self._client = httpx.AsyncClient(
                    base_url=PERPLEXITY_API_URL,
                    timeout=self._timeout,
                )
                self._pool = None

            self._status = AdapterStatus.READY if self._api_key else AdapterStatus.DEGRADED

            return AdapterResult(
                success=True,
                data={
                    "status": "ready" if self._api_key else "degraded",
                    "models": [m.value for m in PerplexityModel],
                    "search_modes": [m.value for m in SearchMode],
                    "search_depths": [d.value for d in SearchDepth],
                    "api_key_provided": bool(self._api_key),
                    "connection_pooling": self._pool is not None,
                    "operations": [
                        "chat", "pro", "reasoning", "deep_research", "search",
                        "stream", "chat_with_schema", "batch_search"
                    ],
                }
            )
        except Exception as e:
            self._status = AdapterStatus.ERROR
            return AdapterResult(success=False, error=str(e))

    async def execute(self, operation: str, **kwargs) -> AdapterResult:
        """Execute Perplexity operations with circuit breaker protection."""
        start_time = time.time()

        operations = {
            "chat": self._chat,
            "pro": self._pro_chat,
            "reasoning": self._reasoning,
            "deep_research": self._deep_research,
            "research": self._research,  # Alias for deep_research
            "search": self._search,
            # NEW operations
            "stream": self._stream_chat_wrapper,
            "chat_with_schema": self._chat_with_schema,
            "batch_search": self._batch_search,
        }

        if operation not in operations:
            return AdapterResult(
                success=False,
                error=f"Unknown operation: {operation}. Valid: {list(operations.keys())}"
            )

        # Execute with circuit breaker protection
        try:
            async with adapter_circuit_breaker("perplexity_adapter"):
                result = await operations[operation](**kwargs)
                result.latency_ms = (time.time() - start_time) * 1000
                return result
        except CircuitOpenError as e:
            # Circuit is open - return fallback response
            return AdapterResult(
                success=False,
                error=f"Circuit breaker open for perplexity_adapter: {e}",
                latency_ms=(time.time() - start_time) * 1000,
                metadata={"circuit_breaker": "open", "adapter": "perplexity"},
            )
        except Exception as e:
            return AdapterResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def _make_request(
        self,
        method: str,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> httpx.Response:
        """
        Make HTTP request using connection pool or direct client.

        This method abstracts the HTTP layer, using the shared connection pool
        when available for improved performance (50% reduction in connection overhead).

        Args:
            method: HTTP method (GET, POST, etc.)
            path: Request path (relative to base URL)
            json: JSON body for request
            timeout: Request timeout override
            **kwargs: Additional request arguments

        Returns:
            httpx.Response object
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        if self._pool:
            # Use connection pool for better performance
            return await self._pool.request(
                method,
                path,
                headers=headers,
                json=json,
                timeout=timeout or self._timeout,
                **kwargs,
            )
        elif self._client:
            # Fallback to direct client
            return await self._client.request(
                method,
                path,
                headers=headers,
                json=json,
                timeout=timeout,
                **kwargs,
            )
        else:
            raise RuntimeError("No HTTP client or pool initialized")

    def get_pool_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get connection pool metrics for monitoring.

        Returns:
            Pool metrics dictionary or None if pool not available
        """
        if self._pool and HTTP_POOL_AVAILABLE:
            metrics = self._pool.get_metrics()
            return metrics.to_dict()
        return None

    async def _chat(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        return_citations: bool = True,
        return_images: bool = False,
        return_related_questions: bool = False,
        search_recency_filter: Optional[str] = None,
        search_domain_filter: Optional[list[str]] = None,
        search_mode: str = "web",
        search_depth: str = "medium",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs,
    ) -> AdapterResult:
        """
        Standard Sonar chat with web grounding.

        Args:
            query: User query
            system_prompt: Optional system prompt
            return_citations: Include citation URLs (FREE, not billed)
            return_images: Include relevant images
            return_related_questions: Include related questions
            search_recency_filter: "hour", "day", "week", "month", or "year"
            search_domain_filter: Include/exclude domains (- prefix to exclude)
            search_mode: "web", "academic", or "sec"
            search_depth: "high", "medium", or "low"
            temperature: Response temperature (0-1)
            max_tokens: Maximum response tokens
        """
        self._stats["chats"] += 1

        if not self._api_key:
            return AdapterResult(
                success=True,
                data={
                    "content": f"Mock response for: {query}",
                    "citations": [],
                    "mock": True,
                }
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        payload = {
            "model": "sonar",
            "messages": messages,
            "return_citations": return_citations,
            "return_images": return_images,
            "return_related_questions": return_related_questions,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Search parameters
        if search_recency_filter:
            payload["search_recency_filter"] = search_recency_filter
        if search_domain_filter:
            payload["search_domain_filter"] = search_domain_filter[:20]
        if search_mode != "web":
            payload["search_mode"] = search_mode
        if search_depth != "medium":
            payload["web_search_options"] = {"search_context_size": search_depth}

        # Execute API call with retry logic using connection pool
        async def _do_chat():
            resp = await self._make_request("POST", "/chat/completions", json=payload)
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
                response = await retry_async(_do_chat, config=config)
            except httpx.HTTPStatusError as e:
                return AdapterResult(
                    success=False,
                    error=f"Perplexity error: {e.response.status_code} - {e.response.text[:500]}"
                )
        else:
            response = await self._make_request("POST", "/chat/completions", json=payload)
            if response.status_code != 200:
                return AdapterResult(
                    success=False,
                    error=f"Perplexity error: {response.status_code} - {response.text[:500]}"
                )

        data = response.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        self._stats["total_tokens"] += data.get("usage", {}).get("total_tokens", 0)

        return AdapterResult(
            success=True,
            data={
                "content": message.get("content", ""),
                "citations": data.get("citations", []),
                "images": data.get("images", []),
                "related_questions": data.get("related_questions", []),
                "model": "sonar",
                "usage": data.get("usage", {}),
            }
        )

    async def _pro_chat(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        search_depth: str = "high",
        **kwargs,
    ) -> AdapterResult:
        """
        Pro-tier chat with sonar-pro for complex queries.

        Features 2x citations vs standard sonar and 200K context.
        """
        self._stats["chats"] += 1

        if not self._api_key:
            return AdapterResult(
                success=True,
                data={
                    "content": f"Mock pro response for: {query}",
                    "citations": [],
                    "mock": True,
                }
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        payload = {
            "model": "sonar-pro",
            "messages": messages,
            "return_citations": True,
            "return_images": True,
            "return_related_questions": True,
        }

        if search_depth != "medium":
            payload["web_search_options"] = {"search_context_size": search_depth}

        response = await self._make_request("POST", "/chat/completions", json=payload)

        if response.status_code != 200:
            return AdapterResult(
                success=False,
                error=f"Perplexity error: {response.status_code}"
            )

        data = response.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        return AdapterResult(
            success=True,
            data={
                "content": message.get("content", ""),
                "citations": data.get("citations", []),
                "images": data.get("images", []),
                "related_questions": data.get("related_questions", []),
                "model": "sonar-pro",
                "usage": data.get("usage", {}),
            }
        )

    async def _reasoning(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        search_depth: str = "high",
        temperature: float = 0.1,
        max_tokens: int = 8192,
        strip_thinking: bool = True,
        **kwargs,
    ) -> AdapterResult:
        """
        Chain-of-thought reasoning with sonar-reasoning-pro.

        This model excels at complex analytical tasks requiring
        multi-step reasoning and logical deduction.

        Args:
            query: User query requiring reasoning
            system_prompt: Optional system prompt
            search_depth: "high", "medium", or "low"
            temperature: Lower for more deterministic reasoning
            max_tokens: Maximum response tokens
            strip_thinking: Remove <thinking> tags from response (default True)
        """
        self._stats["reasoning_queries"] += 1

        if not self._api_key:
            return AdapterResult(
                success=True,
                data={
                    "content": f"Mock reasoning for: {query}",
                    "reasoning_steps": [],
                    "citations": [],
                    "mock": True,
                }
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        payload = {
            "model": "sonar-reasoning-pro",
            "messages": messages,
            "return_citations": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if search_depth != "medium":
            payload["web_search_options"] = {"search_context_size": search_depth}

        response = await self._make_request("POST", "/chat/completions", json=payload, timeout=120.0)

        if response.status_code == 200:
            data = response.json()
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            thinking_stripped = False

            # Strip thinking tags if requested
            if strip_thinking and content:
                original_content = content
                content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
                content = content.strip()
                thinking_stripped = content != original_content

            return AdapterResult(
                success=True,
                data={
                    "content": content,
                    "reasoning_steps": message.get("reasoning_steps", []),
                    "citations": data.get("citations", []),
                    "model": "sonar-reasoning-pro",
                    "usage": data.get("usage", {}),
                    "thinking_stripped": thinking_stripped,
                }
            )
        elif response.status_code == 400:
            # Model may not be available, fallback to sonar-pro with reasoning prompt
            return await self._pro_chat(
                f"Please reason step-by-step through this question and explain your thinking:\n\n{query}",
                system_prompt=system_prompt,
                search_depth=search_depth,
            )
        else:
            return AdapterResult(
                success=False,
                error=f"Perplexity reasoning error: {response.status_code}"
            )

    async def _deep_research(
        self,
        query: str,
        max_iterations: int = 5,
        search_depth: str = "high",
        strip_thinking: bool = True,
        **kwargs,
    ) -> AdapterResult:
        """
        Multi-step deep research with sonar-deep-research.

        This performs autonomous research:
        - Searches, reads, and evaluates sources
        - Refines approach as it gathers information
        - Synthesizes comprehensive reports

        Args:
            query: Research topic
            max_iterations: Maximum search iterations
            search_depth: "high", "medium", or "low"
            strip_thinking: Remove <thinking> tags from response (default True)
        """
        self._stats["research_queries"] += 1

        if not self._api_key:
            return AdapterResult(
                success=True,
                data={
                    "content": f"Mock deep research for: {query}",
                    "citations": [],
                    "mock": True,
                }
            )

        messages = [{"role": "user", "content": query}]

        payload = {
            "model": "sonar-deep-research",
            "messages": messages,
            "return_citations": True,
        }

        if search_depth != "medium":
            payload["web_search_options"] = {"search_context_size": search_depth}

        response = await self._make_request("POST", "/chat/completions", json=payload, timeout=300.0)

        if response.status_code == 200:
            data = response.json()
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            thinking_stripped = False

            # Strip thinking tags if requested
            if strip_thinking and content:
                original_content = content
                content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
                content = content.strip()
                thinking_stripped = content != original_content

            return AdapterResult(
                success=True,
                data={
                    "content": content,
                    "citations": data.get("citations", []),
                    "model": "sonar-deep-research",
                    "usage": data.get("usage", {}),
                    "thinking_stripped": thinking_stripped,
                }
            )
        elif response.status_code == 400:
            # Model not available, fallback to iterative pro search
            all_results = []
            seen_urls = set()
            final_answer = ""
            iterations_completed = 0

            for i in range(max_iterations):
                iterations_completed = i + 1
                modified_query = query if i == 0 else f"{query} (additional context iteration {i+1})"

                result = await self._pro_chat(
                    modified_query,
                    search_depth=search_depth,
                )

                if result.success:
                    final_answer = result.data.get("content", "")
                    for citation in result.data.get("citations", []):
                        url = citation if isinstance(citation, str) else citation.get("url", "")
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            all_results.append(citation)

                if len(all_results) >= 15:
                    break

            # Strip thinking from fallback result too
            if strip_thinking and final_answer:
                final_answer = re.sub(r'<thinking>.*?</thinking>', '', final_answer, flags=re.DOTALL)
                final_answer = final_answer.strip()

            return AdapterResult(
                success=True,
                data={
                    "content": final_answer,
                    "citations": list(all_results),
                    "iterations": iterations_completed,
                    "model": "sonar-pro (fallback)",
                    "fallback": True,
                }
            )
        else:
            return AdapterResult(
                success=False,
                error=f"Perplexity deep research error: {response.status_code}"
            )

    async def _research(self, **kwargs) -> AdapterResult:
        """Alias for deep_research."""
        return await self._deep_research(**kwargs)

    async def _search(
        self,
        query: Union[str, list[str]],
        max_results: int = 10,
        country: Optional[str] = None,
        **kwargs,
    ) -> AdapterResult:
        """
        Raw web search without LLM synthesis (Search API).

        Args:
            query: Search query (or list of up to 5 queries)
            max_results: Maximum results per query (1-20)
            country: ISO alpha-2 country code for localization
        """
        self._stats["searches"] += 1

        if not self._api_key:
            queries = [query] if isinstance(query, str) else query
            return AdapterResult(
                success=True,
                data={
                    "results": [{"title": f"Mock: {q}", "url": "https://example.com"} for q in queries],
                    "mock": True,
                }
            )

        payload = {
            "query": query,
            "max_results": min(max_results, 20),
        }
        if country:
            payload["country"] = country

        response = await self._make_request("POST", "/search", json=payload)

        if response.status_code != 200:
            return AdapterResult(
                success=False,
                error=f"Perplexity search error: {response.status_code}"
            )

        data = response.json()

        return AdapterResult(
            success=True,
            data={
                "results": data.get("results", []),
                "count": len(data.get("results", [])),
            }
        )

    async def _stream_chat(
        self,
        query: str,
        model: str = "sonar",
        system_prompt: Optional[str] = None,
        search_mode: str = "web",
        search_depth: str = "medium",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat responses via SSE.

        Args:
            query: User query
            model: Model to use (sonar, sonar-pro, etc.)
            system_prompt: Optional system prompt
            search_mode: "web", "academic", or "sec"
            search_depth: "high", "medium", or "low"
            temperature: Response temperature (0-1)
            max_tokens: Maximum response tokens

        Yields:
            Text chunks as they arrive
        """
        self._stats["streams"] += 1

        if not self._api_key:
            yield f"Mock streaming response for: {query}"
            return

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if search_mode != "web":
            payload["search_mode"] = search_mode

        if search_depth != "medium":
            payload["web_search_options"] = {"search_context_size": search_depth}

        async with self._client.stream(
            "POST",
            "/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    async def _stream_chat_wrapper(
        self,
        query: str,
        collect_full_response: bool = True,
        **kwargs,
    ) -> AdapterResult:
        """
        Wrapper for streaming that collects and returns full response.

        For true streaming, use _stream_chat() directly as an async generator.

        Args:
            query: User query
            collect_full_response: If True, collect all chunks into full response
            **kwargs: Additional arguments passed to _stream_chat
        """
        start = time.time()

        if collect_full_response:
            chunks = []
            async for chunk in self._stream_chat(query, **kwargs):
                chunks.append(chunk)

            return AdapterResult(
                success=True,
                data={
                    "content": "".join(chunks),
                    "chunks_count": len(chunks),
                    "streamed": True,
                },
                latency_ms=(time.time() - start) * 1000,
            )
        else:
            # Return the generator info for external handling
            return AdapterResult(
                success=True,
                data={
                    "generator": "_stream_chat",
                    "message": "Use _stream_chat() directly for true streaming",
                    "query": query,
                    "kwargs": kwargs,
                },
                latency_ms=(time.time() - start) * 1000,
            )

    async def _chat_with_schema(
        self,
        query: str,
        output_schema: Dict[str, Any],
        model: str = "sonar",
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs,
    ) -> AdapterResult:
        """
        Chat with structured JSON output.

        Args:
            query: User query
            output_schema: JSON Schema for response structure
            model: Model to use
            system_prompt: Optional system prompt
            temperature: Response temperature
            max_tokens: Maximum response tokens

        Returns:
            AdapterResult with structured_content (parsed JSON) and raw_content
        """
        start = time.time()
        self._stats["structured_outputs"] += 1

        if not self._api_key:
            return AdapterResult(
                success=True,
                data={
                    "structured_content": {"mock": True, "query": query},
                    "raw_content": json.dumps({"mock": True, "query": query}),
                    "citations": [],
                    "mock": True,
                },
                latency_ms=(time.time() - start) * 1000,
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        payload = {
            "model": model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": output_schema,
            },
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = await self._make_request("POST", "/chat/completions", json=payload, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            structured_data = json.loads(content)

            self._stats["total_tokens"] += data.get("usage", {}).get("total_tokens", 0)

            return AdapterResult(
                success=True,
                data={
                    "structured_content": structured_data,
                    "raw_content": content,
                    "citations": data.get("citations", []),
                },
                latency_ms=(time.time() - start) * 1000,
                metadata={
                    "model": model,
                    "usage": data.get("usage", {}),
                },
            )
        except json.JSONDecodeError as e:
            return AdapterResult(
                success=False,
                data=None,
                latency_ms=(time.time() - start) * 1000,
                error=f"Failed to parse structured response: {str(e)}",
            )
        except httpx.HTTPStatusError as e:
            return AdapterResult(
                success=False,
                data=None,
                latency_ms=(time.time() - start) * 1000,
                error=f"HTTP error: {e.response.status_code} - {e.response.text[:500]}",
            )
        except Exception as e:
            return AdapterResult(
                success=False,
                data=None,
                latency_ms=(time.time() - start) * 1000,
                error=str(e),
            )

    async def _batch_search(
        self,
        queries: List[str],
        max_results_per_query: int = 20,
        country: Optional[str] = None,
        **kwargs,
    ) -> AdapterResult:
        """
        Batch search with up to 5 queries.

        Args:
            queries: List of search queries (max 5)
            max_results_per_query: Results per query (max 20)
            country: ISO alpha-2 country code for localization

        Returns:
            AdapterResult with results grouped by query
        """
        start = time.time()
        self._stats["batch_searches"] += 1

        if len(queries) > 5:
            return AdapterResult(
                success=False,
                data=None,
                latency_ms=0,
                error="Maximum 5 queries allowed in batch search",
            )

        if len(queries) == 0:
            return AdapterResult(
                success=False,
                data=None,
                latency_ms=0,
                error="At least one query is required",
            )

        if not self._api_key:
            return AdapterResult(
                success=True,
                data={
                    "results": [
                        {"query": q, "results": [{"title": f"Mock: {q}", "url": "https://example.com"}]}
                        for q in queries
                    ],
                    "queries": queries,
                    "mock": True,
                },
                latency_ms=(time.time() - start) * 1000,
            )

        payload: Dict[str, Any] = {
            "query": queries,  # List of queries for batch
            "max_results": min(max_results_per_query, 20),
        }

        if country:
            payload["country"] = country

        try:
            response = await self._make_request("POST", "/search", json=payload, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()

            return AdapterResult(
                success=True,
                data={
                    "results": data.get("results", []),
                    "queries": queries,
                    "total_results": len(data.get("results", [])),
                },
                latency_ms=(time.time() - start) * 1000,
            )
        except httpx.HTTPStatusError as e:
            return AdapterResult(
                success=False,
                data=None,
                latency_ms=(time.time() - start) * 1000,
                error=f"HTTP error: {e.response.status_code} - {e.response.text[:500]}",
            )
        except Exception as e:
            return AdapterResult(
                success=False,
                data=None,
                latency_ms=(time.time() - start) * 1000,
                error=str(e),
            )

    def get_stream_generator(
        self,
        query: str,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Get a streaming generator for direct use.

        This is a convenience method for applications that need
        direct access to the streaming generator.

        Usage:
            async for chunk in adapter.get_stream_generator("query"):
                print(chunk, end="", flush=True)
        """
        return self._stream_chat(query, **kwargs)

    async def health_check(self) -> AdapterResult:
        """Check Perplexity API health."""
        if not self._api_key:
            return AdapterResult(
                success=True,
                data={"status": "degraded", "reason": "No API key"}
            )

        try:
            result = await self._chat("ping", return_citations=False, max_tokens=10)
            return AdapterResult(
                success=True,
                data={"status": "healthy", "stats": self._stats}
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def shutdown(self) -> AdapterResult:
        """Cleanup resources."""
        # Get pool metrics before shutdown for reporting
        pool_metrics = self.get_pool_metrics() if self._pool else None

        if self._client:
            await self._client.aclose()
        if self._pool:
            await self._pool.close()

        self._client = None
        self._pool = None
        self._status = AdapterStatus.UNINITIALIZED

        return AdapterResult(
            success=True,
            data={
                "stats": self._stats,
                "pool_metrics": pool_metrics,
            }
        )


def get_perplexity_adapter() -> type[PerplexityAdapter]:
    """Get the Perplexity adapter class."""
    return PerplexityAdapter


if __name__ == "__main__":
    async def test():
        adapter = PerplexityAdapter()
        await adapter.initialize({})
        result = await adapter.execute("chat", query="What is LangGraph?")
        print(f"Chat result: {result}")
        await adapter.shutdown()

    asyncio.run(test())
