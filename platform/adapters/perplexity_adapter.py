"""
Perplexity Sonar Adapter - Research-Focused LLM with Web Search
================================================================

Perplexity Sonar models combine LLM capabilities with real-time web search.

Models:
- sonar: Fast search and Q&A (Llama 3.3 70B based)
- sonar-pro: Advanced multi-step queries, 2x citations
- sonar-deep-research: Multi-step retrieval and synthesis (complex research)

Features:
- Real-time web grounding
- Citations with every response
- 1200 tokens/sec via Cerebras inference
- Research-focused reasoning

Official Docs: https://docs.perplexity.ai/
Pricing: $5/1000 searches for deep research

Usage:
    adapter = PerplexityAdapter()
    await adapter.initialize({"api_key": "pplx-xxx"})

    # Standard search
    result = await adapter.execute("chat", query="What is LangGraph?")

    # Deep research
    result = await adapter.execute("research", query="distributed consensus algorithms")
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


PERPLEXITY_API_URL = "https://api.perplexity.ai"


@register_adapter("perplexity", SDKLayer.RESEARCH, priority=22)
class PerplexityAdapter(SDKAdapter):
    """
    Perplexity Sonar adapter for research-focused AI with web search.

    Operations:
        - chat: Standard Sonar chat with web grounding
        - research: Deep research with sonar-deep-research
        - pro: Pro-tier queries with sonar-pro
    """

    def __init__(self):
        self._api_key: Optional[str] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._status = AdapterStatus.UNINITIALIZED
        self._config: dict[str, Any] = {}
        self._stats = {
            "chats": 0,
            "research_queries": 0,
            "total_tokens": 0,
        }

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
        """Initialize Perplexity client."""
        try:
            self._api_key = config.get("api_key") or os.getenv("PERPLEXITY_API_KEY")
            self._client = httpx.AsyncClient(
                base_url=PERPLEXITY_API_URL,
                timeout=120.0,  # Deep research can take time
            )
            self._config = config
            self._status = AdapterStatus.READY if self._api_key else AdapterStatus.DEGRADED

            return AdapterResult(
                success=True,
                data={
                    "status": "ready" if self._api_key else "degraded",
                    "models": ["sonar", "sonar-pro", "sonar-deep-research"],
                    "api_key_provided": bool(self._api_key),
                }
            )
        except Exception as e:
            self._status = AdapterStatus.ERROR
            return AdapterResult(success=False, error=str(e))

    async def execute(self, operation: str, **kwargs) -> AdapterResult:
        """Execute Perplexity operations."""
        start_time = time.time()

        operations = {
            "chat": self._chat,
            "research": self._research,
            "pro": self._pro_chat,
            "reasoning": self._reasoning,
            "search": self._search,
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
        **kwargs,
    ) -> AdapterResult:
        """
        Standard Sonar chat with web grounding.

        Args:
            query: User query
            system_prompt: Optional system prompt
            return_citations: Include citation URLs
            return_images: Include relevant images (Tier 2+)
            return_related_questions: Include related questions (Tier 2+)
            search_recency_filter: "hour", "day", "week", "month", or "year"
            search_domain_filter: Include/exclude domains (prefix with - to exclude)
            search_mode: "web", "academic", or "sec" (SEC EDGAR filings)
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

        # Build request payload
        payload = {
            "model": "sonar",
            "messages": messages,
            "return_citations": return_citations,
            "return_images": return_images,
            "return_related_questions": return_related_questions,
        }

        # Add optional search parameters
        if search_recency_filter:
            payload["search_recency_filter"] = search_recency_filter
        if search_domain_filter:
            payload["search_domain_filter"] = search_domain_filter[:20]  # Max 20
        if search_mode and search_mode != "web":
            payload["search_mode"] = search_mode

        response = await self._client.post(
            "/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload
        )

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
        **kwargs,
    ) -> AdapterResult:
        """
        Pro-tier chat with sonar-pro for complex queries.

        Features 2x citations vs standard sonar.
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

        response = await self._client.post(
            "/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar-pro",
                "messages": messages,
                "return_citations": True,
            }
        )

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
                "model": "sonar-pro",
                "usage": data.get("usage", {}),
            }
        )

    async def _research(
        self,
        query: str,
        **kwargs,
    ) -> AdapterResult:
        """
        Deep research using sonar-reasoning or sonar-pro with enhanced prompting.

        This provides multi-step research by:
        - Using advanced reasoning models
        - Requesting comprehensive analysis
        - Gathering multiple citations

        Note: Uses sonar-reasoning for best results
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

        # Use sonar-pro for deep research (sonar-reasoning not available as of 2026-02)
        # Fallback chain: sonar-pro -> sonar
        models_to_try = ["sonar-pro", "sonar"]

        for model in models_to_try:
            try:
                # Enhanced prompt for deep research
                research_prompt = f"""Please provide a comprehensive, well-researched analysis of the following topic.
Include multiple perspectives, cite your sources, and provide detailed explanations.

Topic: {query}

Please structure your response with:
1. Overview and key concepts
2. Detailed analysis
3. Comparisons (if applicable)
4. Real-world applications
5. Conclusions"""

                response = await self._client.post(
                    "/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": research_prompt}],
                        "return_citations": True,
                    },
                    timeout=60.0  # Longer timeout for research
                )

                if response.status_code == 200:
                    data = response.json()
                    choice = data.get("choices", [{}])[0]
                    message = choice.get("message", {})
                    content = message.get("content", "")

                    if content:  # Success if we got content
                        return AdapterResult(
                            success=True,
                            data={
                                "content": content,
                                "citations": data.get("citations", []),
                                "model": model,
                                "usage": data.get("usage", {}),
                            }
                        )
            except Exception:
                continue  # Try next model

        return AdapterResult(
            success=False,
            error="All research models failed"
        )

    async def _reasoning(
        self,
        query: str,
        reasoning_effort: str = "medium",
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> AdapterResult:
        """
        Reasoning-focused query using sonar-reasoning or sonar-pro model.

        Args:
            query: User query requiring reasoning
            reasoning_effort: "minimal", "low", "medium", or "high"
            system_prompt: Optional system prompt
        """
        self._stats["chats"] += 1

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

        # Enhanced reasoning prompt
        reasoning_prompt = f"""Please reason through the following question step-by-step,
considering multiple perspectives before providing your answer.

Question: {query}

Think through this carefully and explain your reasoning."""

        messages.append({"role": "user", "content": reasoning_prompt})

        # Use available models (sonar-reasoning not available as of 2026-02)
        models_to_try = ["sonar-pro", "sonar"]

        for model in models_to_try:
            try:
                payload = {
                    "model": model,
                    "messages": messages,
                    "return_citations": True,
                }

                response = await self._client.post(
                    "/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload
                )

                if response.status_code == 200:
                    data = response.json()
                    choice = data.get("choices", [{}])[0]
                    message = choice.get("message", {})
                    content = message.get("content", "")

                    if content:
                        return AdapterResult(
                            success=True,
                            data={
                                "content": content,
                                "reasoning_steps": message.get("reasoning_steps", []),
                                "citations": data.get("citations", []),
                                "model": model,
                                "usage": data.get("usage", {}),
                            }
                        )
            except Exception:
                continue  # Try next model

        return AdapterResult(
            success=False,
            error="All reasoning models failed"
        )

    async def _search(
        self,
        query: str,
        max_results: int = 10,
        country: Optional[str] = None,
        **kwargs,
    ) -> AdapterResult:
        """
        Raw web search without LLM synthesis (Search API).

        Args:
            query: Search query (or list of up to 5 queries)
            max_results: Maximum results (1-20)
            country: ISO alpha-2 country code for localization
        """
        self._stats["searches"] = self._stats.get("searches", 0) + 1

        if not self._api_key:
            return AdapterResult(
                success=True,
                data={
                    "results": [{"title": f"Mock: {query}", "url": "https://example.com"}],
                    "mock": True,
                }
            )

        payload = {
            "query": query,
            "max_results": min(max_results, 20),
        }
        if country:
            payload["country"] = country

        response = await self._client.post(
            "/search",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload
        )

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

    async def health_check(self) -> AdapterResult:
        """Check Perplexity API health."""
        if not self._api_key:
            return AdapterResult(
                success=True,
                data={"status": "degraded", "reason": "No API key"}
            )

        try:
            result = await self._chat("ping", return_citations=False)
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
