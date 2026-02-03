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
    from .base import SDKAdapter, AdapterResult, AdapterStatus, SDKLayer, register_adapter
except ImportError:
    from base import SDKAdapter, AdapterResult, AdapterStatus, SDKLayer, register_adapter


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
        **kwargs,
    ) -> AdapterResult:
        """
        Standard Sonar chat with web grounding.

        Args:
            query: User query
            system_prompt: Optional system prompt
            return_citations: Include citation URLs
            return_images: Include relevant images
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

        response = await self._client.post(
            "/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar",
                "messages": messages,
                "return_citations": return_citations,
                "return_images": return_images,
            }
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
        Deep research using sonar-deep-research.

        This model:
        - Autonomously searches and reads sources
        - Refines approach as it gathers information
        - Performs multi-step retrieval and synthesis

        Note: Priced at $5/1000 searches
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

        response = await self._client.post(
            "/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "sonar-deep-research",
                "messages": [{"role": "user", "content": query}],
                "return_citations": True,
            }
        )

        if response.status_code != 200:
            return AdapterResult(
                success=False,
                error=f"Perplexity research error: {response.status_code}"
            )

        data = response.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        return AdapterResult(
            success=True,
            data={
                "content": message.get("content", ""),
                "citations": data.get("citations", []),
                "model": "sonar-deep-research",
                "usage": data.get("usage", {}),
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
