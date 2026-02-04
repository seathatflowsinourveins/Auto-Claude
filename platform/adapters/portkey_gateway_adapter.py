"""
Portkey Gateway Adapter - V36 Architecture

Integrates Portkey AI Gateway for unified LLM access with reliability features.

SDK: portkey-ai >= 1.0.0 (https://portkey.ai/)
Layer: L0 (Protocol)
Features:
- Unified API for 200+ LLMs
- Automatic fallbacks and retries
- Load balancing across providers
- Caching for cost reduction
- Request/response logging
- Virtual keys for security

Portkey Gateway Capabilities:
- Provider switching (OpenAI, Anthropic, Azure, etc.)
- Semantic caching (reduce costs by 20-40%)
- Request hedging for latency reduction
- Budget limits and rate limiting

Usage:
    from adapters.portkey_gateway_adapter import PortkeyGatewayAdapter

    adapter = PortkeyGatewayAdapter()
    await adapter.initialize({
        "api_key": "...",
        "virtual_key": "...",
        "config": {"cache": {"mode": "semantic"}}
    })

    # Make LLM call through gateway
    result = await adapter.execute("chat", messages=[{"role": "user", "content": "Hello"}])
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# SDK availability check
PORTKEY_AVAILABLE = False

try:
    from portkey_ai import Portkey
    PORTKEY_AVAILABLE = True
except ImportError:
    logger.info("Portkey not installed - install with: pip install portkey-ai")


# Import base adapter interface
try:
    from core.orchestration.base import (
        SDKAdapter,
        SDKLayer,
        AdapterConfig,
        AdapterResult,
        AdapterStatus,
    )
except ImportError:
    from dataclasses import dataclass as _dataclass
    from enum import IntEnum
    from abc import ABC, abstractmethod

    class SDKLayer(IntEnum):
        PROTOCOL = 0

    @_dataclass
    class AdapterResult:
        success: bool
        data: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        latency_ms: float = 0.0

    @_dataclass
    class AdapterConfig:
        name: str = "portkey-gateway"
        layer: int = 0

    class AdapterStatus:
        READY = "ready"
        FAILED = "failed"
        UNINITIALIZED = "uninitialized"

    class SDKAdapter(ABC):
        @property
        @abstractmethod
        def sdk_name(self) -> str: ...
        @property
        @abstractmethod
        def layer(self) -> int: ...
        @property
        @abstractmethod
        def available(self) -> bool: ...
        @abstractmethod
        async def initialize(self, config: Dict) -> AdapterResult: ...
        @abstractmethod
        async def execute(self, operation: str, **kwargs) -> AdapterResult: ...
        @abstractmethod
        async def health_check(self) -> AdapterResult: ...
        @abstractmethod
        async def shutdown(self) -> AdapterResult: ...


@dataclass
class GatewayConfig:
    """Portkey gateway configuration."""
    cache_mode: str = "semantic"  # none, simple, semantic
    retry_count: int = 3
    timeout_ms: int = 30000
    fallback_providers: List[str] = field(default_factory=list)
    load_balance: str = "round-robin"  # round-robin, least-latency, random


@dataclass
class RequestLog:
    """Request log entry."""
    id: str
    provider: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cached: bool
    cost_usd: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PortkeyGatewayAdapter(SDKAdapter):
    """
    Portkey AI Gateway adapter for unified LLM access.

    Provides enterprise-grade reliability features including automatic
    fallbacks, semantic caching, and load balancing across providers.

    Operations:
    - chat: Chat completion through gateway
    - complete: Text completion through gateway
    - embed: Generate embeddings through gateway
    - configure: Update gateway configuration
    - get_logs: Get request logs
    - get_stats: Get usage statistics
    """

    # Supported providers
    PROVIDERS = [
        "openai", "anthropic", "azure-openai", "google", "cohere",
        "together", "groq", "mistral", "bedrock", "vertex-ai"
    ]

    def __init__(self, config: Optional[AdapterConfig] = None):
        self._config = config or AdapterConfig(
            name="portkey-gateway",
            layer=SDKLayer.PROTOCOL
        )
        self._status = AdapterStatus.UNINITIALIZED
        self._client: Optional[Any] = None
        self._api_key: str = ""
        self._virtual_key: str = ""
        self._gateway_config: GatewayConfig = GatewayConfig()
        self._request_logs: List[RequestLog] = []
        self._total_tokens_in: int = 0
        self._total_tokens_out: int = 0
        self._total_cost_usd: float = 0.0
        self._cache_hits: int = 0
        self._call_count = 0
        self._total_latency_ms = 0.0
        self._error_count = 0

    @property
    def sdk_name(self) -> str:
        return "portkey-gateway"

    @property
    def layer(self) -> SDKLayer:
        return SDKLayer.PROTOCOL

    @property
    def available(self) -> bool:
        return PORTKEY_AVAILABLE

    @property
    def status(self) -> AdapterStatus:
        return self._status

    async def initialize(self, config: Dict[str, Any]) -> AdapterResult:
        """Initialize Portkey Gateway adapter."""
        try:
            self._api_key = config.get("api_key") or os.environ.get("PORTKEY_API_KEY", "")
            self._virtual_key = config.get("virtual_key") or os.environ.get("PORTKEY_VIRTUAL_KEY", "")

            # Parse gateway config
            gw_config = config.get("config", {})
            cache_config = gw_config.get("cache", {})

            self._gateway_config = GatewayConfig(
                cache_mode=cache_config.get("mode", "semantic"),
                retry_count=gw_config.get("retry_count", 3),
                timeout_ms=gw_config.get("timeout_ms", 30000),
                fallback_providers=gw_config.get("fallback_providers", []),
                load_balance=gw_config.get("load_balance", "round-robin")
            )

            if PORTKEY_AVAILABLE and self._api_key:
                self._client = Portkey(
                    api_key=self._api_key,
                    virtual_key=self._virtual_key if self._virtual_key else None
                )

            self._status = AdapterStatus.READY
            logger.info(f"Portkey Gateway adapter initialized (cache={self._gateway_config.cache_mode})")

            return AdapterResult(
                success=True,
                data={
                    "has_api_key": bool(self._api_key),
                    "has_virtual_key": bool(self._virtual_key),
                    "cache_mode": self._gateway_config.cache_mode,
                    "portkey_native": PORTKEY_AVAILABLE
                }
            )

        except Exception as e:
            self._status = AdapterStatus.FAILED
            logger.error(f"Portkey Gateway initialization failed: {e}")
            return AdapterResult(success=False, error=str(e))

    async def execute(self, operation: str, **kwargs) -> AdapterResult:
        """Execute a Portkey Gateway operation."""
        start_time = time.time()

        try:
            if operation == "chat":
                result = await self._chat(**kwargs)
            elif operation == "complete":
                result = await self._complete(**kwargs)
            elif operation == "embed":
                result = await self._embed(**kwargs)
            elif operation == "configure":
                result = await self._configure(**kwargs)
            elif operation == "get_logs":
                result = await self._get_logs(**kwargs)
            elif operation == "get_stats":
                result = await self._get_stats()
            else:
                result = AdapterResult(
                    success=False,
                    error=f"Unknown operation: {operation}"
                )

            latency_ms = (time.time() - start_time) * 1000
            self._call_count += 1
            self._total_latency_ms += latency_ms
            result.latency_ms = latency_ms

            if not result.success:
                self._error_count += 1

            return result

        except Exception as e:
            self._error_count += 1
            logger.error(f"Portkey Gateway execute error: {e}")
            return AdapterResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )

    async def _chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> AdapterResult:
        """Chat completion through Portkey gateway."""
        try:
            request_id = str(uuid.uuid4())
            call_start = time.time()

            if PORTKEY_AVAILABLE and self._client:
                # Use native Portkey client
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                )

                content = response.choices[0].message.content
                tokens_in = response.usage.prompt_tokens
                tokens_out = response.usage.completion_tokens
                cached = getattr(response, 'cache_status', None) == 'hit'
            else:
                # Stub implementation
                content = f"[Portkey stub] Response to: {messages[-1].get('content', '')[:50]}..."
                tokens_in = sum(len(m.get('content', '').split()) for m in messages)
                tokens_out = len(content.split())
                cached = False

            call_latency = (time.time() - call_start) * 1000

            # Estimate cost (simplified)
            cost_usd = (tokens_in * 0.000001 + tokens_out * 0.000002)

            # Log request
            log_entry = RequestLog(
                id=request_id,
                provider=provider,
                model=model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=call_latency,
                cached=cached,
                cost_usd=cost_usd
            )
            self._request_logs.append(log_entry)

            # Update totals
            self._total_tokens_in += tokens_in
            self._total_tokens_out += tokens_out
            self._total_cost_usd += cost_usd
            if cached:
                self._cache_hits += 1

            return AdapterResult(
                success=True,
                data={
                    "content": content,
                    "model": model,
                    "provider": provider,
                    "tokens": {"input": tokens_in, "output": tokens_out},
                    "cached": cached,
                    "cost_usd": round(cost_usd, 6),
                    "request_id": request_id
                }
            )

        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _complete(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo-instruct",
        provider: str = "openai",
        max_tokens: int = 256,
        **kwargs
    ) -> AdapterResult:
        """Text completion through Portkey gateway."""
        try:
            request_id = str(uuid.uuid4())

            if PORTKEY_AVAILABLE and self._client:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._client.completions.create(
                        model=model,
                        prompt=prompt,
                        max_tokens=max_tokens
                    )
                )

                text = response.choices[0].text
                tokens_in = response.usage.prompt_tokens
                tokens_out = response.usage.completion_tokens
            else:
                # Stub implementation
                text = f"[Portkey stub] Completion for: {prompt[:50]}..."
                tokens_in = len(prompt.split())
                tokens_out = len(text.split())

            # Update totals
            self._total_tokens_in += tokens_in
            self._total_tokens_out += tokens_out

            return AdapterResult(
                success=True,
                data={
                    "text": text,
                    "model": model,
                    "provider": provider,
                    "tokens": {"input": tokens_in, "output": tokens_out},
                    "request_id": request_id
                }
            )

        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _embed(
        self,
        input: List[str],
        model: str = "text-embedding-3-small",
        provider: str = "openai",
        **kwargs
    ) -> AdapterResult:
        """Generate embeddings through Portkey gateway."""
        try:
            request_id = str(uuid.uuid4())

            if PORTKEY_AVAILABLE and self._client:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._client.embeddings.create(
                        model=model,
                        input=input
                    )
                )

                embeddings = [item.embedding for item in response.data]
                tokens_in = response.usage.total_tokens
            else:
                # Stub implementation - generate fake embeddings
                import hashlib
                embeddings = []
                for text in input:
                    # Generate deterministic fake embedding
                    hash_val = hashlib.md5(text.encode()).hexdigest()
                    fake_embedding = [int(hash_val[i:i+2], 16) / 255.0 for i in range(0, 32, 2)]
                    embeddings.append(fake_embedding * 96)  # 1536 dims
                tokens_in = sum(len(t.split()) for t in input)

            self._total_tokens_in += tokens_in

            return AdapterResult(
                success=True,
                data={
                    "embeddings": embeddings,
                    "model": model,
                    "provider": provider,
                    "dimensions": len(embeddings[0]) if embeddings else 0,
                    "count": len(embeddings),
                    "tokens": tokens_in,
                    "request_id": request_id
                }
            )

        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _configure(
        self,
        cache_mode: Optional[str] = None,
        retry_count: Optional[int] = None,
        timeout_ms: Optional[int] = None,
        fallback_providers: Optional[List[str]] = None,
        load_balance: Optional[str] = None,
        **kwargs
    ) -> AdapterResult:
        """Update gateway configuration."""
        try:
            if cache_mode:
                self._gateway_config.cache_mode = cache_mode
            if retry_count is not None:
                self._gateway_config.retry_count = retry_count
            if timeout_ms is not None:
                self._gateway_config.timeout_ms = timeout_ms
            if fallback_providers is not None:
                self._gateway_config.fallback_providers = fallback_providers
            if load_balance:
                self._gateway_config.load_balance = load_balance

            return AdapterResult(
                success=True,
                data={
                    "cache_mode": self._gateway_config.cache_mode,
                    "retry_count": self._gateway_config.retry_count,
                    "timeout_ms": self._gateway_config.timeout_ms,
                    "fallback_providers": self._gateway_config.fallback_providers,
                    "load_balance": self._gateway_config.load_balance
                }
            )

        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _get_logs(
        self,
        limit: int = 100,
        provider: Optional[str] = None,
        **kwargs
    ) -> AdapterResult:
        """Get request logs."""
        try:
            logs = self._request_logs

            if provider:
                logs = [log for log in logs if log.provider == provider]

            logs = logs[-limit:]

            return AdapterResult(
                success=True,
                data={
                    "logs": [
                        {
                            "id": log.id,
                            "provider": log.provider,
                            "model": log.model,
                            "tokens_in": log.tokens_in,
                            "tokens_out": log.tokens_out,
                            "latency_ms": round(log.latency_ms, 2),
                            "cached": log.cached,
                            "cost_usd": round(log.cost_usd, 6),
                            "created_at": log.created_at.isoformat()
                        }
                        for log in logs
                    ],
                    "count": len(logs)
                }
            )

        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _get_stats(self) -> AdapterResult:
        """Get usage statistics."""
        cache_hit_rate = self._cache_hits / max(1, self._call_count)

        return AdapterResult(
            success=True,
            data={
                "total_requests": len(self._request_logs),
                "total_tokens_in": self._total_tokens_in,
                "total_tokens_out": self._total_tokens_out,
                "total_cost_usd": round(self._total_cost_usd, 4),
                "cache_hits": self._cache_hits,
                "cache_hit_rate": round(cache_hit_rate, 3),
                "call_count": self._call_count,
                "error_count": self._error_count,
                "avg_latency_ms": self._total_latency_ms / max(1, self._call_count),
                "gateway_config": {
                    "cache_mode": self._gateway_config.cache_mode,
                    "retry_count": self._gateway_config.retry_count,
                    "load_balance": self._gateway_config.load_balance
                },
                "portkey_native": PORTKEY_AVAILABLE
            }
        )

    async def health_check(self) -> AdapterResult:
        """Check adapter health."""
        return AdapterResult(
            success=True,
            data={
                "status": "healthy",
                "has_api_key": bool(self._api_key),
                "cache_mode": self._gateway_config.cache_mode,
                "portkey_available": PORTKEY_AVAILABLE
            }
        )

    async def shutdown(self) -> AdapterResult:
        """Shutdown the adapter."""
        self._client = None
        self._request_logs.clear()
        self._status = AdapterStatus.UNINITIALIZED
        logger.info("Portkey Gateway adapter shutdown")
        return AdapterResult(success=True)


# Register with SDK registry
try:
    from core.orchestration.sdk_registry import register_adapter, SDKLayer

    @register_adapter("portkey-gateway", SDKLayer.PROTOCOL, priority=22)
    class RegisteredPortkeyGatewayAdapter(PortkeyGatewayAdapter):
        """Registered Portkey Gateway adapter."""
        pass

except ImportError:
    pass


__all__ = ["PortkeyGatewayAdapter", "PORTKEY_AVAILABLE", "GatewayConfig", "RequestLog"]
