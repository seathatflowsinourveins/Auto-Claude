"""
SDK Adapter Base Interface - V36 Architecture

This module defines the standard interface for all SDK adapters in the UNLEASH platform.
All adapters must implement this interface to ensure consistent behavior and compatibility
with the orchestration layer.

Architecture Decision: ADR-001 - Unified Adapter Interface
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Dict, Generic, Optional, TypeVar

T = TypeVar('T')


class SDKLayer(IntEnum):
    """SDK Layer classification based on the 8-layer architecture."""
    PROTOCOL = 0        # L0: mcp, anthropic, litellm, openai
    ORCHESTRATION = 1   # L1: langgraph, temporal, claude-flow
    MEMORY = 2          # L2: letta, mem0, graphiti
    STRUCTURED = 3      # L3: instructor, baml, outlines, pydantic-ai
    REASONING = 4       # L4: dspy
    OBSERVABILITY = 5   # L5: langfuse, opik, phoenix, deepeval, ragas
    SAFETY = 6          # L6: guardrails-ai, llm-guard, nemo-guardrails
    PROCESSING = 7      # L7: ast-grep, crawl4ai
    KNOWLEDGE = 8       # L8: graphrag, pyribs, cognee
    RESEARCH = 9        # L9: exa, tavily, jina, perplexity, firecrawl


class AdapterStatus(Enum):
    """Adapter lifecycle status."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class AdapterResult:
    """Standard result container for adapter operations."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_error(self) -> bool:
        """Check if result represents an error."""
        return not self.success or self.error is not None


@dataclass
class AdapterConfig:
    """Configuration for SDK adapters."""
    name: str
    layer: SDKLayer
    enabled: bool = True
    timeout_ms: float = 30000.0
    max_retries: int = 3
    cache_ttl_seconds: float = 300.0
    priority: int = 0  # Higher = higher priority
    metadata: Dict[str, Any] = field(default_factory=dict)


class SDKAdapter(ABC, Generic[T]):
    """
    Abstract base class for all SDK adapters.

    All adapters must implement this interface to be compatible with
    the UltimateOrchestrator and SDK Registry.

    Example Implementation:
        class MyAdapter(SDKAdapter):
            @property
            def sdk_name(self) -> str:
                return "my-sdk"

            @property
            def layer(self) -> SDKLayer:
                return SDKLayer.MEMORY

            @property
            def available(self) -> bool:
                return self._check_sdk_available()

            async def initialize(self, config: Dict) -> AdapterResult:
                # Initialize SDK connection
                return AdapterResult(success=True)

            async def execute(self, operation: str, **kwargs) -> AdapterResult:
                # Execute SDK operation
                return AdapterResult(success=True, data={"result": "..."})

            async def health_check(self) -> AdapterResult:
                return AdapterResult(success=True)

            async def shutdown(self) -> AdapterResult:
                return AdapterResult(success=True)
    """

    def __init__(self, config: Optional[AdapterConfig] = None):
        self._config = config or AdapterConfig(
            name=self.sdk_name,
            layer=self.layer
        )
        self._status = AdapterStatus.UNINITIALIZED
        self._call_count: int = 0
        self._total_latency_ms: float = 0.0
        self._error_count: int = 0
        self._last_health_check: Optional[datetime] = None

    @property
    @abstractmethod
    def sdk_name(self) -> str:
        """Return the SDK name identifier."""
        ...

    @property
    @abstractmethod
    def layer(self) -> SDKLayer:
        """Return the SDK layer (0-8)."""
        ...

    @property
    @abstractmethod
    def available(self) -> bool:
        """Check if the SDK is available (installed, configured, etc.)."""
        ...

    @property
    def config(self) -> AdapterConfig:
        """Get adapter configuration."""
        return self._config

    @property
    def status(self) -> AdapterStatus:
        """Get current adapter status."""
        return self._status

    @property
    def metrics(self) -> Dict[str, Any]:
        """Get adapter metrics."""
        return {
            "call_count": self._call_count,
            "total_latency_ms": self._total_latency_ms,
            "avg_latency_ms": self._total_latency_ms / max(1, self._call_count),
            "error_count": self._error_count,
            "error_rate": self._error_count / max(1, self._call_count),
            "status": self._status.value,
            "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None
        }

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> AdapterResult:
        """
        Initialize the adapter with the given configuration.

        Args:
            config: SDK-specific configuration parameters

        Returns:
            AdapterResult indicating success or failure
        """
        ...

    @abstractmethod
    async def execute(self, operation: str, **kwargs) -> AdapterResult:
        """
        Execute an operation using the SDK.

        Args:
            operation: The operation name to execute
            **kwargs: Operation-specific parameters

        Returns:
            AdapterResult with operation results
        """
        ...

    @abstractmethod
    async def health_check(self) -> AdapterResult:
        """
        Perform a health check on the adapter.

        Returns:
            AdapterResult indicating health status
        """
        ...

    @abstractmethod
    async def shutdown(self) -> AdapterResult:
        """
        Gracefully shutdown the adapter.

        Returns:
            AdapterResult indicating shutdown success
        """
        ...

    def _record_call(self, latency_ms: float, success: bool) -> None:
        """Record call metrics."""
        self._call_count += 1
        self._total_latency_ms += latency_ms
        if not success:
            self._error_count += 1

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sdk={self.sdk_name}, layer={self.layer.name}, status={self._status.value})"


# =============================================================================
# Request Deduplication Support for Adapters
# =============================================================================

# Import deduplication utilities
try:
    from ..deduplication import (
        RequestDeduplicator,
        DeduplicationConfig,
        get_deduplicator_sync,
        deduplicated,
        DeduplicatedAdapter,
    )
    DEDUPLICATION_AVAILABLE = True
except ImportError:
    DEDUPLICATION_AVAILABLE = False
    RequestDeduplicator = None
    DeduplicationConfig = None
    get_deduplicator_sync = None
    deduplicated = None
    DeduplicatedAdapter = None


class DeduplicatedSDKAdapter(SDKAdapter[T]):
    """
    SDK Adapter base class with built-in request deduplication support.

    Extends SDKAdapter to automatically deduplicate API calls using
    hash-based fingerprinting and in-flight request tracking.

    Performance Target: Eliminate 30%+ duplicate calls in batch scenarios.

    Example Implementation:
        class MyAdapter(DeduplicatedSDKAdapter):
            @property
            def sdk_name(self) -> str:
                return "my-sdk"

            @property
            def layer(self) -> SDKLayer:
                return SDKLayer.RESEARCH

            @property
            def available(self) -> bool:
                return True

            async def initialize(self, config: Dict) -> AdapterResult:
                # Initialize deduplication
                self.init_deduplication(default_ttl=60.0, enabled=True)
                return AdapterResult(success=True)

            async def execute(self, operation: str, **kwargs) -> AdapterResult:
                # Use deduplication wrapper
                return await self.execute_deduplicated(
                    operation,
                    self._do_execute,
                    **kwargs
                )

            async def _do_execute(self, operation: str, **kwargs) -> AdapterResult:
                # Actual API call implementation
                ...

            async def health_check(self) -> AdapterResult:
                return AdapterResult(success=True)

            async def shutdown(self) -> AdapterResult:
                return AdapterResult(success=True)
    """

    def __init__(self, config: Optional[AdapterConfig] = None):
        super().__init__(config)
        self._dedup_enabled: bool = False
        self._dedup_default_ttl: float = 60.0
        self._dedup_stats: Dict[str, int] = {
            "cache_hits": 0,
            "in_flight_hits": 0,
            "misses": 0,
        }

    def init_deduplication(
        self,
        default_ttl: float = 60.0,
        enabled: bool = True,
    ) -> None:
        """
        Initialize request deduplication for this adapter.

        Args:
            default_ttl: Default TTL for cached responses (seconds)
            enabled: Enable/disable deduplication
        """
        self._dedup_enabled = enabled and DEDUPLICATION_AVAILABLE
        self._dedup_default_ttl = default_ttl

        if self._dedup_enabled:
            from ..deduplication import get_deduplicator_sync
            # Ensure deduplicator is initialized
            get_deduplicator_sync()

    async def execute_deduplicated(
        self,
        operation: str,
        func,
        ttl: Optional[float] = None,
        **kwargs
    ) -> AdapterResult:
        """
        Execute an operation with deduplication.

        Args:
            operation: Operation name
            func: Async function to execute
            ttl: Optional custom TTL
            **kwargs: Operation parameters

        Returns:
            AdapterResult from cache, in-flight, or new execution
        """
        if not self._dedup_enabled:
            return await func(operation, **kwargs)

        from ..deduplication import get_deduplicator_sync

        deduplicator = get_deduplicator_sync()
        effective_ttl = ttl if ttl is not None else self._dedup_default_ttl

        # Check for deduplication
        result = await deduplicator.check(
            operation,
            adapter_name=self.sdk_name,
            **kwargs
        )

        if result.is_cached:
            self._dedup_stats["cache_hits"] += 1
            cached_result = result.value
            if isinstance(cached_result, AdapterResult):
                cached_result.cached = True
            return cached_result

        if result.is_in_flight:
            self._dedup_stats["in_flight_hits"] += 1
            return await result.future

        self._dedup_stats["misses"] += 1

        # Register as in-flight
        await deduplicator.register_in_flight(result.fingerprint)

        try:
            # Execute the actual operation
            response = await func(operation, **kwargs)

            # Cache the result
            await deduplicator.complete_request(
                result.fingerprint,
                response,
                ttl=effective_ttl,
                adapter_name=self.sdk_name,
                operation=operation,
            )

            return response

        except Exception as e:
            await deduplicator.fail_request(result.fingerprint, e)
            raise

    @property
    def deduplication_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics for this adapter."""
        total = sum(self._dedup_stats.values())
        hits = self._dedup_stats["cache_hits"] + self._dedup_stats["in_flight_hits"]

        return {
            **self._dedup_stats,
            "total": total,
            "hit_rate": hits / max(1, total),
            "enabled": self._dedup_enabled,
        }

    @property
    def metrics(self) -> Dict[str, Any]:
        """Get adapter metrics including deduplication stats."""
        base_metrics = super().metrics
        base_metrics["deduplication"] = self.deduplication_stats
        return base_metrics


# =============================================================================
# Adapter Registration Decorator
# =============================================================================

_registered_adapters: Dict[str, type] = {}


def register_adapter(
    name: str,
    layer: SDKLayer,
    priority: int = 0,
):
    """
    Decorator to register an adapter class with the global registry.

    Args:
        name: Unique adapter name
        layer: SDK layer classification
        priority: Priority for ordering (higher = more preferred)

    Example:
        @register_adapter("my-sdk", SDKLayer.MEMORY, priority=10)
        class MyAdapter(SDKAdapter):
            ...
    """
    def decorator(cls):
        _registered_adapters[name] = {
            "class": cls,
            "layer": layer,
            "priority": priority,
        }
        return cls
    return decorator


def get_registered_adapters() -> Dict[str, Dict[str, Any]]:
    """Get all registered adapters."""
    return _registered_adapters.copy()


def get_adapter_class(name: str) -> Optional[type]:
    """Get adapter class by name."""
    entry = _registered_adapters.get(name)
    return entry["class"] if entry else None


# Export public API
__all__ = [
    # Enums
    "SDKLayer",
    "AdapterStatus",
    # Data classes
    "AdapterResult",
    "AdapterConfig",
    # Base classes
    "SDKAdapter",
    "DeduplicatedSDKAdapter",
    # Registration
    "register_adapter",
    "get_registered_adapters",
    "get_adapter_class",
    # Availability flags
    "DEDUPLICATION_AVAILABLE",
]
