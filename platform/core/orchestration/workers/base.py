"""
Worker Base - SDK Adapter Base Class and Worker Interface

V65 Modular Decomposition - Extracted from ultimate_orchestrator.py

Contains:
- WorkerProtocol: Protocol for worker implementations
- SDKAdapterBase: Base class for all SDK adapters
- AdapterFactory: Factory for creating adapters
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    runtime_checkable,
)

from ..domain.value_objects import (
    CircuitState,
    SDKLayer,
    ExecutionPriority,
    SDKConfig,
    ExecutionContext,
    ExecutionResult,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@runtime_checkable
class WorkerProtocol(Protocol):
    """
    Protocol for worker implementations.

    Workers are responsible for executing operations on SDK adapters.
    """

    @property
    def name(self) -> str:
        """Get worker name."""
        ...

    @property
    def layer(self) -> SDKLayer:
        """Get SDK layer this worker serves."""
        ...

    @property
    def is_available(self) -> bool:
        """Check if worker is available for requests."""
        ...

    async def execute(
        self,
        ctx: ExecutionContext,
        operation: str,
        **kwargs: Any
    ) -> ExecutionResult:
        """Execute an operation."""
        ...

    async def initialize(self) -> bool:
        """Initialize the worker."""
        ...

    async def shutdown(self) -> None:
        """Shutdown the worker."""
        ...


class SDKAdapterBase(ABC, Generic[T]):
    """
    Base class for all SDK adapters.

    Provides common functionality:
    - Circuit breaker pattern
    - Caching support
    - Performance metrics
    - Initialization and shutdown lifecycle
    """

    def __init__(self, config: SDKConfig):
        self.config = config
        self._client: Optional[T] = None
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._initialized = False
        self._call_count = 0
        self._total_latency_ms = 0.0
        self._latency_samples: List[float] = []

        # Circuit breaker state
        self._circuit_state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_successes = 0

        # Circuit breaker configuration
        self._failure_threshold = 5
        self._recovery_timeout = 30.0
        self._half_open_max_calls = 3

    @property
    def name(self) -> str:
        """Get adapter name."""
        return self.config.name

    @property
    def layer(self) -> SDKLayer:
        """Get SDK layer."""
        return self.config.layer

    @property
    def avg_latency_ms(self) -> float:
        """Get average latency in milliseconds."""
        if not self._latency_samples:
            return 0.0
        return sum(self._latency_samples) / len(self._latency_samples)

    def is_available(self) -> bool:
        """Check if adapter is available (circuit breaker check)."""
        if self._circuit_state == CircuitState.CLOSED:
            return True

        if self._circuit_state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self._last_failure_time > self._recovery_timeout:
                self._circuit_state = CircuitState.HALF_OPEN
                self._half_open_successes = 0
                logger.info(f"Circuit breaker HALF_OPEN - testing recovery for {self.name}")
                return True
            return False

        # HALF_OPEN state
        return True

    def record_success(self) -> None:
        """Record a successful call for circuit breaker."""
        self._success_count += 1

        if self._circuit_state == CircuitState.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self._half_open_max_calls:
                self._circuit_state = CircuitState.CLOSED
                self._failure_count = 0
                logger.info(f"Circuit breaker CLOSED - {self.name} recovered")
        elif self._circuit_state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call for circuit breaker."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._circuit_state == CircuitState.HALF_OPEN:
            self._circuit_state = CircuitState.OPEN
            logger.warning(f"Circuit breaker OPEN - {self.name} recovery failed")
        elif self._failure_count >= self._failure_threshold:
            self._circuit_state = CircuitState.OPEN
            logger.warning(f"Circuit breaker OPEN - {self.name} exceeded threshold")

    def _record_latency(self, latency_ms: float) -> None:
        """Record latency sample."""
        self._call_count += 1
        self._total_latency_ms += latency_ms
        self._latency_samples.append(latency_ms)
        # Keep only last 100 samples
        if len(self._latency_samples) > 100:
            self._latency_samples = self._latency_samples[-100:]

    def _get_cached(self, cache_key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if cache_key not in self._cache:
            return None

        timestamp = self._cache_timestamps.get(cache_key, 0)
        if time.time() - timestamp > self.config.cache_ttl_seconds:
            del self._cache[cache_key]
            del self._cache_timestamps[cache_key]
            return None

        return self._cache[cache_key]

    def _set_cached(self, cache_key: str, value: Any) -> None:
        """Set cached value with timestamp."""
        self._cache[cache_key] = value
        self._cache_timestamps[cache_key] = time.time()

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the SDK client.

        Returns True if initialization succeeded, False otherwise.
        """
        ...

    @abstractmethod
    async def execute(
        self,
        ctx: ExecutionContext,
        operation: str = "default",
        **kwargs: Any
    ) -> ExecutionResult:
        """
        Execute an operation on the SDK.

        Args:
            ctx: Execution context
            operation: Operation name
            **kwargs: Operation-specific parameters

        Returns:
            ExecutionResult with success/failure status and data
        """
        ...

    async def execute_with_retry(
        self,
        ctx: ExecutionContext,
        operation: str,
        max_retries: Optional[int] = None,
        **kwargs: Any
    ) -> ExecutionResult:
        """
        Execute with automatic retry on failure.

        Uses exponential backoff between retries.
        """
        retries = max_retries or self.config.max_retries
        last_error = None

        for attempt in range(retries):
            result = await self.execute(ctx, operation, **kwargs)
            if result.success:
                return result

            last_error = result.error
            if attempt < retries - 1:
                # Exponential backoff
                wait_time = (2 ** attempt) * 0.1  # 0.1, 0.2, 0.4, 0.8...
                await asyncio.sleep(wait_time)
                logger.debug(f"Retrying {self.name} operation {operation}, attempt {attempt + 2}")

        return ExecutionResult.failure_result(
            error=f"Max retries ({retries}) exceeded. Last error: {last_error}",
            layer=self.layer,
            adapter_name=self.name
        )

    async def shutdown(self) -> None:
        """Shutdown the adapter and cleanup resources."""
        self._cache.clear()
        self._cache_timestamps.clear()
        self._client = None
        self._initialized = False
        logger.debug(f"Adapter {self.name} shutdown complete")

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return {
            "name": self.name,
            "layer": self.layer.name,
            "initialized": self._initialized,
            "circuit_state": self._circuit_state.name,
            "calls": self._call_count,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "cache_size": len(self._cache),
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "config": {
                "timeout_ms": self.config.timeout_ms,
                "cache_ttl_seconds": self.config.cache_ttl_seconds,
                "max_retries": self.config.max_retries,
            }
        }


# Adapter registry for factory pattern
_adapter_registry: Dict[str, Type[SDKAdapterBase]] = {}


def register_adapter(name: str, adapter_class: Type[SDKAdapterBase]) -> None:
    """Register an adapter class."""
    _adapter_registry[name] = adapter_class


def create_adapter(name: str, config: SDKConfig) -> Optional[SDKAdapterBase]:
    """Create an adapter by name."""
    adapter_class = _adapter_registry.get(name)
    if adapter_class is None:
        logger.warning(f"Unknown adapter: {name}")
        return None
    return adapter_class(config)


class AdapterFactory:
    """
    Factory for creating SDK adapters.

    Supports registration of custom adapter classes and lazy initialization.
    """

    def __init__(self):
        self._registry: Dict[str, Type[SDKAdapterBase]] = {}
        self._instances: Dict[str, SDKAdapterBase] = {}

    def register(self, name: str, adapter_class: Type[SDKAdapterBase]) -> None:
        """Register an adapter class."""
        self._registry[name] = adapter_class

    def create(self, config: SDKConfig) -> Optional[SDKAdapterBase]:
        """Create an adapter instance."""
        adapter_class = self._registry.get(config.name)
        if adapter_class is None:
            return None
        return adapter_class(config)

    def get_or_create(self, config: SDKConfig) -> Optional[SDKAdapterBase]:
        """Get existing adapter or create new one."""
        if config.name in self._instances:
            return self._instances[config.name]

        adapter = self.create(config)
        if adapter:
            self._instances[config.name] = adapter
        return adapter

    def get(self, name: str) -> Optional[SDKAdapterBase]:
        """Get an existing adapter instance."""
        return self._instances.get(name)

    def list_registered(self) -> List[str]:
        """List all registered adapter names."""
        return list(self._registry.keys())

    def list_instances(self) -> List[str]:
        """List all created adapter instance names."""
        return list(self._instances.keys())
