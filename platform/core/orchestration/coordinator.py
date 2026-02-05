"""
Coordinator - High-Level Orchestration

V65 Modular Decomposition - Extracted from ultimate_orchestrator.py

This module provides the high-level coordinator that manages:
- SDK adapter lifecycle
- Request routing and execution
- Event publishing
- Performance metrics aggregation

The Coordinator is a facade that simplifies interaction with the
underlying domain, infrastructure, and worker components.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

from .domain.value_objects import (
    SDKLayer,
    ExecutionPriority,
    SDKConfig,
    ExecutionContext,
    ExecutionResult,
)
from .domain.aggregates import ExecutionSession, AdapterAggregate
from .workers.base import SDKAdapterBase, AdapterFactory

logger = logging.getLogger(__name__)


@dataclass
class CoordinatorConfig:
    """Configuration for the Coordinator."""
    session_id: Optional[str] = None
    max_adapters_per_layer: int = 5
    default_timeout_ms: int = 30000
    enable_metrics: bool = True
    enable_events: bool = True
    enable_caching: bool = True


class Coordinator:
    """
    High-Level Orchestration Coordinator.

    Manages SDK adapters, routes requests, and coordinates execution
    across multiple layers. This is the main entry point for orchestration.

    Features:
    - SDK adapter lifecycle management
    - Request routing with priority support
    - Circuit breaker integration
    - Automatic failover between adapters
    - Performance metrics tracking
    - Event publishing for observability
    """

    def __init__(self, config: Optional[CoordinatorConfig] = None):
        self.config = config or CoordinatorConfig()

        # Session management
        self._session = ExecutionSession(
            session_id=self.config.session_id or self._generate_session_id()
        )

        # Adapter management
        self._adapters: Dict[SDKLayer, List[SDKAdapterBase]] = {
            layer: [] for layer in SDKLayer
        }
        self._primary_adapters: Dict[SDKLayer, SDKAdapterBase] = {}
        self._adapter_factory = AdapterFactory()

        # State
        self._initialized = False
        self._shutdown = False

        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}

    @property
    def session_id(self) -> str:
        """Get current session ID."""
        return self._session.session_id

    @property
    def is_initialized(self) -> bool:
        """Check if coordinator is initialized."""
        return self._initialized

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return hashlib.md5(str(time.time()).encode()).hexdigest()[:12]

    def register_adapter_class(
        self,
        name: str,
        adapter_class: Type[SDKAdapterBase]
    ) -> None:
        """Register an adapter class with the factory."""
        self._adapter_factory.register(name, adapter_class)

    async def add_adapter(
        self,
        config: SDKConfig,
        adapter_class: Optional[Type[SDKAdapterBase]] = None
    ) -> bool:
        """
        Add and initialize an SDK adapter.

        Args:
            config: SDK configuration
            adapter_class: Optional adapter class (uses factory if not provided)

        Returns:
            True if adapter was added successfully
        """
        # Create adapter
        if adapter_class:
            adapter = adapter_class(config)
        else:
            adapter = self._adapter_factory.create(config)
            if adapter is None:
                logger.warning(f"Could not create adapter: {config.name}")
                return False

        # Initialize adapter
        try:
            success = await adapter.initialize()
            if not success:
                logger.warning(f"Adapter initialization failed: {config.name}")
                return False
        except Exception as e:
            logger.error(f"Adapter initialization error: {config.name} - {e}")
            return False

        # Add to layer
        layer_adapters = self._adapters[config.layer]
        if len(layer_adapters) >= self.config.max_adapters_per_layer:
            logger.warning(
                f"Max adapters reached for layer {config.layer.name}, "
                f"replacing oldest"
            )
            old_adapter = layer_adapters.pop(0)
            await old_adapter.shutdown()

        layer_adapters.append(adapter)

        # Set as primary if first for this layer
        if config.layer not in self._primary_adapters:
            self._primary_adapters[config.layer] = adapter
            logger.info(f"Primary adapter for {config.layer.name}: {config.name}")

        return True

    async def initialize(
        self,
        adapter_configs: Optional[List[SDKConfig]] = None
    ) -> bool:
        """
        Initialize the coordinator and all configured adapters.

        Args:
            adapter_configs: Optional list of SDK configurations

        Returns:
            True if initialization succeeded
        """
        if self._initialized:
            return True

        if adapter_configs:
            for config in adapter_configs:
                await self.add_adapter(config)

        self._initialized = True
        logger.info(
            f"Coordinator initialized with {len(self._primary_adapters)} layers"
        )
        return True

    async def execute(
        self,
        layer: SDKLayer,
        operation: str,
        priority: ExecutionPriority = ExecutionPriority.NORMAL,
        adapter_preference: Optional[str] = None,
        **kwargs: Any
    ) -> ExecutionResult:
        """
        Execute an operation on the specified layer.

        Features:
        - Circuit breaker check before execution
        - Auto-failover to secondary adapters on failure
        - Performance metrics tracking

        Args:
            layer: SDK layer to execute on
            operation: Operation name
            priority: Execution priority
            adapter_preference: Optional preferred adapter name
            **kwargs: Operation-specific parameters

        Returns:
            ExecutionResult with success/failure status and data
        """
        # Get available adapters for the layer
        adapters = self._adapters.get(layer, [])
        if not adapters:
            return ExecutionResult.failure_result(
                error=f"No adapter available for layer {layer.name}",
                layer=layer
            )

        # Reorder if preference specified
        if adapter_preference:
            preferred = [a for a in adapters if a.config.name == adapter_preference]
            others = [a for a in adapters if a.config.name != adapter_preference]
            if preferred:
                adapters = preferred + others

        # Create execution context
        ctx = self._session.start_execution(
            request_id=f"{self.session_id}:{time.time()}",
            layer=layer,
            operation=operation,
            priority=priority
        )

        # Try adapters in order
        last_error: Optional[str] = None
        for adapter in adapters:
            # Check circuit breaker
            if not adapter.is_available():
                logger.debug(
                    f"Adapter {adapter.config.name} circuit is open, trying next"
                )
                continue

            start_time = time.time()
            try:
                result = await adapter.execute(ctx, operation=operation, **kwargs)
                result.latency_ms = (time.time() - start_time) * 1000
                result.adapter_name = adapter.config.name

                if result.success:
                    adapter.record_success()
                    self._session.complete_execution(ctx, result, operation)
                    self._emit_events()
                    return result
                else:
                    adapter.record_failure()
                    last_error = result.error
                    logger.warning(
                        f"Adapter {adapter.config.name} failed: {last_error}"
                    )

            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                adapter.record_failure()
                last_error = str(e)
                logger.error(
                    f"Adapter {adapter.config.name} exception: {e}"
                )
                result = ExecutionResult.failure_result(
                    error=str(e),
                    layer=layer,
                    adapter_name=adapter.config.name,
                    latency_ms=latency_ms
                )

        # All adapters failed
        final_result = ExecutionResult.failure_result(
            error=f"All adapters for {layer.name} failed. Last error: {last_error}",
            layer=layer
        )
        self._session.complete_execution(ctx, final_result, operation)
        self._emit_events()
        return final_result

    async def execute_with_timeout(
        self,
        layer: SDKLayer,
        operation: str,
        timeout_ms: Optional[int] = None,
        **kwargs: Any
    ) -> ExecutionResult:
        """Execute with explicit timeout."""
        timeout = timeout_ms or self.config.default_timeout_ms
        try:
            result = await asyncio.wait_for(
                self.execute(layer, operation, **kwargs),
                timeout=timeout / 1000.0
            )
            return result
        except asyncio.TimeoutError:
            return ExecutionResult.failure_result(
                error=f"Operation timed out after {timeout}ms",
                layer=layer,
                latency_ms=float(timeout)
            )

    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to events."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    def _emit_events(self) -> None:
        """Emit pending events to handlers."""
        if not self.config.enable_events:
            return

        events = self._session.drain_events()
        for event in events:
            event_type = event.event_type
            handlers = self._event_handlers.get(event_type, [])
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Event handler error: {e}")

    def get_adapter(
        self,
        layer: SDKLayer,
        name: Optional[str] = None
    ) -> Optional[SDKAdapterBase]:
        """Get an adapter by layer and optional name."""
        adapters = self._adapters.get(layer, [])
        if not adapters:
            return None

        if name:
            for adapter in adapters:
                if adapter.config.name == name:
                    return adapter
            return None

        return self._primary_adapters.get(layer)

    def get_available_layers(self) -> List[SDKLayer]:
        """Get list of layers with available adapters."""
        return [
            layer for layer, adapters in self._adapters.items()
            if adapters and any(a.is_available() for a in adapters)
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        layer_stats = {}
        for layer, adapters in self._adapters.items():
            if adapters:
                layer_stats[layer.name] = [
                    adapter.get_stats() for adapter in adapters
                ]

        return {
            "session_id": self.session_id,
            "initialized": self._initialized,
            "total_executions": self._session.total_executions,
            "success_rate": self._session.success_rate,
            "avg_latency_ms": self._session.avg_latency_ms,
            "layers": layer_stats,
            "active_layers": len(self.get_available_layers())
        }

    async def shutdown(self) -> None:
        """Shutdown the coordinator and all adapters."""
        if self._shutdown:
            return

        self._shutdown = True

        for layer_adapters in self._adapters.values():
            for adapter in layer_adapters:
                try:
                    await adapter.shutdown()
                except Exception as e:
                    logger.error(f"Adapter shutdown error: {e}")

        self._adapters = {layer: [] for layer in SDKLayer}
        self._primary_adapters.clear()
        self._initialized = False

        logger.info("Coordinator shutdown complete")


# Convenience functions for simple usage

_default_coordinator: Optional[Coordinator] = None


async def get_coordinator() -> Coordinator:
    """Get or create the default coordinator."""
    global _default_coordinator
    if _default_coordinator is None:
        _default_coordinator = Coordinator()
    return _default_coordinator


async def execute(
    layer: SDKLayer,
    operation: str,
    **kwargs: Any
) -> ExecutionResult:
    """Execute using the default coordinator."""
    coord = await get_coordinator()
    if not coord.is_initialized:
        await coord.initialize()
    return await coord.execute(layer, operation, **kwargs)
