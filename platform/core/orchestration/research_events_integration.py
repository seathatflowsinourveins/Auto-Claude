"""
Research Events Integration Example - V40 Architecture

Demonstrates how research adapter results flow through the event system:
1. ResearchQueryRequested - When a query is initiated
2. ResearchResultReceived - When results come back
3. ResearchResultCached - When results are cached

This module provides:
- EventAwareResearchAdapter: Wrapper that adds event emission to any adapter
- ResearchEventHandler: Example handlers for research events
- ResearchEventProjection: Read model built from research events
- Example usage patterns

Usage:
    from core.orchestration.research_events_integration import (
        EventAwareResearchAdapter,
        ResearchEventHandler,
        ResearchEventProjection,
        create_research_event_system,
    )

    # Create event-aware adapter
    bus = EventBus()
    exa = ExaAdapter()
    event_aware_exa = EventAwareResearchAdapter(exa, bus)

    # Execute with automatic event emission
    result = await event_aware_exa.execute("search", query="AI safety")

    # Build projection from events
    projection = ResearchEventProjection()
    await event_store.replay_all(projection.handle_event)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from .domain_events import (
    DomainEvent,
    EventBus,
    EventStore,
    EventAggregator,
    EventMetrics,
    ResearchQueryRequestedEvent,
    ResearchResultReceivedEvent,
    ResearchResultCachedEvent,
    ResearchCacheHitEvent,
    ResearchAdapterHealthEvent,
    ResearchEventEmitter,
)


# =============================================================================
# EVENT-AWARE RESEARCH ADAPTER WRAPPER
# =============================================================================

class EventAwareResearchAdapter:
    """
    Wrapper that adds event emission to any research adapter.

    This wrapper intercepts adapter calls and emits appropriate events:
    - Before query: ResearchQueryRequestedEvent
    - After results: ResearchResultReceivedEvent
    - On cache: ResearchResultCachedEvent
    - On cache hit: ResearchCacheHitEvent

    Example:
        exa = ExaAdapter()
        await exa.initialize({})

        bus = EventBus()
        event_exa = EventAwareResearchAdapter(exa, bus)

        # This automatically emits events
        result = await event_exa.execute("search", query="AI agents")
    """

    def __init__(
        self,
        adapter: Any,
        event_bus: EventBus,
        aggregator: Optional[EventAggregator] = None,
        enable_caching: bool = True,
        cache_ttl_seconds: int = 3600,
    ):
        self._adapter = adapter
        self._emitter = ResearchEventEmitter(event_bus, aggregator)
        self._enable_caching = enable_caching
        self._cache_ttl = cache_ttl_seconds

        # Simple in-memory cache for demonstration
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}

    @property
    def sdk_name(self) -> str:
        return getattr(self._adapter, "sdk_name", "unknown")

    async def initialize(self, config: Dict[str, Any]) -> Any:
        """Initialize the underlying adapter."""
        return await self._adapter.initialize(config)

    async def execute(
        self,
        operation: str,
        use_cache: bool = True,
        **kwargs,
    ) -> Any:
        """
        Execute an operation with automatic event emission.

        Args:
            operation: Operation name (search, research, etc.)
            use_cache: Whether to check cache first
            **kwargs: Operation parameters

        Returns:
            Adapter result
        """
        query = kwargs.get("query") or kwargs.get("input") or ""
        cache_key = self._make_cache_key(operation, query, kwargs)

        # Check cache first
        if use_cache and self._enable_caching:
            cached = self._check_cache(cache_key)
            if cached is not None:
                await self._emitter.emit_cache_hit(
                    query=query,
                    cache_key=cache_key,
                    adapter_name=self.sdk_name,
                    cache_type="local",
                    similarity_score=1.0,
                )
                return cached

        # Emit query requested event
        request_id = await self._emitter.emit_query_requested(
            query=query,
            adapter_name=self.sdk_name,
            operation=operation,
            parameters={k: v for k, v in kwargs.items() if k != "api_key"},
        )

        # Execute the actual operation
        start_time = time.time()
        try:
            result = await self._adapter.execute(operation, **kwargs)
            latency_ms = (time.time() - start_time) * 1000

            # Extract result info
            data = getattr(result, "data", {}) or {}
            result_count = (
                len(data.get("results", []))
                or data.get("count", 0)
                or (1 if data else 0)
            )
            cost_dollars = data.get("cost_dollars")
            success = getattr(result, "success", True)
            error = getattr(result, "error", None)

            # Create preview from first result
            preview = ""
            if data.get("results"):
                first = data["results"][0]
                if isinstance(first, dict):
                    preview = first.get("title", "") or first.get("text", "")[:100]

            # Emit result received event
            await self._emitter.emit_result_received(
                request_id=request_id,
                adapter_name=self.sdk_name,
                result_count=result_count,
                latency_ms=latency_ms,
                success=success,
                error_message=error,
                cost_dollars=cost_dollars,
                result_preview=preview,
            )

            # Cache successful results
            if success and self._enable_caching and result_count > 0:
                self._cache[cache_key] = result
                self._cache_timestamps[cache_key] = time.time()

                await self._emitter.emit_result_cached(
                    request_id=request_id,
                    cache_key=cache_key,
                    adapter_name=self.sdk_name,
                    result_count=result_count,
                    cache_type="local",
                    ttl_seconds=self._cache_ttl,
                )

            return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            await self._emitter.emit_result_received(
                request_id=request_id,
                adapter_name=self.sdk_name,
                result_count=0,
                latency_ms=latency_ms,
                success=False,
                error_message=str(e),
            )

            raise

    async def flush_events(self) -> int:
        """Flush any aggregated events."""
        return await self._emitter.flush()

    async def shutdown(self) -> Any:
        """Shutdown the underlying adapter."""
        await self.flush_events()
        return await self._adapter.shutdown()

    def _make_cache_key(
        self,
        operation: str,
        query: str,
        params: Dict[str, Any],
    ) -> str:
        """Create a cache key from operation and parameters."""
        # Simple hash-based key
        import hashlib
        key_parts = [
            self.sdk_name,
            operation,
            query,
            str(sorted(params.items())),
        ]
        key_str = "|".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def _check_cache(self, cache_key: str) -> Optional[Any]:
        """Check cache for a valid entry."""
        if cache_key not in self._cache:
            return None

        # Check expiration
        cached_at = self._cache_timestamps.get(cache_key, 0)
        if time.time() - cached_at > self._cache_ttl:
            del self._cache[cache_key]
            del self._cache_timestamps[cache_key]
            return None

        return self._cache[cache_key]


# =============================================================================
# RESEARCH EVENT HANDLERS
# =============================================================================

class ResearchEventHandler:
    """
    Example handlers for research events.

    These handlers can be used for:
    - Logging and analytics
    - Building read models
    - Triggering downstream workflows
    - Monitoring and alerting
    """

    def __init__(self, name: str = "research_handler"):
        self.name = name
        self._query_count: int = 0
        self._result_count: int = 0
        self._cache_hits: int = 0
        self._errors: int = 0
        self._total_latency_ms: float = 0.0
        self._total_cost: float = 0.0

    async def handle_query_requested(self, event: DomainEvent) -> None:
        """Handle ResearchQueryRequestedEvent."""
        self._query_count += 1
        print(
            f"[{self.name}] Query requested: {event.payload.get('query', '')[:50]}... "
            f"via {event.payload.get('adapter_name')}"
        )

    async def handle_result_received(self, event: DomainEvent) -> None:
        """Handle ResearchResultReceivedEvent."""
        payload = event.payload
        self._result_count += payload.get("result_count", 0)
        self._total_latency_ms += payload.get("latency_ms", 0)

        if not payload.get("success"):
            self._errors += 1

        if payload.get("cost_dollars"):
            self._total_cost += payload["cost_dollars"]

        print(
            f"[{self.name}] Results received: {payload.get('result_count', 0)} results "
            f"in {payload.get('latency_ms', 0):.0f}ms"
        )

    async def handle_result_cached(self, event: DomainEvent) -> None:
        """Handle ResearchResultCachedEvent."""
        payload = event.payload
        print(
            f"[{self.name}] Results cached: {payload.get('result_count', 0)} results "
            f"with TTL {payload.get('ttl_seconds')}s"
        )

    async def handle_cache_hit(self, event: DomainEvent) -> None:
        """Handle ResearchCacheHitEvent."""
        self._cache_hits += 1
        payload = event.payload
        print(
            f"[{self.name}] Cache hit: {payload.get('cache_type')} "
            f"(similarity: {payload.get('similarity_score', 1.0):.2f})"
        )

    async def handle_event(self, event: DomainEvent) -> None:
        """Generic handler that routes to specific handlers."""
        handlers = {
            "ResearchQueryRequestedEvent": self.handle_query_requested,
            "ResearchResultReceivedEvent": self.handle_result_received,
            "ResearchResultCachedEvent": self.handle_result_cached,
            "ResearchCacheHitEvent": self.handle_cache_hit,
        }

        handler = handlers.get(event.event_type)
        if handler:
            await handler(event)

    def get_statistics(self) -> Dict[str, Any]:
        """Get handler statistics."""
        return {
            "query_count": self._query_count,
            "result_count": self._result_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": round(
                self._cache_hits / max(1, self._query_count) * 100, 2
            ),
            "errors": self._errors,
            "error_rate": round(
                self._errors / max(1, self._query_count) * 100, 2
            ),
            "avg_latency_ms": round(
                self._total_latency_ms / max(1, self._query_count - self._cache_hits), 2
            ),
            "total_cost": round(self._total_cost, 4),
        }


# =============================================================================
# RESEARCH EVENT PROJECTION
# =============================================================================

@dataclass
class ResearchQueryRecord:
    """Record of a research query for the projection."""
    request_id: str
    query: str
    adapter_name: str
    operation: str
    timestamp: datetime
    result_count: int = 0
    latency_ms: float = 0.0
    success: bool = True
    cached: bool = False
    cost_dollars: Optional[float] = None


class ResearchEventProjection:
    """
    Read model projection built from research events.

    This projection maintains:
    - Recent query history
    - Per-adapter statistics
    - Cache effectiveness metrics
    - Cost tracking

    Can be rebuilt by replaying events from the event store.
    """

    def __init__(self, max_history: int = 1000):
        self._max_history = max_history
        self._queries: Dict[str, ResearchQueryRecord] = {}
        self._query_history: List[str] = []

        # Per-adapter stats
        self._adapter_stats: Dict[str, Dict[str, Any]] = {}

        # Global stats
        self._total_queries: int = 0
        self._total_results: int = 0
        self._total_cache_hits: int = 0
        self._total_cost: float = 0.0

    async def handle_event(self, event: DomainEvent) -> None:
        """Handle any research event to update projection."""
        if event.event_type == "ResearchQueryRequestedEvent":
            await self._handle_query_requested(event)
        elif event.event_type == "ResearchResultReceivedEvent":
            await self._handle_result_received(event)
        elif event.event_type == "ResearchCacheHitEvent":
            await self._handle_cache_hit(event)

    async def _handle_query_requested(self, event: DomainEvent) -> None:
        """Process query requested event."""
        payload = event.payload
        request_id = payload.get("request_id", "")

        record = ResearchQueryRecord(
            request_id=request_id,
            query=payload.get("query", ""),
            adapter_name=payload.get("adapter_name", ""),
            operation=payload.get("operation", "search"),
            timestamp=event.timestamp,
        )

        self._queries[request_id] = record
        self._query_history.append(request_id)
        self._total_queries += 1

        # Trim history
        if len(self._query_history) > self._max_history:
            old_id = self._query_history.pop(0)
            self._queries.pop(old_id, None)

        # Update adapter stats
        adapter = payload.get("adapter_name", "unknown")
        if adapter not in self._adapter_stats:
            self._adapter_stats[adapter] = {
                "queries": 0,
                "results": 0,
                "errors": 0,
                "cache_hits": 0,
                "total_latency_ms": 0.0,
                "total_cost": 0.0,
            }
        self._adapter_stats[adapter]["queries"] += 1

    async def _handle_result_received(self, event: DomainEvent) -> None:
        """Process result received event."""
        payload = event.payload
        request_id = payload.get("request_id", "")

        if request_id in self._queries:
            record = self._queries[request_id]
            record.result_count = payload.get("result_count", 0)
            record.latency_ms = payload.get("latency_ms", 0)
            record.success = payload.get("success", True)
            record.cost_dollars = payload.get("cost_dollars")

            self._total_results += record.result_count
            if record.cost_dollars:
                self._total_cost += record.cost_dollars

            # Update adapter stats
            adapter = payload.get("adapter_name", "unknown")
            if adapter in self._adapter_stats:
                stats = self._adapter_stats[adapter]
                stats["results"] += record.result_count
                stats["total_latency_ms"] += record.latency_ms
                if not record.success:
                    stats["errors"] += 1
                if record.cost_dollars:
                    stats["total_cost"] += record.cost_dollars

    async def _handle_cache_hit(self, event: DomainEvent) -> None:
        """Process cache hit event."""
        payload = event.payload
        self._total_cache_hits += 1

        adapter = payload.get("adapter_name", "unknown")
        if adapter in self._adapter_stats:
            self._adapter_stats[adapter]["cache_hits"] += 1

    def get_recent_queries(self, limit: int = 10) -> List[ResearchQueryRecord]:
        """Get recent query records."""
        recent_ids = self._query_history[-limit:]
        return [
            self._queries[rid]
            for rid in reversed(recent_ids)
            if rid in self._queries
        ]

    def get_adapter_statistics(self, adapter_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for adapters."""
        if adapter_name:
            stats = self._adapter_stats.get(adapter_name, {})
            queries = stats.get("queries", 0)
            cache_hits = stats.get("cache_hits", 0)

            return {
                **stats,
                "cache_hit_rate": round(cache_hits / max(1, queries) * 100, 2),
                "avg_latency_ms": round(
                    stats.get("total_latency_ms", 0) / max(1, queries - cache_hits), 2
                ),
                "error_rate": round(
                    stats.get("errors", 0) / max(1, queries) * 100, 2
                ),
            }

        return {
            name: self.get_adapter_statistics(name)
            for name in self._adapter_stats
        }

    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global research statistics."""
        return {
            "total_queries": self._total_queries,
            "total_results": self._total_results,
            "total_cache_hits": self._total_cache_hits,
            "cache_hit_rate": round(
                self._total_cache_hits / max(1, self._total_queries) * 100, 2
            ),
            "total_cost": round(self._total_cost, 4),
            "adapters": list(self._adapter_stats.keys()),
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_research_event_system(
    event_bus: Optional[EventBus] = None,
    event_store: Optional[EventStore] = None,
    enable_aggregation: bool = True,
    batch_size: int = 50,
    flush_interval_ms: float = 100.0,
) -> Dict[str, Any]:
    """
    Create a complete research event system.

    Returns a dictionary with:
    - bus: EventBus instance
    - store: EventStore instance
    - metrics: EventMetrics instance
    - aggregator: EventAggregator (if enabled)
    - handler: ResearchEventHandler
    - projection: ResearchEventProjection
    - emitter: ResearchEventEmitter

    Example:
        system = create_research_event_system()

        # Subscribe handlers
        system["bus"].subscribe(
            "ResearchQueryRequestedEvent",
            system["handler"].handle_query_requested,
        )

        # Create event-aware adapter
        exa = ExaAdapter()
        event_exa = EventAwareResearchAdapter(
            exa,
            system["bus"],
            system["aggregator"],
        )
    """
    metrics = EventMetrics()
    bus = event_bus or EventBus(metrics=metrics)
    store = event_store or EventStore()

    aggregator = None
    if enable_aggregation:
        aggregator = EventAggregator(
            bus,
            batch_size=batch_size,
            flush_interval_ms=flush_interval_ms,
            metrics=metrics,
        )

    handler = ResearchEventHandler()
    projection = ResearchEventProjection()
    emitter = ResearchEventEmitter(bus, aggregator)

    # Register handlers
    research_event_types = [
        "ResearchQueryRequestedEvent",
        "ResearchResultReceivedEvent",
        "ResearchResultCachedEvent",
        "ResearchCacheHitEvent",
        "ResearchAdapterHealthEvent",
    ]

    for event_type in research_event_types:
        bus.subscribe(event_type, handler.handle_event, f"research_handler_{event_type}")
        bus.subscribe(event_type, projection.handle_event, f"research_projection_{event_type}")

    return {
        "bus": bus,
        "store": store,
        "metrics": metrics,
        "aggregator": aggregator,
        "handler": handler,
        "projection": projection,
        "emitter": emitter,
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def example_usage():
    """
    Example demonstrating the research events integration.

    This example shows:
    1. Setting up the event system
    2. Wrapping an adapter with event emission
    3. Executing queries with automatic event tracking
    4. Viewing statistics and projections
    """
    print("=== Research Events Integration Example ===\n")

    # Create the event system
    system = create_research_event_system(
        enable_aggregation=True,
        batch_size=10,
        flush_interval_ms=50.0,
    )

    bus = system["bus"]
    handler = system["handler"]
    projection = system["projection"]
    emitter = system["emitter"]

    # Simulate some research queries (without actual adapter)
    print("1. Simulating research queries...\n")

    queries = [
        ("AI agent architectures", "exa"),
        ("LangGraph patterns", "tavily"),
        ("Vector database comparison", "perplexity"),
        ("AI agent architectures", "exa"),  # Will be a cache hit
    ]

    for query, adapter in queries:
        # Emit query requested
        request_id = await emitter.emit_query_requested(
            query=query,
            adapter_name=adapter,
            operation="search",
        )

        # Simulate processing time
        await asyncio.sleep(0.1)

        # Emit result (or cache hit for duplicate)
        if query == "AI agent architectures" and adapter == "exa":
            if projection._total_queries > 0:  # Second time = cache hit
                await emitter.emit_cache_hit(
                    query=query,
                    cache_key=f"cache_{query[:10]}",
                    adapter_name=adapter,
                    similarity_score=0.98,
                )
                continue

        await emitter.emit_result_received(
            request_id=request_id,
            adapter_name=adapter,
            result_count=10,
            latency_ms=150.0,
            success=True,
            cost_dollars=0.002,
        )

        await emitter.emit_result_cached(
            request_id=request_id,
            cache_key=f"cache_{query[:10]}",
            adapter_name=adapter,
            result_count=10,
            ttl_seconds=3600,
        )

    # Flush any remaining events
    await emitter.flush()

    # Wait for events to process
    await asyncio.sleep(0.2)

    # Print statistics
    print("\n2. Handler Statistics:")
    print(handler.get_statistics())

    print("\n3. Projection - Global Statistics:")
    print(projection.get_global_statistics())

    print("\n4. Projection - Adapter Statistics:")
    for adapter, stats in projection.get_adapter_statistics().items():
        print(f"  {adapter}: {stats}")

    print("\n5. Recent Queries:")
    for record in projection.get_recent_queries(5):
        print(f"  - {record.query[:40]}... ({record.adapter_name})")

    print("\n6. Event Bus Metrics:")
    metrics = bus.get_metrics()
    print(f"  Events published: {metrics['events_published']}")
    print(f"  Events handled: {metrics['events_handled']}")
    print(f"  Handlers registered: {metrics['registered_handlers']}")

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    asyncio.run(example_usage())
