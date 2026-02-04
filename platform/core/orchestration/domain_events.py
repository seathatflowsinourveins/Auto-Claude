"""
Domain Events System - V40 Architecture (ADR-029 Enhanced)

Event-driven architecture for the UNLEASH platform.
Provides domain event definitions, event bus, and event sourcing support.

Features:
- Versioned domain events with metadata and schema evolution
- In-memory event bus with async handlers
- Event sourcing with append-only event store
- Outbox pattern integration for reliable external publishing
- Event aggregation for batching multiple events
- Event replay capability for rebuilding state
- Dead letter queue for failed event handlers with retry policies
- Comprehensive metrics/observability for event throughput
- Research adapter event integration

Usage:
    from core.orchestration.domain_events import (
        DomainEvent,
        MemoryStoredEvent,
        SessionStartedEvent,
        ResearchQueryRequestedEvent,
        ResearchResultReceivedEvent,
        ResearchResultCachedEvent,
        EventBus,
        EventStore,
        EventAggregator,
        EventMetrics,
        DeadLetterQueue,
    )

    # Create event bus with metrics
    metrics = EventMetrics()
    bus = EventBus(metrics=metrics)

    # Subscribe to events
    async def on_memory_stored(event: MemoryStoredEvent):
        print(f"Memory {event.memory_id} stored")

    bus.subscribe("MemoryStoredEvent", on_memory_stored)

    # Publish event
    event = MemoryStoredEvent(
        aggregate_id="session-123",
        memory_id="mem-456",
        content_preview="User prefers dark mode",
        memory_type="preference",
    )
    await bus.publish(event)

    # Event sourcing
    store = EventStore()
    await store.append([event])
    events = await store.get_events("session-123")

    # Event aggregation for batching
    aggregator = EventAggregator(bus, batch_size=50, flush_interval_ms=100)
    aggregator.add(event)
    await aggregator.flush()

    # Replay events to rebuild state
    await store.replay_all(handler, from_timestamp=start_time)

    # Research events integration
    research_event = ResearchQueryRequestedEvent(
        aggregate_id="research-123",
        query="AI safety research",
        adapter_name="exa",
    )
    await bus.publish(research_event)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from collections import deque

logger = logging.getLogger(__name__)


# =============================================================================
# EVENT METRICS AND OBSERVABILITY
# =============================================================================

@dataclass
class EventMetrics:
    """
    Comprehensive metrics for event system observability.

    Tracks:
    - Event throughput (events/second)
    - Handler latencies (p50, p95, p99)
    - Error rates by handler and event type
    - Dead letter queue statistics
    - Replay progress
    """

    def __init__(self, window_size: int = 1000):
        self._window_size = window_size
        self._events_published: int = 0
        self._events_handled: int = 0
        self._events_failed: int = 0
        self._events_replayed: int = 0
        self._events_aggregated: int = 0

        # Latency tracking per handler
        self._handler_latencies: Dict[str, deque] = {}

        # Error tracking
        self._errors_by_handler: Dict[str, int] = {}
        self._errors_by_event_type: Dict[str, int] = {}

        # Throughput tracking
        self._throughput_window: deque = deque(maxlen=100)
        self._last_throughput_check: float = time.time()

        # Dead letter stats
        self._dlq_additions: int = 0
        self._dlq_retries: int = 0
        self._dlq_successes: int = 0

    def record_published(self, event_type: str) -> None:
        """Record an event publication."""
        self._events_published += 1
        now = time.time()
        self._throughput_window.append(now)

    def record_handled(self, handler_name: str, latency_ms: float) -> None:
        """Record successful event handling."""
        self._events_handled += 1
        if handler_name not in self._handler_latencies:
            self._handler_latencies[handler_name] = deque(maxlen=self._window_size)
        self._handler_latencies[handler_name].append(latency_ms)

    def record_failed(self, handler_name: str, event_type: str) -> None:
        """Record failed event handling."""
        self._events_failed += 1
        self._errors_by_handler[handler_name] = self._errors_by_handler.get(handler_name, 0) + 1
        self._errors_by_event_type[event_type] = self._errors_by_event_type.get(event_type, 0) + 1

    def record_replayed(self, count: int = 1) -> None:
        """Record replayed events."""
        self._events_replayed += count

    def record_aggregated(self, count: int = 1) -> None:
        """Record aggregated events."""
        self._events_aggregated += count

    def record_dlq_addition(self) -> None:
        """Record event added to dead letter queue."""
        self._dlq_additions += 1

    def record_dlq_retry(self) -> None:
        """Record DLQ retry attempt."""
        self._dlq_retries += 1

    def record_dlq_success(self) -> None:
        """Record successful DLQ reprocessing."""
        self._dlq_successes += 1

    def get_percentile(self, handler_name: str, percentile: float) -> float:
        """Get latency percentile for a handler."""
        latencies = self._handler_latencies.get(handler_name)
        if not latencies:
            return 0.0
        sorted_latencies = sorted(latencies)
        index = int(len(sorted_latencies) * percentile / 100)
        return sorted_latencies[min(index, len(sorted_latencies) - 1)]

    def get_throughput(self) -> float:
        """Get events per second over the last window."""
        now = time.time()
        # Filter to events in last 60 seconds
        recent = [t for t in self._throughput_window if now - t < 60]
        if len(recent) < 2:
            return 0.0
        duration = recent[-1] - recent[0]
        if duration <= 0:
            return 0.0
        return len(recent) / duration

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        handler_stats = {}
        for handler, latencies in self._handler_latencies.items():
            if latencies:
                handler_stats[handler] = {
                    "count": len(latencies),
                    "p50_ms": round(self.get_percentile(handler, 50), 2),
                    "p95_ms": round(self.get_percentile(handler, 95), 2),
                    "p99_ms": round(self.get_percentile(handler, 99), 2),
                    "avg_ms": round(sum(latencies) / len(latencies), 2),
                    "errors": self._errors_by_handler.get(handler, 0),
                }

        return {
            "events_published": self._events_published,
            "events_handled": self._events_handled,
            "events_failed": self._events_failed,
            "events_replayed": self._events_replayed,
            "events_aggregated": self._events_aggregated,
            "throughput_per_sec": round(self.get_throughput(), 2),
            "error_rate": round(
                self._events_failed / max(1, self._events_handled) * 100, 2
            ),
            "handler_stats": handler_stats,
            "errors_by_event_type": dict(self._errors_by_event_type),
            "dead_letter_queue": {
                "additions": self._dlq_additions,
                "retries": self._dlq_retries,
                "successes": self._dlq_successes,
            },
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._events_published = 0
        self._events_handled = 0
        self._events_failed = 0
        self._events_replayed = 0
        self._events_aggregated = 0
        self._handler_latencies.clear()
        self._errors_by_handler.clear()
        self._errors_by_event_type.clear()
        self._throughput_window.clear()
        self._dlq_additions = 0
        self._dlq_retries = 0
        self._dlq_successes = 0


# =============================================================================
# EVENT VERSIONING
# =============================================================================

class EventVersion:
    """
    Semantic versioning for events with schema evolution support.

    Supports:
    - Version parsing and comparison
    - Compatibility checks (same major version)
    - Version ordering
    - Schema migration hints
    """

    def __init__(self, major: int = 1, minor: int = 0, patch: int = 0):
        self.major = major
        self.minor = minor
        self.patch = patch

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __repr__(self) -> str:
        return f"EventVersion({self.major}, {self.minor}, {self.patch})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EventVersion):
            return False
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
        )

    def __lt__(self, other: "EventVersion") -> bool:
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        return self.patch < other.patch

    def __le__(self, other: "EventVersion") -> bool:
        return self == other or self < other

    def __gt__(self, other: "EventVersion") -> bool:
        return not self <= other

    def __ge__(self, other: "EventVersion") -> bool:
        return not self < other

    def __hash__(self) -> int:
        return hash((self.major, self.minor, self.patch))

    def is_compatible(self, other: "EventVersion") -> bool:
        """Check if versions are compatible (same major version)."""
        return self.major == other.major

    def is_backward_compatible(self, other: "EventVersion") -> bool:
        """Check if this version is backward compatible with other."""
        return self.major == other.major and self >= other

    def bump_major(self) -> "EventVersion":
        """Create new version with bumped major."""
        return EventVersion(self.major + 1, 0, 0)

    def bump_minor(self) -> "EventVersion":
        """Create new version with bumped minor."""
        return EventVersion(self.major, self.minor + 1, 0)

    def bump_patch(self) -> "EventVersion":
        """Create new version with bumped patch."""
        return EventVersion(self.major, self.minor, self.patch + 1)

    @classmethod
    def parse(cls, version_str: str) -> "EventVersion":
        """Parse version string (e.g., '1.2.3')."""
        parts = version_str.split(".")
        return cls(
            major=int(parts[0]) if len(parts) > 0 else 1,
            minor=int(parts[1]) if len(parts) > 1 else 0,
            patch=int(parts[2]) if len(parts) > 2 else 0,
        )


@dataclass
class EventSchemaEvolution:
    """
    Schema evolution support for event versioning.

    Provides:
    - Upcasters for migrating old events to new schemas
    - Downcasters for backward compatibility
    - Schema validation
    """

    event_type: str
    current_version: EventVersion
    _upcasters: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = field(
        default_factory=dict
    )
    _downcasters: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = field(
        default_factory=dict
    )

    def register_upcaster(
        self,
        from_version: str,
        to_version: str,
        upcaster: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> None:
        """Register an upcaster for version migration."""
        key = f"{from_version}->{to_version}"
        self._upcasters[key] = upcaster

    def register_downcaster(
        self,
        from_version: str,
        to_version: str,
        downcaster: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> None:
        """Register a downcaster for backward compatibility."""
        key = f"{from_version}->{to_version}"
        self._downcasters[key] = downcaster

    def upcast(
        self,
        event_data: Dict[str, Any],
        from_version: str,
        to_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upcast event data from old version to new version.

        Args:
            event_data: Event data dictionary
            from_version: Source version
            to_version: Target version (defaults to current)

        Returns:
            Migrated event data
        """
        if to_version is None:
            to_version = str(self.current_version)

        if from_version == to_version:
            return event_data

        # Find migration path
        key = f"{from_version}->{to_version}"
        if key in self._upcasters:
            return self._upcasters[key](event_data)

        # Try incremental migration
        from_v = EventVersion.parse(from_version)
        to_v = EventVersion.parse(to_version)

        if from_v >= to_v:
            return event_data

        # Step through versions
        current_data = event_data
        current_v = from_v

        while current_v < to_v:
            # Try minor bump first, then major
            next_minor = current_v.bump_minor()
            next_major = current_v.bump_major()

            minor_key = f"{current_v}->{next_minor}"
            major_key = f"{current_v}->{next_major}"

            if minor_key in self._upcasters and next_minor <= to_v:
                current_data = self._upcasters[minor_key](current_data)
                current_v = next_minor
            elif major_key in self._upcasters and next_major <= to_v:
                current_data = self._upcasters[major_key](current_data)
                current_v = next_major
            else:
                # No path found, return as-is with warning
                logger.warning(
                    f"No migration path from {current_v} to {to_v} for {self.event_type}"
                )
                break

        return current_data


# =============================================================================
# BASE DOMAIN EVENT
# =============================================================================

@dataclass
class DomainEvent:
    """
    Base class for all domain events.

    All domain events must extend this class and define:
    - event_type: String identifier for the event type
    - aggregate_id: ID of the aggregate this event belongs to
    - aggregate_type: Type name of the aggregate

    Events are immutable and carry all information needed to
    reconstruct state changes in event sourcing.
    """

    # Required fields (set by subclasses or explicitly)
    event_type: str = field(default="DomainEvent")
    aggregate_id: str = field(default="")
    aggregate_type: str = field(default="")

    # Auto-generated fields
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: int = field(default=1)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Payload and metadata
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Event schema version
    schema_version: str = field(default="1.0.0")

    def __post_init__(self) -> None:
        """Set event_type from class name if not explicitly set."""
        if self.event_type == "DomainEvent":
            self.event_type = self.__class__.__name__

    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "metadata": self.metadata,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DomainEvent":
        """Deserialize event from dictionary."""
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=data.get("event_type", "DomainEvent"),
            aggregate_id=data.get("aggregate_id", ""),
            aggregate_type=data.get("aggregate_type", ""),
            version=data.get("version", 1),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if isinstance(data.get("timestamp"), str)
            else data.get("timestamp", datetime.now(timezone.utc)),
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {}),
            schema_version=data.get("schema_version", "1.0.0"),
        )

    def with_metadata(self, **kwargs) -> "DomainEvent":
        """Create new event with additional metadata."""
        new_metadata = {**self.metadata, **kwargs}
        return self.__class__(
            event_id=self.event_id,
            event_type=self.event_type,
            aggregate_id=self.aggregate_id,
            aggregate_type=self.aggregate_type,
            version=self.version,
            timestamp=self.timestamp,
            payload=self.payload,
            metadata=new_metadata,
            schema_version=self.schema_version,
        )

    def __hash__(self) -> int:
        return hash(self.event_id)


# =============================================================================
# MEMORY DOMAIN EVENTS
# =============================================================================

@dataclass
class MemoryStoredEvent(DomainEvent):
    """Event raised when a memory entry is stored."""

    memory_id: str = ""
    content_preview: str = ""
    memory_type: str = "generic"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.event_type = "MemoryStoredEvent"
        self.aggregate_type = "Memory"
        self.payload = {
            "memory_id": self.memory_id,
            "content_preview": self.content_preview[:100]
            if self.content_preview
            else "",
            "memory_type": self.memory_type,
        }


@dataclass
class MemoryRetrievedEvent(DomainEvent):
    """Event raised when a memory entry is retrieved."""

    memory_id: str = ""
    query: str = ""
    hit_count: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.event_type = "MemoryRetrievedEvent"
        self.aggregate_type = "Memory"
        self.payload = {
            "memory_id": self.memory_id,
            "query": self.query[:50] if self.query else "",
            "hit_count": self.hit_count,
        }


@dataclass
class MemoryEvictedEvent(DomainEvent):
    """Event raised when a memory entry is evicted."""

    memory_id: str = ""
    eviction_reason: str = "capacity"
    tier: str = "main_context"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.event_type = "MemoryEvictedEvent"
        self.aggregate_type = "Memory"
        self.payload = {
            "memory_id": self.memory_id,
            "eviction_reason": self.eviction_reason,
            "tier": self.tier,
        }


@dataclass
class MemoryPromotedEvent(DomainEvent):
    """Event raised when a memory entry is promoted to a higher tier."""

    memory_id: str = ""
    from_tier: str = ""
    to_tier: str = ""
    reason: str = "frequency"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.event_type = "MemoryPromotedEvent"
        self.aggregate_type = "Memory"
        self.payload = {
            "memory_id": self.memory_id,
            "from_tier": self.from_tier,
            "to_tier": self.to_tier,
            "reason": self.reason,
        }


# =============================================================================
# SESSION DOMAIN EVENTS
# =============================================================================

@dataclass
class SessionStartedEvent(DomainEvent):
    """Event raised when a session starts."""

    session_id: str = ""
    project_path: str = ""
    agent_type: str = "default"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.event_type = "SessionStartedEvent"
        self.aggregate_type = "Session"
        self.aggregate_id = self.session_id
        self.payload = {
            "session_id": self.session_id,
            "project_path": self.project_path,
            "agent_type": self.agent_type,
        }


@dataclass
class SessionEndedEvent(DomainEvent):
    """Event raised when a session ends."""

    session_id: str = ""
    duration_seconds: float = 0.0
    exit_reason: str = "normal"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.event_type = "SessionEndedEvent"
        self.aggregate_type = "Session"
        self.aggregate_id = self.session_id
        self.payload = {
            "session_id": self.session_id,
            "duration_seconds": self.duration_seconds,
            "exit_reason": self.exit_reason,
        }


@dataclass
class SessionCheckpointEvent(DomainEvent):
    """Event raised when a session checkpoint is created."""

    session_id: str = ""
    checkpoint_id: str = ""
    state_summary: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.event_type = "SessionCheckpointEvent"
        self.aggregate_type = "Session"
        self.aggregate_id = self.session_id
        self.payload = {
            "session_id": self.session_id,
            "checkpoint_id": self.checkpoint_id,
            "state_summary": self.state_summary[:200] if self.state_summary else "",
        }


# =============================================================================
# FLOW DOMAIN EVENTS
# =============================================================================

@dataclass
class FlowStartedEvent(DomainEvent):
    """Event raised when a flow execution starts."""

    flow_id: str = ""
    flow_type: str = ""
    input_summary: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.event_type = "FlowStartedEvent"
        self.aggregate_type = "Flow"
        self.aggregate_id = self.flow_id
        self.payload = {
            "flow_id": self.flow_id,
            "flow_type": self.flow_type,
            "input_summary": self.input_summary[:100] if self.input_summary else "",
        }


@dataclass
class FlowCompletedEvent(DomainEvent):
    """Event raised when a flow execution completes."""

    flow_id: str = ""
    flow_type: str = ""
    duration_seconds: float = 0.0
    steps_completed: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.event_type = "FlowCompletedEvent"
        self.aggregate_type = "Flow"
        self.aggregate_id = self.flow_id
        self.payload = {
            "flow_id": self.flow_id,
            "flow_type": self.flow_type,
            "duration_seconds": self.duration_seconds,
            "steps_completed": self.steps_completed,
        }


@dataclass
class FlowFailedEvent(DomainEvent):
    """Event raised when a flow execution fails."""

    flow_id: str = ""
    flow_type: str = ""
    error_message: str = ""
    failed_step: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.event_type = "FlowFailedEvent"
        self.aggregate_type = "Flow"
        self.aggregate_id = self.flow_id
        self.payload = {
            "flow_id": self.flow_id,
            "flow_type": self.flow_type,
            "error_message": self.error_message[:500] if self.error_message else "",
            "failed_step": self.failed_step,
        }


# =============================================================================
# RESEARCH DOMAIN EVENTS
# =============================================================================

@dataclass
class ResearchQueryRequestedEvent(DomainEvent):
    """
    Event raised when a research query is requested.

    Emitted when a research adapter (Exa, Tavily, Perplexity, etc.)
    receives a search or research request.
    """

    query: str = ""
    adapter_name: str = ""
    operation: str = "search"
    parameters: Dict[str, Any] = field(default_factory=dict)
    request_id: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.event_type = "ResearchQueryRequestedEvent"
        self.aggregate_type = "Research"
        if not self.request_id:
            self.request_id = str(uuid.uuid4())
        if not self.aggregate_id:
            self.aggregate_id = self.request_id
        self.payload = {
            "query": self.query[:500] if self.query else "",
            "adapter_name": self.adapter_name,
            "operation": self.operation,
            "parameters": {
                k: v for k, v in self.parameters.items()
                if k not in ("api_key", "key", "secret", "token")  # Exclude secrets
            },
            "request_id": self.request_id,
        }


@dataclass
class ResearchResultReceivedEvent(DomainEvent):
    """
    Event raised when research results are received.

    Emitted when a research adapter successfully returns results.
    """

    request_id: str = ""
    adapter_name: str = ""
    result_count: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    cost_dollars: Optional[float] = None
    result_preview: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.event_type = "ResearchResultReceivedEvent"
        self.aggregate_type = "Research"
        if not self.aggregate_id:
            self.aggregate_id = self.request_id
        self.payload = {
            "request_id": self.request_id,
            "adapter_name": self.adapter_name,
            "result_count": self.result_count,
            "latency_ms": round(self.latency_ms, 2),
            "success": self.success,
            "error_message": self.error_message[:200] if self.error_message else None,
            "cost_dollars": self.cost_dollars,
            "result_preview": self.result_preview[:200] if self.result_preview else "",
        }


@dataclass
class ResearchResultCachedEvent(DomainEvent):
    """
    Event raised when research results are cached.

    Emitted when results are stored in the semantic cache or memory system.
    """

    request_id: str = ""
    cache_key: str = ""
    cache_type: str = "semantic"  # semantic, memory, local
    ttl_seconds: Optional[int] = None
    adapter_name: str = ""
    result_count: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.event_type = "ResearchResultCachedEvent"
        self.aggregate_type = "Research"
        if not self.aggregate_id:
            self.aggregate_id = self.request_id
        self.payload = {
            "request_id": self.request_id,
            "cache_key": self.cache_key,
            "cache_type": self.cache_type,
            "ttl_seconds": self.ttl_seconds,
            "adapter_name": self.adapter_name,
            "result_count": self.result_count,
        }


@dataclass
class ResearchCacheHitEvent(DomainEvent):
    """Event raised when a research query hits the cache."""

    query: str = ""
    cache_key: str = ""
    cache_type: str = "semantic"
    adapter_name: str = ""
    similarity_score: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.event_type = "ResearchCacheHitEvent"
        self.aggregate_type = "Research"
        self.payload = {
            "query": self.query[:200] if self.query else "",
            "cache_key": self.cache_key,
            "cache_type": self.cache_type,
            "adapter_name": self.adapter_name,
            "similarity_score": round(self.similarity_score, 4),
        }


@dataclass
class ResearchAdapterHealthEvent(DomainEvent):
    """Event raised for research adapter health changes."""

    adapter_name: str = ""
    status: str = "healthy"  # healthy, degraded, unhealthy
    latency_ms: float = 0.0
    error_rate: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.event_type = "ResearchAdapterHealthEvent"
        self.aggregate_type = "Research"
        if not self.aggregate_id:
            self.aggregate_id = f"adapter-{self.adapter_name}"
        self.payload = {
            "adapter_name": self.adapter_name,
            "status": self.status,
            "latency_ms": round(self.latency_ms, 2),
            "error_rate": round(self.error_rate, 4),
            "details": self.details,
        }


# =============================================================================
# EVENT HANDLER TYPES
# =============================================================================

EventHandler = Callable[[DomainEvent], Awaitable[None]]
SyncEventHandler = Callable[[DomainEvent], None]


class EventHandlerProtocol(Protocol):
    """Protocol for event handlers."""

    async def handle(self, event: DomainEvent) -> None:
        """Handle an event."""
        ...


# =============================================================================
# DEAD LETTER QUEUE
# =============================================================================

@dataclass
class DeadLetterEntry:
    """Entry in the dead letter queue."""

    event: DomainEvent
    handler_name: str
    error: Exception
    failed_at: datetime
    retry_count: int = 0
    last_retry_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "event_id": self.event.event_id,
            "event_type": self.event.event_type,
            "handler_name": self.handler_name,
            "error": str(self.error),
            "error_type": type(self.error).__name__,
            "failed_at": self.failed_at.isoformat(),
            "retry_count": self.retry_count,
            "last_retry_at": self.last_retry_at.isoformat() if self.last_retry_at else None,
            "metadata": self.metadata,
        }


class DeadLetterQueue:
    """
    Enhanced dead letter queue for failed event handlers.

    Features:
    - Configurable retry policies
    - Exponential backoff
    - Max retry limits
    - Age-based expiration
    - Manual reprocessing
    - Statistics and monitoring
    """

    def __init__(
        self,
        max_retries: int = 5,
        base_delay_seconds: float = 1.0,
        max_delay_seconds: float = 300.0,
        expiration_hours: float = 24.0,
        metrics: Optional[EventMetrics] = None,
    ):
        self._queue: List[DeadLetterEntry] = []
        self._max_retries = max_retries
        self._base_delay = base_delay_seconds
        self._max_delay = max_delay_seconds
        self._expiration_hours = expiration_hours
        self._metrics = metrics
        self._lock = asyncio.Lock()

        # Statistics
        self._total_added: int = 0
        self._total_retried: int = 0
        self._total_expired: int = 0
        self._total_succeeded: int = 0

    async def add(
        self,
        event: DomainEvent,
        handler_name: str,
        error: Exception,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a failed event to the dead letter queue."""
        async with self._lock:
            entry = DeadLetterEntry(
                event=event,
                handler_name=handler_name,
                error=error,
                failed_at=datetime.now(timezone.utc),
                metadata=metadata or {},
            )
            self._queue.append(entry)
            self._total_added += 1

            if self._metrics:
                self._metrics.record_dlq_addition()

            logger.warning(
                f"Event {event.event_id} added to DLQ: {handler_name} - {error}"
            )

    async def retry_all(
        self,
        handler_map: Dict[str, EventHandler],
    ) -> Tuple[int, int]:
        """
        Retry all eligible events in the queue.

        Args:
            handler_map: Map of handler names to handler functions

        Returns:
            Tuple of (succeeded, failed)
        """
        succeeded = 0
        failed = 0
        now = datetime.now(timezone.utc)

        async with self._lock:
            remaining: List[DeadLetterEntry] = []

            for entry in self._queue:
                # Check expiration
                age_hours = (now - entry.failed_at).total_seconds() / 3600
                if age_hours > self._expiration_hours:
                    self._total_expired += 1
                    logger.info(
                        f"Event {entry.event.event_id} expired in DLQ after {age_hours:.1f}h"
                    )
                    continue

                # Check max retries
                if entry.retry_count >= self._max_retries:
                    remaining.append(entry)
                    continue

                # Check backoff
                if entry.last_retry_at:
                    delay = min(
                        self._base_delay * (2 ** entry.retry_count),
                        self._max_delay,
                    )
                    time_since_retry = (now - entry.last_retry_at).total_seconds()
                    if time_since_retry < delay:
                        remaining.append(entry)
                        continue

                # Try to retry
                handler = handler_map.get(entry.handler_name)
                if not handler:
                    remaining.append(entry)
                    continue

                try:
                    entry.retry_count += 1
                    entry.last_retry_at = now
                    self._total_retried += 1

                    if self._metrics:
                        self._metrics.record_dlq_retry()

                    await handler(entry.event)
                    succeeded += 1
                    self._total_succeeded += 1

                    if self._metrics:
                        self._metrics.record_dlq_success()

                    logger.info(
                        f"Event {entry.event.event_id} succeeded on retry #{entry.retry_count}"
                    )

                except Exception as e:
                    entry.error = e
                    remaining.append(entry)
                    failed += 1
                    logger.warning(
                        f"Event {entry.event.event_id} failed retry #{entry.retry_count}: {e}"
                    )

            self._queue = remaining

        return succeeded, failed

    async def retry_single(
        self,
        event_id: str,
        handler: EventHandler,
    ) -> bool:
        """Retry a single event by ID."""
        async with self._lock:
            for i, entry in enumerate(self._queue):
                if entry.event.event_id == event_id:
                    try:
                        entry.retry_count += 1
                        entry.last_retry_at = datetime.now(timezone.utc)
                        await handler(entry.event)
                        self._queue.pop(i)
                        self._total_succeeded += 1
                        return True
                    except Exception as e:
                        entry.error = e
                        return False
        return False

    def get_entries(
        self,
        handler_name: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[DeadLetterEntry]:
        """Get entries from the queue with optional filters."""
        entries = self._queue

        if handler_name:
            entries = [e for e in entries if e.handler_name == handler_name]
        if event_type:
            entries = [e for e in entries if e.event.event_type == event_type]

        return entries[:limit]

    def clear(self, handler_name: Optional[str] = None) -> int:
        """Clear entries, optionally for a specific handler."""
        if handler_name:
            before = len(self._queue)
            self._queue = [e for e in self._queue if e.handler_name != handler_name]
            return before - len(self._queue)
        else:
            count = len(self._queue)
            self._queue.clear()
            return count

    def get_statistics(self) -> Dict[str, Any]:
        """Get DLQ statistics."""
        by_handler: Dict[str, int] = {}
        by_event_type: Dict[str, int] = {}

        for entry in self._queue:
            by_handler[entry.handler_name] = by_handler.get(entry.handler_name, 0) + 1
            by_event_type[entry.event.event_type] = by_event_type.get(entry.event.event_type, 0) + 1

        return {
            "queue_size": len(self._queue),
            "total_added": self._total_added,
            "total_retried": self._total_retried,
            "total_expired": self._total_expired,
            "total_succeeded": self._total_succeeded,
            "by_handler": by_handler,
            "by_event_type": by_event_type,
            "config": {
                "max_retries": self._max_retries,
                "base_delay_seconds": self._base_delay,
                "max_delay_seconds": self._max_delay,
                "expiration_hours": self._expiration_hours,
            },
        }

    def __len__(self) -> int:
        return len(self._queue)


# =============================================================================
# EVENT AGGREGATOR
# =============================================================================

class EventAggregator:
    """
    Event aggregation for batching multiple events.

    Features:
    - Batch events by size or time
    - Configurable flush intervals
    - Automatic flushing
    - Event deduplication (optional)
    - Statistics tracking
    """

    def __init__(
        self,
        event_bus: "EventBus",
        batch_size: int = 50,
        flush_interval_ms: float = 100.0,
        deduplicate: bool = False,
        metrics: Optional[EventMetrics] = None,
    ):
        self._bus = event_bus
        self._batch_size = batch_size
        self._flush_interval_ms = flush_interval_ms
        self._deduplicate = deduplicate
        self._metrics = metrics

        self._pending: List[DomainEvent] = []
        self._seen_ids: Set[str] = set() if deduplicate else set()
        self._batch_start: Optional[float] = None
        self._lock = asyncio.Lock()

        # Statistics
        self._total_added: int = 0
        self._total_flushed: int = 0
        self._total_batches: int = 0
        self._duplicates_skipped: int = 0

    def add(self, event: DomainEvent) -> bool:
        """
        Add an event to the aggregation buffer.

        Args:
            event: Event to add

        Returns:
            True if batch should be flushed (reached max size)
        """
        self._total_added += 1

        # Deduplication check
        if self._deduplicate:
            if event.event_id in self._seen_ids:
                self._duplicates_skipped += 1
                return False
            self._seen_ids.add(event.event_id)

        # Initialize batch start time
        if not self._pending:
            self._batch_start = time.time()

        self._pending.append(event)

        if self._metrics:
            self._metrics.record_aggregated()

        # Check if batch is full
        return len(self._pending) >= self._batch_size

    def should_flush(self) -> bool:
        """Check if batch should be flushed based on size or timeout."""
        if not self._pending:
            return False

        if len(self._pending) >= self._batch_size:
            return True

        if self._batch_start:
            elapsed_ms = (time.time() - self._batch_start) * 1000
            return elapsed_ms >= self._flush_interval_ms

        return False

    async def flush(self) -> int:
        """
        Flush pending events to the event bus.

        Returns:
            Number of events flushed
        """
        async with self._lock:
            if not self._pending:
                return 0

            events = self._pending
            self._pending = []
            self._batch_start = None

            if self._deduplicate:
                # Keep seen IDs for a while to catch late duplicates
                # but limit memory usage
                if len(self._seen_ids) > 10000:
                    self._seen_ids = set(
                        e.event_id for e in events
                    )

            self._total_flushed += len(events)
            self._total_batches += 1

            # Publish batch
            await self._bus.publish_batch(events)

            return len(events)

    async def flush_if_ready(self) -> int:
        """Flush if conditions are met."""
        if self.should_flush():
            return await self.flush()
        return 0

    def pending_count(self) -> int:
        """Get number of pending events."""
        return len(self._pending)

    def clear(self) -> int:
        """Clear pending events without flushing."""
        count = len(self._pending)
        self._pending.clear()
        self._batch_start = None
        return count

    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregator statistics."""
        return {
            "total_added": self._total_added,
            "total_flushed": self._total_flushed,
            "total_batches": self._total_batches,
            "pending": len(self._pending),
            "duplicates_skipped": self._duplicates_skipped,
            "avg_batch_size": round(
                self._total_flushed / max(1, self._total_batches), 1
            ),
            "config": {
                "batch_size": self._batch_size,
                "flush_interval_ms": self._flush_interval_ms,
                "deduplicate": self._deduplicate,
            },
        }


# =============================================================================
# EVENT BUS
# =============================================================================

class EventBus:
    """
    In-memory event bus with async handlers.

    Features:
    - Subscribe to specific event types
    - Wildcard subscription for all events
    - Async and sync handler support
    - Enhanced dead letter queue with retry policies
    - Comprehensive metrics and observability
    - Handler timeout support
    - Event filtering

    Thread Safety:
        The event bus is safe for concurrent use within a single asyncio loop.
        For multi-process scenarios, use EventStore with outbox pattern.
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
        enable_dead_letter: bool = True,
        handler_timeout_seconds: float = 30.0,
        metrics: Optional[EventMetrics] = None,
        dead_letter_queue: Optional[DeadLetterQueue] = None,
    ):
        """
        Initialize the event bus.

        Args:
            max_retries: Maximum retry attempts for failed handlers
            retry_delay_seconds: Delay between retries
            enable_dead_letter: Enable dead letter queue for failed events
            handler_timeout_seconds: Timeout for handler execution
            metrics: Optional EventMetrics instance for observability
            dead_letter_queue: Optional custom DeadLetterQueue
        """
        self._handlers: Dict[str, List[Tuple[str, EventHandler]]] = {}
        self._wildcard_handlers: List[Tuple[str, EventHandler]] = []
        self._handler_names: Dict[EventHandler, str] = {}
        self._max_retries = max_retries
        self._retry_delay = retry_delay_seconds
        self._enable_dead_letter = enable_dead_letter
        self._handler_timeout = handler_timeout_seconds

        # Enhanced metrics
        self._metrics = metrics or EventMetrics()

        # Enhanced dead letter queue
        self._dead_letter_queue = dead_letter_queue or (
            DeadLetterQueue(
                max_retries=max_retries * 2,
                metrics=self._metrics,
            ) if enable_dead_letter else None
        )

        # Legacy compatibility
        self._dead_letter: List[tuple[DomainEvent, Exception]] = []
        self._events_published: int = 0
        self._events_handled: int = 0
        self._events_failed: int = 0
        self._handler_latencies: Dict[str, List[float]] = {}

    def subscribe(
        self,
        event_type: str,
        handler: EventHandler,
        handler_name: Optional[str] = None,
    ) -> None:
        """
        Subscribe a handler to an event type.

        Args:
            event_type: Event type name (e.g., 'MemoryStoredEvent')
                        Use '*' for wildcard subscription
            handler: Async callable that accepts DomainEvent
            handler_name: Optional name for the handler (for metrics/DLQ)
        """
        name = handler_name or getattr(handler, "__name__", str(id(handler)))
        self._handler_names[handler] = name

        if event_type == "*":
            self._wildcard_handlers.append((name, handler))
            logger.debug(f"Registered wildcard handler: {name}")
        else:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append((name, handler))
            logger.debug(f"Registered handler {name} for {event_type}")

    def unsubscribe(
        self,
        event_type: str,
        handler: EventHandler,
    ) -> bool:
        """
        Unsubscribe a handler from an event type.

        Args:
            event_type: Event type name
            handler: Handler to remove

        Returns:
            True if handler was found and removed
        """
        if event_type == "*":
            for i, (name, h) in enumerate(self._wildcard_handlers):
                if h is handler:
                    self._wildcard_handlers.pop(i)
                    self._handler_names.pop(handler, None)
                    return True
            return False

        if event_type in self._handlers:
            for i, (name, h) in enumerate(self._handlers[event_type]):
                if h is handler:
                    self._handlers[event_type].pop(i)
                    self._handler_names.pop(handler, None)
                    return True
        return False

    async def publish(self, event: DomainEvent) -> None:
        """
        Publish event to all subscribers.

        Events are delivered to:
        1. All handlers registered for the specific event type
        2. All wildcard handlers

        Args:
            event: Domain event to publish
        """
        self._events_published += 1
        self._metrics.record_published(event.event_type)

        # Get handlers for this event type
        handlers = list(self._handlers.get(event.event_type, []))
        handlers.extend(self._wildcard_handlers)

        if not handlers:
            logger.debug(f"No handlers for event: {event.event_type}")
            return

        # Execute handlers concurrently
        tasks = [
            self._execute_handler(name, handler, event)
            for name, handler in handlers
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def publish_batch(self, events: List[DomainEvent]) -> None:
        """
        Publish multiple events.

        Events are published concurrently.

        Args:
            events: List of domain events to publish
        """
        tasks = [self.publish(event) for event in events]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def publish_and_persist(
        self,
        event: DomainEvent,
        outbox: "TransactionalOutbox",
    ) -> None:
        """
        Publish event locally and persist to outbox.

        This implements the transactional outbox pattern:
        1. Persist event to outbox (for external consumers)
        2. Publish to local handlers

        Args:
            event: Domain event to publish
            outbox: Outbox for persistence
        """
        # Persist to outbox first (for durability)
        await outbox.add(event)

        # Then publish locally
        await self.publish(event)

    async def _execute_handler(
        self,
        handler_name: str,
        handler: EventHandler,
        event: DomainEvent,
    ) -> None:
        """Execute a single handler with retry logic and timeout."""
        start_time = time.time()
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
                # Execute with timeout
                await asyncio.wait_for(
                    handler(event),
                    timeout=self._handler_timeout,
                )
                self._events_handled += 1

                # Record latency
                latency_ms = (time.time() - start_time) * 1000
                if handler_name not in self._handler_latencies:
                    self._handler_latencies[handler_name] = []
                self._handler_latencies[handler_name].append(latency_ms)

                # Record in metrics
                self._metrics.record_handled(handler_name, latency_ms)

                return

            except asyncio.TimeoutError as e:
                last_error = e
                logger.warning(
                    f"Handler {handler_name} timed out after {self._handler_timeout}s "
                    f"(attempt {attempt + 1}/{self._max_retries})"
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay)

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Handler {handler_name} failed (attempt {attempt + 1}/{self._max_retries}): {e}"
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay)

        # All retries exhausted
        self._events_failed += 1
        self._metrics.record_failed(handler_name, event.event_type)

        if self._enable_dead_letter and last_error:
            # Use enhanced DLQ
            if self._dead_letter_queue:
                await self._dead_letter_queue.add(
                    event=event,
                    handler_name=handler_name,
                    error=last_error,
                    metadata={
                        "initial_attempts": self._max_retries,
                        "total_time_ms": (time.time() - start_time) * 1000,
                    },
                )
            # Also maintain legacy list for backward compatibility
            self._dead_letter.append((event, last_error))
            logger.error(f"Event {event.event_id} sent to dead letter queue: {handler_name}")

    def get_dead_letter_queue(self) -> List[tuple[DomainEvent, Exception]]:
        """Get all events in the dead letter queue (legacy)."""
        return list(self._dead_letter)

    def get_dlq(self) -> Optional[DeadLetterQueue]:
        """Get the enhanced dead letter queue."""
        return self._dead_letter_queue

    def clear_dead_letter_queue(self) -> int:
        """Clear dead letter queue and return count."""
        count = len(self._dead_letter)
        self._dead_letter.clear()
        if self._dead_letter_queue:
            count += self._dead_letter_queue.clear()
        return count

    async def retry_dead_letters(self) -> Tuple[int, int]:
        """
        Retry all events in the dead letter queue.

        Returns:
            Tuple of (succeeded, failed)
        """
        if not self._dead_letter_queue:
            return 0, 0

        # Build handler map
        handler_map: Dict[str, EventHandler] = {}
        for name, handler in self._wildcard_handlers:
            handler_map[name] = handler
        for handlers in self._handlers.values():
            for name, handler in handlers:
                handler_map[name] = handler

        return await self._dead_letter_queue.retry_all(handler_map)

    def get_metrics(self) -> Dict[str, Any]:
        """Get event bus metrics."""
        avg_latencies = {}
        for handler, latencies in self._handler_latencies.items():
            if latencies:
                avg_latencies[handler] = sum(latencies) / len(latencies)

        # Enhanced metrics
        enhanced = self._metrics.get_summary()

        return {
            "events_published": self._events_published,
            "events_handled": self._events_handled,
            "events_failed": self._events_failed,
            "dead_letter_count": len(self._dead_letter) + (
                len(self._dead_letter_queue) if self._dead_letter_queue else 0
            ),
            "registered_handlers": sum(
                len(h) for h in self._handlers.values()
            ) + len(self._wildcard_handlers),
            "event_types_with_handlers": list(self._handlers.keys()),
            "avg_handler_latencies_ms": avg_latencies,
            "enhanced_metrics": enhanced,
            "dlq_statistics": (
                self._dead_letter_queue.get_statistics()
                if self._dead_letter_queue else None
            ),
        }

    def get_handler_map(self) -> Dict[str, EventHandler]:
        """Get map of handler names to handlers for DLQ retry."""
        handler_map: Dict[str, EventHandler] = {}
        for name, handler in self._wildcard_handlers:
            handler_map[name] = handler
        for handlers in self._handlers.values():
            for name, handler in handlers:
                handler_map[name] = handler
        return handler_map


# =============================================================================
# EVENT STORE (EVENT SOURCING)
# =============================================================================

@dataclass
class EventStreamPosition:
    """Position in an event stream."""

    stream_id: str
    version: int
    timestamp: datetime


class EventStore:
    """
    Append-only event store for event sourcing.

    Features:
    - Append events to streams
    - Get events by aggregate ID
    - Replay events for aggregate reconstruction
    - Snapshots for performance
    - Stream positions for consumers

    Thread Safety:
        Safe for concurrent use within a single process.
        For distributed scenarios, use persistent backend.
    """

    def __init__(self) -> None:
        """Initialize in-memory event store."""
        # Events by aggregate_id -> list of events
        self._streams: Dict[str, List[DomainEvent]] = {}

        # Global event log (all events in order)
        self._global_log: List[DomainEvent] = []

        # Snapshots by aggregate_id -> (version, snapshot_data)
        self._snapshots: Dict[str, tuple[int, Dict[str, Any]]] = {}

        # Consumer positions
        self._positions: Dict[str, EventStreamPosition] = {}

    async def append(
        self,
        events: List[DomainEvent],
        expected_version: Optional[int] = None,
    ) -> None:
        """
        Append events to the store.

        Args:
            events: List of events to append
            expected_version: Expected current version (for optimistic concurrency)

        Raises:
            ConcurrencyError: If expected_version doesn't match
        """
        for event in events:
            aggregate_id = event.aggregate_id

            # Initialize stream if needed
            if aggregate_id not in self._streams:
                self._streams[aggregate_id] = []

            stream = self._streams[aggregate_id]

            # Check concurrency
            current_version = len(stream)
            if expected_version is not None and current_version != expected_version:
                raise ConcurrencyError(
                    f"Expected version {expected_version}, got {current_version}"
                )

            # Set version on event
            event.version = current_version + 1

            # Append to stream and global log
            stream.append(event)
            self._global_log.append(event)

            logger.debug(
                f"Appended {event.event_type} to {aggregate_id} (v{event.version})"
            )

    async def get_events(
        self,
        aggregate_id: str,
        after_version: int = 0,
        limit: Optional[int] = None,
    ) -> List[DomainEvent]:
        """
        Get events for an aggregate.

        Args:
            aggregate_id: Aggregate ID to get events for
            after_version: Only return events after this version
            limit: Maximum number of events to return

        Returns:
            List of events in version order
        """
        stream = self._streams.get(aggregate_id, [])
        events = [e for e in stream if e.version > after_version]

        if limit:
            events = events[:limit]

        return events

    async def get_all_events(
        self,
        after_timestamp: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[DomainEvent]:
        """
        Get all events from the global log.

        Args:
            after_timestamp: Only return events after this timestamp
            limit: Maximum number of events

        Returns:
            List of events in chronological order
        """
        events = self._global_log

        if after_timestamp:
            events = [e for e in events if e.timestamp > after_timestamp]

        return events[:limit]

    async def replay(
        self,
        aggregate_id: str,
        handler: Callable[[DomainEvent], None],
        from_version: int = 0,
    ) -> int:
        """
        Replay events for an aggregate.

        Args:
            aggregate_id: Aggregate ID to replay
            handler: Sync handler to call for each event
            from_version: Start replay from this version

        Returns:
            Number of events replayed
        """
        events = await self.get_events(aggregate_id, after_version=from_version)

        for event in events:
            handler(event)

        logger.info(f"Replayed {len(events)} events for {aggregate_id}")
        return len(events)

    async def replay_async(
        self,
        aggregate_id: str,
        handler: EventHandler,
        from_version: int = 0,
    ) -> int:
        """
        Replay events with async handler.

        Args:
            aggregate_id: Aggregate ID to replay
            handler: Async handler to call for each event
            from_version: Start replay from this version

        Returns:
            Number of events replayed
        """
        events = await self.get_events(aggregate_id, after_version=from_version)

        for event in events:
            await handler(event)

        return len(events)

    async def replay_all(
        self,
        handler: EventHandler,
        from_timestamp: Optional[datetime] = None,
        to_timestamp: Optional[datetime] = None,
        event_types: Optional[List[str]] = None,
        batch_size: int = 100,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        metrics: Optional[EventMetrics] = None,
    ) -> int:
        """
        Replay all events from the global log.

        This is useful for rebuilding read models, projections,
        or migrating to new event handlers.

        Args:
            handler: Async handler to call for each event
            from_timestamp: Only replay events after this time
            to_timestamp: Only replay events before this time
            event_types: Only replay these event types (None = all)
            batch_size: Process events in batches for memory efficiency
            progress_callback: Called with (processed, total) during replay
            metrics: Optional metrics to track replay progress

        Returns:
            Number of events replayed
        """
        # Get all events with filters
        all_events = self._global_log

        if from_timestamp:
            all_events = [e for e in all_events if e.timestamp >= from_timestamp]
        if to_timestamp:
            all_events = [e for e in all_events if e.timestamp <= to_timestamp]
        if event_types:
            all_events = [e for e in all_events if e.event_type in event_types]

        total = len(all_events)
        processed = 0

        # Process in batches
        for i in range(0, total, batch_size):
            batch = all_events[i:i + batch_size]

            for event in batch:
                try:
                    await handler(event)
                    processed += 1

                    if metrics:
                        metrics.record_replayed()

                except Exception as e:
                    logger.error(f"Replay failed for event {event.event_id}: {e}")
                    # Continue with next event

            if progress_callback:
                progress_callback(processed, total)

        logger.info(f"Replayed {processed}/{total} events")
        return processed

    async def replay_by_aggregate_type(
        self,
        aggregate_type: str,
        handler: EventHandler,
        from_timestamp: Optional[datetime] = None,
    ) -> int:
        """
        Replay events for all aggregates of a specific type.

        Args:
            aggregate_type: Type of aggregate (e.g., 'Session', 'Memory')
            handler: Async handler to call for each event
            from_timestamp: Only replay events after this time

        Returns:
            Number of events replayed
        """
        events = [
            e for e in self._global_log
            if e.aggregate_type == aggregate_type
            and (from_timestamp is None or e.timestamp >= from_timestamp)
        ]

        for event in events:
            await handler(event)

        logger.info(f"Replayed {len(events)} events for aggregate type {aggregate_type}")
        return len(events)

    async def rebuild_projection(
        self,
        projection_name: str,
        handlers: Dict[str, EventHandler],
        from_timestamp: Optional[datetime] = None,
    ) -> Dict[str, int]:
        """
        Rebuild a projection by replaying relevant events.

        Args:
            projection_name: Name of the projection being rebuilt
            handlers: Map of event_type -> handler
            from_timestamp: Only replay events after this time

        Returns:
            Dict with counts per event type
        """
        counts: Dict[str, int] = {}
        event_types = list(handlers.keys())

        events = [
            e for e in self._global_log
            if e.event_type in event_types
            and (from_timestamp is None or e.timestamp >= from_timestamp)
        ]

        for event in events:
            handler = handlers.get(event.event_type)
            if handler:
                try:
                    await handler(event)
                    counts[event.event_type] = counts.get(event.event_type, 0) + 1
                except Exception as e:
                    logger.error(
                        f"Projection {projection_name} failed on {event.event_id}: {e}"
                    )

        logger.info(
            f"Rebuilt projection {projection_name}: {sum(counts.values())} events"
        )
        return counts

    async def save_snapshot(
        self,
        aggregate_id: str,
        version: int,
        snapshot_data: Dict[str, Any],
    ) -> None:
        """
        Save a snapshot for an aggregate.

        Args:
            aggregate_id: Aggregate ID
            version: Version this snapshot represents
            snapshot_data: Serialized aggregate state
        """
        self._snapshots[aggregate_id] = (version, snapshot_data)
        logger.debug(f"Saved snapshot for {aggregate_id} at v{version}")

    async def get_snapshot(
        self,
        aggregate_id: str,
    ) -> Optional[tuple[int, Dict[str, Any]]]:
        """
        Get the latest snapshot for an aggregate.

        Args:
            aggregate_id: Aggregate ID

        Returns:
            Tuple of (version, snapshot_data) or None
        """
        return self._snapshots.get(aggregate_id)

    async def get_stream_version(self, aggregate_id: str) -> int:
        """Get current version of a stream."""
        stream = self._streams.get(aggregate_id, [])
        return len(stream)

    async def stream_exists(self, aggregate_id: str) -> bool:
        """Check if a stream exists."""
        return aggregate_id in self._streams

    def get_statistics(self) -> Dict[str, Any]:
        """Get event store statistics."""
        return {
            "total_events": len(self._global_log),
            "stream_count": len(self._streams),
            "streams": {
                stream_id: len(events)
                for stream_id, events in self._streams.items()
            },
            "snapshot_count": len(self._snapshots),
        }


class ConcurrencyError(Exception):
    """Raised when optimistic concurrency check fails."""

    pass


# =============================================================================
# TRANSACTIONAL OUTBOX
# =============================================================================

@dataclass
class OutboxMessage:
    """Message in the transactional outbox."""

    message_id: str
    event_type: str
    payload: str  # JSON serialized event
    created_at: datetime
    processed_at: Optional[datetime] = None
    retry_count: int = 0
    last_error: Optional[str] = None

    @property
    def is_processed(self) -> bool:
        return self.processed_at is not None


class TransactionalOutbox:
    """
    Transactional outbox for reliable event publishing.

    Implements the outbox pattern:
    1. Events are persisted to the outbox within the same transaction as state changes
    2. A background worker processes the outbox and publishes to external consumers
    3. Events are marked as processed after successful publishing

    This ensures at-least-once delivery even if the process crashes.
    """

    def __init__(self) -> None:
        """Initialize in-memory outbox."""
        self._messages: Dict[str, OutboxMessage] = {}
        self._processing_lock = asyncio.Lock()

    async def add(self, event: DomainEvent) -> str:
        """
        Add event to the outbox.

        Args:
            event: Domain event to persist

        Returns:
            Message ID
        """
        message_id = str(uuid.uuid4())
        message = OutboxMessage(
            message_id=message_id,
            event_type=event.event_type,
            payload=json.dumps(event.to_dict()),
            created_at=datetime.now(timezone.utc),
        )
        self._messages[message_id] = message
        logger.debug(f"Added event to outbox: {message_id}")
        return message_id

    async def get_pending(
        self,
        limit: int = 100,
        max_age_seconds: Optional[float] = None,
    ) -> List[OutboxMessage]:
        """
        Get pending (unprocessed) messages.

        Args:
            limit: Maximum messages to return
            max_age_seconds: Only return messages older than this

        Returns:
            List of pending messages
        """
        pending = [
            m for m in self._messages.values()
            if not m.is_processed
        ]

        if max_age_seconds:
            cutoff = datetime.now(timezone.utc)
            pending = [
                m for m in pending
                if (cutoff - m.created_at).total_seconds() <= max_age_seconds
            ]

        # Sort by created_at (oldest first)
        pending.sort(key=lambda m: m.created_at)

        return pending[:limit]

    async def mark_processed(
        self,
        message_id: str,
        error: Optional[str] = None,
    ) -> None:
        """
        Mark a message as processed.

        Args:
            message_id: Message ID
            error: Error message if processing failed
        """
        if message_id not in self._messages:
            return

        message = self._messages[message_id]

        if error:
            message.retry_count += 1
            message.last_error = error
        else:
            message.processed_at = datetime.now(timezone.utc)

        logger.debug(
            f"Marked message {message_id} as "
            f"{'processed' if not error else f'failed: {error}'}"
        )

    async def process_batch(
        self,
        publisher: Callable[[DomainEvent], Awaitable[None]],
        limit: int = 100,
        max_retries: int = 3,
    ) -> tuple[int, int]:
        """
        Process a batch of pending messages.

        Args:
            publisher: Async function to publish events externally
            limit: Maximum messages to process
            max_retries: Skip messages with more than this many retries

        Returns:
            Tuple of (processed_count, failed_count)
        """
        async with self._processing_lock:
            pending = await self.get_pending(limit=limit)
            processed = 0
            failed = 0

            for message in pending:
                if message.retry_count >= max_retries:
                    logger.warning(
                        f"Skipping message {message.message_id} "
                        f"(exceeded {max_retries} retries)"
                    )
                    continue

                try:
                    event_data = json.loads(message.payload)
                    event = DomainEvent.from_dict(event_data)
                    await publisher(event)
                    await self.mark_processed(message.message_id)
                    processed += 1

                except Exception as e:
                    await self.mark_processed(message.message_id, error=str(e))
                    failed += 1

            return processed, failed

    async def cleanup(
        self,
        older_than_seconds: float = 86400,  # 24 hours
    ) -> int:
        """
        Remove old processed messages.

        Args:
            older_than_seconds: Remove messages older than this

        Returns:
            Number of messages removed
        """
        cutoff = datetime.now(timezone.utc)
        to_remove = []

        for message_id, message in self._messages.items():
            if not message.is_processed:
                continue
            age = (cutoff - message.created_at).total_seconds()
            if age > older_than_seconds:
                to_remove.append(message_id)

        for message_id in to_remove:
            del self._messages[message_id]

        return len(to_remove)

    def get_statistics(self) -> Dict[str, Any]:
        """Get outbox statistics."""
        total = len(self._messages)
        processed = sum(1 for m in self._messages.values() if m.is_processed)
        pending = total - processed

        return {
            "total_messages": total,
            "processed": processed,
            "pending": pending,
            "failed": sum(
                1 for m in self._messages.values()
                if m.retry_count > 0 and not m.is_processed
            ),
        }


# =============================================================================
# EVENT-SOURCED AGGREGATE BASE
# =============================================================================

T = TypeVar("T", bound="EventSourcedAggregate")


class EventSourcedAggregate(ABC):
    """
    Base class for event-sourced aggregates.

    Subclasses implement:
    - apply_<EventType> methods for each event type
    - create_snapshot/restore_snapshot for snapshot support
    """

    def __init__(self, aggregate_id: str) -> None:
        self._aggregate_id = aggregate_id
        self._version = 0
        self._uncommitted_events: List[DomainEvent] = []

    @property
    def aggregate_id(self) -> str:
        return self._aggregate_id

    @property
    def version(self) -> int:
        return self._version

    @property
    def uncommitted_events(self) -> List[DomainEvent]:
        return list(self._uncommitted_events)

    def apply_event(self, event: DomainEvent) -> None:
        """Apply an event to update aggregate state."""
        method_name = f"apply_{event.event_type}"
        handler = getattr(self, method_name, None)

        if handler:
            handler(event)
        else:
            logger.warning(
                f"No handler for {event.event_type} on {self.__class__.__name__}"
            )

        self._version = event.version

    def raise_event(self, event: DomainEvent) -> None:
        """Raise a new event (for command handling)."""
        event.aggregate_id = self._aggregate_id
        event.version = self._version + 1
        self._uncommitted_events.append(event)
        self.apply_event(event)

    def clear_uncommitted_events(self) -> None:
        """Clear uncommitted events after persistence."""
        self._uncommitted_events.clear()

    @classmethod
    async def load(
        cls: type[T],
        aggregate_id: str,
        event_store: EventStore,
    ) -> T:
        """Load aggregate from event store."""
        aggregate = cls(aggregate_id)

        # Try to load from snapshot first
        snapshot = await event_store.get_snapshot(aggregate_id)
        if snapshot:
            version, snapshot_data = snapshot
            aggregate.restore_snapshot(snapshot_data)
            aggregate._version = version

        # Replay events after snapshot
        events = await event_store.get_events(
            aggregate_id,
            after_version=aggregate._version,
        )

        for event in events:
            aggregate.apply_event(event)

        return aggregate

    async def save(self, event_store: EventStore) -> None:
        """Save uncommitted events to event store."""
        if not self._uncommitted_events:
            return

        await event_store.append(
            self._uncommitted_events,
            expected_version=self._version - len(self._uncommitted_events),
        )
        self.clear_uncommitted_events()

    def create_snapshot(self) -> Dict[str, Any]:
        """Create snapshot of current state (override in subclass)."""
        return {}

    def restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Restore state from snapshot (override in subclass)."""
        pass


# =============================================================================
# MEMORY INTEGRATION
# =============================================================================

class MemoryEventEmitter:
    """
    Event emitter for memory operations.

    Integrates with the memory system to emit domain events
    when memory operations occur.
    """

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus

    async def emit_memory_stored(
        self,
        memory_id: str,
        content: str,
        memory_type: str,
        session_id: str = "",
    ) -> None:
        """Emit event when memory is stored."""
        event = MemoryStoredEvent(
            aggregate_id=session_id or memory_id,
            memory_id=memory_id,
            content_preview=content[:100] if content else "",
            memory_type=memory_type,
        )
        await self._event_bus.publish(event)

    async def emit_memory_retrieved(
        self,
        memory_id: str,
        query: str,
        hit_count: int,
        session_id: str = "",
    ) -> None:
        """Emit event when memory is retrieved."""
        event = MemoryRetrievedEvent(
            aggregate_id=session_id or memory_id,
            memory_id=memory_id,
            query=query,
            hit_count=hit_count,
        )
        await self._event_bus.publish(event)

    async def emit_memory_evicted(
        self,
        memory_id: str,
        reason: str,
        tier: str,
        session_id: str = "",
    ) -> None:
        """Emit event when memory is evicted."""
        event = MemoryEvictedEvent(
            aggregate_id=session_id or memory_id,
            memory_id=memory_id,
            eviction_reason=reason,
            tier=tier,
        )
        await self._event_bus.publish(event)

    async def emit_memory_promoted(
        self,
        memory_id: str,
        from_tier: str,
        to_tier: str,
        reason: str,
        session_id: str = "",
    ) -> None:
        """Emit event when memory is promoted."""
        event = MemoryPromotedEvent(
            aggregate_id=session_id or memory_id,
            memory_id=memory_id,
            from_tier=from_tier,
            to_tier=to_tier,
            reason=reason,
        )
        await self._event_bus.publish(event)


# =============================================================================
# EVENT REGISTRY
# =============================================================================

class EventRegistry:
    """
    Registry for event type metadata and serialization.

    Supports:
    - Event type registration with schema version
    - Event type lookup by name
    - Schema evolution support
    """

    _instance: Optional["EventRegistry"] = None

    def __init__(self) -> None:
        self._event_types: Dict[str, type[DomainEvent]] = {}
        self._schema_versions: Dict[str, str] = {}

    @classmethod
    def instance(cls) -> "EventRegistry":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._register_builtin_events()
        return cls._instance

    def register(
        self,
        event_class: type[DomainEvent],
        schema_version: str = "1.0.0",
    ) -> None:
        """Register an event type."""
        event_type = event_class.__name__
        self._event_types[event_type] = event_class
        self._schema_versions[event_type] = schema_version

    def get_event_class(self, event_type: str) -> Optional[type[DomainEvent]]:
        """Get event class by type name."""
        return self._event_types.get(event_type)

    def get_schema_version(self, event_type: str) -> Optional[str]:
        """Get schema version for event type."""
        return self._schema_versions.get(event_type)

    def deserialize(self, data: Dict[str, Any]) -> DomainEvent:
        """Deserialize event from dictionary using registry."""
        event_type = data.get("event_type", "DomainEvent")
        event_class = self._event_types.get(event_type, DomainEvent)
        return event_class.from_dict(data)

    def _register_builtin_events(self) -> None:
        """Register built-in event types."""
        builtin_events = [
            # Memory events
            MemoryStoredEvent,
            MemoryRetrievedEvent,
            MemoryEvictedEvent,
            MemoryPromotedEvent,
            # Session events
            SessionStartedEvent,
            SessionEndedEvent,
            SessionCheckpointEvent,
            # Flow events
            FlowStartedEvent,
            FlowCompletedEvent,
            FlowFailedEvent,
            # Research events
            ResearchQueryRequestedEvent,
            ResearchResultReceivedEvent,
            ResearchResultCachedEvent,
            ResearchCacheHitEvent,
            ResearchAdapterHealthEvent,
        ]
        for event_class in builtin_events:
            self.register(event_class)


# =============================================================================
# RESEARCH EVENT EMITTER
# =============================================================================

class ResearchEventEmitter:
    """
    Event emitter for research adapter operations.

    Integrates with research adapters (Exa, Tavily, Perplexity, etc.)
    to emit domain events for query tracking, caching, and observability.
    """

    def __init__(
        self,
        event_bus: EventBus,
        aggregator: Optional[EventAggregator] = None,
    ) -> None:
        self._event_bus = event_bus
        self._aggregator = aggregator

    async def emit_query_requested(
        self,
        query: str,
        adapter_name: str,
        operation: str = "search",
        parameters: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """
        Emit event when a research query is requested.

        Returns:
            The request_id for correlation
        """
        event = ResearchQueryRequestedEvent(
            query=query,
            adapter_name=adapter_name,
            operation=operation,
            parameters=parameters or {},
            request_id=request_id or str(uuid.uuid4()),
        )

        if self._aggregator:
            self._aggregator.add(event)
        else:
            await self._event_bus.publish(event)

        return event.request_id

    async def emit_result_received(
        self,
        request_id: str,
        adapter_name: str,
        result_count: int,
        latency_ms: float,
        success: bool = True,
        error_message: Optional[str] = None,
        cost_dollars: Optional[float] = None,
        result_preview: Optional[str] = None,
    ) -> None:
        """Emit event when research results are received."""
        event = ResearchResultReceivedEvent(
            request_id=request_id,
            adapter_name=adapter_name,
            result_count=result_count,
            latency_ms=latency_ms,
            success=success,
            error_message=error_message,
            cost_dollars=cost_dollars,
            result_preview=result_preview or "",
        )

        if self._aggregator:
            self._aggregator.add(event)
        else:
            await self._event_bus.publish(event)

    async def emit_result_cached(
        self,
        request_id: str,
        cache_key: str,
        adapter_name: str,
        result_count: int,
        cache_type: str = "semantic",
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Emit event when research results are cached."""
        event = ResearchResultCachedEvent(
            request_id=request_id,
            cache_key=cache_key,
            cache_type=cache_type,
            ttl_seconds=ttl_seconds,
            adapter_name=adapter_name,
            result_count=result_count,
        )

        if self._aggregator:
            self._aggregator.add(event)
        else:
            await self._event_bus.publish(event)

    async def emit_cache_hit(
        self,
        query: str,
        cache_key: str,
        adapter_name: str,
        cache_type: str = "semantic",
        similarity_score: float = 1.0,
    ) -> None:
        """Emit event when a research query hits the cache."""
        event = ResearchCacheHitEvent(
            query=query,
            cache_key=cache_key,
            cache_type=cache_type,
            adapter_name=adapter_name,
            similarity_score=similarity_score,
        )

        if self._aggregator:
            self._aggregator.add(event)
        else:
            await self._event_bus.publish(event)

    async def emit_adapter_health(
        self,
        adapter_name: str,
        status: str,
        latency_ms: float = 0.0,
        error_rate: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit event for adapter health status changes."""
        event = ResearchAdapterHealthEvent(
            adapter_name=adapter_name,
            status=status,
            latency_ms=latency_ms,
            error_rate=error_rate,
            details=details or {},
        )

        await self._event_bus.publish(event)

    async def flush(self) -> int:
        """Flush aggregated events if using an aggregator."""
        if self._aggregator:
            return await self._aggregator.flush()
        return 0


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Metrics and observability
    "EventMetrics",
    # Versioning
    "EventVersion",
    "EventSchemaEvolution",
    # Base event
    "DomainEvent",
    # Memory events
    "MemoryStoredEvent",
    "MemoryRetrievedEvent",
    "MemoryEvictedEvent",
    "MemoryPromotedEvent",
    # Session events
    "SessionStartedEvent",
    "SessionEndedEvent",
    "SessionCheckpointEvent",
    # Flow events
    "FlowStartedEvent",
    "FlowCompletedEvent",
    "FlowFailedEvent",
    # Research events
    "ResearchQueryRequestedEvent",
    "ResearchResultReceivedEvent",
    "ResearchResultCachedEvent",
    "ResearchCacheHitEvent",
    "ResearchAdapterHealthEvent",
    # Handler types
    "EventHandler",
    "SyncEventHandler",
    "EventHandlerProtocol",
    # Dead letter queue
    "DeadLetterEntry",
    "DeadLetterQueue",
    # Event aggregation
    "EventAggregator",
    # Event bus
    "EventBus",
    # Event sourcing
    "EventStore",
    "EventStreamPosition",
    "EventSourcedAggregate",
    "ConcurrencyError",
    # Outbox pattern
    "OutboxMessage",
    "TransactionalOutbox",
    # Memory integration
    "MemoryEventEmitter",
    # Research integration
    "ResearchEventEmitter",
    # Registry
    "EventRegistry",
]
