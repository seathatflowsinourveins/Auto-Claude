"""
Tests for V65 Modular Orchestration Components

Tests the new modular decomposition:
- domain/value_objects.py
- domain/events.py
- domain/aggregates.py
- workers/base.py
- coordinator.py
- message_bus.py
"""

import asyncio
import sys
import os
import pytest
import time
from datetime import datetime, timezone

# Fix platform module shadowing - add parent to path for proper imports
_platform_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _platform_dir not in sys.path:
    sys.path.insert(0, _platform_dir)


class TestValueObjects:
    """Tests for domain value objects."""

    def test_circuit_state_enum(self):
        """Test CircuitState enum values."""
        from core.orchestration.domain import CircuitState

        assert CircuitState.CLOSED == 0
        assert CircuitState.OPEN == 1
        assert CircuitState.HALF_OPEN == 2

    def test_sdk_layer_enum(self):
        """Test SDKLayer enum has all expected layers."""
        from core.orchestration.domain import SDKLayer

        # Core layers
        assert SDKLayer.OPTIMIZATION
        assert SDKLayer.ORCHESTRATION
        assert SDKLayer.MEMORY
        assert SDKLayer.REASONING
        assert SDKLayer.RESEARCH
        assert SDKLayer.SELF_IMPROVEMENT

        # V18-V27 layers
        assert SDKLayer.STREAMING
        assert SDKLayer.MULTI_MODAL
        assert SDKLayer.SAFETY
        assert SDKLayer.PERSISTENCE
        assert SDKLayer.INFERENCE
        assert SDKLayer.BROWSER_AUTOMATION
        assert SDKLayer.DURABLE_EXECUTION

    def test_execution_priority_enum(self):
        """Test ExecutionPriority enum values."""
        from core.orchestration.domain import ExecutionPriority

        assert ExecutionPriority.CRITICAL.value == 1
        assert ExecutionPriority.HIGH.value == 2
        assert ExecutionPriority.NORMAL.value == 3
        assert ExecutionPriority.LOW.value == 4
        assert ExecutionPriority.BACKGROUND.value == 5

    def test_sdk_config_creation(self):
        """Test SDKConfig dataclass."""
        from core.orchestration.domain import (
            SDKConfig,
            SDKLayer,
            ExecutionPriority,
        )

        config = SDKConfig(
            name="test_sdk",
            layer=SDKLayer.OPTIMIZATION,
            priority=ExecutionPriority.HIGH,
            cache_ttl_seconds=7200,
        )

        assert config.name == "test_sdk"
        assert config.layer == SDKLayer.OPTIMIZATION
        assert config.priority == ExecutionPriority.HIGH
        assert config.cache_ttl_seconds == 7200
        assert config.timeout_ms == 30000  # default
        assert config.max_retries == 3  # default

    def test_sdk_config_empty_name_raises(self):
        """Test SDKConfig validates name is not empty."""
        from core.orchestration.domain import (
            SDKConfig,
            SDKLayer,
        )

        with pytest.raises(ValueError, match="SDK name cannot be empty"):
            SDKConfig(name="", layer=SDKLayer.OPTIMIZATION)

    def test_execution_context_creation(self):
        """Test ExecutionContext dataclass."""
        from core.orchestration.domain import (
            ExecutionContext,
            SDKLayer,
            ExecutionPriority,
        )

        ctx = ExecutionContext(
            request_id="test-123",
            layer=SDKLayer.MEMORY,
            priority=ExecutionPriority.CRITICAL,
            deadline_ms=5000.0,
        )

        assert ctx.request_id == "test-123"
        assert ctx.layer == SDKLayer.MEMORY
        assert ctx.priority == ExecutionPriority.CRITICAL
        assert ctx.deadline_ms == 5000.0
        assert ctx.elapsed_ms >= 0
        assert ctx.remaining_ms is not None
        assert not ctx.is_expired

    def test_execution_result_success(self):
        """Test ExecutionResult success factory."""
        from core.orchestration.domain import (
            ExecutionResult,
            SDKLayer,
        )

        result = ExecutionResult.success_result(
            data={"key": "value"},
            layer=SDKLayer.RESEARCH,
            adapter_name="firecrawl",
            latency_ms=150.0,
            cached=True,
        )

        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.layer == SDKLayer.RESEARCH
        assert result.adapter_name == "firecrawl"
        assert result.latency_ms == 150.0
        assert result.cached is True
        assert result.error is None

    def test_execution_result_failure(self):
        """Test ExecutionResult failure factory."""
        from core.orchestration.domain import (
            ExecutionResult,
            SDKLayer,
        )

        result = ExecutionResult.failure_result(
            error="Connection timeout",
            layer=SDKLayer.MEMORY,
            adapter_name="zep",
            latency_ms=5000.0,
        )

        assert result.success is False
        assert result.error == "Connection timeout"
        assert result.layer == SDKLayer.MEMORY
        assert result.adapter_name == "zep"
        assert result.latency_ms == 5000.0
        assert result.data is None

    def test_execution_result_to_dict(self):
        """Test ExecutionResult serialization."""
        from core.orchestration.domain import (
            ExecutionResult,
            SDKLayer,
        )

        result = ExecutionResult.success_result(
            data="test",
            layer=SDKLayer.OPTIMIZATION,
        )

        data = result.to_dict()
        assert data["success"] is True
        assert data["data"] == "test"
        assert data["layer"] == "OPTIMIZATION"
        assert "timestamp" in data


class TestDomainEvents:
    """Tests for domain events."""

    def test_execution_started_event(self):
        """Test ExecutionStartedEvent creation."""
        from core.orchestration.domain import (
            ExecutionStartedEvent,
            SDKLayer,
            ExecutionPriority,
        )

        event = ExecutionStartedEvent(
            request_id="req-123",
            layer=SDKLayer.REASONING,
            operation="predict",
            priority=ExecutionPriority.HIGH,
            adapter_name="dspy",
        )

        assert event.event_type == "ExecutionStartedEvent"
        assert event.request_id == "req-123"
        assert event.layer == SDKLayer.REASONING
        assert event.operation == "predict"
        assert len(event.event_id) == 16

    def test_execution_completed_event(self):
        """Test ExecutionCompletedEvent creation."""
        from core.orchestration.domain import (
            ExecutionCompletedEvent,
            SDKLayer,
        )

        event = ExecutionCompletedEvent(
            request_id="req-123",
            layer=SDKLayer.RESEARCH,
            operation="scrape",
            adapter_name="firecrawl",
            latency_ms=250.0,
            cached=True,
        )

        assert event.event_type == "ExecutionCompletedEvent"
        assert event.latency_ms == 250.0
        assert event.cached is True

    def test_execution_failed_event(self):
        """Test ExecutionFailedEvent creation."""
        from core.orchestration.domain import (
            ExecutionFailedEvent,
            SDKLayer,
        )

        event = ExecutionFailedEvent(
            request_id="req-123",
            layer=SDKLayer.MEMORY,
            error="Connection refused",
            error_type="ConnectionError",
            retry_count=3,
            is_retryable=False,
        )

        assert event.event_type == "ExecutionFailedEvent"
        assert event.error == "Connection refused"
        assert event.retry_count == 3
        assert event.is_retryable is False

    def test_adapter_health_changed_event(self):
        """Test AdapterHealthChangedEvent creation."""
        from core.orchestration.domain import (
            AdapterHealthChangedEvent,
            SDKLayer,
        )

        event = AdapterHealthChangedEvent(
            adapter_name="zep",
            layer=SDKLayer.MEMORY,
            previous_state="CLOSED",
            new_state="OPEN",
            health_score=0.3,
            failure_count=5,
            reason="Threshold exceeded",
        )

        assert event.adapter_name == "zep"
        assert event.is_degraded is True
        assert event.is_recovered is False

    def test_adapter_health_recovery(self):
        """Test AdapterHealthChangedEvent recovery detection."""
        from core.orchestration.domain import AdapterHealthChangedEvent

        event = AdapterHealthChangedEvent(
            adapter_name="test",
            previous_state="OPEN",
            new_state="CLOSED",
            health_score=0.95,
        )

        assert event.is_recovered is True
        assert event.is_degraded is False


class TestAggregates:
    """Tests for domain aggregates."""

    def test_execution_session_creation(self):
        """Test ExecutionSession creation."""
        from core.orchestration.domain import ExecutionSession

        session = ExecutionSession()

        assert len(session.session_id) == 12
        assert session.total_executions == 0
        assert session.success_rate == 100.0
        assert session.avg_latency_ms == 0.0
        assert session.version == 0

    def test_execution_session_with_id(self):
        """Test ExecutionSession with custom ID."""
        from core.orchestration.domain import ExecutionSession

        session = ExecutionSession(session_id="custom-id-123")
        assert session.session_id == "custom-id-123"

    def test_execution_session_start_execution(self):
        """Test starting an execution in session."""
        from core.orchestration.domain import (
            ExecutionSession,
            SDKLayer,
            ExecutionPriority,
        )

        session = ExecutionSession()
        ctx = session.start_execution(
            request_id="req-1",
            layer=SDKLayer.OPTIMIZATION,
            operation="predict",
            priority=ExecutionPriority.HIGH,
        )

        assert ctx.request_id == "req-1"
        assert ctx.layer == SDKLayer.OPTIMIZATION
        assert len(session.pending_events) == 1

    def test_execution_session_complete_success(self):
        """Test completing a successful execution."""
        from core.orchestration.domain import (
            ExecutionSession,
            ExecutionResult,
            SDKLayer,
        )

        session = ExecutionSession()
        ctx = session.start_execution(
            request_id="req-1",
            layer=SDKLayer.MEMORY,
            operation="add",
        )

        result = ExecutionResult.success_result(
            data="stored",
            layer=SDKLayer.MEMORY,
            latency_ms=50.0,
        )

        session.complete_execution(ctx, result, "add")

        assert session.total_executions == 1
        assert session.successful_executions == 1
        assert session.success_rate == 100.0
        assert session.version == 1

    def test_execution_session_complete_failure(self):
        """Test completing a failed execution."""
        from core.orchestration.domain import (
            ExecutionSession,
            ExecutionResult,
            SDKLayer,
        )

        session = ExecutionSession()
        ctx = session.start_execution(
            request_id="req-1",
            layer=SDKLayer.RESEARCH,
            operation="scrape",
        )

        result = ExecutionResult.failure_result(
            error="Timeout",
            layer=SDKLayer.RESEARCH,
        )

        session.complete_execution(ctx, result, "scrape")

        assert session.total_executions == 1
        assert session.successful_executions == 0
        assert session.success_rate == 0.0

    def test_execution_session_drain_events(self):
        """Test draining events from session."""
        from core.orchestration.domain import (
            ExecutionSession,
            SDKLayer,
        )

        session = ExecutionSession()
        session.start_execution(
            request_id="req-1",
            layer=SDKLayer.OPTIMIZATION,
            operation="predict",
        )

        events = session.drain_events()
        assert len(events) == 1
        assert len(session.pending_events) == 0

    def test_adapter_aggregate_creation(self):
        """Test AdapterAggregate creation."""
        from core.orchestration.domain import (
            AdapterAggregate,
            SDKConfig,
            SDKLayer,
            CircuitState,
        )

        config = SDKConfig(name="test", layer=SDKLayer.MEMORY)
        aggregate = AdapterAggregate(config=config)

        assert aggregate.name == "test"
        assert aggregate.layer == SDKLayer.MEMORY
        assert aggregate.circuit_state == CircuitState.CLOSED
        assert aggregate.is_available is True
        assert aggregate.health_score == 1.0

    def test_adapter_aggregate_record_success(self):
        """Test recording success on adapter aggregate."""
        from core.orchestration.domain import (
            AdapterAggregate,
            SDKConfig,
            SDKLayer,
        )

        config = SDKConfig(name="test", layer=SDKLayer.MEMORY)
        aggregate = AdapterAggregate(config=config)

        aggregate.record_success(latency_ms=50.0)

        assert aggregate.avg_latency_ms == 50.0
        stats = aggregate.get_stats()
        assert stats["success_count"] == 1

    def test_adapter_aggregate_circuit_breaker(self):
        """Test circuit breaker transitions."""
        from core.orchestration.domain import (
            AdapterAggregate,
            SDKConfig,
            SDKLayer,
            CircuitState,
        )

        config = SDKConfig(name="test", layer=SDKLayer.MEMORY)
        aggregate = AdapterAggregate(
            config=config,
            failure_threshold=3,
        )

        # Record failures to open circuit
        for _ in range(3):
            aggregate.record_failure(error="Connection refused")

        assert aggregate.circuit_state == CircuitState.OPEN
        assert aggregate.is_available is False

        # Check pending events
        events = aggregate.drain_events()
        assert len(events) >= 1


class TestMessageBus:
    """Tests for the message bus."""

    def test_message_bus_creation(self):
        """Test MessageBus creation."""
        from core.orchestration.message_bus import MessageBus

        bus = MessageBus()
        assert bus.config.max_queue_size == 10000

    def test_message_bus_subscribe(self):
        """Test subscribing to events."""
        from core.orchestration.message_bus import MessageBus

        bus = MessageBus()
        received = []

        def handler(event):
            received.append(event)

        bus.subscribe("TestEvent", handler)
        bus.publish_sync({"data": "test"}, "TestEvent")

        assert len(received) == 1
        assert received[0]["data"] == "test"

    def test_message_bus_unsubscribe(self):
        """Test unsubscribing from events."""
        from core.orchestration.message_bus import MessageBus

        bus = MessageBus()
        received = []

        def handler(event):
            received.append(event)

        bus.subscribe("TestEvent", handler)
        result = bus.unsubscribe("TestEvent", handler)

        assert result is True
        bus.publish_sync({"data": "test"}, "TestEvent")
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_message_bus_async_publish(self):
        """Test async event publishing."""
        from core.orchestration.message_bus import MessageBus

        bus = MessageBus()
        result = await bus.publish({"data": "async"}, "AsyncEvent")

        assert result is True
        stats = bus.get_stats()
        assert stats["published_count"] == 1

    @pytest.mark.asyncio
    async def test_message_bus_process_all(self):
        """Test processing all events."""
        from core.orchestration.message_bus import MessageBus

        bus = MessageBus()
        received = []

        async def async_handler(event):
            received.append(event)

        bus.subscribe("TestEvent", async_handler)
        await bus.publish({"id": 1}, "TestEvent")
        await bus.publish({"id": 2}, "TestEvent")

        processed = await bus.process_all()
        assert processed == 2
        assert len(received) == 2

    def test_message_bus_stats(self):
        """Test message bus statistics."""
        from core.orchestration.message_bus import MessageBus

        bus = MessageBus()
        stats = bus.get_stats()

        assert "queue_size" in stats
        assert "published_count" in stats
        assert "processed_count" in stats
        assert "failed_count" in stats


class TestCoordinator:
    """Tests for the coordinator."""

    def test_coordinator_creation(self):
        """Test Coordinator creation."""
        from core.orchestration.coordinator import Coordinator

        coord = Coordinator()
        assert coord.is_initialized is False
        assert len(coord.session_id) == 12

    @pytest.mark.asyncio
    async def test_coordinator_initialization(self):
        """Test Coordinator initialization."""
        from core.orchestration.coordinator import Coordinator

        coord = Coordinator()
        result = await coord.initialize()

        assert result is True
        assert coord.is_initialized is True

    def test_coordinator_get_available_layers(self):
        """Test getting available layers."""
        from core.orchestration.coordinator import Coordinator

        coord = Coordinator()
        layers = coord.get_available_layers()

        # Initially no layers available (no adapters registered)
        assert isinstance(layers, list)

    def test_coordinator_stats(self):
        """Test coordinator statistics."""
        from core.orchestration.coordinator import Coordinator

        coord = Coordinator()
        stats = coord.get_stats()

        assert "session_id" in stats
        assert "initialized" in stats
        assert "total_executions" in stats
        assert "success_rate" in stats
        assert "layers" in stats

    @pytest.mark.asyncio
    async def test_coordinator_shutdown(self):
        """Test Coordinator shutdown."""
        from core.orchestration.coordinator import Coordinator

        coord = Coordinator()
        await coord.initialize()
        await coord.shutdown()

        assert coord.is_initialized is False


class TestIntegration:
    """Integration tests for modular orchestration."""

    def test_all_imports_work(self):
        """Test all modular imports work correctly."""
        from core.orchestration import (
            # Domain
            CircuitState,
            ExecutionPriority,
            SDKConfig,
            ExecutionContext,
            ExecutionResult,
            ExecutionStartedEvent,
            ExecutionCompletedEvent,
            ExecutionFailedEvent,
            AdapterHealthChangedEvent,
            ExecutionSession,
            AdapterAggregate,
            # Workers
            WorkerProtocol,
            SDKAdapterBase,
            AdapterFactory,
            # Coordinator
            Coordinator,
            CoordinatorConfig,
            get_coordinator,
            # Message Bus
            MessageBus,
            MessageBusConfig,
            get_message_bus,
        )

        # Just verify all imports succeeded
        assert CircuitState.CLOSED == 0
        assert ExecutionPriority.NORMAL.value == 3
        assert SDKConfig is not None
        assert Coordinator is not None
        assert MessageBus is not None

    def test_backward_compatibility(self):
        """Test backward compatibility with ultimate_orchestrator."""
        # These imports should still work from ultimate_orchestrator
        from core.ultimate_orchestrator import (
            CircuitState,
            SDKLayer,
            ExecutionPriority,
            SDKConfig,
            ExecutionContext,
            ExecutionResult,
            UltimateOrchestrator,
        )

        # Verify types work
        assert CircuitState.CLOSED == 0
        assert SDKLayer.OPTIMIZATION is not None
        assert ExecutionPriority.HIGH.value == 2
