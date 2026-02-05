"""
Aggregates - Transactional Boundaries for Complex Domain Logic

V65 Modular Decomposition - Extracted from ultimate_orchestrator.py

Contains:
- ExecutionSession: Aggregate root for execution sessions
- AdapterAggregate: Aggregate for adapter state management
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .value_objects import (
    CircuitState,
    SDKLayer,
    ExecutionPriority,
    SDKConfig,
    ExecutionContext,
    ExecutionResult,
)
from .events import (
    ExecutionStartedEvent,
    ExecutionCompletedEvent,
    ExecutionFailedEvent,
    AdapterHealthChangedEvent,
)


@dataclass
class ExecutionSession:
    """
    Aggregate root for execution sessions.

    Tracks execution history, manages session state, and enforces
    consistency rules for the session lifecycle.
    """
    session_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    execution_history: List[ExecutionResult] = field(default_factory=list)
    pending_events: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _version: int = 0

    def __post_init__(self):
        if not self.session_id:
            self.session_id = hashlib.md5(
                str(time.time()).encode()
            ).hexdigest()[:12]

    @property
    def version(self) -> int:
        """Get current version for optimistic concurrency."""
        return self._version

    @property
    def total_executions(self) -> int:
        """Get total number of executions."""
        return len(self.execution_history)

    @property
    def successful_executions(self) -> int:
        """Get number of successful executions."""
        return sum(1 for r in self.execution_history if r.success)

    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if not self.execution_history:
            return 100.0
        return (self.successful_executions / self.total_executions) * 100

    @property
    def avg_latency_ms(self) -> float:
        """Get average latency in milliseconds."""
        if not self.execution_history:
            return 0.0
        return sum(r.latency_ms for r in self.execution_history) / len(self.execution_history)

    def start_execution(
        self,
        request_id: str,
        layer: SDKLayer,
        operation: str,
        priority: ExecutionPriority = ExecutionPriority.NORMAL,
        adapter_name: Optional[str] = None
    ) -> ExecutionContext:
        """
        Start a new execution and emit started event.

        Returns an ExecutionContext for the request.
        """
        ctx = ExecutionContext(
            request_id=request_id,
            layer=layer,
            priority=priority
        )

        event = ExecutionStartedEvent(
            request_id=request_id,
            layer=layer,
            operation=operation,
            priority=priority,
            adapter_name=adapter_name
        )
        self.pending_events.append(event)

        return ctx

    def complete_execution(
        self,
        ctx: ExecutionContext,
        result: ExecutionResult,
        operation: str = ""
    ) -> None:
        """
        Complete an execution and emit appropriate event.

        Records the result in history and emits completed/failed event.
        """
        self.execution_history.append(result)
        self._version += 1

        if result.success:
            event = ExecutionCompletedEvent(
                request_id=ctx.request_id,
                layer=ctx.layer,
                operation=operation,
                adapter_name=result.adapter_name,
                latency_ms=result.latency_ms,
                cached=result.cached,
                result_size_bytes=len(str(result.data)) if result.data else 0
            )
        else:
            event = ExecutionFailedEvent(
                request_id=ctx.request_id,
                layer=ctx.layer,
                operation=operation,
                adapter_name=result.adapter_name,
                error=result.error or "Unknown error",
                latency_ms=result.latency_ms
            )

        self.pending_events.append(event)

    def get_layer_stats(self, layer: SDKLayer) -> Dict[str, Any]:
        """Get statistics for a specific layer."""
        layer_results = [r for r in self.execution_history if r.layer == layer]
        if not layer_results:
            return {
                "layer": layer.name,
                "total": 0,
                "success": 0,
                "success_rate": 100.0,
                "avg_latency_ms": 0.0
            }

        successful = sum(1 for r in layer_results if r.success)
        return {
            "layer": layer.name,
            "total": len(layer_results),
            "success": successful,
            "success_rate": (successful / len(layer_results)) * 100,
            "avg_latency_ms": sum(r.latency_ms for r in layer_results) / len(layer_results)
        }

    def get_recent_results(self, limit: int = 10) -> List[ExecutionResult]:
        """Get most recent execution results."""
        return self.execution_history[-limit:]

    def drain_events(self) -> List[Any]:
        """Drain and return all pending events."""
        events = self.pending_events.copy()
        self.pending_events.clear()
        return events

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "version": self._version,
            "total_executions": self.total_executions,
            "success_rate": self.success_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "metadata": self.metadata
        }


@dataclass
class AdapterAggregate:
    """
    Aggregate for adapter state management.

    Manages circuit breaker state, health metrics, and failover logic.
    """
    config: SDKConfig
    circuit_state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    half_open_successes: int = 0
    latency_history: List[float] = field(default_factory=list)
    pending_events: List[Any] = field(default_factory=list)
    _call_count: int = 0
    _initialized: bool = False

    # Circuit breaker configuration
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0
    half_open_max_calls: int = 3
    max_latency_samples: int = 100

    @property
    def name(self) -> str:
        """Get adapter name."""
        return self.config.name

    @property
    def layer(self) -> SDKLayer:
        """Get adapter layer."""
        return self.config.layer

    @property
    def is_available(self) -> bool:
        """Check if adapter is available for requests."""
        if self.circuit_state == CircuitState.CLOSED:
            return True

        if self.circuit_state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time > self.recovery_timeout_seconds:
                self._transition_to_half_open()
                return True
            return False

        # HALF_OPEN state
        return True

    @property
    def health_score(self) -> float:
        """Calculate health score (0.0 to 1.0)."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0
        return self.success_count / total

    @property
    def avg_latency_ms(self) -> float:
        """Get average latency."""
        if not self.latency_history:
            return 0.0
        return sum(self.latency_history) / len(self.latency_history)

    def record_success(self, latency_ms: float = 0.0) -> None:
        """Record a successful call."""
        self._call_count += 1
        self.success_count += 1
        self._record_latency(latency_ms)

        previous_state = self.circuit_state.name

        if self.circuit_state == CircuitState.HALF_OPEN:
            self.half_open_successes += 1
            if self.half_open_successes >= self.half_open_max_calls:
                self._transition_to_closed()
                self._emit_health_event(previous_state, "CLOSED", "Recovery confirmed")
        elif self.circuit_state == CircuitState.CLOSED:
            # Reset failure count on success
            if self.failure_count > 0:
                self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self, latency_ms: float = 0.0, error: str = "") -> None:
        """Record a failed call."""
        self._call_count += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        self._record_latency(latency_ms)

        previous_state = self.circuit_state.name

        if self.circuit_state == CircuitState.HALF_OPEN:
            # Any failure in half-open reopens circuit
            self._transition_to_open()
            self._emit_health_event(previous_state, "OPEN", f"Recovery failed: {error}")
        elif self.failure_count >= self.failure_threshold:
            self._transition_to_open()
            self._emit_health_event(
                previous_state,
                "OPEN",
                f"Threshold exceeded ({self.failure_count} failures)"
            )

    def _record_latency(self, latency_ms: float) -> None:
        """Record latency sample."""
        self.latency_history.append(latency_ms)
        if len(self.latency_history) > self.max_latency_samples:
            self.latency_history = self.latency_history[-self.max_latency_samples:]

    def _transition_to_open(self) -> None:
        """Transition to OPEN state."""
        self.circuit_state = CircuitState.OPEN
        self.half_open_successes = 0

    def _transition_to_half_open(self) -> None:
        """Transition to HALF_OPEN state."""
        self.circuit_state = CircuitState.HALF_OPEN
        self.half_open_successes = 0

    def _transition_to_closed(self) -> None:
        """Transition to CLOSED state."""
        self.circuit_state = CircuitState.CLOSED
        self.failure_count = 0
        self.half_open_successes = 0

    def _emit_health_event(
        self,
        previous_state: str,
        new_state: str,
        reason: str
    ) -> None:
        """Emit health changed event."""
        event = AdapterHealthChangedEvent(
            adapter_name=self.name,
            layer=self.layer,
            previous_state=previous_state,
            new_state=new_state,
            health_score=self.health_score,
            failure_count=self.failure_count,
            success_count=self.success_count,
            avg_latency_ms=self.avg_latency_ms,
            reason=reason
        )
        self.pending_events.append(event)

    def reset(self) -> None:
        """Reset adapter state."""
        previous_state = self.circuit_state.name
        self.circuit_state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_successes = 0
        self.latency_history.clear()

        if previous_state != "CLOSED":
            self._emit_health_event(previous_state, "CLOSED", "Manual reset")

    def drain_events(self) -> List[Any]:
        """Drain and return all pending events."""
        events = self.pending_events.copy()
        self.pending_events.clear()
        return events

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return {
            "name": self.name,
            "layer": self.layer.name,
            "circuit_state": self.circuit_state.name,
            "is_available": self.is_available,
            "health_score": round(self.health_score, 3),
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "call_count": self._call_count,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "initialized": self._initialized
        }
