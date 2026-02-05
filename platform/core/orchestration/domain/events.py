"""
Domain Events - Execution-specific Events

V65 Modular Decomposition - Extracted from ultimate_orchestrator.py

Contains execution-related domain events:
- ExecutionStartedEvent: Fired when execution begins
- ExecutionCompletedEvent: Fired when execution completes successfully
- ExecutionFailedEvent: Fired when execution fails
- AdapterHealthChangedEvent: Fired when adapter health changes
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .value_objects import SDKLayer, ExecutionPriority


@dataclass
class ExecutionStartedEvent:
    """
    Event fired when an execution begins.

    Contains context about the request being processed.
    """
    event_type: str = "ExecutionStartedEvent"
    request_id: str = ""
    layer: Optional[SDKLayer] = None
    operation: str = ""
    priority: ExecutionPriority = ExecutionPriority.NORMAL
    adapter_name: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def event_id(self) -> str:
        """Generate unique event ID."""
        data = f"{self.event_type}:{self.request_id}:{self.timestamp.isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_type": self.event_type,
            "event_id": self.event_id,
            "request_id": self.request_id,
            "layer": self.layer.name if self.layer else None,
            "operation": self.operation,
            "priority": self.priority.name,
            "adapter_name": self.adapter_name,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class ExecutionCompletedEvent:
    """
    Event fired when an execution completes successfully.

    Contains the result and performance metrics.
    """
    event_type: str = "ExecutionCompletedEvent"
    request_id: str = ""
    layer: Optional[SDKLayer] = None
    operation: str = ""
    adapter_name: Optional[str] = None
    latency_ms: float = 0.0
    cached: bool = False
    result_size_bytes: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def event_id(self) -> str:
        """Generate unique event ID."""
        data = f"{self.event_type}:{self.request_id}:{self.timestamp.isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_type": self.event_type,
            "event_id": self.event_id,
            "request_id": self.request_id,
            "layer": self.layer.name if self.layer else None,
            "operation": self.operation,
            "adapter_name": self.adapter_name,
            "latency_ms": self.latency_ms,
            "cached": self.cached,
            "result_size_bytes": self.result_size_bytes,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class ExecutionFailedEvent:
    """
    Event fired when an execution fails.

    Contains error details and context for debugging.
    """
    event_type: str = "ExecutionFailedEvent"
    request_id: str = ""
    layer: Optional[SDKLayer] = None
    operation: str = ""
    adapter_name: Optional[str] = None
    error: str = ""
    error_type: str = ""
    latency_ms: float = 0.0
    retry_count: int = 0
    is_retryable: bool = True
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def event_id(self) -> str:
        """Generate unique event ID."""
        data = f"{self.event_type}:{self.request_id}:{self.timestamp.isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_type": self.event_type,
            "event_id": self.event_id,
            "request_id": self.request_id,
            "layer": self.layer.name if self.layer else None,
            "operation": self.operation,
            "adapter_name": self.adapter_name,
            "error": self.error,
            "error_type": self.error_type,
            "latency_ms": self.latency_ms,
            "retry_count": self.retry_count,
            "is_retryable": self.is_retryable,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class AdapterHealthChangedEvent:
    """
    Event fired when adapter health status changes.

    Tracks circuit breaker state changes and health metrics.
    """
    event_type: str = "AdapterHealthChangedEvent"
    adapter_name: str = ""
    layer: Optional[SDKLayer] = None
    previous_state: str = ""  # CLOSED, OPEN, HALF_OPEN
    new_state: str = ""
    health_score: float = 1.0  # 0.0 to 1.0
    failure_count: int = 0
    success_count: int = 0
    avg_latency_ms: float = 0.0
    reason: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def event_id(self) -> str:
        """Generate unique event ID."""
        data = f"{self.event_type}:{self.adapter_name}:{self.timestamp.isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]

    @property
    def is_degraded(self) -> bool:
        """Check if adapter is in degraded state."""
        return self.new_state in ("OPEN", "HALF_OPEN") or self.health_score < 0.5

    @property
    def is_recovered(self) -> bool:
        """Check if adapter has recovered."""
        return self.previous_state in ("OPEN", "HALF_OPEN") and self.new_state == "CLOSED"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_type": self.event_type,
            "event_id": self.event_id,
            "adapter_name": self.adapter_name,
            "layer": self.layer.name if self.layer else None,
            "previous_state": self.previous_state,
            "new_state": self.new_state,
            "health_score": self.health_score,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "avg_latency_ms": self.avg_latency_ms,
            "reason": self.reason,
            "is_degraded": self.is_degraded,
            "is_recovered": self.is_recovered,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
