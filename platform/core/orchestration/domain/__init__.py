"""
Domain Layer - Value Objects, Events, and Aggregates

This module contains domain-driven design components for the orchestration layer:
- Value Objects: Immutable domain concepts (ExecutionResult, Metrics, etc.)
- Events: Domain events for event-driven architecture
- Aggregates: Transactional boundaries for complex domain logic

V65 Modular Decomposition - Split from ultimate_orchestrator.py
"""

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

from .aggregates import (
    ExecutionSession,
    AdapterAggregate,
)

__all__ = [
    # Value Objects
    "CircuitState",
    "SDKLayer",
    "ExecutionPriority",
    "SDKConfig",
    "ExecutionContext",
    "ExecutionResult",
    # Events
    "ExecutionStartedEvent",
    "ExecutionCompletedEvent",
    "ExecutionFailedEvent",
    "AdapterHealthChangedEvent",
    # Aggregates
    "ExecutionSession",
    "AdapterAggregate",
]
