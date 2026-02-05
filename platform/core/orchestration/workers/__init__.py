"""
Workers Layer - SDK Adapter Base and Worker Interface

V65 Modular Decomposition - Split from ultimate_orchestrator.py

Contains:
- WorkerProtocol: Protocol for worker implementations
- SDKAdapterBase: Base class for SDK adapters
- Worker registry and factory functions
"""

from .base import (
    WorkerProtocol,
    SDKAdapterBase,
    create_adapter,
    AdapterFactory,
)

__all__ = [
    "WorkerProtocol",
    "SDKAdapterBase",
    "create_adapter",
    "AdapterFactory",
]
