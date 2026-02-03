"""
Core orchestration module - V36 Architecture

This module provides the unified orchestration layer for the UNLEASH platform:
- SDKAdapter base class for all adapters
- SDKRegistry for centralized adapter management
- EmbeddingLayer for vector operations
- Infrastructure: Caching, metrics, connection pooling (V5-V8)
- Execution: Batching, backpressure control (V12)

Architecture: 8-Layer SDK Architecture with 39 SDKs
"""

from .base import (
    SDKAdapter,
    SDKLayer,
    AdapterConfig,
    AdapterResult,
    AdapterStatus,
)
from .sdk_registry import (
    SDKRegistry,
    SDKRegistration,
    get_registry,
    register_adapter,
)
from .embedding_layer import (
    EmbeddingLayer,
    EmbeddingConfig,
    EmbeddingModel,
    EmbeddingResult,
    InputType,
    UnleashVectorAdapter,
    QdrantVectorStore,
    create_embedding_layer,
)

# Infrastructure (V5-V8 enhancements)
from .infrastructure import (
    AdaptiveCache,
    SemanticCache,
    PerformanceMetrics,
    AnomalyDetector,
    ConnectionPool,
    RequestDeduplicator,
    WarmupPreloader,
)

# Execution (V12 enhancements)
from .execution import (
    AsyncBatcher,
    BatchItem,
    BackpressureController,
)

__all__ = [
    # Base adapter interface
    "SDKAdapter",
    "SDKLayer",
    "AdapterConfig",
    "AdapterResult",
    "AdapterStatus",
    # Registry
    "SDKRegistry",
    "SDKRegistration",
    "get_registry",
    "register_adapter",
    # Embedding layer
    "EmbeddingLayer",
    "EmbeddingConfig",
    "EmbeddingModel",
    "EmbeddingResult",
    "InputType",
    "UnleashVectorAdapter",
    "QdrantVectorStore",
    "create_embedding_layer",
    # Infrastructure
    "AdaptiveCache",
    "SemanticCache",
    "PerformanceMetrics",
    "AnomalyDetector",
    "ConnectionPool",
    "RequestDeduplicator",
    "WarmupPreloader",
    # Execution
    "AsyncBatcher",
    "BatchItem",
    "BackpressureController",
]
