"""
Core orchestration module - V40 Architecture (ADR-029, ADR-030)

This module provides the unified orchestration layer for the UNLEASH platform.
All exports are directly imported for optimal IDE support.

Exported Components:
--------------------

Core Classes:
- AgentCommSystem: Memory-based agent-to-agent messaging
- ExtremeSwarmController: Advanced multi-agent swarm orchestration
- DomainEvent: Base class for all domain events
- EventBus: Pub/sub event distribution
- EventStore: Event sourcing persistence

Infrastructure:
- SDKAdapter: Base class for all SDK adapters
- SDKRegistry: Centralized adapter management
- EmbeddingLayer: Vector operations and embeddings
- AdaptiveCache, SemanticCache: Intelligent caching
- ConnectionPool: Connection pooling

Execution:
- AsyncBatcher: Request batching
- BackpressureController: Load management
- FlowStateMachine: Formal state management

Message Types and Events:
- AgentMessage: Agent-to-agent message
- MessageType, MessageStatus, MessagePriority: Message enums
- All domain events (Memory*, Session*, Flow*, Research*)

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

# Flow State Machine (V38 enhancements - ADR-028)
from .flow_state_machine import (
    FlowState,
    FlowTransition,
    FlowStateMachine,
    FlowContext,
    FlowExecutor,
    StateHistoryEntry,
)

# Saga Orchestration (V36 - ADR-029)
from .saga import (
    SagaState,
    StepState,
    SagaStep,
    StepResult,
    Saga,
    SagaResult,
    SagaStateStore,
    SagaMetrics,
    SagaOrchestrator,
    create_multi_adapter_saga,
    example_order_processing_saga,
)

# Bulkhead Pattern (V38 - ADR-029)
from .bulkhead import (
    Bulkhead,
    BulkheadConfig,
    BulkheadRegistry,
    BulkheadStats,
    BulkheadState,
    BulkheadError,
    BulkheadFullError,
    BulkheadTimeoutError,
    BulkheadRejectionReason,
    get_bulkhead,
    get_bulkhead_registry,
    create_database_bulkhead,
    create_api_bulkhead,
    create_mcp_bulkhead,
)

# Batch Embedding Pipeline (V39)
from .batch_embedding import (
    BatchEmbeddingPipeline,
    EmbeddingRequest,
    EmbeddingResult as BatchEmbeddingResult,
    EmbeddingCache,
    EmbeddingProvider,
    ProviderModel,
    OpenAIEmbedder,
    VoyageEmbedder,
    JinaEmbedder,
    LocalEmbedder,
    BaseEmbedder,
    create_batch_pipeline,
    quick_embed,
)

# Domain Events (V40 - ADR-029 Enhanced)
from .domain_events import (
    # Metrics and observability
    EventMetrics,
    # Versioning
    EventVersion,
    EventSchemaEvolution,
    # Base event
    DomainEvent,
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
    # Handler types
    EventHandler,
    SyncEventHandler,
    # Dead letter queue
    DeadLetterEntry,
    DeadLetterQueue,
    # Event aggregation
    EventAggregator,
    # Event bus
    EventBus,
    # Event sourcing
    EventStore,
    EventStreamPosition,
    EventSourcedAggregate,
    ConcurrencyError,
    # In-memory outbox pattern
    OutboxMessage,
    TransactionalOutbox as InMemoryTransactionalOutbox,
    # Memory integration (event bus based)
    MemoryEventEmitter as EventBusMemoryEventEmitter,
    # Research integration
    ResearchEventEmitter,
    # Registry
    EventRegistry,
)

# Research Events Integration (V40)
from .research_events_integration import (
    EventAwareResearchAdapter,
    ResearchEventHandler,
    ResearchEventProjection,
    create_research_event_system,
)

# Persistent Transactional Outbox (V39 - SQLite-based reliable event publishing)
from .outbox import (
    # Event types and status
    EventStatus,
    MemoryEventTypes,
    # Core classes
    OutboxEvent,
    TransactionalOutbox,
    OutboxPublisher,
    # Memory integration (outbox based)
    MemoryEventEmitter,
    # Helpers
    create_outbox,
)

# Cache Warming (V39 - ADR-029)
from .cache_warming import (
    # Core classes
    CacheWarmer,
    WarmingStrategy,
    WarmingResult,
    UsagePattern,
    # Enums
    WarmingPriority,
    WarmingStatus,
    # Pre-built strategies
    warm_mcp_connections,
    warm_embedding_cache,
    warm_memory_context,
    warm_research_cache,
    warm_sdk_adapters,
    warm_semantic_cache,
    # Factory functions
    create_default_warmer,
    create_minimal_warmer,
    # Singleton
    get_cache_warmer,
    set_cache_warmer,
)

# Agent Communication (V40 - Memory-based agent-to-agent messaging)
from .agent_comm import (
    # Constants
    LARGE_PAYLOAD_THRESHOLD,
    DEFAULT_MESSAGE_TTL_SECONDS,
    # Enums
    MessageType,
    MessageStatus,
    MessagePriority,
    # Data classes
    AgentMessage,
    MessageAcknowledgment,
    TopicSubscription,
    InboxStats,
    # Events
    MessageSentEvent,
    MessageReceivedEvent,
    MessageAcknowledgedEvent,
    BroadcastSentEvent,
    # Store
    SQLiteMessageStore,
    # Main system
    AgentCommSystem,
    # Factory
    create_agent_comm_system,
)

# Extreme Swarm Patterns (V40 - ADR-030)
from .extreme_swarm import (
    # Enums
    SwarmRole,
    AgentState,
    StrategyState,
    HealthStatus,
    # Core Types
    TokenBudget,
    AgentMetadata,
    TaskRequest,
    TaskResult,
    # Byzantine Fault Tolerance
    BFTVote,
    ByzantineConsensusForSwarm,
    # Speculative Execution
    SpeculativeStrategy,
    SpeculativeExecutionEngine,
    # Adaptive Load Balancing
    AdaptiveLoadBalancer,
    # Event-Driven Coordination
    SwarmEvent,
    AgentJoinedEvent,
    AgentLeftEvent,
    TaskAssignedEvent,
    TaskCompletedEvent,
    ConsensusReachedEvent,
    HealthAlertEvent,
    SwarmEventCoordinator,
    # Self-Healing
    SelfHealingMonitor,
    # Topology
    HierarchicalMeshTopology,
    # Main Controller
    ExtremeSwarmController,
    # Factory
    create_extreme_swarm,
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
    # Flow State Machine (V38)
    "FlowState",
    "FlowTransition",
    "FlowStateMachine",
    "FlowContext",
    "FlowExecutor",
    "StateHistoryEntry",
    # Saga Orchestration (V36 - ADR-029)
    "SagaState",
    "StepState",
    "SagaStep",
    "StepResult",
    "Saga",
    "SagaResult",
    "SagaStateStore",
    "SagaMetrics",
    "SagaOrchestrator",
    "create_multi_adapter_saga",
    "example_order_processing_saga",
    # Bulkhead Pattern (V38 - ADR-029)
    "Bulkhead",
    "BulkheadConfig",
    "BulkheadRegistry",
    "BulkheadStats",
    "BulkheadState",
    "BulkheadError",
    "BulkheadFullError",
    "BulkheadTimeoutError",
    "BulkheadRejectionReason",
    "get_bulkhead",
    "get_bulkhead_registry",
    "create_database_bulkhead",
    "create_api_bulkhead",
    "create_mcp_bulkhead",
    # Batch Embedding Pipeline (V39)
    "BatchEmbeddingPipeline",
    "EmbeddingRequest",
    "BatchEmbeddingResult",
    "EmbeddingCache",
    "EmbeddingProvider",
    "ProviderModel",
    "OpenAIEmbedder",
    "VoyageEmbedder",
    "JinaEmbedder",
    "LocalEmbedder",
    "BaseEmbedder",
    "create_batch_pipeline",
    "quick_embed",
    # Domain Events (V40 Enhanced)
    "EventMetrics",
    "EventVersion",
    "EventSchemaEvolution",
    "DomainEvent",
    "MemoryStoredEvent",
    "MemoryRetrievedEvent",
    "MemoryEvictedEvent",
    "MemoryPromotedEvent",
    "SessionStartedEvent",
    "SessionEndedEvent",
    "SessionCheckpointEvent",
    "FlowStartedEvent",
    "FlowCompletedEvent",
    "FlowFailedEvent",
    # Research events
    "ResearchQueryRequestedEvent",
    "ResearchResultReceivedEvent",
    "ResearchResultCachedEvent",
    "ResearchCacheHitEvent",
    "ResearchAdapterHealthEvent",
    "EventHandler",
    "SyncEventHandler",
    # Dead letter queue
    "DeadLetterEntry",
    "DeadLetterQueue",
    # Event aggregation
    "EventAggregator",
    "EventBus",
    "EventStore",
    "EventStreamPosition",
    "EventSourcedAggregate",
    "ConcurrencyError",
    "OutboxMessage",
    "TransactionalOutbox",
    "MemoryEventEmitter",
    # Research integration
    "ResearchEventEmitter",
    "EventAwareResearchAdapter",
    "ResearchEventHandler",
    "ResearchEventProjection",
    "create_research_event_system",
    "EventRegistry",
    # Cache Warming (V39)
    "CacheWarmer",
    "WarmingStrategy",
    "WarmingResult",
    "UsagePattern",
    "WarmingPriority",
    "WarmingStatus",
    "warm_mcp_connections",
    "warm_embedding_cache",
    "warm_memory_context",
    "warm_research_cache",
    "warm_sdk_adapters",
    "warm_semantic_cache",
    "create_default_warmer",
    "create_minimal_warmer",
    "get_cache_warmer",
    "set_cache_warmer",
    # Extreme Swarm Patterns (V40 - ADR-030)
    "SwarmRole",
    "AgentState",
    "StrategyState",
    "HealthStatus",
    "TokenBudget",
    "AgentMetadata",
    "TaskRequest",
    "TaskResult",
    "BFTVote",
    "ByzantineConsensusForSwarm",
    "SpeculativeStrategy",
    "SpeculativeExecutionEngine",
    "AdaptiveLoadBalancer",
    "SwarmEvent",
    "AgentJoinedEvent",
    "AgentLeftEvent",
    "TaskAssignedEvent",
    "TaskCompletedEvent",
    "ConsensusReachedEvent",
    "HealthAlertEvent",
    "SwarmEventCoordinator",
    "SelfHealingMonitor",
    "HierarchicalMeshTopology",
    "ExtremeSwarmController",
    "create_extreme_swarm",
    # Agent Communication (V40 - Memory-based agent-to-agent messaging)
    "AgentMessage",
    "MessageType",
    "MessageStatus",
    "MessagePriority",
    "MessageAcknowledgment",
    "TopicSubscription",
    "InboxStats",
    "MessageSentEvent",
    "MessageReceivedEvent",
    "MessageAcknowledgedEvent",
    "BroadcastSentEvent",
    "SQLiteMessageStore",
    "AgentCommSystem",
    "create_agent_comm_system",
    "LARGE_PAYLOAD_THRESHOLD",
    "DEFAULT_MESSAGE_TTL_SECONDS",
]
