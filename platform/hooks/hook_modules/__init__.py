#!/usr/bin/env python3
"""
Hook Modules Package - Modular MCP and Hook Utilities

This package provides a modular architecture for MCP (Model Context Protocol)
patterns and hook utilities. All symbols are re-exported here for backward
compatibility with code that previously imported from hook_utils.py.

Modules:
- hook_core: Core hook infrastructure (events, responses, matchers)
- mcp_types: MCP content types and tool results
- knowledge_graph: Knowledge graph entity/relation patterns
- thinking_patterns: Sequential thinking and branching patterns
- letta_patterns: Letta memory blocks, steps, and runs
- mcp_elicitation: URL elicitation patterns
- mcp_sampling: LLM sampling/completion patterns
- mcp_capabilities: Server capabilities and subscriptions
- mcp_tasks: Async task management
- mcp_resources: Prompts and resources
- mcp_completion: Context-aware auto-completion
- mcp_transport: Transport abstraction (stdio, SSE, HTTP)
- mcp_errors: JSON-RPC error handling
- mcp_operations: Ping, roots, pagination, tool annotations
- mcp_task_queue: Task message queue for async operations

Usage:
    # Import everything (backward compatible)
    from hook_modules import *

    # Import specific symbols
    from hook_modules import HookEvent, MCPTask, MemoryBlock

    # Import from specific module (recommended for new code)
    from hook_modules.mcp_tasks import TaskStatus, MCPTask

Version: V1.0.0 (2026-01-30) - Initial modular split from hook_utils.py
"""

# =============================================================================
# CORE HOOK INFRASTRUCTURE
# =============================================================================
from .hook_core import (
    HookEvent,
    PermissionDecision,
    PermissionBehavior,
    BlockDecision,
    HookInput,
    HookResponse,
    HookConfig,
    SessionState,
    TurnCounter,
    ToolMatcher,
    log_event,
    get_logger,
)

# =============================================================================
# MCP CONTENT TYPES
# =============================================================================
from .mcp_types import (
    MCPContent,
    MCPToolResult,
    LogLevel,
    LogMessage,
)

# =============================================================================
# KNOWLEDGE GRAPH
# =============================================================================
from .knowledge_graph import (
    Entity,
    Relation,
    KnowledgeGraph,
)

# =============================================================================
# THINKING PATTERNS
# =============================================================================
from .thinking_patterns import (
    ThoughtData,
    ThinkingSession,
    ThoughtType,
    ThoughtBranch,
    EnhancedThought,
    BranchingThinkingSession,
)

# =============================================================================
# LETTA PATTERNS
# =============================================================================
from .letta_patterns import (
    MemoryBlock,
    BlockManager,
    StepFeedbackType,
    StepFilter,
    StepMetrics,
    StepTrace,
    StepFeedback,
    LettaStep,
    RunStatus,
    StopReasonType,
    RunFilter,
    LettaRun,
)

# =============================================================================
# MCP ELICITATION
# =============================================================================
from .mcp_elicitation import (
    ElicitationMode,
    ElicitationAction,
    ElicitationRequest,
    ElicitationResponse,
)

# =============================================================================
# MCP SAMPLING
# =============================================================================
from .mcp_sampling import (
    ToolChoiceMode,
    SamplingMessage,
    ModelPreferences,
    SamplingTool,
    SamplingRequest,
    SamplingResponse,
)

# =============================================================================
# MCP CAPABILITIES
# =============================================================================
from .mcp_capabilities import (
    ProgressNotification,
    MCPCapabilities,
    ResourceSubscription,
    SubscriptionManager,
)

# =============================================================================
# MCP TASKS
# =============================================================================
from .mcp_tasks import (
    TaskStatus,
    TaskSupport,
    MCPTask,
    TaskRequest,
    TaskResult,
)

# =============================================================================
# MCP RESOURCES
# =============================================================================
from .mcp_resources import (
    PromptArgument,
    MCPPrompt,
    PromptMessage,
    ResourceAnnotations,
    MCPResource,
    ResourceTemplate,
)

# =============================================================================
# MCP COMPLETION
# =============================================================================
from .mcp_completion import (
    CompletionRefType,
    CompletionRequest,
    CompletionResult,
)

# =============================================================================
# MCP TRANSPORT
# =============================================================================
from .mcp_transport import (
    TransportType,
    TransportConfig,
    MCPSession,
)

# =============================================================================
# MCP ERRORS (canonical source for error types)
# =============================================================================
from .mcp_errors import (
    MCPErrorCode,
    ErrorData,
    McpError,
    CancellationNotification,
)

# =============================================================================
# MCP OPERATIONS
# =============================================================================
from .mcp_operations import (
    PingRequest,
    PingResponse,
    MCPRoot,
    RootsCapability,
    ListRootsResult,
    PaginatedRequest,
    PaginatedResult,
    ToolAnnotations,
)

# =============================================================================
# MCP TASK QUEUE
# =============================================================================
from .mcp_task_queue import (
    MCP_LATEST_PROTOCOL_VERSION,
    MCP_DEFAULT_NEGOTIATED_VERSION,
    MODEL_IMMEDIATE_RESPONSE_KEY,
    RELATED_TASK_METADATA_KEY,
    QueuedMessage,
    TaskMessageQueue,
    Resolver,
    is_terminal_status,
    generate_task_id,
)


# =============================================================================
# PUBLIC API
# =============================================================================
__all__ = [
    # Hook Core
    "HookEvent",
    "PermissionDecision",
    "PermissionBehavior",
    "BlockDecision",
    "HookInput",
    "HookResponse",
    "HookConfig",
    "SessionState",
    "TurnCounter",
    "ToolMatcher",
    "log_event",
    "get_logger",
    # MCP Types
    "MCPContent",
    "MCPToolResult",
    "LogLevel",
    "LogMessage",
    # Knowledge Graph
    "Entity",
    "Relation",
    "KnowledgeGraph",
    # Thinking Patterns
    "ThoughtData",
    "ThinkingSession",
    "ThoughtType",
    "ThoughtBranch",
    "EnhancedThought",
    "BranchingThinkingSession",
    # Letta Patterns
    "MemoryBlock",
    "BlockManager",
    "StepFeedbackType",
    "StepFilter",
    "StepMetrics",
    "StepTrace",
    "StepFeedback",
    "LettaStep",
    "RunStatus",
    "StopReasonType",
    "RunFilter",
    "LettaRun",
    # MCP Elicitation
    "ElicitationMode",
    "ElicitationAction",
    "ElicitationRequest",
    "ElicitationResponse",
    # MCP Sampling
    "ToolChoiceMode",
    "SamplingMessage",
    "ModelPreferences",
    "SamplingTool",
    "SamplingRequest",
    "SamplingResponse",
    # MCP Capabilities
    "ProgressNotification",
    "MCPCapabilities",
    "ResourceSubscription",
    "SubscriptionManager",
    # MCP Tasks
    "TaskStatus",
    "TaskSupport",
    "MCPTask",
    "TaskRequest",
    "TaskResult",
    # MCP Resources
    "PromptArgument",
    "MCPPrompt",
    "PromptMessage",
    "ResourceAnnotations",
    "MCPResource",
    "ResourceTemplate",
    # MCP Completion
    "CompletionRefType",
    "CompletionRequest",
    "CompletionResult",
    # MCP Transport
    "TransportType",
    "TransportConfig",
    "MCPSession",
    # MCP Errors
    "MCPErrorCode",
    "ErrorData",
    "McpError",
    "CancellationNotification",
    # MCP Operations
    "PingRequest",
    "PingResponse",
    "MCPRoot",
    "RootsCapability",
    "ListRootsResult",
    "PaginatedRequest",
    "PaginatedResult",
    "ToolAnnotations",
    # MCP Task Queue
    "MCP_LATEST_PROTOCOL_VERSION",
    "MCP_DEFAULT_NEGOTIATED_VERSION",
    "MODEL_IMMEDIATE_RESPONSE_KEY",
    "RELATED_TASK_METADATA_KEY",
    "QueuedMessage",
    "TaskMessageQueue",
    "Resolver",
    "is_terminal_status",
    "generate_task_id",
]

# Version info
__version__ = "1.0.0"
__author__ = "UNLEASH Platform"
