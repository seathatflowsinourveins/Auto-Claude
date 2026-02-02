#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.26.0",
#     "structlog>=24.1.0",
#     "pydantic>=2.0.0",
# ]
# [tool.uv]
# exclude-newer = "2026-02-01"
# ///
"""
Hook Utilities - V10.11 Complete

Shared utilities for all Claude Code hooks with comprehensive MCP support.
Provides consistent logging, configuration, and response handling.

Based on official documentation (2026-01-17):
- Claude Code Hooks API v2.0.10+: https://code.claude.com/docs/en/hooks
- MCP Specification 2025-11-25: https://modelcontextprotocol.io/specification/2025-11-25
- MCP Python SDK + FastMCP: https://github.com/modelcontextprotocol/python-sdk
- Letta Python SDK v1.7.1: https://github.com/letta-ai/letta-python
- Anthropic SDK v0.76.0: https://github.com/anthropics/anthropic-sdk-python
- Pydantic-AI v1.43.0: https://github.com/pydantic/pydantic-ai
- LangGraph v1.0.6: https://github.com/langchain-ai/langgraph
- LangChain MCP Adapters v0.1.5: https://github.com/langchain-ai/langchain-mcp-adapters
- DSPy v3.1.0: https://github.com/stanfordnlp/dspy
- Grafana MCP v1.0+: https://github.com/grafana/mcp-grafana
- Qdrant MCP v1.0+: https://github.com/qdrant/mcp-server-qdrant
- Postgres MCP Pro v1.0+: https://github.com/crystaldba/postgres-mcp
- Sequential Thinking: https://github.com/modelcontextprotocol/servers
- structlog v24.1.0: https://github.com/hynek/structlog
- OpenTelemetry Python v1.20.0: https://github.com/open-telemetry/opentelemetry-python
- Sentry Python SDK v2.0.0: https://github.com/getsentry/sentry-python
- pyribs v0.9.0: https://github.com/icaros-usc/pyribs
- claude-task-master v1.15+: https://github.com/eyaltoledano/claude-task-master
- mcp-shrimp-task-manager v1.1+: https://github.com/cjo4m06/mcp-shrimp-task-manager
- Semgrep MCP v1.0+: https://github.com/semgrep/mcp

V10.11 Complete Features (NEW):
- FastMCP Server Patterns: FastMCPServer, FastMCPTool, FastMCPResource, LifespanContext
- FastMCP Context: MCPContext for progress reporting, logging, lifespan access
- Grafana Observability: GrafanaDashboard, PrometheusQuery, LokiQuery, GrafanaAlert
- Grafana Incidents: GrafanaIncident, IncidentActivity for incident management
- Qdrant Vector Patterns: QdrantCollection, QdrantPoint, VectorSearchResult
- Qdrant Embedding: EmbeddingModel, SemanticSearch for RAG patterns
- Postgres MCP Pro: DatabaseAccessMode, QueryExecution, IndexRecommendation
- Postgres Health: DatabaseHealth, ExplainAnalysis for diagnostics
- Enhanced Sequential Thinking: SequentialThought with branching and revision

V10.10 Complete Features:
- LangGraph Graph Orchestration: GraphNode, GraphEdge for state graph primitives
- LangGraph State Graphs: StateGraph for graph-first orchestration with cycles
- LangGraph Runtime: GraphRuntime, GraphCheckpoint for execution/persistence
- LangGraph Workflows: CyclicalWorkflow for dynamic loop execution
- DSPy Programmatic LM: DSPySignature for declarative module signatures
- DSPy Modules: DSPyModule, DSPyPredict for composable LM programs
- DSPy Optimization: DSPyOptimizer, OptimizationResult for prompt tuning
- DSPy Tools: DSPyTool for MCP tool conversion patterns
- MCP Multi-Server: MCPTransportType, MCPServerConfig, MultiServerClient
- MCP Interceptors: MCPInterceptor, InterceptorChain for middleware patterns
- MCP Sessions: StatefulSession, SessionPool for connection management
- MCP Resources: ResourceLoader for resource and prompt loading

V10.9 Complete Features:
- Task-Master PRD Patterns: PRDTask, PRDWorkflow for PRD-to-implementation
- Task-Master Complexity: ComplexityAnalysis, ComplexityLevel for task sizing
- Task-Master Tagged Lists: TaggedTaskList for multi-context task management
- Task-Master Tool Modes: ToolMode enum (Core/Standard/All), TaskDependencyGraph
- Shrimp Chain-of-Thought: ChainOfThought, ThoughtProcess for structured reasoning
- Shrimp Workflow Modes: TaskWorkflowMode (Planning/Execution/Research/Continuous/Reflection)
- Shrimp Persistence: PersistentTaskMemory for cross-session state
- Shrimp Dependencies: TaskDependency, StructuredWorkflow for guided execution
- Semgrep Security: SecuritySeverity, SecurityFinding, SemgrepRule
- Semgrep AST: ASTNode, ASTAnalyzer for semantic code analysis
- Semgrep Audit: SecurityScanner, SecurityAudit for comprehensive scanning

V10.8 Complete Features:
- Structlog Processor Patterns: LogProcessorProtocol, ContextVarBinding, ContextVarManager
- Structlog Processor Chain: ProcessorChain for composable log transformation
- Structlog Renderers: key_value_renderer, logfmt_renderer functions
- OpenTelemetry Span Patterns: SpanStatusCode, SpanStatus, SpanContext, SpanLink, SpanEvent
- OpenTelemetry Span Class: Full Span implementation with attributes, events, links, timing
- Sentry Scope Patterns: ScopeType enum, Breadcrumb, ScopeData, Scope class
- Sentry Scope Manager: ScopeManager with global/isolation/current scope hierarchy
- pyribs Archive Patterns: Elite, ArchiveAddStatus, ArchiveAddResult, ArchiveStats, Archive
- Quality-Diversity Thinking: ThinkingElite, ThinkingArchive for diverse reasoning exploration

V10.7 Complete Features:
- Tool Runner Patterns (Anthropic SDK): EndStrategy, CompactionControl, ToolCache
- LangGraph Tool Patterns: ToolCall, ToolCallRequest, InjectedState, InjectedStore, ToolRuntime
- Pydantic-AI Instrumentation: InstrumentationLevel, InstrumentationSettings
- Enhanced Thinking Session: Checkpoints, branch merging, statistics, revision tracking
- Thinking Part Parser: Split content into TextPart/ThinkingPart (inline <think> tags)
- Full ToolRunner class: Batch execution, caching, early stopping, iteration limits

V10.6 Complete Features:
- MCP Task Message Queue: FIFO queue with async wait/notify for bidirectional communication
- MCP Resolver: anyio-compatible future-like object for async result passing
- MCP Task Helpers: Terminal state checking, task cancellation, execution context manager
- MCP Task Polling: Generic polling utility until terminal status with configurable intervals
- Letta Run Management: Extended filtering (conversation_id, stop_reason, background mode)
- Letta Project Support: project_id filtering for cloud deployments

V10.5 Advanced Features:
- MCP Error Handling: JSON-RPC error codes (-32700 to -32000), ErrorData class
- MCP Cancellation: Request cancellation with reason tracking
- MCP Ping: Connection health verification (ping/pong)
- MCP Roots: Filesystem boundary definitions with file:// URIs
- MCP Pagination: Cursor-based opaque pagination patterns
- MCP Tool Annotations: Behavioral hints (read_only, destructive, idempotent, open_world)
- Letta Step Tracking: Execution steps with metrics, trace, feedback, filtering

V10.4 Ultimate Features (retained):
- MCP Tasks (Experimental): Async task execution with status lifecycle, TTL, polling
- MCP Prompts: Server prompt templates with arguments and completions
- MCP Resources: Resource templates with URI patterns (RFC 6570) and annotations
- MCP Logging: RFC 5424 syslog severity levels (debug → emergency)
- MCP Completion: Context-aware auto-completion for prompts and resources
- Sequential Thinking: Thought branching, revision tracking, JSONL history
- Transport Abstraction: Session-aware stdio/SSE/HTTP patterns
- Letta Memory Blocks: Core memory block lifecycle management

V10.3 Features (retained):
- MCP Elicitation: Form mode (structured data) and URL mode (OAuth, payments)
- MCP Sampling: Server-initiated LLM calls with tool use support
- MCP Progress Reporting: Progress notifications with tokens
- MCP Resource Subscriptions: Subscribe/Unsubscribe patterns
- MCP Capabilities: Client capability checking and negotiation
- Audio content type support (MCP 2025-11-25)
- Knowledge graph integration patterns

V10.2 Features (retained):
- Full hook event output formats (PreToolUse, PermissionRequest, PostToolUse, UserPromptSubmit, Stop)
- MCP structuredContent support for typed tool responses
- Enhanced input modification with updatedInput
- Context injection via additionalContext for all event types
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import structlog

# Optional Pydantic for schema validation
try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)


class HookEvent(Enum):
    """Claude Code hook event types."""
    SESSION_START = "SessionStart"
    SESSION_END = "SessionEnd"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    PERMISSION_REQUEST = "PermissionRequest"
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    NOTIFICATION = "Notification"
    STOP = "Stop"
    SUBAGENT_STOP = "SubagentStop"
    PRE_COMPACT = "PreCompact"


class PermissionDecision(Enum):
    """Permission decisions for PreToolUse hooks.

    Per official Claude Code hooks spec (https://code.claude.com/docs/en/hooks):
    - "allow": Approve the tool call (optionally with updatedInput to modify)
    - "deny": Block the tool call
    - "ask": Show user permission dialog for approval

    Note: Input modification uses "allow" + updatedInput, NOT a separate "modify" value.
    """
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"  # Official value for user confirmation


class PermissionBehavior(Enum):
    """Permission behaviors for PermissionRequest hooks.

    Per official Claude Code hooks spec:
    - "allow": Auto-approve the permission request
    - "deny": Auto-deny the permission request

    PermissionRequest hooks can include:
    - updatedInput: Modified parameters for the tool
    - message: Custom message to display
    - interrupt: Whether to interrupt processing
    """
    ALLOW = "allow"
    DENY = "deny"


class BlockDecision(Enum):
    """Blocking decisions for PostToolUse/Stop/UserPromptSubmit hooks.

    - "block": Prevent the action and provide reason to Claude
    - None/undefined: Allow the action to proceed
    """
    BLOCK = "block"


@dataclass
class HookInput:
    """
    Parsed hook input from Claude Code.

    All hooks receive JSON via stdin with event-specific data.
    """
    event_type: str = ""
    session_id: str = ""
    tool_name: str = ""
    tool_input: Dict[str, Any] = field(default_factory=dict)
    tool_output: Dict[str, Any] = field(default_factory=dict)
    tool_use_id: str = ""  # For correlating PreToolUse/PostToolUse
    cwd: str = ""
    project_dir: str = ""
    transcript_path: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_stdin(cls) -> "HookInput":
        """Parse hook input from stdin."""
        try:
            data = json.load(sys.stdin)
        except (json.JSONDecodeError, EOFError):
            data = {}

        return cls(
            event_type=data.get("event_type", ""),
            session_id=data.get("session_id", "unknown"),
            tool_name=data.get("tool_name", ""),
            tool_input=data.get("tool_input", {}),
            tool_output=data.get("tool_output", {}),
            tool_use_id=data.get("tool_use_id", ""),
            cwd=data.get("cwd", os.getcwd()),
            project_dir=data.get("project_dir", os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())),
            transcript_path=data.get("transcript_path", ""),
            raw=data
        )


@dataclass
class HookResponse:
    """
    Hook response to send back to Claude Code.

    V10.2 Enhanced - Supports all hook event output formats:
    - PreToolUse: permissionDecision (allow/deny/ask), updatedInput, additionalContext
    - PermissionRequest: decision.behavior (allow/deny), updatedInput, message, interrupt
    - PostToolUse: decision (block), reason, additionalContext
    - UserPromptSubmit: decision (block), reason, additionalContext
    - Stop/SubagentStop: decision (block), reason
    - SessionStart: additionalContext
    """
    # Core decision - for PreToolUse
    decision: Optional[PermissionDecision] = PermissionDecision.ALLOW
    reason: str = ""
    modified_input: Optional[Dict[str, Any]] = None
    additional_context: Optional[str] = None  # Context injection

    # PermissionRequest specific
    permission_behavior: Optional[PermissionBehavior] = None
    permission_message: Optional[str] = None
    permission_interrupt: bool = False

    # PostToolUse/Stop/UserPromptSubmit specific
    block: bool = False  # Set to True to block action

    # Session control
    continue_session: bool = True
    stop_reason: Optional[str] = None
    suppress_output: bool = False
    system_message: Optional[str] = None

    # Custom data
    custom_data: Dict[str, Any] = field(default_factory=dict)

    # Event type hint for formatting
    _event_type: HookEvent = field(default=HookEvent.PRE_TOOL_USE, repr=False)

    def for_event(self, event: HookEvent) -> "HookResponse":
        """Set the event type for proper output formatting."""
        self._event_type = event
        return self

    def to_json(self) -> str:
        """Convert to JSON response for Claude Code.

        Formats output based on event type per official spec:
        https://code.claude.com/docs/en/hooks
        """
        output: Dict[str, Any] = {}
        hook_specific: Dict[str, Any] = {}

        if self._event_type == HookEvent.PRE_TOOL_USE:
            # PreToolUse response format
            if self.decision:
                hook_specific = {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": self.decision.value,
                    "permissionDecisionReason": self.reason,
                }
                if self.modified_input:
                    hook_specific["updatedInput"] = self.modified_input
                if self.additional_context:
                    hook_specific["additionalContext"] = self.additional_context

        elif self._event_type == HookEvent.PERMISSION_REQUEST:
            # PermissionRequest response format
            if self.permission_behavior:
                decision_obj: Dict[str, Any] = {
                    "behavior": self.permission_behavior.value,
                }
                if self.modified_input:
                    decision_obj["updatedInput"] = self.modified_input
                if self.permission_message:
                    decision_obj["message"] = self.permission_message
                if self.permission_interrupt:
                    decision_obj["interrupt"] = True
                hook_specific = {
                    "hookEventName": "PermissionRequest",
                    "decision": decision_obj,
                }

        elif self._event_type == HookEvent.POST_TOOL_USE:
            # PostToolUse response format
            hook_specific = {"hookEventName": "PostToolUse"}
            if self.block:
                output["decision"] = "block"
                output["reason"] = self.reason
            if self.additional_context:
                hook_specific["additionalContext"] = self.additional_context

        elif self._event_type in (HookEvent.STOP, HookEvent.SUBAGENT_STOP):
            # Stop/SubagentStop response format
            if self.block:
                output["decision"] = "block"
                output["reason"] = self.reason

        elif self._event_type == HookEvent.USER_PROMPT_SUBMIT:
            # UserPromptSubmit response format
            hook_specific = {"hookEventName": "UserPromptSubmit"}
            if self.block:
                output["decision"] = "block"
                output["reason"] = self.reason
            if self.additional_context:
                hook_specific["additionalContext"] = self.additional_context

        elif self._event_type == HookEvent.SESSION_START:
            # SessionStart response format
            if self.additional_context:
                hook_specific = {
                    "hookEventName": "SessionStart",
                    "additionalContext": self.additional_context,
                }

        # Common fields
        if not self.continue_session:
            output["continue"] = False
        if self.stop_reason:
            output["stopReason"] = self.stop_reason
        if self.suppress_output:
            output["suppressOutput"] = True
        if self.system_message:
            output["systemMessage"] = self.system_message

        if hook_specific:
            output["hookSpecificOutput"] = hook_specific

        # Custom data
        output.update(self.custom_data)

        return json.dumps(output)

    def send(self) -> None:
        """Send response to stdout and exit appropriately."""
        print(self.to_json())

        # Exit code 2 for blocking errors
        if self.block or self.decision == PermissionDecision.DENY:
            sys.exit(2)
        sys.exit(0)

    @classmethod
    def allow(cls, reason: str = "", context: Optional[str] = None) -> "HookResponse":
        """Create an allow response for PreToolUse."""
        return cls(
            decision=PermissionDecision.ALLOW,
            reason=reason,
            additional_context=context
        )

    @classmethod
    def deny(cls, reason: str) -> "HookResponse":
        """Create a deny response for PreToolUse."""
        return cls(decision=PermissionDecision.DENY, reason=reason)

    @classmethod
    def ask(cls, reason: str = "") -> "HookResponse":
        """Create an ask response for PreToolUse (show user confirmation)."""
        return cls(decision=PermissionDecision.ASK, reason=reason)

    @classmethod
    def modify(cls, updated_input: Dict[str, Any], reason: str = "") -> "HookResponse":
        """Create an allow response with modified input."""
        return cls(
            decision=PermissionDecision.ALLOW,
            reason=reason,
            modified_input=updated_input
        )

    @classmethod
    def block_action(cls, reason: str, event: HookEvent) -> "HookResponse":
        """Create a block response for PostToolUse/Stop/UserPromptSubmit."""
        return cls(block=True, reason=reason, _event_type=event)


class HookConfig:
    """
    Configuration management for hooks.

    Uses environment variables from Claude Code:
    - CLAUDE_PROJECT_DIR: Project root path
    - CLAUDE_CODE_REMOTE: True for web environments
    - CLAUDE_HOOKS_DIR: Hooks directory path
    - CLAUDE_ENV_FILE: File for persisting env vars (SessionStart)
    """

    def __init__(self):
        self.project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd()))
        self.hooks_dir = Path(os.environ.get("CLAUDE_HOOKS_DIR", Path.home() / ".claude" / "v10" / "hooks"))
        self.v10_dir = Path.home() / ".claude" / "v10"
        self.logs_dir = self.v10_dir / "logs"
        self.cache_dir = self.v10_dir / "cache"
        self.is_remote = os.environ.get("CLAUDE_CODE_REMOTE", "").lower() == "true"

        # Letta configuration
        self.letta_url = os.environ.get("LETTA_URL", "http://localhost:8283")
        self.letta_api_key = os.environ.get("LETTA_API_KEY")

        # Ensure directories exist
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_session_env_file(self) -> Path:
        """Get the session environment file path."""
        return self.v10_dir / ".session_env"

    def get_turn_count_file(self) -> Path:
        """Get the turn count file path."""
        return self.v10_dir / ".turn_count"

    def is_kill_switch_active(self) -> bool:
        """Check if kill switch is active."""
        return (Path.home() / ".claude" / "KILL_SWITCH").exists()

    def get_log_file(self, prefix: str = "audit") -> Path:
        """Get daily log file path."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.logs_dir / f"{prefix}_{today}.jsonl"


class SessionState:
    """
    Manages session state persistence across hook invocations.
    """

    def __init__(self, config: HookConfig):
        self.config = config
        self._state_file = config.get_session_env_file()

    def load(self) -> Dict[str, str]:
        """Load session state from file."""
        if not self._state_file.exists():
            return {}

        state = {}
        for line in self._state_file.read_text().split("\n"):
            if "=" in line:
                key, value = line.split("=", 1)
                state[key.strip()] = value.strip()
        return state

    def save(self, state: Dict[str, str]) -> None:
        """Save session state to file."""
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{k}={v}" for k, v in state.items()]
        self._state_file.write_text("\n".join(lines))

    def update(self, key: str, value: str) -> None:
        """Update a single key in session state."""
        state = self.load()
        state[key] = value
        self.save(state)

    def get(self, key: str, default: str = "") -> str:
        """Get a value from session state."""
        return self.load().get(key, default)

    def clear(self) -> None:
        """Clear session state."""
        if self._state_file.exists():
            self._state_file.unlink()


class TurnCounter:
    """
    Manages turn counting for sleeptime triggers.
    """

    def __init__(self, config: HookConfig, frequency: int = 5):
        self.config = config
        self.frequency = frequency
        self._file = config.get_turn_count_file()

    def get(self) -> int:
        """Get current turn count."""
        if not self._file.exists():
            return 0
        try:
            return int(self._file.read_text().strip())
        except ValueError:
            return 0

    def increment(self) -> int:
        """Increment and return turn count."""
        count = self.get() + 1
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._file.write_text(str(count))
        return count

    def reset(self) -> None:
        """Reset turn count."""
        if self._file.exists():
            self._file.unlink()

    def should_consolidate(self) -> bool:
        """Check if memory consolidation should trigger."""
        count = self.get()
        return count >= self.frequency and count % self.frequency == 0


def log_event(
    event_type: str,
    data: Dict[str, Any],
    config: Optional[HookConfig] = None
) -> None:
    """
    Log an event to the daily audit log.

    Args:
        event_type: Type of event (e.g., "file_write", "tool_call")
        data: Event data to log
        config: HookConfig instance (creates new one if not provided)
    """
    if config is None:
        config = HookConfig()

    log_file = config.get_log_file()

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        **data
    }

    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a configured logger for a hook."""
    return structlog.get_logger(name)


class ToolMatcher:
    """
    Tool name matcher for hook configuration.

    Supports patterns from official Claude Code hooks spec:
    - Exact match: "Bash", "Edit", "Write"
    - Regex patterns: "Edit|Write", "mcp__.*"
    - Wildcard: "*" matches all tools
    - MCP tools: "mcp__<server>__<tool>"
    """

    def __init__(self, pattern: str):
        self.pattern = pattern
        self._compiled = None

        if pattern != "*":
            # Convert to regex pattern
            self._compiled = re.compile(f"^{pattern}$")

    def matches(self, tool_name: str) -> bool:
        """Check if tool name matches the pattern."""
        if self.pattern == "*":
            return True
        if self._compiled:
            return bool(self._compiled.match(tool_name))
        return self.pattern == tool_name

    @classmethod
    def mcp_server(cls, server_name: str) -> "ToolMatcher":
        """Create matcher for all tools from an MCP server."""
        return cls(f"mcp__{server_name}__.*")

    @classmethod
    def mcp_tool(cls, server_name: str, tool_name: str) -> "ToolMatcher":
        """Create matcher for a specific MCP tool."""
        return cls(f"mcp__{server_name}__{tool_name}")


@dataclass
class MCPContent:
    """
    MCP content item for tool responses.

    Per MCP Specification 2025-11-25, tool results can contain:
    - text: Plain text content
    - image: Base64-encoded image with mimeType
    - audio: Base64-encoded audio with mimeType
    - resource_link: URI to a resource
    - resource: Embedded resource content
    """
    content_type: str  # "text", "image", "audio", "resource_link", "resource"
    text: Optional[str] = None
    data: Optional[str] = None  # Base64 for image/audio
    mime_type: Optional[str] = None
    uri: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    annotations: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP content format."""
        result: Dict[str, Any] = {"type": self.content_type}

        if self.content_type == "text":
            result["text"] = self.text or ""
        elif self.content_type in ("image", "audio"):
            result["data"] = self.data or ""
            result["mimeType"] = self.mime_type or "application/octet-stream"
        elif self.content_type == "resource_link":
            result["uri"] = self.uri or ""
            if self.name:
                result["name"] = self.name
            if self.description:
                result["description"] = self.description
            if self.mime_type:
                result["mimeType"] = self.mime_type
        elif self.content_type == "resource":
            result["resource"] = {
                "uri": self.uri or "",
                "mimeType": self.mime_type,
                "text": self.text,
            }

        if self.annotations:
            result["annotations"] = self.annotations

        return result

    @classmethod
    def text_content(cls, text: str) -> "MCPContent":
        """Create text content item."""
        return cls(content_type="text", text=text)

    @classmethod
    def image_content(cls, data: str, mime_type: str = "image/png") -> "MCPContent":
        """Create image content item (base64 encoded)."""
        return cls(content_type="image", data=data, mime_type=mime_type)


@dataclass
class MCPToolResult:
    """
    MCP tool result with structured content support.

    Per MCP Specification 2025-11-25:
    - content: Unstructured content array (text, images, resources)
    - structuredContent: Typed JSON object matching outputSchema
    - isError: Whether the result represents an error
    """
    content: list = field(default_factory=list)  # List[MCPContent]
    structured_content: Optional[Dict[str, Any]] = None
    is_error: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP tool result format."""
        result: Dict[str, Any] = {
            "content": [
                c.to_dict() if isinstance(c, MCPContent) else c
                for c in self.content
            ],
            "isError": self.is_error,
        }

        if self.structured_content is not None:
            result["structuredContent"] = self.structured_content
            # For backwards compatibility, also include as text
            if not any(c.get("type") == "text" for c in result["content"]):
                result["content"].insert(0, {
                    "type": "text",
                    "text": json.dumps(self.structured_content)
                })

        return result

    @classmethod
    def success(cls, text: str, structured: Optional[Dict[str, Any]] = None) -> "MCPToolResult":
        """Create successful result with optional structured content."""
        return cls(
            content=[MCPContent.text_content(text)],
            structured_content=structured,
            is_error=False
        )

    @classmethod
    def error(cls, message: str) -> "MCPToolResult":
        """Create error result."""
        return cls(
            content=[MCPContent.text_content(message)],
            is_error=True
        )


# =============================================================================
# V10.3: MCP ELICITATION SUPPORT
# Per MCP Specification 2025-11-25: https://modelcontextprotocol.io/specification/2025-11-25/client/elicitation
# =============================================================================


class ElicitationMode(Enum):
    """MCP elicitation modes for collecting user input."""
    FORM = "form"  # Structured form with JSON Schema validation
    URL = "url"    # Out-of-band URL for OAuth, payments, etc.


class ElicitationAction(Enum):
    """User response actions for elicitation requests."""
    ACCEPT = "accept"    # User provided the requested data
    DECLINE = "decline"  # User declined to provide data
    CANCEL = "cancel"    # User cancelled the dialog


@dataclass
class ElicitationRequest:
    """
    MCP elicitation request for collecting user input.

    Per MCP 2025-11-25 specification:
    - Form mode: Collect structured data using JSON Schema validation
    - URL mode: Redirect to external URL for OAuth, payments, credentials

    Example (Form mode):
        request = ElicitationRequest.form(
            message="Please provide your preferences",
            schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "title": "Full Name"},
                    "email": {"type": "string", "format": "email"},
                    "age": {"type": "integer", "minimum": 0, "maximum": 150}
                },
                "required": ["name", "email"]
            }
        )

    Example (URL mode):
        request = ElicitationRequest.url(
            message="Please complete payment",
            url="https://payment.example.com/checkout?token=xxx",
            description="Secure payment required"
        )
    """
    mode: ElicitationMode
    message: str
    schema: Optional[Dict[str, Any]] = None  # JSON Schema for form mode
    redirect_url: Optional[str] = None  # URL for url mode
    description: Optional[str] = None  # Description for url mode
    timeout_ms: int = 600000  # 10 minutes default

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP elicitation/create request params."""
        if self.mode == ElicitationMode.FORM:
            return {
                "message": self.message,
                "requestedSchema": self.schema or {},
            }
        else:  # URL mode
            return {
                "message": self.message,
                "url": self.redirect_url or "",
                "description": self.description,
            }

    @classmethod
    def form(cls, message: str, schema: Dict[str, Any], timeout_ms: int = 600000) -> "ElicitationRequest":
        """Create a form mode elicitation request with JSON Schema."""
        return cls(
            mode=ElicitationMode.FORM,
            message=message,
            schema=schema,
            timeout_ms=timeout_ms
        )

    @classmethod
    def url_mode(cls, message: str, url: str, description: Optional[str] = None) -> "ElicitationRequest":
        """Create a URL mode elicitation request for OAuth/payments."""
        return cls(
            mode=ElicitationMode.URL,
            message=message,
            redirect_url=url,
            description=description
        )


@dataclass
class ElicitationResponse:
    """
    Response from an elicitation request.

    Per MCP 2025-11-25:
    - action: "accept" | "decline" | "cancel"
    - content: User-provided data (only when action is "accept")
    """
    action: ElicitationAction
    content: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ElicitationResponse":
        """Parse elicitation result from MCP response."""
        action = ElicitationAction(data.get("action", "cancel"))
        content = data.get("content") if action == ElicitationAction.ACCEPT else None
        return cls(action=action, content=content)

    @property
    def accepted(self) -> bool:
        """Check if user accepted and provided data."""
        return self.action == ElicitationAction.ACCEPT and self.content is not None

    @property
    def declined(self) -> bool:
        """Check if user declined to provide data."""
        return self.action == ElicitationAction.DECLINE

    @property
    def cancelled(self) -> bool:
        """Check if user cancelled the dialog."""
        return self.action == ElicitationAction.CANCEL


# =============================================================================
# V10.3: MCP SAMPLING SUPPORT
# Per MCP Specification 2025-11-25: https://modelcontextprotocol.io/specification/2025-11-25/client/sampling
# =============================================================================


class ToolChoiceMode(Enum):
    """Tool choice modes for sampling requests."""
    AUTO = "auto"        # Model decides whether to use tools
    REQUIRED = "required"  # Model MUST use at least one tool
    NONE = "none"        # Model MUST NOT use any tools


@dataclass
class SamplingMessage:
    """
    Message for MCP sampling requests.

    Supports text, image, and audio content types.
    """
    role: str  # "user" or "assistant"
    content: Union[str, Dict[str, Any], List[Dict[str, Any]]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP sampling message format."""
        if isinstance(self.content, str):
            content = {"type": "text", "text": self.content}
        else:
            content = self.content
        return {"role": self.role, "content": content}

    @classmethod
    def user(cls, text: str) -> "SamplingMessage":
        """Create a user message with text content."""
        return cls(role="user", content=text)

    @classmethod
    def assistant(cls, text: str) -> "SamplingMessage":
        """Create an assistant message with text content."""
        return cls(role="assistant", content=text)


@dataclass
class ModelPreferences:
    """
    Model preferences for sampling requests.

    Allows servers to hint at model selection without mandating specific models.
    """
    hints: List[Dict[str, str]] = field(default_factory=list)  # Model name hints
    cost_priority: float = 0.5  # 0-1, higher = prefer cheaper
    speed_priority: float = 0.5  # 0-1, higher = prefer faster
    intelligence_priority: float = 0.5  # 0-1, higher = prefer more capable

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP model preferences format."""
        result: Dict[str, Any] = {}
        if self.hints:
            result["hints"] = self.hints
        if self.cost_priority != 0.5:
            result["costPriority"] = self.cost_priority
        if self.speed_priority != 0.5:
            result["speedPriority"] = self.speed_priority
        if self.intelligence_priority != 0.5:
            result["intelligencePriority"] = self.intelligence_priority
        return result

    @classmethod
    def prefer_claude(cls, variant: str = "sonnet") -> "ModelPreferences":
        """Create preferences hinting at Claude models."""
        return cls(hints=[{"name": f"claude-3-{variant}"}, {"name": "claude"}])

    @classmethod
    def prefer_fast(cls) -> "ModelPreferences":
        """Create preferences prioritizing speed over capability."""
        return cls(speed_priority=0.9, intelligence_priority=0.3, cost_priority=0.5)

    @classmethod
    def prefer_smart(cls) -> "ModelPreferences":
        """Create preferences prioritizing capability over speed."""
        return cls(intelligence_priority=0.9, speed_priority=0.3, cost_priority=0.3)


@dataclass
class SamplingTool:
    """
    Tool definition for sampling requests.

    Servers can provide tools for the LLM to use during sampling.
    """
    name: str
    description: str
    input_schema: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema
        }


@dataclass
class SamplingRequest:
    """
    MCP sampling request for server-initiated LLM calls.

    Per MCP 2025-11-25, servers can request LLM generations from clients:
    - Basic text generation
    - With tool use support (multi-turn tool loops)
    - With model preference hints

    Example (basic):
        request = SamplingRequest(
            messages=[SamplingMessage.user("Write a haiku about coding")],
            max_tokens=100
        )

    Example (with tools):
        request = SamplingRequest(
            messages=[SamplingMessage.user("What's the weather in Paris?")],
            max_tokens=500,
            tools=[SamplingTool(
                name="get_weather",
                description="Get current weather",
                input_schema={"type": "object", "properties": {"city": {"type": "string"}}}
            )],
            tool_choice=ToolChoiceMode.AUTO
        )
    """
    messages: List[SamplingMessage]
    max_tokens: int = 1000
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    model_preferences: Optional[ModelPreferences] = None
    tools: Optional[List[SamplingTool]] = None
    tool_choice: Optional[ToolChoiceMode] = None
    stop_sequences: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP sampling/createMessage request params."""
        result: Dict[str, Any] = {
            "messages": [m.to_dict() for m in self.messages],
            "maxTokens": self.max_tokens,
        }

        if self.system_prompt:
            result["systemPrompt"] = self.system_prompt
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.model_preferences:
            result["modelPreferences"] = self.model_preferences.to_dict()
        if self.tools:
            result["tools"] = [t.to_dict() for t in self.tools]
        if self.tool_choice:
            result["toolChoice"] = {"mode": self.tool_choice.value}
        if self.stop_sequences:
            result["stopSequences"] = self.stop_sequences

        return result


@dataclass
class SamplingResponse:
    """
    Response from a sampling request.

    Per MCP 2025-11-25:
    - content: Text or tool use content
    - model: The model that was used
    - stop_reason: Why generation stopped ("endTurn", "toolUse", "maxTokens", etc.)
    """
    role: str
    content: Union[str, Dict[str, Any], List[Dict[str, Any]]]
    model: str
    stop_reason: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SamplingResponse":
        """Parse sampling result from MCP response."""
        return cls(
            role=data.get("role", "assistant"),
            content=data.get("content", ""),
            model=data.get("model", "unknown"),
            stop_reason=data.get("stopReason", "unknown")
        )

    @property
    def is_tool_use(self) -> bool:
        """Check if response contains tool use requests."""
        return self.stop_reason == "toolUse"

    @property
    def text(self) -> str:
        """Extract text content from response."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, dict) and self.content.get("type") == "text":
            return self.content.get("text", "")
        if isinstance(self.content, list):
            for item in self.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    return item.get("text", "")
        return str(self.content)

    def get_tool_uses(self) -> List[Dict[str, Any]]:
        """Extract tool use requests from response."""
        if isinstance(self.content, list):
            return [c for c in self.content if isinstance(c, dict) and c.get("type") == "tool_use"]
        if isinstance(self.content, dict) and self.content.get("type") == "tool_use":
            return [self.content]
        return []


# =============================================================================
# V10.3: MCP PROGRESS REPORTING
# Per MCP Specification 2025-11-25: Progress notifications for long-running operations
# =============================================================================


@dataclass
class ProgressNotification:
    """
    Progress notification for long-running operations.

    Per MCP 2025-11-25, servers can report progress via notifications/progress.
    The progress_token is provided by the client in request metadata.

    Example:
        progress = ProgressNotification(
            progress_token=metadata.get("progressToken"),
            progress=50,
            total=100,
            message="Processing file 5 of 10..."
        )
    """
    progress_token: str
    progress: int
    total: Optional[int] = None
    message: Optional[str] = None

    def to_notification(self, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Convert to MCP progress notification format."""
        params: Dict[str, Any] = {
            "progressToken": self.progress_token,
            "progress": self.progress,
        }
        if self.total is not None:
            params["total"] = self.total
        if self.message:
            params["message"] = self.message

        notification: Dict[str, Any] = {
            "method": "notifications/progress",
            "params": params,
        }
        return notification

    @property
    def percentage(self) -> Optional[float]:
        """Calculate percentage complete."""
        if self.total and self.total > 0:
            return (self.progress / self.total) * 100
        return None


# =============================================================================
# V10.3: MCP CAPABILITIES
# Per MCP Specification 2025-11-25: Client/Server capability negotiation
# =============================================================================


@dataclass
class MCPCapabilities:
    """
    MCP client/server capabilities for feature negotiation.

    Per MCP 2025-11-25, capabilities are declared during initialization:
    - sampling: Client can process sampling requests
    - elicitation: Client can display elicitation dialogs
    - tools: Client supports tool use in sampling
    - context: Client supports context inclusion (soft-deprecated)
    """
    sampling: bool = False
    sampling_tools: bool = False  # Tool use in sampling
    elicitation: bool = False
    roots: bool = False  # Client can provide filesystem roots

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPCapabilities":
        """Parse capabilities from MCP initialization response."""
        sampling = data.get("sampling")
        return cls(
            sampling=sampling is not None,
            sampling_tools=isinstance(sampling, dict) and "tools" in sampling,
            elicitation=data.get("elicitation") is not None,
            roots=data.get("roots") is not None,
        )

    def supports_sampling(self) -> bool:
        """Check if client supports basic sampling."""
        return self.sampling

    def supports_sampling_tools(self) -> bool:
        """Check if client supports tool use in sampling."""
        return self.sampling and self.sampling_tools

    def supports_elicitation(self) -> bool:
        """Check if client supports elicitation dialogs."""
        return self.elicitation


# =============================================================================
# V10.3: MCP RESOURCE SUBSCRIPTIONS
# Per MCP Specification 2025-11-25: Resource subscription patterns
# =============================================================================


@dataclass
class ResourceSubscription:
    """
    MCP resource subscription for real-time updates.

    Per MCP 2025-11-25, clients can subscribe to resource changes:
    - resources/subscribe: Subscribe to a resource URI
    - resources/unsubscribe: Unsubscribe from a resource URI
    - notifications/resources/updated: Server notifies of changes
    """
    uri: str
    session_id: Optional[str] = None

    def to_subscribe_request(self) -> Dict[str, Any]:
        """Convert to MCP resources/subscribe request."""
        return {
            "method": "resources/subscribe",
            "params": {"uri": self.uri}
        }

    def to_unsubscribe_request(self) -> Dict[str, Any]:
        """Convert to MCP resources/unsubscribe request."""
        return {
            "method": "resources/unsubscribe",
            "params": {"uri": self.uri}
        }


class SubscriptionManager:
    """
    Manager for MCP resource subscriptions.

    Tracks subscribers by URI and session ID, providing update notifications.
    """

    def __init__(self):
        self._subscriptions: Dict[str, Set[str]] = {}  # URI -> set of session IDs

    def subscribe(self, uri: str, session_id: str = "default") -> None:
        """Subscribe a session to a resource URI."""
        if uri not in self._subscriptions:
            self._subscriptions[uri] = set()
        self._subscriptions[uri].add(session_id)

    def unsubscribe(self, uri: str, session_id: str = "default") -> None:
        """Unsubscribe a session from a resource URI."""
        if uri in self._subscriptions:
            self._subscriptions[uri].discard(session_id)
            if not self._subscriptions[uri]:
                del self._subscriptions[uri]

    def get_subscribers(self, uri: str) -> Set[str]:
        """Get all session IDs subscribed to a URI."""
        return self._subscriptions.get(uri, set())

    def has_subscribers(self, uri: str) -> bool:
        """Check if a URI has any subscribers."""
        return uri in self._subscriptions and len(self._subscriptions[uri]) > 0

    def get_subscribed_uris(self, session_id: str = "default") -> List[str]:
        """Get all URIs a session is subscribed to."""
        return [uri for uri, sessions in self._subscriptions.items() if session_id in sessions]

    def create_update_notification(self, uri: str) -> Dict[str, Any]:
        """Create a resource update notification."""
        return {
            "method": "notifications/resources/updated",
            "params": {"uri": uri}
        }


# =============================================================================
# V10.3: KNOWLEDGE GRAPH INTEGRATION
# Based on MCP Memory Server patterns for entity-relation-observation storage
# =============================================================================


@dataclass
class Entity:
    """
    Knowledge graph entity with observations.

    Per MCP Memory Server pattern:
    - name: Unique entity identifier
    - entity_type: Classification (person, project, concept, etc.)
    - observations: List of facts/observations about the entity
    """
    name: str
    entity_type: str
    observations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable format."""
        return {
            "type": "entity",
            "name": self.name,
            "entityType": self.entity_type,
            "observations": self.observations
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """Create from dictionary."""
        return cls(
            name=data.get("name", ""),
            entity_type=data.get("entityType", "unknown"),
            observations=data.get("observations", [])
        )


@dataclass
class Relation:
    """
    Knowledge graph relation between entities.

    Per MCP Memory Server pattern:
    - from_entity: Source entity name
    - to_entity: Target entity name
    - relation_type: Type of relationship (uses active voice)
    """
    from_entity: str
    to_entity: str
    relation_type: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable format."""
        return {
            "type": "relation",
            "from": self.from_entity,
            "to": self.to_entity,
            "relationType": self.relation_type
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        """Create from dictionary."""
        return cls(
            from_entity=data.get("from", ""),
            to_entity=data.get("to", ""),
            relation_type=data.get("relationType", "relates_to")
        )


@dataclass
class KnowledgeGraph:
    """
    Simple knowledge graph storage using entities and relations.

    Based on MCP Memory Server patterns with JSONL persistence.
    """
    entities: List[Entity] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)

    def add_entity(self, entity: Entity) -> bool:
        """Add entity if not already present."""
        if not any(e.name == entity.name for e in self.entities):
            self.entities.append(entity)
            return True
        return False

    def add_relation(self, relation: Relation) -> bool:
        """Add relation if not already present."""
        exists = any(
            r.from_entity == relation.from_entity and
            r.to_entity == relation.to_entity and
            r.relation_type == relation.relation_type
            for r in self.relations
        )
        if not exists:
            self.relations.append(relation)
            return True
        return False

    def add_observation(self, entity_name: str, observation: str) -> bool:
        """Add observation to an entity."""
        for entity in self.entities:
            if entity.name == entity_name:
                if observation not in entity.observations:
                    entity.observations.append(observation)
                    return True
        return False

    def search(self, query: str) -> "KnowledgeGraph":
        """Search entities by query string."""
        query_lower = query.lower()
        matched_entities = [
            e for e in self.entities
            if query_lower in e.name.lower() or
               query_lower in e.entity_type.lower() or
               any(query_lower in obs.lower() for obs in e.observations)
        ]
        matched_names = {e.name for e in matched_entities}
        matched_relations = [
            r for r in self.relations
            if r.from_entity in matched_names and r.to_entity in matched_names
        ]
        return KnowledgeGraph(entities=matched_entities, relations=matched_relations)

    def to_jsonl(self) -> str:
        """Serialize to JSONL format."""
        lines = []
        for entity in self.entities:
            lines.append(json.dumps(entity.to_dict()))
        for relation in self.relations:
            lines.append(json.dumps(relation.to_dict()))
        return "\n".join(lines)

    @classmethod
    def from_jsonl(cls, data: str) -> "KnowledgeGraph":
        """Parse from JSONL format."""
        graph = cls()
        for line in data.strip().split("\n"):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if item.get("type") == "entity":
                    graph.entities.append(Entity.from_dict(item))
                elif item.get("type") == "relation":
                    graph.relations.append(Relation.from_dict(item))
            except json.JSONDecodeError:
                continue
        return graph


# =============================================================================
# V10.4: MCP TASKS (EXPERIMENTAL)
# Per MCP Specification 2025-11-25: https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/tasks
# Durable async task execution for long-running operations
# =============================================================================


class TaskStatus(Enum):
    """
    MCP task status values for async operations.

    Per MCP 2025-11-25 specification:
    - working: Task is actively being processed
    - input_required: Task needs additional input (human-in-the-loop)
    - completed: Task finished successfully
    - failed: Task encountered an error
    - cancelled: Task was cancelled
    """
    WORKING = "working"
    INPUT_REQUIRED = "input_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskSupport(Enum):
    """
    Tool-level task support negotiation.

    Per MCP 2025-11-25:
    - required: Tool always returns async tasks
    - optional: Tool may return tasks based on request
    - forbidden: Tool never returns tasks (default)
    """
    REQUIRED = "required"
    OPTIONAL = "optional"
    FORBIDDEN = "forbidden"


@dataclass
class MCPTask:
    """
    MCP async task for long-running operations.

    Per MCP 2025-11-25, tasks support:
    - Status lifecycle (working → completed/failed/cancelled)
    - TTL-based expiration
    - Polling with recommended intervals
    - Input requirement for human-in-the-loop patterns

    Example:
        task = MCPTask(
            task_id="786512e2-9e0d-44bd-8f29-789f320fe840",
            status=TaskStatus.WORKING,
            poll_interval=5000,
            ttl=60000
        )
    """
    task_id: str
    status: TaskStatus
    created_at: Optional[datetime] = None
    last_updated_at: Optional[datetime] = None
    ttl: int = 60000  # Milliseconds until expiration
    poll_interval: Optional[int] = None  # Recommended polling interval in ms
    message: Optional[str] = None  # Human-readable status message
    input_request: Optional[Dict[str, Any]] = None  # Schema for required input

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP task response format."""
        result: Dict[str, Any] = {
            "taskId": self.task_id,
            "status": self.status.value,
            "ttl": self.ttl,
        }

        if self.created_at:
            result["createdAt"] = self.created_at.isoformat()
        if self.last_updated_at:
            result["lastUpdatedAt"] = self.last_updated_at.isoformat()
        if self.poll_interval:
            result["pollInterval"] = self.poll_interval
        if self.message:
            result["message"] = self.message
        if self.input_request:
            result["inputRequest"] = self.input_request

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPTask":
        """Parse task from MCP response."""
        created = data.get("createdAt")
        updated = data.get("lastUpdatedAt")

        return cls(
            task_id=data.get("taskId", ""),
            status=TaskStatus(data.get("status", "working")),
            created_at=datetime.fromisoformat(created.replace("Z", "+00:00")) if created else None,
            last_updated_at=datetime.fromisoformat(updated.replace("Z", "+00:00")) if updated else None,
            ttl=data.get("ttl", 60000),
            poll_interval=data.get("pollInterval"),
            message=data.get("message"),
            input_request=data.get("inputRequest"),
        )

    @property
    def is_complete(self) -> bool:
        """Check if task reached terminal state."""
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)

    @property
    def needs_input(self) -> bool:
        """Check if task requires additional input."""
        return self.status == TaskStatus.INPUT_REQUIRED


@dataclass
class TaskRequest:
    """
    Request for creating an MCP async task.

    Per MCP 2025-11-25, include task params in tools/call:
        {
            "method": "tools/call",
            "params": {
                "name": "long_operation",
                "arguments": {...},
                "task": {"ttl": 60000}
            }
        }
    """
    ttl: int = 60000  # Task TTL in milliseconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to task request params."""
        return {"ttl": self.ttl}


@dataclass
class TaskResult:
    """
    Result of a completed MCP task.

    Per MCP 2025-11-25, get results via tasks/result:
        {
            "method": "tasks/result",
            "params": {"taskId": "..."}
        }
    """
    task_id: str
    content: List[Dict[str, Any]] = field(default_factory=list)
    structured_content: Optional[Dict[str, Any]] = None
    is_error: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP task result format."""
        result: Dict[str, Any] = {
            "taskId": self.task_id,
            "content": self.content,
            "isError": self.is_error,
        }
        if self.structured_content:
            result["structuredContent"] = self.structured_content
        return result


# =============================================================================
# V10.4: MCP PROMPTS
# Per MCP Specification 2025-11-25: https://modelcontextprotocol.io/specification/2025-11-25/server/prompts
# Server-exposed prompt templates with arguments
# =============================================================================


@dataclass
class PromptArgument:
    """
    Argument definition for MCP prompt templates.

    Per MCP 2025-11-25:
    - name: Unique argument identifier
    - description: Human-readable description
    - required: Whether argument must be provided
    """
    name: str
    description: str = ""
    required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP prompt argument format."""
        result: Dict[str, Any] = {"name": self.name}
        if self.description:
            result["description"] = self.description
        if self.required:
            result["required"] = True
        return result


@dataclass
class MCPPrompt:
    """
    MCP prompt template exposed by servers.

    Per MCP 2025-11-25:
    - Servers expose prompt templates via prompts/list
    - Clients retrieve expanded prompts via prompts/get
    - Arguments are filled by the client

    Example:
        prompt = MCPPrompt(
            name="code_review",
            title="Request Code Review",
            description="Asks the LLM to analyze code quality",
            arguments=[
                PromptArgument("code", "The code to review", required=True),
                PromptArgument("language", "Programming language")
            ]
        )
    """
    name: str
    description: str = ""
    title: Optional[str] = None
    arguments: List[PromptArgument] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP prompt list item format."""
        result: Dict[str, Any] = {
            "name": self.name,
        }
        if self.title:
            result["title"] = self.title
        if self.description:
            result["description"] = self.description
        if self.arguments:
            result["arguments"] = [a.to_dict() for a in self.arguments]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPPrompt":
        """Parse prompt from MCP response."""
        args = [
            PromptArgument(
                name=a.get("name", ""),
                description=a.get("description", ""),
                required=a.get("required", False)
            )
            for a in data.get("arguments", [])
        ]
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            title=data.get("title"),
            arguments=args
        )


@dataclass
class PromptMessage:
    """
    Message content from an expanded prompt.

    Per MCP 2025-11-25, prompts/get returns:
    - description: Expanded description
    - messages: Array of role/content messages
    """
    role: str  # "user" or "assistant"
    content: Union[str, Dict[str, Any], List[Dict[str, Any]]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP prompt message format."""
        if isinstance(self.content, str):
            content = {"type": "text", "text": self.content}
        else:
            content = self.content
        return {"role": self.role, "content": content}


# =============================================================================
# V10.4: MCP RESOURCES (ENHANCED)
# Per MCP Specification 2025-11-25: https://modelcontextprotocol.io/specification/2025-11-25/server/resources
# Resource templates with URI patterns (RFC 6570) and annotations
# =============================================================================


@dataclass
class ResourceAnnotations:
    """
    MCP resource annotations for metadata.

    Per MCP 2025-11-25:
    - audience: Who this resource is intended for (["user"], ["assistant"], ["user", "assistant"])
    - priority: Importance hint 0-1 (higher = more important)
    - lastModified: ISO timestamp of last modification
    """
    audience: Optional[List[str]] = None  # ["user", "assistant"]
    priority: Optional[float] = None  # 0-1 scale
    last_modified: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP annotations format."""
        result: Dict[str, Any] = {}
        if self.audience:
            result["audience"] = self.audience
        if self.priority is not None:
            result["priority"] = self.priority
        if self.last_modified:
            result["lastModified"] = self.last_modified.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceAnnotations":
        """Parse annotations from MCP response."""
        modified = data.get("lastModified")
        return cls(
            audience=data.get("audience"),
            priority=data.get("priority"),
            last_modified=datetime.fromisoformat(modified.replace("Z", "+00:00")) if modified else None
        )


@dataclass
class MCPResource:
    """
    MCP resource exposed by servers.

    Per MCP 2025-11-25:
    - uri: Unique resource identifier (file://, https://, git://, etc.)
    - name: Human-readable name
    - description: Optional description
    - mimeType: Content type hint
    - annotations: Metadata (audience, priority, lastModified)
    """
    uri: str
    name: str
    description: str = ""
    mime_type: Optional[str] = None
    annotations: Optional[ResourceAnnotations] = None
    size: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP resource format."""
        result: Dict[str, Any] = {
            "uri": self.uri,
            "name": self.name,
        }
        if self.description:
            result["description"] = self.description
        if self.mime_type:
            result["mimeType"] = self.mime_type
        if self.annotations:
            result["annotations"] = self.annotations.to_dict()
        if self.size is not None:
            result["size"] = self.size
        return result


@dataclass
class ResourceTemplate:
    """
    MCP resource template with URI pattern.

    Per MCP 2025-11-25, templates use RFC 6570 URI Templates:
    - uriTemplate: Pattern like "file:///{path}" or "db://users/{userId}"
    - Clients fill in template variables to get actual URIs

    Example:
        template = ResourceTemplate(
            uri_template="file:///{path}",
            name="Project Files",
            description="Access files in the project"
        )
    """
    uri_template: str
    name: str
    description: str = ""
    mime_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP resource template format."""
        result: Dict[str, Any] = {
            "uriTemplate": self.uri_template,
            "name": self.name,
        }
        if self.description:
            result["description"] = self.description
        if self.mime_type:
            result["mimeType"] = self.mime_type
        return result

    def expand(self, **variables: str) -> str:
        """Expand template with variables (simple implementation)."""
        result = self.uri_template
        for key, value in variables.items():
            result = result.replace(f"{{{key}}}", value)
        return result


# =============================================================================
# V10.4: MCP LOGGING
# Per MCP Specification 2025-11-25: https://modelcontextprotocol.io/specification/2025-11-25/server/utilities/logging
# RFC 5424 syslog severity levels for structured diagnostics
# =============================================================================


class LogLevel(Enum):
    """
    RFC 5424 syslog severity levels for MCP logging.

    Per MCP 2025-11-25:
    - Servers send log messages via notifications/message
    - Clients can set minimum level via logging/setLevel
    """
    DEBUG = "debug"        # Detailed debugging information
    INFO = "info"          # General informational messages
    NOTICE = "notice"      # Normal but significant conditions
    WARNING = "warning"    # Warning conditions
    ERROR = "error"        # Error conditions
    CRITICAL = "critical"  # Critical conditions
    ALERT = "alert"        # Action must be taken immediately
    EMERGENCY = "emergency"  # System is unusable


@dataclass
class LogMessage:
    """
    MCP log message notification.

    Per MCP 2025-11-25:
    - level: Severity level (RFC 5424)
    - logger: Name of the logger (optional)
    - data: Structured log data (any JSON-serializable value)

    Example:
        msg = LogMessage(
            level=LogLevel.ERROR,
            logger="database",
            data={"error": "Connection failed", "host": "localhost"}
        )
    """
    level: LogLevel
    data: Any
    logger: Optional[str] = None

    def to_notification(self) -> Dict[str, Any]:
        """Convert to MCP notifications/message format."""
        params: Dict[str, Any] = {
            "level": self.level.value,
            "data": self.data,
        }
        if self.logger:
            params["logger"] = self.logger

        return {
            "method": "notifications/message",
            "params": params
        }


# =============================================================================
# V10.4: MCP COMPLETION
# Per MCP Specification 2025-11-25: https://modelcontextprotocol.io/specification/2025-11-25/server/utilities/completion
# Context-aware auto-completion for prompts and resources
# =============================================================================


class CompletionRefType(Enum):
    """Reference types for completion requests."""
    PROMPT = "ref/prompt"
    RESOURCE = "ref/resource"


@dataclass
class CompletionRequest:
    """
    MCP completion request for argument auto-completion.

    Per MCP 2025-11-25:
    - ref: Reference to prompt or resource template
    - argument: Current argument being completed
    - context: Previously provided argument values

    Example:
        request = CompletionRequest(
            ref_type=CompletionRefType.PROMPT,
            ref_name="code_review",
            argument_name="language",
            argument_value="py",
            context={"framework": "flask"}
        )
    """
    ref_type: CompletionRefType
    ref_name: str
    argument_name: str
    argument_value: str = ""
    context: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP completion/complete request params."""
        result: Dict[str, Any] = {
            "ref": {
                "type": self.ref_type.value,
                "name": self.ref_name,
            },
            "argument": {
                "name": self.argument_name,
                "value": self.argument_value,
            }
        }
        if self.context:
            result["context"] = {"arguments": self.context}
        return result


@dataclass
class CompletionResult:
    """
    Result of a completion request.

    Per MCP 2025-11-25:
    - values: List of completion suggestions
    - total: Total number of available completions (if known)
    - hasMore: Whether more completions are available
    """
    values: List[str] = field(default_factory=list)
    total: Optional[int] = None
    has_more: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompletionResult":
        """Parse completion result from MCP response."""
        completion = data.get("completion", {})
        return cls(
            values=completion.get("values", []),
            total=completion.get("total"),
            has_more=completion.get("hasMore", False)
        )


# =============================================================================
# V10.4: SEQUENTIAL THINKING
# Based on MCP Sequential Thinking Server: Extended reasoning with branching
# =============================================================================


@dataclass
class ThoughtData:
    """
    Sequential thinking thought with branching support.

    Per MCP Sequential Thinking Server:
    - Supports multi-step reasoning chains
    - Allows revision of previous thoughts
    - Enables branching for alternative paths

    Example:
        thought = ThoughtData(
            thought="Analyzing the algorithm complexity...",
            thought_number=3,
            total_thoughts=5,
            next_thought_needed=True
        )
    """
    thought: str
    thought_number: int
    total_thoughts: int
    next_thought_needed: bool = True
    is_revision: bool = False
    revises_thought: Optional[int] = None
    branch_from_thought: Optional[int] = None
    branch_id: Optional[str] = None
    needs_more_thoughts: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable format."""
        result: Dict[str, Any] = {
            "thought": self.thought,
            "thoughtNumber": self.thought_number,
            "totalThoughts": self.total_thoughts,
            "nextThoughtNeeded": self.next_thought_needed,
        }

        if self.is_revision:
            result["isRevision"] = True
            if self.revises_thought is not None:
                result["revisesThought"] = self.revises_thought

        if self.branch_from_thought is not None:
            result["branchFromThought"] = self.branch_from_thought
            if self.branch_id:
                result["branchId"] = self.branch_id

        if self.needs_more_thoughts:
            result["needsMoreThoughts"] = True

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThoughtData":
        """Parse thought from dictionary."""
        return cls(
            thought=data.get("thought", ""),
            thought_number=data.get("thoughtNumber", 1),
            total_thoughts=data.get("totalThoughts", 1),
            next_thought_needed=data.get("nextThoughtNeeded", False),
            is_revision=data.get("isRevision", False),
            revises_thought=data.get("revisesThought"),
            branch_from_thought=data.get("branchFromThought"),
            branch_id=data.get("branchId"),
            needs_more_thoughts=data.get("needsMoreThoughts", False),
        )


@dataclass
class ThinkingSession:
    """
    Session for sequential thinking with history tracking.

    Maintains a chain of thoughts with support for:
    - Linear progression
    - Revision of previous thoughts
    - Branching for alternative reasoning paths
    """
    session_id: str
    thoughts: List[ThoughtData] = field(default_factory=list)
    branches: Dict[str, List[ThoughtData]] = field(default_factory=dict)
    current_branch: Optional[str] = None

    def add_thought(self, thought: ThoughtData) -> None:
        """Add a thought to the session."""
        if thought.branch_id:
            if thought.branch_id not in self.branches:
                self.branches[thought.branch_id] = []
            self.branches[thought.branch_id].append(thought)
        else:
            self.thoughts.append(thought)

    def get_history(self, branch_id: Optional[str] = None) -> List[ThoughtData]:
        """Get thought history for main chain or a branch."""
        if branch_id and branch_id in self.branches:
            return self.branches[branch_id]
        return self.thoughts

    def to_jsonl(self) -> str:
        """Serialize to JSONL format."""
        lines = []
        for thought in self.thoughts:
            lines.append(json.dumps(thought.to_dict()))
        for branch_id, branch_thoughts in self.branches.items():
            for thought in branch_thoughts:
                lines.append(json.dumps(thought.to_dict()))
        return "\n".join(lines)

    @classmethod
    def from_jsonl(cls, session_id: str, data: str) -> "ThinkingSession":
        """Parse from JSONL format."""
        session = cls(session_id=session_id)
        for line in data.strip().split("\n"):
            if not line.strip():
                continue
            try:
                thought = ThoughtData.from_dict(json.loads(line))
                session.add_thought(thought)
            except json.JSONDecodeError:
                continue
        return session


# =============================================================================
# V10.4: TRANSPORT ABSTRACTION
# Based on MCP SDK transport patterns: stdio, SSE, Streamable HTTP
# =============================================================================


class TransportType(Enum):
    """MCP transport types."""
    STDIO = "stdio"  # Standard input/output (default)
    SSE = "sse"      # Server-Sent Events
    HTTP = "http"    # Streamable HTTP with sessions


@dataclass
class TransportConfig:
    """
    Configuration for MCP transports.

    Per MCP SDK patterns:
    - stdio: Simple stdin/stdout, no session management
    - SSE: GET for streams, POST for messages, session via query param
    - HTTP: POST/GET/DELETE with mcp-session-id header, resumability
    """
    transport_type: TransportType
    endpoint: Optional[str] = None  # URL for SSE/HTTP
    session_id: Optional[str] = None
    port: int = 3001
    enable_resumability: bool = False  # For HTTP transport

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable format."""
        result: Dict[str, Any] = {
            "type": self.transport_type.value,
        }
        if self.endpoint:
            result["endpoint"] = self.endpoint
        if self.session_id:
            result["sessionId"] = self.session_id
        if self.transport_type == TransportType.HTTP:
            result["port"] = self.port
            result["enableResumability"] = self.enable_resumability
        return result


@dataclass
class MCPSession:
    """
    MCP session state for transport management.

    Tracks session lifecycle for SSE/HTTP transports.
    """
    session_id: str
    transport_type: TransportType
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_event_id: Optional[str] = None  # For SSE resumability
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable format."""
        return {
            "sessionId": self.session_id,
            "transportType": self.transport_type.value,
            "createdAt": self.created_at.isoformat(),
            "lastEventId": self.last_event_id,
            "isActive": self.is_active,
        }


# =============================================================================
# V10.4: LETTA MEMORY BLOCKS
# Based on Letta Python SDK v1.7.1: Core memory block lifecycle management
# =============================================================================


@dataclass
class MemoryBlock:
    """
    Letta core memory block.

    Per Letta SDK v1.7.1:
    - Blocks are named memory segments attached to agents
    - Common labels: "persona", "human", "system"
    - Supports templates, read-only access, and metadata
    """
    block_id: Optional[str] = None
    label: str = ""
    value: str = ""
    description: Optional[str] = None
    limit: int = 5000  # Character limit
    is_template: bool = False
    template_name: Optional[str] = None
    read_only: bool = False
    hidden: bool = False
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Letta block format."""
        result: Dict[str, Any] = {
            "label": self.label,
            "value": self.value,
            "limit": self.limit,
        }
        if self.block_id:
            result["id"] = self.block_id
        if self.description:
            result["description"] = self.description
        if self.is_template:
            result["is_template"] = True
            if self.template_name:
                result["template_name"] = self.template_name
        if self.read_only:
            result["read_only"] = True
        if self.hidden:
            result["hidden"] = True
        if self.metadata:
            result["metadata"] = self.metadata
        if self.tags:
            result["tags"] = self.tags
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryBlock":
        """Parse block from Letta API response."""
        return cls(
            block_id=data.get("id"),
            label=data.get("label", ""),
            value=data.get("value", ""),
            description=data.get("description"),
            limit=data.get("limit", 5000),
            is_template=data.get("is_template", False),
            template_name=data.get("template_name"),
            read_only=data.get("read_only", False),
            hidden=data.get("hidden", False),
            metadata=data.get("metadata"),
            tags=data.get("tags"),
        )

    @classmethod
    def persona(cls, value: str, name: str = "persona") -> "MemoryBlock":
        """Create a persona memory block."""
        return cls(label=name, value=value, description="Agent persona/identity")

    @classmethod
    def human(cls, value: str, name: str = "human") -> "MemoryBlock":
        """Create a human (user) memory block."""
        return cls(label=name, value=value, description="Human/user information")


@dataclass
class BlockManager:
    """
    Manager for Letta memory blocks.

    Provides lifecycle operations:
    - create, retrieve, update, delete
    - attach/detach to agents
    - list with filtering
    """
    blocks: List[MemoryBlock] = field(default_factory=list)

    def add(self, block: MemoryBlock) -> None:
        """Add a block to the manager."""
        self.blocks.append(block)

    def get_by_label(self, label: str) -> Optional[MemoryBlock]:
        """Get a block by its label."""
        for block in self.blocks:
            if block.label == label:
                return block
        return None

    def get_by_id(self, block_id: str) -> Optional[MemoryBlock]:
        """Get a block by its ID."""
        for block in self.blocks:
            if block.block_id == block_id:
                return block
        return None

    def update_value(self, label: str, value: str) -> bool:
        """Update a block's value by label."""
        block = self.get_by_label(label)
        if block:
            if len(value) <= block.limit:
                block.value = value
                return True
        return False

    def to_list(self) -> List[Dict[str, Any]]:
        """Convert all blocks to list format."""
        return [b.to_dict() for b in self.blocks]


# =============================================================================
# V10.5: MCP ERROR HANDLING
# Per MCP Specification 2025-11-25: JSON-RPC 2.0 error codes
# Source: mcp-python/src/mcp/types.py
# =============================================================================


class MCPErrorCode:
    """
    Standard JSON-RPC 2.0 error codes for MCP.

    Per MCP 2025-11-25 specification (from official SDK types.py):
    - PARSE_ERROR: Invalid JSON received
    - INVALID_REQUEST: JSON is not a valid request object
    - METHOD_NOT_FOUND: Method does not exist
    - INVALID_PARAMS: Invalid method parameters
    - INTERNAL_ERROR: Internal JSON-RPC error

    MCP-specific error codes:
    - URL_ELICITATION_REQUIRED: Client must handle URL elicitation (-32042)
    - CONNECTION_CLOSED: Connection was closed unexpectedly (-32000)
    """
    # Standard JSON-RPC errors
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # MCP-specific errors
    URL_ELICITATION_REQUIRED = -32042
    CONNECTION_CLOSED = -32000

    @classmethod
    def get_message(cls, code: int) -> str:
        """Get standard message for error code."""
        messages = {
            cls.PARSE_ERROR: "Parse error",
            cls.INVALID_REQUEST: "Invalid Request",
            cls.METHOD_NOT_FOUND: "Method not found",
            cls.INVALID_PARAMS: "Invalid params",
            cls.INTERNAL_ERROR: "Internal error",
            cls.URL_ELICITATION_REQUIRED: "URL elicitation required",
            cls.CONNECTION_CLOSED: "Connection closed",
        }
        return messages.get(code, "Unknown error")


@dataclass
class ErrorData:
    """
    MCP error data attached to JSON-RPC errors.

    Per MCP 2025-11-25:
    - Additional structured information about the error
    - Optional retry hints and recovery suggestions
    """
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    retry_after_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP error data format."""
        result: Dict[str, Any] = {}
        if self.message:
            result["message"] = self.message
        if self.details:
            result["details"] = self.details
        if self.retry_after_ms is not None:
            result["retryAfterMs"] = self.retry_after_ms
        return result


class McpError(Exception):
    """
    MCP protocol error exception.

    Per MCP 2025-11-25, wraps JSON-RPC error responses for easier handling.

    Example:
        raise McpError(
            code=MCPErrorCode.INVALID_PARAMS,
            message="Missing required parameter 'uri'",
            data=ErrorData(details={"parameter": "uri"})
        )
    """

    def __init__(
        self,
        code: int,
        message: Optional[str] = None,
        data: Optional[ErrorData] = None
    ):
        self.code = code
        self.message = message or MCPErrorCode.get_message(code)
        self.data = data
        super().__init__(f"MCP Error {code}: {self.message}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-RPC error object."""
        result: Dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.data:
            result["data"] = self.data.to_dict()
        return result

    @classmethod
    def parse_error(cls, details: Optional[str] = None) -> "McpError":
        """Create a parse error (-32700)."""
        return cls(
            code=MCPErrorCode.PARSE_ERROR,
            data=ErrorData(message=details) if details else None
        )

    @classmethod
    def invalid_request(cls, details: Optional[str] = None) -> "McpError":
        """Create an invalid request error (-32600)."""
        return cls(
            code=MCPErrorCode.INVALID_REQUEST,
            data=ErrorData(message=details) if details else None
        )

    @classmethod
    def method_not_found(cls, method: str) -> "McpError":
        """Create a method not found error (-32601)."""
        return cls(
            code=MCPErrorCode.METHOD_NOT_FOUND,
            message=f"Method not found: {method}"
        )

    @classmethod
    def invalid_params(cls, details: str) -> "McpError":
        """Create an invalid params error (-32602)."""
        return cls(
            code=MCPErrorCode.INVALID_PARAMS,
            data=ErrorData(message=details)
        )


# =============================================================================
# V10.5: MCP CANCELLATION
# Per MCP Specification 2025-11-25: Request cancellation notifications
# Source: mcp-python/src/mcp/types.py - CancelledNotification
# =============================================================================


@dataclass
class CancellationNotification:
    """
    MCP request cancellation notification.

    Per MCP 2025-11-25 (notifications/cancelled):
    - Sent to cancel a pending request
    - Includes optional reason for cancellation
    - Recipients SHOULD stop processing and respond with error

    Example:
        cancel = CancellationNotification(
            request_id="req-123",
            reason="User cancelled operation"
        )
    """
    request_id: str
    reason: Optional[str] = None

    def to_notification(self) -> Dict[str, Any]:
        """Convert to MCP notifications/cancelled format."""
        params: Dict[str, Any] = {
            "requestId": self.request_id,
        }
        if self.reason:
            params["reason"] = self.reason

        return {
            "method": "notifications/cancelled",
            "params": params
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CancellationNotification":
        """Parse cancellation from notification params."""
        params = data.get("params", {})
        return cls(
            request_id=params.get("requestId", ""),
            reason=params.get("reason")
        )


# =============================================================================
# V10.5: MCP PING
# Per MCP Specification 2025-11-25: Connection health verification
# Source: mcp-python/src/mcp/types.py - PingRequest
# =============================================================================


@dataclass
class PingRequest:
    """
    MCP ping request for connection health verification.

    Per MCP 2025-11-25:
    - Simple request to verify connection is alive
    - Expected response is empty result {}
    - Useful for keep-alive and connection monitoring

    Example:
        ping = PingRequest()
        # Send via: {"method": "ping", "params": {}}
        # Expect: {"result": {}}
    """

    def to_request(self) -> Dict[str, Any]:
        """Convert to MCP ping request format."""
        return {
            "method": "ping",
            "params": {}
        }


@dataclass
class PingResponse:
    """
    MCP ping response (empty result).

    Per MCP 2025-11-25:
    - Empty object response confirms connection is alive
    """
    latency_ms: Optional[float] = None  # Optional timing measurement

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP result format."""
        result: Dict[str, Any] = {}
        if self.latency_ms is not None:
            result["_latencyMs"] = self.latency_ms
        return result


# =============================================================================
# V10.5: MCP ROOTS
# Per MCP Specification 2025-11-25: Filesystem boundary definitions
# Source: mcp-python/src/mcp/types.py - Root, RootsCapability
# =============================================================================


@dataclass
class MCPRoot:
    """
    MCP filesystem root definition.

    Per MCP 2025-11-25:
    - Defines filesystem boundaries that servers can access
    - URIs MUST start with file:// protocol
    - Clients expose roots during initialization
    - Servers subscribe to roots/list_changed notifications

    Example:
        root = MCPRoot(
            uri="file:///home/user/project",
            name="Project Root"
        )
    """
    uri: str  # Must be file:// URI
    name: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate URI starts with file://"""
        if not self.uri.startswith("file://"):
            raise ValueError(f"Root URI must start with file://, got: {self.uri}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP root format."""
        result: Dict[str, Any] = {"uri": self.uri}
        if self.name:
            result["name"] = self.name
        if self.meta:
            result["_meta"] = self.meta
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPRoot":
        """Parse root from MCP response."""
        return cls(
            uri=data.get("uri", ""),
            name=data.get("name"),
            meta=data.get("_meta")
        )

    @classmethod
    def from_path(cls, path: str, name: Optional[str] = None) -> "MCPRoot":
        """Create root from filesystem path."""
        # Convert path to file:// URI
        clean_path = path.replace("\\", "/")
        if not clean_path.startswith("/"):
            clean_path = "/" + clean_path
        return cls(uri=f"file://{clean_path}", name=name or path)


@dataclass
class RootsCapability:
    """
    MCP roots capability declaration.

    Per MCP 2025-11-25:
    - Declared in client capabilities during initialization
    - listChanged: Whether client supports roots/list_changed notifications

    Example:
        capability = RootsCapability(list_changed=True)
    """
    list_changed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP capability format."""
        result: Dict[str, Any] = {}
        if self.list_changed:
            result["listChanged"] = True
        return result


@dataclass
class ListRootsResult:
    """
    Result of roots/list request.

    Per MCP 2025-11-25:
    - Returns array of Root objects
    - Empty array means no filesystem access
    """
    roots: List[MCPRoot] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP result format."""
        return {"roots": [r.to_dict() for r in self.roots]}


# =============================================================================
# V10.5: MCP PAGINATION
# Per MCP Specification 2025-11-25: Cursor-based opaque pagination
# Source: mcp-python/src/mcp/types.py - PaginatedRequest, PaginatedResult
# =============================================================================


@dataclass
class PaginatedRequest:
    """
    Base for MCP paginated requests.

    Per MCP 2025-11-25:
    - cursor: Opaque string from previous nextCursor
    - Servers may limit page sizes
    - Cursor meaning is server-defined

    Example:
        # First request (no cursor)
        request = PaginatedRequest()

        # Subsequent requests (with cursor from previous response)
        request = PaginatedRequest(cursor="abc123")
    """
    cursor: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP request params."""
        result: Dict[str, Any] = {}
        if self.cursor:
            result["cursor"] = self.cursor
        return result


@dataclass
class PaginatedResult:
    """
    Base for MCP paginated results.

    Per MCP 2025-11-25:
    - nextCursor: Opaque cursor for next page (None if last page)
    - Clients should continue requesting until nextCursor is None

    Example:
        result = PaginatedResult(next_cursor="abc123")
        if result.has_more:
            # Fetch next page with result.next_cursor
    """
    next_cursor: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP result format."""
        result: Dict[str, Any] = {}
        if self.next_cursor:
            result["nextCursor"] = self.next_cursor
        return result

    @property
    def has_more(self) -> bool:
        """Check if more pages are available."""
        return self.next_cursor is not None


# =============================================================================
# V10.5: MCP TOOL ANNOTATIONS
# Per MCP Specification 2025-11-25: Behavioral hints for tools
# Source: mcp-python/src/mcp/types.py - ToolAnnotations
# =============================================================================


@dataclass
class ToolAnnotations:
    """
    MCP tool behavioral annotations.

    Per MCP 2025-11-25:
    - title: Human-readable title for UI display
    - read_only_hint: Tool only reads data, no side effects
    - destructive_hint: Tool may cause irreversible changes
    - idempotent_hint: Repeated calls produce same result
    - open_world_hint: Tool interacts with external systems

    These are HINTS - models should NOT assume they're accurate for security.

    Example:
        # Safe read-only tool
        annotations = ToolAnnotations(
            title="List Files",
            read_only_hint=True,
            destructive_hint=False
        )

        # Dangerous tool
        annotations = ToolAnnotations(
            title="Delete Database",
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=False
        )
    """
    title: Optional[str] = None
    read_only_hint: Optional[bool] = None
    destructive_hint: Optional[bool] = None
    idempotent_hint: Optional[bool] = None
    open_world_hint: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP annotations format."""
        result: Dict[str, Any] = {}
        if self.title:
            result["title"] = self.title
        if self.read_only_hint is not None:
            result["readOnlyHint"] = self.read_only_hint
        if self.destructive_hint is not None:
            result["destructiveHint"] = self.destructive_hint
        if self.idempotent_hint is not None:
            result["idempotentHint"] = self.idempotent_hint
        if self.open_world_hint is not None:
            result["openWorldHint"] = self.open_world_hint
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolAnnotations":
        """Parse annotations from MCP tool definition."""
        return cls(
            title=data.get("title"),
            read_only_hint=data.get("readOnlyHint"),
            destructive_hint=data.get("destructiveHint"),
            idempotent_hint=data.get("idempotentHint"),
            open_world_hint=data.get("openWorldHint"),
        )

    @classmethod
    def safe_read(cls, title: str = "") -> "ToolAnnotations":
        """Create annotations for safe read-only tools."""
        return cls(
            title=title or None,
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True
        )

    @classmethod
    def dangerous_write(cls, title: str = "") -> "ToolAnnotations":
        """Create annotations for dangerous write tools."""
        return cls(
            title=title or None,
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=False
        )


# =============================================================================
# V10.5: LETTA STEP TRACKING
# Per Letta Python SDK v1.7.1: Execution step monitoring
# Source: letta-python/src/letta_client/resources/steps/steps.py
# =============================================================================


class StepFeedbackType(Enum):
    """Letta step feedback types."""
    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass
class StepFilter:
    """
    Filter parameters for Letta step queries.

    Per Letta SDK v1.7.1:
    - agent_id: Filter by agent
    - model: Filter by model name
    - feedback: Filter by feedback type
    - has_feedback: Filter by presence of feedback
    - tags: Filter by tags
    - trace_ids: Filter by trace IDs
    - start_date/end_date: Date range (ISO format)

    Example:
        filter = StepFilter(
            agent_id="agent-123",
            feedback=StepFeedbackType.POSITIVE,
            start_date="2026-01-01T00:00:00Z"
        )
    """
    agent_id: Optional[str] = None
    model: Optional[str] = None
    feedback: Optional[StepFeedbackType] = None
    has_feedback: Optional[bool] = None
    tags: Optional[List[str]] = None
    trace_ids: Optional[List[str]] = None
    start_date: Optional[str] = None  # ISO datetime
    end_date: Optional[str] = None    # ISO datetime
    limit: Optional[int] = None
    after: Optional[str] = None  # Step ID for pagination
    before: Optional[str] = None  # Step ID for pagination
    order: str = "desc"  # "asc" or "desc"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Letta API query params."""
        result: Dict[str, Any] = {}
        if self.agent_id:
            result["agent_id"] = self.agent_id
        if self.model:
            result["model"] = self.model
        if self.feedback:
            result["feedback"] = self.feedback.value
        if self.has_feedback is not None:
            result["has_feedback"] = self.has_feedback
        if self.tags:
            result["tags"] = self.tags
        if self.trace_ids:
            result["trace_ids"] = self.trace_ids
        if self.start_date:
            result["start_date"] = self.start_date
        if self.end_date:
            result["end_date"] = self.end_date
        if self.limit:
            result["limit"] = self.limit
        if self.after:
            result["after"] = self.after
        if self.before:
            result["before"] = self.before
        if self.order != "desc":
            result["order"] = self.order
        return result


@dataclass
class StepMetrics:
    """
    Metrics for a Letta execution step.

    Per Letta SDK v1.7.1:
    - Accessed via steps.metrics sub-resource
    - Contains timing, token usage, and cost information
    """
    step_id: str
    latency_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable format."""
        result: Dict[str, Any] = {"step_id": self.step_id}
        if self.latency_ms is not None:
            result["latency_ms"] = self.latency_ms
        if self.input_tokens is not None:
            result["input_tokens"] = self.input_tokens
        if self.output_tokens is not None:
            result["output_tokens"] = self.output_tokens
        if self.total_tokens is not None:
            result["total_tokens"] = self.total_tokens
        if self.cost_usd is not None:
            result["cost_usd"] = self.cost_usd
        if self.model:
            result["model"] = self.model
        return result


@dataclass
class StepTrace:
    """
    Trace information for a Letta execution step.

    Per Letta SDK v1.7.1:
    - Accessed via steps.trace sub-resource
    - Contains execution trace and debug information
    """
    step_id: str
    trace_id: str
    parent_trace_id: Optional[str] = None
    span_id: Optional[str] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable format."""
        result: Dict[str, Any] = {
            "step_id": self.step_id,
            "trace_id": self.trace_id,
        }
        if self.parent_trace_id:
            result["parent_trace_id"] = self.parent_trace_id
        if self.span_id:
            result["span_id"] = self.span_id
        if self.events:
            result["events"] = self.events
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class StepFeedback:
    """
    Feedback for a Letta execution step.

    Per Letta SDK v1.7.1:
    - Accessed via steps.feedback sub-resource
    - Allows rating step outputs for training
    """
    step_id: str
    feedback_type: StepFeedbackType
    comment: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Letta API format."""
        result: Dict[str, Any] = {
            "step_id": self.step_id,
            "feedback": self.feedback_type.value,
        }
        if self.comment:
            result["comment"] = self.comment
        if self.metadata:
            result["metadata"] = self.metadata
        if self.created_at:
            result["created_at"] = self.created_at.isoformat()
        return result


@dataclass
class LettaStep:
    """
    Letta execution step with full details.

    Per Letta SDK v1.7.1:
    - Represents a single agent execution step
    - Contains input, output, metrics, and trace information
    """
    step_id: str
    agent_id: str
    created_at: datetime
    model: Optional[str] = None
    input_messages: List[Dict[str, Any]] = field(default_factory=list)
    output_messages: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Optional[StepMetrics] = None
    trace: Optional[StepTrace] = None
    feedback: Optional[StepFeedback] = None
    tags: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable format."""
        result: Dict[str, Any] = {
            "step_id": self.step_id,
            "agent_id": self.agent_id,
            "created_at": self.created_at.isoformat(),
        }
        if self.model:
            result["model"] = self.model
        if self.input_messages:
            result["input_messages"] = self.input_messages
        if self.output_messages:
            result["output_messages"] = self.output_messages
        if self.metrics:
            result["metrics"] = self.metrics.to_dict()
        if self.trace:
            result["trace"] = self.trace.to_dict()
        if self.feedback:
            result["feedback"] = self.feedback.to_dict()
        if self.tags:
            result["tags"] = self.tags
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LettaStep":
        """Parse step from Letta API response."""
        created = data.get("created_at", "")
        created_dt = datetime.fromisoformat(created.replace("Z", "+00:00")) if created else datetime.now(timezone.utc)

        return cls(
            step_id=data.get("step_id", data.get("id", "")),
            agent_id=data.get("agent_id", ""),
            created_at=created_dt,
            model=data.get("model"),
            input_messages=data.get("input_messages", []),
            output_messages=data.get("output_messages", []),
            tags=data.get("tags"),
        )


# =============================================================================
# V10.5: PROTOCOL CONSTANTS
# Per MCP Specification 2025-11-25: Protocol versioning
# Source: mcp-python/src/mcp/types.py
# =============================================================================

# Protocol version constants
MCP_LATEST_PROTOCOL_VERSION = "2025-11-25"
MCP_DEFAULT_NEGOTIATED_VERSION = "2025-03-26"


# =============================================================================
# V10.6: MCP TASK MESSAGE QUEUE
# Per MCP Python SDK: mcp/shared/experimental/tasks/message_queue.py
# FIFO queue for task-related messages with async wait/notify
# =============================================================================


@dataclass
class QueuedMessage:
    """
    A message queued for delivery via tasks/result.

    Per MCP Tasks spec, messages are stored with their type and a resolver
    for requests that expect responses. This enables bidirectional communication
    through the tasks/result endpoint.

    Attributes:
        message_type: "request" (expects response) or "notification" (one-way)
        message: The JSON-RPC message content
        timestamp: When the message was enqueued
        resolver_id: Optional ID for resolving responses
        original_request_id: Original request ID for routing responses
    """
    message_type: str  # "request" or "notification"
    message: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolver_id: Optional[str] = None
    original_request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable format."""
        result: Dict[str, Any] = {
            "messageType": self.message_type,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.resolver_id:
            result["resolverId"] = self.resolver_id
        if self.original_request_id:
            result["originalRequestId"] = self.original_request_id
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueuedMessage":
        """Parse from dictionary."""
        ts = data.get("timestamp")
        return cls(
            message_type=data.get("messageType", "notification"),
            message=data.get("message", {}),
            timestamp=datetime.fromisoformat(ts) if ts else datetime.now(timezone.utc),
            resolver_id=data.get("resolverId"),
            original_request_id=data.get("originalRequestId"),
        )

    @classmethod
    def request(cls, method: str, params: Dict[str, Any], request_id: str) -> "QueuedMessage":
        """Create a request message."""
        return cls(
            message_type="request",
            message={"method": method, "params": params, "id": request_id},
            original_request_id=request_id,
        )

    @classmethod
    def notification(cls, method: str, params: Dict[str, Any]) -> "QueuedMessage":
        """Create a notification message."""
        return cls(
            message_type="notification",
            message={"method": method, "params": params},
        )


class TaskMessageQueue:
    """
    In-memory FIFO queue for task-related messages.

    Per MCP Python SDK pattern, this enables:
    1. Decoupling request handling from message delivery
    2. Proper bidirectional communication via tasks/result stream
    3. Automatic status management (working <-> input_required)

    For distributed systems, implement with Redis, RabbitMQ, etc.
    """

    def __init__(self):
        self._queues: Dict[str, List[QueuedMessage]] = {}
        self._waiting: Dict[str, bool] = {}  # Simple flag for waiting state

    def enqueue(self, task_id: str, message: QueuedMessage) -> None:
        """Add a message to the queue for a task."""
        if task_id not in self._queues:
            self._queues[task_id] = []
        self._queues[task_id].append(message)
        # Mark that messages are available
        self._waiting[task_id] = True

    def dequeue(self, task_id: str) -> Optional[QueuedMessage]:
        """Remove and return the next message from the queue."""
        queue = self._queues.get(task_id, [])
        if not queue:
            return None
        return queue.pop(0)

    def peek(self, task_id: str) -> Optional[QueuedMessage]:
        """Return the next message without removing it."""
        queue = self._queues.get(task_id, [])
        return queue[0] if queue else None

    def is_empty(self, task_id: str) -> bool:
        """Check if the queue is empty for a task."""
        return len(self._queues.get(task_id, [])) == 0

    def clear(self, task_id: str) -> List[QueuedMessage]:
        """Remove and return all messages from the queue."""
        messages = self._queues.pop(task_id, [])
        self._waiting.pop(task_id, None)
        return messages

    def has_messages(self, task_id: str) -> bool:
        """Check if messages are available."""
        return self._waiting.get(task_id, False) and not self.is_empty(task_id)

    def get_queue_size(self, task_id: str) -> int:
        """Get the number of messages in the queue."""
        return len(self._queues.get(task_id, []))

    def cleanup_all(self) -> None:
        """Clean up all queues."""
        self._queues.clear()
        self._waiting.clear()


# =============================================================================
# V10.6: MCP RESOLVER
# Per MCP Python SDK: mcp/shared/experimental/tasks/resolver.py
# anyio-compatible future-like object for async result passing
# =============================================================================


class Resolver:
    """
    A simple resolver for passing results between operations.

    Per MCP Python SDK pattern, this works like asyncio.Future but
    is designed for synchronous/polling use cases. It provides a way
    to pass a result (or exception) from one operation to another.

    Usage:
        resolver = Resolver()
        # In one operation:
        resolver.set_result("hello")
        # In another operation:
        result = resolver.get_result()  # returns "hello"
    """

    def __init__(self):
        self._value: Any = None
        self._exception: Optional[Exception] = None
        self._completed: bool = False

    def set_result(self, value: Any) -> None:
        """Set the result value and mark as completed."""
        if self._completed:
            raise RuntimeError("Resolver already completed")
        self._value = value
        self._completed = True

    def set_exception(self, exc: Exception) -> None:
        """Set an exception and mark as completed."""
        if self._completed:
            raise RuntimeError("Resolver already completed")
        self._exception = exc
        self._completed = True

    def get_result(self) -> Any:
        """Get the result, or raise the exception if one was set."""
        if not self._completed:
            raise RuntimeError("Resolver not yet completed")
        if self._exception is not None:
            raise self._exception
        return self._value

    def done(self) -> bool:
        """Return True if the resolver has been completed."""
        return self._completed

    def has_exception(self) -> bool:
        """Return True if an exception was set."""
        return self._exception is not None

    def reset(self) -> None:
        """Reset the resolver for reuse."""
        self._value = None
        self._exception = None
        self._completed = False


# =============================================================================
# V10.6: MCP TASK HELPERS
# Per MCP Python SDK: mcp/shared/experimental/tasks/helpers.py
# Task management helper functions and constants
# =============================================================================


# Metadata keys per MCP spec
MODEL_IMMEDIATE_RESPONSE_KEY = "io.modelcontextprotocol/model-immediate-response"
RELATED_TASK_METADATA_KEY = "io.modelcontextprotocol/related-task"


def is_terminal_status(status: Union[str, TaskStatus]) -> bool:
    """
    Check if a task status represents a terminal state.

    Terminal states are those where the task has finished and will not change.
    Per MCP spec: completed, failed, or cancelled.

    Args:
        status: The task status to check (string or TaskStatus enum)

    Returns:
        True if the status is terminal
    """
    status_val = status.value if isinstance(status, TaskStatus) else status
    return status_val in ("completed", "failed", "cancelled")


def generate_task_id() -> str:
    """Generate a unique task ID using UUID4."""
    import uuid
    return str(uuid.uuid4())


@dataclass
class TaskMetadata:
    """
    Metadata for task creation.

    Per MCP spec, tasks can have:
    - ttl: Time-to-live in milliseconds
    - custom metadata fields
    """
    ttl: int = 60000  # Default 60 seconds
    poll_interval: int = 500  # Default 500ms
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP task params."""
        result: Dict[str, Any] = {
            "ttl": self.ttl,
            "pollInterval": self.poll_interval,
        }
        result.update(self.custom)
        return result


def create_initial_task(
    metadata: TaskMetadata,
    task_id: Optional[str] = None
) -> MCPTask:
    """
    Create an MCPTask object with initial working state.

    Per MCP spec, new tasks start in "working" status.

    Args:
        metadata: Task metadata including TTL
        task_id: Optional task ID (generated if not provided)

    Returns:
        A new MCPTask in "working" status
    """
    now = datetime.now(timezone.utc)
    return MCPTask(
        task_id=task_id or generate_task_id(),
        status=TaskStatus.WORKING,
        created_at=now,
        last_updated_at=now,
        ttl=metadata.ttl,
        poll_interval=metadata.poll_interval,
    )


# =============================================================================
# V10.6: MCP TASK POLLING
# Per MCP Python SDK: mcp/shared/experimental/tasks/polling.py
# Generic polling utility for task status checks
# =============================================================================


@dataclass
class TaskPollConfig:
    """
    Configuration for task polling.

    Attributes:
        default_interval_ms: Fallback poll interval if server doesn't specify
        max_attempts: Maximum number of poll attempts (0 = unlimited)
        timeout_ms: Overall timeout in milliseconds (0 = no timeout)
    """
    default_interval_ms: int = 500
    max_attempts: int = 0  # 0 = unlimited
    timeout_ms: int = 0  # 0 = no timeout

    def get_interval(self, task: MCPTask) -> float:
        """Get polling interval in seconds."""
        interval = task.poll_interval if task.poll_interval else self.default_interval_ms
        return interval / 1000.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result: Dict[str, Any] = {
            "defaultIntervalMs": self.default_interval_ms,
        }
        if self.max_attempts > 0:
            result["maxAttempts"] = self.max_attempts
        if self.timeout_ms > 0:
            result["timeoutMs"] = self.timeout_ms
        return result


@dataclass
class TaskPollResult:
    """
    Result of polling for task status.

    Attributes:
        task: The current task state
        attempts: Number of poll attempts made
        elapsed_ms: Total elapsed time in milliseconds
        reached_terminal: Whether task reached terminal status
        timed_out: Whether polling timed out
        max_attempts_reached: Whether max attempts was reached
    """
    task: MCPTask
    attempts: int
    elapsed_ms: float
    reached_terminal: bool = False
    timed_out: bool = False
    max_attempts_reached: bool = False

    @property
    def success(self) -> bool:
        """Check if polling completed successfully with terminal status."""
        return self.reached_terminal and not self.timed_out

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result: Dict[str, Any] = {
            "task": self.task.to_dict(),
            "attempts": self.attempts,
            "elapsed_ms": self.elapsed_ms,
            "reached_terminal": self.reached_terminal,
        }
        if self.timed_out:
            result["timed_out"] = self.timed_out
        if self.max_attempts_reached:
            result["max_attempts_reached"] = self.max_attempts_reached
        return result


# =============================================================================
# V10.6: LETTA RUN MANAGEMENT
# Per Letta Python SDK: letta_client/resources/runs/runs.py
# Extended run filtering and management
# =============================================================================


class RunStatus(Enum):
    """
    Letta run status values.

    Per Letta SDK, runs can have various status values indicating
    their current execution state.
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    def is_terminal(self) -> bool:
        """Check if this status represents a terminal state."""
        return self in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED)


class StopReasonType(Enum):
    """
    Letta run stop reason types.

    Per Letta SDK, runs can stop for various reasons:
    - end_turn: Natural conversation end
    - max_tokens: Token limit reached
    - tool_call: Stopped for tool execution
    - error: Error occurred
    """
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    TOOL_CALL = "tool_call"
    ERROR = "error"
    USER_INTERRUPT = "user_interrupt"


@dataclass
class RunFilter:
    """
    Filter for listing Letta runs.

    Per Letta SDK runs.list() parameters, supports extensive filtering
    for run management and monitoring.

    Attributes:
        active: Filter for active runs
        agent_id: Filter by agent ID
        conversation_id: Filter by conversation ID
        background: Filter for background runs
        statuses: Filter by status values
        stop_reason: Filter by stop reason
        after: Cursor for pagination (after this run ID)
        before: Cursor for pagination (before this run ID)
        limit: Maximum number of runs to return
        order: Sort order ("asc" or "desc")
        order_by: Field to sort by (default: "created_at")
    """
    active: Optional[bool] = None
    agent_id: Optional[str] = None
    conversation_id: Optional[str] = None
    background: Optional[bool] = None
    statuses: Optional[List[str]] = None
    stop_reason: Optional[StopReasonType] = None
    after: Optional[str] = None
    before: Optional[str] = None
    limit: Optional[int] = None
    order: str = "desc"
    order_by: str = "created_at"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API query parameters."""
        result: Dict[str, Any] = {}

        if self.active is not None:
            result["active"] = self.active
        if self.agent_id:
            result["agent_id"] = self.agent_id
        if self.conversation_id:
            result["conversation_id"] = self.conversation_id
        if self.background is not None:
            result["background"] = self.background
        if self.statuses:
            result["statuses"] = self.statuses
        if self.stop_reason:
            result["stop_reason"] = self.stop_reason.value
        if self.after:
            result["after"] = self.after
        if self.before:
            result["before"] = self.before
        if self.limit:
            result["limit"] = self.limit
        if self.order != "desc":
            result["order"] = self.order
        if self.order_by != "created_at":
            result["order_by"] = self.order_by

        return result


@dataclass
class LettaRun:
    """
    Letta run representation.

    Per Letta SDK Run type, represents an agent execution run
    with full metadata and status tracking.
    """
    run_id: str
    agent_id: str
    status: RunStatus
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    conversation_id: Optional[str] = None
    background: bool = False
    stop_reason: Optional[StopReasonType] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result: Dict[str, Any] = {
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "background": self.background,
        }

        if self.created_at:
            result["created_at"] = self.created_at.isoformat()
        if self.completed_at:
            result["completed_at"] = self.completed_at.isoformat()
        if self.conversation_id:
            result["conversation_id"] = self.conversation_id
        if self.stop_reason:
            result["stop_reason"] = self.stop_reason.value
        if self.error_message:
            result["error_message"] = self.error_message
        if self.metadata:
            result["metadata"] = self.metadata

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LettaRun":
        """Parse from API response."""
        created = data.get("created_at")
        completed = data.get("completed_at")
        stop = data.get("stop_reason")

        return cls(
            run_id=data.get("run_id", data.get("id", "")),
            agent_id=data.get("agent_id", ""),
            status=RunStatus(data.get("status", "pending")),
            created_at=datetime.fromisoformat(created.replace("Z", "+00:00")) if created else None,
            completed_at=datetime.fromisoformat(completed.replace("Z", "+00:00")) if completed else None,
            conversation_id=data.get("conversation_id"),
            background=data.get("background", False),
            stop_reason=StopReasonType(stop) if stop else None,
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
        )

    @property
    def is_active(self) -> bool:
        """Check if run is still active."""
        return self.status in (RunStatus.PENDING, RunStatus.RUNNING)

    @property
    def is_complete(self) -> bool:
        """Check if run has completed (success or failure)."""
        return self.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED)


# =============================================================================
# V10.6: LETTA PROJECT SUPPORT
# Per Letta Python SDK: project_id filtering for cloud deployments
# =============================================================================


@dataclass
class ProjectFilter:
    """
    Filter for Letta cloud project-scoped operations.

    Per Letta SDK, cloud deployments can filter by project_id
    for multi-tenant isolation.
    """
    project_id: str
    include_archived: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API query parameters."""
        return {
            "project_id": self.project_id,
            "include_archived": self.include_archived,
        }


@dataclass
class ExtendedStepFilter(StepFilter):
    """
    Extended step filter with project support.

    Per Letta SDK steps.list() with cloud project filtering.
    Inherits all StepFilter fields and adds project_id.
    """
    project_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API query parameters (extends parent)."""
        result = super().to_dict()
        if self.project_id:
            result["project_id"] = self.project_id
        return result


# =============================================================================
# V10.7: ADVANCED TOOL RUNNER PATTERNS
# Based on Anthropic SDK v0.76.0 BetaToolRunner, Pydantic-AI v1.43.0 Agent,
# LangGraph v1.0.6 ToolNode patterns
# =============================================================================


class EndStrategy(Enum):
    """
    Strategy for handling tool execution completion.

    Per Pydantic-AI Agent:
    - EARLY: Stop at first successful tool result (default, faster)
    - EXHAUSTIVE: Execute all pending tools before returning

    Example:
        runner = ToolRunner(end_strategy=EndStrategy.EARLY)
    """
    EARLY = "early"
    EXHAUSTIVE = "exhaustive"


class CompactionMode(Enum):
    """
    Context compaction mode for long conversations.

    Per Anthropic SDK CompactionControl:
    - NONE: No compaction, raise error on context overflow
    - SUMMARIZE: Summarize older messages
    - TRUNCATE: Drop older messages
    - SLIDING_WINDOW: Keep last N messages
    """
    NONE = "none"
    SUMMARIZE = "summarize"
    TRUNCATE = "truncate"
    SLIDING_WINDOW = "sliding_window"


@dataclass
class CompactionControl:
    """
    Context window management for tool runners.

    Per Anthropic SDK v0.76.0:
    Controls how the runner handles context overflow during
    extended tool use conversations.

    Example:
        control = CompactionControl(
            mode=CompactionMode.SLIDING_WINDOW,
            max_tokens=100000,
            window_size=50
        )
    """
    mode: CompactionMode = CompactionMode.NONE
    max_tokens: int = 200000
    window_size: int = 100  # For SLIDING_WINDOW mode
    summary_prompt: Optional[str] = None  # For SUMMARIZE mode

    def should_compact(self, current_tokens: int) -> bool:
        """Check if compaction is needed."""
        if self.mode == CompactionMode.NONE:
            return False
        return current_tokens > self.max_tokens * 0.9


@dataclass
class ToolCache:
    """
    Caching for tool call responses.

    Per Anthropic SDK BetaToolRunner:
    - Caches tool results by tool_use_id
    - Supports TTL-based expiration
    - Useful for expensive or slow tools

    Example:
        cache = ToolCache(ttl_seconds=300)
        cache.set("tool_123", result)
        cached = cache.get("tool_123")
    """
    ttl_seconds: int = 300  # 5 minutes default
    _cache: Dict[str, Tuple[Any, float]] = field(default_factory=dict)

    def get(self, tool_use_id: str) -> Optional[Any]:
        """Get cached result if not expired."""
        if tool_use_id not in self._cache:
            return None
        result, timestamp = self._cache[tool_use_id]
        import time
        if time.time() - timestamp > self.ttl_seconds:
            del self._cache[tool_use_id]
            return None
        return result

    def set(self, tool_use_id: str, result: Any) -> None:
        """Cache a tool result."""
        import time
        self._cache[tool_use_id] = (result, time.time())

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()

    def cleanup_expired(self) -> int:
        """Remove expired entries, return count removed."""
        import time
        now = time.time()
        expired = [k for k, (_, ts) in self._cache.items() if now - ts > self.ttl_seconds]
        for k in expired:
            del self._cache[k]
        return len(expired)


# =============================================================================
# V10.7: LANGGRAPH TOOL CALL REQUEST PATTERN
# Per LangGraph v1.0.6 prebuilt/tool_node.py
# =============================================================================


@dataclass
class ToolCall:
    """
    Represents a tool invocation request.

    Per LangGraph ToolCall:
    - name: Tool name to invoke
    - args: Arguments as dictionary
    - id: Unique identifier for this call
    - type: Always "tool_call"
    """
    name: str
    args: Dict[str, Any]
    id: str
    type: str = "tool_call"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable format."""
        return {
            "name": self.name,
            "args": self.args,
            "id": self.id,
            "type": self.type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        """Parse from dictionary."""
        return cls(
            name=data.get("name", ""),
            args=data.get("args", {}),
            id=data.get("id", ""),
            type=data.get("type", "tool_call"),
        )


@dataclass
class InjectedState:
    """
    State injection marker for tool functions.

    Per LangGraph InjectedState annotation:
    Marks a parameter as receiving injected state rather than
    being part of the tool's input schema.

    Example:
        def my_tool(query: str, state: Annotated[dict, InjectedState()]):
            # state is injected, not from user input
            return state.get("context", "") + query
    """
    key: Optional[str] = None  # Specific key to inject, or entire state

    def __repr__(self) -> str:
        if self.key:
            return f"InjectedState(key={self.key!r})"
        return "InjectedState()"


@dataclass
class InjectedStore:
    """
    Persistent store injection for tools.

    Per LangGraph InjectedStore:
    Provides access to a persistent key-value store across
    tool invocations.

    Example:
        def memory_tool(query: str, store: Annotated[dict, InjectedStore()]):
            # store persists across invocations
            store["last_query"] = query
            return store.get("history", [])
    """
    namespace: Optional[str] = None  # Namespace for store isolation

    def __repr__(self) -> str:
        if self.namespace:
            return f"InjectedStore(namespace={self.namespace!r})"
        return "InjectedStore()"


@dataclass
class ToolRuntime:
    """
    Runtime context for tool execution.

    Per LangGraph ToolRuntime:
    Provides runtime information and services to tools during execution.
    """
    config: Dict[str, Any] = field(default_factory=dict)
    store: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None
    parent_run_id: Optional[str] = None

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get state value."""
        return self.state.get(key, default)


@dataclass
class ToolCallRequest:
    """
    Tool execution request with full context.

    Per LangGraph v1.0.6 ToolCallRequest:
    Contains all information needed to execute a tool including
    the call itself, state, and runtime context.

    Example:
        request = ToolCallRequest(
            tool_call=ToolCall(name="search", args={"q": "test"}, id="1"),
            runtime=ToolRuntime(config={"timeout": 30})
        )
    """
    tool_call: ToolCall
    runtime: ToolRuntime = field(default_factory=ToolRuntime)
    tool_name: Optional[str] = None  # Resolved tool name if different
    tool_args: Optional[Dict[str, Any]] = None  # Resolved args if transformed

    def __post_init__(self) -> None:
        """Set defaults from tool_call."""
        if self.tool_name is None:
            self.tool_name = self.tool_call.name
        if self.tool_args is None:
            self.tool_args = self.tool_call.args.copy()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable format."""
        return {
            "tool_call": self.tool_call.to_dict(),
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "runtime": {
                "config": self.runtime.config,
                "run_id": self.runtime.run_id,
            },
        }


@dataclass
class ToolCallResult:
    """
    Result from tool execution.

    Per LangGraph patterns:
    Contains the result or error from executing a tool call.
    """
    tool_call_id: str
    tool_name: str
    result: Any = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None

    @property
    def is_error(self) -> bool:
        """Check if this is an error result."""
        return self.error is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable format."""
        data: Dict[str, Any] = {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
        }
        if self.error:
            data["error"] = self.error
        else:
            data["result"] = self.result
        if self.duration_ms is not None:
            data["duration_ms"] = self.duration_ms
        return data


# =============================================================================
# V10.7: PYDANTIC-AI INSTRUMENTATION PATTERNS
# Per Pydantic-AI v1.43.0 agent instrumentation settings
# =============================================================================


class InstrumentationLevel(Enum):
    """
    Instrumentation detail level.

    Per Pydantic-AI InstrumentationSettings:
    - OFF: No instrumentation
    - BASIC: Timing and counts only
    - DETAILED: Include inputs/outputs
    - FULL: Include all internal state
    """
    OFF = "off"
    BASIC = "basic"
    DETAILED = "detailed"
    FULL = "full"


@dataclass
class InstrumentationSettings:
    """
    Observability settings for agent/tool execution.

    Per Pydantic-AI v1.43.0:
    Controls what telemetry data is collected during agent runs.

    Example:
        settings = InstrumentationSettings(
            level=InstrumentationLevel.DETAILED,
            trace_tools=True,
            log_to="logfire"
        )
    """
    level: InstrumentationLevel = InstrumentationLevel.OFF
    trace_tools: bool = False
    trace_prompts: bool = False
    trace_outputs: bool = False
    log_to: Optional[str] = None  # "logfire", "stdout", custom handler
    sample_rate: float = 1.0  # 0.0 to 1.0

    @property
    def is_enabled(self) -> bool:
        """Check if any instrumentation is enabled."""
        return self.level != InstrumentationLevel.OFF

    def should_trace(self) -> bool:
        """Check if tracing is enabled based on sample rate."""
        if not self.is_enabled:
            return False
        if self.sample_rate >= 1.0:
            return True
        import random
        return random.random() < self.sample_rate


# =============================================================================
# V10.7: ENHANCED THINKING SESSION
# Enhanced patterns from MCP Sequential Thinking Server
# =============================================================================


@dataclass
class ThinkingCheckpoint:
    """
    Checkpoint in a thinking session for resumption.

    Per MCP Sequential Thinking patterns:
    Allows saving and restoring thinking state.
    """
    checkpoint_id: str
    thought_number: int
    branch_id: Optional[str] = None
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp is None:
            import time
            self.timestamp = time.time()


@dataclass
class BranchMergeResult:
    """
    Result of merging a branch back into main chain.

    Per MCP Sequential Thinking patterns:
    Captures what happened during branch merge.
    """
    merged_thoughts: int
    conflicts: List[int] = field(default_factory=list)  # Conflicting thought numbers
    resolution: str = "append"  # "append", "interleave", "replace"


@dataclass
class EnhancedThinkingSession(ThinkingSession):
    """
    Enhanced thinking session with checkpoints and merge support.

    Extends V10.4 ThinkingSession with patterns from MCP Sequential Thinking:
    - Checkpoint creation and restoration
    - Branch merging
    - Thought revision tracking
    - Session statistics

    Example:
        session = EnhancedThinkingSession(session_id="analysis_1")
        session.add_thought(ThoughtData(thought="Step 1...", thought_number=1, total_thoughts=3))
        checkpoint = session.create_checkpoint()
        # ... continue thinking, can restore if needed
        session.restore_checkpoint(checkpoint.checkpoint_id)
    """
    checkpoints: Dict[str, ThinkingCheckpoint] = field(default_factory=dict)
    revision_history: List[Tuple[int, int]] = field(default_factory=list)  # (original, revision)
    created_at: Optional[float] = None
    updated_at: Optional[float] = None

    def __post_init__(self) -> None:
        import time
        if self.created_at is None:
            self.created_at = time.time()
        self.updated_at = self.created_at

    def add_thought(self, thought: ThoughtData) -> None:
        """Add thought and track revisions."""
        import time
        self.updated_at = time.time()

        # Track revision if applicable
        if thought.is_revision and thought.revises_thought is not None:
            self.revision_history.append((thought.revises_thought, thought.thought_number))

        super().add_thought(thought)

    def create_checkpoint(self, checkpoint_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> ThinkingCheckpoint:
        """Create a checkpoint at current position."""
        import uuid
        if checkpoint_id is None:
            checkpoint_id = str(uuid.uuid4())[:8]

        current_thought = len(self.thoughts)
        if self.current_branch and self.current_branch in self.branches:
            current_thought = len(self.branches[self.current_branch])

        checkpoint = ThinkingCheckpoint(
            checkpoint_id=checkpoint_id,
            thought_number=current_thought,
            branch_id=self.current_branch,
            metadata=metadata or {},
        )
        self.checkpoints[checkpoint_id] = checkpoint
        return checkpoint

    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore to a checkpoint, truncating subsequent thoughts."""
        if checkpoint_id not in self.checkpoints:
            return False

        checkpoint = self.checkpoints[checkpoint_id]

        if checkpoint.branch_id:
            if checkpoint.branch_id in self.branches:
                self.branches[checkpoint.branch_id] = self.branches[checkpoint.branch_id][:checkpoint.thought_number]
            self.current_branch = checkpoint.branch_id
        else:
            self.thoughts = self.thoughts[:checkpoint.thought_number]
            self.current_branch = None

        import time
        self.updated_at = time.time()
        return True

    def merge_branch(self, branch_id: str, resolution: str = "append") -> Optional[BranchMergeResult]:
        """Merge a branch back into the main chain."""
        if branch_id not in self.branches:
            return None

        branch_thoughts = self.branches[branch_id]
        conflicts: List[int] = []

        if resolution == "append":
            # Append all branch thoughts at end
            for thought in branch_thoughts:
                new_thought = ThoughtData(
                    thought=thought.thought,
                    thought_number=len(self.thoughts) + 1,
                    total_thoughts=thought.total_thoughts,
                    next_thought_needed=thought.next_thought_needed,
                    is_revision=False,
                    branch_from_thought=None,
                    branch_id=None,
                )
                self.thoughts.append(new_thought)
        elif resolution == "interleave":
            # Find branch point and interleave
            branch_point = branch_thoughts[0].branch_from_thought if branch_thoughts else 0
            if branch_point:
                for i, thought in enumerate(branch_thoughts):
                    if branch_point + i < len(self.thoughts):
                        conflicts.append(branch_point + i)
                # For now, just append (true interleaving is complex)
                for thought in branch_thoughts:
                    self.thoughts.append(thought)
        elif resolution == "replace":
            # Find branch point and replace from there
            if branch_thoughts:
                branch_point = branch_thoughts[0].branch_from_thought or len(self.thoughts)
                conflicts = list(range(branch_point, len(self.thoughts)))
                self.thoughts = self.thoughts[:branch_point]
                for thought in branch_thoughts:
                    self.thoughts.append(thought)

        # Clean up merged branch
        del self.branches[branch_id]

        import time
        self.updated_at = time.time()

        return BranchMergeResult(
            merged_thoughts=len(branch_thoughts),
            conflicts=conflicts,
            resolution=resolution,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get session statistics."""
        total_branches = len(self.branches)
        total_branch_thoughts = sum(len(b) for b in self.branches.values())

        return {
            "session_id": self.session_id,
            "main_thoughts": len(self.thoughts),
            "total_branches": total_branches,
            "total_branch_thoughts": total_branch_thoughts,
            "total_thoughts": len(self.thoughts) + total_branch_thoughts,
            "revisions": len(self.revision_history),
            "checkpoints": len(self.checkpoints),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "duration_seconds": (self.updated_at - self.created_at) if self.created_at and self.updated_at else None,
        }


# =============================================================================
# V10.7: TOOL RUNNER (ANTHROPIC SDK PATTERN)
# Per Anthropic SDK v0.76.0 BetaToolRunner
# =============================================================================


@dataclass
class ToolRunnerConfig:
    """
    Configuration for ToolRunner.

    Per Anthropic SDK BetaToolRunner:
    Controls tool execution behavior.
    """
    max_iterations: int = 10
    end_strategy: EndStrategy = EndStrategy.EARLY
    compaction: Optional[CompactionControl] = None
    cache: Optional[ToolCache] = None
    instrumentation: Optional[InstrumentationSettings] = None
    timeout_ms: Optional[int] = None
    parallel_tool_calls: bool = True


@dataclass
class ToolRunnerResult:
    """
    Result from a ToolRunner execution.

    Contains all tool call results and metadata from the run.
    """
    tool_results: List[ToolCallResult] = field(default_factory=list)
    iterations: int = 0
    stopped_early: bool = False
    stop_reason: Optional[str] = None
    total_duration_ms: float = 0
    tokens_used: Optional[int] = None

    def add_result(self, result: ToolCallResult) -> None:
        """Add a tool call result."""
        self.tool_results.append(result)
        if result.duration_ms:
            self.total_duration_ms += result.duration_ms

    @property
    def has_errors(self) -> bool:
        """Check if any tool call resulted in error."""
        return any(r.is_error for r in self.tool_results)

    @property
    def successful_results(self) -> List[ToolCallResult]:
        """Get only successful results."""
        return [r for r in self.tool_results if not r.is_error]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable format."""
        return {
            "tool_results": [r.to_dict() for r in self.tool_results],
            "iterations": self.iterations,
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
            "total_duration_ms": self.total_duration_ms,
            "tokens_used": self.tokens_used,
        }


class ToolRunner:
    """
    Advanced tool execution runner.

    Per Anthropic SDK v0.76.0 BetaToolRunner:
    - Manages tool execution loop
    - Supports caching, compaction, and iteration limits
    - Handles early stopping vs exhaustive execution
    - Provides instrumentation hooks

    Example:
        tools = {"search": search_fn, "calculate": calc_fn}
        runner = ToolRunner(tools, config=ToolRunnerConfig(max_iterations=5))

        request = ToolCallRequest(
            tool_call=ToolCall(name="search", args={"q": "test"}, id="1"),
        )
        result = runner.execute(request)
    """

    def __init__(
        self,
        tools: Dict[str, Callable[..., Any]],
        config: Optional[ToolRunnerConfig] = None,
    ) -> None:
        """Initialize tool runner with available tools."""
        self._tools = tools
        self._config = config or ToolRunnerConfig()
        self._iteration_count = 0

    @property
    def tools(self) -> Dict[str, Callable[..., Any]]:
        """Get registered tools."""
        return self._tools

    @property
    def config(self) -> ToolRunnerConfig:
        """Get runner configuration."""
        return self._config

    def execute(self, request: ToolCallRequest) -> ToolCallResult:
        """Execute a single tool call."""
        import time
        start = time.time()

        tool_name = request.tool_name or request.tool_call.name
        tool_args = request.tool_args or request.tool_call.args
        tool_call_id = request.tool_call.id

        # Check cache first
        if self._config.cache:
            cached = self._config.cache.get(tool_call_id)
            if cached is not None:
                return ToolCallResult(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    result=cached,
                    duration_ms=0,
                )

        # Get tool function
        if tool_name not in self._tools:
            return ToolCallResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                error=f"Tool '{tool_name}' not found",
                duration_ms=(time.time() - start) * 1000,
            )

        tool_fn = self._tools[tool_name]

        try:
            # Execute with runtime context if tool supports it
            if "runtime" in tool_args:
                result = tool_fn(**tool_args)
            else:
                # Filter out injected parameters
                clean_args = {k: v for k, v in tool_args.items() if not k.startswith("_")}
                result = tool_fn(**clean_args)

            duration_ms = (time.time() - start) * 1000

            # Cache result
            if self._config.cache:
                self._config.cache.set(tool_call_id, result)

            return ToolCallResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                result=result,
                duration_ms=duration_ms,
            )

        except Exception as e:
            return ToolCallResult(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def execute_batch(self, requests: List[ToolCallRequest]) -> ToolRunnerResult:
        """Execute multiple tool calls according to strategy."""
        import time
        start = time.time()

        runner_result = ToolRunnerResult()

        for i, request in enumerate(requests):
            if i >= self._config.max_iterations:
                runner_result.stopped_early = True
                runner_result.stop_reason = "max_iterations"
                break

            result = self.execute(request)
            runner_result.add_result(result)
            runner_result.iterations = i + 1

            # Early stopping on first success
            if self._config.end_strategy == EndStrategy.EARLY and not result.is_error:
                runner_result.stopped_early = True
                runner_result.stop_reason = "early_success"
                break

        runner_result.total_duration_ms = (time.time() - start) * 1000
        return runner_result


# =============================================================================
# V10.7: THINKING PART PARSER (PYDANTIC-AI PATTERN)
# Per Pydantic-AI v1.43.0 _thinking_part.py
# =============================================================================


@dataclass
class TextPart:
    """Text content part."""
    content: str
    part_type: str = "text"


@dataclass
class ThinkingPart:
    """
    Thinking/reasoning content part.

    Per Pydantic-AI v1.43.0:
    Represents model thinking/reasoning that may be returned
    inline in content (e.g., within <think> tags).
    """
    content: str
    part_type: str = "thinking"


def split_content_into_text_and_thinking(
    content: str,
    thinking_tags: Tuple[str, str] = ("<think>", "</think>"),
) -> List[Union[TextPart, ThinkingPart]]:
    """
    Split content into text and thinking parts.

    Per Pydantic-AI v1.43.0:
    Some models return thinking as inline tags rather than
    separate parts. This extracts them.

    Example:
        parts = split_content_into_text_and_thinking(
            "Let me think... <think>Analyzing options</think> The answer is 42."
        )
        # Returns [TextPart("Let me think... "), ThinkingPart("Analyzing options"), TextPart(" The answer is 42.")]
    """
    start_tag, end_tag = thinking_tags
    parts: List[Union[TextPart, ThinkingPart]] = []

    start_index = content.find(start_tag)
    while start_index >= 0:
        before_think = content[:start_index]
        content = content[start_index + len(start_tag):]

        if before_think:
            parts.append(TextPart(content=before_think))

        end_index = content.find(end_tag)
        if end_index >= 0:
            think_content = content[:end_index]
            content = content[end_index + len(end_tag):]
            parts.append(ThinkingPart(content=think_content))
        else:
            # Unclosed tag - treat rest as text
            parts.append(TextPart(content=content))
            content = ""

        start_index = content.find(start_tag)

    if content:
        parts.append(TextPart(content=content))

    return parts


# =============================================================================
# V10.8: STRUCTLOG PROCESSOR PATTERNS
# Per structlog v24.1.0 contextvars.py and processors.py
# =============================================================================


class LogProcessorProtocol:
    """
    Protocol for log processors.

    Per structlog v24.1.0:
    Processors are callables that transform event dictionaries.
    They form a chain where each processor receives the output of the previous.
    """

    def __call__(
        self,
        logger: Any,
        method_name: str,
        event_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process an event dictionary.

        Args:
            logger: Wrapped logger instance
            method_name: Name of the logging method (debug, info, etc.)
            event_dict: Event dictionary to transform

        Returns:
            Transformed event dictionary
        """
        raise NotImplementedError


@dataclass
class ContextVarBinding:
    """
    A context variable binding for request-scoped logging context.

    Per structlog v24.1.0 contextvars.py:
    Uses contextvars for isolation between concurrent requests.
    Each key-value pair is stored in a separate ContextVar for proper isolation.
    """
    key: str
    value: Any
    prefix: str = "structlog_"

    @property
    def full_key(self) -> str:
        """Get the fully-qualified key with prefix."""
        return f"{self.prefix}{self.key}"


@dataclass
class ContextVarManager:
    """
    Manager for context-local logging context.

    Per structlog v24.1.0 contextvars.py:
    - bind_contextvars: Put keys/values into context-local context
    - unbind_contextvars: Remove keys from context
    - clear_contextvars: Clear all context-local data
    - get_contextvars: Return copy of current context
    - merge_contextvars: Merge context into event dict
    """
    _bindings: Dict[str, ContextVarBinding] = field(default_factory=dict)
    prefix: str = "structlog_"

    def bind(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Bind key-value pairs to context.

        Returns mapping of tokens for later reset.
        """
        tokens = {}
        for key, value in kwargs.items():
            binding = ContextVarBinding(key=key, value=value, prefix=self.prefix)
            self._bindings[key] = binding
            tokens[key] = value  # Simplified token (real impl uses ContextVar.Token)
        return tokens

    def unbind(self, *keys: str) -> None:
        """Remove keys from context."""
        for key in keys:
            self._bindings.pop(key, None)

    def clear(self) -> None:
        """Clear all context bindings."""
        self._bindings.clear()

    def get_context(self) -> Dict[str, Any]:
        """Return copy of current context values."""
        return {k: v.value for k, v in self._bindings.items()}

    def merge_into(self, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge context into event dictionary.

        Uses setdefault to not overwrite existing keys.
        """
        for key, binding in self._bindings.items():
            event_dict.setdefault(key, binding.value)
        return event_dict


@dataclass
class ProcessorChain:
    """
    Chain of log processors.

    Per structlog v24.1.0:
    Processors are composed in sequence. Each receives output of previous.
    """
    processors: List[Callable[[Any, str, Dict[str, Any]], Dict[str, Any]]] = field(default_factory=list)

    def add(self, processor: Callable[[Any, str, Dict[str, Any]], Dict[str, Any]]) -> "ProcessorChain":
        """Add processor to end of chain. Returns self for chaining."""
        self.processors.append(processor)
        return self

    def process(self, logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Run event through all processors in sequence."""
        for processor in self.processors:
            event_dict = processor(logger, method_name, event_dict)
        return event_dict


def key_value_renderer(
    logger: Any,
    method_name: str,
    event_dict: Dict[str, Any]
) -> str:
    """
    Render event as key=value pairs.

    Per structlog v24.1.0 processors.py KeyValueRenderer.
    """
    pairs = []
    event = event_dict.pop("event", "")
    pairs.append(f"event={event}")
    for key, value in sorted(event_dict.items()):
        if isinstance(value, str) and " " in value:
            pairs.append(f'{key}="{value}"')
        else:
            pairs.append(f"{key}={value}")
    return " ".join(pairs)


def logfmt_renderer(
    logger: Any,
    method_name: str,
    event_dict: Dict[str, Any]
) -> str:
    """
    Render event in logfmt format.

    Per structlog v24.1.0 processors.py LogfmtRenderer.
    """
    pairs = []
    for key, value in event_dict.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = str(value).lower()
        elif isinstance(value, str):
            if " " in value or "=" in value or '"' in value:
                value = f'"{value}"'
        pairs.append(f"{key}={value}")
    return " ".join(pairs)


# =============================================================================
# V10.8: OPENTELEMETRY SPAN PATTERNS
# Per OpenTelemetry Python API v1.20.0 trace/span.py
# =============================================================================


class SpanStatusCode(Enum):
    """
    Span status codes.

    Per OpenTelemetry Trace API v1.20.0:
    Status indicates whether the operation completed successfully or had an error.
    """
    UNSET = "UNSET"        # Default, no status set
    OK = "OK"              # Operation completed successfully
    ERROR = "ERROR"        # Operation encountered an error


@dataclass
class SpanStatus:
    """
    Status of a span.

    Per OpenTelemetry Trace API v1.20.0.
    """
    status_code: SpanStatusCode = SpanStatusCode.UNSET
    description: Optional[str] = None

    @classmethod
    def ok(cls) -> "SpanStatus":
        """Create an OK status."""
        return cls(status_code=SpanStatusCode.OK)

    @classmethod
    def error(cls, description: Optional[str] = None) -> "SpanStatus":
        """Create an ERROR status with optional description."""
        return cls(status_code=SpanStatusCode.ERROR, description=description)


@dataclass
class SpanContext:
    """
    Immutable context for a span.

    Per OpenTelemetry Trace API v1.20.0:
    Contains trace_id, span_id, and trace flags for distributed tracing.
    """
    trace_id: str
    span_id: str
    is_remote: bool = False
    trace_flags: int = 1  # Default: sampled
    trace_state: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """Check if context is valid (non-zero IDs)."""
        return bool(self.trace_id and self.span_id)

    @classmethod
    def generate(cls) -> "SpanContext":
        """Generate a new random span context."""
        import uuid
        return cls(
            trace_id=uuid.uuid4().hex,
            span_id=uuid.uuid4().hex[:16]
        )


@dataclass
class SpanLink:
    """
    Link to another span.

    Per OpenTelemetry Trace API v1.20.0:
    Links establish relationships between spans in different traces.
    """
    context: SpanContext
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanEvent:
    """
    Event occurring during a span's lifetime.

    Per OpenTelemetry Trace API v1.20.0:
    Named events with timestamps and optional attributes.
    """
    name: str
    timestamp: Optional[datetime] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class SpanKind(Enum):
    """
    Kind of span.

    Per OpenTelemetry Trace API v1.20.0.
    """
    INTERNAL = "INTERNAL"   # Default, internal operation
    SERVER = "SERVER"       # Server-side handling
    CLIENT = "CLIENT"       # Client-side request
    PRODUCER = "PRODUCER"   # Message producer
    CONSUMER = "CONSUMER"   # Message consumer


@dataclass
class Span:
    """
    A span representing a single operation within a trace.

    Per OpenTelemetry Trace API v1.20.0:
    Spans are the fundamental building blocks of distributed tracing.
    """
    name: str
    context: SpanContext = field(default_factory=SpanContext.generate)
    parent: Optional[SpanContext] = None
    kind: SpanKind = SpanKind.INTERNAL
    status: SpanStatus = field(default_factory=SpanStatus)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    links: List[SpanLink] = field(default_factory=list)
    _ended: bool = False

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now(timezone.utc)

    def set_attribute(self, key: str, value: Any) -> "Span":
        """Set a single attribute."""
        if not self._ended:
            self.attributes[key] = value
        return self

    def set_attributes(self, attributes: Dict[str, Any]) -> "Span":
        """Set multiple attributes."""
        if not self._ended:
            self.attributes.update(attributes)
        return self

    def add_event(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> "Span":
        """Add an event to this span."""
        if not self._ended:
            self.events.append(SpanEvent(
                name=name,
                timestamp=timestamp,
                attributes=attributes or {}
            ))
        return self

    def add_link(self, context: SpanContext, attributes: Optional[Dict[str, Any]] = None) -> "Span":
        """Add a link to another span."""
        if not self._ended:
            self.links.append(SpanLink(context=context, attributes=attributes or {}))
        return self

    def set_status(self, status: SpanStatus) -> "Span":
        """Set the status of this span."""
        if not self._ended:
            self.status = status
        return self

    def end(self, end_time: Optional[datetime] = None) -> None:
        """End this span."""
        if not self._ended:
            self.end_time = end_time or datetime.now(timezone.utc)
            self._ended = True

    def is_recording(self) -> bool:
        """Check if this span is still recording."""
        return not self._ended

    @property
    def duration_ms(self) -> Optional[float]:
        """Get duration in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None


# =============================================================================
# V10.8: SENTRY SCOPE MANAGEMENT PATTERNS
# Per Sentry Python SDK v2.0.0 scope.py
# =============================================================================


class ScopeType(Enum):
    """
    Type of scope.

    Per Sentry Python SDK v2.0.0:
    Different scope types for different isolation needs.
    """
    CURRENT = "current"       # Current scope in context
    ISOLATION = "isolation"   # Isolation scope for request/task
    GLOBAL = "global"         # Global application scope
    MERGED = "merged"         # Merged from all active scopes


@dataclass
class Breadcrumb:
    """
    Breadcrumb for event trail.

    Per Sentry Python SDK v2.0.0:
    Breadcrumbs record events leading up to an error.
    """
    type: str = "default"
    category: Optional[str] = None
    message: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    level: str = "info"
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Sentry-compatible dict."""
        result = {
            "type": self.type,
            "level": self.level,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }
        if self.category:
            result["category"] = self.category
        if self.message:
            result["message"] = self.message
        if self.data:
            result["data"] = self.data
        return result


@dataclass
class ScopeData:
    """
    Data held by a scope.

    Per Sentry Python SDK v2.0.0:
    Scopes hold context data that enriches events.
    """
    user: Optional[Dict[str, Any]] = None
    tags: Dict[str, str] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)
    contexts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    breadcrumbs: List[Breadcrumb] = field(default_factory=list)
    level: Optional[str] = None
    fingerprint: Optional[List[str]] = None
    transaction_name: Optional[str] = None
    span: Optional[Span] = None

    def set_user(self, user: Dict[str, Any]) -> None:
        """Set user information."""
        self.user = user

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag."""
        self.tags[key] = value

    def set_extra(self, key: str, value: Any) -> None:
        """Set extra data."""
        self.extras[key] = value

    def set_context(self, key: str, value: Dict[str, Any]) -> None:
        """Set a context."""
        self.contexts[key] = value

    def add_breadcrumb(self, breadcrumb: Breadcrumb) -> None:
        """Add a breadcrumb."""
        self.breadcrumbs.append(breadcrumb)

    def clear_breadcrumbs(self) -> None:
        """Clear all breadcrumbs."""
        self.breadcrumbs.clear()


@dataclass
class Scope:
    """
    Scope for error context isolation.

    Per Sentry Python SDK v2.0.0:
    Scopes provide context isolation between different parts of code.
    """
    scope_type: ScopeType = ScopeType.CURRENT
    data: ScopeData = field(default_factory=ScopeData)
    _event_processors: List[Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]] = field(default_factory=list)
    _error_processors: List[Callable[[Any], Optional[Any]]] = field(default_factory=list)
    max_breadcrumbs: int = 100

    def set_user(self, user: Dict[str, Any]) -> "Scope":
        """Set user and return self for chaining."""
        self.data.set_user(user)
        return self

    def set_tag(self, key: str, value: str) -> "Scope":
        """Set a tag and return self for chaining."""
        self.data.set_tag(key, value)
        return self

    def set_extra(self, key: str, value: Any) -> "Scope":
        """Set extra data and return self for chaining."""
        self.data.set_extra(key, value)
        return self

    def set_context(self, key: str, value: Dict[str, Any]) -> "Scope":
        """Set context and return self for chaining."""
        self.data.set_context(key, value)
        return self

    def add_breadcrumb(
        self,
        message: Optional[str] = None,
        category: Optional[str] = None,
        type: str = "default",
        level: str = "info",
        data: Optional[Dict[str, Any]] = None
    ) -> "Scope":
        """Add a breadcrumb to the scope."""
        breadcrumb = Breadcrumb(
            type=type,
            category=category,
            message=message,
            data=data or {},
            level=level
        )
        self.data.add_breadcrumb(breadcrumb)
        # Enforce max breadcrumbs
        while len(self.data.breadcrumbs) > self.max_breadcrumbs:
            self.data.breadcrumbs.pop(0)
        return self

    def add_event_processor(
        self,
        processor: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]
    ) -> "Scope":
        """Add an event processor."""
        self._event_processors.append(processor)
        return self

    def apply_to_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply scope data to an event."""
        # Add tags
        if self.data.tags:
            event.setdefault("tags", {}).update(self.data.tags)

        # Add extras
        if self.data.extras:
            event.setdefault("extra", {}).update(self.data.extras)

        # Add user
        if self.data.user:
            event["user"] = self.data.user

        # Add contexts
        if self.data.contexts:
            event.setdefault("contexts", {}).update(self.data.contexts)

        # Add breadcrumbs
        if self.data.breadcrumbs:
            event["breadcrumbs"] = [b.to_dict() for b in self.data.breadcrumbs]

        # Run event processors
        for processor in self._event_processors:
            result = processor(event)
            if result is None:
                return None
            event = result

        return event

    def fork(self) -> "Scope":
        """Create a child scope with copied data."""
        import copy
        new_scope = Scope(scope_type=self.scope_type)
        new_scope.data = copy.deepcopy(self.data)
        new_scope._event_processors = self._event_processors.copy()
        new_scope._error_processors = self._error_processors.copy()
        return new_scope

    def clear(self) -> "Scope":
        """Clear all scope data."""
        self.data = ScopeData()
        return self


@dataclass
class ScopeManager:
    """
    Manager for scope isolation.

    Per Sentry Python SDK v2.0.0:
    Manages scope stack for proper isolation.
    """
    _global_scope: Scope = field(default_factory=lambda: Scope(scope_type=ScopeType.GLOBAL))
    _isolation_scope: Optional[Scope] = None
    _current_scope: Optional[Scope] = None

    def get_global_scope(self) -> Scope:
        """Get the global scope."""
        return self._global_scope

    def get_isolation_scope(self) -> Scope:
        """Get or create isolation scope."""
        if self._isolation_scope is None:
            self._isolation_scope = Scope(scope_type=ScopeType.ISOLATION)
        return self._isolation_scope

    def get_current_scope(self) -> Scope:
        """Get or create current scope."""
        if self._current_scope is None:
            self._current_scope = Scope(scope_type=ScopeType.CURRENT)
        return self._current_scope

    def push_scope(self) -> Scope:
        """Push a new scope and return it."""
        parent = self._current_scope or self._isolation_scope or self._global_scope
        new_scope = parent.fork()
        self._current_scope = new_scope
        return new_scope

    def pop_scope(self) -> None:
        """Pop the current scope."""
        self._current_scope = None

    def get_merged_scope(self) -> Scope:
        """Get a merged scope from all active scopes."""
        merged = Scope(scope_type=ScopeType.MERGED)

        # Merge in order: global -> isolation -> current
        for scope in [self._global_scope, self._isolation_scope, self._current_scope]:
            if scope is not None:
                merged.data.tags.update(scope.data.tags)
                merged.data.extras.update(scope.data.extras)
                merged.data.contexts.update(scope.data.contexts)
                if scope.data.user:
                    merged.data.user = scope.data.user
                merged.data.breadcrumbs.extend(scope.data.breadcrumbs)

        return merged


# =============================================================================
# V10.8: PYRIBS ARCHIVE PATTERNS
# Per pyribs v0.9.0 archives/_archive_base.py
# =============================================================================


@dataclass
class Elite:
    """
    An elite (high-performing solution) in the archive.

    Per pyribs v0.9.0:
    Each elite contains solution parameters, objective value,
    and behavior measures.
    """
    solution: List[float]
    objective: float
    measures: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "solution": self.solution,
            "objective": self.objective,
            "measures": self.measures,
            "metadata": self.metadata,
            "index": self.index
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Elite":
        """Create from dict."""
        return cls(
            solution=data["solution"],
            objective=data["objective"],
            measures=data["measures"],
            metadata=data.get("metadata", {}),
            index=data.get("index")
        )


class ArchiveAddStatus(Enum):
    """
    Status of an add operation.

    Per pyribs v0.9.0.
    """
    NEW = "new"             # Added to empty cell
    IMPROVE = "improve"     # Replaced existing with better
    NOT_ADDED = "not_added" # Did not improve over existing


@dataclass
class ArchiveAddResult:
    """
    Result of adding to an archive.

    Per pyribs v0.9.0.
    """
    status: ArchiveAddStatus
    elite: Optional[Elite] = None
    previous_objective: Optional[float] = None
    improvement: Optional[float] = None


@dataclass
class ArchiveStats:
    """
    Statistics about an archive.

    Per pyribs v0.9.0.
    """
    num_elites: int = 0
    coverage: float = 0.0  # Fraction of cells filled
    qd_score: float = 0.0  # Sum of all objective values
    obj_max: Optional[float] = None
    obj_min: Optional[float] = None
    obj_mean: Optional[float] = None

    def update(self, elites: List[Elite], total_cells: int) -> None:
        """Update stats from elites."""
        self.num_elites = len(elites)
        self.coverage = len(elites) / total_cells if total_cells > 0 else 0.0
        if elites:
            objectives = [e.objective for e in elites]
            self.qd_score = sum(objectives)
            self.obj_max = max(objectives)
            self.obj_min = min(objectives)
            self.obj_mean = self.qd_score / len(objectives)


@dataclass
class Archive:
    """
    Archive for storing elites in quality-diversity optimization.

    Per pyribs v0.9.0 ArchiveBase:
    An archive stores elites, each consisting of a solution with
    evaluated objective and behavior measures.

    This simplified implementation uses a dict-based grid.
    For production, use pyribs directly.
    """
    solution_dim: int
    measure_dim: int
    measure_bounds: List[Tuple[float, float]]
    cells_per_dim: int = 10
    _elites: Dict[Tuple[int, ...], Elite] = field(default_factory=dict)
    _stats: ArchiveStats = field(default_factory=ArchiveStats)

    def __post_init__(self):
        """Calculate total cells."""
        self._total_cells = self.cells_per_dim ** self.measure_dim

    def _measures_to_index(self, measures: List[float]) -> Tuple[int, ...]:
        """Convert behavior measures to grid index."""
        indices = []
        for i, m in enumerate(measures):
            low, high = self.measure_bounds[i]
            # Clamp to bounds
            m = max(low, min(high, m))
            # Normalize to [0, 1]
            normalized = (m - low) / (high - low) if high > low else 0.5
            # Convert to cell index
            idx = int(normalized * self.cells_per_dim)
            idx = min(idx, self.cells_per_dim - 1)
            indices.append(idx)
        return tuple(indices)

    def add(
        self,
        solution: List[float],
        objective: float,
        measures: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ArchiveAddResult:
        """
        Add a solution to the archive.

        Returns:
            ArchiveAddResult indicating what happened.
        """
        index = self._measures_to_index(measures)
        elite = Elite(
            solution=solution,
            objective=objective,
            measures=measures,
            metadata=metadata or {},
            index=hash(index)
        )

        if index not in self._elites:
            # New cell
            self._elites[index] = elite
            self._update_stats()
            return ArchiveAddResult(
                status=ArchiveAddStatus.NEW,
                elite=elite
            )

        existing = self._elites[index]
        if objective > existing.objective:
            # Improvement
            self._elites[index] = elite
            improvement = objective - existing.objective
            self._update_stats()
            return ArchiveAddResult(
                status=ArchiveAddStatus.IMPROVE,
                elite=elite,
                previous_objective=existing.objective,
                improvement=improvement
            )

        # Not added
        return ArchiveAddResult(
            status=ArchiveAddStatus.NOT_ADDED,
            previous_objective=existing.objective
        )

    def _update_stats(self) -> None:
        """Update archive statistics."""
        self._stats.update(list(self._elites.values()), self._total_cells)

    @property
    def stats(self) -> ArchiveStats:
        """Get archive statistics."""
        return self._stats

    def __len__(self) -> int:
        """Number of elites in archive."""
        return len(self._elites)

    @property
    def empty(self) -> bool:
        """Check if archive is empty."""
        return len(self._elites) == 0

    def sample_elites(self, n: int) -> List[Elite]:
        """Sample n random elites from the archive."""
        import random
        elites = list(self._elites.values())
        if n >= len(elites):
            return elites
        return random.sample(elites, n)

    def data(self) -> Dict[str, List[Any]]:
        """
        Return all elite data as dict of lists.

        Per pyribs pattern for batch data access.
        """
        elites = list(self._elites.values())
        return {
            "solution": [e.solution for e in elites],
            "objective": [e.objective for e in elites],
            "measures": [e.measures for e in elites],
            "metadata": [e.metadata for e in elites]
        }

    def clear(self) -> None:
        """Clear the archive."""
        self._elites.clear()
        self._stats = ArchiveStats()


# =============================================================================
# V10.8: THINKING ARCHIVE (Quality-Diversity for Reasoning)
# Combines pyribs patterns with thinking session for diverse reasoning
# =============================================================================


@dataclass
class ThinkingElite:
    """
    An elite thinking branch.

    Combines pyribs Elite pattern with thinking for
    quality-diversity optimization of reasoning.
    """
    thought_id: str
    content: str
    quality_score: float  # Objective
    diversity_measures: List[float]  # Behavior measures
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None

    def to_elite(self) -> Elite:
        """Convert to generic Elite."""
        # Simple hash of content as solution representation
        solution = [float(hash(self.content) % 10000) / 10000]
        return Elite(
            solution=solution,
            objective=self.quality_score,
            measures=self.diversity_measures,
            metadata={
                "thought_id": self.thought_id,
                "parent_id": self.parent_id,
                **self.metadata
            }
        )


class ThinkingArchive:
    """
    Archive for quality-diversity reasoning.

    Stores diverse high-quality thinking branches
    using MAP-Elites inspired patterns from pyribs.
    """

    def __init__(
        self,
        measure_bounds: Optional[List[Tuple[float, float]]] = None,
        cells_per_dim: int = 5
    ):
        """
        Initialize the thinking archive.

        Args:
            measure_bounds: Bounds for diversity measures (default 2D unit bounds)
            cells_per_dim: Number of cells per dimension in the grid
        """
        self.measure_bounds = measure_bounds or [(0.0, 1.0), (0.0, 1.0)]
        self.cells_per_dim = cells_per_dim
        self._thoughts: Dict[str, ThinkingElite] = {}
        self._archive = Archive(
            solution_dim=1,
            measure_dim=len(self.measure_bounds),
            measure_bounds=self.measure_bounds,
            cells_per_dim=self.cells_per_dim
        )

    def add_thought(
        self,
        thought_id: str,
        content: str,
        quality_score: float,
        diversity_measures: List[float],
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ArchiveAddResult:
        """
        Add a thought to the archive.

        Args:
            thought_id: Unique identifier for the thought
            content: The thought content
            quality_score: How good this thought is (higher = better)
            diversity_measures: Behavioral measures for placement
            parent_id: Optional parent thought ID
            metadata: Additional metadata

        Returns:
            Result of the add operation
        """
        elite = ThinkingElite(
            thought_id=thought_id,
            content=content,
            quality_score=quality_score,
            diversity_measures=diversity_measures,
            parent_id=parent_id,
            metadata=metadata or {}
        )

        self._thoughts[thought_id] = elite

        # Add to underlying archive
        return self._archive.add(
            solution=[float(hash(content) % 10000) / 10000],
            objective=quality_score,
            measures=diversity_measures,
            metadata={"thought_id": thought_id}
        )

    @property
    def stats(self) -> ArchiveStats:
        """Get archive statistics."""
        return self._archive.stats

    def sample_diverse_thoughts(self, n: int) -> List[ThinkingElite]:
        """Sample n diverse thoughts from the archive."""
        sampled_elites = self._archive.sample_elites(n)
        result = []
        for elite in sampled_elites:
            thought_id = elite.metadata.get("thought_id")
            if thought_id and thought_id in self._thoughts:
                result.append(self._thoughts[thought_id])
        return result

    def get_best_thought(self) -> Optional[ThinkingElite]:
        """Get the highest quality thought."""
        if not self._thoughts:
            return None
        return max(self._thoughts.values(), key=lambda t: t.quality_score)

    def get_thought(self, thought_id: str) -> Optional[ThinkingElite]:
        """Get a specific thought by ID."""
        return self._thoughts.get(thought_id)

    def __len__(self) -> int:
        """Number of thoughts in archive."""
        return len(self._archive)


# =============================================================================
# V10.9: Task-Master PRD Patterns
# Based on: https://github.com/eyaltoledano/claude-task-master
# =============================================================================


class TaskPriority(Enum):
    """Task priority levels based on task-master patterns."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DEFERRED = "deferred"


class ComplexityLevel(Enum):
    """Task complexity levels for sizing and estimation."""
    TRIVIAL = "trivial"       # < 1 hour
    SIMPLE = "simple"         # 1-4 hours
    MODERATE = "moderate"     # 4-8 hours
    COMPLEX = "complex"       # 1-3 days
    EPIC = "epic"             # > 3 days (requires decomposition)


class ToolMode(Enum):
    """
    Task-master tool mode configuration.

    Controls how many tools are loaded for token efficiency:
    - CORE: 7 essential tools (get_tasks, next_task, set_status, etc.)
    - STANDARD: 15 tools (core + project setup, analysis)
    - ALL: 36 tools (complete set including research, tags, dependencies)
    """
    CORE = "core"           # 7 tools - minimal for execution
    STANDARD = "standard"   # 15 tools - balanced
    ALL = "all"             # 36 tools - full capability


@dataclass
class PRDTask:
    """
    Task parsed from a Product Requirements Document.

    Based on task-master's PRD parsing workflow that converts
    natural language requirements into structured tasks.
    """
    id: str
    title: str
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    complexity: ComplexityLevel = ComplexityLevel.MODERATE
    dependencies: List[str] = field(default_factory=list)
    subtasks: List["PRDTask"] = field(default_factory=list)
    status: str = "pending"
    tags: List[str] = field(default_factory=list)
    test_strategy: Optional[str] = None
    acceptance_criteria: List[str] = field(default_factory=list)
    estimated_hours: Optional[float] = None
    assigned_to: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "complexity": self.complexity.value,
            "dependencies": self.dependencies,
            "subtasks": [s.to_dict() for s in self.subtasks],
            "status": self.status,
            "tags": self.tags,
            "test_strategy": self.test_strategy,
            "acceptance_criteria": self.acceptance_criteria,
            "estimated_hours": self.estimated_hours,
            "assigned_to": self.assigned_to,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PRDTask":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            priority=TaskPriority(data.get("priority", "medium")),
            complexity=ComplexityLevel(data.get("complexity", "moderate")),
            dependencies=data.get("dependencies", []),
            subtasks=[cls.from_dict(s) for s in data.get("subtasks", [])],
            status=data.get("status", "pending"),
            tags=data.get("tags", []),
            test_strategy=data.get("test_strategy"),
            acceptance_criteria=data.get("acceptance_criteria", []),
            estimated_hours=data.get("estimated_hours"),
            assigned_to=data.get("assigned_to"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None
        )


@dataclass
class ComplexityAnalysis:
    """
    Task complexity analysis result.

    Based on task-master's analyze_project_complexity tool
    that assesses task sizing and decomposition needs.
    """
    task_id: str
    level: ComplexityLevel
    score: float  # 0.0 to 1.0
    factors: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    requires_decomposition: bool = False
    suggested_subtask_count: int = 0
    confidence: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "level": self.level.value,
            "score": self.score,
            "factors": self.factors,
            "recommendations": self.recommendations,
            "requires_decomposition": self.requires_decomposition,
            "suggested_subtask_count": self.suggested_subtask_count,
            "confidence": self.confidence
        }


@dataclass
class TaggedTaskList:
    """
    Tagged task list for multi-context management.

    Based on task-master's tagged task lists that allow
    separate, isolated lists for different features/branches.
    """
    tag: str
    tasks: List[PRDTask] = field(default_factory=list)
    description: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_default: bool = False

    def add_task(self, task: PRDTask) -> None:
        """Add a task to this list."""
        if self.tag not in task.tags:
            task.tags.append(self.tag)
        self.tasks.append(task)

    def get_next_task(self) -> Optional[PRDTask]:
        """Get the next executable task (no pending dependencies)."""
        completed_ids = {t.id for t in self.tasks if t.status == "completed"}
        for task in self.tasks:
            if task.status == "pending":
                deps_met = all(dep in completed_ids for dep in task.dependencies)
                if deps_met:
                    return task
        return None

    def get_tasks_by_status(self, status: str) -> List[PRDTask]:
        """Get all tasks with the given status."""
        return [t for t in self.tasks if t.status == status]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tag": self.tag,
            "tasks": [t.to_dict() for t in self.tasks],
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "is_default": self.is_default
        }


class TaskDependencyGraph:
    """
    Task dependency graph for execution ordering.

    Implements topological sort to determine correct
    task execution sequence based on dependencies.
    """

    def __init__(self) -> None:
        self._tasks: Dict[str, PRDTask] = {}
        self._adjacency: Dict[str, List[str]] = {}

    def add_task(self, task: PRDTask) -> None:
        """Add a task to the graph."""
        self._tasks[task.id] = task
        self._adjacency[task.id] = task.dependencies.copy()

    def add_tasks(self, tasks: List[PRDTask]) -> None:
        """Add multiple tasks to the graph."""
        for task in tasks:
            self.add_task(task)

    def get_execution_order(self) -> List[str]:
        """
        Get topologically sorted execution order.

        Returns list of task IDs in order of execution,
        respecting all dependency constraints.
        """
        in_degree: Dict[str, int] = {tid: 0 for tid in self._tasks}

        for task_id, deps in self._adjacency.items():
            for dep in deps:
                if dep in self._tasks:
                    in_degree[task_id] += 1

        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        result: List[str] = []

        while queue:
            # Sort by priority for deterministic ordering
            queue.sort(key=lambda x: (
                self._tasks[x].priority.value if x in self._tasks else "z"
            ))
            task_id = queue.pop(0)
            result.append(task_id)

            # Reduce in-degree for dependent tasks
            for other_id, deps in self._adjacency.items():
                if task_id in deps:
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        queue.append(other_id)

        return result

    def get_blocked_tasks(self) -> List[str]:
        """Get tasks that are blocked by incomplete dependencies."""
        completed = {tid for tid, t in self._tasks.items() if t.status == "completed"}
        blocked = []
        for task_id, task in self._tasks.items():
            if task.status != "completed":
                if not all(dep in completed for dep in task.dependencies):
                    blocked.append(task_id)
        return blocked

    def get_ready_tasks(self) -> List[PRDTask]:
        """Get tasks ready for execution (all deps complete)."""
        completed = {tid for tid, t in self._tasks.items() if t.status == "completed"}
        ready = []
        for task_id, task in self._tasks.items():
            if task.status == "pending":
                if all(dep in completed for dep in task.dependencies):
                    ready.append(task)
        return ready


@dataclass
class PRDWorkflow:
    """
    PRD-to-implementation workflow manager.

    Based on task-master's core workflow:
    1. Parse PRD → 2. Generate tasks → 3. Analyze complexity
    4. Expand subtasks → 5. Execute in order → 6. Verify
    """
    prd_path: Optional[str] = None
    task_lists: Dict[str, TaggedTaskList] = field(default_factory=dict)
    default_tag: str = "master"
    tool_mode: ToolMode = ToolMode.STANDARD
    dependency_graph: Optional[TaskDependencyGraph] = None

    def __post_init__(self) -> None:
        """Initialize with default task list."""
        if self.default_tag not in self.task_lists:
            self.task_lists[self.default_tag] = TaggedTaskList(
                tag=self.default_tag,
                is_default=True
            )
        if self.dependency_graph is None:
            self.dependency_graph = TaskDependencyGraph()

    def parse_prd(self, content: str, num_tasks: Optional[int] = None) -> List[PRDTask]:
        """
        Parse PRD content into tasks.

        This is a simplified pattern - actual implementation
        would use AI to parse natural language.
        """
        # Extract sections that look like requirements
        tasks: List[PRDTask] = []
        lines = content.split("\n")
        current_task: Optional[Dict[str, Any]] = None
        task_counter = 1

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for numbered items or bullet points
            if re.match(r"^(\d+\.|\-|\*)\s+", line):
                if current_task:
                    task = PRDTask(
                        id=f"task_{task_counter}",
                        title=current_task.get("title", f"Task {task_counter}"),
                        description=current_task.get("description", ""),
                    )
                    tasks.append(task)
                    task_counter += 1

                title = re.sub(r"^(\d+\.|\-|\*)\s+", "", line)
                current_task = {"title": title, "description": ""}
            elif current_task:
                current_task["description"] += line + " "

        # Add last task
        if current_task:
            task = PRDTask(
                id=f"task_{task_counter}",
                title=current_task.get("title", f"Task {task_counter}"),
                description=current_task.get("description", "").strip(),
            )
            tasks.append(task)

        # Limit if requested
        if num_tasks and len(tasks) > num_tasks:
            tasks = tasks[:num_tasks]

        # Add to default list and graph
        for task in tasks:
            self.task_lists[self.default_tag].add_task(task)
            if self.dependency_graph is not None:
                self.dependency_graph.add_task(task)

        return tasks

    def get_next_task(self, tag: Optional[str] = None) -> Optional[PRDTask]:
        """Get the next task to execute."""
        tag = tag or self.default_tag
        if tag in self.task_lists:
            return self.task_lists[tag].get_next_task()
        return None

    def set_task_status(self, task_id: str, status: str, tag: Optional[str] = None) -> bool:
        """Update task status."""
        tag = tag or self.default_tag
        if tag in self.task_lists:
            for task in self.task_lists[tag].tasks:
                if task.id == task_id:
                    task.status = status
                    task.updated_at = datetime.now(timezone.utc)
                    return True
        return False

    def expand_task(
        self,
        task_id: str,
        num_subtasks: int = 3,
        tag: Optional[str] = None
    ) -> List[PRDTask]:
        """
        Expand a task into subtasks.

        This is a pattern placeholder - actual implementation
        would use AI for intelligent decomposition.
        """
        tag = tag or self.default_tag
        subtasks: List[PRDTask] = []

        if tag in self.task_lists:
            for task in self.task_lists[tag].tasks:
                if task.id == task_id:
                    for i in range(num_subtasks):
                        subtask = PRDTask(
                            id=f"{task_id}_sub{i+1}",
                            title=f"{task.title} - Part {i+1}",
                            description=f"Subtask {i+1} of {task.title}",
                            priority=task.priority,
                            dependencies=[task_id] if i == 0 else [f"{task_id}_sub{i}"],
                            tags=task.tags.copy()
                        )
                        subtasks.append(subtask)
                        task.subtasks.append(subtask)
                    break

        return subtasks

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prd_path": self.prd_path,
            "task_lists": {k: v.to_dict() for k, v in self.task_lists.items()},
            "default_tag": self.default_tag,
            "tool_mode": self.tool_mode.value
        }


# =============================================================================
# V10.9: Shrimp Task Manager Chain-of-Thought Patterns
# Based on: https://github.com/cjo4m06/mcp-shrimp-task-manager
# =============================================================================


class TaskWorkflowMode(Enum):
    """
    Shrimp task manager workflow modes.

    Each mode represents a different operational approach
    for structured task execution.
    """
    PLANNING = "planning"       # Deep analysis before implementation
    EXECUTION = "execution"     # Active implementation mode
    CONTINUOUS = "continuous"   # Sequential auto-execution
    RESEARCH = "research"       # Systematic exploration mode
    REFLECTION = "reflection"   # Review and improve approach


@dataclass
class ChainOfThought:
    """
    Chain-of-thought reasoning structure.

    Based on shrimp task manager's structured reasoning
    that guides step-by-step problem solving.
    """
    id: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    current_step: int = 0
    mode: TaskWorkflowMode = TaskWorkflowMode.PLANNING
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_step(
        self,
        thought: str,
        observation: Optional[str] = None,
        action: Optional[str] = None,
        result: Optional[str] = None
    ) -> int:
        """
        Add a reasoning step.

        Returns the step index.
        """
        step = {
            "index": len(self.steps),
            "thought": thought,
            "observation": observation,
            "action": action,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self.steps.append(step)
        return step["index"]

    def update_step(
        self,
        index: int,
        observation: Optional[str] = None,
        action: Optional[str] = None,
        result: Optional[str] = None
    ) -> bool:
        """Update an existing step."""
        if 0 <= index < len(self.steps):
            if observation is not None:
                self.steps[index]["observation"] = observation
            if action is not None:
                self.steps[index]["action"] = action
            if result is not None:
                self.steps[index]["result"] = result
            return True
        return False

    def advance(self) -> bool:
        """Move to the next step."""
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            return True
        return False

    def get_current_step(self) -> Optional[Dict[str, Any]]:
        """Get the current step."""
        if 0 <= self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None

    def to_prompt(self) -> str:
        """Convert chain to a reasoning prompt."""
        lines = [f"Chain of Thought (Mode: {self.mode.value}):", ""]
        for step in self.steps:
            lines.append(f"Step {step['index'] + 1}: {step['thought']}")
            if step.get("observation"):
                lines.append(f"  Observation: {step['observation']}")
            if step.get("action"):
                lines.append(f"  Action: {step['action']}")
            if step.get("result"):
                lines.append(f"  Result: {step['result']}")
            lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "steps": self.steps,
            "current_step": self.current_step,
            "mode": self.mode.value,
            "context": self.context,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class ThoughtProcess:
    """
    Thought process tool pattern.

    Based on shrimp's process_thought tool that enables
    structured step-by-step reasoning before task analysis.
    """
    chain: ChainOfThought
    max_steps: int = 10
    enable_reflection: bool = True
    auto_advance: bool = False

    def process(
        self,
        thought: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a thought and add to chain.

        Returns the processed step with any reflections.
        """
        if context:
            self.chain.context.update(context)

        step_idx = self.chain.add_step(thought=thought)

        result = {
            "step_index": step_idx,
            "thought": thought,
            "chain_length": len(self.chain.steps),
            "mode": self.chain.mode.value
        }

        # Add reflection if enabled
        if self.enable_reflection and len(self.chain.steps) > 1:
            result["reflection"] = self._generate_reflection()

        if self.auto_advance:
            self.chain.advance()

        return result

    def _generate_reflection(self) -> str:
        """Generate a reflection on the thought chain."""
        step_count = len(self.chain.steps)
        if step_count <= 1:
            return "Initial thought captured."

        recent_thoughts = [s["thought"] for s in self.chain.steps[-3:]]
        return f"Reasoning chain has {step_count} steps. Recent focus: {recent_thoughts[-1][:50]}..."

    def finalize(self) -> Dict[str, Any]:
        """Finalize the thought process and prepare for action."""
        return {
            "id": self.chain.id,
            "total_steps": len(self.chain.steps),
            "mode": self.chain.mode.value,
            "summary": self.chain.to_prompt(),
            "ready_for_execution": True
        }


@dataclass
class TaskDependency:
    """
    Task dependency tracking.

    Based on shrimp's automatic dependency management
    that tracks relationships between tasks.
    """
    task_id: str
    depends_on: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)
    dependency_type: str = "finish_to_start"  # finish_to_start, start_to_start, etc.
    is_satisfied: bool = False

    def check_satisfaction(self, completed_tasks: Set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        self.is_satisfied = all(dep in completed_tasks for dep in self.depends_on)
        return self.is_satisfied

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "depends_on": self.depends_on,
            "blocks": self.blocks,
            "dependency_type": self.dependency_type,
            "is_satisfied": self.is_satisfied
        }


@dataclass
class PersistentTaskMemory:
    """
    Persistent task memory for cross-session state.

    Based on shrimp's persistent memory that preserves
    tasks and progress across sessions.
    """
    data_dir: Path = field(default_factory=lambda: Path(".taskmaster"))
    profile: str = "default"
    tasks: Dict[str, PRDTask] = field(default_factory=dict)
    chains: Dict[str, ChainOfThought] = field(default_factory=dict)
    dependencies: Dict[str, TaskDependency] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure data directory exists."""
        self.data_dir = Path(self.data_dir)

    def save(self) -> bool:
        """Save current state to disk."""
        try:
            profile_dir = self.data_dir / self.profile
            profile_dir.mkdir(parents=True, exist_ok=True)

            state = {
                "profile": self.profile,
                "tasks": {k: v.to_dict() for k, v in self.tasks.items()},
                "chains": {k: v.to_dict() for k, v in self.chains.items()},
                "dependencies": {k: v.to_dict() for k, v in self.dependencies.items()},
                "metadata": self.metadata,
                "saved_at": datetime.now(timezone.utc).isoformat()
            }

            state_file = profile_dir / "state.json"
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)

            return True
        except Exception:
            return False

    def load(self) -> bool:
        """Load state from disk."""
        try:
            state_file = self.data_dir / self.profile / "state.json"
            if not state_file.exists():
                return False

            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)

            self.tasks = {
                k: PRDTask.from_dict(v) for k, v in state.get("tasks", {}).items()
            }
            # Chains and dependencies would need similar from_dict methods
            self.metadata = state.get("metadata", {})

            return True
        except Exception:
            return False

    def add_task(self, task: PRDTask) -> None:
        """Add a task to memory."""
        self.tasks[task.id] = task
        self.dependencies[task.id] = TaskDependency(
            task_id=task.id,
            depends_on=task.dependencies
        )

    def add_chain(self, chain: ChainOfThought) -> None:
        """Add a thought chain to memory."""
        self.chains[chain.id] = chain

    def get_status(self) -> Dict[str, Any]:
        """Get memory status summary."""
        return {
            "profile": self.profile,
            "task_count": len(self.tasks),
            "chain_count": len(self.chains),
            "dependency_count": len(self.dependencies),
            "pending_tasks": sum(1 for t in self.tasks.values() if t.status == "pending"),
            "completed_tasks": sum(1 for t in self.tasks.values() if t.status == "completed")
        }


@dataclass
class StructuredWorkflow:
    """
    Structured workflow for guided execution.

    Based on shrimp's five operational modes that guide
    the development process from planning to reflection.
    """
    mode: TaskWorkflowMode = TaskWorkflowMode.PLANNING
    memory: Optional[PersistentTaskMemory] = None
    active_chain: Optional[ChainOfThought] = None
    current_task: Optional[PRDTask] = None
    project_rules: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize memory if not provided."""
        if self.memory is None:
            self.memory = PersistentTaskMemory()

    def _get_memory(self) -> PersistentTaskMemory:
        """Get memory, ensuring it's initialized."""
        if self.memory is None:
            self.memory = PersistentTaskMemory()
        return self.memory

    def plan_task(self, description: str) -> ChainOfThought:
        """
        Enter planning mode and create a thought chain.

        Based on shrimp's plan_task tool.
        """
        self.mode = TaskWorkflowMode.PLANNING
        chain = ChainOfThought(
            id=f"chain_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            mode=TaskWorkflowMode.PLANNING
        )
        chain.add_step(
            thought=f"Planning task: {description}",
            observation="Initializing planning phase"
        )
        self.active_chain = chain
        self._get_memory().add_chain(chain)
        return chain

    def execute_task(self, task: PRDTask) -> Dict[str, Any]:
        """
        Enter execution mode for a specific task.

        Based on shrimp's execute_task tool.
        """
        self.mode = TaskWorkflowMode.EXECUTION
        self.current_task = task
        task.status = "in_progress"
        task.updated_at = datetime.now(timezone.utc)

        return {
            "mode": self.mode.value,
            "task_id": task.id,
            "task_title": task.title,
            "started_at": datetime.now(timezone.utc).isoformat()
        }

    def enter_research_mode(self, topic: str) -> Dict[str, Any]:
        """
        Enter research mode for systematic exploration.

        Based on shrimp's research_mode tool.
        """
        self.mode = TaskWorkflowMode.RESEARCH
        chain = ChainOfThought(
            id=f"research_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            mode=TaskWorkflowMode.RESEARCH,
            context={"topic": topic}
        )
        chain.add_step(
            thought=f"Beginning research on: {topic}",
            observation="Entering systematic exploration phase"
        )
        self.active_chain = chain

        return {
            "mode": self.mode.value,
            "chain_id": chain.id,
            "topic": topic
        }

    def reflect_on_task(self, task_id: str) -> Dict[str, Any]:
        """
        Enter reflection mode to review and improve.

        Based on shrimp's reflect_task tool.
        """
        self.mode = TaskWorkflowMode.REFLECTION
        memory = self._get_memory()
        task = memory.tasks.get(task_id)

        if not task:
            return {"error": f"Task {task_id} not found"}

        return {
            "mode": self.mode.value,
            "task_id": task_id,
            "task_status": task.status,
            "reflection_started": datetime.now(timezone.utc).isoformat()
        }

    def verify_task(self, task_id: str) -> Dict[str, Any]:
        """
        Verify task completion against acceptance criteria.

        Based on shrimp's verify_task tool.
        """
        memory = self._get_memory()
        task = memory.tasks.get(task_id)
        if not task:
            return {"verified": False, "error": "Task not found"}

        # Check acceptance criteria
        criteria_met = len(task.acceptance_criteria) == 0  # Default true if no criteria

        return {
            "task_id": task_id,
            "verified": criteria_met,
            "criteria_count": len(task.acceptance_criteria),
            "status": task.status
        }

    def complete_task(self, task_id: str) -> bool:
        """Mark a task as completed."""
        memory = self._get_memory()
        task = memory.tasks.get(task_id)
        if task:
            task.status = "completed"
            task.updated_at = datetime.now(timezone.utc)
            # Update dependency satisfaction
            completed = {tid for tid, t in memory.tasks.items() if t.status == "completed"}
            for dep in memory.dependencies.values():
                dep.check_satisfaction(completed)
            return True
        return False

    def init_project_rules(self, rules: Dict[str, Any]) -> None:
        """
        Initialize project-specific rules.

        Based on shrimp's init_project_rules tool.
        """
        self.project_rules = rules
        self._get_memory().metadata["project_rules"] = rules

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mode": self.mode.value,
            "active_chain_id": self.active_chain.id if self.active_chain else None,
            "current_task_id": self.current_task.id if self.current_task else None,
            "project_rules": self.project_rules,
            "memory_status": self.memory.get_status() if self.memory else None
        }


# =============================================================================
# V10.9: Semgrep Security Audit Patterns
# Based on: https://github.com/semgrep/mcp
# =============================================================================


class SecuritySeverity(Enum):
    """Security finding severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RuleCategory(Enum):
    """Security rule categories."""
    INJECTION = "injection"           # SQL, command, XSS
    AUTH = "authentication"           # Auth bypass, weak auth
    CRYPTO = "cryptography"           # Weak crypto, key management
    SECRETS = "secrets"               # Hardcoded credentials
    CONFIGURATION = "configuration"   # Misconfigurations
    BEST_PRACTICES = "best-practices" # Code quality
    SUPPLY_CHAIN = "supply-chain"     # Dependency issues


@dataclass
class SecurityFinding:
    """
    Security vulnerability finding.

    Based on Semgrep's finding format for
    reporting discovered vulnerabilities.
    """
    id: str
    rule_id: str
    file_path: str
    line_start: int
    line_end: int
    severity: SecuritySeverity
    category: RuleCategory
    message: str
    code_snippet: Optional[str] = None
    fix_suggestion: Optional[str] = None
    cwe_id: Optional[str] = None
    owasp_category: Optional[str] = None
    confidence: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "rule_id": self.rule_id,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
            "code_snippet": self.code_snippet,
            "fix_suggestion": self.fix_suggestion,
            "cwe_id": self.cwe_id,
            "owasp_category": self.owasp_category,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


@dataclass
class ASTNode:
    """
    Abstract Syntax Tree node.

    Based on Semgrep's get_abstract_syntax_tree tool
    for semantic code analysis.
    """
    node_type: str
    value: Optional[str] = None
    children: List["ASTNode"] = field(default_factory=list)
    line: Optional[int] = None
    column: Optional[int] = None
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_type": self.node_type,
            "value": self.value,
            "children": [c.to_dict() for c in self.children],
            "line": self.line,
            "column": self.column,
            "end_line": self.end_line,
            "end_column": self.end_column,
            "metadata": self.metadata
        }

    def find_nodes(self, node_type: str) -> List["ASTNode"]:
        """Find all descendant nodes of a given type."""
        result = []
        if self.node_type == node_type:
            result.append(self)
        for child in self.children:
            result.extend(child.find_nodes(node_type))
        return result


@dataclass
class SemgrepRule:
    """
    Semgrep rule definition.

    Based on Semgrep's rule schema for defining
    custom security scanning rules.
    """
    id: str
    pattern: str
    message: str
    severity: SecuritySeverity = SecuritySeverity.WARNING
    category: RuleCategory = RuleCategory.BEST_PRACTICES
    languages: List[str] = field(default_factory=lambda: ["python"])
    fix: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_yaml(self) -> str:
        """Convert to YAML rule format."""
        rule = {
            "id": self.id,
            "pattern": self.pattern,
            "message": self.message,
            "severity": self.severity.value.upper(),
            "languages": self.languages,
            "metadata": {
                "category": self.category.value,
                **self.metadata
            }
        }
        if self.fix:
            rule["fix"] = self.fix

        # Simple YAML serialization
        lines = ["rules:", "  - id: " + self.id]
        lines.append("    pattern: " + self.pattern)
        lines.append("    message: " + self.message)
        lines.append("    severity: " + self.severity.value.upper())
        lines.append("    languages: [" + ", ".join(self.languages) + "]")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "pattern": self.pattern,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "languages": self.languages,
            "fix": self.fix,
            "metadata": self.metadata
        }


class ASTAnalyzer:
    """
    AST-based code analyzer.

    Based on Semgrep's semantic analysis capabilities
    for understanding code structure.
    """

    def __init__(self) -> None:
        self._patterns: Dict[str, List[SemgrepRule]] = {}
        self._cache: Dict[str, ASTNode] = {}

    def register_rule(self, rule: SemgrepRule) -> None:
        """Register a rule for analysis."""
        for lang in rule.languages:
            if lang not in self._patterns:
                self._patterns[lang] = []
            self._patterns[lang].append(rule)

    def parse_code(self, code: str, language: str = "python") -> ASTNode:
        """
        Parse code into an AST.

        This is a simplified pattern - actual implementation
        would use tree-sitter or similar.
        """
        # Simplified: create a root node representing the code
        root = ASTNode(
            node_type="module",
            metadata={"language": language, "source_length": len(code)}
        )

        # Parse functions (simplified regex-based)
        func_pattern = r"def\s+(\w+)\s*\([^)]*\):"
        for match in re.finditer(func_pattern, code):
            func_node = ASTNode(
                node_type="function_definition",
                value=match.group(1),
                line=code[:match.start()].count("\n") + 1
            )
            root.children.append(func_node)

        # Parse class definitions
        class_pattern = r"class\s+(\w+)\s*[:\(]"
        for match in re.finditer(class_pattern, code):
            class_node = ASTNode(
                node_type="class_definition",
                value=match.group(1),
                line=code[:match.start()].count("\n") + 1
            )
            root.children.append(class_node)

        return root

    def analyze(
        self,
        code: str,
        language: str = "python",
        rules: Optional[List[SemgrepRule]] = None
    ) -> List[SecurityFinding]:
        """
        Analyze code for security issues.

        Returns list of findings.
        """
        findings: List[SecurityFinding] = []

        # Use provided rules or registered ones
        active_rules = rules or self._patterns.get(language, [])

        for rule in active_rules:
            # Simple pattern matching (actual Semgrep uses semantic matching)
            for match in re.finditer(rule.pattern, code):
                line_num = code[:match.start()].count("\n") + 1
                finding = SecurityFinding(
                    id=f"finding_{len(findings) + 1}",
                    rule_id=rule.id,
                    file_path="<code>",
                    line_start=line_num,
                    line_end=line_num,
                    severity=rule.severity,
                    category=rule.category,
                    message=rule.message,
                    code_snippet=match.group(0)[:100],
                    fix_suggestion=rule.fix
                )
                findings.append(finding)

        return findings

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        # Based on Semgrep's language support
        return [
            "python", "javascript", "typescript", "java", "go",
            "ruby", "php", "c", "cpp", "csharp", "rust", "kotlin",
            "scala", "swift", "bash", "yaml", "json", "dockerfile"
        ]


@dataclass
class SecurityScanner:
    """
    Security scanner interface.

    Based on Semgrep MCP's security_check and semgrep_scan tools.
    """
    analyzer: ASTAnalyzer = field(default_factory=ASTAnalyzer)
    rules: List[SemgrepRule] = field(default_factory=list)
    severity_threshold: SecuritySeverity = SecuritySeverity.WARNING

    def __post_init__(self) -> None:
        """Register default security rules."""
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register common security rules."""
        default_rules = [
            SemgrepRule(
                id="hardcoded-password",
                pattern=r"password\s*=\s*['\"][^'\"]+['\"]",
                message="Hardcoded password detected",
                severity=SecuritySeverity.CRITICAL,
                category=RuleCategory.SECRETS
            ),
            SemgrepRule(
                id="sql-injection",
                pattern=r"execute\([^)]*%[^)]*\)",
                message="Potential SQL injection vulnerability",
                severity=SecuritySeverity.CRITICAL,
                category=RuleCategory.INJECTION
            ),
            SemgrepRule(
                id="eval-usage",
                pattern=r"\beval\s*\(",
                message="Use of eval() can lead to code injection",
                severity=SecuritySeverity.ERROR,
                category=RuleCategory.INJECTION
            ),
            SemgrepRule(
                id="weak-crypto",
                pattern=r"(MD5|SHA1)\s*\(",
                message="Weak cryptographic hash function",
                severity=SecuritySeverity.WARNING,
                category=RuleCategory.CRYPTO
            ),
            SemgrepRule(
                id="debug-enabled",
                pattern=r"DEBUG\s*=\s*True",
                message="Debug mode should not be enabled in production",
                severity=SecuritySeverity.WARNING,
                category=RuleCategory.CONFIGURATION
            )
        ]

        for rule in default_rules:
            self.rules.append(rule)
            self.analyzer.register_rule(rule)

    def add_rule(self, rule: SemgrepRule) -> None:
        """Add a custom rule."""
        self.rules.append(rule)
        self.analyzer.register_rule(rule)

    def scan_code(
        self,
        code: str,
        language: str = "python"
    ) -> List[SecurityFinding]:
        """Scan code for security issues."""
        return self.analyzer.analyze(code, language, self.rules)

    def scan_file(self, file_path: str) -> List[SecurityFinding]:
        """Scan a file for security issues."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()

            # Detect language from extension
            ext = Path(file_path).suffix.lower()
            lang_map = {
                ".py": "python",
                ".js": "javascript",
                ".ts": "typescript",
                ".java": "java",
                ".go": "go",
                ".rb": "ruby",
                ".php": "php",
                ".c": "c",
                ".cpp": "cpp",
                ".rs": "rust"
            }
            language = lang_map.get(ext, "python")

            findings = self.scan_code(code, language)

            # Update file paths in findings
            for finding in findings:
                finding.file_path = file_path

            return findings
        except Exception as e:
            return [SecurityFinding(
                id="scan-error",
                rule_id="scanner-error",
                file_path=file_path,
                line_start=0,
                line_end=0,
                severity=SecuritySeverity.ERROR,
                category=RuleCategory.CONFIGURATION,
                message=f"Failed to scan file: {e}"
            )]

    def filter_by_severity(
        self,
        findings: List[SecurityFinding],
        min_severity: Optional[SecuritySeverity] = None
    ) -> List[SecurityFinding]:
        """Filter findings by minimum severity."""
        threshold = min_severity or self.severity_threshold
        severity_order = {
            SecuritySeverity.INFO: 0,
            SecuritySeverity.WARNING: 1,
            SecuritySeverity.ERROR: 2,
            SecuritySeverity.CRITICAL: 3
        }
        min_level = severity_order[threshold]
        return [f for f in findings if severity_order[f.severity] >= min_level]


@dataclass
class SecurityAudit:
    """
    Comprehensive security audit result.

    Aggregates findings from multiple scans
    with summary statistics.
    """
    audit_id: str
    files_scanned: List[str] = field(default_factory=list)
    findings: List[SecurityFinding] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    passed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_findings(self, findings: List[SecurityFinding]) -> None:
        """Add findings to the audit."""
        self.findings.extend(findings)
        # Update pass status based on critical/error findings
        critical_count = sum(1 for f in self.findings if f.severity == SecuritySeverity.CRITICAL)
        error_count = sum(1 for f in self.findings if f.severity == SecuritySeverity.ERROR)
        self.passed = critical_count == 0 and error_count == 0

    def complete(self) -> None:
        """Mark the audit as complete."""
        self.completed_at = datetime.now(timezone.utc)

    def get_summary(self) -> Dict[str, Any]:
        """Get audit summary statistics."""
        severity_counts = {s.value: 0 for s in SecuritySeverity}
        category_counts = {c.value: 0 for c in RuleCategory}

        for finding in self.findings:
            severity_counts[finding.severity.value] += 1
            category_counts[finding.category.value] += 1

        return {
            "audit_id": self.audit_id,
            "files_scanned": len(self.files_scanned),
            "total_findings": len(self.findings),
            "passed": self.passed,
            "severity_breakdown": severity_counts,
            "category_breakdown": category_counts,
            "duration_seconds": (
                (self.completed_at - self.started_at).total_seconds()
                if self.completed_at else None
            )
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "audit_id": self.audit_id,
            "files_scanned": self.files_scanned,
            "findings": [f.to_dict() for f in self.findings],
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "passed": self.passed,
            "summary": self.get_summary(),
            "metadata": self.metadata
        }


# =============================================================================
# V10.10: LangGraph Graph Orchestration Patterns
# Based on: https://github.com/langchain-ai/langgraph
# =============================================================================

class NodeType(Enum):
    """
    Types of nodes in a LangGraph state graph.

    Based on LangGraph's graph-first orchestration primitives.
    """
    START = "start"           # Entry point node
    END = "end"               # Terminal node
    TASK = "task"             # Regular processing node
    CONDITIONAL = "conditional"  # Branching decision node
    TOOL = "tool"             # Tool execution node
    SUBGRAPH = "subgraph"     # Nested graph node


class EdgeType(Enum):
    """
    Types of edges connecting nodes.

    Supports cycles and conditional routing.
    """
    NORMAL = "normal"         # Direct connection
    CONDITIONAL = "conditional"  # Conditional routing
    LOOP = "loop"             # Cyclic edge back to earlier node


@dataclass
class GraphNode:
    """
    A node in a LangGraph state graph.

    Nodes are stateful units that process and transform state.
    """
    node_id: str
    name: str
    node_type: NodeType = NodeType.TASK
    handler: Optional[str] = None  # Function/method name
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "name": self.name,
            "node_type": self.node_type.value,
            "handler": self.handler,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphNode":
        """Create from dictionary."""
        return cls(
            node_id=data["node_id"],
            name=data["name"],
            node_type=NodeType(data.get("node_type", "task")),
            handler=data.get("handler"),
            metadata=data.get("metadata", {})
        )


@dataclass
class GraphEdge:
    """
    An edge connecting two nodes in the graph.

    Supports conditional routing based on state.
    """
    source_id: str
    target_id: str
    edge_type: EdgeType = EdgeType.NORMAL
    condition: Optional[str] = None  # Condition expression
    priority: int = 0  # For ordering multiple edges

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "condition": self.condition,
            "priority": self.priority
        }


@dataclass
class GraphCheckpoint:
    """
    Checkpoint for persisting graph execution state.

    Enables recovery and resumption of workflows.
    """
    checkpoint_id: str
    graph_id: str
    current_node: str
    state: Dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "graph_id": self.graph_id,
            "current_node": self.current_node,
            "state": self.state,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphCheckpoint":
        """Create from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            graph_id=data["graph_id"],
            current_node=data["current_node"],
            state=data.get("state", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
            metadata=data.get("metadata", {})
        )


@dataclass
class StateGraph:
    """
    A state graph for LangGraph-style orchestration.

    Supports:
    - Stateful nodes with handlers
    - Conditional routing
    - Cyclical workflows
    - Checkpointing for persistence
    """
    graph_id: str
    name: str
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: List[GraphEdge] = field(default_factory=list)
    entry_point: Optional[str] = None
    checkpoints: List[GraphCheckpoint] = field(default_factory=list)

    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.node_id] = node
        if node.node_type == NodeType.START:
            self.entry_point = node.node_id

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge connecting nodes."""
        self.edges.append(edge)

    def add_conditional_edge(
        self,
        source_id: str,
        targets: Dict[str, str],  # condition -> target_id
        default: Optional[str] = None
    ) -> None:
        """Add conditional edges from one node to multiple targets."""
        for condition, target_id in targets.items():
            self.edges.append(GraphEdge(
                source_id=source_id,
                target_id=target_id,
                edge_type=EdgeType.CONDITIONAL,
                condition=condition
            ))
        if default:
            self.edges.append(GraphEdge(
                source_id=source_id,
                target_id=default,
                edge_type=EdgeType.NORMAL,
                priority=-1  # Lower priority than conditions
            ))

    def get_successors(self, node_id: str) -> List[GraphEdge]:
        """Get all outgoing edges from a node."""
        return [e for e in self.edges if e.source_id == node_id]

    def get_predecessors(self, node_id: str) -> List[GraphEdge]:
        """Get all incoming edges to a node."""
        return [e for e in self.edges if e.target_id == node_id]

    def has_cycles(self) -> bool:
        """Check if the graph contains cycles."""
        return any(e.edge_type == EdgeType.LOOP for e in self.edges)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "graph_id": self.graph_id,
            "name": self.name,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "entry_point": self.entry_point,
            "has_cycles": self.has_cycles()
        }


class GraphExecutionStatus(Enum):
    """Status of graph execution."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class GraphRuntime:
    """
    Runtime environment for executing state graphs.

    Manages state, checkpointing, and node execution.
    """
    graph: StateGraph
    state: Dict[str, Any] = field(default_factory=dict)
    current_node: Optional[str] = None
    status: GraphExecutionStatus = GraphExecutionStatus.PENDING
    execution_history: List[str] = field(default_factory=list)
    max_iterations: int = 100  # Prevent infinite loops

    def start(self) -> None:
        """Start graph execution from entry point."""
        if self.graph.entry_point:
            self.current_node = self.graph.entry_point
            self.status = GraphExecutionStatus.RUNNING
            self.execution_history.append(self.current_node)

    def step(self) -> Optional[str]:
        """Execute one step and return next node."""
        if not self.current_node or self.status != GraphExecutionStatus.RUNNING:
            return None

        if len(self.execution_history) >= self.max_iterations:
            self.status = GraphExecutionStatus.FAILED
            return None

        node = self.graph.nodes.get(self.current_node)
        if not node:
            return None

        if node.node_type == NodeType.END:
            self.status = GraphExecutionStatus.COMPLETED
            return None

        # Get next node based on edges
        successors = self.graph.get_successors(self.current_node)
        if not successors:
            self.status = GraphExecutionStatus.COMPLETED
            return None

        # Sort by priority and pick first matching
        successors.sort(key=lambda e: e.priority, reverse=True)
        for edge in successors:
            if edge.edge_type == EdgeType.NORMAL or self._evaluate_condition(edge.condition):
                self.current_node = edge.target_id
                self.execution_history.append(self.current_node)
                return self.current_node

        return None

    def _evaluate_condition(self, condition: Optional[str]) -> bool:
        """Evaluate a condition against current state."""
        if not condition:
            return True
        # Simple key-based condition evaluation
        return self.state.get(condition, False)

    def checkpoint(self) -> GraphCheckpoint:
        """Create a checkpoint of current state."""
        cp = GraphCheckpoint(
            checkpoint_id=f"cp_{self.graph.graph_id}_{len(self.graph.checkpoints)}",
            graph_id=self.graph.graph_id,
            current_node=self.current_node or "",
            state=self.state.copy()
        )
        self.graph.checkpoints.append(cp)
        return cp

    def restore(self, checkpoint: GraphCheckpoint) -> None:
        """Restore state from a checkpoint."""
        self.current_node = checkpoint.current_node
        self.state = checkpoint.state.copy()
        self.status = GraphExecutionStatus.RUNNING


@dataclass
class CyclicalWorkflow:
    """
    Workflow that supports cycles and re-entry.

    Enables iterative refinement loops common in
    multi-agent orchestration.
    """
    workflow_id: str
    graph: StateGraph
    loop_limit: int = 10
    current_iteration: int = 0
    loop_state: Dict[str, Any] = field(default_factory=dict)

    def can_continue_loop(self) -> bool:
        """Check if loop can continue."""
        return self.current_iteration < self.loop_limit

    def iterate(self) -> bool:
        """Perform one iteration of the loop."""
        if not self.can_continue_loop():
            return False
        self.current_iteration += 1
        return True

    def reset_loop(self) -> None:
        """Reset loop counter."""
        self.current_iteration = 0
        self.loop_state.clear()

    def set_loop_condition(self, key: str, value: Any) -> None:
        """Set a condition for loop continuation."""
        self.loop_state[key] = value

    def check_exit_condition(self, key: str) -> bool:
        """Check if exit condition is met."""
        return self.loop_state.get(key, False)


# =============================================================================
# V10.10: DSPy Programmatic LM Patterns
# Based on: https://github.com/stanfordnlp/dspy
# =============================================================================

@dataclass
class SignatureField:
    """
    A field in a DSPy signature.

    Fields can be inputs or outputs with descriptions.
    """
    name: str
    description: str = ""
    prefix: str = ""  # Prompt prefix for the field
    is_input: bool = True
    is_optional: bool = False
    default: Optional[Any] = None


@dataclass
class DSPySignature:
    """
    A declarative signature for DSPy modules.

    Signatures define the input/output contract of a module.
    Example: "question -> answer"
    """
    name: str
    inputs: List[SignatureField] = field(default_factory=list)
    outputs: List[SignatureField] = field(default_factory=list)
    instructions: str = ""

    @classmethod
    def from_string(cls, signature_str: str, name: str = "Signature") -> "DSPySignature":
        """
        Parse a signature string like "question, context -> answer".
        """
        parts = signature_str.split("->")
        if len(parts) != 2:
            raise ValueError(f"Invalid signature format: {signature_str}")

        inputs = [SignatureField(name=f.strip(), is_input=True)
                  for f in parts[0].split(",") if f.strip()]
        outputs = [SignatureField(name=f.strip(), is_input=False)
                   for f in parts[1].split(",") if f.strip()]

        return cls(name=name, inputs=inputs, outputs=outputs)

    def to_prompt(self) -> str:
        """Generate a prompt template from the signature."""
        lines = []
        if self.instructions:
            lines.append(f"Instructions: {self.instructions}")

        for inp in self.inputs:
            prefix = inp.prefix or f"{inp.name.title()}:"
            lines.append(f"{prefix} {{{inp.name}}}")

        for out in self.outputs:
            prefix = out.prefix or f"{out.name.title()}:"
            lines.append(f"{prefix}")

        return "\n".join(lines)


@dataclass
class DSPyExample:
    """
    An example for few-shot prompting.

    Contains input/output pairs for demonstrations.
    """
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DSPyModule:
    """
    A composable DSPy module.

    Modules are the building blocks of DSPy programs,
    each with a signature and optional demonstrations.
    """
    name: str
    signature: DSPySignature
    demonstrations: List[DSPyExample] = field(default_factory=list)
    compiled: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def forward(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute the module with given inputs.

        Returns predicted outputs based on signature.
        """
        # Validate inputs
        for inp in self.signature.inputs:
            if not inp.is_optional and inp.name not in kwargs:
                raise ValueError(f"Missing required input: {inp.name}")

        # Build prompt from signature and inputs
        prompt = self.signature.to_prompt()
        for key, value in kwargs.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))

        # Return structured output placeholder
        return {
            "prompt": prompt,
            "inputs": kwargs,
            "demonstrations": len(self.demonstrations),
            "ready": True
        }

    def add_demonstration(self, example: DSPyExample) -> None:
        """Add a demonstration example."""
        self.demonstrations.append(example)

    def compile(self) -> None:
        """Mark module as compiled (optimized)."""
        self.compiled = True


@dataclass
class DSPyPredict:
    """
    The basic predictor module in DSPy.

    Wraps a signature for prediction tasks.
    """
    signature: DSPySignature
    max_tokens: int = 1000
    temperature: float = 0.0

    def __call__(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute prediction."""
        module = DSPyModule(
            name=f"Predict_{self.signature.name}",
            signature=self.signature
        )
        result = module.forward(**kwargs)
        result["max_tokens"] = self.max_tokens
        result["temperature"] = self.temperature
        return result


class OptimizationStrategy(Enum):
    """Strategies for DSPy prompt optimization."""
    BOOTSTRAP = "bootstrap"  # Bootstrap few-shot examples
    MIPRO = "mipro"          # MIPRO optimization
    RANDOM = "random"        # Random search
    BAYESIAN = "bayesian"    # Bayesian optimization


@dataclass
class OptimizationResult:
    """
    Result of DSPy optimization.

    Contains optimized prompts and metrics.
    """
    strategy: OptimizationStrategy
    iterations: int
    best_score: float
    optimized_demos: List[DSPyExample]
    metrics: Dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0


@dataclass
class DSPyOptimizer:
    """
    Optimizer for DSPy modules.

    Implements prompt optimization algorithms.
    """
    strategy: OptimizationStrategy = OptimizationStrategy.BOOTSTRAP
    num_candidates: int = 10
    max_iterations: int = 100
    metric_fn: Optional[str] = None  # Name of metric function

    def optimize(
        self,
        module: DSPyModule,
        train_examples: List[DSPyExample]
    ) -> OptimizationResult:
        """
        Optimize a module using training examples.

        Returns optimized demonstrations and metrics.
        """
        start_time = datetime.now(timezone.utc)

        # Simple bootstrap: select best demonstrations
        selected_demos = train_examples[:self.num_candidates]

        duration = (datetime.now(timezone.utc) - start_time).total_seconds()

        return OptimizationResult(
            strategy=self.strategy,
            iterations=min(len(train_examples), self.max_iterations),
            best_score=0.0,  # Placeholder
            optimized_demos=selected_demos,
            duration_seconds=duration
        )


@dataclass
class DSPyTool:
    """
    A tool in DSPy format.

    Enables MCP tool conversion to DSPy tools.
    """
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema for parameters
    async_enabled: bool = True
    source: str = "mcp"  # Source of the tool (mcp, native, etc.)

    @classmethod
    def from_mcp_tool(cls, mcp_tool: Dict[str, Any]) -> "DSPyTool":
        """
        Convert an MCP tool definition to DSPy format.

        Preserves tool name, description, and parameter schemas.
        """
        return cls(
            name=mcp_tool.get("name", "unknown"),
            description=mcp_tool.get("description", ""),
            parameters=mcp_tool.get("inputSchema", {}),
            async_enabled=True,
            source="mcp"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "async_enabled": self.async_enabled,
            "source": self.source
        }


# =============================================================================
# V10.10: MCP Multi-Server Client Patterns
# Based on: https://github.com/langchain-ai/langchain-mcp-adapters
# =============================================================================

class MCPTransportType(Enum):
    """Transport types for MCP connections."""
    STDIO = "stdio"           # Local subprocess
    HTTP = "http"             # HTTP/REST
    STREAMABLE_HTTP = "streamable_http"  # Streaming HTTP
    SSE = "sse"               # Server-Sent Events
    WEBSOCKET = "websocket"   # WebSocket


@dataclass
class MCPServerConfig:
    """
    Configuration for an MCP server.

    Supports multiple transport types.
    """
    name: str
    transport: MCPTransportType
    # For stdio transport
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    # For HTTP/SSE/WebSocket
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    # Common settings
    timeout_seconds: float = 30.0
    retry_count: int = 3
    stateless: bool = True  # Each call creates fresh session

    def validate(self) -> bool:
        """Validate configuration is complete."""
        if self.transport == MCPTransportType.STDIO:
            return self.command is not None
        else:
            return self.url is not None


@dataclass
class MCPToolArtifact:
    """
    Artifact returned by MCP tool execution.

    Contains machine-parseable data alongside human-readable content.
    """
    content: str
    artifact: Optional[Any] = None
    content_type: str = "text/plain"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPInterceptor:
    """
    Interceptor for MCP tool calls.

    Provides middleware-like control for:
    - Modifying requests before execution
    - Injecting context (user IDs, API keys)
    - Implementing retry logic
    - Access control
    """
    name: str
    priority: int = 0  # Higher priority executes first
    enabled: bool = True

    def before_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Called before tool execution.

        Can modify args or inject context.
        """
        return args

    def after_call(
        self,
        tool_name: str,
        result: MCPToolArtifact,
        context: Dict[str, Any]
    ) -> MCPToolArtifact:
        """
        Called after tool execution.

        Can modify or filter results.
        """
        return result

    def on_error(
        self,
        tool_name: str,
        error: Exception,
        context: Dict[str, Any]
    ) -> Optional[MCPToolArtifact]:
        """
        Called on tool error.

        Can provide fallback result or re-raise.
        """
        return None


@dataclass
class InterceptorChain:
    """
    Chain of interceptors in onion pattern.

    First interceptor is outermost layer.
    """
    interceptors: List[MCPInterceptor] = field(default_factory=list)

    def add(self, interceptor: MCPInterceptor) -> None:
        """Add interceptor to chain."""
        self.interceptors.append(interceptor)
        self.interceptors.sort(key=lambda i: i.priority, reverse=True)

    def before(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute all before hooks."""
        current_args = args
        for interceptor in self.interceptors:
            if interceptor.enabled:
                current_args = interceptor.before_call(tool_name, current_args, context)
        return current_args

    def after(
        self,
        tool_name: str,
        result: MCPToolArtifact,
        context: Dict[str, Any]
    ) -> MCPToolArtifact:
        """Execute all after hooks in reverse order."""
        current_result = result
        for interceptor in reversed(self.interceptors):
            if interceptor.enabled:
                current_result = interceptor.after_call(tool_name, current_result, context)
        return current_result


@dataclass
class StatefulSession:
    """
    A stateful MCP session.

    Maintains connection state across multiple tool calls.
    """
    session_id: str
    server_name: str
    connected: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    state: Dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)

    def is_stale(self, max_idle_seconds: float = 300.0) -> bool:
        """Check if session has been idle too long."""
        idle_time = (datetime.now(timezone.utc) - self.last_activity).total_seconds()
        return idle_time > max_idle_seconds


@dataclass
class SessionPool:
    """
    Pool of MCP sessions for connection reuse.

    Manages stateful sessions with cleanup.
    """
    max_sessions: int = 10
    sessions: Dict[str, StatefulSession] = field(default_factory=dict)
    max_idle_seconds: float = 300.0

    def get(self, server_name: str) -> Optional[StatefulSession]:
        """Get an existing session for a server."""
        session = self.sessions.get(server_name)
        if session and not session.is_stale(self.max_idle_seconds):
            session.touch()
            return session
        return None

    def create(self, server_name: str) -> StatefulSession:
        """Create a new session."""
        self._cleanup_stale()

        if len(self.sessions) >= self.max_sessions:
            # Remove oldest session
            oldest = min(self.sessions.values(), key=lambda s: s.last_activity)
            del self.sessions[oldest.server_name]

        session = StatefulSession(
            session_id=f"sess_{server_name}_{len(self.sessions)}",
            server_name=server_name,
            connected=True
        )
        self.sessions[server_name] = session
        return session

    def release(self, session: StatefulSession) -> None:
        """Release a session back to pool."""
        session.connected = False

    def _cleanup_stale(self) -> None:
        """Remove stale sessions."""
        stale = [name for name, s in self.sessions.items()
                 if s.is_stale(self.max_idle_seconds)]
        for name in stale:
            del self.sessions[name]


@dataclass
class MultiServerClient:
    """
    Client for managing multiple MCP servers.

    Based on langchain-mcp-adapters MultiServerMCPClient.
    """
    servers: Dict[str, MCPServerConfig] = field(default_factory=dict)
    session_pool: SessionPool = field(default_factory=SessionPool)
    interceptor_chain: InterceptorChain = field(default_factory=InterceptorChain)
    tools_cache: Dict[str, List[DSPyTool]] = field(default_factory=dict)

    def add_server(self, config: MCPServerConfig) -> None:
        """Add a server configuration."""
        if config.validate():
            self.servers[config.name] = config

    def get_tools(self, server_name: Optional[str] = None) -> List[DSPyTool]:
        """Get tools from one or all servers."""
        if server_name:
            return self.tools_cache.get(server_name, [])
        # Return all tools
        all_tools: List[DSPyTool] = []
        for tools in self.tools_cache.values():
            all_tools.extend(tools)
        return all_tools

    def register_tools(self, server_name: str, tools: List[Dict[str, Any]]) -> None:
        """Register tools from a server."""
        self.tools_cache[server_name] = [
            DSPyTool.from_mcp_tool(t) for t in tools
        ]

    def get_session(self, server_name: str) -> StatefulSession:
        """Get or create a session for a server."""
        existing = self.session_pool.get(server_name)
        if existing:
            return existing
        return self.session_pool.create(server_name)


@dataclass
class ResourceLoader:
    """
    Loader for MCP resources and prompts.

    Supports loading from multiple servers.
    """
    client: MultiServerClient

    def get_resources(
        self,
        server_name: str,
        uris: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Load resources from a server."""
        # Placeholder for resource loading
        return []

    def get_prompt(
        self,
        server_name: str,
        prompt_name: str,
        arguments: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Load a prompt from a server."""
        # Placeholder for prompt loading
        return []

    def list_prompts(self, server_name: str) -> List[str]:
        """List available prompts on a server."""
        return []


# =============================================================================
# V10.11: FastMCP Server Patterns
# Based on: https://github.com/modelcontextprotocol/python-sdk (FastMCP module)
# =============================================================================


class FastMCPServer:
    """
    FastMCP server implementation with decorator-based tool registration.

    Provides a simplified API for creating MCP servers with automatic
    tool, resource, and prompt registration.

    Example:
        server = FastMCPServer("my-server")

        @server.tool()
        def my_tool(x: int) -> str:
            return f"Result: {x}"
    """

    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        description: Optional[str] = None,
        lifespan: Optional[Callable] = None
    ):
        self.name = name
        self.version = version
        self.description = description
        self.lifespan = lifespan
        self._tools: Dict[str, "FastMCPTool"] = {}
        self._resources: Dict[str, "FastMCPResource"] = {}
        self._prompts: Dict[str, Callable] = {}

    def tool(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Callable:
        """Decorator to register a tool."""
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            self._tools[tool_name] = FastMCPTool(
                name=tool_name,
                description=description or func.__doc__ or "",
                handler=func
            )
            return func
        return decorator

    def resource(
        self,
        uri: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        mime_type: str = "text/plain"
    ) -> Callable:
        """Decorator to register a resource."""
        def decorator(func: Callable) -> Callable:
            resource_name = name or func.__name__
            self._resources[uri] = FastMCPResource(
                uri=uri,
                name=resource_name,
                description=description or func.__doc__ or "",
                mime_type=mime_type,
                handler=func
            )
            return func
        return decorator

    def prompt(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Callable:
        """Decorator to register a prompt."""
        def decorator(func: Callable) -> Callable:
            prompt_name = name or func.__name__
            self._prompts[prompt_name] = func
            return func
        return decorator

    def list_tools(self) -> List[str]:
        """List registered tool names."""
        return list(self._tools.keys())

    def list_resources(self) -> List[str]:
        """List registered resource URIs."""
        return list(self._resources.keys())

    def list_prompts(self) -> List[str]:
        """List registered prompt names."""
        return list(self._prompts.keys())

    async def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
        context: Optional["MCPContext"] = None
    ) -> Any:
        """Call a registered tool."""
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        tool = self._tools[name]
        return await tool.execute(arguments, context)


@dataclass
class FastMCPTool:
    """
    FastMCP tool definition with handler.

    Supports async and sync handlers with automatic argument validation.
    """
    name: str
    description: str
    handler: Callable
    annotations: Optional[ToolAnnotations] = None

    async def execute(
        self,
        arguments: Dict[str, Any],
        context: Optional["MCPContext"] = None
    ) -> Any:
        """Execute the tool handler."""
        import inspect
        if inspect.iscoroutinefunction(self.handler):
            return await self.handler(**arguments)
        return self.handler(**arguments)

    def to_schema(self) -> Dict[str, Any]:
        """Convert to MCP tool schema."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }


@dataclass
class FastMCPResource:
    """
    FastMCP resource definition.

    Resources provide readable data to clients (files, API data, etc.).
    """
    uri: str
    name: str
    description: str
    handler: Callable
    mime_type: str = "text/plain"

    async def read(self, context: Optional["MCPContext"] = None) -> str:
        """Read the resource content."""
        import asyncio
        if inspect.iscoroutinefunction(self.handler):
            return await self.handler()
        return self.handler()

    def to_schema(self) -> Dict[str, Any]:
        """Convert to MCP resource schema."""
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type
        }


@dataclass
class LifespanContext:
    """
    Lifespan context for FastMCP servers.

    Provides startup/shutdown lifecycle management with dependency injection.
    """
    server: FastMCPServer
    state: Dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        """Store a value in lifespan state."""
        self.state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from lifespan state."""
        return self.state.get(key, default)

    async def __aenter__(self) -> "LifespanContext":
        """Enter lifespan context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit lifespan context."""
        self.state.clear()


@dataclass
class MCPContext:
    """
    Context available to FastMCP tools during execution.

    Provides access to progress reporting, logging, and lifespan state.
    """
    request_id: str
    server: FastMCPServer
    lifespan: Optional[LifespanContext] = None

    async def report_progress(
        self,
        progress: float,
        total: Optional[float] = None,
        message: Optional[str] = None
    ) -> None:
        """Report progress to the client."""
        # Would send progress notification
        pass

    def log(
        self,
        level: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a message."""
        pass

    def get_lifespan_state(self, key: str) -> Any:
        """Get value from lifespan state."""
        if self.lifespan:
            return self.lifespan.get(key)
        return None


# =============================================================================
# V10.11: Grafana MCP Observability Patterns
# Based on: https://github.com/grafana/mcp-grafana
# =============================================================================


class DashboardPanelType(str, Enum):
    """Types of Grafana dashboard panels."""
    GRAPH = "graph"
    STAT = "stat"
    TABLE = "table"
    GAUGE = "gauge"
    HEATMAP = "heatmap"
    TIMESERIES = "timeseries"
    LOGS = "logs"
    BARCHART = "barchart"
    PIE = "pie"
    TEXT = "text"


@dataclass
class GrafanaPanel:
    """
    Grafana dashboard panel configuration.

    Represents a single visualization in a dashboard.
    """
    id: int
    title: str
    panel_type: DashboardPanelType
    targets: List[Dict[str, Any]] = field(default_factory=list)
    grid_pos: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0, "w": 12, "h": 8})
    options: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        """Convert to Grafana panel JSON."""
        return {
            "id": self.id,
            "title": self.title,
            "type": self.panel_type.value,
            "targets": self.targets,
            "gridPos": self.grid_pos,
            "options": self.options
        }


@dataclass
class GrafanaDashboard:
    """
    Grafana dashboard definition.

    Contains multiple panels and configuration.
    """
    uid: str
    title: str
    panels: List[GrafanaPanel] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    refresh: str = "5s"
    time_from: str = "now-1h"
    time_to: str = "now"
    version: int = 1

    def add_panel(self, panel: GrafanaPanel) -> None:
        """Add a panel to the dashboard."""
        self.panels.append(panel)

    def to_json(self) -> Dict[str, Any]:
        """Convert to Grafana dashboard JSON."""
        return {
            "uid": self.uid,
            "title": self.title,
            "panels": [p.to_json() for p in self.panels],
            "tags": self.tags,
            "refresh": self.refresh,
            "time": {"from": self.time_from, "to": self.time_to},
            "version": self.version
        }


@dataclass
class PrometheusQuery:
    """
    Prometheus query configuration.

    Used for metrics querying in Grafana.
    """
    expr: str
    legend_format: str = "{{label}}"
    ref_id: str = "A"
    instant: bool = False
    range_query: bool = True
    interval: str = ""

    def to_target(self) -> Dict[str, Any]:
        """Convert to Grafana target format."""
        return {
            "datasource": {"type": "prometheus"},
            "expr": self.expr,
            "legendFormat": self.legend_format,
            "refId": self.ref_id,
            "instant": self.instant,
            "range": self.range_query,
            "interval": self.interval
        }


@dataclass
class LokiQuery:
    """
    Loki log query configuration.

    Used for log querying in Grafana.
    """
    expr: str
    ref_id: str = "A"
    max_lines: int = 1000
    query_type: str = "range"  # "range" or "instant"

    def to_target(self) -> Dict[str, Any]:
        """Convert to Grafana target format."""
        return {
            "datasource": {"type": "loki"},
            "expr": self.expr,
            "refId": self.ref_id,
            "maxLines": self.max_lines,
            "queryType": self.query_type
        }


class AlertState(str, Enum):
    """Grafana alert states."""
    OK = "ok"
    PENDING = "pending"
    ALERTING = "alerting"
    NO_DATA = "no_data"
    ERROR = "error"


@dataclass
class GrafanaAlert:
    """
    Grafana alerting rule.

    Defines conditions for triggering alerts.
    """
    uid: str
    title: str
    condition: str
    queries: List[Dict[str, Any]] = field(default_factory=list)
    state: AlertState = AlertState.OK
    folder_uid: str = ""
    rule_group: str = "default"
    for_duration: str = "5m"
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        """Convert to Grafana alert rule JSON."""
        return {
            "uid": self.uid,
            "title": self.title,
            "condition": self.condition,
            "data": self.queries,
            "state": self.state.value,
            "folderUID": self.folder_uid,
            "ruleGroup": self.rule_group,
            "for": self.for_duration,
            "labels": self.labels,
            "annotations": self.annotations
        }


class IncidentSeverity(str, Enum):
    """Grafana incident severity levels."""
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    WARNING = "warning"
    INFO = "info"


class IncidentStatus(str, Enum):
    """Grafana incident status."""
    OPEN = "open"
    ACTIVE = "active"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass
class IncidentActivity:
    """
    Activity entry in a Grafana incident.

    Represents timeline events during incident management.
    """
    id: str
    incident_id: str
    activity_type: str  # "comment", "status_change", "assignment", etc.
    body: str
    created_at: str
    user_id: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON format."""
        return {
            "id": self.id,
            "incidentId": self.incident_id,
            "activityType": self.activity_type,
            "body": self.body,
            "createdAt": self.created_at,
            "userId": self.user_id
        }


@dataclass
class GrafanaIncident:
    """
    Grafana incident management.

    Represents an active incident with timeline.
    """
    id: str
    title: str
    severity: IncidentSeverity
    status: IncidentStatus = IncidentStatus.OPEN
    description: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    activities: List[IncidentActivity] = field(default_factory=list)
    created_at: Optional[str] = None
    resolved_at: Optional[str] = None

    def add_activity(self, activity: IncidentActivity) -> None:
        """Add an activity to the incident."""
        self.activities.append(activity)

    def resolve(self, resolution_note: str = "") -> None:
        """Mark incident as resolved."""
        self.status = IncidentStatus.RESOLVED
        from datetime import datetime, timezone
        self.resolved_at = datetime.now(timezone.utc).isoformat()

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON format."""
        return {
            "id": self.id,
            "title": self.title,
            "severity": self.severity.value,
            "status": self.status.value,
            "description": self.description,
            "labels": self.labels,
            "activities": [a.to_json() for a in self.activities],
            "createdAt": self.created_at,
            "resolvedAt": self.resolved_at
        }


# =============================================================================
# V10.11: Qdrant MCP Vector Patterns
# Based on: https://github.com/qdrant/mcp-server-qdrant
# =============================================================================


class DistanceMetric(str, Enum):
    """Qdrant distance metrics for vector comparison."""
    COSINE = "Cosine"
    EUCLID = "Euclid"
    DOT = "Dot"
    MANHATTAN = "Manhattan"


class IndexType(str, Enum):
    """Qdrant vector index types."""
    HNSW = "hnsw"
    FLAT = "flat"


@dataclass
class VectorConfig:
    """
    Qdrant vector configuration.

    Defines vector dimensions and distance metric.
    """
    size: int
    distance: DistanceMetric = DistanceMetric.COSINE
    on_disk: bool = False

    def to_json(self) -> Dict[str, Any]:
        """Convert to Qdrant format."""
        return {
            "size": self.size,
            "distance": self.distance.value,
            "on_disk": self.on_disk
        }


@dataclass
class QdrantCollection:
    """
    Qdrant collection configuration.

    A collection is a named set of points (vectors with payloads).
    """
    name: str
    vectors_config: VectorConfig
    shard_number: int = 1
    replication_factor: int = 1
    write_consistency_factor: int = 1
    on_disk_payload: bool = True

    def to_create_params(self) -> Dict[str, Any]:
        """Convert to collection creation parameters."""
        return {
            "vectors": self.vectors_config.to_json(),
            "shard_number": self.shard_number,
            "replication_factor": self.replication_factor,
            "write_consistency_factor": self.write_consistency_factor,
            "on_disk_payload": self.on_disk_payload
        }


@dataclass
class QdrantPoint:
    """
    Qdrant point (vector with payload).

    Represents a single vector in a collection.
    """
    id: Union[str, int]
    vector: List[float]
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        """Convert to Qdrant point format."""
        return {
            "id": self.id,
            "vector": self.vector,
            "payload": self.payload
        }


@dataclass
class VectorSearchResult:
    """
    Result from a Qdrant vector search.

    Contains matching points with scores.
    """
    id: Union[str, int]
    score: float
    payload: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[List[float]] = None

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "VectorSearchResult":
        """Create from Qdrant response."""
        return cls(
            id=data["id"],
            score=data["score"],
            payload=data.get("payload", {}),
            vector=data.get("vector")
        )


class EmbeddingProvider(str, Enum):
    """Embedding model providers."""
    OPENAI = "openai"
    FASTEMBED = "fastembed"
    OLLAMA = "ollama"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


@dataclass
class EmbeddingModel:
    """
    Embedding model configuration for Qdrant.

    Defines how text is converted to vectors.
    """
    provider: EmbeddingProvider
    model_name: str
    dimensions: int
    batch_size: int = 32

    def get_embedding_function(self) -> Callable[[str], List[float]]:
        """Get embedding function (placeholder)."""
        def embed(text: str) -> List[float]:
            # Placeholder - would use actual embedding model
            return [0.0] * self.dimensions
        return embed


@dataclass
class SemanticSearch:
    """
    Semantic search configuration for RAG patterns.

    Combines embedding model with Qdrant collection.
    """
    collection: QdrantCollection
    embedding_model: EmbeddingModel
    top_k: int = 10
    score_threshold: Optional[float] = None

    def build_query(
        self,
        query_vector: List[float],
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build Qdrant search query."""
        query = {
            "vector": query_vector,
            "limit": self.top_k,
            "with_payload": True,
            "with_vector": False
        }
        if self.score_threshold is not None:
            query["score_threshold"] = self.score_threshold
        if filter_conditions:
            query["filter"] = filter_conditions
        return query

    def search(
        self,
        query_text: str,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """
        Perform semantic search (placeholder).

        Would embed query and search Qdrant.
        """
        embed_fn = self.embedding_model.get_embedding_function()
        query_vector = embed_fn(query_text)
        # Would call Qdrant with self.build_query(query_vector, filter_conditions)
        return []


# =============================================================================
# V10.11: Postgres MCP Pro Patterns
# Based on: https://github.com/crystaldba/postgres-mcp
# =============================================================================


class DatabaseAccessMode(str, Enum):
    """Database access permission levels."""
    UNRESTRICTED = "unrestricted"  # Full DDL/DML access
    RESTRICTED = "restricted"      # SELECT + safe modifications
    READ_ONLY = "read_only"        # SELECT only


class QueryType(str, Enum):
    """Types of SQL queries."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    DDL = "ddl"
    UNKNOWN = "unknown"


@dataclass
class QueryExecution:
    """
    SQL query execution result.

    Contains query results and execution metadata.
    """
    query: str
    query_type: QueryType
    rows: List[Dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    columns: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if query executed successfully."""
        return self.error is None


class PgIndexType(str, Enum):
    """PostgreSQL index types."""
    BTREE = "btree"
    HASH = "hash"
    GIN = "gin"
    GIST = "gist"
    BRIN = "brin"


@dataclass
class IndexRecommendation:
    """
    Index recommendation from Postgres MCP Pro.

    Based on Anytime Algorithm analysis.
    """
    table_name: str
    column_names: List[str]
    index_type: PgIndexType
    estimated_benefit: float  # Improvement factor
    create_statement: str
    reason: str
    priority: int = 5  # 1-10, higher = more important

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON format."""
        return {
            "table": self.table_name,
            "columns": self.column_names,
            "indexType": self.index_type.value,
            "estimatedBenefit": self.estimated_benefit,
            "createStatement": self.create_statement,
            "reason": self.reason,
            "priority": self.priority
        }


class HealthStatus(str, Enum):
    """Database health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class DatabaseHealth:
    """
    PostgreSQL database health status.

    Includes key metrics and status indicators.
    """
    status: HealthStatus
    connection_count: int = 0
    max_connections: int = 100
    database_size: str = "0 MB"
    cache_hit_ratio: float = 0.0
    active_queries: int = 0
    blocked_queries: int = 0
    replication_lag_seconds: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def connection_usage_percent(self) -> float:
        """Calculate connection pool usage."""
        if self.max_connections == 0:
            return 0.0
        return (self.connection_count / self.max_connections) * 100

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON format."""
        return {
            "status": self.status.value,
            "connectionCount": self.connection_count,
            "maxConnections": self.max_connections,
            "connectionUsagePercent": self.connection_usage_percent,
            "databaseSize": self.database_size,
            "cacheHitRatio": self.cache_hit_ratio,
            "activeQueries": self.active_queries,
            "blockedQueries": self.blocked_queries,
            "replicationLagSeconds": self.replication_lag_seconds,
            "warnings": self.warnings
        }


class ExplainFormat(str, Enum):
    """EXPLAIN output formats."""
    TEXT = "text"
    JSON = "json"
    XML = "xml"
    YAML = "yaml"


@dataclass
class ExplainAnalysis:
    """
    PostgreSQL EXPLAIN ANALYZE result.

    Provides query plan analysis for optimization.
    """
    query: str
    plan: Dict[str, Any]
    total_cost: float = 0.0
    actual_time_ms: float = 0.0
    rows_estimated: int = 0
    rows_actual: int = 0
    shared_hit_blocks: int = 0
    shared_read_blocks: int = 0

    @property
    def estimation_accuracy(self) -> float:
        """Calculate row estimation accuracy."""
        if self.rows_estimated == 0:
            return 0.0 if self.rows_actual > 0 else 1.0
        return min(self.rows_estimated, self.rows_actual) / max(self.rows_estimated, self.rows_actual)

    @property
    def cache_hit_ratio(self) -> float:
        """Calculate buffer cache hit ratio for this query."""
        total_blocks = self.shared_hit_blocks + self.shared_read_blocks
        if total_blocks == 0:
            return 1.0
        return self.shared_hit_blocks / total_blocks

    def get_slow_nodes(self, threshold_ms: float = 100.0) -> List[Dict[str, Any]]:
        """Find slow nodes in the query plan."""
        slow_nodes = []

        def traverse(node: Dict[str, Any]) -> None:
            actual_time = node.get("Actual Total Time", 0)
            if actual_time > threshold_ms:
                slow_nodes.append({
                    "node_type": node.get("Node Type"),
                    "actual_time_ms": actual_time,
                    "rows": node.get("Actual Rows", 0)
                })
            for child in node.get("Plans", []):
                traverse(child)

        traverse(self.plan)
        return slow_nodes


# =============================================================================
# V10.11: Enhanced Sequential Thinking Patterns
# Based on: https://github.com/modelcontextprotocol/servers
# =============================================================================


class ThoughtType(str, Enum):
    """Types of sequential thoughts."""
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    ANALYSIS = "analysis"
    CONCLUSION = "conclusion"
    QUESTION = "question"
    REVISION = "revision"
    BRANCH = "branch"


@dataclass
class ThoughtBranch:
    """
    A branch in sequential thinking.

    Represents an alternative line of reasoning.
    """
    id: str
    parent_thought_id: str
    name: str
    created_at: str
    merged: bool = False
    merged_into: Optional[str] = None


@dataclass
class EnhancedThought:
    """
    Enhanced sequential thought with branching and revision.

    Extends ThoughtData with branching and revision tracking.
    """
    id: str
    content: str
    thought_type: ThoughtType
    thought_number: int
    total_thoughts: int
    next_thought_needed: bool = True
    is_revision: bool = False
    revises_thought_id: Optional[str] = None
    branch_id: Optional[str] = None
    branch_from_thought_id: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON format."""
        return {
            "id": self.id,
            "content": self.content,
            "thoughtType": self.thought_type.value,
            "thoughtNumber": self.thought_number,
            "totalThoughts": self.total_thoughts,
            "nextThoughtNeeded": self.next_thought_needed,
            "isRevision": self.is_revision,
            "revisesThoughtId": self.revises_thought_id,
            "branchId": self.branch_id,
            "branchFromThoughtId": self.branch_from_thought_id,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


@dataclass
class BranchingThinkingSession:
    """
    Thinking session with branching and revision support.

    Extends beyond simple sequential thinking to support
    branching thought paths and revision tracking.
    """
    session_id: str
    thoughts: List[EnhancedThought] = field(default_factory=list)
    branches: List[ThoughtBranch] = field(default_factory=list)
    current_branch_id: Optional[str] = None
    created_at: Optional[str] = None

    def add_thought(self, thought: EnhancedThought) -> None:
        """Add a thought to the session."""
        thought.branch_id = self.current_branch_id
        self.thoughts.append(thought)

    def create_branch(
        self,
        name: str,
        from_thought_id: str
    ) -> ThoughtBranch:
        """Create a new branch from a thought."""
        from datetime import datetime, timezone
        import uuid

        branch = ThoughtBranch(
            id=str(uuid.uuid4()),
            parent_thought_id=from_thought_id,
            name=name,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        self.branches.append(branch)
        return branch

    def switch_branch(self, branch_id: str) -> None:
        """Switch to a different branch."""
        if any(b.id == branch_id for b in self.branches):
            self.current_branch_id = branch_id

    def get_branch_thoughts(self, branch_id: Optional[str] = None) -> List[EnhancedThought]:
        """Get thoughts for a specific branch."""
        target_branch = branch_id or self.current_branch_id
        return [t for t in self.thoughts if t.branch_id == target_branch]

    def revise_thought(
        self,
        thought_id: str,
        new_content: str,
        reason: str = ""
    ) -> EnhancedThought:
        """Create a revision of an existing thought."""
        import uuid

        original = next((t for t in self.thoughts if t.id == thought_id), None)
        if not original:
            raise ValueError(f"Thought {thought_id} not found")

        revision = EnhancedThought(
            id=str(uuid.uuid4()),
            content=new_content,
            thought_type=ThoughtType.REVISION,
            thought_number=len(self.thoughts) + 1,
            total_thoughts=original.total_thoughts + 1,
            is_revision=True,
            revises_thought_id=thought_id,
            branch_id=self.current_branch_id,
            metadata={"revision_reason": reason}
        )
        self.add_thought(revision)
        return revision

    def get_revision_chain(self, thought_id: str) -> List[EnhancedThought]:
        """Get all revisions of a thought."""
        chain = []
        current_id = thought_id

        while current_id:
            thought = next((t for t in self.thoughts if t.id == current_id), None)
            if thought:
                chain.append(thought)
                # Find revision that revises this thought
                revision = next(
                    (t for t in self.thoughts if t.revises_thought_id == current_id),
                    None
                )
                current_id = revision.id if revision else None
            else:
                break

        return chain

    def to_json(self) -> Dict[str, Any]:
        """Convert session to JSON format."""
        return {
            "sessionId": self.session_id,
            "thoughts": [t.to_json() for t in self.thoughts],
            "branches": [
                {
                    "id": b.id,
                    "parentThoughtId": b.parent_thought_id,
                    "name": b.name,
                    "createdAt": b.created_at,
                    "merged": b.merged,
                    "mergedInto": b.merged_into
                }
                for b in self.branches
            ],
            "currentBranchId": self.current_branch_id,
            "createdAt": self.created_at
        }


# Convenience exports
__all__ = [
    # Enums
    "HookEvent",
    "PermissionDecision",
    "PermissionBehavior",
    "BlockDecision",
    # Core classes
    "HookInput",
    "HookResponse",
    "HookConfig",
    "SessionState",
    "TurnCounter",
    # MCP integration (V10.2)
    "ToolMatcher",
    "MCPContent",
    "MCPToolResult",
    # MCP Elicitation (V10.3)
    "ElicitationMode",
    "ElicitationAction",
    "ElicitationRequest",
    "ElicitationResponse",
    # MCP Sampling (V10.3)
    "ToolChoiceMode",
    "SamplingMessage",
    "ModelPreferences",
    "SamplingTool",
    "SamplingRequest",
    "SamplingResponse",
    # MCP Progress (V10.3)
    "ProgressNotification",
    # MCP Capabilities (V10.3)
    "MCPCapabilities",
    # MCP Subscriptions (V10.3)
    "ResourceSubscription",
    "SubscriptionManager",
    # Knowledge Graph (V10.3)
    "Entity",
    "Relation",
    "KnowledgeGraph",
    # MCP Tasks (V10.4)
    "TaskStatus",
    "TaskSupport",
    "MCPTask",
    "TaskRequest",
    "TaskResult",
    # MCP Prompts (V10.4)
    "PromptArgument",
    "MCPPrompt",
    "PromptMessage",
    # MCP Resources (V10.4)
    "ResourceAnnotations",
    "MCPResource",
    "ResourceTemplate",
    # MCP Logging (V10.4)
    "LogLevel",
    "LogMessage",
    # MCP Completion (V10.4)
    "CompletionRefType",
    "CompletionRequest",
    "CompletionResult",
    # Sequential Thinking (V10.4)
    "ThoughtData",
    "ThinkingSession",
    # Transport Abstraction (V10.4)
    "TransportType",
    "TransportConfig",
    "MCPSession",
    # Letta Memory Blocks (V10.4)
    "MemoryBlock",
    "BlockManager",
    # MCP Error Handling (V10.5)
    "MCPErrorCode",
    "ErrorData",
    "McpError",
    # MCP Cancellation (V10.5)
    "CancellationNotification",
    # MCP Ping (V10.5)
    "PingRequest",
    "PingResponse",
    # MCP Roots (V10.5)
    "MCPRoot",
    "RootsCapability",
    "ListRootsResult",
    # MCP Pagination (V10.5)
    "PaginatedRequest",
    "PaginatedResult",
    # MCP Tool Annotations (V10.5)
    "ToolAnnotations",
    # Letta Step Tracking (V10.5)
    "StepFeedbackType",
    "StepFilter",
    "StepMetrics",
    "StepTrace",
    "StepFeedback",
    "LettaStep",
    # Protocol Constants (V10.5)
    "MCP_LATEST_PROTOCOL_VERSION",
    "MCP_DEFAULT_NEGOTIATED_VERSION",
    # MCP Task Message Queue (V10.6)
    "QueuedMessage",
    "TaskMessageQueue",
    # MCP Resolver (V10.6)
    "Resolver",
    # MCP Task Helpers (V10.6)
    "MODEL_IMMEDIATE_RESPONSE_KEY",
    "RELATED_TASK_METADATA_KEY",
    "is_terminal_status",
    "generate_task_id",
    "TaskMetadata",
    "create_initial_task",
    # MCP Task Polling (V10.6)
    "TaskPollConfig",
    "TaskPollResult",
    # Letta Run Management (V10.6)
    "RunStatus",
    "StopReasonType",
    "RunFilter",
    "LettaRun",
    # Letta Project Support (V10.6)
    "ProjectFilter",
    "ExtendedStepFilter",
    # Tool Runner Patterns (V10.7)
    "EndStrategy",
    "CompactionMode",
    "CompactionControl",
    "ToolCache",
    # LangGraph Tool Patterns (V10.7)
    "ToolCall",
    "InjectedState",
    "InjectedStore",
    "ToolRuntime",
    "ToolCallRequest",
    "ToolCallResult",
    # Pydantic-AI Instrumentation (V10.7)
    "InstrumentationLevel",
    "InstrumentationSettings",
    # Enhanced Thinking (V10.7)
    "ThinkingCheckpoint",
    "BranchMergeResult",
    "EnhancedThinkingSession",
    # Tool Runner (V10.7)
    "ToolRunnerConfig",
    "ToolRunnerResult",
    "ToolRunner",
    # Thinking Part Parser (V10.7)
    "TextPart",
    "ThinkingPart",
    "split_content_into_text_and_thinking",
    # Structlog Processor Patterns (V10.8)
    "LogProcessorProtocol",
    "ContextVarBinding",
    "ContextVarManager",
    "ProcessorChain",
    "key_value_renderer",
    "logfmt_renderer",
    # OpenTelemetry Span Patterns (V10.8)
    "SpanStatusCode",
    "SpanStatus",
    "SpanContext",
    "SpanLink",
    "SpanEvent",
    "SpanKind",
    "Span",
    # Sentry Scope Patterns (V10.8)
    "ScopeType",
    "Breadcrumb",
    "ScopeData",
    "Scope",
    "ScopeManager",
    # pyribs Archive Patterns (V10.8)
    "Elite",
    "ArchiveAddStatus",
    "ArchiveAddResult",
    "ArchiveStats",
    "Archive",
    # Quality-Diversity Thinking (V10.8)
    "ThinkingElite",
    "ThinkingArchive",
    # Task-Master PRD Patterns (V10.9)
    "TaskPriority",
    "ComplexityLevel",
    "ToolMode",
    "PRDTask",
    "ComplexityAnalysis",
    "TaggedTaskList",
    "TaskDependencyGraph",
    "PRDWorkflow",
    # Shrimp Chain-of-Thought Patterns (V10.9)
    "TaskWorkflowMode",
    "ChainOfThought",
    "ThoughtProcess",
    "TaskDependency",
    "PersistentTaskMemory",
    "StructuredWorkflow",
    # Semgrep Security Patterns (V10.9)
    "SecuritySeverity",
    "RuleCategory",
    "SecurityFinding",
    "ASTNode",
    "SemgrepRule",
    "ASTAnalyzer",
    "SecurityScanner",
    "SecurityAudit",
    # LangGraph Graph Orchestration (V10.10)
    "NodeType",
    "EdgeType",
    "GraphNode",
    "GraphEdge",
    "GraphCheckpoint",
    "StateGraph",
    "GraphExecutionStatus",
    "GraphRuntime",
    "CyclicalWorkflow",
    # DSPy Programmatic LM Patterns (V10.10)
    "SignatureField",
    "DSPySignature",
    "DSPyExample",
    "DSPyModule",
    "DSPyPredict",
    "OptimizationStrategy",
    "OptimizationResult",
    "DSPyOptimizer",
    "DSPyTool",
    # MCP Multi-Server Patterns (V10.10)
    "MCPTransportType",
    "MCPServerConfig",
    "MCPToolArtifact",
    "MCPInterceptor",
    "InterceptorChain",
    "StatefulSession",
    "SessionPool",
    "MultiServerClient",
    "ResourceLoader",
    # FastMCP Server Patterns (V10.11)
    "FastMCPServer",
    "FastMCPTool",
    "FastMCPResource",
    "LifespanContext",
    "MCPContext",
    # Grafana MCP Observability (V10.11)
    "DashboardPanelType",
    "GrafanaPanel",
    "GrafanaDashboard",
    "PrometheusQuery",
    "LokiQuery",
    "AlertState",
    "GrafanaAlert",
    "IncidentSeverity",
    "IncidentStatus",
    "IncidentActivity",
    "GrafanaIncident",
    # Qdrant MCP Vector Patterns (V10.11)
    "DistanceMetric",
    "VectorConfig",
    "QdrantCollection",
    "QdrantPoint",
    "VectorSearchResult",
    "EmbeddingProvider",
    "EmbeddingModel",
    "SemanticSearch",
    # Postgres MCP Pro Patterns (V10.11)
    "DatabaseAccessMode",
    "QueryType",
    "QueryExecution",
    "PgIndexType",
    "IndexRecommendation",
    "HealthStatus",
    "DatabaseHealth",
    "ExplainFormat",
    "ExplainAnalysis",
    # Enhanced Sequential Thinking (V10.11)
    "ThoughtType",
    "ThoughtBranch",
    "EnhancedThought",
    "BranchingThinkingSession",
    # Functions
    "log_event",
    "get_logger",
    # Constants
    "PYDANTIC_AVAILABLE",
]
