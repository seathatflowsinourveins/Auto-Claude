#!/usr/bin/env python3
"""
Hook Core Module - Essential Hook Types and Configuration

This module contains the fundamental hook types used across all Claude Code hooks.
Extracted from hook_utils.py for modular architecture.

Exports:
- HookEvent: Enum of all hook event types
- PermissionDecision: Allow/Deny/Ask decisions
- PermissionBehavior: Permission request behaviors
- BlockDecision: Blocking decisions
- HookInput: Parsed hook input from Claude Code
- HookResponse: Response to send back to Claude Code
- HookConfig: Configuration management
- SessionState: Session state persistence
- TurnCounter: Turn counting for context management
- ToolMatcher: Tool name pattern matching
- log_event: Structured logging utility
- get_logger: Logger factory

Version: V1.0.0 (2026-01-30) - Extracted from hook_utils.py V10.11
"""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

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

    Supports all hook event output formats:
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
        """Convert to JSON response for Claude Code."""
        output: Dict[str, Any] = {}
        hook_specific: Dict[str, Any] = {}

        if self._event_type == HookEvent.PRE_TOOL_USE:
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
            hook_specific = {"hookEventName": "PostToolUse"}
            if self.block:
                output["decision"] = "block"
                output["reason"] = self.reason
            if self.additional_context:
                hook_specific["additionalContext"] = self.additional_context

        elif self._event_type in (HookEvent.STOP, HookEvent.SUBAGENT_STOP):
            if self.block:
                output["decision"] = "block"
                output["reason"] = self.reason

        elif self._event_type == HookEvent.USER_PROMPT_SUBMIT:
            hook_specific = {"hookEventName": "UserPromptSubmit"}
            if self.block:
                output["decision"] = "block"
                output["reason"] = self.reason
            if self.additional_context:
                hook_specific["additionalContext"] = self.additional_context

        elif self._event_type == HookEvent.SESSION_START:
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
        try:
            state = {}
            for line in self._state_file.read_text().split("\n"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    state[key.strip()] = value.strip()
            return state
        except Exception:
            return {}

    def save(self, state: Dict[str, str]) -> None:
        """Save session state to file."""
        lines = [f"{k}={v}" for k, v in state.items()]
        self._state_file.write_text("\n".join(lines))

    def update(self, **kwargs: str) -> None:
        """Update specific state values."""
        state = self.load()
        state.update(kwargs)
        self.save(state)


class TurnCounter:
    """
    Counts tool calls/turns within a session for context management.

    Used by strategic compact hook to trigger proactive compaction.
    """

    def __init__(self, config: HookConfig):
        self.config = config
        self._count_file = config.get_turn_count_file()

    def load(self) -> int:
        """Load current turn count."""
        if not self._count_file.exists():
            return 0
        try:
            return int(self._count_file.read_text().strip())
        except (ValueError, OSError):
            return 0

    def increment(self) -> int:
        """Increment and return new turn count."""
        count = self.load() + 1
        self._count_file.write_text(str(count))
        return count

    def reset(self) -> None:
        """Reset turn count to zero."""
        self._count_file.write_text("0")

    def get_and_increment(self) -> int:
        """Atomic get-and-increment."""
        return self.increment()


class ToolMatcher:
    """
    Matches tool names against patterns for hook filtering.

    Supports:
    - Exact match: "Write"
    - Wildcard: "mcp__*"
    - Regex: "^(Write|Edit)$"
    """

    def __init__(self, patterns: List[str]):
        self.patterns = patterns
        self._compiled: List[re.Pattern[str]] = []
        for p in patterns:
            if p == "*":
                self._compiled.append(re.compile(".*"))
            elif "*" in p and not p.startswith("^"):
                # Convert glob to regex
                regex = p.replace("*", ".*")
                self._compiled.append(re.compile(f"^{regex}$"))
            else:
                self._compiled.append(re.compile(p))

    def matches(self, tool_name: str) -> bool:
        """Check if tool name matches any pattern."""
        return any(p.match(tool_name) for p in self._compiled)


def log_event(
    event: str,
    data: Dict[str, Any],
    log_file: Optional[Path] = None,
    level: str = "info"
) -> None:
    """
    Log an event to file and optionally structlog.

    Args:
        event: Event name
        data: Event data
        log_file: Optional file path for JSONL logging
        level: Log level (info, warning, error)
    """
    logger = structlog.get_logger(__name__)
    log_method = getattr(logger, level, logger.info)

    # Add timestamp
    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    data["event"] = event

    # Log to structlog
    log_method(event, **data)

    # Log to file if specified
    if log_file:
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(data) + "\n")
        except OSError:
            pass


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger for a module."""
    return structlog.get_logger(name)


# Export all symbols
__all__ = [
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
]
