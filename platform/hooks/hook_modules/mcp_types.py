#!/usr/bin/env python3
"""
MCP Types Module - Core MCP Content and Tool Types

This module contains MCP content types, tool results, and error handling.
Extracted from hook_utils.py for modular architecture.

Exports:
- MCPContent: Structured content for MCP tools
- MCPToolResult: Tool execution results
- MCPErrorCode: JSON-RPC error codes
- ErrorData: Error metadata
- McpError: MCP error exception
- LogLevel: RFC 5424 syslog levels

Version: V1.0.0 (2026-01-30) - Extracted from hook_utils.py V10.11
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


@dataclass
class MCPContent:
    """
    MCP content wrapper for tool responses.

    Supports MCP structuredContent format per MCP 2025-11-25 spec.
    Used for returning rich content from tools.
    """
    type: str = "text"  # text, image, audio, resource
    text: Optional[str] = None
    data: Optional[str] = None  # Base64 for images/audio
    mime_type: Optional[str] = None
    uri: Optional[str] = None  # For resource type
    blob: Optional[str] = None  # For blob content

    # Structured content (MCP 2025-11-25)
    structured: Optional[Dict[str, Any]] = None
    annotations: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP content format."""
        content: Dict[str, Any] = {"type": self.type}

        if self.type == "text" and self.text:
            content["text"] = self.text
        elif self.type == "image" and self.data:
            content["data"] = self.data
            content["mimeType"] = self.mime_type if self.mime_type else "image/png"
        elif self.type == "audio" and self.data:
            content["data"] = self.data
            content["mimeType"] = self.mime_type or "audio/wav"
        elif self.type == "resource":
            if self.uri:
                content["resource"] = {"uri": self.uri}
                if self.text:
                    content["resource"]["text"] = self.text
                if self.blob:
                    content["resource"]["blob"] = self.blob
                    content["resource"]["mimeType"] = self.mime_type

        if self.structured:
            content["structuredContent"] = self.structured
        if self.annotations:
            content["annotations"] = self.annotations

        return content

    @classmethod
    def text_content(cls, text: str) -> "MCPContent":
        """Create text content."""
        return cls(type="text", text=text)

    @classmethod
    def image_content(cls, data: str, mime_type: str = "image/png") -> "MCPContent":
        """Create image content from base64 data."""
        return cls(type="image", data=data, mime_type=mime_type)

    @classmethod
    def structured_content(cls, data: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> "MCPContent":
        """Create structured content."""
        structured: Dict[str, Any] = {"data": data}
        if schema:
            structured["schema"] = schema
        return cls(type="text", structured=structured)


@dataclass
class MCPToolResult:
    """
    Result from an MCP tool execution.

    Follows MCP tool result format with content array.
    """
    content: List[MCPContent] = field(default_factory=list)
    is_error: bool = False
    error_code: Optional[int] = None
    error_message: Optional[str] = None

    # Structured output (MCP 2025-11-25)
    structured_content: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP result format."""
        result: Dict[str, Any] = {
            "content": [c.to_dict() for c in self.content],
            "isError": self.is_error
        }

        if self.structured_content:
            result["structuredContent"] = self.structured_content

        if self.is_error and self.error_code:
            result["errorCode"] = self.error_code
            result["errorMessage"] = self.error_message

        return result

    @classmethod
    def success(cls, text: str) -> "MCPToolResult":
        """Create successful text result."""
        return cls(content=[MCPContent.text_content(text)])

    @classmethod
    def error(cls, message: str, code: int = -32000) -> "MCPToolResult":
        """Create error result."""
        return cls(
            content=[MCPContent.text_content(f"Error: {message}")],
            is_error=True,
            error_code=code,
            error_message=message
        )

    @classmethod
    def structured(cls, data: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> "MCPToolResult":
        """Create structured result."""
        return cls(
            content=[MCPContent.text_content(str(data))],
            structured_content={"data": data, "schema": schema} if schema else {"data": data}
        )


class MCPErrorCode:
    """
    MCP JSON-RPC error codes per MCP 2025-11-25 spec.

    Standard JSON-RPC codes:
    -32700: Parse error
    -32600: Invalid request
    -32601: Method not found
    -32602: Invalid params
    -32603: Internal error

    MCP-specific codes:
    -32000 to -32099: Reserved for MCP
    """
    # JSON-RPC standard errors
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # MCP-specific errors
    REQUEST_CANCELLED = -32000
    CONTENT_TOO_LARGE = -32001
    RESOURCE_NOT_FOUND = -32002
    RESOURCE_UNAVAILABLE = -32003
    TOOL_NOT_FOUND = -32004
    TOOL_EXECUTION_FAILED = -32005
    PROMPT_NOT_FOUND = -32006
    COMPLETION_NOT_AVAILABLE = -32007
    ELICITATION_DENIED = -32008
    SAMPLING_DENIED = -32009

    @classmethod
    def get_message(cls, code: int) -> str:
        """Get default message for error code."""
        messages = {
            cls.PARSE_ERROR: "Parse error",
            cls.INVALID_REQUEST: "Invalid request",
            cls.METHOD_NOT_FOUND: "Method not found",
            cls.INVALID_PARAMS: "Invalid params",
            cls.INTERNAL_ERROR: "Internal error",
            cls.REQUEST_CANCELLED: "Request cancelled",
            cls.CONTENT_TOO_LARGE: "Content too large",
            cls.RESOURCE_NOT_FOUND: "Resource not found",
            cls.RESOURCE_UNAVAILABLE: "Resource unavailable",
            cls.TOOL_NOT_FOUND: "Tool not found",
            cls.TOOL_EXECUTION_FAILED: "Tool execution failed",
            cls.PROMPT_NOT_FOUND: "Prompt not found",
            cls.COMPLETION_NOT_AVAILABLE: "Completion not available",
            cls.ELICITATION_DENIED: "Elicitation denied",
            cls.SAMPLING_DENIED: "Sampling denied",
        }
        return messages.get(code, "Unknown error")


@dataclass
class ErrorData:
    """
    Additional error data for MCP errors.

    Provides context for error recovery and debugging.
    """
    type: str = "error"
    retryable: bool = False
    retry_after_seconds: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP error data format."""
        data: Dict[str, Any] = {"type": self.type}
        if self.retryable:
            data["retryable"] = True
            if self.retry_after_seconds:
                data["retryAfterSeconds"] = self.retry_after_seconds
        if self.details:
            data["details"] = self.details
        return data


class McpError(Exception):
    """
    MCP-specific exception with JSON-RPC error code.

    Provides structured error information for MCP responses.
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
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-RPC error format."""
        error: Dict[str, Any] = {
            "code": self.code,
            "message": self.message
        }
        if self.data:
            error["data"] = self.data.to_dict()
        return error

    @classmethod
    def parse_error(cls, details: Optional[str] = None) -> "McpError":
        """Create parse error."""
        data = ErrorData(details={"parseError": details}) if details else None
        return cls(MCPErrorCode.PARSE_ERROR, data=data)

    @classmethod
    def invalid_params(cls, param: str, reason: str) -> "McpError":
        """Create invalid params error."""
        return cls(
            MCPErrorCode.INVALID_PARAMS,
            f"Invalid parameter '{param}': {reason}",
            ErrorData(details={"param": param, "reason": reason})
        )

    @classmethod
    def tool_not_found(cls, tool_name: str) -> "McpError":
        """Create tool not found error."""
        return cls(
            MCPErrorCode.TOOL_NOT_FOUND,
            f"Tool '{tool_name}' not found",
            ErrorData(details={"tool": tool_name})
        )

    @classmethod
    def internal_error(cls, message: str, retryable: bool = False) -> "McpError":
        """Create internal error."""
        return cls(
            MCPErrorCode.INTERNAL_ERROR,
            message,
            ErrorData(retryable=retryable)
        )


class LogLevel(Enum):
    """
    RFC 5424 syslog severity levels for MCP logging.

    From most to least severe:
    - emergency: System is unusable
    - alert: Action must be taken immediately
    - critical: Critical conditions
    - error: Error conditions
    - warning: Warning conditions
    - notice: Normal but significant condition
    - info: Informational messages
    - debug: Debug-level messages
    """
    EMERGENCY = "emergency"
    ALERT = "alert"
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    NOTICE = "notice"
    INFO = "info"
    DEBUG = "debug"

    @classmethod
    def from_string(cls, level: str) -> "LogLevel":
        """Convert string to LogLevel."""
        mapping = {
            "emergency": cls.EMERGENCY,
            "alert": cls.ALERT,
            "critical": cls.CRITICAL,
            "error": cls.ERROR,
            "warning": cls.WARNING,
            "warn": cls.WARNING,
            "notice": cls.NOTICE,
            "info": cls.INFO,
            "debug": cls.DEBUG,
        }
        return mapping.get(level.lower(), cls.INFO)

    @property
    def severity(self) -> int:
        """Get numeric severity (0=emergency, 7=debug)."""
        order = [
            self.EMERGENCY, self.ALERT, self.CRITICAL, self.ERROR,
            self.WARNING, self.NOTICE, self.INFO, self.DEBUG
        ]
        return order.index(self)


@dataclass
class LogMessage:
    """
    MCP log message for server logging.

    Sent via notifications/logging/message.
    """
    level: LogLevel
    logger: str
    message: str
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP log message format."""
        msg: Dict[str, Any] = {
            "level": self.level.value,
            "logger": self.logger,
            "message": self.message
        }
        if self.data:
            msg["data"] = self.data
        return msg


# Export all symbols
__all__ = [
    "MCPContent",
    "MCPToolResult",
    "MCPErrorCode",
    "ErrorData",
    "McpError",
    "LogLevel",
    "LogMessage",
]
