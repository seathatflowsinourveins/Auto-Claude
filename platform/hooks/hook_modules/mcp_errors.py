#!/usr/bin/env python3
"""
MCP Errors Module - JSON-RPC Error Handling

This module contains MCP error handling patterns per JSON-RPC 2.0.
Extracted from hook_utils.py for modular architecture.

Exports:
- MCPErrorCode: Standard error codes
- ErrorData: Error data attachment
- McpError: MCP protocol exception
- CancellationNotification: Request cancellation

Version: V1.0.0 (2026-01-30) - Extracted from hook_utils.py V10.11
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


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

    @classmethod
    def is_retriable(cls, code: int) -> bool:
        """Check if error is retriable."""
        # Internal errors and connection issues may be retriable
        return code in (cls.INTERNAL_ERROR, cls.CONNECTION_CLOSED)


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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ErrorData":
        """Parse from dictionary."""
        return cls(
            message=data.get("message"),
            details=data.get("details"),
            retry_after_ms=data.get("retryAfterMs")
        )


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
    def from_dict(cls, data: Dict[str, Any]) -> "McpError":
        """Parse from JSON-RPC error object."""
        error_data = None
        if "data" in data:
            error_data = ErrorData.from_dict(data["data"])
        return cls(
            code=data.get("code", MCPErrorCode.INTERNAL_ERROR),
            message=data.get("message"),
            data=error_data
        )

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

    @classmethod
    def internal_error(cls, details: Optional[str] = None) -> "McpError":
        """Create an internal error (-32603)."""
        return cls(
            code=MCPErrorCode.INTERNAL_ERROR,
            data=ErrorData(message=details) if details else None
        )

    @property
    def is_retriable(self) -> bool:
        """Check if this error is retriable."""
        return MCPErrorCode.is_retriable(self.code)


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


# Export all symbols
__all__ = [
    "MCPErrorCode",
    "ErrorData",
    "McpError",
    "CancellationNotification",
]
