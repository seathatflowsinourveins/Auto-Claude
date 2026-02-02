#!/usr/bin/env python3
"""
MCP Elicitation Module - User Input Collection Patterns

This module contains MCP elicitation patterns for collecting user input.
Extracted from hook_utils.py for modular architecture.

Exports:
- ElicitationMode: Form vs URL elicitation modes
- ElicitationAction: User response actions
- ElicitationRequest: Request for user input
- ElicitationResponse: User's response to elicitation

Version: V1.0.0 (2026-01-30) - Extracted from hook_utils.py V10.11
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class ElicitationMode(Enum):
    """MCP elicitation modes for collecting user input.

    Per MCP Specification 2025-11-25:
    - FORM: Structured form with JSON Schema validation
    - URL: Out-of-band URL for OAuth, payments, etc.
    """
    FORM = "form"
    URL = "url"


class ElicitationAction(Enum):
    """User response actions for elicitation requests.

    Per MCP Specification 2025-11-25:
    - ACCEPT: User provided the requested data
    - DECLINE: User declined to provide data
    - CANCEL: User cancelled the dialog
    """
    ACCEPT = "accept"
    DECLINE = "decline"
    CANCEL = "cancel"


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
        request = ElicitationRequest.url_mode(
            message="Please complete payment",
            url="https://payment.example.com/checkout?token=xxx",
            description="Secure payment required"
        )
    """
    mode: ElicitationMode
    message: str
    schema: Optional[Dict[str, Any]] = None
    redirect_url: Optional[str] = None
    description: Optional[str] = None
    timeout_ms: int = 600000  # 10 minutes default

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP elicitation/create request params."""
        if self.mode == ElicitationMode.FORM:
            return {
                "message": self.message,
                "requestedSchema": self.schema or {},
            }
        else:  # URL mode
            result: Dict[str, Any] = {
                "message": self.message,
                "url": self.redirect_url or "",
            }
            if self.description:
                result["description"] = self.description
            return result

    @classmethod
    def form(
        cls,
        message: str,
        schema: Dict[str, Any],
        timeout_ms: int = 600000
    ) -> "ElicitationRequest":
        """Create a form mode elicitation request with JSON Schema."""
        return cls(
            mode=ElicitationMode.FORM,
            message=message,
            schema=schema,
            timeout_ms=timeout_ms
        )

    @classmethod
    def url_mode(
        cls,
        message: str,
        url: str,
        description: Optional[str] = None
    ) -> "ElicitationRequest":
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


# Export all symbols
__all__ = [
    "ElicitationMode",
    "ElicitationAction",
    "ElicitationRequest",
    "ElicitationResponse",
]
