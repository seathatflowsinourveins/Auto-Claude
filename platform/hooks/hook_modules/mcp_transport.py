#!/usr/bin/env python3
"""
MCP Transport Module - Transport Abstraction Layer

This module contains MCP transport patterns for stdio, SSE, and HTTP.
Extracted from hook_utils.py for modular architecture.

Exports:
- TransportType: Transport type enum
- TransportConfig: Transport configuration
- MCPSession: Session state for transports

Version: V1.0.0 (2026-01-30) - Extracted from hook_utils.py V10.11
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class TransportType(Enum):
    """MCP transport types.

    Per MCP SDK patterns:
    - STDIO: Standard input/output (default for CLI tools)
    - SSE: Server-Sent Events (for web clients)
    - HTTP: Streamable HTTP with sessions (for stateful connections)
    """
    STDIO = "stdio"
    SSE = "sse"
    HTTP = "http"


@dataclass
class TransportConfig:
    """
    Configuration for MCP transports.

    Per MCP SDK patterns:
    - stdio: Simple stdin/stdout, no session management
    - SSE: GET for streams, POST for messages, session via query param
    - HTTP: POST/GET/DELETE with mcp-session-id header, resumability

    Example:
        config = TransportConfig(
            transport_type=TransportType.HTTP,
            endpoint="http://localhost:3001",
            enable_resumability=True
        )
    """
    transport_type: TransportType
    endpoint: Optional[str] = None  # URL for SSE/HTTP
    session_id: Optional[str] = None
    port: int = 3001
    enable_resumability: bool = False  # For HTTP transport
    timeout_ms: int = 30000

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
        if self.timeout_ms != 30000:
            result["timeoutMs"] = self.timeout_ms
        return result

    @classmethod
    def stdio(cls) -> "TransportConfig":
        """Create a stdio transport config."""
        return cls(transport_type=TransportType.STDIO)

    @classmethod
    def sse(cls, endpoint: str, session_id: Optional[str] = None) -> "TransportConfig":
        """Create an SSE transport config."""
        return cls(
            transport_type=TransportType.SSE,
            endpoint=endpoint,
            session_id=session_id
        )

    @classmethod
    def http(
        cls,
        endpoint: str,
        port: int = 3001,
        enable_resumability: bool = False
    ) -> "TransportConfig":
        """Create an HTTP transport config with session support."""
        return cls(
            transport_type=TransportType.HTTP,
            endpoint=endpoint,
            port=port,
            enable_resumability=enable_resumability
        )


@dataclass
class MCPSession:
    """
    MCP session state for transport management.

    Tracks session lifecycle for SSE/HTTP transports.
    Provides session resumability for HTTP transport.

    Example:
        session = MCPSession(
            session_id="abc-123",
            transport_type=TransportType.HTTP
        )
    """
    session_id: str
    transport_type: TransportType
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_event_id: Optional[str] = None  # For SSE resumability
    is_active: bool = True
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable format."""
        result: Dict[str, Any] = {
            "sessionId": self.session_id,
            "transportType": self.transport_type.value,
            "createdAt": self.created_at.isoformat(),
            "isActive": self.is_active,
        }
        if self.last_event_id:
            result["lastEventId"] = self.last_event_id
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPSession":
        """Parse session from dictionary."""
        created = data.get("createdAt")
        return cls(
            session_id=data.get("sessionId", ""),
            transport_type=TransportType(data.get("transportType", "stdio")),
            created_at=datetime.fromisoformat(created.replace("Z", "+00:00")) if created else datetime.now(timezone.utc),
            last_event_id=data.get("lastEventId"),
            is_active=data.get("isActive", True),
            metadata=data.get("metadata")
        )

    def deactivate(self) -> None:
        """Mark session as inactive."""
        self.is_active = False

    def update_event_id(self, event_id: str) -> None:
        """Update the last event ID for resumability."""
        self.last_event_id = event_id


# Export all symbols
__all__ = [
    "TransportType",
    "TransportConfig",
    "MCPSession",
]
