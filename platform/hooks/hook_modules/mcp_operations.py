#!/usr/bin/env python3
"""
MCP Operations Module - Ping, Roots, Pagination, Tool Annotations

This module contains MCP operational patterns for health checks,
filesystem roots, pagination, and tool annotations.
Extracted from hook_utils.py for modular architecture.

Exports:
- PingRequest: Connection health verification
- PingResponse: Ping response
- MCPRoot: Filesystem root definition
- RootsCapability: Roots capability declaration
- ListRootsResult: Roots list result
- PaginatedRequest: Cursor-based pagination request
- PaginatedResult: Pagination result
- ToolAnnotations: Tool behavioral hints

Version: V1.0.0 (2026-01-30) - Extracted from hook_utils.py V10.11
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# =============================================================================
# PING
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
# ROOTS
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

    def __post_init__(self) -> None:
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
            uri=data.get("uri", "file:///"),
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ListRootsResult":
        """Parse from MCP response."""
        roots = [MCPRoot.from_dict(r) for r in data.get("roots", [])]
        return cls(roots=roots)


# =============================================================================
# PAGINATION
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
# TOOL ANNOTATIONS
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

    @classmethod
    def safe_write(cls, title: str = "") -> "ToolAnnotations":
        """Create annotations for safe, idempotent write tools."""
        return cls(
            title=title or None,
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True
        )


# Export all symbols
__all__ = [
    "PingRequest",
    "PingResponse",
    "MCPRoot",
    "RootsCapability",
    "ListRootsResult",
    "PaginatedRequest",
    "PaginatedResult",
    "ToolAnnotations",
]
