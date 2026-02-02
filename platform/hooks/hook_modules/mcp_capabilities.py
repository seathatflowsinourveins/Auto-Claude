#!/usr/bin/env python3
"""
MCP Capabilities Module - Progress and Capability Negotiation

This module contains MCP capability negotiation and progress reporting patterns.
Extracted from hook_utils.py for modular architecture.

Exports:
- ProgressNotification: Progress reporting for long-running operations
- MCPCapabilities: Client/server capability negotiation
- ResourceSubscription: Resource subscription patterns
- SubscriptionManager: Manager for resource subscriptions

Version: V1.0.0 (2026-01-30) - Extracted from hook_utils.py V10.11
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set


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

    def to_notification(self) -> Dict[str, Any]:
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

    def is_complete(self) -> bool:
        """Check if progress indicates completion."""
        if self.total:
            return self.progress >= self.total
        return False


@dataclass
class MCPCapabilities:
    """
    MCP client/server capabilities for feature negotiation.

    Per MCP 2025-11-25, capabilities are declared during initialization:
    - sampling: Client can process sampling requests
    - elicitation: Client can display elicitation dialogs
    - tools: Client supports tool use in sampling
    - roots: Client can provide filesystem roots
    """
    sampling: bool = False
    sampling_tools: bool = False
    elicitation: bool = False
    roots: bool = False

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP capabilities format."""
        caps: Dict[str, Any] = {}
        if self.sampling:
            caps["sampling"] = {"tools": {}} if self.sampling_tools else {}
        if self.elicitation:
            caps["elicitation"] = {}
        if self.roots:
            caps["roots"] = {}
        return caps

    def supports_sampling(self) -> bool:
        """Check if client supports basic sampling."""
        return self.sampling

    def supports_sampling_tools(self) -> bool:
        """Check if client supports tool use in sampling."""
        return self.sampling and self.sampling_tools

    def supports_elicitation(self) -> bool:
        """Check if client supports elicitation dialogs."""
        return self.elicitation


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

    def __init__(self) -> None:
        self._subscriptions: Dict[str, Set[str]] = {}

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

    def clear_session(self, session_id: str = "default") -> None:
        """Remove all subscriptions for a session."""
        empty_uris = []
        for uri, sessions in self._subscriptions.items():
            sessions.discard(session_id)
            if not sessions:
                empty_uris.append(uri)
        for uri in empty_uris:
            del self._subscriptions[uri]


# Export all symbols
__all__ = [
    "ProgressNotification",
    "MCPCapabilities",
    "ResourceSubscription",
    "SubscriptionManager",
]
