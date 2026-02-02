"""
Zep Compatibility Layer for Python 3.14+

The native zep_python SDK fails on Python 3.14 due to Pydantic type inference issues:
  "unable to infer type for attribute 'message'"

This compatibility layer provides HTTP-based access to Zep's memory API,
bypassing the SDK's Pydantic models that cause issues.

Usage:
    from core.memory.zep_compat import ZepCompat, ZepMessage

    client = ZepCompat(api_key="your-key")
    client.add_memory(session_id, messages)
    memory = client.get_memory(session_id)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


@dataclass
class ZepMessage:
    """A message in a Zep session."""
    role: str
    content: str
    role_type: str = "user"  # user, assistant, system
    metadata: Dict[str, Any] = field(default_factory=dict)
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ZepMemory:
    """Memory retrieved from Zep."""
    messages: List[ZepMessage] = field(default_factory=list)
    summary: Optional[str] = None
    facts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ZepSearchResult:
    """A search result from Zep."""
    message: ZepMessage
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ZepCompat:
    """
    HTTP-based Zep client compatible with Python 3.14+.

    Provides memory operations without using the native zep_python SDK
    which has Pydantic compatibility issues on Python 3.14.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        enabled: bool = True
    ):
        # V44 FIX: Environment-configurable Zep URL (was hardcoded)
        import os
        if base_url is None:
            base_url = os.environ.get("ZEP_BASE_URL", "https://api.getzep.com")
        """
        Initialize Zep client.

        Args:
            api_key: Zep API key (optional for testing)
            base_url: Zep API base URL
            enabled: Whether to actually make API calls
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.enabled = enabled and HTTPX_AVAILABLE and api_key is not None

        if self.enabled:
            self.client = httpx.Client(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
        else:
            self.client = None

        # Local storage for testing when disabled
        self._local_sessions: Dict[str, List[ZepMessage]] = {}

    def _to_api_message(self, msg: ZepMessage) -> dict:
        """Convert ZepMessage to API format."""
        return {
            "uuid": msg.uuid,
            "role": msg.role,
            "role_type": msg.role_type,
            "content": msg.content,
            "metadata": msg.metadata,
            "created_at": msg.created_at
        }

    def _from_api_message(self, data: dict) -> ZepMessage:
        """Convert API response to ZepMessage."""
        return ZepMessage(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            role_type=data.get("role_type", "user"),
            metadata=data.get("metadata", {}),
            uuid=data.get("uuid", str(uuid.uuid4())),
            created_at=data.get("created_at", datetime.utcnow().isoformat())
        )

    def add_memory(
        self,
        session_id: str,
        messages: List[ZepMessage]
    ) -> dict:
        """
        Add messages to a session's memory.

        Args:
            session_id: The session identifier
            messages: List of messages to add

        Returns:
            API response or local confirmation
        """
        if not self.enabled:
            if session_id not in self._local_sessions:
                self._local_sessions[session_id] = []
            self._local_sessions[session_id].extend(messages)
            return {"status": "ok", "local": True, "count": len(messages)}

        api_messages = [self._to_api_message(m) for m in messages]
        response = self.client.post(
            f"/api/v2/sessions/{session_id}/memory",
            json={"messages": api_messages}
        )
        response.raise_for_status()
        return response.json()

    def get_memory(
        self,
        session_id: str,
        lastn: int = 10
    ) -> ZepMemory:
        """
        Get memory for a session.

        Args:
            session_id: The session identifier
            lastn: Number of recent messages to retrieve

        Returns:
            ZepMemory containing messages and summary
        """
        if not self.enabled:
            messages = self._local_sessions.get(session_id, [])[-lastn:]
            return ZepMemory(messages=messages)

        response = self.client.get(
            f"/api/v2/sessions/{session_id}/memory",
            params={"lastn": lastn}
        )
        response.raise_for_status()
        data = response.json()

        return ZepMemory(
            messages=[self._from_api_message(m) for m in data.get("messages", [])],
            summary=data.get("summary"),
            facts=data.get("facts", []),
            metadata=data.get("metadata", {})
        )

    def search(
        self,
        session_id: str,
        query: str,
        limit: int = 5,
        search_type: str = "similarity"
    ) -> List[ZepSearchResult]:
        """
        Search session memory.

        Args:
            session_id: The session identifier
            query: Search query
            limit: Maximum results to return
            search_type: Type of search (similarity, mmr)

        Returns:
            List of search results
        """
        if not self.enabled:
            # Simple local search
            messages = self._local_sessions.get(session_id, [])
            results = []
            query_lower = query.lower()
            for msg in messages:
                if query_lower in msg.content.lower():
                    results.append(ZepSearchResult(
                        message=msg,
                        score=1.0 if query_lower in msg.content.lower() else 0.0
                    ))
            return results[:limit]

        response = self.client.post(
            f"/api/v2/sessions/{session_id}/search",
            json={
                "text": query,
                "limit": limit,
                "search_type": search_type
            }
        )
        response.raise_for_status()
        data = response.json()

        return [
            ZepSearchResult(
                message=self._from_api_message(r.get("message", {})),
                score=r.get("score", 0.0),
                metadata=r.get("metadata", {})
            )
            for r in data.get("results", [])
        ]

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and its memory.

        Args:
            session_id: The session identifier

        Returns:
            True if successful
        """
        if not self.enabled:
            if session_id in self._local_sessions:
                del self._local_sessions[session_id]
            return True

        response = self.client.delete(f"/api/v2/sessions/{session_id}")
        return response.status_code in (200, 204)

    def get_session(self, session_id: str) -> Optional[dict]:
        """
        Get session metadata.

        Args:
            session_id: The session identifier

        Returns:
            Session data or None
        """
        if not self.enabled:
            if session_id in self._local_sessions:
                return {
                    "session_id": session_id,
                    "message_count": len(self._local_sessions[session_id])
                }
            return None

        response = self.client.get(f"/api/v2/sessions/{session_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def create_session(
        self,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> dict:
        """
        Create a new session.

        Args:
            session_id: Optional session ID (generated if not provided)
            metadata: Optional session metadata

        Returns:
            Created session data
        """
        session_id = session_id or str(uuid.uuid4())

        if not self.enabled:
            self._local_sessions[session_id] = []
            return {"session_id": session_id, "metadata": metadata or {}}

        response = self.client.post(
            "/api/v2/sessions",
            json={
                "session_id": session_id,
                "metadata": metadata or {}
            }
        )
        response.raise_for_status()
        return response.json()


# Compatibility exports
ZEP_COMPAT_AVAILABLE = True

def get_zep_client(
    api_key: Optional[str] = None,
    base_url: str = "https://api.getzep.com"
) -> ZepCompat:
    """Factory function to create a Zep client."""
    return ZepCompat(api_key=api_key, base_url=base_url)


# For drop-in replacement of zep_python
class AsyncZepClient:
    """Async wrapper (placeholder for future implementation)."""
    pass


class ZepClient(ZepCompat):
    """Alias for backward compatibility with zep_python naming."""
    pass
