#!/usr/bin/env python3
"""
Memory Providers - SDK-specific implementations.
Wraps Letta, Zep, and Mem0 SDKs with a unified interface.
"""

from __future__ import annotations

import os
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Optional
from datetime import datetime

import structlog
from dotenv import load_dotenv

from .types import (
    MemoryEntry,
    MemoryProvider,
    MemoryTier,
    SearchResult,
    ProviderConfig,
)

load_dotenv()
logger = structlog.get_logger(__name__)

# Check SDK availability
try:
    from mem0 import Memory as Mem0Memory
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    logger.warning("mem0 not available - install with: pip install mem0ai")

try:
    from zep_python import ZepClient
    from zep_python.memory import Session, Memory as ZepMemory
    ZEP_AVAILABLE = True
except ImportError as e:
    ZEP_AVAILABLE = False
    logger.warning(f"zep-python not installed: {e}")
except Exception as e:
    # Pydantic V1 ConfigError on Python 3.14, or other compatibility issues
    ZEP_AVAILABLE = False
    logger.warning(f"zep-python not available (compatibility issue): {e}")

try:
    import letta
    from letta import create_client
    LETTA_AVAILABLE = True
except ImportError:
    LETTA_AVAILABLE = False
    logger.warning("letta not available - install with: pip install letta")

# Cross-session memory (platform module)
try:
    import sys
    from pathlib import Path
    platform_path = Path(__file__).parent.parent.parent / "platform" / "core"
    if str(platform_path) not in sys.path:
        sys.path.insert(0, str(platform_path))
    from cross_session_memory import (
        CrossSessionMemory,
        get_memory_store,
        Memory as CrossSessionMemoryEntry,
    )
    CROSS_SESSION_AVAILABLE = True
except ImportError as e:
    CROSS_SESSION_AVAILABLE = False
    logger.warning(f"cross_session_memory not available: {e}")


class MemoryProviderBase(ABC):
    """Abstract base class for memory providers."""

    provider_type: MemoryProvider

    @abstractmethod
    async def store(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> MemoryEntry:
        """Store a memory entry."""
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 10,
        user_id: Optional[str] = None,
    ) -> list[SearchResult]:
        """Search for memories."""
        pass

    @abstractmethod
    async def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory by ID."""
        pass

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory entry."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the provider connection."""
        pass


class Mem0Provider(MemoryProviderBase):
    """Mem0 memory provider for persistent cross-session memory."""

    provider_type = MemoryProvider.MEM0

    def __init__(self, config: Optional[ProviderConfig] = None):
        if not MEM0_AVAILABLE:
            raise ImportError("mem0 not installed")

        self.config = config or ProviderConfig(provider=MemoryProvider.MEM0)

        # Initialize Mem0 with optional API key
        mem0_config = {}
        if self.config.api_key:
            mem0_config["api_key"] = self.config.api_key
        if self.config.options:
            mem0_config.update(self.config.options)

        self.memory = Mem0Memory(mem0_config) if mem0_config else Mem0Memory()
        logger.info("mem0_provider_initialized")

    async def store(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> MemoryEntry:
        """Store memory using Mem0."""
        loop = asyncio.get_event_loop()

        # Mem0 uses sync API, run in executor
        def _store():
            result = self.memory.add(
                content,
                user_id=user_id or "default",
                metadata=metadata or {},
            )
            return result

        result = await loop.run_in_executor(None, _store)

        # Extract ID from result
        memory_id = result.get("id", str(datetime.now().timestamp()))

        return MemoryEntry(
            id=memory_id,
            content=content,
            tier=MemoryTier.ARCHIVAL,
            provider=MemoryProvider.MEM0,
            metadata=metadata or {},
        )

    async def search(
        self,
        query: str,
        limit: int = 10,
        user_id: Optional[str] = None,
    ) -> list[SearchResult]:
        """Search memories using Mem0."""
        loop = asyncio.get_event_loop()

        def _search():
            results = self.memory.search(
                query,
                user_id=user_id or "default",
                limit=limit,
            )
            return results

        results = await loop.run_in_executor(None, _search)

        return [
            SearchResult(
                id=r.get("id", ""),
                content=r.get("memory", r.get("text", "")),
                score=r.get("score", 0.0),
                metadata=r.get("metadata", {}),
                provider=MemoryProvider.MEM0,
            )
            for r in results
        ]

    async def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory from Mem0."""
        loop = asyncio.get_event_loop()

        def _get():
            try:
                result = self.memory.get(memory_id)
                return result
            except Exception:
                return None

        result = await loop.run_in_executor(None, _get)

        if result:
            return MemoryEntry(
                id=memory_id,
                content=result.get("memory", result.get("text", "")),
                tier=MemoryTier.ARCHIVAL,
                provider=MemoryProvider.MEM0,
                metadata=result.get("metadata", {}),
            )
        return None

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory from Mem0."""
        loop = asyncio.get_event_loop()

        def _delete():
            try:
                self.memory.delete(memory_id)
                return True
            except Exception:
                return False

        return await loop.run_in_executor(None, _delete)

    async def close(self) -> None:
        """Close Mem0 provider (no-op for mem0)."""
        pass


class ZepProvider(MemoryProviderBase):
    """Zep memory provider for session-based memory."""

    provider_type = MemoryProvider.ZEP

    def __init__(self, config: Optional[ProviderConfig] = None):
        if not ZEP_AVAILABLE:
            raise ImportError("zep-python not installed")

        self.config = config or ProviderConfig(provider=MemoryProvider.ZEP)

        # Initialize Zep client
        api_key = self.config.api_key or os.getenv("ZEP_API_KEY")
        base_url = self.config.base_url or os.getenv("ZEP_API_URL", "http://localhost:8000")

        self.client = ZepClient(api_key=api_key, base_url=base_url)
        self._sessions: dict[str, Any] = {}
        logger.info("zep_provider_initialized", base_url=base_url)

    async def _get_or_create_session(self, session_id: str) -> Any:
        """Get or create a Zep session."""
        if session_id not in self._sessions:
            try:
                session = await asyncio.to_thread(
                    self.client.memory.get_session,
                    session_id,
                )
            except Exception:
                # Create new session
                session = await asyncio.to_thread(
                    self.client.memory.add_session,
                    Session(session_id=session_id),
                )
            self._sessions[session_id] = session
        return self._sessions[session_id]

    async def store(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> MemoryEntry:
        """Store memory using Zep."""
        session_id = user_id or "default"
        await self._get_or_create_session(session_id)

        # Store as Zep memory
        memory = ZepMemory(
            messages=[{"role": "system", "content": content}],
            metadata=metadata or {},
        )

        await asyncio.to_thread(
            self.client.memory.add_memory,
            session_id,
            memory,
        )

        memory_id = f"{session_id}:{datetime.now().timestamp()}"

        return MemoryEntry(
            id=memory_id,
            content=content,
            tier=MemoryTier.ARCHIVAL,
            provider=MemoryProvider.ZEP,
            metadata=metadata or {},
        )

    async def search(
        self,
        query: str,
        limit: int = 10,
        user_id: Optional[str] = None,
    ) -> list[SearchResult]:
        """Search memories using Zep."""
        session_id = user_id or "default"

        try:
            results = await asyncio.to_thread(
                self.client.memory.search_memory,
                session_id,
                query,
                limit=limit,
            )

            return [
                SearchResult(
                    id=f"{session_id}:{i}",
                    content=r.message.get("content", "") if hasattr(r, "message") else str(r),
                    score=r.score if hasattr(r, "score") else 0.0,
                    metadata=r.metadata if hasattr(r, "metadata") else {},
                    provider=MemoryProvider.ZEP,
                )
                for i, r in enumerate(results)
            ]
        except Exception as e:
            logger.warning("zep_search_failed", error=str(e))
            return []

    async def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get session memory from Zep."""
        try:
            parts = memory_id.split(":")
            session_id = parts[0] if parts else memory_id

            memory = await asyncio.to_thread(
                self.client.memory.get_memory,
                session_id,
            )

            if memory and memory.messages:
                return MemoryEntry(
                    id=memory_id,
                    content=memory.messages[-1].get("content", ""),
                    tier=MemoryTier.ARCHIVAL,
                    provider=MemoryProvider.ZEP,
                    metadata=memory.metadata or {},
                )
        except Exception as e:
            logger.warning("zep_get_failed", error=str(e))

        return None

    async def delete(self, memory_id: str) -> bool:
        """Delete a Zep session."""
        try:
            parts = memory_id.split(":")
            session_id = parts[0] if parts else memory_id

            await asyncio.to_thread(
                self.client.memory.delete_memory,
                session_id,
            )

            if session_id in self._sessions:
                del self._sessions[session_id]

            return True
        except Exception as e:
            logger.warning("zep_delete_failed", error=str(e))
            return False

    async def close(self) -> None:
        """Close Zep provider."""
        self._sessions.clear()


class LettaProvider(MemoryProviderBase):
    """Letta memory provider for agent-style memory management."""

    provider_type = MemoryProvider.LETTA

    def __init__(self, config: Optional[ProviderConfig] = None):
        if not LETTA_AVAILABLE:
            raise ImportError("letta not installed")

        self.config = config or ProviderConfig(provider=MemoryProvider.LETTA)

        # Initialize Letta client
        self.client = create_client()
        self._agents: dict[str, Any] = {}
        logger.info("letta_provider_initialized")

    async def _get_or_create_agent(self, agent_id: str) -> Any:
        """Get or create a Letta agent."""
        if agent_id not in self._agents:
            try:
                agent = self.client.get_agent(agent_id)
            except Exception:
                # Create new agent
                agent = self.client.create_agent(name=agent_id)
            self._agents[agent_id] = agent
        return self._agents[agent_id]

    async def store(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> MemoryEntry:
        """Store memory using Letta archival memory."""
        agent_id = user_id or "default"

        loop = asyncio.get_event_loop()

        def _store():
            try:
                agent = self._get_or_create_agent(agent_id)
                # Insert into archival memory
                passage = self.client.insert_archival_memory(
                    agent_id=agent.id,
                    memory=content,
                )
                return passage
            except Exception as e:
                logger.warning("letta_store_fallback", error=str(e))
                return None

        result = await loop.run_in_executor(None, _store)

        memory_id = result.id if result and hasattr(result, "id") else str(datetime.now().timestamp())

        return MemoryEntry(
            id=memory_id,
            content=content,
            tier=MemoryTier.ARCHIVAL,
            provider=MemoryProvider.LETTA,
            metadata=metadata or {},
        )

    async def search(
        self,
        query: str,
        limit: int = 10,
        user_id: Optional[str] = None,
    ) -> list[SearchResult]:
        """Search Letta archival memory."""
        agent_id = user_id or "default"

        loop = asyncio.get_event_loop()

        def _search():
            try:
                agent = self._get_or_create_agent(agent_id)
                results = self.client.get_archival_memory(
                    agent_id=agent.id,
                    limit=limit,
                )
                return results
            except Exception as e:
                logger.warning("letta_search_failed", error=str(e))
                return []

        results = await loop.run_in_executor(None, _search)

        # Filter by query relevance (simple keyword matching)
        query_lower = query.lower()
        filtered = [
            r for r in results
            if query_lower in (r.text if hasattr(r, "text") else str(r)).lower()
        ]

        return [
            SearchResult(
                id=r.id if hasattr(r, "id") else str(i),
                content=r.text if hasattr(r, "text") else str(r),
                score=1.0,  # Letta doesn't provide scores
                metadata={},
                provider=MemoryProvider.LETTA,
            )
            for i, r in enumerate(filtered[:limit])
        ]

    async def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get archival memory from Letta."""
        # Letta doesn't support direct ID lookup, return None
        return None

    async def delete(self, memory_id: str) -> bool:
        """Delete from Letta archival memory."""
        loop = asyncio.get_event_loop()

        def _delete():
            try:
                self.client.delete_archival_memory(memory_id)
                return True
            except Exception:
                return False

        return await loop.run_in_executor(None, _delete)

    async def close(self) -> None:
        """Close Letta provider."""
        self._agents.clear()


class CrossSessionProvider(MemoryProviderBase):
    """
    Cross-session memory provider for persistent memory across Claude Code sessions.

    This wraps the platform's CrossSessionMemory to provide:
    - File-based persistence that survives restarts
    - Session tracking and handoff
    - Temporal knowledge with decisions/learnings/facts
    - Context generation for new sessions
    """

    provider_type = MemoryProvider.CROSS_SESSION

    def __init__(self, config: Optional[ProviderConfig] = None):
        if not CROSS_SESSION_AVAILABLE:
            raise ImportError("cross_session_memory not available")

        self.config = config or ProviderConfig(provider=MemoryProvider.CROSS_SESSION)
        self._store = get_memory_store()
        logger.info("cross_session_provider_initialized", path=str(self._store.base_path))

    async def store(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> MemoryEntry:
        """Store memory using cross-session storage."""
        loop = asyncio.get_event_loop()

        # Determine memory type from metadata
        memory_type = "context"
        importance = 0.5
        tags = []

        if metadata:
            memory_type = metadata.get("type", metadata.get("memory_type", "context"))
            importance = metadata.get("importance", 0.5)
            tags = metadata.get("tags", [])

        def _store():
            memory = self._store.add(
                content=content,
                memory_type=memory_type,
                importance=importance,
                tags=tags,
                metadata=metadata or {},
            )
            return memory

        result = await loop.run_in_executor(None, _store)

        return MemoryEntry(
            id=result.id,
            content=result.content,
            tier=MemoryTier.ARCHIVAL,
            provider=MemoryProvider.CROSS_SESSION,
            metadata=metadata or {},
            importance=importance,
        )

    async def search(
        self,
        query: str,
        limit: int = 10,
        user_id: Optional[str] = None,
    ) -> list[SearchResult]:
        """Search cross-session memories."""
        loop = asyncio.get_event_loop()

        def _search():
            return self._store.search(query, limit=limit)

        results = await loop.run_in_executor(None, _search)

        return [
            SearchResult(
                id=r.id,
                content=r.content,
                score=r.importance,  # Use importance as score
                metadata={
                    "type": r.memory_type,
                    "tags": r.tags,
                    "created_at": r.created_at,
                    "session_id": r.session_id,
                },
                provider=MemoryProvider.CROSS_SESSION,
            )
            for r in results
        ]

    async def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory by ID."""
        if memory_id in self._store._memories:
            m = self._store._memories[memory_id]
            return MemoryEntry(
                id=m.id,
                content=m.content,
                tier=MemoryTier.ARCHIVAL,
                provider=MemoryProvider.CROSS_SESSION,
                metadata={
                    "type": m.memory_type,
                    "tags": m.tags,
                    "session_id": m.session_id,
                },
                importance=m.importance,
            )
        return None

    async def delete(self, memory_id: str) -> bool:
        """Invalidate a memory entry."""
        return self._store.invalidate(memory_id, reason="deleted via API")

    async def close(self) -> None:
        """Close cross-session provider (saves to disk)."""
        self._store._save()

    # Cross-session specific methods
    def start_session(self, task_summary: str = "") -> Any:
        """Start a new tracking session."""
        return self._store.start_session(task_summary)

    def end_session(self, summary: Optional[str] = None) -> None:
        """End the current session."""
        self._store.end_session(summary)

    def get_session_context(self, max_tokens: int = 4000) -> str:
        """Get context from previous sessions for handoff."""
        return self._store.get_session_context(max_tokens)

    def get_decisions(self, limit: int = 20) -> list[Any]:
        """Get recent decisions."""
        return self._store.get_decisions(limit)

    def get_learnings(self, limit: int = 20) -> list[Any]:
        """Get recent learnings."""
        return self._store.get_learnings(limit)

    def remember_decision(self, content: str, importance: float = 0.7, tags: list[str] = None) -> Any:
        """Remember a decision made."""
        return self._store.add(content, memory_type="decision", importance=importance, tags=tags or [])

    def remember_learning(self, content: str, importance: float = 0.6, tags: list[str] = None) -> Any:
        """Remember something learned."""
        return self._store.add(content, memory_type="learning", importance=importance, tags=tags or [])

    def remember_fact(self, content: str, importance: float = 0.5, tags: list[str] = None) -> Any:
        """Remember a fact."""
        return self._store.add(content, memory_type="fact", importance=importance, tags=tags or [])


def create_provider(
    provider_type: MemoryProvider,
    config: Optional[ProviderConfig] = None,
) -> MemoryProviderBase:
    """Factory function to create a memory provider."""
    providers = {
        MemoryProvider.MEM0: Mem0Provider,
        MemoryProvider.ZEP: ZepProvider,
        MemoryProvider.LETTA: LettaProvider,
        MemoryProvider.CROSS_SESSION: CrossSessionProvider,
    }

    if provider_type not in providers:
        raise ValueError(f"Unknown provider: {provider_type}")

    provider_class = providers[provider_type]
    return provider_class(config)


def get_available_providers() -> dict[MemoryProvider, bool]:
    """Get availability status of all providers."""
    return {
        MemoryProvider.LOCAL: True,  # Always available
        MemoryProvider.MEM0: MEM0_AVAILABLE,
        MemoryProvider.ZEP: ZEP_AVAILABLE,
        MemoryProvider.LETTA: LETTA_AVAILABLE,
        MemoryProvider.CROSS_SESSION: CROSS_SESSION_AVAILABLE,
    }
