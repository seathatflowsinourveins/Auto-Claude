"""
Letta Backend - V36 Architecture

Letta-based storage for core and archival memory tiers.
Integrates with Letta Cloud or self-hosted Letta server.

Uses Letta SDK 1.7.6+ API patterns:
- Core memory (blocks): client.agents.blocks.*
- Archival memory (passages): client.agents.passages.*

Usage:
    from core.memory.backends.letta import LettaTierBackend

    backend = LettaTierBackend(
        tier=MemoryTier.ARCHIVAL_MEMORY,
        agent_id="agent-xxxx",
        api_key=os.environ["LETTA_API_KEY"]
    )
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

from .base import (
    MemoryEntry,
    MemoryLayer,
    MemoryNamespace,
    MemoryPriority,
    MemoryTier,
    TierBackend,
    MemoryBackend,
)

logger = logging.getLogger(__name__)


class LettaTierBackend(TierBackend[MemoryEntry]):
    """
    Letta-based storage for core and archival memory.

    Uses Letta SDK 1.7.6+ API (verified patterns):
    - Core memory (blocks):
      - client.agents.blocks.list(agent_id)
      - client.agents.blocks.retrieve(block_label, agent_id=...)
      - client.agents.blocks.update(block_label, agent_id=..., value=...)
      - client.agents.blocks.attach(block_id, agent_id=...)
      - client.agents.blocks.detach(block_id, agent_id=...)
    - Archival memory (passages):
      - client.agents.passages.create(agent_id, text=..., tags=...)
      - client.agents.passages.search(agent_id, query=..., top_k=...)
      - client.agents.passages.delete(agent_id, memory_id=...)

    V36 Consolidated: Unified from memory_tiers.py and unified_memory_gateway.py.
    """

    def __init__(
        self,
        tier: MemoryTier,
        agent_id: str,
        letta_client: Optional[Any] = None,
        base_url: str = "https://api.letta.com",
        api_key: Optional[str] = None
    ) -> None:
        """Initialize Letta tier backend.

        Args:
            tier: Memory tier (CORE_MEMORY, RECALL_MEMORY, ARCHIVAL)
            agent_id: Letta agent ID
            letta_client: Optional pre-configured Letta client
            base_url: Letta API URL (Cloud: https://api.letta.com, Local: http://localhost:8283)
            api_key: API key for Letta Cloud (reads from LETTA_API_KEY env var if not provided)
        """
        self.tier = tier
        self.agent_id = agent_id
        self.base_url = base_url or os.environ.get("LETTA_BASE_URL", "https://api.letta.com")
        self.api_key = api_key or os.environ.get("LETTA_API_KEY")
        self._client = letta_client
        self._local_cache: Dict[str, MemoryEntry] = {}

    def _get_client(self) -> Any:
        """Get or create Letta client with connection pooling."""
        if self._client is None:
            try:
                from letta_client import Letta
                import httpx

                # Cloud connection requires api_key
                if self.api_key:
                    self._client = Letta(
                        api_key=self.api_key,
                        base_url=self.base_url,
                        http_client=httpx.Client(
                            limits=httpx.Limits(
                                max_connections=50,
                                max_keepalive_connections=10
                            ),
                            timeout=httpx.Timeout(30.0)
                        )
                    )
                else:
                    # Self-hosted (local) connection
                    self._client = Letta(
                        base_url=self.base_url,
                        http_client=httpx.Client(
                            limits=httpx.Limits(
                                max_connections=50,
                                max_keepalive_connections=10
                            ),
                            timeout=httpx.Timeout(30.0)
                        )
                    )
            except ImportError:
                logger.warning("Letta SDK not installed - install with: pip install letta-client")
                return None
        return self._client

    async def get(self, key: str) -> Optional[MemoryEntry]:
        """Get memory entry by key."""
        # Check local cache first
        if key in self._local_cache:
            entry = self._local_cache[key]
            entry.touch()
            return entry

        # Try Letta API
        client = self._get_client()
        if client is None:
            return None

        try:
            if self.tier == MemoryTier.CORE_MEMORY:
                # Core memory: get by block label using blocks.retrieve()
                block = client.agents.blocks.retrieve(key, agent_id=self.agent_id)
                if block:
                    entry = MemoryEntry(
                        id=key,
                        content=getattr(block, 'value', str(block)),
                        tier=self.tier,
                        block_label=getattr(block, 'label', key),
                        priority=MemoryPriority.HIGH
                    )
                    self._local_cache[key] = entry
                    return entry
            else:
                # Archival: search using passages.search()
                results = client.agents.passages.search(
                    agent_id=self.agent_id,
                    query=f"id:{key}",
                    top_k=1
                )
                passages = getattr(results, 'results', getattr(results, 'passages', results)) or []
                if passages:
                    passage = passages[0] if hasattr(passages, '__getitem__') else next(iter(passages), None)
                    if passage:
                        entry = MemoryEntry(
                            id=key,
                            content=getattr(passage, 'content', getattr(passage, 'text', str(passage))),
                            tier=self.tier,
                            embedding=getattr(passage, 'embedding', None)
                        )
                        self._local_cache[key] = entry
                        return entry
        except Exception as e:
            logger.error(f"Letta get failed for key '{key}': {e}")

        return None

    async def put(self, key: str, value: MemoryEntry) -> None:
        """Store memory entry."""
        self._local_cache[key] = value

        client = self._get_client()
        if client is None:
            return

        try:
            if self.tier == MemoryTier.CORE_MEMORY:
                if value.block_label:
                    # Update existing block
                    client.agents.blocks.update(
                        value.block_label,
                        agent_id=self.agent_id,
                        value=value.content
                    )
                else:
                    # Create new block and attach
                    block = client.blocks.create(
                        label=key,
                        value=value.content
                    )
                    client.agents.blocks.attach(
                        block.id,
                        agent_id=self.agent_id
                    )
            else:
                # Archival memory: create passage
                tags = list(value.tags) if value.tags else []
                if key:
                    tags.append(f"id:{key}")
                client.agents.passages.create(
                    agent_id=self.agent_id,
                    text=value.content,
                    tags=tags if tags else None
                )
        except Exception as e:
            logger.error(f"Letta put failed for key '{key}': {e}")

    async def delete(self, key: str) -> bool:
        """Delete memory entry by key."""
        if key in self._local_cache:
            del self._local_cache[key]

        client = self._get_client()
        if client is None:
            return True

        try:
            if self.tier == MemoryTier.CORE_MEMORY:
                # Detach block
                try:
                    block = client.agents.blocks.retrieve(key, agent_id=self.agent_id)
                    if block and hasattr(block, 'id'):
                        client.agents.blocks.detach(block.id, agent_id=self.agent_id)
                except Exception:
                    logger.warning(f"Block '{key}' not found for detach")
            else:
                # Archival: delete passage by ID
                try:
                    client.agents.passages.delete(
                        agent_id=self.agent_id,
                        memory_id=key
                    )
                except Exception:
                    logger.warning(f"Passage '{key}' delete may not be supported")
            return True
        except Exception as e:
            logger.error(f"Letta delete failed for key '{key}': {e}")
            return False

    async def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Search memory tier for matching entries."""
        results: List[MemoryEntry] = []

        # Search local cache first
        query_lower = query.lower()
        for entry in self._local_cache.values():
            if query_lower in entry.content.lower():
                results.append(entry)

        client = self._get_client()
        if client is None:
            return results[:limit]

        try:
            if self.tier == MemoryTier.CORE_MEMORY:
                # Search core blocks
                blocks = client.agents.blocks.list(self.agent_id)
                for block in blocks:
                    block_value = getattr(block, 'value', '')
                    if block_value and query_lower in block_value.lower():
                        entry = MemoryEntry(
                            id=getattr(block, 'label', str(block)),
                            content=block_value,
                            tier=self.tier,
                            block_label=getattr(block, 'label', None),
                            priority=MemoryPriority.HIGH
                        )
                        if entry.id not in [r.id for r in results]:
                            results.append(entry)
            else:
                # Archival: semantic search
                search_results = client.agents.passages.search(
                    agent_id=self.agent_id,
                    query=query,
                    top_k=limit
                )
                passages = getattr(search_results, 'results', getattr(search_results, 'passages', search_results)) or []
                for passage in passages:
                    passage_text = getattr(passage, 'content', getattr(passage, 'text', str(passage)))
                    passage_id = getattr(passage, 'id', None) or hashlib.md5(passage_text.encode()).hexdigest()[:8]
                    entry = MemoryEntry(
                        id=passage_id,
                        content=passage_text,
                        tier=self.tier,
                        embedding=getattr(passage, 'embedding', None)
                    )
                    if entry.id not in [r.id for r in results]:
                        results.append(entry)
        except Exception as e:
            logger.error(f"Letta search failed: {e}")

        return results[:limit]

    async def list_all(self) -> List[MemoryEntry]:
        """List all entries from local cache."""
        return list(self._local_cache.values())

    async def count(self) -> int:
        """Get entry count from local cache."""
        return len(self._local_cache)


class LettaMemoryBackend(MemoryBackend):
    """
    Letta agent memory backend for the 5-layer gateway.

    Connects to Letta server for project-specific conversational memory.
    Each project has its own agent with persistent memory blocks.

    V36 Consolidated: From unified_memory_gateway.py LettaMemoryBackend.
    """

    # Project-specific Letta agent IDs
    AGENT_REGISTRY: Dict[str, Dict[str, str]] = {
        "unleash": {
            "id": "agent-daee71d2-193b-485e-bda4-ee44752635fe",
            "name": "claude-code-ecosystem-test"
        },
        "witness": {
            "id": "agent-bbcc0a74-5ff8-4ccd-83bc-b7282c952589",
            "name": "state-of-witness-creative-brain"
        },
        "trading": {
            "id": "agent-5676da61-c57c-426e-a0f6-390fd9dfcf94",
            "name": "alphaforge-dev-orchestrator"
        }
    }

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        project: str = "unleash"
    ):
        """Initialize Letta memory backend.

        Args:
            base_url: Letta API URL. Use "https://api.letta.com" for Cloud,
                     "http://localhost:8283" for self-hosted.
            api_key: API key for Letta Cloud (required for Cloud, optional for local).
            project: Project name to select agent from registry.
        """
        self.base_url = base_url or os.environ.get("LETTA_BASE_URL", "https://api.letta.com")
        self.api_key = api_key or os.environ.get("LETTA_API_KEY")
        self.project = project
        self.agent_info = self.AGENT_REGISTRY.get(project, self.AGENT_REGISTRY["unleash"])
        self._client = None

    @property
    def layer(self) -> MemoryLayer:
        return MemoryLayer.LETTA

    async def _get_client(self):
        """Get or create Letta client with connection pooling."""
        if self._client is None:
            try:
                from letta_client import Letta
                import httpx

                if self.api_key:
                    self._client = Letta(
                        api_key=self.api_key,
                        base_url=self.base_url,
                        http_client=httpx.Client(
                            limits=httpx.Limits(
                                max_connections=100,
                                max_keepalive_connections=20
                            ),
                            timeout=httpx.Timeout(30.0)
                        )
                    )
                else:
                    self._client = Letta(
                        base_url=self.base_url,
                        http_client=httpx.Client(
                            limits=httpx.Limits(
                                max_connections=50,
                                max_keepalive_connections=10
                            ),
                            timeout=httpx.Timeout(30.0)
                        )
                    )
            except ImportError:
                logger.warning("Letta client not available - install with: pip install letta-client")
                return None
        return self._client

    async def search(
        self,
        query: str,
        max_results: int = 10,
        namespace: Optional[MemoryNamespace] = None,
        tags: Optional[List[str]] = None,
        start_datetime: Optional[str] = None,
        end_datetime: Optional[str] = None
    ) -> List[MemoryEntry]:
        """Search Letta agent archival memory (passages)."""
        client = await self._get_client()
        if not client:
            return []

        try:
            # Build search parameters with temporal and tag filtering
            search_params = {
                "agent_id": self.agent_info["id"],
                "query": query,
                "top_k": max_results
            }

            if start_datetime:
                search_params["start_datetime"] = start_datetime
            if end_datetime:
                search_params["end_datetime"] = end_datetime
            if tags:
                search_params["tags"] = tags
                search_params["tag_match_mode"] = "any"

            # Use Letta's passages search (archival memory)
            search_results = client.agents.passages.search(**search_params)

            entries = []
            passages = getattr(search_results, 'results', getattr(search_results, 'passages', search_results)) or []
            for i, passage in enumerate(passages):
                content = getattr(passage, 'content', getattr(passage, 'text', str(passage)))
                score = getattr(passage, 'score', 1.0 - (i * 0.1))
                entries.append(MemoryEntry(
                    id=f"letta_{self.project}_{i}",
                    content=content,
                    tier=MemoryTier.ARCHIVAL_MEMORY,
                    namespace=namespace or MemoryNamespace.CONTEXT,
                    metadata={
                        "agent_id": self.agent_info["id"],
                        "project": self.project,
                        "passage_id": getattr(passage, 'id', None),
                        "relevance_score": float(score)
                    }
                ))

            return entries

        except Exception as e:
            logger.error("Letta search failed: %s", e)
            return []

    async def store(
        self,
        content: str,
        namespace: MemoryNamespace,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store memory by creating a passage in the Letta agent."""
        client = await self._get_client()
        if not client:
            return ""

        try:
            # Create passage with namespace tag
            tags = [namespace.value]
            if metadata and "tags" in metadata:
                tags.extend(metadata["tags"])

            client.agents.passages.create(
                self.agent_info["id"],
                text=content,
                tags=tags
            )

            # Return generated ID
            return hashlib.md5(f"{content[:50]}:{namespace.value}".encode()).hexdigest()[:12]

        except Exception as e:
            logger.error(f"Letta store failed: {e}")
            return ""

    async def health_check(self) -> bool:
        """Check if Letta backend is healthy."""
        client = await self._get_client()
        return client is not None


__all__ = [
    "LettaTierBackend",
    "LettaMemoryBackend",
]
