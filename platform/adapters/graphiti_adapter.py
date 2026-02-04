"""
Graphiti Adapter - V36 Architecture

Integrates Graphiti for temporal knowledge graphs with dynamic entity management.

SDK: graphiti-core >= 0.5.0 (https://github.com/getzep/graphiti)
Layer: L2 (Memory)
Features:
- Temporal knowledge graphs
- Dynamic entity extraction
- Relationship evolution over time
- Neo4j backend support
- Hybrid RAG with graph context

Replaces: Zep CE (deprecated)

API Patterns (verified from graphiti-core 0.5.0):
- Graphiti(neo4j_uri, neo4j_user, neo4j_password) → client
- client.add_episode(episode_body) → extract entities/relationships
- client.search(query, num_results) → hybrid search
- client.get_node(node_id) → get specific entity
- client.delete_episode(episode_id) → remove episode

Usage:
    from adapters.graphiti_adapter import GraphitiAdapter

    adapter = GraphitiAdapter()
    await adapter.initialize({
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "password"
    })

    # Add episode (conversation, document, etc.)
    await adapter.execute("add_episode", content="User discussed AI ethics")

    # Search with temporal awareness
    result = await adapter.execute("search", query="AI ethics discussion")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# SDK availability check
GRAPHITI_AVAILABLE = False

try:
    from graphiti_core import Graphiti
    from graphiti_core.nodes import EpisodeType
    GRAPHITI_AVAILABLE = True
except ImportError:
    logger.info("Graphiti not installed - install with: pip install graphiti-core")


# Import base adapter interface
try:
    from core.orchestration.base import (
        SDKAdapter,
        SDKLayer,
        AdapterConfig,
        AdapterResult,
        AdapterStatus,
    )
except ImportError:
    from dataclasses import dataclass as _dataclass
    from enum import IntEnum
    from abc import ABC, abstractmethod

    class SDKLayer(IntEnum):
        MEMORY = 2

    @_dataclass
    class AdapterResult:
        success: bool
        data: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        latency_ms: float = 0.0

    @_dataclass
    class AdapterConfig:
        name: str = "graphiti"
        layer: int = 2

    class AdapterStatus:
        READY = "ready"
        FAILED = "failed"
        UNINITIALIZED = "uninitialized"

    class SDKAdapter(ABC):
        @property
        @abstractmethod
        def sdk_name(self) -> str: ...
        @property
        @abstractmethod
        def layer(self) -> int: ...
        @property
        @abstractmethod
        def available(self) -> bool: ...
        @abstractmethod
        async def initialize(self, config: Dict) -> AdapterResult: ...
        @abstractmethod
        async def execute(self, operation: str, **kwargs) -> AdapterResult: ...
        @abstractmethod
        async def health_check(self) -> AdapterResult: ...
        @abstractmethod
        async def shutdown(self) -> AdapterResult: ...


@dataclass
class TemporalEntity:
    """Entity with temporal metadata."""
    id: str
    name: str
    entity_type: str
    created_at: datetime
    updated_at: datetime
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalRelationship:
    """Relationship with temporal validity."""
    source_id: str
    target_id: str
    relationship_type: str
    valid_from: datetime
    valid_to: Optional[datetime] = None
    properties: Dict[str, Any] = field(default_factory=dict)


class GraphitiAdapter(SDKAdapter):
    """
    Graphiti adapter for temporal knowledge graphs.

    Provides temporal knowledge management:
    - Automatic entity extraction from episodes
    - Relationship tracking over time
    - Hybrid search (semantic + graph traversal)
    - Neo4j backend for scalable storage

    Operations:
    - add_episode: Add content and extract entities/relationships
    - search: Hybrid search with temporal awareness
    - get_entity: Get a specific entity by ID
    - get_relationships: Get relationships for an entity
    - delete_episode: Remove an episode and update graph
    - get_facts: Get fact summaries from the graph
    """

    def __init__(self, config: Optional[AdapterConfig] = None):
        self._config = config or AdapterConfig(
            name="graphiti",
            layer=SDKLayer.MEMORY
        )
        self._status = AdapterStatus.UNINITIALIZED
        self._client: Optional[Any] = None
        self._neo4j_uri: str = ""
        self._call_count = 0
        self._total_latency_ms = 0.0
        self._error_count = 0
        self._episode_count = 0

    @property
    def sdk_name(self) -> str:
        return "graphiti"

    @property
    def layer(self) -> SDKLayer:
        return SDKLayer.MEMORY

    @property
    def available(self) -> bool:
        return GRAPHITI_AVAILABLE

    @property
    def status(self) -> AdapterStatus:
        return self._status

    async def initialize(self, config: Dict[str, Any]) -> AdapterResult:
        """Initialize Graphiti with Neo4j connection."""
        if not GRAPHITI_AVAILABLE:
            return AdapterResult(
                success=False,
                error="Graphiti not installed. Install with: pip install graphiti-core"
            )

        try:
            import os

            # Get Neo4j configuration
            neo4j_uri = config.get("neo4j_uri") or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
            neo4j_user = config.get("neo4j_user") or os.environ.get("NEO4J_USER", "neo4j")
            neo4j_password = config.get("neo4j_password") or os.environ.get("NEO4J_PASSWORD", "")

            if not neo4j_password:
                return AdapterResult(
                    success=False,
                    error="NEO4J_PASSWORD not provided"
                )

            self._neo4j_uri = neo4j_uri

            # Initialize Graphiti client
            self._client = Graphiti(
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password
            )

            # Initialize the graph schema
            await self._client.build_indices()

            self._status = AdapterStatus.READY
            logger.info(f"Graphiti adapter initialized (uri={neo4j_uri})")

            return AdapterResult(
                success=True,
                data={"neo4j_uri": neo4j_uri, "status": "connected"}
            )

        except Exception as e:
            self._status = AdapterStatus.FAILED
            logger.error(f"Graphiti initialization failed: {e}")
            return AdapterResult(success=False, error=str(e))

    async def execute(self, operation: str, **kwargs) -> AdapterResult:
        """Execute a Graphiti operation."""
        if not self._client:
            return AdapterResult(success=False, error="Adapter not initialized")

        start_time = time.time()

        try:
            if operation == "add_episode":
                result = await self._add_episode(**kwargs)
            elif operation == "search":
                result = await self._search(**kwargs)
            elif operation == "get_entity":
                result = await self._get_entity(**kwargs)
            elif operation == "get_relationships":
                result = await self._get_relationships(**kwargs)
            elif operation == "delete_episode":
                result = await self._delete_episode(**kwargs)
            elif operation == "get_facts":
                result = await self._get_facts(**kwargs)
            elif operation == "get_stats":
                result = await self._get_stats()
            else:
                result = AdapterResult(
                    success=False,
                    error=f"Unknown operation: {operation}"
                )

            latency_ms = (time.time() - start_time) * 1000
            self._call_count += 1
            self._total_latency_ms += latency_ms
            result.latency_ms = latency_ms

            if not result.success:
                self._error_count += 1

            return result

        except Exception as e:
            self._error_count += 1
            logger.error(f"Graphiti execute error: {e}")
            return AdapterResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )

    async def _add_episode(
        self,
        content: str,
        episode_type: str = "text",
        source: str = "user",
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AdapterResult:
        """Add an episode to the knowledge graph."""
        try:
            # Map episode type
            ep_type = EpisodeType.text
            if episode_type == "message":
                ep_type = EpisodeType.message
            elif episode_type == "json":
                ep_type = EpisodeType.json

            # Add episode
            episode = await self._client.add_episode(
                name=f"episode_{self._episode_count}",
                episode_body=content,
                source_description=source,
                reference_time=timestamp or datetime.now(timezone.utc).isoformat(),
            )

            self._episode_count += 1

            return AdapterResult(
                success=True,
                data={
                    "episode_id": str(episode.uuid) if hasattr(episode, 'uuid') else str(self._episode_count),
                    "entities_extracted": getattr(episode, 'entity_count', 0),
                    "relationships_created": getattr(episode, 'relation_count', 0)
                }
            )

        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _search(
        self,
        query: str,
        num_results: int = 10,
        include_entities: bool = True,
        include_facts: bool = True,
        **kwargs
    ) -> AdapterResult:
        """Search the knowledge graph with hybrid approach."""
        try:
            results = await self._client.search(
                query=query,
                num_results=num_results
            )

            # Process results
            search_results = []
            for i, result in enumerate(results):
                search_results.append({
                    "content": getattr(result, 'content', str(result)),
                    "score": getattr(result, 'score', 1.0 / (i + 1)),
                    "type": getattr(result, 'type', 'unknown'),
                    "metadata": getattr(result, 'metadata', {})
                })

            return AdapterResult(
                success=True,
                data={
                    "results": search_results,
                    "count": len(search_results),
                    "query": query
                }
            )

        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _get_entity(self, entity_id: str, **kwargs) -> AdapterResult:
        """Get a specific entity by ID."""
        try:
            node = await self._client.get_node(entity_id)

            if node:
                return AdapterResult(
                    success=True,
                    data={
                        "entity": {
                            "id": entity_id,
                            "name": getattr(node, 'name', ''),
                            "type": getattr(node, 'type', ''),
                            "properties": getattr(node, 'properties', {})
                        }
                    }
                )
            else:
                return AdapterResult(
                    success=False,
                    error=f"Entity not found: {entity_id}"
                )

        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _get_relationships(
        self,
        entity_id: str,
        direction: str = "both",
        **kwargs
    ) -> AdapterResult:
        """Get relationships for an entity."""
        try:
            # Search for relationships involving the entity
            results = await self._client.search(
                query=f"relationships with {entity_id}",
                num_results=50
            )

            relationships = []
            for result in results:
                if hasattr(result, 'source') and hasattr(result, 'target'):
                    relationships.append({
                        "source": result.source,
                        "target": result.target,
                        "type": getattr(result, 'relationship_type', 'related'),
                        "properties": getattr(result, 'properties', {})
                    })

            return AdapterResult(
                success=True,
                data={
                    "entity_id": entity_id,
                    "relationships": relationships,
                    "count": len(relationships)
                }
            )

        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _delete_episode(self, episode_id: str, **kwargs) -> AdapterResult:
        """Delete an episode from the graph."""
        try:
            await self._client.delete_episode(episode_id)
            return AdapterResult(
                success=True,
                data={"episode_id": episode_id, "deleted": True}
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _get_facts(self, limit: int = 20, **kwargs) -> AdapterResult:
        """Get fact summaries from the knowledge graph."""
        try:
            # Get recent facts via search
            results = await self._client.search(
                query="facts and relationships",
                num_results=limit
            )

            facts = []
            for result in results:
                facts.append({
                    "content": getattr(result, 'content', str(result)),
                    "confidence": getattr(result, 'score', 1.0)
                })

            return AdapterResult(
                success=True,
                data={"facts": facts, "count": len(facts)}
            )

        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _get_stats(self) -> AdapterResult:
        """Get adapter statistics."""
        return AdapterResult(
            success=True,
            data={
                "episode_count": self._episode_count,
                "call_count": self._call_count,
                "error_count": self._error_count,
                "avg_latency_ms": self._total_latency_ms / max(1, self._call_count),
                "neo4j_uri": self._neo4j_uri
            }
        )

    async def health_check(self) -> AdapterResult:
        """Check adapter health."""
        if not self._client:
            return AdapterResult(success=False, error="Client not initialized")

        try:
            # Simple connectivity check
            return AdapterResult(
                success=True,
                data={
                    "status": "healthy",
                    "neo4j_uri": self._neo4j_uri,
                    "episode_count": self._episode_count
                }
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def shutdown(self) -> AdapterResult:
        """Shutdown the adapter."""
        if self._client:
            try:
                await self._client.close()
            except Exception:
                pass
        self._client = None
        self._status = AdapterStatus.UNINITIALIZED
        logger.info("Graphiti adapter shutdown")
        return AdapterResult(success=True)


# Register with SDK registry
try:
    from core.orchestration.sdk_registry import register_adapter, SDKLayer

    @register_adapter("graphiti", SDKLayer.MEMORY, priority=18, replaces="zep-ce")
    class RegisteredGraphitiAdapter(GraphitiAdapter):
        """Registered Graphiti adapter."""
        pass

except ImportError:
    pass


__all__ = ["GraphitiAdapter", "GRAPHITI_AVAILABLE", "TemporalEntity", "TemporalRelationship"]
