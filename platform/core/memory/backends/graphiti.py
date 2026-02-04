"""
Graphiti Backend - V36 Architecture (Placeholder)

Graphiti-based temporal knowledge graph backend for entity relationships and episodic memory.
Integrates with Graphiti/Zep Cloud for advanced knowledge graph capabilities.

Graphiti Features (when implemented):
- Temporal knowledge graphs with bi-temporal support
- Episodic memory with automatic fact extraction
- Entity and relationship extraction
- Semantic search over graph structure
- Time-aware queries (point-in-time, as-of)

References:
- Graphiti: https://github.com/getzep/graphiti
- Zep Cloud: https://www.getzep.com/

TODO: This is a placeholder implementation. Real implementation requires:
1. graphiti-core package installation: pip install graphiti-core
2. Zep Cloud API credentials
3. Neo4j database connection (Graphiti uses Neo4j)
4. LLM provider for entity extraction (OpenAI, Anthropic)

Usage (future):
    from core.memory.backends.graphiti import GraphitiBackend

    backend = GraphitiBackend(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="...",
        zep_api_key=os.environ["ZEP_API_KEY"]
    )

    # Add episode (conversation turn)
    episode_id = await backend.add_episode(
        content="User asked about authentication patterns",
        source="conversation",
        source_description="Chat session 123",
    )

    # Search with temporal awareness
    results = await backend.search(
        query="authentication",
        num_results=10,
        center_node_uuid=None  # Optional: focus search around a node
    )
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    MemoryBackend,
    MemoryEntry,
    MemoryLayer,
    MemoryNamespace,
    MemoryPriority,
    MemoryTier,
    TierBackend,
    generate_memory_id,
)

logger = logging.getLogger(__name__)


# =============================================================================
# GRAPHITI DATA TYPES
# =============================================================================

class EntityType(str, Enum):
    """Types of entities that can be extracted from episodes."""
    PERSON = "person"
    ORGANIZATION = "organization"
    CONCEPT = "concept"
    TOOL = "tool"
    TECHNOLOGY = "technology"
    LOCATION = "location"
    EVENT = "event"
    CUSTOM = "custom"


class RelationType(str, Enum):
    """Types of relationships between entities."""
    USES = "uses"
    KNOWS = "knows"
    RELATES_TO = "relates_to"
    DEPENDS_ON = "depends_on"
    CREATED_BY = "created_by"
    PART_OF = "part_of"
    SIMILAR_TO = "similar_to"
    TEMPORAL = "temporal"  # Time-based relationship
    CAUSAL = "causal"      # Cause-effect relationship


class EpisodeType(str, Enum):
    """Types of episodes (conversational units)."""
    MESSAGE = "message"        # Single message
    CONVERSATION = "conversation"  # Full conversation
    DOCUMENT = "document"      # Document ingestion
    OBSERVATION = "observation"  # System observation
    REFLECTION = "reflection"   # Agent reflection


@dataclass
class GraphitiEntity:
    """
    An entity in the knowledge graph.

    Entities represent nodes (persons, concepts, tools, etc.)
    that are extracted from episodes and linked via relationships.
    """
    id: str
    name: str
    entity_type: EntityType
    summary: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Temporal validity
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None

    # Metadata
    properties: Dict[str, Any] = field(default_factory=dict)
    source_episode_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "summary": self.summary,
            "created_at": self.created_at.isoformat(),
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
            "properties": self.properties,
            "source_episode_ids": self.source_episode_ids,
        }


@dataclass
class GraphitiRelation:
    """
    A relationship between two entities in the knowledge graph.

    Relationships are edges connecting entities, with optional
    temporal validity and extracted facts.
    """
    id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    fact: str = ""  # The extracted fact describing this relationship
    weight: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Temporal validity
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None

    # Episode tracking
    episode_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "fact": self.fact,
            "weight": self.weight,
            "created_at": self.created_at.isoformat(),
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
            "episode_ids": self.episode_ids,
        }


@dataclass
class GraphitiEpisode:
    """
    An episode in the temporal knowledge graph.

    Episodes represent units of information (messages, documents)
    that are processed to extract entities and relationships.
    """
    id: str
    content: str
    episode_type: EpisodeType
    source: str
    source_description: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Reference timestamp (when the event happened, not when ingested)
    reference_time: Optional[datetime] = None

    # Extraction results
    extracted_entities: List[str] = field(default_factory=list)  # Entity IDs
    extracted_relations: List[str] = field(default_factory=list)  # Relation IDs

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "episode_type": self.episode_type.value,
            "source": self.source,
            "source_description": self.source_description,
            "created_at": self.created_at.isoformat(),
            "reference_time": self.reference_time.isoformat() if self.reference_time else None,
            "extracted_entities": self.extracted_entities,
            "extracted_relations": self.extracted_relations,
            "metadata": self.metadata,
        }


@dataclass
class GraphitiSearchResult:
    """Result from a Graphiti search operation."""
    uuid: str
    content: str
    score: float
    entity: Optional[GraphitiEntity] = None
    relation: Optional[GraphitiRelation] = None
    episode: Optional[GraphitiEpisode] = None

    # Temporal context
    valid_at: Optional[datetime] = None
    temporal_distance: Optional[float] = None  # How far from query time


# =============================================================================
# GRAPHITI TIER BACKEND (Placeholder)
# =============================================================================

class GraphitiTierBackend(TierBackend[MemoryEntry]):
    """
    Graphiti-based storage for temporal knowledge graphs.

    This is a PLACEHOLDER implementation that stores data in memory.
    Real implementation requires:
    - Neo4j database connection
    - Graphiti SDK (graphiti-core)
    - LLM for entity/relationship extraction

    TODO: Implement full Graphiti integration when available.

    Planned Graphiti SDK usage pattern:
        from graphiti_core import Graphiti
        from graphiti_core.nodes import EpisodeType

        graphiti = Graphiti(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
        )

        # Add episode
        await graphiti.add_episode(
            name="conversation_123",
            episode_body=content,
            source=EpisodeType.message,
            source_description="Chat session",
            reference_time=datetime.now(),
        )

        # Search
        results = await graphiti.search(
            query=query,
            num_results=10,
        )
    """

    def __init__(
        self,
        tier: MemoryTier = MemoryTier.ARCHIVAL_MEMORY,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        zep_api_key: Optional[str] = None,
        llm_provider: str = "openai",
    ) -> None:
        """
        Initialize Graphiti tier backend.

        Args:
            tier: Memory tier (usually ARCHIVAL_MEMORY for knowledge graphs)
            neo4j_uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            zep_api_key: Zep Cloud API key (alternative to self-hosted)
            llm_provider: LLM provider for extraction ("openai", "anthropic")

        TODO: Initialize real Graphiti client when implemented:
            self._graphiti = Graphiti(
                neo4j_uri=neo4j_uri or os.environ.get("NEO4J_URI"),
                neo4j_user=neo4j_user or os.environ.get("NEO4J_USER"),
                neo4j_password=neo4j_password or os.environ.get("NEO4J_PASSWORD"),
            )
        """
        self.tier = tier
        self.neo4j_uri = neo4j_uri or os.environ.get("NEO4J_URI")
        self.neo4j_user = neo4j_user or os.environ.get("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.environ.get("NEO4J_PASSWORD")
        self.zep_api_key = zep_api_key or os.environ.get("ZEP_API_KEY")
        self.llm_provider = llm_provider

        # Placeholder in-memory storage
        # TODO: Remove when real Graphiti integration is implemented
        self._local_cache: Dict[str, MemoryEntry] = {}
        self._episodes: Dict[str, GraphitiEpisode] = {}
        self._entities: Dict[str, GraphitiEntity] = {}
        self._relations: Dict[str, GraphitiRelation] = {}

        self._graphiti_client = None  # TODO: Real Graphiti client

        logger.info(
            f"GraphitiTierBackend initialized (PLACEHOLDER MODE) - "
            f"Neo4j URI: {self.neo4j_uri or 'Not configured'}"
        )

    def _get_client(self) -> Optional[Any]:
        """
        Get or create Graphiti client.

        TODO: Implement real client initialization:
            if self._graphiti_client is None:
                try:
                    from graphiti_core import Graphiti

                    self._graphiti_client = Graphiti(
                        neo4j_uri=self.neo4j_uri,
                        neo4j_user=self.neo4j_user,
                        neo4j_password=self.neo4j_password,
                    )
                except ImportError:
                    logger.warning(
                        "Graphiti SDK not installed - install with: "
                        "pip install graphiti-core"
                    )
                    return None
            return self._graphiti_client
        """
        # PLACEHOLDER: No real client yet
        if self._graphiti_client is None:
            logger.debug(
                "Graphiti client not available - using placeholder storage. "
                "Install graphiti-core for full functionality."
            )
        return self._graphiti_client

    async def get(self, key: str) -> Optional[MemoryEntry]:
        """Get memory entry by key."""
        # Check local cache first
        if key in self._local_cache:
            entry = self._local_cache[key]
            entry.touch()
            return entry

        # TODO: Query Graphiti graph database
        # client = self._get_client()
        # if client:
        #     node = await client.get_node(key)
        #     if node:
        #         return self._node_to_entry(node)

        return None

    async def put(self, key: str, value: MemoryEntry) -> None:
        """Store memory entry."""
        self._local_cache[key] = value

        # TODO: Store in Graphiti as episode or entity
        # client = self._get_client()
        # if client:
        #     await client.add_episode(
        #         name=key,
        #         episode_body=value.content,
        #         source=EpisodeType.message,
        #         reference_time=value.created_at,
        #     )

    async def delete(self, key: str) -> bool:
        """Delete memory entry by key."""
        deleted = False

        if key in self._local_cache:
            del self._local_cache[key]
            deleted = True

        if key in self._episodes:
            del self._episodes[key]
            deleted = True

        # TODO: Delete from Graphiti
        # client = self._get_client()
        # if client:
        #     await client.delete_node(key)

        return deleted

    async def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """
        Search memory tier for matching entries.

        TODO: Use Graphiti's semantic search:
            client = self._get_client()
            if client:
                results = await client.search(
                    query=query,
                    num_results=limit,
                )
                return [self._result_to_entry(r) for r in results]
        """
        results: List[MemoryEntry] = []
        query_lower = query.lower()

        # PLACEHOLDER: Simple text search over cached entries
        for entry in self._local_cache.values():
            if query_lower in entry.content.lower():
                results.append(entry)
                if len(results) >= limit:
                    break

        # Also search episodes
        for episode in self._episodes.values():
            if query_lower in episode.content.lower():
                entry = self._episode_to_entry(episode)
                if entry.id not in [r.id for r in results]:
                    results.append(entry)
                    if len(results) >= limit:
                        break

        return results[:limit]

    async def list_all(self) -> List[MemoryEntry]:
        """List all entries."""
        entries = list(self._local_cache.values())

        # Include episodes as entries
        for episode in self._episodes.values():
            entry = self._episode_to_entry(episode)
            if entry.id not in [e.id for e in entries]:
                entries.append(entry)

        return entries

    async def count(self) -> int:
        """Get entry count."""
        return len(self._local_cache) + len(self._episodes)

    # =========================================================================
    # EPISODE MANAGEMENT (Graphiti-specific)
    # =========================================================================

    async def add_episode(
        self,
        content: str,
        source: str,
        source_description: str = "",
        episode_type: EpisodeType = EpisodeType.MESSAGE,
        reference_time: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add an episode to the knowledge graph.

        Episodes are the primary unit of information ingestion in Graphiti.
        They are processed to extract entities and relationships.

        Args:
            content: The episode content (message, document text, etc.)
            source: Source identifier
            source_description: Human-readable source description
            episode_type: Type of episode
            reference_time: When the event occurred (defaults to now)
            metadata: Additional metadata

        Returns:
            Episode ID

        TODO: Use real Graphiti SDK:
            client = self._get_client()
            if client:
                result = await client.add_episode(
                    name=f"episode_{generate_memory_id(content)}",
                    episode_body=content,
                    source=episode_type.value,
                    source_description=source_description,
                    reference_time=reference_time or datetime.now(timezone.utc),
                )
                return result.uuid
        """
        episode_id = f"ep_{generate_memory_id(content)}"

        episode = GraphitiEpisode(
            id=episode_id,
            content=content,
            episode_type=episode_type,
            source=source,
            source_description=source_description,
            reference_time=reference_time or datetime.now(timezone.utc),
            metadata=metadata or {},
        )

        # PLACEHOLDER: Store in memory
        self._episodes[episode_id] = episode

        # Also create a MemoryEntry for unified access
        entry = self._episode_to_entry(episode)
        self._local_cache[episode_id] = entry

        # TODO: Extract entities and relationships using LLM
        # This is where Graphiti's magic happens - it uses an LLM to
        # extract entities, relationships, and facts from the episode
        extracted = await self._extract_entities_placeholder(content)
        episode.extracted_entities = [e.id for e in extracted["entities"]]
        episode.extracted_relations = [r.id for r in extracted["relations"]]

        logger.debug(f"Added episode {episode_id}: {len(episode.extracted_entities)} entities extracted")
        return episode_id

    async def get_episode(self, episode_id: str) -> Optional[GraphitiEpisode]:
        """Get episode by ID."""
        return self._episodes.get(episode_id)

    async def search_episodes(
        self,
        query: str,
        limit: int = 10,
        valid_at: Optional[datetime] = None,
    ) -> List[GraphitiSearchResult]:
        """
        Search episodes with optional temporal filtering.

        TODO: Use Graphiti's temporal search capabilities:
            client = self._get_client()
            if client:
                results = await client.search(
                    query=query,
                    num_results=limit,
                    # Graphiti supports temporal queries
                )
        """
        results: List[GraphitiSearchResult] = []
        query_lower = query.lower()

        for episode in self._episodes.values():
            if query_lower in episode.content.lower():
                # Simple relevance score based on match position
                pos = episode.content.lower().find(query_lower)
                score = 1.0 - (pos / len(episode.content)) if pos >= 0 else 0.0

                results.append(GraphitiSearchResult(
                    uuid=episode.id,
                    content=episode.content,
                    score=score,
                    episode=episode,
                    valid_at=episode.reference_time,
                ))

        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    # =========================================================================
    # ENTITY MANAGEMENT
    # =========================================================================

    async def get_entity(self, entity_id: str) -> Optional[GraphitiEntity]:
        """Get entity by ID."""
        return self._entities.get(entity_id)

    async def search_entities(
        self,
        query: str,
        entity_type: Optional[EntityType] = None,
        limit: int = 10,
    ) -> List[GraphitiEntity]:
        """
        Search entities by name or properties.

        TODO: Use Graphiti's entity search:
            client = self._get_client()
            if client:
                results = await client.search_entities(
                    query=query,
                    entity_type=entity_type,
                    limit=limit,
                )
        """
        results: List[GraphitiEntity] = []
        query_lower = query.lower()

        for entity in self._entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue
            if query_lower in entity.name.lower() or query_lower in entity.summary.lower():
                results.append(entity)
                if len(results) >= limit:
                    break

        return results

    async def get_related_entities(
        self,
        entity_id: str,
        relation_types: Optional[List[RelationType]] = None,
        limit: int = 10,
    ) -> List[Tuple[GraphitiEntity, GraphitiRelation]]:
        """
        Get entities related to a given entity.

        TODO: Use Graphiti's graph traversal:
            client = self._get_client()
            if client:
                results = await client.get_related(
                    node_id=entity_id,
                    relation_types=relation_types,
                )
        """
        results: List[Tuple[GraphitiEntity, GraphitiRelation]] = []

        for relation in self._relations.values():
            if relation_types and relation.relation_type not in relation_types:
                continue

            related_id = None
            if relation.source_id == entity_id:
                related_id = relation.target_id
            elif relation.target_id == entity_id:
                related_id = relation.source_id

            if related_id and related_id in self._entities:
                results.append((self._entities[related_id], relation))
                if len(results) >= limit:
                    break

        return results

    # =========================================================================
    # FACT EXTRACTION (Placeholder)
    # =========================================================================

    async def _extract_entities_placeholder(
        self,
        content: str
    ) -> Dict[str, List[Any]]:
        """
        Placeholder for entity/relationship extraction.

        Real implementation would use an LLM to extract:
        - Named entities (people, organizations, concepts)
        - Relationships between entities
        - Temporal facts (when things happened)
        - Factual statements

        TODO: Implement with LLM (Graphiti uses OpenAI by default):
            from graphiti_core.llm import extract_entities, extract_relations

            entities = await extract_entities(content, llm_client)
            relations = await extract_relations(content, entities, llm_client)
        """
        entities: List[GraphitiEntity] = []
        relations: List[GraphitiRelation] = []

        # PLACEHOLDER: Very basic extraction using simple patterns
        # This is NOT how Graphiti works - it uses LLM-based extraction

        # Look for capitalized words as potential entities
        import re
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)

        seen_names = set()
        for name in capitalized[:5]:  # Limit to avoid noise
            if name not in seen_names and len(name) > 2:
                seen_names.add(name)
                entity_id = f"ent_{hashlib.md5(name.encode()).hexdigest()[:8]}"

                entity = GraphitiEntity(
                    id=entity_id,
                    name=name,
                    entity_type=EntityType.CONCEPT,  # Default type
                    summary=f"Entity extracted from: {content[:50]}...",
                )
                entities.append(entity)
                self._entities[entity_id] = entity

        # Create relationships between consecutive entities
        for i in range(len(entities) - 1):
            relation_id = f"rel_{entities[i].id}_{entities[i+1].id}"
            relation = GraphitiRelation(
                id=relation_id,
                source_id=entities[i].id,
                target_id=entities[i+1].id,
                relation_type=RelationType.RELATES_TO,
                fact=f"{entities[i].name} relates to {entities[i+1].name}",
            )
            relations.append(relation)
            self._relations[relation_id] = relation

        return {"entities": entities, "relations": relations}

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _episode_to_entry(self, episode: GraphitiEpisode) -> MemoryEntry:
        """Convert GraphitiEpisode to MemoryEntry for unified interface."""
        return MemoryEntry(
            id=episode.id,
            content=episode.content,
            tier=self.tier,
            priority=MemoryPriority.NORMAL,
            content_type="episode",
            source=episode.source,
            created_at=episode.created_at,
            metadata={
                "episode_type": episode.episode_type.value,
                "source_description": episode.source_description,
                "reference_time": episode.reference_time.isoformat() if episode.reference_time else None,
                "extracted_entities": episode.extracted_entities,
                "extracted_relations": episode.extracted_relations,
                "is_graphiti": True,
                **episode.metadata,
            },
        )

    def _result_to_entry(self, result: GraphitiSearchResult) -> MemoryEntry:
        """Convert GraphitiSearchResult to MemoryEntry."""
        return MemoryEntry(
            id=result.uuid,
            content=result.content,
            tier=self.tier,
            priority=MemoryPriority.NORMAL,
            content_type="graphiti_result",
            metadata={
                "score": result.score,
                "valid_at": result.valid_at.isoformat() if result.valid_at else None,
                "temporal_distance": result.temporal_distance,
                "is_graphiti": True,
            },
        )


# =============================================================================
# GRAPHITI MEMORY BACKEND (Gateway Layer)
# =============================================================================

class GraphitiMemoryBackend(MemoryBackend):
    """
    Graphiti memory backend for the 5-layer unified memory gateway.

    Provides the GRAPH layer (Layer 4) in the memory architecture:
    - Layer 1: Letta (project-specific agents)
    - Layer 2: Claude-mem (observations)
    - Layer 3: Episodic (conversation archive)
    - Layer 4: Graph (entity relationships) <-- This backend
    - Layer 5: Static (CLAUDE.md configuration)

    TODO: Implement full Graphiti/Zep Cloud integration.
    """

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        zep_api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize Graphiti memory backend.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            zep_api_key: Zep Cloud API key (alternative to self-hosted)
        """
        self._tier_backend = GraphitiTierBackend(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            zep_api_key=zep_api_key,
        )
        logger.info("GraphitiMemoryBackend initialized (PLACEHOLDER)")

    @property
    def layer(self) -> MemoryLayer:
        """Return the layer this backend represents."""
        return MemoryLayer.GRAPH

    async def search(
        self,
        query: str,
        max_results: int = 10,
        namespace: Optional[MemoryNamespace] = None,
    ) -> List[MemoryEntry]:
        """
        Search the knowledge graph.

        Uses semantic search over entities and relationships.
        """
        return await self._tier_backend.search(query, max_results)

    async def store(
        self,
        content: str,
        namespace: MemoryNamespace,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store content by adding it as an episode.

        The episode will be processed to extract entities and relationships.
        """
        return await self._tier_backend.add_episode(
            content=content,
            source=namespace.value,
            source_description=f"Stored via unified gateway ({namespace.value})",
            metadata=metadata,
        )

    async def health_check(self) -> bool:
        """
        Check if the Graphiti backend is healthy.

        TODO: Actually check Neo4j/Zep connection.
        """
        # PLACEHOLDER: Always return True since we're using in-memory storage
        return True

    # Expose tier backend methods
    async def add_episode(
        self,
        content: str,
        source: str,
        source_description: str = "",
        episode_type: EpisodeType = EpisodeType.MESSAGE,
        reference_time: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add an episode to the knowledge graph."""
        return await self._tier_backend.add_episode(
            content=content,
            source=source,
            source_description=source_description,
            episode_type=episode_type,
            reference_time=reference_time,
            metadata=metadata,
        )

    async def get_entity(self, entity_id: str) -> Optional[GraphitiEntity]:
        """Get entity by ID."""
        return await self._tier_backend.get_entity(entity_id)

    async def search_entities(
        self,
        query: str,
        entity_type: Optional[EntityType] = None,
        limit: int = 10,
    ) -> List[GraphitiEntity]:
        """Search entities."""
        return await self._tier_backend.search_entities(query, entity_type, limit)

    async def get_related_entities(
        self,
        entity_id: str,
        relation_types: Optional[List[RelationType]] = None,
        limit: int = 10,
    ) -> List[Tuple[GraphitiEntity, GraphitiRelation]]:
        """Get related entities."""
        return await self._tier_backend.get_related_entities(
            entity_id, relation_types, limit
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

_graphiti_backend: Optional[GraphitiTierBackend] = None


def get_graphiti_backend(
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    zep_api_key: Optional[str] = None,
) -> GraphitiTierBackend:
    """
    Get or create the singleton Graphiti backend.

    Args:
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        zep_api_key: Zep Cloud API key

    Returns:
        GraphitiTierBackend instance
    """
    global _graphiti_backend

    if _graphiti_backend is None:
        _graphiti_backend = GraphitiTierBackend(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            zep_api_key=zep_api_key,
        )

    return _graphiti_backend


def reset_graphiti_backend() -> None:
    """Reset the singleton instance (for testing)."""
    global _graphiti_backend
    _graphiti_backend = None


__all__ = [
    # Data types
    "EntityType",
    "RelationType",
    "EpisodeType",
    "GraphitiEntity",
    "GraphitiRelation",
    "GraphitiEpisode",
    "GraphitiSearchResult",
    # Backends
    "GraphitiTierBackend",
    "GraphitiMemoryBackend",
    # Factory functions
    "get_graphiti_backend",
    "reset_graphiti_backend",
]
