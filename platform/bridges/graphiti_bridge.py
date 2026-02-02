#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "neo4j>=5.0.0",
#     "qdrant-client>=1.7.0",
#     "structlog>=24.1.0",
#     "numpy>=1.26.0",
# ]
# ///
"""
Graphiti Knowledge Graph Bridge - Ultimate Autonomous Platform

Unified bridge combining:
- Neo4j: Bi-temporal knowledge graph with relationship traversal
- Qdrant: Vector similarity search for semantic matching

Architecture based on Graphiti patterns:
- Nodes: Entities (agents, tasks, decisions, artifacts)
- Edges: Relationships with temporal metadata
- Vectors: Embeddings for semantic search

References:
- https://github.com/getzep/graphiti
- GOALS_TRACKING.md: Memory Architecture (Letta 80% / Graphiti 20%)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class EntityType(Enum):
    """Types of entities in the knowledge graph."""
    AGENT = "agent"
    TASK = "task"
    DECISION = "decision"
    ARTIFACT = "artifact"
    SESSION = "session"
    MEMORY = "memory"
    PATTERN = "pattern"
    ERROR = "error"


class RelationType(Enum):
    """Types of relationships between entities."""
    CREATED = "CREATED"
    ASSIGNED_TO = "ASSIGNED_TO"
    COMPLETED = "COMPLETED"
    DEPENDS_ON = "DEPENDS_ON"
    REFERENCES = "REFERENCES"
    LEARNED_FROM = "LEARNED_FROM"
    CAUSED = "CAUSED"
    PART_OF = "PART_OF"
    FOLLOWS = "FOLLOWS"
    SIMILAR_TO = "SIMILAR_TO"


@dataclass
class Entity:
    """A node in the knowledge graph."""
    id: str
    type: EntityType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_from: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_to: Optional[datetime] = None  # None = currently valid

    def to_neo4j_params(self) -> Dict[str, Any]:
        """Convert to Neo4j parameters."""
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "properties": json.dumps(self.properties),
            "created_at": self.created_at.isoformat(),
            "valid_from": self.valid_from.isoformat(),
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
        }


@dataclass
class Relationship:
    """An edge in the knowledge graph."""
    id: str
    type: RelationType
    source_id: str
    target_id: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_from: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_to: Optional[datetime] = None

    def to_neo4j_params(self) -> Dict[str, Any]:
        """Convert to Neo4j parameters."""
        return {
            "id": self.id,
            "type": self.type.value,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "properties": json.dumps(self.properties),
            "created_at": self.created_at.isoformat(),
            "valid_from": self.valid_from.isoformat(),
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
        }


class GraphitiBridge:
    """
    Unified knowledge graph bridge for the Ultimate Autonomous Platform.

    Combines Neo4j graph database with Qdrant vector store for:
    - Bi-temporal queries (what was true at time T?)
    - Relationship traversal (graph patterns)
    - Semantic similarity search (embeddings)
    """

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_auth: Tuple[str, str] = ("neo4j", "alphaforge2024"),
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "graphiti_entities",
        embedding_dim: int = 384,
    ):
        self.neo4j_uri = neo4j_uri
        self.neo4j_auth = neo4j_auth
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

        self._neo4j_driver = None
        self._qdrant_client = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize connections to Neo4j and Qdrant."""
        try:
            # Initialize Neo4j
            from neo4j import GraphDatabase
            self._neo4j_driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=self.neo4j_auth
            )

            # Verify connection
            with self._neo4j_driver.session() as session:
                session.run("RETURN 1")

            # Create constraints and indexes
            await self._setup_neo4j_schema()

            logger.info("neo4j_initialized", uri=self.neo4j_uri)

        except Exception as e:
            logger.error("neo4j_init_failed", error=str(e))
            return False

        try:
            # Initialize Qdrant
            from qdrant_client import QdrantClient
            from qdrant_client.models import VectorParams, Distance

            self._qdrant_client = QdrantClient(url=self.qdrant_url)

            # Ensure collection exists
            collections = self._qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                self._qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info("qdrant_collection_created", name=self.collection_name)

            logger.info("qdrant_initialized", url=self.qdrant_url)

        except Exception as e:
            logger.error("qdrant_init_failed", error=str(e))
            return False

        self._initialized = True
        return True

    async def _setup_neo4j_schema(self) -> None:
        """Setup Neo4j constraints and indexes for the knowledge graph."""
        with self._neo4j_driver.session() as session:
            # Create uniqueness constraint on entity ID
            try:
                session.run("""
                    CREATE CONSTRAINT entity_id IF NOT EXISTS
                    FOR (e:Entity) REQUIRE e.id IS UNIQUE
                """)
            except Exception:
                pass  # Constraint may already exist

            # Create index on entity type
            try:
                session.run("""
                    CREATE INDEX entity_type IF NOT EXISTS
                    FOR (e:Entity) ON (e.type)
                """)
            except Exception:
                pass

            # Create index on valid_from for temporal queries
            try:
                session.run("""
                    CREATE INDEX entity_valid_from IF NOT EXISTS
                    FOR (e:Entity) ON (e.valid_from)
                """)
            except Exception:
                pass

    async def add_entity(self, entity: Entity) -> bool:
        """
        Add an entity to both Neo4j and Qdrant.

        Neo4j stores the entity properties and relationships.
        Qdrant stores the embedding for similarity search.
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Add to Neo4j
            with self._neo4j_driver.session() as session:
                params = entity.to_neo4j_params()
                session.run("""
                    CREATE (e:Entity {
                        id: $id,
                        type: $type,
                        name: $name,
                        properties: $properties,
                        created_at: $created_at,
                        valid_from: $valid_from,
                        valid_to: $valid_to
                    })
                """, params)

            # Add to Qdrant if embedding exists
            if entity.embedding:
                from qdrant_client.models import PointStruct

                point = PointStruct(
                    id=self._generate_point_id(entity.id),
                    vector=entity.embedding,
                    payload={
                        "entity_id": entity.id,
                        "type": entity.type.value,
                        "name": entity.name,
                        "created_at": entity.created_at.isoformat(),
                    }
                )
                self._qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=[point]
                )

            logger.info("entity_added", id=entity.id, type=entity.type.value)
            return True

        except Exception as e:
            logger.error("entity_add_failed", id=entity.id, error=str(e))
            return False

    async def add_relationship(self, relationship: Relationship) -> bool:
        """Add a relationship between two entities in Neo4j."""
        if not self._initialized:
            await self.initialize()

        try:
            with self._neo4j_driver.session() as session:
                params = relationship.to_neo4j_params()
                session.run(f"""
                    MATCH (source:Entity {{id: $source_id}})
                    MATCH (target:Entity {{id: $target_id}})
                    CREATE (source)-[r:{relationship.type.value} {{
                        id: $id,
                        properties: $properties,
                        created_at: $created_at,
                        valid_from: $valid_from,
                        valid_to: $valid_to
                    }}]->(target)
                """, params)

            logger.info(
                "relationship_added",
                type=relationship.type.value,
                source=relationship.source_id,
                target=relationship.target_id
            )
            return True

        except Exception as e:
            logger.error("relationship_add_failed", error=str(e))
            return False

    async def query_graph(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query on Neo4j."""
        if not self._initialized:
            await self.initialize()

        results = []
        with self._neo4j_driver.session() as session:
            result = session.run(cypher, params or {})
            for record in result:
                results.append(dict(record))

        return results

    async def find_similar_entities(
        self,
        embedding: List[float],
        entity_type: Optional[EntityType] = None,
        limit: int = 10,
        score_threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Find entities similar to the given embedding using Qdrant.
        Returns list of (entity_id, similarity_score) tuples.
        """
        if not self._initialized:
            await self.initialize()

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Build filter if entity type specified
        query_filter = None
        if entity_type:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="type",
                        match=MatchValue(value=entity_type.value)
                    )
                ]
            )

        results = self._qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold
        )

        return [(r.payload["entity_id"], r.score) for r in results]

    async def get_entity_history(
        self,
        entity_id: str,
        as_of: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get the history of an entity (bi-temporal query).
        If as_of is provided, returns the state at that point in time.
        """
        if not self._initialized:
            await self.initialize()

        if as_of:
            # Point-in-time query
            cypher = """
                MATCH (e:Entity {id: $id})
                WHERE e.valid_from <= $as_of
                AND (e.valid_to IS NULL OR e.valid_to > $as_of)
                RETURN e
            """
            params = {"id": entity_id, "as_of": as_of.isoformat()}
        else:
            # Full history
            cypher = """
                MATCH (e:Entity {id: $id})
                RETURN e
                ORDER BY e.valid_from
            """
            params = {"id": entity_id}

        return await self.query_graph(cypher, params)

    async def find_connected_entities(
        self,
        entity_id: str,
        relationship_type: Optional[RelationType] = None,
        direction: str = "both",  # "in", "out", "both"
        max_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """Find entities connected to the given entity via relationships."""
        if not self._initialized:
            await self.initialize()

        # Build relationship pattern
        rel_pattern = f":{relationship_type.value}" if relationship_type else ""

        if direction == "out":
            pattern = f"-[r{rel_pattern}*1..{max_depth}]->"
        elif direction == "in":
            pattern = f"<-[r{rel_pattern}*1..{max_depth}]-"
        else:
            pattern = f"-[r{rel_pattern}*1..{max_depth}]-"

        cypher = f"""
            MATCH (start:Entity {{id: $id}}){pattern}(connected:Entity)
            RETURN DISTINCT connected.id as id, connected.type as type,
                   connected.name as name, connected.properties as properties
        """

        return await self.query_graph(cypher, {"id": entity_id})

    async def record_decision(
        self,
        decision_id: str,
        description: str,
        context: Dict[str, Any],
        outcome: Optional[str] = None,
        related_entities: Optional[List[str]] = None
    ) -> bool:
        """
        Record a decision in the knowledge graph.

        Useful for tracking architectural decisions, debugging choices, etc.
        """
        entity = Entity(
            id=decision_id,
            type=EntityType.DECISION,
            name=description[:80],
            properties={
                "description": description,
                "context": context,
                "outcome": outcome,
            }
        )

        success = await self.add_entity(entity)

        # Link to related entities
        if success and related_entities:
            for related_id in related_entities:
                rel = Relationship(
                    id=f"rel-{uuid.uuid4().hex[:8]}",
                    type=RelationType.REFERENCES,
                    source_id=decision_id,
                    target_id=related_id
                )
                await self.add_relationship(rel)

        return success

    async def record_pattern(
        self,
        pattern_name: str,
        description: str,
        examples: List[str],
        embedding: Optional[List[float]] = None
    ) -> bool:
        """Record a discovered pattern for future reference."""
        entity = Entity(
            id=f"pattern-{uuid.uuid4().hex[:8]}",
            type=EntityType.PATTERN,
            name=pattern_name,
            properties={
                "description": description,
                "examples": examples,
                "usage_count": 0,
            },
            embedding=embedding
        )
        return await self.add_entity(entity)

    async def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        if not self._initialized:
            await self.initialize()

        stats = {}

        # Neo4j stats
        with self._neo4j_driver.session() as session:
            # Node count by type
            result = session.run("""
                MATCH (e:Entity)
                RETURN e.type as type, count(e) as count
            """)
            stats["entities_by_type"] = {r["type"]: r["count"] for r in result}

            # Total relationships
            result = session.run("""
                MATCH ()-[r]->()
                RETURN count(r) as count
            """)
            stats["total_relationships"] = result.single()["count"]

        # Qdrant stats
        collection_info = self._qdrant_client.get_collection(self.collection_name)
        stats["vectors_count"] = collection_info.points_count
        stats["vectors_indexed"] = collection_info.indexed_vectors_count

        return stats

    def _generate_point_id(self, entity_id: str) -> int:
        """Generate a numeric ID for Qdrant from string entity ID."""
        return int(hashlib.md5(entity_id.encode()).hexdigest()[:15], 16)

    async def close(self) -> None:
        """Close connections."""
        if self._neo4j_driver:
            self._neo4j_driver.close()
        logger.info("graphiti_bridge_closed")


# Demo and testing
async def main():
    """Demo the Graphiti bridge."""
    print("=" * 50)
    print("GRAPHITI KNOWLEDGE GRAPH BRIDGE DEMO")
    print("=" * 50)

    bridge = GraphitiBridge()

    print("\nInitializing connections...")
    success = await bridge.initialize()
    if not success:
        print("Failed to initialize bridge")
        return

    print("Connections established!")

    # Create some test entities
    print("\nCreating test entities...")

    # Agent entity
    agent = Entity(
        id="agent-queen-001",
        type=EntityType.AGENT,
        name="Queen Coordinator",
        properties={
            "role": "coordinator",
            "topology": "hierarchical",
            "capabilities": ["task_assignment", "worker_management"]
        }
    )
    await bridge.add_entity(agent)
    print(f"  Created: {agent.name}")

    # Task entity
    task = Entity(
        id="task-test-001",
        type=EntityType.TASK,
        name="Integration Test Task",
        properties={
            "priority": 7,
            "status": "pending"
        }
    )
    await bridge.add_entity(task)
    print(f"  Created: {task.name}")

    # Decision entity
    await bridge.record_decision(
        decision_id="decision-001",
        description="Use hierarchical topology for swarm coordination",
        context={
            "alternatives": ["mesh", "star", "ring"],
            "rationale": "Better for task decomposition"
        },
        outcome="Implemented in coordinator.py",
        related_entities=["agent-queen-001"]
    )
    print("  Created: Architecture decision")

    # Create relationship
    rel = Relationship(
        id="rel-001",
        type=RelationType.ASSIGNED_TO,
        source_id="task-test-001",
        target_id="agent-queen-001"
    )
    await bridge.add_relationship(rel)
    print("  Created: Task-Agent relationship")

    # Query the graph
    print("\nQuerying knowledge graph...")

    # Find connected entities
    connected = await bridge.find_connected_entities(
        "agent-queen-001",
        direction="in",
        max_depth=2
    )
    print(f"  Entities connected to Queen: {len(connected)}")
    for c in connected:
        print(f"    - {c['name']} ({c['type']})")

    # Get statistics
    print("\nGraph Statistics:")
    stats = await bridge.get_statistics()
    print(f"  Entities by type: {stats['entities_by_type']}")
    print(f"  Total relationships: {stats['total_relationships']}")
    print(f"  Vectors stored: {stats['vectors_count']}")

    await bridge.close()
    print("\nDemo complete!")


if __name__ == "__main__":
    asyncio.run(main())
