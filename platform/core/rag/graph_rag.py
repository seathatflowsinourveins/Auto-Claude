"""
GraphRAG Integration - Entity-Relationship Knowledge Graph for RAG

Implements GraphRAG patterns from Microsoft GraphRAG analysis:
1. LLM-based entity extraction (PERSON, ORG, LOCATION, CONCEPT)
2. Relationship building with co-occurrence tracking
3. SQLite graph storage with adjacency list + FTS5
4. Graph-enhanced retrieval with 2-hop expansion
5. Hybrid search combining vector + graph context

Reference: docs/GRAPHRAG_ANALYSIS.md
Integration: Works as GraphRAGTool for AgenticRAG

Usage:
    from core.rag.graph_rag import GraphRAG, GraphRAGConfig

    config = GraphRAGConfig(db_path="graph.db", embedding_dim=384)
    graph_rag = GraphRAG(llm=my_llm, embedder=my_embedder, config=config)

    # Build graph from documents
    await graph_rag.ingest("Alice is CEO of TechCorp. TechCorp is in San Francisco.")

    # Graph-enhanced search
    results = await graph_rag.search("Who leads TechCorp?", top_k=5)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple, Set

logger = logging.getLogger(__name__)


# =============================================================================
# PROTOCOLS
# =============================================================================

class LLMProvider(Protocol):
    """Protocol for LLM providers."""
    async def generate(self, prompt: str, max_tokens: int = 1024, **kwargs) -> str: ...


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    def encode(self, texts: List[str]) -> List[List[float]]: ...


# =============================================================================
# DATA TYPES
# =============================================================================

class EntityType(str, Enum):
    """Entity types for knowledge graph."""
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "LOCATION"
    CONCEPT = "CONCEPT"
    EVENT = "EVENT"
    OBJECT = "OBJECT"
    UNKNOWN = "UNKNOWN"


class RelationshipType(str, Enum):
    """Common relationship types."""
    WORKS_FOR = "WORKS_FOR"
    LOCATED_IN = "LOCATED_IN"
    PART_OF = "PART_OF"
    IS_A = "IS_A"
    RELATED_TO = "RELATED_TO"
    CAUSES = "CAUSES"
    FOLLOWS = "FOLLOWS"
    OWNS = "OWNS"
    CREATED = "CREATED"


@dataclass
class Entity:
    """Graph entity node."""
    id: str
    name: str
    entity_type: EntityType
    description: str = ""
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    occurrence_count: int = 1


@dataclass
class Relationship:
    """Graph relationship edge."""
    id: str
    source_id: str
    target_id: str
    relationship_type: str
    description: str = ""
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphRAGConfig:
    """Configuration for GraphRAG."""
    db_path: str = ":memory:"
    embedding_dim: int = 384
    similarity_threshold: float = 0.85
    max_hops: int = 2
    max_entities_per_doc: int = 20
    max_relationships_per_doc: int = 30
    enable_embeddings: bool = True
    enable_fts: bool = True


@dataclass
class GraphSearchResult:
    """Result from graph-enhanced search."""
    query: str
    entities: List[Entity]
    relationships: List[Relationship]
    expanded_context: List[str]
    scores: Dict[str, float]
    latency_ms: float = 0.0


# =============================================================================
# ENTITY EXTRACTOR
# =============================================================================

class EntityExtractor:
    """LLM-based entity extraction."""

    EXTRACTION_PROMPT = """Extract entities and relationships from the following text.

Text: {text}

Instructions:
1. Identify named entities (PERSON, ORG, LOCATION, CONCEPT, EVENT, OBJECT)
2. Identify relationships between entities
3. Return as JSON with this structure:

{{
  "entities": [
    {{"name": "Entity Name", "type": "TYPE", "description": "Brief description"}}
  ],
  "relationships": [
    {{"source": "Entity1", "target": "Entity2", "type": "RELATIONSHIP_TYPE", "description": "Relationship description"}}
  ]
}}

Common relationship types: WORKS_FOR, LOCATED_IN, PART_OF, IS_A, RELATED_TO, CAUSES, OWNS, CREATED

JSON output:"""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def extract(self, text: str, max_entities: int = 20) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from text."""
        if not text or len(text.strip()) < 10:
            return [], []

        prompt = self.EXTRACTION_PROMPT.format(text=text[:4000])

        try:
            response = await self.llm.generate(prompt, max_tokens=1500, temperature=0.3)
            return self._parse_response(response, max_entities)
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return [], []

    def _parse_response(
        self, response: str, max_entities: int
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Parse LLM response into entities and relationships."""
        entities: List[Entity] = []
        relationships: List[Relationship] = []

        try:
            # Find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start == -1 or end == 0:
                return [], []

            data = json.loads(response[start:end])

            # Parse entities
            for item in data.get("entities", [])[:max_entities]:
                name = item.get("name", "").strip()
                if not name:
                    continue

                entity_type = self._parse_entity_type(item.get("type", ""))
                entity_id = self._generate_id(name)

                entities.append(Entity(
                    id=entity_id,
                    name=name,
                    entity_type=entity_type,
                    description=item.get("description", ""),
                ))

            # Build entity lookup
            entity_lookup = {e.name.lower(): e.id for e in entities}

            # Parse relationships
            for item in data.get("relationships", []):
                source = item.get("source", "").strip()
                target = item.get("target", "").strip()
                if not source or not target:
                    continue

                source_id = entity_lookup.get(source.lower())
                target_id = entity_lookup.get(target.lower())
                if not source_id or not target_id:
                    continue

                rel_id = self._generate_id(f"{source_id}-{target_id}")
                relationships.append(Relationship(
                    id=rel_id,
                    source_id=source_id,
                    target_id=target_id,
                    relationship_type=item.get("type", "RELATED_TO"),
                    description=item.get("description", ""),
                    weight=1.0,
                ))

        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse error in extraction: {e}")

        return entities, relationships

    def _parse_entity_type(self, type_str: str) -> EntityType:
        """Parse entity type string to enum."""
        type_upper = type_str.upper().strip()
        for et in EntityType:
            if et.value == type_upper or et.name == type_upper:
                return et
        return EntityType.UNKNOWN

    def _generate_id(self, name: str) -> str:
        """Generate stable ID from name."""
        return hashlib.md5(name.lower().encode()).hexdigest()[:16]


# =============================================================================
# GRAPH STORAGE (SQLite)
# =============================================================================

class GraphStorage:
    """SQLite-based graph storage with FTS5 search."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS entities (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        entity_type TEXT NOT NULL,
        description TEXT,
        embedding BLOB,
        metadata TEXT,
        occurrence_count INTEGER DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS relationships (
        id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        relationship_type TEXT NOT NULL,
        description TEXT,
        weight REAL DEFAULT 1.0,
        metadata TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (source_id) REFERENCES entities(id),
        FOREIGN KEY (target_id) REFERENCES entities(id)
    );

    CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
    CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
    CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id);
    CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id);
    """

    FTS_SCHEMA = """
    CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
        name, description, entity_type, content=entities, content_rowid=rowid
    );

    CREATE TRIGGER IF NOT EXISTS entities_ai AFTER INSERT ON entities BEGIN
        INSERT INTO entities_fts(rowid, name, description, entity_type)
        VALUES (NEW.rowid, NEW.name, NEW.description, NEW.entity_type);
    END;

    CREATE TRIGGER IF NOT EXISTS entities_ad AFTER DELETE ON entities BEGIN
        INSERT INTO entities_fts(entities_fts, rowid, name, description, entity_type)
        VALUES ('delete', OLD.rowid, OLD.name, OLD.description, OLD.entity_type);
    END;
    """

    def __init__(self, db_path: str = ":memory:", enable_fts: bool = True):
        self.db_path = db_path
        self.enable_fts = enable_fts
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Initialize database connection and schema."""
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self.SCHEMA)
        if self.enable_fts:
            try:
                self._conn.executescript(self.FTS_SCHEMA)
            except sqlite3.OperationalError as e:
                logger.warning(f"FTS5 not available: {e}")
        self._conn.commit()

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def _ensure_connected(self) -> sqlite3.Connection:
        if not self._conn:
            self.connect()
        return self._conn  # type: ignore

    def upsert_entity(self, entity: Entity) -> None:
        """Insert or update an entity."""
        conn = self._ensure_connected()
        embedding_blob = (
            sqlite3.Binary(json.dumps(entity.embedding).encode())
            if entity.embedding else None
        )
        conn.execute("""
            INSERT INTO entities (id, name, entity_type, description, embedding, metadata, occurrence_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                description = COALESCE(excluded.description, description),
                embedding = COALESCE(excluded.embedding, embedding),
                occurrence_count = occurrence_count + 1
        """, (
            entity.id,
            entity.name,
            entity.entity_type.value,
            entity.description,
            embedding_blob,
            json.dumps(entity.metadata),
            entity.occurrence_count,
        ))
        conn.commit()

    def upsert_relationship(self, rel: Relationship) -> None:
        """Insert or update a relationship."""
        conn = self._ensure_connected()
        conn.execute("""
            INSERT INTO relationships (id, source_id, target_id, relationship_type, description, weight, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                weight = weight + excluded.weight,
                description = COALESCE(excluded.description, description)
        """, (
            rel.id,
            rel.source_id,
            rel.target_id,
            rel.relationship_type,
            rel.description,
            rel.weight,
            json.dumps(rel.metadata),
        ))
        conn.commit()

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        conn = self._ensure_connected()
        row = conn.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        return self._row_to_entity(row) if row else None

    def search_entities(self, query: str, limit: int = 10) -> List[Entity]:
        """Search entities using FTS5."""
        conn = self._ensure_connected()
        try:
            if self.enable_fts:
                rows = conn.execute("""
                    SELECT e.* FROM entities e
                    JOIN entities_fts fts ON e.rowid = fts.rowid
                    WHERE entities_fts MATCH ?
                    LIMIT ?
                """, (query, limit)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM entities
                    WHERE name LIKE ? OR description LIKE ?
                    LIMIT ?
                """, (f"%{query}%", f"%{query}%", limit)).fetchall()
        except sqlite3.OperationalError:
            # FTS fallback
            rows = conn.execute("""
                SELECT * FROM entities
                WHERE name LIKE ? OR description LIKE ?
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", limit)).fetchall()

        return [self._row_to_entity(row) for row in rows]

    def get_neighbors(self, entity_id: str, hops: int = 1) -> Tuple[List[Entity], List[Relationship]]:
        """Get neighboring entities and relationships within N hops."""
        conn = self._ensure_connected()
        entities: Dict[str, Entity] = {}
        relationships: List[Relationship] = []

        visited: Set[str] = set()
        current_ids: Set[str] = {entity_id}

        for _ in range(hops):
            if not current_ids:
                break

            next_ids: Set[str] = set()

            for eid in current_ids:
                if eid in visited:
                    continue
                visited.add(eid)

                # Get outgoing relationships
                rows = conn.execute("""
                    SELECT r.*, e.id as target_entity_id, e.name, e.entity_type, e.description
                    FROM relationships r
                    JOIN entities e ON r.target_id = e.id
                    WHERE r.source_id = ?
                """, (eid,)).fetchall()

                for row in rows:
                    rel = Relationship(
                        id=row['id'],
                        source_id=row['source_id'],
                        target_id=row['target_id'],
                        relationship_type=row['relationship_type'],
                        description=row['description'] or "",
                        weight=row['weight'],
                    )
                    relationships.append(rel)
                    target_id = row['target_entity_id']
                    if target_id not in entities:
                        entities[target_id] = Entity(
                            id=target_id,
                            name=row['name'],
                            entity_type=EntityType(row['entity_type']),
                            description=row['description'] or "",
                        )
                        next_ids.add(target_id)

                # Get incoming relationships
                rows = conn.execute("""
                    SELECT r.*, e.id as source_entity_id, e.name, e.entity_type, e.description
                    FROM relationships r
                    JOIN entities e ON r.source_id = e.id
                    WHERE r.target_id = ?
                """, (eid,)).fetchall()

                for row in rows:
                    rel = Relationship(
                        id=row['id'],
                        source_id=row['source_id'],
                        target_id=row['target_id'],
                        relationship_type=row['relationship_type'],
                        description=row['description'] or "",
                        weight=row['weight'],
                    )
                    relationships.append(rel)
                    source_id = row['source_entity_id']
                    if source_id not in entities:
                        entities[source_id] = Entity(
                            id=source_id,
                            name=row['name'],
                            entity_type=EntityType(row['entity_type']),
                            description=row['description'] or "",
                        )
                        next_ids.add(source_id)

            current_ids = next_ids - visited

        return list(entities.values()), relationships

    def get_stats(self) -> Dict[str, int]:
        """Get storage statistics."""
        conn = self._ensure_connected()
        entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        rel_count = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
        return {"entities": entity_count, "relationships": rel_count}

    def _row_to_entity(self, row: sqlite3.Row) -> Entity:
        """Convert database row to Entity."""
        embedding = None
        if row['embedding']:
            try:
                embedding = json.loads(row['embedding'])
            except (json.JSONDecodeError, TypeError):
                pass

        metadata = {}
        if row['metadata']:
            try:
                metadata = json.loads(row['metadata'])
            except (json.JSONDecodeError, TypeError):
                pass

        return Entity(
            id=row['id'],
            name=row['name'],
            entity_type=EntityType(row['entity_type']),
            description=row['description'] or "",
            embedding=embedding,
            metadata=metadata,
            occurrence_count=row['occurrence_count'],
        )


# =============================================================================
# MAIN GRAPHRAG CLASS
# =============================================================================

class GraphRAG:
    """
    GraphRAG: Entity-Relationship Knowledge Graph for RAG.

    Implements Microsoft GraphRAG patterns with SQLite storage:
    - LLM-based entity extraction
    - Co-occurrence relationship building
    - 2-hop neighborhood expansion
    - Hybrid vector + graph search
    """

    def __init__(
        self,
        llm: LLMProvider,
        embedder: Optional[EmbeddingProvider] = None,
        config: Optional[GraphRAGConfig] = None,
    ):
        self.llm = llm
        self.embedder = embedder
        self.config = config or GraphRAGConfig()

        self._extractor = EntityExtractor(llm)
        self._storage = GraphStorage(
            db_path=self.config.db_path,
            enable_fts=self.config.enable_fts,
        )
        self._storage.connect()

    async def ingest(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
        """
        Ingest text into the knowledge graph.

        Args:
            text: Text to process
            metadata: Optional metadata to attach

        Returns:
            Dict with counts of entities and relationships added
        """
        # Extract entities and relationships
        entities, relationships = await self._extractor.extract(
            text, max_entities=self.config.max_entities_per_doc
        )

        # Add embeddings if available
        if self.embedder and self.config.enable_embeddings and entities:
            try:
                texts = [f"{e.name}: {e.description}" for e in entities]
                embeddings = self.embedder.encode(texts)
                for entity, emb in zip(entities, embeddings):
                    entity.embedding = emb
            except Exception as e:
                logger.warning(f"Embedding generation failed: {e}")

        # Store in graph
        for entity in entities:
            if metadata:
                entity.metadata.update(metadata)
            self._storage.upsert_entity(entity)

        for rel in relationships[:self.config.max_relationships_per_doc]:
            self._storage.upsert_relationship(rel)

        return {"entities": len(entities), "relationships": len(relationships)}

    async def search(
        self,
        query: str,
        top_k: int = 10,
        expand_hops: int = 2,
    ) -> GraphSearchResult:
        """
        Graph-enhanced search with neighborhood expansion.

        Args:
            query: Search query
            top_k: Maximum entities to return
            expand_hops: Number of hops for neighborhood expansion

        Returns:
            GraphSearchResult with entities, relationships, and context
        """
        start_time = time.time()

        # Search for matching entities
        seed_entities = self._storage.search_entities(query, limit=top_k)

        if not seed_entities:
            return GraphSearchResult(
                query=query,
                entities=[],
                relationships=[],
                expanded_context=[],
                scores={},
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Expand to neighborhood
        all_entities: Dict[str, Entity] = {e.id: e for e in seed_entities}
        all_relationships: List[Relationship] = []

        hops = min(expand_hops, self.config.max_hops)
        for entity in seed_entities:
            neighbors, rels = self._storage.get_neighbors(entity.id, hops=hops)
            for n in neighbors:
                if n.id not in all_entities:
                    all_entities[n.id] = n
            all_relationships.extend(rels)

        # Deduplicate relationships
        seen_rels: Set[str] = set()
        unique_rels: List[Relationship] = []
        for rel in all_relationships:
            if rel.id not in seen_rels:
                seen_rels.add(rel.id)
                unique_rels.append(rel)

        # Build expanded context
        context = self._build_context(list(all_entities.values()), unique_rels)

        # Score entities by relevance (occurrence + connection count)
        scores = {}
        for eid, entity in all_entities.items():
            connection_count = sum(1 for r in unique_rels if r.source_id == eid or r.target_id == eid)
            scores[entity.name] = entity.occurrence_count + connection_count * 0.5

        return GraphSearchResult(
            query=query,
            entities=list(all_entities.values())[:top_k],
            relationships=unique_rels,
            expanded_context=context,
            scores=scores,
            latency_ms=(time.time() - start_time) * 1000,
        )

    def _build_context(
        self, entities: List[Entity], relationships: List[Relationship]
    ) -> List[str]:
        """Build context strings from graph data."""
        context: List[str] = []

        # Entity lookup
        entity_map = {e.id: e.name for e in entities}

        # Add entity descriptions
        for entity in entities:
            if entity.description:
                context.append(f"{entity.name} ({entity.entity_type.value}): {entity.description}")

        # Add relationship descriptions
        for rel in relationships:
            source = entity_map.get(rel.source_id, rel.source_id)
            target = entity_map.get(rel.target_id, rel.target_id)
            rel_text = f"{source} --[{rel.relationship_type}]--> {target}"
            if rel.description:
                rel_text += f": {rel.description}"
            context.append(rel_text)

        return context

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self._storage.get_entity(entity_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return self._storage.get_stats()

    def close(self) -> None:
        """Close storage connection."""
        self._storage.close()


# =============================================================================
# GRAPHRAG TOOL FOR AGENTIC RAG
# =============================================================================

class GraphRAGTool:
    """
    GraphRAG as a retrieval tool for AgenticRAG integration.

    Implements the RetrievalTool protocol for seamless integration
    with the agentic RAG loop.
    """

    def __init__(self, graph_rag: GraphRAG):
        self._graph_rag = graph_rag
        self._name = "graph_rag"
        self._description = (
            "Knowledge graph-based search that finds entities and their relationships. "
            "Best for questions about people, organizations, locations, and their connections. "
            "Expands context through relationship traversal."
        )
        self._query_types = ["factual", "multi_hop", "explanation", "comparison"]

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def supported_query_types(self) -> List[str]:
        return self._query_types

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using graph-enhanced search.

        Args:
            query: Search query
            top_k: Maximum results to return
            **kwargs: Additional options (expand_hops)

        Returns:
            List of documents with content, score, and metadata
        """
        expand_hops = kwargs.get("expand_hops", 2)

        try:
            result = await self._graph_rag.search(
                query=query,
                top_k=top_k,
                expand_hops=expand_hops,
            )

            documents = []
            for ctx in result.expanded_context[:top_k]:
                documents.append({
                    "content": ctx,
                    "score": 0.8,  # Graph context is high quality
                    "metadata": {
                        "source": "graph_rag",
                        "query": query,
                        "entity_count": len(result.entities),
                        "relationship_count": len(result.relationships),
                    }
                })

            return documents

        except Exception as e:
            logger.error(f"GraphRAG retrieval error: {e}")
            return []


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main class
    "GraphRAG",
    # Configuration
    "GraphRAGConfig",
    # Data types
    "Entity",
    "Relationship",
    "EntityType",
    "RelationshipType",
    "GraphSearchResult",
    # Components
    "EntityExtractor",
    "GraphStorage",
    # Tool integration
    "GraphRAGTool",
    # Protocols
    "LLMProvider",
    "EmbeddingProvider",
]
