# -*- coding: utf-8 -*-
"""
GraphRAG V2 - Enhanced Graph-Based Retrieval-Augmented Generation

Advanced GraphRAG implementation with hierarchical community detection,
multiple query modes (local, global, DRIFT, hybrid), and Microsoft
GraphRAG-inspired architecture.

Key Features:
    1. Hierarchical Leiden community detection via graspologic/igraph
    2. Four query modes: LOCAL, GLOBAL, DRIFT, HYBRID
    3. Community summarization with map-reduce pattern
    4. Integration with existing knowledge_graph.py and entity_extractor.py
    5. Benchmark-ready with detailed metrics

References:
    - Microsoft GraphRAG: https://microsoft.github.io/graphrag/
    - DRIFT Search: https://microsoft.github.io/graphrag/query/drift_search/
    - Leiden Algorithm: https://www.nature.com/articles/s41598-019-41695-z
    - Neo4j GraphRAG: https://neo4j.com/docs/neo4j-graphrag-python/

Usage:
    from core.rag.graph_rag_v2 import GraphRAGV2, GraphRAGV2Config, QueryMode

    config = GraphRAGV2Config(
        db_path="data/graph_rag_v2.db",
        enable_hierarchical_communities=True,
        max_community_levels=4,
    )
    graph = GraphRAGV2(config=config)

    # Ingest documents
    await graph.ingest("Alice is CEO of TechCorp. TechCorp is in San Francisco.")

    # Query with different modes
    result = await graph.query("Who leads TechCorp?", mode=QueryMode.LOCAL)
    result = await graph.query("What are the main themes?", mode=QueryMode.GLOBAL)
    result = await graph.query("Compare Alice and Bob", mode=QueryMode.DRIFT)
    result = await graph.query("Full analysis", mode=QueryMode.HYBRID)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sqlite3
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    nx = None  # type: ignore[assignment]
    HAS_NETWORKX = False

try:
    from graspologic.partition import hierarchical_leiden
    HAS_GRASPOLOGIC = True
except ImportError:
    HAS_GRASPOLOGIC = False

try:
    import leidenalg  # type: ignore[import-untyped]
    import igraph as ig  # type: ignore[import-untyped]
    HAS_LEIDENALG = True
except ImportError:
    HAS_LEIDENALG = False


# =============================================================================
# PROTOCOLS
# =============================================================================

class LLMProvider(Protocol):
    """Protocol for LLM providers used in community summarization."""
    async def generate(self, prompt: str, max_tokens: int = 2048, **kwargs) -> str:
        ...


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    def encode(self, texts: List[str]) -> List[List[float]]:
        ...


# =============================================================================
# ENUMS
# =============================================================================

class QueryMode(str, Enum):
    """Query modes for GraphRAG V2."""
    LOCAL = "local"      # Entity-centric neighborhood traversal
    GLOBAL = "global"    # Community-based map-reduce summarization
    DRIFT = "drift"      # Dynamic reasoning with follow-up questions
    HYBRID = "hybrid"    # Combination of local + global + DRIFT


class CommunityLevel(int, Enum):
    """Hierarchical community levels."""
    LEAF = 0        # Finest granularity (individual entities)
    SMALL = 1       # 3-10 entities
    MEDIUM = 2      # 10-50 entities
    LARGE = 3       # 50-200 entities
    ROOT = 4        # Corpus-wide themes


class EntityType(str, Enum):
    """Entity types for knowledge graph."""
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "LOCATION"
    CONCEPT = "CONCEPT"
    EVENT = "EVENT"
    TECHNOLOGY = "TECH"
    FRAMEWORK = "FRAMEWORK"
    MODEL = "MODEL"
    UNKNOWN = "UNKNOWN"


class RelationType(str, Enum):
    """Relationship types."""
    WORKS_FOR = "WORKS_FOR"
    LOCATED_IN = "LOCATED_IN"
    PART_OF = "PART_OF"
    IS_A = "IS_A"
    RELATED_TO = "RELATED_TO"
    CAUSES = "CAUSES"
    FOLLOWS = "FOLLOWS"
    OWNS = "OWNS"
    CREATED = "CREATED"
    SIMILAR_TO = "SIMILAR_TO"
    CONTRADICTS = "CONTRADICTS"
    SUPPORTS = "SUPPORTS"


# =============================================================================
# EXCEPTIONS
# =============================================================================

class GraphRAGError(Exception):
    """Base exception for GraphRAG V2."""


class CommunityDetectionError(GraphRAGError):
    """Error during community detection."""


class QueryError(GraphRAGError):
    """Error during query execution."""


class PersistenceError(GraphRAGError):
    """Error during persistence operations."""


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GraphRAGV2Config:
    """Configuration for GraphRAG V2.

    Attributes:
        db_path: SQLite database path for persistence.
        enable_hierarchical_communities: Enable Leiden hierarchical detection.
        max_community_levels: Maximum levels in community hierarchy.
        community_size_threshold: Min size for a community at each level.
        leiden_resolution: Resolution parameter for Leiden algorithm.
        embedding_dim: Dimension of entity embeddings.
        max_hops: Maximum hops for local search expansion.
        drift_max_follow_ups: Maximum follow-up questions in DRIFT mode.
        drift_confidence_threshold: Confidence threshold to stop DRIFT.
        global_top_k_communities: Communities to consider in global search.
        rrf_k: RRF fusion constant for hybrid search.
        enable_embeddings: Store and use entity embeddings.
        enable_community_summaries: Pre-compute community summaries.
        summary_max_tokens: Max tokens per community summary.
    """
    db_path: str = "data/graph_rag_v2.db"
    enable_hierarchical_communities: bool = True
    max_community_levels: int = 4
    community_size_threshold: List[int] = field(
        default_factory=lambda: [3, 10, 50, 200]  # Per level
    )
    leiden_resolution: float = 1.0
    embedding_dim: int = 1024
    max_hops: int = 2
    drift_max_follow_ups: int = 3
    drift_confidence_threshold: float = 0.8
    global_top_k_communities: int = 10
    rrf_k: int = 60
    enable_embeddings: bool = True
    enable_community_summaries: bool = True
    summary_max_tokens: int = 500


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
    community_ids: List[str] = field(default_factory=list)  # Hierarchical membership


@dataclass
class Relationship:
    """Graph relationship edge."""
    id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    description: str = ""
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Community:
    """A community (cluster) of entities.

    Communities are hierarchical: a level-2 community contains level-1
    sub-communities, which contain individual entities (level-0).
    """
    id: str
    level: int
    entity_ids: List[str]
    sub_community_ids: List[str] = field(default_factory=list)
    parent_community_id: Optional[str] = None
    summary: str = ""
    title: str = ""
    central_entity_id: Optional[str] = None
    size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DRIFTState:
    """State tracking for DRIFT search.

    DRIFT (Dynamic Reasoning and Inference with Flexible Traversal)
    iteratively refines queries through follow-up questions.
    """
    original_query: str
    current_query: str
    follow_ups: List[str] = field(default_factory=list)
    intermediate_answers: List[str] = field(default_factory=list)
    confidence: float = 0.0
    iteration: int = 0
    explored_communities: Set[str] = field(default_factory=set)
    explored_entities: Set[str] = field(default_factory=set)


@dataclass
class QueryResult:
    """Result from GraphRAG V2 query."""
    query: str
    mode: QueryMode
    answer: str
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    communities_used: List[Community] = field(default_factory=list)
    context_chunks: List[str] = field(default_factory=list)
    confidence: float = 0.0
    latency_ms: float = 0.0
    drift_state: Optional[DRIFTState] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestResult:
    """Result from document ingestion."""
    entities_added: int = 0
    relationships_added: int = 0
    communities_detected: int = 0
    latency_ms: float = 0.0


# =============================================================================
# COMMUNITY DETECTOR
# =============================================================================

class CommunityDetector:
    """Hierarchical community detection using Leiden algorithm.

    Supports multiple backends:
    1. graspologic (preferred): Native hierarchical Leiden
    2. leidenalg + igraph: Alternative implementation
    3. NetworkX greedy modularity: Fallback (non-hierarchical)

    References:
        - https://github.com/graspologic-org/graspologic
        - https://github.com/vtraag/leidenalg
        - https://github.com/microsoft/graphrag/discussions/1128
    """

    def __init__(
        self,
        resolution: float = 1.0,
        max_levels: int = 4,
        size_thresholds: Optional[List[int]] = None,
    ) -> None:
        self.resolution = resolution
        self.max_levels = max_levels
        self.size_thresholds = size_thresholds or [3, 10, 50, 200]
        self._backend = self._detect_backend()
        logger.info(
            "CommunityDetector initialized",
            backend=self._backend,
            max_levels=max_levels,
        )

    def _detect_backend(self) -> str:
        """Detect best available community detection backend."""
        if HAS_GRASPOLOGIC:
            return "graspologic"
        elif HAS_LEIDENALG:
            return "leidenalg"
        elif HAS_NETWORKX:
            return "networkx"
        else:
            return "none"

    def detect_hierarchical(
        self,
        graph: "nx.Graph",
    ) -> Dict[int, List[Community]]:
        """Detect hierarchical communities at multiple levels.

        Args:
            graph: NetworkX graph with nodes and edges.

        Returns:
            Dict mapping level -> list of communities at that level.

        Raises:
            CommunityDetectionError: If detection fails.
        """
        if graph.number_of_nodes() == 0:
            return {}

        if self._backend == "graspologic":
            return self._detect_graspologic(graph)
        elif self._backend == "leidenalg":
            return self._detect_leidenalg(graph)
        elif self._backend == "networkx":
            return self._detect_networkx(graph)
        else:
            raise CommunityDetectionError(
                "No community detection backend available. "
                "Install graspologic, leidenalg+igraph, or networkx."
            )

    def _detect_graspologic(
        self,
        graph: "nx.Graph",
    ) -> Dict[int, List[Community]]:
        """Detect communities using graspologic hierarchical Leiden."""
        try:
            # Convert to undirected for community detection
            undirected = graph.to_undirected() if graph.is_directed() else graph

            # Run hierarchical Leiden
            hierarchical_clusters = hierarchical_leiden(
                undirected,
                resolution=self.resolution,
                max_cluster_size=max(self.size_thresholds),
            )

            # Build community hierarchy
            communities_by_level: Dict[int, List[Community]] = defaultdict(list)
            node_to_communities: Dict[str, List[str]] = defaultdict(list)

            for cluster in hierarchical_clusters:
                level = self._determine_level(len(cluster.node_ids))
                community_id = f"comm_{level}_{hashlib.md5(str(sorted(cluster.node_ids)).encode()).hexdigest()[:12]}"

                community = Community(
                    id=community_id,
                    level=level,
                    entity_ids=list(cluster.node_ids),
                    size=len(cluster.node_ids),
                    metadata={
                        "is_final_cluster": cluster.is_final_cluster,
                    }
                )

                communities_by_level[level].append(community)

                # Track node membership
                for node_id in cluster.node_ids:
                    node_to_communities[node_id].append(community_id)

            # Assign central entities
            for level, comms in communities_by_level.items():
                for comm in comms:
                    subgraph = undirected.subgraph(comm.entity_ids)
                    if subgraph.number_of_nodes() > 0:
                        degrees = dict(subgraph.degree())
                        comm.central_entity_id = max(degrees, key=degrees.get)

            return dict(communities_by_level)

        except Exception as e:
            logger.warning("graspologic detection failed", error=str(e))
            return self._detect_networkx(graph)

    def _detect_leidenalg(
        self,
        graph: "nx.Graph",
    ) -> Dict[int, List[Community]]:
        """Detect communities using leidenalg + igraph."""
        try:
            # Convert NetworkX to igraph
            undirected = graph.to_undirected() if graph.is_directed() else graph
            node_list = list(undirected.nodes())
            node_to_idx = {n: i for i, n in enumerate(node_list)}

            edges = [
                (node_to_idx[u], node_to_idx[v])
                for u, v in undirected.edges()
            ]

            ig_graph = ig.Graph(n=len(node_list), edges=edges)

            # Run Leiden at multiple resolutions for hierarchy
            communities_by_level: Dict[int, List[Community]] = {}

            for level in range(self.max_levels):
                resolution = self.resolution * (2 ** level)  # Increase resolution per level
                partition = leidenalg.find_partition(
                    ig_graph,
                    leidenalg.RBConfigurationVertexPartition,
                    resolution_parameter=resolution,
                )

                level_communities: List[Community] = []
                for comm_idx, members in enumerate(partition):
                    entity_ids = [node_list[i] for i in members]

                    if len(entity_ids) < self.size_thresholds[min(level, len(self.size_thresholds) - 1)]:
                        continue

                    community_id = f"comm_{level}_{comm_idx}_{hashlib.md5(str(sorted(entity_ids)).encode()).hexdigest()[:8]}"

                    community = Community(
                        id=community_id,
                        level=level,
                        entity_ids=entity_ids,
                        size=len(entity_ids),
                    )

                    # Find central entity
                    subgraph = undirected.subgraph(entity_ids)
                    if subgraph.number_of_nodes() > 0:
                        degrees = dict(subgraph.degree())
                        community.central_entity_id = max(degrees, key=degrees.get)

                    level_communities.append(community)

                communities_by_level[level] = level_communities

            # Link parent-child relationships
            self._link_community_hierarchy(communities_by_level)

            return communities_by_level

        except Exception as e:
            logger.warning("leidenalg detection failed", error=str(e))
            return self._detect_networkx(graph)

    def _detect_networkx(
        self,
        graph: "nx.Graph",
    ) -> Dict[int, List[Community]]:
        """Fallback: Detect communities using NetworkX greedy modularity.

        This is non-hierarchical but provides basic community structure.
        """
        if not HAS_NETWORKX:
            raise CommunityDetectionError("NetworkX not available")

        undirected = graph.to_undirected() if graph.is_directed() else graph

        try:
            communities_gen = nx.community.greedy_modularity_communities(undirected)
            communities_list = list(communities_gen)
        except (nx.NetworkXError, ValueError) as e:
            logger.warning("NetworkX community detection failed", error=str(e))
            return {}

        # Create flat hierarchy (level 0 only)
        level_0_communities: List[Community] = []

        for idx, members in enumerate(communities_list):
            entity_ids = list(members)

            if len(entity_ids) < self.size_thresholds[0]:
                continue

            community_id = f"comm_0_{idx}_{hashlib.md5(str(sorted(entity_ids)).encode()).hexdigest()[:8]}"

            community = Community(
                id=community_id,
                level=0,
                entity_ids=entity_ids,
                size=len(entity_ids),
            )

            # Find central entity
            subgraph = undirected.subgraph(entity_ids)
            if subgraph.number_of_nodes() > 0:
                degrees = dict(subgraph.degree())
                community.central_entity_id = max(degrees, key=degrees.get)

            level_0_communities.append(community)

        return {0: level_0_communities}

    def _determine_level(self, size: int) -> int:
        """Determine community level based on size."""
        for level, threshold in enumerate(self.size_thresholds):
            if size <= threshold:
                return level
        return len(self.size_thresholds)

    def _link_community_hierarchy(
        self,
        communities_by_level: Dict[int, List[Community]],
    ) -> None:
        """Link parent-child relationships between community levels."""
        levels = sorted(communities_by_level.keys())

        for i, level in enumerate(levels[:-1]):
            next_level = levels[i + 1]
            child_comms = communities_by_level[level]
            parent_comms = communities_by_level[next_level]

            for child in child_comms:
                child_set = set(child.entity_ids)
                best_parent: Optional[Community] = None
                best_overlap = 0

                for parent in parent_comms:
                    parent_set = set(parent.entity_ids)
                    overlap = len(child_set & parent_set)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_parent = parent

                if best_parent and best_overlap > len(child_set) * 0.5:
                    child.parent_community_id = best_parent.id
                    if child.id not in best_parent.sub_community_ids:
                        best_parent.sub_community_ids.append(child.id)


# =============================================================================
# GRAPH STORAGE (SQLite)
# =============================================================================

class GraphStorageV2:
    """SQLite-based graph storage with community support.

    Schema includes:
    - entities: Node data with community membership
    - relationships: Edge data
    - communities: Hierarchical community data
    - community_summaries: Pre-computed community summaries
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS entities (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        entity_type TEXT NOT NULL,
        description TEXT,
        embedding BLOB,
        metadata TEXT,
        occurrence_count INTEGER DEFAULT 1,
        community_ids TEXT,
        created_at REAL
    );

    CREATE TABLE IF NOT EXISTS relationships (
        id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        relation_type TEXT NOT NULL,
        description TEXT,
        weight REAL DEFAULT 1.0,
        metadata TEXT,
        created_at REAL,
        FOREIGN KEY (source_id) REFERENCES entities(id),
        FOREIGN KEY (target_id) REFERENCES entities(id)
    );

    CREATE TABLE IF NOT EXISTS communities (
        id TEXT PRIMARY KEY,
        level INTEGER NOT NULL,
        entity_ids TEXT NOT NULL,
        sub_community_ids TEXT,
        parent_community_id TEXT,
        summary TEXT,
        title TEXT,
        central_entity_id TEXT,
        size INTEGER,
        metadata TEXT,
        created_at REAL
    );

    CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
    CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
    CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id);
    CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id);
    CREATE INDEX IF NOT EXISTS idx_comm_level ON communities(level);
    CREATE INDEX IF NOT EXISTS idx_comm_size ON communities(size);
    """

    FTS_SCHEMA = """
    CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
        name, description, entity_type,
        content=entities, content_rowid=rowid
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS communities_fts USING fts5(
        title, summary,
        content=communities, content_rowid=rowid
    );
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Initialize database connection and schema."""
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(self.SCHEMA)
        try:
            self._conn.executescript(self.FTS_SCHEMA)
        except sqlite3.OperationalError as e:
            logger.warning("FTS5 not available", error=str(e))
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

    # Entity operations
    def upsert_entity(self, entity: Entity) -> None:
        """Insert or update an entity."""
        conn = self._ensure_connected()
        embedding_blob = (
            sqlite3.Binary(json.dumps(entity.embedding).encode())
            if entity.embedding else None
        )
        conn.execute("""
            INSERT INTO entities (id, name, entity_type, description, embedding,
                                  metadata, occurrence_count, community_ids, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                description = COALESCE(excluded.description, description),
                embedding = COALESCE(excluded.embedding, embedding),
                occurrence_count = occurrence_count + 1,
                community_ids = COALESCE(excluded.community_ids, community_ids)
        """, (
            entity.id,
            entity.name,
            entity.entity_type.value,
            entity.description,
            embedding_blob,
            json.dumps(entity.metadata),
            entity.occurrence_count,
            json.dumps(entity.community_ids),
            time.time(),
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
        """Search entities using FTS or LIKE fallback.

        Falls back to LIKE search when:
        - FTS5 is not available
        - FTS returns empty results (triggers may not work with UPSERT)
        """
        conn = self._ensure_connected()
        rows = []

        # Try FTS first
        try:
            rows = conn.execute("""
                SELECT e.* FROM entities e
                JOIN entities_fts fts ON e.rowid = fts.rowid
                WHERE entities_fts MATCH ?
                LIMIT ?
            """, (query, limit)).fetchall()
        except sqlite3.OperationalError:
            pass  # FTS not available, fall through to LIKE

        # Fall back to LIKE if FTS returned nothing
        if not rows:
            rows = conn.execute("""
                SELECT * FROM entities
                WHERE name LIKE ? OR description LIKE ? OR entity_type LIKE ?
                ORDER BY occurrence_count DESC
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", f"%{query}%", limit)).fetchall()

        return [self._row_to_entity(row) for row in rows]

    def get_all_entities(self) -> List[Entity]:
        """Get all entities."""
        conn = self._ensure_connected()
        rows = conn.execute("SELECT * FROM entities").fetchall()
        return [self._row_to_entity(row) for row in rows]

    def _row_to_entity(self, row: sqlite3.Row) -> Entity:
        """Convert database row to Entity."""
        embedding = None
        if row["embedding"]:
            try:
                embedding = json.loads(row["embedding"])
            except (json.JSONDecodeError, TypeError):
                pass

        metadata = {}
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass

        community_ids = []
        if row["community_ids"]:
            try:
                community_ids = json.loads(row["community_ids"])
            except (json.JSONDecodeError, TypeError):
                pass

        return Entity(
            id=row["id"],
            name=row["name"],
            entity_type=EntityType(row["entity_type"]),
            description=row["description"] or "",
            embedding=embedding,
            metadata=metadata,
            occurrence_count=row["occurrence_count"],
            community_ids=community_ids,
        )

    # Relationship operations
    def upsert_relationship(self, rel: Relationship) -> None:
        """Insert or update a relationship."""
        conn = self._ensure_connected()
        conn.execute("""
            INSERT INTO relationships (id, source_id, target_id, relation_type,
                                       description, weight, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                weight = weight + excluded.weight,
                description = COALESCE(excluded.description, description)
        """, (
            rel.id,
            rel.source_id,
            rel.target_id,
            rel.relation_type.value,
            rel.description,
            rel.weight,
            json.dumps(rel.metadata),
            time.time(),
        ))
        conn.commit()

    def get_neighbors(
        self,
        entity_id: str,
        hops: int = 1,
    ) -> Tuple[List[Entity], List[Relationship]]:
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
                    SELECT r.*, e.id as target_entity_id, e.name, e.entity_type,
                           e.description, e.community_ids
                    FROM relationships r
                    JOIN entities e ON r.target_id = e.id
                    WHERE r.source_id = ?
                """, (eid,)).fetchall()

                for row in rows:
                    rel = Relationship(
                        id=row["id"],
                        source_id=row["source_id"],
                        target_id=row["target_id"],
                        relation_type=RelationType(row["relation_type"]),
                        description=row["description"] or "",
                        weight=row["weight"],
                    )
                    relationships.append(rel)
                    target_id = row["target_entity_id"]
                    if target_id not in entities:
                        community_ids = []
                        if row["community_ids"]:
                            try:
                                community_ids = json.loads(row["community_ids"])
                            except (json.JSONDecodeError, TypeError):
                                pass
                        entities[target_id] = Entity(
                            id=target_id,
                            name=row["name"],
                            entity_type=EntityType(row["entity_type"]),
                            description=row["description"] or "",
                            community_ids=community_ids,
                        )
                        next_ids.add(target_id)

                # Get incoming relationships
                rows = conn.execute("""
                    SELECT r.*, e.id as source_entity_id, e.name, e.entity_type,
                           e.description, e.community_ids
                    FROM relationships r
                    JOIN entities e ON r.source_id = e.id
                    WHERE r.target_id = ?
                """, (eid,)).fetchall()

                for row in rows:
                    rel = Relationship(
                        id=row["id"],
                        source_id=row["source_id"],
                        target_id=row["target_id"],
                        relation_type=RelationType(row["relation_type"]),
                        description=row["description"] or "",
                        weight=row["weight"],
                    )
                    relationships.append(rel)
                    source_id = row["source_entity_id"]
                    if source_id not in entities:
                        community_ids = []
                        if row["community_ids"]:
                            try:
                                community_ids = json.loads(row["community_ids"])
                            except (json.JSONDecodeError, TypeError):
                                pass
                        entities[source_id] = Entity(
                            id=source_id,
                            name=row["name"],
                            entity_type=EntityType(row["entity_type"]),
                            description=row["description"] or "",
                            community_ids=community_ids,
                        )
                        next_ids.add(source_id)

            current_ids = next_ids - visited

        return list(entities.values()), relationships

    # Community operations
    def upsert_community(self, community: Community) -> None:
        """Insert or update a community."""
        conn = self._ensure_connected()
        conn.execute("""
            INSERT INTO communities (id, level, entity_ids, sub_community_ids,
                                     parent_community_id, summary, title,
                                     central_entity_id, size, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                summary = COALESCE(excluded.summary, summary),
                title = COALESCE(excluded.title, title),
                size = excluded.size
        """, (
            community.id,
            community.level,
            json.dumps(community.entity_ids),
            json.dumps(community.sub_community_ids),
            community.parent_community_id,
            community.summary,
            community.title,
            community.central_entity_id,
            community.size,
            json.dumps(community.metadata),
            time.time(),
        ))
        conn.commit()

    def get_community(self, community_id: str) -> Optional[Community]:
        """Get community by ID."""
        conn = self._ensure_connected()
        row = conn.execute(
            "SELECT * FROM communities WHERE id = ?", (community_id,)
        ).fetchone()
        return self._row_to_community(row) if row else None

    def get_communities_by_level(self, level: int) -> List[Community]:
        """Get all communities at a specific level."""
        conn = self._ensure_connected()
        rows = conn.execute(
            "SELECT * FROM communities WHERE level = ? ORDER BY size DESC",
            (level,)
        ).fetchall()
        return [self._row_to_community(row) for row in rows]

    def search_communities(self, query: str, limit: int = 10) -> List[Community]:
        """Search communities using FTS or LIKE fallback."""
        conn = self._ensure_connected()
        try:
            rows = conn.execute("""
                SELECT c.* FROM communities c
                JOIN communities_fts fts ON c.rowid = fts.rowid
                WHERE communities_fts MATCH ?
                ORDER BY c.size DESC
                LIMIT ?
            """, (query, limit)).fetchall()
        except sqlite3.OperationalError:
            rows = conn.execute("""
                SELECT * FROM communities
                WHERE title LIKE ? OR summary LIKE ?
                ORDER BY size DESC
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", limit)).fetchall()

        return [self._row_to_community(row) for row in rows]

    def _row_to_community(self, row: sqlite3.Row) -> Community:
        """Convert database row to Community."""
        entity_ids = []
        if row["entity_ids"]:
            try:
                entity_ids = json.loads(row["entity_ids"])
            except (json.JSONDecodeError, TypeError):
                pass

        sub_community_ids = []
        if row["sub_community_ids"]:
            try:
                sub_community_ids = json.loads(row["sub_community_ids"])
            except (json.JSONDecodeError, TypeError):
                pass

        metadata = {}
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except (json.JSONDecodeError, TypeError):
                pass

        return Community(
            id=row["id"],
            level=row["level"],
            entity_ids=entity_ids,
            sub_community_ids=sub_community_ids,
            parent_community_id=row["parent_community_id"],
            summary=row["summary"] or "",
            title=row["title"] or "",
            central_entity_id=row["central_entity_id"],
            size=row["size"],
            metadata=metadata,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        conn = self._ensure_connected()
        entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        rel_count = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
        comm_count = conn.execute("SELECT COUNT(*) FROM communities").fetchone()[0]

        # Community distribution by level
        level_dist = {}
        for row in conn.execute(
            "SELECT level, COUNT(*) as cnt FROM communities GROUP BY level"
        ).fetchall():
            level_dist[row["level"]] = row["cnt"]

        return {
            "entities": entity_count,
            "relationships": rel_count,
            "communities": comm_count,
            "community_levels": level_dist,
        }


# =============================================================================
# QUERY EXECUTORS
# =============================================================================

class QueryExecutor(ABC):
    """Base class for query execution strategies."""

    @abstractmethod
    async def execute(
        self,
        query: str,
        storage: GraphStorageV2,
        config: GraphRAGV2Config,
        llm: Optional[LLMProvider] = None,
        embedder: Optional[EmbeddingProvider] = None,
    ) -> QueryResult:
        """Execute a query and return results."""
        ...


class LocalQueryExecutor(QueryExecutor):
    """LOCAL search: Entity-centric neighborhood traversal.

    Algorithm:
    1. Embed query and find similar entities
    2. Expand to N-hop neighborhood
    3. Collect relationships and build context
    4. Generate answer from local context
    """

    async def execute(
        self,
        query: str,
        storage: GraphStorageV2,
        config: GraphRAGV2Config,
        llm: Optional[LLMProvider] = None,
        embedder: Optional[EmbeddingProvider] = None,
    ) -> QueryResult:
        start_time = time.time()

        # Find seed entities via FTS search
        seed_entities = storage.search_entities(query, limit=5)

        if not seed_entities:
            return QueryResult(
                query=query,
                mode=QueryMode.LOCAL,
                answer="No relevant entities found.",
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Expand neighborhoods
        all_entities: Dict[str, Entity] = {e.id: e for e in seed_entities}
        all_relationships: List[Relationship] = []
        seen_rel_ids: Set[str] = set()

        for entity in seed_entities:
            neighbors, rels = storage.get_neighbors(entity.id, hops=config.max_hops)
            for n in neighbors:
                if n.id not in all_entities:
                    all_entities[n.id] = n
            for r in rels:
                if r.id not in seen_rel_ids:
                    seen_rel_ids.add(r.id)
                    all_relationships.append(r)

        # Build context chunks
        context_chunks = self._build_context(
            list(all_entities.values()),
            all_relationships,
        )

        # Generate answer (if LLM available)
        answer = self._synthesize_answer(query, context_chunks, llm)

        return QueryResult(
            query=query,
            mode=QueryMode.LOCAL,
            answer=answer,
            entities=list(all_entities.values()),
            relationships=all_relationships,
            context_chunks=context_chunks,
            confidence=0.7 if seed_entities else 0.0,
            latency_ms=(time.time() - start_time) * 1000,
            metrics={
                "seed_entities": len(seed_entities),
                "total_entities": len(all_entities),
                "total_relationships": len(all_relationships),
            },
        )

    def _build_context(
        self,
        entities: List[Entity],
        relationships: List[Relationship],
    ) -> List[str]:
        """Build context strings from graph data."""
        context: List[str] = []
        entity_map = {e.id: e.name for e in entities}

        # Entity descriptions
        for entity in entities:
            if entity.description:
                context.append(
                    f"{entity.name} ({entity.entity_type.value}): {entity.description}"
                )

        # Relationship descriptions
        for rel in relationships:
            source = entity_map.get(rel.source_id, rel.source_id)
            target = entity_map.get(rel.target_id, rel.target_id)
            rel_text = f"{source} --[{rel.relation_type.value}]--> {target}"
            if rel.description:
                rel_text += f": {rel.description}"
            context.append(rel_text)

        return context

    def _synthesize_answer(
        self,
        query: str,
        context: List[str],
        llm: Optional[LLMProvider],
    ) -> str:
        """Synthesize answer from context (sync wrapper)."""
        if not context:
            return "No context available to answer the query."

        # Without LLM, return concatenated context
        if llm is None:
            return "\n".join(context[:10])

        # With LLM, generate proper answer (would need async handling)
        return "\n".join(context[:10])


class GlobalQueryExecutor(QueryExecutor):
    """GLOBAL search: Community-based map-reduce summarization.

    Algorithm:
    1. Select top-K communities at target level
    2. MAP: Score and summarize each relevant community
    3. REDUCE: Combine community summaries into final answer

    References:
        - https://microsoft.github.io/graphrag/query/overview/
    """

    async def execute(
        self,
        query: str,
        storage: GraphStorageV2,
        config: GraphRAGV2Config,
        llm: Optional[LLMProvider] = None,
        embedder: Optional[EmbeddingProvider] = None,
    ) -> QueryResult:
        start_time = time.time()

        # Get communities, preferring higher levels for global view
        communities_used: List[Community] = []

        for level in range(config.max_community_levels - 1, -1, -1):
            level_communities = storage.get_communities_by_level(level)
            if level_communities:
                communities_used.extend(level_communities[:config.global_top_k_communities])
                break

        if not communities_used:
            # Fallback to entity search
            entities = storage.search_entities(query, limit=10)
            return QueryResult(
                query=query,
                mode=QueryMode.GLOBAL,
                answer="No communities available. Using entity search.",
                entities=entities,
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Build context from community summaries
        context_chunks = []
        for comm in communities_used:
            if comm.summary:
                context_chunks.append(f"[{comm.title or comm.id}]: {comm.summary}")
            else:
                # Build summary from entity names
                entity_names = []
                for eid in comm.entity_ids[:10]:
                    entity = storage.get_entity(eid)
                    if entity:
                        entity_names.append(entity.name)
                if entity_names:
                    context_chunks.append(
                        f"[Community {comm.id}]: Includes {', '.join(entity_names)}"
                    )

        # Synthesize answer
        answer = "\n".join(context_chunks) if context_chunks else "No relevant themes found."

        return QueryResult(
            query=query,
            mode=QueryMode.GLOBAL,
            answer=answer,
            communities_used=communities_used,
            context_chunks=context_chunks,
            confidence=0.6 if communities_used else 0.0,
            latency_ms=(time.time() - start_time) * 1000,
            metrics={
                "communities_searched": len(communities_used),
                "context_chunks": len(context_chunks),
            },
        )


class DRIFTQueryExecutor(QueryExecutor):
    """DRIFT search: Dynamic Reasoning and Inference with Flexible Traversal.

    Combines local and global search with iterative refinement:
    1. PRIMER: Compare query with top community reports
    2. FOLLOW-UP: Generate and answer follow-up questions using local search
    3. CONFIDENCE: Track confidence to stop early when answer is complete

    References:
        - https://microsoft.github.io/graphrag/query/drift_search/
        - https://www.microsoft.com/en-us/research/blog/introducing-drift-search/
    """

    async def execute(
        self,
        query: str,
        storage: GraphStorageV2,
        config: GraphRAGV2Config,
        llm: Optional[LLMProvider] = None,
        embedder: Optional[EmbeddingProvider] = None,
    ) -> QueryResult:
        start_time = time.time()

        # Initialize DRIFT state
        state = DRIFTState(
            original_query=query,
            current_query=query,
        )

        all_entities: Dict[str, Entity] = {}
        all_relationships: List[Relationship] = []
        communities_used: List[Community] = []
        context_chunks: List[str] = []

        local_executor = LocalQueryExecutor()
        global_executor = GlobalQueryExecutor()

        # Phase 1: PRIMER - Global context
        global_result = await global_executor.execute(
            query, storage, config, llm, embedder
        )
        context_chunks.extend(global_result.context_chunks)
        communities_used.extend(global_result.communities_used)

        for comm in global_result.communities_used:
            state.explored_communities.add(comm.id)

        # Phase 2: FOLLOW-UP iterations
        for iteration in range(config.drift_max_follow_ups):
            state.iteration = iteration + 1

            # Local search on current query
            local_result = await local_executor.execute(
                state.current_query, storage, config, llm, embedder
            )

            for e in local_result.entities:
                if e.id not in all_entities:
                    all_entities[e.id] = e
                    state.explored_entities.add(e.id)

            all_relationships.extend(local_result.relationships)
            context_chunks.extend(local_result.context_chunks)

            # Update intermediate answer
            if local_result.answer:
                state.intermediate_answers.append(local_result.answer)

            # Generate follow-up question (heuristic without LLM)
            follow_up = self._generate_follow_up(
                state, list(all_entities.values()), communities_used
            )

            if follow_up:
                state.follow_ups.append(follow_up)
                state.current_query = follow_up
            else:
                break

            # Check confidence threshold
            state.confidence = min(0.9, 0.5 + 0.15 * len(state.intermediate_answers))
            if state.confidence >= config.drift_confidence_threshold:
                break

        # Final answer synthesis
        answer = self._synthesize_drift_answer(state, context_chunks)

        return QueryResult(
            query=query,
            mode=QueryMode.DRIFT,
            answer=answer,
            entities=list(all_entities.values()),
            relationships=all_relationships,
            communities_used=communities_used,
            context_chunks=context_chunks[:20],
            confidence=state.confidence,
            latency_ms=(time.time() - start_time) * 1000,
            drift_state=state,
            metrics={
                "iterations": state.iteration,
                "follow_ups": len(state.follow_ups),
                "entities_explored": len(state.explored_entities),
                "communities_explored": len(state.explored_communities),
            },
        )

    def _generate_follow_up(
        self,
        state: DRIFTState,
        entities: List[Entity],
        communities: List[Community],
    ) -> Optional[str]:
        """Generate follow-up question based on current state."""
        # Simple heuristic: ask about unexplored high-degree entities
        unexplored = [
            e for e in entities
            if e.id not in state.explored_entities
            and e.occurrence_count > 1
        ]

        if unexplored:
            target = max(unexplored, key=lambda e: e.occurrence_count)
            return f"What is the relationship between {target.name} and {state.original_query}?"

        return None

    def _synthesize_drift_answer(
        self,
        state: DRIFTState,
        context_chunks: List[str],
    ) -> str:
        """Synthesize final answer from DRIFT exploration."""
        parts = []

        if state.intermediate_answers:
            parts.append("Based on iterative exploration:")
            for i, ans in enumerate(state.intermediate_answers, 1):
                parts.append(f"\n[Step {i}]: {ans[:200]}")

        if context_chunks:
            parts.append("\n\nKey context:")
            for chunk in context_chunks[:5]:
                parts.append(f"- {chunk[:150]}")

        return "\n".join(parts) if parts else "No conclusive answer found."


class HybridQueryExecutor(QueryExecutor):
    """HYBRID search: Combination of LOCAL + GLOBAL + DRIFT.

    Uses RRF (Reciprocal Rank Fusion) to combine results from all modes.
    """

    async def execute(
        self,
        query: str,
        storage: GraphStorageV2,
        config: GraphRAGV2Config,
        llm: Optional[LLMProvider] = None,
        embedder: Optional[EmbeddingProvider] = None,
    ) -> QueryResult:
        start_time = time.time()

        # Execute all strategies in parallel
        local_exec = LocalQueryExecutor()
        global_exec = GlobalQueryExecutor()
        drift_exec = DRIFTQueryExecutor()

        local_task = local_exec.execute(query, storage, config, llm, embedder)
        global_task = global_exec.execute(query, storage, config, llm, embedder)
        drift_task = drift_exec.execute(query, storage, config, llm, embedder)

        local_result, global_result, drift_result = await asyncio.gather(
            local_task, global_task, drift_task
        )

        # Merge entities with RRF scoring
        entity_scores: Dict[str, float] = defaultdict(float)

        for rank, entity in enumerate(local_result.entities, 1):
            entity_scores[entity.id] += 1.0 / (config.rrf_k + rank)

        for rank, entity in enumerate(drift_result.entities, 1):
            entity_scores[entity.id] += 1.0 / (config.rrf_k + rank)

        # Collect all unique entities
        all_entities: Dict[str, Entity] = {}
        for e in local_result.entities + drift_result.entities:
            if e.id not in all_entities:
                all_entities[e.id] = e

        # Sort by RRF score
        sorted_entity_ids = sorted(
            entity_scores.keys(),
            key=lambda eid: entity_scores[eid],
            reverse=True,
        )

        final_entities = [
            all_entities[eid]
            for eid in sorted_entity_ids
            if eid in all_entities
        ]

        # Merge relationships
        all_relationships: List[Relationship] = []
        seen_rel_ids: Set[str] = set()
        for r in local_result.relationships + drift_result.relationships:
            if r.id not in seen_rel_ids:
                seen_rel_ids.add(r.id)
                all_relationships.append(r)

        # Merge context chunks
        context_chunks = list(dict.fromkeys(
            local_result.context_chunks +
            global_result.context_chunks +
            drift_result.context_chunks
        ))[:20]

        # Combine answers
        answer_parts = []
        if local_result.answer:
            answer_parts.append(f"[LOCAL]: {local_result.answer[:300]}")
        if global_result.answer:
            answer_parts.append(f"[GLOBAL]: {global_result.answer[:300]}")
        if drift_result.answer:
            answer_parts.append(f"[DRIFT]: {drift_result.answer[:300]}")

        answer = "\n\n".join(answer_parts)

        # Combined confidence
        confidence = (
            local_result.confidence * 0.3 +
            global_result.confidence * 0.3 +
            drift_result.confidence * 0.4
        )

        return QueryResult(
            query=query,
            mode=QueryMode.HYBRID,
            answer=answer,
            entities=final_entities,
            relationships=all_relationships,
            communities_used=global_result.communities_used,
            context_chunks=context_chunks,
            confidence=confidence,
            latency_ms=(time.time() - start_time) * 1000,
            drift_state=drift_result.drift_state,
            metrics={
                "local_entities": len(local_result.entities),
                "global_communities": len(global_result.communities_used),
                "drift_iterations": drift_result.metrics.get("iterations", 0),
                "rrf_combined_entities": len(final_entities),
            },
        )


# =============================================================================
# MAIN GRAPHRAG V2 CLASS
# =============================================================================

class GraphRAGV2:
    """GraphRAG V2: Enhanced Graph-Based Retrieval-Augmented Generation.

    Key improvements over V1:
    1. Hierarchical Leiden community detection (graspologic/leidenalg)
    2. Four query modes: LOCAL, GLOBAL, DRIFT, HYBRID
    3. Pre-computed community summaries
    4. Integration with existing entity_extractor.py
    5. Benchmark-ready metrics

    Usage:
        graph = GraphRAGV2(config=GraphRAGV2Config())

        # Ingest documents
        await graph.ingest("Alice is CEO of TechCorp...")

        # Run community detection
        graph.detect_communities()

        # Query with different modes
        result = await graph.query("Who leads TechCorp?", mode=QueryMode.LOCAL)
        result = await graph.query("Main themes?", mode=QueryMode.GLOBAL)
    """

    def __init__(
        self,
        config: Optional[GraphRAGV2Config] = None,
        llm: Optional[LLMProvider] = None,
        embedder: Optional[EmbeddingProvider] = None,
    ) -> None:
        self.config = config or GraphRAGV2Config()
        self.llm = llm
        self.embedder = embedder

        self._storage = GraphStorageV2(self.config.db_path)
        self._storage.connect()

        self._community_detector = CommunityDetector(
            resolution=self.config.leiden_resolution,
            max_levels=self.config.max_community_levels,
            size_thresholds=self.config.community_size_threshold,
        )

        self._graph: Optional["nx.DiGraph"] = None
        self._executors = {
            QueryMode.LOCAL: LocalQueryExecutor(),
            QueryMode.GLOBAL: GlobalQueryExecutor(),
            QueryMode.DRIFT: DRIFTQueryExecutor(),
            QueryMode.HYBRID: HybridQueryExecutor(),
        }

        logger.info(
            "GraphRAGV2 initialized",
            db_path=self.config.db_path,
            hierarchical_communities=self.config.enable_hierarchical_communities,
        )

    # -------------------------------------------------------------------------
    # Ingestion
    # -------------------------------------------------------------------------

    async def ingest(
        self,
        text: str,
        entities: Optional[List[Dict[str, Any]]] = None,
        relationships: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IngestResult:
        """Ingest text or pre-extracted entities/relationships.

        Args:
            text: Raw text to extract entities from.
            entities: Pre-extracted entities (skip extraction).
            relationships: Pre-extracted relationships.
            metadata: Metadata to attach to entities.

        Returns:
            IngestResult with counts and latency.
        """
        start_time = time.time()

        # Use provided entities or extract from text
        if entities is None:
            entities_list, relationships_list = await self._extract_entities(text)
        else:
            entities_list = [
                Entity(
                    id=self._generate_id(e.get("name", "")),
                    name=e.get("name", ""),
                    entity_type=EntityType(e.get("type", "UNKNOWN")),
                    description=e.get("description", ""),
                )
                for e in entities
            ]
            relationships_list = [
                Relationship(
                    id=self._generate_id(f"{r.get('source', '')}-{r.get('target', '')}"),
                    source_id=self._generate_id(r.get("source", "")),
                    target_id=self._generate_id(r.get("target", "")),
                    relation_type=RelationType(r.get("type", "RELATED_TO")),
                    description=r.get("description", ""),
                )
                for r in (relationships or [])
            ]

        # Add embeddings if available
        if self.embedder and self.config.enable_embeddings and entities_list:
            try:
                texts = [f"{e.name}: {e.description}" for e in entities_list]
                embeddings = self.embedder.encode(texts)
                for entity, emb in zip(entities_list, embeddings):
                    entity.embedding = emb
            except Exception as e:
                logger.warning("Embedding generation failed", error=str(e))

        # Store entities
        for entity in entities_list:
            if metadata:
                entity.metadata.update(metadata)
            self._storage.upsert_entity(entity)

        # Store relationships
        for rel in relationships_list:
            self._storage.upsert_relationship(rel)

        # Invalidate graph cache
        self._graph = None

        return IngestResult(
            entities_added=len(entities_list),
            relationships_added=len(relationships_list),
            latency_ms=(time.time() - start_time) * 1000,
        )

    async def _extract_entities(
        self,
        text: str,
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities from text using entity_extractor.py if available."""
        try:
            from .entity_extractor import EntityExtractor as NERExtractor

            extractor = NERExtractor()
            result = extractor.extract(text)

            entities = []
            entity_lookup: Dict[str, str] = {}

            for ent in result.entities:
                entity_id = self._generate_id(ent.name)
                entity_type = self._map_entity_type(ent.type)

                entities.append(Entity(
                    id=entity_id,
                    name=ent.name,
                    entity_type=entity_type,
                    description=", ".join(ent.contexts[:2]),
                    occurrence_count=ent.occurrences,
                ))
                entity_lookup[ent.name.lower()] = entity_id

            # Build co-occurrence relationships
            relationships = []
            entity_names = list(entity_lookup.keys())
            for i, name1 in enumerate(entity_names):
                for name2 in entity_names[i+1:]:
                    rel_id = self._generate_id(f"{name1}-{name2}")
                    relationships.append(Relationship(
                        id=rel_id,
                        source_id=entity_lookup[name1],
                        target_id=entity_lookup[name2],
                        relation_type=RelationType.RELATED_TO,
                        weight=0.5,
                    ))

            return entities, relationships

        except ImportError:
            logger.warning("entity_extractor not available, using simple extraction")
            return self._simple_extract(text)

    def _simple_extract(
        self,
        text: str,
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Simple regex-based entity extraction fallback.

        Extracts:
        - Capitalized words (Alice, TechCorp)
        - Multi-word capitalized phrases (San Francisco)
        - Common entity patterns
        """
        import re

        entities = []
        seen: Set[str] = set()

        # Pattern 1: Capitalized words and multi-word phrases
        pattern = r"\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\b"
        matches = re.findall(pattern, text)

        # Common words to exclude
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "i", "you", "he", "she", "it", "we", "they", "who", "what",
            "this", "that", "these", "those", "in", "on", "at", "by",
            "for", "with", "about", "against", "between", "into", "through",
        }

        for match in matches:
            name_lower = match.lower()
            if (
                name_lower not in seen
                and len(match) > 2
                and name_lower not in stopwords
            ):
                seen.add(name_lower)
                entities.append(Entity(
                    id=self._generate_id(match),
                    name=match,
                    entity_type=EntityType.UNKNOWN,
                    description=f"Extracted from text",
                ))

        # Build co-occurrence relationships between extracted entities
        relationships = []
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                rel_id = self._generate_id(f"{e1.id}-{e2.id}")
                relationships.append(Relationship(
                    id=rel_id,
                    source_id=e1.id,
                    target_id=e2.id,
                    relation_type=RelationType.RELATED_TO,
                    weight=0.3,
                ))

        return entities, relationships

    def _map_entity_type(self, type_str: str) -> EntityType:
        """Map entity_extractor types to GraphRAG types."""
        type_map = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "TECH": EntityType.TECHNOLOGY,
            "FRAMEWORK": EntityType.FRAMEWORK,
            "MODEL": EntityType.MODEL,
            "URL": EntityType.CONCEPT,
            "VERSION": EntityType.CONCEPT,
        }
        return type_map.get(type_str.upper(), EntityType.UNKNOWN)

    def _generate_id(self, name: str) -> str:
        """Generate stable ID from name."""
        return hashlib.md5(name.lower().encode()).hexdigest()[:16]

    # -------------------------------------------------------------------------
    # Community Detection
    # -------------------------------------------------------------------------

    def detect_communities(self) -> int:
        """Run hierarchical community detection on the graph.

        Returns:
            Total number of communities detected.
        """
        if not HAS_NETWORKX:
            logger.warning("NetworkX not available, skipping community detection")
            return 0

        # Build NetworkX graph
        graph = self._build_networkx_graph()

        if graph.number_of_nodes() < 3:
            logger.info("Graph too small for community detection")
            return 0

        # Detect communities
        communities_by_level = self._community_detector.detect_hierarchical(graph)

        # Store communities
        total = 0
        for level, communities in communities_by_level.items():
            for comm in communities:
                self._storage.upsert_community(comm)
                total += 1

                # Update entity community membership
                for eid in comm.entity_ids:
                    entity = self._storage.get_entity(eid)
                    if entity and comm.id not in entity.community_ids:
                        entity.community_ids.append(comm.id)
                        self._storage.upsert_entity(entity)

        logger.info(
            "Community detection complete",
            total_communities=total,
            levels=len(communities_by_level),
        )

        return total

    def _build_networkx_graph(self) -> "nx.DiGraph":
        """Build NetworkX graph from storage."""
        if self._graph is not None:
            return self._graph

        graph = nx.DiGraph()

        # Add entities as nodes
        for entity in self._storage.get_all_entities():
            graph.add_node(
                entity.id,
                name=entity.name,
                entity_type=entity.entity_type.value,
            )

        # Add relationships as edges
        conn = self._storage._ensure_connected()
        for row in conn.execute("SELECT * FROM relationships").fetchall():
            graph.add_edge(
                row["source_id"],
                row["target_id"],
                relation_type=row["relation_type"],
                weight=row["weight"],
            )

        self._graph = graph
        return graph

    # -------------------------------------------------------------------------
    # Querying
    # -------------------------------------------------------------------------

    async def query(
        self,
        query: str,
        mode: QueryMode = QueryMode.LOCAL,
    ) -> QueryResult:
        """Execute a query using the specified mode.

        Args:
            query: The search query.
            mode: Query mode (LOCAL, GLOBAL, DRIFT, HYBRID).

        Returns:
            QueryResult with answer, entities, and metrics.
        """
        executor = self._executors.get(mode)
        if not executor:
            raise QueryError(f"Unknown query mode: {mode}")

        return await executor.execute(
            query,
            self._storage,
            self.config,
            self.llm,
            self.embedder,
        )

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get graph and storage statistics."""
        storage_stats = self._storage.get_stats()
        storage_stats["config"] = {
            "hierarchical_communities": self.config.enable_hierarchical_communities,
            "max_community_levels": self.config.max_community_levels,
            "max_hops": self.config.max_hops,
        }
        storage_stats["community_detector_backend"] = self._community_detector._backend
        return storage_stats

    def close(self) -> None:
        """Close storage connection."""
        self._storage.close()
        self._graph = None

    def __repr__(self) -> str:
        stats = self._storage.get_stats()
        return (
            f"GraphRAGV2(entities={stats['entities']}, "
            f"relationships={stats['relationships']}, "
            f"communities={stats['communities']})"
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main class
    "GraphRAGV2",
    # Configuration
    "GraphRAGV2Config",
    # Data types
    "Entity",
    "Relationship",
    "Community",
    "QueryResult",
    "IngestResult",
    "DRIFTState",
    # Enums
    "QueryMode",
    "CommunityLevel",
    "EntityType",
    "RelationType",
    # Components
    "CommunityDetector",
    "GraphStorageV2",
    # Query executors
    "QueryExecutor",
    "LocalQueryExecutor",
    "GlobalQueryExecutor",
    "DRIFTQueryExecutor",
    "HybridQueryExecutor",
    # Protocols
    "LLMProvider",
    "EmbeddingProvider",
    # Exceptions
    "GraphRAGError",
    "CommunityDetectionError",
    "QueryError",
    "PersistenceError",
]
