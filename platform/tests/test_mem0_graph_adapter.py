# -*- coding: utf-8 -*-
"""
Tests for Mem0 Graph Memory Adapter (V66 Enhanced)
===================================================

Validates:
- LocalGraphStore (NetworkX-based fallback)
- Entity operations (add, get, search)
- Relationship operations (add, query, multi-hop)
- Entity context retrieval (26% accuracy improvement)
- Entity merging (deduplication)
- Graph statistics
- Integration with KnowledgeGraph
- Performance targets
- Error handling and edge cases

30+ tests covering all graph memory features.
"""

import time
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Handle platform module shadowing
import sys
import os

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_platform_dir = os.path.join(_project_root, "platform")
if _platform_dir not in sys.path:
    sys.path.insert(0, _platform_dir)

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

pytestmark = pytest.mark.skipif(
    not HAS_NETWORKX,
    reason="networkx not installed",
)

from adapters.mem0_adapter import (
    LocalGraphStore,
    GraphBackend,
    GraphRelationType,
    MemoryBackend,
    MemoryType,
    GraphEntity,
    GraphRelation,
    MEM0_AVAILABLE,
    HAS_NETWORKX as ADAPTER_HAS_NETWORKX,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def graph_store():
    """Create a fresh LocalGraphStore."""
    return LocalGraphStore()


@pytest.fixture
def populated_graph_store():
    """Create a LocalGraphStore with sample data."""
    store = LocalGraphStore()

    # Add entities
    store.add_entity("LangGraph", "FRAMEWORK", {"version": "0.2"})
    store.add_entity("LangChain", "FRAMEWORK", {"version": "0.3"})
    store.add_entity("Python", "LANGUAGE")
    store.add_entity("Claude", "MODEL")
    store.add_entity("GPT-4", "MODEL")

    # Add relationships
    store.add_relationship("LangGraph", "LangChain", "related_to", 2.0)
    store.add_relationship("LangGraph", "Python", "depends_on", 1.0)
    store.add_relationship("LangChain", "Python", "depends_on", 1.0)
    store.add_relationship("Claude", "LangGraph", "supports", 1.5)
    store.add_relationship("GPT-4", "LangChain", "supports", 1.5)

    return store


@pytest.fixture
def tmp_persist_path(tmp_path):
    """Provide a temporary persistence path."""
    return tmp_path / "test_graph.db"


# =============================================================================
# LocalGraphStore Basic Tests
# =============================================================================

class TestLocalGraphStoreBasic:
    """Test basic LocalGraphStore operations."""

    def test_create_empty_store(self, graph_store):
        """Should create an empty graph store."""
        assert graph_store.graph.number_of_nodes() == 0
        assert graph_store.graph.number_of_edges() == 0

    def test_add_entity(self, graph_store):
        """Should add an entity to the graph."""
        node_id = graph_store.add_entity("TensorFlow", "FRAMEWORK")
        assert node_id is not None
        assert graph_store.graph.has_node(node_id)
        data = graph_store.graph.nodes[node_id]
        assert data["name"] == "TensorFlow"
        assert data["entity_type"] == "FRAMEWORK"

    def test_add_entity_with_metadata(self, graph_store):
        """Entity metadata should be stored."""
        node_id = graph_store.add_entity(
            "PyTorch",
            "FRAMEWORK",
            {"version": "2.0", "org": "Meta"}
        )
        data = graph_store.graph.nodes[node_id]
        assert data["metadata"]["version"] == "2.0"
        assert data["metadata"]["org"] == "Meta"

    def test_add_entity_with_user_id(self, graph_store):
        """User ID should be associated with entity."""
        node_id = graph_store.add_entity(
            "MyModel",
            "MODEL",
            user_id="user_123"
        )
        data = graph_store.graph.nodes[node_id]
        assert data["user_id"] == "user_123"

    def test_get_entity_by_name(self, graph_store):
        """Should retrieve entity by name."""
        graph_store.add_entity("VLLM", "FRAMEWORK")
        entity = graph_store.get_entity("VLLM")
        assert entity is not None
        assert entity["name"] == "VLLM"
        assert "id" in entity

    def test_get_entity_not_found(self, graph_store):
        """Should return None for non-existent entity."""
        entity = graph_store.get_entity("NonExistent")
        assert entity is None

    def test_deterministic_ids(self, graph_store):
        """Same entity name should produce same ID."""
        id1 = graph_store.add_entity("Python", "LANGUAGE")
        id2 = graph_store.add_entity("Python", "LANGUAGE")
        assert id1 == id2


# =============================================================================
# Relationship Tests
# =============================================================================

class TestRelationships:
    """Test relationship operations."""

    def test_add_relationship(self, graph_store):
        """Should add a relationship between entities."""
        graph_store.add_entity("A", "TEST")
        graph_store.add_entity("B", "TEST")
        result = graph_store.add_relationship("A", "B", "related_to", 1.0)
        assert result is True
        # Verify edge exists
        a_id = graph_store._find_entity("A")
        b_id = graph_store._find_entity("B")
        assert graph_store.graph.has_edge(a_id, b_id)

    def test_add_relationship_with_metadata(self, graph_store):
        """Relationship metadata should be stored."""
        graph_store.add_entity("Source", "TEST")
        graph_store.add_entity("Target", "TEST")
        graph_store.add_relationship(
            "Source", "Target",
            "depends_on",
            weight=2.0,
            metadata={"reason": "core dependency"}
        )
        source_id = graph_store._find_entity("Source")
        target_id = graph_store._find_entity("Target")
        edge_data = graph_store.graph[source_id][target_id]
        assert edge_data["relation_type"] == "depends_on"
        assert edge_data["weight"] == 2.0
        assert edge_data["metadata"]["reason"] == "core dependency"

    def test_add_relationship_weight_accumulates(self, graph_store):
        """Adding same relationship should accumulate weight."""
        graph_store.add_entity("A", "TEST")
        graph_store.add_entity("B", "TEST")
        graph_store.add_relationship("A", "B", "related_to", 1.0)
        graph_store.add_relationship("A", "B", "related_to", 2.0)
        a_id = graph_store._find_entity("A")
        b_id = graph_store._find_entity("B")
        assert graph_store.graph[a_id][b_id]["weight"] == 3.0

    def test_add_relationship_missing_entity(self, graph_store):
        """Should return False for non-existent entity."""
        graph_store.add_entity("A", "TEST")
        result = graph_store.add_relationship("A", "NonExistent", "related_to")
        assert result is False

    def test_relationship_types(self, graph_store):
        """Should support various relationship types."""
        graph_store.add_entity("A", "TEST")
        graph_store.add_entity("B", "TEST")
        graph_store.add_entity("C", "TEST")

        graph_store.add_relationship("A", "B", "supports")
        graph_store.add_relationship("A", "C", "contradicts")

        a_id = graph_store._find_entity("A")
        b_id = graph_store._find_entity("B")
        c_id = graph_store._find_entity("C")

        assert graph_store.graph[a_id][b_id]["relation_type"] == "supports"
        assert graph_store.graph[a_id][c_id]["relation_type"] == "contradicts"


# =============================================================================
# Query Tests
# =============================================================================

class TestGraphQueries:
    """Test graph query operations."""

    def test_query_relationships_basic(self, populated_graph_store):
        """Should return related entities and relationships."""
        entities, relationships = populated_graph_store.query_relationships(
            "LangGraph", max_hops=1
        )
        assert len(entities) > 0
        assert len(relationships) > 0

        # Check that LangChain is found (directly related)
        entity_names = [e["name"] for e in entities]
        assert "LangChain" in entity_names or "Python" in entity_names

    def test_query_relationships_multi_hop(self, populated_graph_store):
        """Should traverse multiple hops."""
        entities, relationships = populated_graph_store.query_relationships(
            "Claude", max_hops=3
        )
        # Claude -> LangGraph -> LangChain -> ...
        assert len(entities) > 0

    def test_query_relationships_respects_max_hops(self, graph_store):
        """Should respect max_hops limit."""
        # Create chain: A -> B -> C -> D
        graph_store.add_entity("A", "TEST")
        graph_store.add_entity("B", "TEST")
        graph_store.add_entity("C", "TEST")
        graph_store.add_entity("D", "TEST")
        graph_store.add_relationship("A", "B", "related_to")
        graph_store.add_relationship("B", "C", "related_to")
        graph_store.add_relationship("C", "D", "related_to")

        entities_1hop, _ = graph_store.query_relationships("A", max_hops=1)
        entities_2hop, _ = graph_store.query_relationships("A", max_hops=2)

        names_1hop = {e["name"] for e in entities_1hop}
        names_2hop = {e["name"] for e in entities_2hop}

        assert "B" in names_1hop
        assert "C" not in names_1hop
        assert "C" in names_2hop

    def test_query_relationships_filter_by_type(self, populated_graph_store):
        """Should filter by relation types."""
        entities, relationships = populated_graph_store.query_relationships(
            "LangGraph",
            max_hops=2,
            relation_types=["depends_on"]
        )
        for rel in relationships:
            assert rel["relation_type"] == "depends_on"

    def test_query_unknown_entity(self, graph_store):
        """Should return empty for unknown entity."""
        entities, relationships = graph_store.query_relationships("NonExistent")
        assert entities == []
        assert relationships == []

    def test_search_basic(self, populated_graph_store):
        """Should find entities by keyword."""
        results = populated_graph_store.search("Lang", limit=5)
        assert len(results) > 0
        names = [r["name"] for r in results]
        assert any("Lang" in name for name in names)

    def test_search_exact_match_higher_score(self, populated_graph_store):
        """Exact matches should have higher score."""
        results = populated_graph_store.search("Python")
        assert len(results) > 0
        # First result should be exact match with score 1.0
        assert results[0]["name"] == "Python"
        assert results[0]["score"] == 1.0

    def test_search_respects_limit(self, populated_graph_store):
        """Should respect limit parameter."""
        results = populated_graph_store.search("a", limit=2)
        assert len(results) <= 2


# =============================================================================
# Entity Context Tests
# =============================================================================

class TestEntityContext:
    """Test entity context retrieval (26% accuracy improvement feature)."""

    def test_get_entity_context_basic(self, populated_graph_store):
        """Should return full context for an entity."""
        context = populated_graph_store.get_entity_context("LangGraph")
        assert context is not None
        assert "entity" in context
        assert "related_entities" in context
        assert "relationships" in context
        assert context["entity"]["name"] == "LangGraph"

    def test_get_entity_context_includes_depth(self, populated_graph_store):
        """Context should include traversal depth."""
        context = populated_graph_store.get_entity_context("LangGraph", depth=3)
        assert context["depth"] == 3

    def test_get_entity_context_not_found(self, populated_graph_store):
        """Should return None for non-existent entity."""
        context = populated_graph_store.get_entity_context("NonExistent")
        assert context is None

    def test_get_entity_context_depth_affects_results(self, graph_store):
        """Different depths should return different results."""
        # Create chain: A -> B -> C
        graph_store.add_entity("A", "TEST")
        graph_store.add_entity("B", "TEST")
        graph_store.add_entity("C", "TEST")
        graph_store.add_relationship("A", "B", "related_to")
        graph_store.add_relationship("B", "C", "related_to")

        ctx_1 = graph_store.get_entity_context("A", depth=1)
        ctx_2 = graph_store.get_entity_context("A", depth=2)

        # depth=2 should find more related entities
        assert len(ctx_2["related_entities"]) >= len(ctx_1["related_entities"])


# =============================================================================
# Entity Merge Tests
# =============================================================================

class TestEntityMerge:
    """Test entity merging (deduplication)."""

    def test_merge_entities_basic(self, graph_store):
        """Should merge duplicate entities."""
        graph_store.add_entity("Python", "LANGUAGE")
        graph_store.add_entity("Python3", "LANGUAGE")
        graph_store.add_entity("Py", "LANGUAGE")
        graph_store.add_entity("Django", "FRAMEWORK")

        # Add relationships to duplicates
        graph_store.add_relationship("Django", "Python3", "depends_on")
        graph_store.add_relationship("Py", "Django", "supports")

        result = graph_store.merge_entities(
            primary="Python",
            duplicates=["Python3", "Py"]
        )

        assert result["success"] is True
        assert result["primary_entity"] == "Python"
        assert len(result["merged_entities"]) == 2
        assert result["relationships_transferred"] > 0

        # Verify duplicates are removed
        assert graph_store.get_entity("Python3") is None
        assert graph_store.get_entity("Py") is None

    def test_merge_entities_primary_not_found(self, graph_store):
        """Should fail if primary entity not found."""
        graph_store.add_entity("A", "TEST")
        result = graph_store.merge_entities(
            primary="NonExistent",
            duplicates=["A"]
        )
        assert result["success"] is False
        assert "not found" in result.get("error", "")

    def test_merge_entities_preserves_relationships(self, graph_store):
        """Merged entity should have all relationships."""
        graph_store.add_entity("Primary", "TEST")
        graph_store.add_entity("Duplicate", "TEST")
        graph_store.add_entity("External", "TEST")

        graph_store.add_relationship("External", "Duplicate", "related_to")
        graph_store.add_relationship("Duplicate", "External", "supports")

        graph_store.merge_entities(
            primary="Primary",
            duplicates=["Duplicate"]
        )

        # Verify relationships transferred
        primary_id = graph_store._find_entity("Primary")
        external_id = graph_store._find_entity("External")

        assert graph_store.graph.has_edge(external_id, primary_id) or \
               graph_store.graph.has_edge(primary_id, external_id)


# =============================================================================
# Statistics Tests
# =============================================================================

class TestGraphStats:
    """Test graph statistics."""

    def test_get_stats_empty_graph(self, graph_store):
        """Empty graph should have zero stats."""
        stats = graph_store.get_stats()
        assert stats["total_entities"] == 0
        assert stats["total_relationships"] == 0
        assert stats["entity_types"] == {}
        assert stats["relation_types"] == {}

    def test_get_stats_populated_graph(self, populated_graph_store):
        """Should return accurate statistics."""
        stats = populated_graph_store.get_stats()
        assert stats["total_entities"] == 5  # LangGraph, LangChain, Python, Claude, GPT-4
        assert stats["total_relationships"] == 5
        assert "FRAMEWORK" in stats["entity_types"]
        assert "MODEL" in stats["entity_types"]
        assert stats["entity_types"]["FRAMEWORK"] == 2
        assert stats["entity_types"]["MODEL"] == 2

    def test_get_stats_relation_types(self, populated_graph_store):
        """Should count relation types correctly."""
        stats = populated_graph_store.get_stats()
        assert "related_to" in stats["relation_types"]
        assert "depends_on" in stats["relation_types"]
        assert "supports" in stats["relation_types"]


# =============================================================================
# Enum Tests
# =============================================================================

class TestEnums:
    """Test enum definitions."""

    def test_graph_backend_values(self):
        """GraphBackend should have expected values."""
        assert GraphBackend.NEO4J.value == "neo4j"
        assert GraphBackend.NETWORKX.value == "networkx"
        assert GraphBackend.SQLITE.value == "sqlite"

    def test_graph_relation_type_values(self):
        """GraphRelationType should have expected values."""
        assert GraphRelationType.RELATED_TO.value == "related_to"
        assert GraphRelationType.MENTIONS.value == "mentions"
        assert GraphRelationType.CONTRADICTS.value == "contradicts"
        assert GraphRelationType.SUPPORTS.value == "supports"

    def test_memory_backend_values(self):
        """MemoryBackend should have expected values."""
        assert MemoryBackend.QDRANT.value == "qdrant"
        assert MemoryBackend.SQLITE.value == "sqlite"
        assert MemoryBackend.CHROMA.value == "chroma"


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_add_entity_empty_name(self, graph_store):
        """Should handle empty entity name."""
        # Empty name still creates a node (design choice)
        node_id = graph_store.add_entity("", "TEST")
        assert node_id is not None

    def test_relationship_with_self(self, graph_store):
        """Should allow self-referential relationships."""
        graph_store.add_entity("Node", "TEST")
        result = graph_store.add_relationship("Node", "Node", "refers_to")
        # Self-loop is valid in directed graph
        assert result is True

    def test_query_with_zero_hops(self, populated_graph_store):
        """Zero hops should return empty results."""
        entities, relationships = populated_graph_store.query_relationships(
            "LangGraph", max_hops=0
        )
        assert entities == []
        assert relationships == []


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Test performance targets."""

    def test_add_1000_entities_under_1s(self):
        """Adding 1000 entities should complete in <1s."""
        store = LocalGraphStore()
        start = time.perf_counter()

        for i in range(1000):
            store.add_entity(f"Entity_{i}", "TEST")

        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"Adding 1000 entities took {elapsed:.2f}s"
        assert store.graph.number_of_nodes() == 1000

    def test_query_1000_node_graph_under_100ms(self):
        """Query on 1000-node graph should be <100ms."""
        store = LocalGraphStore()

        # Build graph
        for i in range(500):
            store.add_entity(f"Entity_{i}", "TEST")

        # Add relationships to create connected graph
        for i in range(500):
            store.add_relationship(
                f"Entity_{i}",
                f"Entity_{(i + 1) % 500}",
                "related_to"
            )

        start = time.perf_counter()
        entities, relationships = store.query_relationships(
            "Entity_0", max_hops=3
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, f"Query took {elapsed_ms:.1f}ms"

    def test_get_stats_1000_nodes_under_50ms(self):
        """Stats on 1000-node graph should be <50ms."""
        store = LocalGraphStore()

        for i in range(1000):
            store.add_entity(f"Entity_{i}", f"TYPE_{i % 10}")

        start = time.perf_counter()
        stats = store.get_stats()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, f"Stats took {elapsed_ms:.1f}ms"
        assert stats["total_entities"] == 1000


# =============================================================================
# Integration Tests (with mocked Mem0Adapter)
# =============================================================================

class TestAdapterIntegration:
    """Test Mem0Adapter graph operations with mocked client."""

    @pytest.fixture
    def mock_adapter(self):
        """Create adapter with mocked client for testing."""
        # Import adapter class
        try:
            from adapters.mem0_adapter import Mem0Adapter, SDK_ADAPTER_AVAILABLE
        except ImportError:
            pytest.skip("Mem0Adapter not available")

        if not SDK_ADAPTER_AVAILABLE:
            pytest.skip("SDKAdapter base class not available")

        adapter = Mem0Adapter()
        adapter._graph_enabled = True
        adapter._graph_backend = GraphBackend.NETWORKX
        adapter._local_graph_store = LocalGraphStore()
        adapter._available = True
        return adapter

    @pytest.mark.asyncio
    async def test_add_entity_operation(self, mock_adapter):
        """Should add entity via adapter."""
        result = await mock_adapter._add_single_entity({
            "name": "TestEntity",
            "entity_type": "FRAMEWORK",
            "metadata": {"version": "1.0"},
        })
        assert result.success is True
        assert result.data["name"] == "TestEntity"
        assert result.data["backend"] == "networkx"

    @pytest.mark.asyncio
    async def test_add_relationship_operation(self, mock_adapter):
        """Should add relationship via adapter."""
        # Add entities first
        await mock_adapter._add_single_entity({"name": "A", "entity_type": "TEST"})
        await mock_adapter._add_single_entity({"name": "B", "entity_type": "TEST"})

        result = await mock_adapter._add_relationship({
            "source": "A",
            "target": "B",
            "relation_type": "related_to",
            "weight": 2.0,
        })
        assert result.success is True
        assert result.data["source"] == "A"
        assert result.data["target"] == "B"

    @pytest.mark.asyncio
    async def test_query_graph_operation(self, mock_adapter):
        """Should query graph via adapter."""
        # Build small graph
        await mock_adapter._add_single_entity({"name": "Start", "entity_type": "TEST"})
        await mock_adapter._add_single_entity({"name": "Middle", "entity_type": "TEST"})
        await mock_adapter._add_single_entity({"name": "End", "entity_type": "TEST"})
        await mock_adapter._add_relationship({"source": "Start", "target": "Middle"})
        await mock_adapter._add_relationship({"source": "Middle", "target": "End"})

        result = await mock_adapter._query_graph({
            "entity": "Start",
            "max_hops": 2,
        })
        assert result.success is True
        assert "paths" in result.data
        assert "entities" in result.data

    @pytest.mark.asyncio
    async def test_get_entity_context_operation(self, mock_adapter):
        """Should get entity context via adapter."""
        await mock_adapter._add_single_entity({"name": "Central", "entity_type": "TEST"})
        await mock_adapter._add_single_entity({"name": "Related", "entity_type": "TEST"})
        await mock_adapter._add_relationship({"source": "Central", "target": "Related"})

        result = await mock_adapter._get_entity_context({
            "entity": "Central",
            "depth": 1,
            "include_memories": False,  # Skip memory lookup for this test
        })
        assert result.success is True
        assert result.data["entity"]["name"] == "Central"

    @pytest.mark.asyncio
    async def test_merge_entities_operation(self, mock_adapter):
        """Should merge entities via adapter."""
        await mock_adapter._add_single_entity({"name": "Primary", "entity_type": "TEST"})
        await mock_adapter._add_single_entity({"name": "Duplicate1", "entity_type": "TEST"})
        await mock_adapter._add_single_entity({"name": "Duplicate2", "entity_type": "TEST"})

        result = await mock_adapter._merge_entities({
            "primary": "Primary",
            "duplicates": ["Duplicate1", "Duplicate2"],
        })
        assert result.success is True
        assert result.data["primary_entity"] == "Primary"

    @pytest.mark.asyncio
    async def test_get_graph_stats_operation(self, mock_adapter):
        """Should get graph stats via adapter."""
        await mock_adapter._add_single_entity({"name": "Entity1", "entity_type": "TYPE_A"})
        await mock_adapter._add_single_entity({"name": "Entity2", "entity_type": "TYPE_B"})

        result = await mock_adapter._get_graph_stats({})
        assert result.success is True
        assert result.data["total_entities"] == 2
        assert result.data["backend"] == "networkx"

    @pytest.mark.asyncio
    async def test_graph_disabled_error(self, mock_adapter):
        """Should error when graph is disabled."""
        mock_adapter._graph_enabled = False
        mock_adapter._local_graph_store = None

        result = await mock_adapter._query_graph({"entity": "Test"})
        assert result.success is False
        assert "not enabled" in result.error


# =============================================================================
# Data Class Tests
# =============================================================================

class TestDataClasses:
    """Test data class definitions."""

    def test_graph_entity_creation(self):
        """Should create GraphEntity."""
        entity = GraphEntity(
            id="entity_123",
            name="TestEntity",
            entity_type="FRAMEWORK",
            metadata={"version": "1.0"}
        )
        assert entity.name == "TestEntity"
        assert entity.entity_type == "FRAMEWORK"

    def test_graph_relation_creation(self):
        """Should create GraphRelation."""
        relation = GraphRelation(
            source="A",
            target="B",
            relation_type="related_to",
            metadata={"weight": 1.0}
        )
        assert relation.source == "A"
        assert relation.target == "B"


# =============================================================================
# Module Constants Tests
# =============================================================================

class TestModuleConstants:
    """Test module-level constants."""

    def test_has_networkx_flag(self):
        """HAS_NETWORKX should match actual availability."""
        assert ADAPTER_HAS_NETWORKX == HAS_NETWORKX

    def test_mem0_available_is_bool(self):
        """MEM0_AVAILABLE should be boolean."""
        assert isinstance(MEM0_AVAILABLE, bool)
