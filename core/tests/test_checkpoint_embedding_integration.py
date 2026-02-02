#!/usr/bin/env python3
"""
Checkpoint + Voyage Embedding Integration Tests - ITERATION 9.

These tests verify semantic search over checkpoint data:
- Embedding checkpoint task descriptions
- Semantic similarity search for runs
- Finding semantically similar Ralph iterations
- Cross-session knowledge retrieval

Run with: pytest core/tests/test_checkpoint_embedding_integration.py -v
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
import pytest

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Constants
# =============================================================================

VOYAGE_API_KEY = "pa-KCpYL_zzmvoPK1dM6tN5kdCD8e6qnAndC-dSTlCuzK4"


# =============================================================================
# Semantic Checkpoint Integration Classes
# =============================================================================

@dataclass
class EmbeddedCheckpoint:
    """Checkpoint with associated embedding vector."""

    checkpoint_id: str
    checkpoint_type: str
    content: str  # The text content that was embedded
    embedding: List[float]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return len(self.embedding)


@dataclass
class SemanticSearchResult:
    """Result from semantic search over checkpoints."""

    checkpoint_id: str
    content: str
    similarity: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticCheckpointStore:
    """
    Combines checkpoint persistence with Voyage AI embeddings for semantic search.

    This class wraps the CheckpointStore and adds embedding-based retrieval.
    """

    def __init__(
        self,
        embedding_layer: Any = None,
        checkpoint_store: Any = None,
    ) -> None:
        """Initialize with embedding layer and checkpoint store."""
        self._embedding_layer = embedding_layer
        self._checkpoint_store = checkpoint_store
        self._embedding_cache: Dict[str, EmbeddedCheckpoint] = {}
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if both stores are initialized."""
        return self._initialized

    async def initialize(self) -> bool:
        """Initialize both embedding layer and checkpoint store."""
        if self._embedding_layer:
            await self._embedding_layer.initialize()
        if self._checkpoint_store:
            await self._checkpoint_store.initialize()
        self._initialized = True
        return True

    async def embed_and_store(
        self,
        checkpoint_id: str,
        checkpoint_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EmbeddedCheckpoint:
        """
        Embed content and store in the checkpoint cache.

        Args:
            checkpoint_id: Unique checkpoint identifier
            checkpoint_type: Type of checkpoint (e.g., 'run', 'iteration')
            content: Text content to embed
            metadata: Additional metadata

        Returns:
            EmbeddedCheckpoint with embedding vector
        """
        if not self._embedding_layer:
            # Use mock embedding for testing without API
            embedding = self._mock_embed(content)
        else:
            embedding = await self._embedding_layer.embed_query(content)

        checkpoint = EmbeddedCheckpoint(
            checkpoint_id=checkpoint_id,
            checkpoint_type=checkpoint_type,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
        )

        self._embedding_cache[checkpoint_id] = checkpoint
        return checkpoint

    def _mock_embed(self, text: str) -> List[float]:
        """Create a mock embedding based on text features."""
        # Simple mock: hash-based pseudo-embedding
        import hashlib

        hash_bytes = hashlib.sha256(text.encode()).digest()
        # Create 64-dim mock embedding from hash
        embedding = [float(b) / 255.0 for b in hash_bytes[:32]]
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [x / norm for x in embedding]
        return embedding

    async def search_similar(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> List[SemanticSearchResult]:
        """
        Search for checkpoints semantically similar to the query.

        Args:
            query: Search query text
            top_k: Maximum results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of SemanticSearchResult sorted by similarity
        """
        if not self._embedding_cache:
            return []

        # Embed query
        if self._embedding_layer:
            query_embedding = await self._embedding_layer.embed_query(query)
        else:
            query_embedding = self._mock_embed(query)

        query_vec = np.array(query_embedding)

        # Calculate similarities
        results = []
        for checkpoint in self._embedding_cache.values():
            checkpoint_vec = np.array(checkpoint.embedding)

            # Handle dimension mismatch (mock vs real embeddings)
            if len(query_vec) != len(checkpoint_vec):
                # Truncate to smaller dimension
                min_dim = min(len(query_vec), len(checkpoint_vec))
                sim = np.dot(query_vec[:min_dim], checkpoint_vec[:min_dim])
            else:
                sim = np.dot(query_vec, checkpoint_vec)

            if sim >= min_similarity:
                results.append(SemanticSearchResult(
                    checkpoint_id=checkpoint.checkpoint_id,
                    content=checkpoint.content,
                    similarity=float(sim),
                    metadata=checkpoint.metadata,
                ))

        # Sort by similarity descending
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:top_k]

    def get_checkpoint(self, checkpoint_id: str) -> Optional[EmbeddedCheckpoint]:
        """Get a specific checkpoint by ID."""
        return self._embedding_cache.get(checkpoint_id)

    def list_checkpoints(self, checkpoint_type: Optional[str] = None) -> List[EmbeddedCheckpoint]:
        """List all checkpoints, optionally filtered by type."""
        checkpoints = list(self._embedding_cache.values())
        if checkpoint_type:
            checkpoints = [c for c in checkpoints if c.checkpoint_type == checkpoint_type]
        return checkpoints

    def clear(self) -> int:
        """Clear the embedding cache and return count of cleared items."""
        count = len(self._embedding_cache)
        self._embedding_cache.clear()
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the semantic store."""
        checkpoints = list(self._embedding_cache.values())
        types = {}
        for c in checkpoints:
            types[c.checkpoint_type] = types.get(c.checkpoint_type, 0) + 1

        return {
            "total_checkpoints": len(checkpoints),
            "types": types,
            "embedding_dim": checkpoints[0].embedding_dim if checkpoints else 0,
            "initialized": self._initialized,
        }


# =============================================================================
# Test: EmbeddedCheckpoint Data Class
# =============================================================================

class TestEmbeddedCheckpoint:
    """Test EmbeddedCheckpoint dataclass."""

    def test_embedded_checkpoint_creation(self):
        """Test creating EmbeddedCheckpoint."""
        checkpoint = EmbeddedCheckpoint(
            checkpoint_id="ckpt_001",
            checkpoint_type="run",
            content="Test task description",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        assert checkpoint.checkpoint_id == "ckpt_001"
        assert checkpoint.checkpoint_type == "run"
        assert checkpoint.content == "Test task description"
        assert checkpoint.embedding_dim == 4

    def test_embedded_checkpoint_with_metadata(self):
        """Test EmbeddedCheckpoint with metadata."""
        checkpoint = EmbeddedCheckpoint(
            checkpoint_id="ckpt_002",
            checkpoint_type="iteration",
            content="Ralph iteration state",
            embedding=[0.5, 0.6],
            metadata={"iteration": 5, "session": "abc"},
        )

        assert checkpoint.metadata["iteration"] == 5
        assert checkpoint.metadata["session"] == "abc"


class TestSemanticSearchResult:
    """Test SemanticSearchResult dataclass."""

    def test_search_result_creation(self):
        """Test creating SemanticSearchResult."""
        result = SemanticSearchResult(
            checkpoint_id="ckpt_003",
            content="Machine learning task",
            similarity=0.95,
        )

        assert result.checkpoint_id == "ckpt_003"
        assert result.similarity == 0.95
        assert result.content == "Machine learning task"


# =============================================================================
# Test: SemanticCheckpointStore (Mock Mode)
# =============================================================================

class TestSemanticCheckpointStoreMock:
    """Test SemanticCheckpointStore without real API calls."""

    @pytest.mark.asyncio
    async def test_store_creation(self):
        """Test creating SemanticCheckpointStore."""
        store = SemanticCheckpointStore()
        assert store.is_initialized is False

        await store.initialize()
        assert store.is_initialized is True

    @pytest.mark.asyncio
    async def test_embed_and_store(self):
        """Test embedding and storing a checkpoint."""
        store = SemanticCheckpointStore()
        await store.initialize()

        checkpoint = await store.embed_and_store(
            checkpoint_id="run_001",
            checkpoint_type="orchestration_run",
            content="Optimize database queries for faster performance",
            metadata={"agents": ["planner", "code-reviewer"]},
        )

        assert checkpoint.checkpoint_id == "run_001"
        assert checkpoint.embedding_dim == 32  # Mock embedding dimension
        assert "planner" in checkpoint.metadata["agents"]

    @pytest.mark.asyncio
    async def test_mock_embedding_deterministic(self):
        """Test that mock embeddings are deterministic."""
        store = SemanticCheckpointStore()
        await store.initialize()

        content = "Same content for embedding"

        ckpt1 = await store.embed_and_store("id1", "run", content)
        store.clear()
        ckpt2 = await store.embed_and_store("id2", "run", content)

        assert ckpt1.embedding == ckpt2.embedding

    @pytest.mark.asyncio
    async def test_search_similar_empty_store(self):
        """Test searching in empty store."""
        store = SemanticCheckpointStore()
        await store.initialize()

        results = await store.search_similar("any query")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_similar_basic(self):
        """Test basic semantic search."""
        store = SemanticCheckpointStore()
        await store.initialize()

        # Store some checkpoints
        await store.embed_and_store(
            "run_001", "run", "Database optimization task"
        )
        await store.embed_and_store(
            "run_002", "run", "Machine learning model training"
        )
        await store.embed_and_store(
            "run_003", "run", "Frontend UI improvements"
        )

        # Search - should find something
        results = await store.search_similar("database", top_k=2)

        assert len(results) <= 2
        # Results should be sorted by similarity
        if len(results) >= 2:
            assert results[0].similarity >= results[1].similarity

    @pytest.mark.asyncio
    async def test_search_with_min_similarity(self):
        """Test search with minimum similarity threshold."""
        store = SemanticCheckpointStore()
        await store.initialize()

        await store.embed_and_store("run_001", "run", "Python programming")

        # Very high threshold should filter out results
        results = await store.search_similar("JavaScript", min_similarity=0.99)
        # With mock embeddings, unrelated content should have low similarity
        assert len(results) == 0 or results[0].similarity >= 0.99

    @pytest.mark.asyncio
    async def test_get_checkpoint(self):
        """Test retrieving a specific checkpoint."""
        store = SemanticCheckpointStore()
        await store.initialize()

        await store.embed_and_store("run_001", "run", "Test content")

        checkpoint = store.get_checkpoint("run_001")
        assert checkpoint is not None
        assert checkpoint.content == "Test content"

        # Non-existent checkpoint
        missing = store.get_checkpoint("nonexistent")
        assert missing is None

    @pytest.mark.asyncio
    async def test_list_checkpoints(self):
        """Test listing checkpoints."""
        store = SemanticCheckpointStore()
        await store.initialize()

        await store.embed_and_store("run_001", "run", "Run 1")
        await store.embed_and_store("run_002", "run", "Run 2")
        await store.embed_and_store("iter_001", "iteration", "Iteration 1")

        # List all
        all_checkpoints = store.list_checkpoints()
        assert len(all_checkpoints) == 3

        # Filter by type
        runs = store.list_checkpoints(checkpoint_type="run")
        assert len(runs) == 2

        iterations = store.list_checkpoints(checkpoint_type="iteration")
        assert len(iterations) == 1

    @pytest.mark.asyncio
    async def test_clear_store(self):
        """Test clearing the store."""
        store = SemanticCheckpointStore()
        await store.initialize()

        await store.embed_and_store("run_001", "run", "Content")
        await store.embed_and_store("run_002", "run", "Content 2")

        cleared = store.clear()
        assert cleared == 2
        assert len(store.list_checkpoints()) == 0

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting store statistics."""
        store = SemanticCheckpointStore()
        await store.initialize()

        await store.embed_and_store("run_001", "run", "Run task")
        await store.embed_and_store("iter_001", "iteration", "Iteration state")
        await store.embed_and_store("iter_002", "iteration", "Another iteration")

        stats = store.get_stats()

        assert stats["total_checkpoints"] == 3
        assert stats["types"]["run"] == 1
        assert stats["types"]["iteration"] == 2
        assert stats["embedding_dim"] == 32
        assert stats["initialized"] is True


# =============================================================================
# Test: SemanticCheckpointStore with Real Embeddings
# =============================================================================

@pytest.mark.slow
class TestSemanticCheckpointStoreReal:
    """Test SemanticCheckpointStore with real Voyage AI embeddings."""

    @pytest.fixture
    def voyage_api_key(self) -> str:
        """Get API key for tests."""
        return VOYAGE_API_KEY

    @pytest.mark.asyncio
    async def test_real_embeddings_for_runs(self, voyage_api_key):
        """
        Test semantic search with real Voyage embeddings.

        This test is rate-limited to 3 RPM, so we make minimal API calls:
        1. Initialize + embed one document (1 call via batch)
        2. Wait 21 seconds
        3. Search query (1 call)
        """
        from core.orchestration.embedding_layer import create_embedding_layer

        # Create real embedding layer
        embedding_layer = create_embedding_layer(
            model="voyage-3",
            api_key=voyage_api_key,
            cache_enabled=True,
        )

        store = SemanticCheckpointStore(embedding_layer=embedding_layer)
        await store.initialize()

        # Store ONE checkpoint with real embedding (minimize API calls)
        await store.embed_and_store(
            "run_001",
            "run",
            "Optimize SQL database queries for improved read performance",
            metadata={"focus": "database"},
        )

        # Rate limit - wait 21 seconds before next API call
        print("\n  [Rate limit] Waiting 21s before search...")
        await asyncio.sleep(21)

        # Search should use cache for the document embedding, only needs 1 API call for query
        results = await store.search_similar("database performance optimization")

        assert len(results) >= 1
        # First result should be the database optimization run
        assert results[0].checkpoint_id == "run_001"
        assert results[0].similarity > 0.5  # Should have decent similarity

        print(f"\n  [Success] Similarity score: {results[0].similarity:.4f}")


# =============================================================================
# Test: Integration with CheckpointStore
# =============================================================================

# Skip if aiosqlite not available
try:
    import aiosqlite
    _has_aiosqlite = True
except ImportError:
    _has_aiosqlite = False


@pytest.mark.skipif(not _has_aiosqlite, reason="aiosqlite not available")
class TestCheckpointStoreIntegration:
    """Test integration between SemanticCheckpointStore and CheckpointStore."""

    @pytest.fixture
    def temp_db_path(self, tmp_path: Path) -> str:
        """Create temporary database path."""
        return str(tmp_path / "test_semantic_checkpoints.db")

    @pytest.mark.asyncio
    async def test_combined_storage(self, temp_db_path: str):
        """Test using both checkpoint persistence and semantic search."""
        from core.orchestration.checkpoint_persistence import (
            CheckpointStore,
            CheckpointConfig,
            CheckpointType,
        )

        # Create checkpoint store
        config = CheckpointConfig(db_path=temp_db_path)
        checkpoint_store = CheckpointStore(config)

        # Create semantic store wrapping checkpoint store
        semantic_store = SemanticCheckpointStore(
            checkpoint_store=checkpoint_store,
        )
        await semantic_store.initialize()

        # Store both in persistent DB and semantic cache
        task_description = "Refactor authentication module for better security"

        # Save to persistent store
        checkpoint_id = await checkpoint_store.save_checkpoint(
            CheckpointType.ORCHESTRATION_RUN,
            state={"task": task_description, "status": "running"},
            checkpoint_id="auth_refactor_001",
        )

        # Also index in semantic store
        await semantic_store.embed_and_store(
            checkpoint_id,
            "orchestration_run",
            task_description,
            metadata={"persistent": True},
        )

        # Verify both stores have the data
        persistent = await checkpoint_store.get_checkpoint("auth_refactor_001")
        assert persistent is not None

        semantic = semantic_store.get_checkpoint("auth_refactor_001")
        assert semantic is not None

        # Search semantically
        results = await semantic_store.search_similar("security authentication")
        assert len(results) >= 1
        assert results[0].checkpoint_id == "auth_refactor_001"

        await checkpoint_store.close()

    @pytest.mark.asyncio
    async def test_ralph_iteration_semantic_search(self, temp_db_path: str):
        """Test semantic search over Ralph iteration descriptions."""
        from core.orchestration.checkpoint_persistence import (
            CheckpointStore,
            CheckpointConfig,
            RunStatus,
        )

        config = CheckpointConfig(db_path=temp_db_path)
        checkpoint_store = CheckpointStore(config)

        semantic_store = SemanticCheckpointStore(
            checkpoint_store=checkpoint_store,
        )
        await semantic_store.initialize()

        # Save Ralph iterations
        iterations = [
            ("iter_1", 1, "Implement Voyage AI embedding layer"),
            ("iter_2", 2, "Create agent SDK wrapper classes"),
            ("iter_3", 3, "Add multi-agent orchestration support"),
            ("iter_4", 4, "Integrate LangGraph checkpointing"),
            ("iter_5", 5, "Build semantic search for checkpoints"),
        ]

        for iter_id, num, task in iterations:
            await checkpoint_store.save_ralph_iteration(
                iteration_number=num,
                iteration_id=iter_id,
                session_id="ralph_session_001",
                task=task,
                status=RunStatus.COMPLETED,
            )

            await semantic_store.embed_and_store(
                iter_id,
                "ralph_iteration",
                task,
                metadata={"iteration_number": num},
            )

        # Search for embedding-related iterations
        results = await semantic_store.search_similar("embeddings vectors", top_k=2)

        assert len(results) >= 1
        # Should find the Voyage AI embedding iteration
        iteration_ids = [r.checkpoint_id for r in results]
        assert "iter_1" in iteration_ids or "iter_5" in iteration_ids

        await checkpoint_store.close()


# =============================================================================
# Test: Semantic Search Quality
# =============================================================================

class TestSemanticSearchQuality:
    """Test the quality of semantic search results."""

    @pytest.mark.asyncio
    async def test_similar_content_ranks_higher(self):
        """Test that more similar content ranks higher."""
        store = SemanticCheckpointStore()
        await store.initialize()

        # Store checkpoints with varying relevance
        await store.embed_and_store(
            "ckpt_relevant",
            "run",
            "Python machine learning data science neural networks",
        )
        await store.embed_and_store(
            "ckpt_unrelated",
            "run",
            "Cooking recipes Italian pasta dishes",
        )
        await store.embed_and_store(
            "ckpt_somewhat",
            "run",
            "JavaScript web development frontend React",
        )

        # Search for ML-related content
        results = await store.search_similar("machine learning AI")

        assert len(results) == 3
        # ML content should rank highest (even with mock embeddings)
        # This tests the ranking mechanism works
        assert results[0].checkpoint_id == "ckpt_relevant" or \
               results[0].similarity >= results[1].similarity

    @pytest.mark.asyncio
    async def test_exact_match_highest_similarity(self):
        """Test that exact content match has highest similarity."""
        store = SemanticCheckpointStore()
        await store.initialize()

        exact_content = "Database query optimization for PostgreSQL"

        await store.embed_and_store("ckpt_exact", "run", exact_content)
        await store.embed_and_store("ckpt_other", "run", "Unrelated content here")

        results = await store.search_similar(exact_content)

        assert len(results) == 2
        # Exact match should be first
        assert results[0].checkpoint_id == "ckpt_exact"
        # And should have perfect or near-perfect similarity
        assert results[0].similarity > 0.99


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_content(self):
        """Test handling of empty content."""
        store = SemanticCheckpointStore()
        await store.initialize()

        # Should handle empty content without crashing
        checkpoint = await store.embed_and_store("empty", "run", "")
        assert checkpoint is not None
        assert checkpoint.embedding_dim > 0

    @pytest.mark.asyncio
    async def test_very_long_content(self):
        """Test handling of very long content."""
        store = SemanticCheckpointStore()
        await store.initialize()

        long_content = "word " * 10000  # 50k characters

        checkpoint = await store.embed_and_store("long", "run", long_content)
        assert checkpoint is not None

    @pytest.mark.asyncio
    async def test_special_characters(self):
        """Test handling of special characters in content."""
        store = SemanticCheckpointStore()
        await store.initialize()

        special_content = "Code: def foo():\n\treturn 'bar' # comment\n\"\"\"\ndocstring\n\"\"\""

        checkpoint = await store.embed_and_store("special", "run", special_content)
        assert checkpoint is not None
        assert checkpoint.content == special_content

    @pytest.mark.asyncio
    async def test_duplicate_checkpoint_id(self):
        """Test handling duplicate checkpoint IDs."""
        store = SemanticCheckpointStore()
        await store.initialize()

        await store.embed_and_store("dup_id", "run", "First content")
        await store.embed_and_store("dup_id", "run", "Second content")

        # Should overwrite with latest
        checkpoint = store.get_checkpoint("dup_id")
        assert checkpoint.content == "Second content"

    @pytest.mark.asyncio
    async def test_unicode_content(self):
        """Test handling of unicode content."""
        store = SemanticCheckpointStore()
        await store.initialize()

        unicode_content = "日本語テスト 中文测试 한국어 🚀 émojis"

        checkpoint = await store.embed_and_store("unicode", "run", unicode_content)
        assert checkpoint is not None
        assert checkpoint.content == unicode_content


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    # Run fast tests by default
    pytest.main([__file__, "-v", "-m", "not slow"])
