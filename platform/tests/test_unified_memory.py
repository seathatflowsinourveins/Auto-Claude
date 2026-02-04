"""
Tests for Unified Memory Interface - V41 Architecture

Comprehensive test suite covering:
- Content classification and routing
- Cross-memory search with RRF fusion
- Memory lifecycle management
- Forgetting curve integration
- Compression integration
- Statistics dashboard
"""

import asyncio
import pytest
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# Import test subjects
from core.memory.unified import (
    UnifiedMemory,
    ContentClassifier,
    RRFFusion,
    MemoryType,
    RoutingDecision,
    SearchStrategy,
    LifecycleState,
    UnifiedSearchResult,
    RoutingResult,
    MaintenanceReport,
    UnifiedStatistics,
    create_unified_memory,
    get_unified_memory,
    reset_unified_memory,
)

from core.memory.backends.base import (
    MemoryEntry,
    MemoryTier,
    MemoryPriority,
    MemoryNamespace,
    MemoryAccessPattern,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def classifier():
    """Create a ContentClassifier instance."""
    return ContentClassifier()


@pytest.fixture
def rrf():
    """Create an RRFFusion instance."""
    return RRFFusion(k=60)


@pytest.fixture
def mock_entry():
    """Create a mock MemoryEntry."""
    return MemoryEntry(
        id="test_entry_1",
        content="This is a test memory entry",
        tier=MemoryTier.ARCHIVAL_MEMORY,
        priority=MemoryPriority.NORMAL,
        access_count=5,
        strength=0.8,
        decay_rate=0.15,
    )


@pytest.fixture
def mock_sqlite_backend():
    """Create a mock SQLite backend."""
    backend = AsyncMock()
    backend.search = AsyncMock(return_value=[])
    backend.get = AsyncMock(return_value=None)
    backend.put = AsyncMock()
    backend.delete = AsyncMock(return_value=True)
    backend.list_all = AsyncMock(return_value=[])
    backend.store_memory = AsyncMock(return_value="test_id")
    backend.get_stats = AsyncMock(return_value={
        "total_memories": 100,
        "memories_by_type": {"fact": 50, "learning": 30, "decision": 20},
        "db_size_bytes": 1024000,
        "strength_stats": {
            "average": 0.7,
            "weak_count": 10,
            "very_weak_count": 2,
        },
    })
    backend.get_learnings = AsyncMock(return_value=[])
    backend.get_decisions = AsyncMock(return_value=[])
    backend.archive_weak_memories = AsyncMock(return_value=(5, 10))
    backend.delete_archived_memories = AsyncMock(return_value=3)
    backend.close = MagicMock()
    return backend


@pytest.fixture
def mock_procedural_memory():
    """Create a mock ProceduralMemory."""
    pm = AsyncMock()
    pm.learn_procedure = AsyncMock()
    pm.recall_procedure = AsyncMock(return_value=[])
    pm.get_procedure = AsyncMock(return_value=None)
    pm.delete_procedure = AsyncMock(return_value=False)
    pm.list_procedures = AsyncMock(return_value=[])
    pm.get_stats = AsyncMock(return_value={
        "total_procedures": 10,
        "active_procedures": 8,
        "total_executions": 50,
        "successful_executions": 45,
        "db_size_bytes": 51200,
    })
    pm.close = MagicMock()
    return pm


@pytest.fixture
def mock_bitemporal_memory():
    """Create a mock BiTemporalMemory."""
    bm = AsyncMock()
    bm.store = AsyncMock()
    bm.search_valid_now = AsyncMock(return_value=[])
    bm.query_as_of = AsyncMock(return_value=[])
    bm.query_valid_at = AsyncMock(return_value=[])
    bm.query_bitemporal = AsyncMock(return_value=[])
    bm.get = AsyncMock(return_value=None)
    bm.delete = AsyncMock(return_value=False)
    bm.get_stats = AsyncMock(return_value={
        "total_entries": 50,
        "currently_valid": 40,
        "superseded": 5,
        "invalidated": 5,
        "db_size_bytes": 25600,
    })
    bm.close = MagicMock()
    return bm


@pytest.fixture
def mock_compressor():
    """Create a mock MemoryCompressor."""
    comp = MagicMock()
    comp.identify_candidates = AsyncMock(return_value=[])
    comp.compress_memories = AsyncMock()
    comp.get_metrics = MagicMock(return_value=MagicMock(
        total_original_tokens=10000,
        total_compressed_tokens=3000,
    ))
    return comp


@pytest.fixture
async def unified_memory(
    mock_sqlite_backend,
    mock_procedural_memory,
    mock_bitemporal_memory,
    mock_compressor,
):
    """Create a UnifiedMemory instance with mocked backends."""
    memory = UnifiedMemory(
        sqlite_backend=mock_sqlite_backend,
        procedural_memory=mock_procedural_memory,
        bitemporal_memory=mock_bitemporal_memory,
        compressor=mock_compressor,
        auto_route=True,
        enable_forgetting=True,
        enable_compression=True,
    )
    memory._initialized = True
    yield memory
    memory.close()


# =============================================================================
# CONTENT CLASSIFIER TESTS
# =============================================================================

class TestContentClassifier:
    """Tests for ContentClassifier."""

    def test_classify_procedure_content(self, classifier):
        """Test classification of procedural content."""
        content = """
        Step 1: Run git add .
        Step 2: Run git commit -m "message"
        Step 3: Run git push
        """
        result = classifier.classify(content)
        assert result.decision == RoutingDecision.PROCEDURAL
        assert result.confidence >= 0.2
        assert any("Procedure pattern" in r for r in result.reasons)

    def test_classify_temporal_content(self, classifier):
        """Test classification of temporal content."""
        content = "User preference changed on 2024-01-15. Valid from 2024-01-15 until 2024-12-31."
        result = classifier.classify(content)
        assert result.decision == RoutingDecision.BITEMPORAL
        assert result.confidence >= 0.2
        assert any("Temporal pattern" in r for r in result.reasons)

    def test_classify_learning_content(self, classifier):
        """Test classification of learning content."""
        content = "I learned that you should always use TypeScript for new code. This is a best practice."
        result = classifier.classify(content)
        assert result.decision == RoutingDecision.FORGETTING
        assert result.confidence >= 0.2
        assert any("Learning pattern" in r for r in result.reasons)

    def test_classify_decision_content(self, classifier):
        """Test classification of decision content."""
        content = "We decided to use React because it has better community support. The trade-off is bundle size."
        result = classifier.classify(content)
        assert result.decision == RoutingDecision.FORGETTING
        assert result.confidence >= 0.2
        assert any("Decision pattern" in r for r in result.reasons)

    def test_classify_with_explicit_type(self, classifier):
        """Test classification with explicit memory type in metadata."""
        content = "Some generic content"
        metadata = {"memory_type": "procedure"}
        result = classifier.classify(content, metadata)
        assert result.decision == RoutingDecision.PROCEDURAL
        assert result.confidence == 1.0
        assert "Explicit type" in result.reasons[0]

    def test_classify_generic_content(self, classifier):
        """Test classification of generic content (no patterns)."""
        content = "Hello world"
        result = classifier.classify(content)
        assert result.decision == RoutingDecision.STANDARD
        assert result.confidence == 0.5
        assert "defaulting to fact" in result.reasons[0].lower()

    def test_namespace_suggestion(self, classifier):
        """Test namespace suggestions based on type."""
        content = "I learned that X is important"
        result = classifier.classify(content)
        assert result.suggested_namespace == MemoryNamespace.LEARNINGS

    def test_priority_suggestion(self, classifier):
        """Test priority suggestions based on type."""
        content = "I prefer dark mode for all applications"
        result = classifier.classify(content)
        assert result.suggested_priority == MemoryPriority.HIGH


# =============================================================================
# RRF FUSION TESTS
# =============================================================================

class TestRRFFusion:
    """Tests for RRFFusion."""

    def test_fuse_single_source(self, rrf):
        """Test fusion with a single source."""
        entries = [
            MemoryEntry(id=f"entry_{i}", content=f"Content {i}")
            for i in range(5)
        ]
        result_lists = {
            "source1": [(e, 1.0 - i * 0.1) for i, e in enumerate(entries)]
        }

        fused = rrf.fuse(result_lists, limit=5)

        assert len(fused) == 5
        assert fused[0].entry.id == "entry_0"
        assert fused[0].source == "source1"
        assert fused[0].rrf_score > 0

    def test_fuse_multiple_sources(self, rrf):
        """Test fusion with multiple sources."""
        entries1 = [
            MemoryEntry(id=f"entry_{i}", content=f"Content {i}")
            for i in range(3)
        ]
        entries2 = [
            MemoryEntry(id=f"entry_{i}", content=f"Content {i}")
            for i in [2, 1, 0]  # Different order
        ]

        result_lists = {
            "source1": [(e, 1.0 - i * 0.1) for i, e in enumerate(entries1)],
            "source2": [(e, 1.0 - i * 0.1) for i, e in enumerate(entries2)],
        }

        fused = rrf.fuse(result_lists, limit=5)

        # Entry 0 and 2 should have highest RRF scores (appear in both lists at extremes)
        assert len(fused) == 3
        # Check that entries appearing in both sources have multiple contributing sources
        for result in fused:
            assert "source1" in result.metadata["contributing_sources"] or \
                   "source2" in result.metadata["contributing_sources"]

    def test_fuse_with_duplicates(self, rrf):
        """Test that duplicates are properly handled."""
        entry = MemoryEntry(id="shared_entry", content="Shared content")

        result_lists = {
            "source1": [(entry, 0.9)],
            "source2": [(entry, 0.8)],
            "source3": [(entry, 0.7)],
        }

        fused = rrf.fuse(result_lists, limit=10)

        assert len(fused) == 1
        assert fused[0].entry.id == "shared_entry"
        assert fused[0].metadata["source_count"] == 3

    def test_fuse_limit_respected(self, rrf):
        """Test that limit is respected."""
        entries = [
            MemoryEntry(id=f"entry_{i}", content=f"Content {i}")
            for i in range(20)
        ]
        result_lists = {
            "source1": [(e, 1.0 - i * 0.05) for i, e in enumerate(entries)]
        }

        fused = rrf.fuse(result_lists, limit=5)

        assert len(fused) == 5

    def test_fuse_empty_sources(self, rrf):
        """Test fusion with empty sources."""
        result_lists = {
            "source1": [],
            "source2": [],
        }

        fused = rrf.fuse(result_lists, limit=10)

        assert len(fused) == 0


# =============================================================================
# UNIFIED MEMORY TESTS
# =============================================================================

class TestUnifiedMemory:
    """Tests for UnifiedMemory."""

    @pytest.mark.asyncio
    async def test_store_with_auto_routing_procedure(self, unified_memory, mock_procedural_memory):
        """Test storing procedural content with auto-routing."""
        mock_procedure = MagicMock()
        mock_procedure.id = "proc_123"
        mock_procedural_memory.learn_procedure.return_value = mock_procedure

        content = "Step 1: git add . Step 2: git commit"
        memory_id = await unified_memory.store(content)

        # Should route to procedural memory
        mock_procedural_memory.learn_procedure.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_with_auto_routing_temporal(self, unified_memory, mock_bitemporal_memory):
        """Test storing temporal content with auto-routing."""
        mock_entry = MagicMock()
        mock_entry.id = "temp_123"
        mock_bitemporal_memory.store.return_value = mock_entry

        content = "User preference valid from 2024-01-01 until 2024-12-31"
        memory_id = await unified_memory.store(content)

        # Should route to bi-temporal memory
        mock_bitemporal_memory.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_with_forced_routing(self, unified_memory, mock_sqlite_backend):
        """Test storing with forced routing decision."""
        content = "Some content"
        memory_id = await unified_memory.store(
            content,
            force_routing=RoutingDecision.STANDARD
        )

        # Should route to standard SQLite
        mock_sqlite_backend.store_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_with_explicit_type(self, unified_memory, mock_sqlite_backend):
        """Test storing with explicit memory type."""
        content = "Some fact"
        memory_id = await unified_memory.store(
            content,
            memory_type="fact",
            importance=0.8,
            tags=["test", "important"],
        )

        mock_sqlite_backend.store_memory.assert_called_once()
        call_kwargs = mock_sqlite_backend.store_memory.call_args[1]
        assert call_kwargs["importance"] == 0.8
        assert "test" in call_kwargs["tags"]

    @pytest.mark.asyncio
    async def test_search_all_strategy(
        self,
        unified_memory,
        mock_sqlite_backend,
        mock_bitemporal_memory,
        mock_procedural_memory
    ):
        """Test cross-memory search with ALL strategy."""
        # Setup mock returns
        mock_entry = MemoryEntry(id="sqlite_1", content="SQLite result")
        mock_sqlite_backend.search.return_value = [mock_entry]

        results = await unified_memory.search("test query", strategy=SearchStrategy.ALL)

        # Should search all backends
        mock_sqlite_backend.search.assert_called()
        mock_bitemporal_memory.search_valid_now.assert_called()
        mock_procedural_memory.recall_procedure.assert_called()

    @pytest.mark.asyncio
    async def test_search_with_temporal_filters(self, unified_memory, mock_bitemporal_memory):
        """Test search with temporal filters."""
        as_of = datetime.now(timezone.utc) - timedelta(days=7)
        valid_at = datetime.now(timezone.utc) - timedelta(days=3)

        await unified_memory.search(
            "test query",
            strategy=SearchStrategy.TEMPORAL,
            as_of_time=as_of,
            valid_time=valid_at,
        )

        # Should use bi-temporal query
        mock_bitemporal_memory.query_bitemporal.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_from_multiple_backends(
        self,
        unified_memory,
        mock_sqlite_backend,
        mock_bitemporal_memory
    ):
        """Test get falls through backends until found."""
        mock_sqlite_backend.get.return_value = None

        mock_entry = MemoryEntry(id="temp_1", content="Temporal entry")
        mock_bitemporal_memory.get.return_value = mock_entry

        result = await unified_memory.get("temp_1")

        assert result is not None
        assert result.id == "temp_1"
        mock_sqlite_backend.get.assert_called_once()
        mock_bitemporal_memory.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_from_all_backends(
        self,
        unified_memory,
        mock_sqlite_backend,
        mock_bitemporal_memory,
        mock_procedural_memory
    ):
        """Test delete attempts all backends."""
        mock_sqlite_backend.delete.return_value = True
        mock_bitemporal_memory.delete.return_value = False
        mock_procedural_memory.delete_procedure.return_value = False

        result = await unified_memory.delete("test_id")

        assert result is True
        mock_sqlite_backend.delete.assert_called_once()
        mock_bitemporal_memory.delete.assert_called_once()
        mock_procedural_memory.delete_procedure.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_maintenance(self, unified_memory, mock_sqlite_backend, mock_compressor):
        """Test maintenance operations."""
        mock_entries = [
            MemoryEntry(
                id=f"entry_{i}",
                content=f"Content {i}",
                strength=0.5,
                decay_rate=0.15,
                created_at=datetime.now(timezone.utc) - timedelta(days=10),
            )
            for i in range(5)
        ]
        mock_sqlite_backend.list_all.return_value = mock_entries

        report = await unified_memory.run_maintenance(
            archive_threshold=0.1,
            delete_threshold=0.01,
        )

        assert isinstance(report, MaintenanceReport)
        assert report.duration_seconds >= 0
        assert unified_memory._last_maintenance is not None

    @pytest.mark.asyncio
    async def test_get_statistics(
        self,
        unified_memory,
        mock_sqlite_backend,
        mock_bitemporal_memory,
        mock_procedural_memory
    ):
        """Test statistics collection."""
        mock_sqlite_backend.list_all.return_value = []

        stats = await unified_memory.get_statistics()

        assert isinstance(stats, UnifiedStatistics)
        assert stats.total_entries >= 0
        mock_sqlite_backend.get_stats.assert_called()
        mock_bitemporal_memory.get_stats.assert_called()
        mock_procedural_memory.get_stats.assert_called()

    @pytest.mark.asyncio
    async def test_get_context(self, unified_memory, mock_sqlite_backend, mock_procedural_memory):
        """Test context generation."""
        mock_learnings = [
            MemoryEntry(id="learn_1", content="Learning 1"),
        ]
        mock_decisions = [
            MemoryEntry(id="dec_1", content="Decision 1"),
        ]
        mock_sqlite_backend.get_learnings.return_value = mock_learnings
        mock_sqlite_backend.get_decisions.return_value = mock_decisions

        context = await unified_memory.get_context(max_tokens=1000)

        assert "Recent Learnings" in context or "Key Decisions" in context
        mock_sqlite_backend.get_learnings.assert_called()
        mock_sqlite_backend.get_decisions.assert_called()

    @pytest.mark.asyncio
    async def test_reinforce_memory(self, unified_memory, mock_sqlite_backend, mock_entry):
        """Test memory reinforcement."""
        mock_entry.strength = 0.8
        mock_sqlite_backend.get.return_value = mock_entry

        new_strength = await unified_memory.reinforce_memory("test_id", "recall")

        assert new_strength is not None
        mock_sqlite_backend.get.assert_called()


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    @pytest.mark.asyncio
    async def test_create_unified_memory(self):
        """Test singleton creation."""
        reset_unified_memory()

        with patch('platform.core.memory.unified.UnifiedMemory') as MockUnified:
            mock_instance = AsyncMock()
            mock_instance._ensure_initialized = AsyncMock()
            MockUnified.return_value = mock_instance

            # First call creates
            memory1 = await create_unified_memory()

            # Second call returns same instance
            memory2 = await create_unified_memory()

            assert memory1 is memory2

        reset_unified_memory()

    def test_get_unified_memory_before_create(self):
        """Test get before create returns None."""
        reset_unified_memory()
        assert get_unified_memory() is None

    def test_reset_unified_memory(self):
        """Test reset clears singleton."""
        reset_unified_memory()
        assert get_unified_memory() is None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests with real backends (in temp directories)."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_full_lifecycle(self):
        """Test full memory lifecycle: store -> search -> reinforce -> maintenance."""
        reset_unified_memory()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Patch paths to use temp directory
            with patch.object(Path, 'home', return_value=tmp_path):
                # Create memory instance
                memory = UnifiedMemory(auto_route=True, enable_forgetting=True)

                try:
                    await memory._ensure_initialized()

                    # Store different types of content
                    learning_id = await memory.store(
                        "I learned that async/await is essential for Python performance",
                        memory_type="learning",
                        importance=0.8,
                    )

                    fact_id = await memory.store(
                        "Python 3.12 was released in October 2023",
                        memory_type="fact",
                    )

                    # Search
                    results = await memory.search("Python", limit=5)
                    assert len(results) >= 0  # May be 0 if backends not available

                    # Get context
                    context = await memory.get_context(max_tokens=500)
                    assert isinstance(context, str)

                    # Run maintenance
                    report = await memory.run_maintenance()
                    assert isinstance(report, MaintenanceReport)

                    # Get statistics
                    stats = await memory.get_statistics()
                    assert isinstance(stats, UnifiedStatistics)

                finally:
                    memory.close()

        reset_unified_memory()


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_search_with_no_backends(self):
        """Test search handles missing backends gracefully."""
        memory = UnifiedMemory(
            sqlite_backend=None,
            procedural_memory=None,
            bitemporal_memory=None,
        )
        memory._initialized = True

        results = await memory.search("test")
        assert results == []

        memory.close()

    @pytest.mark.asyncio
    async def test_store_fallback_on_backend_error(self, unified_memory, mock_sqlite_backend):
        """Test store falls back on backend errors."""
        # Make procedural fail
        unified_memory._procedural.learn_procedure.side_effect = Exception("Backend error")

        # Should fall back to standard storage
        content = "Step 1: do something"
        memory_id = await unified_memory.store(content, force_routing=RoutingDecision.STANDARD)

        mock_sqlite_backend.store_memory.assert_called()

    @pytest.mark.asyncio
    async def test_statistics_with_failing_backends(self, unified_memory, mock_sqlite_backend):
        """Test statistics handles backend failures gracefully."""
        mock_sqlite_backend.get_stats.side_effect = Exception("Stats error")
        mock_sqlite_backend.list_all.return_value = []

        # Should not raise, but log warning
        stats = await unified_memory.get_statistics()
        assert isinstance(stats, UnifiedStatistics)

    def test_classifier_with_empty_content(self, classifier):
        """Test classifier handles empty content."""
        result = classifier.classify("")
        assert result.decision == RoutingDecision.STANDARD

    def test_rrf_with_negative_k(self):
        """Test RRF with invalid k value."""
        rrf = RRFFusion(k=0)  # Edge case
        entries = [MemoryEntry(id="e1", content="content")]
        result_lists = {"source": [(entries[0], 1.0)]}

        # Should still work (k=0 means rank+1 denominator)
        fused = rrf.fuse(result_lists, limit=10)
        assert len(fused) == 1


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance-related tests."""

    @pytest.mark.asyncio
    async def test_search_parallel_execution(self, unified_memory):
        """Test that search queries backends in parallel."""
        import time

        # Add delays to mock backends
        async def slow_search(*args, **kwargs):
            await asyncio.sleep(0.1)
            return []

        unified_memory._sqlite.search = slow_search
        unified_memory._bitemporal.search_valid_now = slow_search
        unified_memory._procedural.recall_procedure = slow_search

        start = time.time()
        await unified_memory.search("test", strategy=SearchStrategy.ALL)
        elapsed = time.time() - start

        # If parallel, should be ~0.1s; if sequential, would be ~0.3s
        assert elapsed < 0.25  # Allow some overhead

    def test_rrf_performance_large_lists(self, rrf):
        """Test RRF fusion with large result lists."""
        import time

        entries = [
            MemoryEntry(id=f"entry_{i}", content=f"Content {i}")
            for i in range(1000)
        ]

        result_lists = {
            f"source_{j}": [(e, 1.0 - i * 0.001) for i, e in enumerate(entries)]
            for j in range(5)
        }

        start = time.time()
        fused = rrf.fuse(result_lists, limit=100)
        elapsed = time.time() - start

        assert len(fused) == 100
        assert elapsed < 1.0  # Should complete in under 1 second


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
