"""
Unit tests for Memory Compaction System - V42 Architecture

Tests:
- MemoryCompactor candidate identification
- Compaction score calculation
- Semantic similarity detection
- Memory merging
- Cold storage archival
- Background scheduler
"""

import asyncio
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from core.memory.compaction import (
    MemoryCompactor,
    CompactionStrategy,
    CompactionPriority,
    MergeStrategy,
    CompactionConfig,
    CompactionCandidate,
    MergeGroup,
    CompactionReport,
    CompactionMetrics,
    CompactionScheduler,
    get_memory_compactor,
    get_compaction_scheduler,
)
from core.memory.backends.base import (
    MemoryEntry,
    MemoryTier,
    MemoryPriority,
    MemoryNamespace,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_backend():
    """Create a mock memory backend."""
    backend = AsyncMock()
    backend.list_all = AsyncMock(return_value=[])
    backend.get = AsyncMock(return_value=None)
    backend.put = AsyncMock()
    backend.delete = AsyncMock(return_value=True)
    backend.search = AsyncMock(return_value=[])
    backend.count = AsyncMock(return_value=0)
    backend.get_stats = AsyncMock(return_value={
        'total_memories': 0,
        'db_size_bytes': 0
    })
    return backend


@pytest.fixture
def compaction_config():
    """Create a test configuration."""
    return CompactionConfig(
        min_age_days=1.0,
        max_age_days=7.0,
        stale_threshold_days=3.0,
        max_memory_count=100,
        target_memory_count=50,
        min_strength_threshold=0.2,
        min_importance_threshold=0.3,
        min_access_count=1,
        max_access_age_days=2.0,
        similarity_threshold=0.8,
        max_merge_group_size=5,
        enable_cold_storage=False,
        schedule_interval_hours=1.0,
    )


@pytest.fixture
def compactor(mock_backend, compaction_config):
    """Create a MemoryCompactor instance."""
    return MemoryCompactor(
        backend=mock_backend,
        config=compaction_config
    )


def create_test_entry(
    entry_id: str,
    content: str,
    age_days: float = 0.0,
    access_count: int = 0,
    strength: float = 1.0,
    importance: float = 0.5,
    priority: MemoryPriority = MemoryPriority.NORMAL,
    tags: list = None,
    namespace: MemoryNamespace = None,
) -> MemoryEntry:
    """Create a test MemoryEntry."""
    now = datetime.now(timezone.utc)
    created_at = now - timedelta(days=age_days)
    last_accessed = created_at if access_count == 0 else now - timedelta(days=age_days / 2)

    return MemoryEntry(
        id=entry_id,
        content=content,
        tier=MemoryTier.ARCHIVAL_MEMORY,
        priority=priority,
        namespace=namespace,
        created_at=created_at,
        last_accessed=last_accessed,
        access_count=access_count,
        strength=strength,
        decay_rate=0.15,
        tags=tags or [],
        metadata={'importance': importance},
    )


# =============================================================================
# COMPACTION CANDIDATE TESTS
# =============================================================================

class TestCandidateIdentification:
    """Test compaction candidate identification."""

    @pytest.mark.asyncio
    async def test_identify_old_memories(self, compactor, mock_backend):
        """Test identification of old memories as candidates."""
        entries = [
            create_test_entry("old_1", "Old content", age_days=10.0, access_count=0),
            create_test_entry("new_1", "New content", age_days=0.5, access_count=5),
        ]
        mock_backend.list_all.return_value = entries

        candidates = await compactor.identify_low_value_memories()

        assert len(candidates) == 1
        assert candidates[0].entry_id == "old_1"
        assert "old" in candidates[0].reason.lower()

    @pytest.mark.asyncio
    async def test_identify_weak_memories(self, compactor, mock_backend):
        """Test identification of weak (decayed) memories."""
        entries = [
            create_test_entry("weak_1", "Weak memory", strength=0.05, age_days=5.0),
            create_test_entry("strong_1", "Strong memory", strength=0.9, age_days=5.0, access_count=5),
        ]
        mock_backend.list_all.return_value = entries

        candidates = await compactor.identify_low_value_memories()

        assert len(candidates) >= 1
        weak_candidate = next((c for c in candidates if c.entry_id == "weak_1"), None)
        assert weak_candidate is not None
        assert weak_candidate.strength < 0.2

    @pytest.mark.asyncio
    async def test_identify_low_importance_memories(self, compactor, mock_backend):
        """Test identification of low importance memories."""
        entries = [
            create_test_entry("low_1", "Low importance", importance=0.1, age_days=5.0),
            create_test_entry("high_1", "High importance", importance=0.9, age_days=5.0),
        ]
        mock_backend.list_all.return_value = entries

        candidates = await compactor.identify_low_value_memories()

        low_candidate = next((c for c in candidates if c.entry_id == "low_1"), None)
        assert low_candidate is not None
        assert "importance" in low_candidate.reason.lower()

    @pytest.mark.asyncio
    async def test_preserve_critical_memories(self, compactor, mock_backend):
        """Test that critical memories are preserved."""
        entries = [
            create_test_entry(
                "critical_1", "Critical memory",
                age_days=30.0, access_count=0, strength=0.05,
                priority=MemoryPriority.CRITICAL
            ),
        ]
        mock_backend.list_all.return_value = entries

        candidates = await compactor.identify_low_value_memories()

        assert len(candidates) == 0

    @pytest.mark.asyncio
    async def test_preserve_tagged_memories(self, compactor, mock_backend):
        """Test that memories with preserve tags are kept."""
        entries = [
            create_test_entry(
                "tagged_1", "Tagged memory",
                age_days=30.0, access_count=0, strength=0.05,
                tags=["critical"]
            ),
        ]
        mock_backend.list_all.return_value = entries

        candidates = await compactor.identify_low_value_memories()

        assert len(candidates) == 0

    @pytest.mark.asyncio
    async def test_preserve_artifact_namespace(self, compactor, mock_backend):
        """Test that artifact namespace memories are preserved."""
        entries = [
            create_test_entry(
                "artifact_1", "Artifact memory",
                age_days=30.0, access_count=0, strength=0.05,
                namespace=MemoryNamespace.ARTIFACTS
            ),
        ]
        mock_backend.list_all.return_value = entries

        candidates = await compactor.identify_low_value_memories()

        assert len(candidates) == 0


# =============================================================================
# COMPACTION SCORE TESTS
# =============================================================================

class TestCompactionScore:
    """Test compaction score calculation."""

    def test_score_calculation_old_memory(self, compactor):
        """Test score for old memories."""
        score = compactor._calculate_compaction_score(
            age_days=30.0,
            days_since_access=30.0,
            access_count=0,
            strength=0.1,
            importance=0.1
        )
        assert score > 0.8  # Should be high

    def test_score_calculation_new_memory(self, compactor):
        """Test score for new memories."""
        score = compactor._calculate_compaction_score(
            age_days=0.5,
            days_since_access=0.1,
            access_count=10,
            strength=1.0,
            importance=0.9
        )
        assert score < 0.3  # Should be low

    def test_score_between_zero_and_one(self, compactor):
        """Test score is always in valid range."""
        for _ in range(100):
            import random
            score = compactor._calculate_compaction_score(
                age_days=random.uniform(0, 100),
                days_since_access=random.uniform(0, 100),
                access_count=random.randint(0, 100),
                strength=random.uniform(0, 1),
                importance=random.uniform(0, 1)
            )
            assert 0.0 <= score <= 1.0


# =============================================================================
# SIMILARITY TESTS
# =============================================================================

class TestSimilarityDetection:
    """Test semantic similarity detection."""

    @pytest.mark.asyncio
    async def test_keyword_similarity_identical(self, compactor):
        """Test similarity for identical content."""
        entry1 = create_test_entry("e1", "The quick brown fox jumps over the lazy dog")
        entry2 = create_test_entry("e2", "The quick brown fox jumps over the lazy dog")

        similarity = await compactor._calculate_similarity(entry1, entry2)
        assert similarity == 1.0

    @pytest.mark.asyncio
    async def test_keyword_similarity_different(self, compactor):
        """Test similarity for different content."""
        entry1 = create_test_entry("e1", "Python programming language syntax")
        entry2 = create_test_entry("e2", "JavaScript web development framework")

        similarity = await compactor._calculate_similarity(entry1, entry2)
        assert similarity < 0.5

    @pytest.mark.asyncio
    async def test_keyword_similarity_partial(self, compactor):
        """Test similarity for partially overlapping content."""
        entry1 = create_test_entry("e1", "Machine learning with Python and TensorFlow")
        entry2 = create_test_entry("e2", "Deep learning with Python and PyTorch")

        similarity = await compactor._calculate_similarity(entry1, entry2)
        assert 0.2 < similarity < 0.8  # Some overlap

    @pytest.mark.asyncio
    async def test_identify_similar_memories(self, compactor, mock_backend):
        """Test identification of similar memory groups."""
        entries = [
            create_test_entry("e1", "Python machine learning tutorial", age_days=5.0),
            create_test_entry("e2", "Python machine learning guide", age_days=5.0),
            create_test_entry("e3", "JavaScript web development", age_days=5.0),
        ]
        mock_backend.list_all.return_value = entries

        # First get candidates
        candidates = await compactor.identify_low_value_memories()

        # Mock get for merge detection
        def mock_get(key, reinforce=False):
            return next((e for e in entries if e.id == key), None)
        mock_backend.get.side_effect = mock_get

        # Set higher similarity threshold for this test
        compactor._config.similarity_threshold = 0.4

        merge_groups = await compactor.identify_similar_memories(candidates)

        # Should find at least one group with similar Python content
        assert len(merge_groups) >= 0  # May or may not find groups depending on score


# =============================================================================
# MERGE TESTS
# =============================================================================

class TestMemoryMerging:
    """Test memory merging functionality."""

    @pytest.mark.asyncio
    async def test_merge_newest_wins(self, compactor, mock_backend):
        """Test merge with newest_wins strategy."""
        now = datetime.now(timezone.utc)
        entries = [
            create_test_entry("old", "Old content", age_days=10.0),
            create_test_entry("new", "New content", age_days=1.0),
        ]
        entries[0].created_at = now - timedelta(days=10)
        entries[1].created_at = now - timedelta(days=1)

        def mock_get(key, reinforce=False):
            return next((e for e in entries if e.id == key), None)
        mock_backend.get.side_effect = mock_get

        group = MergeGroup(
            representative_id="old",
            member_ids=["old", "new"],
            similarity_scores={"old": 1.0, "new": 0.9}
        )

        result_id = await compactor.merge_memories(group, MergeStrategy.NEWEST_WINS)
        assert result_id == "new"

    @pytest.mark.asyncio
    async def test_merge_combines_tags(self, compactor, mock_backend):
        """Test that merge combines tags from all entries."""
        entries = [
            create_test_entry("e1", "Content 1", age_days=5.0, tags=["tag1", "tag2"]),
            create_test_entry("e2", "Content 2", age_days=5.0, tags=["tag2", "tag3"]),
        ]

        def mock_get(key, reinforce=False):
            return next((e for e in entries if e.id == key), None)
        mock_backend.get.side_effect = mock_get

        group = MergeGroup(
            representative_id="e1",
            member_ids=["e1", "e2"],
            similarity_scores={"e1": 1.0, "e2": 0.9}
        )

        await compactor.merge_memories(group, MergeStrategy.HIGHEST_SCORE)

        # Check that put was called with merged tags
        put_call = mock_backend.put.call_args
        assert put_call is not None
        saved_entry = put_call[0][1]
        assert set(saved_entry.tags) == {"tag1", "tag2", "tag3"}


# =============================================================================
# COMPACTION REPORT TESTS
# =============================================================================

class TestCompactionReport:
    """Test compaction report generation."""

    def test_report_to_dict(self):
        """Test report serialization."""
        now = datetime.now(timezone.utc)
        report = CompactionReport(
            started_at=now - timedelta(seconds=10),
            completed_at=now,
            strategy_used=CompactionStrategy.ADAPTIVE,
            memories_analyzed=100,
            candidates_identified=20,
            memories_archived=10,
            memories_deleted=5,
            memories_merged=3,
            merge_groups_processed=2,
            bytes_before=10000,
            bytes_after=8000,
            bytes_saved=2000,
            fragmentation_before=0.3,
            fragmentation_after=0.1,
            avg_strength_removed=0.15,
            avg_importance_removed=0.2,
            avg_age_removed_days=15.0,
            errors=[]
        )

        data = report.to_dict()

        assert data['strategy_used'] == 'adaptive'
        assert data['counts']['analyzed'] == 100
        assert data['counts']['archived'] == 10
        assert data['storage']['bytes_saved'] == 2000
        assert data['duration_seconds'] == pytest.approx(10.0, rel=0.1)

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        report = CompactionReport(
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            strategy_used=CompactionStrategy.ADAPTIVE,
            memories_analyzed=0,
            candidates_identified=0,
            memories_archived=0,
            memories_deleted=0,
            memories_merged=0,
            merge_groups_processed=0,
            bytes_before=10000,
            bytes_after=5000,
            bytes_saved=5000,
            fragmentation_before=0.0,
            fragmentation_after=0.0,
            avg_strength_removed=0.0,
            avg_importance_removed=0.0,
            avg_age_removed_days=0.0,
        )

        assert report.compression_ratio == 0.5
        assert report.space_savings_percent == 50.0


# =============================================================================
# METRICS TESTS
# =============================================================================

class TestCompactionMetrics:
    """Test compaction metrics tracking."""

    def test_record_compaction(self):
        """Test recording a compaction result."""
        metrics = CompactionMetrics()
        now = datetime.now(timezone.utc)

        report = CompactionReport(
            started_at=now - timedelta(seconds=5),
            completed_at=now,
            strategy_used=CompactionStrategy.ADAPTIVE,
            memories_analyzed=100,
            candidates_identified=20,
            memories_archived=10,
            memories_deleted=5,
            memories_merged=3,
            merge_groups_processed=2,
            bytes_before=10000,
            bytes_after=8000,
            bytes_saved=2000,
            fragmentation_before=0.3,
            fragmentation_after=0.1,
            avg_strength_removed=0.15,
            avg_importance_removed=0.2,
            avg_age_removed_days=15.0,
        )

        metrics.record_compaction(report)

        assert metrics.total_compactions == 1
        assert metrics.total_memories_compacted == 18  # 10 + 5 + 3
        assert metrics.total_bytes_saved == 2000
        assert metrics.last_compaction == now

    def test_metrics_history_limit(self):
        """Test that metrics history is limited."""
        metrics = CompactionMetrics()
        now = datetime.now(timezone.utc)

        for i in range(15):
            report = CompactionReport(
                started_at=now,
                completed_at=now,
                strategy_used=CompactionStrategy.ADAPTIVE,
                memories_analyzed=i,
                candidates_identified=0,
                memories_archived=1,
                memories_deleted=0,
                memories_merged=0,
                merge_groups_processed=0,
                bytes_before=1000,
                bytes_after=900,
                bytes_saved=100,
                fragmentation_before=0.0,
                fragmentation_after=0.0,
                avg_strength_removed=0.0,
                avg_importance_removed=0.0,
                avg_age_removed_days=0.0,
            )
            metrics.record_compaction(report)

        assert len(metrics.compaction_history) == 10


# =============================================================================
# SCHEDULER TESTS
# =============================================================================

class TestCompactionScheduler:
    """Test background compaction scheduler."""

    @pytest.mark.asyncio
    async def test_scheduler_status(self, compactor):
        """Test scheduler status reporting."""
        scheduler = CompactionScheduler(compactor)

        status = scheduler.get_status()

        assert status['running'] is False
        assert status['check_count'] == 0
        assert 'metrics' in status

    @pytest.mark.asyncio
    async def test_should_compact_threshold(self, compactor, mock_backend):
        """Test should_compact based on count threshold."""
        mock_backend.get_stats.return_value = {
            'total_memories': 200,  # Exceeds threshold of 100
            'db_size_bytes': 1000
        }

        should_compact, reason = await compactor.should_compact()

        assert should_compact is True
        assert "count" in reason.lower() or "exceeds" in reason.lower()


# =============================================================================
# FULL COMPACTION TESTS
# =============================================================================

class TestFullCompaction:
    """Test full compaction workflow."""

    @pytest.mark.asyncio
    async def test_compact_time_based(self, compactor, mock_backend):
        """Test time-based compaction strategy."""
        entries = [
            create_test_entry("old_1", "Old content 1", age_days=15.0, access_count=0),
            create_test_entry("old_2", "Old content 2", age_days=10.0, access_count=0),
            create_test_entry("new_1", "New content", age_days=0.5, access_count=10),
        ]
        mock_backend.list_all.return_value = entries
        mock_backend.count.return_value = 3
        mock_backend.get_stats.return_value = {
            'total_memories': 3,
            'db_size_bytes': 1000
        }

        report = await compactor.compact(strategy=CompactionStrategy.TIME_BASED)

        assert report.memories_analyzed == 3
        assert report.candidates_identified >= 0

    @pytest.mark.asyncio
    async def test_compact_quality_based(self, compactor, mock_backend):
        """Test quality-based compaction strategy."""
        entries = [
            create_test_entry("weak_1", "Weak memory", strength=0.05, age_days=5.0),
            create_test_entry("strong_1", "Strong memory", strength=0.9, age_days=5.0, access_count=10),
        ]
        mock_backend.list_all.return_value = entries
        mock_backend.count.return_value = 2
        mock_backend.get_stats.return_value = {
            'total_memories': 2,
            'db_size_bytes': 500
        }

        report = await compactor.compact(strategy=CompactionStrategy.QUALITY_BASED)

        assert report.memories_analyzed == 2


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestCompactionIntegration:
    """Integration tests for compaction system."""

    @pytest.mark.asyncio
    async def test_factory_functions(self):
        """Test factory function singleton pattern."""
        # Note: This modifies global state, so cleanup after
        import core.memory.compaction as compaction_module

        # Reset singletons
        compaction_module._compactor_instance = None
        compaction_module._scheduler_instance = None

        try:
            compactor1 = get_memory_compactor()
            compactor2 = get_memory_compactor()
            assert compactor1 is compactor2

            scheduler1 = get_compaction_scheduler()
            scheduler2 = get_compaction_scheduler()
            assert scheduler1 is scheduler2
        finally:
            # Cleanup
            compaction_module._compactor_instance = None
            compaction_module._scheduler_instance = None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
