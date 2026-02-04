"""
Tests for Memory Compression System

Tests the semantic compression functionality including:
- Compression strategies (extractive, abstractive, hierarchical, clustering)
- Compression pipeline
- Decompression and caching
- Background compressor
- Compression metrics
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from core.memory.compression import (
    CompressionStrategy,
    CompressionTrigger,
    CompressionConfig,
    CompressionResult,
    CompressionMetrics,
    CompressedMemory,
    ExtractiveStrategy,
    AbstractiveStrategy,
    HierarchicalStrategy,
    ClusteringStrategy,
    MemoryCompressor,
    BackgroundCompressor,
)
from core.memory.backends.base import (
    MemoryEntry,
    MemoryTier,
    MemoryPriority,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_memories():
    """Create sample memory entries for testing."""
    now = datetime.now(timezone.utc)
    old_date = now - timedelta(days=30)

    return [
        MemoryEntry(
            id="mem_001",
            content="The authentication system uses JWT tokens for session management. "
                    "Tokens expire after 24 hours and must be refreshed using the refresh token endpoint.",
            tier=MemoryTier.ARCHIVAL_MEMORY,
            priority=MemoryPriority.NORMAL,
            created_at=old_date,
            last_accessed=old_date + timedelta(days=5),
            access_count=3,
            tags=["auth", "jwt", "security"],
            metadata={"importance": 0.6},
        ),
        MemoryEntry(
            id="mem_002",
            content="User passwords are hashed using bcrypt with a cost factor of 12. "
                    "We decided to use bcrypt over argon2 due to wider library support.",
            tier=MemoryTier.ARCHIVAL_MEMORY,
            priority=MemoryPriority.NORMAL,
            created_at=old_date,
            last_accessed=old_date + timedelta(days=3),
            access_count=2,
            tags=["auth", "security", "password"],
            metadata={"importance": 0.7},
        ),
        MemoryEntry(
            id="mem_003",
            content="The API rate limiting is set to 100 requests per minute per user. "
                    "This applies to all authenticated endpoints. Rate limits are stored in Redis.",
            tier=MemoryTier.ARCHIVAL_MEMORY,
            priority=MemoryPriority.NORMAL,
            created_at=old_date,
            last_accessed=old_date + timedelta(days=7),
            access_count=4,
            tags=["api", "security", "rate-limiting"],
            metadata={"importance": 0.5},
        ),
        MemoryEntry(
            id="mem_004",
            content="Security headers are configured in the middleware layer. "
                    "We use helmet.js for Express applications to set HSTS, CSP, and X-Frame-Options.",
            tier=MemoryTier.ARCHIVAL_MEMORY,
            priority=MemoryPriority.NORMAL,
            created_at=old_date,
            last_accessed=old_date + timedelta(days=10),
            access_count=1,
            tags=["security", "headers", "middleware"],
            metadata={"importance": 0.4},
        ),
    ]


@pytest.fixture
def compression_config():
    """Create compression config for testing."""
    return CompressionConfig(
        default_strategy=CompressionStrategy.EXTRACTIVE,
        min_age_days=1.0,  # Lower for testing
        min_group_size=2,
        max_group_size=10,
        target_compression_ratio=0.3,
        similarity_threshold=0.5,
        cache_decompressed=True,
        cache_ttl_seconds=60,
    )


@pytest.fixture
def mock_backend():
    """Create a mock backend for testing."""
    backend = AsyncMock()
    backend.get = AsyncMock(return_value=None)
    backend.put = AsyncMock()
    backend.list_all = AsyncMock(return_value=[])
    backend.get_stats = AsyncMock(return_value={"total_memories": 0})
    return backend


# =============================================================================
# EXTRACTIVE STRATEGY TESTS
# =============================================================================

class TestExtractiveStrategy:
    """Tests for extractive compression strategy."""

    @pytest.mark.asyncio
    async def test_compress_extracts_key_sentences(self, sample_memories, compression_config):
        """Test that extractive strategy extracts key sentences."""
        strategy = ExtractiveStrategy()
        summary, key_facts, relationships = await strategy.compress(
            sample_memories, compression_config
        )

        # Should produce a summary
        assert len(summary) > 0

        # Should extract key facts
        assert len(key_facts) > 0

        # Summary should be shorter than combined originals
        original_length = sum(len(m.content) for m in sample_memories)
        assert len(summary) < original_length

    @pytest.mark.asyncio
    async def test_compress_preserves_important_content(self, sample_memories, compression_config):
        """Test that important terms are preserved."""
        strategy = ExtractiveStrategy()
        summary, key_facts, _ = await strategy.compress(
            sample_memories, compression_config
        )

        combined = summary.lower() + " ".join(key_facts).lower()

        # Check for presence of key terms
        important_terms = ["jwt", "authentication", "security", "password", "bcrypt"]
        found_terms = sum(1 for term in important_terms if term in combined)

        # At least some important terms should be preserved
        assert found_terms >= 2

    def test_score_retention_calculates_correctly(self, sample_memories):
        """Test retention scoring."""
        strategy = ExtractiveStrategy()

        summary = "JWT authentication with bcrypt password hashing."
        key_facts = ["Uses JWT tokens", "Bcrypt for passwords"]

        score = strategy.score_retention(sample_memories, summary, key_facts)

        # Score should be between 0 and 1
        assert 0.0 <= score <= 1.0
        # With some matching content, should have reasonable retention
        assert score > 0.1

    def test_split_sentences_handles_various_formats(self):
        """Test sentence splitting."""
        strategy = ExtractiveStrategy()

        text = "First sentence. Second sentence! Third sentence? Final sentence."
        sentences = strategy._split_sentences(text)

        assert len(sentences) == 4
        assert "First sentence" in sentences[0]

    def test_score_sentence_weights_position(self):
        """Test that sentence position affects score."""
        strategy = ExtractiveStrategy()

        first_score = strategy._score_sentence(
            "This is a test sentence.",
            position=0,
            total_sentences=5,
            access_count=0,
            method="tfidf"
        )

        last_score = strategy._score_sentence(
            "This is a test sentence.",
            position=4,
            total_sentences=5,
            access_count=0,
            method="tfidf"
        )

        middle_score = strategy._score_sentence(
            "This is a test sentence.",
            position=2,
            total_sentences=5,
            access_count=0,
            method="tfidf"
        )

        # First and last should score higher than middle
        assert first_score > middle_score
        assert last_score > middle_score


# =============================================================================
# ABSTRACTIVE STRATEGY TESTS
# =============================================================================

class TestAbstractiveStrategy:
    """Tests for abstractive compression strategy."""

    @pytest.mark.asyncio
    async def test_falls_back_to_extractive_without_llm(
        self, sample_memories, compression_config
    ):
        """Test fallback when no LLM provider."""
        strategy = AbstractiveStrategy(llm_provider=None)
        summary, key_facts, _ = await strategy.compress(
            sample_memories, compression_config
        )

        # Should still produce output via fallback
        assert len(summary) > 0
        assert len(key_facts) > 0

    @pytest.mark.asyncio
    async def test_uses_llm_when_available(
        self, sample_memories, compression_config
    ):
        """Test LLM integration."""
        mock_llm = MagicMock(return_value="""
Summary: Authentication uses JWT with bcrypt password hashing.

Key Facts:
- JWT tokens for session management
- Bcrypt for password hashing
- Rate limiting at 100 req/min

Relationships:
- User -> authenticates_via -> JWT
        """)

        strategy = AbstractiveStrategy(llm_provider=mock_llm)
        summary, key_facts, relationships = await strategy.compress(
            sample_memories, compression_config
        )

        # Should have called LLM
        mock_llm.assert_called_once()

        # Should parse response
        assert len(summary) > 0


# =============================================================================
# HIERARCHICAL STRATEGY TESTS
# =============================================================================

class TestHierarchicalStrategy:
    """Tests for hierarchical compression strategy."""

    @pytest.mark.asyncio
    async def test_creates_multi_level_compression(
        self, sample_memories, compression_config
    ):
        """Test hierarchical compression with multiple levels."""
        strategy = HierarchicalStrategy()

        # Increase min group size for hierarchical to kick in
        compression_config.min_group_size = 2
        compression_config.hierarchy_levels = 2

        summary, key_facts, _ = await strategy.compress(
            sample_memories, compression_config
        )

        assert len(summary) > 0
        assert len(key_facts) > 0

    @pytest.mark.asyncio
    async def test_handles_small_groups(self, compression_config):
        """Test handling of groups smaller than hierarchy requires."""
        strategy = HierarchicalStrategy()

        small_memories = [
            MemoryEntry(id="mem_1", content="Small memory content."),
            MemoryEntry(id="mem_2", content="Another small memory."),
        ]

        summary, key_facts, _ = await strategy.compress(
            small_memories, compression_config
        )

        # Should still work with small groups
        assert len(summary) > 0


# =============================================================================
# CLUSTERING STRATEGY TESTS
# =============================================================================

class TestClusteringStrategy:
    """Tests for clustering compression strategy."""

    @pytest.mark.asyncio
    async def test_groups_similar_memories(
        self, sample_memories, compression_config
    ):
        """Test that similar memories are grouped."""
        strategy = ClusteringStrategy()

        clusters = await strategy._cluster_by_keywords(
            sample_memories, compression_config
        )

        # Should create at least one cluster
        assert len(clusters) >= 1
        # Total memories should be preserved
        total = sum(len(c) for c in clusters)
        assert total == len(sample_memories)

    @pytest.mark.asyncio
    async def test_compress_with_clustering(
        self, sample_memories, compression_config
    ):
        """Test full clustering compression."""
        strategy = ClusteringStrategy()

        summary, key_facts, relationships = await strategy.compress(
            sample_memories, compression_config
        )

        assert len(summary) > 0

    def test_cosine_similarity_calculation(self):
        """Test cosine similarity."""
        strategy = ClusteringStrategy()

        # Same vectors
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert strategy._cosine_similarity(a, b) == pytest.approx(1.0)

        # Orthogonal vectors
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert strategy._cosine_similarity(a, b) == pytest.approx(0.0)

        # Similar vectors
        a = [1.0, 1.0, 0.0]
        b = [1.0, 0.0, 0.0]
        sim = strategy._cosine_similarity(a, b)
        assert 0.5 < sim < 1.0


# =============================================================================
# MEMORY COMPRESSOR TESTS
# =============================================================================

class TestMemoryCompressor:
    """Tests for main compressor class."""

    @pytest.mark.asyncio
    async def test_compress_memories(
        self, sample_memories, compression_config, mock_backend
    ):
        """Test full compression flow."""
        # Setup mock backend
        async def mock_get(key):
            for m in sample_memories:
                if m.id == key:
                    return m
            return None

        mock_backend.get = mock_get
        mock_backend.list_all = AsyncMock(return_value=sample_memories)

        compressor = MemoryCompressor(
            backend=mock_backend,
            config=compression_config
        )

        memory_ids = [m.id for m in sample_memories]
        result = await compressor.compress_memories(
            memory_ids,
            strategy=CompressionStrategy.EXTRACTIVE,
            trigger=CompressionTrigger.MANUAL
        )

        # Should return valid result
        assert isinstance(result, CompressionResult)
        assert result.compressed_id.startswith("compressed_")
        assert len(result.original_ids) == len(memory_ids)
        assert 0 < result.compression_ratio < 1.0
        assert 0 <= result.retention_score <= 1.0

    @pytest.mark.asyncio
    async def test_decompress_with_caching(
        self, sample_memories, compression_config, mock_backend
    ):
        """Test decompression with cache."""
        # Setup mock
        compressed_meta = {
            "is_compressed": True,
            "original_ids": ["mem_001", "mem_002"],
            "key_facts": ["Fact 1", "Fact 2"],
        }
        compressed_entry = MemoryEntry(
            id="compressed_abc",
            content="Compressed summary",
            metadata=compressed_meta
        )

        mock_backend.get = AsyncMock(side_effect=[
            compressed_entry,  # First call returns compressed
            sample_memories[0],  # Second returns original 1
            sample_memories[1],  # Third returns original 2
        ])

        compressor = MemoryCompressor(
            backend=mock_backend,
            config=compression_config
        )

        # First decompression
        result1 = await compressor.decompress("compressed_abc")
        assert len(result1) == 2

        # Second should use cache
        result2 = await compressor.decompress("compressed_abc")
        assert len(result2) == 2

        # Cache should have been used
        assert compressor._metrics.cache_hits >= 1

    @pytest.mark.asyncio
    async def test_identify_candidates(
        self, sample_memories, compression_config, mock_backend
    ):
        """Test candidate identification."""
        mock_backend.list_all = AsyncMock(return_value=sample_memories)

        compressor = MemoryCompressor(
            backend=mock_backend,
            config=compression_config
        )

        groups = await compressor.identify_candidates(
            trigger=CompressionTrigger.SCHEDULED
        )

        # Should identify at least one group
        # (depends on sample_memories meeting criteria)
        assert isinstance(groups, list)

    def test_get_metrics(self, compression_config, mock_backend):
        """Test metrics retrieval."""
        compressor = MemoryCompressor(
            backend=mock_backend,
            config=compression_config
        )

        metrics = compressor.get_metrics()

        assert isinstance(metrics, CompressionMetrics)
        assert metrics.total_compressions == 0  # No compressions yet

    def test_clear_cache(self, compression_config, mock_backend):
        """Test cache clearing."""
        compressor = MemoryCompressor(
            backend=mock_backend,
            config=compression_config
        )

        # Add something to cache
        compressor._cache["test_id"] = ([], datetime.now(timezone.utc))
        assert len(compressor._cache) == 1

        compressor.clear_cache()
        assert len(compressor._cache) == 0


# =============================================================================
# BACKGROUND COMPRESSOR TESTS
# =============================================================================

class TestBackgroundCompressor:
    """Tests for background compression scheduler."""

    @pytest.mark.asyncio
    async def test_run_once(self, compression_config, mock_backend):
        """Test single compression run."""
        mock_backend.list_all = AsyncMock(return_value=[])

        compressor = MemoryCompressor(
            backend=mock_backend,
            config=compression_config
        )
        background = BackgroundCompressor(compressor)

        result = await background.run_once()

        assert "run_count" in result
        assert result["run_count"] == 1

    def test_get_status(self, compression_config, mock_backend):
        """Test status retrieval."""
        compressor = MemoryCompressor(
            backend=mock_backend,
            config=compression_config
        )
        background = BackgroundCompressor(compressor)

        status = background.get_status()

        assert "running" in status
        assert status["running"] is False
        assert "run_count" in status


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestCompressionConfig:
    """Tests for compression configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CompressionConfig()

        assert config.default_strategy == CompressionStrategy.EXTRACTIVE
        assert config.min_age_days == 7.0
        assert config.min_group_size == 3
        assert config.target_compression_ratio == 0.3

    def test_custom_values(self):
        """Test custom configuration."""
        config = CompressionConfig(
            default_strategy=CompressionStrategy.CLUSTERING,
            min_age_days=14.0,
            min_group_size=5,
            similarity_threshold=0.8
        )

        assert config.default_strategy == CompressionStrategy.CLUSTERING
        assert config.min_age_days == 14.0
        assert config.min_group_size == 5


# =============================================================================
# RESULT SERIALIZATION TESTS
# =============================================================================

class TestResultSerialization:
    """Tests for result serialization."""

    def test_compression_result_to_dict(self):
        """Test CompressionResult serialization."""
        result = CompressionResult(
            compressed_id="compressed_abc",
            original_ids=["mem_1", "mem_2"],
            strategy=CompressionStrategy.EXTRACTIVE,
            original_token_count=1000,
            compressed_token_count=300,
            compression_ratio=0.3,
            retention_score=0.85,
            coherence_score=0.9,
            coverage_score=0.8,
            trigger=CompressionTrigger.MANUAL,
        )

        data = result.to_dict()

        assert data["compressed_id"] == "compressed_abc"
        assert data["strategy"] == "extractive"
        assert data["compression_ratio"] == 0.3
        assert "compressed_at" in data

    def test_compression_metrics_to_dict(self):
        """Test CompressionMetrics serialization."""
        metrics = CompressionMetrics(
            total_compressions=10,
            total_original_tokens=5000,
            total_compressed_tokens=1500,
            average_compression_ratio=0.3,
            cache_hits=5,
            cache_misses=2,
        )

        data = metrics.to_dict()

        assert data["total_compressions"] == 10
        assert data["average_compression_ratio"] == 0.3


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for compression system."""

    @pytest.mark.asyncio
    async def test_full_compression_decompression_cycle(
        self, sample_memories, compression_config, mock_backend
    ):
        """Test complete compression and decompression cycle."""
        # Track stored entries
        stored_entries = {}

        async def mock_put(key, value):
            stored_entries[key] = value

        async def mock_get(key):
            if key in stored_entries:
                return stored_entries[key]
            for m in sample_memories:
                if m.id == key:
                    return m
            return None

        mock_backend.put = mock_put
        mock_backend.get = mock_get
        mock_backend.list_all = AsyncMock(return_value=sample_memories)

        compressor = MemoryCompressor(
            backend=mock_backend,
            config=compression_config
        )

        # Compress
        memory_ids = [m.id for m in sample_memories]
        result = await compressor.compress_memories(
            memory_ids,
            strategy=CompressionStrategy.EXTRACTIVE
        )

        # Verify compression
        assert result.compressed_id in stored_entries
        assert result.compression_ratio < 1.0

        # Decompress
        originals = await compressor.decompress(result.compressed_id)

        # Should recover all originals
        assert len(originals) == len(memory_ids)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
