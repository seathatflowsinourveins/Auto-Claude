"""
Ultimate Research Swarm Tests
=============================

Tests for the UltimateResearchSwarm - the crown jewel of UNLEASH.
"""

import pytest
import asyncio
from typing import List
from dataclasses import dataclass

# Import the module under test
try:
    from platform.core.ultimate_research_swarm import (
        UltimateResearchSwarm,
        ResearchSwarmConfig,
        ResearchDepth,
        ResearchAgentType,
        ResearchSource,
        SynthesizedResult,
        SynthesisQueen,
        ResearchMemoryManager,
        UltimateResearchResult,
        get_ultimate_swarm,
        quick_research,
        comprehensive_research,
        deep_research,
    )
except ImportError:
    from core.ultimate_research_swarm import (
        UltimateResearchSwarm,
        ResearchSwarmConfig,
        ResearchDepth,
        ResearchAgentType,
        ResearchSource,
        SynthesizedResult,
        SynthesisQueen,
        ResearchMemoryManager,
        UltimateResearchResult,
        get_ultimate_swarm,
        quick_research,
        comprehensive_research,
        deep_research,
    )


# =============================================================================
# Test Data
# =============================================================================

@pytest.fixture
def sample_sources() -> List[ResearchSource]:
    """Sample research sources for testing."""
    return [
        ResearchSource(
            tool="exa",
            title="LangGraph StateGraph Guide",
            url="https://example.com/langgraph",
            content="LangGraph StateGraph is a powerful abstraction for building stateful workflows...",
            score=0.95,
        ),
        ResearchSource(
            tool="tavily",
            title="LangGraph Tutorial",
            url="https://example.com/tutorial",
            content="StateGraph allows you to define nodes and edges that represent your workflow...",
            score=0.88,
        ),
        ResearchSource(
            tool="jina",
            title="Official Documentation",
            url="https://langchain.com/docs",
            content="The StateGraph class provides a way to orchestrate multi-step agent workflows...",
            score=0.92,
        ),
    ]


@pytest.fixture
def swarm_config() -> ResearchSwarmConfig:
    """Default swarm configuration for tests."""
    return ResearchSwarmConfig(
        max_agents=4,
        topology="hierarchical",
        synthesis_max_kb=4,
    )


# =============================================================================
# ResearchSource Tests
# =============================================================================

class TestResearchSource:
    """Tests for ResearchSource dataclass."""

    def test_create_source(self):
        """Test basic source creation."""
        source = ResearchSource(
            tool="exa",
            title="Test Title",
            url="https://example.com",
            content="Test content",
            score=0.9,
        )
        assert source.tool == "exa"
        assert source.title == "Test Title"
        assert source.score == 0.9
        assert source.content_hash != ""

    def test_content_hash_generated(self):
        """Test that content hash is auto-generated."""
        source = ResearchSource(
            tool="tavily",
            title="Title",
            url="",
            content="Some unique content here",
        )
        assert len(source.content_hash) == 32  # MD5 hex length

    def test_same_content_same_hash(self):
        """Test that same content produces same hash."""
        content = "This is the same content"
        source1 = ResearchSource(tool="exa", title="A", url="", content=content)
        source2 = ResearchSource(tool="tavily", title="B", url="", content=content)
        assert source1.content_hash == source2.content_hash


# =============================================================================
# SynthesisQueen Tests
# =============================================================================

class TestSynthesisQueen:
    """Tests for the SynthesisQueen."""

    def test_synthesize_empty_sources(self, swarm_config):
        """Test synthesis with no sources."""
        queen = SynthesisQueen(swarm_config)
        result = queen.synthesize([], "test query")

        assert result.summary == "No sources found for this query."
        assert result.confidence == 0.0
        assert len(result.sources) == 0

    def test_synthesize_deduplication(self, swarm_config):
        """Test that duplicate content is deduplicated."""
        queen = SynthesisQueen(swarm_config)

        # Create sources with duplicate content
        sources = [
            ResearchSource(tool="exa", title="A", url="", content="Same content here", score=0.9),
            ResearchSource(tool="tavily", title="B", url="", content="Same content here", score=0.8),
            ResearchSource(tool="jina", title="C", url="", content="Different content", score=0.7),
        ]

        result = queen.synthesize(sources, "test")

        # Should deduplicate to 2 unique sources
        assert len(result.sources) == 2

    def test_synthesize_sorts_by_score(self, swarm_config):
        """Test that sources are sorted by score."""
        queen = SynthesisQueen(swarm_config)

        sources = [
            ResearchSource(tool="exa", title="Low", url="", content="Content A", score=0.5),
            ResearchSource(tool="tavily", title="High", url="", content="Content B", score=0.95),
            ResearchSource(tool="jina", title="Mid", url="", content="Content C", score=0.75),
        ]

        result = queen.synthesize(sources, "test")

        # First source should have highest score
        assert result.sources[0].score == 0.95
        assert result.sources[0].title == "High"

    def test_synthesize_extracts_findings(self, swarm_config, sample_sources):
        """Test key findings extraction."""
        queen = SynthesisQueen(swarm_config)
        result = queen.synthesize(sample_sources, "LangGraph")

        assert len(result.key_findings) > 0
        assert len(result.key_findings) <= 5  # Max 5 findings

    def test_synthesize_calculates_confidence(self, swarm_config, sample_sources):
        """Test confidence calculation."""
        queen = SynthesisQueen(swarm_config)
        result = queen.synthesize(sample_sources, "test")

        # With 3 tools and good scores, confidence should be reasonable
        assert 0.5 < result.confidence < 1.0

    def test_synthesize_builds_summary(self, swarm_config, sample_sources):
        """Test summary building."""
        queen = SynthesisQueen(swarm_config)
        result = queen.synthesize(sample_sources, "test")

        assert len(result.summary) > 0
        # Summary should contain content from sources
        assert "LangGraph" in result.summary or "StateGraph" in result.summary


# =============================================================================
# UltimateResearchSwarm Tests
# =============================================================================

class TestUltimateResearchSwarm:
    """Tests for the UltimateResearchSwarm."""

    @pytest.fixture
    def swarm(self, swarm_config):
        """Create a swarm instance."""
        return UltimateResearchSwarm(swarm_config)

    @pytest.mark.asyncio
    async def test_initialize(self, swarm):
        """Test swarm initialization."""
        status = await swarm.initialize()

        # Should return dict with status for each component
        assert isinstance(status, dict)
        # At least one key should exist
        assert len(status) > 0

    @pytest.mark.asyncio
    async def test_research_quick(self, swarm):
        """Test quick research mode."""
        await swarm.initialize()

        result = await swarm.research(
            "LangGraph StateGraph",
            depth=ResearchDepth.QUICK,
        )

        assert isinstance(result, UltimateResearchResult)
        assert result.query == "LangGraph StateGraph"
        assert result.depth == ResearchDepth.QUICK
        # Quick should use minimal agents
        assert result.agents_spawned <= 2

        await swarm.shutdown()

    @pytest.mark.asyncio
    async def test_research_comprehensive(self, swarm):
        """Test comprehensive research mode."""
        await swarm.initialize()

        result = await swarm.research(
            "distributed consensus algorithms",
            depth=ResearchDepth.COMPREHENSIVE,
        )

        assert isinstance(result, UltimateResearchResult)
        assert result.depth == ResearchDepth.COMPREHENSIVE
        # Comprehensive should spawn more agents
        assert result.agents_spawned >= 1

        await swarm.shutdown()

    @pytest.mark.asyncio
    async def test_research_with_memory_key(self, swarm):
        """Test research with memory storage."""
        await swarm.initialize()

        result = await swarm.research(
            "test query",
            depth=ResearchDepth.QUICK,
            memory_key="test_memory_key",
        )

        assert result.memory_key == "test_memory_key"

        await swarm.shutdown()

    @pytest.mark.asyncio
    async def test_get_stats(self, swarm):
        """Test statistics tracking."""
        await swarm.initialize()

        # Execute a query
        await swarm.research("test", depth=ResearchDepth.QUICK)

        stats = swarm.get_stats()
        assert stats["total_queries"] >= 1
        assert "agents_spawned" in stats

        await swarm.shutdown()

    @pytest.mark.asyncio
    async def test_get_adapter_status(self, swarm):
        """Test adapter status retrieval."""
        await swarm.initialize()

        status = swarm.get_adapter_status()
        assert isinstance(status, dict)

        await swarm.shutdown()


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_get_ultimate_swarm_singleton(self):
        """Test that get_ultimate_swarm returns singleton."""
        swarm1 = get_ultimate_swarm()
        swarm2 = get_ultimate_swarm()
        assert swarm1 is swarm2


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full research pipeline."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_research_pipeline(self):
        """Test complete research pipeline."""
        swarm = UltimateResearchSwarm()
        await swarm.initialize()

        # Execute research
        result = await swarm.research(
            "Python async/await patterns",
            depth=ResearchDepth.STANDARD,
        )

        # Verify result structure
        assert result.query == "Python async/await patterns"
        assert isinstance(result.summary, str)
        assert isinstance(result.key_findings, list)
        assert isinstance(result.patterns, list)
        assert isinstance(result.sources, list)
        assert result.latency_ms > 0

        await swarm.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_memory_recall(self):
        """Test memory storage and recall."""
        swarm = UltimateResearchSwarm()
        await swarm.initialize()

        # First query - should search
        result1 = await swarm.research(
            "unique test topic for memory",
            depth=ResearchDepth.QUICK,
            memory_key="test_recall_key",
        )

        assert not result1.recalled_from_memory

        # Second query with same topic should potentially hit memory
        # (depends on memory backend availability)
        result2 = await swarm.research(
            "unique test topic for memory",
            depth=ResearchDepth.QUICK,
            check_memory_first=True,
        )

        # At minimum, query should complete
        assert result2.query == "unique test topic for memory"

        await swarm.shutdown()


# =============================================================================
# ResearchDepth Tests
# =============================================================================

class TestResearchDepth:
    """Tests for ResearchDepth enum."""

    def test_depth_values(self):
        """Test depth enum values."""
        assert ResearchDepth.QUICK.value == "quick"
        assert ResearchDepth.STANDARD.value == "standard"
        assert ResearchDepth.COMPREHENSIVE.value == "comprehensive"
        assert ResearchDepth.DEEP.value == "deep"

    def test_depth_from_string(self):
        """Test creating depth from string."""
        depth = ResearchDepth("comprehensive")
        assert depth == ResearchDepth.COMPREHENSIVE


# =============================================================================
# ResearchAgentType Tests
# =============================================================================

class TestResearchAgentType:
    """Tests for ResearchAgentType enum."""

    def test_agent_types(self):
        """Test agent type values."""
        assert ResearchAgentType.EXA_NEURAL.value == "exa-neural"
        assert ResearchAgentType.TAVILY_AI.value == "tavily-ai"
        assert ResearchAgentType.JINA_READER.value == "jina-reader"
        assert ResearchAgentType.PERPLEXITY_DEEP.value == "perplexity"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
