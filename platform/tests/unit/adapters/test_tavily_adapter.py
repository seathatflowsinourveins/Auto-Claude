"""
Unit Tests for TavilyAdapter

Tests the Tavily AI search adapter in isolation with mocked dependencies.
"""

import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

# Import adapter under test
try:
    from adapters.tavily_adapter import (
        TavilyAdapter,
        TavilySearchDepth,
        TavilyTopic,
        TavilyResearchModel,
        TavilyCitationFormat,
        TAVILY_AVAILABLE,
    )
    from core.orchestration.base import (
        AdapterResult,
        AdapterStatus,
        SDKLayer,
    )
except ImportError:
    pytest.skip("Tavily adapter not available", allow_module_level=True)


@pytest.fixture
def clean_env():
    """Ensure no API keys in environment so adapter runs in mock mode."""
    with patch.dict(os.environ, {}, clear=True):
        yield


class TestTavilyAdapterProperties:
    """Tests for TavilyAdapter basic properties."""

    def test_sdk_name(self):
        """Adapter should return 'tavily' as sdk_name."""
        adapter = TavilyAdapter()
        assert adapter.sdk_name == "tavily"

    def test_layer(self):
        """Adapter should be in RESEARCH layer."""
        adapter = TavilyAdapter()
        assert adapter.layer == SDKLayer.RESEARCH

    def test_available_depends_on_sdk(self):
        """available property should reflect SDK installation."""
        adapter = TavilyAdapter()
        assert adapter.available == TAVILY_AVAILABLE


class TestTavilyAdapterInitialization:
    """Tests for TavilyAdapter initialization."""

    @pytest.mark.asyncio
    async def test_initialization_without_api_key(self, clean_env):
        """Initialization without API key should enter degraded mode."""
        adapter = TavilyAdapter()
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            result = await adapter.initialize({})

        assert result.success is True
        assert result.data.get("status") == "degraded"

    @pytest.mark.asyncio
    async def test_initialization_with_api_key(self, clean_env):
        """Initialization with API key should succeed."""
        adapter = TavilyAdapter()
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            with patch('adapters.tavily_adapter.TavilyClient'):
                with patch('adapters.tavily_adapter.AsyncTavilyClient'):
                    result = await adapter.initialize({"api_key": "test-key"})

        assert result.success is True
        assert result.data.get("status") == "ready"

    @pytest.mark.asyncio
    async def test_initialization_lists_features(self, clean_env):
        """Initialization with API key should list available features."""
        adapter = TavilyAdapter()
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            with patch('adapters.tavily_adapter.TavilyClient'):
                with patch('adapters.tavily_adapter.AsyncTavilyClient'):
                    result = await adapter.initialize({"api_key": "test-key"})

        assert "features" in result.data
        expected_features = ["search", "research", "extract", "qna", "context", "map", "crawl"]
        for feature in expected_features:
            assert feature in result.data["features"]

    @pytest.mark.asyncio
    async def test_initialization_lists_search_depths(self, clean_env):
        """Initialization with API key should list search depths."""
        adapter = TavilyAdapter()
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            with patch('adapters.tavily_adapter.TavilyClient'):
                with patch('adapters.tavily_adapter.AsyncTavilyClient'):
                    result = await adapter.initialize({"api_key": "test-key"})

        assert "search_depths" in result.data
        assert "basic" in result.data["search_depths"]
        assert "advanced" in result.data["search_depths"]


class TestTavilyAdapterSearch:
    """Tests for search operation."""

    @pytest.fixture
    def adapter(self, clean_env):
        return TavilyAdapter()

    @pytest.mark.asyncio
    async def test_search_basic(self, adapter):
        """Basic search should work in mock mode."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        result = await adapter.execute("search", query="test query")

        assert result.success is True
        assert "results" in result.data
        assert result.data.get("mock") is True

    @pytest.mark.asyncio
    async def test_search_with_depth_advanced(self, adapter):
        """Search with advanced depth should work."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        result = await adapter.execute(
            "search",
            query="test",
            search_depth="advanced"
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_search_with_topic_news(self, adapter):
        """Search with news topic should work."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        result = await adapter.execute(
            "search",
            query="AI news",
            topic="news"
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_search_with_topic_finance(self, adapter):
        """Search with finance topic should work."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        result = await adapter.execute(
            "search",
            query="stock market",
            topic="finance"
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_search_with_domain_filters(self, adapter):
        """Search with domain filters should work."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        result = await adapter.execute(
            "search",
            query="test",
            include_domains=["github.com"],
            exclude_domains=["reddit.com"]
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_search_with_time_range(self, adapter):
        """Search with time range should work."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        result = await adapter.execute(
            "search",
            query="recent news",
            time_range="week"
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_search_with_country(self, adapter):
        """Search with country should work."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        result = await adapter.execute(
            "search",
            query="local news",
            country="US"
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_search_include_answer(self, adapter):
        """Search with include_answer should work."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        result = await adapter.execute(
            "search",
            query="What is Python?",
            include_answer=True
        )

        assert result.success is True
        assert "answer" in result.data

    @pytest.mark.asyncio
    async def test_search_include_images(self, adapter):
        """Search with include_images should work."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        result = await adapter.execute(
            "search",
            query="cats",
            include_images=True
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_search_increments_stats(self, adapter):
        """Search should increment stats."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        initial = adapter._stats["searches"]
        await adapter.execute("search", query="test")

        assert adapter._stats["searches"] == initial + 1


class TestTavilyAdapterResearch:
    """Tests for research operation."""

    @pytest.fixture
    def adapter(self, clean_env):
        return TavilyAdapter()

    @pytest.mark.asyncio
    async def test_research_synchronous(self, adapter):
        """Synchronous research should work."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        result = await adapter.execute(
            "research",
            query="AI safety"
        )

        assert result.success is True
        assert "report" in result.data or "mock" in result.data

    @pytest.mark.asyncio
    async def test_research_async_mode(self, adapter):
        """Async research mode should work."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        result = await adapter.execute(
            "research",
            query="AI safety",
            async_mode=True
        )

        assert result.success is True
        assert "report" in result.data or "request_id" in result.data or "mock" in result.data

    @pytest.mark.asyncio
    async def test_research_with_citation_format(self, adapter):
        """Research with citation format should work."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        result = await adapter.execute(
            "research",
            query="AI research",
            citation_format="apa"
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_research_increments_stats(self, adapter):
        """Research should increment stats."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        initial = adapter._stats["research_queries"]
        await adapter.execute("research", query="test")

        assert adapter._stats["research_queries"] == initial + 1


class TestTavilyAdapterGetResearch:
    """Tests for get_research operation."""

    @pytest.fixture
    def adapter(self, clean_env):
        return TavilyAdapter()

    @pytest.mark.asyncio
    async def test_get_research_mock(self, adapter):
        """Get research should work in mock mode."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        result = await adapter.execute(
            "get_research",
            request_id="test-request-id"
        )

        assert result.success is True
        assert "request_id" in result.data


class TestTavilyAdapterExtract:
    """Tests for extract operation."""

    @pytest.fixture
    def adapter(self, clean_env):
        return TavilyAdapter()

    @pytest.mark.asyncio
    async def test_extract_single_url(self, adapter):
        """Extract from single URL should work."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        result = await adapter.execute(
            "extract",
            urls=["https://example.com"]
        )

        assert result.success is True
        assert "results" in result.data

    @pytest.mark.asyncio
    async def test_extract_multiple_urls(self, adapter):
        """Extract from multiple URLs should work."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        result = await adapter.execute(
            "extract",
            urls=["https://example.com", "https://example.org"]
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_extract_increments_stats(self, adapter):
        """Extract should increment stats."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        initial = adapter._stats["extractions"]
        await adapter.execute("extract", urls=["https://example.com", "https://example.org"])

        assert adapter._stats["extractions"] == initial + 2


class TestTavilyAdapterQnA:
    """Tests for qna operation."""

    @pytest.fixture
    def adapter(self, clean_env):
        return TavilyAdapter()

    @pytest.mark.asyncio
    async def test_qna_basic(self, adapter):
        """QnA should work."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        result = await adapter.execute(
            "qna",
            query="What is the capital of France?"
        )

        assert result.success is True
        assert "answer" in result.data


class TestTavilyAdapterContext:
    """Tests for context operation."""

    @pytest.fixture
    def adapter(self, clean_env):
        return TavilyAdapter()

    @pytest.mark.asyncio
    async def test_context_basic(self, adapter):
        """Context should work."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        result = await adapter.execute(
            "context",
            query="Python programming"
        )

        assert result.success is True
        assert "context" in result.data

    @pytest.mark.asyncio
    async def test_context_with_max_tokens(self, adapter):
        """Context with max_tokens should work."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        result = await adapter.execute(
            "context",
            query="Python",
            max_tokens=2000
        )

        assert result.success is True


class TestTavilyAdapterMap:
    """Tests for map operation."""

    @pytest.fixture
    def adapter(self, clean_env):
        return TavilyAdapter()

    @pytest.mark.asyncio
    async def test_map_website(self, adapter):
        """Map should work."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        result = await adapter.execute(
            "map",
            url="https://example.com"
        )

        assert result.success is True
        assert "results" in result.data

    @pytest.mark.asyncio
    async def test_map_with_depth(self, adapter):
        """Map with depth should work."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        result = await adapter.execute(
            "map",
            url="https://example.com",
            max_depth=3
        )

        assert result.success is True


class TestTavilyAdapterCrawl:
    """Tests for crawl operation."""

    @pytest.fixture
    def adapter(self, clean_env):
        return TavilyAdapter()

    @pytest.mark.asyncio
    async def test_crawl_website(self, adapter):
        """Crawl should work."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        result = await adapter.execute(
            "crawl",
            url="https://example.com"
        )

        assert result.success is True
        assert "results" in result.data

    @pytest.mark.asyncio
    async def test_crawl_with_format(self, adapter):
        """Crawl with format should work."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        result = await adapter.execute(
            "crawl",
            url="https://example.com",
            format="markdown"
        )

        assert result.success is True


class TestTavilyAdapterHealthAndShutdown:
    """Tests for health check and shutdown."""

    @pytest.fixture
    def adapter(self, clean_env):
        return TavilyAdapter()

    @pytest.mark.asyncio
    async def test_health_check_degraded(self, adapter):
        """Health check in degraded mode should work."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
            result = await adapter.health_check()

        assert result.success is True
        assert result.data.get("status") in ["healthy", "degraded"]

    @pytest.mark.asyncio
    async def test_shutdown_returns_stats(self, adapter):
        """Shutdown should return stats."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        await adapter.execute("search", query="test")
        result = await adapter.shutdown()

        assert result.success is True
        assert "stats" in result.data

    @pytest.mark.asyncio
    async def test_get_stats(self, adapter):
        """get_stats should return stats dictionary."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        await adapter.execute("search", query="test")
        stats = adapter.get_stats()

        assert "searches" in stats
        assert stats["searches"] >= 1


class TestTavilyAdapterErrorHandling:
    """Tests for error handling."""

    @pytest.fixture
    def adapter(self, clean_env):
        return TavilyAdapter()

    @pytest.mark.asyncio
    async def test_unknown_operation_returns_error(self, adapter):
        """Unknown operation should return error."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        result = await adapter.execute("__invalid__")

        assert result.success is False
        assert "Unknown operation" in result.error

    @pytest.mark.asyncio
    async def test_error_includes_latency(self, adapter):
        """Error results should include latency."""
        with patch('adapters.tavily_adapter.TAVILY_AVAILABLE', True):
            await adapter.initialize({})
        result = await adapter.execute("__invalid__")

        assert result.latency_ms >= 0


class TestTavilyEnums:
    """Tests for Tavily enums."""

    def test_search_depth_values(self):
        """Verify search depth enum values."""
        assert TavilySearchDepth.BASIC.value == "basic"
        assert TavilySearchDepth.ADVANCED.value == "advanced"
        assert TavilySearchDepth.FAST.value == "fast"

    def test_topic_values(self):
        """Verify topic enum values."""
        assert TavilyTopic.GENERAL.value == "general"
        assert TavilyTopic.NEWS.value == "news"
        assert TavilyTopic.FINANCE.value == "finance"

    def test_research_model_values(self):
        """Verify research model enum values."""
        assert TavilyResearchModel.MINI.value == "mini"
        assert TavilyResearchModel.PRO.value == "pro"
        assert TavilyResearchModel.AUTO.value == "auto"

    def test_citation_format_values(self):
        """Verify citation format enum values."""
        assert TavilyCitationFormat.NUMBERED.value == "numbered"
        assert TavilyCitationFormat.MLA.value == "mla"
        assert TavilyCitationFormat.APA.value == "apa"
        assert TavilyCitationFormat.CHICAGO.value == "chicago"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
