"""
Unit Tests for ExaAdapter

Tests the Exa AI neural search adapter in isolation with mocked dependencies.
"""

import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

# Import adapter under test
try:
    from adapters.exa_adapter import (
        ExaAdapter,
        ExaSearchType,
        ExaCategory,
        ExaSearchResult,
        EXA_AVAILABLE,
    )
    from core.orchestration.base import (
        AdapterResult,
        AdapterStatus,
        SDKLayer,
    )
except ImportError:
    pytest.skip("Exa adapter not available", allow_module_level=True)


# Keys to clear from env so adapter enters mock mode
_EXA_ENV_KEYS = {"EXA_API_KEY"}


@pytest.fixture
def clean_env():
    """Ensure no API keys in environment so adapter runs in mock mode."""
    with patch.dict(os.environ, {}, clear=True):
        yield


class TestExaAdapterProperties:
    """Tests for ExaAdapter basic properties."""

    def test_sdk_name(self):
        """Adapter should return 'exa' as sdk_name."""
        adapter = ExaAdapter()
        assert adapter.sdk_name == "exa"

    def test_layer(self):
        """Adapter should be in RESEARCH layer."""
        adapter = ExaAdapter()
        assert adapter.layer.name == "RESEARCH"

    def test_available_depends_on_sdk(self):
        """available property should reflect SDK installation."""
        adapter = ExaAdapter()
        assert adapter.available == EXA_AVAILABLE


class TestExaAdapterInitialization:
    """Tests for ExaAdapter initialization."""

    @pytest.mark.asyncio
    async def test_initialization_without_api_key(self, clean_env):
        """Initialization without API key should enter degraded mode."""
        adapter = ExaAdapter()
        result = await adapter.initialize({})

        assert result.success is True
        assert result.data.get("status") == "degraded"

    @pytest.mark.asyncio
    async def test_initialization_with_api_key(self, clean_env):
        """Initialization with API key should succeed."""
        adapter = ExaAdapter()
        with patch('adapters.exa_adapter.EXA_AVAILABLE', True):
            with patch('adapters.exa_adapter.Exa') as mock_exa:
                result = await adapter.initialize({"api_key": "test-key"})

        assert result.success is True
        assert result.data.get("status") == "ready"
        assert "features" in result.data

    @pytest.mark.asyncio
    async def test_initialization_lists_features(self, clean_env):
        """Initialization with API key should list available features."""
        adapter = ExaAdapter()
        with patch('adapters.exa_adapter.EXA_AVAILABLE', True):
            with patch('adapters.exa_adapter.Exa'):
                result = await adapter.initialize({"api_key": "test-key"})

        assert "features" in result.data
        expected_features = ["search", "get_contents", "find_similar", "answer"]
        for feature in expected_features:
            assert feature in result.data["features"]

    @pytest.mark.asyncio
    async def test_initialization_without_sdk(self, clean_env):
        """Initialization without SDK installed should fail."""
        adapter = ExaAdapter()
        with patch('adapters.exa_adapter.EXA_AVAILABLE', False):
            result = await adapter.initialize({"api_key": "test-key"})

        assert result.success is False
        assert "not installed" in result.error.lower()


class TestExaAdapterSearch:
    """Tests for search operation."""

    @pytest.fixture
    def adapter(self, clean_env):
        """Create adapter in mock mode for testing."""
        return ExaAdapter()

    @pytest.mark.asyncio
    async def test_search_basic_query(self, adapter):
        """Basic search should work in mock mode."""
        await adapter.initialize({})
        result = await adapter.execute("search", query="test query")

        assert result.success is True
        assert "results" in result.data
        assert result.data.get("mock") is True

    @pytest.mark.asyncio
    async def test_search_with_type_fast(self, adapter):
        """Search with fast type should work."""
        await adapter.initialize({})
        result = await adapter.execute("search", query="test", type="fast")

        assert result.success is True
        assert result.data.get("type") == "fast"

    @pytest.mark.asyncio
    async def test_search_with_type_deep(self, adapter):
        """Search with deep type should work."""
        await adapter.initialize({})
        result = await adapter.execute("search", query="test", type="deep")

        assert result.success is True
        assert result.data.get("type") == "deep"

    @pytest.mark.asyncio
    async def test_search_with_type_neural(self, adapter):
        """Search with neural type should work."""
        await adapter.initialize({})
        result = await adapter.execute("search", query="test", type="neural")

        assert result.success is True
        assert result.data.get("type") == "neural"

    @pytest.mark.asyncio
    async def test_search_with_domain_filters(self, adapter):
        """Search with domain filters should work."""
        await adapter.initialize({})
        result = await adapter.execute(
            "search",
            query="test",
            include_domains=["github.com", "docs.python.org"],
            exclude_domains=["stackoverflow.com"]
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_search_with_date_filters(self, adapter):
        """Search with date filters should work."""
        await adapter.initialize({})
        result = await adapter.execute(
            "search",
            query="test",
            start_published_date="2024-01-01",
            end_published_date="2026-01-01"
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_search_with_category(self, adapter):
        """Search with category should work."""
        await adapter.initialize({})
        result = await adapter.execute(
            "search",
            query="test",
            category="research paper"
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_search_increments_stats(self, adapter):
        """Search should increment stats counter."""
        await adapter.initialize({})
        initial_count = adapter._stats["searches"]
        await adapter.execute("search", query="test")

        assert adapter._stats["searches"] == initial_count + 1


class TestExaAdapterGetContents:
    """Tests for get_contents operation."""

    @pytest.fixture
    def adapter(self, clean_env):
        return ExaAdapter()

    @pytest.mark.asyncio
    async def test_get_contents_single_url(self, adapter):
        """Get contents for single URL should work."""
        await adapter.initialize({})
        result = await adapter.execute(
            "get_contents",
            urls=["https://example.com"]
        )

        assert result.success is True
        assert "contents" in result.data

    @pytest.mark.asyncio
    async def test_get_contents_multiple_urls(self, adapter):
        """Get contents for multiple URLs should work."""
        await adapter.initialize({})
        result = await adapter.execute(
            "get_contents",
            urls=["https://example1.com", "https://example2.com"]
        )

        assert result.success is True
        assert len(result.data.get("contents", [])) >= 1

    @pytest.mark.asyncio
    async def test_get_contents_with_summary(self, adapter):
        """Get contents with summary option should work."""
        await adapter.initialize({})
        result = await adapter.execute(
            "get_contents",
            urls=["https://example.com"],
            summary=True
        )

        assert result.success is True


class TestExaAdapterFindSimilar:
    """Tests for find_similar operation."""

    @pytest.fixture
    def adapter(self, clean_env):
        return ExaAdapter()

    @pytest.mark.asyncio
    async def test_find_similar_basic(self, adapter):
        """Find similar content should work."""
        await adapter.initialize({})
        result = await adapter.execute(
            "find_similar",
            url="https://example.com/article"
        )

        assert result.success is True
        assert "results" in result.data

    @pytest.mark.asyncio
    async def test_find_similar_exclude_source(self, adapter):
        """Find similar with exclude_source_domain should work."""
        await adapter.initialize({})
        result = await adapter.execute(
            "find_similar",
            url="https://example.com/article",
            exclude_source_domain=True
        )

        assert result.success is True


class TestExaAdapterAnswer:
    """Tests for answer operation."""

    @pytest.fixture
    def adapter(self, clean_env):
        return ExaAdapter()

    @pytest.mark.asyncio
    async def test_answer_basic(self, adapter):
        """Answer operation should work."""
        await adapter.initialize({})
        result = await adapter.execute(
            "answer",
            query="What is Python?"
        )

        assert result.success is True
        assert "answer" in result.data


class TestExaAdapterResearch:
    """Tests for research operation."""

    @pytest.fixture
    def adapter(self, clean_env):
        return ExaAdapter()

    @pytest.mark.asyncio
    async def test_research_basic(self, adapter):
        """Research operation should work in mock mode."""
        await adapter.initialize({})
        result = await adapter.execute(
            "research",
            instructions="Compare vector databases"
        )

        assert result.success is True
        assert "result" in result.data or "mock" in result.data


class TestExaAdapterCompanyResearch:
    """Tests for company_research operation."""

    @pytest.fixture
    def adapter(self, clean_env):
        return ExaAdapter()

    @pytest.mark.asyncio
    async def test_company_research_basic(self, adapter):
        """Company research should work."""
        await adapter.initialize({})
        result = await adapter.execute(
            "company_research",
            domain="anthropic.com"
        )

        assert result.success is True
        assert "company" in result.data


class TestExaAdapterLinkedInSearch:
    """Tests for linkedin_search operation."""

    @pytest.fixture
    def adapter(self, clean_env):
        return ExaAdapter()

    @pytest.mark.asyncio
    async def test_linkedin_search_people(self, adapter):
        """LinkedIn people search should work."""
        await adapter.initialize({})
        result = await adapter.execute(
            "linkedin_search",
            query="AI researcher",
            search_type="people"
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_linkedin_search_companies(self, adapter):
        """LinkedIn company search should work."""
        await adapter.initialize({})
        result = await adapter.execute(
            "linkedin_search",
            query="AI startup",
            search_type="company"
        )

        assert result.success is True


class TestExaAdapterCodeSearch:
    """Tests for code_search operation."""

    @pytest.fixture
    def adapter(self, clean_env):
        return ExaAdapter()

    @pytest.mark.asyncio
    async def test_code_search_basic(self, adapter):
        """Code search should work."""
        await adapter.initialize({})
        result = await adapter.execute(
            "code_search",
            query="async Python decorator"
        )

        assert result.success is True
        assert "results" in result.data

    @pytest.mark.asyncio
    async def test_code_search_with_language(self, adapter):
        """Code search with language filter should work."""
        await adapter.initialize({})
        result = await adapter.execute(
            "code_search",
            query="async decorator",
            language="Python"
        )

        assert result.success is True


class TestExaAdapterHealthAndShutdown:
    """Tests for health check and shutdown."""

    @pytest.fixture
    def adapter(self, clean_env):
        return ExaAdapter()

    @pytest.mark.asyncio
    async def test_health_check_degraded(self, adapter):
        """Health check in degraded mode should return degraded status."""
        await adapter.initialize({})
        result = await adapter.health_check()

        # In mock mode (no SDK or no API key), health_check returns degraded
        if EXA_AVAILABLE:
            assert result.success is True
            assert result.data.get("status") in ["healthy", "degraded"]
        else:
            # SDK not installed - health_check returns False
            assert result.success is False

    @pytest.mark.asyncio
    async def test_shutdown_returns_stats(self, adapter):
        """Shutdown should return stats."""
        await adapter.initialize({})
        await adapter.execute("search", query="test")
        result = await adapter.shutdown()

        assert result.success is True
        assert "stats" in result.data

    @pytest.mark.asyncio
    async def test_shutdown_cleanup(self, adapter):
        """Shutdown should cleanup client."""
        await adapter.initialize({})
        await adapter.shutdown()

        assert adapter._client is None
        assert adapter._status.value == AdapterStatus.UNINITIALIZED.value


class TestExaAdapterErrorHandling:
    """Tests for error handling."""

    @pytest.fixture
    def adapter(self, clean_env):
        return ExaAdapter()

    @pytest.mark.asyncio
    async def test_unknown_operation_returns_error(self, adapter):
        """Unknown operation should return error, not raise."""
        await adapter.initialize({})
        result = await adapter.execute("__invalid_operation__")

        assert result.success is False
        assert result.error is not None
        assert "Unknown operation" in result.error

    @pytest.mark.asyncio
    async def test_error_includes_latency(self, adapter):
        """Error results should include latency."""
        await adapter.initialize({})
        result = await adapter.execute("__invalid__")

        assert result.latency_ms >= 0


class TestExaSearchType:
    """Tests for ExaSearchType enum."""

    def test_search_type_values(self):
        """Verify search type enum values."""
        assert ExaSearchType.FAST.value == "fast"
        assert ExaSearchType.AUTO.value == "auto"
        assert ExaSearchType.NEURAL.value == "neural"
        assert ExaSearchType.KEYWORD.value == "keyword"
        assert ExaSearchType.DEEP.value == "deep"


class TestExaCategory:
    """Tests for ExaCategory enum."""

    def test_category_values(self):
        """Verify category enum values."""
        assert ExaCategory.COMPANY.value == "company"
        assert ExaCategory.RESEARCH_PAPER.value == "research paper"
        assert ExaCategory.NEWS.value == "news"
        assert ExaCategory.GITHUB.value == "github"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
