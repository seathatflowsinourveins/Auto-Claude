"""
Tests for Request Deduplication Module
======================================

Validates:
- Request fingerprinting
- Cache hit/miss behavior
- In-flight request tracking
- TTL expiration
- Statistics collection
- Thread-safety
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

# Import deduplication module
from core.deduplication import (
    RequestDeduplicator,
    DeduplicationConfig,
    DeduplicationStrategy,
    DeduplicationStats,
    CachedResponse,
    DeduplicationResult,
    get_deduplicator,
    get_deduplicator_sync,
    reset_deduplicator,
    deduplicated,
    DeduplicatedAdapter,
)


class TestRequestFingerprinting:
    """Test request fingerprint generation."""

    def test_same_params_same_fingerprint(self):
        """Identical parameters should produce identical fingerprints."""
        dedup = RequestDeduplicator()

        fp1 = dedup._generate_fingerprint(
            "search",
            adapter_name="test",
            query="hello world",
            max_results=10,
        )

        fp2 = dedup._generate_fingerprint(
            "search",
            adapter_name="test",
            query="hello world",
            max_results=10,
        )

        assert fp1 == fp2

    def test_different_params_different_fingerprint(self):
        """Different parameters should produce different fingerprints."""
        dedup = RequestDeduplicator()

        fp1 = dedup._generate_fingerprint(
            "search",
            adapter_name="test",
            query="hello",
        )

        fp2 = dedup._generate_fingerprint(
            "search",
            adapter_name="test",
            query="world",
        )

        assert fp1 != fp2

    def test_param_order_independence(self):
        """Parameter order should not affect fingerprint."""
        dedup = RequestDeduplicator()

        fp1 = dedup._generate_fingerprint(
            "search",
            query="test",
            max_results=10,
            include_content=True,
        )

        fp2 = dedup._generate_fingerprint(
            "search",
            include_content=True,
            max_results=10,
            query="test",
        )

        assert fp1 == fp2

    def test_case_normalization(self):
        """String values should be normalized for matching."""
        dedup = RequestDeduplicator()

        fp1 = dedup._generate_fingerprint(
            "search",
            query="HELLO WORLD",
        )

        fp2 = dedup._generate_fingerprint(
            "search",
            query="hello world",
        )

        assert fp1 == fp2

    def test_whitespace_normalization(self):
        """Whitespace should be normalized."""
        dedup = RequestDeduplicator()

        fp1 = dedup._generate_fingerprint(
            "search",
            query="  hello world  ",
        )

        fp2 = dedup._generate_fingerprint(
            "search",
            query="hello world",
        )

        assert fp1 == fp2


class TestCacheOperations:
    """Test cache hit/miss behavior."""

    @pytest.mark.asyncio
    async def test_cache_miss_on_first_request(self):
        """First request should result in cache miss."""
        dedup = RequestDeduplicator()

        result = await dedup.check("search", query="test")

        assert not result.is_cached
        assert not result.is_in_flight
        assert result.fingerprint != ""

    @pytest.mark.asyncio
    async def test_cache_hit_after_completion(self):
        """Subsequent requests should hit cache."""
        dedup = RequestDeduplicator()

        # First request
        result1 = await dedup.check("search", query="test")
        assert not result1.is_cached

        # Complete the request
        await dedup.register_in_flight(result1.fingerprint)
        await dedup.complete_request(
            result1.fingerprint,
            {"data": "test result"},
            ttl=60.0,
            operation="search",
        )

        # Second request should hit cache
        result2 = await dedup.check("search", query="test")

        assert result2.is_cached
        assert result2.value == {"data": "test result"}

    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Expired cache entries should be treated as miss."""
        config = DeduplicationConfig(default_ttl_seconds=0.1)  # 100ms TTL
        dedup = RequestDeduplicator(config)

        # First request
        result1 = await dedup.check("search", query="test")
        await dedup.register_in_flight(result1.fingerprint)
        await dedup.complete_request(
            result1.fingerprint,
            {"data": "test"},
            ttl=0.1,
        )

        # Wait for expiration
        await asyncio.sleep(0.15)

        # Should be a miss
        result2 = await dedup.check("search", query="test")

        assert not result2.is_cached

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """Oldest entries should be evicted when at capacity."""
        config = DeduplicationConfig(max_cache_size=2)
        dedup = RequestDeduplicator(config)

        # Fill cache
        for i in range(3):
            result = await dedup.check("search", query=f"test{i}")
            await dedup.register_in_flight(result.fingerprint)
            await dedup.complete_request(
                result.fingerprint,
                {"data": f"result{i}"},
            )

        # First entry should be evicted
        result0 = await dedup.check("search", query="test0")
        assert not result0.is_cached

        # Last entry should still be cached
        result2 = await dedup.check("search", query="test2")
        assert result2.is_cached


class TestInFlightTracking:
    """Test in-flight request tracking."""

    @pytest.mark.asyncio
    async def test_in_flight_request_tracking(self):
        """Concurrent identical requests should share the same future."""
        dedup = RequestDeduplicator()

        # First request registers as in-flight
        result1 = await dedup.check("search", query="test")
        future = await dedup.register_in_flight(result1.fingerprint)

        # Second request should detect in-flight
        result2 = await dedup.check("search", query="test")

        assert result2.is_in_flight
        assert result2.future is not None

    @pytest.mark.asyncio
    async def test_in_flight_completion(self):
        """Completing in-flight request should resolve futures."""
        dedup = RequestDeduplicator()

        # First request
        result1 = await dedup.check("search", query="test")
        future = await dedup.register_in_flight(result1.fingerprint)

        # Simulate second request waiting
        async def wait_for_result():
            result2 = await dedup.check("search", query="test")
            if result2.is_in_flight:
                return await result2.future
            return result2.value

        # Complete the first request
        await dedup.complete_request(
            result1.fingerprint,
            {"data": "shared result"},
        )

        # Both should get the same result
        shared_result = await wait_for_result()
        assert shared_result == {"data": "shared result"}

    @pytest.mark.asyncio
    async def test_in_flight_failure_propagation(self):
        """Failed in-flight requests should propagate errors."""
        dedup = RequestDeduplicator()

        result1 = await dedup.check("search", query="test")
        await dedup.register_in_flight(result1.fingerprint)

        # Fail the request
        await dedup.fail_request(
            result1.fingerprint,
            ValueError("API error"),
        )

        # In-flight entry should be removed
        result2 = await dedup.check("search", query="test")
        assert not result2.is_in_flight


class TestStatistics:
    """Test statistics collection."""

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        """Statistics should be accurately tracked."""
        dedup = RequestDeduplicator()

        # Cache miss
        result1 = await dedup.check("search", query="test1")
        await dedup.register_in_flight(result1.fingerprint)
        await dedup.complete_request(result1.fingerprint, {"data": "1"})

        # Cache hit
        await dedup.check("search", query="test1")

        # Another cache miss
        result2 = await dedup.check("search", query="test2")
        await dedup.register_in_flight(result2.fingerprint)
        await dedup.complete_request(result2.fingerprint, {"data": "2"})

        stats = dedup.get_stats()

        assert stats["total_requests"] == 3
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 2
        assert stats["hit_rate"] > 0

    @pytest.mark.asyncio
    async def test_hit_rate_calculation(self):
        """Hit rate should be calculated correctly."""
        dedup = RequestDeduplicator()

        # 2 misses
        for i in range(2):
            result = await dedup.check("search", query=f"test{i}")
            await dedup.register_in_flight(result.fingerprint)
            await dedup.complete_request(result.fingerprint, {"data": str(i)})

        # 2 hits
        await dedup.check("search", query="test0")
        await dedup.check("search", query="test1")

        stats = dedup.get_stats()

        # 2 hits out of 4 total requests = 50%
        assert stats["hit_rate"] == 0.5


class TestExecuteWrapper:
    """Test the execute convenience method."""

    @pytest.mark.asyncio
    async def test_execute_with_deduplication(self):
        """Execute should handle the full deduplication flow."""
        dedup = RequestDeduplicator()
        call_count = 0

        async def mock_api_call(query: str) -> dict:
            nonlocal call_count
            call_count += 1
            return {"result": query}

        # First call executes
        result1 = await dedup.execute(
            mock_api_call,
            operation="search",
            query="test",
        )

        # Second call should use cache
        result2 = await dedup.execute(
            mock_api_call,
            operation="search",
            query="test",
        )

        assert result1 == {"result": "test"}
        assert result2 == {"result": "test"}
        assert call_count == 1  # Only one actual API call


class TestDecorator:
    """Test the @deduplicated decorator."""

    @pytest.mark.asyncio
    async def test_deduplicated_decorator(self):
        """Decorator should add deduplication to async functions."""
        call_count = 0

        @deduplicated("test_adapter", operation="search")
        async def search(query: str) -> dict:
            nonlocal call_count
            call_count += 1
            return {"result": query}

        # First call
        result1 = await search(query="test")

        # Second call should use cache
        result2 = await search(query="test")

        assert result1 == {"result": "test"}
        assert result2 == {"result": "test"}
        assert call_count == 1


class TestMixin:
    """Test the DeduplicatedAdapter mixin."""

    @pytest.mark.asyncio
    async def test_mixin_integration(self):
        """Mixin should provide deduplication capabilities."""

        class TestAdapter(DeduplicatedAdapter):
            def __init__(self):
                self.init_deduplication(
                    adapter_name="test_adapter",
                    default_ttl=60.0,
                )
                self.call_count = 0

            async def search(self, query: str) -> dict:
                return await self.deduplicated_call(
                    self._do_search,
                    operation="search",
                    query=query,
                )

            async def _do_search(self, query: str) -> dict:
                self.call_count += 1
                return {"result": query}

        adapter = TestAdapter()

        # First call
        result1 = await adapter.search("test")

        # Second call should use cache
        result2 = await adapter.search("test")

        assert result1 == {"result": "test"}
        assert result2 == {"result": "test"}
        assert adapter.call_count == 1


class TestThreadSafety:
    """Test thread-safety under concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Concurrent identical requests should share results."""
        dedup = RequestDeduplicator()
        call_count = 0

        async def mock_api_call(query: str) -> dict:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # Simulate API latency
            return {"result": query, "call": call_count}

        # Fire 10 concurrent requests
        tasks = [
            dedup.execute(mock_api_call, operation="search", query="test")
            for _ in range(10)
        ]

        results = await asyncio.gather(*tasks)

        # All should get the same result
        assert all(r["result"] == "test" for r in results)
        # Only one actual API call should have been made
        assert call_count == 1


class TestCleanup:
    """Test cleanup and maintenance operations."""

    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """Clear should remove all entries."""
        dedup = RequestDeduplicator()

        # Add some entries
        for i in range(5):
            result = await dedup.check("search", query=f"test{i}")
            await dedup.register_in_flight(result.fingerprint)
            await dedup.complete_request(result.fingerprint, {"data": str(i)})

        # Clear
        count = await dedup.clear()

        assert count == 5
        assert dedup.get_stats()["cache_size"] == 0

    @pytest.mark.asyncio
    async def test_invalidate_by_adapter(self):
        """Invalidate should remove entries for specific adapter."""
        dedup = RequestDeduplicator()

        # Add entries for different adapters
        for adapter in ["adapter1", "adapter2"]:
            result = await dedup.check("search", adapter_name=adapter, query="test")
            await dedup.register_in_flight(result.fingerprint)
            await dedup.complete_request(
                result.fingerprint,
                {"data": adapter},
                adapter_name=adapter,
            )

        # Invalidate only adapter1
        count = await dedup.invalidate(adapter_name="adapter1")

        assert count == 1

        # adapter1 should be gone
        result1 = await dedup.check("search", adapter_name="adapter1", query="test")
        assert not result1.is_cached

        # adapter2 should still be cached
        result2 = await dedup.check("search", adapter_name="adapter2", query="test")
        assert result2.is_cached


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
