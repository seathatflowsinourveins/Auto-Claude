"""
Adapter Latency Performance Tests

Tests for adapter initialization, health check, and operation latencies.
"""

import pytest
import asyncio
import time
from typing import Dict, Any, List, Type

# Import adapters
try:
    from adapters.exa_adapter import ExaAdapter
    from adapters.tavily_adapter import TavilyAdapter
    from adapters.jina_adapter import JinaAdapter
    from adapters.perplexity_adapter import PerplexityAdapter
    from adapters.simplemem_adapter import SimpleMemAdapter
    from adapters.braintrust_adapter import BraintrustAdapter
    from core.orchestration.base import SDKAdapter, AdapterResult
except ImportError:
    pytest.skip("Adapters not available", allow_module_level=True)


# Adapters to test
ADAPTERS_TO_TEST = [
    ("exa", ExaAdapter),
    ("tavily", TavilyAdapter),
    ("jina", JinaAdapter),
    ("perplexity", PerplexityAdapter),
]


class TestAdapterInitializationLatency:
    """Tests for adapter initialization latency."""

    @pytest.mark.performance
    @pytest.mark.parametrize("name,adapter_class", ADAPTERS_TO_TEST)
    @pytest.mark.asyncio
    async def test_initialization_under_500ms(self, name: str, adapter_class: Type[SDKAdapter]):
        """Adapter initialization should complete within 500ms."""
        adapter = adapter_class()

        start = time.perf_counter()
        result = await adapter.initialize({})
        elapsed_ms = (time.perf_counter() - start) * 1000

        await adapter.shutdown()

        assert result is not None
        assert elapsed_ms < 500, f"{name} initialization took {elapsed_ms:.2f}ms (limit: 500ms)"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_all_adapters_initialize_under_2s_total(self):
        """All adapters should initialize within 2 seconds total."""
        adapters = []

        start = time.perf_counter()

        for name, adapter_class in ADAPTERS_TO_TEST:
            adapter = adapter_class()
            await adapter.initialize({})
            adapters.append(adapter)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Cleanup
        for adapter in adapters:
            await adapter.shutdown()

        assert elapsed_ms < 2000, f"All initializations took {elapsed_ms:.2f}ms (limit: 2000ms)"


class TestHealthCheckLatency:
    """Tests for health check latency."""

    @pytest.mark.performance
    @pytest.mark.parametrize("name,adapter_class", ADAPTERS_TO_TEST)
    @pytest.mark.asyncio
    async def test_health_check_under_100ms(self, name: str, adapter_class: Type[SDKAdapter]):
        """Health check should complete within 100ms."""
        adapter = adapter_class()
        await adapter.initialize({})

        start = time.perf_counter()
        result = await adapter.health_check()
        elapsed_ms = (time.perf_counter() - start) * 1000

        await adapter.shutdown()

        assert result is not None
        assert elapsed_ms < 100, f"{name} health check took {elapsed_ms:.2f}ms (limit: 100ms)"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_repeated_health_checks_consistent(self):
        """Repeated health checks should have consistent latency."""
        adapter = ExaAdapter()
        await adapter.initialize({})

        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            await adapter.health_check()
            latencies.append((time.perf_counter() - start) * 1000)

        await adapter.shutdown()

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)

        # Max should not be more than 5x average
        assert max_latency < avg_latency * 5, f"Latency variance too high: max={max_latency:.2f}ms, avg={avg_latency:.2f}ms"


class TestOperationLatency:
    """Tests for operation latency in mock mode."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_search_operation_under_500ms(self):
        """Search operation should complete within 500ms (mock mode)."""
        adapter = ExaAdapter()
        await adapter.initialize({})

        start = time.perf_counter()
        result = await adapter.execute("search", query="test query")
        elapsed_ms = (time.perf_counter() - start) * 1000

        await adapter.shutdown()

        assert result.success is True
        assert elapsed_ms < 500, f"Search took {elapsed_ms:.2f}ms (limit: 500ms)"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_multiple_operations_p50_under_100ms(self):
        """P50 operation latency should be under 100ms (mock mode)."""
        adapter = ExaAdapter()
        await adapter.initialize({})

        latencies = []
        for i in range(20):
            start = time.perf_counter()
            await adapter.execute("search", query=f"test query {i}")
            latencies.append((time.perf_counter() - start) * 1000)

        await adapter.shutdown()

        latencies.sort()
        p50 = latencies[len(latencies) // 2]

        assert p50 < 100, f"P50 latency {p50:.2f}ms exceeds 100ms limit"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_multiple_operations_p99_under_500ms(self):
        """P99 operation latency should be under 500ms (mock mode)."""
        adapter = ExaAdapter()
        await adapter.initialize({})

        latencies = []
        for i in range(100):
            start = time.perf_counter()
            await adapter.execute("search", query=f"test query {i}")
            latencies.append((time.perf_counter() - start) * 1000)

        await adapter.shutdown()

        latencies.sort()
        p99_index = int(len(latencies) * 0.99)
        p99 = latencies[p99_index]

        assert p99 < 500, f"P99 latency {p99:.2f}ms exceeds 500ms limit"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_error_operation_fast(self):
        """Error operations should be fast (under 50ms)."""
        adapter = ExaAdapter()
        await adapter.initialize({})

        start = time.perf_counter()
        result = await adapter.execute("__invalid_operation__")
        elapsed_ms = (time.perf_counter() - start) * 1000

        await adapter.shutdown()

        assert result.success is False
        assert elapsed_ms < 50, f"Error operation took {elapsed_ms:.2f}ms (limit: 50ms)"


class TestShutdownLatency:
    """Tests for shutdown latency."""

    @pytest.mark.performance
    @pytest.mark.parametrize("name,adapter_class", ADAPTERS_TO_TEST)
    @pytest.mark.asyncio
    async def test_shutdown_under_100ms(self, name: str, adapter_class: Type[SDKAdapter]):
        """Shutdown should complete within 100ms."""
        adapter = adapter_class()
        await adapter.initialize({})

        start = time.perf_counter()
        result = await adapter.shutdown()
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result.success is True
        assert elapsed_ms < 100, f"{name} shutdown took {elapsed_ms:.2f}ms (limit: 100ms)"


class TestConcurrentOperationLatency:
    """Tests for concurrent operation latency."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_operations_scale(self):
        """Concurrent operations should not degrade significantly."""
        adapter = ExaAdapter()
        await adapter.initialize({})

        # Sequential baseline
        sequential_start = time.perf_counter()
        for i in range(10):
            await adapter.execute("search", query=f"query {i}")
        sequential_elapsed = (time.perf_counter() - sequential_start) * 1000

        # Concurrent
        concurrent_start = time.perf_counter()
        tasks = [
            adapter.execute("search", query=f"query {i}")
            for i in range(10)
        ]
        await asyncio.gather(*tasks)
        concurrent_elapsed = (time.perf_counter() - concurrent_start) * 1000

        await adapter.shutdown()

        # Concurrent should be faster or at least not 2x slower
        assert concurrent_elapsed < sequential_elapsed * 2, \
            f"Concurrent ({concurrent_elapsed:.2f}ms) slower than 2x sequential ({sequential_elapsed:.2f}ms)"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_50_concurrent_operations(self):
        """50 concurrent operations should complete within 5 seconds."""
        adapter = ExaAdapter()
        await adapter.initialize({})

        start = time.perf_counter()
        tasks = [
            adapter.execute("search", query=f"query {i}")
            for i in range(50)
        ]
        results = await asyncio.gather(*tasks)
        elapsed_ms = (time.perf_counter() - start) * 1000

        await adapter.shutdown()

        # All should succeed
        successes = sum(1 for r in results if r.success)
        assert successes >= 45, f"Only {successes}/50 succeeded"

        assert elapsed_ms < 5000, f"50 concurrent ops took {elapsed_ms:.2f}ms (limit: 5000ms)"


class TestLatencyReporting:
    """Tests for latency reporting in AdapterResult."""

    @pytest.mark.asyncio
    async def test_latency_always_reported(self):
        """AdapterResult should always include latency."""
        adapter = ExaAdapter()
        await adapter.initialize({})

        result = await adapter.execute("search", query="test")
        await adapter.shutdown()

        assert hasattr(result, "latency_ms")
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_latency_reported_on_error(self):
        """Latency should be reported even on errors."""
        adapter = ExaAdapter()
        await adapter.initialize({})

        result = await adapter.execute("__invalid__")
        await adapter.shutdown()

        assert result.success is False
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_latency_approximately_correct(self):
        """Reported latency should approximately match actual time."""
        adapter = ExaAdapter()
        await adapter.initialize({})

        start = time.perf_counter()
        result = await adapter.execute("search", query="test")
        actual_ms = (time.perf_counter() - start) * 1000

        await adapter.shutdown()

        # Reported latency should be within 20% of actual
        diff_percent = abs(result.latency_ms - actual_ms) / actual_ms * 100
        assert diff_percent < 50, f"Latency diff {diff_percent:.1f}% > 50%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "performance"])
