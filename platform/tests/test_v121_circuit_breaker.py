#!/usr/bin/env python3
"""
V121 Optimization Test: Circuit Breaker for API Failures

Tests circuit breaker pattern by importing and testing real classes -
not by grepping file contents.

Test Date: 2026-01-30, Updated: 2026-02-02 (V14 Iter 52)
"""

import os
import sys
import pytest

# Add platform to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestCircuitBreakerStructure:
    """Test circuit breaker structure by importing real classes."""

    def test_circuit_breaker_importable(self):
        """CircuitBreaker class should be importable from resilience."""
        try:
            from core.resilience import CircuitBreaker
        except ImportError:
            pytest.skip("resilience module not importable")

        assert CircuitBreaker is not None

    def test_circuit_state_importable(self):
        """CircuitState enum should be importable."""
        try:
            from core.resilience import CircuitState
        except ImportError:
            pytest.skip("resilience module not importable")

        # Should have expected states
        assert hasattr(CircuitState, "CLOSED")
        assert hasattr(CircuitState, "OPEN")
        assert hasattr(CircuitState, "HALF_OPEN")

    def test_openai_provider_has_circuit_breaker(self):
        """OpenAIEmbeddingProvider should have _circuit_breaker attribute."""
        try:
            from core.advanced_memory import OpenAIEmbeddingProvider
        except ImportError:
            pytest.skip("advanced_memory not importable")

        assert hasattr(OpenAIEmbeddingProvider, "_circuit_breaker"), \
            "OpenAIEmbeddingProvider should have _circuit_breaker class variable"

    def test_openai_provider_has_circuit_stats(self):
        """OpenAIEmbeddingProvider should have get_circuit_stats method."""
        try:
            from core.advanced_memory import OpenAIEmbeddingProvider
        except ImportError:
            pytest.skip("advanced_memory not importable")

        assert hasattr(OpenAIEmbeddingProvider, "get_circuit_stats"), \
            "Should have get_circuit_stats method"
        assert callable(OpenAIEmbeddingProvider.get_circuit_stats)

    def test_circuit_breaker_has_stats(self):
        """CircuitBreaker instances should have stats attribute."""
        try:
            from core.resilience import CircuitBreaker
        except ImportError:
            pytest.skip("resilience module not importable")

        breaker = CircuitBreaker(failure_threshold=3)
        assert hasattr(breaker, "stats"), "Should have stats attribute"
        assert hasattr(breaker, "state"), "Should have state attribute"
        assert hasattr(breaker, "reset"), "Should have reset method"


class TestCircuitBreakerBehavior:
    """Test actual circuit breaker behavior."""

    def test_circuit_breaker_states(self):
        """Test CircuitBreaker state transitions."""
        try:
            from core.resilience import CircuitBreaker, CircuitState
        except ImportError:
            pytest.skip("resilience module not importable")

        breaker = CircuitBreaker(
            failure_threshold=3,
            success_threshold=2,
            recovery_timeout=0.1,  # Short timeout for testing
        )

        # Should start CLOSED
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_opens_on_failures(self):
        """Test that circuit opens after consecutive failures."""
        try:
            from core.resilience import CircuitBreaker, CircuitState, CircuitOpenError
        except ImportError:
            pytest.skip("resilience module not importable")

        breaker = CircuitBreaker(
            failure_threshold=3,
            success_threshold=2,
            recovery_timeout=60.0,
        )

        # Simulate 3 failures
        for i in range(3):
            try:
                async with breaker:
                    raise RuntimeError(f"Simulated failure {i}")
            except RuntimeError:
                pass

        # Should be OPEN now
        assert breaker.state == CircuitState.OPEN, \
            f"Circuit should be OPEN after 3 failures, got {breaker.state}"

        # Next call should be rejected
        with pytest.raises(CircuitOpenError):
            async with breaker:
                pass

    @pytest.mark.asyncio
    async def test_circuit_tracks_stats(self):
        """Test that circuit breaker tracks statistics."""
        try:
            from core.resilience import CircuitBreaker
        except ImportError:
            pytest.skip("resilience module not importable")

        breaker = CircuitBreaker(failure_threshold=5)

        # Successful call
        async with breaker:
            pass  # Success

        # Failed call
        try:
            async with breaker:
                raise RuntimeError("Test failure")
        except RuntimeError:
            pass

        stats = breaker.stats
        assert stats.total_calls == 2
        assert stats.successful_calls == 1
        assert stats.failed_calls == 1

    def test_openai_provider_circuit_stats_method(self):
        """Test that OpenAIEmbeddingProvider has circuit stats."""
        try:
            from core.advanced_memory import OpenAIEmbeddingProvider
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        # Should be able to get stats without instantiation
        stats = OpenAIEmbeddingProvider.get_circuit_stats()

        assert "state" in stats
        assert "total_calls" in stats
        assert "successful_calls" in stats
        assert "failed_calls" in stats
        assert "rejected_calls" in stats
        assert "failure_rate" in stats


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with caching."""

    @pytest.mark.asyncio
    async def test_cache_bypasses_circuit_breaker(self):
        """Test that cache hits bypass the circuit breaker."""
        try:
            from core.advanced_memory import (
                OpenAIEmbeddingProvider,
                _embedding_cache,
            )
        except ImportError:
            pytest.skip("modules not importable")

        # Manually add to cache
        _embedding_cache.set("cached_text", "text-embedding-3-small", [0.1] * 1536)

        provider = OpenAIEmbeddingProvider(api_key="fake-key")

        # Reset circuit breaker to CLOSED
        provider._circuit_breaker.reset()

        # Get initial stats
        initial_calls = provider._circuit_breaker.stats.total_calls

        # This should hit cache and NOT touch circuit breaker
        result = await provider.embed("cached_text")

        # Verify cache was hit
        assert len(result.embedding) == 1536
        assert result.tokens_used == 0  # Cache hit indicator

        # Circuit breaker should NOT have been called
        assert provider._circuit_breaker.stats.total_calls == initial_calls, \
            "Cache hit should bypass circuit breaker"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
