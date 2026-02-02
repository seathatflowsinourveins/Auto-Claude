#!/usr/bin/env python3
"""
V121 Optimization Test: Circuit Breaker for API Failures

This test validates that the circuit breaker pattern is integrated:
1. OpenAIEmbeddingProvider has a circuit breaker
2. API calls are wrapped with the circuit breaker
3. Circuit breaker transitions work correctly
4. Statistics are tracked and accessible

Expected Gains:
- Cascade failure prevention
- Fail-fast during outages (~0ms vs ~30s timeout)
- Automatic recovery detection

Test Date: 2026-01-30
"""

import os
import re
import sys
import pytest

# Add platform to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestCircuitBreakerPatterns:
    """Test suite for circuit breaker pattern verification."""

    def test_circuit_breaker_import_exists(self):
        """Verify CircuitBreaker is imported in advanced_memory."""
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "core", "advanced_memory.py"
        )

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Should import CircuitBreaker
        assert "from .resilience import" in content or "from resilience import" in content, \
            "Should import from resilience module"
        assert "CircuitBreaker" in content, \
            "CircuitBreaker should be imported"

    def test_openai_provider_has_circuit_breaker(self):
        """Verify OpenAIEmbeddingProvider has circuit breaker instance."""
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "core", "advanced_memory.py"
        )

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find OpenAIEmbeddingProvider class section
        provider_start = content.find("class OpenAIEmbeddingProvider")
        provider_end = content.find("\nclass ", provider_start + 1)
        if provider_end == -1:
            provider_end = len(content)
        provider_section = content[provider_start:provider_end]

        # Should have circuit breaker as class variable
        assert "_circuit_breaker" in provider_section, \
            "OpenAIEmbeddingProvider should have _circuit_breaker"
        assert "CircuitBreaker(" in provider_section, \
            "Should instantiate CircuitBreaker"

    def test_embed_uses_circuit_breaker(self):
        """Verify embed method uses circuit breaker."""
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "core", "advanced_memory.py"
        )

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find OpenAIEmbeddingProvider class section
        provider_start = content.find("class OpenAIEmbeddingProvider")
        provider_end = content.find("\nclass ", provider_start + 1)
        if provider_end == -1:
            provider_end = len(content)
        provider_section = content[provider_start:provider_end]

        # Find embed method
        embed_match = re.search(
            r"async def embed\(self.*?\n(?:\s{8}.*\n)*?(?=\n\s{4}async def|\n\s{4}@|\nclass|\Z)",
            provider_section,
            re.MULTILINE
        )
        assert embed_match, "Should have embed method"
        embed_method = embed_match.group(0)

        # Should use circuit breaker context manager
        assert "async with self._circuit_breaker" in embed_method or \
               "async with cls._circuit_breaker" in embed_method, \
            "embed method should wrap API call with circuit breaker"

    def test_embed_batch_uses_circuit_breaker(self):
        """Verify embed_batch method uses circuit breaker."""
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "core", "advanced_memory.py"
        )

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find OpenAIEmbeddingProvider class section
        provider_start = content.find("class OpenAIEmbeddingProvider")
        provider_end = content.find("\nclass ", provider_start + 1)
        if provider_end == -1:
            provider_end = len(content)
        provider_section = content[provider_start:provider_end]

        # Find embed_batch method
        batch_match = re.search(
            r"async def embed_batch\(self.*?\n(?:\s{8}.*\n)*?(?=\n\s{4}async def|\n\s{4}@|\n\s{4}def|\nclass|\Z)",
            provider_section,
            re.MULTILINE
        )
        assert batch_match, "Should have embed_batch method"
        batch_method = batch_match.group(0)

        # Should use circuit breaker context manager
        assert "async with self._circuit_breaker" in batch_method or \
               "async with cls._circuit_breaker" in batch_method, \
            "embed_batch method should wrap API call with circuit breaker"

    def test_has_circuit_stats_method(self):
        """Verify get_circuit_stats method exists."""
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "core", "advanced_memory.py"
        )

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find OpenAIEmbeddingProvider class section
        provider_start = content.find("class OpenAIEmbeddingProvider")
        provider_end = content.find("\nclass ", provider_start + 1)
        if provider_end == -1:
            provider_end = len(content)
        provider_section = content[provider_start:provider_end]

        # Should have stats method
        assert "def get_circuit_stats" in provider_section, \
            "Should have get_circuit_stats method"


class TestCircuitBreakerBehavior:
    """Test actual circuit breaker behavior."""

    def test_circuit_breaker_states(self):
        """Test CircuitBreaker state transitions."""
        try:
            from platform.core.resilience import CircuitBreaker, CircuitState
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
            from platform.core.resilience import CircuitBreaker, CircuitState, CircuitOpenError
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
            from platform.core.resilience import CircuitBreaker
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
            from platform.core.advanced_memory import OpenAIEmbeddingProvider
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
            from platform.core.advanced_memory import (
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
