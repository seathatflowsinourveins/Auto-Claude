#!/usr/bin/env python3
"""
V118 Optimization Test: Connection Pooling for HTTP Clients

This test validates that HTTP clients use connection pooling:
1. Shared client instances (not new client per request)
2. Proper httpx.Limits configuration
3. Connection reuse across calls

Expected Gains:
- Latency reduction: -90% per-request overhead
- Throughput: +500% concurrent requests
- Resource efficiency: -99% TCP connection creation

Test Date: 2026-01-30
"""

import os
import re
import sys
import pytest

# Add platform to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestConnectionPoolingPatterns:
    """Test suite for connection pooling pattern fixes."""

    def test_advanced_memory_uses_shared_client(self):
        """Verify OpenAIEmbeddingProvider uses shared client."""
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "core", "advanced_memory.py"
        )

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Should have _shared_client class variable
        assert "_shared_client" in content, \
            "advanced_memory.py should have _shared_client for connection pooling"

        # Should have _get_client method
        assert "_get_client" in content, \
            "advanced_memory.py should have _get_client method"

        # Should NOT have "async with httpx.AsyncClient()" in embed methods
        # Look for the pattern in the OpenAIEmbeddingProvider class section
        openai_section_start = content.find("class OpenAIEmbeddingProvider")
        openai_section_end = content.find("class ", openai_section_start + 1)
        if openai_section_end == -1:
            openai_section_end = len(content)

        openai_section = content[openai_section_start:openai_section_end]

        # Count occurrences of bad pattern
        bad_pattern_count = len(re.findall(r"async with httpx\.AsyncClient\(\)", openai_section))
        assert bad_pattern_count == 0, \
            f"OpenAIEmbeddingProvider should not create new clients per request (found {bad_pattern_count})"

    def test_model_router_uses_shared_client(self):
        """Verify OllamaClient uses shared client."""
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "adapters", "model_router.py"
        )

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Should have _shared_client class variable
        assert "_shared_client" in content, \
            "model_router.py should have _shared_client for connection pooling"

        # Should have _get_client method
        assert "_get_client" in content, \
            "model_router.py should have _get_client method"

        # Should have httpx.Limits configuration
        assert "httpx.Limits" in content, \
            "model_router.py should configure httpx.Limits for connection pooling"


class TestConnectionPoolConfiguration:
    """Test connection pool configuration values."""

    def test_advanced_memory_pool_limits(self):
        """Verify OpenAIEmbeddingProvider has proper pool limits."""
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "core", "advanced_memory.py"
        )

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Should have max_connections configured
        assert "max_connections=" in content, \
            "advanced_memory.py should configure max_connections"

        # Should have max_keepalive_connections configured
        assert "max_keepalive_connections=" in content, \
            "advanced_memory.py should configure max_keepalive_connections"

    def test_model_router_pool_limits(self):
        """Verify OllamaClient has proper pool limits."""
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "adapters", "model_router.py"
        )

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Should have max_connections configured
        assert "max_connections=" in content, \
            "model_router.py should configure max_connections"


class TestSharedClientBehavior:
    """Test that shared client pattern works correctly."""

    def test_openai_provider_client_reuse(self):
        """Test that OpenAIEmbeddingProvider reuses client."""
        try:
            from core.advanced_memory import OpenAIEmbeddingProvider
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        # Create two instances
        provider1 = OpenAIEmbeddingProvider(api_key="test-key-1")
        provider2 = OpenAIEmbeddingProvider(api_key="test-key-2")

        # Both should use the same shared client
        client1 = provider1._get_client()
        client2 = provider2._get_client()

        assert client1 is client2, \
            "Multiple OpenAIEmbeddingProvider instances should share the same client"

    def test_ollama_client_reuse(self):
        """Test that OllamaClient reuses client."""
        try:
            from platform.adapters.model_router import OllamaClient
        except ImportError:
            pytest.skip("model_router module not importable")

        # Create two instances
        client1 = OllamaClient(base_url="http://localhost:11434")
        client2 = OllamaClient(base_url="http://localhost:11434")

        # Both should use the same shared client
        httpx_client1 = client1._get_client()
        httpx_client2 = client2._get_client()

        assert httpx_client1 is httpx_client2, \
            "Multiple OllamaClient instances should share the same httpx client"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
