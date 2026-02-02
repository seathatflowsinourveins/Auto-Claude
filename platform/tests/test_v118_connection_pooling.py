#!/usr/bin/env python3
"""
V118 Optimization Test: Connection Pooling for HTTP Clients

Tests that HTTP clients use connection pooling by actually importing
and testing the classes - not by grepping file contents.

Test Date: 2026-01-30, Updated: 2026-02-02 (V14 Iter 50)
"""

import os
import sys
import pytest

# Add platform to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestConnectionPoolingReal:
    """Test connection pooling by importing and testing real classes."""

    def test_openai_provider_has_shared_client(self):
        """OpenAIEmbeddingProvider class should have _shared_client attribute."""
        try:
            from core.advanced_memory import OpenAIEmbeddingProvider
        except ImportError:
            pytest.skip("advanced_memory not importable")

        # Class-level shared client should exist
        assert hasattr(OpenAIEmbeddingProvider, "_shared_client"), \
            "OpenAIEmbeddingProvider should have _shared_client class variable"

    def test_openai_provider_has_get_client_method(self):
        """OpenAIEmbeddingProvider should have _get_client method."""
        try:
            from core.advanced_memory import OpenAIEmbeddingProvider
        except ImportError:
            pytest.skip("advanced_memory not importable")

        provider = OpenAIEmbeddingProvider(api_key="test-key")
        assert hasattr(provider, "_get_client"), \
            "OpenAIEmbeddingProvider should have _get_client method"
        assert callable(provider._get_client)

    def test_openai_provider_client_reuse(self):
        """Multiple OpenAIEmbeddingProvider instances should share the same client."""
        try:
            from core.advanced_memory import OpenAIEmbeddingProvider
        except ImportError:
            pytest.skip("advanced_memory not importable")

        provider1 = OpenAIEmbeddingProvider(api_key="test-key-1")
        provider2 = OpenAIEmbeddingProvider(api_key="test-key-2")

        client1 = provider1._get_client()
        client2 = provider2._get_client()

        assert client1 is client2, \
            "Multiple OpenAIEmbeddingProvider instances should share the same client"

    def test_openai_provider_client_is_httpx(self):
        """Shared client should be an httpx.AsyncClient."""
        try:
            from core.advanced_memory import OpenAIEmbeddingProvider
            import httpx
        except ImportError:
            pytest.skip("advanced_memory or httpx not importable")

        provider = OpenAIEmbeddingProvider(api_key="test-key")
        client = provider._get_client()
        assert isinstance(client, httpx.AsyncClient), \
            f"Expected httpx.AsyncClient, got {type(client).__name__}"

    def test_ollama_client_has_shared_client(self):
        """OllamaClient class should have _shared_client attribute."""
        try:
            from adapters.model_router import OllamaClient
        except ImportError:
            pytest.skip("model_router not importable")

        assert hasattr(OllamaClient, "_shared_client"), \
            "OllamaClient should have _shared_client class variable"

    def test_ollama_client_reuse(self):
        """Multiple OllamaClient instances should share the same httpx client."""
        try:
            from adapters.model_router import OllamaClient
        except ImportError:
            pytest.skip("model_router not importable")

        client1 = OllamaClient(base_url="http://localhost:11434")
        client2 = OllamaClient(base_url="http://localhost:11434")

        httpx_client1 = client1._get_client()
        httpx_client2 = client2._get_client()

        assert httpx_client1 is httpx_client2, \
            "Multiple OllamaClient instances should share the same httpx client"

    def test_ollama_client_is_httpx_async(self):
        """OllamaClient's shared client should be an httpx.AsyncClient."""
        try:
            from adapters.model_router import OllamaClient
            import httpx
        except ImportError:
            pytest.skip("model_router or httpx not importable")

        client = OllamaClient(base_url="http://localhost:11434")
        httpx_client = client._get_client()

        assert isinstance(httpx_client, httpx.AsyncClient), \
            f"Expected httpx.AsyncClient, got {type(httpx_client).__name__}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
