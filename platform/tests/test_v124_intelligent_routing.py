#!/usr/bin/env python3
"""
V124 Optimization Test: Intelligent Model Routing

Tests content detection and model routing by importing and testing
real classes - not by grepping file contents.

Test Date: 2026-01-30, Updated: 2026-02-02 (V14 Iter 52)
"""

import os
import sys
import pytest

# Add platform to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestContentDetectionStructure:
    """Test content type detection structure by importing real classes."""

    def test_content_type_enum_importable(self):
        """ContentType enum should be importable with all values."""
        try:
            from core.advanced_memory import ContentType
        except ImportError:
            pytest.skip("advanced_memory not importable")

        assert hasattr(ContentType, "CODE"), "Should have CODE"
        assert hasattr(ContentType, "TEXT"), "Should have TEXT"
        assert hasattr(ContentType, "MULTILINGUAL"), "Should have MULTILINGUAL"
        assert hasattr(ContentType, "MIXED"), "Should have MIXED"
        assert hasattr(ContentType, "UNKNOWN"), "Should have UNKNOWN"

    def test_detect_content_type_importable(self):
        """detect_content_type function should be importable and callable."""
        try:
            from core.advanced_memory import detect_content_type
        except ImportError:
            pytest.skip("advanced_memory not importable")

        assert callable(detect_content_type)

    def test_content_type_values(self):
        """ContentType values should be the expected strings."""
        try:
            from core.advanced_memory import ContentType
        except ImportError:
            pytest.skip("advanced_memory not importable")

        assert ContentType.CODE.value == "code"
        assert ContentType.TEXT.value == "text"
        assert ContentType.MULTILINGUAL.value == "multilingual"
        assert ContentType.MIXED.value == "mixed"
        assert ContentType.UNKNOWN.value == "unknown"

    def test_detect_returns_content_type(self):
        """detect_content_type should return a ContentType enum value."""
        try:
            from core.advanced_memory import detect_content_type, ContentType
        except ImportError:
            pytest.skip("advanced_memory not importable")

        result = detect_content_type("Hello world, this is a test sentence.")
        assert isinstance(result, ContentType), \
            f"Should return ContentType, got {type(result)}"


class TestContentDetectionBehavior:
    """Test actual content detection behavior."""

    def test_detect_python_code(self):
        """Test detection of Python code."""
        try:
            from core.advanced_memory import detect_content_type, ContentType
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        # Python function
        result = detect_content_type("def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)")
        assert result == ContentType.CODE, f"Expected CODE, got {result}"

        # Python class
        result = detect_content_type("class UserService:\n    def __init__(self):\n        self.users = {}")
        assert result == ContentType.CODE, f"Expected CODE, got {result}"

        # Python imports
        result = detect_content_type("from typing import List, Dict\nimport asyncio\n\nasync def fetch(): pass")
        assert result == ContentType.CODE, f"Expected CODE, got {result}"

    def test_detect_javascript_code(self):
        """Test detection of JavaScript code."""
        try:
            from core.advanced_memory import detect_content_type, ContentType
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        # JavaScript function
        result = detect_content_type("function calculateSum(a, b) {\n    return a + b;\n}")
        assert result == ContentType.CODE, f"Expected CODE, got {result}"

        # ES6 syntax
        result = detect_content_type("const fetchData = async () => {\n    const response = await api.get();\n    return response;\n};")
        assert result == ContentType.CODE, f"Expected CODE, got {result}"

        # ES6 imports
        result = detect_content_type("import { useState, useEffect } from 'react';\n\nconst App = () => { };")
        assert result == ContentType.CODE, f"Expected CODE, got {result}"

    def test_detect_natural_text(self):
        """Test detection of natural language text."""
        try:
            from core.advanced_memory import detect_content_type, ContentType
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        # Simple text
        result = detect_content_type("The user prefers dark mode and concise responses.")
        assert result == ContentType.TEXT, f"Expected TEXT, got {result}"

        # Documentation
        result = detect_content_type("This component handles user authentication and manages session state. It provides methods for login, logout, and session refresh.")
        assert result == ContentType.TEXT, f"Expected TEXT, got {result}"

    def test_detect_multilingual_content(self):
        """Test detection of multilingual content."""
        try:
            from core.advanced_memory import detect_content_type, ContentType
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        # Chinese
        result = detect_content_type("这是一个测试文本，用于验证多语言检测功能。")
        assert result == ContentType.MULTILINGUAL, f"Expected MULTILINGUAL, got {result}"

        # Japanese
        result = detect_content_type("これはテストです。多言語検出をテストしています。")
        assert result == ContentType.MULTILINGUAL, f"Expected MULTILINGUAL, got {result}"

        # Korean
        result = detect_content_type("이것은 다국어 감지 기능을 테스트하기 위한 텍스트입니다.")
        assert result == ContentType.MULTILINGUAL, f"Expected MULTILINGUAL, got {result}"

    def test_detect_unknown_for_short_text(self):
        """Test that short text returns UNKNOWN."""
        try:
            from core.advanced_memory import detect_content_type, ContentType
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        result = detect_content_type("Hi")
        assert result == ContentType.UNKNOWN, f"Expected UNKNOWN for short text, got {result}"

        result = detect_content_type("")
        assert result == ContentType.UNKNOWN, f"Expected UNKNOWN for empty text, got {result}"


class TestEmbeddingRouterStructure:
    """Test EmbeddingRouter class structure by importing real class."""

    def test_embedding_router_importable(self):
        """EmbeddingRouter class should be importable."""
        try:
            from core.advanced_memory import EmbeddingRouter
        except ImportError:
            pytest.skip("advanced_memory not importable")

        assert EmbeddingRouter is not None

    def test_embedding_router_has_required_methods(self):
        """EmbeddingRouter should have embed, embed_batch, select_provider, get_routing_stats."""
        try:
            from core.advanced_memory import EmbeddingRouter
        except ImportError:
            pytest.skip("advanced_memory not importable")

        router = EmbeddingRouter()
        assert hasattr(router, "embed"), "Should have embed method"
        assert hasattr(router, "embed_batch"), "Should have embed_batch method"
        assert hasattr(router, "select_provider"), "Should have select_provider method"
        assert hasattr(router, "get_routing_stats"), "Should have get_routing_stats method"

    def test_create_embedding_router_importable(self):
        """create_embedding_router factory should be importable."""
        try:
            from core.advanced_memory import create_embedding_router
        except ImportError:
            pytest.skip("advanced_memory not importable")

        assert callable(create_embedding_router)


class TestEmbeddingRouterBehavior:
    """Test actual EmbeddingRouter behavior."""

    def test_router_initialization(self):
        """Test EmbeddingRouter initializes correctly."""
        try:
            from core.advanced_memory import EmbeddingRouter
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        router = EmbeddingRouter()

        # Should have initial stats
        stats = router.get_routing_stats()
        assert stats["total_routed"] == 0
        assert "by_content_type" in stats
        assert "by_provider" in stats
        assert "available_providers" in stats

    def test_router_prefer_local(self):
        """Test router with prefer_local flag."""
        try:
            from core.advanced_memory import EmbeddingRouter
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        router = EmbeddingRouter(prefer_local=True)
        stats = router.get_routing_stats()

        # With prefer_local, API providers should be marked unavailable
        assert stats["available_providers"]["local"] is True
        assert stats["available_providers"]["voyage"] is False
        assert stats["available_providers"]["openai"] is False

    @pytest.mark.asyncio
    async def test_router_embed_local(self):
        """Test router embedding with local provider."""
        try:
            from core.advanced_memory import (
                create_embedding_router,
                ContentType,
            )
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        # Force local-only mode
        router = create_embedding_router(prefer_local=True)

        result = await router.embed("Test text for embedding")
        assert result is not None
        assert len(result.embedding) > 0

        stats = router.get_routing_stats()
        assert stats["total_routed"] == 1
        assert stats["by_provider"]["local"] >= 1

    @pytest.mark.asyncio
    async def test_router_embed_with_force_type(self):
        """Test router with forced content type."""
        try:
            from core.advanced_memory import (
                create_embedding_router,
                ContentType,
            )
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        router = create_embedding_router(prefer_local=True)

        # Even text should be treated as code
        result = await router.embed(
            "The user prefers dark mode",
            force_type=ContentType.CODE
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_router_batch_embed(self):
        """Test router batch embedding."""
        try:
            from core.advanced_memory import create_embedding_router
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        router = create_embedding_router(prefer_local=True)

        texts = [
            "def hello(): pass",
            "User documentation",
            "class Widget: pass",
        ]
        results = await router.embed_batch(texts)

        assert len(results) == 3
        for r in results:
            assert len(r.embedding) > 0

        stats = router.get_routing_stats()
        assert stats["total_routed"] == 3


class TestRoutingStats:
    """Test routing statistics tracking."""

    @pytest.mark.asyncio
    async def test_stats_by_content_type(self):
        """Test that stats track content type correctly."""
        try:
            from core.advanced_memory import (
                create_embedding_router,
                ContentType,
            )
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        router = create_embedding_router(prefer_local=True)

        # Embed different content types
        await router.embed("def test(): pass")  # CODE
        await router.embed("Natural language text here for testing")  # TEXT
        await router.embed("class Widget:\n    def __init__(self): pass")  # CODE

        stats = router.get_routing_stats()
        assert stats["total_routed"] == 3
        # At least some should be classified as code
        assert stats["by_content_type"]["code"] >= 1

    @pytest.mark.asyncio
    async def test_stats_reset(self):
        """Test that stats can be reset."""
        try:
            from core.advanced_memory import create_embedding_router
        except ImportError:
            pytest.skip("advanced_memory module not importable")

        router = create_embedding_router(prefer_local=True)

        await router.embed("Test text")
        assert router.get_routing_stats()["total_routed"] == 1

        router.reset_stats()
        assert router.get_routing_stats()["total_routed"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
