"""
V44 Test Configuration and Fixtures

Shared fixtures and configuration for all test suites.
Fixes stdlib 'platform' module collision.
"""

import sys
import os
from pathlib import Path

# CRITICAL: Fix platform namespace collision BEFORE any other imports.
# Our package directory is named 'platform' which shadows Python's stdlib
# 'platform' module. We must ensure stdlib is loaded first.
_platform_pkg_dir = str(Path(__file__).parent.parent)
_unleash_root = str(Path(__file__).parent.parent.parent)

# Temporarily remove paths that could resolve to our 'platform' package
_original_path = sys.path[:]
sys.path = [p for p in sys.path if os.path.normpath(p) not in (
    os.path.normpath(_platform_pkg_dir),
    os.path.normpath(_unleash_root),
)]

# Force-load stdlib platform module
import importlib
if 'platform' in sys.modules:
    _platform_mod = sys.modules['platform']
    if not hasattr(_platform_mod, 'python_version'):
        del sys.modules['platform']
        import platform  # noqa: F811
else:
    import platform  # noqa: F811

# Restore paths
sys.path = _original_path

# Add platform dir for 'from adapters...' and 'from core...' imports
# IMPORTANT: _platform_pkg_dir must be inserted LAST (so it ends up at index 0)
# to ensure 'from core...' resolves to platform/core/ not unleash/core/
if _unleash_root not in sys.path:
    sys.path.insert(0, _unleash_root)
if _platform_pkg_dir not in sys.path:
    sys.path.insert(0, _platform_pkg_dir)

import pytest
import asyncio
from typing import Dict, Any, List


# =============================================================================
# Async Configuration
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Adapter Fixtures
# =============================================================================

@pytest.fixture
async def simplemem_adapter():
    """Create and initialize a SimpleMem adapter."""
    try:
        from adapters.simplemem_adapter import SimpleMemAdapter
        adapter = SimpleMemAdapter()
        await adapter.initialize({"max_tokens": 4096})
        yield adapter
        await adapter.shutdown()
    except ImportError:
        pytest.skip("SimpleMemAdapter not available")


@pytest.fixture
async def braintrust_adapter():
    """Create and initialize a Braintrust adapter."""
    try:
        from adapters.braintrust_adapter import BraintrustAdapter
        adapter = BraintrustAdapter()
        await adapter.initialize({"project": "test-project"})
        yield adapter
        await adapter.shutdown()
    except ImportError:
        pytest.skip("BraintrustAdapter not available")


@pytest.fixture
async def a2a_adapter():
    """Create and initialize an A2A Protocol adapter."""
    try:
        from adapters.a2a_protocol_adapter import A2AProtocolAdapter
        adapter = A2AProtocolAdapter()
        await adapter.initialize({
            "agent_id": "test-agent",
            "capabilities": ["testing"]
        })
        yield adapter
        await adapter.shutdown()
    except ImportError:
        pytest.skip("A2AProtocolAdapter not available")


@pytest.fixture
async def ragatouille_adapter():
    """Create and initialize a RAGatouille adapter."""
    try:
        from adapters.ragatouille_adapter import RAGatouilleAdapter
        adapter = RAGatouilleAdapter()
        await adapter.initialize({})
        yield adapter
        await adapter.shutdown()
    except ImportError:
        pytest.skip("RAGatouilleAdapter not available")


@pytest.fixture
async def portkey_adapter():
    """Create and initialize a Portkey Gateway adapter."""
    try:
        from adapters.portkey_gateway_adapter import PortkeyGatewayAdapter
        adapter = PortkeyGatewayAdapter()
        await adapter.initialize({})
        yield adapter
        await adapter.shutdown()
    except ImportError:
        pytest.skip("PortkeyGatewayAdapter not available")


# =============================================================================
# Memory Fixtures
# =============================================================================

@pytest.fixture
def memory_entry_factory():
    """Factory for creating memory entries."""
    try:
        from core.memory.backends.base import MemoryEntry, MemoryTier
    except ImportError:
        pytest.skip("Memory backend not available")

    def create_entry(
        id: str = "test-entry",
        content: str = "Test content",
        tier: str = "main_context",
        **kwargs
    ):
        return MemoryEntry(
            id=id,
            content=content,
            tier=MemoryTier(tier),
            **kwargs
        )

    return create_entry


@pytest.fixture
async def in_memory_backend():
    """Create an in-memory tier backend."""
    try:
        from core.memory.backends.in_memory import InMemoryTierBackend
        backend = InMemoryTierBackend()
        yield backend
        await backend.clear()
    except ImportError:
        pytest.skip("InMemoryTierBackend not available")


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_documents() -> List[Dict[str, Any]]:
    """Sample documents for testing."""
    return [
        {
            "id": "doc-1",
            "content": "Python is a versatile programming language used in web development, data science, and AI.",
            "metadata": {"topic": "programming", "difficulty": "beginner"}
        },
        {
            "id": "doc-2",
            "content": "Machine learning models learn patterns from training data to make predictions.",
            "metadata": {"topic": "ml", "difficulty": "intermediate"}
        },
        {
            "id": "doc-3",
            "content": "Natural language processing enables computers to understand human language.",
            "metadata": {"topic": "nlp", "difficulty": "advanced"}
        },
        {
            "id": "doc-4",
            "content": "Deep learning uses neural networks with many layers to extract features.",
            "metadata": {"topic": "deep_learning", "difficulty": "advanced"}
        },
    ]


@pytest.fixture
def sample_agents() -> List[Dict[str, Any]]:
    """Sample agent configurations for testing."""
    return [
        {
            "agent_id": "coder-agent",
            "name": "Code Writer",
            "description": "Writes and reviews code",
            "capabilities": ["coding", "review", "debugging"]
        },
        {
            "agent_id": "research-agent",
            "name": "Researcher",
            "description": "Conducts research and analysis",
            "capabilities": ["research", "analysis", "summarization"]
        },
        {
            "agent_id": "test-agent",
            "name": "Tester",
            "description": "Tests and validates code",
            "capabilities": ["testing", "qa", "validation"]
        },
    ]


# =============================================================================
# Markers
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
    config.addinivalue_line(
        "markers", "requires_sdk(sdk_name): marks test as requiring a specific SDK"
    )
    config.addinivalue_line(
        "markers", "requires_api_key(key_name): marks test as requiring an API key"
    )


# =============================================================================
# Skip Helpers
# =============================================================================

def skip_if_no_sdk(sdk_name: str):
    """Skip test if SDK is not available."""
    sdk_checks = {
        "letta": lambda: __import__("letta"),
        "graphiti": lambda: __import__("graphiti_core"),
        "openai_swarm": lambda: __import__("swarm"),
        "strands": lambda: __import__("strands"),
        "cognee": lambda: __import__("cognee"),
        "ragatouille": lambda: __import__("ragatouille"),
        "braintrust": lambda: __import__("braintrust"),
        "portkey": lambda: __import__("portkey_ai"),
    }

    if sdk_name in sdk_checks:
        try:
            sdk_checks[sdk_name]()
            return False
        except ImportError:
            return True
    return False


@pytest.fixture
def check_sdk():
    """Fixture to check SDK availability."""
    def _check(sdk_name: str):
        if skip_if_no_sdk(sdk_name):
            pytest.skip(f"{sdk_name} SDK not available")
    return _check


# =============================================================================
# Research Adapter Fixtures
# =============================================================================

@pytest.fixture
async def exa_adapter():
    """Create and initialize an Exa adapter."""
    try:
        from adapters.exa_adapter import ExaAdapter
        adapter = ExaAdapter()
        await adapter.initialize({})
        yield adapter
        await adapter.shutdown()
    except ImportError:
        pytest.skip("ExaAdapter not available")


@pytest.fixture
async def tavily_adapter():
    """Create and initialize a Tavily adapter."""
    try:
        from adapters.tavily_adapter import TavilyAdapter
        adapter = TavilyAdapter()
        await adapter.initialize({})
        yield adapter
        await adapter.shutdown()
    except ImportError:
        pytest.skip("TavilyAdapter not available")


@pytest.fixture
async def jina_adapter():
    """Create and initialize a Jina adapter."""
    try:
        from adapters.jina_adapter import JinaAdapter
        adapter = JinaAdapter()
        await adapter.initialize({})
        yield adapter
        await adapter.shutdown()
    except ImportError:
        pytest.skip("JinaAdapter not available")


@pytest.fixture
async def perplexity_adapter():
    """Create and initialize a Perplexity adapter."""
    try:
        from adapters.perplexity_adapter import PerplexityAdapter
        adapter = PerplexityAdapter()
        await adapter.initialize({})
        yield adapter
        await adapter.shutdown()
    except ImportError:
        pytest.skip("PerplexityAdapter not available")


# =============================================================================
# MCP Fixtures
# =============================================================================

@pytest.fixture
def mcp_server():
    """Create a FastMCP server for testing."""
    try:
        from hooks.hook_utils import FastMCPServer
        return FastMCPServer("test-server", version="1.0.0")
    except ImportError:
        pytest.skip("FastMCPServer not available")


@pytest.fixture
def mcp_context(mcp_server):
    """Create an MCP context for testing."""
    try:
        from hooks.hook_utils import MCPContext
        return MCPContext(request_id="test-001", server=mcp_server)
    except ImportError:
        pytest.skip("MCPContext not available")


@pytest.fixture
def lifespan_context(mcp_server):
    """Create a lifespan context for testing."""
    try:
        from hooks.hook_utils import LifespanContext
        return LifespanContext(server=mcp_server)
    except ImportError:
        pytest.skip("LifespanContext not available")


# =============================================================================
# Search Results Fixtures
# =============================================================================

@pytest.fixture
def sample_search_results() -> List[Dict[str, Any]]:
    """Sample search results for testing."""
    return [
        {
            "title": "Introduction to Machine Learning",
            "url": "https://example.com/ml-intro",
            "content": "Machine learning is a branch of artificial intelligence...",
            "score": 0.95,
            "published_date": "2026-01-15"
        },
        {
            "title": "Deep Learning Fundamentals",
            "url": "https://example.com/deep-learning",
            "content": "Deep learning uses neural networks with many layers...",
            "score": 0.87,
            "published_date": "2026-01-10"
        },
        {
            "title": "Natural Language Processing Guide",
            "url": "https://example.com/nlp-guide",
            "content": "NLP enables computers to understand human language...",
            "score": 0.82,
            "published_date": "2026-01-05"
        },
    ]


@pytest.fixture
def sample_embeddings() -> List[List[float]]:
    """Sample embeddings for testing."""
    import random
    random.seed(42)
    return [
        [random.random() for _ in range(768)]
        for _ in range(5)
    ]


# NOTE: Additional markers merged into pytest_configure above (removed duplicate)
