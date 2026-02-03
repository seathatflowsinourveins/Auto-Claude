"""
V36 Test Configuration and Fixtures

Shared fixtures and configuration for all test suites.
"""

import pytest
import asyncio
import sys
import os
from typing import Dict, Any, List
from pathlib import Path

# Add platform to Python path
platform_path = Path(__file__).parent.parent
if str(platform_path) not in sys.path:
    sys.path.insert(0, str(platform_path))

# Also add the parent (unleash root) for absolute imports
root_path = platform_path.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))


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
        from platform.adapters.simplemem_adapter import SimpleMemAdapter
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
        from platform.adapters.braintrust_adapter import BraintrustAdapter
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
        from platform.adapters.a2a_protocol_adapter import A2AProtocolAdapter
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
        from platform.adapters.ragatouille_adapter import RAGatouilleAdapter
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
        from platform.adapters.portkey_gateway_adapter import PortkeyGatewayAdapter
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
        from platform.core.memory.backends.base import MemoryEntry, MemoryTier
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
        from platform.core.memory.backends.in_memory import InMemoryTierBackend
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
        "markers", "requires_sdk(sdk_name): marks test as requiring a specific SDK"
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
