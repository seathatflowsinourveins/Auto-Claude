"""
End-to-End Orchestration Tests - ITERATION 7.

These tests verify the complete orchestration pipeline:
- Agent creation and configuration
- Multi-framework coordination
- Real Voyage AI embeddings integration
- Memory operations with semantic search
- Complete workflow execution

This integrates all components from previous iterations:
- ITERATION 4: Agent SDK Layer
- ITERATION 5: Multi-Agent Coordination
- ITERATION 6: Voyage AI Embeddings
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest


# =============================================================================
# Constants
# =============================================================================

VOYAGE_API_KEY = "pa-KCpYL_zzmvoPK1dM6tN5kdCD8e6qnAndC-dSTlCuzK4"


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def voyage_api_key() -> str:
    """Get the Voyage AI API key."""
    return VOYAGE_API_KEY


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test file structure
        src_dir = Path(tmpdir) / "src"
        src_dir.mkdir()
        (src_dir / "main.py").write_text("print('Hello, World!')")
        (src_dir / "utils.py").write_text("def add(a, b): return a + b")
        yield tmpdir


@pytest.fixture
def sample_knowledge_base() -> list[dict]:
    """Sample knowledge base for semantic search."""
    return [
        {"id": "doc1", "content": "Machine learning is artificial intelligence that learns from data."},
        {"id": "doc2", "content": "Python is a programming language popular for data science."},
        {"id": "doc3", "content": "Neural networks process information using connected nodes."},
        {"id": "doc4", "content": "Deep learning uses multiple layers of neural networks."},
        {"id": "doc5", "content": "Natural language processing handles human language with AI."},
    ]


# =============================================================================
# Component Availability Tests
# =============================================================================

class TestOrchestrationComponentsAvailable:
    """Verify all orchestration components are available."""

    def test_agent_sdk_layer_available(self):
        """Test Agent SDK Layer is importable."""
        from core.orchestration.agent_sdk_layer import (
            Agent,
            AgentConfig,
            AgentResult,
            CLAUDE_AGENT_SDK_AVAILABLE,
            ANTHROPIC_AVAILABLE,
        )
        assert CLAUDE_AGENT_SDK_AVAILABLE or ANTHROPIC_AVAILABLE
        assert Agent is not None
        assert AgentConfig is not None

    def test_multi_agent_coordination_available(self):
        """Test multi-agent coordination is available."""
        from core.orchestration import (
            UnifiedOrchestrator,
            OrchestrationResult,
        )
        from core.orchestration.agent_sdk_layer import ANTHROPIC_AVAILABLE
        from core.orchestration.langgraph_agents import LANGGRAPH_AVAILABLE
        assert UnifiedOrchestrator is not None
        assert LANGGRAPH_AVAILABLE or ANTHROPIC_AVAILABLE

    def test_embedding_layer_available(self):
        """Test embedding layer is available."""
        from core.orchestration.embedding_layer import (
            EmbeddingLayer,
            create_embedding_layer,
            embed_texts,
            VOYAGE_AVAILABLE,
        )
        assert VOYAGE_AVAILABLE is True
        assert EmbeddingLayer is not None

    def test_all_exports_from_orchestration(self):
        """Test all key exports from orchestration module."""
        from core.orchestration import (
            # Multi-agent
            UnifiedOrchestrator,
            OrchestrationResult,
            # Embeddings
            EmbeddingLayer,
            create_embedding_layer,
            embed_texts,
        )
        from core.orchestration.agent_sdk_layer import (
            # Agent SDK
            Agent,
            AgentConfig,
            create_agent,
        )
        # All imports should succeed
        assert Agent is not None
        assert AgentConfig is not None
        assert UnifiedOrchestrator is not None
        assert EmbeddingLayer is not None


# =============================================================================
# Agent + Embedding Integration Tests
# =============================================================================

class TestAgentEmbeddingIntegration:
    """Test agent operations with embedding support."""

    def test_agent_config_with_memory_tools(self):
        """Test agent configuration includes memory tools."""
        from core.orchestration.agent_sdk_layer import AgentConfig

        config = AgentConfig(
            name="memory_agent",
            tools=["Read", "Write", "SaveMemory", "SearchMemory"],
        )
        assert "SaveMemory" in config.tools
        assert "SearchMemory" in config.tools

    def test_agent_builds_memory_tools(self):
        """Test agent builds memory tool definitions."""
        from core.orchestration.agent_sdk_layer import Agent, AgentConfig

        config = AgentConfig(
            name="memory_agent",
            tools=["SaveMemory", "SearchMemory"],
        )
        agent = Agent(config)
        tools = agent._build_tools()

        tool_names = [t["name"] for t in tools]
        assert "save_memory" in tool_names
        assert "search_memory" in tool_names

    @pytest.mark.asyncio
    async def test_agent_with_file_operations(self, temp_workspace):
        """Test agent can read files."""
        from core.orchestration.agent_sdk_layer import Agent, AgentConfig

        config = AgentConfig(
            name="file_agent",
            tools=["Read", "ListDir"],
        )
        agent = Agent(config)

        # Execute read operation
        main_file = str(Path(temp_workspace) / "src" / "main.py")
        tool_calls = [{"name": "read_file", "input": {"path": main_file}, "id": "call_1"}]
        results = await agent._execute_tools(tool_calls)

        assert len(results) == 1
        assert "Hello, World!" in results[0]["content"]


# =============================================================================
# Multi-Agent + Embedding Workflow Tests
# =============================================================================

class TestMultiAgentEmbeddingWorkflow:
    """Test multi-agent workflows with embedding integration."""

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self):
        """Test orchestrator initializes correctly."""
        from core.orchestration import UnifiedOrchestrator

        orchestrator = UnifiedOrchestrator()
        assert orchestrator is not None
        assert orchestrator.available_frameworks is not None

    @pytest.mark.asyncio
    async def test_orchestrator_with_embedding_context(self, sample_knowledge_base):
        """Test orchestrator can work with embedded context."""
        from core.orchestration import UnifiedOrchestrator
        from core.orchestration.embedding_layer import EmbeddingLayer

        # Create orchestrator
        orchestrator = UnifiedOrchestrator()

        # Create embedding layer (no API call needed for config)
        layer = EmbeddingLayer()

        assert orchestrator is not None
        assert layer is not None
        assert layer.config.model == "voyage-4-large"  # Default model is now voyage-4-large


# =============================================================================
# Semantic Search Workflow Tests
# =============================================================================

class TestSemanticSearchWorkflow:
    """Test semantic search workflows."""

    def test_knowledge_base_structure(self, sample_knowledge_base):
        """Test knowledge base fixture structure."""
        assert len(sample_knowledge_base) == 5
        assert all("id" in doc for doc in sample_knowledge_base)
        assert all("content" in doc for doc in sample_knowledge_base)

    def test_embedding_config_for_search(self):
        """Test embedding configuration for semantic search."""
        from core.orchestration.embedding_layer import (
            EmbeddingConfig,
            InputType,
        )

        config = EmbeddingConfig(
            model="voyage-4-large",
            cache_enabled=True,
        )
        assert config.model == "voyage-4-large"
        assert config.cache_enabled is True

        # Verify input types
        assert InputType.DOCUMENT.value == "document"
        assert InputType.QUERY.value == "query"


# =============================================================================
# Agent Execution Pipeline Tests
# =============================================================================

class TestAgentExecutionPipeline:
    """Test complete agent execution pipeline."""

    @pytest.mark.asyncio
    async def test_agent_tool_execution(self, temp_workspace):
        """Test agent executes tools correctly."""
        from core.orchestration.agent_sdk_layer import Agent, AgentConfig

        config = AgentConfig(
            name="exec_agent",
            tools=["Read", "ListDir", "Bash"],
        )
        agent = Agent(config)

        # Test list directory
        tool_calls = [{
            "name": "list_directory",
            "input": {"path": temp_workspace},
            "id": "call_1"
        }]
        results = await agent._execute_tools(tool_calls)

        assert len(results) == 1
        assert "src" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_parallel_tool_execution(self, temp_workspace):
        """Test agent executes multiple tools in parallel."""
        from core.orchestration.agent_sdk_layer import Agent, AgentConfig

        config = AgentConfig(
            name="parallel_agent",
            tools=["Read", "ListDir"],
        )
        agent = Agent(config)

        # Multiple tool calls
        main_path = str(Path(temp_workspace) / "src" / "main.py")
        utils_path = str(Path(temp_workspace) / "src" / "utils.py")

        tool_calls = [
            {"name": "read_file", "input": {"path": main_path}, "id": "call_1"},
            {"name": "read_file", "input": {"path": utils_path}, "id": "call_2"},
        ]

        start = time.time()
        results = await agent._execute_tools(tool_calls)
        duration = time.time() - start

        assert len(results) == 2
        assert "Hello" in results[0]["content"]
        assert "add" in results[1]["content"]

    @pytest.mark.asyncio
    async def test_error_handling_in_pipeline(self):
        """Test error handling in execution pipeline."""
        from core.orchestration.agent_sdk_layer import Agent, AgentConfig

        config = AgentConfig(
            name="error_agent",
            tools=["Read"],
        )
        agent = Agent(config)

        # Try to read non-existent file
        tool_calls = [{
            "name": "read_file",
            "input": {"path": "/nonexistent/file.txt"},
            "id": "call_1"
        }]
        results = await agent._execute_tools(tool_calls)

        assert len(results) == 1
        # Should contain error message
        content = results[0]["content"].lower()
        assert "error" in content or "not found" in content


# =============================================================================
# Orchestration Result Verification Tests
# =============================================================================

class TestOrchestrationResults:
    """Test orchestration result structures."""

    def test_orchestration_result_structure(self):
        """Test OrchestrationResult dataclass."""
        from core.orchestration import OrchestrationResult

        result = OrchestrationResult(
            success=True,
            output={"key": "value"},
            framework="claude_agent_sdk",
            metadata={"execution_time": 1.5, "agent_count": 2},
        )

        assert result.success is True
        assert result.output["key"] == "value"
        assert result.framework == "claude_agent_sdk"
        assert result.metadata["execution_time"] == 1.5
        assert result.metadata["agent_count"] == 2
        assert result.error is None

    def test_agent_result_structure(self):
        """Test AgentResult dataclass."""
        from core.orchestration.agent_sdk_layer import AgentResult

        result = AgentResult(
            output="Task completed",
            tool_calls=[{"name": "read_file", "input": {"path": "test.txt"}}],
            messages=[{"role": "assistant", "content": "Done"}],
            success=True,
        )

        assert result.success is True
        assert result.output == "Task completed"
        assert len(result.tool_calls) == 1
        assert result.error is None


# =============================================================================
# Embedding Result Verification Tests
# =============================================================================

class TestEmbeddingResults:
    """Test embedding result structures."""

    def test_embedding_result_structure(self):
        """Test EmbeddingResult dataclass."""
        from core.orchestration.embedding_layer import EmbeddingResult

        result = EmbeddingResult(
            embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            model="voyage-3",
            input_type="document",
            total_tokens=100,
            cached_count=0,
            dimension=3,
        )

        assert result.count == 2
        assert result.dimension == 3
        assert result.model == "voyage-3"
        assert result.total_tokens == 100

    def test_embedding_model_dimensions(self):
        """Verify embedding model dimension constants."""
        from core.orchestration.embedding_layer import EmbeddingModel

        # Voyage 4 Series - AVAILABLE (200M tokens each)
        assert EmbeddingModel.VOYAGE_4_LARGE.dimension == 1024
        assert EmbeddingModel.VOYAGE_4.dimension == 1024
        assert EmbeddingModel.VOYAGE_4_LITE.dimension == 1024
        assert EmbeddingModel.VOYAGE_CODE_3.dimension == 2048

        # Legacy Voyage 3 Series - DEPRECATED (0 tokens on free tier)
        assert EmbeddingModel.VOYAGE_3.dimension == 1024
        assert EmbeddingModel.VOYAGE_3_LARGE.dimension == 1024
        assert EmbeddingModel.VOYAGE_3_LITE.dimension == 512

    def test_embedding_model_free_tier_availability(self):
        """Verify which models are available on free tier."""
        from core.orchestration.embedding_layer import EmbeddingModel

        # Voyage 4 Series - AVAILABLE on free tier
        assert EmbeddingModel.VOYAGE_4_LARGE.is_available_free_tier is True
        assert EmbeddingModel.VOYAGE_4.is_available_free_tier is True
        assert EmbeddingModel.VOYAGE_4_LITE.is_available_free_tier is True
        assert EmbeddingModel.VOYAGE_CODE_3.is_available_free_tier is True

        # Voyage 3 Series - NOT available on free tier (0 tokens)
        assert EmbeddingModel.VOYAGE_3.is_available_free_tier is False
        assert EmbeddingModel.VOYAGE_3_LARGE.is_available_free_tier is False
        assert EmbeddingModel.VOYAGE_3_LITE.is_available_free_tier is False


# =============================================================================
# Integration Workflow Tests
# =============================================================================

class TestIntegrationWorkflows:
    """Test complete integration workflows."""

    @pytest.mark.asyncio
    async def test_agent_creation_factory(self):
        """Test agent creation factory function."""
        from core.orchestration.agent_sdk_layer import create_agent

        agent = await create_agent(
            name="factory_test_agent",
            model="claude-sonnet-4-20250514",
            tools=["Read", "Write", "Bash"],
        )

        assert agent.config.name == "factory_test_agent"
        assert agent.config.tools == ["Read", "Write", "Bash"]

    @pytest.mark.asyncio
    async def test_embedding_layer_factory(self):
        """Test embedding layer factory function."""
        from core.orchestration.embedding_layer import create_embedding_layer

        layer = create_embedding_layer(
            model="voyage-3",
            cache_enabled=True,
        )

        assert layer.config.model == "voyage-3"
        assert layer.config.cache_enabled is True
        assert layer.is_initialized is False

    @pytest.mark.asyncio
    async def test_orchestrator_framework_selection(self):
        """Test orchestrator framework selection."""
        from core.orchestration import UnifiedOrchestrator

        orchestrator = UnifiedOrchestrator()
        frameworks = orchestrator.available_frameworks

        # Should have at least one framework
        assert len(frameworks) >= 1

        # Test framework selection
        selected = orchestrator._select_framework(prefer=None)
        assert selected in frameworks


# =============================================================================
# Performance Baseline Tests
# =============================================================================

class TestPerformanceBaselines:
    """Establish performance baselines for orchestration."""

    def test_agent_creation_speed(self):
        """Test agent creation is fast."""
        from core.orchestration.agent_sdk_layer import Agent, AgentConfig

        start = time.time()
        for _ in range(100):
            config = AgentConfig(name="test", tools=["Read"])
            agent = Agent(config)
        duration = time.time() - start

        # 100 agent creations should take < 1 second
        assert duration < 1.0, f"Agent creation too slow: {duration}s for 100 agents"

    def test_embedding_layer_creation_speed(self):
        """Test embedding layer creation is fast."""
        from core.orchestration.embedding_layer import create_embedding_layer

        start = time.time()
        for _ in range(100):
            layer = create_embedding_layer(model="voyage-3")
        duration = time.time() - start

        # 100 layer creations should take < 1 second
        assert duration < 1.0, f"Layer creation too slow: {duration}s for 100 layers"

    def test_orchestrator_creation_speed(self):
        """Test orchestrator creation is fast."""
        from core.orchestration import UnifiedOrchestrator

        start = time.time()
        for _ in range(10):
            orchestrator = UnifiedOrchestrator()
        duration = time.time() - start

        # 10 orchestrator creations should take < 2 seconds
        assert duration < 2.0, f"Orchestrator creation too slow: {duration}s for 10"


# =============================================================================
# Slow API Integration Tests
# =============================================================================

@pytest.mark.slow
class TestRealAPIIntegration:
    """Test real API integration (slow tests with rate limiting)."""

    @pytest.mark.asyncio
    async def test_embedding_with_orchestrator_context(self, voyage_api_key, sample_knowledge_base):
        """
        Test embeddings work with orchestrator context.
        This is a real API test - respects rate limits.
        """
        from core.orchestration.embedding_layer import create_embedding_layer
        from core.orchestration import UnifiedOrchestrator

        # Create embedding layer with real API - use voyage-4-large (200M tokens available)
        layer = create_embedding_layer(
            model="voyage-4-large",
            api_key=voyage_api_key,
            cache_enabled=True,
        )
        await layer.initialize()

        # Create orchestrator
        orchestrator = UnifiedOrchestrator()

        # Embed knowledge base documents
        documents = [doc["content"] for doc in sample_knowledge_base]
        doc_result = await layer.embed_documents(documents)

        assert doc_result.count == 5
        assert doc_result.dimension == 1024

        # Verify orchestrator is ready
        assert len(orchestrator.available_frameworks) >= 1


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    # Run fast tests by default
    pytest.main([__file__, "-v", "-m", "not slow"])
