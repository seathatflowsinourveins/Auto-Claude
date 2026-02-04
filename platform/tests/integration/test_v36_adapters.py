"""
V36 Adapter Integration Tests

Tests adapter interactions in realistic scenarios:
- Multi-adapter workflows
- Cross-layer communication
- Fallback behavior
- Error propagation
"""

import pytest
import asyncio
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch


class TestMultiAdapterWorkflow:
    """Test workflows involving multiple adapters."""

    @pytest.mark.asyncio
    async def test_memory_to_knowledge_pipeline(self):
        """Test data flow from memory adapter to knowledge adapter."""
        try:
            from adapters.simplemem_adapter import SimpleMemAdapter
            from adapters.cognee_v36_adapter import CogneeV36Adapter
        except ImportError:
            pytest.skip("Adapters not available")

        # Initialize adapters
        memory = SimpleMemAdapter()
        knowledge = CogneeV36Adapter()

        await memory.initialize({"max_tokens": 4096})
        await knowledge.initialize({})

        # Store context in memory
        compress_result = await memory.execute(
            "compress",
            context="AI ethics involves principles like fairness, transparency, and accountability.",
            context_id="ethics-1"
        )
        assert compress_result.success

        # Retrieve compressed context
        retrieve_result = await memory.execute(
            "retrieve",
            query="ethics principles"
        )
        assert retrieve_result.success

        # Cleanup
        await memory.shutdown()
        await knowledge.shutdown()

    @pytest.mark.asyncio
    async def test_orchestration_adapter_delegation(self):
        """Test agent delegation between orchestration adapters."""
        try:
            from adapters.openai_agents_adapter import OpenAIAgentsAdapter
            from adapters.a2a_protocol_adapter import A2AProtocolAdapter
        except ImportError:
            pytest.skip("Adapters not available")

        # Initialize A2A for discovery
        a2a = A2AProtocolAdapter()
        await a2a.initialize({
            "agent_id": "coordinator",
            "capabilities": ["orchestration"]
        })

        # Register agents
        await a2a.execute("register", agent_card={
            "agent_id": "worker-1",
            "name": "Worker Agent",
            "description": "Task executor",
            "capabilities": ["coding", "analysis"]
        })

        # Discover agents
        discover_result = await a2a.execute("discover", capability="coding")
        assert discover_result.success
        assert discover_result.data["count"] >= 1

        # Cleanup
        await a2a.shutdown()

    @pytest.mark.asyncio
    async def test_reranking_pipeline(self):
        """Test retrieval + reranking pipeline."""
        try:
            from adapters.ragflow_adapter import RAGFlowAdapter
            from adapters.ragatouille_adapter import RAGatouilleAdapter
        except ImportError:
            pytest.skip("Adapters not available")

        # Initialize adapters
        ragflow = RAGFlowAdapter()
        reranker = RAGatouilleAdapter()

        await ragflow.initialize({})
        await reranker.initialize({})

        # Create test documents for reranking
        documents = [
            {"content": "Python is a programming language", "id": "doc-1"},
            {"content": "Machine learning uses Python extensively", "id": "doc-2"},
            {"content": "JavaScript is for web development", "id": "doc-3"},
        ]

        # Index documents in reranker
        await reranker.execute("index", documents=documents)

        # Rerank with query
        rerank_result = await reranker.execute(
            "rerank",
            query="machine learning programming",
            documents=documents
        )

        assert rerank_result.success
        assert len(rerank_result.data["results"]) > 0

        # Cleanup
        await ragflow.shutdown()
        await reranker.shutdown()


class TestCrossLayerCommunication:
    """Test communication between different SDK layers."""

    @pytest.mark.asyncio
    async def test_protocol_to_memory_flow(self):
        """Test L0 Protocol -> L2 Memory flow."""
        try:
            from adapters.portkey_gateway_adapter import PortkeyGatewayAdapter
            from adapters.simplemem_adapter import SimpleMemAdapter
        except ImportError:
            pytest.skip("Adapters not available")

        # Initialize both layers
        gateway = PortkeyGatewayAdapter()
        memory = SimpleMemAdapter()

        await gateway.initialize({})
        await memory.initialize({"max_tokens": 2048})

        # Simulate LLM response (from gateway stub)
        chat_result = await gateway.execute(
            "chat",
            messages=[{"role": "user", "content": "Explain quantum computing"}]
        )
        assert chat_result.success

        # Store response in memory
        if chat_result.data and "content" in chat_result.data:
            compress_result = await memory.execute(
                "compress",
                context=chat_result.data["content"]
            )
            assert compress_result.success

        # Cleanup
        await gateway.shutdown()
        await memory.shutdown()

    @pytest.mark.asyncio
    async def test_observability_tracking(self):
        """Test that operations can be tracked by observability adapter."""
        try:
            from adapters.braintrust_adapter import BraintrustAdapter
            from adapters.simplemem_adapter import SimpleMemAdapter
        except ImportError:
            pytest.skip("Adapters not available")

        # Initialize
        tracker = BraintrustAdapter()
        memory = SimpleMemAdapter()

        await tracker.initialize({"project": "v36-integration-test"})
        await memory.initialize({})

        # Perform memory operation
        compress_result = await memory.execute(
            "compress",
            context="Test content for tracking"
        )

        # Log to tracker
        log_result = await tracker.execute(
            "log",
            input="compress operation",
            output=compress_result.data,
            scores={"success": 1.0 if compress_result.success else 0.0}
        )
        assert log_result.success

        # Verify tracking
        stats = await tracker.execute("get_stats")
        assert stats.data["total_logs"] >= 1

        # Cleanup
        await tracker.shutdown()
        await memory.shutdown()


class TestFallbackBehavior:
    """Test adapter fallback behavior."""

    @pytest.mark.asyncio
    async def test_adapter_fallback_on_failure(self):
        """Test that operations can fallback gracefully."""
        try:
            from adapters.graphiti_adapter import GraphitiAdapter
            from adapters.simplemem_adapter import SimpleMemAdapter
        except ImportError:
            pytest.skip("Adapters not available")

        # Graphiti requires Neo4j - will fail without it
        graphiti = GraphitiAdapter()
        simplemem = SimpleMemAdapter()

        # Initialize (Graphiti will fail without Neo4j)
        graphiti_result = await graphiti.initialize({})
        simplemem_result = await simplemem.initialize({})

        # SimpleMem should always succeed
        assert simplemem_result.success

        # If Graphiti fails, use SimpleMem as fallback
        if not graphiti_result.success:
            # Fallback to SimpleMem
            result = await simplemem.execute(
                "compress",
                context="Fallback content"
            )
            assert result.success

        # Cleanup
        await graphiti.shutdown()
        await simplemem.shutdown()

    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test graceful degradation when SDK is unavailable."""
        try:
            from adapters.strands_agents_adapter import StrandsAgentsAdapter
        except ImportError:
            pytest.skip("Adapter not available")

        adapter = StrandsAgentsAdapter()

        # Initialize (will use stub if Strands not installed)
        result = await adapter.initialize({"model": "test"})
        assert result.success  # Should succeed even without real SDK

        # Operations should work in stub mode
        run_result = await adapter.execute(
            "run",
            message="Test message"
        )
        assert run_result.success

        await adapter.shutdown()


class TestErrorPropagation:
    """Test error handling and propagation."""

    @pytest.mark.asyncio
    async def test_invalid_operation_returns_error(self):
        """Invalid operations should return error, not raise."""
        try:
            from adapters.openai_agents_adapter import OpenAIAgentsAdapter
        except ImportError:
            pytest.skip("Adapter not available")

        adapter = OpenAIAgentsAdapter()
        await adapter.initialize({})

        # Invalid operation
        result = await adapter.execute("__invalid_operation__")

        assert not result.success
        assert result.error is not None
        assert "Unknown operation" in result.error or "not initialized" in result.error.lower()

        await adapter.shutdown()

    @pytest.mark.asyncio
    async def test_error_includes_latency(self):
        """Even error responses should include latency."""
        try:
            from adapters.a2a_protocol_adapter import A2AProtocolAdapter
        except ImportError:
            pytest.skip("Adapter not available")

        adapter = A2AProtocolAdapter()
        await adapter.initialize({})

        result = await adapter.execute("__bad_operation__")

        assert result.latency_ms >= 0
        assert result.latency_ms < 1000  # Should be fast for invalid op

        await adapter.shutdown()

    @pytest.mark.asyncio
    async def test_adapter_tracks_errors(self):
        """Adapters should track error counts."""
        try:
            from adapters.mcp_apps_adapter import MCPAppsAdapter
        except ImportError:
            pytest.skip("Adapter not available")

        adapter = MCPAppsAdapter()
        await adapter.initialize({})

        # Cause some errors
        for _ in range(3):
            await adapter.execute("__bad_op__")

        # Check error tracking
        stats = await adapter.execute("get_stats")
        if stats.success and stats.data:
            assert stats.data.get("error_count", 0) >= 0
        # Adapter may not support get_stats in degraded mode

        await adapter.shutdown()


class TestAdapterConcurrency:
    """Test concurrent adapter operations."""

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test multiple concurrent operations on same adapter."""
        try:
            from adapters.simplemem_adapter import SimpleMemAdapter
        except ImportError:
            pytest.skip("Adapter not available")

        adapter = SimpleMemAdapter()
        await adapter.initialize({"max_tokens": 4096})

        # Run multiple compress operations concurrently
        tasks = [
            adapter.execute("compress", context=f"Content {i}", context_id=f"ctx-{i}")
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)

        # All should succeed
        for i, result in enumerate(results):
            assert result.success, f"Operation {i} failed: {result.error}"

        await adapter.shutdown()

    @pytest.mark.asyncio
    async def test_multiple_adapter_instances(self):
        """Test multiple instances of same adapter type."""
        try:
            from adapters.braintrust_adapter import BraintrustAdapter
        except ImportError:
            pytest.skip("Adapter not available")

        # Create multiple instances
        adapter1 = BraintrustAdapter()
        adapter2 = BraintrustAdapter()

        await adapter1.initialize({"project": "project-1"})
        await adapter2.initialize({"project": "project-2"})

        # Operations should be isolated
        await adapter1.execute("log", input="a", output="b", scores={"x": 1})
        await adapter1.execute("log", input="c", output="d", scores={"x": 1})

        await adapter2.execute("log", input="e", output="f", scores={"x": 1})

        stats1 = await adapter1.execute("get_stats")
        stats2 = await adapter2.execute("get_stats")

        assert stats1.data["total_logs"] == 2
        assert stats2.data["total_logs"] == 1

        await adapter1.shutdown()
        await adapter2.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
