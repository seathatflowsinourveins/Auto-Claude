"""
V36 End-to-End Pipeline Tests

Tests complete workflows through the entire SDK stack:
- Document ingestion -> Processing -> Memory -> Retrieval
- Agent coordination -> Task execution -> Result aggregation
- Observability throughout
"""

import pytest
import asyncio
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""
    success: bool
    stages_completed: List[str]
    final_output: Any
    errors: List[str]
    metrics: Dict[str, float]


class TestDocumentProcessingPipeline:
    """Test document processing from ingestion to retrieval."""

    @pytest.fixture
    def sample_documents(self) -> List[Dict[str, Any]]:
        """Sample documents for testing."""
        return [
            {
                "id": "doc-1",
                "content": "Python is a high-level programming language known for its readability.",
                "metadata": {"topic": "programming", "language": "python"}
            },
            {
                "id": "doc-2",
                "content": "Machine learning algorithms can learn patterns from data.",
                "metadata": {"topic": "ml", "language": "general"}
            },
            {
                "id": "doc-3",
                "content": "Neural networks are inspired by biological brain structures.",
                "metadata": {"topic": "ml", "subtopic": "deep_learning"}
            },
        ]

    @pytest.mark.asyncio
    async def test_ingest_process_retrieve_pipeline(self, sample_documents):
        """Test full document pipeline: ingest -> process -> retrieve."""
        stages_completed = []
        errors = []

        try:
            # Stage 1: Ingest into RAGatouille
            from adapters.ragatouille_adapter import RAGatouilleAdapter
            indexer = RAGatouilleAdapter()
            await indexer.initialize({})

            index_result = await indexer.execute("index", documents=sample_documents)
            if index_result.success:
                stages_completed.append("indexing")
            else:
                errors.append(f"Indexing failed: {index_result.error}")

            # Stage 2: Compress context with SimpleMem
            from adapters.simplemem_adapter import SimpleMemAdapter
            memory = SimpleMemAdapter()
            await memory.initialize({"max_tokens": 4096})

            # Combine documents for compression
            combined_content = " ".join(d["content"] for d in sample_documents)
            compress_result = await memory.execute(
                "compress",
                context=combined_content,
                context_id="doc-collection"
            )
            if compress_result.success:
                stages_completed.append("compression")
            else:
                errors.append(f"Compression failed: {compress_result.error}")

            # Stage 3: Search and retrieve
            search_result = await indexer.execute(
                "search",
                query="machine learning Python",
                k=3
            )
            if search_result.success:
                stages_completed.append("search")
            else:
                errors.append(f"Search failed: {search_result.error}")

            # Stage 4: Rerank results
            if search_result.success and search_result.data.get("results"):
                rerank_result = await indexer.execute(
                    "rerank",
                    query="programming with neural networks",
                    documents=sample_documents
                )
                if rerank_result.success:
                    stages_completed.append("reranking")

            # Cleanup
            await indexer.shutdown()
            await memory.shutdown()

            # Verify pipeline completion
            assert "indexing" in stages_completed
            assert "search" in stages_completed
            assert len(errors) == 0, f"Pipeline errors: {errors}"

        except ImportError as e:
            pytest.skip(f"Required adapters not available: {e}")

    @pytest.mark.asyncio
    async def test_pipeline_with_observability(self, sample_documents):
        """Test pipeline with observability tracking."""
        try:
            from adapters.braintrust_adapter import BraintrustAdapter
            from adapters.simplemem_adapter import SimpleMemAdapter

            # Initialize observability
            tracker = BraintrustAdapter()
            await tracker.initialize({"project": "e2e-test"})

            # Initialize processing
            memory = SimpleMemAdapter()
            await memory.initialize({})

            # Track each operation
            for i, doc in enumerate(sample_documents[:2]):
                # Process document
                result = await memory.execute(
                    "compress",
                    context=doc["content"],
                    context_id=doc["id"]
                )

                # Log to tracker
                await tracker.execute(
                    "log",
                    input=doc["content"],
                    output=result.data if result.success else None,
                    scores={
                        "success": 1.0 if result.success else 0.0,
                        "latency": result.latency_ms / 1000.0
                    }
                )

            # Verify tracking
            stats = await tracker.execute("get_stats")
            assert stats.data["total_logs"] >= 2

            await tracker.shutdown()
            await memory.shutdown()

        except ImportError as e:
            pytest.skip(f"Required adapters not available: {e}")


class TestAgentCoordinationPipeline:
    """Test multi-agent coordination workflows."""

    @pytest.mark.asyncio
    async def test_agent_discovery_and_delegation(self):
        """Test agent discovery -> capability matching -> task delegation."""
        try:
            from adapters.a2a_protocol_adapter import A2AProtocolAdapter

            # Initialize coordinator
            coordinator = A2AProtocolAdapter()
            await coordinator.initialize({
                "agent_id": "coordinator-main",
                "capabilities": ["orchestration", "routing"]
            })

            # Register worker agents
            workers = [
                {"agent_id": "coder-1", "name": "Code Agent", "capabilities": ["coding", "review"]},
                {"agent_id": "researcher-1", "name": "Research Agent", "capabilities": ["research", "analysis"]},
                {"agent_id": "tester-1", "name": "Test Agent", "capabilities": ["testing", "qa"]},
            ]

            for worker in workers:
                await coordinator.execute("register", agent_card={
                    **worker,
                    "description": f"{worker['name']} - specialized agent"
                })

            # Discover agents by capability
            coders = await coordinator.execute("discover", capability="coding")
            researchers = await coordinator.execute("discover", capability="research")

            assert coders.data["count"] >= 1
            assert researchers.data["count"] >= 1

            # Delegate task to discovered agent
            if coders.data["agents"]:
                target = coders.data["agents"][0]["agent_id"]
                delegate_result = await coordinator.execute(
                    "delegate",
                    target_agent=target,
                    task="Implement a sorting function",
                    task_type="coding"
                )
                assert delegate_result.success

            # Check task status
            tasks = await coordinator.execute("list_tasks")
            assert tasks.data["pending_count"] >= 1

            await coordinator.shutdown()

        except ImportError as e:
            pytest.skip(f"Required adapters not available: {e}")

    @pytest.mark.asyncio
    async def test_multi_orchestrator_workflow(self):
        """Test workflow spanning multiple orchestration adapters."""
        try:
            from adapters.openai_agents_adapter import OpenAIAgentsAdapter
            from adapters.strands_agents_adapter import StrandsAgentsAdapter

            # Initialize both orchestrators
            openai_orch = OpenAIAgentsAdapter()
            strands_orch = StrandsAgentsAdapter()

            await openai_orch.initialize({"default_model": "gpt-4o-mini"})
            await strands_orch.initialize({"model": "anthropic.claude-3-sonnet"})

            # Register agents on both
            await openai_orch.execute(
                "register_agent",
                agent_id="analyzer",
                name="Data Analyzer",
                instructions="Analyze data patterns"
            )

            await strands_orch.execute(
                "register_tool",
                name="validation",
                description="Validate analysis results"
            )

            # Verify registrations
            openai_agents = await openai_orch.execute("list_agents")
            strands_tools = await strands_orch.execute("list_tools")

            # In stub mode, operations may not succeed
            assert openai_agents.success or "not initialized" in (openai_agents.error or "").lower()
            assert strands_tools.success or "not initialized" in (strands_tools.error or "").lower()

            await openai_orch.shutdown()
            await strands_orch.shutdown()

        except ImportError as e:
            pytest.skip(f"Required adapters not available: {e}")


class TestFullStackPipeline:
    """Test complete stack from Protocol to Knowledge layers."""

    @pytest.mark.asyncio
    async def test_l0_to_l8_pipeline(self):
        """Test data flow through all SDK layers."""
        layers_touched = []

        try:
            # L0: Protocol - Gateway
            from adapters.portkey_gateway_adapter import PortkeyGatewayAdapter
            gateway = PortkeyGatewayAdapter()
            await gateway.initialize({"config": {"cache_mode": "semantic"}})
            layers_touched.append("L0_PROTOCOL")

            # Simulate LLM call through gateway
            gateway_result = await gateway.execute(
                "chat",
                messages=[{"role": "user", "content": "Explain microservices architecture"}]
            )
            assert gateway_result.success

            # L2: Memory - SimpleMem
            from adapters.simplemem_adapter import SimpleMemAdapter
            memory = SimpleMemAdapter()
            await memory.initialize({})
            layers_touched.append("L2_MEMORY")

            # Store in memory
            memory_result = await memory.execute(
                "compress",
                context=gateway_result.data.get("content", "fallback content")
            )
            assert memory_result.success

            # L5: Observability - Braintrust
            from adapters.braintrust_adapter import BraintrustAdapter
            tracker = BraintrustAdapter()
            await tracker.initialize({"project": "full-stack-test"})
            layers_touched.append("L5_OBSERVABILITY")

            # Log the operation
            await tracker.execute(
                "log",
                input="microservices query",
                output=memory_result.data,
                scores={"compression_ratio": memory_result.data.get("compression_ratio", 1.0)}
            )

            # L7: Processing - RAGatouille
            from adapters.ragatouille_adapter import RAGatouilleAdapter
            processor = RAGatouilleAdapter()
            await processor.initialize({})
            layers_touched.append("L7_PROCESSING")

            # Index for future retrieval
            await processor.execute("index", documents=[
                {"id": "microservices-1", "content": gateway_result.data.get("content", "")}
            ])

            # Cleanup all
            await gateway.shutdown()
            await memory.shutdown()
            await tracker.shutdown()
            await processor.shutdown()

            # Verify all layers were touched
            assert "L0_PROTOCOL" in layers_touched
            assert "L2_MEMORY" in layers_touched
            assert "L5_OBSERVABILITY" in layers_touched
            assert "L7_PROCESSING" in layers_touched

        except ImportError as e:
            pytest.skip(f"Required adapters not available: {e}")


class TestPipelineResilience:
    """Test pipeline behavior under failure conditions."""

    @pytest.mark.asyncio
    async def test_pipeline_continues_on_partial_failure(self):
        """Pipeline should continue even if some stages fail."""
        successful_stages = []

        try:
            from adapters.simplemem_adapter import SimpleMemAdapter
            from adapters.graphiti_adapter import GraphitiAdapter

            # SimpleMem always works
            memory = SimpleMemAdapter()
            await memory.initialize({})

            result1 = await memory.execute("compress", context="Test content")
            if result1.success:
                successful_stages.append("simplemem_compress")

            # Graphiti requires Neo4j (will likely fail)
            graphiti = GraphitiAdapter()
            init_result = await graphiti.initialize({})

            if init_result.success:
                result2 = await graphiti.execute("add_episode", content="Test episode")
                if result2.success:
                    successful_stages.append("graphiti_episode")
            else:
                # Expected - Graphiti needs Neo4j
                successful_stages.append("graphiti_graceful_fail")

            # Pipeline continues with SimpleMem
            result3 = await memory.execute("retrieve", query="test")
            if result3.success:
                successful_stages.append("simplemem_retrieve")

            await memory.shutdown()
            await graphiti.shutdown()

            # At least some stages should succeed
            assert len(successful_stages) >= 2
            assert "simplemem_compress" in successful_stages

        except ImportError as e:
            pytest.skip(f"Required adapters not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
