#!/usr/bin/env python3
"""
V14 Iteration 45: REAL Service Integration Tests

Unlike most platform tests that check file patterns or mock behavior,
these tests make ACTUAL API calls to verify real service connectivity.

Services tested:
- Voyage AI (embedding API)
- Letta Cloud (memory/agent API)
- DSPy (module creation)
- LangGraph (workflow execution)
- MemoryMetrics (functional behavior)

HONEST: These are the first tests that actually call real services.
"""

import os
import sys
import pytest

# Add platform to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# =============================================================================
# REAL API Tests (require API keys)
# =============================================================================

class TestRealVoyageAI:
    """Test actual Voyage AI API connectivity."""

    @pytest.fixture(autouse=True)
    def check_voyage_key(self):
        if not os.environ.get("VOYAGE_API_KEY"):
            pytest.skip("VOYAGE_API_KEY not set")

    def test_voyage_embed_real_api(self):
        """Make a REAL Voyage AI API call and validate response."""
        import voyageai
        client = voyageai.Client()
        result = client.embed(["UNLEASH platform test embedding"], model="voyage-3")

        assert result.embeddings is not None
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 1024  # voyage-3 dim
        assert result.total_tokens > 0

    def test_voyage_batch_embed(self):
        """Test batch embedding with real API."""
        import voyageai
        client = voyageai.Client()
        texts = ["first document", "second document", "third document"]
        result = client.embed(texts, model="voyage-3")

        assert len(result.embeddings) == 3
        for emb in result.embeddings:
            assert len(emb) == 1024

    def test_voyage_rerank_real_api(self):
        """Test real Voyage reranking."""
        import voyageai
        client = voyageai.Client()
        result = client.rerank(
            query="What is machine learning?",
            documents=[
                "Machine learning is a subset of AI",
                "The weather is nice today",
                "Deep learning uses neural networks",
            ],
            model="rerank-2",
        )
        assert result.results is not None
        assert len(result.results) == 3
        # ML-related docs should rank higher
        top_doc_idx = result.results[0].index
        assert top_doc_idx in [0, 2]  # Either ML or DL doc


class TestRealLettaCloud:
    """Test actual Letta Cloud API connectivity."""

    @pytest.fixture(autouse=True)
    def check_letta_key(self):
        if not os.environ.get("LETTA_API_KEY"):
            pytest.skip("LETTA_API_KEY not set")

    def test_letta_list_agents(self):
        """List real agents from Letta Cloud."""
        from letta_client import Letta
        client = Letta(api_key=os.environ["LETTA_API_KEY"])
        agents = list(client.agents.list())

        assert len(agents) > 0
        # Verify agent structure
        for agent in agents:
            assert agent.id is not None
            assert agent.name is not None

    def test_letta_read_blocks(self):
        """Read memory blocks from a real agent."""
        from letta_client import Letta
        client = Letta(api_key=os.environ["LETTA_API_KEY"])
        agent_id = "agent-daee71d2-193b-485e-bda4-ee44752635fe"

        blocks = list(client.agents.blocks.list(agent_id=agent_id))
        assert len(blocks) > 0

        labels = [b.label for b in blocks]
        # UNLEASH agent should have these blocks
        assert "persona" in labels or "human" in labels

    def test_letta_search_passages(self):
        """Search archival memory with real API."""
        from letta_client import Letta
        client = Letta(api_key=os.environ["LETTA_API_KEY"])
        agent_id = "agent-daee71d2-193b-485e-bda4-ee44752635fe"

        results = client.agents.passages.search(
            agent_id=agent_id,
            query="iteration",
            top_k=3,
        )
        # Results structure validated
        assert results is not None


class TestRealDSPy:
    """Test actual DSPy functionality (not mocked)."""

    def test_dspy_create_module(self):
        """Create a real DSPy module."""
        import dspy

        class SimpleQA(dspy.Signature):
            """Answer questions."""
            question = dspy.InputField()
            answer = dspy.OutputField()

        predictor = dspy.Predict(SimpleQA)
        assert predictor is not None
        assert hasattr(predictor, "forward")

    def test_dspy_chain_of_thought(self):
        """Create a real ChainOfThought module."""
        import dspy

        class ReasonedQA(dspy.Signature):
            question = dspy.InputField()
            reasoning = dspy.OutputField()
            answer = dspy.OutputField()

        cot = dspy.ChainOfThought(ReasonedQA)
        assert cot is not None

    def test_dspy_adapter_creates_module(self):
        """Test the UNLEASH DSPy adapter creates real modules."""
        from adapters.dspy_adapter import DSPyAdapter, DSPY_AVAILABLE
        assert DSPY_AVAILABLE is True

        adapter = DSPyAdapter()
        sig = adapter.create_signature("SimpleTest", {"question": "input"}, {"answer": "output"})
        assert sig is not None


class TestRealLangGraph:
    """Test actual LangGraph workflow execution."""

    def test_langgraph_basic_graph(self):
        """Execute a real LangGraph graph."""
        from langgraph.graph import StateGraph
        from typing import TypedDict

        class State(TypedDict):
            value: str
            processed: bool

        def process(state):
            return {"value": state["value"].upper(), "processed": True}

        graph = StateGraph(State)
        graph.add_node("process", process)
        graph.set_entry_point("process")
        graph.set_finish_point("process")
        app = graph.compile()

        result = app.invoke({"value": "test", "processed": False})
        assert result["value"] == "TEST"
        assert result["processed"] is True

    def test_langgraph_multi_step(self):
        """Execute a multi-step LangGraph workflow."""
        from langgraph.graph import StateGraph
        from typing import TypedDict

        class State(TypedDict):
            steps: list

        def step_a(state):
            return {"steps": state["steps"] + ["a"]}

        def step_b(state):
            return {"steps": state["steps"] + ["b"]}

        graph = StateGraph(State)
        graph.add_node("a", step_a)
        graph.add_node("b", step_b)
        graph.add_edge("a", "b")
        graph.set_entry_point("a")
        graph.set_finish_point("b")
        app = graph.compile()

        result = app.invoke({"steps": []})
        assert result["steps"] == ["a", "b"]

    def test_langgraph_adapter_creates_graph(self):
        """Test the UNLEASH LangGraph adapter."""
        from adapters.langgraph_adapter import LangGraphAdapter, LANGGRAPH_AVAILABLE
        assert LANGGRAPH_AVAILABLE is True

        adapter = LangGraphAdapter()
        graph = adapter.create_graph("test")
        assert graph is not None


# =============================================================================
# REAL Functional Tests (no external API needed)
# =============================================================================

class TestRealMemoryMetrics:
    """Test MemoryMetrics with actual method calls, not string matching."""

    def test_metrics_record_and_retrieve(self):
        """Record real metrics and verify retrieval."""
        from core.advanced_memory import MemoryMetrics, get_memory_stats, reset_memory_metrics
        reset_memory_metrics()
        m = MemoryMetrics()

        # Record actual metrics
        m.record_embed_call("voyage", "voyage-3", False, 0.045, 100)
        m.record_embed_call("voyage", "voyage-3", True, 0.002, 0)
        m.record_embed_call("openai", "text-3-small", False, 0.1, 200)

        stats = get_memory_stats()
        assert stats["embedding"]["calls"] == 3
        assert stats["embedding"]["cache_hits"] == 1
        assert stats["embedding"]["cache_misses"] == 2
        assert stats["embedding"]["tokens_total"] == 300

    def test_metrics_error_tracking(self):
        """Test error recording and stats."""
        from core.advanced_memory import MemoryMetrics, get_memory_stats, reset_memory_metrics
        reset_memory_metrics()
        m = MemoryMetrics()

        m.record_embed_error("voyage", "voyage-3", "timeout")
        m.record_embed_error("voyage", "voyage-3", "rate_limit")

        stats = get_memory_stats()
        assert stats["embedding"]["errors"] == 2

    def test_metrics_latency_percentiles(self):
        """Test latency percentile calculation."""
        from core.advanced_memory import MemoryMetrics, get_memory_stats, reset_memory_metrics
        reset_memory_metrics()
        m = MemoryMetrics()

        # Record various latencies
        for latency in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
            m.record_embed_call("test", "model", False, latency, 10)

        stats = get_memory_stats()
        p50 = stats["embedding"]["latency_p50_ms"]
        p95 = stats["embedding"]["latency_p95_ms"]
        p99 = stats["embedding"]["latency_p99_ms"]

        assert p50 > 0
        assert p95 > p50
        assert p99 >= p95

    def test_metrics_search_tracking(self):
        """Test search call recording."""
        from core.advanced_memory import MemoryMetrics, get_memory_stats, reset_memory_metrics
        reset_memory_metrics()
        m = MemoryMetrics()

        m.record_search("default_index", 0.05)
        m.record_search("default_index", 0.03)

        stats = get_memory_stats()
        assert stats["search"]["calls"] == 2

    def test_metrics_reset_works(self):
        """Verify reset clears all counters."""
        from core.advanced_memory import MemoryMetrics, get_memory_stats, reset_memory_metrics

        m = MemoryMetrics()
        m.record_embed_call("test", "model", False, 0.01, 10)
        reset_memory_metrics()

        stats = get_memory_stats()
        assert stats["embedding"]["calls"] == 0
        assert stats["embedding"]["errors"] == 0


class TestRealAdapterAvailability:
    """Verify adapter availability flags match actual imports."""

    def test_dspy_available_matches_import(self):
        """DSPy adapter says available AND dspy actually imports."""
        from adapters.dspy_adapter import DSPY_AVAILABLE
        assert DSPY_AVAILABLE is True
        import dspy  # Actually importable
        assert dspy.__version__

    def test_langgraph_available_matches_import(self):
        """LangGraph adapter says available AND langgraph actually imports."""
        from adapters.langgraph_adapter import LANGGRAPH_AVAILABLE
        assert LANGGRAPH_AVAILABLE is True
        import langgraph  # Actually importable

    def test_mem0_available_matches_import(self):
        """Mem0 adapter says available AND mem0 actually imports."""
        from adapters.mem0_adapter import MEM0_AVAILABLE
        assert MEM0_AVAILABLE is True
        import mem0  # Actually importable

    def test_voyage_adapter_available(self):
        """Voyage adapter VOYAGE_AVAILABLE is True now that embedding_layer exists."""
        from adapters.dspy_voyage_retriever import VOYAGE_AVAILABLE
        assert VOYAGE_AVAILABLE is True, \
            "VOYAGE_AVAILABLE should be True (core.orchestration.embedding_layer created in V14 Iter 46)"

        from adapters.letta_voyage_adapter import VOYAGE_AVAILABLE as LV
        assert LV is True, "Letta-Voyage adapter should also be available"

        # And the raw SDK works
        import voyageai
        assert voyageai is not None


class TestRealEmbeddingLayer:
    """Test the core.orchestration.embedding_layer with REAL Voyage AI API."""

    @pytest.fixture(autouse=True)
    def check_voyage_key(self):
        if not os.environ.get("VOYAGE_API_KEY"):
            pytest.skip("VOYAGE_API_KEY not set")

    @pytest.mark.asyncio
    async def test_embedding_layer_real_api(self):
        """Create embedding layer and make real Voyage API call."""
        from core.orchestration.embedding_layer import create_embedding_layer, InputType

        layer = create_embedding_layer(model="voyage-3", cache_enabled=True)
        await layer.initialize()

        result = await layer.embed(["test embedding"], input_type=InputType.QUERY)
        assert len(result.embeddings) == 1
        assert len(result.embeddings[0]) == 1024
        assert result.total_tokens > 0
        assert result.model == "voyage-3"

    @pytest.mark.asyncio
    async def test_embedding_layer_cache(self):
        """Verify cache returns cached results on repeated calls."""
        from core.orchestration.embedding_layer import create_embedding_layer, InputType

        layer = create_embedding_layer(model="voyage-3", cache_enabled=True)
        await layer.initialize()

        # First call - real API
        r1 = await layer.embed(["cache test"], input_type=InputType.DOCUMENT)
        assert r1.cached is False

        # Second call - should be cached
        r2 = await layer.embed(["cache test"], input_type=InputType.DOCUMENT)
        assert r2.cached is True
        assert layer._cache_hits >= 1

    @pytest.mark.asyncio
    async def test_embedding_layer_batch(self):
        """Test batch embedding through the layer."""
        from core.orchestration.embedding_layer import create_embedding_layer, InputType

        layer = create_embedding_layer(model="voyage-3")
        await layer.initialize()

        texts = ["doc one", "doc two", "doc three"]
        result = await layer.embed(texts, input_type=InputType.DOCUMENT)
        assert len(result.embeddings) == 3


class TestRealOrchestratorIntegration:
    """Test that the orchestrator actually integrates components."""

    def test_orchestrator_initializes_real_sdks(self):
        """Verify orchestrator creates real SDK connections."""
        from core.ecosystem_orchestrator import get_orchestrator_v2
        o = get_orchestrator_v2()

        # These should be real adapter instances, not None
        assert o.has_dspy is True
        assert o.has_langgraph is True
        assert o.has_mem0 is True

    def test_orchestrator_v2_status_accurate(self):
        """Verify status report matches actual availability."""
        from core.ecosystem_orchestrator import get_orchestrator_v2
        o = get_orchestrator_v2()
        status = o.v2_status()

        assert "adapters" in status
        assert status["adapters"]["dspy"]["available"] is True
        assert status["adapters"]["langgraph"]["available"] is True
        assert status["adapters"]["mem0"]["available"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
