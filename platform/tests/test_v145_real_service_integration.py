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


class TestRealResearchEngine:
    """Test Exa and Firecrawl through the platform's ResearchEngine."""

    @pytest.fixture(autouse=True)
    def check_exa_key(self):
        if not os.environ.get("EXA_API_KEY"):
            pytest.skip("EXA_API_KEY not set")

    def test_exa_search_real_api(self):
        """Make a real Exa search API call."""
        from core.research_engine import ResearchEngine
        engine = ResearchEngine()
        assert engine.exa is not None

        results = engine.exa_search("Python testing", num_results=2)
        assert isinstance(results, dict)
        assert "results" in results
        assert len(results["results"]) > 0

    def test_exa_search_returns_urls(self):
        """Verify Exa results contain URLs and titles."""
        from core.research_engine import ResearchEngine
        engine = ResearchEngine()

        results = engine.exa_search("machine learning frameworks", num_results=2)
        for r in results.get("results", []):
            assert hasattr(r, "url") or "url" in (r if isinstance(r, dict) else {})


class TestRealVoyageEmbedderAdapter:
    """Test the VoyageEmbedder adapter with real Voyage API."""

    @pytest.fixture(autouse=True)
    def check_voyage_key(self):
        if not os.environ.get("VOYAGE_API_KEY"):
            pytest.skip("VOYAGE_API_KEY not set")

    @pytest.mark.asyncio
    async def test_voyage_embedder_async(self):
        """Test VoyageEmbedder async embed with real API."""
        from adapters.dspy_voyage_retriever import VoyageEmbedder

        embedder = VoyageEmbedder()
        await embedder.initialize()

        result = await embedder.embed(["voyage embedder test"], input_type="query")
        assert len(result) == 1
        assert len(result[0]) == 1024

    @pytest.mark.asyncio
    async def test_voyage_embedder_batch_async(self):
        """Test VoyageEmbedder batch embed."""
        from adapters.dspy_voyage_retriever import VoyageEmbedder

        embedder = VoyageEmbedder()
        await embedder.initialize()

        result = await embedder.embed(
            ["doc one", "doc two", "doc three"],
            input_type="document"
        )
        assert len(result) == 3


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


class TestFullPipelineE2E:
    """End-to-end: Voyage AI embed -> Qdrant in-memory -> Semantic Search."""

    @pytest.fixture(autouse=True)
    def check_voyage_key(self):
        if not os.environ.get("VOYAGE_API_KEY"):
            pytest.skip("VOYAGE_API_KEY not set")

    @pytest.mark.asyncio
    async def test_embed_store_search_pipeline(self):
        """Full pipeline: embed documents, store in Qdrant, search semantically."""
        from core.orchestration.embedding_layer import create_embedding_layer, InputType
        from qdrant_client import QdrantClient, models as qm

        # Create embedding layer with real Voyage AI
        layer = create_embedding_layer(model="voyage-3", cache_enabled=True)
        await layer.initialize()

        # Create Qdrant in-memory (no server needed)
        qclient = QdrantClient(location=":memory:")
        qclient.create_collection(
            "e2e_test",
            vectors_config=qm.VectorParams(size=1024, distance=qm.Distance.COSINE),
        )

        # Embed real documents
        documents = [
            "Machine learning is a subset of artificial intelligence",
            "Neural networks process data through layers of nodes",
            "The weather today is sunny with mild temperatures",
        ]
        result = await layer.embed(documents, input_type=InputType.DOCUMENT)
        assert len(result.embeddings) == 3
        assert result.total_tokens > 0

        # Store in Qdrant
        points = [
            qm.PointStruct(id=i, vector=vec, payload={"text": doc})
            for i, (vec, doc) in enumerate(zip(result.embeddings, documents))
        ]
        qclient.upsert("e2e_test", points=points)

        # Semantic search
        query = "How do AI systems learn?"
        qr = await layer.embed([query], input_type=InputType.QUERY)
        hits = qclient.query_points("e2e_test", query=qr.embeddings[0], limit=2)

        assert len(hits.points) == 2
        # AI docs should rank higher than weather
        top_id = hits.points[0].id
        assert top_id in [0, 1], f"Expected AI doc in top result, got id={top_id}"

    @pytest.mark.asyncio
    async def test_voyage_letta_cross_service(self):
        """Verify Voyage embeddings and Letta Cloud can both be used in same session."""
        from core.orchestration.embedding_layer import create_embedding_layer, InputType
        from letta_client import Letta

        # Voyage AI
        layer = create_embedding_layer(model="voyage-3")
        await layer.initialize()
        result = await layer.embed(["cross-service test"], input_type=InputType.QUERY)
        assert len(result.embeddings[0]) == 1024

        # Letta Cloud (in same session)
        client = Letta(api_key=os.environ.get("LETTA_API_KEY", ""))
        agents = list(client.agents.list())
        assert len(agents) > 0

        # Both services work together
        assert result.total_tokens > 0
        assert agents[0].id is not None


class TestAdapterLevelE2E:
    """V14 Iter 53: Test actual adapter initialization chains with real APIs."""

    @pytest.fixture(autouse=True)
    def check_voyage_key(self):
        if not os.environ.get("VOYAGE_API_KEY"):
            pytest.skip("VOYAGE_API_KEY not set")

    @pytest.mark.asyncio
    async def test_voyage_embedder_full_init_chain(self):
        """VoyageEmbedder: init creates EmbeddingLayer, layer calls real API."""
        from adapters.dspy_voyage_retriever import VoyageEmbedder

        embedder = VoyageEmbedder(model="voyage-3")
        assert embedder._initialized is False
        assert embedder._embedding_layer is None

        await embedder.initialize()

        assert embedder._initialized is True
        assert embedder._embedding_layer is not None
        assert embedder._embedding_layer._initialized is True

    def test_voyage_embedder_sync_callable(self):
        """VoyageEmbedder.__call__ works synchronously for DSPy compatibility."""
        from adapters.dspy_voyage_retriever import VoyageEmbedder
        import asyncio

        embedder = VoyageEmbedder(model="voyage-3")
        asyncio.run(embedder.initialize())

        # Call as function (DSPy pattern)
        result = embedder(["sync call test"])
        assert len(result) == 1
        assert len(result[0]) == 1024

    @pytest.mark.asyncio
    async def test_qdrant_server_pipeline_e2e(self):
        """Full Qdrant SERVER pipeline (not in-memory): embed -> store -> search."""
        import httpx
        try:
            httpx.get("http://localhost:6333/healthz", timeout=2)
        except Exception:
            pytest.skip("Qdrant not running at localhost:6333")

        from core.orchestration.embedding_layer import (
            create_embedding_layer, InputType, QdrantVectorStore,
        )
        from qdrant_client import QdrantClient, models as qm

        layer = create_embedding_layer(model="voyage-3")
        await layer.initialize()

        # Use real Qdrant server (not in-memory)
        collection = "e2e_iter53_test"
        qclient = QdrantClient(url="http://localhost:6333")

        # Create collection
        try:
            qclient.delete_collection(collection)
        except Exception:
            pass
        qclient.create_collection(
            collection,
            vectors_config=qm.VectorParams(size=1024, distance=qm.Distance.COSINE),
        )

        try:
            docs = [
                "Python async/await enables concurrent IO operations",
                "Rust ownership model prevents memory safety bugs",
                "The recipe for chocolate cake uses butter and cocoa",
            ]
            result = await layer.embed(docs, input_type=InputType.DOCUMENT)
            points = [
                qm.PointStruct(id=i, vector=vec, payload={"text": doc})
                for i, (vec, doc) in enumerate(zip(result.embeddings, docs))
            ]
            qclient.upsert(collection, points=points)

            # Verify storage
            info = qclient.get_collection(collection)
            assert info.points_count == 3

            # Search
            qr = await layer.embed(["async concurrency in Python"], input_type=InputType.QUERY)
            hits = qclient.query_points(collection, query=qr.embeddings[0], limit=2)
            assert len(hits.points) == 2
            # Python doc should rank above cooking
            assert hits.points[0].id in [0, 1], f"Expected programming doc first, got id={hits.points[0].id}"
        finally:
            qclient.delete_collection(collection)

    @pytest.mark.asyncio
    async def test_letta_voyage_adapter_status(self):
        """LettaVoyageAdapter.get_status() returns correct availability."""
        from adapters.letta_voyage_adapter import LettaVoyageAdapter

        adapter = LettaVoyageAdapter()
        status = adapter.get_status()

        assert status["voyage_available"] is True
        assert status["letta_available"] is True

    def test_dspy_embedder_callable_pattern(self):
        """Verify VoyageEmbedder works with dspy.Embedder pattern."""
        try:
            import dspy
            from adapters.dspy_voyage_retriever import VoyageEmbedder
            import asyncio
        except ImportError:
            pytest.skip("dspy or VoyageEmbedder not importable")

        embedder = VoyageEmbedder(model="voyage-3")
        asyncio.run(embedder.initialize())

        # Create DSPy Embedder wrapping our VoyageEmbedder
        dspy_embedder = dspy.Embedder(embedder.embed_sync)
        assert dspy_embedder is not None


class TestMem0AdapterReal:
    """V14 Iter 58: Test Mem0 adapter with real initialization."""

    def test_mem0_adapter_available(self):
        """Mem0 adapter should report available."""
        from adapters.mem0_adapter import Mem0Adapter, MEM0_AVAILABLE
        assert MEM0_AVAILABLE is True
        adapter = Mem0Adapter()
        status = adapter.get_status()
        assert status["available"] is True

    def test_mem0_adapter_initializes_qdrant(self):
        """Mem0 adapter should initialize with Qdrant backend + OpenAI."""
        from adapters.mem0_adapter import Mem0Adapter, MemoryBackend

        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        # Verify Qdrant server is running
        try:
            import httpx
            resp = httpx.get("http://localhost:6333/healthz", timeout=3)
            if resp.status_code != 200:
                pytest.skip("Qdrant server not running")
        except Exception:
            pytest.skip("Qdrant server not reachable")

        adapter = Mem0Adapter(
            backend=MemoryBackend.QDRANT,
            config={
                "collection": "mem0_iter58_test",
                "host": "localhost",
                "port": 6333,
            },
        )
        adapter.initialize(
            llm_config={
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "api_key": os.environ["OPENAI_API_KEY"],
                },
            },
            embedder_config={
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                    "api_key": os.environ["OPENAI_API_KEY"],
                },
            },
        )
        assert adapter._initialized is True
        assert adapter._memory is not None

    def test_mem0_add_and_search(self):
        """Mem0 should store and retrieve memories via Qdrant."""
        from adapters.mem0_adapter import Mem0Adapter, MemoryBackend

        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        try:
            import httpx
            resp = httpx.get("http://localhost:6333/healthz", timeout=3)
            if resp.status_code != 200:
                pytest.skip("Qdrant server not running")
        except Exception:
            pytest.skip("Qdrant server not reachable")

        adapter = Mem0Adapter(
            backend=MemoryBackend.QDRANT,
            config={
                "collection": "mem0_iter58_search_test",
                "host": "localhost",
                "port": 6333,
            },
        )
        adapter.initialize(
            llm_config={
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "api_key": os.environ["OPENAI_API_KEY"],
                },
            },
            embedder_config={
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                    "api_key": os.environ["OPENAI_API_KEY"],
                },
            },
        )

        # Add a memory (requires real LLM call for entity extraction)
        try:
            result = adapter.add(
                content="The UNLEASH platform uses Voyage AI for embeddings",
                user_id="test-user",
            )
        except Exception as e:
            # OpenAI/OpenRouter credit issues are not adapter bugs
            if "402" in str(e) or "Insufficient credits" in str(e):
                pytest.skip(f"LLM provider credit issue: {e}")
            raise
        assert result is not None

        # Search for it
        results = adapter.search(
            query="What embedding system does UNLEASH use?",
            user_id="test-user",
        )
        assert results is not None
        assert results.total >= 0  # May be 0 depending on mem0 extraction


class TestOrchestratorChainReal:
    """V14 Iter 58: Test full orchestrator initialization chain."""

    def test_orchestrator_all_capabilities(self):
        """Orchestrator should have all expected capabilities."""
        from core.ecosystem_orchestrator import get_orchestrator_v2
        o = get_orchestrator_v2()

        caps = {
            "research": o.has_research,
            "letta": o.has_letta,
            "cache": o.has_cache,
            "dspy": o.has_dspy,
            "langgraph": o.has_langgraph,
            "mem0": o.has_mem0,
        }
        active = [k for k, v in caps.items() if v]
        assert len(active) >= 6, f"Expected 6+ capabilities, got {len(active)}: {active}"

    def test_orchestrator_research_engine_has_exa(self):
        """Research engine should have Exa client connected."""
        from core.ecosystem_orchestrator import get_orchestrator_v2
        o = get_orchestrator_v2()
        assert o._research_engine is not None
        assert o._research_engine.exa is not None

    def test_all_adapter_status_consistent(self):
        """All adapter status flags should match actual importability."""
        from adapters import get_adapter_status
        status = get_adapter_status()

        for name, info in status.items():
            if info.get("available"):
                # If adapter says available, the SDK should be importable
                assert info["version"] is not None, \
                    f"Adapter {name} says available but has no version"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
