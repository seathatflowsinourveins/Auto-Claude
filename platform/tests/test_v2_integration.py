"""
V2 SDK Stack Integration Tests

Validates the complete adapter and pipeline architecture:
- Adapter availability and initialization
- Pipeline functionality
- Cross-component integration
- Error handling and graceful degradation

Run: pytest test_v2_integration.py -v
"""

import asyncio
import pytest
from typing import Dict, Any, List
from datetime import datetime
import sys
import os

# Add platform to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# ADAPTER AVAILABILITY TESTS
# =============================================================================

class TestAdapterAvailability:
    """Test that all adapters are importable and report correct status."""

    def test_adapters_module_imports(self):
        """Test that adapters module imports successfully."""
        from adapters import (
            ADAPTER_STATUS,
            get_adapter_status,
            get_dspy_adapter,
            get_langgraph_adapter,
            get_mem0_adapter,
            get_llm_reasoners_adapter,
        )
        assert callable(get_adapter_status)
        assert isinstance(ADAPTER_STATUS, dict)

    def test_adapter_status_tracking(self):
        """Test that adapter status is properly tracked."""
        from adapters import get_adapter_status

        status = get_adapter_status()

        # All expected adapters should be registered
        expected_adapters = ["dspy", "langgraph", "mem0", "llm_reasoners"]
        for adapter in expected_adapters:
            assert adapter in status, f"Adapter {adapter} not registered"
            assert "available" in status[adapter]
            assert "initialized" in status[adapter]

    def test_dspy_adapter_structure(self):
        """Test DSPy adapter has expected structure."""
        from adapters.dspy_adapter import (
            DSPyAdapter,
            DSPY_AVAILABLE,
            OptimizationResult,
            CompilationResult,
        )

        adapter = DSPyAdapter()

        # Check methods exist
        assert hasattr(adapter, "configure")
        assert hasattr(adapter, "create_signature")
        assert hasattr(adapter, "create_module")
        assert hasattr(adapter, "optimize")
        assert hasattr(adapter, "compile")
        assert hasattr(adapter, "get_status")

    def test_langgraph_adapter_structure(self):
        """Test LangGraph adapter has expected structure."""
        from adapters.langgraph_adapter import (
            LangGraphAdapter,
            LANGGRAPH_AVAILABLE,
            NodeType,
            WorkflowResult,
        )

        adapter = LangGraphAdapter()

        # Check methods exist
        assert hasattr(adapter, "create_graph")
        assert hasattr(adapter, "add_node")
        assert hasattr(adapter, "add_edge")
        assert hasattr(adapter, "add_conditional_edge")
        assert hasattr(adapter, "execute")
        assert hasattr(adapter, "get_status")

    def test_mem0_adapter_structure(self):
        """Test Mem0 adapter has expected structure."""
        from adapters.mem0_adapter import (
            Mem0Adapter,
            MEM0_AVAILABLE,
            MemoryEntry,
            SearchResult,
        )

        adapter = Mem0Adapter()

        # Check methods exist
        assert hasattr(adapter, "initialize")
        assert hasattr(adapter, "add")
        assert hasattr(adapter, "search")
        assert hasattr(adapter, "get")
        assert hasattr(adapter, "delete")
        assert hasattr(adapter, "get_status")

    def test_llm_reasoners_adapter_structure(self):
        """Test llm-reasoners adapter has expected structure."""
        from adapters.llm_reasoners_adapter import (
            LLMReasonersAdapter,
            LLM_REASONERS_AVAILABLE,
            ReasoningAlgorithm,
            ReasoningResult,
            ThoughtNode,
        )

        adapter = LLMReasonersAdapter()

        # Check methods exist
        assert hasattr(adapter, "reason")
        assert hasattr(adapter, "get_status")

        # Check enums
        assert hasattr(ReasoningAlgorithm, "MCTS")
        assert hasattr(ReasoningAlgorithm, "TOT")
        assert hasattr(ReasoningAlgorithm, "GOT")


# =============================================================================
# PIPELINE AVAILABILITY TESTS
# =============================================================================

class TestPipelineAvailability:
    """Test that all pipelines are importable and report correct status."""

    def test_pipelines_module_imports(self):
        """Test that pipelines module imports successfully."""
        from pipelines import (
            PIPELINE_STATUS,
            get_pipeline_status,
            get_deep_research_pipeline,
            get_self_improvement_pipeline,
        )
        assert callable(get_pipeline_status)
        assert isinstance(PIPELINE_STATUS, dict)

    def test_deep_research_pipeline_structure(self):
        """Test DeepResearchPipeline has expected structure."""
        from pipelines.deep_research_pipeline import (
            DeepResearchPipeline,
            PIPELINE_AVAILABLE,
            ResearchDepth,
            ResearchStrategy,
            ResearchResult,
            Source,
        )

        pipeline = DeepResearchPipeline()

        # Check methods exist
        assert hasattr(pipeline, "research")
        assert hasattr(pipeline, "get_status")

        # Check enums
        assert hasattr(ResearchDepth, "QUICK")
        assert hasattr(ResearchDepth, "COMPREHENSIVE")
        assert hasattr(ResearchStrategy, "SEMANTIC")
        assert hasattr(ResearchStrategy, "HYBRID")

    def test_self_improvement_pipeline_structure(self):
        """Test SelfImprovementPipeline has expected structure."""
        from pipelines.self_improvement_pipeline import (
            SelfImprovementPipeline,
            PIPELINE_AVAILABLE,
            ImprovementStrategy,
            ImprovementResult,
            Workflow,
            WorkflowStep,
        )

        pipeline = SelfImprovementPipeline()

        # Check methods exist
        assert hasattr(pipeline, "improve")
        assert hasattr(pipeline, "get_status")

        # Check enums
        assert hasattr(ImprovementStrategy, "GENETIC")
        assert hasattr(ImprovementStrategy, "GRADIENT")
        assert hasattr(ImprovementStrategy, "HYBRID")


# =============================================================================
# ECOSYSTEM ORCHESTRATOR V2 TESTS
# =============================================================================

class TestEcosystemOrchestratorV2:
    """Test EcosystemOrchestratorV2 integration."""

    def test_orchestrator_v2_imports(self):
        """Test that orchestrator V2 imports successfully."""
        from core.ecosystem_orchestrator import (
            EcosystemOrchestratorV2,
            get_orchestrator_v2,
            ecosystem_v2,
        )
        assert callable(get_orchestrator_v2)
        assert callable(ecosystem_v2)

    def test_orchestrator_v2_initialization(self):
        """Test orchestrator V2 initializes without errors."""
        from core.ecosystem_orchestrator import get_orchestrator_v2

        orchestrator = get_orchestrator_v2()

        # Check V2-specific attributes
        assert hasattr(orchestrator, "_dspy_adapter")
        assert hasattr(orchestrator, "_langgraph_adapter")
        assert hasattr(orchestrator, "_mem0_adapter")
        assert hasattr(orchestrator, "_llm_reasoners_adapter")
        assert hasattr(orchestrator, "_deep_research_pipeline")
        assert hasattr(orchestrator, "_self_improvement_pipeline")

    def test_orchestrator_v2_status(self):
        """Test orchestrator V2 reports status correctly."""
        from core.ecosystem_orchestrator import get_orchestrator_v2

        orchestrator = get_orchestrator_v2()
        status = orchestrator.v2_status()

        # Check status structure
        assert "adapters" in status
        assert "pipelines" in status
        assert "total_v2_components" in status
        assert "available_v2_components" in status

        # Check adapter status
        expected_adapters = ["dspy", "langgraph", "mem0", "llm_reasoners"]
        for adapter in expected_adapters:
            assert adapter in status["adapters"]

    def test_orchestrator_v2_has_properties(self):
        """Test orchestrator V2 has availability properties."""
        from core.ecosystem_orchestrator import get_orchestrator_v2

        orchestrator = get_orchestrator_v2()

        # These should return booleans
        assert isinstance(orchestrator.has_dspy, bool)
        assert isinstance(orchestrator.has_langgraph, bool)
        assert isinstance(orchestrator.has_mem0, bool)
        assert isinstance(orchestrator.has_llm_reasoners, bool)

    def test_orchestrator_v2_methods_exist(self):
        """Test orchestrator V2 has all expected methods."""
        from core.ecosystem_orchestrator import get_orchestrator_v2

        orchestrator = get_orchestrator_v2()

        # Sync methods
        assert hasattr(orchestrator, "optimize_prompt")
        assert hasattr(orchestrator, "create_workflow")
        assert hasattr(orchestrator, "remember")
        assert hasattr(orchestrator, "recall")

        # Async methods
        assert hasattr(orchestrator, "reason_v2")
        assert hasattr(orchestrator, "deep_research_v2")
        assert hasattr(orchestrator, "improve_workflow")


# =============================================================================
# DATA CLASS TESTS
# =============================================================================

class TestDataClasses:
    """Test that data classes are properly defined."""

    def test_reasoning_result_dataclass(self):
        """Test ReasoningResult dataclass."""
        from adapters.llm_reasoners_adapter import ReasoningResult, ThoughtNode

        node = ThoughtNode(
            id="test",
            content="Test content",
            parent_id=None,
            children=[],
            score=0.5,
            depth=0,
            metadata={},
        )

        result = ReasoningResult(
            answer="Test answer",
            confidence=0.8,
            reasoning_path=[node],
            algorithm_used="mcts",
            total_nodes_explored=1,
            execution_time=0.1,
            metadata={},
        )

        assert result.answer == "Test answer"
        assert result.confidence == 0.8
        assert len(result.reasoning_path) == 1

    def test_memory_entry_dataclass(self):
        """Test MemoryEntry dataclass."""
        from adapters.mem0_adapter import MemoryEntry

        entry = MemoryEntry(
            id="test-id",
            content="Test content",
            metadata={"key": "value"},
            created_at=datetime.now(),
            user_id="user-1",
            agent_id=None,
            run_id=None,
        )

        assert entry.id == "test-id"
        assert entry.content == "Test content"
        assert entry.user_id == "user-1"

    def test_research_result_dataclass(self):
        """Test ResearchResult dataclass."""
        from pipelines.deep_research_pipeline import ResearchResult, ResearchDepth

        result = ResearchResult(
            query="Test query",
            sources=[],
            synthesis="Test synthesis",
            confidence=0.9,
            reasoning_path=None,
            total_sources=0,
            depth=ResearchDepth.STANDARD,
            execution_time=0.5,
            metadata={},
        )

        assert result.query == "Test query"
        assert result.confidence == 0.9
        assert result.depth == ResearchDepth.STANDARD


# =============================================================================
# GRACEFUL DEGRADATION TESTS
# =============================================================================

class TestGracefulDegradation:
    """Test that components degrade gracefully when SDKs unavailable."""

    def test_adapters_handle_missing_dependencies(self):
        """Test adapters work even when dependencies missing."""
        from adapters.dspy_adapter import DSPyAdapter, DSPY_AVAILABLE
        from adapters.langgraph_adapter import LangGraphAdapter, LANGGRAPH_AVAILABLE
        from adapters.mem0_adapter import Mem0Adapter, MEM0_AVAILABLE
        from adapters.llm_reasoners_adapter import LLMReasonersAdapter, LLM_REASONERS_AVAILABLE

        # All adapters should instantiate without errors
        dspy = DSPyAdapter()
        langgraph = LangGraphAdapter()
        mem0 = Mem0Adapter()
        llm_reasoners = LLMReasonersAdapter()

        # Status should reflect availability
        assert dspy.get_status()["available"] == DSPY_AVAILABLE
        assert langgraph.get_status()["available"] == LANGGRAPH_AVAILABLE
        assert mem0.get_status()["available"] == MEM0_AVAILABLE
        assert llm_reasoners.get_status()["available"] == LLM_REASONERS_AVAILABLE

    def test_pipelines_handle_missing_components(self):
        """Test pipelines work even when components missing."""
        from pipelines.deep_research_pipeline import DeepResearchPipeline
        from pipelines.self_improvement_pipeline import SelfImprovementPipeline

        # Pipelines should instantiate without errors
        research = DeepResearchPipeline()
        improvement = SelfImprovementPipeline()

        # Status should reflect component availability
        research_status = research.get_status()
        assert "exa" in research_status
        assert "firecrawl" in research_status
        assert "crawler" in research_status

    def test_orchestrator_v2_handles_missing_adapters(self):
        """Test orchestrator V2 works even when adapters missing."""
        from core.ecosystem_orchestrator import get_orchestrator_v2

        orchestrator = get_orchestrator_v2()

        # Should not raise errors
        status = orchestrator.v2_status()

        # Should report correct availability
        assert "available_v2_components" in status
        assert isinstance(status["available_v2_components"], int)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Test integration between components."""

    def test_orchestrator_v2_extends_v1(self):
        """Test that V2 orchestrator extends V1 functionality."""
        from core.ecosystem_orchestrator import (
            EcosystemOrchestrator,
            EcosystemOrchestratorV2,
        )

        assert issubclass(EcosystemOrchestratorV2, EcosystemOrchestrator)

    def test_adapter_registration(self):
        """Test that all adapters are properly registered."""
        from adapters import get_adapter_status

        status = get_adapter_status()

        # Should have all adapters
        assert len(status) >= 4

        # Each adapter should have required fields
        for name, info in status.items():
            assert "available" in info
            assert "initialized" in info

    def test_pipeline_registration(self):
        """Test that all pipelines are properly registered."""
        from pipelines import get_pipeline_status

        status = get_pipeline_status()

        # Should have registered pipelines
        assert len(status) >= 2

        # Each pipeline should have required fields
        for name, info in status.items():
            assert "available" in info
            assert "dependencies" in info

    def test_full_stack_status(self):
        """Test getting status of the full V2 stack."""
        from core.ecosystem_orchestrator import get_orchestrator_v2
        from adapters import get_adapter_status
        from pipelines import get_pipeline_status

        orchestrator = get_orchestrator_v2()

        # Get all statuses
        adapter_status = get_adapter_status()
        pipeline_status = get_pipeline_status()
        orchestrator_status = orchestrator.v2_status()

        # All should be consistent
        for adapter in ["dspy", "langgraph", "mem0", "llm_reasoners"]:
            adapter_avail = adapter_status.get(adapter, {}).get("available", False)
            orch_avail = orchestrator_status["adapters"].get(adapter, False)
            # Both should agree on availability
            assert adapter_avail == orch_avail, f"Mismatch for {adapter}"


# =============================================================================
# ASYNC TESTS
# =============================================================================

class TestAsyncOperations:
    """Test async operations work correctly."""

    @pytest.mark.asyncio
    async def test_reason_v2_returns_result(self):
        """Test reason_v2 returns a result structure."""
        from core.ecosystem_orchestrator import get_orchestrator_v2

        orchestrator = get_orchestrator_v2()

        result = await orchestrator.reason_v2(
            problem="What is 2 + 2?",
            algorithm="mcts",
            max_depth=3,
        )

        # Should return dict with expected structure
        assert isinstance(result, dict)
        assert "answer" in result or "error" in result

    @pytest.mark.asyncio
    async def test_deep_research_v2_returns_result(self):
        """Test deep_research_v2 returns a result structure."""
        from core.ecosystem_orchestrator import get_orchestrator_v2

        orchestrator = get_orchestrator_v2()

        result = await orchestrator.deep_research_v2(
            query="Python async patterns",
            depth="quick",
        )

        # Should return dict with expected structure
        assert isinstance(result, dict)
        # Will have sources or error depending on API availability
        assert "sources" in result or "error" in result


# =============================================================================
# VALIDATION REPORT
# =============================================================================

def generate_validation_report() -> Dict[str, Any]:
    """Generate a comprehensive validation report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "adapters": {},
        "pipelines": {},
        "orchestrator": {},
        "summary": {},
    }

    # Test adapters
    adapter_tests = [
        ("dspy", "adapters.dspy_adapter", "DSPyAdapter", "DSPY_AVAILABLE"),
        ("langgraph", "adapters.langgraph_adapter", "LangGraphAdapter", "LANGGRAPH_AVAILABLE"),
        ("mem0", "adapters.mem0_adapter", "Mem0Adapter", "MEM0_AVAILABLE"),
        ("llm_reasoners", "adapters.llm_reasoners_adapter", "LLMReasonersAdapter", "LLM_REASONERS_AVAILABLE"),
    ]

    for name, module_path, class_name, avail_flag in adapter_tests:
        try:
            module = __import__(module_path, fromlist=[class_name, avail_flag])
            adapter_class = getattr(module, class_name)
            available = getattr(module, avail_flag)
            adapter = adapter_class()
            status = adapter.get_status()

            report["adapters"][name] = {
                "importable": True,
                "available": available,
                "status": status,
                "error": None,
            }
        except Exception as e:
            report["adapters"][name] = {
                "importable": False,
                "available": False,
                "status": None,
                "error": str(e),
            }

    # Test pipelines
    pipeline_tests = [
        ("deep_research", "pipelines.deep_research_pipeline", "DeepResearchPipeline"),
        ("self_improvement", "pipelines.self_improvement_pipeline", "SelfImprovementPipeline"),
    ]

    for name, module_path, class_name in pipeline_tests:
        try:
            module = __import__(module_path, fromlist=[class_name])
            pipeline_class = getattr(module, class_name)
            pipeline = pipeline_class()
            status = pipeline.get_status()

            report["pipelines"][name] = {
                "importable": True,
                "status": status,
                "error": None,
            }
        except Exception as e:
            report["pipelines"][name] = {
                "importable": False,
                "status": None,
                "error": str(e),
            }

    # Test orchestrator V2
    try:
        from core.ecosystem_orchestrator import get_orchestrator_v2
        orchestrator = get_orchestrator_v2()
        v2_status = orchestrator.v2_status()

        report["orchestrator"] = {
            "importable": True,
            "status": v2_status,
            "error": None,
        }
    except Exception as e:
        report["orchestrator"] = {
            "importable": False,
            "status": None,
            "error": str(e),
        }

    # Summary
    total_adapters = len(adapter_tests)
    available_adapters = sum(
        1 for a in report["adapters"].values()
        if a.get("importable") and a.get("available")
    )

    total_pipelines = len(pipeline_tests)
    available_pipelines = sum(
        1 for p in report["pipelines"].values()
        if p.get("importable")
    )

    report["summary"] = {
        "total_adapters": total_adapters,
        "available_adapters": available_adapters,
        "total_pipelines": total_pipelines,
        "available_pipelines": available_pipelines,
        "orchestrator_v2_ready": report["orchestrator"].get("importable", False),
        "overall_status": "READY" if report["orchestrator"].get("importable") else "DEGRADED",
    }

    return report


if __name__ == "__main__":
    # Run validation and print report
    print("=" * 70)
    print("V2 SDK STACK VALIDATION REPORT")
    print("=" * 70)

    report = generate_validation_report()

    print(f"\nTimestamp: {report['timestamp']}")

    print("\n--- ADAPTERS ---")
    for name, info in report["adapters"].items():
        status = "✓" if info["importable"] and info["available"] else "○" if info["importable"] else "✗"
        print(f"  {status} {name}: {'available' if info['available'] else 'not available'}")
        if info["error"]:
            print(f"      Error: {info['error']}")

    print("\n--- PIPELINES ---")
    for name, info in report["pipelines"].items():
        status = "✓" if info["importable"] else "✗"
        print(f"  {status} {name}")
        if info["error"]:
            print(f"      Error: {info['error']}")

    print("\n--- ORCHESTRATOR V2 ---")
    orch = report["orchestrator"]
    status = "✓" if orch["importable"] else "✗"
    print(f"  {status} EcosystemOrchestratorV2")
    if orch["error"]:
        print(f"      Error: {orch['error']}")

    print("\n--- SUMMARY ---")
    summary = report["summary"]
    print(f"  Adapters: {summary['available_adapters']}/{summary['total_adapters']} available")
    print(f"  Pipelines: {summary['available_pipelines']}/{summary['total_pipelines']} importable")
    print(f"  Orchestrator V2: {'Ready' if summary['orchestrator_v2_ready'] else 'Not Ready'}")
    print(f"\n  OVERALL STATUS: {summary['overall_status']}")

    print("\n" + "=" * 70)
