"""
V14 Iteration Verification Tests

Purpose: Validate actual system state for V14 iteration loop.
Tests what IS working, not what SHOULD be working.
Run: pytest tests/v14_iteration_tests.py -v
"""

import json
import os
import sys
from pathlib import Path

import pytest

BASE = Path("Z:/insider/AUTO CLAUDE/unleash")


# ============================================================================
# SECTION 1: Core Infrastructure
# ============================================================================

class TestGitRepository:
    """Git repo must be initialized and clean."""

    def test_git_initialized(self):
        assert (BASE / ".git").exists(), "Git not initialized"

    def test_gitignore_exists(self):
        assert (BASE / ".gitignore").exists()

    def test_iteration_state_exists(self):
        path = BASE / "iteration-state.json"
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["version"] == "14.0"
        assert data["iteration"] >= 29


class TestIterationLoop:
    """V14 iteration loop architecture must exist."""

    def test_loop_document_exists(self):
        path = BASE / "active" / "ITERATION_LOOP_V14.md"
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "OBSERVE" in content
        assert "ORIENT" in content
        assert "DECIDE" in content
        assert "ACT" in content
        assert "LEARN" in content
        assert "VERIFY" in content

    def test_loop_has_letta_patterns(self):
        path = BASE / "active" / "ITERATION_LOOP_V14.md"
        content = path.read_text(encoding="utf-8")
        assert "Letta" in content
        assert "blocks.update" in content

    def test_loop_has_opik_patterns(self):
        path = BASE / "active" / "ITERATION_LOOP_V14.md"
        content = path.read_text(encoding="utf-8")
        assert "opik" in content.lower()
        assert "@track" in content or "track" in content


# ============================================================================
# SECTION 2: Core Module Imports
# ============================================================================

class TestCoreModuleImports:
    """All core platform modules must import without error."""

    @pytest.fixture(autouse=True)
    def setup_path(self):
        platform_path = str(BASE / "platform")
        if platform_path not in sys.path:
            sys.path.insert(0, platform_path)

    def test_ralph_loop_imports(self):
        import core.ralph_loop
        assert hasattr(core.ralph_loop, "RalphLoop")

    def test_learning_imports(self):
        import core.learning
        assert hasattr(core.learning, "LearningEngine")

    def test_self_improvement_imports(self):
        import core.self_improvement

    def test_cross_session_memory_imports(self):
        import core.cross_session_memory

    def test_memory_tiers_imports(self):
        import core.memory_tiers

    def test_unified_confidence_imports(self):
        import core.unified_confidence


# ============================================================================
# SECTION 3: SDK Directories (Actual State)
# ============================================================================

class TestSDKDirectories:
    """Verify SDKs exist across all 3 locations."""

    SDK_LOCATIONS = [
        BASE / "sdks",
        BASE / "platform" / "sdks",
        BASE / "stack",
    ]

    def _find_sdk(self, name: str) -> bool:
        """Check if SDK exists in any location (including stack subdirs)."""
        for loc in self.SDK_LOCATIONS:
            if (loc / name).exists():
                return True
            # Check stack tier subdirectories
            if loc.name == "stack":
                for tier in loc.iterdir():
                    if tier.is_dir() and (tier / name).exists():
                        return True
        return False

    @pytest.mark.parametrize("sdk_name", [
        # P0 Critical (must exist)
        "anthropic", "claude-flow", "letta", "opik", "mcp-python-sdk",
        "instructor", "litellm", "langgraph",
        # P1 Important
        "dspy", "pydantic-ai", "pyribs", "promptfoo", "mem0",
        "ragas", "deepeval", "guardrails-ai", "crawl4ai", "ast-grep",
        # P2 Specialized
        "graphrag", "outlines", "baml", "langfuse", "zep",
        "llm-guard", "nemo-guardrails", "arize-phoenix",
    ])
    def test_p0_p1_p2_sdk_exists(self, sdk_name: str):
        assert self._find_sdk(sdk_name), (
            f"SDK {sdk_name} not found in sdks/, platform/sdks/, or stack/"
        )

    def test_sdk_count_minimum(self):
        """Must have at least 28 unique SDKs across all locations."""
        all_sdks = set()
        for loc in [BASE / "sdks", BASE / "platform" / "sdks"]:
            if loc.exists():
                for d in loc.iterdir():
                    if d.is_dir():
                        all_sdks.add(d.name)
        assert len(all_sdks) >= 28, f"Found {len(all_sdks)} SDKs, need 28+"


# ============================================================================
# SECTION 4: Cross-Session Memory
# ============================================================================

class TestCrossSessionMemory:
    """Verify cross-session memory infrastructure."""

    def test_bootstrap_document_exists(self):
        path = BASE / "CROSS_SESSION_BOOTSTRAP_V40.md"
        assert path.exists()

    def test_claude_md_exists(self):
        assert (BASE / "CLAUDE.md").exists()

    def test_config_env_exists(self):
        assert (BASE / ".config" / ".env").exists()


# ============================================================================
# SECTION 5: Ralph Loop State
# ============================================================================

class TestRalphLoopState:
    """Verify Ralph Loop V13 state is intact."""

    def test_ralph_loop_file_size(self):
        """Ralph loop must be substantial (V13 = ~10K lines)."""
        path = BASE / "platform" / "core" / "ralph_loop.py"
        assert path.exists()
        lines = path.read_text(encoding="utf-8").count("\n")
        assert lines >= 10000, f"Ralph loop only {lines} lines, expected 10000+"

    def test_iteration_state_has_learnings(self):
        data = json.loads((BASE / "iteration-state.json").read_text(encoding="utf-8"))
        assert "learnings" in data
        assert len(data["learnings"]) >= 50, (
            f"Only {len(data['learnings'])} learnings, expected 50+"
        )


# ============================================================================
# SECTION 6: Opik Tracing
# ============================================================================

class TestOpikTracing:
    """Verify Opik observability is wired."""

    def test_opik_installed(self):
        import opik
        assert hasattr(opik, "track")

    def test_opik_integration_module_exists(self):
        path = BASE / "platform" / "core" / "opik_integration.py"
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "OpikClient" in content

    def test_opik_supports_default_mode(self):
        """OpikClient must work without API key (default mode)."""
        path = BASE / "platform" / "core" / "opik_integration.py"
        content = path.read_text(encoding="utf-8")
        assert "default mode" in content.lower() or "anonymous" in content.lower()

    def test_opik_v14_imports(self):
        """V14 enhanced observability module must import."""
        platform_path = str(BASE / "platform")
        if platform_path not in sys.path:
            sys.path.insert(0, platform_path)
        from core.opik_v14 import OpikV14, AgentOptimizer, OutputGuardrail
        assert OpikV14 is not None
        assert AgentOptimizer is not None
        assert OutputGuardrail is not None

    def test_opik_track_decorator_works(self):
        """@opik.track must execute without error."""
        import opik

        @opik.track(name="v14_test_probe")
        def probe():
            return {"status": "ok"}

        result = probe()
        assert result["status"] == "ok"


# ============================================================================
# SECTION 7: Letta Cloud Integration
# ============================================================================

class TestLettaCloudIntegration:
    """Verify Letta Cloud cross-session memory."""

    def test_cross_session_memory_module_exists(self):
        path = BASE / "platform" / "core" / "cross_session_memory.py"
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "letta_sync" in content
        assert "CrossSessionMemory" in content

    def test_cross_session_memory_has_letta_sync(self):
        """Must have Letta Cloud sync method."""
        path = BASE / "platform" / "core" / "cross_session_memory.py"
        content = path.read_text(encoding="utf-8")
        assert "_sync_to_letta" in content
        assert "passages.create" in content

    def test_advanced_memory_has_letta(self):
        path = BASE / "platform" / "core" / "advanced_memory.py"
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "Letta" in content or "letta" in content

    def test_letta_agent_id_configured(self):
        """ECOSYSTEM agent ID must be in cross-session memory."""
        path = BASE / "platform" / "core" / "cross_session_memory.py"
        content = path.read_text(encoding="utf-8")
        assert "agent-daee71d2" in content


# ============================================================================
# SECTION 8: Self-Improvement Integration
# ============================================================================

class TestSelfImprovement:
    """Verify self-improvement infrastructure."""

    @pytest.fixture(autouse=True)
    def setup_path(self):
        platform_path = str(BASE / "platform")
        if platform_path not in sys.path:
            sys.path.insert(0, platform_path)

    def test_self_improvement_imports_with_confidence(self):
        """unified_confidence must load (fixed import path)."""
        from core.self_improvement import HAS_CONFIDENCE
        assert HAS_CONFIDENCE, "unified_confidence not loading - import path broken"

    def test_letta_client_available(self):
        """Letta client must be importable for persistence."""
        from core.self_improvement import HAS_LETTA
        assert HAS_LETTA, "letta_client not importable"

    def test_friction_detector_works(self):
        """Friction detection must trigger on 2+ repeated errors."""
        from core.self_improvement import SelfImprovementOrchestrator
        orch = SelfImprovementOrchestrator(session_id="test-v14")
        f1 = orch.friction_detector.detect_repeated_error("TestError", "same msg")
        assert f1 is None, "Should not trigger on first error"
        f2 = orch.friction_detector.detect_repeated_error("TestError", "same msg")
        assert f2 is not None, "Should trigger on second occurrence"
        assert f2.occurrence_count == 2

    def test_pattern_store_available(self):
        """Pattern store must be initialized when HAS_CONFIDENCE is True."""
        from core.self_improvement import SelfImprovementOrchestrator, HAS_CONFIDENCE
        if HAS_CONFIDENCE:
            orch = SelfImprovementOrchestrator(session_id="test-v14")
            assert orch.pattern_store is not None

    def test_letta_agent_id_in_orchestrator(self):
        """ECOSYSTEM agent ID must be configured."""
        from core.self_improvement import SelfImprovementOrchestrator
        assert "agent-daee71d2" in SelfImprovementOrchestrator.LETTA_AGENT_ID


# ============================================================================
# SECTION 9: End-to-End Integration
# ============================================================================

class TestEndToEndIntegration:
    """Verify cross-component integration."""

    @pytest.fixture(autouse=True)
    def setup_path(self):
        platform_path = str(BASE / "platform")
        if platform_path not in sys.path:
            sys.path.insert(0, platform_path)

    def test_self_improvement_letta_sync_capability(self):
        """SelfImprovementOrchestrator must support letta_sync parameter."""
        from core.self_improvement import SelfImprovementOrchestrator, HAS_LETTA
        orch = SelfImprovementOrchestrator(session_id="test", letta_sync=True)
        assert orch.letta_sync == HAS_LETTA  # True only if letta_client available

    def test_cross_session_memory_write_read_cycle(self):
        """CrossSessionMemory must persist across instances."""
        import tempfile
        from core.cross_session_memory import CrossSessionMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            csm = CrossSessionMemory(base_path=Path(tmpdir))
            mem = csm.add(content="V14 test memory", memory_type="learning", importance=0.9)
            assert mem.id is not None
            results = csm.search("V14 test")
            assert len(results) >= 1
            # Reload and verify persistence
            csm2 = CrossSessionMemory(base_path=Path(tmpdir))
            results2 = csm2.search("V14")
            assert len(results2) >= 1

    def test_core_module_count(self):
        """Must have at least 12 importable core modules."""
        import importlib
        modules = [
            "core.ralph_loop", "core.learning", "core.self_improvement",
            "core.cross_session_memory", "core.memory_tiers",
            "core.unified_confidence", "core.opik_integration",
            "core.opik_v14", "core.advanced_memory",
            "core.v14_optimizations", "core.v50_integration",
            "core.code_intel_observability",
        ]
        imported = 0
        for mod in modules:
            try:
                importlib.import_module(mod)
                imported += 1
            except Exception:
                pass
        assert imported >= 12, f"Only {imported}/12 core modules importing"


# ============================================================================
# SECTION 10: DSPy Integration
# ============================================================================

class TestDSPyIntegration:
    """Verify DSPy 2.6.5 is functional."""

    def test_dspy_imports(self):
        import dspy
        assert hasattr(dspy, "Predict")
        assert hasattr(dspy, "ChainOfThought")
        assert hasattr(dspy, "Signature")

    def test_dspy_version(self):
        import dspy
        assert dspy.__version__.startswith("2.6")

    def test_dspy_miprov2_available(self):
        from dspy.teleprompt import MIPROv2
        assert MIPROv2 is not None

    def test_dspy_signature_creation(self):
        import dspy
        sig = dspy.Signature("question -> answer")
        assert sig is not None

    def test_dspy_predict_module(self):
        import dspy
        sig = dspy.Signature("question -> answer")
        pred = dspy.Predict(sig)
        assert pred is not None


# ============================================================================
# SECTION 11: LiteLLM, Instructor, LangGraph
# ============================================================================

class TestLiteLLMIntegration:
    """Verify LiteLLM multi-provider gateway."""

    def test_litellm_imports(self):
        import litellm
        assert hasattr(litellm, "completion")
        assert hasattr(litellm, "acompletion")
        assert hasattr(litellm, "embedding")

    def test_litellm_provider_list(self):
        import litellm
        assert hasattr(litellm, "provider_list")
        assert len(litellm.provider_list) > 5


class TestInstructorIntegration:
    """Verify Instructor structured output library."""

    def test_instructor_imports(self):
        import instructor
        assert hasattr(instructor, "from_anthropic")
        assert hasattr(instructor, "from_openai")

    def test_instructor_version(self):
        import instructor
        assert instructor.__version__.startswith("1.")


class TestLangGraphIntegration:
    """Verify LangGraph stateful agent framework."""

    def test_langgraph_state_graph(self):
        from langgraph.graph import StateGraph, END
        assert StateGraph is not None
        assert END is not None

    def test_langgraph_graph_creation(self):
        from langgraph.graph import StateGraph, END
        from typing import TypedDict

        class State(TypedDict):
            value: str

        graph = StateGraph(State)
        assert graph is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
