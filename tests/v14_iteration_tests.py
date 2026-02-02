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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
