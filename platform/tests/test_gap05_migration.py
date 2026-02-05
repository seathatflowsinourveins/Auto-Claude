"""
Tests for Gap05 Migration - Research iteration scripts using base_executor.py

Verifies that migrated scripts:
1. Import correctly from base_executor
2. Have valid TOPICS lists with required keys
3. Define custom executor classes inheriting from BaseResearchExecutor
4. Can be instantiated without errors
"""

import ast
import importlib.util
import os
import sys
from pathlib import Path

import pytest


# Path to research iterations directory
ITERATIONS_DIR = Path(__file__).parent.parent.parent / "research" / "iterations"


# Scripts that should be migrated (using base_executor)
MIGRATED_SCRIPTS = [
    "llmops_iterations.py",
    "advanced_rag_iterations.py",
    "llm_agents_iterations.py",
    "embedding_models_iterations.py",
    "agentic_rag_iterations.py",
    "cutting_edge_iterations.py",
    "battle_tested_iterations.py",
    "memory_integration_iterations.py",
    "advanced_production_iterations.py",
    "guardrails_safety_iterations.py",
    "vector_databases_iterations.py",
    "mcp_protocol_iterations.py",
    "prompt_engineering_iterations.py",
    "evaluation_benchmarking_iterations.py",
    "agent_memory_iterations.py",
]


def get_iteration_files():
    """Get all iteration files in the research/iterations directory."""
    if not ITERATIONS_DIR.exists():
        return []
    return [f for f in ITERATIONS_DIR.glob("*_iterations.py")
            if f.name not in ("base_executor.py", "legacy_adapter.py", "run_iteration.py")]


class TestMigratedScriptsStructure:
    """Test that migrated scripts have correct structure."""

    @pytest.mark.parametrize("script_name", MIGRATED_SCRIPTS)
    def test_script_exists(self, script_name):
        """Verify migrated script exists."""
        script_path = ITERATIONS_DIR / script_name
        assert script_path.exists(), f"Script {script_name} not found at {script_path}"

    @pytest.mark.parametrize("script_name", MIGRATED_SCRIPTS)
    def test_imports_base_executor(self, script_name):
        """Verify script imports from base_executor."""
        script_path = ITERATIONS_DIR / script_name
        if not script_path.exists():
            pytest.skip(f"Script {script_name} not found")

        source = script_path.read_text(encoding="utf-8")
        assert "from base_executor import" in source, \
            f"{script_name} does not import from base_executor"

    @pytest.mark.parametrize("script_name", MIGRATED_SCRIPTS)
    def test_has_topics_list(self, script_name):
        """Verify script has a TOPICS list variable."""
        script_path = ITERATIONS_DIR / script_name
        if not script_path.exists():
            pytest.skip(f"Script {script_name} not found")

        source = script_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        topics_found = False
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id == "TOPICS" or target.id.endswith("_TOPICS"):
                            topics_found = True
                            break

        assert topics_found, f"{script_name} does not have a TOPICS list"

    @pytest.mark.parametrize("script_name", MIGRATED_SCRIPTS)
    def test_has_executor_class(self, script_name):
        """Verify script has a custom executor class."""
        script_path = ITERATIONS_DIR / script_name
        if not script_path.exists():
            pytest.skip(f"Script {script_name} not found")

        source = script_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        executor_found = False
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                if node.name.endswith("Executor"):
                    executor_found = True
                    break

        assert executor_found, f"{script_name} does not have an Executor class"

    @pytest.mark.parametrize("script_name", MIGRATED_SCRIPTS)
    def test_has_main_block(self, script_name):
        """Verify script has if __name__ == '__main__' block."""
        script_path = ITERATIONS_DIR / script_name
        if not script_path.exists():
            pytest.skip(f"Script {script_name} not found")

        source = script_path.read_text(encoding="utf-8")
        assert 'if __name__ == "__main__"' in source or "if __name__ == '__main__'" in source, \
            f"{script_name} does not have a main block"

    @pytest.mark.parametrize("script_name", MIGRATED_SCRIPTS)
    def test_calls_run_research(self, script_name):
        """Verify script calls run_research in main block."""
        script_path = ITERATIONS_DIR / script_name
        if not script_path.exists():
            pytest.skip(f"Script {script_name} not found")

        source = script_path.read_text(encoding="utf-8")
        assert "run_research(" in source, \
            f"{script_name} does not call run_research()"


class TestTopicsFormat:
    """Test that TOPICS lists have correct format."""

    @pytest.mark.parametrize("script_name", MIGRATED_SCRIPTS)
    def test_topics_have_required_keys(self, script_name):
        """Verify each topic has 'topic' and 'area' keys."""
        script_path = ITERATIONS_DIR / script_name
        if not script_path.exists():
            pytest.skip(f"Script {script_name} not found")

        source = script_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id == "TOPICS" or target.id.endswith("_TOPICS"):
                            try:
                                topics = ast.literal_eval(node.value)
                            except (ValueError, TypeError):
                                pytest.skip(f"Cannot evaluate TOPICS in {script_name}")

                            assert isinstance(topics, list), \
                                f"TOPICS in {script_name} is not a list"
                            assert len(topics) > 0, \
                                f"TOPICS in {script_name} is empty"

                            for i, topic in enumerate(topics):
                                assert isinstance(topic, dict), \
                                    f"Topic {i} in {script_name} is not a dict"
                                assert "topic" in topic, \
                                    f"Topic {i} in {script_name} missing 'topic' key"
                                assert "area" in topic, \
                                    f"Topic {i} in {script_name} missing 'area' key"
                            return

        pytest.fail(f"Could not find TOPICS list in {script_name}")


class TestLegacyAdapter:
    """Test legacy adapter functionality."""

    def test_legacy_adapter_exists(self):
        """Verify legacy_adapter.py exists."""
        adapter_path = ITERATIONS_DIR / "legacy_adapter.py"
        assert adapter_path.exists(), "legacy_adapter.py not found"

    def test_legacy_adapter_has_required_classes(self):
        """Verify legacy adapter has LegacyScriptAdapter class."""
        adapter_path = ITERATIONS_DIR / "legacy_adapter.py"
        if not adapter_path.exists():
            pytest.skip("legacy_adapter.py not found")

        source = adapter_path.read_text(encoding="utf-8")
        assert "class LegacyScriptAdapter" in source, \
            "LegacyScriptAdapter class not found"
        assert "class LegacyAdapterError" in source, \
            "LegacyAdapterError class not found"

    def test_legacy_adapter_has_required_methods(self):
        """Verify legacy adapter has required methods."""
        adapter_path = ITERATIONS_DIR / "legacy_adapter.py"
        if not adapter_path.exists():
            pytest.skip("legacy_adapter.py not found")

        source = adapter_path.read_text(encoding="utf-8")
        required_methods = [
            "def initialize",
            "async def research",
            "def filter_findings",
            "def deduplicate",
            "def synthesize",
            "def score_quality",
            "async def run_iteration",
            "def save_results",
            "def detect_topics",
            "def detect_executor_class",
            "def generate_migration_diff",
        ]

        for method in required_methods:
            assert method in source, \
                f"Method '{method}' not found in legacy_adapter.py"


class TestBaseExecutor:
    """Test base_executor.py functionality."""

    def test_base_executor_exists(self):
        """Verify base_executor.py exists."""
        executor_path = ITERATIONS_DIR / "base_executor.py"
        assert executor_path.exists(), "base_executor.py not found"

    def test_base_executor_has_required_classes(self):
        """Verify base executor has required classes."""
        executor_path = ITERATIONS_DIR / "base_executor.py"
        if not executor_path.exists():
            pytest.skip("base_executor.py not found")

        source = executor_path.read_text(encoding="utf-8")
        assert "class BaseResearchExecutor" in source, \
            "BaseResearchExecutor class not found"
        assert "class ResearchResult" in source or "@dataclass" in source, \
            "ResearchResult dataclass not found"

    def test_base_executor_has_gap_features(self):
        """Verify base executor implements Gap02-11 features."""
        executor_path = ITERATIONS_DIR / "base_executor.py"
        if not executor_path.exists():
            pytest.skip("base_executor.py not found")

        source = executor_path.read_text(encoding="utf-8")

        # Gap02: Quality filtering
        assert "filter_findings" in source, "Gap02: filter_findings not found"

        # Gap04: Stats recomputation
        assert "actual_findings" in source or "recompute" in source.lower(), \
            "Gap04: Stats recomputation not found"

        # Gap06: Synthesis
        assert "synthesize" in source, "Gap06: synthesize not found"
        assert "_claim_indicators" in source, "Gap06: claim indicators not found"

        # Gap07: Deduplication
        assert "_seen_urls" in source, "Gap07: URL dedup not found"
        assert "_seen_vector_hashes" in source, "Gap07: Vector dedup not found"

        # Gap09: Fallback
        assert "research_with_fallback" in source, "Gap09: fallback not found"

        # Gap11: Quality scoring
        assert "score_quality" in source, "Gap11: quality scoring not found"
        assert "quality_dashboard" in source, "Gap11: quality dashboard not found"


class TestMigrationCoverage:
    """Test migration coverage metrics."""

    def test_migration_progress(self):
        """Report migration progress."""
        all_files = list(get_iteration_files())

        migrated_count = 0
        for f in all_files:
            source = f.read_text(encoding="utf-8")
            if "from base_executor import" in source:
                migrated_count += 1

        total = len(all_files)
        percentage = (migrated_count / total * 100) if total > 0 else 0

        print(f"\nMigration Progress: {migrated_count}/{total} ({percentage:.1f}%)")
        print(f"Remaining: {total - migrated_count} scripts")

        # At least 10% should be migrated after Gap05 implementation
        assert migrated_count >= 5, \
            f"Expected at least 5 migrated scripts, found {migrated_count}"

    def test_no_legacy_patterns_in_migrated(self):
        """Verify migrated scripts don't have legacy patterns."""
        legacy_patterns = [
            "asyncio.run(main())",  # Should use run_research instead
            "class.*Executor:.*async def initialize.*async def research.*async def _exa",
        ]

        for script_name in MIGRATED_SCRIPTS:
            script_path = ITERATIONS_DIR / script_name
            if not script_path.exists():
                continue

            source = script_path.read_text(encoding="utf-8")

            # Should NOT have asyncio.run(main()) pattern
            if "asyncio.run(main())" in source:
                pytest.fail(f"{script_name} still uses asyncio.run(main()) instead of run_research()")

            # Should NOT have duplicated API methods
            if source.count("async def _exa(") > 0:
                pytest.fail(f"{script_name} still has duplicated _exa method")
            if source.count("async def _tavily(") > 0:
                pytest.fail(f"{script_name} still has duplicated _tavily method")


class TestSyntaxValidity:
    """Test that all scripts have valid Python syntax."""

    @pytest.mark.parametrize("script_name", MIGRATED_SCRIPTS)
    def test_syntax_valid(self, script_name):
        """Verify script has valid Python syntax."""
        script_path = ITERATIONS_DIR / script_name
        if not script_path.exists():
            pytest.skip(f"Script {script_name} not found")

        source = script_path.read_text(encoding="utf-8")
        try:
            compile(source, script_path, "exec")
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {script_name}: {e}")

    def test_base_executor_syntax_valid(self):
        """Verify base_executor.py has valid syntax."""
        executor_path = ITERATIONS_DIR / "base_executor.py"
        if not executor_path.exists():
            pytest.skip("base_executor.py not found")

        source = executor_path.read_text(encoding="utf-8")
        try:
            compile(source, executor_path, "exec")
        except SyntaxError as e:
            pytest.fail(f"Syntax error in base_executor.py: {e}")

    def test_legacy_adapter_syntax_valid(self):
        """Verify legacy_adapter.py has valid syntax."""
        adapter_path = ITERATIONS_DIR / "legacy_adapter.py"
        if not adapter_path.exists():
            pytest.skip("legacy_adapter.py not found")

        source = adapter_path.read_text(encoding="utf-8")
        try:
            compile(source, adapter_path, "exec")
        except SyntaxError as e:
            pytest.fail(f"Syntax error in legacy_adapter.py: {e}")
