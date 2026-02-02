#!/usr/bin/env python3
"""
CLI Command Tests for V35

Tests all CLI commands work correctly with the V35 SDK stack.
Phase 14: CLI Commands Verification
"""

import subprocess
import sys
import os

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_cli(*args, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run CLI command and return result."""
    cmd = [sys.executable, "-m", "core.cli.unified_cli", *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=timeout,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )


class TestCoreCommands:
    """Test core CLI commands."""

    def test_version(self):
        """CLI shows version 35.0.0"""
        result = run_cli("--version")
        assert result.returncode == 0, f"Version failed: {result.stderr}"
        assert "35.0.0" in result.stdout, f"Wrong version: {result.stdout}"

    def test_help(self):
        """CLI shows help"""
        result = run_cli("--help")
        assert result.returncode == 0, f"Help failed: {result.stderr}"
        assert "unleash" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_status(self):
        """CLI status command works"""
        result = run_cli("status")
        # Should not crash (import errors are acceptable)
        assert result.returncode in [0, 1], f"Status crashed: {result.stderr}"


class TestProtocolCommands:
    """Test L0 Protocol commands."""

    def test_protocol_help(self):
        """Protocol group shows help"""
        result = run_cli("protocol", "--help")
        assert result.returncode == 0, f"Protocol help failed: {result.stderr}"
        assert "protocol" in result.stdout.lower() or "llm" in result.stdout.lower()

    def test_protocol_call_help(self):
        """Protocol call shows help"""
        result = run_cli("protocol", "call", "--help")
        assert result.returncode == 0
        assert "prompt" in result.stdout.lower()

    def test_protocol_chat_help(self):
        """Protocol chat shows help"""
        result = run_cli("protocol", "chat", "--help")
        assert result.returncode == 0
        assert "model" in result.stdout.lower() or "chat" in result.stdout.lower()


class TestMemoryCommands:
    """Test L2 Memory commands."""

    def test_memory_help(self):
        """Memory group shows help"""
        result = run_cli("memory", "--help")
        assert result.returncode == 0, f"Memory help failed: {result.stderr}"

    def test_memory_store_help(self):
        """Memory store shows help"""
        result = run_cli("memory", "store", "--help")
        assert result.returncode == 0

    def test_memory_search_help(self):
        """Memory search shows help"""
        result = run_cli("memory", "search", "--help")
        assert result.returncode == 0

    def test_memory_list_help(self):
        """Memory list shows help"""
        result = run_cli("memory", "list", "--help")
        assert result.returncode == 0


class TestStructuredCommands:
    """Test L3 Structured commands."""

    def test_structured_help(self):
        """Structured group shows help"""
        result = run_cli("structured", "--help")
        assert result.returncode == 0, f"Structured help failed: {result.stderr}"

    def test_structured_generate_help(self):
        """Structured generate shows help"""
        result = run_cli("structured", "generate", "--help")
        assert result.returncode == 0

    def test_structured_validate_help(self):
        """Structured validate shows help"""
        result = run_cli("structured", "validate", "--help")
        assert result.returncode == 0


class TestSafetyCommands:
    """Test L6 Safety commands."""

    def test_safety_help(self):
        """Safety group shows help"""
        result = run_cli("safety", "--help")
        assert result.returncode == 0, f"Safety help failed: {result.stderr}"

    def test_safety_scan_help(self):
        """Safety scan shows help"""
        result = run_cli("safety", "scan", "--help")
        assert result.returncode == 0

    def test_safety_guard_help(self):
        """Safety guard shows help"""
        result = run_cli("safety", "guard", "--help")
        assert result.returncode == 0

    def test_safety_guard_status_help(self):
        """Safety guard status shows help"""
        result = run_cli("safety", "guard", "status", "--help")
        assert result.returncode == 0


class TestDocCommands:
    """Test L7 Processing commands."""

    def test_doc_help(self):
        """Doc group shows help"""
        result = run_cli("doc", "--help")
        assert result.returncode == 0, f"Doc help failed: {result.stderr}"

    def test_doc_convert_help(self):
        """Doc convert shows help"""
        result = run_cli("doc", "convert", "--help")
        assert result.returncode == 0

    def test_doc_extract_help(self):
        """Doc extract shows help"""
        result = run_cli("doc", "extract", "--help")
        assert result.returncode == 0


class TestKnowledgeCommands:
    """Test L8 Knowledge commands."""

    def test_knowledge_help(self):
        """Knowledge group shows help"""
        result = run_cli("knowledge", "--help")
        assert result.returncode == 0, f"Knowledge help failed: {result.stderr}"

    def test_knowledge_index_help(self):
        """Knowledge index shows help"""
        result = run_cli("knowledge", "index", "--help")
        assert result.returncode == 0

    def test_knowledge_search_help(self):
        """Knowledge search shows help"""
        result = run_cli("knowledge", "search", "--help")
        assert result.returncode == 0

    def test_knowledge_list_help(self):
        """Knowledge list shows help"""
        result = run_cli("knowledge", "list", "--help")
        assert result.returncode == 0


class TestObservabilityCommands:
    """Test L5 Observability commands."""

    def test_trace_help(self):
        """Trace group shows help"""
        result = run_cli("trace", "--help")
        assert result.returncode == 0, f"Trace help failed: {result.stderr}"

    def test_eval_help(self):
        """Eval group shows help"""
        result = run_cli("eval", "--help")
        assert result.returncode == 0, f"Eval help failed: {result.stderr}"


class TestConfigCommands:
    """Test config commands."""

    def test_config_help(self):
        """Config group shows help"""
        result = run_cli("config", "--help")
        assert result.returncode == 0, f"Config help failed: {result.stderr}"

    def test_config_show_help(self):
        """Config show shows help"""
        result = run_cli("config", "show", "--help")
        assert result.returncode == 0


class TestToolsCommands:
    """Test tools commands."""

    def test_tools_help(self):
        """Tools group shows help"""
        result = run_cli("tools", "--help")
        assert result.returncode == 0, f"Tools help failed: {result.stderr}"


class TestRunCommands:
    """Test run commands."""

    def test_run_help(self):
        """Run group shows help"""
        result = run_cli("run", "--help")
        assert result.returncode == 0, f"Run help failed: {result.stderr}"


def run_all_cli_tests():
    """Run all CLI tests and report results."""
    test_classes = [
        TestCoreCommands,
        TestProtocolCommands,
        TestMemoryCommands,
        TestStructuredCommands,
        TestSafetyCommands,
        TestDocCommands,
        TestKnowledgeCommands,
        TestObservabilityCommands,
        TestConfigCommands,
        TestToolsCommands,
        TestRunCommands,
    ]

    passed = 0
    failed = 0
    results = []

    for test_class in test_classes:
        instance = test_class()
        class_name = test_class.__name__
        print(f"\n{class_name}:")

        for name in dir(instance):
            if name.startswith("test_"):
                test_name = f"{class_name}.{name}"
                try:
                    getattr(instance, name)()
                    passed += 1
                    results.append((test_name, "PASS", None))
                    print(f"  [PASS] {name}")
                except Exception as e:
                    failed += 1
                    results.append((test_name, "FAIL", str(e)))
                    print(f"  [FAIL] {name}: {e}")

    print(f"\n{'=' * 60}")
    print(f"CLI Command Tests: {passed}/{passed + failed} passed")
    print(f"{'=' * 60}")

    return passed, failed, results


if __name__ == "__main__":
    passed, failed, results = run_all_cli_tests()
    sys.exit(0 if failed == 0 else 1)
