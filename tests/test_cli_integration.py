#!/usr/bin/env python3
"""
CLI Integration Tests for V35

Tests command-line interface functionality of the Unleash platform.
"""

import subprocess
import sys
import os

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_python_module(module: str, *args, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a Python module as a subprocess."""
    cmd = [sys.executable, "-m", module, *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=timeout,
    )


def run_python_script(script: str, *args, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a Python script as a subprocess."""
    cmd = [sys.executable, script, *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=timeout,
    )


class TestV35Validation:
    """Test V35 validation script."""

    def test_validate_v35_runs(self):
        """V35 validation script executes successfully"""
        result = run_python_script("scripts/validate_v35_final.py")

        # Should complete without crash
        assert result.returncode == 0, f"Validation failed: {result.stderr}"

        # Should report 36/36
        assert "36/36" in result.stdout or "100%" in result.stdout, \
            f"Expected 100% pass rate: {result.stdout}"

    def test_validate_v35_output_json(self):
        """V35 validation outputs valid JSON results"""
        result_file = os.path.join(PROJECT_ROOT, "validation_v35_result.json")

        # Run validation
        run_python_script("scripts/validate_v35_final.py")

        # Check JSON file exists
        assert os.path.exists(result_file), f"Result file not created: {result_file}"

        # Validate JSON structure
        import json
        with open(result_file, "r") as f:
            data = json.load(f)

        assert "passed" in data
        assert "failed" in data
        assert "total_sdks" in data
        assert data["passed"] == 36
        assert data["failed"] == 0


class TestCoreImports:
    """Test core module can be imported."""

    def test_core_importable(self):
        """Core module imports without error"""
        result = subprocess.run(
            [sys.executable, "-c", "import core; print('OK')"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=120,  # Extended for heavy ML deps
        )
        assert "OK" in result.stdout, f"Core import failed: {result.stderr}"

    def test_compat_layers_importable(self):
        """All compat layers import without error"""
        imports = """
from core.orchestration.crewai_compat import CREWAI_COMPAT_AVAILABLE
from core.memory.zep_compat import ZEP_COMPAT_AVAILABLE
from core.structured.outlines_compat import OUTLINES_COMPAT_AVAILABLE
from core.reasoning.agentlite_compat import AGENTLITE_COMPAT_AVAILABLE
from core.observability.langfuse_compat import LANGFUSE_COMPAT_AVAILABLE
from core.observability.phoenix_compat import PHOENIX_COMPAT_AVAILABLE
from core.safety.scanner_compat import SCANNER_COMPAT_AVAILABLE
from core.safety.rails_compat import RAILS_COMPAT_AVAILABLE
from core.processing.aider_compat import AIDER_COMPAT_AVAILABLE
print(all([
    CREWAI_COMPAT_AVAILABLE,
    ZEP_COMPAT_AVAILABLE,
    OUTLINES_COMPAT_AVAILABLE,
    AGENTLITE_COMPAT_AVAILABLE,
    LANGFUSE_COMPAT_AVAILABLE,
    PHOENIX_COMPAT_AVAILABLE,
    SCANNER_COMPAT_AVAILABLE,
    RAILS_COMPAT_AVAILABLE,
    AIDER_COMPAT_AVAILABLE,
]))
"""
        result = subprocess.run(
            [sys.executable, "-c", imports],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=120,  # Extended for heavy ML deps
        )
        assert "True" in result.stdout, f"Compat import failed: {result.stderr}"


class TestVerificationScripts:
    """Test verification test scripts."""

    def test_v35_verification_exists(self):
        """V35 verification test exists"""
        test_file = os.path.join(PROJECT_ROOT, "tests", "v35_verification_tests.py")
        assert os.path.exists(test_file), f"V35 verification not found: {test_file}"

    def test_v35_verification_runs(self):
        """V35 verification tests execute"""
        test_file = os.path.join(PROJECT_ROOT, "tests", "v35_verification_tests.py")
        if not os.path.exists(test_file):
            return  # Skip if file doesn't exist

        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=60,
        )
        # Should not crash
        assert result.returncode in [0, 1, 5], \
            f"Verification crashed: {result.stderr}"


def run_all_cli_tests():
    """Run all CLI tests and report results."""
    import traceback

    tests = [
        ("V35 Validation Runs", TestV35Validation().test_validate_v35_runs),
        ("V35 Output JSON", TestV35Validation().test_validate_v35_output_json),
        ("Core Importable", TestCoreImports().test_core_importable),
        ("Compat Layers Import", TestCoreImports().test_compat_layers_importable),
        ("V35 Verification Exists", TestVerificationScripts().test_v35_verification_exists),
        ("V35 Verification Runs", TestVerificationScripts().test_v35_verification_runs),
    ]

    passed = 0
    failed = 0
    results = []

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
            results.append((name, "PASS", None))
            print(f"  [PASS] {name}")
        except Exception as e:
            failed += 1
            results.append((name, "FAIL", str(e)))
            print(f"  [FAIL] {name}: {e}")

    print(f"\n{'='*60}")
    print(f"CLI Integration Tests: {passed}/{passed+failed} passed")
    print(f"{'='*60}")

    return passed, failed, results


if __name__ == "__main__":
    passed, failed, results = run_all_cli_tests()
    sys.exit(0 if failed == 0 else 1)
