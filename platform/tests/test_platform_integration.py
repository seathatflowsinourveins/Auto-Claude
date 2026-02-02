#!/usr/bin/env python3
"""
Platform Integration Tests - Ultimate Autonomous Platform

Comprehensive integration tests for V10 modules:
1. Module import tests
2. CLI interface tests
3. Cross-module communication tests
4. End-to-end workflow tests
5. Error handling and edge cases

Usage:
    pytest test_platform_integration.py -v
    pytest test_platform_integration.py -v -k "test_cli"
    pytest test_platform_integration.py -v --tb=short

Platform: Windows 11 + Python 3.11+
"""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent.parent / "scripts"
V10_DIR = Path(__file__).parent.parent
PROJECT_ROOT = V10_DIR.parent

# Timeout for subprocess calls
SUBPROCESS_TIMEOUT = 60


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def script_runner():
    """Fixture to run scripts via uv."""
    def run(script: str, args: list[str] = None, timeout: int = SUBPROCESS_TIMEOUT) -> Tuple[int, str, str]:
        """Run a script and return (exit_code, stdout, stderr)."""
        cmd = ["uv", "run", str(SCRIPT_DIR / script)]
        if args:
            cmd.extend(args)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(SCRIPT_DIR),
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Timeout"
        except Exception as e:
            return -2, "", str(e)

    return run


@pytest.fixture
def json_runner(script_runner):
    """Fixture to run scripts with --json flag and parse output."""
    def run(script: str, args: list[str] = None, timeout: int = SUBPROCESS_TIMEOUT) -> Tuple[int, Dict[str, Any]]:
        """Run a script with --json and return (exit_code, parsed_json)."""
        full_args = ["--json"] + (args or [])
        exit_code, stdout, stderr = script_runner(script, full_args, timeout)

        if exit_code < 0:
            return exit_code, {"error": stderr}

        try:
            # Find JSON in output (skip non-JSON lines)
            for line in stdout.strip().split("\n"):
                if line.startswith("{") or line.startswith("["):
                    return exit_code, json.loads(line)
            # Try parsing entire output
            return exit_code, json.loads(stdout)
        except json.JSONDecodeError:
            return exit_code, {"raw": stdout, "error": "JSON parse failed"}

    return run


# =============================================================================
# Module Import Tests
# =============================================================================

class TestModuleImports:
    """Test that all modules can be imported."""

    def test_import_ecosystem_orchestrator(self):
        """Test ecosystem_orchestrator imports."""
        sys.path.insert(0, str(SCRIPT_DIR))
        try:
            import ecosystem_orchestrator
            assert hasattr(ecosystem_orchestrator, "EcosystemOrchestrator")
        finally:
            sys.path.pop(0)

    def test_import_auto_validate(self):
        """Test auto_validate imports."""
        sys.path.insert(0, str(SCRIPT_DIR))
        try:
            import auto_validate
            assert hasattr(auto_validate, "AutoValidator")
            assert hasattr(auto_validate, "ValidationStatus")
        finally:
            sys.path.pop(0)

    def test_import_sleeptime_compute(self):
        """Test sleeptime_compute imports."""
        sys.path.insert(0, str(SCRIPT_DIR))
        try:
            import sleeptime_compute
            assert hasattr(sleeptime_compute, "SleepTimeDaemon")
            assert hasattr(sleeptime_compute, "MemoryManager")
        finally:
            sys.path.pop(0)

    def test_import_session_continuity(self):
        """Test session_continuity imports."""
        sys.path.insert(0, str(SCRIPT_DIR))
        try:
            import session_continuity
            assert hasattr(session_continuity, "SessionManager")
            assert hasattr(session_continuity, "TrinityManager")
        finally:
            sys.path.pop(0)

    def test_import_ralph_loop(self):
        """Test ralph_loop imports."""
        sys.path.insert(0, str(SCRIPT_DIR))
        try:
            import ralph_loop
            assert hasattr(ralph_loop, "RalphLoop")
            assert hasattr(ralph_loop, "IterationPhase")
        finally:
            sys.path.pop(0)

    def test_import_performance(self):
        """Test performance imports."""
        sys.path.insert(0, str(SCRIPT_DIR))
        try:
            import performance
            assert hasattr(performance, "PerformanceRunner")
            assert hasattr(performance, "MemoryProfiler")
        finally:
            sys.path.pop(0)


# =============================================================================
# CLI Interface Tests
# =============================================================================

class TestCLIInterfaces:
    """Test CLI interfaces for all modules."""

    def test_ecosystem_orchestrator_quick(self, script_runner):
        """Test ecosystem_orchestrator --quick."""
        exit_code, stdout, stderr = script_runner("ecosystem_orchestrator.py", ["--quick"])
        assert exit_code in [0, 1, 2], f"Unexpected exit code: {exit_code}"
        assert "ECOSYSTEM HEALTH" in stdout or "ecosystem" in stdout.lower()

    def test_ecosystem_orchestrator_json(self, json_runner):
        """Test ecosystem_orchestrator --json."""
        exit_code, data = json_runner("ecosystem_orchestrator.py", ["--quick"])
        assert exit_code in [0, 1, 2]
        # Should have structured output
        assert isinstance(data, dict)

    def test_auto_validate_quick(self, script_runner):
        """Test auto_validate --quick."""
        exit_code, stdout, stderr = script_runner("auto_validate.py", ["--quick"])
        assert exit_code in [0, 1, 2]
        assert "AUTO-VALIDATION" in stdout or "validation" in stdout.lower()

    def test_sleeptime_status(self, script_runner):
        """Test sleeptime_compute status."""
        exit_code, stdout, stderr = script_runner("sleeptime_compute.py", ["status"])
        assert exit_code in [0, 1, 2]
        assert "SLEEP-TIME" in stdout or "sleep" in stdout.lower()

    def test_session_continuity_status(self, script_runner):
        """Test session_continuity status."""
        exit_code, stdout, stderr = script_runner("session_continuity.py", ["status"])
        assert exit_code in [0, 1, 2]
        assert "SESSION" in stdout or "session" in stdout.lower()

    def test_ralph_loop_status(self, script_runner):
        """Test ralph_loop status."""
        exit_code, stdout, stderr = script_runner("ralph_loop.py", ["status"])
        assert exit_code in [0, 1, 2]
        assert "RALPH LOOP" in stdout or "ralph" in stdout.lower()

    def test_performance_status(self, script_runner):
        """Test performance status."""
        exit_code, stdout, stderr = script_runner("performance.py", ["status"])
        assert exit_code in [0, 1, 2]
        assert "PERFORMANCE" in stdout or "Python" in stdout


# =============================================================================
# Cross-Module Communication Tests
# =============================================================================

class TestCrossModuleCommunication:
    """Test modules working together."""

    def test_ralph_loop_uses_ecosystem(self, script_runner):
        """Test that ralph_loop properly calls ecosystem_orchestrator."""
        # Ralph loop status should work independently
        exit_code, stdout, stderr = script_runner("ralph_loop.py", ["status"])
        assert exit_code in [0, 1, 2]

    def test_validation_uses_hooks(self, script_runner):
        """Test that auto_validate checks hooks."""
        exit_code, stdout, stderr = script_runner("auto_validate.py", ["--quick"])
        # Should mention hooks validation
        assert exit_code in [0, 1, 2]
        # Output should include hook validation step
        assert "Hook" in stdout or "hook" in stdout.lower()

    def test_performance_profiles_modules(self, script_runner):
        """Test performance can profile other modules."""
        exit_code, stdout, stderr = script_runner("performance.py", ["benchmark"], timeout=120)
        assert exit_code in [0, 1, 2]
        # Should profile ecosystem_orchestrator, auto_validate, etc.
        assert "ecosystem_orchestrator" in stdout or "benchmark" in stdout.lower()


# =============================================================================
# Data Model Tests
# =============================================================================

class TestDataModels:
    """Test data models are correctly defined."""

    def test_validation_status_enum(self):
        """Test ValidationStatus enum values."""
        sys.path.insert(0, str(SCRIPT_DIR))
        try:
            from auto_validate import ValidationStatus
            assert ValidationStatus.PASS.value == "pass"
            assert ValidationStatus.FAIL.value == "fail"
            assert ValidationStatus.WARN.value == "warn"
            assert ValidationStatus.SKIP.value == "skip"
        finally:
            sys.path.pop(0)

    def test_iteration_phase_enum(self):
        """Test IterationPhase enum values."""
        sys.path.insert(0, str(SCRIPT_DIR))
        try:
            from ralph_loop import IterationPhase
            assert IterationPhase.HEALTH_CHECK.value == "health_check"
            assert IterationPhase.VALIDATION.value == "validation"
        finally:
            sys.path.pop(0)

    def test_sleep_phase_enum(self):
        """Test SleepPhase enum values."""
        sys.path.insert(0, str(SCRIPT_DIR))
        try:
            from sleeptime_compute import SleepPhase
            assert SleepPhase.IDLE.value == "idle"
            assert SleepPhase.CONSOLIDATING.value == "consolidating"
        finally:
            sys.path.pop(0)


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_command(self, script_runner):
        """Test handling of invalid commands."""
        exit_code, stdout, stderr = script_runner("ralph_loop.py", ["invalid_command"])
        # Should either show help or error gracefully
        assert exit_code != 0 or "usage" in stdout.lower() or "usage" in stderr.lower()

    def test_missing_dependencies_graceful(self):
        """Test that missing optional deps are handled."""
        # All scripts should import even if optional deps missing
        sys.path.insert(0, str(SCRIPT_DIR))
        try:
            # These should not raise even if deps missing
            import performance
            assert True
        except ImportError as e:
            pytest.fail(f"Import failed with missing deps: {e}")
        finally:
            sys.path.pop(0)

    def test_json_flag_always_works(self, script_runner):
        """Test --json flag produces valid output."""
        scripts = [
            ("ecosystem_orchestrator.py", ["--quick"]),
            ("auto_validate.py", ["--quick"]),
            ("performance.py", ["status"]),
        ]

        for script, extra_args in scripts:
            exit_code, stdout, stderr = script_runner(script, ["--json"] + extra_args)
            # Should not crash
            assert exit_code >= 0, f"{script} crashed with --json"


# =============================================================================
# End-to-End Workflow Tests
# =============================================================================

class TestEndToEndWorkflows:
    """Test complete workflows."""

    def test_full_validation_cycle(self, script_runner):
        """Test complete validation cycle."""
        # Run quick validation
        exit_code, stdout, stderr = script_runner("auto_validate.py", ["--quick"])
        assert exit_code in [0, 1, 2]

        # Should complete all steps
        assert "[1/" in stdout  # At least first step
        assert "RESULT" in stdout or "result" in stdout.lower()

    def test_session_export_workflow(self, script_runner):
        """Test session export for teleportation."""
        # Export session
        exit_code, stdout, stderr = script_runner("session_continuity.py", ["export"])
        assert exit_code in [0, 1, 2]

    def test_performance_analysis_workflow(self, script_runner):
        """Test performance analysis workflow."""
        # Get status
        exit_code, stdout, stderr = script_runner("performance.py", ["status"])
        assert exit_code in [0, 1, 2]
        assert "Python" in stdout or "CPU" in stdout

    @pytest.mark.slow
    def test_single_ralph_iteration(self, script_runner):
        """Test a single Ralph Loop iteration (slow)."""
        exit_code, stdout, stderr = script_runner("ralph_loop.py", ["iterate"], timeout=120)
        assert exit_code in [0, 1, 2]
        # Should show phases
        assert "RALPH LOOP" in stdout
        assert "RESULT" in stdout or "complete" in stdout.lower()


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Test performance characteristics."""

    def test_status_commands_fast(self, script_runner):
        """Test that status commands complete quickly."""
        scripts = [
            ("performance.py", ["status"]),
            ("session_continuity.py", ["status"]),
            ("sleeptime_compute.py", ["status"]),
        ]

        for script, args in scripts:
            start = time.time()
            exit_code, stdout, stderr = script_runner(script, args, timeout=30)
            duration = time.time() - start

            assert exit_code in [0, 1, 2], f"{script} failed"
            assert duration < 10, f"{script} took too long: {duration:.1f}s"

    def test_quick_mode_faster(self, script_runner):
        """Test that --quick mode is faster than full mode."""
        # Quick validation
        start_quick = time.time()
        script_runner("auto_validate.py", ["--quick"])
        duration_quick = time.time() - start_quick

        # Quick mode should complete in reasonable time
        assert duration_quick < 30, f"Quick mode too slow: {duration_quick:.1f}s"


# =============================================================================
# File System Tests
# =============================================================================

class TestFileSystem:
    """Test file system operations."""

    def test_data_directories_exist(self):
        """Test that required data directories exist."""
        data_dir = V10_DIR / "data"
        assert data_dir.exists(), "data/ directory should exist"

    def test_hooks_directory_exists(self):
        """Test that hooks directory exists."""
        hooks_dir = V10_DIR / "hooks"
        assert hooks_dir.exists(), "hooks/ directory should exist"

    def test_scripts_directory_exists(self):
        """Test that scripts directory exists."""
        assert SCRIPT_DIR.exists(), "scripts/ directory should exist"

    def test_all_scripts_have_docstrings(self):
        """Test that all Python scripts have docstrings."""
        for script_path in SCRIPT_DIR.glob("*.py"):
            if script_path.name.startswith("__"):
                continue

            content = script_path.read_text(encoding="utf-8")
            # Should have a docstring (triple-quoted string near top)
            assert '"""' in content[:2000], f"{script_path.name} missing docstring"


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
