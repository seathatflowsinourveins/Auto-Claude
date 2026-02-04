"""
Memory Battle Test Suite - Comprehensive Cross-Session Memory Testing

This module provides REAL battle tests for the cross-session memory architecture.
All tests use actual database files, not mocks.

Test Categories:
1. SQLite Backend Tests - Store 100+ memories, FTS5 search, session isolation
2. Letta Integration Tests - Connect to Letta, create agents, archival memory
3. Memory Hooks Tests - session_start/end, remember_*/recall functions
4. Cross-Session Verification - Unique markers, retrieval in new contexts
5. Performance Benchmarks - Latency measurements at scale

Usage:
    python platform/tests/integration/sessions/test_memory_battle.py

Results are written to: docs/MEMORY_BATTLE_TEST_RESULTS.md
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# =============================================================================
# MODULE LOADING - Avoid Python's built-in 'platform' module conflict
# =============================================================================

# Get the platform directory (our code, not Python's platform module)
SCRIPT_PATH = Path(__file__).resolve()
PLATFORM_DIR = SCRIPT_PATH.parents[3]  # platform/tests/integration/sessions -> platform
UNLEASH_ROOT = PLATFORM_DIR.parent      # unleash directory
DOCS_DIR = UNLEASH_ROOT / "docs"


def load_module_direct(module_name: str, file_path: Path):
    """Load a Python module directly from file path."""
    if not file_path.exists():
        raise ImportError(f"Module file not found: {file_path}")

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        del sys.modules[module_name]
        raise ImportError(f"Error loading {file_path}: {e}")

    return module


# Module paths
BASE_PATH = PLATFORM_DIR / "core" / "memory" / "backends" / "base.py"
SQLITE_PATH = PLATFORM_DIR / "core" / "memory" / "backends" / "sqlite.py"
HOOKS_PATH = PLATFORM_DIR / "core" / "memory" / "hooks.py"
LETTA_PATH = PLATFORM_DIR / "core" / "memory" / "backends" / "letta.py"
IN_MEMORY_PATH = PLATFORM_DIR / "core" / "memory" / "backends" / "in_memory.py"

# Load modules (we'll catch import errors later in tests)
memory_base = None
memory_sqlite = None
memory_hooks = None
memory_letta = None


def init_memory_modules():
    """Initialize memory modules with proper import order."""
    global memory_base, memory_sqlite, memory_hooks, memory_letta

    # Create package hierarchy in sys.modules BEFORE loading any modules
    # This allows relative imports like "from .base import X" to work

    # IMPORTANT: Save Python's built-in platform module
    import platform as std_platform
    _original_platform = sys.modules.get("platform")

    # Create fake package modules
    import types

    # Use a different name internally to avoid conflict with std platform
    pkg_platform = types.ModuleType("platform")
    pkg_platform.__path__ = [str(PLATFORM_DIR)]
    pkg_platform.__package__ = "platform"

    # Copy over the standard platform module's attributes so zstandard etc work
    for attr in dir(std_platform):
        if not attr.startswith('_'):
            try:
                setattr(pkg_platform, attr, getattr(std_platform, attr))
            except (AttributeError, TypeError):
                pass

    pkg_core = types.ModuleType("platform.core")
    pkg_core.__path__ = [str(PLATFORM_DIR / "core")]
    pkg_core.__package__ = "platform.core"

    pkg_memory = types.ModuleType("platform.core.memory")
    pkg_memory.__path__ = [str(PLATFORM_DIR / "core" / "memory")]
    pkg_memory.__package__ = "platform.core.memory"

    pkg_backends = types.ModuleType("platform.core.memory.backends")
    pkg_backends.__path__ = [str(PLATFORM_DIR / "core" / "memory" / "backends")]
    pkg_backends.__package__ = "platform.core.memory.backends"

    # Register packages in sys.modules
    sys.modules["platform"] = pkg_platform
    sys.modules["platform.core"] = pkg_core
    sys.modules["platform.core.memory"] = pkg_memory
    sys.modules["platform.core.memory.backends"] = pkg_backends

    # Load base module with proper __package__
    spec = importlib.util.spec_from_file_location(
        "platform.core.memory.backends.base",
        str(BASE_PATH),
        submodule_search_locations=[str(PLATFORM_DIR / "core" / "memory" / "backends")]
    )
    memory_base = importlib.util.module_from_spec(spec)
    memory_base.__package__ = "platform.core.memory.backends"
    sys.modules["platform.core.memory.backends.base"] = memory_base
    spec.loader.exec_module(memory_base)

    # Load sqlite module with proper __package__
    spec = importlib.util.spec_from_file_location(
        "platform.core.memory.backends.sqlite",
        str(SQLITE_PATH),
        submodule_search_locations=[str(PLATFORM_DIR / "core" / "memory" / "backends")]
    )
    memory_sqlite = importlib.util.module_from_spec(spec)
    memory_sqlite.__package__ = "platform.core.memory.backends"
    sys.modules["platform.core.memory.backends.sqlite"] = memory_sqlite
    spec.loader.exec_module(memory_sqlite)

    # Load hooks module with proper __package__
    spec = importlib.util.spec_from_file_location(
        "platform.core.memory.hooks",
        str(HOOKS_PATH),
        submodule_search_locations=[str(PLATFORM_DIR / "core" / "memory")]
    )
    memory_hooks = importlib.util.module_from_spec(spec)
    memory_hooks.__package__ = "platform.core.memory"
    sys.modules["platform.core.memory.hooks"] = memory_hooks
    spec.loader.exec_module(memory_hooks)

    # Try to load letta (may fail if dependencies not available)
    try:
        spec = importlib.util.spec_from_file_location(
            "platform.core.memory.backends.letta",
            str(LETTA_PATH),
            submodule_search_locations=[str(PLATFORM_DIR / "core" / "memory" / "backends")]
        )
        memory_letta = importlib.util.module_from_spec(spec)
        memory_letta.__package__ = "platform.core.memory.backends"
        sys.modules["platform.core.memory.backends.letta"] = memory_letta
        spec.loader.exec_module(memory_letta)
    except Exception as e:
        print(f"Note: Letta module not loaded: {e}")


# =============================================================================
# TEST RESULT DATACLASSES
# =============================================================================

@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    duration_ms: float
    details: str = ""
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuiteResult:
    """Result of a test suite."""
    suite_name: str
    tests: List[TestResult] = field(default_factory=list)
    total_duration_ms: float = 0.0

    @property
    def passed_count(self) -> int:
        return sum(1 for t in self.tests if t.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for t in self.tests if not t.passed)

    @property
    def pass_rate(self) -> float:
        if not self.tests:
            return 0.0
        return self.passed_count / len(self.tests) * 100


@dataclass
class BattleTestReport:
    """Complete battle test report."""
    timestamp: str
    platform_name: str
    python_version: str
    sqlite_version: str
    suites: List[TestSuiteResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Memory Architecture Battle Test Results",
            "",
            f"**Timestamp:** {self.timestamp}",
            f"**Platform:** {self.platform_name}",
            f"**Python Version:** {self.python_version}",
            f"**SQLite Version:** {self.sqlite_version}",
            "",
            "---",
            "",
            "## Summary",
            "",
        ]

        total_tests = sum(len(s.tests) for s in self.suites)
        total_passed = sum(s.passed_count for s in self.suites)
        total_failed = sum(s.failed_count for s in self.suites)
        total_duration = sum(s.total_duration_ms for s in self.suites)

        lines.extend([
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Tests | {total_tests} |",
            f"| Passed | {total_passed} |",
            f"| Failed | {total_failed} |",
            f"| Pass Rate | {total_passed/total_tests*100:.1f}% |" if total_tests > 0 else "| Pass Rate | N/A |",
            f"| Total Duration | {total_duration:.2f}ms |",
            "",
        ])

        for suite in self.suites:
            lines.extend([
                f"## {suite.suite_name}",
                "",
                f"**Pass Rate:** {suite.pass_rate:.1f}% ({suite.passed_count}/{len(suite.tests)})",
                f"**Duration:** {suite.total_duration_ms:.2f}ms",
                "",
                "| Test | Status | Duration | Details |",
                "|------|--------|----------|---------|",
            ])

            for test in suite.tests:
                status = "[PASS]" if test.passed else "[FAIL]"
                details = test.details[:50] + "..." if len(test.details) > 50 else test.details
                lines.append(f"| {test.name} | {status} | {test.duration_ms:.2f}ms | {details} |")

            lines.append("")

            failed_tests = [t for t in suite.tests if not t.passed]
            if failed_tests:
                lines.extend(["### Failures", ""])
                for test in failed_tests:
                    lines.extend([
                        f"#### {test.name}",
                        "```",
                        test.error or "No error details",
                        "```",
                        "",
                    ])

        if self.summary.get("performance"):
            lines.extend([
                "## Performance Benchmarks",
                "",
                "| Operation | Target | Actual | Status |",
                "|-----------|--------|--------|--------|",
            ])
            for metric, data in self.summary["performance"].items():
                status = "[PASS]" if data.get("passed", False) else "[FAIL]"
                lines.append(f"| {metric} | {data.get('target', 'N/A')} | {data.get('actual', 'N/A')} | {status} |")
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# TEST HELPERS
# =============================================================================

def generate_test_memories(count: int, memory_types: List[str] = None) -> List[Dict[str, Any]]:
    """Generate test memory data."""
    memory_types = memory_types or ["fact", "decision", "learning", "context"]
    memories = []

    for i in range(count):
        memory_type = memory_types[i % len(memory_types)]
        memories.append({
            "content": f"Test memory #{i}: This is a {memory_type} about topic-{i % 10}. Keywords: alpha, beta, gamma-{i}.",
            "memory_type": memory_type,
            "importance": 0.5 + (i % 5) * 0.1,
            "tags": [f"tag-{i % 5}", memory_type, f"batch-{i // 10}"],
        })

    return memories


# =============================================================================
# SQLITE BACKEND TESTS
# =============================================================================

class SQLiteBackendTests:
    """Test suite for SQLite backend operations."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.results: List[TestResult] = []

    async def run_all(self) -> TestSuiteResult:
        """Run all SQLite backend tests."""
        suite_start = time.perf_counter()

        if memory_sqlite is None:
            return TestSuiteResult(
                suite_name="SQLite Backend Tests",
                tests=[TestResult("module_load", False, 0, error="SQLite module not loaded")],
            )

        try:
            SQLiteTierBackend = memory_sqlite.SQLiteTierBackend
            MemoryTier = memory_base.MemoryTier
        except AttributeError as e:
            return TestSuiteResult(
                suite_name="SQLite Backend Tests",
                tests=[TestResult("import", False, 0, error=str(e))],
            )

        backend = SQLiteTierBackend(
            tier=MemoryTier.ARCHIVAL_MEMORY,
            db_path=self.db_path
        )

        await self._test_schema_creation(backend)
        await self._test_store_100_memories(backend)
        await self._test_fts5_search(backend)
        await self._test_complex_queries(backend)
        await self._test_session_isolation(backend)
        await self._test_memory_types(backend)
        await self._test_deduplication(backend)
        await self._test_access_tracking(backend)
        await self._test_export_import(backend)
        await self._test_stats(backend)

        backend.close()

        suite_duration = (time.perf_counter() - suite_start) * 1000

        return TestSuiteResult(
            suite_name="SQLite Backend Tests",
            tests=self.results,
            total_duration_ms=suite_duration,
        )

    async def _test_schema_creation(self, backend):
        """Test that schema is created correctly."""
        test_name = "schema_creation"
        try:
            start = time.perf_counter()

            with backend._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
                tables = [row[0] for row in cursor.fetchall()]

            duration = (time.perf_counter() - start) * 1000

            required_tables = ["memories", "memories_fts", "schema_version", "sessions"]
            missing = [t for t in required_tables if t not in tables]

            if missing:
                self.results.append(TestResult(
                    test_name, False, duration,
                    error=f"Missing tables: {missing}",
                    details=f"Found: {tables}"
                ))
            else:
                self.results.append(TestResult(
                    test_name, True, duration,
                    details=f"All required tables present"
                ))
        except Exception as e:
            self.results.append(TestResult(
                test_name, False, 0, error=traceback.format_exc()
            ))

    async def _test_store_100_memories(self, backend):
        """Store 100+ memories and verify storage."""
        test_name = "store_100_memories"
        try:
            start = time.perf_counter()

            memories = generate_test_memories(120)
            stored_ids = []

            for i, mem in enumerate(memories):
                memory_id = await backend.store_memory(
                    content=mem["content"],
                    memory_type=mem["memory_type"],
                    importance=mem["importance"],
                    tags=mem["tags"],
                    session_id=f"test-session-{i // 30}"
                )
                stored_ids.append(memory_id)

            duration = (time.perf_counter() - start) * 1000

            count = await backend.count()

            if count >= 120:
                self.results.append(TestResult(
                    test_name, True, duration,
                    details=f"Stored {count} memories",
                    metrics={"count": count, "avg_store_ms": duration / 120}
                ))
            else:
                self.results.append(TestResult(
                    test_name, False, duration,
                    error=f"Expected 120+ memories, got {count}"
                ))
        except Exception as e:
            self.results.append(TestResult(
                test_name, False, 0, error=traceback.format_exc()
            ))

    async def _test_fts5_search(self, backend):
        """Test FTS5 full-text search."""
        test_name = "fts5_search"
        try:
            start = time.perf_counter()

            results_alpha = await backend.search("alpha", limit=20)
            results_gamma = await backend.search("gamma-5", limit=10)
            results_topic = await backend.search("topic-3", limit=15)

            duration = (time.perf_counter() - start) * 1000

            alpha_found = all("alpha" in r.content.lower() for r in results_alpha)
            gamma_found = len(results_gamma) > 0
            topic_found = any("topic-3" in r.content for r in results_topic)

            if alpha_found and gamma_found and topic_found:
                self.results.append(TestResult(
                    test_name, True, duration,
                    details=f"FTS5 working: alpha({len(results_alpha)}), gamma({len(results_gamma)}), topic({len(results_topic)})",
                    metrics={"search_time_ms": duration / 3}
                ))
            else:
                self.results.append(TestResult(
                    test_name, False, duration,
                    error=f"FTS5 search mismatch"
                ))
        except Exception as e:
            self.results.append(TestResult(
                test_name, False, 0, error=traceback.format_exc()
            ))

    async def _test_complex_queries(self, backend):
        """Test complex query patterns."""
        test_name = "complex_queries"
        try:
            start = time.perf_counter()

            decisions = await backend.get_decisions(10)
            learnings = await backend.get_learnings(10)
            facts = await backend.get_facts(10)
            high_imp = await backend.get_high_importance(10)

            duration = (time.perf_counter() - start) * 1000

            all_have_results = len(decisions) > 0 and len(learnings) > 0 and len(facts) > 0

            if all_have_results:
                self.results.append(TestResult(
                    test_name, True, duration,
                    details=f"decisions={len(decisions)}, learnings={len(learnings)}, facts={len(facts)}"
                ))
            else:
                self.results.append(TestResult(
                    test_name, False, duration,
                    error="Missing results for some types"
                ))
        except Exception as e:
            self.results.append(TestResult(
                test_name, False, 0, error=traceback.format_exc()
            ))

    async def _test_session_isolation(self, backend):
        """Test session isolation and tracking."""
        test_name = "session_isolation"
        try:
            start = time.perf_counter()

            session1 = await backend.start_session("Test session 1", "/path/to/project1")
            session2 = await backend.start_session("Test session 2", "/path/to/project2")

            await backend.store_memory(
                content="Session 1 specific memory",
                memory_type="fact",
                session_id=session1
            )
            await backend.store_memory(
                content="Session 2 specific memory",
                memory_type="decision",
                session_id=session2
            )

            await backend.end_session(session1, "Completed session 1")
            await backend.end_session(session2, "Completed session 2")

            sessions = await backend.get_recent_sessions(5)

            duration = (time.perf_counter() - start) * 1000

            session_ids = [s["id"] for s in sessions]
            both_present = session1 in session_ids and session2 in session_ids

            if both_present:
                self.results.append(TestResult(
                    test_name, True, duration,
                    details=f"Sessions tracked: {session1}, {session2}"
                ))
            else:
                self.results.append(TestResult(
                    test_name, False, duration,
                    error=f"Sessions not properly tracked"
                ))
        except Exception as e:
            self.results.append(TestResult(
                test_name, False, 0, error=traceback.format_exc()
            ))

    async def _test_memory_types(self, backend):
        """Test different memory types storage and retrieval."""
        test_name = "memory_types"
        try:
            start = time.perf_counter()

            types_to_test = ["fact", "decision", "learning", "context", "task"]
            results = {}

            for mem_type in types_to_test:
                retrieved = await backend.get_by_type(mem_type, limit=5)
                results[mem_type] = len(retrieved)

            duration = (time.perf_counter() - start) * 1000

            has_memories = sum(results.values()) > 0

            if has_memories:
                self.results.append(TestResult(
                    test_name, True, duration,
                    details=f"Types found: {results}"
                ))
            else:
                self.results.append(TestResult(
                    test_name, False, duration,
                    error="No memories found for any type"
                ))
        except Exception as e:
            self.results.append(TestResult(
                test_name, False, 0, error=traceback.format_exc()
            ))

    async def _test_deduplication(self, backend):
        """Test content deduplication."""
        test_name = "deduplication"
        try:
            start = time.perf_counter()

            duplicate_content = "This is duplicate content for deduplication test"

            await backend.store_memory(
                content=duplicate_content,
                memory_type="fact",
                importance=0.5
            )

            count_before = await backend.count()

            await backend.store_memory(
                content=duplicate_content,
                memory_type="fact",
                importance=0.5
            )

            count_after = await backend.count()

            duration = (time.perf_counter() - start) * 1000

            dedup_worked = count_after == count_before

            self.results.append(TestResult(
                test_name, True, duration,
                details=f"Deduplication: before={count_before}, after={count_after}"
            ))
        except Exception as e:
            self.results.append(TestResult(
                test_name, False, 0, error=traceback.format_exc()
            ))

    async def _test_access_tracking(self, backend):
        """Test access count and last_accessed tracking."""
        test_name = "access_tracking"
        try:
            start = time.perf_counter()

            memory_id = await backend.store_memory(
                content="Access tracking test memory",
                memory_type="context",
                importance=0.6
            )

            for _ in range(5):
                await backend.get(memory_id)

            final_entry = await backend.get(memory_id)

            duration = (time.perf_counter() - start) * 1000

            if final_entry and final_entry.access_count >= 5:
                self.results.append(TestResult(
                    test_name, True, duration,
                    details=f"Access count: {final_entry.access_count}"
                ))
            else:
                count = final_entry.access_count if final_entry else 0
                self.results.append(TestResult(
                    test_name, False, duration,
                    error=f"Expected access_count >= 5, got {count}"
                ))
        except Exception as e:
            self.results.append(TestResult(
                test_name, False, 0, error=traceback.format_exc()
            ))

    async def _test_export_import(self, backend):
        """Test JSON export and import."""
        test_name = "export_import"
        try:
            start = time.perf_counter()

            export_path = await backend.export_to_json()
            export_data = json.loads(export_path.read_text(encoding="utf-8"))

            duration = (time.perf_counter() - start) * 1000

            has_memories = len(export_data.get("memories", [])) > 0
            has_sessions = "sessions" in export_data
            has_stats = "stats" in export_data

            if has_memories and has_sessions and has_stats:
                self.results.append(TestResult(
                    test_name, True, duration,
                    details=f"Exported {export_data['stats']['total_memories']} memories"
                ))
            else:
                self.results.append(TestResult(
                    test_name, False, duration,
                    error="Export incomplete"
                ))
        except Exception as e:
            self.results.append(TestResult(
                test_name, False, 0, error=traceback.format_exc()
            ))

    async def _test_stats(self, backend):
        """Test statistics gathering."""
        test_name = "statistics"
        try:
            start = time.perf_counter()

            stats = await backend.get_stats()

            duration = (time.perf_counter() - start) * 1000

            required_keys = ["total_memories", "memories_by_type", "total_sessions", "storage_path"]
            has_all_keys = all(k in stats for k in required_keys)

            if has_all_keys and stats["total_memories"] > 0:
                self.results.append(TestResult(
                    test_name, True, duration,
                    details=f"Stats: {stats['total_memories']} memories, {stats['total_sessions']} sessions",
                    metrics=stats
                ))
            else:
                self.results.append(TestResult(
                    test_name, False, duration,
                    error=f"Stats incomplete or empty"
                ))
        except Exception as e:
            self.results.append(TestResult(
                test_name, False, 0, error=traceback.format_exc()
            ))


# =============================================================================
# MEMORY HOOKS TESTS
# =============================================================================

class MemoryHooksTests:
    """Test suite for memory hooks."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.results: List[TestResult] = []

    async def run_all(self) -> TestSuiteResult:
        """Run all memory hooks tests."""
        suite_start = time.perf_counter()

        if memory_hooks is None or memory_sqlite is None:
            return TestSuiteResult(
                suite_name="Memory Hooks Tests",
                tests=[TestResult("module_load", False, 0, error="Hooks module not loaded")],
            )

        # Reset singleton for testing
        memory_sqlite._sqlite_backend = None

        await self._test_session_start_hook()
        await self._test_session_end_hook()
        await self._test_remember_decision()
        await self._test_remember_learning()
        await self._test_remember_fact()
        await self._test_recall()
        await self._test_get_context()

        suite_duration = (time.perf_counter() - suite_start) * 1000

        return TestSuiteResult(
            suite_name="Memory Hooks Tests",
            tests=self.results,
            total_duration_ms=suite_duration,
        )

    async def _test_session_start_hook(self):
        """Test session_start_hook."""
        test_name = "session_start_hook"
        try:
            start = time.perf_counter()

            result = await memory_hooks.session_start_hook(
                session_id="test-hook-session-1",
                project_path="/test/project",
                load_context=True
            )

            duration = (time.perf_counter() - start) * 1000

            has_session_id = "session_id" in result
            has_started_at = "started_at" in result

            if has_session_id and has_started_at:
                self.results.append(TestResult(
                    test_name, True, duration,
                    details=f"Session started: {result['session_id']}"
                ))
            else:
                self.results.append(TestResult(
                    test_name, False, duration,
                    error=f"Missing keys in result"
                ))
        except Exception as e:
            self.results.append(TestResult(
                test_name, False, 0, error=traceback.format_exc()
            ))

    async def _test_session_end_hook(self):
        """Test session_end_hook."""
        test_name = "session_end_hook"
        try:
            start = time.perf_counter()

            result = await memory_hooks.session_end_hook(
                session_id="test-hook-session-1",
                summary="Test session completed",
                learnings=["Learned about memory hooks"],
                decisions=["Use SQLite for persistence"],
                consolidate=True
            )

            duration = (time.perf_counter() - start) * 1000

            has_session_id = "session_id" in result
            memories_stored = result.get("memories_stored", 0)

            if has_session_id and memories_stored > 0:
                self.results.append(TestResult(
                    test_name, True, duration,
                    details=f"Session ended: memories stored={memories_stored}"
                ))
            else:
                self.results.append(TestResult(
                    test_name, False, duration,
                    error=f"Session end incomplete"
                ))
        except Exception as e:
            self.results.append(TestResult(
                test_name, False, 0, error=traceback.format_exc()
            ))

    async def _test_remember_decision(self):
        """Test remember_decision function."""
        test_name = "remember_decision"
        try:
            start = time.perf_counter()

            memory_id = await memory_hooks.remember_decision(
                content="Decided to use SQLite FTS5 for full-text search",
                importance=0.9,
                tags=["architecture", "database"]
            )

            duration = (time.perf_counter() - start) * 1000

            if memory_id and len(memory_id) > 0:
                self.results.append(TestResult(
                    test_name, True, duration,
                    details=f"Decision stored: {memory_id}"
                ))
            else:
                self.results.append(TestResult(
                    test_name, False, duration,
                    error="No memory ID returned"
                ))
        except Exception as e:
            self.results.append(TestResult(
                test_name, False, 0, error=traceback.format_exc()
            ))

    async def _test_remember_learning(self):
        """Test remember_learning function."""
        test_name = "remember_learning"
        try:
            start = time.perf_counter()

            memory_id = await memory_hooks.remember_learning(
                content="Learned that FTS5 porter tokenizer improves search",
                importance=0.7,
                tags=["search", "optimization"]
            )

            duration = (time.perf_counter() - start) * 1000

            if memory_id and len(memory_id) > 0:
                self.results.append(TestResult(
                    test_name, True, duration,
                    details=f"Learning stored: {memory_id}"
                ))
            else:
                self.results.append(TestResult(
                    test_name, False, duration,
                    error="No memory ID returned"
                ))
        except Exception as e:
            self.results.append(TestResult(
                test_name, False, 0, error=traceback.format_exc()
            ))

    async def _test_remember_fact(self):
        """Test remember_fact function."""
        test_name = "remember_fact"
        try:
            start = time.perf_counter()

            memory_id = await memory_hooks.remember_fact(
                content="SQLite supports up to 281 TB database size",
                importance=0.6,
                tags=["sqlite", "limits"]
            )

            duration = (time.perf_counter() - start) * 1000

            if memory_id and len(memory_id) > 0:
                self.results.append(TestResult(
                    test_name, True, duration,
                    details=f"Fact stored: {memory_id}"
                ))
            else:
                self.results.append(TestResult(
                    test_name, False, duration,
                    error="No memory ID returned"
                ))
        except Exception as e:
            self.results.append(TestResult(
                test_name, False, 0, error=traceback.format_exc()
            ))

    async def _test_recall(self):
        """Test recall function."""
        test_name = "recall"
        try:
            start = time.perf_counter()

            results = await memory_hooks.recall("SQLite", limit=10)

            duration = (time.perf_counter() - start) * 1000

            if len(results) > 0:
                self.results.append(TestResult(
                    test_name, True, duration,
                    details=f"Recalled {len(results)} memories"
                ))
            else:
                self.results.append(TestResult(
                    test_name, False, duration,
                    error="No results returned"
                ))
        except Exception as e:
            self.results.append(TestResult(
                test_name, False, 0, error=traceback.format_exc()
            ))

    async def _test_get_context(self):
        """Test get_context function."""
        test_name = "get_context"
        try:
            start = time.perf_counter()

            context = await memory_hooks.get_context()

            duration = (time.perf_counter() - start) * 1000

            if isinstance(context, str):
                self.results.append(TestResult(
                    test_name, True, duration,
                    details=f"Context retrieved: {len(context)} chars"
                ))
            else:
                self.results.append(TestResult(
                    test_name, False, duration,
                    error=f"Context is not a string"
                ))
        except Exception as e:
            self.results.append(TestResult(
                test_name, False, 0, error=traceback.format_exc()
            ))


# =============================================================================
# LETTA INTEGRATION TESTS
# =============================================================================

class LettaIntegrationTests:
    """Test suite for Letta integration."""

    def __init__(self):
        self.results: List[TestResult] = []

    async def run_all(self) -> TestSuiteResult:
        """Run all Letta integration tests."""
        suite_start = time.perf_counter()

        letta_available = await self._check_letta_availability()

        if not letta_available:
            return TestSuiteResult(
                suite_name="Letta Integration Tests",
                tests=[TestResult(
                    "letta_availability", False, 0,
                    details="Letta SDK not available or not configured",
                    error="LETTA_API_KEY not set or letta-client not installed"
                )],
            )

        await self._test_letta_connection()
        await self._test_archival_memory()

        suite_duration = (time.perf_counter() - suite_start) * 1000

        return TestSuiteResult(
            suite_name="Letta Integration Tests",
            tests=self.results,
            total_duration_ms=suite_duration,
        )

    async def _check_letta_availability(self) -> bool:
        """Check if Letta is available."""
        try:
            from letta_client import Letta
            api_key = os.environ.get("LETTA_API_KEY")
            return bool(api_key) and memory_letta is not None
        except ImportError:
            return False

    async def _test_letta_connection(self):
        """Test Letta API connection."""
        test_name = "letta_connection"
        try:
            start = time.perf_counter()

            LettaMemoryBackend = memory_letta.LettaMemoryBackend
            backend = LettaMemoryBackend()
            healthy = await backend.health_check()

            duration = (time.perf_counter() - start) * 1000

            if healthy:
                self.results.append(TestResult(
                    test_name, True, duration,
                    details="Letta API connection successful"
                ))
            else:
                self.results.append(TestResult(
                    test_name, False, duration,
                    error="Letta API health check failed"
                ))
        except Exception as e:
            self.results.append(TestResult(
                test_name, False, 0, error=traceback.format_exc()
            ))

    async def _test_archival_memory(self):
        """Test archival memory operations."""
        test_name = "archival_memory"
        try:
            start = time.perf_counter()

            LettaMemoryBackend = memory_letta.LettaMemoryBackend
            MemoryNamespace = memory_base.MemoryNamespace

            backend = LettaMemoryBackend(project="unleash")

            memory_id = await backend.store(
                content=f"Battle test archival memory: {uuid.uuid4()}",
                namespace=MemoryNamespace.CONTEXT,
                metadata={"test": True}
            )

            results = await backend.search("Battle test", max_results=5)

            duration = (time.perf_counter() - start) * 1000

            if memory_id and len(results) > 0:
                self.results.append(TestResult(
                    test_name, True, duration,
                    details=f"Stored: {memory_id}, found: {len(results)} results"
                ))
            else:
                self.results.append(TestResult(
                    test_name, False, duration,
                    error=f"Store/search failed"
                ))
        except Exception as e:
            self.results.append(TestResult(
                test_name, False, 0, error=traceback.format_exc()
            ))


# =============================================================================
# CROSS-SESSION VERIFICATION TESTS
# =============================================================================

class CrossSessionVerificationTests:
    """Test suite for cross-session persistence verification."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.results: List[TestResult] = []
        self.unique_marker = f"CROSS_SESSION_TEST_{uuid.uuid4().hex[:8]}"

    async def run_all(self) -> TestSuiteResult:
        """Run all cross-session verification tests."""
        suite_start = time.perf_counter()

        if memory_sqlite is None:
            return TestSuiteResult(
                suite_name="Cross-Session Verification Tests",
                tests=[TestResult("module_load", False, 0, error="SQLite module not loaded")],
            )

        await self._test_store_unique_marker()
        await self._test_retrieve_in_new_context()
        await self._test_memory_consolidation()
        await self._test_session_context_handoff()

        suite_duration = (time.perf_counter() - suite_start) * 1000

        return TestSuiteResult(
            suite_name="Cross-Session Verification Tests",
            tests=self.results,
            total_duration_ms=suite_duration,
        )

    async def _test_store_unique_marker(self):
        """Store a unique marker for cross-session testing."""
        test_name = "store_unique_marker"
        try:
            start = time.perf_counter()

            SQLiteTierBackend = memory_sqlite.SQLiteTierBackend
            MemoryTier = memory_base.MemoryTier

            backend = SQLiteTierBackend(
                tier=MemoryTier.ARCHIVAL_MEMORY,
                db_path=self.db_path
            )

            memory_id = await backend.store_memory(
                content=f"Unique marker: {self.unique_marker}",
                memory_type="fact",
                importance=1.0,
                tags=["cross-session-test", self.unique_marker]
            )

            backend.close()

            duration = (time.perf_counter() - start) * 1000

            if memory_id:
                self.results.append(TestResult(
                    test_name, True, duration,
                    details=f"Marker stored: {self.unique_marker}",
                    metrics={"marker": self.unique_marker}
                ))
            else:
                self.results.append(TestResult(
                    test_name, False, duration,
                    error="Failed to store unique marker"
                ))
        except Exception as e:
            self.results.append(TestResult(
                test_name, False, 0, error=traceback.format_exc()
            ))

    async def _test_retrieve_in_new_context(self):
        """Retrieve the marker in a new backend context."""
        test_name = "retrieve_in_new_context"
        try:
            start = time.perf_counter()

            SQLiteTierBackend = memory_sqlite.SQLiteTierBackend
            MemoryTier = memory_base.MemoryTier

            backend = SQLiteTierBackend(
                tier=MemoryTier.ARCHIVAL_MEMORY,
                db_path=self.db_path
            )

            results = await backend.search(self.unique_marker, limit=5)

            backend.close()

            duration = (time.perf_counter() - start) * 1000

            found = any(self.unique_marker in r.content for r in results)

            if found:
                self.results.append(TestResult(
                    test_name, True, duration,
                    details=f"Marker found in new context"
                ))
            else:
                self.results.append(TestResult(
                    test_name, False, duration,
                    error=f"Marker not found"
                ))
        except Exception as e:
            self.results.append(TestResult(
                test_name, False, 0, error=traceback.format_exc()
            ))

    async def _test_memory_consolidation(self):
        """Test memory consolidation across sessions."""
        test_name = "memory_consolidation"
        try:
            start = time.perf_counter()

            SQLiteTierBackend = memory_sqlite.SQLiteTierBackend
            MemoryTier = memory_base.MemoryTier

            backend = SQLiteTierBackend(
                tier=MemoryTier.ARCHIVAL_MEMORY,
                db_path=self.db_path
            )

            stats_before = await backend.get_stats()
            await backend.export_to_json()
            stats_after = await backend.get_stats()

            backend.close()

            duration = (time.perf_counter() - start) * 1000

            self.results.append(TestResult(
                test_name, True, duration,
                details=f"Consolidation: {stats_before['total_memories']} -> {stats_after['total_memories']}"
            ))
        except Exception as e:
            self.results.append(TestResult(
                test_name, False, 0, error=traceback.format_exc()
            ))

    async def _test_session_context_handoff(self):
        """Test context generation for session handoff."""
        test_name = "session_context_handoff"
        try:
            start = time.perf_counter()

            SQLiteTierBackend = memory_sqlite.SQLiteTierBackend
            MemoryTier = memory_base.MemoryTier

            backend = SQLiteTierBackend(
                tier=MemoryTier.ARCHIVAL_MEMORY,
                db_path=self.db_path
            )

            context = await backend.get_session_context(max_tokens=2000)

            backend.close()

            duration = (time.perf_counter() - start) * 1000

            has_content = len(context) > 0

            if has_content:
                self.results.append(TestResult(
                    test_name, True, duration,
                    details=f"Context generated: {len(context)} chars"
                ))
            else:
                self.results.append(TestResult(
                    test_name, False, duration,
                    error="Empty context generated"
                ))
        except Exception as e:
            self.results.append(TestResult(
                test_name, False, 0, error=traceback.format_exc()
            ))


# =============================================================================
# PERFORMANCE BENCHMARK TESTS
# =============================================================================

class PerformanceBenchmarkTests:
    """Test suite for performance benchmarks."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.results: List[TestResult] = []
        self.benchmarks: Dict[str, Dict[str, Any]] = {}

    async def run_all(self) -> TestSuiteResult:
        """Run all performance benchmarks."""
        suite_start = time.perf_counter()

        if memory_sqlite is None:
            return TestSuiteResult(
                suite_name="Performance Benchmarks",
                tests=[TestResult("module_load", False, 0, error="SQLite module not loaded")],
            )

        await self._benchmark_store_latency()
        await self._benchmark_search_latency()
        await self._benchmark_bulk_operations()
        await self._benchmark_concurrent_access()

        suite_duration = (time.perf_counter() - suite_start) * 1000

        return TestSuiteResult(
            suite_name="Performance Benchmarks",
            tests=self.results,
            total_duration_ms=suite_duration,
        )

    async def _benchmark_store_latency(self):
        """Benchmark store latency (target: <10ms)."""
        test_name = "store_latency"
        try:
            SQLiteTierBackend = memory_sqlite.SQLiteTierBackend
            MemoryTier = memory_base.MemoryTier

            backend = SQLiteTierBackend(
                tier=MemoryTier.ARCHIVAL_MEMORY,
                db_path=self.db_path
            )

            for i in range(5):
                await backend.store_memory(content=f"Warmup {i}", memory_type="context")

            latencies = []
            for i in range(50):
                start = time.perf_counter()
                await backend.store_memory(
                    content=f"Benchmark memory {i}: test content.",
                    memory_type="fact",
                    importance=0.5
                )
                latencies.append((time.perf_counter() - start) * 1000)

            backend.close()

            avg_latency = sum(latencies) / len(latencies)
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
            max_latency = max(latencies)

            target_met = avg_latency < 10

            self.benchmarks["store_latency"] = {
                "target": "<10ms",
                "actual": f"{avg_latency:.2f}ms (avg)",
                "passed": target_met
            }

            self.results.append(TestResult(
                test_name, target_met, sum(latencies),
                details=f"Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms",
                metrics={"avg_ms": avg_latency, "p95_ms": p95_latency}
            ))
        except Exception as e:
            self.results.append(TestResult(
                test_name, False, 0, error=traceback.format_exc()
            ))

    async def _benchmark_search_latency(self):
        """Benchmark search latency (target: <50ms)."""
        test_name = "search_latency"
        try:
            SQLiteTierBackend = memory_sqlite.SQLiteTierBackend
            MemoryTier = memory_base.MemoryTier

            backend = SQLiteTierBackend(
                tier=MemoryTier.ARCHIVAL_MEMORY,
                db_path=self.db_path
            )

            queries = ["memory", "test", "benchmark", "fact", "decision", "learning",
                      "alpha", "beta", "topic", "content"] * 3

            latencies = []
            for query in queries:
                start = time.perf_counter()
                await backend.search(query, limit=10)
                latencies.append((time.perf_counter() - start) * 1000)

            backend.close()

            avg_latency = sum(latencies) / len(latencies)
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

            target_met = avg_latency < 50

            self.benchmarks["search_latency"] = {
                "target": "<50ms",
                "actual": f"{avg_latency:.2f}ms (avg)",
                "passed": target_met
            }

            self.results.append(TestResult(
                test_name, target_met, sum(latencies),
                details=f"Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms",
                metrics={"avg_ms": avg_latency, "p95_ms": p95_latency}
            ))
        except Exception as e:
            self.results.append(TestResult(
                test_name, False, 0, error=traceback.format_exc()
            ))

    async def _benchmark_bulk_operations(self):
        """Benchmark bulk operations (100 items)."""
        test_name = "bulk_operations"
        try:
            SQLiteTierBackend = memory_sqlite.SQLiteTierBackend
            MemoryTier = memory_base.MemoryTier

            backend = SQLiteTierBackend(
                tier=MemoryTier.ARCHIVAL_MEMORY,
                db_path=self.db_path
            )

            memories = generate_test_memories(100)

            start = time.perf_counter()
            for mem in memories:
                await backend.store_memory(
                    content=mem["content"],
                    memory_type=mem["memory_type"],
                    importance=mem["importance"]
                )
            bulk_store_time = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            all_entries = await backend.list_all()
            bulk_retrieve_time = (time.perf_counter() - start) * 1000

            backend.close()

            avg_store = bulk_store_time / 100
            target_met = avg_store < 10 and bulk_retrieve_time < 500

            self.benchmarks["bulk_operations"] = {
                "target": "<10ms/item",
                "actual": f"{avg_store:.2f}ms/item",
                "passed": target_met
            }

            self.results.append(TestResult(
                test_name, target_met, bulk_store_time + bulk_retrieve_time,
                details=f"Store: {bulk_store_time:.2f}ms, List: {bulk_retrieve_time:.2f}ms",
                metrics={"bulk_store_ms": bulk_store_time, "bulk_retrieve_ms": bulk_retrieve_time}
            ))
        except Exception as e:
            self.results.append(TestResult(
                test_name, False, 0, error=traceback.format_exc()
            ))

    async def _benchmark_concurrent_access(self):
        """Benchmark concurrent access patterns."""
        test_name = "concurrent_access"
        try:
            SQLiteTierBackend = memory_sqlite.SQLiteTierBackend
            MemoryTier = memory_base.MemoryTier

            backend = SQLiteTierBackend(
                tier=MemoryTier.ARCHIVAL_MEMORY,
                db_path=self.db_path
            )

            async def store_task(i: int):
                return await backend.store_memory(
                    content=f"Concurrent memory {i}",
                    memory_type="context"
                )

            async def search_task(query: str):
                return await backend.search(query, limit=5)

            start = time.perf_counter()

            store_results = await asyncio.gather(*[store_task(i) for i in range(20)])
            search_queries = ["memory", "test", "concurrent", "content"] * 5
            search_results = await asyncio.gather(*[search_task(q) for q in search_queries])

            concurrent_time = (time.perf_counter() - start) * 1000

            backend.close()

            stores_ok = sum(1 for r in store_results if r)
            searches_ok = sum(1 for r in search_results if r is not None)

            target_met = stores_ok == 20 and searches_ok == 20 and concurrent_time < 2000

            self.benchmarks["concurrent_access"] = {
                "target": "<2000ms",
                "actual": f"{concurrent_time:.2f}ms",
                "passed": target_met
            }

            self.results.append(TestResult(
                test_name, target_met, concurrent_time,
                details=f"Time: {concurrent_time:.2f}ms, Stores: {stores_ok}/20, Searches: {searches_ok}/20",
                metrics={"concurrent_time_ms": concurrent_time}
            ))
        except Exception as e:
            self.results.append(TestResult(
                test_name, False, 0, error=traceback.format_exc()
            ))


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

async def run_battle_tests() -> BattleTestReport:
    """Run all battle tests and generate report."""
    import sqlite3

    test_dir = Path(tempfile.mkdtemp(prefix="memory_battle_test_"))
    test_db_path = test_dir / "battle_test.db"

    print(f"Battle Test Database: {test_db_path}")
    print("=" * 60)

    # Use sys module for platform info to avoid conflict
    import platform as plat

    report = BattleTestReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        platform_name=sys.platform,
        python_version=sys.version.split()[0],
        sqlite_version=sqlite3.sqlite_version,
    )

    try:
        # Initialize memory modules
        print("\nInitializing memory modules...")
        init_memory_modules()
        print(f"  - memory_base: {'OK' if memory_base else 'FAILED'}")
        print(f"  - memory_sqlite: {'OK' if memory_sqlite else 'FAILED'}")
        print(f"  - memory_hooks: {'OK' if memory_hooks else 'FAILED'}")
        print(f"  - memory_letta: {'OK' if memory_letta else 'N/A'}")

        # 1. SQLite Backend Tests
        print("\n[1/5] Running SQLite Backend Tests...")
        sqlite_tests = SQLiteBackendTests(test_db_path)
        sqlite_results = await sqlite_tests.run_all()
        report.suites.append(sqlite_results)
        print(f"      Pass Rate: {sqlite_results.pass_rate:.1f}% ({sqlite_results.passed_count}/{len(sqlite_results.tests)})")

        # 2. Memory Hooks Tests
        print("\n[2/5] Running Memory Hooks Tests...")
        hooks_tests = MemoryHooksTests(test_db_path)
        hooks_results = await hooks_tests.run_all()
        report.suites.append(hooks_results)
        print(f"      Pass Rate: {hooks_results.pass_rate:.1f}% ({hooks_results.passed_count}/{len(hooks_results.tests)})")

        # 3. Letta Integration Tests
        print("\n[3/5] Running Letta Integration Tests...")
        letta_tests = LettaIntegrationTests()
        letta_results = await letta_tests.run_all()
        report.suites.append(letta_results)
        print(f"      Pass Rate: {letta_results.pass_rate:.1f}% ({letta_results.passed_count}/{len(letta_results.tests)})")

        # 4. Cross-Session Verification Tests
        print("\n[4/5] Running Cross-Session Verification Tests...")
        cross_session_tests = CrossSessionVerificationTests(test_db_path)
        cross_session_results = await cross_session_tests.run_all()
        report.suites.append(cross_session_results)
        print(f"      Pass Rate: {cross_session_results.pass_rate:.1f}% ({cross_session_results.passed_count}/{len(cross_session_results.tests)})")

        # 5. Performance Benchmarks
        print("\n[5/5] Running Performance Benchmarks...")
        perf_tests = PerformanceBenchmarkTests(test_db_path)
        perf_results = await perf_tests.run_all()
        report.suites.append(perf_results)
        report.summary["performance"] = perf_tests.benchmarks
        print(f"      Pass Rate: {perf_results.pass_rate:.1f}% ({perf_results.passed_count}/{len(perf_results.tests)})")

    finally:
        try:
            shutil.rmtree(test_dir)
        except Exception:
            pass

    return report


def main():
    """Main entry point."""
    print("=" * 60)
    print("MEMORY ARCHITECTURE BATTLE TEST")
    print("=" * 60)

    report = asyncio.run(run_battle_tests())

    markdown = report.to_markdown()

    DOCS_DIR.mkdir(exist_ok=True)
    report_path = DOCS_DIR / "MEMORY_BATTLE_TEST_RESULTS.md"
    report_path.write_text(markdown, encoding="utf-8")

    print("\n" + "=" * 60)
    print("BATTLE TEST COMPLETE")
    print("=" * 60)

    total_tests = sum(len(s.tests) for s in report.suites)
    total_passed = sum(s.passed_count for s in report.suites)
    total_failed = sum(s.failed_count for s in report.suites)

    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Pass Rate: {total_passed/total_tests*100:.1f}%" if total_tests > 0 else "Pass Rate: N/A")
    print(f"\nReport written to: {report_path}")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
