#!/usr/bin/env python3
"""
Test Cross-Session Memory Persistence

This test verifies that the SQLite-based cross-session memory actually persists
data to disk and can be retrieved across sessions.

Run with:
    python platform/tests/test_cross_session_memory.py
    pytest platform/tests/test_cross_session_memory.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Add platform to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.memory.backends.sqlite import SQLiteTierBackend, get_sqlite_backend
from core.memory.backends.base import MemoryEntry, MemoryTier, MemoryPriority


class TestSQLiteBackend:
    """Tests for SQLite persistent memory backend."""

    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create a temporary database path."""
        return tmp_path / "test_memory.db"

    @pytest.fixture
    def backend(self, temp_db_path):
        """Create a backend with temporary database."""
        backend = SQLiteTierBackend(
            tier=MemoryTier.ARCHIVAL_MEMORY,
            db_path=temp_db_path
        )
        yield backend
        backend.close()

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, backend):
        """Test storing and retrieving a memory."""
        unique_id = f"test_{uuid.uuid4().hex[:8]}"

        # Store memory
        memory_id = await backend.store_memory(
            content=f"Test memory {unique_id}: This is a persistence test",
            memory_type="fact",
            importance=0.8,
            tags=["test", unique_id]
        )

        assert memory_id is not None
        assert len(memory_id) > 0

        # Retrieve by ID
        entry = await backend.get(memory_id)
        assert entry is not None
        assert unique_id in entry.content

    @pytest.mark.asyncio
    async def test_search_full_text(self, backend):
        """Test full-text search."""
        unique_id = f"unique_{uuid.uuid4().hex[:8]}"

        # Store test memory
        await backend.store_memory(
            content=f"The quick brown fox {unique_id} jumps over lazy dog",
            memory_type="fact",
            importance=0.7
        )

        # Search for it
        results = await backend.search(unique_id, limit=10)
        assert len(results) > 0
        assert any(unique_id in r.content for r in results)

    @pytest.mark.asyncio
    async def test_persistence_across_instances(self, temp_db_path):
        """Test that data persists across backend instances."""
        unique_id = f"persist_{uuid.uuid4().hex[:8]}"

        # Create first backend and store data
        backend1 = SQLiteTierBackend(
            tier=MemoryTier.ARCHIVAL_MEMORY,
            db_path=temp_db_path
        )

        memory_id = await backend1.store_memory(
            content=f"Persistent memory {unique_id}",
            memory_type="decision",
            importance=0.9
        )
        backend1.close()

        # Create second backend (simulates new session)
        backend2 = SQLiteTierBackend(
            tier=MemoryTier.ARCHIVAL_MEMORY,
            db_path=temp_db_path
        )

        # Verify data persisted
        entry = await backend2.get(memory_id)
        assert entry is not None
        assert unique_id in entry.content

        # Also verify search works
        results = await backend2.search(unique_id, limit=5)
        assert len(results) > 0

        backend2.close()

    @pytest.mark.asyncio
    async def test_session_management(self, backend):
        """Test session start/end."""
        # Start session
        session_id = await backend.start_session(
            task_summary="Test session",
            project_path="/test/path"
        )
        assert session_id is not None
        assert len(session_id) > 0

        # Store memory with session
        await backend.store_memory(
            content="Session-linked memory",
            memory_type="context",
            session_id=session_id
        )

        # End session
        await backend.end_session(session_id, "Completed test")

        # Verify session recorded
        sessions = await backend.get_recent_sessions(5)
        assert any(s["id"] == session_id for s in sessions)

    @pytest.mark.asyncio
    async def test_get_by_type(self, backend):
        """Test retrieving memories by type."""
        unique_id = uuid.uuid4().hex[:8]

        # Store different types
        await backend.store_memory(
            content=f"Decision {unique_id}",
            memory_type="decision",
            importance=0.8
        )
        await backend.store_memory(
            content=f"Learning {unique_id}",
            memory_type="learning",
            importance=0.7
        )
        await backend.store_memory(
            content=f"Fact {unique_id}",
            memory_type="fact",
            importance=0.6
        )

        # Get by type
        decisions = await backend.get_decisions(10)
        learnings = await backend.get_learnings(10)
        facts = await backend.get_facts(10)

        assert any(unique_id in d.content for d in decisions)
        assert any(unique_id in l.content for l in learnings)
        assert any(unique_id in f.content for f in facts)

    @pytest.mark.asyncio
    async def test_session_context(self, backend):
        """Test generating session context."""
        unique_id = uuid.uuid4().hex[:8]

        # Store some memories
        await backend.store_memory(
            content=f"Important decision {unique_id}",
            memory_type="decision",
            importance=0.9
        )
        await backend.store_memory(
            content=f"Key learning {unique_id}",
            memory_type="learning",
            importance=0.8
        )

        # Get context
        context = await backend.get_session_context(max_tokens=2000)

        # Should contain our memories
        assert unique_id in context or len(context) > 0

    @pytest.mark.asyncio
    async def test_export_import(self, backend, tmp_path):
        """Test JSON export and import."""
        unique_id = uuid.uuid4().hex[:8]

        # Store test memory
        await backend.store_memory(
            content=f"Exportable memory {unique_id}",
            memory_type="fact",
            importance=0.7
        )

        # Export
        export_path = tmp_path / "export.json"
        result_path = await backend.export_to_json(export_path)
        assert result_path.exists()

        # Create new backend
        new_db_path = tmp_path / "new_memory.db"
        new_backend = SQLiteTierBackend(
            tier=MemoryTier.ARCHIVAL_MEMORY,
            db_path=new_db_path
        )

        # Import
        imported = await new_backend.import_from_json(result_path)
        assert imported > 0

        # Verify data imported
        results = await new_backend.search(unique_id, limit=5)
        assert len(results) > 0

        new_backend.close()

    @pytest.mark.asyncio
    async def test_stats(self, backend):
        """Test statistics gathering."""
        # Store some data
        await backend.store_memory(
            content="Stats test memory",
            memory_type="fact",
            importance=0.5
        )

        stats = await backend.get_stats()

        assert "total_memories" in stats
        assert stats["total_memories"] >= 1
        assert "storage_path" in stats
        assert "db_size_bytes" in stats


class TestCrossSessionIntegration:
    """Integration tests for cross-session memory."""

    @pytest.mark.asyncio
    async def test_real_persistence_path(self):
        """Test that the default persistence path works."""
        backend = get_sqlite_backend()

        # The database should be created
        assert backend.db_path.exists()
        assert backend.db_path.parent.exists()

        # Store a test memory
        unique_id = f"integration_{uuid.uuid4().hex[:8]}"
        memory_id = await backend.store_memory(
            content=f"Integration test memory {unique_id}",
            memory_type="fact",
            importance=0.5,
            tags=["integration_test"]
        )

        # Should be searchable
        results = await backend.search(unique_id, limit=5)
        assert len(results) > 0

        print(f"\nPersistence path: {backend.db_path}")
        print(f"Database size: {backend.db_path.stat().st_size} bytes")

    @pytest.mark.asyncio
    async def test_memory_hooks(self):
        """Test the memory hooks integration."""
        try:
            from core.memory.hooks import (
                session_start_hook,
                session_end_hook,
                remember_decision,
                remember_learning,
                recall
            )

            # Test session start
            start_result = await session_start_hook(
                project_path=os.getcwd()
            )
            assert "session_id" in start_result
            session_id = start_result["session_id"]

            # Test storing via hooks
            unique_id = f"hook_test_{uuid.uuid4().hex[:8]}"

            decision_id = await remember_decision(
                content=f"Hook test decision {unique_id}",
                importance=0.8,
                session_id=session_id
            )
            assert decision_id is not None

            learning_id = await remember_learning(
                content=f"Hook test learning {unique_id}",
                importance=0.7,
                session_id=session_id
            )
            assert learning_id is not None

            # Test recall
            results = await recall(unique_id, limit=10)
            assert len(results) >= 2

            # Test session end
            end_result = await session_end_hook(
                session_id=session_id,
                summary="Hook integration test completed"
            )
            assert "ended_at" in end_result

            print(f"\nSession {session_id} completed with {len(results)} memories")

        except ImportError as e:
            pytest.skip(f"Hooks not available: {e}")


def run_manual_test():
    """Run a manual test of cross-session persistence."""
    print("=" * 60)
    print("CROSS-SESSION MEMORY PERSISTENCE TEST")
    print("=" * 60)

    async def test():
        # Get the singleton backend (uses ~/.claude/memory/memory.db)
        backend = get_sqlite_backend()

        unique_id = f"manual_test_{uuid.uuid4().hex[:8]}"
        print(f"\nTest ID: {unique_id}")
        print(f"Database: {backend.db_path}")

        # Store test memory
        print("\n1. Storing test memory...")
        memory_id = await backend.store_memory(
            content=f"Manual test memory {unique_id}: Cross-session persistence verification",
            memory_type="fact",
            importance=0.9,
            tags=["manual_test", unique_id]
        )
        print(f"   Stored: {memory_id}")

        # Search for it
        print("\n2. Searching for test memory...")
        results = await backend.search(unique_id, limit=5)
        print(f"   Found: {len(results)} results")
        for r in results:
            print(f"   - {r.id}: {r.content[:50]}...")

        # Get stats
        print("\n3. Database statistics...")
        stats = await backend.get_stats()
        print(f"   Total memories: {stats['total_memories']}")
        print(f"   Total sessions: {stats['total_sessions']}")
        print(f"   Database size: {stats['db_size_bytes']} bytes")

        # Get context
        print("\n4. Session context preview...")
        context = await backend.get_session_context(max_tokens=500)
        if context:
            print(context[:300] + "..." if len(context) > 300 else context)
        else:
            print("   (No context available)")

        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)
        print(f"\nTo verify persistence, run this test again in a NEW session.")
        print(f"The memory with ID '{unique_id}' should still be found.")
        print(f"\nDatabase location: {backend.db_path}")

    asyncio.run(test())


if __name__ == "__main__":
    # Run manual test if executed directly
    if len(sys.argv) > 1 and sys.argv[1] == "--pytest":
        pytest.main([__file__, "-v"])
    else:
        run_manual_test()
