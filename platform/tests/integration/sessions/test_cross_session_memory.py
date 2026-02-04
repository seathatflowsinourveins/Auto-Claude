"""
Cross-Session Memory Tests

Tests for memory persistence across sessions, including:
- Memory persistence after session close
- Retrieval of previous session data
- Tier-based persistence
- Namespace isolation
- TTL handling across sessions
"""

import pytest
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any

# Import memory components
try:
    from core.memory.backends.base import MemoryEntry, MemoryTier, MemoryPriority
    from core.memory.backends.in_memory import InMemoryTierBackend
except ImportError:
    pytest.skip("Memory backend modules not available", allow_module_level=True)


class TestMemoryPersistenceAcrossSessions:
    """Tests for memory persistence across sessions."""

    @pytest.fixture
    def memory_entry_factory(self):
        """Factory for creating memory entries."""
        def create(
            id: str,
            content: str,
            tier: MemoryTier = MemoryTier.MAIN_CONTEXT,
            **kwargs
        ) -> MemoryEntry:
            return MemoryEntry(
                id=id,
                content=content,
                tier=tier,
                **kwargs
            )
        return create

    @pytest.mark.asyncio
    async def test_memory_persists_in_backend(self, memory_entry_factory):
        """Memory stored in backend should persist."""
        backend = InMemoryTierBackend()

        # Session 1: Store memory
        entry = memory_entry_factory(
            id="persist-test-1",
            content="Important information from session 1"
        )
        await backend.put(entry.id, entry)

        # Verify stored
        count = await backend.count()
        assert count == 1

        # Session 2: Retrieve (same backend instance simulates persistence)
        retrieved = await backend.get("persist-test-1")
        assert retrieved is not None
        assert retrieved.content == "Important information from session 1"

    @pytest.mark.asyncio
    async def test_multiple_entries_persist(self, memory_entry_factory):
        """Multiple entries should all persist."""
        backend = InMemoryTierBackend()

        # Store multiple entries
        entries = [
            memory_entry_factory(id=f"entry-{i}", content=f"Content {i}")
            for i in range(10)
        ]
        for entry in entries:
            await backend.put(entry.id, entry)

        # Verify all persist
        count = await backend.count()
        assert count == 10

        # Retrieve specific entries
        for i in range(10):
            entry = await backend.get(f"entry-{i}")
            assert entry is not None
            assert entry.content == f"Content {i}"

    @pytest.mark.asyncio
    async def test_entry_updates_persist(self, memory_entry_factory):
        """Entry updates should persist."""
        backend = InMemoryTierBackend()

        # Store initial entry
        entry = memory_entry_factory(id="update-test", content="Original content")
        await backend.put(entry.id, entry)

        # Update entry
        updated = memory_entry_factory(id="update-test", content="Updated content")
        await backend.put(updated.id, updated)

        # Verify update persists
        retrieved = await backend.get("update-test")
        assert retrieved.content == "Updated content"


class TestMemoryTierPersistence:
    """Tests for tier-based memory persistence."""

    @pytest.fixture
    def backend(self):
        return InMemoryTierBackend()

    @pytest.mark.asyncio
    async def test_main_context_tier_persistence(self, backend):
        """Main context tier entries should persist."""
        entry = MemoryEntry(
            id="main-ctx-1",
            content="Main context data",
            tier=MemoryTier.MAIN_CONTEXT
        )
        await backend.put(entry.id, entry)

        retrieved = await backend.get("main-ctx-1")
        assert retrieved is not None
        assert retrieved.tier == MemoryTier.MAIN_CONTEXT

    @pytest.mark.asyncio
    async def test_core_memory_tier_persistence(self, backend):
        """Core memory tier entries should persist."""
        entry = MemoryEntry(
            id="core-mem-1",
            content="Core memory data",
            tier=MemoryTier.CORE_MEMORY
        )
        await backend.put(entry.id, entry)

        retrieved = await backend.get("core-mem-1")
        assert retrieved is not None
        assert retrieved.tier == MemoryTier.CORE_MEMORY

    @pytest.mark.asyncio
    async def test_archival_memory_tier_persistence(self, backend):
        """Archival memory tier entries should persist."""
        entry = MemoryEntry(
            id="archival-1",
            content="Archival data",
            tier=MemoryTier.ARCHIVAL_MEMORY
        )
        await backend.put(entry.id, entry)

        retrieved = await backend.get("archival-1")
        assert retrieved is not None
        assert retrieved.tier == MemoryTier.ARCHIVAL_MEMORY

    @pytest.mark.asyncio
    async def test_recall_memory_tier_persistence(self, backend):
        """Recall memory tier entries should persist."""
        entry = MemoryEntry(
            id="recall-1",
            content="Recall data",
            tier=MemoryTier.RECALL_MEMORY
        )
        await backend.put(entry.id, entry)

        retrieved = await backend.get("recall-1")
        assert retrieved is not None
        assert retrieved.tier == MemoryTier.RECALL_MEMORY

    @pytest.mark.asyncio
    async def test_mixed_tiers_coexist(self, backend):
        """Entries from different tiers should coexist."""
        entries = [
            MemoryEntry(id="t1", content="Main", tier=MemoryTier.MAIN_CONTEXT),
            MemoryEntry(id="t2", content="Core", tier=MemoryTier.CORE_MEMORY),
            MemoryEntry(id="t3", content="Archival", tier=MemoryTier.ARCHIVAL_MEMORY),
            MemoryEntry(id="t4", content="Recall", tier=MemoryTier.RECALL_MEMORY),
        ]

        for entry in entries:
            await backend.put(entry.id, entry)

        # All should be retrievable
        for i, tier in enumerate([MemoryTier.MAIN_CONTEXT, MemoryTier.CORE_MEMORY,
                                   MemoryTier.ARCHIVAL_MEMORY, MemoryTier.RECALL_MEMORY]):
            retrieved = await backend.get(f"t{i+1}")
            assert retrieved is not None
            assert retrieved.tier == tier


class TestMemoryNamespaceIsolation:
    """Tests for memory namespace isolation."""

    @pytest.mark.asyncio
    async def test_different_backends_isolated(self):
        """Different backend instances should be isolated."""
        backend1 = InMemoryTierBackend()
        backend2 = InMemoryTierBackend()

        entry = MemoryEntry(id="isolated-1", content="Backend 1 data")
        await backend1.put(entry.id, entry)

        # Backend 2 should not have the entry
        retrieved = await backend2.get("isolated-1")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_same_id_different_backends(self):
        """Same ID in different backends should be independent."""
        backend1 = InMemoryTierBackend()
        backend2 = InMemoryTierBackend()

        e1 = MemoryEntry(id="shared-id", content="Content from 1")
        await backend1.put(e1.id, e1)
        e2 = MemoryEntry(id="shared-id", content="Content from 2")
        await backend2.put(e2.id, e2)

        retrieved1 = await backend1.get("shared-id")
        retrieved2 = await backend2.get("shared-id")

        assert retrieved1.content == "Content from 1"
        assert retrieved2.content == "Content from 2"


class TestMemoryMetadataPersistence:
    """Tests for memory metadata persistence."""

    @pytest.fixture
    def backend(self):
        return InMemoryTierBackend()

    @pytest.mark.asyncio
    async def test_metadata_persists(self, backend):
        """Entry metadata should persist."""
        entry = MemoryEntry(
            id="meta-1",
            content="Content with metadata",
            metadata={"source": "test", "version": 1, "tags": ["important"]}
        )
        await backend.put(entry.id, entry)

        retrieved = await backend.get("meta-1")
        assert retrieved.metadata["source"] == "test"
        assert retrieved.metadata["version"] == 1
        assert "important" in retrieved.metadata["tags"]

    @pytest.mark.asyncio
    async def test_strength_persists(self, backend):
        """Entry strength should persist."""
        entry = MemoryEntry(
            id="imp-1",
            content="Important content",
            strength=0.95
        )
        await backend.put(entry.id, entry)

        retrieved = await backend.get("imp-1")
        assert retrieved.strength == 0.95

    @pytest.mark.asyncio
    async def test_priority_persists(self, backend):
        """Entry priority should persist."""
        entry = MemoryEntry(
            id="prio-1",
            content="High priority",
            priority=MemoryPriority.CRITICAL
        )
        await backend.put(entry.id, entry)

        retrieved = await backend.get("prio-1")
        assert retrieved.priority == MemoryPriority.CRITICAL

    @pytest.mark.asyncio
    async def test_access_count_persists(self, backend):
        """Access count should persist."""
        entry = MemoryEntry(
            id="access-1",
            content="Accessed content",
            access_count=42
        )
        await backend.put(entry.id, entry)

        retrieved = await backend.get("access-1")
        assert retrieved.access_count >= 42  # get() may increment access_count

    @pytest.mark.asyncio
    async def test_embedding_persists(self, backend):
        """Entry embedding should persist."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        entry = MemoryEntry(
            id="embed-1",
            content="Content with embedding",
            embedding=embedding
        )
        await backend.put(entry.id, entry)

        retrieved = await backend.get("embed-1")
        assert retrieved.embedding == embedding


class TestMemoryListingAcrossSessions:
    """Tests for memory listing across sessions."""

    @pytest.fixture
    def backend(self):
        return InMemoryTierBackend()

    @pytest.mark.asyncio
    async def test_list_all_entries(self, backend):
        """List should return all stored entries."""
        for i in range(5):
            entry = MemoryEntry(id=f"list-{i}", content=f"Content {i}")
            await backend.put(entry.id, entry)

        entries = await backend.list_all()
        assert len(entries) == 5

    @pytest.mark.asyncio
    async def test_list_returns_all(self, backend):
        """List_all should return all entries."""
        for i in range(10):
            entry = MemoryEntry(id=f"limit-{i}", content=f"Content {i}")
            await backend.put(entry.id, entry)

        entries = await backend.list_all()
        assert len(entries) == 10

    @pytest.mark.asyncio
    async def test_list_after_delete(self, backend):
        """List should reflect deletions."""
        for i in range(5):
            entry = MemoryEntry(id=f"del-{i}", content=f"Content {i}")
            await backend.put(entry.id, entry)

        await backend.delete("del-2")

        entries = await backend.list_all()
        assert len(entries) == 4
        ids = [e.id for e in entries]
        assert "del-2" not in ids


class TestMemoryClearAndRecovery:
    """Tests for memory clear and recovery patterns."""

    @pytest.fixture
    def backend(self):
        return InMemoryTierBackend()

    @pytest.mark.asyncio
    async def test_clear_removes_all(self, backend):
        """Clear should remove all entries."""
        for i in range(10):
            entry = MemoryEntry(id=f"clear-{i}", content=f"Content {i}")
            await backend.put(entry.id, entry)

        backend.clear()

        count = await backend.count()
        assert count == 0

    @pytest.mark.asyncio
    async def test_can_store_after_clear(self, backend):
        """Should be able to store after clear."""
        e_before = MemoryEntry(id="before", content="Before clear")
        await backend.put(e_before.id, e_before)
        backend.clear()
        e_after = MemoryEntry(id="after", content="After clear")
        await backend.put(e_after.id, e_after)

        retrieved = await backend.get("after")
        assert retrieved is not None
        assert retrieved.content == "After clear"


class TestMemoryEntryTimestamps:
    """Tests for memory entry timestamps."""

    @pytest.fixture
    def backend(self):
        return InMemoryTierBackend()

    @pytest.mark.asyncio
    async def test_created_at_persists(self, backend):
        """Created timestamp should persist."""
        now = datetime.now(timezone.utc)
        entry = MemoryEntry(
            id="ts-1",
            content="Timestamped",
            created_at=now
        )
        await backend.put(entry.id, entry)

        retrieved = await backend.get("ts-1")
        assert retrieved.created_at == now

    @pytest.mark.asyncio
    async def test_last_accessed_persists(self, backend):
        """Last accessed timestamp should persist."""
        now = datetime.now(timezone.utc)
        entry = MemoryEntry(
            id="ts-2",
            content="Timestamped",
            last_accessed=now
        )
        await backend.put(entry.id, entry)

        retrieved = await backend.get("ts-2")
        assert retrieved.last_accessed >= now  # get() may update last_accessed


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
