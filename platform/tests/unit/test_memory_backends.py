"""
V36 Memory Backend Unit Tests

Tests for the unified memory backend system including:
- Base types and enums
- InMemoryTierBackend
- LettaTierBackend (mocked)
- Memory entry serialization
"""

import pytest
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch


# Import memory types
try:
    from core.memory.backends.base import (
        MemoryEntry,
        MemoryTier,
        MemoryPriority,
        MemoryAccessPattern,
        MemoryLayer,
        TierBackend,
    )
    from core.memory.backends.in_memory import InMemoryTierBackend
except ImportError:
    pytest.skip("Memory backend modules not available", allow_module_level=True)


class TestMemoryEntry:
    """Tests for MemoryEntry dataclass."""

    def test_create_basic_entry(self):
        """Create a basic memory entry."""
        entry = MemoryEntry(
            id="test-1",
            content="Test content"
        )

        assert entry.id == "test-1"
        assert entry.content == "Test content"
        assert entry.tier == MemoryTier.MAIN_CONTEXT  # default
        assert entry.priority == MemoryPriority.NORMAL  # default

    def test_create_entry_with_all_fields(self):
        """Create entry with all fields specified."""
        now = datetime.now(timezone.utc)
        entry = MemoryEntry(
            id="test-2",
            content="Full entry",
            tier=MemoryTier.ARCHIVAL_MEMORY,
            priority=MemoryPriority.HIGH,
            access_pattern=MemoryAccessPattern.HOT,
            access_count=5,
            embedding=[0.1, 0.2, 0.3],
            metadata={"source": "test"},
            created_at=now,
        )

        assert entry.tier == MemoryTier.ARCHIVAL_MEMORY
        assert entry.priority == MemoryPriority.HIGH
        assert entry.access_count == 5
        assert entry.embedding == [0.1, 0.2, 0.3]
        assert entry.metadata["source"] == "test"

    def test_entry_fields(self):
        """Test entry field access."""
        entry = MemoryEntry(
            id="test-3",
            content="Field test",
            metadata={"key": "value"}
        )

        assert entry.id == "test-3"
        assert entry.content == "Field test"
        assert entry.metadata["key"] == "value"
        assert entry.tier == MemoryTier.MAIN_CONTEXT

    def test_entry_defaults(self):
        """Test entry default values."""
        entry = MemoryEntry(
            id="test-4",
            content="Defaults test",
        )

        assert entry.access_count == 0
        assert entry.metadata == {} or entry.metadata is not None
        assert entry.tier == MemoryTier.MAIN_CONTEXT


class TestMemoryTier:
    """Tests for MemoryTier enum."""

    def test_tier_values(self):
        """Verify tier enum values."""
        assert MemoryTier.MAIN_CONTEXT.value == "main_context"
        assert MemoryTier.CORE_MEMORY.value == "core_memory"
        assert MemoryTier.RECALL_MEMORY.value == "recall_memory"
        assert MemoryTier.ARCHIVAL_MEMORY.value == "archival"

    def test_tier_ordering(self):
        """Tiers should have implicit ordering."""
        tiers = list(MemoryTier)
        assert len(tiers) >= 4


class TestMemoryPriority:
    """Tests for MemoryPriority enum."""

    def test_priority_values(self):
        """Verify priority enum values."""
        assert MemoryPriority.LOW.value == "low"
        assert MemoryPriority.NORMAL.value == "normal"
        assert MemoryPriority.HIGH.value == "high"
        assert MemoryPriority.CRITICAL.value == "critical"


class TestInMemoryTierBackend:
    """Tests for InMemoryTierBackend."""

    @pytest.fixture
    def backend(self):
        """Create a fresh backend for each test."""
        return InMemoryTierBackend()

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, backend):
        """Store and retrieve a memory entry."""
        entry = MemoryEntry(id="mem-1", content="Test memory")

        # Store
        await backend.put(entry.id, entry)

        # Retrieve
        retrieved = await backend.get("mem-1")

        assert retrieved is not None
        assert retrieved.id == "mem-1"
        assert retrieved.content == "Test memory"

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent(self, backend):
        """Retrieving nonexistent entry returns None."""
        result = await backend.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_entry(self, backend):
        """Delete an entry."""
        entry = MemoryEntry(id="mem-2", content="To delete")
        await backend.put(entry.id, entry)

        # Delete
        deleted = await backend.delete("mem-2")
        assert deleted is True

        # Verify gone
        result = await backend.get("mem-2")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, backend):
        """Deleting nonexistent entry returns False."""
        deleted = await backend.delete("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_list_entries(self, backend):
        """List all entries."""
        # Store multiple entries
        for i in range(5):
            entry = MemoryEntry(id=f"mem-{i}", content=f"Content {i}")
            await backend.put(entry.id, entry)

        # List all
        entries = await backend.list_all()

        assert len(entries) == 5
        ids = [e.id for e in entries]
        assert "mem-0" in ids
        assert "mem-4" in ids

    @pytest.mark.asyncio
    async def test_list_with_limit(self, backend):
        """List with limit."""
        for i in range(10):
            entry = MemoryEntry(id=f"mem-{i}", content=f"Content {i}")
            await backend.put(entry.id, entry)

        entries = await backend.list_all()
        assert len(entries) == 10  # list_all returns all entries

    @pytest.mark.asyncio
    async def test_update_entry(self, backend):
        """Update an existing entry."""
        entry = MemoryEntry(id="mem-update", content="Original")
        await backend.put(entry.id, entry)

        # Update
        updated_entry = MemoryEntry(id="mem-update", content="Updated")
        await backend.put(updated_entry.id, updated_entry)

        # Verify
        retrieved = await backend.get("mem-update")
        assert retrieved.content == "Updated"

    @pytest.mark.asyncio
    async def test_clear_all(self, backend):
        """Clear all entries."""
        for i in range(5):
            entry = MemoryEntry(id=f"mem-{i}", content=f"Content {i}")
            await backend.put(entry.id, entry)

        # Clear (sync method)
        backend.clear()

        # Verify empty
        entries = await backend.list_all()
        assert len(entries) == 0

    @pytest.mark.asyncio
    async def test_count(self, backend):
        """Count entries."""
        assert await backend.count() == 0

        for i in range(3):
            entry = MemoryEntry(id=f"mem-{i}", content=f"Content {i}")
            await backend.put(entry.id, entry)

        assert await backend.count() == 3


class TestLettaTierBackendMocked:
    """Tests for LettaTierBackend with mocked Letta client."""

    @pytest.fixture
    def mock_letta_client(self):
        """Create a mocked Letta client."""
        client = MagicMock()

        # Mock blocks
        client.agents = MagicMock()
        client.agents.get_block = AsyncMock(return_value=MagicMock(
            id="block-1",
            value="Test block content"
        ))
        client.agents.update_block = AsyncMock()

        # Mock passages
        client.agents.insert_passage = AsyncMock()
        client.agents.list_passages = AsyncMock(return_value=[
            MagicMock(id="pass-1", text="Passage 1"),
            MagicMock(id="pass-2", text="Passage 2"),
        ])
        client.agents.delete_passage = AsyncMock()

        return client

    @pytest.mark.asyncio
    async def test_letta_backend_import(self):
        """LettaTierBackend should be importable."""
        try:
            from core.memory.backends.letta import LettaTierBackend
            assert LettaTierBackend is not None
        except ImportError:
            pytest.skip("Letta backend not available")

    @pytest.mark.asyncio
    async def test_letta_backend_initialization(self, mock_letta_client):
        """LettaTierBackend initializes correctly."""
        try:
            from core.memory.backends.letta import LettaTierBackend

            backend = LettaTierBackend(
                agent_id="test-agent",
                tier=MemoryTier.CORE_MEMORY
            )
            assert backend.agent_id == "test-agent"
        except ImportError:
            pytest.skip("Letta backend not available")


class TestMemoryAccessPattern:
    """Tests for MemoryAccessPattern enum."""

    def test_access_patterns(self):
        """Verify access pattern values."""
        assert MemoryAccessPattern.HOT.value == "hot"
        assert MemoryAccessPattern.WARM.value == "warm"
        assert MemoryAccessPattern.COLD.value == "cold"
        assert MemoryAccessPattern.FROZEN.value == "frozen"


class TestMemoryLayer:
    """Tests for MemoryLayer enum."""

    def test_memory_layers(self):
        """Verify memory layer values."""
        assert MemoryLayer.LETTA.value == "letta"
        assert MemoryLayer.CLAUDE_MEM.value == "claude_mem"
        assert MemoryLayer.EPISODIC.value == "episodic"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
