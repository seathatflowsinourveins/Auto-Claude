"""
Test Memory Tier Optimization - Quick verification.

Run: python -m pytest platform/core/test_memory_tiers.py -v
"""

import asyncio
import sys
from pathlib import Path

# Ensure the core module is importable
sys.path.insert(0, str(Path(__file__).parent))

import pytest
from memory_tiers import (
    MemoryTier,
    MemoryPriority,
    MemoryAccessPattern,
    MemoryEntry,
    MemoryTierManager,
    MemoryTierIntegration,
    get_tier_manager,
    reset_memory_system,
)


class TestMemoryTiers:
    """Test memory tier functionality."""

    def setup_method(self) -> None:
        """Reset singletons before each test."""
        reset_memory_system()

    @pytest.mark.asyncio
    async def test_basic_remember_recall(self) -> None:
        """Test basic memory operations."""
        manager = MemoryTierManager()

        # Store in main context
        entry = await manager.remember(
            content="Test memory content",
            tier=MemoryTier.MAIN_CONTEXT,
            priority=MemoryPriority.NORMAL
        )

        assert entry.id is not None
        assert entry.content == "Test memory content"
        assert entry.tier == MemoryTier.MAIN_CONTEXT

        # Recall by ID
        recalled = await manager.recall(entry.id)
        assert recalled is not None
        assert recalled.content == entry.content
        assert recalled.access_count == 1  # Touched on recall

    @pytest.mark.asyncio
    async def test_tier_hierarchy(self) -> None:
        """Test storing in different tiers."""
        manager = MemoryTierManager()

        # Store in each tier
        main = await manager.remember("Main context data", tier=MemoryTier.MAIN_CONTEXT)
        core = await manager.remember("Core memory fact", tier=MemoryTier.CORE_MEMORY)
        recall = await manager.remember("Recall conversation", tier=MemoryTier.RECALL_MEMORY)
        archival = await manager.remember("Archival knowledge", tier=MemoryTier.ARCHIVAL_MEMORY)

        assert main.tier == MemoryTier.MAIN_CONTEXT
        assert core.tier == MemoryTier.CORE_MEMORY
        assert recall.tier == MemoryTier.RECALL_MEMORY
        assert archival.tier == MemoryTier.ARCHIVAL_MEMORY

    @pytest.mark.asyncio
    async def test_search_across_tiers(self) -> None:
        """Test searching memories."""
        manager = MemoryTierManager()

        # Store related memories
        await manager.remember("Python programming tips", tier=MemoryTier.MAIN_CONTEXT)
        await manager.remember("Python best practices", tier=MemoryTier.CORE_MEMORY)
        await manager.remember("JavaScript frameworks", tier=MemoryTier.RECALL_MEMORY)

        # Search
        results = await manager.search("Python", limit=5)

        assert len(results) >= 2
        assert all("python" in r.entry.content.lower() for r in results)

    @pytest.mark.asyncio
    async def test_convenience_methods(self) -> None:
        """Test convenience methods for common patterns."""
        manager = MemoryTierManager()

        # Store fact
        fact = await manager.remember_fact("Claude is an AI assistant")
        assert fact.tier == MemoryTier.CORE_MEMORY
        assert fact.priority == MemoryPriority.HIGH

        # Store preference (never evict)
        pref = await manager.remember_preference("User prefers dark mode")
        assert pref.tier == MemoryTier.CORE_MEMORY
        assert pref.priority == MemoryPriority.CRITICAL

        # Store context
        ctx = await manager.remember_context("Working on memory optimization", ttl_minutes=30)
        assert ctx.tier == MemoryTier.MAIN_CONTEXT
        assert "ttl_30m" in ctx.tags

    @pytest.mark.asyncio
    async def test_forget(self) -> None:
        """Test memory deletion."""
        manager = MemoryTierManager()

        entry = await manager.remember("Temporary data")
        entry_id = entry.id

        # Verify exists
        recalled = await manager.recall(entry_id)
        assert recalled is not None

        # Delete
        deleted = await manager.forget(entry_id)
        assert deleted is True

        # Verify gone
        recalled_after = await manager.recall(entry_id)
        assert recalled_after is None

    @pytest.mark.asyncio
    async def test_access_pattern_updates(self) -> None:
        """Test access pattern tracking."""
        manager = MemoryTierManager()

        entry = await manager.remember("Frequently accessed data")

        # Access multiple times
        for _ in range(15):
            await manager.recall(entry.id)

        # Check updated
        updated = await manager.recall(entry.id)
        assert updated is not None
        assert updated.access_count >= 15

    @pytest.mark.asyncio
    async def test_stats(self) -> None:
        """Test statistics tracking."""
        manager = MemoryTierManager()

        # Store some entries
        await manager.remember("Entry 1", tier=MemoryTier.MAIN_CONTEXT)
        await manager.remember("Entry 2", tier=MemoryTier.CORE_MEMORY)
        await manager.remember("Entry 3", tier=MemoryTier.RECALL_MEMORY)

        stats = manager.get_stats()
        assert stats.total_entries == 3
        assert stats.entries_by_tier[MemoryTier.MAIN_CONTEXT] == 1
        assert stats.entries_by_tier[MemoryTier.CORE_MEMORY] == 1
        assert stats.entries_by_tier[MemoryTier.RECALL_MEMORY] == 1

    @pytest.mark.asyncio
    async def test_tier_usage(self) -> None:
        """Test tier usage reporting."""
        manager = MemoryTierManager(
            main_context_tokens=1000,
            core_memory_tokens=500
        )

        await manager.remember("Some content for main context")

        usage = await manager.get_tier_usage()

        assert MemoryTier.MAIN_CONTEXT in usage
        assert "entries" in usage[MemoryTier.MAIN_CONTEXT]
        assert "utilization" in usage[MemoryTier.MAIN_CONTEXT]

    @pytest.mark.asyncio
    async def test_integration_bridge(self) -> None:
        """Test integration bridge functionality."""
        manager = MemoryTierManager()
        integration = MemoryTierIntegration(manager)

        # Store experience (high reward → core memory)
        entry = await integration.store_experience(
            experience_id="exp_001",
            state="Initial state",
            action="Take action",
            result="Positive outcome",
            reward=0.9
        )

        assert entry.tier == MemoryTier.CORE_MEMORY
        assert "experience" in entry.content_type

        # Store low reward experience (→ archival)
        low_entry = await integration.store_experience(
            experience_id="exp_002",
            state="Bad state",
            action="Wrong action",
            result="Negative outcome",
            reward=0.1
        )

        assert low_entry.tier == MemoryTier.ARCHIVAL_MEMORY

    @pytest.mark.asyncio
    async def test_learned_pattern_storage(self) -> None:
        """Test storing learned patterns."""
        manager = MemoryTierManager()
        integration = MemoryTierIntegration(manager)

        entry = await integration.store_learned_pattern(
            pattern_key="api_signature",
            pattern_value={"method": "create", "args": ["content"]},
            confidence=0.95
        )

        assert entry.tier == MemoryTier.CORE_MEMORY
        assert entry.priority == MemoryPriority.HIGH  # High confidence

    @pytest.mark.asyncio
    async def test_singleton_behavior(self) -> None:
        """Test singleton pattern."""
        reset_memory_system()

        manager1 = get_tier_manager()
        manager2 = get_tier_manager()

        assert manager1 is manager2

        reset_memory_system()

        manager3 = get_tier_manager()
        assert manager3 is not manager1


def run_quick_verification() -> None:
    """Run quick verification without pytest."""
    async def verify() -> None:
        print("=" * 60)
        print("Memory Tier Optimization - Quick Verification")
        print("=" * 60)

        reset_memory_system()
        manager = MemoryTierManager()

        # Test 1: Basic operations
        print("\n[1] Testing basic remember/recall...")
        entry = await manager.remember(
            "Test content for verification",
            tier=MemoryTier.MAIN_CONTEXT
        )
        recalled = await manager.recall(entry.id)
        assert recalled is not None
        print(f"    ✓ Stored and recalled: {entry.id}")

        # Test 2: Tier hierarchy
        print("\n[2] Testing tier hierarchy...")
        for tier in MemoryTier:
            e = await manager.remember(f"Content for {tier.value}", tier=tier)
            print(f"    ✓ {tier.value}: {e.id}")

        # Test 3: Search
        print("\n[3] Testing search...")
        await manager.remember("Python programming guide", tier=MemoryTier.CORE_MEMORY)
        await manager.remember("Python async patterns", tier=MemoryTier.RECALL_MEMORY)
        results = await manager.search("Python")
        print(f"    ✓ Found {len(results)} results for 'Python'")

        # Test 4: Stats
        print("\n[4] Testing statistics...")
        stats = manager.get_stats()
        print(f"    ✓ Total entries: {stats.total_entries}")
        for tier, count in stats.entries_by_tier.items():
            if count > 0:
                print(f"      - {tier.value}: {count}")

        # Test 5: Integration
        print("\n[5] Testing integration bridge...")
        integration = MemoryTierIntegration(manager)
        exp = await integration.store_experience(
            "test_exp", "state", "action", "result", 0.85
        )
        print(f"    ✓ Experience stored in {exp.tier.value}")

        # Test 6: Memory Pressure (V2)
        print("\n[6] Testing memory pressure (V2)...")
        pressure = await manager.check_pressure(MemoryTier.MAIN_CONTEXT)
        print(f"    ✓ Main context pressure: {pressure.value}")

        levels = manager.get_pressure_levels()
        print(f"    ✓ All pressure levels retrieved: {len(levels)} tiers")

        report = await manager.get_pressure_report()
        print(f"    ✓ Pressure report: {report[MemoryTier.MAIN_CONTEXT.value]['level']}")

        # Test 7: Sleep-Time Agent (V2)
        print("\n[7] Testing sleep-time agent (V2)...")
        status = manager.get_sleep_agent_status()
        print(f"    ✓ Sleep agent status: running={status.get('running', False)}")

        # Test consolidation (without starting agent loop)
        results = await manager.run_consolidation_now()
        print(f"    ✓ Manual consolidation: {len(results)} results")

        print("\n" + "=" * 60)
        print("All verifications passed!")
        print("=" * 60)

    asyncio.run(verify())


if __name__ == "__main__":
    run_quick_verification()
