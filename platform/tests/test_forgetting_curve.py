"""
Tests for Forgetting Curve and Memory Strength Decay - V40

Tests the Ebbinghaus forgetting curve implementation and memory consolidation.
"""

import asyncio
import math
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add parent directory to path to avoid conflict with built-in platform module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from the forgetting module
from core.memory.forgetting import (
    DEFAULT_DECAY_RATES,
    REINFORCEMENT_MULTIPLIERS,
    STRENGTH_ARCHIVE_THRESHOLD,
    STRENGTH_DELETE_THRESHOLD,
    ConsolidationReport,
    DecayCategory,
    ForgettingCurve,
    MemoryConsolidationTask,
    MemoryStrength,
    apply_forgetting_to_entry,
    get_memory_strength,
    reinforce_memory,
)
from core.memory.backends.base import (
    MemoryEntry,
    MemoryPriority,
    MemoryTier,
)


class TestForgettingCurve:
    """Tests for ForgettingCurve class."""

    def test_calculate_strength_no_decay(self):
        """Strength should not decay when no time has passed."""
        curve = ForgettingCurve(decay_rate=0.15)
        strength = curve.calculate_strength(
            initial_strength=1.0,
            days_elapsed=0.0
        )
        assert strength == 1.0

    def test_calculate_strength_with_decay(self):
        """Strength should decay over time."""
        curve = ForgettingCurve(decay_rate=0.15)
        strength = curve.calculate_strength(
            initial_strength=1.0,
            days_elapsed=7.0  # 1 week
        )
        # Expected: 1.0 * e^(-0.15 * 7) = 1.0 * e^(-1.05) ≈ 0.35
        expected = math.exp(-0.15 * 7)
        assert abs(strength - expected) < 0.01

    def test_calculate_strength_importance_slows_decay(self):
        """Higher importance should result in slower decay."""
        curve = ForgettingCurve(decay_rate=0.15)

        # Low importance
        low_importance = curve.calculate_strength(
            initial_strength=1.0,
            days_elapsed=7.0,
            importance=0.1
        )

        # High importance
        high_importance = curve.calculate_strength(
            initial_strength=1.0,
            days_elapsed=7.0,
            importance=0.9
        )

        # High importance should retain more strength
        assert high_importance > low_importance

    def test_calculate_strength_category_modifiers(self):
        """Different categories should have different decay rates."""
        curve = ForgettingCurve(decay_rate=0.15)

        # Semantic memories decay slowest
        semantic = curve.calculate_strength(
            initial_strength=1.0,
            days_elapsed=7.0,
            category=DecayCategory.SEMANTIC
        )

        # Episodic memories decay faster
        episodic = curve.calculate_strength(
            initial_strength=1.0,
            days_elapsed=7.0,
            category=DecayCategory.EPISODIC
        )

        assert semantic > episodic

    def test_calculate_reinforcement(self):
        """Reinforcement should boost strength."""
        curve = ForgettingCurve()
        new_strength, interval = curve.calculate_reinforcement(
            current_strength=0.5,
            access_type="recall",
            days_since_last=1.0,
            reinforcement_count=0
        )

        assert new_strength > 0.5
        assert interval > 0

    def test_reinforcement_diminishing_returns(self):
        """Reinforcement should have diminishing returns at high strength."""
        curve = ForgettingCurve()

        # Low strength - full reinforcement
        low_new, _ = curve.calculate_reinforcement(
            current_strength=0.3,
            access_type="recall"
        )
        low_gain = low_new - 0.3

        # High strength - diminished reinforcement
        high_new, _ = curve.calculate_reinforcement(
            current_strength=0.9,
            access_type="recall"
        )
        high_gain = high_new - 0.9

        assert low_gain > high_gain

    def test_predict_retention(self):
        """Should correctly predict when strength reaches target."""
        curve = ForgettingCurve()
        days = curve.predict_retention(
            initial_strength=1.0,
            decay_rate=0.15,
            target_strength=0.5
        )

        # At decay_rate=0.15, should take ~4.6 days to reach 0.5
        # t = -ln(0.5/1.0) / 0.15 = -ln(0.5) / 0.15 ≈ 4.62
        expected = -math.log(0.5) / 0.15
        assert abs(days - expected) < 0.1


class TestMemoryStrength:
    """Tests for MemoryStrength dataclass."""

    def test_to_dict_from_dict_roundtrip(self):
        """Should serialize and deserialize correctly."""
        original = MemoryStrength(
            strength=0.75,
            initial_strength=1.0,
            decay_rate=0.1,
            reinforcement_count=5,
            decay_category=DecayCategory.PROCEDURAL
        )

        data = original.to_dict()
        restored = MemoryStrength.from_dict(data)

        assert restored.strength == original.strength
        assert restored.decay_rate == original.decay_rate
        assert restored.reinforcement_count == original.reinforcement_count
        assert restored.decay_category == original.decay_category


class TestMemoryEntryIntegration:
    """Tests for MemoryEntry forgetting curve integration."""

    def test_calculate_current_strength_new_memory(self):
        """New memory should have full strength."""
        entry = MemoryEntry(
            id="test-1",
            content="Test content",
            strength=1.0,
            decay_rate=0.15,
            last_reinforced=datetime.now(timezone.utc)
        )

        strength = entry.calculate_current_strength()
        assert strength > 0.99  # Should be very close to 1.0

    def test_calculate_current_strength_old_memory(self):
        """Old memory should have decayed strength."""
        past = datetime.now(timezone.utc) - timedelta(days=7)
        entry = MemoryEntry(
            id="test-1",
            content="Test content",
            strength=1.0,
            decay_rate=0.15,
            last_reinforced=past
        )

        strength = entry.calculate_current_strength()
        assert strength < 0.5  # Should have decayed significantly

    def test_calculate_current_strength_critical_priority(self):
        """Critical priority should decay very slowly."""
        past = datetime.now(timezone.utc) - timedelta(days=30)

        critical = MemoryEntry(
            id="critical-1",
            content="Critical content",
            strength=1.0,
            decay_rate=0.15,
            priority=MemoryPriority.CRITICAL,
            last_reinforced=past
        )

        normal = MemoryEntry(
            id="normal-1",
            content="Normal content",
            strength=1.0,
            decay_rate=0.15,
            priority=MemoryPriority.NORMAL,
            last_reinforced=past
        )

        critical_strength = critical.calculate_current_strength()
        normal_strength = normal.calculate_current_strength()

        assert critical_strength > normal_strength

    def test_reinforce_boosts_strength(self):
        """Reinforce method should boost strength."""
        past = datetime.now(timezone.utc) - timedelta(days=3)
        entry = MemoryEntry(
            id="test-1",
            content="Test content",
            strength=0.8,
            decay_rate=0.15,
            last_reinforced=past,
            reinforcement_count=2
        )

        # Calculate decayed strength before reinforcement
        before = entry.calculate_current_strength()

        # Reinforce
        new_strength = entry.reinforce(access_type="recall")

        assert new_strength > before
        assert entry.reinforcement_count == 3
        assert entry.last_reinforced > past

    def test_is_weak_property(self):
        """is_weak should return True when strength below threshold."""
        past = datetime.now(timezone.utc) - timedelta(days=60)
        entry = MemoryEntry(
            id="test-1",
            content="Test content",
            strength=1.0,
            decay_rate=0.15,
            last_reinforced=past
        )

        # After 60 days with decay_rate=0.15, strength should be very low
        assert entry.is_weak is True

    def test_is_very_weak_property(self):
        """is_very_weak should return True when strength below delete threshold."""
        past = datetime.now(timezone.utc) - timedelta(days=120)
        entry = MemoryEntry(
            id="test-1",
            content="Test content",
            strength=1.0,
            decay_rate=0.15,
            last_reinforced=past
        )

        # After 120 days, should be very weak
        assert entry.is_very_weak is True


class TestApplyForgettingToEntry:
    """Tests for apply_forgetting_to_entry function."""

    def test_apply_forgetting_with_metadata(self):
        """Should work with entries that have strength metadata."""
        entry = MagicMock()
        entry.metadata = {
            "strength_data": {
                "strength": 0.8,
                "initial_strength": 1.0,
                "decay_rate": 0.15,
                "last_reinforced": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat(),
                "reinforcement_count": 2,
                "decay_category": "factual",
                "optimal_review_interval": 3.0,
                "last_review_intervals": [1.0, 2.0],
            },
            "importance": 0.7
        }

        strength = apply_forgetting_to_entry(entry)
        assert 0.0 <= strength <= 1.0


class TestMemoryConsolidationTask:
    """Tests for MemoryConsolidationTask."""

    @pytest.mark.asyncio
    async def test_consolidation_processes_memories(self):
        """Consolidation should process all memories."""
        # Create mock backend
        backend = AsyncMock()
        entries = [
            MemoryEntry(
                id=f"test-{i}",
                content=f"Content {i}",
                strength=1.0,
                decay_rate=0.15,
                last_reinforced=datetime.now(timezone.utc) - timedelta(days=i)
            )
            for i in range(5)
        ]
        backend.list_all.return_value = entries
        backend.put = AsyncMock()
        backend.delete = AsyncMock(return_value=True)

        # Run consolidation
        consolidator = MemoryConsolidationTask(backend)
        report = await consolidator.run()

        assert report.memories_processed == 5
        assert isinstance(report.average_strength_before, float)
        assert isinstance(report.average_strength_after, float)

    @pytest.mark.asyncio
    async def test_consolidation_archives_weak_memories(self):
        """Consolidation should archive weak memories."""
        backend = AsyncMock()
        archive_backend = AsyncMock()

        # Create a weak memory (very old)
        weak_entry = MemoryEntry(
            id="weak-1",
            content="Weak content",
            strength=0.05,
            decay_rate=0.15,
            last_reinforced=datetime.now(timezone.utc) - timedelta(days=90)
        )

        backend.list_all.return_value = [weak_entry]
        backend.put = AsyncMock()
        backend.delete = AsyncMock(return_value=True)
        archive_backend.put = AsyncMock()

        consolidator = MemoryConsolidationTask(
            backend,
            archive_backend=archive_backend,
            archive_threshold=0.1
        )
        report = await consolidator.run()

        # Should have archived the weak memory
        assert report.memories_archived >= 1 or report.memories_deleted >= 1

    @pytest.mark.asyncio
    async def test_consolidation_deletes_very_weak_memories(self):
        """Consolidation should delete very weak memories."""
        backend = AsyncMock()

        # Create a very weak memory
        very_weak = MemoryEntry(
            id="very-weak-1",
            content="Very weak content",
            strength=0.005,
            decay_rate=0.15,
            last_reinforced=datetime.now(timezone.utc) - timedelta(days=180)
        )

        backend.list_all.return_value = [very_weak]
        backend.delete = AsyncMock(return_value=True)

        consolidator = MemoryConsolidationTask(
            backend,
            delete_threshold=0.01
        )
        report = await consolidator.run()

        # Should have deleted the very weak memory
        assert report.memories_deleted >= 1


class TestConsolidationReport:
    """Tests for ConsolidationReport."""

    def test_to_dict(self):
        """Report should serialize to dict correctly."""
        report = ConsolidationReport(
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc) + timedelta(seconds=5),
            memories_processed=100,
            memories_archived=10,
            memories_deleted=5,
            memories_reinforced=85,
            average_strength_before=0.6,
            average_strength_after=0.7,
            errors=[]
        )

        data = report.to_dict()
        assert data["memories_processed"] == 100
        assert data["memories_archived"] == 10
        assert data["duration_seconds"] > 0


class TestDefaultDecayRates:
    """Tests for decay rate constants."""

    def test_critical_decays_slowest(self):
        """Critical priority should have lowest decay rate."""
        assert DEFAULT_DECAY_RATES["critical"] < DEFAULT_DECAY_RATES["normal"]

    def test_low_decays_fastest(self):
        """Low priority should have highest decay rate."""
        assert DEFAULT_DECAY_RATES["low"] > DEFAULT_DECAY_RATES["normal"]

    def test_all_rates_positive(self):
        """All decay rates should be positive."""
        for rate in DEFAULT_DECAY_RATES.values():
            assert rate > 0


class TestReinforcementMultipliers:
    """Tests for reinforcement multiplier constants."""

    def test_recall_strongest(self):
        """Active recall should provide strongest reinforcement."""
        assert REINFORCEMENT_MULTIPLIERS["recall"] >= max(
            REINFORCEMENT_MULTIPLIERS["review"],
            REINFORCEMENT_MULTIPLIERS["reference"],
            REINFORCEMENT_MULTIPLIERS["passive"]
        )

    def test_passive_weakest(self):
        """Passive exposure should provide weakest reinforcement."""
        assert REINFORCEMENT_MULTIPLIERS["passive"] <= min(
            REINFORCEMENT_MULTIPLIERS["recall"],
            REINFORCEMENT_MULTIPLIERS["review"]
        )


class TestThresholds:
    """Tests for threshold constants."""

    def test_archive_threshold_reasonable(self):
        """Archive threshold should be between 0 and 1."""
        assert 0 < STRENGTH_ARCHIVE_THRESHOLD < 1

    def test_delete_threshold_lower_than_archive(self):
        """Delete threshold should be lower than archive threshold."""
        assert STRENGTH_DELETE_THRESHOLD < STRENGTH_ARCHIVE_THRESHOLD


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
