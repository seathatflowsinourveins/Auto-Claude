"""
Forgetting Curve and Memory Strength Decay - V40 Advanced Memory Research

Implements Ebbinghaus forgetting curve for realistic memory decay:
- Memory strength (0.0-1.0) decays over time
- Reinforcement on access boosts strength
- Spaced repetition provides cumulative benefits
- Importance-weighted decay (critical memories decay slower)
- Background consolidation prunes weak memories

Formula: strength = initial_strength * e^(-decay_rate * days_since_reinforcement)

Usage:
    from core.memory.forgetting import (
        ForgettingCurve,
        MemoryStrength,
        MemoryConsolidationTask,
        apply_forgetting_to_entry,
        reinforce_memory,
    )

    # Calculate current strength
    curve = ForgettingCurve(decay_rate=0.1)
    current_strength = curve.calculate_strength(
        initial_strength=1.0,
        days_elapsed=3.0
    )

    # Reinforce on access
    reinforce_memory(entry, access_type="recall")

    # Run consolidation
    consolidator = MemoryConsolidationTask(backend)
    report = await consolidator.run()
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Default decay rates by priority (per day)
DEFAULT_DECAY_RATES: Dict[str, float] = {
    "critical": 0.01,   # Critical memories decay very slowly
    "high": 0.05,       # High priority decays slowly
    "normal": 0.15,     # Normal decay rate
    "low": 0.30,        # Low priority decays quickly
}

# Reinforcement multipliers by access type
REINFORCEMENT_MULTIPLIERS: Dict[str, float] = {
    "recall": 0.2,      # Active recall provides strong reinforcement
    "review": 0.15,     # Explicit review
    "reference": 0.1,   # Referenced by other memories
    "passive": 0.05,    # Passive exposure (shown but not acted on)
    "search_hit": 0.08, # Appeared in search results
}

# Spaced repetition bonus thresholds (days since last reinforcement)
SPACED_REPETITION_THRESHOLDS = [1, 3, 7, 14, 30, 60]
SPACED_REPETITION_BONUS = 0.1  # Bonus per optimal spacing

# Minimum strength before archival/deletion
STRENGTH_ARCHIVE_THRESHOLD = 0.1
STRENGTH_DELETE_THRESHOLD = 0.01


# =============================================================================
# DATA CLASSES
# =============================================================================

class DecayCategory(str, Enum):
    """Categories affecting decay rate."""
    FACTUAL = "factual"          # Facts decay normally
    PROCEDURAL = "procedural"    # How-to knowledge decays slower
    EPISODIC = "episodic"        # Events decay faster
    SEMANTIC = "semantic"        # Concepts decay slowest
    CONTEXTUAL = "contextual"    # Context decays fastest


@dataclass
class MemoryStrength:
    """Memory strength tracking data.

    These fields extend MemoryEntry for forgetting curve support.
    """
    strength: float = 1.0           # Current strength (0.0-1.0)
    initial_strength: float = 1.0   # Strength at creation
    decay_rate: float = 0.15        # Decay rate per day
    last_reinforced: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reinforcement_count: int = 0    # Total reinforcements
    decay_category: DecayCategory = DecayCategory.FACTUAL

    # Spaced repetition tracking
    optimal_review_interval: float = 1.0  # Days until optimal review
    last_review_intervals: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "strength": self.strength,
            "initial_strength": self.initial_strength,
            "decay_rate": self.decay_rate,
            "last_reinforced": self.last_reinforced.isoformat(),
            "reinforcement_count": self.reinforcement_count,
            "decay_category": self.decay_category.value,
            "optimal_review_interval": self.optimal_review_interval,
            "last_review_intervals": self.last_review_intervals[-10:],  # Keep last 10
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryStrength":
        """Create from dictionary."""
        return cls(
            strength=data.get("strength", 1.0),
            initial_strength=data.get("initial_strength", 1.0),
            decay_rate=data.get("decay_rate", 0.15),
            last_reinforced=datetime.fromisoformat(data["last_reinforced"])
                if data.get("last_reinforced") else datetime.now(timezone.utc),
            reinforcement_count=data.get("reinforcement_count", 0),
            decay_category=DecayCategory(data.get("decay_category", "factual")),
            optimal_review_interval=data.get("optimal_review_interval", 1.0),
            last_review_intervals=data.get("last_review_intervals", []),
        )


@dataclass
class ConsolidationReport:
    """Report from memory consolidation task."""
    started_at: datetime
    completed_at: datetime
    memories_processed: int
    memories_archived: int
    memories_deleted: int
    memories_reinforced: int
    average_strength_before: float
    average_strength_after: float
    errors: List[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Get task duration in seconds."""
        return (self.completed_at - self.started_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "memories_processed": self.memories_processed,
            "memories_archived": self.memories_archived,
            "memories_deleted": self.memories_deleted,
            "memories_reinforced": self.memories_reinforced,
            "average_strength_before": self.average_strength_before,
            "average_strength_after": self.average_strength_after,
            "errors": self.errors,
        }


# =============================================================================
# FORGETTING CURVE
# =============================================================================

class ForgettingCurve:
    """Implements Ebbinghaus forgetting curve with extensions.

    The forgetting curve models how memory strength decays exponentially
    over time without reinforcement:

        S(t) = S_0 * e^(-lambda * t)

    Where:
        S(t) = strength at time t
        S_0 = initial strength
        lambda = decay rate
        t = time elapsed (days)

    Extensions:
        - Importance-weighted decay (critical memories decay slower)
        - Spaced repetition bonuses (optimal review timing)
        - Category-based decay modifiers
    """

    def __init__(
        self,
        decay_rate: float = 0.15,
        importance_weight: float = 1.0,
        category_modifiers: Optional[Dict[DecayCategory, float]] = None
    ) -> None:
        """Initialize forgetting curve.

        Args:
            decay_rate: Base decay rate per day (0.0-1.0)
            importance_weight: Multiplier for importance (0.0-1.0 = slower decay)
            category_modifiers: Decay rate multipliers by category
        """
        self.decay_rate = decay_rate
        self.importance_weight = importance_weight
        self.category_modifiers = category_modifiers or {
            DecayCategory.FACTUAL: 1.0,
            DecayCategory.PROCEDURAL: 0.7,
            DecayCategory.EPISODIC: 1.3,
            DecayCategory.SEMANTIC: 0.5,
            DecayCategory.CONTEXTUAL: 1.5,
        }

    def calculate_strength(
        self,
        initial_strength: float,
        days_elapsed: float,
        decay_rate: Optional[float] = None,
        importance: float = 0.5,
        category: DecayCategory = DecayCategory.FACTUAL
    ) -> float:
        """Calculate current memory strength using forgetting curve.

        Args:
            initial_strength: Strength at last reinforcement (0.0-1.0)
            days_elapsed: Days since last reinforcement
            decay_rate: Override decay rate (uses instance default if None)
            importance: Memory importance (0.0-1.0), higher = slower decay
            category: Memory category for decay modifier

        Returns:
            Current strength (0.0-1.0)
        """
        if days_elapsed <= 0:
            return initial_strength

        # Get effective decay rate
        effective_rate = decay_rate or self.decay_rate

        # Apply importance weighting (higher importance = slower decay)
        importance_factor = 1.0 - (importance * self.importance_weight * 0.5)
        effective_rate *= importance_factor

        # Apply category modifier
        category_modifier = self.category_modifiers.get(category, 1.0)
        effective_rate *= category_modifier

        # Ebbinghaus forgetting curve: S = S_0 * e^(-lambda * t)
        strength = initial_strength * math.exp(-effective_rate * days_elapsed)

        # Clamp to valid range
        return max(0.0, min(1.0, strength))

    def calculate_reinforcement(
        self,
        current_strength: float,
        access_type: str = "recall",
        days_since_last: float = 0.0,
        reinforcement_count: int = 0
    ) -> Tuple[float, float]:
        """Calculate new strength after reinforcement.

        Args:
            current_strength: Current strength before reinforcement
            access_type: Type of access (recall, review, reference, passive)
            days_since_last: Days since last reinforcement
            reinforcement_count: Previous reinforcement count

        Returns:
            Tuple of (new_strength, optimal_next_interval)
        """
        # Base reinforcement amount
        base_reinforcement = REINFORCEMENT_MULTIPLIERS.get(access_type, 0.05)

        # Spaced repetition bonus - optimal spacing provides extra boost
        spacing_bonus = self._calculate_spacing_bonus(days_since_last, reinforcement_count)

        # Calculate new strength
        reinforcement = base_reinforcement + spacing_bonus

        # Diminishing returns at high strength
        if current_strength > 0.8:
            reinforcement *= (1.0 - current_strength) * 2

        new_strength = min(1.0, current_strength + reinforcement)

        # Calculate optimal next review interval (spaced repetition)
        optimal_interval = self._calculate_optimal_interval(
            reinforcement_count + 1,
            new_strength
        )

        return new_strength, optimal_interval

    def _calculate_spacing_bonus(
        self,
        days_since_last: float,
        reinforcement_count: int
    ) -> float:
        """Calculate bonus for optimal spaced repetition timing."""
        if reinforcement_count == 0:
            return SPACED_REPETITION_BONUS  # First review always gets bonus

        # Get expected interval based on reinforcement count
        threshold_index = min(reinforcement_count - 1, len(SPACED_REPETITION_THRESHOLDS) - 1)
        expected_interval = SPACED_REPETITION_THRESHOLDS[threshold_index]

        # Calculate how close to optimal timing
        ratio = days_since_last / expected_interval

        # Optimal is between 0.8 and 1.2 of expected interval
        if 0.8 <= ratio <= 1.2:
            return SPACED_REPETITION_BONUS
        elif 0.5 <= ratio <= 1.5:
            return SPACED_REPETITION_BONUS * 0.5
        else:
            return 0.0

    def _calculate_optimal_interval(
        self,
        reinforcement_count: int,
        current_strength: float
    ) -> float:
        """Calculate optimal interval until next review."""
        # Base interval from spaced repetition schedule
        index = min(reinforcement_count - 1, len(SPACED_REPETITION_THRESHOLDS) - 1)
        base_interval = SPACED_REPETITION_THRESHOLDS[index]

        # Adjust based on current strength (stronger = longer interval OK)
        strength_factor = 0.5 + current_strength

        return base_interval * strength_factor

    def predict_retention(
        self,
        initial_strength: float,
        decay_rate: float,
        target_strength: float = 0.5
    ) -> float:
        """Predict days until memory decays to target strength.

        Args:
            initial_strength: Starting strength
            decay_rate: Decay rate per day
            target_strength: Target strength level

        Returns:
            Days until target strength reached
        """
        if target_strength >= initial_strength:
            return 0.0
        if decay_rate <= 0:
            return float('inf')

        # Solve: target = initial * e^(-rate * t) for t
        # t = -ln(target/initial) / rate
        return -math.log(target_strength / initial_strength) / decay_rate


# =============================================================================
# ENTRY INTEGRATION
# =============================================================================

def get_memory_strength(entry: Any) -> MemoryStrength:
    """Extract or create MemoryStrength from a MemoryEntry.

    Args:
        entry: MemoryEntry with optional strength metadata

    Returns:
        MemoryStrength instance
    """
    metadata = getattr(entry, 'metadata', {}) or {}
    strength_data = metadata.get('strength_data')

    if strength_data:
        return MemoryStrength.from_dict(strength_data)

    # Create default based on entry priority
    priority = getattr(entry, 'priority', None)
    priority_value = priority.value if hasattr(priority, 'value') else str(priority or 'normal')
    decay_rate = DEFAULT_DECAY_RATES.get(priority_value, 0.15)

    return MemoryStrength(
        strength=1.0,
        initial_strength=1.0,
        decay_rate=decay_rate,
        last_reinforced=getattr(entry, 'created_at', datetime.now(timezone.utc)),
    )


def apply_forgetting_to_entry(
    entry: Any,
    curve: Optional[ForgettingCurve] = None
) -> float:
    """Apply forgetting curve to calculate current strength.

    Args:
        entry: MemoryEntry to process
        curve: ForgettingCurve instance (uses default if None)

    Returns:
        Current strength value (0.0-1.0)
    """
    curve = curve or ForgettingCurve()
    strength_data = get_memory_strength(entry)

    # Calculate days elapsed
    now = datetime.now(timezone.utc)
    delta = now - strength_data.last_reinforced
    days_elapsed = delta.total_seconds() / 86400

    # Get importance from metadata
    metadata = getattr(entry, 'metadata', {}) or {}
    importance = metadata.get('importance', 0.5)

    # Calculate current strength
    current_strength = curve.calculate_strength(
        initial_strength=strength_data.initial_strength,
        days_elapsed=days_elapsed,
        decay_rate=strength_data.decay_rate,
        importance=importance,
        category=strength_data.decay_category
    )

    return current_strength


def reinforce_memory(
    entry: Any,
    access_type: str = "recall",
    curve: Optional[ForgettingCurve] = None
) -> Tuple[float, float]:
    """Reinforce memory on access, updating strength.

    Args:
        entry: MemoryEntry to reinforce
        access_type: Type of access (recall, review, reference, passive)
        curve: ForgettingCurve instance (uses default if None)

    Returns:
        Tuple of (new_strength, optimal_next_interval_days)
    """
    curve = curve or ForgettingCurve()
    strength_data = get_memory_strength(entry)

    # Calculate current strength (with decay)
    current_strength = apply_forgetting_to_entry(entry, curve)

    # Calculate days since last reinforcement
    now = datetime.now(timezone.utc)
    delta = now - strength_data.last_reinforced
    days_since_last = delta.total_seconds() / 86400

    # Calculate reinforcement
    new_strength, optimal_interval = curve.calculate_reinforcement(
        current_strength=current_strength,
        access_type=access_type,
        days_since_last=days_since_last,
        reinforcement_count=strength_data.reinforcement_count
    )

    # Update strength data
    strength_data.strength = new_strength
    strength_data.initial_strength = new_strength
    strength_data.last_reinforced = now
    strength_data.reinforcement_count += 1
    strength_data.optimal_review_interval = optimal_interval
    strength_data.last_review_intervals.append(days_since_last)

    # Store back in entry metadata
    metadata = getattr(entry, 'metadata', {}) or {}
    metadata['strength_data'] = strength_data.to_dict()
    entry.metadata = metadata

    return new_strength, optimal_interval


# =============================================================================
# CONSOLIDATION TASK
# =============================================================================

class MemoryBackendProtocol(Protocol):
    """Protocol for memory backends supporting consolidation."""

    async def list_all(self) -> List[Any]: ...
    async def delete(self, key: str) -> bool: ...
    async def put(self, key: str, value: Any) -> None: ...


class MemoryConsolidationTask:
    """Background task for memory consolidation.

    Periodically reviews all memories and:
    - Updates strength values based on decay
    - Archives weak memories (strength < 0.1)
    - Deletes very weak memories (strength < 0.01)
    - Generates consolidation reports
    """

    def __init__(
        self,
        backend: MemoryBackendProtocol,
        archive_backend: Optional[MemoryBackendProtocol] = None,
        curve: Optional[ForgettingCurve] = None,
        archive_threshold: float = STRENGTH_ARCHIVE_THRESHOLD,
        delete_threshold: float = STRENGTH_DELETE_THRESHOLD
    ) -> None:
        """Initialize consolidation task.

        Args:
            backend: Primary memory backend
            archive_backend: Optional backend for archived memories
            curve: ForgettingCurve instance
            archive_threshold: Strength below which to archive
            delete_threshold: Strength below which to delete
        """
        self.backend = backend
        self.archive_backend = archive_backend
        self.curve = curve or ForgettingCurve()
        self.archive_threshold = archive_threshold
        self.delete_threshold = delete_threshold

        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def run(self) -> ConsolidationReport:
        """Run a single consolidation pass.

        Returns:
            ConsolidationReport with results
        """
        started_at = datetime.now(timezone.utc)

        memories_processed = 0
        memories_archived = 0
        memories_deleted = 0
        memories_reinforced = 0
        strength_sum_before = 0.0
        strength_sum_after = 0.0
        errors: List[str] = []

        try:
            entries = await self.backend.list_all()

            for entry in entries:
                try:
                    memories_processed += 1

                    # Calculate current strength
                    current_strength = apply_forgetting_to_entry(entry, self.curve)
                    strength_sum_before += current_strength

                    # Check if memory should be archived/deleted
                    if current_strength < self.delete_threshold:
                        # Delete very weak memories
                        entry_id = getattr(entry, 'id', str(entry))
                        await self.backend.delete(entry_id)
                        memories_deleted += 1
                        logger.debug(f"Deleted weak memory {entry_id} (strength={current_strength:.3f})")
                        continue

                    elif current_strength < self.archive_threshold:
                        # Archive weak memories
                        if self.archive_backend:
                            entry_id = getattr(entry, 'id', str(entry))

                            # Mark as archived in metadata
                            metadata = getattr(entry, 'metadata', {}) or {}
                            metadata['archived_at'] = datetime.now(timezone.utc).isoformat()
                            metadata['archived_strength'] = current_strength
                            entry.metadata = metadata

                            await self.archive_backend.put(entry_id, entry)
                            await self.backend.delete(entry_id)
                            memories_archived += 1
                            logger.debug(f"Archived memory {entry_id} (strength={current_strength:.3f})")
                            continue

                    # Update strength in metadata
                    strength_data = get_memory_strength(entry)
                    strength_data.strength = current_strength

                    metadata = getattr(entry, 'metadata', {}) or {}
                    metadata['strength_data'] = strength_data.to_dict()
                    entry.metadata = metadata

                    # Save updated entry
                    entry_id = getattr(entry, 'id', str(entry))
                    await self.backend.put(entry_id, entry)

                    strength_sum_after += current_strength
                    memories_reinforced += 1

                except Exception as e:
                    entry_id = getattr(entry, 'id', 'unknown')
                    errors.append(f"Error processing {entry_id}: {str(e)}")
                    logger.warning(f"Error in consolidation for {entry_id}: {e}")

        except Exception as e:
            errors.append(f"Fatal error: {str(e)}")
            logger.error(f"Consolidation task failed: {e}")

        completed_at = datetime.now(timezone.utc)

        return ConsolidationReport(
            started_at=started_at,
            completed_at=completed_at,
            memories_processed=memories_processed,
            memories_archived=memories_archived,
            memories_deleted=memories_deleted,
            memories_reinforced=memories_reinforced,
            average_strength_before=strength_sum_before / max(memories_processed, 1),
            average_strength_after=strength_sum_after / max(memories_reinforced, 1),
            errors=errors
        )

    async def start_periodic(self, interval_hours: float = 24.0) -> None:
        """Start periodic consolidation task.

        Args:
            interval_hours: Hours between consolidation runs
        """
        if self._running:
            logger.warning("Consolidation task already running")
            return

        self._running = True

        async def periodic_runner():
            while self._running:
                try:
                    report = await self.run()
                    logger.info(
                        f"Consolidation complete: {report.memories_processed} processed, "
                        f"{report.memories_archived} archived, {report.memories_deleted} deleted"
                    )
                except Exception as e:
                    logger.error(f"Periodic consolidation failed: {e}")

                await asyncio.sleep(interval_hours * 3600)

        self._task = asyncio.create_task(periodic_runner())
        logger.info(f"Started periodic consolidation (interval={interval_hours}h)")

    async def stop(self) -> None:
        """Stop periodic consolidation task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Stopped periodic consolidation")


# =============================================================================
# SQLITE SCHEMA MIGRATION
# =============================================================================

FORGETTING_MIGRATION_SQL = """
-- Migration for forgetting curve support (V2)
-- Adds strength tracking columns to memories table

-- Add strength column (0.0-1.0)
ALTER TABLE memories ADD COLUMN strength REAL NOT NULL DEFAULT 1.0;

-- Add decay rate column (per day)
ALTER TABLE memories ADD COLUMN decay_rate REAL NOT NULL DEFAULT 0.15;

-- Add last reinforcement timestamp
ALTER TABLE memories ADD COLUMN last_reinforced TEXT;

-- Add reinforcement count
ALTER TABLE memories ADD COLUMN reinforcement_count INTEGER NOT NULL DEFAULT 0;

-- Add index on strength for efficient queries
CREATE INDEX IF NOT EXISTS idx_memories_strength ON memories(strength);

-- Add index on last_reinforced for decay calculations
CREATE INDEX IF NOT EXISTS idx_memories_reinforced ON memories(last_reinforced);

-- Update schema version
INSERT INTO schema_version (version, applied_at) VALUES (2, datetime('now'));
"""

FORGETTING_SCHEMA_CHECK_SQL = """
SELECT COUNT(*) FROM pragma_table_info('memories') WHERE name = 'strength';
"""


async def apply_forgetting_migration(db_path: str) -> bool:
    """Apply forgetting curve migration to SQLite database.

    Args:
        db_path: Path to SQLite database

    Returns:
        True if migration applied, False if already migrated
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        # Check if migration already applied
        cursor = conn.execute(FORGETTING_SCHEMA_CHECK_SQL)
        if cursor.fetchone()[0] > 0:
            logger.info("Forgetting curve migration already applied")
            return False

        # Apply migration
        for statement in FORGETTING_MIGRATION_SQL.split(';'):
            statement = statement.strip()
            if statement:
                try:
                    conn.execute(statement)
                except sqlite3.OperationalError as e:
                    # Ignore "column already exists" errors
                    if "duplicate column name" not in str(e).lower():
                        raise

        conn.commit()
        logger.info("Applied forgetting curve migration")
        return True

    finally:
        conn.close()


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Core classes
    "ForgettingCurve",
    "MemoryStrength",
    "DecayCategory",
    "ConsolidationReport",

    # Constants
    "DEFAULT_DECAY_RATES",
    "REINFORCEMENT_MULTIPLIERS",
    "SPACED_REPETITION_THRESHOLDS",
    "STRENGTH_ARCHIVE_THRESHOLD",
    "STRENGTH_DELETE_THRESHOLD",

    # Entry integration
    "get_memory_strength",
    "apply_forgetting_to_entry",
    "reinforce_memory",

    # Consolidation
    "MemoryConsolidationTask",

    # Migration
    "apply_forgetting_migration",
    "FORGETTING_MIGRATION_SQL",
]
