"""
Memory Compaction System for Long-Running Sessions - V42 Architecture

Provides intelligent memory compaction to optimize storage and performance
for long-running agent sessions.

Features:
- MemoryCompactor class for compaction operations
- Identification of low-value memories (low access, old)
- Semantic clustering to merge similar memories
- Archive to cold storage
- Index defragmentation
- Background compaction scheduler

Compaction Strategies:
1. Time-based: Compact memories older than threshold
2. Size-based: Compact when memory exceeds budget
3. Quality-based: Compact low-importance memories

Usage:
    from core.memory.compaction import (
        MemoryCompactor,
        CompactionStrategy,
        CompactionConfig,
        CompactionScheduler,
        CompactionReport,
    )

    # Create compactor
    compactor = MemoryCompactor(backend)

    # Run compaction
    report = await compactor.compact()

    # Start background scheduler
    scheduler = CompactionScheduler(compactor)
    await scheduler.start()
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .backends.base import MemoryEntry
    from .backends.sqlite import SQLiteTierBackend

logger = logging.getLogger(__name__)


# =============================================================================
# PROTOCOLS
# =============================================================================

class CompactionBackendProtocol(Protocol):
    """Protocol for backends supporting compaction."""

    async def list_all(self) -> List[Any]: ...
    async def get(self, key: str, reinforce: bool = False) -> Optional[Any]: ...
    async def put(self, key: str, value: Any) -> None: ...
    async def delete(self, key: str) -> bool: ...
    async def search(self, query: str, limit: int = 10) -> List[Any]: ...
    async def count(self) -> int: ...
    async def get_stats(self) -> Dict[str, Any]: ...


# =============================================================================
# ENUMS AND CONFIGURATION
# =============================================================================

class CompactionStrategy(str, Enum):
    """Compaction strategies for memory optimization."""
    TIME_BASED = "time_based"         # Compact memories older than threshold
    SIZE_BASED = "size_based"         # Compact when exceeding memory budget
    QUALITY_BASED = "quality_based"   # Compact low-importance memories
    ADAPTIVE = "adaptive"             # Combine strategies based on conditions
    AGGRESSIVE = "aggressive"         # Maximum compaction for memory pressure


class CompactionPriority(str, Enum):
    """Priority levels for compaction decisions."""
    PRESERVE = "preserve"      # Never compact (critical memories)
    LOW = "low"                # Compact only under pressure
    NORMAL = "normal"          # Standard compaction rules
    HIGH = "high"              # Compact eagerly
    IMMEDIATE = "immediate"    # Compact immediately


class MergeStrategy(str, Enum):
    """Strategies for merging similar memories."""
    NEWEST_WINS = "newest_wins"     # Keep most recent content
    OLDEST_WINS = "oldest_wins"     # Preserve original content
    COMBINE = "combine"             # Merge content intelligently
    HIGHEST_SCORE = "highest_score" # Keep highest importance/strength


@dataclass
class CompactionConfig:
    """Configuration for memory compaction."""
    # Time-based thresholds
    min_age_days: float = 7.0                # Minimum age before compaction
    max_age_days: float = 30.0               # Age at which compaction is aggressive
    stale_threshold_days: float = 14.0       # Days without access = stale

    # Size-based thresholds
    max_memory_count: int = 10000            # Trigger compaction above this
    target_memory_count: int = 5000          # Target count after compaction
    max_storage_bytes: int = 100 * 1024 * 1024  # 100MB threshold

    # Quality-based thresholds
    min_strength_threshold: float = 0.1      # Below this = candidate
    min_importance_threshold: float = 0.2    # Below this = candidate
    min_access_count: int = 2                # Below this = candidate
    max_access_age_days: float = 7.0         # Days since last access

    # Merge settings
    similarity_threshold: float = 0.85       # Similarity for merging
    max_merge_group_size: int = 10           # Max memories to merge together
    merge_strategy: MergeStrategy = MergeStrategy.HIGHEST_SCORE

    # Archive settings
    enable_cold_storage: bool = True         # Archive instead of delete
    cold_storage_path: Optional[str] = None  # Path for cold storage
    archive_batch_size: int = 100            # Batch size for archiving

    # Index settings
    defragment_threshold: float = 0.3        # Fragmentation level to trigger
    vacuum_after_compact: bool = True        # Run VACUUM after compaction

    # Scheduler settings
    schedule_interval_hours: float = 6.0     # Background check interval
    enable_auto_compact: bool = True         # Auto-compact when thresholds hit
    max_compact_duration_seconds: float = 300.0  # Max time per compaction run

    # Preservation
    preserve_namespaces: List[str] = field(default_factory=lambda: ["artifacts"])
    preserve_tags: List[str] = field(default_factory=lambda: ["critical", "permanent"])
    min_reinforcement_to_preserve: int = 5   # Reinforcements to preserve


@dataclass
class CompactionCandidate:
    """A memory entry identified for compaction."""
    entry_id: str
    content: str
    age_days: float
    days_since_access: float
    access_count: int
    strength: float
    importance: float
    priority: CompactionPriority
    reason: str
    score: float  # Higher = more likely to compact

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_id": self.entry_id,
            "age_days": round(self.age_days, 2),
            "days_since_access": round(self.days_since_access, 2),
            "access_count": self.access_count,
            "strength": round(self.strength, 4),
            "importance": round(self.importance, 4),
            "priority": self.priority.value,
            "reason": self.reason,
            "score": round(self.score, 4),
        }


@dataclass
class MergeGroup:
    """A group of similar memories to be merged."""
    representative_id: str
    member_ids: List[str]
    similarity_scores: Dict[str, float]
    combined_content: Optional[str] = None
    merged_tags: List[str] = field(default_factory=list)
    merged_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompactionReport:
    """Report from a compaction operation."""
    started_at: datetime
    completed_at: datetime
    strategy_used: CompactionStrategy

    # Counts
    memories_analyzed: int
    candidates_identified: int
    memories_archived: int
    memories_deleted: int
    memories_merged: int
    merge_groups_processed: int

    # Storage metrics
    bytes_before: int
    bytes_after: int
    bytes_saved: int
    fragmentation_before: float
    fragmentation_after: float

    # Quality metrics
    avg_strength_removed: float
    avg_importance_removed: float
    avg_age_removed_days: float

    # Errors
    errors: List[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Get compaction duration."""
        return (self.completed_at - self.started_at).total_seconds()

    @property
    def compression_ratio(self) -> float:
        """Get compression ratio."""
        if self.bytes_before == 0:
            return 1.0
        return self.bytes_after / self.bytes_before

    @property
    def space_savings_percent(self) -> float:
        """Get space savings percentage."""
        if self.bytes_before == 0:
            return 0.0
        return (self.bytes_saved / self.bytes_before) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": round(self.duration_seconds, 2),
            "strategy_used": self.strategy_used.value,
            "counts": {
                "analyzed": self.memories_analyzed,
                "candidates": self.candidates_identified,
                "archived": self.memories_archived,
                "deleted": self.memories_deleted,
                "merged": self.memories_merged,
                "merge_groups": self.merge_groups_processed,
            },
            "storage": {
                "bytes_before": self.bytes_before,
                "bytes_after": self.bytes_after,
                "bytes_saved": self.bytes_saved,
                "compression_ratio": round(self.compression_ratio, 4),
                "space_savings_percent": round(self.space_savings_percent, 2),
            },
            "fragmentation": {
                "before": round(self.fragmentation_before, 4),
                "after": round(self.fragmentation_after, 4),
            },
            "removed_averages": {
                "strength": round(self.avg_strength_removed, 4),
                "importance": round(self.avg_importance_removed, 4),
                "age_days": round(self.avg_age_removed_days, 2),
            },
            "errors": self.errors,
        }


@dataclass
class CompactionMetrics:
    """Aggregated metrics for compaction system."""
    total_compactions: int = 0
    total_memories_compacted: int = 0
    total_bytes_saved: int = 0
    total_merge_groups: int = 0
    average_compression_ratio: float = 1.0
    last_compaction: Optional[datetime] = None
    compaction_history: List[Dict[str, Any]] = field(default_factory=list)

    def record_compaction(self, report: CompactionReport) -> None:
        """Record a compaction result."""
        self.total_compactions += 1
        self.total_memories_compacted += (
            report.memories_archived + report.memories_deleted + report.memories_merged
        )
        self.total_bytes_saved += report.bytes_saved
        self.total_merge_groups += report.merge_groups_processed
        self.last_compaction = report.completed_at

        # Update rolling average
        if self.total_compactions == 1:
            self.average_compression_ratio = report.compression_ratio
        else:
            self.average_compression_ratio = (
                0.9 * self.average_compression_ratio + 0.1 * report.compression_ratio
            )

        # Keep last 10 compaction summaries
        self.compaction_history.append({
            "timestamp": report.completed_at.isoformat(),
            "strategy": report.strategy_used.value,
            "memories_compacted": (
                report.memories_archived + report.memories_deleted + report.memories_merged
            ),
            "bytes_saved": report.bytes_saved,
        })
        self.compaction_history = self.compaction_history[-10:]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_compactions": self.total_compactions,
            "total_memories_compacted": self.total_memories_compacted,
            "total_bytes_saved": self.total_bytes_saved,
            "total_merge_groups": self.total_merge_groups,
            "average_compression_ratio": round(self.average_compression_ratio, 4),
            "last_compaction": self.last_compaction.isoformat() if self.last_compaction else None,
            "recent_history": self.compaction_history,
        }


# =============================================================================
# MEMORY COMPACTOR
# =============================================================================

class MemoryCompactor:
    """
    Main memory compaction system for long-running sessions.

    Provides:
    - Identification of low-value memories
    - Semantic clustering and merging of similar memories
    - Archive to cold storage
    - Index defragmentation
    - Multiple compaction strategies
    """

    def __init__(
        self,
        backend: Optional[CompactionBackendProtocol] = None,
        config: Optional[CompactionConfig] = None,
        cold_storage_backend: Optional[CompactionBackendProtocol] = None,
        embedding_provider: Optional[Callable[[str], List[float]]] = None
    ) -> None:
        """
        Initialize memory compactor.

        Args:
            backend: Memory storage backend
            config: Compaction configuration
            cold_storage_backend: Backend for archived memories
            embedding_provider: Function to generate embeddings for similarity
        """
        self._backend = backend
        self._config = config or CompactionConfig()
        self._cold_storage = cold_storage_backend
        self._embedding_provider = embedding_provider
        self._metrics = CompactionMetrics()
        self._running = False

    @property
    def backend(self) -> CompactionBackendProtocol:
        """Get backend, initializing if needed."""
        if self._backend is None:
            from .backends.sqlite import get_sqlite_backend
            self._backend = get_sqlite_backend()
        return self._backend

    @property
    def config(self) -> CompactionConfig:
        """Get compaction configuration."""
        return self._config

    @property
    def metrics(self) -> CompactionMetrics:
        """Get compaction metrics."""
        return self._metrics

    # =========================================================================
    # CANDIDATE IDENTIFICATION
    # =========================================================================

    async def identify_low_value_memories(self) -> List[CompactionCandidate]:
        """
        Identify memories that are candidates for compaction.

        Criteria:
        - Low access count
        - Old age
        - Low strength (forgetting curve)
        - Low importance
        - Not recently accessed

        Returns:
            List of CompactionCandidate objects sorted by compaction score
        """
        candidates: List[CompactionCandidate] = []
        now = datetime.now(timezone.utc)

        try:
            entries = await self.backend.list_all()

            for entry in entries:
                # Skip preserved entries
                if self._should_preserve(entry):
                    continue

                # Calculate metrics
                entry_id = getattr(entry, 'id', str(entry))
                content = getattr(entry, 'content', '')
                created_at = getattr(entry, 'created_at', now)
                last_accessed = getattr(entry, 'last_accessed', created_at)
                access_count = getattr(entry, 'access_count', 0)
                strength = getattr(entry, 'strength', 1.0)
                metadata = getattr(entry, 'metadata', {}) or {}
                importance = metadata.get('importance', 0.5)
                reinforcement_count = getattr(entry, 'reinforcement_count', 0)

                # Ensure timezone-aware datetimes
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                if last_accessed and last_accessed.tzinfo is None:
                    last_accessed = last_accessed.replace(tzinfo=timezone.utc)
                if last_accessed is None:
                    last_accessed = created_at

                age_days = (now - created_at).total_seconds() / 86400
                days_since_access = (now - last_accessed).total_seconds() / 86400

                # Determine if candidate and why
                reasons = []
                priority = CompactionPriority.NORMAL

                # Time-based checks
                if age_days >= self._config.max_age_days:
                    reasons.append(f"very old ({age_days:.1f} days)")
                    priority = CompactionPriority.HIGH
                elif age_days >= self._config.min_age_days:
                    reasons.append(f"old ({age_days:.1f} days)")

                # Access-based checks
                if days_since_access >= self._config.stale_threshold_days:
                    reasons.append(f"stale ({days_since_access:.1f} days since access)")
                    priority = CompactionPriority.HIGH
                elif days_since_access >= self._config.max_access_age_days:
                    reasons.append(f"not recent ({days_since_access:.1f} days)")

                if access_count < self._config.min_access_count:
                    reasons.append(f"low access ({access_count})")

                # Quality-based checks
                current_strength = self._calculate_current_strength(entry)
                if current_strength < self._config.min_strength_threshold:
                    reasons.append(f"weak ({current_strength:.2f})")
                    priority = CompactionPriority.HIGH

                if importance < self._config.min_importance_threshold:
                    reasons.append(f"low importance ({importance:.2f})")

                # Must have at least one reason to be a candidate
                if not reasons:
                    continue

                # Skip highly reinforced memories
                if reinforcement_count >= self._config.min_reinforcement_to_preserve:
                    continue

                # Calculate compaction score (higher = more likely to compact)
                score = self._calculate_compaction_score(
                    age_days=age_days,
                    days_since_access=days_since_access,
                    access_count=access_count,
                    strength=current_strength,
                    importance=importance
                )

                candidates.append(CompactionCandidate(
                    entry_id=entry_id,
                    content=content,
                    age_days=age_days,
                    days_since_access=days_since_access,
                    access_count=access_count,
                    strength=current_strength,
                    importance=importance,
                    priority=priority,
                    reason="; ".join(reasons),
                    score=score
                ))

        except Exception as e:
            logger.error(f"Error identifying low-value memories: {e}")

        # Sort by score descending (highest score = most likely to compact)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def _should_preserve(self, entry: Any) -> bool:
        """Check if entry should be preserved from compaction."""
        # Check priority
        priority = getattr(entry, 'priority', None)
        if priority and hasattr(priority, 'value'):
            if priority.value == 'critical':
                return True

        # Check namespace
        namespace = getattr(entry, 'namespace', None)
        if namespace:
            ns_value = namespace.value if hasattr(namespace, 'value') else str(namespace)
            if ns_value in self._config.preserve_namespaces:
                return True

        # Check tags
        tags = getattr(entry, 'tags', []) or []
        for tag in tags:
            if tag in self._config.preserve_tags:
                return True

        return False

    def _calculate_current_strength(self, entry: Any) -> float:
        """Calculate current memory strength with decay."""
        if hasattr(entry, 'calculate_current_strength'):
            return entry.calculate_current_strength()

        # Fallback calculation
        strength = getattr(entry, 'strength', 1.0)
        decay_rate = getattr(entry, 'decay_rate', 0.15)
        last_reinforced = getattr(entry, 'last_reinforced', None)
        created_at = getattr(entry, 'created_at', None)

        reference_time = last_reinforced or created_at
        if reference_time is None:
            return strength

        now = datetime.now(timezone.utc)
        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=timezone.utc)

        days_elapsed = (now - reference_time).total_seconds() / 86400
        if days_elapsed <= 0:
            return strength

        # Ebbinghaus decay
        return strength * math.exp(-decay_rate * days_elapsed)

    def _calculate_compaction_score(
        self,
        age_days: float,
        days_since_access: float,
        access_count: int,
        strength: float,
        importance: float
    ) -> float:
        """
        Calculate compaction score (0-1, higher = more likely to compact).

        Factors:
        - Age: Older = higher score
        - Access recency: Less recent = higher score
        - Access count: Lower = higher score
        - Strength: Lower = higher score
        - Importance: Lower = higher score
        """
        score = 0.0

        # Age factor (0-0.25)
        age_factor = min(1.0, age_days / self._config.max_age_days) * 0.25
        score += age_factor

        # Access recency factor (0-0.25)
        recency_factor = min(1.0, days_since_access / self._config.stale_threshold_days) * 0.25
        score += recency_factor

        # Access count factor (0-0.15)
        count_factor = max(0, 1.0 - access_count / 10.0) * 0.15
        score += count_factor

        # Strength factor (0-0.2)
        strength_factor = (1.0 - strength) * 0.2
        score += strength_factor

        # Importance factor (0-0.15)
        importance_factor = (1.0 - importance) * 0.15
        score += importance_factor

        return min(1.0, score)

    # =========================================================================
    # SEMANTIC CLUSTERING AND MERGING
    # =========================================================================

    async def identify_similar_memories(
        self,
        candidates: Optional[List[CompactionCandidate]] = None
    ) -> List[MergeGroup]:
        """
        Identify groups of similar memories that can be merged.

        Uses semantic similarity (embeddings) or keyword overlap to group
        memories that contain redundant information.

        Args:
            candidates: Optional pre-identified candidates. If None, scans all.

        Returns:
            List of MergeGroup objects
        """
        merge_groups: List[MergeGroup] = []

        if candidates is None:
            candidates = await self.identify_low_value_memories()

        if len(candidates) < 2:
            return []

        # Get full entries for comparison
        entries: Dict[str, Any] = {}
        for candidate in candidates:
            entry = await self.backend.get(candidate.entry_id, reinforce=False)
            if entry:
                entries[candidate.entry_id] = entry

        if len(entries) < 2:
            return []

        # Calculate similarities
        used_ids: Set[str] = set()
        entry_ids = list(entries.keys())

        for i, id1 in enumerate(entry_ids):
            if id1 in used_ids:
                continue

            group_members = [id1]
            similarities = {id1: 1.0}

            for j, id2 in enumerate(entry_ids):
                if i >= j or id2 in used_ids:
                    continue

                similarity = await self._calculate_similarity(
                    entries[id1], entries[id2]
                )

                if similarity >= self._config.similarity_threshold:
                    group_members.append(id2)
                    similarities[id2] = similarity

                    if len(group_members) >= self._config.max_merge_group_size:
                        break

            # Only create group if we have multiple members
            if len(group_members) > 1:
                # Mark all as used
                for member_id in group_members:
                    used_ids.add(member_id)

                merge_groups.append(MergeGroup(
                    representative_id=id1,
                    member_ids=group_members,
                    similarity_scores=similarities
                ))

        logger.info(f"Identified {len(merge_groups)} merge groups")
        return merge_groups

    async def _calculate_similarity(self, entry1: Any, entry2: Any) -> float:
        """Calculate semantic similarity between two entries."""
        content1 = getattr(entry1, 'content', '')
        content2 = getattr(entry2, 'content', '')

        # Try embedding-based similarity first
        if self._embedding_provider:
            try:
                emb1 = getattr(entry1, 'embedding', None)
                emb2 = getattr(entry2, 'embedding', None)

                if emb1 is None:
                    emb1 = self._embedding_provider(content1)
                if emb2 is None:
                    emb2 = self._embedding_provider(content2)

                return self._cosine_similarity(emb1, emb2)
            except Exception as e:
                logger.debug(f"Embedding similarity failed, using keyword: {e}")

        # Fallback to keyword-based similarity (Jaccard)
        return self._keyword_similarity(content1, content2)

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b) or len(a) == 0:
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def _keyword_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity based on keywords."""
        import re

        # Extract significant words (4+ characters, lowercase)
        words1 = set(w.lower() for w in re.findall(r'\b\w{4,}\b', text1))
        words2 = set(w.lower() for w in re.findall(r'\b\w{4,}\b', text2))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    async def merge_memories(
        self,
        group: MergeGroup,
        strategy: Optional[MergeStrategy] = None
    ) -> str:
        """
        Merge a group of similar memories into one.

        Args:
            group: MergeGroup to merge
            strategy: Merge strategy to use

        Returns:
            ID of the resulting merged memory
        """
        strategy = strategy or self._config.merge_strategy

        # Fetch all entries
        entries: List[Any] = []
        for member_id in group.member_ids:
            entry = await self.backend.get(member_id, reinforce=False)
            if entry:
                entries.append(entry)

        if not entries:
            raise ValueError("No entries found for merge group")

        # Determine representative based on strategy
        if strategy == MergeStrategy.NEWEST_WINS:
            representative = max(entries, key=lambda e: getattr(e, 'created_at', datetime.min))
        elif strategy == MergeStrategy.OLDEST_WINS:
            representative = min(entries, key=lambda e: getattr(e, 'created_at', datetime.max))
        elif strategy == MergeStrategy.HIGHEST_SCORE:
            representative = max(entries, key=lambda e: (
                getattr(e, 'strength', 0) * (getattr(e, 'metadata', {}) or {}).get('importance', 0.5)
            ))
        else:  # COMBINE
            representative = entries[0]

        # Merge metadata
        merged_tags: Set[str] = set()
        merged_metadata: Dict[str, Any] = {}
        total_access_count = 0
        max_importance = 0.0
        max_strength = 0.0

        for entry in entries:
            tags = getattr(entry, 'tags', []) or []
            merged_tags.update(tags)

            metadata = getattr(entry, 'metadata', {}) or {}
            for key, value in metadata.items():
                if key not in merged_metadata:
                    merged_metadata[key] = value

            total_access_count += getattr(entry, 'access_count', 0)
            max_importance = max(max_importance, metadata.get('importance', 0.5))
            max_strength = max(max_strength, getattr(entry, 'strength', 1.0))

        # Update representative with merged data
        representative.tags = list(merged_tags)
        representative.metadata = {
            **merged_metadata,
            'merged_from': [e.id for e in entries if e != representative],
            'merged_at': datetime.now(timezone.utc).isoformat(),
            'importance': max_importance,
        }
        representative.access_count = total_access_count
        representative.strength = max_strength

        # If COMBINE strategy, merge content
        if strategy == MergeStrategy.COMBINE:
            contents = [getattr(e, 'content', '') for e in entries]
            # Simple deduplication - keep unique sentences
            sentences: Set[str] = set()
            for content in contents:
                for sentence in content.split('. '):
                    sentence = sentence.strip()
                    if sentence and len(sentence) > 10:
                        sentences.add(sentence)
            representative.content = '. '.join(sorted(sentences))

        # Save merged entry
        await self.backend.put(representative.id, representative)

        # Delete other entries
        for entry in entries:
            if entry.id != representative.id:
                await self.backend.delete(entry.id)

        logger.debug(f"Merged {len(entries)} memories into {representative.id}")
        return representative.id

    # =========================================================================
    # ARCHIVAL TO COLD STORAGE
    # =========================================================================

    async def archive_to_cold_storage(
        self,
        candidates: List[CompactionCandidate]
    ) -> Tuple[int, int]:
        """
        Archive candidates to cold storage.

        Args:
            candidates: Candidates to archive

        Returns:
            Tuple of (archived_count, bytes_saved)
        """
        if not self._config.enable_cold_storage:
            return 0, 0

        if self._cold_storage is None:
            logger.warning("Cold storage not configured, skipping archive")
            return 0, 0

        archived_count = 0
        bytes_saved = 0

        for candidate in candidates:
            try:
                entry = await self.backend.get(candidate.entry_id, reinforce=False)
                if not entry:
                    continue

                # Mark as archived
                metadata = getattr(entry, 'metadata', {}) or {}
                metadata['archived_at'] = datetime.now(timezone.utc).isoformat()
                metadata['archived_reason'] = candidate.reason
                entry.metadata = metadata

                # Store in cold storage
                await self._cold_storage.put(candidate.entry_id, entry)

                # Delete from primary storage
                content_size = len(getattr(entry, 'content', ''))
                await self.backend.delete(candidate.entry_id)

                archived_count += 1
                bytes_saved += content_size

            except Exception as e:
                logger.warning(f"Failed to archive {candidate.entry_id}: {e}")

        logger.info(f"Archived {archived_count} memories to cold storage")
        return archived_count, bytes_saved

    # =========================================================================
    # INDEX DEFRAGMENTATION
    # =========================================================================

    async def defragment_indexes(self) -> float:
        """
        Defragment database indexes.

        Returns:
            New fragmentation level (0.0-1.0)
        """
        if not self._config.vacuum_after_compact:
            return 0.0

        try:
            # For SQLite backend, run VACUUM
            if hasattr(self.backend, '_get_connection'):
                with self.backend._get_connection() as conn:
                    # Analyze before
                    cursor = conn.execute("PRAGMA freelist_count")
                    freelist_before = cursor.fetchone()[0]
                    cursor = conn.execute("PRAGMA page_count")
                    page_count = cursor.fetchone()[0]

                    frag_before = freelist_before / max(page_count, 1)

                    # Run VACUUM
                    conn.execute("VACUUM")

                    # Reanalyze indexes
                    conn.execute("ANALYZE")

                    # Check after
                    cursor = conn.execute("PRAGMA freelist_count")
                    freelist_after = cursor.fetchone()[0]
                    cursor = conn.execute("PRAGMA page_count")
                    page_count_after = cursor.fetchone()[0]

                    frag_after = freelist_after / max(page_count_after, 1)

                    logger.info(
                        f"Defragmented indexes: {frag_before:.2%} -> {frag_after:.2%}"
                    )
                    return frag_after

        except Exception as e:
            logger.warning(f"Index defragmentation failed: {e}")

        return 0.0

    async def get_fragmentation_level(self) -> float:
        """Get current fragmentation level (0.0-1.0)."""
        try:
            if hasattr(self.backend, '_get_connection'):
                with self.backend._get_connection() as conn:
                    cursor = conn.execute("PRAGMA freelist_count")
                    freelist = cursor.fetchone()[0]
                    cursor = conn.execute("PRAGMA page_count")
                    page_count = cursor.fetchone()[0]
                    return freelist / max(page_count, 1)
        except Exception:
            pass
        return 0.0

    # =========================================================================
    # MAIN COMPACTION
    # =========================================================================

    async def compact(
        self,
        strategy: Optional[CompactionStrategy] = None,
        max_candidates: Optional[int] = None
    ) -> CompactionReport:
        """
        Run memory compaction.

        Args:
            strategy: Compaction strategy to use
            max_candidates: Maximum candidates to process

        Returns:
            CompactionReport with results
        """
        started_at = datetime.now(timezone.utc)
        strategy = strategy or CompactionStrategy.ADAPTIVE
        errors: List[str] = []

        # Get initial stats
        stats = await self.backend.get_stats()
        bytes_before = stats.get('db_size_bytes', 0)
        frag_before = await self.get_fragmentation_level()

        # Identify candidates
        candidates = await self.identify_low_value_memories()
        memories_analyzed = await self.backend.count()

        # Apply strategy-specific filtering
        if strategy == CompactionStrategy.TIME_BASED:
            candidates = [c for c in candidates if c.age_days >= self._config.min_age_days]
        elif strategy == CompactionStrategy.QUALITY_BASED:
            candidates = [c for c in candidates if c.strength < self._config.min_strength_threshold]
        elif strategy == CompactionStrategy.AGGRESSIVE:
            # Use all candidates
            pass
        # ADAPTIVE and SIZE_BASED use score-based selection

        if max_candidates:
            candidates = candidates[:max_candidates]

        # Tracking
        archived_count = 0
        deleted_count = 0
        merged_count = 0
        merge_groups_count = 0
        strength_sum = 0.0
        importance_sum = 0.0
        age_sum = 0.0

        # Phase 1: Identify and process merge groups
        try:
            merge_groups = await self.identify_similar_memories(candidates)
            merge_groups_count = len(merge_groups)

            for group in merge_groups:
                try:
                    await self.merge_memories(group)
                    merged_count += len(group.member_ids) - 1  # -1 for representative

                    # Remove merged items from candidates
                    merged_ids = set(group.member_ids)
                    candidates = [c for c in candidates if c.entry_id not in merged_ids]
                except Exception as e:
                    errors.append(f"Merge failed: {e}")

        except Exception as e:
            errors.append(f"Merge identification failed: {e}")

        # Phase 2: Archive or delete remaining candidates
        archive_candidates = [c for c in candidates if c.priority != CompactionPriority.IMMEDIATE]
        delete_candidates = [c for c in candidates if c.priority == CompactionPriority.IMMEDIATE]

        # Archive
        if archive_candidates:
            try:
                count, _ = await self.archive_to_cold_storage(archive_candidates)
                archived_count = count

                for c in archive_candidates[:count]:
                    strength_sum += c.strength
                    importance_sum += c.importance
                    age_sum += c.age_days

            except Exception as e:
                errors.append(f"Archive failed: {e}")

        # Delete (for immediate priority or if no cold storage)
        for candidate in delete_candidates:
            try:
                await self.backend.delete(candidate.entry_id)
                deleted_count += 1
                strength_sum += candidate.strength
                importance_sum += candidate.importance
                age_sum += candidate.age_days
            except Exception as e:
                errors.append(f"Delete failed for {candidate.entry_id}: {e}")

        # Phase 3: Defragment if needed
        if frag_before >= self._config.defragment_threshold:
            frag_after = await self.defragment_indexes()
        else:
            frag_after = frag_before

        # Get final stats
        stats_after = await self.backend.get_stats()
        bytes_after = stats_after.get('db_size_bytes', 0)

        # Calculate averages
        total_removed = archived_count + deleted_count + merged_count
        avg_strength = strength_sum / max(total_removed, 1)
        avg_importance = importance_sum / max(total_removed, 1)
        avg_age = age_sum / max(total_removed, 1)

        completed_at = datetime.now(timezone.utc)

        report = CompactionReport(
            started_at=started_at,
            completed_at=completed_at,
            strategy_used=strategy,
            memories_analyzed=memories_analyzed,
            candidates_identified=len(candidates) + merged_count,
            memories_archived=archived_count,
            memories_deleted=deleted_count,
            memories_merged=merged_count,
            merge_groups_processed=merge_groups_count,
            bytes_before=bytes_before,
            bytes_after=bytes_after,
            bytes_saved=max(0, bytes_before - bytes_after),
            fragmentation_before=frag_before,
            fragmentation_after=frag_after,
            avg_strength_removed=avg_strength,
            avg_importance_removed=avg_importance,
            avg_age_removed_days=avg_age,
            errors=errors
        )

        # Record metrics
        self._metrics.record_compaction(report)

        logger.info(
            f"Compaction complete: archived={archived_count}, deleted={deleted_count}, "
            f"merged={merged_count}, saved={report.bytes_saved} bytes"
        )

        return report

    async def should_compact(self) -> Tuple[bool, str]:
        """
        Check if compaction should run based on current conditions.

        Returns:
            Tuple of (should_compact, reason)
        """
        stats = await self.backend.get_stats()

        # Size-based check
        count = stats.get('total_memories', 0)
        if count >= self._config.max_memory_count:
            return True, f"Memory count ({count}) exceeds threshold"

        # Storage-based check
        db_size = stats.get('db_size_bytes', 0)
        if db_size >= self._config.max_storage_bytes:
            return True, f"Storage ({db_size} bytes) exceeds threshold"

        # Fragmentation check
        frag = await self.get_fragmentation_level()
        if frag >= self._config.defragment_threshold:
            return True, f"Fragmentation ({frag:.1%}) exceeds threshold"

        # Time-based check
        if self._metrics.last_compaction:
            hours_since = (
                datetime.now(timezone.utc) - self._metrics.last_compaction
            ).total_seconds() / 3600
            if hours_since >= self._config.schedule_interval_hours * 2:
                return True, f"Time since last compaction ({hours_since:.1f}h)"

        return False, "No compaction needed"


# =============================================================================
# BACKGROUND SCHEDULER
# =============================================================================

class CompactionScheduler:
    """
    Background scheduler for automatic memory compaction.

    Monitors memory conditions and triggers compaction when thresholds
    are exceeded.
    """

    def __init__(
        self,
        compactor: MemoryCompactor,
        config: Optional[CompactionConfig] = None
    ) -> None:
        """
        Initialize compaction scheduler.

        Args:
            compactor: Memory compactor instance
            config: Configuration (uses compactor's config if None)
        """
        self.compactor = compactor
        self.config = config or compactor.config

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_check: Optional[datetime] = None
        self._check_count = 0

    async def start(self) -> None:
        """Start the background scheduler."""
        if self._running:
            logger.warning("Compaction scheduler already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            f"Started compaction scheduler (interval={self.config.schedule_interval_hours}h)"
        )

    async def stop(self) -> None:
        """Stop the background scheduler."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("Stopped compaction scheduler")

    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                self._last_check = datetime.now(timezone.utc)
                self._check_count += 1

                # Check if compaction needed
                should_compact, reason = await self.compactor.should_compact()

                if should_compact and self.config.enable_auto_compact:
                    logger.info(f"Triggering automatic compaction: {reason}")

                    try:
                        report = await asyncio.wait_for(
                            self.compactor.compact(strategy=CompactionStrategy.ADAPTIVE),
                            timeout=self.config.max_compact_duration_seconds
                        )
                        logger.info(
                            f"Auto-compaction complete: "
                            f"{report.memories_archived + report.memories_deleted + report.memories_merged} "
                            f"memories compacted"
                        )
                    except asyncio.TimeoutError:
                        logger.warning("Auto-compaction timed out")
                    except Exception as e:
                        logger.error(f"Auto-compaction failed: {e}")

                # Wait for next interval
                await asyncio.sleep(self.config.schedule_interval_hours * 3600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)  # Wait before retry

    async def run_once(self) -> CompactionReport:
        """Run compaction once (manual trigger)."""
        return await self.compactor.compact()

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            "running": self._running,
            "check_count": self._check_count,
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "interval_hours": self.config.schedule_interval_hours,
            "auto_compact_enabled": self.config.enable_auto_compact,
            "metrics": self.compactor.metrics.to_dict(),
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

_compactor_instance: Optional[MemoryCompactor] = None
_scheduler_instance: Optional[CompactionScheduler] = None


def get_memory_compactor(
    backend: Optional[CompactionBackendProtocol] = None,
    config: Optional[CompactionConfig] = None,
    embedding_provider: Optional[Callable[[str], List[float]]] = None
) -> MemoryCompactor:
    """
    Get or create the singleton memory compactor.

    Args:
        backend: Optional backend (uses default if None)
        config: Optional configuration
        embedding_provider: Optional embedding function

    Returns:
        MemoryCompactor instance
    """
    global _compactor_instance
    if _compactor_instance is None:
        _compactor_instance = MemoryCompactor(
            backend=backend,
            config=config,
            embedding_provider=embedding_provider
        )
    return _compactor_instance


def get_compaction_scheduler(
    compactor: Optional[MemoryCompactor] = None
) -> CompactionScheduler:
    """
    Get or create the singleton compaction scheduler.

    Args:
        compactor: Optional compactor (creates one if None)

    Returns:
        CompactionScheduler instance
    """
    global _scheduler_instance
    if _scheduler_instance is None:
        if compactor is None:
            compactor = get_memory_compactor()
        _scheduler_instance = CompactionScheduler(compactor)
    return _scheduler_instance


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def compact_session_memories(
    strategy: CompactionStrategy = CompactionStrategy.ADAPTIVE,
    max_candidates: Optional[int] = None
) -> CompactionReport:
    """
    Convenience function to compact memories for current session.

    Args:
        strategy: Compaction strategy
        max_candidates: Maximum candidates to process

    Returns:
        CompactionReport
    """
    compactor = get_memory_compactor()
    return await compactor.compact(strategy=strategy, max_candidates=max_candidates)


async def get_compaction_candidates() -> List[Dict[str, Any]]:
    """
    Get list of compaction candidates.

    Returns:
        List of candidate dictionaries
    """
    compactor = get_memory_compactor()
    candidates = await compactor.identify_low_value_memories()
    return [c.to_dict() for c in candidates]


async def get_compaction_status() -> Dict[str, Any]:
    """
    Get compaction system status.

    Returns:
        Status dictionary
    """
    compactor = get_memory_compactor()
    should_compact, reason = await compactor.should_compact()

    return {
        "should_compact": should_compact,
        "reason": reason,
        "metrics": compactor.metrics.to_dict(),
        "config": {
            "max_memory_count": compactor.config.max_memory_count,
            "min_age_days": compactor.config.min_age_days,
            "min_strength_threshold": compactor.config.min_strength_threshold,
        }
    }


__all__ = [
    # Enums
    "CompactionStrategy",
    "CompactionPriority",
    "MergeStrategy",

    # Configuration
    "CompactionConfig",

    # Data classes
    "CompactionCandidate",
    "MergeGroup",
    "CompactionReport",
    "CompactionMetrics",

    # Main classes
    "MemoryCompactor",
    "CompactionScheduler",

    # Factory functions
    "get_memory_compactor",
    "get_compaction_scheduler",

    # Convenience functions
    "compact_session_memories",
    "get_compaction_candidates",
    "get_compaction_status",
]
