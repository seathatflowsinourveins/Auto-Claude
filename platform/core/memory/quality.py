"""
Memory Quality Tracking System - V36 Architecture

This module provides comprehensive memory quality tracking, including:
- Relevance scoring based on access patterns and decay
- Stale/conflicting memory detection
- Retrieval quality metrics (NDCG, MRR)
- Consolidation recommendations

Integration:
    from core.memory.quality import (
        MemoryQualityTracker,
        MemoryQualityMetrics,
        ConflictReport,
        RetrievalMetrics,
        ConsolidationRecommendation,
    )

Usage:
    tracker = MemoryQualityTracker(backend)

    # Analyze single memory
    metrics = await tracker.analyze_memory("mem_abc123")

    # Find stale memories
    stale_ids = await tracker.get_stale_memories(threshold=0.3)

    # Detect conflicts
    conflicts = await tracker.detect_conflicts()

    # Measure retrieval quality
    test_queries = [("auth pattern", ["mem_1", "mem_2"])]
    retrieval_metrics = await tracker.measure_retrieval_quality(test_queries)

    # Get consolidation recommendations
    recommendations = await tracker.get_consolidation_recommendations()
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .backends.sqlite import SQLiteTierBackend
    from .backends.base import MemoryEntry

logger = logging.getLogger(__name__)


# =============================================================================
# QUALITY METRIC DATA CLASSES
# =============================================================================

class ConflictType(str, Enum):
    """Types of memory conflicts."""
    CONTRADICTORY = "contradictory"      # Directly conflicting information
    DUPLICATE = "duplicate"              # Near-duplicate content
    SUPERSEDED = "superseded"           # One memory supersedes another
    TEMPORAL_OVERLAP = "temporal_overlap"  # Conflicting time-based info


class ConsolidationAction(str, Enum):
    """Recommended consolidation actions."""
    ARCHIVE = "archive"          # Move to archival tier
    DELETE = "delete"            # Remove entirely
    MERGE = "merge"              # Merge with another memory
    UPDATE = "update"            # Update content
    KEEP = "keep"                # No action needed
    DEMOTE = "demote"            # Reduce priority
    PROMOTE = "promote"          # Increase priority


@dataclass
class MemoryQualityMetrics:
    """Quality metrics for a single memory entry."""
    id: str
    relevance_score: float       # 0-1 based on access patterns
    freshness_score: float       # 0-1 based on age
    consistency_score: float     # 0-1 based on conflict detection
    overall_quality: float       # Weighted combination

    # Additional details
    access_count: int = 0
    days_since_access: float = 0.0
    days_since_creation: float = 0.0
    conflict_count: int = 0

    # Computed flags
    is_stale: bool = False
    is_orphaned: bool = False    # No related memories
    needs_review: bool = False

    # Metadata
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "relevance_score": round(self.relevance_score, 4),
            "freshness_score": round(self.freshness_score, 4),
            "consistency_score": round(self.consistency_score, 4),
            "overall_quality": round(self.overall_quality, 4),
            "access_count": self.access_count,
            "days_since_access": round(self.days_since_access, 2),
            "days_since_creation": round(self.days_since_creation, 2),
            "conflict_count": self.conflict_count,
            "is_stale": self.is_stale,
            "is_orphaned": self.is_orphaned,
            "needs_review": self.needs_review,
            "computed_at": self.computed_at.isoformat(),
        }


@dataclass
class ConflictReport:
    """Report of conflicting memories."""
    memory_id_1: str
    memory_id_2: str
    conflict_type: ConflictType
    confidence: float            # 0-1 confidence in conflict detection
    description: str
    suggested_resolution: str

    # Content excerpts for review
    content_1_excerpt: str = ""
    content_2_excerpt: str = ""

    # Timestamps for context
    created_at_1: Optional[datetime] = None
    created_at_2: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "memory_id_1": self.memory_id_1,
            "memory_id_2": self.memory_id_2,
            "conflict_type": self.conflict_type.value,
            "confidence": round(self.confidence, 4),
            "description": self.description,
            "suggested_resolution": self.suggested_resolution,
            "content_1_excerpt": self.content_1_excerpt,
            "content_2_excerpt": self.content_2_excerpt,
            "created_at_1": self.created_at_1.isoformat() if self.created_at_1 else None,
            "created_at_2": self.created_at_2.isoformat() if self.created_at_2 else None,
        }


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval quality assessment."""
    ndcg: float                  # Normalized Discounted Cumulative Gain
    mrr: float                   # Mean Reciprocal Rank
    precision_at_k: Dict[int, float]  # Precision at k (k=1,3,5,10)
    recall_at_k: Dict[int, float]     # Recall at k (k=1,3,5,10)

    # Per-query breakdown
    query_count: int = 0
    queries_with_results: int = 0
    average_result_count: float = 0.0

    # Timing
    average_query_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "ndcg": round(self.ndcg, 4),
            "mrr": round(self.mrr, 4),
            "precision_at_k": {k: round(v, 4) for k, v in self.precision_at_k.items()},
            "recall_at_k": {k: round(v, 4) for k, v in self.recall_at_k.items()},
            "query_count": self.query_count,
            "queries_with_results": self.queries_with_results,
            "average_result_count": round(self.average_result_count, 2),
            "average_query_time_ms": round(self.average_query_time_ms, 2),
        }


@dataclass
class ConsolidationRecommendation:
    """Recommendation for memory consolidation."""
    memory_id: str
    action: ConsolidationAction
    reason: str
    priority: int                # 1 (highest) to 5 (lowest)

    # Related memories (for merge/update actions)
    related_ids: List[str] = field(default_factory=list)

    # Quality context
    quality_score: float = 0.0
    freshness_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "memory_id": self.memory_id,
            "action": self.action.value,
            "reason": self.reason,
            "priority": self.priority,
            "related_ids": self.related_ids,
            "quality_score": round(self.quality_score, 4),
            "freshness_score": round(self.freshness_score, 4),
        }


@dataclass
class QualityReport:
    """Comprehensive quality report for memory system."""
    total_memories: int
    healthy_count: int
    stale_count: int
    conflicting_count: int
    orphaned_count: int

    # Score distributions
    average_quality: float
    quality_distribution: Dict[str, int]  # excellent/good/fair/poor counts

    # Recommendations
    recommendations: List[ConsolidationRecommendation]

    # Retrieval quality (if tested)
    retrieval_metrics: Optional[RetrievalMetrics] = None

    # Timing
    analysis_time_ms: float = 0.0
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_memories": self.total_memories,
            "healthy_count": self.healthy_count,
            "stale_count": self.stale_count,
            "conflicting_count": self.conflicting_count,
            "orphaned_count": self.orphaned_count,
            "average_quality": round(self.average_quality, 4),
            "quality_distribution": self.quality_distribution,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "retrieval_metrics": self.retrieval_metrics.to_dict() if self.retrieval_metrics else None,
            "analysis_time_ms": round(self.analysis_time_ms, 2),
            "generated_at": self.generated_at.isoformat(),
        }


# =============================================================================
# QUALITY CONFIGURATION
# =============================================================================

@dataclass
class QualityConfig:
    """Configuration for quality tracking."""
    # Score weights for overall quality calculation
    relevance_weight: float = 0.4
    freshness_weight: float = 0.3
    consistency_weight: float = 0.3

    # Decay parameters
    access_decay_days: float = 30.0       # Days for 50% relevance decay
    freshness_decay_days: float = 90.0    # Days for 50% freshness decay

    # Thresholds
    stale_threshold: float = 0.3          # Below this = stale
    conflict_similarity_threshold: float = 0.85  # Above this = potential duplicate
    orphan_threshold_days: float = 180.0  # No access in this time = orphan

    # Retrieval quality
    retrieval_k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])


# =============================================================================
# MEMORY QUALITY TRACKER
# =============================================================================

class MemoryQualityTracker:
    """
    Tracks and reports on memory quality.

    Provides:
    - Individual memory quality analysis
    - Stale memory detection
    - Conflict detection across memories
    - Retrieval quality measurement (NDCG, MRR)
    - Consolidation recommendations
    """

    def __init__(
        self,
        backend: Optional["SQLiteTierBackend"] = None,
        config: Optional[QualityConfig] = None
    ) -> None:
        """Initialize the quality tracker.

        Args:
            backend: SQLite backend for memory access. If None, uses singleton.
            config: Quality configuration. Uses defaults if None.
        """
        self._backend = backend
        self._config = config or QualityConfig()
        self._cache: Dict[str, MemoryQualityMetrics] = {}
        self._cache_ttl_seconds = 300  # 5 minute cache
        self._cache_time: Dict[str, datetime] = {}

    @property
    def backend(self) -> "SQLiteTierBackend":
        """Get the backend, initializing if needed."""
        if self._backend is None:
            from .backends.sqlite import get_sqlite_backend
            self._backend = get_sqlite_backend()
        return self._backend

    # =========================================================================
    # INDIVIDUAL MEMORY ANALYSIS
    # =========================================================================

    async def analyze_memory(self, memory_id: str) -> MemoryQualityMetrics:
        """Analyze quality of a single memory.

        Args:
            memory_id: ID of the memory to analyze

        Returns:
            MemoryQualityMetrics with comprehensive quality scores

        Raises:
            ValueError: If memory not found
        """
        # Check cache
        if memory_id in self._cache:
            cache_time = self._cache_time.get(memory_id)
            if cache_time:
                age = (datetime.now(timezone.utc) - cache_time).total_seconds()
                if age < self._cache_ttl_seconds:
                    return self._cache[memory_id]

        # Fetch memory
        entry = await self.backend.get(memory_id)
        if entry is None:
            raise ValueError(f"Memory not found: {memory_id}")

        now = datetime.now(timezone.utc)

        # Calculate days since access/creation
        days_since_access = 0.0
        if entry.last_accessed:
            days_since_access = (now - entry.last_accessed).total_seconds() / 86400

        days_since_creation = 0.0
        if entry.created_at:
            days_since_creation = (now - entry.created_at).total_seconds() / 86400

        # Calculate relevance score based on access patterns
        relevance_score = self._calculate_relevance_score(
            access_count=entry.access_count,
            days_since_access=days_since_access,
            days_since_creation=days_since_creation,
        )

        # Calculate freshness score based on age
        freshness_score = self._calculate_freshness_score(days_since_creation)

        # Calculate consistency score (check for conflicts)
        consistency_score, conflict_count = await self._calculate_consistency_score(entry)

        # Calculate overall quality (weighted combination)
        overall_quality = (
            self._config.relevance_weight * relevance_score +
            self._config.freshness_weight * freshness_score +
            self._config.consistency_weight * consistency_score
        )

        # Determine flags
        is_stale = overall_quality < self._config.stale_threshold
        is_orphaned = days_since_access > self._config.orphan_threshold_days
        needs_review = is_stale or conflict_count > 0 or is_orphaned

        metrics = MemoryQualityMetrics(
            id=memory_id,
            relevance_score=relevance_score,
            freshness_score=freshness_score,
            consistency_score=consistency_score,
            overall_quality=overall_quality,
            access_count=entry.access_count,
            days_since_access=days_since_access,
            days_since_creation=days_since_creation,
            conflict_count=conflict_count,
            is_stale=is_stale,
            is_orphaned=is_orphaned,
            needs_review=needs_review,
        )

        # Cache result
        self._cache[memory_id] = metrics
        self._cache_time[memory_id] = now

        return metrics

    def _calculate_relevance_score(
        self,
        access_count: int,
        days_since_access: float,
        days_since_creation: float,
    ) -> float:
        """Calculate relevance score based on access patterns.

        Uses exponential decay with access frequency boost.
        """
        # Base score from access frequency
        # Log scale to avoid over-weighting high access counts
        frequency_score = min(1.0, math.log1p(access_count) / math.log1p(100))

        # Time decay factor (exponential decay)
        decay_rate = math.log(2) / self._config.access_decay_days
        time_factor = math.exp(-decay_rate * days_since_access)

        # Combine with access boost
        # Recent accesses boost relevance more than old accesses
        if access_count == 0:
            # Never accessed - use creation time
            age_penalty = math.exp(-decay_rate * days_since_creation * 0.5)
            return 0.3 * age_penalty

        relevance = frequency_score * 0.4 + time_factor * 0.6
        return max(0.0, min(1.0, relevance))

    def _calculate_freshness_score(self, days_since_creation: float) -> float:
        """Calculate freshness score based on memory age.

        Uses exponential decay model.
        """
        decay_rate = math.log(2) / self._config.freshness_decay_days
        freshness = math.exp(-decay_rate * days_since_creation)
        return max(0.0, min(1.0, freshness))

    async def _calculate_consistency_score(
        self,
        entry: "MemoryEntry"
    ) -> Tuple[float, int]:
        """Calculate consistency score by checking for conflicts.

        Returns:
            Tuple of (consistency_score, conflict_count)
        """
        # Search for potentially conflicting memories
        conflicts = await self._find_conflicts_for_memory(entry)
        conflict_count = len(conflicts)

        if conflict_count == 0:
            return 1.0, 0

        # Penalize based on conflict severity
        # More conflicts and higher confidence = lower consistency
        total_severity = sum(c.confidence for c in conflicts)
        max_severity = conflict_count * 1.0  # Max confidence per conflict

        consistency = 1.0 - (total_severity / max_severity) * 0.8
        return max(0.0, min(1.0, consistency)), conflict_count

    # =========================================================================
    # STALE MEMORY DETECTION
    # =========================================================================

    async def get_stale_memories(self, threshold: float = 0.3) -> List[str]:
        """Find memories that should be archived or removed.

        Args:
            threshold: Quality threshold below which memories are considered stale

        Returns:
            List of memory IDs that are stale
        """
        stale_ids: List[str] = []

        all_memories = await self.backend.list_all()

        for entry in all_memories:
            try:
                metrics = await self.analyze_memory(entry.id)
                if metrics.overall_quality < threshold:
                    stale_ids.append(entry.id)
            except Exception as e:
                logger.warning(f"Failed to analyze memory {entry.id}: {e}")

        return stale_ids

    async def get_orphaned_memories(self) -> List[str]:
        """Find memories that haven't been accessed in a long time.

        Returns:
            List of orphaned memory IDs
        """
        orphaned_ids: List[str] = []
        threshold_days = self._config.orphan_threshold_days
        now = datetime.now(timezone.utc)

        all_memories = await self.backend.list_all()

        for entry in all_memories:
            if entry.last_accessed:
                days_since = (now - entry.last_accessed).total_seconds() / 86400
                if days_since > threshold_days:
                    orphaned_ids.append(entry.id)
            elif entry.created_at:
                # Never accessed, check creation date
                days_since = (now - entry.created_at).total_seconds() / 86400
                if days_since > threshold_days:
                    orphaned_ids.append(entry.id)

        return orphaned_ids

    # =========================================================================
    # CONFLICT DETECTION
    # =========================================================================

    async def detect_conflicts(self) -> List[ConflictReport]:
        """Detect conflicting information across memories.

        Checks for:
        - Duplicate/near-duplicate content
        - Contradictory information
        - Superseded information

        Returns:
            List of ConflictReport objects
        """
        conflicts: List[ConflictReport] = []
        all_memories = await self.backend.list_all()

        # Build comparison pairs (avoid O(n^2) by using content hashing)
        content_groups: Dict[str, List["MemoryEntry"]] = {}

        for entry in all_memories:
            # Group by content similarity hash (first 100 chars normalized)
            content_key = self._content_similarity_key(entry.content)
            if content_key not in content_groups:
                content_groups[content_key] = []
            content_groups[content_key].append(entry)

        # Check within groups for duplicates
        for group_entries in content_groups.values():
            if len(group_entries) > 1:
                # Check pairwise within group
                for i, entry1 in enumerate(group_entries):
                    for entry2 in group_entries[i + 1:]:
                        conflict = self._check_duplicate_conflict(entry1, entry2)
                        if conflict:
                            conflicts.append(conflict)

        # Check for contradictory information across all memories
        # (This is a simplified check - real implementation would use NLP)
        contradiction_conflicts = await self._detect_contradictions(all_memories)
        conflicts.extend(contradiction_conflicts)

        # Check for superseded information (newer info replaces older)
        superseded_conflicts = self._detect_superseded(all_memories)
        conflicts.extend(superseded_conflicts)

        return conflicts

    async def _find_conflicts_for_memory(
        self,
        entry: "MemoryEntry"
    ) -> List[ConflictReport]:
        """Find conflicts for a specific memory entry."""
        conflicts: List[ConflictReport] = []

        # Search for similar content
        try:
            similar = await self.backend.search(entry.content[:100], limit=10)
        except Exception:
            similar = []

        for other in similar:
            if other.id == entry.id:
                continue

            # Check for duplicate
            duplicate = self._check_duplicate_conflict(entry, other)
            if duplicate:
                conflicts.append(duplicate)
                continue

            # Check for contradiction (simplified)
            contradiction = self._check_contradiction(entry, other)
            if contradiction:
                conflicts.append(contradiction)

        return conflicts

    def _content_similarity_key(self, content: str) -> str:
        """Generate a key for grouping similar content."""
        # Normalize: lowercase, remove punctuation, take first 100 chars
        normalized = re.sub(r'[^\w\s]', '', content.lower())[:100]
        # Hash the first few words
        words = normalized.split()[:10]
        return hashlib.md5(' '.join(words).encode()).hexdigest()[:8]

    def _check_duplicate_conflict(
        self,
        entry1: "MemoryEntry",
        entry2: "MemoryEntry"
    ) -> Optional[ConflictReport]:
        """Check if two entries are duplicates."""
        similarity = SequenceMatcher(
            None,
            entry1.content.lower(),
            entry2.content.lower()
        ).ratio()

        if similarity >= self._config.conflict_similarity_threshold:
            # Determine which is newer
            newer_id = entry1.id
            older_id = entry2.id
            if entry1.created_at and entry2.created_at:
                if entry2.created_at > entry1.created_at:
                    newer_id, older_id = older_id, newer_id

            return ConflictReport(
                memory_id_1=entry1.id,
                memory_id_2=entry2.id,
                conflict_type=ConflictType.DUPLICATE,
                confidence=similarity,
                description=f"Near-duplicate content detected ({similarity:.1%} similar)",
                suggested_resolution=f"Consider merging or removing {older_id} (older entry)",
                content_1_excerpt=entry1.content[:200],
                content_2_excerpt=entry2.content[:200],
                created_at_1=entry1.created_at,
                created_at_2=entry2.created_at,
            )

        return None

    def _check_contradiction(
        self,
        entry1: "MemoryEntry",
        entry2: "MemoryEntry"
    ) -> Optional[ConflictReport]:
        """Check for contradictory information between entries.

        This is a simplified heuristic check. A production system would use
        NLP/LLM for semantic contradiction detection.
        """
        # Simple heuristic: look for negation patterns
        negation_patterns = [
            (r'\bis\b', r'\bis not\b'),
            (r'\bwill\b', r'\bwill not\b'),
            (r'\bshould\b', r'\bshould not\b'),
            (r'\bcan\b', r'\bcannot\b'),
            (r'\btrue\b', r'\bfalse\b'),
            (r'\benabled?\b', r'\bdisabled?\b'),
            (r'\byes\b', r'\bno\b'),
        ]

        content1_lower = entry1.content.lower()
        content2_lower = entry2.content.lower()

        for pos_pattern, neg_pattern in negation_patterns:
            pos_in_1 = bool(re.search(pos_pattern, content1_lower))
            neg_in_1 = bool(re.search(neg_pattern, content1_lower))
            pos_in_2 = bool(re.search(pos_pattern, content2_lower))
            neg_in_2 = bool(re.search(neg_pattern, content2_lower))

            # Check for opposite patterns
            if (pos_in_1 and neg_in_2) or (neg_in_1 and pos_in_2):
                # Check if they're about similar topics
                topic_similarity = SequenceMatcher(
                    None,
                    content1_lower[:100],
                    content2_lower[:100]
                ).ratio()

                if topic_similarity > 0.3:  # Related topics
                    return ConflictReport(
                        memory_id_1=entry1.id,
                        memory_id_2=entry2.id,
                        conflict_type=ConflictType.CONTRADICTORY,
                        confidence=0.6,  # Heuristic-based, moderate confidence
                        description="Potential contradictory information detected",
                        suggested_resolution="Review both memories and determine which is current/correct",
                        content_1_excerpt=entry1.content[:200],
                        content_2_excerpt=entry2.content[:200],
                        created_at_1=entry1.created_at,
                        created_at_2=entry2.created_at,
                    )

        return None

    async def _detect_contradictions(
        self,
        memories: List["MemoryEntry"]
    ) -> List[ConflictReport]:
        """Detect contradictions across all memories."""
        conflicts: List[ConflictReport] = []

        # Group by tags for more targeted comparison
        tag_groups: Dict[str, List["MemoryEntry"]] = {}
        for entry in memories:
            for tag in entry.tags:
                if tag not in tag_groups:
                    tag_groups[tag] = []
                tag_groups[tag].append(entry)

        # Check within tag groups (more likely to have related info)
        checked_pairs: Set[Tuple[str, str]] = set()

        for tag_entries in tag_groups.values():
            if len(tag_entries) > 1:
                for i, entry1 in enumerate(tag_entries):
                    for entry2 in tag_entries[i + 1:]:
                        pair_key = tuple(sorted([entry1.id, entry2.id]))
                        if pair_key in checked_pairs:
                            continue
                        checked_pairs.add(pair_key)

                        conflict = self._check_contradiction(entry1, entry2)
                        if conflict:
                            conflicts.append(conflict)

        return conflicts

    def _detect_superseded(
        self,
        memories: List["MemoryEntry"]
    ) -> List[ConflictReport]:
        """Detect memories that have been superseded by newer information."""
        conflicts: List[ConflictReport] = []

        # Group by content type and tags
        type_tag_groups: Dict[str, List["MemoryEntry"]] = {}

        for entry in memories:
            key = f"{entry.content_type}:{','.join(sorted(entry.tags[:3]))}"
            if key not in type_tag_groups:
                type_tag_groups[key] = []
            type_tag_groups[key].append(entry)

        for group_entries in type_tag_groups.values():
            if len(group_entries) < 2:
                continue

            # Sort by creation time
            sorted_entries = sorted(
                [e for e in group_entries if e.created_at],
                key=lambda x: x.created_at,
                reverse=True  # Newest first
            )

            if len(sorted_entries) < 2:
                continue

            # Check if newer entries reference or update older ones
            for i, newer in enumerate(sorted_entries[:-1]):
                for older in sorted_entries[i + 1:]:
                    # Check content overlap
                    overlap = SequenceMatcher(
                        None,
                        newer.content[:200].lower(),
                        older.content[:200].lower()
                    ).ratio()

                    if 0.4 < overlap < 0.85:  # Partial overlap = possible update
                        # Check if newer is more recent by significant margin
                        if newer.created_at and older.created_at:
                            days_diff = (newer.created_at - older.created_at).days
                            if days_diff > 7:  # At least a week apart
                                conflicts.append(ConflictReport(
                                    memory_id_1=newer.id,
                                    memory_id_2=older.id,
                                    conflict_type=ConflictType.SUPERSEDED,
                                    confidence=0.5 + overlap * 0.3,
                                    description=f"Newer memory may supersede older ({days_diff} days apart)",
                                    suggested_resolution=f"Review if {older.id} should be archived",
                                    content_1_excerpt=newer.content[:200],
                                    content_2_excerpt=older.content[:200],
                                    created_at_1=newer.created_at,
                                    created_at_2=older.created_at,
                                ))

        return conflicts

    # =========================================================================
    # RETRIEVAL QUALITY METRICS
    # =========================================================================

    async def measure_retrieval_quality(
        self,
        queries: List[Tuple[str, List[str]]]
    ) -> RetrievalMetrics:
        """Measure NDCG and MRR for a set of test queries.

        Args:
            queries: List of (query_text, expected_memory_ids) tuples.
                     expected_memory_ids should be in relevance order.

        Returns:
            RetrievalMetrics with NDCG, MRR, precision@k, recall@k
        """
        if not queries:
            return RetrievalMetrics(
                ndcg=0.0,
                mrr=0.0,
                precision_at_k={k: 0.0 for k in self._config.retrieval_k_values},
                recall_at_k={k: 0.0 for k in self._config.retrieval_k_values},
            )

        ndcg_scores: List[float] = []
        rr_scores: List[float] = []  # Reciprocal ranks
        precision_sums: Dict[int, float] = {k: 0.0 for k in self._config.retrieval_k_values}
        recall_sums: Dict[int, float] = {k: 0.0 for k in self._config.retrieval_k_values}

        total_time_ms = 0.0
        total_results = 0
        queries_with_results = 0

        for query_text, expected_ids in queries:
            if not expected_ids:
                continue

            # Execute search
            import time
            start_time = time.time()
            results = await self.backend.search(query_text, limit=max(self._config.retrieval_k_values))
            elapsed_ms = (time.time() - start_time) * 1000
            total_time_ms += elapsed_ms

            result_ids = [r.id for r in results]
            total_results += len(result_ids)

            if result_ids:
                queries_with_results += 1

            # Calculate NDCG
            ndcg = self._calculate_ndcg(result_ids, expected_ids)
            ndcg_scores.append(ndcg)

            # Calculate Reciprocal Rank
            rr = self._calculate_reciprocal_rank(result_ids, expected_ids)
            rr_scores.append(rr)

            # Calculate Precision@K and Recall@K
            for k in self._config.retrieval_k_values:
                precision_sums[k] += self._calculate_precision_at_k(result_ids, expected_ids, k)
                recall_sums[k] += self._calculate_recall_at_k(result_ids, expected_ids, k)

        query_count = len(queries)

        return RetrievalMetrics(
            ndcg=sum(ndcg_scores) / query_count if query_count else 0.0,
            mrr=sum(rr_scores) / query_count if query_count else 0.0,
            precision_at_k={k: v / query_count for k, v in precision_sums.items()},
            recall_at_k={k: v / query_count for k, v in recall_sums.items()},
            query_count=query_count,
            queries_with_results=queries_with_results,
            average_result_count=total_results / query_count if query_count else 0.0,
            average_query_time_ms=total_time_ms / query_count if query_count else 0.0,
        )

    def _calculate_ndcg(
        self,
        result_ids: List[str],
        expected_ids: List[str]
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain.

        NDCG measures ranking quality, considering position in results.
        """
        if not expected_ids:
            return 0.0

        # Create relevance scores (position-based: first expected = highest relevance)
        relevance_map = {
            id_: len(expected_ids) - i
            for i, id_ in enumerate(expected_ids)
        }

        # Calculate DCG
        dcg = 0.0
        for i, result_id in enumerate(result_ids):
            rel = relevance_map.get(result_id, 0)
            if rel > 0:
                dcg += rel / math.log2(i + 2)  # Position 1 = log2(2)

        # Calculate ideal DCG (all expected results in perfect order)
        idcg = 0.0
        for i, expected_id in enumerate(expected_ids[:len(result_ids)]):
            rel = len(expected_ids) - i
            idcg += rel / math.log2(i + 2)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def _calculate_reciprocal_rank(
        self,
        result_ids: List[str],
        expected_ids: List[str]
    ) -> float:
        """Calculate Reciprocal Rank (1/position of first relevant result)."""
        expected_set = set(expected_ids)

        for i, result_id in enumerate(result_ids):
            if result_id in expected_set:
                return 1.0 / (i + 1)

        return 0.0

    def _calculate_precision_at_k(
        self,
        result_ids: List[str],
        expected_ids: List[str],
        k: int
    ) -> float:
        """Calculate Precision@K (relevant in top-k / k)."""
        if k == 0:
            return 0.0

        expected_set = set(expected_ids)
        top_k = result_ids[:k]

        relevant_count = sum(1 for r in top_k if r in expected_set)
        return relevant_count / k

    def _calculate_recall_at_k(
        self,
        result_ids: List[str],
        expected_ids: List[str],
        k: int
    ) -> float:
        """Calculate Recall@K (relevant in top-k / total relevant)."""
        if not expected_ids:
            return 0.0

        expected_set = set(expected_ids)
        top_k = result_ids[:k]

        relevant_count = sum(1 for r in top_k if r in expected_set)
        return relevant_count / len(expected_ids)

    # =========================================================================
    # CONSOLIDATION RECOMMENDATIONS
    # =========================================================================

    async def get_consolidation_recommendations(
        self,
        max_recommendations: int = 50
    ) -> List[ConsolidationRecommendation]:
        """Get recommendations for memory consolidation.

        Analyzes all memories and provides actionable recommendations.

        Args:
            max_recommendations: Maximum number of recommendations to return

        Returns:
            List of ConsolidationRecommendation sorted by priority
        """
        recommendations: List[ConsolidationRecommendation] = []

        all_memories = await self.backend.list_all()
        conflicts = await self.detect_conflicts()

        # Build conflict map
        conflict_map: Dict[str, List[ConflictReport]] = {}
        for conflict in conflicts:
            for mem_id in [conflict.memory_id_1, conflict.memory_id_2]:
                if mem_id not in conflict_map:
                    conflict_map[mem_id] = []
                conflict_map[mem_id].append(conflict)

        # Analyze each memory
        for entry in all_memories:
            try:
                metrics = await self.analyze_memory(entry.id)
            except Exception as e:
                logger.warning(f"Failed to analyze {entry.id}: {e}")
                continue

            # Determine recommendation
            recommendation = self._determine_recommendation(
                entry,
                metrics,
                conflict_map.get(entry.id, [])
            )

            if recommendation:
                recommendations.append(recommendation)

        # Sort by priority and limit
        recommendations.sort(key=lambda r: (r.priority, -r.quality_score))
        return recommendations[:max_recommendations]

    def _determine_recommendation(
        self,
        entry: "MemoryEntry",
        metrics: MemoryQualityMetrics,
        conflicts: List[ConflictReport]
    ) -> Optional[ConsolidationRecommendation]:
        """Determine consolidation recommendation for a memory."""
        # High quality, no issues - no recommendation needed
        if metrics.overall_quality > 0.7 and not conflicts and not metrics.is_orphaned:
            return None

        # Check for duplicates first
        duplicate_conflicts = [
            c for c in conflicts if c.conflict_type == ConflictType.DUPLICATE
        ]
        if duplicate_conflicts:
            return ConsolidationRecommendation(
                memory_id=entry.id,
                action=ConsolidationAction.MERGE,
                reason=f"Duplicate of {duplicate_conflicts[0].memory_id_2}",
                priority=2,
                related_ids=[c.memory_id_2 for c in duplicate_conflicts if c.memory_id_2 != entry.id],
                quality_score=metrics.overall_quality,
                freshness_score=metrics.freshness_score,
            )

        # Check for contradictions
        contradiction_conflicts = [
            c for c in conflicts if c.conflict_type == ConflictType.CONTRADICTORY
        ]
        if contradiction_conflicts:
            return ConsolidationRecommendation(
                memory_id=entry.id,
                action=ConsolidationAction.UPDATE,
                reason=f"Contradicts {contradiction_conflicts[0].memory_id_2}",
                priority=1,  # High priority - contradictions need resolution
                related_ids=[c.memory_id_2 for c in contradiction_conflicts if c.memory_id_2 != entry.id],
                quality_score=metrics.overall_quality,
                freshness_score=metrics.freshness_score,
            )

        # Check for superseded
        superseded_conflicts = [
            c for c in conflicts if c.conflict_type == ConflictType.SUPERSEDED
        ]
        if superseded_conflicts:
            # Check if this is the older entry
            for c in superseded_conflicts:
                if c.memory_id_2 == entry.id:  # This is the older one
                    return ConsolidationRecommendation(
                        memory_id=entry.id,
                        action=ConsolidationAction.ARCHIVE,
                        reason=f"Superseded by {c.memory_id_1}",
                        priority=3,
                        related_ids=[c.memory_id_1],
                        quality_score=metrics.overall_quality,
                        freshness_score=metrics.freshness_score,
                    )

        # Very stale memories
        if metrics.overall_quality < 0.1:
            return ConsolidationRecommendation(
                memory_id=entry.id,
                action=ConsolidationAction.DELETE,
                reason="Very low quality score, likely obsolete",
                priority=4,
                quality_score=metrics.overall_quality,
                freshness_score=metrics.freshness_score,
            )

        # Stale but not critical
        if metrics.is_stale:
            return ConsolidationRecommendation(
                memory_id=entry.id,
                action=ConsolidationAction.ARCHIVE,
                reason=f"Quality score ({metrics.overall_quality:.2f}) below threshold",
                priority=4,
                quality_score=metrics.overall_quality,
                freshness_score=metrics.freshness_score,
            )

        # Orphaned memories
        if metrics.is_orphaned:
            return ConsolidationRecommendation(
                memory_id=entry.id,
                action=ConsolidationAction.DEMOTE,
                reason=f"No access in {metrics.days_since_access:.0f} days",
                priority=5,
                quality_score=metrics.overall_quality,
                freshness_score=metrics.freshness_score,
            )

        return None

    # =========================================================================
    # COMPREHENSIVE REPORT
    # =========================================================================

    async def generate_quality_report(
        self,
        include_retrieval_test: bool = False,
        test_queries: Optional[List[Tuple[str, List[str]]]] = None
    ) -> QualityReport:
        """Generate comprehensive quality report for the memory system.

        Args:
            include_retrieval_test: Whether to include retrieval quality metrics
            test_queries: Test queries for retrieval evaluation

        Returns:
            QualityReport with full analysis
        """
        import time
        start_time = time.time()

        all_memories = await self.backend.list_all()
        total_memories = len(all_memories)

        # Analyze all memories
        quality_scores: List[float] = []
        stale_count = 0
        orphaned_count = 0

        for entry in all_memories:
            try:
                metrics = await self.analyze_memory(entry.id)
                quality_scores.append(metrics.overall_quality)
                if metrics.is_stale:
                    stale_count += 1
                if metrics.is_orphaned:
                    orphaned_count += 1
            except Exception as e:
                logger.warning(f"Failed to analyze {entry.id}: {e}")

        # Detect conflicts
        conflicts = await self.detect_conflicts()
        conflicting_ids = set()
        for conflict in conflicts:
            conflicting_ids.add(conflict.memory_id_1)
            conflicting_ids.add(conflict.memory_id_2)

        # Get recommendations
        recommendations = await self.get_consolidation_recommendations()

        # Calculate quality distribution
        excellent = sum(1 for q in quality_scores if q >= 0.8)
        good = sum(1 for q in quality_scores if 0.6 <= q < 0.8)
        fair = sum(1 for q in quality_scores if 0.4 <= q < 0.6)
        poor = sum(1 for q in quality_scores if q < 0.4)

        # Retrieval metrics (optional)
        retrieval_metrics = None
        if include_retrieval_test and test_queries:
            retrieval_metrics = await self.measure_retrieval_quality(test_queries)

        analysis_time_ms = (time.time() - start_time) * 1000

        return QualityReport(
            total_memories=total_memories,
            healthy_count=total_memories - stale_count - orphaned_count,
            stale_count=stale_count,
            conflicting_count=len(conflicting_ids),
            orphaned_count=orphaned_count,
            average_quality=sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
            quality_distribution={
                "excellent": excellent,
                "good": good,
                "fair": fair,
                "poor": poor,
            },
            recommendations=recommendations,
            retrieval_metrics=retrieval_metrics,
            analysis_time_ms=analysis_time_ms,
        )

    # =========================================================================
    # CACHE MANAGEMENT
    # =========================================================================

    def clear_cache(self) -> None:
        """Clear the quality metrics cache."""
        self._cache.clear()
        self._cache_time.clear()

    def set_cache_ttl(self, ttl_seconds: int) -> None:
        """Set cache TTL in seconds."""
        self._cache_ttl_seconds = ttl_seconds


# =============================================================================
# FACTORY AND CONVENIENCE FUNCTIONS
# =============================================================================

_tracker_instance: Optional[MemoryQualityTracker] = None


def get_quality_tracker(
    backend: Optional["SQLiteTierBackend"] = None,
    config: Optional[QualityConfig] = None
) -> MemoryQualityTracker:
    """Get or create the singleton quality tracker.

    Args:
        backend: Optional backend (uses singleton if None)
        config: Optional config (uses defaults if None)

    Returns:
        MemoryQualityTracker instance
    """
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = MemoryQualityTracker(backend, config)
    return _tracker_instance


async def analyze_memory_quality(memory_id: str) -> MemoryQualityMetrics:
    """Convenience function to analyze a single memory.

    Args:
        memory_id: ID of the memory to analyze

    Returns:
        MemoryQualityMetrics
    """
    tracker = get_quality_tracker()
    return await tracker.analyze_memory(memory_id)


async def get_stale_memories(threshold: float = 0.3) -> List[str]:
    """Convenience function to get stale memory IDs.

    Args:
        threshold: Quality threshold

    Returns:
        List of stale memory IDs
    """
    tracker = get_quality_tracker()
    return await tracker.get_stale_memories(threshold)


async def detect_memory_conflicts() -> List[ConflictReport]:
    """Convenience function to detect memory conflicts.

    Returns:
        List of ConflictReport objects
    """
    tracker = get_quality_tracker()
    return await tracker.detect_conflicts()


async def get_memory_quality_report() -> QualityReport:
    """Convenience function to generate full quality report.

    Returns:
        QualityReport with comprehensive analysis
    """
    tracker = get_quality_tracker()
    return await tracker.generate_quality_report()


__all__ = [
    # Enums
    "ConflictType",
    "ConsolidationAction",

    # Data classes
    "MemoryQualityMetrics",
    "ConflictReport",
    "RetrievalMetrics",
    "ConsolidationRecommendation",
    "QualityReport",
    "QualityConfig",

    # Main class
    "MemoryQualityTracker",

    # Factory functions
    "get_quality_tracker",

    # Convenience functions
    "analyze_memory_quality",
    "get_stale_memories",
    "detect_memory_conflicts",
    "get_memory_quality_report",
]
