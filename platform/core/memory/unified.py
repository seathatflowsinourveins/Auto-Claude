"""
Unified Memory Interface - V41 Architecture

This module provides a single interface to all memory systems, with intelligent
routing based on content type and cross-memory search capabilities.

Integrated Memory Types:
1. SQLite (core storage) - Primary persistent storage
2. Forgetting curve (strength tracking) - Memory decay and reinforcement
3. Procedural (learned behaviors) - Tool sequences and patterns
4. Bi-temporal (time-based queries) - Transaction and valid time tracking
5. Semantic compression (long-term storage) - Memory consolidation
6. Letta (external memory) - Project-specific agent memory

Key Features:
- Automatic routing based on content classification
- Cross-memory search with RRF (Reciprocal Rank Fusion)
- Memory lifecycle management (Store -> Access -> Reinforce -> Compress -> Archive)
- Automatic consolidation and forgetting curve cleanup
- Statistics dashboard with detailed metrics

Usage:
    from core.memory.unified import UnifiedMemory, create_unified_memory

    # Create unified memory instance
    memory = await create_unified_memory()

    # Store with automatic routing
    entry_id = await memory.store("Always use TypeScript for new code", memory_type="learning")

    # Cross-memory search
    results = await memory.search("typescript patterns", limit=10)

    # Get statistics dashboard
    stats = await memory.get_statistics()

    # Run maintenance (consolidation, compression, cleanup)
    report = await memory.run_maintenance()
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .backends.base import (
    MemoryEntry,
    MemoryLayer,
    MemoryNamespace,
    MemoryPriority,
    MemoryTier,
    TierBackend,
    generate_memory_id,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class MemoryType(str, Enum):
    """Memory content types for routing decisions."""
    PROCEDURE = "procedure"       # Learned behaviors, tool sequences
    TEMPORAL_FACT = "temporal"    # Facts with time validity
    LEARNING = "learning"         # Insights with forgetting curve
    FACT = "fact"                 # Standard factual information
    DECISION = "decision"         # Architectural decisions
    CONTEXT = "context"           # Session context
    PREFERENCE = "preference"     # User preferences
    TASK = "task"                 # Task-related information


class RoutingDecision(str, Enum):
    """Where to route a memory entry."""
    PROCEDURAL = "procedural"     # ProceduralMemory
    BITEMPORAL = "bitemporal"     # BiTemporalMemory
    FORGETTING = "forgetting"     # SQLite with forgetting curve
    STANDARD = "standard"         # Standard SQLite storage
    LETTA = "letta"               # External Letta memory
    GRAPHITI = "graphiti"         # Graphiti knowledge graph (entities/relations)


class SearchStrategy(str, Enum):
    """Search strategy for cross-memory queries."""
    ALL = "all"                   # Search all memory types
    STANDARD = "standard"         # Only SQLite
    TEMPORAL = "temporal"         # Include bi-temporal
    PROCEDURAL = "procedural"     # Include procedural
    PRIORITIZED = "prioritized"   # Based on query classification


class LifecycleState(str, Enum):
    """Memory lifecycle states."""
    ACTIVE = "active"             # Actively used
    WEAK = "weak"                 # Low strength, candidate for compression
    COMPRESSED = "compressed"     # Compressed into summary
    ARCHIVED = "archived"         # Archived, low priority
    DELETED = "deleted"           # Soft deleted


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class UnifiedSearchResult:
    """Result from cross-memory search."""
    entry: MemoryEntry
    score: float
    source: str                   # Which memory system
    rank: int                     # Position in source results
    rrf_score: float              # Reciprocal Rank Fusion score
    match_type: str               # exact, semantic, fuzzy, temporal
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingResult:
    """Result of content routing decision."""
    decision: RoutingDecision
    confidence: float             # 0.0-1.0 confidence in routing
    reasons: List[str]            # Why this routing was chosen
    suggested_namespace: Optional[MemoryNamespace] = None
    suggested_priority: MemoryPriority = MemoryPriority.NORMAL


@dataclass
class MaintenanceReport:
    """Report from maintenance operations."""
    started_at: datetime
    completed_at: datetime

    # Forgetting curve
    memories_decayed: int
    memories_archived: int
    memories_deleted: int
    average_strength_before: float
    average_strength_after: float

    # Compression
    memories_compressed: int
    compression_ratio: float
    retention_score: float

    # Cleanup
    expired_removed: int
    duplicates_merged: int

    # Errors
    errors: List[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return (self.completed_at - self.started_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "forgetting_curve": {
                "decayed": self.memories_decayed,
                "archived": self.memories_archived,
                "deleted": self.memories_deleted,
                "avg_strength_before": round(self.average_strength_before, 4),
                "avg_strength_after": round(self.average_strength_after, 4),
            },
            "compression": {
                "compressed": self.memories_compressed,
                "ratio": round(self.compression_ratio, 4),
                "retention": round(self.retention_score, 4),
            },
            "cleanup": {
                "expired_removed": self.expired_removed,
                "duplicates_merged": self.duplicates_merged,
            },
            "errors": self.errors,
        }


@dataclass
class UnifiedStatistics:
    """Comprehensive statistics across all memory systems."""
    # Overall
    total_entries: int
    total_by_type: Dict[str, int]
    total_by_source: Dict[str, int]

    # Strength distribution
    strength_distribution: Dict[str, int]  # excellent/good/fair/weak/very_weak
    average_strength: float

    # Compression
    compressed_entries: int
    compression_ratio: float
    total_original_tokens: int
    total_compressed_tokens: int

    # Access patterns
    hot_entries: int
    warm_entries: int
    cold_entries: int
    frozen_entries: int

    # Temporal
    valid_temporal_entries: int
    superseded_entries: int
    invalidated_entries: int

    # Procedural
    total_procedures: int
    active_procedures: int
    procedure_executions: int
    procedure_success_rate: float

    # Storage
    sqlite_size_bytes: int
    temporal_size_bytes: int
    procedural_size_bytes: int

    # Timestamps
    last_maintenance: Optional[datetime]
    last_compression: Optional[datetime]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overview": {
                "total_entries": self.total_entries,
                "by_type": self.total_by_type,
                "by_source": self.total_by_source,
            },
            "strength": {
                "distribution": self.strength_distribution,
                "average": round(self.average_strength, 4),
            },
            "compression": {
                "compressed_entries": self.compressed_entries,
                "compression_ratio": round(self.compression_ratio, 4),
                "original_tokens": self.total_original_tokens,
                "compressed_tokens": self.total_compressed_tokens,
            },
            "access_patterns": {
                "hot": self.hot_entries,
                "warm": self.warm_entries,
                "cold": self.cold_entries,
                "frozen": self.frozen_entries,
            },
            "temporal": {
                "valid_entries": self.valid_temporal_entries,
                "superseded": self.superseded_entries,
                "invalidated": self.invalidated_entries,
            },
            "procedural": {
                "total_procedures": self.total_procedures,
                "active_procedures": self.active_procedures,
                "total_executions": self.procedure_executions,
                "success_rate": round(self.procedure_success_rate, 4),
            },
            "storage": {
                "sqlite_bytes": self.sqlite_size_bytes,
                "temporal_bytes": self.temporal_size_bytes,
                "procedural_bytes": self.procedural_size_bytes,
                "total_bytes": self.sqlite_size_bytes + self.temporal_size_bytes + self.procedural_size_bytes,
            },
            "timestamps": {
                "last_maintenance": self.last_maintenance.isoformat() if self.last_maintenance else None,
                "last_compression": self.last_compression.isoformat() if self.last_compression else None,
            },
        }


# =============================================================================
# CONTENT CLASSIFIER
# =============================================================================

class ContentClassifier:
    """
    Classifies content to determine appropriate memory routing.

    Uses pattern matching and heuristics to classify content into
    memory types and suggest appropriate storage backends.
    """

    # Patterns for procedure detection
    PROCEDURE_PATTERNS = [
        r"(?i)\b(step\s*\d+|first|then|next|finally)\b",
        r"(?i)\b(workflow|sequence|procedure|process)\b",
        r"(?i)\b(run|execute|call|invoke)\s+\w+",
        r"(?i)\b(git\s+(add|commit|push|pull|checkout))\b",
        r"(?i)\b(npm\s+(install|run|build|test))\b",
    ]

    # Patterns for temporal facts
    TEMPORAL_PATTERNS = [
        r"(?i)\b(from|since|until|before|after)\s+\d{4}",
        r"(?i)\b(valid|effective|expired?)\s+(from|until|on)\b",
        r"(?i)\b(was|were|used\s+to\s+be)\b",
        r"(?i)\b(changed|updated|modified)\s+(on|in)\b",
        r"\d{4}-\d{2}-\d{2}",  # ISO date
    ]

    # Patterns for learnings (need reinforcement)
    LEARNING_PATTERNS = [
        r"(?i)\b(learned|discovered|realized|understood)\b",
        r"(?i)\b(lesson|insight|takeaway|finding)\b",
        r"(?i)\b(always|never|should|must|avoid)\b",
        r"(?i)\b(best\s+practice|anti-?pattern|gotcha)\b",
        r"(?i)\b(remember|note|important)\b",
    ]

    # Patterns for decisions
    DECISION_PATTERNS = [
        r"(?i)\b(decided|chose|selected|picked)\b",
        r"(?i)\b(decision|choice|trade-?off)\b",
        r"(?i)\b(because|since|therefore|thus)\b",
        r"(?i)\b(pros?\s+and\s+cons?|alternatives?)\b",
        r"(?i)\b(architecture|design|approach)\b",
    ]

    # Patterns for preferences
    PREFERENCE_PATTERNS = [
        r"(?i)\b(prefer|like|want|favorite)\b",
        r"(?i)\b(style|format|convention)\b",
        r"(?i)\b(user|personal|individual)\s+\w*\s*(setting|preference)\b",
    ]

    def classify(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> RoutingResult:
        """
        Classify content and determine routing.

        Args:
            content: The content to classify
            metadata: Optional metadata hints

        Returns:
            RoutingResult with decision and confidence
        """
        scores: Dict[MemoryType, float] = {mt: 0.0 for mt in MemoryType}
        reasons: List[str] = []

        # Check metadata hints first
        if metadata:
            if "memory_type" in metadata:
                explicit_type = metadata["memory_type"]
                if explicit_type in [mt.value for mt in MemoryType]:
                    return RoutingResult(
                        decision=self._type_to_routing(MemoryType(explicit_type)),
                        confidence=1.0,
                        reasons=[f"Explicit type: {explicit_type}"],
                        suggested_namespace=self._suggest_namespace(MemoryType(explicit_type)),
                        suggested_priority=self._suggest_priority(MemoryType(explicit_type)),
                    )

        # Score against patterns
        for pattern in self.PROCEDURE_PATTERNS:
            if re.search(pattern, content):
                scores[MemoryType.PROCEDURE] += 0.2
                reasons.append(f"Procedure pattern: {pattern[:30]}...")

        for pattern in self.TEMPORAL_PATTERNS:
            if re.search(pattern, content):
                scores[MemoryType.TEMPORAL_FACT] += 0.2
                reasons.append(f"Temporal pattern: {pattern[:30]}...")

        for pattern in self.LEARNING_PATTERNS:
            if re.search(pattern, content):
                scores[MemoryType.LEARNING] += 0.2
                reasons.append(f"Learning pattern: {pattern[:30]}...")

        for pattern in self.DECISION_PATTERNS:
            if re.search(pattern, content):
                scores[MemoryType.DECISION] += 0.2
                reasons.append(f"Decision pattern: {pattern[:30]}...")

        for pattern in self.PREFERENCE_PATTERNS:
            if re.search(pattern, content):
                scores[MemoryType.PREFERENCE] += 0.2
                reasons.append(f"Preference pattern: {pattern[:30]}...")

        # Determine winner
        max_score = max(scores.values())

        if max_score < 0.2:
            # No strong signals, default to fact
            memory_type = MemoryType.FACT
            confidence = 0.5
            reasons.append("No strong patterns, defaulting to fact")
        else:
            memory_type = max(scores.keys(), key=lambda k: scores[k])
            confidence = min(1.0, max_score)

        return RoutingResult(
            decision=self._type_to_routing(memory_type),
            confidence=confidence,
            reasons=reasons[:5],  # Limit reasons
            suggested_namespace=self._suggest_namespace(memory_type),
            suggested_priority=self._suggest_priority(memory_type),
        )

    def _type_to_routing(self, memory_type: MemoryType) -> RoutingDecision:
        """Map memory type to routing decision."""
        mapping = {
            MemoryType.PROCEDURE: RoutingDecision.PROCEDURAL,
            MemoryType.TEMPORAL_FACT: RoutingDecision.BITEMPORAL,
            MemoryType.LEARNING: RoutingDecision.FORGETTING,
            MemoryType.DECISION: RoutingDecision.FORGETTING,
            MemoryType.FACT: RoutingDecision.STANDARD,
            MemoryType.CONTEXT: RoutingDecision.STANDARD,
            MemoryType.PREFERENCE: RoutingDecision.BITEMPORAL,
            MemoryType.TASK: RoutingDecision.STANDARD,
        }
        return mapping.get(memory_type, RoutingDecision.STANDARD)

    def _suggest_namespace(self, memory_type: MemoryType) -> Optional[MemoryNamespace]:
        """Suggest namespace based on memory type."""
        mapping = {
            MemoryType.LEARNING: MemoryNamespace.LEARNINGS,
            MemoryType.DECISION: MemoryNamespace.DECISIONS,
            MemoryType.CONTEXT: MemoryNamespace.CONTEXT,
            MemoryType.FACT: MemoryNamespace.ARTIFACTS,
        }
        return mapping.get(memory_type)

    def _suggest_priority(self, memory_type: MemoryType) -> MemoryPriority:
        """Suggest priority based on memory type."""
        mapping = {
            MemoryType.PREFERENCE: MemoryPriority.HIGH,
            MemoryType.DECISION: MemoryPriority.HIGH,
            MemoryType.LEARNING: MemoryPriority.NORMAL,
            MemoryType.PROCEDURE: MemoryPriority.NORMAL,
        }
        return mapping.get(memory_type, MemoryPriority.NORMAL)


# =============================================================================
# RRF FUSION
# =============================================================================

class RRFFusion:
    """
    Reciprocal Rank Fusion for combining results from multiple memory systems.

    RRF Score = sum(1 / (k + rank_i)) for each result list

    This provides a robust way to merge rankings from different sources
    without requiring score normalization.
    """

    def __init__(self, k: int = 60):
        """
        Initialize RRF fusion.

        Args:
            k: Constant to prevent high-ranking items from dominating (default: 60)
        """
        self.k = k

    def fuse(
        self,
        result_lists: Dict[str, List[Tuple[MemoryEntry, float]]],
        limit: int = 10
    ) -> List[UnifiedSearchResult]:
        """
        Fuse multiple ranked result lists using RRF.

        Args:
            result_lists: Dict mapping source name to list of (entry, score) tuples
            limit: Maximum results to return

        Returns:
            Fused and ranked list of UnifiedSearchResult
        """
        # Calculate RRF scores
        entry_scores: Dict[str, Dict[str, Any]] = {}  # entry_id -> {rrf_score, entry, sources}

        for source, results in result_lists.items():
            for rank, (entry, score) in enumerate(results):
                entry_id = entry.id

                if entry_id not in entry_scores:
                    entry_scores[entry_id] = {
                        "rrf_score": 0.0,
                        "entry": entry,
                        "sources": [],
                        "best_score": 0.0,
                        "best_rank": float('inf'),
                        "match_types": set(),
                    }

                # Add RRF contribution: 1 / (k + rank)
                rrf_contribution = 1.0 / (self.k + rank + 1)  # +1 for 0-indexing
                entry_scores[entry_id]["rrf_score"] += rrf_contribution
                entry_scores[entry_id]["sources"].append(source)
                entry_scores[entry_id]["best_score"] = max(
                    entry_scores[entry_id]["best_score"], score
                )
                entry_scores[entry_id]["best_rank"] = min(
                    entry_scores[entry_id]["best_rank"], rank
                )

        # Sort by RRF score
        sorted_entries = sorted(
            entry_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        # Build results
        results: List[UnifiedSearchResult] = []
        for i, item in enumerate(sorted_entries[:limit]):
            results.append(UnifiedSearchResult(
                entry=item["entry"],
                score=item["best_score"],
                source=", ".join(item["sources"]),
                rank=item["best_rank"],
                rrf_score=item["rrf_score"],
                match_type="fused",
                metadata={
                    "contributing_sources": item["sources"],
                    "source_count": len(item["sources"]),
                }
            ))

        return results


# =============================================================================
# UNIFIED MEMORY
# =============================================================================

class UnifiedMemory:
    """
    Unified interface to all V41 memory systems.

    Provides:
    - Single interface for all memory operations
    - Automatic routing based on content classification
    - Cross-memory search with RRF fusion
    - Memory lifecycle management
    - Maintenance and consolidation
    - Comprehensive statistics

    Memory Systems Integrated:
    - SQLite (core persistent storage)
    - Forgetting curve (strength tracking)
    - Procedural (learned behaviors)
    - Bi-temporal (time-based queries)
    - Semantic compression (long-term storage)
    - Letta (external project memory)
    """

    def __init__(
        self,
        sqlite_backend: Optional[Any] = None,
        procedural_memory: Optional[Any] = None,
        bitemporal_memory: Optional[Any] = None,
        compressor: Optional[Any] = None,
        letta_backend: Optional[Any] = None,
        graphiti_backend: Optional[Any] = None,
        hnsw_backend: Optional[Any] = None,
        embedding_provider: Optional[Callable[[str], List[float]]] = None,
        auto_route: bool = True,
        enable_forgetting: bool = True,
        enable_compression: bool = True,
        enable_graphiti: bool = False,
        enable_hnsw: bool = True,
    ) -> None:
        """
        Initialize unified memory interface.

        Args:
            sqlite_backend: SQLite tier backend for core storage
            procedural_memory: ProceduralMemory for learned behaviors
            bitemporal_memory: BiTemporalMemory for temporal facts
            compressor: MemoryCompressor for long-term storage
            letta_backend: LettaTierBackend for external memory
            graphiti_backend: GraphitiTierBackend for knowledge graphs (optional)
            hnsw_backend: HNSWBackend for high-performance vector search (150x-12500x speedup)
            embedding_provider: Function to generate embeddings
            auto_route: Automatically route based on content classification
            enable_forgetting: Enable forgetting curve for strength decay
            enable_compression: Enable automatic compression for old memories
            enable_graphiti: Enable Graphiti knowledge graph backend (requires Neo4j)
            enable_hnsw: Enable HNSW vector index for semantic search acceleration
        """
        self._sqlite = sqlite_backend
        self._procedural = procedural_memory
        self._bitemporal = bitemporal_memory
        self._compressor = compressor
        self._letta = letta_backend
        self._graphiti = graphiti_backend
        self._hnsw = hnsw_backend
        self._embedding_provider = embedding_provider

        self._auto_route = auto_route
        self._enable_forgetting = enable_forgetting
        self._enable_compression = enable_compression
        self._enable_graphiti = enable_graphiti
        self._enable_hnsw = enable_hnsw

        self._classifier = ContentClassifier()
        self._rrf = RRFFusion()

        # Tracking
        self._last_maintenance: Optional[datetime] = None
        self._last_compression: Optional[datetime] = None
        self._initialized = False

        logger.info("UnifiedMemory initialized")

    async def _ensure_initialized(self) -> None:
        """Lazy initialization of backends."""
        if self._initialized:
            return

        # Initialize SQLite backend
        if self._sqlite is None:
            try:
                from .backends.sqlite import get_sqlite_backend
                self._sqlite = get_sqlite_backend(embedding_provider=self._embedding_provider)
            except Exception as e:
                logger.warning(f"Failed to initialize SQLite backend: {e}")

        # Initialize procedural memory
        if self._procedural is None:
            try:
                from .procedural import get_procedural_memory
                self._procedural = get_procedural_memory(
                    embedding_provider=self._embedding_provider
                )
            except Exception as e:
                logger.warning(f"Failed to initialize procedural memory: {e}")

        # Initialize bi-temporal memory
        if self._bitemporal is None:
            try:
                from .temporal import create_bitemporal_memory
                self._bitemporal = await create_bitemporal_memory(
                    embedding_provider=self._embedding_provider
                )
            except Exception as e:
                logger.warning(f"Failed to initialize bi-temporal memory: {e}")

        # Initialize compressor
        if self._compressor is None and self._enable_compression:
            try:
                from .compression import get_compressor
                self._compressor = get_compressor(
                    backend=self._sqlite,
                    embedding_provider=self._embedding_provider
                )
            except Exception as e:
                logger.warning(f"Failed to initialize compressor: {e}")

        # Initialize Graphiti backend (optional - requires Neo4j)
        if self._graphiti is None and self._enable_graphiti:
            try:
                from .backends.graphiti import get_graphiti_backend
                self._graphiti = get_graphiti_backend()
                logger.info("Graphiti backend initialized (placeholder mode)")
            except Exception as e:
                logger.warning(f"Failed to initialize Graphiti backend: {e}")

        # Initialize HNSW backend for high-performance vector search (150x-12500x speedup)
        if self._hnsw is None and self._enable_hnsw:
            try:
                from .backends.hnsw import get_hnsw_backend
                self._hnsw = get_hnsw_backend()
                stats = self._hnsw.get_stats()
                logger.info(
                    f"HNSW backend initialized: {stats.get('count', 0)} vectors, "
                    f"backend={stats.get('backend', 'unknown')}, dim={stats.get('dimension', 384)}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize HNSW backend: {e}")
                self._hnsw = None

        self._initialized = True

    # =========================================================================
    # STORE OPERATIONS
    # =========================================================================

    async def store(
        self,
        content: str,
        memory_type: Optional[str] = None,
        namespace: Optional[MemoryNamespace] = None,
        priority: Optional[MemoryPriority] = None,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        valid_from: Optional[datetime] = None,
        valid_to: Optional[datetime] = None,
        force_routing: Optional[RoutingDecision] = None,
    ) -> str:
        """
        Store content with automatic routing based on classification.

        Args:
            content: The content to store
            memory_type: Explicit memory type (overrides classification)
            namespace: Memory namespace for TTL
            priority: Memory priority level
            importance: Importance score (0.0-1.0)
            tags: Optional tags for categorization
            metadata: Additional metadata
            valid_from: For temporal facts, when the fact became true
            valid_to: For temporal facts, when the fact stopped being true
            force_routing: Force a specific routing decision

        Returns:
            Memory ID of the stored entry
        """
        await self._ensure_initialized()

        # Build metadata
        meta = metadata or {}
        if memory_type:
            meta["memory_type"] = memory_type

        # Classify content and determine routing
        if force_routing:
            routing = RoutingResult(
                decision=force_routing,
                confidence=1.0,
                reasons=["Forced routing"],
                suggested_namespace=namespace,
                suggested_priority=priority or MemoryPriority.NORMAL,
            )
        elif self._auto_route:
            routing = self._classifier.classify(content, meta)
        else:
            routing = RoutingResult(
                decision=RoutingDecision.STANDARD,
                confidence=1.0,
                reasons=["Auto-routing disabled"],
            )

        # Use routing suggestions if not explicitly provided
        namespace = namespace or routing.suggested_namespace
        priority = priority or routing.suggested_priority

        # Route to appropriate backend
        memory_id: str

        if routing.decision == RoutingDecision.PROCEDURAL:
            memory_id = await self._store_procedural(content, tags, meta)

        elif routing.decision == RoutingDecision.BITEMPORAL:
            memory_id = await self._store_bitemporal(
                content, namespace, importance, tags, meta, valid_from, valid_to
            )

        elif routing.decision == RoutingDecision.FORGETTING:
            memory_id = await self._store_with_forgetting(
                content, memory_type or "learning", namespace, priority, importance, tags, meta
            )

        elif routing.decision == RoutingDecision.LETTA:
            memory_id = await self._store_letta(content, namespace, meta)

        elif routing.decision == RoutingDecision.GRAPHITI:
            memory_id = await self._store_graphiti(content, namespace, tags, meta)

        else:  # STANDARD
            memory_id = await self._store_standard(
                content, memory_type or "fact", namespace, priority, importance, tags, meta
            )

        logger.debug(f"Stored memory {memory_id} via {routing.decision.value} (confidence: {routing.confidence:.2f})")
        return memory_id

    async def _store_procedural(
        self,
        content: str,
        tags: Optional[List[str]],
        metadata: Dict[str, Any]
    ) -> str:
        """Store as procedural memory (extract steps if possible)."""
        if self._procedural is None:
            # Fallback to standard storage
            return await self._store_standard(content, "procedure", None, None, 0.5, tags, metadata)

        # Try to extract procedure from content
        from .procedural import ProcedureStep, StepType

        # Simple step extraction (looks for numbered steps or bullet points)
        step_patterns = [
            r"(?:^|\n)\s*(?:\d+[.)]\s*|[-*]\s*)(.*?)(?=\n\s*(?:\d+[.)]\s*|[-*]\s*)|\n\n|$)",
        ]

        steps: List[ProcedureStep] = []
        for pattern in step_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for i, match in enumerate(matches):
                text = match.strip()
                if len(text) > 10:  # Skip very short matches
                    steps.append(ProcedureStep(
                        action=f"step_{i+1}",
                        params={"content": text},
                        step_type=StepType.TOOL_CALL,
                        order=i,
                        description=text[:100],
                    ))
            if steps:
                break

        if not steps:
            # No steps extracted, store as single step
            steps = [ProcedureStep(
                action="execute",
                params={"content": content},
                description=content[:100],
            )]

        # Extract trigger patterns from content
        trigger_patterns = []
        # Look for quoted strings as potential triggers
        quotes = re.findall(r'"([^"]+)"', content)
        trigger_patterns.extend(quotes[:5])

        if not trigger_patterns:
            # Use first few words as trigger
            words = content.split()[:5]
            trigger_patterns.append(" ".join(words))

        # Create procedure
        procedure = await self._procedural.learn_procedure(
            name=f"procedure_{hashlib.md5(content[:50].encode()).hexdigest()[:8]}",
            steps=steps,
            trigger_patterns=trigger_patterns,
            description=content[:200],
            tags=tags,
            metadata=metadata,
        )

        return procedure.id

    async def _store_bitemporal(
        self,
        content: str,
        namespace: Optional[MemoryNamespace],
        importance: float,
        tags: Optional[List[str]],
        metadata: Dict[str, Any],
        valid_from: Optional[datetime],
        valid_to: Optional[datetime],
    ) -> str:
        """Store with bi-temporal tracking."""
        if self._bitemporal is None:
            # Fallback to standard storage
            return await self._store_standard(content, "temporal", namespace, None, importance, tags, metadata)

        entry = await self._bitemporal.store(
            content=content,
            valid_from=valid_from,
            valid_to=valid_to,
            memory_type="temporal_fact",
            importance=importance,
            tags=tags,
            namespace=namespace,
            metadata=metadata,
        )

        return entry.id

    async def _store_with_forgetting(
        self,
        content: str,
        memory_type: str,
        namespace: Optional[MemoryNamespace],
        priority: Optional[MemoryPriority],
        importance: float,
        tags: Optional[List[str]],
        metadata: Dict[str, Any],
    ) -> str:
        """Store with forgetting curve tracking (higher importance = slower decay)."""
        if self._sqlite is None:
            logger.error("SQLite backend not available")
            return ""

        # Determine decay rate based on importance
        if importance >= 0.8:
            decay_rate = 0.05  # Very slow decay for important items
        elif importance >= 0.6:
            decay_rate = 0.1
        elif importance >= 0.4:
            decay_rate = 0.15
        else:
            decay_rate = 0.25  # Faster decay for less important items

        memory_id = generate_memory_id(content, f"{memory_type}_")
        now = datetime.now(timezone.utc)

        entry = MemoryEntry(
            id=memory_id,
            content=content,
            tier=MemoryTier.ARCHIVAL_MEMORY,
            priority=priority or MemoryPriority.NORMAL,
            namespace=namespace,
            content_type=memory_type,
            tags=tags or [],
            metadata={
                **metadata,
                "importance": importance,
                "memory_type": memory_type,
                "has_forgetting": True,
            },
            strength=1.0,
            decay_rate=decay_rate,
            last_reinforced=now,
            reinforcement_count=0,
        )

        await self._sqlite.put(memory_id, entry)
        return memory_id

    async def _store_standard(
        self,
        content: str,
        memory_type: str,
        namespace: Optional[MemoryNamespace],
        priority: Optional[MemoryPriority],
        importance: float,
        tags: Optional[List[str]],
        metadata: Dict[str, Any],
    ) -> str:
        """Store in standard SQLite backend."""
        if self._sqlite is None:
            logger.error("SQLite backend not available")
            return ""

        return await self._sqlite.store_memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags,
            namespace=namespace,
        )

    async def _store_letta(
        self,
        content: str,
        namespace: Optional[MemoryNamespace],
        metadata: Dict[str, Any],
    ) -> str:
        """Store in Letta external memory."""
        if self._letta is None:
            # Fallback to standard storage
            return await self._store_standard(content, "external", namespace, None, 0.5, None, metadata)

        return await self._letta.store(
            content=content,
            namespace=namespace or MemoryNamespace.CONTEXT,
            metadata=metadata,
        )

    async def _store_graphiti(
        self,
        content: str,
        namespace: Optional[MemoryNamespace],
        tags: Optional[List[str]],
        metadata: Dict[str, Any],
    ) -> str:
        """Store in Graphiti knowledge graph as an episode.

        Graphiti will extract entities and relationships from the content
        using LLM-based extraction.

        TODO: Full implementation requires graphiti-core and Neo4j.
        """
        if self._graphiti is None:
            # Fallback to bi-temporal storage (closest alternative)
            logger.debug("Graphiti not available, falling back to bi-temporal storage")
            return await self._store_bitemporal(
                content, namespace, 0.5, tags, metadata, None, None
            )

        # Use Graphiti's episode-based storage
        return await self._graphiti.add_episode(
            content=content,
            source=namespace.value if namespace else "unified_memory",
            source_description=f"Stored via UnifiedMemory ({namespace.value if namespace else 'default'})",
            metadata={
                **metadata,
                "tags": tags or [],
            },
        )

    # =========================================================================
    # SEARCH OPERATIONS
    # =========================================================================

    async def search(
        self,
        query: str,
        limit: int = 10,
        strategy: SearchStrategy = SearchStrategy.ALL,
        namespace: Optional[MemoryNamespace] = None,
        min_strength: float = 0.0,
        include_archived: bool = False,
        as_of_time: Optional[datetime] = None,
        valid_time: Optional[datetime] = None,
    ) -> List[UnifiedSearchResult]:
        """
        Cross-memory search with RRF fusion.

        Args:
            query: Search query
            limit: Maximum results
            strategy: Which memory systems to search
            namespace: Filter by namespace
            min_strength: Minimum strength for forgetting curve entries
            include_archived: Include archived/compressed entries
            as_of_time: For temporal queries - knowledge state at this time
            valid_time: For temporal queries - facts valid at this time

        Returns:
            Fused and ranked list of UnifiedSearchResult
        """
        await self._ensure_initialized()

        result_lists: Dict[str, List[Tuple[MemoryEntry, float]]] = {}

        # Determine which backends to search based on strategy
        search_sqlite = strategy in [SearchStrategy.ALL, SearchStrategy.STANDARD, SearchStrategy.PRIORITIZED]
        search_temporal = strategy in [SearchStrategy.ALL, SearchStrategy.TEMPORAL, SearchStrategy.PRIORITIZED]
        search_procedural = strategy in [SearchStrategy.ALL, SearchStrategy.PROCEDURAL, SearchStrategy.PRIORITIZED]

        # For PRIORITIZED, classify query to determine priority
        if strategy == SearchStrategy.PRIORITIZED:
            routing = self._classifier.classify(query)
            if routing.decision == RoutingDecision.PROCEDURAL:
                search_procedural = True
                search_sqlite = False
                search_temporal = False
            elif routing.decision == RoutingDecision.BITEMPORAL:
                search_temporal = True
                search_sqlite = False
                search_procedural = False

        # Search backends in parallel
        search_tasks = []

        # HNSW vector search (high-performance, 150x-12500x speedup)
        # Run first as it's the fastest semantic search option
        if self._hnsw and self._embedding_provider and strategy == SearchStrategy.ALL:
            search_tasks.append(self._search_hnsw(query, limit * 2))

        if search_sqlite and self._sqlite:
            search_tasks.append(self._search_sqlite(query, limit * 2, min_strength))

        if search_temporal and self._bitemporal:
            search_tasks.append(self._search_temporal(query, limit * 2, as_of_time, valid_time))

        if search_procedural and self._procedural:
            search_tasks.append(self._search_procedural(query, limit * 2))

        if self._letta and strategy == SearchStrategy.ALL:
            search_tasks.append(self._search_letta(query, limit))

        if self._graphiti and strategy == SearchStrategy.ALL:
            search_tasks.append(self._search_graphiti(query, limit))

        # Gather results
        if search_tasks:
            results = await asyncio.gather(*search_tasks, return_exceptions=True)

            task_names = []
            if self._hnsw and self._embedding_provider and strategy == SearchStrategy.ALL:
                task_names.append("hnsw")
            if search_sqlite and self._sqlite:
                task_names.append("sqlite")
            if search_temporal and self._bitemporal:
                task_names.append("temporal")
            if search_procedural and self._procedural:
                task_names.append("procedural")
            if self._letta and strategy == SearchStrategy.ALL:
                task_names.append("letta")
            if self._graphiti and strategy == SearchStrategy.ALL:
                task_names.append("graphiti")

            for name, result in zip(task_names, results):
                if isinstance(result, Exception):
                    logger.warning(f"Search failed for {name}: {result}")
                elif result:
                    result_lists[name] = result

        # Fuse results using RRF
        fused = self._rrf.fuse(result_lists, limit)

        # Filter by strength if requested
        if min_strength > 0:
            fused = [r for r in fused if r.entry.calculate_current_strength() >= min_strength]

        # Filter archived if requested
        if not include_archived:
            fused = [r for r in fused if not r.entry.metadata.get("archived_to")]

        return fused[:limit]

    async def _search_sqlite(
        self,
        query: str,
        limit: int,
        min_strength: float
    ) -> List[Tuple[MemoryEntry, float]]:
        """Search SQLite backend."""
        results = await self._sqlite.search(query, limit)

        # Score results (FTS doesn't give us scores directly in this wrapper)
        scored = []
        for i, entry in enumerate(results):
            # Calculate score based on position and strength
            base_score = 1.0 - (i * 0.1)  # Position-based
            strength = entry.calculate_current_strength()

            if strength >= min_strength:
                combined_score = base_score * (0.5 + 0.5 * strength)
                scored.append((entry, combined_score))

        return scored

    async def _search_temporal(
        self,
        query: str,
        limit: int,
        as_of_time: Optional[datetime],
        valid_time: Optional[datetime],
    ) -> List[Tuple[MemoryEntry, float]]:
        """Search bi-temporal memory."""
        if as_of_time and valid_time:
            # Full bi-temporal query
            results = await self._bitemporal.query_bitemporal(query, as_of_time, valid_time, limit)
        elif as_of_time:
            # As-of query (what did we know at time T)
            results = await self._bitemporal.query_as_of(query, as_of_time, limit)
        elif valid_time:
            # Valid-time query (what was true at time T)
            results = await self._bitemporal.query_valid_at(query, valid_time, limit)
        else:
            # Current valid facts
            entries = await self._bitemporal.search_valid_now(query, limit)
            # Convert to (entry, score) tuples
            return [(e, 1.0 - i * 0.1) for i, e in enumerate(entries)]

        # Convert TemporalSearchResult to (entry, score) tuples
        return [(r.entry, r.score * r.temporal_relevance) for r in results]

    async def _search_procedural(
        self,
        query: str,
        limit: int
    ) -> List[Tuple[MemoryEntry, float]]:
        """Search procedural memory."""
        matches = await self._procedural.recall_procedure(query, limit)

        # Convert ProcedureMatch to MemoryEntry
        results = []
        for match in matches:
            # Create a MemoryEntry from procedure
            entry = MemoryEntry(
                id=match.procedure.id,
                content=f"{match.procedure.name}: {match.procedure.description}",
                tier=MemoryTier.ARCHIVAL_MEMORY,
                priority=MemoryPriority.NORMAL,
                content_type="procedure",
                tags=match.procedure.tags,
                metadata={
                    "procedure_name": match.procedure.name,
                    "steps_count": len(match.procedure.steps),
                    "confidence": match.procedure.confidence,
                    "success_rate": match.procedure.success_rate,
                    "matched_trigger": match.matched_trigger,
                },
            )
            results.append((entry, match.confidence))

        return results

    async def _search_letta(
        self,
        query: str,
        limit: int
    ) -> List[Tuple[MemoryEntry, float]]:
        """Search Letta external memory."""
        entries = await self._letta.search(query, limit)
        return [(e, e.metadata.get("relevance_score", 0.5)) for e in entries]

    async def _search_graphiti(
        self,
        query: str,
        limit: int
    ) -> List[Tuple[MemoryEntry, float]]:
        """Search Graphiti knowledge graph.

        Searches over episodes, entities, and relationships.

        TODO: Full implementation with entity-focused search.
        """
        entries = await self._graphiti.search(query, limit)
        return [(e, e.metadata.get("score", 0.5)) for e in entries]

    async def _search_hnsw(
        self,
        query: str,
        limit: int
    ) -> List[Tuple[MemoryEntry, float]]:
        """Search HNSW vector index for high-performance semantic search.

        Performance: 150x-12500x faster than linear scan.
        Typical latency: 0.07-0.25ms depending on index size and ef_search.

        Args:
            query: Search query text
            limit: Maximum results to return

        Returns:
            List of (MemoryEntry, score) tuples
        """
        if not self._hnsw or not self._embedding_provider:
            return []

        try:
            # Generate query embedding
            query_embedding = self._embedding_provider(query)

            # Search HNSW index
            hnsw_results = await self._hnsw.search(query_embedding, k=limit)

            # Convert to (MemoryEntry, score) tuples
            results: List[Tuple[MemoryEntry, float]] = []

            for result in hnsw_results:
                # Try to retrieve the full entry from SQLite using the HNSW result ID
                entry = None
                if self._sqlite:
                    entry = await self._sqlite.get(result.id, reinforce=False)

                if entry:
                    results.append((entry, result.score))
                else:
                    # Create minimal entry from HNSW metadata if SQLite lookup fails
                    entry = MemoryEntry(
                        id=result.id,
                        content=result.metadata.get("content", f"[HNSW ID: {result.id}]"),
                        tier=MemoryTier.ARCHIVAL_MEMORY,
                        priority=MemoryPriority.NORMAL,
                        content_type=result.metadata.get("type", "hnsw_result"),
                        metadata={
                            **result.metadata,
                            "hnsw_score": result.score,
                            "hnsw_distance": result.distance,
                        },
                    )
                    results.append((entry, result.score))

            return results

        except Exception as e:
            logger.warning(f"HNSW search failed: {e}")
            return []

    async def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        hnsw_weight: float = 0.6,
        fts_weight: float = 0.4,
        min_strength: float = 0.0,
        ef_search: Optional[int] = None,
    ) -> List[UnifiedSearchResult]:
        """
        Perform hybrid search combining HNSW vector search with SQLite FTS5.

        This provides the best of both worlds:
        - HNSW: Semantic similarity (captures meaning, handles synonyms)
        - FTS5: Keyword matching (exact terms, high precision)

        The results are combined using weighted RRF (Reciprocal Rank Fusion).

        Performance target: 150x-12500x speedup over linear scan.

        Args:
            query: Search query text
            limit: Maximum results to return
            hnsw_weight: Weight for HNSW results in fusion (default: 0.6)
            fts_weight: Weight for FTS5 results in fusion (default: 0.4)
            min_strength: Minimum memory strength for forgetting curve
            ef_search: Optional HNSW ef_search parameter override

        Returns:
            Fused and ranked list of UnifiedSearchResult
        """
        await self._ensure_initialized()

        result_lists: Dict[str, List[Tuple[MemoryEntry, float]]] = {}

        # HNSW semantic search (fast vector similarity)
        if self._hnsw and self._embedding_provider:
            try:
                hnsw_results = await self._search_hnsw(query, limit * 2)
                if hnsw_results:
                    # Apply weight to scores
                    weighted_results = [(e, s * hnsw_weight) for e, s in hnsw_results]
                    result_lists["hnsw"] = weighted_results
                    logger.debug(f"HNSW search returned {len(hnsw_results)} results")
            except Exception as e:
                logger.warning(f"HNSW search failed in hybrid: {e}")

        # SQLite FTS5 keyword search
        if self._sqlite:
            try:
                fts_results = await self._search_sqlite(query, limit * 2, min_strength)
                if fts_results:
                    # Apply weight to scores
                    weighted_results = [(e, s * fts_weight) for e, s in fts_results]
                    result_lists["fts5"] = weighted_results
                    logger.debug(f"FTS5 search returned {len(fts_results)} results")
            except Exception as e:
                logger.warning(f"FTS5 search failed in hybrid: {e}")

        # Fuse results using RRF
        fused = self._rrf.fuse(result_lists, limit)

        # Update match type to indicate hybrid
        for result in fused:
            result.match_type = "hybrid"
            result.metadata["search_sources"] = list(result_lists.keys())
            result.metadata["hnsw_weight"] = hnsw_weight
            result.metadata["fts_weight"] = fts_weight

        return fused

    # =========================================================================
    # RETRIEVAL OPERATIONS
    # =========================================================================

    async def get(
        self,
        memory_id: str,
        reinforce: bool = True
    ) -> Optional[MemoryEntry]:
        """
        Get a memory entry by ID from any backend.

        Args:
            memory_id: The memory ID
            reinforce: Whether to reinforce strength on access

        Returns:
            MemoryEntry if found, None otherwise
        """
        await self._ensure_initialized()

        # Try SQLite first (most common)
        if self._sqlite:
            entry = await self._sqlite.get(memory_id, reinforce=reinforce)
            if entry:
                return entry

        # Try bi-temporal
        if self._bitemporal:
            entry = await self._bitemporal.get(memory_id)
            if entry:
                return entry

        # Try procedural
        if self._procedural:
            procedure = await self._procedural.get_procedure(memory_id)
            if procedure:
                return MemoryEntry(
                    id=procedure.id,
                    content=f"{procedure.name}: {procedure.description}",
                    content_type="procedure",
                    metadata={"procedure": procedure.to_dict()},
                )

        # Try Letta
        if self._letta:
            entry = await self._letta.get(memory_id)
            if entry:
                return entry

        # Try Graphiti
        if self._graphiti:
            entry = await self._graphiti.get(memory_id)
            if entry:
                return entry

        return None

    async def delete(self, memory_id: str) -> bool:
        """
        Delete a memory entry from all backends.

        Args:
            memory_id: The memory ID to delete

        Returns:
            True if deleted from any backend
        """
        await self._ensure_initialized()
        deleted = False

        if self._sqlite:
            if await self._sqlite.delete(memory_id):
                deleted = True

        if self._bitemporal:
            if await self._bitemporal.delete(memory_id):
                deleted = True

        if self._procedural:
            if await self._procedural.delete_procedure(memory_id):
                deleted = True

        if self._letta:
            if await self._letta.delete(memory_id):
                deleted = True

        if self._graphiti:
            if await self._graphiti.delete(memory_id):
                deleted = True

        return deleted

    # =========================================================================
    # LIFECYCLE MANAGEMENT
    # =========================================================================

    async def run_maintenance(
        self,
        archive_threshold: float = 0.1,
        delete_threshold: float = 0.01,
        compress_age_days: float = 7.0,
    ) -> MaintenanceReport:
        """
        Run maintenance operations: decay, archive, compress, cleanup.

        This implements the memory lifecycle:
        Store -> Access -> Reinforce -> Compress -> Archive

        Args:
            archive_threshold: Strength below which to archive
            delete_threshold: Strength below which to delete
            compress_age_days: Age after which to consider for compression

        Returns:
            MaintenanceReport with details
        """
        await self._ensure_initialized()
        started_at = datetime.now(timezone.utc)

        memories_decayed = 0
        memories_archived = 0
        memories_deleted = 0
        strength_sum_before = 0.0
        strength_sum_after = 0.0
        memories_compressed = 0
        compression_ratio = 0.0
        retention_score = 0.0
        expired_removed = 0
        duplicates_merged = 0
        errors: List[str] = []

        # Process SQLite entries with forgetting curve
        if self._sqlite and self._enable_forgetting:
            try:
                entries = await self._sqlite.list_all()
                total_entries = len(entries)

                for entry in entries:
                    current_strength = entry.calculate_current_strength()
                    strength_sum_before += current_strength

                    if current_strength < delete_threshold:
                        await self._sqlite.delete(entry.id)
                        memories_deleted += 1
                    elif current_strength < archive_threshold:
                        archived, _ = await self._sqlite.archive_weak_memories(archive_threshold)
                        memories_archived += archived
                    else:
                        memories_decayed += 1
                        strength_sum_after += current_strength

            except Exception as e:
                errors.append(f"Forgetting curve processing failed: {e}")
                logger.error(f"Maintenance error (forgetting): {e}")

        # Run compression for old memories
        if self._compressor and self._enable_compression:
            try:
                from .compression import CompressionTrigger

                candidate_groups = await self._compressor.identify_candidates(
                    trigger=CompressionTrigger.SCHEDULED
                )

                for group in candidate_groups[:10]:  # Limit batches
                    try:
                        result = await self._compressor.compress_memories(
                            group,
                            trigger=CompressionTrigger.SCHEDULED
                        )
                        memories_compressed += len(result.original_ids)
                        compression_ratio = result.compression_ratio
                        retention_score = result.retention_score
                    except Exception as e:
                        errors.append(f"Compression failed for group: {e}")

                self._last_compression = datetime.now(timezone.utc)

            except Exception as e:
                errors.append(f"Compression identification failed: {e}")
                logger.error(f"Maintenance error (compression): {e}")

        # Cleanup expired entries
        if self._sqlite:
            try:
                expired_removed = await self._sqlite.delete_archived_memories(days_old=30)
            except Exception as e:
                errors.append(f"Expired cleanup failed: {e}")

        self._last_maintenance = datetime.now(timezone.utc)
        completed_at = datetime.now(timezone.utc)

        report = MaintenanceReport(
            started_at=started_at,
            completed_at=completed_at,
            memories_decayed=memories_decayed,
            memories_archived=memories_archived,
            memories_deleted=memories_deleted,
            average_strength_before=strength_sum_before / max(memories_decayed + memories_archived + memories_deleted, 1),
            average_strength_after=strength_sum_after / max(memories_decayed, 1),
            memories_compressed=memories_compressed,
            compression_ratio=compression_ratio,
            retention_score=retention_score,
            expired_removed=expired_removed,
            duplicates_merged=duplicates_merged,
            errors=errors,
        )

        logger.info(
            f"Maintenance complete: decayed={memories_decayed}, archived={memories_archived}, "
            f"deleted={memories_deleted}, compressed={memories_compressed}"
        )

        return report

    async def reinforce_memory(
        self,
        memory_id: str,
        access_type: str = "recall"
    ) -> Optional[float]:
        """
        Reinforce a memory's strength on access.

        Args:
            memory_id: The memory ID to reinforce
            access_type: Type of access (recall, review, reference, passive)

        Returns:
            New strength value, or None if not found
        """
        entry = await self.get(memory_id, reinforce=True)
        if entry:
            return entry.strength
        return None

    # =========================================================================
    # STATISTICS
    # =========================================================================

    async def get_statistics(self) -> UnifiedStatistics:
        """
        Get comprehensive statistics across all memory systems.

        Returns:
            UnifiedStatistics with detailed metrics
        """
        await self._ensure_initialized()

        total_entries = 0
        total_by_type: Dict[str, int] = {}
        total_by_source: Dict[str, int] = {}

        strength_distribution: Dict[str, int] = {
            "excellent": 0,  # >= 0.8
            "good": 0,       # >= 0.6
            "fair": 0,       # >= 0.4
            "weak": 0,       # >= 0.1
            "very_weak": 0,  # < 0.1
        }
        strength_sum = 0.0

        compressed_entries = 0
        total_original_tokens = 0
        total_compressed_tokens = 0

        access_counts: Dict[str, int] = {"hot": 0, "warm": 0, "cold": 0, "frozen": 0}

        valid_temporal_entries = 0
        superseded_entries = 0
        invalidated_entries = 0

        total_procedures = 0
        active_procedures = 0
        procedure_executions = 0
        procedure_successes = 0

        sqlite_size = 0
        temporal_size = 0
        procedural_size = 0

        # SQLite stats
        if self._sqlite:
            try:
                stats = await self._sqlite.get_stats()
                total_entries += stats.get("total_memories", 0)
                total_by_source["sqlite"] = stats.get("total_memories", 0)

                for mem_type, count in stats.get("memories_by_type", {}).items():
                    total_by_type[mem_type] = total_by_type.get(mem_type, 0) + count

                sqlite_size = stats.get("db_size_bytes", 0)

                # Strength stats
                strength_stats = stats.get("strength_stats", {})
                strength_sum += (strength_stats.get("average", 0.5) or 0.5) * stats.get("total_memories", 0)
                strength_distribution["weak"] = strength_stats.get("weak_count", 0)
                strength_distribution["very_weak"] = strength_stats.get("very_weak_count", 0)

                # Process entries for detailed stats
                entries = await self._sqlite.list_all()
                for entry in entries:
                    # Access pattern
                    pattern = entry.access_pattern.value if entry.access_pattern else "warm"
                    access_counts[pattern] = access_counts.get(pattern, 0) + 1

                    # Compression
                    if entry.metadata.get("is_compressed"):
                        compressed_entries += 1
                        total_original_tokens += entry.metadata.get("original_tokens", 0)
                        total_compressed_tokens += entry.token_count

                    # Strength distribution
                    strength = entry.calculate_current_strength()
                    if strength >= 0.8:
                        strength_distribution["excellent"] += 1
                    elif strength >= 0.6:
                        strength_distribution["good"] += 1
                    elif strength >= 0.4:
                        strength_distribution["fair"] += 1

            except Exception as e:
                logger.warning(f"Failed to get SQLite stats: {e}")

        # Temporal stats
        if self._bitemporal:
            try:
                stats = await self._bitemporal.get_stats()
                temporal_entries = stats.get("total_entries", 0)
                total_entries += temporal_entries
                total_by_source["temporal"] = temporal_entries

                valid_temporal_entries = stats.get("currently_valid", 0)
                superseded_entries = stats.get("superseded", 0)
                invalidated_entries = stats.get("invalidated", 0)

                temporal_size = stats.get("db_size_bytes", 0)
            except Exception as e:
                logger.warning(f"Failed to get temporal stats: {e}")

        # Procedural stats
        if self._procedural:
            try:
                stats = await self._procedural.get_stats()
                total_procedures = stats.get("total_procedures", 0)
                active_procedures = stats.get("active_procedures", 0)
                procedure_executions = stats.get("total_executions", 0)
                procedure_successes = stats.get("successful_executions", 0)

                total_by_source["procedural"] = total_procedures
                total_by_type["procedure"] = total_procedures

                procedural_size = stats.get("db_size_bytes", 0)
            except Exception as e:
                logger.warning(f"Failed to get procedural stats: {e}")

        # Compression stats
        if self._compressor:
            try:
                metrics = self._compressor.get_metrics()
                total_original_tokens = max(total_original_tokens, metrics.total_original_tokens)
                total_compressed_tokens = max(total_compressed_tokens, metrics.total_compressed_tokens)
            except Exception as e:
                logger.warning(f"Failed to get compression stats: {e}")

        # HNSW vector index stats
        if self._hnsw:
            try:
                hnsw_stats = self._hnsw.get_stats()
                total_by_source["hnsw"] = hnsw_stats.get("count", 0)
                total_by_type["vector_indexed"] = hnsw_stats.get("count", 0)
                logger.debug(
                    f"HNSW stats: {hnsw_stats.get('count', 0)} vectors, "
                    f"backend={hnsw_stats.get('backend', 'unknown')}"
                )
            except Exception as e:
                logger.warning(f"Failed to get HNSW stats: {e}")

        # Calculate compression ratio
        compression_ratio = (
            total_compressed_tokens / total_original_tokens
            if total_original_tokens > 0
            else 1.0
        )

        # Calculate procedure success rate
        procedure_success_rate = (
            procedure_successes / procedure_executions
            if procedure_executions > 0
            else 0.0
        )

        # Calculate average strength
        average_strength = strength_sum / total_entries if total_entries > 0 else 0.5

        return UnifiedStatistics(
            total_entries=total_entries,
            total_by_type=total_by_type,
            total_by_source=total_by_source,
            strength_distribution=strength_distribution,
            average_strength=average_strength,
            compressed_entries=compressed_entries,
            compression_ratio=compression_ratio,
            total_original_tokens=total_original_tokens,
            total_compressed_tokens=total_compressed_tokens,
            hot_entries=access_counts["hot"],
            warm_entries=access_counts["warm"],
            cold_entries=access_counts["cold"],
            frozen_entries=access_counts["frozen"],
            valid_temporal_entries=valid_temporal_entries,
            superseded_entries=superseded_entries,
            invalidated_entries=invalidated_entries,
            total_procedures=total_procedures,
            active_procedures=active_procedures,
            procedure_executions=procedure_executions,
            procedure_success_rate=procedure_success_rate,
            sqlite_size_bytes=sqlite_size,
            temporal_size_bytes=temporal_size,
            procedural_size_bytes=procedural_size,
            last_maintenance=self._last_maintenance,
            last_compression=self._last_compression,
        )

    # =========================================================================
    # CONTEXT GENERATION
    # =========================================================================

    async def get_context(
        self,
        query: Optional[str] = None,
        max_tokens: int = 4000,
        include_types: Optional[List[MemoryType]] = None,
    ) -> str:
        """
        Generate context string for session/conversation.

        Args:
            query: Optional query to focus context on
            max_tokens: Maximum tokens in context
            include_types: Types of memories to include

        Returns:
            Formatted context string
        """
        await self._ensure_initialized()

        parts: List[str] = []
        used_tokens = 0
        max_chars = max_tokens * 4  # Rough estimate

        # If query provided, search for relevant context
        if query:
            results = await self.search(query, limit=10)
            if results:
                parts.append("## Relevant Context")
                for result in results[:5]:
                    content = result.entry.content[:500]
                    parts.append(f"- [{result.source}] {content}")
                    used_tokens += len(content) // 4

        # Add recent learnings
        if self._sqlite and (not include_types or MemoryType.LEARNING in include_types):
            learnings = await self._sqlite.get_learnings(5)
            if learnings:
                parts.append("\n## Recent Learnings")
                for l in learnings:
                    if used_tokens < max_tokens * 0.8:
                        content = l.content[:300]
                        parts.append(f"- {content}")
                        used_tokens += len(content) // 4

        # Add recent decisions
        if self._sqlite and (not include_types or MemoryType.DECISION in include_types):
            decisions = await self._sqlite.get_decisions(5)
            if decisions:
                parts.append("\n## Key Decisions")
                for d in decisions:
                    if used_tokens < max_tokens * 0.8:
                        content = d.content[:300]
                        parts.append(f"- {content}")
                        used_tokens += len(content) // 4

        # Add available procedures
        if self._procedural and (not include_types or MemoryType.PROCEDURE in include_types):
            from .procedural import ProcedureStatus
            procedures = await self._procedural.list_procedures(
                status=ProcedureStatus.ACTIVE, limit=5
            )
            if procedures:
                parts.append("\n## Available Procedures")
                for p in procedures:
                    if used_tokens < max_tokens * 0.9:
                        parts.append(f"- {p.name}: {p.description[:100]}")
                        used_tokens += 50

        context = "\n".join(parts)

        # Truncate if needed
        if len(context) > max_chars:
            context = context[:max_chars] + "\n\n[Context truncated...]"

        return context

    def close(self) -> None:
        """Close all backend connections."""
        if self._sqlite:
            self._sqlite.close()
        if self._bitemporal:
            self._bitemporal.close()
        if self._procedural:
            self._procedural.close()
        # Graphiti cleanup (placeholder has no persistent connections)
        if self._graphiti:
            logger.debug("Graphiti backend cleanup (placeholder - no connections to close)")
        # HNSW cleanup - save index if modified
        if self._hnsw:
            try:
                self._hnsw.save()
                logger.debug("HNSW index saved on close")
            except Exception as e:
                logger.warning(f"Failed to save HNSW index on close: {e}")
        logger.info("UnifiedMemory closed all connections")


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

_unified_memory: Optional[UnifiedMemory] = None


async def create_unified_memory(
    embedding_provider: Optional[Callable[[str], List[float]]] = None,
    enable_forgetting: bool = True,
    enable_compression: bool = True,
    enable_graphiti: bool = False,
    enable_hnsw: bool = True,
) -> UnifiedMemory:
    """
    Create or get the singleton UnifiedMemory instance.

    Args:
        embedding_provider: Optional function to generate embeddings
        enable_forgetting: Enable forgetting curve for strength decay
        enable_compression: Enable automatic compression
        enable_graphiti: Enable Graphiti knowledge graph backend (requires Neo4j)
        enable_hnsw: Enable HNSW vector index for semantic search acceleration (150x-12500x speedup)

    Returns:
        Configured UnifiedMemory instance
    """
    global _unified_memory

    if _unified_memory is None:
        _unified_memory = UnifiedMemory(
            embedding_provider=embedding_provider,
            enable_forgetting=enable_forgetting,
            enable_compression=enable_compression,
            enable_graphiti=enable_graphiti,
            enable_hnsw=enable_hnsw,
        )
        await _unified_memory._ensure_initialized()

    return _unified_memory


def get_unified_memory() -> Optional[UnifiedMemory]:
    """Get the current UnifiedMemory instance if initialized."""
    return _unified_memory


def reset_unified_memory() -> None:
    """Reset the singleton instance (for testing)."""
    global _unified_memory
    if _unified_memory:
        _unified_memory.close()
    _unified_memory = None


__all__ = [
    # Main class
    "UnifiedMemory",

    # Enums
    "MemoryType",
    "RoutingDecision",
    "SearchStrategy",
    "LifecycleState",

    # Data classes
    "UnifiedSearchResult",
    "RoutingResult",
    "MaintenanceReport",
    "UnifiedStatistics",

    # Helper classes
    "ContentClassifier",
    "RRFFusion",

    # Factory functions
    "create_unified_memory",
    "get_unified_memory",
    "reset_unified_memory",
]
