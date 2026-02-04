"""
Memory Backend Base Classes - V36 Architecture

This module defines the abstract base classes and common types for memory backends.
All memory backends (Letta, Mem0, Graphiti, In-Memory) implement these interfaces.

V36 Consolidation:
- Unified TierBackend and MemoryBackend concepts
- Common MemoryEntry structure across all backends
- Shared utility functions

Usage:
    from core.memory.backends.base import (
        MemoryBackend,
        TierBackend,
        MemoryEntry,
        MemoryTier,
        MemoryPriority,
        MemoryAccessPattern,
    )
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar


# =============================================================================
# MEMORY TIER DEFINITIONS
# =============================================================================

class MemoryTier(str, Enum):
    """Letta/MemGPT-inspired memory tiers."""
    MAIN_CONTEXT = "main_context"    # RAM analog, ~8K tokens, fastest
    CORE_MEMORY = "core_memory"      # Always visible, compressed facts
    RECALL_MEMORY = "recall_memory"  # Searchable conversation history
    ARCHIVAL_MEMORY = "archival"     # Long-term vector DB, unlimited


class MemoryPriority(str, Enum):
    """Priority levels for memory entries."""
    CRITICAL = "critical"    # Never evict (user preferences, persona)
    HIGH = "high"            # Rarely evict (recent context, active tasks)
    NORMAL = "normal"        # Standard eviction rules
    LOW = "low"              # Evict first when space needed


class MemoryAccessPattern(str, Enum):
    """Access patterns for intelligent caching."""
    HOT = "hot"          # Frequently accessed, keep in higher tier
    WARM = "warm"        # Occasional access
    COLD = "cold"        # Rarely accessed, can demote
    FROZEN = "frozen"    # Archive candidate


class MemoryLayer(str, Enum):
    """Memory layers in the 5-layer gateway stack."""
    LETTA = "letta"           # Layer 1: Project-specific agents
    CLAUDE_MEM = "claude_mem"  # Layer 2: Observations
    EPISODIC = "episodic"      # Layer 3: Conversation archive
    GRAPH = "graph"            # Layer 4: Entity relationships
    STATIC = "static"          # Layer 5: CLAUDE.md configuration


class MemoryNamespace(str, Enum):
    """Memory namespaces for TTL management."""
    ARTIFACTS = "artifacts"    # Permanent - final outputs
    SHARED = "shared"          # 30 min - coordination state
    PATTERNS = "patterns"      # 7 days - learned tactics
    DECISIONS = "decisions"    # 7 days - architecture choices
    EVENTS = "events"          # 30 days - audit trail
    CONTEXT = "context"        # Session - current context
    LEARNINGS = "learnings"    # 7 days - extracted learnings


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MemoryEntry:
    """A single memory entry that can live in any tier.

    V36 Unified: Combines fields from both memory_tiers.py and unified_memory_gateway.py
    to provide a single, consistent memory entry structure.

    V40 Extension: Adds forgetting curve and memory strength decay support.
    """
    id: str
    content: str
    tier: MemoryTier = MemoryTier.MAIN_CONTEXT
    priority: MemoryPriority = MemoryPriority.NORMAL
    access_pattern: MemoryAccessPattern = MemoryAccessPattern.WARM

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0

    # Content metadata
    token_count: int = 0
    content_type: str = "text"  # text, fact, preference, context, task
    source: str = ""  # Where this memory came from
    namespace: Optional[MemoryNamespace] = None

    # Relationships
    related_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Letta-specific
    block_label: Optional[str] = None  # For core memory blocks
    embedding: Optional[List[float]] = None  # For semantic search

    # TTL support
    ttl_seconds: Optional[int] = None

    # V36: Cached content hash for deduplication
    _content_hash: Optional[str] = field(default=None, init=False, repr=False)

    # V40: Forgetting curve and memory strength decay
    strength: float = 1.0           # Current strength (0.0-1.0)
    decay_rate: float = 0.15        # Decay rate per day
    last_reinforced: Optional[datetime] = None  # Last access/reinforcement time
    reinforcement_count: int = 0    # Total reinforcements

    def touch(self) -> None:
        """Update access metadata."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1
        self._update_access_pattern()

    def _update_access_pattern(self) -> None:
        """Update access pattern based on access history."""
        now = datetime.now(timezone.utc)
        age_hours = (now - self.last_accessed).total_seconds() / 3600

        # Simple heuristic: access count + recency
        if self.access_count > 10 and age_hours < 1:
            self.access_pattern = MemoryAccessPattern.HOT
        elif self.access_count > 5 and age_hours < 24:
            self.access_pattern = MemoryAccessPattern.WARM
        elif age_hours > 168:  # 1 week
            self.access_pattern = MemoryAccessPattern.FROZEN
        else:
            self.access_pattern = MemoryAccessPattern.COLD

    @property
    def is_expired(self) -> bool:
        """Check if this entry has expired based on TTL."""
        if self.ttl_seconds is None:
            return False
        age = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def content_hash(self) -> str:
        """Generate a hash of the content for deduplication.

        Caches hash on first computation to avoid repeated MD5 calculations.
        """
        if self._content_hash is None:
            self._content_hash = hashlib.md5(self.content.encode()).hexdigest()[:12]
        return self._content_hash

    def calculate_current_strength(self) -> float:
        """Calculate current memory strength using Ebbinghaus forgetting curve.

        Formula: strength = initial_strength * e^(-decay_rate * days_since_reinforcement)

        Returns:
            Current strength value (0.0-1.0)
        """
        import math

        # Get reference time for decay calculation
        reference_time = self.last_reinforced or self.created_at
        if reference_time is None:
            return self.strength

        now = datetime.now(timezone.utc)

        # Ensure reference_time is timezone-aware
        if reference_time.tzinfo is None:
            reference_time = reference_time.replace(tzinfo=timezone.utc)

        # Calculate days elapsed since last reinforcement
        delta = now - reference_time
        days_elapsed = delta.total_seconds() / 86400  # Seconds per day

        if days_elapsed <= 0:
            return self.strength

        # Apply importance weighting (higher importance = slower decay)
        importance = self.metadata.get('importance', 0.5) if self.metadata else 0.5
        importance_factor = 1.0 - (importance * 0.5)
        effective_decay_rate = self.decay_rate * importance_factor

        # Priority-based decay adjustment
        if self.priority == MemoryPriority.CRITICAL:
            effective_decay_rate *= 0.1  # Critical memories decay very slowly
        elif self.priority == MemoryPriority.HIGH:
            effective_decay_rate *= 0.5  # High priority decays slowly
        elif self.priority == MemoryPriority.LOW:
            effective_decay_rate *= 2.0  # Low priority decays faster

        # Ebbinghaus forgetting curve: S = S_0 * e^(-lambda * t)
        current_strength = self.strength * math.exp(-effective_decay_rate * days_elapsed)

        # Clamp to valid range
        return max(0.0, min(1.0, current_strength))

    def reinforce(self, access_type: str = "recall") -> float:
        """Reinforce memory on access, boosting strength.

        Args:
            access_type: Type of access - "recall", "review", "reference", "passive"

        Returns:
            New strength value after reinforcement
        """
        # Calculate current decayed strength first
        current_strength = self.calculate_current_strength()

        # Reinforcement amounts by access type
        reinforcement_amounts = {
            "recall": 0.2,      # Active recall provides strong reinforcement
            "review": 0.15,     # Explicit review
            "reference": 0.1,   # Referenced by other memories
            "passive": 0.05,    # Passive exposure
            "search_hit": 0.08, # Appeared in search results
        }

        reinforcement = reinforcement_amounts.get(access_type, 0.05)

        # Spaced repetition bonus if enough time has passed
        reference_time = self.last_reinforced or self.created_at
        if reference_time:
            if reference_time.tzinfo is None:
                reference_time = reference_time.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            days_since_last = (now - reference_time).total_seconds() / 86400

            # Optimal spacing thresholds (days)
            optimal_thresholds = [1, 3, 7, 14, 30, 60]
            threshold_idx = min(self.reinforcement_count, len(optimal_thresholds) - 1)
            expected_interval = optimal_thresholds[threshold_idx]

            # Bonus for optimal timing (0.8-1.2x expected interval)
            if expected_interval > 0:
                ratio = days_since_last / expected_interval
                if 0.8 <= ratio <= 1.2:
                    reinforcement += 0.1  # Spaced repetition bonus

        # Diminishing returns at high strength
        if current_strength > 0.8:
            reinforcement *= (1.0 - current_strength) * 2

        # Update strength
        new_strength = min(1.0, current_strength + reinforcement)
        self.strength = new_strength
        self.last_reinforced = datetime.now(timezone.utc)
        self.reinforcement_count += 1

        # Also update access metadata
        self.touch()

        return new_strength

    @property
    def is_weak(self) -> bool:
        """Check if memory strength is below archive threshold."""
        return self.calculate_current_strength() < 0.1

    @property
    def is_very_weak(self) -> bool:
        """Check if memory strength is below delete threshold."""
        return self.calculate_current_strength() < 0.01


@dataclass
class TierConfig:
    """Configuration for a memory tier."""
    tier: MemoryTier
    max_tokens: int
    max_entries: int
    eviction_threshold: float = 0.8  # Start evicting at 80% capacity
    ttl_hours: Optional[int] = None  # Time-to-live for entries

    # Letta API config (for core/archival)
    letta_enabled: bool = False
    letta_agent_id: Optional[str] = None


@dataclass
class MemoryStats:
    """Statistics for memory system."""
    total_entries: int
    entries_by_tier: Dict[MemoryTier, int]
    total_tokens: int
    tokens_by_tier: Dict[MemoryTier, int]
    hit_rate: float
    eviction_count: int
    promotion_count: int  # Moved to higher tier
    demotion_count: int   # Moved to lower tier


@dataclass
class MemorySearchResult:
    """Result from memory search."""
    entry: MemoryEntry
    score: float
    tier: MemoryTier
    match_type: str  # exact, semantic, fuzzy


@dataclass
class MemoryQuery:
    """A query against the memory gateway."""
    query_text: str
    layers: Optional[List[MemoryLayer]] = None  # None = all layers
    namespace: Optional[MemoryNamespace] = None
    max_results: int = 15
    min_relevance: float = 0.5
    include_metadata: bool = True
    project: Optional[str] = None  # For Letta agent selection


@dataclass
class MemoryResult:
    """Result from a memory query."""
    entries: List[MemoryEntry]
    total_found: int
    sources_queried: List[MemoryLayer]
    duration_ms: float
    deduplicated_count: int = 0


# =============================================================================
# ABSTRACT BACKENDS
# =============================================================================

T = TypeVar('T')


class TierBackend(ABC, Generic[T]):
    """Abstract backend for tier storage.

    V36: Unified interface for all memory tier backends.
    Implementations: InMemoryTierBackend, LettaTierBackend, Mem0TierBackend, GraphitiTierBackend
    """

    @abstractmethod
    async def get(self, key: str) -> Optional[T]:
        """Get entry by key."""
        pass

    @abstractmethod
    async def put(self, key: str, value: T) -> None:
        """Store entry."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete entry."""
        pass

    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[T]:
        """Search entries."""
        pass

    @abstractmethod
    async def list_all(self) -> List[T]:
        """List all entries."""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Get entry count."""
        pass


class MemoryBackend(ABC):
    """Abstract base class for gateway memory backends.

    V36: Interface for the 5-layer gateway backends.
    Each layer (Letta, Claude-mem, Episodic, Graph, Static) implements this.
    """

    @property
    @abstractmethod
    def layer(self) -> MemoryLayer:
        """Return the layer this backend represents."""
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 10,
        namespace: Optional[MemoryNamespace] = None
    ) -> List[MemoryEntry]:
        """Search this memory backend."""
        pass

    @abstractmethod
    async def store(
        self,
        content: str,
        namespace: MemoryNamespace,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a memory entry. Returns the entry ID."""
        pass

    async def health_check(self) -> bool:
        """Check if this backend is healthy."""
        return True


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_memory_id(content: str, prefix: str = "") -> str:
    """Generate a unique memory ID from content."""
    timestamp = str(time.time())
    hash_input = f"{content[:100]}:{timestamp}"
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:12]
    return f"{prefix}{hash_value}" if prefix else hash_value


def estimate_tokens(content: str) -> int:
    """Estimate token count (rough: 4 chars per token)."""
    return len(content) // 4


# TTL configuration in seconds
TTL_CONFIG: Dict[MemoryNamespace, Optional[int]] = {
    MemoryNamespace.ARTIFACTS: None,              # Permanent
    MemoryNamespace.SHARED: 30 * 60,              # 30 minutes
    MemoryNamespace.PATTERNS: 7 * 24 * 60 * 60,   # 7 days
    MemoryNamespace.DECISIONS: 7 * 24 * 60 * 60,  # 7 days
    MemoryNamespace.EVENTS: 30 * 24 * 60 * 60,    # 30 days
    MemoryNamespace.CONTEXT: None,                # Session-scoped
    MemoryNamespace.LEARNINGS: 7 * 24 * 60 * 60,  # 7 days
}


__all__ = [
    # Enums
    "MemoryTier",
    "MemoryPriority",
    "MemoryAccessPattern",
    "MemoryLayer",
    "MemoryNamespace",
    # Data classes
    "MemoryEntry",
    "TierConfig",
    "MemoryStats",
    "MemorySearchResult",
    "MemoryQuery",
    "MemoryResult",
    # Abstract backends
    "TierBackend",
    "MemoryBackend",
    # Utilities
    "generate_memory_id",
    "estimate_tokens",
    "TTL_CONFIG",
]
