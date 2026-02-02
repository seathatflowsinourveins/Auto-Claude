#!/usr/bin/env python3
"""
Memory Layer Types - Shared Models and Enums
Part of the V33 Unified Memory Architecture.
"""

from __future__ import annotations

from enum import Enum
from datetime import datetime
from typing import Any, Optional
from dataclasses import dataclass, field

from pydantic import BaseModel, Field


class MemoryTier(str, Enum):
    """Memory tiers in the three-tier architecture."""
    CORE = "core"           # Always in-context
    ARCHIVAL = "archival"   # Vector-retrieved
    TEMPORAL = "temporal"   # Knowledge graph


class MemoryProvider(str, Enum):
    """Available memory providers."""
    LOCAL = "local"         # Local file-based
    LETTA = "letta"         # Letta Cloud
    ZEP = "zep"             # Zep memory
    MEM0 = "mem0"           # Mem0 persistence
    CROSS_SESSION = "cross_session"  # Cross-session persistent memory


class ConsolidationStrategy(str, Enum):
    """Memory consolidation strategies."""
    SUMMARIZE = "summarize"     # Create summaries
    COMPRESS = "compress"       # Merge similar entries
    PRUNE = "prune"             # Remove low-importance
    HIERARCHICAL = "hierarchical"  # Build hierarchies


class MemoryEntry(BaseModel):
    """A single memory entry."""
    id: str
    content: str
    tier: MemoryTier = MemoryTier.ARCHIVAL
    provider: MemoryProvider = MemoryProvider.LOCAL
    metadata: dict[str, Any] = Field(default_factory=dict)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    embedding: Optional[list[float]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    access_count: int = 0


class MemoryBlock(BaseModel):
    """A memory block for core memory."""
    label: str
    content: str
    max_tokens: int = 2000
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """Result from a memory search."""
    id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    provider: MemoryProvider = MemoryProvider.LOCAL


class MemoryStats(BaseModel):
    """Statistics about memory usage."""
    total_entries: int = 0
    entries_by_tier: dict[str, int] = Field(default_factory=dict)
    entries_by_provider: dict[str, int] = Field(default_factory=dict)
    total_tokens_estimated: int = 0
    last_consolidation: Optional[datetime] = None


@dataclass
class ProviderConfig:
    """Configuration for a memory provider."""
    provider: MemoryProvider
    enabled: bool = True
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryConfig:
    """Configuration for the memory system."""
    default_provider: MemoryProvider = MemoryProvider.LOCAL
    providers: list[ProviderConfig] = field(default_factory=list)
    consolidation_enabled: bool = True
    consolidation_strategy: ConsolidationStrategy = ConsolidationStrategy.SUMMARIZE
    auto_sync: bool = True
    embedding_model: str = "text-embedding-3-small"
