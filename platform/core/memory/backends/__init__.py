"""
Memory Backends - V36 Architecture

This package contains memory backend implementations for different storage systems.

Backends:
- base.py: MemoryBackend ABC and common types (V36 canonical)
- letta.py: Letta memory integration (V36 canonical)
- in_memory.py: In-memory backend for testing (V36 canonical)
- mem0.py: Mem0 SDK integration (TODO)
- graphiti.py: Graphiti temporal knowledge graphs (TODO)

Usage:
    from core.memory.backends import LettaTierBackend, InMemoryTierBackend
    from core.memory.backends.base import MemoryEntry, MemoryTier
"""

from __future__ import annotations

import importlib
from typing import Any, Dict

# Lazy loading cache
_loaded: Dict[str, Any] = {}

# Module mapping for lazy loading - V36 prefers modular files, fallback to legacy
_BACKEND_MAP = {
    # V36 Canonical - from modular files
    "MemoryBackend": (".base", "MemoryBackend"),
    "TierBackend": (".base", "TierBackend"),
    "MemoryEntry": (".base", "MemoryEntry"),
    "MemoryTier": (".base", "MemoryTier"),
    "MemoryPriority": (".base", "MemoryPriority"),
    "MemoryAccessPattern": (".base", "MemoryAccessPattern"),
    "MemoryLayer": (".base", "MemoryLayer"),
    "MemoryNamespace": (".base", "MemoryNamespace"),
    "TierConfig": (".base", "TierConfig"),
    "MemoryStats": (".base", "MemoryStats"),
    "MemorySearchResult": (".base", "MemorySearchResult"),
    "MemoryQuery": (".base", "MemoryQuery"),
    "MemoryResult": (".base", "MemoryResult"),
    "TTL_CONFIG": (".base", "TTL_CONFIG"),

    # V36 In-memory backend
    "InMemoryTierBackend": (".in_memory", "InMemoryTierBackend"),

    # V36 Letta backends
    "LettaTierBackend": (".letta", "LettaTierBackend"),
    "LettaMemoryBackend": (".letta", "LettaMemoryBackend"),

    # V36 SQLite backend (REAL persistent cross-session memory)
    "SQLiteTierBackend": (".sqlite", "SQLiteTierBackend"),
    "get_sqlite_backend": (".sqlite", "get_sqlite_backend"),

    # V36 Graphiti backend (temporal knowledge graphs - PLACEHOLDER)
    "GraphitiTierBackend": (".graphiti", "GraphitiTierBackend"),
    "GraphitiMemoryBackend": (".graphiti", "GraphitiMemoryBackend"),
    "get_graphiti_backend": (".graphiti", "get_graphiti_backend"),
    "GraphitiEntity": (".graphiti", "GraphitiEntity"),
    "GraphitiRelation": (".graphiti", "GraphitiRelation"),
    "GraphitiEpisode": (".graphiti", "GraphitiEpisode"),
    "EntityType": (".graphiti", "EntityType"),
    "RelationType": (".graphiti", "RelationType"),
    "EpisodeType": (".graphiti", "EpisodeType"),

    # V41 HNSW backend (high-performance vector search, 150x-12500x speedup)
    "HNSWBackend": (".hnsw", "HNSWBackend"),
    "HNSWConfig": (".hnsw", "HNSWConfig"),
    "HNSWSearchResult": (".hnsw", "HNSWSearchResult"),
    "HNSWLibBackend": (".hnsw", "HNSWLibBackend"),
    "PurePythonHNSWBackend": (".hnsw", "PurePythonHNSWBackend"),
    "get_hnsw_backend": (".hnsw", "get_hnsw_backend"),
    "reset_hnsw_backend": (".hnsw", "reset_hnsw_backend"),
    "HNSWLIB_AVAILABLE": (".hnsw", "HNSWLIB_AVAILABLE"),
    "NUMPY_AVAILABLE": (".hnsw", "NUMPY_AVAILABLE"),

    # Legacy compatibility - from unified_memory_gateway.py
    "LettaArchivesBackend": ("...unified_memory_gateway", "LettaArchivesBackend"),
    "ClaudeMemBackend": ("...unified_memory_gateway", "ClaudeMemBackend"),
    "EpisodicMemoryBackend": ("...unified_memory_gateway", "EpisodicMemoryBackend"),
    "GraphMemoryBackend": ("...unified_memory_gateway", "GraphMemoryBackend"),
    "StaticMemoryBackend": ("...unified_memory_gateway", "StaticMemoryBackend"),
}


def __getattr__(name: str) -> Any:
    """Lazy load backends on first access."""
    if name in _loaded:
        return _loaded[name]

    if name in _BACKEND_MAP:
        module_path, attr_name = _BACKEND_MAP[name]
        try:
            module = importlib.import_module(module_path, package=__name__)
            value = getattr(module, attr_name)
            _loaded[name] = value
            return value
        except (ImportError, AttributeError) as e:
            raise AttributeError(f"Cannot import backend {name}: {e}")

    raise AttributeError(f"module 'platform.core.memory.backends' has no attribute '{name}'")


def __dir__():
    """Return all available names for tab completion."""
    return list(__all__)


__all__ = [
    # V36 Base types (canonical)
    "MemoryBackend",
    "TierBackend",
    "MemoryEntry",
    "MemoryTier",
    "MemoryPriority",
    "MemoryAccessPattern",
    "MemoryLayer",
    "MemoryNamespace",
    "TierConfig",
    "MemoryStats",
    "MemorySearchResult",
    "MemoryQuery",
    "MemoryResult",
    "TTL_CONFIG",
    # V36 Backends (canonical)
    "InMemoryTierBackend",
    "LettaTierBackend",
    "LettaMemoryBackend",
    # V36 SQLite backend (REAL persistent cross-session memory)
    "SQLiteTierBackend",
    "get_sqlite_backend",
    # V36 Graphiti backend (temporal knowledge graphs - PLACEHOLDER)
    "GraphitiTierBackend",
    "GraphitiMemoryBackend",
    "get_graphiti_backend",
    "GraphitiEntity",
    "GraphitiRelation",
    "GraphitiEpisode",
    "EntityType",
    "RelationType",
    "EpisodeType",
    # V41 HNSW backend (high-performance vector search, 150x-12500x speedup)
    "HNSWBackend",
    "HNSWConfig",
    "HNSWSearchResult",
    "HNSWLibBackend",
    "PurePythonHNSWBackend",
    "get_hnsw_backend",
    "reset_hnsw_backend",
    "HNSWLIB_AVAILABLE",
    "NUMPY_AVAILABLE",
    # Legacy compatibility
    "LettaArchivesBackend",
    "ClaudeMemBackend",
    "EpisodicMemoryBackend",
    "GraphMemoryBackend",
    "StaticMemoryBackend",
]
