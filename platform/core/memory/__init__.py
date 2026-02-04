"""
Unified Memory System - V41 Architecture

This package consolidates all memory implementations into a clean, modular structure.
Uses lazy loading for optimal startup performance.

Exported Components:
--------------------

Core Memory Classes:
- UnifiedMemory: Single interface to all memory systems with auto-routing
- ForgettingCurve: Memory strength decay with Ebbinghaus model
- ProceduralMemory: Learned behavior patterns and procedures
- BiTemporalMemory: Dual-timeline memory (event time + system time)
- SemanticCompressor: See CompressionStrategyBase and strategies

Factory Functions:
- create_unified_memory: Create UnifiedMemory with default configuration
- create_memory_gateway: Create UnifiedMemoryGateway
- create_tier_manager: Create MemoryTierManager

V36 Modular Structure:
```
memory/
├── __init__.py              # Public API (this file)
├── backends/
│   ├── __init__.py          # Backend exports
│   ├── base.py              # MemoryEntry, TierBackend, MemoryBackend ABCs (CANONICAL)
│   ├── letta.py             # LettaTierBackend, LettaMemoryBackend (CANONICAL)
│   ├── in_memory.py         # InMemoryTierBackend (CANONICAL)
│   ├── mem0.py              # Mem0 integration (TODO)
│   └── graphiti.py          # Graphiti temporal graphs (TODO)
```

Import Patterns:
    # Primary API - V36 canonical types
    from core.memory import MemoryTierManager, UnifiedMemoryGateway
    from core.memory.backends import MemoryEntry, LettaTierBackend

    # Legacy compatibility (deprecated)
    from core.memory import MemorySystem, CoreMemory  # -> use UnifiedMemoryGateway

Migration Guide:
- MemorySystem -> UnifiedMemoryGateway (preferred) or MemoryTierManager
- CoreMemory -> UnifiedMemoryGateway.core_layer
- ArchivalMemory -> UnifiedMemoryGateway.archival_layer
- CrossSessionMemory -> UnifiedMemoryGateway.cross_session()
- MemoryEntry (memory_tiers) -> backends.base.MemoryEntry (V36 canonical)
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Dict

# =============================================================================
# LEGACY COMPATIBILITY - Direct imports for `from .memory import X` syntax
# =============================================================================
# These must be explicitly imported because `from X import Y` doesn't trigger
# __getattr__. The legacy classes live in ../memory.py (parent memory.py file).
# NOTE: `from ..memory import X` is circular because this package IS core.memory.
# We must load the legacy file directly using importlib.
try:
    import importlib.util as _ilu
    import sys as _sys
    from pathlib import Path as _P
    _legacy_path = str(_P(__file__).parent.parent / "memory.py")
    _legacy_mod_name = "core._memory_legacy"
    _spec = _ilu.spec_from_file_location(_legacy_mod_name, _legacy_path)
    _legacy = _ilu.module_from_spec(_spec)
    _sys.modules[_legacy_mod_name] = _legacy  # Register before exec for dataclass compat
    _spec.loader.exec_module(_legacy)
    MemorySystem = _legacy.MemorySystem
    CoreMemory = _legacy.CoreMemory
    ArchivalMemory = _legacy.ArchivalMemory
    MemoryBlock = _legacy.MemoryBlock
    TemporalGraph = _legacy.TemporalGraph
except Exception:
    # Fallback if legacy file doesn't exist
    MemorySystem = None
    CoreMemory = None
    ArchivalMemory = None
    MemoryBlock = None
    TemporalGraph = None

# Type hints only - no runtime import
if TYPE_CHECKING:
    # V36 canonical types
    from .backends.base import (
        MemoryEntry, MemoryTier, MemoryPriority, MemoryAccessPattern,
        MemoryLayer, MemoryNamespace, TierConfig, MemoryStats,
        MemorySearchResult, MemoryQuery, MemoryResult,
        TierBackend, MemoryBackend,
    )
    from .backends.in_memory import InMemoryTierBackend
    from .backends.letta import LettaTierBackend, LettaMemoryBackend
    # System classes
    from ..memory_tiers import MemoryTierManager, MemoryPressureLevel, SleepTimeAgent
    from ..unified_memory_gateway import UnifiedMemoryGateway
    # Legacy compatibility
    from ..memory import MemorySystem, CoreMemory, ArchivalMemory, MemoryBlock
    from ..cross_session_memory import CrossSessionMemory
    from ..advanced_memory import AdvancedMemorySystem, MemoryConsolidator

# Lazy loading cache
_loaded: Dict[str, Any] = {}
_loading: set = set()  # Guard against recursive imports

# Module mapping for lazy loading
# V36: Prefers modular backends/, falls back to legacy files
_MODULE_MAP = {
    # V36 CANONICAL - Base types from backends.base (NEW)
    "MemoryEntry": (".backends.base", "MemoryEntry"),
    "MemoryTier": (".backends.base", "MemoryTier"),
    "MemoryPriority": (".backends.base", "MemoryPriority"),
    "MemoryAccessPattern": (".backends.base", "MemoryAccessPattern"),
    "MemoryLayer": (".backends.base", "MemoryLayer"),
    "MemoryNamespace": (".backends.base", "MemoryNamespace"),
    "TierConfig": (".backends.base", "TierConfig"),
    "MemoryStats": (".backends.base", "MemoryStats"),
    "MemorySearchResult": (".backends.base", "MemorySearchResult"),
    "MemoryQuery": (".backends.base", "MemoryQuery"),
    "MemoryResult": (".backends.base", "MemoryResult"),
    "TierBackend": (".backends.base", "TierBackend"),
    "MemoryBackend": (".backends.base", "MemoryBackend"),

    # V36 CANONICAL - Backends (NEW)
    "InMemoryTierBackend": (".backends.in_memory", "InMemoryTierBackend"),
    "LettaTierBackend": (".backends.letta", "LettaTierBackend"),
    "LettaMemoryBackend": (".backends.letta", "LettaMemoryBackend"),

    # Primary system classes (from memory_tiers.py - kept for MemoryTierManager)
    "MemoryTierManager": ("..memory_tiers", "MemoryTierManager"),
    "MemoryPressureLevel": ("..memory_tiers", "MemoryPressureLevel"),
    "SleepTimeAgent": ("..memory_tiers", "SleepTimeAgent"),
    "MemoryTierIntegration": ("..memory_tiers", "MemoryTierIntegration"),

    # Unified Gateway (from unified_memory_gateway.py)
    "UnifiedMemoryGateway": ("..unified_memory_gateway", "UnifiedMemoryGateway"),
    "LettaArchivesBackend": ("..unified_memory_gateway", "LettaArchivesBackend"),
    "ClaudeMemBackend": ("..unified_memory_gateway", "ClaudeMemBackend"),
    "EpisodicMemoryBackend": ("..unified_memory_gateway", "EpisodicMemoryBackend"),
    "GraphMemoryBackend": ("..unified_memory_gateway", "GraphMemoryBackend"),
    "StaticMemoryBackend": ("..unified_memory_gateway", "StaticMemoryBackend"),

    # Advanced Memory (from advanced_memory.py)
    "AdvancedMemorySystem": ("..advanced_memory", "AdvancedMemorySystem"),
    "MemoryConsolidator": ("..advanced_memory", "MemoryConsolidator"),
    "MemoryMetrics": ("..advanced_memory", "MemoryMetrics"),

    # Cross-session (from cross_session_memory.py)
    "CrossSessionMemory": ("..cross_session_memory", "CrossSessionMemory"),

    # V36 SQLite persistent backend (REAL cross-session memory)
    "SQLiteTierBackend": (".backends.sqlite", "SQLiteTierBackend"),
    "get_sqlite_backend": (".backends.sqlite", "get_sqlite_backend"),

    # V36 Memory hooks for session lifecycle
    "session_start_hook": (".hooks", "session_start_hook"),
    "session_end_hook": (".hooks", "session_end_hook"),
    "remember_decision": (".hooks", "remember_decision"),
    "remember_learning": (".hooks", "remember_learning"),
    "remember_fact": (".hooks", "remember_fact"),
    "recall": (".hooks", "recall"),
    "get_context": (".hooks", "get_context"),

    # V36 Memory quality tracking
    "MemoryQualityTracker": (".quality", "MemoryQualityTracker"),
    "MemoryQualityMetrics": (".quality", "MemoryQualityMetrics"),
    "ConflictReport": (".quality", "ConflictReport"),
    "RetrievalMetrics": (".quality", "RetrievalMetrics"),
    "ConsolidationRecommendation": (".quality", "ConsolidationRecommendation"),
    "QualityReport": (".quality", "QualityReport"),
    "QualityConfig": (".quality", "QualityConfig"),
    "ConflictType": (".quality", "ConflictType"),
    "ConsolidationAction": (".quality", "ConsolidationAction"),
    "get_quality_tracker": (".quality", "get_quality_tracker"),
    "analyze_memory_quality": (".quality", "analyze_memory_quality"),
    "get_stale_memories": (".quality", "get_stale_memories"),
    "detect_memory_conflicts": (".quality", "detect_memory_conflicts"),
    "get_memory_quality_report": (".quality", "get_memory_quality_report"),

    # V36 Memory compression for long-term storage
    "MemoryCompressor": (".compression", "MemoryCompressor"),
    "CompressionStrategy": (".compression", "CompressionStrategy"),
    "CompressionTrigger": (".compression", "CompressionTrigger"),
    "CompressionConfig": (".compression", "CompressionConfig"),
    "CompressionResult": (".compression", "CompressionResult"),
    "CompressionMetrics": (".compression", "CompressionMetrics"),
    "CompressedMemory": (".compression", "CompressedMemory"),
    "CompressionStrategyBase": (".compression", "CompressionStrategyBase"),
    "ExtractiveStrategy": (".compression", "ExtractiveStrategy"),
    "AbstractiveStrategy": (".compression", "AbstractiveStrategy"),
    "HierarchicalStrategy": (".compression", "HierarchicalStrategy"),
    "ClusteringStrategy": (".compression", "ClusteringStrategy"),
    "BackgroundCompressor": (".compression", "BackgroundCompressor"),
    "get_compressor": (".compression", "get_compressor"),
    "get_background_compressor": (".compression", "get_background_compressor"),
    "compress_old_memories": (".compression", "compress_old_memories"),
    "get_compression_stats": (".compression", "get_compression_stats"),
    "decompress_memory": (".compression", "decompress_memory"),

    # V40 Forgetting curve and memory strength decay
    "ForgettingCurve": (".forgetting", "ForgettingCurve"),
    "MemoryStrength": (".forgetting", "MemoryStrength"),
    "DecayCategory": (".forgetting", "DecayCategory"),
    "ForgettingConsolidationReport": (".forgetting", "ConsolidationReport"),
    "MemoryConsolidationTask": (".forgetting", "MemoryConsolidationTask"),
    "apply_forgetting_to_entry": (".forgetting", "apply_forgetting_to_entry"),
    "reinforce_memory": (".forgetting", "reinforce_memory"),
    "get_memory_strength": (".forgetting", "get_memory_strength"),
    "apply_forgetting_migration": (".forgetting", "apply_forgetting_migration"),
    "DEFAULT_DECAY_RATES": (".forgetting", "DEFAULT_DECAY_RATES"),
    "REINFORCEMENT_MULTIPLIERS": (".forgetting", "REINFORCEMENT_MULTIPLIERS"),
    "STRENGTH_ARCHIVE_THRESHOLD": (".forgetting", "STRENGTH_ARCHIVE_THRESHOLD"),
    "STRENGTH_DELETE_THRESHOLD": (".forgetting", "STRENGTH_DELETE_THRESHOLD"),

    # V40 Bi-temporal memory model (Zep research)
    "BiTemporalMemory": (".temporal", "BiTemporalMemory"),
    "TemporalMemoryEntry": (".temporal", "TemporalMemoryEntry"),
    "TemporalSearchResult": (".temporal", "TemporalSearchResult"),
    "TemporalAggregation": (".temporal", "TemporalAggregation"),
    "create_bitemporal_memory": (".temporal", "create_bitemporal_memory"),
    "get_bitemporal_memory": (".temporal", "get_bitemporal_memory"),
    "reset_bitemporal_memory": (".temporal", "reset_bitemporal_memory"),
    "BITEMPORAL_SCHEMA_VERSION": (".temporal", "BITEMPORAL_SCHEMA_VERSION"),

    # V40 Procedural memory for learned behaviors
    "ProceduralMemory": (".procedural", "ProceduralMemory"),
    "Procedure": (".procedural", "Procedure"),
    "ProcedureStep": (".procedural", "ProcedureStep"),
    "ProcedureMatch": (".procedural", "ProcedureMatch"),
    "ExecutionResult": (".procedural", "ExecutionResult"),
    "ProcedureStatus": (".procedural", "ProcedureStatus"),
    "StepType": (".procedural", "StepType"),
    "ExecutionOutcome": (".procedural", "ExecutionOutcome"),
    "get_procedural_memory": (".procedural", "get_procedural_memory"),
    "learn_procedure": (".procedural", "learn_procedure"),
    "recall_procedure": (".procedural", "recall_procedure"),
    "execute_procedure": (".procedural", "execute_procedure"),
    "MIN_CONFIDENCE_FOR_AUTO_EXECUTE": (".procedural", "MIN_CONFIDENCE_FOR_AUTO_EXECUTE"),

    # V41 Unified Memory Interface (single interface to all memory systems)
    "UnifiedMemory": (".unified", "UnifiedMemory"),
    "MemoryType": (".unified", "MemoryType"),
    "RoutingDecision": (".unified", "RoutingDecision"),
    "SearchStrategy": (".unified", "SearchStrategy"),
    "LifecycleState": (".unified", "LifecycleState"),
    "UnifiedSearchResult": (".unified", "UnifiedSearchResult"),
    "RoutingResult": (".unified", "RoutingResult"),
    "MaintenanceReport": (".unified", "MaintenanceReport"),
    "UnifiedStatistics": (".unified", "UnifiedStatistics"),
    "ContentClassifier": (".unified", "ContentClassifier"),
    "RRFFusion": (".unified", "RRFFusion"),
    "create_unified_memory": (".unified", "create_unified_memory"),
    "get_unified_memory": (".unified", "get_unified_memory"),
    "reset_unified_memory": (".unified", "reset_unified_memory"),

    # V42 Memory Compaction for long-running sessions
    "MemoryCompactor": (".compaction", "MemoryCompactor"),
    "CompactionStrategy": (".compaction", "CompactionStrategy"),
    "CompactionPriority": (".compaction", "CompactionPriority"),
    "MergeStrategy": (".compaction", "MergeStrategy"),
    "CompactionConfig": (".compaction", "CompactionConfig"),
    "CompactionCandidate": (".compaction", "CompactionCandidate"),
    "MergeGroup": (".compaction", "MergeGroup"),
    "CompactionReport": (".compaction", "CompactionReport"),
    "CompactionMetrics": (".compaction", "CompactionMetrics"),
    "CompactionScheduler": (".compaction", "CompactionScheduler"),
    "get_memory_compactor": (".compaction", "get_memory_compactor"),
    "get_compaction_scheduler": (".compaction", "get_compaction_scheduler"),
    "compact_session_memories": (".compaction", "compact_session_memories"),
    "get_compaction_candidates": (".compaction", "get_compaction_candidates"),
    "get_compaction_status": (".compaction", "get_compaction_status"),

    # Legacy compatibility (from memory.py) - DEPRECATED
    # NOTE: These are eagerly loaded at module top via importlib file loader
    # to avoid circular import (this package IS core.memory, so ..memory is self).
    # They are NOT in the lazy loader map.
}


def __getattr__(name: str) -> Any:
    """Lazy load modules on first access."""
    if name in _loaded:
        return _loaded[name]

    # Guard against recursive imports
    if name in _loading:
        raise AttributeError(f"Recursive import detected for {name}")

    if name in _MODULE_MAP:
        _loading.add(name)
        module_path, attr_name = _MODULE_MAP[name]
        try:
            module = importlib.import_module(module_path, package=__name__)
            value = getattr(module, attr_name)
            _loaded[name] = value
            return value
        except (ImportError, AttributeError) as e:
            raise AttributeError(f"Cannot import {name}: {e}")
        finally:
            _loading.discard(name)

    # Availability flags
    if name == "MEMORY_TIER_AVAILABLE":
        try:
            importlib.import_module("..memory_tiers", package=__name__)
            _loaded[name] = True
        except ImportError:
            _loaded[name] = False
        return _loaded[name]

    if name == "UNIFIED_GATEWAY_AVAILABLE":
        try:
            importlib.import_module("..unified_memory_gateway", package=__name__)
            _loaded[name] = True
        except ImportError:
            _loaded[name] = False
        return _loaded[name]

    if name == "ADVANCED_MEMORY_AVAILABLE":
        try:
            importlib.import_module("..advanced_memory", package=__name__)
            _loaded[name] = True
        except ImportError:
            _loaded[name] = False
        return _loaded[name]

    raise AttributeError(f"module 'platform.core.memory' has no attribute '{name}'")


def __dir__():
    """Return all available names for tab completion."""
    return list(__all__)


# Factory functions for common use cases
def create_memory_gateway(
    agent_id: str = "default",
    enable_letta: bool = True,
    enable_graphiti: bool = False,
    **kwargs
) -> "UnifiedMemoryGateway":
    """
    Create a configured UnifiedMemoryGateway.

    Args:
        agent_id: Agent identifier
        enable_letta: Enable Letta memory backend
        enable_graphiti: Enable Graphiti temporal graph
        **kwargs: Additional configuration

    Returns:
        Configured UnifiedMemoryGateway instance
    """
    gateway_cls = __getattr__("UnifiedMemoryGateway")
    return gateway_cls(agent_id=agent_id, **kwargs)


def create_tier_manager(
    max_tiers: int = 3,
    enable_compression: bool = True,
    **kwargs
) -> "MemoryTierManager":
    """
    Create a configured MemoryTierManager.

    Args:
        max_tiers: Number of memory tiers
        enable_compression: Enable memory compression
        **kwargs: Additional configuration

    Returns:
        Configured MemoryTierManager instance
    """
    manager_cls = __getattr__("MemoryTierManager")
    return manager_cls(**kwargs)


__all__ = [
    # Availability flags
    "MEMORY_TIER_AVAILABLE",
    "UNIFIED_GATEWAY_AVAILABLE",
    "ADVANCED_MEMORY_AVAILABLE",

    # V36 CANONICAL - Base types (from backends.base)
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
    "TierBackend",
    "MemoryBackend",

    # V36 CANONICAL - Backends
    "InMemoryTierBackend",
    "LettaTierBackend",
    "LettaMemoryBackend",

    # Primary system classes
    "MemoryTierManager",
    "MemoryPressureLevel",
    "SleepTimeAgent",
    "MemoryTierIntegration",

    # Unified Gateway
    "UnifiedMemoryGateway",
    "LettaArchivesBackend",
    "ClaudeMemBackend",
    "EpisodicMemoryBackend",
    "GraphMemoryBackend",
    "StaticMemoryBackend",

    # Advanced Memory
    "AdvancedMemorySystem",
    "MemoryConsolidator",
    "MemoryMetrics",

    # Cross-session
    "CrossSessionMemory",

    # V36 SQLite persistent backend (REAL cross-session memory)
    "SQLiteTierBackend",
    "get_sqlite_backend",

    # V36 Memory hooks for session lifecycle
    "session_start_hook",
    "session_end_hook",
    "remember_decision",
    "remember_learning",
    "remember_fact",
    "recall",
    "get_context",

    # V36 Memory quality tracking
    "MemoryQualityTracker",
    "MemoryQualityMetrics",
    "ConflictReport",
    "RetrievalMetrics",
    "ConsolidationRecommendation",
    "QualityReport",
    "QualityConfig",
    "ConflictType",
    "ConsolidationAction",
    "get_quality_tracker",
    "analyze_memory_quality",
    "get_stale_memories",
    "detect_memory_conflicts",
    "get_memory_quality_report",

    # V36 Memory compression for long-term storage
    "MemoryCompressor",
    "CompressionStrategy",
    "CompressionTrigger",
    "CompressionConfig",
    "CompressionResult",
    "CompressionMetrics",
    "CompressedMemory",
    "CompressionStrategyBase",
    "ExtractiveStrategy",
    "AbstractiveStrategy",
    "HierarchicalStrategy",
    "ClusteringStrategy",
    "BackgroundCompressor",
    "get_compressor",
    "get_background_compressor",
    "compress_old_memories",
    "get_compression_stats",
    "decompress_memory",

    # V40 Forgetting curve and memory strength decay
    "ForgettingCurve",
    "MemoryStrength",
    "DecayCategory",
    "ForgettingConsolidationReport",
    "MemoryConsolidationTask",
    "apply_forgetting_to_entry",
    "reinforce_memory",
    "get_memory_strength",
    "apply_forgetting_migration",
    "DEFAULT_DECAY_RATES",
    "REINFORCEMENT_MULTIPLIERS",
    "STRENGTH_ARCHIVE_THRESHOLD",
    "STRENGTH_DELETE_THRESHOLD",

    # V40 Bi-temporal memory model (Zep research)
    "BiTemporalMemory",
    "TemporalMemoryEntry",
    "TemporalSearchResult",
    "TemporalAggregation",
    "create_bitemporal_memory",
    "get_bitemporal_memory",
    "reset_bitemporal_memory",
    "BITEMPORAL_SCHEMA_VERSION",

    # V40 Procedural memory for learned behaviors
    "ProceduralMemory",
    "Procedure",
    "ProcedureStep",
    "ProcedureMatch",
    "ExecutionResult",
    "ProcedureStatus",
    "StepType",
    "ExecutionOutcome",
    "get_procedural_memory",
    "learn_procedure",
    "recall_procedure",
    "execute_procedure",
    "MIN_CONFIDENCE_FOR_AUTO_EXECUTE",

    # V41 Unified Memory Interface (single interface to all memory systems)
    "UnifiedMemory",
    "MemoryType",
    "RoutingDecision",
    "SearchStrategy",
    "LifecycleState",
    "UnifiedSearchResult",
    "RoutingResult",
    "MaintenanceReport",
    "UnifiedStatistics",
    "ContentClassifier",
    "RRFFusion",
    "create_unified_memory",
    "get_unified_memory",
    "reset_unified_memory",

    # V42 Memory Compaction for long-running sessions
    "MemoryCompactor",
    "CompactionStrategy",
    "CompactionPriority",
    "MergeStrategy",
    "CompactionConfig",
    "CompactionCandidate",
    "MergeGroup",
    "CompactionReport",
    "CompactionMetrics",
    "CompactionScheduler",
    "get_memory_compactor",
    "get_compaction_scheduler",
    "compact_session_memories",
    "get_compaction_candidates",
    "get_compaction_status",

    # Legacy compatibility (DEPRECATED)
    "MemorySystem",
    "CoreMemory",
    "ArchivalMemory",
    "MemoryBlock",
    "TemporalGraph",

    # Factory functions
    "create_memory_gateway",
    "create_tier_manager",
]
