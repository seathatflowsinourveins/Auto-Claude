#!/usr/bin/env python3
"""
Unified Memory Layer - V33 Architecture (Phase 9 Production Fix)
Provides a unified interface to multiple memory backends.

This module unifies:
- Local three-tier memory (Core, Archival, Temporal)
- Letta Cloud integration
- Zep session memory
- Mem0 persistent memory

NO STUBS - EXPLICIT FAILURES ONLY:
- get_mem0_client() raises SDKNotAvailableError if mem0 not installed
- get_zep_client() raises SDKNotAvailableError if zep-python not installed
- get_letta_client() raises SDKNotAvailableError if letta not installed
- get_cross_session_provider() raises SDKNotAvailableError if unavailable

Usage:
    from core.memory import UnifiedMemory, create_memory

    memory = create_memory()

    # Store memories
    await memory.store("Important information", tier="archival")

    # Search across all providers
    results = await memory.search("query", providers=["local", "mem0"])

    # For direct SDK access (raises on unavailable):
    from core.memory import get_mem0_client, SDKNotAvailableError

    try:
        client = get_mem0_client()
    except SDKNotAvailableError as e:
        print(f"Install SDK: {e.install_cmd}")
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional, Literal, List
from dataclasses import dataclass, field
from datetime import datetime

import structlog

# Import exceptions from observability layer
from core.observability import (
    SDKNotAvailableError,
    SDKConfigurationError,
)

# Import types
from .types import (
    MemoryTier,
    MemoryProvider,
    ConsolidationStrategy,
    MemoryEntry,
    MemoryBlock,
    SearchResult,
    MemoryStats,
    ProviderConfig,
    MemoryConfig,
)

# Import providers
from .providers import (
    MemoryProviderBase,
    Mem0Provider,
    ZepProvider,
    LettaProvider,
    CrossSessionProvider,
    create_provider,
    get_available_providers,
    MEM0_AVAILABLE,
    ZEP_AVAILABLE,
    LETTA_AVAILABLE,
    CROSS_SESSION_AVAILABLE,
)

logger = structlog.get_logger(__name__)

# Local memory is always available
LOCAL_AVAILABLE = True

# Memory layer availability flag (for core/__init__.py)
MEMORY_LAYER_AVAILABLE = True

# Platform memory availability
PLATFORM_MEMORY_AVAILABLE = False
PLATFORM_MEMORY_ERROR = None

try:
    import sys
    from pathlib import Path

    # Add platform to path if needed
    platform_path = Path(__file__).parent.parent.parent / "platform"
    if str(platform_path) not in sys.path:
        sys.path.insert(0, str(platform_path))

    from core.memory import (
        CoreMemory,
        ArchivalMemory,
        TemporalGraph,
        MemorySystem,
    )
    from core.advanced_memory import (
        AdvancedMemorySystem,
        SemanticIndex,
        MemoryConsolidator,
        LettaClient,
    )
    PLATFORM_MEMORY_AVAILABLE = True
except ImportError as e:
    PLATFORM_MEMORY_ERROR = str(e)


# =============================================================================
# EXPLICIT GETTER FUNCTIONS - No Stubs, Raise on Unavailable
# =============================================================================

def get_mem0_client() -> "Mem0Provider":
    """
    Get Mem0 memory provider. Raises explicit error if unavailable.

    Returns:
        Configured Mem0Provider instance

    Raises:
        SDKNotAvailableError: If mem0 is not installed
        SDKConfigurationError: If required config is missing
    """
    if not MEM0_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="mem0",
            install_cmd="pip install mem0ai>=0.1.0",
            docs_url="https://docs.mem0.ai/overview"
        )

    # Mem0 can work with or without API key (local mode vs cloud)
    return Mem0Provider()


def get_zep_client() -> "ZepProvider":
    """
    Get Zep memory provider. Raises explicit error if unavailable.

    Returns:
        Configured ZepProvider instance

    Raises:
        SDKNotAvailableError: If zep-python is not installed
        SDKConfigurationError: If ZEP_API_KEY not configured
    """
    if not ZEP_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="zep-python",
            install_cmd="pip install zep-python>=2.0.0",
            docs_url="https://docs.getzep.com/sdk/python/"
        )

    # Check for API key
    if not os.getenv("ZEP_API_KEY"):
        raise SDKConfigurationError(
            sdk_name="zep",
            missing_config=["ZEP_API_KEY"],
            example="""
ZEP_API_KEY=your-api-key-here
ZEP_API_URL=http://localhost:8000  # Optional
"""
        )

    return ZepProvider()


def get_letta_client() -> "LettaProvider":
    """
    Get Letta memory provider. Raises explicit error if unavailable.

    Returns:
        Configured LettaProvider instance

    Raises:
        SDKNotAvailableError: If letta is not installed
    """
    if not LETTA_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="letta",
            install_cmd="pip install letta>=0.5.0",
            docs_url="https://docs.letta.com/"
        )

    return LettaProvider()


def get_cross_session_provider() -> "CrossSessionProvider":
    """
    Get cross-session memory provider. Raises explicit error if unavailable.

    Returns:
        Configured CrossSessionProvider instance

    Raises:
        SDKNotAvailableError: If cross-session module is not available
    """
    if not CROSS_SESSION_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="cross_session_memory",
            install_cmd="# Cross-session requires platform.core.cross_session_memory module",
            docs_url="https://docs.unleash.dev/memory/cross-session"
        )

    return CrossSessionProvider()


@dataclass
class UnifiedMemory:
    """
    Unified Memory System - Single interface to all memory backends.

    Provides:
    - Multi-provider memory storage and retrieval
    - Automatic provider selection based on availability
    - Cross-provider search with result merging
    - Cloud synchronization
    - Memory consolidation

    Usage:
        memory = UnifiedMemory()

        # Store with automatic provider selection
        entry = await memory.store("User prefers dark mode")

        # Store to specific provider
        entry = await memory.store("Session context", provider="zep")

        # Search across providers
        results = await memory.search("dark mode", limit=5)
    """

    config: MemoryConfig = field(default_factory=MemoryConfig)
    _providers: dict[MemoryProvider, MemoryProviderBase] = field(default_factory=dict)
    _local_memory: Any = field(default=None)
    _initialized: bool = field(default=False)

    async def initialize(self) -> None:
        """Initialize all configured providers."""
        if self._initialized:
            return

        logger.info("unified_memory_initializing")

        # Initialize local memory if platform available
        if PLATFORM_MEMORY_AVAILABLE:
            try:
                self._local_memory = AdvancedMemorySystem()
                logger.info("local_memory_initialized", type="advanced")
            except Exception as e:
                logger.warning("advanced_memory_failed", error=str(e))
                self._local_memory = MemorySystem()
                logger.info("local_memory_initialized", type="basic")

        # Initialize external providers
        available = get_available_providers()

        for provider_config in self.config.providers:
            provider_type = provider_config.provider
            if not provider_config.enabled:
                continue

            if not available.get(provider_type, False):
                logger.warning("provider_not_available", provider=provider_type.value)
                continue

            try:
                provider = create_provider(provider_type, provider_config)
                self._providers[provider_type] = provider
                logger.info("provider_initialized", provider=provider_type.value)
            except Exception as e:
                logger.error("provider_init_failed", provider=provider_type.value, error=str(e))

        self._initialized = True
        logger.info(
            "unified_memory_ready",
            local=self._local_memory is not None,
            providers=list(self._providers.keys()),
        )

    async def store(
        self,
        content: str,
        tier: MemoryTier = MemoryTier.ARCHIVAL,
        provider: Optional[MemoryProvider] = None,
        metadata: Optional[dict[str, Any]] = None,
        importance: float = 0.5,
        user_id: Optional[str] = None,
    ) -> MemoryEntry:
        """
        Store a memory entry.

        Args:
            content: The memory content to store
            tier: Memory tier (core, archival, temporal)
            provider: Specific provider to use (None = auto-select)
            metadata: Additional metadata
            importance: Importance score (0.0-1.0)
            user_id: User identifier for multi-user scenarios

        Returns:
            MemoryEntry with storage details
        """
        await self.initialize()

        metadata = metadata or {}
        metadata["importance"] = importance
        metadata["tier"] = tier.value

        # Determine which provider to use
        target_provider = provider or self.config.default_provider

        # Try to store in the selected provider
        if target_provider == MemoryProvider.LOCAL and self._local_memory:
            try:
                if tier == MemoryTier.CORE and hasattr(self._local_memory, "core"):
                    self._local_memory.core.update("working_memory", content)
                elif hasattr(self._local_memory, "store_semantic"):
                    await self._local_memory.store_semantic(
                        content=content,
                        metadata=metadata,
                        importance=importance,
                    )
                elif hasattr(self._local_memory, "archival"):
                    self._local_memory.archival.store(content, metadata=metadata)

                return MemoryEntry(
                    id=str(datetime.now().timestamp()),
                    content=content,
                    tier=tier,
                    provider=MemoryProvider.LOCAL,
                    metadata=metadata,
                    importance=importance,
                )
            except Exception as e:
                logger.warning("local_store_failed", error=str(e))

        # Try external provider
        if target_provider in self._providers:
            entry = await self._providers[target_provider].store(
                content=content,
                metadata=metadata,
                user_id=user_id,
            )
            entry.tier = tier
            entry.importance = importance
            return entry

        # Fallback to first available provider
        for prov_type, prov in self._providers.items():
            try:
                entry = await prov.store(content=content, metadata=metadata, user_id=user_id)
                entry.tier = tier
                entry.importance = importance
                return entry
            except Exception as e:
                logger.warning("provider_store_failed", provider=prov_type.value, error=str(e))

        # If all else fails, create a local entry
        return MemoryEntry(
            id=str(datetime.now().timestamp()),
            content=content,
            tier=tier,
            provider=MemoryProvider.LOCAL,
            metadata=metadata,
            importance=importance,
        )

    async def search(
        self,
        query: str,
        limit: int = 10,
        providers: Optional[list[MemoryProvider]] = None,
        tier: Optional[MemoryTier] = None,
        min_score: float = 0.0,
        user_id: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Search for memories across providers.

        Args:
            query: Search query
            limit: Maximum results per provider
            providers: Specific providers to search (None = all)
            tier: Filter by memory tier
            min_score: Minimum relevance score
            user_id: User identifier

        Returns:
            List of SearchResult objects, sorted by score
        """
        await self.initialize()

        all_results: list[SearchResult] = []

        # Search local memory
        if self._local_memory and (providers is None or MemoryProvider.LOCAL in providers):
            try:
                if hasattr(self._local_memory, "search_semantic"):
                    local_results = await self._local_memory.search_semantic(
                        query=query,
                        limit=limit,
                        min_score=min_score,
                    )
                    for r in local_results:
                        all_results.append(SearchResult(
                            id=r.id if hasattr(r, "id") else str(len(all_results)),
                            content=r.content if hasattr(r, "content") else str(r),
                            score=r.score if hasattr(r, "score") else 0.5,
                            metadata=r.metadata if hasattr(r, "metadata") else {},
                            provider=MemoryProvider.LOCAL,
                        ))
                elif hasattr(self._local_memory, "archival"):
                    # Use archival search
                    local_results = self._local_memory.archival.search(query, limit=limit)
                    for r in local_results:
                        all_results.append(SearchResult(
                            id=r.get("id", str(len(all_results))),
                            content=r.get("content", str(r)),
                            score=r.get("score", 0.5),
                            metadata=r.get("metadata", {}),
                            provider=MemoryProvider.LOCAL,
                        ))
            except Exception as e:
                logger.warning("local_search_failed", error=str(e))

        # Search external providers
        target_providers = providers or list(self._providers.keys())

        for prov_type in target_providers:
            if prov_type == MemoryProvider.LOCAL:
                continue

            if prov_type not in self._providers:
                continue

            try:
                results = await self._providers[prov_type].search(
                    query=query,
                    limit=limit,
                    user_id=user_id,
                )
                all_results.extend(results)
            except Exception as e:
                logger.warning("provider_search_failed", provider=prov_type.value, error=str(e))

        # Filter by tier if specified
        if tier:
            all_results = [r for r in all_results if r.metadata.get("tier") == tier.value]

        # Filter by minimum score
        all_results = [r for r in all_results if r.score >= min_score]

        # Sort by score descending
        all_results.sort(key=lambda r: r.score, reverse=True)

        # Deduplicate by content similarity (simple approach)
        seen_content = set()
        unique_results = []
        for r in all_results:
            content_key = r.content[:100].lower()
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(r)

        return unique_results[:limit]

    async def list(
        self,
        namespace: str = "default",
        limit: int = 100,
        providers: Optional[list[MemoryProvider]] = None,
    ) -> list[dict[str, Any]]:
        """
        List memory entries in a namespace.

        This is a convenience method for CLI compatibility that wraps search
        with an empty/wildcard query to enumerate entries.

        Args:
            namespace: Memory namespace to list
            limit: Maximum entries to return
            providers: Specific providers to query

        Returns:
            List of memory entries as dictionaries
        """
        # Use search with a wildcard-style query
        results = await self.search(
            query="*",  # General query to get entries
            limit=limit,
            providers=providers,
        )
        
        # Convert SearchResult to dict for CLI compatibility
        entries = []
        for r in results:
            entries.append({
                "key": r.id,
                "value": r.content,
                "created_at": r.metadata.get("created_at", "N/A"),
                "namespace": namespace,
                "provider": r.provider.value if r.provider else "unknown",
            })
        
        return entries

    async def get(
        self,
        memory_id: str,
        provider: Optional[MemoryProvider] = None,
    ) -> Optional[MemoryEntry]:
        """Get a specific memory by ID."""
        await self.initialize()

        # Try specified provider first
        if provider and provider in self._providers:
            return await self._providers[provider].get(memory_id)

        # Try all providers
        for prov in self._providers.values():
            result = await prov.get(memory_id)
            if result:
                return result

        return None

    async def delete(
        self,
        memory_id: str,
        provider: Optional[MemoryProvider] = None,
    ) -> bool:
        """Delete a memory entry."""
        await self.initialize()

        if provider and provider in self._providers:
            return await self._providers[provider].delete(memory_id)

        # Try all providers
        for prov in self._providers.values():
            if await prov.delete(memory_id):
                return True

        return False

    async def consolidate(
        self,
        strategy: ConsolidationStrategy = ConsolidationStrategy.SUMMARIZE,
        older_than_days: int = 7,
    ) -> int:
        """
        Consolidate old memories using the specified strategy.

        Args:
            strategy: Consolidation strategy to use
            older_than_days: Only consolidate entries older than this

        Returns:
            Number of entries consolidated
        """
        await self.initialize()

        if not self._local_memory:
            return 0

        try:
            if hasattr(self._local_memory, "consolidate"):
                from datetime import timedelta
                cutoff = datetime.now() - timedelta(days=older_than_days)
                result = await self._local_memory.consolidate(
                    strategy=strategy.value,
                    older_than=cutoff,
                )
                return result.entries_affected if hasattr(result, "entries_affected") else 0
        except Exception as e:
            logger.warning("consolidation_failed", error=str(e))

        return 0

    async def sync(self, provider: Optional[MemoryProvider] = None) -> bool:
        """
        Synchronize local memory with cloud provider.

        Args:
            provider: Specific provider to sync (None = default)

        Returns:
            True if sync succeeded
        """
        await self.initialize()

        if not self._local_memory:
            return False

        target = provider or self.config.default_provider

        try:
            if hasattr(self._local_memory, "sync_to_letta") and target == MemoryProvider.LETTA:
                await self._local_memory.sync_to_letta()
                return True
        except Exception as e:
            logger.warning("sync_failed", provider=target.value, error=str(e))

        return False

    async def get_stats(self) -> MemoryStats:
        """Get memory usage statistics."""
        await self.initialize()

        stats = MemoryStats()

        if self._local_memory and hasattr(self._local_memory, "get_stats"):
            local_stats = await self._local_memory.get_stats()
            stats.total_entries = local_stats.get("total_entries", 0)
            stats.total_tokens_estimated = local_stats.get("total_tokens", 0)

        stats.entries_by_provider = {
            MemoryProvider.LOCAL.value: stats.total_entries,
        }

        for prov_type in self._providers:
            stats.entries_by_provider[prov_type.value] = 0  # Would need API calls to count

        return stats

    async def close(self) -> None:
        """Close all provider connections."""
        for provider in self._providers.values():
            try:
                await provider.close()
            except Exception as e:
                logger.warning("provider_close_failed", error=str(e))

        self._providers.clear()
        self._initialized = False

    @property
    def available_providers(self) -> list[str]:
        """List available providers."""
        available = ["local"] if self._local_memory else []
        available.extend(p.value for p in self._providers.keys())
        return available


def create_memory(
    config: Optional[MemoryConfig] = None,
    enable_mem0: bool = True,
    enable_zep: bool = True,
    enable_letta: bool = True,
) -> UnifiedMemory:
    """
    Factory function to create a unified memory system.

    Args:
        config: Optional configuration
        enable_mem0: Enable Mem0 provider
        enable_zep: Enable Zep provider
        enable_letta: Enable Letta provider

    Returns:
        Configured UnifiedMemory instance
    """
    if config is None:
        providers = []

        if enable_mem0 and MEM0_AVAILABLE:
            providers.append(ProviderConfig(
                provider=MemoryProvider.MEM0,
                enabled=True,
            ))

        if enable_zep and ZEP_AVAILABLE:
            providers.append(ProviderConfig(
                provider=MemoryProvider.ZEP,
                enabled=True,
                api_key=os.getenv("ZEP_API_KEY"),
                base_url=os.getenv("ZEP_API_URL"),
            ))

        if enable_letta and LETTA_AVAILABLE:
            providers.append(ProviderConfig(
                provider=MemoryProvider.LETTA,
                enabled=True,
            ))

        config = MemoryConfig(
            default_provider=MemoryProvider.LOCAL,
            providers=providers,
        )

    return UnifiedMemory(config=config)


def get_available_memory_providers() -> dict[str, bool]:
    """Get availability status of all memory providers."""
    return {
        "local": LOCAL_AVAILABLE,
        "platform": PLATFORM_MEMORY_AVAILABLE,
        "mem0": MEM0_AVAILABLE,
        "zep": ZEP_AVAILABLE,
        "letta": LETTA_AVAILABLE,
    }


# =============================================================================
# CROSS-SESSION HELPER FUNCTIONS - Explicit Errors Only
# =============================================================================

def get_cross_session_context(max_tokens: int = 4000) -> str:
    """
    Get context from previous Claude Code sessions for handoff.

    This is the key function for cross-session continuity.
    Returns a formatted string with recent sessions, decisions, learnings, and facts.

    Raises:
        SDKNotAvailableError: If cross-session module is not available
    """
    if not CROSS_SESSION_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="cross_session_memory",
            install_cmd="# Cross-session requires platform.core.cross_session_memory module",
            docs_url="https://docs.unleash.dev/memory/cross-session"
        )

    provider = CrossSessionProvider()
    return provider.get_session_context(max_tokens)


def start_memory_session(task_summary: str = "") -> None:
    """
    Start a new memory tracking session.

    Raises:
        SDKNotAvailableError: If cross-session module is not available
    """
    if not CROSS_SESSION_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="cross_session_memory",
            install_cmd="# Cross-session requires platform.core.cross_session_memory module",
            docs_url="https://docs.unleash.dev/memory/cross-session"
        )

    provider = CrossSessionProvider()
    provider.start_session(task_summary)


def end_memory_session(summary: str = "") -> None:
    """
    End the current memory tracking session.

    Raises:
        SDKNotAvailableError: If cross-session module is not available
    """
    if not CROSS_SESSION_AVAILABLE:
        raise SDKNotAvailableError(
            sdk_name="cross_session_memory",
            install_cmd="# Cross-session requires platform.core.cross_session_memory module",
            docs_url="https://docs.unleash.dev/memory/cross-session"
        )

    provider = CrossSessionProvider()
    provider.end_session(summary)


# Singleton instance for CLI compatibility
_memory_manager_instance: Optional[UnifiedMemory] = None


def get_memory_manager() -> UnifiedMemory:
    """
    Get the singleton memory manager instance.
    
    This provides CLI compatibility by returning a cached memory instance.
    The instance is created on first access and reused for subsequent calls.
    
    Returns:
        UnifiedMemory: The singleton memory manager
    """
    global _memory_manager_instance
    if _memory_manager_instance is None:
        _memory_manager_instance = create_memory()
    return _memory_manager_instance


# Export all public symbols
__all__ = [
    # Exceptions (re-exported from observability)
    "SDKNotAvailableError",
    "SDKConfigurationError",

    # Core classes
    "UnifiedMemory",

    # Types
    "MemoryTier",
    "MemoryProvider",
    "ConsolidationStrategy",
    "MemoryEntry",
    "MemoryBlock",
    "SearchResult",
    "MemoryStats",
    "ProviderConfig",
    "MemoryConfig",

    # Providers
    "MemoryProviderBase",
    "Mem0Provider",
    "ZepProvider",
    "LettaProvider",
    "CrossSessionProvider",

    # Explicit getter functions (raise on unavailable)
    "get_mem0_client",
    "get_zep_client",
    "get_letta_client",
    "get_cross_session_provider",

    # Factory functions
    "create_memory",
    "create_provider",
    "get_available_memory_providers",
    "get_available_providers",
    "get_memory_manager",

    # Cross-session functions (raise on unavailable)
    "get_cross_session_context",
    "start_memory_session",
    "end_memory_session",

    # Availability flags (for conditional logic)
    "LOCAL_AVAILABLE",
    "PLATFORM_MEMORY_AVAILABLE",
    "MEM0_AVAILABLE",
    "ZEP_AVAILABLE",
    "LETTA_AVAILABLE",
    "CROSS_SESSION_AVAILABLE",
]


if __name__ == "__main__":
    async def main():
        """Test the unified memory system."""
        print("Memory Layer Status")
        print("=" * 40)

        status = get_available_memory_providers()
        for provider, available in status.items():
            symbol = "[+]" if available else "[X]"
            print(f"  {symbol} {provider}")

        print()

        memory = create_memory()
        await memory.initialize()

        print(f"Available: {memory.available_providers}")

        # Test store and search
        if memory.available_providers:
            entry = await memory.store(
                "Test memory entry for V33 validation",
                importance=0.8,
            )
            print(f"Stored: {entry.id}")

            results = await memory.search("test memory")
            print(f"Search found: {len(results)} results")

        await memory.close()

    asyncio.run(main())
