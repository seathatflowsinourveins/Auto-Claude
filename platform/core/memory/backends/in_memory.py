"""
In-Memory Backend - V36 Architecture

Simple in-memory storage for testing and fast operations.
Does not persist across restarts.

Usage:
    from core.memory.backends.in_memory import InMemoryTierBackend

    backend = InMemoryTierBackend()
    await backend.put("key1", entry)
    result = await backend.get("key1")
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .base import MemoryEntry, TierBackend


class InMemoryTierBackend(TierBackend[MemoryEntry]):
    """In-memory storage for main context and fast operations.

    V36 Consolidated: Unified implementation from memory_tiers.py.
    Thread-safe for single-process usage.
    """

    def __init__(self) -> None:
        self._storage: Dict[str, MemoryEntry] = {}

    async def get(self, key: str) -> Optional[MemoryEntry]:
        """Get entry by key, updating access metadata."""
        entry = self._storage.get(key)
        if entry:
            entry.touch()
        return entry

    async def put(self, key: str, value: MemoryEntry) -> None:
        """Store entry."""
        self._storage[key] = value

    async def delete(self, key: str) -> bool:
        """Delete entry."""
        if key in self._storage:
            del self._storage[key]
            return True
        return False

    async def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Simple substring search for in-memory.

        For semantic search, use backends with embedding support.
        """
        query_lower = query.lower()
        results = [
            entry for entry in self._storage.values()
            if query_lower in entry.content.lower()
        ]
        # Sort by access count (most accessed first)
        results.sort(key=lambda e: e.access_count, reverse=True)
        return results[:limit]

    async def list_all(self) -> List[MemoryEntry]:
        """List all entries."""
        return list(self._storage.values())

    async def count(self) -> int:
        """Get entry count."""
        return len(self._storage)

    def clear(self) -> None:
        """Clear all entries (for testing)."""
        self._storage.clear()


__all__ = ["InMemoryTierBackend"]
