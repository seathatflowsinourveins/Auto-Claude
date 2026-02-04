#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pydantic>=2.0.0",
# ]
# ///
"""
Memory System - Letta + Graphiti Pattern Implementation

Three-tier memory architecture:
1. Core Memory: Always in-context memory blocks
2. Archival Memory: Vector-retrieved external storage
3. Temporal Graph: Knowledge graph with temporal awareness

Based on:
- Letta docs: https://docs.letta.com/guides/agents/memory
- Graphiti: Temporal knowledge graphs for AI agents

Usage:
    from memory import MemorySystem, MemoryBlock

    memory = MemorySystem(agent_id="agent-001")
    memory.core.update("user_context", "User prefers dark mode")
    memory.archival.store("conversation-123", embeddings, metadata)
    memory.temporal.add_fact(subject, predicate, object)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

# Structured logging
try:
    from .logging_config import get_logger, generate_correlation_id
    _logger = get_logger("memory")
except ImportError:
    import logging
    _logger = logging.getLogger(__name__)
    generate_correlation_id = lambda: "corr-fallback"


# =============================================================================
# Core Memory (Tier 1) - Always In-Context
# =============================================================================

class MemoryBlock(BaseModel):
    """
    Structured section of agent's context window.

    Memory blocks are Letta's core abstraction. They persist across all
    interactions and are always visible (no retrieval needed).

    Example:
        >>> block = MemoryBlock(label="user_context", max_tokens=2000)
        >>> block.update("User prefers concise responses")
        >>> print(block.render())
    """

    label: str = Field(..., description="Descriptive label (e.g., 'user_context')")
    content: str = Field(default="", description="Block content")
    max_tokens: int = Field(default=2000, description="Maximum token budget")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def update(self, new_content: str) -> bool:
        """Update block content. Returns True if successful."""
        estimated_tokens = len(new_content) // 4  # Rough estimate
        if estimated_tokens <= self.max_tokens:
            self.content = new_content
            self.updated_at = datetime.now(timezone.utc)
            return True
        return False

    def append(self, addition: str, separator: str = "\n") -> bool:
        """Append to block content."""
        combined = f"{self.content}{separator}{addition}" if self.content else addition
        return self.update(combined)

    def render(self) -> str:
        """Render block for context injection."""
        return f"<{self.label}>\n{self.content}\n</{self.label}>"

    def estimated_tokens(self) -> int:
        """Estimate current token count."""
        return len(self.content) // 4


class CoreMemory:
    """
    Core memory manager - always in-context memory blocks.

    Standard blocks:
    - system_persona: Agent identity and behavior
    - user_context: User preferences and state
    - task_state: Current task progress
    - working_memory: Recent observations
    """

    STANDARD_BLOCKS = {
        "system_persona": 3000,
        "user_context": 2000,
        "task_state": 2000,
        "working_memory": 5000,
    }

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.blocks: Dict[str, MemoryBlock] = {}

        # Initialize standard blocks
        for label, max_tokens in self.STANDARD_BLOCKS.items():
            self.blocks[label] = MemoryBlock(label=label, max_tokens=max_tokens)

    def get(self, label: str) -> Optional[str]:
        """Get block content by label."""
        block = self.blocks.get(label)
        return block.content if block else None

    def update(self, label: str, content: str) -> bool:
        """Update or create a block."""
        if label in self.blocks:
            return self.blocks[label].update(content)
        else:
            # Create new custom block
            self.blocks[label] = MemoryBlock(label=label, content=content)
            return True

    def append(self, label: str, content: str) -> bool:
        """Append to a block."""
        if label not in self.blocks:
            return self.update(label, content)
        return self.blocks[label].append(content)

    def render_all(self) -> str:
        """Render all blocks for context injection."""
        rendered = []
        for block in self.blocks.values():
            if block.content:  # Skip empty blocks
                rendered.append(block.render())
        return "\n\n".join(rendered)

    def total_tokens(self) -> int:
        """Estimate total token usage."""
        return sum(b.estimated_tokens() for b in self.blocks.values())

    def export(self) -> Dict[str, str]:
        """Export all blocks for checkpointing."""
        return {label: block.content for label, block in self.blocks.items()}

    def import_state(self, state: Dict[str, str]) -> None:
        """Import blocks from checkpoint."""
        for label, content in state.items():
            self.update(label, content)


# =============================================================================
# Archival Memory (Tier 2) - Vector-Retrieved
# =============================================================================

@dataclass
class ArchivalEntry:
    """Entry in archival memory."""

    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


class ArchivalMemoryBackend(ABC):
    """Abstract backend for archival memory."""

    @abstractmethod
    async def store(self, entry: ArchivalEntry) -> bool:
        """Store an entry."""
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ArchivalEntry]:
        """Search entries by semantic similarity."""
        pass

    @abstractmethod
    async def get(self, entry_id: str) -> Optional[ArchivalEntry]:
        """Get entry by ID."""
        pass

    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """Delete entry by ID."""
        pass


class LocalArchivalMemory(ArchivalMemoryBackend):
    """
    Local file-based archival memory.

    For development/testing. Production should use Qdrant/Pinecone.
    """

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.entries: Dict[str, ArchivalEntry] = {}
        self._load()

    def _load(self) -> None:
        """Load entries from disk."""
        index_file = self.storage_path / "index.json"
        if index_file.exists():
            with open(index_file) as f:
                data = json.load(f)
                for entry_data in data.get("entries", []):
                    entry = ArchivalEntry(
                        id=entry_data["id"],
                        content=entry_data["content"],
                        metadata=entry_data.get("metadata", {}),
                        created_at=datetime.fromisoformat(entry_data["created_at"])
                    )
                    self.entries[entry.id] = entry

    def _save(self) -> None:
        """Save entries to disk."""
        index_file = self.storage_path / "index.json"
        data = {
            "entries": [e.to_dict() for e in self.entries.values()]
        }
        with open(index_file, "w") as f:
            json.dump(data, f, indent=2)

    async def store(self, entry: ArchivalEntry) -> bool:
        """Store an entry."""
        self.entries[entry.id] = entry
        self._save()
        return True

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ArchivalEntry]:
        """Search entries (basic keyword matching for local backend)."""
        query_lower = query.lower()
        results = []

        for entry in self.entries.values():
            # Basic keyword matching
            if query_lower in entry.content.lower():
                # Apply filters
                if filters:
                    match = all(
                        entry.metadata.get(k) == v
                        for k, v in filters.items()
                    )
                    if not match:
                        continue
                results.append(entry)

        # Sort by relevance (simple: more keyword occurrences = higher)
        results.sort(
            key=lambda e: e.content.lower().count(query_lower),
            reverse=True
        )
        return results[:limit]

    async def get(self, entry_id: str) -> Optional[ArchivalEntry]:
        """Get entry by ID."""
        return self.entries.get(entry_id)

    async def delete(self, entry_id: str) -> bool:
        """Delete entry by ID."""
        if entry_id in self.entries:
            del self.entries[entry_id]
            self._save()
            return True
        return False


class ArchivalMemory:
    """
    Archival memory manager - vector-retrieved external storage.

    Categories:
    - episodic: Past conversations and interactions
    - semantic: Domain knowledge and facts
    - procedural: How-to knowledge and patterns
    """

    def __init__(self, agent_id: str, backend: ArchivalMemoryBackend):
        self.agent_id = agent_id
        self.backend = backend

    async def store(
        self,
        content: str,
        category: str = "episodic",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store content in archival memory."""
        entry_id = str(uuid4())
        entry = ArchivalEntry(
            id=entry_id,
            content=content,
            metadata={
                "agent_id": self.agent_id,
                "category": category,
                **(metadata or {})
            }
        )
        await self.backend.store(entry)

        _logger.info(
            "Stored entry in archival memory",
            component="memory.archival",
            agent_id=self.agent_id,
            entry_id=entry_id,
            category=category,
            content_length=len(content),
        )
        return entry_id

    async def search(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10
    ) -> List[ArchivalEntry]:
        """Search archival memory."""
        filters = {"agent_id": self.agent_id}
        if category:
            filters["category"] = category

        _logger.debug(
            "Searching archival memory",
            component="memory.archival",
            agent_id=self.agent_id,
            query_length=len(query),
            category=category,
            limit=limit,
            sample_rate=0.2,
        )

        results = await self.backend.search(query, limit, filters)

        _logger.info(
            "Archival memory search completed",
            component="memory.archival",
            agent_id=self.agent_id,
            results_count=len(results),
            category=category,
            sample_rate=0.5,
        )
        return results

    async def recall(self, query: str, limit: int = 5) -> str:
        """Recall relevant memories as formatted string."""
        entries = await self.search(query, limit=limit)
        if not entries:
            _logger.debug(
                "No memories found for recall",
                component="memory.archival",
                agent_id=self.agent_id,
                query_length=len(query),
            )
            return "No relevant memories found."

        memories = []
        for i, entry in enumerate(entries, 1):
            memories.append(f"{i}. [{entry.metadata.get('category', 'unknown')}] {entry.content}")

        return "\n".join(memories)


# =============================================================================
# Temporal Knowledge Graph (Tier 3) - Graphiti Pattern
# =============================================================================

class TemporalFact(BaseModel):
    """
    A fact with temporal validity.

    Tracks not just what is true, but when it became true and when it stopped
    being true. This enables querying "what was true at time X".
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    subject: str = Field(..., description="Entity the fact is about")
    predicate: str = Field(..., description="Relationship type")
    object: str = Field(..., description="Related entity or value")
    valid_from: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_to: Optional[datetime] = Field(default=None, description="None = still valid")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = Field(default="observation", description="How we learned this")

    def is_current(self) -> bool:
        """Check if fact is currently valid."""
        now = datetime.now(timezone.utc)
        return self.valid_from <= now and (self.valid_to is None or self.valid_to > now)

    def invalidate(self, when: Optional[datetime] = None) -> None:
        """Mark fact as no longer valid."""
        self.valid_to = when or datetime.now(timezone.utc)

    def to_triple(self) -> str:
        """Return as (subject, predicate, object) string."""
        return f"({self.subject}, {self.predicate}, {self.object})"


class TemporalGraph:
    """
    Temporal knowledge graph - Graphiti pattern.

    Key features:
    - Track fact validity over time
    - Handle contradictions by invalidating old facts
    - Query state at any point in time
    - ~18% improvement on multi-session reasoning
    """

    def __init__(self, agent_id: str, storage_path: Path):
        self.agent_id = agent_id
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.facts: Dict[str, TemporalFact] = {}
        self._load()

    def _load(self) -> None:
        """Load facts from disk."""
        facts_file = self.storage_path / "temporal_facts.json"
        if facts_file.exists():
            with open(facts_file) as f:
                data = json.load(f)
                for fact_data in data.get("facts", []):
                    fact = TemporalFact(
                        id=fact_data["id"],
                        subject=fact_data["subject"],
                        predicate=fact_data["predicate"],
                        object=fact_data["object"],
                        valid_from=datetime.fromisoformat(fact_data["valid_from"]),
                        valid_to=datetime.fromisoformat(fact_data["valid_to"]) if fact_data.get("valid_to") else None,
                        confidence=fact_data.get("confidence", 1.0),
                        source=fact_data.get("source", "observation")
                    )
                    self.facts[fact.id] = fact

    def _save(self) -> None:
        """Save facts to disk."""
        facts_file = self.storage_path / "temporal_facts.json"
        data = {
            "agent_id": self.agent_id,
            "facts": [
                {
                    "id": f.id,
                    "subject": f.subject,
                    "predicate": f.predicate,
                    "object": f.object,
                    "valid_from": f.valid_from.isoformat(),
                    "valid_to": f.valid_to.isoformat() if f.valid_to else None,
                    "confidence": f.confidence,
                    "source": f.source
                }
                for f in self.facts.values()
            ]
        }
        with open(facts_file, "w") as f:
            json.dump(data, f, indent=2)

    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source: str = "observation"
    ) -> TemporalFact:
        """
        Add a fact, potentially invalidating contradictory facts.

        If we learn "user prefers dark mode" but previously knew "user prefers
        light mode", the old fact gets invalidated with valid_to set to now.
        """
        # Check for contradictions
        contradictions = self.find_facts(subject, predicate, current_only=True)
        now = datetime.now(timezone.utc)

        for old_fact in contradictions:
            if old_fact.object != obj:
                old_fact.invalidate(now)

        # Add new fact
        fact = TemporalFact(
            subject=subject,
            predicate=predicate,
            object=obj,
            valid_from=now,
            confidence=confidence,
            source=source
        )
        self.facts[fact.id] = fact
        self._save()
        return fact

    def find_facts(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        current_only: bool = False
    ) -> List[TemporalFact]:
        """Find facts matching criteria."""
        results = []
        for fact in self.facts.values():
            if subject and fact.subject != subject:
                continue
            if predicate and fact.predicate != predicate:
                continue
            if obj and fact.object != obj:
                continue
            if current_only and not fact.is_current():
                continue
            results.append(fact)
        return results

    def query_at_time(
        self,
        timestamp: datetime,
        subject: Optional[str] = None
    ) -> List[TemporalFact]:
        """Get all facts valid at a specific time."""
        results = []
        for fact in self.facts.values():
            if fact.valid_from <= timestamp:
                if fact.valid_to is None or fact.valid_to > timestamp:
                    if subject is None or fact.subject == subject:
                        results.append(fact)
        return results

    def get_entity_history(self, subject: str) -> List[TemporalFact]:
        """Get all facts about an entity, ordered by time."""
        facts = self.find_facts(subject=subject)
        return sorted(facts, key=lambda f: f.valid_from)

    def current_knowledge(self) -> str:
        """Render current knowledge for context injection."""
        current = [f for f in self.facts.values() if f.is_current()]
        if not current:
            return "No current knowledge in temporal graph."

        lines = ["Current Knowledge:"]
        for fact in current:
            lines.append(f"- {fact.to_triple()} (confidence: {fact.confidence:.2f})")

        return "\n".join(lines)


# =============================================================================
# Unified Memory System
# =============================================================================

class MemorySystem:
    """
    Unified three-tier memory system.

    Provides seamless access to:
    - Core memory (always in-context)
    - Archival memory (vector-retrieved)
    - Temporal graph (knowledge graph with time)
    """

    def __init__(
        self,
        agent_id: str,
        storage_base: Optional[Path] = None
    ):
        self.agent_id = agent_id

        _logger.info(
            "Initializing memory system",
            component="memory",
            agent_id=agent_id,
            storage_path=str(storage_base) if storage_base else "default",
        )

        if storage_base is None:
            storage_base = Path.home() / ".uap" / "memory" / agent_id

        storage_base.mkdir(parents=True, exist_ok=True)

        # Initialize tiers
        self.core = CoreMemory(agent_id)

        archival_backend = LocalArchivalMemory(storage_base / "archival")
        self.archival = ArchivalMemory(agent_id, archival_backend)

        self.temporal = TemporalGraph(agent_id, storage_base / "temporal")

        _logger.info(
            "Memory system initialized",
            component="memory",
            agent_id=agent_id,
            core_blocks=len(self.core.blocks),
            temporal_facts=len(self.temporal.facts),
        )

    def render_context(self, include_temporal: bool = True) -> str:
        """Render all memory for context injection."""
        _logger.debug(
            "Rendering memory context",
            component="memory",
            agent_id=self.agent_id,
            include_temporal=include_temporal,
            sample_rate=0.1,
        )

        parts = [self.core.render_all()]

        if include_temporal:
            knowledge = self.temporal.current_knowledge()
            if "No current knowledge" not in knowledge:
                parts.append(f"\n<knowledge_graph>\n{knowledge}\n</knowledge_graph>")

        return "\n\n".join(parts)

    def export_state(self) -> Dict[str, Any]:
        """Export full memory state for checkpointing."""
        _logger.info(
            "Exporting memory state",
            component="memory",
            agent_id=self.agent_id,
            core_blocks=len(self.core.blocks),
            temporal_facts=len(self.temporal.facts),
        )

        return {
            "agent_id": self.agent_id,
            "core": self.core.export(),
            "temporal_facts": [
                {
                    "id": f.id,
                    "subject": f.subject,
                    "predicate": f.predicate,
                    "object": f.object,
                    "valid_from": f.valid_from.isoformat(),
                    "valid_to": f.valid_to.isoformat() if f.valid_to else None,
                    "confidence": f.confidence,
                    "source": f.source
                }
                for f in self.temporal.facts.values()
            ]
        }

    async def import_state(self, state: Dict[str, Any]) -> None:
        """Import memory state from checkpoint."""
        _logger.info(
            "Importing memory state",
            component="memory",
            agent_id=self.agent_id,
            has_core=("core" in state),
            has_temporal=("temporal_facts" in state),
        )

        if "core" in state:
            self.core.import_state(state["core"])

        # Temporal facts would need to be reloaded
        # (implementation depends on checkpoint format)


# =============================================================================
# Demo / Test
# =============================================================================

def main():
    """Demo the memory system."""
    print("=" * 60)
    print("MEMORY SYSTEM DEMO")
    print("=" * 60)
    print()

    # Create memory system
    memory = MemorySystem(agent_id="demo-agent")

    # Test core memory
    print("[>>] Testing Core Memory...")
    memory.core.update("system_persona", "You are a helpful coding assistant.")
    memory.core.update("user_context", "User is working on a Python project.")
    memory.core.update("task_state", "Currently debugging an import error.")
    print(f"  Core tokens: {memory.core.total_tokens()}")

    # Test temporal graph
    print("\n[>>] Testing Temporal Graph...")
    memory.temporal.add_fact("user", "prefers", "dark_mode", source="explicit")
    memory.temporal.add_fact("project", "language", "python", source="inferred")
    memory.temporal.add_fact("project", "framework", "fastapi", source="inferred")

    # Change a fact (old fact gets invalidated)
    print("  Updating preference...")
    memory.temporal.add_fact("user", "prefers", "light_mode", source="explicit")

    current = memory.temporal.find_facts(subject="user", current_only=True)
    print(f"  Current facts about user: {[f.to_triple() for f in current]}")

    # Render context
    print("\n[>>] Full Context:")
    print("-" * 40)
    print(memory.render_context())
    print("-" * 40)

    print("\n[OK] Memory system demo complete")


if __name__ == "__main__":
    main()
