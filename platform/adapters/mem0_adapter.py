"""
Mem0 Adapter for Unleashed Platform

Mem0 provides a universal memory layer for AI assistants with multiple backends.

Key features:
- Universal memory layer for AI assistants
- Multiple backends: SQLite, Supabase, Pinecone, Weaviate
- Graph memory with Neo4j support
- MCP protocol compatible

Repository: https://github.com/mem0ai/mem0
Stars: 45,700 | Version: 1.0.0 | License: MIT
"""

import os
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Check Mem0 availability
MEM0_AVAILABLE = False
mem0 = None
Memory = None

try:
    from mem0 import Memory as _Memory
    import mem0 as _mem0

    mem0 = _mem0
    Memory = _Memory
    MEM0_AVAILABLE = True
except ImportError:
    pass

# Register adapter status
from . import register_adapter
register_adapter("mem0", MEM0_AVAILABLE, "1.0.0" if MEM0_AVAILABLE else None)


class MemoryBackend(Enum):
    """Supported memory backends."""
    SQLITE = "sqlite"
    SUPABASE = "supabase"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    CHROMA = "chroma"


class MemoryType(Enum):
    """Types of memory storage."""
    SHORT_TERM = "short_term"  # Session-level
    LONG_TERM = "long_term"    # Persistent across sessions
    EPISODIC = "episodic"      # Event-based memories
    SEMANTIC = "semantic"      # Factual knowledge
    PROCEDURAL = "procedural"  # How-to knowledge


@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str
    content: str
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    memory_type: MemoryType = MemoryType.LONG_TERM
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    score: Optional[float] = None


@dataclass
class SearchResult:
    """Result from memory search."""
    memories: List[MemoryEntry]
    total: int
    query: str
    search_time: float


class Mem0Adapter:
    """
    Adapter for Mem0 memory layer.

    Mem0 provides intelligent memory management for AI applications,
    supporting multiple backends and automatic memory categorization.
    """

    def __init__(
        self,
        backend: MemoryBackend = MemoryBackend.SQLITE,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Mem0 adapter.

        Args:
            backend: Memory storage backend
            config: Backend-specific configuration
        """
        self._available = MEM0_AVAILABLE
        self.backend = backend
        self.config = config or {}
        self._memory: Optional[Any] = None
        self._initialized = False

    def get_status(self) -> Dict[str, Any]:
        """Get adapter status."""
        return {
            "available": self._available,
            "backend": self.backend.value,
            "initialized": self._initialized,
            "memory_active": self._memory is not None,
        }

    def _check_available(self):
        """Check if Mem0 is available, raise error if not."""
        if not self._available:
            raise ImportError(
                "Mem0 is not installed. Install with: pip install mem0ai"
            )

    def initialize(
        self,
        llm_config: Optional[Dict[str, Any]] = None,
        embedder_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the memory instance.

        Args:
            llm_config: LLM configuration for memory processing
            embedder_config: Embedder configuration for vector storage
        """
        self._check_available()
        config = {
            "version": "v1.1",
        }

        # Backend-specific configuration
        if self.backend == MemoryBackend.SQLITE:
            config["vector_store"] = {
                "provider": "chroma",
                "config": {
                    "collection_name": "unleashed_memories",
                    "path": self.config.get("db_path", "./.mem0/chroma"),
                },
            }
        elif self.backend == MemoryBackend.QDRANT:
            config["vector_store"] = {
                "provider": "qdrant",
                "config": {
                    "collection_name": self.config.get("collection", "unleashed"),
                    "host": self.config.get("host", "localhost"),
                    "port": self.config.get("port", 6333),
                },
            }
        elif self.backend == MemoryBackend.PINECONE:
            config["vector_store"] = {
                "provider": "pinecone",
                "config": {
                    "api_key": self.config.get("api_key", os.getenv("PINECONE_API_KEY")),
                    "environment": self.config.get("environment", "us-west1-gcp"),
                    "index_name": self.config.get("index", "unleashed"),
                },
            }

        # LLM configuration
        if llm_config:
            config["llm"] = llm_config
        else:
            config["llm"] = {
                "provider": "anthropic",
                "config": {
                    "model": "claude-3-haiku-20240307",
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                },
            }

        # Embedder configuration
        if embedder_config:
            config["embedder"] = embedder_config
        else:
            config["embedder"] = {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                },
            }

        self._memory = Memory.from_config(config)
        self._initialized = True
        return self

    def add(
        self,
        content: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        memory_type: MemoryType = MemoryType.LONG_TERM,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEntry:
        """
        Add a memory.

        Args:
            content: Memory content (text)
            user_id: User identifier
            agent_id: Agent identifier
            session_id: Session identifier
            memory_type: Type of memory
            metadata: Additional metadata

        Returns:
            Created MemoryEntry
        """
        self._ensure_initialized()

        # Build kwargs
        kwargs = {}
        if user_id:
            kwargs["user_id"] = user_id
        if agent_id:
            kwargs["agent_id"] = agent_id
        if session_id:
            kwargs["run_id"] = session_id

        # Add memory type to metadata
        full_metadata = metadata or {}
        full_metadata["memory_type"] = memory_type.value
        kwargs["metadata"] = full_metadata

        # Add to Mem0
        result = self._memory.add(content, **kwargs)

        # Handle different result formats
        if isinstance(result, dict):
            memory_id = result.get("id", result.get("results", [{}])[0].get("id", ""))
        elif isinstance(result, list) and len(result) > 0:
            memory_id = result[0].get("id", "")
        else:
            memory_id = str(result)

        return MemoryEntry(
            id=memory_id,
            content=content,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            memory_type=memory_type,
            metadata=full_metadata,
        )

    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 10,
        memory_type: Optional[MemoryType] = None,
    ) -> SearchResult:
        """
        Search memories.

        Args:
            query: Search query
            user_id: Filter by user
            agent_id: Filter by agent
            limit: Maximum results
            memory_type: Filter by memory type

        Returns:
            SearchResult with matching memories
        """
        import time
        start_time = time.time()

        self._ensure_initialized()

        # Build kwargs
        kwargs = {"limit": limit}
        if user_id:
            kwargs["user_id"] = user_id
        if agent_id:
            kwargs["agent_id"] = agent_id

        # Search
        results = self._memory.search(query, **kwargs)

        # Parse results
        memories = []
        raw_results = results.get("results", results) if isinstance(results, dict) else results

        for item in raw_results:
            if isinstance(item, dict):
                entry = MemoryEntry(
                    id=item.get("id", ""),
                    content=item.get("memory", item.get("text", "")),
                    user_id=item.get("user_id"),
                    agent_id=item.get("agent_id"),
                    metadata=item.get("metadata", {}),
                    score=item.get("score"),
                )

                # Filter by memory type if specified
                if memory_type:
                    if entry.metadata.get("memory_type") != memory_type.value:
                        continue

                memories.append(entry)

        search_time = time.time() - start_time

        return SearchResult(
            memories=memories,
            total=len(memories),
            query=query,
            search_time=search_time,
        )

    def get(self, memory_id: str) -> Optional[MemoryEntry]:
        """
        Get a specific memory by ID.

        Args:
            memory_id: Memory identifier

        Returns:
            MemoryEntry or None
        """
        self._ensure_initialized()

        try:
            result = self._memory.get(memory_id)
            if result:
                return MemoryEntry(
                    id=result.get("id", memory_id),
                    content=result.get("memory", result.get("text", "")),
                    user_id=result.get("user_id"),
                    agent_id=result.get("agent_id"),
                    metadata=result.get("metadata", {}),
                )
        except Exception:
            pass
        return None

    def get_all(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[MemoryEntry]:
        """
        Get all memories for a user/agent.

        Args:
            user_id: Filter by user
            agent_id: Filter by agent
            limit: Maximum results

        Returns:
            List of MemoryEntry
        """
        self._ensure_initialized()

        kwargs = {}
        if user_id:
            kwargs["user_id"] = user_id
        if agent_id:
            kwargs["agent_id"] = agent_id

        results = self._memory.get_all(**kwargs)

        memories = []
        raw_results = results.get("results", results) if isinstance(results, dict) else results

        for item in raw_results[:limit]:
            if isinstance(item, dict):
                memories.append(MemoryEntry(
                    id=item.get("id", ""),
                    content=item.get("memory", item.get("text", "")),
                    user_id=item.get("user_id"),
                    agent_id=item.get("agent_id"),
                    metadata=item.get("metadata", {}),
                ))

        return memories

    def update(
        self,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryEntry:
        """
        Update a memory.

        Args:
            memory_id: Memory identifier
            content: New content
            metadata: Updated metadata

        Returns:
            Updated MemoryEntry
        """
        self._ensure_initialized()

        update_data = {"memory": content}
        if metadata:
            update_data["metadata"] = metadata

        self._memory.update(memory_id, update_data)

        return MemoryEntry(
            id=memory_id,
            content=content,
            metadata=metadata or {},
            updated_at=datetime.now(),
        )

    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory.

        Args:
            memory_id: Memory identifier

        Returns:
            True if deleted
        """
        self._ensure_initialized()

        try:
            self._memory.delete(memory_id)
            return True
        except Exception:
            return False

    def delete_all(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> int:
        """
        Delete all memories for a user/agent.

        Args:
            user_id: Filter by user
            agent_id: Filter by agent

        Returns:
            Number of deleted memories
        """
        self._ensure_initialized()

        kwargs = {}
        if user_id:
            kwargs["user_id"] = user_id
        if agent_id:
            kwargs["agent_id"] = agent_id

        # Get count first
        all_memories = self.get_all(**kwargs)
        count = len(all_memories)

        self._memory.delete_all(**kwargs)
        return count

    def history(
        self,
        memory_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get memory history (if supported by backend).

        Args:
            memory_id: Memory identifier
            limit: Maximum history entries

        Returns:
            List of history entries
        """
        self._ensure_initialized()

        try:
            return self._memory.history(memory_id)[:limit]
        except Exception:
            return []

    def reset(self):
        """Reset all memories (use with caution)."""
        self._ensure_initialized()
        self._memory.reset()

    def _ensure_initialized(self):
        """Ensure memory is initialized."""
        if not self._initialized or self._memory is None:
            self.initialize()


class UnifiedMemoryLayer:
    """
    Combines Mem0 with other memory systems for comprehensive memory.

    This provides a unified interface across:
    - Mem0 (short-term, vector-based)
    - Letta (long-term, stateful)
    - Graphiti (temporal, graph-based)
    """

    def __init__(
        self,
        enable_letta: bool = False,
        enable_graphiti: bool = False,
    ):
        """
        Initialize unified memory layer.

        Args:
            enable_letta: Enable Letta integration
            enable_graphiti: Enable Graphiti integration
        """
        self.mem0 = Mem0Adapter()
        self._letta = None
        self._graphiti = None

        self.enable_letta = enable_letta
        self.enable_graphiti = enable_graphiti

    async def remember(
        self,
        content: str,
        metadata: Dict[str, Any],
        memory_type: MemoryType = MemoryType.LONG_TERM,
    ) -> Dict[str, Any]:
        """
        Store across all memory layers.

        Args:
            content: Content to remember
            metadata: Associated metadata
            memory_type: Type of memory

        Returns:
            Dict with storage results
        """
        results = {}

        # Always store in Mem0
        mem0_entry = self.mem0.add(
            content=content,
            user_id=metadata.get("user_id"),
            agent_id=metadata.get("agent_id"),
            session_id=metadata.get("session_id"),
            memory_type=memory_type,
            metadata=metadata,
        )
        results["mem0"] = {"id": mem0_entry.id, "success": True}

        # Store in Letta if enabled
        if self.enable_letta and self._letta:
            try:
                letta_result = await self._letta.add_memory(content, metadata)
                results["letta"] = {"success": True, "result": letta_result}
            except Exception as e:
                results["letta"] = {"success": False, "error": str(e)}

        # Store in Graphiti if enabled
        if self.enable_graphiti and self._graphiti:
            try:
                graphiti_result = await self._graphiti.add_episode(
                    content=content,
                    timestamp=metadata.get("timestamp"),
                    entity_type=metadata.get("type", "fact"),
                )
                results["graphiti"] = {"success": True, "result": graphiti_result}
            except Exception as e:
                results["graphiti"] = {"success": False, "error": str(e)}

        return results

    async def recall(
        self,
        query: str,
        strategy: str = "hybrid",
        limit: int = 10,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve using best strategy.

        Args:
            query: Search query
            strategy: Retrieval strategy (fast/deep/hybrid)
            limit: Maximum results
            user_id: Filter by user

        Returns:
            Dict with search results
        """
        results = {}

        if strategy == "fast":
            # Use only Mem0 for speed
            mem0_results = self.mem0.search(query, user_id=user_id, limit=limit)
            results["mem0"] = [
                {"id": m.id, "content": m.content, "score": m.score}
                for m in mem0_results.memories
            ]

        elif strategy == "deep":
            # Use Graphiti for deep graph search
            if self.enable_graphiti and self._graphiti:
                graphiti_results = await self._graphiti.search(query, include_edges=True)
                results["graphiti"] = graphiti_results
            else:
                # Fallback to Mem0
                mem0_results = self.mem0.search(query, user_id=user_id, limit=limit)
                results["mem0"] = [
                    {"id": m.id, "content": m.content, "score": m.score}
                    for m in mem0_results.memories
                ]

        else:  # hybrid
            # Combine all sources
            mem0_results = self.mem0.search(query, user_id=user_id, limit=limit)
            results["mem0"] = [
                {"id": m.id, "content": m.content, "score": m.score}
                for m in mem0_results.memories
            ]

            if self.enable_graphiti and self._graphiti:
                try:
                    graphiti_results = await self._graphiti.search(query)
                    results["graphiti"] = graphiti_results
                except Exception:
                    pass

            # Merge and deduplicate
            results["merged"] = self._merge_results(results)

        return results

    def _merge_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Merge and deduplicate results from multiple sources."""
        merged = []
        seen_content = set()

        # Add Mem0 results first (usually highest quality)
        for item in results.get("mem0", []):
            content = item.get("content", "")
            if content and content not in seen_content:
                merged.append(item)
                seen_content.add(content)

        # Add Graphiti results
        for item in results.get("graphiti", []):
            content = item.get("content", item.get("text", ""))
            if content and content not in seen_content:
                merged.append({"content": content, "source": "graphiti"})
                seen_content.add(content)

        return merged
