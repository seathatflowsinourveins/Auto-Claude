"""
Cross-Session Memory Persistence - Unleashed Platform V3

This module provides persistent memory that survives across Claude Code sessions,
enabling true autonomous operation with context retention.

Key Features:
- File-based persistence (survives restarts)
- Semantic search via embeddings
- Temporal knowledge tracking
- Session handoff with full context
- Integration with Zep/Graphiti patterns
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Memory:
    """A single memory unit with temporal metadata."""
    id: str
    content: str
    memory_type: str  # "fact", "decision", "learning", "context", "task"
    created_at: str
    updated_at: str
    session_id: str
    importance: float = 0.5  # 0.0 to 1.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None  # None means still valid

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Memory:
        return cls(**data)


@dataclass
class Session:
    """Session metadata for cross-session tracking."""
    id: str
    started_at: str
    ended_at: Optional[str] = None
    task_summary: str = ""
    memories_created: int = 0
    decisions_made: List[str] = field(default_factory=list)
    learnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Session:
        return cls(**data)


@dataclass
class MemoryIndex:
    """Index for fast memory retrieval."""
    by_type: Dict[str, List[str]] = field(default_factory=dict)  # type -> [memory_ids]
    by_tag: Dict[str, List[str]] = field(default_factory=dict)   # tag -> [memory_ids]
    by_session: Dict[str, List[str]] = field(default_factory=dict)  # session -> [memory_ids]
    recent: List[str] = field(default_factory=list)  # Most recent memory IDs


# =============================================================================
# MEMORY STORE
# =============================================================================

class CrossSessionMemory:
    """
    Persistent memory store that survives across Claude Code sessions.

    Storage structure:
    .claude/
      unleash_memory/
        memories.json      - All memories
        sessions.json      - Session history
        index.json         - Search index
        embeddings.json    - Embedding cache (optional)

    V16 ENHANCEMENT: Optional Letta Cloud sync for true cross-device persistence.
    When letta_sync=True, memories are backed up to Letta passages for:
    - Multi-machine access
    - Cloud backup/recovery
    - Integration with Letta agent memory
    """

    def __init__(
        self,
        base_path: Optional[Path] = None,
        letta_sync: bool = False,
        letta_api_key: Optional[str] = None,
        letta_agent_id: Optional[str] = None
    ):
        """Initialize cross-session memory.

        V16 ENHANCEMENT: Added optional Letta Cloud sync.

        Args:
            base_path: Local storage path. Defaults to ~/.claude/unleash_memory
            letta_sync: If True, sync memories to Letta Cloud passages.
            letta_api_key: Letta API key. Falls back to LETTA_API_KEY env var.
            letta_agent_id: Letta agent ID for sync. Falls back to UNLEASH agent.
        """
        self.base_path = base_path or Path.home() / ".claude" / "unleash_memory"
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.memories_file = self.base_path / "memories.json"
        self.sessions_file = self.base_path / "sessions.json"
        self.index_file = self.base_path / "index.json"

        self._memories: Dict[str, Memory] = {}
        self._sessions: Dict[str, Session] = {}
        self._index = MemoryIndex()
        self._current_session: Optional[Session] = None

        # V16: Letta Cloud sync configuration
        self._letta_sync = letta_sync
        self._letta_api_key = letta_api_key or os.environ.get("LETTA_API_KEY")
        self._letta_agent_id = letta_agent_id or "agent-daee71d2-193b-485e-bda4-ee44752635fe"  # UNLEASH default
        self._letta_client = None

        self._load()

    def _load(self) -> None:
        """Load memories from disk."""
        # Load memories
        if self.memories_file.exists():
            try:
                with open(self.memories_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._memories = {k: Memory.from_dict(v) for k, v in data.items()}
                logger.info(f"Loaded {len(self._memories)} memories from disk")
            except Exception as e:
                logger.error(f"Failed to load memories: {e}")
                self._memories = {}

        # Load sessions
        if self.sessions_file.exists():
            try:
                with open(self.sessions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._sessions = {k: Session.from_dict(v) for k, v in data.items()}
                logger.info(f"Loaded {len(self._sessions)} session records")
            except Exception as e:
                logger.error(f"Failed to load sessions: {e}")
                self._sessions = {}

        # Load or rebuild index
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._index = MemoryIndex(**data)
            except Exception as e:
                logger.warning(f"Failed to load index, rebuilding: {e}")
                self._rebuild_index()
        else:
            self._rebuild_index()

    def _save(self) -> None:
        """Save memories to disk and optionally sync to Letta Cloud."""
        try:
            # Save memories
            with open(self.memories_file, 'w', encoding='utf-8') as f:
                data = {k: v.to_dict() for k, v in self._memories.items()}
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Save sessions
            with open(self.sessions_file, 'w', encoding='utf-8') as f:
                data = {k: v.to_dict() for k, v in self._sessions.items()}
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Save index
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self._index), f, indent=2)

            # V16: Sync to Letta Cloud if enabled
            if self._letta_sync:
                self._sync_to_letta()

        except Exception as e:
            logger.error(f"Failed to save memory store: {e}")

    def _get_letta_client(self):
        """Get or create Letta client for cloud sync."""
        if self._letta_client is None and self._letta_api_key:
            try:
                from letta_client import Letta
                import httpx

                self._letta_client = Letta(
                    api_key=self._letta_api_key,
                    base_url="https://api.letta.com",
                    http_client=httpx.Client(
                        limits=httpx.Limits(max_connections=50, max_keepalive_connections=10),
                        timeout=httpx.Timeout(30.0)
                    )
                )
            except ImportError:
                logger.warning("Letta client not available for cloud sync")
                return None
        return self._letta_client

    def _sync_to_letta(self) -> None:
        """Sync important memories to Letta Cloud passages (DEPRECATED - use async version).

        V16 ENHANCEMENT: Backs up high-importance memories to Letta Cloud.
        Only syncs memories with importance >= 0.7 to avoid noise.

        V42 NOTE: This method blocks. Use _sync_to_letta_async() for async contexts.
        """
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, schedule the async version
            asyncio.create_task(self._sync_to_letta_async())
        except RuntimeError:
            # No event loop running - run blocking (legacy behavior)
            self._sync_to_letta_blocking()

    def _sync_to_letta_blocking(self) -> None:
        """Original blocking sync (for legacy/sync contexts)."""
        client = self._get_letta_client()
        if not client:
            return

        try:
            # Filter high-importance memories
            important_memories = [
                m for m in self._memories.values()
                if m.importance >= 0.7
            ]

            if not important_memories:
                return

            # Batch sync (up to 50 at a time)
            batch_size = 50
            for i in range(0, len(important_memories), batch_size):
                batch = important_memories[i:i + batch_size]

                for memory in batch:
                    # Format for Letta passage
                    passage_text = (
                        f"[{memory.memory_type}] {memory.content}\n"
                        f"Tags: {', '.join(memory.tags)}\n"
                        f"Created: {memory.created_at}\n"
                        f"Importance: {memory.importance}"
                    )

                    try:
                        client.agents.passages.create(
                            self._letta_agent_id,
                            text=passage_text,
                            tags=["cross-session", memory.memory_type] + memory.tags
                        )
                    except Exception as e:
                        logger.debug(f"Failed to sync memory {memory.id}: {e}")

            logger.info(f"Synced {len(important_memories)} memories to Letta Cloud")

        except Exception as e:
            logger.error(f"Letta sync failed: {e}")

    async def _sync_to_letta_async(self) -> None:
        """V42 ASYNC: Sync important memories to Letta Cloud passages (non-blocking).

        Wraps blocking Letta SDK calls with asyncio.to_thread() for async safety.
        """
        client = self._get_letta_client()
        if not client:
            return

        try:
            # Filter high-importance memories
            important_memories = [
                m for m in self._memories.values()
                if m.importance >= 0.7
            ]

            if not important_memories:
                return

            # Batch sync (up to 50 at a time)
            batch_size = 50
            for i in range(0, len(important_memories), batch_size):
                batch = important_memories[i:i + batch_size]

                for memory in batch:
                    # Format for Letta passage
                    passage_text = (
                        f"[{memory.memory_type}] {memory.content}\n"
                        f"Tags: {', '.join(memory.tags)}\n"
                        f"Created: {memory.created_at}\n"
                        f"Importance: {memory.importance}"
                    )

                    try:
                        # V42 FIX: Wrap sync Letta SDK call with asyncio.to_thread()
                        await asyncio.to_thread(
                            client.agents.passages.create,
                            self._letta_agent_id,
                            text=passage_text,
                            tags=["cross-session", memory.memory_type] + memory.tags
                        )
                    except Exception as e:
                        logger.debug(f"Failed to sync memory {memory.id}: {e}")

            logger.info(f"Synced {len(important_memories)} memories to Letta Cloud (async)")

        except Exception as e:
            logger.error(f"Letta sync failed: {e}")

    def sync_from_letta(self, limit: int = 100) -> int:
        """Download memories from Letta Cloud passages (DEPRECATED - use async version).

        V16 ENHANCEMENT: Import memories from Letta Cloud to local storage.
        V42 NOTE: This method blocks. Use sync_from_letta_async() for async contexts.

        Args:
            limit: Maximum passages to import.

        Returns:
            Number of memories imported.
        """
        client = self._get_letta_client()
        if not client:
            return 0

        try:
            # Search for cross-session tagged passages
            results = client.agents.passages.search(
                self._letta_agent_id,
                query="cross-session",
                top_k=limit,
                tags=["cross-session"]
            )

            # V21 FIX: Use .results not .passages
            passages = getattr(results, 'results', results) or []
            imported = 0

            for passage in passages:
                passage_id = getattr(passage, 'id', None)
                if not passage_id or passage_id in self._memories:
                    continue

                # V23 FIX: Search results use .content, not .text
                content = getattr(passage, 'content', getattr(passage, 'text', str(passage)))

                # Create memory from passage
                memory = Memory(
                    id=passage_id,
                    content=content,
                    memory_type="imported",
                    created_at=datetime.now(timezone.utc).isoformat(),
                    updated_at=datetime.now(timezone.utc).isoformat(),
                    session_id="letta_import",
                    importance=0.7,
                    tags=["letta_import"],
                    metadata={"source": "letta_cloud"}
                )

                self._memories[passage_id] = memory
                imported += 1

            if imported > 0:
                self._save()
                logger.info(f"Imported {imported} memories from Letta Cloud")

            return imported

        except Exception as e:
            logger.error(f"Failed to sync from Letta: {e}")
            return 0

    async def sync_from_letta_async(self, limit: int = 100) -> int:
        """V42 ASYNC: Download memories from Letta Cloud passages (non-blocking).

        Wraps blocking Letta SDK calls with asyncio.to_thread() for async safety.

        Args:
            limit: Maximum passages to import.

        Returns:
            Number of memories imported.
        """
        client = self._get_letta_client()
        if not client:
            return 0

        try:
            # V42 FIX: Wrap sync Letta SDK call with asyncio.to_thread()
            results = await asyncio.to_thread(
                client.agents.passages.search,
                self._letta_agent_id,
                query="cross-session",
                top_k=limit,
                tags=["cross-session"]
            )

            # V21 FIX: Use .results not .passages
            passages = getattr(results, 'results', results) or []
            imported = 0

            for passage in passages:
                passage_id = getattr(passage, 'id', None)
                if not passage_id or passage_id in self._memories:
                    continue

                # V23 FIX: Search results use .content, not .text
                content = getattr(passage, 'content', getattr(passage, 'text', str(passage)))

                # Create memory from passage
                memory = Memory(
                    id=passage_id,
                    content=content,
                    memory_type="imported",
                    created_at=datetime.now(timezone.utc).isoformat(),
                    updated_at=datetime.now(timezone.utc).isoformat(),
                    session_id="letta_import",
                    importance=0.7,
                    tags=["letta_import"],
                    metadata={"source": "letta_cloud"}
                )

                self._memories[passage_id] = memory
                imported += 1

            if imported > 0:
                self._save()
                logger.info(f"Imported {imported} memories from Letta Cloud (async)")

            return imported

        except Exception as e:
            logger.error(f"Failed to sync from Letta: {e}")
            return 0

    # =========================================================================
    # V131: LETTA BLOCKS API SYNC (Core Memory)
    # =========================================================================

    def get_blocks(self) -> Dict[str, str]:
        """V131: Get all memory blocks from Letta agent.

        Returns:
            Dict mapping block label to block value.
        """
        client = self._get_letta_client()
        if not client:
            return {}

        try:
            blocks = list(client.agents.blocks.list(self._letta_agent_id))
            return {
                getattr(block, 'label', 'unknown'): getattr(block, 'value', '')
                for block in blocks
            }
        except Exception as e:
            logger.error(f"Failed to get Letta blocks: {e}")
            return {}

    def get_block(self, block_label: str) -> Optional[str]:
        """V131: Get a specific memory block from Letta agent.

        Args:
            block_label: Block label (e.g., "human", "persona", "system_state")

        Returns:
            Block value or None if not found.
        """
        client = self._get_letta_client()
        if not client:
            return None

        try:
            # CRITICAL: block_label is POSITIONAL, agent_id is KEYWORD
            block = client.agents.blocks.retrieve(block_label, agent_id=self._letta_agent_id)
            return getattr(block, 'value', None)
        except Exception as e:
            logger.debug(f"Block '{block_label}' not found: {e}")
            return None

    def update_block(self, block_label: str, value: str) -> bool:
        """V131: Update a memory block on Letta agent.

        Args:
            block_label: Block label to update
            value: New value for the block (max ~4000 chars)

        Returns:
            True if successful.
        """
        client = self._get_letta_client()
        if not client:
            return False

        try:
            # Truncate if too long (Letta blocks have ~4000 char limit)
            if len(value) > 4000:
                value = value[:3900] + "\n[truncated...]"

            # CRITICAL: block_label is POSITIONAL, agent_id is KEYWORD
            client.agents.blocks.update(
                block_label,
                agent_id=self._letta_agent_id,
                value=value
            )
            logger.info(f"Updated Letta block '{block_label}' ({len(value)} chars)")
            return True
        except Exception as e:
            logger.error(f"Failed to update block '{block_label}': {e}")
            return False

    def sync_state_to_block(self, block_label: str = "system_state") -> bool:
        """V131: Sync current session state to a Letta block.

        This syncs critical active state to Core Memory so it's always
        in the agent's context on next session.

        Args:
            block_label: Target block label (default: "system_state")

        Returns:
            True if successful.
        """
        if not self._letta_sync:
            return False

        # Build state summary
        state_parts = []

        # Current session info
        if self._current_session:
            state_parts.append(f"# Active Session: {self._current_session.id}")
            state_parts.append(f"Started: {self._current_session.started_at}")
            if self._current_session.task_summary:
                state_parts.append(f"Task: {self._current_session.task_summary}")

        # Recent decisions (most critical for context)
        decisions = self.get_decisions(limit=5)
        if decisions:
            state_parts.append("\n## Recent Decisions")
            for d in decisions:
                state_parts.append(f"- {d.content[:150]}")

        # Active learnings
        learnings = self.get_learnings(limit=3)
        if learnings:
            state_parts.append("\n## Key Learnings")
            for l in learnings:
                state_parts.append(f"- {l.content[:150]}")

        # Critical facts
        facts = [m for m in self._memories.values()
                if m.memory_type == "fact" and m.importance >= 0.8 and m.valid_to is None]
        facts = sorted(facts, key=lambda m: m.importance, reverse=True)[:3]
        if facts:
            state_parts.append("\n## Critical Facts")
            for f in facts:
                state_parts.append(f"- {f.content[:150]}")

        # Add timestamp
        state_parts.append(f"\n---\nLast sync: {datetime.now(timezone.utc).isoformat()}")

        state_value = "\n".join(state_parts)
        return self.update_block(block_label, state_value)

    def load_state_from_block(self, block_label: str = "system_state") -> Dict[str, Any]:
        """V131: Load session state from a Letta block.

        Call this on session start to restore context from previous sessions.

        Args:
            block_label: Source block label (default: "system_state")

        Returns:
            Parsed state dict with sections.
        """
        value = self.get_block(block_label)
        if not value:
            return {}

        result = {
            "raw": value,
            "decisions": [],
            "learnings": [],
            "facts": [],
            "session_id": None,
            "last_sync": None
        }

        # Parse sections (simple markdown parsing)
        current_section = None
        for line in value.split('\n'):
            line = line.strip()

            if line.startswith('# Active Session:'):
                result["session_id"] = line.replace('# Active Session:', '').strip()
            elif line.startswith('Last sync:'):
                result["last_sync"] = line.replace('Last sync:', '').strip()
            elif line.startswith('## Recent Decisions'):
                current_section = "decisions"
            elif line.startswith('## Key Learnings'):
                current_section = "learnings"
            elif line.startswith('## Critical Facts'):
                current_section = "facts"
            elif line.startswith('- ') and current_section:
                result[current_section].append(line[2:])

        return result

    def sync_blocks_bidirectional(self) -> Dict[str, Any]:
        """V131: Perform bidirectional sync between local and Letta blocks.

        1. Load state from Letta blocks
        2. Merge with local state
        3. Push updated state back to blocks

        Returns:
            Sync statistics.
        """
        if not self._letta_sync:
            return {"status": "disabled", "synced": False}

        stats = {
            "status": "success",
            "synced": True,
            "blocks_read": 0,
            "blocks_updated": 0,
            "items_imported": 0
        }

        try:
            # Load current block state
            remote_state = self.load_state_from_block()
            if remote_state.get("raw"):
                stats["blocks_read"] = 1

                # Import remote decisions/learnings if not already local
                for decision_text in remote_state.get("decisions", []):
                    # Check if already exists locally
                    existing = self.search(decision_text[:50], memory_type="decision", limit=1)
                    if not existing:
                        self.add(decision_text, memory_type="decision", importance=0.7,
                                tags=["letta_import"], metadata={"source": "letta_block"})
                        stats["items_imported"] += 1

            # Sync current state back to block
            if self.sync_state_to_block():
                stats["blocks_updated"] = 1

        except Exception as e:
            stats["status"] = f"error: {e}"
            stats["synced"] = False
            logger.error(f"Bidirectional sync failed: {e}")

        return stats

    def create_shared_block(self, label: str, value: str, limit: int = 4000) -> Optional[str]:
        """V131: Create a shared block that can be attached to multiple agents.

        Args:
            label: Block label (unique identifier)
            value: Initial block content
            limit: Character limit (default 4000)

        Returns:
            Block ID if successful, None otherwise.
        """
        client = self._get_letta_client()
        if not client:
            return None

        try:
            block = client.blocks.create(label=label, value=value, limit=limit)
            block_id = getattr(block, 'id', None)
            logger.info(f"Created shared block '{label}' with ID: {block_id}")
            return block_id
        except Exception as e:
            logger.error(f"Failed to create shared block: {e}")
            return None

    def attach_shared_block(self, block_id: str) -> bool:
        """V131: Attach a shared block to this agent.

        Args:
            block_id: ID of the shared block to attach

        Returns:
            True if successful.
        """
        client = self._get_letta_client()
        if not client:
            return False

        try:
            # SDK signature: attach(block_id, *, agent_id=...) - block_id is positional
            client.agents.blocks.attach(block_id, agent_id=self._letta_agent_id)
            logger.info(f"Attached shared block {block_id} to agent")
            return True
        except Exception as e:
            logger.error(f"Failed to attach shared block: {e}")
            return False

    def detach_block(self, block_id: str) -> bool:
        """V131: Detach a block from this agent.

        Args:
            block_id: ID of the block to detach

        Returns:
            True if successful.
        """
        client = self._get_letta_client()
        if not client:
            return False

        try:
            # SDK signature: detach(block_id, *, agent_id=...) - block_id is positional
            client.agents.blocks.detach(block_id, agent_id=self._letta_agent_id)
            logger.info(f"Detached block {block_id} from agent")
            return True
        except Exception as e:
            logger.error(f"Failed to detach block: {e}")
            return False

    def get_blocks_stats(self) -> Dict[str, Any]:
        """V131: Get statistics about Letta blocks.

        Returns:
            Block statistics including labels, sizes, and sync status.
        """
        stats = {
            "letta_sync_enabled": self._letta_sync,
            "letta_agent_id": self._letta_agent_id if self._letta_sync else None,
            "blocks": {},
            "total_blocks": 0,
            "total_chars": 0
        }

        if not self._letta_sync:
            return stats

        blocks = self.get_blocks()
        stats["total_blocks"] = len(blocks)
        for label, value in blocks.items():
            char_count = len(value) if value else 0
            stats["blocks"][label] = {
                "chars": char_count,
                "usage_pct": round((char_count / 4000) * 100, 1)
            }
            stats["total_chars"] += char_count

        return stats

    # =========================================================================
    # V133: LETTA SLEEP-TIME AGENT SUPPORT
    # =========================================================================

    def enable_sleeptime(self, frequency: int = 5) -> bool:
        """V133: Enable sleep-time agent for background memory consolidation.

        Sleep-time agents automatically consolidate memory in the background:
        - Run every N conversation steps (default: 5)
        - Generate "learned context" from conversation history
        - Update memory blocks with consolidated knowledge

        Args:
            frequency: Consolidation frequency in conversation steps (default: 5)

        Returns:
            True if sleep-time was enabled successfully, False otherwise.
        """
        client = self._get_letta_client()
        if not client:
            logger.warning("Cannot enable sleep-time: Letta client not available")
            return False

        try:
            # Enable sleep-time on the agent
            client.agents.update(self._letta_agent_id, enable_sleeptime=True)

            # Configure frequency via the managed group
            agent = client.agents.retrieve(self._letta_agent_id)
            if hasattr(agent, 'managed_group') and agent.managed_group:
                group_id = agent.managed_group.id
                # V43 FIX: client.groups API does NOT exist in letta-client 1.7.7
                # The frequency configuration must be done via Letta Cloud dashboard
                # or when the SDK adds groups.update() support
                logger.info(
                    f"Sleep-time enabled for agent {self._letta_agent_id} "
                    f"(group_id={group_id}, frequency={frequency} - configure via Letta Cloud dashboard)"
                )
            else:
                logger.info(f"Sleep-time enabled for agent {self._letta_agent_id} (default frequency)")

            return True
        except Exception as e:
            logger.error(f"Failed to enable sleep-time agent: {e}")
            return False

    def disable_sleeptime(self) -> bool:
        """V133: Disable sleep-time agent.

        Returns:
            True if sleep-time was disabled successfully, False otherwise.
        """
        client = self._get_letta_client()
        if not client:
            return False

        try:
            client.agents.update(self._letta_agent_id, enable_sleeptime=False)
            logger.info(f"Sleep-time disabled for agent {self._letta_agent_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to disable sleep-time agent: {e}")
            return False

    def get_sleeptime_status(self) -> Dict[str, Any]:
        """V133: Get sleep-time agent status and configuration.

        Returns:
            Dict with enabled status, frequency, and group info.
        """
        status = {
            "enabled": False,
            "frequency": None,
            "group_id": None,
            "agent_id": self._letta_agent_id,
            "letta_available": False
        }

        client = self._get_letta_client()
        if not client:
            return status

        status["letta_available"] = True

        try:
            agent = client.agents.retrieve(self._letta_agent_id)

            # Check if sleeptime is enabled
            if hasattr(agent, 'enable_sleeptime'):
                status["enabled"] = agent.enable_sleeptime

            # Get group configuration
            if hasattr(agent, 'managed_group') and agent.managed_group:
                status["group_id"] = agent.managed_group.id
                # V43 FIX: client.groups.retrieve() does NOT exist in letta-client 1.7.7
                # Default to frequency 5 (Letta Cloud default) - actual value must be checked via dashboard
                status["frequency"] = 5  # Default frequency, configure via Letta Cloud dashboard

            return status
        except Exception as e:
            logger.error(f"Failed to get sleep-time status: {e}")
            return status

    def trigger_sleeptime_consolidation(self) -> bool:
        """V133: Manually trigger sleep-time consolidation.

        This forces an immediate consolidation cycle without waiting for
        the automatic step-based trigger.

        Note: This is a convenience method. In normal operation, sleep-time
        consolidation happens automatically based on the configured frequency.

        Returns:
            True if triggered successfully, False otherwise.
        """
        client = self._get_letta_client()
        if not client:
            return False

        try:
            # First ensure sleep-time is enabled
            status = self.get_sleeptime_status()
            if not status.get("enabled"):
                # Enable it first
                self.enable_sleeptime()

            # Send a message to trigger consolidation
            # The sleep-time agent will run after this based on step count
            client.agents.messages.create(
                self._letta_agent_id,
                messages=[{
                    "role": "user",
                    "content": "[SYSTEM] Sleep-time consolidation trigger - please consolidate recent context."
                }]
            )

            logger.info("Triggered sleep-time consolidation")
            return True
        except Exception as e:
            logger.error(f"Failed to trigger sleep-time consolidation: {e}")
            return False

    def configure_mandatory_triggers(
        self,
        on_session_end: bool = True,
        on_important_memory: bool = True,
        on_memory_count: int = 10,
        importance_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """V133: Configure mandatory sleep-time consolidation triggers.

        Sets up automatic triggers for memory consolidation:
        - Session end: Consolidate when ending a session
        - Important memories: Consolidate when high-importance memory is added
        - Memory count: Consolidate every N new memories

        Args:
            on_session_end: Trigger consolidation when session ends
            on_important_memory: Trigger when importance >= threshold
            on_memory_count: Trigger every N new memories (0 to disable)
            importance_threshold: Importance level that triggers consolidation

        Returns:
            Configuration dict.
        """
        self._sleeptime_triggers = {
            "on_session_end": on_session_end,
            "on_important_memory": on_important_memory,
            "on_memory_count": on_memory_count,
            "importance_threshold": importance_threshold,
            "memories_since_last_trigger": 0
        }

        # Store triggers in metadata for persistence
        trigger_memory = Memory(
            id="sleeptime_config",
            content=json.dumps(self._sleeptime_triggers),
            memory_type="system",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            session_id="system",
            importance=1.0,
            tags=["system", "sleeptime", "config"]
        )
        self._memories["sleeptime_config"] = trigger_memory
        self._save()

        logger.info(f"Configured mandatory sleep-time triggers: {self._sleeptime_triggers}")
        return self._sleeptime_triggers

    def get_mandatory_triggers(self) -> Dict[str, Any]:
        """V133: Get current mandatory trigger configuration.

        Returns:
            Trigger configuration dict.
        """
        if hasattr(self, '_sleeptime_triggers'):
            return self._sleeptime_triggers

        # Try to load from persisted config
        if "sleeptime_config" in self._memories:
            try:
                config = json.loads(self._memories["sleeptime_config"].content)
                self._sleeptime_triggers = config
                return config
            except Exception:
                pass

        # Default configuration
        return {
            "on_session_end": True,
            "on_important_memory": True,
            "on_memory_count": 10,
            "importance_threshold": 0.8,
            "memories_since_last_trigger": 0
        }

    def _check_mandatory_triggers(self, memory: Memory) -> None:
        """V133: Internal method to check and fire mandatory triggers.

        Called automatically when a new memory is added.
        """
        triggers = self.get_mandatory_triggers()

        should_consolidate = False
        trigger_reason = None

        # Check importance trigger
        if triggers.get("on_important_memory") and memory.importance >= triggers.get("importance_threshold", 0.8):
            should_consolidate = True
            trigger_reason = f"high_importance ({memory.importance:.2f})"

        # Check memory count trigger
        if triggers.get("on_memory_count", 0) > 0:
            count = triggers.get("memories_since_last_trigger", 0) + 1
            if count >= triggers.get("on_memory_count"):
                should_consolidate = True
                trigger_reason = f"memory_count ({count})"
                triggers["memories_since_last_trigger"] = 0
            else:
                triggers["memories_since_last_trigger"] = count

            # Update stored triggers
            if hasattr(self, '_sleeptime_triggers'):
                self._sleeptime_triggers = triggers

        if should_consolidate and self._letta_sync:
            logger.info(f"Mandatory sleep-time trigger fired: {trigger_reason}")
            self.trigger_sleeptime_consolidation()

    def _rebuild_index(self) -> None:
        """Rebuild the memory index."""
        self._index = MemoryIndex()

        for memory_id, memory in self._memories.items():
            # Index by type
            if memory.memory_type not in self._index.by_type:
                self._index.by_type[memory.memory_type] = []
            self._index.by_type[memory.memory_type].append(memory_id)

            # Index by tags
            for tag in memory.tags:
                if tag not in self._index.by_tag:
                    self._index.by_tag[tag] = []
                self._index.by_tag[tag].append(memory_id)

            # Index by session
            if memory.session_id not in self._index.by_session:
                self._index.by_session[memory.session_id] = []
            self._index.by_session[memory.session_id].append(memory_id)

        # Sort recent by creation time
        sorted_memories = sorted(
            self._memories.values(),
            key=lambda m: m.created_at,
            reverse=True
        )
        self._index.recent = [m.id for m in sorted_memories[:100]]

    def _generate_id(self, content: str) -> str:
        """Generate a unique memory ID."""
        timestamp = str(time.time())
        return hashlib.sha256(f"{content}{timestamp}".encode()).hexdigest()[:16]

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    def start_session(self, task_summary: str = "") -> Session:
        """Start a new session."""
        session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:12]
        now = datetime.now(timezone.utc).isoformat()

        self._current_session = Session(
            id=session_id,
            started_at=now,
            task_summary=task_summary
        )
        self._sessions[session_id] = self._current_session
        self._save()

        logger.info(f"Started new session: {session_id}")
        return self._current_session

    def end_session(self, summary: Optional[str] = None) -> None:
        """End the current session."""
        if self._current_session:
            self._current_session.ended_at = datetime.now(timezone.utc).isoformat()
            if summary:
                self._current_session.task_summary = summary
            self._save()
            logger.info(f"Ended session: {self._current_session.id}")

    def get_current_session(self) -> Optional[Session]:
        """Get the current session."""
        return self._current_session

    def get_session_history(self, limit: int = 10) -> List[Session]:
        """Get recent session history."""
        sorted_sessions = sorted(
            self._sessions.values(),
            key=lambda s: s.started_at,
            reverse=True
        )
        return sorted_sessions[:limit]

    # =========================================================================
    # MEMORY OPERATIONS
    # =========================================================================

    def add(
        self,
        content: str,
        memory_type: str = "context",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Memory:
        """Add a new memory."""
        now = datetime.now(timezone.utc).isoformat()
        session_id = self._current_session.id if self._current_session else "no_session"

        memory = Memory(
            id=self._generate_id(content),
            content=content,
            memory_type=memory_type,
            created_at=now,
            updated_at=now,
            session_id=session_id,
            importance=importance,
            tags=tags or [],
            metadata=metadata or {},
            valid_from=now
        )

        self._memories[memory.id] = memory
        self._update_index(memory)

        if self._current_session:
            self._current_session.memories_created += 1
            if memory_type == "decision":
                self._current_session.decisions_made.append(content[:100])
            elif memory_type == "learning":
                self._current_session.learnings.append(content[:100])

        self._save()
        return memory

    def _update_index(self, memory: Memory) -> None:
        """Update index with new memory."""
        # Index by type
        if memory.memory_type not in self._index.by_type:
            self._index.by_type[memory.memory_type] = []
        if memory.id not in self._index.by_type[memory.memory_type]:
            self._index.by_type[memory.memory_type].append(memory.id)

        # Index by tags
        for tag in memory.tags:
            if tag not in self._index.by_tag:
                self._index.by_tag[tag] = []
            if memory.id not in self._index.by_tag[tag]:
                self._index.by_tag[tag].append(memory.id)

        # Index by session
        if memory.session_id not in self._index.by_session:
            self._index.by_session[memory.session_id] = []
        if memory.id not in self._index.by_session[memory.session_id]:
            self._index.by_session[memory.session_id].append(memory.id)

        # Update recent
        if memory.id in self._index.recent:
            self._index.recent.remove(memory.id)
        self._index.recent.insert(0, memory.id)
        self._index.recent = self._index.recent[:100]

    def search(
        self,
        query: str,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
        min_importance: float = 0.0
    ) -> List[Memory]:
        """
        Search memories with text matching.

        For production, this would use embeddings for semantic search.
        Current implementation uses simple text matching.
        """
        query_lower = query.lower()
        results: List[Tuple[Memory, float]] = []

        # Get candidate memory IDs
        candidate_ids = set(self._memories.keys())

        # Filter by type
        if memory_type and memory_type in self._index.by_type:
            candidate_ids &= set(self._index.by_type[memory_type])

        # Filter by tags
        if tags:
            tag_ids = set()
            for tag in tags:
                if tag in self._index.by_tag:
                    tag_ids |= set(self._index.by_tag[tag])
            candidate_ids &= tag_ids

        # Score and filter
        for memory_id in candidate_ids:
            memory = self._memories[memory_id]

            # Skip if below importance threshold
            if memory.importance < min_importance:
                continue

            # Skip if invalidated
            if memory.valid_to is not None:
                continue

            # Calculate relevance score (simple text matching)
            content_lower = memory.content.lower()
            score = 0.0

            # Exact match bonus
            if query_lower in content_lower:
                score += 0.5

            # Word overlap
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())
            overlap = len(query_words & content_words)
            if query_words:
                score += (overlap / len(query_words)) * 0.3

            # Recency bonus
            try:
                age_hours = (datetime.now(timezone.utc) -
                           datetime.fromisoformat(memory.created_at.replace('Z', '+00:00'))).total_seconds() / 3600
                recency_score = max(0, 1 - (age_hours / 168))  # Decay over 1 week
                score += recency_score * 0.1
            except Exception:
                pass

            # Importance bonus
            score += memory.importance * 0.1

            if score > 0:
                results.append((memory, score))

        # Sort by score and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, _ in results[:limit]]

    def get_recent(self, limit: int = 10, memory_type: Optional[str] = None) -> List[Memory]:
        """Get most recent memories."""
        if memory_type:
            ids = self._index.by_type.get(memory_type, [])
        else:
            ids = self._index.recent

        memories = [self._memories[mid] for mid in ids[:limit] if mid in self._memories]
        return sorted(memories, key=lambda m: m.created_at, reverse=True)

    def get_by_type(self, memory_type: str) -> List[Memory]:
        """Get all memories of a specific type."""
        ids = self._index.by_type.get(memory_type, [])
        return [self._memories[mid] for mid in ids if mid in self._memories]

    def get_decisions(self, limit: int = 20) -> List[Memory]:
        """Get recent decisions."""
        return self.get_recent(limit=limit, memory_type="decision")

    def get_learnings(self, limit: int = 20) -> List[Memory]:
        """Get recent learnings."""
        return self.get_recent(limit=limit, memory_type="learning")

    def invalidate(self, memory_id: str, reason: str = "") -> bool:
        """Invalidate a memory (mark as no longer valid)."""
        if memory_id in self._memories:
            memory = self._memories[memory_id]
            memory.valid_to = datetime.now(timezone.utc).isoformat()
            memory.metadata["invalidation_reason"] = reason
            self._save()
            return True
        return False

    # =========================================================================
    # CONTEXT GENERATION
    # =========================================================================

    def get_session_context(self, max_tokens: int = 4000) -> str:
        """
        Generate context from previous sessions for handoff.

        This is the key function for cross-session continuity.
        """
        context_parts = []

        # Recent sessions summary
        recent_sessions = self.get_session_history(limit=5)
        if recent_sessions:
            context_parts.append("## Recent Sessions")
            for session in recent_sessions:
                context_parts.append(f"- **{session.started_at[:10]}**: {session.task_summary or 'No summary'}")
                if session.decisions_made:
                    context_parts.append(f"  Decisions: {', '.join(session.decisions_made[:3])}")

        # Key decisions
        decisions = self.get_decisions(limit=10)
        if decisions:
            context_parts.append("\n## Key Decisions")
            for d in decisions:
                context_parts.append(f"- [{d.created_at[:10]}] {d.content[:200]}")

        # Learnings
        learnings = self.get_learnings(limit=10)
        if learnings:
            context_parts.append("\n## Learnings")
            for l in learnings:
                context_parts.append(f"- {l.content[:200]}")

        # Important facts
        facts = [m for m in self._memories.values()
                if m.memory_type == "fact" and m.importance >= 0.7 and m.valid_to is None]
        facts = sorted(facts, key=lambda m: m.importance, reverse=True)[:10]
        if facts:
            context_parts.append("\n## Important Facts")
            for f in facts:
                context_parts.append(f"- {f.content[:200]}")

        context = "\n".join(context_parts)

        # Truncate if too long (rough token estimate: 4 chars per token)
        max_chars = max_tokens * 4
        if len(context) > max_chars:
            context = context[:max_chars] + "\n\n[Context truncated...]"

        return context

    def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        return {
            "total_memories": len(self._memories),
            "total_sessions": len(self._sessions),
            "memories_by_type": {k: len(v) for k, v in self._index.by_type.items()},
            "unique_tags": len(self._index.by_tag),
            "storage_path": str(self.base_path)
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_memory_store: Optional[CrossSessionMemory] = None


def get_memory_store() -> CrossSessionMemory:
    """Get or create the singleton memory store."""
    global _memory_store
    if _memory_store is None:
        _memory_store = CrossSessionMemory()
    return _memory_store


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def remember_decision(content: str, importance: float = 0.7, tags: Optional[List[str]] = None) -> Memory:
    """Remember a decision made."""
    store = get_memory_store()
    return store.add(content, memory_type="decision", importance=importance, tags=tags)


def remember_learning(content: str, importance: float = 0.6, tags: Optional[List[str]] = None) -> Memory:
    """Remember something learned."""
    store = get_memory_store()
    return store.add(content, memory_type="learning", importance=importance, tags=tags)


def remember_fact(content: str, importance: float = 0.5, tags: Optional[List[str]] = None) -> Memory:
    """Remember a fact."""
    store = get_memory_store()
    return store.add(content, memory_type="fact", importance=importance, tags=tags)


def remember_context(content: str, tags: Optional[List[str]] = None) -> Memory:
    """Remember context information."""
    store = get_memory_store()
    return store.add(content, memory_type="context", importance=0.4, tags=tags)


def recall(query: str, limit: int = 10) -> List[Memory]:
    """Recall memories matching a query."""
    store = get_memory_store()
    return store.search(query, limit=limit)


def get_context_for_new_session() -> str:
    """Get context to start a new session with previous knowledge."""
    store = get_memory_store()
    return store.get_session_context()


# =============================================================================
# MAIN - DEMO
# =============================================================================

def demo():
    """Demonstrate cross-session memory."""
    print("=" * 60)
    print("CROSS-SESSION MEMORY - DEMO")
    print("=" * 60)

    store = get_memory_store()

    # Start a new session
    session = store.start_session("Testing cross-session memory system")
    print(f"\nStarted session: {session.id}")

    # Add some memories
    print("\nAdding memories...")

    remember_decision(
        "Use DSPy for optimization layer - best benchmarks (+35% BIG-Bench)",
        importance=0.9,
        tags=["architecture", "optimization", "dspy"]
    )

    remember_decision(
        "Use LangGraph for orchestration - fastest latency, enterprise-proven",
        importance=0.9,
        tags=["architecture", "orchestration", "langgraph"]
    )

    remember_learning(
        "Zep/Graphiti achieves 94.8% DMR accuracy vs 68.5% for Mem0",
        importance=0.8,
        tags=["memory", "benchmarks", "zep"]
    )

    remember_fact(
        "Claude Opus 4.5 achieves 80.9% on SWE-bench - highest coding benchmark",
        importance=0.85,
        tags=["reasoning", "benchmarks", "claude"]
    )

    remember_context(
        "User prioritizes performance over cost for SDK selection",
        tags=["user-preference", "requirements"]
    )

    # Search memories
    print("\nSearching for 'optimization'...")
    results = recall("optimization", limit=5)
    for r in results:
        print(f"  - [{r.memory_type}] {r.content[:60]}...")

    # Get context for new session
    print("\n" + "=" * 60)
    print("CONTEXT FOR NEW SESSION")
    print("=" * 60)
    context = get_context_for_new_session()
    print(context)

    # Stats
    print("\n" + "=" * 60)
    print("MEMORY STORE STATS")
    print("=" * 60)
    stats = store.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # End session
    store.end_session("Completed cross-session memory demo")


if __name__ == "__main__":
    demo()
