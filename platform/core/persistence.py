"""
UAP Persistence Module - Session state persistence and recovery.

Implements the Shift Handoff pattern for cross-session continuity:
- Save session state to disk/memory
- Restore state for new sessions
- Checkpoint critical progress
- Support graceful degradation

Based on Claude Agent SDK session management patterns.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import shutil
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class PersistenceBackend(str, Enum):
    """Storage backends for persistence."""

    FILE = "file"              # Local filesystem
    MEMORY = "memory"          # In-memory (for testing)
    SQLITE = "sqlite"          # SQLite database
    REDIS = "redis"            # Redis cache


class CheckpointType(str, Enum):
    """Types of checkpoints."""

    AUTO = "auto"              # Automatic periodic checkpoint
    MANUAL = "manual"          # User-triggered checkpoint
    MILESTONE = "milestone"    # Task milestone achieved
    ERROR = "error"            # Checkpoint before error handling
    HANDOFF = "handoff"        # Session handoff checkpoint


# =============================================================================
# Data Models
# =============================================================================

class SessionMetadata(BaseModel):
    """Metadata about a persisted session."""

    session_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    agent_id: Optional[str] = None
    task_description: Optional[str] = None
    status: str = "active"
    iteration_count: int = 0
    checkpoint_count: int = 0
    total_tokens_used: int = 0
    tags: List[str] = Field(default_factory=list)


class Checkpoint(BaseModel):
    """A snapshot of session state at a point in time."""

    id: str
    session_id: str
    checkpoint_type: CheckpointType = CheckpointType.AUTO
    created_at: datetime = Field(default_factory=datetime.now)
    iteration: int = 0
    state_hash: str = ""
    description: Optional[str] = None

    # State data
    memory_state: Dict[str, Any] = Field(default_factory=dict)
    task_state: Dict[str, Any] = Field(default_factory=dict)
    context_state: Dict[str, Any] = Field(default_factory=dict)


class SessionState(BaseModel):
    """Complete state of a session."""

    metadata: SessionMetadata
    checkpoints: List[Checkpoint] = Field(default_factory=list)
    current_state: Dict[str, Any] = Field(default_factory=dict)

    # Components state
    memory_snapshot: Dict[str, Any] = Field(default_factory=dict)
    harness_snapshot: Dict[str, Any] = Field(default_factory=dict)
    executor_snapshot: Dict[str, Any] = Field(default_factory=dict)

    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        """Get the most recent checkpoint."""
        if not self.checkpoints:
            return None
        return max(self.checkpoints, key=lambda c: c.created_at)


# =============================================================================
# Persistence Manager
# =============================================================================

class PersistenceManager:
    """
    Manages session state persistence and recovery.

    Supports multiple backends and provides:
    - Automatic checkpointing
    - State compression
    - Incremental saves
    - Recovery from crashes
    """

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        backend: PersistenceBackend = PersistenceBackend.FILE,
        compress: bool = True,
        max_checkpoints: int = 10,
    ):
        self._storage_path = storage_path or Path.cwd() / ".uap_sessions"
        self._backend = backend
        self._compress = compress
        self._max_checkpoints = max_checkpoints
        self._memory_store: Dict[str, SessionState] = {}

        # Ensure storage directory exists
        if backend == PersistenceBackend.FILE:
            self._storage_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------

    def create_session(
        self,
        session_id: str,
        agent_id: Optional[str] = None,
        task_description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> SessionState:
        """
        Create a new session.

        Args:
            session_id: Unique session identifier
            agent_id: Associated agent ID
            task_description: Description of the task
            tags: Tags for categorization

        Returns:
            New SessionState
        """
        metadata = SessionMetadata(
            session_id=session_id,
            agent_id=agent_id,
            task_description=task_description,
            tags=tags or [],
        )

        state = SessionState(metadata=metadata)
        self._save_state(session_id, state)

        logger.info(f"Created session: {session_id}")
        return state

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Load a session by ID."""
        return self._load_state(session_id)

    def list_sessions(
        self,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[SessionMetadata]:
        """
        List all sessions matching criteria.

        Args:
            status: Filter by status
            tags: Filter by tags (any match)
            limit: Maximum results

        Returns:
            List of session metadata
        """
        sessions = []

        if self._backend == PersistenceBackend.MEMORY:
            for state in self._memory_store.values():
                sessions.append(state.metadata)
        else:
            for path in self._storage_path.glob("*.json*"):
                state = self._load_state(path.stem.replace(".json", ""))
                if state:
                    sessions.append(state.metadata)

        # Apply filters
        if status:
            sessions = [s for s in sessions if s.status == status]

        if tags:
            tag_set = set(tags)
            sessions = [s for s in sessions if tag_set & set(s.tags)]

        # Sort by updated_at descending
        sessions.sort(key=lambda s: s.updated_at, reverse=True)

        return sessions[:limit]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its checkpoints."""
        if self._backend == PersistenceBackend.MEMORY:
            if session_id in self._memory_store:
                del self._memory_store[session_id]
                return True
            return False

        session_file = self._get_session_path(session_id)
        if session_file.exists():
            session_file.unlink()

            # Also delete checkpoint directory
            checkpoint_dir = self._storage_path / f"{session_id}_checkpoints"
            if checkpoint_dir.exists():
                shutil.rmtree(checkpoint_dir)

            logger.info(f"Deleted session: {session_id}")
            return True

        return False

    # -------------------------------------------------------------------------
    # State Operations
    # -------------------------------------------------------------------------

    def save_state(
        self,
        session_id: str,
        state_key: str,
        state_value: Any,
    ) -> bool:
        """
        Save a state value to the session.

        Args:
            session_id: Session identifier
            state_key: Key for the state value
            state_value: Value to save (must be JSON serializable)

        Returns:
            True if saved successfully
        """
        session = self._load_state(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return False

        session.current_state[state_key] = state_value
        session.metadata.updated_at = datetime.now()

        self._save_state(session_id, session)
        return True

    def get_state(
        self,
        session_id: str,
        state_key: str,
        default: Any = None,
    ) -> Any:
        """Get a state value from the session."""
        session = self._load_state(session_id)
        if not session:
            return default

        return session.current_state.get(state_key, default)

    def update_metadata(
        self,
        session_id: str,
        **updates: Any,
    ) -> bool:
        """Update session metadata fields."""
        session = self._load_state(session_id)
        if not session:
            return False

        for key, value in updates.items():
            if hasattr(session.metadata, key):
                setattr(session.metadata, key, value)

        session.metadata.updated_at = datetime.now()
        self._save_state(session_id, session)
        return True

    # -------------------------------------------------------------------------
    # Checkpoints
    # -------------------------------------------------------------------------

    def create_checkpoint(
        self,
        session_id: str,
        checkpoint_type: CheckpointType = CheckpointType.AUTO,
        description: Optional[str] = None,
        memory_state: Optional[Dict[str, Any]] = None,
        task_state: Optional[Dict[str, Any]] = None,
        context_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[Checkpoint]:
        """
        Create a checkpoint of the current session state.

        Args:
            session_id: Session identifier
            checkpoint_type: Type of checkpoint
            description: Human-readable description
            memory_state: Memory system state
            task_state: Task/harness state
            context_state: Context window state

        Returns:
            Created Checkpoint or None
        """
        session = self._load_state(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return None

        # Generate checkpoint ID
        checkpoint_id = f"{session_id}_cp_{len(session.checkpoints):04d}"

        # Compute state hash
        state_data = {
            "memory": memory_state or {},
            "task": task_state or {},
            "context": context_state or {},
        }
        state_hash = hashlib.sha256(
            json.dumps(state_data, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        checkpoint = Checkpoint(
            id=checkpoint_id,
            session_id=session_id,
            checkpoint_type=checkpoint_type,
            iteration=session.metadata.iteration_count,
            state_hash=state_hash,
            description=description,
            memory_state=memory_state or {},
            task_state=task_state or {},
            context_state=context_state or {},
        )

        session.checkpoints.append(checkpoint)
        session.metadata.checkpoint_count = len(session.checkpoints)

        # Prune old checkpoints if needed
        if len(session.checkpoints) > self._max_checkpoints:
            # Keep milestones and the most recent auto checkpoints
            milestones = [
                c for c in session.checkpoints
                if c.checkpoint_type in (CheckpointType.MILESTONE, CheckpointType.HANDOFF)
            ]
            auto = [
                c for c in session.checkpoints
                if c.checkpoint_type not in (CheckpointType.MILESTONE, CheckpointType.HANDOFF)
            ]
            auto.sort(key=lambda c: c.created_at, reverse=True)

            keep_count = self._max_checkpoints - len(milestones)
            session.checkpoints = milestones + auto[:max(0, keep_count)]

        self._save_state(session_id, session)
        logger.info(f"Created checkpoint: {checkpoint_id} ({checkpoint_type.value})")

        return checkpoint

    def restore_checkpoint(
        self,
        session_id: str,
        checkpoint_id: Optional[str] = None,
    ) -> Optional[Checkpoint]:
        """
        Restore from a checkpoint.

        Args:
            session_id: Session identifier
            checkpoint_id: Specific checkpoint (or latest if None)

        Returns:
            Restored Checkpoint or None
        """
        session = self._load_state(session_id)
        if not session:
            return None

        if checkpoint_id:
            checkpoint = next(
                (c for c in session.checkpoints if c.id == checkpoint_id),
                None
            )
        else:
            checkpoint = session.get_latest_checkpoint()

        if not checkpoint:
            logger.warning(f"No checkpoint found for session: {session_id}")
            return None

        logger.info(f"Restored checkpoint: {checkpoint.id}")
        return checkpoint

    def list_checkpoints(
        self,
        session_id: str,
        checkpoint_type: Optional[CheckpointType] = None,
    ) -> List[Checkpoint]:
        """List checkpoints for a session."""
        session = self._load_state(session_id)
        if not session:
            return []

        checkpoints = session.checkpoints

        if checkpoint_type:
            checkpoints = [c for c in checkpoints if c.checkpoint_type == checkpoint_type]

        return sorted(checkpoints, key=lambda c: c.created_at, reverse=True)

    # -------------------------------------------------------------------------
    # Shift Handoff Support
    # -------------------------------------------------------------------------

    def create_handoff(
        self,
        session_id: str,
        summary: str,
        next_steps: List[str],
        critical_context: Dict[str, Any],
    ) -> Optional[str]:
        """
        Create a handoff package for session transfer.

        Args:
            session_id: Session identifier
            summary: Summary of work done
            next_steps: Recommended next actions
            critical_context: Must-preserve context

        Returns:
            Handoff checkpoint ID
        """
        checkpoint = self.create_checkpoint(
            session_id=session_id,
            checkpoint_type=CheckpointType.HANDOFF,
            description=summary,
            context_state={
                "summary": summary,
                "next_steps": next_steps,
                "critical_context": critical_context,
            },
        )

        if checkpoint:
            # Also save to a separate handoff file for easy access (file backend only)
            if self._backend == PersistenceBackend.FILE:
                handoff_path = self._storage_path / f"{session_id}_handoff.json"
                handoff_data = {
                    "checkpoint_id": checkpoint.id,
                    "created_at": checkpoint.created_at.isoformat(),
                    "summary": summary,
                    "next_steps": next_steps,
                    "critical_context": critical_context,
                }

                with open(handoff_path, "w") as f:
                    json.dump(handoff_data, f, indent=2, default=str)

            return checkpoint.id

        return None

    def load_handoff(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load the most recent handoff for a session."""
        # Try file first if using file backend
        if self._backend == PersistenceBackend.FILE:
            handoff_path = self._storage_path / f"{session_id}_handoff.json"
            if handoff_path.exists():
                with open(handoff_path) as f:
                    return json.load(f)

        # Fall back to checkpoint (works for all backends)
        session = self._load_state(session_id)
        if session:
            # Find the most recent handoff checkpoint
            handoff_checkpoints = [
                c for c in session.checkpoints
                if c.checkpoint_type == CheckpointType.HANDOFF
            ]
            if handoff_checkpoints:
                checkpoint = max(handoff_checkpoints, key=lambda c: c.created_at)
                return {
                    "checkpoint_id": checkpoint.id,
                    "summary": checkpoint.description,
                    "next_steps": checkpoint.context_state.get("next_steps", []),
                    "critical_context": checkpoint.context_state.get("critical_context", {}),
                }

        return None

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _get_session_path(self, session_id: str) -> Path:
        """Get the file path for a session."""
        ext = ".json.gz" if self._compress else ".json"
        return self._storage_path / f"{session_id}{ext}"

    def _save_state(self, session_id: str, state: SessionState) -> None:
        """Save session state to storage."""
        if self._backend == PersistenceBackend.MEMORY:
            self._memory_store[session_id] = state
            return

        session_path = self._get_session_path(session_id)
        data = state.model_dump_json(indent=2)

        if self._compress:
            with gzip.open(session_path, "wt", encoding="utf-8") as f:
                f.write(data)
        else:
            with open(session_path, "w", encoding="utf-8") as f:
                f.write(data)

    def _load_state(self, session_id: str) -> Optional[SessionState]:
        """Load session state from storage."""
        if self._backend == PersistenceBackend.MEMORY:
            return self._memory_store.get(session_id)

        # Try compressed first, then uncompressed
        for ext in [".json.gz", ".json"]:
            session_path = self._storage_path / f"{session_id}{ext}"
            if session_path.exists():
                try:
                    if ext == ".json.gz":
                        with gzip.open(session_path, "rt", encoding="utf-8") as f:
                            data = f.read()
                    else:
                        with open(session_path, encoding="utf-8") as f:
                            data = f.read()

                    return SessionState.model_validate_json(data)

                except Exception as e:
                    logger.error(f"Failed to load session {session_id}: {e}")
                    return None

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get persistence statistics."""
        sessions = self.list_sessions(limit=1000)

        total_checkpoints = 0
        by_status: Dict[str, int] = {}

        for meta in sessions:
            total_checkpoints += meta.checkpoint_count
            status = meta.status
            by_status[status] = by_status.get(status, 0) + 1

        return {
            "total_sessions": len(sessions),
            "total_checkpoints": total_checkpoints,
            "by_status": by_status,
            "storage_path": str(self._storage_path),
            "backend": self._backend.value,
            "compress": self._compress,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_persistence_manager(
    storage_path: Optional[Path] = None,
    backend: PersistenceBackend = PersistenceBackend.FILE,
) -> PersistenceManager:
    """
    Factory function to create a PersistenceManager.

    Args:
        storage_path: Where to store session data
        backend: Storage backend to use

    Returns:
        Configured PersistenceManager
    """
    return PersistenceManager(
        storage_path=storage_path,
        backend=backend,
    )


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate the persistence system."""
    import uuid

    print("=" * 60)
    print("UAP Persistence System Demo")
    print("=" * 60)

    # Create manager with memory backend for demo
    manager = PersistenceManager(backend=PersistenceBackend.MEMORY)
    print(f"\nCreated persistence manager (backend: memory)")

    # Create a session
    session_id = str(uuid.uuid4())[:8]
    print(f"\n[Create Session: {session_id}]")

    session = manager.create_session(
        session_id=session_id,
        agent_id="demo-agent",
        task_description="Demo task for persistence testing",
        tags=["demo", "test"],
    )
    print(f"  Created at: {session.metadata.created_at}")

    # Save some state
    print("\n[Save State]")
    manager.save_state(session_id, "counter", 42)
    manager.save_state(session_id, "items", ["a", "b", "c"])
    print("  Saved: counter=42, items=['a', 'b', 'c']")

    # Create checkpoints
    print("\n[Create Checkpoints]")
    for i in range(3):
        manager.update_metadata(session_id, iteration_count=i + 1)
        checkpoint = manager.create_checkpoint(
            session_id=session_id,
            checkpoint_type=CheckpointType.AUTO,
            description=f"Iteration {i + 1} checkpoint",
            task_state={"iteration": i + 1},
        )
        print(f"  Created: {checkpoint.id if checkpoint else 'failed'}")

    # Create a milestone
    milestone = manager.create_checkpoint(
        session_id=session_id,
        checkpoint_type=CheckpointType.MILESTONE,
        description="Demo milestone achieved",
        task_state={"milestone": "complete"},
    )
    print(f"  Milestone: {milestone.id if milestone else 'failed'}")

    # List checkpoints
    print("\n[List Checkpoints]")
    checkpoints = manager.list_checkpoints(session_id)
    for cp in checkpoints:
        print(f"  - {cp.id} ({cp.checkpoint_type.value}): {cp.description}")

    # Restore from checkpoint
    print("\n[Restore Latest Checkpoint]")
    restored = manager.restore_checkpoint(session_id)
    if restored:
        print(f"  Restored: {restored.id}")
        print(f"  State hash: {restored.state_hash}")

    # Create handoff
    print("\n[Create Handoff]")
    handoff_id = manager.create_handoff(
        session_id=session_id,
        summary="Completed demo iterations with milestone",
        next_steps=["Continue with phase 2", "Run validation tests"],
        critical_context={"demo_complete": True},
    )
    print(f"  Handoff ID: {handoff_id}")

    # Load handoff
    handoff = manager.load_handoff(session_id)
    if handoff:
        print(f"  Summary: {handoff['summary']}")
        print(f"  Next steps: {handoff['next_steps']}")

    # Stats
    print("\n[Persistence Stats]")
    stats = manager.get_stats()
    print(f"  Total sessions: {stats['total_sessions']}")
    print(f"  Total checkpoints: {stats['total_checkpoints']}")
    print(f"  By status: {stats['by_status']}")


if __name__ == "__main__":
    demo()
