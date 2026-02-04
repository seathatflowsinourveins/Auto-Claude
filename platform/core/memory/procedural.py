"""
Procedural Memory - Learned Behavior Patterns (V40 Architecture)

This module implements procedural memory for storing and recalling learned behaviors,
tool sequences, and execution patterns. It enables agents to learn from successful
executions and replay those patterns when similar situations arise.

Architecture:
- Procedure: A sequence of steps representing a learned behavior
- ProcedureStep: Individual action within a procedure
- ProceduralMemory: SQLite-backed storage with FTS5 for pattern matching
- Learning: Extract procedures from successful tool executions
- Recall: Match user queries to known procedures using triggers

V40 Features:
- Confidence-based execution with outcome tracking
- Pattern extraction from tool call sequences
- Similarity-based trigger matching
- Session-aware learning with provenance

Storage Schema:
    ~/.claude/memory/procedural.db
        procedures          # Procedure definitions
        procedure_steps     # Individual steps
        procedure_triggers  # Trigger patterns (FTS5)
        execution_history   # Outcome tracking

Usage:
    from core.memory.procedural import ProceduralMemory

    # Initialize
    pm = ProceduralMemory()

    # Learn from successful execution
    procedure = await pm.learn_procedure(
        name="git_commit_workflow",
        steps=[
            ProcedureStep(action="git_status", params={}),
            ProcedureStep(action="git_add", params={"files": "."}),
            ProcedureStep(action="git_commit", params={"message": "$MESSAGE"}),
        ],
        trigger_patterns=["commit changes", "save my work", "git commit"],
        source_session="session_123"
    )

    # Find matching procedures
    candidates = await pm.recall_procedure("I want to commit my changes")

    # Execute with tracking
    result = await pm.execute_procedure(procedure.id, params={"MESSAGE": "feat: add feature"})
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class ProcedureStatus(str, Enum):
    """Status of a procedure."""
    ACTIVE = "active"          # Available for use
    DEPRECATED = "deprecated"  # Superseded by another procedure
    DISABLED = "disabled"      # Manually disabled
    LEARNING = "learning"      # Still being refined


class StepType(str, Enum):
    """Type of procedure step."""
    TOOL_CALL = "tool_call"       # Call a tool/function
    CONDITION = "condition"        # Conditional branch
    LOOP = "loop"                  # Loop construct
    SUB_PROCEDURE = "sub_procedure"  # Call another procedure
    WAIT = "wait"                  # Wait for condition
    PARALLEL = "parallel"          # Execute steps in parallel


class ExecutionOutcome(str, Enum):
    """Outcome of a procedure execution."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    ABORTED = "aborted"


# Confidence thresholds
MIN_CONFIDENCE_FOR_AUTO_EXECUTE = 0.8
CONFIDENCE_DECAY_PER_FAILURE = 0.1
CONFIDENCE_BOOST_PER_SUCCESS = 0.05
MAX_CONFIDENCE = 1.0
MIN_CONFIDENCE = 0.1


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ProcedureStep:
    """A single step within a procedure."""
    action: str                          # Tool/action name
    params: Dict[str, Any] = field(default_factory=dict)  # Parameters (may contain variables)
    step_type: StepType = StepType.TOOL_CALL
    order: int = 0                       # Execution order
    description: str = ""                # Human-readable description
    required: bool = True                # Is this step mandatory?
    timeout_ms: int = 30000              # Step timeout
    retry_count: int = 0                 # Retry attempts on failure
    condition: Optional[str] = None      # Condition expression for conditional steps
    on_failure: str = "abort"            # abort, skip, retry, fallback
    fallback_action: Optional[str] = None  # Fallback action if primary fails

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "action": self.action,
            "params": self.params,
            "step_type": self.step_type.value,
            "order": self.order,
            "description": self.description,
            "required": self.required,
            "timeout_ms": self.timeout_ms,
            "retry_count": self.retry_count,
            "condition": self.condition,
            "on_failure": self.on_failure,
            "fallback_action": self.fallback_action,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcedureStep":
        """Create from dictionary."""
        return cls(
            action=data["action"],
            params=data.get("params", {}),
            step_type=StepType(data.get("step_type", "tool_call")),
            order=data.get("order", 0),
            description=data.get("description", ""),
            required=data.get("required", True),
            timeout_ms=data.get("timeout_ms", 30000),
            retry_count=data.get("retry_count", 0),
            condition=data.get("condition"),
            on_failure=data.get("on_failure", "abort"),
            fallback_action=data.get("fallback_action"),
        )


@dataclass
class Procedure:
    """A learned procedure representing a sequence of actions."""
    id: str
    name: str
    description: str
    steps: List[ProcedureStep]
    trigger_patterns: List[str]          # Patterns that trigger this procedure
    success_count: int = 0
    failure_count: int = 0
    last_executed: Optional[datetime] = None
    confidence: float = 0.5              # Confidence score (0.0 - 1.0)
    status: ProcedureStatus = ProcedureStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_session: Optional[str] = None  # Session where this was learned
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1                     # Procedure version
    parent_id: Optional[str] = None      # Parent procedure if derived

    @property
    def total_executions(self) -> int:
        """Total number of executions."""
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if self.total_executions == 0:
            return 0.0
        return self.success_count / self.total_executions

    def update_confidence(self, success: bool) -> None:
        """Update confidence based on execution outcome."""
        if success:
            self.success_count += 1
            self.confidence = min(MAX_CONFIDENCE, self.confidence + CONFIDENCE_BOOST_PER_SUCCESS)
        else:
            self.failure_count += 1
            self.confidence = max(MIN_CONFIDENCE, self.confidence - CONFIDENCE_DECAY_PER_FAILURE)
        self.last_executed = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "trigger_patterns": self.trigger_patterns,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "last_executed": self.last_executed.isoformat() if self.last_executed else None,
            "confidence": self.confidence,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "source_session": self.source_session,
            "tags": self.tags,
            "metadata": self.metadata,
            "version": self.version,
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Procedure":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            steps=[ProcedureStep.from_dict(s) for s in data.get("steps", [])],
            trigger_patterns=data.get("trigger_patterns", []),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            last_executed=datetime.fromisoformat(data["last_executed"]) if data.get("last_executed") else None,
            confidence=data.get("confidence", 0.5),
            status=ProcedureStatus(data.get("status", "active")),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(timezone.utc),
            source_session=data.get("source_session"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            version=data.get("version", 1),
            parent_id=data.get("parent_id"),
        )


@dataclass
class ProcedureMatch:
    """Result of matching a query to a procedure."""
    procedure: Procedure
    score: float                 # Match score (0.0 - 1.0)
    matched_trigger: str         # The trigger pattern that matched
    match_type: str              # exact, fuzzy, semantic
    confidence: float            # Combined confidence (score * procedure.confidence)

    @property
    def should_auto_execute(self) -> bool:
        """Whether this match is confident enough for auto-execution."""
        return self.confidence >= MIN_CONFIDENCE_FOR_AUTO_EXECUTE


@dataclass
class ExecutionResult:
    """Result of executing a procedure."""
    procedure_id: str
    outcome: ExecutionOutcome
    steps_completed: int
    steps_total: int
    duration_ms: float
    outputs: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    modified_params: Dict[str, Any] = field(default_factory=dict)
    executed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# SQLite Schema
# =============================================================================

PROCEDURAL_SCHEMA_VERSION = 1

PROCEDURAL_SCHEMA_SQL = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

-- Procedures table
CREATE TABLE IF NOT EXISTS procedures (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    confidence REAL NOT NULL DEFAULT 0.5,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    last_executed TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    source_session TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    parent_id TEXT,
    tags TEXT,  -- JSON array
    metadata TEXT,  -- JSON object
    FOREIGN KEY (parent_id) REFERENCES procedures(id)
);

-- Procedure steps table
CREATE TABLE IF NOT EXISTS procedure_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    procedure_id TEXT NOT NULL,
    action TEXT NOT NULL,
    params TEXT,  -- JSON object
    step_type TEXT NOT NULL DEFAULT 'tool_call',
    step_order INTEGER NOT NULL,
    description TEXT,
    required INTEGER NOT NULL DEFAULT 1,
    timeout_ms INTEGER NOT NULL DEFAULT 30000,
    retry_count INTEGER NOT NULL DEFAULT 0,
    condition TEXT,
    on_failure TEXT NOT NULL DEFAULT 'abort',
    fallback_action TEXT,
    FOREIGN KEY (procedure_id) REFERENCES procedures(id) ON DELETE CASCADE
);

-- Trigger patterns with FTS5 for fast matching
CREATE VIRTUAL TABLE IF NOT EXISTS procedure_triggers USING fts5(
    procedure_id,
    pattern,
    tokenize='porter unicode61'
);

-- Execution history for learning
CREATE TABLE IF NOT EXISTS execution_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    procedure_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    steps_completed INTEGER NOT NULL,
    steps_total INTEGER NOT NULL,
    duration_ms REAL NOT NULL,
    outputs TEXT,  -- JSON
    errors TEXT,  -- JSON array
    executed_at TEXT NOT NULL,
    session_id TEXT,
    FOREIGN KEY (procedure_id) REFERENCES procedures(id) ON DELETE CASCADE
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_procedures_status ON procedures(status);
CREATE INDEX IF NOT EXISTS idx_procedures_confidence ON procedures(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_procedures_name ON procedures(name);
CREATE INDEX IF NOT EXISTS idx_steps_procedure ON procedure_steps(procedure_id);
CREATE INDEX IF NOT EXISTS idx_steps_order ON procedure_steps(procedure_id, step_order);
CREATE INDEX IF NOT EXISTS idx_history_procedure ON execution_history(procedure_id);
CREATE INDEX IF NOT EXISTS idx_history_executed ON execution_history(executed_at DESC);
"""

PROCEDURAL_TRIGGERS_SQL = """
-- Sync FTS on procedure delete
CREATE TRIGGER IF NOT EXISTS procedures_ad AFTER DELETE ON procedures BEGIN
    DELETE FROM procedure_triggers WHERE procedure_id = OLD.id;
END;
"""


# =============================================================================
# PROCEDURAL MEMORY IMPLEMENTATION
# =============================================================================

class ProceduralMemory:
    """
    SQLite-backed procedural memory for learned behaviors.

    Provides:
    - Procedure learning from tool sequences
    - Pattern-based trigger matching
    - Confidence-based execution
    - Outcome tracking and learning

    Storage: ~/.claude/memory/procedural.db
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        embedding_provider: Optional[Callable[[str], List[float]]] = None
    ) -> None:
        """Initialize procedural memory.

        Args:
            db_path: Path to SQLite database. Defaults to ~/.claude/memory/procedural.db
            embedding_provider: Optional function to generate embeddings for semantic matching
        """
        self.db_path = db_path or Path.home() / ".claude" / "memory" / "procedural.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_provider = embedding_provider
        self._connection: Optional[sqlite3.Connection] = None
        self._init_db()
        logger.info(f"Procedural memory initialized at {self.db_path}")

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            )
            if not cursor.fetchone():
                conn.executescript(PROCEDURAL_SCHEMA_SQL)
                conn.executescript(PROCEDURAL_TRIGGERS_SQL)
                conn.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (PROCEDURAL_SCHEMA_VERSION, datetime.now(timezone.utc).isoformat())
                )
                conn.commit()
                logger.info("Created procedural memory database with schema v%d", PROCEDURAL_SCHEMA_VERSION)

    @contextmanager
    def _get_connection(self):
        """Get database connection with context management."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute("PRAGMA foreign_keys=ON")

        try:
            yield self._connection
        except Exception:
            self._connection.rollback()
            raise

    def _generate_id(self, name: str) -> str:
        """Generate a unique procedure ID."""
        timestamp = str(time.time())
        hash_input = f"{name}:{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    # =========================================================================
    # LEARNING API
    # =========================================================================

    async def learn_procedure(
        self,
        name: str,
        steps: List[ProcedureStep],
        trigger_patterns: List[str],
        description: str = "",
        source_session: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        initial_confidence: float = 0.5
    ) -> Procedure:
        """Learn a new procedure from a sequence of steps.

        Args:
            name: Procedure name
            steps: List of procedure steps
            trigger_patterns: Patterns that should trigger this procedure
            description: Human-readable description
            source_session: Session ID where this was learned
            tags: Optional tags for categorization
            metadata: Optional metadata
            initial_confidence: Starting confidence (default 0.5)

        Returns:
            The created Procedure
        """
        procedure_id = self._generate_id(name)
        now = datetime.now(timezone.utc)

        # Set step order
        for i, step in enumerate(steps):
            step.order = i

        procedure = Procedure(
            id=procedure_id,
            name=name,
            description=description,
            steps=steps,
            trigger_patterns=trigger_patterns,
            confidence=initial_confidence,
            source_session=source_session,
            tags=tags or [],
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )

        await self._store_procedure(procedure)
        logger.info(f"Learned procedure: {name} (id={procedure_id}, {len(steps)} steps)")
        return procedure

    async def learn_from_execution(
        self,
        tool_calls: List[Dict[str, Any]],
        success: bool,
        name: Optional[str] = None,
        trigger_patterns: Optional[List[str]] = None,
        source_session: Optional[str] = None
    ) -> Optional[Procedure]:
        """Extract a procedure from a successful tool execution sequence.

        Args:
            tool_calls: List of tool calls with name, params, and result
            success: Whether the overall execution was successful
            name: Optional procedure name (auto-generated if not provided)
            trigger_patterns: Patterns to associate with this procedure
            source_session: Session ID

        Returns:
            Created Procedure if extraction successful, None otherwise
        """
        if not success or len(tool_calls) < 2:
            return None

        # Extract steps from tool calls
        steps = []
        for i, call in enumerate(tool_calls):
            step = ProcedureStep(
                action=call.get("name", call.get("tool", "unknown")),
                params=self._parameterize(call.get("params", call.get("arguments", {}))),
                step_type=StepType.TOOL_CALL,
                order=i,
                description=call.get("description", ""),
            )
            steps.append(step)

        # Generate name from tool sequence if not provided
        if name is None:
            action_names = [s.action for s in steps[:3]]
            name = "_".join(action_names) + "_workflow"

        # Generate trigger patterns if not provided
        if trigger_patterns is None:
            trigger_patterns = self._extract_triggers(steps)

        return await self.learn_procedure(
            name=name,
            steps=steps,
            trigger_patterns=trigger_patterns,
            description=f"Learned from {len(tool_calls)} tool calls",
            source_session=source_session,
            initial_confidence=0.6 if success else 0.3
        )

    def _parameterize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert concrete values to parameterized variables where appropriate.

        This allows procedures to be reused with different inputs.
        """
        parameterized = {}
        for key, value in params.items():
            if isinstance(value, str) and len(value) > 50:
                # Long strings likely to be user-specific, make them variables
                parameterized[key] = f"${key.upper()}"
            elif key in ("message", "content", "text", "query", "prompt"):
                # Common user-input fields
                parameterized[key] = f"${key.upper()}"
            else:
                parameterized[key] = value
        return parameterized

    def _extract_triggers(self, steps: List[ProcedureStep]) -> List[str]:
        """Extract trigger patterns from procedure steps."""
        triggers = []
        action_names = [s.action for s in steps]

        # Add action-based triggers
        if action_names:
            triggers.append(" ".join(action_names[:2]))

        # Add common synonyms for known actions
        action_triggers = {
            "git_commit": ["commit changes", "save my work", "create commit"],
            "git_push": ["push changes", "push to remote", "sync to github"],
            "file_write": ["write file", "save file", "create file"],
            "file_read": ["read file", "open file", "show file"],
            "search": ["search for", "find", "look for"],
            "test_run": ["run tests", "test code", "verify tests"],
        }

        for action in action_names:
            if action in action_triggers:
                triggers.extend(action_triggers[action])

        return triggers[:10]  # Limit to 10 triggers

    async def _store_procedure(self, procedure: Procedure) -> None:
        """Store a procedure in the database."""
        with self._get_connection() as conn:
            # Insert procedure
            conn.execute("""
                INSERT OR REPLACE INTO procedures (
                    id, name, description, status, confidence,
                    success_count, failure_count, last_executed,
                    created_at, updated_at, source_session,
                    version, parent_id, tags, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                procedure.id,
                procedure.name,
                procedure.description,
                procedure.status.value,
                procedure.confidence,
                procedure.success_count,
                procedure.failure_count,
                procedure.last_executed.isoformat() if procedure.last_executed else None,
                procedure.created_at.isoformat(),
                procedure.updated_at.isoformat(),
                procedure.source_session,
                procedure.version,
                procedure.parent_id,
                json.dumps(procedure.tags),
                json.dumps(procedure.metadata),
            ))

            # Delete old steps and triggers
            conn.execute("DELETE FROM procedure_steps WHERE procedure_id = ?", (procedure.id,))
            conn.execute("DELETE FROM procedure_triggers WHERE procedure_id = ?", (procedure.id,))

            # Insert steps
            for step in procedure.steps:
                conn.execute("""
                    INSERT INTO procedure_steps (
                        procedure_id, action, params, step_type, step_order,
                        description, required, timeout_ms, retry_count,
                        condition, on_failure, fallback_action
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    procedure.id,
                    step.action,
                    json.dumps(step.params),
                    step.step_type.value,
                    step.order,
                    step.description,
                    1 if step.required else 0,
                    step.timeout_ms,
                    step.retry_count,
                    step.condition,
                    step.on_failure,
                    step.fallback_action,
                ))

            # Insert trigger patterns
            for pattern in procedure.trigger_patterns:
                conn.execute(
                    "INSERT INTO procedure_triggers (procedure_id, pattern) VALUES (?, ?)",
                    (procedure.id, pattern)
                )

            conn.commit()

    # =========================================================================
    # RECALL API
    # =========================================================================

    async def recall_procedure(
        self,
        query: str,
        limit: int = 5,
        min_confidence: float = 0.3,
        include_deprecated: bool = False
    ) -> List[ProcedureMatch]:
        """Find procedures matching a query.

        Args:
            query: User query or intent
            limit: Maximum number of results
            min_confidence: Minimum procedure confidence
            include_deprecated: Whether to include deprecated procedures

        Returns:
            List of ProcedureMatch objects, ranked by relevance
        """
        matches: List[ProcedureMatch] = []

        with self._get_connection() as conn:
            # FTS5 search on trigger patterns
            try:
                cursor = conn.execute("""
                    SELECT p.*, pt.pattern, bm25(procedure_triggers) as score
                    FROM procedures p
                    JOIN procedure_triggers pt ON p.id = pt.procedure_id
                    WHERE procedure_triggers MATCH ?
                    AND p.confidence >= ?
                    AND (p.status = 'active' OR (? AND p.status = 'deprecated'))
                    ORDER BY score
                    LIMIT ?
                """, (query, min_confidence, include_deprecated, limit * 2))

                for row in cursor.fetchall():
                    procedure = await self._load_procedure(row["id"])
                    if procedure:
                        score = min(1.0, -row["score"])  # BM25 scores are negative
                        matches.append(ProcedureMatch(
                            procedure=procedure,
                            score=score,
                            matched_trigger=row["pattern"],
                            match_type="fuzzy",
                            confidence=score * procedure.confidence
                        ))
            except sqlite3.OperationalError:
                # FTS may not have data yet, fall back to LIKE search
                cursor = conn.execute("""
                    SELECT DISTINCT p.id
                    FROM procedures p
                    JOIN procedure_triggers pt ON p.id = pt.procedure_id
                    WHERE pt.pattern LIKE ?
                    AND p.confidence >= ?
                    AND (p.status = 'active' OR (? AND p.status = 'deprecated'))
                    LIMIT ?
                """, (f"%{query}%", min_confidence, include_deprecated, limit))

                for row in cursor.fetchall():
                    procedure = await self._load_procedure(row["id"])
                    if procedure:
                        matches.append(ProcedureMatch(
                            procedure=procedure,
                            score=0.5,
                            matched_trigger=query,
                            match_type="like",
                            confidence=0.5 * procedure.confidence
                        ))

        # Add semantic search if embedding provider available
        if self.embedding_provider and len(matches) < limit:
            semantic_matches = await self._semantic_recall(query, limit - len(matches), min_confidence)
            existing_ids = {m.procedure.id for m in matches}
            for match in semantic_matches:
                if match.procedure.id not in existing_ids:
                    matches.append(match)

        # Sort by confidence and limit
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches[:limit]

    async def _semantic_recall(
        self,
        query: str,
        limit: int,
        min_confidence: float
    ) -> List[ProcedureMatch]:
        """Semantic search for procedures using embeddings."""
        if not self.embedding_provider:
            return []

        try:
            query_embedding = self.embedding_provider(query)
        except Exception as e:
            logger.warning(f"Failed to generate query embedding: {e}")
            return []

        matches = []
        procedures = await self.list_procedures(status=ProcedureStatus.ACTIVE)

        for procedure in procedures:
            if procedure.confidence < min_confidence:
                continue

            # Compare with trigger pattern embeddings
            best_score = 0.0
            best_trigger = ""
            for trigger in procedure.trigger_patterns:
                try:
                    trigger_embedding = self.embedding_provider(trigger)
                    similarity = self._cosine_similarity(query_embedding, trigger_embedding)
                    if similarity > best_score:
                        best_score = similarity
                        best_trigger = trigger
                except Exception:
                    continue

            if best_score > 0.5:  # Threshold
                matches.append(ProcedureMatch(
                    procedure=procedure,
                    score=best_score,
                    matched_trigger=best_trigger,
                    match_type="semantic",
                    confidence=best_score * procedure.confidence
                ))

        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches[:limit]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    async def _load_procedure(self, procedure_id: str) -> Optional[Procedure]:
        """Load a procedure from the database."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM procedures WHERE id = ?",
                (procedure_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None

            # Load steps
            steps_cursor = conn.execute(
                "SELECT * FROM procedure_steps WHERE procedure_id = ? ORDER BY step_order",
                (procedure_id,)
            )
            steps = []
            for step_row in steps_cursor.fetchall():
                steps.append(ProcedureStep(
                    action=step_row["action"],
                    params=json.loads(step_row["params"]) if step_row["params"] else {},
                    step_type=StepType(step_row["step_type"]),
                    order=step_row["step_order"],
                    description=step_row["description"] or "",
                    required=bool(step_row["required"]),
                    timeout_ms=step_row["timeout_ms"],
                    retry_count=step_row["retry_count"],
                    condition=step_row["condition"],
                    on_failure=step_row["on_failure"],
                    fallback_action=step_row["fallback_action"],
                ))

            # Load triggers
            triggers_cursor = conn.execute(
                "SELECT pattern FROM procedure_triggers WHERE procedure_id = ?",
                (procedure_id,)
            )
            triggers = [r["pattern"] for r in triggers_cursor.fetchall()]

            return Procedure(
                id=row["id"],
                name=row["name"],
                description=row["description"] or "",
                steps=steps,
                trigger_patterns=triggers,
                success_count=row["success_count"],
                failure_count=row["failure_count"],
                last_executed=datetime.fromisoformat(row["last_executed"]) if row["last_executed"] else None,
                confidence=row["confidence"],
                status=ProcedureStatus(row["status"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                source_session=row["source_session"],
                tags=json.loads(row["tags"]) if row["tags"] else [],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                version=row["version"],
                parent_id=row["parent_id"],
            )

    # =========================================================================
    # EXECUTION API
    # =========================================================================

    async def execute_procedure(
        self,
        procedure_id: str,
        params: Optional[Dict[str, Any]] = None,
        executor: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        session_id: Optional[str] = None,
        dry_run: bool = False
    ) -> ExecutionResult:
        """Execute a procedure with parameter substitution.

        Args:
            procedure_id: ID of the procedure to execute
            params: Parameters to substitute into steps
            executor: Function to execute each step (action, params) -> result
            session_id: Session ID for tracking
            dry_run: If True, don't actually execute, just validate

        Returns:
            ExecutionResult with outcome and details
        """
        procedure = await self._load_procedure(procedure_id)
        if not procedure:
            return ExecutionResult(
                procedure_id=procedure_id,
                outcome=ExecutionOutcome.FAILURE,
                steps_completed=0,
                steps_total=0,
                duration_ms=0,
                errors=["Procedure not found"]
            )

        params = params or {}
        start_time = time.time()
        outputs: Dict[str, Any] = {}
        errors: List[str] = []
        steps_completed = 0

        for step in procedure.steps:
            if dry_run:
                steps_completed += 1
                continue

            # Substitute parameters
            step_params = self._substitute_params(step.params, params, outputs)

            # Execute step
            if executor:
                try:
                    result = await self._execute_step(executor, step, step_params)
                    outputs[f"step_{step.order}"] = result
                    steps_completed += 1
                except Exception as e:
                    error_msg = f"Step {step.order} ({step.action}) failed: {str(e)}"
                    errors.append(error_msg)
                    logger.warning(error_msg)

                    if step.on_failure == "abort":
                        break
                    elif step.on_failure == "skip":
                        continue
                    elif step.on_failure == "fallback" and step.fallback_action:
                        try:
                            result = await self._execute_step(
                                executor,
                                ProcedureStep(action=step.fallback_action, params=step_params),
                                step_params
                            )
                            outputs[f"step_{step.order}_fallback"] = result
                            steps_completed += 1
                        except Exception as fe:
                            errors.append(f"Fallback failed: {str(fe)}")
                            if step.required:
                                break
            else:
                # No executor, just validate
                steps_completed += 1

        duration_ms = (time.time() - start_time) * 1000

        # Determine outcome
        if steps_completed == len(procedure.steps) and not errors:
            outcome = ExecutionOutcome.SUCCESS
        elif steps_completed > 0:
            outcome = ExecutionOutcome.PARTIAL
        else:
            outcome = ExecutionOutcome.FAILURE

        result = ExecutionResult(
            procedure_id=procedure_id,
            outcome=outcome,
            steps_completed=steps_completed,
            steps_total=len(procedure.steps),
            duration_ms=duration_ms,
            outputs=outputs,
            errors=errors,
            modified_params=params
        )

        # Record execution and update confidence
        if not dry_run:
            await self._record_execution(result, session_id)
            procedure.update_confidence(outcome == ExecutionOutcome.SUCCESS)
            await self._store_procedure(procedure)

        return result

    async def _execute_step(
        self,
        executor: Callable[[str, Dict[str, Any]], Any],
        step: ProcedureStep,
        params: Dict[str, Any]
    ) -> Any:
        """Execute a single step with retries."""
        last_error = None
        for attempt in range(step.retry_count + 1):
            try:
                return await executor(step.action, params)
            except Exception as e:
                last_error = e
                if attempt < step.retry_count:
                    logger.debug(f"Retry {attempt + 1}/{step.retry_count} for {step.action}")
                    continue
        raise last_error or Exception("Unknown execution error")

    def _substitute_params(
        self,
        step_params: Dict[str, Any],
        user_params: Dict[str, Any],
        outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Substitute variables in step parameters."""
        result = {}
        for key, value in step_params.items():
            if isinstance(value, str) and value.startswith("$"):
                var_name = value[1:]
                if var_name in user_params:
                    result[key] = user_params[var_name]
                elif var_name.lower() in user_params:
                    result[key] = user_params[var_name.lower()]
                elif f"step_{var_name}" in outputs:
                    result[key] = outputs[f"step_{var_name}"]
                else:
                    result[key] = value  # Keep as variable
            else:
                result[key] = value
        return result

    async def _record_execution(
        self,
        result: ExecutionResult,
        session_id: Optional[str]
    ) -> None:
        """Record execution in history."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO execution_history (
                    procedure_id, outcome, steps_completed, steps_total,
                    duration_ms, outputs, errors, executed_at, session_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.procedure_id,
                result.outcome.value,
                result.steps_completed,
                result.steps_total,
                result.duration_ms,
                json.dumps(result.outputs),
                json.dumps(result.errors),
                result.executed_at.isoformat(),
                session_id,
            ))
            conn.commit()

    # =========================================================================
    # MANAGEMENT API
    # =========================================================================

    async def get_procedure(self, procedure_id: str) -> Optional[Procedure]:
        """Get a procedure by ID."""
        return await self._load_procedure(procedure_id)

    async def list_procedures(
        self,
        status: Optional[ProcedureStatus] = None,
        limit: int = 100,
        order_by: str = "confidence"
    ) -> List[Procedure]:
        """List all procedures.

        Args:
            status: Filter by status
            limit: Maximum number to return
            order_by: Sort field (confidence, created_at, success_count)

        Returns:
            List of procedures
        """
        order_map = {
            "confidence": "confidence DESC",
            "created_at": "created_at DESC",
            "success_count": "success_count DESC",
            "name": "name ASC",
        }
        order_clause = order_map.get(order_by, "confidence DESC")

        with self._get_connection() as conn:
            if status:
                cursor = conn.execute(
                    f"SELECT id FROM procedures WHERE status = ? ORDER BY {order_clause} LIMIT ?",
                    (status.value, limit)
                )
            else:
                cursor = conn.execute(
                    f"SELECT id FROM procedures ORDER BY {order_clause} LIMIT ?",
                    (limit,)
                )

            procedures = []
            for row in cursor.fetchall():
                procedure = await self._load_procedure(row["id"])
                if procedure:
                    procedures.append(procedure)

            return procedures

    async def delete_procedure(self, procedure_id: str) -> bool:
        """Delete a procedure.

        Args:
            procedure_id: ID of procedure to delete

        Returns:
            True if deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM procedures WHERE id = ?",
                (procedure_id,)
            )
            conn.commit()
            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted procedure: {procedure_id}")
            return deleted

    async def deprecate_procedure(
        self,
        procedure_id: str,
        replacement_id: Optional[str] = None
    ) -> bool:
        """Mark a procedure as deprecated.

        Args:
            procedure_id: ID of procedure to deprecate
            replacement_id: Optional ID of replacement procedure

        Returns:
            True if deprecated, False if not found
        """
        procedure = await self._load_procedure(procedure_id)
        if not procedure:
            return False

        procedure.status = ProcedureStatus.DEPRECATED
        if replacement_id:
            procedure.metadata["replaced_by"] = replacement_id
        procedure.updated_at = datetime.now(timezone.utc)

        await self._store_procedure(procedure)
        logger.info(f"Deprecated procedure: {procedure_id}")
        return True

    async def get_execution_history(
        self,
        procedure_id: Optional[str] = None,
        limit: int = 50
    ) -> List[ExecutionResult]:
        """Get execution history.

        Args:
            procedure_id: Filter by procedure ID
            limit: Maximum number of results

        Returns:
            List of ExecutionResult
        """
        with self._get_connection() as conn:
            if procedure_id:
                cursor = conn.execute("""
                    SELECT * FROM execution_history
                    WHERE procedure_id = ?
                    ORDER BY executed_at DESC
                    LIMIT ?
                """, (procedure_id, limit))
            else:
                cursor = conn.execute("""
                    SELECT * FROM execution_history
                    ORDER BY executed_at DESC
                    LIMIT ?
                """, (limit,))

            results = []
            for row in cursor.fetchall():
                results.append(ExecutionResult(
                    procedure_id=row["procedure_id"],
                    outcome=ExecutionOutcome(row["outcome"]),
                    steps_completed=row["steps_completed"],
                    steps_total=row["steps_total"],
                    duration_ms=row["duration_ms"],
                    outputs=json.loads(row["outputs"]) if row["outputs"] else {},
                    errors=json.loads(row["errors"]) if row["errors"] else [],
                    executed_at=datetime.fromisoformat(row["executed_at"]),
                ))
            return results

    async def get_stats(self) -> Dict[str, Any]:
        """Get procedural memory statistics."""
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM procedures").fetchone()[0]
            active = conn.execute(
                "SELECT COUNT(*) FROM procedures WHERE status = 'active'"
            ).fetchone()[0]
            executions = conn.execute("SELECT COUNT(*) FROM execution_history").fetchone()[0]
            successes = conn.execute(
                "SELECT COUNT(*) FROM execution_history WHERE outcome = 'success'"
            ).fetchone()[0]

            avg_confidence = conn.execute(
                "SELECT AVG(confidence) FROM procedures WHERE status = 'active'"
            ).fetchone()[0] or 0.0

            return {
                "total_procedures": total,
                "active_procedures": active,
                "total_executions": executions,
                "successful_executions": successes,
                "success_rate": successes / executions if executions > 0 else 0.0,
                "average_confidence": round(avg_confidence, 3),
                "storage_path": str(self.db_path),
                "db_size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
            }

    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None


# =============================================================================
# SINGLETON AND HOOKS
# =============================================================================

_procedural_memory: Optional[ProceduralMemory] = None


def get_procedural_memory(
    embedding_provider: Optional[Callable[[str], List[float]]] = None
) -> ProceduralMemory:
    """Get or create the singleton ProceduralMemory instance."""
    global _procedural_memory
    if _procedural_memory is None:
        _procedural_memory = ProceduralMemory(embedding_provider=embedding_provider)
    return _procedural_memory


# Hook functions for integration with memory system
async def learn_procedure(
    name: str,
    steps: List[ProcedureStep],
    trigger_patterns: List[str],
    **kwargs
) -> Procedure:
    """Hook: Learn a new procedure."""
    pm = get_procedural_memory()
    return await pm.learn_procedure(name, steps, trigger_patterns, **kwargs)


async def recall_procedure(
    query: str,
    limit: int = 5,
    min_confidence: float = 0.3
) -> List[ProcedureMatch]:
    """Hook: Recall procedures matching a query."""
    pm = get_procedural_memory()
    return await pm.recall_procedure(query, limit, min_confidence)


async def execute_procedure(
    procedure_id: str,
    params: Optional[Dict[str, Any]] = None,
    executor: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    **kwargs
) -> ExecutionResult:
    """Hook: Execute a procedure."""
    pm = get_procedural_memory()
    return await pm.execute_procedure(procedure_id, params, executor, **kwargs)


__all__ = [
    # Enums
    "ProcedureStatus",
    "StepType",
    "ExecutionOutcome",
    # Data classes
    "ProcedureStep",
    "Procedure",
    "ProcedureMatch",
    "ExecutionResult",
    # Main class
    "ProceduralMemory",
    # Singleton
    "get_procedural_memory",
    # Hooks
    "learn_procedure",
    "recall_procedure",
    "execute_procedure",
    # Constants
    "MIN_CONFIDENCE_FOR_AUTO_EXECUTE",
]
