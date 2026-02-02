#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pydantic>=2.0.0",
# ]
# ///
"""
Auto-Claude Cooperation System

Enables systematic cooperation between Claude instances across sessions.
Implements the Session Handoff Protocol for transferring context and progress.

Key Features:
1. Session Handoff Protocol - Transfer work between sessions
2. Decision Tracking - Record and share key decisions
3. Constraint Discovery - Share learned constraints
4. Progress Synchronization - Track task completion

Based on multi-agent coordination patterns from Claude SDK.

Usage:
    from cooperation import SessionHandoff, CooperationManager

    # At end of session
    handoff = manager.create_handoff(
        completed_steps=["Research", "Design"],
        current_step="Implementation",
        remaining_steps=["Testing", "Deployment"]
    )

    # At start of next session
    context = manager.load_handoff(handoff_id)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# =============================================================================
# Decision Tracking
# =============================================================================

class Decision(BaseModel):
    """
    A key decision made during development.

    Captures what was decided, why, and any alternatives considered.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    description: str = Field(..., description="What needed to be decided")
    choice: str = Field(..., description="What was chosen")
    rationale: str = Field(..., description="Why this was chosen")
    alternatives: List[str] = Field(default_factory=list, description="Other options considered")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: str = Field(default="high", description="high, medium, low")

    def to_context_string(self) -> str:
        """Format for context injection."""
        return f"- **{self.description}**: {self.choice} (reason: {self.rationale})"


# =============================================================================
# Constraint Discovery
# =============================================================================

class Constraint(BaseModel):
    """
    A constraint discovered during development.

    These are things we learned that limit what's possible or advisable.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    description: str = Field(..., description="What the constraint is")
    category: str = Field(default="technical", description="technical, business, time, resource")
    severity: str = Field(default="medium", description="critical, high, medium, low")
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = Field(default="observation", description="How we learned this")

    def to_context_string(self) -> str:
        """Format for context injection."""
        return f"- [{self.severity.upper()}] {self.description}"


# =============================================================================
# Session Handoff
# =============================================================================

class SessionHandoff(BaseModel):
    """
    Context for transferring work between Claude sessions.

    This is the core unit of cooperation - it captures everything needed
    for a new session to seamlessly continue work.
    """

    # Identity
    id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str = Field(..., description="Identifier for the overall task")
    parent_session: str = Field(default="", description="Session that created this handoff")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Progress State
    completed_steps: List[str] = Field(default_factory=list)
    current_step: str = Field(default="")
    remaining_steps: List[str] = Field(default_factory=list)
    progress_percent: int = Field(default=0, ge=0, le=100)

    # Knowledge Transfer
    key_decisions: List[Decision] = Field(default_factory=list)
    discovered_constraints: List[Constraint] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    important_findings: List[str] = Field(default_factory=list)

    # File State
    modified_files: List[str] = Field(default_factory=list)
    created_files: List[str] = Field(default_factory=list)
    deleted_files: List[str] = Field(default_factory=list)

    # Memory Snapshot
    core_memory_export: Dict[str, str] = Field(default_factory=dict)
    relevant_facts: List[Dict[str, Any]] = Field(default_factory=list)

    # Execution Context
    last_error: Optional[str] = Field(default=None)
    last_command: Optional[str] = Field(default=None)
    environment_notes: List[str] = Field(default_factory=list)

    def to_context_string(self) -> str:
        """Generate comprehensive context string for next session."""
        sections = []

        # Header
        sections.append(f"""## Session Handoff
**Task ID**: {self.task_id}
**Created**: {self.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}
**Progress**: {self.progress_percent}%
""")

        # Progress
        sections.append(f"""### Progress State
**Completed**: {', '.join(self.completed_steps) if self.completed_steps else 'None'}
**Current**: {self.current_step or 'Not started'}
**Remaining**: {', '.join(self.remaining_steps) if self.remaining_steps else 'None'}
""")

        # Key Decisions
        if self.key_decisions:
            decisions_str = "\n".join(d.to_context_string() for d in self.key_decisions)
            sections.append(f"""### Key Decisions Made
{decisions_str}
""")

        # Constraints
        if self.discovered_constraints:
            constraints_str = "\n".join(c.to_context_string() for c in self.discovered_constraints)
            sections.append(f"""### Discovered Constraints
{constraints_str}
""")

        # Open Questions
        if self.open_questions:
            questions_str = "\n".join(f"- {q}" for q in self.open_questions)
            sections.append(f"""### Open Questions
{questions_str}
""")

        # Important Findings
        if self.important_findings:
            findings_str = "\n".join(f"- {f}" for f in self.important_findings)
            sections.append(f"""### Important Findings
{findings_str}
""")

        # File Changes
        file_changes = []
        if self.created_files:
            file_changes.append(f"**Created**: {', '.join(self.created_files)}")
        if self.modified_files:
            file_changes.append(f"**Modified**: {', '.join(self.modified_files)}")
        if self.deleted_files:
            file_changes.append(f"**Deleted**: {', '.join(self.deleted_files)}")

        if file_changes:
            sections.append(f"""### File Changes
{chr(10).join(file_changes)}
""")

        # Errors/Issues
        if self.last_error:
            sections.append(f"""### Last Error
```
{self.last_error}
```
""")

        # Environment Notes
        if self.environment_notes:
            notes_str = "\n".join(f"- {n}" for n in self.environment_notes)
            sections.append(f"""### Environment Notes
{notes_str}
""")

        return "\n".join(sections)

    def to_compact_summary(self) -> str:
        """Generate a compact one-liner summary."""
        return (
            f"Task {self.task_id}: {self.progress_percent}% complete, "
            f"{len(self.completed_steps)} done, {len(self.remaining_steps)} remaining, "
            f"{len(self.key_decisions)} decisions, {len(self.discovered_constraints)} constraints"
        )


# =============================================================================
# Cooperation Manager
# =============================================================================

class CooperationManager:
    """
    Manages cooperation between Claude sessions.

    Handles creating, storing, and loading session handoffs.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            storage_path = Path.home() / ".uap" / "cooperation"

        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.current_session_id = str(uuid4())
        self.decisions: List[Decision] = []
        self.constraints: List[Constraint] = []
        self.questions: List[str] = []
        self.findings: List[str] = []
        self.files_created: List[str] = []
        self.files_modified: List[str] = []
        self.files_deleted: List[str] = []

    # -------------------------------------------------------------------------
    # Recording During Session
    # -------------------------------------------------------------------------

    def record_decision(
        self,
        description: str,
        choice: str,
        rationale: str,
        alternatives: Optional[List[str]] = None,
        confidence: str = "high"
    ) -> Decision:
        """Record a key decision made during this session."""
        decision = Decision(
            description=description,
            choice=choice,
            rationale=rationale,
            alternatives=alternatives or [],
            confidence=confidence
        )
        self.decisions.append(decision)
        return decision

    def record_constraint(
        self,
        description: str,
        category: str = "technical",
        severity: str = "medium",
        source: str = "observation"
    ) -> Constraint:
        """Record a constraint discovered during this session."""
        constraint = Constraint(
            description=description,
            category=category,
            severity=severity,
            source=source
        )
        self.constraints.append(constraint)
        return constraint

    def record_question(self, question: str) -> None:
        """Record an open question that needs resolution."""
        if question not in self.questions:
            self.questions.append(question)

    def record_finding(self, finding: str) -> None:
        """Record an important finding."""
        if finding not in self.findings:
            self.findings.append(finding)

    def record_file_created(self, file_path: str) -> None:
        """Record that a file was created."""
        if file_path not in self.files_created:
            self.files_created.append(file_path)

    def record_file_modified(self, file_path: str) -> None:
        """Record that a file was modified."""
        if file_path not in self.files_modified and file_path not in self.files_created:
            self.files_modified.append(file_path)

    def record_file_deleted(self, file_path: str) -> None:
        """Record that a file was deleted."""
        if file_path not in self.files_deleted:
            self.files_deleted.append(file_path)
        # Remove from created/modified if present
        if file_path in self.files_created:
            self.files_created.remove(file_path)
        if file_path in self.files_modified:
            self.files_modified.remove(file_path)

    # -------------------------------------------------------------------------
    # Creating Handoffs
    # -------------------------------------------------------------------------

    def create_handoff(
        self,
        task_id: str,
        completed_steps: List[str],
        current_step: str,
        remaining_steps: List[str],
        core_memory_export: Optional[Dict[str, str]] = None,
        last_error: Optional[str] = None,
        last_command: Optional[str] = None,
        environment_notes: Optional[List[str]] = None
    ) -> SessionHandoff:
        """Create a session handoff for the next session."""

        # Calculate progress
        total_steps = len(completed_steps) + 1 + len(remaining_steps)  # +1 for current
        progress = int((len(completed_steps) / total_steps) * 100) if total_steps > 0 else 0

        handoff = SessionHandoff(
            task_id=task_id,
            parent_session=self.current_session_id,
            completed_steps=completed_steps,
            current_step=current_step,
            remaining_steps=remaining_steps,
            progress_percent=progress,
            key_decisions=self.decisions.copy(),
            discovered_constraints=self.constraints.copy(),
            open_questions=self.questions.copy(),
            important_findings=self.findings.copy(),
            modified_files=self.files_modified.copy(),
            created_files=self.files_created.copy(),
            deleted_files=self.files_deleted.copy(),
            core_memory_export=core_memory_export or {},
            last_error=last_error,
            last_command=last_command,
            environment_notes=environment_notes or []
        )

        # Save to disk
        self._save_handoff(handoff)

        return handoff

    def _save_handoff(self, handoff: SessionHandoff) -> Path:
        """Save handoff to disk."""
        handoff_dir = self.storage_path / handoff.task_id
        handoff_dir.mkdir(parents=True, exist_ok=True)

        file_path = handoff_dir / f"{handoff.id}.json"
        with open(file_path, "w") as f:
            json.dump(handoff.model_dump(mode="json"), f, indent=2, default=str)

        # Also save as "latest" for easy access
        latest_path = handoff_dir / "latest.json"
        with open(latest_path, "w") as f:
            json.dump(handoff.model_dump(mode="json"), f, indent=2, default=str)

        return file_path

    # -------------------------------------------------------------------------
    # Loading Handoffs
    # -------------------------------------------------------------------------

    def load_handoff(self, task_id: str, handoff_id: Optional[str] = None) -> Optional[SessionHandoff]:
        """Load a handoff by task ID (latest) or specific handoff ID."""
        handoff_dir = self.storage_path / task_id

        if not handoff_dir.exists():
            return None

        if handoff_id:
            file_path = handoff_dir / f"{handoff_id}.json"
        else:
            file_path = handoff_dir / "latest.json"

        if not file_path.exists():
            return None

        with open(file_path) as f:
            data = json.load(f)

        # Convert back to objects
        data["key_decisions"] = [Decision(**d) for d in data.get("key_decisions", [])]
        data["discovered_constraints"] = [Constraint(**c) for c in data.get("discovered_constraints", [])]

        return SessionHandoff(**data)

    def list_handoffs(self, task_id: str) -> List[str]:
        """List all handoff IDs for a task."""
        handoff_dir = self.storage_path / task_id

        if not handoff_dir.exists():
            return []

        return [
            f.stem for f in handoff_dir.glob("*.json")
            if f.stem != "latest"
        ]

    def get_latest_context(self, task_id: str) -> Optional[str]:
        """Get context string from latest handoff for a task."""
        handoff = self.load_handoff(task_id)
        if handoff:
            return handoff.to_context_string()
        return None


# =============================================================================
# Task Coordination (Multi-Agent)
# =============================================================================

@dataclass
class WorkerAssignment:
    """Assignment of work to a worker agent."""

    worker_id: str
    task_description: str
    expected_outputs: List[str]
    dependencies: List[str]  # Other worker IDs this depends on
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[str] = None


class TaskCoordinator:
    """
    Coordinates work between multiple Claude worker agents.

    Pattern:
    1. Coordinator decomposes task into subtasks
    2. Workers execute subtasks in parallel (respecting dependencies)
    3. Results aggregated and synthesized by coordinator
    """

    def __init__(self, task_id: str, storage_path: Optional[Path] = None):
        self.task_id = task_id

        if storage_path is None:
            storage_path = Path.home() / ".uap" / "coordination"

        self.storage_path = storage_path / task_id
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.assignments: Dict[str, WorkerAssignment] = {}
        self.shared_context: Dict[str, Any] = {}
        self._load_state()

    def _load_state(self) -> None:
        """Load coordinator state from disk."""
        state_file = self.storage_path / "state.json"
        if state_file.exists():
            with open(state_file) as f:
                data = json.load(f)
                self.shared_context = data.get("shared_context", {})
                for worker_id, assignment_data in data.get("assignments", {}).items():
                    self.assignments[worker_id] = WorkerAssignment(**assignment_data)

    def _save_state(self) -> None:
        """Save coordinator state to disk."""
        state_file = self.storage_path / "state.json"
        data = {
            "task_id": self.task_id,
            "shared_context": self.shared_context,
            "assignments": {
                worker_id: {
                    "worker_id": a.worker_id,
                    "task_description": a.task_description,
                    "expected_outputs": a.expected_outputs,
                    "dependencies": a.dependencies,
                    "status": a.status,
                    "result": a.result
                }
                for worker_id, a in self.assignments.items()
            }
        }
        with open(state_file, "w") as f:
            json.dump(data, f, indent=2)

    def assign_work(
        self,
        worker_id: str,
        task_description: str,
        expected_outputs: List[str],
        dependencies: Optional[List[str]] = None
    ) -> WorkerAssignment:
        """Assign work to a worker."""
        assignment = WorkerAssignment(
            worker_id=worker_id,
            task_description=task_description,
            expected_outputs=expected_outputs,
            dependencies=dependencies or []
        )
        self.assignments[worker_id] = assignment
        self._save_state()
        return assignment

    def get_ready_assignments(self) -> List[WorkerAssignment]:
        """Get assignments whose dependencies are all completed."""
        ready = []
        for assignment in self.assignments.values():
            if assignment.status != "pending":
                continue

            # Check if all dependencies are completed
            deps_met = all(
                self.assignments.get(dep_id, WorkerAssignment(
                    worker_id=dep_id,
                    task_description="",
                    expected_outputs=[],
                    dependencies=[]
                )).status == "completed"
                for dep_id in assignment.dependencies
            )

            if deps_met:
                ready.append(assignment)

        return ready

    def mark_in_progress(self, worker_id: str) -> None:
        """Mark assignment as in progress."""
        if worker_id in self.assignments:
            self.assignments[worker_id].status = "in_progress"
            self._save_state()

    def mark_completed(self, worker_id: str, result: str) -> None:
        """Mark assignment as completed with result."""
        if worker_id in self.assignments:
            self.assignments[worker_id].status = "completed"
            self.assignments[worker_id].result = result
            self._save_state()

    def mark_failed(self, worker_id: str, error: str) -> None:
        """Mark assignment as failed."""
        if worker_id in self.assignments:
            self.assignments[worker_id].status = "failed"
            self.assignments[worker_id].result = f"ERROR: {error}"
            self._save_state()

    def share_context(self, key: str, value: Any) -> None:
        """Share context data with all workers."""
        self.shared_context[key] = value
        self._save_state()

    def get_shared_context(self, key: str) -> Optional[Any]:
        """Get shared context data."""
        return self.shared_context.get(key)

    def get_progress_summary(self) -> str:
        """Get a summary of coordination progress."""
        total = len(self.assignments)
        if total == 0:
            return "No assignments yet."

        completed = sum(1 for a in self.assignments.values() if a.status == "completed")
        in_progress = sum(1 for a in self.assignments.values() if a.status == "in_progress")
        failed = sum(1 for a in self.assignments.values() if a.status == "failed")
        pending = total - completed - in_progress - failed

        lines = [
            f"Task Coordination Progress ({self.task_id})",
            f"  Total: {total}",
            f"  Completed: {completed}",
            f"  In Progress: {in_progress}",
            f"  Pending: {pending}",
            f"  Failed: {failed}"
        ]

        if self.assignments:
            lines.append("\nAssignments:")
            for worker_id, assignment in self.assignments.items():
                status_emoji = {
                    "pending": "[ ]",
                    "in_progress": "[>]",
                    "completed": "[x]",
                    "failed": "[!]"
                }.get(assignment.status, "[ ]")
                lines.append(f"  {status_emoji} {worker_id}: {assignment.task_description[:50]}...")

        return "\n".join(lines)


# =============================================================================
# Demo / Test
# =============================================================================

def main():
    """Demo the cooperation system."""
    import shutil

    print("=" * 60)
    print("AUTO-CLAUDE COOPERATION DEMO")
    print("=" * 60)
    print()

    # Clean up old demo data
    demo_path = Path.home() / ".uap" / "cooperation" / "demo-task"
    if demo_path.exists():
        shutil.rmtree(demo_path)

    # Create cooperation manager
    manager = CooperationManager()

    # Simulate Session 1
    print("[>>] Session 1: Planning and Research")
    print("-" * 40)

    manager.record_decision(
        description="Database choice",
        choice="PostgreSQL",
        rationale="Need JSONB support and strong consistency",
        alternatives=["MongoDB", "SQLite"],
        confidence="high"
    )

    manager.record_decision(
        description="API framework",
        choice="FastAPI",
        rationale="Async support, automatic OpenAPI docs",
        alternatives=["Flask", "Django REST"],
        confidence="high"
    )

    manager.record_constraint(
        description="Must support Python 3.11+",
        category="technical",
        severity="high"
    )

    manager.record_finding("Existing auth system uses JWT tokens")
    manager.record_question("Should we support OAuth2 providers?")

    manager.record_file_created("docs/api-spec.md")
    manager.record_file_created("src/models/user.py")

    # Create handoff
    handoff = manager.create_handoff(
        task_id="demo-task",
        completed_steps=["Requirements Analysis", "Database Design"],
        current_step="API Design",
        remaining_steps=["Implementation", "Testing", "Deployment"],
        environment_notes=["Docker Compose available", "CI/CD via GitHub Actions"]
    )

    print(f"  Created handoff: {handoff.id}")
    print(f"  Progress: {handoff.progress_percent}%")
    print()

    # Simulate Session 2 loading handoff
    print("[>>] Session 2: Loading Previous Context")
    print("-" * 40)

    loaded = manager.load_handoff("demo-task")
    if loaded:
        print(loaded.to_context_string())
    print()

    # Test Task Coordinator
    print("[>>] Testing Task Coordinator")
    print("-" * 40)

    coord = TaskCoordinator("demo-coordination")

    coord.assign_work(
        worker_id="researcher",
        task_description="Research authentication best practices",
        expected_outputs=["auth-report.md"]
    )

    coord.assign_work(
        worker_id="designer",
        task_description="Design database schema",
        expected_outputs=["schema.sql"],
        dependencies=["researcher"]
    )

    coord.assign_work(
        worker_id="builder",
        task_description="Implement API endpoints",
        expected_outputs=["api.py"],
        dependencies=["designer"]
    )

    # Check ready assignments
    ready = coord.get_ready_assignments()
    print(f"  Ready to execute: {[a.worker_id for a in ready]}")

    # Simulate completion
    coord.mark_in_progress("researcher")
    coord.mark_completed("researcher", "Completed auth research. JWT recommended.")

    ready = coord.get_ready_assignments()
    print(f"  After researcher done: {[a.worker_id for a in ready]}")

    print()
    print(coord.get_progress_summary())

    print()
    print("[OK] Cooperation system demo complete")


if __name__ == "__main__":
    main()
