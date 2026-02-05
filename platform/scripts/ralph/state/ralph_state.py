"""Ralph Loop State Definitions.

This module defines all state types used in the Ralph Loop,
following LangGraph-style TypedDict patterns for state management.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict


class IterationPhase(str, Enum):
    """Phases of a Ralph Loop iteration."""

    INIT = "init"
    HEALTH_CHECK = "health_check"
    VALIDATION = "validation"
    CONSOLIDATION = "consolidation"
    SESSION_UPDATE = "session_update"
    REPORTING = "reporting"
    COMPLETE = "complete"


class IterationStatus(str, Enum):
    """Status of an iteration."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    WARNING = "warning"
    FAILED = "failed"


@dataclass
class PhaseResult:
    """Result of a single phase."""

    phase: IterationPhase
    status: IterationStatus
    message: str
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "phase": self.phase.value,
            "status": self.status.value,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "details": self.details,
        }


@dataclass
class IterationReport:
    """Complete report for a Ralph Loop iteration."""

    iteration_number: int
    started_at: str
    completed_at: str
    total_duration_ms: float
    overall_status: IterationStatus
    phases: List[PhaseResult]
    summary: Dict[str, Any]
    recommendations: List[str]

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "iteration_number": self.iteration_number,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_duration_ms": self.total_duration_ms,
            "overall_status": self.overall_status.value,
            "phases": [p.to_dict() for p in self.phases],
            "summary": self.summary,
            "recommendations": self.recommendations,
        }


class RalphState(TypedDict, total=False):
    """LangGraph-style state for Ralph Loop iteration.

    This TypedDict defines the state that flows through the graph nodes.
    Each node can read and update specific fields.
    """

    # Iteration metadata
    iteration_number: int
    started_at: str
    completed_at: str

    # Phase results
    health_result: Optional[PhaseResult]
    validation_result: Optional[PhaseResult]
    consolidation_result: Optional[PhaseResult]
    session_result: Optional[PhaseResult]

    # Aggregated data
    phases: List[PhaseResult]
    overall_status: IterationStatus
    summary: Dict[str, Any]
    recommendations: List[str]

    # Configuration
    script_dir: Path
    data_dir: Path
    memory_dir: Path
    reports_dir: Path


def create_initial_state(
    iteration_number: int,
    script_dir: Path,
    data_dir: Path,
    memory_dir: Path,
    reports_dir: Path,
) -> RalphState:
    """Create initial state for a new iteration.

    Args:
        iteration_number: The current iteration number.
        script_dir: Path to the scripts directory.
        data_dir: Path to the data directory.
        memory_dir: Path to the memory directory.
        reports_dir: Path to the reports directory.

    Returns:
        A new RalphState with initialized values.
    """
    return RalphState(
        iteration_number=iteration_number,
        started_at=datetime.now(timezone.utc).isoformat(),
        completed_at="",
        health_result=None,
        validation_result=None,
        consolidation_result=None,
        session_result=None,
        phases=[],
        overall_status=IterationStatus.PENDING,
        summary={},
        recommendations=[],
        script_dir=script_dir,
        data_dir=data_dir,
        memory_dir=memory_dir,
        reports_dir=reports_dir,
    )


def finalize_state(state: RalphState) -> RalphState:
    """Finalize state after all phases complete.

    Collects phase results, calculates overall status, and generates
    recommendations.

    Args:
        state: The current state after all phases.

    Returns:
        Updated state with finalized values.
    """
    # Collect all phase results
    phases = []
    for result_key in ["health_result", "validation_result", "consolidation_result", "session_result"]:
        result = state.get(result_key)
        if result is not None:
            phases.append(result)

    # Calculate overall status
    failed = sum(1 for p in phases if p.status == IterationStatus.FAILED)
    warnings = sum(1 for p in phases if p.status == IterationStatus.WARNING)
    success = sum(1 for p in phases if p.status == IterationStatus.SUCCESS)

    if failed > 0:
        overall_status = IterationStatus.FAILED
    elif warnings > 1:
        overall_status = IterationStatus.WARNING
    else:
        overall_status = IterationStatus.SUCCESS

    # Generate recommendations
    recommendations = []
    for p in phases:
        if p.status == IterationStatus.FAILED:
            recommendations.append(f"Fix {p.phase.value}: {p.message}")
        elif p.status == IterationStatus.WARNING:
            recommendations.append(f"Investigate {p.phase.value}: {p.message}")

    # Create summary
    summary = {
        "success": success,
        "warnings": warnings,
        "failed": failed,
        "phases_run": len(phases),
    }

    # Update state
    state["phases"] = phases
    state["overall_status"] = overall_status
    state["summary"] = summary
    state["recommendations"] = recommendations
    state["completed_at"] = datetime.now(timezone.utc).isoformat()

    return state


def state_to_report(state: RalphState, total_duration_ms: float) -> IterationReport:
    """Convert finalized state to an IterationReport.

    Args:
        state: The finalized state.
        total_duration_ms: Total duration of the iteration.

    Returns:
        An IterationReport object.
    """
    return IterationReport(
        iteration_number=state["iteration_number"],
        started_at=state["started_at"],
        completed_at=state["completed_at"],
        total_duration_ms=total_duration_ms,
        overall_status=state["overall_status"],
        phases=state["phases"],
        summary=state["summary"],
        recommendations=state["recommendations"],
    )


def create_memory_block(report: IterationReport, memory_dir: Path) -> str:
    """Create and save a memory block from an iteration report.

    Args:
        report: The iteration report.
        memory_dir: Directory to save the memory block.

    Returns:
        The block ID.
    """
    now = datetime.now(timezone.utc).isoformat()
    content = (
        f"Iteration #{report.iteration_number} [{report.overall_status.value}]: "
        f"{report.summary.get('success', 0)} success, "
        f"{report.summary.get('warnings', 0)} warnings, "
        f"{report.summary.get('failed', 0)} failed. "
    )
    if report.recommendations:
        content += "Recommendations: " + "; ".join(report.recommendations[:3])

    block_id = hashlib.sha256(
        f"iter:{report.iteration_number}:{now}".encode()
    ).hexdigest()[:16]

    block = {
        "id": block_id,
        "type": "working",
        "content": content,
        "created_at": now,
        "updated_at": now,
        "metadata": {
            "topic": "ralph_loop",
            "iteration": report.iteration_number,
            "status": report.overall_status.value,
            "summary": report.summary,
        },
        "embedding_hash": hashlib.md5(content.encode()).hexdigest(),
    }

    block_file = memory_dir / f"{block_id}.json"
    block_file.write_text(json.dumps(block, indent=2), encoding="utf-8")

    return block_id
