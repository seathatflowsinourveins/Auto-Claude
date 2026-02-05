"""Metrics Collection for Ralph Loop.

This module provides functions for calculating and aggregating
metrics from Ralph Loop iterations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..state.ralph_state import (
    IterationReport,
    IterationStatus,
    PhaseResult,
    RalphState,
)


def calculate_phase_metrics(phases: List[PhaseResult]) -> Dict[str, Any]:
    """Calculate metrics from a list of phase results.

    Args:
        phases: List of phase results from an iteration.

    Returns:
        Dictionary containing calculated metrics.
    """
    if not phases:
        return {
            "total_phases": 0,
            "success_count": 0,
            "warning_count": 0,
            "failed_count": 0,
            "success_rate": 0.0,
            "total_duration_ms": 0.0,
            "avg_duration_ms": 0.0,
        }

    success_count = sum(1 for p in phases if p.status == IterationStatus.SUCCESS)
    warning_count = sum(1 for p in phases if p.status == IterationStatus.WARNING)
    failed_count = sum(1 for p in phases if p.status == IterationStatus.FAILED)
    total_duration = sum(p.duration_ms for p in phases)

    return {
        "total_phases": len(phases),
        "success_count": success_count,
        "warning_count": warning_count,
        "failed_count": failed_count,
        "success_rate": success_count / len(phases) if phases else 0.0,
        "total_duration_ms": total_duration,
        "avg_duration_ms": total_duration / len(phases) if phases else 0.0,
    }


def aggregate_iteration_metrics(
    reports: List[IterationReport],
) -> Dict[str, Any]:
    """Aggregate metrics across multiple iteration reports.

    Args:
        reports: List of iteration reports.

    Returns:
        Dictionary containing aggregated metrics.
    """
    if not reports:
        return {
            "total_iterations": 0,
            "success_iterations": 0,
            "warning_iterations": 0,
            "failed_iterations": 0,
            "success_rate": 0.0,
            "avg_duration_ms": 0.0,
            "total_duration_ms": 0.0,
            "phase_success_rates": {},
        }

    success_count = sum(
        1 for r in reports if r.overall_status == IterationStatus.SUCCESS
    )
    warning_count = sum(
        1 for r in reports if r.overall_status == IterationStatus.WARNING
    )
    failed_count = sum(
        1 for r in reports if r.overall_status == IterationStatus.FAILED
    )
    total_duration = sum(r.total_duration_ms for r in reports)

    # Calculate per-phase success rates
    phase_stats: Dict[str, Dict[str, int]] = {}
    for report in reports:
        for phase in report.phases:
            phase_name = phase.phase.value
            if phase_name not in phase_stats:
                phase_stats[phase_name] = {"success": 0, "total": 0}
            phase_stats[phase_name]["total"] += 1
            if phase.status == IterationStatus.SUCCESS:
                phase_stats[phase_name]["success"] += 1

    phase_success_rates = {
        name: stats["success"] / stats["total"] if stats["total"] > 0 else 0.0
        for name, stats in phase_stats.items()
    }

    return {
        "total_iterations": len(reports),
        "success_iterations": success_count,
        "warning_iterations": warning_count,
        "failed_iterations": failed_count,
        "success_rate": success_count / len(reports) if reports else 0.0,
        "avg_duration_ms": total_duration / len(reports) if reports else 0.0,
        "total_duration_ms": total_duration,
        "phase_success_rates": phase_success_rates,
    }


def get_trend_analysis(
    reports: List[IterationReport],
    window_size: int = 10,
) -> Dict[str, Any]:
    """Analyze trends in iteration performance.

    Args:
        reports: List of iteration reports, ordered chronologically.
        window_size: Number of recent reports to analyze.

    Returns:
        Dictionary containing trend analysis.
    """
    if len(reports) < 2:
        return {
            "trend": "insufficient_data",
            "recent_success_rate": 0.0,
            "overall_success_rate": 0.0,
            "improvement": 0.0,
        }

    # Overall metrics
    overall_success = sum(
        1 for r in reports if r.overall_status == IterationStatus.SUCCESS
    )
    overall_rate = overall_success / len(reports)

    # Recent window metrics
    recent = reports[-window_size:] if len(reports) >= window_size else reports
    recent_success = sum(
        1 for r in recent if r.overall_status == IterationStatus.SUCCESS
    )
    recent_rate = recent_success / len(recent)

    # Calculate improvement
    improvement = recent_rate - overall_rate

    # Determine trend
    if improvement > 0.1:
        trend = "improving"
    elif improvement < -0.1:
        trend = "declining"
    else:
        trend = "stable"

    return {
        "trend": trend,
        "recent_success_rate": recent_rate,
        "overall_success_rate": overall_rate,
        "improvement": improvement,
        "window_size": len(recent),
        "total_iterations": len(reports),
    }


def calculate_health_score(state: RalphState) -> float:
    """Calculate overall health score from state.

    Args:
        state: Current Ralph Loop state.

    Returns:
        Health score from 0.0 to 1.0.
    """
    phases = state.get("phases", [])
    if not phases:
        return 0.0

    # Weight each status
    weights = {
        IterationStatus.SUCCESS: 1.0,
        IterationStatus.WARNING: 0.5,
        IterationStatus.FAILED: 0.0,
        IterationStatus.PENDING: 0.5,
        IterationStatus.RUNNING: 0.5,
    }

    total_weight = sum(weights.get(p.status, 0.5) for p in phases)
    return total_weight / len(phases)
