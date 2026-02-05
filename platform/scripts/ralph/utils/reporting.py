"""Report Generation for Ralph Loop.

This module provides functions for generating and formatting
reports from Ralph Loop iterations.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..state.ralph_state import (
    IterationReport,
    IterationStatus,
    PhaseResult,
    RalphState,
)


def save_report(report: IterationReport, reports_dir: Path) -> Path:
    """Save an iteration report to disk.

    Args:
        report: The iteration report to save.
        reports_dir: Directory to save reports in.

    Returns:
        Path to the saved report file.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / f"iteration_{report.iteration_number:04d}.json"
    report_file.write_text(
        json.dumps(report.to_dict(), indent=2),
        encoding="utf-8"
    )
    return report_file


def print_iteration_header(iteration_number: int) -> None:
    """Print the header for an iteration.

    Args:
        iteration_number: The current iteration number.
    """
    print("=" * 60)
    print(f"RALPH LOOP ITERATION #{iteration_number}")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)


def print_phase_result(phase_num: int, total_phases: int, result: PhaseResult) -> None:
    """Print the result of a single phase.

    Args:
        phase_num: Current phase number (1-indexed).
        total_phases: Total number of phases.
        result: The phase result to print.
    """
    status_str = result.status.value.upper()
    print(f"[{phase_num}/{total_phases}] {result.phase.value}: [{status_str}] {result.message}")


def print_iteration_result(
    overall_status: IterationStatus,
    summary: Dict[str, Any],
    total_duration_ms: float,
    recommendations: List[str],
) -> None:
    """Print the result of an iteration.

    Args:
        overall_status: The overall iteration status.
        summary: Summary statistics.
        total_duration_ms: Total duration in milliseconds.
        recommendations: List of recommendations.
    """
    print("-" * 60)
    print(f"RESULT: [{overall_status.value.upper()}]")
    print(
        f"Summary: SUCCESS={summary.get('success', 0)}, "
        f"WARN={summary.get('warnings', 0)}, "
        f"FAIL={summary.get('failed', 0)}"
    )
    print(f"Duration: {total_duration_ms:.0f}ms")

    if recommendations:
        print("\nRecommendations:")
        for rec in recommendations[:3]:
            print(f"  - {rec}")

    print("=" * 60)


def print_status(
    iteration_count: int,
    reports_generated: int,
    reports_dir: str,
    latest_report: Optional[Dict[str, Any]] = None,
) -> None:
    """Print the current Ralph Loop status.

    Args:
        iteration_count: Total iterations run.
        reports_generated: Number of reports generated.
        reports_dir: Path to reports directory.
        latest_report: Latest report data if available.
    """
    print("=" * 50)
    print("RALPH LOOP STATUS")
    print("=" * 50)
    print(f"Iteration Count:    {iteration_count}")
    print(f"Reports Generated:  {reports_generated}")
    print(f"Reports Directory:  {reports_dir}")

    if latest_report:
        print("-" * 50)
        print(f"Latest Iteration:   #{latest_report['iteration_number']}")
        print(f"Status:             {latest_report['overall_status']}")
        print(f"Duration:           {latest_report['total_duration_ms']:.0f}ms")

    print("=" * 50)


def format_report(report: Dict[str, Any]) -> str:
    """Format a report for display.

    Args:
        report: Report data as dictionary.

    Returns:
        Formatted string representation.
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"RALPH LOOP ITERATION #{report['iteration_number']} REPORT")
    lines.append("=" * 60)
    lines.append(f"Started:   {report['started_at'][:19]}")
    lines.append(f"Completed: {report['completed_at'][:19]}")
    lines.append(f"Duration:  {report['total_duration_ms']:.0f}ms")
    lines.append(f"Status:    {report['overall_status']}")
    lines.append("-" * 60)
    lines.append("PHASES:")

    for phase in report.get("phases", []):
        status = phase.get("status", "unknown").upper()
        phase_name = phase.get("phase", "unknown")
        message = phase.get("message", "")
        lines.append(f"  [{status:7}] {phase_name}: {message}")

    if report.get("recommendations"):
        lines.append("-" * 60)
        lines.append("RECOMMENDATIONS:")
        for rec in report["recommendations"]:
            lines.append(f"  - {rec}")

    lines.append("=" * 60)
    return "\n".join(lines)


def generate_summary_report(
    reports: List[IterationReport],
    include_details: bool = False,
) -> str:
    """Generate a summary report across multiple iterations.

    Args:
        reports: List of iteration reports.
        include_details: Whether to include detailed per-iteration info.

    Returns:
        Formatted summary report string.
    """
    if not reports:
        return "No reports available."

    lines = []
    lines.append("=" * 60)
    lines.append("RALPH LOOP SUMMARY REPORT")
    lines.append("=" * 60)

    # Overall stats
    total = len(reports)
    success = sum(1 for r in reports if r.overall_status == IterationStatus.SUCCESS)
    warnings = sum(1 for r in reports if r.overall_status == IterationStatus.WARNING)
    failed = sum(1 for r in reports if r.overall_status == IterationStatus.FAILED)
    avg_duration = sum(r.total_duration_ms for r in reports) / total

    lines.append(f"Total Iterations: {total}")
    lines.append(f"Success Rate:     {success/total*100:.1f}%")
    lines.append(f"  SUCCESS: {success}")
    lines.append(f"  WARNING: {warnings}")
    lines.append(f"  FAILED:  {failed}")
    lines.append(f"Avg Duration:     {avg_duration:.0f}ms")

    if include_details:
        lines.append("-" * 60)
        lines.append("ITERATION DETAILS:")
        for report in reports[-10:]:  # Last 10
            status = report.overall_status.value.upper()
            lines.append(
                f"  #{report.iteration_number:04d}: [{status:7}] "
                f"{report.total_duration_ms:.0f}ms"
            )

    lines.append("=" * 60)
    return "\n".join(lines)
