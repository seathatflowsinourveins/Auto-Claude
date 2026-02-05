"""Health Check Node for Ralph Loop.

This module implements the health check phase that runs
ecosystem_orchestrator.py to check system health.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from ..state.ralph_state import RalphState

from ..state.ralph_state import IterationPhase, IterationStatus, PhaseResult

logger = structlog.get_logger(__name__)


def health_check_node(state: "RalphState") -> "RalphState":
    """Run ecosystem health check phase.

    This node executes ecosystem_orchestrator.py to check the health
    of all platform components.

    Args:
        state: Current Ralph Loop state.

    Returns:
        Updated state with health_result.
    """
    start = time.perf_counter()
    script_dir = state.get("script_dir", Path(__file__).parent.parent)
    script = script_dir / "ecosystem_orchestrator.py"

    if not script.exists():
        result = PhaseResult(
            phase=IterationPhase.HEALTH_CHECK,
            status=IterationStatus.WARNING,
            message="Ecosystem orchestrator not found",
            duration_ms=0,
        )
        state["health_result"] = result
        return state

    try:
        proc_result = subprocess.run(
            [sys.executable, str(script), "--quick", "--json"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(script_dir),
        )
        duration = (time.perf_counter() - start) * 1000

        # Try to find JSON in output (may have non-JSON prefix from logging)
        stdout = proc_result.stdout.strip()
        json_data = None

        # First try direct parse
        try:
            json_data = json.loads(stdout)
        except json.JSONDecodeError:
            # Search for JSON object in output
            lines = stdout.split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("{"):
                    try:
                        json_str = "\n".join(lines[i:])
                        json_data = json.loads(json_str)
                        break
                    except json.JSONDecodeError:
                        continue

        if json_data:
            summary = json_data.get("summary", {})
            overall = json_data.get("overall_status", "unknown")

            # Map overall_status to IterationStatus
            if overall == "healthy":
                status = IterationStatus.SUCCESS
            elif overall in ("degraded", "unknown"):
                status = IterationStatus.WARNING
            else:  # unavailable
                status = IterationStatus.FAILED

            result = PhaseResult(
                phase=IterationPhase.HEALTH_CHECK,
                status=status,
                message=f"OK={summary.get('healthy', 0)}, WARN={summary.get('degraded', 0)}, ERR={summary.get('unavailable', 0)}",
                duration_ms=duration,
                details={"overall": overall, "summary": summary},
            )
            state["health_result"] = result
            return state

        # Fallback: check exit code if no valid JSON found
        if proc_result.returncode == 0:
            result = PhaseResult(
                phase=IterationPhase.HEALTH_CHECK,
                status=IterationStatus.SUCCESS,
                message="Health check passed (exit code 0)",
                duration_ms=duration,
            )
        elif proc_result.returncode == 1:
            result = PhaseResult(
                phase=IterationPhase.HEALTH_CHECK,
                status=IterationStatus.WARNING,
                message="Health check warnings (exit code 1)",
                duration_ms=duration,
            )
        else:
            result = PhaseResult(
                phase=IterationPhase.HEALTH_CHECK,
                status=IterationStatus.FAILED,
                message=f"Health check failed (exit code {proc_result.returncode})",
                duration_ms=duration,
            )

        state["health_result"] = result
        return state

    except subprocess.TimeoutExpired:
        duration = (time.perf_counter() - start) * 1000
        result = PhaseResult(
            phase=IterationPhase.HEALTH_CHECK,
            status=IterationStatus.FAILED,
            message="Health check timed out after 60s",
            duration_ms=duration,
        )
        state["health_result"] = result
        return state

    except FileNotFoundError as e:
        duration = (time.perf_counter() - start) * 1000
        result = PhaseResult(
            phase=IterationPhase.HEALTH_CHECK,
            status=IterationStatus.FAILED,
            message=f"Python executable not found: {e}",
            duration_ms=duration,
        )
        state["health_result"] = result
        return state

    except OSError as e:
        duration = (time.perf_counter() - start) * 1000
        result = PhaseResult(
            phase=IterationPhase.HEALTH_CHECK,
            status=IterationStatus.FAILED,
            message=f"OS error during health check: {e}",
            duration_ms=duration,
        )
        state["health_result"] = result
        return state
