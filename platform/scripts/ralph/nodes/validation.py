"""Validation Node for Ralph Loop.

This module implements the validation phase that runs
auto_validate.py to validate platform components.
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


def validation_node(state: "RalphState") -> "RalphState":
    """Run validation pipeline phase.

    This node executes auto_validate.py to run validation checks
    on platform components.

    Args:
        state: Current Ralph Loop state.

    Returns:
        Updated state with validation_result.
    """
    start = time.perf_counter()
    script_dir = state.get("script_dir", Path(__file__).parent.parent)
    script = script_dir / "auto_validate.py"

    if not script.exists():
        result = PhaseResult(
            phase=IterationPhase.VALIDATION,
            status=IterationStatus.WARNING,
            message="Auto-validator not found",
            duration_ms=0,
        )
        state["validation_result"] = result
        return state

    try:
        proc_result = subprocess.run(
            [sys.executable, str(script), "--quick", "--json"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(script_dir),
        )
        duration = (time.perf_counter() - start) * 1000

        # Parse JSON output (appears after the status lines)
        lines = proc_result.stdout.strip().split("\n")
        json_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("{"):
                json_start = i
                break

        if json_start is not None:
            try:
                json_str = "\n".join(lines[json_start:])
                data = json.loads(json_str)
                summary = data.get("summary", {})
                overall = data.get("overall_status", "unknown")

                if overall == "pass":
                    status = IterationStatus.SUCCESS
                elif overall == "warn":
                    status = IterationStatus.WARNING
                else:
                    status = IterationStatus.FAILED

                result = PhaseResult(
                    phase=IterationPhase.VALIDATION,
                    status=status,
                    message=f"PASS={summary.get('pass', 0)}, WARN={summary.get('warn', 0)}, FAIL={summary.get('fail', 0)}",
                    duration_ms=duration,
                    details={"overall": overall, "summary": summary},
                )
                state["validation_result"] = result
                return state

            except json.JSONDecodeError:
                pass  # Fall through to exit code handling

        # Fallback: check exit code
        if proc_result.returncode == 0:
            status = IterationStatus.SUCCESS
        elif proc_result.returncode == 1:
            status = IterationStatus.WARNING
        else:
            status = IterationStatus.FAILED

        result = PhaseResult(
            phase=IterationPhase.VALIDATION,
            status=status,
            message=f"Exit code: {proc_result.returncode}",
            duration_ms=duration,
        )
        state["validation_result"] = result
        return state

    except subprocess.TimeoutExpired:
        duration = (time.perf_counter() - start) * 1000
        result = PhaseResult(
            phase=IterationPhase.VALIDATION,
            status=IterationStatus.FAILED,
            message="Validation timed out after 120s",
            duration_ms=duration,
        )
        state["validation_result"] = result
        return state

    except FileNotFoundError as e:
        duration = (time.perf_counter() - start) * 1000
        result = PhaseResult(
            phase=IterationPhase.VALIDATION,
            status=IterationStatus.FAILED,
            message=f"Python executable not found: {e}",
            duration_ms=duration,
        )
        state["validation_result"] = result
        return state

    except OSError as e:
        duration = (time.perf_counter() - start) * 1000
        result = PhaseResult(
            phase=IterationPhase.VALIDATION,
            status=IterationStatus.FAILED,
            message=f"OS error during validation: {e}",
            duration_ms=duration,
        )
        state["validation_result"] = result
        return state
