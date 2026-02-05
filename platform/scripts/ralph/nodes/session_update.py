"""Session Update Node for Ralph Loop.

This module implements the session update phase that runs
session_continuity.py to update session state.
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


def session_update_node(state: "RalphState") -> "RalphState":
    """Run session update phase.

    This node executes session_continuity.py to update the
    current session state.

    Args:
        state: Current Ralph Loop state.

    Returns:
        Updated state with session_result.
    """
    start = time.perf_counter()
    script_dir = state.get("script_dir", Path(__file__).parent.parent)
    script = script_dir / "session_continuity.py"

    if not script.exists():
        result = PhaseResult(
            phase=IterationPhase.SESSION_UPDATE,
            status=IterationStatus.WARNING,
            message="Session continuity not found",
            duration_ms=0,
        )
        state["session_result"] = result
        return state

    try:
        proc_result = subprocess.run(
            [sys.executable, str(script), "status", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(script_dir),
        )
        duration = (time.perf_counter() - start) * 1000

        if proc_result.returncode == 0:
            try:
                # Find JSON in output
                lines = proc_result.stdout.strip().split("\n")
                for i, line in enumerate(lines):
                    if line.strip().startswith("{"):
                        data = json.loads("\n".join(lines[i:]))
                        session_id = data.get("session_id", "unknown")
                        result = PhaseResult(
                            phase=IterationPhase.SESSION_UPDATE,
                            status=IterationStatus.SUCCESS,
                            message=f"Session {session_id[:8]}... active",
                            duration_ms=duration,
                            details=data,
                        )
                        state["session_result"] = result
                        return state

            except (json.JSONDecodeError, IndexError):
                pass  # Fall through to default success

        # Default success if exit code is 0
        if proc_result.returncode == 0:
            result = PhaseResult(
                phase=IterationPhase.SESSION_UPDATE,
                status=IterationStatus.SUCCESS,
                message="Session updated",
                duration_ms=duration,
            )
        else:
            result = PhaseResult(
                phase=IterationPhase.SESSION_UPDATE,
                status=IterationStatus.WARNING,
                message=f"Session update returned code {proc_result.returncode}",
                duration_ms=duration,
            )

        state["session_result"] = result
        return state

    except subprocess.TimeoutExpired:
        duration = (time.perf_counter() - start) * 1000
        result = PhaseResult(
            phase=IterationPhase.SESSION_UPDATE,
            status=IterationStatus.WARNING,
            message="Session update timed out after 30s",
            duration_ms=duration,
        )
        state["session_result"] = result
        return state

    except FileNotFoundError as e:
        duration = (time.perf_counter() - start) * 1000
        result = PhaseResult(
            phase=IterationPhase.SESSION_UPDATE,
            status=IterationStatus.WARNING,
            message=f"Python executable not found: {e}",
            duration_ms=duration,
        )
        state["session_result"] = result
        return state

    except OSError as e:
        duration = (time.perf_counter() - start) * 1000
        result = PhaseResult(
            phase=IterationPhase.SESSION_UPDATE,
            status=IterationStatus.WARNING,
            message=f"OS error during session update: {e}",
            duration_ms=duration,
        )
        state["session_result"] = result
        return state
