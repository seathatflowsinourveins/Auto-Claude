"""Consolidation Node for Ralph Loop.

This module implements the memory consolidation phase that runs
sleeptime_compute.py to consolidate WORKING memory blocks into LEARNED.
"""

from __future__ import annotations

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


def consolidation_node(state: "RalphState") -> "RalphState":
    """Run memory consolidation phase.

    This node executes sleeptime_compute.py to consolidate
    WORKING memory blocks into LEARNED memory blocks.

    Args:
        state: Current Ralph Loop state.

    Returns:
        Updated state with consolidation_result.
    """
    start = time.perf_counter()
    script_dir = state.get("script_dir", Path(__file__).parent.parent)
    script = script_dir / "sleeptime_compute.py"

    if not script.exists():
        result = PhaseResult(
            phase=IterationPhase.CONSOLIDATION,
            status=IterationStatus.WARNING,
            message="Sleep-time compute not found",
            duration_ms=0,
        )
        state["consolidation_result"] = result
        return state

    try:
        proc_result = subprocess.run(
            [sys.executable, str(script), "consolidate", "--json"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(script_dir),
        )
        duration = (time.perf_counter() - start) * 1000

        if proc_result.returncode == 0:
            result = PhaseResult(
                phase=IterationPhase.CONSOLIDATION,
                status=IterationStatus.SUCCESS,
                message="Memory consolidation completed",
                duration_ms=duration,
            )
        else:
            result = PhaseResult(
                phase=IterationPhase.CONSOLIDATION,
                status=IterationStatus.WARNING,
                message="Memory consolidation skipped",
                duration_ms=duration,
            )

        state["consolidation_result"] = result
        return state

    except subprocess.TimeoutExpired:
        duration = (time.perf_counter() - start) * 1000
        result = PhaseResult(
            phase=IterationPhase.CONSOLIDATION,
            status=IterationStatus.WARNING,
            message="Consolidation timed out after 60s",
            duration_ms=duration,
        )
        state["consolidation_result"] = result
        return state

    except FileNotFoundError as e:
        duration = (time.perf_counter() - start) * 1000
        result = PhaseResult(
            phase=IterationPhase.CONSOLIDATION,
            status=IterationStatus.WARNING,
            message=f"Python executable not found: {e}",
            duration_ms=duration,
        )
        state["consolidation_result"] = result
        return state

    except OSError as e:
        duration = (time.perf_counter() - start) * 1000
        result = PhaseResult(
            phase=IterationPhase.CONSOLIDATION,
            status=IterationStatus.WARNING,
            message=f"OS error during consolidation: {e}",
            duration_ms=duration,
        )
        state["consolidation_result"] = result
        return state
