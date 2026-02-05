#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.26.0",
#     "pydantic>=2.0.0",
#     "structlog>=24.1.0",
# ]
# [tool.uv]
# exclude-newer = "2026-02-01"
# ///
"""
Ralph Loop Runner - V10 Ultimate Autonomous Platform

Unified orchestration for autonomous improvement iterations.
Coordinates all platform modules for seamless operation.

This file serves as the entry point that runs the modular Ralph Loop.
The modular implementation is in the ralph/ package with LangGraph-style
graph construction.

Module Structure:
    ralph/
    ├── __init__.py     # Package exports
    ├── graph.py        # LangGraph-style graph construction
    ├── nodes/          # Individual phase nodes
    │   ├── health_check.py
    │   ├── validation.py
    │   ├── consolidation.py
    │   └── session_update.py
    ├── state/          # State definitions
    │   └── ralph_state.py
    └── utils/          # Utility functions
        ├── metrics.py
        └── reporting.py

Modules Orchestrated:
1. ecosystem_orchestrator.py - Health monitoring
2. auto_validate.py - Validation pipeline
3. sleeptime_compute.py - Memory consolidation
4. session_continuity.py - Session management

Usage:
    uv run ralph_loop.py status         # Show complete status
    uv run ralph_loop.py iterate        # Run one iteration
    uv run ralph_loop.py daemon         # Run continuous loop
    uv run ralph_loop.py init           # Initialize all systems
    uv run ralph_loop.py report         # Generate iteration report

Platform: Windows 11 + Python 3.11+
Architecture: V10 Optimized (Verified, Minimal, Seamless)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

# Try to import from modular package, fall back to inline implementation
try:
    from ralph import (
        GraphConfig,
        IterationReport,
        IterationStatus,
        create_default_config,
        create_ralph_graph,
        run_iteration as graph_run_iteration,
        format_report,
        print_status,
    )
    USE_MODULAR = True
except ImportError:
    USE_MODULAR = False
    # Fall back to inline implementation if ralph package not available
    from dataclasses import dataclass, field
    from enum import Enum
    import hashlib

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

# Configuration
SCRIPT_DIR = Path(__file__).parent
V10_DIR = SCRIPT_DIR.parent
UNLEASH_DIR = V10_DIR.parent
DATA_DIR = V10_DIR / "data"
MEMORY_DIR = DATA_DIR / "memory"
REPORTS_DIR = DATA_DIR / "reports"

# Ensure directories exist
MEMORY_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Windows compatibility
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

logger = structlog.get_logger(__name__)


class ModuleRunner:
    """Runs V10 platform modules (fallback implementation)."""

    def __init__(self) -> None:
        self.script_dir = SCRIPT_DIR

    async def run_ecosystem_check(self) -> "PhaseResult":
        """Run ecosystem health check."""
        start = time.perf_counter()
        script = self.script_dir / "ecosystem_orchestrator.py"

        if not script.exists():
            return PhaseResult(
                phase=IterationPhase.HEALTH_CHECK,
                status=IterationStatus.WARNING,
                message="Ecosystem orchestrator not found",
                duration_ms=0,
            )

        try:
            result = subprocess.run(
                [sys.executable, str(script), "--quick", "--json"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.script_dir),
            )
            duration = (time.perf_counter() - start) * 1000

            stdout = result.stdout.strip()
            json_data = None

            try:
                json_data = json.loads(stdout)
            except json.JSONDecodeError:
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
                if overall == "healthy":
                    status = IterationStatus.SUCCESS
                elif overall in ("degraded", "unknown"):
                    status = IterationStatus.WARNING
                else:
                    status = IterationStatus.FAILED

                return PhaseResult(
                    phase=IterationPhase.HEALTH_CHECK,
                    status=status,
                    message=f"OK={summary.get('healthy', 0)}, WARN={summary.get('degraded', 0)}, ERR={summary.get('unavailable', 0)}",
                    duration_ms=duration,
                    details={"overall": overall, "summary": summary},
                )

            if result.returncode == 0:
                return PhaseResult(
                    phase=IterationPhase.HEALTH_CHECK,
                    status=IterationStatus.SUCCESS,
                    message="Health check passed (exit code 0)",
                    duration_ms=duration,
                )
            elif result.returncode == 1:
                return PhaseResult(
                    phase=IterationPhase.HEALTH_CHECK,
                    status=IterationStatus.WARNING,
                    message="Health check warnings (exit code 1)",
                    duration_ms=duration,
                )
            else:
                return PhaseResult(
                    phase=IterationPhase.HEALTH_CHECK,
                    status=IterationStatus.FAILED,
                    message=f"Health check failed (exit code {result.returncode})",
                    duration_ms=duration,
                )

        except subprocess.TimeoutExpired:
            duration = (time.perf_counter() - start) * 1000
            return PhaseResult(
                phase=IterationPhase.HEALTH_CHECK,
                status=IterationStatus.FAILED,
                message="Health check timed out after 60s",
                duration_ms=duration,
            )
        except OSError as e:
            duration = (time.perf_counter() - start) * 1000
            return PhaseResult(
                phase=IterationPhase.HEALTH_CHECK,
                status=IterationStatus.FAILED,
                message=str(e),
                duration_ms=duration,
            )

    async def run_validation(self) -> "PhaseResult":
        """Run auto-validation pipeline."""
        start = time.perf_counter()
        script = self.script_dir / "auto_validate.py"

        if not script.exists():
            return PhaseResult(
                phase=IterationPhase.VALIDATION,
                status=IterationStatus.WARNING,
                message="Auto-validator not found",
                duration_ms=0,
            )

        try:
            result = subprocess.run(
                [sys.executable, str(script), "--quick", "--json"],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self.script_dir),
            )
            duration = (time.perf_counter() - start) * 1000

            lines = result.stdout.strip().split("\n")
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

                    return PhaseResult(
                        phase=IterationPhase.VALIDATION,
                        status=status,
                        message=f"PASS={summary.get('pass', 0)}, WARN={summary.get('warn', 0)}, FAIL={summary.get('fail', 0)}",
                        duration_ms=duration,
                        details={"overall": overall, "summary": summary},
                    )
                except json.JSONDecodeError:
                    pass

            if result.returncode == 0:
                status = IterationStatus.SUCCESS
            elif result.returncode == 1:
                status = IterationStatus.WARNING
            else:
                status = IterationStatus.FAILED

            return PhaseResult(
                phase=IterationPhase.VALIDATION,
                status=status,
                message=f"Exit code: {result.returncode}",
                duration_ms=duration,
            )

        except subprocess.TimeoutExpired:
            duration = (time.perf_counter() - start) * 1000
            return PhaseResult(
                phase=IterationPhase.VALIDATION,
                status=IterationStatus.FAILED,
                message="Validation timed out after 120s",
                duration_ms=duration,
            )
        except OSError as e:
            duration = (time.perf_counter() - start) * 1000
            return PhaseResult(
                phase=IterationPhase.VALIDATION,
                status=IterationStatus.FAILED,
                message=str(e),
                duration_ms=duration,
            )

    async def run_consolidation(self) -> "PhaseResult":
        """Run sleep-time memory consolidation."""
        start = time.perf_counter()
        script = self.script_dir / "sleeptime_compute.py"

        if not script.exists():
            return PhaseResult(
                phase=IterationPhase.CONSOLIDATION,
                status=IterationStatus.WARNING,
                message="Sleep-time compute not found",
                duration_ms=0,
            )

        try:
            result = subprocess.run(
                [sys.executable, str(script), "consolidate", "--json"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.script_dir),
            )
            duration = (time.perf_counter() - start) * 1000

            return PhaseResult(
                phase=IterationPhase.CONSOLIDATION,
                status=IterationStatus.SUCCESS if result.returncode == 0 else IterationStatus.WARNING,
                message=f"Memory consolidation {'completed' if result.returncode == 0 else 'skipped'}",
                duration_ms=duration,
            )

        except subprocess.TimeoutExpired:
            duration = (time.perf_counter() - start) * 1000
            return PhaseResult(
                phase=IterationPhase.CONSOLIDATION,
                status=IterationStatus.WARNING,
                message="Consolidation timed out after 60s",
                duration_ms=duration,
            )
        except OSError as e:
            duration = (time.perf_counter() - start) * 1000
            return PhaseResult(
                phase=IterationPhase.CONSOLIDATION,
                status=IterationStatus.WARNING,
                message=str(e),
                duration_ms=duration,
            )

    async def run_session_update(self) -> "PhaseResult":
        """Update session state."""
        start = time.perf_counter()
        script = self.script_dir / "session_continuity.py"

        if not script.exists():
            return PhaseResult(
                phase=IterationPhase.SESSION_UPDATE,
                status=IterationStatus.WARNING,
                message="Session continuity not found",
                duration_ms=0,
            )

        try:
            result = subprocess.run(
                [sys.executable, str(script), "status", "--json"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.script_dir),
            )
            duration = (time.perf_counter() - start) * 1000

            if result.returncode == 0:
                try:
                    lines = result.stdout.strip().split("\n")
                    for i, line in enumerate(lines):
                        if line.strip().startswith("{"):
                            data = json.loads("\n".join(lines[i:]))
                            return PhaseResult(
                                phase=IterationPhase.SESSION_UPDATE,
                                status=IterationStatus.SUCCESS,
                                message=f"Session {data.get('session_id', 'unknown')[:8]}... active",
                                duration_ms=duration,
                                details=data,
                            )
                except (json.JSONDecodeError, IndexError):
                    pass

            return PhaseResult(
                phase=IterationPhase.SESSION_UPDATE,
                status=IterationStatus.SUCCESS,
                message="Session updated",
                duration_ms=duration,
            )

        except subprocess.TimeoutExpired:
            duration = (time.perf_counter() - start) * 1000
            return PhaseResult(
                phase=IterationPhase.SESSION_UPDATE,
                status=IterationStatus.WARNING,
                message="Session update timed out after 30s",
                duration_ms=duration,
            )
        except OSError as e:
            duration = (time.perf_counter() - start) * 1000
            return PhaseResult(
                phase=IterationPhase.SESSION_UPDATE,
                status=IterationStatus.WARNING,
                message=str(e),
                duration_ms=duration,
            )


class RalphLoop:
    """Main Ralph Loop orchestrator.

    Uses the modular ralph/ package if available, otherwise falls back
    to the inline implementation.
    """

    def __init__(self) -> None:
        self.iteration_count = self._load_iteration_count()

        if USE_MODULAR:
            self.graph = create_ralph_graph()
            self.config = GraphConfig(
                script_dir=SCRIPT_DIR,
                data_dir=DATA_DIR,
                memory_dir=MEMORY_DIR,
                reports_dir=REPORTS_DIR,
                verbose=True,
            )
        else:
            self.runner = ModuleRunner()

    def _load_iteration_count(self) -> int:
        """Load current iteration count from disk."""
        count_file = DATA_DIR / "iteration_count.txt"
        if count_file.exists():
            try:
                return int(count_file.read_text().strip())
            except (ValueError, OSError):
                pass
        return 0

    def _save_iteration_count(self) -> None:
        """Save iteration count to disk."""
        count_file = DATA_DIR / "iteration_count.txt"
        try:
            temp_file = count_file.with_suffix(".tmp")
            temp_file.write_text(str(self.iteration_count), encoding="utf-8")
            temp_file.replace(count_file)
        except OSError as e:
            try:
                with open(count_file, "w", encoding="utf-8") as f:
                    f.write(str(self.iteration_count))
            except OSError:
                print(f"[WARN] Could not save iteration count: {e}")

    async def run_iteration(self) -> IterationReport:
        """Run a single Ralph Loop iteration."""
        self.iteration_count += 1

        if USE_MODULAR:
            report = graph_run_iteration(
                graph=self.graph,
                iteration_number=self.iteration_count,
                config=self.config,
            )
            self._save_iteration_count()
            return report

        # Fallback implementation
        return await self._run_iteration_fallback()

    async def _run_iteration_fallback(self) -> IterationReport:
        """Run iteration using fallback implementation."""
        start_time = time.perf_counter()
        started_at = datetime.now(timezone.utc).isoformat()

        print("=" * 60)
        print(f"RALPH LOOP ITERATION #{self.iteration_count}")
        print("=" * 60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)

        phases: List[PhaseResult] = []

        print("[1/4] Running ecosystem health check...", end=" ", flush=True)
        result = await self.runner.run_ecosystem_check()
        phases.append(result)
        print(f"[{result.status.value.upper()}] {result.message}")

        print("[2/4] Running validation pipeline...", end=" ", flush=True)
        result = await self.runner.run_validation()
        phases.append(result)
        print(f"[{result.status.value.upper()}] {result.message}")

        print("[3/4] Running memory consolidation...", end=" ", flush=True)
        result = await self.runner.run_consolidation()
        phases.append(result)
        print(f"[{result.status.value.upper()}] {result.message}")

        print("[4/4] Updating session state...", end=" ", flush=True)
        result = await self.runner.run_session_update()
        phases.append(result)
        print(f"[{result.status.value.upper()}] {result.message}")

        completed_at = datetime.now(timezone.utc).isoformat()
        total_duration = (time.perf_counter() - start_time) * 1000

        failed = sum(1 for p in phases if p.status == IterationStatus.FAILED)
        warnings = sum(1 for p in phases if p.status == IterationStatus.WARNING)
        success = sum(1 for p in phases if p.status == IterationStatus.SUCCESS)

        if failed > 0:
            overall_status = IterationStatus.FAILED
        elif warnings > 1:
            overall_status = IterationStatus.WARNING
        else:
            overall_status = IterationStatus.SUCCESS

        recommendations = []
        for p in phases:
            if p.status == IterationStatus.FAILED:
                recommendations.append(f"Fix {p.phase.value}: {p.message}")
            elif p.status == IterationStatus.WARNING:
                recommendations.append(f"Investigate {p.phase.value}: {p.message}")

        summary = {
            "success": success,
            "warnings": warnings,
            "failed": failed,
            "phases_run": len(phases),
        }

        print("-" * 60)
        print(f"RESULT: [{overall_status.value.upper()}]")
        print(f"Summary: SUCCESS={success}, WARN={warnings}, FAIL={failed}")
        print(f"Duration: {total_duration:.0f}ms")

        if recommendations:
            print("\nRecommendations:")
            for rec in recommendations[:3]:
                print(f"  - {rec}")

        print("=" * 60)

        report = IterationReport(
            iteration_number=self.iteration_count,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_ms=total_duration,
            overall_status=overall_status,
            phases=phases,
            summary=summary,
            recommendations=recommendations,
        )

        self._save_report(report)
        self._save_iteration_count()
        self._store_memory_block(report)

        return report

    def _save_report(self, report: IterationReport) -> None:
        """Save iteration report to disk."""
        report_file = REPORTS_DIR / f"iteration_{report.iteration_number:04d}.json"
        report_file.write_text(
            json.dumps(report.to_dict(), indent=2),
            encoding="utf-8"
        )

    def _store_memory_block(self, report: IterationReport) -> None:
        """Store iteration result as a WORKING memory block for consolidation."""
        import hashlib

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

        block_file = MEMORY_DIR / f"{block_id}.json"
        block_file.write_text(json.dumps(block, indent=2), encoding="utf-8")

    async def run_daemon(self, interval_seconds: int = 300, max_iterations: int = 0) -> None:
        """Run Ralph Loop as a continuous daemon."""
        print("=" * 60)
        print("RALPH LOOP DAEMON - V10 Ultimate Platform")
        print("=" * 60)
        print(f"Interval: {interval_seconds}s")
        print(f"Max iterations: {'unlimited' if max_iterations == 0 else max_iterations}")
        print("-" * 60)

        iterations_run = 0
        while True:
            await self.run_iteration()
            iterations_run += 1

            if max_iterations > 0 and iterations_run >= max_iterations:
                print(f"\nReached max iterations ({max_iterations}). Stopping.")
                break

            print(f"\nNext iteration in {interval_seconds}s...")
            await asyncio.sleep(interval_seconds)

    def get_status(self) -> Dict[str, Any]:
        """Get current Ralph Loop status."""
        reports = list(REPORTS_DIR.glob("iteration_*.json"))
        latest_report = None

        if reports:
            latest_file = max(reports, key=lambda f: f.stat().st_mtime)
            try:
                latest_report = json.loads(latest_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        return {
            "iteration_count": self.iteration_count,
            "reports_generated": len(reports),
            "latest_report": latest_report,
            "reports_dir": str(REPORTS_DIR),
        }

    async def initialize_all(self) -> None:
        """Initialize all platform systems."""
        print("=" * 60)
        print("RALPH LOOP INITIALIZATION")
        print("=" * 60)

        print("[1/3] Initializing knowledge base...", end=" ", flush=True)
        script = SCRIPT_DIR / "session_continuity.py"
        if script.exists():
            result = subprocess.run(
                [sys.executable, str(script), "init"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                print("[OK]")
            else:
                print("[WARN]")
        else:
            print("[SKIP]")

        print("[2/3] Creating initial session...", end=" ", flush=True)
        if script.exists():
            result = subprocess.run(
                [sys.executable, str(script), "status"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                print("[OK]")
            else:
                print("[WARN]")
        else:
            print("[SKIP]")

        print("[3/3] Running initial validation...", end=" ", flush=True)
        script = SCRIPT_DIR / "auto_validate.py"
        if script.exists():
            result = subprocess.run(
                [sys.executable, str(script), "--quick"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode <= 1:
                print("[OK]")
            else:
                print("[WARN]")
        else:
            print("[SKIP]")

        print("-" * 60)
        print("Initialization complete!")
        print("=" * 60)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ralph Loop Runner - V10 Ultimate Platform",
    )
    parser.add_argument(
        "command",
        choices=["status", "iterate", "daemon", "init", "report"],
        help="Command to execute",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Daemon interval in seconds (default: 300)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        help="Max iterations for daemon (0=unlimited)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )

    args = parser.parse_args()
    loop = RalphLoop()

    if args.command == "status":
        status = loop.get_status()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            if USE_MODULAR:
                print_status(
                    iteration_count=status["iteration_count"],
                    reports_generated=status["reports_generated"],
                    reports_dir=status["reports_dir"],
                    latest_report=status["latest_report"],
                )
            else:
                print("=" * 50)
                print("RALPH LOOP STATUS")
                print("=" * 50)
                print(f"Iteration Count:    {status['iteration_count']}")
                print(f"Reports Generated:  {status['reports_generated']}")
                print(f"Reports Directory:  {status['reports_dir']}")
                if status["latest_report"]:
                    report = status["latest_report"]
                    print("-" * 50)
                    print(f"Latest Iteration:   #{report['iteration_number']}")
                    print(f"Status:             {report['overall_status']}")
                    print(f"Duration:           {report['total_duration_ms']:.0f}ms")
                print("=" * 50)

    elif args.command == "iterate":
        report = await loop.run_iteration()
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))

    elif args.command == "daemon":
        await loop.run_daemon(args.interval, args.max)

    elif args.command == "init":
        await loop.initialize_all()

    elif args.command == "report":
        status = loop.get_status()
        if status["latest_report"]:
            if args.json:
                print(json.dumps(status["latest_report"], indent=2))
            else:
                if USE_MODULAR:
                    print(format_report(status["latest_report"]))
                else:
                    report = status["latest_report"]
                    print("=" * 60)
                    print(f"RALPH LOOP ITERATION #{report['iteration_number']} REPORT")
                    print("=" * 60)
                    print(f"Started:   {report['started_at'][:19]}")
                    print(f"Completed: {report['completed_at'][:19]}")
                    print(f"Duration:  {report['total_duration_ms']:.0f}ms")
                    print(f"Status:    {report['overall_status']}")
                    print("-" * 60)
                    print("PHASES:")
                    for phase in report["phases"]:
                        print(f"  [{phase['status'].upper():7}] {phase['phase']}: {phase['message']}")
                    if report["recommendations"]:
                        print("-" * 60)
                        print("RECOMMENDATIONS:")
                        for rec in report["recommendations"]:
                            print(f"  - {rec}")
                    print("=" * 60)
        else:
            print("No reports available. Run 'ralph_loop.py iterate' first.")


if __name__ == "__main__":
    asyncio.run(main())
