#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "structlog>=24.1.0",
#     "pydantic>=2.0.0",
#     "psutil>=5.9.0",
# ]
# [tool.uv]
# exclude-newer = "2026-02-01"
# ///
"""
Performance Optimization Module - Ultimate Autonomous Platform

Comprehensive profiling and optimization for Ralph Loop iterations:
1. Async-aware profiling (yappi-style coroutine tracking)
2. Memory profiling with tracemalloc
3. Execution timing with context managers
4. Hot path identification
5. Token efficiency metrics
6. Optimization recommendations

Usage:
    uv run performance.py status          # Show performance metrics
    uv run performance.py profile <cmd>   # Profile a command
    uv run performance.py memory          # Memory analysis
    uv run performance.py optimize        # Run optimization analysis
    uv run performance.py benchmark       # Run benchmark suite

Platform: Windows 11 + Python 3.11+
Architecture: V10 Optimized (Verified, Minimal, Seamless)
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import gc
import io
import json
import os
import pstats
import subprocess
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

import structlog

# Optional imports for enhanced profiling
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
V10_DIR = SCRIPT_DIR.parent
DATA_DIR = V10_DIR / "data"
PERF_DIR = DATA_DIR / "performance"
PERF_DIR.mkdir(parents=True, exist_ok=True)

# Windows compatibility
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

logger = structlog.get_logger(__name__)

T = TypeVar("T")


# =============================================================================
# Data Models
# =============================================================================

class ProfileType(str, Enum):
    """Type of profiling to perform."""
    CPU = "cpu"
    MEMORY = "memory"
    ASYNC = "async"
    IO = "io"
    FULL = "full"


class OptimizationLevel(str, Enum):
    """Severity of optimization recommendation."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class TimingResult:
    """Result of a timing measurement."""
    name: str
    duration_ms: float
    start_time: float
    end_time: float
    memory_delta_mb: float = 0.0
    cpu_percent: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemorySnapshot:
    """Memory state at a point in time."""
    timestamp: str
    rss_mb: float
    vms_mb: float
    heap_mb: float
    gc_objects: int
    top_allocations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ProfileResult:
    """Result of a profiling session."""
    profile_type: ProfileType
    duration_ms: float
    timestamp: str
    function_stats: List[Dict[str, Any]] = field(default_factory=list)
    memory_stats: Optional[MemorySnapshot] = None
    async_stats: Dict[str, Any] = field(default_factory=dict)
    hotspots: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class OptimizationRecommendation:
    """A specific optimization recommendation."""
    level: OptimizationLevel
    category: str
    title: str
    description: str
    impact: str
    action: str
    code_location: Optional[str] = None


@dataclass
class PerformanceReport:
    """Complete performance analysis report."""
    timestamp: str
    duration_ms: float
    system_info: Dict[str, Any]
    timings: List[TimingResult] = field(default_factory=list)
    profiles: List[ProfileResult] = field(default_factory=list)
    memory: Optional[MemorySnapshot] = None
    recommendations: List[OptimizationRecommendation] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Timing Utilities
# =============================================================================

class Timer:
    """High-precision timer for measuring execution time."""

    def __init__(self, name: str = "unnamed"):
        self.name = name
        self.start_time: float = 0
        self.end_time: float = 0
        self.start_memory: int = 0
        self.end_memory: int = 0
        self._running = False

    def start(self) -> "Timer":
        """Start the timer."""
        self.start_time = time.perf_counter()
        if HAS_PSUTIL:
            proc = psutil.Process()
            self.start_memory = proc.memory_info().rss
        self._running = True
        return self

    def stop(self) -> TimingResult:
        """Stop the timer and return result."""
        self.end_time = time.perf_counter()
        if HAS_PSUTIL:
            proc = psutil.Process()
            self.end_memory = proc.memory_info().rss
        self._running = False

        duration_ms = (self.end_time - self.start_time) * 1000
        memory_delta_mb = (self.end_memory - self.start_memory) / (1024 * 1024)

        return TimingResult(
            name=self.name,
            duration_ms=duration_ms,
            start_time=self.start_time,
            end_time=self.end_time,
            memory_delta_mb=memory_delta_mb,
        )

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self._running:
            return (time.perf_counter() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000


@contextmanager
def timed(name: str = "operation"):
    """Context manager for timing operations."""
    timer = Timer(name).start()
    try:
        yield timer
    finally:
        result = timer.stop()
        logger.debug(f"{name} completed", duration_ms=result.duration_ms)


def timed_async(name: str = "async_operation"):
    """Decorator for timing async functions."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            timer = Timer(name).start()
            try:
                return await func(*args, **kwargs)
            finally:
                result = timer.stop()
                logger.debug(f"{name} completed", duration_ms=result.duration_ms)
        return wrapper
    return decorator


# =============================================================================
# Memory Profiling
# =============================================================================

class MemoryProfiler:
    """Memory profiling utilities."""

    def __init__(self):
        self.snapshots: List[MemorySnapshot] = []
        self._tracing = False

    def start_tracing(self) -> None:
        """Start memory tracing."""
        if not self._tracing:
            tracemalloc.start()
            self._tracing = True

    def stop_tracing(self) -> None:
        """Stop memory tracing."""
        if self._tracing:
            tracemalloc.stop()
            self._tracing = False

    def take_snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot."""
        gc.collect()  # Force garbage collection first

        # Get process memory
        rss_mb = 0.0
        vms_mb = 0.0
        if HAS_PSUTIL:
            proc = psutil.Process()
            mem_info = proc.memory_info()
            rss_mb = mem_info.rss / (1024 * 1024)
            vms_mb = mem_info.vms / (1024 * 1024)

        # Get heap allocations from tracemalloc
        heap_mb = 0.0
        top_allocations = []
        if self._tracing:
            current, peak = tracemalloc.get_traced_memory()
            heap_mb = current / (1024 * 1024)

            # Get top allocations
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics("lineno")[:10]
            for stat in top_stats:
                top_allocations.append({
                    "file": str(stat.traceback),
                    "size_kb": stat.size / 1024,
                    "count": stat.count,
                })

        # GC stats
        gc_objects = len(gc.get_objects())

        snapshot = MemorySnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            rss_mb=rss_mb,
            vms_mb=vms_mb,
            heap_mb=heap_mb,
            gc_objects=gc_objects,
            top_allocations=top_allocations,
        )
        self.snapshots.append(snapshot)
        return snapshot

    def get_memory_growth(self) -> float:
        """Get memory growth since first snapshot in MB."""
        if len(self.snapshots) < 2:
            return 0.0
        return self.snapshots[-1].rss_mb - self.snapshots[0].rss_mb

    def detect_leaks(self, threshold_mb: float = 10.0) -> List[Dict[str, Any]]:
        """Detect potential memory leaks."""
        leaks = []
        growth = self.get_memory_growth()

        if growth > threshold_mb:
            leaks.append({
                "type": "memory_growth",
                "growth_mb": growth,
                "message": f"Memory grew by {growth:.1f}MB during profiling",
            })

        # Check for growing object count
        if len(self.snapshots) >= 2:
            obj_growth = self.snapshots[-1].gc_objects - self.snapshots[0].gc_objects
            if obj_growth > 10000:
                leaks.append({
                    "type": "object_growth",
                    "count": obj_growth,
                    "message": f"Object count grew by {obj_growth} during profiling",
                })

        return leaks


# =============================================================================
# CPU Profiling
# =============================================================================

class CPUProfiler:
    """CPU profiling utilities using cProfile."""

    def __init__(self):
        self.profiler: Optional[cProfile.Profile] = None
        self.results: List[ProfileResult] = []

    def start(self) -> None:
        """Start CPU profiling."""
        self.profiler = cProfile.Profile()
        self.profiler.enable()

    def stop(self) -> ProfileResult:
        """Stop profiling and return results."""
        if not self.profiler:
            raise RuntimeError("Profiler not started")

        self.profiler.disable()

        # Get stats
        stream = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=stream)
        stats.sort_stats("cumulative")
        stats.print_stats(20)

        # Parse function stats
        function_stats = []
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            filename, line, name = func
            function_stats.append({
                "function": name,
                "file": filename,
                "line": line,
                "calls": nc,
                "total_time_ms": tt * 1000,
                "cumulative_time_ms": ct * 1000,
                "time_per_call_ms": (ct / nc * 1000) if nc > 0 else 0,
            })

        # Sort by cumulative time
        function_stats.sort(key=lambda x: x["cumulative_time_ms"], reverse=True)

        # Identify hotspots (functions taking >5% of total time)
        total_time = sum(f["cumulative_time_ms"] for f in function_stats[:1]) or 1
        hotspots = [
            {
                "function": f["function"],
                "file": f["file"],
                "time_ms": f["cumulative_time_ms"],
                "percent": (f["cumulative_time_ms"] / total_time) * 100,
            }
            for f in function_stats[:10]
            if f["cumulative_time_ms"] > total_time * 0.05
        ]

        result = ProfileResult(
            profile_type=ProfileType.CPU,
            duration_ms=total_time,
            timestamp=datetime.now(timezone.utc).isoformat(),
            function_stats=function_stats[:20],
            hotspots=hotspots,
        )
        self.results.append(result)
        return result

    @contextmanager
    def profile(self):
        """Context manager for profiling."""
        self.start()
        try:
            yield self
        finally:
            self.stop()


# =============================================================================
# Async Profiling
# =============================================================================

class AsyncProfiler:
    """Profiler for asyncio coroutines."""

    def __init__(self):
        self.coroutine_times: Dict[str, List[float]] = {}
        self.task_counts: Dict[str, int] = {}
        self._start_time: float = 0

    def start(self) -> None:
        """Start async profiling."""
        self._start_time = time.perf_counter()

    def record_coroutine(self, name: str, duration_ms: float) -> None:
        """Record a coroutine execution."""
        if name not in self.coroutine_times:
            self.coroutine_times[name] = []
            self.task_counts[name] = 0

        self.coroutine_times[name].append(duration_ms)
        self.task_counts[name] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get async profiling statistics."""
        stats = {}

        for name, times in self.coroutine_times.items():
            if times:
                stats[name] = {
                    "count": len(times),
                    "total_ms": sum(times),
                    "avg_ms": sum(times) / len(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                }

        return {
            "coroutines": stats,
            "total_tasks": sum(self.task_counts.values()),
            "unique_coroutines": len(self.coroutine_times),
        }

    def wrap_coroutine(self, name: str):
        """Decorator to wrap and profile a coroutine."""
        def decorator(coro_func):
            @wraps(coro_func)
            async def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return await coro_func(*args, **kwargs)
                finally:
                    duration = (time.perf_counter() - start) * 1000
                    self.record_coroutine(name, duration)
            return wrapper
        return decorator


# =============================================================================
# System Information
# =============================================================================

def get_system_info() -> Dict[str, Any]:
    """Get current system information."""
    info = {
        "python_version": sys.version,
        "platform": sys.platform,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if HAS_PSUTIL:
        info["cpu_count"] = psutil.cpu_count()
        info["cpu_percent"] = psutil.cpu_percent(interval=0.1)

        mem = psutil.virtual_memory()
        info["memory_total_gb"] = mem.total / (1024**3)
        info["memory_available_gb"] = mem.available / (1024**3)
        info["memory_percent"] = mem.percent

        disk = psutil.disk_usage("/")
        info["disk_total_gb"] = disk.total / (1024**3)
        info["disk_free_gb"] = disk.free / (1024**3)

    return info


# =============================================================================
# Optimization Analyzer
# =============================================================================

class OptimizationAnalyzer:
    """Analyzes performance data and generates recommendations."""

    def __init__(self):
        self.recommendations: List[OptimizationRecommendation] = []

    def analyze_profile(self, profile: ProfileResult) -> List[OptimizationRecommendation]:
        """Analyze a profile and generate recommendations."""
        recs = []

        # Check for slow functions
        for hotspot in profile.hotspots:
            if hotspot["percent"] > 20:
                recs.append(OptimizationRecommendation(
                    level=OptimizationLevel.HIGH,
                    category="CPU",
                    title=f"Hot Function: {hotspot['function']}",
                    description=f"Function takes {hotspot['percent']:.1f}% of execution time",
                    impact="Significant CPU time reduction possible",
                    action="Consider optimizing or caching results",
                    code_location=hotspot.get("file"),
                ))

        return recs

    def analyze_memory(self, snapshot: MemorySnapshot) -> List[OptimizationRecommendation]:
        """Analyze memory snapshot for issues."""
        recs = []

        # High memory usage
        if snapshot.rss_mb > 500:
            recs.append(OptimizationRecommendation(
                level=OptimizationLevel.MEDIUM,
                category="Memory",
                title="High Memory Usage",
                description=f"Process using {snapshot.rss_mb:.1f}MB RSS",
                impact="May cause OOM on constrained systems",
                action="Review large data structures and caching",
            ))

        # Too many objects
        if snapshot.gc_objects > 100000:
            recs.append(OptimizationRecommendation(
                level=OptimizationLevel.MEDIUM,
                category="Memory",
                title="High Object Count",
                description=f"{snapshot.gc_objects:,} objects in memory",
                impact="GC pressure may cause latency spikes",
                action="Review object creation patterns",
            ))

        return recs

    def analyze_timings(self, timings: List[TimingResult]) -> List[OptimizationRecommendation]:
        """Analyze timing results for optimization opportunities."""
        recs = []

        for timing in timings:
            if timing.duration_ms > 5000:
                recs.append(OptimizationRecommendation(
                    level=OptimizationLevel.HIGH,
                    category="Latency",
                    title=f"Slow Operation: {timing.name}",
                    description=f"Operation took {timing.duration_ms:.0f}ms",
                    impact="User-facing latency impact",
                    action="Consider async execution or caching",
                ))
            elif timing.duration_ms > 1000:
                recs.append(OptimizationRecommendation(
                    level=OptimizationLevel.MEDIUM,
                    category="Latency",
                    title=f"Moderate Latency: {timing.name}",
                    description=f"Operation took {timing.duration_ms:.0f}ms",
                    impact="May impact responsiveness",
                    action="Review for optimization opportunities",
                ))

        return recs

    def generate_report(
        self,
        timings: List[TimingResult],
        profiles: List[ProfileResult],
        memory: Optional[MemorySnapshot],
    ) -> PerformanceReport:
        """Generate a complete performance report."""
        all_recs = []

        for profile in profiles:
            all_recs.extend(self.analyze_profile(profile))

        if memory:
            all_recs.extend(self.analyze_memory(memory))

        all_recs.extend(self.analyze_timings(timings))

        # Sort by level
        level_order = {
            OptimizationLevel.CRITICAL: 0,
            OptimizationLevel.HIGH: 1,
            OptimizationLevel.MEDIUM: 2,
            OptimizationLevel.LOW: 3,
            OptimizationLevel.INFO: 4,
        }
        all_recs.sort(key=lambda r: level_order[r.level])

        # Generate summary
        summary = {
            "total_recommendations": len(all_recs),
            "by_level": {},
            "by_category": {},
        }

        for rec in all_recs:
            summary["by_level"][rec.level.value] = summary["by_level"].get(rec.level.value, 0) + 1
            summary["by_category"][rec.category] = summary["by_category"].get(rec.category, 0) + 1

        total_duration = sum(t.duration_ms for t in timings)

        return PerformanceReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_ms=total_duration,
            system_info=get_system_info(),
            timings=timings,
            profiles=profiles,
            memory=memory,
            recommendations=all_recs,
            summary=summary,
        )


# =============================================================================
# Performance Runner
# =============================================================================

class PerformanceRunner:
    """Main performance analysis runner."""

    def __init__(self):
        self.memory_profiler = MemoryProfiler()
        self.cpu_profiler = CPUProfiler()
        self.async_profiler = AsyncProfiler()
        self.analyzer = OptimizationAnalyzer()
        self.timings: List[TimingResult] = []

    async def profile_module(self, module_path: Path) -> ProfileResult:
        """Profile a Python module."""
        timer = Timer(module_path.name).start()

        try:
            result = subprocess.run(
                [sys.executable, str(module_path), "--quick", "--json"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(module_path.parent),
            )
            timing = timer.stop()
            self.timings.append(timing)

            return ProfileResult(
                profile_type=ProfileType.CPU,
                duration_ms=timing.duration_ms,
                timestamp=datetime.now(timezone.utc).isoformat(),
                async_stats={
                    "exit_code": result.returncode,
                    "stdout_lines": len(result.stdout.split("\n")),
                    "stderr_lines": len(result.stderr.split("\n")),
                },
            )
        except Exception as e:
            timing = timer.stop()
            return ProfileResult(
                profile_type=ProfileType.CPU,
                duration_ms=timing.duration_ms,
                timestamp=datetime.now(timezone.utc).isoformat(),
                async_stats={"error": str(e)},
            )

    async def run_benchmark_suite(self) -> PerformanceReport:
        """Run benchmarks on all V10 modules."""
        print("=" * 60)
        print("PERFORMANCE BENCHMARK SUITE - V10 Ultimate Platform")
        print("=" * 60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)

        # Start memory tracing
        self.memory_profiler.start_tracing()
        initial_snapshot = self.memory_profiler.take_snapshot()

        profiles = []

        # Profile each module
        modules = [
            SCRIPT_DIR / "ecosystem_orchestrator.py",
            SCRIPT_DIR / "auto_validate.py",
            SCRIPT_DIR / "sleeptime_compute.py",
            SCRIPT_DIR / "session_continuity.py",
        ]

        for i, module in enumerate(modules, 1):
            if module.exists():
                print(f"[{i}/{len(modules)}] Profiling {module.name}...", end=" ", flush=True)
                profile = await self.profile_module(module)
                profiles.append(profile)
                print(f"[{profile.duration_ms:.0f}ms]")
            else:
                print(f"[{i}/{len(modules)}] {module.name} not found [SKIP]")

        # Final memory snapshot
        final_snapshot = self.memory_profiler.take_snapshot()
        self.memory_profiler.stop_tracing()

        # Generate report
        report = self.analyzer.generate_report(
            timings=self.timings,
            profiles=profiles,
            memory=final_snapshot,
        )

        # Print summary
        print("-" * 60)
        print(f"BENCHMARK COMPLETE")
        print(f"Total Duration: {report.duration_ms:.0f}ms")
        print(f"Memory Growth: {self.memory_profiler.get_memory_growth():.1f}MB")
        print(f"Recommendations: {len(report.recommendations)}")

        if report.recommendations:
            print("\nTop Recommendations:")
            for rec in report.recommendations[:5]:
                print(f"  [{rec.level.value.upper()}] {rec.title}")

        print("=" * 60)

        # Save report
        report_file = PERF_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self._save_report(report, report_file)
        print(f"Report saved: {report_file}")

        return report

    def _save_report(self, report: PerformanceReport, path: Path) -> None:
        """Save report to JSON file."""
        seen = set()

        def serialize(obj, depth: int = 0):
            # Prevent infinite recursion
            if depth > 20:
                return str(obj)[:100]

            obj_id = id(obj)
            if obj_id in seen and hasattr(obj, "__dict__"):
                return f"<circular ref: {type(obj).__name__}>"
            seen.add(obj_id)

            if obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, list):
                return [serialize(i, depth + 1) for i in obj[:100]]  # Limit list size
            elif isinstance(obj, dict):
                return {str(k): serialize(v, depth + 1) for k, v in list(obj.items())[:50]}
            elif hasattr(obj, "__dict__"):
                return {k: serialize(v, depth + 1) for k, v in obj.__dict__.items() if not k.startswith("_")}
            else:
                return str(obj)[:200]

        with open(path, "w") as f:
            json.dump(serialize(report), f, indent=2)

    async def get_status(self) -> Dict[str, Any]:
        """Get current performance status."""
        snapshot = self.memory_profiler.take_snapshot()
        system = get_system_info()

        # Count report files
        report_files = list(PERF_DIR.glob("*.json"))

        return {
            "system": system,
            "memory": {
                "rss_mb": snapshot.rss_mb,
                "gc_objects": snapshot.gc_objects,
            },
            "reports": {
                "count": len(report_files),
                "directory": str(PERF_DIR),
            },
        }


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Performance Optimization Module for Ultimate Autonomous Platform",
    )
    parser.add_argument(
        "command",
        choices=["status", "profile", "memory", "optimize", "benchmark"],
        help="Command to run",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()
    runner = PerformanceRunner()

    if args.command == "status":
        status = await runner.get_status()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print("=" * 50)
            print("PERFORMANCE STATUS")
            print("=" * 50)
            print(f"Python: {status['system']['python_version'].split()[0]}")
            if HAS_PSUTIL:
                print(f"CPU: {status['system']['cpu_percent']}% ({status['system']['cpu_count']} cores)")
                print(f"Memory: {status['system']['memory_percent']}% used")
            print(f"Process RSS: {status['memory']['rss_mb']:.1f}MB")
            print(f"GC Objects: {status['memory']['gc_objects']:,}")
            print(f"Reports: {status['reports']['count']} saved")
            print("=" * 50)

    elif args.command == "memory":
        profiler = MemoryProfiler()
        profiler.start_tracing()
        snapshot = profiler.take_snapshot()
        profiler.stop_tracing()

        if args.json:
            print(json.dumps({
                "rss_mb": snapshot.rss_mb,
                "vms_mb": snapshot.vms_mb,
                "heap_mb": snapshot.heap_mb,
                "gc_objects": snapshot.gc_objects,
                "top_allocations": snapshot.top_allocations,
            }, indent=2))
        else:
            print("=" * 50)
            print("MEMORY ANALYSIS")
            print("=" * 50)
            print(f"RSS: {snapshot.rss_mb:.1f}MB")
            print(f"VMS: {snapshot.vms_mb:.1f}MB")
            print(f"Heap (traced): {snapshot.heap_mb:.1f}MB")
            print(f"GC Objects: {snapshot.gc_objects:,}")
            if snapshot.top_allocations:
                print("\nTop Allocations:")
                for alloc in snapshot.top_allocations[:5]:
                    print(f"  {alloc['size_kb']:.1f}KB - {alloc['file'][:50]}")
            print("=" * 50)

    elif args.command == "benchmark":
        await runner.run_benchmark_suite()

    elif args.command == "optimize":
        print("Running optimization analysis...")
        report = await runner.run_benchmark_suite()

        print("\n" + "=" * 60)
        print("OPTIMIZATION RECOMMENDATIONS")
        print("=" * 60)

        if not report.recommendations:
            print("No optimization issues detected!")
        else:
            for rec in report.recommendations:
                print(f"\n[{rec.level.value.upper()}] {rec.title}")
                print(f"  Category: {rec.category}")
                print(f"  {rec.description}")
                print(f"  Impact: {rec.impact}")
                print(f"  Action: {rec.action}")
                if rec.code_location:
                    print(f"  Location: {rec.code_location}")

        print("=" * 60)

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
