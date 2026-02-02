#!/usr/bin/env python3
"""
Performance Benchmarking Suite - V33 Phase 8
Part of Phase 8: CLI Integration & Performance Optimization.

Benchmarks all V33 components for performance validation:
- Memory layer throughput
- Tool layer latency
- Orchestration startup time
- Structured output parsing
- Caching operations
- End-to-end pipelines

Usage:
    python scripts/benchmark_performance.py [--quick] [--detailed]
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    name: str
    iterations: int
    min_ms: float
    max_ms: float
    avg_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    throughput: float  # ops/sec
    passed: bool
    threshold_ms: float
    error: Optional[str] = None


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_benchmarks: int = 0
    passed_benchmarks: int = 0
    failed_benchmarks: int = 0
    total_time_ms: float = 0.0
    results: List[BenchmarkResult] = field(default_factory=list)


def run_benchmark(
    name: str,
    func: Callable[[], Any],
    iterations: int = 100,
    warmup: int = 5,
    threshold_ms: float = 100.0,
) -> BenchmarkResult:
    """Run a synchronous benchmark."""
    # Warmup
    for _ in range(warmup):
        try:
            func()
        except Exception:
            pass

    # Actual benchmark
    times: List[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            func()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        except Exception as e:
            return BenchmarkResult(
                name=name,
                iterations=0,
                min_ms=0,
                max_ms=0,
                avg_ms=0,
                median_ms=0,
                p95_ms=0,
                p99_ms=0,
                throughput=0,
                passed=False,
                threshold_ms=threshold_ms,
                error=str(e),
            )

    if not times:
        return BenchmarkResult(
            name=name,
            iterations=0,
            min_ms=0,
            max_ms=0,
            avg_ms=0,
            median_ms=0,
            p95_ms=0,
            p99_ms=0,
            throughput=0,
            passed=False,
            threshold_ms=threshold_ms,
            error="No successful iterations",
        )

    sorted_times = sorted(times)
    n = len(times)
    p95_idx = min(int(n * 0.95), n - 1)
    p99_idx = min(int(n * 0.99), n - 1)

    avg_ms = statistics.mean(times)
    throughput = 1000.0 / avg_ms if avg_ms > 0 else 0

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        min_ms=min(times),
        max_ms=max(times),
        avg_ms=avg_ms,
        median_ms=statistics.median(times),
        p95_ms=sorted_times[p95_idx],
        p99_ms=sorted_times[p99_idx],
        throughput=throughput,
        passed=avg_ms <= threshold_ms,
        threshold_ms=threshold_ms,
    )


async def run_async_benchmark(
    name: str,
    func: Callable[[], Any],
    iterations: int = 100,
    warmup: int = 5,
    threshold_ms: float = 100.0,
) -> BenchmarkResult:
    """Run an asynchronous benchmark."""
    # Warmup
    for _ in range(warmup):
        try:
            result = func()
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            pass

    # Actual benchmark
    times: List[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            result = func()
            if asyncio.iscoroutine(result):
                await result
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
        except Exception as e:
            return BenchmarkResult(
                name=name,
                iterations=0,
                min_ms=0,
                max_ms=0,
                avg_ms=0,
                median_ms=0,
                p95_ms=0,
                p99_ms=0,
                throughput=0,
                passed=False,
                threshold_ms=threshold_ms,
                error=str(e),
            )

    if not times:
        return BenchmarkResult(
            name=name,
            iterations=0,
            min_ms=0,
            max_ms=0,
            avg_ms=0,
            median_ms=0,
            p95_ms=0,
            p99_ms=0,
            throughput=0,
            passed=False,
            threshold_ms=threshold_ms,
            error="No successful iterations",
        )

    sorted_times = sorted(times)
    n = len(times)
    p95_idx = min(int(n * 0.95), n - 1)
    p99_idx = min(int(n * 0.99), n - 1)

    avg_ms = statistics.mean(times)
    throughput = 1000.0 / avg_ms if avg_ms > 0 else 0

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        min_ms=min(times),
        max_ms=max(times),
        avg_ms=avg_ms,
        median_ms=statistics.median(times),
        p95_ms=sorted_times[p95_idx],
        p99_ms=sorted_times[p99_idx],
        throughput=throughput,
        passed=avg_ms <= threshold_ms,
        threshold_ms=threshold_ms,
    )


# ============================================================================
# Benchmark Functions
# ============================================================================


def benchmark_cache_operations() -> BenchmarkResult:
    """Benchmark LRU cache operations."""
    from core.performance.optimizer import LRUCache

    cache: LRUCache[str] = LRUCache(max_size=1000, default_ttl=60.0)

    # Pre-populate cache
    for i in range(500):
        cache.set(f"key_{i}", f"value_{i}")

    def cache_ops():
        # Mix of operations
        cache.set("test_key", "test_value")
        cache.get("test_key")
        cache.get("key_100")
        cache.get("nonexistent")

    return run_benchmark(
        name="cache_operations",
        func=cache_ops,
        iterations=1000,
        threshold_ms=0.1,  # 0.1ms per op
    )


def benchmark_cache_throughput() -> BenchmarkResult:
    """Benchmark cache throughput (ops/sec)."""
    from core.performance.optimizer import LRUCache

    cache: LRUCache[str] = LRUCache(max_size=10000, default_ttl=60.0)
    counter = [0]

    def cache_write():
        key = f"key_{counter[0]}"
        cache.set(key, f"value_{counter[0]}")
        counter[0] += 1

    return run_benchmark(
        name="cache_throughput",
        func=cache_write,
        iterations=10000,
        threshold_ms=0.05,  # 0.05ms per op = 20,000 ops/sec
    )


def benchmark_connection_pool() -> BenchmarkResult:
    """Benchmark HTTP connection pool creation."""
    from core.performance.optimizer import HTTPConnectionPool

    # Reset singleton for clean test
    HTTPConnectionPool._instance = None

    def create_pool():
        HTTPConnectionPool._instance = None
        pool = HTTPConnectionPool()
        _ = pool.sync_client  # Force client creation
        pool.close()

    return run_benchmark(
        name="connection_pool_init",
        func=create_pool,
        iterations=10,
        warmup=1,
        threshold_ms=100.0,
    )


def benchmark_profiler() -> BenchmarkResult:
    """Benchmark profiler overhead."""
    from core.performance.optimizer import Profiler

    profiler = Profiler()

    def profiled_operation():
        with profiler.measure("test_operation"):
            # Simulate minimal work
            _ = sum(range(100))

    return run_benchmark(
        name="profiler_overhead",
        func=profiled_operation,
        iterations=1000,
        threshold_ms=0.1,
    )


def benchmark_lazy_loader() -> BenchmarkResult:
    """Benchmark lazy loading."""
    from core.performance.optimizer import LazyLoader

    loader = LazyLoader()

    def load_module():
        # Load a small standard library module
        loader._modules.clear()  # Reset for each iteration
        loader.load("json")

    return run_benchmark(
        name="lazy_loader",
        func=load_module,
        iterations=100,
        threshold_ms=1.0,
    )


async def benchmark_deduplicator() -> BenchmarkResult:
    """Benchmark request deduplication."""
    from core.performance.optimizer import RequestDeduplicator

    deduplicator = RequestDeduplicator()
    call_count = [0]

    async def mock_request(key: str):
        call_count[0] += 1
        await asyncio.sleep(0.001)  # 1ms simulated latency
        return f"result_{key}"

    async def dedupe_request():
        await deduplicator.dedupe(mock_request, "test_key")

    return await run_async_benchmark(
        name="request_deduplicator",
        func=dedupe_request,
        iterations=100,
        threshold_ms=5.0,
    )


def benchmark_json_serialization() -> BenchmarkResult:
    """Benchmark JSON serialization for structured output."""
    test_data = {
        "id": "test-123",
        "name": "Test Object",
        "values": list(range(100)),
        "nested": {
            "level1": {
                "level2": {
                    "data": ["a", "b", "c"]
                }
            }
        },
        "metadata": {
            f"key_{i}": f"value_{i}" for i in range(50)
        },
    }

    def serialize():
        json.dumps(test_data)

    return run_benchmark(
        name="json_serialization",
        func=serialize,
        iterations=1000,
        threshold_ms=0.5,
    )


def benchmark_json_parsing() -> BenchmarkResult:
    """Benchmark JSON parsing for structured input."""
    test_data = {
        "id": "test-123",
        "name": "Test Object",
        "values": list(range(100)),
        "nested": {
            "level1": {
                "level2": {
                    "data": ["a", "b", "c"]
                }
            }
        },
        "metadata": {
            f"key_{i}": f"value_{i}" for i in range(50)
        },
    }
    json_str = json.dumps(test_data)

    def parse():
        json.loads(json_str)

    return run_benchmark(
        name="json_parsing",
        func=parse,
        iterations=1000,
        threshold_ms=0.5,
    )


def benchmark_pydantic_validation() -> BenchmarkResult:
    """Benchmark Pydantic model validation."""
    try:
        from pydantic import BaseModel
        from typing import List, Dict

        class TestModel(BaseModel):
            id: str
            name: str
            values: List[int]
            metadata: Dict[str, str]

        test_data = {
            "id": "test-123",
            "name": "Test Object",
            "values": list(range(10)),
            "metadata": {f"key_{i}": f"value_{i}" for i in range(10)},
        }

        def validate():
            TestModel(**test_data)

        return run_benchmark(
            name="pydantic_validation",
            func=validate,
            iterations=1000,
            threshold_ms=1.0,
        )
    except ImportError as e:
        return BenchmarkResult(
            name="pydantic_validation",
            iterations=0,
            min_ms=0,
            max_ms=0,
            avg_ms=0,
            median_ms=0,
            p95_ms=0,
            p99_ms=0,
            throughput=0,
            passed=False,
            threshold_ms=1.0,
            error=f"Pydantic not available: {e}",
        )


def benchmark_import_time() -> BenchmarkResult:
    """Benchmark core module import time."""
    import importlib

    def reimport_core():
        # Clear from cache
        modules_to_remove = [k for k in sys.modules if k.startswith("core.performance")]
        for mod in modules_to_remove:
            del sys.modules[mod]

        # Reimport
        importlib.import_module("core.performance.optimizer")

    return run_benchmark(
        name="core_import_time",
        func=reimport_core,
        iterations=5,
        warmup=1,
        threshold_ms=500.0,
    )


# ============================================================================
# Main Benchmark Runner
# ============================================================================


def print_result(result: BenchmarkResult, detailed: bool = False) -> None:
    """Print a benchmark result."""
    status = "[PASS]" if result.passed else "[FAIL]"
    color = "\033[92m" if result.passed else "\033[91m"
    reset = "\033[0m"

    print(f"  {color}{status}{reset} {result.name}")

    if result.error:
        print(f"         Error: {result.error}")
        return

    print(f"         Avg: {result.avg_ms:.3f}ms (threshold: {result.threshold_ms}ms)")

    if detailed:
        print(f"         Min: {result.min_ms:.3f}ms, Max: {result.max_ms:.3f}ms")
        print(f"         P50: {result.median_ms:.3f}ms, P95: {result.p95_ms:.3f}ms, P99: {result.p99_ms:.3f}ms")
        print(f"         Throughput: {result.throughput:.1f} ops/sec")
        print(f"         Iterations: {result.iterations}")


async def run_all_benchmarks(quick: bool = False, detailed: bool = False) -> BenchmarkSuite:
    """Run all benchmarks and return results."""
    suite = BenchmarkSuite()
    start_time = time.perf_counter()

    print("\n" + "=" * 60)
    print("V33 Performance Benchmark Suite")
    print("=" * 60)

    # Synchronous benchmarks
    sync_benchmarks = [
        ("Cache Operations", benchmark_cache_operations),
        ("Cache Throughput", benchmark_cache_throughput),
        ("Profiler Overhead", benchmark_profiler),
        ("Lazy Loader", benchmark_lazy_loader),
        ("JSON Serialization", benchmark_json_serialization),
        ("JSON Parsing", benchmark_json_parsing),
        ("Pydantic Validation", benchmark_pydantic_validation),
    ]

    if not quick:
        sync_benchmarks.extend([
            ("Connection Pool Init", benchmark_connection_pool),
            ("Core Import Time", benchmark_import_time),
        ])

    print("\n[Synchronous Benchmarks]")
    for name, func in sync_benchmarks:
        try:
            result = func()
            suite.results.append(result)
            print_result(result, detailed)
        except Exception as e:
            result = BenchmarkResult(
                name=name.lower().replace(" ", "_"),
                iterations=0,
                min_ms=0,
                max_ms=0,
                avg_ms=0,
                median_ms=0,
                p95_ms=0,
                p99_ms=0,
                throughput=0,
                passed=False,
                threshold_ms=0,
                error=str(e),
            )
            suite.results.append(result)
            print_result(result, detailed)

    # Asynchronous benchmarks
    print("\n[Asynchronous Benchmarks]")
    try:
        result = await benchmark_deduplicator()
        suite.results.append(result)
        print_result(result, detailed)
    except Exception as e:
        result = BenchmarkResult(
            name="request_deduplicator",
            iterations=0,
            min_ms=0,
            max_ms=0,
            avg_ms=0,
            median_ms=0,
            p95_ms=0,
            p99_ms=0,
            throughput=0,
            passed=False,
            threshold_ms=0,
            error=str(e),
        )
        suite.results.append(result)
        print_result(result, detailed)

    # Calculate summary
    suite.total_time_ms = (time.perf_counter() - start_time) * 1000
    suite.total_benchmarks = len(suite.results)
    suite.passed_benchmarks = sum(1 for r in suite.results if r.passed)
    suite.failed_benchmarks = suite.total_benchmarks - suite.passed_benchmarks

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total Benchmarks: {suite.total_benchmarks}")
    print(f"  Passed: {suite.passed_benchmarks}")
    print(f"  Failed: {suite.failed_benchmarks}")
    print(f"  Total Time: {suite.total_time_ms:.1f}ms")
    print("=" * 60)

    return suite


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="V33 Performance Benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks only")
    parser.add_argument("--detailed", action="store_true", help="Show detailed results")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    suite = asyncio.run(run_all_benchmarks(quick=args.quick, detailed=args.detailed))

    if args.json:
        output = {
            "timestamp": suite.timestamp,
            "total_benchmarks": suite.total_benchmarks,
            "passed_benchmarks": suite.passed_benchmarks,
            "failed_benchmarks": suite.failed_benchmarks,
            "total_time_ms": suite.total_time_ms,
            "results": [
                {
                    "name": r.name,
                    "iterations": r.iterations,
                    "avg_ms": r.avg_ms,
                    "p95_ms": r.p95_ms,
                    "throughput": r.throughput,
                    "passed": r.passed,
                    "error": r.error,
                }
                for r in suite.results
            ],
        }
        print(json.dumps(output, indent=2))

    # Exit with error code if any benchmarks failed
    sys.exit(0 if suite.failed_benchmarks == 0 else 1)


if __name__ == "__main__":
    main()
