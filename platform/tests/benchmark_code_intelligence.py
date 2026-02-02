#!/usr/bin/env python3
"""
UNLEASH Code Intelligence Benchmark Suite
==========================================
Part of the Unified Code Intelligence Architecture (5-Layer, 2026)

Benchmarks:
1. Embedding throughput (chunks/second)
2. Search latency (ms per query)
3. Search accuracy (relevance ranking)
4. Memory usage
5. Cold start time

Usage:
    python platform/tests/benchmark_code_intelligence.py [--quick] [--full]

Requirements:
    pip install voyageai qdrant-client psutil tabulate
    export VOYAGE_API_KEY=your_key
"""

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import psutil

# Project setup
PROJECT_ROOT = Path("Z:/insider/AUTO CLAUDE/unleash")
sys.path.insert(0, str(PROJECT_ROOT))

QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "unleash_code"


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    name: str
    metric: str
    value: float
    unit: str
    target: float | None = None
    passed: bool | None = None

    def __post_init__(self):
        if self.target is not None:
            self.passed = self.value <= self.target if "latency" in self.metric.lower() else self.value >= self.target


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    results: list[BenchmarkResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    def add(self, name: str, metric: str, value: float, unit: str, target: float | None = None):
        self.results.append(BenchmarkResult(name, metric, value, unit, target))

    def print_report(self):
        from tabulate import tabulate

        print("\n" + "=" * 70)
        print("CODE INTELLIGENCE BENCHMARK REPORT")
        print("=" * 70)

        rows = []
        for r in self.results:
            status = ""
            if r.passed is True:
                status = "PASS"
            elif r.passed is False:
                status = "FAIL"

            target_str = f"{r.target:.2f}" if r.target else "-"
            rows.append([r.name, r.metric, f"{r.value:.2f}", r.unit, target_str, status])

        print(tabulate(
            rows,
            headers=["Benchmark", "Metric", "Value", "Unit", "Target", "Status"],
            tablefmt="grid",
        ))

        print(f"\nTotal benchmark time: {self.elapsed:.1f}s")

        # Summary
        passed = sum(1 for r in self.results if r.passed is True)
        failed = sum(1 for r in self.results if r.passed is False)
        if failed > 0:
            print(f"\nRESULT: {failed} benchmark(s) below target")
        else:
            print(f"\nRESULT: All {passed} benchmarks passed")


def benchmark_embedding_throughput(num_samples: int = 100) -> BenchmarkResult:
    """Benchmark embedding generation throughput."""
    import voyageai  # type: ignore[import-untyped]

    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        return BenchmarkResult("Embedding", "Throughput", 0, "chunks/s", 10, False)

    voyage = voyageai.Client(api_key=api_key)  # type: ignore[attr-defined]

    # Generate test code samples
    samples = [f"def function_{i}(x): return x + {i}" for i in range(num_samples)]

    # Benchmark in batches of 32
    batch_size = 32
    start = time.time()
    total_embedded = 0

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        voyage.embed(batch, model="voyage-code-3", input_type="document")
        total_embedded += len(batch)

    elapsed = time.time() - start
    throughput = total_embedded / elapsed

    # Target: 10 chunks/second minimum
    return BenchmarkResult("Embedding", "Throughput", throughput, "chunks/s", 10.0)


def benchmark_search_latency(num_queries: int = 20) -> BenchmarkResult:
    """Benchmark semantic search latency."""
    import voyageai  # type: ignore[import-untyped]
    from qdrant_client import QdrantClient

    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        return BenchmarkResult("Search", "Latency", 9999, "ms", 100, False)

    voyage = voyageai.Client(api_key=api_key)  # type: ignore[attr-defined]
    qdrant = QdrantClient(QDRANT_URL)

    # Check if collection has data
    info = qdrant.get_collection(QDRANT_COLLECTION)
    if info.points_count == 0:
        return BenchmarkResult("Search", "Latency", 0, "ms", 100, None)

    queries = [
        "authentication login user",
        "database query sql",
        "api request http",
        "error handling exception",
        "file read write",
    ]

    latencies = []

    for query in queries[:num_queries]:
        # Time the full search (embed + vector search)
        start = time.time()

        # Embed query
        result = voyage.embed([query], model="voyage-code-3", input_type="query")
        query_vector = result.embeddings[0]

        # Search
        qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=5,
        )

        elapsed_ms = (time.time() - start) * 1000
        latencies.append(elapsed_ms)

    avg_latency = sum(latencies) / len(latencies)

    # Target: <100ms average latency
    return BenchmarkResult("Search", "Latency (avg)", avg_latency, "ms", 100.0)


def benchmark_qdrant_query_latency() -> BenchmarkResult:
    """Benchmark pure Qdrant query latency (without embedding)."""
    from qdrant_client import QdrantClient

    qdrant = QdrantClient(QDRANT_URL)

    # Check if collection has data
    info = qdrant.get_collection(QDRANT_COLLECTION)
    if info.points_count == 0:
        return BenchmarkResult("Qdrant", "Query Latency", 0, "ms", 15, None)

    # Random query vector
    query_vector = [0.5] * 1024

    latencies = []
    for _ in range(50):
        start = time.time()
        qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=5,
        )
        elapsed_ms = (time.time() - start) * 1000
        latencies.append(elapsed_ms)

    avg_latency = sum(latencies) / len(latencies)

    # Target: <15ms (spec says 15ms for Qdrant)
    return BenchmarkResult("Qdrant", "Query Latency (avg)", avg_latency, "ms", 15.0)


def benchmark_cold_start() -> BenchmarkResult:
    """Benchmark Qdrant client initialization time."""
    import gc
    gc.collect()

    start = time.time()

    from qdrant_client import QdrantClient
    client = QdrantClient(QDRANT_URL)
    _ = client.get_collection(QDRANT_COLLECTION)

    elapsed_ms = (time.time() - start) * 1000

    # Target: <5000ms (5 seconds)
    return BenchmarkResult("System", "Cold Start", elapsed_ms, "ms", 5000.0)


def benchmark_memory_usage() -> BenchmarkResult:
    """Benchmark memory usage of the embedding pipeline."""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024 * 1024)

    # Target: <500MB for the process
    return BenchmarkResult("System", "Memory Usage", memory_mb, "MB", 500.0)


def benchmark_collection_stats() -> list[BenchmarkResult]:
    """Get collection statistics."""
    from qdrant_client import QdrantClient

    qdrant = QdrantClient(QDRANT_URL)
    info = qdrant.get_collection(QDRANT_COLLECTION)

    return [
        BenchmarkResult("Collection", "Points Count", info.points_count or 0, "points"),
        BenchmarkResult("Collection", "Indexed Vectors", info.indexed_vectors_count or 0, "vectors"),
    ]


def run_quick_benchmarks() -> BenchmarkSuite:
    """Run quick benchmark suite."""
    suite = BenchmarkSuite()

    print("Running quick benchmarks...")

    # System benchmarks
    suite.results.append(benchmark_cold_start())
    suite.results.append(benchmark_memory_usage())

    # Collection stats
    suite.results.extend(benchmark_collection_stats())

    # Qdrant query latency
    suite.results.append(benchmark_qdrant_query_latency())

    return suite


def run_full_benchmarks() -> BenchmarkSuite:
    """Run full benchmark suite."""
    suite = BenchmarkSuite()

    print("Running full benchmarks (this may take a few minutes)...")

    # System benchmarks
    print("  [1/5] Cold start...")
    suite.results.append(benchmark_cold_start())

    print("  [2/5] Memory usage...")
    suite.results.append(benchmark_memory_usage())

    # Collection stats
    print("  [3/5] Collection stats...")
    suite.results.extend(benchmark_collection_stats())

    # Performance benchmarks
    print("  [4/5] Qdrant query latency...")
    suite.results.append(benchmark_qdrant_query_latency())

    if os.getenv("VOYAGE_API_KEY"):
        print("  [5/5] Embedding throughput...")
        suite.results.append(benchmark_embedding_throughput(num_samples=50))

        print("  [6/6] Search latency...")
        suite.results.append(benchmark_search_latency(num_queries=10))
    else:
        print("  [5/5] Skipping embedding benchmarks (VOYAGE_API_KEY not set)")

    return suite


def main():
    import argparse

    parser = argparse.ArgumentParser(description="UNLEASH Code Intelligence Benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks only")
    parser.add_argument("--full", action="store_true", help="Run full benchmark suite")

    args = parser.parse_args()

    if args.quick:
        suite = run_quick_benchmarks()
    else:
        suite = run_full_benchmarks()

    suite.print_report()

    # Exit with error if any benchmark failed
    failed = sum(1 for r in suite.results if r.passed is False)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
