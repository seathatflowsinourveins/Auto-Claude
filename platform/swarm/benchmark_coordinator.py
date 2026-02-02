#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "qdrant-client>=1.7.0",
#     "structlog>=24.1.0",
# ]
# ///
"""
Swarm Coordinator Performance Benchmark - Ultimate Autonomous Platform

Benchmarks the swarm coordinator under various conditions:
- Topology comparison (Hierarchical vs Mesh)
- Task throughput
- Worker scaling
- Memory operation latency

Results help inform agent selection decisions (AGENT_SELECTION_GUIDE.md).
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Any

sys.path.insert(0, str(__file__).rsplit("\\", 1)[0])

from coordinator import (
    Agent, AgentRole, Task, Topology,
    SwarmMemory, QueenCoordinator, MeshCoordinator, SwarmOrchestrator
)


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    name: str
    topology: str
    workers: int
    tasks: int
    total_time_ms: float
    avg_task_time_ms: float
    tasks_per_second: float
    memory_ops: int
    memory_latency_ms: float


class CoordinatorBenchmark:
    """Benchmark suite for swarm coordinator."""

    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        self.qdrant_url = qdrant_url
        self.results: List[BenchmarkResult] = []

    async def benchmark_hierarchical(
        self,
        num_workers: int = 5,
        num_tasks: int = 50
    ) -> BenchmarkResult:
        """Benchmark hierarchical (Queen) topology."""
        memory = SwarmMemory(self.qdrant_url)
        queen = QueenCoordinator(memory)

        # Register workers
        workers = []
        for i in range(num_workers):
            worker = Agent(
                id=f"bench-hier-worker-{i:03d}",
                name=f"Hierarchical Worker {i}",
                role=AgentRole.WORKER,
                capabilities=["benchmark", "test"]
            )
            await queen.register_worker(worker)
            workers.append(worker)

        # Create tasks
        tasks = []
        for i in range(num_tasks):
            task = Task(
                id=f"bench-hier-task-{i:04d}",
                description=f"Benchmark task {i}",
                priority=5 + (i % 5)
            )
            tasks.append(task)

        # Benchmark task submission and assignment
        start_time = time.perf_counter()
        memory_ops = 0

        for task in tasks:
            await queen.submit_task(task)
            memory_ops += 1

            assigned = await queen.assign_task(task.id)
            memory_ops += 1

            if assigned:
                await queen.complete_task(task.id, {"status": "benchmark"})
                memory_ops += 1

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000

        result = BenchmarkResult(
            name="hierarchical_throughput",
            topology="hierarchical",
            workers=num_workers,
            tasks=num_tasks,
            total_time_ms=total_time_ms,
            avg_task_time_ms=total_time_ms / num_tasks,
            tasks_per_second=num_tasks / (total_time_ms / 1000),
            memory_ops=memory_ops,
            memory_latency_ms=total_time_ms / memory_ops
        )

        self.results.append(result)
        return result

    async def benchmark_mesh(
        self,
        num_workers: int = 5,
        num_tasks: int = 50
    ) -> BenchmarkResult:
        """Benchmark mesh (peer-to-peer) topology using gossip protocol."""
        memory = SwarmMemory(self.qdrant_url)
        mesh = MeshCoordinator(f"mesh-benchmark-{num_workers}w", memory)

        # Add local tasks to the mesh node
        tasks = []
        for i in range(num_tasks):
            task = Task(
                id=f"bench-mesh-task-{i:04d}",
                description=f"Mesh benchmark task {i}",
                priority=5 + (i % 5)
            )
            tasks.append(task)
            mesh.local_tasks.append(task)

        # Simulate peer discovery and gossip
        start_time = time.perf_counter()
        memory_ops = 0

        # Gossip state for each task
        for task in tasks:
            # Each gossip broadcasts state
            await mesh.gossip_state()
            memory_ops += 1

            # Discover peers (simulates mesh coordination)
            await mesh.discover_peers()
            memory_ops += 1

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000

        result = BenchmarkResult(
            name="mesh_throughput",
            topology="mesh",
            workers=num_workers,
            tasks=num_tasks,
            total_time_ms=total_time_ms,
            avg_task_time_ms=total_time_ms / num_tasks,
            tasks_per_second=num_tasks / (total_time_ms / 1000),
            memory_ops=memory_ops,
            memory_latency_ms=total_time_ms / memory_ops
        )

        self.results.append(result)
        return result

    async def benchmark_memory_latency(
        self,
        num_operations: int = 100
    ) -> Dict[str, float]:
        """Benchmark raw memory operation latency."""
        memory = SwarmMemory(self.qdrant_url)

        # Store operations
        store_times = []
        for i in range(num_operations):
            start = time.perf_counter()
            await memory.store(
                f"bench/latency/{i}",
                {"data": f"test-{i}", "index": i}
            )
            store_times.append((time.perf_counter() - start) * 1000)

        return {
            "store_min_ms": min(store_times),
            "store_max_ms": max(store_times),
            "store_avg_ms": statistics.mean(store_times),
            "store_p50_ms": statistics.median(store_times),
            "store_p95_ms": sorted(store_times)[int(num_operations * 0.95)],
        }

    async def benchmark_scaling(
        self,
        worker_counts: List[int] = [2, 5, 10, 20],
        tasks_per_worker: int = 10
    ) -> List[BenchmarkResult]:
        """Benchmark how throughput scales with worker count."""
        scaling_results = []

        for num_workers in worker_counts:
            num_tasks = num_workers * tasks_per_worker

            # Test hierarchical
            hier_result = await self.benchmark_hierarchical(num_workers, num_tasks)
            hier_result.name = f"hierarchical_scale_{num_workers}w"
            scaling_results.append(hier_result)

            # Test mesh
            mesh_result = await self.benchmark_mesh(num_workers, num_tasks)
            mesh_result.name = f"mesh_scale_{num_workers}w"
            scaling_results.append(mesh_result)

        return scaling_results

    def generate_report(self) -> str:
        """Generate a benchmark report."""
        lines = []
        lines.append("=" * 60)
        lines.append("SWARM COORDINATOR BENCHMARK REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Group by topology
        hier_results = [r for r in self.results if r.topology == "hierarchical"]
        mesh_results = [r for r in self.results if r.topology == "mesh"]

        if hier_results:
            lines.append("HIERARCHICAL TOPOLOGY (Queen Coordinator)")
            lines.append("-" * 40)
            for r in hier_results:
                lines.append(f"  {r.name}")
                lines.append(f"    Workers: {r.workers}, Tasks: {r.tasks}")
                lines.append(f"    Total Time: {r.total_time_ms:.2f}ms")
                lines.append(f"    Tasks/sec: {r.tasks_per_second:.1f}")
                lines.append(f"    Avg Task Time: {r.avg_task_time_ms:.2f}ms")
                lines.append(f"    Memory Latency: {r.memory_latency_ms:.2f}ms")
                lines.append("")

        if mesh_results:
            lines.append("MESH TOPOLOGY (Gossip Protocol)")
            lines.append("-" * 40)
            for r in mesh_results:
                lines.append(f"  {r.name}")
                lines.append(f"    Peers: {r.workers}, Tasks: {r.tasks}")
                lines.append(f"    Total Time: {r.total_time_ms:.2f}ms")
                lines.append(f"    Tasks/sec: {r.tasks_per_second:.1f}")
                lines.append(f"    Avg Task Time: {r.avg_task_time_ms:.2f}ms")
                lines.append(f"    Memory Latency: {r.memory_latency_ms:.2f}ms")
                lines.append("")

        # Summary comparison
        if hier_results and mesh_results:
            lines.append("TOPOLOGY COMPARISON")
            lines.append("-" * 40)

            hier_avg = statistics.mean([r.tasks_per_second for r in hier_results])
            mesh_avg = statistics.mean([r.tasks_per_second for r in mesh_results])

            lines.append(f"  Hierarchical Avg Throughput: {hier_avg:.1f} tasks/sec")
            lines.append(f"  Mesh Avg Throughput: {mesh_avg:.1f} tasks/sec")

            if hier_avg > mesh_avg:
                diff = ((hier_avg / mesh_avg) - 1) * 100
                lines.append(f"  Winner: Hierarchical (+{diff:.1f}% faster)")
            else:
                diff = ((mesh_avg / hier_avg) - 1) * 100
                lines.append(f"  Winner: Mesh (+{diff:.1f}% faster)")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


async def main():
    """Run the benchmark suite."""
    print("=" * 60)
    print("SWARM COORDINATOR BENCHMARK")
    print("=" * 60)

    benchmark = CoordinatorBenchmark()

    # Memory latency test
    print("\n[1/3] Testing memory latency...")
    latency = await benchmark.benchmark_memory_latency(50)
    print(f"  Store latency: {latency['store_avg_ms']:.2f}ms avg, "
          f"{latency['store_p95_ms']:.2f}ms p95")

    # Basic throughput tests
    print("\n[2/3] Testing basic throughput...")
    hier = await benchmark.benchmark_hierarchical(5, 25)
    print(f"  Hierarchical: {hier.tasks_per_second:.1f} tasks/sec")

    mesh = await benchmark.benchmark_mesh(5, 25)
    print(f"  Mesh: {mesh.tasks_per_second:.1f} tasks/sec")

    # Scaling tests
    print("\n[3/3] Testing scaling (2, 5, 10 workers)...")
    scaling = await benchmark.benchmark_scaling([2, 5, 10], 5)
    for r in scaling:
        print(f"  {r.name}: {r.tasks_per_second:.1f} tasks/sec")

    # Generate report
    print("\n")
    print(benchmark.generate_report())

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
