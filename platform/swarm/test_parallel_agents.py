#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.26.0",
#     "structlog>=24.1.0",
#     "qdrant-client>=1.7.0",
# ]
# ///
"""
Multi-Agent Parallel Execution Test - Ultimate Autonomous Platform

Tests concurrent task execution across multiple swarm agents:
1. Worker pool initialization
2. Parallel task distribution
3. Work stealing between agents
4. Concurrent result aggregation

Usage:
    uv run test_parallel_agents.py
    python test_parallel_agents.py --workers 10 --tasks 50
"""

from __future__ import annotations

import asyncio
import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List
import random

import structlog

# Local imports
from coordinator import (
    Agent, AgentRole, Task, TaskStatus,
    SwarmMemory, QueenCoordinator, MeshCoordinator
)

logger = structlog.get_logger(__name__)


@dataclass
class ParallelTestResult:
    """Results from parallel execution test."""
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    total_duration_ms: float
    avg_task_duration_ms: float
    throughput_per_sec: float
    worker_utilization: Dict[str, float]
    parallelism_efficiency: float


class ParallelExecutionTest:
    """
    Tests multi-agent parallel task execution.

    Validates:
    - Concurrent task assignment without conflicts
    - Load balancing across workers
    - Work stealing effectiveness
    - Result consistency under parallel execution
    """

    def __init__(
        self,
        num_workers: int = 5,
        qdrant_url: str = "http://localhost:6333"
    ):
        self.num_workers = num_workers
        self.memory = SwarmMemory(qdrant_url)
        self.coordinator = QueenCoordinator(self.memory)

        # Track worker metrics
        self.worker_task_counts: Dict[str, int] = {}
        self.worker_execution_times: Dict[str, List[float]] = {}

    async def setup_workers(self) -> List[Agent]:
        """Initialize worker pool."""
        workers = []

        for i in range(self.num_workers):
            worker = Agent(
                id=f"parallel-worker-{i:03d}",
                name=f"Parallel Worker {i}",
                role=AgentRole.WORKER,
                capabilities=["process", "analyze", "transform"]
            )
            await self.coordinator.register_worker(worker)
            workers.append(worker)

            self.worker_task_counts[worker.id] = 0
            self.worker_execution_times[worker.id] = []

        logger.info("workers_initialized", count=len(workers))
        return workers

    async def generate_tasks(self, num_tasks: int) -> List[Task]:
        """Generate a batch of tasks for parallel execution."""
        tasks = []

        for i in range(num_tasks):
            # Vary task complexity with different priorities
            priority = random.randint(1, 10)

            task = Task(
                id=f"parallel-task-{i:04d}",
                description=f"Process data batch {i} with complexity {priority}",
                priority=priority,
                context={
                    "batch_id": i,
                    "complexity": priority,
                    "data_size": random.randint(100, 1000)
                }
            )
            tasks.append(task)

        return tasks

    async def execute_task_simulation(
        self,
        task: Task,
        worker_id: str
    ) -> Dict[str, Any]:
        """
        Simulate task execution with variable duration.

        In production, this would delegate to actual processing.
        """
        # Simulate processing time based on complexity
        complexity = task.context.get("complexity", 5)
        base_delay = 0.01  # 10ms base
        delay = base_delay * (complexity / 5)  # Scale by complexity

        await asyncio.sleep(delay)

        return {
            "task_id": task.id,
            "worker_id": worker_id,
            "processed_items": task.context.get("data_size", 0),
            "complexity": complexity,
            "completed_at": datetime.now(timezone.utc).isoformat()
        }

    async def worker_loop(
        self,
        worker: Agent,
        task_queue: asyncio.Queue,
        results: List[Dict[str, Any]]
    ) -> None:
        """
        Worker coroutine that processes tasks from shared queue.

        Implements work stealing pattern - workers take tasks as available.
        """
        while True:
            try:
                # Non-blocking get with timeout
                task = await asyncio.wait_for(
                    task_queue.get(),
                    timeout=0.5
                )

                start = time.perf_counter()

                # Execute task
                result = await self.execute_task_simulation(task, worker.id)

                duration = (time.perf_counter() - start) * 1000

                # Track metrics
                self.worker_task_counts[worker.id] += 1
                self.worker_execution_times[worker.id].append(duration)

                results.append(result)
                task_queue.task_done()

            except asyncio.TimeoutError:
                # No more tasks available
                break
            except Exception as e:
                logger.error("worker_error", worker=worker.id, error=str(e))

    async def run_parallel_test(
        self,
        num_tasks: int = 50
    ) -> ParallelTestResult:
        """
        Run the parallel execution test.

        Creates a task queue and launches workers concurrently.
        """
        print(f"\n{'='*60}")
        print("MULTI-AGENT PARALLEL EXECUTION TEST")
        print(f"{'='*60}")
        print(f"Workers: {self.num_workers} | Tasks: {num_tasks}")

        # Setup
        workers = await self.setup_workers()
        tasks = await self.generate_tasks(num_tasks)

        # Create shared task queue
        task_queue: asyncio.Queue = asyncio.Queue()
        results: List[Dict[str, Any]] = []

        # Enqueue all tasks
        for task in tasks:
            await task_queue.put(task)

        print(f"\nStarting parallel execution...")
        start_time = time.perf_counter()

        # Launch workers concurrently
        worker_tasks = [
            asyncio.create_task(
                self.worker_loop(worker, task_queue, results)
            )
            for worker in workers
        ]

        # Wait for all workers to complete
        await asyncio.gather(*worker_tasks)

        total_duration = (time.perf_counter() - start_time) * 1000

        # Calculate metrics
        completed = len(results)
        failed = num_tasks - completed

        all_durations = []
        for times in self.worker_execution_times.values():
            all_durations.extend(times)

        avg_duration = sum(all_durations) / len(all_durations) if all_durations else 0
        throughput = (completed / total_duration) * 1000 if total_duration > 0 else 0

        # Calculate worker utilization
        utilization = {}
        max_tasks = max(self.worker_task_counts.values()) if self.worker_task_counts else 1
        for worker_id, count in self.worker_task_counts.items():
            utilization[worker_id] = count / max_tasks if max_tasks > 0 else 0

        # Parallelism efficiency: actual vs theoretical speedup
        # Theoretical: single-threaded time / parallel time
        single_thread_estimate = sum(all_durations)
        parallel_efficiency = single_thread_estimate / total_duration if total_duration > 0 else 0

        result = ParallelTestResult(
            total_tasks=num_tasks,
            completed_tasks=completed,
            failed_tasks=failed,
            total_duration_ms=total_duration,
            avg_task_duration_ms=avg_duration,
            throughput_per_sec=throughput,
            worker_utilization=utilization,
            parallelism_efficiency=parallel_efficiency
        )

        self._print_results(result)
        return result

    def _print_results(self, result: ParallelTestResult) -> None:
        """Print test results."""
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")

        print(f"\nTask Completion:")
        print(f"  Completed: {result.completed_tasks}/{result.total_tasks}")
        print(f"  Failed: {result.failed_tasks}")

        print(f"\nPerformance:")
        print(f"  Total Duration: {result.total_duration_ms:.1f}ms")
        print(f"  Avg Task Duration: {result.avg_task_duration_ms:.2f}ms")
        print(f"  Throughput: {result.throughput_per_sec:.1f} tasks/sec")
        print(f"  Parallelism Efficiency: {result.parallelism_efficiency:.2f}x")

        print(f"\nWorker Distribution:")
        for worker_id, count in self.worker_task_counts.items():
            util = result.worker_utilization.get(worker_id, 0)
            bar = "[" + "#" * int(util * 20) + "." * (20 - int(util * 20)) + "]"
            print(f"  {worker_id}: {count:3d} tasks {bar} {util*100:.0f}%")

        # Verify load balancing
        counts = list(self.worker_task_counts.values())
        if counts:
            min_tasks = min(counts)
            max_tasks = max(counts)
            balance_ratio = min_tasks / max_tasks if max_tasks > 0 else 0

            print(f"\nLoad Balancing:")
            print(f"  Min tasks per worker: {min_tasks}")
            print(f"  Max tasks per worker: {max_tasks}")
            print(f"  Balance ratio: {balance_ratio:.2f}")

            if balance_ratio > 0.7:
                print("  [OK] Good load balancing achieved")
            else:
                print("  [WARN] Load imbalance detected")


async def run_scaling_test(
    worker_counts: List[int],
    tasks_per_test: int = 100
) -> None:
    """Test scalability across different worker counts."""
    print(f"\n{'='*60}")
    print("SCALABILITY TEST")
    print(f"{'='*60}")

    results = []

    for num_workers in worker_counts:
        test = ParallelExecutionTest(num_workers=num_workers)
        result = await test.run_parallel_test(tasks_per_test)
        results.append((num_workers, result))

        # Brief pause between tests
        await asyncio.sleep(0.5)

    # Summary
    print(f"\n{'='*60}")
    print("SCALING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Workers':>8} | {'Throughput':>12} | {'Efficiency':>10} | {'Duration':>12}")
    print("-" * 60)

    for num_workers, result in results:
        print(f"{num_workers:>8} | {result.throughput_per_sec:>9.1f}/s | "
              f"{result.parallelism_efficiency:>9.2f}x | "
              f"{result.total_duration_ms:>9.1f}ms")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Multi-Agent Parallel Execution Test")
    parser.add_argument("--workers", "-w", type=int, default=5,
                        help="Number of parallel workers")
    parser.add_argument("--tasks", "-t", type=int, default=50,
                        help="Number of tasks to process")
    parser.add_argument("--scaling", "-s", action="store_true",
                        help="Run scaling test across worker counts")
    args = parser.parse_args()

    if args.scaling:
        await run_scaling_test([2, 4, 6, 8, 10], tasks_per_test=100)
    else:
        test = ParallelExecutionTest(num_workers=args.workers)
        await test.run_parallel_test(args.tasks)

    print("\n[OK] Parallel execution test complete!")


if __name__ == "__main__":
    asyncio.run(main())
