"""
UAP Core Benchmarks - Performance measurement for all modules.

Measures:
- Execution time (mean, median, p95, p99)
- Memory usage (peak, average)
- Throughput (operations/second)
- Latency distribution
"""

import time
import gc
import statistics
import tracemalloc
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
from enum import Enum
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    iterations: int
    total_time_ms: float
    mean_time_ms: float
    median_time_ms: float
    min_time_ms: float
    max_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    std_dev_ms: float
    ops_per_second: float
    peak_memory_mb: float
    avg_memory_mb: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time_ms": round(self.total_time_ms, 3),
            "mean_time_ms": round(self.mean_time_ms, 3),
            "median_time_ms": round(self.median_time_ms, 3),
            "min_time_ms": round(self.min_time_ms, 3),
            "max_time_ms": round(self.max_time_ms, 3),
            "p95_time_ms": round(self.p95_time_ms, 3),
            "p99_time_ms": round(self.p99_time_ms, 3),
            "std_dev_ms": round(self.std_dev_ms, 3),
            "ops_per_second": round(self.ops_per_second, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 3),
            "avg_memory_mb": round(self.avg_memory_mb, 3),
        }

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Iterations: {self.iterations}\n"
            f"  Mean: {self.mean_time_ms:.3f}ms | Median: {self.median_time_ms:.3f}ms\n"
            f"  Min: {self.min_time_ms:.3f}ms | Max: {self.max_time_ms:.3f}ms\n"
            f"  P95: {self.p95_time_ms:.3f}ms | P99: {self.p99_time_ms:.3f}ms\n"
            f"  Throughput: {self.ops_per_second:.2f} ops/sec\n"
            f"  Memory: Peak {self.peak_memory_mb:.3f}MB | Avg {self.avg_memory_mb:.3f}MB"
        )


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile of data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


# =============================================================================
# Benchmark Suite
# =============================================================================

class BenchmarkSuite:
    """Suite for running benchmarks with memory tracking."""

    def __init__(self, warmup_iterations: int = 3):
        self.warmup_iterations = warmup_iterations
        self.results: List[BenchmarkResult] = []

    def run_benchmark(
        self,
        name: str,
        func: Callable,
        iterations: int = 100,
        setup: Optional[Callable] = None,
        teardown: Optional[Callable] = None,
    ) -> BenchmarkResult:
        """
        Run a benchmark with timing and memory tracking.

        Args:
            name: Benchmark name
            func: Function to benchmark (called with no args)
            iterations: Number of iterations
            setup: Optional setup function called before each iteration
            teardown: Optional teardown function called after each iteration
        """
        # Warmup
        for _ in range(self.warmup_iterations):
            if setup:
                setup()
            func()
            if teardown:
                teardown()

        # Force garbage collection before benchmark
        gc.collect()

        # Track times and memory
        times_ms: List[float] = []
        memory_samples: List[float] = []

        # Start memory tracking
        tracemalloc.start()

        total_start = time.perf_counter()

        for _ in range(iterations):
            if setup:
                setup()

            # Time the function
            start = time.perf_counter()
            func()
            end = time.perf_counter()

            times_ms.append((end - start) * 1000)

            # Sample memory
            current, peak = tracemalloc.get_traced_memory()
            memory_samples.append(current / (1024 * 1024))  # Convert to MB

            if teardown:
                teardown()

        total_end = time.perf_counter()

        # Get peak memory
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Calculate statistics
        total_time_ms = (total_end - total_start) * 1000
        mean_time = statistics.mean(times_ms)
        median_time = statistics.median(times_ms)
        std_dev = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0

        result = BenchmarkResult(
            name=name,
            iterations=iterations,
            total_time_ms=total_time_ms,
            mean_time_ms=mean_time,
            median_time_ms=median_time,
            min_time_ms=min(times_ms),
            max_time_ms=max(times_ms),
            p95_time_ms=percentile(times_ms, 95),
            p99_time_ms=percentile(times_ms, 99),
            std_dev_ms=std_dev,
            ops_per_second=iterations / (total_time_ms / 1000),
            peak_memory_mb=peak_memory / (1024 * 1024),
            avg_memory_mb=statistics.mean(memory_samples) if memory_samples else 0.0,
        )

        self.results.append(result)
        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmark results."""
        return {
            "total_benchmarks": len(self.results),
            "results": [r.to_dict() for r in self.results],
        }


# =============================================================================
# Module-Specific Benchmarks
# =============================================================================

class MemoryBenchmark:
    """Benchmarks for memory system modules."""

    @staticmethod
    def run(suite: BenchmarkSuite, iterations: int = 100) -> List[BenchmarkResult]:
        """Run all memory benchmarks."""
        results = []

        from core.memory import MemorySystem
        from core.advanced_memory import (
            AdvancedMemorySystem,
            SemanticIndex,
            LocalEmbeddingProvider,
        )
        import asyncio

        # Benchmark: MemorySystem creation
        results.append(suite.run_benchmark(
            name="MemorySystem.create",
            func=lambda: MemorySystem(agent_id="bench-agent"),
            iterations=iterations,
        ))

        # Benchmark: Core memory update
        memory = MemorySystem(agent_id="bench-agent")
        results.append(suite.run_benchmark(
            name="CoreMemory.update",
            func=lambda: memory.core.update("test_key", "test_value"),
            iterations=iterations,
        ))

        # Benchmark: Core memory get
        memory.core.update("bench_key", "bench_value")
        results.append(suite.run_benchmark(
            name="CoreMemory.get",
            func=lambda: memory.core.get("bench_key"),
            iterations=iterations,
        ))

        # Benchmark: SemanticIndex add (uses async, so we wrap it)
        embedding_provider = LocalEmbeddingProvider(dimensions=384)
        index = SemanticIndex(embedding_provider=embedding_provider)
        counter = [0]

        def add_entry():
            counter[0] += 1
            asyncio.run(
                index.add(
                    f"entry-{counter[0]}",
                    f"Content for entry {counter[0]}",
                    {"type": "test"},
                )
            )

        results.append(suite.run_benchmark(
            name="SemanticIndex.add",
            func=add_entry,
            iterations=iterations,
        ))

        # Benchmark: AdvancedMemorySystem creation
        results.append(suite.run_benchmark(
            name="AdvancedMemorySystem.create",
            func=lambda: AdvancedMemorySystem(agent_id="bench-agent"),
            iterations=iterations // 2,  # Slower operation
        ))

        return results


class ThinkingBenchmark:
    """Benchmarks for thinking/reasoning modules."""

    @staticmethod
    def run(suite: BenchmarkSuite, iterations: int = 100) -> List[BenchmarkResult]:
        """Run all thinking benchmarks."""
        results = []

        from core.thinking import ThinkingEngine
        from core.ultrathink import (
            UltrathinkEngine,
            detect_thinking_level,
            ThinkingLevel,
            CoTPhase,
            TreeOfThoughts,
            ConfidenceCalibrator,
        )

        # Benchmark: ThinkingEngine creation
        results.append(suite.run_benchmark(
            name="ThinkingEngine.create",
            func=lambda: ThinkingEngine(),
            iterations=iterations,
        ))

        # Benchmark: Thinking level detection
        prompts = [
            "Quick question about Python",
            "Let me think about this architecture",
            "I need to hardthink about this complex problem",
            "Let me ultrathink through all possibilities",
        ]
        prompt_idx = [0]
        def detect_level():
            prompt_idx[0] = (prompt_idx[0] + 1) % len(prompts)
            return detect_thinking_level(prompts[prompt_idx[0]])

        results.append(suite.run_benchmark(
            name="detect_thinking_level",
            func=detect_level,
            iterations=iterations,
        ))

        # Benchmark: UltrathinkEngine creation
        results.append(suite.run_benchmark(
            name="UltrathinkEngine.create",
            func=lambda: UltrathinkEngine(),
            iterations=iterations,
        ))

        # Benchmark: Chain creation and steps
        engine = UltrathinkEngine()
        def create_and_step():
            chain = engine.begin_chain("Test analysis", level=ThinkingLevel.THINK)
            engine.add_step(chain.id, CoTPhase.UNDERSTAND, "Understanding the problem")
            engine.conclude_chain(chain.id, "Conclusion reached")

        results.append(suite.run_benchmark(
            name="UltrathinkEngine.chain_lifecycle",
            func=create_and_step,
            iterations=iterations,
        ))

        # Benchmark: TreeOfThoughts exploration
        tot = TreeOfThoughts(max_depth=3, max_branches=2)
        def explore_tree():
            root = tot.create_root("Should we use approach A or B?")
            tot.add_branch(root.id, "Approach A with modifications", 0.7)
            tot.add_branch(root.id, "Approach B with extensions", 0.8)
            tot.prune()  # No threshold parameter - uses pruning_threshold from init
            return tot.get_promising_branches()

        results.append(suite.run_benchmark(
            name="TreeOfThoughts.explore",
            func=explore_tree,
            iterations=iterations,
        ))

        # Benchmark: Confidence calibration
        from core.ultrathink import EvidenceItem
        calibrator = ConfidenceCalibrator(prior_confidence=0.5)
        test_evidence = [
            EvidenceItem(
                content="Test evidence",
                source="benchmark",
                strength=0.8,
                relevance=0.9,
                contradicts=False,
            )
        ]

        def calibrate():
            return calibrator.calibrate(
                base_confidence=0.6,
                evidence=test_evidence,
                chain_coherence=0.7,
            )

        results.append(suite.run_benchmark(
            name="ConfidenceCalibrator.calibrate",
            func=calibrate,
            iterations=iterations,
        ))

        return results


class OrchestrationBenchmark:
    """Benchmarks for orchestration modules."""

    @staticmethod
    def run(suite: BenchmarkSuite, iterations: int = 100) -> List[BenchmarkResult]:
        """Run all orchestration benchmarks."""
        results = []

        from core.orchestrator import (
            Orchestrator,
            Topology,
            AgentRole,
            AgentCapability,
            TaskPriority,
        )
        from core.skills import SkillRegistry, create_skill_registry
        from core.tool_registry import ToolRegistry, create_tool_registry

        # Benchmark: Orchestrator creation
        results.append(suite.run_benchmark(
            name="Orchestrator.create",
            func=lambda: Orchestrator(topology=Topology.HIERARCHICAL),
            iterations=iterations,
        ))

        # Benchmark: Agent registration
        orch = Orchestrator(topology=Topology.MESH)
        counter = [0]
        def register_agent():
            counter[0] += 1
            orch.register_agent(
                agent_id=f"agent-{counter[0]}",
                role=AgentRole.SPECIALIST,
                capabilities=[
                    AgentCapability(name="testing", proficiency=0.8),
                ],
            )

        results.append(suite.run_benchmark(
            name="Orchestrator.register_agent",
            func=register_agent,
            iterations=iterations,
        ))

        # Benchmark: SkillRegistry with builtins
        results.append(suite.run_benchmark(
            name="SkillRegistry.create_with_builtins",
            func=lambda: create_skill_registry(include_builtins=True),
            iterations=iterations,
        ))

        # Benchmark: Skill search
        registry = create_skill_registry(include_builtins=True)
        queries = ["code review", "testing", "thinking", "debugging"]
        query_idx = [0]
        def search_skills():
            query_idx[0] = (query_idx[0] + 1) % len(queries)
            return registry.find_relevant(queries[query_idx[0]], max_skills=3)

        results.append(suite.run_benchmark(
            name="SkillRegistry.find_relevant",
            func=search_skills,
            iterations=iterations,
        ))

        # Benchmark: ToolRegistry with builtins
        results.append(suite.run_benchmark(
            name="ToolRegistry.create_with_builtins",
            func=lambda: create_tool_registry(include_builtins=True),
            iterations=iterations,
        ))

        # Benchmark: Tool search
        tools = create_tool_registry(include_builtins=True)
        def search_tools():
            return tools.search("read")

        results.append(suite.run_benchmark(
            name="ToolRegistry.search",
            func=search_tools,
            iterations=iterations,
        ))

        return results


class ResilienceBenchmark:
    """Benchmarks for resilience modules."""

    @staticmethod
    def run(suite: BenchmarkSuite, iterations: int = 100) -> List[BenchmarkResult]:
        """Run all resilience benchmarks."""
        results = []

        from core.resilience import (
            CircuitBreaker,
            RetryPolicy,
            RetryStrategy,
            RateLimiter,
            RateLimitStrategy,
            BackpressureManager,
            HealthChecker,
            TelemetryCollector,
            ResilienceHandler,
        )

        # Benchmark: CircuitBreaker creation
        results.append(suite.run_benchmark(
            name="CircuitBreaker.create",
            func=lambda: CircuitBreaker(failure_threshold=5, recovery_timeout=30.0),
            iterations=iterations,
        ))

        # Benchmark: CircuitBreaker state and stats access (sync operations)
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

        def check_circuit():
            _ = cb.state
            _ = cb.stats
            return cb._should_attempt_reset()

        results.append(suite.run_benchmark(
            name="CircuitBreaker.state_check",
            func=check_circuit,
            iterations=iterations,
        ))

        # Benchmark: RetryPolicy creation
        results.append(suite.run_benchmark(
            name="RetryPolicy.create",
            func=lambda: RetryPolicy(
                max_retries=3,
                strategy=RetryStrategy.EXPONENTIAL,
                base_delay=1.0,
            ),
            iterations=iterations,
        ))

        # Benchmark: RetryPolicy delay calculation
        policy = RetryPolicy(max_retries=5, strategy=RetryStrategy.DECORRELATED_JITTER)
        attempt_num = [0]

        def calc_delay():
            attempt_num[0] = (attempt_num[0] % 5) + 1
            return policy.calculate_delay(attempt_num[0])

        results.append(suite.run_benchmark(
            name="RetryPolicy.calculate_delay",
            func=calc_delay,
            iterations=iterations,
        ))

        # Benchmark: RateLimiter creation
        results.append(suite.run_benchmark(
            name="RateLimiter.create",
            func=lambda: RateLimiter(
                tokens_per_second=100.0,
                bucket_size=100,
            ),
            iterations=iterations,
        ))

        # Benchmark: RateLimiter internal refill (sync operation)
        limiter = RateLimiter(tokens_per_second=10000.0, bucket_size=10000)

        def limiter_check():
            limiter._refill()  # Sync method
            return limiter._tokens

        results.append(suite.run_benchmark(
            name="RateLimiter.refill",
            func=limiter_check,
            iterations=iterations,
        ))

        # Benchmark: BackpressureManager
        bp = BackpressureManager()
        results.append(suite.run_benchmark(
            name="BackpressureManager.should_accept",
            func=lambda: bp.should_accept_request(priority=1),
            iterations=iterations,
        ))

        # Benchmark: HealthChecker
        health = HealthChecker()
        health.register_check("test", lambda: True)
        results.append(suite.run_benchmark(
            name="HealthChecker.check_all",
            func=lambda: health.check_all(),
            iterations=iterations,
        ))

        # Benchmark: TelemetryCollector
        telemetry = TelemetryCollector()
        counter = [0]
        def record_metric():
            counter[0] += 1
            telemetry.record_counter(f"test.counter", 1, {"iteration": str(counter[0])})

        results.append(suite.run_benchmark(
            name="TelemetryCollector.record_counter",
            func=record_metric,
            iterations=iterations,
        ))

        # Benchmark: ResilienceHandler creation
        results.append(suite.run_benchmark(
            name="ResilienceHandler.create",
            func=lambda: ResilienceHandler(),
            iterations=iterations // 2,  # Slower operation
        ))

        return results


# =============================================================================
# Run All Benchmarks
# =============================================================================

def run_all_benchmarks(iterations: int = 100) -> Dict[str, Any]:
    """
    Run all benchmarks and return comprehensive results.

    Args:
        iterations: Number of iterations per benchmark

    Returns:
        Dictionary with all benchmark results
    """
    suite = BenchmarkSuite(warmup_iterations=3)

    print("=" * 60)
    print("UAP Core Benchmarks")
    print("=" * 60)

    # Run memory benchmarks
    print("\n[Memory Benchmarks]")
    memory_results = MemoryBenchmark.run(suite, iterations)
    for r in memory_results:
        print(f"  {r.name}: {r.mean_time_ms:.3f}ms (p95: {r.p95_time_ms:.3f}ms)")

    # Run thinking benchmarks
    print("\n[Thinking Benchmarks]")
    thinking_results = ThinkingBenchmark.run(suite, iterations)
    for r in thinking_results:
        print(f"  {r.name}: {r.mean_time_ms:.3f}ms (p95: {r.p95_time_ms:.3f}ms)")

    # Run orchestration benchmarks
    print("\n[Orchestration Benchmarks]")
    orch_results = OrchestrationBenchmark.run(suite, iterations)
    for r in orch_results:
        print(f"  {r.name}: {r.mean_time_ms:.3f}ms (p95: {r.p95_time_ms:.3f}ms)")

    # Run resilience benchmarks
    print("\n[Resilience Benchmarks]")
    resilience_results = ResilienceBenchmark.run(suite, iterations)
    for r in resilience_results:
        print(f"  {r.name}: {r.mean_time_ms:.3f}ms (p95: {r.p95_time_ms:.3f}ms)")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_results = suite.results

    # Find slowest operations
    slowest = sorted(all_results, key=lambda r: r.mean_time_ms, reverse=True)[:5]
    print("\n[Slowest Operations]")
    for r in slowest:
        print(f"  {r.name}: {r.mean_time_ms:.3f}ms")

    # Find highest memory usage
    highest_memory = sorted(all_results, key=lambda r: r.peak_memory_mb, reverse=True)[:5]
    print("\n[Highest Memory Usage]")
    for r in highest_memory:
        print(f"  {r.name}: {r.peak_memory_mb:.3f}MB peak")

    # Find highest throughput
    highest_throughput = sorted(all_results, key=lambda r: r.ops_per_second, reverse=True)[:5]
    print("\n[Highest Throughput]")
    for r in highest_throughput:
        print(f"  {r.name}: {r.ops_per_second:.2f} ops/sec")

    return suite.get_summary()


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import json

    # Run with default iterations
    results = run_all_benchmarks(iterations=100)

    # Save results to JSON
    output_path = Path(__file__).parent / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
