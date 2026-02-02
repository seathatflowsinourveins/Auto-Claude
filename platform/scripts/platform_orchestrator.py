#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.26.0",
#     "structlog>=24.1.0",
#     "qdrant-client>=1.7.0",
#     "neo4j>=5.0.0",
# ]
# ///
"""
Platform Orchestrator - Ultimate Autonomous Platform

Single entry point to initialize, monitor, and coordinate all platform components:

Components:
1. Infrastructure Layer (Qdrant, Neo4j, Letta)
2. Swarm Coordinator (Queen topology)
3. Knowledge Graph (Graphiti Bridge)
4. Integration Layer (Auto-Claude routing)
5. Memory Persistence (Vector + Graph)

Usage:
    uv run platform_orchestrator.py status        # Show component status
    uv run platform_orchestrator.py start         # Start platform
    uv run platform_orchestrator.py demo          # Run demo workflow
    uv run platform_orchestrator.py health        # Health check
    uv run platform_orchestrator.py benchmark     # Performance benchmark
"""

from __future__ import annotations

import asyncio
import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

import structlog

T = TypeVar('T')


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator for retry logic with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Multiplier for exponential backoff
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay
                        )
                        structlog.get_logger().warning(
                            "retry_attempt",
                            function=func.__name__,
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            delay=delay,
                            error=str(e)
                        )
                        await asyncio.sleep(delay)
            raise last_exception

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay
                        )
                        structlog.get_logger().warning(
                            "retry_attempt",
                            function=func.__name__,
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            delay=delay,
                            error=str(e)
                        )
                        time.sleep(delay)
            raise last_exception

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern for failing fast on repeated errors.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failing fast, requests rejected immediately
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_requests: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests

        self._failure_count = 0
        self._last_failure_time: float = 0
        self._state = "CLOSED"
        self._half_open_successes = 0

    @property
    def state(self) -> str:
        """Get current circuit state, checking for timeout-based transitions."""
        if self._state == "OPEN":
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = "HALF_OPEN"
                self._half_open_successes = 0
        return self._state

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state == "HALF_OPEN":
            self._half_open_successes += 1
            if self._half_open_successes >= self.half_open_requests:
                self._state = "CLOSED"
                self._failure_count = 0
        else:
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == "HALF_OPEN":
            self._state = "OPEN"
        elif self._failure_count >= self.failure_threshold:
            self._state = "OPEN"

    def can_execute(self) -> bool:
        """Check if a request can proceed."""
        return self.state != "OPEN"


class Bulkhead:
    """
    Bulkhead pattern for resource isolation.

    Limits concurrent requests to prevent one slow service from
    consuming all available resources (threads, connections, etc.).

    Usage:
        bulkhead = Bulkhead(max_concurrent=5)
        async with bulkhead:
            await some_operation()
    """

    def __init__(self, max_concurrent: int = 10, timeout: float = 30.0):
        """
        Initialize bulkhead.

        Args:
            max_concurrent: Maximum concurrent operations allowed
            timeout: Timeout for acquiring a slot (seconds)
        """
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._current_count = 0
        self._rejected_count = 0
        self._lock = asyncio.Lock()

    @property
    def current_load(self) -> int:
        """Current number of active operations."""
        return self._current_count

    @property
    def available_slots(self) -> int:
        """Number of available slots."""
        return self.max_concurrent - self._current_count

    @property
    def rejected_count(self) -> int:
        """Total number of rejected requests."""
        return self._rejected_count

    async def __aenter__(self):
        """Acquire a slot in the bulkhead."""
        try:
            acquired = await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.timeout
            )
            if acquired:
                async with self._lock:
                    self._current_count += 1
                return self
        except asyncio.TimeoutError:
            async with self._lock:
                self._rejected_count += 1
            raise BulkheadFullError(
                f"Bulkhead full ({self.max_concurrent} concurrent operations)"
            )

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release the slot."""
        async with self._lock:
            self._current_count -= 1
        self._semaphore.release()
        return False


class BulkheadFullError(Exception):
    """Raised when the bulkhead has no available slots."""
    pass


class GracefulShutdown:
    """
    Handles graceful shutdown for the platform.

    Features:
    - Signal handling (SIGTERM, SIGINT)
    - Timeout-based forced shutdown
    - Cleanup callbacks
    - Drain period for in-flight requests

    Usage:
        shutdown = GracefulShutdown()
        shutdown.register_cleanup(cleanup_func)

        async with shutdown.run():
            await serve()  # Runs until shutdown signal
    """

    def __init__(
        self,
        shutdown_timeout: float = 30.0,
        drain_timeout: float = 10.0
    ):
        """
        Initialize graceful shutdown handler.

        Args:
            shutdown_timeout: Maximum time to wait for cleanup (seconds)
            drain_timeout: Time to drain in-flight requests (seconds)
        """
        self.shutdown_timeout = shutdown_timeout
        self.drain_timeout = drain_timeout

        self._shutdown_event = asyncio.Event()
        self._cleanup_callbacks: List[Callable] = []
        self._is_shutting_down = False
        self._shutdown_reason: Optional[str] = None

    @property
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self._is_shutting_down

    @property
    def shutdown_reason(self) -> Optional[str]:
        """Get the reason for shutdown."""
        return self._shutdown_reason

    def register_cleanup(self, callback: Callable) -> None:
        """Register a cleanup callback to run during shutdown."""
        self._cleanup_callbacks.append(callback)

    def request_shutdown(self, reason: str = "manual") -> None:
        """Request graceful shutdown."""
        if not self._is_shutting_down:
            self._is_shutting_down = True
            self._shutdown_reason = reason
            self._shutdown_event.set()
            logger.info("shutdown_requested", reason=reason)

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()

    async def _run_cleanup(self) -> None:
        """Run all registered cleanup callbacks."""
        logger.info("running_cleanup_callbacks", count=len(self._cleanup_callbacks))

        for callback in self._cleanup_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await asyncio.wait_for(callback(), timeout=5.0)
                else:
                    callback()
            except asyncio.TimeoutError:
                logger.warning("cleanup_callback_timeout", callback=callback.__name__)
            except Exception as e:
                logger.error("cleanup_callback_error", callback=callback.__name__, error=str(e))

    async def run(self):
        """
        Context manager for running with graceful shutdown support.

        Sets up signal handlers and waits for shutdown.
        """
        import signal

        loop = asyncio.get_running_loop()

        # Setup signal handlers (Unix-style, Windows compatible)
        def signal_handler(sig):
            sig_name = signal.Signals(sig).name
            logger.info("signal_received", signal=sig_name)
            self.request_shutdown(f"signal:{sig_name}")

        # Register signal handlers
        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            # Use alternative approach
            pass

        return self

    async def __aenter__(self):
        """Enter the shutdown context."""
        await self.run()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the shutdown context - run cleanup."""
        logger.info("shutdown_initiated", timeout=self.shutdown_timeout)

        # Drain period - allow in-flight requests to complete
        logger.info("draining_requests", timeout=self.drain_timeout)
        await asyncio.sleep(min(1.0, self.drain_timeout))

        # Run cleanup callbacks with timeout
        try:
            await asyncio.wait_for(
                self._run_cleanup(),
                timeout=self.shutdown_timeout
            )
        except asyncio.TimeoutError:
            logger.warning("shutdown_timeout", forced=True)

        logger.info("shutdown_complete")
        return False


class HealthProbes:
    """
    Kubernetes-style health probes for the platform.

    Provides:
    - Liveness probe: Is the process running?
    - Readiness probe: Is the service ready to accept requests?
    - Startup probe: Has the service finished starting?
    """

    def __init__(self, orchestrator: 'PlatformOrchestrator'):
        self.orchestrator = orchestrator
        self._started = False
        self._ready = False
        self._startup_time: Optional[float] = None

    def mark_started(self) -> None:
        """Mark the service as started."""
        self._started = True
        self._startup_time = time.time()

    def mark_ready(self) -> None:
        """Mark the service as ready."""
        self._ready = True

    def mark_not_ready(self) -> None:
        """Mark the service as not ready (e.g., during maintenance)."""
        self._ready = False

    async def liveness(self) -> Dict[str, Any]:
        """
        Liveness probe - checks if process is running.

        Returns unhealthy only if the process is in a bad state
        that requires restart.
        """
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": time.time() - self._startup_time if self._startup_time else 0
        }

    async def readiness(self) -> Dict[str, Any]:
        """
        Readiness probe - checks if service can accept requests.

        Checks that critical components are available.
        """
        if not self._ready:
            return {
                "status": "not_ready",
                "reason": "service_initializing"
            }

        # Check critical components (Qdrant and Neo4j are critical)
        status = await self.orchestrator.check_health()

        critical_healthy = all(
            c.status == ComponentStatus.HEALTHY
            for c in status.components
            if c.name in ["Qdrant", "Neo4j"]
        )

        return {
            "status": "ready" if critical_healthy else "not_ready",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components_healthy": sum(
                1 for c in status.components
                if c.status == ComponentStatus.HEALTHY
            ),
            "components_total": len(status.components)
        }

    async def startup(self) -> Dict[str, Any]:
        """
        Startup probe - checks if service has finished starting.

        Used by Kubernetes to know when to start liveness checks.
        """
        return {
            "status": "started" if self._started else "starting",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Add paths for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "bridges"))
sys.path.insert(0, str(project_root / "swarm"))
sys.path.insert(0, str(project_root / "demo"))

logger = structlog.get_logger(__name__)


class ComponentStatus(Enum):
    """Status of a platform component."""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    STARTING = "starting"


@dataclass
class ComponentHealth:
    """Health information for a component."""
    name: str
    status: ComponentStatus
    message: str = ""
    latency_ms: float = 0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlatformStatus:
    """Overall platform status."""
    healthy: bool
    components: List[ComponentHealth]
    timestamp: str
    summary: str


class PlatformOrchestrator:
    """
    Central orchestrator for the Ultimate Autonomous Platform.

    Coordinates:
    - Infrastructure services (Qdrant, Neo4j, Letta)
    - Swarm coordination
    - Knowledge graph operations
    - Auto-Claude integration
    - Memory persistence

    Features:
    - Circuit breaker pattern for each component
    - Bulkhead pattern for resource isolation
    - Retry with exponential backoff for transient failures
    - Graceful degradation when components unavailable
    """

    def __init__(self):
        self.config = {
            "qdrant_url": "http://localhost:6333",
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_auth": ("neo4j", "alphaforge2024"),
            "letta_url": "http://localhost:8283",
            "auto_claude_url": "http://localhost:3000",
        }

        # Circuit breakers for each component (fail fast on repeated errors)
        self._circuit_breakers = {
            "qdrant": CircuitBreaker(failure_threshold=3, recovery_timeout=30.0),
            "neo4j": CircuitBreaker(failure_threshold=3, recovery_timeout=45.0),
            "letta": CircuitBreaker(failure_threshold=3, recovery_timeout=30.0),
            "auto_claude": CircuitBreaker(failure_threshold=5, recovery_timeout=60.0),
        }

        # Bulkheads for resource isolation (limit concurrent operations)
        self._bulkheads = {
            "health_check": Bulkhead(max_concurrent=10, timeout=30.0),
            "swarm_tasks": Bulkhead(max_concurrent=50, timeout=60.0),
            "knowledge_graph": Bulkhead(max_concurrent=20, timeout=45.0),
        }

        self._swarm = None
        self._graphiti = None
        self._integration = None

        # Health probes for Kubernetes
        self._probes = HealthProbes(self)
        self._shutdown = GracefulShutdown(shutdown_timeout=30.0, drain_timeout=5.0)

    async def check_health(self) -> PlatformStatus:
        """Check health of all platform components."""
        components = []
        timestamp = datetime.now(timezone.utc).isoformat()

        # Check Qdrant
        components.append(await self._check_qdrant())

        # Check Neo4j
        components.append(await self._check_neo4j())

        # Check Letta
        components.append(await self._check_letta())

        # Check Auto-Claude
        components.append(await self._check_auto_claude())

        # Calculate overall status
        healthy_count = sum(1 for c in components if c.status == ComponentStatus.HEALTHY)
        total = len(components)

        return PlatformStatus(
            healthy=healthy_count >= 3,  # Need at least 3 core services
            components=components,
            timestamp=timestamp,
            summary=f"{healthy_count}/{total} components healthy"
        )

    async def _check_qdrant(self) -> ComponentHealth:
        """Check Qdrant vector database with circuit breaker protection."""
        import time

        circuit = self._circuit_breakers["qdrant"]

        # Check if circuit is open (failing fast)
        if not circuit.can_execute():
            return ComponentHealth(
                name="Qdrant",
                status=ComponentStatus.UNAVAILABLE,
                message=f"Circuit OPEN (recovering in {circuit.recovery_timeout}s)",
                details={"circuit_state": circuit.state}
            )

        start = time.perf_counter()

        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.config['qdrant_url']}/collections",
                    timeout=5.0
                )
                latency = (time.perf_counter() - start) * 1000

                if resp.status_code == 200:
                    data = resp.json()
                    collections = data.get("result", {}).get("collections", [])
                    circuit.record_success()
                    return ComponentHealth(
                        name="Qdrant",
                        status=ComponentStatus.HEALTHY,
                        message=f"{len(collections)} collections",
                        latency_ms=latency,
                        details={
                            "collections": [c["name"] for c in collections],
                            "circuit_state": circuit.state
                        }
                    )

        except Exception as e:
            circuit.record_failure()
            return ComponentHealth(
                name="Qdrant",
                status=ComponentStatus.UNAVAILABLE,
                message=str(e),
                details={"circuit_state": circuit.state}
            )

        circuit.record_failure()
        return ComponentHealth(
            name="Qdrant",
            status=ComponentStatus.DEGRADED,
            message="Unexpected response",
            details={"circuit_state": circuit.state}
        )

    async def _check_neo4j(self) -> ComponentHealth:
        """Check Neo4j graph database with circuit breaker protection."""
        import time

        circuit = self._circuit_breakers["neo4j"]

        # Check if circuit is open
        if not circuit.can_execute():
            return ComponentHealth(
                name="Neo4j",
                status=ComponentStatus.UNAVAILABLE,
                message=f"Circuit OPEN (recovering in {circuit.recovery_timeout}s)",
                details={"circuit_state": circuit.state}
            )

        start = time.perf_counter()

        try:
            from neo4j import GraphDatabase

            driver = GraphDatabase.driver(
                self.config["neo4j_uri"],
                auth=self.config["neo4j_auth"]
            )

            with driver.session() as session:
                result = session.run("RETURN 1 as n")
                _ = result.single()
                latency = (time.perf_counter() - start) * 1000

                # Get node count
                count_result = session.run(
                    "MATCH (n) RETURN count(n) as count"
                )
                record = count_result.single()
                node_count = record["count"] if record else 0

            driver.close()

            circuit.record_success()
            return ComponentHealth(
                name="Neo4j",
                status=ComponentStatus.HEALTHY,
                message=f"{node_count} nodes",
                latency_ms=latency,
                details={"node_count": node_count, "circuit_state": circuit.state}
            )

        except Exception as e:
            circuit.record_failure()
            return ComponentHealth(
                name="Neo4j",
                status=ComponentStatus.UNAVAILABLE,
                message=str(e),
                details={"circuit_state": circuit.state}
            )

    async def _check_letta(self) -> ComponentHealth:
        """Check Letta memory server with circuit breaker protection."""
        import time

        circuit = self._circuit_breakers["letta"]

        # Check if circuit is open
        if not circuit.can_execute():
            return ComponentHealth(
                name="Letta",
                status=ComponentStatus.UNAVAILABLE,
                message=f"Circuit OPEN (recovering in {circuit.recovery_timeout}s)",
                details={"circuit_state": circuit.state}
            )

        start = time.perf_counter()

        try:
            import httpx
            async with httpx.AsyncClient() as client:
                # Letta serves HTML at root, check if server responds
                resp = await client.get(
                    f"{self.config['letta_url']}/",
                    timeout=5.0
                )
                latency = (time.perf_counter() - start) * 1000

                # If we get any HTML response, server is healthy
                if resp.status_code == 200 and "Letta" in resp.text:
                    circuit.record_success()
                    return ComponentHealth(
                        name="Letta",
                        status=ComponentStatus.HEALTHY,
                        message="Server responding",
                        latency_ms=latency,
                        details={"circuit_state": circuit.state}
                    )
                elif resp.status_code == 200:
                    circuit.record_success()
                    return ComponentHealth(
                        name="Letta",
                        status=ComponentStatus.HEALTHY,
                        message="Server up",
                        latency_ms=latency,
                        details={"circuit_state": circuit.state}
                    )

        except Exception as e:
            circuit.record_failure()
            return ComponentHealth(
                name="Letta",
                status=ComponentStatus.UNAVAILABLE,
                message=str(e),
                details={"circuit_state": circuit.state}
            )

        circuit.record_failure()
        return ComponentHealth(
            name="Letta",
            status=ComponentStatus.DEGRADED,
            message="Unexpected response",
            details={"circuit_state": circuit.state}
        )

    async def _check_auto_claude(self) -> ComponentHealth:
        """Check Auto-Claude IDE backend with circuit breaker protection."""
        import time

        circuit = self._circuit_breakers["auto_claude"]

        # Check if circuit is open
        if not circuit.can_execute():
            return ComponentHealth(
                name="Auto-Claude",
                status=ComponentStatus.UNAVAILABLE,
                message=f"Circuit OPEN (recovering in {circuit.recovery_timeout}s)",
                details={"circuit_state": circuit.state, "optional": True}
            )

        start = time.perf_counter()

        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.config['auto_claude_url']}/api/health",
                    timeout=5.0
                )
                latency = (time.perf_counter() - start) * 1000

                if resp.status_code == 200:
                    circuit.record_success()
                    return ComponentHealth(
                        name="Auto-Claude",
                        status=ComponentStatus.HEALTHY,
                        message="IDE backend responding",
                        latency_ms=latency,
                        details={"circuit_state": circuit.state}
                    )

        except Exception:
            # Auto-Claude is optional - don't count as circuit failure
            return ComponentHealth(
                name="Auto-Claude",
                status=ComponentStatus.UNAVAILABLE,
                message="Not running (optional)",
                details={"circuit_state": circuit.state, "optional": True}
            )

        circuit.record_failure()
        return ComponentHealth(
            name="Auto-Claude",
            status=ComponentStatus.DEGRADED,
            message="Unexpected response",
            details={"circuit_state": circuit.state}
        )

    def get_circuit_states(self) -> Dict[str, str]:
        """Get current state of all circuit breakers."""
        return {
            name: cb.state
            for name, cb in self._circuit_breakers.items()
        }

    def reset_circuit(self, component: str) -> bool:
        """Manually reset a circuit breaker to CLOSED state."""
        if component in self._circuit_breakers:
            cb = self._circuit_breakers[component]
            cb._state = "CLOSED"
            cb._failure_count = 0
            cb._half_open_successes = 0
            logger.info("circuit_reset", component=component)
            return True
        return False

    def get_bulkhead_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current status of all bulkheads."""
        return {
            name: {
                "current_load": bh.current_load,
                "available_slots": bh.available_slots,
                "max_concurrent": bh.max_concurrent,
                "rejected_count": bh.rejected_count,
                "utilization_pct": round(bh.current_load / bh.max_concurrent * 100, 1)
            }
            for name, bh in self._bulkheads.items()
        }

    async def initialize(self) -> bool:
        """Initialize all platform components."""
        print("\n[>>] Initializing Platform Components...")

        try:
            from coordinator import SwarmMemory, QueenCoordinator
            from graphiti_bridge import GraphitiBridge
            from swarm_autoclaude_integration import SwarmAutoClaudeIntegration

            # Initialize swarm
            memory = SwarmMemory(self.config["qdrant_url"])
            self._swarm = QueenCoordinator(memory)
            print("  [OK] Swarm Coordinator initialized")

            # Initialize Graphiti
            self._graphiti = GraphitiBridge(
                neo4j_uri=self.config["neo4j_uri"],
                neo4j_auth=self.config["neo4j_auth"],
                qdrant_url=self.config["qdrant_url"]
            )
            await self._graphiti.initialize()
            print("  [OK] Graphiti Bridge initialized")

            # Initialize integration
            self._integration = SwarmAutoClaudeIntegration()
            print("  [OK] Integration Layer initialized")

            return True

        except Exception as e:
            print(f"  [ERR] Initialization failed: {e}")
            return False

    async def run_demo(self, task: Optional[str] = None) -> bool:
        """Run a demo workflow through the platform."""
        if not task:
            task = "Review and optimize the swarm coordinator for production deployment"

        try:
            from e2e_workflow_demo import EndToEndWorkflow

            workflow = EndToEndWorkflow(verbose=True)
            result = await workflow.run_demo(task)

            return result.success

        except Exception as e:
            print(f"[ERR] Demo failed: {e}")
            return False

    async def run_benchmark(self) -> Dict[str, Any]:
        """Run performance benchmarks."""
        try:
            from benchmark_coordinator import CoordinatorBenchmark

            benchmark = CoordinatorBenchmark()
            results = await benchmark.run_full_benchmark()

            return results

        except Exception as e:
            print(f"[ERR] Benchmark failed: {e}")
            return {"error": str(e)}

    async def serve(self, port: int = 8080) -> None:
        """
        Run the orchestrator as a long-running service with graceful shutdown.

        Provides health check endpoints and waits for shutdown signal.
        """
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json
        import threading

        # Mark as started
        self._probes.mark_started()

        # Initialize components
        success = await self.initialize()
        if not success:
            print("[ERR] Failed to initialize - not starting server")
            return

        # Mark as ready
        self._probes.mark_ready()

        # Simple HTTP server for health probes
        orchestrator = self

        class HealthHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress default logging

            def do_GET(self):
                loop = asyncio.new_event_loop()
                try:
                    if self.path == "/health/live" or self.path == "/healthz":
                        result = loop.run_until_complete(orchestrator._probes.liveness())
                        status_code = 200
                    elif self.path == "/health/ready" or self.path == "/readyz":
                        result = loop.run_until_complete(orchestrator._probes.readiness())
                        status_code = 200 if result.get("status") == "ready" else 503
                    elif self.path == "/health/startup":
                        result = loop.run_until_complete(orchestrator._probes.startup())
                        status_code = 200 if result.get("status") == "started" else 503
                    elif self.path == "/health" or self.path == "/":
                        full_status = loop.run_until_complete(orchestrator.check_health())
                        result = {
                            "healthy": full_status.healthy,
                            "summary": full_status.summary,
                            "timestamp": full_status.timestamp,
                            "components": [
                                {
                                    "name": c.name,
                                    "status": c.status.value,
                                    "message": c.message,
                                    "latency_ms": c.latency_ms
                                }
                                for c in full_status.components
                            ]
                        }
                        status_code = 200 if full_status.healthy else 503
                    else:
                        result = {"error": "not found"}
                        status_code = 404
                finally:
                    loop.close()

                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result, indent=2).encode())

        # Start HTTP server in a thread
        server = HTTPServer(("0.0.0.0", port), HealthHandler)
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        print(f"\n[OK] Platform service running on port {port}")
        print(f"[>>] Health endpoints:")
        print(f"     GET /health/live   - Liveness probe")
        print(f"     GET /health/ready  - Readiness probe")
        print(f"     GET /health        - Full health status")
        print(f"\n[>>] Press Ctrl+C to stop...")

        # Register cleanup
        async def cleanup():
            self._probes.mark_not_ready()
            server.shutdown()
            print("[OK] HTTP server stopped")

        self._shutdown.register_cleanup(cleanup)

        # Wait for shutdown signal
        async with self._shutdown:
            await self._shutdown.wait_for_shutdown()

        print("\n[OK] Service stopped gracefully")

    async def get_probe_status(self, probe_type: str) -> Dict[str, Any]:
        """Get status from a specific health probe."""
        if probe_type == "liveness":
            return await self._probes.liveness()
        elif probe_type == "readiness":
            return await self._probes.readiness()
        elif probe_type == "startup":
            return await self._probes.startup()
        else:
            return {"error": f"Unknown probe type: {probe_type}"}


def print_status(status: PlatformStatus, show_circuits: bool = False) -> None:
    """Print platform status in a formatted way."""
    print(f"\n{'='*60}")
    print("ULTIMATE AUTONOMOUS PLATFORM STATUS")
    print(f"{'='*60}")
    print(f"Timestamp: {status.timestamp}")
    print(f"Overall: {'[HEALTHY]' if status.healthy else '[DEGRADED]'}")
    print(f"Summary: {status.summary}")

    print(f"\n{'Component':<15} {'Status':<12} {'Latency':<12} {'Message'}")
    print("-" * 60)

    status_icons = {
        ComponentStatus.HEALTHY: "[OK]",
        ComponentStatus.DEGRADED: "[WARN]",
        ComponentStatus.UNAVAILABLE: "[ERR]",
        ComponentStatus.UNKNOWN: "[?]",
        ComponentStatus.STARTING: "[>>]"
    }

    circuit_icons = {
        "CLOSED": "",
        "OPEN": " [CB:OPEN]",
        "HALF_OPEN": " [CB:TEST]"
    }

    for c in status.components:
        icon = status_icons.get(c.status, "[?]")
        latency = f"{c.latency_ms:.0f}ms" if c.latency_ms > 0 else "-"
        circuit_state = c.details.get("circuit_state", "CLOSED") if c.details else "CLOSED"
        circuit_suffix = circuit_icons.get(circuit_state, "")
        print(f"{c.name:<15} {icon:<12} {latency:<12} {c.message}{circuit_suffix}")

    # Show circuit breaker summary if requested
    if show_circuits:
        print(f"\n{'Circuit Breakers:':<60}")
        print("-" * 60)
        for c in status.components:
            if c.details and "circuit_state" in c.details:
                state = c.details["circuit_state"]
                state_icon = "[OK]" if state == "CLOSED" else "[!!]" if state == "OPEN" else "[??]"
                print(f"  {c.name:<13} {state_icon} {state}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ultimate Autonomous Platform Orchestrator"
    )
    parser.add_argument(
        "command",
        choices=["status", "start", "demo", "health", "benchmark", "reset", "serve"],
        help="Command to execute"
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        help="Task description for demo"
    )
    parser.add_argument(
        "--circuits", "-c",
        action="store_true",
        help="Show circuit breaker details"
    )
    parser.add_argument(
        "--component",
        type=str,
        help="Component name for reset command"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="Port for serve command (default: 8080)"
    )
    args = parser.parse_args()

    orchestrator = PlatformOrchestrator()

    if args.command in ["status", "health"]:
        status = await orchestrator.check_health()
        print_status(status, show_circuits=args.circuits)

    elif args.command == "start":
        success = await orchestrator.initialize()
        if success:
            print("\n[OK] Platform initialized successfully!")
            status = await orchestrator.check_health()
            print_status(status)
        else:
            print("\n[ERR] Platform initialization failed")
            sys.exit(1)

    elif args.command == "demo":
        print("\n[>>] Running E2E Demo Workflow...")
        success = await orchestrator.run_demo(args.task)
        if not success:
            sys.exit(1)

    elif args.command == "benchmark":
        print("\n[>>] Running Performance Benchmarks...")
        results = await orchestrator.run_benchmark()
        if "error" in results:
            print(f"[ERR] {results['error']}")
            sys.exit(1)

    elif args.command == "reset":
        if args.component:
            # Reset specific circuit breaker
            if orchestrator.reset_circuit(args.component):
                print(f"[OK] Circuit breaker for '{args.component}' reset to CLOSED")
            else:
                print(f"[ERR] Unknown component: {args.component}")
                print(f"Valid components: {list(orchestrator._circuit_breakers.keys())}")
                sys.exit(1)
        else:
            # Reset all circuit breakers
            for component in orchestrator._circuit_breakers:
                orchestrator.reset_circuit(component)
            print("[OK] All circuit breakers reset to CLOSED")

        # Show updated status
        status = await orchestrator.check_health()
        print_status(status, show_circuits=True)

    elif args.command == "serve":
        print("\n[>>] Starting Platform Service with Graceful Shutdown...")
        await orchestrator.serve(port=args.port)


if __name__ == "__main__":
    asyncio.run(main())
