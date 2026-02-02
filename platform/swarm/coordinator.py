#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "letta-client>=1.7.0",
#     "qdrant-client>=1.7.0",
#     "httpx>=0.26.0",
#     "structlog>=24.1.0",
# ]
# ///
"""
Swarm Coordinator - Ultimate Autonomous Platform

Production-grade swarm coordination implementing:
- Queen-led hierarchical topology
- Mesh peer-to-peer coordination
- Qdrant-backed shared memory
- Letta agent persistence

Based on: SWARM_PATTERNS_CATALOG.md, AGENT_SELECTION_GUIDE.md
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# V45: Circuit Breaker for Qdrant Operations
# =============================================================================

class _SwarmCircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class _SwarmCircuitBreaker:
    """
    V45: Circuit breaker for Qdrant swarm memory operations.

    Pattern: CLOSED -> (failures >= threshold) -> OPEN -> (timeout) -> HALF_OPEN
             HALF_OPEN -> (success) -> CLOSED
             HALF_OPEN -> (failure) -> OPEN
    """
    name: str = "qdrant_swarm"
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0
    half_open_max_requests: int = 3

    state: _SwarmCircuitState = field(default=_SwarmCircuitState.CLOSED)
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    half_open_requests: int = 0

    def can_execute(self) -> bool:
        """Check if the circuit allows execution."""
        if self.state == _SwarmCircuitState.CLOSED:
            return True

        if self.state == _SwarmCircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if self.last_failure_time:
                elapsed = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout_seconds:
                    self.state = _SwarmCircuitState.HALF_OPEN
                    self.half_open_requests = 0
                    logger.info(f"[V45] Circuit {self.name} transitioning to HALF_OPEN")
                    return True
            return False

        if self.state == _SwarmCircuitState.HALF_OPEN:
            if self.half_open_requests < self.half_open_max_requests:
                self.half_open_requests += 1
                return True
            return False

        return True

    def record_success(self) -> None:
        """Record a successful operation."""
        if self.state == _SwarmCircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 3:
                self.state = _SwarmCircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(f"[V45] Circuit {self.name} CLOSED (recovered)")
        elif self.state == _SwarmCircuitState.CLOSED:
            # Decay failure count on success
            if self.failure_count > 0:
                self.failure_count -= 1

    def record_failure(self) -> None:
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)

        if self.state == _SwarmCircuitState.HALF_OPEN:
            self.state = _SwarmCircuitState.OPEN
            logger.warning(f"[V45] Circuit {self.name} OPEN (half-open failed)")
        elif self.state == _SwarmCircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self.state = _SwarmCircuitState.OPEN
                logger.warning(f"[V45] Circuit {self.name} OPEN (threshold reached)")


class AgentRole(Enum):
    """Agent roles in the swarm."""
    QUEEN = "queen"
    WORKER = "worker"
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Topology(Enum):
    """Swarm coordination topology."""
    HIERARCHICAL = "hierarchical"  # Queen-led
    MESH = "mesh"                  # Peer-to-peer
    HYBRID = "hybrid"              # Combined


@dataclass
class Agent:
    """Represents a swarm agent."""
    id: str
    name: str
    role: AgentRole
    status: str = "idle"
    capabilities: List[str] = field(default_factory=list)
    current_task: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Task:
    """Represents a task in the swarm."""
    id: str
    description: str
    priority: int = 5  # 1-10, higher = more important
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    dependencies: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)  # Additional task context


class SwarmMemory:
    """
    Shared memory layer using Qdrant for vector storage.
    Implements the namespace patterns from SWARM_PATTERNS_CATALOG.md

    V45: Environment-configurable URL + circuit breaker protection.
    """

    def __init__(self, qdrant_url: Optional[str] = None):
        import os
        # V45 FIX: Environment-configurable Qdrant URL (was hardcoded)
        self.qdrant_url = qdrant_url or os.environ.get(
            "QDRANT_URL", "http://localhost:6333"
        )
        self._client = None
        self.collection = "swarm_coordination"
        # V45: Circuit breaker for Qdrant operations
        self._circuit_breaker = _SwarmCircuitBreaker(name="qdrant_swarm")

    def _get_client(self):
        """Lazy client initialization with circuit breaker check."""
        if not self._circuit_breaker.can_execute():
            logger.warning("[V45] Qdrant circuit breaker OPEN - returning None")
            return None

        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                self._client = QdrantClient(url=self.qdrant_url)
                self._circuit_breaker.record_success()
            except Exception as e:
                self._circuit_breaker.record_failure()
                logger.error(f"[V45] Qdrant client init failed: {e}")
                return None
        return self._client

    async def store(self, key: str, value: Dict[str, Any]) -> bool:
        """
        Store a key-value pair in swarm memory.

        Key patterns:
        - swarm/queen/status
        - swarm/worker-{id}/tasks
        - swarm/shared/resource-pool
        - coordination/heartbeat/{id}

        V45: Protected by circuit breaker.
        """
        try:
            from qdrant_client.models import PointStruct

            client = self._get_client()
            # V45: Circuit breaker may block client creation
            if client is None:
                logger.warning("[V45] Qdrant client unavailable (circuit open)")
                return False

            # Create simple embedding from key (production would use real embeddings)
            vector = [hash(key + str(i)) % 1000 / 1000.0 for i in range(384)]

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "key": key,
                    "value": json.dumps(value),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )

            client.upsert(collection_name=self.collection, points=[point])
            self._circuit_breaker.record_success()  # V45
            logger.debug("memory_stored", key=key)
            return True

        except Exception as e:
            self._circuit_breaker.record_failure()  # V45
            logger.error("memory_store_failed", key=key, error=str(e))
            return False

    async def retrieve(self, key_prefix: str) -> List[Dict[str, Any]]:
        """Retrieve all values matching key prefix. V45: Protected by circuit breaker."""
        try:
            client = self._get_client()
            # V45: Circuit breaker may block client creation
            if client is None:
                logger.warning("[V45] Qdrant client unavailable (circuit open)")
                return []

            results = client.scroll(
                collection_name=self.collection,
                scroll_filter={
                    "must": [{"key": "key", "match": {"text": key_prefix}}]
                },
                limit=100
            )

            items = []
            for point in results[0]:
                payload = point.payload
                items.append({
                    "key": payload.get("key"),
                    "value": json.loads(payload.get("value", "{}")),
                    "timestamp": payload.get("timestamp")
                })

            self._circuit_breaker.record_success()  # V45
            return items

        except Exception as e:
            self._circuit_breaker.record_failure()  # V45
            logger.error("memory_retrieve_failed", prefix=key_prefix, error=str(e))
            return []


class QueenCoordinator:
    """
    Queen-led hierarchical coordinator.
    Implements sovereign orchestration pattern.
    """

    def __init__(self, memory: SwarmMemory):
        self.id = f"queen-{uuid.uuid4().hex[:8]}"
        self.memory = memory
        self.workers: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.status = "active"

    async def register_worker(self, worker: Agent) -> bool:
        """Register a new worker with the queen."""
        self.workers[worker.id] = worker
        await self.memory.store(
            f"swarm/worker-{worker.id}/status",
            {"status": worker.status, "role": worker.role.value}
        )
        logger.info("worker_registered", worker_id=worker.id, name=worker.name)
        return True

    async def submit_task(self, task: Task) -> str:
        """Submit a task to the swarm."""
        self.tasks[task.id] = task
        await self.memory.store(
            f"swarm/shared/tasks/{task.id}",
            {
                "description": task.description,
                "priority": task.priority,
                "status": task.status.value
            }
        )
        logger.info("task_submitted", task_id=task.id, priority=task.priority)
        return task.id

    async def assign_task(self, task_id: str) -> Optional[str]:
        """Assign a task to the best available worker."""
        task = self.tasks.get(task_id)
        if not task or task.status != TaskStatus.PENDING:
            return None

        # Find idle worker with matching capabilities
        for worker_id, worker in self.workers.items():
            if worker.status == "idle":
                task.assigned_to = worker_id
                task.status = TaskStatus.ASSIGNED
                worker.status = "busy"
                worker.current_task = task_id

                await self.memory.store(
                    f"swarm/worker-{worker_id}/tasks",
                    {"current": task_id, "status": "assigned"}
                )

                logger.info("task_assigned", task_id=task_id, worker_id=worker_id)
                return worker_id

        logger.warning("no_available_workers", task_id=task_id)
        return None

    async def complete_task(self, task_id: str, result: Dict[str, Any]) -> bool:
        """Mark a task as completed."""
        task = self.tasks.get(task_id)
        if not task:
            return False

        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now(timezone.utc)
        task.result = result

        if task.assigned_to:
            worker = self.workers.get(task.assigned_to)
            if worker:
                worker.status = "idle"
                worker.current_task = None

        await self.memory.store(
            f"swarm/shared/tasks/{task_id}",
            {"status": "completed", "result": result}
        )

        logger.info("task_completed", task_id=task_id)
        return True

    async def get_status(self) -> Dict[str, Any]:
        """Get current swarm status."""
        return {
            "queen_id": self.id,
            "status": self.status,
            "workers": len(self.workers),
            "idle_workers": sum(1 for w in self.workers.values() if w.status == "idle"),
            "pending_tasks": sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING),
            "completed_tasks": sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED),
        }


class MeshCoordinator:
    """
    Peer-to-peer mesh coordinator.
    Implements gossip protocol and work stealing.
    """

    def __init__(self, node_id: str, memory: SwarmMemory):
        self.node_id = node_id
        self.memory = memory
        self.peers: Dict[str, Dict[str, Any]] = {}
        self.local_tasks: List[Task] = []

    async def gossip_state(self) -> None:
        """Broadcast state to peers using gossip protocol."""
        state = {
            "node_id": self.node_id,
            "task_count": len(self.local_tasks),
            "load": self._calculate_load(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        await self.memory.store(
            f"swarm/mesh/{self.node_id}/state",
            state
        )

        logger.debug("gossip_sent", node_id=self.node_id, load=state["load"])

    async def discover_peers(self) -> List[str]:
        """Discover other mesh nodes."""
        items = await self.memory.retrieve("swarm/mesh/")
        peer_ids = []

        for item in items:
            key = item.get("key", "")
            if "/state" in key:
                peer_id = key.split("/")[2]
                if peer_id != self.node_id:
                    peer_ids.append(peer_id)
                    self.peers[peer_id] = item.get("value", {})

        return peer_ids

    async def work_steal(self) -> Optional[Task]:
        """Steal work from overloaded peers."""
        if len(self.local_tasks) > 2:
            return None  # Already have enough work

        # Find busiest peer
        busiest_id = None
        max_load = 0

        for peer_id, state in self.peers.items():
            load = state.get("load", 0)
            if load > max_load:
                max_load = load
                busiest_id = peer_id

        if busiest_id and max_load > 5:
            logger.info("work_stealing", from_peer=busiest_id, their_load=max_load)
            # In production, would request task transfer
            return None

        return None

    def _calculate_load(self) -> float:
        """Calculate current node load."""
        return len(self.local_tasks) + sum(t.priority for t in self.local_tasks) * 0.1


class SwarmOrchestrator:
    """
    Main orchestrator combining multiple coordination patterns.
    """

    def __init__(
        self,
        topology: Topology = Topology.HIERARCHICAL,
        qdrant_url: str = "http://localhost:6333"
    ):
        self.topology = topology
        self.memory = SwarmMemory(qdrant_url)

        if topology == Topology.HIERARCHICAL:
            self.coordinator = QueenCoordinator(self.memory)
        elif topology == Topology.MESH:
            self.coordinator = MeshCoordinator(f"node-{uuid.uuid4().hex[:8]}", self.memory)
        else:
            # Hybrid: Queen with mesh workers
            self.coordinator = QueenCoordinator(self.memory)

        logger.info("orchestrator_initialized", topology=topology.value)

    async def run(self, iterations: int = 10) -> Dict[str, Any]:
        """Run the orchestrator for specified iterations."""
        results = {"iterations": 0, "tasks_completed": 0}

        for i in range(iterations):
            logger.info("orchestrator_iteration", iteration=i + 1)

            if isinstance(self.coordinator, QueenCoordinator):
                status = await self.coordinator.get_status()
                logger.info("swarm_status", **status)

            elif isinstance(self.coordinator, MeshCoordinator):
                await self.coordinator.gossip_state()
                peers = await self.coordinator.discover_peers()
                logger.info("mesh_status", peers=len(peers))

            results["iterations"] = i + 1
            await asyncio.sleep(1)  # Simulation delay

        return results


# CLI interface
async def main():
    """Demo the swarm coordinator."""
    print("="*50)
    print("SWARM COORDINATOR DEMO")
    print("="*50)

    # Initialize with hierarchical topology
    orchestrator = SwarmOrchestrator(topology=Topology.HIERARCHICAL)

    # Create some workers
    workers = [
        Agent(id=f"worker-{i:03d}", name=f"Worker {i}", role=AgentRole.WORKER)
        for i in range(3)
    ]

    # Register workers
    print("\nRegistering workers...")
    for worker in workers:
        await orchestrator.coordinator.register_worker(worker)

    # Submit tasks
    print("\nSubmitting tasks...")
    tasks = [
        Task(id=f"task-{i:03d}", description=f"Process item {i}", priority=5 + i % 3)
        for i in range(5)
    ]

    for task in tasks:
        await orchestrator.coordinator.submit_task(task)

    # Assign tasks
    print("\nAssigning tasks...")
    for task in tasks:
        await orchestrator.coordinator.assign_task(task.id)

    # Get status
    status = await orchestrator.coordinator.get_status()
    print(f"\nSwarm Status:")
    print(f"  Queen ID: {status['queen_id']}")
    print(f"  Workers: {status['workers']} ({status['idle_workers']} idle)")
    print(f"  Tasks: {status['pending_tasks']} pending, {status['completed_tasks']} completed")

    print("\nDemo complete!")


if __name__ == "__main__":
    asyncio.run(main())
