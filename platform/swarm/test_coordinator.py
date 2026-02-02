#!/usr/bin/env python3
"""
Swarm Coordinator Test Suite
Tests the coordinator prototype without requiring Qdrant connection
"""

import asyncio
import sys
from datetime import datetime, timezone
from typing import Dict, Any

# Add parent to path for imports
sys.path.insert(0, str(__file__).rsplit("\\", 1)[0])

from coordinator import (
    Agent, AgentRole, Task, TaskStatus, Topology,
    SwarmMemory, QueenCoordinator, MeshCoordinator, SwarmOrchestrator
)


class MockSwarmMemory(SwarmMemory):
    """Mock memory that doesn't require Qdrant."""

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    async def store(self, key: str, value: Dict[str, Any]) -> bool:
        self._store[key] = {
            "key": key,
            "value": value,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        return True

    async def retrieve(self, key_prefix: str) -> list:
        return [
            item for key, item in self._store.items()
            if key.startswith(key_prefix)
        ]


async def test_queen_coordinator():
    """Test hierarchical queen-led coordination."""
    print("\n=== Testing QueenCoordinator ===")

    memory = MockSwarmMemory()
    queen = QueenCoordinator(memory)

    # Create workers
    workers = [
        Agent(id=f"worker-{i:03d}", name=f"Worker {i}", role=AgentRole.WORKER)
        for i in range(3)
    ]

    # Register workers
    print("Registering 3 workers...")
    for worker in workers:
        result = await queen.register_worker(worker)
        assert result, f"Failed to register {worker.id}"

    # Submit tasks
    print("Submitting 5 tasks...")
    tasks = [
        Task(id=f"task-{i:03d}", description=f"Process item {i}", priority=5 + i % 3)
        for i in range(5)
    ]

    for task in tasks:
        task_id = await queen.submit_task(task)
        assert task_id == task.id, f"Task ID mismatch"

    # Assign tasks
    print("Assigning tasks to workers...")
    assigned = 0
    for task in tasks:
        worker_id = await queen.assign_task(task.id)
        if worker_id:
            assigned += 1
            print(f"  {task.id} -> {worker_id}")

    # Complete some tasks
    print("Completing tasks...")
    await queen.complete_task("task-000", {"status": "success", "output": "done"})
    await queen.complete_task("task-001", {"status": "success", "output": "done"})

    # Get status
    status = await queen.get_status()
    print(f"\nSwarm Status:")
    print(f"  Queen ID: {status['queen_id']}")
    print(f"  Workers: {status['workers']} ({status['idle_workers']} idle)")
    print(f"  Pending: {status['pending_tasks']}, Completed: {status['completed_tasks']}")

    # Verify counts
    assert status['workers'] == 3
    assert status['completed_tasks'] == 2

    print("  PASS: QueenCoordinator tests passed!")
    return True


async def test_mesh_coordinator():
    """Test peer-to-peer mesh coordination."""
    print("\n=== Testing MeshCoordinator ===")

    memory = MockSwarmMemory()

    # Create multiple mesh nodes
    nodes = [
        MeshCoordinator(f"node-{i:03d}", memory)
        for i in range(3)
    ]

    # Add tasks to nodes
    print("Adding tasks to nodes...")
    nodes[0].local_tasks = [Task(id=f"t{i}", description=f"Task {i}") for i in range(5)]
    nodes[1].local_tasks = [Task(id=f"t{i}", description=f"Task {i}") for i in range(2)]
    nodes[2].local_tasks = []

    # Gossip state from all nodes
    print("Gossiping state...")
    for node in nodes:
        await node.gossip_state()

    # Node 2 discovers peers
    print("Discovering peers from node-002...")
    peers = await nodes[2].discover_peers()
    print(f"  Found {len(peers)} peers: {peers}")

    # Check work stealing
    print("Testing work stealing...")
    stolen = await nodes[2].work_steal()
    if nodes[2].peers:
        loads = {pid: state.get("load", 0) for pid, state in nodes[2].peers.items()}
        print(f"  Peer loads: {loads}")

    # Verify load calculation
    node0_load = nodes[0]._calculate_load()
    print(f"  Node 0 load: {node0_load}")
    assert node0_load > 0, "Load should be positive with tasks"

    print("  PASS: MeshCoordinator tests passed!")
    return True


async def test_orchestrator():
    """Test the main orchestrator."""
    print("\n=== Testing SwarmOrchestrator ===")

    # Test hierarchical mode (skip actual run to avoid Qdrant)
    print("Creating hierarchical orchestrator...")
    orchestrator = SwarmOrchestrator(topology=Topology.HIERARCHICAL)
    assert orchestrator.topology == Topology.HIERARCHICAL
    assert isinstance(orchestrator.coordinator, QueenCoordinator)
    print("  PASS: Hierarchical orchestrator created")

    # Test mesh mode
    print("Creating mesh orchestrator...")
    mesh_orchestrator = SwarmOrchestrator(topology=Topology.MESH)
    assert mesh_orchestrator.topology == Topology.MESH
    assert isinstance(mesh_orchestrator.coordinator, MeshCoordinator)
    print("  PASS: Mesh orchestrator created")

    print("  PASS: SwarmOrchestrator tests passed!")
    return True


async def test_memory_namespaces():
    """Test memory namespace patterns."""
    print("\n=== Testing Memory Namespaces ===")

    memory = MockSwarmMemory()

    # Test namespace patterns from SWARM_PATTERNS_CATALOG
    namespaces = {
        "swarm/queen/status": {"status": "active", "workers": 5},
        "swarm/worker-001/tasks": {"current": "task-001", "completed": 3},
        "swarm/shared/resource-pool": {"memory": 1024, "cpu": 4},
        "coordination/heartbeat/worker-001": {"last_seen": "2026-01-18T23:00:00Z"},
        "coordination/consensus/vote-123": {"proposer": "node-001", "value": 42},
    }

    print("Storing namespace entries...")
    for key, value in namespaces.items():
        result = await memory.store(key, value)
        assert result, f"Failed to store {key}"
        print(f"  {key} -> stored")

    # Test retrieval by prefix
    print("\nRetrieving by prefix...")
    swarm_items = await memory.retrieve("swarm/")
    print(f"  swarm/* -> {len(swarm_items)} items")
    assert len(swarm_items) == 3

    coord_items = await memory.retrieve("coordination/")
    print(f"  coordination/* -> {len(coord_items)} items")
    assert len(coord_items) == 2

    print("  PASS: Memory namespace tests passed!")
    return True


async def main():
    """Run all tests."""
    print("=" * 50)
    print("SWARM COORDINATOR TEST SUITE")
    print("=" * 50)

    results = []

    try:
        results.append(("queen_coordinator", await test_queen_coordinator()))
    except Exception as e:
        print(f"  FAIL: {e}")
        results.append(("queen_coordinator", False))

    try:
        results.append(("mesh_coordinator", await test_mesh_coordinator()))
    except Exception as e:
        print(f"  FAIL: {e}")
        results.append(("mesh_coordinator", False))

    try:
        results.append(("orchestrator", await test_orchestrator()))
    except Exception as e:
        print(f"  FAIL: {e}")
        results.append(("orchestrator", False))

    try:
        results.append(("memory_namespaces", await test_memory_namespaces()))
    except Exception as e:
        print(f"  FAIL: {e}")
        results.append(("memory_namespaces", False))

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All swarm coordinator tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
