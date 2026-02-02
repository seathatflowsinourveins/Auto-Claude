#!/usr/bin/env python3
"""
Live Qdrant Integration Test for Swarm Coordinator
Tests real Qdrant connection and swarm memory operations
"""

import asyncio
import sys
from datetime import datetime, timezone

# Add parent to path for imports
sys.path.insert(0, str(__file__).rsplit("\\", 1)[0])

from coordinator import (
    Agent, AgentRole, Task, TaskStatus, Topology,
    SwarmMemory, QueenCoordinator, SwarmOrchestrator
)


async def test_qdrant_connection():
    """Test basic Qdrant connectivity."""
    print("\n=== Testing Qdrant Connection ===")

    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url="http://localhost:6333")
        collections = client.get_collections()

        print(f"  Connected to Qdrant")
        print(f"  Collections: {len(collections.collections)}")
        for col in collections.collections:
            print(f"    - {col.name}")

        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


async def test_swarm_memory_store():
    """Test SwarmMemory store operation with real Qdrant."""
    print("\n=== Testing SwarmMemory Store ===")

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance

        # Ensure collection exists
        client = QdrantClient(url="http://localhost:6333")
        collection_name = "swarm_coordination"

        try:
            client.get_collection(collection_name)
            print(f"  Collection '{collection_name}' exists")
        except Exception:
            print(f"  Creating collection '{collection_name}'...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            print(f"  Collection created")

        # Test store operation
        memory = SwarmMemory("http://localhost:6333")

        # Store test data
        test_key = f"test/swarm/timestamp-{datetime.now(timezone.utc).isoformat()}"
        test_value = {
            "type": "test",
            "message": "Live Qdrant integration test",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        result = await memory.store(test_key, test_value)

        if result:
            print(f"  Stored: {test_key}")
            print(f"  Value: {test_value}")
            return True
        else:
            print(f"  Failed to store")
            return False

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


async def test_queen_with_qdrant():
    """Test QueenCoordinator with real Qdrant backend."""
    print("\n=== Testing QueenCoordinator with Qdrant ===")

    try:
        memory = SwarmMemory("http://localhost:6333")
        queen = QueenCoordinator(memory)

        # Register a worker
        worker = Agent(
            id="live-worker-001",
            name="Live Test Worker",
            role=AgentRole.WORKER,
            capabilities=["test", "integration"]
        )

        registered = await queen.register_worker(worker)
        print(f"  Worker registered: {registered}")

        # Submit a task
        task = Task(
            id="live-task-001",
            description="Live Qdrant integration test task",
            priority=7
        )

        task_id = await queen.submit_task(task)
        print(f"  Task submitted: {task_id}")

        # Assign the task
        assigned_worker = await queen.assign_task(task_id)
        print(f"  Task assigned to: {assigned_worker}")

        # Complete the task
        completed = await queen.complete_task(task_id, {
            "status": "success",
            "test": "live_qdrant",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        print(f"  Task completed: {completed}")

        # Get status
        status = await queen.get_status()
        print(f"\n  Swarm Status:")
        print(f"    Queen ID: {status['queen_id']}")
        print(f"    Workers: {status['workers']}")
        print(f"    Completed: {status['completed_tasks']}")

        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_orchestrator_with_qdrant():
    """Test SwarmOrchestrator with real Qdrant."""
    print("\n=== Testing SwarmOrchestrator with Qdrant ===")

    try:
        orchestrator = SwarmOrchestrator(
            topology=Topology.HIERARCHICAL,
            qdrant_url="http://localhost:6333"
        )

        # Register workers
        for i in range(2):
            worker = Agent(
                id=f"orchestrator-worker-{i:03d}",
                name=f"Orchestrator Worker {i}",
                role=AgentRole.WORKER
            )
            await orchestrator.coordinator.register_worker(worker)

        print(f"  Registered 2 workers")

        # Submit and process tasks
        for i in range(3):
            task = Task(
                id=f"orch-task-{i:03d}",
                description=f"Orchestrator test task {i}",
                priority=5 + i
            )
            await orchestrator.coordinator.submit_task(task)
            await orchestrator.coordinator.assign_task(task.id)

        print(f"  Submitted and assigned 3 tasks")

        # Get final status
        status = await orchestrator.coordinator.get_status()
        print(f"\n  Final Status:")
        print(f"    Queen: {status['queen_id']}")
        print(f"    Workers: {status['workers']} ({status['idle_workers']} idle)")
        print(f"    Pending: {status['pending_tasks']}")

        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


async def main():
    """Run all live Qdrant tests."""
    print("=" * 50)
    print("LIVE QDRANT INTEGRATION TESTS")
    print("=" * 50)

    results = []

    # Run tests
    results.append(("qdrant_connection", await test_qdrant_connection()))
    results.append(("swarm_memory_store", await test_swarm_memory_store()))
    results.append(("queen_with_qdrant", await test_queen_with_qdrant()))
    results.append(("orchestrator_with_qdrant", await test_orchestrator_with_qdrant()))

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
        print("All live Qdrant tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
