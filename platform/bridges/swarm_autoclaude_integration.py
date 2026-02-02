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
Swarm-AutoClaude Integration - Ultimate Autonomous Platform

Full integration between swarm coordinator and Auto-Claude task system.
Enables intelligent task routing based on task type and urgency.

Architecture:
1. Swarm generates tasks from coordinator decisions
2. Integration layer routes to Auto-Claude or handles locally
3. Results flow back to swarm memory for coordination

References:
- auto_claude_bridge.py: Task format and queue management
- coordinator.py: Swarm coordination patterns
- AGENT_SELECTION_GUIDE.md: When to use which component
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "swarm"))

from auto_claude_bridge import (
    AutoClaudeBridge, AutoClaudeTask, TaskType, TaskPriority, TaskResult
)
from coordinator import (
    Agent, AgentRole, Task, TaskStatus, SwarmMemory, QueenCoordinator
)

logger = structlog.get_logger(__name__)


class RoutingDecision(Enum):
    """Where to route a task."""
    AUTO_CLAUDE = "auto_claude"  # Send to Auto-Claude IDE
    SWARM_LOCAL = "swarm_local"  # Handle within swarm
    GRAPHITI = "graphiti"        # Knowledge graph operation
    HYBRID = "hybrid"            # Both Auto-Claude and swarm


@dataclass
class IntegrationConfig:
    """Configuration for swarm-autoclaude integration."""
    auto_claude_url: str = "http://localhost:3000"
    qdrant_url: str = "http://localhost:6333"
    code_review_threshold: int = 3  # Files > threshold go to Auto-Claude
    test_generation_threshold: int = 100  # LOC > threshold go to Auto-Claude
    enable_async_results: bool = True
    max_parallel_auto_claude_tasks: int = 5


class SwarmAutoClaudeIntegration:
    """
    Integration layer between Swarm Coordinator and Auto-Claude.

    Responsibilities:
    1. Intelligent task routing based on complexity
    2. Bidirectional result flow
    3. State synchronization between systems
    4. Hybrid task execution (parallel swarm + Auto-Claude)
    """

    def __init__(self, config: Optional[IntegrationConfig] = None):
        self.config = config or IntegrationConfig()
        self.bridge = AutoClaudeBridge(auto_claude_url=self.config.auto_claude_url)
        self.memory = SwarmMemory(self.config.qdrant_url)
        self.coordinator = QueenCoordinator(self.memory)

        self._active_hybrid_tasks: Dict[str, Dict[str, Any]] = {}
        self._routing_stats: Dict[str, int] = {
            "auto_claude": 0,
            "swarm_local": 0,
            "graphiti": 0,
            "hybrid": 0,
        }

    def analyze_task(self, task: Task) -> RoutingDecision:
        """
        Analyze a task to determine optimal routing.

        Decision matrix:
        - Code review with many files -> Auto-Claude
        - Simple status queries -> Swarm local
        - Documentation generation -> Auto-Claude
        - Task coordination -> Swarm local
        - Architecture decisions -> Hybrid
        """
        desc_lower = task.description.lower()

        # High-complexity tasks go to Auto-Claude
        if any(kw in desc_lower for kw in ["review", "audit", "security", "test"]):
            return RoutingDecision.AUTO_CLAUDE

        # Documentation tasks go to Auto-Claude
        if any(kw in desc_lower for kw in ["document", "docstring", "readme"]):
            return RoutingDecision.AUTO_CLAUDE

        # Knowledge queries go to Graphiti
        if any(kw in desc_lower for kw in ["find", "search", "what", "who", "when"]):
            return RoutingDecision.GRAPHITI

        # Architecture decisions are hybrid
        if any(kw in desc_lower for kw in ["design", "architect", "decide", "plan"]):
            return RoutingDecision.HYBRID

        # Default to swarm local
        return RoutingDecision.SWARM_LOCAL

    async def submit_task(
        self,
        description: str,
        files: Optional[List[str]] = None,
        priority: int = 5,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Submit a task to the integrated system.

        The system automatically determines optimal routing.
        Returns task ID for tracking.
        """
        # Create swarm task
        task = Task(
            id=f"integrated-{datetime.now(timezone.utc).strftime('%H%M%S')}-{hash(description) % 10000:04d}",
            description=description,
            priority=priority,
            context=context or {}
        )

        if files:
            task.context["files"] = files

        # Determine routing
        routing = self.analyze_task(task)
        self._routing_stats[routing.value] += 1

        logger.info(
            "task_routed",
            task_id=task.id,
            routing=routing.value,
            priority=priority
        )

        if routing == RoutingDecision.AUTO_CLAUDE:
            return await self._route_to_auto_claude(task, files or [])

        elif routing == RoutingDecision.SWARM_LOCAL:
            return await self._route_to_swarm(task)

        elif routing == RoutingDecision.GRAPHITI:
            return await self._route_to_graphiti(task)

        elif routing == RoutingDecision.HYBRID:
            return await self._route_hybrid(task, files or [])

        return task.id

    async def _route_to_auto_claude(self, task: Task, files: List[str]) -> str:
        """Route task to Auto-Claude."""
        # Convert to Auto-Claude format
        ac_task = AutoClaudeTask.from_swarm_task({
            "id": task.id,
            "description": task.description,
            "priority": task.priority,
            "files": files,
            "context": task.context
        })

        # Submit with callback to sync results
        await self.bridge.submit_task(
            ac_task,
            callback=lambda r: asyncio.create_task(self._sync_result_to_swarm(r))
        )

        return task.id

    async def _route_to_swarm(self, task: Task) -> str:
        """Route task to swarm coordinator."""
        await self.coordinator.submit_task(task)

        # Auto-assign if we have idle workers
        await self.coordinator.assign_task(task.id)

        return task.id

    async def _route_to_graphiti(self, task: Task) -> str:
        """Route task to Graphiti knowledge graph."""
        # For now, store in swarm memory with graphiti prefix
        await self.memory.store(
            f"graphiti/queries/{task.id}",
            {
                "description": task.description,
                "status": "pending",
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        )
        return task.id

    async def _route_hybrid(self, task: Task, files: List[str]) -> str:
        """
        Route task to both Auto-Claude and swarm.

        Used for complex tasks that benefit from parallel processing:
        - Auto-Claude handles code generation/review
        - Swarm handles coordination and status tracking
        """
        # Track hybrid task
        self._active_hybrid_tasks[task.id] = {
            "swarm_status": "pending",
            "auto_claude_status": "pending",
            "started_at": datetime.now(timezone.utc).isoformat()
        }

        # Submit to both systems
        await self._route_to_swarm(task)
        await self._route_to_auto_claude(task, files)

        return task.id

    async def _sync_result_to_swarm(self, result: TaskResult) -> None:
        """Sync Auto-Claude result back to swarm memory."""
        await self.memory.store(
            f"swarm/results/{result.task_id}",
            {
                "success": result.success,
                "output": result.output,
                "error": result.error,
                "duration_ms": result.duration_ms,
                "files_modified": result.files_modified,
                "completed_at": result.completed_at.isoformat()
            }
        )

        # Update hybrid task if applicable
        if result.task_id in self._active_hybrid_tasks:
            self._active_hybrid_tasks[result.task_id]["auto_claude_status"] = \
                "completed" if result.success else "failed"

        logger.info(
            "result_synced_to_swarm",
            task_id=result.task_id,
            success=result.success
        )

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get comprehensive task status from all systems."""
        status = {
            "task_id": task_id,
            "swarm_status": None,
            "auto_claude_status": None,
            "hybrid": task_id in self._active_hybrid_tasks
        }

        # Check swarm
        swarm_status = await self.coordinator.get_status()
        for t in swarm_status.get("pending_tasks", []) + swarm_status.get("completed_tasks", []):
            if t == task_id:
                status["swarm_status"] = "found"
                break

        # Check Auto-Claude
        result = await self.bridge.check_result(task_id)
        if result:
            status["auto_claude_status"] = "completed" if result.success else "failed"
            status["auto_claude_result"] = {
                "output": result.output,
                "duration_ms": result.duration_ms
            }

        return status

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get statistics on task routing."""
        total = sum(self._routing_stats.values())
        return {
            "total_tasks": total,
            "routing_breakdown": self._routing_stats,
            "routing_percentages": {
                k: (v / total * 100) if total > 0 else 0
                for k, v in self._routing_stats.items()
            }
        }


# Demo and testing
async def main():
    """Demo the swarm-autoclaude integration."""
    print("=" * 60)
    print("SWARM-AUTOCLAUDE INTEGRATION DEMO")
    print("=" * 60)

    integration = SwarmAutoClaudeIntegration()

    # Register a worker for local tasks
    worker = Agent(
        id="integration-worker-001",
        name="Integration Worker",
        role=AgentRole.WORKER,
        capabilities=["review", "test", "document"]
    )
    await integration.coordinator.register_worker(worker)
    print(f"\nRegistered worker: {worker.name}")

    # Submit various tasks to see routing decisions
    print("\nSubmitting tasks with intelligent routing...\n")

    tasks = [
        ("Review the coordinator.py for security issues", ["coordinator.py"]),
        ("Generate unit tests for the bridge module", ["auto_claude_bridge.py"]),
        ("Check swarm status and report", None),
        ("Design the architecture for multi-region deployment", ["architecture.md"]),
        ("Find all agents that handle code review", None),
    ]

    for desc, files in tasks:
        task_id = await integration.submit_task(
            description=desc,
            files=files,
            priority=7
        )
        routing = integration.analyze_task(Task(id="x", description=desc, priority=5))
        print(f"  [{routing.value:12}] {desc[:50]}... -> {task_id}")

    # Show routing statistics
    print("\nRouting Statistics:")
    stats = integration.get_routing_stats()
    print(f"  Total tasks: {stats['total_tasks']}")
    for route, count in stats['routing_breakdown'].items():
        pct = stats['routing_percentages'][route]
        print(f"  {route}: {count} ({pct:.1f}%)")

    # Get queue status
    print("\nAuto-Claude Queue:")
    queue_status = integration.bridge.get_queue_status()
    print(f"  Pending: {queue_status['pending']}")
    print(f"  Queue path: {queue_status['queue_path']}")

    print("\nSwarm Status:")
    swarm_status = await integration.coordinator.get_status()
    print(f"  Workers: {swarm_status['workers']}")
    print(f"  Pending tasks: {swarm_status['pending_tasks']}")

    print("\nDemo complete!")


if __name__ == "__main__":
    asyncio.run(main())
