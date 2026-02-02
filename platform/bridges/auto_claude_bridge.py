#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.26.0",
#     "structlog>=24.1.0",
# ]
# ///
"""
Auto-Claude Task Bridge - Ultimate Autonomous Platform

Bridge between the Swarm Coordinator and Auto-Claude IDE.
Enables bidirectional task flow between Claude Code swarm agents and
Auto-Claude's agentic workflows.

Task Flow:
1. Swarm Agent -> Bridge -> Auto-Claude Task Queue
2. Auto-Claude Result -> Bridge -> Swarm Memory

Based on: SWARM_PATTERNS_CATALOG.md, Auto-Claude planner.md analysis
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


class TaskType(Enum):
    """Auto-Claude task types from planner.md."""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    SECURITY_AUDIT = "security_audit"
    PERFORMANCE = "performance"


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class AutoClaudeTask:
    """
    Task format compatible with Auto-Claude planner.

    Based on analysis of auto-claude/apps/backend/prompts/planner.md:
    - Uses structured JSON for task definitions
    - Supports context injection
    - Includes dependency tracking
    """
    id: str
    type: TaskType
    title: str
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    context: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    timeout_seconds: int = 300
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_planner_format(self) -> Dict[str, Any]:
        """Convert to Auto-Claude planner format."""
        return {
            "task_id": self.id,
            "task_type": self.type.value,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "context": {
                "files": self.files,
                "dependencies": self.dependencies,
                **self.context
            },
            "metadata": {
                "created_at": self.created_at.isoformat(),
                "timeout_seconds": self.timeout_seconds,
                **self.metadata
            }
        }

    @classmethod
    def from_swarm_task(cls, swarm_task: Dict[str, Any]) -> "AutoClaudeTask":
        """Convert from swarm coordinator task format."""
        # Map swarm priority (1-10) to TaskPriority
        swarm_priority = swarm_task.get("priority", 5)
        if swarm_priority >= 8:
            priority = TaskPriority.CRITICAL
        elif swarm_priority >= 6:
            priority = TaskPriority.HIGH
        elif swarm_priority >= 4:
            priority = TaskPriority.MEDIUM
        else:
            priority = TaskPriority.LOW

        # Infer task type from description
        description = swarm_task.get("description", "").lower()
        if "review" in description:
            task_type = TaskType.CODE_REVIEW
        elif "test" in description:
            task_type = TaskType.TESTING
        elif "debug" in description:
            task_type = TaskType.DEBUGGING
        elif "doc" in description:
            task_type = TaskType.DOCUMENTATION
        elif "security" in description or "audit" in description:
            task_type = TaskType.SECURITY_AUDIT
        elif "refactor" in description:
            task_type = TaskType.REFACTORING
        elif "perf" in description or "optim" in description:
            task_type = TaskType.PERFORMANCE
        else:
            task_type = TaskType.CODE_GENERATION

        return cls(
            id=swarm_task.get("id", str(uuid.uuid4())),
            type=task_type,
            title=swarm_task.get("description", "Untitled Task")[:80],
            description=swarm_task.get("description", ""),
            priority=priority,
            context=swarm_task.get("context", {}),
            dependencies=swarm_task.get("dependencies", []),
            files=swarm_task.get("files", []),
            metadata={"source": "swarm_coordinator"}
        )


@dataclass
class TaskResult:
    """Result from Auto-Claude task execution."""
    task_id: str
    success: bool
    output: Any
    error: Optional[str] = None
    duration_ms: float = 0
    files_modified: List[str] = field(default_factory=list)
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AutoClaudeBridge:
    """
    Bridge between Swarm Coordinator and Auto-Claude.

    Responsibilities:
    1. Convert swarm tasks to Auto-Claude format
    2. Queue tasks for Auto-Claude processing
    3. Receive and route results back to swarm
    4. Handle task lifecycle and timeouts
    """

    def __init__(
        self,
        auto_claude_url: str = "http://localhost:3000",
        queue_path: Optional[Path] = None
    ):
        self.auto_claude_url = auto_claude_url
        self.queue_path = queue_path or Path.home() / ".claude" / "v10" / "task_queue"
        self.queue_path.mkdir(parents=True, exist_ok=True)

        self._pending_tasks: Dict[str, AutoClaudeTask] = {}
        self._results: Dict[str, TaskResult] = {}
        self._callbacks: Dict[str, Callable[[TaskResult], None]] = {}

    async def submit_task(
        self,
        task: AutoClaudeTask,
        callback: Optional[Callable[[TaskResult], None]] = None
    ) -> str:
        """
        Submit a task to Auto-Claude.

        Returns task ID for tracking.
        """
        self._pending_tasks[task.id] = task

        if callback:
            self._callbacks[task.id] = callback

        # Write to file-based queue (works without Auto-Claude server)
        task_file = self.queue_path / f"{task.id}.json"
        task_file.write_text(
            json.dumps(task.to_planner_format(), indent=2, default=str),
            encoding="utf-8"
        )

        logger.info(
            "task_submitted",
            task_id=task.id,
            task_type=task.type.value,
            priority=task.priority.value
        )

        # Try to submit to Auto-Claude server if available
        await self._try_http_submit(task)

        return task.id

    async def _try_http_submit(self, task: AutoClaudeTask) -> bool:
        """Try to submit task via HTTP to Auto-Claude server."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.auto_claude_url}/api/tasks",
                    json=task.to_planner_format()
                )
                if response.status_code in (200, 201, 202):
                    logger.info("task_http_submitted", task_id=task.id)
                    return True
                else:
                    logger.warning(
                        "task_http_failed",
                        task_id=task.id,
                        status=response.status_code
                    )
        except Exception as e:
            logger.debug("auto_claude_unavailable", error=str(e))
        return False

    async def check_result(self, task_id: str) -> Optional[TaskResult]:
        """Check if a task has completed."""
        # Check cached results
        if task_id in self._results:
            return self._results[task_id]

        # Check result file
        result_file = self.queue_path / f"{task_id}.result.json"
        if result_file.exists():
            try:
                data = json.loads(result_file.read_text())
                result = TaskResult(
                    task_id=task_id,
                    success=data.get("success", False),
                    output=data.get("output"),
                    error=data.get("error"),
                    duration_ms=data.get("duration_ms", 0),
                    files_modified=data.get("files_modified", [])
                )
                self._results[task_id] = result

                # Trigger callback if registered
                if task_id in self._callbacks:
                    self._callbacks[task_id](result)
                    del self._callbacks[task_id]

                return result
            except Exception as e:
                logger.error("result_parse_error", task_id=task_id, error=str(e))

        return None

    async def wait_for_result(
        self,
        task_id: str,
        timeout: float = 300.0,
        poll_interval: float = 1.0
    ) -> Optional[TaskResult]:
        """Wait for a task to complete with timeout."""
        elapsed = 0.0

        while elapsed < timeout:
            result = await self.check_result(task_id)
            if result:
                return result

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        logger.warning("task_timeout", task_id=task_id, timeout=timeout)
        return None

    def complete_task(self, task_id: str, result: TaskResult) -> None:
        """
        Mark a task as complete (called by Auto-Claude or manually).
        """
        self._results[task_id] = result

        # Write result file
        result_file = self.queue_path / f"{task_id}.result.json"
        result_file.write_text(
            json.dumps({
                "task_id": task_id,
                "success": result.success,
                "output": result.output,
                "error": result.error,
                "duration_ms": result.duration_ms,
                "files_modified": result.files_modified,
                "completed_at": result.completed_at.isoformat()
            }, indent=2, default=str),
            encoding="utf-8"
        )

        # Remove from pending
        if task_id in self._pending_tasks:
            del self._pending_tasks[task_id]

        # Trigger callback
        if task_id in self._callbacks:
            self._callbacks[task_id](result)
            del self._callbacks[task_id]

        logger.info(
            "task_completed",
            task_id=task_id,
            success=result.success,
            duration_ms=result.duration_ms
        )

    def get_pending_tasks(self) -> List[AutoClaudeTask]:
        """Get all pending tasks."""
        return list(self._pending_tasks.values())

    def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status."""
        return {
            "pending": len(self._pending_tasks),
            "completed": len(self._results),
            "queue_path": str(self.queue_path),
            "tasks": [
                {
                    "id": t.id,
                    "type": t.type.value,
                    "priority": t.priority.value,
                    "title": t.title[:50]
                }
                for t in self._pending_tasks.values()
            ]
        }


class SwarmToAutoClaude:
    """
    Integration layer for Swarm Coordinator to Auto-Claude.

    Provides high-level methods for swarm agents to interact
    with Auto-Claude's task system.
    """

    def __init__(self, bridge: Optional[AutoClaudeBridge] = None):
        self.bridge = bridge or AutoClaudeBridge()

    async def request_code_review(
        self,
        files: List[str],
        context: str = "",
        priority: TaskPriority = TaskPriority.MEDIUM
    ) -> str:
        """Request code review from Auto-Claude."""
        task = AutoClaudeTask(
            id=f"review-{uuid.uuid4().hex[:8]}",
            type=TaskType.CODE_REVIEW,
            title=f"Code Review: {len(files)} files",
            description=f"Review the following files for quality, security, and best practices.\n\n{context}",
            priority=priority,
            files=files
        )
        return await self.bridge.submit_task(task)

    async def request_test_generation(
        self,
        target_file: str,
        test_type: str = "unit",
        priority: TaskPriority = TaskPriority.MEDIUM
    ) -> str:
        """Request test generation for a file."""
        task = AutoClaudeTask(
            id=f"test-{uuid.uuid4().hex[:8]}",
            type=TaskType.TESTING,
            title=f"Generate {test_type} tests for {Path(target_file).name}",
            description=f"Generate comprehensive {test_type} tests for {target_file}",
            priority=priority,
            files=[target_file],
            context={"test_type": test_type}
        )
        return await self.bridge.submit_task(task)

    async def request_documentation(
        self,
        files: List[str],
        doc_type: str = "docstring",
        priority: TaskPriority = TaskPriority.LOW
    ) -> str:
        """Request documentation generation."""
        task = AutoClaudeTask(
            id=f"doc-{uuid.uuid4().hex[:8]}",
            type=TaskType.DOCUMENTATION,
            title=f"Generate {doc_type} documentation",
            description=f"Generate {doc_type} documentation for the specified files.",
            priority=priority,
            files=files,
            context={"doc_type": doc_type}
        )
        return await self.bridge.submit_task(task)

    async def request_security_audit(
        self,
        files: List[str],
        priority: TaskPriority = TaskPriority.HIGH
    ) -> str:
        """Request security audit."""
        task = AutoClaudeTask(
            id=f"security-{uuid.uuid4().hex[:8]}",
            type=TaskType.SECURITY_AUDIT,
            title=f"Security Audit: {len(files)} files",
            description="Perform security audit checking for vulnerabilities, injection risks, and OWASP top 10.",
            priority=priority,
            files=files
        )
        return await self.bridge.submit_task(task)


# Demo/test
async def main():
    """Demo the Auto-Claude bridge."""
    print("=" * 50)
    print("AUTO-CLAUDE TASK BRIDGE DEMO")
    print("=" * 50)

    bridge = AutoClaudeBridge()
    swarm = SwarmToAutoClaude(bridge)

    # Create some tasks
    print("\nSubmitting tasks...")

    # Code review request
    review_id = await swarm.request_code_review(
        files=["coordinator.py", "test_coordinator.py"],
        context="Review the swarm coordinator implementation",
        priority=TaskPriority.HIGH
    )
    print(f"  Code Review: {review_id}")

    # Test generation request
    test_id = await swarm.request_test_generation(
        target_file="auto_claude_bridge.py",
        test_type="integration"
    )
    print(f"  Test Generation: {test_id}")

    # Security audit request
    security_id = await swarm.request_security_audit(
        files=["coordinator.py"],
        priority=TaskPriority.CRITICAL
    )
    print(f"  Security Audit: {security_id}")

    # Get queue status
    status = bridge.get_queue_status()
    print(f"\nQueue Status:")
    print(f"  Pending: {status['pending']}")
    print(f"  Completed: {status['completed']}")
    print(f"  Queue Path: {status['queue_path']}")

    # Simulate completing a task
    print("\nSimulating task completion...")
    bridge.complete_task(review_id, TaskResult(
        task_id=review_id,
        success=True,
        output={
            "issues_found": 3,
            "suggestions": ["Add type hints", "Improve error handling", "Add docstrings"]
        },
        duration_ms=1500,
        files_modified=[]
    ))

    # Check result
    result = await bridge.check_result(review_id)
    if result:
        print(f"  Task {review_id} completed!")
        print(f"  Success: {result.success}")
        print(f"  Output: {result.output}")

    print("\nDemo complete!")


if __name__ == "__main__":
    asyncio.run(main())
