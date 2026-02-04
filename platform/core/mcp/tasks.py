"""
MCP Tasks Primitive - Long-running async operation tracking.

This module implements the MCP Tasks primitive for managing long-running
operations in the Model Context Protocol:

- tasks/list: List all tasks and their status
- tasks/create: Create a new task
- tasks/get: Get task status and result

Tasks enable asynchronous workflows where:
1. Client requests a long-running operation
2. Server returns a task ID immediately
3. Client polls or subscribes for completion
4. Server notifies when task completes

Usage:
    from core.mcp.tasks import TaskManager, Task, TaskStatus

    # Create task manager
    manager = TaskManager()

    # Create a new task
    task = await manager.create_task(
        name="research",
        description="Research AI patterns",
        metadata={"query": "LangGraph patterns 2025"},
    )

    # Update task progress
    await manager.update_progress(task.id, progress=0.5, status_message="Searching...")

    # Complete task
    await manager.complete_task(task.id, result={"findings": [...]})

    # List tasks
    tasks = await manager.list_tasks(status=TaskStatus.RUNNING)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class TaskStatus(str, Enum):
    """Task lifecycle status."""

    PENDING = "pending"      # Created but not started
    RUNNING = "running"      # Currently executing
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"        # Failed with error
    CANCELLED = "cancelled"  # Cancelled by user
    TIMEOUT = "timeout"      # Exceeded time limit


class TaskPriority(str, Enum):
    """Task execution priority."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# Data Models
# =============================================================================

class TaskMetadata(BaseModel):
    """Metadata for a task."""

    source: Optional[str] = Field(default=None, description="Task source/creator")
    server: Optional[str] = Field(default=None, description="MCP server handling task")
    tool: Optional[str] = Field(default=None, description="Tool being executed")
    args: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    tags: List[str] = Field(default_factory=list, description="Task tags")
    correlation_id: Optional[str] = Field(default=None, description="Correlation ID")


class TaskProgress(BaseModel):
    """Progress information for a task."""

    percentage: float = Field(default=0.0, ge=0.0, le=1.0)
    message: Optional[str] = Field(default=None)
    current_step: Optional[str] = Field(default=None)
    total_steps: Optional[int] = Field(default=None)
    completed_steps: Optional[int] = Field(default=None)
    estimated_completion: Optional[float] = Field(default=None)


class Task(BaseModel):
    """A long-running task in the MCP system."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Task name")
    description: str = Field(default="", description="Task description")
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    priority: TaskPriority = Field(default=TaskPriority.NORMAL)

    # Timing
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = Field(default=None)
    completed_at: Optional[float] = Field(default=None)
    timeout_seconds: Optional[float] = Field(default=None)

    # Progress
    progress: TaskProgress = Field(default_factory=TaskProgress)

    # Results
    result: Optional[Any] = Field(default=None)
    error: Optional[str] = Field(default=None)
    error_details: Optional[Dict[str, Any]] = Field(default=None)

    # Metadata
    metadata: TaskMetadata = Field(default_factory=TaskMetadata)

    @property
    def is_terminal(self) -> bool:
        """Check if task is in a terminal state."""
        return self.status in {
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMEOUT,
        }

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get task duration in seconds."""
        if not self.started_at:
            return None
        end_time = self.completed_at or time.time()
        return end_time - self.started_at

    @property
    def is_expired(self) -> bool:
        """Check if task has exceeded timeout."""
        if not self.timeout_seconds or not self.started_at:
            return False
        elapsed = time.time() - self.started_at
        return elapsed > self.timeout_seconds

    def to_mcp_response(self) -> Dict[str, Any]:
        """Convert to MCP tasks/get response format."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "progress": {
                "percentage": self.progress.percentage,
                "message": self.progress.message,
            },
            "createdAt": datetime.fromtimestamp(self.created_at).isoformat(),
            "startedAt": datetime.fromtimestamp(self.started_at).isoformat() if self.started_at else None,
            "completedAt": datetime.fromtimestamp(self.completed_at).isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata.model_dump(exclude_none=True),
        }


class TaskCreateRequest(BaseModel):
    """Request to create a new task."""

    name: str = Field(..., description="Task name")
    description: str = Field(default="", description="Task description")
    priority: TaskPriority = Field(default=TaskPriority.NORMAL)
    timeout_seconds: Optional[float] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskListRequest(BaseModel):
    """Request to list tasks."""

    status: Optional[TaskStatus] = Field(default=None)
    server: Optional[str] = Field(default=None)
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    include_completed: bool = Field(default=False)


# =============================================================================
# Task Manager
# =============================================================================

class TaskManager:
    """Manager for MCP tasks.

    Provides the core functionality for the MCP Tasks primitive:
    - Create and track long-running tasks
    - Update task progress
    - Subscribe to task completion
    - Clean up completed tasks
    """

    def __init__(
        self,
        cleanup_interval_seconds: float = 300.0,
        completed_task_ttl_seconds: float = 3600.0,
        max_tasks: int = 10000,
    ):
        """Initialize the task manager.

        Args:
            cleanup_interval_seconds: Interval for cleaning up old tasks
            completed_task_ttl_seconds: TTL for completed tasks before cleanup
            max_tasks: Maximum number of tasks to retain
        """
        self._tasks: Dict[str, Task] = {}
        self._subscribers: Dict[str, List[Callable[[Task], None]]] = {}
        self._cleanup_interval = cleanup_interval_seconds
        self._completed_ttl = completed_task_ttl_seconds
        self._max_tasks = max_tasks
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Start the task manager."""
        if self._initialized:
            return

        self._initialized = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("[MCP_TASKS] Task manager initialized")

    async def shutdown(self) -> None:
        """Shutdown the task manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        self._initialized = False
        logger.info("[MCP_TASKS] Task manager shutdown")

    # -------------------------------------------------------------------------
    # MCP Protocol Methods
    # -------------------------------------------------------------------------

    async def tasks_create(self, request: TaskCreateRequest) -> Dict[str, Any]:
        """MCP tasks/create method.

        Creates a new task and returns its ID.

        Args:
            request: Task creation request

        Returns:
            Dict with task ID and initial status
        """
        task = await self.create_task(
            name=request.name,
            description=request.description,
            priority=request.priority,
            timeout_seconds=request.timeout_seconds,
            metadata=request.metadata,
        )

        return {
            "id": task.id,
            "status": task.status.value,
            "createdAt": datetime.fromtimestamp(task.created_at).isoformat(),
        }

    async def tasks_list(self, request: TaskListRequest) -> Dict[str, Any]:
        """MCP tasks/list method.

        Lists tasks matching the criteria.

        Args:
            request: List request parameters

        Returns:
            Dict with tasks array and pagination info
        """
        tasks = await self.list_tasks(
            status=request.status,
            server=request.server,
            include_completed=request.include_completed,
        )

        # Apply pagination
        total = len(tasks)
        tasks = tasks[request.offset:request.offset + request.limit]

        return {
            "tasks": [t.to_mcp_response() for t in tasks],
            "total": total,
            "offset": request.offset,
            "limit": request.limit,
        }

    async def tasks_get(self, task_id: str) -> Dict[str, Any]:
        """MCP tasks/get method.

        Gets the status and result of a task.

        Args:
            task_id: ID of the task

        Returns:
            Task details including status and result

        Raises:
            KeyError: If task not found
        """
        task = await self.get_task(task_id)
        if not task:
            raise KeyError(f"Task not found: {task_id}")

        return task.to_mcp_response()

    # -------------------------------------------------------------------------
    # Core Task Operations
    # -------------------------------------------------------------------------

    async def create_task(
        self,
        name: str,
        description: str = "",
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout_seconds: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """Create a new task.

        Args:
            name: Task name
            description: Task description
            priority: Task priority
            timeout_seconds: Optional timeout
            metadata: Optional metadata

        Returns:
            Created task
        """
        task_metadata = TaskMetadata(**(metadata or {}))

        task = Task(
            name=name,
            description=description,
            priority=priority,
            timeout_seconds=timeout_seconds,
            metadata=task_metadata,
        )

        async with self._lock:
            # Enforce max tasks limit
            if len(self._tasks) >= self._max_tasks:
                await self._evict_oldest_tasks()

            self._tasks[task.id] = task

        logger.info(f"[MCP_TASKS] Created task {task.id}: {name}")
        return task

    async def start_task(self, task_id: str) -> bool:
        """Mark a task as started.

        Args:
            task_id: ID of the task

        Returns:
            True if task was started
        """
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status != TaskStatus.PENDING:
                return False

            task.status = TaskStatus.RUNNING
            task.started_at = time.time()

        self._notify_subscribers(task)
        return True

    async def update_progress(
        self,
        task_id: str,
        progress: Optional[float] = None,
        status_message: Optional[str] = None,
        current_step: Optional[str] = None,
        total_steps: Optional[int] = None,
        completed_steps: Optional[int] = None,
    ) -> bool:
        """Update task progress.

        Args:
            task_id: ID of the task
            progress: Progress percentage (0-1)
            status_message: Human-readable status message
            current_step: Current step name
            total_steps: Total number of steps
            completed_steps: Number of completed steps

        Returns:
            True if task was updated
        """
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.is_terminal:
                return False

            if progress is not None:
                task.progress.percentage = max(0.0, min(1.0, progress))
            if status_message is not None:
                task.progress.message = status_message
            if current_step is not None:
                task.progress.current_step = current_step
            if total_steps is not None:
                task.progress.total_steps = total_steps
            if completed_steps is not None:
                task.progress.completed_steps = completed_steps

        self._notify_subscribers(task)
        return True

    async def complete_task(
        self,
        task_id: str,
        result: Any = None,
    ) -> bool:
        """Mark a task as completed.

        Args:
            task_id: ID of the task
            result: Task result

        Returns:
            True if task was completed
        """
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.is_terminal:
                return False

            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            task.result = result
            task.progress.percentage = 1.0

        logger.info(f"[MCP_TASKS] Completed task {task_id}")
        self._notify_subscribers(task)
        return True

    async def fail_task(
        self,
        task_id: str,
        error: str,
        error_details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Mark a task as failed.

        Args:
            task_id: ID of the task
            error: Error message
            error_details: Optional error details

        Returns:
            True if task was marked as failed
        """
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.is_terminal:
                return False

            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            task.error = error
            task.error_details = error_details

        logger.warning(f"[MCP_TASKS] Failed task {task_id}: {error}")
        self._notify_subscribers(task)
        return True

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task.

        Args:
            task_id: ID of the task

        Returns:
            True if task was cancelled
        """
        async with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.is_terminal:
                return False

            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()

        logger.info(f"[MCP_TASKS] Cancelled task {task_id}")
        self._notify_subscribers(task)
        return True

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID.

        Args:
            task_id: ID of the task

        Returns:
            Task or None if not found
        """
        return self._tasks.get(task_id)

    async def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        server: Optional[str] = None,
        include_completed: bool = False,
    ) -> List[Task]:
        """List tasks matching criteria.

        Args:
            status: Filter by status
            server: Filter by server
            include_completed: Include completed tasks

        Returns:
            List of matching tasks
        """
        tasks = []

        for task in self._tasks.values():
            # Status filter
            if status and task.status != status:
                continue

            # Server filter
            if server and task.metadata.server != server:
                continue

            # Completed filter
            if not include_completed and task.is_terminal:
                continue

            tasks.append(task)

        # Sort by created time (newest first)
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks

    # -------------------------------------------------------------------------
    # Subscription
    # -------------------------------------------------------------------------

    def subscribe(self, task_id: str, callback: Callable[[Task], None]) -> None:
        """Subscribe to task updates.

        Args:
            task_id: ID of the task
            callback: Callback for updates
        """
        if task_id not in self._subscribers:
            self._subscribers[task_id] = []
        self._subscribers[task_id].append(callback)

    def unsubscribe(self, task_id: str, callback: Callable[[Task], None]) -> None:
        """Unsubscribe from task updates.

        Args:
            task_id: ID of the task
            callback: Callback to remove
        """
        if task_id in self._subscribers:
            try:
                self._subscribers[task_id].remove(callback)
            except ValueError:
                pass

    def _notify_subscribers(self, task: Task) -> None:
        """Notify subscribers of task update."""
        callbacks = self._subscribers.get(task.id, [])
        for callback in callbacks:
            try:
                callback(task)
            except Exception as e:
                logger.error(f"[MCP_TASKS] Subscriber callback error: {e}")

        # Clean up subscriptions for terminal tasks
        if task.is_terminal and task.id in self._subscribers:
            del self._subscribers[task.id]

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    async def _cleanup_loop(self) -> None:
        """Background task to cleanup old tasks."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired_tasks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[MCP_TASKS] Cleanup error: {e}")

    async def _cleanup_expired_tasks(self) -> None:
        """Remove expired completed tasks."""
        now = time.time()
        to_remove = []

        async with self._lock:
            for task_id, task in self._tasks.items():
                if task.is_terminal and task.completed_at:
                    age = now - task.completed_at
                    if age > self._completed_ttl:
                        to_remove.append(task_id)

            for task_id in to_remove:
                del self._tasks[task_id]
                self._subscribers.pop(task_id, None)

        if to_remove:
            logger.info(f"[MCP_TASKS] Cleaned up {len(to_remove)} expired tasks")

    async def _evict_oldest_tasks(self) -> None:
        """Evict oldest completed tasks to make room for new ones."""
        # Get completed tasks sorted by completion time
        completed = [
            t for t in self._tasks.values()
            if t.is_terminal
        ]
        completed.sort(key=lambda t: t.completed_at or t.created_at)

        # Remove oldest 10%
        evict_count = max(1, len(completed) // 10)
        for task in completed[:evict_count]:
            del self._tasks[task.id]
            self._subscribers.pop(task.id, None)

    def get_stats(self) -> Dict[str, Any]:
        """Get task manager statistics."""
        by_status: Dict[str, int] = {}
        for task in self._tasks.values():
            status = task.status.value
            by_status[status] = by_status.get(status, 0) + 1

        return {
            "total_tasks": len(self._tasks),
            "by_status": by_status,
            "active_subscriptions": sum(len(s) for s in self._subscribers.values()),
            "initialized": self._initialized,
        }


# =============================================================================
# Global Instance
# =============================================================================

_task_manager: Optional[TaskManager] = None


def get_task_manager() -> TaskManager:
    """Get global task manager instance."""
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager


async def initialize_task_manager() -> TaskManager:
    """Initialize and return the global task manager."""
    manager = get_task_manager()
    await manager.initialize()
    return manager
