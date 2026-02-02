#!/usr/bin/env python3
"""
MCP Tasks Module - Async Task Execution Patterns

This module contains MCP task patterns for long-running async operations.
Extracted from hook_utils.py for modular architecture.

Exports:
- TaskStatus: Task status values
- TaskSupport: Tool-level task support negotiation
- MCPTask: Async task for long-running operations
- TaskRequest: Request for creating tasks
- TaskResult: Result of completed tasks

Version: V1.0.0 (2026-01-30) - Extracted from hook_utils.py V10.11
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskStatus(Enum):
    """
    MCP task status values for async operations.

    Per MCP 2025-11-25 specification:
    - WORKING: Task is actively being processed
    - INPUT_REQUIRED: Task needs additional input (human-in-the-loop)
    - COMPLETED: Task finished successfully
    - FAILED: Task encountered an error
    - CANCELLED: Task was cancelled
    """
    WORKING = "working"
    INPUT_REQUIRED = "input_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskSupport(Enum):
    """
    Tool-level task support negotiation.

    Per MCP 2025-11-25:
    - REQUIRED: Tool always returns async tasks
    - OPTIONAL: Tool may return tasks based on request
    - FORBIDDEN: Tool never returns tasks (default)
    """
    REQUIRED = "required"
    OPTIONAL = "optional"
    FORBIDDEN = "forbidden"


@dataclass
class MCPTask:
    """
    MCP async task for long-running operations.

    Per MCP 2025-11-25, tasks support:
    - Status lifecycle (working → completed/failed/cancelled)
    - TTL-based expiration
    - Polling with recommended intervals
    - Input requirement for human-in-the-loop patterns

    Example:
        task = MCPTask(
            task_id="786512e2-9e0d-44bd-8f29-789f320fe840",
            status=TaskStatus.WORKING,
            poll_interval=5000,
            ttl=60000
        )
    """
    task_id: str
    status: TaskStatus
    created_at: Optional[datetime] = None
    last_updated_at: Optional[datetime] = None
    ttl: int = 60000
    poll_interval: Optional[int] = None
    message: Optional[str] = None
    input_request: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP task response format."""
        result: Dict[str, Any] = {
            "taskId": self.task_id,
            "status": self.status.value,
            "ttl": self.ttl,
        }

        if self.created_at:
            result["createdAt"] = self.created_at.isoformat()
        if self.last_updated_at:
            result["lastUpdatedAt"] = self.last_updated_at.isoformat()
        if self.poll_interval:
            result["pollInterval"] = self.poll_interval
        if self.message:
            result["message"] = self.message
        if self.input_request:
            result["inputRequest"] = self.input_request

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPTask":
        """Parse task from MCP response."""
        created = data.get("createdAt")
        updated = data.get("lastUpdatedAt")

        return cls(
            task_id=data.get("taskId", ""),
            status=TaskStatus(data.get("status", "working")),
            created_at=datetime.fromisoformat(created.replace("Z", "+00:00")) if created else None,
            last_updated_at=datetime.fromisoformat(updated.replace("Z", "+00:00")) if updated else None,
            ttl=data.get("ttl", 60000),
            poll_interval=data.get("pollInterval"),
            message=data.get("message"),
            input_request=data.get("inputRequest"),
        )

    @property
    def is_complete(self) -> bool:
        """Check if task reached terminal state."""
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)

    @property
    def needs_input(self) -> bool:
        """Check if task requires additional input."""
        return self.status == TaskStatus.INPUT_REQUIRED

    @property
    def is_working(self) -> bool:
        """Check if task is still processing."""
        return self.status == TaskStatus.WORKING


@dataclass
class TaskRequest:
    """
    Request for creating an MCP async task.

    Per MCP 2025-11-25, include task params in tools/call:
        {
            "method": "tools/call",
            "params": {
                "name": "long_operation",
                "arguments": {...},
                "task": {"ttl": 60000}
            }
        }
    """
    ttl: int = 60000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to task request params."""
        return {"ttl": self.ttl}


@dataclass
class TaskResult:
    """
    Result of a completed MCP task.

    Per MCP 2025-11-25, get results via tasks/result:
        {
            "method": "tasks/result",
            "params": {"taskId": "..."}
        }
    """
    task_id: str
    content: List[Dict[str, Any]] = field(default_factory=list)
    structured_content: Optional[Dict[str, Any]] = None
    is_error: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP task result format."""
        result: Dict[str, Any] = {
            "taskId": self.task_id,
            "content": self.content,
            "isError": self.is_error,
        }
        if self.structured_content:
            result["structuredContent"] = self.structured_content
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskResult":
        """Parse result from MCP response."""
        return cls(
            task_id=data.get("taskId", ""),
            content=data.get("content", []),
            structured_content=data.get("structuredContent"),
            is_error=data.get("isError", False)
        )


# Export all symbols
__all__ = [
    "TaskStatus",
    "TaskSupport",
    "MCPTask",
    "TaskRequest",
    "TaskResult",
]
