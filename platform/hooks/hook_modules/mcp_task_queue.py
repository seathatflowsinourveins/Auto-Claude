#!/usr/bin/env python3
"""
MCP Task Queue Module - Async Task Message Queue

This module contains MCP task queue patterns for async operations.
Extracted from hook_utils.py for modular architecture.

Exports:
- QueuedMessage: Message in task queue
- TaskMessageQueue: FIFO queue for task messages
- Resolver: Simple future-like result passing
- generate_task_id: UUID task ID generation
- is_terminal_status: Check terminal task status

Version: V1.0.0 (2026-01-30) - Extracted from hook_utils.py V10.11
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union


# Protocol version constants
MCP_LATEST_PROTOCOL_VERSION = "2025-11-25"
MCP_DEFAULT_NEGOTIATED_VERSION = "2025-03-26"

# Metadata keys per MCP spec
MODEL_IMMEDIATE_RESPONSE_KEY = "io.modelcontextprotocol/model-immediate-response"
RELATED_TASK_METADATA_KEY = "io.modelcontextprotocol/related-task"


@dataclass
class QueuedMessage:
    """
    A message queued for delivery via tasks/result.

    Per MCP Tasks spec, messages are stored with their type and a resolver
    for requests that expect responses. This enables bidirectional communication
    through the tasks/result endpoint.

    Attributes:
        message_type: "request" (expects response) or "notification" (one-way)
        message: The JSON-RPC message content
        timestamp: When the message was enqueued
        resolver_id: Optional ID for resolving responses
        original_request_id: Original request ID for routing responses
    """
    message_type: str  # "request" or "notification"
    message: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolver_id: Optional[str] = None
    original_request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable format."""
        result: Dict[str, Any] = {
            "messageType": self.message_type,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.resolver_id:
            result["resolverId"] = self.resolver_id
        if self.original_request_id:
            result["originalRequestId"] = self.original_request_id
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueuedMessage":
        """Parse from dictionary."""
        ts = data.get("timestamp")
        return cls(
            message_type=data.get("messageType", "notification"),
            message=data.get("message", {}),
            timestamp=datetime.fromisoformat(ts) if ts else datetime.now(timezone.utc),
            resolver_id=data.get("resolverId"),
            original_request_id=data.get("originalRequestId"),
        )

    @classmethod
    def request(cls, method: str, params: Dict[str, Any], request_id: str) -> "QueuedMessage":
        """Create a request message."""
        return cls(
            message_type="request",
            message={"method": method, "params": params, "id": request_id},
            original_request_id=request_id,
        )

    @classmethod
    def notification(cls, method: str, params: Dict[str, Any]) -> "QueuedMessage":
        """Create a notification message."""
        return cls(
            message_type="notification",
            message={"method": method, "params": params},
        )


class TaskMessageQueue:
    """
    In-memory FIFO queue for task-related messages.

    Per MCP Python SDK pattern, this enables:
    1. Decoupling request handling from message delivery
    2. Proper bidirectional communication via tasks/result stream
    3. Automatic status management (working <-> input_required)

    For distributed systems, implement with Redis, RabbitMQ, etc.

    Example:
        queue = TaskMessageQueue()
        queue.enqueue("task-123", QueuedMessage.notification("progress", {"value": 50}))
        msg = queue.dequeue("task-123")
    """

    def __init__(self) -> None:
        self._queues: Dict[str, List[QueuedMessage]] = {}
        self._waiting: Dict[str, bool] = {}  # Simple flag for waiting state

    def enqueue(self, task_id: str, message: QueuedMessage) -> None:
        """Add a message to the queue for a task."""
        if task_id not in self._queues:
            self._queues[task_id] = []
        self._queues[task_id].append(message)
        # Mark that messages are available
        self._waiting[task_id] = True

    def dequeue(self, task_id: str) -> Optional[QueuedMessage]:
        """Remove and return the next message from the queue."""
        queue = self._queues.get(task_id, [])
        if not queue:
            return None
        return queue.pop(0)

    def peek(self, task_id: str) -> Optional[QueuedMessage]:
        """Return the next message without removing it."""
        queue = self._queues.get(task_id, [])
        return queue[0] if queue else None

    def is_empty(self, task_id: str) -> bool:
        """Check if the queue is empty for a task."""
        return len(self._queues.get(task_id, [])) == 0

    def clear(self, task_id: str) -> List[QueuedMessage]:
        """Remove and return all messages from the queue."""
        messages = self._queues.pop(task_id, [])
        self._waiting.pop(task_id, None)
        return messages

    def has_messages(self, task_id: str) -> bool:
        """Check if messages are available."""
        return self._waiting.get(task_id, False) and not self.is_empty(task_id)

    def get_queue_size(self, task_id: str) -> int:
        """Get the number of messages in the queue."""
        return len(self._queues.get(task_id, []))

    def cleanup_all(self) -> None:
        """Clean up all queues."""
        self._queues.clear()
        self._waiting.clear()

    def get_all_task_ids(self) -> List[str]:
        """Get all task IDs with queued messages."""
        return list(self._queues.keys())


class Resolver:
    """
    A simple resolver for passing results between operations.

    Per MCP Python SDK pattern, this works like asyncio.Future but
    is designed for synchronous/polling use cases. It provides a way
    to pass a result (or exception) from one operation to another.

    Usage:
        resolver = Resolver()
        # In one operation:
        resolver.set_result("hello")
        # In another operation:
        result = resolver.get_result()  # returns "hello"
    """

    def __init__(self) -> None:
        self._value: Any = None
        self._exception: Optional[Exception] = None
        self._completed: bool = False

    def set_result(self, value: Any) -> None:
        """Set the result value and mark as completed."""
        if self._completed:
            raise RuntimeError("Resolver already completed")
        self._value = value
        self._completed = True

    def set_exception(self, exc: Exception) -> None:
        """Set an exception and mark as completed."""
        if self._completed:
            raise RuntimeError("Resolver already completed")
        self._exception = exc
        self._completed = True

    def get_result(self) -> Any:
        """Get the result, or raise the exception if one was set."""
        if not self._completed:
            raise RuntimeError("Resolver not yet completed")
        if self._exception is not None:
            raise self._exception
        return self._value

    def done(self) -> bool:
        """Return True if the resolver has been completed."""
        return self._completed

    def has_exception(self) -> bool:
        """Return True if an exception was set."""
        return self._exception is not None

    def reset(self) -> None:
        """Reset the resolver for reuse."""
        self._value = None
        self._exception = None
        self._completed = False


def is_terminal_status(status: Union[str, Any]) -> bool:
    """
    Check if a task status represents a terminal state.

    Terminal states are those where the task has finished and will not change.
    Per MCP spec: completed, failed, or cancelled.

    Args:
        status: The task status to check (string or enum with .value)

    Returns:
        True if the status is terminal
    """
    if isinstance(status, str):
        status_val = status
    elif hasattr(status, "value"):
        status_val = getattr(status, "value")
    else:
        status_val = str(status)
    return status_val in ("completed", "failed", "cancelled")


def generate_task_id() -> str:
    """Generate a unique task ID using UUID4."""
    return str(uuid.uuid4())


# Export all symbols
__all__ = [
    "MCP_LATEST_PROTOCOL_VERSION",
    "MCP_DEFAULT_NEGOTIATED_VERSION",
    "MODEL_IMMEDIATE_RESPONSE_KEY",
    "RELATED_TASK_METADATA_KEY",
    "QueuedMessage",
    "TaskMessageQueue",
    "Resolver",
    "is_terminal_status",
    "generate_task_id",
]
