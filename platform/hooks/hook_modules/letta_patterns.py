#!/usr/bin/env python3
"""
Letta Patterns Module - Memory Block and Step Types

This module contains Letta SDK patterns for memory management.
Extracted from hook_utils.py for modular architecture.

Exports:
- MemoryBlock: Core memory block
- BlockManager: Memory block lifecycle management
- StepFeedbackType: Step feedback types
- StepFilter: Step filtering criteria
- StepMetrics: Step execution metrics
- LettaStep: Step execution data
- RunStatus: Run status enum
- StopReasonType: Stop reason types
- RunFilter: Run filtering criteria
- LettaRun: Run execution data

Version: V1.0.0 (2026-01-30) - Extracted from hook_utils.py V10.11
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


@dataclass
class MemoryBlock:
    """
    Letta memory block for core memory.

    Core memory blocks are always in context and agent-modifiable.
    Standard blocks: human, persona
    Custom blocks can be created for specific use cases.

    Per Letta SDK 1.7.x:
    - blocks.retrieve(block_label, agent_id=agent_id)
    - blocks.update(block_label, agent_id=agent_id, value=...)
    - blocks.list(agent_id) returns iterator
    """
    label: str
    value: str
    limit: int = 5000  # Character limit
    agent_id: Optional[str] = None
    block_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        data: Dict[str, Any] = {
            "label": self.label,
            "value": self.value,
            "limit": self.limit
        }
        if self.agent_id:
            data["agent_id"] = self.agent_id
        if self.block_id:
            data["id"] = self.block_id
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryBlock":
        """Create from dictionary."""
        return cls(
            label=data.get("label", ""),
            value=data.get("value", ""),
            limit=data.get("limit", 5000),
            agent_id=data.get("agent_id"),
            block_id=data.get("id")
        )

    @property
    def usage_percent(self) -> float:
        """Calculate usage percentage."""
        if self.limit <= 0:
            return 0.0
        return (len(self.value) / self.limit) * 100


class BlockManager:
    """
    Manager for Letta memory blocks.

    Provides lifecycle management for core memory blocks.
    Uses lazy initialization for Letta client.
    """

    DEFAULT_AGENT_ID = os.environ.get(
        "LETTA_UNLEASH_AGENT_ID",
        "agent-daee71d2-193b-485e-bda4-ee44752635fe"
    )

    def __init__(self, agent_id: Optional[str] = None):
        self.agent_id = agent_id or self.DEFAULT_AGENT_ID
        self._client = None

    def _get_client(self):
        """Get Letta client with lazy initialization."""
        if self._client is None:
            try:
                from letta_client import Letta
                api_key = os.environ.get("LETTA_API_KEY")
                if api_key:
                    self._client = Letta(
                        api_key=api_key,
                        base_url="https://api.letta.com"
                    )
            except ImportError:
                pass
        return self._client

    def list_blocks(self) -> List[MemoryBlock]:
        """List all memory blocks for the agent."""
        client = self._get_client()
        if not client:
            return []

        try:
            blocks = []
            for block in client.agents.blocks.list(self.agent_id):
                blocks.append(MemoryBlock(
                    label=getattr(block, 'label', ''),
                    value=getattr(block, 'value', ''),
                    limit=getattr(block, 'limit', 5000),
                    agent_id=self.agent_id,
                    block_id=getattr(block, 'id', None)
                ))
            return blocks
        except Exception:
            return []

    def get_block(self, label: str) -> Optional[MemoryBlock]:
        """Get a specific memory block by label."""
        client = self._get_client()
        if not client:
            return None

        try:
            block = client.agents.blocks.retrieve(label, agent_id=self.agent_id)
            return MemoryBlock(
                label=getattr(block, 'label', label),
                value=getattr(block, 'value', ''),
                limit=getattr(block, 'limit', 5000),
                agent_id=self.agent_id,
                block_id=getattr(block, 'id', None)
            )
        except Exception:
            return None

    def update_block(self, label: str, value: str) -> bool:
        """Update a memory block's value."""
        client = self._get_client()
        if not client:
            return False

        try:
            client.agents.blocks.update(label, agent_id=self.agent_id, value=value)
            return True
        except Exception:
            return False


class StepFeedbackType(Enum):
    """Types of step feedback."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class StepFilter:
    """
    Filter criteria for Letta steps.

    Used with client.agents.steps.list() for filtering.
    """
    agent_id: str
    limit: int = 100
    step_type: Optional[str] = None
    status: Optional[str] = None
    after: Optional[str] = None
    before: Optional[str] = None

    def to_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs for API call."""
        kwargs: Dict[str, Any] = {"limit": self.limit}
        if self.step_type:
            kwargs["step_type"] = self.step_type
        if self.status:
            kwargs["status"] = self.status
        if self.after:
            kwargs["after"] = self.after
        if self.before:
            kwargs["before"] = self.before
        return kwargs


@dataclass
class StepMetrics:
    """
    Execution metrics for a step.

    Tracks performance and resource usage.
    """
    duration_ms: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    model_name: Optional[str] = None
    tool_calls: int = 0


@dataclass
class StepTrace:
    """
    Trace data for debugging steps.

    Contains detailed execution path information.
    """
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class StepFeedback:
    """
    Feedback for a step.

    Used for learning and improvement.
    """
    feedback_type: StepFeedbackType
    comment: Optional[str] = None
    rating: Optional[float] = None


@dataclass
class LettaStep:
    """
    Letta step execution data.

    Represents a single step in agent execution.
    """
    step_id: str
    agent_id: str
    step_type: str
    status: str
    created_at: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Optional[StepMetrics] = None
    trace: Optional[StepTrace] = None
    feedback: Optional[StepFeedback] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        data: Dict[str, Any] = {
            "id": self.step_id,
            "agent_id": self.agent_id,
            "step_type": self.step_type,
            "status": self.status,
            "created_at": self.created_at,
            "messages": self.messages
        }
        if self.metrics:
            data["metrics"] = {
                "duration_ms": self.metrics.duration_ms,
                "tokens_input": self.metrics.tokens_input,
                "tokens_output": self.metrics.tokens_output,
                "model_name": self.metrics.model_name,
                "tool_calls": self.metrics.tool_calls
            }
        if self.trace:
            data["trace"] = {
                "trace_id": self.trace.trace_id,
                "span_id": self.trace.span_id,
                "parent_span_id": self.trace.parent_span_id,
                "events": self.trace.events
            }
        if self.feedback:
            data["feedback"] = {
                "type": self.feedback.feedback_type.value,
                "comment": self.feedback.comment,
                "rating": self.feedback.rating
            }
        return data


class RunStatus(Enum):
    """Letta run status values."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @classmethod
    def is_terminal(cls, status: str) -> bool:
        """Check if status is terminal."""
        return status in (cls.COMPLETED.value, cls.FAILED.value, cls.CANCELLED.value)


class StopReasonType(Enum):
    """Letta stop reason types."""
    END_TURN = "end_turn"
    TOOL_CALL = "tool_call"
    MAX_TOKENS = "max_tokens"
    ERROR = "error"
    USER_INTERRUPT = "user_interrupt"


@dataclass
class RunFilter:
    """
    Filter criteria for Letta runs.

    Used with client.agents.runs.list() for filtering.
    """
    agent_id: str
    limit: int = 100
    status: Optional[RunStatus] = None
    conversation_id: Optional[str] = None
    stop_reason: Optional[StopReasonType] = None
    is_background: Optional[bool] = None
    project_id: Optional[str] = None

    def to_kwargs(self) -> Dict[str, Any]:
        """Convert to kwargs for API call."""
        kwargs: Dict[str, Any] = {"limit": self.limit}
        if self.status:
            kwargs["status"] = self.status.value
        if self.conversation_id:
            kwargs["conversation_id"] = self.conversation_id
        if self.stop_reason:
            kwargs["stop_reason"] = self.stop_reason.value
        if self.is_background is not None:
            kwargs["is_background"] = self.is_background
        if self.project_id:
            kwargs["project_id"] = self.project_id
        return kwargs


@dataclass
class LettaRun:
    """
    Letta run execution data.

    Represents a complete agent execution run.
    """
    run_id: str
    agent_id: str
    status: RunStatus
    created_at: str
    completed_at: Optional[str] = None
    stop_reason: Optional[StopReasonType] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    steps: List[LettaStep] = field(default_factory=list)
    metrics: Optional[StepMetrics] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        data: Dict[str, Any] = {
            "id": self.run_id,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "messages": self.messages
        }
        if self.completed_at:
            data["completed_at"] = self.completed_at
        if self.stop_reason:
            data["stop_reason"] = self.stop_reason.value
        if self.steps:
            data["steps"] = [s.to_dict() for s in self.steps]
        if self.metrics:
            data["metrics"] = {
                "duration_ms": self.metrics.duration_ms,
                "tokens_input": self.metrics.tokens_input,
                "tokens_output": self.metrics.tokens_output
            }
        if self.error:
            data["error"] = self.error
        return data

    @property
    def is_complete(self) -> bool:
        """Check if run is complete."""
        return self.status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED)

    @property
    def duration_ms(self) -> Optional[float]:
        """Calculate run duration in milliseconds."""
        if not self.completed_at:
            return None
        try:
            start = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
            end = datetime.fromisoformat(self.completed_at.replace("Z", "+00:00"))
            return (end - start).total_seconds() * 1000
        except (ValueError, TypeError):
            return None


# Export all symbols
__all__ = [
    "MemoryBlock",
    "BlockManager",
    "StepFeedbackType",
    "StepFilter",
    "StepMetrics",
    "StepTrace",
    "StepFeedback",
    "LettaStep",
    "RunStatus",
    "StopReasonType",
    "RunFilter",
    "LettaRun",
]
