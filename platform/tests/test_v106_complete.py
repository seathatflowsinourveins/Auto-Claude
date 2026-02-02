#!/usr/bin/env python3
"""
Test Suite for V10.6 Complete Hooks

Tests the new features added based on deep SDK source code research:
- MCP Task Message Queue: QueuedMessage, TaskMessageQueue (FIFO queue with async wait/notify)
- MCP Resolver: Future-like object for passing results between operations
- MCP Task Helpers: Terminal status checking, task ID generation, metadata, initial task creation
- MCP Task Polling: Configurable polling with intervals until terminal status
- Letta Run Management: RunStatus, StopReasonType, RunFilter, LettaRun
- Letta Project Support: ProjectFilter, ExtendedStepFilter for cloud multi-tenant deployments

Based on:
- mcp-python/src/mcp/shared/experimental/tasks/message_queue.py
- mcp-python/src/mcp/shared/experimental/tasks/resolver.py
- mcp-python/src/mcp/shared/experimental/tasks/helpers.py
- mcp-python/src/mcp/shared/experimental/tasks/polling.py
- letta-python/src/letta_client/resources/runs/runs.py
- letta-python/src/letta_client/resources/steps/steps.py

Run with: python -m pytest tests/test_v106_complete.py -v
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add hooks directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))

from hook_utils import (
    # V10.6: MCP Task Message Queue
    QueuedMessage,
    TaskMessageQueue,
    # V10.6: MCP Resolver
    Resolver,
    # V10.6: MCP Task Helpers
    MODEL_IMMEDIATE_RESPONSE_KEY,
    RELATED_TASK_METADATA_KEY,
    is_terminal_status,
    generate_task_id,
    TaskMetadata,
    create_initial_task,
    # V10.6: MCP Task Polling
    TaskPollConfig,
    TaskPollResult,
    # V10.6: Letta Run Management
    RunStatus,
    StopReasonType,
    RunFilter,
    LettaRun,
    # V10.6: Letta Project Support
    ProjectFilter,
    ExtendedStepFilter,
    # V10.4 Dependencies
    TaskStatus,
    MCPTask,
    # V10.5 Dependencies
    StepFeedbackType,
)


class TestQueuedMessage:
    """Test QueuedMessage dataclass for queue entries."""

    def test_request_message(self):
        """Test request-type queued message."""
        msg = QueuedMessage(
            message_type="request",
            message={"method": "tools/call", "params": {"name": "test"}},
            resolver_id="resolver-123"
        )

        assert msg.message_type == "request"
        assert msg.message["method"] == "tools/call"
        assert msg.resolver_id == "resolver-123"
        assert msg.timestamp is not None

    def test_notification_message(self):
        """Test notification-type queued message."""
        msg = QueuedMessage(
            message_type="notification",
            message={"method": "notifications/progress", "params": {"progress": 0.5}}
        )

        assert msg.message_type == "notification"
        assert msg.resolver_id is None

    def test_message_with_original_request_id(self):
        """Test message tracking original request."""
        msg = QueuedMessage(
            message_type="request",
            message={"jsonrpc": "2.0", "id": 42},
            original_request_id="req-original-123"
        )

        assert msg.original_request_id == "req-original-123"

    def test_to_dict(self):
        """Test message serialization (uses camelCase per SDK)."""
        msg = QueuedMessage(
            message_type="request",
            message={"method": "test"},
            resolver_id="res-1"
        )
        result = msg.to_dict()

        assert result["messageType"] == "request"
        assert result["message"]["method"] == "test"
        assert result["resolverId"] == "res-1"
        assert "timestamp" in result


class TestTaskMessageQueue:
    """Test TaskMessageQueue for FIFO message handling."""

    def test_enqueue_dequeue(self):
        """Test basic enqueue and dequeue operations."""
        queue = TaskMessageQueue()
        msg1 = QueuedMessage("request", {"id": 1})
        msg2 = QueuedMessage("request", {"id": 2})

        queue.enqueue("task-1", msg1)
        queue.enqueue("task-1", msg2)

        result = queue.dequeue("task-1")
        assert result.message["id"] == 1  # FIFO order

        result = queue.dequeue("task-1")
        assert result.message["id"] == 2

    def test_peek(self):
        """Test peek without removing message."""
        queue = TaskMessageQueue()
        msg = QueuedMessage("request", {"test": True})

        queue.enqueue("task-1", msg)

        # Peek should return message without removing
        peeked = queue.peek("task-1")
        assert peeked.message["test"] is True

        # Message should still be in queue
        assert not queue.is_empty("task-1")

        # Dequeue should return same message
        dequeued = queue.dequeue("task-1")
        assert dequeued.message["test"] is True

    def test_is_empty(self):
        """Test empty queue detection."""
        queue = TaskMessageQueue()

        # Empty for non-existent task
        assert queue.is_empty("task-new") is True

        # Not empty after enqueue
        queue.enqueue("task-1", QueuedMessage("request", {}))
        assert queue.is_empty("task-1") is False

        # Empty after dequeue
        queue.dequeue("task-1")
        assert queue.is_empty("task-1") is True

    def test_clear(self):
        """Test clearing all messages for a task."""
        queue = TaskMessageQueue()

        queue.enqueue("task-1", QueuedMessage("request", {"id": 1}))
        queue.enqueue("task-1", QueuedMessage("request", {"id": 2}))
        queue.enqueue("task-1", QueuedMessage("request", {"id": 3}))

        # Clear should return all messages
        cleared = queue.clear("task-1")
        assert len(cleared) == 3
        assert cleared[0].message["id"] == 1

        # Queue should be empty
        assert queue.is_empty("task-1") is True

    def test_dequeue_empty(self):
        """Test dequeue from empty queue returns None."""
        queue = TaskMessageQueue()

        result = queue.dequeue("nonexistent-task")
        assert result is None

    def test_queue_isolation(self):
        """Test that different tasks have isolated queues."""
        queue = TaskMessageQueue()

        queue.enqueue("task-1", QueuedMessage("request", {"task": 1}))
        queue.enqueue("task-2", QueuedMessage("request", {"task": 2}))

        msg1 = queue.dequeue("task-1")
        msg2 = queue.dequeue("task-2")

        assert msg1.message["task"] == 1
        assert msg2.message["task"] == 2


class TestResolver:
    """Test Resolver for future-like result passing."""

    def test_set_and_get_result(self):
        """Test setting and retrieving result."""
        resolver = Resolver()

        assert resolver.done() is False

        resolver.set_result({"status": "success", "value": 42})

        assert resolver.done() is True
        result = resolver.get_result()
        assert result["status"] == "success"
        assert result["value"] == 42

    def test_set_exception(self):
        """Test setting and retrieving exception."""
        import pytest

        resolver = Resolver()
        error = ValueError("Something went wrong")

        resolver.set_exception(error)

        assert resolver.done() is True

        with pytest.raises(ValueError) as exc_info:
            resolver.get_result()
        assert "Something went wrong" in str(exc_info.value)

    def test_get_result_before_done(self):
        """Test getting result before it's set."""
        import pytest

        resolver = Resolver()

        with pytest.raises(RuntimeError) as exc_info:
            resolver.get_result()
        assert "not yet completed" in str(exc_info.value).lower()

    def test_set_result_twice(self):
        """Test that setting result twice raises error."""
        import pytest

        resolver = Resolver()
        resolver.set_result("first")

        with pytest.raises(RuntimeError) as exc_info:
            resolver.set_result("second")
        assert "already" in str(exc_info.value).lower()

    def test_reset(self):
        """Test resolver can be reset for reuse."""
        resolver = Resolver()
        resolver.set_result("first")
        assert resolver.done() is True

        resolver.reset()
        assert resolver.done() is False

        resolver.set_result("second")
        assert resolver.get_result() == "second"


class TestMCPTaskHelpers:
    """Test MCP task helper functions and classes."""

    def test_metadata_keys(self):
        """Test metadata key constants."""
        assert MODEL_IMMEDIATE_RESPONSE_KEY == "io.modelcontextprotocol/model-immediate-response"
        assert RELATED_TASK_METADATA_KEY == "io.modelcontextprotocol/related-task"

    def test_is_terminal_status_strings(self):
        """Test terminal status checking with strings."""
        # Terminal statuses
        assert is_terminal_status("completed") is True
        assert is_terminal_status("failed") is True
        assert is_terminal_status("cancelled") is True

        # Non-terminal statuses
        assert is_terminal_status("pending") is False
        assert is_terminal_status("running") is False
        assert is_terminal_status("unknown") is False

    def test_is_terminal_status_enums(self):
        """Test terminal status checking with TaskStatus enum."""
        assert is_terminal_status(TaskStatus.COMPLETED) is True
        assert is_terminal_status(TaskStatus.FAILED) is True
        assert is_terminal_status(TaskStatus.CANCELLED) is True

        assert is_terminal_status(TaskStatus.WORKING) is False
        assert is_terminal_status(TaskStatus.INPUT_REQUIRED) is False

    def test_generate_task_id(self):
        """Test task ID generation (returns UUID per MCP SDK)."""
        task_id1 = generate_task_id()
        task_id2 = generate_task_id()

        # Should be unique UUIDs
        assert task_id1 != task_id2

        # Should be valid UUID format (36 chars with dashes)
        assert len(task_id1) == 36
        assert len(task_id2) == 36
        assert task_id1.count("-") == 4

    def test_task_metadata_defaults(self):
        """Test TaskMetadata default values."""
        metadata = TaskMetadata()

        assert metadata.ttl == 60000  # 60 seconds default
        assert metadata.poll_interval == 500  # 500ms default

    def test_task_metadata_custom(self):
        """Test TaskMetadata with custom values (uses custom dict)."""
        metadata = TaskMetadata(
            ttl=120000,
            poll_interval=1000,
            custom={
                MODEL_IMMEDIATE_RESPONSE_KEY: True,
                RELATED_TASK_METADATA_KEY: "task-parent-123"
            }
        )

        assert metadata.ttl == 120000
        assert metadata.poll_interval == 1000
        assert metadata.custom[MODEL_IMMEDIATE_RESPONSE_KEY] is True
        assert metadata.custom[RELATED_TASK_METADATA_KEY] == "task-parent-123"

    def test_task_metadata_to_dict(self):
        """Test TaskMetadata serialization with metadata keys."""
        metadata = TaskMetadata(
            custom={
                MODEL_IMMEDIATE_RESPONSE_KEY: True,
                RELATED_TASK_METADATA_KEY: "task-parent"
            }
        )
        result = metadata.to_dict()

        assert result["ttl"] == 60000
        assert result["pollInterval"] == 500
        assert result[MODEL_IMMEDIATE_RESPONSE_KEY] is True
        assert result[RELATED_TASK_METADATA_KEY] == "task-parent"

    def test_create_initial_task_auto_id(self):
        """Test creating initial task with auto-generated ID."""
        metadata = TaskMetadata()
        task = create_initial_task(metadata)

        # ID is a UUID (36 chars)
        assert len(task.task_id) == 36
        assert task.status == TaskStatus.WORKING
        assert task.ttl == 60000

    def test_create_initial_task_custom_id(self):
        """Test creating initial task with custom ID."""
        metadata = TaskMetadata(ttl=30000)
        task = create_initial_task(metadata, task_id="task-custom-123")

        assert task.task_id == "task-custom-123"
        assert task.status == TaskStatus.WORKING
        assert task.ttl == 30000


class TestTaskPollConfig:
    """Test TaskPollConfig for polling configuration."""

    def test_default_config(self):
        """Test default polling configuration."""
        config = TaskPollConfig()

        assert config.default_interval_ms == 500
        assert config.max_attempts == 0  # No limit
        assert config.timeout_ms == 0  # No timeout

    def test_custom_config(self):
        """Test custom polling configuration."""
        config = TaskPollConfig(
            default_interval_ms=1000,
            max_attempts=10,
            timeout_ms=30000
        )

        assert config.default_interval_ms == 1000
        assert config.max_attempts == 10
        assert config.timeout_ms == 30000

    def test_to_dict(self):
        """Test configuration serialization (only non-zero values)."""
        config = TaskPollConfig(
            default_interval_ms=750,
            max_attempts=5,
            timeout_ms=15000
        )
        result = config.to_dict()

        assert result["defaultIntervalMs"] == 750
        assert result["maxAttempts"] == 5
        assert result["timeoutMs"] == 15000

    def test_to_dict_defaults(self):
        """Test that default config only includes interval."""
        config = TaskPollConfig()
        result = config.to_dict()

        assert result["defaultIntervalMs"] == 500
        assert "maxAttempts" not in result  # 0 is not included
        assert "timeoutMs" not in result  # 0 is not included


class TestTaskPollResult:
    """Test TaskPollResult for polling outcomes."""

    def test_successful_poll(self):
        """Test successful polling result."""
        task = MCPTask(
            task_id="task-123",
            status=TaskStatus.COMPLETED
        )
        result = TaskPollResult(
            task=task,
            attempts=3,
            elapsed_ms=1500.0,
            reached_terminal=True
        )

        assert result.task.task_id == "task-123"
        assert result.attempts == 3
        assert result.elapsed_ms == 1500.0
        assert result.reached_terminal is True
        assert result.success is True

    def test_poll_timeout(self):
        """Test polling that timed out."""
        task = MCPTask(
            task_id="task-456",
            status=TaskStatus.WORKING
        )
        result = TaskPollResult(
            task=task,
            attempts=10,
            elapsed_ms=30000.0,
            reached_terminal=False,
            timed_out=True
        )

        assert result.reached_terminal is False
        assert result.timed_out is True
        assert result.success is False

    def test_to_dict(self):
        """Test result serialization."""
        task = MCPTask(task_id="task-789", status=TaskStatus.COMPLETED)
        result = TaskPollResult(
            task=task,
            attempts=5,
            elapsed_ms=2500.0,
            reached_terminal=True
        )
        output = result.to_dict()

        assert "task" in output
        assert output["attempts"] == 5
        assert output["elapsed_ms"] == 2500.0
        assert output["reached_terminal"] is True


class TestRunStatus:
    """Test RunStatus enum for Letta run states."""

    def test_status_values(self):
        """Verify run status values match Letta SDK."""
        assert RunStatus.PENDING.value == "pending"
        assert RunStatus.RUNNING.value == "running"
        assert RunStatus.COMPLETED.value == "completed"
        assert RunStatus.FAILED.value == "failed"
        assert RunStatus.CANCELLED.value == "cancelled"

    def test_is_terminal(self):
        """Test terminal status detection."""
        assert RunStatus.COMPLETED.is_terminal() is True
        assert RunStatus.FAILED.is_terminal() is True
        assert RunStatus.CANCELLED.is_terminal() is True

        assert RunStatus.PENDING.is_terminal() is False
        assert RunStatus.RUNNING.is_terminal() is False


class TestStopReasonType:
    """Test StopReasonType enum for run completion reasons."""

    def test_stop_reason_values(self):
        """Verify stop reason values match Letta SDK."""
        assert StopReasonType.END_TURN.value == "end_turn"
        assert StopReasonType.MAX_TOKENS.value == "max_tokens"
        assert StopReasonType.TOOL_CALL.value == "tool_call"
        assert StopReasonType.ERROR.value == "error"
        assert StopReasonType.USER_INTERRUPT.value == "user_interrupt"


class TestRunFilter:
    """Test RunFilter for Letta run queries."""

    def test_filter_by_agent(self):
        """Test filtering runs by agent ID."""
        filter = RunFilter(agent_id="agent-123")
        result = filter.to_dict()

        assert result["agent_id"] == "agent-123"

    def test_filter_by_status(self):
        """Test filtering by multiple statuses."""
        filter = RunFilter(statuses=["running", "pending"])
        result = filter.to_dict()

        assert result["statuses"] == ["running", "pending"]

    def test_filter_active_runs(self):
        """Test filtering for active runs only."""
        filter = RunFilter(active=True)
        result = filter.to_dict()

        assert result["active"] is True

    def test_filter_by_conversation(self):
        """Test filtering by conversation ID."""
        filter = RunFilter(conversation_id="conv-456")
        result = filter.to_dict()

        assert result["conversation_id"] == "conv-456"

    def test_filter_background_runs(self):
        """Test filtering background vs foreground runs."""
        filter = RunFilter(background=True)
        result = filter.to_dict()

        assert result["background"] is True

    def test_filter_by_stop_reason(self):
        """Test filtering by stop reason."""
        filter = RunFilter(stop_reason=StopReasonType.TOOL_CALL)
        result = filter.to_dict()

        assert result["stop_reason"] == "tool_call"

    def test_full_filter(self):
        """Test filter with all options."""
        filter = RunFilter(
            active=False,
            agent_id="agent-789",
            conversation_id="conv-000",
            background=False,
            statuses=["completed", "failed"],
            stop_reason=StopReasonType.ERROR,
            limit=50,
            after="run-last",
            order="asc"  # Non-default to include in output
        )
        result = filter.to_dict()

        assert result["active"] is False
        assert result["agent_id"] == "agent-789"
        assert result["conversation_id"] == "conv-000"
        assert result["background"] is False
        assert result["statuses"] == ["completed", "failed"]
        assert result["stop_reason"] == "error"
        assert result["limit"] == 50
        assert result["after"] == "run-last"
        assert result["order"] == "asc"  # Default "desc" is excluded


class TestLettaRun:
    """Test LettaRun for run representation."""

    def test_basic_run(self):
        """Test basic run creation."""
        run = LettaRun(
            run_id="run-123",
            agent_id="agent-456",
            status=RunStatus.RUNNING
        )

        assert run.run_id == "run-123"
        assert run.agent_id == "agent-456"
        assert run.status == RunStatus.RUNNING

    def test_completed_run(self):
        """Test completed run with all fields."""
        run = LettaRun(
            run_id="run-789",
            agent_id="agent-000",
            status=RunStatus.COMPLETED,
            conversation_id="conv-111",
            stop_reason=StopReasonType.END_TURN,
            background=False,
            created_at=datetime(2026, 1, 17, 12, 0, 0, tzinfo=timezone.utc),
            completed_at=datetime(2026, 1, 17, 12, 1, 30, tzinfo=timezone.utc),
            metadata={"version": "1.0"}
        )

        assert run.status == RunStatus.COMPLETED
        assert run.stop_reason == StopReasonType.END_TURN
        assert run.conversation_id == "conv-111"
        assert run.metadata["version"] == "1.0"

    def test_run_to_dict(self):
        """Test run serialization."""
        run = LettaRun(
            run_id="run-abc",
            agent_id="agent-def",
            status=RunStatus.FAILED,
            stop_reason=StopReasonType.ERROR,
            background=True
        )
        result = run.to_dict()

        assert result["run_id"] == "run-abc"
        assert result["agent_id"] == "agent-def"
        assert result["status"] == "failed"
        assert result["stop_reason"] == "error"
        assert result["background"] is True

    def test_run_from_dict(self):
        """Test parsing run from API response."""
        data = {
            "id": "run-xyz",
            "agent_id": "agent-uvw",
            "status": "completed",
            "conversation_id": "conv-rst",
            "stop_reason": "end_turn",
            "background": False,
            "created_at": "2026-01-17T12:00:00Z"
        }
        run = LettaRun.from_dict(data)

        assert run.run_id == "run-xyz"
        assert run.agent_id == "agent-uvw"
        assert run.status == RunStatus.COMPLETED
        assert run.stop_reason == StopReasonType.END_TURN

    def test_run_is_active(self):
        """Test checking if run is active (property, not method)."""
        active_run = LettaRun(
            run_id="run-1",
            agent_id="agent-1",
            status=RunStatus.RUNNING
        )
        completed_run = LettaRun(
            run_id="run-2",
            agent_id="agent-2",
            status=RunStatus.COMPLETED
        )

        assert active_run.is_active is True
        assert completed_run.is_active is False
        assert active_run.is_complete is False
        assert completed_run.is_complete is True


class TestProjectFilter:
    """Test ProjectFilter for cloud multi-tenant support."""

    def test_basic_project_filter(self):
        """Test basic project filter (includes all fields)."""
        filter = ProjectFilter(project_id="proj-123")
        result = filter.to_dict()

        assert result["project_id"] == "proj-123"
        assert result["include_archived"] is False  # Default value included

    def test_project_filter_with_archived(self):
        """Test project filter including archived items."""
        filter = ProjectFilter(
            project_id="proj-456",
            include_archived=True
        )
        result = filter.to_dict()

        assert result["project_id"] == "proj-456"
        assert result["include_archived"] is True


class TestExtendedStepFilter:
    """Test ExtendedStepFilter for cloud project support."""

    def test_extended_filter_inherits(self):
        """Test that ExtendedStepFilter inherits from StepFilter."""
        filter = ExtendedStepFilter(
            agent_id="agent-123",
            model="claude-3-opus",
            project_id="proj-789"
        )
        result = filter.to_dict()

        # Inherited fields
        assert result["agent_id"] == "agent-123"
        assert result["model"] == "claude-3-opus"

        # Extended field
        assert result["project_id"] == "proj-789"

    def test_extended_filter_full(self):
        """Test full extended filter with all options."""
        filter = ExtendedStepFilter(
            agent_id="agent-000",
            model="claude-3-sonnet",
            feedback=StepFeedbackType.POSITIVE,
            has_feedback=True,
            tags=["important", "reviewed"],
            limit=100,
            order="asc",
            project_id="proj-enterprise"
        )
        result = filter.to_dict()

        assert result["agent_id"] == "agent-000"
        assert result["feedback"] == "positive"
        assert result["tags"] == ["important", "reviewed"]
        assert result["limit"] == 100
        assert result["project_id"] == "proj-enterprise"


class TestV106Integration:
    """Integration tests for V10.6 features working together."""

    def test_task_lifecycle(self):
        """Test complete task lifecycle with queue and resolver."""
        # Create task
        metadata = TaskMetadata(ttl=30000, poll_interval=100)
        task = create_initial_task(metadata, task_id="task-lifecycle-1")

        # Create message queue and resolver
        queue = TaskMessageQueue()
        resolver = Resolver()

        # Enqueue request with resolver
        request_msg = QueuedMessage(
            message_type="request",
            message={"method": "tools/call", "params": {"name": "test_tool"}},
            resolver_id="resolver-12345"  # Use string ID, not resolver attribute
        )
        queue.enqueue(task.task_id, request_msg)

        # Simulate processing
        msg = queue.dequeue(task.task_id)
        assert msg.message["method"] == "tools/call"

        # Complete task and resolve
        task.status = TaskStatus.COMPLETED
        task.result = {"success": True}
        resolver.set_result(task.result)

        # Verify
        assert is_terminal_status(task.status) is True
        assert resolver.done() is True
        assert resolver.get_result()["success"] is True

    def test_run_with_steps_filter(self):
        """Test filtering runs and their steps."""
        # Create run
        run = LettaRun(
            run_id="run-integration",
            agent_id="agent-test",
            status=RunStatus.COMPLETED,
            conversation_id="conv-main"
        )

        # Create step filter for this run's project
        step_filter = ExtendedStepFilter(
            agent_id=run.agent_id,
            project_id="proj-test",
            feedback=StepFeedbackType.POSITIVE
        )

        # Verify filter includes all necessary fields
        result = step_filter.to_dict()
        assert result["agent_id"] == "agent-test"
        assert result["project_id"] == "proj-test"
        assert result["feedback"] == "positive"

    def test_polling_with_terminal_check(self):
        """Test polling configuration with terminal status checking."""
        config = TaskPollConfig(
            default_interval_ms=100,
            max_attempts=5,
            timeout_ms=1000
        )

        # Simulate polling
        task = MCPTask(task_id="task-poll", status=TaskStatus.WORKING)
        attempts = 0
        start_time = time.time()

        # Simulate status transitions (WORKING -> INPUT_REQUIRED -> WORKING -> COMPLETED)
        statuses = [TaskStatus.WORKING, TaskStatus.INPUT_REQUIRED, TaskStatus.WORKING, TaskStatus.COMPLETED]

        for status in statuses:
            attempts += 1
            task.status = status
            if is_terminal_status(task.status):
                break

        elapsed_ms = (time.time() - start_time) * 1000

        result = TaskPollResult(
            task=task,
            attempts=attempts,
            elapsed_ms=elapsed_ms,
            reached_terminal=is_terminal_status(task.status)
        )

        assert result.reached_terminal is True
        assert result.attempts == 4  # Working, InputRequired, Working, Completed
        assert config.max_attempts == 5  # Use the config to avoid unused warning


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
