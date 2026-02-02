#!/usr/bin/env python3
"""
Test Suite for V10.5 Advanced Hooks

Tests the new features added based on official SDK source code research:
- MCP Error Handling: JSON-RPC error codes, ErrorData, McpError
- MCP Cancellation: CancellationNotification
- MCP Ping: PingRequest, PingResponse
- MCP Roots: MCPRoot, RootsCapability, ListRootsResult
- MCP Pagination: PaginatedRequest, PaginatedResult
- MCP Tool Annotations: ToolAnnotations with behavioral hints
- Letta Step Tracking: StepFilter, StepMetrics, StepTrace, StepFeedback, LettaStep

Run with: python -m pytest tests/test_v105_advanced.py -v
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add hooks directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))

from hook_utils import (
    # V10.5: MCP Error Handling
    MCPErrorCode,
    ErrorData,
    McpError,
    # V10.5: MCP Cancellation
    CancellationNotification,
    # V10.5: MCP Ping
    PingRequest,
    PingResponse,
    # V10.5: MCP Roots
    MCPRoot,
    RootsCapability,
    ListRootsResult,
    # V10.5: MCP Pagination
    PaginatedRequest,
    PaginatedResult,
    # V10.5: MCP Tool Annotations
    ToolAnnotations,
    # V10.5: Letta Step Tracking
    StepFeedbackType,
    StepFilter,
    StepMetrics,
    StepTrace,
    StepFeedback,
    LettaStep,
    # V10.5: Protocol Constants
    MCP_LATEST_PROTOCOL_VERSION,
    MCP_DEFAULT_NEGOTIATED_VERSION,
)


class TestMCPErrorCode:
    """Test MCPErrorCode class with JSON-RPC error codes."""

    def test_standard_error_codes(self):
        """Verify standard JSON-RPC error codes per spec."""
        assert MCPErrorCode.PARSE_ERROR == -32700
        assert MCPErrorCode.INVALID_REQUEST == -32600
        assert MCPErrorCode.METHOD_NOT_FOUND == -32601
        assert MCPErrorCode.INVALID_PARAMS == -32602
        assert MCPErrorCode.INTERNAL_ERROR == -32603

    def test_mcp_specific_error_codes(self):
        """Verify MCP-specific error codes."""
        assert MCPErrorCode.URL_ELICITATION_REQUIRED == -32042
        assert MCPErrorCode.CONNECTION_CLOSED == -32000

    def test_get_message(self):
        """Test error code to message mapping."""
        assert MCPErrorCode.get_message(MCPErrorCode.PARSE_ERROR) == "Parse error"
        assert MCPErrorCode.get_message(MCPErrorCode.INVALID_REQUEST) == "Invalid Request"
        assert MCPErrorCode.get_message(MCPErrorCode.METHOD_NOT_FOUND) == "Method not found"
        assert MCPErrorCode.get_message(-99999) == "Unknown error"


class TestErrorData:
    """Test ErrorData class for structured error information."""

    def test_error_data_to_dict(self):
        """Test ErrorData serialization."""
        data = ErrorData(
            message="Invalid parameter type",
            details={"expected": "string", "got": "integer"},
            retry_after_ms=5000
        )
        result = data.to_dict()

        assert result["message"] == "Invalid parameter type"
        assert result["details"]["expected"] == "string"
        assert result["retryAfterMs"] == 5000

    def test_empty_error_data(self):
        """Test empty ErrorData returns empty dict."""
        data = ErrorData()
        assert data.to_dict() == {}


class TestMcpError:
    """Test McpError exception class."""

    def test_basic_error(self):
        """Test basic MCP error creation."""
        error = McpError(code=MCPErrorCode.PARSE_ERROR)
        assert error.code == -32700
        assert error.message == "Parse error"
        assert str(error) == "MCP Error -32700: Parse error"

    def test_error_with_custom_message(self):
        """Test MCP error with custom message."""
        error = McpError(
            code=MCPErrorCode.INVALID_PARAMS,
            message="Missing required field 'name'"
        )
        assert error.message == "Missing required field 'name'"

    def test_error_with_data(self):
        """Test MCP error with attached data."""
        error = McpError(
            code=MCPErrorCode.INTERNAL_ERROR,
            data=ErrorData(message="Database connection failed")
        )
        result = error.to_dict()
        assert result["code"] == -32603
        assert result["data"]["message"] == "Database connection failed"

    def test_factory_methods(self):
        """Test error factory methods."""
        parse_err = McpError.parse_error("Invalid JSON at position 42")
        assert parse_err.code == MCPErrorCode.PARSE_ERROR

        invalid_req = McpError.invalid_request("Missing 'method' field")
        assert invalid_req.code == MCPErrorCode.INVALID_REQUEST

        not_found = McpError.method_not_found("tools/execute")
        assert "tools/execute" in not_found.message

        invalid_params = McpError.invalid_params("'uri' must be a string")
        assert invalid_params.code == MCPErrorCode.INVALID_PARAMS


class TestCancellationNotification:
    """Test CancellationNotification for request cancellation."""

    def test_cancellation_to_notification(self):
        """Test cancellation notification format."""
        cancel = CancellationNotification(
            request_id="req-12345",
            reason="User pressed cancel"
        )
        result = cancel.to_notification()

        assert result["method"] == "notifications/cancelled"
        assert result["params"]["requestId"] == "req-12345"
        assert result["params"]["reason"] == "User pressed cancel"

    def test_cancellation_without_reason(self):
        """Test cancellation without reason."""
        cancel = CancellationNotification(request_id="req-456")
        result = cancel.to_notification()

        assert "reason" not in result["params"]

    def test_cancellation_from_dict(self):
        """Test parsing cancellation from notification."""
        data = {
            "method": "notifications/cancelled",
            "params": {
                "requestId": "req-789",
                "reason": "Timeout exceeded"
            }
        }
        cancel = CancellationNotification.from_dict(data)
        assert cancel.request_id == "req-789"
        assert cancel.reason == "Timeout exceeded"


class TestPingRequest:
    """Test PingRequest for connection health checks."""

    def test_ping_request_format(self):
        """Test ping request format."""
        ping = PingRequest()
        result = ping.to_request()

        assert result["method"] == "ping"
        assert result["params"] == {}


class TestPingResponse:
    """Test PingResponse for ping acknowledgment."""

    def test_ping_response_empty(self):
        """Test basic ping response (empty)."""
        pong = PingResponse()
        assert pong.to_dict() == {}

    def test_ping_response_with_latency(self):
        """Test ping response with latency measurement."""
        pong = PingResponse(latency_ms=42.5)
        result = pong.to_dict()
        assert result["_latencyMs"] == 42.5


class TestMCPRoot:
    """Test MCPRoot for filesystem boundary definitions."""

    def test_valid_root(self):
        """Test valid file:// root creation."""
        root = MCPRoot(
            uri="file:///home/user/project",
            name="Project Root"
        )
        result = root.to_dict()

        assert result["uri"] == "file:///home/user/project"
        assert result["name"] == "Project Root"

    def test_root_with_meta(self):
        """Test root with metadata."""
        root = MCPRoot(
            uri="file:///var/data",
            name="Data Directory",
            meta={"version": "1.0", "readonly": True}
        )
        result = root.to_dict()
        assert result["_meta"]["version"] == "1.0"

    def test_invalid_root_uri(self):
        """Test that non-file:// URIs raise ValueError."""
        import pytest
        with pytest.raises(ValueError) as exc_info:
            MCPRoot(uri="https://example.com/files")
        assert "must start with file://" in str(exc_info.value)

    def test_root_from_path(self):
        """Test creating root from filesystem path."""
        root = MCPRoot.from_path("/home/user/code", name="Code")
        assert root.uri == "file:///home/user/code"
        assert root.name == "Code"

    def test_root_from_windows_path(self):
        """Test creating root from Windows path."""
        root = MCPRoot.from_path("C:\\Users\\test", name="Test")
        assert root.uri == "file:///C:/Users/test"


class TestRootsCapability:
    """Test RootsCapability for capability declaration."""

    def test_roots_capability_default(self):
        """Test default roots capability (no list_changed)."""
        cap = RootsCapability()
        assert cap.to_dict() == {}

    def test_roots_capability_with_list_changed(self):
        """Test roots capability with list_changed support."""
        cap = RootsCapability(list_changed=True)
        result = cap.to_dict()
        assert result["listChanged"] is True


class TestListRootsResult:
    """Test ListRootsResult for roots/list response."""

    def test_list_roots_result(self):
        """Test list roots result format."""
        roots = [
            MCPRoot(uri="file:///project1", name="Project 1"),
            MCPRoot(uri="file:///project2", name="Project 2"),
        ]
        result = ListRootsResult(roots=roots)
        output = result.to_dict()

        assert len(output["roots"]) == 2
        assert output["roots"][0]["uri"] == "file:///project1"


class TestPaginatedRequest:
    """Test PaginatedRequest for cursor-based pagination."""

    def test_paginated_request_no_cursor(self):
        """Test first request without cursor."""
        req = PaginatedRequest()
        assert req.to_dict() == {}

    def test_paginated_request_with_cursor(self):
        """Test subsequent request with cursor."""
        req = PaginatedRequest(cursor="eyJwb3MiOjEwMH0=")
        result = req.to_dict()
        assert result["cursor"] == "eyJwb3MiOjEwMH0="


class TestPaginatedResult:
    """Test PaginatedResult for pagination responses."""

    def test_paginated_result_with_next(self):
        """Test result with next cursor."""
        result = PaginatedResult(next_cursor="abc123")
        assert result.has_more is True
        assert result.to_dict()["nextCursor"] == "abc123"

    def test_paginated_result_last_page(self):
        """Test result for last page (no next cursor)."""
        result = PaginatedResult()
        assert result.has_more is False
        assert result.to_dict() == {}


class TestToolAnnotations:
    """Test ToolAnnotations for tool behavioral hints."""

    def test_full_annotations(self):
        """Test tool annotations with all hints."""
        annotations = ToolAnnotations(
            title="Delete Files",
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=False,
            open_world_hint=True
        )
        result = annotations.to_dict()

        assert result["title"] == "Delete Files"
        assert result["readOnlyHint"] is False
        assert result["destructiveHint"] is True
        assert result["idempotentHint"] is False
        assert result["openWorldHint"] is True

    def test_safe_read_factory(self):
        """Test safe read-only tool annotations factory."""
        annotations = ToolAnnotations.safe_read("List Directory")
        result = annotations.to_dict()

        assert result["title"] == "List Directory"
        assert result["readOnlyHint"] is True
        assert result["destructiveHint"] is False
        assert result["idempotentHint"] is True

    def test_dangerous_write_factory(self):
        """Test dangerous write tool annotations factory."""
        annotations = ToolAnnotations.dangerous_write("Drop Table")
        result = annotations.to_dict()

        assert result["readOnlyHint"] is False
        assert result["destructiveHint"] is True
        assert result["idempotentHint"] is False

    def test_from_dict(self):
        """Test parsing annotations from dict."""
        data = {
            "title": "Read File",
            "readOnlyHint": True,
            "destructiveHint": False
        }
        annotations = ToolAnnotations.from_dict(data)
        assert annotations.title == "Read File"
        assert annotations.read_only_hint is True


class TestStepFeedbackType:
    """Test StepFeedbackType enum."""

    def test_feedback_values(self):
        """Verify feedback type values."""
        assert StepFeedbackType.POSITIVE.value == "positive"
        assert StepFeedbackType.NEGATIVE.value == "negative"


class TestStepFilter:
    """Test StepFilter for Letta step queries."""

    def test_step_filter_full(self):
        """Test full step filter with all options."""
        filter = StepFilter(
            agent_id="agent-123",
            model="claude-3-opus",
            feedback=StepFeedbackType.POSITIVE,
            has_feedback=True,
            tags=["important", "reviewed"],
            trace_ids=["trace-1", "trace-2"],
            start_date="2026-01-01T00:00:00Z",
            end_date="2026-01-31T23:59:59Z",
            limit=50,
            after="step-100",
            order="asc"
        )
        result = filter.to_dict()

        assert result["agent_id"] == "agent-123"
        assert result["model"] == "claude-3-opus"
        assert result["feedback"] == "positive"
        assert result["has_feedback"] is True
        assert result["tags"] == ["important", "reviewed"]
        assert result["limit"] == 50
        assert result["order"] == "asc"

    def test_step_filter_minimal(self):
        """Test minimal step filter (defaults)."""
        filter = StepFilter()
        result = filter.to_dict()
        assert result == {}  # All defaults excluded


class TestStepMetrics:
    """Test StepMetrics for execution metrics."""

    def test_step_metrics_full(self):
        """Test full step metrics."""
        metrics = StepMetrics(
            step_id="step-123",
            latency_ms=1250.5,
            input_tokens=500,
            output_tokens=750,
            total_tokens=1250,
            cost_usd=0.0125,
            model="claude-3-opus"
        )
        result = metrics.to_dict()

        assert result["step_id"] == "step-123"
        assert result["latency_ms"] == 1250.5
        assert result["input_tokens"] == 500
        assert result["output_tokens"] == 750
        assert result["total_tokens"] == 1250
        assert result["cost_usd"] == 0.0125


class TestStepTrace:
    """Test StepTrace for execution tracing."""

    def test_step_trace_full(self):
        """Test full step trace."""
        trace = StepTrace(
            step_id="step-123",
            trace_id="trace-456",
            parent_trace_id="trace-000",
            span_id="span-789",
            events=[{"type": "llm_call", "duration_ms": 500}],
            metadata={"version": "1.0"}
        )
        result = trace.to_dict()

        assert result["step_id"] == "step-123"
        assert result["trace_id"] == "trace-456"
        assert result["parent_trace_id"] == "trace-000"
        assert len(result["events"]) == 1


class TestStepFeedback:
    """Test StepFeedback for step rating."""

    def test_step_feedback(self):
        """Test step feedback creation."""
        feedback = StepFeedback(
            step_id="step-123",
            feedback_type=StepFeedbackType.POSITIVE,
            comment="Great response!",
            metadata={"reviewer": "user-42"}
        )
        result = feedback.to_dict()

        assert result["step_id"] == "step-123"
        assert result["feedback"] == "positive"
        assert result["comment"] == "Great response!"


class TestLettaStep:
    """Test LettaStep for full step representation."""

    def test_letta_step_full(self):
        """Test full Letta step creation."""
        step = LettaStep(
            step_id="step-123",
            agent_id="agent-456",
            created_at=datetime(2026, 1, 17, 12, 0, 0, tzinfo=timezone.utc),
            model="claude-3-opus",
            input_messages=[{"role": "user", "content": "Hello"}],
            output_messages=[{"role": "assistant", "content": "Hi there!"}],
            tags=["greeting", "test"]
        )
        result = step.to_dict()

        assert result["step_id"] == "step-123"
        assert result["agent_id"] == "agent-456"
        assert result["model"] == "claude-3-opus"
        assert len(result["input_messages"]) == 1
        assert len(result["output_messages"]) == 1

    def test_letta_step_from_dict(self):
        """Test parsing Letta step from API response."""
        data = {
            "id": "step-789",
            "agent_id": "agent-000",
            "created_at": "2026-01-17T12:00:00Z",
            "model": "claude-3-sonnet",
            "tags": ["parsed"]
        }
        step = LettaStep.from_dict(data)
        assert step.step_id == "step-789"
        assert step.model == "claude-3-sonnet"


class TestProtocolConstants:
    """Test MCP protocol version constants."""

    def test_latest_protocol_version(self):
        """Verify latest protocol version."""
        assert MCP_LATEST_PROTOCOL_VERSION == "2025-11-25"

    def test_default_negotiated_version(self):
        """Verify default negotiated version."""
        assert MCP_DEFAULT_NEGOTIATED_VERSION == "2025-03-26"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
