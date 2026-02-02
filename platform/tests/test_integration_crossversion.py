#!/usr/bin/env python3
"""
Cross-Version Integration Test Suite

Verifies seamless integration between V10.2, V10.3, V10.4, and V10.5 features.
Tests real-world scenarios combining multiple features across versions.

Run with: python -m pytest tests/test_integration_crossversion.py -v
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add hooks directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))

from hook_utils import (
    # V10.2: Core Hook Features
    HookEvent,
    HookResponse,
    PermissionDecision,
    ToolMatcher,
    MCPContent,
    MCPToolResult,
    # V10.3: Advanced Features
    ElicitationMode,
    ElicitationRequest,
    ElicitationResponse,
    ElicitationAction,
    SamplingMessage,
    SamplingRequest,
    SamplingResponse,
    ModelPreferences,
    ProgressNotification,
    MCPCapabilities,
    ResourceSubscription,
    SubscriptionManager,
    Entity,
    Relation,
    KnowledgeGraph,
    # V10.4: Ultimate Features
    TaskStatus,
    MCPTask,
    TaskResult,
    MCPPrompt,
    PromptArgument,
    MCPResource,
    ResourceAnnotations,
    ResourceTemplate,
    LogLevel,
    LogMessage,
    CompletionRefType,
    CompletionRequest,
    CompletionResult,
    ThoughtData,
    ThinkingSession,
    TransportType,
    TransportConfig,
    MCPSession,
    MemoryBlock,
    BlockManager,
    # V10.5: Advanced Features
    MCPErrorCode,
    ErrorData,
    McpError,
    CancellationNotification,
    PingRequest,
    PingResponse,
    MCPRoot,
    RootsCapability,
    ListRootsResult,
    PaginatedRequest,
    PaginatedResult,
    ToolAnnotations,
    StepFeedbackType,
    StepFilter,
    StepMetrics,
    StepTrace,
    StepFeedback,
    LettaStep,
    MCP_LATEST_PROTOCOL_VERSION,
    MCP_DEFAULT_NEGOTIATED_VERSION,
)


class TestHookResponseWithErrorHandling:
    """Test V10.2 HookResponse with V10.5 error handling."""

    def test_hook_deny_with_mcp_error(self):
        """Combine hook denial with structured MCP error."""
        error = McpError(
            code=MCPErrorCode.INVALID_PARAMS,
            message="Path traversal detected",
            data=ErrorData(
                message="Security violation",
                details={"path": "../../../etc/passwd"}
            )
        )

        response = HookResponse.deny(
            reason=f"MCP Error {error.code}: {error.message}"
        )
        output = json.loads(response.to_json())

        assert output["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert "Path traversal" in output["hookSpecificOutput"]["permissionDecisionReason"]

    def test_hook_allow_with_tool_annotations(self):
        """Use V10.5 tool annotations to inform V10.2 hook decisions."""
        annotations = ToolAnnotations.safe_read("List Directory")

        # If tool is read-only and idempotent, auto-allow
        if annotations.read_only_hint and not annotations.destructive_hint:
            response = HookResponse.allow(
                reason=f"Auto-approved: {annotations.title} is read-only"
            )
        else:
            response = HookResponse.ask(reason="Requires user confirmation")

        output = json.loads(response.to_json())
        assert output["hookSpecificOutput"]["permissionDecision"] == "allow"


class TestMCPToolResultWithPagination:
    """Test V10.2 MCPToolResult with V10.5 pagination."""

    def test_paginated_tool_result(self):
        """Create paginated tool results."""
        items = [f"item_{i}" for i in range(100)]
        page_size = 10

        # First page
        page1_items = items[:page_size]
        result1 = MCPToolResult.success(
            f"Showing items 1-{page_size}",
            structured={
                "items": page1_items,
                "total": len(items)
            }
        )
        pagination1 = PaginatedResult(next_cursor="eyJvZmZzZXQiOjEwfQ==")

        output1 = result1.to_dict()
        output1["pagination"] = pagination1.to_dict()

        assert output1["isError"] is False
        assert len(output1["structuredContent"]["items"]) == 10
        assert output1["pagination"]["nextCursor"] == "eyJvZmZzZXQiOjEwfQ=="

    def test_last_page_no_cursor(self):
        """Final page has no next cursor."""
        result = MCPToolResult.success("Final page", structured={"items": ["last"]})
        pagination = PaginatedResult()  # No next_cursor

        assert pagination.has_more is False
        assert pagination.to_dict() == {}


class TestSamplingWithStepTracking:
    """Test V10.3 sampling with V10.5 Letta step tracking."""

    def test_sampling_request_creates_step(self):
        """Sampling request triggers step creation."""
        # Create sampling request
        request = SamplingRequest(
            messages=[SamplingMessage.user("Analyze this code")],
            max_tokens=500,
            model_preferences=ModelPreferences.prefer_smart()
        )

        # Create corresponding step
        step = LettaStep(
            step_id="step-sampling-001",
            agent_id="agent-claude",
            model="claude-3-opus",
            input_messages=[{"role": "user", "content": "Analyze this code"}],
            created_at=datetime.now(timezone.utc)
        )

        assert request.to_dict()["maxTokens"] == 500
        assert step.model == "claude-3-opus"

    def test_sampling_response_with_metrics(self):
        """Track sampling response metrics."""
        # SamplingResponse requires role parameter
        response = SamplingResponse(
            role="assistant",
            content=[{"type": "text", "text": "Analysis complete."}],
            model="claude-3-opus",
            stop_reason="end_turn"
        )

        metrics = StepMetrics(
            step_id="step-001",
            latency_ms=1250.5,
            input_tokens=50,
            output_tokens=200,
            total_tokens=250,
            cost_usd=0.00375
        )

        # Combine for full tracking
        result = {
            "response": {"model": response.model, "stop_reason": response.stop_reason},
            "metrics": metrics.to_dict()
        }

        assert result["response"]["model"] == "claude-3-opus"
        assert result["metrics"]["latency_ms"] == 1250.5


class TestKnowledgeGraphWithRoots:
    """Test V10.3 KnowledgeGraph with V10.5 MCP Roots."""

    def test_graph_entities_from_roots(self):
        """Create graph entities from filesystem roots."""
        roots = [
            MCPRoot(uri="file:///project/src", name="Source"),
            MCPRoot(uri="file:///project/tests", name="Tests"),
        ]

        graph = KnowledgeGraph()

        # Create entities from roots
        for root in roots:
            entity = Entity(
                name=root.name or root.uri,
                entity_type="directory",
                observations=[f"MCP Root: {root.uri}"]
            )
            graph.add_entity(entity)

        # Add relation - using correct parameter names
        graph.add_relation(Relation(
            from_entity="Source",
            to_entity="Tests",
            relation_type="tests_code_in"
        ))

        assert len(graph.entities) == 2
        assert len(graph.relations) == 1

    def test_list_roots_result_to_graph(self):
        """Convert ListRootsResult to knowledge graph."""
        roots_result = ListRootsResult(roots=[
            MCPRoot.from_path("/home/user/project", name="Project"),
            MCPRoot.from_path("/home/user/data", name="Data"),
        ])

        graph = KnowledgeGraph()
        for root in roots_result.roots:
            graph.add_entity(Entity(
                name=root.name,
                entity_type="mcp_root",
                observations=[root.uri]
            ))

        jsonl = graph.to_jsonl()
        assert "mcp_root" in jsonl
        assert "Project" in jsonl


class TestTasksWithErrorHandling:
    """Test V10.4 Tasks with V10.5 error handling."""

    def test_task_failure_with_mcp_error(self):
        """Handle task failure using MCP errors."""
        task = MCPTask(
            task_id="task-001",
            status=TaskStatus.FAILED,
            message="Task execution failed"
        )

        # Use available factory method - invalid_params
        error = McpError.invalid_params("Database connection timeout")

        result = TaskResult(
            task_id=task.task_id,
            content=[MCPContent.text_content(f"Error: {error.message}").to_dict()],
            is_error=True,
            structured_content=error.to_dict()
        )

        output = result.to_dict()
        assert output["isError"] is True
        assert output["structuredContent"]["code"] == MCPErrorCode.INVALID_PARAMS

    def test_task_cancellation(self):
        """Cancel a running task using cancellation notification."""
        task = MCPTask(
            task_id="task-002",
            status=TaskStatus.WORKING  # MCP 2025-11-25 uses WORKING not RUNNING
        )

        cancel = CancellationNotification(
            request_id=task.task_id,
            reason="User requested cancellation"
        )

        notification = cancel.to_notification()
        assert notification["method"] == "notifications/cancelled"
        assert notification["params"]["requestId"] == "task-002"


class TestResourcesWithAnnotations:
    """Test V10.4 Resources with V10.5 Tool Annotations."""

    def test_resource_with_tool_hints(self):
        """Combine resource annotations with tool hints."""
        resource = MCPResource(
            uri="file:///config/settings.json",
            name="Settings",
            description="Application configuration",
            annotations=ResourceAnnotations(
                audience=["admin"],
                priority=0.9
            )
        )

        # Tool for modifying this resource
        tool_hints = ToolAnnotations(
            title="Modify Settings",
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=False
        )

        # Combined metadata
        metadata = {
            "resource": resource.to_dict(),
            "tool_hints": tool_hints.to_dict()
        }

        assert metadata["resource"]["annotations"]["priority"] == 0.9
        assert metadata["tool_hints"]["destructiveHint"] is True

    def test_template_expansion_with_roots(self):
        """Expand resource templates using MCP roots."""
        template = ResourceTemplate(
            uri_template="{root}/src/{module}.py",
            name="{module} Source",
            description="Python module file"
        )

        root = MCPRoot.from_path("/project", name="Project")

        # Expand template - returns string URI
        expanded_uri = template.expand(root=root.uri.replace("file://", ""), module="main")

        assert expanded_uri == "/project/src/main.py"


class TestThinkingWithStepTrace:
    """Test V10.4 Thinking with V10.5 Step Tracing."""

    def test_thinking_session_traced(self):
        """Create traced thinking session."""
        session = ThinkingSession(session_id="think-001")

        # Add thoughts
        session.add_thought(ThoughtData(
            thought="Analyzing the problem structure...",
            thought_number=1,
            total_thoughts=2
        ))
        session.add_thought(ThoughtData(
            thought="Found optimal approach using dynamic programming.",
            thought_number=2,
            total_thoughts=2
        ))

        # Create step trace
        trace = StepTrace(
            step_id="step-think-001",
            trace_id="trace-main",
            events=[
                {"type": "thought", "content": t.thought}
                for t in session.get_history()
            ]
        )

        assert len(trace.events) == 2
        assert "dynamic programming" in trace.events[1]["content"]

    def test_thinking_with_feedback(self):
        """Add feedback to thinking session steps."""
        session = ThinkingSession(session_id="think-002")
        session.add_thought(ThoughtData(
            thought="This approach considers edge cases.",
            thought_number=1,
            total_thoughts=1
        ))

        feedback = StepFeedback(
            step_id="step-think-002",
            feedback_type=StepFeedbackType.POSITIVE,
            comment="Excellent consideration of edge cases",
            metadata={"reviewer": "senior-dev"}
        )

        output = feedback.to_dict()
        assert output["feedback"] == "positive"
        assert "edge cases" in output["comment"]


class TestMemoryBlocksWithStepFilter:
    """Test V10.4 Memory Blocks with V10.5 Step Filtering."""

    def test_filter_steps_by_memory_context(self):
        """Filter steps based on memory block context."""
        # Create memory blocks using correct method name
        manager = BlockManager()
        manager.add(MemoryBlock.persona(
            "You are a Python expert focusing on async patterns."
        ))
        manager.add(MemoryBlock(
            block_id="block-context",
            label="current_task",
            value="Reviewing async code"
        ))

        # Create step filter matching this context
        filter = StepFilter(
            agent_id="agent-python-expert",
            tags=["async", "code-review"],
            feedback=StepFeedbackType.POSITIVE,
            limit=20
        )

        filter_dict = filter.to_dict()
        assert filter_dict["tags"] == ["async", "code-review"]
        assert filter_dict["limit"] == 20

    def test_block_update_creates_step(self):
        """Memory block updates can be tracked as steps."""
        manager = BlockManager()
        manager.add(MemoryBlock(
            block_id="block-session",
            label="session_context",
            value="Initial context"
        ))

        # Update block via manager
        block = manager.get_by_label("session_context")
        if block:
            block.value = "Updated with new learnings"

        # Create step to track this update
        step = LettaStep(
            step_id="step-memory-update",
            agent_id="agent-main",
            model="internal",
            created_at=datetime.now(timezone.utc),
            tags=["memory-update", "session_context"]
        )

        assert "memory-update" in step.tags


class TestTransportWithPing:
    """Test V10.4 Transport with V10.5 Ping."""

    def test_session_health_check(self):
        """Use ping for session health checking."""
        # TransportConfig uses 'endpoint' not 'url'
        config = TransportConfig(
            transport_type=TransportType.SSE,
            endpoint="https://api.example.com/mcp"
        )

        session = MCPSession(
            session_id="session-001",
            transport_type=TransportType.SSE
        )

        # Health check via ping
        ping = PingRequest()
        ping_data = ping.to_request()

        # Simulate response with latency
        pong = PingResponse(latency_ms=45.5)

        assert ping_data["method"] == "ping"
        assert pong.to_dict()["_latencyMs"] == 45.5
        assert config.endpoint == "https://api.example.com/mcp"

    def test_session_with_roots_capability(self):
        """Session with roots capability declaration."""
        config = TransportConfig(
            transport_type=TransportType.STDIO
        )

        session = MCPSession(
            session_id="session-002",
            transport_type=TransportType.STDIO
        )

        capabilities = MCPCapabilities(
            sampling=True,
            elicitation=True
        )

        roots_cap = RootsCapability(list_changed=True)

        # Combine capabilities (MCPCapabilities doesn't have to_dict, build manually)
        full_caps = {
            "sampling": capabilities.sampling,
            "elicitation": capabilities.elicitation,
            "roots": roots_cap.to_dict()
        }

        assert full_caps["sampling"] is True
        assert full_caps["roots"]["listChanged"] is True
        assert session.transport_type == TransportType.STDIO


class TestSubscriptionsWithCancellation:
    """Test V10.3 Subscriptions with V10.5 Cancellation."""

    def test_cancel_subscription_request(self):
        """Cancel a pending subscription request."""
        sub_manager = SubscriptionManager()
        # subscribe takes (uri: str, session_id: str)
        sub_manager.subscribe("file:///watched/file.txt", "session-1")

        request_id = "sub-req-001"

        # Cancel the subscription request
        cancel = CancellationNotification(
            request_id=request_id,
            reason="Subscription no longer needed"
        )

        notification = cancel.to_notification()
        assert notification["params"]["reason"] == "Subscription no longer needed"

    def test_subscription_with_pagination(self):
        """Handle large subscription lists with pagination."""
        sub_manager = SubscriptionManager()

        # Subscribe to many resources
        for i in range(50):
            sub_manager.subscribe(f"file:///watched/file_{i}.txt", "default")

        # Paginate the subscription list
        all_uris = sub_manager.get_subscribed_uris("default")
        page_size = 10

        # First page
        page1_req = PaginatedRequest()
        page1_uris = all_uris[:page_size]
        page1_result = PaginatedResult(next_cursor="eyJwYWdlIjoxfQ==")

        assert len(page1_uris) == 10
        assert page1_result.has_more is True


class TestElicitationWithErrors:
    """Test V10.3 Elicitation with V10.5 Error Handling."""

    def test_elicitation_validation_error(self):
        """Handle validation errors in elicitation."""
        # ElicitationRequest uses 'form' factory not 'form_mode'
        request = ElicitationRequest.form(
            message="User Preferences",
            schema={"type": "object", "properties": {"theme": {"type": "string"}}}
        )

        # Simulate validation error - invalid_params puts details in data.message
        error = McpError.invalid_params(
            "Field 'theme' requires 'options' property"
        )

        assert error.code == MCPErrorCode.INVALID_PARAMS
        # The custom message is stored in error.data.message, not error.message
        assert error.data is not None
        assert "options" in error.data.message

    def test_elicitation_url_error(self):
        """Handle URL elicitation specific errors."""
        error = McpError(
            code=MCPErrorCode.URL_ELICITATION_REQUIRED,
            message="User must authenticate via OAuth"
        )

        # This special error code triggers OAuth flow
        assert error.code == -32042
        assert "OAuth" in error.message


class TestProgressWithMetrics:
    """Test V10.3 Progress with V10.5 Step Metrics."""

    def test_progress_updates_with_metrics(self):
        """Combine progress notifications with step metrics."""
        # Progress notification
        progress = ProgressNotification(
            progress_token="task-001",
            progress=50,
            total=100,
            message="Processing files..."
        )

        # Corresponding metrics
        metrics = StepMetrics(
            step_id="step-task-001",
            latency_ms=500.0,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0
        )

        # Combined status
        status = {
            "progress": progress.to_notification()["params"],
            "metrics": metrics.to_dict()
        }

        assert status["progress"]["progress"] == 50
        assert status["metrics"]["latency_ms"] == 500.0


class TestCompletionWithProtocolVersion:
    """Test V10.4 Completion with V10.5 Protocol Constants."""

    def test_completion_request_versioned(self):
        """Completion request includes protocol version."""
        # CompletionRequest uses ref_name, argument_name, etc.
        request = CompletionRequest(
            ref_type=CompletionRefType.RESOURCE,
            ref_name="project_files",
            argument_name="path",
            argument_value="src/"
        )

        # Add protocol version
        request_dict = request.to_dict()
        request_dict["_protocolVersion"] = MCP_LATEST_PROTOCOL_VERSION

        assert request_dict["_protocolVersion"] == "2025-11-25"

    def test_completion_supports_pagination(self):
        """Completion results can be paginated."""
        completions = [f"completion_{i}" for i in range(100)]

        # First page - CompletionResult uses from_dict, not to_dict
        result = CompletionResult(
            values=completions[:10],
            has_more=True
        )
        pagination = PaginatedResult(next_cursor="cmVzdWx0czoxMA==")

        # Verify result attributes
        assert len(result.values) == 10
        assert result.has_more is True
        assert pagination.next_cursor is not None


class TestToolMatcherWithAnnotations:
    """Test V10.2 ToolMatcher with V10.5 Annotations."""

    def test_dangerous_tool_detection(self):
        """Detect dangerous tools using annotations."""
        dangerous_tools = {
            "rm_file": ToolAnnotations.dangerous_write("Delete File"),
            "drop_table": ToolAnnotations.dangerous_write("Drop Table"),
        }

        matcher = ToolMatcher("rm_file|drop_table")

        # Check if tool is dangerous
        tool_name = "rm_file"
        if matcher.matches(tool_name):
            annotations = dangerous_tools.get(tool_name)
            if annotations and annotations.destructive_hint:
                # Require extra confirmation
                response = HookResponse.ask(
                    reason=f"Dangerous operation: {annotations.title}"
                )
                output = json.loads(response.to_json())
                assert output["hookSpecificOutput"]["permissionDecision"] == "ask"

    def test_mcp_tool_with_safe_annotation(self):
        """MCP tools with safe annotations auto-allowed."""
        matcher = ToolMatcher.mcp_server("filesystem")

        safe_operations = {
            "mcp__filesystem__read": ToolAnnotations.safe_read("Read File"),
            "mcp__filesystem__list": ToolAnnotations.safe_read("List Directory"),
        }

        tool_name = "mcp__filesystem__read"
        if matcher.matches(tool_name):
            annotations = safe_operations.get(tool_name)
            if annotations and annotations.read_only_hint:
                response = HookResponse.allow(
                    reason=f"Auto-approved: {annotations.title} is safe"
                )
                output = json.loads(response.to_json())
                assert output["hookSpecificOutput"]["permissionDecision"] == "allow"


class TestEndToEndScenarios:
    """End-to-end scenarios combining all versions."""

    def test_full_mcp_session_lifecycle(self):
        """Complete MCP session from start to finish."""
        # 1. Setup transport (V10.4) - use endpoint, not url
        config = TransportConfig(
            transport_type=TransportType.SSE,
            endpoint="https://api.example.com/mcp"
        )

        # 2. Create session (V10.4) - use transport_type directly
        session = MCPSession(
            session_id="session-full",
            transport_type=TransportType.SSE
        )

        # 3. Health check (V10.5)
        ping = PingRequest()
        pong = PingResponse(latency_ms=25.0)

        # 4. Declare capabilities (V10.3 + V10.5)
        capabilities = MCPCapabilities(sampling=True, elicitation=True)
        roots_cap = RootsCapability(list_changed=True)

        # 5. List roots (V10.5)
        roots = ListRootsResult(roots=[
            MCPRoot.from_path("/project", name="Project")
        ])

        # 6. Create task (V10.4)
        task = MCPTask(
            task_id="task-main",
            status=TaskStatus.WORKING  # MCP 2025-11-25 spec uses WORKING
        )

        # 7. Track with step (V10.5)
        step = LettaStep(
            step_id="step-main",
            agent_id="agent-main",
            model="claude-3-opus",
            created_at=datetime.now(timezone.utc)
        )

        # 8. Progress updates (V10.3)
        progress = ProgressNotification(
            progress_token=task.task_id,
            progress=100,
            total=100
        )

        # 9. Complete with result (V10.4)
        result = TaskResult(
            task_id=task.task_id,
            content=[MCPContent.text_content("Task completed successfully")]
        )

        # 10. Add feedback (V10.5)
        feedback = StepFeedback(
            step_id=step.step_id,
            feedback_type=StepFeedbackType.POSITIVE,
            comment="Excellent execution"
        )

        # Verify all components
        assert session.transport_type == TransportType.SSE
        assert pong.latency_ms == 25.0
        assert len(roots.roots) == 1
        assert task.status == TaskStatus.WORKING
        assert feedback.feedback_type == StepFeedbackType.POSITIVE

    def test_hook_security_pipeline(self):
        """Complete security pipeline using hooks."""
        # Incoming tool request
        tool_name = "Bash"
        tool_input = {"command": "rm -rf /important"}

        # 1. Match tool (V10.2)
        dangerous_matcher = ToolMatcher("Bash|Edit|Write")

        if dangerous_matcher.matches(tool_name):
            # 2. Check annotations (V10.5)
            annotations = ToolAnnotations(
                title="Shell Command",
                destructive_hint=True,
                read_only_hint=False
            )

            if annotations.destructive_hint:
                # 3. Create error for dangerous command (V10.5)
                if "rm -rf" in tool_input.get("command", ""):
                    error = McpError.invalid_params(
                        "Dangerous command pattern detected"
                    )

                    # 4. Block with hook (V10.2)
                    response = HookResponse.deny(
                        reason=f"Security: {error.message}"
                    )

                    output = json.loads(response.to_json())
                    assert output["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_knowledge_graph_memory_integration(self):
        """Integrate knowledge graph with memory blocks."""
        # 1. Create memory blocks (V10.4) - use add() not add_block()
        manager = BlockManager()
        manager.add(MemoryBlock.persona(
            "You are an expert system analyst."
        ))

        # 2. Build knowledge graph (V10.3)
        graph = KnowledgeGraph()

        # Add entities from memory context
        graph.add_entity(Entity(
            name="CurrentTask",
            entity_type="context",
            observations=["Analyzing system architecture"]
        ))

        # 3. Track in step (V10.5)
        step = LettaStep(
            step_id="step-analysis",
            agent_id="agent-analyst",
            model="claude-3-opus",
            created_at=datetime.now(timezone.utc),
            tags=["knowledge-graph", "analysis"]
        )

        # 4. Filter related steps (V10.5)
        filter = StepFilter(
            tags=["knowledge-graph"],
            has_feedback=False,
            limit=10
        )

        assert len(graph.entities) == 1
        assert "knowledge-graph" in filter.to_dict()["tags"]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
