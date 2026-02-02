#!/usr/bin/env python3
"""
Test Suite for V10.4 Ultimate Hooks

Tests all V10.4 features based on official MCP 2025-11-25 specification and SDK research:
- MCP Tasks: Async task lifecycle and status management
- MCP Prompts: Server-exposed prompt templates
- MCP Resources: Resource templates with RFC 6570 patterns
- MCP Logging: RFC 5424 syslog severity levels
- MCP Completion: Context-aware auto-completion
- Sequential Thinking: Extended reasoning with branching
- Transport Abstraction: stdio, SSE, HTTP session patterns
- Letta Memory Blocks: Core memory block lifecycle

Run with: python -m pytest tests/test_v104_ultimate.py -v
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add hooks directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))

from hook_utils import (
    # MCP Tasks (V10.4)
    TaskStatus,
    TaskSupport,
    MCPTask,
    TaskRequest,
    TaskResult,
    # MCP Prompts (V10.4)
    PromptArgument,
    MCPPrompt,
    PromptMessage,
    # MCP Resources (V10.4)
    ResourceAnnotations,
    MCPResource,
    ResourceTemplate,
    # MCP Logging (V10.4)
    LogLevel,
    LogMessage,
    # MCP Completion (V10.4)
    CompletionRefType,
    CompletionRequest,
    CompletionResult,
    # Sequential Thinking (V10.4)
    ThoughtData,
    ThinkingSession,
    # Transport Abstraction (V10.4)
    TransportType,
    TransportConfig,
    MCPSession,
    # Letta Memory Blocks (V10.4)
    MemoryBlock,
    BlockManager,
)


# =============================================================================
# MCP TASKS TESTS
# =============================================================================


class TestTaskStatus:
    """Test TaskStatus enum per MCP 2025-11-25 specification."""

    def test_all_status_values(self):
        """Verify all task status values from MCP spec."""
        assert TaskStatus.WORKING.value == "working"
        assert TaskStatus.INPUT_REQUIRED.value == "input_required"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"

    def test_status_count(self):
        """Ensure we have all 5 status types."""
        assert len(TaskStatus) == 5


class TestTaskSupport:
    """Test TaskSupport enum for tool-level task negotiation."""

    def test_support_values(self):
        """Verify task support negotiation values."""
        assert TaskSupport.REQUIRED.value == "required"
        assert TaskSupport.OPTIONAL.value == "optional"
        assert TaskSupport.FORBIDDEN.value == "forbidden"


class TestMCPTask:
    """Test MCPTask class for async task lifecycle."""

    def test_basic_task_creation(self):
        """Test creating a basic MCP task."""
        task = MCPTask(
            task_id="test-123",
            status=TaskStatus.WORKING,
            ttl=60000,
            poll_interval=5000
        )
        output = task.to_dict()

        assert output["taskId"] == "test-123"
        assert output["status"] == "working"
        assert output["ttl"] == 60000
        assert output["pollInterval"] == 5000

    def test_task_with_message(self):
        """Test task with human-readable status message."""
        task = MCPTask(
            task_id="task-456",
            status=TaskStatus.WORKING,
            message="Processing 50% complete..."
        )
        output = task.to_dict()

        assert output["message"] == "Processing 50% complete..."

    def test_task_input_required(self):
        """Test task requiring additional input (human-in-the-loop)."""
        task = MCPTask(
            task_id="task-789",
            status=TaskStatus.INPUT_REQUIRED,
            input_request={
                "type": "object",
                "properties": {
                    "approval": {"type": "boolean"}
                }
            }
        )
        output = task.to_dict()

        assert output["status"] == "input_required"
        assert "inputRequest" in output
        assert task.needs_input is True

    def test_task_completion_states(self):
        """Test terminal state detection."""
        completed = MCPTask(task_id="1", status=TaskStatus.COMPLETED)
        failed = MCPTask(task_id="2", status=TaskStatus.FAILED)
        cancelled = MCPTask(task_id="3", status=TaskStatus.CANCELLED)
        working = MCPTask(task_id="4", status=TaskStatus.WORKING)

        assert completed.is_complete is True
        assert failed.is_complete is True
        assert cancelled.is_complete is True
        assert working.is_complete is False

    def test_task_from_dict(self):
        """Test parsing task from MCP response."""
        data = {
            "taskId": "parsed-task",
            "status": "completed",
            "ttl": 30000,
            "message": "Done!",
            "createdAt": "2026-01-17T12:00:00Z"
        }
        task = MCPTask.from_dict(data)

        assert task.task_id == "parsed-task"
        assert task.status == TaskStatus.COMPLETED
        assert task.ttl == 30000
        assert task.is_complete is True


class TestTaskRequest:
    """Test TaskRequest for creating async tasks."""

    def test_task_request_format(self):
        """Test task request params format."""
        request = TaskRequest(ttl=120000)
        output = request.to_dict()

        assert output["ttl"] == 120000


class TestTaskResult:
    """Test TaskResult for completed task outputs."""

    def test_task_result_basic(self):
        """Test basic task result."""
        result = TaskResult(
            task_id="result-123",
            content=[{"type": "text", "text": "Operation completed"}],
            is_error=False
        )
        output = result.to_dict()

        assert output["taskId"] == "result-123"
        assert output["isError"] is False
        assert len(output["content"]) == 1

    def test_task_result_with_structured(self):
        """Test task result with structured content."""
        result = TaskResult(
            task_id="structured-task",
            content=[{"type": "text", "text": "Data ready"}],
            structured_content={"rows": 100, "columns": 5}
        )
        output = result.to_dict()

        assert output["structuredContent"]["rows"] == 100


# =============================================================================
# MCP PROMPTS TESTS
# =============================================================================


class TestPromptArgument:
    """Test PromptArgument for prompt template parameters."""

    def test_basic_argument(self):
        """Test basic argument definition."""
        arg = PromptArgument(
            name="language",
            description="Programming language",
            required=True
        )
        output = arg.to_dict()

        assert output["name"] == "language"
        assert output["description"] == "Programming language"
        assert output["required"] is True

    def test_optional_argument(self):
        """Test optional argument (required omitted when False)."""
        arg = PromptArgument(name="style", description="Code style")
        output = arg.to_dict()

        assert "required" not in output


class TestMCPPrompt:
    """Test MCPPrompt for server-exposed prompt templates."""

    def test_prompt_creation(self):
        """Test creating a prompt template."""
        prompt = MCPPrompt(
            name="code_review",
            title="Request Code Review",
            description="Asks the LLM to analyze code quality",
            arguments=[
                PromptArgument("code", "The code to review", required=True),
                PromptArgument("language", "Programming language")
            ]
        )
        output = prompt.to_dict()

        assert output["name"] == "code_review"
        assert output["title"] == "Request Code Review"
        assert len(output["arguments"]) == 2

    def test_prompt_from_dict(self):
        """Test parsing prompt from MCP response."""
        data = {
            "name": "explain",
            "description": "Explain a concept",
            "arguments": [
                {"name": "topic", "description": "Topic to explain", "required": True}
            ]
        }
        prompt = MCPPrompt.from_dict(data)

        assert prompt.name == "explain"
        assert len(prompt.arguments) == 1
        assert prompt.arguments[0].required is True


class TestPromptMessage:
    """Test PromptMessage for expanded prompt content."""

    def test_text_message(self):
        """Test text content message."""
        msg = PromptMessage(role="user", content="Explain this code")
        output = msg.to_dict()

        assert output["role"] == "user"
        assert output["content"]["type"] == "text"
        assert output["content"]["text"] == "Explain this code"

    def test_structured_message(self):
        """Test message with pre-structured content."""
        msg = PromptMessage(
            role="assistant",
            content={"type": "image", "data": "base64..."}
        )
        output = msg.to_dict()

        assert output["content"]["type"] == "image"


# =============================================================================
# MCP RESOURCES TESTS
# =============================================================================


class TestResourceAnnotations:
    """Test ResourceAnnotations for resource metadata."""

    def test_full_annotations(self):
        """Test annotations with all fields."""
        now = datetime.now(timezone.utc)
        annotations = ResourceAnnotations(
            audience=["user", "assistant"],
            priority=0.9,
            last_modified=now
        )
        output = annotations.to_dict()

        assert output["audience"] == ["user", "assistant"]
        assert output["priority"] == 0.9
        assert "lastModified" in output

    def test_from_dict(self):
        """Test parsing annotations from MCP response."""
        data = {
            "audience": ["user"],
            "priority": 0.5,
            "lastModified": "2026-01-17T10:00:00Z"
        }
        annotations = ResourceAnnotations.from_dict(data)

        assert annotations.audience == ["user"]
        assert annotations.priority == 0.5
        assert annotations.last_modified is not None


class TestMCPResource:
    """Test MCPResource for server-exposed resources."""

    def test_file_resource(self):
        """Test file resource creation."""
        resource = MCPResource(
            uri="file:///project/src/main.py",
            name="main.py",
            description="Application entry point",
            mime_type="text/x-python",
            size=1024
        )
        output = resource.to_dict()

        assert output["uri"] == "file:///project/src/main.py"
        assert output["name"] == "main.py"
        assert output["mimeType"] == "text/x-python"
        assert output["size"] == 1024

    def test_resource_with_annotations(self):
        """Test resource with metadata annotations."""
        resource = MCPResource(
            uri="file:///config.json",
            name="Configuration",
            annotations=ResourceAnnotations(priority=1.0)
        )
        output = resource.to_dict()

        assert output["annotations"]["priority"] == 1.0


class TestResourceTemplate:
    """Test ResourceTemplate for URI pattern matching."""

    def test_template_creation(self):
        """Test RFC 6570 template creation."""
        template = ResourceTemplate(
            uri_template="file:///{path}",
            name="Project Files",
            description="Access files in the project"
        )
        output = template.to_dict()

        assert output["uriTemplate"] == "file:///{path}"
        assert output["name"] == "Project Files"

    def test_template_expansion(self):
        """Test URI template expansion."""
        template = ResourceTemplate(
            uri_template="db://users/{userId}/profile",
            name="User Profile"
        )
        expanded = template.expand(userId="12345")

        assert expanded == "db://users/12345/profile"

    def test_multi_variable_expansion(self):
        """Test template with multiple variables."""
        template = ResourceTemplate(
            uri_template="git://{repo}/{branch}/{path}",
            name="Git Files"
        )
        expanded = template.expand(repo="myrepo", branch="main", path="src/app.py")

        assert expanded == "git://myrepo/main/src/app.py"


# =============================================================================
# MCP LOGGING TESTS
# =============================================================================


class TestLogLevel:
    """Test LogLevel enum per RFC 5424 syslog spec."""

    def test_all_log_levels(self):
        """Verify all RFC 5424 severity levels."""
        assert LogLevel.DEBUG.value == "debug"
        assert LogLevel.INFO.value == "info"
        assert LogLevel.NOTICE.value == "notice"
        assert LogLevel.WARNING.value == "warning"
        assert LogLevel.ERROR.value == "error"
        assert LogLevel.CRITICAL.value == "critical"
        assert LogLevel.ALERT.value == "alert"
        assert LogLevel.EMERGENCY.value == "emergency"

    def test_log_level_count(self):
        """Ensure we have all 8 RFC 5424 levels."""
        assert len(LogLevel) == 8


class TestLogMessage:
    """Test LogMessage for MCP logging notifications."""

    def test_basic_log(self):
        """Test basic log message."""
        msg = LogMessage(
            level=LogLevel.INFO,
            data="Server started successfully"
        )
        output = msg.to_notification()

        assert output["method"] == "notifications/message"
        assert output["params"]["level"] == "info"
        assert output["params"]["data"] == "Server started successfully"

    def test_structured_log_with_logger(self):
        """Test structured log with logger name."""
        msg = LogMessage(
            level=LogLevel.ERROR,
            logger="database",
            data={"error": "Connection failed", "host": "localhost", "port": 5432}
        )
        output = msg.to_notification()

        assert output["params"]["logger"] == "database"
        assert output["params"]["data"]["error"] == "Connection failed"


# =============================================================================
# MCP COMPLETION TESTS
# =============================================================================


class TestCompletionRefType:
    """Test CompletionRefType enum for reference types."""

    def test_ref_types(self):
        """Verify completion reference types."""
        assert CompletionRefType.PROMPT.value == "ref/prompt"
        assert CompletionRefType.RESOURCE.value == "ref/resource"


class TestCompletionRequest:
    """Test CompletionRequest for argument auto-completion."""

    def test_basic_request(self):
        """Test basic completion request."""
        request = CompletionRequest(
            ref_type=CompletionRefType.PROMPT,
            ref_name="code_review",
            argument_name="language",
            argument_value="py"
        )
        output = request.to_dict()

        assert output["ref"]["type"] == "ref/prompt"
        assert output["ref"]["name"] == "code_review"
        assert output["argument"]["name"] == "language"
        assert output["argument"]["value"] == "py"

    def test_request_with_context(self):
        """Test completion request with previous argument values."""
        request = CompletionRequest(
            ref_type=CompletionRefType.PROMPT,
            ref_name="api_request",
            argument_name="endpoint",
            argument_value="/us",
            context={"method": "GET", "base_url": "https://api.example.com"}
        )
        output = request.to_dict()

        assert output["context"]["arguments"]["method"] == "GET"


class TestCompletionResult:
    """Test CompletionResult for auto-completion suggestions."""

    def test_basic_result(self):
        """Test basic completion result."""
        data = {
            "completion": {
                "values": ["python", "pytorch", "pydantic"],
                "total": 10,
                "hasMore": True
            }
        }
        result = CompletionResult.from_dict(data)

        assert result.values == ["python", "pytorch", "pydantic"]
        assert result.total == 10
        assert result.has_more is True

    def test_empty_result(self):
        """Test empty completion result."""
        result = CompletionResult.from_dict({"completion": {}})

        assert result.values == []
        assert result.has_more is False


# =============================================================================
# SEQUENTIAL THINKING TESTS
# =============================================================================


class TestThoughtData:
    """Test ThoughtData for sequential thinking."""

    def test_basic_thought(self):
        """Test basic thought creation."""
        thought = ThoughtData(
            thought="Analyzing the problem structure...",
            thought_number=1,
            total_thoughts=5,
            next_thought_needed=True
        )
        output = thought.to_dict()

        assert output["thought"] == "Analyzing the problem structure..."
        assert output["thoughtNumber"] == 1
        assert output["totalThoughts"] == 5
        assert output["nextThoughtNeeded"] is True

    def test_revision_thought(self):
        """Test thought that revises a previous thought."""
        thought = ThoughtData(
            thought="Actually, the time complexity is O(n log n)...",
            thought_number=4,
            total_thoughts=5,
            is_revision=True,
            revises_thought=2
        )
        output = thought.to_dict()

        assert output["isRevision"] is True
        assert output["revisesThought"] == 2

    def test_branching_thought(self):
        """Test thought that branches from a previous thought."""
        thought = ThoughtData(
            thought="Exploring alternative approach...",
            thought_number=3,
            total_thoughts=5,
            branch_from_thought=2,
            branch_id="alt-approach-1"
        )
        output = thought.to_dict()

        assert output["branchFromThought"] == 2
        assert output["branchId"] == "alt-approach-1"

    def test_from_dict(self):
        """Test parsing thought from dictionary."""
        data = {
            "thought": "Testing parsing...",
            "thoughtNumber": 2,
            "totalThoughts": 3,
            "nextThoughtNeeded": False,
            "needsMoreThoughts": True
        }
        thought = ThoughtData.from_dict(data)

        assert thought.thought == "Testing parsing..."
        assert thought.next_thought_needed is False
        assert thought.needs_more_thoughts is True


class TestThinkingSession:
    """Test ThinkingSession for managing thought chains."""

    def test_session_creation(self):
        """Test creating a thinking session."""
        session = ThinkingSession(session_id="session-1")

        assert session.session_id == "session-1"
        assert len(session.thoughts) == 0

    def test_adding_thoughts(self):
        """Test adding thoughts to session."""
        session = ThinkingSession(session_id="session-2")
        session.add_thought(ThoughtData(
            thought="First thought",
            thought_number=1,
            total_thoughts=2
        ))
        session.add_thought(ThoughtData(
            thought="Second thought",
            thought_number=2,
            total_thoughts=2
        ))

        assert len(session.thoughts) == 2
        assert session.thoughts[0].thought == "First thought"

    def test_branching_in_session(self):
        """Test adding branched thoughts."""
        session = ThinkingSession(session_id="session-3")
        session.add_thought(ThoughtData(
            thought="Main thought",
            thought_number=1,
            total_thoughts=2
        ))
        session.add_thought(ThoughtData(
            thought="Branch thought",
            thought_number=2,
            total_thoughts=2,
            branch_id="branch-A",
            branch_from_thought=1
        ))

        assert len(session.thoughts) == 1
        assert "branch-A" in session.branches
        assert len(session.branches["branch-A"]) == 1

    def test_get_history(self):
        """Test getting thought history."""
        session = ThinkingSession(session_id="session-4")
        session.add_thought(ThoughtData(thought="Main", thought_number=1, total_thoughts=1))
        session.add_thought(ThoughtData(
            thought="Branch",
            thought_number=1,
            total_thoughts=1,
            branch_id="b1"
        ))

        main_history = session.get_history()
        branch_history = session.get_history(branch_id="b1")

        assert len(main_history) == 1
        assert len(branch_history) == 1
        assert branch_history[0].thought == "Branch"

    def test_jsonl_serialization(self):
        """Test JSONL serialization and parsing."""
        session = ThinkingSession(session_id="session-5")
        session.add_thought(ThoughtData(
            thought="Thought one",
            thought_number=1,
            total_thoughts=2
        ))
        session.add_thought(ThoughtData(
            thought="Thought two",
            thought_number=2,
            total_thoughts=2
        ))

        jsonl = session.to_jsonl()
        restored = ThinkingSession.from_jsonl("restored", jsonl)

        assert len(restored.thoughts) == 2
        assert restored.thoughts[0].thought == "Thought one"


# =============================================================================
# TRANSPORT ABSTRACTION TESTS
# =============================================================================


class TestTransportType:
    """Test TransportType enum for MCP transports."""

    def test_transport_types(self):
        """Verify all transport types."""
        assert TransportType.STDIO.value == "stdio"
        assert TransportType.SSE.value == "sse"
        assert TransportType.HTTP.value == "http"


class TestTransportConfig:
    """Test TransportConfig for transport settings."""

    def test_stdio_config(self):
        """Test stdio transport configuration."""
        config = TransportConfig(transport_type=TransportType.STDIO)
        output = config.to_dict()

        assert output["type"] == "stdio"

    def test_sse_config(self):
        """Test SSE transport configuration."""
        config = TransportConfig(
            transport_type=TransportType.SSE,
            endpoint="http://localhost:3000/sse",
            session_id="sse-session-1"
        )
        output = config.to_dict()

        assert output["type"] == "sse"
        assert output["endpoint"] == "http://localhost:3000/sse"
        assert output["sessionId"] == "sse-session-1"

    def test_http_config_with_resumability(self):
        """Test HTTP transport with resumability enabled."""
        config = TransportConfig(
            transport_type=TransportType.HTTP,
            endpoint="http://localhost:3001/mcp",
            port=3001,
            enable_resumability=True
        )
        output = config.to_dict()

        assert output["type"] == "http"
        assert output["port"] == 3001
        assert output["enableResumability"] is True


class TestMCPSession:
    """Test MCPSession for session state management."""

    def test_session_creation(self):
        """Test creating an MCP session."""
        session = MCPSession(
            session_id="sess-123",
            transport_type=TransportType.HTTP
        )
        output = session.to_dict()

        assert output["sessionId"] == "sess-123"
        assert output["transportType"] == "http"
        assert output["isActive"] is True

    def test_session_with_resumability(self):
        """Test session with last event ID for resumability."""
        session = MCPSession(
            session_id="sess-456",
            transport_type=TransportType.SSE,
            last_event_id="event-789"
        )
        output = session.to_dict()

        assert output["lastEventId"] == "event-789"


# =============================================================================
# LETTA MEMORY BLOCKS TESTS
# =============================================================================


class TestMemoryBlock:
    """Test MemoryBlock for Letta core memory."""

    def test_basic_block(self):
        """Test basic memory block creation."""
        block = MemoryBlock(
            label="persona",
            value="I am a helpful assistant.",
            limit=5000
        )
        output = block.to_dict()

        assert output["label"] == "persona"
        assert output["value"] == "I am a helpful assistant."
        assert output["limit"] == 5000

    def test_template_block(self):
        """Test memory block template."""
        block = MemoryBlock(
            label="greeting",
            value="Hello, {name}!",
            is_template=True,
            template_name="standard_greeting"
        )
        output = block.to_dict()

        assert output["is_template"] is True
        assert output["template_name"] == "standard_greeting"

    def test_read_only_block(self):
        """Test read-only memory block."""
        block = MemoryBlock(
            label="system",
            value="Core system instructions",
            read_only=True,
            hidden=True
        )
        output = block.to_dict()

        assert output["read_only"] is True
        assert output["hidden"] is True

    def test_block_with_metadata(self):
        """Test block with metadata and tags."""
        block = MemoryBlock(
            label="context",
            value="User preferences...",
            metadata={"source": "onboarding", "version": 2},
            tags=["user", "preferences", "settings"]
        )
        output = block.to_dict()

        assert output["metadata"]["source"] == "onboarding"
        assert "preferences" in output["tags"]

    def test_from_dict(self):
        """Test parsing block from Letta API response."""
        data = {
            "id": "block-123",
            "label": "human",
            "value": "User name: Alice",
            "limit": 2000,
            "read_only": False
        }
        block = MemoryBlock.from_dict(data)

        assert block.block_id == "block-123"
        assert block.label == "human"
        assert block.limit == 2000

    def test_persona_factory(self):
        """Test persona block factory method."""
        block = MemoryBlock.persona("I am Claude, a helpful AI assistant.")

        assert block.label == "persona"
        assert block.description == "Agent persona/identity"

    def test_human_factory(self):
        """Test human block factory method."""
        block = MemoryBlock.human("User: Bob, prefers formal communication")

        assert block.label == "human"
        assert block.description == "Human/user information"


class TestBlockManager:
    """Test BlockManager for memory block lifecycle."""

    def test_add_and_get_by_label(self):
        """Test adding blocks and retrieving by label."""
        manager = BlockManager()
        manager.add(MemoryBlock(label="persona", value="Assistant"))
        manager.add(MemoryBlock(label="human", value="User info"))

        persona = manager.get_by_label("persona")

        assert persona is not None
        assert persona.value == "Assistant"

    def test_get_by_id(self):
        """Test retrieving block by ID."""
        manager = BlockManager()
        manager.add(MemoryBlock(block_id="id-1", label="test", value="data"))

        block = manager.get_by_id("id-1")

        assert block is not None
        assert block.label == "test"

    def test_update_value(self):
        """Test updating block value."""
        manager = BlockManager()
        manager.add(MemoryBlock(label="notes", value="Initial", limit=100))

        success = manager.update_value("notes", "Updated content")

        assert success is True
        assert manager.get_by_label("notes").value == "Updated content"

    def test_update_respects_limit(self):
        """Test that update respects character limit."""
        manager = BlockManager()
        manager.add(MemoryBlock(label="short", value="x", limit=10))

        success = manager.update_value("short", "This is too long for the limit")

        assert success is False
        assert manager.get_by_label("short").value == "x"

    def test_to_list(self):
        """Test converting all blocks to list."""
        manager = BlockManager()
        manager.add(MemoryBlock(label="one", value="1"))
        manager.add(MemoryBlock(label="two", value="2"))

        blocks = manager.to_list()

        assert len(blocks) == 2
        assert blocks[0]["label"] == "one"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestV104Integration:
    """Integration tests for V10.4 features working together."""

    def test_task_with_prompt_completion(self):
        """Test task that requires prompt argument completion."""
        # Create a prompt with arguments
        prompt = MCPPrompt(
            name="analysis",
            arguments=[PromptArgument("data_source", "Source to analyze", required=True)]
        )

        # Request completion for the argument
        completion_req = CompletionRequest(
            ref_type=CompletionRefType.PROMPT,
            ref_name=prompt.name,
            argument_name="data_source",
            argument_value="db"
        )

        # Simulate completion result
        completion_result = CompletionResult(
            values=["database", "db_primary", "db_replica"]
        )

        # Create async task for analysis
        task = MCPTask(
            task_id="analysis-task",
            status=TaskStatus.WORKING,
            message="Analyzing data source..."
        )

        assert prompt.name == completion_req.ref_name
        assert len(completion_result.values) == 3
        assert not task.is_complete

    def test_resource_with_logging(self):
        """Test resource access with logging."""
        # Define a resource
        resource = MCPResource(
            uri="file:///logs/app.log",
            name="Application Logs",
            annotations=ResourceAnnotations(
                audience=["user"],
                priority=0.8
            )
        )

        # Log access to resource
        log = LogMessage(
            level=LogLevel.INFO,
            logger="resource_manager",
            data={"action": "read", "uri": resource.uri}
        )

        notification = log.to_notification()

        assert resource.annotations.priority == 0.8
        assert notification["params"]["data"]["uri"] == resource.uri

    def test_thinking_with_memory_blocks(self):
        """Test sequential thinking that updates memory blocks."""
        # Create thinking session
        session = ThinkingSession(session_id="memory-update")
        session.add_thought(ThoughtData(
            thought="User mentioned preference for dark mode",
            thought_number=1,
            total_thoughts=2
        ))

        # Update memory based on thinking
        manager = BlockManager()
        manager.add(MemoryBlock(label="preferences", value="", limit=1000))
        manager.update_value("preferences", "dark_mode: true")

        assert len(session.thoughts) == 1
        assert "dark_mode" in manager.get_by_label("preferences").value

    def test_transport_session_with_tasks(self):
        """Test managing tasks across transport sessions."""
        # Create HTTP session
        session = MCPSession(
            session_id="http-sess-1",
            transport_type=TransportType.HTTP
        )

        # Create task within session
        task = MCPTask(
            task_id=f"{session.session_id}:task-1",
            status=TaskStatus.WORKING,
            ttl=60000
        )

        assert session.session_id in task.task_id
        assert session.is_active


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
