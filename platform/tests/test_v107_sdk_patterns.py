#!/usr/bin/env python3
"""
Tests for V10.7 SDK Patterns

Tests the advanced patterns integrated from:
- Anthropic SDK v0.76.0: ToolRunner, CompactionControl, ToolCache
- Pydantic-AI v1.43.0: EndStrategy, InstrumentationSettings
- LangGraph v1.0.6: ToolCall, ToolCallRequest, InjectedState, InjectedStore
- MCP Sequential Thinking: EnhancedThinkingSession, checkpoints, branch merging
"""

import pytest
import time
from hooks.hook_utils import (
    # Tool Runner Patterns
    EndStrategy,
    CompactionMode,
    CompactionControl,
    ToolCache,
    # LangGraph Patterns
    ToolCall,
    InjectedState,
    InjectedStore,
    ToolRuntime,
    ToolCallRequest,
    ToolCallResult,
    # Instrumentation
    InstrumentationLevel,
    InstrumentationSettings,
    # Enhanced Thinking
    ThinkingCheckpoint,
    BranchMergeResult,
    EnhancedThinkingSession,
    ThoughtData,
    # Tool Runner
    ToolRunnerConfig,
    ToolRunnerResult,
    ToolRunner,
    # Thinking Part Parser
    TextPart,
    ThinkingPart,
    split_content_into_text_and_thinking,
)


# =============================================================================
# EndStrategy Tests
# =============================================================================

class TestEndStrategy:
    """Tests for EndStrategy enum (Pydantic-AI pattern)."""

    def test_early_strategy(self):
        """Test EARLY strategy value."""
        assert EndStrategy.EARLY.value == "early"

    def test_exhaustive_strategy(self):
        """Test EXHAUSTIVE strategy value."""
        assert EndStrategy.EXHAUSTIVE.value == "exhaustive"


# =============================================================================
# CompactionControl Tests
# =============================================================================

class TestCompactionMode:
    """Tests for CompactionMode enum (Anthropic SDK pattern)."""

    def test_compaction_modes(self):
        """Test all compaction modes exist."""
        assert CompactionMode.NONE.value == "none"
        assert CompactionMode.SUMMARIZE.value == "summarize"
        assert CompactionMode.TRUNCATE.value == "truncate"
        assert CompactionMode.SLIDING_WINDOW.value == "sliding_window"


class TestCompactionControl:
    """Tests for CompactionControl (Anthropic SDK pattern)."""

    def test_default_values(self):
        """Test default compaction control."""
        control = CompactionControl()
        assert control.mode == CompactionMode.NONE
        assert control.max_tokens == 200000
        assert control.window_size == 100

    def test_should_compact_none_mode(self):
        """Test that NONE mode never compacts."""
        control = CompactionControl(mode=CompactionMode.NONE)
        assert not control.should_compact(199000)
        assert not control.should_compact(250000)

    def test_should_compact_window_mode(self):
        """Test sliding window compaction trigger."""
        control = CompactionControl(
            mode=CompactionMode.SLIDING_WINDOW,
            max_tokens=100000,
        )
        # Below 90% threshold
        assert not control.should_compact(80000)
        # Above 90% threshold
        assert control.should_compact(95000)

    def test_custom_summary_prompt(self):
        """Test custom summary prompt."""
        control = CompactionControl(
            mode=CompactionMode.SUMMARIZE,
            summary_prompt="Summarize the key points",
        )
        assert control.summary_prompt == "Summarize the key points"


# =============================================================================
# ToolCache Tests
# =============================================================================

class TestToolCache:
    """Tests for ToolCache (Anthropic SDK pattern)."""

    def test_set_and_get(self):
        """Test basic cache set and get."""
        cache = ToolCache(ttl_seconds=300)
        cache.set("tool_123", {"result": "success"})
        assert cache.get("tool_123") == {"result": "success"}

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = ToolCache()
        assert cache.get("nonexistent") is None

    def test_cache_expiration(self):
        """Test cache expiration."""
        cache = ToolCache(ttl_seconds=0)  # Immediate expiration
        cache.set("tool_123", "result")
        time.sleep(0.1)  # Wait for expiration
        assert cache.get("tool_123") is None

    def test_clear_cache(self):
        """Test clearing cache."""
        cache = ToolCache()
        cache.set("tool_1", "result1")
        cache.set("tool_2", "result2")
        cache.clear()
        assert cache.get("tool_1") is None
        assert cache.get("tool_2") is None

    def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        cache = ToolCache(ttl_seconds=0)
        cache.set("tool_1", "result1")
        cache.set("tool_2", "result2")
        time.sleep(0.1)
        removed = cache.cleanup_expired()
        assert removed == 2


# =============================================================================
# ToolCall Tests
# =============================================================================

class TestToolCall:
    """Tests for ToolCall (LangGraph pattern)."""

    def test_basic_tool_call(self):
        """Test basic tool call creation."""
        call = ToolCall(name="search", args={"query": "test"}, id="tc_1")
        assert call.name == "search"
        assert call.args == {"query": "test"}
        assert call.id == "tc_1"
        assert call.type == "tool_call"

    def test_to_dict(self):
        """Test tool call serialization."""
        call = ToolCall(name="calculate", args={"x": 1, "y": 2}, id="tc_2")
        result = call.to_dict()
        assert result == {
            "name": "calculate",
            "args": {"x": 1, "y": 2},
            "id": "tc_2",
            "type": "tool_call",
        }

    def test_from_dict(self):
        """Test tool call deserialization."""
        data = {
            "name": "fetch",
            "args": {"url": "https://example.com"},
            "id": "tc_3",
            "type": "tool_call",
        }
        call = ToolCall.from_dict(data)
        assert call.name == "fetch"
        assert call.args == {"url": "https://example.com"}
        assert call.id == "tc_3"


# =============================================================================
# InjectedState and InjectedStore Tests
# =============================================================================

class TestInjectedState:
    """Tests for InjectedState (LangGraph pattern)."""

    def test_basic_injection(self):
        """Test basic state injection marker."""
        state = InjectedState()
        assert state.key is None
        assert repr(state) == "InjectedState()"

    def test_keyed_injection(self):
        """Test keyed state injection."""
        state = InjectedState(key="user_context")
        assert state.key == "user_context"
        assert "user_context" in repr(state)


class TestInjectedStore:
    """Tests for InjectedStore (LangGraph pattern)."""

    def test_basic_store(self):
        """Test basic store injection marker."""
        store = InjectedStore()
        assert store.namespace is None
        assert repr(store) == "InjectedStore()"

    def test_namespaced_store(self):
        """Test namespaced store injection."""
        store = InjectedStore(namespace="memory")
        assert store.namespace == "memory"
        assert "memory" in repr(store)


# =============================================================================
# ToolRuntime Tests
# =============================================================================

class TestToolRuntime:
    """Tests for ToolRuntime (LangGraph pattern)."""

    def test_empty_runtime(self):
        """Test empty runtime creation."""
        runtime = ToolRuntime()
        assert runtime.config == {}
        assert runtime.store == {}
        assert runtime.state == {}
        assert runtime.run_id is None

    def test_runtime_with_config(self):
        """Test runtime with configuration."""
        runtime = ToolRuntime(
            config={"timeout": 30, "retry": 3},
            run_id="run_123",
        )
        assert runtime.get_config("timeout") == 30
        assert runtime.get_config("retry") == 3
        assert runtime.get_config("missing", default=0) == 0

    def test_runtime_with_state(self):
        """Test runtime with state."""
        runtime = ToolRuntime(state={"user": "test_user"})
        assert runtime.get_state("user") == "test_user"
        assert runtime.get_state("missing", default="anon") == "anon"


# =============================================================================
# ToolCallRequest Tests
# =============================================================================

class TestToolCallRequest:
    """Tests for ToolCallRequest (LangGraph pattern)."""

    def test_basic_request(self):
        """Test basic tool call request."""
        call = ToolCall(name="search", args={"q": "test"}, id="1")
        request = ToolCallRequest(tool_call=call)
        assert request.tool_call == call
        assert request.tool_name == "search"
        assert request.tool_args == {"q": "test"}

    def test_request_with_runtime(self):
        """Test request with runtime context."""
        call = ToolCall(name="api_call", args={"endpoint": "/users"}, id="2")
        runtime = ToolRuntime(config={"api_key": "secret"})
        request = ToolCallRequest(tool_call=call, runtime=runtime)
        assert request.runtime.get_config("api_key") == "secret"

    def test_request_to_dict(self):
        """Test request serialization."""
        call = ToolCall(name="compute", args={"x": 5}, id="3")
        request = ToolCallRequest(tool_call=call)
        result = request.to_dict()
        assert result["tool_call"]["name"] == "compute"
        assert result["tool_name"] == "compute"
        assert result["tool_args"] == {"x": 5}


# =============================================================================
# ToolCallResult Tests
# =============================================================================

class TestToolCallResult:
    """Tests for ToolCallResult."""

    def test_success_result(self):
        """Test successful tool result."""
        result = ToolCallResult(
            tool_call_id="tc_1",
            tool_name="search",
            result={"items": [1, 2, 3]},
            duration_ms=50.5,
        )
        assert not result.is_error
        assert result.result == {"items": [1, 2, 3]}

    def test_error_result(self):
        """Test error tool result."""
        result = ToolCallResult(
            tool_call_id="tc_2",
            tool_name="api_call",
            error="Connection timeout",
            duration_ms=5000.0,
        )
        assert result.is_error
        assert result.error == "Connection timeout"

    def test_result_to_dict(self):
        """Test result serialization."""
        result = ToolCallResult(
            tool_call_id="tc_3",
            tool_name="compute",
            result=42,
        )
        data = result.to_dict()
        assert data["tool_call_id"] == "tc_3"
        assert data["result"] == 42
        assert "error" not in data


# =============================================================================
# InstrumentationSettings Tests
# =============================================================================

class TestInstrumentationLevel:
    """Tests for InstrumentationLevel enum (Pydantic-AI pattern)."""

    def test_level_values(self):
        """Test all instrumentation levels."""
        assert InstrumentationLevel.OFF.value == "off"
        assert InstrumentationLevel.BASIC.value == "basic"
        assert InstrumentationLevel.DETAILED.value == "detailed"
        assert InstrumentationLevel.FULL.value == "full"


class TestInstrumentationSettings:
    """Tests for InstrumentationSettings (Pydantic-AI pattern)."""

    def test_default_settings(self):
        """Test default instrumentation settings."""
        settings = InstrumentationSettings()
        assert settings.level == InstrumentationLevel.OFF
        assert not settings.is_enabled
        assert not settings.trace_tools

    def test_enabled_settings(self):
        """Test enabled instrumentation."""
        settings = InstrumentationSettings(
            level=InstrumentationLevel.DETAILED,
            trace_tools=True,
            log_to="logfire",
        )
        assert settings.is_enabled
        assert settings.trace_tools
        assert settings.log_to == "logfire"

    def test_should_trace_disabled(self):
        """Test tracing when disabled."""
        settings = InstrumentationSettings(level=InstrumentationLevel.OFF)
        assert not settings.should_trace()

    def test_should_trace_full_sample(self):
        """Test tracing with full sample rate."""
        settings = InstrumentationSettings(
            level=InstrumentationLevel.BASIC,
            sample_rate=1.0,
        )
        # Should always trace at 100% sample rate
        for _ in range(10):
            assert settings.should_trace()


# =============================================================================
# Enhanced Thinking Session Tests
# =============================================================================

class TestThinkingCheckpoint:
    """Tests for ThinkingCheckpoint."""

    def test_basic_checkpoint(self):
        """Test basic checkpoint creation."""
        checkpoint = ThinkingCheckpoint(
            checkpoint_id="cp_1",
            thought_number=5,
        )
        assert checkpoint.checkpoint_id == "cp_1"
        assert checkpoint.thought_number == 5
        assert checkpoint.timestamp is not None

    def test_checkpoint_with_metadata(self):
        """Test checkpoint with metadata."""
        checkpoint = ThinkingCheckpoint(
            checkpoint_id="cp_2",
            thought_number=3,
            branch_id="branch_alt",
            metadata={"reason": "exploring alternative"},
        )
        assert checkpoint.branch_id == "branch_alt"
        assert checkpoint.metadata["reason"] == "exploring alternative"


class TestBranchMergeResult:
    """Tests for BranchMergeResult."""

    def test_successful_merge(self):
        """Test successful branch merge result."""
        result = BranchMergeResult(
            merged_thoughts=5,
            conflicts=[],
            resolution="append",
        )
        assert result.merged_thoughts == 5
        assert len(result.conflicts) == 0

    def test_merge_with_conflicts(self):
        """Test merge result with conflicts."""
        result = BranchMergeResult(
            merged_thoughts=3,
            conflicts=[2, 3, 4],
            resolution="replace",
        )
        assert len(result.conflicts) == 3


class TestEnhancedThinkingSession:
    """Tests for EnhancedThinkingSession."""

    def test_session_creation(self):
        """Test enhanced session creation."""
        session = EnhancedThinkingSession(session_id="sess_1")
        assert session.session_id == "sess_1"
        assert len(session.thoughts) == 0
        assert session.created_at is not None

    def test_add_thought_updates_timestamp(self):
        """Test that adding thought updates timestamp."""
        session = EnhancedThinkingSession(session_id="sess_2")
        initial_time = session.updated_at
        time.sleep(0.01)
        thought = ThoughtData(
            thought="Step 1",
            thought_number=1,
            total_thoughts=3,
        )
        session.add_thought(thought)
        assert session.updated_at > initial_time

    def test_create_and_restore_checkpoint(self):
        """Test checkpoint creation and restoration."""
        session = EnhancedThinkingSession(session_id="sess_3")

        # Add some thoughts
        for i in range(5):
            session.add_thought(ThoughtData(
                thought=f"Step {i+1}",
                thought_number=i+1,
                total_thoughts=10,
            ))

        # Create checkpoint after 5 thoughts
        checkpoint = session.create_checkpoint()
        assert len(session.checkpoints) == 1

        # Add more thoughts
        for i in range(5, 8):
            session.add_thought(ThoughtData(
                thought=f"Step {i+1}",
                thought_number=i+1,
                total_thoughts=10,
            ))

        assert len(session.thoughts) == 8

        # Restore checkpoint
        restored = session.restore_checkpoint(checkpoint.checkpoint_id)
        assert restored
        assert len(session.thoughts) == 5

    def test_revision_tracking(self):
        """Test revision history tracking."""
        session = EnhancedThinkingSession(session_id="sess_4")

        # Original thought
        session.add_thought(ThoughtData(
            thought="Original approach",
            thought_number=1,
            total_thoughts=3,
        ))

        # Revision of thought 1
        session.add_thought(ThoughtData(
            thought="Revised approach",
            thought_number=2,
            total_thoughts=3,
            is_revision=True,
            revises_thought=1,
        ))

        assert len(session.revision_history) == 1
        assert session.revision_history[0] == (1, 2)

    def test_merge_branch_append(self):
        """Test branch merging with append resolution."""
        session = EnhancedThinkingSession(session_id="sess_5")

        # Add main chain thoughts
        session.add_thought(ThoughtData(
            thought="Main step 1",
            thought_number=1,
            total_thoughts=5,
        ))

        # Add branch thoughts
        branch_thought = ThoughtData(
            thought="Branch step 1",
            thought_number=1,
            total_thoughts=2,
            branch_id="branch_1",
            branch_from_thought=1,
        )
        session.add_thought(branch_thought)

        assert "branch_1" in session.branches

        # Merge branch
        result = session.merge_branch("branch_1", resolution="append")
        assert result is not None
        assert result.merged_thoughts == 1
        assert "branch_1" not in session.branches
        assert len(session.thoughts) == 2

    def test_get_statistics(self):
        """Test session statistics."""
        session = EnhancedThinkingSession(session_id="sess_6")

        # Add thoughts
        for i in range(3):
            session.add_thought(ThoughtData(
                thought=f"Step {i+1}",
                thought_number=i+1,
                total_thoughts=5,
            ))

        # Add a branch
        session.add_thought(ThoughtData(
            thought="Branch thought",
            thought_number=1,
            total_thoughts=2,
            branch_id="alt",
        ))

        # Create checkpoint
        session.create_checkpoint()

        stats = session.get_statistics()
        assert stats["session_id"] == "sess_6"
        assert stats["main_thoughts"] == 3
        assert stats["total_branches"] == 1
        assert stats["total_branch_thoughts"] == 1
        assert stats["total_thoughts"] == 4
        assert stats["checkpoints"] == 1


# =============================================================================
# ToolRunner Tests
# =============================================================================

class TestToolRunnerConfig:
    """Tests for ToolRunnerConfig."""

    def test_default_config(self):
        """Test default runner configuration."""
        config = ToolRunnerConfig()
        assert config.max_iterations == 10
        assert config.end_strategy == EndStrategy.EARLY
        assert config.parallel_tool_calls

    def test_custom_config(self):
        """Test custom runner configuration."""
        cache = ToolCache(ttl_seconds=60)
        config = ToolRunnerConfig(
            max_iterations=5,
            end_strategy=EndStrategy.EXHAUSTIVE,
            cache=cache,
            timeout_ms=5000,
        )
        assert config.max_iterations == 5
        assert config.end_strategy == EndStrategy.EXHAUSTIVE
        assert config.cache is cache
        assert config.timeout_ms == 5000


class TestToolRunnerResult:
    """Tests for ToolRunnerResult."""

    def test_empty_result(self):
        """Test empty runner result."""
        result = ToolRunnerResult()
        assert len(result.tool_results) == 0
        assert result.iterations == 0
        assert not result.has_errors

    def test_add_result(self):
        """Test adding tool results."""
        runner_result = ToolRunnerResult()
        runner_result.add_result(ToolCallResult(
            tool_call_id="tc_1",
            tool_name="search",
            result="found it",
            duration_ms=100.0,
        ))
        assert len(runner_result.tool_results) == 1
        assert runner_result.total_duration_ms == 100.0

    def test_has_errors(self):
        """Test error detection in results."""
        runner_result = ToolRunnerResult()
        runner_result.add_result(ToolCallResult(
            tool_call_id="tc_1",
            tool_name="ok_tool",
            result="success",
        ))
        assert not runner_result.has_errors

        runner_result.add_result(ToolCallResult(
            tool_call_id="tc_2",
            tool_name="bad_tool",
            error="failed",
        ))
        assert runner_result.has_errors

    def test_successful_results(self):
        """Test filtering successful results."""
        runner_result = ToolRunnerResult()
        runner_result.add_result(ToolCallResult(
            tool_call_id="tc_1", tool_name="good", result="ok"
        ))
        runner_result.add_result(ToolCallResult(
            tool_call_id="tc_2", tool_name="bad", error="fail"
        ))
        runner_result.add_result(ToolCallResult(
            tool_call_id="tc_3", tool_name="good2", result="ok2"
        ))

        successful = runner_result.successful_results
        assert len(successful) == 2


class TestToolRunner:
    """Tests for ToolRunner (Anthropic SDK pattern)."""

    def test_execute_simple_tool(self):
        """Test executing a simple tool."""
        def add(a: int, b: int) -> int:
            return a + b

        runner = ToolRunner(tools={"add": add})
        call = ToolCall(name="add", args={"a": 2, "b": 3}, id="tc_1")
        request = ToolCallRequest(tool_call=call)

        result = runner.execute(request)
        assert not result.is_error
        assert result.result == 5

    def test_execute_missing_tool(self):
        """Test executing a non-existent tool."""
        runner = ToolRunner(tools={})
        call = ToolCall(name="missing", args={}, id="tc_1")
        request = ToolCallRequest(tool_call=call)

        result = runner.execute(request)
        assert result.is_error
        assert "not found" in result.error

    def test_execute_with_exception(self):
        """Test handling tool exceptions."""
        def failing_tool() -> None:
            raise ValueError("Tool failed intentionally")

        runner = ToolRunner(tools={"fail": failing_tool})
        call = ToolCall(name="fail", args={}, id="tc_1")
        request = ToolCallRequest(tool_call=call)

        result = runner.execute(request)
        assert result.is_error
        assert "Tool failed intentionally" in result.error

    def test_execute_with_cache(self):
        """Test tool caching."""
        call_count = {"count": 0}

        def expensive_tool() -> str:
            call_count["count"] += 1
            return "result"

        cache = ToolCache(ttl_seconds=60)
        config = ToolRunnerConfig(cache=cache)
        runner = ToolRunner(tools={"expensive": expensive_tool}, config=config)

        call = ToolCall(name="expensive", args={}, id="tc_cached")
        request = ToolCallRequest(tool_call=call)

        # First call
        result1 = runner.execute(request)
        assert result1.result == "result"

        # Second call should be cached
        result2 = runner.execute(request)
        assert result2.result == "result"
        assert result2.duration_ms == 0  # Cached, no time spent

        # Tool should only be called once
        assert call_count["count"] == 1

    def test_execute_batch_early_stop(self):
        """Test batch execution with early stopping."""
        def search(query: str) -> str:
            return f"found: {query}"

        config = ToolRunnerConfig(end_strategy=EndStrategy.EARLY)
        runner = ToolRunner(tools={"search": search}, config=config)

        requests = [
            ToolCallRequest(tool_call=ToolCall(name="search", args={"query": "a"}, id="1")),
            ToolCallRequest(tool_call=ToolCall(name="search", args={"query": "b"}, id="2")),
            ToolCallRequest(tool_call=ToolCall(name="search", args={"query": "c"}, id="3")),
        ]

        result = runner.execute_batch(requests)
        # Early strategy stops after first success
        assert len(result.tool_results) == 1
        assert result.stopped_early
        assert result.stop_reason == "early_success"

    def test_execute_batch_exhaustive(self):
        """Test batch execution with exhaustive strategy."""
        def process(x: int) -> int:
            return x * 2

        config = ToolRunnerConfig(end_strategy=EndStrategy.EXHAUSTIVE)
        runner = ToolRunner(tools={"process": process}, config=config)

        requests = [
            ToolCallRequest(tool_call=ToolCall(name="process", args={"x": 1}, id="1")),
            ToolCallRequest(tool_call=ToolCall(name="process", args={"x": 2}, id="2")),
            ToolCallRequest(tool_call=ToolCall(name="process", args={"x": 3}, id="3")),
        ]

        result = runner.execute_batch(requests)
        # Exhaustive executes all
        assert len(result.tool_results) == 3
        assert not result.stopped_early

    def test_execute_batch_max_iterations(self):
        """Test batch execution respects max iterations."""
        def simple() -> str:
            return "done"

        config = ToolRunnerConfig(
            max_iterations=2,
            end_strategy=EndStrategy.EXHAUSTIVE,
        )
        runner = ToolRunner(tools={"simple": simple}, config=config)

        requests = [
            ToolCallRequest(tool_call=ToolCall(name="simple", args={}, id=str(i)))
            for i in range(5)
        ]

        result = runner.execute_batch(requests)
        assert len(result.tool_results) == 2
        assert result.stopped_early
        assert result.stop_reason == "max_iterations"


# =============================================================================
# Thinking Part Parser Tests
# =============================================================================

class TestTextPart:
    """Tests for TextPart."""

    def test_text_part(self):
        """Test text part creation."""
        part = TextPart(content="Hello world")
        assert part.content == "Hello world"
        assert part.part_type == "text"


class TestThinkingPart:
    """Tests for ThinkingPart."""

    def test_thinking_part(self):
        """Test thinking part creation."""
        part = ThinkingPart(content="Analyzing the problem...")
        assert part.content == "Analyzing the problem..."
        assert part.part_type == "thinking"


class TestSplitContentIntoTextAndThinking:
    """Tests for split_content_into_text_and_thinking (Pydantic-AI pattern)."""

    def test_no_thinking_tags(self):
        """Test content without thinking tags."""
        result = split_content_into_text_and_thinking("Just regular text")
        assert len(result) == 1
        assert isinstance(result[0], TextPart)
        assert result[0].content == "Just regular text"

    def test_single_thinking_block(self):
        """Test single thinking block."""
        content = "Before <think>thinking here</think> After"
        result = split_content_into_text_and_thinking(content)
        assert len(result) == 3
        assert isinstance(result[0], TextPart)
        assert result[0].content == "Before "
        assert isinstance(result[1], ThinkingPart)
        assert result[1].content == "thinking here"
        assert isinstance(result[2], TextPart)
        assert result[2].content == " After"

    def test_multiple_thinking_blocks(self):
        """Test multiple thinking blocks."""
        content = "A <think>first</think> B <think>second</think> C"
        result = split_content_into_text_and_thinking(content)
        assert len(result) == 5
        assert result[0].content == "A "
        assert result[1].content == "first"
        assert result[2].content == " B "
        assert result[3].content == "second"
        assert result[4].content == " C"

    def test_adjacent_thinking_blocks(self):
        """Test adjacent thinking blocks."""
        content = "<think>one</think><think>two</think>"
        result = split_content_into_text_and_thinking(content)
        assert len(result) == 2
        assert all(isinstance(p, ThinkingPart) for p in result)
        assert result[0].content == "one"
        assert result[1].content == "two"

    def test_unclosed_thinking_tag(self):
        """Test unclosed thinking tag."""
        content = "Before <think>unclosed thinking"
        result = split_content_into_text_and_thinking(content)
        assert len(result) == 2
        assert result[0].content == "Before "
        # Unclosed tag content treated as text
        assert isinstance(result[1], TextPart)

    def test_custom_thinking_tags(self):
        """Test custom thinking tags."""
        content = "Before [THINK]custom thinking[/THINK] After"
        result = split_content_into_text_and_thinking(
            content,
            thinking_tags=("[THINK]", "[/THINK]")
        )
        assert len(result) == 3
        assert result[1].content == "custom thinking"

    def test_empty_thinking_block(self):
        """Test empty thinking block."""
        content = "Before <think></think> After"
        result = split_content_into_text_and_thinking(content)
        assert len(result) == 3
        assert result[1].content == ""


# =============================================================================
# Integration Tests
# =============================================================================

class TestV107Integration:
    """Integration tests combining V10.7 patterns."""

    def test_tool_runner_with_thinking_session(self):
        """Test tool runner integrated with thinking session."""
        session = EnhancedThinkingSession(session_id="integration_1")

        def analyze(text: str) -> dict:
            # Simulate analysis
            return {"sentiment": "positive", "confidence": 0.9}

        runner = ToolRunner(tools={"analyze": analyze})

        # Record thought before tool call
        session.add_thought(ThoughtData(
            thought="Preparing to analyze user input",
            thought_number=1,
            total_thoughts=3,
        ))

        # Execute tool
        call = ToolCall(name="analyze", args={"text": "Great product!"}, id="1")
        result = runner.execute(ToolCallRequest(tool_call=call))

        # Record result in thinking
        session.add_thought(ThoughtData(
            thought=f"Analysis complete: {result.result}",
            thought_number=2,
            total_thoughts=3,
        ))

        assert len(session.thoughts) == 2
        assert not result.is_error

    def test_instrumented_tool_runner(self):
        """Test tool runner with instrumentation settings."""
        instrumentation = InstrumentationSettings(
            level=InstrumentationLevel.DETAILED,
            trace_tools=True,
            sample_rate=1.0,
        )

        config = ToolRunnerConfig(instrumentation=instrumentation)

        def tracked_tool() -> str:
            return "tracked"

        runner = ToolRunner(tools={"tracked": tracked_tool}, config=config)

        assert runner.config.instrumentation.is_enabled
        assert runner.config.instrumentation.should_trace()

    def test_cached_runner_with_compaction(self):
        """Test cached runner with compaction control."""
        cache = ToolCache(ttl_seconds=300)
        compaction = CompactionControl(
            mode=CompactionMode.SLIDING_WINDOW,
            max_tokens=50000,
            window_size=20,
        )

        config = ToolRunnerConfig(
            cache=cache,
            compaction=compaction,
            max_iterations=5,
        )

        def search(q: str) -> str:
            return f"Results for: {q}"

        runner = ToolRunner(tools={"search": search}, config=config)

        # First call
        call1 = ToolCall(name="search", args={"q": "test"}, id="tc_1")
        result1 = runner.execute(ToolCallRequest(tool_call=call1))

        # Second call (should be cached)
        result2 = runner.execute(ToolCallRequest(tool_call=call1))

        assert result1.result == result2.result
        assert result2.duration_ms == 0  # Cached


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
