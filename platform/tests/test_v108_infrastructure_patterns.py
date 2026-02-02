#!/usr/bin/env python3
"""
Tests for V10.8 Infrastructure Patterns.

Covers:
- Structlog Processor Patterns (LogProcessorProtocol, ContextVarBinding, ContextVarManager, ProcessorChain)
- Structlog Renderers (key_value_renderer, logfmt_renderer)
- OpenTelemetry Span Patterns (SpanStatusCode, SpanStatus, SpanContext, SpanLink, SpanEvent, SpanKind, Span)
- Sentry Scope Patterns (ScopeType, Breadcrumb, ScopeData, Scope, ScopeManager)
- pyribs Archive Patterns (Elite, ArchiveAddStatus, ArchiveAddResult, ArchiveStats, Archive)
- Quality-Diversity Thinking (ThinkingElite, ThinkingArchive)
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
import time

# Add hooks directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))

from hook_utils import (
    # Structlog Processor Patterns
    LogProcessorProtocol,
    ContextVarBinding,
    ContextVarManager,
    ProcessorChain,
    key_value_renderer,
    logfmt_renderer,
    # OpenTelemetry Span Patterns
    SpanStatusCode,
    SpanStatus,
    SpanContext,
    SpanLink,
    SpanEvent,
    SpanKind,
    Span,
    # Sentry Scope Patterns
    ScopeType,
    Breadcrumb,
    ScopeData,
    Scope,
    ScopeManager,
    # pyribs Archive Patterns
    Elite,
    ArchiveAddStatus,
    ArchiveAddResult,
    ArchiveStats,
    Archive,
    # Quality-Diversity Thinking
    ThinkingElite,
    ThinkingArchive,
)


# =============================================================================
# STRUCTLOG PROCESSOR PATTERN TESTS
# =============================================================================


class TestContextVarBinding:
    """Tests for ContextVarBinding."""

    def test_basic_binding(self):
        """Test creating a basic binding."""
        binding = ContextVarBinding(key="request_id", value="abc123")
        assert binding.key == "request_id"
        assert binding.value == "abc123"
        assert binding.prefix == "structlog_"

    def test_full_key(self):
        """Test full key property."""
        binding = ContextVarBinding(key="user_id", value=42)
        assert binding.full_key == "structlog_user_id"

    def test_custom_prefix(self):
        """Test custom prefix."""
        binding = ContextVarBinding(key="trace_id", value="xyz", prefix="custom_")
        assert binding.full_key == "custom_trace_id"


class TestContextVarManager:
    """Tests for ContextVarManager."""

    def test_bind_and_get(self):
        """Test binding and getting context."""
        manager = ContextVarManager()
        tokens = manager.bind(request_id="req-123", user_id=42)

        assert "request_id" in tokens
        assert "user_id" in tokens

        context = manager.get_context()
        assert context["request_id"] == "req-123"
        assert context["user_id"] == 42

    def test_unbind(self):
        """Test unbinding keys."""
        manager = ContextVarManager()
        manager.bind(a=1, b=2, c=3)
        manager.unbind("b")

        context = manager.get_context()
        assert "a" in context
        assert "b" not in context
        assert "c" in context

    def test_clear(self):
        """Test clearing all bindings."""
        manager = ContextVarManager()
        manager.bind(x=1, y=2)
        manager.clear()

        assert manager.get_context() == {}

    def test_merge_into(self):
        """Test merging context into event dict."""
        manager = ContextVarManager()
        manager.bind(trace_id="trace-abc", span_id="span-123")

        event_dict = {"event": "test", "level": "info"}
        result = manager.merge_into(event_dict)

        assert result["trace_id"] == "trace-abc"
        assert result["span_id"] == "span-123"
        assert result["event"] == "test"

    def test_merge_preserves_existing(self):
        """Test that merge doesn't overwrite existing keys."""
        manager = ContextVarManager()
        manager.bind(key1="context_value")

        event_dict = {"key1": "original_value", "key2": "other"}
        result = manager.merge_into(event_dict)

        assert result["key1"] == "original_value"  # Not overwritten


class TestProcessorChain:
    """Tests for ProcessorChain."""

    def test_add_processor(self):
        """Test adding processors."""
        chain = ProcessorChain()

        def processor1(logger, method, event):
            event["p1"] = True
            return event

        result = chain.add(processor1)
        assert result is chain  # Returns self for chaining
        assert len(chain.processors) == 1

    def test_process_chain(self):
        """Test processing through chain."""
        chain = ProcessorChain()

        def add_timestamp(logger, method, event):
            event["timestamp"] = "2026-01-17"
            return event

        def add_level(logger, method, event):
            event["level"] = method
            return event

        chain.add(add_timestamp).add(add_level)

        result = chain.process(None, "info", {"event": "test"})
        assert result["timestamp"] == "2026-01-17"
        assert result["level"] == "info"
        assert result["event"] == "test"


class TestRenderers:
    """Tests for structlog renderers."""

    def test_key_value_renderer(self):
        """Test key-value renderer."""
        event_dict = {"event": "user_login", "user_id": 42, "success": True}
        result = key_value_renderer(None, "info", event_dict)

        assert "event=user_login" in result
        assert "success=True" in result
        assert "user_id=42" in result

    def test_key_value_renderer_quotes_spaces(self):
        """Test that values with spaces are quoted."""
        event_dict = {"event": "test", "message": "hello world"}
        result = key_value_renderer(None, "info", event_dict)

        assert 'message="hello world"' in result

    def test_logfmt_renderer(self):
        """Test logfmt renderer."""
        event_dict = {"event": "test", "count": 5, "enabled": True}
        result = logfmt_renderer(None, "info", event_dict)

        assert "event=test" in result
        assert "count=5" in result
        assert "enabled=true" in result  # Booleans lowercase

    def test_logfmt_skips_none(self):
        """Test that None values are skipped."""
        event_dict = {"event": "test", "optional": None}
        result = logfmt_renderer(None, "info", event_dict)

        assert "optional" not in result


# =============================================================================
# OPENTELEMETRY SPAN PATTERN TESTS
# =============================================================================


class TestSpanStatusCode:
    """Tests for SpanStatusCode enum."""

    def test_status_codes(self):
        """Test all status code values."""
        assert SpanStatusCode.UNSET.value == "UNSET"
        assert SpanStatusCode.OK.value == "OK"
        assert SpanStatusCode.ERROR.value == "ERROR"


class TestSpanStatus:
    """Tests for SpanStatus."""

    def test_default_status(self):
        """Test default status is UNSET."""
        status = SpanStatus()
        assert status.status_code == SpanStatusCode.UNSET
        assert status.description is None

    def test_ok_factory(self):
        """Test OK factory method."""
        status = SpanStatus.ok()
        assert status.status_code == SpanStatusCode.OK

    def test_error_factory(self):
        """Test error factory method."""
        status = SpanStatus.error("Something went wrong")
        assert status.status_code == SpanStatusCode.ERROR
        assert status.description == "Something went wrong"


class TestSpanContext:
    """Tests for SpanContext."""

    def test_generate_context(self):
        """Test generating random context."""
        ctx = SpanContext.generate()
        assert len(ctx.trace_id) == 32  # UUID hex
        assert len(ctx.span_id) == 16
        assert ctx.is_valid

    def test_is_valid(self):
        """Test validity check."""
        valid = SpanContext(trace_id="abc123", span_id="def456")
        assert valid.is_valid

        invalid = SpanContext(trace_id="", span_id="")
        assert not invalid.is_valid


class TestSpanEvent:
    """Tests for SpanEvent."""

    def test_event_creation(self):
        """Test creating an event."""
        event = SpanEvent(name="exception", attributes={"type": "ValueError"})
        assert event.name == "exception"
        assert event.attributes["type"] == "ValueError"
        assert event.timestamp is not None

    def test_auto_timestamp(self):
        """Test automatic timestamp."""
        before = datetime.now(timezone.utc)
        event = SpanEvent(name="test")
        after = datetime.now(timezone.utc)

        assert before <= event.timestamp <= after


class TestSpanKind:
    """Tests for SpanKind enum."""

    def test_span_kinds(self):
        """Test all span kind values."""
        assert SpanKind.INTERNAL.value == "INTERNAL"
        assert SpanKind.SERVER.value == "SERVER"
        assert SpanKind.CLIENT.value == "CLIENT"
        assert SpanKind.PRODUCER.value == "PRODUCER"
        assert SpanKind.CONSUMER.value == "CONSUMER"


class TestSpan:
    """Tests for Span."""

    def test_span_creation(self):
        """Test creating a span."""
        span = Span(name="test-operation")
        assert span.name == "test-operation"
        assert span.context.is_valid
        assert span.kind == SpanKind.INTERNAL
        assert span.is_recording()

    def test_set_attribute(self):
        """Test setting attributes."""
        span = Span(name="test")
        result = span.set_attribute("http.method", "GET")

        assert result is span  # Chaining
        assert span.attributes["http.method"] == "GET"

    def test_set_attributes(self):
        """Test setting multiple attributes."""
        span = Span(name="test")
        span.set_attributes({"a": 1, "b": 2, "c": 3})

        assert span.attributes["a"] == 1
        assert span.attributes["b"] == 2
        assert span.attributes["c"] == 3

    def test_add_event(self):
        """Test adding events."""
        span = Span(name="test")
        span.add_event("exception", {"message": "error occurred"})

        assert len(span.events) == 1
        assert span.events[0].name == "exception"

    def test_add_link(self):
        """Test adding links."""
        span = Span(name="test")
        linked_ctx = SpanContext.generate()
        span.add_link(linked_ctx, {"relationship": "follows"})

        assert len(span.links) == 1
        assert span.links[0].context == linked_ctx

    def test_set_status(self):
        """Test setting status."""
        span = Span(name="test")
        span.set_status(SpanStatus.ok())

        assert span.status.status_code == SpanStatusCode.OK

    def test_end_span(self):
        """Test ending a span."""
        span = Span(name="test")
        assert span.is_recording()

        span.end()
        assert not span.is_recording()
        assert span.end_time is not None

    def test_duration(self):
        """Test duration calculation."""
        span = Span(name="test")
        time.sleep(0.01)  # 10ms
        span.end()

        assert span.duration_ms is not None
        assert span.duration_ms >= 10

    def test_no_modify_after_end(self):
        """Test that span cannot be modified after end."""
        span = Span(name="test")
        span.end()

        span.set_attribute("new_key", "value")
        assert "new_key" not in span.attributes


# =============================================================================
# SENTRY SCOPE PATTERN TESTS
# =============================================================================


class TestScopeType:
    """Tests for ScopeType enum."""

    def test_scope_types(self):
        """Test all scope type values."""
        assert ScopeType.CURRENT.value == "current"
        assert ScopeType.ISOLATION.value == "isolation"
        assert ScopeType.GLOBAL.value == "global"
        assert ScopeType.MERGED.value == "merged"


class TestBreadcrumb:
    """Tests for Breadcrumb."""

    def test_breadcrumb_creation(self):
        """Test creating a breadcrumb."""
        crumb = Breadcrumb(
            type="http",
            category="xhr",
            message="GET /api/users",
            level="info"
        )
        assert crumb.type == "http"
        assert crumb.category == "xhr"
        assert crumb.message == "GET /api/users"
        assert crumb.timestamp is not None

    def test_to_dict(self):
        """Test converting to dict."""
        crumb = Breadcrumb(
            type="navigation",
            category="route",
            message="Navigate to /home",
            data={"from": "/login"}
        )
        d = crumb.to_dict()

        assert d["type"] == "navigation"
        assert d["category"] == "route"
        assert d["message"] == "Navigate to /home"
        assert d["data"]["from"] == "/login"


class TestScopeData:
    """Tests for ScopeData."""

    def test_set_user(self):
        """Test setting user."""
        data = ScopeData()
        data.set_user({"id": "user-123", "email": "test@example.com"})

        assert data.user["id"] == "user-123"

    def test_set_tag(self):
        """Test setting tags."""
        data = ScopeData()
        data.set_tag("environment", "production")

        assert data.tags["environment"] == "production"

    def test_add_breadcrumb(self):
        """Test adding breadcrumbs."""
        data = ScopeData()
        crumb = Breadcrumb(message="test")
        data.add_breadcrumb(crumb)

        assert len(data.breadcrumbs) == 1


class TestScope:
    """Tests for Scope."""

    def test_scope_creation(self):
        """Test creating a scope."""
        scope = Scope()
        assert scope.scope_type == ScopeType.CURRENT
        assert scope.data is not None

    def test_chaining(self):
        """Test method chaining."""
        scope = (
            Scope()
            .set_user({"id": "123"})
            .set_tag("version", "1.0")
            .set_extra("debug", True)
        )

        assert scope.data.user["id"] == "123"
        assert scope.data.tags["version"] == "1.0"
        assert scope.data.extras["debug"] is True

    def test_add_breadcrumb(self):
        """Test adding breadcrumbs."""
        scope = Scope()
        scope.add_breadcrumb(message="Step 1", category="process")
        scope.add_breadcrumb(message="Step 2", category="process")

        assert len(scope.data.breadcrumbs) == 2

    def test_max_breadcrumbs(self):
        """Test max breadcrumbs limit."""
        scope = Scope(max_breadcrumbs=3)
        for i in range(5):
            scope.add_breadcrumb(message=f"Crumb {i}")

        assert len(scope.data.breadcrumbs) == 3

    def test_apply_to_event(self):
        """Test applying scope to event."""
        scope = (
            Scope()
            .set_user({"id": "user-1"})
            .set_tag("env", "test")
            .set_extra("debug", True)
            .add_breadcrumb(message="action")
        )

        event = {"type": "error", "message": "test error"}
        result = scope.apply_to_event(event)

        assert result["user"]["id"] == "user-1"
        assert result["tags"]["env"] == "test"
        assert result["extra"]["debug"] is True
        assert len(result["breadcrumbs"]) == 1

    def test_event_processor(self):
        """Test event processor."""
        scope = Scope()

        def add_release(event):
            event["release"] = "1.0.0"
            return event

        scope.add_event_processor(add_release)

        event = {"type": "error"}
        result = scope.apply_to_event(event)

        assert result["release"] == "1.0.0"

    def test_event_processor_can_drop(self):
        """Test that event processor can drop event."""
        scope = Scope()

        def drop_event(event):
            return None  # Drop the event

        scope.add_event_processor(drop_event)

        event = {"type": "error"}
        result = scope.apply_to_event(event)

        assert result is None

    def test_fork(self):
        """Test forking a scope."""
        parent = Scope().set_tag("parent", "true").set_user({"id": "1"})
        child = parent.fork()

        # Child has parent's data
        assert child.data.tags["parent"] == "true"
        assert child.data.user["id"] == "1"

        # Modifications don't affect parent
        child.set_tag("child", "true")
        assert "child" not in parent.data.tags


class TestScopeManager:
    """Tests for ScopeManager."""

    def test_get_global_scope(self):
        """Test getting global scope."""
        manager = ScopeManager()
        global_scope = manager.get_global_scope()

        assert global_scope.scope_type == ScopeType.GLOBAL

    def test_get_isolation_scope(self):
        """Test getting isolation scope."""
        manager = ScopeManager()
        iso_scope = manager.get_isolation_scope()

        assert iso_scope.scope_type == ScopeType.ISOLATION

    def test_push_pop_scope(self):
        """Test push/pop scope."""
        manager = ScopeManager()
        manager.get_global_scope().set_tag("global", "true")

        pushed = manager.push_scope()
        pushed.set_tag("pushed", "true")

        assert pushed.data.tags["global"] == "true"  # Inherited
        assert pushed.data.tags["pushed"] == "true"

        manager.pop_scope()
        current = manager.get_current_scope()
        assert "pushed" not in current.data.tags

    def test_get_merged_scope(self):
        """Test getting merged scope."""
        manager = ScopeManager()

        manager.get_global_scope().set_tag("level", "global")
        manager.get_isolation_scope().set_tag("level", "isolation")
        manager.push_scope().set_tag("level", "current")

        merged = manager.get_merged_scope()
        # Last one wins
        assert merged.data.tags["level"] == "current"
        assert merged.scope_type == ScopeType.MERGED


# =============================================================================
# PYRIBS ARCHIVE PATTERN TESTS
# =============================================================================


class TestElite:
    """Tests for Elite."""

    def test_elite_creation(self):
        """Test creating an elite."""
        elite = Elite(
            solution=[1.0, 2.0, 3.0],
            objective=0.95,
            measures=[0.5, 0.5]
        )
        assert elite.solution == [1.0, 2.0, 3.0]
        assert elite.objective == 0.95
        assert elite.measures == [0.5, 0.5]

    def test_to_dict(self):
        """Test converting to dict."""
        elite = Elite(
            solution=[1.0],
            objective=0.9,
            measures=[0.3, 0.7],
            metadata={"generation": 5}
        )
        d = elite.to_dict()

        assert d["solution"] == [1.0]
        assert d["objective"] == 0.9
        assert d["measures"] == [0.3, 0.7]
        assert d["metadata"]["generation"] == 5

    def test_from_dict(self):
        """Test creating from dict."""
        data = {
            "solution": [1.0, 2.0],
            "objective": 0.85,
            "measures": [0.4, 0.6],
            "metadata": {"id": 1}
        }
        elite = Elite.from_dict(data)

        assert elite.solution == [1.0, 2.0]
        assert elite.objective == 0.85


class TestArchiveAddStatus:
    """Tests for ArchiveAddStatus enum."""

    def test_add_statuses(self):
        """Test all add status values."""
        assert ArchiveAddStatus.NEW.value == "new"
        assert ArchiveAddStatus.IMPROVE.value == "improve"
        assert ArchiveAddStatus.NOT_ADDED.value == "not_added"


class TestArchiveStats:
    """Tests for ArchiveStats."""

    def test_update_stats(self):
        """Test updating stats from elites."""
        stats = ArchiveStats()
        elites = [
            Elite(solution=[1.0], objective=0.5, measures=[0.1, 0.1]),
            Elite(solution=[2.0], objective=0.8, measures=[0.5, 0.5]),
            Elite(solution=[3.0], objective=1.0, measures=[0.9, 0.9]),
        ]
        stats.update(elites, total_cells=100)

        assert stats.num_elites == 3
        assert stats.coverage == 0.03  # 3/100
        assert stats.qd_score == 2.3  # 0.5 + 0.8 + 1.0
        assert stats.obj_max == 1.0
        assert stats.obj_min == 0.5


class TestArchive:
    """Tests for Archive."""

    def test_archive_creation(self):
        """Test creating an archive."""
        archive = Archive(
            solution_dim=10,
            measure_dim=2,
            measure_bounds=[(0.0, 1.0), (0.0, 1.0)],
            cells_per_dim=5
        )
        assert archive.solution_dim == 10
        assert archive.measure_dim == 2
        assert archive.empty
        assert len(archive) == 0

    def test_add_new(self):
        """Test adding to empty cell."""
        archive = Archive(
            solution_dim=3,
            measure_dim=2,
            measure_bounds=[(0.0, 1.0), (0.0, 1.0)]
        )

        result = archive.add(
            solution=[1.0, 2.0, 3.0],
            objective=0.9,
            measures=[0.5, 0.5]
        )

        assert result.status == ArchiveAddStatus.NEW
        assert result.elite is not None
        assert len(archive) == 1

    def test_add_improve(self):
        """Test improving existing cell."""
        archive = Archive(
            solution_dim=2,
            measure_dim=2,
            measure_bounds=[(0.0, 1.0), (0.0, 1.0)]
        )

        # Add initial
        archive.add(solution=[1.0, 1.0], objective=0.5, measures=[0.5, 0.5])

        # Add better solution to same cell
        result = archive.add(
            solution=[2.0, 2.0],
            objective=0.9,
            measures=[0.5, 0.5]  # Same cell
        )

        assert result.status == ArchiveAddStatus.IMPROVE
        assert result.improvement == 0.4
        assert result.previous_objective == 0.5

    def test_add_not_added(self):
        """Test not adding worse solution."""
        archive = Archive(
            solution_dim=2,
            measure_dim=2,
            measure_bounds=[(0.0, 1.0), (0.0, 1.0)]
        )

        archive.add(solution=[1.0, 1.0], objective=0.9, measures=[0.5, 0.5])

        result = archive.add(
            solution=[2.0, 2.0],
            objective=0.5,  # Worse
            measures=[0.5, 0.5]
        )

        assert result.status == ArchiveAddStatus.NOT_ADDED
        assert len(archive) == 1  # Still just one

    def test_sample_elites(self):
        """Test sampling elites."""
        archive = Archive(
            solution_dim=1,
            measure_dim=2,
            measure_bounds=[(0.0, 1.0), (0.0, 1.0)]
        )

        # Add several elites in different cells
        for i in range(5):
            archive.add(
                solution=[float(i)],
                objective=i * 0.1,
                measures=[i * 0.2, i * 0.2]
            )

        sampled = archive.sample_elites(3)
        assert len(sampled) == 3

    def test_data(self):
        """Test getting all data."""
        archive = Archive(
            solution_dim=1,
            measure_dim=2,
            measure_bounds=[(0.0, 1.0), (0.0, 1.0)]
        )

        archive.add(solution=[1.0], objective=0.5, measures=[0.1, 0.1])
        archive.add(solution=[2.0], objective=0.7, measures=[0.9, 0.9])

        data = archive.data()
        assert len(data["solution"]) == 2
        assert len(data["objective"]) == 2

    def test_clear(self):
        """Test clearing archive."""
        archive = Archive(
            solution_dim=1,
            measure_dim=2,
            measure_bounds=[(0.0, 1.0), (0.0, 1.0)]
        )

        archive.add(solution=[1.0], objective=0.5, measures=[0.5, 0.5])
        assert len(archive) == 1

        archive.clear()
        assert archive.empty
        assert len(archive) == 0

    def test_stats(self):
        """Test archive statistics."""
        archive = Archive(
            solution_dim=1,
            measure_dim=2,
            measure_bounds=[(0.0, 1.0), (0.0, 1.0)],
            cells_per_dim=10
        )

        archive.add(solution=[1.0], objective=0.5, measures=[0.1, 0.1])
        archive.add(solution=[2.0], objective=0.9, measures=[0.9, 0.9])

        stats = archive.stats
        assert stats.num_elites == 2
        assert stats.qd_score == 1.4
        assert stats.obj_max == 0.9
        assert stats.obj_min == 0.5


# =============================================================================
# QUALITY-DIVERSITY THINKING TESTS
# =============================================================================


class TestThinkingElite:
    """Tests for ThinkingElite."""

    def test_thinking_elite_creation(self):
        """Test creating a thinking elite."""
        elite = ThinkingElite(
            thought_id="thought-1",
            content="Let me analyze this problem...",
            quality_score=0.85,
            diversity_measures=[0.3, 0.7]
        )

        assert elite.thought_id == "thought-1"
        assert elite.quality_score == 0.85
        assert elite.parent_id is None

    def test_to_elite(self):
        """Test converting to generic Elite."""
        thinking = ThinkingElite(
            thought_id="t1",
            content="Thinking content",
            quality_score=0.9,
            diversity_measures=[0.5, 0.5],
            parent_id="t0"
        )

        elite = thinking.to_elite()
        assert elite.objective == 0.9
        assert elite.measures == [0.5, 0.5]
        assert elite.metadata["thought_id"] == "t1"
        assert elite.metadata["parent_id"] == "t0"


class TestThinkingArchive:
    """Tests for ThinkingArchive."""

    def test_archive_creation(self):
        """Test creating a thinking archive."""
        archive = ThinkingArchive()
        assert len(archive) == 0

    def test_add_thought(self):
        """Test adding a thought."""
        archive = ThinkingArchive()

        result = archive.add_thought(
            thought_id="thought-1",
            content="First approach: use dynamic programming",
            quality_score=0.8,
            diversity_measures=[0.3, 0.7]
        )

        assert result.status == ArchiveAddStatus.NEW
        assert len(archive) == 1

    def test_get_thought(self):
        """Test getting a specific thought."""
        archive = ThinkingArchive()

        archive.add_thought(
            thought_id="t1",
            content="Approach 1",
            quality_score=0.7,
            diversity_measures=[0.2, 0.2]
        )

        thought = archive.get_thought("t1")
        assert thought is not None
        assert thought.content == "Approach 1"

        missing = archive.get_thought("nonexistent")
        assert missing is None

    def test_get_best_thought(self):
        """Test getting the best thought."""
        archive = ThinkingArchive()

        archive.add_thought("t1", "Low quality", 0.3, [0.1, 0.1])
        archive.add_thought("t2", "High quality", 0.9, [0.5, 0.5])
        archive.add_thought("t3", "Medium quality", 0.6, [0.9, 0.9])

        best = archive.get_best_thought()
        assert best is not None
        assert best.thought_id == "t2"
        assert best.quality_score == 0.9

    def test_sample_diverse_thoughts(self):
        """Test sampling diverse thoughts."""
        archive = ThinkingArchive(cells_per_dim=5)

        # Add thoughts in different cells
        archive.add_thought("t1", "Approach 1", 0.8, [0.1, 0.1])
        archive.add_thought("t2", "Approach 2", 0.7, [0.5, 0.5])
        archive.add_thought("t3", "Approach 3", 0.9, [0.9, 0.9])

        sampled = archive.sample_diverse_thoughts(2)
        assert len(sampled) <= 2
        for thought in sampled:
            assert isinstance(thought, ThinkingElite)

    def test_stats(self):
        """Test archive statistics."""
        archive = ThinkingArchive()

        archive.add_thought("t1", "A", 0.5, [0.2, 0.2])
        archive.add_thought("t2", "B", 0.9, [0.8, 0.8])

        stats = archive.stats
        assert stats.num_elites == 2
        assert stats.qd_score == 1.4

    def test_parent_tracking(self):
        """Test parent-child thought tracking."""
        archive = ThinkingArchive()

        archive.add_thought("root", "Initial thought", 0.5, [0.5, 0.5])
        archive.add_thought(
            "child",
            "Refined thought",
            0.7,
            [0.6, 0.6],
            parent_id="root"
        )

        child = archive.get_thought("child")
        assert child.parent_id == "root"


class TestV108Integration:
    """Integration tests for V10.8 patterns."""

    def test_span_with_scope(self):
        """Test using spans within scopes."""
        scope = Scope()
        span = Span(name="http-request")

        scope.data.span = span
        scope.set_context("trace", {
            "trace_id": span.context.trace_id,
            "span_id": span.context.span_id
        })

        span.set_attribute("http.method", "GET")
        span.add_event("request_received")

        assert scope.data.span is span
        assert "trace" in scope.data.contexts

    def test_processor_with_context_manager(self):
        """Test processor chain with context manager."""
        ctx_manager = ContextVarManager()
        chain = ProcessorChain()

        # Add context merging as first processor
        def merge_context(logger, method, event):
            return ctx_manager.merge_into(event)

        chain.add(merge_context)

        # Bind request context
        ctx_manager.bind(request_id="req-123", user_id=42)

        # Process event
        result = chain.process(None, "info", {"event": "user_action"})

        assert result["request_id"] == "req-123"
        assert result["user_id"] == 42

    def test_thinking_archive_exploration(self):
        """Test exploration pattern with thinking archive."""
        archive = ThinkingArchive(
            measure_bounds=[(0.0, 1.0), (0.0, 1.0)],
            cells_per_dim=3
        )

        # Simulate exploration loop
        thoughts = [
            ("t1", "Dynamic programming approach", 0.7, [0.2, 0.3]),
            ("t2", "Greedy algorithm approach", 0.6, [0.8, 0.2]),
            ("t3", "Divide and conquer approach", 0.8, [0.5, 0.8]),
            ("t4", "Improved DP with memoization", 0.9, [0.25, 0.35]),  # Near t1
        ]

        for thought_id, content, quality, measures in thoughts:
            archive.add_thought(thought_id, content, quality, measures)

        # Get diverse sample
        diverse = archive.sample_diverse_thoughts(3)
        assert len(diverse) >= 1

        # Best thought should be t4 or t3
        best = archive.get_best_thought()
        assert best.quality_score >= 0.8

    def test_scope_manager_with_breadcrumbs(self):
        """Test scope manager with breadcrumb trail."""
        manager = ScopeManager()

        # Set global context
        manager.get_global_scope().set_tag("app", "test-app")

        # Start request isolation
        iso = manager.get_isolation_scope()
        iso.set_user({"id": "user-123"})
        iso.add_breadcrumb(message="Request received", category="http")

        # Push transaction scope
        txn = manager.push_scope()
        txn.add_breadcrumb(message="Starting transaction", category="db")
        txn.set_tag("transaction", "create_order")

        # Get merged for error reporting
        merged = manager.get_merged_scope()
        event = merged.apply_to_event({"type": "error"})

        assert event["tags"]["app"] == "test-app"
        assert event["tags"]["transaction"] == "create_order"
        # Merged scope extends breadcrumbs from all scopes
        # iso has 1, txn (forked from iso) has 2 (inherited + new), so merged gets all
        assert len(event["breadcrumbs"]) >= 2


# Run tests if executed directly
if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
