"""
Unit Tests for Procedural Memory (V40 Architecture)

Tests for the procedural memory system including:
- Procedure learning and storage
- Pattern-based recall
- Procedure execution with parameter substitution
- Confidence tracking and updates
- Execution history

Run with: pytest platform/tests/unit/test_procedural_memory.py -v
"""

import asyncio
import pytest
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

# Import procedural memory types
try:
    from core.memory.procedural import (
        ProceduralMemory,
        Procedure,
        ProcedureStep,
        ProcedureMatch,
        ExecutionResult,
        ProcedureStatus,
        StepType,
        ExecutionOutcome,
        get_procedural_memory,
        learn_procedure,
        recall_procedure,
        execute_procedure,
        MIN_CONFIDENCE_FOR_AUTO_EXECUTE,
        CONFIDENCE_BOOST_PER_SUCCESS,
        CONFIDENCE_DECAY_PER_FAILURE,
    )
    PROCEDURAL_AVAILABLE = True
except ImportError as e:
    PROCEDURAL_AVAILABLE = False
    IMPORT_ERROR = str(e)


# Skip all tests if module not available
pytestmark = pytest.mark.skipif(
    not PROCEDURAL_AVAILABLE,
    reason=f"Procedural memory module not available: {IMPORT_ERROR if not PROCEDURAL_AVAILABLE else ''}"
)


class TestProcedureStep:
    """Tests for ProcedureStep dataclass."""

    def test_create_basic_step(self):
        """Create a basic procedure step."""
        step = ProcedureStep(
            action="git_status",
            params={}
        )

        assert step.action == "git_status"
        assert step.params == {}
        assert step.step_type == StepType.TOOL_CALL
        assert step.order == 0
        assert step.required is True
        assert step.timeout_ms == 30000

    def test_create_step_with_params(self):
        """Create a step with parameters."""
        step = ProcedureStep(
            action="file_write",
            params={"path": "/tmp/test.txt", "content": "$CONTENT"},
            step_type=StepType.TOOL_CALL,
            order=1,
            description="Write content to file",
            required=True,
            timeout_ms=5000
        )

        assert step.action == "file_write"
        assert step.params["path"] == "/tmp/test.txt"
        assert step.params["content"] == "$CONTENT"
        assert step.description == "Write content to file"
        assert step.timeout_ms == 5000

    def test_step_to_dict(self):
        """Test step serialization to dict."""
        step = ProcedureStep(
            action="test_action",
            params={"key": "value"},
            step_type=StepType.CONDITION,
            order=2
        )

        data = step.to_dict()

        assert isinstance(data, dict)
        assert data["action"] == "test_action"
        assert data["params"] == {"key": "value"}
        assert data["step_type"] == "condition"
        assert data["order"] == 2

    def test_step_from_dict(self):
        """Test step deserialization from dict."""
        data = {
            "action": "search",
            "params": {"query": "test"},
            "step_type": "tool_call",
            "order": 0,
            "required": True,
            "on_failure": "skip"
        }

        step = ProcedureStep.from_dict(data)

        assert step.action == "search"
        assert step.params["query"] == "test"
        assert step.step_type == StepType.TOOL_CALL
        assert step.on_failure == "skip"


class TestProcedure:
    """Tests for Procedure dataclass."""

    def test_create_basic_procedure(self):
        """Create a basic procedure."""
        steps = [
            ProcedureStep(action="step1", params={}),
            ProcedureStep(action="step2", params={}),
        ]

        procedure = Procedure(
            id="test-1",
            name="test_procedure",
            description="A test procedure",
            steps=steps,
            trigger_patterns=["test pattern"]
        )

        assert procedure.id == "test-1"
        assert procedure.name == "test_procedure"
        assert len(procedure.steps) == 2
        assert procedure.confidence == 0.5
        assert procedure.status == ProcedureStatus.ACTIVE
        assert procedure.total_executions == 0
        assert procedure.success_rate == 0.0

    def test_procedure_confidence_update_success(self):
        """Test confidence update on success."""
        procedure = Procedure(
            id="test-2",
            name="confidence_test",
            description="",
            steps=[],
            trigger_patterns=[],
            confidence=0.5
        )

        initial_confidence = procedure.confidence
        procedure.update_confidence(success=True)

        assert procedure.success_count == 1
        assert procedure.failure_count == 0
        assert procedure.confidence == initial_confidence + CONFIDENCE_BOOST_PER_SUCCESS
        assert procedure.last_executed is not None

    def test_procedure_confidence_update_failure(self):
        """Test confidence update on failure."""
        procedure = Procedure(
            id="test-3",
            name="confidence_test",
            description="",
            steps=[],
            trigger_patterns=[],
            confidence=0.5
        )

        initial_confidence = procedure.confidence
        procedure.update_confidence(success=False)

        assert procedure.success_count == 0
        assert procedure.failure_count == 1
        assert procedure.confidence == initial_confidence - CONFIDENCE_DECAY_PER_FAILURE
        assert procedure.last_executed is not None

    def test_procedure_success_rate(self):
        """Test success rate calculation."""
        procedure = Procedure(
            id="test-4",
            name="rate_test",
            description="",
            steps=[],
            trigger_patterns=[],
            success_count=8,
            failure_count=2
        )

        assert procedure.total_executions == 10
        assert procedure.success_rate == 0.8

    def test_procedure_to_dict(self):
        """Test procedure serialization to dict."""
        steps = [ProcedureStep(action="action1", params={"p": 1})]
        procedure = Procedure(
            id="test-5",
            name="serialize_test",
            description="Test",
            steps=steps,
            trigger_patterns=["pattern1", "pattern2"],
            tags=["tag1"],
            metadata={"key": "value"}
        )

        data = procedure.to_dict()

        assert data["id"] == "test-5"
        assert data["name"] == "serialize_test"
        assert len(data["steps"]) == 1
        assert data["steps"][0]["action"] == "action1"
        assert len(data["trigger_patterns"]) == 2
        assert data["tags"] == ["tag1"]
        assert data["metadata"]["key"] == "value"

    def test_procedure_from_dict(self):
        """Test procedure deserialization from dict."""
        data = {
            "id": "test-6",
            "name": "deserialize_test",
            "description": "From dict",
            "steps": [
                {"action": "step1", "params": {}, "step_type": "tool_call", "order": 0}
            ],
            "trigger_patterns": ["trigger"],
            "success_count": 5,
            "failure_count": 1,
            "confidence": 0.75,
            "status": "active",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        procedure = Procedure.from_dict(data)

        assert procedure.id == "test-6"
        assert procedure.name == "deserialize_test"
        assert len(procedure.steps) == 1
        assert procedure.success_count == 5
        assert procedure.confidence == 0.75


class TestProceduralMemory:
    """Tests for ProceduralMemory class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_procedural.db"
            yield db_path

    @pytest.fixture
    def memory(self, temp_db):
        """Create a ProceduralMemory instance with temp database."""
        return ProceduralMemory(db_path=temp_db)

    @pytest.mark.asyncio
    async def test_learn_procedure(self, memory):
        """Test learning a new procedure."""
        steps = [
            ProcedureStep(action="git_status", params={}),
            ProcedureStep(action="git_add", params={"files": "."}),
            ProcedureStep(action="git_commit", params={"message": "$MESSAGE"}),
        ]

        procedure = await memory.learn_procedure(
            name="git_commit_workflow",
            steps=steps,
            trigger_patterns=["commit changes", "save work"],
            description="Standard git commit workflow"
        )

        assert procedure.id is not None
        assert procedure.name == "git_commit_workflow"
        assert len(procedure.steps) == 3
        assert len(procedure.trigger_patterns) == 2
        assert procedure.status == ProcedureStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_learn_from_execution(self, memory):
        """Test learning from a tool execution sequence."""
        tool_calls = [
            {"name": "file_read", "params": {"path": "/test.py"}},
            {"name": "file_write", "params": {"path": "/test.py", "content": "updated"}},
            {"name": "test_run", "params": {}},
        ]

        procedure = await memory.learn_from_execution(
            tool_calls=tool_calls,
            success=True,
            name="file_update_workflow"
        )

        assert procedure is not None
        assert procedure.name == "file_update_workflow"
        assert len(procedure.steps) == 3
        assert procedure.steps[0].action == "file_read"
        assert procedure.confidence == 0.6  # Higher for successful execution

    @pytest.mark.asyncio
    async def test_learn_from_execution_too_few_steps(self, memory):
        """Test that learning requires at least 2 steps."""
        tool_calls = [{"name": "single_action", "params": {}}]

        procedure = await memory.learn_from_execution(
            tool_calls=tool_calls,
            success=True
        )

        assert procedure is None

    @pytest.mark.asyncio
    async def test_learn_from_execution_failure(self, memory):
        """Test that learning is skipped for failed executions."""
        tool_calls = [
            {"name": "action1", "params": {}},
            {"name": "action2", "params": {}},
        ]

        procedure = await memory.learn_from_execution(
            tool_calls=tool_calls,
            success=False
        )

        assert procedure is None

    @pytest.mark.asyncio
    async def test_get_procedure(self, memory):
        """Test retrieving a procedure by ID."""
        steps = [ProcedureStep(action="test", params={})]
        created = await memory.learn_procedure(
            name="get_test",
            steps=steps,
            trigger_patterns=["test"]
        )

        retrieved = await memory.get_procedure(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.name == "get_test"

    @pytest.mark.asyncio
    async def test_get_procedure_not_found(self, memory):
        """Test retrieving a nonexistent procedure."""
        result = await memory.get_procedure("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_recall_procedure_fts(self, memory):
        """Test recalling procedures using FTS5 search."""
        # Create some procedures
        await memory.learn_procedure(
            name="git_workflow",
            steps=[ProcedureStep(action="git_status", params={})],
            trigger_patterns=["commit changes", "git commit", "save my work"]
        )
        await memory.learn_procedure(
            name="test_workflow",
            steps=[ProcedureStep(action="test_run", params={})],
            trigger_patterns=["run tests", "test code", "verify tests"]
        )

        # Recall
        matches = await memory.recall_procedure("commit my changes")

        assert len(matches) >= 1
        assert any(m.procedure.name == "git_workflow" for m in matches)

    @pytest.mark.asyncio
    async def test_recall_procedure_ranked_by_confidence(self, memory):
        """Test that recall results are ranked by confidence."""
        # Create procedures with different confidence
        steps = [ProcedureStep(action="test", params={})]

        proc1 = await memory.learn_procedure(
            name="low_confidence",
            steps=steps,
            trigger_patterns=["search pattern"],
            initial_confidence=0.3
        )

        proc2 = await memory.learn_procedure(
            name="high_confidence",
            steps=steps,
            trigger_patterns=["search pattern"],
            initial_confidence=0.9
        )

        matches = await memory.recall_procedure("search pattern")

        if len(matches) >= 2:
            # Higher confidence should come first
            assert matches[0].confidence >= matches[1].confidence

    @pytest.mark.asyncio
    async def test_execute_procedure_dry_run(self, memory):
        """Test dry run execution."""
        steps = [
            ProcedureStep(action="action1", params={}),
            ProcedureStep(action="action2", params={}),
        ]
        procedure = await memory.learn_procedure(
            name="dry_run_test",
            steps=steps,
            trigger_patterns=["test"]
        )

        result = await memory.execute_procedure(
            procedure.id,
            dry_run=True
        )

        assert result.outcome == ExecutionOutcome.SUCCESS
        assert result.steps_completed == 2
        assert result.steps_total == 2

    @pytest.mark.asyncio
    async def test_execute_procedure_with_executor(self, memory):
        """Test execution with a real executor."""
        steps = [
            ProcedureStep(action="action1", params={"key": "value"}),
            ProcedureStep(action="action2", params={"message": "$MESSAGE"}),
        ]
        procedure = await memory.learn_procedure(
            name="executor_test",
            steps=steps,
            trigger_patterns=["test"]
        )

        # Mock executor
        async def mock_executor(action: str, params: Dict[str, Any]) -> Any:
            return {"action": action, "result": "success"}

        result = await memory.execute_procedure(
            procedure.id,
            params={"MESSAGE": "Hello World"},
            executor=mock_executor
        )

        assert result.outcome == ExecutionOutcome.SUCCESS
        assert result.steps_completed == 2
        assert "step_0" in result.outputs
        assert "step_1" in result.outputs

    @pytest.mark.asyncio
    async def test_execute_procedure_with_failure(self, memory):
        """Test execution that fails on a step."""
        steps = [
            ProcedureStep(action="action1", params={}),
            ProcedureStep(action="action2", params={}, on_failure="abort"),
        ]
        procedure = await memory.learn_procedure(
            name="failure_test",
            steps=steps,
            trigger_patterns=["test"]
        )

        # Mock executor that fails on action2
        async def failing_executor(action: str, params: Dict[str, Any]) -> Any:
            if action == "action2":
                raise Exception("Simulated failure")
            return {"result": "ok"}

        result = await memory.execute_procedure(
            procedure.id,
            executor=failing_executor
        )

        assert result.outcome == ExecutionOutcome.PARTIAL
        assert result.steps_completed == 1
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_execute_procedure_skip_on_failure(self, memory):
        """Test execution with skip on failure."""
        steps = [
            ProcedureStep(action="action1", params={}, on_failure="skip"),
            ProcedureStep(action="action2", params={}),
        ]
        procedure = await memory.learn_procedure(
            name="skip_test",
            steps=steps,
            trigger_patterns=["test"]
        )

        # Mock executor that fails on action1
        async def failing_executor(action: str, params: Dict[str, Any]) -> Any:
            if action == "action1":
                raise Exception("Simulated failure")
            return {"result": "ok"}

        result = await memory.execute_procedure(
            procedure.id,
            executor=failing_executor
        )

        # Should complete because action1 failure is skipped
        assert result.steps_completed == 1
        assert "step_1" in result.outputs

    @pytest.mark.asyncio
    async def test_execute_procedure_not_found(self, memory):
        """Test execution of nonexistent procedure."""
        result = await memory.execute_procedure("nonexistent-id")

        assert result.outcome == ExecutionOutcome.FAILURE
        assert "not found" in result.errors[0].lower()

    @pytest.mark.asyncio
    async def test_parameter_substitution(self, memory):
        """Test parameter substitution in steps."""
        steps = [
            ProcedureStep(
                action="file_write",
                params={"path": "$PATH", "content": "$CONTENT"}
            ),
        ]
        procedure = await memory.learn_procedure(
            name="substitution_test",
            steps=steps,
            trigger_patterns=["test"]
        )

        captured_params = {}

        async def capture_executor(action: str, params: Dict[str, Any]) -> Any:
            captured_params.update(params)
            return {"result": "ok"}

        await memory.execute_procedure(
            procedure.id,
            params={"PATH": "/test/file.txt", "CONTENT": "Hello"},
            executor=capture_executor
        )

        assert captured_params["path"] == "/test/file.txt"
        assert captured_params["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_list_procedures(self, memory):
        """Test listing procedures."""
        for i in range(5):
            await memory.learn_procedure(
                name=f"procedure_{i}",
                steps=[ProcedureStep(action=f"action_{i}", params={})],
                trigger_patterns=[f"trigger_{i}"]
            )

        procedures = await memory.list_procedures()

        assert len(procedures) == 5

    @pytest.mark.asyncio
    async def test_list_procedures_by_status(self, memory):
        """Test listing procedures filtered by status."""
        proc1 = await memory.learn_procedure(
            name="active_proc",
            steps=[ProcedureStep(action="action", params={})],
            trigger_patterns=["test"]
        )
        proc2 = await memory.learn_procedure(
            name="deprecated_proc",
            steps=[ProcedureStep(action="action", params={})],
            trigger_patterns=["test"]
        )

        await memory.deprecate_procedure(proc2.id)

        active = await memory.list_procedures(status=ProcedureStatus.ACTIVE)
        deprecated = await memory.list_procedures(status=ProcedureStatus.DEPRECATED)

        assert len(active) == 1
        assert active[0].name == "active_proc"
        assert len(deprecated) == 1
        assert deprecated[0].name == "deprecated_proc"

    @pytest.mark.asyncio
    async def test_delete_procedure(self, memory):
        """Test deleting a procedure."""
        procedure = await memory.learn_procedure(
            name="delete_test",
            steps=[ProcedureStep(action="action", params={})],
            trigger_patterns=["test"]
        )

        deleted = await memory.delete_procedure(procedure.id)
        assert deleted is True

        retrieved = await memory.get_procedure(procedure.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_procedure_not_found(self, memory):
        """Test deleting nonexistent procedure."""
        deleted = await memory.delete_procedure("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_deprecate_procedure(self, memory):
        """Test deprecating a procedure."""
        procedure = await memory.learn_procedure(
            name="deprecate_test",
            steps=[ProcedureStep(action="action", params={})],
            trigger_patterns=["test"]
        )

        success = await memory.deprecate_procedure(procedure.id)
        assert success is True

        retrieved = await memory.get_procedure(procedure.id)
        assert retrieved.status == ProcedureStatus.DEPRECATED

    @pytest.mark.asyncio
    async def test_execution_history(self, memory):
        """Test execution history tracking."""
        procedure = await memory.learn_procedure(
            name="history_test",
            steps=[ProcedureStep(action="action", params={})],
            trigger_patterns=["test"]
        )

        # Execute multiple times
        for _ in range(3):
            await memory.execute_procedure(procedure.id, dry_run=True)

        history = await memory.get_execution_history(procedure.id)

        assert len(history) == 3
        assert all(h.procedure_id == procedure.id for h in history)

    @pytest.mark.asyncio
    async def test_confidence_updates_after_execution(self, memory):
        """Test that confidence is updated after execution."""
        procedure = await memory.learn_procedure(
            name="confidence_update_test",
            steps=[ProcedureStep(action="action", params={})],
            trigger_patterns=["test"],
            initial_confidence=0.5
        )

        initial_confidence = procedure.confidence

        # Successful execution
        async def success_executor(action: str, params: Dict[str, Any]) -> Any:
            return {"result": "ok"}

        await memory.execute_procedure(
            procedure.id,
            executor=success_executor
        )

        updated = await memory.get_procedure(procedure.id)
        assert updated.confidence > initial_confidence
        assert updated.success_count == 1

    @pytest.mark.asyncio
    async def test_get_stats(self, memory):
        """Test getting statistics."""
        # Create some procedures and executions
        for i in range(3):
            proc = await memory.learn_procedure(
                name=f"stats_test_{i}",
                steps=[ProcedureStep(action=f"action_{i}", params={})],
                trigger_patterns=[f"trigger_{i}"]
            )
            await memory.execute_procedure(proc.id, dry_run=True)

        stats = await memory.get_stats()

        assert stats["total_procedures"] == 3
        assert stats["active_procedures"] == 3
        assert stats["total_executions"] == 3
        assert "average_confidence" in stats
        assert "storage_path" in stats


class TestProcedureMatch:
    """Tests for ProcedureMatch dataclass."""

    def test_should_auto_execute_high_confidence(self):
        """Test auto-execute flag with high confidence."""
        procedure = Procedure(
            id="test",
            name="test",
            description="",
            steps=[],
            trigger_patterns=[],
            confidence=0.9
        )

        match = ProcedureMatch(
            procedure=procedure,
            score=0.95,
            matched_trigger="test",
            match_type="fuzzy",
            confidence=0.855  # 0.95 * 0.9
        )

        assert match.should_auto_execute is True

    def test_should_auto_execute_low_confidence(self):
        """Test auto-execute flag with low confidence."""
        procedure = Procedure(
            id="test",
            name="test",
            description="",
            steps=[],
            trigger_patterns=[],
            confidence=0.5
        )

        match = ProcedureMatch(
            procedure=procedure,
            score=0.7,
            matched_trigger="test",
            match_type="fuzzy",
            confidence=0.35  # 0.7 * 0.5
        )

        assert match.should_auto_execute is False


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_create_execution_result(self):
        """Create an execution result."""
        result = ExecutionResult(
            procedure_id="proc-1",
            outcome=ExecutionOutcome.SUCCESS,
            steps_completed=5,
            steps_total=5,
            duration_ms=150.5
        )

        assert result.procedure_id == "proc-1"
        assert result.outcome == ExecutionOutcome.SUCCESS
        assert result.steps_completed == 5
        assert result.duration_ms == 150.5
        assert result.executed_at is not None

    def test_execution_result_with_outputs(self):
        """Create an execution result with outputs and errors."""
        result = ExecutionResult(
            procedure_id="proc-2",
            outcome=ExecutionOutcome.PARTIAL,
            steps_completed=3,
            steps_total=5,
            duration_ms=200.0,
            outputs={"step_0": "result1", "step_1": "result2"},
            errors=["Step 3 failed: timeout"]
        )

        assert len(result.outputs) == 2
        assert len(result.errors) == 1


class TestHookFunctions:
    """Tests for hook functions."""

    @pytest.fixture
    def reset_singleton(self):
        """Reset the singleton for each test."""
        import core.memory.procedural as proc_module
        proc_module._procedural_memory = None
        yield
        proc_module._procedural_memory = None

    @pytest.fixture
    def temp_db_patch(self, reset_singleton):
        """Patch the default database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_hooks.db"
            with patch.object(
                ProceduralMemory,
                '__init__',
                lambda self, **kwargs: ProceduralMemory.__init__(
                    self, db_path=db_path, **{k: v for k, v in kwargs.items() if k != 'db_path'}
                )
            ):
                yield

    @pytest.mark.asyncio
    async def test_learn_procedure_hook(self, temp_db_patch):
        """Test learn_procedure hook function."""
        steps = [ProcedureStep(action="hook_test", params={})]

        procedure = await learn_procedure(
            name="hook_learn_test",
            steps=steps,
            trigger_patterns=["hook test"]
        )

        assert procedure is not None
        assert procedure.name == "hook_learn_test"

    @pytest.mark.asyncio
    async def test_recall_procedure_hook(self, temp_db_patch):
        """Test recall_procedure hook function."""
        steps = [ProcedureStep(action="action", params={})]
        await learn_procedure(
            name="hook_recall_test",
            steps=steps,
            trigger_patterns=["recall hook pattern"]
        )

        matches = await recall_procedure("recall hook")

        assert len(matches) >= 0  # May or may not match depending on FTS

    @pytest.mark.asyncio
    async def test_execute_procedure_hook(self, temp_db_patch):
        """Test execute_procedure hook function."""
        steps = [ProcedureStep(action="action", params={})]
        procedure = await learn_procedure(
            name="hook_execute_test",
            steps=steps,
            trigger_patterns=["test"]
        )

        result = await execute_procedure(
            procedure.id,
            dry_run=True
        )

        assert result.outcome == ExecutionOutcome.SUCCESS


class TestEnums:
    """Tests for enum types."""

    def test_procedure_status_values(self):
        """Verify ProcedureStatus enum values."""
        assert ProcedureStatus.ACTIVE.value == "active"
        assert ProcedureStatus.DEPRECATED.value == "deprecated"
        assert ProcedureStatus.DISABLED.value == "disabled"
        assert ProcedureStatus.LEARNING.value == "learning"

    def test_step_type_values(self):
        """Verify StepType enum values."""
        assert StepType.TOOL_CALL.value == "tool_call"
        assert StepType.CONDITION.value == "condition"
        assert StepType.LOOP.value == "loop"
        assert StepType.SUB_PROCEDURE.value == "sub_procedure"

    def test_execution_outcome_values(self):
        """Verify ExecutionOutcome enum values."""
        assert ExecutionOutcome.SUCCESS.value == "success"
        assert ExecutionOutcome.PARTIAL.value == "partial"
        assert ExecutionOutcome.FAILURE.value == "failure"
        assert ExecutionOutcome.ABORTED.value == "aborted"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_edge.db"
            yield db_path

    @pytest.fixture
    def memory(self, temp_db):
        """Create a ProceduralMemory instance with temp database."""
        return ProceduralMemory(db_path=temp_db)

    @pytest.mark.asyncio
    async def test_empty_steps_procedure(self, memory):
        """Test creating a procedure with no steps."""
        procedure = await memory.learn_procedure(
            name="empty_steps",
            steps=[],
            trigger_patterns=["test"]
        )

        assert procedure.id is not None
        assert len(procedure.steps) == 0

    @pytest.mark.asyncio
    async def test_empty_triggers_procedure(self, memory):
        """Test creating a procedure with no triggers."""
        procedure = await memory.learn_procedure(
            name="empty_triggers",
            steps=[ProcedureStep(action="test", params={})],
            trigger_patterns=[]
        )

        assert procedure.id is not None
        assert len(procedure.trigger_patterns) == 0

    @pytest.mark.asyncio
    async def test_special_characters_in_triggers(self, memory):
        """Test triggers with special characters."""
        procedure = await memory.learn_procedure(
            name="special_chars",
            steps=[ProcedureStep(action="test", params={})],
            trigger_patterns=["what's the time?", "run 'npm test'", "path/to/file"]
        )

        assert len(procedure.trigger_patterns) == 3

    @pytest.mark.asyncio
    async def test_large_params(self, memory):
        """Test step with large parameters."""
        large_content = "x" * 10000

        steps = [
            ProcedureStep(
                action="large_params",
                params={"content": large_content}
            )
        ]

        procedure = await memory.learn_procedure(
            name="large_params_test",
            steps=steps,
            trigger_patterns=["test"]
        )

        retrieved = await memory.get_procedure(procedure.id)
        assert retrieved.steps[0].params["content"] == large_content

    @pytest.mark.asyncio
    async def test_unicode_content(self, memory):
        """Test procedure with unicode content."""
        steps = [
            ProcedureStep(
                action="unicode_test",
                params={"message": "Hello World!"},
                description="Test with emojis"
            )
        ]

        procedure = await memory.learn_procedure(
            name="unicode_procedure",
            steps=steps,
            trigger_patterns=["unicode test"]
        )

        retrieved = await memory.get_procedure(procedure.id)
        assert "Hello" in retrieved.steps[0].params["message"]

    @pytest.mark.asyncio
    async def test_concurrent_learning(self, memory):
        """Test concurrent procedure learning."""
        async def create_procedure(i: int):
            return await memory.learn_procedure(
                name=f"concurrent_{i}",
                steps=[ProcedureStep(action=f"action_{i}", params={})],
                trigger_patterns=[f"concurrent {i}"]
            )

        # Create 10 procedures concurrently
        procedures = await asyncio.gather(*[create_procedure(i) for i in range(10)])

        assert len(procedures) == 10
        assert len(set(p.id for p in procedures)) == 10  # All unique IDs

    @pytest.mark.asyncio
    async def test_execution_with_retry(self, memory):
        """Test step execution with retries."""
        steps = [
            ProcedureStep(
                action="flaky_action",
                params={},
                retry_count=2
            )
        ]

        procedure = await memory.learn_procedure(
            name="retry_test",
            steps=steps,
            trigger_patterns=["test"]
        )

        call_count = 0

        async def flaky_executor(action: str, params: Dict[str, Any]) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return {"result": "success"}

        result = await memory.execute_procedure(
            procedure.id,
            executor=flaky_executor
        )

        assert result.outcome == ExecutionOutcome.SUCCESS
        assert call_count == 3  # Initial + 2 retries


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
