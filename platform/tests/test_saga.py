"""
Tests for Saga Orchestration Pattern - V36 Architecture

Tests cover:
- Basic saga execution
- Compensation on failure
- State persistence and recovery
- Idempotency keys
- Metrics tracking
- Timeout handling
- Retry logic
"""

import asyncio
import os
import tempfile
import time
from typing import Any, Dict

import pytest

from core.orchestration.saga import (
    Saga,
    SagaMetrics,
    SagaOrchestrator,
    SagaResult,
    SagaState,
    SagaStateStore,
    SagaStep,
    StepResult,
    StepState,
    example_order_processing_saga,
)


class TestSagaStep:
    """Tests for SagaStep dataclass."""

    def test_step_creation(self):
        """Test basic step creation."""
        async def execute(ctx: Dict[str, Any]) -> str:
            return "executed"

        async def compensate(ctx: Dict[str, Any], result: Any) -> None:
            pass

        step = SagaStep(
            name="test_step",
            execute=execute,
            compensate=compensate,
            timeout_seconds=10.0,
        )

        assert step.name == "test_step"
        assert step.timeout_seconds == 10.0
        assert step.idempotency_key is not None

    def test_step_custom_idempotency_key(self):
        """Test step with custom idempotency key."""
        async def execute(ctx: Dict[str, Any]) -> str:
            return "executed"

        async def compensate(ctx: Dict[str, Any], result: Any) -> None:
            pass

        step = SagaStep(
            name="test_step",
            execute=execute,
            compensate=compensate,
            idempotency_key="custom-key-123",
        )

        assert step.idempotency_key == "custom-key-123"


class TestSaga:
    """Tests for Saga dataclass."""

    def test_saga_creation(self):
        """Test basic saga creation."""
        async def execute(ctx: Dict[str, Any]) -> str:
            return "executed"

        async def compensate(ctx: Dict[str, Any], result: Any) -> None:
            pass

        saga = Saga(
            saga_id="test-saga-001",
            steps=[
                SagaStep(name="step1", execute=execute, compensate=compensate),
            ],
            context={"key": "value"},
        )

        assert saga.saga_id == "test-saga-001"
        assert len(saga.steps) == 1
        assert saga.context == {"key": "value"}

    def test_saga_auto_id(self):
        """Test saga with auto-generated ID."""
        async def execute(ctx: Dict[str, Any]) -> str:
            return "executed"

        async def compensate(ctx: Dict[str, Any], result: Any) -> None:
            pass

        saga = Saga(
            saga_id="",
            steps=[SagaStep(name="step1", execute=execute, compensate=compensate)],
        )

        assert saga.saga_id != ""
        assert len(saga.saga_id) > 0

    def test_saga_idempotency_key_generation(self):
        """Test deterministic idempotency key generation."""
        async def execute(ctx: Dict[str, Any]) -> str:
            return "executed"

        async def compensate(ctx: Dict[str, Any], result: Any) -> None:
            pass

        saga1 = Saga(
            saga_id="test-saga",
            steps=[SagaStep(name="step1", execute=execute, compensate=compensate)],
            context={"query": "test"},
        )

        saga2 = Saga(
            saga_id="test-saga",
            steps=[SagaStep(name="step1", execute=execute, compensate=compensate)],
            context={"query": "test"},
        )

        assert saga1.generate_idempotency_key() == saga2.generate_idempotency_key()


class TestStepResult:
    """Tests for StepResult dataclass."""

    def test_step_result_duration(self):
        """Test step duration calculation."""
        result = StepResult(
            step_name="test_step",
            state=StepState.COMPLETED,
            started_at=1000.0,
            completed_at=1001.5,
        )

        assert result.duration_ms == 1500.0

    def test_step_result_duration_incomplete(self):
        """Test step duration when incomplete."""
        result = StepResult(
            step_name="test_step",
            state=StepState.EXECUTING,
            started_at=1000.0,
        )

        assert result.duration_ms == 0.0

    def test_step_result_to_dict(self):
        """Test serialization to dict."""
        result = StepResult(
            step_name="test_step",
            state=StepState.COMPLETED,
            result={"data": "value"},
            started_at=1000.0,
            completed_at=1001.0,
        )

        d = result.to_dict()
        assert d["step_name"] == "test_step"
        assert d["state"] == "COMPLETED"
        assert d["result"] == {"data": "value"}


class TestSagaMetrics:
    """Tests for SagaMetrics."""

    def test_metrics_recording(self):
        """Test metrics recording."""
        metrics = SagaMetrics()

        metrics.record_saga_complete(success=True, duration_ms=100.0, compensated=False)
        metrics.record_saga_complete(success=False, duration_ms=200.0, compensated=True)

        assert metrics.total_sagas == 2
        assert metrics.successful_sagas == 1
        assert metrics.failed_sagas == 1
        assert metrics.compensation_count == 1

    def test_metrics_rates(self):
        """Test success and compensation rates."""
        metrics = SagaMetrics()

        metrics.record_saga_complete(success=True, duration_ms=100.0, compensated=False)
        metrics.record_saga_complete(success=True, duration_ms=100.0, compensated=False)
        metrics.record_saga_complete(success=False, duration_ms=100.0, compensated=True)
        metrics.record_saga_complete(success=False, duration_ms=100.0, compensated=True)

        assert metrics.success_rate == 0.5
        assert metrics.compensation_rate == 0.5

    def test_step_metrics(self):
        """Test step duration tracking."""
        metrics = SagaMetrics()

        metrics.record_step("step1", 100.0)
        metrics.record_step("step1", 200.0)
        metrics.record_step("step1", 150.0)

        stats = metrics.get_step_stats("step1")
        assert stats["count"] == 3
        assert stats["avg_ms"] == 150.0
        assert stats["min_ms"] == 100.0
        assert stats["max_ms"] == 200.0


class TestSagaStateStore:
    """Tests for SQLite state persistence."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_saga_persistence(self, temp_db):
        """Test saga state persistence."""
        store = SagaStateStore(db_path=temp_db)

        async def execute(ctx: Dict[str, Any]) -> str:
            return "executed"

        async def compensate(ctx: Dict[str, Any], result: Any) -> None:
            pass

        saga = Saga(
            saga_id="persist-test",
            steps=[SagaStep(name="step1", execute=execute, compensate=compensate)],
            context={"key": "value"},
        )

        await store.save_saga(saga, SagaState.RUNNING, started_at=time.time())

        retrieved = await store.get_saga("persist-test")
        assert retrieved is not None
        assert retrieved["saga_id"] == "persist-test"
        assert retrieved["state"] == "RUNNING"
        assert retrieved["context"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_step_persistence(self, temp_db):
        """Test step state persistence."""
        store = SagaStateStore(db_path=temp_db)

        async def execute(ctx: Dict[str, Any]) -> str:
            return "executed"

        async def compensate(ctx: Dict[str, Any], result: Any) -> None:
            pass

        saga = Saga(
            saga_id="step-persist-test",
            steps=[SagaStep(name="step1", execute=execute, compensate=compensate)],
        )

        await store.save_saga(saga, SagaState.RUNNING, started_at=time.time())

        step_result = StepResult(
            step_name="step1",
            state=StepState.COMPLETED,
            result={"data": "value"},
            started_at=time.time(),
            completed_at=time.time(),
        )

        await store.save_step("step-persist-test", 0, step_result)

        retrieved = await store.get_saga("step-persist-test")
        assert len(retrieved["steps"]) == 1
        assert retrieved["steps"][0]["step_name"] == "step1"

    @pytest.mark.asyncio
    async def test_idempotency(self, temp_db):
        """Test idempotency key storage and retrieval."""
        store = SagaStateStore(db_path=temp_db)

        await store.store_idempotency(
            "test-key",
            "saga-001",
            "step1",
            {"result": "cached"},
            ttl_seconds=3600.0,
        )

        cached = await store.check_idempotency("test-key")
        assert cached is not None
        assert cached["saga_id"] == "saga-001"
        assert cached["result"] == {"result": "cached"}


class TestSagaOrchestrator:
    """Tests for SagaOrchestrator."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_successful_saga_execution(self, temp_db):
        """Test successful saga execution."""
        orchestrator = SagaOrchestrator(db_path=temp_db)

        execution_order = []

        async def step1_execute(ctx: Dict[str, Any]) -> str:
            execution_order.append("step1_execute")
            return "step1_result"

        async def step1_compensate(ctx: Dict[str, Any], result: Any) -> None:
            execution_order.append("step1_compensate")

        async def step2_execute(ctx: Dict[str, Any]) -> str:
            execution_order.append("step2_execute")
            return "step2_result"

        async def step2_compensate(ctx: Dict[str, Any], result: Any) -> None:
            execution_order.append("step2_compensate")

        saga = Saga(
            saga_id="success-test",
            steps=[
                SagaStep(name="step1", execute=step1_execute, compensate=step1_compensate),
                SagaStep(name="step2", execute=step2_execute, compensate=step2_compensate),
            ],
        )

        result = await orchestrator.execute(saga)

        assert result.success
        assert result.state == SagaState.COMPLETED
        assert result.completed_steps == ["step1", "step2"]
        assert execution_order == ["step1_execute", "step2_execute"]
        assert len(result.compensated_steps) == 0

    @pytest.mark.asyncio
    async def test_saga_compensation_on_failure(self, temp_db):
        """Test automatic compensation when a step fails."""
        orchestrator = SagaOrchestrator(db_path=temp_db)

        execution_order = []

        async def step1_execute(ctx: Dict[str, Any]) -> str:
            execution_order.append("step1_execute")
            return "step1_result"

        async def step1_compensate(ctx: Dict[str, Any], result: Any) -> None:
            execution_order.append("step1_compensate")

        async def step2_execute(ctx: Dict[str, Any]) -> str:
            execution_order.append("step2_execute")
            raise ValueError("Step 2 failed!")

        async def step2_compensate(ctx: Dict[str, Any], result: Any) -> None:
            execution_order.append("step2_compensate")

        saga = Saga(
            saga_id="compensation-test",
            steps=[
                SagaStep(name="step1", execute=step1_execute, compensate=step1_compensate),
                SagaStep(name="step2", execute=step2_execute, compensate=step2_compensate),
            ],
        )

        result = await orchestrator.execute(saga)

        assert not result.success
        assert result.state == SagaState.FAILED
        assert result.failed_at_step == 1
        assert result.completed_steps == ["step1"]
        assert result.compensated_steps == ["step1"]
        assert "Step 2 failed" in result.error
        assert execution_order == ["step1_execute", "step2_execute", "step1_compensate"]

    @pytest.mark.asyncio
    async def test_saga_timeout_handling(self, temp_db):
        """Test timeout handling."""
        orchestrator = SagaOrchestrator(db_path=temp_db)

        async def slow_execute(ctx: Dict[str, Any]) -> str:
            await asyncio.sleep(5.0)
            return "never_reached"

        async def compensate(ctx: Dict[str, Any], result: Any) -> None:
            pass

        saga = Saga(
            saga_id="timeout-test",
            steps=[
                SagaStep(
                    name="slow_step",
                    execute=slow_execute,
                    compensate=compensate,
                    timeout_seconds=0.1,
                ),
            ],
        )

        result = await orchestrator.execute(saga)

        assert not result.success
        assert result.failed_at_step == 0
        assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_saga_retry_logic(self, temp_db):
        """Test retry on failure."""
        orchestrator = SagaOrchestrator(db_path=temp_db)

        attempt_count = 0

        async def flaky_execute(ctx: Dict[str, Any]) -> str:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError(f"Attempt {attempt_count} failed")
            return "success_on_third"

        async def compensate(ctx: Dict[str, Any], result: Any) -> None:
            pass

        saga = Saga(
            saga_id="retry-test",
            steps=[
                SagaStep(
                    name="flaky_step",
                    execute=flaky_execute,
                    compensate=compensate,
                    retry_count=2,
                    retry_delay_seconds=0.01,
                ),
            ],
        )

        result = await orchestrator.execute(saga)

        assert result.success
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_saga_context_sharing(self, temp_db):
        """Test context sharing between steps."""
        orchestrator = SagaOrchestrator(db_path=temp_db)

        async def step1_execute(ctx: Dict[str, Any]) -> Dict[str, Any]:
            return {"step1_data": "from_step1"}

        async def step2_execute(ctx: Dict[str, Any]) -> Dict[str, Any]:
            # Should have access to step1 result
            step1_result = ctx.get("_result_step1", {})
            return {"step2_data": step1_result.get("step1_data", "not_found")}

        async def compensate(ctx: Dict[str, Any], result: Any) -> None:
            pass

        saga = Saga(
            saga_id="context-test",
            steps=[
                SagaStep(name="step1", execute=step1_execute, compensate=compensate),
                SagaStep(name="step2", execute=step2_execute, compensate=compensate),
            ],
            context={"initial": "value"},
        )

        result = await orchestrator.execute(saga)

        assert result.success
        assert len(result.step_results) == 2
        step2_result = result.step_results[1].result
        assert step2_result["step2_data"] == "from_step1"

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, temp_db):
        """Test metrics are properly tracked."""
        orchestrator = SagaOrchestrator(db_path=temp_db)

        async def execute(ctx: Dict[str, Any]) -> str:
            return "done"

        async def compensate(ctx: Dict[str, Any], result: Any) -> None:
            pass

        saga = Saga(
            saga_id="metrics-test",
            steps=[
                SagaStep(name="step1", execute=execute, compensate=compensate),
            ],
        )

        await orchestrator.execute(saga)

        metrics = orchestrator.get_metrics()
        assert metrics["total_sagas"] == 1
        assert metrics["successful_sagas"] == 1
        assert metrics["total_steps_executed"] == 1

    @pytest.mark.asyncio
    async def test_example_order_saga(self, temp_db):
        """Test the example order processing saga."""
        orchestrator = SagaOrchestrator(db_path=temp_db)

        saga = await example_order_processing_saga()
        result = await orchestrator.execute(saga)

        assert result.success
        assert result.state == SagaState.COMPLETED
        assert len(result.completed_steps) == 3
        assert "reserve_inventory" in result.completed_steps
        assert "charge_payment" in result.completed_steps
        assert "confirm_order" in result.completed_steps


class TestSagaOrchestratorNoPersistence:
    """Tests for SagaOrchestrator without persistence."""

    @pytest.mark.asyncio
    async def test_execution_without_persistence(self):
        """Test saga execution without database persistence."""
        orchestrator = SagaOrchestrator(enable_persistence=False)

        async def execute(ctx: Dict[str, Any]) -> str:
            return "done"

        async def compensate(ctx: Dict[str, Any], result: Any) -> None:
            pass

        saga = Saga(
            saga_id="no-persist-test",
            steps=[
                SagaStep(name="step1", execute=execute, compensate=compensate),
            ],
        )

        result = await orchestrator.execute(saga)

        assert result.success
        assert result.state == SagaState.COMPLETED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
