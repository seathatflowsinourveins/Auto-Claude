#!/usr/bin/env python3
"""
Tests for the Orchestration Observability Layer (V33.10).

Tests verify Opik integration for tracing agent execution and coordination.
Tests are skipped if Opik SDK is not available.

Run with: pytest core/tests/test_orchestration_observability.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import pytest

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Test: Import Availability
# =============================================================================

class TestImportAvailability:
    """Test that observability modules can be imported."""

    def test_observability_module_importable(self):
        """Verify orchestration_observability module imports correctly."""
        from core.orchestration import orchestration_observability
        assert orchestration_observability is not None

    def test_opik_availability_flag(self):
        """Verify OPIK_AVAILABLE flag is boolean."""
        from core.orchestration.orchestration_observability import OPIK_AVAILABLE
        assert isinstance(OPIK_AVAILABLE, bool)

    def test_evaluator_availability_flag(self):
        """Verify EVALUATOR_AVAILABLE flag is boolean."""
        from core.orchestration.orchestration_observability import EVALUATOR_AVAILABLE
        assert isinstance(EVALUATOR_AVAILABLE, bool)

    def test_combined_availability_flag(self):
        """Verify ORCHESTRATION_OBSERVABILITY_AVAILABLE flag."""
        from core.orchestration.orchestration_observability import (
            ORCHESTRATION_OBSERVABILITY_AVAILABLE,
        )
        assert isinstance(ORCHESTRATION_OBSERVABILITY_AVAILABLE, bool)

    def test_orchestration_exports_observability(self):
        """Verify orchestration __init__ exports observability symbols."""
        from core.orchestration import (
            ORCHESTRATION_OBSERVABILITY_AVAILABLE,
            OPIK_AVAILABLE,
            EVALUATOR_AVAILABLE,
        )
        assert isinstance(ORCHESTRATION_OBSERVABILITY_AVAILABLE, bool)
        assert isinstance(OPIK_AVAILABLE, bool)
        assert isinstance(EVALUATOR_AVAILABLE, bool)


# =============================================================================
# Test: Configuration Classes
# =============================================================================

class TestObservabilityConfig:
    """Test ObservabilityConfig dataclass structure."""

    def test_default_config(self):
        """Test ObservabilityConfig with defaults."""
        from core.orchestration.orchestration_observability import ObservabilityConfig

        config = ObservabilityConfig()
        assert config.project_name == "unleash-orchestration"
        assert config.trace_agent_execution is True
        assert config.trace_coordination is True
        assert config.trace_ralph_iteration is True
        assert config.evaluate_outputs is True
        assert "orchestration" in config.tags

    def test_custom_config(self):
        """Test ObservabilityConfig with custom values."""
        from core.orchestration.orchestration_observability import ObservabilityConfig

        config = ObservabilityConfig(
            project_name="test-project",
            trace_coordination=False,
            evaluate_outputs=False,
            tags=["test", "custom"],
        )
        assert config.project_name == "test-project"
        assert config.trace_coordination is False
        assert config.evaluate_outputs is False
        assert config.tags == ["test", "custom"]


class TestTraceLevel:
    """Test TraceLevel enum."""

    def test_trace_level_values(self):
        """Test trace level enum values."""
        from core.orchestration.orchestration_observability import TraceLevel

        assert TraceLevel.MINIMAL.value == "minimal"
        assert TraceLevel.STANDARD.value == "standard"
        assert TraceLevel.DETAILED.value == "detailed"
        assert TraceLevel.DEBUG.value == "debug"


# =============================================================================
# Test: ObservableOrchestrator Class
# =============================================================================

class TestObservableOrchestratorBasic:
    """Test ObservableOrchestrator basic functionality (no API key required)."""

    def test_orchestrator_creation(self):
        """Test creating ObservableOrchestrator."""
        from core.orchestration.orchestration_observability import (
            ObservableOrchestrator,
            ObservabilityConfig,
        )

        config = ObservabilityConfig()
        orchestrator = ObservableOrchestrator(config=config)
        assert orchestrator is not None
        assert orchestrator.config == config
        assert orchestrator.is_initialized is False

    def test_orchestrator_default_config(self):
        """Test ObservableOrchestrator with default config."""
        from core.orchestration.orchestration_observability import ObservableOrchestrator

        orchestrator = ObservableOrchestrator()
        assert orchestrator.config.project_name == "unleash-orchestration"

    def test_opik_available_property(self):
        """Test opik_available property."""
        from core.orchestration.orchestration_observability import (
            ObservableOrchestrator,
            OPIK_AVAILABLE,
        )

        orchestrator = ObservableOrchestrator()
        assert orchestrator.opik_available == OPIK_AVAILABLE

    def test_get_stats(self):
        """Test get_stats method."""
        from core.orchestration.orchestration_observability import ObservableOrchestrator

        orchestrator = ObservableOrchestrator()
        stats = orchestrator.get_stats()
        assert "total_traces" in stats
        assert "successful_traces" in stats
        assert "failed_traces" in stats
        assert "total_agent_calls" in stats
        assert "evaluation_count" in stats

    def test_agents_property_before_init(self):
        """Test agents property before initialization."""
        from core.orchestration.orchestration_observability import ObservableOrchestrator

        orchestrator = ObservableOrchestrator()
        agents = orchestrator.agents
        assert isinstance(agents, dict)


# =============================================================================
# Test: Factory Functions
# =============================================================================

class TestFactoryFunctions:
    """Test factory functions."""

    def test_get_observable_orchestrator_sync(self):
        """Test synchronous getter (uninitialized)."""
        from core.orchestration.orchestration_observability import (
            get_observable_orchestrator_sync,
        )

        orchestrator = get_observable_orchestrator_sync()
        assert orchestrator is not None
        assert orchestrator.is_initialized is False

    def test_get_observable_orchestrator_sync_with_config(self):
        """Test synchronous getter with custom config."""
        from core.orchestration.orchestration_observability import (
            get_observable_orchestrator_sync,
            ObservabilityConfig,
        )

        config = ObservabilityConfig(project_name="custom")
        orchestrator = get_observable_orchestrator_sync(config=config)
        assert orchestrator.config.project_name == "custom"


# =============================================================================
# Test: Tracing Decorator
# =============================================================================

class TestTracingDecorator:
    """Test the traced() decorator."""

    def test_traced_decorator_returns_function(self):
        """Test that traced decorator returns a callable."""
        from core.orchestration.orchestration_observability import traced

        @traced(name="test_func")
        async def test_func():
            return "result"

        assert callable(test_func)

    @pytest.mark.asyncio
    async def test_traced_decorator_async_execution(self):
        """Test traced decorator on async function."""
        from core.orchestration.orchestration_observability import traced

        @traced(name="async_test")
        async def async_test():
            return "async_result"

        result = await async_test()
        assert result == "async_result"

    def test_traced_decorator_sync_execution(self):
        """Test traced decorator on sync function."""
        from core.orchestration.orchestration_observability import traced

        @traced(name="sync_test")
        def sync_test():
            return "sync_result"

        result = sync_test()
        assert result == "sync_result"


class TestTraceOrchestrationDecorator:
    """Test trace_orchestration convenience decorator."""

    def test_trace_orchestration_returns_decorator(self):
        """Test trace_orchestration returns a decorator."""
        from core.orchestration.orchestration_observability import trace_orchestration

        decorator = trace_orchestration(project="test")
        assert callable(decorator)


# =============================================================================
# Test: Initialization (requires orchestrator availability)
# =============================================================================

class TestOrchestratorInitialization:
    """Test ObservableOrchestrator initialization."""

    @pytest.mark.asyncio
    async def test_initialize_creates_orchestrator(self):
        """Test initialization creates internal orchestrator."""
        from core.orchestration.orchestration_observability import (
            ObservableOrchestrator,
            ORCHESTRATOR_AVAILABLE,
        )

        if not ORCHESTRATOR_AVAILABLE:
            pytest.skip("CoreOrchestrator not available")

        orchestrator = ObservableOrchestrator()
        result = await orchestrator.initialize()

        assert result is True
        assert orchestrator.is_initialized is True

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        """Test initialization is idempotent."""
        from core.orchestration.orchestration_observability import (
            ObservableOrchestrator,
            ORCHESTRATOR_AVAILABLE,
        )

        if not ORCHESTRATOR_AVAILABLE:
            pytest.skip("CoreOrchestrator not available")

        orchestrator = ObservableOrchestrator()
        await orchestrator.initialize()
        result = await orchestrator.initialize()

        assert result is True


# =============================================================================
# Test: Coordinate Agents (requires orchestrator)
# =============================================================================

class TestCoordinateAgents:
    """Test coordinate_agents with observability."""

    @pytest.mark.asyncio
    async def test_coordinate_agents_basic(self):
        """Test basic coordination with tracing."""
        from core.orchestration.orchestration_observability import (
            create_observable_orchestrator,
            ORCHESTRATOR_AVAILABLE,
        )

        if not ORCHESTRATOR_AVAILABLE:
            pytest.skip("CoreOrchestrator not available")

        orchestrator = await create_observable_orchestrator()

        result = await orchestrator.coordinate_agents(
            task="test task",
            agents=["ralph"],
        )

        assert result is not None
        assert result.task_id.startswith("task_")
        assert "trace_id" in (result.metadata or {})

    @pytest.mark.asyncio
    async def test_coordinate_agents_custom_trace_id(self):
        """Test coordination with custom trace ID."""
        from core.orchestration.orchestration_observability import (
            create_observable_orchestrator,
            ORCHESTRATOR_AVAILABLE,
        )

        if not ORCHESTRATOR_AVAILABLE:
            pytest.skip("CoreOrchestrator not available")

        orchestrator = await create_observable_orchestrator()

        result = await orchestrator.coordinate_agents(
            task="test task",
            trace_id="custom_trace_123",
        )

        assert result.metadata.get("trace_id") == "custom_trace_123"


# =============================================================================
# Test: Status and Stats
# =============================================================================

class TestStatusAndStats:
    """Test status and statistics reporting."""

    @pytest.mark.asyncio
    async def test_get_status_includes_observability(self):
        """Test get_status includes observability info."""
        from core.orchestration.orchestration_observability import (
            create_observable_orchestrator,
            ORCHESTRATOR_AVAILABLE,
        )

        if not ORCHESTRATOR_AVAILABLE:
            pytest.skip("CoreOrchestrator not available")

        orchestrator = await create_observable_orchestrator()
        status = orchestrator.get_status()

        assert "observability" in status
        assert "opik_available" in status["observability"]
        assert "evaluator_available" in status["observability"]
        assert "config" in status["observability"]
        assert "stats" in status["observability"]


# =============================================================================
# Test: Opik Integration (requires OPIK_API_KEY)
# =============================================================================

_has_opik_key = bool(os.environ.get("OPIK_API_KEY"))


@pytest.mark.skipif(
    not _has_opik_key,
    reason="OPIK_API_KEY not available"
)
class TestOpikIntegration:
    """
    Tests that require a valid OPIK_API_KEY and make real API calls.
    """

    @pytest.mark.asyncio
    async def test_traced_coordination_with_opik(self):
        """Test coordination is traced to Opik dashboard."""
        from core.orchestration.orchestration_observability import (
            create_observable_orchestrator,
            OPIK_AVAILABLE,
            ORCHESTRATOR_AVAILABLE,
        )

        if not OPIK_AVAILABLE:
            pytest.skip("Opik not installed")
        if not ORCHESTRATOR_AVAILABLE:
            pytest.skip("CoreOrchestrator not available")

        orchestrator = await create_observable_orchestrator()

        result = await orchestrator.coordinate_agents(
            task="test opik tracing",
            agents=["ralph"],
        )

        assert result is not None
        assert result.metadata.get("traced") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
