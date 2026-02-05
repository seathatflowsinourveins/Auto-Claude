"""
Test Suite for Adaptive Topology Manager - V66

Comprehensive tests for:
- Topology types (hierarchical, mesh, ring, star)
- Adaptive switching logic
- Health-based, cost-based, load-based switching
- Graceful migration
- Fallback mechanisms
- Metrics tracking

Test count: 40+ tests covering all functionality
"""

import asyncio
import pytest
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

# Import the module under test
import sys
import os

# Add platform to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.orchestration.topology_manager import (
    # Enums
    TopologyType,
    SwitchReason,
    MigrationState,
    # Configuration
    TopologyThresholds,
    TopologyConfig,
    TopologyMetrics,
    SwarmContext,
    # Topologies
    BaseTopology,
    HierarchicalTopology,
    MeshTopology,
    RingTopology,
    StarTopology,
    TopologyFactory,
    # Manager
    TopologyManager,
    TopologySwitchEvent,
    # Convenience
    get_topology_manager,
    switch_topology,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def topology_thresholds():
    """Default topology thresholds for testing."""
    return TopologyThresholds(
        health_threshold_low=0.70,
        health_threshold_high=0.90,
        cost_variance_threshold=2.0,
        load_threshold_high=0.80,
        load_threshold_low=0.30,
        failure_rate_critical=0.20,
        failure_rate_acceptable=0.05,
        min_agents_for_mesh=3,
        max_agents_for_star=10,
        optimal_agents_for_ring=8,
        complexity_threshold_simple=0.3,
        complexity_threshold_complex=0.7,
    )


@pytest.fixture
def healthy_context():
    """Context representing a healthy swarm."""
    return SwarmContext(
        agent_count=10,
        health_ratio=0.95,
        current_load=0.50,
        failure_rate=0.02,
        avg_task_complexity=0.5,
        total_cost=100.0,
        healthy_agents=9,
        degraded_agents=1,
        failed_agents=0,
    )


@pytest.fixture
def degraded_context():
    """Context representing a degraded swarm."""
    return SwarmContext(
        agent_count=10,
        health_ratio=0.60,
        current_load=0.85,
        failure_rate=0.25,
        avg_task_complexity=0.5,
        total_cost=150.0,
        healthy_agents=5,
        degraded_agents=3,
        failed_agents=2,
    )


@pytest.fixture
def low_load_context():
    """Context with low load."""
    return SwarmContext(
        agent_count=10,
        health_ratio=0.95,
        current_load=0.20,
        failure_rate=0.01,
        avg_task_complexity=0.5,
        total_cost=50.0,
        healthy_agents=10,
        degraded_agents=0,
        failed_agents=0,
    )


@pytest.fixture
def high_load_context():
    """Context with high load."""
    return SwarmContext(
        agent_count=10,
        health_ratio=0.85,
        current_load=0.90,
        failure_rate=0.05,
        avg_task_complexity=0.5,
        total_cost=200.0,
        healthy_agents=8,
        degraded_agents=2,
        failed_agents=0,
    )


@pytest.fixture
async def topology_manager(topology_thresholds):
    """Create and initialize a topology manager."""
    manager = TopologyManager(thresholds=topology_thresholds)
    await manager.initialize(TopologyType.HIERARCHICAL)
    yield manager
    await manager.shutdown()


# =============================================================================
# TOPOLOGY TYPE TESTS
# =============================================================================


class TestTopologyTypes:
    """Tests for individual topology implementations."""

    @pytest.mark.asyncio
    async def test_hierarchical_topology_creation(self):
        """Test hierarchical topology can be created."""
        topology = HierarchicalTopology()
        assert topology.topology_type == TopologyType.HIERARCHICAL
        assert topology.config.expected_response_ms == 200.0
        assert topology.config.requires_coordinator is True

    @pytest.mark.asyncio
    async def test_hierarchical_topology_activation(self):
        """Test hierarchical topology can be activated."""
        topology = HierarchicalTopology()
        result = await topology.activate()
        assert result is True
        assert topology.is_active is True

    @pytest.mark.asyncio
    async def test_hierarchical_add_queen(self):
        """Test adding queen to hierarchical topology."""
        topology = HierarchicalTopology()
        await topology.activate()

        result = await topology.add_agent("queen-1", {"role": "queen"})
        assert result is True
        assert topology._coordinator == "queen-1"

    @pytest.mark.asyncio
    async def test_hierarchical_add_sub_coordinator(self):
        """Test adding sub-coordinator to hierarchical topology."""
        topology = HierarchicalTopology()
        await topology.activate()

        await topology.add_agent("queen-1", {"role": "queen"})
        result = await topology.add_agent("sub-1", {"role": "sub_coordinator"})

        assert result is True
        assert "sub-1" in topology._sub_coordinators

    @pytest.mark.asyncio
    async def test_hierarchical_add_worker(self):
        """Test adding worker to hierarchical topology."""
        topology = HierarchicalTopology()
        await topology.activate()

        await topology.add_agent("queen-1", {"role": "queen"})
        await topology.add_agent("sub-1", {"role": "sub_coordinator"})
        result = await topology.add_agent("worker-1", {"role": "worker"})

        assert result is True
        assert topology._worker_assignments.get("worker-1") == "sub-1"

    @pytest.mark.asyncio
    async def test_hierarchical_route_message(self):
        """Test message routing in hierarchical topology."""
        topology = HierarchicalTopology()
        await topology.activate()

        await topology.add_agent("queen-1", {"role": "queen"})
        await topology.add_agent("sub-1", {"role": "sub_coordinator"})
        await topology.add_agent("worker-1", {"role": "worker"})
        await topology.add_agent("worker-2", {"role": "worker"})

        route = await topology.route_message("worker-1", "worker-2", "test")

        assert route is not None
        assert "queen-1" in route  # Routes through coordinator

    @pytest.mark.asyncio
    async def test_mesh_topology_creation(self):
        """Test mesh topology can be created."""
        topology = MeshTopology()
        assert topology.topology_type == TopologyType.MESH
        assert topology.config.expected_response_ms == 150.0
        assert topology.config.requires_coordinator is False

    @pytest.mark.asyncio
    async def test_mesh_topology_full_connectivity(self):
        """Test mesh creates full connectivity."""
        topology = MeshTopology()

        await topology.add_agent("agent-1", {})
        await topology.add_agent("agent-2", {})
        await topology.add_agent("agent-3", {})
        await topology.activate()

        # Each agent should be connected to all others
        assert len(topology._connections["agent-1"]) == 2
        assert "agent-2" in topology._connections["agent-1"]
        assert "agent-3" in topology._connections["agent-1"]

    @pytest.mark.asyncio
    async def test_mesh_direct_routing(self):
        """Test mesh provides direct routing."""
        topology = MeshTopology()

        await topology.add_agent("agent-1", {})
        await topology.add_agent("agent-2", {})
        await topology.activate()

        route = await topology.route_message("agent-1", "agent-2", "test")

        assert route == ["agent-1", "agent-2"]  # Direct route

    @pytest.mark.asyncio
    async def test_ring_topology_creation(self):
        """Test ring topology can be created."""
        topology = RingTopology()
        assert topology.topology_type == TopologyType.RING
        assert topology.config.supports_parallel is False

    @pytest.mark.asyncio
    async def test_ring_maintains_order(self):
        """Test ring maintains agent order."""
        topology = RingTopology()

        await topology.add_agent("agent-1", {})
        await topology.add_agent("agent-2", {})
        await topology.add_agent("agent-3", {})
        await topology.activate()

        assert topology._ring_order == ["agent-1", "agent-2", "agent-3"]

    @pytest.mark.asyncio
    async def test_ring_token_passing(self):
        """Test ring token passing mechanism."""
        topology = RingTopology()

        await topology.add_agent("agent-1", {})
        await topology.add_agent("agent-2", {})
        await topology.add_agent("agent-3", {})
        await topology.activate()

        assert topology._token_holder == "agent-1"

        next_holder = await topology.pass_token()
        assert next_holder == "agent-2"

        next_holder = await topology.pass_token()
        assert next_holder == "agent-3"

        next_holder = await topology.pass_token()
        assert next_holder == "agent-1"  # Wrapped around

    @pytest.mark.asyncio
    async def test_ring_sequential_routing(self):
        """Test ring routes sequentially."""
        topology = RingTopology()

        await topology.add_agent("agent-1", {})
        await topology.add_agent("agent-2", {})
        await topology.add_agent("agent-3", {})
        await topology.activate()

        route = await topology.route_message("agent-1", "agent-3", "test")

        # Should go through agent-2
        assert route == ["agent-1", "agent-2", "agent-3"]

    @pytest.mark.asyncio
    async def test_star_topology_creation(self):
        """Test star topology can be created."""
        topology = StarTopology()
        assert topology.topology_type == TopologyType.STAR
        assert topology.config.requires_coordinator is True

    @pytest.mark.asyncio
    async def test_star_first_agent_becomes_center(self):
        """Test first agent becomes center in star."""
        topology = StarTopology()

        await topology.add_agent("agent-1", {})

        assert topology._center == "agent-1"
        assert len(topology._spokes) == 0

    @pytest.mark.asyncio
    async def test_star_subsequent_agents_become_spokes(self):
        """Test subsequent agents become spokes."""
        topology = StarTopology()

        await topology.add_agent("center", {})
        await topology.add_agent("spoke-1", {})
        await topology.add_agent("spoke-2", {})

        assert topology._center == "center"
        assert "spoke-1" in topology._spokes
        assert "spoke-2" in topology._spokes

    @pytest.mark.asyncio
    async def test_star_routes_through_center(self):
        """Test star routes all messages through center."""
        topology = StarTopology()

        await topology.add_agent("center", {})
        await topology.add_agent("spoke-1", {})
        await topology.add_agent("spoke-2", {})
        await topology.activate()

        route = await topology.route_message("spoke-1", "spoke-2", "test")

        assert route == ["spoke-1", "center", "spoke-2"]


# =============================================================================
# TOPOLOGY FACTORY TESTS
# =============================================================================


class TestTopologyFactory:
    """Tests for topology factory."""

    def test_factory_creates_hierarchical(self):
        """Test factory creates hierarchical topology."""
        topology = TopologyFactory.create(TopologyType.HIERARCHICAL)
        assert isinstance(topology, HierarchicalTopology)

    def test_factory_creates_mesh(self):
        """Test factory creates mesh topology."""
        topology = TopologyFactory.create(TopologyType.MESH)
        assert isinstance(topology, MeshTopology)

    def test_factory_creates_ring(self):
        """Test factory creates ring topology."""
        topology = TopologyFactory.create(TopologyType.RING)
        assert isinstance(topology, RingTopology)

    def test_factory_creates_star(self):
        """Test factory creates star topology."""
        topology = TopologyFactory.create(TopologyType.STAR)
        assert isinstance(topology, StarTopology)

    def test_factory_raises_for_unknown_type(self):
        """Test factory raises error for unknown type."""
        with pytest.raises(ValueError, match="Unknown topology type"):
            # Create a fake enum value
            TopologyFactory._registry.pop(TopologyType.HIERARCHICAL, None)
            try:
                TopologyFactory.create(TopologyType.HIERARCHICAL)
            finally:
                TopologyFactory._registry[TopologyType.HIERARCHICAL] = HierarchicalTopology


# =============================================================================
# TOPOLOGY MANAGER TESTS
# =============================================================================


class TestTopologyManager:
    """Tests for topology manager."""

    @pytest.mark.asyncio
    async def test_manager_initialization(self):
        """Test manager can be initialized."""
        manager = TopologyManager()
        result = await manager.initialize(TopologyType.HIERARCHICAL)

        assert result is True
        assert manager.current_topology == TopologyType.HIERARCHICAL
        assert manager.migration_state == MigrationState.IDLE

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_manager_default_topology(self):
        """Test manager uses default topology."""
        manager = TopologyManager(default_topology=TopologyType.MESH)
        await manager.initialize()

        assert manager.current_topology == TopologyType.MESH

        await manager.shutdown()

    @pytest.mark.asyncio
    async def test_manager_add_agent(self, topology_manager):
        """Test manager can add agents."""
        result = await topology_manager.add_agent("test-agent", {"role": "worker"})

        assert result is True
        assert topology_manager.agent_count == 1

    @pytest.mark.asyncio
    async def test_manager_remove_agent(self, topology_manager):
        """Test manager can remove agents."""
        await topology_manager.add_agent("test-agent", {"role": "worker"})
        result = await topology_manager.remove_agent("test-agent")

        assert result is True
        assert topology_manager.agent_count == 0

    @pytest.mark.asyncio
    async def test_manager_remove_nonexistent_agent(self, topology_manager):
        """Test removing nonexistent agent returns False."""
        result = await topology_manager.remove_agent("nonexistent")
        assert result is False


# =============================================================================
# ADAPTIVE SWITCHING TESTS
# =============================================================================


class TestAdaptiveSwitching:
    """Tests for adaptive topology switching logic."""

    @pytest.mark.asyncio
    async def test_recommend_mesh_on_low_health(self, topology_manager, degraded_context):
        """Test recommends mesh when health is low."""
        recommended, reason = topology_manager.recommend_topology(degraded_context)

        assert recommended == TopologyType.MESH
        assert reason == SwitchReason.HEALTH_DEGRADED

    @pytest.mark.asyncio
    async def test_recommend_mesh_on_high_failure_rate(self, topology_manager):
        """Test recommends mesh when failure rate is high."""
        context = SwarmContext(
            agent_count=10,
            health_ratio=0.80,
            failure_rate=0.25,  # >20%
        )

        recommended, reason = topology_manager.recommend_topology(context)

        assert recommended == TopologyType.MESH
        assert reason == SwitchReason.FAILURE_RATE_HIGH

    @pytest.mark.asyncio
    async def test_recommend_mesh_on_high_load(self, topology_manager, high_load_context):
        """Test recommends mesh when load is high."""
        recommended, reason = topology_manager.recommend_topology(high_load_context)

        assert recommended == TopologyType.MESH
        assert reason == SwitchReason.LOAD_CHANGE

    @pytest.mark.asyncio
    async def test_recommend_hierarchical_on_low_load_high_health(
        self, topology_manager, low_load_context
    ):
        """Test recommends hierarchical on low load with high health."""
        recommended, reason = topology_manager.recommend_topology(low_load_context)

        assert recommended == TopologyType.HIERARCHICAL
        assert reason == SwitchReason.LOAD_CHANGE

    @pytest.mark.asyncio
    async def test_recommend_star_for_few_agents(self, topology_manager):
        """Test recommends star when agent count is low."""
        context = SwarmContext(
            agent_count=2,  # Below min_agents_for_mesh
            health_ratio=0.90,
        )

        recommended, reason = topology_manager.recommend_topology(context)

        assert recommended == TopologyType.STAR
        assert reason == SwitchReason.AGENT_COUNT_CHANGE

    @pytest.mark.asyncio
    async def test_recommend_star_for_simple_tasks(self, topology_manager):
        """Test recommends star for simple tasks."""
        context = SwarmContext(
            agent_count=5,
            health_ratio=0.90,
            current_load=0.50,
            failure_rate=0.02,
            avg_task_complexity=0.2,  # Simple tasks
        )

        recommended, reason = topology_manager.recommend_topology(context)

        assert recommended == TopologyType.STAR
        assert reason == SwitchReason.TASK_COMPLEXITY

    @pytest.mark.asyncio
    async def test_switch_topology_success(self, topology_manager):
        """Test successful topology switch."""
        event = await topology_manager.switch_topology(
            TopologyType.MESH,
            SwitchReason.MANUAL_OVERRIDE,
        )

        assert event is not None
        assert event.success is True
        assert event.to_topology == TopologyType.MESH
        assert topology_manager.current_topology == TopologyType.MESH

    @pytest.mark.asyncio
    async def test_switch_preserves_agents(self, topology_manager):
        """Test topology switch preserves agents."""
        await topology_manager.add_agent("agent-1", {})
        await topology_manager.add_agent("agent-2", {})

        await topology_manager.switch_topology(
            TopologyType.MESH,
            SwitchReason.MANUAL_OVERRIDE,
        )

        assert topology_manager.agent_count == 2

    @pytest.mark.asyncio
    async def test_evaluate_and_switch(self, topology_manager, degraded_context):
        """Test evaluate_and_switch triggers switch on degraded context."""
        event = await topology_manager.evaluate_and_switch(degraded_context, force=True)

        assert event is not None
        assert event.to_topology == TopologyType.MESH
        assert topology_manager.current_topology == TopologyType.MESH

    @pytest.mark.asyncio
    async def test_evaluate_no_switch_when_optimal(self, topology_manager, healthy_context):
        """Test no switch when current topology is optimal."""
        # First switch to get into a known state
        await topology_manager.switch_topology(
            TopologyType.HIERARCHICAL,
            SwitchReason.MANUAL_OVERRIDE,
        )

        # Healthy context should maintain hierarchical
        event = await topology_manager.evaluate_and_switch(healthy_context, force=True)

        # May return None or same topology
        if event:
            assert event.to_topology == TopologyType.HIERARCHICAL


# =============================================================================
# MIGRATION AND FALLBACK TESTS
# =============================================================================


class TestMigrationAndFallback:
    """Tests for migration and fallback behavior."""

    @pytest.mark.asyncio
    async def test_migration_state_progression(self, topology_manager):
        """Test migration state progresses correctly."""
        states_observed = []

        original_switch = topology_manager.switch_topology

        async def track_switch(*args, **kwargs):
            states_observed.append(topology_manager.migration_state)
            return await original_switch(*args, **kwargs)

        # The migration happens internally
        await topology_manager.switch_topology(
            TopologyType.MESH,
            SwitchReason.MANUAL_OVERRIDE,
        )

        # Final state should be IDLE
        assert topology_manager.migration_state == MigrationState.IDLE

    @pytest.mark.asyncio
    async def test_switch_records_event(self, topology_manager):
        """Test switch records event in history."""
        await topology_manager.switch_topology(
            TopologyType.MESH,
            SwitchReason.MANUAL_OVERRIDE,
        )

        history = topology_manager.get_switch_history()

        # Should have initial switch + new switch
        assert len(history) >= 1
        assert history[0]["to_topology"] == "mesh"

    @pytest.mark.asyncio
    async def test_switch_callback_triggered(self, topology_manager):
        """Test switch triggers registered callbacks."""
        callback_events = []

        def callback(event: TopologySwitchEvent):
            callback_events.append(event)

        topology_manager.on_topology_switch(callback)

        await topology_manager.switch_topology(
            TopologyType.MESH,
            SwitchReason.MANUAL_OVERRIDE,
        )

        assert len(callback_events) == 1
        assert callback_events[0].to_topology == TopologyType.MESH

    @pytest.mark.asyncio
    async def test_concurrent_switch_blocked(self, topology_manager):
        """Test concurrent switches are blocked."""
        # Simulate a long-running migration
        topology_manager._migration_state = MigrationState.MIGRATING

        result = await topology_manager.switch_topology(
            TopologyType.MESH,
            SwitchReason.MANUAL_OVERRIDE,
        )

        assert result is None

        # Reset state
        topology_manager._migration_state = MigrationState.IDLE


# =============================================================================
# METRICS TESTS
# =============================================================================


class TestMetrics:
    """Tests for metrics tracking."""

    def test_topology_metrics_initialization(self):
        """Test TopologyMetrics initialization."""
        metrics = TopologyMetrics(topology_type=TopologyType.MESH)

        assert metrics.total_operations == 0
        assert metrics.success_rate == 1.0
        assert metrics.avg_latency_ms == 0.0

    def test_topology_metrics_record_operation(self):
        """Test recording operations."""
        metrics = TopologyMetrics(topology_type=TopologyType.MESH)

        metrics.record_operation(success=True, latency_ms=100.0)
        metrics.record_operation(success=True, latency_ms=200.0)
        metrics.record_operation(success=False, latency_ms=150.0)

        assert metrics.total_operations == 3
        assert metrics.successful_operations == 2
        assert metrics.failed_operations == 1
        assert metrics.success_rate == 2 / 3
        assert metrics.failure_rate == 1 / 3
        assert metrics.avg_latency_ms == 150.0

    @pytest.mark.asyncio
    async def test_manager_records_operations(self, topology_manager):
        """Test manager records operations to metrics."""
        topology_manager.record_operation(success=True, latency_ms=50.0)
        topology_manager.record_operation(success=True, latency_ms=100.0)

        metrics = topology_manager.get_metrics(TopologyType.HIERARCHICAL)

        assert metrics["total_operations"] == 2
        assert metrics["avg_latency_ms"] == 75.0

    @pytest.mark.asyncio
    async def test_manager_get_all_metrics(self, topology_manager):
        """Test getting all topology metrics."""
        metrics = topology_manager.get_metrics()

        assert "hierarchical" in metrics
        assert "mesh" in metrics
        assert "ring" in metrics
        assert "star" in metrics

    @pytest.mark.asyncio
    async def test_manager_get_status(self, topology_manager):
        """Test manager status includes all info."""
        await topology_manager.add_agent("test-agent", {})

        status = topology_manager.get_status()

        assert status["current_topology"] == "hierarchical"
        assert status["migration_state"] == "idle"
        assert status["agent_count"] == 1
        assert "topology_info" in status
        assert "metrics" in status


# =============================================================================
# SWARM CONTEXT TESTS
# =============================================================================


class TestSwarmContext:
    """Tests for SwarmContext."""

    def test_context_to_dict(self, healthy_context):
        """Test context serialization."""
        data = healthy_context.to_dict()

        assert data["agent_count"] == 10
        assert data["health_ratio"] == 0.95
        assert data["current_load"] == 0.50
        assert data["failure_rate"] == 0.02

    def test_context_defaults(self):
        """Test context has sensible defaults."""
        context = SwarmContext()

        assert context.agent_count == 0
        assert context.health_ratio == 1.0
        assert context.current_load == 0.0
        assert context.failure_rate == 0.0


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.mark.asyncio
    async def test_get_topology_manager_creates_singleton(self):
        """Test get_topology_manager creates singleton."""
        # Reset singleton
        import core.orchestration.topology_manager as tm
        tm._default_manager = None

        manager1 = await get_topology_manager()
        manager2 = await get_topology_manager()

        assert manager1 is manager2

        await manager1.shutdown()
        tm._default_manager = None

    @pytest.mark.asyncio
    async def test_switch_topology_convenience(self):
        """Test switch_topology convenience function."""
        import core.orchestration.topology_manager as tm
        tm._default_manager = None

        event = await switch_topology(TopologyType.MESH)

        assert event is not None
        assert event.to_topology == TopologyType.MESH

        manager = await get_topology_manager()
        await manager.shutdown()
        tm._default_manager = None


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_route_message_without_topology(self):
        """Test routing without active topology."""
        manager = TopologyManager()
        # Not initialized

        route = await manager.route_message("a", "b", "test")
        assert route is None

    @pytest.mark.asyncio
    async def test_route_to_nonexistent_agent(self, topology_manager):
        """Test routing to nonexistent agent."""
        await topology_manager.add_agent("agent-1", {})

        route = await topology_manager.route_message("agent-1", "nonexistent", "test")
        assert route is None

    @pytest.mark.asyncio
    async def test_switch_event_serialization(self):
        """Test TopologySwitchEvent serialization."""
        event = TopologySwitchEvent(
            event_id="test-123",
            timestamp=datetime.now(timezone.utc),
            from_topology=TopologyType.HIERARCHICAL,
            to_topology=TopologyType.MESH,
            reason=SwitchReason.HEALTH_DEGRADED,
            context=SwarmContext(agent_count=5),
            duration_ms=150.0,
            success=True,
        )

        data = event.to_dict()

        assert data["event_id"] == "test-123"
        assert data["from_topology"] == "hierarchical"
        assert data["to_topology"] == "mesh"
        assert data["reason"] == "health_degraded"
        assert data["duration_ms"] == 150.0
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_remove_center_from_star(self):
        """Test removing center from star promotes spoke."""
        topology = StarTopology()

        await topology.add_agent("center", {})
        await topology.add_agent("spoke-1", {})
        await topology.add_agent("spoke-2", {})
        await topology.activate()

        # Remove center
        result = await topology.remove_agent("center")

        assert result is True
        assert topology._center in ["spoke-1", "spoke-2"]
        assert topology.agent_count == 2

    @pytest.mark.asyncio
    async def test_ring_remove_token_holder(self):
        """Test removing token holder updates token."""
        topology = RingTopology()

        await topology.add_agent("agent-1", {})
        await topology.add_agent("agent-2", {})
        await topology.add_agent("agent-3", {})
        await topology.activate()

        assert topology._token_holder == "agent-1"

        await topology.remove_agent("agent-1")

        assert topology._token_holder in ["agent-2", "agent-3"]
        assert len(topology._ring_order) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
