"""
Adaptive Topology Switching System - V66 Architecture

Implements dynamic topology reconfiguration for swarm orchestration with:
- Multiple topology types: hierarchical, mesh, ring, star
- Adaptive switching based on task complexity, agent count, failure rate
- Health-based, cost-based, and load-based switching triggers
- Graceful migration between topologies with fallback mechanisms
- Real-time metrics tracking per topology

Architecture Decision: ADR-031 - Adaptive Topology Management

Reference Performance Targets:
- Hierarchical: 0.20s response (Queen-led coordination)
- Mesh: 0.15s response (peer-to-peer, no SPOF)
- Ring: Sequential processing for ordered tasks
- Star: Central coordinator for simple tasks

Version: V1.0.0 (February 2026)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class TopologyType(str, Enum):
    """Supported topology types for swarm orchestration."""

    HIERARCHICAL = "hierarchical"  # Queen-led coordination (0.20s response)
    MESH = "mesh"  # Peer-to-peer (0.15s response, no SPOF)
    RING = "ring"  # Sequential processing (ordered tasks)
    STAR = "star"  # Central coordinator (simple tasks)


class SwitchReason(str, Enum):
    """Reasons for topology switching."""

    HEALTH_DEGRADED = "health_degraded"
    COST_OPTIMIZATION = "cost_optimization"
    LOAD_CHANGE = "load_change"
    TASK_COMPLEXITY = "task_complexity"
    AGENT_COUNT_CHANGE = "agent_count_change"
    FAILURE_RATE_HIGH = "failure_rate_high"
    MANUAL_OVERRIDE = "manual_override"
    RECOVERY = "recovery"
    INITIAL = "initial"


class MigrationState(str, Enum):
    """Migration state machine states."""

    IDLE = "idle"
    PREPARING = "preparing"
    MIGRATING = "migrating"
    VALIDATING = "validating"
    COMPLETED = "completed"
    ROLLING_BACK = "rolling_back"
    FAILED = "failed"


# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================


@dataclass
class TopologyThresholds:
    """Thresholds for adaptive topology switching."""

    # Health-based switching
    health_threshold_low: float = 0.70  # Switch to mesh if health < 70%
    health_threshold_high: float = 0.90  # Switch to hierarchical if health > 90%

    # Cost-based switching
    cost_variance_threshold: float = 2.0  # Switch if >200% cost variation

    # Load-based switching
    load_threshold_high: float = 0.80  # High load -> mesh
    load_threshold_low: float = 0.30  # Low load -> hierarchical

    # Failure rate thresholds
    failure_rate_critical: float = 0.20  # >20% failure -> mesh
    failure_rate_acceptable: float = 0.05  # <5% failure -> can use hierarchical

    # Agent count thresholds
    min_agents_for_mesh: int = 3  # Minimum agents for mesh topology
    max_agents_for_star: int = 10  # Maximum agents for star topology
    optimal_agents_for_ring: int = 8  # Optimal agent count for ring

    # Task complexity
    complexity_threshold_simple: float = 0.3  # Simple tasks -> star
    complexity_threshold_complex: float = 0.7  # Complex tasks -> hierarchical


@dataclass
class TopologyConfig:
    """Configuration for a specific topology type."""

    topology_type: TopologyType
    expected_response_ms: float
    optimal_agent_count: Tuple[int, int]  # (min, max)
    supports_fault_tolerance: bool = True
    requires_coordinator: bool = True
    supports_parallel: bool = True
    cost_multiplier: float = 1.0


@dataclass
class TopologyMetrics:
    """Metrics tracked per topology."""

    topology_type: TopologyType
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_latency_ms: float = 0.0
    active_time_seconds: float = 0.0
    switch_count: int = 0
    last_active: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_operations == 0:
            return 1.0
        return self.successful_operations / self.total_operations

    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.total_operations == 0:
            return 0.0
        return self.total_latency_ms / self.total_operations

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_operations == 0:
            return 0.0
        return self.failed_operations / self.total_operations

    def record_operation(self, success: bool, latency_ms: float) -> None:
        """Record an operation."""
        self.total_operations += 1
        self.total_latency_ms += latency_ms
        if success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "topology_type": self.topology_type.value,
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "active_time_seconds": self.active_time_seconds,
            "switch_count": self.switch_count,
            "last_active": self.last_active.isoformat() if self.last_active else None,
        }


@dataclass
class SwarmContext:
    """Current swarm context for topology decisions."""

    agent_count: int = 0
    health_ratio: float = 1.0
    current_load: float = 0.0
    failure_rate: float = 0.0
    avg_task_complexity: float = 0.5
    total_cost: float = 0.0
    healthy_agents: int = 0
    degraded_agents: int = 0
    failed_agents: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "agent_count": self.agent_count,
            "health_ratio": self.health_ratio,
            "current_load": self.current_load,
            "failure_rate": self.failure_rate,
            "avg_task_complexity": self.avg_task_complexity,
            "total_cost": self.total_cost,
            "healthy_agents": self.healthy_agents,
            "degraded_agents": self.degraded_agents,
            "failed_agents": self.failed_agents,
        }


# =============================================================================
# TOPOLOGY IMPLEMENTATIONS
# =============================================================================


class BaseTopology(ABC):
    """Abstract base class for topology implementations."""

    def __init__(self, topology_type: TopologyType, config: TopologyConfig):
        self.topology_type = topology_type
        self.config = config
        self._agents: Dict[str, Any] = {}
        self._coordinator: Optional[str] = None
        self._active = False
        self._created_at = datetime.now(timezone.utc)

    @property
    def agent_count(self) -> int:
        """Get number of agents."""
        return len(self._agents)

    @property
    def is_active(self) -> bool:
        """Check if topology is active."""
        return self._active

    @abstractmethod
    async def activate(self) -> bool:
        """Activate the topology."""
        ...

    @abstractmethod
    async def deactivate(self) -> bool:
        """Deactivate the topology."""
        ...

    @abstractmethod
    async def add_agent(self, agent_id: str, metadata: Dict[str, Any]) -> bool:
        """Add an agent to the topology."""
        ...

    @abstractmethod
    async def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the topology."""
        ...

    @abstractmethod
    async def route_message(
        self, source: str, target: str, message: Any
    ) -> Optional[List[str]]:
        """
        Route a message from source to target.

        Returns:
            List of agent IDs in the route, or None if no route exists
        """
        ...

    @abstractmethod
    def get_topology_info(self) -> Dict[str, Any]:
        """Get topology information."""
        ...


class HierarchicalTopology(BaseTopology):
    """
    Hierarchical topology with Queen-led coordination.

    Structure:
    - Queen (top-level coordinator)
    - Sub-coordinators (regional/domain)
    - Workers (task executors)

    Response time target: 0.20s
    """

    def __init__(self):
        config = TopologyConfig(
            topology_type=TopologyType.HIERARCHICAL,
            expected_response_ms=200.0,
            optimal_agent_count=(5, 50),
            supports_fault_tolerance=True,
            requires_coordinator=True,
            supports_parallel=True,
            cost_multiplier=1.2,
        )
        super().__init__(TopologyType.HIERARCHICAL, config)
        self._sub_coordinators: Dict[str, List[str]] = {}  # sub_coord -> workers
        self._worker_assignments: Dict[str, str] = {}  # worker -> sub_coord

    async def activate(self) -> bool:
        """Activate hierarchical topology."""
        self._active = True
        logger.info("[Hierarchical] Topology activated")
        return True

    async def deactivate(self) -> bool:
        """Deactivate hierarchical topology."""
        self._active = False
        logger.info("[Hierarchical] Topology deactivated")
        return True

    async def add_agent(self, agent_id: str, metadata: Dict[str, Any]) -> bool:
        """Add agent to hierarchy."""
        role = metadata.get("role", "worker")
        self._agents[agent_id] = metadata

        if role == "queen":
            self._coordinator = agent_id
        elif role == "sub_coordinator":
            self._sub_coordinators[agent_id] = []
        else:
            # Assign to least loaded sub-coordinator
            best_sub = self._find_best_sub_coordinator()
            if best_sub:
                self._sub_coordinators[best_sub].append(agent_id)
                self._worker_assignments[agent_id] = best_sub

        return True

    async def remove_agent(self, agent_id: str) -> bool:
        """Remove agent from hierarchy."""
        if agent_id not in self._agents:
            return False

        # Clean up assignments
        if agent_id in self._worker_assignments:
            sub_coord = self._worker_assignments.pop(agent_id)
            if sub_coord in self._sub_coordinators:
                self._sub_coordinators[sub_coord].remove(agent_id)

        if agent_id in self._sub_coordinators:
            # Reassign workers
            workers = self._sub_coordinators.pop(agent_id)
            for worker in workers:
                best_sub = self._find_best_sub_coordinator()
                if best_sub:
                    self._sub_coordinators[best_sub].append(worker)
                    self._worker_assignments[worker] = best_sub

        if agent_id == self._coordinator:
            self._coordinator = None

        del self._agents[agent_id]
        return True

    def _find_best_sub_coordinator(self) -> Optional[str]:
        """Find sub-coordinator with least workers."""
        if not self._sub_coordinators:
            return None
        return min(self._sub_coordinators, key=lambda k: len(self._sub_coordinators[k]))

    async def route_message(
        self, source: str, target: str, message: Any
    ) -> Optional[List[str]]:
        """Route message through hierarchy."""
        if source not in self._agents or target not in self._agents:
            return None

        route = [source]

        # Route up to coordinator then down
        if self._coordinator:
            route.append(self._coordinator)

        # Find target's sub-coordinator
        target_sub = self._worker_assignments.get(target)
        if target_sub:
            route.append(target_sub)

        route.append(target)
        return route

    def get_topology_info(self) -> Dict[str, Any]:
        """Get topology information."""
        return {
            "type": self.topology_type.value,
            "active": self._active,
            "agent_count": self.agent_count,
            "coordinator": self._coordinator,
            "sub_coordinators": len(self._sub_coordinators),
            "worker_distribution": {
                sub: len(workers)
                for sub, workers in self._sub_coordinators.items()
            },
        }


class MeshTopology(BaseTopology):
    """
    Mesh topology with peer-to-peer communication.

    Features:
    - No single point of failure
    - Direct agent-to-agent communication
    - Self-organizing network

    Response time target: 0.15s
    """

    def __init__(self):
        config = TopologyConfig(
            topology_type=TopologyType.MESH,
            expected_response_ms=150.0,
            optimal_agent_count=(3, 20),
            supports_fault_tolerance=True,
            requires_coordinator=False,
            supports_parallel=True,
            cost_multiplier=1.0,
        )
        super().__init__(TopologyType.MESH, config)
        self._connections: Dict[str, Set[str]] = defaultdict(set)

    async def activate(self) -> bool:
        """Activate mesh topology."""
        # Establish full mesh connections
        agents = list(self._agents.keys())
        for i, agent in enumerate(agents):
            for other in agents[i + 1 :]:
                self._connections[agent].add(other)
                self._connections[other].add(agent)

        self._active = True
        logger.info("[Mesh] Topology activated with %d connections", len(agents))
        return True

    async def deactivate(self) -> bool:
        """Deactivate mesh topology."""
        self._connections.clear()
        self._active = False
        logger.info("[Mesh] Topology deactivated")
        return True

    async def add_agent(self, agent_id: str, metadata: Dict[str, Any]) -> bool:
        """Add agent with connections to all existing agents."""
        self._agents[agent_id] = metadata

        # Connect to all existing agents (full mesh)
        for existing_agent in self._agents:
            if existing_agent != agent_id:
                self._connections[agent_id].add(existing_agent)
                self._connections[existing_agent].add(agent_id)

        return True

    async def remove_agent(self, agent_id: str) -> bool:
        """Remove agent and its connections."""
        if agent_id not in self._agents:
            return False

        # Remove all connections
        for connected in self._connections[agent_id]:
            self._connections[connected].discard(agent_id)
        del self._connections[agent_id]
        del self._agents[agent_id]
        return True

    async def route_message(
        self, source: str, target: str, message: Any
    ) -> Optional[List[str]]:
        """Direct routing in mesh (single hop)."""
        if source not in self._agents or target not in self._agents:
            return None

        # Direct connection in mesh
        return [source, target]

    def get_topology_info(self) -> Dict[str, Any]:
        """Get topology information."""
        total_connections = sum(len(conns) for conns in self._connections.values()) // 2
        return {
            "type": self.topology_type.value,
            "active": self._active,
            "agent_count": self.agent_count,
            "total_connections": total_connections,
            "avg_connections_per_agent": (
                total_connections * 2 / max(1, self.agent_count)
            ),
        }


class RingTopology(BaseTopology):
    """
    Ring topology for sequential processing.

    Features:
    - Ordered task processing
    - Token passing mechanism
    - Predictable message flow

    Best for: Tasks requiring ordered execution
    """

    def __init__(self):
        config = TopologyConfig(
            topology_type=TopologyType.RING,
            expected_response_ms=250.0,  # Varies with ring size
            optimal_agent_count=(4, 12),
            supports_fault_tolerance=False,
            requires_coordinator=False,
            supports_parallel=False,
            cost_multiplier=0.8,
        )
        super().__init__(TopologyType.RING, config)
        self._ring_order: List[str] = []
        self._token_holder: Optional[str] = None

    async def activate(self) -> bool:
        """Activate ring topology."""
        self._ring_order = list(self._agents.keys())
        if self._ring_order:
            self._token_holder = self._ring_order[0]
        self._active = True
        logger.info("[Ring] Topology activated with %d nodes", len(self._ring_order))
        return True

    async def deactivate(self) -> bool:
        """Deactivate ring topology."""
        self._ring_order = []
        self._token_holder = None
        self._active = False
        logger.info("[Ring] Topology deactivated")
        return True

    async def add_agent(self, agent_id: str, metadata: Dict[str, Any]) -> bool:
        """Add agent to ring end."""
        self._agents[agent_id] = metadata
        self._ring_order.append(agent_id)

        if len(self._ring_order) == 1:
            self._token_holder = agent_id

        return True

    async def remove_agent(self, agent_id: str) -> bool:
        """Remove agent from ring."""
        if agent_id not in self._agents:
            return False

        idx = self._ring_order.index(agent_id)
        self._ring_order.remove(agent_id)
        del self._agents[agent_id]

        # Update token holder if removed
        if self._token_holder == agent_id and self._ring_order:
            self._token_holder = self._ring_order[idx % len(self._ring_order)]

        return True

    async def pass_token(self) -> Optional[str]:
        """Pass token to next agent in ring."""
        if not self._ring_order or not self._token_holder:
            return None

        current_idx = self._ring_order.index(self._token_holder)
        next_idx = (current_idx + 1) % len(self._ring_order)
        self._token_holder = self._ring_order[next_idx]
        return self._token_holder

    async def route_message(
        self, source: str, target: str, message: Any
    ) -> Optional[List[str]]:
        """Route through ring (clockwise)."""
        if source not in self._agents or target not in self._agents:
            return None

        source_idx = self._ring_order.index(source)
        target_idx = self._ring_order.index(target)

        # Route clockwise
        route = []
        current = source_idx
        while current != target_idx:
            route.append(self._ring_order[current])
            current = (current + 1) % len(self._ring_order)
        route.append(self._ring_order[target_idx])

        return route

    def get_topology_info(self) -> Dict[str, Any]:
        """Get topology information."""
        return {
            "type": self.topology_type.value,
            "active": self._active,
            "agent_count": self.agent_count,
            "ring_order": self._ring_order,
            "token_holder": self._token_holder,
        }


class StarTopology(BaseTopology):
    """
    Star topology with central coordinator.

    Features:
    - Simple architecture
    - Central control point
    - Easy to manage

    Best for: Simple tasks with few agents
    """

    def __init__(self):
        config = TopologyConfig(
            topology_type=TopologyType.STAR,
            expected_response_ms=180.0,
            optimal_agent_count=(2, 10),
            supports_fault_tolerance=False,
            requires_coordinator=True,
            supports_parallel=True,
            cost_multiplier=0.9,
        )
        super().__init__(TopologyType.STAR, config)
        self._center: Optional[str] = None
        self._spokes: Set[str] = set()

    async def activate(self) -> bool:
        """Activate star topology."""
        if not self._center:
            # Select first agent as center
            if self._agents:
                self._center = next(iter(self._agents.keys()))
                self._spokes = set(self._agents.keys()) - {self._center}

        self._active = True
        logger.info("[Star] Topology activated with center %s", self._center)
        return True

    async def deactivate(self) -> bool:
        """Deactivate star topology."""
        self._active = False
        logger.info("[Star] Topology deactivated")
        return True

    async def add_agent(self, agent_id: str, metadata: Dict[str, Any]) -> bool:
        """Add agent as spoke (or center if first)."""
        self._agents[agent_id] = metadata

        if not self._center:
            self._center = agent_id
        else:
            self._spokes.add(agent_id)

        return True

    async def remove_agent(self, agent_id: str) -> bool:
        """Remove agent from star."""
        if agent_id not in self._agents:
            return False

        if agent_id == self._center:
            # Promote a spoke to center
            if self._spokes:
                self._center = self._spokes.pop()
            else:
                self._center = None
        else:
            self._spokes.discard(agent_id)

        del self._agents[agent_id]
        return True

    async def route_message(
        self, source: str, target: str, message: Any
    ) -> Optional[List[str]]:
        """Route through center hub."""
        if source not in self._agents or target not in self._agents:
            return None

        # All messages go through center
        if source == self._center:
            return [source, target]
        elif target == self._center:
            return [source, target]
        else:
            return [source, self._center, target] if self._center else None

    def get_topology_info(self) -> Dict[str, Any]:
        """Get topology information."""
        return {
            "type": self.topology_type.value,
            "active": self._active,
            "agent_count": self.agent_count,
            "center": self._center,
            "spoke_count": len(self._spokes),
        }


# =============================================================================
# TOPOLOGY FACTORY
# =============================================================================


class TopologyFactory:
    """Factory for creating topology instances."""

    _registry: Dict[TopologyType, type] = {
        TopologyType.HIERARCHICAL: HierarchicalTopology,
        TopologyType.MESH: MeshTopology,
        TopologyType.RING: RingTopology,
        TopologyType.STAR: StarTopology,
    }

    @classmethod
    def create(cls, topology_type: TopologyType) -> BaseTopology:
        """Create a topology instance."""
        topology_class = cls._registry.get(topology_type)
        if not topology_class:
            raise ValueError(f"Unknown topology type: {topology_type}")
        return topology_class()

    @classmethod
    def register(cls, topology_type: TopologyType, topology_class: type) -> None:
        """Register a custom topology type."""
        cls._registry[topology_type] = topology_class


# =============================================================================
# TOPOLOGY MANAGER
# =============================================================================


@dataclass
class TopologySwitchEvent:
    """Event representing a topology switch."""

    event_id: str
    timestamp: datetime
    from_topology: Optional[TopologyType]
    to_topology: TopologyType
    reason: SwitchReason
    context: SwarmContext
    duration_ms: float
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "from_topology": self.from_topology.value if self.from_topology else None,
            "to_topology": self.to_topology.value,
            "reason": self.reason.value,
            "context": self.context.to_dict(),
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
        }


class TopologyManager:
    """
    Adaptive Topology Manager for swarm orchestration.

    Features:
    - Dynamic topology switching based on context
    - Health-based, cost-based, and load-based triggers
    - Graceful migration between topologies
    - Automatic fallback on errors
    - Comprehensive metrics tracking

    Usage:
        manager = TopologyManager()
        await manager.initialize(TopologyType.HIERARCHICAL)

        # Update context and let manager decide
        context = SwarmContext(health_ratio=0.65, current_load=0.85)
        await manager.evaluate_and_switch(context)
    """

    def __init__(
        self,
        thresholds: Optional[TopologyThresholds] = None,
        default_topology: TopologyType = TopologyType.HIERARCHICAL,
    ):
        """
        Initialize the topology manager.

        Args:
            thresholds: Custom thresholds for switching decisions
            default_topology: Default topology to use
        """
        self.thresholds = thresholds or TopologyThresholds()
        self.default_topology = default_topology

        # Current state
        self._current_topology: Optional[BaseTopology] = None
        self._current_type: Optional[TopologyType] = None
        self._migration_state = MigrationState.IDLE
        self._lock = asyncio.Lock()

        # Metrics per topology
        self._metrics: Dict[TopologyType, TopologyMetrics] = {
            t: TopologyMetrics(topology_type=t) for t in TopologyType
        }

        # Event history
        self._switch_history: List[TopologySwitchEvent] = []
        self._max_history = 100

        # Callbacks
        self._on_switch_callbacks: List[Callable[[TopologySwitchEvent], None]] = []

        # Agent registry (preserved across topology switches)
        self._agents: Dict[str, Dict[str, Any]] = {}

        # Context tracking
        self._last_context: Optional[SwarmContext] = None
        self._last_evaluation: Optional[datetime] = None
        self._evaluation_interval_seconds = 10.0

        logger.info("[TopologyManager] Initialized with default: %s", default_topology.value)

    @property
    def current_topology(self) -> Optional[TopologyType]:
        """Get current topology type."""
        return self._current_type

    @property
    def migration_state(self) -> MigrationState:
        """Get current migration state."""
        return self._migration_state

    @property
    def agent_count(self) -> int:
        """Get total agent count."""
        return len(self._agents)

    async def initialize(self, initial_topology: Optional[TopologyType] = None) -> bool:
        """
        Initialize the topology manager with initial topology.

        Args:
            initial_topology: Initial topology type (uses default if not provided)

        Returns:
            True if initialization succeeded
        """
        topology_type = initial_topology or self.default_topology

        async with self._lock:
            try:
                self._current_topology = TopologyFactory.create(topology_type)
                await self._current_topology.activate()
                self._current_type = topology_type
                self._migration_state = MigrationState.IDLE

                # Record initial switch
                event = TopologySwitchEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc),
                    from_topology=None,
                    to_topology=topology_type,
                    reason=SwitchReason.INITIAL,
                    context=SwarmContext(),
                    duration_ms=0.0,
                    success=True,
                )
                self._record_switch_event(event)

                logger.info("[TopologyManager] Initialized with topology: %s", topology_type.value)
                return True

            except Exception as e:
                logger.error("[TopologyManager] Initialization failed: %s", e)
                return False

    async def shutdown(self) -> bool:
        """Shutdown the topology manager."""
        async with self._lock:
            if self._current_topology:
                await self._current_topology.deactivate()
                self._current_topology = None
                self._current_type = None

            logger.info("[TopologyManager] Shutdown complete")
            return True

    async def add_agent(self, agent_id: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add an agent to the current topology.

        Args:
            agent_id: Unique agent identifier
            metadata: Optional agent metadata

        Returns:
            True if agent was added successfully
        """
        metadata = metadata or {}

        async with self._lock:
            self._agents[agent_id] = metadata

            if self._current_topology:
                result = await self._current_topology.add_agent(agent_id, metadata)
                if result:
                    logger.debug("[TopologyManager] Agent added: %s", agent_id)
                return result

            return True

    async def remove_agent(self, agent_id: str) -> bool:
        """
        Remove an agent from the current topology.

        Args:
            agent_id: Agent identifier to remove

        Returns:
            True if agent was removed successfully
        """
        async with self._lock:
            if agent_id not in self._agents:
                return False

            del self._agents[agent_id]

            if self._current_topology:
                result = await self._current_topology.remove_agent(agent_id)
                if result:
                    logger.debug("[TopologyManager] Agent removed: %s", agent_id)
                return result

            return True

    def recommend_topology(self, context: SwarmContext) -> Tuple[TopologyType, SwitchReason]:
        """
        Recommend optimal topology based on current context.

        Args:
            context: Current swarm context

        Returns:
            Tuple of (recommended_topology, reason)
        """
        # Health-based recommendation (highest priority)
        if context.health_ratio < self.thresholds.health_threshold_low:
            return TopologyType.MESH, SwitchReason.HEALTH_DEGRADED

        # Failure rate check
        if context.failure_rate > self.thresholds.failure_rate_critical:
            return TopologyType.MESH, SwitchReason.FAILURE_RATE_HIGH

        # Load-based recommendation
        if context.current_load > self.thresholds.load_threshold_high:
            return TopologyType.MESH, SwitchReason.LOAD_CHANGE

        if context.current_load < self.thresholds.load_threshold_low:
            if context.health_ratio > self.thresholds.health_threshold_high:
                return TopologyType.HIERARCHICAL, SwitchReason.LOAD_CHANGE

        # Agent count considerations
        if context.agent_count < self.thresholds.min_agents_for_mesh:
            return TopologyType.STAR, SwitchReason.AGENT_COUNT_CHANGE

        if context.agent_count > self.thresholds.max_agents_for_star:
            if context.avg_task_complexity < self.thresholds.complexity_threshold_simple:
                return TopologyType.HIERARCHICAL, SwitchReason.TASK_COMPLEXITY

        # Task complexity considerations
        if context.avg_task_complexity < self.thresholds.complexity_threshold_simple:
            return TopologyType.STAR, SwitchReason.TASK_COMPLEXITY

        if context.avg_task_complexity > self.thresholds.complexity_threshold_complex:
            return TopologyType.HIERARCHICAL, SwitchReason.TASK_COMPLEXITY

        # Default: maintain current or use hierarchical
        return self._current_type or TopologyType.HIERARCHICAL, SwitchReason.RECOVERY

    async def evaluate_and_switch(
        self, context: SwarmContext, force: bool = False
    ) -> Optional[TopologySwitchEvent]:
        """
        Evaluate context and switch topology if needed.

        Args:
            context: Current swarm context
            force: Force evaluation even if interval not elapsed

        Returns:
            Switch event if a switch occurred, None otherwise
        """
        self._last_context = context

        # Check evaluation interval
        if not force and self._last_evaluation:
            elapsed = (datetime.now(timezone.utc) - self._last_evaluation).total_seconds()
            if elapsed < self._evaluation_interval_seconds:
                return None

        self._last_evaluation = datetime.now(timezone.utc)

        # Get recommendation
        recommended, reason = self.recommend_topology(context)

        # Check if switch is needed
        if recommended == self._current_type and not force:
            return None

        # Perform switch
        return await self.switch_topology(recommended, reason, context)

    async def switch_topology(
        self,
        target_topology: TopologyType,
        reason: SwitchReason,
        context: Optional[SwarmContext] = None,
    ) -> Optional[TopologySwitchEvent]:
        """
        Switch to a new topology.

        Args:
            target_topology: Target topology type
            reason: Reason for the switch
            context: Optional context snapshot

        Returns:
            Switch event if successful, None on failure
        """
        if self._migration_state != MigrationState.IDLE:
            logger.warning("[TopologyManager] Migration already in progress")
            return None

        context = context or SwarmContext()
        start_time = time.time()
        from_topology = self._current_type

        async with self._lock:
            try:
                self._migration_state = MigrationState.PREPARING

                # Create new topology
                new_topology = TopologyFactory.create(target_topology)

                self._migration_state = MigrationState.MIGRATING

                # Migrate agents to new topology
                for agent_id, metadata in self._agents.items():
                    await new_topology.add_agent(agent_id, metadata)

                # Activate new topology
                await new_topology.activate()

                self._migration_state = MigrationState.VALIDATING

                # Deactivate old topology
                if self._current_topology:
                    await self._current_topology.deactivate()

                # Switch
                self._current_topology = new_topology
                self._current_type = target_topology

                self._migration_state = MigrationState.COMPLETED

                # Update metrics
                duration_ms = (time.time() - start_time) * 1000
                self._metrics[target_topology].switch_count += 1
                self._metrics[target_topology].last_active = datetime.now(timezone.utc)

                # Create event
                event = TopologySwitchEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc),
                    from_topology=from_topology,
                    to_topology=target_topology,
                    reason=reason,
                    context=context,
                    duration_ms=duration_ms,
                    success=True,
                )
                self._record_switch_event(event)
                self._emit_switch_callbacks(event)

                self._migration_state = MigrationState.IDLE

                logger.info(
                    "[TopologyManager] Switched topology: %s -> %s (reason: %s, duration: %.2fms)",
                    from_topology.value if from_topology else "none",
                    target_topology.value,
                    reason.value,
                    duration_ms,
                )

                return event

            except Exception as e:
                self._migration_state = MigrationState.ROLLING_BACK

                logger.error("[TopologyManager] Switch failed, rolling back: %s", e)

                # Rollback to hierarchical as safe default
                try:
                    await self._rollback_to_default()
                except Exception as rollback_error:
                    logger.error("[TopologyManager] Rollback failed: %s", rollback_error)

                self._migration_state = MigrationState.FAILED

                # Record failed event
                duration_ms = (time.time() - start_time) * 1000
                event = TopologySwitchEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc),
                    from_topology=from_topology,
                    to_topology=target_topology,
                    reason=reason,
                    context=context,
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e),
                )
                self._record_switch_event(event)

                self._migration_state = MigrationState.IDLE
                return event

    async def _rollback_to_default(self) -> None:
        """Rollback to default topology (hierarchical)."""
        fallback = TopologyFactory.create(TopologyType.HIERARCHICAL)

        for agent_id, metadata in self._agents.items():
            await fallback.add_agent(agent_id, metadata)

        await fallback.activate()
        self._current_topology = fallback
        self._current_type = TopologyType.HIERARCHICAL

        logger.info("[TopologyManager] Rolled back to HIERARCHICAL")

    def _record_switch_event(self, event: TopologySwitchEvent) -> None:
        """Record a switch event in history."""
        self._switch_history.append(event)
        if len(self._switch_history) > self._max_history:
            self._switch_history.pop(0)

    def _emit_switch_callbacks(self, event: TopologySwitchEvent) -> None:
        """Emit switch event to callbacks."""
        for callback in self._on_switch_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error("[TopologyManager] Callback error: %s", e)

    def on_topology_switch(self, callback: Callable[[TopologySwitchEvent], None]) -> None:
        """Register a callback for topology switch events."""
        self._on_switch_callbacks.append(callback)

    def record_operation(self, success: bool, latency_ms: float) -> None:
        """Record an operation for current topology metrics."""
        if self._current_type:
            self._metrics[self._current_type].record_operation(success, latency_ms)

    async def route_message(
        self, source: str, target: str, message: Any
    ) -> Optional[List[str]]:
        """Route a message through current topology."""
        if self._current_topology:
            return await self._current_topology.route_message(source, target, message)
        return None

    def get_topology_info(self) -> Dict[str, Any]:
        """Get current topology information."""
        if self._current_topology:
            return self._current_topology.get_topology_info()
        return {"type": None, "active": False}

    def get_metrics(self, topology_type: Optional[TopologyType] = None) -> Dict[str, Any]:
        """
        Get metrics for a specific or all topologies.

        Args:
            topology_type: Optional specific topology (returns all if None)

        Returns:
            Metrics dictionary
        """
        if topology_type:
            return self._metrics[topology_type].to_dict()

        return {t.value: m.to_dict() for t, m in self._metrics.items()}

    def get_switch_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent switch history."""
        recent = self._switch_history[-limit:]
        return [event.to_dict() for event in reversed(recent)]

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive manager status."""
        return {
            "current_topology": self._current_type.value if self._current_type else None,
            "migration_state": self._migration_state.value,
            "agent_count": self.agent_count,
            "topology_info": self.get_topology_info(),
            "metrics": self.get_metrics(),
            "recent_switches": len(self._switch_history),
            "last_evaluation": (
                self._last_evaluation.isoformat() if self._last_evaluation else None
            ),
            "last_context": self._last_context.to_dict() if self._last_context else None,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


_default_manager: Optional[TopologyManager] = None


async def get_topology_manager() -> TopologyManager:
    """Get or create the default topology manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = TopologyManager()
        await _default_manager.initialize()
    return _default_manager


async def switch_topology(
    target: TopologyType, reason: SwitchReason = SwitchReason.MANUAL_OVERRIDE
) -> Optional[TopologySwitchEvent]:
    """Switch topology using default manager."""
    manager = await get_topology_manager()
    return await manager.switch_topology(target, reason)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "TopologyType",
    "SwitchReason",
    "MigrationState",
    # Configuration
    "TopologyThresholds",
    "TopologyConfig",
    "TopologyMetrics",
    "SwarmContext",
    # Topology implementations
    "BaseTopology",
    "HierarchicalTopology",
    "MeshTopology",
    "RingTopology",
    "StarTopology",
    "TopologyFactory",
    # Manager
    "TopologyManager",
    "TopologySwitchEvent",
    # Convenience
    "get_topology_manager",
    "switch_topology",
]
