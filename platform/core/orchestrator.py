"""
Agent Orchestrator - Unified Multi-Agent Coordination System.

Synthesizes patterns from:
- swarm/coordinator.py (Queen/Mesh topologies)
- core/cooperation.py (Task coordination, handoffs)
- core/executor.py (ReAct execution loop)
- core/skills.py (Capability management)

Key Features:
1. Topology Selection - Auto-switch HIERARCHICAL <-> MESH based on scale
2. Task Decomposition - Break complex tasks into DAG of subtasks
3. Capability Matching - Match tasks to agents by declared capabilities
4. Swarm Intelligence - Emergent coordination through simple rules
5. Load Balancing - Work stealing and queue-based dispatch
6. Observability - Metrics for throughput, utilization, latency
"""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class Topology(str, Enum):
    """Network topology for agent coordination."""

    HIERARCHICAL = "hierarchical"  # Queen-led, centralized dispatch
    MESH = "mesh"                  # Peer-to-peer, gossip-based
    HYBRID = "hybrid"              # Dynamic switching based on load
    SOLO = "solo"                  # Single agent, no coordination


class AgentRole(str, Enum):
    """Roles agents can assume in the swarm."""

    ORCHESTRATOR = "orchestrator"  # Coordinates other agents
    SPECIALIST = "specialist"      # Domain expert (coding, research, etc.)
    GENERALIST = "generalist"      # Can handle varied tasks
    VALIDATOR = "validator"        # Reviews and validates work
    SCOUT = "scout"                # Explores and gathers information
    RESEARCHER = "researcher"      # Deep research and information synthesis


class TaskPriority(str, Enum):
    """Task priority levels."""

    CRITICAL = "critical"  # Immediate execution required
    HIGH = "high"          # Execute soon
    NORMAL = "normal"      # Standard priority
    LOW = "low"            # Can wait
    BACKGROUND = "background"  # Execute when idle


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"        # Not yet started
    QUEUED = "queued"          # In execution queue
    ASSIGNED = "assigned"      # Assigned to agent
    IN_PROGRESS = "in_progress"  # Currently executing
    BLOCKED = "blocked"        # Waiting on dependencies
    COMPLETED = "completed"    # Successfully finished
    FAILED = "failed"          # Execution failed
    CANCELLED = "cancelled"    # Manually cancelled


class AgentStatus(str, Enum):
    """Agent availability status."""

    IDLE = "idle"              # Ready for work
    BUSY = "busy"              # Currently executing
    OVERLOADED = "overloaded"  # Too many tasks queued
    OFFLINE = "offline"        # Not available
    PAUSED = "paused"          # Temporarily paused


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class AgentCapability:
    """Describes what an agent can do."""

    name: str
    proficiency: float = 1.0  # 0.0 to 1.0
    max_concurrent: int = 1   # How many tasks of this type at once

    def matches(self, requirement: str) -> Tuple[bool, float]:
        """Check if capability matches a requirement."""
        # Exact match
        if self.name.lower() == requirement.lower():
            return True, self.proficiency
        # Partial match (e.g., "python" matches "python-coding")
        if requirement.lower() in self.name.lower():
            return True, self.proficiency * 0.8
        if self.name.lower() in requirement.lower():
            return True, self.proficiency * 0.8
        return False, 0.0


@dataclass
class Agent:
    """Represents an agent in the orchestration system."""

    agent_id: str
    role: AgentRole = AgentRole.GENERALIST
    status: AgentStatus = AgentStatus.IDLE
    capabilities: List[AgentCapability] = field(default_factory=list)
    current_tasks: List[str] = field(default_factory=list)  # Task IDs
    max_concurrent_tasks: int = 3
    load_factor: float = 0.0  # 0.0 to 1.0+
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_available(self) -> bool:
        """Check if agent can accept more work."""
        return (
            self.status in (AgentStatus.IDLE, AgentStatus.BUSY)
            and len(self.current_tasks) < self.max_concurrent_tasks
            and self.load_factor < 1.0
        )

    def update_load(self) -> None:
        """Recalculate load factor based on current tasks."""
        self.load_factor = len(self.current_tasks) / max(self.max_concurrent_tasks, 1)
        if self.load_factor >= 1.0:
            self.status = AgentStatus.OVERLOADED
        elif self.load_factor > 0:
            self.status = AgentStatus.BUSY
        else:
            self.status = AgentStatus.IDLE


@dataclass
class TaskDependency:
    """Represents a dependency between tasks."""

    task_id: str
    dependency_type: str = "completion"  # completion, data, resource
    required_output: Optional[str] = None  # Specific output key needed


@dataclass
class Task:
    """Represents a task to be executed."""

    task_id: str
    description: str
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    required_capabilities: List[str] = field(default_factory=list)
    dependencies: List[TaskDependency] = field(default_factory=list)
    assigned_agent: Optional[str] = None
    parent_task_id: Optional[str] = None  # For subtasks
    subtask_ids: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout_seconds: int = 300  # 5 minutes default
    retry_count: int = 0
    max_retries: int = 3

    @property
    def is_ready(self) -> bool:
        """Check if all dependencies are satisfied."""
        # Implemented by orchestrator checking dependency status
        return self.status == TaskStatus.QUEUED

    @property
    def duration_ms(self) -> Optional[float]:
        """Calculate execution duration in milliseconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None


@dataclass
class TaskDecomposition:
    """Result of decomposing a complex task into subtasks."""

    original_task_id: str
    subtasks: List[Task]
    dependency_graph: Dict[str, List[str]]  # task_id -> dependent_task_ids
    estimated_total_duration: Optional[float] = None


@dataclass
class OrchestratorMetrics:
    """Metrics for orchestration performance."""

    total_tasks_submitted: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    active_agents: int = 0
    idle_agents: int = 0
    average_task_duration_ms: float = 0.0
    average_queue_time_ms: float = 0.0
    tasks_per_second: float = 0.0
    agent_utilization: float = 0.0
    topology_switches: int = 0
    work_steals: int = 0


# =============================================================================
# DECOMPOSITION STRATEGIES
# =============================================================================


class DecompositionStrategy(ABC):
    """Abstract base for task decomposition strategies."""

    @abstractmethod
    def decompose(self, task: Task) -> TaskDecomposition:
        """Break a task into subtasks."""
        pass


class SequentialDecomposition(DecompositionStrategy):
    """Decompose into sequential steps."""

    def __init__(self, step_descriptions: List[str]):
        self.step_descriptions = step_descriptions

    def decompose(self, task: Task) -> TaskDecomposition:
        """Create sequential subtasks where each depends on the previous."""
        subtasks = []
        dependencies: Dict[str, List[str]] = {}

        prev_task_id: Optional[str] = None
        for i, desc in enumerate(self.step_descriptions):
            subtask_id = f"{task.task_id}-step-{i+1}"
            deps = []
            if prev_task_id:
                deps.append(TaskDependency(task_id=prev_task_id))

            subtask = Task(
                task_id=subtask_id,
                description=f"{task.description}: {desc}",
                priority=task.priority,
                status=TaskStatus.PENDING,
                required_capabilities=task.required_capabilities,
                dependencies=deps,
                parent_task_id=task.task_id,
                context=task.context.copy(),
            )
            subtasks.append(subtask)
            dependencies[subtask_id] = [prev_task_id] if prev_task_id else []
            prev_task_id = subtask_id

        return TaskDecomposition(
            original_task_id=task.task_id,
            subtasks=subtasks,
            dependency_graph=dependencies,
        )


class ParallelDecomposition(DecompositionStrategy):
    """Decompose into parallel independent subtasks."""

    def __init__(self, subtask_descriptions: List[Tuple[str, List[str]]]):
        """
        Args:
            subtask_descriptions: List of (description, capabilities) tuples
        """
        self.subtask_descriptions = subtask_descriptions

    def decompose(self, task: Task) -> TaskDecomposition:
        """Create parallel subtasks that can execute independently."""
        subtasks = []
        dependencies: Dict[str, List[str]] = {}

        for i, (desc, caps) in enumerate(self.subtask_descriptions):
            subtask_id = f"{task.task_id}-parallel-{i+1}"

            subtask = Task(
                task_id=subtask_id,
                description=f"{task.description}: {desc}",
                priority=task.priority,
                status=TaskStatus.QUEUED,  # Ready immediately
                required_capabilities=caps or task.required_capabilities,
                dependencies=[],  # No dependencies
                parent_task_id=task.task_id,
                context=task.context.copy(),
            )
            subtasks.append(subtask)
            dependencies[subtask_id] = []

        return TaskDecomposition(
            original_task_id=task.task_id,
            subtasks=subtasks,
            dependency_graph=dependencies,
        )


class DAGDecomposition(DecompositionStrategy):
    """Decompose into a directed acyclic graph of subtasks."""

    def __init__(
        self,
        nodes: List[Tuple[str, str, List[str]]],  # (id_suffix, description, capabilities)
        edges: List[Tuple[str, str]],  # (from_suffix, to_suffix)
    ):
        self.nodes = nodes
        self.edges = edges

    def decompose(self, task: Task) -> TaskDecomposition:
        """Create DAG of subtasks with explicit dependencies."""
        subtasks = []
        dependencies: Dict[str, List[str]] = {}
        id_map: Dict[str, str] = {}  # suffix -> full task_id

        # Create all subtasks
        for suffix, desc, caps in self.nodes:
            subtask_id = f"{task.task_id}-{suffix}"
            id_map[suffix] = subtask_id
            dependencies[subtask_id] = []

            subtask = Task(
                task_id=subtask_id,
                description=f"{task.description}: {desc}",
                priority=task.priority,
                status=TaskStatus.PENDING,
                required_capabilities=caps or task.required_capabilities,
                dependencies=[],
                parent_task_id=task.task_id,
                context=task.context.copy(),
            )
            subtasks.append(subtask)

        # Add edges as dependencies
        for from_suffix, to_suffix in self.edges:
            from_id = id_map[from_suffix]
            to_id = id_map[to_suffix]
            dependencies[to_id].append(from_id)

            # Update subtask dependencies
            for subtask in subtasks:
                if subtask.task_id == to_id:
                    subtask.dependencies.append(TaskDependency(task_id=from_id))

        # Mark tasks with no dependencies as queued (ready)
        for subtask in subtasks:
            if not subtask.dependencies:
                subtask.status = TaskStatus.QUEUED

        return TaskDecomposition(
            original_task_id=task.task_id,
            subtasks=subtasks,
            dependency_graph=dependencies,
        )


# =============================================================================
# SWARM INTELLIGENCE BEHAVIORS
# =============================================================================


class SwarmBehavior(ABC):
    """Abstract base for swarm intelligence behaviors."""

    @abstractmethod
    async def execute(self, orchestrator: Orchestrator) -> None:
        """Execute the swarm behavior."""
        pass


class WorkStealingBehavior(SwarmBehavior):
    """Idle agents steal work from overloaded agents."""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold  # Steal when agent load > threshold

    async def execute(self, orchestrator: Orchestrator) -> None:
        """Move tasks from overloaded to idle agents."""
        idle_agents = [a for a in orchestrator._agents.values() if a.status == AgentStatus.IDLE]
        overloaded = [a for a in orchestrator._agents.values() if a.load_factor > self.threshold]

        for victim in overloaded:
            if not idle_agents:
                break

            # Find stealable tasks (queued but not started)
            for task_id in list(victim.current_tasks):
                task = orchestrator._tasks.get(task_id)
                if task and task.status in (TaskStatus.QUEUED, TaskStatus.ASSIGNED):
                    thief = idle_agents.pop(0)
                    await orchestrator._reassign_task(task_id, thief.agent_id)
                    orchestrator._metrics.work_steals += 1

                    if not idle_agents:
                        break


class LoadBalancingBehavior(SwarmBehavior):
    """Distribute load evenly across agents."""

    async def execute(self, orchestrator: Orchestrator) -> None:
        """Rebalance tasks to achieve even distribution."""
        if len(orchestrator._agents) < 2:
            return

        avg_load = sum(a.load_factor for a in orchestrator._agents.values()) / len(orchestrator._agents)

        # Identify agents far from average
        underloaded = [a for a in orchestrator._agents.values() if a.load_factor < avg_load * 0.5]
        overloaded = [a for a in orchestrator._agents.values() if a.load_factor > avg_load * 1.5]

        # Move tasks from overloaded to underloaded
        for over in overloaded:
            for under in underloaded:
                if over.load_factor <= avg_load:
                    break

                # Find a task to move
                for task_id in list(over.current_tasks):
                    task = orchestrator._tasks.get(task_id)
                    if task and task.status in (TaskStatus.QUEUED, TaskStatus.ASSIGNED):
                        # Check capability match
                        if orchestrator._check_capability_match(under, task):
                            await orchestrator._reassign_task(task_id, under.agent_id)
                            break


class TopologyAdaptationBehavior(SwarmBehavior):
    """Automatically switch topology based on conditions."""

    def __init__(self, hierarchical_threshold: int = 5, mesh_threshold: int = 20):
        self.hierarchical_threshold = hierarchical_threshold
        self.mesh_threshold = mesh_threshold

    async def execute(self, orchestrator: Orchestrator) -> None:
        """Switch topology based on agent count and load."""
        agent_count = len(orchestrator._agents)

        if orchestrator._topology == Topology.HYBRID:
            # Auto-select based on scale
            if agent_count <= self.hierarchical_threshold:
                orchestrator._effective_topology = Topology.HIERARCHICAL
            elif agent_count >= self.mesh_threshold:
                orchestrator._effective_topology = Topology.MESH
            else:
                # Between thresholds, use load to decide
                avg_load = sum(a.load_factor for a in orchestrator._agents.values()) / max(agent_count, 1)
                if avg_load > 0.7:
                    orchestrator._effective_topology = Topology.MESH  # Distribute load
                else:
                    orchestrator._effective_topology = Topology.HIERARCHICAL  # Centralized control

            if orchestrator._effective_topology != orchestrator._last_effective_topology:
                orchestrator._last_effective_topology = orchestrator._effective_topology
                orchestrator._metrics.topology_switches += 1


class SwarmResearchBehavior(SwarmBehavior):
    """
    Coordinate distributed research across RESEARCHER agents.

    Features:
    - Parallel query distribution
    - Result aggregation and synthesis
    - Source deduplication
    - Citation consolidation
    """

    def __init__(self, max_parallel_queries: int = 3):
        self.max_parallel_queries = max_parallel_queries
        self._research_engine = None  # Lazy load

    def _get_research_engine(self):
        """Lazy load research engine."""
        if self._research_engine is None:
            try:
                from .research_engine import get_engine
                self._research_engine = get_engine()
            except ImportError:
                pass
        return self._research_engine

    async def execute(self, orchestrator: Orchestrator) -> None:
        """
        Distribute research tasks to RESEARCHER agents.

        For complex queries, splits into sub-queries and assigns to idle researchers.
        """
        engine = self._get_research_engine()
        if not engine:
            return

        # Find researcher agents
        researchers = [
            a for a in orchestrator._agents.values()
            if a.role == AgentRole.RESEARCHER and a.status == AgentStatus.IDLE
        ]

        if not researchers:
            return

        # Find research tasks in queue
        research_tasks = [
            orchestrator._tasks[tid] for tid in orchestrator._task_queue
            if tid in orchestrator._tasks
            and "research" in orchestrator._tasks[tid].description.lower()
            and orchestrator._tasks[tid].status == TaskStatus.PENDING
        ]

        if not research_tasks:
            return

        # Distribute research tasks to researchers
        for task, agent in zip(research_tasks[:len(researchers)], researchers):
            task.status = TaskStatus.ASSIGNED
            task.assigned_to = agent.agent_id
            agent.status = AgentStatus.BUSY

            # Create sub-tasks for parallel research
            sub_queries = self._decompose_query(task.description)

            # Store research sub-tasks as metadata
            task.metadata = task.metadata or {}
            task.metadata["research_sub_queries"] = sub_queries
            task.metadata["research_sources"] = ["exa", "firecrawl"]

    def _decompose_query(self, query: str) -> List[str]:
        """
        Decompose a complex research query into sub-queries.

        Simple heuristic: split by 'and', 'vs', 'compare' keywords.
        """
        sub_queries = [query]

        for splitter in [" and ", " vs ", " versus ", " compare "]:
            if splitter in query.lower():
                parts = query.lower().split(splitter)
                sub_queries = [p.strip() for p in parts if p.strip()]
                break

        return sub_queries[:self.max_parallel_queries]


# =============================================================================
# ORCHESTRATOR
# =============================================================================


class Orchestrator:
    """
    Unified multi-agent orchestration system.

    Features:
    - Dynamic topology selection (hierarchical, mesh, hybrid)
    - Task decomposition with dependency management
    - Capability-based agent matching
    - Swarm intelligence behaviors
    - Work stealing and load balancing
    - Comprehensive metrics and observability
    """

    def __init__(
        self,
        orchestrator_id: str = "",
        topology: Topology = Topology.HYBRID,
        enable_swarm_behaviors: bool = True,
        behavior_interval_seconds: float = 5.0,
    ):
        self.orchestrator_id = orchestrator_id or f"orch-{uuid.uuid4().hex[:8]}"
        self._topology = topology
        self._effective_topology = topology if topology != Topology.HYBRID else Topology.HIERARCHICAL
        self._last_effective_topology = self._effective_topology

        # Storage
        self._agents: Dict[str, Agent] = {}
        self._tasks: Dict[str, Task] = {}
        self._task_queue: List[str] = []  # Priority queue (task IDs)
        self._completed_tasks: Dict[str, Task] = {}

        # Swarm behaviors
        self._swarm_behaviors: List[SwarmBehavior] = []
        if enable_swarm_behaviors:
            self._swarm_behaviors = [
                TopologyAdaptationBehavior(),
                WorkStealingBehavior(),
                LoadBalancingBehavior(),
                SwarmResearchBehavior(),
            ]
        self._behavior_interval = behavior_interval_seconds
        self._behavior_task: Optional[asyncio.Task] = None

        # Callbacks
        self._on_task_complete: List[Callable[[Task], None]] = []
        self._on_task_failed: List[Callable[[Task], None]] = []
        self._on_agent_status_change: List[Callable[[Agent], None]] = []

        # Metrics
        self._metrics = OrchestratorMetrics()
        self._start_time = time.time()
        self._task_durations: List[float] = []

        # Running state
        self._is_running = False

    # -------------------------------------------------------------------------
    # Agent Management
    # -------------------------------------------------------------------------

    def register_agent(
        self,
        agent_id: str,
        role: AgentRole = AgentRole.GENERALIST,
        capabilities: Optional[List[AgentCapability]] = None,
        max_concurrent_tasks: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Agent:
        """Register a new agent with the orchestrator."""
        agent = Agent(
            agent_id=agent_id,
            role=role,
            status=AgentStatus.IDLE,
            capabilities=capabilities or [],
            max_concurrent_tasks=max_concurrent_tasks,
            metadata=metadata or {},
        )
        self._agents[agent_id] = agent
        self._update_agent_metrics()
        return agent

    def unregister_agent(self, agent_id: str) -> bool:
        """Remove an agent from the orchestrator."""
        if agent_id not in self._agents:
            return False

        agent = self._agents[agent_id]

        # Reassign any current tasks
        for task_id in list(agent.current_tasks):
            task = self._tasks.get(task_id)
            if task and task.status in (TaskStatus.ASSIGNED, TaskStatus.QUEUED):
                task.assigned_agent = None
                task.status = TaskStatus.QUEUED
                self._task_queue.append(task_id)

        del self._agents[agent_id]
        self._update_agent_metrics()
        return True

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def update_agent_status(self, agent_id: str, status: AgentStatus) -> bool:
        """Update an agent's status."""
        agent = self._agents.get(agent_id)
        if not agent:
            return False

        old_status = agent.status
        agent.status = status
        agent.last_heartbeat = datetime.now(timezone.utc)

        if old_status != status:
            for callback in self._on_agent_status_change:
                callback(agent)

        self._update_agent_metrics()
        return True

    def heartbeat(self, agent_id: str) -> bool:
        """Record agent heartbeat."""
        agent = self._agents.get(agent_id)
        if not agent:
            return False

        agent.last_heartbeat = datetime.now(timezone.utc)
        return True

    # -------------------------------------------------------------------------
    # Task Management
    # -------------------------------------------------------------------------

    def submit_task(
        self,
        description: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        required_capabilities: Optional[List[str]] = None,
        dependencies: Optional[List[TaskDependency]] = None,
        context: Optional[Dict[str, Any]] = None,
        timeout_seconds: int = 300,
    ) -> str:
        """Submit a new task to the orchestrator."""
        task_id = f"task-{uuid.uuid4().hex[:12]}"

        task = Task(
            task_id=task_id,
            description=description,
            priority=priority,
            status=TaskStatus.PENDING,
            required_capabilities=required_capabilities or [],
            dependencies=dependencies or [],
            context=context or {},
            timeout_seconds=timeout_seconds,
        )

        self._tasks[task_id] = task
        self._metrics.total_tasks_submitted += 1

        # Check if ready to queue (no dependencies or all satisfied)
        if self._check_dependencies_satisfied(task):
            task.status = TaskStatus.QUEUED
            self._enqueue_task(task_id)
        else:
            task.status = TaskStatus.BLOCKED

        return task_id

    def decompose_task(
        self,
        task_id: str,
        strategy: DecompositionStrategy,
    ) -> TaskDecomposition:
        """Decompose a task into subtasks using the given strategy."""
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        decomposition = strategy.decompose(task)

        # Register subtasks
        for subtask in decomposition.subtasks:
            self._tasks[subtask.task_id] = subtask
            task.subtask_ids.append(subtask.task_id)
            self._metrics.total_tasks_submitted += 1

            # Queue ready subtasks
            if subtask.status == TaskStatus.QUEUED:
                self._enqueue_task(subtask.task_id)

        # Mark parent as having subtasks
        task.status = TaskStatus.IN_PROGRESS

        return decomposition

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self._tasks.get(task_id) or self._completed_tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or queued task."""
        task = self._tasks.get(task_id)
        if not task:
            return False

        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            return False

        task.status = TaskStatus.CANCELLED

        # Remove from queue
        if task_id in self._task_queue:
            self._task_queue.remove(task_id)

        # Unassign from agent
        if task.assigned_agent:
            agent = self._agents.get(task.assigned_agent)
            if agent and task_id in agent.current_tasks:
                agent.current_tasks.remove(task_id)
                agent.update_load()

        return True

    async def complete_task(
        self,
        task_id: str,
        result: Any = None,
        error: Optional[str] = None,
    ) -> bool:
        """Mark a task as completed (successfully or with error)."""
        task = self._tasks.get(task_id)
        if not task:
            return False

        task.completed_at = datetime.now(timezone.utc)

        if error:
            task.status = TaskStatus.FAILED
            task.error = error
            self._metrics.total_tasks_failed += 1

            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.QUEUED
                task.error = None
                self._enqueue_task(task_id)
            else:
                for callback in self._on_task_failed:
                    callback(task)
        else:
            task.status = TaskStatus.COMPLETED
            task.result = result
            self._metrics.total_tasks_completed += 1

            if task.duration_ms:
                self._task_durations.append(task.duration_ms)

            for callback in self._on_task_complete:
                callback(task)

        # Unassign from agent
        if task.assigned_agent:
            agent = self._agents.get(task.assigned_agent)
            if agent and task_id in agent.current_tasks:
                agent.current_tasks.remove(task_id)
                agent.update_load()

        # Check if this unblocks dependent tasks
        await self._check_dependent_tasks(task_id)

        # Check if parent task is complete
        if task.parent_task_id:
            await self._check_parent_completion(task.parent_task_id)

        # Move to completed storage
        self._completed_tasks[task_id] = task
        del self._tasks[task_id]

        self._update_metrics()
        return True

    # -------------------------------------------------------------------------
    # Task Assignment
    # -------------------------------------------------------------------------

    async def assign_pending_tasks(self) -> int:
        """Assign queued tasks to available agents."""
        assigned_count = 0

        # Sort queue by priority
        self._sort_task_queue()

        for task_id in list(self._task_queue):
            task = self._tasks.get(task_id)
            if not task or task.status != TaskStatus.QUEUED:
                continue

            # Find best agent
            agent = self._find_best_agent(task)
            if agent:
                await self._assign_task(task_id, agent.agent_id)
                assigned_count += 1

        return assigned_count

    async def _assign_task(self, task_id: str, agent_id: str) -> bool:
        """Assign a task to a specific agent."""
        task = self._tasks.get(task_id)
        agent = self._agents.get(agent_id)

        if not task or not agent:
            return False

        task.assigned_agent = agent_id
        task.status = TaskStatus.ASSIGNED
        task.started_at = datetime.now(timezone.utc)

        agent.current_tasks.append(task_id)
        agent.update_load()

        # Remove from queue
        if task_id in self._task_queue:
            self._task_queue.remove(task_id)

        return True

    async def _reassign_task(self, task_id: str, new_agent_id: str) -> bool:
        """Reassign a task to a different agent."""
        task = self._tasks.get(task_id)
        if not task:
            return False

        # Remove from old agent
        if task.assigned_agent:
            old_agent = self._agents.get(task.assigned_agent)
            if old_agent and task_id in old_agent.current_tasks:
                old_agent.current_tasks.remove(task_id)
                old_agent.update_load()

        # Assign to new agent
        return await self._assign_task(task_id, new_agent_id)

    def _find_best_agent(self, task: Task) -> Optional[Agent]:
        """Find the best agent for a task based on capabilities and load."""
        best_agent: Optional[Agent] = None
        best_score = -1.0

        for agent in self._agents.values():
            if not agent.is_available:
                continue

            # Calculate score based on capability match and load
            score = self._calculate_agent_score(agent, task)
            if score > best_score:
                best_score = score
                best_agent = agent

        return best_agent

    def _calculate_agent_score(self, agent: Agent, task: Task) -> float:
        """Calculate how well an agent matches a task."""
        if not task.required_capabilities:
            # No requirements - prefer lower load
            return 1.0 - agent.load_factor

        # Calculate capability match score
        capability_score = 0.0
        matched_capabilities = 0

        for requirement in task.required_capabilities:
            for capability in agent.capabilities:
                matches, proficiency = capability.matches(requirement)
                if matches:
                    capability_score += proficiency
                    matched_capabilities += 1
                    break

        if matched_capabilities == 0:
            return 0.0  # No capability match

        # Normalize capability score
        capability_score /= len(task.required_capabilities)

        # Combine with load factor (prefer lower load)
        load_score = 1.0 - agent.load_factor

        # Weight: 70% capability, 30% load
        return 0.7 * capability_score + 0.3 * load_score

    def _check_capability_match(self, agent: Agent, task: Task) -> bool:
        """Check if an agent has required capabilities for a task."""
        if not task.required_capabilities:
            return True

        for requirement in task.required_capabilities:
            matched = False
            for capability in agent.capabilities:
                matches, _ = capability.matches(requirement)
                if matches:
                    matched = True
                    break
            if not matched:
                return False

        return True

    # -------------------------------------------------------------------------
    # Dependency Management
    # -------------------------------------------------------------------------

    def _check_dependencies_satisfied(self, task: Task) -> bool:
        """Check if all task dependencies are satisfied."""
        for dep in task.dependencies:
            dep_task = self.get_task(dep.task_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        return True

    async def _check_dependent_tasks(self, completed_task_id: str) -> None:
        """Check if completing a task unblocks any dependent tasks."""
        for task in self._tasks.values():
            if task.status != TaskStatus.BLOCKED:
                continue

            # Check if this task depends on the completed task
            depends_on_completed = any(
                dep.task_id == completed_task_id
                for dep in task.dependencies
            )

            if depends_on_completed and self._check_dependencies_satisfied(task):
                task.status = TaskStatus.QUEUED
                self._enqueue_task(task.task_id)

    async def _check_parent_completion(self, parent_task_id: str) -> None:
        """Check if all subtasks of a parent are complete."""
        parent = self._tasks.get(parent_task_id)
        if not parent:
            return

        all_complete = True
        any_failed = False
        results = {}

        for subtask_id in parent.subtask_ids:
            subtask = self.get_task(subtask_id)
            if subtask:
                if subtask.status == TaskStatus.COMPLETED:
                    results[subtask_id] = subtask.result
                elif subtask.status == TaskStatus.FAILED:
                    any_failed = True
                    results[subtask_id] = f"FAILED: {subtask.error}"
                else:
                    all_complete = False

        if all_complete:
            if any_failed:
                await self.complete_task(parent_task_id, error="One or more subtasks failed")
            else:
                await self.complete_task(parent_task_id, result=results)

    # -------------------------------------------------------------------------
    # Queue Management
    # -------------------------------------------------------------------------

    def _enqueue_task(self, task_id: str) -> None:
        """Add a task to the priority queue."""
        if task_id not in self._task_queue:
            self._task_queue.append(task_id)

    def _sort_task_queue(self) -> None:
        """Sort the task queue by priority."""
        priority_order = {
            TaskPriority.CRITICAL: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.NORMAL: 2,
            TaskPriority.LOW: 3,
            TaskPriority.BACKGROUND: 4,
        }

        def get_priority(task_id: str) -> int:
            task = self._tasks.get(task_id)
            if task:
                return priority_order.get(task.priority, 5)
            return 999

        self._task_queue.sort(key=get_priority)

    # -------------------------------------------------------------------------
    # Swarm Behaviors
    # -------------------------------------------------------------------------

    async def _run_swarm_behaviors(self) -> None:
        """Background task to run swarm behaviors periodically."""
        while self._is_running:
            try:
                for behavior in self._swarm_behaviors:
                    await behavior.execute(self)

                # Also assign any pending tasks
                await self.assign_pending_tasks()

            except Exception:
                pass  # Swarm behaviors should not crash orchestrator

            await asyncio.sleep(self._behavior_interval)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start the orchestrator."""
        if self._is_running:
            return

        self._is_running = True
        self._start_time = time.time()

        # Start swarm behavior loop
        if self._swarm_behaviors:
            self._behavior_task = asyncio.create_task(self._run_swarm_behaviors())

    async def stop(self) -> None:
        """Stop the orchestrator."""
        self._is_running = False

        if self._behavior_task:
            self._behavior_task.cancel()
            try:
                await self._behavior_task
            except asyncio.CancelledError:
                pass

    # -------------------------------------------------------------------------
    # Metrics and Observability
    # -------------------------------------------------------------------------

    def _update_agent_metrics(self) -> None:
        """Update agent-related metrics."""
        self._metrics.active_agents = sum(
            1 for a in self._agents.values()
            if a.status in (AgentStatus.BUSY, AgentStatus.OVERLOADED)
        )
        self._metrics.idle_agents = sum(
            1 for a in self._agents.values()
            if a.status == AgentStatus.IDLE
        )

        if self._agents:
            self._metrics.agent_utilization = sum(
                a.load_factor for a in self._agents.values()
            ) / len(self._agents)

    def _update_metrics(self) -> None:
        """Update task-related metrics."""
        if self._task_durations:
            self._metrics.average_task_duration_ms = sum(self._task_durations) / len(self._task_durations)

        elapsed = time.time() - self._start_time
        if elapsed > 0:
            self._metrics.tasks_per_second = self._metrics.total_tasks_completed / elapsed

        self._update_agent_metrics()

    def get_metrics(self) -> OrchestratorMetrics:
        """Get current orchestration metrics."""
        self._update_metrics()
        return self._metrics

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status."""
        return {
            "orchestrator_id": self.orchestrator_id,
            "topology": self._topology.value,
            "effective_topology": self._effective_topology.value,
            "is_running": self._is_running,
            "agents": {
                "total": len(self._agents),
                "active": self._metrics.active_agents,
                "idle": self._metrics.idle_agents,
            },
            "tasks": {
                "pending": len(self._task_queue),
                "in_progress": sum(1 for t in self._tasks.values() if t.status == TaskStatus.IN_PROGRESS),
                "completed": self._metrics.total_tasks_completed,
                "failed": self._metrics.total_tasks_failed,
            },
            "metrics": {
                "tasks_per_second": round(self._metrics.tasks_per_second, 2),
                "avg_duration_ms": round(self._metrics.average_task_duration_ms, 2),
                "agent_utilization": round(self._metrics.agent_utilization, 2),
                "work_steals": self._metrics.work_steals,
                "topology_switches": self._metrics.topology_switches,
            },
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_orchestrator(
    orchestrator_id: str = "",
    topology: Topology = Topology.HYBRID,
    enable_swarm: bool = True,
) -> Orchestrator:
    """Create an orchestrator instance with default configuration."""
    return Orchestrator(
        orchestrator_id=orchestrator_id,
        topology=topology,
        enable_swarm_behaviors=enable_swarm,
    )


def create_decomposition(
    strategy_type: str,
    **kwargs,
) -> DecompositionStrategy:
    """Factory for creating decomposition strategies."""
    if strategy_type == "sequential":
        return SequentialDecomposition(kwargs.get("steps", []))
    elif strategy_type == "parallel":
        return ParallelDecomposition(kwargs.get("subtasks", []))
    elif strategy_type == "dag":
        return DAGDecomposition(kwargs.get("nodes", []), kwargs.get("edges", []))
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


# =============================================================================
# DEMO
# =============================================================================


async def demo():
    """Demonstrate orchestrator capabilities."""
    print("=" * 60)
    print("Agent Orchestrator Demo")
    print("=" * 60)

    # Create orchestrator
    orch = create_orchestrator(
        orchestrator_id="demo-orchestrator",
        topology=Topology.HYBRID,
        enable_swarm=True,
    )

    # Register agents with different capabilities
    orch.register_agent(
        "agent-coder",
        role=AgentRole.SPECIALIST,
        capabilities=[
            AgentCapability("python", proficiency=0.9),
            AgentCapability("testing", proficiency=0.7),
        ],
        max_concurrent_tasks=3,
    )

    orch.register_agent(
        "agent-researcher",
        role=AgentRole.SPECIALIST,
        capabilities=[
            AgentCapability("research", proficiency=0.95),
            AgentCapability("documentation", proficiency=0.8),
        ],
        max_concurrent_tasks=2,
    )

    orch.register_agent(
        "agent-general",
        role=AgentRole.GENERALIST,
        capabilities=[
            AgentCapability("python", proficiency=0.6),
            AgentCapability("research", proficiency=0.5),
        ],
        max_concurrent_tasks=5,
    )

    print(f"\nRegistered {len(orch._agents)} agents")

    # Start orchestrator
    await orch.start()
    print("Orchestrator started")

    # Submit simple tasks
    task1_id = orch.submit_task(
        "Write Python unit tests",
        priority=TaskPriority.HIGH,
        required_capabilities=["python", "testing"],
    )
    print(f"\nSubmitted task: {task1_id}")

    task2_id = orch.submit_task(
        "Research best practices",
        priority=TaskPriority.NORMAL,
        required_capabilities=["research"],
    )
    print(f"Submitted task: {task2_id}")

    # Assign tasks
    assigned = await orch.assign_pending_tasks()
    print(f"\nAssigned {assigned} tasks")

    # Show assignments
    for task_id in [task1_id, task2_id]:
        task = orch.get_task(task_id)
        if task:
            print(f"  - {task.description}: assigned to {task.assigned_agent}")

    # Submit a complex task and decompose it
    complex_task_id = orch.submit_task(
        "Build and test a new feature",
        priority=TaskPriority.HIGH,
        required_capabilities=["python"],
    )

    # Decompose into sequential steps
    decomposition = orch.decompose_task(
        complex_task_id,
        SequentialDecomposition([
            "Design the feature architecture",
            "Implement core functionality",
            "Write unit tests",
            "Run integration tests",
        ])
    )

    print(f"\nDecomposed task into {len(decomposition.subtasks)} subtasks:")
    for subtask in decomposition.subtasks:
        deps = [d.task_id for d in subtask.dependencies]
        print(f"  - {subtask.task_id}: {subtask.description[:40]}... (deps: {deps})")

    # Simulate task completion
    await orch.complete_task(task1_id, result="Tests written successfully")
    await orch.complete_task(task2_id, result="Research complete")

    # Complete subtasks sequentially
    for subtask in decomposition.subtasks:
        # Assign the subtask
        await orch.assign_pending_tasks()
        subtask = orch.get_task(subtask.task_id)
        if subtask and subtask.status == TaskStatus.ASSIGNED:
            await orch.complete_task(subtask.task_id, result=f"Completed: {subtask.description[:30]}")

    # Get status
    status = orch.get_status()
    print(f"\n{'-' * 60}")
    print("Orchestrator Status:")
    print(f"  Topology: {status['effective_topology']}")
    print(f"  Agents: {status['agents']['total']} total, {status['agents']['active']} active")
    print(f"  Tasks completed: {status['tasks']['completed']}")
    print(f"  Tasks failed: {status['tasks']['failed']}")
    print(f"  Work steals: {status['metrics']['work_steals']}")

    # Stop orchestrator
    await orch.stop()
    print("\nOrchestrator stopped")


if __name__ == "__main__":
    asyncio.run(demo())
