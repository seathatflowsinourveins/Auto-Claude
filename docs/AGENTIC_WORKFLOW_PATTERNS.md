# Agentic Workflow Patterns for Claude Code CLI

## Overview

This document provides comprehensive patterns and best practices for building autonomous AI agents using Claude Code CLI. These patterns enable robust, fault-tolerant, and maintainable agentic systems that can operate autonomously while maintaining safety and reliability.

### Key Principles

1. **State Management** - Explicit state machines for predictable behavior
2. **Fault Tolerance** - Graceful degradation and recovery mechanisms
3. **Observability** - Comprehensive logging and checkpointing
4. **Human Oversight** - Strategic intervention points

---

## 1. Agent Lifecycle Patterns

### State Machine Architecture

```
IDLE → PLANNING → EXECUTING → CHECKPOINTING → COMPLETED
       ↑______________|________________|
            (retry/resume)
                    |
                    ↓
                 FAILED
```

### Implementation

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, Callable, Awaitable
import asyncio
import json
from datetime import datetime


class AgentState(Enum):
    """Agent lifecycle states."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    CHECKPOINTING = "checkpointing"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class ExecutionContext:
    """Context passed through agent execution."""
    task: str
    plan: Optional[List[Dict[str, Any]]] = None
    current_step: int = 0
    results: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class Checkpoint:
    """Serializable checkpoint for resumption."""
    state: AgentState
    context: ExecutionContext
    timestamp: datetime
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "context": {
                "task": self.context.task,
                "plan": self.context.plan,
                "current_step": self.context.current_step,
                "results": self.context.results,
                "metadata": self.context.metadata,
            },
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        context = ExecutionContext(
            task=data["context"]["task"],
            plan=data["context"]["plan"],
            current_step=data["context"]["current_step"],
            results=data["context"]["results"],
            metadata=data["context"]["metadata"],
        )
        return cls(
            state=AgentState(data["state"]),
            context=context,
            timestamp=datetime.fromisoformat(data["timestamp"]),
            version=data.get("version", "1.0"),
        )


class Agent:
    """
    Autonomous agent with state machine lifecycle.
    
    Usage:
        agent = Agent()
        result = await agent.run("Analyze the codebase and suggest improvements")
    """
    
    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        max_retries: int = 3,
    ):
        self.state = AgentState.IDLE
        self.context: Optional[ExecutionContext] = None
        self.checkpoint_dir = checkpoint_dir
        self.max_retries = max_retries
        self._state_handlers: Dict[AgentState, Callable] = {}
        self._transition_hooks: List[Callable] = []
    
    def on_state_change(self, hook: Callable[[AgentState, AgentState], Awaitable[None]]):
        """Register a hook for state transitions."""
        self._transition_hooks.append(hook)
    
    async def _transition_to(self, new_state: AgentState) -> None:
        """Transition to a new state with hooks."""
        old_state = self.state
        for hook in self._transition_hooks:
            await hook(old_state, new_state)
        self.state = new_state
    
    async def run(self, task: str, resume_from: Optional[Checkpoint] = None) -> Any:
        """
        Execute the agent's main loop.
        
        Args:
            task: The task description
            resume_from: Optional checkpoint to resume from
            
        Returns:
            The final result of execution
        """
        if resume_from:
            self.state = resume_from.state
            self.context = resume_from.context
        else:
            self.context = ExecutionContext(task=task, started_at=datetime.now())
        
        retries = 0
        
        while self.state not in (AgentState.COMPLETED, AgentState.FAILED):
            try:
                if self.state == AgentState.IDLE:
                    await self._transition_to(AgentState.PLANNING)
                
                elif self.state == AgentState.PLANNING:
                    self.context.plan = await self.plan(task)
                    await self._transition_to(AgentState.EXECUTING)
                
                elif self.state == AgentState.EXECUTING:
                    result = await self.execute(self.context.plan)
                    self.context.results.append(result)
                    await self._transition_to(AgentState.CHECKPOINTING)
                
                elif self.state == AgentState.CHECKPOINTING:
                    await self.save_checkpoint()
                    if self.context.current_step >= len(self.context.plan):
                        self.context.completed_at = datetime.now()
                        await self._transition_to(AgentState.COMPLETED)
                    else:
                        await self._transition_to(AgentState.EXECUTING)
                
                elif self.state == AgentState.PAUSED:
                    await asyncio.sleep(1)  # Wait for resume signal
                    
            except Exception as e:
                retries += 1
                if retries >= self.max_retries:
                    await self._transition_to(AgentState.FAILED)
                    raise AgentExecutionError(f"Max retries exceeded: {e}") from e
                await self.save_checkpoint()
                await asyncio.sleep(2 ** retries)  # Exponential backoff
        
        return self.context.results
    
    async def plan(self, task: str) -> List[Dict[str, Any]]:
        """
        Generate an execution plan for the task.
        Override this method to implement custom planning logic.
        """
        # Default implementation - override for actual planning
        return [{"action": "execute", "task": task}]
    
    async def execute(self, plan: List[Dict[str, Any]]) -> Any:
        """
        Execute a single step of the plan.
        Override this method to implement custom execution logic.
        """
        step = plan[self.context.current_step]
        self.context.current_step += 1
        # Default implementation - override for actual execution
        return {"step": step, "status": "completed"}
    
    async def save_checkpoint(self) -> None:
        """Persist current state for later resumption."""
        if not self.checkpoint_dir:
            return
        
        checkpoint = Checkpoint(
            state=self.state,
            context=self.context,
            timestamp=datetime.now(),
        )
        
        checkpoint_path = f"{self.checkpoint_dir}/checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint.to_dict(), f, indent=2)
    
    async def pause(self) -> None:
        """Pause execution."""
        await self._transition_to(AgentState.PAUSED)
    
    async def resume(self) -> None:
        """Resume from paused state."""
        if self.state == AgentState.PAUSED:
            await self._transition_to(AgentState.EXECUTING)


class AgentExecutionError(Exception):
    """Raised when agent execution fails."""
    pass
```

### State Transition Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        AGENT LIFECYCLE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────┐    start    ┌──────────┐                            │
│   │ IDLE │────────────▶│ PLANNING │                            │
│   └──────┘             └────┬─────┘                            │
│                             │ plan ready                        │
│                             ▼                                   │
│   ┌────────┐          ┌───────────┐          ┌──────────────┐  │
│   │ FAILED │◀─error───│ EXECUTING │─success─▶│CHECKPOINTING │  │
│   └────────┘          └───────────┘          └──────┬───────┘  │
│        ▲                    ▲                       │          │
│        │                    │ more steps            │          │
│        │                    └───────────────────────┤          │
│        │                                            │          │
│        │               ┌───────────┐                │          │
│        └───max retries─│  PAUSED   │◀──pause────────┤          │
│                        └───────────┘                │          │
│                             │ resume                │ done     │
│                             ▼                       ▼          │
│                        ┌───────────┐          ┌───────────┐   │
│                        │ EXECUTING │          │ COMPLETED │   │
│                        └───────────┘          └───────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Result Type Pattern

A functional approach to error handling that makes success and failure explicit in the type system.

### Implementation

```python
from dataclasses import dataclass
from typing import TypeVar, Generic, Union, Callable, Awaitable, Optional

T = TypeVar('T')
U = TypeVar('U')


@dataclass
class Ok(Generic[T]):
    """Represents a successful result."""
    value: T
    
    def is_ok(self) -> bool:
        return True
    
    def is_err(self) -> bool:
        return False
    
    def unwrap(self) -> T:
        return self.value
    
    def unwrap_or(self, default: T) -> T:
        return self.value
    
    def map(self, fn: Callable[[T], U]) -> "Result[U]":
        return Ok(fn(self.value))
    
    async def map_async(self, fn: Callable[[T], Awaitable[U]]) -> "Result[U]":
        return Ok(await fn(self.value))


@dataclass
class Err:
    """Represents an error result."""
    error: str
    code: str = "UNKNOWN"
    details: Optional[dict] = None
    
    def is_ok(self) -> bool:
        return False
    
    def is_err(self) -> bool:
        return True
    
    def unwrap(self):
        raise ResultError(f"Called unwrap on Err: {self.error}")
    
    def unwrap_or(self, default: T) -> T:
        return default
    
    def map(self, fn: Callable) -> "Err":
        return self
    
    async def map_async(self, fn: Callable) -> "Err":
        return self


Result = Union[Ok[T], Err]


class ResultError(Exception):
    """Raised when unwrapping an Err."""
    pass


# Helper functions
def ok(value: T) -> Ok[T]:
    """Create a successful result."""
    return Ok(value)


def err(error: str, code: str = "UNKNOWN", details: Optional[dict] = None) -> Err:
    """Create an error result."""
    return Err(error, code, details)


# Pattern matching helper
def match_result(
    result: Result[T],
    on_ok: Callable[[T], U],
    on_err: Callable[[Err], U]
) -> U:
    """Pattern match on a Result."""
    if isinstance(result, Ok):
        return on_ok(result.value)
    return on_err(result)


# Usage example
async def fetch_data(url: str) -> Result[dict]:
    """Example function returning a Result type."""
    try:
        # Simulated fetch
        data = {"status": "success", "url": url}
        return ok(data)
    except Exception as e:
        return err(str(e), code="FETCH_ERROR")


async def process_with_result():
    """Demonstrate Result type usage."""
    result = await fetch_data("https://api.example.com/data")
    
    # Pattern matching
    output = match_result(
        result,
        on_ok=lambda data: f"Got data: {data}",
        on_err=lambda e: f"Error {e.code}: {e.error}"
    )
    
    # Chaining with map
    transformed = result.map(lambda d: d.get("status", "unknown"))
    
    # Safe unwrapping
    value = result.unwrap_or({"status": "default"})
    
    return output
```

---

## 3. Multi-Agent Coordination

### Swarm Patterns

#### Hierarchical (Master-Worker)

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import asyncio


class WorkerStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    FAILED = "failed"


@dataclass
class Task:
    id: str
    description: str
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)
    result: Optional[Any] = None


@dataclass
class Worker:
    id: str
    status: WorkerStatus = WorkerStatus.IDLE
    current_task: Optional[Task] = None
    capabilities: List[str] = field(default_factory=list)


class MasterAgent:
    """
    Master agent that coordinates worker agents in a hierarchical pattern.
    """
    
    def __init__(self, max_workers: int = 5):
        self.workers: Dict[str, Worker] = {}
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.completed_tasks: Dict[str, Task] = {}
        self.max_workers = max_workers
        self._running = False
    
    def register_worker(self, worker_id: str, capabilities: List[str] = None) -> Worker:
        """Register a new worker agent."""
        worker = Worker(id=worker_id, capabilities=capabilities or [])
        self.workers[worker_id] = worker
        return worker
    
    async def submit_task(self, task: Task) -> None:
        """Submit a task to the queue."""
        # Priority queue uses (priority, task) tuples; negate for highest-first
        await self.task_queue.put((-task.priority, task))
    
    async def assign_task(self, worker: Worker, task: Task) -> Result[Any]:
        """Assign a task to a worker."""
        worker.status = WorkerStatus.BUSY
        worker.current_task = task
        
        try:
            # Execute task (would be actual worker execution)
            result = await self._execute_on_worker(worker, task)
            task.result = result
            self.completed_tasks[task.id] = task
            worker.status = WorkerStatus.IDLE
            worker.current_task = None
            return ok(result)
        except Exception as e:
            worker.status = WorkerStatus.FAILED
            return err(str(e), code="WORKER_EXECUTION_ERROR")
    
    async def _execute_on_worker(self, worker: Worker, task: Task) -> Any:
        """Execute task on worker - override for actual implementation."""
        await asyncio.sleep(0.1)  # Simulated work
        return {"task_id": task.id, "worker_id": worker.id, "status": "completed"}
    
    def get_available_worker(self, required_capabilities: List[str] = None) -> Optional[Worker]:
        """Find an available worker with required capabilities."""
        for worker in self.workers.values():
            if worker.status == WorkerStatus.IDLE:
                if not required_capabilities:
                    return worker
                if all(cap in worker.capabilities for cap in required_capabilities):
                    return worker
        return None
    
    async def run(self) -> None:
        """Main coordination loop."""
        self._running = True
        
        while self._running:
            if self.task_queue.empty():
                await asyncio.sleep(0.1)
                continue
            
            worker = self.get_available_worker()
            if not worker:
                await asyncio.sleep(0.1)
                continue
            
            _, task = await self.task_queue.get()
            
            # Check dependencies
            deps_met = all(
                dep_id in self.completed_tasks 
                for dep_id in task.dependencies
            )
            
            if deps_met:
                asyncio.create_task(self.assign_task(worker, task))
            else:
                # Re-queue with lower priority
                await self.task_queue.put((-task.priority + 1, task))
    
    async def stop(self) -> None:
        """Stop the coordination loop."""
        self._running = False
```

#### Mesh (Peer-to-Peer)

```python
from dataclasses import dataclass, field
from typing import Dict, Set, Any, Callable, Awaitable
import asyncio


@dataclass
class PeerMessage:
    sender_id: str
    message_type: str
    payload: Any
    timestamp: float


class PeerAgent:
    """
    Peer agent for mesh-based coordination.
    Agents communicate directly with each other.
    """
    
    def __init__(self, agent_id: str):
        self.id = agent_id
        self.peers: Dict[str, "PeerAgent"] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.inbox: asyncio.Queue = asyncio.Queue()
        self._running = False
    
    def connect(self, peer: "PeerAgent") -> None:
        """Establish bidirectional connection with peer."""
        self.peers[peer.id] = peer
        peer.peers[self.id] = self
    
    def disconnect(self, peer_id: str) -> None:
        """Disconnect from peer."""
        if peer_id in self.peers:
            peer = self.peers.pop(peer_id)
            if self.id in peer.peers:
                peer.peers.pop(self.id)
    
    def on_message(self, message_type: str, handler: Callable[[PeerMessage], Awaitable[Any]]):
        """Register a message handler."""
        self.message_handlers[message_type] = handler
    
    async def send(self, peer_id: str, message_type: str, payload: Any) -> Result[None]:
        """Send message to a specific peer."""
        if peer_id not in self.peers:
            return err(f"Unknown peer: {peer_id}", code="UNKNOWN_PEER")
        
        message = PeerMessage(
            sender_id=self.id,
            message_type=message_type,
            payload=payload,
            timestamp=asyncio.get_event_loop().time()
        )
        await self.peers[peer_id].inbox.put(message)
        return ok(None)
    
    async def broadcast(self, message_type: str, payload: Any) -> None:
        """Broadcast message to all peers."""
        for peer_id in self.peers:
            await self.send(peer_id, message_type, payload)
    
    async def run(self) -> None:
        """Process incoming messages."""
        self._running = True
        
        while self._running:
            try:
                message = await asyncio.wait_for(self.inbox.get(), timeout=1.0)
                
                if message.message_type in self.message_handlers:
                    handler = self.message_handlers[message.message_type]
                    await handler(message)
                    
            except asyncio.TimeoutError:
                continue
    
    async def stop(self) -> None:
        """Stop message processing."""
        self._running = False


# Example usage
async def setup_mesh_network():
    """Create a mesh network of peer agents."""
    agents = [PeerAgent(f"agent_{i}") for i in range(5)]
    
    # Connect all agents in a mesh
    for i, agent in enumerate(agents):
        for j, other in enumerate(agents):
            if i != j:
                agent.connect(other)
    
    # Register handlers
    async def handle_task_request(message: PeerMessage):
        print(f"Received task from {message.sender_id}: {message.payload}")
    
    for agent in agents:
        agent.on_message("task_request", handle_task_request)
    
    return agents
```

#### Pipeline (Sequential)

```python
from dataclasses import dataclass
from typing import List, Any, Callable, Awaitable, Optional
import asyncio


@dataclass
class PipelineStage:
    name: str
    handler: Callable[[Any], Awaitable[Any]]
    timeout: Optional[float] = None
    retry_count: int = 1


class PipelineAgent:
    """
    Pipeline agent for sequential multi-stage processing.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.stages: List[PipelineStage] = []
        self.middleware: List[Callable] = []
    
    def add_stage(
        self,
        name: str,
        handler: Callable[[Any], Awaitable[Any]],
        timeout: Optional[float] = None,
        retry_count: int = 1
    ) -> "PipelineAgent":
        """Add a processing stage to the pipeline."""
        self.stages.append(PipelineStage(
            name=name,
            handler=handler,
            timeout=timeout,
            retry_count=retry_count
        ))
        return self
    
    def use_middleware(self, middleware: Callable[[Any, Callable], Awaitable[Any]]) -> "PipelineAgent":
        """Add middleware for cross-cutting concerns."""
        self.middleware.append(middleware)
        return self
    
    async def execute(self, initial_input: Any) -> Result[Any]:
        """Execute the pipeline."""
        current_value = initial_input
        
        for stage in self.stages:
            result = await self._execute_stage(stage, current_value)
            
            if isinstance(result, Err):
                return result
            
            current_value = result.value
        
        return ok(current_value)
    
    async def _execute_stage(self, stage: PipelineStage, input_value: Any) -> Result[Any]:
        """Execute a single stage with retry logic."""
        last_error = None
        
        for attempt in range(stage.retry_count):
            try:
                # Apply middleware
                handler = stage.handler
                for mw in reversed(self.middleware):
                    handler = lambda v, h=handler, m=mw: m(v, h)
                
                # Execute with optional timeout
                if stage.timeout:
                    result = await asyncio.wait_for(
                        handler(input_value),
                        timeout=stage.timeout
                    )
                else:
                    result = await handler(input_value)
                
                return ok(result)
                
            except asyncio.TimeoutError:
                last_error = f"Stage '{stage.name}' timed out after {stage.timeout}s"
            except Exception as e:
                last_error = str(e)
            
            if attempt < stage.retry_count - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return err(last_error, code=f"STAGE_{stage.name.upper()}_FAILED")


# Example usage
async def build_analysis_pipeline():
    """Build a code analysis pipeline."""
    
    async def parse_code(code: str) -> dict:
        return {"ast": "parsed", "code": code}
    
    async def analyze_complexity(data: dict) -> dict:
        data["complexity"] = "low"
        return data
    
    async def generate_report(data: dict) -> str:
        return f"Analysis complete: {data}"
    
    # Logging middleware
    async def logging_middleware(value: Any, next_handler: Callable) -> Any:
        print(f"Processing: {type(value)}")
        result = await next_handler(value)
        print(f"Result: {type(result)}")
        return result
    
    pipeline = (
        PipelineAgent("code-analyzer")
        .use_middleware(logging_middleware)
        .add_stage("parse", parse_code, timeout=30.0)
        .add_stage("analyze", analyze_complexity, timeout=60.0, retry_count=3)
        .add_stage("report", generate_report, timeout=10.0)
    )
    
    return pipeline
```

### Claude-Flow Integration

```yaml
# claude-flow.yaml - Multi-agent workflow configuration
version: "1.0"

agents:
  planner:
    role: "Task Planner"
    model: "claude-sonnet-4-20250514"
    system_prompt: |
      You are a planning agent. Break down complex tasks into
      actionable steps that can be executed by worker agents.
    
  executor:
    role: "Task Executor"
    model: "claude-sonnet-4-20250514"
    system_prompt: |
      You are an execution agent. Complete tasks assigned to you
      and report results back to the coordinator.
    
  reviewer:
    role: "Quality Reviewer"
    model: "claude-sonnet-4-20250514"
    system_prompt: |
      You are a review agent. Verify the quality of completed
      work and suggest improvements.

workflow:
  type: "hierarchical"
  coordinator: "planner"
  
  stages:
    - name: "planning"
      agent: "planner"
      output: "task_plan"
      
    - name: "execution"
      agent: "executor"
      input: "task_plan"
      parallel: true
      max_concurrent: 3
      
    - name: "review"
      agent: "reviewer"
      input: "execution_results"
      
checkpoints:
  enabled: true
  interval: "after_each_stage"
  storage: "./checkpoints"
```

```python
# claude_flow_runner.py
from dataclasses import dataclass
from typing import Dict, Any, List
import yaml
import asyncio


@dataclass
class ClaudeFlowConfig:
    """Configuration for Claude Flow workflows."""
    agents: Dict[str, Dict[str, Any]]
    workflow: Dict[str, Any]
    checkpoints: Dict[str, Any]
    
    @classmethod
    def from_yaml(cls, path: str) -> "ClaudeFlowConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            agents=data.get("agents", {}),
            workflow=data.get("workflow", {}),
            checkpoints=data.get("checkpoints", {})
        )


class ClaudeFlowRunner:
    """Execute Claude Flow workflows."""
    
    def __init__(self, config: ClaudeFlowConfig):
        self.config = config
        self.agents: Dict[str, Agent] = {}
        self.results: Dict[str, Any] = {}
    
    async def initialize(self) -> None:
        """Initialize all agents from config."""
        for agent_id, agent_config in self.config.agents.items():
            self.agents[agent_id] = await self._create_agent(agent_id, agent_config)
    
    async def _create_agent(self, agent_id: str, config: Dict[str, Any]) -> Agent:
        """Create an agent instance from config."""
        # Implementation would create actual LLM-backed agents
        return Agent()
    
    async def run(self, task: str) -> Result[Dict[str, Any]]:
        """Execute the workflow."""
        await self.initialize()
        
        for stage in self.config.workflow.get("stages", []):
            stage_name = stage["name"]
            agent_id = stage["agent"]
            
            if stage.get("parallel", False):
                results = await self._run_parallel(agent_id, stage)
            else:
                results = await self._run_sequential(agent_id, stage)
            
            self.results[stage_name] = results
            
            if self.config.checkpoints.get("enabled"):
                await self._save_checkpoint(stage_name)
        
        return ok(self.results)
    
    async def _run_parallel(self, agent_id: str, stage: Dict[str, Any]) -> List[Any]:
        """Run stage with parallel execution."""
        max_concurrent = stage.get("max_concurrent", 5)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_limit(task):
            async with semaphore:
                return await self.agents[agent_id].run(task)
        
        # Would get tasks from previous stage
        tasks = self.results.get(stage.get("input"), [])
        return await asyncio.gather(*[run_with_limit(t) for t in tasks])
    
    async def _run_sequential(self, agent_id: str, stage: Dict[str, Any]) -> Any:
        """Run stage sequentially."""
        input_data = self.results.get(stage.get("input"))
        return await self.agents[agent_id].run(str(input_data))
    
    async def _save_checkpoint(self, stage_name: str) -> None:
        """Save checkpoint after stage completion."""
        checkpoint_dir = self.config.checkpoints.get("storage", "./checkpoints")
        # Implementation would save checkpoint to storage
        pass
```

---

## 4. Memory & Context Management

### mem0 Integration Pattern

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
import json


@dataclass
class Memory:
    """A single memory entry."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    importance: float = 0.5


class MemoryStore:
    """
    Memory management system inspired by mem0.
    Provides storage, retrieval, and context management.
    """
    
    def __init__(
        self,
        max_memories: int = 1000,
        relevance_threshold: float = 0.7
    ):
        self.memories: Dict[str, Memory] = {}
        self.max_memories = max_memories
        self.relevance_threshold = relevance_threshold
        self._embeddings_cache: Dict[str, List[float]] = {}
    
    def _generate_id(self, content: str) -> str:
        """Generate unique ID for memory."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def store(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5
    ) -> Memory:
        """Store a new memory."""
        memory_id = self._generate_id(content)
        
        # Check for duplicates
        if memory_id in self.memories:
            existing = self.memories[memory_id]
            existing.access_count += 1
            existing.last_accessed = datetime.now()
            return existing
        
        # Create embedding (would use actual embedding model)
        embedding = await self._create_embedding(content)
        
        memory = Memory(
            id=memory_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            importance=importance
        )
        
        # Evict old memories if at capacity
        if len(self.memories) >= self.max_memories:
            await self._evict_least_important()
        
        self.memories[memory_id] = memory
        return memory
    
    async def recall(
        self,
        query: str,
        limit: int = 5,
        min_relevance: Optional[float] = None
    ) -> List[Memory]:
        """Recall memories relevant to query."""
        threshold = min_relevance or self.relevance_threshold
        query_embedding = await self._create_embedding(query)
        
        scored_memories = []
        for memory in self.memories.values():
            if memory.embedding:
                score = self._cosine_similarity(query_embedding, memory.embedding)
                if score >= threshold:
                    scored_memories.append((score, memory))
        
        # Sort by relevance and return top results
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, memory in scored_memories[:limit]:
            memory.access_count += 1
            memory.last_accessed = datetime.now()
            results.append(memory)
        
        return results
    
    async def search(
        self,
        filters: Dict[str, Any],
        limit: int = 10
    ) -> List[Memory]:
        """Search memories by metadata filters."""
        results = []
        
        for memory in self.memories.values():
            match = all(
                memory.metadata.get(k) == v 
                for k, v in filters.items()
            )
            if match:
                results.append(memory)
                if len(results) >= limit:
                    break
        
        return results
    
    async def forget(self, memory_id: str) -> bool:
        """Remove a specific memory."""
        if memory_id in self.memories:
            del self.memories[memory_id]
            return True
        return False
    
    async def summarize(self, memories: List[Memory]) -> str:
        """Summarize multiple memories into coherent context."""
        # Would use LLM for actual summarization
        contents = [m.content for m in memories]
        return " | ".join(contents)
    
    async def _create_embedding(self, text: str) -> List[float]:
        """Create embedding vector for text."""
        # Would use actual embedding model (e.g., OpenAI, sentence-transformers)
        # This is a placeholder
        return [0.0] * 1536
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
    
    async def _evict_least_important(self) -> None:
        """Remove least important memory."""
        if not self.memories:
            return
        
        # Score by importance and recency
        def score(m: Memory) -> float:
            age_hours = (datetime.now() - m.last_accessed).total_seconds() / 3600
            recency_score = 1 / (1 + age_hours)
            return m.importance * 0.6 + recency_score * 0.4
        
        least_important = min(self.memories.values(), key=score)
        del self.memories[least_important.id]


# Context window management
class ContextManager:
    """
    Manages context window for LLM interactions.
    Handles token counting and context pruning.
    """
    
    def __init__(
        self,
        max_tokens: int = 100000,
        reserved_output_tokens: int = 4096
    ):
        self.max_tokens = max_tokens
        self.reserved_output_tokens = reserved_output_tokens
        self.available_tokens = max_tokens - reserved_output_tokens
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        Uses approximation; would use actual tokenizer in production.
        """
        # Rough approximation: ~4 characters per token
        return len(text) // 4
    
    def fit_context(
        self,
        system_prompt: str,
        memories: List[Memory],
        user_message: str,
        priority_order: List[str] = None
    ) -> Dict[str, Any]:
        """
        Fit context within token budget.
        
        Returns dict with fitted context and metadata.
        """
        system_tokens = self.count_tokens(system_prompt)
        user_tokens = self.count_tokens(user_message)
        
        remaining = self.available_tokens - system_tokens - user_tokens
        
        if remaining <= 0:
            # User message too long - truncate
            max_user = self.available_tokens - system_tokens - 1000
            user_message = self._truncate(user_message, max_user)
            remaining = 1000
        
        # Add memories by priority/relevance
        included_memories = []
        memory_text = ""
        
        for memory in memories:
            memory_tokens = self.count_tokens(memory.content)
            if memory_tokens <= remaining:
                included_memories.append(memory)
                memory_text += f"\n- {memory.content}"
                remaining -= memory_tokens
            else:
                break
        
        return {
            "system_prompt": system_prompt,
            "memory_context": memory_text,
            "user_message": user_message,
            "total_tokens": self.available_tokens - remaining,
            "memories_included": len(included_memories),
            "memories_excluded": len(memories) - len(included_memories),
        }
    
    def _truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit token budget."""
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars - 3] + "..."
```

### Token Budget Management

```python
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class PruningStrategy(Enum):
    """Strategies for pruning context when over budget."""
    OLDEST_FIRST = "oldest_first"
    LOWEST_RELEVANCE = "lowest_relevance"
    SUMMARIZE = "summarize"
    SLIDING_WINDOW = "sliding_window"


@dataclass
class TokenBudget:
    """Token budget allocation."""
    total: int
    system: int
    memory: int
    conversation: int
    output_reserve: int
    
    @property
    def available_input(self) -> int:
        return self.total - self.output_reserve
    
    def validate(self) -> bool:
        allocated = self.system + self.memory + self.conversation
        return allocated <= self.available_input


class TokenBudgetManager:
    """
    Manages token budget across different context components.
    """
    
    def __init__(
        self,
        model_context_limit: int = 200000,
        default_output_reserve: int = 8192
    ):
        self.model_context_limit = model_context_limit
        self.default_output_reserve = default_output_reserve
    
    def create_budget(
        self,
        system_ratio: float = 0.1,
        memory_ratio: float = 0.3,
        conversation_ratio: float = 0.5,
        output_reserve: Optional[int] = None
    ) -> TokenBudget:
        """Create a token budget with specified ratios."""
        reserve = output_reserve or self.default_output_reserve
        available = self.model_context_limit - reserve
        
        return TokenBudget(
            total=self.model_context_limit,
            system=int(available * system_ratio),
            memory=int(available * memory_ratio),
            conversation=int(available * conversation_ratio),
            output_reserve=reserve
        )
    
    def prune_to_budget(
        self,
        messages: List[dict],
        budget: int,
        strategy: PruningStrategy = PruningStrategy.OLDEST_FIRST
    ) -> Tuple[List[dict], int]:
        """
        Prune messages to fit within budget.
        
        Returns (pruned_messages, actual_token_count)
        """
        if strategy == PruningStrategy.OLDEST_FIRST:
            return self._prune_oldest_first(messages, budget)
        elif strategy == PruningStrategy.SLIDING_WINDOW:
            return self._prune_sliding_window(messages, budget)
        elif strategy == PruningStrategy.SUMMARIZE:
            return self._prune_with_summary(messages, budget)
        else:
            return self._prune_oldest_first(messages, budget)
    
    def _count_message_tokens(self, message: dict) -> int:
        """Count tokens in a message."""
        content = message.get("content", "")
        # Rough approximation
        return len(content) // 4 + 4  # +4 for message overhead
    
    def _prune_oldest_first(
        self,
        messages: List[dict],
        budget: int
    ) -> Tuple[List[dict], int]:
        """Remove oldest messages first, keeping system message."""
        if not messages:
            return [], 0
        
        result = []
        total_tokens = 0
        
        # Always keep system message if present
        system_messages = [m for m in messages if m.get("role") == "system"]
        other_messages = [m for m in messages if m.get("role") != "system"]
        
        for msg in system_messages:
            tokens = self._count_message_tokens(msg)
            result.append(msg)
            total_tokens += tokens
        
        # Add messages from newest to oldest
        for msg in reversed(other_messages):
            tokens = self._count_message_tokens(msg)
            if total_tokens + tokens <= budget:
                result.insert(len(system_messages), msg)
                total_tokens += tokens
            else:
                break
        
        return result, total_tokens
    
    def _prune_sliding_window(
        self,
        messages: List[dict],
        budget: int
    ) -> Tuple[List[dict], int]:
        """Keep only the most recent messages within budget."""
        return self._prune_oldest_first(messages, budget)
    
    def _prune_with_summary(
        self,
        messages: List[dict],
        budget: int
    ) -> Tuple[List[dict], int]:
        """Summarize old messages instead of removing them."""
        # Would use LLM to summarize older messages
        # For now, falls back to oldest-first
        return self._prune_oldest_first(messages, budget)
```

---

## 5. Tool Execution Framework

### Pre/Post Hooks Pattern

```python
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Awaitable, List, Optional
from datetime import datetime
import asyncio
import traceback


@dataclass
class ToolInvocation:
    """Record of a tool invocation."""
    tool_name: str
    args: Dict[str, Any]
    started_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None


# Type aliases for hook functions
BeforeHook = Callable[[str, Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]
AfterHook = Callable[[str, Any, Optional[Exception]], Awaitable[None]]
ErrorHook = Callable[[str, Dict[str, Any], Exception], Awaitable[Optional[Any]]]


class ToolExecutor:
    """
    Tool execution framework with hooks and error handling.
    """
    
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.before_hooks: List[BeforeHook] = []
        self.after_hooks: List[AfterHook] = []
        self.error_hooks: List[ErrorHook] = []
        self.invocations: List[ToolInvocation] = []
        self.timeout_default: float = 30.0
    
    def register(
        self,
        name: str,
        fn: Callable[..., Awaitable[Any]],
        description: str = "",
        schema: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a tool with optional schema."""
        self.tools[name] = {
            "fn": fn,
            "description": description,
            "schema": schema or {}
        }
    
    def before(self, hook: BeforeHook) -> None:
        """Register a before-execution hook."""
        self.before_hooks.append(hook)
    
    def after(self, hook: AfterHook) -> None:
        """Register an after-execution hook."""
        self.after_hooks.append(hook)
    
    def on_error(self, hook: ErrorHook) -> None:
        """Register an error hook."""
        self.error_hooks.append(hook)
    
    async def execute(
        self,
        name: str,
        args: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> Result[Any]:
        """
        Execute a tool with full hook lifecycle.
        
        Lifecycle:
        1. Before hooks (can modify args or cancel)
        2. Tool execution
        3. After hooks (for logging, metrics)
        4. Error hooks (on failure, can provide fallback)
        """
        if name not in self.tools:
            return err(f"Unknown tool: {name}", code="UNKNOWN_TOOL")
        
        invocation = ToolInvocation(
            tool_name=name,
            args=args.copy(),
            started_at=datetime.now()
        )
        self.invocations.append(invocation)
        
        # Run before hooks
        modified_args = args
        for hook in self.before_hooks:
            result = await hook(name, modified_args)
            if result is None:
                # Hook cancelled execution
                return err("Execution cancelled by before hook", code="HOOK_CANCELLED")
            modified_args = result
        
        # Execute tool
        try:
            tool_fn = self.tools[name]["fn"]
            execution_timeout = timeout or self.timeout_default
            
            result = await asyncio.wait_for(
                tool_fn(**modified_args),
                timeout=execution_timeout
            )
            
            invocation.completed_at = datetime.now()
            invocation.result = result
            invocation.duration_ms = (
                invocation.completed_at - invocation.started_at
            ).total_seconds() * 1000
            
            # Run after hooks
            for hook in self.after_hooks:
                await hook(name, result, None)
            
            return ok(result)
            
        except asyncio.TimeoutError:
            error = f"Tool '{name}' timed out after {execution_timeout}s"
            invocation.error = error
            
            # Run error hooks
            for hook in self.error_hooks:
                fallback = await hook(name, modified_args, TimeoutError(error))
                if fallback is not None:
                    return ok(fallback)
            
            return err(error, code="TIMEOUT")
            
        except Exception as e:
            invocation.error = str(e)
            invocation.completed_at = datetime.now()
            
            # Run after hooks with error
            for hook in self.after_hooks:
                await hook(name, None, e)
            
            # Run error hooks for potential recovery
            for hook in self.error_hooks:
                fallback = await hook(name, modified_args, e)
                if fallback is not None:
                    return ok(fallback)
            
            return err(str(e), code="EXECUTION_ERROR", details={
                "traceback": traceback.format_exc()
            })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics."""
        if not self.invocations:
            return {}
        
        successful = [i for i in self.invocations if i.error is None]
        failed = [i for i in self.invocations if i.error is not None]
        
        durations = [i.duration_ms for i in successful if i.duration_ms]
        
        return {
            "total_invocations": len(self.invocations),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.invocations),
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "by_tool": self._metrics_by_tool()
        }
    
    def _metrics_by_tool(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics grouped by tool."""
        by_tool: Dict[str, List[ToolInvocation]] = {}
        
        for inv in self.invocations:
            if inv.tool_name not in by_tool:
                by_tool[inv.tool_name] = []
            by_tool[inv.tool_name].append(inv)
        
        return {
            name: {
                "count": len(invs),
                "errors": sum(1 for i in invs if i.error),
                "avg_ms": sum(i.duration_ms or 0 for i in invs) / len(invs)
            }
            for name, invs in by_tool.items()
        }


# Example usage with hooks
async def setup_tool_executor():
    """Set up a tool executor with hooks."""
    executor = ToolExecutor()
    
    # Register tools
    async def read_file(path: str) -> str:
        with open(path) as f:
            return f.read()
    
    async def write_file(path: str, content: str) -> bool:
        with open(path, 'w') as f:
            f.write(content)
        return True
    
    executor.register("read_file", read_file, "Read a file from disk")
    executor.register("write_file", write_file, "Write content to a file")
    
    # Logging hook
    async def log_before(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        print(f"[{datetime.now()}] Executing {name} with {args}")
        return args
    
    async def log_after(name: str, result: Any, error: Optional[Exception]) -> None:
        if error:
            print(f"[{datetime.now()}] {name} failed: {error}")
        else:
            print(f"[{datetime.now()}] {name} completed successfully")
    
    # Validation hook
    async def validate_paths(name: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if "path" in args:
            path = args["path"]
            if ".." in path or path.startswith("/"):
                print(f"Blocked suspicious path: {path}")
                return None  # Cancel execution
        return args
    
    # Error recovery hook
    async def handle_file_error(
        name: str,
        args: Dict[str, Any],
        error: Exception
    ) -> Optional[Any]:
        if name == "read_file" and isinstance(error, FileNotFoundError):
            return ""  # Return empty string for missing files
        return None  # Let error propagate
    
    executor.before(validate_paths)
    executor.before(log_before)
    executor.after(log_after)
    executor.on_error(handle_file_error)
    
    return executor
```

---

## 6. Streaming Patterns

### LLM Response Streaming

```python
from typing import AsyncIterator, Optional, Callable, Awaitable
import asyncio
import httpx


async def stream_llm_response(
    prompt: str,
    api_url: str,
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 4096
) -> AsyncIterator[str]:
    """
    Stream responses from an LLM API.
    
    Yields chunks of text as they arrive.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True
    }
    
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            api_url,
            headers=headers,
            json=payload,
            timeout=120.0
        ) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        break
                    # Parse and yield content
                    # (Actual parsing depends on API format)
                    yield data


class StreamingAccumulator:
    """
    Accumulates streaming chunks with callback support.
    """
    
    def __init__(
        self,
        on_chunk: Optional[Callable[[str], Awaitable[None]]] = None,
        on_complete: Optional[Callable[[str], Awaitable[None]]] = None
    ):
        self.chunks: list = []
        self.on_chunk = on_chunk
        self.on_complete = on_complete
    
    async def process_stream(self, stream: AsyncIterator[str]) -> str:
        """Process a stream, accumulating chunks."""
        async for chunk in stream:
            self.chunks.append(chunk)
            if self.on_chunk:
                await self.on_chunk(chunk)
        
        full_response = "".join(self.chunks)
        
        if self.on_complete:
            await self.on_complete(full_response)
        
        return full_response
    
    @property
    def current_text(self) -> str:
        """Get accumulated text so far."""
        return "".join(self.chunks)


# Streaming with transformation
async def transform_stream(
    source: AsyncIterator[str],
    transform: Callable[[str], str]
) -> AsyncIterator[str]:
    """Apply transformation to each chunk in a stream."""
    async for chunk in source:
        yield transform(chunk)


# Streaming with filtering
async def filter_stream(
    source: AsyncIterator[str],
    predicate: Callable[[str], bool]
) -> AsyncIterator[str]:
    """Filter chunks from a stream."""
    async for chunk in source:
        if predicate(chunk):
            yield chunk
```

### Backpressure Handling

```python
from typing import AsyncIterator, Optional
import asyncio
from dataclasses import dataclass


@dataclass
class BackpressureConfig:
    """Configuration for backpressure handling."""
    max_queue_size: int = 100
    high_watermark: float = 0.8  # Start slowing at 80%
    low_watermark: float = 0.3   # Resume normal at 30%
    slow_interval: float = 0.01  # Delay when under pressure
    check_interval: float = 0.001


class BackpressureQueue:
    """
    Async queue with backpressure support.
    
    Implements flow control to prevent memory exhaustion
    when consumers are slower than producers.
    """
    
    def __init__(self, config: Optional[BackpressureConfig] = None):
        self.config = config or BackpressureConfig()
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        self._under_pressure = False
        self._closed = False
    
    @property
    def pressure_level(self) -> float:
        """Current queue fullness as fraction."""
        return self.queue.qsize() / self.config.max_queue_size
    
    @property
    def under_pressure(self) -> bool:
        """Check if queue is under pressure."""
        level = self.pressure_level
        
        if self._under_pressure and level < self.config.low_watermark:
            self._under_pressure = False
        elif not self._under_pressure and level > self.config.high_watermark:
            self._under_pressure = True
        
        return self._under_pressure
    
    async def put(self, item) -> None:
        """Put item with backpressure handling."""
        if self._closed:
            raise RuntimeError("Queue is closed")
        
        # Apply backpressure
        while self.queue.qsize() >= self.config.max_queue_size:
            await asyncio.sleep(self.config.slow_interval)
        
        if self.under_pressure:
            await asyncio.sleep(self.config.slow_interval)
        
        await self.queue.put(item)
    
    async def get(self) -> any:
        """Get item from queue."""
        return await self.queue.get()
    
    def close(self) -> None:
        """Signal that no more items will be added."""
        self._closed = True


async def process_with_backpressure(
    stream: AsyncIterator[str],
    processor: Callable[[str], Awaitable[None]],
    config: Optional[BackpressureConfig] = None
) -> None:
    """
    Process a stream with backpressure handling.
    
    Uses a bounded queue between producer and consumer
    to prevent memory exhaustion.
    """
    queue = BackpressureQueue(config)
    
    async def producer():
        try:
            async for chunk in stream:
                await queue.put(chunk)
        finally:
            queue.close()
    
    async def consumer():
        while True:
            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=1.0)
                await processor(chunk)
            except asyncio.TimeoutError:
                if queue._closed and queue.queue.empty():
                    break
    
    await asyncio.gather(producer(), consumer())


# Rate-limited streaming
class RateLimitedStream:
    """
    Rate-limited async iterator.
    
    Ensures chunks are yielded no faster than specified rate.
    """
    
    def __init__(
        self,
        source: AsyncIterator[str],
        chunks_per_second: float = 10.0
    ):
        self.source = source
        self.interval = 1.0 / chunks_per_second
        self.last_yield_time = 0.0
    
    def __aiter__(self):
        return self
    
    async def __anext__(self) -> str:
        # Enforce rate limit
        now = asyncio.get_event_loop().time()
        elapsed = now - self.last_yield_time
        
        if elapsed < self.interval:
            await asyncio.sleep(self.interval - elapsed)
        
        chunk = await self.source.__anext__()
        self.last_yield_time = asyncio.get_event_loop().time()
        
        return chunk
```

---

## 7. Error Recovery Patterns

### Exponential Backoff with Jitter

```python
import asyncio
import random
from typing import TypeVar, Callable, Awaitable, Optional, List, Type
from dataclasses import dataclass


T = TypeVar('T')


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter_factor: float = 0.1
    exponential_base: float = 2.0
    retryable_exceptions: Optional[List[Type[Exception]]] = None


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    config: Optional[RetryConfig] = None
) -> T:
    """
    Execute a function with exponential backoff retry.
    
    Args:
        fn: Async function to execute
        config: Retry configuration
        
    Returns:
        Result of successful execution
        
    Raises:
        Last exception if all retries exhausted
    """
    cfg = config or RetryConfig()
    last_exception = None
    
    for attempt in range(cfg.max_attempts):
        try:
            return await fn()
            
        except Exception as e:
            last_exception = e
            
            # Check if exception is retryable
            if cfg.retryable_exceptions:
                if not any(isinstance(e, exc_type) for exc_type in cfg.retryable_exceptions):
                    raise
            
            # Don't sleep after last attempt
            if attempt < cfg.max_attempts - 1:
                delay = calculate_backoff_delay(
                    attempt=attempt,
                    base_delay=cfg.base_delay,
                    max_delay=cfg.max_delay,
                    jitter_factor=cfg.jitter_factor,
                    exponential_base=cfg.exponential_base
                )
                await asyncio.sleep(delay)
    
    raise last_exception


def calculate_backoff_delay(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_factor: float = 0.1,
    exponential_base: float = 2.0
) -> float:
    """
    Calculate delay with exponential backoff and jitter.
    
    Formula: min(base * (exponential_base ^ attempt), max) * (1 + random_jitter)
    """
    # Exponential backoff
    delay = base_delay * (exponential_base ** attempt)
    
    # Cap at max delay
    delay = min(delay, max_delay)
    
    # Add jitter (±jitter_factor percentage)
    jitter = random.uniform(-jitter_factor, jitter_factor)
    delay = delay * (1 + jitter)
    
    return max(0, delay)  # Ensure non-negative


class RetryableOperation:
    """
    Decorator class for retryable operations.
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
    
    def __call__(self, fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        async def wrapper(*args, **kwargs) -> T:
            return await with_retry(
                lambda: fn(*args, **kwargs),
                self.config
            )
        return wrapper


# Usage example
@RetryableOperation(RetryConfig(
    max_attempts=5,
    base_delay=0.5,
    retryable_exceptions=[ConnectionError, TimeoutError]
))
async def fetch_with_retry(url: str) -> dict:
    """Fetch URL with automatic retry."""
    # Implementation
    pass
```

### Circuit Breaker Pattern

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Awaitable, TypeVar, Optional, List
from datetime import datetime, timedelta
import asyncio


T = TypeVar('T')


class CircuitState(Enum):
    """States of the circuit breaker."""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Failing, requests are rejected
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes to close from half-open
    reset_timeout: float = 60.0         # Seconds before half-open
    half_open_max_calls: int = 3        # Max concurrent calls in half-open
    excluded_exceptions: List[type] = field(default_factory=list)


@dataclass
class CircuitStats:
    """Statistics for circuit breaker."""
    failures: int = 0
    successes: int = 0
    rejections: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state_changes: List[tuple] = field(default_factory=list)


class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.
    
    Prevents cascading failures by failing fast when a service is unhealthy.
    
    State transitions:
    - CLOSED -> OPEN: When failures exceed threshold
    - OPEN -> HALF_OPEN: After reset timeout
    - HALF_OPEN -> CLOSED: When successes reach threshold  
    - HALF_OPEN -> OPEN: On any failure
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitStats()
        self._half_open_semaphore = asyncio.Semaphore(
            self.config.half_open_max_calls
        )
        self._lock = asyncio.Lock()
    
    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        """
        Execute function through circuit breaker.
        
        Raises:
            CircuitOpenError: If circuit is open
        """
        async with self._lock:
            self._check_state_transition()
        
        if self.state == CircuitState.OPEN:
            self.stats.rejections += 1
            raise CircuitOpenError(
                f"Circuit '{self.name}' is open. "
                f"Retry after {self._time_until_half_open():.1f}s"
            )
        
        # In half-open, limit concurrent calls
        if self.state == CircuitState.HALF_OPEN:
            if not self._half_open_semaphore.locked():
                acquired = await asyncio.wait_for(
                    self._half_open_semaphore.acquire(),
                    timeout=0.1
                )
                if not acquired:
                    raise CircuitOpenError(
                        f"Circuit '{self.name}' half-open limit reached"
                    )
        
        try:
            result = await fn()
            await self._on_success()
            return result
            
        except Exception as e:
            # Check if exception should be counted
            if not any(isinstance(e, exc) for exc in self.config.excluded_exceptions):
                await self._on_failure()
            raise
        
        finally:
            if self.state == CircuitState.HALF_OPEN:
                self._half_open_semaphore.release()
    
    def _check_state_transition(self) -> None:
        """Check if state should transition."""
        if self.state == CircuitState.OPEN:
            if self._time_until_half_open() <= 0:
                self._transition_to(CircuitState.HALF_OPEN)
    
    def _time_until_half_open(self) -> float:
        """Seconds until circuit can transition to half-open."""
        if not self.stats.last_failure_time:
            return 0
        
        elapsed = (datetime.now() - self.stats.last_failure_time).total_seconds()
        return max(0, self.config.reset_timeout - elapsed)
    
    async def _on_success(self) -> None:
        """Handle successful call."""
        async with self._lock:
            self.stats.successes += 1
            self.stats.last_success_time = datetime.now()
            
            if self.state == CircuitState.HALF_OPEN:
                if self.stats.successes >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
    
    async def _on_failure(self) -> None:
        """Handle failed call."""
        async with self._lock:
            self.stats.failures += 1
            self.stats.last_failure_time = datetime.now()
            
            if self.state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self.state == CircuitState.CLOSED:
                if self.stats.failures >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state."""
        old_state = self.state
        self.state = new_state
        self.stats.state_changes.append((old_state, new_state, datetime.now()))
        
        # Reset counters on state change
        if new_state == CircuitState.CLOSED:
            self.stats.failures = 0
        elif new_state == CircuitState.HALF_OPEN:
            self.stats.successes = 0
    
    def get_status(self) -> dict:
        """Get current circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failures": self.stats.failures,
            "successes": self.stats.successes,
            "rejections": self.stats.rejections,
            "time_until_half_open": self._time_until_half_open()
            if self.state == CircuitState.OPEN else None
        }


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# Circuit breaker registry for managing multiple breakers
class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self.breakers: dict[str, CircuitBreaker] = {}
    
    def get_or_create(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker."""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name, config)
        return self.breakers[name]
    
    def get_all_status(self) -> dict:
        """Get status of all circuit breakers."""
        return {
            name: breaker.get_status()
            for name, breaker in self.breakers.items()
        }


# Global registry
circuit_registry = CircuitBreakerRegistry()


# Decorator for circuit breaker
def with_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None
):
    """Decorator to wrap function with circuit breaker."""
    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        breaker = circuit_registry.get_or_create(name, config)
        
        async def wrapper(*args, **kwargs) -> T:
            return await breaker.call(lambda: fn(*args, **kwargs))
        
        return wrapper
    return decorator
```

### Checkpoint/Resume Pattern

```python
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, TypeVar, Generic
from datetime import datetime
import json
import hashlib
from pathlib import Path


T = TypeVar('T')


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    id: str
    created_at: datetime
    version: str
    task_id: str
    step_index: int
    total_steps: int
    tags: List[str] = field(default_factory=list)


@dataclass
class Checkpoint(Generic[T]):
    """
    Serializable checkpoint for task resumption.
    """
    metadata: CheckpointMetadata
    state: T
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize checkpoint to dictionary."""
        return {
            "metadata": {
                "id": self.metadata.id,
                "created_at": self.metadata.created_at.isoformat(),
                "version": self.metadata.version,
                "task_id": self.metadata.task_id,
                "step_index": self.metadata.step_index,
                "total_steps": self.metadata.total_steps,
                "tags": self.metadata.tags
            },
            "state": self._serialize_state(self.state),
            "context": self.context
        }
    
    def _serialize_state(self, state: T) -> Any:
        """Serialize state - override for custom types."""
        if hasattr(state, 'to_dict'):
            return state.to_dict()
        if hasattr(state, '__dict__'):
            return state.__dict__
        return state
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], state_factory: callable = None) -> "Checkpoint[T]":
        """Deserialize checkpoint from dictionary."""
        metadata = CheckpointMetadata(
            id=data["metadata"]["id"],
            created_at=datetime.fromisoformat(data["metadata"]["created_at"]),
            version=data["metadata"]["version"],
            task_id=data["metadata"]["task_id"],
            step_index=data["metadata"]["step_index"],
            total_steps=data["metadata"]["total_steps"],
            tags=data["metadata"].get("tags", [])
        )
        
        state = data["state"]
        if state_factory:
            state = state_factory(state)
        
        return cls(
            metadata=metadata,
            state=state,
            context=data.get("context", {})
        )


class CheckpointManager:
    """
    Manages checkpoints for task resumption.
    """
    
    def __init__(
        self,
        storage_dir: str,
        version: str = "1.0",
        max_checkpoints: int = 10
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.version = version
        self.max_checkpoints = max_checkpoints
    
    def _generate_id(self, task_id: str, step_index: int) -> str:
        """Generate unique checkpoint ID."""
        content = f"{task_id}:{step_index}:{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def save(
        self,
        task_id: str,
        step_index: int,
        total_steps: int,
        state: T,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> Checkpoint[T]:
        """Save a checkpoint."""
        checkpoint = Checkpoint(
            metadata=CheckpointMetadata(
                id=self._generate_id(task_id, step_index),
                created_at=datetime.now(),
                version=self.version,
                task_id=task_id,
                step_index=step_index,
                total_steps=total_steps,
                tags=tags or []
            ),
            state=state,
            context=context or {}
        )
        
        # Save to file
        filepath = self.storage_dir / f"{task_id}_{checkpoint.metadata.id}.json"
        with open(filepath, 'w') as f:
            json.dump(checkpoint.to_dict(), f, indent=2, default=str)
        
        # Cleanup old checkpoints
        await self._cleanup_old_checkpoints(task_id)
        
        return checkpoint
    
    async def load_latest(
        self,
        task_id: str,
        state_factory: callable = None
    ) -> Optional[Checkpoint[T]]:
        """Load the most recent checkpoint for a task."""
        checkpoints = await self.list_checkpoints(task_id)
        
        if not checkpoints:
            return None
        
        # Sort by step index descending
        latest = max(checkpoints, key=lambda c: c["step_index"])
        
        filepath = self.storage_dir / f"{task_id}_{latest['id']}.json"
        with open(filepath) as f:
            data = json.load(f)
        
        return Checkpoint.from_dict(data, state_factory)
    
    async def list_checkpoints(self, task_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for a task."""
        checkpoints = []
        
        for filepath in self.storage_dir.glob(f"{task_id}_*.json"):
            with open(filepath) as f:
                data = json.load(f)
                checkpoints.append({
                    "id": data["metadata"]["id"],
                    "step_index": data["metadata"]["step_index"],
                    "created_at": data["metadata"]["created_at"],
                    "filepath": str(filepath)
                })
        
        return sorted(checkpoints, key=lambda x: x["step_index"])
    
    async def delete_checkpoint(self, task_id: str, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint."""
        filepath = self.storage_dir / f"{task_id}_{checkpoint_id}.json"
        
        if filepath.exists():
            filepath.unlink()
            return True
        return False
    
    async def _cleanup_old_checkpoints(self, task_id: str) -> None:
        """Remove old checkpoints beyond max limit."""
        checkpoints = await self.list_checkpoints(task_id)
        
        if len(checkpoints) > self.max_checkpoints:
            # Keep only the most recent
            to_delete = checkpoints[:-self.max_checkpoints]
            for cp in to_delete:
                await self.delete_checkpoint(task_id, cp["id"])


# Resumable task wrapper
class ResumableTask:
    """
    Wrapper for tasks that can be checkpointed and resumed.
    """
    
    def __init__(
        self,
        task_id: str,
        checkpoint_manager: CheckpointManager
    ):
        self.task_id = task_id
        self.checkpoint_manager = checkpoint_manager
        self.current_step = 0
        self.total_steps = 0
        self.state: Any = None
    
    async def run(
        self,
        steps: List[Callable[[], Awaitable[Any]]],
        initial_state: Any = None,
        resume: bool = True
    ) -> Any:
        """
        Execute steps with automatic checkpointing.
        
        Args:
            steps: List of async functions to execute
            initial_state: Starting state
            resume: Whether to resume from checkpoint
            
        Returns:
            Final state after all steps
        """
        self.total_steps = len(steps)
        self.state = initial_state
        
        # Try to resume from checkpoint
        if resume:
            checkpoint = await self.checkpoint_manager.load_latest(self.task_id)
            if checkpoint:
                self.current_step = checkpoint.metadata.step_index + 1
                self.state = checkpoint.state
                print(f"Resuming from step {self.current_step}/{self.total_steps}")
        
        # Execute remaining steps
        for i in range(self.current_step, self.total_steps):
            step_fn = steps[i]
            
            try:
                self.state = await step_fn()
                
                # Checkpoint after each step
                await self.checkpoint_manager.save(
                    task_id=self.task_id,
                    step_index=i,
                    total_steps=self.total_steps,
                    state=self.state,
                    context={"step_name": step_fn.__name__}
                )
                
                self.current_step = i + 1
                
            except Exception as e:
                # Save checkpoint before re-raising
                await self.checkpoint_manager.save(
                    task_id=self.task_id,
                    step_index=i,
                    total_steps=self.total_steps,
                    state=self.state,
                    context={"error": str(e)},
                    tags=["failed"]
                )
                raise
        
        return self.state
```

---

## 8. Human-in-the-Loop Patterns

### Confirmation Gates

```python
from dataclasses import dataclass
from typing import Callable, Awaitable, Optional, List, Any
from enum import Enum
import asyncio


class ConfirmationLevel(Enum):
    """Levels of confirmation required."""
    NONE = "none"           # No confirmation needed
    NOTIFY = "notify"       # Notify but proceed
    CONFIRM = "confirm"     # Require explicit approval
    REVIEW = "review"       # Detailed review required


@dataclass
class ConfirmationRequest:
    """Request for human confirmation."""
    action: str
    description: str
    level: ConfirmationLevel
    details: Optional[dict] = None
    timeout: Optional[float] = None  # Seconds to wait


@dataclass
class ConfirmationResponse:
    """Response from human confirmation."""
    approved: bool
    feedback: Optional[str] = None
    modified_action: Optional[dict] = None


class ConfirmationGate:
    """
    Gate that requires human confirmation before proceeding.
    """
    
    def __init__(
        self,
        prompt_fn: Callable[[ConfirmationRequest], Awaitable[ConfirmationResponse]],
        default_timeout: float = 300.0  # 5 minutes
    ):
        self.prompt_fn = prompt_fn
        self.default_timeout = default_timeout
        self.confirmation_log: List[tuple] = []
    
    async def request_confirmation(
        self,
        action: str,
        description: str,
        level: ConfirmationLevel = ConfirmationLevel.CONFIRM,
        details: Optional[dict] = None,
        timeout: Optional[float] = None
    ) -> ConfirmationResponse:
        """
        Request human confirmation for an action.
        
        Args:
            action: Short action name
            description: Detailed description
            level: Confirmation level required
            details: Additional details to show
            timeout: Override default timeout
            
        Returns:
            ConfirmationResponse with approval status
        """
        if level == ConfirmationLevel.NONE:
            return ConfirmationResponse(approved=True)
        
        request = ConfirmationRequest(
            action=action,
            description=description,
            level=level,
            details=details,
            timeout=timeout or self.default_timeout
        )
        
        if level == ConfirmationLevel.NOTIFY:
            # Just notify, don't wait for response
            asyncio.create_task(self.prompt_fn(request))
            return ConfirmationResponse(approved=True)
        
        # Wait for confirmation
        try:
            response = await asyncio.wait_for(
                self.prompt_fn(request),
                timeout=request.timeout
            )
        except asyncio.TimeoutError:
            response = ConfirmationResponse(
                approved=False,
                feedback="Confirmation timed out"
            )
        
        # Log confirmation
        self.confirmation_log.append((request, response))
        
        return response


async def with_confirmation(
    gate: ConfirmationGate,
    action: str,
    description: str,
    fn: Callable[[], Awaitable[Any]],
    level: ConfirmationLevel = ConfirmationLevel.CONFIRM
) -> Result[Any]:
    """
    Execute a function with human confirmation.
    
    Usage:
        result = await with_confirmation(
            gate,
            "delete_files",
            "Delete 10 temporary files from /tmp",
            lambda: delete_temp_files(),
            level=ConfirmationLevel.CONFIRM
        )
    """
    response = await gate.request_confirmation(
        action=action,
        description=description,
        level=level
    )
    
    if not response.approved:
        return err(
            f"User cancelled: {response.feedback or 'No reason provided'}",
            code="USER_CANCELLED"
        )
    
    try:
        result = await fn()
        return ok(result)
    except Exception as e:
        return err(str(e), code="EXECUTION_ERROR")


# CLI-based confirmation
async def cli_confirmation_prompt(request: ConfirmationRequest) -> ConfirmationResponse:
    """
    Terminal-based confirmation prompt.
    """
    print("\n" + "=" * 60)
    print(f"ACTION: {request.action}")
    print(f"DESCRIPTION: {request.description}")
    
    if request.details:
        print("\nDETAILS:")
        for key, value in request.details.items():
            print(f"  {key}: {value}")
    
    print("=" * 60)
    
    if request.level == ConfirmationLevel.NOTIFY:
        print("[NOTIFY] Proceeding automatically...")
        return ConfirmationResponse(approved=True)
    
    prompt = "Approve? [y/n]: " if request.level == ConfirmationLevel.CONFIRM else \
             "Review complete, approve? [y/n]: "
    
    # In real implementation, would use async input
    response = input(prompt).strip().lower()
    
    approved = response in ('y', 'yes')
    feedback = None
    
    if not approved:
        feedback = input("Reason (optional): ").strip() or None
    
    return ConfirmationResponse(approved=approved, feedback=feedback)


# Confirmation policies
@dataclass
class ConfirmationPolicy:
    """Policy for when confirmation is required."""
    patterns: List[str]  # Action patterns requiring confirmation
    level: ConfirmationLevel
    
    def matches(self, action: str) -> bool:
        """Check if action matches policy patterns."""
        import fnmatch
        return any(fnmatch.fnmatch(action, pattern) for pattern in self.patterns)


class PolicyBasedConfirmation:
    """
    Confirmation gate with configurable policies.
    """
    
    def __init__(
        self,
        gate: ConfirmationGate,
        policies: Optional[List[ConfirmationPolicy]] = None
    ):
        self.gate = gate
        self.policies = policies or [
            # Default policies
            ConfirmationPolicy(["delete*", "remove*"], ConfirmationLevel.CONFIRM),
            ConfirmationPolicy(["deploy*", "publish*"], ConfirmationLevel.REVIEW),
            ConfirmationPolicy(["read*", "list*"], ConfirmationLevel.NONE),
        ]
    
    def get_level_for_action(self, action: str) -> ConfirmationLevel:
        """Determine confirmation level for an action."""
        for policy in self.policies:
            if policy.matches(action):
                return policy.level
        return ConfirmationLevel.CONFIRM  # Default to confirm
    
    async def execute(
        self,
        action: str,
        description: str,
        fn: Callable[[], Awaitable[Any]],
        details: Optional[dict] = None,
        override_level: Optional[ConfirmationLevel] = None
    ) -> Result[Any]:
        """Execute with policy-based confirmation."""
        level = override_level or self.get_level_for_action(action)
        
        response = await self.gate.request_confirmation(
            action=action,
            description=description,
            level=level,
            details=details
        )
        
        if not response.approved:
            return err(f"Cancelled: {response.feedback}", code="CANCELLED")
        
        return ok(await fn())
```

### Review Checkpoints

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Awaitable
from datetime import datetime
from enum import Enum


class ReviewStatus(Enum):
    """Status of a review checkpoint."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


@dataclass
class ReviewItem:
    """An item requiring review."""
    id: str
    category: str
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReviewCheckpoint:
    """
    A checkpoint where human review is required.
    """
    id: str
    name: str
    description: str
    items: List[ReviewItem]
    status: ReviewStatus = ReviewStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    reviewed_at: Optional[datetime] = None
    reviewer_notes: Optional[str] = None
    modifications: Dict[str, Any] = field(default_factory=dict)


class ReviewWorkflow:
    """
    Manages review checkpoints in agentic workflows.
    """
    
    def __init__(
        self,
        review_handler: Callable[[ReviewCheckpoint], Awaitable[ReviewCheckpoint]]
    ):
        self.review_handler = review_handler
        self.checkpoints: Dict[str, ReviewCheckpoint] = {}
        self.completed_reviews: List[ReviewCheckpoint] = []
    
    async def create_checkpoint(
        self,
        name: str,
        description: str,
        items: List[ReviewItem]
    ) -> ReviewCheckpoint:
        """Create a review checkpoint."""
        checkpoint = ReviewCheckpoint(
            id=f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=name,
            description=description,
            items=items
        )
        self.checkpoints[checkpoint.id] = checkpoint
        return checkpoint
    
    async def request_review(
        self,
        checkpoint: ReviewCheckpoint
    ) -> ReviewCheckpoint:
        """
        Request human review and wait for completion.
        
        Returns:
            Updated checkpoint with review status
        """
        reviewed = await self.review_handler(checkpoint)
        reviewed.reviewed_at = datetime.now()
        
        self.checkpoints[reviewed.id] = reviewed
        self.completed_reviews.append(reviewed)
        
        return reviewed
    
    async def pause_for_review(
        self,
        name: str,
        description: str,
        items: List[ReviewItem],
        on_approved: Callable[[], Awaitable[Any]],
        on_rejected: Optional[Callable[[str], Awaitable[Any]]] = None,
        on_modified: Optional[Callable[[Dict], Awaitable[Any]]] = None
    ) -> Any:
        """
        Pause execution for review and handle outcome.
        
        Args:
            name: Checkpoint name
            description: What's being reviewed
            items: Items to review
            on_approved: Called if approved
            on_rejected: Called if rejected
            on_modified: Called if modifications made
            
        Returns:
            Result of the appropriate handler
        """
        checkpoint = await self.create_checkpoint(name, description, items)
        reviewed = await self.request_review(checkpoint)
        
        if reviewed.status == ReviewStatus.APPROVED:
            return await on_approved()
        
        elif reviewed.status == ReviewStatus.REJECTED:
            if on_rejected:
                return await on_rejected(reviewed.reviewer_notes or "No reason provided")
            raise ReviewRejectedError(reviewed.reviewer_notes)
        
        elif reviewed.status == ReviewStatus.MODIFIED:
            if on_modified:
                return await on_modified(reviewed.modifications)
            return await on_approved()  # Proceed with modifications
        
        raise ValueError(f"Unknown review status: {reviewed.status}")


class ReviewRejectedError(Exception):
    """Raised when a review is rejected."""
    pass


# Interactive review handler
async def interactive_review_handler(checkpoint: ReviewCheckpoint) -> ReviewCheckpoint:
    """
    Terminal-based interactive review handler.
    """
    print("\n" + "=" * 70)
    print(f"REVIEW CHECKPOINT: {checkpoint.name}")
    print(f"Description: {checkpoint.description}")
    print("=" * 70)
    
    print(f"\nItems to review ({len(checkpoint.items)}):\n")
    
    for i, item in enumerate(checkpoint.items, 1):
        print(f"  [{i}] {item.category}")
        print(f"      {item.content}")
        if item.metadata:
            for k, v in item.metadata.items():
                print(f"      {k}: {v}")
        print()
    
    print("-" * 70)
    print("Options: [a]pprove, [r]eject, [m]odify, [v]iew details")
    
    while True:
        choice = input("Your choice: ").strip().lower()
        
        if choice == 'a':
            checkpoint.status = ReviewStatus.APPROVED
            break
        
        elif choice == 'r':
            notes = input("Rejection reason: ").strip()
            checkpoint.status = ReviewStatus.REJECTED
            checkpoint.reviewer_notes = notes
            break
        
        elif choice == 'm':
            print("Enter modifications (JSON format, empty line to finish):")
            mod_lines = []
            while True:
                line = input()
                if not line:
                    break
                mod_lines.append(line)
            
            import json
            try:
                checkpoint.modifications = json.loads("\n".join(mod_lines))
                checkpoint.status = ReviewStatus.MODIFIED
                break
            except json.JSONDecodeError:
                print("Invalid JSON, try again")
        
        elif choice == 'v':
            # Show full details
            import json
            print(json.dumps(
                [{"id": i.id, "content": i.content, "metadata": i.metadata}
                 for i in checkpoint.items],
                indent=2
            ))
        
        else:
            print("Unknown option")
    
    return checkpoint


# Staged review workflow
class StagedReviewWorkflow:
    """
    Multi-stage review workflow for complex operations.
    """
    
    def __init__(self, review_workflow: ReviewWorkflow):
        self.workflow = review_workflow
        self.stages: List[Dict[str, Any]] = []
        self.current_stage = 0
    
    def add_stage(
        self,
        name: str,
        executor: Callable[[], Awaitable[List[ReviewItem]]],
        requires_review: bool = True
    ) -> "StagedReviewWorkflow":
        """Add a stage to the workflow."""
        self.stages.append({
            "name": name,
            "executor": executor,
            "requires_review": requires_review
        })
        return self
    
    async def run(self) -> List[Any]:
        """Execute all stages with reviews."""
        results = []
        
        for i, stage in enumerate(self.stages):
            self.current_stage = i
            
            # Execute stage
            items = await stage["executor"]()
            
            if stage["requires_review"] and items:
                # Request review
                checkpoint = await self.workflow.create_checkpoint(
                    name=f"Stage {i + 1}: {stage['name']}",
                    description=f"Review results from {stage['name']}",
                    items=items
                )
                
                reviewed = await self.workflow.request_review(checkpoint)
                
                if reviewed.status == ReviewStatus.REJECTED:
                    raise ReviewRejectedError(
                        f"Stage {stage['name']} rejected: {reviewed.reviewer_notes}"
                    )
                
                results.append({
                    "stage": stage["name"],
                    "items": items,
                    "review_status": reviewed.status.value,
                    "modifications": reviewed.modifications
                })
            else:
                results.append({
                    "stage": stage["name"],
                    "items": items,
                    "review_status": "skipped"
                })
        
        return results
```

---

## 9. Best Practices Summary

### Design Principles

1. **Explicit State** - Always know what state your agent is in
2. **Fail Gracefully** - Use Result types and circuit breakers
3. **Checkpoint Often** - Save progress for resumption
4. **Human Oversight** - Strategic confirmation gates
5. **Token Awareness** - Manage context window explicitly

### Anti-Patterns to Avoid

| Anti-Pattern | Problem | Solution |
|--------------|---------|----------|
| Unbounded retries | Resource exhaustion | Use max attempts with backoff |
| Silent failures | Lost errors | Use Result types |
| Monolithic agents | Hard to debug | Use pipelines and stages |
| No checkpoints | Lost progress | Checkpoint after each step |
| Infinite context | Token overflow | Prune with strategies |

### Testing Patterns

```python
import pytest
from unittest.mock import AsyncMock, patch


class TestAgent:
    """Test patterns for agent code."""
    
    @pytest.fixture
    def agent(self):
        return Agent(checkpoint_dir="./test_checkpoints")
    
    @pytest.mark.asyncio
    async def test_state_transitions(self, agent):
        """Test agent moves through expected states."""
        states_visited = []
        
        async def track_state(old_state, new_state):
            states_visited.append(new_state)
        
        agent.on_state_change(track_state)
        
        await agent.run("test task")
        
        assert AgentState.PLANNING in states_visited
        assert AgentState.EXECUTING in states_visited
        assert AgentState.COMPLETED in states_visited
    
    @pytest.mark.asyncio
    async def test_checkpoint_resume(self, agent):
        """Test agent can resume from checkpoint."""
        # Create checkpoint at step 2
        checkpoint = Checkpoint(
            state=AgentState.EXECUTING,
            context=ExecutionContext(
                task="resume test",
                plan=[{"step": 1}, {"step": 2}, {"step": 3}],
                current_step=2
            ),
            timestamp=datetime.now()
        )
        
        result = await agent.run("resume test", resume_from=checkpoint)
        
        # Should only execute remaining steps
        assert agent.context.current_step == 3
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens(self):
        """Test circuit breaker opens after failures."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=3))
        
        async def failing_fn():
            raise Exception("Simulated failure")
        
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call(failing_fn)
        
        assert breaker.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff(self):
        """Test exponential backoff behavior."""
        attempt_times = []
        
        async def tracked_fn():
            attempt_times.append(asyncio.get_event_loop().time())
            raise Exception("Retry me")
        
        with pytest.raises(Exception):
            await with_retry(tracked_fn, RetryConfig(max_attempts=3, base_delay=0.1))
        
        # Verify increasing delays
        assert len(attempt_times) == 3
        delay1 = attempt_times[1] - attempt_times[0]
        delay2 = attempt_times[2] - attempt_times[1]
        assert delay2 > delay1  # Exponential increase
```

---

## Appendix: Quick Reference

### State Machine Template

```python
agent = Agent(checkpoint_dir="./checkpoints")
result = await agent.run("your task here")
```

### Result Type Template

```python
result = await some_operation()
if isinstance(result, Err):
    handle_error(result.error)
else:
    use_value(result.value)
```

### Retry Template

```python
result = await with_retry(
    lambda: risky_operation(),
    RetryConfig(max_attempts=5, base_delay=1.0)
)
```

### Circuit Breaker Template

```python
@with_circuit_breaker("external-api")
async def call_api():
    return await httpx.get(url)
```

### Confirmation Template

```python
result = await with_confirmation(
    gate,
    "action_name",
    "Description of what will happen",
    lambda: perform_action()
)
```

---

*Document Version: 1.0*
*Last Updated: 2026-01-24*
*Patterns tested with Claude Code CLI v1.0.21+*
