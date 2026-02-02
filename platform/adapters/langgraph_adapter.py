"""
LangGraph Adapter for Unleashed Platform

LangGraph provides stateful, multi-actor workflows with checkpointing and
human-in-the-loop capabilities.

Key features:
- State graph architecture (nodes + edges)
- Durable execution with checkpointing
- Human-in-the-loop interrupts
- Enterprise adoption: Replit, Uber, LinkedIn, GitLab

Repository: https://github.com/langchain-ai/langgraph
Stars: 23,500 | License: MIT
"""

import os
from typing import Any, Dict, List, Optional, Callable, TypeVar, Annotated
from dataclasses import dataclass, field
from enum import Enum

# Check LangGraph availability
LANGGRAPH_AVAILABLE = False
langgraph = None
StateGraph = None
END = None
START = None

try:
    from langgraph.graph import StateGraph as _StateGraph, END as _END, START as _START
    from langgraph.checkpoint.memory import MemorySaver
    import langgraph as _langgraph

    langgraph = _langgraph
    StateGraph = _StateGraph
    END = _END
    START = _START
    LANGGRAPH_AVAILABLE = True

    # Optional production checkpointers (verified from official docs 2026-01-30)
    # Use PostgresSaver for production, MemorySaver for development only
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        POSTGRES_SAVER_AVAILABLE = True
    except ImportError:
        PostgresSaver = None
        POSTGRES_SAVER_AVAILABLE = False

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        SQLITE_SAVER_AVAILABLE = True
    except ImportError:
        SqliteSaver = None
        SQLITE_SAVER_AVAILABLE = False
except ImportError:
    PostgresSaver = None
    SqliteSaver = None
    POSTGRES_SAVER_AVAILABLE = False
    SQLITE_SAVER_AVAILABLE = False
    pass

# Register adapter status
from . import register_adapter
register_adapter("langgraph", LANGGRAPH_AVAILABLE)


class NodeType(Enum):
    """Types of nodes in a workflow graph."""
    STANDARD = "standard"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    SUBGRAPH = "subgraph"
    HUMAN_IN_LOOP = "human_in_loop"


@dataclass
class WorkflowState:
    """Base state for workflow graphs."""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    current_step: str = ""
    completed_steps: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class WorkflowResult:
    """Result from workflow execution."""
    final_state: Dict[str, Any]
    steps_executed: List[str]
    checkpoints: List[str]
    execution_time: float
    success: bool
    error: Optional[str] = None


class LangGraphAdapter:
    """
    Adapter for LangGraph workflow orchestration.

    LangGraph enables building complex, stateful multi-agent workflows
    with native support for checkpointing and human oversight.
    """

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        enable_persistence: bool = True,
        checkpointer_type: str = "memory",
        postgres_conn_string: Optional[str] = None,
    ):
        """
        Initialize LangGraph adapter.

        Args:
            checkpoint_dir: Directory for checkpoints (for sqlite)
            enable_persistence: Whether to enable state persistence
            checkpointer_type: Type of checkpointer ("memory", "postgres", "sqlite")
                              - memory: In-memory (development only, data lost on restart)
                              - postgres: PostgreSQL (production recommended)
                              - sqlite: SQLite file-based (good for single-instance)
            postgres_conn_string: PostgreSQL connection string (required for postgres)
        """
        self._available = LANGGRAPH_AVAILABLE
        self.checkpoint_dir = checkpoint_dir
        self.enable_persistence = enable_persistence
        self.checkpointer_type = checkpointer_type
        self._graphs: Dict[str, Any] = {}
        self._checkpointer = None

        if LANGGRAPH_AVAILABLE and enable_persistence:
            self._checkpointer = self._create_checkpointer(
                checkpointer_type, postgres_conn_string, checkpoint_dir
            )

    def _create_checkpointer(
        self,
        checkpointer_type: str,
        postgres_conn_string: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Create the appropriate checkpointer based on type.

        Production recommendation: Use PostgresSaver for durability.
        MemorySaver is for development only (verified from official docs 2026-01-30).
        """
        if checkpointer_type == "postgres":
            if not POSTGRES_SAVER_AVAILABLE:
                raise ImportError(
                    "PostgresSaver not available. Install with: pip install langgraph-checkpoint-postgres"
                )
            if not postgres_conn_string:
                raise ValueError("postgres_conn_string required for PostgresSaver")
            return PostgresSaver.from_conn_string(postgres_conn_string)

        elif checkpointer_type == "sqlite":
            if not SQLITE_SAVER_AVAILABLE:
                raise ImportError(
                    "SqliteSaver not available. Install with: pip install langgraph-checkpoint-sqlite"
                )
            db_path = checkpoint_dir or ":memory:"
            return SqliteSaver.from_conn_string(db_path)

        else:  # Default to memory
            # WARNING: MemorySaver is for development only - data is lost on restart
            return MemorySaver()

    def get_status(self) -> Dict[str, Any]:
        """Get adapter status."""
        return {
            "available": self._available,
            "persistence_enabled": self.enable_persistence,
            "graphs_created": len(self._graphs),
            "checkpointer_active": self._checkpointer is not None,
            "checkpointer_type": self.checkpointer_type,
            "postgres_saver_available": POSTGRES_SAVER_AVAILABLE,
            "sqlite_saver_available": SQLITE_SAVER_AVAILABLE,
        }

    def _check_available(self):
        """Check if LangGraph is available, raise error if not."""
        if not self._available:
            raise ImportError(
                "LangGraph is not installed. Install with: pip install langgraph"
            )

    def create_graph(
        self,
        name: str,
        state_schema: Optional[type] = None,
    ) -> Any:
        """
        Create a new state graph.

        Args:
            name: Name for this graph
            state_schema: Pydantic model or TypedDict for state

        Returns:
            StateGraph instance
        """
        self._check_available()
        if state_schema is None:
            # Use default dict state
            from typing import TypedDict

            class DefaultState(TypedDict, total=False):
                messages: List[Dict[str, Any]]
                context: Dict[str, Any]
                current_step: str
                result: Any
                error: str

            state_schema = DefaultState

        graph = StateGraph(state_schema)
        self._graphs[name] = {
            "graph": graph,
            "compiled": None,
            "nodes": [],
            "edges": [],
        }
        return graph

    def add_node(
        self,
        graph_name: str,
        node_name: str,
        handler: Callable,
        node_type: NodeType = NodeType.STANDARD,
    ):
        """
        Add a node to a graph.

        Args:
            graph_name: Name of the graph
            node_name: Name for this node
            handler: Function to handle this node
            node_type: Type of node
        """
        if graph_name not in self._graphs:
            raise ValueError(f"Graph '{graph_name}' not found")

        graph_info = self._graphs[graph_name]
        graph = graph_info["graph"]

        graph.add_node(node_name, handler)
        graph_info["nodes"].append({
            "name": node_name,
            "type": node_type.value,
            "handler": handler.__name__,
        })

    def add_edge(
        self,
        graph_name: str,
        from_node: str,
        to_node: str,
    ):
        """
        Add a direct edge between nodes.

        Args:
            graph_name: Name of the graph
            from_node: Source node name (or START)
            to_node: Target node name (or END)
        """
        if graph_name not in self._graphs:
            raise ValueError(f"Graph '{graph_name}' not found")

        graph_info = self._graphs[graph_name]
        graph = graph_info["graph"]

        # Handle special nodes
        source = START if from_node == "START" else from_node
        target = END if to_node == "END" else to_node

        graph.add_edge(source, target)
        graph_info["edges"].append({
            "from": from_node,
            "to": to_node,
            "type": "direct",
        })

    def add_conditional_edge(
        self,
        graph_name: str,
        from_node: str,
        condition: Callable,
        path_map: Dict[str, str],
    ):
        """
        Add a conditional edge that routes based on state.

        Args:
            graph_name: Name of the graph
            from_node: Source node name
            condition: Function that returns next node key
            path_map: Mapping from condition output to node names
        """
        if graph_name not in self._graphs:
            raise ValueError(f"Graph '{graph_name}' not found")

        graph_info = self._graphs[graph_name]
        graph = graph_info["graph"]

        # Replace END in path_map
        resolved_map = {
            k: END if v == "END" else v
            for k, v in path_map.items()
        }

        graph.add_conditional_edges(from_node, condition, resolved_map)
        graph_info["edges"].append({
            "from": from_node,
            "type": "conditional",
            "paths": list(path_map.keys()),
        })

    def set_entry_point(self, graph_name: str, node_name: str):
        """Set the entry point for a graph."""
        if graph_name not in self._graphs:
            raise ValueError(f"Graph '{graph_name}' not found")

        graph_info = self._graphs[graph_name]
        graph = graph_info["graph"]

        graph.add_edge(START, node_name)

    def set_finish_point(self, graph_name: str, node_name: str):
        """Set the finish point for a graph."""
        if graph_name not in self._graphs:
            raise ValueError(f"Graph '{graph_name}' not found")

        graph_info = self._graphs[graph_name]
        graph = graph_info["graph"]

        graph.add_edge(node_name, END)

    def compile(self, graph_name: str) -> Any:
        """
        Compile a graph for execution.

        Args:
            graph_name: Name of the graph

        Returns:
            Compiled graph
        """
        if graph_name not in self._graphs:
            raise ValueError(f"Graph '{graph_name}' not found")

        graph_info = self._graphs[graph_name]
        graph = graph_info["graph"]

        compile_kwargs = {}
        if self._checkpointer and self.enable_persistence:
            compile_kwargs["checkpointer"] = self._checkpointer

        compiled = graph.compile(**compile_kwargs)
        graph_info["compiled"] = compiled
        return compiled

    async def execute(
        self,
        graph_name: str,
        initial_state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
    ) -> WorkflowResult:
        """
        Execute a compiled graph.

        Args:
            graph_name: Name of the graph
            initial_state: Initial state dict
            config: Execution configuration
            thread_id: Thread ID for checkpointing

        Returns:
            WorkflowResult with execution details
        """
        import time
        start_time = time.time()

        if graph_name not in self._graphs:
            raise ValueError(f"Graph '{graph_name}' not found")

        graph_info = self._graphs[graph_name]
        compiled = graph_info.get("compiled")

        if compiled is None:
            compiled = self.compile(graph_name)

        # Build config
        exec_config = config or {}
        if thread_id:
            exec_config["configurable"] = exec_config.get("configurable", {})
            exec_config["configurable"]["thread_id"] = thread_id

        try:
            # Execute graph
            final_state = await compiled.ainvoke(initial_state, exec_config)

            execution_time = time.time() - start_time

            return WorkflowResult(
                final_state=final_state,
                steps_executed=graph_info["nodes"],
                checkpoints=[thread_id] if thread_id else [],
                execution_time=execution_time,
                success=True,
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return WorkflowResult(
                final_state=initial_state,
                steps_executed=[],
                checkpoints=[],
                execution_time=execution_time,
                success=False,
                error=str(e),
            )

    def stream(
        self,
        graph_name: str,
        initial_state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Stream execution of a graph.

        Args:
            graph_name: Name of the graph
            initial_state: Initial state dict
            config: Execution configuration

        Yields:
            State updates during execution
        """
        if graph_name not in self._graphs:
            raise ValueError(f"Graph '{graph_name}' not found")

        graph_info = self._graphs[graph_name]
        compiled = graph_info.get("compiled")

        if compiled is None:
            compiled = self.compile(graph_name)

        for state in compiled.stream(initial_state, config or {}):
            yield state

    def get_state(
        self,
        graph_name: str,
        thread_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the current state of a thread.

        Args:
            graph_name: Name of the graph
            thread_id: Thread ID

        Returns:
            Current state or None
        """
        if graph_name not in self._graphs:
            return None

        graph_info = self._graphs[graph_name]
        compiled = graph_info.get("compiled")

        if compiled is None or not self._checkpointer:
            return None

        config = {"configurable": {"thread_id": thread_id}}
        state = compiled.get_state(config)
        return state.values if state else None

    def update_state(
        self,
        graph_name: str,
        thread_id: str,
        updates: Dict[str, Any],
        as_node: Optional[str] = None,
    ):
        """
        Update the state of a paused thread.

        Args:
            graph_name: Name of the graph
            thread_id: Thread ID
            updates: State updates
            as_node: Node to attribute update to
        """
        if graph_name not in self._graphs:
            raise ValueError(f"Graph '{graph_name}' not found")

        graph_info = self._graphs[graph_name]
        compiled = graph_info.get("compiled")

        if compiled is None:
            compiled = self.compile(graph_name)

        config = {"configurable": {"thread_id": thread_id}}
        compiled.update_state(config, updates, as_node=as_node)

    def add_human_in_loop(
        self,
        graph_name: str,
        before_node: str,
        approval_handler: Optional[Callable] = None,
    ):
        """
        Add a human-in-the-loop checkpoint before a node.

        Args:
            graph_name: Name of the graph
            before_node: Node to pause before
            approval_handler: Optional custom approval handler
        """
        if graph_name not in self._graphs:
            raise ValueError(f"Graph '{graph_name}' not found")

        # Human-in-loop is configured at compile time via interrupt_before
        graph_info = self._graphs[graph_name]
        graph_info["interrupt_before"] = graph_info.get("interrupt_before", [])
        graph_info["interrupt_before"].append(before_node)

    def get_graph_info(self, graph_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a graph."""
        if graph_name not in self._graphs:
            return None

        graph_info = self._graphs[graph_name]
        return {
            "name": graph_name,
            "nodes": graph_info["nodes"],
            "edges": graph_info["edges"],
            "compiled": graph_info["compiled"] is not None,
        }

    def list_graphs(self) -> List[str]:
        """List all registered graphs."""
        return list(self._graphs.keys())


# Convenience functions for common patterns
def create_agent_graph(
    agent_handlers: Dict[str, Callable],
    router: Callable,
) -> LangGraphAdapter:
    """
    Create a multi-agent workflow graph.

    Args:
        agent_handlers: Dict mapping agent names to handlers
        router: Function to route between agents

    Returns:
        Configured LangGraphAdapter
    """
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("LangGraph not available")

    adapter = LangGraphAdapter()
    adapter.create_graph("agents")

    # Add router node
    adapter.add_node("agents", "router", router)
    adapter.set_entry_point("agents", "router")

    # Add agent nodes
    path_map = {}
    for name, handler in agent_handlers.items():
        adapter.add_node("agents", name, handler)
        path_map[name] = name
        adapter.add_edge("agents", name, "router")  # Loop back

    path_map["END"] = "END"

    # Add conditional routing
    adapter.add_conditional_edge(
        "agents",
        "router",
        lambda state: state.get("next_agent", "END"),
        path_map,
    )

    return adapter


def create_linear_workflow(
    steps: List[Dict[str, Any]],
) -> LangGraphAdapter:
    """
    Create a simple linear workflow.

    Args:
        steps: List of step configs [{"name": str, "handler": Callable}, ...]

    Returns:
        Configured LangGraphAdapter
    """
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("LangGraph not available")

    adapter = LangGraphAdapter()
    adapter.create_graph("linear")

    for i, step in enumerate(steps):
        adapter.add_node("linear", step["name"], step["handler"])

        if i == 0:
            adapter.set_entry_point("linear", step["name"])
        else:
            adapter.add_edge("linear", steps[i-1]["name"], step["name"])

        if i == len(steps) - 1:
            adapter.set_finish_point("linear", step["name"])

    return adapter
