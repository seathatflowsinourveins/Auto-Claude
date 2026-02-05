"""Ralph Loop Graph Construction.

This module provides the LangGraph-style graph construction for the
Ralph Loop. It composes the node modules into a sequential graph
that executes the 4 phases of the Ralph Loop.

Usage:
    from ralph.graph import create_ralph_graph, run_iteration

    # Create the graph
    graph = create_ralph_graph()

    # Run an iteration
    result = run_iteration(graph, iteration_number=1, config=config)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

from .nodes import (
    health_check_node,
    validation_node,
    consolidation_node,
    session_update_node,
)
from .state import (
    IterationReport,
    IterationStatus,
    PhaseResult,
    RalphState,
)
from .state.ralph_state import (
    create_initial_state,
    create_memory_block,
    finalize_state,
    state_to_report,
)
from .utils import (
    print_iteration_header,
    print_iteration_result,
    save_report,
)

# Type for node functions
NodeFunc = Callable[[RalphState], RalphState]


@dataclass
class GraphConfig:
    """Configuration for the Ralph Loop graph."""

    script_dir: Path
    data_dir: Path
    memory_dir: Path
    reports_dir: Path
    verbose: bool = True


@dataclass
class GraphNode:
    """A node in the Ralph Loop graph."""

    name: str
    func: NodeFunc
    description: str


class RalphGraph:
    """LangGraph-style graph for Ralph Loop execution.

    This class manages the sequential execution of Ralph Loop phases
    as a graph of nodes.
    """

    def __init__(self) -> None:
        """Initialize the graph with empty node list."""
        self._nodes: List[GraphNode] = []
        self._edges: List[tuple[str, str]] = []

    def add_node(self, name: str, func: NodeFunc, description: str = "") -> "RalphGraph":
        """Add a node to the graph.

        Args:
            name: Unique name for the node.
            func: Function that processes RalphState.
            description: Human-readable description.

        Returns:
            Self for method chaining.
        """
        self._nodes.append(GraphNode(name=name, func=func, description=description))
        return self

    def add_edge(self, from_node: str, to_node: str) -> "RalphGraph":
        """Add an edge between nodes.

        Args:
            from_node: Source node name.
            to_node: Target node name.

        Returns:
            Self for method chaining.
        """
        self._edges.append((from_node, to_node))
        return self

    def compile(self) -> "CompiledGraph":
        """Compile the graph for execution.

        Returns:
            A CompiledGraph ready for execution.
        """
        return CompiledGraph(self._nodes, self._edges)


class CompiledGraph:
    """A compiled graph ready for execution."""

    def __init__(self, nodes: List[GraphNode], edges: List[tuple[str, str]]) -> None:
        """Initialize with nodes and edges.

        Args:
            nodes: List of graph nodes.
            edges: List of edges between nodes.
        """
        self._nodes = nodes
        self._edges = edges
        self._node_map = {node.name: node for node in nodes}

    def invoke(self, state: RalphState, verbose: bool = True) -> RalphState:
        """Execute the graph on the given state.

        Args:
            state: Initial state.
            verbose: Whether to print progress.

        Returns:
            Final state after all nodes execute.
        """
        current_state = state
        total_nodes = len(self._nodes)

        for i, node in enumerate(self._nodes, 1):
            if verbose:
                phase_name = node.name.replace("_", " ").title()
                print(f"[{i}/{total_nodes}] Running {phase_name}...", end=" ", flush=True)

            current_state = node.func(current_state)

            if verbose:
                # Get the result from state
                result_key = f"{node.name}_result"
                if node.name == "health_check":
                    result_key = "health_result"
                elif node.name == "session_update":
                    result_key = "session_result"
                else:
                    result_key = f"{node.name}_result"

                result = current_state.get(result_key)
                if result:
                    status = result.status.value.upper()
                    print(f"[{status}] {result.message}")
                else:
                    print("[DONE]")

        return current_state


def create_ralph_graph() -> CompiledGraph:
    """Create and compile the Ralph Loop graph.

    The graph executes 4 phases in sequence:
    1. Health Check - Verify system health
    2. Validation - Run validation checks
    3. Consolidation - Consolidate memory blocks
    4. Session Update - Update session state

    Returns:
        A compiled graph ready for execution.
    """
    graph = RalphGraph()

    # Add nodes
    graph.add_node(
        "health_check",
        health_check_node,
        "Run ecosystem health check",
    )
    graph.add_node(
        "validation",
        validation_node,
        "Run validation pipeline",
    )
    graph.add_node(
        "consolidation",
        consolidation_node,
        "Run memory consolidation",
    )
    graph.add_node(
        "session_update",
        session_update_node,
        "Update session state",
    )

    # Add edges (sequential flow)
    graph.add_edge("health_check", "validation")
    graph.add_edge("validation", "consolidation")
    graph.add_edge("consolidation", "session_update")

    return graph.compile()


def run_iteration(
    graph: CompiledGraph,
    iteration_number: int,
    config: GraphConfig,
) -> IterationReport:
    """Run a single Ralph Loop iteration using the graph.

    Args:
        graph: The compiled graph to execute.
        iteration_number: Current iteration number.
        config: Graph configuration.

    Returns:
        IterationReport with results.
    """
    start_time = time.perf_counter()

    if config.verbose:
        print_iteration_header(iteration_number)

    # Create initial state
    state = create_initial_state(
        iteration_number=iteration_number,
        script_dir=config.script_dir,
        data_dir=config.data_dir,
        memory_dir=config.memory_dir,
        reports_dir=config.reports_dir,
    )

    # Execute the graph
    final_state = graph.invoke(state, verbose=config.verbose)

    # Finalize state
    final_state = finalize_state(final_state)

    # Calculate total duration
    total_duration_ms = (time.perf_counter() - start_time) * 1000

    # Create report
    report = state_to_report(final_state, total_duration_ms)

    if config.verbose:
        print_iteration_result(
            overall_status=report.overall_status,
            summary=report.summary,
            total_duration_ms=total_duration_ms,
            recommendations=report.recommendations,
        )

    # Save report
    save_report(report, config.reports_dir)

    # Store memory block
    create_memory_block(report, config.memory_dir)

    return report


def create_default_config() -> GraphConfig:
    """Create a default configuration based on script location.

    Returns:
        GraphConfig with default paths.
    """
    script_dir = Path(__file__).parent.parent
    v10_dir = script_dir.parent
    data_dir = v10_dir / "data"
    memory_dir = data_dir / "memory"
    reports_dir = data_dir / "reports"

    # Ensure directories exist
    memory_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    return GraphConfig(
        script_dir=script_dir,
        data_dir=data_dir,
        memory_dir=memory_dir,
        reports_dir=reports_dir,
        verbose=True,
    )
