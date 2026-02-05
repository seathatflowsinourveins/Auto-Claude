"""Ralph Loop Modular Package.

This package provides a LangGraph-style modular implementation of the
Ralph Loop, enabling iterative improvement with automatic checkpointing.

Structure:
    ralph/
    ├── __init__.py     # This file - package exports
    ├── graph.py        # LangGraph-style graph construction
    ├── nodes/          # Individual phase nodes
    │   ├── health_check.py
    │   ├── validation.py
    │   ├── consolidation.py
    │   └── session_update.py
    ├── state/          # State definitions
    │   └── ralph_state.py
    └── utils/          # Utility functions
        ├── metrics.py
        └── reporting.py

Usage:
    from ralph import create_ralph_graph, run_iteration, create_default_config

    # Create configuration
    config = create_default_config()

    # Create the graph
    graph = create_ralph_graph()

    # Run iterations
    for i in range(10):
        report = run_iteration(graph, iteration_number=i+1, config=config)
        if report.overall_status == IterationStatus.FAILED:
            break
"""

from .graph import (
    CompiledGraph,
    GraphConfig,
    RalphGraph,
    create_default_config,
    create_ralph_graph,
    run_iteration,
)
from .state import (
    IterationPhase,
    IterationReport,
    IterationStatus,
    PhaseResult,
    RalphState,
)
from .nodes import (
    health_check_node,
    validation_node,
    consolidation_node,
    session_update_node,
)
from .utils import (
    calculate_phase_metrics,
    aggregate_iteration_metrics,
    save_report,
    print_iteration_header,
    print_iteration_result,
    print_status,
    format_report,
)

__version__ = "1.0.0"

__all__ = [
    # Graph
    "CompiledGraph",
    "GraphConfig",
    "RalphGraph",
    "create_default_config",
    "create_ralph_graph",
    "run_iteration",
    # State
    "IterationPhase",
    "IterationReport",
    "IterationStatus",
    "PhaseResult",
    "RalphState",
    # Nodes
    "health_check_node",
    "validation_node",
    "consolidation_node",
    "session_update_node",
    # Utils
    "calculate_phase_metrics",
    "aggregate_iteration_metrics",
    "save_report",
    "print_iteration_header",
    "print_iteration_result",
    "print_status",
    "format_report",
]
