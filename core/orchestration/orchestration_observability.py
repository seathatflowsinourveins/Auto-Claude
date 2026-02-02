#!/usr/bin/env python3
"""
Orchestration Observability - Opik Integration for Tracing Agent Execution.
Part of V33.10 Architecture - RALPH ITERATION 2.

This module provides observability for the orchestration layer by:
1. Wrapping CoreOrchestrator methods with Opik tracing
2. Capturing metrics for agent coordination and execution
3. Integrating with OpikEvaluator for quality assessment
4. Providing dashboard-ready telemetry data

Usage:
    from core.orchestration.orchestration_observability import (
        create_observable_orchestrator,
        ObservableOrchestrator,
    )

    # Create a traced orchestrator
    orchestrator = await create_observable_orchestrator()

    # All operations are now traced
    result = await orchestrator.coordinate_agents(
        task="validate hooks",
        agents=["validator"],
    )
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Sequence

import structlog

# =============================================================================
# Import Orchestrator (local import to avoid circular)
# =============================================================================

try:
    from core.orchestrator import (
        CoreOrchestrator as _CoreOrchestrator,
        AgentConfig as _AgentConfig,
        AgentRole as _AgentRole,
        TaskResult as _TaskResult,
        TaskStatus as _TaskStatus,
        AGENT_SDK_AVAILABLE,
    )
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False
    _CoreOrchestrator = None  # type: ignore[assignment,misc]
    _AgentConfig = None  # type: ignore[assignment,misc]
    _AgentRole = None  # type: ignore[assignment,misc]
    _TaskResult = None  # type: ignore[assignment,misc]
    _TaskStatus = None  # type: ignore[assignment,misc]
    AGENT_SDK_AVAILABLE = False

# =============================================================================
# Import Opik (observability SDK)
# =============================================================================

try:
    import opik as _opik_module
    from opik import track as _opik_track, opik_context as _opik_context
    OPIK_AVAILABLE = True
except ImportError:
    OPIK_AVAILABLE = False
    _opik_module = None  # type: ignore[assignment]
    _opik_track = None  # type: ignore[assignment]
    _opik_context = None  # type: ignore[assignment]

# =============================================================================
# Import OpikEvaluator for quality metrics
# =============================================================================

try:
    from core.observability.opik_evaluator import (
        OpikEvaluator as _OpikEvaluator,
        MetricType as _MetricType,
        EvaluationResult as _EvaluationResult,
    )
    EVALUATOR_AVAILABLE = True
except ImportError:
    EVALUATOR_AVAILABLE = False
    _OpikEvaluator = None  # type: ignore[assignment,misc]
    _MetricType = None  # type: ignore[assignment,misc]
    _EvaluationResult = None  # type: ignore[assignment,misc]


log = structlog.get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ObservabilityConfig:
    """Configuration for orchestration observability."""

    project_name: str = "unleash-orchestration"
    trace_agent_execution: bool = True
    trace_coordination: bool = True
    trace_ralph_iteration: bool = True
    evaluate_outputs: bool = True
    capture_metadata: bool = True
    tags: List[str] = field(default_factory=lambda: ["orchestration", "v33.10"])


class TraceLevel(str, Enum):
    """Tracing detail levels."""

    MINIMAL = "minimal"      # Only top-level operations
    STANDARD = "standard"    # Methods + basic metadata
    DETAILED = "detailed"    # All methods + full metadata
    DEBUG = "debug"          # Everything including internal calls


# =============================================================================
# Tracing Decorator Factory
# =============================================================================

F = TypeVar("F", bound=Callable[..., Any])


def traced(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    capture_input: bool = True,
    capture_output: bool = True,
) -> Callable[[F], F]:
    """
    Decorator to add Opik tracing to a function.

    Falls back to no-op if Opik is unavailable.

    Args:
        name: Custom span name (defaults to function name)
        tags: Tags to add to the span
        capture_input: Whether to capture function inputs
        capture_output: Whether to capture function outputs

    Returns:
        Decorated function with tracing
    """
    def decorator(func: F) -> F:
        if not OPIK_AVAILABLE:
            return func

        span_name = name or func.__name__
        span_tags = tags or []

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()

            # Prepare input metadata
            input_data = {}
            if capture_input:
                # Don't capture 'self' for methods
                if args and hasattr(args[0], "__class__"):
                    input_data["args"] = str(args[1:])[:500]  # Limit size
                else:
                    input_data["args"] = str(args)[:500]
                input_data["kwargs"] = str({
                    k: v for k, v in kwargs.items()
                    if not k.startswith("_")
                })[:500]

            try:
                # Execute with tracing
                tracked_func = _opik_track(
                    name=span_name,
                    tags=span_tags,
                )(func)

                result = await tracked_func(*args, **kwargs)

                duration_ms = (time.time() - start_time) * 1000

                # Log success
                log.debug(
                    "traced_operation_completed",
                    operation=span_name,
                    duration_ms=round(duration_ms, 2),
                    success=True,
                )

                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                log.error(
                    "traced_operation_failed",
                    operation=span_name,
                    duration_ms=round(duration_ms, 2),
                    error=str(e),
                )
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if OPIK_AVAILABLE:
                tracked_func = _opik_track(
                    name=span_name,
                    tags=span_tags,
                )(func)
                return tracked_func(*args, **kwargs)
            return func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


# =============================================================================
# Observable Orchestrator
# =============================================================================

class ObservableOrchestrator:
    """
    CoreOrchestrator wrapper with Opik observability integration.

    Provides automatic tracing for:
    - Agent coordination operations
    - Individual agent executions
    - Ralph loop iterations
    - Status checks and health monitoring

    Example:
        orchestrator = await create_observable_orchestrator()

        # All operations are traced to Opik dashboard
        result = await orchestrator.coordinate_agents(
            task="validate configuration",
            agents=["validator"],
        )
    """

    def __init__(
        self,
        orchestrator: Optional["CoreOrchestrator"] = None,
        config: Optional[ObservabilityConfig] = None,
    ):
        """
        Initialize observable orchestrator.

        Args:
            orchestrator: Existing CoreOrchestrator instance (creates new if None)
            config: Observability configuration
        """
        self.config = config or ObservabilityConfig()
        self._orchestrator: Optional["CoreOrchestrator"] = orchestrator
        self._evaluator: Optional["OpikEvaluator"] = None
        self._initialized = False

        # Statistics
        self._stats = {
            "total_traces": 0,
            "successful_traces": 0,
            "failed_traces": 0,
            "total_agent_calls": 0,
            "evaluation_count": 0,
        }

    async def initialize(self) -> bool:
        """
        Initialize the observable orchestrator.

        Creates CoreOrchestrator if not provided and sets up
        Opik client and evaluator.

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        # Create orchestrator if needed
        if self._orchestrator is None:
            if not ORCHESTRATOR_AVAILABLE or _CoreOrchestrator is None:
                log.error("orchestrator_not_available")
                return False
            self._orchestrator = _CoreOrchestrator()

        # Initialize wrapped orchestrator
        orchestrator_init = await self._orchestrator.initialize()

        # Initialize evaluator if available
        if EVALUATOR_AVAILABLE and self.config.evaluate_outputs and _OpikEvaluator is not None:
            self._evaluator = _OpikEvaluator(
                project_name=f"{self.config.project_name}-eval"
            )

        # Configure Opik if available
        if OPIK_AVAILABLE and _opik_module is not None:
            try:
                api_key = os.getenv("OPIK_API_KEY")
                if api_key:
                    _opik_module.configure(api_key=api_key)
                    log.info("opik_configured", project=self.config.project_name)
            except Exception as e:
                log.warning("opik_configuration_failed", error=str(e))

        self._initialized = orchestrator_init

        log.info(
            "observable_orchestrator_initialized",
            opik_available=OPIK_AVAILABLE,
            evaluator_available=self._evaluator is not None,
            orchestrator_ready=orchestrator_init,
        )

        return self._initialized

    @property
    def is_initialized(self) -> bool:
        """Check if orchestrator is initialized."""
        return self._initialized

    @property
    def opik_available(self) -> bool:
        """Check if Opik tracing is available."""
        return OPIK_AVAILABLE

    @property
    def agents(self) -> Dict[str, Any]:
        """Get registered agents from wrapped orchestrator."""
        if self._orchestrator:
            return self._orchestrator.agents
        return {}

    async def coordinate_agents(
        self,
        task: str,
        agents: Optional[List[str]] = None,
        trace_id: Optional[str] = None,
    ) -> Any:  # Returns TaskResult-like object
        """
        Coordinate agents with full observability tracing.

        Creates a parent span for the coordination and child spans
        for each agent execution.

        Args:
            task: Task description
            agents: Specific agents to use (auto-selects if None)
            trace_id: Optional trace ID for correlation

        Returns:
            TaskResult with execution details and tracing metadata
        """
        if not self._initialized:
            await self.initialize()

        trace_id = trace_id or f"coord_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        self._stats["total_traces"] += 1

        log.info(
            "starting_traced_coordination",
            trace_id=trace_id,
            task=task[:100],
            agents=agents,
        )

        try:
            # Build span metadata
            metadata = {
                "trace_id": trace_id,
                "task": task[:500],
                "requested_agents": agents,
                "timestamp": datetime.now().isoformat(),
                "config_tags": self.config.tags,
            }

            # Execute with tracing wrapper
            if OPIK_AVAILABLE and self.config.trace_coordination:
                result = await self._traced_coordinate(
                    task=task,
                    agents=agents,
                    metadata=metadata,
                )
            else:
                # Fallback without tracing
                result = await self._orchestrator.coordinate_agents(
                    task=task,
                    agents=agents,
                )

            # Add trace metadata to result
            duration_ms = (time.time() - start_time) * 1000
            if result.metadata is None:
                result.metadata = {}
            result.metadata["trace_id"] = trace_id
            result.metadata["traced"] = OPIK_AVAILABLE
            result.metadata["observability_duration_ms"] = round(duration_ms, 2)

            # Evaluate output quality if enabled
            if self._evaluator and self.config.evaluate_outputs:
                await self._evaluate_result(task, result)

            self._stats["successful_traces"] += 1

            log.info(
                "traced_coordination_completed",
                trace_id=trace_id,
                status=result.status.value if hasattr(result.status, 'value') else str(result.status),
                duration_ms=round(duration_ms, 2),
            )

            return result

        except Exception as e:
            self._stats["failed_traces"] += 1
            log.error(
                "traced_coordination_failed",
                trace_id=trace_id,
                error=str(e),
            )
            raise

    async def _traced_coordinate(
        self,
        task: str,
        agents: Optional[List[str]],
        metadata: Dict[str, Any],  # noqa: ARG002 - reserved for future use
    ) -> Any:  # Returns TaskResult-like object
        """Execute coordination with Opik tracing."""
        if not OPIK_AVAILABLE or _opik_track is None or self._orchestrator is None:
            if self._orchestrator is None:
                raise RuntimeError("Orchestrator not initialized")
            return await self._orchestrator.coordinate_agents(task, agents)

        # Create traced version of coordinate_agents
        @_opik_track(
            name="coordinate_agents",
            tags=self.config.tags + ["coordination"],
        )
        async def tracked_coordinate():
            assert self._orchestrator is not None  # Type narrowing
            return await self._orchestrator.coordinate_agents(task, agents)

        return await tracked_coordinate()

    async def run_ralph_iteration(
        self,
        iteration_id: Optional[str] = None,
    ) -> Any:  # Returns TaskResult-like object
        """
        Run a Ralph Loop iteration with full tracing.

        Args:
            iteration_id: Optional iteration ID for correlation

        Returns:
            TaskResult from the iteration
        """
        if not self._initialized:
            await self.initialize()

        if self._orchestrator is None:
            raise RuntimeError("Orchestrator not initialized")

        iteration_id = iteration_id or f"ralph_{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        log.info("starting_traced_ralph_iteration", iteration_id=iteration_id)

        try:
            if OPIK_AVAILABLE and self.config.trace_ralph_iteration and _opik_track is not None:
                orchestrator = self._orchestrator  # Capture for closure

                @_opik_track(
                    name="ralph_iteration",
                    tags=self.config.tags + ["ralph", "iteration"],
                )
                async def tracked_iteration():
                    return await orchestrator.run_ralph_iteration()

                result = await tracked_iteration()
            else:
                result = await self._orchestrator.run_ralph_iteration()

            duration_ms = (time.time() - start_time) * 1000

            if result.metadata is None:
                result.metadata = {}
            result.metadata["iteration_id"] = iteration_id
            result.metadata["iteration_duration_ms"] = round(duration_ms, 2)

            log.info(
                "traced_ralph_iteration_completed",
                iteration_id=iteration_id,
                status=result.status.value if hasattr(result.status, 'value') else str(result.status),
                duration_ms=round(duration_ms, 2),
            )

            return result

        except Exception as e:
            log.error(
                "traced_ralph_iteration_failed",
                iteration_id=iteration_id,
                error=str(e),
            )
            raise

    async def _evaluate_result(
        self,
        task: str,
        result: Any,  # TaskResult-like object
    ) -> Optional[Any]:  # Returns EvaluationResult-like object
        """
        Evaluate task result quality using OpikEvaluator.

        Args:
            task: Original task description
            result: Task execution result

        Returns:
            EvaluationResult if evaluation succeeded
        """
        if not self._evaluator:
            return None

        try:
            # Extract output text for evaluation
            output_text = ""
            if result.output:
                if isinstance(result.output, dict):
                    # Extract text from agent results
                    results = result.output.get("results", [])
                    output_text = " ".join(
                        r.get("output", "") for r in results if isinstance(r, dict)
                    )
                elif isinstance(result.output, str):
                    output_text = result.output

            if not output_text:
                return None

            # Evaluate relevance
            eval_result = await self._evaluator.evaluate_relevance(
                input=task,
                output=output_text,
            )

            self._stats["evaluation_count"] += 1

            log.debug(
                "result_evaluated",
                task_id=result.task_id,
                relevance_score=eval_result.score,
                passed=eval_result.passed,
            )

            return eval_result

        except Exception as e:
            log.warning("evaluation_failed", error=str(e))
            return None

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status with observability stats."""
        base_status = {}
        if self._orchestrator:
            base_status = self._orchestrator.get_status()

        return {
            **base_status,
            "observability": {
                "opik_available": OPIK_AVAILABLE,
                "evaluator_available": self._evaluator is not None,
                "config": {
                    "project_name": self.config.project_name,
                    "trace_coordination": self.config.trace_coordination,
                    "evaluate_outputs": self.config.evaluate_outputs,
                    "tags": self.config.tags,
                },
                "stats": self._stats,
            },
        }

    def get_stats(self) -> Dict[str, int]:
        """Get observability statistics."""
        return self._stats.copy()


# =============================================================================
# Factory Functions
# =============================================================================

async def create_observable_orchestrator(
    config: Optional[ObservabilityConfig] = None,
    **kwargs: Any,
) -> ObservableOrchestrator:
    """
    Create and initialize an ObservableOrchestrator.

    Args:
        config: Observability configuration
        **kwargs: Additional orchestrator configuration

    Returns:
        Initialized ObservableOrchestrator
    """
    orchestrator = ObservableOrchestrator(config=config)
    await orchestrator.initialize()
    return orchestrator


def get_observable_orchestrator_sync(
    config: Optional[ObservabilityConfig] = None,
) -> ObservableOrchestrator:
    """
    Get an ObservableOrchestrator synchronously (not initialized).

    Args:
        config: Observability configuration

    Returns:
        ObservableOrchestrator (call initialize() before use)
    """
    return ObservableOrchestrator(config=config)


# =============================================================================
# Convenience Decorators
# =============================================================================

def trace_orchestration(
    project: str = "unleash-orchestration",
    tags: Optional[List[str]] = None,
):
    """
    Decorator to add orchestration tracing to any function.

    Args:
        project: Project name for Opik
        tags: Tags for the trace

    Returns:
        Decorated function with tracing
    """
    return traced(
        tags=(tags or []) + ["orchestration", project],
        capture_input=True,
        capture_output=True,
    )


# =============================================================================
# Module Exports
# =============================================================================

# Layer availability
ORCHESTRATION_OBSERVABILITY_AVAILABLE = ORCHESTRATOR_AVAILABLE and (OPIK_AVAILABLE or EVALUATOR_AVAILABLE)

__all__ = [
    # Main class
    "ObservableOrchestrator",
    # Configuration
    "ObservabilityConfig",
    "TraceLevel",
    # Factory functions
    "create_observable_orchestrator",
    "get_observable_orchestrator_sync",
    # Decorators
    "traced",
    "trace_orchestration",
    # Availability flags
    "OPIK_AVAILABLE",
    "EVALUATOR_AVAILABLE",
    "ORCHESTRATION_OBSERVABILITY_AVAILABLE",
]
