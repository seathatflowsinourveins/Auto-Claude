"""
Unified Pipeline System - Unleashed Platform V3

This module provides high-level pipelines that combine all SDK layers
for common autonomous agent tasks.

Pipelines:
- Deep Research Pipeline: Research → Extract → Reason → Remember
- Self-Improvement Pipeline: Evaluate → Evolve → Optimize → Deploy
- Code Analysis Pipeline: Analyze → Reason → Generate → Test
- Autonomous Task Pipeline: Plan → Execute → Verify → Learn
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional

# Structured logging
try:
    from .logging_config import get_logger, generate_correlation_id
    _logger = get_logger("rag_pipeline")
except ImportError:
    import logging
    _logger = logging.getLogger(__name__)
    generate_correlation_id = lambda: "corr-fallback"


# =============================================================================
# PIPELINE DATA STRUCTURES
# =============================================================================

class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class PipelineStep:
    """A single step in a pipeline."""
    name: str
    layer: str
    operation: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    status: PipelineStatus = PipelineStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    latency_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    pipeline_name: str
    status: PipelineStatus
    steps: List[PipelineStep]
    total_latency_ms: float
    outputs: Dict[str, Any]
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# BASE PIPELINE
# =============================================================================

class Pipeline:
    """Base class for all pipelines."""

    def __init__(self, name: str):
        self.name = name
        self._steps: List[PipelineStep] = []
        self._orchestrator = None

    async def _get_orchestrator(self):
        """Lazy load the orchestrator."""
        if self._orchestrator is None:
            from .ultimate_orchestrator import get_orchestrator
            self._orchestrator = await get_orchestrator()
        return self._orchestrator

    def add_step(self, name: str, layer: str, operation: str, **inputs) -> Pipeline:
        """Add a step to the pipeline."""
        self._steps.append(PipelineStep(
            name=name,
            layer=layer,
            operation=operation,
            inputs=inputs
        ))
        return self

    async def execute(self, **initial_inputs) -> PipelineResult:
        """Execute the pipeline."""
        start_time = time.time()
        outputs = initial_inputs.copy()
        executed_steps = []
        correlation_id = generate_correlation_id()

        # Log pipeline start with structured context
        _logger.info(
            "Pipeline execution started",
            pipeline=self.name,
            correlation_id=correlation_id,
            step_count=len(self._steps),
            inputs=list(initial_inputs.keys()),
        )

        try:
            orch = await self._get_orchestrator()

            for step_index, step in enumerate(self._steps):
                step.status = PipelineStatus.RUNNING
                step.started_at = datetime.now(timezone.utc).isoformat()

                # Log step start
                _logger.debug(
                    "Pipeline step started",
                    pipeline=self.name,
                    correlation_id=correlation_id,
                    step_name=step.name,
                    step_index=step_index,
                    layer=step.layer,
                    operation=step.operation,
                )

                # Resolve input references
                resolved_inputs = {}
                for key, value in step.inputs.items():
                    if isinstance(value, str) and value.startswith("$"):
                        # Reference to previous output
                        ref_key = value[1:]
                        resolved_inputs[key] = outputs.get(ref_key, value)
                    else:
                        resolved_inputs[key] = value

                # Add any matching outputs as inputs
                for key, value in outputs.items():
                    if key not in resolved_inputs:
                        resolved_inputs[key] = value

                # Execute the step
                step_start = time.time()

                try:
                    from .ultimate_orchestrator import SDKLayer

                    layer_map = {
                        "optimization": SDKLayer.OPTIMIZATION,
                        "orchestration": SDKLayer.ORCHESTRATION,
                        "memory": SDKLayer.MEMORY,
                        "reasoning": SDKLayer.REASONING,
                        "research": SDKLayer.RESEARCH,
                        "self_improvement": SDKLayer.SELF_IMPROVEMENT,
                    }

                    layer = layer_map.get(step.layer.lower())
                    if layer:
                        result = await orch.execute(layer, step.operation, **resolved_inputs)

                        step.outputs = result.data if result.success else {}
                        step.status = PipelineStatus.COMPLETED if result.success else PipelineStatus.FAILED
                        step.error = result.error

                        # Merge outputs
                        if result.success and result.data:
                            outputs[step.name] = result.data
                            if isinstance(result.data, dict):
                                outputs.update(result.data)
                    else:
                        step.status = PipelineStatus.FAILED
                        step.error = f"Unknown layer: {step.layer}"

                except Exception as e:
                    step.status = PipelineStatus.FAILED
                    step.error = str(e)
                    _logger.exception(
                        "Pipeline step execution failed",
                        pipeline=self.name,
                        correlation_id=correlation_id,
                        step_name=step.name,
                        error_type=type(e).__name__,
                    )

                step.latency_ms = (time.time() - step_start) * 1000
                step.completed_at = datetime.now(timezone.utc).isoformat()
                executed_steps.append(step)

                # Log step completion
                _logger.info(
                    "Pipeline step completed",
                    pipeline=self.name,
                    correlation_id=correlation_id,
                    step_name=step.name,
                    step_index=step_index,
                    status=step.status.name,
                    duration_ms=step.latency_ms,
                    error=step.error,
                )

                # Stop on failure unless configured otherwise
                if step.status == PipelineStatus.FAILED:
                    break

            # Determine overall status
            failed_steps = [s for s in executed_steps if s.status == PipelineStatus.FAILED]
            overall_status = PipelineStatus.FAILED if failed_steps else PipelineStatus.COMPLETED
            total_latency = (time.time() - start_time) * 1000

            # Log pipeline completion
            _logger.info(
                "Pipeline execution completed",
                pipeline=self.name,
                correlation_id=correlation_id,
                status=overall_status.name,
                total_duration_ms=total_latency,
                steps_executed=len(executed_steps),
                steps_failed=len(failed_steps),
                error=failed_steps[0].error if failed_steps else None,
            )

            return PipelineResult(
                pipeline_name=self.name,
                status=overall_status,
                steps=executed_steps,
                total_latency_ms=total_latency,
                outputs=outputs,
                error=failed_steps[0].error if failed_steps else None,
                metadata={"correlation_id": correlation_id},
            )

        except Exception as e:
            total_latency = (time.time() - start_time) * 1000
            _logger.exception(
                "Pipeline execution failed",
                pipeline=self.name,
                correlation_id=correlation_id,
                duration_ms=total_latency,
                error_type=type(e).__name__,
            )
            return PipelineResult(
                pipeline_name=self.name,
                status=PipelineStatus.FAILED,
                steps=executed_steps,
                total_latency_ms=total_latency,
                outputs=outputs,
                error=str(e),
                metadata={"correlation_id": correlation_id},
            )


# =============================================================================
# PRE-BUILT PIPELINES
# =============================================================================

class DeepResearchPipeline(Pipeline):
    """
    Deep Research Pipeline

    Flow: Search → Extract → Reason → Remember
    """

    def __init__(self):
        super().__init__("deep_research")

    async def research(self, topic: str, depth: str = "comprehensive") -> PipelineResult:
        """Execute deep research on a topic."""
        # Build pipeline dynamically based on depth
        self._steps = []

        # Step 1: Web research
        self.add_step(
            "web_search",
            "research",
            "scrape",
            url=f"https://www.google.com/search?q={topic.replace(' ', '+')}"
        )

        # Step 2: Extract key information
        self.add_step(
            "extract_info",
            "research",
            "extract",
            url="$web_search.url"
        )

        # Step 3: Reason about findings
        self.add_step(
            "analyze",
            "reasoning",
            "completion",
            messages=[{
                "role": "user",
                "content": f"Analyze the following research on '{topic}': $extract_info"
            }]
        )

        # Step 4: Remember key findings
        self.add_step(
            "remember",
            "memory",
            "add",
            content=f"Research findings on {topic}: $analyze",
            session_id="research"
        )

        return await self.execute(topic=topic, depth=depth)


class SelfImprovementPipeline(Pipeline):
    """
    Self-Improvement Pipeline

    Flow: Evaluate → Evolve → Optimize → Validate
    """

    def __init__(self):
        super().__init__("self_improvement")

    async def improve(
        self,
        target: str,
        current_version: Any,
        fitness_function: str = "accuracy",
        generations: int = 10
    ) -> PipelineResult:
        """Improve a target (prompt, workflow, etc.) through evolution."""
        self._steps = []

        # Step 1: Evaluate current version
        self.add_step(
            "evaluate",
            "optimization",
            "predict",
            prompt=f"Evaluate this {target}: {current_version}"
        )

        # Step 2: Evolve using MAP-Elites
        self.add_step(
            "evolve",
            "self_improvement",
            "evolve",
            generations=generations
        )

        # Step 3: Optimize best candidates with DSPy
        self.add_step(
            "optimize",
            "optimization",
            "optimize",
            module=current_version
        )

        # Step 4: Remember the improvement
        self.add_step(
            "remember",
            "memory",
            "add",
            content=f"Improved {target}: $optimize",
            memory_type="learning"
        )

        return await self.execute(
            target=target,
            current_version=current_version,
            fitness_function=fitness_function
        )


class AutonomousTaskPipeline(Pipeline):
    """
    Autonomous Task Pipeline

    Flow: Plan → Execute → Verify → Learn
    """

    def __init__(self):
        super().__init__("autonomous_task")

    async def execute_task(self, task: str, context: Optional[str] = None) -> PipelineResult:
        """Execute a task autonomously."""
        self._steps = []

        # Step 1: Recall relevant context
        self.add_step(
            "recall_context",
            "memory",
            "search",
            query=task
        )

        # Step 2: Plan the approach
        self.add_step(
            "plan",
            "reasoning",
            "completion",
            messages=[{
                "role": "user",
                "content": f"Plan how to accomplish: {task}\n\nContext: $recall_context"
            }]
        )

        # Step 3: Create execution workflow
        self.add_step(
            "orchestrate",
            "orchestration",
            "execute",
            inputs={"task": task, "plan": "$plan"}
        )

        # Step 4: Remember what we learned
        self.add_step(
            "learn",
            "memory",
            "add",
            content=f"Completed task '{task}': $orchestrate",
            memory_type="learning"
        )

        return await self.execute(task=task, context=context or "")


# =============================================================================
# PIPELINE FACTORY
# =============================================================================

class PipelineFactory:
    """Factory for creating and managing pipelines."""

    _pipelines = {
        "deep_research": DeepResearchPipeline,
        "self_improvement": SelfImprovementPipeline,
        "autonomous_task": AutonomousTaskPipeline,
    }

    @classmethod
    def create(cls, pipeline_type: str) -> Pipeline:
        """Create a pipeline by type."""
        pipeline_class = cls._pipelines.get(pipeline_type)
        if pipeline_class:
            return pipeline_class()
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")

    @classmethod
    def list_pipelines(cls) -> List[str]:
        """List available pipeline types."""
        return list(cls._pipelines.keys())

    @classmethod
    def register(cls, name: str, pipeline_class: type) -> None:
        """Register a custom pipeline."""
        cls._pipelines[name] = pipeline_class


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def deep_research(topic: str, depth: str = "comprehensive") -> PipelineResult:
    """Quick access to deep research pipeline."""
    pipeline = DeepResearchPipeline()
    return await pipeline.research(topic, depth)


async def self_improve(target: str, current_version: Any, generations: int = 10) -> PipelineResult:
    """Quick access to self-improvement pipeline."""
    pipeline = SelfImprovementPipeline()
    return await pipeline.improve(target, current_version, generations=generations)


async def autonomous_task(task: str, context: Optional[str] = None) -> PipelineResult:
    """Quick access to autonomous task pipeline."""
    pipeline = AutonomousTaskPipeline()
    return await pipeline.execute_task(task, context)


# =============================================================================
# MAIN - DEMO
# =============================================================================

async def demo():
    """Demonstrate the unified pipeline system."""
    print("=" * 60)
    print("UNIFIED PIPELINE SYSTEM - DEMO")
    print("=" * 60)

    # List available pipelines
    print("\nAvailable Pipelines:")
    for name in PipelineFactory.list_pipelines():
        print(f"  - {name}")

    # Test Deep Research Pipeline
    print("\n" + "-" * 40)
    print("Testing Deep Research Pipeline")
    print("-" * 40)

    result = await deep_research("DSPy prompt optimization benchmarks 2025")

    print(f"Status: {result.status.name}")
    print(f"Total Latency: {result.total_latency_ms:.2f}ms")
    print(f"Steps Executed: {len(result.steps)}")

    for step in result.steps:
        status_icon = "✅" if step.status == PipelineStatus.COMPLETED else "❌"
        print(f"  {status_icon} {step.name}: {step.latency_ms:.2f}ms")

    # Test Autonomous Task Pipeline
    print("\n" + "-" * 40)
    print("Testing Autonomous Task Pipeline")
    print("-" * 40)

    result = await autonomous_task("Analyze the best SDK stack for AI agents")

    print(f"Status: {result.status.name}")
    print(f"Total Latency: {result.total_latency_ms:.2f}ms")

    for step in result.steps:
        status_icon = "✅" if step.status == PipelineStatus.COMPLETED else "❌"
        print(f"  {status_icon} {step.name}: {step.latency_ms:.2f}ms")


if __name__ == "__main__":
    asyncio.run(demo())
