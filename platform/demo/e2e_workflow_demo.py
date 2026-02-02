#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.26.0",
#     "structlog>=24.1.0",
#     "qdrant-client>=1.7.0",
#     "neo4j>=5.0.0",
#     "rich>=13.7.0",
# ]
# ///
"""
End-to-End Workflow Demo - Ultimate Autonomous Platform

Demonstrates the complete workflow from task submission through all components:

1. Task Submission → Swarm Coordinator
2. Intelligent Routing → Auto-Claude or Local
3. Knowledge Graph → Decision Recording
4. Memory Persistence → Qdrant + Letta
5. Result Aggregation → Final Output

This demo exercises all major platform components in a realistic workflow.

Usage:
    uv run e2e_workflow_demo.py
    python e2e_workflow_demo.py --verbose
"""

from __future__ import annotations

import asyncio
import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "bridges"))
sys.path.insert(0, str(Path(__file__).parent.parent / "swarm"))

import structlog

# Try rich for beautiful output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.live import Live
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

logger = structlog.get_logger(__name__)


@dataclass
class WorkflowStep:
    """A step in the workflow."""
    name: str
    component: str
    status: str = "pending"  # pending, running, success, failed
    duration_ms: float = 0
    output: Any = None
    error: Optional[str] = None


@dataclass
class WorkflowResult:
    """Result of the complete workflow."""
    success: bool
    total_duration_ms: float
    steps: List[WorkflowStep]
    final_output: Any


class EndToEndWorkflow:
    """
    Orchestrates the complete end-to-end workflow.

    Demonstrates:
    - Swarm task coordination (Queen topology)
    - Intelligent task routing
    - Knowledge graph recording
    - Memory persistence
    - Multi-component integration
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.steps: List[WorkflowStep] = []
        self._start_time: float = 0

    async def run_demo(self, task_description: str) -> WorkflowResult:
        """Run the complete demo workflow."""
        self._start_time = time.perf_counter()

        self._print_header("ULTIMATE AUTONOMOUS PLATFORM - E2E WORKFLOW DEMO")

        # Step 1: Initialize Components
        step1 = await self._step_init_components()
        self.steps.append(step1)
        if step1.status == "failed":
            return self._create_result(False, step1.error)

        # Step 2: Submit Task to Swarm
        step2 = await self._step_submit_task(task_description)
        self.steps.append(step2)

        # Step 3: Intelligent Routing
        step3 = await self._step_route_task(task_description)
        self.steps.append(step3)

        # Step 4: Record in Knowledge Graph
        step4 = await self._step_record_knowledge(task_description)
        self.steps.append(step4)

        # Step 5: Store in Memory
        step5 = await self._step_persist_memory(task_description)
        self.steps.append(step5)

        # Step 6: Aggregate Results
        step6 = await self._step_aggregate_results()
        self.steps.append(step6)

        # Print summary
        self._print_summary()

        return self._create_result(
            all(s.status == "success" for s in self.steps),
            step6.output
        )

    async def _step_init_components(self) -> WorkflowStep:
        """Initialize all platform components."""
        step = WorkflowStep(
            name="Initialize Components",
            component="Platform",
            status="running"
        )
        self._print_step(step, "Initializing platform components...")
        start = time.perf_counter()

        try:
            # Import and initialize components
            from coordinator import SwarmMemory, QueenCoordinator, Agent, AgentRole
            from graphiti_bridge import GraphitiBridge
            from swarm_autoclaude_integration import SwarmAutoClaudeIntegration

            self.memory = SwarmMemory("http://localhost:6333")
            self.coordinator = QueenCoordinator(self.memory)
            self.graphiti = GraphitiBridge()
            self.integration = SwarmAutoClaudeIntegration()

            # Initialize Graphiti (Neo4j + Qdrant)
            await self.graphiti.initialize()

            # Register a demo worker
            worker = Agent(
                id="e2e-demo-worker-001",
                name="E2E Demo Worker",
                role=AgentRole.WORKER,
                capabilities=["demo", "test", "review"]
            )
            await self.coordinator.register_worker(worker)

            step.status = "success"
            step.output = {
                "components": ["SwarmMemory", "QueenCoordinator", "GraphitiBridge", "Integration"],
                "worker_registered": worker.id
            }

        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            logger.error("init_failed", error=str(e))

        step.duration_ms = (time.perf_counter() - start) * 1000
        self._print_step(step)
        return step

    async def _step_submit_task(self, description: str) -> WorkflowStep:
        """Submit task to swarm coordinator."""
        step = WorkflowStep(
            name="Submit Task",
            component="Swarm Coordinator",
            status="running"
        )
        self._print_step(step, f"Submitting: {description[:50]}...")
        start = time.perf_counter()

        try:
            from coordinator import Task

            task = Task(
                id=f"e2e-demo-{datetime.now(timezone.utc).strftime('%H%M%S')}",
                description=description,
                priority=8
            )

            task_id = await self.coordinator.submit_task(task)
            assigned = await self.coordinator.assign_task(task_id)

            step.status = "success"
            step.output = {
                "task_id": task_id,
                "assigned_to": assigned,
                "priority": task.priority
            }
            self.current_task = task

        except Exception as e:
            step.status = "failed"
            step.error = str(e)

        step.duration_ms = (time.perf_counter() - start) * 1000
        self._print_step(step)
        return step

    async def _step_route_task(self, description: str) -> WorkflowStep:
        """Route task through intelligent routing system."""
        step = WorkflowStep(
            name="Intelligent Routing",
            component="Integration Layer",
            status="running"
        )
        self._print_step(step, "Analyzing task for optimal routing...")
        start = time.perf_counter()

        try:
            from coordinator import Task

            # Analyze routing
            analysis_task = Task(id="analysis", description=description, priority=5)
            routing = self.integration.analyze_task(analysis_task)

            # Submit through integration layer
            integrated_task_id = await self.integration.submit_task(
                description=description,
                files=["demo_file.py"],
                priority=8,
                context={"demo": True, "iteration": 6}
            )

            step.status = "success"
            step.output = {
                "routing_decision": routing.value,
                "integrated_task_id": integrated_task_id,
                "routing_stats": self.integration.get_routing_stats()
            }

        except Exception as e:
            step.status = "failed"
            step.error = str(e)

        step.duration_ms = (time.perf_counter() - start) * 1000
        self._print_step(step)
        return step

    async def _step_record_knowledge(self, description: str) -> WorkflowStep:
        """Record task and decision in knowledge graph."""
        step = WorkflowStep(
            name="Knowledge Graph",
            component="Graphiti Bridge",
            status="running"
        )
        self._print_step(step, "Recording in Neo4j knowledge graph...")
        start = time.perf_counter()

        try:
            from graphiti_bridge import Entity, EntityType, Relationship, RelationType

            # Record task entity
            task_entity = Entity(
                id=f"task-demo-{datetime.now(timezone.utc).strftime('%H%M%S')}",
                type=EntityType.TASK,
                name=description[:80],
                properties={
                    "full_description": description,
                    "demo": True,
                    "iteration": 6
                }
            )
            await self.graphiti.add_entity(task_entity)

            # Record decision
            await self.graphiti.record_decision(
                decision_id=f"decision-demo-{datetime.now(timezone.utc).strftime('%H%M%S')}",
                description="E2E Demo workflow execution",
                context={
                    "task": description,
                    "components_used": ["Swarm", "Graphiti", "Integration"]
                },
                outcome="Demonstrated full platform integration",
                related_entities=[task_entity.id]
            )

            # Get statistics
            stats = await self.graphiti.get_statistics()

            step.status = "success"
            step.output = {
                "entity_created": task_entity.id,
                "graph_stats": stats
            }

        except Exception as e:
            step.status = "failed"
            step.error = str(e)

        step.duration_ms = (time.perf_counter() - start) * 1000
        self._print_step(step)
        return step

    async def _step_persist_memory(self, description: str) -> WorkflowStep:
        """Persist workflow memory to Qdrant."""
        step = WorkflowStep(
            name="Memory Persistence",
            component="Swarm Memory (Qdrant)",
            status="running"
        )
        self._print_step(step, "Persisting to vector memory...")
        start = time.perf_counter()

        try:
            # Store workflow execution record
            workflow_record = {
                "type": "e2e_workflow",
                "description": description,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "iteration": 6,
                "steps_completed": len([s for s in self.steps if s.status == "success"]),
                "total_steps": len(self.steps) + 2  # Include this and next step
            }

            success = await self.memory.store(
                f"workflows/e2e-demo/{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
                workflow_record
            )

            step.status = "success" if success else "failed"
            step.output = {
                "stored": success,
                "key": f"workflows/e2e-demo/{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
                "record": workflow_record
            }

        except Exception as e:
            step.status = "failed"
            step.error = str(e)

        step.duration_ms = (time.perf_counter() - start) * 1000
        self._print_step(step)
        return step

    async def _step_aggregate_results(self) -> WorkflowStep:
        """Aggregate results from all components."""
        step = WorkflowStep(
            name="Aggregate Results",
            component="Orchestrator",
            status="running"
        )
        self._print_step(step, "Aggregating final results...")
        start = time.perf_counter()

        try:
            # Collect all outputs
            successful_steps = [s for s in self.steps if s.status == "success"]
            failed_steps = [s for s in self.steps if s.status == "failed"]

            # Get final status from components
            swarm_status = await self.coordinator.get_status()
            routing_stats = self.integration.get_routing_stats()
            graph_stats = await self.graphiti.get_statistics()

            step.status = "success"
            step.output = {
                "workflow_success": len(failed_steps) == 0,
                "steps_succeeded": len(successful_steps),
                "steps_failed": len(failed_steps),
                "swarm": {
                    "workers": swarm_status["workers"],
                    "completed_tasks": swarm_status["completed_tasks"]
                },
                "routing": routing_stats,
                "knowledge_graph": graph_stats,
                "total_duration_ms": (time.perf_counter() - self._start_time) * 1000
            }

        except Exception as e:
            step.status = "failed"
            step.error = str(e)

        step.duration_ms = (time.perf_counter() - start) * 1000
        self._print_step(step)
        return step

    def _create_result(self, success: bool, output: Any) -> WorkflowResult:
        """Create the final workflow result."""
        total_duration = (time.perf_counter() - self._start_time) * 1000
        return WorkflowResult(
            success=success,
            total_duration_ms=total_duration,
            steps=self.steps,
            final_output=output
        )

    def _print_header(self, title: str) -> None:
        """Print workflow header."""
        if RICH_AVAILABLE:
            console.print(Panel.fit(
                f"[bold cyan]{title}[/bold cyan]\n"
                f"[dim]Iteration 6 - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC[/dim]",
                border_style="cyan"
            ))
        else:
            print("=" * 60)
            print(title)
            print("=" * 60)

    def _print_step(self, step: WorkflowStep, message: str = None) -> None:
        """Print step status."""
        # Use ASCII-safe icons for Windows compatibility
        status_icons = {
            "pending": "[...]",
            "running": "[>>]",
            "success": "[OK]",
            "failed": "[ERR]"
        }

        icon = status_icons.get(step.status, "[?]")

        if message:
            output = f"{icon} [{step.component}] {step.name}: {message}"
        elif step.status == "success":
            duration = f"({step.duration_ms:.0f}ms)"
            output = f"{icon} [{step.component}] {step.name} {duration}"
        elif step.status == "failed":
            output = f"{icon} [{step.component}] {step.name}: {step.error}"
        else:
            output = f"{icon} [{step.component}] {step.name}"

        if RICH_AVAILABLE:
            color = {"success": "green", "failed": "red", "running": "yellow"}.get(step.status, "white")
            console.print(f"[{color}]{output}[/{color}]")
        else:
            print(output)

    def _print_summary(self) -> None:
        """Print workflow summary."""
        total_duration = (time.perf_counter() - self._start_time) * 1000
        success_count = len([s for s in self.steps if s.status == "success"])
        total_count = len(self.steps)

        if RICH_AVAILABLE:
            table = Table(title="Workflow Summary", show_header=True)
            table.add_column("Step", style="cyan")
            table.add_column("Component", style="magenta")
            table.add_column("Status", justify="center")
            table.add_column("Duration", justify="right")

            for step in self.steps:
                status = "[green]OK[/green]" if step.status == "success" else "[red]FAIL[/red]"
                table.add_row(
                    step.name,
                    step.component,
                    status,
                    f"{step.duration_ms:.0f}ms"
                )

            console.print()
            console.print(table)
            console.print()
            console.print(f"[bold]Total: {success_count}/{total_count} steps succeeded in {total_duration:.0f}ms[/bold]")
        else:
            print("\n" + "=" * 60)
            print("WORKFLOW SUMMARY")
            print("-" * 60)
            for step in self.steps:
                status = "OK" if step.status == "success" else "FAIL"
                print(f"  {status:4} | {step.name:25} | {step.duration_ms:.0f}ms")
            print("-" * 60)
            print(f"Total: {success_count}/{total_count} steps in {total_duration:.0f}ms")
            print("=" * 60)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="E2E Workflow Demo")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--task", "-t", type=str,
                        default="Review and optimize the swarm coordinator for production deployment",
                        help="Task description to demo")
    args = parser.parse_args()

    workflow = EndToEndWorkflow(verbose=args.verbose)
    result = await workflow.run_demo(args.task)

    if result.success:
        print("\n[SUCCESS] E2E Workflow completed successfully!")
    else:
        print("\n[WARNING] E2E Workflow had failures")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
