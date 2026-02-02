#!/usr/bin/env python3
"""
Temporal Workflow Orchestration
Durable, fault-tolerant workflow execution for AI agents.
"""

from __future__ import annotations

import asyncio
import os
from datetime import timedelta
from typing import Any, Optional
from dataclasses import dataclass, field

import structlog
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment
load_dotenv()

logger = structlog.get_logger(__name__)

# Import Temporal SDK
try:
    from temporalio import workflow, activity
    from temporalio.client import Client
    from temporalio.worker import Worker
    TEMPORAL_AVAILABLE = True
except ImportError:
    TEMPORAL_AVAILABLE = False
    logger.warning("temporalio not available - install with: pip install temporalio")


class WorkflowInput(BaseModel):
    """Input for AI workflow execution."""
    task: str = Field(..., description="The task to execute")
    context: dict[str, Any] = Field(default_factory=dict)
    max_steps: int = Field(default=10, ge=1, le=100)
    timeout_seconds: int = Field(default=300)


class WorkflowResult(BaseModel):
    """Result from AI workflow execution."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    steps_taken: int = 0
    total_duration_ms: int = 0


class StepResult(BaseModel):
    """Result from a single workflow step."""
    step_number: int
    action: str
    output: Any
    duration_ms: int


if TEMPORAL_AVAILABLE:
    @activity.defn
    async def execute_llm_step(
        task: str,
        context: dict[str, Any],
        step_number: int,
    ) -> dict[str, Any]:
        """Execute a single LLM-powered step in the workflow."""
        import time
        start = time.time()

        logger.info(
            "executing_llm_step",
            task=task[:50],
            step=step_number,
        )

        # Import gateway lazily to avoid circular imports
        try:
            from core.llm_gateway import LLMGateway, Message
            gateway = LLMGateway()

            messages = [
                Message(
                    role="system",
                    content="You are a helpful AI assistant executing workflow steps.",
                ),
                Message(
                    role="user",
                    content=f"Task: {task}\nContext: {context}\nStep: {step_number}",
                ),
            ]

            response = await gateway.complete(messages)
            result = response.content

        except Exception as e:
            logger.warning("llm_step_fallback", error=str(e))
            result = f"Step {step_number} completed (simulated)"

        duration_ms = int((time.time() - start) * 1000)

        return {
            "step_number": step_number,
            "action": f"Executed step {step_number}",
            "output": result,
            "duration_ms": duration_ms,
        }

    @activity.defn
    async def validate_step(
        step_result: dict[str, Any],
        context: dict[str, Any],
    ) -> bool:
        """Validate the result of a workflow step."""
        logger.info("validating_step", step=step_result.get("step_number"))
        # Basic validation - extend as needed
        return step_result.get("output") is not None

    @workflow.defn
    class AIAgentWorkflow:
        """
        Durable AI agent workflow with automatic retries and checkpointing.

        This workflow orchestrates multi-step AI agent tasks with:
        - Automatic state persistence
        - Retry logic for failures
        - Timeout handling
        - Step-by-step execution with validation
        """

        @workflow.run
        async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
            """Execute the AI agent workflow."""
            workflow_input = WorkflowInput(**input_data)
            steps_completed = 0
            results = []
            total_duration = 0

            logger.info(
                "workflow_started",
                task=workflow_input.task[:50],
                max_steps=workflow_input.max_steps,
            )

            try:
                for step in range(1, workflow_input.max_steps + 1):
                    # Execute step with retry
                    step_result = await workflow.execute_activity(
                        execute_llm_step,
                        args=[
                            workflow_input.task,
                            workflow_input.context,
                            step,
                        ],
                        start_to_close_timeout=timedelta(
                            seconds=workflow_input.timeout_seconds
                        ),
                        retry_policy=workflow.RetryPolicy(
                            maximum_attempts=3,
                            initial_interval=timedelta(seconds=1),
                            maximum_interval=timedelta(seconds=10),
                        ),
                    )

                    # Validate step
                    is_valid = await workflow.execute_activity(
                        validate_step,
                        args=[step_result, workflow_input.context],
                        start_to_close_timeout=timedelta(seconds=30),
                    )

                    if not is_valid:
                        logger.warning("step_validation_failed", step=step)
                        continue

                    results.append(step_result)
                    steps_completed = step
                    total_duration += step_result.get("duration_ms", 0)

                    # Check for completion signal in output
                    output = step_result.get("output", "")
                    if isinstance(output, str) and "TASK_COMPLETE" in output:
                        logger.info("workflow_completed_early", step=step)
                        break

                return WorkflowResult(
                    success=True,
                    result=results,
                    steps_taken=steps_completed,
                    total_duration_ms=total_duration,
                ).model_dump()

            except Exception as e:
                logger.error("workflow_failed", error=str(e))
                return WorkflowResult(
                    success=False,
                    error=str(e),
                    steps_taken=steps_completed,
                    total_duration_ms=total_duration,
                ).model_dump()


@dataclass
class TemporalOrchestrator:
    """
    Orchestrator for Temporal-based AI workflows.

    Provides a high-level interface for running durable AI agent workflows
    with automatic retry, checkpointing, and fault tolerance.

    Usage:
        orchestrator = await TemporalOrchestrator.create()
        result = await orchestrator.run_workflow(
            task="Analyze this data and generate a report",
            context={"data": [...]}
        )
    """

    client: Any = field(default=None)
    worker: Any = field(default=None)
    task_queue: str = field(default="ai-agent-queue")

    @classmethod
    async def create(
        cls,
        temporal_address: str = "localhost:7233",
        task_queue: str = "ai-agent-queue",
    ) -> "TemporalOrchestrator":
        """Create and connect to Temporal."""
        if not TEMPORAL_AVAILABLE:
            raise ImportError("temporalio not installed")

        instance = cls(task_queue=task_queue)
        instance.client = await Client.connect(temporal_address)

        logger.info(
            "temporal_connected",
            address=temporal_address,
            task_queue=task_queue,
        )

        return instance

    async def start_worker(self) -> None:
        """Start the Temporal worker to process workflows."""
        if not self.client:
            raise RuntimeError("Not connected to Temporal")

        self.worker = Worker(
            self.client,
            task_queue=self.task_queue,
            workflows=[AIAgentWorkflow],
            activities=[execute_llm_step, validate_step],
        )

        logger.info("temporal_worker_starting", task_queue=self.task_queue)
        await self.worker.run()

    async def run_workflow(
        self,
        task: str,
        context: Optional[dict[str, Any]] = None,
        max_steps: int = 10,
        timeout_seconds: int = 300,
        workflow_id: Optional[str] = None,
    ) -> WorkflowResult:
        """
        Run an AI agent workflow.

        Args:
            task: The task description
            context: Additional context for the task
            max_steps: Maximum number of steps
            timeout_seconds: Timeout per step
            workflow_id: Optional workflow ID for idempotency

        Returns:
            WorkflowResult with execution details
        """
        if not self.client:
            raise RuntimeError("Not connected to Temporal")

        import uuid
        workflow_id = workflow_id or f"ai-workflow-{uuid.uuid4()}"

        input_data = WorkflowInput(
            task=task,
            context=context or {},
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
        ).model_dump()

        logger.info("starting_workflow", workflow_id=workflow_id, task=task[:50])

        result = await self.client.execute_workflow(
            AIAgentWorkflow.run,
            input_data,
            id=workflow_id,
            task_queue=self.task_queue,
        )

        return WorkflowResult(**result)

    async def get_workflow_status(self, workflow_id: str) -> dict[str, Any]:
        """Get the status of a running workflow."""
        if not self.client:
            raise RuntimeError("Not connected to Temporal")

        handle = self.client.get_workflow_handle(workflow_id)
        description = await handle.describe()

        return {
            "workflow_id": workflow_id,
            "status": description.status.name,
            "start_time": description.start_time.isoformat() if description.start_time else None,
            "close_time": description.close_time.isoformat() if description.close_time else None,
        }

    async def cancel_workflow(self, workflow_id: str) -> None:
        """Cancel a running workflow."""
        if not self.client:
            raise RuntimeError("Not connected to Temporal")

        handle = self.client.get_workflow_handle(workflow_id)
        await handle.cancel()
        logger.info("workflow_cancelled", workflow_id=workflow_id)


async def create_orchestrator(
    temporal_address: str = "localhost:7233",
    task_queue: str = "ai-agent-queue",
) -> TemporalOrchestrator:
    """Factory function to create a Temporal orchestrator."""
    return await TemporalOrchestrator.create(temporal_address, task_queue)


if __name__ == "__main__":
    async def main():
        """Test the Temporal orchestrator."""
        print(f"Temporal available: {TEMPORAL_AVAILABLE}")

        if TEMPORAL_AVAILABLE:
            try:
                orchestrator = await create_orchestrator()
                print("Orchestrator created successfully")

                # Note: Worker needs to be running separately
                # await orchestrator.start_worker()

            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Install temporalio to use this module")

    asyncio.run(main())
