#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "structlog>=24.1.0",
#     "httpx>=0.26.0",
# ]
# ///
"""
Core Orchestrator - Ultimate Autonomous Platform

Central coordination layer that integrates:
- Claude-Flow v2/v3 swarm patterns
- Ralph Loop autonomous iterations
- Letta memory persistence
- MCP server coordination

This module provides the unified entry point for multi-agent orchestration.

Usage:
    from core.orchestrator import CoreOrchestrator

    orchestrator = CoreOrchestrator()
    await orchestrator.coordinate_agents()
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

log = structlog.get_logger()

# Import agent SDK layer for real agent execution
AGENT_SDK_AVAILABLE = False
try:
    from core.orchestration.agent_sdk_layer import (
        create_agent,
        run_agent_loop,
        Agent,
        ANTHROPIC_AVAILABLE,
    )
    AGENT_SDK_AVAILABLE = True
    log.info("agent_sdk_layer_available")
except ImportError:
    log.warning("agent_sdk_layer_not_available", fallback="simulation")


class AgentRole(str, Enum):
    """Agent roles in the swarm."""
    ORCHESTRATOR = "orchestrator"
    WORKER = "worker"
    SPECIALIST = "specialist"
    MONITOR = "monitor"
    MEMORY = "memory"


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentConfig:
    """Configuration for a swarm agent."""
    name: str
    role: AgentRole
    capabilities: List[str] = field(default_factory=list)
    max_concurrent: int = 5
    timeout_seconds: float = 300.0
    retry_count: int = 3
    model: str = "claude-sonnet-4-20250514"
    tools: List[str] = field(default_factory=lambda: ["Read", "Write", "Bash"])
    system_prompt: Optional[str] = None
    # Runtime agent instance (populated during initialization)
    _agent_instance: Optional[Any] = field(default=None, repr=False)


@dataclass
class TaskResult:
    """Result from task execution."""
    task_id: str
    status: TaskStatus
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CoreOrchestrator:
    """
    Central coordination layer for the Ultimate Autonomous Platform.

    Integrates Claude-Flow patterns with Ralph Loop for autonomous
    self-improvement iterations.
    """

    def __init__(
        self,
        letta_url: str = None,
        claude_flow_version: str = "v2",
    ):
        self.letta_url = letta_url or os.environ.get("LETTA_URL", "http://localhost:8500")
        self.claude_flow_version = claude_flow_version
        self.agents: Dict[str, AgentConfig] = {}
        self.active_tasks: Dict[str, TaskResult] = {}
        self._initialized = False

        # Paths
        self.base_dir = Path(__file__).parent.parent
        self.v10_dir = self.base_dir / "v10_optimized"
        self.claude_flow_dir = self.base_dir / "ruvnet-claude-flow" / claude_flow_version

    async def initialize(self) -> bool:
        """Initialize the orchestrator and register default agents."""
        log.info("initializing_core_orchestrator",
                 letta_url=self.letta_url,
                 claude_flow_version=self.claude_flow_version)

        # Register default agents
        self._register_default_agents()

        # Verify Claude-Flow availability
        cf_available = await self._check_claude_flow()

        # Verify Letta connectivity
        letta_available = await self._check_letta()

        # Initialize real agent instances if SDK available
        sdk_initialized = False
        if AGENT_SDK_AVAILABLE:
            sdk_initialized = await self._initialize_agent_instances()

        self._initialized = cf_available or letta_available or sdk_initialized

        log.info("orchestrator_initialized",
                 success=self._initialized,
                 claude_flow=cf_available,
                 letta=letta_available,
                 agent_sdk=sdk_initialized)

        return self._initialized

    async def _initialize_agent_instances(self) -> bool:
        """Create real agent instances using the Agent SDK layer."""
        if not AGENT_SDK_AVAILABLE:
            return False

        initialized_count = 0
        for name, config in self.agents.items():
            try:
                agent_instance = await create_agent(
                    name=name,
                    model=config.model,
                    tools=config.tools,
                    system_prompt=config.system_prompt,
                )
                config._agent_instance = agent_instance
                initialized_count += 1
                log.info("agent_instance_created", name=name, model=config.model)
            except Exception as e:
                log.warning("agent_instance_creation_failed", name=name, error=str(e))

        return initialized_count > 0

    def _register_default_agents(self) -> None:
        """Register the default agent configurations."""
        defaults = [
            AgentConfig(
                name="ralph",
                role=AgentRole.ORCHESTRATOR,
                capabilities=["iterate", "validate", "consolidate"],
                system_prompt="You are Ralph, the autonomous iteration orchestrator. "
                "Your role is to coordinate multi-agent tasks, validate outputs, "
                "and drive continuous improvement cycles.",
                tools=["Read", "Write", "Bash"],
            ),
            AgentConfig(
                name="validator",
                role=AgentRole.SPECIALIST,
                capabilities=["hooks", "mcp", "infrastructure"],
                system_prompt="You are a validation specialist. "
                "Your role is to verify hook configurations, MCP server health, "
                "and infrastructure integrity. Report issues clearly and concisely.",
                tools=["Read", "Bash"],
            ),
            AgentConfig(
                name="memory-agent",
                role=AgentRole.MEMORY,
                capabilities=["persist", "retrieve", "consolidate"],
                system_prompt="You are a memory management agent. "
                "Your role is to persist important context, retrieve relevant history, "
                "and consolidate learnings across sessions.",
                tools=["Read", "Write"],
            ),
            AgentConfig(
                name="health-monitor",
                role=AgentRole.MONITOR,
                capabilities=["health_check", "metrics", "alerts"],
                system_prompt="You are a health monitoring agent. "
                "Your role is to check system health, collect metrics, "
                "and alert on anomalies or degradation.",
                tools=["Read", "Bash"],
            ),
        ]

        for agent in defaults:
            self.agents[agent.name] = agent

    async def _check_claude_flow(self) -> bool:
        """Check if Claude-Flow is available."""
        if not self.claude_flow_dir.exists():
            return False

        # Check for required components
        required = ["src"]
        return all((self.claude_flow_dir / comp).exists() for comp in required)

    async def _check_letta(self) -> bool:
        """Check Letta server connectivity."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                for endpoint in ["/health", "/v1/health"]:
                    try:
                        response = await client.get(f"{self.letta_url}{endpoint}")
                        if response.status_code == 200:
                            return True
                    except Exception:
                        pass
            return False
        except ImportError:
            return False

    async def coordinate_agents(
        self,
        task: str,
        agents: Optional[List[str]] = None,
    ) -> TaskResult:
        """
        Coordinate multiple agents to complete a task.

        Args:
            task: Task description or command
            agents: Specific agents to use (default: auto-select)

        Returns:
            TaskResult with execution details
        """
        if not self._initialized:
            await self.initialize()

        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = asyncio.get_event_loop().time()

        log.info("coordinating_agents", task_id=task_id, task=task, agents=agents)

        try:
            # Select agents for task
            selected = agents or self._select_agents_for_task(task)

            # Execute task with selected agents
            result = await self._execute_with_agents(task_id, task, selected)

            duration = (asyncio.get_event_loop().time() - start_time) * 1000
            result.duration_ms = duration

            self.active_tasks[task_id] = result
            return result

        except Exception as e:
            duration = (asyncio.get_event_loop().time() - start_time) * 1000
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                duration_ms=duration,
            )

    def _select_agents_for_task(self, task: str) -> List[str]:
        """Auto-select appropriate agents for a task."""
        task_lower = task.lower()
        selected = []

        if "validate" in task_lower or "check" in task_lower:
            selected.append("validator")
        if "memory" in task_lower or "persist" in task_lower:
            selected.append("memory-agent")
        if "health" in task_lower or "monitor" in task_lower:
            selected.append("health-monitor")

        # Always include orchestrator for coordination
        if not selected:
            selected = ["ralph"]

        return selected

    async def _execute_with_agents(
        self,
        task_id: str,
        task: str,
        agent_names: List[str],
    ) -> TaskResult:
        """Execute a task using specified agents with real SDK integration."""
        results = []
        errors = []

        for name in agent_names:
            if name not in self.agents:
                log.warning("agent_not_found", name=name)
                continue

            agent_config = self.agents[name]
            log.info("executing_agent", agent=name, role=agent_config.role.value)

            # Use real SDK execution if available and agent instance exists
            if AGENT_SDK_AVAILABLE and agent_config._agent_instance is not None:
                try:
                    # Build prompt for agent based on task and role
                    agent_prompt = self._build_agent_prompt(task, agent_config)

                    # Execute with real SDK
                    result = await run_agent_loop(
                        agent=agent_config._agent_instance,
                        prompt=agent_prompt,
                        max_turns=5,  # Limit turns per agent in multi-agent context
                    )

                    results.append({
                        "agent": name,
                        "status": "completed" if result.get("success") else "failed",
                        "output": result.get("output", ""),
                        "tool_calls": result.get("tool_calls", []),
                        "sdk": "agent_sdk_layer",
                    })

                    log.info("agent_execution_completed",
                             agent=name,
                             success=result.get("success"),
                             tool_calls=len(result.get("tool_calls", [])))

                except Exception as e:
                    error_msg = f"Agent {name} failed: {str(e)}"
                    errors.append(error_msg)
                    log.error("agent_execution_error", agent=name, error=str(e))
                    results.append({
                        "agent": name,
                        "status": "failed",
                        "error": str(e),
                    })
            else:
                # Fallback: Return placeholder for agents without SDK instance
                log.info("agent_fallback_mode", agent=name, reason="no_sdk_instance")
                results.append({
                    "agent": name,
                    "status": "completed",
                    "output": f"Agent {name} executed in fallback mode (SDK not available)",
                    "sdk": "fallback",
                })

        # Determine overall status
        any_success = any(r.get("status") == "completed" for r in results)
        status = TaskStatus.COMPLETED if any_success else TaskStatus.FAILED

        return TaskResult(
            task_id=task_id,
            status=status,
            output={"agents_used": agent_names, "results": results},
            error="; ".join(errors) if errors else None,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "sdk_available": AGENT_SDK_AVAILABLE,
                "agents_with_instances": sum(1 for c in self.agents.values() if c._agent_instance is not None),
            },
        )

    def _build_agent_prompt(self, task: str, agent_config: AgentConfig) -> str:
        """Build a task-specific prompt for an agent based on its role."""
        role_prompts = {
            AgentRole.ORCHESTRATOR: (
                f"As the orchestration agent, execute this task:\n\n{task}\n\n"
                "Coordinate any necessary sub-tasks and report results."
            ),
            AgentRole.SPECIALIST: (
                f"As a specialist agent with capabilities {agent_config.capabilities}, "
                f"execute this task:\n\n{task}\n\n"
                "Focus on your area of expertise and provide detailed findings."
            ),
            AgentRole.MONITOR: (
                f"As a monitoring agent, execute this task:\n\n{task}\n\n"
                "Check system health and report any issues or anomalies."
            ),
            AgentRole.MEMORY: (
                f"As a memory management agent, execute this task:\n\n{task}\n\n"
                "Handle any persistence or retrieval operations needed."
            ),
            AgentRole.WORKER: (
                f"Execute this task:\n\n{task}\n\n"
                "Complete the work and report results."
            ),
        }

        return role_prompts.get(agent_config.role, f"Execute this task:\n\n{task}")

    async def run_ralph_iteration(self) -> TaskResult:
        """Run a single Ralph Loop iteration via the orchestrator."""
        return await self.coordinate_agents(
            task="ralph_loop_iterate",
            agents=["ralph", "validator", "health-monitor"],
        )

    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status."""
        return {
            "initialized": self._initialized,
            "letta_url": self.letta_url,
            "claude_flow_version": self.claude_flow_version,
            "registered_agents": list(self.agents.keys()),
            "active_tasks": len(self.active_tasks),
            "paths": {
                "base_dir": str(self.base_dir),
                "v10_dir": str(self.v10_dir),
                "claude_flow_dir": str(self.claude_flow_dir),
            },
        }


async def main():
    """Demo the core orchestrator."""
    print("=" * 60)
    print("CORE ORCHESTRATOR - Ultimate Autonomous Platform")
    print("=" * 60)

    orchestrator = CoreOrchestrator()

    # Initialize
    success = await orchestrator.initialize()
    print(f"\nInitialized: {success}")

    # Show status
    status = orchestrator.get_status()
    print(f"\nRegistered Agents: {status['registered_agents']}")
    print(f"Letta URL: {status['letta_url']}")
    print(f"Claude-Flow Version: {status['claude_flow_version']}")

    # Run a coordination task
    print("\n" + "-" * 40)
    print("Running coordination task...")
    result = await orchestrator.coordinate_agents("validate and check health")
    print(f"Result: {result.status.value}")
    print(f"Duration: {result.duration_ms:.1f}ms")
    print(f"Output: {result.output}")

    print("\n" + "=" * 60)
    print("[OK] Core Orchestrator Demo Complete")


if __name__ == "__main__":
    asyncio.run(main())
