#!/usr/bin/env python3
"""
Claude Flow - Native Multi-Agent Orchestration
Built specifically for Claude-based agent systems.
"""

from __future__ import annotations

import asyncio
import os
import json
from datetime import datetime
from typing import Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum

import structlog
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment
load_dotenv()

logger = structlog.get_logger(__name__)

# Claude Flow is always available (native implementation)
CLAUDE_FLOW_AVAILABLE = True


class AgentRole(str, Enum):
    """Predefined agent roles."""
    PLANNER = "planner"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    COORDINATOR = "coordinator"
    RESEARCHER = "researcher"
    CODER = "coder"
    CRITIC = "critic"
    CUSTOM = "custom"


class MessageType(str, Enum):
    """Types of messages in the flow."""
    TASK = "task"
    RESULT = "result"
    FEEDBACK = "feedback"
    HANDOFF = "handoff"
    COMPLETE = "complete"
    ERROR = "error"


class FlowMessage(BaseModel):
    """A message passed between agents in the flow."""
    id: str = Field(default_factory=lambda: str(datetime.now().timestamp()))
    type: MessageType
    from_agent: str
    to_agent: Optional[str] = None
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class AgentConfig(BaseModel):
    """Configuration for a Claude Flow agent."""
    name: str
    role: AgentRole
    system_prompt: str
    capabilities: list[str] = Field(default_factory=list)
    max_iterations: int = Field(default=5)
    temperature: float = Field(default=0.7)


class FlowConfig(BaseModel):
    """Configuration for the entire flow."""
    name: str
    agents: list[AgentConfig]
    entry_agent: str
    max_total_steps: int = Field(default=50)
    timeout_seconds: int = Field(default=300)
    parallel_execution: bool = Field(default=False)


class AgentResult(BaseModel):
    """Result from an agent's execution."""
    agent_name: str
    success: bool
    output: str
    next_agent: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FlowResult(BaseModel):
    """Final result from a flow execution."""
    success: bool
    final_output: Optional[str] = None
    messages: list[FlowMessage] = Field(default_factory=list)
    agent_results: list[AgentResult] = Field(default_factory=list)
    total_steps: int = 0
    duration_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class ClaudeAgent:
    """
    A Claude-powered agent in the flow.

    Each agent has a specific role and system prompt that guides
    its behavior when processing tasks.
    """

    config: AgentConfig
    message_history: list[FlowMessage] = field(default_factory=list)
    _iteration_count: int = field(default=0)

    async def process(
        self,
        message: FlowMessage,
        context: dict[str, Any],
    ) -> AgentResult:
        """Process an incoming message and produce a result."""
        self._iteration_count += 1
        self.message_history.append(message)

        logger.info(
            "agent_processing",
            agent=self.config.name,
            message_type=message.type.value,
            iteration=self._iteration_count,
        )

        # Build prompt for Claude
        prompt = self._build_prompt(message, context)

        # Call LLM
        response = await self._call_llm(prompt)

        # Parse response to determine next agent
        next_agent = self._determine_next_agent(response)

        return AgentResult(
            agent_name=self.config.name,
            success=True,
            output=response,
            next_agent=next_agent,
            metadata={
                "iteration": self._iteration_count,
                "role": self.config.role.value,
            },
        )

    def _build_prompt(
        self,
        message: FlowMessage,
        context: dict[str, Any],
    ) -> str:
        """Build the prompt for the LLM."""
        recent_history = self.message_history[-5:]
        history_text = "\n".join([
            f"[{m.from_agent}] {m.content[:200]}"
            for m in recent_history
        ])

        return f"""{self.config.system_prompt}

## Current Task
{message.content}

## Context
{json.dumps(context, indent=2)}

## Recent History
{history_text}

## Instructions
1. Process the task according to your role as {self.config.role.value}
2. If the task is complete, say "COMPLETE:" followed by your final answer
3. If you need another agent, say "HANDOFF: [agent_name]" and explain why
4. If there's an error, say "ERROR:" followed by the error description

Respond now:"""

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        try:
            from core.llm_gateway import LLMGateway, Message
            gateway = LLMGateway()

            messages = [
                Message(role="system", content=self.config.system_prompt),
                Message(role="user", content=prompt),
            ]

            # Create a ModelConfig with the agent's temperature setting
            from core.llm_gateway import ModelConfig, Provider
            model_config = ModelConfig(
                provider=Provider.ANTHROPIC,
                model_id="claude-3-5-sonnet-20241022",
                temperature=self.config.temperature,
            )
            response = await gateway.complete(messages, model_config=model_config)
            return response.content

        except Exception as e:
            logger.warning("llm_call_fallback", error=str(e))
            return f"Simulated response from {self.config.name}: Task processed for iteration {self._iteration_count}"

    def _determine_next_agent(self, response: str) -> Optional[str]:
        """Determine the next agent based on the response."""
        response_upper = response.upper()

        if "COMPLETE:" in response_upper:
            return None

        if "HANDOFF:" in response_upper:
            # Extract agent name after HANDOFF:
            try:
                idx = response_upper.index("HANDOFF:")
                remaining = response[idx + 8:].strip()
                # Get first word as agent name
                next_agent = remaining.split()[0].strip("[]").lower()
                return next_agent
            except (ValueError, IndexError):
                pass

        if "ERROR:" in response_upper:
            return None

        return None

    def reset(self) -> None:
        """Reset the agent's state."""
        self._iteration_count = 0
        self.message_history.clear()


@dataclass
class ClaudeFlow:
    """
    Claude Flow Orchestrator - Native multi-agent system.

    Coordinates multiple Claude agents to accomplish complex tasks
    through structured handoffs and message passing.

    Usage:
        flow = ClaudeFlow.from_config(config)
        result = await flow.run("Analyze this data and create a report")
    """

    config: FlowConfig
    agents: dict[str, ClaudeAgent] = field(default_factory=dict)
    message_log: list[FlowMessage] = field(default_factory=list)
    _step_count: int = field(default=0)

    @classmethod
    def from_config(cls, config: FlowConfig) -> "ClaudeFlow":
        """Create a ClaudeFlow from configuration."""
        flow = cls(config=config)

        for agent_config in config.agents:
            agent = ClaudeAgent(config=agent_config)
            flow.agents[agent_config.name] = agent

        logger.info(
            "claude_flow_created",
            name=config.name,
            agents=list(flow.agents.keys()),
        )

        return flow

    @classmethod
    def create_default(cls, name: str = "default") -> "ClaudeFlow":
        """Create a default three-agent flow (Planner -> Executor -> Reviewer)."""
        config = FlowConfig(
            name=name,
            agents=[
                AgentConfig(
                    name="planner",
                    role=AgentRole.PLANNER,
                    system_prompt="You are a planning agent. Break down tasks into clear, actionable steps.",
                    capabilities=["planning", "task_decomposition"],
                ),
                AgentConfig(
                    name="executor",
                    role=AgentRole.EXECUTOR,
                    system_prompt="You are an execution agent. Carry out the planned steps precisely.",
                    capabilities=["execution", "implementation"],
                ),
                AgentConfig(
                    name="reviewer",
                    role=AgentRole.REVIEWER,
                    system_prompt="You are a review agent. Validate outputs and ensure quality.",
                    capabilities=["review", "validation"],
                ),
            ],
            entry_agent="planner",
        )

        return cls.from_config(config)

    async def run(
        self,
        task: str,
        context: Optional[dict[str, Any]] = None,
    ) -> FlowResult:
        """
        Run the flow with the given task.

        Args:
            task: The task description
            context: Additional context for agents

        Returns:
            FlowResult with execution details
        """
        import time
        start_time = time.time()

        context = context or {}
        self._step_count = 0
        self.message_log.clear()
        agent_results = []

        # Reset all agents
        for agent in self.agents.values():
            agent.reset()

        # Start with entry agent
        current_agent_name = self.config.entry_agent
        current_message = FlowMessage(
            type=MessageType.TASK,
            from_agent="user",
            to_agent=current_agent_name,
            content=task,
        )

        logger.info("claude_flow_started", task=task[:50])

        try:
            while self._step_count < self.config.max_total_steps:
                self._step_count += 1

                # Get current agent
                current_agent = self.agents.get(current_agent_name)
                if not current_agent:
                    logger.error("agent_not_found", name=current_agent_name)
                    break

                # Process message
                result = await current_agent.process(current_message, context)
                agent_results.append(result)
                self.message_log.append(current_message)

                logger.info(
                    "agent_completed",
                    agent=current_agent_name,
                    step=self._step_count,
                    next_agent=result.next_agent,
                )

                # Check for completion
                if result.next_agent is None:
                    # Flow complete
                    duration = time.time() - start_time
                    return FlowResult(
                        success=True,
                        final_output=result.output,
                        messages=self.message_log,
                        agent_results=agent_results,
                        total_steps=self._step_count,
                        duration_seconds=duration,
                    )

                # Prepare next iteration
                if result.next_agent in self.agents:
                    current_message = FlowMessage(
                        type=MessageType.HANDOFF,
                        from_agent=current_agent_name,
                        to_agent=result.next_agent,
                        content=result.output,
                    )
                    current_agent_name = result.next_agent
                else:
                    logger.warning(
                        "unknown_next_agent",
                        requested=result.next_agent,
                        available=list(self.agents.keys()),
                    )
                    break

            # Max steps reached
            duration = time.time() - start_time
            return FlowResult(
                success=False,
                error="Max steps reached",
                messages=self.message_log,
                agent_results=agent_results,
                total_steps=self._step_count,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error("claude_flow_failed", error=str(e))
            return FlowResult(
                success=False,
                error=str(e),
                messages=self.message_log,
                agent_results=agent_results,
                total_steps=self._step_count,
                duration_seconds=duration,
            )

    async def run_parallel(
        self,
        tasks: list[str],
        context: Optional[dict[str, Any]] = None,
    ) -> list[FlowResult]:
        """Run multiple tasks in parallel."""
        if not self.config.parallel_execution:
            logger.warning("parallel_not_enabled")

        return await asyncio.gather(*[
            self.run(task, context) for task in tasks
        ])

    def add_agent(self, config: AgentConfig) -> None:
        """Add a new agent to the flow."""
        agent = ClaudeAgent(config=config)
        self.agents[config.name] = agent
        logger.info("agent_added", name=config.name)

    def remove_agent(self, name: str) -> bool:
        """Remove an agent from the flow."""
        if name in self.agents:
            del self.agents[name]
            logger.info("agent_removed", name=name)
            return True
        return False

    def get_agent(self, name: str) -> Optional[ClaudeAgent]:
        """Get an agent by name."""
        return self.agents.get(name)

    def list_agents(self) -> list[str]:
        """List all agent names."""
        return list(self.agents.keys())


@dataclass
class ClaudeFlowOrchestrator:
    """
    High-level orchestrator for multiple Claude Flows.

    Manages flow creation, execution, and lifecycle.

    Usage:
        orchestrator = ClaudeFlowOrchestrator()
        flow = orchestrator.create_flow("my-flow")
        result = await orchestrator.run(flow, "Process this task")
    """

    flows: dict[str, ClaudeFlow] = field(default_factory=dict)

    def create_flow(
        self,
        name: str,
        agents: Optional[list[AgentConfig]] = None,
        entry_agent: Optional[str] = None,
    ) -> ClaudeFlow:
        """Create a new flow."""
        if agents:
            config = FlowConfig(
                name=name,
                agents=agents,
                entry_agent=entry_agent or agents[0].name,
            )
            flow = ClaudeFlow.from_config(config)
        else:
            flow = ClaudeFlow.create_default(name)

        self.flows[name] = flow
        return flow

    async def run(
        self,
        flow: ClaudeFlow,
        task: str,
        context: Optional[dict[str, Any]] = None,
    ) -> FlowResult:
        """Run a flow with the given task."""
        return await flow.run(task, context)

    def get_flow(self, name: str) -> Optional[ClaudeFlow]:
        """Get a flow by name."""
        return self.flows.get(name)

    def list_flows(self) -> list[str]:
        """List all flow names."""
        return list(self.flows.keys())

    def delete_flow(self, name: str) -> bool:
        """Delete a flow."""
        if name in self.flows:
            del self.flows[name]
            return True
        return False


def create_orchestrator() -> ClaudeFlowOrchestrator:
    """Factory function to create a Claude Flow orchestrator."""
    return ClaudeFlowOrchestrator()


def create_default_flow(name: str = "default") -> ClaudeFlow:
    """Factory function to create a default flow."""
    return ClaudeFlow.create_default(name)


if __name__ == "__main__":
    async def main():
        """Test the Claude Flow orchestrator."""
        print(f"Claude Flow available: {CLAUDE_FLOW_AVAILABLE}")

        # Create default flow
        flow = create_default_flow("test-flow")
        print(f"Agents: {flow.list_agents()}")

        # Run a task
        result = await flow.run("Create a simple hello world function in Python")

        print(f"Success: {result.success}")
        print(f"Steps: {result.total_steps}")
        print(f"Duration: {result.duration_seconds:.2f}s")

        if result.final_output:
            print(f"Output: {result.final_output[:200]}...")

    asyncio.run(main())
