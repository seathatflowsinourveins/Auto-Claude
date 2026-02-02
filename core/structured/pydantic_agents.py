#!/usr/bin/env python3
"""
Pydantic AI Agents - Agent Framework with Memory Integration
Part of the V33 Structured Output Layer.

Uses pydantic-ai for building agents with built-in Pydantic validation,
dependency injection, and memory integration with the V33 memory layer.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any, Optional, TypeVar, Generic, Callable, Awaitable
from datetime import datetime
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field

try:
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.openai import OpenAIModel
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    Agent = None
    RunContext = None
    AnthropicModel = None
    OpenAIModel = None


# Type variables
T = TypeVar("T", bound=BaseModel)
DepsT = TypeVar("DepsT")


# ============================================================================
# Agent Configuration
# ============================================================================

class AgentProvider(str, Enum):
    """Supported AI providers for agents."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class AgentRole(str, Enum):
    """Pre-defined agent roles."""
    ASSISTANT = "assistant"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    CODER = "coder"
    REVIEWER = "reviewer"
    PLANNER = "planner"


@dataclass
class AgentConfig:
    """Configuration for a Pydantic AI agent."""
    name: str
    role: AgentRole = AgentRole.ASSISTANT
    provider: AgentProvider = AgentProvider.ANTHROPIC
    model: str = "claude-sonnet-4-20250514"
    system_prompt: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    retries: int = 3

    # Memory integration
    enable_memory: bool = True
    memory_namespace: str = "agent"

    # Tool configuration
    tools: list[Callable] = field(default_factory=list)


# ============================================================================
# Agent Dependencies
# ============================================================================

@dataclass
class AgentDeps:
    """
    Dependencies injected into agent runs.

    Provides access to memory, tools, and session context.
    """
    session_id: str = ""
    user_id: str = ""
    memory: Optional[Any] = None  # V33 UnifiedMemory instance
    tools: Optional[Any] = None   # V33 ToolLayer instance
    context: dict[str, Any] = field(default_factory=dict)

    async def store_memory(
        self,
        content: str,
        metadata: Optional[dict] = None,
        importance: float = 0.5,
    ) -> Optional[str]:
        """Store content in memory if available."""
        if self.memory:
            try:
                entry = await self.memory.store(
                    content=content,
                    metadata=metadata or {},
                    importance=importance,
                )
                return entry.id if entry else None
            except Exception:
                return None
        return None

    async def search_memory(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """Search memory if available."""
        if self.memory:
            try:
                results = await self.memory.search(query, limit=limit)
                return [
                    {"content": r.content, "score": r.score, "metadata": r.metadata}
                    for r in results
                ]
            except Exception:
                return []
        return []


# ============================================================================
# Agent Result Models
# ============================================================================

class AgentMessage(BaseModel):
    """A single message in agent conversation."""
    role: str = Field(description="Message role (user, assistant, system)")
    content: str = Field(description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRun(BaseModel):
    """Result of an agent run."""
    success: bool
    output: Any = None
    messages: list[AgentMessage] = Field(default_factory=list)
    error: Optional[str] = None
    tokens_used: int = 0
    latency_ms: float = 0.0
    memories_stored: int = 0
    memories_retrieved: int = 0


class ResearchOutput(BaseModel):
    """Output for research agent."""
    findings: list[str] = Field(description="Key research findings")
    sources: list[str] = Field(default_factory=list, description="Sources consulted")
    summary: str = Field(description="Executive summary")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in findings")


class AnalysisOutput(BaseModel):
    """Output for analysis agent."""
    insights: list[str] = Field(description="Key insights")
    data_points: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)


class PlanOutput(BaseModel):
    """Output for planning agent."""
    goal: str = Field(description="The planning goal")
    steps: list[str] = Field(description="Ordered steps to achieve goal")
    dependencies: dict[str, list[str]] = Field(default_factory=dict)
    estimated_duration: str = Field(default="", description="Estimated total duration")
    risks: list[str] = Field(default_factory=list)


class CodeOutput(BaseModel):
    """Output for coding agent."""
    code: str = Field(description="Generated code")
    language: str = Field(description="Programming language")
    explanation: str = Field(description="Code explanation")
    tests: list[str] = Field(default_factory=list, description="Suggested tests")


# ============================================================================
# Pydantic AI Agent Wrapper
# ============================================================================

class PydanticAIAgent:
    """
    Wrapper for pydantic-ai agents with V33 integration.

    Provides memory integration, dependency injection, and
    structured output validation.
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize the agent.

        Args:
            config: Agent configuration
        """
        self.config = config
        self._agent = None
        self._deps: Optional[AgentDeps] = None

        if PYDANTIC_AI_AVAILABLE:
            self._init_agent()

    def _init_agent(self):
        """Initialize the pydantic-ai agent."""
        # pydantic-ai models use env vars automatically
        if self.config.provider == AgentProvider.ANTHROPIC:
            model = AnthropicModel(self.config.model)
        else:
            model = OpenAIModel(self.config.model)

        system_prompt = self.config.system_prompt or self._default_system_prompt()

        self._agent = Agent(
            model=model,
            system_prompt=system_prompt,
            retries=self.config.retries,
            deps_type=AgentDeps,
        )

        # Register tools
        for tool in self.config.tools:
            self._agent.tool(tool)

    def _default_system_prompt(self) -> str:
        """Get default system prompt based on role."""
        prompts = {
            AgentRole.ASSISTANT: "You are a helpful assistant. Provide clear, accurate responses.",
            AgentRole.RESEARCHER: "You are a research assistant. Find and synthesize information thoroughly.",
            AgentRole.ANALYST: "You are a data analyst. Analyze information and provide actionable insights.",
            AgentRole.CODER: "You are a coding assistant. Write clean, well-documented code.",
            AgentRole.REVIEWER: "You are a code reviewer. Identify issues and suggest improvements.",
            AgentRole.PLANNER: "You are a planning assistant. Create detailed, actionable plans.",
        }
        return prompts.get(self.config.role, prompts[AgentRole.ASSISTANT])

    def attach_memory(self, memory: Any) -> None:
        """Attach V33 UnifiedMemory instance."""
        if self._deps is None:
            self._deps = AgentDeps()
        self._deps.memory = memory

    def attach_tools(self, tools: Any) -> None:
        """Attach V33 ToolLayer instance."""
        if self._deps is None:
            self._deps = AgentDeps()
        self._deps.tools = tools

    async def run(
        self,
        prompt: str,
        response_model: Optional[type] = None,
        context: Optional[dict] = None,
        **kwargs: Any,
    ) -> AgentRun:
        """
        Run the agent with the given prompt.

        Args:
            prompt: User prompt
            response_model: Optional Pydantic model for structured output
            context: Additional context for the run
            **kwargs: Additional run parameters

        Returns:
            AgentRun with the result
        """
        start_time = datetime.now()
        memories_stored = 0
        memories_retrieved = 0

        try:
            # Prepare dependencies
            deps = self._deps or AgentDeps()
            if context:
                deps.context.update(context)

            # Search memory for relevant context
            if deps.memory and self.config.enable_memory:
                memories = await deps.search_memory(prompt, limit=3)
                memories_retrieved = len(memories)
                if memories:
                    memory_context = "\n".join([m["content"] for m in memories])
                    prompt = f"Relevant context:\n{memory_context}\n\nUser query: {prompt}"

            if not PYDANTIC_AI_AVAILABLE or self._agent is None:
                # Fallback without pydantic-ai
                return await self._fallback_run(prompt, response_model, deps)

            # Run agent
            if response_model:
                result = await self._agent.run(
                    prompt,
                    deps=deps,
                    result_type=response_model,
                    **kwargs,
                )
            else:
                result = await self._agent.run(prompt, deps=deps, **kwargs)

            # Store result in memory
            if deps.memory and self.config.enable_memory:
                await deps.store_memory(
                    content=f"Agent {self.config.name} response: {str(result.data)[:500]}",
                    metadata={"agent": self.config.name, "role": self.config.role.value},
                    importance=0.6,
                )
                memories_stored = 1

            latency = (datetime.now() - start_time).total_seconds() * 1000

            return AgentRun(
                success=True,
                output=result.data,
                messages=[
                    AgentMessage(role="user", content=prompt),
                    AgentMessage(role="assistant", content=str(result.data)),
                ],
                latency_ms=latency,
                memories_stored=memories_stored,
                memories_retrieved=memories_retrieved,
            )

        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds() * 1000
            return AgentRun(
                success=False,
                error=str(e),
                latency_ms=latency,
                memories_retrieved=memories_retrieved,
            )

    async def _fallback_run(
        self,
        prompt: str,
        response_model: Optional[type],
        deps: AgentDeps,
    ) -> AgentRun:
        """Fallback when pydantic-ai not available."""
        try:
            from .instructor_chains import InstructorClient

            client = InstructorClient(
                provider="anthropic" if self.config.provider == AgentProvider.ANTHROPIC else "openai",
                model=self.config.model,
            )

            if response_model:
                result = await client.extract(
                    text=prompt,
                    response_model=response_model,
                    system_prompt=self._default_system_prompt(),
                )

                if result.success:
                    return AgentRun(
                        success=True,
                        output=result.data,
                        latency_ms=result.latency_ms,
                    )
                else:
                    return AgentRun(
                        success=False,
                        error=result.error,
                        latency_ms=result.latency_ms,
                    )
            else:
                # Simple text response
                return AgentRun(
                    success=False,
                    error="pydantic-ai not available for unstructured responses",
                )

        except Exception as e:
            return AgentRun(
                success=False,
                error=f"Fallback failed: {str(e)}",
            )


# ============================================================================
# Pre-built Agent Types
# ============================================================================

def create_research_agent(
    name: str = "researcher",
    memory: Optional[Any] = None,
    **kwargs: Any,
) -> PydanticAIAgent:
    """Create a research-focused agent."""
    config = AgentConfig(
        name=name,
        role=AgentRole.RESEARCHER,
        system_prompt="""You are an expert researcher. Your task is to:
1. Thoroughly investigate the given topic
2. Identify key findings and patterns
3. Cite sources when available
4. Provide a confidence assessment
5. Summarize findings concisely""",
        **kwargs,
    )
    agent = PydanticAIAgent(config)
    if memory:
        agent.attach_memory(memory)
    return agent


def create_analyst_agent(
    name: str = "analyst",
    memory: Optional[Any] = None,
    **kwargs: Any,
) -> PydanticAIAgent:
    """Create an analysis-focused agent."""
    config = AgentConfig(
        name=name,
        role=AgentRole.ANALYST,
        system_prompt="""You are a data analyst. Your task is to:
1. Analyze the provided information objectively
2. Identify key insights and patterns
3. Quantify findings where possible
4. Provide actionable recommendations
5. Flag potential risks""",
        **kwargs,
    )
    agent = PydanticAIAgent(config)
    if memory:
        agent.attach_memory(memory)
    return agent


def create_planner_agent(
    name: str = "planner",
    memory: Optional[Any] = None,
    **kwargs: Any,
) -> PydanticAIAgent:
    """Create a planning-focused agent."""
    config = AgentConfig(
        name=name,
        role=AgentRole.PLANNER,
        system_prompt="""You are a planning expert. Your task is to:
1. Understand the goal clearly
2. Break it into achievable steps
3. Identify dependencies between steps
4. Estimate effort and duration
5. Anticipate potential risks""",
        **kwargs,
    )
    agent = PydanticAIAgent(config)
    if memory:
        agent.attach_memory(memory)
    return agent


def create_coder_agent(
    name: str = "coder",
    memory: Optional[Any] = None,
    **kwargs: Any,
) -> PydanticAIAgent:
    """Create a coding-focused agent."""
    config = AgentConfig(
        name=name,
        role=AgentRole.CODER,
        system_prompt="""You are an expert programmer. Your task is to:
1. Write clean, efficient code
2. Follow best practices and patterns
3. Include appropriate documentation
4. Consider edge cases
5. Suggest tests for the code""",
        temperature=0.3,  # Lower temperature for code
        **kwargs,
    )
    agent = PydanticAIAgent(config)
    if memory:
        agent.attach_memory(memory)
    return agent


# ============================================================================
# Convenience Functions
# ============================================================================

def create_agent(
    name: str,
    role: AgentRole = AgentRole.ASSISTANT,
    memory: Optional[Any] = None,
    tools: Optional[Any] = None,
    **kwargs: Any,
) -> PydanticAIAgent:
    """
    Factory function to create a PydanticAIAgent.

    Args:
        name: Agent name
        role: Agent role
        memory: Optional V33 memory instance
        tools: Optional V33 tools instance
        **kwargs: Additional configuration

    Returns:
        Configured PydanticAIAgent instance
    """
    config = AgentConfig(name=name, role=role, **kwargs)
    agent = PydanticAIAgent(config)

    if memory:
        agent.attach_memory(memory)
    if tools:
        agent.attach_tools(tools)

    return agent


# Export availability
__all__ = [
    "PydanticAIAgent",
    "AgentConfig",
    "AgentDeps",
    "AgentRole",
    "AgentProvider",
    "AgentRun",
    "AgentMessage",
    "ResearchOutput",
    "AnalysisOutput",
    "PlanOutput",
    "CodeOutput",
    "create_agent",
    "create_research_agent",
    "create_analyst_agent",
    "create_planner_agent",
    "create_coder_agent",
    "PYDANTIC_AI_AVAILABLE",
]
