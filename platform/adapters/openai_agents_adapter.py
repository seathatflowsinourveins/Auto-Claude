"""
OpenAI Agents SDK Adapter - V36 Architecture

Integrates OpenAI's Agents SDK (Swarm) for multi-agent orchestration.

SDK: openai-swarm (https://github.com/openai/swarm)
Layer: L1 (Orchestration)
Features:
- Lightweight multi-agent orchestration
- Handoff between agents
- Function calling with automatic execution
- Context management across agents

API Patterns (verified from openai-swarm 0.1.0):
- Swarm() → client for agent orchestration
- Agent(name, instructions, functions, tool_choice) → define agents
- client.run(agent, messages, context_variables) → execute agent
- Handoff patterns via function returns

Usage:
    from adapters.openai_agents_adapter import OpenAIAgentsAdapter

    adapter = OpenAIAgentsAdapter()
    await adapter.initialize({})

    result = await adapter.execute("run_agent", agent_name="coder", messages=[...])
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# SDK availability check
OPENAI_AGENTS_AVAILABLE = False
Swarm = None
Agent = None

try:
    from swarm import Swarm, Agent
    OPENAI_AGENTS_AVAILABLE = True
except ImportError:
    try:
        # Alternative import path
        from openai_swarm import Swarm, Agent
        OPENAI_AGENTS_AVAILABLE = True
    except ImportError:
        logger.info("OpenAI Swarm not installed - install with: pip install openai-swarm")


# Import base adapter interface
try:
    from core.orchestration.base import (
        SDKAdapter,
        SDKLayer,
        AdapterConfig,
        AdapterResult,
        AdapterStatus,
    )
except ImportError:
    # Fallback for standalone usage
    from dataclasses import dataclass as _dataclass
    from enum import IntEnum
    from abc import ABC, abstractmethod

    class SDKLayer(IntEnum):
        ORCHESTRATION = 1

    @_dataclass
    class AdapterResult:
        success: bool
        data: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        latency_ms: float = 0.0
        cached: bool = False

    @_dataclass
    class AdapterConfig:
        name: str = "openai-agents"
        layer: int = 1

    class AdapterStatus:
        READY = "ready"
        FAILED = "failed"
        UNINITIALIZED = "uninitialized"

    class SDKAdapter(ABC):
        @property
        @abstractmethod
        def sdk_name(self) -> str: ...
        @property
        @abstractmethod
        def layer(self) -> int: ...
        @property
        @abstractmethod
        def available(self) -> bool: ...
        @abstractmethod
        async def initialize(self, config: Dict) -> AdapterResult: ...
        @abstractmethod
        async def execute(self, operation: str, **kwargs) -> AdapterResult: ...
        @abstractmethod
        async def health_check(self) -> AdapterResult: ...
        @abstractmethod
        async def shutdown(self) -> AdapterResult: ...


@dataclass
class AgentDefinition:
    """Definition for a Swarm agent."""
    name: str
    instructions: str
    functions: List[Callable] = field(default_factory=list)
    tool_choice: Optional[str] = None
    model: str = "gpt-4o-mini"


class OpenAIAgentsAdapter(SDKAdapter):
    """
    OpenAI Agents SDK (Swarm) adapter for multi-agent orchestration.

    Provides lightweight agent coordination with:
    - Named agents with specific instructions
    - Inter-agent handoffs
    - Function calling with automatic execution
    - Context variable propagation

    Example:
        adapter = OpenAIAgentsAdapter()
        await adapter.initialize({"api_key": "sk-..."})

        # Define and run agents
        result = await adapter.execute("run_agent",
            agent_name="researcher",
            messages=[{"role": "user", "content": "Find info about X"}],
            context_variables={"topic": "AI"}
        )
    """

    def __init__(self, config: Optional[AdapterConfig] = None):
        self._config = config or AdapterConfig(
            name="openai-agents",
            layer=SDKLayer.ORCHESTRATION
        )
        self._status = AdapterStatus.UNINITIALIZED
        self._client: Optional[Any] = None
        self._agents: Dict[str, Any] = {}
        self._call_count = 0
        self._total_latency_ms = 0.0
        self._error_count = 0

    @property
    def sdk_name(self) -> str:
        return "openai-agents"

    @property
    def layer(self) -> SDKLayer:
        return SDKLayer.ORCHESTRATION

    @property
    def available(self) -> bool:
        return OPENAI_AGENTS_AVAILABLE

    @property
    def status(self) -> AdapterStatus:
        return self._status

    async def initialize(self, config: Dict[str, Any]) -> AdapterResult:
        """Initialize the Swarm client."""
        if not OPENAI_AGENTS_AVAILABLE:
            return AdapterResult(
                success=False,
                error="OpenAI Swarm SDK not installed. Install with: pip install openai-swarm"
            )

        try:
            import os

            # Get API key from config or environment
            api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return AdapterResult(
                    success=False,
                    error="OPENAI_API_KEY not provided"
                )

            # Initialize Swarm client
            from openai import OpenAI
            openai_client = OpenAI(api_key=api_key)
            self._client = Swarm(client=openai_client)

            # Register default agents
            await self._register_default_agents(config)

            self._status = AdapterStatus.READY
            logger.info("OpenAI Agents SDK initialized successfully")

            return AdapterResult(
                success=True,
                data={"agents_registered": list(self._agents.keys())}
            )

        except Exception as e:
            self._status = AdapterStatus.FAILED
            logger.error(f"OpenAI Agents initialization failed: {e}")
            return AdapterResult(success=False, error=str(e))

    async def _register_default_agents(self, config: Dict[str, Any]) -> None:
        """Register default agents for common use cases."""
        model = config.get("model", "gpt-4o-mini")

        # Researcher agent
        self._agents["researcher"] = Agent(
            name="Researcher",
            instructions="""You are a research specialist. Your role is to:
            1. Find relevant information on the given topic
            2. Synthesize findings into clear summaries
            3. Cite sources when possible
            Be thorough but concise.""",
            model=model
        )

        # Coder agent
        self._agents["coder"] = Agent(
            name="Coder",
            instructions="""You are a coding specialist. Your role is to:
            1. Write clean, efficient code
            2. Follow best practices and patterns
            3. Include comments for complex logic
            4. Consider edge cases and error handling
            Output code in markdown code blocks.""",
            model=model
        )

        # Reviewer agent
        self._agents["reviewer"] = Agent(
            name="Reviewer",
            instructions="""You are a code review specialist. Your role is to:
            1. Identify bugs and potential issues
            2. Suggest improvements for performance
            3. Check for security vulnerabilities
            4. Ensure code follows best practices
            Be constructive and specific in feedback.""",
            model=model
        )

        # Coordinator agent
        self._agents["coordinator"] = Agent(
            name="Coordinator",
            instructions="""You are a task coordinator. Your role is to:
            1. Break down complex tasks into subtasks
            2. Delegate to appropriate specialists
            3. Synthesize results from multiple agents
            4. Ensure quality and completeness
            Use handoff functions to delegate work.""",
            model=model
        )

    async def execute(self, operation: str, **kwargs) -> AdapterResult:
        """Execute an operation using the Swarm client."""
        if not self._client:
            return AdapterResult(success=False, error="Adapter not initialized")

        start_time = time.time()

        try:
            if operation == "run_agent":
                result = await self._run_agent(**kwargs)
            elif operation == "register_agent":
                result = await self._register_agent(**kwargs)
            elif operation == "list_agents":
                result = await self._list_agents()
            elif operation == "handoff":
                result = await self._handoff(**kwargs)
            else:
                result = AdapterResult(
                    success=False,
                    error=f"Unknown operation: {operation}"
                )

            latency_ms = (time.time() - start_time) * 1000
            self._call_count += 1
            self._total_latency_ms += latency_ms
            result.latency_ms = latency_ms

            if not result.success:
                self._error_count += 1

            return result

        except Exception as e:
            self._error_count += 1
            logger.error(f"OpenAI Agents execute error: {e}")
            return AdapterResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )

    async def _run_agent(
        self,
        agent_name: str,
        messages: List[Dict[str, str]],
        context_variables: Optional[Dict[str, Any]] = None,
        max_turns: int = 10,
        **kwargs
    ) -> AdapterResult:
        """Run an agent with the given messages."""
        if agent_name not in self._agents:
            return AdapterResult(
                success=False,
                error=f"Agent not found: {agent_name}. Available: {list(self._agents.keys())}"
            )

        agent = self._agents[agent_name]

        # Run in thread to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.run(
                agent=agent,
                messages=messages,
                context_variables=context_variables or {},
                max_turns=max_turns
            )
        )

        # Extract response data
        return AdapterResult(
            success=True,
            data={
                "messages": [m.__dict__ if hasattr(m, '__dict__') else m for m in response.messages],
                "agent": response.agent.name if response.agent else None,
                "context_variables": response.context_variables
            }
        )

    async def _register_agent(
        self,
        name: str,
        instructions: str,
        functions: Optional[List[Callable]] = None,
        model: str = "gpt-4o-mini",
        **kwargs
    ) -> AdapterResult:
        """Register a new agent."""
        try:
            agent = Agent(
                name=name,
                instructions=instructions,
                functions=functions or [],
                model=model
            )
            self._agents[name.lower()] = agent

            return AdapterResult(
                success=True,
                data={"agent_name": name, "registered": True}
            )
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _list_agents(self) -> AdapterResult:
        """List all registered agents."""
        agents_info = []
        for name, agent in self._agents.items():
            agents_info.append({
                "name": name,
                "full_name": agent.name,
                "model": getattr(agent, 'model', 'gpt-4o-mini'),
                "functions_count": len(agent.functions) if hasattr(agent, 'functions') else 0
            })

        return AdapterResult(
            success=True,
            data={"agents": agents_info, "count": len(agents_info)}
        )

    async def _handoff(
        self,
        from_agent: str,
        to_agent: str,
        messages: List[Dict[str, str]],
        context_variables: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AdapterResult:
        """Perform a handoff from one agent to another."""
        if to_agent not in self._agents:
            return AdapterResult(
                success=False,
                error=f"Target agent not found: {to_agent}"
            )

        # Run the target agent with the conversation context
        return await self._run_agent(
            agent_name=to_agent,
            messages=messages,
            context_variables={
                **(context_variables or {}),
                "handoff_from": from_agent
            }
        )

    async def health_check(self) -> AdapterResult:
        """Check adapter health."""
        if not OPENAI_AGENTS_AVAILABLE:
            return AdapterResult(success=False, error="SDK not available")

        if not self._client:
            return AdapterResult(success=False, error="Client not initialized")

        return AdapterResult(
            success=True,
            data={
                "status": "healthy",
                "agents_count": len(self._agents),
                "call_count": self._call_count,
                "avg_latency_ms": self._total_latency_ms / max(1, self._call_count)
            }
        )

    async def shutdown(self) -> AdapterResult:
        """Shutdown the adapter."""
        self._client = None
        self._agents.clear()
        self._status = AdapterStatus.UNINITIALIZED
        logger.info("OpenAI Agents adapter shutdown")
        return AdapterResult(success=True)


# Register with SDK registry if available
try:
    from core.orchestration.sdk_registry import register_adapter, SDKLayer

    @register_adapter("openai-agents", SDKLayer.ORCHESTRATION, priority=15)
    class RegisteredOpenAIAgentsAdapter(OpenAIAgentsAdapter):
        """Registered OpenAI Agents adapter."""
        pass

except ImportError:
    pass


__all__ = ["OpenAIAgentsAdapter", "OPENAI_AGENTS_AVAILABLE", "AgentDefinition"]
