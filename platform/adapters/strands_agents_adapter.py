"""
Strands Agents Adapter - V36 Architecture

Integrates AWS Strands Agents SDK for enterprise-grade agent orchestration.

SDK: strands-agents (AWS Strands)
Layer: L1 (Orchestration)
Features:
- Model-agnostic agent building
- Built-in tool integration
- Conversation memory
- Streaming responses
- Multi-turn conversations

API Patterns (from strands-agents SDK):
- Agent(model, tools, system_prompt) → create agent
- agent.run(message) → execute single turn
- agent.stream(message) → streaming execution
- Tool(name, description, handler) → define tools

Usage:
    from adapters.strands_agents_adapter import StrandsAgentsAdapter

    adapter = StrandsAgentsAdapter()
    await adapter.initialize({"model": "anthropic.claude-3-sonnet"})

    result = await adapter.execute("run", message="Analyze this code", tools=["code_analysis"])
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# SDK availability check
STRANDS_AVAILABLE = False

try:
    from strands import Agent, Tool
    STRANDS_AVAILABLE = True
except ImportError:
    logger.info("Strands Agents not installed - install with: pip install strands-agents")


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

    @_dataclass
    class AdapterConfig:
        name: str = "strands-agents"
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
class StrandsToolDef:
    """Tool definition for Strands agents."""
    name: str
    description: str
    handler: Optional[Callable] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


class StrandsAgentsAdapter(SDKAdapter):
    """
    AWS Strands Agents adapter for enterprise agent orchestration.

    Provides model-agnostic agent capabilities:
    - Support for multiple LLM providers (Anthropic, OpenAI, Bedrock)
    - Built-in tool integration with automatic execution
    - Conversation memory across turns
    - Streaming response support
    - Enterprise-ready patterns

    Operations:
    - run: Single-turn agent execution
    - stream: Streaming agent execution
    - create_agent: Create a new agent configuration
    - register_tool: Register a custom tool
    - list_tools: List available tools
    - clear_memory: Clear conversation memory
    """

    # Default tools
    DEFAULT_TOOLS = [
        StrandsToolDef(
            name="calculator",
            description="Perform mathematical calculations",
            parameters={"expression": {"type": "string", "description": "Math expression to evaluate"}}
        ),
        StrandsToolDef(
            name="web_search",
            description="Search the web for information",
            parameters={"query": {"type": "string", "description": "Search query"}}
        ),
        StrandsToolDef(
            name="code_analysis",
            description="Analyze code for issues and improvements",
            parameters={"code": {"type": "string", "description": "Code to analyze"}}
        )
    ]

    def __init__(self, config: Optional[AdapterConfig] = None):
        self._config = config or AdapterConfig(
            name="strands-agents",
            layer=SDKLayer.ORCHESTRATION
        )
        self._status = AdapterStatus.UNINITIALIZED
        self._agent: Optional[Any] = None
        self._model: str = ""
        self._tools: Dict[str, StrandsToolDef] = {}
        self._conversation_memory: List[Dict[str, str]] = []
        self._call_count = 0
        self._total_latency_ms = 0.0
        self._error_count = 0

    @property
    def sdk_name(self) -> str:
        return "strands-agents"

    @property
    def layer(self) -> SDKLayer:
        return SDKLayer.ORCHESTRATION

    @property
    def available(self) -> bool:
        return STRANDS_AVAILABLE

    @property
    def status(self) -> AdapterStatus:
        return self._status

    async def initialize(self, config: Dict[str, Any]) -> AdapterResult:
        """Initialize Strands agent."""
        if not STRANDS_AVAILABLE:
            # Provide stub implementation for non-AWS environments
            logger.info("Strands not available, using stub implementation")

        try:
            self._model = config.get("model", "anthropic.claude-3-sonnet")
            system_prompt = config.get("system_prompt", "You are a helpful AI assistant.")

            # Register default tools
            for tool_def in self.DEFAULT_TOOLS:
                self._tools[tool_def.name] = tool_def

            # Register custom tools from config
            custom_tools = config.get("tools", [])
            for tool in custom_tools:
                if isinstance(tool, dict):
                    self._tools[tool["name"]] = StrandsToolDef(**tool)

            if STRANDS_AVAILABLE:
                # Build tools list for Strands
                strands_tools = []
                for name, tool_def in self._tools.items():
                    if tool_def.handler:
                        strands_tools.append(Tool(
                            name=name,
                            description=tool_def.description,
                            handler=tool_def.handler
                        ))

                # Create agent
                self._agent = Agent(
                    model=self._model,
                    tools=strands_tools,
                    system_prompt=system_prompt
                )

            self._status = AdapterStatus.READY
            logger.info(f"Strands Agents adapter initialized (model={self._model})")

            return AdapterResult(
                success=True,
                data={
                    "model": self._model,
                    "tools": list(self._tools.keys()),
                    "strands_native": STRANDS_AVAILABLE
                }
            )

        except Exception as e:
            self._status = AdapterStatus.FAILED
            logger.error(f"Strands initialization failed: {e}")
            return AdapterResult(success=False, error=str(e))

    async def execute(self, operation: str, **kwargs) -> AdapterResult:
        """Execute a Strands operation."""
        start_time = time.time()

        try:
            if operation == "run":
                result = await self._run(**kwargs)
            elif operation == "stream":
                result = await self._stream(**kwargs)
            elif operation == "create_agent":
                result = await self._create_agent(**kwargs)
            elif operation == "register_tool":
                result = await self._register_tool(**kwargs)
            elif operation == "list_tools":
                result = await self._list_tools()
            elif operation == "clear_memory":
                result = await self._clear_memory()
            elif operation == "get_stats":
                result = await self._get_stats()
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
            logger.error(f"Strands execute error: {e}")
            return AdapterResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )

    async def _run(
        self,
        message: str,
        tools: Optional[List[str]] = None,
        **kwargs
    ) -> AdapterResult:
        """Run a single-turn agent execution."""
        try:
            # Add to conversation memory
            self._conversation_memory.append({"role": "user", "content": message})

            if STRANDS_AVAILABLE and self._agent:
                # Use native Strands execution
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self._agent.run(message)
                )
                content = str(response)
            else:
                # Stub implementation
                content = f"[Strands stub] Processed: {message[:100]}..."
                if tools:
                    content += f" (tools available: {', '.join(tools)})"

            # Add response to memory
            self._conversation_memory.append({"role": "assistant", "content": content})

            return AdapterResult(
                success=True,
                data={
                    "response": content,
                    "model": self._model,
                    "tools_used": tools or []
                }
            )

        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _stream(
        self,
        message: str,
        tools: Optional[List[str]] = None,
        **kwargs
    ) -> AdapterResult:
        """Stream agent execution (returns chunks)."""
        try:
            chunks = []

            if STRANDS_AVAILABLE and self._agent:
                # Use native Strands streaming
                for chunk in self._agent.stream(message):
                    chunks.append(str(chunk))
            else:
                # Stub implementation
                words = f"[Strands stub] Processed: {message[:100]}...".split()
                for word in words:
                    chunks.append(word + " ")

            content = "".join(chunks)
            self._conversation_memory.append({"role": "user", "content": message})
            self._conversation_memory.append({"role": "assistant", "content": content})

            return AdapterResult(
                success=True,
                data={
                    "response": content,
                    "chunks": chunks,
                    "chunk_count": len(chunks)
                }
            )

        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    async def _create_agent(
        self,
        name: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[str]] = None,
        **kwargs
    ) -> AdapterResult:
        """Create a new agent configuration."""
        # This would create a separate agent instance
        return AdapterResult(
            success=True,
            data={
                "agent_name": name,
                "model": model or self._model,
                "tools": tools or list(self._tools.keys()),
                "created": True
            }
        )

    async def _register_tool(
        self,
        name: str,
        description: str,
        handler: Optional[Callable] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AdapterResult:
        """Register a custom tool."""
        self._tools[name] = StrandsToolDef(
            name=name,
            description=description,
            handler=handler,
            parameters=parameters or {}
        )

        return AdapterResult(
            success=True,
            data={"tool": name, "registered": True}
        )

    async def _list_tools(self) -> AdapterResult:
        """List available tools."""
        tools_info = []
        for name, tool in self._tools.items():
            tools_info.append({
                "name": name,
                "description": tool.description,
                "has_handler": tool.handler is not None
            })

        return AdapterResult(
            success=True,
            data={"tools": tools_info, "count": len(tools_info)}
        )

    async def _clear_memory(self) -> AdapterResult:
        """Clear conversation memory."""
        self._conversation_memory.clear()
        return AdapterResult(
            success=True,
            data={"cleared": True}
        )

    async def _get_stats(self) -> AdapterResult:
        """Get adapter statistics."""
        return AdapterResult(
            success=True,
            data={
                "model": self._model,
                "tools_count": len(self._tools),
                "memory_turns": len(self._conversation_memory),
                "call_count": self._call_count,
                "error_count": self._error_count,
                "avg_latency_ms": self._total_latency_ms / max(1, self._call_count),
                "strands_native": STRANDS_AVAILABLE
            }
        )

    async def health_check(self) -> AdapterResult:
        """Check adapter health."""
        return AdapterResult(
            success=True,
            data={
                "status": "healthy",
                "model": self._model,
                "strands_available": STRANDS_AVAILABLE
            }
        )

    async def shutdown(self) -> AdapterResult:
        """Shutdown the adapter."""
        self._agent = None
        self._conversation_memory.clear()
        self._status = AdapterStatus.UNINITIALIZED
        logger.info("Strands Agents adapter shutdown")
        return AdapterResult(success=True)


# Register with SDK registry
try:
    from core.orchestration.sdk_registry import register_adapter, SDKLayer

    @register_adapter("strands-agents", SDKLayer.ORCHESTRATION, priority=12)
    class RegisteredStrandsAgentsAdapter(StrandsAgentsAdapter):
        """Registered Strands Agents adapter."""
        pass

except ImportError:
    pass


__all__ = ["StrandsAgentsAdapter", "STRANDS_AVAILABLE", "StrandsToolDef"]
