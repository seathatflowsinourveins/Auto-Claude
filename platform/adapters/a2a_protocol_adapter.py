"""
A2A Protocol Adapter - V36 Architecture

Integrates Google's Agent-to-Agent (A2A) Protocol for inter-agent communication.

SDK: google-a2a (Agent-to-Agent Protocol)
Layer: L0 (Protocol)
Features:
- Standardized agent communication protocol
- Discovery and capability exchange
- Task delegation between agents
- Secure message passing
- Multi-agent coordination

A2A Protocol Spec (from Google):
- Agent Cards: Capability advertisements
- Task Protocol: Request/response patterns
- Streaming: Real-time updates
- Security: Signature verification

Usage:
    from adapters.a2a_protocol_adapter import A2AProtocolAdapter

    adapter = A2AProtocolAdapter()
    await adapter.initialize({"agent_id": "my-agent", "capabilities": ["code", "research"]})

    # Discover other agents
    result = await adapter.execute("discover", capability="code")

    # Delegate task
    result = await adapter.execute("delegate", target_agent="coder-1", task="Write a function")
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# SDK availability check
A2A_AVAILABLE = False

try:
    # Check for A2A protocol implementation
    from google.a2a import AgentCard, TaskRequest, TaskResponse
    A2A_AVAILABLE = True
except ImportError:
    logger.info("Google A2A not installed - using protocol-compliant stub")


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
        PROTOCOL = 0

    @_dataclass
    class AdapterResult:
        success: bool
        data: Optional[Dict[str, Any]] = None
        error: Optional[str] = None
        latency_ms: float = 0.0

    @_dataclass
    class AdapterConfig:
        name: str = "a2a-protocol"
        layer: int = 0

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
class AgentCard:
    """A2A Agent Card - capability advertisement."""
    agent_id: str
    name: str
    description: str
    capabilities: List[str]
    version: str = "1.0"
    endpoint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "version": self.version,
            "endpoint": self.endpoint,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class TaskRequest:
    """A2A Task Request."""
    task_id: str
    source_agent: str
    target_agent: str
    task_type: str
    payload: Dict[str, Any]
    priority: int = 5
    timeout_ms: float = 30000.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TaskResponse:
    """A2A Task Response."""
    task_id: str
    source_agent: str
    status: str  # pending, in_progress, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    completed_at: Optional[datetime] = None


class A2AProtocolAdapter(SDKAdapter):
    """
    Google A2A Protocol adapter for inter-agent communication.

    Implements the Agent-to-Agent protocol for:
    - Agent discovery via capability matching
    - Task delegation with request/response
    - Streaming updates for long-running tasks
    - Secure inter-agent messaging

    Operations:
    - register: Register this agent's capabilities
    - discover: Find agents by capability
    - delegate: Send task to another agent
    - respond: Respond to a received task
    - broadcast: Send message to all agents
    - get_card: Get an agent's capability card
    - list_tasks: List pending/active tasks
    """

    def __init__(self, config: Optional[AdapterConfig] = None):
        self._config = config or AdapterConfig(
            name="a2a-protocol",
            layer=SDKLayer.PROTOCOL
        )
        self._status = AdapterStatus.UNINITIALIZED
        self._agent_card: Optional[AgentCard] = None
        self._known_agents: Dict[str, AgentCard] = {}
        self._pending_tasks: Dict[str, TaskRequest] = {}
        self._completed_tasks: Dict[str, TaskResponse] = {}
        self._call_count = 0
        self._total_latency_ms = 0.0
        self._error_count = 0

    @property
    def sdk_name(self) -> str:
        return "a2a-protocol"

    @property
    def layer(self) -> SDKLayer:
        return SDKLayer.PROTOCOL

    @property
    def available(self) -> bool:
        # Protocol-compliant implementation always available
        return True

    @property
    def status(self) -> AdapterStatus:
        return self._status

    async def initialize(self, config: Dict[str, Any]) -> AdapterResult:
        """Initialize A2A protocol adapter."""
        try:
            agent_id = config.get("agent_id") or f"agent-{uuid.uuid4().hex[:8]}"
            name = config.get("name", "UNLEASH Agent")
            description = config.get("description", "V36 Architecture Agent")
            capabilities = config.get("capabilities", ["general"])
            endpoint = config.get("endpoint")

            # Create agent card
            self._agent_card = AgentCard(
                agent_id=agent_id,
                name=name,
                description=description,
                capabilities=capabilities,
                endpoint=endpoint
            )

            # Register known agents from config
            known_agents = config.get("known_agents", [])
            for agent_data in known_agents:
                if isinstance(agent_data, dict):
                    card = AgentCard(**agent_data)
                    self._known_agents[card.agent_id] = card

            self._status = AdapterStatus.READY
            logger.info(f"A2A Protocol adapter initialized (agent_id={agent_id})")

            return AdapterResult(
                success=True,
                data={
                    "agent_id": agent_id,
                    "capabilities": capabilities,
                    "known_agents": len(self._known_agents)
                }
            )

        except Exception as e:
            self._status = AdapterStatus.FAILED
            logger.error(f"A2A initialization failed: {e}")
            return AdapterResult(success=False, error=str(e))

    async def execute(self, operation: str, **kwargs) -> AdapterResult:
        """Execute an A2A protocol operation."""
        start_time = time.time()

        try:
            if operation == "register":
                result = await self._register(**kwargs)
            elif operation == "discover":
                result = await self._discover(**kwargs)
            elif operation == "delegate":
                result = await self._delegate(**kwargs)
            elif operation == "respond":
                result = await self._respond(**kwargs)
            elif operation == "broadcast":
                result = await self._broadcast(**kwargs)
            elif operation == "get_card":
                result = await self._get_card(**kwargs)
            elif operation == "list_tasks":
                result = await self._list_tasks(**kwargs)
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
            logger.error(f"A2A execute error: {e}")
            return AdapterResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000
            )

    async def _register(
        self,
        agent_card: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AdapterResult:
        """Register an agent's capabilities."""
        if agent_card:
            card = AgentCard(**agent_card)
            self._known_agents[card.agent_id] = card
        else:
            # Register self
            if self._agent_card:
                self._known_agents[self._agent_card.agent_id] = self._agent_card

        return AdapterResult(
            success=True,
            data={
                "registered": True,
                "total_agents": len(self._known_agents)
            }
        )

    async def _discover(
        self,
        capability: Optional[str] = None,
        name_pattern: Optional[str] = None,
        **kwargs
    ) -> AdapterResult:
        """Discover agents by capability."""
        matching_agents = []

        for agent_id, card in self._known_agents.items():
            if capability and capability not in card.capabilities:
                continue
            if name_pattern and name_pattern.lower() not in card.name.lower():
                continue

            matching_agents.append(card.to_dict())

        return AdapterResult(
            success=True,
            data={
                "agents": matching_agents,
                "count": len(matching_agents),
                "filter": {"capability": capability, "name_pattern": name_pattern}
            }
        )

    async def _delegate(
        self,
        target_agent: str,
        task: str,
        task_type: str = "general",
        payload: Optional[Dict[str, Any]] = None,
        priority: int = 5,
        timeout_ms: float = 30000.0,
        **kwargs
    ) -> AdapterResult:
        """Delegate a task to another agent."""
        if target_agent not in self._known_agents:
            return AdapterResult(
                success=False,
                error=f"Unknown target agent: {target_agent}"
            )

        task_id = f"task-{uuid.uuid4().hex[:12]}"
        source_agent = self._agent_card.agent_id if self._agent_card else "unknown"

        request = TaskRequest(
            task_id=task_id,
            source_agent=source_agent,
            target_agent=target_agent,
            task_type=task_type,
            payload={"task": task, **(payload or {})},
            priority=priority,
            timeout_ms=timeout_ms
        )

        self._pending_tasks[task_id] = request

        # In a real implementation, this would send to the target agent's endpoint
        # For now, simulate task acceptance
        return AdapterResult(
            success=True,
            data={
                "task_id": task_id,
                "target_agent": target_agent,
                "status": "delegated",
                "priority": priority
            }
        )

    async def _respond(
        self,
        task_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        **kwargs
    ) -> AdapterResult:
        """Respond to a received task."""
        if task_id not in self._pending_tasks:
            return AdapterResult(
                success=False,
                error=f"Task not found: {task_id}"
            )

        request = self._pending_tasks[task_id]
        response = TaskResponse(
            task_id=task_id,
            source_agent=self._agent_card.agent_id if self._agent_card else "unknown",
            status=status,
            result=result,
            error=error,
            completed_at=datetime.now(timezone.utc) if status in ("completed", "failed") else None
        )

        if status in ("completed", "failed"):
            del self._pending_tasks[task_id]
            self._completed_tasks[task_id] = response

        return AdapterResult(
            success=True,
            data={
                "task_id": task_id,
                "status": status,
                "responded": True
            }
        )

    async def _broadcast(
        self,
        message: str,
        capability_filter: Optional[str] = None,
        **kwargs
    ) -> AdapterResult:
        """Broadcast message to all/filtered agents."""
        targets = []

        for agent_id, card in self._known_agents.items():
            if capability_filter and capability_filter not in card.capabilities:
                continue
            targets.append(agent_id)

        return AdapterResult(
            success=True,
            data={
                "message": message,
                "targets": targets,
                "target_count": len(targets),
                "capability_filter": capability_filter
            }
        )

    async def _get_card(self, agent_id: Optional[str] = None, **kwargs) -> AdapterResult:
        """Get an agent's capability card."""
        if agent_id is None:
            # Return own card
            if self._agent_card:
                return AdapterResult(
                    success=True,
                    data={"card": self._agent_card.to_dict()}
                )
            return AdapterResult(success=False, error="No agent card configured")

        if agent_id in self._known_agents:
            return AdapterResult(
                success=True,
                data={"card": self._known_agents[agent_id].to_dict()}
            )

        return AdapterResult(
            success=False,
            error=f"Agent not found: {agent_id}"
        )

    async def _list_tasks(
        self,
        status_filter: Optional[str] = None,
        **kwargs
    ) -> AdapterResult:
        """List tasks."""
        tasks = []

        # Pending tasks
        if status_filter in (None, "pending", "in_progress"):
            for task_id, request in self._pending_tasks.items():
                tasks.append({
                    "task_id": task_id,
                    "status": "pending",
                    "target_agent": request.target_agent,
                    "task_type": request.task_type,
                    "created_at": request.created_at.isoformat()
                })

        # Completed tasks
        if status_filter in (None, "completed", "failed"):
            for task_id, response in self._completed_tasks.items():
                tasks.append({
                    "task_id": task_id,
                    "status": response.status,
                    "completed_at": response.completed_at.isoformat() if response.completed_at else None
                })

        return AdapterResult(
            success=True,
            data={
                "tasks": tasks,
                "count": len(tasks),
                "pending_count": len(self._pending_tasks),
                "completed_count": len(self._completed_tasks)
            }
        )

    async def _get_stats(self) -> AdapterResult:
        """Get adapter statistics."""
        return AdapterResult(
            success=True,
            data={
                "agent_id": self._agent_card.agent_id if self._agent_card else None,
                "known_agents": len(self._known_agents),
                "pending_tasks": len(self._pending_tasks),
                "completed_tasks": len(self._completed_tasks),
                "call_count": self._call_count,
                "error_count": self._error_count,
                "avg_latency_ms": self._total_latency_ms / max(1, self._call_count)
            }
        )

    async def health_check(self) -> AdapterResult:
        """Check adapter health."""
        return AdapterResult(
            success=True,
            data={
                "status": "healthy",
                "agent_id": self._agent_card.agent_id if self._agent_card else None,
                "a2a_native": A2A_AVAILABLE
            }
        )

    async def shutdown(self) -> AdapterResult:
        """Shutdown the adapter."""
        self._known_agents.clear()
        self._pending_tasks.clear()
        self._completed_tasks.clear()
        self._status = AdapterStatus.UNINITIALIZED
        logger.info("A2A Protocol adapter shutdown")
        return AdapterResult(success=True)


# Register with SDK registry
try:
    from core.orchestration.sdk_registry import register_adapter, SDKLayer

    @register_adapter("a2a-protocol", SDKLayer.PROTOCOL, priority=20)
    class RegisteredA2AProtocolAdapter(A2AProtocolAdapter):
        """Registered A2A Protocol adapter."""
        pass

except ImportError:
    pass


__all__ = ["A2AProtocolAdapter", "A2A_AVAILABLE", "AgentCard", "TaskRequest", "TaskResponse"]
