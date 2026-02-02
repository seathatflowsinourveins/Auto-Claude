#!/usr/bin/env python3
"""
MCP Sampling Module - Server-Initiated LLM Calls

This module contains MCP sampling patterns for server-initiated LLM generations.
Extracted from hook_utils.py for modular architecture.

Exports:
- ToolChoiceMode: Tool choice modes for sampling
- SamplingMessage: Messages for sampling requests
- ModelPreferences: Model selection preferences
- SamplingTool: Tool definitions for sampling
- SamplingRequest: Complete sampling request
- SamplingResponse: Response from sampling

Version: V1.0.0 (2026-01-30) - Extracted from hook_utils.py V10.11
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ToolChoiceMode(Enum):
    """Tool choice modes for sampling requests.

    Per MCP Specification 2025-11-25:
    - AUTO: Model decides whether to use tools
    - REQUIRED: Model MUST use at least one tool
    - NONE: Model MUST NOT use any tools
    """
    AUTO = "auto"
    REQUIRED = "required"
    NONE = "none"


@dataclass
class SamplingMessage:
    """
    Message for MCP sampling requests.

    Supports text, image, and audio content types.
    """
    role: str  # "user" or "assistant"
    content: Union[str, Dict[str, Any], List[Dict[str, Any]]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP sampling message format."""
        if isinstance(self.content, str):
            content: Any = {"type": "text", "text": self.content}
        else:
            content = self.content
        return {"role": self.role, "content": content}

    @classmethod
    def user(cls, text: str) -> "SamplingMessage":
        """Create a user message with text content."""
        return cls(role="user", content=text)

    @classmethod
    def assistant(cls, text: str) -> "SamplingMessage":
        """Create an assistant message with text content."""
        return cls(role="assistant", content=text)


@dataclass
class ModelPreferences:
    """
    Model preferences for sampling requests.

    Allows servers to hint at model selection without mandating specific models.
    Priority values range from 0-1 where higher values indicate stronger preference.
    """
    hints: List[Dict[str, str]] = field(default_factory=list)
    cost_priority: float = 0.5
    speed_priority: float = 0.5
    intelligence_priority: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP model preferences format."""
        result: Dict[str, Any] = {}
        if self.hints:
            result["hints"] = self.hints
        if self.cost_priority != 0.5:
            result["costPriority"] = self.cost_priority
        if self.speed_priority != 0.5:
            result["speedPriority"] = self.speed_priority
        if self.intelligence_priority != 0.5:
            result["intelligencePriority"] = self.intelligence_priority
        return result

    @classmethod
    def prefer_claude(cls, variant: str = "sonnet") -> "ModelPreferences":
        """Create preferences hinting at Claude models."""
        return cls(hints=[{"name": f"claude-3-{variant}"}, {"name": "claude"}])

    @classmethod
    def prefer_fast(cls) -> "ModelPreferences":
        """Create preferences prioritizing speed over capability."""
        return cls(speed_priority=0.9, intelligence_priority=0.3, cost_priority=0.5)

    @classmethod
    def prefer_smart(cls) -> "ModelPreferences":
        """Create preferences prioritizing capability over speed."""
        return cls(intelligence_priority=0.9, speed_priority=0.3, cost_priority=0.3)

    @classmethod
    def prefer_cheap(cls) -> "ModelPreferences":
        """Create preferences prioritizing cost efficiency."""
        return cls(cost_priority=0.9, speed_priority=0.5, intelligence_priority=0.3)


@dataclass
class SamplingTool:
    """
    Tool definition for sampling requests.

    Servers can provide tools for the LLM to use during sampling.
    """
    name: str
    description: str
    input_schema: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema
        }


@dataclass
class SamplingRequest:
    """
    MCP sampling request for server-initiated LLM calls.

    Per MCP 2025-11-25, servers can request LLM generations from clients:
    - Basic text generation
    - With tool use support (multi-turn tool loops)
    - With model preference hints

    Example (basic):
        request = SamplingRequest(
            messages=[SamplingMessage.user("Write a haiku about coding")],
            max_tokens=100
        )

    Example (with tools):
        request = SamplingRequest(
            messages=[SamplingMessage.user("What's the weather in Paris?")],
            max_tokens=500,
            tools=[SamplingTool(
                name="get_weather",
                description="Get current weather",
                input_schema={"type": "object", "properties": {"city": {"type": "string"}}}
            )],
            tool_choice=ToolChoiceMode.AUTO
        )
    """
    messages: List[SamplingMessage]
    max_tokens: int = 1000
    system_prompt: Optional[str] = None
    temperature: Optional[float] = None
    model_preferences: Optional[ModelPreferences] = None
    tools: Optional[List[SamplingTool]] = None
    tool_choice: Optional[ToolChoiceMode] = None
    stop_sequences: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP sampling/createMessage request params."""
        result: Dict[str, Any] = {
            "messages": [m.to_dict() for m in self.messages],
            "maxTokens": self.max_tokens,
        }

        if self.system_prompt:
            result["systemPrompt"] = self.system_prompt
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.model_preferences:
            result["modelPreferences"] = self.model_preferences.to_dict()
        if self.tools:
            result["tools"] = [t.to_dict() for t in self.tools]
        if self.tool_choice:
            result["toolChoice"] = {"mode": self.tool_choice.value}
        if self.stop_sequences:
            result["stopSequences"] = self.stop_sequences

        return result


@dataclass
class SamplingResponse:
    """
    Response from a sampling request.

    Per MCP 2025-11-25:
    - content: Text or tool use content
    - model: The model that was used
    - stop_reason: Why generation stopped ("endTurn", "toolUse", "maxTokens", etc.)
    """
    role: str
    content: Union[str, Dict[str, Any], List[Dict[str, Any]]]
    model: str
    stop_reason: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SamplingResponse":
        """Parse sampling result from MCP response."""
        return cls(
            role=data.get("role", "assistant"),
            content=data.get("content", ""),
            model=data.get("model", "unknown"),
            stop_reason=data.get("stopReason", "unknown")
        )

    @property
    def is_tool_use(self) -> bool:
        """Check if response contains tool use requests."""
        return self.stop_reason == "toolUse"

    @property
    def text(self) -> str:
        """Extract text content from response."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, dict) and self.content.get("type") == "text":
            return str(self.content.get("text", ""))
        if isinstance(self.content, list):
            for item in self.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    return str(item.get("text", ""))
        return str(self.content)

    def get_tool_uses(self) -> List[Dict[str, Any]]:
        """Extract tool use requests from response."""
        if isinstance(self.content, list):
            return [c for c in self.content if isinstance(c, dict) and c.get("type") == "tool_use"]
        if isinstance(self.content, dict) and self.content.get("type") == "tool_use":
            return [self.content]
        return []


# Export all symbols
__all__ = [
    "ToolChoiceMode",
    "SamplingMessage",
    "ModelPreferences",
    "SamplingTool",
    "SamplingRequest",
    "SamplingResponse",
]
