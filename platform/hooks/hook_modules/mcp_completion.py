#!/usr/bin/env python3
"""
MCP Completion Module - Context-Aware Auto-Completion

This module contains MCP completion patterns for argument auto-completion.
Extracted from hook_utils.py for modular architecture.

Exports:
- CompletionRefType: Reference types for completion
- CompletionRequest: Completion request
- CompletionResult: Completion result

Version: V1.0.0 (2026-01-30) - Extracted from hook_utils.py V10.11
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CompletionRefType(Enum):
    """Reference types for completion requests.

    Per MCP 2025-11-25:
    - PROMPT: Reference to a prompt template
    - RESOURCE: Reference to a resource template
    """
    PROMPT = "ref/prompt"
    RESOURCE = "ref/resource"


@dataclass
class CompletionRequest:
    """
    MCP completion request for argument auto-completion.

    Per MCP 2025-11-25:
    - ref: Reference to prompt or resource template
    - argument: Current argument being completed
    - context: Previously provided argument values

    Example:
        request = CompletionRequest(
            ref_type=CompletionRefType.PROMPT,
            ref_name="code_review",
            argument_name="language",
            argument_value="py",
            context={"framework": "flask"}
        )
    """
    ref_type: CompletionRefType
    ref_name: str
    argument_name: str
    argument_value: str = ""
    context: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP completion/complete request params."""
        result: Dict[str, Any] = {
            "ref": {
                "type": self.ref_type.value,
                "name": self.ref_name,
            },
            "argument": {
                "name": self.argument_name,
                "value": self.argument_value,
            }
        }
        if self.context:
            result["context"] = {"arguments": self.context}
        return result

    @classmethod
    def for_prompt(
        cls,
        prompt_name: str,
        argument_name: str,
        argument_value: str = "",
        context: Optional[Dict[str, str]] = None
    ) -> "CompletionRequest":
        """Create a completion request for a prompt argument."""
        return cls(
            ref_type=CompletionRefType.PROMPT,
            ref_name=prompt_name,
            argument_name=argument_name,
            argument_value=argument_value,
            context=context
        )

    @classmethod
    def for_resource(
        cls,
        resource_template: str,
        argument_name: str,
        argument_value: str = "",
        context: Optional[Dict[str, str]] = None
    ) -> "CompletionRequest":
        """Create a completion request for a resource template argument."""
        return cls(
            ref_type=CompletionRefType.RESOURCE,
            ref_name=resource_template,
            argument_name=argument_name,
            argument_value=argument_value,
            context=context
        )


@dataclass
class CompletionResult:
    """
    Result of a completion request.

    Per MCP 2025-11-25:
    - values: List of completion suggestions
    - total: Total number of available completions (if known)
    - hasMore: Whether more completions are available
    """
    values: List[str] = field(default_factory=list)
    total: Optional[int] = None
    has_more: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompletionResult":
        """Parse completion result from MCP response."""
        completion = data.get("completion", {})
        return cls(
            values=completion.get("values", []),
            total=completion.get("total"),
            has_more=completion.get("hasMore", False)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP completion result format."""
        result: Dict[str, Any] = {
            "completion": {
                "values": self.values,
            }
        }
        if self.total is not None:
            result["completion"]["total"] = self.total
        if self.has_more:
            result["completion"]["hasMore"] = True
        return result

    @property
    def is_empty(self) -> bool:
        """Check if no completions were returned."""
        return len(self.values) == 0


# Export all symbols
__all__ = [
    "CompletionRefType",
    "CompletionRequest",
    "CompletionResult",
]
