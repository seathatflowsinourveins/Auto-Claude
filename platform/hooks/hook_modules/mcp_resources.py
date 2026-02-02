#!/usr/bin/env python3
"""
MCP Resources Module - Prompts, Resources, and Templates

This module contains MCP resource patterns for prompts, files, and templates.
Extracted from hook_utils.py for modular architecture.

Exports:
- PromptArgument: Argument definition for prompts
- MCPPrompt: Prompt template exposed by servers
- PromptMessage: Message content from expanded prompt
- ResourceAnnotations: Metadata for resources
- MCPResource: Resource exposed by servers
- ResourceTemplate: Resource template with URI pattern

Version: V1.0.0 (2026-01-30) - Extracted from hook_utils.py V10.11
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


@dataclass
class PromptArgument:
    """
    Argument definition for MCP prompt templates.

    Per MCP 2025-11-25:
    - name: Unique argument identifier
    - description: Human-readable description
    - required: Whether argument must be provided
    """
    name: str
    description: str = ""
    required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP prompt argument format."""
        result: Dict[str, Any] = {"name": self.name}
        if self.description:
            result["description"] = self.description
        if self.required:
            result["required"] = True
        return result


@dataclass
class MCPPrompt:
    """
    MCP prompt template exposed by servers.

    Per MCP 2025-11-25:
    - Servers expose prompt templates via prompts/list
    - Clients retrieve expanded prompts via prompts/get
    - Arguments are filled by the client

    Example:
        prompt = MCPPrompt(
            name="code_review",
            title="Request Code Review",
            description="Asks the LLM to analyze code quality",
            arguments=[
                PromptArgument("code", "The code to review", required=True),
                PromptArgument("language", "Programming language")
            ]
        )
    """
    name: str
    description: str = ""
    title: Optional[str] = None
    arguments: List[PromptArgument] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP prompt list item format."""
        result: Dict[str, Any] = {
            "name": self.name,
        }
        if self.title:
            result["title"] = self.title
        if self.description:
            result["description"] = self.description
        if self.arguments:
            result["arguments"] = [a.to_dict() for a in self.arguments]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPPrompt":
        """Parse prompt from MCP response."""
        args = [
            PromptArgument(
                name=a.get("name", ""),
                description=a.get("description", ""),
                required=a.get("required", False)
            )
            for a in data.get("arguments", [])
        ]
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            title=data.get("title"),
            arguments=args
        )

    def get_required_args(self) -> List[str]:
        """Get names of required arguments."""
        return [a.name for a in self.arguments if a.required]


@dataclass
class PromptMessage:
    """
    Message content from an expanded prompt.

    Per MCP 2025-11-25, prompts/get returns:
    - description: Expanded description
    - messages: Array of role/content messages
    """
    role: str
    content: Union[str, Dict[str, Any], List[Dict[str, Any]]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP prompt message format."""
        if isinstance(self.content, str):
            content: Any = {"type": "text", "text": self.content}
        else:
            content = self.content
        return {"role": self.role, "content": content}

    @classmethod
    def user(cls, text: str) -> "PromptMessage":
        """Create a user message."""
        return cls(role="user", content=text)

    @classmethod
    def assistant(cls, text: str) -> "PromptMessage":
        """Create an assistant message."""
        return cls(role="assistant", content=text)


@dataclass
class ResourceAnnotations:
    """
    MCP resource annotations for metadata.

    Per MCP 2025-11-25:
    - audience: Who this resource is intended for
    - priority: Importance hint 0-1 (higher = more important)
    - lastModified: ISO timestamp of last modification
    """
    audience: Optional[List[str]] = None
    priority: Optional[float] = None
    last_modified: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP annotations format."""
        result: Dict[str, Any] = {}
        if self.audience:
            result["audience"] = self.audience
        if self.priority is not None:
            result["priority"] = self.priority
        if self.last_modified:
            result["lastModified"] = self.last_modified.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceAnnotations":
        """Parse annotations from MCP response."""
        modified = data.get("lastModified")
        return cls(
            audience=data.get("audience"),
            priority=data.get("priority"),
            last_modified=datetime.fromisoformat(modified.replace("Z", "+00:00")) if modified else None
        )

    @classmethod
    def for_user(cls, priority: float = 0.5) -> "ResourceAnnotations":
        """Create annotations for user-facing resources."""
        return cls(audience=["user"], priority=priority)

    @classmethod
    def for_assistant(cls, priority: float = 0.5) -> "ResourceAnnotations":
        """Create annotations for assistant-only resources."""
        return cls(audience=["assistant"], priority=priority)

    @classmethod
    def for_both(cls, priority: float = 0.5) -> "ResourceAnnotations":
        """Create annotations for both user and assistant."""
        return cls(audience=["user", "assistant"], priority=priority)


@dataclass
class MCPResource:
    """
    MCP resource exposed by servers.

    Per MCP 2025-11-25:
    - uri: Unique resource identifier (file://, https://, git://, etc.)
    - name: Human-readable name
    - description: Optional description
    - mimeType: Content type hint
    - annotations: Metadata (audience, priority, lastModified)
    """
    uri: str
    name: str
    description: str = ""
    mime_type: Optional[str] = None
    annotations: Optional[ResourceAnnotations] = None
    size: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP resource format."""
        result: Dict[str, Any] = {
            "uri": self.uri,
            "name": self.name,
        }
        if self.description:
            result["description"] = self.description
        if self.mime_type:
            result["mimeType"] = self.mime_type
        if self.annotations:
            result["annotations"] = self.annotations.to_dict()
        if self.size is not None:
            result["size"] = self.size
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPResource":
        """Parse resource from MCP response."""
        annotations = None
        if "annotations" in data:
            annotations = ResourceAnnotations.from_dict(data["annotations"])
        return cls(
            uri=data.get("uri", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            mime_type=data.get("mimeType"),
            annotations=annotations,
            size=data.get("size")
        )

    @classmethod
    def file(
        cls,
        path: str,
        name: Optional[str] = None,
        mime_type: str = "text/plain"
    ) -> "MCPResource":
        """Create a file resource."""
        import os
        return cls(
            uri=f"file:///{path.replace(os.sep, '/')}",
            name=name or os.path.basename(path),
            mime_type=mime_type
        )


@dataclass
class ResourceTemplate:
    """
    MCP resource template with URI pattern.

    Per MCP 2025-11-25, templates use RFC 6570 URI Templates:
    - uriTemplate: Pattern like "file:///{path}" or "db://users/{userId}"
    - Clients fill in template variables to get actual URIs

    Example:
        template = ResourceTemplate(
            uri_template="file:///{path}",
            name="Project Files",
            description="Access files in the project"
        )
    """
    uri_template: str
    name: str
    description: str = ""
    mime_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP resource template format."""
        result: Dict[str, Any] = {
            "uriTemplate": self.uri_template,
            "name": self.name,
        }
        if self.description:
            result["description"] = self.description
        if self.mime_type:
            result["mimeType"] = self.mime_type
        return result

    def expand(self, **variables: str) -> str:
        """Expand template with variables (simple implementation)."""
        result = self.uri_template
        for key, value in variables.items():
            result = result.replace(f"{{{key}}}", value)
        return result

    def to_resource(self, **variables: str) -> MCPResource:
        """Expand template and create a resource."""
        return MCPResource(
            uri=self.expand(**variables),
            name=self.name,
            description=self.description,
            mime_type=self.mime_type
        )


# Export all symbols
__all__ = [
    "PromptArgument",
    "MCPPrompt",
    "PromptMessage",
    "ResourceAnnotations",
    "MCPResource",
    "ResourceTemplate",
]
