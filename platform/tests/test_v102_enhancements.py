#!/usr/bin/env python3
"""
Test Suite for V10.2 Enhanced Hooks

Tests the new features added based on official documentation research:
- hook_utils.py: Full hook event output formats, MCP structured content
- letta_sync_v2.py: Tool rules, MCP server management, compaction settings

Run with: python -m pytest tests/test_v102_enhancements.py -v
"""

import json
import sys
from pathlib import Path

# Add hooks directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "hooks"))

from hook_utils import (
    HookEvent,
    PermissionDecision,
    PermissionBehavior,
    BlockDecision,
    HookInput,
    HookResponse,
    ToolMatcher,
    MCPContent,
    MCPToolResult,
)


class TestPermissionDecision:
    """Test PermissionDecision enum values match official spec."""

    def test_permission_values(self):
        """Verify enum values match Claude Code hooks spec."""
        assert PermissionDecision.ALLOW.value == "allow"
        assert PermissionDecision.DENY.value == "deny"
        assert PermissionDecision.ASK.value == "ask"


class TestPermissionBehavior:
    """Test PermissionBehavior enum for PermissionRequest events."""

    def test_behavior_values(self):
        """Verify PermissionRequest behavior values."""
        assert PermissionBehavior.ALLOW.value == "allow"
        assert PermissionBehavior.DENY.value == "deny"


class TestBlockDecision:
    """Test BlockDecision enum for PostToolUse/Stop events."""

    def test_block_value(self):
        """Verify block decision value."""
        assert BlockDecision.BLOCK.value == "block"


class TestHookResponse:
    """Test HookResponse class with V10.2 enhancements."""

    def test_pretooluse_allow(self):
        """Test PreToolUse allow response format."""
        response = HookResponse.allow(reason="Safe operation")
        output = json.loads(response.to_json())

        assert "hookSpecificOutput" in output
        assert output["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
        assert output["hookSpecificOutput"]["permissionDecision"] == "allow"
        assert output["hookSpecificOutput"]["permissionDecisionReason"] == "Safe operation"

    def test_pretooluse_deny(self):
        """Test PreToolUse deny response format."""
        response = HookResponse.deny(reason="Dangerous command")
        output = json.loads(response.to_json())

        assert output["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert output["hookSpecificOutput"]["permissionDecisionReason"] == "Dangerous command"

    def test_pretooluse_ask(self):
        """Test PreToolUse ask response format."""
        response = HookResponse.ask(reason="User confirmation needed")
        output = json.loads(response.to_json())

        assert output["hookSpecificOutput"]["permissionDecision"] == "ask"

    def test_pretooluse_modify(self):
        """Test PreToolUse with updatedInput."""
        response = HookResponse.modify(
            updated_input={"command": "ls -la"},
            reason="Sanitized command"
        )
        output = json.loads(response.to_json())

        assert output["hookSpecificOutput"]["permissionDecision"] == "allow"
        assert output["hookSpecificOutput"]["updatedInput"] == {"command": "ls -la"}

    def test_pretooluse_with_context(self):
        """Test PreToolUse with additionalContext."""
        response = HookResponse.allow(
            reason="Approved",
            context="Previous session learned: prefer async patterns"
        )
        output = json.loads(response.to_json())

        assert output["hookSpecificOutput"]["additionalContext"] == "Previous session learned: prefer async patterns"

    def test_permissionrequest_allow(self):
        """Test PermissionRequest allow response format."""
        response = HookResponse(
            permission_behavior=PermissionBehavior.ALLOW,
            permission_message="Auto-approved by policy",
            _event_type=HookEvent.PERMISSION_REQUEST
        )
        output = json.loads(response.to_json())

        assert output["hookSpecificOutput"]["hookEventName"] == "PermissionRequest"
        assert output["hookSpecificOutput"]["decision"]["behavior"] == "allow"
        assert output["hookSpecificOutput"]["decision"]["message"] == "Auto-approved by policy"

    def test_permissionrequest_deny_with_interrupt(self):
        """Test PermissionRequest deny with interrupt."""
        response = HookResponse(
            permission_behavior=PermissionBehavior.DENY,
            permission_message="Blocked by security policy",
            permission_interrupt=True,
            _event_type=HookEvent.PERMISSION_REQUEST
        )
        output = json.loads(response.to_json())

        assert output["hookSpecificOutput"]["decision"]["behavior"] == "deny"
        assert output["hookSpecificOutput"]["decision"]["interrupt"] is True

    def test_posttooluse_block(self):
        """Test PostToolUse block response format."""
        response = HookResponse.block_action(
            reason="Tool output contains sensitive data",
            event=HookEvent.POST_TOOL_USE
        )
        output = json.loads(response.to_json())

        assert output["decision"] == "block"
        assert output["reason"] == "Tool output contains sensitive data"
        assert output["hookSpecificOutput"]["hookEventName"] == "PostToolUse"

    def test_stop_block(self):
        """Test Stop block response format."""
        response = HookResponse.block_action(
            reason="Tasks incomplete",
            event=HookEvent.STOP
        )
        output = json.loads(response.to_json())

        assert output["decision"] == "block"
        assert output["reason"] == "Tasks incomplete"

    def test_userpromptsubmit_block(self):
        """Test UserPromptSubmit block response format."""
        response = HookResponse.block_action(
            reason="Prompt contains blocked content",
            event=HookEvent.USER_PROMPT_SUBMIT
        )
        output = json.loads(response.to_json())

        assert output["decision"] == "block"
        assert output["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"

    def test_sessionstart_context(self):
        """Test SessionStart with additionalContext."""
        response = HookResponse(
            additional_context="Welcome back! Previous session focused on API design.",
            _event_type=HookEvent.SESSION_START
        )
        output = json.loads(response.to_json())

        assert output["hookSpecificOutput"]["hookEventName"] == "SessionStart"
        assert "Previous session" in output["hookSpecificOutput"]["additionalContext"]

    def test_common_fields(self):
        """Test common output fields."""
        response = HookResponse(
            continue_session=False,
            stop_reason="Session ended by hook",
            suppress_output=True,
            system_message="Custom system message"
        )
        output = json.loads(response.to_json())

        assert output["continue"] is False
        assert output["stopReason"] == "Session ended by hook"
        assert output["suppressOutput"] is True
        assert output["systemMessage"] == "Custom system message"


class TestToolMatcher:
    """Test ToolMatcher class for tool name matching."""

    def test_exact_match(self):
        """Test exact tool name matching."""
        matcher = ToolMatcher("Bash")
        assert matcher.matches("Bash") is True
        assert matcher.matches("Edit") is False

    def test_wildcard_match(self):
        """Test wildcard matching all tools."""
        matcher = ToolMatcher("*")
        assert matcher.matches("Bash") is True
        assert matcher.matches("Edit") is True
        assert matcher.matches("mcp__memory__create") is True

    def test_regex_pattern(self):
        """Test regex pattern matching."""
        matcher = ToolMatcher("Edit|Write")
        assert matcher.matches("Edit") is True
        assert matcher.matches("Write") is True
        assert matcher.matches("Read") is False

    def test_mcp_server_match(self):
        """Test MCP server pattern matching."""
        matcher = ToolMatcher.mcp_server("memory")
        assert matcher.matches("mcp__memory__create_entities") is True
        assert matcher.matches("mcp__memory__search") is True
        assert matcher.matches("mcp__filesystem__read") is False

    def test_mcp_tool_match(self):
        """Test specific MCP tool matching."""
        matcher = ToolMatcher.mcp_tool("memory", "create_entities")
        assert matcher.matches("mcp__memory__create_entities") is True
        assert matcher.matches("mcp__memory__search") is False


class TestMCPContent:
    """Test MCPContent class for MCP content items."""

    def test_text_content(self):
        """Test text content creation."""
        content = MCPContent.text_content("Hello, world!")
        result = content.to_dict()

        assert result["type"] == "text"
        assert result["text"] == "Hello, world!"

    def test_image_content(self):
        """Test image content creation."""
        content = MCPContent.image_content("base64data==", "image/png")
        result = content.to_dict()

        assert result["type"] == "image"
        assert result["data"] == "base64data=="
        assert result["mimeType"] == "image/png"

    def test_resource_link(self):
        """Test resource link content."""
        content = MCPContent(
            content_type="resource_link",
            uri="file:///project/src/main.py",
            name="main.py",
            description="Main application entry",
            mime_type="text/x-python"
        )
        result = content.to_dict()

        assert result["type"] == "resource_link"
        assert result["uri"] == "file:///project/src/main.py"
        assert result["name"] == "main.py"
        assert result["mimeType"] == "text/x-python"

    def test_content_with_annotations(self):
        """Test content with annotations."""
        content = MCPContent(
            content_type="text",
            text="Important info",
            annotations={
                "audience": ["user", "assistant"],
                "priority": 0.9
            }
        )
        result = content.to_dict()

        assert result["annotations"]["audience"] == ["user", "assistant"]
        assert result["annotations"]["priority"] == 0.9


class TestMCPToolResult:
    """Test MCPToolResult class for MCP tool responses."""

    def test_success_result(self):
        """Test successful tool result."""
        result = MCPToolResult.success("Operation completed")
        output = result.to_dict()

        assert output["isError"] is False
        assert output["content"][0]["type"] == "text"
        assert output["content"][0]["text"] == "Operation completed"

    def test_error_result(self):
        """Test error tool result."""
        result = MCPToolResult.error("Something went wrong")
        output = result.to_dict()

        assert output["isError"] is True
        assert "Something went wrong" in output["content"][0]["text"]

    def test_structured_content(self):
        """Test structured content per MCP 2025-11-25 spec."""
        structured = {
            "temperature": 22.5,
            "conditions": "Partly cloudy",
            "humidity": 65
        }
        result = MCPToolResult.success(
            "Weather data retrieved",
            structured=structured
        )
        output = result.to_dict()

        assert output["structuredContent"] == structured
        # Should also have text content for backwards compatibility
        assert any(c.get("type") == "text" for c in output["content"])


class TestLettaSyncV2Features:
    """Test LettaSyncV2 V2.1 enhancements."""

    def test_tool_rule_types_defined(self):
        """Verify all 9 tool rule types from Letta SDK are defined."""
        from letta_sync_v2 import LettaSyncV2

        expected_rules = [
            "ChildToolRule",
            "InitToolRule",
            "TerminalToolRule",
            "ConditionalToolRule",
            "ContinueToolRule",
            "RequiredBeforeExitToolRule",
            "MaxCountPerStepToolRule",
            "ParentToolRule",
            "RequiresApprovalToolRule",
        ]

        for rule in expected_rules:
            assert rule in LettaSyncV2.TOOL_RULE_TYPES

    def test_init_with_options(self):
        """Test initialization with V2.1 options."""
        from letta_sync_v2 import LettaSyncV2

        sync = LettaSyncV2(
            enable_tool_rules=True,
            enable_mcp_management=True,
            timeout=60.0,
            max_retries=5
        )

        assert sync.enable_tool_rules is True
        assert sync.enable_mcp_management is True
        assert sync.timeout == 60.0
        assert sync.max_retries == 5


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
