#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
MCP Guard Hook - V10 Optimized

Pre-tool validation for MCP server calls.
Implements lightweight security checks without blocking legitimate operations.

Returns:
    - allow: Proceed with the tool call
    - deny: Block the tool call with reason
    - modify: Allow with modified input
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Tuple


# Dangerous patterns to block
BLOCKED_PATTERNS = {
    "mcp__filesystem__write_file": [
        r"\.env$",
        r"\.env\.",
        r"/\.git/",
        r"\\\.git\\",
        r"secrets?/",
        r"\.pem$",
        r"\.key$",
        r"id_rsa",
        r"\.ssh/",
    ],
    "mcp__filesystem__read_file": [
        r"\.env$",
        r"\.env\.",
        r"/\.git/config",
        r"secrets?/.*\.(json|yaml|yml|env)",
        r"\.pem$",
        r"\.key$",
        r"id_rsa",
    ],
    "mcp__filesystem__delete_file": [
        r".*",  # Block all deletes by default
    ],
}

# Allowed patterns override blocks
ALLOWED_OVERRIDES = {
    "mcp__filesystem__write_file": [
        r"CLAUDE\.local\.md$",
        r"\.claude/",
    ],
}


def check_tool_call(tool_name: str, tool_input: Dict[str, Any]) -> Tuple[str, str, Dict]:
    """
    Check if a tool call should be allowed.
    
    Returns:
        Tuple of (decision, reason, modified_input)
        decision: "allow", "deny", or "modify"
    """
    # Extract file path from various input formats
    file_path = (
        tool_input.get("path") or 
        tool_input.get("file_path") or 
        tool_input.get("filepath") or
        tool_input.get("target") or
        ""
    )
    
    # Normalize path
    file_path = str(file_path).replace("\\", "/")
    
    # Check allowed overrides first
    if tool_name in ALLOWED_OVERRIDES:
        for pattern in ALLOWED_OVERRIDES[tool_name]:
            if re.search(pattern, file_path, re.IGNORECASE):
                return "allow", "Matched allowed override", tool_input
    
    # Check blocked patterns
    if tool_name in BLOCKED_PATTERNS:
        for pattern in BLOCKED_PATTERNS[tool_name]:
            if re.search(pattern, file_path, re.IGNORECASE):
                return "deny", f"Blocked by security pattern: {pattern}", tool_input
    
    # Check for path traversal attempts
    if ".." in file_path:
        return "deny", "Path traversal detected", tool_input
    
    # Check for absolute paths outside project
    project_dir = os.environ.get("PROJECT_DIR", os.getcwd())
    if file_path.startswith("/") or (len(file_path) > 1 and file_path[1] == ":"):
        # Absolute path - ensure it's within allowed directories
        allowed_roots = [
            project_dir,
            str(Path.home() / ".claude"),
            str(Path.home() / "projects"),
        ]
        
        is_allowed = any(
            file_path.lower().startswith(root.lower().replace("\\", "/"))
            for root in allowed_roots
        )
        
        if not is_allowed:
            return "deny", f"Path outside allowed directories: {file_path}", tool_input
    
    return "allow", "Passed all checks", tool_input


def main():
    # Read hook input from stdin
    try:
        hook_input = json.load(sys.stdin)
    except json.JSONDecodeError:
        # No input, allow by default
        print(json.dumps({"decision": "allow"}))
        return
    
    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})
    
    # Only process MCP tools
    if not tool_name.startswith("mcp__"):
        print(json.dumps({"decision": "allow"}))
        return
    
    decision, reason, modified_input = check_tool_call(tool_name, tool_input)
    
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": reason,
        }
    }
    
    if decision == "modify":
        output["hookSpecificOutput"]["updatedInput"] = modified_input
    
    print(json.dumps(output))
    
    # Exit code 0 for allow/modify, 2 for deny
    if decision == "deny":
        sys.exit(2)


if __name__ == "__main__":
    main()
