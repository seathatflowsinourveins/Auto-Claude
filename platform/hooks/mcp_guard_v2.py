#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
MCP Guard Hook V2 - V10 Optimized

Enhanced PreToolUse validation for MCP server calls.
Now supports input modification (v2.0.10+) in addition to allow/deny.

Key Features:
- Input modification for safe transformations
- Path normalization and sanitization
- Kill switch support
- Audit logging of decisions
- Allowlist overrides

Based on Claude Code Hooks v2.0.10+:
https://code.claude.com/docs/en/hooks
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

# Try to use hook_utils (optional enhancement)
try:
    from hook_utils import HookConfig  # noqa: F401
    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False


# Security patterns for blocking
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
        r"credentials",
        r"\.aws/",
        r"\.kube/config",
    ],
    "mcp__filesystem__read_file": [
        r"\.env$",
        r"\.env\.",
        r"/\.git/config",
        r"secrets?/.*\.(json|yaml|yml|env|pem|key)",
        r"\.pem$",
        r"\.key$",
        r"id_rsa",
        r"\.aws/credentials",
    ],
    "mcp__filesystem__delete_file": [
        r".*",  # Block all deletes by default
    ],
}

# Patterns that override blocks (always allowed)
ALLOWED_OVERRIDES = {
    "mcp__filesystem__write_file": [
        r"CLAUDE\.local\.md$",
        r"\.claude/",
        r"\.claude\\",
        r"node_modules/\.cache",
        r"__pycache__",
    ],
    "mcp__filesystem__read_file": [
        r"\.claude/",
        r"CLAUDE\.md$",
        r"CLAUDE\.local\.md$",
    ],
}

# Input transformations (v2.0.10+)
# These modify inputs to make them safer rather than blocking
INPUT_TRANSFORMS = {
    "mcp__filesystem__write_file": {
        # Normalize line endings to LF
        "content": lambda x: x.replace("\r\n", "\n") if isinstance(x, str) else x,
    },
}


def normalize_path(path: str) -> str:
    """Normalize a file path for consistent checking."""
    return str(path).replace("\\", "/").lower()


def get_project_dir() -> Path:
    """Get the project directory from environment."""
    return Path(os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd()))


def is_path_allowed(file_path: str, tool_name: str) -> Tuple[bool, str]:
    """
    Check if a path is allowed for the given tool.

    Returns:
        Tuple of (is_allowed, reason)
    """
    normalized = normalize_path(file_path)
    project_dir = normalize_path(str(get_project_dir()))

    # Check allowed overrides first (they bypass blocks)
    if tool_name in ALLOWED_OVERRIDES:
        for pattern in ALLOWED_OVERRIDES[tool_name]:
            if re.search(pattern, normalized, re.IGNORECASE):
                return True, f"Allowed by override: {pattern}"

    # Check blocked patterns
    if tool_name in BLOCKED_PATTERNS:
        for pattern in BLOCKED_PATTERNS[tool_name]:
            if re.search(pattern, normalized, re.IGNORECASE):
                return False, f"Blocked by security pattern: {pattern}"

    # Check for path traversal
    if ".." in normalized:
        return False, "Path traversal detected"

    # Check absolute paths are within allowed directories
    if normalized.startswith("/") or (len(normalized) > 1 and normalized[1] == ":"):
        allowed_roots = [
            project_dir,
            normalize_path(str(Path.home() / ".claude")),
            normalize_path(str(Path.home() / "projects")),
        ]

        is_allowed = any(
            normalized.startswith(root)
            for root in allowed_roots
        )

        if not is_allowed:
            return False, f"Path outside allowed directories: {file_path}"

    return True, "Path allowed"


def transform_input(tool_name: str, tool_input: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Apply input transformations for safety.

    Returns:
        Tuple of (modified_input, was_modified)
    """
    if tool_name not in INPUT_TRANSFORMS:
        return tool_input, False

    modified = dict(tool_input)
    was_modified = False

    for key, transform_fn in INPUT_TRANSFORMS[tool_name].items():
        if key in modified:
            original = modified[key]
            transformed = transform_fn(original)
            if transformed != original:
                modified[key] = transformed
                was_modified = True

    return modified, was_modified


def check_tool_call(
    tool_name: str,
    tool_input: Dict[str, Any]
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """
    Check if an MCP tool call should be allowed.

    Returns:
        Tuple of (decision, reason, modified_input)
        decision: "allow", "deny", or "modify"
    """
    # Check kill switch first
    if (Path.home() / ".claude" / "KILL_SWITCH").exists():
        return "deny", "Kill switch is active", None

    # Extract file path from various input formats
    file_path = (
        tool_input.get("path") or
        tool_input.get("file_path") or
        tool_input.get("filepath") or
        tool_input.get("target") or
        ""
    )

    # Check path security
    if file_path:
        is_allowed, reason = is_path_allowed(str(file_path), tool_name)
        if not is_allowed:
            return "deny", reason, None

    # Apply input transformations
    modified_input, was_modified = transform_input(tool_name, tool_input)

    if was_modified:
        return "modify", "Input transformed for safety", modified_input

    return "allow", "Passed all security checks", None


def log_decision(
    tool_name: str,
    decision: str,
    reason: str,
    tool_input: Dict[str, Any]
) -> None:
    """Log the security decision for audit trail."""
    log_dir = Path.home() / ".claude" / "v10" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = log_dir / f"security_{today}.jsonl"

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "decision": decision,
        "reason": reason,
        "path": tool_input.get("path") or tool_input.get("file_path", ""),
    }

    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    # Read hook input from stdin
    try:
        hook_input = json.load(sys.stdin)
    except json.JSONDecodeError:
        print(json.dumps({"decision": "allow"}))
        return

    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})

    # Only process MCP tools
    if not tool_name.startswith("mcp__"):
        print(json.dumps({"decision": "allow"}))
        return

    # Check the tool call
    decision, reason, modified_input = check_tool_call(tool_name, tool_input)

    # Log the decision
    log_decision(tool_name, decision, reason, tool_input)

    # Build response per official Claude Code hooks spec
    # https://code.claude.com/docs/en/hooks
    # permissionDecision must be "allow", "deny", or "ask" (NOT "modify")
    # Use "allow" + updatedInput for input modification

    # Map internal decision to official values
    if decision == "modify":
        official_decision = "allow"  # Modification uses "allow" + updatedInput
    else:
        official_decision = decision  # "allow" or "deny"

    hook_output: Dict[str, Any] = {
        "hookEventName": "PreToolUse",
        "permissionDecision": official_decision,
        "permissionDecisionReason": reason,
    }

    # v2.0.10+: Include modified input when modifying (with "allow" decision)
    if decision == "modify" and modified_input:
        hook_output["updatedInput"] = modified_input

    output: Dict[str, Any] = {"hookSpecificOutput": hook_output}

    print(json.dumps(output))

    # Exit code: 0 for allow (including modified), 2 for deny
    if decision == "deny":
        sys.exit(2)


if __name__ == "__main__":
    main()
