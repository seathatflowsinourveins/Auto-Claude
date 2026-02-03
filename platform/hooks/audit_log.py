#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Audit Log Hook - V10 Optimized

PostToolUse hook for logging file modifications.
Creates an audit trail of all Write and Edit operations.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def get_log_file() -> Path:
    """Get the audit log file path."""
    log_dir = Path.home() / ".claude" / "v10" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Daily log rotation
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return log_dir / f"audit_{today}.jsonl"


def log_event(event: Dict[str, Any]):
    """Append event to audit log."""
    log_file = get_log_file()
    
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def extract_file_info(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """Extract file information from tool input."""
    file_path = (
        tool_input.get("file_path") or
        tool_input.get("path") or
        tool_input.get("filepath") or
        tool_input.get("target") or
        "unknown"
    )
    
    content = tool_input.get("content", "")
    content_preview = content[:200] + "..." if len(content) > 200 else content
    
    return {
        "file_path": str(file_path),
        "content_length": len(content),
        "content_preview": content_preview,
        "line_count": content.count("\n") + 1 if content else 0
    }


def main():
    # Read hook input
    try:
        hook_input = json.load(sys.stdin)
    except json.JSONDecodeError:
        hook_input = {}
    
    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})
    tool_output = hook_input.get("tool_output", {})
    session_id = hook_input.get("session_id", "unknown")
    
    # Only log Write and Edit operations
    if tool_name not in ("Write", "Edit"):
        print(json.dumps({"status": "skipped", "reason": "Not a file modification"}))
        return
    
    # Extract file info
    file_info = extract_file_info(tool_name, tool_input)
    
    # Create audit event
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "tool": tool_name,
        "file_path": file_info["file_path"],
        "content_length": file_info["content_length"],
        "line_count": file_info["line_count"],
        "success": tool_output.get("success", True),
        "cwd": hook_input.get("cwd", os.getcwd())
    }
    
    # Log the event
    log_event(event)
    
    print(json.dumps({
        "status": "logged",
        "file": file_info["file_path"],
        "lines": file_info["line_count"]
    }))


if __name__ == "__main__":
    main()
