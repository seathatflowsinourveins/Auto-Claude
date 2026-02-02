#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
Bash Guard Hook - V10 Optimized

Pre-tool validation for Bash commands.
Blocks dangerous commands while allowing development workflows.
"""

import json
import re
import sys
from typing import Tuple


# Explicitly blocked commands/patterns
BLOCKED_COMMANDS = [
    r"^rm\s+-rf\s+[/\\]",            # rm -rf /
    r"^rm\s+-rf\s+~",                 # rm -rf ~
    r"^rm\s+-rf\s+\*",                # rm -rf *
    r"^del\s+/[sq]",                  # del /s or /q (Windows)
    r"^format\s+",                    # format drive
    r"^mkfs\.",                       # make filesystem
    r"^dd\s+if=.*of=/dev/",           # dd to device
    r"^:\s*\(\)\s*{\s*:\|:\s*&\s*};", # Fork bomb
    r"^chmod\s+777\s+[/\\]",          # chmod 777 /
    r"^chmod\s+-R\s+777",             # recursive 777
    r"^sudo\s+",                      # sudo anything
    r"^su\s+-",                       # su -
    r"^curl\s+.*\|\s*(ba)?sh",        # curl | bash
    r"^wget\s+.*\|\s*(ba)?sh",        # wget | bash
    r"^eval\s+.*\$\(",                # eval with command substitution
    r">\s*/dev/sd",                   # write to disk device
    r">\s*/etc/passwd",               # overwrite passwd
    r">\s*/etc/shadow",               # overwrite shadow
    r"^net\s+user\s+.*\s+/add",       # Windows add user
    r"^reg\s+delete",                 # Windows registry delete
    r"^shutdown",                     # shutdown command
    r"^reboot",                       # reboot command
    r"^halt",                         # halt command
    r"^poweroff",                     # poweroff command
]

# Commands requiring confirmation (logged but allowed)
WARN_COMMANDS = [
    r"^rm\s+-rf",          # rm -rf (with safe path)
    r"^rm\s+-r\s+",        # rm -r
    r"^git\s+push.*-f",    # force push
    r"^git\s+push.*force", # force push
    r"^git\s+reset.*hard", # hard reset
    r"^npm\s+publish",     # npm publish
    r"^pip\s+.*--user",    # pip install to user
    r"^docker\s+rm\s+-f",  # docker force remove
    r"^docker\s+system\s+prune", # docker prune
]

# Explicitly allowed safe patterns
ALLOWED_PATTERNS = [
    r"^npm\s+(run|test|install|ci|build|start)",
    r"^yarn\s+(run|test|install|build|start)",
    r"^pnpm\s+(run|test|install|build|start)",
    r"^bun\s+(run|test|install|build|start)",
    r"^uv\s+(run|pip|sync|add)",
    r"^pip\s+install",
    r"^python\s+-m\s+(pytest|pip|venv)",
    r"^cargo\s+(build|test|run|check|clippy)",
    r"^git\s+(status|diff|log|branch|checkout|stash|fetch|pull)",
    r"^git\s+add\s+",
    r"^git\s+commit\s+",
    r"^git\s+push\s+(?!.*-f)(?!.*force)",  # push without force
    r"^ls\s+",
    r"^dir\s+",
    r"^cat\s+",
    r"^type\s+",
    r"^head\s+",
    r"^tail\s+",
    r"^grep\s+",
    r"^find\s+",
    r"^echo\s+",
    r"^pwd$",
    r"^cd\s+",
    r"^mkdir\s+",
    r"^touch\s+",
    r"^cp\s+(?!.*-rf)",  # cp without -rf
    r"^mv\s+",
    r"^docker\s+(ps|images|logs|inspect)",
    r"^kubectl\s+(get|describe|logs)",
]


def check_bash_command(command: str) -> Tuple[str, str]:
    """
    Check if a bash command should be allowed.
    
    Returns:
        Tuple of (decision, reason)
    """
    # Normalize command
    cmd = command.strip()
    cmd_lower = cmd.lower()
    
    # Check kill switch
    from pathlib import Path
    kill_switch = Path.home() / ".claude" / "KILL_SWITCH"
    if kill_switch.exists():
        return "deny", "Kill switch is active - all operations blocked"
    
    # Check blocked commands first
    for pattern in BLOCKED_COMMANDS:
        if re.search(pattern, cmd, re.IGNORECASE):
            return "deny", f"Blocked dangerous command pattern: {pattern}"
    
    # Check explicitly allowed patterns
    for pattern in ALLOWED_PATTERNS:
        if re.match(pattern, cmd, re.IGNORECASE):
            return "allow", "Matched safe command pattern"
    
    # Check warning patterns (allow but log)
    for pattern in WARN_COMMANDS:
        if re.search(pattern, cmd, re.IGNORECASE):
            # Log warning but allow
            import sys
            print(f"WARNING: Potentially risky command: {cmd}", file=sys.stderr)
            return "allow", "Allowed with warning"
    
    # Default: allow unknown commands (user has accepted permissions)
    return "allow", "Default allow for unmatched command"


def main():
    try:
        hook_input = json.load(sys.stdin)
    except json.JSONDecodeError:
        print(json.dumps({"decision": "allow"}))
        return
    
    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})
    
    # Only process Bash tool
    if tool_name != "Bash":
        print(json.dumps({"decision": "allow"}))
        return
    
    command = tool_input.get("command", "")
    decision, reason = check_bash_command(command)
    
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": reason,
        }
    }
    
    print(json.dumps(output))
    
    if decision == "deny":
        sys.exit(2)


if __name__ == "__main__":
    main()
