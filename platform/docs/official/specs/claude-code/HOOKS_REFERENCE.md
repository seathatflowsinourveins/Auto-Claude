# Claude Code Hooks Reference - Complete Documentation

> **Source**: https://code.claude.com/docs/en/hooks
> **Downloaded**: January 17, 2026

## Overview

Claude Code hooks are automated scripts that run at specific events during your Claude Code session. They allow you to validate commands, approve/deny tool usage, inject context, and perform cleanup tasks.

## Configuration

Hooks are configured in settings files:
- `~/.claude/settings.json` - User settings
- `.claude/settings.json` - Project settings
- `.claude/settings.local.json` - Local project settings
- Managed policy settings (enterprise)

### Structure

```json
{
  "hooks": {
    "EventName": [
      {
        "matcher": "ToolPattern",
        "hooks": [
          {
            "type": "command",
            "command": "your-command-here"
          }
        ]
      }
    ]
  }
}
```

**Key fields:**
- `matcher`: Pattern to match tool names (case-sensitive, supports regex like `Edit|Write` or `*` for all)
- `hooks`: Array of hook definitions
- `type`: `"command"` for bash or `"prompt"` for LLM-based evaluation
- `command`: Bash command to execute
- `prompt`: LLM prompt (for prompt-based hooks)
- `timeout`: Optional timeout in seconds (default: 60)

## Hook Events

### PreToolUse
Runs after Claude creates tool parameters and before processing the tool call.

**Common matchers:**
- `Bash`, `Task`, `Glob`, `Grep`, `Read`, `Edit`, `Write`, `WebFetch`, `WebSearch`

**Output control:**
- `"allow"` - Auto-approve with optional `updatedInput` to modify parameters
- `"deny"` - Block the tool call
- `"ask"` - Request user confirmation

### PermissionRequest
Runs when a permission dialog is shown to the user. Allows auto-approval or denial.

### PostToolUse
Runs immediately after a tool completes successfully.

**Output control:**
- `"block"` - Prompts Claude with reason
- `undefined` - No action

### Notification
Runs when Claude Code sends notifications.

**Matchers:**
- `permission_prompt`, `idle_prompt`, `auth_success`, `elicitation_dialog`

### UserPromptSubmit
Runs when the user submits a prompt, before Claude processes it.

**Use cases:** Add context, validate prompts, block sensitive inputs

### Stop
Runs when the main Claude Code agent finishes responding.

**Output control:**
- `"block"` - Prevent stopping, requires `reason` for Claude

### SubagentStop
Runs when a subagent (Task tool) finishes responding.

### PreCompact
Runs before compacting the context window.

**Matchers:** `manual`, `auto`

### SessionStart
Runs when a session starts or resumes.

**Matchers:** `startup`, `resume`, `clear`, `compact`

**Special feature:** Access to `CLAUDE_ENV_FILE` environment variable for persisting environment variables:

```bash
#!/bin/bash
if [ -n "$CLAUDE_ENV_FILE" ]; then
  echo 'export NODE_ENV=production' >> "$CLAUDE_ENV_FILE"
  echo 'export API_KEY=your-api-key' >> "$CLAUDE_ENV_FILE"
fi
exit 0
```

### SessionEnd
Runs when a session ends.

**Reason field values:** `clear`, `logout`, `prompt_input_exit`, `other`

## Hook Input

All hooks receive JSON via stdin:

```json
{
  "session_id": "string",
  "transcript_path": "string",
  "cwd": "string",
  "permission_mode": "string",
  "hook_event_name": "string",
  "tool_name": "string",
  "tool_input": {},
  "tool_use_id": "string"
}
```

### Tool-Specific Input Examples

**Bash:**
```json
{
  "tool_name": "Bash",
  "tool_input": {
    "command": "psql -c 'SELECT * FROM users'",
    "description": "Query the users table",
    "timeout": 120000
  }
}
```

**Write:**
```json
{
  "tool_name": "Write",
  "tool_input": {
    "file_path": "/path/to/file.txt",
    "content": "file content"
  }
}
```

**Edit:**
```json
{
  "tool_name": "Edit",
  "tool_input": {
    "file_path": "/path/to/file.txt",
    "old_string": "original text",
    "new_string": "replacement text",
    "replace_all": false
  }
}
```

**Read:**
```json
{
  "tool_name": "Read",
  "tool_input": {
    "file_path": "/path/to/file.txt",
    "offset": 0,
    "limit": 10
  }
}
```

## Hook Output

### Exit Codes

- **Exit 0**: Success. Stdout shown in verbose mode (except `UserPromptSubmit`/`SessionStart` where it's added as context)
- **Exit 2**: Blocking error. Only stderr used as error message
- **Other codes**: Non-blocking error. Stderr shown in verbose mode

### JSON Output Format

Hooks can return structured JSON in stdout (exit code 0 only):

```json
{
  "continue": true,
  "stopReason": "string",
  "suppressOutput": true,
  "systemMessage": "string",
  "hookSpecificOutput": {
    "hookEventName": "string"
  }
}
```

### PreToolUse JSON Output

```json
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "allow|deny|ask",
    "permissionDecisionReason": "string",
    "updatedInput": { "field": "value" },
    "additionalContext": "string"
  }
}
```

### PermissionRequest JSON Output

```json
{
  "hookSpecificOutput": {
    "hookEventName": "PermissionRequest",
    "decision": {
      "behavior": "allow|deny",
      "updatedInput": { "field": "value" },
      "message": "string",
      "interrupt": false
    }
  }
}
```

### PostToolUse JSON Output

```json
{
  "decision": "block" | undefined,
  "reason": "string",
  "hookSpecificOutput": {
    "hookEventName": "PostToolUse",
    "additionalContext": "string"
  }
}
```

### UserPromptSubmit JSON Output

```json
{
  "decision": "block" | undefined,
  "reason": "string",
  "hookSpecificOutput": {
    "hookEventName": "UserPromptSubmit",
    "additionalContext": "string"
  }
}
```

**Plain text alternative:** Simply print text to stdout with exit code 0 to add context.

### Stop/SubagentStop JSON Output

```json
{
  "decision": "block" | undefined,
  "reason": "string"
}
```

### SessionStart JSON Output

```json
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": "string"
  }
}
```

## Prompt-Based Hooks

Use LLM evaluation instead of bash for complex decisions:

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "prompt",
            "prompt": "Evaluate if Claude should stop: $ARGUMENTS. Check if all tasks are complete.",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

**Response schema:**
```json
{
  "ok": true | false,
  "reason": "Explanation for the decision"
}
```

**Supported events:** `Stop`, `SubagentStop`, `UserPromptSubmit`, `PreToolUse`, `PermissionRequest`

## Component-Scoped Hooks

Hooks can be defined in Skills, Agents, and Slash Commands using frontmatter:

```yaml
---
name: secure-operations
description: Perform operations with security checks
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./scripts/security-check.sh"
---
```

**Supported events:** `PreToolUse`, `PostToolUse`, `Stop`

**Additional option:** `once: true` to run hook only once per session (skills and slash commands only)

## MCP Tool Integration

MCP tools follow the pattern: `mcp__<server>__<tool>`

Examples:
- `mcp__memory__create_entities`
- `mcp__filesystem__read_file`
- `mcp__github__search_repositories`

Match MCP tools in hooks:
```json
{
  "matcher": "mcp__memory__.*"
}
```

## Hook Execution Details

- **Timeout**: 60 seconds default (configurable per command)
- **Parallelization**: All matching hooks run in parallel
- **Deduplication**: Identical commands are automatically deduplicated
- **Environment variables:**
  - `CLAUDE_PROJECT_DIR` - Project root directory
  - `CLAUDE_CODE_REMOTE` - "true" for web, empty for CLI
  - `CLAUDE_ENV_FILE` - File to persist environment variables (SessionStart only)

## Security Best Practices

1. **Validate and sanitize inputs** - Never trust input data blindly
2. **Always quote shell variables** - Use `"$VAR"` not `$VAR`
3. **Block path traversal** - Check for `..` in file paths
4. **Use absolute paths** - Specify full paths for scripts
5. **Skip sensitive files** - Avoid `.env`, `.git/`, keys, etc.

**Disclaimer:** Hooks execute arbitrary shell commands automatically. You are solely responsible for the commands you configure. Hooks can modify, delete, or access any files your user account can access.

## Debugging

Enable debug mode with:
```bash
claude --debug
```

**Basic troubleshooting:**
1. Check configuration with `/hooks` command
2. Verify JSON syntax in settings
3. Test commands manually first
4. Ensure scripts are executable
5. Review logs with `--debug` flag
