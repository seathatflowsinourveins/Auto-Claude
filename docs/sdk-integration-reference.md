# Claude SDK Integration Reference
## Comprehensive Documentation for Autonomous Agentic Systems
**Last Updated**: 2026-01-22 | **Ralph Loop Iteration**: 1

---

## Table of Contents
1. [Claude Code Overview](#1-claude-code-overview)
2. [Tool Use Patterns](#2-tool-use-patterns)
3. [MCP Architecture](#3-mcp-architecture)
4. [Extended Thinking](#4-extended-thinking)
5. [Hooks System](#5-hooks-system)
6. [Integration Patterns](#6-integration-patterns)
7. [Best Practices](#7-best-practices)

---

## 1. Claude Code Overview

### Core Architecture
Claude Code is Anthropic's official CLI tool - an **agentic coding tool** that brings AI-assisted development directly into the terminal.

**Key Characteristics:**
- **Terminal-native**: Works where developers already work
- **Agentic**: Takes direct action (edits files, runs commands, creates commits)
- **Composable**: Follows Unix philosophy - scriptable and pipeable
- **Enterprise-ready**: Support for Claude API, AWS, GCP with built-in security

### Core Capabilities
1. **Build features from descriptions** - Plain English → Implementation plans → Code
2. **Debug and fix issues** - Analyze codebases, identify problems, implement fixes
3. **Navigate codebases** - Full project structure understanding, MCP integrations
4. **Automate tasks** - Lint fixes, merge conflicts, release notes, CI/CD

### Multi-Platform Ecosystem
- CLI (terminal-native)
- Web version (claude.ai/code)
- Desktop app
- IDE integrations (VS Code, JetBrains)
- Browser extension (beta)
- CI/CD integration (GitHub Actions, GitLab CI/CD)
- Slack integration

### Extension Points
1. **MCP (Model Context Protocol)** - Custom data sources & external tools
2. **Skills & Output Styles** - Customize behavior
3. **Plugins** - Add functionality from marketplace
4. **Deployment Options** - Claude API, Amazon Bedrock, Google Vertex AI, Microsoft Foundry

---

## 2. Tool Use Patterns

### Tool Types

#### Client Tools
Execute on your systems:
- User-defined custom tools
- Anthropic-defined tools (computer use, text editor)

```python
# Tool Definition Schema
{
    "name": "get_weather",
    "description": "Get the current weather in a given location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
            }
        },
        "required": ["location"]
    }
}
```

#### Server Tools
Execute on Anthropic's servers:
- `web_search_20250305` - Web search
- `web_fetch_20250305` - Web fetch

### Tool Use Flow

```
1. Provide Claude with tools + user prompt
2. Claude decides to use a tool (stop_reason: "tool_use")
3. Execute tool and return results (tool_result)
4. Claude formulates final response
```

### Parallel Tool Use
Claude can call multiple tools in parallel within a single response:

```json
{
  "content": [
    {"type": "tool_use", "id": "toolu_01...", "name": "get_weather", "input": {...}},
    {"type": "tool_use", "id": "toolu_02...", "name": "get_time", "input": {...}}
  ]
}
```

**Important**: All `tool_result` blocks must be provided in the subsequent user message.

### Sequential Tools
For dependent operations, Claude chains tools using outputs from previous calls:

```
User: "What's the weather where I am?"
→ get_location → "San Francisco, CA"
→ get_weather(location="San Francisco, CA") → "59°F"
→ Final response
```

### Strict Tool Use (Structured Outputs)
Add `strict: true` for guaranteed schema validation:

```json
{
    "name": "tool_name",
    "strict": true,
    "input_schema": {...}
}
```

---

## 3. MCP Architecture

### Client-Server Model

```
┌─────────────────┐
│   MCP Host      │  (Claude Desktop, Claude Code)
│                 │
│  ┌───────────┐  │
│  │MCP Client │──┼──→ MCP Server 1 (tools, resources, prompts)
│  └───────────┘  │
│  ┌───────────┐  │
│  │MCP Client │──┼──→ MCP Server 2 (tools, resources, prompts)
│  └───────────┘  │
└─────────────────┘
```

### Two-Layer Architecture

#### Data Layer (JSON-RPC 2.0)
- Lifecycle management (initialization, capability negotiation, termination)
- Server features (tools, resources, prompts)
- Client features (sampling, elicitation, logging)
- Utility features (notifications, progress tracking)

#### Transport Layer
- **Stdio transport**: Standard I/O for local processes (optimal performance)
- **Streamable HTTP**: HTTP POST with SSE (enables remote communication)

### Capability Negotiation

```json
// Initialize Request
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-06-18",
    "capabilities": {"elicitation": {}},
    "clientInfo": {"name": "example-client", "version": "1.0.0"}
  }
}

// Server Response
{
  "jsonrpc": "2.0",
  "result": {
    "protocolVersion": "2025-06-18",
    "capabilities": {
      "tools": {"listChanged": true},
      "resources": {}
    },
    "serverInfo": {"name": "example-server", "version": "1.0.0"}
  },
  "id": 1
}
```

### Core Primitives

**Server Primitives:**
| Primitive | Discovery | Execution | Purpose |
|-----------|-----------|-----------|---------|
| Tools | `tools/list` | `tools/call` | Executable functions |
| Resources | `resources/list` | `resources/read` | Data sources |
| Prompts | `prompts/list` | `prompts/get` | Reusable templates |

**Client Primitives:**
- **Sampling**: Servers request LLM completions
- **Elicitation**: Servers request user input
- **Logging**: Servers send log messages

### Converting MCP Tools to Claude Format

```python
async def get_claude_tools(mcp_session: ClientSession):
    mcp_tools = await mcp_session.list_tools()
    return [{
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": tool.inputSchema  # Rename inputSchema → input_schema
    } for tool in mcp_tools.tools]
```

---

## 4. Extended Thinking

### Supported Models
- Claude Opus 4.5, 4.1, 4
- Claude Sonnet 4.5, 4, 3.7 (deprecated)
- Claude Haiku 4.5

### Enabling Extended Thinking

```python
response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000  # Min: 1024
    },
    messages=[{"role": "user", "content": "Complex problem..."}]
)
```

### Response Format

```json
{
  "content": [
    {
      "type": "thinking",
      "thinking": "Let me analyze this step by step...",
      "signature": "WaUjzkypQ2mUEVM36O2TxuC06KN8xyfbJwyem2dw3URve..."
    },
    {
      "type": "text",
      "text": "Based on my analysis..."
    }
  ]
}
```

### Summarized Thinking (Claude 4 Models)
- Returns summary of full thinking process
- Charged for full thinking tokens (not summary)
- First lines more verbose for prompt engineering

### Extended Thinking with Tool Use

**Critical Requirements:**
1. Tool choice limited to `auto` or `none` (not `any` or specific tool)
2. **Must preserve thinking blocks** when passing tool results back
3. Cannot toggle thinking mid-turn (entire assistant turn = single mode)

```python
# After first response with thinking + tool_use
continuation = client.messages.create(
    model="claude-opus-4-5",
    thinking={"type": "enabled", "budget_tokens": 10000},
    tools=[weather_tool],
    messages=[
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "content": [thinking_block, tool_use_block]},  # MUST include both
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "...", "content": "..."}]}
    ]
)
```

### Interleaved Thinking (Claude 4 Only)
Enable with beta header: `interleaved-thinking-2025-05-14`

```
Turn 1: [thinking] + [tool_use]
↓ tool result
Turn 2: [thinking] "Got result, now..." + [tool_use]  ← thinking BETWEEN tool calls
↓ tool result
Turn 3: [thinking] + [text]
```

**Benefits:**
- Reason about intermediate results
- Chain tools with reasoning steps between
- `budget_tokens` can exceed `max_tokens`

### Budget Recommendations
- **Starting point**: 16k+ tokens for complex tasks
- **Above 32k**: Use batch processing to avoid network timeouts
- **Minimum**: 1,024 tokens

---

## 5. Hooks System

### Hook Lifecycle

```
SessionStart → UserPromptSubmit → PreToolUse → PermissionRequest →
PostToolUse → Stop → SubagentStop → PreCompact → SessionEnd
```

### Hook Types

#### Command Hooks
```json
{
  "type": "command",
  "command": "python validate.py",
  "timeout": 60
}
```

#### Prompt-Based Hooks (Stop/SubagentStop only)
```json
{
  "type": "prompt",
  "prompt": "Evaluate if Claude should stop: $ARGUMENTS",
  "timeout": 30
}
```

### Hook Events Reference

| Event | Trigger | Use Case |
|-------|---------|----------|
| **PreToolUse** | Before tool execution | Validate/modify params, approve/deny |
| **PermissionRequest** | Permission dialog | Auto-approve/deny permissions |
| **PostToolUse** | After tool execution | Validate results, feedback |
| **UserPromptSubmit** | User submits prompt | Validate, add context |
| **Stop** | Claude finishes | Continue/stop decision |
| **SubagentStop** | Subagent completes | Continue/stop decision |
| **SessionStart** | Session begins | Load context, set env vars |
| **PreCompact** | Before compaction | Pre-compact operations |
| **SessionEnd** | Session terminates | Cleanup, logging |

### Hook Configuration

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash|Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "python security_check.py",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

### Hook Input/Output

**Input (JSON via stdin):**
```json
{
  "session_id": "string",
  "transcript_path": "path/to/transcript.jsonl",
  "cwd": "current/working/directory",
  "tool_name": "Bash",
  "tool_input": {"command": "...", "description": "..."}
}
```

**Output (Exit codes):**
- **0**: Success (parse stdout for JSON control)
- **2**: Blocking error (show stderr)
- **Other**: Non-blocking error

**Control Output:**
```json
{
  "continue": true,
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "allow|deny|ask",
    "updatedInput": {"field": "new_value"},
    "additionalContext": "context for claude"
  }
}
```

### Environment Variables
- `CLAUDE_PROJECT_DIR` - Project root
- `CLAUDE_ENV_FILE` - Persist env vars (SessionStart only)
- `CLAUDE_CODE_REMOTE` - "true" for web
- `${CLAUDE_PLUGIN_ROOT}` - Plugin directory

---

## 6. Integration Patterns

### Agentic Core Loop
```
while task_incomplete:
    1. Gather context (search, read, explore)
    2. Take action (tool use)
    3. Verify work (tests, checks)
    4. Repeat or complete
```

### Subagents for Parallelization
- Each subagent has isolated context window
- Use for independent, parallelizable tasks
- Coordinate via shared filesystem/database

### Compaction for Long Sessions
- Automatic summarization after context fills
- Preserves key decisions and state
- ~10-15 minute sessions before compaction

### Memory Integration Patterns

**CLAUDE.md Hierarchy:**
1. `~/.claude/CLAUDE.md` - Global user instructions
2. `project/CLAUDE.md` - Project instructions (checked in)
3. `project/CLAUDE.local.md` - Private project instructions

**Cross-Session Memory:**
- Serena MCP memories (`read_memory`, `write_memory`)
- Episodic memory (conversation history search)
- Claude-mem (observation persistence)

---

## 7. Best Practices

### Tool Design
1. Clear, concise descriptions
2. Well-documented parameters with examples
3. Strict schemas for production (`strict: true`)
4. Error handling in tool implementations

### Extended Thinking
1. Start with minimum budget (1,024), increase incrementally
2. Use batch processing for >32k budgets
3. Always preserve thinking blocks in tool use flows
4. Plan thinking strategy at turn start (no mid-turn toggle)

### Hooks
1. Validate all stdin input
2. Quote shell variables
3. Use absolute paths or `$CLAUDE_PROJECT_DIR`
4. Skip sensitive files (.env, .git/, keys)
5. Set appropriate timeouts

### Performance
1. Use parallel tool calls for independent operations
2. Stream extended thinking responses
3. Cache system prompts (preserved when thinking changes)
4. Monitor token usage for cost optimization

### Security
1. Block dangerous commands (rm -rf, etc.)
2. Validate file paths (no traversal)
3. Auto-approve only safe operations (reads, docs)
4. Use hooks for guardrails

---

## Official Resources

| Resource | URL |
|----------|-----|
| Claude Code Docs | https://code.claude.com/docs/en/overview |
| Tool Use | https://platform.claude.com/docs/en/docs/build-with-claude/tool-use |
| Extended Thinking | https://platform.claude.com/docs/en/docs/build-with-claude/extended-thinking |
| MCP Specification | https://modelcontextprotocol.io/specification/latest |
| Hooks Reference | https://code.claude.com/docs/en/hooks |
| Agent SDK Python | https://github.com/anthropics/claude-agent-sdk-python |
| Agent SDK TypeScript | https://github.com/anthropics/claude-agent-sdk-typescript |

---

*Generated by Ralph Loop Iteration 1 - Deep SDK Documentation Research Phase*
