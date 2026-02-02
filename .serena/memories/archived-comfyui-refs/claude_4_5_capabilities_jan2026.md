# Claude 4.5 Complete Capabilities Reference (January 2026)

## Extended Thinking (ULTRATHINK)

### Core Parameters
```python
response = client.messages.create(
    model="claude-opus-4-5-20251101",  # or claude-sonnet-4-5
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000  # Minimum 1,024, recommend 16k+ for complex tasks
    },
    messages=[{"role": "user", "content": "Complex question..."}]
)
```

### Key Behaviors
- **Summarized Thinking**: Claude 4 models return summaries, billed for full tokens
- **Budget vs Max Tokens**: `budget_tokens` can exceed `max_tokens` with interleaved thinking
- **Thinking Preservation**: Claude Opus 4.5+ preserves thinking blocks by default
- **Prompt Caching**: Works with extended thinking, thinking output not cached

### Interleaved Thinking (Beta)
```python
# Add beta header for tool use with thinking
response = client.messages.create(
    model="claude-opus-4-5-20251101",
    extra_headers={"anthropic-beta": "interleaved-thinking-2025-05-14"},
    thinking={"type": "enabled", "budget_tokens": 10000},
    tools=[...],
    messages=[...]
)
```

### Tool Use with Thinking - CRITICAL Pattern
```python
# MUST preserve thinking blocks when continuing tool use conversations
thinking_block = next((b for b in response.content if b.type == 'thinking'), None)
tool_use_block = next((b for b in response.content if b.type == 'tool_use'), None)

continuation = client.messages.create(
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[
        {"role": "user", "content": "Original question"},
        {"role": "assistant", "content": [thinking_block, tool_use_block]},  # MUST include both!
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use_block.id, "content": "..."}]}
    ]
)
```

### Budget Token Guidelines
| Task Complexity | Recommended Budget |
|-----------------|-------------------|
| Simple Q&A | 1,024 - 4,000 |
| Code review | 8,000 - 16,000 |
| Complex debugging | 16,000 - 32,000 |
| Architecture design | 32,000 - 64,000 |
| Deep research | 64,000 - 128,000 |

---

## Claude Agent SDK (Production-Grade Agents)

### Installation
```bash
# Claude Code runtime required
curl -fsSL https://claude.ai/install.sh | bash  # macOS/Linux
winget install Anthropic.ClaudeCode  # Windows

# SDK packages
pip install claude-agent-sdk  # Python
npm install @anthropic-ai/claude-agent-sdk  # TypeScript
```

### Core Query Pattern
```python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions

async def main():
    async for message in query(
        prompt="Find and fix the bug in auth.py",
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Edit", "Bash", "Glob", "Grep"],
            permission_mode="acceptEdits"
        )
    ):
        if hasattr(message, "result"):
            print(message.result)

asyncio.run(main())
```

### Built-in Tools
| Tool | Purpose |
|------|---------|
| **Read** | Read any file in working directory |
| **Write** | Create new files |
| **Edit** | Make precise edits to existing files |
| **Bash** | Run terminal commands, scripts, git |
| **Glob** | Find files by pattern (`**/*.ts`) |
| **Grep** | Search file contents with regex |
| **WebSearch** | Search the web |
| **WebFetch** | Fetch and parse web content |
| **AskUserQuestion** | Clarifying questions with options |

### Permission Modes
| Mode | Behavior | Use Case |
|------|----------|----------|
| `acceptEdits` | Auto-approves file edits | Trusted workflows |
| `bypassPermissions` | Runs without prompts | CI/CD pipelines |
| `default` | Requires `canUseTool` callback | Custom approval flows |

### Hooks System
```python
from claude_agent_sdk import query, ClaudeAgentOptions, HookMatcher

async def log_file_change(input_data, tool_use_id, context):
    file_path = input_data.get('tool_input', {}).get('file_path', 'unknown')
    with open('./audit.log', 'a') as f:
        f.write(f"{datetime.now()}: modified {file_path}\n")
    return {}

async for message in query(
    prompt="Refactor utils.py",
    options=ClaudeAgentOptions(
        permission_mode="acceptEdits",
        hooks={
            "PostToolUse": [HookMatcher(matcher="Edit|Write", hooks=[log_file_change])]
        }
    )
):
    pass
```

**Available Hooks**: `PreToolUse`, `PostToolUse`, `Stop`, `SessionStart`, `SessionEnd`, `UserPromptSubmit`

### Subagents (Multi-Agent Orchestration)
```python
from claude_agent_sdk import query, ClaudeAgentOptions, AgentDefinition

async for message in query(
    prompt="Use the code-reviewer agent to review this codebase",
    options=ClaudeAgentOptions(
        allowed_tools=["Read", "Glob", "Grep", "Task"],  # Task enables subagents
        agents={
            "code-reviewer": AgentDefinition(
                description="Expert code reviewer for quality and security.",
                prompt="Analyze code quality and suggest improvements.",
                tools=["Read", "Glob", "Grep"]
            ),
            "security-auditor": AgentDefinition(
                description="Security vulnerability scanner.",
                prompt="Find OWASP Top 10 vulnerabilities.",
                tools=["Read", "Glob", "Grep"]
            )
        }
    )
):
    if hasattr(message, "result"):
        print(message.result)
```

### MCP Server Integration
```python
async for message in query(
    prompt="Open example.com and describe what you see",
    options=ClaudeAgentOptions(
        mcp_servers={
            "playwright": {"command": "npx", "args": ["@playwright/mcp@latest"]},
            "filesystem": {"command": "npx", "args": ["@anthropic/mcp-server-filesystem"]},
            "postgres": {"command": "npx", "args": ["@anthropic/mcp-server-postgres"]}
        }
    )
):
    pass
```

### Session Management (Context Persistence)
```python
# First query - capture session ID
session_id = None
async for message in query(
    prompt="Read the authentication module",
    options=ClaudeAgentOptions(allowed_tools=["Read", "Glob"])
):
    if hasattr(message, 'subtype') and message.subtype == 'init':
        session_id = message.session_id

# Resume with full context
async for message in query(
    prompt="Now find all places that call it",  # "it" = auth module from context
    options=ClaudeAgentOptions(resume=session_id)
):
    if hasattr(message, "result"):
        print(message.result)
```

### Claude Code Configuration Integration
```python
# Use project-level Claude Code configs
options=ClaudeAgentOptions(
    setting_sources=["project"],  # Enables .claude/ directory configs
    # Now agent has access to:
    # - Skills from .claude/skills/SKILL.md
    # - Commands from .claude/commands/*.md
    # - Memory from CLAUDE.md
)
```

---

## Multi-Agent Architecture Patterns (2026 Best Practices)

### Pattern 1: Minimal Tool Architecture
From Vercel research - frontier models prefer fewer, more powerful tools:
```python
# Instead of 20 custom tools, give agent:
options=ClaudeAgentOptions(
    allowed_tools=["Bash", "Read", "Write", "Edit"],  # Just 4 tools
    permission_mode="acceptEdits"
)
# Claude figures out how to compose these for complex tasks
```

### Pattern 2: Specialist Subagent Swarm
```python
agents={
    "planner": AgentDefinition(
        description="Creates implementation plans",
        prompt="Break down complex tasks into actionable steps",
        tools=["Read", "Glob"]
    ),
    "implementer": AgentDefinition(
        description="Writes production code",
        prompt="Implement features following project patterns",
        tools=["Read", "Edit", "Write", "Bash"]
    ),
    "reviewer": AgentDefinition(
        description="Reviews code quality",
        prompt="Find bugs, security issues, style violations",
        tools=["Read", "Glob", "Grep"]
    ),
    "tester": AgentDefinition(
        description="Writes and runs tests",
        prompt="Create comprehensive test coverage",
        tools=["Read", "Edit", "Bash"]
    )
}
```

### Pattern 3: Context Engineering for Multi-Agent
From LangChain research (Jan 2026):
- **Context Isolation**: Each agent gets only relevant context
- **Handoff Protocols**: Structured message passing between agents
- **Shared Memory**: Use MCP servers for cross-agent state

### Pattern 4: Hierarchical Orchestration
```
Orchestrator Agent (claude-opus-4-5)
├── Planning Phase
│   └── planner-agent (claude-haiku-4-5) - cost-efficient
├── Implementation Phase
│   └── implementer-agent (claude-sonnet-4-5) - balanced
├── Review Phase
│   └── reviewer-agent (claude-opus-4-5) - thorough
└── Verification Phase
    └── tester-agent (claude-sonnet-4-5) - balanced
```

---

## Project-Specific Integration

### WITNESS (Creative)
```python
# Real-time creative exploration
options=ClaudeAgentOptions(
    allowed_tools=["Read", "Edit", "Bash"],
    mcp_servers={
        "touchdesigner": {"command": "python", "args": ["td-mcp-server.py"]},
        "comfyui": {"command": "npx", "args": ["comfyui-mcp"]}
    },
    agents={
        "shader-generator": AgentDefinition(
            description="GLSL 4.60+ shader expert",
            prompt="Generate TouchDesigner-compatible shaders",
            tools=["Read", "Edit"]
        )
    }
)
```

### TRADING (AlphaForge)
```python
# Development lifecycle orchestration (NOT hot path)
options=ClaudeAgentOptions(
    allowed_tools=["Read", "Edit", "Bash", "Glob", "Grep"],
    permission_mode="default",  # Require approval for trading code
    agents={
        "risk-validator": AgentDefinition(
            description="Trading risk limit validator",
            prompt="Verify position sizing, circuit breakers, daily limits",
            tools=["Read", "Grep"]
        ),
        "security-auditor": AgentDefinition(
            description="Trading security specialist",
            prompt="Find injection, authentication, authorization issues",
            tools=["Read", "Glob", "Grep"]
        )
    }
)
```

### UNLEASH (Meta-Platform)
```python
# Full capability access for meta-development
options=ClaudeAgentOptions(
    allowed_tools=["Read", "Edit", "Write", "Bash", "Glob", "Grep", "WebSearch", "WebFetch", "Task"],
    setting_sources=["project"],  # Use all Claude Code configs
    mcp_servers={
        "memory": {"command": "npx", "args": ["@anthropic/mcp-server-memory"]},
        "github": {"command": "npx", "args": ["@anthropic/mcp-server-github"]}
    }
)
```

---

## Cost Optimization

### Model Selection by Task
| Model | Cost (M tokens) | Best For |
|-------|-----------------|----------|
| claude-opus-4-5 | $5/$25 | Complex reasoning, architecture |
| claude-sonnet-4-5 | $3/$15 | Balanced coding tasks |
| claude-haiku-4-5 | $0.25/$1.25 | Fast iteration, subagents |

### Subagent Cost Strategy
- Use **haiku** for high-frequency subagents (validators, formatters)
- Use **sonnet** for implementation subagents
- Reserve **opus** for orchestrator and critical decisions

---

## Authentication Options
- **Anthropic API**: `ANTHROPIC_API_KEY=...`
- **Amazon Bedrock**: `CLAUDE_CODE_USE_BEDROCK=1` + AWS credentials
- **Google Vertex AI**: `CLAUDE_CODE_USE_VERTEX=1` + GCP credentials
- **Microsoft Azure**: `CLAUDE_CODE_USE_FOUNDRY=1` + Azure credentials

---

*Generated: 2026-01-25*
*Sources: platform.claude.com, Anthropic news, Exa research*
