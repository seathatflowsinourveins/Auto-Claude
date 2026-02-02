# Official Documentation & SDK Index

> **Downloaded**: January 17, 2026
> **Purpose**: Local copies of all official documentation and SDKs for V10.1 system

---

## Quick Reference

| Component | Version | Location |
|-----------|---------|----------|
| MCP Python SDK | `mcp` (latest) | `sdks/mcp-python/` |
| MCP TypeScript SDK | `@modelcontextprotocol/sdk` v2.0.0-alpha.0 | `sdks/mcp-typescript/` |
| MCP Go SDK | `github.com/modelcontextprotocol/go-sdk` | `sdks/mcp-go/` |
| MCP Reference Servers | 7 servers | `sdks/mcp-servers/` |
| Letta Python SDK | `letta-client` v1.7.1 | `sdks/letta-python/` |
| Letta Server | `letta` v0.16.2 | `sdks/letta-server/` |
| MCP Specification | 2025-11-25 | `specs/mcp/` |
| Claude Code Hooks | v2.0.10+ | `specs/claude-code/` |
| Letta API | Latest | `specs/letta/` |

---

## SDKs Directory

### MCP Python SDK
```
sdks/mcp-python/
├── src/mcp/           # Main SDK source
├── examples/          # Usage examples
├── docs/              # API documentation
└── pyproject.toml     # Package: mcp
```

**Key imports:**
```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
```

### MCP TypeScript SDK
```
sdks/mcp-typescript/
├── packages/          # Monorepo packages
│   ├── sdk/           # Main SDK
│   └── client-shared/ # Shared client code
├── examples/          # Usage examples
└── package.json       # @modelcontextprotocol/sdk v2.0.0-alpha.0
```

**Key imports:**
```typescript
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
```

### MCP Go SDK
```
sdks/mcp-go/
├── mcp/               # Core MCP types and interfaces
├── client/            # Client implementation
├── server/            # Server implementation
└── go.mod             # Module: github.com/modelcontextprotocol/go-sdk
```

### MCP Reference Servers
```
sdks/mcp-servers/src/
├── everything/        # All-in-one demo server
├── fetch/             # HTTP fetch server
├── filesystem/        # File operations
├── git/               # Git operations
├── memory/            # Knowledge graph memory
├── sequentialthinking/ # Extended reasoning
└── time/              # Time utilities
```

### Letta Python SDK
```
sdks/letta-python/
├── src/letta_client/  # Main SDK source
│   ├── agents/        # Agent management
│   ├── blocks/        # Memory blocks
│   ├── tools/         # Tool management
│   └── types/         # Type definitions (SleeptimeManagerUpdate, etc.)
├── examples/          # Usage examples
└── pyproject.toml     # Package: letta-client v1.7.1
```

**Key imports:**
```python
from letta_client import Letta
from letta_client.types import SleeptimeManagerUpdate
```

### Letta Server
```
sdks/letta-server/
├── letta/             # Server source code
│   ├── agent.py       # Agent implementation
│   ├── memory.py      # Memory system
│   └── server/        # REST API server
├── examples/          # Example configurations
└── pyproject.toml     # Package: letta v0.16.2
```

---

## Specifications Directory

### MCP Specification
```
specs/mcp/
├── docs/              # Specification documentation
│   └── specification/ # Protocol versions
│       └── 2025-11-25/
│           ├── basic/     # Lifecycle, transports
│           ├── client/    # Sampling, roots
│           └── server/    # Tools, resources, prompts
├── schema/            # JSON Schema files
└── README.md          # Overview
```

**Key specification files:**
- `docs/specification/2025-11-25/basic/lifecycle.mdx` - Protocol lifecycle
- `docs/specification/2025-11-25/server/tools.mdx` - Tool definitions
- `docs/specification/2025-11-25/client/sampling.mdx` - LLM sampling

### Claude Code Hooks
```
specs/claude-code/
└── HOOKS_REFERENCE.md  # Complete hooks documentation
```

**Hook Events:**
- `SessionStart`, `SessionEnd` - Session lifecycle
- `PreToolUse`, `PostToolUse` - Tool call interception
- `UserPromptSubmit` - Prompt validation
- `Stop`, `SubagentStop` - Completion control
- `Notification`, `PreCompact` - Misc events

**Permission Decisions (PreToolUse):**
- `"allow"` - Approve (optionally with `updatedInput`)
- `"deny"` - Block the tool call
- `"ask"` - Request user confirmation

### Letta API
```
specs/letta/
└── API_REFERENCE.md    # API and Sleeptime documentation
```

**Key Sleeptime configuration:**
```python
# Create agent with sleeptime
agent = client.agents.create(
    name="my-agent",
    enable_sleeptime=True
)

# Configure frequency
from letta_client.types import SleeptimeManagerUpdate
client.groups.update(
    group_id=group_id,
    manager_config=SleeptimeManagerUpdate(sleeptime_agent_frequency=5)
)
```

---

## Online Documentation

| Resource | URL |
|----------|-----|
| MCP Specification | https://modelcontextprotocol.io/specification/2025-11-25 |
| MCP Python SDK | https://github.com/modelcontextprotocol/python-sdk |
| MCP TypeScript SDK | https://github.com/modelcontextprotocol/typescript-sdk |
| Claude Code Hooks | https://code.claude.com/docs/en/hooks |
| Letta API | https://docs.letta.com/api-reference |
| Letta Sleeptime | https://docs.letta.com/guides/agents/architectures/sleeptime |

---

## Usage in V10.1

The V10.1 hooks system uses patterns from these SDKs:

### hook_utils.py
Uses Claude Code hooks specification for:
- `HookInput.from_stdin()` - Parse JSON input
- `HookResponse.to_json()` - Generate correct output format
- `PermissionDecision` enum - `ALLOW`, `DENY`, `ASK`

### letta_sync_v2.py
Uses Letta Python SDK patterns:
```python
from letta_client import Letta
from letta_client.types import SleeptimeManagerUpdate

client = Letta(base_url="http://localhost:8283")
agent = client.agents.create(enable_sleeptime=True)
```

### mcp_guard_v2.py
Implements MCP security best practices:
- Input validation before tool execution
- Path sanitization for filesystem operations
- Audit logging for compliance

---

*Last updated: January 17, 2026*
