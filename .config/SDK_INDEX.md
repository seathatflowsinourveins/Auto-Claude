# SDK & Repository Index

> **Location**: `Z:/insider/AUTO CLAUDE/unleash/`
> **Last Updated**: 2026-01-19
> **Structure Version**: 2.0 (Consolidated)

## 📁 Folder Structure

```
unleash/
├── .config/              # Centralized configuration
│   ├── .env              # Environment variables (your API keys)
│   ├── .env.template     # Template for new setups
│   ├── SDK_INDEX.md      # This file
│   └── README.md         # Setup instructions
│
├── sdks/                 # ALL SDKs consolidated
│   ├── anthropic/        # Claude/Anthropic SDKs
│   ├── letta/            # Memory & agent SDKs
│   ├── mcp/              # Model Context Protocol
│   ├── graphiti/         # Knowledge graphs (Zep/Graphiti)
│   ├── exa/              # AI search examples
│   └── claude-flow/      # Multi-agent orchestration
│
├── platform/             # Core platform code
│   ├── core/             # Orchestrator, memory, integrations
│   ├── scripts/          # CLI tools, automation
│   └── tests/            # Test suites
│
├── apps/                 # Applications
│   ├── backend/          # API backend
│   └── frontend/         # Web frontend
│
├── docs/                 # Documentation
│   ├── ULTIMATE_ARCHITECTURE.md
│   ├── INTEGRATION_ROADMAP.md
│   └── ...
│
└── archived/             # Archived/outdated files
```

---

## Quick Reference

| Category | New Path | Purpose |
|----------|----------|---------|
| Config | `unleash/.config/` | All environment config |
| SDKs | `unleash/sdks/` | All SDK repositories |
| Platform | `unleash/platform/` | Core platform code |
| Apps | `unleash/apps/` | Applications |
| Docs | `unleash/docs/` | Documentation |

---

## 1. Anthropic SDKs (`sdks/anthropic/`)

### claude-agent-sdk-python
- **Purpose**: Official Python SDK for building agents
- **Path**: `sdks/anthropic/claude-agent-sdk-python/`
- **Install**: `pip install anthropic-agent-sdk`

### claude-agent-sdk-typescript
- **Purpose**: Official TypeScript SDK for agents
- **Path**: `sdks/anthropic/claude-agent-sdk-typescript/`
- **Install**: `npm install @anthropic-ai/agent-sdk`

### claude-agent-sdk-demos
- **Purpose**: Example agents (email, research)
- **Path**: `sdks/anthropic/claude-agent-sdk-demos/`
- **Examples**: `email-agent/`, `research-agent/`

### claude-cookbooks
- **Purpose**: Recipes and patterns for Claude
- **Path**: `sdks/anthropic/claude-cookbooks/`
- **Key dirs**: `tool_use/`, `skills/`, `third_party/`

### claude-quickstarts
- **Purpose**: Quick start templates
- **Path**: `sdks/anthropic/claude-quickstarts/`

### claude-plugins-official
- **Purpose**: Official plugin implementations
- **Path**: `sdks/anthropic/claude-plugins-official/`
- **Plugins**: asana, context7, firebase, github, gitlab, linear, playwright, serena, slack, stripe, supabase

### claude-code-action
- **Purpose**: GitHub Actions integration
- **Path**: `sdks/anthropic/claude-code-action/`

### anthropic-sdk-typescript
- **Purpose**: Core Anthropic TypeScript SDK
- **Path**: `sdks/anthropic/anthropic-sdk-typescript/`
- **Install**: `npm install @anthropic-ai/sdk`

---

## 2. Letta SDKs (`sdks/letta/`)

### letta (core)
- **Purpose**: Main Letta server and framework
- **Path**: `sdks/letta/letta/`
- **Start**: `letta server --port 8500`

### letta-python
- **Purpose**: Python client SDK
- **Path**: `sdks/letta/letta-python/`
- **Install**: `pip install letta-client`

### letta-node
- **Purpose**: Node.js client SDK
- **Path**: `sdks/letta/letta-node/`
- **Install**: `npm install @letta/client`

### deep-research
- **Purpose**: Research agent with Exa + Firecrawl
- **Path**: `sdks/letta/deep-research/`
- **Run**: `python research.py`

### sleep-time-compute
- **Purpose**: Background memory processing
- **Path**: `sdks/letta/sleep-time-compute/`

### ai-memory-sdk
- **Purpose**: Memory SDK
- **Path**: `sdks/letta/ai-memory-sdk/`

---

## 3. MCP SDKs (`sdks/mcp/`)

### servers
- **Purpose**: Official MCP server implementations
- **Path**: `sdks/mcp/servers/`
- **Key servers**: filesystem, memory, postgres, github, slack

### python-sdk
- **Purpose**: MCP Python SDK
- **Path**: `sdks/mcp/python-sdk/`
- **Install**: `pip install mcp`

### typescript-sdk
- **Purpose**: MCP TypeScript SDK
- **Path**: `sdks/mcp/typescript-sdk/`
- **Install**: `npm install @modelcontextprotocol/sdk`

### inspector
- **Purpose**: MCP debugging tool
- **Path**: `sdks/mcp/inspector/`

---

## 4. Graphiti/Zep (`sdks/graphiti/`)

### graphiti
- **Purpose**: Temporal knowledge graphs
- **Path**: `sdks/graphiti/graphiti/`
- **MCP Server**: `mcp_server/` (ready to integrate!)

### zep
- **Purpose**: Memory layer for LLM apps
- **Path**: `sdks/graphiti/zep/`

### zep-python
- **Purpose**: Python SDK
- **Path**: `sdks/graphiti/zep-python/`
- **Install**: `pip install zep-python`

---

## 5. Claude-Flow (`sdks/claude-flow/`)

### v2
- **Purpose**: Multi-agent orchestration (stable)
- **Path**: `sdks/claude-flow/v2/`
- **Key**: Swarm patterns, coordination

### v3
- **Purpose**: Next-gen orchestration (TypeScript fixed)
- **Path**: `sdks/claude-flow/v3/`
- **Key**: Hierarchical mesh topology

---

## 6. Exa (`sdks/exa/`)

### Examples
- `company-researcher/` - Company research agent
- `exa-deepseek-chat/` - DeepSeek integration
- `exa-hallucination-detector/` - Fact checking

---

## 7. Platform (`platform/`)

### Core Modules
| Module | Purpose |
|--------|---------|
| `core/orchestrator.py` | Multi-agent orchestration |
| `core/advanced_memory.py` | Letta integration |
| `core/firecrawl_integration.py` | Web scraping |
| `core/mcp_discovery.py` | Server discovery |
| `core/resilience.py` | Circuit breaker, retry |
| `core/ultrathink.py` | Extended thinking |

### Scripts
| Script | Purpose |
|--------|---------|
| `scripts/auto_research.py` | Firecrawl + Exa research |
| `scripts/ecosystem_orchestrator.py` | Platform status |
| `scripts/ralph_loop.py` | Autonomous improvement |
| `scripts/auto_validate.py` | Validation suite |

---

## Environment Configuration

**Location**: `unleash/.config/.env`

### Required Keys
```bash
ANTHROPIC_API_KEY=sk-ant-...
FIRECRAWL_API_KEY=fc-...     # ✅ Configured
EXA_API_KEY=...
LETTA_URL=http://localhost:8500
```

### Optional Keys
```bash
GITHUB_TOKEN=ghp_...
OPENAI_API_KEY=sk-...
NEO4J_PASSWORD=...
```

---

## Quick Start

```bash
# 1. Navigate to unleash
cd "Z:/insider/AUTO CLAUDE/unleash"

# 2. Your .env is already configured with Firecrawl
# Add other API keys as needed:
code .config/.env

# 3. Start Letta server (for memory)
cd sdks/letta/letta && letta server --port 8500

# 4. Run platform validation
cd platform && uv run scripts/auto_validate.py

# 5. Test Firecrawl research
cd platform && python scripts/auto_research.py test
```

---

## GitHub Repositories

| SDK | Repository |
|-----|------------|
| Firecrawl | https://github.com/firecrawl/firecrawl |
| Exa | https://github.com/exa-labs |
| Anthropic SDKs | https://github.com/anthropics |
| Letta | https://github.com/letta-ai |
| MCP | https://github.com/modelcontextprotocol |
| Graphiti | https://github.com/getzep/graphiti |
| Claude-Flow | https://github.com/ruvnet/claude-flow |

---

*Generated: 2026-01-19 | Structure Version 2.0*
