# Unleash Platform - Centralized Configuration

> **All SDKs, repos, and environment configuration in one place**

## Directory Structure

```
.config/
├── README.md           # This file
├── .env.template       # Master environment template (copy to .env)
├── .env                # Your actual secrets (gitignored)
└── SDK_INDEX.md        # Complete inventory of all SDKs and repos
```

## Quick Setup

### 1. Create your `.env` file

```bash
cp .env.template .env
```

### 2. Edit with your API keys

**Required keys for core functionality:**
```
ANTHROPIC_API_KEY=sk-ant-...
FIRECRAWL_API_KEY=fc-...
EXA_API_KEY=...
```

**Optional but recommended:**
```
GITHUB_TOKEN=ghp_...
LETTA_URL=http://localhost:8500
```

### 3. Load environment

**PowerShell:**
```powershell
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}
```

**Bash:**
```bash
export $(grep -v '^#' .env | xargs)
```

## SDK Locations

| Category | Path | Contents |
|----------|------|----------|
| Core Platform | `../v10_optimized/` | Orchestrator, memory, Firecrawl |
| Anthropic SDKs | `../anthropic/` | Agent SDK, cookbooks, plugins |
| Letta | `../letta-ai/` | Memory server, clients, research |
| MCP | `../mcp-official/` | Protocol SDKs, servers |
| Graphiti | `../zep-graphiti/` | Knowledge graphs, MCP server |
| Claude-Flow | `../ruvnet-claude-flow/` | Multi-agent orchestration |

## MCP Configuration

My MCP config is at: `C:/Users/42/.claude/.mcp.json`

**Configured servers:**
- `filesystem` - File operations
- `memory` - Key-value store
- `sequential-thinking` - Extended reasoning
- `letta` - Long-term memory
- `firecrawl` - Web scraping/crawling
- `exa` - AI search
- `github` - Repository operations
- `postgres` - Database
- `brave-search` - Web search
- `slack` - Notifications

## Global Shortcuts

Available in Claude Code:
- `platform-status` - Check ecosystem health
- `ralph-iterate` - Run improvement loop
- `validate` - Run validation suite
- `research` - Run auto-research pipeline
- `sdk-index` - View SDK inventory

## Files to Gitignore

Add to your `.gitignore`:
```
.config/.env
*.env
!*.env.template
!*.env.example
```

---

*Centralized config created: 2026-01-19*
