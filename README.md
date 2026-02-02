# 🚀 Unleash Platform

> **Unified AI Development Platform** - All SDKs, tools, and configuration in one place

## Quick Start

```bash
# 1. Your Firecrawl API key is already configured
# Add other API keys:
code .config/.env

# 2. Start Letta memory server
cd sdks/letta/letta && letta server --port 8500

# 3. Run auto-research
cd platform && python scripts/auto_research.py research "your topic" --sources 10
```

## 📁 Structure

```
unleash/
├── .config/          # ⚙️ Configuration (API keys, settings)
├── sdks/             # 📦 All SDKs consolidated
│   ├── anthropic/    #    Claude SDKs, cookbooks, plugins
│   ├── letta/        #    Memory server, clients
│   ├── mcp/          #    Model Context Protocol
│   ├── graphiti/     #    Knowledge graphs
│   ├── exa/          #    AI search examples
│   └── claude-flow/  #    Multi-agent orchestration
├── platform/         # 🎯 Core platform code
├── apps/             # 📱 Applications
├── docs/             # 📚 Documentation
└── archived/         # 📦 Archived files
```

## Key Locations

| What | Where |
|------|-------|
| API Keys | `.config/.env` |
| SDK Index | `.config/SDK_INDEX.md` |
| Platform Core | `platform/core/` |
| Auto-Research | `platform/scripts/auto_research.py` |
| Letta Server | `sdks/letta/letta/` |

## Configured Services

| Service | Status | API Key |
|---------|--------|---------|
| Firecrawl | ✅ Ready | `fc-ba99d5b...` |
| Exa | ⚠️ Needs key | `.config/.env` |
| Letta | ✅ localhost:8500 | - |
| Anthropic | ⚠️ Needs key | `.config/.env` |

## Commands

```bash
# Research a topic (Exa + Firecrawl)
python platform/scripts/auto_research.py research "AI agents 2026"

# Scrape a single URL
python platform/scripts/auto_research.py scrape https://example.com

# Crawl a website
python platform/scripts/auto_research.py crawl https://docs.example.com --depth 2

# Run validation
python platform/scripts/auto_validate.py

# Ralph Loop iteration
python platform/scripts/ralph_loop.py iterate
```

---

*Organized: 2026-01-19*
