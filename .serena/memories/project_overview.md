# Unleash Platform - Project Overview

## Purpose
Unleash is an autonomous AI development platform (V10 ULTRAMAX) providing:
- Unified research pipelines with 7+ SDKs
- Cross-session memory persistence
- MCP server ecosystem integration
- Ralph Loop autonomous iteration

## Tech Stack
- **Language**: Python 3.12+
- **Package Manager**: uv (with pyproject.toml)
- **Type Checking**: Pyright (strict mode)
- **Testing**: pytest with markers
- **Linting**: ruff

## Directory Structure
```
unleash/
├── platform/           # Core platform code
│   ├── core/          # Research engine, SDK integrations
│   ├── cli/           # Command line interface
│   └── sdks/          # SDK integrations (serena, exa, etc.)
├── v10_optimized/     # V10 optimized scripts
│   └── scripts/       # Ralph loop, orchestrator
├── auto-claude/       # Auto-Claude configurations
├── ruvnet-claude-flow/ # Claude Flow v3 integration
├── .serena/           # Serena project config
├── .config/           # Environment configuration
├── docs/              # Documentation
└── tests/             # Test suites
```

## Key Components
1. **Research Engine v3.0** - Multi-source research with Exa, Firecrawl
2. **SDK Ecosystem** - Crawl4AI, LightRAG, LlamaIndex, GraphRAG, Tavily, LangGraph, MCP
3. **Ralph Loop** - Autonomous improvement iteration
4. **Claude-Flow v3** - Multi-agent orchestration

## Integration Points
- Serena: Semantic code analysis (port 24282 dashboard)
- Qdrant: Vector memory storage (port 6333)
- PostgreSQL: Structured data (port 5432)
- TouchDesigner: Creative visualization (port 9981)

## Code Conventions
- Type hints required on all functions
- Docstrings for public APIs
- Async/await for I/O operations
- Pydantic for validation
- structlog for logging
