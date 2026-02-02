# Architecture V31.1 Deep Research Synthesis

## Overview
This memory captures the comprehensive deep research findings integrated into V31.1 of the UNLEASH architecture.

## Research Completed (2026-01-23)

### 1. Claude Agent SDK (from Exa Deep Research)
- **Python SDK**: `pip install claude-agent-sdk`
- **Two Modes**: `query()` for one-off, `ClaudeSDKClient` for sessions
- **Extended Thinking**: Up to 128K tokens via `thinking` blocks
- **TypeScript V2**: `createSession()`, `resumeSession()`, `session.send()`
- **Tool Use**: JSON schema-based tool definitions with agentic loop handling

### 2. Graphiti Temporal Knowledge Graph (Local SDK + Exa)
- **Local Path**: `Z:\insider\AUTO CLAUDE\unleash\sdks\graphiti\graphiti_core\`
- **Bi-temporal Tracking**: `valid_from` and `valid_to` on edges
- **Episode Ingestion**: LLM-driven entity/relationship extraction
- **Hybrid Search**: Semantic (0.4) + BM25 (0.3) + Graph (0.3)
- **MCP Server**: FastMCP integration with semaphore control

### 3. MCP Registry & Discovery (Exa Deep Research)
- **Official Registry**: `registry.modelcontextprotocol.io`
- **FastMCP 3.0**: Components, Providers, Transforms pattern
- **OAuth 2.1**: RemoteAuthProvider with PKCE
- **A2A Protocol**: Agent-to-Agent peer communication
- **ACP Protocol**: REST-based agent endpoints
- **Kubernetes**: Helm charts and production deployment patterns

### 4. everything-claude-code (Local Analysis)
- **9 Agents**: planner, architect, tdd-guide, code-reviewer, security-reviewer, build-error-resolver, e2e-runner, refactor-cleaner, doc-updater
- **11 Skills**: continuous-learning, verification-loop, eval-harness, backend-patterns, frontend-patterns, etc.
- **14 Commands**: /plan, /tdd, /verify, /eval, /orchestrate, etc.
- **Handoff Documents**: Structured context passing between agents
- **Verification Loop**: 6-phase (build, types, lint, tests, security, diff)

## Key Integration Patterns

### Memory Backend Priority
1. claude-mem (MCP) - Recent session state (0.9 weight)
2. Serena memories - Project-specific (0.85 weight)
3. Episodic memory - Conversation history (0.8 weight)
4. Graphiti - Temporal KG (0.75 weight)
5. Letta - Long-term agent memory (0.7 weight)

### ImportanceScorer Formula
```
importance = 0.35*type + 0.30*content + 0.20*temporal + 0.15*source
```

### Thinking Token Budgets
| Complexity | Tokens | Use Case |
|------------|--------|----------|
| TRIVIAL | 0 | Status, reads |
| LOW | 4,000 | Simple edits |
| MEDIUM | 16,000 | Multi-file changes |
| HIGH | 64,000 | Architecture |
| ULTRATHINK | 128,000 | System design |

## Files Updated
- `Z:\insider\AUTO CLAUDE\unleash\ARCHITECTURE_OPTIMIZATION_SYNTHESIS.md` - V31.1 (1669 lines)

## Next Steps
1. Implement real Graphiti connection (not fallback)
2. Deploy Neo4j instance for temporal KG
3. Configure session hooks for cross-session persistence
4. Validate end-to-end integration

---
Last Updated: 2026-01-23
