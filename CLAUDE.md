# Claude Code Configuration - UNLEASH V6

## Behavioral Rules (CRITICAL)

- Do what has been asked; nothing more, nothing less
- ALWAYS read a file before editing it
- NEVER create files unless absolutely necessary - prefer editing existing
- NEVER save to root folder - use `/platform`, `/docs`, `/scripts`, `/tests`
- NEVER commit secrets, .env files, or credentials
- NEVER continuously poll status after spawning swarms - wait for results
- After 2 failed corrections, `/clear` and write a better prompt

## File Organization

| Directory | Purpose |
|-----------|---------|
| `/platform/core` | Core modules (orchestration, RAG, memory) |
| `/platform/adapters` | SDK adapters (40+ adapters) |
| `/platform/tests` | Test files |
| `/docs` | Documentation and markdown |
| `/scripts` | Utility scripts |
| `/research` | Research iteration configs |

## Build & Test

```bash
npm run build    # Build project
npm test         # Run tests
npm run lint     # Lint code
```

ALWAYS run tests after code changes. ALWAYS verify build before committing.

## Concurrency: 1 MESSAGE = ALL OPERATIONS

- ALWAYS batch ALL file reads/writes/edits in ONE message
- ALWAYS spawn ALL agents in ONE message with full instructions
- ALWAYS batch ALL Bash commands in ONE message
- Use `run_in_background: true` for all agent Task calls
- After spawning agents, STOP - do NOT check status

## 3-Tier Model Routing (ADR-026)

| Tier | Handler | Latency | Use Case |
|------|---------|---------|----------|
| **1** | Agent Booster (WASM) | <1ms | Simple transforms (var to const, add types) |
| **2** | Haiku | ~500ms | Simple tasks, complexity <30% |
| **3** | Sonnet/Opus | 2-5s | Complex reasoning, architecture, security |

**IMPORTANT**: Check for `[AGENT_BOOSTER_AVAILABLE]` or `[TASK_MODEL_RECOMMENDATION]` before spawning agents. Use Edit tool directly when Agent Booster is available.

## Swarm Configuration (Anti-Drift)

```bash
npx @claude-flow/cli@latest swarm init --topology hierarchical --max-agents 8 --strategy specialized
```

**Mandatory settings:**
- Topology: `hierarchical` (prevents drift via single coordinator)
- Max agents: 6-8 (tight coordination)
- Strategy: `specialized` (clear role boundaries)
- Consensus: `raft` (leader maintains authoritative state)
- Memory namespace: shared across all agents

**Checkpoint rules:**
- Run frequent checkpoints via `post-task` hooks
- Store state in memory before complex operations
- Use `/rewind` to restore on failure

## Memory Persistence

**Session Start:**
```bash
npx @claude-flow/cli@latest memory search --query "project:$(basename $PWD) architecture" --limit 10
```

**Session End:**
```bash
npx @claude-flow/cli@latest memory store --key "session-$(date +%Y%m%d)" --value "SUMMARY" --namespace decisions --ttl 30d
```

**Cross-session patterns:**
- Store architectural decisions in `decisions` namespace
- Store code patterns in `patterns` namespace
- Use `--tags` for filtering: `--tags "api,auth,critical"`
- Confidence decay: decisions/patterns never decay, progress decays in 7 days

## Research Tools (Priority Order)

| Tool | Use Case | Speed |
|------|----------|-------|
| **Exa** | Neural semantic search, code context | Fast: <350ms |
| **Tavily** | Factual research with citations | Sub-second |
| **Context7** | Library documentation lookup | Fast |
| **Jina** | Embeddings, reranking, HTML to MD | ~500ms |
| **Firecrawl** | Deep web scraping, JS-heavy sites | Variable |
| **Perplexity** | AI synthesis when others fail | Backup |

**Query pattern:**
1. Start with Exa for semantic search
2. Add Tavily for citations
3. Use Context7 for library-specific docs
4. Fall back to Firecrawl for deep extraction

## Error Recovery

**API Failures:**
1. Retry with exponential backoff (1s, 2s, 4s)
2. Fall back to alternative tool (Exa -> Tavily -> Perplexity)
3. Log failure to memory for pattern detection
4. If 3+ failures, alert and switch to manual mode

**Swarm Failures:**
1. Check agent logs: `npx @claude-flow/cli@latest agent logs <id>`
2. Restore from checkpoint: `/rewind`
3. Reduce agent count and retry
4. Store failure pattern in memory for future avoidance

**Context Overflow:**
1. Run `/compact` with focus instructions
2. Clear unrelated context with `/clear`
3. Use subagents for investigation (separate context)
4. Store important state to memory before clearing

## Token Efficiency

- Use subagents for file exploration (keeps main context clean)
- Scope investigations narrowly - avoid "explore everything"
- Prefer `--limit` flags on searches
- Use streaming for long operations
- Compact frequently during long sessions

## Security Rules

- NEVER hardcode API keys or secrets
- Always validate user input at system boundaries
- Sanitize file paths (prevent directory traversal)
- Run security scan after security-related changes:
  ```bash
  npx @claude-flow/cli@latest security scan
  ```

## Agent Types (Quick Reference)

**Core:** `coder`, `reviewer`, `tester`, `planner`, `researcher`
**Specialized:** `security-architect`, `memory-specialist`, `performance-engineer`
**Swarm:** `hierarchical-coordinator`, `mesh-coordinator`
**GitHub:** `pr-manager`, `code-review-swarm`, `issue-tracker`

## CLI Quick Reference

```bash
# Initialization
npx @claude-flow/cli@latest init --wizard
npx @claude-flow/cli@latest doctor --fix

# Agent management
npx @claude-flow/cli@latest agent spawn -t coder --name my-coder
npx @claude-flow/cli@latest swarm init --v3-mode

# Memory operations
npx @claude-flow/cli@latest memory store --key "KEY" --value "VALUE" --namespace NS
npx @claude-flow/cli@latest memory search --query "QUERY" --limit 10
npx @claude-flow/cli@latest memory retrieve --key "KEY" --namespace NS
```

## Architecture Principles

- Domain-Driven Design with bounded contexts
- Files under 500 lines
- Typed interfaces for all public APIs
- TDD London School (mock-first) for new code
- Event sourcing for state changes
- Input validation at system boundaries

## Project Config

- **Topology**: hierarchical-mesh
- **Max Agents**: 15
- **Memory**: hybrid (SQLite + HNSW vectors)
- **HNSW**: M=16, efConstruction=200

## Hooks Configuration

```json
{
  "PostToolUse": [
    {"matcher": "Edit(**/*.py)", "hooks": [{"type": "command", "command": "python -m py_compile %CLAUDE_FILE_PATHS%"}]},
    {"matcher": "Edit(**/*.ts)", "hooks": [{"type": "command", "command": "npx tsc --noEmit"}]}
  ],
  "Stop": [
    {"hooks": [{"type": "prompt", "prompt": "Verify tests pass before completing"}]}
  ]
}
```

## External Documentation

For detailed patterns, see:
- `docs/essential/UNLEASHED_PATTERNS.md` - Production patterns
- `docs/RESEARCH_STACK_2026.md` - Research tool details
- `docs/ADR-027-MCP-CONFIGURATION-OPTIMIZATION.md` - MCP optimization
- `docs/gap-resolution/` - Gap resolution guides

## Support

- Documentation: https://github.com/ruvnet/claude-flow
- Issues: https://github.com/ruvnet/claude-flow/issues
