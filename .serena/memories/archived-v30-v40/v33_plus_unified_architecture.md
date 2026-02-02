# V33+ Unified Architecture - Complete Integration

**Created**: 2026-01-23 (Ralph Loop Iteration 1)
**Document**: `Z:\insider\AUTO CLAUDE\unleash\UNIFIED_ARCHITECTURE_V33_PLUS.md`

## Key V33+ Enhancements Over V33

### 1. In-Process MCP Servers (NEW)
- **10-100x faster** than stdio transport
- Tool calls: <0.1ms vs 2-5ms
- Using `createSdkMcpServer()` from @anthropic-ai/claude-code/sdk

### 2. PydanticAI 5-Level Multi-Agent
- Level 1: Single agent workflows
- Level 2: Agent delegation (tool-based)
- Level 3: Programmatic hand-off (human-in-loop)
- Level 4: Graph-based control (LangGraph)
- Level 5: Deep Agents (autonomous with full capabilities)

### 3. Everything-Claude-Code Integration
- 9 agents: planner, architect, tdd-guide, code-reviewer, security-reviewer, build-error-resolver, e2e-runner, refactor-cleaner, doc-updater
- 14 commands: /plan, /tdd, /verify, /eval, /orchestrate, /code-review, etc.
- Verification loop with 6 phases

### 4. Enhanced Hook System (9 Events)
```
SessionStart → PreToolUse → PostToolUse → PreCompact → SessionEnd
              ↓
         UserPromptSubmit, Notification, Stop, SubagentStop
```

### 5. A2A Protocol v1.1
- DID + DIDComm v2 authentication
- Capability negotiation before hand-off
- Message signing for audit trail

## Architecture Stack (24 Layers)

| Range | Purpose | Key SDKs |
|-------|---------|----------|
| 0 | Infrastructure | K8s, Redis, QuestDB |
| 1-5 | Core Runtime | LiteLLM, Pydantic-AI, Temporal, Letta |
| 6-10 | Intelligence | DSPy, LlamaIndex, pyribs |
| 11-15 | Safety | NeMo, LLM-Guard, DeepEval |
| 16-20 | Integration | Claude SDK, MCP, A2A |
| 21-24 | Project-Specific | Unleash/Witness/Trading |

## Memory Architecture (HiAgent V33+)

```
Working (≤100) → Episodic (≤1000) → Semantic → Procedural
   Session          30 days        Permanent   Permanent
```

Promotion triggers:
- Working→Episodic: On compaction
- Episodic→Semantic: Access count ≥3
- Any→Procedural: Contains code pattern

## Thinking Budgets

| Level | Tokens | Model | Trigger |
|-------|--------|-------|---------|
| trivial | 0 | haiku | "simple", "quick" |
| low | 4K | sonnet | "think", "explain" |
| medium | 10K | sonnet | "think hard", "debug" |
| high | 32K | opus | "think harder", "architecture" |
| ultrathink | 128K | opus | "ultrathink", "megathink" |

## Cross-Session Bootstrap

1. letta-session-start.py → Memory blocks
2. project-bootstrap.py → Project detection
3. memory-gateway-hook.py → Unified search

ImportanceScorer: 0.35*type + 0.30*content + 0.20*temporal + 0.15*source

## Quick Reference

| Task | SDK |
|------|-----|
| Prompt optimization | DSPy GEPA |
| Agent orchestration | Pydantic-AI |
| Durable workflows | Temporal |
| Safety rails | NeMo + LLM-Guard |
| Evaluation | Opik + DeepEval |
| Quality-diversity | pyribs |

## Research Sources

1. Exa Deep Research 2026 (Claude Code + MCP ecosystem)
2. unleash/ULTIMATE_UNLEASH_ARCHITECTURE_V30.md
3. everything-claude-code-full/ (9 agents, 14 commands)
4. Claude Code SDK deep analysis (undocumented features)
5. PydanticAI multi-agent documentation
