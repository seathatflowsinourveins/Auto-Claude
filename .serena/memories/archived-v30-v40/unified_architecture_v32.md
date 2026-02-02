# V32 Unified Architecture Summary

## Quick Reference

V32 synthesizes V30 (21-Layer) + V31 (Integration) + V11 (Memory) + V13 (Ralph Loop) + 2026 deep research.

## Key Additions in V32

1. **Protocol Interoperability (Layer 22)**
   - MCP (tools) + A2A (agent-to-agent) + ACP (cross-framework)
   - AgentCard discovery, Task routing, Protocol bridging

2. **FastMCP 3.0 Patterns**
   - Components: tools, resources, prompts
   - Providers: decorators, files, OpenAPI, remote
   - Transforms: namespacing, filtering, authorization
   - `provider.with_namespace("api")` → "tool" → "api_tool"
   - `server.disable(tags={"internal"})` for visibility control

3. **A2A Protocol (Google, 50+ companies)**
   - gRPC/HTTP2 transport
   - TaskState: SUBMITTED → WORKING → COMPLETED/FAILED
   - AgentCard for capability discovery

4. **Enhanced Memory Architecture**
   - Working → Episodic → Semantic → Procedural (HiAgent)
   - Sleep-time compute: 5x token reduction
   - Hybrid retrieval: BM25 → Dense → Reranker

5. **Hook Pipeline (everything-claude-code)**
   - SessionStart: letta-session-start, project-bootstrap, memory-gateway
   - PreToolUse: dev-server-guard, git-push-pause, strategic-compact
   - PostToolUse: prettier, tsc, console-log-warning
   - Stop: memory-consolidate
   - SessionEnd: continuous-learning pattern extraction

## Full Document

`Z:\insider\AUTO CLAUDE\unleash\UNIFIED_ARCHITECTURE_V32.md` (1,600+ lines)

## Project Stacks

| Project | Primary Stack |
|---------|---------------|
| UNLEASH | Temporal + LangGraph + DSPy GEPA + pyribs |
| WITNESS | pyribs MAP-Elites + TouchDesigner MCP + GLSL |
| TRADING | Temporal + NeMo + Graphiti bi-temporal |

## Memory Search Order

1. `mcp__serena__read_memory(name)` - If you know exact name
2. `mcp__serena__list_memories()` - Browse available
3. `mcp__claude_mem__search(query)` - Semantic search
4. `mcp__episodic_memory__search(query)` - Conversations

## Task Complexity → Thinking Tokens

| Level | Tokens | Use Case |
|-------|--------|----------|
| TRIVIAL | 0 | Status, simple reads |
| LOW | 4,000 | Simple edits |
| MEDIUM | 16,000 | Code changes |
| HIGH | 64,000 | Architecture |
| ULTRATHINK | 128,000 | System design |

---
Created: 2026-01-23
