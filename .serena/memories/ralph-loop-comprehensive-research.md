# Ralph Loop Comprehensive Research Findings

## Research Date: 2026-01-21
## Status: Complete Investigation

---

## 1. Core Philosophy

Ralph Loop is an **autonomous exploration methodology** that repeatedly drives AI agents through self-referential loops until defined tasks are verifiably complete.

**Key Metaphor**: Software development as a pottery wheel - repeatedly spinning a single process until each micro-task is satisfied, treating failures as data ("throwing it back on the wheel").

**Critical Innovation**: Fresh context each iteration prevents context rot, while external state (Git, files) maintains continuity.

---

## 2. The Ralph Tenets

| Tenet | Explanation |
|-------|-------------|
| **Fresh Context Is Reliability** | Each iteration clears context. Re-read specs, plan, code every cycle. |
| **Backpressure Over Prescription** | Don't prescribe how; create gates that reject bad work. |
| **The Plan Is Disposable** | Regeneration costs one planning loop. Cheap. |
| **Disk Is State, Git Is Memory** | Files are the handoff mechanism. |
| **Steer With Signals, Not Scripts** | Add signs, not scripts. |
| **Let Ralph Ralph** | Sit *on* the loop, not *in* it. |

---

## 3. Key Implementations

### 3.1 snarktank/ralph (Bash)
- Simple Bash loop orchestrator
- Supports Amp CLI and Claude Code
- Git-backed memory with PRD-driven development
- Exit condition: `<promise>COMPLETE</promise>`

### 3.2 ralph-orchestrator (Rust)
- Multi-backend support (Claude Code, Kiro, Gemini CLI, Codex, Amp, Copilot CLI, OpenCode)
- Hat-based personas for specialized roles
- 20+ presets, event-driven coordination
- Interactive TUI monitoring

### 3.3 Ralph Loop Enhanced (Local Python)
- Two-phase workflow (planning → implementation)
- Three verification gates (tests, screenshots, background agent)
- PreCompact hook for state persistence
- 12/12 features complete

---

## 4. Semantic Search MCP Options

| MCP | Features | Priority |
|-----|----------|----------|
| mcp-vector-search | ChromaDB, zero-config, native Claude integration | HIGH |
| sourcerer-mcp | Tree-sitter AST, chromem-go, token-efficient | HIGH |
| semantic-search-mcp | FastEmbed + Jina + Tree-sitter | MEDIUM |
| codebase-mcp-pgvector | PostgreSQL + pgvector | MEDIUM |

---

## 5. Memory Architecture

### Three-Tier Model
1. **Short-term**: progress.txt, prd.json, working memory
2. **Medium-term**: Git commits, checkpoints, AGENTS.md
3. **Long-term**: episodic-memory, claude-mem, Qdrant vectors, Serena memories

---

## 6. Best Practices

### Task Sizing
- Right-sized: Add a database column, UI component, server action
- Too big: "Build entire dashboard", "Add authentication" (split these)

### Verification Gates
1. Test Gate → pytest/npm test must pass
2. Build Gate → cargo build/npm build must succeed
3. Screenshot Gate → UI changes verified visually
4. Background Gate → Separate agent reviews code

### Context Management
- Use --max-iterations (prevents runaway, ~$10.42/hour)
- Enable autoHandoff at 90% context
- Semantic search instead of reading entire files
- Fresh context each iteration (no context rot)

---

## 7. Integration Recommendation

**Primary Stack**:
- Orchestration: ralph-orchestrator
- Semantic Search: mcp-vector-search
- Verification: ralph-verify.py
- Memory: episodic-memory + claude-mem

**Lightweight Stack**:
- Orchestration: snarktank-ralph
- Semantic Search: sourcerer-mcp
- Memory: progress.txt + Git

---

## 8. SDKs Cloned

| SDK | Path | Purpose |
|-----|------|---------|
| snarktank-ralph | sdks/snarktank-ralph | Original Bash orchestrator |
| ralph-orchestrator | sdks/ralph-orchestrator | Rust multi-backend |
| sourcerer-mcp | sdks/sourcerer-mcp | Token-efficient search |
| mcp-vector-search | sdks/mcp-vector-search | Zero-config search |

**Total SDKs in Collection**: 123

---

*Last Updated: 2026-01-21*
