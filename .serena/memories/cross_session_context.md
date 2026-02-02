# Cross-Session Context for Unleash Platform

## Session Continuity Protocol

### On Session Start
1. Activate Serena: `mcp__plugin_serena_serena__activate_project("unleash")`
2. If timeout: Kill and restart Serena processes
3. Load memories from `.serena/memories/`
4. Check SDK status: 7 SDKs should be operational

### Known Issues & Solutions

#### Serena Timeout Fix
```powershell
# Kill stuck processes
Stop-Process -Name "serena" -Force -ErrorAction SilentlyContinue
# Wait for socket cleanup
Start-Sleep -Seconds 5
# Retry activation
```

#### SDK Verification
- Crawl4AI: LLM-optimized web crawling
- LightRAG: Lightweight GraphRAG alternative
- LlamaIndex: RAG indexing (PyPI version)
- GraphRAG: Partial support (indexer adapters)
- Tavily: AI-powered web search
- LangGraph: Stateful agent workflows
- MCP: Model Context Protocol SDK

### Critical Paths
- Research Engine: `platform/core/research_engine.py`
- SDK Integrations: `platform/core/sdk_integrations.py`
- Ralph Loop: `v10_optimized/scripts/ralph_loop.py`
- Orchestrator: `v10_optimized/scripts/ecosystem_orchestrator.py`

### Memory Persistence Layers
1. **Serena Memories**: `.serena/memories/*.md` (this file)
2. **Claude-Mem**: MCP-based observation storage
3. **Episodic Memory**: Conversation archive search
4. **Qdrant Vectors**: Semantic embeddings (port 6333)

### Session Handoff Checklist
- [ ] Save current task state to claude-mem
- [ ] Update memories if new insights discovered
- [ ] Document any new patterns or solutions
- [ ] Note pending tasks for next session

## Quick Reference

### Activate Unleash in New Session
```
1. Open Claude Code
2. Wait for MCP servers to start (~30 seconds)
3. Say: "Activate unleash project in Serena"
4. If timeout: Follow Serena Timeout Fix above
```

### Resume Previous Work
```
1. Search episodic memory for last session
2. Load relevant claude-mem observations
3. Check todo list state
4. Continue from last checkpoint
```
