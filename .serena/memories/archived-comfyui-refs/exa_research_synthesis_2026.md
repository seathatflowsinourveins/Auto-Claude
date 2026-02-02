# Exa Deep Research Synthesis - January 2026

**Created**: 2026-01-23
**Research Method**: Exa AI deep_researcher with exa-research-pro model

## Research 1: Claude Code CLI Patterns 2026

### Extended Thinking Evolution
- **Dynamic budgets**: 4K → 10K → 32K → 128K tokens
- **Keyword triggers**: "think", "think hard", "think harder", "ultrathink"
- **Model pairing**: Budget determines optimal model (haiku/sonnet/opus)

### MCP Connector (New in 2026)
- Remote server support via SSE transport
- Tool allowlists for security
- Capability discovery protocol

### Multi-Agent Orchestration
- Handoff documents between agents
- Workflow types: feature, bugfix, refactor, security
- Agent specialization patterns

### Hooks System
- Pre-tool, post-tool, session-start, session-stop
- Pattern extraction on stop hooks
- Continuous learning integration

## Research 2: Memory Architecture Best Practices 2026

### HiAgent Hierarchical Memory
- 4-tier architecture (working/episodic/semantic/procedural)
- Automatic promotion based on access patterns
- Capacity limits prevent context bloat

### Letta Sleep-Time Compute
- Background consolidation agents
- Runs every 5 interactions
- Semantic compression of episodic data

### Graphiti Temporal Knowledge Graphs
- Point-in-time queries
- Event timestamps vs ingestion timestamps
- Relationship decay modeling

### Context Fusion Techniques
- Two-phase retrieval (dense + salience)
- 45% payload reduction achieved
- Multi-source weighted merging

### Vector DB Patterns
- Qdrant for primary embeddings
- Hybrid search (dense + sparse)
- Namespace isolation per project

## Research 3: MCP Ecosystem 2026

### A2A Protocol v1.1
- DIDs + DIDComm v2 authentication
- Capability negotiation
- Message signing for audit

### Creative AI Servers
- comfyui-mcp: Workflow execution
- touchdesigner-creative: Real-time visuals
- blender-mcp: 3D rendering
- fal-ai: Cloud inference

### Security Enhancements
- mTLS for server auth
- OPA policy evaluation
- Tool allowlist/blocklist
- Rate limiting per client

### Performance Optimizations
- Connection pooling
- Message batching
- Streaming responses
- Cache invalidation protocols

## Key Patterns for Implementation

### 1. Thinking Budget Selection
```python
def select_budget(task: str) -> int:
    if has_ultrathink_keyword(task): return 128000
    if has_architecture_keywords(task): return 32000
    if has_debugging_keywords(task): return 10000
    if has_simple_keywords(task): return 0
    return 4000  # default low
```

### 2. Memory Promotion
```python
def should_promote(item: MemoryItem) -> str:
    if item.access_count >= 3:
        return "semantic"
    if contains_code_pattern(item.content):
        return "procedural"
    return None  # stay in current tier
```

### 3. Context Fusion
```python
def fuse_context(query: str, max_tokens: int = 8000):
    candidates = dense_retrieve(query, top_k=50)
    scored = salience_score(candidates, query)
    return merge_by_weight(scored, max_tokens)
```
