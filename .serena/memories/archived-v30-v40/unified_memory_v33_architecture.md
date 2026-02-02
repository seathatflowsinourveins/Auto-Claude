# V33 Unified Architecture - Cross-Session Memory

**Created**: 2026-01-23
**Source**: Deep research synthesis from Exa + local repo analysis
**Document**: `Z:\insider\AUTO CLAUDE\unleash\UNIFIED_ARCHITECTURE_V33.md`

## Core V33 Innovations

### 1. Dynamic Extended Thinking Budgets
```python
THINKING_BUDGETS = {
    "trivial": 0,       # Direct response, no thinking
    "low": 4000,        # Brief outline (2-3 sentences)  
    "medium": 10000,    # Explain approach + key decisions
    "high": 32000,      # Step-by-step with alternatives
    "ultrathink": 128000 # Full multi-angle analysis
}
```

**Keyword Triggers**:
- "think" → low (4K)
- "think hard" → medium (10K)  
- "think harder" → high (32K)
- "ultrathink" / "megathink" → ultrathink (128K)

**Semantic Complexity Indicators** (auto-detect):
- architecture, design, trade-off → HIGH
- debug, fix, error → MEDIUM
- explain, describe → LOW
- simple, quick, trivial → TRIVIAL

### 2. HiAgent Hierarchical Memory (4 Tiers)

| Tier | Capacity | Retention | Use Case |
|------|----------|-----------|----------|
| Working | ≤100 items | Session | Immediate context |
| Episodic | ≤1000 items | 30 days | Conversation history |
| Semantic | Unlimited | Permanent | Distilled knowledge |
| Procedural | Unlimited | Permanent | Learned patterns |

**Promotion Rules**:
- Working→Episodic: On compaction
- Episodic→Semantic: Access count ≥3
- Any→Procedural: Contains code pattern

### 3. Letta Sleep-Time Consolidation
```python
# Triggers background agent every 5 interactions
letta.trigger_consolidation(
    context_window=last_5_turns,
    memory_types=["episodic", "semantic"],
    priority="batch"
)
```

### 4. Context Fusion (45% Payload Reduction)
Two-phase retrieval:
1. **Phase 1**: Dense retrieval (top-50 candidates)
2. **Phase 2**: Salience-weighted merging

```python
merged = fusion_merge(
    sources=[episodic, semantic, procedural],
    weights=[0.3, 0.4, 0.3],
    max_tokens=8000
)
```

### 5. A2A Protocol v1.1 (Agent-to-Agent)
- DIDs + DIDComm v2 for authentication
- Capability negotiation before handoff
- Message signing for audit trail

### 6. MCP Connector (Remote Servers)
```json
{
  "mcpServers": {
    "remote-creative": {
      "transport": "sse",
      "url": "https://mcp.example.com/creative",
      "allowedTools": ["generate_image", "style_transfer"]
    }
  }
}
```

## Integration Points

### Cross-Session Hook: `cross_session_v33.py`
**Location**: `C:\Users\42\.claude\hooks\cross_session_v33.py`

Key functions:
- `detect_task_complexity()` → Returns ThinkingConfig with budget + model + prompt
- `HiAgentMemoryV33.add()` → Adds to working memory with auto-promotion
- `CrossSessionV33.pre_task_setup()` → Complexity detection + memory retrieval
- `CrossSessionV33.post_task_capture()` → Pattern extraction + learning
- `CrossSessionV33.pre_compact()` → Memory promotion before compaction

### Files Created
1. `UNIFIED_ARCHITECTURE_V33.md` - Full architecture (~900 lines)
2. `cross_session_v33.py` - Integration hooks (~700 lines)

## Research Sources Synthesized
1. **Exa Deep Research**: Claude Code 2026, MCP ecosystem, memory architectures
2. **Local Repos**: everything-claude-code-full, github-advanced-unleash
3. **Patterns Extracted**: continuous-learning, verification-loop, strategic-compact
4. **SDKs Analyzed**: EvoAgentX AFlow, Opik observability, Agent SDK
