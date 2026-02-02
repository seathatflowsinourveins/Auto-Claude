# Unified Bootstrap V31 - Instant Session Access

**Created**: 2026-01-23
**Purpose**: Single entry point for all enhanced capabilities

---

## 🚀 QUICK START - Read This First

### New Session Checklist
```
1. ✅ Memory loaded automatically via hooks
2. ✅ Project detected (UNLEASH|WITNESS|TRADING)
3. ✅ SDK patterns accessible
4. ✅ Self-improvement loop ready
5. ✅ MCP servers mapped
```

---

## 📚 MEMORY MAP (Read Order)

| Priority | Memory | Purpose |
|----------|--------|---------|
| 🔴 **1** | `sdk_code_patterns_v31` | Production-ready code patterns |
| 🟠 **2** | `mcp_access_optimization_v1` | MCP server usage guide |
| 🟡 **3** | `self_improvement_loop_v1` | 6-phase learning loop |
| 🟢 **4** | `sdk_ecosystem_v30` | Quick SDK reference |
| 🔵 **5** | `memory_architecture_v11` | System architecture |

---

## 🎯 CORE CAPABILITIES (V31)

### Agent Communication (A2A)
```python
from litellm.a2a_protocol import A2AClient
client = A2AClient()
response = await client.send_task(agent_id="...", task="...")
```

### Durable Execution (Temporal)
```python
from temporalio import workflow
@workflow.defn
class MyWorkflow:
    @workflow.run
    async def run(self, data): ...
```

### Prompt Optimization (DSPy GEPA)
```python
from dspy import GEPA
gepa = GEPA(metric=metric_with_feedback, auto="light")
optimized = gepa.compile(student=program, trainset=data)
```

### Quality-Diversity (pyribs)
```python
from ribs.archives import GridArchive
from ribs.schedulers import Scheduler
scheduler = Scheduler(archive, emitters)
solutions = scheduler.ask()
scheduler.tell(objectives, measures)
```

### Safety Pipeline (NeMo + LLM-Guard)
```python
from nemoguardrails import LLMRails
from llm_guard import scan_prompt, scan_output
# 3-layer: input → guardrails → output
```

---

## 🔧 MCP QUICK ACCESS

### Memory (fastest first)
- `mcp__serena__read_memory(name)` - Code-aware
- `mcp__claude_mem__search(query)` - Observations
- `mcp__episodic_memory__search(query)` - Conversations

### Research
- `mcp__exa__web_search_exa(query)` - Web search
- `mcp__exa__get_code_context_exa(query)` - Code examples
- `mcp__context7__query_docs(libraryId, query)` - Official docs

### Creative (Witness)
- `mcp__touchdesigner_creative__create_node(type, params)`
- `mcp__touchdesigner_creative__execute_script(code)`

### Code (Serena)
- `mcp__serena__find_symbol(name_path, include_body=True)`
- `mcp__serena__search_for_pattern(pattern)`

---

## 🔄 SELF-IMPROVEMENT LOOP

```
RESEARCH → ABSORB → EXTRACT → INTEGRATE → APPLY → EVALUATE
   │                                              │
   └──────────────── FEEDBACK ────────────────────┘
```

### Triggers
- Session start: Load patterns
- Task failure: Research improvements
- Every 10 tasks: Consolidate learnings

---

## 📊 PROJECT STACKS

### UNLEASH (Meta-Development)
```
Temporal + LangGraph → Pydantic-AI + CrewAI → DSPy GEPA + pyribs
```

### WITNESS (Creative)
```
pyribs MAP-Elites → TouchDesigner MCP → LiveKit + Pipecat
```

### TRADING (AlphaForge)
```
Temporal (crash-proof) → NeMo Guardrails → Guardrails AI
```

---

## 📁 FILE LOCATIONS

### SDK Documentation
- `Z:\insider\AUTO CLAUDE\unleash\sdks\SDK_INTEGRATION_PATTERNS_V30.md`
- `Z:\insider\AUTO CLAUDE\unleash\sdks\SDK_QUICK_REFERENCE.md`
- `Z:\insider\AUTO CLAUDE\unleash\DEEP_DIVE_SDK_REFERENCE.md`

### Memory Files
- Serena: `.serena/memories/` (19+ files)
- Episodic: `~/.claude/conversations/`
- Claude-mem: MCP-managed

---

## ⚡ INSTANT ACTIONS

### Research Something New
```
1. mcp__exa__web_search_exa("topic 2026")
2. mcp__exa__get_code_context_exa("topic implementation")
3. Store: mcp__serena__write_memory(name, content)
```

### Find Past Solution
```
1. mcp__serena__list_memories() → find relevant
2. mcp__serena__read_memory(name)
3. Or: mcp__claude_mem__search(query)
```

### Apply Pattern
```
1. Read: mcp__serena__read_memory("sdk_code_patterns_v31")
2. Find: Search for relevant section
3. Apply: Use code pattern as template
```

---

## 🎓 KEY INSIGHTS (V31)

1. **A2A Protocol**: Google-backed, 50+ companies, JSON-RPC 2.0
2. **FastMCP v2**: Server composition via mount(), enterprise OAuth
3. **DSPy GEPA**: 10-14% better than MIPROv2, feedback-driven
4. **Temporal**: Crash-proof workflows, Pydantic-AI integration
5. **pyribs**: GridArchive + CMA-ME, ask-tell interface
6. **LLM-Guard**: 15+ input scanners, 20+ output scanners

---

## 🔐 SAFETY DEFAULTS

```python
# Always use for user-facing
input_scanners = [PromptInjection(), Toxicity(), Secrets()]
output_scanners = [FactualConsistency(), Bias(), MaliciousURLs()]
```

---

*Bootstrap Version: 31.0 | Session-Ready | All Systems Go*
