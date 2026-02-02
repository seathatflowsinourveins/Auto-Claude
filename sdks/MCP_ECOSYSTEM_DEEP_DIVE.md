# MCP Ecosystem Deep Dive Reference

> **Created**: 2026-01-27 | **Repositories**: 11 | **Purpose**: Comprehensive reference for MCP ecosystem SDKs
> **Location**: `Z:\insider\AUTO CLAUDE\unleash\sdks\mcp-ecosystem\`

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Repository Overview](#repository-overview)
3. [Tier 1: Core AI Infrastructure](#tier-1-core-ai-infrastructure)
   - [Claude-Flow V3](#claude-flow-v3)
   - [Letta (MemGPT)](#letta-memgpt)
4. [Tier 2: Search & Research](#tier-2-search--research)
   - [Exa MCP](#exa-mcp)
   - [Tavily Suite](#tavily-suite)
   - [Jina MCP](#jina-mcp)
   - [Context7](#context7)
5. [Tier 3: Development Tools](#tier-3-development-tools)
   - [Everything Claude Code](#everything-claude-code)
   - [Opik](#opik)
6. [Tier 4: Research Resources](#tier-4-research-resources)
   - [System Prompts Collection](#system-prompts-collection)
7. [Integration Patterns](#integration-patterns)
8. [Usage Matrix](#usage-matrix)

---

## Executive Summary

This document provides a comprehensive analysis of 11 MCP ecosystem repositories cloned for the UNLEASH SDK platform. These tools provide:

| Category | Repositories | Primary Use |
|----------|--------------|-------------|
| **AI Orchestration** | claude-flow | Multi-agent swarms, self-learning |
| **Stateful Agents** | letta | Persistent memory, self-improvement |
| **Web Search** | exa-mcp-server, tavily-*, jina-mcp | LLM-optimized search |
| **Documentation** | context7 | Library documentation retrieval |
| **Observability** | opik | AI system monitoring, evaluation |
| **Development** | everything-claude-code | Claude Code plugins, skills, agents |
| **Research** | system-prompts-and-models | AI system prompt analysis |

---

## Repository Overview

| Repository | Stars | Type | Status |
|------------|-------|------|--------|
| `claude-flow` | 12.4K | npm | ✅ V3 Alpha |
| `letta` | 14K+ | Python/API | ✅ Active |
| `exa-mcp-server` | 500+ | npm | ✅ Active |
| `tavily-python` | 1K+ | Python | ✅ Active |
| `tavily-mcp` | 500+ | npm | ✅ Active |
| `tavily-js` | 200+ | npm | ✅ Active |
| `jina-mcp` | 300+ | npm | ✅ Active |
| `context7` | 8K+ | MCP | ✅ Active |
| `everything-claude-code` | 2K+ | Plugin | ✅ Active |
| `opik` | 5K+ | Python | ✅ Active |
| `system-prompts-and-models` | 10K+ | Research | ✅ Active |

---

## Tier 1: Core AI Infrastructure

### Claude-Flow V3

**Location**: `mcp-ecosystem/claude-flow/`
**Install**: `npm install -g claude-flow@v3alpha`
**Type**: Enterprise AI orchestration platform

#### Architecture

```
User → CLI/MCP → Intelligent Router → Swarm Coordination → 60+ Agents → Memory → LLM Providers
                      ↓                      ↓
                 3-Tier Routing         Self-Learning
                 (WASM/Haiku/Opus)      (SONA/EWC++)
```

#### Key Capabilities

| Feature | Description | Performance |
|---------|-------------|-------------|
| **60+ Agents** | Specialized roles across 8 categories | Dynamic spawning |
| **SONA** | Self-Optimizing Neural Architecture | <0.05ms adaptation |
| **EWC++** | Prevents catastrophic forgetting | 95%+ knowledge preservation |
| **MoE** | 8 expert networks with dynamic gating | Task-based routing |
| **HNSW** | Vector search acceleration | 150x-12,500x faster |
| **Flash Attention** | Accelerated attention compute | 2.49x-7.47x speedup |
| **Agent Booster** | WASM code transforms without LLM | 352x faster, $0 cost |
| **RuVector** | PostgreSQL vector extension | 77+ SQL functions, 61µs search |

#### Agent Categories (64 Total)

1. **Core Development** (5): coordinator, developer, designer, tester, reviewer
2. **Swarm Coordination** (3): swarm-coordinator, queen, worker
3. **Hive-Mind Consensus** (3): hive-mind, consensus-builder, vote-aggregator
4. **Distributed Consensus** (7): raft-leader, raft-follower, pbft-primary, pbft-replica, paxos-*
5. **Performance** (5): performance-monitor, load-balancer, cache-manager, rate-limiter, circuit-breaker
6. **GitHub Integration** (12): github-issue-creator, github-pr-creator, github-reviewer, github-merger, etc.
7. **SPARC Methodology** (4): sparc-specification, sparc-pseudocode, sparc-architecture, sparc-refinement
8. **Specialized Engineers** (8): ml-engineer, devops-engineer, security-analyst, data-engineer, etc.

#### 3-Tier Model Routing

| Tier | Handler | Latency | Cost | Use Cases |
|------|---------|---------|------|-----------|
| **1** | Agent Booster (WASM) | <1ms | $0 | var→const, add-types, remove-console |
| **2** | Haiku/Sonnet | 500ms-2s | $0.0002-$0.003 | Bug fixes, refactoring |
| **3** | Opus | 2-5s | $0.015 | Architecture, security design |

#### Swarm Topologies

```
Hierarchical (Default)    Mesh              Ring              Star
      Queen               Agent ↔ Agent     A1 → A2           Hub
     /  |  \                ↕       ↕         ↓                 ↓
    W1  W2  W3           Agent ↔ Agent     A3 ← A4         A1 A2 A3
```

#### MCP Tools (87+)

- **Swarm Management** (16): swarm_init, swarm_status, agent_spawn, task_orchestrate
- **Neural & AI** (15): neural_train, neural_patterns, cognitive_analyze
- **Memory** (10): memory_store, memory_query, memory_backup
- **Performance** (10): metrics_collect, bottleneck_identify, cost_assess
- **GitHub** (6): repo_analyze, pr_enhance, issue_track
- **Workflow** (8): workflow_create, pipeline_setup, event_trigger

#### Quick Start

```bash
# Install
npm install -g claude-flow@v3alpha

# Add to Claude Code
claude mcp add claude-flow -- npx claude-flow@v3alpha mcp start

# Start swarm
./claude-flow swarm "implement feature X" --strategy development --review
```

---

### Letta (MemGPT)

**Location**: `mcp-ecosystem/letta/`
**Install**: `pip install letta-client` (Python) | `npm install @letta-ai/letta-client` (TS)
**Type**: Stateful AI agents platform with persistent memory

#### Core Concept

Letta creates AI agents that **remember and improve over time**:

```python
from letta_client import Letta

client = Letta(api_key=os.getenv("LETTA_API_KEY"))

# Create agent with memory
agent = client.agents.create(
    model="openai/gpt-5.2",
    memory_blocks=[
        {"label": "human", "value": "Name: User. Context: ..."},
        {"label": "persona", "value": "I am a helpful assistant."}
    ],
    tools=["web_search", "fetch_webpage"]
)

# Agent remembers across conversations
response = client.agents.messages.create(agent.id, input="Remember this...")
```

#### Architecture

```
┌─────────────────────────────────────────────────────┐
│                    LettaAgent                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │   Memory    │  │   Tools     │  │  Summarizer │  │
│  │   Blocks    │  │   Executor  │  │   Agent     │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  │
│          │               │               │          │
│          └───────────────┼───────────────┘          │
│                          ↓                          │
│  ┌─────────────────────────────────────────────────┐│
│  │              BaseAgent (Abstract)               ││
│  │  - step(): Execute agent loop                   ││
│  │  - step_stream(): Streaming execution           ││
│  │  - _rebuild_memory_async(): Memory management   ││
│  └─────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
```

#### Key Classes (from source analysis)

| Class | File | Purpose |
|-------|------|---------|
| `BaseAgent` | `agents/base_agent.py` | Abstract base with step/stream methods |
| `LettaAgent` | `agents/letta_agent.py` | Main agent implementation |
| `EphemeralSummaryAgent` | `agents/ephemeral_summary_agent.py` | Memory summarization |
| `MessageManager` | `services/message_manager.py` | Message persistence |
| `PassageManager` | `services/passage_manager.py` | Long-term memory passages |
| `AgentManager` | `services/agent_manager.py` | Agent lifecycle |
| `ToolExecutionManager` | `services/tool_executor/` | Tool execution |

#### Memory System

```
Memory Blocks
├── Human Block: User context, preferences
├── Persona Block: Agent identity, personality
└── Custom Blocks: Domain-specific knowledge

Long-term Storage
├── Passages: Semantic chunks with embeddings
├── Archives: Full conversation history
└── Summaries: Compressed context
```

#### Multi-Agent Support

```python
# From letta/groups/
from letta.groups import DynamicMultiAgent, SupervisorMultiAgent

# Dynamic: Agents coordinate peer-to-peer
dynamic = DynamicMultiAgent(agents=[agent1, agent2, agent3])

# Supervisor: One agent coordinates others
supervisor = SupervisorMultiAgent(
    supervisor=coordinator,
    workers=[agent1, agent2]
)
```

#### LLM Provider Support

From `letta/llm_api/`:
- Anthropic (Claude)
- OpenAI (GPT)
- Google AI
- DeepSeek
- Azure OpenAI
- Cohere
- Together AI
- OpenRouter

---

## Tier 2: Search & Research

### Exa MCP

**Location**: `mcp-ecosystem/exa-mcp-server/`
**Type**: Neural search API for LLMs

#### Features

| Endpoint | Purpose | Best For |
|----------|---------|----------|
| `/search` | Semantic web search | Research queries |
| `/contents` | Content extraction | Full page text |
| `/find_similar` | Similar page discovery | Related content |
| `/deep_research` | Multi-step research | Comprehensive reports |

#### Usage

```typescript
// MCP tool call
mcp__exa__web_search_exa({ query: "AI orchestration patterns 2026" })
mcp__exa__deep_research_exa({
  query: "comprehensive analysis of X",
  depth: "thorough"
})
```

---

### Tavily Suite

**Location**: `mcp-ecosystem/tavily-python/`, `tavily-mcp/`, `tavily-js/`
**Type**: LLM-optimized search, extract, crawl, research APIs

#### API Endpoints

| API | Description | Use Case |
|-----|-------------|----------|
| **Search** | Web search optimized for LLMs | Fact retrieval |
| **Extract** | Content extraction from URLs | Page parsing |
| **Crawl** | Website traversal with instructions | Site mapping |
| **Map** | URL discovery and structure | Site structure |
| **Research** | Comprehensive research reports | Deep analysis |

#### Python Usage

```python
from tavily import TavilyClient

client = TavilyClient(api_key="tvly-...")

# Search
results = client.search("latest AI developments", max_results=10)

# RAG context generation
context = client.get_search_context("What happened at event X?")

# Quick Q&A
answer = client.qna_search("Who is the CEO of X?")

# Extract content
content = client.extract(urls=["https://example.com"])

# Crawl with instructions
crawled = client.crawl(
    url="https://docs.example.com",
    max_depth=3,
    instructions="Find all API documentation"
)

# Deep research
research = client.research(
    input="Research topic X",
    model="pro",
    citation_format="apa"
)
```

#### Search Depth Options

| Depth | Latency | Relevance | Content Type |
|-------|---------|-----------|--------------|
| `ultra-fast` | Lowest | Lower | NLP summary |
| `fast` | Low | Good | Chunks |
| `basic` | Medium | High | NLP summary |
| `advanced` | Higher | Highest | Chunks |

---

### Jina MCP

**Location**: `mcp-ecosystem/jina-mcp/`
**Type**: Neural search and embedding APIs

#### Features

- Reader API: Clean content extraction
- Search API: Semantic search
- Embeddings: Vector generation
- Reranker: Result reordering

---

### Context7

**Location**: `mcp-ecosystem/context7/`
**Type**: Library documentation retrieval

#### MCP Tools

```typescript
// Resolve library to Context7 ID
mcp__context7__resolve-library-id({
  libraryName: "react",
  query: "hooks usage"
})

// Query documentation
mcp__context7__query-docs({
  libraryId: "/facebook/react",
  query: "useEffect cleanup function examples"
})
```

#### Best Practices

- Call `resolve-library-id` first to get valid library ID
- Keep queries specific (not "auth" but "JWT authentication setup")
- Max 3 calls per question
- Prioritize libraries with higher Code Snippet counts

---

## Tier 3: Development Tools

### Everything Claude Code

**Location**: `mcp-ecosystem/everything-claude-code/`
**Type**: Complete Claude Code plugin ecosystem

#### Structure

```
everything-claude-code/
├── agents/           # 9 specialized agents
│   ├── planner.md
│   ├── architect.md
│   ├── code-reviewer.md
│   ├── security-reviewer.md
│   ├── tdd-guide.md
│   ├── build-error-resolver.md
│   ├── e2e-runner.md
│   ├── refactor-cleaner.md
│   └── doc-updater.md
├── skills/           # 11+ skills
│   ├── continuous-learning/SKILL.md
│   ├── verification-loop/SKILL.md
│   ├── eval-harness/SKILL.md
│   ├── tdd-workflow/SKILL.md
│   ├── security-review/SKILL.md
│   ├── backend-patterns/SKILL.md
│   ├── frontend-patterns/SKILL.md
│   └── ...
├── commands/         # 14 slash commands
├── rules/            # 6 coding rules
└── hooks/            # Lifecycle hooks
```

#### Key Skills

##### Continuous Learning

Auto-extracts reusable patterns from sessions:

```json
{
  "min_session_length": 10,
  "patterns_to_detect": [
    "error_resolution",
    "user_corrections",
    "workarounds",
    "debugging_techniques",
    "project_specific"
  ]
}
```

Hook integration:
```json
{
  "hooks": {
    "Stop": [{
      "matcher": "*",
      "hooks": [{
        "type": "command",
        "command": "~/.claude/skills/continuous-learning/evaluate-session.sh"
      }]
    }]
  }
}
```

##### Verification Loop

6-phase verification after code changes:

1. **BUILD**: `npm run build` - must pass
2. **TYPES**: `npx tsc --noEmit` - report errors
3. **LINT**: `npm run lint` - fix issues
4. **TESTS**: `npm test --coverage` - 80%+ coverage
5. **SECURITY**: Check for secrets, console.log
6. **DIFF**: Review changes

Output format:
```
VERIFICATION REPORT
==================
Build:     [PASS/FAIL]
Types:     [PASS/FAIL] (X errors)
Lint:      [PASS/FAIL] (X warnings)
Tests:     [PASS/FAIL] (X/Y passed, Z% coverage)
Security:  [PASS/FAIL] (X issues)
Diff:      [X files changed]

Overall:   [READY/NOT READY] for PR
```

##### Eval Harness (EDD)

Eval-Driven Development framework:

```markdown
## EVAL DEFINITION: feature-xyz

### Capability Evals
1. Can create new user account
2. Can validate email format

### Regression Evals
1. Existing login still works
2. Session management unchanged

### Success Metrics
- pass@3 > 90% for capability evals
- pass^3 = 100% for regression evals
```

Metrics:
- **pass@k**: "At least one success in k attempts"
- **pass^k**: "All k trials succeed"

#### Agents

| Agent | Purpose | Model |
|-------|---------|-------|
| `planner` | Implementation planning | sonnet |
| `architect` | System design | opus |
| `code-reviewer` | Quality review | sonnet |
| `security-reviewer` | Security analysis | opus |
| `tdd-guide` | Test-driven development | opus |
| `build-error-resolver` | Fix build failures | sonnet |
| `e2e-runner` | Playwright E2E testing | sonnet |
| `refactor-cleaner` | Dead code cleanup | sonnet |
| `doc-updater` | Documentation updates | haiku |

---

### Opik

**Location**: `mcp-ecosystem/opik/`
**Type**: AI observability and evaluation platform

#### Features

- **Tracing**: Automatic Claude call logging
- **Evaluation**: 50+ metrics (hallucination, relevance, bias)
- **Monitoring**: Production dashboards
- **Self-hosted**: Docker deployment

#### Quick Start

```python
import opik
from opik.integrations.anthropic import track_anthropic

# Auto-trace all Claude calls
client = track_anthropic(anthropic.Anthropic())

# Manual function tracing
@opik.track(name="my_function", tags=["production"])
def my_llm_pipeline(prompt): ...

# Evaluation metrics
from opik.evaluation.metrics import (
    Hallucination,
    AnswerRelevance,
    ContextPrecision,  # RAG
    AgentTaskCompletionJudge,  # Agents
    GenderBiasJudge,  # Bias
)
```

---

## Tier 4: Research Resources

### System Prompts Collection

**Location**: `mcp-ecosystem/system-prompts-and-models-of-ai-tools/`
**Type**: Research collection of AI system prompts

#### Contents

- 30,000+ lines of AI system prompts
- Multiple AI tools covered (Amp, etc.)
- Extraction methods documented

#### Use Cases

- Understanding AI tool behavior
- Prompt engineering research
- Comparative analysis

---

## Integration Patterns

### Pattern 1: Research Pipeline

```
Exa (deep_research) → Tavily (extract) → Context7 (docs) → Claude (synthesis)
```

### Pattern 2: Agent Orchestration

```
claude-flow (swarm) → Letta (stateful agents) → Opik (monitoring)
```

### Pattern 3: Development Workflow

```
everything-claude-code (skills) → claude-flow (agents) → Opik (evaluation)
```

### Pattern 4: Memory-Enhanced Agents

```
Letta (agent) → Qdrant (vectors) → claude-flow (coordination)
```

---

## Usage Matrix

| Task | Primary Tool | Fallback |
|------|--------------|----------|
| Web search | Exa | Tavily |
| Content extraction | Tavily Extract | Jina Reader |
| Library docs | Context7 | WebFetch |
| Multi-agent coordination | claude-flow | Manual Task tool |
| Stateful agents | Letta | claude-mem + hooks |
| AI monitoring | Opik | Custom logging |
| Code evaluation | Eval Harness | Manual testing |
| Pattern learning | Continuous Learning | Manual skills |

---

## Quick Reference Commands

```bash
# Claude-Flow
npm install -g claude-flow@v3alpha
claude mcp add claude-flow -- npx claude-flow@v3alpha mcp start
./claude-flow swarm "task" --strategy development

# Letta
pip install letta-client
# See Python examples above

# Tavily
pip install tavily-python
# See Python examples above

# Opik
pip install opik
opik configure
# See Python examples above

# Everything Claude Code
# Copy agents/skills to ~/.claude/
```

---

## Appendix: Repository Locations

```
Z:\insider\AUTO CLAUDE\unleash\sdks\mcp-ecosystem\
├── claude-flow/                    # V3 Enterprise Orchestration
├── letta/                          # Stateful AI Agents (MemGPT)
├── exa-mcp-server/                 # Neural Search
├── tavily-python/                  # Python Search/Extract/Crawl
├── tavily-mcp/                     # MCP Search Server
├── tavily-js/                      # JavaScript SDK
├── jina-mcp/                       # Neural Search/Embeddings
├── context7/                       # Library Documentation
├── everything-claude-code/         # Claude Code Plugin
├── opik/                           # AI Observability
└── system-prompts-and-models/      # System Prompt Research
```

---

*Generated: 2026-01-27 | Total Analysis: 11 repositories, 50K+ lines of source code reviewed*
