# Integration Guide: Claude-Flow + Letta + Opik

> **Purpose**: Production-ready integration patterns for combining multi-agent orchestration, stateful memory, and AI observability
> **Created**: 2026-01-27
> **Source**: MCP Ecosystem Deep Dive Research

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATION LAYER                         │
│                      (Claude-Flow V3)                           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │  Queen  │  │ Worker  │  │ Worker  │  │ Worker  │            │
│  │(Planner)│  │(Coder)  │  │(Tester) │  │(Reviewer)│           │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘            │
│       │            │            │            │                  │
│       └────────────┴────────────┴────────────┘                  │
│                          │                                      │
└──────────────────────────┼──────────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────────┐
│                     MEMORY LAYER                                │
│                       (Letta)                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Core Memory │  │ Archival    │  │ Recall      │             │
│  │  (persona,   │  │ (passages,  │  │ (search,    │             │
│  │   human)     │  │  embeddings)│  │  retrieve)  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└──────────────────────────┼──────────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────────┐
│                   OBSERVABILITY LAYER                           │
│                       (Opik)                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Traces    │  │   Metrics   │  │ Evaluation  │             │
│  │ (LLM calls, │  │ (latency,   │  │ (50+ metrics│             │
│  │  tool use)  │  │  tokens)    │  │  RAG, Agent)│             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start: Combined Stack

### 1. Installation

```bash
# Claude-Flow V3 (orchestration)
npm install -g claude-flow@v3alpha

# Letta (memory)
pip install letta-client

# Opik (observability)
pip install opik
opik configure  # Interactive setup
```

### 2. Environment Variables

```bash
# Claude-Flow
export CLAUDE_API_KEY="your-anthropic-key"

# Letta
export LETTA_API_KEY="your-letta-key"  # From app.letta.com

# Opik (self-hosted or cloud)
export OPIK_API_KEY="your-opik-key"
export OPIK_WORKSPACE="default"
export OPIK_PROJECT_NAME="my-project"
```

### 3. Basic Integration Script

```python
"""
Combined integration: Claude-Flow + Letta + Opik
"""
import opik
from opik.integrations.anthropic import track_anthropic
from letta_client import Letta
import subprocess
import json

# Initialize Opik tracing
opik.configure(project_name="claude-flow-letta-integration")

# Initialize Letta client
letta = Letta(api_key=os.environ["LETTA_API_KEY"])

# Create or retrieve Letta agent for memory
def get_or_create_agent(name: str, project: str):
    """Get existing agent or create new one with memory blocks."""
    agents = letta.agents.list()
    for agent in agents:
        if agent.name == name:
            return agent

    return letta.agents.create(
        name=name,
        model="openai/gpt-4o",
        memory_blocks=[
            {"label": "project", "value": f"Project: {project}"},
            {"label": "decisions", "value": "Key decisions made:"},
            {"label": "learnings", "value": "Patterns learned:"},
        ]
    )

# Trace Claude-Flow swarm execution
@opik.track(name="swarm_execution")
def run_swarm(task: str, strategy: str = "development"):
    """Execute Claude-Flow swarm with tracing."""
    result = subprocess.run(
        ["claude-flow", "swarm", task, "--strategy", strategy, "--json"],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout) if result.returncode == 0 else None

# Save swarm results to Letta memory
def persist_to_letta(agent_id: str, content: str, category: str = "learnings"):
    """Save swarm insights to Letta archival memory."""
    letta.agents.passages.create(
        agent_id=agent_id,
        text=content,
        metadata={"category": category, "source": "claude-flow"}
    )

# Recall relevant context from Letta
def recall_from_letta(agent_id: str, query: str, top_k: int = 5):
    """Search Letta memory for relevant context."""
    results = letta.agents.passages.search(
        agent_id=agent_id,
        query=query,
        top_k=top_k
    )
    return [r.text for r in results]

# Main integration loop
def integrated_workflow(task: str, project: str):
    """
    Full integration workflow:
    1. Recall relevant context from Letta
    2. Execute Claude-Flow swarm
    3. Trace with Opik
    4. Persist learnings back to Letta
    """
    # Get/create Letta agent
    agent = get_or_create_agent(f"{project}-agent", project)

    # Recall context
    context = recall_from_letta(agent.id, task)
    print(f"Retrieved {len(context)} relevant memories")

    # Execute swarm (traced by Opik)
    result = run_swarm(task)

    if result:
        # Persist insights
        persist_to_letta(
            agent.id,
            f"Task: {task}\nResult: {result.get('summary', 'N/A')}",
            category="decisions"
        )

    return result

if __name__ == "__main__":
    integrated_workflow(
        task="Implement user authentication with JWT",
        project="my-app"
    )
```

---

## Integration Pattern 1: Swarm Memory Sharing

### Problem
Multiple agents in a Claude-Flow swarm need to share learned context.

### Solution
Use Letta as the shared memory backend.

```python
from claude_flow import Swarm, Agent
from letta_client import Letta

class MemoryAwareSwarm:
    """Swarm with shared Letta memory."""

    def __init__(self, project: str):
        self.letta = Letta()
        self.agent_id = self._get_or_create_agent(project)

    def _get_or_create_agent(self, project: str) -> str:
        # Create Letta agent for shared memory
        agent = self.letta.agents.create(
            name=f"swarm-memory-{project}",
            memory_blocks=[
                {"label": "context", "value": "Shared swarm context"},
                {"label": "decisions", "value": "Swarm decisions log"},
            ]
        )
        return agent.id

    def store_finding(self, agent_name: str, finding: str):
        """Store agent finding in shared memory."""
        self.letta.agents.passages.create(
            agent_id=self.agent_id,
            text=f"[{agent_name}] {finding}",
            metadata={"agent": agent_name, "type": "finding"}
        )

    def get_context(self, query: str) -> list:
        """Get relevant context for any agent."""
        return self.letta.agents.passages.search(
            agent_id=self.agent_id,
            query=query,
            top_k=5
        )

    async def execute(self, task: str, strategy: str = "development"):
        """Execute swarm with memory integration."""
        # Pre-load context
        context = self.get_context(task)

        # Run swarm (pseudo-code - actual claude-flow API may differ)
        result = await Swarm(strategy=strategy).execute(
            task=task,
            context="\n".join([c.text for c in context])
        )

        # Store result
        self.store_finding("swarm", f"Completed: {task}")

        return result
```

---

## Integration Pattern 2: Traced Multi-Agent Workflows

### Problem
Need visibility into multi-agent execution for debugging and optimization.

### Solution
Wrap each agent step with Opik tracing.

```python
import opik
from opik import track

class TracedAgent:
    """Agent with full Opik tracing."""

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    @track(name="agent_think")
    def think(self, context: str) -> str:
        """Traced thinking step."""
        # Agent reasoning logic
        return f"[{self.name}] Analysis: ..."

    @track(name="agent_act")
    def act(self, decision: str) -> dict:
        """Traced action step."""
        # Agent action logic
        return {"action": decision, "result": "success"}

    @track(name="agent_step")
    def step(self, task: str, context: str = "") -> dict:
        """Full traced agent step."""
        thought = self.think(context)
        action = self.act(thought)
        return {
            "agent": self.name,
            "thought": thought,
            "action": action
        }

class TracedSwarm:
    """Swarm with comprehensive tracing."""

    def __init__(self, agents: list[TracedAgent]):
        self.agents = agents

    @track(name="swarm_execution", tags=["orchestration"])
    def execute(self, task: str):
        """Traced swarm execution."""
        results = []
        for agent in self.agents:
            result = agent.step(task)
            results.append(result)
        return results

# Usage with Opik dashboard
swarm = TracedSwarm([
    TracedAgent("planner", "planning"),
    TracedAgent("coder", "implementation"),
    TracedAgent("reviewer", "review"),
])

# All calls traced and visible in Opik dashboard
swarm.execute("Implement feature X")
```

---

## Integration Pattern 3: Evaluation-Driven Development

### Problem
Need to measure agent quality and catch regressions.

### Solution
Combine Opik metrics with Claude-Flow verification.

```python
from opik.evaluation import evaluate
from opik.evaluation.metrics import (
    Hallucination,
    AnswerRelevance,
    ContextPrecision,
    AgentTaskCompletionJudge,
)

# Define evaluation dataset
eval_dataset = [
    {
        "task": "Write a function to validate email",
        "expected": "Function with regex validation",
        "context": "Python, no external libraries"
    },
    {
        "task": "Fix authentication bug",
        "expected": "Token refresh logic corrected",
        "context": "JWT-based auth system"
    },
]

# Evaluation metrics for coding agents
coding_metrics = [
    AgentTaskCompletionJudge(name="task_completion"),
    AnswerRelevance(name="code_relevance"),
]

# Run evaluation
def eval_agent(task: dict) -> dict:
    """Evaluate single agent task."""
    result = run_swarm(task["task"])
    return {
        "output": result.get("code", ""),
        "context": task["context"]
    }

# Execute evaluation
results = evaluate(
    experiment_name="coding-agent-v1",
    dataset=eval_dataset,
    task=eval_agent,
    scoring_metrics=coding_metrics
)

# Results visible in Opik dashboard with:
# - pass@k metrics
# - Regression tracking
# - Agent comparison
```

---

## Integration Pattern 4: Self-Improving Agents

### Problem
Agents should learn from successes and failures.

### Solution
Letta memory + Opik feedback loop.

```python
class SelfImprovingAgent:
    """Agent that learns from traced feedback."""

    def __init__(self, name: str):
        self.name = name
        self.letta = Letta()
        self.agent_id = self._create_agent()

    def _create_agent(self) -> str:
        agent = self.letta.agents.create(
            name=f"self-improving-{self.name}",
            memory_blocks=[
                {"label": "successes", "value": "Patterns that worked:"},
                {"label": "failures", "value": "Patterns to avoid:"},
                {"label": "strategies", "value": "Current strategies:"},
            ]
        )
        return agent.id

    @opik.track(name="execute_with_learning")
    def execute(self, task: str) -> dict:
        """Execute task with learning integration."""

        # 1. Recall relevant patterns
        successes = self.letta.agents.passages.search(
            agent_id=self.agent_id,
            query=task,
            top_k=3
        )

        # 2. Execute with context
        result = self._do_task(task, successes)

        # 3. Get feedback from Opik traces
        # (In real implementation, this would come from Opik API)
        feedback = self._get_trace_feedback()

        # 4. Learn from outcome
        self._learn(task, result, feedback)

        return result

    def _learn(self, task: str, result: dict, feedback: dict):
        """Store learning in Letta memory."""
        if feedback.get("success", False):
            self.letta.agents.passages.create(
                agent_id=self.agent_id,
                text=f"SUCCESS: {task} -> {result.get('approach')}",
                metadata={"type": "success", "score": feedback.get("score", 0)}
            )
        else:
            self.letta.agents.passages.create(
                agent_id=self.agent_id,
                text=f"FAILURE: {task} -> Avoid: {result.get('approach')}",
                metadata={"type": "failure", "reason": feedback.get("reason")}
            )
```

---

## Integration Pattern 5: Hierarchical Memory with Paging

### Problem
Large context windows still fill up; need intelligent memory management.

### Solution
Combine Letta's archival memory with tiered paging.

```python
"""
MemGPT-style paging with Letta + local hot cache.
Based on memgpt_paging.py from UNLEASH SDK.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from letta_client import Letta

@dataclass
class MemoryTier:
    HOT = "hot"      # In-context, immediate access
    WARM = "warm"    # Letta archival, fast retrieval
    COLD = "cold"    # Qdrant vectors, semantic search

class TieredMemoryManager:
    """Multi-tier memory with automatic paging."""

    def __init__(self, max_hot_items: int = 50):
        self.letta = Letta()
        self.agent_id = None
        self.hot_cache = []  # In-context items
        self.max_hot_items = max_hot_items

    def add(self, content: str, importance: float = 0.5):
        """Add to hot tier, page out if needed."""
        entry = {
            "content": content,
            "importance": importance,
            "timestamp": datetime.now(timezone.utc),
            "tier": MemoryTier.HOT
        }

        self.hot_cache.append(entry)

        # Page out if over capacity
        if len(self.hot_cache) > self.max_hot_items:
            self._page_out()

    def _page_out(self):
        """Move least important items to warm tier (Letta)."""
        # Sort by importance (ascending) and age
        self.hot_cache.sort(key=lambda x: (x["importance"], x["timestamp"]))

        # Page out bottom 20%
        page_count = len(self.hot_cache) // 5
        to_page = self.hot_cache[:page_count]
        self.hot_cache = self.hot_cache[page_count:]

        # Store in Letta archival
        for entry in to_page:
            self.letta.agents.passages.create(
                agent_id=self.agent_id,
                text=entry["content"],
                metadata={
                    "importance": entry["importance"],
                    "paged_at": datetime.now(timezone.utc).isoformat()
                }
            )

    def recall(self, query: str, top_k: int = 10) -> list:
        """Recall from all tiers."""
        results = []

        # 1. Hot tier (exact match)
        for entry in self.hot_cache:
            if query.lower() in entry["content"].lower():
                results.append(entry["content"])

        # 2. Warm tier (Letta semantic)
        letta_results = self.letta.agents.passages.search(
            agent_id=self.agent_id,
            query=query,
            top_k=top_k - len(results)
        )
        results.extend([r.text for r in letta_results])

        return results[:top_k]
```

---

## MCP Server Configuration

### Combined MCP Setup for Claude Code

```json
{
  "mcpServers": {
    "claude-flow": {
      "type": "stdio",
      "command": "npx",
      "args": ["claude-flow@v3alpha", "mcp", "start"],
      "env": {
        "CLAUDE_API_KEY": "${CLAUDE_API_KEY}"
      }
    },
    "letta": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@letta-ai/letta-mcp"],
      "env": {
        "LETTA_API_KEY": "${LETTA_API_KEY}"
      }
    },
    "opik": {
      "type": "stdio",
      "command": "python",
      "args": ["-m", "opik.mcp_server"],
      "env": {
        "OPIK_API_KEY": "${OPIK_API_KEY}"
      }
    }
  }
}
```

---

## Best Practices

### 1. Memory Lifecycle
```
Session Start → Load from Letta (warm) → Work in hot cache →
Session End → Persist to Letta → Summarize to cold (Qdrant)
```

### 2. Tracing Granularity
- **Swarm level**: Always trace
- **Agent level**: Trace when debugging
- **Tool level**: Trace for performance tuning

### 3. Evaluation Cadence
- **CI/CD**: Run eval suite on every PR
- **Nightly**: Full regression suite
- **Weekly**: Human review of edge cases

### 4. Cost Optimization
- Use Haiku for worker agents (3x cost savings)
- Page out aggressively (Letta storage is cheap)
- Sample traces in production (10% default)

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Letta search returns empty | No passages stored | Check agent_id, add test data |
| Opik traces missing | Not configured | Run `opik configure` |
| Claude-Flow timeout | Long task | Add `--timeout 300` |
| Memory paging slow | Too many items | Reduce max_hot_items |

---

## See Also

- `MCP_ECOSYSTEM_DEEP_DIVE.md` - Full repository analysis
- `~/.claude/scripts/memgpt_paging.py` - MemGPT-style paging implementation
- `~/.claude/rules/orchestration.md` - Agent orchestration patterns
